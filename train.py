"""
RL Agent Training Script
=========================
Trains a PPO agent on the ICT/SMC trading environment.

Usage:
    python train.py --data QQQ_data.csv --timesteps 500000
    python train.py --data QQQ_data.csv --resume checkpoints/best_model
    python train.py --data QQQ_data.csv --eval checkpoints/best_model
"""

import argparse
import os
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from trading_env import TradingEnv
from data_feed import HistoricalDataFeed
from ict_adapter import ICTSignalAdapter


# ── Custom callback ───────────────────────────────────────────────────────────
class TradeMetricsCallback(BaseCallback):
    """Logs trading-specific metrics (trades, win rate) to TensorBoard."""

    def __init__(self, log_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq          = log_freq
        self.episode_trades    = []
        self.episode_win_rates = []

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            for info in self.locals.get('infos', []):
                if 'total_trades' in info:
                    self.episode_trades.append(info['total_trades'])
                    self.episode_win_rates.append(info['win_rate'])

            if self.episode_trades:
                avg_trades = np.mean(self.episode_trades[-20:])
                avg_wr     = np.mean(self.episode_win_rates[-20:])
                self.logger.record("trade/avg_trades",   avg_trades)
                self.logger.record("trade/avg_win_rate", avg_wr)
                if self.verbose:
                    print(f"  Step {self.n_calls}: "
                          f"avg_trades={avg_trades:.1f}  "
                          f"avg_win_rate={avg_wr:.2%}")
        return True


# ── Environment factory ───────────────────────────────────────────────────────
def make_env(data_path: str, initial_balance: float = 10_000.0,
             randomize_start: bool = True):
    def _init():
        feed = HistoricalDataFeed.from_csv(data_path, randomize_start=randomize_start)
        # max_steps: full usable data length so episodes can run the whole dataset
        max_steps = max(1, feed.n_rows - HistoricalDataFeed.HISTORY_LEN - 1)
        env = TradingEnv(
            data_feed       = feed,
            ict_detector    = ICTSignalAdapter(),
            initial_balance = initial_balance,
            max_steps       = max_steps,
        )
        return Monitor(env)
    return _init


# ── Resolve helpers ───────────────────────────────────────────────────────────
def resolve_data(path: str) -> str:
    if os.path.exists(path):
        return path
    for candidate in ["QQQ_data.csv", "nasdaq_data.csv", "SPY_data.csv"]:
        if os.path.exists(candidate):
            print(f"[warn] '{path}' not found — using '{candidate}'")
            return candidate
    raise FileNotFoundError(f"Data file not found: {path}")


def resolve_model(path: str) -> str:
    if os.path.exists(path) or os.path.exists(path + ".zip"):
        return path
    for candidate in [
        "checkpoints/best_model",
        "models/ppo_ict/best_model",
        "models/ppo_ict/ppo_ict_final",
    ]:
        if os.path.exists(candidate + ".zip") or os.path.exists(candidate):
            print(f"[warn] '{path}' not found — using '{candidate}'")
            return candidate
    raise FileNotFoundError(f"Model not found: {path}")


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    data_path = resolve_data(args.data)

    print(f"{'='*60}")
    print(f"  RL Trading Agent - Training")
    print(f"  Data:      {data_path}")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"{'='*60}\n")

    os.makedirs("checkpoints",        exist_ok=True)
    os.makedirs("logs/tensorboard",   exist_ok=True)
    os.makedirs("logs/eval",          exist_ok=True)

    # VecNormalize stabilises training on noisy financial observations
    train_env = DummyVecEnv([make_env(data_path, randomize_start=True)])
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=True,
        clip_obs=10.0, clip_reward=10.0,
    )

    eval_env = DummyVecEnv([make_env(data_path, randomize_start=False)])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
    )

    # PPO tuned for swing trading
    model_kwargs = dict(
        policy         = "MlpPolicy",
        env            = train_env,
        learning_rate  = 1e-4,       # slower for noisy financial data
        n_steps        = 4096,       # larger rollouts to capture full swings
        batch_size     = 128,
        n_epochs       = 10,
        gamma          = 0.995,      # high discount — swing returns span many bars
        gae_lambda     = 0.97,
        clip_range     = 0.15,
        ent_coef       = 0.05,       # high entropy — prevent hold-forever collapse
        vf_coef        = 0.5,
        max_grad_norm  = 0.5,
        policy_kwargs  = dict(
            net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
        ),
        verbose           = 1,
        tensorboard_log   = "logs/tensorboard/",
    )

    if args.resume:
        model_path = resolve_model(args.resume)
        print(f"Resuming from: {model_path}\n")
        model = PPO.load(model_path, env=train_env)
    else:
        model = PPO(**model_kwargs)

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path = "checkpoints/",
            log_path             = "logs/eval/",
            eval_freq            = 10_000,
            n_eval_episodes      = 5,
            deterministic        = True,
        ),
        CheckpointCallback(
            save_freq   = 50_000,
            save_path   = "checkpoints/",
            name_prefix = "trading_agent",
        ),
        TradeMetricsCallback(log_freq=5000, verbose=1),
    ]

    print("Starting training...\n")
    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"checkpoints/trading_agent_final_{timestamp}"
    model.save(model_path)
    train_env.save(f"{model_path}_vecnormalize.pkl")

    print(f"\nModel saved:        {model_path}.zip")
    print(f"VecNormalize saved: {model_path}_vecnormalize.pkl")
    print("TensorBoard:        tensorboard --logdir logs/tensorboard")


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(args):
    data_path  = resolve_data(args.data)
    model_path = resolve_model(args.eval)

    print(f"{'='*60}")
    print(f"  RL Trading Agent - Evaluation")
    print(f"  Model: {model_path}")
    print(f"  Data:  {data_path}")
    print(f"{'='*60}\n")

    eval_env     = DummyVecEnv([make_env(data_path, randomize_start=True)])
    vecnorm_path = f"{model_path}_vecnormalize.pkl"
    if os.path.exists(vecnorm_path):
        eval_env = VecNormalize.load(vecnorm_path, eval_env)
        eval_env.training    = False
        eval_env.norm_reward = False

    model       = PPO.load(model_path, env=eval_env)
    all_results = []

    for ep in range(args.episodes):
        obs  = eval_env.reset()
        done = False
        info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, done_arr, info_arr = eval_env.step(action)
            done = bool(done_arr[0])
            info = info_arr[0]

        all_results.append(info)
        print(f"  Episode {ep+1}/{args.episodes}: "
              f"equity=${info['equity']:,.2f}  "
              f"trades={info['total_trades']}  "
              f"win_rate={info['win_rate']:.1%}  "
              f"drawdown={info['drawdown']:.1%}")

    print(f"\n{'─'*60}")
    print("  SUMMARY")
    print(f"{'─'*60}")
    avg_equity = np.mean([r['equity']       for r in all_results])
    avg_trades = np.mean([r['total_trades'] for r in all_results])
    avg_wr     = np.mean([r['win_rate']     for r in all_results])
    avg_dd     = np.mean([r['drawdown']     for r in all_results])
    print(f"  Avg Final Equity:   ${avg_equity:,.2f}")
    print(f"  Avg Trades/Episode: {avg_trades:.1f}")
    print(f"  Avg Win Rate:       {avg_wr:.1%}")
    print(f"  Avg Max Drawdown:   {avg_dd:.1%}")
    print(f"  Return:             {((avg_equity / 10_000) - 1) * 100:+.2f}%")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RL Trading Agent")
    parser.add_argument("--data",       required=True,            help="OHLCV CSV file")
    parser.add_argument("--timesteps",  type=int, default=500_000)
    parser.add_argument("--resume",     type=str, default=None,   help="Resume from checkpoint")
    parser.add_argument("--eval",       type=str, default=None,   help="Evaluate model path")
    parser.add_argument("--episodes",   type=int, default=10,     help="Eval episodes")
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
