"""
RL Agent Training Script
=========================
Trains a PPO agent on the ICT/SMC trading environment using your
existing veteran_trader_v2 data and signal detection.

Usage:
    # Train on historical data
    python rl_trading_agent/train.py --data QQQ_data.csv --timesteps 500000

    # Resume from checkpoint
    python rl_trading_agent/train.py --data QQQ_data.csv --resume checkpoints/best_model

    # Evaluate a trained model
    python rl_trading_agent/train.py --data QQQ_data.csv --eval checkpoints/best_model
"""

import argparse
import os
import sys
import numpy as np
from datetime import datetime

# Ensure parent dir (stock/) is importable
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from veteran_trader_v2 import load_data, TraderConfig
from rl_trading_agent.trading_env import TradingEnv
from rl_trading_agent.data_feed import HistoricalDataFeed
from rl_trading_agent.ict_adapter import ICTSignalAdapter


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------
class TradeMetricsCallback(BaseCallback):
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
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
                          f"avg_trades={avg_trades:.1f}, avg_win_rate={avg_wr:.2%}")
        return True


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def make_env(data_path: str, initial_balance: float = 10_000.0):
    """
    Build a monitored TradingEnv.

    The OHLCV data is loaded once; ICTSignalAdapter builds SmartMoneyAnalyzer
    upfront so signals are O(1) lookups during training.
    """
    def _init():
        ohlcv = load_data(data_path)
        config = TraderConfig()
        feed   = HistoricalDataFeed.from_ohlcv(ohlcv, randomize_start=True)
        ict    = ICTSignalAdapter(ohlcv, config)
        env    = TradingEnv(
            data_feed=feed,
            ict_detector=ict,
            initial_balance=initial_balance,
            max_steps=5000,
        )
        return Monitor(env)
    return _init


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args):
    print(f"{'='*60}")
    print(f"  RL Trading Agent — Training")
    print(f"  Data:       {args.data}")
    print(f"  Timesteps:  {args.timesteps:,}")
    print(f"{'='*60}\n")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs",        exist_ok=True)

    train_env = DummyVecEnv([make_env(args.data)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                              clip_obs=10.0, clip_reward=10.0)

    eval_env = DummyVecEnv([make_env(args.data)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                             clip_obs=10.0)

    # PPO tuned for swing trading (4H candles, hold days to weeks):
    # - Slower learning rate: financial data is noisy, avoid overfitting
    # - Larger n_steps: captures full swing cycles (entry → hold → exit)
    # - Higher gamma: longer-horizon credit assignment for multi-day trades
    # - Tighter clip_range: stability on noisy rewards
    # - Higher entropy: more exploration of setups before converging
    model_kwargs = dict(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.97,
        clip_range=0.15,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
        ),
        verbose=1,
        tensorboard_log="logs/tensorboard/",
    )

    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=train_env)
    else:
        model = PPO(**model_kwargs)

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path="checkpoints/",
            log_path="logs/eval/",
            eval_freq=10_000,
            n_eval_episodes=5,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=50_000,
            save_path="checkpoints/",
            name_prefix="trading_agent",
        ),
        TradeMetricsCallback(log_freq=5000, verbose=1),
    ]

    print("Starting training...\n")
    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                progress_bar=True)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"checkpoints/trading_agent_final_{timestamp}"
    model.save(model_path)
    train_env.save(f"{model_path}_vecnormalize.pkl")
    print(f"\nModel saved to: {model_path}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(args):
    print(f"{'='*60}")
    print(f"  RL Trading Agent — Evaluation")
    print(f"  Model: {args.eval}")
    print(f"  Data:  {args.data}")
    print(f"{'='*60}\n")

    eval_env     = DummyVecEnv([make_env(args.data)])
    vecnorm_path = f"{args.eval}_vecnormalize.pkl"
    if os.path.exists(vecnorm_path):
        eval_env = VecNormalize.load(vecnorm_path, eval_env)
        eval_env.training  = False
        eval_env.norm_reward = False

    model       = PPO.load(args.eval, env=eval_env)
    all_results = []

    for ep in range(args.episodes):
        obs  = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
        info = info[0]
        all_results.append(info)
        print(f"  Episode {ep+1}/{args.episodes}: "
              f"equity=${info['equity']:,.2f}, "
              f"trades={info['total_trades']}, "
              f"win_rate={info['win_rate']:.1%}, "
              f"max_dd={info['drawdown']:.1%}")

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RL Trading Agent")
    parser.add_argument("--data",      required=True)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--resume",    type=str, default=None)
    parser.add_argument("--eval",      type=str, default=None)
    parser.add_argument("--episodes",  type=int, default=10)
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
