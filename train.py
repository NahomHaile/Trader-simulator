#!/usr/bin/env python3
"""
PPO Stock Trading Agent — ICT/Smart Money Strategy
===================================================
Trains a PPO agent on OHLCV data using the ICT/SMC trading environment.

Usage:
    python train.py --data QQQ_data.csv --timesteps 500000
    python train.py --data qqq_daily_5y.csv --timesteps 1000000 --out models/qqq_ict
"""
import argparse
import os

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from data_feed import CSVDataFeed
from trading_env import TradingEnv
from ict_adapter import ICTSignalAdapter


TRAIN_SPLIT = 0.80


def make_env(df: pd.DataFrame, random_start: bool = True, max_steps: int = 200):
    def _init():
        feed = CSVDataFeed(df, random_start=random_start)
        ict  = ICTSignalAdapter()
        env  = TradingEnv(feed, ict, max_steps=max_steps)
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO ICT trading agent")
    parser.add_argument("--data",      required=True,              help="OHLCV CSV file")
    parser.add_argument("--timesteps", type=int, default=500_000,  help="Training timesteps")
    parser.add_argument("--out",       default="models/ppo_ict",   help="Model output directory")
    parser.add_argument("--lr",        type=float, default=3e-4,   help="Learning rate")
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    # ── resolve data file ─────────────────────────────────────────────────────
    data_path = args.data
    if not os.path.exists(data_path):
        for candidate in ["QQQ_data.csv", "nasdaq_data.csv", "SPY_data.csv"]:
            if os.path.exists(candidate):
                print(f"[warn] '{data_path}' not found — using '{candidate}'")
                data_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"Data file not found: {args.data}\n"
                f"Run: python data_extractor.py --output {args.data}"
            )

    # ── load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(data_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    split    = int(len(df) * TRAIN_SPLIT)
    train_df = df.iloc[:split].reset_index(drop=True)
    eval_df  = df.iloc[split:].reset_index(drop=True)

    date_range = f"{df['Date'].iloc[0].date()} – {df['Date'].iloc[-1].date()}"
    print(f"Data:  {len(df)} rows  |  {date_range}")
    print(f"Train: {len(train_df)} rows   Eval: {len(eval_df)} rows\n")

    # ── environments ──────────────────────────────────────────────────────────
    # Training: random episode start, 200-bar episodes
    # Eval: fixed start at beginning of eval set, full eval set
    eval_max_steps = max(1, len(eval_df) - CSVDataFeed.HISTORY_LEN - 1)

    train_env = DummyVecEnv([make_env(train_df, random_start=True,  max_steps=200)])
    eval_env  = DummyVecEnv([make_env(eval_df,  random_start=False, max_steps=eval_max_steps)])

    os.makedirs(args.out, exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)

    # ── callbacks ─────────────────────────────────────────────────────────────
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.out,
        log_path="logs",
        eval_freq=10_000,
        n_eval_episodes=1,
        deterministic=True,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=args.out,
        name_prefix="ppo_ict",
    )

    # ── PPO model ─────────────────────────────────────────────────────────────
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=args.seed,
        verbose=1,
        tensorboard_log="logs/tensorboard",
    )

    print(f"Training PPO for {args.timesteps:,} timesteps …")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_cb, ckpt_cb],
    )

    final_path = os.path.join(args.out, "ppo_ict_final")
    model.save(final_path)
    print(f"\nSaved → {final_path}.zip")
    print("TensorBoard: tensorboard --logdir logs/tensorboard")


if __name__ == "__main__":
    main()
