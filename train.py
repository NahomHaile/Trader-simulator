#!/usr/bin/env python3
"""
PPO Stock Trading Agent — Stable-Baselines3 + Gymnasium
========================================================
Usage:
    python train.py --data QQQ_data.csv --timesteps 500000
    python train.py --data qqq_daily_5y.csv --timesteps 1000000 --out models/qqq_agent
"""
import argparse
import os

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor


# ── Hyper-params ──────────────────────────────────────────────────────────────
LOOKBACK      = 20          # observation window (trading days)
INITIAL_CASH  = 10_000.0   # starting portfolio value ($)
TRADE_COST    = 0.001       # 0.1 % round-trip commission per trade
TRAIN_SPLIT   = 0.80        # fraction of data used for training


# ── Feature engineering ───────────────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators and append as columns."""
    c = df["Close"]

    # Price returns
    df["ret_1"]  = c.pct_change(1)
    df["ret_5"]  = c.pct_change(5)
    df["ret_20"] = c.pct_change(20)

    # RSI-14
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    # MACD (normalized by price)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    df["macd"]        = macd / c
    df["macd_signal"] = macd.ewm(span=9, adjust=False).mean() / c

    # Bollinger-band position  (-1 = lower band, +1 = upper band)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_pos"] = (c - sma20) / (2 * std20 + 1e-9)

    # Volume relative to 20-day average
    df["vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)

    # ATR-14 (normalized by price)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - c.shift()).abs(),
        (df["Low"]  - c.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean() / c

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── Gymnasium environment ─────────────────────────────────────────────────────
class StockTradingEnv(gym.Env):
    """
    Discrete stock trading environment.

    Actions
    -------
    0 – hold
    1 – buy  (invest all available cash)
    2 – sell (liquidate entire position)

    Observation
    -----------
    Flattened LOOKBACK × n_features window  +  [position_flag, cash_ratio]
    """

    metadata = {"render_modes": []}

    FEATURE_COLS = [
        "ret_1", "ret_5", "ret_20",
        "rsi", "macd", "macd_signal",
        "bb_pos", "vol_ratio", "atr",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        lookback: int   = LOOKBACK,
        initial_cash: float = INITIAL_CASH,
        trade_cost: float   = TRADE_COST,
    ):
        super().__init__()
        self.df           = df.reset_index(drop=True)
        self.lookback     = lookback
        self.initial_cash = initial_cash
        self.trade_cost   = trade_cost
        self.n_features   = len(self.FEATURE_COLS)

        obs_dim = lookback * self.n_features + 2
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._start_idx = lookback
        self._end_idx   = len(df) - 1

    # ── helpers ───────────────────────────────────────────────────────────────
    def _price(self) -> float:
        return float(self.df.loc[self.current_step, "Close"])

    def _portfolio_value(self) -> float:
        return self.cash + self.shares * self._price()

    def _get_obs(self) -> np.ndarray:
        window = self.df.loc[
            self.current_step - self.lookback : self.current_step - 1,
            self.FEATURE_COLS,
        ].values.astype(np.float32)
        window = np.clip(window, -10.0, 10.0)

        position_flag = np.float32(self.shares > 0)
        pv = self._portfolio_value()
        cash_ratio = np.float32(self.cash / (pv + 1e-9))
        return np.concatenate([window.flatten(), [position_flag, cash_ratio]])

    # ── gym interface ─────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self._start_idx
        self.cash         = self.initial_cash
        self.shares       = 0.0
        self.prev_value   = self.initial_cash
        return self._get_obs(), {}

    def step(self, action: int):
        price = self._price()

        if action == 1 and self.cash > 1.0:          # buy
            shares_bought = self.cash * (1 - self.trade_cost) / price
            self.shares  += shares_bought
            self.cash     = 0.0

        elif action == 2 and self.shares > 0.0:       # sell
            self.cash  += self.shares * price * (1 - self.trade_cost)
            self.shares = 0.0

        self.current_step += 1
        current_value = self._portfolio_value()

        # Log-return reward (proportional, unbiased across price scales)
        reward          = float(np.log(current_value / (self.prev_value + 1e-9)))
        self.prev_value = current_value

        terminated = self.current_step >= self._end_idx
        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        pv = self._portfolio_value()
        ret = (pv / self.initial_cash - 1) * 100
        print(f"  step={self.current_step}  price={self._price():.2f}"
              f"  pv=${pv:,.0f} ({ret:+.1f}%)"
              f"  shares={self.shares:.3f}  cash=${self.cash:,.0f}")


# ── Training entry-point ──────────────────────────────────────────────────────
def make_env(df: pd.DataFrame):
    def _init():
        env = StockTradingEnv(df)
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO stock trading agent")
    parser.add_argument("--data",      required=True,               help="OHLCV CSV file")
    parser.add_argument("--timesteps", type=int, default=500_000,   help="Training timesteps")
    parser.add_argument("--out",       default="models/ppo_stock",  help="Model output directory")
    parser.add_argument("--lr",        type=float, default=3e-4,    help="Learning rate")
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    # ── resolve data file ────────────────────────────────────────────────────
    data_path = args.data
    if not os.path.exists(data_path):
        candidates = [
            "QQQ_data.csv", "QQQ_data_signals.csv",
            "nasdaq_data.csv", "SPY_data.csv",
        ]
        for c in candidates:
            if os.path.exists(c):
                print(f"[warn] '{data_path}' not found — using '{c}' instead")
                data_path = c
                break
        else:
            raise FileNotFoundError(
                f"Data file not found: {args.data}\n"
                f"Run: python data_extractor.py --output {args.data}"
            )

    # ── load & featurise ─────────────────────────────────────────────────────
    df = pd.read_csv(data_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df = add_features(df)

    date_range = f"{df['Date'].iloc[0].date()} – {df['Date'].iloc[-1].date()}"
    print(f"Data: {len(df)} rows  |  {date_range}")

    split     = int(len(df) * TRAIN_SPLIT)
    train_df  = df.iloc[:split].reset_index(drop=True)
    eval_df   = df.iloc[split:].reset_index(drop=True)
    print(f"Train: {len(train_df)} rows   Eval: {len(eval_df)} rows\n")

    # ── environments ─────────────────────────────────────────────────────────
    train_env = DummyVecEnv([make_env(train_df)])
    eval_env  = DummyVecEnv([make_env(eval_df)])

    os.makedirs(args.out, exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)

    # ── callbacks ────────────────────────────────────────────────────────────
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
        name_prefix="ppo_stock",
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

    final_path = os.path.join(args.out, "ppo_stock_final")
    model.save(final_path)
    print(f"\nSaved → {final_path}.zip")
    print("TensorBoard: tensorboard --logdir logs/tensorboard")


if __name__ == "__main__":
    main()
