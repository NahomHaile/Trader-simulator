"""
Out-of-Sample Backtest
======================
Splits data by date into train/test periods.
Runs the trained model ONLY on the test period (data it never saw during training).
This is the true test of whether the model learned real patterns or just memorized.

Usage:
    python backtest_oos.py --data QQQ_data.csv --model checkpoints/trading_agent_final_20260402_002531 --train-end 2024-01-01
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from trading_env import TradingEnv, TradeAction
from data_feed import CSVDataFeed, HistoricalDataFeed
from ict_adapter import ICTSignalAdapter


class FixedStartFeed(CSVDataFeed):
    """
    Data feed that always starts at bar HISTORY_LEN and runs to the end.
    Used for the OOS backtest — no random start, one clean run through.
    """
    def reset(self, seed=None):
        self.current_idx = self.history_len


def load_and_split(data_path: str, train_end: str):
    """
    Load CSV and split into train/test by date.
    Returns (train_df, test_df, split_date).
    """
    peek = pd.read_csv(data_path, nrows=3, header=None)
    is_yfinance = str(peek.iloc[1, 0]).lower() == "ticker"
    if is_yfinance:
        df = pd.read_csv(data_path, skiprows=[1, 2])
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    else:
        df = pd.read_csv(data_path)
        if "Datetime" in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)

    df.columns = [c.strip().title() if c != "Date" else "Date" for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], utc=False, errors="coerce")
    if hasattr(df["Date"].dt, "tz") and df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)
    df.dropna(subset=["Date"], inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    split_date = pd.Timestamp(train_end)
    train_df = df[df["Date"] < split_date].reset_index(drop=True)
    test_df  = df[df["Date"] >= split_date].reset_index(drop=True)

    return train_df, test_df, split_date


def run_full_episode(model, env, start_price: float, end_price: float) -> dict:
    """Run one deterministic episode from start to end of data."""
    obs = env.reset()
    done = False
    info = {}
    equity_curve = []
    trade_log    = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, done_arr, info_arr = env.step(action)
        done = bool(done_arr[0])
        info = info_arr[0]

        equity_curve.append(info["equity"])

        action_val = int(action[0][0])
        if TradeAction(action_val) != TradeAction.HOLD:
            trade_log.append({
                "step":   info["step"],
                "action": TradeAction(action_val).name,
                "equity": info["equity"],
            })

    # B&H calculated from actual test period prices passed in — not from feed
    bh_return    = (end_price - start_price) / (start_price + 1e-9) * 100
    agent_return = (info["equity"] - 10_000.0) / 10_000.0 * 100

    # Max drawdown
    peak   = 10_000.0
    max_dd = 0.0
    for eq in equity_curve:
        peak   = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / (peak + 1e-9))

    # Annualised Sharpe
    sharpe = 0.0
    if len(equity_curve) >= 2:
        rets = np.diff([10_000.0] + equity_curve) / (
            np.array([10_000.0] + equity_curve[:-1]) + 1e-9
        )
        if rets.std() > 0:
            sharpe = float(rets.mean() / rets.std() * np.sqrt(252))

    return {
        "agent_return":   agent_return,
        "buyhold_return": bh_return,
        "beats_market":   agent_return > bh_return,
        "final_equity":   info["equity"],
        "total_trades":   info["total_trades"],
        "win_rate":       info["win_rate"],
        "max_drawdown":   max_dd * 100,
        "sharpe_ratio":   sharpe,
        "equity_curve":   equity_curve,
        "trade_log":      trade_log,
    }


def backtest(args):
    # ── resolve paths ─────────────────────────────────────────────────────────
    data_path = args.data
    if not os.path.exists(data_path):
        for candidate in ["QQQ_data.csv", "nasdaq_data.csv", "SPY_data.csv"]:
            if os.path.exists(candidate):
                print(f"[warn] '{data_path}' not found — using '{candidate}'")
                data_path = candidate
                break
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")

    model_path = args.model
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        for candidate in [
            "checkpoints/best_model",
            "checkpoints/trading_agent_final_20260402_002531",
        ]:
            if os.path.exists(candidate + ".zip") or os.path.exists(candidate):
                print(f"[warn] '{model_path}' not found — using '{candidate}'")
                model_path = candidate
                break
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

    # ── split data ────────────────────────────────────────────────────────────
    train_df, test_df, split_date = load_and_split(data_path, args.train_end)

    print(f"""
{'='*60}
  OUT-OF-SAMPLE BACKTEST
  Model:       {model_path}
  Data:        {data_path}
  Train period: {train_df['Date'].iloc[0].date()} → {train_df['Date'].iloc[-1].date()} ({len(train_df)} bars)
  Test period:  {test_df['Date'].iloc[0].date()} → {test_df['Date'].iloc[-1].date()} ({len(test_df)} bars)
{'='*60}
""")

    if len(test_df) < CSVDataFeed.HISTORY_LEN + 10:
        raise ValueError(
            f"Test period too short ({len(test_df)} bars). "
            f"Try an earlier --train-end date."
        )

    # ── build test environment ────────────────────────────────────────────────
    # Use the full dataset for the feed so ICT signals have history context,
    # but the episode starts exactly at the train/test split date.
    full_df, _, _ = load_and_split(data_path, args.train_end)
    # Re-use load_and_split just to get the parsed full_df; re-load without split
    _peek = pd.read_csv(data_path, nrows=3, header=None)
    _is_yf = str(_peek.iloc[1, 0]).lower() == "ticker"
    if _is_yf:
        full_df = pd.read_csv(data_path, skiprows=[1, 2])
        full_df.rename(columns={full_df.columns[0]: "Date"}, inplace=True)
    else:
        full_df = pd.read_csv(data_path)
        if "Datetime" in full_df.columns:
            full_df.rename(columns={"Datetime": "Date"}, inplace=True)
    full_df.columns = [c.strip().title() if c != "Date" else "Date" for c in full_df.columns]
    full_df["Date"] = pd.to_datetime(full_df["Date"], utc=False, errors="coerce")
    if hasattr(full_df["Date"].dt, "tz") and full_df["Date"].dt.tz is not None:
        full_df["Date"] = full_df["Date"].dt.tz_localize(None)
    full_df.dropna(subset=["Date"], inplace=True)
    full_df.sort_values("Date", inplace=True)
    full_df.reset_index(drop=True, inplace=True)

    # Find the index in full_df where the test period starts
    split_idx = full_df[full_df["Date"] >= split_date].index[0]
    # Make sure there's enough history before split for HISTORY_LEN
    start_idx = max(CSVDataFeed.HISTORY_LEN, split_idx)

    test_feed = FixedStartFeed(full_df, random_start=False)
    # Override reset to start at split point with full history available
    test_feed.current_idx = start_idx

    max_steps = len(full_df) - start_idx - 1

    def make_test_env():
        feed = FixedStartFeed(full_df, random_start=False)
        feed.current_idx = start_idx
        env = TradingEnv(
            data_feed       = feed,
            ict_detector    = ICTSignalAdapter(),
            initial_balance = 10_000.0,
            max_steps       = max_steps,
        )
        return Monitor(env)

    test_env = DummyVecEnv([make_test_env])

    vecnorm_path = f"{model_path}_vecnormalize.pkl"
    if os.path.exists(vecnorm_path):
        test_env = VecNormalize.load(vecnorm_path, test_env)
        test_env.training    = False
        test_env.norm_reward = False
        print("Loaded VecNormalize stats.\n")

    model = PPO.load(model_path, env=test_env)

    # ── run multiple episodes over the test period ────────────────────────────
    # Each episode starts at a random point within the test period so we get
    # statistical variety rather than running the exact same bars 10 times.
    print(f"Running {args.episodes} episodes over unseen test data...\n")

    rng = np.random.default_rng(42)
    # Minimum 60 bars per episode so there's enough data to trade
    max_ep_start = len(full_df) - 60 - 1

    all_results = []
    for ep in range(args.episodes):
        # Random start within the test period
        ep_start = int(rng.integers(start_idx, max(start_idx + 1, max_ep_start)))
        ep_end   = len(full_df) - 1
        ep_steps = ep_end - ep_start

        # B&H prices taken directly from the dataframe — no feed needed
        ep_start_price = float(full_df["Close"].iloc[ep_start])
        ep_end_price   = float(full_df["Close"].iloc[ep_end])

        def make_ep_env(s=ep_start, steps=ep_steps):
            feed = FixedStartFeed(full_df, random_start=False)
            feed.current_idx = s
            env = TradingEnv(
                data_feed       = feed,
                ict_detector    = ICTSignalAdapter(),
                initial_balance = 10_000.0,
                max_steps       = steps,
            )
            return Monitor(env)

        ep_env = DummyVecEnv([make_ep_env])
        if os.path.exists(vecnorm_path):
            ep_env = VecNormalize.load(vecnorm_path, ep_env)
            ep_env.training    = False
            ep_env.norm_reward = False

        result = run_full_episode(model, ep_env, ep_start_price, ep_end_price)
        all_results.append(result)

        marker = ""
        if result["beats_market"] and result["agent_return"] > 0:
            marker = " << BEAT MARKET"
        elif result["agent_return"] > 0:
            marker = " (profitable)"

        print(f"  Episode {ep+1:2d}/{args.episodes} | "
              f"Agent: {result['agent_return']:+7.2f}% | "
              f"B&H: {result['buyhold_return']:+7.2f}% | "
              f"Trades: {result['total_trades']:3d} | "
              f"WR: {result['win_rate']:5.1%} | "
              f"MaxDD: {result['max_drawdown']:5.1f}%"
              f"{marker}")

    # ── summary ───────────────────────────────────────────────────────────────
    agent_returns = [r["agent_return"]   for r in all_results]
    bh_returns    = [r["buyhold_return"] for r in all_results]
    win_rates     = [r["win_rate"]       for r in all_results]
    drawdowns     = [r["max_drawdown"]   for r in all_results]
    sharpes       = [r["sharpe_ratio"]   for r in all_results]
    trades        = [r["total_trades"]   for r in all_results]
    beats_count   = sum(r["beats_market"] for r in all_results)
    profitable    = sum(r["agent_return"] > 0 for r in all_results)

    avg_agent = float(np.mean(agent_returns))
    avg_bh    = float(np.mean(bh_returns))

    print(f"""
{'='*60}
  OUT-OF-SAMPLE RESULTS ({args.episodes} episodes on UNSEEN data)
{'='*60}

  TEST PERIOD: {test_df['Date'].iloc[0].date()} → {test_df['Date'].iloc[-1].date()}

  AGENT PERFORMANCE
  ─────────────────────────────────────────
  Avg Return:        {avg_agent:+.2f}%
  Median Return:     {np.median(agent_returns):+.2f}%
  Best Episode:      {np.max(agent_returns):+.2f}%
  Worst Episode:     {np.min(agent_returns):+.2f}%
  Std Dev:           {np.std(agent_returns):.2f}%
  Avg Win Rate:      {np.mean(win_rates):.1%}
  Avg Trades/Ep:     {np.mean(trades):.1f}
  Avg Max Drawdown:  {np.mean(drawdowns):.1f}%
  Avg Sharpe Ratio:  {np.mean(sharpes):.2f}

  BUY & HOLD (same test period)
  ─────────────────────────────────────────
  Avg Return:        {avg_bh:+.2f}%

  HEAD-TO-HEAD
  ─────────────────────────────────────────
  Agent beats market:  {beats_count}/{args.episodes} ({beats_count/args.episodes:.1%})
  Agent is profitable: {profitable}/{args.episodes} ({profitable/args.episodes:.1%})
  Avg excess return:   {avg_agent - avg_bh:+.2f}%
""")

    # Verdict
    print("  VERDICT")
    print("  ─────────────────────────────────────────")
    profit_rate = profitable / args.episodes

    if avg_agent > 0 and avg_agent > avg_bh and profit_rate >= 0.70:
        print("  PASS — Agent is profitable and beating market on UNSEEN data.")
        print("  This model is ready for paper trading.")
        verdict = "PASS"
    elif avg_agent > 0 and profit_rate >= 0.60:
        print("  MARGINAL PASS — Agent is profitable on unseen data but not")
        print("  consistently beating buy-and-hold.")
        print("  Consider more training before paper trading.")
        verdict = "MARGINAL"
    else:
        print("  FAIL — Agent is not reliably profitable on unseen data.")
        print("  The model likely overfit the training period.")
        print("  Recommendations:")
        print("  - Get more diverse training data (different market regimes)")
        print("  - Train on 4H data instead of daily")
        print("  - Reduce model complexity (smaller network)")
        verdict = "FAIL"

    # ── save results ──────────────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    save_data = {
        "timestamp":    datetime.now().isoformat(),
        "model":        model_path,
        "data":         data_path,
        "train_end":    args.train_end,
        "train_bars":   len(train_df),
        "test_bars":    len(test_df),
        "test_start":   str(test_df["Date"].iloc[0].date()),
        "test_end":     str(test_df["Date"].iloc[-1].date()),
        "verdict":      verdict,
        "summary": {
            "avg_agent_return":   avg_agent,
            "avg_buyhold_return": avg_bh,
            "avg_excess_return":  avg_agent - avg_bh,
            "profitable_pct":     profit_rate,
            "beats_market_pct":   beats_count / args.episodes,
            "avg_win_rate":       float(np.mean(win_rates)),
            "avg_max_drawdown":   float(np.mean(drawdowns)),
            "avg_sharpe":         float(np.mean(sharpes)),
            "avg_trades":         float(np.mean(trades)),
        },
        "episodes": [{
            "agent_return":   r["agent_return"],
            "buyhold_return": r["buyhold_return"],
            "beats_market":   r["beats_market"],
            "total_trades":   r["total_trades"],
            "win_rate":       r["win_rate"],
            "max_drawdown":   r["max_drawdown"],
            "sharpe_ratio":   r["sharpe_ratio"],
        } for r in all_results],
    }

    out_path = "logs/oos_backtest_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Full results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Out-of-Sample Backtest")
    parser.add_argument("--data",       required=True,                         help="OHLCV CSV file")
    parser.add_argument("--model",      required=True,                         help="Trained model path")
    parser.add_argument("--train-end",  default="2024-01-01",                  help="Date to split train/test (YYYY-MM-DD)")
    parser.add_argument("--episodes",   type=int, default=10,                  help="Episodes to run over test period")
    args = parser.parse_args()
    backtest(args)


if __name__ == "__main__":
    main()
