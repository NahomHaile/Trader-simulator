"""
RL Agent Performance Evaluation
=================================
Runs the trained agent through historical data and compares it
against a simple buy-and-hold benchmark (the market).

Usage:
    python evaluate.py --data QQQ_data.csv --model models/ppo_ict/best_model --episodes 50
"""

import argparse
import json
import os
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from trading_env import TradingEnv, TradeAction
from data_feed import HistoricalDataFeed
from ict_adapter import ICTSignalAdapter


def run_single_episode(model, env, feed: HistoricalDataFeed) -> dict:
    """Run one deterministic episode and collect detailed metrics."""
    obs = env.reset()

    # Capture start price after reset (data_feed is now positioned at episode start)
    start_price = feed.get_current_bar().get('close', 0.0)

    equity_curve  = []
    actions_taken = []
    info          = {}
    done          = False
    step          = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, done_arr, info_arr = env.step(action)
        info = info_arr[0]
        done = bool(done_arr[0])

        equity_curve.append(info['equity'])

        # action shape: (n_envs, action_dim) for MultiDiscrete
        action_name = TradeAction(int(action[0][0])).name
        if action_name != 'HOLD':
            actions_taken.append({
                'step':   step,
                'action': action_name,
                'equity': info['equity'],
            })
        step += 1

    end_price   = feed.get_current_bar().get('close', start_price)
    bh_return   = ((end_price - start_price) / start_price * 100) if start_price > 0 else 0.0

    initial_balance = 10_000.0
    final_equity    = info.get('equity', initial_balance)
    agent_return    = (final_equity - initial_balance) / initial_balance * 100

    # Max drawdown
    peak   = initial_balance
    max_dd = 0.0
    for eq in equity_curve:
        peak   = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / (peak + 1e-9))

    # Annualised Sharpe (daily proxy)
    sharpe = 0.0
    if len(equity_curve) >= 2:
        rets = np.diff([initial_balance] + equity_curve) / (
            np.array([initial_balance] + equity_curve[:-1]) + 1e-9
        )
        if rets.std() > 0:
            sharpe = float(rets.mean() / rets.std() * np.sqrt(252))

    return {
        'agent_return':   agent_return,
        'buyhold_return': bh_return,
        'beats_market':   agent_return > bh_return,
        'final_equity':   final_equity,
        'total_trades':   info.get('total_trades', 0),
        'win_rate':       info.get('win_rate', 0.0),
        'max_drawdown':   max_dd * 100,
        'sharpe_ratio':   sharpe,
        'steps':          step,
        'equity_curve':   equity_curve,
        'actions':        actions_taken,
    }


def evaluate(args):
    print(f"""
{'='*60}
  RL SWING TRADER — EVALUATION
  Model:    {args.model}
  Data:     {args.data}
  Episodes: {args.episodes}
{'='*60}
""")

    # ── resolve data file ─────────────────────────────────────────────────────
    data_path = args.data
    if not os.path.exists(data_path):
        for candidate in ["QQQ_data.csv", "nasdaq_data.csv", "SPY_data.csv"]:
            if os.path.exists(candidate):
                print(f"[warn] '{data_path}' not found — using '{candidate}'\n")
                data_path = candidate
                break
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")

    # ── resolve model path ────────────────────────────────────────────────────
    model_path = args.model
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        candidates = [
            "models/ppo_ict/best_model",
            "models/ppo_ict/ppo_ict_final",
            "models/ppo_stock/best_model",
            "models/ppo_stock/ppo_stock_final",
        ]
        for candidate in candidates:
            if os.path.exists(candidate + ".zip") or os.path.exists(candidate):
                print(f"[warn] '{model_path}' not found — using '{candidate}'\n")
                model_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Run train.py first to generate a model."
            )

    # ── load model once (not per episode) ─────────────────────────────────────
    # Build a throwaway env just to load the model with the right action/obs spaces
    _probe_feed = HistoricalDataFeed.from_csv(data_path, randomize_start=False)
    _probe_env  = DummyVecEnv([lambda: Monitor(
        TradingEnv(_probe_feed, ICTSignalAdapter(), max_steps=200)
    )])

    vecnorm_path = f"{model_path}_vecnormalize.pkl"
    if os.path.exists(vecnorm_path):
        _probe_env = VecNormalize.load(vecnorm_path, _probe_env)
        _probe_env.training    = False
        _probe_env.norm_reward = False

    model = PPO.load(model_path, env=_probe_env)

    # ── episode loop ──────────────────────────────────────────────────────────
    all_results     = []
    beats_count     = 0
    profitable_count = 0

    for ep in range(args.episodes):
        feed    = HistoricalDataFeed.from_csv(data_path, randomize_start=True)
        raw_env = TradingEnv(
            data_feed    = feed,
            ict_detector = ICTSignalAdapter(),
            max_steps    = max(1, feed.n_rows - HistoricalDataFeed.HISTORY_LEN - 1),
        )
        ep_env = DummyVecEnv([lambda e=raw_env: Monitor(e)])

        if os.path.exists(vecnorm_path):
            ep_env = VecNormalize.load(vecnorm_path, ep_env)
            ep_env.training    = False
            ep_env.norm_reward = False

        result = run_single_episode(model, ep_env, feed)
        all_results.append(result)

        if result['beats_market']:
            beats_count += 1
        if result['agent_return'] > 0:
            profitable_count += 1

        marker = ""
        if result['beats_market'] and result['agent_return'] > 0:
            marker = " << BEAT MARKET"
        elif result['agent_return'] > 0:
            marker = " (profitable)"

        print(f"  Episode {ep+1:3d}/{args.episodes} | "
              f"Agent: {result['agent_return']:+7.2f}% | "
              f"B&H: {result['buyhold_return']:+7.2f}% | "
              f"Trades: {result['total_trades']:3d} | "
              f"WR: {result['win_rate']:5.1%} | "
              f"MaxDD: {result['max_drawdown']:5.1f}%"
              f"{marker}")

    # ── summary ───────────────────────────────────────────────────────────────
    agent_returns = [r['agent_return']   for r in all_results]
    bh_returns    = [r['buyhold_return'] for r in all_results]
    win_rates     = [r['win_rate']       for r in all_results]
    drawdowns     = [r['max_drawdown']   for r in all_results]
    sharpes       = [r['sharpe_ratio']   for r in all_results]
    trades        = [r['total_trades']   for r in all_results]

    print(f"""
{'='*60}
  RESULTS SUMMARY ({args.episodes} episodes)
{'='*60}

  AGENT PERFORMANCE
  ─────────────────────────────────────────
  Avg Return:        {np.mean(agent_returns):+.2f}%
  Median Return:     {np.median(agent_returns):+.2f}%
  Best Episode:      {np.max(agent_returns):+.2f}%
  Worst Episode:     {np.min(agent_returns):+.2f}%
  Std Dev:           {np.std(agent_returns):.2f}%
  Avg Win Rate:      {np.mean(win_rates):.1%}
  Avg Trades/Ep:     {np.mean(trades):.1f}
  Avg Max Drawdown:  {np.mean(drawdowns):.1f}%
  Avg Sharpe Ratio:  {np.mean(sharpes):.2f}

  BUY & HOLD BENCHMARK
  ─────────────────────────────────────────
  Avg Return:        {np.mean(bh_returns):+.2f}%
  Median Return:     {np.median(bh_returns):+.2f}%

  HEAD-TO-HEAD
  ─────────────────────────────────────────
  Agent beats market:    {beats_count}/{args.episodes} ({beats_count/args.episodes:.1%})
  Agent is profitable:   {profitable_count}/{args.episodes} ({profitable_count/args.episodes:.1%})
  Avg excess return:     {np.mean(agent_returns) - np.mean(bh_returns):+.2f}%
""")

    print("  VERDICT")
    print("  ─────────────────────────────────────────")
    avg_agent = np.mean(agent_returns)
    avg_bh    = np.mean(bh_returns)
    profit_rate = profitable_count / args.episodes

    if avg_agent > avg_bh and profit_rate > 0.6:
        print("  The agent is BEATING the market and consistently profitable.")
        print("  Consider fine-tuning on 4H data next.")
    elif profit_rate > 0.6:
        print("  The agent is PROFITABLE but not beating buy-and-hold.")
        print("  Try training for more timesteps or tuning reward weights.")
    elif profit_rate > 0.4:
        print("  The agent is INCONSISTENT — profitable sometimes, not others.")
        print("  Needs more training data or reward shaping adjustments.")
    else:
        print("  The agent is UNDERPERFORMING. Recommendations:")
        print("  - Train for more timesteps (try 1M+)")
        print("  - Check if ICT signals are mapping correctly")
        print("  - Adjust reward weights (see RewardCalculator)")
        print("  - Try training on more diverse data")

    # ── save results ──────────────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    save_data = {
        'timestamp':   datetime.now().isoformat(),
        'model':       model_path,
        'data':        data_path,
        'num_episodes': args.episodes,
        'summary': {
            'avg_agent_return':   float(avg_agent),
            'avg_buyhold_return': float(avg_bh),
            'avg_excess_return':  float(avg_agent - avg_bh),
            'beats_market_pct':   beats_count / args.episodes,
            'profitable_pct':     profit_rate,
            'avg_win_rate':       float(np.mean(win_rates)),
            'avg_max_drawdown':   float(np.mean(drawdowns)),
            'avg_sharpe':         float(np.mean(sharpes)),
            'avg_trades':         float(np.mean(trades)),
        },
        'episodes': [{
            'agent_return':   r['agent_return'],
            'buyhold_return': r['buyhold_return'],
            'beats_market':   r['beats_market'],
            'total_trades':   r['total_trades'],
            'win_rate':       r['win_rate'],
            'max_drawdown':   r['max_drawdown'],
            'sharpe_ratio':   r['sharpe_ratio'],
        } for r in all_results],
    }

    out_path = "logs/evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Full results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL Trading Agent")
    parser.add_argument("--data",     required=True,          help="Path to OHLCV CSV")
    parser.add_argument("--model",    required=True,          help="Path to trained model (.zip)")
    parser.add_argument("--episodes", type=int, default=50,   help="Number of eval episodes")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
