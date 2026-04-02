#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    BACKTEST ENGINE v1.0                                     ║
║                    ─────────────────────                                   ║
║  Simulates how the Veteran Trader v2 signals would have performed          ║
║  historically. Tracks P&L, win rate, max drawdown, Sharpe ratio,           ║
║  and per-trade breakdowns for both LONG and SHORT positions.               ║
║                                                                            ║
║  USAGE:                                                                    ║
║    python backtester.py QQQ_data.csv                                       ║
║    python backtester.py QQQ_data.csv --risk-profile aggressive             ║
║    python backtester.py QQQ_data.csv --smt-compare SPY_data.csv            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Import the trader
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from veteran_trader_v2 import (
    OHLCV, TraderConfig, RiskProfile, VeteranTrader, Signal, SignalContext,
    load_data
)


class PositionType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Trade:
    entry_date: str
    exit_date: str
    position: PositionType
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    shares: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # "take_profit", "stop_loss", "signal_flip", "end_of_data"
    signal_context: str
    conviction: float
    holding_days: int


@dataclass
class BacktestResult:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    avg_holding_days: float = 0.0
    long_trades: int = 0
    short_trades: int = 0
    long_wins: int = 0
    short_wins: int = 0
    long_pnl: float = 0.0
    short_pnl: float = 0.0
    best_trade: Optional[Trade] = None
    worst_trade: Optional[Trade] = None
    equity_curve: list = field(default_factory=list)
    trades: list = field(default_factory=list)


def run_backtest(data: list[OHLCV], config: TraderConfig,
                 smt_data: Optional[list[OHLCV]] = None) -> BacktestResult:
    """Run a full backtest simulation."""

    print("  Running signal generation for backtest...")
    trader = VeteranTrader(data, config, smt_data)
    signals = trader.analyze()

    capital = config.starting_capital
    current_capital = capital
    peak_capital = capital
    max_dd = 0.0
    max_dd_pct = 0.0

    result = BacktestResult()
    result.equity_curve = [capital]

    # State
    in_position = False
    position_type = None
    entry_price = 0.0
    entry_date = ""
    shares = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    entry_context = ""
    entry_conviction = 0.0
    entry_idx = 0

    daily_returns = []
    consecutive_losses = 0
    cooldown_remaining = 0

    for idx, sig in enumerate(signals):
        price = sig.price

        # Check exit conditions if in position
        if in_position:
            hit_stop = False
            hit_tp = False
            bar_idx = idx + trader.n - len(signals)
            actual_high = data[bar_idx].high
            actual_low = data[bar_idx].low

            if position_type == PositionType.LONG:
                if actual_low <= stop_loss:
                    hit_stop = True
                    exit_price = stop_loss
                elif actual_high >= take_profit:
                    hit_tp = True
                    exit_price = take_profit

                # Signal flip to sell
                signal_flip = sig.signal in (Signal.SELL, Signal.STRONG_SELL)

            else:  # SHORT
                if actual_high >= stop_loss:
                    hit_stop = True
                    exit_price = stop_loss
                elif actual_low <= take_profit:
                    hit_tp = True
                    exit_price = take_profit

                signal_flip = sig.signal in (Signal.BUY, Signal.STRONG_BUY)

            should_exit = hit_stop or hit_tp or signal_flip

            if should_exit:
                if hit_stop:
                    exit_price = stop_loss
                    exit_reason = "stop_loss"
                elif hit_tp:
                    exit_price = take_profit
                    exit_reason = "take_profit"
                else:
                    exit_price = price
                    exit_reason = "signal_flip"

                # Calculate P&L
                if position_type == PositionType.LONG:
                    pnl = (exit_price - entry_price) * shares
                else:
                    pnl = (entry_price - exit_price) * shares

                pnl_pct = pnl / (entry_price * shares) * 100 if entry_price * shares > 0 else 0
                holding_days = idx - entry_idx

                trade = Trade(
                    entry_date=entry_date, exit_date=sig.date,
                    position=position_type,
                    entry_price=round(entry_price, 2),
                    exit_price=round(exit_price, 2),
                    stop_loss=round(stop_loss, 2),
                    take_profit=round(take_profit, 2),
                    shares=round(shares, 2),
                    pnl=round(pnl, 2), pnl_pct=round(pnl_pct, 2),
                    exit_reason=exit_reason,
                    signal_context=entry_context,
                    conviction=entry_conviction,
                    holding_days=holding_days,
                )
                result.trades.append(trade)

                current_capital += pnl
                trader.current_capital = current_capital
                in_position = False

                # Track consecutive losses for cooldown
                if pnl <= 0:
                    consecutive_losses += 1
                    if consecutive_losses >= config.max_consecutive_losses:
                        cooldown_remaining = config.cooldown_after_streak
                        consecutive_losses = 0
                else:
                    consecutive_losses = 0

                if cooldown_remaining > 0:
                    cooldown_remaining -= 1

        # Track equity
        result.equity_curve.append(current_capital)
        if current_capital > peak_capital:
            peak_capital = current_capital
        dd = peak_capital - current_capital
        dd_pct = dd / peak_capital * 100 if peak_capital > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct

        # Daily return
        if len(result.equity_curve) >= 2:
            prev_eq = result.equity_curve[-2]
            if prev_eq > 0:
                daily_returns.append((current_capital - prev_eq) / prev_eq)

        # Entry conditions
        if not in_position and cooldown_remaining == 0:
            is_buy = sig.signal in (Signal.STRONG_BUY, Signal.BUY)
            is_sell = sig.signal in (Signal.STRONG_SELL, Signal.SELL)

            if is_buy or is_sell:
                in_position = True
                entry_price = price
                entry_date = sig.date
                stop_loss = sig.stop_loss
                take_profit = sig.take_profit
                entry_context = sig.context.value
                entry_conviction = sig.conviction
                entry_idx = idx

                if is_buy:
                    position_type = PositionType.LONG
                else:
                    position_type = PositionType.SHORT

                # Position sizing
                risk_per_share = abs(price - stop_loss)
                if risk_per_share > 0:
                    max_risk = current_capital * config.max_risk_per_trade
                    shares = max_risk / risk_per_share
                    max_shares = (current_capital * config.max_position_pct) / price
                    shares = min(shares, max_shares)
                else:
                    shares = 0

    # Close any open position at end
    if in_position and len(signals) > 0:
        last_price = signals[-1].price
        if position_type == PositionType.LONG:
            pnl = (last_price - entry_price) * shares
        else:
            pnl = (entry_price - last_price) * shares
        pnl_pct = pnl / (entry_price * shares) * 100 if entry_price * shares > 0 else 0

        trade = Trade(
            entry_date=entry_date, exit_date=signals[-1].date,
            position=position_type,
            entry_price=round(entry_price, 2),
            exit_price=round(last_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            shares=round(shares, 2),
            pnl=round(pnl, 2), pnl_pct=round(pnl_pct, 2),
            exit_reason="end_of_data",
            signal_context=entry_context,
            conviction=entry_conviction,
            holding_days=len(signals) - entry_idx,
        )
        result.trades.append(trade)
        current_capital += pnl

    # Compute stats
    result.total_trades = len(result.trades)
    if result.total_trades == 0:
        return result

    winners = [t for t in result.trades if t.pnl > 0]
    losers = [t for t in result.trades if t.pnl <= 0]

    result.winning_trades = len(winners)
    result.losing_trades = len(losers)
    result.total_pnl = round(sum(t.pnl for t in result.trades), 2)
    result.total_pnl_pct = round(result.total_pnl / capital * 100, 2)
    result.max_drawdown = round(max_dd, 2)
    result.max_drawdown_pct = round(max_dd_pct, 2)
    result.win_rate = round(len(winners) / result.total_trades * 100, 1)
    result.avg_win = round(sum(t.pnl for t in winners) / len(winners), 2) if winners else 0
    result.avg_loss = round(sum(t.pnl for t in losers) / len(losers), 2) if losers else 0
    gross_profit = sum(t.pnl for t in winners)
    gross_loss = abs(sum(t.pnl for t in losers))
    result.profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf')
    result.avg_holding_days = round(sum(t.holding_days for t in result.trades) / result.total_trades, 1)

    # Sharpe ratio (annualized)
    if daily_returns and len(daily_returns) > 1:
        mean_ret = sum(daily_returns) / len(daily_returns)
        std_ret = math.sqrt(sum((r - mean_ret) ** 2 for r in daily_returns) / (len(daily_returns) - 1))
        result.sharpe_ratio = round((mean_ret / std_ret) * math.sqrt(252), 2) if std_ret > 0 else 0
    else:
        result.sharpe_ratio = 0

    # Long/Short breakdown
    longs = [t for t in result.trades if t.position == PositionType.LONG]
    shorts = [t for t in result.trades if t.position == PositionType.SHORT]
    result.long_trades = len(longs)
    result.short_trades = len(shorts)
    result.long_wins = len([t for t in longs if t.pnl > 0])
    result.short_wins = len([t for t in shorts if t.pnl > 0])
    result.long_pnl = round(sum(t.pnl for t in longs), 2)
    result.short_pnl = round(sum(t.pnl for t in shorts), 2)

    if result.trades:
        result.best_trade = max(result.trades, key=lambda t: t.pnl)
        result.worst_trade = min(result.trades, key=lambda t: t.pnl)

    return result


def print_backtest_report(result: BacktestResult, config: TraderConfig):
    """Print a detailed backtest report."""
    print()
    print("=" * 70)
    print("              BACKTEST RESULTS — VETERAN TRADER v2.0")
    print("=" * 70)

    if result.total_trades == 0:
        print("  No trades were generated during the backtest period.")
        print("  Try a longer time period or more aggressive risk profile.")
        return

    print(f"  Starting Capital:  ${config.starting_capital:>12,.2f}")
    print(f"  Final Capital:     ${config.starting_capital + result.total_pnl:>12,.2f}")
    print(f"  Total P&L:         ${result.total_pnl:>12,.2f}  ({result.total_pnl_pct:+.2f}%)")
    print()

    print("-" * 70)
    print("  PERFORMANCE METRICS")
    print("-" * 70)
    print(f"    Total Trades:      {result.total_trades}")
    print(f"    Win Rate:          {result.win_rate}%")
    print(f"    Profit Factor:     {result.profit_factor}")
    print(f"    Sharpe Ratio:      {result.sharpe_ratio}")
    print(f"    Max Drawdown:      ${result.max_drawdown:,.2f} ({result.max_drawdown_pct:.2f}%)")
    print(f"    Avg Holding Days:  {result.avg_holding_days}")
    print()
    print(f"    Avg Win:           ${result.avg_win:>10,.2f}")
    print(f"    Avg Loss:          ${result.avg_loss:>10,.2f}")
    ratio = abs(result.avg_win / result.avg_loss) if result.avg_loss != 0 else 0
    print(f"    Win/Loss Ratio:    {ratio:.2f}x")
    print()

    print("-" * 70)
    print("  LONG vs SHORT BREAKDOWN")
    print("-" * 70)
    long_wr = (result.long_wins / result.long_trades * 100) if result.long_trades > 0 else 0
    short_wr = (result.short_wins / result.short_trades * 100) if result.short_trades > 0 else 0
    print(f"    LONG  trades: {result.long_trades:>4}  |  wins: {result.long_wins:>3} ({long_wr:.0f}%)  |  P&L: ${result.long_pnl:>10,.2f}")
    print(f"    SHORT trades: {result.short_trades:>4}  |  wins: {result.short_wins:>3} ({short_wr:.0f}%)  |  P&L: ${result.short_pnl:>10,.2f}")
    print()

    # Exit reason breakdown
    exit_reasons = {}
    for t in result.trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
    print("-" * 70)
    print("  EXIT REASONS")
    print("-" * 70)
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason:<20} {count:>4}")
    print()

    # Context performance
    ctx_stats = {}
    for t in result.trades:
        ctx = t.signal_context
        if ctx not in ctx_stats:
            ctx_stats[ctx] = {"trades": 0, "wins": 0, "pnl": 0.0}
        ctx_stats[ctx]["trades"] += 1
        if t.pnl > 0:
            ctx_stats[ctx]["wins"] += 1
        ctx_stats[ctx]["pnl"] += t.pnl

    print("-" * 70)
    print("  PERFORMANCE BY SIGNAL CONTEXT")
    print("-" * 70)
    for ctx, stats in sorted(ctx_stats.items(), key=lambda x: -x[1]["pnl"]):
        wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
        print(f"    {ctx:<34} {stats['trades']:>3} trades  {wr:>5.0f}% WR  ${stats['pnl']:>10,.2f}")
    print()

    # Best/Worst trades
    if result.best_trade:
        t = result.best_trade
        print("-" * 70)
        print("  BEST TRADE")
        print(f"    {t.position.value} {t.entry_date} -> {t.exit_date}  ${t.entry_price} -> ${t.exit_price}")
        print(f"    P&L: ${t.pnl:,.2f} ({t.pnl_pct:+.2f}%)  |  {t.signal_context}  |  {t.exit_reason}")
    if result.worst_trade:
        t = result.worst_trade
        print("  WORST TRADE")
        print(f"    {t.position.value} {t.entry_date} -> {t.exit_date}  ${t.entry_price} -> ${t.exit_price}")
        print(f"    P&L: ${t.pnl:,.2f} ({t.pnl_pct:+.2f}%)  |  {t.signal_context}  |  {t.exit_reason}")
    print()

    # Recent trades
    print("-" * 70)
    print("  LAST 15 TRADES")
    print("-" * 70)
    for t in result.trades[-15:]:
        arrow = ">>" if t.position == PositionType.LONG else "<<"
        win = "W" if t.pnl > 0 else "L"
        print(f"    {t.entry_date} -> {t.exit_date}  {arrow} {t.position.value:<5}  "
              f"${t.entry_price:>9,.2f} -> ${t.exit_price:>9,.2f}  "
              f"{win} ${t.pnl:>9,.2f}  ({t.exit_reason})")
    print()

    print("=" * 70)
    print("  DISCLAIMER: Past performance does not guarantee future results.")
    print("=" * 70)
    print()


def save_backtest(result: BacktestResult, output_path: str):
    """Save backtest results to CSV and JSON."""
    # Trades CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Entry_Date", "Exit_Date", "Position", "Entry_Price", "Exit_Price",
            "Stop_Loss", "Take_Profit", "Shares", "PnL", "PnL_%",
            "Exit_Reason", "Context", "Conviction", "Holding_Days"
        ])
        for t in result.trades:
            writer.writerow([
                t.entry_date, t.exit_date, t.position.value,
                t.entry_price, t.exit_price, t.stop_loss, t.take_profit,
                t.shares, t.pnl, t.pnl_pct,
                t.exit_reason, t.signal_context, t.conviction, t.holding_days,
            ])

    # Summary JSON
    json_path = output_path.replace(".csv", "_summary.json")
    summary = {
        "total_trades": result.total_trades,
        "win_rate": result.win_rate,
        "total_pnl": result.total_pnl,
        "total_pnl_pct": result.total_pnl_pct,
        "profit_factor": result.profit_factor,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "max_drawdown_pct": result.max_drawdown_pct,
        "avg_holding_days": result.avg_holding_days,
        "long_trades": result.long_trades,
        "short_trades": result.short_trades,
        "long_pnl": result.long_pnl,
        "short_pnl": result.short_pnl,
        "equity_curve": result.equity_curve,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  Trades CSV:    {output_path}")
    print(f"  Summary JSON:  {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Backtest the Veteran Trader v2 signals")
    parser.add_argument("datafile", help="OHLCV CSV file")
    parser.add_argument("--risk-profile", choices=["conservative", "moderate", "aggressive"],
                        default="moderate")
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--max-risk", type=float, default=None)
    parser.add_argument("--smt-compare", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print()
    print("  ============================================")
    print("       BACKTESTER — VETERAN TRADER v2.0")
    print("  ============================================")
    print()

    config = TraderConfig(risk_profile=RiskProfile(args.risk_profile), starting_capital=args.capital)
    config.adjust_for_profile()
    if args.max_risk:
        config.max_risk_per_trade = args.max_risk

    print(f"  Risk Profile:   {config.risk_profile.value.upper()}")
    print(f"  Capital:        ${config.starting_capital:,.2f}")
    print()

    data = load_data(args.datafile)
    smt_data = None
    if args.smt_compare:
        print(f"  Loading SMT comparison: {args.smt_compare}")
        smt_data = load_data(args.smt_compare)

    result = run_backtest(data, config, smt_data)
    print_backtest_report(result, config)

    output = args.output
    if output is None:
        base = os.path.splitext(os.path.basename(args.datafile))[0]
        output = f"{base}_backtest.csv"
    save_backtest(result, output)


if __name__ == "__main__":
    main()
