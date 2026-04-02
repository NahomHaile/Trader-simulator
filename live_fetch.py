#!/usr/bin/env python3
"""
Live NASDAQ Data Fetcher v2
=============================
Downloads real OHLCV data from Yahoo Finance and feeds it into
the Veteran Trader Agent v2 (Smart Money Concepts edition).

SETUP (one time):
    pip install yfinance pandas

USAGE:
    python live_fetch.py                                    # NASDAQ Composite
    python live_fetch.py --ticker AAPL                      # Single stock
    python live_fetch.py --ticker AAPL MSFT NVDA            # Multiple tickers
    python live_fetch.py --ticker QQQ --smt-compare SPY     # Cross-asset SMT
    python live_fetch.py --risk-profile aggressive --capital 50000
    python live_fetch.py --no-analyze                       # Just download
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def check_dependencies():
    try:
        import yfinance  # noqa: F401
        import pandas     # noqa: F401
    except ImportError:
        print()
        print("  Missing dependencies. Install with:")
        print("      pip install yfinance pandas")
        print("  or: py -m pip install yfinance pandas")
        print()
        sys.exit(1)


def fetch_data(ticker: str, period: str, output_dir: str = ".") -> str:
    import yfinance as yf

    safe_name = ticker.replace("^", "").replace(".", "_").replace("/", "_")
    output_path = os.path.join(output_dir, f"{safe_name}_data.csv")

    print(f"  Fetching {ticker} ({period} of daily data)...")

    try:
        tkr = yf.Ticker(ticker)
        df = tkr.history(period=period, interval="1d")
    except Exception as e:
        print(f"  ERROR: Failed to fetch {ticker}: {e}")
        return None

    if df.empty:
        print(f"  ERROR: No data returned for {ticker}. Check the ticker symbol.")
        return None

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df.index = df.index.strftime("%Y-%m-%d")
    df.dropna(inplace=True)
    df = df[df["Volume"] > 0]

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].round(2)
    df["Volume"] = df["Volume"].astype(int)

    df.to_csv(output_path)

    print(f"  Downloaded {len(df)} trading days")
    print(f"    Date range: {df.index[0]} -> {df.index[-1]}")
    print(f"    Last close:  ${df['Close'].iloc[-1]:,.2f}")
    print(f"    Saved to:    {output_path}")

    return output_path


def run_trader(csv_path: str, risk_profile: str, capital: float,
               max_risk: float = None, smt_csv: str = None):
    # Try v2 first, fall back to v1
    script_dir = os.path.dirname(os.path.abspath(__file__))
    trader_v2 = os.path.join(script_dir, "veteran_trader_v2.py")
    trader_v1 = os.path.join(script_dir, "veteran_trader.py")

    if os.path.exists(trader_v2):
        trader_script = trader_v2
        version = "v2"
    elif os.path.exists(trader_v1):
        trader_script = trader_v1
        version = "v1"
    else:
        print(f"  Cannot find veteran_trader_v2.py or veteran_trader.py in {script_dir}")
        return

    base = os.path.splitext(os.path.basename(csv_path))[0]
    output_csv = f"{base}_signals_v2.csv" if version == "v2" else f"{base}_signals.csv"

    cmd = [
        sys.executable, trader_script,
        csv_path,
        "--risk-profile", risk_profile,
        "--capital", str(capital),
        "--output", output_csv,
    ]
    if max_risk is not None:
        cmd.extend(["--max-risk", str(max_risk)])
    if smt_csv and version == "v2":
        cmd.extend(["--smt-compare", smt_csv])

    print()
    print(f"  Running Veteran Trader {version} on {os.path.basename(csv_path)}...")
    if smt_csv:
        print(f"  SMT comparison: {os.path.basename(smt_csv)}")
    print()

    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch live market data and run Veteran Trader v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python live_fetch.py                                      # QQQ (default)
  python live_fetch.py --ticker QQQ                         # QQQ
  python live_fetch.py --ticker QQQ --smt-compare SPY       # QQQ vs SPY divergence
  python live_fetch.py --ticker AAPL MSFT NVDA              # Multiple tickers
  python live_fetch.py --risk-profile conservative          # Conservative mode
  python live_fetch.py --ticker QQQ --capital 25000         # Custom capital
        """,
    )
    parser.add_argument("--ticker", nargs="+", default=["QQQ"],
                        help="Yahoo Finance ticker(s). Default: QQQ")
    parser.add_argument("--period", default="2y",
                        help="How far back. Options: 1mo, 3mo, 6mo, 1y, 2y, 5y, max")
    parser.add_argument("--smt-compare", default=None,
                        help="Correlated ticker for cross-asset SMT divergence (e.g. SPY)")
    parser.add_argument("--risk-profile", choices=["conservative", "moderate", "aggressive"],
                        default="moderate")
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--max-risk", type=float, default=None)
    parser.add_argument("--no-analyze", action="store_true",
                        help="Only download data, don't run the trader")
    parser.add_argument("--output-dir", default=".")

    args = parser.parse_args()
    check_dependencies()

    print()
    print("  ============================================")
    print("       LIVE FETCH -> VETERAN TRADER v2")
    print(f"       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  ============================================")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # Fetch SMT comparison asset if requested
    smt_csv = None
    if args.smt_compare:
        print(f"  --- Fetching SMT comparison: {args.smt_compare} ---")
        smt_csv = fetch_data(args.smt_compare, args.period, args.output_dir)
        print()

    for ticker in args.ticker:
        print(f"  {'=' * 50}")
        print(f"  Ticker: {ticker}")
        print(f"  {'=' * 50}")

        csv_path = fetch_data(ticker, args.period, args.output_dir)

        if csv_path and not args.no_analyze:
            run_trader(csv_path, args.risk_profile, args.capital, args.max_risk, smt_csv)

        print()

    print("  Done. This is an educational tool, not financial advice.")
    print()


if __name__ == "__main__":
    main()
