#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    AUTO-RUN DAILY TRADER                                   ║
║                    ────────────────────                                    ║
║  One-click daily runner: fetches live data, runs analysis, sends alerts,   ║
║  and logs everything. Set it up with Windows Task Scheduler to run         ║
║  automatically before market open.                                         ║
║                                                                            ║
║  USAGE:                                                                    ║
║    python auto_run.py                          # Use alert_config.json     ║
║    python auto_run.py --ticker QQQ AAPL NVDA   # Override tickers          ║
║    python auto_run.py --install-schedule       # Set up Windows Task       ║
║    python auto_run.py --backtest               # Also run backtest         ║
║                                                                            ║
║  SCHEDULED SETUP (runs every weekday at 9:00 AM):                          ║
║    python auto_run.py --install-schedule                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime


script_dir = os.path.dirname(os.path.abspath(__file__))


def check_dependencies():
    try:
        import yfinance  # noqa: F401
        import pandas     # noqa: F401
    except ImportError:
        print("  Missing dependencies: pip install yfinance pandas")
        sys.exit(1)


def fetch_ticker(ticker: str, period: str = "2y") -> str:
    """Download data for a single ticker. Returns CSV path."""
    import yfinance as yf

    safe = ticker.replace("^", "").replace(".", "_").replace("/", "_")
    path = os.path.join(script_dir, f"{safe}_data.csv")

    print(f"  Fetching {ticker}...")
    try:
        df = yf.Ticker(ticker).history(period=period, interval="1d")
    except Exception as e:
        print(f"    ERROR: {e}")
        return None

    if df.empty:
        print(f"    ERROR: No data for {ticker}")
        return None

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df.index = df.index.strftime("%Y-%m-%d")
    df.dropna(inplace=True)
    df = df[df["Volume"] > 0]
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].round(2)
    df["Volume"] = df["Volume"].astype(int)
    df.to_csv(path)

    print(f"    {len(df)} days -> ${df['Close'].iloc[-1]:,.2f} (last close)")
    return path


def run_analysis(csv_path: str, config: dict, smt_csv: str = None):
    """Run veteran_trader_v2 and return the output CSV path."""
    trader_script = os.path.join(script_dir, "veteran_trader_v2.py")
    if not os.path.exists(trader_script):
        print(f"  ERROR: veteran_trader_v2.py not found")
        return None

    base = os.path.splitext(os.path.basename(csv_path))[0]
    output = os.path.join(script_dir, f"{base}_signals_v2.csv")

    cmd = [
        sys.executable, trader_script, csv_path,
        "--risk-profile", config.get("risk_profile", "moderate"),
        "--capital", str(config.get("capital", 100000)),
        "--output", output,
    ]
    if smt_csv:
        cmd.extend(["--smt-compare", smt_csv])

    subprocess.run(cmd)
    return output


def run_alerts_for(csv_path: str, config: dict, smt_csv: str = None):
    """Run the alert system on a CSV."""
    alerts_script = os.path.join(script_dir, "alerts.py")
    if not os.path.exists(alerts_script):
        print(f"  alerts.py not found — skipping notifications")
        return

    cmd = [sys.executable, alerts_script, csv_path]
    if smt_csv:
        cmd.extend(["--smt-compare", smt_csv])

    subprocess.run(cmd)


def run_backtest_for(csv_path: str, config: dict, smt_csv: str = None):
    """Run the backtester on a CSV."""
    bt_script = os.path.join(script_dir, "backtester.py")
    if not os.path.exists(bt_script):
        print(f"  backtester.py not found — skipping backtest")
        return

    cmd = [
        sys.executable, bt_script, csv_path,
        "--risk-profile", config.get("risk_profile", "moderate"),
        "--capital", str(config.get("capital", 100000)),
    ]
    if smt_csv:
        cmd.extend(["--smt-compare", smt_csv])

    subprocess.run(cmd)


def log_run(tickers: list, log_path: str = None):
    """Log this run to a file."""
    if log_path is None:
        log_path = os.path.join(script_dir, "run_log.txt")

    entry = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Tickers: {', '.join(tickers)}\n"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)


def install_schedule():
    """Create a Windows Task Scheduler task to run daily at 9:00 AM."""
    if sys.platform != "win32":
        print("  Task scheduling is only supported on Windows.")
        print("  On Mac/Linux, add this to your crontab:")
        print(f"    0 9 * * 1-5 cd {script_dir} && python3 auto_run.py")
        return

    python_path = sys.executable
    script_path = os.path.join(script_dir, "auto_run.py")

    # Create the task via schtasks
    task_name = "VeteranTrader_DailyRun"
    cmd = [
        "schtasks", "/create",
        "/tn", task_name,
        "/tr", f'"{python_path}" "{script_path}"',
        "/sc", "weekly",
        "/d", "MON,TUE,WED,THU,FRI",
        "/st", "09:00",
        "/f",  # Force overwrite
    ]

    print(f"  Creating scheduled task: {task_name}")
    print(f"  Schedule: Weekdays at 9:00 AM")
    print(f"  Python:   {python_path}")
    print(f"  Script:   {script_path}")
    print()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("  Task created successfully!")
            print()
            print("  To verify:  schtasks /query /tn VeteranTrader_DailyRun")
            print("  To remove:  schtasks /delete /tn VeteranTrader_DailyRun /f")
            print("  To run now: schtasks /run /tn VeteranTrader_DailyRun")
        else:
            print(f"  Failed to create task: {result.stderr}")
            print()
            print("  Try running this command as Administrator, or create the task manually:")
            print(f"  1. Open Task Scheduler (search 'Task Scheduler' in Start)")
            print(f"  2. Create Basic Task -> Name: VeteranTrader")
            print(f"  3. Trigger: Weekly, Mon-Fri, 9:00 AM")
            print(f"  4. Action: Start a Program")
            print(f"     Program: {python_path}")
            print(f"     Arguments: \"{script_path}\"")
            print(f"     Start in: {script_dir}")
    except Exception as e:
        print(f"  Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Daily auto-run: fetch, analyze, alert")
    parser.add_argument("--ticker", nargs="*", default=None,
                        help="Override tickers (default: from alert_config.json)")
    parser.add_argument("--smt-compare", default=None,
                        help="SMT comparison ticker")
    parser.add_argument("--period", default="2y")
    parser.add_argument("--backtest", action="store_true",
                        help="Also run backtester")
    parser.add_argument("--install-schedule", action="store_true",
                        help="Install Windows Task Scheduler job")
    parser.add_argument("--no-alerts", action="store_true",
                        help="Skip alert notifications")
    args = parser.parse_args()

    if args.install_schedule:
        install_schedule()
        return

    check_dependencies()

    # Load config
    config_path = os.path.join(script_dir, "alert_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {
            "tickers": ["QQQ"],
            "risk_profile": "moderate",
            "capital": 100000,
            "smt_compare": "",
        }

    tickers = args.ticker or config.get("tickers", ["QQQ"])
    smt_ticker = args.smt_compare or config.get("smt_compare", "")

    print()
    print("  ============================================")
    print("       VETERAN TRADER — DAILY AUTO-RUN")
    print(f"       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  ============================================")
    print()
    print(f"  Tickers:      {', '.join(tickers)}")
    print(f"  Risk Profile: {config.get('risk_profile', 'moderate').upper()}")
    if smt_ticker:
        print(f"  SMT Compare:  {smt_ticker}")
    print()

    # Fetch SMT data if needed
    smt_csv = None
    if smt_ticker:
        smt_csv = fetch_ticker(smt_ticker, args.period)

    # Process each ticker
    for ticker in tickers:
        print()
        print(f"  {'=' * 50}")
        print(f"  {ticker}")
        print(f"  {'=' * 50}")

        # 1. Fetch
        csv_path = fetch_ticker(ticker, args.period)
        if not csv_path:
            continue

        # 2. Analyze
        print()
        run_analysis(csv_path, config, smt_csv)

        # 3. Alerts
        if not args.no_alerts:
            print()
            print(f"  Checking alerts for {ticker}...")
            run_alerts_for(csv_path, config, smt_csv)

        # 4. Backtest (optional)
        if args.backtest:
            print()
            print(f"  Running backtest for {ticker}...")
            run_backtest_for(csv_path, config, smt_csv)

    # Log
    log_run(tickers)

    print()
    print("  ============================================")
    print("       DAILY RUN COMPLETE")
    print("  ============================================")
    print()


if __name__ == "__main__":
    main()
