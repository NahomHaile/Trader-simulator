#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ALPACA PAPER TRADING RUNNER                             ║
║                    ───────────────────────────                             ║
║  Loads the trained RL agent and executes paper trades via Alpaca's API.    ║
║                                                                            ║
║  SETUP:                                                                    ║
║    Set environment variables:                                              ║
║        ALPACA_API_KEY=your_key_here                                        ║
║        ALPACA_SECRET_KEY=your_secret_here                                  ║
║    Or create alpaca_keys.json with {"api_key": "...", "secret_key": "..."}  ║
║                                                                            ║
║  USAGE:                                                                    ║
║    python paper_trade.py                        # Run once, default QQQ    ║
║    python paper_trade.py --symbol QQQ           # Explicit symbol          ║
║    python paper_trade.py --loop                 # Run daily in a loop      ║
║    python paper_trade.py --dry-run              # Print actions, no orders ║
║    python paper_trade.py --model checkpoints/best_model                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from collections import deque

import numpy as np

LOG_FILE     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_trade_log.txt")
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_trade_history.json")

def log(msg: str):
    """Write to both console and log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def save_daily_record(date: str, symbol: str, action: str, equity: float,
                      price: float, qty: int, has_position: bool, pnl: float = 0.0):
    """Append today's result to the history JSON file."""
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    history.append({
        "date":         date,
        "symbol":       symbol,
        "action":       action,
        "equity":       equity,
        "price":        price,
        "qty":          qty,
        "has_position": has_position,
        "unrealized_pnl": pnl,
    })
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def weekly_summary():
    """Log a weekly performance summary every Friday."""
    if not os.path.exists(HISTORY_FILE):
        return
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    if len(history) < 2:
        return

    # Last 5 trading days
    week = history[-5:] if len(history) >= 5 else history
    start_equity = week[0]["equity"]
    end_equity   = week[-1]["equity"]
    week_pnl     = end_equity - start_equity
    week_pct     = (week_pnl / start_equity) * 100

    trades  = [r for r in week if r["action"] in ("LONG", "SHORT", "CLOSE")]
    entries = [r for r in week if r["action"] in ("LONG", "SHORT")]

    # All-time stats
    all_start  = history[0]["equity"]
    all_pnl    = end_equity - all_start
    all_pct    = (all_pnl / all_start) * 100
    all_trades = [r for r in history if r["action"] in ("LONG", "SHORT")]

    log("")
    log("  ======== WEEKLY SUMMARY ========")
    log(f"  Period:        {week[0]['date']} to {week[-1]['date']}")
    log(f"  Week P&L:      ${week_pnl:+,.2f}  ({week_pct:+.2f}%)")
    log(f"  Trades this week: {len(entries)}")
    log(f"  Current equity:   ${end_equity:,.2f}")
    log(f"  All-time P&L:     ${all_pnl:+,.2f}  ({all_pct:+.2f}%)")
    log(f"  Total entries:    {len(all_trades)}")
    log("  ================================")
    log("")

# ── Make stock/ importable ────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

# ── Constants ─────────────────────────────────────────────────────────────────
PAPER_BASE  = "https://paper-api.alpaca.markets/v2"
DATA_BASE   = "https://data.alpaca.markets/v2"
KEYS_FILE   = os.path.join(_here, "alpaca_keys.json")
DEFAULT_MODEL = os.path.join(_here, "checkpoints", "trading_agent_final_20260405_180641")
DEFAULT_VECNORM = DEFAULT_MODEL + "_vecnormalize.pkl"

SIZE_MAP   = {0: 0.02, 1: 0.03, 2: 0.05, 3: 0.08}   # fraction of equity per action[1]
ACTION_MAP = {0: "HOLD", 1: "LONG", 2: "SHORT", 3: "CLOSE"}
OBS_DIM    = 28


# ── API Client ────────────────────────────────────────────────────────────────

class AlpacaClient:
    """Thin urllib wrapper for Alpaca paper trading REST API."""

    def __init__(self, api_key: str, secret_key: str):
        self.headers = {
            "APCA-API-KEY-ID":     api_key,
            "APCA-API-SECRET-KEY": secret_key,
            "Content-Type":        "application/json",
        }

    def _request(self, method: str, url: str, body: dict = None) -> dict:
        data = json.dumps(body).encode() if body else None
        req  = urllib.request.Request(url, data=data, headers=self.headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            msg = e.read().decode()
            raise RuntimeError(f"Alpaca {method} {url} → {e.code}: {msg}") from e

    def get_account(self) -> dict:
        return self._request("GET", f"{PAPER_BASE}/account")

    def get_position(self, symbol: str) -> dict | None:
        try:
            return self._request("GET", f"{PAPER_BASE}/positions/{symbol}")
        except RuntimeError as e:
            if "404" in str(e):
                return None
            raise

    def get_bars(self, symbol: str, limit: int = 150) -> list[dict]:
        """Fetch recent daily OHLCV bars via Yahoo Finance (free, no subscription needed)."""
        import yfinance as yf
        df = yf.Ticker(symbol).history(period="1y", interval="1d")
        if df.empty:
            return []
        df = df[["Open", "High", "Low", "Close", "Volume"]].tail(limit)
        return [
            {
                "open":      row["Open"],
                "high":      row["High"],
                "low":       row["Low"],
                "close":     row["Close"],
                "volume":    row["Volume"],
                "timestamp": str(idx.date()),
            }
            for idx, row in df.iterrows()
        ]

    def submit_order(self, symbol: str, notional: float, side: str,
                     order_type: str = "market", tif: str = "day") -> dict:
        """Place a fractional dollar-based market order. notional = dollar amount."""
        body = {
            "symbol":        symbol,
            "notional":      str(round(notional, 2)),
            "side":          side,
            "type":          order_type,
            "time_in_force": tif,
        }
        return self._request("POST", f"{PAPER_BASE}/orders", body)

    def close_position(self, symbol: str) -> dict:
        """Close entire position for a symbol."""
        req = urllib.request.Request(
            f"{PAPER_BASE}/positions/{symbol}",
            headers=self.headers,
            method="DELETE",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {}  # no position to close
            raise


# ── Observation builder (mirrors TradingEnv logic) ────────────────────────────

def enrich_bar(bar: dict, history: list[dict]) -> dict:
    """Add derived features to a bar dict (mirrors PaperTradingFeed._enrich_bar)."""
    enriched = dict(bar)
    n = len(history)
    enriched["returns"]    = (history[-1]["close"] - history[-2]["close"]) / history[-2]["close"] if n >= 2 else 0.0
    enriched["returns_5"]  = (history[-1]["close"] - history[-7]["close"]) / history[-7]["close"] if n >= 7 else 0.0
    if n >= 31:
        closes = [h["close"] for h in history[-31:]]
        enriched["returns_20"] = (history[-1]["close"] - history[-31]["close"]) / history[-31]["close"]
        enriched["volatility"] = float(np.std(closes) / (np.mean(closes) + 1e-9))
    else:
        enriched["returns_20"] = 0.0
        enriched["volatility"] = 0.0
    if n >= 30:
        avg_vol = np.mean([h["volume"] for h in history[-30:]])
        enriched["volume_ratio"] = bar.get("volume", 1) / max(avg_vol, 1)
    else:
        enriched["volume_ratio"] = 1.0
    enriched["spread"] = (bar.get("high", 0) - bar.get("low", 0)) / max(bar.get("close", 1), 1)
    return enriched


def detect_ict_signals(history: list[dict]) -> dict:
    """ICT signal detection fallback (mirrors ICTSignalAdapter._detect_signals)."""
    sig = {k: 0.0 for k in (
        "bullish_fvg", "bearish_fvg", "bullish_ob", "bearish_ob",
        "market_structure", "fvg_distance", "ob_strength", "liquidity_sweep",
    )}
    if len(history) < 3:
        return sig

    c1, c2, c3 = history[-3], history[-2], history[-1]

    # Fair Value Gaps
    if c3["low"] > c1["high"]:
        sig["bullish_fvg"]  = 1.0
        sig["fvg_distance"] = min((c3["low"] - c1["high"]) / (c2["close"] + 1e-9), 1.0)
    if c3["high"] < c1["low"]:
        sig["bearish_fvg"]  = 1.0
        sig["fvg_distance"] = min((c1["low"] - c3["high"]) / (c2["close"] + 1e-9), 1.0)

    # Order Blocks
    if len(history) >= 5:
        for i in range(len(history) - 5, len(history) - 1):
            ca, cb = history[i], history[i + 1]
            if ca["close"] < ca["open"] and cb["close"] > cb["open"] and cb["close"] > ca["high"]:
                sig["bullish_ob"] = 1.0
                sig["ob_strength"] = min((cb["close"] - ca["low"]) / (ca["close"] + 1e-9), 1.0)
            if ca["close"] > ca["open"] and cb["close"] < cb["open"] and cb["close"] < ca["low"]:
                sig["bearish_ob"] = 1.0
                sig["ob_strength"] = min((ca["high"] - cb["close"]) / (ca["close"] + 1e-9), 1.0)

    # Market Structure
    if len(history) >= 10:
        highs = [c["high"] for c in history[-10:]]
        lows  = [c["low"]  for c in history[-10:]]
        if highs[-1] > max(highs[:5]) and min(lows[-5:]) > min(lows[:5]):
            sig["market_structure"] = 1.0
        elif lows[-1] < min(lows[:5]) and max(highs[-5:]) < max(highs[:5]):
            sig["market_structure"] = -1.0

    # Liquidity Sweep
    if len(history) >= 20:
        recent_high = max(c["high"] for c in history[-20:-1])
        recent_low  = min(c["low"]  for c in history[-20:-1])
        last = history[-1]
        if ((last["high"] > recent_high and last["close"] < recent_high) or
                (last["low"] < recent_low  and last["close"] > recent_low)):
            sig["liquidity_sweep"] = 1.0

    return sig


def build_observation(market_data: dict, ict: dict, account: dict, step: int) -> np.ndarray:
    """Build the 21-feature observation vector (mirrors ObservationBuilder.build)."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    obs[0]  = market_data.get("returns",      0.0)
    obs[1]  = market_data.get("returns_5",    0.0)
    obs[2]  = market_data.get("returns_20",   0.0)
    obs[3]  = market_data.get("volatility",   0.0)
    obs[4]  = market_data.get("volume_ratio", 1.0)
    obs[5]  = market_data.get("spread",       0.0)

    obs[6]  = ict.get("bullish_fvg",      0.0)
    obs[7]  = ict.get("bearish_fvg",      0.0)
    obs[8]  = ict.get("bullish_ob",       0.0)
    obs[9]  = ict.get("bearish_ob",       0.0)
    obs[10] = ict.get("market_structure", 0.0)
    obs[11] = ict.get("fvg_distance",     0.0)
    obs[12] = ict.get("ob_strength",      0.0)
    obs[13] = ict.get("liquidity_sweep",  0.0)

    # Session signals — daily bars have no intraday session data, default 0
    obs[14] = 0.0  # session_asia
    obs[15] = 0.0  # session_london
    obs[16] = 0.0  # session_ny
    obs[17] = 0.0  # judas_swing_bull
    obs[18] = 0.0  # judas_swing_bear
    obs[19] = 0.0  # asia_range_sweep_high
    obs[20] = 0.0  # asia_range_sweep_low

    obs[21] = account.get("is_long",        0.0)
    obs[22] = account.get("is_short",       0.0)
    obs[23] = account.get("unrealized_pct", 0.0)
    obs[24] = account.get("drawdown",       0.0)

    obs[25] = (step % 30)  / 30.0    # weekly cycle proxy
    obs[26] = (step % 126) / 126.0   # monthly cycle proxy
    obs[27] = min(account.get("bars_in_position", 0), 60) / 60.0

    return np.clip(obs, -10.0, 10.0)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_path: str, vecnorm_path: str):
    """Load PPO model + VecNormalize for inference."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        import gymnasium as gym
        from gymnasium import spaces
    except ImportError:
        print("  ERROR: stable-baselines3 not installed.")
        print("         pip install stable-baselines3")
        sys.exit(1)

    # Minimal env with matching obs/action shape so VecNormalize loads correctly
    class _MinimalEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(-10., 10., (OBS_DIM,), np.float32)
            self.action_space      = spaces.MultiDiscrete([4, 4, 4])
        def reset(self, **kw):
            return np.zeros(OBS_DIM, np.float32), {}
        def step(self, a):
            return np.zeros(OBS_DIM, np.float32), 0., False, False, {}

    dummy = DummyVecEnv([_MinimalEnv])

    vec_env = None
    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, dummy)
        vec_env.training    = False
        vec_env.norm_reward = False
        print(f"  VecNormalize: {os.path.basename(vecnorm_path)}")
    else:
        print(f"  WARNING: VecNormalize file not found ({vecnorm_path}) — using raw observations")

    model = PPO.load(model_path, env=dummy if vec_env is None else vec_env)
    print(f"  Model:        {os.path.basename(model_path)}.zip")
    return model, vec_env


def normalize_obs(obs: np.ndarray, vec_env) -> np.ndarray:
    if vec_env is None:
        return obs.reshape(1, -1)
    return vec_env.normalize_obs(obs.reshape(1, -1))


# ── Main runner ───────────────────────────────────────────────────────────────

def load_keys() -> tuple[str, str]:
    """Load Alpaca API keys from env vars or alpaca_keys.json."""
    api_key    = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    if api_key and secret_key:
        return api_key, secret_key

    if os.path.exists(KEYS_FILE):
        with open(KEYS_FILE) as f:
            cfg = json.load(f)
        return cfg["api_key"], cfg["secret_key"]

    print()
    print("  ERROR: Alpaca API keys not found.")
    print("  Set env vars:  ALPACA_API_KEY  and  ALPACA_SECRET_KEY")
    print(f"  Or create:    {KEYS_FILE}")
    print('  Format:       {{"api_key": "...", "secret_key": "..."}}')
    print()
    sys.exit(1)


def run_once(symbol: str, model, vec_env, client: AlpacaClient,
             dry_run: bool, step_counter: list, capital_override: float = None):
    """Run one decision cycle: observe → predict → execute."""

    log("")
    log(f"  {'-'*60}")
    log(f"  {symbol} — starting decision cycle")
    log(f"  {'-'*60}")

    # ── 1. Fetch recent bars ───────────────────────────────────────────
    log("  Fetching bars from Yahoo Finance...")
    bars = client.get_bars(symbol, limit=150)
    if len(bars) < 10:
        log(f"  ERROR: Only {len(bars)} bars returned — aborting this cycle")
        return

    history = bars  # list of dicts, oldest→newest

    # ── 2. Build market data features ─────────────────────────────────
    current_bar  = history[-1]
    market_data  = enrich_bar(current_bar, history)
    ict_signals  = detect_ict_signals(history)

    log(f"  Latest bar: {current_bar.get('timestamp', 'N/A')[:10]}  "
        f"close=${current_bar['close']:,.2f}  "
        f"vol={current_bar['volume']:,}")

    # ── 3. Sync account state with Alpaca ─────────────────────────────
    acct          = client.get_account()
    equity        = float(acct["equity"])
    cash          = float(acct["cash"])
    peak_equity   = float(acct.get("last_equity", equity))  # approximate
    drawdown      = max(0.0, (peak_equity - equity) / (peak_equity + 1e-9))

    position      = client.get_position(symbol)
    is_long       = 0.0
    is_short      = 0.0
    unrealized_pct = 0.0
    bars_in_pos   = step_counter[1]   # bars since last open (tracked externally)

    if position:
        qty = float(position["qty"])
        if qty > 0:
            is_long  = 1.0
        elif qty < 0:
            is_short = 1.0
        unrealized_pnl  = float(position.get("unrealized_pl", 0.0))
        unrealized_pct  = unrealized_pnl / max(equity, 1.0)

    account_state = {
        "is_long":          is_long,
        "is_short":         is_short,
        "unrealized_pct":   unrealized_pct,
        "drawdown":         drawdown,
        "bars_in_position": bars_in_pos,
    }

    log(f"  Account: equity=${equity:,.2f}  cash=${cash:,.2f}  "
        f"drawdown={drawdown:.1%}  "
        f"{'LONG' if is_long else 'SHORT' if is_short else 'FLAT'}")

    # ── 4. Build + normalize observation ──────────────────────────────
    step  = step_counter[0]
    obs   = build_observation(market_data, ict_signals, account_state, step)
    obs_n = normalize_obs(obs, vec_env)

    # ── 5. Predict action ─────────────────────────────────────────────
    action, _ = model.predict(obs_n, deterministic=True)
    action     = action[0] if hasattr(action[0], '__len__') else action

    trade_decision = int(action[0])
    size_idx       = int(action[1])
    setup_filter   = int(action[2])
    size_pct       = SIZE_MAP[size_idx]
    action_name    = ACTION_MAP[trade_decision]

    log(f"  Agent decision: {action_name}  size={size_pct:.0%}  filter={setup_filter}")

    # ── 6. Execute order ───────────────────────────────────────────────
    price  = current_bar["close"]
    has_position = bool(position and float(position["qty"]) != 0)

    sizing_capital = capital_override if capital_override else equity

    if trade_decision == 1 and not has_position:      # LONG
        notional = round(sizing_capital * size_pct, 2)
        log(f"  {'[DRY RUN] ' if dry_run else ''}BUY ${notional:.2f} of {symbol} @ ~${price:,.2f}")
        if not dry_run:
            order = client.submit_order(symbol, notional, "buy")
            log(f"    Order ID: {order.get('id', 'N/A')}")
        step_counter[1] = 0

    elif trade_decision == 2 and not has_position:    # SHORT
        notional = round(sizing_capital * size_pct, 2)
        log(f"  {'[DRY RUN] ' if dry_run else ''}SELL SHORT ${notional:.2f} of {symbol} @ ~${price:,.2f}")
        if not dry_run:
            order = client.submit_order(symbol, notional, "sell")
            log(f"    Order ID: {order.get('id', 'N/A')}")
        step_counter[1] = 0

    elif trade_decision == 3 and has_position:        # CLOSE
        log(f"  {'[DRY RUN] ' if dry_run else ''}CLOSE position in {symbol}")
        if not dry_run:
            client.close_position(symbol)
        step_counter[1] = 0

    else:
        log(f"  Holding — no order placed.")
        if has_position:
            step_counter[1] += 1

    step_counter[0] += 1

    # ── 7. ICT signal summary ──────────────────────────────────────────
    active = [k for k, v in ict_signals.items() if v > 0.5]
    if active:
        log(f"  ICT signals: {', '.join(active)}")

    # ── 8. Save daily record ───────────────────────────────────────────
    notional_executed = round(equity * size_pct, 2) if trade_decision in (1, 2) and not has_position else 0.0
    save_daily_record(
        date         = current_bar.get("timestamp", "")[:10],
        symbol       = symbol,
        action       = action_name,
        equity       = equity,
        price        = price,
        qty          = notional_executed,
        has_position = has_position or trade_decision in (1, 2),
        pnl          = float(position.get("unrealized_pl", 0.0)) if position else 0.0,
    )

    # ── 9. Weekly summary on Fridays ───────────────────────────────────
    if datetime.now().weekday() == 4:   # 4 = Friday
        weekly_summary()

    log("")


def main():
    parser = argparse.ArgumentParser(
        description="Run trained RL agent on Alpaca paper trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--symbol",   default="QQQ",         help="Ticker to trade (default: QQQ)")
    parser.add_argument("--model",    default=DEFAULT_MODEL,  help="Path to model checkpoint (no .zip)")
    parser.add_argument("--vecnorm",  default=None,           help="Path to VecNormalize .pkl")
    parser.add_argument("--loop",     action="store_true",    help="Loop once per day (default: run once)")
    parser.add_argument("--interval", type=int, default=86400,help="Loop interval in seconds (default: 86400 = 1 day)")
    parser.add_argument("--dry-run",  action="store_true",    help="Print actions, don't submit orders")
    parser.add_argument("--capital",  type=float, default=None,
                        help="Override capital for position sizing (e.g. 20 for $20 account)")
    args = parser.parse_args()

    vecnorm_path = args.vecnorm or (args.model + "_vecnormalize.pkl")

    print()
    print("  =" * 35)
    print("  ALPACA PAPER TRADING RUNNER")
    print(f"  Symbol:  {args.symbol}")
    print(f"  {'[DRY RUN MODE]' if args.dry_run else 'LIVE PAPER TRADING'}")
    print("  =" * 35)
    print()

    # Load keys + client
    api_key, secret_key = load_keys()
    client = AlpacaClient(api_key, secret_key)

    # Verify connection
    try:
        acct = client.get_account()
        print(f"  Connected to Alpaca paper account: {acct['id']}")
        print(f"  Equity: ${float(acct['equity']):,.2f}  Cash: ${float(acct['cash']):,.2f}")
    except Exception as e:
        print(f"  ERROR connecting to Alpaca: {e}")
        sys.exit(1)

    # Load model
    print()
    print("  Loading model...")
    model, vec_env = load_model(args.model, vecnorm_path)
    print("  Model loaded.")
    print()

    # step_counter[0] = global step count, step_counter[1] = bars in current position
    step_counter = [0, 0]

    if args.loop:
        print(f"  Running in loop mode (every {args.interval}s). Ctrl+C to stop.")
        while True:
            try:
                run_once(args.symbol, model, vec_env, client, args.dry_run, step_counter, args.capital)
            except Exception as e:
                print(f"  ERROR in cycle: {e}")
            print(f"  Sleeping {args.interval}s until next bar...")
            time.sleep(args.interval)
    else:
        run_once(args.symbol, model, vec_env, client, args.dry_run, step_counter, args.capital)
        print("  Done. Run with --loop to run continuously, or schedule daily via Task Scheduler.")


if __name__ == "__main__":
    main()
