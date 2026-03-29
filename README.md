# Veteran Trader v2.0 — Smart Money Concepts Engine

A disciplined, pattern-recognizing trading signal engine that combines classical technical analysis with ICT / Smart Money Concepts to generate actionable buy and short signals on NASDAQ and other equities.

Built to mimic the behavior of an experienced swing trader — patient, disciplined, and data-driven.

---

## Core Philosophy

- **Patience** — Wait for high-conviction setups. The system holds 75-85% of days and only trades when multiple indicators align.
- **Discipline** — Follow the system. Respect stop-losses. Never fight the trend. Automatic cooldowns after loss streaks.
- **Pattern Recognition** — 15+ technical indicators running simultaneously alongside Smart Money Concept detection.
- **Risk First** — Preserve capital above all else. ATR-based position sizing, portfolio heat limits, and configurable risk profiles.

---

## Features

### Classical Technical Analysis
- Moving Average crossovers (EMA 9/21, SMA 50/200 golden/death cross)
- RSI, MACD, Stochastic Oscillator, Money Flow Index
- Bollinger Bands, Keltner Channels, volatility squeeze detection
- On-Balance Volume, Cumulative Volume Delta
- Parabolic SAR, ADX trend strength
- Candlestick pattern detection (engulfing, hammer, morning/evening star, three soldiers/crows, doji)
- Pivot point support/resistance levels

### Smart Money Concepts (ICT)
- **Fair Value Gaps (FVGs)** — Detects 3-candle imbalances and tracks fill entries
- **Liquidity Sweeps** — Identifies stop hunts above/below key swing levels with reversal confirmation
- **SMT Divergences** — Internal (price vs RSI/OBV/CVD) and cross-asset (e.g., QQQ vs SPY)
- **Market Structure** — Break of Structure (BOS) and Change of Character (CHoCH) detection
- **Order Blocks** — Last opposing candle before displacement, with retest entry signals
- **Displacement Candles** — Large impulsive moves showing institutional commitment
- **Premium/Discount Zones** — Maps price within the current dealing range

### Signal Types
Every signal includes a **context** explaining the setup:
- High Confidence / Low Risk (the best — high conviction, tight risk, 3:1+ R:R, multiple SMC confirmations)
- Fair Value Gap Entry
- Liquidity Sweep Reversal
- SMT Divergence
- Order Block Entry
- Break of Structure
- Displacement / Momentum Entry
- Multi-System Confluence
- Trend Continuation, Breakout, Reversal, Momentum Shift

### Risk Profiles

| Setting | Risk/Trade | Confirmations | Trades/Days | Best For |
|---------|-----------|---------------|-------------|----------|
| Conservative | 1% | 4 required | ~4% | Capital preservation |
| Moderate | 2% | 3 required | ~12% | Balanced approach |
| Aggressive | 3% | 2 required | ~24% | More opportunities |

---

## Project Structure

```
stock/
├── veteran_trader_v2.py     # Core engine — indicators + smart money + signals
├── veteran_trader.py         # v1 backup (classical only)
├── live_fetch.py             # Downloads live data from Yahoo Finance
├── data_extractor.py         # Generates sample OHLCV data for testing
├── backtester.py             # Simulates historical performance
├── alerts.py                 # Discord / Telegram / desktop notifications
├── auto_run.py               # Daily auto-runner with scheduling
├── alert_config.json         # Your personal settings (not tracked in git)
└── alert_config_TEMPLATE.json
```

---

## Setup

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/NahomHaile/Trader-simulator.git
cd Trader-simulator
pip install yfinance pandas
```

### Configuration

Copy the template and add your webhook URLs:

```bash
cp alert_config_TEMPLATE.json alert_config.json
```

Edit `alert_config.json`:

```json
{
  "desktop_notifications": true,
  "discord_webhook_url": "YOUR_DISCORD_WEBHOOK_URL",
  "telegram_bot_token": "",
  "telegram_chat_id": "",
  "alert_on": {
    "strong_buy": true,
    "buy": true,
    "strong_sell": true,
    "sell": true,
    "lean_buy": false,
    "lean_sell": false,
    "high_confidence_low_risk": true
  },
  "min_conviction_for_alert": 0.5,
  "tickers": ["QQQ"],
  "risk_profile": "moderate",
  "capital": 100000,
  "smt_compare": "SPY"
}
```

---

## Usage

### Daily Analysis (the main command)
```bash
python auto_run.py
```
Fetches live data, runs the full analysis, and sends Discord alerts if there's a trade signal.

### Manual Analysis
```bash
python live_fetch.py --ticker QQQ
python live_fetch.py --ticker AAPL MSFT NVDA TSLA
python live_fetch.py --ticker QQQ --smt-compare SPY
python live_fetch.py --ticker QQQ --risk-profile aggressive
```

### Backtesting
```bash
python backtester.py QQQ_data.csv
python backtester.py QQQ_data.csv --risk-profile aggressive --smt-compare SPY_data.csv
```

### Alert Testing
```bash
python alerts.py --test
```

### Schedule Auto-Run (Windows)
```bash
python auto_run.py --install-schedule
```
Creates a Windows Task Scheduler job that runs every weekday at 9:00 AM.

### Sample Data (no internet needed)
```bash
python data_extractor.py
python veteran_trader_v2.py nasdaq_data.csv
```

---

## How Signals Work

The system evaluates every trading day across two layers:

**Layer 1 — Classical Technicals:** Trend direction (EMA stacks, PSAR), momentum (RSI, MACD, Stochastic), volume (OBV, CVD, volume surges), volatility (Bollinger Bands, ATR, squeeze), candlestick patterns, and support/resistance.

**Layer 2 — Smart Money Concepts:** Fair value gaps, liquidity sweeps, market structure breaks, order block retests, displacement candles, SMT divergences, and premium/discount zone awareness.

Both layers feed into a scoring engine. When enough signals agree and conviction crosses the threshold, it generates a trade signal with:
- Direction (LONG or SHORT)
- Entry price
- Stop loss (2x ATR)
- Take profit (2.5-3x risk)
- Position size (based on risk %)
- Context (why this signal fired)

The discipline engine can override signals — reducing conviction when trading against the trend, forcing cooldowns after loss streaks, and refusing to trade in low-clarity conditions.

---

## Example Alert Output

```
🔴 STRONG SELL — QQQ $588.00

========================================
ACTION: STRONG SHORT QQQ (bet on drop)

WHAT TO DO:
  1. SHORT/SELL QQQ near $588.00
     (or buy inverse ETF like SQQQ)
  2. Set STOP LOSS at $604.50
     ($16.50 above entry)
  3. Set TAKE PROFIT at $546.75
     ($41.25 below entry)
  4. Position size: 18.0% of capital

  If price rises to $604.50 -> GET OUT
  If price drops to $546.75 -> TAKE PROFIT
========================================

Confidence: HIGH (86%)
Setup Type: Liquidity Sweep Reversal
Risk/Reward: 2.5x (risking $1 to make $2.5)
```

---

## Supported Tickers

Any valid Yahoo Finance ticker works:

| Ticker | Index/ETF |
|--------|-----------|
| `^IXIC` | NASDAQ Composite |
| `QQQ` | NASDAQ 100 ETF |
| `SPY` | S&P 500 ETF |
| `^GSPC` | S&P 500 Index |
| `^DJI` | Dow Jones |

Plus any individual stock: `AAPL`, `MSFT`, `NVDA`, `TSLA`, `AMZN`, `GOOG`, etc.

---

## Disclaimer

This is an **educational tool**. It is **not financial advice**. Past patterns and backtest results do not guarantee future performance. Always do your own research, never risk more than you can afford to lose, and consider paper trading before using real capital.

---

## Tech Stack

- Python 3.10+
- pandas / numpy for data processing
- yfinance for live market data
- Discord Webhooks / Telegram Bot API for notifications
- Windows Task Scheduler for automation
- Zero external dependencies for the core engine (pure Python)

---

## License

MIT
