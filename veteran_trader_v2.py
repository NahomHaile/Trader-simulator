#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    VETERAN TRADER AGENT v2.0                               ║
║                    ─────────────────────────                               ║
║  Advanced signal engine combining classical technicals with ICT / Smart    ║
║  Money Concepts: Fair Value Gaps, Liquidity Sweeps, SMT Divergences,       ║
║  Market Structure, Order Blocks, and Displacement analysis.                ║
║                                                                            ║
║  NEW IN v2:                                                                ║
║    • Fair Value Gaps (bullish/bearish FVG detection + fill entries)         ║
║    • Liquidity Sweeps (stop hunts above/below key levels)                  ║
║    • SMT Divergences (internal structure + cross-indicator)                 ║
║    • Market Structure (BOS, CHoCH, swing points)                           ║
║    • Order Blocks (last opposing candle before displacement)               ║
║    • Displacement Detection (large impulsive candles)                      ║
║    • Context-Aware Signals (WHY, not just what)                            ║
║    • High Confidence / Low Risk special signals                            ║
║    • Premium/Discount zone awareness                                       ║
║                                                                            ║
║  USAGE:                                                                    ║
║    python veteran_trader_v2.py nasdaq_data.csv                             ║
║    python veteran_trader_v2.py data.csv --risk-profile conservative        ║
║    python veteran_trader_v2.py data.csv --capital 50000 --max-risk 0.01    ║
║    python veteran_trader_v2.py data.csv --smt-compare SPY_data.csv         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ════════════════════════════════════════════════════════════════════════════════
#  ENUMS & TYPES
# ════════════════════════════════════════════════════════════════════════════════

class Signal(Enum):
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    LEAN_BUY = "LEAN BUY"
    HOLD = "HOLD"
    LEAN_SELL = "LEAN SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"
    NO_TRADE = "NO TRADE"


class SignalContext(Enum):
    """WHY the signal fired — not just the direction, but the setup type."""
    HIGH_CONFIDENCE_LOW_RISK = "High Confidence / Low Risk"
    FVG_ENTRY = "Fair Value Gap Entry"
    LIQUIDITY_SWEEP = "Liquidity Sweep Reversal"
    SMT_DIVERGENCE = "SMT Divergence"
    ORDER_BLOCK_ENTRY = "Order Block Entry"
    DISPLACEMENT_ENTRY = "Displacement / Momentum Entry"
    BREAK_OF_STRUCTURE = "Break of Structure"
    TREND_CONTINUATION = "Trend Continuation"
    BREAKOUT = "Breakout"
    REVERSAL = "Mean Reversion / Reversal"
    MOMENTUM_SHIFT = "Momentum Shift"
    PREMIUM_DISCOUNT = "Premium/Discount Zone"
    CONFLUENCE = "Multi-System Confluence"
    STANDARD = "Standard Technical Signal"


class Regime(Enum):
    STRONG_UPTREND = "Strong Uptrend"
    UPTREND = "Uptrend"
    SIDEWAYS = "Sideways / Choppy"
    DOWNTREND = "Downtrend"
    STRONG_DOWNTREND = "Strong Downtrend"
    VOLATILE = "Volatile / Uncertain"


class RiskProfile(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class StructureType(Enum):
    HIGHER_HIGH = "HH"
    HIGHER_LOW = "HL"
    LOWER_HIGH = "LH"
    LOWER_LOW = "LL"
    BREAK_OF_STRUCTURE = "BOS"
    CHANGE_OF_CHARACTER = "CHoCH"


# ════════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class TraderConfig:
    risk_profile: RiskProfile = RiskProfile.MODERATE
    starting_capital: float = 100_000.0

    # Risk management
    max_risk_per_trade: float = 0.02
    max_portfolio_heat: float = 0.06
    max_consecutive_losses: int = 3
    cooldown_after_streak: int = 5

    # Patience
    min_signal_strength: float = 0.55
    min_confirmations: int = 3
    wait_for_retest: bool = True

    # Position management
    take_profit_ratio: float = 2.5
    trailing_stop_pct: float = 0.02
    scale_in: bool = True
    max_position_pct: float = 0.25

    # Classic thresholds
    volume_surge_threshold: float = 1.5
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    bb_squeeze_threshold: float = 0.03

    # Smart Money thresholds
    fvg_min_gap_atr: float = 0.3        # Min FVG size as ATR multiple
    displacement_min_body_atr: float = 1.5  # Min body for displacement candle
    liquidity_sweep_lookback: int = 20   # Bars to find swing highs/lows
    liquidity_sweep_threshold: float = 0.002  # How far price must poke past level
    order_block_lookback: int = 15
    swing_lookback: int = 5              # Bars each side for swing detection
    structure_lookback: int = 50

    # High Confidence / Low Risk thresholds
    hclr_min_conviction: float = 0.80
    hclr_max_atr_risk_pct: float = 1.5  # Max risk as % of price via ATR
    hclr_min_rr: float = 3.0            # Min R:R for HCLR signal

    def adjust_for_profile(self):
        if self.risk_profile == RiskProfile.CONSERVATIVE:
            self.max_risk_per_trade = 0.01
            self.max_portfolio_heat = 0.03
            self.min_signal_strength = 0.65
            self.min_confirmations = 4
            self.take_profit_ratio = 3.0
            self.max_position_pct = 0.15
            self.max_consecutive_losses = 2
            self.cooldown_after_streak = 8
            self.hclr_min_conviction = 0.85
            self.hclr_min_rr = 3.5
        elif self.risk_profile == RiskProfile.AGGRESSIVE:
            self.max_risk_per_trade = 0.03
            self.max_portfolio_heat = 0.10
            self.min_signal_strength = 0.40
            self.min_confirmations = 2
            self.take_profit_ratio = 2.0
            self.max_position_pct = 0.35
            self.max_consecutive_losses = 4
            self.cooldown_after_streak = 3
            self.hclr_min_conviction = 0.72
            self.hclr_min_rr = 2.5


# ════════════════════════════════════════════════════════════════════════════════
#  DATA
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class OHLCV:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


def load_data(filepath: str) -> list[OHLCV]:
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)

    rows = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = {h.strip().lower(): h for h in reader.fieldnames}

        col_map = {}
        for target, aliases in {
            "date": ["date", "datetime", "time", "timestamp", "day"],
            "open": ["open", "open price", "o"],
            "high": ["high", "high price", "h"],
            "low": ["low", "low price", "l"],
            "close": ["close", "close price", "c", "adj close", "adjusted close"],
            "volume": ["volume", "vol", "v"],
        }.items():
            for alias in aliases:
                if alias in headers:
                    col_map[target] = headers[alias]
                    break

        missing = [k for k in ["date", "open", "high", "low", "close", "volume"] if k not in col_map]
        if missing:
            print(f"ERROR: Missing columns: {missing}")
            sys.exit(1)

        for row in reader:
            try:
                rows.append(OHLCV(
                    date=row[col_map["date"]].strip(),
                    open=float(row[col_map["open"]].replace(",", "")),
                    high=float(row[col_map["high"]].replace(",", "")),
                    low=float(row[col_map["low"]].replace(",", "")),
                    close=float(row[col_map["close"]].replace(",", "")),
                    volume=int(float(row[col_map["volume"]].replace(",", ""))),
                ))
            except (ValueError, KeyError):
                continue

    if len(rows) < 50:
        print(f"ERROR: Need at least 50 data points, got {len(rows)}")
        sys.exit(1)

    print(f"  Loaded {len(rows)} trading days from {rows[0].date} to {rows[-1].date}")
    return rows


# ════════════════════════════════════════════════════════════════════════════════
#  SMART MONEY STRUCTURES
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class FairValueGap:
    """A 3-candle imbalance where price left a gap."""
    index: int
    direction: str          # "bullish" or "bearish"
    top: float              # Upper edge of the gap
    bottom: float           # Lower edge of the gap
    midpoint: float
    size: float             # Gap size in price
    size_atr: float         # Gap size as ATR multiple
    filled: bool = False
    fill_index: Optional[int] = None
    created_date: str = ""


@dataclass
class LiquiditySweep:
    """Price poking past a key level to grab stops, then reversing."""
    index: int
    direction: str          # "bullish" (swept lows, going up) or "bearish" (swept highs, going down)
    swept_level: float      # The level that was swept
    sweep_extreme: float    # How far price went past
    reversal_close: float   # Where price closed after sweep
    strength: float         # 0-1 quality score
    created_date: str = ""


@dataclass
class OrderBlock:
    """Last opposing candle before a displacement move."""
    index: int
    direction: str          # "bullish" or "bearish"
    top: float
    bottom: float
    midpoint: float
    displacement_size: float  # How strong was the move away
    tested: bool = False
    created_date: str = ""


@dataclass
class SwingPoint:
    """A confirmed swing high or low."""
    index: int
    type: str               # "high" or "low"
    price: float
    date: str = ""


@dataclass
class StructureBreak:
    """Break of Structure (BOS) or Change of Character (CHoCH)."""
    index: int
    type: StructureType
    level: float
    direction: str          # "bullish" or "bearish"
    date: str = ""


@dataclass
class SMTDivergence:
    """Smart Money Technique divergence."""
    index: int
    type: str               # "bullish" or "bearish"
    description: str
    strength: float
    date: str = ""


# ════════════════════════════════════════════════════════════════════════════════
#  TECHNICAL INDICATORS ENGINE
# ════════════════════════════════════════════════════════════════════════════════

class Indicators:
    def __init__(self, data: list[OHLCV]):
        self.data = data
        self.n = len(data)
        self.closes = [d.close for d in data]
        self.highs = [d.high for d in data]
        self.lows = [d.low for d in data]
        self.opens = [d.open for d in data]
        self.volumes = [d.volume for d in data]
        self._compute_all()

    def _compute_all(self):
        # Moving Averages
        self.sma_10 = self._sma(self.closes, 10)
        self.sma_20 = self._sma(self.closes, 20)
        self.sma_50 = self._sma(self.closes, 50)
        self.sma_200 = self._sma(self.closes, 200)
        self.ema_9 = self._ema(self.closes, 9)
        self.ema_21 = self._ema(self.closes, 21)
        self.ema_55 = self._ema(self.closes, 55)

        # VWAP approximation (cumulative)
        self.vwap = self._vwap()

        # Volume
        self.vol_sma_20 = self._sma(self.volumes, 20)
        self.vol_ratio = self._safe_divide(self.volumes, self.vol_sma_20)
        self.obv = self._obv()
        self.obv_sma = self._sma(self.obv, 20)
        self.cvd = self._cvd()  # Cumulative Volume Delta approximation

        # Momentum
        self.rsi_14 = self._rsi(14)
        self.rsi_7 = self._rsi(7)
        self.macd_line, self.macd_signal, self.macd_hist = self._macd()
        self.stoch_k, self.stoch_d = self._stochastic(14, 3)
        self.roc_10 = self._roc(10)
        self.roc_20 = self._roc(20)
        self.mfi = self._mfi(14)  # Money Flow Index

        # Volatility
        self.bb_upper, self.bb_middle, self.bb_lower = self._bollinger(20, 2.0)
        self.bb_width = self._bb_width()
        self.atr_14 = self._atr(14)
        self.atr_pct = [(a / c * 100 if (a is not None and c) else 0) for a, c in zip(self.atr_14, self.closes)]

        # Keltner Channels (for squeeze detection)
        self.kc_upper, self.kc_lower = self._keltner(20, 1.5)

        # Trend
        self.adx = self._adx(14)
        self.psar = self._parabolic_sar()

        # Support & Resistance
        self.pivot_points = self._pivot_points()

        # Candlestick patterns
        self.candle_patterns = self._detect_candle_patterns()

    # ── Core Calculations ──────────────────────────────────────────────────

    def _sma(self, series, period):
        result = [None] * self.n
        for i in range(period - 1, self.n):
            result[i] = sum(series[i - period + 1:i + 1]) / period
        return result

    def _ema(self, series, period):
        result = [None] * self.n
        k = 2 / (period + 1)
        start = period - 1
        if start >= self.n:
            return result
        result[start] = sum(series[:period]) / period
        for i in range(start + 1, self.n):
            result[i] = series[i] * k + result[i - 1] * (1 - k)
        return result

    def _rsi(self, period):
        result = [None] * self.n
        deltas = [0] + [self.closes[i] - self.closes[i - 1] for i in range(1, self.n)]
        gains = [max(0, d) for d in deltas]
        losses = [max(0, -d) for d in deltas]
        if period >= self.n:
            return result
        avg_gain = sum(gains[1:period + 1]) / period
        avg_loss = sum(losses[1:period + 1]) / period
        for i in range(period, self.n):
            if i > period:
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0:
                result[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                result[i] = 100 - (100 / (1 + rs))
        return result

    def _macd(self, fast=12, slow=26, signal=9):
        ema_fast = self._ema(self.closes, fast)
        ema_slow = self._ema(self.closes, slow)
        macd_line = [None] * self.n
        for i in range(self.n):
            if ema_fast[i] is not None and ema_slow[i] is not None:
                macd_line[i] = ema_fast[i] - ema_slow[i]
        macd_vals = [v if v is not None else 0 for v in macd_line]
        sig = self._ema(macd_vals, signal)
        histogram = [None] * self.n
        for i in range(self.n):
            if macd_line[i] is not None and sig[i] is not None:
                histogram[i] = macd_line[i] - sig[i]
        return macd_line, sig, histogram

    def _stochastic(self, period, smooth):
        k = [None] * self.n
        for i in range(period - 1, self.n):
            low_min = min(self.lows[i - period + 1:i + 1])
            high_max = max(self.highs[i - period + 1:i + 1])
            if high_max != low_min:
                k[i] = ((self.closes[i] - low_min) / (high_max - low_min)) * 100
            else:
                k[i] = 50.0
        d = self._sma([v if v is not None else 50 for v in k], smooth)
        return k, d

    def _roc(self, period):
        result = [None] * self.n
        for i in range(period, self.n):
            if self.closes[i - period] != 0:
                result[i] = ((self.closes[i] - self.closes[i - period]) / self.closes[i - period]) * 100
        return result

    def _bollinger(self, period, num_std):
        upper, middle, lower = [None] * self.n, [None] * self.n, [None] * self.n
        for i in range(period - 1, self.n):
            window = self.closes[i - period + 1:i + 1]
            mean = sum(window) / period
            std = math.sqrt(sum((x - mean) ** 2 for x in window) / period)
            middle[i] = mean
            upper[i] = mean + num_std * std
            lower[i] = mean - num_std * std
        return upper, middle, lower

    def _bb_width(self):
        result = [None] * self.n
        for i in range(self.n):
            if self.bb_upper[i] and self.bb_middle[i] and self.bb_middle[i] != 0:
                result[i] = (self.bb_upper[i] - self.bb_lower[i]) / self.bb_middle[i]
        return result

    def _atr(self, period):
        tr = [0.0] * self.n
        for i in range(1, self.n):
            tr[i] = max(
                self.highs[i] - self.lows[i],
                abs(self.highs[i] - self.closes[i - 1]),
                abs(self.lows[i] - self.closes[i - 1]),
            )
        tr[0] = self.highs[0] - self.lows[0]
        atr = [None] * self.n
        if period < self.n:
            atr[period - 1] = sum(tr[:period]) / period
            for i in range(period, self.n):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr

    def _adx(self, period):
        result = [None] * self.n
        if self.n < period * 2:
            return result
        plus_dm = [0.0] * self.n
        minus_dm = [0.0] * self.n
        tr = [0.0] * self.n
        for i in range(1, self.n):
            up = self.highs[i] - self.highs[i - 1]
            down = self.lows[i - 1] - self.lows[i]
            plus_dm[i] = up if (up > down and up > 0) else 0
            minus_dm[i] = down if (down > up and down > 0) else 0
            tr[i] = max(self.highs[i] - self.lows[i], abs(self.highs[i] - self.closes[i - 1]), abs(self.lows[i] - self.closes[i - 1]))
        atr_s = self._smooth(tr, period)
        plus_s = self._smooth(plus_dm, period)
        minus_s = self._smooth(minus_dm, period)
        dx = [None] * self.n
        for i in range(self.n):
            if atr_s[i] and atr_s[i] > 0 and plus_s[i] is not None:
                plus_di = (plus_s[i] / atr_s[i]) * 100
                minus_di = (minus_s[i] / atr_s[i]) * 100
                denom = plus_di + minus_di
                dx[i] = abs(plus_di - minus_di) / denom * 100 if denom > 0 else 0
        for i in range(period * 2 - 1, self.n):
            vals = [dx[j] for j in range(i - period + 1, i + 1) if dx[j] is not None]
            if vals:
                result[i] = sum(vals) / len(vals)
        return result

    def _smooth(self, series, period):
        result = [None] * self.n
        if period >= self.n:
            return result
        result[period] = sum(series[1:period + 1])
        for i in range(period + 1, self.n):
            if result[i - 1] is not None:
                result[i] = result[i - 1] - result[i - 1] / period + series[i]
        return result

    def _obv(self):
        result = [0.0] * self.n
        for i in range(1, self.n):
            if self.closes[i] > self.closes[i - 1]:
                result[i] = result[i - 1] + self.volumes[i]
            elif self.closes[i] < self.closes[i - 1]:
                result[i] = result[i - 1] - self.volumes[i]
            else:
                result[i] = result[i - 1]
        return result

    def _cvd(self):
        """Cumulative Volume Delta — approximation using candle body vs range."""
        result = [0.0] * self.n
        for i in range(self.n):
            rng = self.highs[i] - self.lows[i]
            if rng > 0:
                buying = ((self.closes[i] - self.lows[i]) / rng) * self.volumes[i]
                selling = ((self.highs[i] - self.closes[i]) / rng) * self.volumes[i]
                delta = buying - selling
            else:
                delta = 0
            result[i] = (result[i - 1] if i > 0 else 0) + delta
        return result

    def _mfi(self, period):
        """Money Flow Index — volume-weighted RSI."""
        result = [None] * self.n
        if self.n < period + 1:
            return result
        tp = [(self.highs[i] + self.lows[i] + self.closes[i]) / 3 for i in range(self.n)]
        mf = [tp[i] * self.volumes[i] for i in range(self.n)]

        for i in range(period, self.n):
            pos_flow = sum(mf[j] for j in range(i - period + 1, i + 1) if j > 0 and tp[j] > tp[j - 1])
            neg_flow = sum(mf[j] for j in range(i - period + 1, i + 1) if j > 0 and tp[j] < tp[j - 1])
            if neg_flow == 0:
                result[i] = 100.0
            else:
                result[i] = 100 - (100 / (1 + pos_flow / neg_flow))
        return result

    def _vwap(self):
        """Rolling VWAP approximation (20-day)."""
        result = [None] * self.n
        period = 20
        for i in range(period - 1, self.n):
            tp_vol_sum = sum(
                ((self.highs[j] + self.lows[j] + self.closes[j]) / 3) * self.volumes[j]
                for j in range(i - period + 1, i + 1)
            )
            vol_sum = sum(self.volumes[j] for j in range(i - period + 1, i + 1))
            result[i] = tp_vol_sum / vol_sum if vol_sum > 0 else None
        return result

    def _keltner(self, period, mult):
        """Keltner Channels — used with Bollinger for squeeze detection."""
        ema = self._ema(self.closes, period)
        atr = self._atr(period)
        upper = [None] * self.n
        lower = [None] * self.n
        for i in range(self.n):
            if ema[i] is not None and atr[i] is not None:
                upper[i] = ema[i] + mult * atr[i]
                lower[i] = ema[i] - mult * atr[i]
        return upper, lower

    def _parabolic_sar(self, af_start=0.02, af_step=0.02, af_max=0.20):
        result = [None] * self.n
        if self.n < 2:
            return result
        uptrend = True
        af = af_start
        ep = self.highs[0]
        sar = self.lows[0]
        result[0] = sar
        for i in range(1, self.n):
            prev_sar = sar
            sar = prev_sar + af * (ep - prev_sar)
            if uptrend:
                sar = min(sar, self.lows[i - 1])
                if i >= 2:
                    sar = min(sar, self.lows[i - 2])
                if self.lows[i] < sar:
                    uptrend = False
                    sar = ep
                    ep = self.lows[i]
                    af = af_start
                else:
                    if self.highs[i] > ep:
                        ep = self.highs[i]
                        af = min(af + af_step, af_max)
            else:
                sar = max(sar, self.highs[i - 1])
                if i >= 2:
                    sar = max(sar, self.highs[i - 2])
                if self.highs[i] > sar:
                    uptrend = True
                    sar = ep
                    ep = self.highs[i]
                    af = af_start
                else:
                    if self.lows[i] < ep:
                        ep = self.lows[i]
                        af = min(af + af_step, af_max)
            result[i] = sar
        return result

    def _pivot_points(self):
        result = [None] * self.n
        for i in range(1, self.n):
            pivot = (self.highs[i - 1] + self.lows[i - 1] + self.closes[i - 1]) / 3
            r1 = 2 * pivot - self.lows[i - 1]
            s1 = 2 * pivot - self.highs[i - 1]
            r2 = pivot + (self.highs[i - 1] - self.lows[i - 1])
            s2 = pivot - (self.highs[i - 1] - self.lows[i - 1])
            result[i] = {"P": pivot, "R1": r1, "R2": r2, "S1": s1, "S2": s2}
        return result

    def _detect_candle_patterns(self):
        patterns = [[] for _ in range(self.n)]
        for i in range(2, self.n):
            body = self.closes[i] - self.opens[i]
            abs_body = abs(body)
            candle_range = self.highs[i] - self.lows[i]
            upper_wick = self.highs[i] - max(self.opens[i], self.closes[i])
            lower_wick = min(self.opens[i], self.closes[i]) - self.lows[i]
            prev_body = self.closes[i - 1] - self.opens[i - 1]
            if candle_range == 0:
                continue
            body_ratio = abs_body / candle_range

            # Hammer
            if lower_wick > abs_body * 2 and upper_wick < abs_body * 0.5 and body_ratio < 0.4:
                if i >= 3 and self.closes[i] < self.closes[i - 3]:
                    patterns[i].append(("HAMMER", 0.7, "bullish"))
                else:
                    patterns[i].append(("HANGING_MAN", 0.6, "bearish"))

            # Shooting Star
            if upper_wick > abs_body * 2 and lower_wick < abs_body * 0.5 and body_ratio < 0.4:
                if i >= 3 and self.closes[i] > self.closes[i - 3]:
                    patterns[i].append(("SHOOTING_STAR", 0.65, "bearish"))
                else:
                    patterns[i].append(("INVERTED_HAMMER", 0.6, "bullish"))

            # Engulfing
            if prev_body < 0 and body > 0 and abs_body > abs(prev_body):
                if self.opens[i] <= self.closes[i - 1] and self.closes[i] >= self.opens[i - 1]:
                    patterns[i].append(("BULLISH_ENGULFING", 0.8, "bullish"))
            if prev_body > 0 and body < 0 and abs_body > abs(prev_body):
                if self.opens[i] >= self.closes[i - 1] and self.closes[i] <= self.opens[i - 1]:
                    patterns[i].append(("BEARISH_ENGULFING", 0.8, "bearish"))

            # Doji
            if body_ratio < 0.1:
                patterns[i].append(("DOJI", 0.5, "neutral"))

            # Morning / Evening Star
            if i >= 2:
                prev2_body = self.closes[i - 2] - self.opens[i - 2]
                mid_body = abs(self.closes[i - 1] - self.opens[i - 1])
                mid_range = self.highs[i - 1] - self.lows[i - 1]
                if prev2_body < 0 and mid_range > 0 and mid_body < mid_range * 0.3 and body > 0 and abs_body > abs(prev2_body) * 0.5:
                    patterns[i].append(("MORNING_STAR", 0.85, "bullish"))
                if prev2_body > 0 and mid_range > 0 and mid_body < mid_range * 0.3 and body < 0 and abs_body > abs(prev2_body) * 0.5:
                    patterns[i].append(("EVENING_STAR", 0.85, "bearish"))

            # Three soldiers / crows
            if i >= 2:
                bodies = [self.closes[i - j] - self.opens[i - j] for j in range(3)]
                if all(b > 0 for b in bodies) and bodies[0] > bodies[1] * 0.5:
                    patterns[i].append(("THREE_WHITE_SOLDIERS", 0.75, "bullish"))
                if all(b < 0 for b in bodies) and abs(bodies[0]) > abs(bodies[1]) * 0.5:
                    patterns[i].append(("THREE_BLACK_CROWS", 0.75, "bearish"))

        return patterns

    def _safe_divide(self, a, b):
        result = [None] * self.n
        for i in range(self.n):
            if b[i] and b[i] != 0:
                result[i] = a[i] / b[i]
        return result


# ════════════════════════════════════════════════════════════════════════════════
#  SMART MONEY CONCEPTS ENGINE
# ════════════════════════════════════════════════════════════════════════════════

class SmartMoneyAnalyzer:
    """
    Detects ICT / Smart Money Concepts:
    - Fair Value Gaps (FVGs)
    - Liquidity Sweeps
    - Order Blocks
    - Market Structure (BOS / CHoCH)
    - Displacement candles
    - Premium / Discount zones
    - SMT Divergences (internal + cross-asset)
    """

    def __init__(self, data: list[OHLCV], indicators: Indicators, config: TraderConfig,
                 smt_data: Optional[list[OHLCV]] = None):
        self.data = data
        self.ind = indicators
        self.config = config
        self.n = len(data)
        self.smt_data = smt_data

        # Detected structures
        self.fvgs: list[list[FairValueGap]] = [[] for _ in range(self.n)]
        self.liquidity_sweeps: list[list[LiquiditySweep]] = [[] for _ in range(self.n)]
        self.order_blocks: list[list[OrderBlock]] = [[] for _ in range(self.n)]
        self.swing_points: list[SwingPoint] = []
        self.structure_breaks: list[list[StructureBreak]] = [[] for _ in range(self.n)]
        self.smt_divergences: list[list[SMTDivergence]] = [[] for _ in range(self.n)]
        self.displacement: list[Optional[str]] = [None] * self.n  # "bullish", "bearish", or None
        self.premium_discount: list[Optional[str]] = [None] * self.n  # "premium", "discount", "equilibrium"
        self.squeeze: list[bool] = [False] * self.n

        # Active (unfilled) FVGs for tracking
        self._active_fvgs: list[FairValueGap] = []

        self._analyze_all()

    def _analyze_all(self):
        """Run all smart money detection passes."""
        self._detect_swing_points()
        self._detect_structure_breaks()
        self._detect_displacement()
        self._detect_fvgs()
        self._detect_order_blocks()
        self._detect_liquidity_sweeps()
        self._detect_premium_discount()
        self._detect_squeeze()
        self._detect_smt_divergences()
        self._update_fvg_fills()

    # ── Swing Points ───────────────────────────────────────────────────────

    def _detect_swing_points(self):
        lb = self.config.swing_lookback
        for i in range(lb, self.n - lb):
            # Swing High
            is_high = all(self.data[i].high >= self.data[j].high for j in range(i - lb, i + lb + 1) if j != i)
            if is_high:
                self.swing_points.append(SwingPoint(i, "high", self.data[i].high, self.data[i].date))

            # Swing Low
            is_low = all(self.data[i].low <= self.data[j].low for j in range(i - lb, i + lb + 1) if j != i)
            if is_low:
                self.swing_points.append(SwingPoint(i, "low", self.data[i].low, self.data[i].date))

    # ── Market Structure (BOS / CHoCH) ─────────────────────────────────────

    def _detect_structure_breaks(self):
        # Pre-sorted for pointer-based lookup
        swing_highs = sorted([sp for sp in self.swing_points if sp.type == "high"], key=lambda s: s.index)
        swing_lows = sorted([sp for sp in self.swing_points if sp.type == "low"], key=lambda s: s.index)

        hi_ptr = 0
        lo_ptr = 0

        for i in range(self.config.swing_lookback + 1, self.n):
            # Advance pointers
            while hi_ptr < len(swing_highs) and swing_highs[hi_ptr].index < i:
                hi_ptr += 1
            while lo_ptr < len(swing_lows) and swing_lows[lo_ptr].index < i:
                lo_ptr += 1

            recent_sh = swing_highs[max(0, hi_ptr - 4):hi_ptr]
            recent_sl = swing_lows[max(0, lo_ptr - 4):lo_ptr]

            if len(recent_sh) < 2 or len(recent_sl) < 2:
                continue

            # Break of Structure — bullish: price closes above most recent swing high
            last_sh = recent_sh[-1]
            if self.data[i].close > last_sh.price and self.data[i - 1].close <= last_sh.price:
                # Is this BOS (continuation) or CHoCH (reversal)?
                prev_two_sh = recent_sh[-2:]
                if len(prev_two_sh) >= 2 and prev_two_sh[-1].price < prev_two_sh[-2].price:
                    # Was making lower highs, now broke above = CHoCH
                    self.structure_breaks[i].append(StructureBreak(
                        i, StructureType.CHANGE_OF_CHARACTER, last_sh.price, "bullish", self.data[i].date
                    ))
                else:
                    self.structure_breaks[i].append(StructureBreak(
                        i, StructureType.BREAK_OF_STRUCTURE, last_sh.price, "bullish", self.data[i].date
                    ))

            # Break of Structure — bearish: price closes below most recent swing low
            last_sl = recent_sl[-1]
            if self.data[i].close < last_sl.price and self.data[i - 1].close >= last_sl.price:
                prev_two_sl = recent_sl[-2:]
                if len(prev_two_sl) >= 2 and prev_two_sl[-1].price > prev_two_sl[-2].price:
                    self.structure_breaks[i].append(StructureBreak(
                        i, StructureType.CHANGE_OF_CHARACTER, last_sl.price, "bearish", self.data[i].date
                    ))
                else:
                    self.structure_breaks[i].append(StructureBreak(
                        i, StructureType.BREAK_OF_STRUCTURE, last_sl.price, "bearish", self.data[i].date
                    ))

    # ── Displacement Detection ─────────────────────────────────────────────

    def _detect_displacement(self):
        """Large impulsive candles that show institutional commitment."""
        for i in range(1, self.n):
            atr = self.ind.atr_14[i]
            if atr is None:
                continue
            body = abs(self.data[i].close - self.data[i].open)
            if body >= atr * self.config.displacement_min_body_atr:
                if self.data[i].close > self.data[i].open:
                    self.displacement[i] = "bullish"
                else:
                    self.displacement[i] = "bearish"

    # ── Fair Value Gaps ────────────────────────────────────────────────────

    def _detect_fvgs(self):
        """
        FVG = 3-candle pattern where candle 3 doesn't overlap with candle 1.
        Bullish FVG: candle1.high < candle3.low (gap up)
        Bearish FVG: candle1.low > candle3.high (gap down)
        """
        for i in range(2, self.n):
            atr = self.ind.atr_14[i]
            if atr is None or atr == 0:
                continue

            c1 = self.data[i - 2]
            c3 = self.data[i]

            # Bullish FVG
            if c3.low > c1.high:
                gap_size = c3.low - c1.high
                if gap_size >= atr * self.config.fvg_min_gap_atr:
                    fvg = FairValueGap(
                        index=i, direction="bullish",
                        top=c3.low, bottom=c1.high,
                        midpoint=(c3.low + c1.high) / 2,
                        size=gap_size, size_atr=gap_size / atr,
                        created_date=self.data[i].date,
                    )
                    self.fvgs[i].append(fvg)
                    self._active_fvgs.append(fvg)

            # Bearish FVG
            if c1.low > c3.high:
                gap_size = c1.low - c3.high
                if gap_size >= atr * self.config.fvg_min_gap_atr:
                    fvg = FairValueGap(
                        index=i, direction="bearish",
                        top=c1.low, bottom=c3.high,
                        midpoint=(c1.low + c3.high) / 2,
                        size=gap_size, size_atr=gap_size / atr,
                        created_date=self.data[i].date,
                    )
                    self.fvgs[i].append(fvg)
                    self._active_fvgs.append(fvg)

    def _update_fvg_fills(self):
        """Track when price fills back into FVGs."""
        for fvg in self._active_fvgs:
            for i in range(fvg.index + 1, self.n):
                if fvg.direction == "bullish":
                    # Filled when price drops into the gap
                    if self.data[i].low <= fvg.midpoint:
                        fvg.filled = True
                        fvg.fill_index = i
                        break
                else:
                    # Filled when price rises into the gap
                    if self.data[i].high >= fvg.midpoint:
                        fvg.filled = True
                        fvg.fill_index = i
                        break

    # ── Order Blocks ───────────────────────────────────────────────────────

    def _detect_order_blocks(self):
        """
        Order Block = the last opposing candle before a displacement move.
        Bullish OB: last bearish candle before a bullish displacement
        Bearish OB: last bullish candle before a bearish displacement
        """
        for i in range(1, self.n):
            if self.displacement[i] is None:
                continue

            # Look back for the last opposing candle
            lookback = min(self.config.order_block_lookback, i)
            if self.displacement[i] == "bullish":
                # Find last bearish candle before this displacement
                for j in range(i - 1, i - lookback - 1, -1):
                    if self.data[j].close < self.data[j].open:
                        atr = self.ind.atr_14[i] or 1
                        disp_size = abs(self.data[i].close - self.data[i].open) / atr
                        ob = OrderBlock(
                            index=j, direction="bullish",
                            top=self.data[j].open, bottom=self.data[j].close,
                            midpoint=(self.data[j].open + self.data[j].close) / 2,
                            displacement_size=disp_size,
                            created_date=self.data[j].date,
                        )
                        self.order_blocks[i].append(ob)
                        break

            elif self.displacement[i] == "bearish":
                for j in range(i - 1, i - lookback - 1, -1):
                    if self.data[j].close > self.data[j].open:
                        atr = self.ind.atr_14[i] or 1
                        disp_size = abs(self.data[i].close - self.data[i].open) / atr
                        ob = OrderBlock(
                            index=j, direction="bearish",
                            top=self.data[j].close, bottom=self.data[j].open,
                            midpoint=(self.data[j].close + self.data[j].open) / 2,
                            displacement_size=disp_size,
                            created_date=self.data[j].date,
                        )
                        self.order_blocks[i].append(ob)
                        break

    # ── Liquidity Sweeps ───────────────────────────────────────────────────

    def _detect_liquidity_sweeps(self):
        """
        Liquidity Sweep = price pokes past a key swing high/low (grabbing stops)
        then reverses and closes back inside. Classic stop hunt.
        """
        # Pre-sort for efficient lookup
        sorted_highs = sorted([sp for sp in self.swing_points if sp.type == "high"], key=lambda s: s.index)
        sorted_lows = sorted([sp for sp in self.swing_points if sp.type == "low"], key=lambda s: s.index)

        for i in range(2, self.n):
            price = self.data[i]
            lb = self.config.liquidity_sweep_lookback
            threshold = self.config.liquidity_sweep_threshold

            # Recent swing highs within lookback (buy-side liquidity)
            recent_highs = [sp for sp in sorted_highs
                           if i - lb <= sp.index < i - 1]

            for sh in recent_highs:
                poke_pct = (price.high - sh.price) / sh.price if sh.price > 0 else 0
                if poke_pct > threshold and price.close < sh.price:
                    strength = min(1.0, poke_pct / 0.01 * 0.3 + 0.4)
                    if price.close < price.open:
                        strength += 0.2
                    self.liquidity_sweeps[i].append(LiquiditySweep(
                        index=i, direction="bearish",
                        swept_level=sh.price, sweep_extreme=price.high,
                        reversal_close=price.close,
                        strength=min(1.0, strength),
                        created_date=self.data[i].date,
                    ))

            # Recent swing lows within lookback (sell-side liquidity)
            recent_lows = [sp for sp in sorted_lows
                          if i - lb <= sp.index < i - 1]

            for sl in recent_lows:
                poke_pct = (sl.price - price.low) / sl.price if sl.price > 0 else 0
                if poke_pct > threshold and price.close > sl.price:
                    strength = min(1.0, poke_pct / 0.01 * 0.3 + 0.4)
                    if price.close > price.open:
                        strength += 0.2
                    self.liquidity_sweeps[i].append(LiquiditySweep(
                        index=i, direction="bullish",
                        swept_level=sl.price, sweep_extreme=price.low,
                        reversal_close=price.close,
                        strength=min(1.0, strength),
                        created_date=self.data[i].date,
                    ))

    # ── Premium / Discount Zones ───────────────────────────────────────────

    def _detect_premium_discount(self):
        """
        Based on the current dealing range (recent swing high to swing low):
        - Above 70% = Premium (look to sell)
        - Below 30% = Discount (look to buy)
        - Near 50% = Equilibrium
        """
        # Pre-sort swing points by index for efficient lookup
        sorted_highs = sorted([sp for sp in self.swing_points if sp.type == "high"], key=lambda s: s.index)
        sorted_lows = sorted([sp for sp in self.swing_points if sp.type == "low"], key=lambda s: s.index)

        hi_ptr = 0
        lo_ptr = 0

        for i in range(20, self.n):
            # Advance pointers to include swings up to bar i
            while hi_ptr < len(sorted_highs) and sorted_highs[hi_ptr].index <= i:
                hi_ptr += 1
            while lo_ptr < len(sorted_lows) and sorted_lows[lo_ptr].index <= i:
                lo_ptr += 1

            if hi_ptr == 0 or lo_ptr == 0:
                continue

            # Take last 5 of each
            recent_hi = sorted_highs[max(0, hi_ptr - 5):hi_ptr]
            recent_lo = sorted_lows[max(0, lo_ptr - 5):lo_ptr]

            range_high = max(sp.price for sp in recent_hi)
            range_low = min(sp.price for sp in recent_lo)

            if range_high == range_low:
                continue

            position = (self.data[i].close - range_low) / (range_high - range_low)

            if position > 0.7:
                self.premium_discount[i] = "premium"
            elif position < 0.3:
                self.premium_discount[i] = "discount"
            else:
                self.premium_discount[i] = "equilibrium"

    # ── Volatility Squeeze ─────────────────────────────────────────────────

    def _detect_squeeze(self):
        """BB inside Keltner = squeeze (energy building)."""
        for i in range(self.n):
            bb_u = self.ind.bb_upper[i]
            bb_l = self.ind.bb_lower[i]
            kc_u = self.ind.kc_upper[i]
            kc_l = self.ind.kc_lower[i]
            if all(v is not None for v in [bb_u, bb_l, kc_u, kc_l]):
                self.squeeze[i] = (bb_u < kc_u and bb_l > kc_l)

    # ── SMT Divergences ────────────────────────────────────────────────────

    def _detect_smt_divergences(self):
        """
        SMT = Smart Money Technique divergence.

        Internal SMT: Price makes new high/low but momentum indicators don't confirm.
        Cross-asset SMT: If smt_data provided, compare swing highs/lows between assets.
        """
        self._detect_internal_smt()
        if self.smt_data:
            self._detect_cross_asset_smt()

    def _detect_internal_smt(self):
        """Compare price swing structure with RSI, OBV, CVD for divergences."""
        swing_highs = [sp for sp in self.swing_points if sp.type == "high"]
        swing_lows = [sp for sp in self.swing_points if sp.type == "low"]

        # Bearish SMT: Price makes higher high but RSI/OBV makes lower high
        for k in range(1, len(swing_highs)):
            sh1 = swing_highs[k - 1]
            sh2 = swing_highs[k]
            if sh2.price > sh1.price:  # Price HH
                rsi1 = self.ind.rsi_14[sh1.index]
                rsi2 = self.ind.rsi_14[sh2.index]
                obv1 = self.ind.obv[sh1.index]
                obv2 = self.ind.obv[sh2.index]
                cvd1 = self.ind.cvd[sh1.index]
                cvd2 = self.ind.cvd[sh2.index]

                div_count = 0
                descs = []
                if rsi1 is not None and rsi2 is not None and rsi2 < rsi1:
                    div_count += 1
                    descs.append("RSI lower high")
                if obv2 < obv1:
                    div_count += 1
                    descs.append("OBV lower high")
                if cvd2 < cvd1:
                    div_count += 1
                    descs.append("CVD lower high")

                if div_count >= 2:
                    strength = min(1.0, 0.3 * div_count + 0.2)
                    idx = sh2.index
                    if idx < self.n:
                        self.smt_divergences[idx].append(SMTDivergence(
                            index=idx, type="bearish",
                            description=f"Price HH but {', '.join(descs)}",
                            strength=strength, date=self.data[idx].date,
                        ))

        # Bullish SMT: Price makes lower low but RSI/OBV makes higher low
        for k in range(1, len(swing_lows)):
            sl1 = swing_lows[k - 1]
            sl2 = swing_lows[k]
            if sl2.price < sl1.price:  # Price LL
                rsi1 = self.ind.rsi_14[sl1.index]
                rsi2 = self.ind.rsi_14[sl2.index]
                obv1 = self.ind.obv[sl1.index]
                obv2 = self.ind.obv[sl2.index]
                cvd1 = self.ind.cvd[sl1.index]
                cvd2 = self.ind.cvd[sl2.index]

                div_count = 0
                descs = []
                if rsi1 is not None and rsi2 is not None and rsi2 > rsi1:
                    div_count += 1
                    descs.append("RSI higher low")
                if obv2 > obv1:
                    div_count += 1
                    descs.append("OBV higher low")
                if cvd2 > cvd1:
                    div_count += 1
                    descs.append("CVD higher low")

                if div_count >= 2:
                    strength = min(1.0, 0.3 * div_count + 0.2)
                    idx = sl2.index
                    if idx < self.n:
                        self.smt_divergences[idx].append(SMTDivergence(
                            index=idx, type="bullish",
                            description=f"Price LL but {', '.join(descs)}",
                            strength=strength, date=self.data[idx].date,
                        ))

    def _detect_cross_asset_smt(self):
        """
        Compare swing structure between primary and correlated asset.
        E.g., QQQ makes new high but SPY doesn't = bearish SMT divergence.
        """
        if not self.smt_data or len(self.smt_data) < self.n:
            return

        smt_highs = self.smt_data
        lb = self.config.swing_lookback

        for i in range(lb + 20, min(self.n, len(smt_highs))):
            # Check if primary makes new high but SMT asset doesn't
            lookback_range = range(max(0, i - 20), i)

            primary_high = max(self.data[j].high for j in lookback_range)
            smt_high = max(smt_highs[j].high for j in lookback_range)

            if self.data[i].high > primary_high:
                # Primary made new high
                if smt_highs[i].high < smt_high:
                    self.smt_divergences[i].append(SMTDivergence(
                        index=i, type="bearish",
                        description="Cross-asset: primary new high, correlated asset failed",
                        strength=0.8, date=self.data[i].date,
                    ))

            primary_low = min(self.data[j].low for j in lookback_range)
            smt_low = min(smt_highs[j].low for j in lookback_range)

            if self.data[i].low < primary_low:
                if smt_highs[i].low > smt_low:
                    self.smt_divergences[i].append(SMTDivergence(
                        index=i, type="bullish",
                        description="Cross-asset: primary new low, correlated asset held",
                        strength=0.8, date=self.data[i].date,
                    ))

    # ── Public Getters ─────────────────────────────────────────────────────

    def get_active_fvgs_at(self, i: int) -> list[FairValueGap]:
        """Return FVGs that are still unfilled at bar i."""
        return [fvg for fvg in self._active_fvgs
                if fvg.index < i and (not fvg.filled or (fvg.fill_index is not None and fvg.fill_index >= i))]

    def get_recent_order_blocks(self, i: int, lookback: int = 30) -> list[OrderBlock]:
        """Return recent untested order blocks."""
        obs = []
        for j in range(max(0, i - lookback), i):
            obs.extend(self.order_blocks[j])
        return obs

    def get_nearby_liquidity(self, i: int, lookback: int = 10) -> list[LiquiditySweep]:
        """Return recent liquidity sweeps."""
        sweeps = []
        for j in range(max(0, i - lookback), i + 1):
            sweeps.extend(self.liquidity_sweeps[j])
        return sweeps


# ════════════════════════════════════════════════════════════════════════════════
#  TRADE SIGNAL
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeSignal:
    date: str
    signal: Signal
    context: SignalContext
    conviction: float
    price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float
    risk_reward: float
    regime: Regime
    reasons: list
    smart_money_reasons: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    fvg_count: int = 0
    active_sweeps: int = 0
    structure_events: int = 0
    smt_signals: int = 0
    premium_discount_zone: str = ""


# ════════════════════════════════════════════════════════════════════════════════
#  VETERAN TRADER BRAIN v2
# ════════════════════════════════════════════════════════════════════════════════

class VeteranTrader:
    def __init__(self, data: list[OHLCV], config: TraderConfig,
                 smt_data: Optional[list[OHLCV]] = None):
        self.data = data
        self.config = config
        self.n = len(data)

        print("  Computing technical indicators...")
        self.ind = Indicators(data)

        print("  Analyzing smart money concepts...")
        self.smc = SmartMoneyAnalyzer(data, self.ind, config, smt_data)

        self.signals: list[TradeSignal] = []
        self.consecutive_losses = 0
        self.cooldown_remaining = 0

    def analyze(self) -> list[TradeSignal]:
        start_idx = min(200, self.n - 1)
        if start_idx < 55:
            start_idx = 55

        print("  Running signal analysis...")
        for i in range(start_idx, self.n):
            signal = self._evaluate_day(i)
            self.signals.append(signal)

        return self.signals

    def _evaluate_day(self, i: int) -> TradeSignal:
        reasons = []
        sm_reasons = []
        warnings = []
        bullish_score = 0.0
        bearish_score = 0.0
        price = self.data[i].close

        # ── 1. Market Regime ───────────────────────────────────────────────
        regime = self._detect_regime(i)

        # ── 2. Classical Technicals (weighted: 3.0) ────────────────────────
        t_b, t_s, t_r = self._trend_signals(i)
        bullish_score += t_b
        bearish_score += t_s
        reasons.extend(t_r)

        # ── 3. Momentum (weighted: 3.0) ────────────────────────────────────
        m_b, m_s, m_r = self._momentum_signals(i)
        bullish_score += m_b
        bearish_score += m_s
        reasons.extend(m_r)

        # ── 4. Volume (weighted: 2.0) ──────────────────────────────────────
        v_b, v_s, v_r = self._volume_signals(i)
        bullish_score += v_b
        bearish_score += v_s
        reasons.extend(v_r)

        # ── 5. Volatility (weighted: 2.0) ──────────────────────────────────
        b_b, b_s, b_r = self._volatility_signals(i)
        bullish_score += b_b
        bearish_score += b_s
        reasons.extend(b_r)

        # ── 6. Candlestick Patterns (weighted: 1.5) ───────────────────────
        c_b, c_s, c_r = self._pattern_signals(i)
        bullish_score += c_b
        bearish_score += c_s
        reasons.extend(c_r)

        # ── 7. Support/Resistance (weighted: 1.0) ─────────────────────────
        sr_b, sr_s, sr_r = self._support_resistance_signals(i)
        bullish_score += sr_b
        bearish_score += sr_s
        reasons.extend(sr_r)

        # ══════════════════════════════════════════════════════════════════
        # ── 8. SMART MONEY CONCEPTS (weighted: 5.0 total) ────────────────
        # ══════════════════════════════════════════════════════════════════

        # FVG Signals
        fvg_b, fvg_s, fvg_r = self._fvg_signals(i)
        bullish_score += fvg_b
        bearish_score += fvg_s
        sm_reasons.extend(fvg_r)

        # Liquidity Sweep Signals
        liq_b, liq_s, liq_r = self._liquidity_sweep_signals(i)
        bullish_score += liq_b
        bearish_score += liq_s
        sm_reasons.extend(liq_r)

        # Market Structure Signals
        ms_b, ms_s, ms_r = self._structure_signals(i)
        bullish_score += ms_b
        bearish_score += ms_s
        sm_reasons.extend(ms_r)

        # Order Block Signals
        ob_b, ob_s, ob_r = self._order_block_signals(i)
        bullish_score += ob_b
        bearish_score += ob_s
        sm_reasons.extend(ob_r)

        # Displacement Signals
        disp_b, disp_s, disp_r = self._displacement_signals(i)
        bullish_score += disp_b
        bearish_score += disp_s
        sm_reasons.extend(disp_r)

        # SMT Divergence Signals
        smt_b, smt_s, smt_r = self._smt_signals(i)
        bullish_score += smt_b
        bearish_score += smt_s
        sm_reasons.extend(smt_r)

        # Premium / Discount
        pd_b, pd_s, pd_r = self._premium_discount_signals(i)
        bullish_score += pd_b
        bearish_score += pd_s
        sm_reasons.extend(pd_r)

        # ── Normalize ──────────────────────────────────────────────────────
        gross = bullish_score + bearish_score
        if gross > 0:
            dominance = abs(bullish_score - bearish_score) / gross
            strength = min(1.0, max(bullish_score, bearish_score) / 5.0)
            conviction = dominance * 0.6 + strength * 0.4
            conviction = min(1.0, conviction)
        else:
            conviction = 0.0

        net_score = bullish_score - bearish_score

        # ── Determine Context ──────────────────────────────────────────────
        context = self._determine_context(i, net_score, conviction, sm_reasons, reasons)

        # ── Discipline ─────────────────────────────────────────────────────
        signal, conviction, disc_warnings = self._apply_discipline(
            net_score, conviction, regime, i, context
        )
        warnings.extend(disc_warnings)

        # ── Risk Management ────────────────────────────────────────────────
        stop_loss, take_profit, position_pct, rr_ratio = self._compute_risk(signal, price, i)

        # ── High Confidence / Low Risk Check ───────────────────────────────
        signal, context = self._check_hclr(signal, context, conviction, rr_ratio, i, sm_reasons)

        # ── Count SMC elements ─────────────────────────────────────────────
        active_fvgs = self.smc.get_active_fvgs_at(i)
        active_sweeps = self.smc.get_nearby_liquidity(i, 5)
        pd_zone = self.smc.premium_discount[i] or ""

        return TradeSignal(
            date=self.data[i].date, signal=signal, context=context,
            conviction=round(conviction, 3), price=round(price, 2),
            stop_loss=round(stop_loss, 2), take_profit=round(take_profit, 2),
            position_size_pct=round(position_pct, 4),
            risk_reward=round(rr_ratio, 2), regime=regime,
            reasons=reasons, smart_money_reasons=sm_reasons,
            warnings=warnings, fvg_count=len(active_fvgs),
            active_sweeps=len(active_sweeps),
            structure_events=len(self.smc.structure_breaks[i]),
            smt_signals=len(self.smc.smt_divergences[i]),
            premium_discount_zone=pd_zone,
        )

    # ── Smart Money Signal Generators ──────────────────────────────────────

    def _fvg_signals(self, i: int) -> tuple:
        bull = bear = 0.0
        reasons = []

        # New FVGs created at this bar
        for fvg in self.smc.fvgs[i]:
            if fvg.direction == "bullish":
                bull += 0.6 * min(2.0, fvg.size_atr)
                reasons.append(f"[FVG] Bullish Fair Value Gap formed ({fvg.size_atr:.1f}x ATR)")
            else:
                bear += 0.6 * min(2.0, fvg.size_atr)
                reasons.append(f"[FVG] Bearish Fair Value Gap formed ({fvg.size_atr:.1f}x ATR)")

        # Price filling into active FVGs (entry opportunity)
        active_fvgs = self.smc.get_active_fvgs_at(i)
        price = self.data[i].close
        for fvg in active_fvgs:
            if fvg.direction == "bullish" and self.data[i].low <= fvg.top and price >= fvg.midpoint:
                bull += 1.0
                reasons.append(f"[FVG ENTRY] Price filling into bullish FVG (gap: ${fvg.bottom:.0f}-${fvg.top:.0f})")
            elif fvg.direction == "bearish" and self.data[i].high >= fvg.bottom and price <= fvg.midpoint:
                bear += 1.0
                reasons.append(f"[FVG ENTRY] Price filling into bearish FVG (gap: ${fvg.bottom:.0f}-${fvg.top:.0f})")

        return bull, bear, reasons

    def _liquidity_sweep_signals(self, i: int) -> tuple:
        bull = bear = 0.0
        reasons = []

        for sweep in self.smc.liquidity_sweeps[i]:
            if sweep.direction == "bullish":
                bull += 1.2 * sweep.strength
                reasons.append(
                    f"[SWEEP] Bullish liquidity sweep — swept ${sweep.swept_level:.2f} lows, "
                    f"reversed to ${sweep.reversal_close:.2f} (strength: {sweep.strength:.2f})"
                )
            else:
                bear += 1.2 * sweep.strength
                reasons.append(
                    f"[SWEEP] Bearish liquidity sweep — swept ${sweep.swept_level:.2f} highs, "
                    f"reversed to ${sweep.reversal_close:.2f} (strength: {sweep.strength:.2f})"
                )

        return bull, bear, reasons

    def _structure_signals(self, i: int) -> tuple:
        bull = bear = 0.0
        reasons = []

        for sb in self.smc.structure_breaks[i]:
            if sb.type == StructureType.CHANGE_OF_CHARACTER:
                weight = 1.5  # CHoCH is higher conviction
                label = "CHoCH"
            else:
                weight = 1.0
                label = "BOS"

            if sb.direction == "bullish":
                bull += weight
                reasons.append(f"[STRUCTURE] Bullish {label} — broke above ${sb.level:.2f}")
            else:
                bear += weight
                reasons.append(f"[STRUCTURE] Bearish {label} — broke below ${sb.level:.2f}")

        return bull, bear, reasons

    def _order_block_signals(self, i: int) -> tuple:
        bull = bear = 0.0
        reasons = []
        price = self.data[i].close

        recent_obs = self.smc.get_recent_order_blocks(i, 30)
        atr = self.ind.atr_14[i] or 1

        for ob in recent_obs:
            if ob.tested:
                continue

            # Check if price is testing the order block
            if ob.direction == "bullish" and abs(price - ob.midpoint) < atr * 0.5:
                if price >= ob.bottom:
                    bull += 0.8 * min(1.5, ob.displacement_size / 2)
                    reasons.append(
                        f"[ORDER BLOCK] Testing bullish OB at ${ob.bottom:.0f}-${ob.top:.0f} "
                        f"(displacement: {ob.displacement_size:.1f}x ATR)"
                    )
                    ob.tested = True

            elif ob.direction == "bearish" and abs(price - ob.midpoint) < atr * 0.5:
                if price <= ob.top:
                    bear += 0.8 * min(1.5, ob.displacement_size / 2)
                    reasons.append(
                        f"[ORDER BLOCK] Testing bearish OB at ${ob.bottom:.0f}-${ob.top:.0f} "
                        f"(displacement: {ob.displacement_size:.1f}x ATR)"
                    )
                    ob.tested = True

        return bull, bear, reasons

    def _displacement_signals(self, i: int) -> tuple:
        bull = bear = 0.0
        reasons = []

        disp = self.smc.displacement[i]
        if disp is None:
            return bull, bear, reasons

        atr = self.ind.atr_14[i] or 1
        body = abs(self.data[i].close - self.data[i].open)
        mult = body / atr

        if disp == "bullish":
            bull += 0.7 * min(2.0, mult / 1.5)
            reasons.append(f"[DISPLACEMENT] Bullish displacement candle ({mult:.1f}x ATR body)")
        else:
            bear += 0.7 * min(2.0, mult / 1.5)
            reasons.append(f"[DISPLACEMENT] Bearish displacement candle ({mult:.1f}x ATR body)")

        # Displacement + squeeze release = extra weight
        if self.smc.squeeze[max(0, i - 1)] and not self.smc.squeeze[i]:
            if disp == "bullish":
                bull += 0.5
                reasons.append("[DISPLACEMENT] Squeeze fire — bullish breakout from compression")
            else:
                bear += 0.5
                reasons.append("[DISPLACEMENT] Squeeze fire — bearish breakdown from compression")

        return bull, bear, reasons

    def _smt_signals(self, i: int) -> tuple:
        bull = bear = 0.0
        reasons = []

        for smt in self.smc.smt_divergences[i]:
            if smt.type == "bullish":
                bull += 1.3 * smt.strength
                reasons.append(f"[SMT] Bullish divergence: {smt.description} (str: {smt.strength:.2f})")
            else:
                bear += 1.3 * smt.strength
                reasons.append(f"[SMT] Bearish divergence: {smt.description} (str: {smt.strength:.2f})")

        return bull, bear, reasons

    def _premium_discount_signals(self, i: int) -> tuple:
        bull = bear = 0.0
        reasons = []

        zone = self.smc.premium_discount[i]
        if zone == "discount":
            bull += 0.4
            reasons.append("[P/D] Price in DISCOUNT zone — favorable for longs")
        elif zone == "premium":
            bear += 0.4
            reasons.append("[P/D] Price in PREMIUM zone — favorable for shorts")

        return bull, bear, reasons

    # ── Context Determination ──────────────────────────────────────────────

    def _determine_context(self, i, net_score, conviction, sm_reasons, reasons) -> SignalContext:
        """Determine the primary context/setup type for this signal."""
        # Count SMC hits
        has_sweep = any("[SWEEP]" in r for r in sm_reasons)
        has_fvg_entry = any("[FVG ENTRY]" in r for r in sm_reasons)
        has_structure = any("[STRUCTURE]" in r for r in sm_reasons)
        has_ob = any("[ORDER BLOCK]" in r for r in sm_reasons)
        has_displacement = any("[DISPLACEMENT]" in r for r in sm_reasons)
        has_smt = any("[SMT]" in r for r in sm_reasons)
        has_pd = any("[P/D]" in r for r in sm_reasons)

        smc_count = sum([has_sweep, has_fvg_entry, has_structure, has_ob, has_displacement, has_smt])

        # Multi-system confluence (classical + SMC agreeing)
        classical_reasons = len(reasons)
        if smc_count >= 3 and classical_reasons >= 3:
            return SignalContext.CONFLUENCE

        # Specific contexts (priority order)
        if has_sweep:
            return SignalContext.LIQUIDITY_SWEEP
        if has_smt:
            return SignalContext.SMT_DIVERGENCE
        if has_fvg_entry:
            return SignalContext.FVG_ENTRY
        if has_ob:
            return SignalContext.ORDER_BLOCK_ENTRY
        if has_structure:
            return SignalContext.BREAK_OF_STRUCTURE
        if has_displacement:
            return SignalContext.DISPLACEMENT_ENTRY

        # Classical contexts
        if any("golden cross" in r.lower() or "death cross" in r.lower() for r in reasons):
            return SignalContext.BREAKOUT
        if any("oversold" in r.lower() or "overbought" in r.lower() for r in reasons):
            return SignalContext.REVERSAL
        if any("MACD" in r and "crossover" in r for r in reasons):
            return SignalContext.MOMENTUM_SHIFT
        if any("above EMA" in r or "below EMA" in r for r in reasons):
            return SignalContext.TREND_CONTINUATION

        return SignalContext.STANDARD

    # ── High Confidence / Low Risk ─────────────────────────────────────────

    def _check_hclr(self, signal, context, conviction, rr_ratio, i, sm_reasons) -> tuple:
        """
        Upgrade signal to STRONG if it meets HCLR criteria:
        - High conviction
        - Low risk (tight stop via ATR)
        - Strong R:R
        - Multiple SMC confirmations
        """
        if signal in (Signal.HOLD, Signal.NO_TRADE):
            return signal, context

        atr = self.ind.atr_14[i]
        price = self.data[i].close
        if atr is None or price == 0:
            return signal, context

        atr_risk_pct = (atr * 2 / price) * 100  # Risk as % of price

        smc_count = sum(1 for r in sm_reasons if r.startswith("["))

        if (conviction >= self.config.hclr_min_conviction
                and atr_risk_pct <= self.config.hclr_max_atr_risk_pct
                and rr_ratio >= self.config.hclr_min_rr
                and smc_count >= 2):
            # Upgrade to STRONG + HCLR context
            if signal in (Signal.BUY, Signal.LEAN_BUY):
                return Signal.STRONG_BUY, SignalContext.HIGH_CONFIDENCE_LOW_RISK
            elif signal in (Signal.SELL, Signal.LEAN_SELL):
                return Signal.STRONG_SELL, SignalContext.HIGH_CONFIDENCE_LOW_RISK

        return signal, context

    # ── Classical Signal Generators (same as v1, kept compact) ─────────────

    def _detect_regime(self, i: int) -> Regime:
        sma50 = self.ind.sma_50[i]
        sma200 = self.ind.sma_200[i]
        adx = self.ind.adx[i]
        atr_pct = self.ind.atr_pct[i]
        price = self.data[i].close
        if sma50 is None or sma200 is None:
            return Regime.VOLATILE if (adx and adx < 20) else Regime.SIDEWAYS
        above_50 = price > sma50
        above_200 = price > sma200
        sma50_above_200 = sma50 > sma200
        strong = adx and adx > 30
        if above_50 and above_200 and sma50_above_200:
            return Regime.STRONG_UPTREND if strong else Regime.UPTREND
        elif not above_50 and not above_200 and not sma50_above_200:
            return Regime.STRONG_DOWNTREND if strong else Regime.DOWNTREND
        elif atr_pct and atr_pct > 2.5:
            return Regime.VOLATILE
        return Regime.SIDEWAYS

    def _trend_signals(self, i):
        bull = bear = 0.0
        reasons = []
        price = self.data[i].close
        ema9 = self.ind.ema_9[i]
        ema21 = self.ind.ema_21[i]
        sma50 = self.ind.sma_50[i]
        sma200 = self.ind.sma_200[i]
        psar = self.ind.psar[i]

        if ema9 and ema21:
            if ema9 > ema21 and price > ema9:
                bull += 0.8; reasons.append("Bullish: Price above EMA 9/21 stack")
            elif ema9 < ema21 and price < ema9:
                bear += 0.8; reasons.append("Bearish: Price below EMA 9/21 stack")
            prev9 = self.ind.ema_9[i - 1] if i > 0 else None
            prev21 = self.ind.ema_21[i - 1] if i > 0 else None
            if prev9 and prev21:
                if prev9 <= prev21 and ema9 > ema21:
                    bull += 1.0; reasons.append("Bullish: Fresh EMA 9/21 golden cross")
                elif prev9 >= prev21 and ema9 < ema21:
                    bear += 1.0; reasons.append("Bearish: Fresh EMA 9/21 death cross")

        if sma50 and sma200 and i > 0:
            p50 = self.ind.sma_50[i - 1]
            p200 = self.ind.sma_200[i - 1]
            if p50 and p200:
                if p50 <= p200 and sma50 > sma200:
                    bull += 1.2; reasons.append("STRONG Bullish: 50/200 Golden Cross")
                elif p50 >= p200 and sma50 < sma200:
                    bear += 1.2; reasons.append("STRONG Bearish: 50/200 Death Cross")

        if psar:
            if price > psar:
                bull += 0.4; reasons.append("Bullish: Price above Parabolic SAR")
            else:
                bear += 0.4; reasons.append("Bearish: Price below Parabolic SAR")

        return bull, bear, reasons

    def _momentum_signals(self, i):
        bull = bear = 0.0
        reasons = []
        rsi = self.ind.rsi_14[i]
        macd_h = self.ind.macd_hist[i]
        macd_line = self.ind.macd_line[i]
        macd_sig = self.ind.macd_signal[i]
        stoch_k = self.ind.stoch_k[i]
        mfi = self.ind.mfi[i]

        if rsi is not None:
            if rsi < self.config.rsi_oversold:
                bull += 1.0; reasons.append(f"Bullish: RSI oversold at {rsi:.1f}")
            elif rsi < 40:
                bull += 0.4; reasons.append(f"Lean Bullish: RSI approaching oversold ({rsi:.1f})")
            elif rsi > self.config.rsi_overbought:
                bear += 1.0; reasons.append(f"Bearish: RSI overbought at {rsi:.1f}")
            elif rsi > 60:
                bear += 0.3; reasons.append(f"Lean Bearish: RSI elevated ({rsi:.1f})")

            if i >= 14:
                rsi_prev = self.ind.rsi_14[i - 14]
                if rsi_prev is not None:
                    if self.data[i].close > self.data[i - 14].close and rsi < rsi_prev and rsi > 50:
                        bear += 0.7; reasons.append("Bearish: RSI negative divergence")
                    elif self.data[i].close < self.data[i - 14].close and rsi > rsi_prev and rsi < 50:
                        bull += 0.7; reasons.append("Bullish: RSI positive divergence")

        if macd_h is not None and i > 0:
            prev_h = self.ind.macd_hist[i - 1]
            if prev_h is not None:
                if prev_h <= 0 and macd_h > 0:
                    bull += 0.9; reasons.append("Bullish: MACD histogram crossed above zero")
                elif prev_h >= 0 and macd_h < 0:
                    bear += 0.9; reasons.append("Bearish: MACD histogram crossed below zero")
            if macd_line is not None and macd_sig is not None and i > 0:
                pml = self.ind.macd_line[i - 1]
                pms = self.ind.macd_signal[i - 1]
                if pml is not None and pms is not None:
                    if pml <= pms and macd_line > macd_sig:
                        bull += 0.8; reasons.append("Bullish: MACD bullish crossover")
                    elif pml >= pms and macd_line < macd_sig:
                        bear += 0.8; reasons.append("Bearish: MACD bearish crossover")

        if stoch_k is not None:
            if stoch_k < 20:
                bull += 0.5; reasons.append(f"Bullish: Stochastic oversold ({stoch_k:.0f})")
            elif stoch_k > 80:
                bear += 0.5; reasons.append(f"Bearish: Stochastic overbought ({stoch_k:.0f})")

        # Money Flow Index
        if mfi is not None:
            if mfi < 20:
                bull += 0.6; reasons.append(f"Bullish: MFI oversold ({mfi:.0f}) — institutional buying")
            elif mfi > 80:
                bear += 0.6; reasons.append(f"Bearish: MFI overbought ({mfi:.0f}) — institutional selling")

        return bull, bear, reasons

    def _volume_signals(self, i):
        bull = bear = 0.0
        reasons = []
        vol_ratio = self.ind.vol_ratio[i]
        obv = self.ind.obv[i]
        obv_sma = self.ind.obv_sma[i]
        price_up = self.data[i].close > self.data[i].open

        if vol_ratio is not None:
            if vol_ratio > self.config.volume_surge_threshold:
                if price_up:
                    bull += 0.8; reasons.append(f"Bullish: Volume surge ({vol_ratio:.1f}x avg) on up day")
                else:
                    bear += 0.8; reasons.append(f"Bearish: Volume surge ({vol_ratio:.1f}x avg) on down day")
            elif vol_ratio < 0.5:
                reasons.append(f"Low volume ({vol_ratio:.1f}x avg) — weak conviction")

        if obv_sma is not None:
            if obv > obv_sma:
                bull += 0.4; reasons.append("Bullish: OBV above its moving average")
            else:
                bear += 0.4; reasons.append("Bearish: OBV below its moving average")
            if i >= 20:
                if self.data[i].close > self.data[i - 20].close and obv < self.ind.obv[i - 20]:
                    bear += 0.6; reasons.append("Bearish: OBV divergence (distribution)")
                elif self.data[i].close < self.data[i - 20].close and obv > self.ind.obv[i - 20]:
                    bull += 0.6; reasons.append("Bullish: OBV divergence (accumulation)")

        return bull, bear, reasons

    def _volatility_signals(self, i):
        bull = bear = 0.0
        reasons = []
        price = self.data[i].close
        bb_upper = self.ind.bb_upper[i]
        bb_lower = self.ind.bb_lower[i]
        bb_width = self.ind.bb_width[i]

        if bb_upper and bb_lower:
            if price <= bb_lower:
                bull += 0.7; reasons.append("Bullish: Price at lower Bollinger Band")
            elif price >= bb_upper:
                bear += 0.6; reasons.append("Bearish: Price at upper Bollinger Band")

            if bb_width and bb_width < self.config.bb_squeeze_threshold:
                reasons.append(f"Bollinger Squeeze (width: {bb_width:.4f}) — breakout imminent")

            if i >= 3:
                cu = sum(1 for j in range(i - 2, i + 1) if self.ind.bb_upper[j] and self.data[j].close > self.ind.bb_upper[j] * 0.998)
                cl = sum(1 for j in range(i - 2, i + 1) if self.ind.bb_lower[j] and self.data[j].close < self.ind.bb_lower[j] * 1.002)
                if cu >= 2:
                    bull += 0.5; reasons.append("Bullish: Walking upper Bollinger Band")
                if cl >= 2:
                    bear += 0.5; reasons.append("Bearish: Walking lower Bollinger Band")

        return bull, bear, reasons

    def _pattern_signals(self, i):
        bull = bear = 0.0
        reasons = []
        for name, strength, direction in self.ind.candle_patterns[i]:
            label = name.replace("_", " ").title()
            if direction == "bullish":
                bull += strength; reasons.append(f"Bullish Pattern: {label}")
            elif direction == "bearish":
                bear += strength; reasons.append(f"Bearish Pattern: {label}")
            else:
                reasons.append(f"Neutral Pattern: {label}")
        return bull, bear, reasons

    def _support_resistance_signals(self, i):
        bull = bear = 0.0
        reasons = []
        pivots = self.ind.pivot_points[i]
        if pivots is None:
            return bull, bear, reasons
        price = self.data[i].close
        atr = self.ind.atr_14[i] or 1
        if abs(price - pivots["S1"]) < atr * 0.5:
            bull += 0.5; reasons.append(f"Bullish: Near S1 support ({pivots['S1']:.0f})")
        elif abs(price - pivots["S2"]) < atr * 0.5:
            bull += 0.7; reasons.append(f"Bullish: Near S2 strong support ({pivots['S2']:.0f})")
        if abs(price - pivots["R1"]) < atr * 0.5:
            bear += 0.5; reasons.append(f"Bearish: Near R1 resistance ({pivots['R1']:.0f})")
        elif abs(price - pivots["R2"]) < atr * 0.5:
            bear += 0.7; reasons.append(f"Bearish: Near R2 strong resistance ({pivots['R2']:.0f})")
        return bull, bear, reasons

    # ── Discipline ─────────────────────────────────────────────────────────

    def _apply_discipline(self, net_score, conviction, regime, i, context) -> tuple:
        warnings = []

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            warnings.append(f"DISCIPLINE: Cooldown ({self.cooldown_remaining}d left)")
            return Signal.NO_TRADE, conviction * 0.3, warnings

        # SMC contexts get a slight conviction pass (they're high-edge setups)
        smc_contexts = {
            SignalContext.LIQUIDITY_SWEEP, SignalContext.SMT_DIVERGENCE,
            SignalContext.FVG_ENTRY, SignalContext.CONFLUENCE,
            SignalContext.ORDER_BLOCK_ENTRY, SignalContext.HIGH_CONFIDENCE_LOW_RISK,
        }
        effective_threshold = self.config.min_signal_strength
        if context in smc_contexts:
            effective_threshold *= 0.85  # 15% lower bar for SMC setups

        if conviction < effective_threshold:
            warnings.append(f"PATIENCE: Conviction {conviction:.2f} < {effective_threshold:.2f}")
            return Signal.HOLD, conviction, warnings

        # Don't fight regime (reduced penalty for SMC setups)
        regime_penalty = 0.5 if context not in smc_contexts else 0.75
        if regime in (Regime.STRONG_DOWNTREND, Regime.DOWNTREND) and net_score > 0:
            conviction *= regime_penalty
            warnings.append("DISCIPLINE: Buying against downtrend")
        elif regime in (Regime.STRONG_UPTREND, Regime.UPTREND) and net_score < 0:
            conviction *= regime_penalty
            warnings.append("DISCIPLINE: Selling against uptrend")

        if regime == Regime.SIDEWAYS and conviction < 0.65 and context not in smc_contexts:
            warnings.append("PATIENCE: Sideways chop — waiting")
            return Signal.HOLD, conviction, warnings

        if regime == Regime.VOLATILE and conviction < 0.70 and context not in smc_contexts:
            warnings.append("PATIENCE: Volatile — need higher conviction")
            return Signal.HOLD, conviction, warnings

        if net_score > 0:
            if conviction > 0.85:
                signal = Signal.STRONG_BUY
            elif conviction > 0.65:
                signal = Signal.BUY
            else:
                signal = Signal.LEAN_BUY
        elif net_score < 0:
            if conviction > 0.85:
                signal = Signal.STRONG_SELL
            elif conviction > 0.65:
                signal = Signal.SELL
            else:
                signal = Signal.LEAN_SELL
        else:
            signal = Signal.HOLD

        return signal, conviction, warnings

    # ── Risk Management ────────────────────────────────────────────────────

    def _compute_risk(self, signal, price, i):
        atr = self.ind.atr_14[i] or price * 0.015
        is_buy = signal in (Signal.STRONG_BUY, Signal.BUY, Signal.LEAN_BUY)
        is_sell = signal in (Signal.STRONG_SELL, Signal.SELL, Signal.LEAN_SELL)

        if is_buy:
            stop_loss = price - (atr * 2.0)
            risk = price - stop_loss
            take_profit = price + (risk * self.config.take_profit_ratio)
        elif is_sell:
            stop_loss = price + (atr * 2.0)
            risk = stop_loss - price
            take_profit = price - (risk * self.config.take_profit_ratio)
        else:
            return price * 0.97, price * 1.05, 0.0, 0.0

        rr_ratio = self.config.take_profit_ratio
        risk_per_share = abs(price - stop_loss)
        if risk_per_share > 0:
            capital = self.config.starting_capital
            max_risk_dollars = capital * self.config.max_risk_per_trade
            shares = max_risk_dollars / risk_per_share
            position_value = shares * price
            position_pct = min(position_value / capital, self.config.max_position_pct)
        else:
            position_pct = 0.0

        return stop_loss, take_profit, position_pct, rr_ratio


# ════════════════════════════════════════════════════════════════════════════════
#  REPORT GENERATOR v2
# ════════════════════════════════════════════════════════════════════════════════

def generate_report(trader: VeteranTrader, output_path: str = "trade_signals.csv"):
    signals = trader.signals
    if not signals:
        print("  No signals generated.")
        return

    # CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Date", "Price", "Signal", "Context", "Conviction", "Regime",
            "Stop_Loss", "Take_Profit", "Position_%", "R:R",
            "Zone", "FVGs", "Sweeps", "Structure", "SMT",
            "Reasons", "Smart_Money_Reasons", "Warnings"
        ])
        for s in signals:
            writer.writerow([
                s.date, s.price, s.signal.value, s.context.value,
                s.conviction, s.regime.value,
                s.stop_loss, s.take_profit, f"{s.position_size_pct:.4f}",
                s.risk_reward, s.premium_discount_zone,
                s.fvg_count, s.active_sweeps, s.structure_events, s.smt_signals,
                " | ".join(s.reasons),
                " | ".join(s.smart_money_reasons),
                " | ".join(s.warnings) if s.warnings else "",
            ])

    # JSON
    json_path = output_path.replace(".csv", ".json")
    json_data = []
    for s in signals:
        json_data.append({
            "date": s.date, "price": s.price,
            "signal": s.signal.value, "context": s.context.value,
            "conviction": s.conviction, "regime": s.regime.value,
            "stop_loss": s.stop_loss, "take_profit": s.take_profit,
            "position_size_pct": s.position_size_pct,
            "risk_reward": s.risk_reward,
            "premium_discount_zone": s.premium_discount_zone,
            "fvg_count": s.fvg_count, "active_sweeps": s.active_sweeps,
            "structure_events": s.structure_events, "smt_signals": s.smt_signals,
            "reasons": s.reasons, "smart_money_reasons": s.smart_money_reasons,
            "warnings": s.warnings,
        })
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    # Console
    actionable = [s for s in signals if s.signal not in (Signal.HOLD, Signal.NO_TRADE)]
    buys = [s for s in signals if s.signal in (Signal.STRONG_BUY, Signal.BUY, Signal.LEAN_BUY)]
    sells = [s for s in signals if s.signal in (Signal.STRONG_SELL, Signal.SELL, Signal.LEAN_SELL)]

    # Context distribution
    ctx_counts = {}
    for s in actionable:
        c = s.context.value
        ctx_counts[c] = ctx_counts.get(c, 0) + 1

    # SMC stats
    total_fvg_signals = sum(1 for s in signals if any("[FVG" in r for r in s.smart_money_reasons))
    total_sweep_signals = sum(1 for s in signals if any("[SWEEP]" in r for r in s.smart_money_reasons))
    total_structure_signals = sum(1 for s in signals if any("[STRUCTURE]" in r for r in s.smart_money_reasons))
    total_smt_signals = sum(1 for s in signals if any("[SMT]" in r for r in s.smart_money_reasons))
    total_ob_signals = sum(1 for s in signals if any("[ORDER BLOCK]" in r for r in s.smart_money_reasons))
    total_disp_signals = sum(1 for s in signals if any("[DISPLACEMENT]" in r for r in s.smart_money_reasons))
    hclr_signals = [s for s in signals if s.context == SignalContext.HIGH_CONFIDENCE_LOW_RISK]

    regime_counts = {}
    for s in signals:
        r = s.regime.value
        regime_counts[r] = regime_counts.get(r, 0) + 1

    avg_conviction = sum(s.conviction for s in actionable) / len(actionable) if actionable else 0

    print()
    print("=" * 70)
    print("           VETERAN TRADER v2.0 — ANALYSIS REPORT")
    print("           Smart Money Concepts + Classical Technicals")
    print("=" * 70)
    print(f"  Period:        {signals[0].date}  ->  {signals[-1].date}")
    print(f"  Days Analyzed: {len(signals)}")
    print(f"  Risk Profile:  {trader.config.risk_profile.value.upper()}")
    print(f"  Capital:       ${trader.config.starting_capital:>12,.2f}")
    print()
    print("-" * 70)
    print("  SIGNAL DISTRIBUTION")
    print("-" * 70)
    for sig_type in Signal:
        count = sum(1 for s in signals if s.signal == sig_type)
        if count > 0:
            bar = "#" * min(40, count)
            print(f"    {sig_type.value:<16} {count:>4}  {bar}")
    print(f"    {'':─<40}")
    print(f"    Actionable:      {len(actionable)} / {len(signals)} ({len(actionable)/len(signals)*100:.1f}% of days)")
    print(f"    Avg Conviction:  {avg_conviction:.3f}")
    print()
    print("-" * 70)
    print("  SMART MONEY CONCEPTS DETECTED")
    print("-" * 70)
    print(f"    Fair Value Gaps:      {total_fvg_signals:>4} days with FVG signals")
    print(f"    Liquidity Sweeps:     {total_sweep_signals:>4} days with sweep signals")
    print(f"    Structure Breaks:     {total_structure_signals:>4} days with BOS/CHoCH")
    print(f"    SMT Divergences:      {total_smt_signals:>4} days with SMT signals")
    print(f"    Order Block Tests:    {total_ob_signals:>4} days with OB entries")
    print(f"    Displacement Candles: {total_disp_signals:>4} days with displacement")
    print()

    if hclr_signals:
        print("-" * 70)
        print(f"  HIGH CONFIDENCE / LOW RISK SIGNALS: {len(hclr_signals)}")
        print("-" * 70)
        for s in hclr_signals[-10:]:
            emoji = ">>" if "BUY" in s.signal.value else "<<"
            print(f"    {s.date}  {emoji} {s.signal.value:<14} conv:{s.conviction:.2f}  ${s.price:>10,.2f}  R:R {s.risk_reward:.1f}x")
            if s.smart_money_reasons:
                for r in s.smart_money_reasons[:3]:
                    print(f"      {r}")
        print()

    print("-" * 70)
    print("  SIGNAL CONTEXT BREAKDOWN")
    print("-" * 70)
    for ctx, count in sorted(ctx_counts.items(), key=lambda x: -x[1]):
        bar = "#" * min(30, count)
        print(f"    {ctx:<32} {count:>3}  {bar}")
    print()

    print("-" * 70)
    print("  MARKET REGIMES")
    print("-" * 70)
    for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
        pct = count / len(signals) * 100
        bar = "#" * int(pct / 2.5)
        print(f"    {regime:<24} {count:>4} ({pct:>5.1f}%) {bar}")
    print()

    # Latest signal (detailed)
    latest = signals[-1]
    print("-" * 70)
    print("  LATEST SIGNAL")
    print("-" * 70)
    print(f"    Date:        {latest.date}")
    print(f"    Price:       ${latest.price:,.2f}")
    print(f"    Signal:      {latest.signal.value}")
    print(f"    Context:     {latest.context.value}")
    print(f"    Conviction:  {latest.conviction:.3f}")
    print(f"    Regime:      {latest.regime.value}")
    print(f"    Zone:        {latest.premium_discount_zone or 'N/A'}")
    print(f"    Stop Loss:   ${latest.stop_loss:,.2f}")
    print(f"    Take Profit: ${latest.take_profit:,.2f}")
    print(f"    Position:    {latest.position_size_pct*100:.2f}% of capital")
    print(f"    Risk/Reward: {latest.risk_reward:.1f}x")
    print(f"    Active FVGs: {latest.fvg_count}")
    if latest.reasons:
        print("    -- Technical Reasons --")
        for r in latest.reasons[:8]:
            print(f"      {r}")
    if latest.smart_money_reasons:
        print("    -- Smart Money Reasons --")
        for r in latest.smart_money_reasons[:8]:
            print(f"      {r}")
    if latest.warnings:
        print("    -- Warnings --")
        for w in latest.warnings:
            print(f"      {w}")
    print()

    # Recent actionable
    print("-" * 70)
    print("  RECENT ACTIONABLE SIGNALS (last 15)")
    print("-" * 70)
    recent = [s for s in signals[-60:] if s.signal not in (Signal.HOLD, Signal.NO_TRADE)][-15:]
    for s in recent:
        emoji = ">>" if "BUY" in s.signal.value else "<<"
        ctx_short = s.context.value[:20]
        zone_tag = f"[{s.premium_discount_zone}]" if s.premium_discount_zone else ""
        print(f"    {s.date}  {emoji} {s.signal.value:<14} {s.conviction:.2f}  ${s.price:>9,.2f}  {ctx_short:<20} {zone_tag}")
    print()

    print("=" * 70)
    print(f"  Signals CSV:  {output_path}")
    print(f"  Detail JSON:  {json_path}")
    print()
    print("  DISCLAIMER: Educational tool only. NOT financial advice.")
    print("  Past patterns do not guarantee future results.")
    print("=" * 70)
    print()


# ════════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Veteran Trader Agent v2.0 — Smart Money + Classical Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python veteran_trader_v2.py nasdaq_data.csv
  python veteran_trader_v2.py QQQ_data.csv --risk-profile aggressive
  python veteran_trader_v2.py QQQ_data.csv --smt-compare SPY_data.csv
  python veteran_trader_v2.py data.csv --capital 50000 --max-risk 0.01
        """,
    )
    parser.add_argument("datafile", help="Path to OHLCV CSV file")
    parser.add_argument("--risk-profile", choices=["conservative", "moderate", "aggressive"],
                        default="moderate")
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--max-risk", type=float, default=None)
    parser.add_argument("--smt-compare", default=None,
                        help="Path to a correlated asset CSV for cross-asset SMT divergence")
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    print()
    print("  ============================================")
    print("       VETERAN TRADER AGENT v2.0")
    print("       Smart Money Concepts Engine")
    print("  ============================================")
    print()

    config = TraderConfig(risk_profile=RiskProfile(args.risk_profile), starting_capital=args.capital)
    config.adjust_for_profile()
    if args.max_risk is not None:
        config.max_risk_per_trade = args.max_risk

    print(f"  Risk Profile:   {config.risk_profile.value.upper()}")
    print(f"  Capital:        ${config.starting_capital:,.2f}")
    print(f"  Max Risk/Trade: {config.max_risk_per_trade*100:.1f}%")
    print(f"  Min Conviction: {config.min_signal_strength:.2f}")
    print()

    data = load_data(args.datafile)

    smt_data = None
    if args.smt_compare:
        print(f"  Loading SMT comparison asset: {args.smt_compare}")
        smt_data = load_data(args.smt_compare)

    trader = VeteranTrader(data, config, smt_data)
    trader.analyze()

    output = args.output
    if output is None:
        base = os.path.splitext(os.path.basename(args.datafile))[0]
        output = f"{base}_signals_v2.csv"

    generate_report(trader, output)


if __name__ == "__main__":
    main()