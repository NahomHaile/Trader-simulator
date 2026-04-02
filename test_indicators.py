#!/usr/bin/env python3
"""
Unit tests for Veteran Trader v2 — indicators and SMC detection.

Run with:  python -m pytest test_indicators.py -v
       or:  python test_indicators.py
"""

import math
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from veteran_trader_v2 import (
    OHLCV, Indicators, TraderConfig, RiskProfile,
    SmartMoneyAnalyzer, VeteranTrader, Signal, load_data
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_bars(closes, opens=None, highs=None, lows=None, volumes=None):
    """Build a list of OHLCV bars from a list of close prices."""
    n = len(closes)
    opens   = opens   or closes
    highs   = highs   or [c * 1.005 for c in closes]
    lows    = lows    or [c * 0.995 for c in closes]
    volumes = volumes or [1_000_000] * n
    return [
        OHLCV(
            date=f"2024-01-{i+1:02d}",
            open=opens[i], high=highs[i], low=lows[i],
            close=closes[i], volume=volumes[i],
        )
        for i in range(n)
    ]


def rising(start=100.0, step=1.0, n=60):
    return [start + i * step for i in range(n)]


def falling(start=160.0, step=1.0, n=60):
    return [start - i * step for i in range(n)]


# ── EMA Tests ─────────────────────────────────────────────────────────────────

class TestEMA(unittest.TestCase):
    # Indicators exposes: ema_9, ema_21, ema_55 and sma_10, sma_20, sma_50, sma_200

    def test_ema_starts_at_none_until_period(self):
        bars = make_bars(rising(n=60))
        ind = Indicators(bars)
        # ema_21 should be None for indices 0-19, a value from index 20 onward
        for i in range(20):
            self.assertIsNone(ind.ema_21[i], f"ema_21[{i}] should be None")
        self.assertIsNotNone(ind.ema_21[20])

    def test_ema_rises_on_rising_prices(self):
        bars = make_bars(rising(n=60))
        ind = Indicators(bars)
        values = [v for v in ind.ema_21 if v is not None]
        for i in range(1, len(values)):
            self.assertGreater(values[i], values[i - 1])

    def test_ema_55_lags_ema_21_on_rising_prices(self):
        bars = make_bars(rising(n=120))
        ind = Indicators(bars)
        # Slower EMA should be below faster EMA on a rising series
        last = -1
        for i in range(len(bars)):
            if ind.ema_21[i] is not None and ind.ema_55[i] is not None:
                last = i
        self.assertGreater(last, 0)
        self.assertGreater(ind.ema_21[last], ind.ema_55[last])

    def test_sma_200_requires_200_bars(self):
        bars = make_bars(rising(n=210))
        ind = Indicators(bars)
        self.assertIsNone(ind.sma_200[198])
        self.assertIsNotNone(ind.sma_200[199])


# ── RSI Tests ─────────────────────────────────────────────────────────────────

class TestRSI(unittest.TestCase):

    def test_rsi_bounds(self):
        closes = rising(n=80) + falling(n=40)
        bars = make_bars(closes)
        ind = Indicators(bars)
        for v in ind.rsi_14:
            if v is not None:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 100.0)

    def test_rsi_high_on_strong_uptrend(self):
        # Consistent gains should push RSI above 70
        bars = make_bars(rising(start=100, step=2, n=60))
        ind = Indicators(bars)
        last_rsi = next(v for v in reversed(ind.rsi_14) if v is not None)
        self.assertGreater(last_rsi, 70.0)

    def test_rsi_low_on_strong_downtrend(self):
        # Consistent losses should push RSI below 30
        bars = make_bars(falling(start=200, step=2, n=60))
        ind = Indicators(bars)
        last_rsi = next(v for v in reversed(ind.rsi_14) if v is not None)
        self.assertLess(last_rsi, 30.0)

    def test_rsi_starts_none_before_period(self):
        bars = make_bars(rising(n=30))
        ind = Indicators(bars)
        for i in range(14):
            self.assertIsNone(ind.rsi_14[i])


# ── ATR Tests ─────────────────────────────────────────────────────────────────

class TestATR(unittest.TestCase):

    def test_atr_is_positive(self):
        bars = make_bars(rising(n=50))
        ind = Indicators(bars)
        for v in ind.atr_14:
            if v is not None:
                self.assertGreater(v, 0.0)

    def test_atr_larger_on_wider_candles(self):
        n = 60
        closes = rising(n=n)
        # Narrow candles: 0.1% range
        narrow_highs = [c * 1.001 for c in closes]
        narrow_lows  = [c * 0.999 for c in closes]
        # Wide candles: 2% range
        wide_highs   = [c * 1.02  for c in closes]
        wide_lows    = [c * 0.98  for c in closes]

        narrow_bars = make_bars(closes, highs=narrow_highs, lows=narrow_lows)
        wide_bars   = make_bars(closes, highs=wide_highs,   lows=wide_lows)

        narrow_ind = Indicators(narrow_bars)
        wide_ind   = Indicators(wide_bars)

        last_narrow = next(v for v in reversed(narrow_ind.atr_14) if v is not None)
        last_wide   = next(v for v in reversed(wide_ind.atr_14)   if v is not None)
        self.assertGreater(last_wide, last_narrow)

    def test_atr_starts_at_bar_13(self):
        bars = make_bars(rising(n=30))
        ind = Indicators(bars)
        self.assertIsNone(ind.atr_14[12])
        self.assertIsNotNone(ind.atr_14[13])


# ── MACD Tests ────────────────────────────────────────────────────────────────

class TestMACD(unittest.TestCase):

    def test_macd_positive_on_strong_uptrend(self):
        # Fast EMA > Slow EMA when trending up
        bars = make_bars(rising(start=100, step=1, n=100))
        ind = Indicators(bars)
        last_macd = next(v for v in reversed(ind.macd_line) if v is not None)
        self.assertGreater(last_macd, 0.0)

    def test_macd_negative_on_strong_downtrend(self):
        bars = make_bars(falling(start=200, step=1, n=100))
        ind = Indicators(bars)
        last_macd = next(v for v in reversed(ind.macd_line) if v is not None)
        self.assertLess(last_macd, 0.0)

    def test_macd_histogram_is_macd_minus_signal(self):
        bars = make_bars(rising(n=100))
        ind = Indicators(bars)
        for i in range(len(bars)):
            m = ind.macd_line[i]
            s = ind.macd_signal[i]
            h = ind.macd_hist[i]
            if m is not None and s is not None and h is not None:
                self.assertAlmostEqual(h, m - s, places=8)


# ── Bollinger Band Tests ───────────────────────────────────────────────────────

class TestBollingerBands(unittest.TestCase):

    def test_price_inside_bands(self):
        closes = rising(n=60)
        bars = make_bars(closes)
        ind = Indicators(bars)
        for i in range(len(bars)):
            if ind.bb_upper[i] and ind.bb_lower[i]:
                # The middle band is the 20-period SMA of closes
                self.assertGreater(ind.bb_upper[i], ind.bb_lower[i])

    def test_bands_wider_on_volatile_prices(self):
        import random
        random.seed(42)
        n = 60
        smooth = rising(n=n)
        volatile = [100 + random.uniform(-10, 10) for _ in range(n)]

        smooth_ind   = Indicators(make_bars(smooth))
        volatile_ind = Indicators(make_bars(volatile))

        last_smooth_width   = next(v for v in reversed(smooth_ind.bb_width)   if v is not None)
        last_volatile_width = next(v for v in reversed(volatile_ind.bb_width) if v is not None)
        self.assertGreater(last_volatile_width, last_smooth_width)


# ── SMC: FVG Detection Tests ──────────────────────────────────────────────────

class TestFVGDetection(unittest.TestCase):

    def _make_bullish_fvg_bars(self):
        """
        Create bars with a textbook bullish FVG:
          bar[i-1]: low=100, high=102, close=101
          bar[i]  : open=103, close=108, high=110, low=103  (big up candle)
          bar[i+1]: open=109, high=112, low=106, close=111
        The gap between bar[i-1].high (102) and bar[i+1].low (106) is the FVG.
        """
        bars = make_bars(rising(start=95, step=1, n=60))
        # Override three specific bars to create the FVG at index 30
        bars[29] = OHLCV("2024-01-30", open=100, high=102, low=100, close=101, volume=1_000_000)
        bars[30] = OHLCV("2024-01-31", open=103, high=110, low=103, close=108, volume=2_000_000)
        bars[31] = OHLCV("2024-02-01", open=109, high=112, low=106, close=111, volume=1_000_000)
        return bars

    def test_bullish_fvg_is_detected(self):
        bars = self._make_bullish_fvg_bars()
        config = TraderConfig()
        ind = Indicators(bars)
        smc = SmartMoneyAnalyzer(bars, ind, config)
        # There should be a bullish FVG around bar 30
        all_fvgs = [fvg for fvg_list in smc.fvgs for fvg in fvg_list]
        bullish = [f for f in all_fvgs if f.direction == "bullish"]
        self.assertTrue(len(bullish) > 0, "Expected at least one bullish FVG to be detected")

    def test_fvg_top_above_bottom(self):
        bars = self._make_bullish_fvg_bars()
        config = TraderConfig()
        ind = Indicators(bars)
        smc = SmartMoneyAnalyzer(bars, ind, config)
        for fvg_list in smc.fvgs:
            for fvg in fvg_list:
                self.assertGreater(fvg.top, fvg.bottom)
                self.assertAlmostEqual(fvg.midpoint, (fvg.top + fvg.bottom) / 2)


# ── SMC: SMT Date Alignment Tests ─────────────────────────────────────────────

class TestSMTAlignment(unittest.TestCase):

    def test_smt_aligned_matches_by_date(self):
        """smt_aligned[i] should be the bar whose date matches primary data[i]."""
        primary = make_bars(rising(start=100, n=60))
        # SMT data with an extra leading bar (simulating different file length)
        extra = OHLCV("2023-12-31", 99, 100, 98, 99, 500_000)
        smt = [extra] + make_bars(rising(start=90, n=60))
        # Override smt bar dates to match primary
        for i, bar in enumerate(primary):
            smt[i + 1] = OHLCV(bar.date, smt[i + 1].open, smt[i + 1].high,
                                smt[i + 1].low, smt[i + 1].close, smt[i + 1].volume)

        config = TraderConfig()
        ind = Indicators(primary)
        smc = SmartMoneyAnalyzer(primary, ind, config, smt_data=smt)

        for i, bar in enumerate(primary):
            aligned = smc.smt_aligned[i]
            if aligned is not None:
                self.assertEqual(aligned.date, bar.date,
                    f"smt_aligned[{i}].date={aligned.date} != primary date {bar.date}")

    def test_smt_missing_dates_produce_none(self):
        """If SMT data is missing a date, smt_aligned[i] should be None for that bar."""
        primary = make_bars(rising(start=100, n=10))
        # SMT data that is missing the date of primary[3]
        smt_bars = [
            OHLCV(primary[i].date, 90, 92, 88, 91, 500_000)
            for i in range(10) if i != 3
        ]
        config = TraderConfig()
        ind = Indicators(primary)
        smc = SmartMoneyAnalyzer(primary, ind, config, smt_data=smt_bars)
        self.assertIsNone(smc.smt_aligned[3])


# ── Integration: load_data skipped-row warning ────────────────────────────────

class TestLoadDataWarning(unittest.TestCase):

    def test_bad_row_warns_and_continues(self):
        """A corrupt row should print a warning but not crash load_data."""
        import tempfile
        from contextlib import redirect_stdout
        import io as _io

        # Build 52 valid rows + 1 bad one (load_data requires >=50 rows)
        lines = ["Date,Open,High,Low,Close,Volume\n"]
        for day in range(1, 53):
            lines.append(f"2024-01-{day:02d},100,102,99,101,1000000\n")
        # Insert one bad row in the middle
        lines.insert(26, "2024-01-BAD,BAD,DATA,HERE,CRASH,BOOM\n")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                         delete=False, encoding="utf-8") as f:
            f.writelines(lines)
            tmp_path = f.name

        buf = _io.StringIO()
        try:
            with redirect_stdout(buf):
                rows = load_data(tmp_path)
        finally:
            os.unlink(tmp_path)

        output = buf.getvalue()
        self.assertIn("WARNING", output, "Expected a WARNING message for the bad row")
        self.assertEqual(len(rows), 52, "Should have loaded 52 valid rows, skipping the bad one")


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
