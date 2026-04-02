"""
ICT Signal Adapter
===================
Bridges the existing ICT/Smart Money detection logic from veteran_trader_v2.py
to the RL environment's observation space.

On construction it receives the full OHLCV dataset, builds Indicators and
SmartMoneyAnalyzer upfront (the same way VeteranTrader does), and pre-computes
a running-market-structure vector.

get_signals(price_history, current_idx=None)
  - When current_idx is supplied (the data feed cursor), it reads from the
    pre-computed SmartMoneyAnalyzer arrays → rich, consistent signals.
  - When current_idx is None (e.g., during short integration tests with
    synthetic data), it falls back to the reference implementation so the
    environment still functions.
"""

import os
import sys
from typing import Optional

# Make parent directory importable so veteran_trader_v2 is reachable
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from veteran_trader_v2 import OHLCV, Indicators, TraderConfig, SmartMoneyAnalyzer


class ICTSignalAdapter:
    """
    Adapter for ICT signal detection backed by SmartMoneyAnalyzer.

    Parameters
    ----------
    ohlcv_data : list[OHLCV]
        Full price history loaded via veteran_trader_v2.load_data().
    config : TraderConfig | None
        Trader config (uses defaults if None).
    """

    # Expected keys in the signals dict
    SIGNAL_KEYS = (
        'bullish_fvg', 'bearish_fvg',
        'bullish_ob',  'bearish_ob',
        'market_structure',
        'fvg_distance',
        'ob_strength',
        'liquidity_sweep',
    )

    def __init__(self, ohlcv_data: list, config: Optional[TraderConfig] = None):
        self.ohlcv_data = ohlcv_data
        self.config = config or TraderConfig()

        self.ind: Optional[Indicators] = None
        self.smc: Optional[SmartMoneyAnalyzer] = None
        self._running_structure: list[float] = []

        if ohlcv_data:
            self.ind = Indicators(ohlcv_data)
            self.smc = SmartMoneyAnalyzer(ohlcv_data, self.ind, self.config)
            self._running_structure = self._compute_running_structure()

    # ------------------------------------------------------------------
    # Public interface (called by TradingEnv every step)
    # ------------------------------------------------------------------

    def get_signals(self, price_history: list[dict],
                    current_idx: Optional[int] = None) -> dict:
        """
        Generate ICT signals.

        Parameters
        ----------
        price_history : list[dict]
            Recent OHLCV dicts (most recent last).  Used only for the
            fallback reference implementation.
        current_idx : int | None
            Current bar index into ohlcv_data / SmartMoneyAnalyzer arrays.
            When provided the pre-computed signals are used.
        """
        if (current_idx is not None
                and self.smc is not None
                and 0 <= current_idx < len(self.ohlcv_data)):
            return self._extract_from_smc(current_idx)

        # Fallback: reference implementation on raw dicts
        return self._detect_signals(price_history)

    # ------------------------------------------------------------------
    # Pre-computed path (rich signals from SmartMoneyAnalyzer)
    # ------------------------------------------------------------------

    def _extract_from_smc(self, idx: int) -> dict:
        signals = self._empty_signals()

        bar   = self.ohlcv_data[idx]
        close = bar.close
        atr   = (self.ind.atr_14[idx] or 1.0)  # ATR used for normalisation

        # ── Fair Value Gaps ────────────────────────────────────────────
        active_fvgs = self.smc.get_active_fvgs_at(idx)
        nearest_dist = float('inf')
        for fvg in active_fvgs:
            dist = abs(close - fvg.midpoint)
            if dist < nearest_dist:
                nearest_dist = dist
            if fvg.direction == 'bullish':
                signals['bullish_fvg'] = 1.0
            else:
                signals['bearish_fvg'] = 1.0

        if active_fvgs:
            # Normalise distance: 0 = price is at the FVG, 1 = 5+ ATRs away
            signals['fvg_distance'] = min(nearest_dist / (atr * 5.0 + 1e-9), 1.0)

        # ── Order Blocks ───────────────────────────────────────────────
        recent_obs = self.smc.get_recent_order_blocks(idx, lookback=30)
        best_strength = 0.0
        for ob in recent_obs:
            if ob.direction == 'bullish':
                signals['bullish_ob'] = 1.0
            elif ob.direction == 'bearish':
                signals['bearish_ob'] = 1.0
            # displacement_size is in ATR multiples; 5 ATRs ≈ a very strong move
            strength = min(ob.displacement_size / 5.0, 1.0)
            if strength > best_strength:
                best_strength = strength
        signals['ob_strength'] = best_strength

        # ── Market Structure (running direction) ───────────────────────
        signals['market_structure'] = self._running_structure[idx]

        # ── Liquidity Sweep ────────────────────────────────────────────
        # Check current bar and the two previous bars for a recent sweep
        for j in range(max(0, idx - 2), idx + 1):
            if self.smc.liquidity_sweeps[j]:
                signals['liquidity_sweep'] = 1.0
                break

        return signals

    def _compute_running_structure(self) -> list:
        """
        Build a list where each element is the most recent known market
        structure direction up to that bar: +1 = bullish, -1 = bearish, 0 = neutral.
        """
        n = len(self.ohlcv_data)
        structure = [0.0] * n
        current = 0.0
        for i in range(n):
            for sb in self.smc.structure_breaks[i]:
                if sb.direction == 'bullish':
                    current = 1.0
                elif sb.direction == 'bearish':
                    current = -1.0
            structure[i] = current
        return structure

    # ------------------------------------------------------------------
    # Fallback / reference implementation (works on raw dicts, no SMC)
    # ------------------------------------------------------------------

    def _detect_signals(self, history: list[dict]) -> dict:
        """Reference implementation — used when no idx is available."""
        signals = self._empty_signals()

        if len(history) < 3:
            return signals

        # --- Fair Value Gap detection ---
        c1, c2, c3 = history[-3], history[-2], history[-1]

        if c3['low'] > c1['high']:
            signals['bullish_fvg'] = 1.0
            gap_size = c3['low'] - c1['high']
            signals['fvg_distance'] = min(gap_size / (c2['close'] + 1e-9), 1.0)

        if c3['high'] < c1['low']:
            signals['bearish_fvg'] = 1.0
            gap_size = c1['low'] - c3['high']
            signals['fvg_distance'] = min(gap_size / (c2['close'] + 1e-9), 1.0)

        # --- Order Block detection ---
        if len(history) >= 5:
            for i in range(len(history) - 5, len(history) - 1):
                candle      = history[i]
                next_candle = history[i + 1]

                if (candle['close'] < candle['open']
                        and next_candle['close'] > next_candle['open']
                        and next_candle['close'] > candle['high']):
                    signals['bullish_ob'] = 1.0
                    move = next_candle['close'] - candle['low']
                    signals['ob_strength'] = min(move / (candle['close'] + 1e-9), 1.0)

                if (candle['close'] > candle['open']
                        and next_candle['close'] < next_candle['open']
                        and next_candle['close'] < candle['low']):
                    signals['bearish_ob'] = 1.0
                    move = candle['high'] - next_candle['close']
                    signals['ob_strength'] = min(move / (candle['close'] + 1e-9), 1.0)

        # --- Market Structure ---
        if len(history) >= 10:
            highs = [c['high'] for c in history[-10:]]
            lows  = [c['low']  for c in history[-10:]]
            hh = highs[-1] > max(highs[:5])
            hl = min(lows[-5:]) > min(lows[:5])
            ll = lows[-1]  < min(lows[:5])
            lh = max(highs[-5:]) < max(highs[:5])
            if hh and hl:
                signals['market_structure'] = 1.0
            elif ll and lh:
                signals['market_structure'] = -1.0

        # --- Liquidity Sweep ---
        if len(history) >= 20:
            recent_high = max(c['high'] for c in history[-20:-1])
            recent_low  = min(c['low']  for c in history[-20:-1])
            last = history[-1]
            if ((last['high'] > recent_high and last['close'] < recent_high)
                    or (last['low'] < recent_low and last['close'] > recent_low)):
                signals['liquidity_sweep'] = 1.0

        return signals

    @staticmethod
    def _empty_signals() -> dict:
        return {
            'bullish_fvg':      0.0,
            'bearish_fvg':      0.0,
            'bullish_ob':       0.0,
            'bearish_ob':       0.0,
            'market_structure': 0.0,
            'fvg_distance':     0.0,
            'ob_strength':      0.0,
            'liquidity_sweep':  0.0,
        }
