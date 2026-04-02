"""
ICT Signal Adapter — Session-Aware
=====================================
Full ICT/Smart Money signal detection with:
- Asia, London, New York session tracking
- Previous Day High/Low (PDH/PDL)
- Judas Swing detection at NY open
- Session liquidity sweeps
- FVG, OB, Market Structure (original signals)

Requires 4H candle data with timestamps to detect sessions.
Falls back gracefully on daily candles (session signals stay neutral).
"""

from datetime import datetime, time
from collections import deque


class SessionTracker:
    """
    Tracks trading sessions and their key levels.

    Sessions (ET / Eastern Time):
        Asia:       7:00 PM - 12:00 AM  (19:00 - 00:00)
        London:     2:00 AM -  5:00 AM  (02:00 - 05:00)
        NY Open:    9:30 AM - 11:00 AM  (09:30 - 11:00)
        NY Session: 9:30 AM -  4:00 PM  (09:30 - 16:00)

    On 4H candles, sessions are approximated:
        Bar ending ~00:00 → Asia
        Bar ending ~04:00 → London
        Bar ending ~08:00 → Pre-NY
        Bar ending ~12:00 → NY Morning (includes open)
        Bar ending ~16:00 → NY Afternoon
        Bar ending ~20:00 → Asia start
    """

    def __init__(self):
        self.previous_day_high = 0.0
        self.previous_day_low = float('inf')
        self.current_day_high = 0.0
        self.current_day_low = float('inf')
        self.asia_high = 0.0
        self.asia_low = float('inf')
        self.london_high = 0.0
        self.london_low = float('inf')
        self._current_date = None
        self._session_bars = deque(maxlen=50)

    def update(self, bar: dict):
        """Update session levels with new bar data."""
        timestamp = bar.get('timestamp', None)
        high = bar.get('high', 0)
        low = bar.get('low', float('inf'))

        bar_date = None
        bar_time = None
        if timestamp:
            if isinstance(timestamp, str):
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    bar_date = dt.date()
                    bar_time = dt.time()
                except (ValueError, AttributeError):
                    pass
            elif isinstance(timestamp, datetime):
                bar_date = timestamp.date()
                bar_time = timestamp.time()

        # Track day transitions
        if bar_date and bar_date != self._current_date:
            if self._current_date is not None:
                self.previous_day_high = self.current_day_high
                self.previous_day_low = self.current_day_low
            self.current_day_high = high
            self.current_day_low = low
            self.asia_high = 0.0
            self.asia_low = float('inf')
            self.london_high = 0.0
            self.london_low = float('inf')
            self._current_date = bar_date
        else:
            self.current_day_high = max(self.current_day_high, high)
            self.current_day_low = min(self.current_day_low, low)

        # Update session-specific levels
        if bar_time:
            session = self._classify_session(bar_time)
            if session == 'asia':
                self.asia_high = max(self.asia_high, high)
                self.asia_low = min(self.asia_low, low)
            elif session == 'london':
                self.london_high = max(self.london_high, high)
                self.london_low = min(self.london_low, low)

        self._session_bars.append(bar)

    def _classify_session(self, t: time) -> str:
        if t >= time(19, 0) or t < time(2, 0):
            return 'asia'
        elif time(2, 0) <= t < time(5, 0):
            return 'london'
        elif time(5, 0) <= t < time(9, 30):
            return 'pre_ny'
        elif time(9, 30) <= t < time(16, 0):
            return 'ny'
        else:
            return 'post_ny'

    def get_current_session(self, bar: dict) -> str:
        timestamp = bar.get('timestamp', None)
        if timestamp:
            if isinstance(timestamp, str):
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    return self._classify_session(dt.time())
                except (ValueError, AttributeError):
                    pass
            elif isinstance(timestamp, datetime):
                return self._classify_session(timestamp.time())
        return 'unknown'

    def get_levels(self) -> dict:
        return {
            'pdh': self.previous_day_high,
            'pdl': self.previous_day_low if self.previous_day_low != float('inf') else 0.0,
            'current_day_high': self.current_day_high,
            'current_day_low': self.current_day_low if self.current_day_low != float('inf') else 0.0,
            'asia_high': self.asia_high,
            'asia_low': self.asia_low if self.asia_low != float('inf') else 0.0,
            'london_high': self.london_high,
            'london_low': self.london_low if self.london_low != float('inf') else 0.0,
        }


class ICTSignalAdapter:
    """
    Session-aware ICT signal detection.

    Returns 15 signals:
        Base (8):    bullish_fvg, bearish_fvg, fvg_distance,
                     bullish_ob, bearish_ob, ob_strength,
                     market_structure, liquidity_sweep

        Session (7): session_asia, session_london, session_ny,
                     judas_swing_bull, judas_swing_bear,
                     asia_range_sweep_high, asia_range_sweep_low
    """

    def __init__(self, your_ict_detector=None):
        self.detector = your_ict_detector
        self.session_tracker = SessionTracker()
        self._initialized = False

    def get_signals(self, price_history: list) -> dict:
        """Generate ICT signals including session-aware detection."""
        if len(price_history) < 3:
            return self._empty_signals()

        # Bootstrap session tracker with full history on first call
        if not self._initialized and len(price_history) > 1:
            for bar in price_history[:-1]:
                self.session_tracker.update(bar)
            self._initialized = True

        current_bar = price_history[-1]
        self.session_tracker.update(current_bar)

        signals = self._detect_base_signals(price_history)

        # Session signals: gracefully neutral when no timestamps present
        has_timestamps = current_bar.get('timestamp') is not None
        if has_timestamps:
            session_signals = self._detect_session_signals(price_history)
        else:
            session_signals = self._neutral_session_signals()
        signals.update(session_signals)

        return signals

    # ── session detection ─────────────────────────────────────────────────────
    def _detect_session_signals(self, history: list) -> dict:
        signals = self._neutral_session_signals()
        current_bar = history[-1]
        levels = self.session_tracker.get_levels()
        session = self.session_tracker.get_current_session(current_bar)

        if session == 'asia':
            signals['session_asia'] = 1.0
        elif session == 'london':
            signals['session_london'] = 1.0
        elif session == 'ny':
            signals['session_ny'] = 1.0

        # Judas Swing (NY Open): price sweeps PDH/PDL then closes back
        if session == 'ny' and levels['pdh'] > 0 and levels['pdl'] > 0:
            high  = current_bar['high']
            low   = current_bar['low']
            close = current_bar['close']
            pdh, pdl = levels['pdh'], levels['pdl']

            if high > pdh and close < pdh:      # bearish Judas
                signals['judas_swing_bear'] = 1.0
            if low < pdl and close > pdl:       # bullish Judas
                signals['judas_swing_bull'] = 1.0

        # Asia Range Sweep: London/NY raids Asia session liquidity
        if session in ('london', 'ny'):
            asia_h = levels['asia_high']
            asia_l = levels['asia_low']
            if asia_h > 0 and asia_l > 0:
                high  = current_bar['high']
                low   = current_bar['low']
                close = current_bar['close']
                if high > asia_h and close < asia_h:
                    signals['asia_range_sweep_high'] = 1.0
                if low < asia_l and close > asia_l:
                    signals['asia_range_sweep_low'] = 1.0

        return signals

    @staticmethod
    def _neutral_session_signals() -> dict:
        return {
            'session_asia': 0.0,
            'session_london': 0.0,
            'session_ny': 0.0,
            'judas_swing_bull': 0.0,
            'judas_swing_bear': 0.0,
            'asia_range_sweep_high': 0.0,
            'asia_range_sweep_low': 0.0,
        }

    # ── base ICT detection ────────────────────────────────────────────────────
    def _detect_base_signals(self, history: list) -> dict:
        signals = self._empty_signals()
        if len(history) < 3:
            return signals

        c1, c2, c3 = history[-3], history[-2], history[-1]

        # Fair Value Gap
        if c3['low'] > c1['high']:
            signals['bullish_fvg'] = 1.0
            gap_size = c3['low'] - c1['high']
            signals['fvg_distance'] = min(gap_size / (c2['close'] + 1e-9), 1.0)
        elif c3['high'] < c1['low']:
            signals['bearish_fvg'] = 1.0
            gap_size = c1['low'] - c3['high']
            signals['fvg_distance'] = min(gap_size / (c2['close'] + 1e-9), 1.0)

        # Order Blocks
        if len(history) >= 5:
            for i in range(len(history) - 5, len(history) - 2):
                candle      = history[i]
                next_candle = history[i + 1]
                # Bullish OB: bearish candle followed by strong bullish move
                if (candle['close'] < candle['open']
                        and next_candle['close'] > next_candle['open']
                        and next_candle['close'] > candle['high']):
                    signals['bullish_ob'] = 1.0
                    move = next_candle['close'] - candle['low']
                    signals['ob_strength'] = min(move / (candle['close'] + 1e-9), 1.0)
                # Bearish OB: bullish candle followed by strong bearish move
                if (candle['close'] > candle['open']
                        and next_candle['close'] < next_candle['open']
                        and next_candle['close'] < candle['low']):
                    signals['bearish_ob'] = 1.0
                    move = candle['high'] - next_candle['close']
                    signals['ob_strength'] = min(move / (candle['close'] + 1e-9), 1.0)

        # Market Structure (Higher Highs/Lows vs Lower Lows/Highs)
        if len(history) >= 10:
            recent = history[-10:]
            highs  = [c['high'] for c in recent]
            lows   = [c['low']  for c in recent]
            hh = highs[-1] > max(highs[:5])
            hl = min(lows[-5:]) > min(lows[:5])
            ll = lows[-1] < min(lows[:5])
            lh = max(highs[-5:]) < max(highs[:5])
            if hh and hl:
                signals['market_structure'] = 1.0
            elif ll and lh:
                signals['market_structure'] = -1.0

        # Liquidity Sweep (wick above recent high or below recent low, closes back)
        if len(history) >= 20:
            recent_high = max(c['high'] for c in history[-20:-1])
            recent_low  = min(c['low']  for c in history[-20:-1])
            last = history[-1]
            swept_high = last['high'] > recent_high and last['close'] < recent_high
            swept_low  = last['low']  < recent_low  and last['close'] > recent_low
            if swept_high or swept_low:
                signals['liquidity_sweep'] = 1.0

        return signals

    @staticmethod
    def _empty_signals() -> dict:
        return {
            'bullish_fvg': 0.0, 'bearish_fvg': 0.0,
            'bullish_ob': 0.0,  'bearish_ob': 0.0,
            'market_structure': 0.0, 'fvg_distance': 0.0,
            'ob_strength': 0.0, 'liquidity_sweep': 0.0,
            'session_asia': 0.0, 'session_london': 0.0, 'session_ny': 0.0,
            'judas_swing_bull': 0.0, 'judas_swing_bear': 0.0,
            'asia_range_sweep_high': 0.0, 'asia_range_sweep_low': 0.0,
        }
