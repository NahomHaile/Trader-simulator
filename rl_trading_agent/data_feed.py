"""
Data Feed Adapter
==================
Bridges veteran_trader_v2 market data to the RL environment.

Key additions vs. the original template:

  HistoricalDataFeed.from_ohlcv(ohlcv_list)
      Converts a list[OHLCV] (from veteran_trader_v2.load_data) directly
      into the feed, avoiding a second CSV parse.

  HistoricalDataFeed.from_veteran_csv(filepath)
      Thin wrapper: calls veteran_trader_v2.load_data then from_ohlcv.

The cursor attribute on HistoricalDataFeed is the current bar index into
raw_data.  ICTSignalAdapter reads this to look up pre-computed SMC signals.
"""

import os
import sys
import numpy as np
from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

# Make parent directory importable
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)


class BaseDataFeed(ABC):
    """Abstract base for market data feeds."""

    @abstractmethod
    def reset(self, seed: Optional[int] = None):
        ...

    @abstractmethod
    def step(self):
        ...

    @abstractmethod
    def get_current_bar(self) -> dict:
        ...

    @abstractmethod
    def get_history(self) -> list[dict]:
        ...


class PaperTradingFeed(BaseDataFeed):
    """
    Live paper trading data feed.

    Implement _fetch_next_bar() with your actual data source (Alpaca, IBKR, etc).
    Expected return format:
        {'open': float, 'high': float, 'low': float,
         'close': float, 'volume': float, 'timestamp': str}
    """

    def __init__(self, symbol: str = "ES", lookback: int = 100):
        self.symbol = symbol
        self.lookback = lookback
        self.history = deque(maxlen=lookback)
        self._current_bar = {}
        self.cursor = 0

    def reset(self, seed: Optional[int] = None):
        self.history.clear()
        self.cursor = 0
        for _ in range(self.lookback):
            bar = self._fetch_next_bar()
            if bar:
                self.history.append(bar)
        if self.history:
            self._current_bar = self.history[-1]

    def step(self):
        bar = self._fetch_next_bar()
        if bar:
            self.history.append(bar)
            self._current_bar = bar
            self.cursor += 1

    def get_current_bar(self) -> dict:
        return self._enrich_bar(self._current_bar)

    def get_history(self) -> list[dict]:
        return list(self.history)

    def _enrich_bar(self, bar: dict) -> dict:
        enriched = dict(bar)
        hist = list(self.history)
        if len(hist) >= 2:
            enriched['returns'] = (hist[-1]['close'] - hist[-2]['close']) / hist[-2]['close']
        else:
            enriched['returns'] = 0.0
        if len(hist) >= 7:
            enriched['returns_5'] = (hist[-1]['close'] - hist[-7]['close']) / hist[-7]['close']
        else:
            enriched['returns_5'] = 0.0
        if len(hist) >= 31:
            enriched['returns_20'] = (hist[-1]['close'] - hist[-31]['close']) / hist[-31]['close']
            closes = [h['close'] for h in list(hist)[-31:]]
            enriched['volatility'] = float(np.std(closes) / (np.mean(closes) + 1e-9))
        else:
            enriched['returns_20'] = 0.0
            enriched['volatility'] = 0.0
        if len(hist) >= 30:
            volumes = [h['volume'] for h in list(hist)[-30:]]
            avg_vol = np.mean(volumes)
            enriched['volume_ratio'] = bar.get('volume', 1) / max(avg_vol, 1)
        else:
            enriched['volume_ratio'] = 1.0
        enriched['spread'] = ((bar.get('high', 0) - bar.get('low', 0))
                               / max(bar.get('close', 1), 1))
        return enriched

    def _fetch_next_bar(self) -> dict:
        """
        INTEGRATION POINT: implement with your live data source.
        """
        raise NotImplementedError(
            "Implement _fetch_next_bar() with your data source."
        )


class HistoricalDataFeed(BaseDataFeed):
    """
    Replays historical OHLCV data for backtesting / agent pre-training.

    The cursor attribute tracks the current position in raw_data and is
    used by ICTSignalAdapter to look up pre-computed SmartMoneyAnalyzer
    signals at the correct bar index.
    """

    def __init__(self, data: list[dict], lookback: int = 100,
                 randomize_start: bool = True):
        self.raw_data = data
        self.lookback = lookback
        self.randomize_start = randomize_start
        self.history = deque(maxlen=lookback)
        self.cursor = 0
        self._rng = np.random.default_rng()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.history.clear()
        if self.randomize_start:
            max_start = max(0, len(self.raw_data) - 2000)
            self.cursor = int(self._rng.integers(0, max(max_start, 1)))
        else:
            self.cursor = 0
        # Pre-fill: load bars [cursor-lookback .. cursor] so the current bar is
        # always history[-1] and get_current_bar() never returns stale/zero data.
        start = max(0, self.cursor - self.lookback)
        for idx in range(start, min(self.cursor + 1, len(self.raw_data))):
            self.history.append(self.raw_data[idx])

    def step(self):
        self.cursor += 1
        if self.cursor < len(self.raw_data):
            self.history.append(self.raw_data[self.cursor])

    def get_current_bar(self) -> dict:
        if not self.history:
            return {'open': 0, 'high': 0, 'low': 0, 'close': 0, 'volume': 0}
        bar = dict(self.history[-1])
        hist = list(self.history)
        if len(hist) >= 2:
            bar['returns'] = (hist[-1]['close'] - hist[-2]['close']) / hist[-2]['close']
        else:
            bar['returns'] = 0.0
        if len(hist) >= 7:
            bar['returns_5'] = (hist[-1]['close'] - hist[-7]['close']) / hist[-7]['close']
        else:
            bar['returns_5'] = 0.0
        if len(hist) >= 31:
            bar['returns_20'] = (hist[-1]['close'] - hist[-31]['close']) / hist[-31]['close']
            closes = [h['close'] for h in hist[-31:]]
            bar['volatility'] = float(np.std(closes) / (np.mean(closes) + 1e-9))
        else:
            bar['returns_20'] = 0.0
            bar['volatility'] = 0.0
        if len(hist) >= 30:
            volumes = [h['volume'] for h in hist[-30:]]
            bar['volume_ratio'] = bar.get('volume', 1) / max(np.mean(volumes), 1)
        else:
            bar['volume_ratio'] = 1.0
        bar['spread'] = ((bar.get('high', 0) - bar.get('low', 0))
                          / max(bar.get('close', 1), 1))
        return bar

    def get_history(self) -> list[dict]:
        return list(self.history)

    @property
    def done(self) -> bool:
        return self.cursor >= len(self.raw_data) - 1

    # ------------------------------------------------------------------
    # Constructors that understand veteran_trader_v2 data
    # ------------------------------------------------------------------

    @classmethod
    def from_ohlcv(cls, ohlcv_list: list, **kwargs) -> 'HistoricalDataFeed':
        """
        Build a feed directly from a list[OHLCV] returned by load_data().

        Use this when you already have the OHLCV list (e.g., you loaded it
        to pass to ICTSignalAdapter) to avoid parsing the CSV twice.
        """
        data = [
            {
                'open':   bar.open,
                'high':   bar.high,
                'low':    bar.low,
                'close':  bar.close,
                'volume': bar.volume,
            }
            for bar in ohlcv_list
        ]
        return cls(data=data, **kwargs)

    @classmethod
    def from_veteran_csv(cls, filepath: str, **kwargs) -> 'HistoricalDataFeed':
        """
        Load an OHLCV CSV using veteran_trader_v2.load_data(), then build
        a HistoricalDataFeed from the result.

        Returns
        -------
        feed : HistoricalDataFeed
        ohlcv_list : list[OHLCV]
            Also returned so you can pass it to ICTSignalAdapter without
            reading the file twice.
        """
        from veteran_trader_v2 import load_data
        ohlcv_list = load_data(filepath)
        return cls.from_ohlcv(ohlcv_list, **kwargs), ohlcv_list

    @classmethod
    def from_csv(cls, filepath: str, **kwargs) -> 'HistoricalDataFeed':
        """
        Load from a generic CSV (pandas).
        Columns required (case-insensitive): open, high, low, close, volume.
        """
        import pandas as pd
        df = pd.read_csv(filepath)
        df.columns = [c.lower().strip() for c in df.columns]
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
        data = df[required].to_dict('records')
        return cls(data=data, **kwargs)
