"""
CSV Data Feed
=============
Wraps a pandas OHLCV DataFrame into a step-by-step market data feed
compatible with TradingEnv.
"""

import numpy as np
import pandas as pd
from typing import Optional


class CSVDataFeed:
    """
    Stateful data feed over a CSV-loaded OHLCV DataFrame.

    Interface expected by TradingEnv:
        reset(seed)          – reset position (random or fixed start)
        step()               – advance one bar (clamped at last bar)
        get_current_bar()    – dict of current OHLCV + derived features
        get_history()        – list of last HISTORY_LEN bar dicts
        is_done (property)   – True when at or past the last bar
    """

    HISTORY_LEN = 50  # bars of history passed to ICT detector

    def __init__(
        self,
        df: pd.DataFrame,
        history_len: int = HISTORY_LEN,
        random_start: bool = True,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.history_len = history_len
        self.random_start = random_start
        self._rng = np.random.default_rng()
        self._precompute_features()
        self.current_idx = history_len  # safe default; overwritten by reset()

    # ── feature computation ───────────────────────────────────────────────────
    def _precompute_features(self):
        c = self.df["Close"]
        self.df["returns"]      = c.pct_change(1).fillna(0)
        self.df["returns_5"]    = c.pct_change(5).fillna(0)
        self.df["returns_20"]   = c.pct_change(20).fillna(0)
        self.df["volatility"]   = c.pct_change().rolling(20).std().fillna(0)
        self.df["volume_ratio"] = (
            self.df["Volume"] / self.df["Volume"].rolling(20).mean()
        ).fillna(1.0)
        self.df["spread"] = ((self.df["High"] - self.df["Low"]) / c).fillna(0)

    # ── gym-style interface ───────────────────────────────────────────────────
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self.random_start:
            # Leave at least 200 bars remaining for the episode
            max_start = len(self.df) - 200
            if max_start > self.history_len:
                self.current_idx = int(self._rng.integers(self.history_len, max_start))
            else:
                self.current_idx = self.history_len
        else:
            self.current_idx = self.history_len

    def step(self):
        """Advance one bar. Clamped so it never goes past the last row."""
        if self.current_idx < len(self.df) - 1:
            self.current_idx += 1

    @property
    def is_done(self) -> bool:
        return self.current_idx >= len(self.df) - 1

    # ── data accessors ────────────────────────────────────────────────────────
    def get_current_bar(self) -> dict:
        row = self.df.iloc[self.current_idx]
        return {
            "open":         float(row["Open"]),
            "high":         float(row["High"]),
            "low":          float(row["Low"]),
            "close":        float(row["Close"]),
            "volume":       float(row["Volume"]),
            "returns":      float(row["returns"]),
            "returns_5":    float(row["returns_5"]),
            "returns_20":   float(row["returns_20"]),
            "volatility":   float(row["volatility"]),
            "volume_ratio": float(row["volume_ratio"]),
            "spread":       float(row["spread"]),
        }

    def get_history(self) -> list:
        """Return the last history_len bars as a list of OHLCV dicts."""
        start = max(0, self.current_idx - self.history_len)
        rows = self.df.iloc[start : self.current_idx + 1]
        return [
            {
                "open":   float(r["Open"]),
                "high":   float(r["High"]),
                "low":    float(r["Low"]),
                "close":  float(r["Close"]),
                "volume": float(r["Volume"]),
            }
            for _, r in rows.iterrows()
        ]
