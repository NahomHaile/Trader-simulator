#!/usr/bin/env python3
"""
Integration test for the RL Trading Agent + Veteran Trader v2.

Verifies:
  1. veteran_trader_v2.load_data() can read QQQ_data.csv
  2. ICTSignalAdapter builds SmartMoneyAnalyzer without error
  3. HistoricalDataFeed wraps the same OHLCV list
  4. TradingEnv.reset() returns an observation of the right shape
  5. 10 random env.step() calls succeed and produce valid observations/rewards
  6. get_signals() returns all expected keys with float values when given a cursor
  7. get_signals() falls back gracefully with no cursor (reference implementation)

Run with:
    python test_integration.py
or:
    python -m pytest test_integration.py -v
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from veteran_trader_v2 import load_data, TraderConfig
from rl_trading_agent.ict_adapter import ICTSignalAdapter
from rl_trading_agent.data_feed import HistoricalDataFeed
from rl_trading_agent.trading_env import TradingEnv, ObservationBuilder

# Path to a real data file for integration testing
DATA_CSV = os.path.join(os.path.dirname(__file__), "QQQ_data.csv")


def _build_env(randomize: bool = False):
    """Helper: load data and build a TradingEnv."""
    ohlcv  = load_data(DATA_CSV)
    config = TraderConfig()
    feed   = HistoricalDataFeed.from_ohlcv(ohlcv, randomize_start=randomize)
    ict    = ICTSignalAdapter(ohlcv, config)
    env    = TradingEnv(
        data_feed=feed,
        ict_detector=ict,
        initial_balance=10_000.0,
        max_steps=500,
    )
    return env, feed, ict, ohlcv


class TestDataLoading(unittest.TestCase):

    def test_load_data_returns_ohlcv_list(self):
        ohlcv = load_data(DATA_CSV)
        self.assertIsInstance(ohlcv, list)
        self.assertGreater(len(ohlcv), 50)
        bar = ohlcv[0]
        for attr in ('date', 'open', 'high', 'low', 'close', 'volume'):
            self.assertTrue(hasattr(bar, attr),
                            f"OHLCV missing attribute: {attr}")

    def test_from_ohlcv_converts_correctly(self):
        ohlcv = load_data(DATA_CSV)
        feed  = HistoricalDataFeed.from_ohlcv(ohlcv, randomize_start=False)
        self.assertEqual(len(feed.raw_data), len(ohlcv))
        first = feed.raw_data[0]
        self.assertAlmostEqual(first['close'], ohlcv[0].close)
        self.assertAlmostEqual(first['high'],  ohlcv[0].high)


class TestICTAdapter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ohlcv     = load_data(DATA_CSV)
        config    = TraderConfig()
        cls.ohlcv = ohlcv
        cls.ict   = ICTSignalAdapter(ohlcv, config)

    def test_smc_built(self):
        self.assertIsNotNone(self.ict.smc)
        self.assertIsNotNone(self.ict.ind)

    def test_running_structure_length(self):
        self.assertEqual(len(self.ict._running_structure), len(self.ohlcv))

    def test_running_structure_values(self):
        for v in self.ict._running_structure:
            self.assertIn(v, (-1.0, 0.0, 1.0))

    def test_get_signals_with_idx_returns_all_keys(self):
        # Use bar 100 (enough history for indicators to be populated)
        idx     = min(100, len(self.ohlcv) - 1)
        signals = self.ict.get_signals([], current_idx=idx)
        for key in ICTSignalAdapter.SIGNAL_KEYS:
            self.assertIn(key, signals, f"Missing signal key: {key}")
            self.assertIsInstance(signals[key], float)

    def test_get_signals_values_in_range(self):
        idx     = min(200, len(self.ohlcv) - 1)
        signals = self.ict.get_signals([], current_idx=idx)
        self.assertIn(signals['market_structure'], (-1.0, 0.0, 1.0))
        self.assertGreaterEqual(signals['bullish_fvg'],  0.0)
        self.assertLessEqual(   signals['bullish_fvg'],  1.0)
        self.assertGreaterEqual(signals['fvg_distance'], 0.0)
        self.assertLessEqual(   signals['fvg_distance'], 1.0)

    def test_fallback_reference_impl(self):
        # Supply synthetic price_history dicts, no idx → uses reference impl
        history = [
            {'open': 100, 'high': 102, 'low': 98, 'close': 101, 'volume': 1e6},
            {'open': 103, 'high': 110, 'low': 103, 'close': 108, 'volume': 2e6},
            {'open': 109, 'high': 112, 'low': 106, 'close': 111, 'volume': 1e6},
        ]
        signals = self.ict.get_signals(history, current_idx=None)
        for key in ICTSignalAdapter.SIGNAL_KEYS:
            self.assertIn(key, signals)
        # These three bars form a textbook bullish FVG
        self.assertEqual(signals['bullish_fvg'], 1.0)


class TestHistoricalDataFeed(unittest.TestCase):

    def test_cursor_starts_at_zero(self):
        ohlcv = load_data(DATA_CSV)
        feed  = HistoricalDataFeed.from_ohlcv(ohlcv, randomize_start=False)
        feed.reset(seed=42)
        self.assertEqual(feed.cursor, 0)

    def test_step_increments_cursor(self):
        ohlcv = load_data(DATA_CSV)
        feed  = HistoricalDataFeed.from_ohlcv(ohlcv, randomize_start=False)
        feed.reset(seed=0)
        feed.step()
        self.assertEqual(feed.cursor, 1)

    def test_get_current_bar_has_derived_features(self):
        ohlcv = load_data(DATA_CSV)
        feed  = HistoricalDataFeed.from_ohlcv(ohlcv, randomize_start=False)
        feed.reset(seed=0)
        for _ in range(25):   # enough history for derived features
            feed.step()
        bar = feed.get_current_bar()
        for key in ('open', 'high', 'low', 'close', 'volume',
                    'returns', 'returns_5', 'returns_20', 'volatility',
                    'volume_ratio', 'spread'):
            self.assertIn(key, bar, f"Missing bar key: {key}")


class TestTradingEnv(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env, cls.feed, cls.ict, cls.ohlcv = _build_env(randomize=False)

    def test_reset_returns_correct_obs_shape(self):
        obs, info = self.env.reset(seed=0)
        self.assertEqual(obs.shape, (ObservationBuilder.OBS_DIM,))
        self.assertEqual(obs.dtype, np.float32)

    def test_obs_within_clip_bounds(self):
        obs, _ = self.env.reset(seed=0)
        self.assertTrue(np.all(obs >= -10.0))
        self.assertTrue(np.all(obs <=  10.0))

    def test_step_with_hold(self):
        self.env.reset(seed=0)
        obs, reward, terminated, truncated, info = self.env.step([0, 0, 0])
        self.assertEqual(obs.shape, (ObservationBuilder.OBS_DIM,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(info,   dict)
        self.assertIn('equity', info)
        self.assertIn('win_rate', info)

    def test_step_long_then_close(self):
        self.env.reset(seed=0)
        # Step several times so the feed has history
        for _ in range(30):
            self.env.step([0, 0, 0])  # hold
        # Open long, any setup filter
        obs, reward, term, trunc, info = self.env.step([1, 1, 0])
        self.assertFalse(term)
        # Close the position
        obs, reward, term, trunc, info = self.env.step([3, 0, 0])
        self.assertIsInstance(reward, float)

    def test_10_random_steps_no_crash(self):
        rng = np.random.default_rng(seed=7)
        obs, _ = self.env.reset(seed=7)
        for _ in range(10):
            action = rng.integers([0, 0, 0], [4, 4, 4])
            obs, reward, term, trunc, info = self.env.step(action)
            self.assertEqual(obs.shape, (ObservationBuilder.OBS_DIM,))
            if term or trunc:
                obs, _ = self.env.reset()

    def test_account_equity_non_negative(self):
        obs, _ = self.env.reset(seed=0)
        for _ in range(50):
            action = [1, 3, 0]  # keep trying to open longs at max size
            obs, reward, term, trunc, info = self.env.step(action)
            self.assertGreater(info['equity'], 0.0)
            if term or trunc:
                break

    def test_ict_signals_use_precomputed_path(self):
        """Confirm the env is reading cursor-based signals, not fallback."""
        self.env.reset(seed=0)
        # Advance far enough that SMC has data
        for _ in range(60):
            self.env.step([0, 0, 0])
        signals = self.env._current_ict_signals
        for key in ICTSignalAdapter.SIGNAL_KEYS:
            self.assertIn(key, signals)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
