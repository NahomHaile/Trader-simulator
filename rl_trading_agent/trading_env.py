"""
RL Trading Environment for ICT/Smart Money Strategy
=====================================================
Custom Gymnasium environment that wraps the existing Veteran Trader logic.
The agent learns: entry/exit timing, position sizing, and setup filtering.

Changes vs. original template
------------------------------
- _update_market_state() passes the data feed's cursor to get_signals() so
  ICTSignalAdapter can look up pre-computed SmartMoneyAnalyzer signals instead
  of re-running detection on every step.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action definitions
# ---------------------------------------------------------------------------
class TradeAction(IntEnum):
    HOLD            = 0
    LONG_ENTRY      = 1
    SHORT_ENTRY     = 2
    CLOSE_POSITION  = 3


@dataclass
class Position:
    direction:       str
    entry_price:     float = 0.0
    size:            float = 0.0
    entry_step:      int   = 0
    unrealized_pnl:  float = 0.0


@dataclass
class AccountState:
    balance:         float = 10_000.0
    initial_balance: float = 10_000.0
    equity:          float = 10_000.0
    position:        Optional[Position] = None
    trade_history:   list  = field(default_factory=list)
    peak_equity:     float = 10_000.0
    total_trades:    int   = 0
    winning_trades:  int   = 0


# ---------------------------------------------------------------------------
# Observation Builder
# ---------------------------------------------------------------------------
class ObservationBuilder:
    """
    Builds the 21-feature state vector the agent sees each step.

    Slots
    -----
    [0-5]   Market data  : returns (1, 5, 20-bar), volatility, volume ratio, spread
    [6-13]  ICT signals  : bullish/bearish FVG & OB, market structure, FVG distance,
                           OB strength, liquidity sweep
    [14-17] Account state: long/short flags, unrealised PnL, drawdown
    [18-20] Temporal     : intraday progress, weekly progress, bars in position
    """

    OBS_DIM = 21

    @staticmethod
    def build(market_data: dict, ict_signals: dict,
              account: AccountState, current_step: int) -> np.ndarray:
        obs = np.zeros(ObservationBuilder.OBS_DIM, dtype=np.float32)

        obs[0]  = market_data.get('returns',      0.0)
        obs[1]  = market_data.get('returns_5',    0.0)
        obs[2]  = market_data.get('returns_20',   0.0)
        obs[3]  = market_data.get('volatility',   0.0)
        obs[4]  = market_data.get('volume_ratio', 1.0)
        obs[5]  = market_data.get('spread',       0.0)

        obs[6]  = ict_signals.get('bullish_fvg',       0.0)
        obs[7]  = ict_signals.get('bearish_fvg',       0.0)
        obs[8]  = ict_signals.get('bullish_ob',        0.0)
        obs[9]  = ict_signals.get('bearish_ob',        0.0)
        obs[10] = ict_signals.get('market_structure',  0.0)
        obs[11] = ict_signals.get('fvg_distance',      0.0)
        obs[12] = ict_signals.get('ob_strength',       0.0)
        obs[13] = ict_signals.get('liquidity_sweep',   0.0)

        obs[14] = 1.0 if account.position and account.position.direction == 'long'  else 0.0
        obs[15] = 1.0 if account.position and account.position.direction == 'short' else 0.0
        obs[16] = (account.position.unrealized_pnl / account.balance
                   if account.position else 0.0)
        obs[17] = (account.equity - account.peak_equity) / (account.peak_equity + 1e-9)

        # Temporal features tuned for 4H candles (6/day, ~30/week, ~126/month)
        obs[18] = (current_step % 30) / 30.0                # weekly cycle progress
        obs[19] = (current_step % 126) / 126.0              # monthly cycle progress
        bars_in = (current_step - (account.position.entry_step
                                   if account.position else current_step))
        obs[20] = min(bars_in, 60) / 60.0                   # bars in position, ~10 trading days

        return np.clip(obs, -10.0, 10.0)


# ---------------------------------------------------------------------------
# Reward Function
# ---------------------------------------------------------------------------
class RewardCalculator:
    """
    Reward shaping tuned for 4H candle swing trading.

    Key differences from intraday:
    - Zero holding penalty  (holding IS the strategy)
    - 8% drawdown tolerance (swing trades have larger swings)
    - Harsh overtrading penalty (patience is rewarded)
    - R:R bonus for >2% winners (encourages 2R+ setups)
    - Cooldown of 12 bars (~2 days of 4H) before trade counter decays
    """

    def __init__(
        self,
        pnl_weight:          float = 1.5,
        drawdown_penalty:    float = 1.5,
        holding_penalty:     float = 0.0,
        overtrading_penalty: float = 0.02,
        win_bonus:           float = 0.02,
        rr_bonus_weight:     float = 0.01,
    ):
        self.pnl_weight          = pnl_weight
        self.drawdown_penalty    = drawdown_penalty
        self.holding_penalty     = holding_penalty
        self.overtrading_penalty = overtrading_penalty
        self.win_bonus           = win_bonus
        self.rr_bonus_weight     = rr_bonus_weight
        self._recent_trades  = 0
        self._trade_cooldown = 0

    def calculate(self, pnl_change: float, account: AccountState,
                  action: int, was_trade: bool) -> float:
        reward = 0.0

        # 1. PnL signal
        reward += self.pnl_weight * (pnl_change / max(account.balance, 1.0))

        # 2. Drawdown penalty — wider 8% tolerance for swing
        drawdown = (account.peak_equity - account.equity) / (account.peak_equity + 1e-9)
        if drawdown > 0.08:
            reward -= self.drawdown_penalty * drawdown

        # 3. Holding cost (zero for swing)
        if account.position is not None:
            reward -= self.holding_penalty

        # 4. Overtrading penalty — stricter cooldown (~2 days of 4H bars)
        if was_trade:
            self._recent_trades += 1
            self._trade_cooldown = 12
        elif self._trade_cooldown > 0:
            self._trade_cooldown -= 1
            if self._trade_cooldown == 0:
                self._recent_trades = max(0, self._recent_trades - 1)

        if self._recent_trades > 2:
            reward -= self.overtrading_penalty * self._recent_trades

        # 5. Win bonus + R:R bonus for >2% gains
        if was_trade and action == TradeAction.CLOSE_POSITION:
            last = account.trade_history[-1] if account.trade_history else None
            if last and last.get('pnl', 0) > 0:
                reward += self.win_bonus
                pnl_pct = last['pnl'] / max(account.balance, 1.0)
                if pnl_pct > 0.02:
                    reward += self.rr_bonus_weight * (pnl_pct / 0.02)

        return reward

    def reset(self):
        self._recent_trades  = 0
        self._trade_cooldown = 0


# ---------------------------------------------------------------------------
# Main Gymnasium Environment
# ---------------------------------------------------------------------------
class TradingEnv(gym.Env):
    """
    Gymnasium environment for RL-based ICT/SMC trading.

    Action Space  MultiDiscrete([4, 4, 4])
    ─────────────────────────────────────────────────
    [0] trade_decision  : 0=hold, 1=long, 2=short, 3=close
    [1] position_size   : 0=1%, 1=2%, 2=3%, 3=5%  of equity
    [2] setup_filter    : 0=any, 1=FVG-only, 2=OB-only, 3=confluence

    Observation Space   Box(21,) — see ObservationBuilder
    """

    metadata = {"render_modes": ["human", "log"]}

    SIZE_MAP   = {0: 0.02, 1: 0.03, 2: 0.05, 3: 0.08}  # swing sizes
    FILTER_MAP = {0: 'any', 1: 'fvg_only', 2: 'ob_only', 3: 'confluence'}

    def __init__(
        self,
        data_feed,
        ict_detector,
        initial_balance: float = 10_000.0,
        max_steps:       int   = 1_500,  # ~1 year of 4H candles
        render_mode:     str   = None,
    ):
        super().__init__()

        self.data_feed       = data_feed
        self.ict_detector    = ict_detector
        self.initial_balance = initial_balance
        self.max_steps       = max_steps
        self.render_mode     = render_mode

        self.action_space = spaces.MultiDiscrete([4, 4, 4])
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(ObservationBuilder.OBS_DIM,),
            dtype=np.float32,
        )

        self.account = AccountState(
            balance=initial_balance,
            initial_balance=initial_balance,
            equity=initial_balance,
            peak_equity=initial_balance,
        )
        self.reward_calc  = RewardCalculator()
        self.current_step = 0
        self._current_market_data  = {}
        self._current_ict_signals  = {}

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.account = AccountState(
            balance=self.initial_balance,
            initial_balance=self.initial_balance,
            equity=self.initial_balance,
            peak_equity=self.initial_balance,
        )
        self.reward_calc.reset()
        self.current_step = 0
        self.data_feed.reset(seed=seed)
        self._update_market_state()
        obs  = ObservationBuilder.build(
            self._current_market_data, self._current_ict_signals,
            self.account, self.current_step,
        )
        return obs, self._get_info()

    # ------------------------------------------------------------------
    def step(self, action):
        trade_decision = TradeAction(action[0])
        size_pct       = self.SIZE_MAP[action[1]]
        setup_filter   = self.FILTER_MAP[action[2]]

        prev_equity = self.account.equity
        was_trade   = False

        if not self._passes_filter(trade_decision, setup_filter):
            trade_decision = TradeAction.HOLD

        if trade_decision == TradeAction.LONG_ENTRY and self.account.position is None:
            self._open_position('long', size_pct)
            was_trade = True
        elif trade_decision == TradeAction.SHORT_ENTRY and self.account.position is None:
            self._open_position('short', size_pct)
            was_trade = True
        elif trade_decision == TradeAction.CLOSE_POSITION and self.account.position is not None:
            self._close_position()
            was_trade = True

        self.current_step += 1
        self.data_feed.step()
        self._update_market_state()
        self._update_unrealized_pnl()

        pnl_change = self.account.equity - prev_equity
        reward = self.reward_calc.calculate(
            pnl_change, self.account, action[0], was_trade
        )

        terminated = False
        truncated  = False
        if self.account.equity <= self.initial_balance * 0.50:
            terminated = True
            reward -= 1.0
        # Truncate when max_steps is hit OR the data feed has no more bars
        feed_done = getattr(self.data_feed, 'done', False)
        if self.current_step >= self.max_steps or feed_done:
            truncated = True

        obs = ObservationBuilder.build(
            self._current_market_data, self._current_ict_signals,
            self.account, self.current_step,
        )
        return obs, reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    def _open_position(self, direction: str, size_pct: float):
        price = self._current_market_data.get('close', 1.0)
        size  = (self.account.equity * size_pct) / max(price, 1e-9)
        self.account.position = Position(
            direction=direction, entry_price=price,
            size=size, entry_step=self.current_step,
        )
        self.account.total_trades += 1
        logger.debug(f"OPEN {direction} @ {price:.2f}, size={size:.4f}")

    def _close_position(self):
        pos   = self.account.position
        price = self._current_market_data.get('close', pos.entry_price)
        pnl   = ((price - pos.entry_price) if pos.direction == 'long'
                  else (pos.entry_price - price)) * pos.size
        self.account.balance   += pnl
        self.account.equity     = self.account.balance
        self.account.peak_equity = max(self.account.peak_equity, self.account.equity)
        if pnl > 0:
            self.account.winning_trades += 1
        self.account.trade_history.append({
            'direction':   pos.direction,
            'entry_price': pos.entry_price,
            'exit_price':  price,
            'size':        pos.size,
            'pnl':         pnl,
            'bars_held':   self.current_step - pos.entry_step,
        })
        logger.debug(f"CLOSE {pos.direction} @ {price:.2f}, PnL={pnl:.2f}")
        self.account.position = None

    def _update_unrealized_pnl(self):
        if self.account.position is None:
            self.account.equity = self.account.balance
            return
        pos   = self.account.position
        price = self._current_market_data.get('close', pos.entry_price)
        pos.unrealized_pnl = ((price - pos.entry_price) if pos.direction == 'long'
                               else (pos.entry_price - price)) * pos.size
        self.account.equity      = self.account.balance + pos.unrealized_pnl
        self.account.peak_equity = max(self.account.peak_equity, self.account.equity)

    def _passes_filter(self, decision: TradeAction, setup_filter: str) -> bool:
        if decision in (TradeAction.HOLD, TradeAction.CLOSE_POSITION):
            return True
        s       = self._current_ict_signals
        has_fvg = s.get('bullish_fvg', 0) or s.get('bearish_fvg', 0)
        has_ob  = s.get('bullish_ob',  0) or s.get('bearish_ob',  0)
        if setup_filter == 'any':
            return True
        if setup_filter == 'fvg_only':
            return bool(has_fvg)
        if setup_filter == 'ob_only':
            return bool(has_ob)
        if setup_filter == 'confluence':
            return bool(has_fvg and has_ob)
        return True

    def _update_market_state(self):
        """
        Pull current market data and ICT signals.

        The data feed's cursor is passed to get_signals() so ICTSignalAdapter
        can look up the pre-computed SmartMoneyAnalyzer result for this bar.
        """
        self._current_market_data = self.data_feed.get_current_bar()
        cursor = getattr(self.data_feed, 'cursor', None)
        self._current_ict_signals = self.ict_detector.get_signals(
            self.data_feed.get_history(),
            current_idx=cursor,
        )

    def _get_info(self) -> dict:
        win_rate = (self.account.winning_trades
                    / max(self.account.total_trades, 1))
        return {
            'equity':       self.account.equity,
            'balance':      self.account.balance,
            'total_trades': self.account.total_trades,
            'win_rate':     win_rate,
            'drawdown':     ((self.account.peak_equity - self.account.equity)
                             / (self.account.peak_equity + 1e-9)),
            'step':         self.current_step,
            'has_position': self.account.position is not None,
        }
