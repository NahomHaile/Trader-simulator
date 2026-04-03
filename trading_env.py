"""
RL Trading Environment for ICT/Smart Money Strategy
=====================================================
Custom Gymnasium environment that wraps around your existing trading logic.
The agent learns: entry/exit timing, position sizing, and setup filtering.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ── Action definitions ────────────────────────────────────────────────────────
class TradeAction(IntEnum):
    HOLD            = 0
    LONG_ENTRY      = 1
    SHORT_ENTRY     = 2
    CLOSE_POSITION  = 3


@dataclass
class Position:
    direction:      str
    entry_price:    float = 0.0
    size:           float = 0.0
    entry_step:     int   = 0
    unrealized_pnl: float = 0.0


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


# ── Observation builder ───────────────────────────────────────────────────────
class ObservationBuilder:
    """
    28-feature state vector:
        [0-5]   Market data  : returns_1, returns_5, returns_20, volatility,
                               volume_ratio, spread
        [6-13]  ICT signals  : bullish_fvg, bearish_fvg, bullish_ob, bearish_ob,
                               market_structure, fvg_distance, ob_strength,
                               liquidity_sweep
        [14-20] Session      : session_asia, session_london, session_ny,
                               judas_swing_bull, judas_swing_bear,
                               asia_range_sweep_high, asia_range_sweep_low
        [21-24] Account      : long_flag, short_flag, unrealized_pnl_pct, drawdown
        [25-27] Temporal     : weekly_cycle, monthly_cycle, bars_in_position
    """
    OBS_DIM = 28

    @staticmethod
    def build(market_data: dict, ict_signals: dict,
              account: AccountState, current_step: int) -> np.ndarray:
        obs = np.zeros(ObservationBuilder.OBS_DIM, dtype=np.float32)

        obs[0] = market_data.get('returns',      0.0)
        obs[1] = market_data.get('returns_5',    0.0)
        obs[2] = market_data.get('returns_20',   0.0)
        obs[3] = market_data.get('volatility',   0.0)
        obs[4] = market_data.get('volume_ratio', 1.0)
        obs[5] = market_data.get('spread',       0.0)

        obs[6]  = ict_signals.get('bullish_fvg',      0.0)
        obs[7]  = ict_signals.get('bearish_fvg',      0.0)
        obs[8]  = ict_signals.get('bullish_ob',       0.0)
        obs[9]  = ict_signals.get('bearish_ob',       0.0)
        obs[10] = ict_signals.get('market_structure', 0.0)
        obs[11] = ict_signals.get('fvg_distance',     0.0)
        obs[12] = ict_signals.get('ob_strength',      0.0)
        obs[13] = ict_signals.get('liquidity_sweep',  0.0)

        obs[14] = ict_signals.get('session_asia',           0.0)
        obs[15] = ict_signals.get('session_london',         0.0)
        obs[16] = ict_signals.get('session_ny',             0.0)
        obs[17] = ict_signals.get('judas_swing_bull',       0.0)
        obs[18] = ict_signals.get('judas_swing_bear',       0.0)
        obs[19] = ict_signals.get('asia_range_sweep_high',  0.0)
        obs[20] = ict_signals.get('asia_range_sweep_low',   0.0)

        obs[21] = 1.0 if (account.position and account.position.direction == 'long')  else 0.0
        obs[22] = 1.0 if (account.position and account.position.direction == 'short') else 0.0
        obs[23] = (account.position.unrealized_pnl / (account.balance + 1e-9)
                   if account.position else 0.0)
        obs[24] = (account.peak_equity - account.equity) / (account.peak_equity + 1e-9)

        obs[25] = (current_step % 30)  / 30.0
        obs[26] = (current_step % 126) / 126.0
        bars_held = (current_step - account.position.entry_step
                     if account.position else 0)
        obs[27] = min(bars_held, 60) / 60.0

        return np.clip(obs, -10.0, 10.0)


# ── Reward calculator ─────────────────────────────────────────────────────────
class RewardCalculator:
    """
    Reward shaping tuned for swing trading (4H candles, hold days–weeks).
    """

    # Max bars a position can be held before forced close
    MAX_HOLD_BARS     = 180
    # Bars held before overstay penalty ramps up
    OVERSTAY_BARS     = 120
    OVERSTAY_RATE     = 0.0003  # penalty per bar beyond OVERSTAY_BARS
    # Bars flat before inactivity nudge
    INACTIVITY_BARS   = 30
    INACTIVITY_RATE   = 0.0005  # penalty per bar beyond INACTIVITY_BARS
    # Bonus just for completing (closing) a trade
    CLOSE_BONUS       = 0.01
    # Hard cooldown between trades (bars) — enforced in TradingEnv, not reward
    ENTRY_COOLDOWN_BARS = 30

    def __init__(
        self,
        pnl_weight:          float = 1.5,
        drawdown_penalty:    float = 1.5,
        holding_penalty:     float = 0.0,
        overtrading_penalty: float = 0.05,
        win_bonus:           float = 0.02,
        rr_bonus_weight:     float = 0.01,
    ):
        self.pnl_weight          = pnl_weight
        self.drawdown_penalty    = drawdown_penalty
        self.holding_penalty     = holding_penalty
        self.overtrading_penalty = overtrading_penalty
        self.win_bonus           = win_bonus
        self.rr_bonus_weight     = rr_bonus_weight
        self._recent_trades      = 0
        self._trade_cooldown     = 0

    def calculate(self, pnl_change: float, account: AccountState,
                  action: int, was_trade: bool,
                  bars_in_position: int = 0, bars_flat: int = 0) -> float:
        reward = 0.0

        # 1. PnL signal
        reward += self.pnl_weight * (pnl_change / max(account.balance, 1.0))

        # 2. Drawdown penalty (8% threshold — swing trades swing more)
        drawdown = (account.peak_equity - account.equity) / (account.peak_equity + 1e-9)
        if drawdown > 0.08:
            reward -= self.drawdown_penalty * drawdown

        # 3. Holding cost (zero for swing)
        if account.position is not None:
            reward -= self.holding_penalty

        # 4. Overstay penalty — ramps up after OVERSTAY_BARS
        # Prevents hold-forever exploit: agent must learn to exit
        if bars_in_position > self.OVERSTAY_BARS:
            reward -= self.OVERSTAY_RATE * (bars_in_position - self.OVERSTAY_BARS)

        # 5. Inactivity penalty — nudge after INACTIVITY_BARS flat
        # Prevents park-in-cash exploit: agent must keep engaging
        if bars_flat > self.INACTIVITY_BARS:
            reward -= self.INACTIVITY_RATE * (bars_flat - self.INACTIVITY_BARS)

        # 6. Overtrading penalty — fires only at trade time, not every step
        if was_trade:
            self._recent_trades += 1
            self._trade_cooldown = RewardCalculator.ENTRY_COOLDOWN_BARS
            if self._recent_trades > 1:
                reward -= self.overtrading_penalty * (self._recent_trades ** 2)
        elif self._trade_cooldown > 0:
            self._trade_cooldown -= 1
            if self._trade_cooldown == 0:
                self._recent_trades = max(0, self._recent_trades - 1)

        # 7. Trade completion bonus + tiered return bonuses on closes
        if was_trade and action == TradeAction.CLOSE_POSITION:
            reward += self.CLOSE_BONUS
            last_trade = account.trade_history[-1] if account.trade_history else None
            if last_trade and last_trade.get('pnl', 0) > 0:
                reward += self.win_bonus
                pnl_pct = last_trade['pnl'] / max(account.balance, 1.0)

                # Tiered return bonuses — each tier stacks on top of the previous
                # Encourages the agent to hold through strong moves instead of exiting early
                if pnl_pct > 0.01:   # > 1%  — decent winner
                    reward += 0.05
                if pnl_pct > 0.02:   # > 2%  — solid winner (old single bonus)
                    reward += self.rr_bonus_weight * (pnl_pct / 0.02)
                if pnl_pct > 0.04:   # > 4%  — strong winner
                    reward += 0.10
                if pnl_pct > 0.08:   # > 8%  — home run trade
                    reward += 0.20

        # 8. Continuous equity growth bonus
        # Rewards the agent each step that overall portfolio return is positive,
        # scaled by how large the gain is — reinforces compounding good runs
        portfolio_return = (account.equity - account.initial_balance) / (account.initial_balance + 1e-9)
        if portfolio_return > 0.05:   # > 5% up on the episode
            reward += 0.02 * portfolio_return
        if portfolio_return > 0.10:  # > 10% up — extra push to keep growing
            reward += 0.05 * portfolio_return

        return reward

    def reset(self):
        """Must be called on every episode reset to prevent state leaking."""
        self._recent_trades  = 0
        self._trade_cooldown = 0


# ── Main environment ──────────────────────────────────────────────────────────
class TradingEnv(gym.Env):
    """
    Gymnasium environment for RL-based ICT/SMC swing trading.

    Action Space (MultiDiscrete [4, 4, 4]):
        [0] Trade decision : 0=hold, 1=long, 2=short, 3=close
        [1] Position size  : 0=2%, 1=3%, 2=5%, 3=8% of equity
        [2] Setup filter   : 0=any, 1=FVG only, 2=OB only, 3=FVG+OB confluence

    Observation Space:
        Box(28,) — see ObservationBuilder

    Args:
        data_feed    : CSVDataFeed instance (or compatible data source)
        ict_detector : ICTSignalAdapter instance (or compatible detector)
        initial_balance : starting portfolio value
        max_steps    : maximum bars per episode (set to data length for eval)
        render_mode  : None | "human" | "log"
    """

    metadata = {"render_modes": ["human", "log"]}

    SIZE_MAP   = {0: 0.10, 1: 0.20, 2: 0.40, 3: 0.60}
    FILTER_MAP = {0: 'any', 1: 'fvg_only', 2: 'ob_only', 3: 'confluence'}

    def __init__(
        self,
        data_feed,
        ict_detector,
        initial_balance: float = 10_000.0,
        max_steps:       int   = 200,
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

        self.account         = self._fresh_account()
        self.reward_calc     = RewardCalculator()
        self.current_step    = 0
        self._bars_flat      = 0    # bars elapsed with no open position
        self._entry_cooldown = 0    # bars remaining before next entry is allowed
        self._current_market_data  = {}
        self._current_ict_signals  = {}

    # ── gym interface ─────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.account         = self._fresh_account()
        self.reward_calc.reset()
        self.current_step    = 0
        self._bars_flat      = 0
        self._entry_cooldown = 0
        self.data_feed.reset(seed=seed)
        self._update_market_state()
        return ObservationBuilder.build(
            self._current_market_data,
            self._current_ict_signals,
            self.account,
            self.current_step,
        ), self._get_info()

    def step(self, action):
        trade_decision = TradeAction(action[0])
        size_pct       = self.SIZE_MAP[action[1]]
        setup_filter   = self.FILTER_MAP[action[2]]

        prev_equity = self.account.equity
        was_trade   = False

        # Apply setup filter (may downgrade entry to hold)
        if not self._passes_filter(trade_decision, setup_filter):
            trade_decision = TradeAction.HOLD

        # Hard cooldown — block new entries for ENTRY_COOLDOWN_BARS after close
        if self._entry_cooldown > 0:
            self._entry_cooldown -= 1
            if trade_decision in (TradeAction.LONG_ENTRY, TradeAction.SHORT_ENTRY):
                trade_decision = TradeAction.HOLD

        # Execute trade
        # SHORT_ENTRY disabled — QQQ is a long-biased asset, shorting hurts win rate
        if trade_decision == TradeAction.LONG_ENTRY and self.account.position is None:
            self._open_position('long', size_pct)
            was_trade = True
        elif trade_decision == TradeAction.SHORT_ENTRY:
            trade_decision = TradeAction.HOLD  # treat as hold
        elif trade_decision == TradeAction.CLOSE_POSITION and self.account.position is not None:
            self._close_position()
            self._entry_cooldown = RewardCalculator.ENTRY_COOLDOWN_BARS
            was_trade = True

        # Force-close if held too long — prevents hold-forever exploit
        if self.account.position is not None:
            bars_held = self.current_step - self.account.position.entry_step
            if bars_held >= RewardCalculator.MAX_HOLD_BARS:
                self._close_position()
                self._entry_cooldown = RewardCalculator.ENTRY_COOLDOWN_BARS
                was_trade = True
                action = (TradeAction.CLOSE_POSITION, action[1], action[2])

        # Track bars with no position (for inactivity penalty)
        # Don't count cooldown bars — agent physically can't trade during cooldown
        if self.account.position is None and self._entry_cooldown == 0:
            self._bars_flat += 1
        else:
            self._bars_flat = 0

        # Capture position duration before advancing (used in reward)
        bars_in_position = (
            self.current_step - self.account.position.entry_step
            if self.account.position else 0
        )

        # Advance market — always safe: data_feed.step() is clamped
        self.current_step += 1
        self.data_feed.step()
        self._update_market_state()
        self._update_unrealized_pnl()

        # Reward
        pnl_change = self.account.equity - prev_equity
        reward = self.reward_calc.calculate(
            pnl_change, self.account, action[0], was_trade,
            bars_in_position=bars_in_position,
            bars_flat=self._bars_flat,
        )

        # Termination
        terminated = self.account.equity <= self.initial_balance * 0.50
        truncated  = (self.current_step >= self.max_steps or self.data_feed.is_done)

        if terminated:
            reward -= 1.0

        obs = ObservationBuilder.build(
            self._current_market_data,
            self._current_ict_signals,
            self.account,
            self.current_step,
        )
        return obs, reward, terminated, truncated, self._get_info()

    # ── trade helpers ─────────────────────────────────────────────────────────
    def _open_position(self, direction: str, size_pct: float):
        price = self._current_market_data['close']
        size  = (self.account.equity * size_pct) / (price + 1e-9)
        self.account.position = Position(
            direction=direction, entry_price=price,
            size=size, entry_step=self.current_step,
        )
        self.account.total_trades += 1
        logger.debug(f"OPEN {direction} @ {price:.2f}  size={size:.4f}")

    def _close_position(self):
        pos   = self.account.position
        price = self._current_market_data['close']
        pnl   = ((price - pos.entry_price) if pos.direction == 'long'
                 else (pos.entry_price - price)) * pos.size

        self.account.balance    += pnl
        self.account.equity      = self.account.balance
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
        logger.debug(f"CLOSE {pos.direction} @ {price:.2f}  PnL={pnl:.2f}")
        self.account.position = None

    def _update_unrealized_pnl(self):
        if self.account.position is None:
            self.account.equity = self.account.balance
            return
        pos   = self.account.position
        price = self._current_market_data['close']
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
        if setup_filter == 'any':        return True
        if setup_filter == 'fvg_only':   return bool(has_fvg)
        if setup_filter == 'ob_only':    return bool(has_ob)
        if setup_filter == 'confluence': return bool(has_fvg and has_ob)
        return True

    def _update_market_state(self):
        self._current_market_data = self.data_feed.get_current_bar()
        self._current_ict_signals = self.ict_detector.get_signals(
            self.data_feed.get_history()
        )

    # ── helpers ───────────────────────────────────────────────────────────────
    def _fresh_account(self) -> AccountState:
        return AccountState(
            balance=self.initial_balance,
            initial_balance=self.initial_balance,
            equity=self.initial_balance,
            peak_equity=self.initial_balance,
        )

    def _get_info(self) -> dict:
        win_rate = self.account.winning_trades / max(self.account.total_trades, 1)
        return {
            'equity':       self.account.equity,
            'balance':      self.account.balance,
            'total_trades': self.account.total_trades,
            'win_rate':     win_rate,
            'drawdown':     (self.account.peak_equity - self.account.equity) / (self.account.peak_equity + 1e-9),
            'step':         self.current_step,
            'has_position': self.account.position is not None,
        }

    def render(self):
        pv  = self.account.equity
        ret = (pv / self.initial_balance - 1) * 100
        print(f"  step={self.current_step:4d}  "
              f"price={self._current_market_data.get('close', 0):.2f}  "
              f"equity=${pv:,.0f} ({ret:+.1f}%)  "
              f"trades={self.account.total_trades}")
