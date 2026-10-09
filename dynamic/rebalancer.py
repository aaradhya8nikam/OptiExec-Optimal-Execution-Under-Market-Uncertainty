"""
Dynamic Adaptive Re-Optimization Engine.
Dynamically re-solves optimal execution trajectory upon intraday market regime shifts (volatility spike, liquidity shock).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from config import OrderConfig, MarketParameters, ExecutionSchedule, OrderSide
from strategies.almgren_chriss import AlmgrenChrissExecutionStrategy


@dataclass
class DynamicExecutionState:
    """Current live execution state at interval k."""
    current_interval: int
    elapsed_time_min: float
    executed_shares: float
    remaining_shares: float
    current_price: float
    updated_annual_vol: float
    updated_adv: float
    updated_temp_impact_coef: float


@dataclass
class DynamicRebalancingResult:
    """Comparison between the original static schedule and the dynamically re-optimized schedule."""
    original_schedule: ExecutionSchedule
    rebalanced_schedule: ExecutionSchedule
    rebalance_step: int
    rebalance_time_min: float
    volatility_shock_pct: float
    liquidity_shock_pct: float
    old_kappa: float
    new_kappa: float
    urgency_shift: str  # "ACCELERATE", "DECELERATE", "UNCHANGED"
    combined_trajectory: np.ndarray  # Spliced inventory path (actual executed + new future)
    combined_trades: np.ndarray       # Spliced trade slices


class DynamicExecutionEngine:
    """
    Manages mid-horizon re-optimization for adaptive algorithmic execution.
    """
    def __init__(self):
        self.ac_strategy = AlmgrenChrissExecutionStrategy()

    def simulate_dynamic_shock(
        self,
        order: OrderConfig,
        initial_params: MarketParameters,
        shock_step: int = 4,
        volatility_multiplier: float = 1.8,   # +80% volatility surge
        liquidity_multiplier: float = 0.6     # -40% liquidity drop
    ) -> DynamicRebalancingResult:
        """
        Simulates an intraday execution where market conditions abruptly shift at `shock_step`.
        Re-calculates the optimal execution path for the remaining inventory.
        """
        N = order.num_intervals
        tau = order.interval_duration_min
        T = order.time_horizon_min
        X = order.total_quantity

        if not (1 <= shock_step < N):
            shock_step = max(1, N // 2)

        # 1. Baseline initial schedule
        original_sched = self.ac_strategy.generate_schedule(order, initial_params)
        old_kappa = original_sched.metadata.get("kappa", 0.0)

        # 2. Inventory executed up to shock_step
        trades_done = original_sched.trade_sizes[:shock_step]
        shares_done = float(np.sum(trades_done))
        remaining_shares = max(X - shares_done, 0.0)

        remaining_intervals = N - shock_step
        remaining_time_min = remaining_intervals * tau

        # 3. Create shocked market parameters
        shocked_annual_vol = initial_params.annual_volatility * volatility_multiplier
        shocked_interval_vol = initial_params.interval_volatility * volatility_multiplier
        shocked_adv = initial_params.adv * liquidity_multiplier
        # Market impact eta scales inversely with liquidity
        shocked_eta = initial_params.temp_impact_coef / max(liquidity_multiplier, 0.1)

        shocked_params = MarketParameters(
            symbol=initial_params.symbol,
            annual_volatility=shocked_annual_vol,
            daily_volatility=initial_params.daily_volatility * volatility_multiplier,
            interval_volatility=shocked_interval_vol,
            adv=shocked_adv,
            half_spread_bps=initial_params.half_spread_bps * (1.0 / max(liquidity_multiplier, 0.5)),
            temp_impact_coef=shocked_eta,
            perm_impact_coef=initial_params.perm_impact_coef,
            intraday_volume_profile=None
        )

        # 4. Re-optimize for remaining horizon
        residual_order = OrderConfig(
            symbol=order.symbol,
            side=order.side,
            total_quantity=remaining_shares,
            start_price=order.start_price,
            time_horizon_min=remaining_time_min,
            num_intervals=remaining_intervals,
            risk_aversion=order.risk_aversion
        )

        rebalanced_sched = self.ac_strategy.generate_schedule(residual_order, shocked_params)
        new_kappa = rebalanced_sched.metadata.get("kappa", 0.0)

        # 5. Determine urgency shift
        if new_kappa > old_kappa * 1.05:
            urgency_shift = "ACCELERATE (High Volatility Dominates -> Trade Faster)"
        elif new_kappa < old_kappa * 0.95:
            urgency_shift = "DECELERATE (Impact Cost Dominates -> Trade Slower)"
        else:
            urgency_shift = "UNCHANGED (Balanced Shocks)"

        # 6. Build seamless combined trajectory
        # Combined trades: first `shock_step` original trades + remaining rebalanced trades
        combined_trades = np.concatenate([trades_done, rebalanced_sched.trade_sizes])
        combined_inventory = np.zeros(N + 1)
        combined_inventory[0] = X
        combined_inventory[1:] = np.maximum(X - np.cumsum(combined_trades), 0.0)
        combined_inventory[-1] = 0.0

        return DynamicRebalancingResult(
            original_schedule=original_sched,
            rebalanced_schedule=rebalanced_sched,
            rebalance_step=shock_step,
            rebalance_time_min=shock_step * tau,
            volatility_shock_pct=(volatility_multiplier - 1.0) * 100.0,
            liquidity_shock_pct=(liquidity_multiplier - 1.0) * 100.0,
            old_kappa=old_kappa,
            new_kappa=new_kappa,
            urgency_shift=urgency_shift,
            combined_trajectory=combined_inventory,
            combined_trades=combined_trades
        )
