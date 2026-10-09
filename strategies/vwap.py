"""
Volume-Weighted Average Price (VWAP) Strategy.
Allocates trade slices proportionally to expected intraday market volume: n_k = X * w_k.
"""

import numpy as np
from .base import BaseExecutionStrategy
from config import OrderConfig, MarketParameters, ExecutionSchedule
from data.market_data import generate_u_shape_volume_profile


class VWAPExecutionStrategy(BaseExecutionStrategy):
    """
    Volume-Weighted Average Price (VWAP) execution strategy.
    Matches the historical/forecasted intraday volume curve to maximize liquidity capture.
    """
    def __init__(self):
        super().__init__(name="VWAP")

    def generate_schedule(
        self,
        order: OrderConfig,
        market_params: MarketParameters
    ) -> ExecutionSchedule:
        order.validate()
        N = order.num_intervals
        tau = order.interval_duration_min
        X = order.total_quantity

        # Use provided intraday volume profile or default U-curve
        if market_params.intraday_volume_profile is not None and len(market_params.intraday_volume_profile) == N:
            weights = market_params.intraday_volume_profile
        else:
            weights = generate_u_shape_volume_profile(N)

        weights = weights / np.sum(weights)

        time_steps = np.array([k * tau for k in range(N + 1)])
        trade_sizes = X * weights

        # Cumulative inventory trajectory
        cum_executed = np.cumsum(trade_sizes)
        inventory = np.zeros(N + 1)
        inventory[0] = X
        inventory[1:] = np.maximum(X - cum_executed, 0.0)
        inventory[-1] = 0.0

        trading_rates = trade_sizes / tau

        exp_cost, exp_var, utility = self.compute_theoretical_metrics(
            trade_sizes, inventory, order, market_params
        )

        return ExecutionSchedule(
            strategy_name=self.name,
            time_steps_min=time_steps,
            trade_sizes=trade_sizes,
            inventory_remaining=inventory,
            trading_rates=trading_rates,
            expected_cost=exp_cost,
            expected_variance=exp_var,
            expected_std_cost=np.sqrt(exp_var),
            utility=utility,
            metadata={"type": "Benchmark", "weights": weights.tolist()}
        )
