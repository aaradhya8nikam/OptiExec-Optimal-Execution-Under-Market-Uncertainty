"""
Time-Weighted Average Price (TWAP) Strategy.
Splits the total order uniformly across time intervals: n_k = X / N.
Mathematically corresponds to the risk-neutral Almgren-Chriss baseline (lambda = 0).
"""

import numpy as np
from .base import BaseExecutionStrategy
from config import OrderConfig, MarketParameters, ExecutionSchedule


class TWAPExecutionStrategy(BaseExecutionStrategy):
    """
    Time-Weighted Average Price (TWAP) execution strategy.
    Executes equal quantities of shares per interval across the entire execution window.
    """
    def __init__(self):
        super().__init__(name="TWAP")

    def generate_schedule(
        self,
        order: OrderConfig,
        market_params: MarketParameters
    ) -> ExecutionSchedule:
        order.validate()
        N = order.num_intervals
        tau = order.interval_duration_min
        X = order.total_quantity

        time_steps = np.array([k * tau for k in range(N + 1)])
        slice_size = X / N
        trade_sizes = np.full(N, slice_size)

        # Inventory decreases linearly: x_k = X - k * (X / N)
        k_indices = np.arange(N + 1)
        inventory = X * (1.0 - k_indices / N)
        inventory[-1] = 0.0  # Exact terminal boundary

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
            metadata={"type": "Benchmark", "profile": "Uniform Time Distribution"}
        )
