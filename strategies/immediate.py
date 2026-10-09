"""
Immediate Execution Strategy (Block / Fast Execution).
Executes 100% of order volume in the first interval to minimize market risk at the expense of impact.
"""

import numpy as np
from .base import BaseExecutionStrategy
from config import OrderConfig, MarketParameters, ExecutionSchedule


class ImmediateExecutionStrategy(BaseExecutionStrategy):
    """
    Immediate Execution strategy: executes all shares in interval 1.
    Eliminates inventory holding risk, but generates maximum market impact.
    """
    def __init__(self):
        super().__init__(name="Immediate Execution")

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
        trade_sizes = np.zeros(N)
        trade_sizes[0] = X

        inventory = np.zeros(N + 1)
        inventory[0] = X
        # Remaining inventory for t=1..N is 0

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
            metadata={"type": "Benchmark", "urgency": "Maximum"}
        )
