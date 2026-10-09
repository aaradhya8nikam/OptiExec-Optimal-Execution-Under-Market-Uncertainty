"""
Discrete Path Execution Simulator.
Tracks step-by-step price evolution, temporary impact, permanent impact drift, cash flows, and shortfall.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from config import OrderConfig, MarketParameters, ExecutionSchedule, OrderSide
from models.market_impact import MarketImpactModel


@dataclass
class IntervalExecutionRecord:
    """Detailed record for a single execution interval."""
    interval_idx: int
    time_min: float
    shares_traded: float
    inventory_remaining: float
    market_price_pre: float
    execution_price: float
    temporary_impact: float
    permanent_impact: float
    cash_flow: float
    interval_shortfall: float


@dataclass
class SimulatedExecutionResult:
    """Aggregate result for executing a strategy on a specific price path."""
    strategy_name: str
    records: List[IntervalExecutionRecord]
    total_shares: float
    decision_price: float
    average_execution_price: float
    implementation_shortfall_total: float
    implementation_shortfall_bps: float
    total_temporary_cost: float
    total_permanent_cost: float
    final_market_price: float


class ExecutionSimulator:
    """
    Simulates the exact trade execution of an institutional order schedule
    along a realized market price trajectory.
    """
    def __init__(self, impact_model: MarketImpactModel):
        self.impact_model = impact_model

    def execute_path(
        self,
        schedule: ExecutionSchedule,
        price_path: np.ndarray,
        order: OrderConfig,
        market_params: MarketParameters
    ) -> SimulatedExecutionResult:
        """
        Executes the schedule against a single realized price path.
        price_path shape: (num_intervals + 1,) representing [S_0, S_1, ... S_N]
        """
        side_sign = 1.0 if order.side == OrderSide.BUY else -1.0
        N = order.num_intervals
        tau = order.interval_duration_min
        X = order.total_quantity
        S0 = order.start_price

        records: List[IntervalExecutionRecord] = []
        current_market_price = price_path[0]
        total_cash_flow = 0.0
        total_temp_cost = 0.0
        total_perm_cost = 0.0

        for k in range(N):
            n_k = schedule.trade_sizes[k]
            x_rem = schedule.inventory_remaining[k + 1]
            time_min = (k + 1) * tau

            if n_k > 0:
                trade_rate = n_k / max(tau, 1e-4)
                temp_impact = self.impact_model.temporary_impact(trade_rate, current_market_price)
                perm_impact = self.impact_model.permanent_impact(n_k)

                # Execution price paid for BUY (+), received for SELL (-)
                exec_price = current_market_price + side_sign * temp_impact
                cash_interval = n_k * exec_price
                
                temp_cost_interval = n_k * temp_impact
                perm_cost_interval = n_k * perm_impact

                total_cash_flow += cash_interval
                total_temp_cost += temp_cost_interval
                total_perm_cost += perm_cost_interval
            else:
                exec_price = current_market_price
                temp_impact = 0.0
                perm_impact = 0.0
                cash_interval = 0.0

            # Shortfall relative to decision price S0 for this interval's trade
            if order.side == OrderSide.BUY:
                interval_is = n_k * (exec_price - S0)
            else:
                interval_is = n_k * (S0 - exec_price)

            records.append(
                IntervalExecutionRecord(
                    interval_idx=k + 1,
                    time_min=time_min,
                    shares_traded=n_k,
                    inventory_remaining=x_rem,
                    market_price_pre=current_market_price,
                    execution_price=exec_price,
                    temporary_impact=temp_impact,
                    permanent_impact=perm_impact,
                    cash_flow=cash_interval,
                    interval_shortfall=interval_is
                )
            )

            # Advance market price: exogenous movement + endogenous permanent impact
            exogenous_drift = price_path[k + 1] - price_path[k]
            current_market_price = current_market_price + exogenous_drift + side_sign * perm_impact

        # Total metrics
        avg_exec_price = total_cash_flow / max(X, 1e-9)
        if order.side == OrderSide.BUY:
            total_is = total_cash_flow - X * S0
        else:
            total_is = X * S0 - total_cash_flow

        is_bps = (total_is / (X * S0)) * 10_000.0

        return SimulatedExecutionResult(
            strategy_name=schedule.strategy_name,
            records=records,
            total_shares=X,
            decision_price=S0,
            average_execution_price=avg_exec_price,
            implementation_shortfall_total=total_is,
            implementation_shortfall_bps=is_bps,
            total_temporary_cost=total_temp_cost,
            total_permanent_cost=total_perm_cost,
            final_market_price=current_market_price
        )
