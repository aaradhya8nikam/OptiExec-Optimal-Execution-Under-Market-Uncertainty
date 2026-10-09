"""
Risk-Cost Efficient Frontier Generator.
Maps the continuum of optimal execution strategies across risk aversion parameter lambda (λ).
Plots: Expected Cost E[x] versus Execution Risk Std(x) = sqrt(V[x]).
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
from config import OrderConfig, MarketParameters, ExecutionSchedule
from strategies.almgren_chriss import AlmgrenChrissExecutionStrategy
from strategies.twap import TWAPExecutionStrategy
from strategies.vwap import VWAPExecutionStrategy
from strategies.immediate import ImmediateExecutionStrategy


@dataclass
class EfficientFrontierPoint:
    """A single point along the Almgren-Chriss Efficient Frontier."""
    risk_aversion: float
    kappa: float
    expected_cost: float
    expected_cost_bps: float
    expected_variance: float
    expected_std_cost: float
    expected_std_bps: float
    utility: float
    schedule: ExecutionSchedule


@dataclass
class EfficientFrontierResult:
    """The collection of frontier points and benchmark positions."""
    frontier_points: List[EfficientFrontierPoint]
    benchmark_points: Dict[str, Dict[str, float]]
    user_point: EfficientFrontierPoint


class EfficientFrontierEngine:
    """
    Computes the Almgren-Chriss Efficient Frontier curve across risk-aversion values.
    """
    def __init__(self, num_points: int = 50):
        self.num_points = num_points

    def generate_frontier(
        self,
        order: OrderConfig,
        market_params: MarketParameters,
        min_lambda: float = 1e-9,
        max_lambda: float = 1e-3
    ) -> EfficientFrontierResult:
        """
        Calculates optimal schedules for log-spaced lambda values and benchmarks.
        """
        lambdas = np.logspace(np.log10(min_lambda), np.log10(max_lambda), self.num_points)
        notional = order.total_quantity * order.start_price

        ac_strategy = AlmgrenChrissExecutionStrategy()
        points: List[EfficientFrontierPoint] = []

        for lam in lambdas:
            test_order = OrderConfig(
                symbol=order.symbol,
                side=order.side,
                total_quantity=order.total_quantity,
                start_price=order.start_price,
                time_horizon_min=order.time_horizon_min,
                num_intervals=order.num_intervals,
                risk_aversion=lam
            )
            sched = ac_strategy.generate_schedule(test_order, market_params)
            cost_bps = (sched.expected_cost / notional) * 10_000.0
            std_bps = (sched.expected_std_cost / notional) * 10_000.0

            points.append(
                EfficientFrontierPoint(
                    risk_aversion=lam,
                    kappa=sched.metadata.get("kappa", 0.0),
                    expected_cost=sched.expected_cost,
                    expected_cost_bps=cost_bps,
                    expected_variance=sched.expected_variance,
                    expected_std_cost=sched.expected_std_cost,
                    expected_std_bps=std_bps,
                    utility=sched.utility,
                    schedule=sched
                )
            )

        # Benchmark positions
        benchmarks = {}
        for strat in [ImmediateExecutionStrategy(), TWAPExecutionStrategy(), VWAPExecutionStrategy()]:
            bench_sched = strat.generate_schedule(order, market_params)
            benchmarks[strat.name] = {
                "expected_cost": bench_sched.expected_cost,
                "expected_cost_bps": (bench_sched.expected_cost / notional) * 10_000.0,
                "expected_std_cost": bench_sched.expected_std_cost,
                "expected_std_bps": (bench_sched.expected_std_cost / notional) * 10_000.0,
            }

        # User's current lambda point
        user_sched = ac_strategy.generate_schedule(order, market_params)
        user_point = EfficientFrontierPoint(
            risk_aversion=order.risk_aversion,
            kappa=user_sched.metadata.get("kappa", 0.0),
            expected_cost=user_sched.expected_cost,
            expected_cost_bps=(user_sched.expected_cost / notional) * 10_000.0,
            expected_variance=user_sched.expected_variance,
            expected_std_cost=user_sched.expected_std_cost,
            expected_std_bps=(user_sched.expected_std_cost / notional) * 10_000.0,
            utility=user_sched.utility,
            schedule=user_sched
        )

        return EfficientFrontierResult(
            frontier_points=points,
            benchmark_points=benchmarks,
            user_point=user_point
        )
