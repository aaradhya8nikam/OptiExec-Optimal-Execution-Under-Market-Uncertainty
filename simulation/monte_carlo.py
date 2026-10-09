"""
Vectorized Monte Carlo Simulation Engine.
Evaluates multiple execution strategies across thousands of stochastic market paths.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from config import OrderConfig, MarketParameters, ExecutionSchedule, SimulationConfig, OrderSide
from models.market_impact import MarketImpactModel
from models.price_process import PriceProcessSimulator


@dataclass
class StrategySimulationSummary:
    """Statistical summary of Monte Carlo simulation for a specific strategy."""
    strategy_name: str
    num_paths: int
    mean_shortfall_total: float
    median_shortfall_total: float
    std_shortfall_total: float
    mean_shortfall_bps: float
    median_shortfall_bps: float
    std_shortfall_bps: float
    var_95_bps: float           # 95% Value at Risk (95th percentile cost)
    var_99_bps: float           # 99% Value at Risk
    cvar_95_bps: float          # Expected Shortfall (tail mean above 95th percentile)
    worst_case_bps: float       # Maximum simulated cost
    best_case_bps: float        # Minimum simulated cost
    mean_execution_price: float
    std_execution_price: float
    all_shortfalls_bps: np.ndarray
    all_avg_prices: np.ndarray


@dataclass
class MonteCarloSimulationResult:
    """Complete collection of Monte Carlo results across all evaluated strategies."""
    order: OrderConfig
    market_params: MarketParameters
    sim_config: SimulationConfig
    exogenous_price_paths: np.ndarray  # Shape: (num_paths, num_intervals + 1)
    strategy_summaries: Dict[str, StrategySimulationSummary]


class MonteCarloEngine:
    """
    Simulates execution trajectories across thousands of stochastic price paths.
    Uses vectorized NumPy operations for speed.
    """
    def __init__(self, impact_model: Optional[MarketImpactModel] = None):
        self.impact_model = impact_model

    def run_simulations(
        self,
        order: OrderConfig,
        market_params: MarketParameters,
        schedules: List[ExecutionSchedule],
        sim_config: Optional[SimulationConfig] = None
    ) -> MonteCarloSimulationResult:
        """
        Executes all provided schedules across identical Monte Carlo price paths.
        """
        if sim_config is None:
            sim_config = SimulationConfig()

        order.validate()
        impact = self.impact_model or MarketImpactModel(
            eta=market_params.temp_impact_coef,
            gamma=market_params.perm_impact_coef,
            half_spread_bps=market_params.half_spread_bps
        )

        # 1. Generate exogenous baseline price paths
        price_sim = PriceProcessSimulator(
            start_price=order.start_price,
            interval_vol=market_params.interval_volatility,
            num_intervals=order.num_intervals,
            model_type=sim_config.price_model
        )
        base_paths = price_sim.generate_exogenous_paths(
            num_paths=sim_config.num_paths,
            seed=sim_config.random_seed
        )

        summaries: Dict[str, StrategySimulationSummary] = {}
        for schedule in schedules:
            summary = self._simulate_strategy_vectorized(
                schedule=schedule,
                base_paths=base_paths,
                order=order,
                market_params=market_params,
                impact=impact
            )
            summaries[schedule.strategy_name] = summary

        return MonteCarloSimulationResult(
            order=order,
            market_params=market_params,
            sim_config=sim_config,
            exogenous_price_paths=base_paths,
            strategy_summaries=summaries
        )

    def _simulate_strategy_vectorized(
        self,
        schedule: ExecutionSchedule,
        base_paths: np.ndarray,
        order: OrderConfig,
        market_params: MarketParameters,
        impact: MarketImpactModel
    ) -> StrategySimulationSummary:
        """
        Simulates one strategy across all price paths simultaneously.
        """
        M, N_plus_1 = base_paths.shape
        N = order.num_intervals
        tau = order.interval_duration_min
        X = order.total_quantity
        S0 = order.start_price
        side_sign = 1.0 if order.side == OrderSide.BUY else -1.0

        trade_sizes = schedule.trade_sizes  # shape (N,)
        # Precompute temporary impacts for each interval: shape (N,)
        trade_rates = trade_sizes / max(tau, 1e-4)
        
        half_spread_abs = S0 * (market_params.half_spread_bps * 1e-4)
        temp_impacts = np.where(
            trade_sizes > 0,
            half_spread_abs + impact.eta * (trade_rates ** impact.power_alpha),
            0.0
        )
        perm_impacts = impact.gamma * trade_sizes

        # Track market price path with endogenous cumulative permanent impact
        # At interval k, cumulative permanent impact before trade is sum(perm_impacts[:k])
        cum_perm_impact = np.zeros(N)
        if N > 1:
            cum_perm_impact[1:] = np.cumsum(perm_impacts[:-1])

        # Effective market price for each interval across all paths:
        # Pre-trade market price at interval k: base_paths[:, k] + side_sign * cum_perm_impact[k]
        # Execution price: pre_trade_price + side_sign * temp_impacts[k]
        # shape of pre_trade_prices: (M, N)
        pre_trade_prices = base_paths[:, :N] + side_sign * cum_perm_impact[np.newaxis, :]
        exec_prices = pre_trade_prices + side_sign * temp_impacts[np.newaxis, :]

        # Total cash flows across all paths: sum_k (n_k * exec_price_k)
        # shape: (M,)
        total_cash_flows = np.sum(exec_prices * trade_sizes[np.newaxis, :], axis=1)
        avg_prices = total_cash_flows / X

        # Implementation Shortfall in dollars
        if order.side == OrderSide.BUY:
            shortfall_dollars = total_cash_flows - X * S0
        else:
            shortfall_dollars = X * S0 - total_cash_flows

        # Implementation Shortfall in basis points (1 bp = 0.01% of notional)
        shortfall_bps = (shortfall_dollars / (X * S0)) * 10_000.0

        # Percentiles & Tail Risk
        var_95 = float(np.percentile(shortfall_bps, 95.0))
        var_99 = float(np.percentile(shortfall_bps, 99.0))
        tail_95_mask = shortfall_bps >= var_95
        cvar_95 = float(np.mean(shortfall_bps[tail_95_mask])) if np.any(tail_95_mask) else var_95

        return StrategySimulationSummary(
            strategy_name=schedule.strategy_name,
            num_paths=M,
            mean_shortfall_total=float(np.mean(shortfall_dollars)),
            median_shortfall_total=float(np.median(shortfall_dollars)),
            std_shortfall_total=float(np.std(shortfall_dollars, ddof=1)),
            mean_shortfall_bps=float(np.mean(shortfall_bps)),
            median_shortfall_bps=float(np.median(shortfall_bps)),
            std_shortfall_bps=float(np.std(shortfall_bps, ddof=1)),
            var_95_bps=var_95,
            var_99_bps=var_99,
            cvar_95_bps=cvar_95,
            worst_case_bps=float(np.max(shortfall_bps)),
            best_case_bps=float(np.min(shortfall_bps)),
            mean_execution_price=float(np.mean(avg_prices)),
            std_execution_price=float(np.std(avg_prices, ddof=1)),
            all_shortfalls_bps=shortfall_bps,
            all_avg_prices=avg_prices
        )
