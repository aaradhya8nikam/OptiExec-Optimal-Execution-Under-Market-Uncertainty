"""
Base Execution Strategy abstract class.
Provides shared evaluation methods for expected cost, cost variance, and schedule packaging.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from config import OrderConfig, MarketParameters, ExecutionSchedule


class BaseExecutionStrategy(ABC):
    """
    Abstract Base Class for institutional execution algorithms.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_schedule(
        self,
        order: OrderConfig,
        market_params: MarketParameters
    ) -> ExecutionSchedule:
        """
        Generates the discrete execution schedule [x_0, x_1, ... x_N] and trade sizes [n_1, ... n_N].
        """
        pass

    def compute_theoretical_metrics(
        self,
        trade_sizes: np.ndarray,
        inventory: np.ndarray,
        order: OrderConfig,
        market_params: MarketParameters
    ) -> tuple[float, float, float]:
        """
        Computes the theoretical Almgren-Chriss (2000) Expected Cost E[x] and Variance V[x].
        
        Expected Cost E[x]:
          E[x] = 0.5 * gamma * X^2 + (eta / tau) * sum(n_k^2) + half_spread * X
        
        Cost Variance V[x]:
          V[x] = sigma^2 * tau * sum(x_k^2)
        
        Returns:
        --------
        (expected_cost, expected_variance, utility)
        """
        X = order.total_quantity
        tau = order.interval_duration_min
        eta = market_params.temp_impact_coef
        gamma = market_params.perm_impact_coef
        sigma_abs = order.start_price * market_params.interval_volatility
        half_spread_cost = order.start_price * (market_params.half_spread_bps * 1e-4) * X

        # Expected transaction cost
        perm_cost = 0.5 * gamma * (X ** 2)
        temp_cost = (eta / max(tau, 1e-4)) * np.sum(trade_sizes ** 2)
        expected_cost = float(perm_cost + temp_cost + half_spread_cost)

        # Variance of shortfall: V[x] = sigma_tau^2 * sum_{k=1}^N x_k^2
        # Note: inventory is [x_0, x_1, ... x_N], so x_1..x_N are the unexecuted amounts after each trade
        x_post_trades = inventory[1:]
        expected_variance = float((sigma_abs ** 2) * np.sum(x_post_trades ** 2))

        # Expected utility (objective function value)
        utility = float(expected_cost + order.risk_aversion * expected_variance)

        return expected_cost, expected_variance, utility
