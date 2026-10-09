"""
Almgren-Chriss (2000) Optimal Execution Strategy.
Implements the exact discrete analytical hyperbolic solution and numerical QP solver.
Minimizes: Objective U(x) = E[x] + lambda * V[x]
"""

from typing import Optional
import numpy as np
from scipy.optimize import minimize
from .base import BaseExecutionStrategy
from config import OrderConfig, MarketParameters, ExecutionSchedule


class AlmgrenChrissExecutionStrategy(BaseExecutionStrategy):
    """
    Optimal Execution under the Almgren-Chriss (2000) framework.
    Finds the optimal inventory trajectory balancing market impact cost vs timing volatility risk.
    """
    def __init__(self, use_numerical_solver: bool = False):
        super().__init__(name="Almgren-Chriss (Optimal)")
        self.use_numerical_solver = use_numerical_solver

    def calculate_kappa(
        self,
        order: OrderConfig,
        market_params: MarketParameters
    ) -> float:
        """
        Calculates the urgency parameter kappa (κ):
        cosh(kappa * tau) = 1 + (lambda * sigma_abs^2 * tau) / (2 * eta)
        """
        tau = order.interval_duration_min
        eta = market_params.temp_impact_coef
        sigma_abs = order.start_price * market_params.interval_volatility
        lam = order.risk_aversion

        if lam <= 1e-15 or eta <= 0:
            return 0.0

        cosh_val = 1.0 + (lam * (sigma_abs ** 2) * tau) / (2.0 * eta)
        # Numerical protection for arccosh
        cosh_val = max(cosh_val, 1.0)
        
        # kappa * tau = arccosh(cosh_val)
        kappa_tau = np.arccosh(cosh_val)
        kappa = kappa_tau / tau
        return float(kappa)

    def generate_schedule(
        self,
        order: OrderConfig,
        market_params: MarketParameters
    ) -> ExecutionSchedule:
        order.validate()
        N = order.num_intervals
        tau = order.interval_duration_min
        T = order.time_horizon_min
        X = order.total_quantity

        if self.use_numerical_solver:
            inventory, trade_sizes = self._solve_numerical(order, market_params)
        else:
            inventory, trade_sizes = self._solve_analytical(order, market_params)

        time_steps = np.array([k * tau for k in range(N + 1)])
        trading_rates = trade_sizes / tau

        exp_cost, exp_var, utility = self.compute_theoretical_metrics(
            trade_sizes, inventory, order, market_params
        )

        kappa = self.calculate_kappa(order, market_params)
        half_life = np.log(2.0) / kappa if kappa > 1e-8 else float("inf")

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
            metadata={
                "type": "Optimal",
                "kappa": kappa,
                "half_life_min": half_life,
                "solver": "Numerical QP" if self.use_numerical_solver else "Analytical Hyperbolic"
            }
        )

    def _solve_analytical(
        self,
        order: OrderConfig,
        market_params: MarketParameters
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Analytical solution: x_j = sinh(kappa * (T - t_j)) / sinh(kappa * T) * X
        """
        N = order.num_intervals
        tau = order.interval_duration_min
        T = order.time_horizon_min
        X = order.total_quantity

        kappa = self.calculate_kappa(order, market_params)

        # If risk-neutral (lambda ~ 0), linear TWAP is the exact limit
        if kappa < 1e-7:
            k_indices = np.arange(N + 1)
            inventory = X * (1.0 - k_indices / N)
            inventory[-1] = 0.0
            trade_sizes = np.full(N, X / N)
            return inventory, trade_sizes

        # General hyperbolic trajectory
        t_j = np.array([j * tau for j in range(N + 1)])
        
        # Prevent numerical overflow for high kappa * T
        if kappa * T > 50.0:
            # Exponential approximation for high urgency
            inventory = X * np.exp(-kappa * t_j)
            inventory[-1] = 0.0
        else:
            sinh_total = np.sinh(kappa * T)
            inventory = X * (np.sinh(kappa * (T - t_j)) / sinh_total)
            inventory[-1] = 0.0

        # Compute interval trade sizes: n_j = x_{j-1} - x_j
        trade_sizes = inventory[:-1] - inventory[1:]
        # Ensure exact sum and non-negativity
        trade_sizes = np.maximum(trade_sizes, 0.0)
        sum_trades = np.sum(trade_sizes)
        if sum_trades > 0:
            trade_sizes = trade_sizes * (X / sum_trades)

        # Re-derive inventory from validated trade sizes for strict consistency
        inventory[0] = X
        inventory[1:] = np.maximum(X - np.cumsum(trade_sizes), 0.0)
        inventory[-1] = 0.0

        return inventory, trade_sizes

    def _solve_numerical(
        self,
        order: OrderConfig,
        market_params: MarketParameters
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Quadratic Programming solver for general/constrained optimal execution.
        Minimizes: sum( (eta/tau) * n_k^2 ) + lambda * sigma^2 * sum( x_k^2 )
        subject to: sum(n_k) = X, n_k >= 0
        """
        N = order.num_intervals
        tau = order.interval_duration_min
        X = order.total_quantity
        eta = market_params.temp_impact_coef
        sigma_abs = order.start_price * market_params.interval_volatility
        lam = order.risk_aversion

        # Objective function over trade sizes n = [n_1, ... n_N]
        def objective(n: np.ndarray) -> float:
            cum_n = np.cumsum(n)
            x_rem = np.maximum(X - cum_n, 0.0)
            
            temp_cost = (eta / tau) * np.sum(n ** 2)
            var_cost = lam * (sigma_abs ** 2) * np.sum(x_rem ** 2)
            return temp_cost + var_cost

        # Initial guess: TWAP
        n0 = np.full(N, X / N)
        bounds = [(0.0, X) for _ in range(N)]
        constraints = [{"type": "eq", "fun": lambda n: np.sum(n) - X}]

        res = minimize(objective, n0, method="SLSQP", bounds=bounds, constraints=constraints)
        trade_sizes = res.x if res.success else n0
        trade_sizes = np.maximum(trade_sizes, 0.0)
        trade_sizes = trade_sizes * (X / np.sum(trade_sizes))

        inventory = np.zeros(N + 1)
        inventory[0] = X
        inventory[1:] = np.maximum(X - np.cumsum(trade_sizes), 0.0)
        inventory[-1] = 0.0

        return inventory, trade_sizes
