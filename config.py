"""
Optimal Execution Engine - Configuration and Data Structures
Defines domain dataclasses for Orders, Market Parameters, Simulation, and Schedules.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
import numpy as np


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class PriceModelType(str, Enum):
    ABM = "ABM"  # Arithmetic Brownian Motion
    GBM = "GBM"  # Geometric Brownian Motion


@dataclass
class OrderConfig:
    """Configuration for an institutional large order."""
    symbol: str = "RELIANCE.NS"
    side: OrderSide = OrderSide.BUY
    total_quantity: float = 100_000.0  # Number of shares
    start_price: float = 1_000.0       # Initial decision price (S_0)
    time_horizon_min: float = 60.0     # Total execution window in minutes (T)
    num_intervals: int = 12            # Number of discrete trading intervals (N)
    risk_aversion: float = 1e-6        # Risk aversion parameter lambda (λ)

    @property
    def interval_duration_min(self) -> float:
        """Duration of each discrete interval tau = T / N in minutes."""
        return self.time_horizon_min / self.num_intervals

    @property
    def total_shares_signed(self) -> float:
        """Signed quantity: positive for BUY, negative for SELL."""
        return self.total_quantity if self.side == OrderSide.BUY else -self.total_quantity

    def validate(self) -> None:
        if self.total_quantity <= 0:
            raise ValueError("Total quantity must be strictly positive.")
        if self.start_price <= 0:
            raise ValueError("Start price must be strictly positive.")
        if self.time_horizon_min <= 0:
            raise ValueError("Time horizon must be strictly positive.")
        if self.num_intervals <= 0:
            raise ValueError("Number of intervals must be >= 1.")
        if self.risk_aversion < 0:
            raise ValueError("Risk aversion lambda must be >= 0.")


@dataclass
class MarketParameters:
    """Asset liquidity and volatility parameters."""
    symbol: str = "RELIANCE.NS"
    annual_volatility: float = 0.25      # e.g., 25% annualized vol
    daily_volatility: float = 0.0157     # ~ 0.25 / sqrt(252)
    interval_volatility: float = 0.0022  # sigma per interval tau
    adv: float = 5_000_000.0             # Average Daily Volume (shares)
    half_spread_bps: float = 2.5         # Half bid-ask spread in basis points (1 bp = 0.0001)
    temp_impact_coef: float = 2.5e-7     # Temporary impact parameter eta (η)
    perm_impact_coef: float = 5.0e-8     # Permanent impact parameter gamma (γ)
    intraday_volume_profile: Optional[np.ndarray] = None  # Expected volume fraction per interval

    @property
    def half_spread_abs(self) -> float:
        """Half spread in absolute currency terms per share."""
        return self.half_spread_bps * 1e-4

    @classmethod
    def estimate_default_impacts(cls, price: float, adv: float, daily_vol: float) -> tuple[float, float]:
        """
        Standard heuristic impact estimation (Almgren et al. 2005):
        gamma ~= 0.1 * (daily_vol * price) / ADV
        eta ~= 0.5 * (daily_vol * price) / (0.1 * ADV)
        """
        gamma = 0.1 * (daily_vol * price) / max(adv, 1.0)
        eta = 0.5 * (daily_vol * price) / max(0.1 * adv, 1.0)
        return eta, gamma


@dataclass
class ExecutionSchedule:
    """Generated trading schedule for an execution strategy."""
    strategy_name: str
    time_steps_min: np.ndarray          # Array of interval timestamps [0, tau, 2*tau, ... T]
    trade_sizes: np.ndarray             # Shares traded in each interval [n_1, n_2, ... n_N]
    inventory_remaining: np.ndarray     # Inventory remaining at each step [x_0, x_1, ... x_N]
    trading_rates: np.ndarray           # Trade rate (shares/min) [v_1, v_2, ... v_N]
    expected_cost: float = 0.0          # E[x] expected transaction cost (Implementation Shortfall)
    expected_variance: float = 0.0      # V[x] variance of execution cost
    expected_std_cost: float = 0.0      # sqrt(V[x]) execution cost standard deviation
    utility: float = 0.0                # E[x] + lambda * V[x]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationConfig:
    """Parameters for Monte Carlo and execution path simulations."""
    num_paths: int = 1000
    random_seed: Optional[int] = 42
    price_model: PriceModelType = PriceModelType.ABM
    include_drift: bool = False
    drift_annual: float = 0.0
