"""
Stochastic Price Process Simulator (ABM & GBM).
Simulates asset price paths driven by Brownian motion with endogenous permanent market impact.
"""

from typing import Optional
import numpy as np
from config import PriceModelType


def generate_abm_price_paths(
    start_price: float,
    interval_vol_abs: float,
    num_intervals: int,
    num_paths: int = 1000,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Generates exogenous un-impacted Arithmetic Brownian Motion (ABM) price paths.
    S_k = S_{k-1} + sigma_abs * xi_k, where xi_k ~ N(0, 1)

    Returns:
    --------
    np.ndarray of shape (num_paths, num_intervals + 1)
    """
    if seed is not None:
        np.random.seed(seed)

    # Standard normal increments
    xi = np.random.normal(0.0, 1.0, size=(num_paths, num_intervals))
    increments = interval_vol_abs * xi
    
    # Prepend zero increment at t=0
    zero_col = np.zeros((num_paths, 1))
    all_increments = np.hstack([zero_col, increments])
    
    # Cumulative sum starting at start_price
    paths = start_price + np.cumsum(all_increments, axis=1)
    return np.maximum(paths, 0.01)


def generate_gbm_price_paths(
    start_price: float,
    interval_vol_pct: float,
    num_intervals: int,
    num_paths: int = 1000,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Generates exogenous Geometric Brownian Motion (GBM) price paths:
    S_k = S_{k-1} * exp( -0.5 * sigma^2 + sigma * xi_k )

    Returns:
    --------
    np.ndarray of shape (num_paths, num_intervals + 1)
    """
    if seed is not None:
        np.random.seed(seed)

    xi = np.random.normal(0.0, 1.0, size=(num_paths, num_intervals))
    log_increments = -0.5 * (interval_vol_pct ** 2) + interval_vol_pct * xi
    
    zero_col = np.zeros((num_paths, 1))
    all_log_increments = np.hstack([zero_col, log_increments])
    
    log_paths = np.log(start_price) + np.cumsum(all_log_increments, axis=1)
    return np.exp(log_paths)


class PriceProcessSimulator:
    """
    Manages price process simulations with exogenous Brownian shocks
    and endogenous permanent price impact.
    """
    def __init__(
        self,
        start_price: float,
        interval_vol: float,
        num_intervals: int,
        model_type: PriceModelType = PriceModelType.ABM
    ):
        self.start_price = start_price
        self.interval_vol = interval_vol
        self.num_intervals = num_intervals
        self.model_type = model_type

    def generate_exogenous_paths(self, num_paths: int = 1000, seed: Optional[int] = 42) -> np.ndarray:
        """Generates exogenous baseline price paths without trading impact."""
        if self.model_type == PriceModelType.ABM:
            # interval_vol in absolute price terms (S_0 * vol_pct)
            interval_vol_abs = self.start_price * self.interval_vol
            return generate_abm_price_paths(
                start_price=self.start_price,
                interval_vol_abs=interval_vol_abs,
                num_intervals=self.num_intervals,
                num_paths=num_paths,
                seed=seed
            )
        else:
            return generate_gbm_price_paths(
                start_price=self.start_price,
                interval_vol_pct=self.interval_vol,
                num_intervals=self.num_intervals,
                num_paths=num_paths,
                seed=seed
            )
