"""
Models module for Optimal Execution Engine.
Includes Almgren-Chriss market impact functions, stochastic price processes, and ML impact estimator.
"""

from .market_impact import (
    MarketImpactModel,
    estimate_almgren_impact_params,
    calculate_instantaneous_cost
)

from .price_process import (
    PriceProcessSimulator,
    generate_abm_price_paths,
    generate_gbm_price_paths
)

from .ml_impact import MLImpactEstimator
