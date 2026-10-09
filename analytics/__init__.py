"""
Analytics module for Optimal Execution Engine.
Includes returns and volatility modeling, and liquidity/volume profiling.
"""

from .returns_volatility import (
    calculate_returns,
    calculate_historical_volatility,
    calculate_parkinson_volatility,
    calculate_garman_klass_volatility,
    calculate_rolling_volatility,
    scale_volatility_to_interval
)

from .liquidity_volume import (
    calculate_adv,
    estimate_intraday_volume_profile,
    calculate_participation_rate,
    calculate_order_to_adv_ratio,
    calculate_liquidity_summary
)
