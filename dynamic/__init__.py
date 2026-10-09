"""
Dynamic execution module for Optimal Execution Engine.
Includes Adaptive Rebalancer for mid-execution volatility and liquidity shocks.
"""

from .rebalancer import (
    DynamicExecutionEngine,
    DynamicExecutionState,
    DynamicRebalancingResult
)
