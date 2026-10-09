"""
Simulation module for Optimal Execution Engine.
Provides path-by-path execution simulator and multithreaded/vectorized Monte Carlo engine.
"""

from .execution_simulator import (
    ExecutionSimulator,
    SimulatedExecutionResult,
    IntervalExecutionRecord
)

from .monte_carlo import (
    MonteCarloEngine,
    MonteCarloSimulationResult,
    StrategySimulationSummary
)
