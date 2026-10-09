"""
Evaluation module for Optimal Execution Engine.
Includes Institutional Performance Metrics, Efficient Frontier generator, and Robustness/Sensitivity Matrix.
"""

from .metrics import (
    calculate_implementation_shortfall,
    calculate_execution_efficiency_ratio,
    compute_strategy_comparison_table
)

from .efficient_frontier import (
    EfficientFrontierEngine,
    EfficientFrontierPoint,
    EfficientFrontierResult
)

from .robustness import (
    RobustnessEngine,
    RobustnessResult,
    SensitivityExperiment
)
