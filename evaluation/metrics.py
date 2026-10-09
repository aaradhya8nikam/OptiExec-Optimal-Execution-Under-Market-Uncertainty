"""
Institutional Performance and Execution Risk Metrics.
Calculates Implementation Shortfall, execution slippage, tracking error, and comparative summary tables.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from config import OrderConfig, ExecutionSchedule
from simulation.monte_carlo import StrategySimulationSummary


def calculate_implementation_shortfall(
    executed_cash_flow: float,
    total_shares: float,
    decision_price: float,
    is_buy: bool = True
) -> tuple[float, float]:
    """
    Computes Implementation Shortfall (IS) in dollars and basis points.
    IS_buy  = Executed_Cash - Shares * S_0
    IS_sell = Shares * S_0 - Executed_Cash
    IS_bps  = (IS / (Shares * S_0)) * 10,000
    """
    notional = total_shares * decision_price
    if is_buy:
        shortfall_dollars = executed_cash_flow - notional
    else:
        shortfall_dollars = notional - executed_cash_flow

    shortfall_bps = (shortfall_dollars / max(notional, 1e-9)) * 10_000.0
    return float(shortfall_dollars), float(shortfall_bps)


def calculate_execution_efficiency_ratio(mean_shortfall_bps: float, std_shortfall_bps: float) -> float:
    """
    Risk-adjusted execution cost ratio: higher volatility relative to cost is penalized.
    EER = Mean Cost / Std Dev of Cost (lower is generally better, representing predictable low cost).
    """
    if std_shortfall_bps <= 1e-6:
        return float("inf")
    return float(mean_shortfall_bps / std_shortfall_bps)


def compute_strategy_comparison_table(
    summaries: Dict[str, StrategySimulationSummary],
    schedules: Dict[str, ExecutionSchedule]
) -> pd.DataFrame:
    """
    Creates a comprehensive side-by-side performance comparison dataframe.
    """
    rows = []
    for name, summary in summaries.items():
        sched = schedules.get(name)
        exp_cost = sched.expected_cost if sched else summary.mean_shortfall_total
        exp_std = sched.expected_std_cost if sched else summary.std_shortfall_total

        rows.append({
            "Strategy": name,
            "Exp. Cost ($)": f"{exp_cost:,.2f}",
            "Sim. Mean Cost ($)": f"{summary.mean_shortfall_total:,.2f}",
            "Mean IS (bps)": f"{summary.mean_shortfall_bps:.2f}",
            "Median IS (bps)": f"{summary.median_shortfall_bps:.2f}",
            "Std Dev (bps)": f"{summary.std_shortfall_bps:.2f}",
            "95% VaR (bps)": f"{summary.var_95_bps:.2f}",
            "95% CVaR (bps)": f"{summary.cvar_95_bps:.2f}",
            "Worst Case (bps)": f"{summary.worst_case_bps:.2f}",
            "Avg Exec Price": f"{summary.mean_execution_price:.2f}"
        })

    return pd.DataFrame(rows)
