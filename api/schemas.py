"""
Pydantic Request & Response Schemas for the FastAPI Execution Engine REST API.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class OrderRequest(BaseModel):
    symbol: str = "RELIANCE.NS"
    side: str = "BUY"  # BUY or SELL
    total_quantity: float = Field(default=100_000.0, gt=0)
    start_price: float = Field(default=1350.0, gt=0)
    time_horizon_min: float = Field(default=60.0, gt=0)
    num_intervals: int = Field(default=12, ge=2, le=100)
    risk_aversion: float = Field(default=1e-6, ge=0)


class MarketParametersRequest(BaseModel):
    symbol: str = "RELIANCE.NS"
    annual_volatility: float = Field(default=0.22, gt=0)
    adv: float = Field(default=7_500_000.0, gt=0)
    half_spread_bps: float = Field(default=2.0, ge=0)
    temp_impact_coef: Optional[float] = None
    perm_impact_coef: Optional[float] = None


class ExecutionScheduleItem(BaseModel):
    strategy_name: str
    time_steps_min: List[float]
    trade_sizes: List[float]
    inventory_remaining: List[float]
    trading_rates: List[float]
    expected_cost: float
    expected_cost_bps: float
    expected_variance: float
    expected_std_cost: float
    expected_std_bps: float
    utility: float
    metadata: Dict[str, Any] = {}


class StrategySchedulesResponse(BaseModel):
    order: OrderRequest
    market_params: Dict[str, Any]
    schedules: Dict[str, ExecutionScheduleItem]


class MonteCarloRequest(BaseModel):
    order: OrderRequest
    market_params: MarketParametersRequest
    num_paths: int = Field(default=500, ge=50, le=5000)
    random_seed: Optional[int] = 42


class StrategySummaryItem(BaseModel):
    strategy_name: str
    num_paths: int
    mean_shortfall_total: float
    median_shortfall_total: float
    std_shortfall_total: float
    mean_shortfall_bps: float
    median_shortfall_bps: float
    std_shortfall_bps: float
    var_95_bps: float
    var_99_bps: float
    cvar_95_bps: float
    worst_case_bps: float
    best_case_bps: float
    mean_execution_price: float
    std_execution_price: float
    shortfall_histogram: Dict[str, Any] = {}


class MonteCarloResponse(BaseModel):
    order: OrderRequest
    num_paths: int
    sample_price_paths: List[List[float]]
    time_steps_min: List[float]
    strategy_summaries: Dict[str, StrategySummaryItem]


class EfficientFrontierPointItem(BaseModel):
    risk_aversion: float
    kappa: float
    expected_cost_bps: float
    expected_std_bps: float
    utility: float


class EfficientFrontierResponse(BaseModel):
    frontier: List[EfficientFrontierPointItem]
    benchmarks: Dict[str, Dict[str, float]]
    user_point: EfficientFrontierPointItem


class DynamicShockRequest(BaseModel):
    order: OrderRequest
    market_params: MarketParametersRequest
    shock_step: int = Field(default=4, ge=1)
    volatility_multiplier: float = Field(default=2.0, gt=0)
    liquidity_multiplier: float = Field(default=0.5, gt=0)


class DynamicShockResponse(BaseModel):
    rebalance_step: int
    rebalance_time_min: float
    old_kappa: float
    new_kappa: float
    urgency_shift: str
    time_steps_min: List[float]
    original_inventory: List[float]
    dynamic_inventory: List[float]
    original_trade_sizes: List[float]
    dynamic_trade_sizes: List[float]
