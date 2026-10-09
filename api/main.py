"""
FastAPI Server for Optimal Execution Engine.
Exposes REST endpoints for market analytics, optimal schedule calculation, Monte Carlo simulation,
Efficient Frontier, parameter robustness, and dynamic re-optimization.
"""

import sys
import os
# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, List
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from config import OrderConfig, MarketParameters, OrderSide, SimulationConfig
from data.market_data import MarketDataLoader, generate_u_shape_volume_profile
from analytics.returns_volatility import (
    calculate_historical_volatility,
    calculate_parkinson_volatility,
    calculate_garman_klass_volatility,
    scale_volatility_to_interval
)
from analytics.liquidity_volume import calculate_liquidity_summary
from models.market_impact import estimate_almgren_impact_params
from strategies.almgren_chriss import AlmgrenChrissExecutionStrategy
from strategies.twap import TWAPExecutionStrategy
from strategies.vwap import VWAPExecutionStrategy
from strategies.immediate import ImmediateExecutionStrategy
from simulation.monte_carlo import MonteCarloEngine
from evaluation.efficient_frontier import EfficientFrontierEngine
from evaluation.robustness import RobustnessEngine
from dynamic.rebalancer import DynamicExecutionEngine

from .schemas import (
    OrderRequest,
    MarketParametersRequest,
    ExecutionScheduleItem,
    StrategySchedulesResponse,
    MonteCarloRequest,
    MonteCarloResponse,
    StrategySummaryItem,
    EfficientFrontierResponse,
    EfficientFrontierPointItem,
    DynamicShockRequest,
    DynamicShockResponse
)

# Initialize FastAPI App
app = FastAPI(
    title="Optimal Execution Engine API",
    description="Quantitative Execution Engine using Almgren-Chriss Framework, Monte Carlo, and Dynamic Adaptive Execution.",
    version="2.0.0"
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_domain_objects(order_req: OrderRequest, mkt_req: MarketParametersRequest):
    """Converts Pydantic requests to domain dataclasses."""
    side = OrderSide.BUY if order_req.side.upper() == "BUY" else OrderSide.SELL
    order = OrderConfig(
        symbol=order_req.symbol,
        side=side,
        total_quantity=order_req.total_quantity,
        start_price=order_req.start_price,
        time_horizon_min=order_req.time_horizon_min,
        num_intervals=order_req.num_intervals,
        risk_aversion=order_req.risk_aversion
    )

    daily_vol = mkt_req.annual_volatility / np.sqrt(252.0)
    interval_vol = scale_volatility_to_interval(mkt_req.annual_volatility, order.interval_duration_min)

    h_eta, h_gamma = estimate_almgren_impact_params(order.start_price, mkt_req.adv, daily_vol)
    eta = mkt_req.temp_impact_coef if mkt_req.temp_impact_coef is not None else h_eta
    gamma = mkt_req.perm_impact_coef if mkt_req.perm_impact_coef is not None else h_gamma

    market_params = MarketParameters(
        symbol=mkt_req.symbol,
        annual_volatility=mkt_req.annual_volatility,
        daily_volatility=daily_vol,
        interval_volatility=interval_vol,
        adv=mkt_req.adv,
        half_spread_bps=mkt_req.half_spread_bps,
        temp_impact_coef=eta,
        perm_impact_coef=gamma
    )
    return order, market_params


# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------

@app.get("/api/health")
def health_check():
    return {"status": "online", "version": "2.0.0", "engine": "Optimal Execution Engine (Almgren-Chriss)"}


@app.get("/api/assets")
def get_preset_assets():
    """Returns preset institutional equity parameters."""
    assets = []
    for sym, p in MarketDataLoader.PRESET_ASSETS.items():
        assets.append({
            "symbol": sym,
            "start_price": p["start_price"],
            "annual_vol": p["annual_vol"],
            "adv": p["adv"],
            "spread_bps": p["spread_bps"]
        })
    return {"assets": assets}


@app.get("/api/market-data/{symbol}")
def get_market_data(symbol: str):
    """Retrieves recent OHLCV bars, volatility estimators, and liquidity profile."""
    loader = MarketDataLoader()
    df = loader.get_market_data(symbol=symbol, days=30)

    hist_vol = calculate_historical_volatility(df["close"])
    park_vol = calculate_parkinson_volatility(df["high"], df["low"])
    gk_vol = calculate_garman_klass_volatility(df["open"], df["high"], df["low"], df["close"])
    u_profile = generate_u_shape_volume_profile(12).tolist()

    bars = []
    for _, row in df.tail(60).iterrows():
        bars.append({
            "timestamp": str(row["timestamp"]) if "timestamp" in row else "",
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "vwap": float(row.get("vwap", row["close"]))
        })

    return {
        "symbol": symbol,
        "bars": bars,
        "volatility_estimators": {
            "close_to_close": hist_vol,
            "parkinson": park_vol,
            "garman_klass": gk_vol
        },
        "u_shape_volume_profile": u_profile
    }


@app.post("/api/execute/schedules", response_model=StrategySchedulesResponse)
def compute_schedules(order_req: OrderRequest, mkt_req: MarketParametersRequest):
    """Generates optimal and benchmark execution trajectories."""
    order, market_params = _build_domain_objects(order_req, mkt_req)
    notional = order.total_quantity * order.start_price

    strategies = [
        AlmgrenChrissExecutionStrategy(),
        TWAPExecutionStrategy(),
        VWAPExecutionStrategy(),
        ImmediateExecutionStrategy()
    ]

    sched_map = {}
    for strat in strategies:
        sched = strat.generate_schedule(order, market_params)
        cost_bps = (sched.expected_cost / notional) * 10_000.0
        std_bps = (sched.expected_std_cost / notional) * 10_000.0

        sched_map[strat.name] = ExecutionScheduleItem(
            strategy_name=strat.name,
            time_steps_min=sched.time_steps_min.tolist(),
            trade_sizes=sched.trade_sizes.tolist(),
            inventory_remaining=sched.inventory_remaining.tolist(),
            trading_rates=sched.trading_rates.tolist(),
            expected_cost=float(sched.expected_cost),
            expected_cost_bps=float(cost_bps),
            expected_variance=float(sched.expected_variance),
            expected_std_cost=float(sched.expected_std_cost),
            expected_std_bps=float(std_bps),
            utility=float(sched.utility),
            metadata=sched.metadata
        )

    return StrategySchedulesResponse(
        order=order_req,
        market_params={
            "temp_impact_coef": market_params.temp_impact_coef,
            "perm_impact_coef": market_params.perm_impact_coef,
            "interval_volatility": market_params.interval_volatility,
            "daily_volatility": market_params.daily_volatility
        },
        schedules=sched_map
    )


@app.post("/api/simulate/monte-carlo", response_model=MonteCarloResponse)
def run_monte_carlo(req: MonteCarloRequest):
    """Runs vectorized multi-path stochastic price simulation and computes tail risk metrics."""
    order, market_params = _build_domain_objects(req.order, req.market_params)

    schedules = [
        AlmgrenChrissExecutionStrategy().generate_schedule(order, market_params),
        TWAPExecutionStrategy().generate_schedule(order, market_params),
        VWAPExecutionStrategy().generate_schedule(order, market_params),
        ImmediateExecutionStrategy().generate_schedule(order, market_params)
    ]

    mc_engine = MonteCarloEngine()
    sim_config = SimulationConfig(num_paths=req.num_paths, random_seed=req.random_seed)
    res = mc_engine.run_simulations(order, market_params, schedules, sim_config)

    # Prepare sample paths for frontend charting (first 30 paths)
    sample_paths = res.exogenous_price_paths[:30].tolist()
    time_steps = [k * order.interval_duration_min for k in range(order.num_intervals + 1)]

    summary_items = {}
    for name, s in res.strategy_summaries.items():
        # Compute histogram bins
        counts, bin_edges = np.histogram(s.all_shortfalls_bps, bins=25)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        summary_items[name] = StrategySummaryItem(
            strategy_name=name,
            num_paths=s.num_paths,
            mean_shortfall_total=s.mean_shortfall_total,
            median_shortfall_total=s.median_shortfall_total,
            std_shortfall_total=s.std_shortfall_total,
            mean_shortfall_bps=s.mean_shortfall_bps,
            median_shortfall_bps=s.median_shortfall_bps,
            std_shortfall_bps=s.std_shortfall_bps,
            var_95_bps=s.var_95_bps,
            var_99_bps=s.var_99_bps,
            cvar_95_bps=s.cvar_95_bps,
            worst_case_bps=s.worst_case_bps,
            best_case_bps=s.best_case_bps,
            mean_execution_price=s.mean_execution_price,
            std_execution_price=s.std_execution_price,
            shortfall_histogram={
                "counts": counts.tolist(),
                "bins": [round(float(b), 2) for b in bin_centers]
            }
        )

    return MonteCarloResponse(
        order=req.order,
        num_paths=req.num_paths,
        sample_price_paths=sample_paths,
        time_steps_min=time_steps,
        strategy_summaries=summary_items
    )


@app.post("/api/evaluate/efficient-frontier", response_model=EfficientFrontierResponse)
def get_efficient_frontier(order_req: OrderRequest, mkt_req: MarketParametersRequest):
    """Calculates the continuum of optimal strategies across risk aversion parameter λ."""
    order, market_params = _build_domain_objects(order_req, mkt_req)
    engine = EfficientFrontierEngine(num_points=35)
    ef_res = engine.generate_frontier(order, market_params)

    frontier_items = [
        EfficientFrontierPointItem(
            risk_aversion=p.risk_aversion,
            kappa=p.kappa,
            expected_cost_bps=p.expected_cost_bps,
            expected_std_bps=p.expected_std_bps,
            utility=p.utility
        )
        for p in ef_res.frontier_points
    ]

    u = ef_res.user_point
    user_item = EfficientFrontierPointItem(
        risk_aversion=u.risk_aversion,
        kappa=u.kappa,
        expected_cost_bps=u.expected_cost_bps,
        expected_std_bps=u.expected_std_bps,
        utility=u.utility
    )

    return EfficientFrontierResponse(
        frontier=frontier_items,
        benchmarks=ef_res.benchmark_points,
        user_point=user_item
    )


@app.post("/api/evaluate/robustness")
def run_robustness_test(order_req: OrderRequest, mkt_req: MarketParametersRequest):
    """Stress tests model under parameter uncertainty (misspecified volatility & impact)."""
    order, market_params = _build_domain_objects(order_req, mkt_req)
    engine = RobustnessEngine(num_paths=200)
    res = engine.run_robustness_analysis(order, market_params)

    return {
        "conclusion": res.research_conclusion,
        "volatility_experiments": [
            {
                "multiplier": e.multiplier,
                "true_value": e.true_param_value,
                "ac_cost_bps": e.ac_mean_cost_bps,
                "twap_cost_bps": e.twap_mean_cost_bps,
                "vwap_cost_bps": e.vwap_mean_cost_bps,
                "ac_outperforms": e.ac_outperforms_twap
            }
            for e in res.volatility_experiments
        ],
        "impact_experiments": [
            {
                "multiplier": e.multiplier,
                "true_value": e.true_param_value,
                "ac_cost_bps": e.ac_mean_cost_bps,
                "twap_cost_bps": e.twap_mean_cost_bps,
                "vwap_cost_bps": e.vwap_mean_cost_bps,
                "ac_outperforms": e.ac_outperforms_twap
            }
            for e in res.impact_experiments
        ]
    }


@app.post("/api/dynamic/shock", response_model=DynamicShockResponse)
def simulate_dynamic_shock(req: DynamicShockRequest):
    """Simulates mid-execution market shock and re-optimizes remaining trajectory."""
    order, market_params = _build_domain_objects(req.order, req.market_params)
    dyn_engine = DynamicExecutionEngine()

    res = dyn_engine.simulate_dynamic_shock(
        order=order,
        initial_params=market_params,
        shock_step=req.shock_step,
        volatility_multiplier=req.volatility_multiplier,
        liquidity_multiplier=req.liquidity_multiplier
    )

    time_steps = [k * order.interval_duration_min for k in range(order.num_intervals + 1)]

    return DynamicShockResponse(
        rebalance_step=res.rebalance_step,
        rebalance_time_min=res.rebalance_time_min,
        old_kappa=res.old_kappa,
        new_kappa=res.new_kappa,
        urgency_shift=res.urgency_shift,
        time_steps_min=time_steps,
        original_inventory=res.original_schedule.inventory_remaining.tolist(),
        dynamic_inventory=res.combined_trajectory.tolist(),
        original_trade_sizes=res.original_schedule.trade_sizes.tolist(),
        dynamic_trade_sizes=res.combined_trades.tolist()
    )


# -------------------------------------------------------------
# Static Files & Frontend Routing
# -------------------------------------------------------------
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def serve_frontend_root():
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Optimal Execution Engine API is running. Access /docs for API documentation."}
