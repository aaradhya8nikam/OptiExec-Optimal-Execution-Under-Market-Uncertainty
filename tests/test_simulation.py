"""
Tests for simulation, Monte Carlo engine, Efficient Frontier, and Dynamic Rebalancer.
"""

import pytest
import numpy as np
from config import OrderConfig, MarketParameters, SimulationConfig, OrderSide
from models.market_impact import MarketImpactModel
from strategies.almgren_chriss import AlmgrenChrissExecutionStrategy
from strategies.twap import TWAPExecutionStrategy
from simulation.execution_simulator import ExecutionSimulator
from simulation.monte_carlo import MonteCarloEngine
from evaluation.efficient_frontier import EfficientFrontierEngine
from evaluation.robustness import RobustnessEngine
from dynamic.rebalancer import DynamicExecutionEngine


@pytest.fixture
def sim_setup():
    order = OrderConfig(
        symbol="INFY.NS",
        side=OrderSide.BUY,
        total_quantity=50_000.0,
        start_price=1800.0,
        time_horizon_min=60.0,
        num_intervals=12,
        risk_aversion=1e-6
    )
    market = MarketParameters(
        symbol="INFY.NS",
        annual_volatility=0.25,
        daily_volatility=0.0157,
        interval_volatility=0.0022,
        adv=5_000_000.0,
        half_spread_bps=2.5,
        temp_impact_coef=2.5e-7,
        perm_impact_coef=5.0e-8
    )
    return order, market


def test_single_path_execution_simulator(sim_setup):
    order, market = sim_setup
    impact = MarketImpactModel(eta=market.temp_impact_coef, gamma=market.perm_impact_coef)
    simulator = ExecutionSimulator(impact)
    
    ac_strat = AlmgrenChrissExecutionStrategy()
    sched = ac_strat.generate_schedule(order, market)
    
    price_path = np.linspace(1800.0, 1810.0, order.num_intervals + 1)
    res = simulator.execute_path(sched, price_path, order, market)
    
    assert len(res.records) == order.num_intervals
    assert res.total_shares == 50_000.0
    assert res.average_execution_price > 0.0
    assert res.implementation_shortfall_total != 0.0


def test_monte_carlo_engine(sim_setup):
    order, market = sim_setup
    ac_strat = AlmgrenChrissExecutionStrategy()
    twap_strat = TWAPExecutionStrategy()
    
    schedules = [
        ac_strat.generate_schedule(order, market),
        twap_strat.generate_schedule(order, market)
    ]
    
    mc = MonteCarloEngine()
    sim_config = SimulationConfig(num_paths=100, random_seed=42)
    mc_res = mc.run_simulations(order, market, schedules, sim_config)
    
    assert len(mc_res.strategy_summaries) == 2
    ac_summary = mc_res.strategy_summaries["Almgren-Chriss (Optimal)"]
    assert ac_summary.num_paths == 100
    assert ac_summary.var_95_bps > 0.0
    assert ac_summary.cvar_95_bps >= ac_summary.var_95_bps


def test_efficient_frontier_engine(sim_setup):
    order, market = sim_setup
    ef_engine = EfficientFrontierEngine(num_points=10)
    ef_res = ef_engine.generate_frontier(order, market)
    
    assert len(ef_res.frontier_points) == 10
    assert "TWAP" in ef_res.benchmark_points
    assert "Immediate Execution" in ef_res.benchmark_points
    assert ef_res.user_point is not None


def test_dynamic_rebalancer(sim_setup):
    order, market = sim_setup
    dyn_engine = DynamicExecutionEngine()
    
    dyn_res = dyn_engine.simulate_dynamic_shock(
        order=order,
        initial_params=market,
        shock_step=4,
        volatility_multiplier=2.0,
        liquidity_multiplier=0.5
    )
    
    assert len(dyn_res.combined_trades) == order.num_intervals
    assert np.isclose(np.sum(dyn_res.combined_trades), order.total_quantity)
    assert dyn_res.new_kappa > dyn_res.old_kappa
