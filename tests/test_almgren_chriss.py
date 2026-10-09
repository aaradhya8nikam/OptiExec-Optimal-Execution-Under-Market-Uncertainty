"""
Tests for Almgren-Chriss (2000) optimal execution strategy.
"""

import pytest
import numpy as np
from config import OrderConfig, MarketParameters, OrderSide
from strategies.almgren_chriss import AlmgrenChrissExecutionStrategy
from strategies.twap import TWAPExecutionStrategy


@pytest.fixture
def base_setup():
    order = OrderConfig(
        symbol="RELIANCE.NS",
        side=OrderSide.BUY,
        total_quantity=100_000.0,
        start_price=1000.0,
        time_horizon_min=60.0,
        num_intervals=12,
        risk_aversion=1e-6
    )
    market = MarketParameters(
        symbol="RELIANCE.NS",
        annual_volatility=0.25,
        daily_volatility=0.0157,
        interval_volatility=0.0022,
        adv=5_000_000.0,
        half_spread_bps=2.5,
        temp_impact_coef=2.5e-7,
        perm_impact_coef=5.0e-8
    )
    return order, market


def test_almgren_chriss_basic_properties(base_setup):
    order, market = base_setup
    strat = AlmgrenChrissExecutionStrategy(use_numerical_solver=False)
    sched = strat.generate_schedule(order, market)

    assert np.isclose(np.sum(sched.trade_sizes), order.total_quantity)
    assert sched.inventory_remaining[0] == order.total_quantity
    assert sched.inventory_remaining[-1] == 0.0
    assert np.all(sched.trade_sizes >= 0.0)
    # Inventory is monotonically non-increasing
    assert np.all(np.diff(sched.inventory_remaining) <= 1e-6)


def test_risk_neutral_limit_equals_twap(base_setup):
    order, market = base_setup
    order.risk_aversion = 0.0  # Risk neutral
    
    ac_strat = AlmgrenChrissExecutionStrategy()
    twap_strat = TWAPExecutionStrategy()
    
    ac_sched = ac_strat.generate_schedule(order, market)
    twap_sched = twap_strat.generate_schedule(order, market)

    assert np.allclose(ac_sched.trade_sizes, twap_sched.trade_sizes, atol=1e-3)
    assert np.allclose(ac_sched.inventory_remaining, twap_sched.inventory_remaining, atol=1e-3)


def test_risk_averse_front_loading(base_setup):
    order, market = base_setup
    order.risk_aversion = 1e-4  # High risk aversion
    
    ac_strat = AlmgrenChrissExecutionStrategy()
    sched = ac_strat.generate_schedule(order, market)

    # First interval trade should be significantly larger than the last interval
    assert sched.trade_sizes[0] > sched.trade_sizes[-1] * 2.0


def test_numerical_vs_analytical_agreement(base_setup):
    order, market = base_setup
    order.risk_aversion = 1e-6
    
    analytical_strat = AlmgrenChrissExecutionStrategy(use_numerical_solver=False)
    numerical_strat = AlmgrenChrissExecutionStrategy(use_numerical_solver=True)
    
    ana_sched = analytical_strat.generate_schedule(order, market)
    num_sched = numerical_strat.generate_schedule(order, market)

    # Both should have close trade sizes and expected cost
    assert np.allclose(ana_sched.trade_sizes, num_sched.trade_sizes, atol=500.0)
    assert np.isclose(ana_sched.expected_cost, num_sched.expected_cost, rtol=0.05)
