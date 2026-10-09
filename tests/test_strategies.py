"""
Tests for execution strategies: Immediate, TWAP, and VWAP.
"""

import pytest
import numpy as np
from config import OrderConfig, MarketParameters, OrderSide
from strategies.immediate import ImmediateExecutionStrategy
from strategies.twap import TWAPExecutionStrategy
from strategies.vwap import VWAPExecutionStrategy


@pytest.fixture
def sample_setup():
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


def test_immediate_strategy(sample_setup):
    order, market = sample_setup
    strat = ImmediateExecutionStrategy()
    sched = strat.generate_schedule(order, market)

    assert sched.trade_sizes[0] == 100_000.0
    assert np.all(sched.trade_sizes[1:] == 0.0)
    assert sched.inventory_remaining[0] == 100_000.0
    assert np.all(sched.inventory_remaining[1:] == 0.0)
    assert sched.expected_variance == 0.0  # Zero timing risk
    assert sched.expected_cost > 0.0


def test_twap_strategy(sample_setup):
    order, market = sample_setup
    strat = TWAPExecutionStrategy()
    sched = strat.generate_schedule(order, market)

    expected_slice = 100_000.0 / 12
    assert np.allclose(sched.trade_sizes, expected_slice)
    assert np.isclose(np.sum(sched.trade_sizes), 100_000.0)
    assert sched.inventory_remaining[0] == 100_000.0
    assert sched.inventory_remaining[-1] == 0.0


def test_vwap_strategy(sample_setup):
    order, market = sample_setup
    strat = VWAPExecutionStrategy()
    sched = strat.generate_schedule(order, market)

    assert np.isclose(np.sum(sched.trade_sizes), 100_000.0)
    assert sched.inventory_remaining[0] == 100_000.0
    assert sched.inventory_remaining[-1] == 0.0
    assert np.all(sched.trade_sizes > 0.0)
