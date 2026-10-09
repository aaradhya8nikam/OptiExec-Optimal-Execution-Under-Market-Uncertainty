"""
Tests for returns, volatility estimators, and liquidity analytics.
"""

import pytest
import numpy as np
import pandas as pd
from analytics.returns_volatility import (
    calculate_returns,
    calculate_historical_volatility,
    calculate_parkinson_volatility,
    calculate_garman_klass_volatility,
    calculate_rolling_volatility,
    scale_volatility_to_interval
)
from analytics.liquidity_volume import (
    calculate_adv,
    estimate_intraday_volume_profile,
    calculate_participation_rate,
    calculate_order_to_adv_ratio,
    calculate_liquidity_summary
)
from data.market_data import generate_synthetic_ohlcv, generate_u_shape_volume_profile


def test_returns_calculation():
    prices = np.array([100.0, 105.0, 102.0, 110.0])
    log_ret = calculate_returns(prices, method="log")
    simple_ret = calculate_returns(prices, method="simple")
    
    assert len(log_ret) == 3
    assert np.isclose(log_ret[0], np.log(105.0 / 100.0))
    assert np.isclose(simple_ret[0], 0.05)


def test_volatility_estimators():
    df = generate_synthetic_ohlcv(days=10, intervals_per_day=75, seed=42)
    
    hist_vol = calculate_historical_volatility(df["close"])
    park_vol = calculate_parkinson_volatility(df["high"], df["low"])
    gk_vol = calculate_garman_klass_volatility(df["open"], df["high"], df["low"], df["close"])
    
    assert 0.10 < hist_vol < 0.40
    assert 0.10 < park_vol < 0.40
    assert 0.10 < gk_vol < 0.40


def test_u_shape_volume_profile():
    profile = generate_u_shape_volume_profile(12)
    assert len(profile) == 12
    assert np.isclose(np.sum(profile), 1.0)
    # Ends should be higher than the midpoint
    assert profile[0] > profile[6]
    assert profile[-1] > profile[6]


def test_liquidity_analytics():
    df = generate_synthetic_ohlcv(days=5, intervals_per_day=75, adv=5_000_000.0, seed=42)
    adv = calculate_adv(df)
    assert adv > 1_000_000.0

    part_rate = calculate_participation_rate(100_000, 60.0, adv)
    assert 0.0 < part_rate < 1.0

    order_adv_ratio = calculate_order_to_adv_ratio(100_000, adv)
    assert 0.0 < order_adv_ratio < 0.1

    summary = calculate_liquidity_summary(df, 100_000, 60.0)
    assert "adv" in summary
    assert "participation_rate_pct" in summary
