"""
Tests for market impact models and ML impact estimator.
"""

import pytest
import numpy as np
from models.market_impact import MarketImpactModel, estimate_almgren_impact_params
from models.ml_impact import MLImpactEstimator


def test_linear_impact_model():
    model = MarketImpactModel(eta=2.5e-7, gamma=5.0e-8, half_spread_bps=2.0)
    
    # 10,000 shares / 5 min = 2,000 shares/min
    trade_rate = 2000.0
    price = 1000.0
    
    temp_impact = model.temporary_impact(trade_rate, price)
    perm_impact = model.permanent_impact(10_000.0)
    
    expected_half_spread = 1000.0 * (2.0 * 1e-4)  # 0.20
    expected_eta_cost = 2.5e-7 * 2000.0           # 0.0005
    assert np.isclose(temp_impact, expected_half_spread + expected_eta_cost)
    assert np.isclose(perm_impact, 5.0e-8 * 10_000.0)


def test_impact_parameter_estimation():
    eta, gamma = estimate_almgren_impact_params(price=1000.0, adv=5_000_000.0, daily_vol=0.015)
    assert eta > 0.0
    assert gamma > 0.0
    assert eta > gamma


def test_ml_impact_estimator():
    estimator = MLImpactEstimator(model_type="ridge")
    pred_eta = estimator.predict_impact_coefficient(
        order_to_adv=0.02,
        annual_vol=0.25,
        spread_bps=2.5,
        participation_rate=0.08
    )
    assert pred_eta > 0.0
    importances = estimator.get_feature_importances()
    assert len(importances) == 5
