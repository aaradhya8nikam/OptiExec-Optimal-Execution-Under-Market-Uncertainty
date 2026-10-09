"""
Integration Tests for FastAPI Backend REST Endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "online"


def test_api_assets():
    res = client.get("/api/assets")
    assert res.status_code == 200
    assets = res.json()["assets"]
    assert len(assets) > 0
    assert any(a["symbol"] == "RELIANCE.NS" for a in assets)


def test_api_market_data():
    res = client.get("/api/market-data/RELIANCE.NS")
    assert res.status_code == 200
    data = res.json()
    assert "bars" in data
    assert "volatility_estimators" in data
    assert "garman_klass" in data["volatility_estimators"]


def test_api_execute_schedules():
    payload = {
        "order_req": {
            "symbol": "INFY.NS",
            "side": "BUY",
            "total_quantity": 50000.0,
            "start_price": 1820.0,
            "time_horizon_min": 60.0,
            "num_intervals": 12,
            "risk_aversion": 1e-6
        },
        "mkt_req": {
            "symbol": "INFY.NS",
            "annual_volatility": 0.26,
            "adv": 5200000.0,
            "half_spread_bps": 2.5
        }
    }
    res = client.post("/api/execute/schedules", json=payload)
    assert res.status_code == 200
    schedules = res.json()["schedules"]
    assert "Almgren-Chriss (Optimal)" in schedules
    assert "TWAP" in schedules
    assert "VWAP" in schedules
    assert len(schedules["Almgren-Chriss (Optimal)"]["trade_sizes"]) == 12


def test_api_monte_carlo():
    payload = {
        "order": {
            "symbol": "INFY.NS",
            "side": "BUY",
            "total_quantity": 50000.0,
            "start_price": 1820.0,
            "time_horizon_min": 60.0,
            "num_intervals": 12,
            "risk_aversion": 1e-6
        },
        "market_params": {
            "symbol": "INFY.NS",
            "annual_volatility": 0.26,
            "adv": 5200000.0,
            "half_spread_bps": 2.5
        },
        "num_paths": 100,
        "random_seed": 42
    }
    res = client.post("/api/simulate/monte-carlo", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "strategy_summaries" in data
    assert "Almgren-Chriss (Optimal)" in data["strategy_summaries"]
    ac_sum = data["strategy_summaries"]["Almgren-Chriss (Optimal)"]
    assert ac_sum["var_95_bps"] > 0.0


def test_api_dynamic_shock():
    payload = {
        "order": {
            "symbol": "INFY.NS",
            "side": "BUY",
            "total_quantity": 50000.0,
            "start_price": 1820.0,
            "time_horizon_min": 60.0,
            "num_intervals": 12,
            "risk_aversion": 1e-6
        },
        "market_params": {
            "symbol": "INFY.NS",
            "annual_volatility": 0.26,
            "adv": 5200000.0,
            "half_spread_bps": 2.5
        },
        "shock_step": 4,
        "volatility_multiplier": 2.0,
        "liquidity_multiplier": 0.5
    }
    res = client.post("/api/dynamic/shock", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["rebalance_step"] == 4
    assert len(data["dynamic_trade_sizes"]) == 12
