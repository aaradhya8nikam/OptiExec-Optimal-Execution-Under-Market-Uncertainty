"""
Order Input and Parameter Controls Component.
Renders sidebar configuration for institutional orders, market parameters, and impact estimation.
"""

from typing import Tuple, Dict, Any
import streamlit as st
import numpy as np
from config import OrderConfig, MarketParameters, OrderSide
from data.market_data import MarketDataLoader
from models.market_impact import estimate_almgren_impact_params
from models.ml_impact import MLImpactEstimator
from analytics.returns_volatility import scale_volatility_to_interval


def render_order_sidebar() -> Tuple[OrderConfig, MarketParameters, Dict[str, Any]]:
    """
    Renders the sidebar controls and returns (OrderConfig, MarketParameters, ui_settings).
    """
    st.sidebar.markdown("## ⚙️ Order Configuration")
    
    # 1. Preset Assets or Custom
    symbols = list(MarketDataLoader.PRESET_ASSETS.keys()) + ["Custom Asset"]
    selected_symbol = st.sidebar.selectbox("Asset Symbol", symbols, index=0)

    if selected_symbol == "Custom Asset":
        symbol_name = st.sidebar.text_input("Custom Symbol", value="TATAMOTORS.NS")
        default_price = 950.0
        default_vol = 0.28
        default_adv = 8_000_000.0
        default_spread = 2.5
    else:
        symbol_name = selected_symbol
        preset = MarketDataLoader.PRESET_ASSETS[selected_symbol]
        default_price = preset["start_price"]
        default_vol = preset["annual_vol"]
        default_adv = preset["adv"]
        default_spread = preset["spread_bps"]

    # 2. Order Details
    col_side, col_price = st.sidebar.columns(2)
    with col_side:
        side_str = st.selectbox("Direction", ["BUY", "SELL"], index=0)
        side = OrderSide.BUY if side_str == "BUY" else OrderSide.SELL
    with col_price:
        start_price = st.number_input("Decision Price ($/₹)", value=float(default_price), min_value=0.1, step=10.0)

    order_qty = st.sidebar.number_input(
        "Order Quantity (Shares)",
        value=100_000,
        min_value=100,
        max_value=100_000_000,
        step=10_000
    )

    col_h, col_n = st.sidebar.columns(2)
    with col_h:
        time_horizon_min = st.number_input("Window (Mins)", value=60.0, min_value=5.0, max_value=480.0, step=15.0)
    with col_n:
        num_intervals = st.number_input("Intervals (N)", value=12, min_value=2, max_value=100, step=1)

    # 3. Risk Aversion
    st.sidebar.markdown("### 🎯 Risk Preference")
    risk_preset = st.sidebar.select_slider(
        "Risk Profile",
        options=["Risk Neutral (TWAP)", "Low Risk Averse", "Balanced (Institutional)", "High Risk Averse", "Urgent / Fast"],
        value="Balanced (Institutional)"
    )
    
    preset_lambda_map = {
        "Risk Neutral (TWAP)": 0.0,
        "Low Risk Averse": 1e-8,
        "Balanced (Institutional)": 1e-6,
        "High Risk Averse": 5e-5,
        "Urgent / Fast": 1e-3
    }
    default_lambda = preset_lambda_map[risk_preset]
    
    use_custom_lambda = st.sidebar.checkbox("Custom λ (log scale)", value=False)
    if use_custom_lambda:
        log_lambda = st.sidebar.slider("log10(λ)", min_value=-9.0, max_value=-2.0, value=-6.0, step=0.5)
        risk_aversion = 10.0 ** log_lambda
    else:
        risk_aversion = default_lambda

    # 4. Market Microstructure Parameters
    st.sidebar.markdown("### 📊 Market Microstructure")
    annual_vol = st.sidebar.slider("Annualized Volatility (σ)", min_value=0.05, max_value=0.90, value=float(default_vol), step=0.01)
    adv = st.sidebar.number_input("Average Daily Volume (ADV)", value=float(default_adv), min_value=10_000.0, step=500_000.0)
    half_spread_bps = st.sidebar.slider("Half-Spread (bps)", min_value=0.5, max_value=15.0, value=float(default_spread), step=0.5)

    # Calculate interval vol
    interval_dur = time_horizon_min / num_intervals
    interval_vol = scale_volatility_to_interval(annual_vol, interval_dur)
    daily_vol = annual_vol / np.sqrt(252.0)

    # Impact parameter calibration
    heuristic_eta, heuristic_gamma = estimate_almgren_impact_params(start_price, adv, daily_vol)
    
    impact_mode = st.sidebar.radio("Impact Calibration", ["Heuristic (Almgren)", "ML Calibrated", "Manual"], index=0)
    if impact_mode == "ML Calibrated":
        ml_est = MLImpactEstimator()
        part_rate = (order_qty / (adv * (time_horizon_min / 375.0)))
        calib_eta = ml_est.predict_impact_coefficient(order_qty / adv, annual_vol, half_spread_bps * 2, part_rate)
        calib_gamma = heuristic_gamma
    elif impact_mode == "Manual":
        calib_eta = st.sidebar.number_input("Temp Impact η", value=float(heuristic_eta), format="%.3e")
        calib_gamma = st.sidebar.number_input("Perm Impact γ", value=float(heuristic_gamma), format="%.3e")
    else:
        calib_eta = heuristic_eta
        calib_gamma = heuristic_gamma

    order = OrderConfig(
        symbol=symbol_name,
        side=side,
        total_quantity=float(order_qty),
        start_price=float(start_price),
        time_horizon_min=float(time_horizon_min),
        num_intervals=int(num_intervals),
        risk_aversion=float(risk_aversion)
    )

    market_params = MarketParameters(
        symbol=symbol_name,
        annual_volatility=float(annual_vol),
        daily_volatility=float(daily_vol),
        interval_volatility=float(interval_vol),
        adv=float(adv),
        half_spread_bps=float(half_spread_bps),
        temp_impact_coef=float(calib_eta),
        perm_impact_coef=float(calib_gamma)
    )

    ui_settings = {
        "num_mc_paths": st.sidebar.slider("Monte Carlo Paths", min_value=100, max_value=2500, value=500, step=100),
        "impact_mode": impact_mode
    }

    return order, market_params, ui_settings
