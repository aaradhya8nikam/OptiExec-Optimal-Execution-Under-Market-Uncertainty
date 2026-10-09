"""
Market Analytics UI Component.
Visualizes historical OHLCV data, volatility estimators, volume profiles, and liquidity diagnostics.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import OrderConfig, MarketParameters
from analytics.returns_volatility import (
    calculate_historical_volatility,
    calculate_parkinson_volatility,
    calculate_garman_klass_volatility,
    calculate_rolling_volatility
)
from analytics.liquidity_volume import calculate_liquidity_summary, estimate_intraday_volume_profile


def render_market_analytics_tab(df: pd.DataFrame, order: OrderConfig, market_params: MarketParameters):
    """
    Renders the complete Market Conditions and Liquidity analytics tab.
    """
    st.markdown("### 📈 Market Microstructure & Liquidity Analysis")
    
    # 1. Top Diagnostic KPI Metrics
    liq = calculate_liquidity_summary(df, order.total_quantity, order.time_horizon_min)
    notional = order.total_quantity * order.start_price
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Order Notional", f"${notional:,.0f}" if "$" in order.symbol or not "NS" in order.symbol else f"₹{notional:,.0f}")
    c2.metric("Order-to-ADV", f"{liq['order_to_adv_pct']:.2f}%")
    c3.metric("Participation Rate", f"{liq['participation_rate_pct']:.2f}%")
    c4.metric("Liquidity Status", liq["liquidity_regime"].split("(")[0].strip())

    st.markdown("---")

    # 2. Interactive OHLCV Candlestick + Volume Chart
    st.markdown("#### 🕯️ Asset Price & Intraday Volume Dynamics")
    display_df = df.tail(150).copy() if len(df) > 150 else df.copy()

    fig_ohlc = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price Action (OHLC)", "Volume Traded")
    )

    fig_ohlc.add_trace(
        go.Candlestick(
            x=display_df["timestamp"] if "timestamp" in display_df.columns else display_df.index,
            open=display_df["open"],
            high=display_df["high"],
            low=display_df["low"],
            close=display_df["close"],
            name="OHLC",
            increasing_line_color="#00E676",
            decreasing_line_color="#FF5252"
        ),
        row=1, col=1
    )

    if "vwap" in display_df.columns:
        fig_ohlc.add_trace(
            go.Scatter(
                x=display_df["timestamp"] if "timestamp" in display_df.columns else display_df.index,
                y=display_df["vwap"],
                line=dict(color="#FFD700", width=1.5, dash="dot"),
                name="Market VWAP"
            ),
            row=1, col=1
        )

    fig_ohlc.add_trace(
        go.Bar(
            x=display_df["timestamp"] if "timestamp" in display_df.columns else display_df.index,
            y=display_df["volume"],
            marker_color="rgba(100, 181, 246, 0.7)",
            name="Volume"
        ),
        row=2, col=1
    )

    fig_ohlc.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_ohlc, use_container_width=True)

    # 3. Volatility Engine Estimators Comparison & Intraday U-Curve
    col_vol, col_curve = st.columns(2)

    with col_vol:
        st.markdown("#### 🔬 Volatility Engine Breakdown")
        hist_vol = calculate_historical_volatility(df["close"])
        park_vol = calculate_parkinson_volatility(df["high"], df["low"])
        gk_vol = calculate_garman_klass_volatility(df["open"], df["high"], df["low"], df["close"])

        vol_df = pd.DataFrame({
            "Estimator": ["Close-to-Close (Standard)", "Parkinson (High-Low Range)", "Garman-Klass (OHLC Combined)", "Configured Volatility"],
            "Annualized Vol": [f"{hist_vol * 100:.2f}%", f"{park_vol * 100:.2f}%", f"{gk_vol * 100:.2f}%", f"{market_params.annual_volatility * 100:.2f}%"],
            "Interval σ (tau)": [
                f"{hist_vol / np.sqrt(252*75) * 100:.3f}%",
                f"{park_vol / np.sqrt(252*75) * 100:.3f}%",
                f"{gk_vol / np.sqrt(252*75) * 100:.3f}%",
                f"{market_params.interval_volatility * 100:.3f}%"
            ]
        })
        st.dataframe(vol_df, use_container_width=True, hide_index=True)

        st.info(
            "💡 **Garman-Klass estimator** captures intraday opening jumps and extreme wicks, "
            "providing up to 8x statistical efficiency over simple close-to-close returns."
        )

    with col_curve:
        st.markdown("#### 🌊 Expected Intraday Volume Distribution")
        vol_profile = estimate_intraday_volume_profile(df, num_target_intervals=order.num_intervals)
        
        fig_u = go.Figure()
        fig_u.add_trace(go.Bar(
            x=[f"Int {i+1} ({int((i+1)*order.interval_duration_min)}m)" for i in range(order.num_intervals)],
            y=vol_profile * 100.0,
            marker_color="#AB47BC",
            name="Volume Weight %"
        ))
        fig_u.update_layout(
            template="plotly_dark",
            height=280,
            margin=dict(l=30, r=30, t=30, b=30),
            yaxis_title="% of Day's Volume",
            xaxis_title="Execution Interval"
        )
        st.plotly_chart(fig_u, use_container_width=True)
