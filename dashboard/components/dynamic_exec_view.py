"""
Dynamic Adaptive Re-Optimization UI Component.
Simulates mid-execution market shocks and visualizes live trajectory adaptation.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from config import OrderConfig, MarketParameters
from dynamic.rebalancer import DynamicExecutionEngine


def render_dynamic_exec_tab(order: OrderConfig, market_params: MarketParameters):
    """
    Renders the Dynamic Adaptive Re-optimization and intraday shock simulation tab.
    """
    st.markdown("### 🔄 Dynamic Adaptive Re-Optimization Engine")
    st.caption(
        "Standard execution models compute a static schedule at t=0. "
        "The Dynamic Engine monitors live market conditions and re-solves the optimal trajectory "
        "for the remaining inventory when volatility spikes or liquidity vanishes."
    )

    # 1. Shock Controls
    col_k, col_v, col_l = st.columns(3)
    with col_k:
        shock_step = st.slider(
            "Shock Trigger Interval",
            min_value=1,
            max_value=max(2, order.num_intervals - 1),
            value=max(1, order.num_intervals // 3)
        )
    with col_v:
        vol_multiplier = st.slider("Volatility Multiplier (Shock)", min_value=0.5, max_value=3.0, value=2.0, step=0.1)
    with col_l:
        liq_multiplier = st.slider("Liquidity Multiplier (ADV)", min_value=0.2, max_value=1.5, value=0.5, step=0.1)

    # Execute dynamic shock simulation
    dyn_engine = DynamicExecutionEngine()
    dyn_res = dyn_engine.simulate_dynamic_shock(
        order=order,
        initial_params=market_params,
        shock_step=shock_step,
        volatility_multiplier=vol_multiplier,
        liquidity_multiplier=liq_multiplier
    )

    # 2. Urgency Shift Diagnosis Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Shock Step", f"Interval {dyn_res.rebalance_step} ({dyn_res.rebalance_time_min:.1f}m)")
    c2.metric("Pre-Shock κ", f"{dyn_res.old_kappa:.4f}")
    c3.metric("Post-Shock κ", f"{dyn_res.new_kappa:.4f}")
    c4.metric("Urgency Adaptation", dyn_res.urgency_shift.split("(")[0].strip())

    st.info(f"⚡ **Regime Shift Verdict:** {dyn_res.urgency_shift}")

    # 3. Visual Comparison: Static vs Dynamic Inventory Trajectory
    st.markdown("#### 📉 Trajectory Adaptation: Static vs Adaptive Execution")
    fig_dyn = go.Figure()

    time_steps = np.linspace(0, order.time_horizon_min, order.num_intervals + 1)
    orig_inv = dyn_res.original_schedule.inventory_remaining
    dyn_inv = dyn_res.combined_trajectory

    fig_dyn.add_trace(go.Scatter(
        x=time_steps,
        y=orig_inv,
        mode="lines+markers",
        name="Static Frozen Plan (No Adaptation)",
        line=dict(color="#FF5252", dash="dash", width=2.5)
    ))

    fig_dyn.add_trace(go.Scatter(
        x=time_steps,
        y=dyn_inv,
        mode="lines+markers",
        name="Dynamic Adaptive Trajectory",
        line=dict(color="#00E676", width=3.5)
    ))

    # Add vertical shock line
    fig_dyn.add_vline(
        x=dyn_res.rebalance_time_min,
        line_dash="dot",
        line_color="#FFEB3B",
        annotation_text="Shock Occurred ⚠️",
        annotation_position="top left"
    )

    fig_dyn.update_layout(
        template="plotly_dark",
        height=420,
        xaxis_title="Time Elapsed (Minutes)",
        yaxis_title="Remaining Inventory (Shares)",
        margin=dict(l=40, r=40, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_dyn, use_container_width=True)

    # 4. Sliced Trade Sizes Comparison
    st.markdown("#### 📊 Trade Slices: Static vs Dynamic")
    fig_slices = go.Figure()
    intervals = [f"Int {k+1}" for k in range(order.num_intervals)]

    fig_slices.add_trace(go.Bar(
        x=intervals,
        y=dyn_res.original_schedule.trade_sizes,
        name="Static Original Trade Slices",
        marker_color="rgba(255, 82, 82, 0.7)"
    ))

    fig_slices.add_trace(go.Bar(
        x=intervals,
        y=dyn_res.combined_trades,
        name="Dynamic Adaptive Trade Slices",
        marker_color="rgba(0, 230, 118, 0.85)"
    ))

    fig_slices.update_layout(
        barmode="group",
        template="plotly_dark",
        height=380,
        xaxis_title="Trading Interval",
        yaxis_title="Shares Traded",
        margin=dict(l=40, r=40, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_slices, use_container_width=True)
