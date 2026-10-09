"""
Execution Trajectory and Schedule View Component.
Visualizes remaining inventory x(t), trade slice sizes n_k, trading rates, and theoretical cost metrics.
"""

from typing import Dict
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from config import OrderConfig, MarketParameters, ExecutionSchedule


def render_trajectory_tab(
    schedules: Dict[str, ExecutionSchedule],
    order: OrderConfig,
    market_params: MarketParameters
):
    """
    Renders the Execution Schedule and Trajectory comparison view.
    """
    st.markdown("### 🏹 Optimal Execution Trajectory & Order Schedule")

    ac_sched = schedules.get("Almgren-Chriss (Optimal)")
    twap_sched = schedules.get("TWAP")
    vwap_sched = schedules.get("VWAP")
    imm_sched = schedules.get("Immediate Execution")

    notional = order.total_quantity * order.start_price

    # 1. Top Urgency / Optimal Metric Banner
    if ac_sched:
        kappa = ac_sched.metadata.get("kappa", 0.0)
        half_life = ac_sched.metadata.get("half_life_min", float("inf"))
        hl_str = f"{half_life:.1f} mins" if half_life < 1000 else "N/A (Linear TWAP)"
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Urgency Parameter (κ)", f"{kappa:.4f} min⁻¹")
        c2.metric("Inventory Half-Life", hl_str)
        c3.metric("Optimal Exp. Cost", f"${ac_sched.expected_cost:,.2f}" if "$" in order.symbol or not "NS" in order.symbol else f"₹{ac_sched.expected_cost:,.2f}")
        c4.metric("Exp. Shortfall (bps)", f"{(ac_sched.expected_cost / notional) * 10000:.2f} bps")

    st.markdown("---")

    # 2. Inventory Remaining Trajectory Plot
    col_traj, col_slices = st.columns([1.1, 0.9])

    with col_traj:
        st.markdown("#### 📉 Inventory Depletion Path $x(t)$")
        fig_inv = go.Figure()

        colors = {
            "Almgren-Chriss (Optimal)": "#00E5FF",
            "TWAP": "#FFB300",
            "VWAP": "#E040FB",
            "Immediate Execution": "#FF5252"
        }

        for name, sched in schedules.items():
            dash = "solid" if "Almgren" in name else "dash"
            width = 3.5 if "Almgren" in name else 2.0
            fig_inv.add_trace(go.Scatter(
                x=sched.time_steps_min,
                y=sched.inventory_remaining,
                mode="lines+markers",
                name=name,
                line=dict(color=colors.get(name, "#FFFFFF"), width=width, dash=dash),
                marker=dict(size=6)
            ))

        fig_inv.update_layout(
            template="plotly_dark",
            height=420,
            xaxis_title="Time Elapsed (Minutes)",
            yaxis_title="Remaining Inventory (Shares)",
            margin=dict(l=40, r=40, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_inv, use_container_width=True)

    with col_slices:
        st.markdown("#### 📊 Interval Trade Slices $n_k$")
        fig_bar = go.Figure()

        intervals = [f"Int {k+1}" for k in range(order.num_intervals)]
        
        if ac_sched:
            fig_bar.add_trace(go.Bar(
                x=intervals,
                y=ac_sched.trade_sizes,
                name="Almgren-Chriss",
                marker_color="#00E5FF"
            ))
        if twap_sched:
            fig_bar.add_trace(go.Bar(
                x=intervals,
                y=twap_sched.trade_sizes,
                name="TWAP",
                marker_color="#FFB300"
            ))
        if vwap_sched:
            fig_bar.add_trace(go.Bar(
                x=intervals,
                y=vwap_sched.trade_sizes,
                name="VWAP",
                marker_color="#E040FB"
            ))

        fig_bar.update_layout(
            barmode="group",
            template="plotly_dark",
            height=420,
            xaxis_title="Trading Interval",
            yaxis_title="Shares Traded",
            margin=dict(l=40, r=40, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # 3. Schedule Table & Theoretical Metrics
    st.markdown("#### 📋 Detailed Interval Trade Schedule (Almgren-Chriss Optimal)")
    if ac_sched:
        sched_rows = []
        cum_shares = 0.0
        for k in range(order.num_intervals):
            n_k = ac_sched.trade_sizes[k]
            cum_shares += n_k
            rem_shares = ac_sched.inventory_remaining[k + 1]
            rate = ac_sched.trading_rates[k]
            time_start = k * order.interval_duration_min
            time_end = (k + 1) * order.interval_duration_min

            sched_rows.append({
                "Interval": f"Interval {k+1}",
                "Time Window (min)": f"{time_start:.1f} - {time_end:.1f}m",
                "Trade Slice (Shares)": f"{n_k:,.1f}",
                "Trade Rate (shares/min)": f"{rate:,.1f}",
                "Cumulative Done": f"{cum_shares:,.1f}",
                "% Completed": f"{(cum_shares / order.total_quantity) * 100:.1f}%",
                "Remaining Shares": f"{rem_shares:,.1f}"
            })

        st.dataframe(pd.DataFrame(sched_rows), use_container_width=True, hide_index=True)
