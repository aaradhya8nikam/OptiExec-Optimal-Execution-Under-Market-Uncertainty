"""
Monte Carlo Simulation Results and Risk Analysis UI Component.
Visualizes stochastic price paths, shortfall distributions, tail risk (VaR 95%, CVaR), and boxplots.
"""

from typing import Dict
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from config import OrderConfig, MarketParameters, ExecutionSchedule
from simulation.monte_carlo import MonteCarloSimulationResult
from evaluation.metrics import compute_strategy_comparison_table


def render_monte_carlo_tab(
    mc_result: MonteCarloSimulationResult,
    schedules: Dict[str, ExecutionSchedule],
    order: OrderConfig
):
    """
    Renders the Monte Carlo Simulation and Cost vs Risk Distribution tab.
    """
    st.markdown("### 🎲 Monte Carlo Stochastic Simulation & Tail Risk Analysis")
    st.caption(
        f"Simulated {mc_result.sim_config.num_paths:,} stochastic market paths driven by Brownian motion "
        f"with endogenous permanent price impact and instantaneous temporary liquidity consumption."
    )

    summaries = mc_result.strategy_summaries

    # 1. High-Level Summary Comparison Table
    st.markdown("#### 📊 Comparative Performance Matrix")
    comp_df = compute_strategy_comparison_table(summaries, schedules)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 2. Stochastic Price Paths & Implementation Shortfall Histogram
    col_paths, col_hist = st.columns(2)

    with col_paths:
        st.markdown("#### 📉 Simulated Market Price Bundles")
        fig_paths = go.Figure()

        paths = mc_result.exogenous_price_paths
        num_display = min(50, len(paths))
        time_axis = np.linspace(0, order.time_horizon_min, order.num_intervals + 1)

        for i in range(num_display):
            fig_paths.add_trace(go.Scatter(
                x=time_axis,
                y=paths[i],
                mode="lines",
                line=dict(width=0.8, color="rgba(0, 229, 255, 0.25)"),
                showlegend=False,
                hoverinfo="skip"
            ))

        # Add Mean path
        mean_path = np.mean(paths, axis=0)
        fig_paths.add_trace(go.Scatter(
            x=time_axis,
            y=mean_path,
            mode="lines",
            line=dict(width=3.0, color="#FFEB3B", dash="dash"),
            name="Expected Price Path"
        ))

        fig_paths.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Time Elapsed (Minutes)",
            yaxis_title="Stock Price ($/₹)",
            margin=dict(l=40, r=40, t=30, b=40)
        )
        st.plotly_chart(fig_paths, use_container_width=True)

    with col_hist:
        st.markdown("#### 🎯 Implementation Shortfall Distribution (bps)")
        fig_hist = go.Figure()

        colors = {
            "Almgren-Chriss (Optimal)": "#00E5FF",
            "TWAP": "#FFB300",
            "VWAP": "#E040FB",
            "Immediate Execution": "#FF5252"
        }

        for name, summary in summaries.items():
            if name != "Immediate Execution":  # Keep histogram readable by omitting extreme outlier
                fig_hist.add_trace(go.Histogram(
                    x=summary.all_shortfalls_bps,
                    name=name,
                    opacity=0.6,
                    marker_color=colors.get(name, "#FFFFFF"),
                    nbinsx=40
                ))

        fig_hist.update_layout(
            barmode="overlay",
            template="plotly_dark",
            height=400,
            xaxis_title="Implementation Shortfall (basis points)",
            yaxis_title="Frequency Count",
            margin=dict(l=40, r=40, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # 3. Execution Risk Boxplots & Tail Risk Comparison
    st.markdown("#### 📦 Execution Cost Dispersion & Tail Risk (VaR 95% / CVaR)")
    
    # Prepare boxplot dataframe
    box_data = []
    for name, summary in summaries.items():
        for val in summary.all_shortfalls_bps:
            box_data.append({"Strategy": name, "Shortfall (bps)": val})

    box_df = pd.DataFrame(box_data)
    
    fig_box = px.box(
        box_df,
        x="Strategy",
        y="Shortfall (bps)",
        color="Strategy",
        color_discrete_map=colors,
        template="plotly_dark"
    )
    fig_box.update_layout(
        height=380,
        margin=dict(l=40, r=40, t=30, b=40),
        showlegend=False
    )
    st.plotly_chart(fig_box, use_container_width=True)
