"""
Risk-Cost Efficient Frontier UI Component.
Visualizes the continuous trade-off between Expected Impact Cost E[x] and Volatility Risk Std(x).
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from config import OrderConfig, MarketParameters
from evaluation.efficient_frontier import EfficientFrontierEngine


def render_efficient_frontier_tab(order: OrderConfig, market_params: MarketParameters):
    """
    Renders the interactive Almgren-Chriss Efficient Frontier visualization.
    """
    st.markdown("### 🌐 Almgren-Chriss Risk-Cost Efficient Frontier")
    st.caption(
        "The Efficient Frontier illustrates the fundamental quantitative trade-off: "
        "trading faster reduces volatility risk but incurs heavy market impact, "
        "while trading slower reduces impact costs but exposes the order to market drift."
    )

    # Compute frontier
    ef_engine = EfficientFrontierEngine(num_points=40)
    ef_res = ef_engine.generate_frontier(order, market_params)

    # Extract coordinates
    frontier_cost_bps = [p.expected_cost_bps for p in ef_res.frontier_points]
    frontier_std_bps = [p.expected_std_bps for p in ef_res.frontier_points]
    kappas = [p.kappa for p in ef_res.frontier_points]

    fig_ef = go.Figure()

    # 1. Continuous Frontier Curve
    fig_ef.add_trace(go.Scatter(
        x=frontier_std_bps,
        y=frontier_cost_bps,
        mode="lines+markers",
        name="Optimal Frontier (Almgren-Chriss)",
        line=dict(color="#00E5FF", width=3),
        marker=dict(
            size=6,
            color=kappas,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=dict(text="Urgency (κ)"))
        ),
        text=[f"λ = {p.risk_aversion:.1e}<br>κ = {p.kappa:.4f}<br>Cost: {p.expected_cost_bps:.2f} bps<br>Risk: {p.expected_std_bps:.2f} bps" for p in ef_res.frontier_points],
        hoverinfo="text"
    ))

    # 2. Selected User Strategy
    u = ef_res.user_point
    fig_ef.add_trace(go.Scatter(
        x=[u.expected_std_bps],
        y=[u.expected_cost_bps],
        mode="markers+text",
        name="Current Selected Policy (λ)",
        marker=dict(symbol="star", size=18, color="#FFEA00", line=dict(color="#000", width=1.5)),
        text=["Selected Policy ⭐"],
        textposition="top center"
    ))

    # 3. Benchmark Strategies
    bench_colors = {
        "TWAP": "#FFB300",
        "VWAP": "#E040FB",
        "Immediate Execution": "#FF5252"
    }

    for name, coords in ef_res.benchmark_points.items():
        if name != "Immediate Execution":  # Keep axis in readable range
            fig_ef.add_trace(go.Scatter(
                x=[coords["expected_std_bps"]],
                y=[coords["expected_cost_bps"]],
                mode="markers+text",
                name=f"Benchmark: {name}",
                marker=dict(size=12, color=bench_colors.get(name, "#FFFFFF")),
                text=[name],
                textposition="bottom right"
            ))

    fig_ef.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title="Execution Volatility Risk Std(Cost) [basis points]",
        yaxis_title="Expected Execution Cost E(Cost) [basis points]",
        margin=dict(l=40, r=40, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_ef, use_container_width=True)

    # 4. Explanatory Insights Card
    st.markdown("#### 🔍 Frontier Interpretations")
    c1, c2 = st.columns(2)
    with c1:
        st.success(
            "🟢 **Left End of Frontier (High Urgency / High λ)**:\n\n"
            "- Focuses on minimizing price volatility uncertainty.\n"
            "- Front-loads volume aggressively into early intervals.\n"
            "- Recommended during high-volatility regimes or impending macro announcements."
        )
    with c2:
        st.info(
            "🟡 **Right End of Frontier (Low Urgency / Risk-Neutral λ -> 0)**:\n\n"
            "- Minimizes temporary and permanent impact footprint.\n"
            "- Slices order evenly over time (converges to TWAP).\n"
            "- Recommended in calm, low-volatility, or highly liquid markets."
        )
