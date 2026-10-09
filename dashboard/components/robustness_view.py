"""
Model Uncertainty and Robustness Analysis UI Component.
Evaluates research hypothesis on optimal execution robustness under misspecified parameters.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from config import OrderConfig, MarketParameters
from evaluation.robustness import RobustnessEngine


def render_robustness_tab(order: OrderConfig, market_params: MarketParameters):
    """
    Renders the Model Uncertainty & Parameter Robustness research tab.
    """
    st.markdown("### 🔬 Model Uncertainty & Parameter Robustness Analysis")
    st.markdown(
        "> **Core Research Question:** *How robust is an optimal execution strategy when volatility, "
        "liquidity, and market-impact parameters are estimated imperfectly from historical data?*"
    )

    if st.button("🚀 Run Comprehensive Robustness Stress-Test", type="primary"):
        with st.spinner("Executing parameter perturbation Monte Carlo simulations..."):
            engine = RobustnessEngine(num_paths=300)
            res = engine.run_robustness_analysis(order, market_params)

        # 1. Research Conclusion Banner
        st.success(f"📌 **{res.research_conclusion}**")

        st.markdown("---")

        # 2. Sensitivity Charts (Volatility & Impact Shocks)
        col_vol_sens, col_imp_sens = st.columns(2)

        with col_vol_sens:
            st.markdown("#### ⚡ Volatility Misspecification Stress")
            vol_mults = [e.multiplier for e in res.volatility_experiments]
            ac_vol_costs = [e.ac_mean_cost_bps for e in res.volatility_experiments]
            twap_vol_costs = [e.twap_mean_cost_bps for e in res.volatility_experiments]
            vwap_vol_costs = [e.vwap_mean_cost_bps for e in res.volatility_experiments]

            fig_v = go.Figure()
            fig_v.add_trace(go.Scatter(x=vol_mults, y=ac_vol_costs, mode="lines+markers", name="Almgren-Chriss", line=dict(color="#00E5FF", width=3)))
            fig_v.add_trace(go.Scatter(x=vol_mults, y=twap_vol_costs, mode="lines+markers", name="TWAP", line=dict(color="#FFB300", dash="dash")))
            fig_v.add_trace(go.Scatter(x=vol_mults, y=vwap_vol_costs, mode="lines+markers", name="VWAP", line=dict(color="#E040FB", dash="dot")))

            fig_v.update_layout(
                template="plotly_dark",
                height=380,
                xaxis_title="True Volatility Multiplier (vs Estimated)",
                yaxis_title="Realized Shortfall (bps)",
                margin=dict(l=40, r=40, t=30, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_v, use_container_width=True)

        with col_imp_sens:
            st.markdown("#### 🌊 Market Impact (η) Misspecification Stress")
            imp_mults = [e.multiplier for e in res.impact_experiments]
            ac_imp_costs = [e.ac_mean_cost_bps for e in res.impact_experiments]
            twap_imp_costs = [e.twap_mean_cost_bps for e in res.impact_experiments]
            vwap_imp_costs = [e.vwap_mean_cost_bps for e in res.impact_experiments]

            fig_i = go.Figure()
            fig_i.add_trace(go.Scatter(x=imp_mults, y=ac_imp_costs, mode="lines+markers", name="Almgren-Chriss", line=dict(color="#00E5FF", width=3)))
            fig_i.add_trace(go.Scatter(x=imp_mults, y=twap_imp_costs, mode="lines+markers", name="TWAP", line=dict(color="#FFB300", dash="dash")))
            fig_i.add_trace(go.Scatter(x=imp_mults, y=vwap_imp_costs, mode="lines+markers", name="VWAP", line=dict(color="#E040FB", dash="dot")))

            fig_i.update_layout(
                template="plotly_dark",
                height=380,
                xaxis_title="True Impact Multiplier (vs Estimated)",
                yaxis_title="Realized Shortfall (bps)",
                margin=dict(l=40, r=40, t=30, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_i, use_container_width=True)

        # 3. Full Stress-Test Summary Table
        st.markdown("#### 📋 Comprehensive Stress-Test Realization Table")
        st.dataframe(res.summary_matrix, use_container_width=True, hide_index=True)
    else:
        st.info("Click the button above to execute the multi-scenario parameter uncertainty simulation.")
