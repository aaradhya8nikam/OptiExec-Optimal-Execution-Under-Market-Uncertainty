"""
Optimal Execution Engine - Streamlit Master Quantitative Dashboard.
Integrates Almgren-Chriss optimization, TWAP, VWAP, Monte Carlo simulation,
Efficient Frontier, Robustness Analysis, and Dynamic Adaptive Execution.
"""

import sys
import os
# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
import pandas as pd
from config import OrderConfig, MarketParameters, SimulationConfig
from data.market_data import MarketDataLoader
from strategies.almgren_chriss import AlmgrenChrissExecutionStrategy
from strategies.twap import TWAPExecutionStrategy
from strategies.vwap import VWAPExecutionStrategy
from strategies.immediate import ImmediateExecutionStrategy
from simulation.monte_carlo import MonteCarloEngine

# UI Components
from dashboard.components.order_input import render_order_sidebar
from dashboard.components.market_analytics import render_market_analytics_tab
from dashboard.components.trajectory_view import render_trajectory_tab
from dashboard.components.monte_carlo_view import render_monte_carlo_tab
from dashboard.components.efficient_frontier_view import render_efficient_frontier_tab
from dashboard.components.robustness_view import render_robustness_tab
from dashboard.components.dynamic_exec_view import render_dynamic_exec_tab


# Page Configuration
st.set_page_config(
    page_title="Optimal Execution Engine | Almgren-Chriss",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Institutional CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00E5FF, #7C4DFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #B0BEC5;
        margin-bottom: 1.5rem;
    }
    .stMetric {
        background-color: #1E222D;
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid #2A2E39;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E222D;
        border-radius: 6px 6px 0 0;
        padding: 8px 18px;
        color: #CFD8DC;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2979FF !important;
        color: #FFFFFF !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_cached_market_data(symbol: str):
    loader = MarketDataLoader()
    return loader.get_market_data(symbol=symbol, days=30)


@st.cache_data
def run_cached_monte_carlo(
    _order_dict: dict,
    _market_dict: dict,
    num_paths: int
):
    # Reconstruct objects
    order = OrderConfig(**_order_dict)
    market_params = MarketParameters(**_market_dict)

    # Generate schedules
    ac = AlmgrenChrissExecutionStrategy().generate_schedule(order, market_params)
    twap = TWAPExecutionStrategy().generate_schedule(order, market_params)
    vwap = VWAPExecutionStrategy().generate_schedule(order, market_params)
    imm = ImmediateExecutionStrategy().generate_schedule(order, market_params)

    schedules = [ac, twap, vwap, imm]

    mc_engine = MonteCarloEngine()
    sim_config = SimulationConfig(num_paths=num_paths, random_seed=42)
    mc_res = mc_engine.run_simulations(order, market_params, schedules, sim_config)
    
    sched_map = {s.strategy_name: s for s in schedules}
    return mc_res, sched_map


def main():
    # 1. Header Banner
    st.markdown('<div class="main-header">⚡ OPTIMAL EXECUTION ENGINE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Institutional Algorithmic Trading & Optimal Order Slicing '
        '(Almgren–Chriss Framework · TWAP · VWAP · Monte Carlo · Robustness)</div>',
        unsafe_allow_html=True
    )

    # 2. Sidebar Parameters
    order, market_params, ui_settings = render_order_sidebar()

    # 3. Ingest Market Data
    df = get_cached_market_data(order.symbol)

    # 4. Compute Strategies
    ac_strat = AlmgrenChrissExecutionStrategy()
    twap_strat = TWAPExecutionStrategy()
    vwap_strat = VWAPExecutionStrategy()
    imm_strat = ImmediateExecutionStrategy()

    ac_sched = ac_strat.generate_schedule(order, market_params)
    twap_sched = twap_strat.generate_schedule(order, market_params)
    vwap_sched = vwap_strat.generate_schedule(order, market_params)
    imm_sched = imm_strat.generate_schedule(order, market_params)

    schedules = {
        ac_sched.strategy_name: ac_sched,
        twap_sched.strategy_name: twap_sched,
        vwap_sched.strategy_name: vwap_sched,
        imm_sched.strategy_name: imm_sched
    }

    # 5. Run Monte Carlo Simulation
    order_dict = {
        "symbol": order.symbol,
        "side": order.side,
        "total_quantity": order.total_quantity,
        "start_price": order.start_price,
        "time_horizon_min": order.time_horizon_min,
        "num_intervals": order.num_intervals,
        "risk_aversion": order.risk_aversion
    }
    market_dict = {
        "symbol": market_params.symbol,
        "annual_volatility": market_params.annual_volatility,
        "daily_volatility": market_params.daily_volatility,
        "interval_volatility": market_params.interval_volatility,
        "adv": market_params.adv,
        "half_spread_bps": market_params.half_spread_bps,
        "temp_impact_coef": market_params.temp_impact_coef,
        "perm_impact_coef": market_params.perm_impact_coef
    }

    mc_result, _ = run_cached_monte_carlo(order_dict, market_dict, ui_settings["num_mc_paths"])

    # 6. Navigation Tabs
    tabs = st.tabs([
        "📊 Market Conditions",
        "🏹 Optimal Trajectory",
        "🎲 Monte Carlo & Risk",
        "🌐 Efficient Frontier",
        "🔬 Robustness Analysis",
        "🔄 Dynamic Rebalancing",
        "📚 Methodology & Math"
    ])

    with tabs[0]:
        render_market_analytics_tab(df, order, market_params)

    with tabs[1]:
        render_trajectory_tab(schedules, order, market_params)

    with tabs[2]:
        render_monte_carlo_tab(mc_result, schedules, order)

    with tabs[3]:
        render_efficient_frontier_tab(order, market_params)

    with tabs[4]:
        render_robustness_tab(order, market_params)

    with tabs[5]:
        render_dynamic_exec_tab(order, market_params)

    with tabs[6]:
        st.markdown("### 📚 Mathematical Foundation: Almgren-Chriss (2000)")
        st.markdown(r"""
        #### 1. Price Dynamics with Permanent Impact
        $$S_k = S_{k-1} + \sigma \tau^{1/2} \xi_k - \tau \gamma(v_k)$$
        where $\xi_k \sim \mathcal{N}(0, 1)$ is exogenous price volatility and $\gamma$ is permanent price impact.

        #### 2. Execution Price with Temporary Impact
        $$\tilde{S}_k = S_{k-1} - \eta(v_k) - \frac{1}{2}\text{Spread}$$
        where $\eta(v_k) = \eta \frac{n_k}{\tau}$ is the temporary liquidity friction.

        #### 3. Expected Total Cost & Variance
        $$\mathbb{E}[x] = \frac{1}{2}\gamma X^2 + \frac{\eta}{\tau}\sum_{k=1}^N n_k^2 + \frac{1}{2}\text{Spread} \cdot X$$
        $$\mathbb{V}[x] = \sigma^2 \tau \sum_{k=1}^N x_k^2$$

        #### 4. Optimal Inventory Trajectory
        $$\min_{\{x_k\}} U(x) = \mathbb{E}[x] + \lambda \mathbb{V}[x]$$
        The analytical discrete solution is given by:
        $$x_j = \frac{\sinh(\kappa (T - t_j))}{\sinh(\kappa T)} X$$
        where the urgency parameter $\kappa$ satisfies:
        $$\cosh(\kappa \tau) = 1 + \frac{\lambda \sigma^2 \tau^2}{2 \eta}$$
        """)


if __name__ == "__main__":
    main()
