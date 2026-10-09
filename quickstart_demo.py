import sys
import os

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
from config import OrderConfig, MarketParameters, OrderSide, SimulationConfig
from data.market_data import MarketDataLoader
from strategies.almgren_chriss import AlmgrenChrissExecutionStrategy
from strategies.twap import TWAPExecutionStrategy
from strategies.vwap import VWAPExecutionStrategy
from strategies.immediate import ImmediateExecutionStrategy
from simulation.monte_carlo import MonteCarloEngine
from evaluation.metrics import compute_strategy_comparison_table
from dynamic.rebalancer import DynamicExecutionEngine


def run_demo():
    print("=" * 80)
    print(" [*] OPTIMAL EXECUTION ENGINE - INSTITUTIONAL DEMO WALKTHROUGH")
    print("=" * 80)

    # ---------------------------------------------------------
    # STEP 1: Define Institutional Large Order
    # ---------------------------------------------------------
    print("\n[1] Submitting Institutional Order:")
    order = OrderConfig(
        symbol="RELIANCE.NS",
        side=OrderSide.BUY,
        total_quantity=100_000.0,       # 100,000 shares
        start_price=1_350.0,            # Initial decision price S_0
        time_horizon_min=60.0,          # 60 minute execution window (T)
        num_intervals=12,               # 12 intervals (tau = 5 mins each)
        risk_aversion=1e-6              # Risk aversion lambda (λ)
    )
    notional = order.total_quantity * order.start_price
    print(f"    • Asset Symbol    : {order.symbol}")
    print(f"    • Direction       : {order.side.value}")
    print(f"    • Quantity        : {order.total_quantity:,.0f} shares")
    print(f"    • Decision Price  : ₹{order.start_price:,.2f}")
    print(f"    • Order Notional  : ₹{notional:,.2f}")
    print(f"    • Horizon / Bins  : {order.time_horizon_min:.0f} mins across {order.num_intervals} intervals ({order.interval_duration_min:.0f}m per bin)")
    print(f"    • Risk Aversion λ : {order.risk_aversion:.1e}")

    # ---------------------------------------------------------
    # STEP 2: Ingest Market Data & Estimate Microstructure
    # ---------------------------------------------------------
    print("\n[2] Ingesting Market Data & Estimating Microstructure Parameters:")
    loader = MarketDataLoader()
    df = loader.get_market_data(order.symbol, days=30)
    
    annual_vol = 0.22
    adv = 7_500_000.0
    daily_vol = annual_vol / np.sqrt(252.0)
    interval_vol = annual_vol * np.sqrt(order.interval_duration_min / (252.0 * 375.0))
    
    market_params = MarketParameters(
        symbol=order.symbol,
        annual_volatility=annual_vol,
        daily_volatility=daily_vol,
        interval_volatility=interval_vol,
        adv=adv,
        half_spread_bps=2.0,
        temp_impact_coef=2.5e-7,
        perm_impact_coef=5.0e-8
    )

    part_rate = (order.total_quantity / (adv * (order.time_horizon_min / 375.0))) * 100.0
    print(f"    • Annual Volatility (σ) : {market_params.annual_volatility * 100:.1f}%")
    print(f"    • 5-min Interval Vol (σ): {market_params.interval_volatility * 100:.3f}%")
    print(f"    • Average Daily Volume  : {market_params.adv:,.0f} shares")
    print(f"    • Half Bid-Ask Spread   : {market_params.half_spread_bps:.1f} bps")
    print(f"    • Participation Rate    : {part_rate:.2f}% of market volume")
    print(f"    • Temp Impact Param (η) : {market_params.temp_impact_coef:.3e}")
    print(f"    • Perm Impact Param (γ) : {market_params.perm_impact_coef:.3e}")

    # ---------------------------------------------------------
    # STEP 3: Generate Execution Strategies (Almgren-Chriss, TWAP, VWAP)
    # ---------------------------------------------------------
    print("\n[3] Generating Optimal & Benchmark Execution Schedules:")
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

    kappa = ac_sched.metadata.get("kappa", 0.0)
    half_life = ac_sched.metadata.get("half_life_min", 0.0)
    print(f"    • Almgren-Chriss Urgency κ : {kappa:.4f} min⁻¹ (Inventory Half-life: {half_life:.1f} mins)")
    print("\n    Optimal Interval Trade Slices (n_k):")
    for k in range(order.num_intervals):
        time_m = (k + 1) * order.interval_duration_min
        print(f"      - Int {k+1:02d} ({time_m:02.0f}m): AC={ac_sched.trade_sizes[k]:8,.0f} shs | TWAP={twap_sched.trade_sizes[k]:8,.0f} shs | Rem Inventory={ac_sched.inventory_remaining[k+1]:8,.0f} shs")

    # ---------------------------------------------------------
    # STEP 4: Run Multi-Path Monte Carlo Simulation (500 Paths)
    # ---------------------------------------------------------
    print("\n[4] Running Vectorized Monte Carlo Simulation (500 Stochastic Paths)...")
    mc_engine = MonteCarloEngine()
    sim_config = SimulationConfig(num_paths=500, random_seed=42)
    mc_result = mc_engine.run_simulations(order, market_params, list(schedules.values()), sim_config)

    print("\n[5] Performance & Risk Comparison (Implementation Shortfall):")
    comp_df = compute_strategy_comparison_table(mc_result.strategy_summaries, schedules)
    print(comp_df.to_string(index=False))

    # ---------------------------------------------------------
    # STEP 6: Dynamic Adaptive Re-Optimization on Market Shock
    # ---------------------------------------------------------
    print("\n[6] Simulating Mid-Trade Shock & Dynamic Adaptive Re-Optimization:")
    print("    Scenario: At Interval 4 (20 mins in), sudden earnings news causes Volatility +100% and Liquidity -50%!")
    
    dyn_engine = DynamicExecutionEngine()
    dyn_res = dyn_engine.simulate_dynamic_shock(
        order=order,
        initial_params=market_params,
        shock_step=4,
        volatility_multiplier=2.0,
        liquidity_multiplier=0.5
    )

    print(f"    • Pre-Shock Urgency κ  : {dyn_res.old_kappa:.4f} min⁻¹")
    print(f"    • Post-Shock Urgency κ : {dyn_res.new_kappa:.4f} min⁻¹")
    print(f"    • Adaptation Verdict   : {dyn_res.urgency_shift}")
    print(f"    • Re-optimized Remaining Trades: {np.round(dyn_res.rebalanced_schedule.trade_sizes, 0)}")

    print("\n" + "=" * 80)
    print(" ✅ DEMO COMPLETE! Launch interactive dashboard with: python run_dashboard.py")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
