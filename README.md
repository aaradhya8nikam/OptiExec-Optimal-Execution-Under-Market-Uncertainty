# Optimal Execution Engine (Almgren–Chriss Framework)

A quantitative finance execution framework and research simulation platform designed to solve the institutional large order execution problem: **minimizing market impact and timing volatility risk**.

---

## Project Architecture & File Hierarchy

```
TRADING Project/
├── config.py                         # Enums and dataclasses (OrderConfig, MarketParameters, ExecutionSchedule)
├── requirements.txt                  # Python dependencies
├── run_dashboard.py                  # One-click Streamlit dashboard launcher
├── data/
│   ├── __init__.py
│   ├── market_data.py                # OHLCV data loader, synthetic generator, U-shape intraday volume profile
│   └── sample_data_generator.py      # Pre-caching scripts for sample equities
├── analytics/
│   ├── __init__.py
│   ├── returns_volatility.py         # Parkinson, Garman-Klass, Close-to-Close & rolling volatility estimators
│   └── liquidity_volume.py           # ADV, Order-to-ADV, participation rate, and bid-ask spread analytics
├── models/
│   ├── __init__.py
│   ├── market_impact.py              # Almgren-Chriss linear & power-law temporary/permanent impact models
│   ├── price_process.py              # Stochastic Brownian motion price simulator (ABM & GBM)
│   └── ml_impact.py                  # Machine learning / Ridge impact parameter calibration
├── strategies/
│   ├── __init__.py
│   ├── base.py                       # Base strategy class with theoretical cost/variance metrics
│   ├── immediate.py                  # Immediate Execution benchmark (100% at t=0)
│   ├── twap.py                       # Time-Weighted Average Price (uniform slicing)
│   ├── vwap.py                       # Volume-Weighted Average Price (U-curve volume matching)
│   └── almgren_chriss.py             # Almgren-Chriss closed-form analytical & numerical QP solvers
├── simulation/
│   ├── __init__.py
│   ├── execution_simulator.py        # Path-by-path trade logger with realized slippage & impact
│   └── monte_carlo.py                # Vectorized multi-path Monte Carlo engine (VaR 95%, CVaR)
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                    # Implementation Shortfall ($ and bps), efficiency ratios
│   ├── efficient_frontier.py         # Continuous Cost vs Risk Efficient Frontier generator
│   └── robustness.py                 # Parameter uncertainty & model misspecification stress-test
├── dynamic/
│   ├── __init__.py
│   └── rebalancer.py                 # Dynamic adaptive re-optimization for intraday volatility/liquidity shocks
├── dashboard/
│   ├── __init__.py
│   ├── app.py                        # Master Streamlit quantitative dashboard
│   └── components/
│       ├── __init__.py
│       ├── order_input.py            # Sidebar controls & parameter inputs
│       ├── market_analytics.py       # Candlestick chart & volatility estimators breakdown
│       ├── trajectory_view.py        # Inventory trajectory & interval trade slices view
│       ├── monte_carlo_view.py       # Stochastic price paths & shortfall distribution view
│       ├── efficient_frontier_view.py# Cost-Risk Efficient Frontier interactive plot
│       ├── robustness_view.py        # Parameter uncertainty & stress-test view
│       └── dynamic_exec_view.py      # Mid-trade shock & adaptive rebalancer view
└── tests/
    ├── __init__.py
    ├── test_analytics.py             # Tests for returns, volatility, and volume curves
    ├── test_market_impact.py         # Tests for impact formulas & ML estimator
    ├── test_strategies.py            # Tests for Immediate, TWAP, and VWAP
    ├── test_almgren_chriss.py        # Tests for hyperbolic solver, limits, and QP agreement
    └── test_simulation.py            # Tests for simulator, Monte Carlo, and dynamic rebalancing
```

---

## Mathematical Foundations

### 1. Market Price Dynamics with Permanent Impact
$$S_k = S_{k-1} + \sigma \sqrt{\tau} \xi_k - \text{sign}(\text{side}) \cdot \gamma n_k$$
where $\xi_k \sim \mathcal{N}(0, 1)$ is exogenous price innovation, $\tau = T / N$, and $\gamma$ is permanent price impact.

### 2. Execution Price with Temporary Impact & Half-Spread
$$\tilde{S}_k = S_{k-1} + \text{sign}(\text{side}) \left( \eta \frac{n_k}{\tau} + \frac{1}{2}\text{Spread} \right)$$

### 3. Almgren-Chriss (2000) Optimization Objective
$$\min_{\{x_k\}} U(x) = \mathbb{E}[x] + \lambda \mathbb{V}[x]$$
where:
$$\mathbb{E}[x] = \frac{1}{2}\gamma X^2 + \frac{\eta}{\tau}\sum_{k=1}^N n_k^2 + \frac{1}{2}\text{Spread} \cdot X$$
$$\mathbb{V}[x] = \sigma^2 \tau \sum_{k=1}^N x_k^2$$

### 4. Closed-Form Analytical Trajectory
$$x_j = \frac{\sinh(\kappa (T - t_j))}{\sinh(\kappa T)} X \quad \text{for } j=0, 1, \dots, N$$
where the urgency parameter $\kappa$ satisfies:
$$\cosh(\kappa \tau) = 1 + \frac{\lambda \sigma^2 \tau^2}{2 \eta}$$
- As $\lambda \to 0$ (risk-neutral): $x_j \to X(1 - j/N)$ (**TWAP**).
- As $\lambda \to \infty$ (risk-averse): order is front-loaded into early intervals.

---

## How to Run

### 1. Launch the Modern Web Application (FastAPI + Modern Web Frontend)
```bash
python run_server.py
```
- **Web UI:** [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger REST API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **OpenAPI JSON Schema:** [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

### 2. Run the Full Automated Test Suite (24 Tests)
```bash
python -m pytest -v
```

### 3. Run the Quickstart Terminal Demo
```bash
python quickstart_demo.py
```

### 4. Optional Streamlit Dashboard
```bash
python run_dashboard.py
```

---

## Core Research Finding
**Hypothesis:** *How robust is an optimal execution strategy when volatility, liquidity, and market-impact parameters are estimated imperfectly?*

**Result:** The Almgren-Chriss optimal trajectory maintains superior or competitive risk-adjusted cost efficiency over benchmark TWAP and VWAP across misspecified volatility ($\pm 50\%$) and market impact ($0.5\times$ to $3.0\times$) regimes. By front-loading volume according to the trader's risk tolerance $\lambda$, it limits severe tail losses ($\text{VaR}_{95\%}$) during adverse price drift.
