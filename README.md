# OptiExec — Optimal Execution Under Market-Impact Uncertainty

**A quantitative finance engine for optimizing large-order execution using the Almgren–Chriss framework, stochastic simulation, and risk-aware optimization.**

OptiExec addresses a fundamental problem in institutional trading: how should a large buy or sell order be executed over time to minimize transaction costs while controlling price volatility risk?

Executing an order too aggressively can increase market impact and liquidity costs, while executing too slowly exposes the trader to adverse price movements. OptiExec models this trade-off, generates optimized execution schedules, and evaluates their performance against established benchmark strategies.

## Key Features

* **Optimal Execution:** Analytical and numerical optimization using the Almgren–Chriss framework.
* **Benchmark Strategies:** Compare Immediate Execution, Time-Weighted Average Price (TWAP), and Volume-Weighted Average Price (VWAP).
* **Market Analytics:** Estimate volatility, average daily volume (ADV), participation rates, liquidity, and intraday volume profiles.
* **Market Impact Modeling:** Support temporary and permanent market impact using linear and power-law models.
* **Monte Carlo Simulation:** Evaluate execution strategies across stochastic price paths and analyze execution-cost distributions, Value at Risk (VaR), and Conditional Value at Risk (CVaR).
* **Cost–Risk Efficient Frontier:** Analyze the trade-off between expected execution costs and execution risk under different risk-aversion levels.
* **Robustness Analysis:** Stress-test strategies under uncertain volatility and market-impact estimates.
* **Dynamic Re-optimization:** Adapt execution schedules when market volatility or liquidity changes.
* **Interactive Dashboard:** Visualize execution trajectories, market conditions, simulation outcomes, and strategy comparisons.
* **ML-Based Calibration:** Experiment with Ridge regression to estimate market-impact parameters from market features.

## Project Architecture

```text
OptiExec/
├── config.py
├── requirements.txt
├── run_server.py
├── run_dashboard.py
├── quickstart_demo.py
├── data/
│   ├── market_data.py
│   └── sample_data_generator.py
├── analytics/
│   ├── returns_volatility.py
│   └── liquidity_volume.py
├── models/
│   ├── market_impact.py
│   ├── price_process.py
│   └── ml_impact.py
├── strategies/
│   ├── base.py
│   ├── immediate.py
│   ├── twap.py
│   ├── vwap.py
│   └── almgren_chriss.py
├── simulation/
│   ├── execution_simulator.py
│   └── monte_carlo.py
├── evaluation/
│   ├── metrics.py
│   ├── efficient_frontier.py
│   └── robustness.py
├── dynamic/
│   └── rebalancer.py
├── dashboard/
│   ├── app.py
│   └── components/
└── tests/
```

*The tree highlights the main modules. Refer to the repository for the complete file hierarchy and implementation.*

## Mathematical Framework

### 1. Execution Objective

The Almgren–Chriss framework balances expected execution costs against the risk of holding an unexecuted position.

$$
\min_x \left(
\mathbb{E}[C] + \lambda\operatorname{Var}(C)
\right)
$$

Where:

* \(C\): total execution cost relative to the decision price.
* \(\lambda\): risk-aversion parameter.
* \(X\): initial order quantity.
* \(x_k\): remaining inventory after interval \(k\).
* \(\sigma\): price volatility.
* \(\eta\): temporary market-impact coefficient.
* \(\gamma\): permanent market-impact coefficient.

### 2. Price Dynamics

A simplified arithmetic Brownian motion model with permanent market impact is:

$$
S_k=S_{k-1}+\sigma\sqrt{\tau}\xi_k-\gamma n_k
$$

Here, \(\xi_k\sim\mathcal{N}(0,1)\), \(\tau\) is the interval duration, and \(n_k\) is the signed trade quantity under the chosen impact convention.

### 3. Execution Price

For a buy order, a simplified execution-price model is:

$$
\widetilde S_k=S_{k-1}
+\eta\frac{n_k}{\tau}
+\frac{\text{Spread}}{2}
$$

The model accounts for temporary market impact and half the bid–ask spread. Sell orders require consistent trade-direction and impact sign conventions.

### 4. Optimal Inventory Trajectory

Under the standard continuous-time Almgren–Chriss assumptions, the remaining inventory follows:

$$
x(t)=X\frac{\sinh(\kappa(T-t))}
{\sinh(\kappa T)}
$$

where

$$
\kappa=\sqrt{\frac{\lambda\sigma^2}{\eta}}
$$

for the corresponding continuous-time linear-impact formulation.

The trajectory determines how much inventory remains at each point during execution. Greater risk aversion generally leads to faster execution, while lower risk aversion favors spreading trades over a longer horizon.

*Discrete-time implementations must use equations consistent with their time-step and impact-coefficient conventions.*

## Research Objective

**Research question:** How robust is an optimal execution strategy when volatility, liquidity, and market-impact parameters are estimated imperfectly?

OptiExec investigates this question by comparing optimized execution against TWAP and VWAP under different market conditions and parameter-misspecification scenarios.

The evaluation examines:

* Expected implementation shortfall.
* Execution-cost variance and tail risk.
* Performance under volatility and market-impact estimation errors.
* Sensitivity to order size, execution horizon, and risk aversion.
* The effect of changing market conditions on execution decisions.

The objective is to determine when model-based optimization provides meaningful benefits and when simpler benchmark strategies may be more robust.

## Technology Stack

| Technology   | Purpose                                 |
| ------------ | --------------------------------------- |
| Python       | Core implementation                     |
| NumPy        | Numerical computation and simulation    |
| Pandas       | Market-data processing                  |
| SciPy        | Numerical optimization                  |
| Streamlit    | Interactive dashboard                   |
| Plotly       | Interactive financial visualizations    |
| FastAPI      | REST API, if enabled                    |
| scikit-learn | Ridge regression for impact calibration |
| Pytest       | Automated testing                       |

## Getting Started

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd OptiExec
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Quickstart Demo

```bash
python quickstart_demo.py
```

### 5. Launch the API Server

```bash
python run_server.py
```

If the FastAPI server is configured to use port 8000, access:

* Web application: http://localhost:8000
* API documentation: http://localhost:8000/docs
* OpenAPI schema: http://localhost:8000/openapi.json

### 6. Launch the Streamlit Dashboard

In a separate terminal:

```bash
python run_dashboard.py
```

### 7. Run the Test Suite

```bash
python -m pytest -v
```

The commands assume the corresponding scripts and dependencies are present in the repository.

## Project Status

OptiExec is a quantitative research and simulation project. Its purpose is to investigate execution strategies through mathematical modeling, numerical optimization, and controlled simulation—not to predict stock prices or guarantee profitable trades.

Results should be reported only after running the experiments and validating the implementation. Simulated performance does not establish real-market profitability, and any conclusions depend on the accuracy of the market-impact assumptions and data.

## Future Enhancements

* Calibrate market-impact models using historical execution data.
* Incorporate realistic intraday volume and liquidity profiles.
* Evaluate performance across different market regimes.
* Improve adaptive re-optimization under market shocks.
* Extend the framework to multiple assets and portfolio-level execution.

## Disclaimer

OptiExec is intended for educational and quantitative research purposes. It is not investment advice and does not execute live trades by default.
