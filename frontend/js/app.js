/**
 * Optimal Execution Engine - Client-Side Controller
 * Interfaces with FastAPI endpoints and renders interactive Chart.js visualizations.
 */

// State
let appState = {
  order: null,
  marketParams: null,
  schedules: null,
  monteCarlo: null,
  efficientFrontier: null,
  charts: {}
};

const API_BASE = "";

// Initialize App
document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupEventListeners();
  loadAssetPresets();
  triggerFullOptimization();
});

// Setup Tab Navigation
function setupTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(btn => {
    btn.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".view-pane").forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add("active");

      // Trigger resize for charts in newly active tab
      Object.values(appState.charts).forEach(c => c && c.resize && c.resize());
    });
  });
}

// Event Listeners for inputs
function setupEventListeners() {
  document.getElementById("btn-optimize").addEventListener("click", () => {
    triggerFullOptimization();
  });

  document.getElementById("asset-select").addEventListener("change", (e) => {
    const symbol = e.target.value;
    loadMarketDataForAsset(symbol);
  });

  document.getElementById("risk-slider").addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    const lambdaVal = Math.pow(10, val);
    document.getElementById("lambda-display").innerText = lambdaVal.toExponential(1);
  });

  document.getElementById("btn-run-shock").addEventListener("click", () => {
    runDynamicShock();
  });

  document.getElementById("btn-run-robustness").addEventListener("click", () => {
    runRobustnessTest();
  });
}

// Ingest Form Values
function getFormValues() {
  const symbol = document.getElementById("asset-select").value;
  const side = document.getElementById("side-select").value;
  const totalQuantity = parseFloat(document.getElementById("order-qty").value);
  const startPrice = parseFloat(document.getElementById("order-price").value);
  const timeHorizon = parseFloat(document.getElementById("order-horizon").value);
  const numIntervals = parseInt(document.getElementById("order-intervals").value);

  const riskSliderVal = parseFloat(document.getElementById("risk-slider").value);
  const riskAversion = Math.pow(10, riskSliderVal);

  const annualVol = parseFloat(document.getElementById("market-vol").value);
  const adv = parseFloat(document.getElementById("market-adv").value);
  const halfSpreadBps = parseFloat(document.getElementById("market-spread").value);

  return {
    order: {
      symbol,
      side,
      total_quantity: totalQuantity,
      start_price: startPrice,
      time_horizon_min: timeHorizon,
      num_intervals: numIntervals,
      risk_aversion: riskAversion
    },
    market_params: {
      symbol,
      annual_volatility: annualVol,
      adv: adv,
      half_spread_bps: halfSpreadBps
    }
  };
}

// Fetch Preset Assets
async function loadAssetPresets() {
  try {
    const res = await fetch(`${API_BASE}/api/assets`);
    const data = await res.json();
    const select = document.getElementById("asset-select");
    select.innerHTML = "";
    data.assets.forEach(a => {
      const opt = document.createElement("option");
      opt.value = a.symbol;
      opt.text = a.symbol;
      select.appendChild(opt);
    });
  } catch (err) {
    console.error("Failed to load preset assets:", err);
  }
}

// Load Asset Market Data
async function loadMarketDataForAsset(symbol) {
  try {
    const res = await fetch(`${API_BASE}/api/market-data/${symbol}`);
    const data = await res.json();
    
    // Update inputs
    if (data.bars && data.bars.length > 0) {
      const lastBar = data.bars[data.bars.length - 1];
      document.getElementById("order-price").value = lastBar.close;
    }
    if (data.volatility_estimators) {
      document.getElementById("market-vol").value = data.volatility_estimators.garman_klass.toFixed(3);
    }
    
    renderMarketDataCharts(data);
  } catch (err) {
    console.error("Failed to fetch market data:", err);
  }
}

// Main Optimization Execution
async function triggerFullOptimization() {
  const { order, market_params } = getFormValues();

  // 1. Calculate Schedules
  try {
    const schedRes = await fetch(`${API_BASE}/api/execute/schedules`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ order_req: order, mkt_req: market_params })
    });
    const schedData = await schedRes.json();
    appState.schedules = schedData.schedules;
    renderTopMetrics(order, schedData);
    renderTrajectoryCharts(schedData.schedules);
    renderScheduleTable(schedData.schedules["Almgren-Chriss (Optimal)"], order);
  } catch (err) {
    console.error("Schedule computation error:", err);
  }

  // 2. Run Monte Carlo Simulation
  try {
    const mcRes = await fetch(`${API_BASE}/api/simulate/monte-carlo`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        order,
        market_params,
        num_paths: 500,
        random_seed: 42
      })
    });
    const mcData = await mcRes.json();
    appState.monteCarlo = mcData;
    renderMonteCarloView(mcData);
  } catch (err) {
    console.error("Monte Carlo error:", err);
  }

  // 3. Generate Efficient Frontier
  try {
    const efRes = await fetch(`${API_BASE}/api/evaluate/efficient-frontier`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ order_req: order, mkt_req: market_params })
    });
    const efData = await efRes.json();
    appState.efficientFrontier = efData;
    renderEfficientFrontierChart(efData);
  } catch (err) {
    console.error("Efficient Frontier error:", err);
  }

  // Load market analytics
  loadMarketDataForAsset(order.symbol);
}

// Render Top Summary Metric Cards
function renderTopMetrics(order, schedData) {
  const notional = order.total_quantity * order.start_price;
  const ac = schedData.schedules["Almgren-Chriss (Optimal)"];
  const kappa = ac.metadata.kappa || 0.0;
  const halfLife = ac.metadata.half_life_min || 0.0;

  document.getElementById("metric-notional").innerText = `$${notional.toLocaleString('en-US', {maximumFractionDigits: 0})}`;
  document.getElementById("metric-urgency").innerText = `${kappa.toFixed(4)} min⁻¹`;
  document.getElementById("metric-halflife").innerText = halfLife < 1000 ? `${halfLife.toFixed(1)} mins` : "Linear";
  document.getElementById("metric-expcost").innerText = `$${ac.expected_cost.toLocaleString('en-US', {maximumFractionDigits: 2})}`;
  document.getElementById("metric-expcost-bps").innerText = `${ac.expected_cost_bps.toFixed(2)} bps`;
}

// Render Trajectory Charts
function renderTrajectoryCharts(schedules) {
  const timeLabels = schedules["TWAP"].time_steps_min.map(t => `${t.toFixed(0)}m`);
  const intervalLabels = Array.from({length: schedules["TWAP"].trade_sizes.length}, (_, i) => `Int ${i+1}`);

  // 1. Inventory Chart
  const ctxInv = document.getElementById("chart-inventory").getContext("2d");
  if (appState.charts.inventory) appState.charts.inventory.destroy();

  appState.charts.inventory = new Chart(ctxInv, {
    type: "line",
    data: {
      labels: timeLabels,
      datasets: [
        {
          label: "Almgren-Chriss (Optimal)",
          data: schedules["Almgren-Chriss (Optimal)"].inventory_remaining,
          borderColor: "#00E5FF",
          backgroundColor: "rgba(0, 229, 255, 0.1)",
          borderWidth: 3,
          tension: 0.2
        },
        {
          label: "TWAP",
          data: schedules["TWAP"].inventory_remaining,
          borderColor: "#FFB300",
          borderDash: [5, 5],
          borderWidth: 2,
          tension: 0
        },
        {
          label: "VWAP",
          data: schedules["VWAP"].inventory_remaining,
          borderColor: "#E040FB",
          borderDash: [3, 3],
          borderWidth: 2,
          tension: 0.2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#94A3B8" } } },
      scales: {
        x: { grid: { color: "#212B42" }, ticks: { color: "#94A3B8" } },
        y: { grid: { color: "#212B42" }, ticks: { color: "#94A3B8" } }
      }
    }
  });

  // 2. Trade Slices Bar Chart
  const ctxSlices = document.getElementById("chart-slices").getContext("2d");
  if (appState.charts.slices) appState.charts.slices.destroy();

  appState.charts.slices = new Chart(ctxSlices, {
    type: "bar",
    data: {
      labels: intervalLabels,
      datasets: [
        {
          label: "Almgren-Chriss",
          data: schedules["Almgren-Chriss (Optimal)"].trade_sizes,
          backgroundColor: "#00E5FF"
        },
        {
          label: "TWAP",
          data: schedules["TWAP"].trade_sizes,
          backgroundColor: "#FFB300"
        },
        {
          label: "VWAP",
          data: schedules["VWAP"].trade_sizes,
          backgroundColor: "#E040FB"
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#94A3B8" } } },
      scales: {
        x: { grid: { color: "#212B42" }, ticks: { color: "#94A3B8" } },
        y: { grid: { color: "#212B42" }, ticks: { color: "#94A3B8" } }
      }
    }
  });
}

// Render Schedule Table
function renderScheduleTable(acSched, order) {
  const tbody = document.getElementById("schedule-table-body");
  tbody.innerHTML = "";
  let cum = 0;

  for (let k = 0; k < acSched.trade_sizes.length; k++) {
    const size = acSched.trade_sizes[k];
    cum += size;
    const pct = (cum / order.total_quantity) * 100;
    const rem = acSched.inventory_remaining[k + 1];
    const rate = acSched.trading_rates[k];

    const row = document.createElement("tr");
    row.innerHTML = `
      <td>Interval ${k + 1}</td>
      <td>${(k * (order.time_horizon_min / order.num_intervals)).toFixed(0)}m - ${((k + 1) * (order.time_horizon_min / order.num_intervals)).toFixed(0)}m</td>
      <td style="color: #00E5FF; font-weight: 600;">${size.toLocaleString('en-US', {maximumFractionDigits: 0})}</td>
      <td>${rate.toLocaleString('en-US', {maximumFractionDigits: 0})} sh/m</td>
      <td>${cum.toLocaleString('en-US', {maximumFractionDigits: 0})}</td>
      <td>${pct.toFixed(1)}%</td>
      <td>${rem.toLocaleString('en-US', {maximumFractionDigits: 0})}</td>
    `;
    tbody.appendChild(row);
  }
}

// Render Monte Carlo Visuals & Table
function renderMonteCarloView(mcData) {
  // 1. Comparison Table
  const tbody = document.getElementById("mc-table-body");
  tbody.innerHTML = "";

  const summaries = mcData.strategy_summaries;
  for (const [name, s] of Object.entries(summaries)) {
    const row = document.createElement("tr");
    const badgeClass = name.includes("Almgren") ? "badge-ac" : name === "TWAP" ? "badge-twap" : name === "VWAP" ? "badge-vwap" : "badge-imm";
    row.innerHTML = `
      <td><span class="badge ${badgeClass}">${name}</span></td>
      <td>$${s.mean_shortfall_total.toLocaleString('en-US', {maximumFractionDigits: 2})}</td>
      <td style="font-weight: 700; color: #00E5FF;">${s.mean_shortfall_bps.toFixed(2)}</td>
      <td>${s.median_shortfall_bps.toFixed(2)}</td>
      <td>${s.std_shortfall_bps.toFixed(2)}</td>
      <td style="color: #FFB300;">${s.var_95_bps.toFixed(2)}</td>
      <td style="color: #FF5252;">${s.cvar_95_bps.toFixed(2)}</td>
      <td>${s.worst_case_bps.toFixed(2)}</td>
      <td>$${s.mean_execution_price.toFixed(2)}</td>
    `;
    tbody.appendChild(row);
  }

  // 2. Price Paths Chart
  const ctxPaths = document.getElementById("chart-mc-paths").getContext("2d");
  if (appState.charts.mcPaths) appState.charts.mcPaths.destroy();

  const timeLabels = mcData.time_steps_min.map(t => `${t.toFixed(0)}m`);
  const datasets = mcData.sample_price_paths.map((p, idx) => ({
    label: `Path ${idx+1}`,
    data: p,
    borderColor: "rgba(0, 229, 255, 0.15)",
    borderWidth: 1,
    pointRadius: 0,
    fill: false
  }));

  appState.charts.mcPaths = new Chart(ctxPaths, {
    type: "line",
    data: { labels: timeLabels, datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: "#212B42" }, ticks: { color: "#94A3B8" } },
        y: { grid: { color: "#212B42" }, ticks: { color: "#94A3B8" } }
      }
    }
  });

  // 3. Shortfall Histogram
  const ctxHist = document.getElementById("chart-mc-hist").getContext("2d");
  if (appState.charts.mcHist) appState.charts.mcHist.destroy();

  const acHist = summaries["Almgren-Chriss (Optimal)"].shortfall_histogram;
  const twapHist = summaries["TWAP"].shortfall_histogram;

  appState.charts.mcHist = new Chart(ctxHist, {
    type: "bar",
    data: {
      labels: acHist.bins.map(b => `${b} bps`),
      datasets: [
        {
          label: "Almgren-Chriss",
          data: acHist.counts,
          backgroundColor: "rgba(0, 229, 255, 0.7)"
        },
        {
          label: "TWAP",
          data: twapHist.counts,
          backgroundColor: "rgba(255, 179, 0, 0.6)"
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#94A3B8" } } },
      scales: {
        x: { grid: { color: "#212B42" }, ticks: { color: "#94A3B8" } },
        y: { grid: { color: "#212B42" }, ticks: { color: "#94A3B8" } }
      }
    }
  });
}

// Render Efficient Frontier Chart
function renderEfficientFrontierChart(efData) {
  const ctxEf = document.getElementById("chart-ef").getContext("2d");
  if (appState.charts.ef) appState.charts.ef.destroy();

  const frontierPoints = efData.frontier.map(p => ({
    x: p.expected_std_bps,
    y: p.expected_cost_bps
  }));

  const userPoint = [{
    x: efData.user_point.expected_std_bps,
    y: efData.user_point.expected_cost_bps
  }];

  const twapPoint = [{
    x: efData.benchmarks["TWAP"].expected_std_bps,
    y: efData.benchmarks["TWAP"].expected_cost_bps
  }];

  const vwapPoint = [{
    x: efData.benchmarks["VWAP"].expected_std_bps,
    y: efData.benchmarks["VWAP"].expected_cost_bps
  }];

  appState.charts.ef = new Chart(ctxEf, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Almgren-Chriss Frontier",
          data: frontierPoints,
          showLine: true,
          borderColor: "#00E5FF",
          borderWidth: 3,
          pointBackgroundColor: "#00E5FF",
          pointRadius: 4
        },
        {
          label: "Selected Policy (★)",
          data: userPoint,
          pointBackgroundColor: "#FFEB3B",
          pointRadius: 9,
          pointHoverRadius: 11
        },
        {
          label: "TWAP Benchmark",
          data: twapPoint,
          pointBackgroundColor: "#FFB300",
          pointRadius: 7
        },
        {
          label: "VWAP Benchmark",
          data: vwapPoint,
          pointBackgroundColor: "#E040FB",
          pointRadius: 7
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#94A3B8" } } },
      scales: {
        x: {
          title: { display: true, text: "Volatility Risk Std(Cost) [bps]", color: "#94A3B8" },
          grid: { color: "#212B42" },
          ticks: { color: "#94A3B8" }
        },
        y: {
          title: { display: true, text: "Expected Shortfall E(Cost) [bps]", color: "#94A3B8" },
          grid: { color: "#212B42" },
          ticks: { color: "#94A3B8" }
        }
      }
    }
  });
}

// Render Market Data Candlestick & Volume
function renderMarketDataCharts(data) {
  const ctxVol = document.getElementById("chart-mkt-volume").getContext("2d");
  if (appState.charts.mktVol) appState.charts.mktVol.destroy();

  const labels = data.u_shape_volume_profile.map((_, i) => `Bin ${i+1}`);
  const pctValues = data.u_shape_volume_profile.map(v => (v * 100).toFixed(2));

  appState.charts.mktVol = new Chart(ctxVol, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Intraday Volume Weight (%)",
        data: pctValues,
        backgroundColor: "rgba(171, 71, 188, 0.8)"
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#94A3B8" } } },
      scales: {
        x: { grid: { color: "#212B42" }, ticks: { color: "#94A3B8" } },
        y: { grid: { color: "#212B42" }, ticks: { color: "#94A3B8" } }
      }
    }
  });
}

// Run Dynamic Shock
async function runDynamicShock() {
  const { order, market_params } = getFormValues();
  const shockStep = parseInt(document.getElementById("shock-step").value);
  const volMult = parseFloat(document.getElementById("shock-vol").value);
  const liqMult = parseFloat(document.getElementById("shock-liq").value);

  try {
    const res = await fetch(`${API_BASE}/api/dynamic/shock`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        order,
        market_params,
        shock_step: shockStep,
        volatility_multiplier: volMult,
        liquidity_multiplier: liqMult
      })
    });
    const shockData = await res.json();

    document.getElementById("shock-verdict").innerText = shockData.urgency_shift;
    document.getElementById("shock-old-kappa").innerText = `${shockData.old_kappa.toFixed(4)} min⁻¹`;
    document.getElementById("shock-new-kappa").innerText = `${shockData.new_kappa.toFixed(4)} min⁻¹`;

    // Render comparison chart
    const ctx = document.getElementById("chart-shock").getContext("2d");
    if (appState.charts.shock) appState.charts.shock.destroy();

    const labels = shockData.time_steps_min.map(t => `${t.toFixed(0)}m`);

    appState.charts.shock = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Static Frozen Plan",
            data: shockData.original_inventory,
            borderColor: "#FF5252",
            borderDash: [5, 5],
            borderWidth: 2
          },
          {
            label: "Dynamic Adaptive Trajectory",
            data: shockData.dynamic_inventory,
            borderColor: "#00E676",
            borderWidth: 3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#94A3B8" } } },
        scales: {
          x: { grid: { color: "#212B42" }, ticks: { color: "#94A3B8" } },
          y: { grid: { color: "#212B42" }, ticks: { color: "#94A3B8" } }
        }
      }
    });
  } catch (err) {
    console.error("Dynamic shock error:", err);
  }
}

// Run Robustness Stress Test
async function runRobustnessTest() {
  const { order, market_params } = getFormValues();
  const btn = document.getElementById("btn-run-robustness");
  btn.innerText = "Running Stress-Test...";

  try {
    const res = await fetch(`${API_BASE}/api/evaluate/robustness`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ order_req: order, mkt_req: market_params })
    });
    const data = await res.json();

    document.getElementById("robustness-conclusion").innerText = data.conclusion;
    
    // Render volatility sensitivity chart
    const ctx = document.getElementById("chart-robustness-vol").getContext("2d");
    if (appState.charts.robVol) appState.charts.robVol.destroy();

    const mults = data.volatility_experiments.map(e => `${e.multiplier}x`);
    const acCosts = data.volatility_experiments.map(e => e.ac_cost_bps);
    const twapCosts = data.volatility_experiments.map(e => e.twap_cost_bps);
    const vwapCosts = data.volatility_experiments.map(e => e.vwap_cost_bps);

    appState.charts.robVol = new Chart(ctx, {
      type: "line",
      data: {
        labels: mults,
        datasets: [
          { label: "Almgren-Chriss", data: acCosts, borderColor: "#00E5FF", borderWidth: 3 },
          { label: "TWAP", data: twapCosts, borderColor: "#FFB300", borderDash: [4, 4] },
          { label: "VWAP", data: vwapCosts, borderColor: "#E040FB", borderDash: [2, 2] }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#94A3B8" } } },
        scales: {
          x: { title: { display: true, text: "True Volatility (Multiplier)", color: "#94A3B8" }, grid: { color: "#212B42" }, ticks: { color: "#94A3B8" } },
          y: { title: { display: true, text: "Shortfall (bps)", color: "#94A3B8" }, grid: { color: "#212B42" }, ticks: { color: "#94A3B8" } }
        }
      }
    });

    btn.innerText = "Re-Run Stress-Test";
  } catch (err) {
    console.error("Robustness error:", err);
    btn.innerText = "Run Stress-Test";
  }
}
