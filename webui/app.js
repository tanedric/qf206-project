const state = {
  options: null,
  results: null,
  universeOptions: [],
  selectedStrategies: new Set(),
  rankingSort: {
    column: "Sharpe Ratio",
    direction: "desc",
  },
  loadingTimer: null,
  loadingStartedAt: null,
  loadingExpectedMs: null,
  loadingPhaseIndex: 0,
};

const loadingMessages = [
  "Fetching market data...",
  "Generating signals...",
  "Running portfolio optimisation...",
  "Calculating metrics...",
];

const regimeColors = {
  bad: "rgba(244, 114, 182, 0.20)",
  neutral: "rgba(253, 224, 71, 0.18)",
  good: "rgba(134, 239, 172, 0.18)",
};

document.addEventListener("DOMContentLoaded", async () => {
  setupTabs();
  setupComboBox();
  setupComparisonMode();
  setupRunButton();
  window.addEventListener("resize", resizeVisiblePlots);
  await loadOptions();
});

async function loadOptions() {
  const response = await fetch("/api/options");
  if (!response.ok) {
    throw new Error("Failed to load dashboard options.");
  }
  state.options = await response.json();
  buildStrategySelector();
  populateSelectDefaults();
  buildUniverseOptions();
  bindStaticSettings();
  bindExportLinks();
  await loadLatestResults();
}

async function loadLatestResults() {
  try {
    const response = await fetch("/api/latest");
    if (!response.ok) {
      renderEmptyDashboard();
      return;
    }
    state.results = await response.json();
    applyInputsToControls(state.results.meta.inputs);
    enableDownloads(state.results.meta.artifacts);
    renderDashboard();
  } catch (_error) {
    renderEmptyDashboard();
  }
}

function applyInputsToControls(inputs) {
  document.getElementById("start-date").value = inputs.start_date;
  document.getElementById("end-date").value = inputs.end_date;
  document.getElementById("sentiment-window").value = inputs.sentiment_window_days;
  document.getElementById("transaction-cost").value = inputs.transaction_cost_bps;
  document.getElementById("initial-capital").value = inputs.initial_capital;
  setComboSelection(inputs.sector_filter || "All S&P 500");
  state.selectedStrategies = new Set(inputs.strategies);
  document.querySelectorAll("#strategy-selector .chip").forEach((chip) => {
    const strategy = chip.dataset.strategy;
    const isSelected = state.selectedStrategies.has(strategy);
    chip.classList.toggle("selected", isSelected);
    chip.querySelector("input").checked = isSelected;
  });
  syncStrategySelects(inputs.focus_strategy, inputs.baseline_strategy);
  applyComparisonModeVisibility();
}

function buildStrategySelector() {
  const container = document.getElementById("strategy-selector");
  container.innerHTML = "";
  const defaults = state.options.defaults;
  state.selectedStrategies = new Set(state.options.strategies);
  state.options.strategies.forEach((strategy) => {
    const chip = document.createElement("label");
    chip.className = "chip selected";
    chip.dataset.strategy = strategy;
    chip.innerHTML = `
      <input type="checkbox" checked />
      <span class="chip-swatch" style="background:${state.options.strategyColors[strategy]};"></span>
      <span>${strategy}</span>
    `;
    chip.addEventListener("click", (event) => {
      event.preventDefault();
      const willSelect = !chip.classList.contains("selected");
      if (!willSelect && state.selectedStrategies.size === 1) {
        return;
      }
      chip.classList.toggle("selected", willSelect);
      chip.querySelector("input").checked = willSelect;
      if (willSelect) {
        state.selectedStrategies.add(strategy);
      } else {
        state.selectedStrategies.delete(strategy);
      }
      syncStrategySelects(defaults.focus_strategy, defaults.baseline_strategy);
    });
    container.appendChild(chip);
  });
}

function populateSelectDefaults() {
  document.getElementById("start-date").value = state.options.defaults.start_date;
  document.getElementById("end-date").value = state.options.defaults.end_date;
  document.getElementById("sentiment-window").value = state.options.defaults.sentiment_window_days;
  document.getElementById("transaction-cost").value = state.options.defaults.transaction_cost_bps;
  document.getElementById("initial-capital").value = state.options.defaults.initial_capital;
  syncStrategySelects(state.options.defaults.focus_strategy, state.options.defaults.baseline_strategy);
  document.getElementById("allocation-strategy").addEventListener("change", renderAllocationTab);
  applyComparisonModeVisibility();
}

function buildUniverseOptions() {
  const groups = state.options.universeFilters;
  state.universeOptions = [
    ...(groups.featured || []),
    groups.all_option,
    ...groups.sectors,
    ...groups.sub_industries,
  ];
  setComboSelection(state.options.defaults.sector_filter);
  renderComboOptions("");
}

function bindStaticSettings() {
  document.getElementById("meta-benchmark").textContent = state.options.staticSettings.benchmark;
  document.getElementById("meta-rebalance").textContent = state.options.staticSettings.rebalance_frequency;
  document.getElementById("meta-membership").textContent = state.options.staticSettings.membership_source;
}

function bindExportLinks() {
  ["download-csv", "download-png", "download-report"].forEach((id) => {
    document.getElementById(id).classList.add("disabled");
  });
}

function syncStrategySelects(defaultFocus, defaultBaseline) {
  const focusSelect = document.getElementById("focus-strategy");
  const baselineSelect = document.getElementById("baseline-strategy");
  const allocationSelect = document.getElementById("allocation-strategy");
  const selected = Array.from(state.selectedStrategies);

  [focusSelect, baselineSelect, allocationSelect].forEach((select) => {
    select.innerHTML = "";
    selected.forEach((strategy) => {
      const option = document.createElement("option");
      option.value = strategy;
      option.textContent = strategy;
      select.appendChild(option);
    });
  });

  const focusValue = selected.includes(focusSelect.dataset.value || defaultFocus)
    ? (focusSelect.dataset.value || defaultFocus)
    : (selected.includes(defaultFocus) ? defaultFocus : selected[0]);
  const baselineValue = selected.includes(baselineSelect.dataset.value || defaultBaseline)
    ? (baselineSelect.dataset.value || defaultBaseline)
    : (selected.includes(defaultBaseline) ? defaultBaseline : selected[0]);

  focusSelect.value = focusValue;
  baselineSelect.value = baselineValue;
  allocationSelect.value = focusValue;

  focusSelect.dataset.value = focusValue;
  baselineSelect.dataset.value = baselineValue;
  allocationSelect.dataset.value = focusValue;

  focusSelect.onchange = () => {
    focusSelect.dataset.value = focusSelect.value;
    if (state.results) {
      renderDashboard();
    }
  };
  baselineSelect.onchange = () => {
    baselineSelect.dataset.value = baselineSelect.value;
    if (state.results) {
      renderDashboard();
    }
  };
  allocationSelect.onchange = () => {
    allocationSelect.dataset.value = allocationSelect.value;
    renderAllocationTab();
  };
}

function setupComboBox() {
  const combo = document.getElementById("sector-combo");
  const input = document.getElementById("sector-search");
  const toggle = document.getElementById("sector-toggle");
  const menu = document.getElementById("sector-menu");

  toggle.addEventListener("click", () => {
    combo.classList.toggle("focused", true);
    menu.classList.toggle("hidden");
    renderComboOptions("");
  });

  input.addEventListener("focus", () => {
    combo.classList.add("focused");
    menu.classList.remove("hidden");
    renderComboOptions("");
    input.select();
  });

  input.addEventListener("input", () => {
    renderComboOptions(input.value);
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      const firstVisible = menu.querySelector(".combo-option");
      if (firstVisible) {
        event.preventDefault();
        chooseComboOption(firstVisible.dataset.value, firstVisible.textContent.trim());
      }
    }
    if (event.key === "Escape") {
      menu.classList.add("hidden");
      combo.classList.remove("focused");
    }
  });

  document.addEventListener("click", (event) => {
    if (!combo.contains(event.target)) {
      menu.classList.add("hidden");
      combo.classList.remove("focused");
    }
  });
}

function setupComparisonMode() {
  document.getElementById("comparison-mode").addEventListener("change", () => {
    applyComparisonModeVisibility();
    if (state.results) {
      renderDashboard();
    }
  });
}

function applyComparisonModeVisibility() {
  const mode = document.getElementById("comparison-mode").value;
  const wrapper = document.getElementById("focus-baseline-fields");
  const deltaCard = document.getElementById("kpi-delta-card");
  wrapper.classList.toggle("hidden-row", mode !== "focus-baseline");
  deltaCard.classList.toggle("hidden-card", mode !== "focus-baseline");
}

function renderComboOptions(filterText) {
  const menu = document.getElementById("sector-menu");
  const query = (filterText || "").trim().toLowerCase();
  menu.innerHTML = "";

  const grouped = state.universeOptions.reduce((acc, option) => {
    if (query && !option.label.toLowerCase().includes(query)) {
      return acc;
    }
    acc[option.group] = acc[option.group] || [];
    acc[option.group].push(option);
    return acc;
  }, {});

  Object.entries(grouped).forEach(([group, options]) => {
    const label = document.createElement("div");
    label.className = "combo-group-label";
    label.textContent = group;
    menu.appendChild(label);
    options.forEach((option) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "combo-option";
      button.dataset.value = option.value;
      button.textContent = option.label;
      button.addEventListener("click", () => chooseComboOption(option.value, option.label));
      menu.appendChild(button);
    });
  });

  if (!menu.children.length) {
    const empty = document.createElement("div");
    empty.className = "combo-group-label";
    empty.textContent = "No matching filters";
    menu.appendChild(empty);
  }
}

function setComboSelection(value) {
  const normalized = (value || "").trim().toLowerCase();
  const selected = state.universeOptions.find((option) => option.value === value)
    || state.universeOptions.find((option) => option.value.toLowerCase() === normalized)
    || state.universeOptions.find((option) => option.value.toLowerCase().includes(normalized))
    || state.universeOptions[0];
  document.getElementById("sector-search").value = selected.label;
  document.getElementById("sector-value").value = selected.value;
}

function chooseComboOption(value, label) {
  document.getElementById("sector-search").value = label;
  document.getElementById("sector-value").value = value;
  document.getElementById("sector-menu").classList.add("hidden");
  document.getElementById("sector-combo").classList.remove("focused");
}

function setupRunButton() {
  document.getElementById("run-backtest").addEventListener("click", runBacktest);
}

async function runBacktest() {
  const payload = {
    start_date: document.getElementById("start-date").value,
    end_date: document.getElementById("end-date").value,
    sector_filter: document.getElementById("sector-value").value || "All S&P 500",
    strategies: Array.from(state.selectedStrategies),
    sentiment_window_days: Number(document.getElementById("sentiment-window").value),
    transaction_cost_bps: Number(document.getElementById("transaction-cost").value),
    initial_capital: Number(document.getElementById("initial-capital").value),
    focus_strategy: document.getElementById("focus-strategy").value,
    baseline_strategy: document.getElementById("baseline-strategy").value,
  };

  showLoading(payload);
  try {
    const response = await fetch("/api/run-backtest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(formatApiError(data.detail || data));
    }
    state.results = data;
    enableDownloads(data.meta.artifacts);
    renderDashboard();
    window.setTimeout(scrollToResults, 1500);
  } catch (error) {
    window.alert(`Backtest failed.\n\n${error.message}`);
  } finally {
    hideLoading();
  }
}

function showLoading(payload) {
  const overlay = document.getElementById("loading-overlay");
  const message = document.getElementById("loading-message");
  const eta = document.getElementById("loading-eta");
  const elapsed = document.getElementById("loading-elapsed");
  const progressBar = document.getElementById("loading-progress-bar");
  state.loadingStartedAt = Date.now();
  state.loadingPhaseIndex = 0;
  state.loadingExpectedMs = estimateRuntimeMs(payload);
  message.textContent = loadingMessages[0];
  eta.textContent = `ETA: ${formatDuration(Math.round(state.loadingExpectedMs / 1000))}`;
  elapsed.textContent = "Elapsed: 0s";
  progressBar.style.width = "4%";
  overlay.classList.remove("hidden");
  state.loadingTimer = window.setInterval(() => {
    const elapsedMs = Date.now() - state.loadingStartedAt;
    const elapsedSec = Math.max(0, Math.round(elapsedMs / 1000));
    const expectedMs = state.loadingExpectedMs || 1;
    const progress = estimateProgress(elapsedMs, expectedMs);
    const remainingSec = Math.max(0, Math.round((expectedMs - elapsedMs) / 1000));
    elapsed.textContent = `Elapsed: ${formatDuration(elapsedSec)}`;
    eta.textContent = elapsedMs < expectedMs
      ? `ETA: ${formatDuration(remainingSec)}`
      : "ETA: wrapping up...";
    progressBar.style.width = `${Math.max(4, Math.min(progress * 100, 94))}%`;

    const nextPhase = Math.min(
      loadingMessages.length - 1,
      Math.floor((progress * loadingMessages.length))
    );
    if (nextPhase !== state.loadingPhaseIndex) {
      state.loadingPhaseIndex = nextPhase;
      message.textContent = loadingMessages[nextPhase];
    }
  }, 1000);
}

function hideLoading() {
  const overlay = document.getElementById("loading-overlay");
  const progressBar = document.getElementById("loading-progress-bar");
  progressBar.style.width = "100%";
  overlay.classList.add("hidden");
  if (state.loadingTimer) {
    window.clearInterval(state.loadingTimer);
    state.loadingTimer = null;
  }
  state.loadingStartedAt = null;
  state.loadingExpectedMs = null;
}

function enableDownloads(artifacts) {
  document.getElementById("download-csv").href = artifacts.csv;
  document.getElementById("download-png").href = artifacts.png;
  document.getElementById("download-report").href = artifacts.report;
  ["download-csv", "download-png", "download-report"].forEach((id) => {
    document.getElementById(id).classList.remove("disabled");
  });
}

function renderDashboard() {
  setTabsVisibility(true);
  showHeroEmptyState(false);
  renderKpis();
  renderHeroChart();
  renderOverviewTab();
  renderRiskTab();
  renderAllocationTab();
  renderDiagnosticsTab();
  renderRegimeTab();
  renderMethodologyTab();
  window.setTimeout(resizeVisiblePlots, 60);
}

function renderEmptyDashboard() {
  setTabsVisibility(false);
  showHeroEmptyState(true);
}

function setTabsVisibility(show) {
  const tabsShell = document.getElementById("tabs-shell");
  tabsShell.classList.toggle("hidden-tabs", !show);
}

function renderKpis() {
  const selected = getStrategiesInView();
  const metrics = getPerformanceMap();
  const comparisonMode = document.getElementById("comparison-mode").value;
  const focus = document.getElementById("focus-strategy").value;
  const baseline = document.getElementById("baseline-strategy").value;
  const summaryRows = state.results.summaryRanking.filter((row) => selected.includes(row.strategy));
  const best = summaryRows[0];
  const metricStrategy = comparisonMode === "all"
    ? (best ? best.strategy : focus)
    : focus;
  const focusMetrics = metrics.get(metricStrategy) || {};
  const baselineMetrics = metrics.get(baseline) || {};
  const terminalValue = getLatestEquityValue(metricStrategy);
  const returnDelta = toNumber(focusMetrics["Annual Return"]) - toNumber(baselineMetrics["Annual Return"]);
  const annualLabel = document.getElementById("kpi-annual-label");
  const focusName = document.getElementById("kpi-focus-name");
  const sharpeSub = document.getElementById("kpi-sharpe-sub");
  const drawdownSub = document.getElementById("kpi-drawdown-sub");
  const terminalSub = document.getElementById("kpi-terminal-sub");

  document.getElementById("kpi-best-strategy").textContent = best ? best.strategy : "-";
  document.getElementById("kpi-annual-return").textContent = formatPercent(focusMetrics["Annual Return"]);
  document.getElementById("kpi-sharpe").textContent = formatNumber(focusMetrics["Sharpe Ratio"]);
  document.getElementById("kpi-drawdown").textContent = formatPercent(focusMetrics["Max Drawdown"]);
  document.getElementById("kpi-terminal").textContent = formatCurrency(terminalValue);
  document.getElementById("kpi-delta").textContent = `${returnDelta >= 0 ? "+" : ""}${formatPercent(returnDelta)}`;
  annualLabel.textContent = comparisonMode === "all"
    ? "Best Strategy Annual Return"
    : "Focus Strategy Annual Return";
  focusName.textContent = metricStrategy;
  sharpeSub.textContent = `Risk-adjusted return for ${metricStrategy}`;
  drawdownSub.textContent = `Worst peak-to-trough loss for ${metricStrategy}`;
  terminalSub.textContent = `${metricStrategy} scaled by selected initial capital`;
  document.getElementById("kpi-baseline-name").textContent = `Annual return vs ${baseline}`;
}

function renderHeroChart() {
  const strategies = getStrategiesInView();
  const equity = state.results.equityCurves;
  const traces = strategies.map((strategy) => ({
    type: "scatter",
    mode: "lines",
    name: strategy,
    x: equity.map((row) => row.date),
    y: equity.map((row) => row[strategy]),
    line: { width: 3, color: state.results.meta.strategyColors[strategy] },
    hovertemplate: "%{x}<br>%{fullData.name}: %{y:$,.2f}<extra></extra>",
  }));

  Plotly.newPlot("hero-chart", traces, buildLayout({
    yaxisTitle: "Portfolio Value",
    showLegend: true,
  }), { responsive: true, displayModeBar: false });
}

function showHeroEmptyState(show) {
  const emptyState = document.getElementById("hero-empty-state");
  const heroPlot = document.getElementById("hero-chart");
  emptyState.classList.toggle("hidden-card", !show);
  heroPlot.classList.toggle("hidden-plot", show);
  if (show && heroPlot.data) {
    Plotly.purge(heroPlot);
  }
}

function renderOverviewTab() {
  const strategies = getStrategiesInView();
  const metrics = state.results.performanceMetrics.filter((row) => strategies.includes(row.strategy));
  const plotMetrics = ["Annual Return", "Sharpe Ratio", "Max Drawdown", "Calmar Ratio"];
  const traces = plotMetrics.map((metric) => ({
    type: "bar",
    name: metric,
    x: metrics.map((row) => row.strategy),
    y: metrics.map((row) => row[metric]),
    hovertemplate: `${metric}<br>%{x}: %{y:.4f}<extra></extra>`,
  }));

  Plotly.newPlot("overview-summary-chart", traces, buildLayout({
    barmode: "group",
    yaxisTitle: "Metric Value",
    showLegend: true,
  }), { responsive: true, displayModeBar: false });

  const rankingRows = state.results.summaryRanking.filter((row) => strategies.includes(row.strategy));
  renderRankingTable(rankingRows);
}

function renderRankingTable(rows) {
  const table = document.getElementById("ranking-table");
  const columns = ["strategy", "Annual Return", "Sharpe Ratio", "Max Drawdown", "Terminal Value"];
  const terminalByStrategy = new Map(getStrategiesInView().map((strategy) => [strategy, getLatestEquityValue(strategy)]));
  const sortableRows = rows.map((row) => ({
    ...row,
    "Terminal Value": terminalByStrategy.get(row.strategy),
  }));
  const sortedRows = sortRankingRows(sortableRows, state.rankingSort.column, state.rankingSort.direction);
  table.innerHTML = `
    <thead>
      <tr>${columns.map((col) => {
        const isActive = state.rankingSort.column === col;
        const arrow = !isActive ? "" : (state.rankingSort.direction === "asc" ? "▲" : "▼");
        return `<th class="sortable" data-column="${col}">${col}<span class="sort-indicator">${arrow}</span></th>`;
      }).join("")}</tr>
    </thead>
    <tbody>
      ${sortedRows.map((row) => `
        <tr>
          <td>${row.strategy}</td>
          <td>${formatPercent(row["Annual Return"])}</td>
          <td>${formatNumber(row["Sharpe Ratio"])}</td>
          <td>${formatPercent(row["Max Drawdown"])}</td>
          <td>${formatCurrency(row["Terminal Value"])}</td>
        </tr>
      `).join("")}
    </tbody>
  `;
  table.querySelectorAll("th.sortable").forEach((header) => {
    header.addEventListener("click", () => {
      const column = header.dataset.column;
      if (state.rankingSort.column === column) {
        state.rankingSort.direction = state.rankingSort.direction === "asc" ? "desc" : "asc";
      } else {
        state.rankingSort.column = column;
        state.rankingSort.direction = column === "strategy" ? "asc" : "desc";
      }
      renderRankingTable(rows);
    });
  });
}

function renderRiskTab() {
  const strategies = getStrategiesInView();
  const equity = state.results.equityCurves;
  const dates = equity.map((row) => row.date);
  const drawdownTraces = [];
  const volTraces = [];
  const sharpeTraces = [];
  const cvarTraces = [];

  strategies.forEach((strategy) => {
    const series = equity.map((row) => toNumber(row[strategy]));
    const returns = computeDailyReturns(series);
    drawdownTraces.push({
      type: "scatter",
      mode: "lines",
      name: strategy,
      x: dates,
      y: computeDrawdown(series).map((value) => value * 100),
      line: { color: state.results.meta.strategyColors[strategy], width: 2.2 },
    });
    volTraces.push({
      type: "scatter",
      mode: "lines",
      name: strategy,
      x: dates,
      y: rollingWindow(returns, 21, (window) => std(window) * Math.sqrt(252) * 100),
      line: { color: state.results.meta.strategyColors[strategy], width: 2.2 },
    });
    sharpeTraces.push({
      type: "scatter",
      mode: "lines",
      name: strategy,
      x: dates,
      y: rollingWindow(returns, 21, (window) => {
        const sigma = std(window);
        if (!Number.isFinite(sigma) || sigma === 0) return null;
        return ((mean(window) * 252) - 0.02) / (sigma * Math.sqrt(252));
      }),
      line: { color: state.results.meta.strategyColors[strategy], width: 2.2 },
    });
    cvarTraces.push({
      type: "scatter",
      mode: "lines",
      name: strategy,
      x: dates,
      y: rollingWindow(returns, 21, (window) => computeCvar(window, 0.05) * 100),
      line: { color: state.results.meta.strategyColors[strategy], width: 2.2 },
    });
  });

  Plotly.newPlot("risk-drawdown-chart", drawdownTraces, buildLayout({
    yaxisTitle: "Drawdown (%)",
    showLegend: true,
  }), { responsive: true, displayModeBar: false });
  Plotly.newPlot("risk-vol-chart", volTraces, buildLayout({
    yaxisTitle: "Volatility (%)",
    showLegend: true,
  }), { responsive: true, displayModeBar: false });
  Plotly.newPlot("risk-sharpe-chart", sharpeTraces, buildLayout({
    yaxisTitle: "Sharpe",
    showLegend: true,
    shapes: [zeroLine()],
  }), { responsive: true, displayModeBar: false });
  Plotly.newPlot("risk-cvar-chart", cvarTraces, buildLayout({
    yaxisTitle: "CVaR (%)",
    showLegend: true,
  }), { responsive: true, displayModeBar: false });
}

function renderAllocationTab() {
  if (!state.results) return;
  const compareImage = document.getElementById("allocation-compare-image");
  if (compareImage) {
    compareImage.src = `/metrics/fig6_weights.png?ts=${Date.now()}`;
  }
  const strategy = document.getElementById("allocation-strategy").value;
  const weights = state.results.rebalanceWeights.filter((row) => row.strategy === strategy);
  if (!weights.length) {
    Plotly.purge("allocation-top5-chart");
    Plotly.purge("allocation-heatmap-chart");
    Plotly.purge("allocation-turnover-chart");
    Plotly.purge("allocation-concentration-chart");
    return;
  }

  const grouped = groupWeightsByDate(weights);
  const topTickers = rankTickers(grouped, 5);
  const areaTraces = buildTop5AreaTraces(grouped, topTickers, strategy);
  Plotly.newPlot("allocation-top5-chart", areaTraces, buildLayout({
    yaxisTitle: "Weight",
    showLegend: true,
  }), { responsive: true, displayModeBar: false });

  const heatmap = buildWeightHeatmap(grouped, 10);
  Plotly.newPlot("allocation-heatmap-chart", [{
    type: "heatmap",
    x: heatmap.dates,
    y: heatmap.tickers,
    z: heatmap.matrix,
    colorscale: "Blues",
    hovertemplate: "%{x}<br>%{y}: %{z:.2%}<extra></extra>",
  }], buildLayout({
    yaxisTitle: "Ticker",
    showLegend: false,
  }), { responsive: true, displayModeBar: false });

  const turnover = state.results.turnoverByRebalance.filter((row) => row.strategy === strategy);
  Plotly.newPlot("allocation-turnover-chart", [
    {
      type: "scatter",
      mode: "lines",
      name: "Turnover",
      x: turnover.map((row) => row.date),
      y: turnover.map((row) => row.turnover),
      line: { color: state.results.meta.strategyColors[strategy], width: 2.4 },
      yaxis: "y",
    },
    {
      type: "scatter",
      mode: "lines",
      name: "Active Positions",
      x: turnover.map((row) => row.date),
      y: turnover.map((row) => row.active_positions),
      line: { color: "#0f2747", width: 1.8, dash: "dot" },
      yaxis: "y2",
    },
  ], buildLayout({
    yaxisTitle: "Turnover",
    showLegend: true,
    yaxis2: {
      title: "Active Positions",
      overlaying: "y",
      side: "right",
      gridcolor: "rgba(0,0,0,0)",
    },
  }), { responsive: true, displayModeBar: false });

  const concentration = state.results.concentrationDiagnostics.find((row) => row.strategy === strategy) || {};
  const concentrationMetrics = [
    "Peak Max Weight",
    "Average Max Weight",
    "Average Top-5 Weight Share",
    "Average Effective Bets",
  ];
  Plotly.newPlot("allocation-concentration-chart", [{
    type: "bar",
    x: concentrationMetrics,
    y: concentrationMetrics.map((metric) => concentration[metric]),
    marker: { color: state.results.meta.strategyColors[strategy] },
    hovertemplate: "%{x}: %{y:.4f}<extra></extra>",
  }], buildLayout({
    yaxisTitle: "Value",
    showLegend: false,
  }), { responsive: true, displayModeBar: false });
}

function renderDiagnosticsTab() {
  const strategies = getStrategiesInView();
  const predictionRows = state.results.predictionDiagnostics.filter((row) => strategies.includes(row.strategy) && row["Hit Rate"] !== null);
  const benchmarkRows = state.results.benchmarkDiagnostics.filter((row) => strategies.includes(row.strategy));

  Plotly.newPlot("diag-prediction-chart", [
    metricTrace(predictionRows, "Hit Rate"),
    metricTrace(predictionRows, "Pearson IC"),
    metricTrace(predictionRows, "Spearman Rank IC"),
    metricTrace(predictionRows, "RMSE"),
  ], buildLayout({
    barmode: "group",
    yaxisTitle: "Metric Value",
    showLegend: true,
  }), { responsive: true, displayModeBar: false });

  Plotly.newPlot("diag-benchmark-chart", [
    metricTrace(benchmarkRows, "Alpha"),
    metricTrace(benchmarkRows, "Beta"),
    metricTrace(benchmarkRows, "Information Ratio"),
    metricTrace(benchmarkRows, "Tracking Error"),
  ], buildLayout({
    barmode: "group",
    yaxisTitle: "Metric Value",
    showLegend: true,
  }), { responsive: true, displayModeBar: false });
}

function renderRegimeTab() {
  const strategies = getStrategiesInView();
  const traces = strategies.map((strategy) => ({
    type: "scatter",
    mode: "lines",
    name: strategy,
    x: state.results.equityCurves.map((row) => row.date),
    y: state.results.equityCurves.map((row) => row[strategy]),
    line: { width: 2.2, color: state.results.meta.strategyColors[strategy] },
  }));

  Plotly.newPlot("regime-overlay-chart", traces, buildLayout({
    yaxisTitle: "Portfolio Value",
    showLegend: true,
    shapes: buildRegimeShapes(),
  }), { responsive: true, displayModeBar: false });

  const counts = state.results.regimeByRebalance.reduce((acc, row) => {
    acc[row.regime] = (acc[row.regime] || 0) + 1;
    return acc;
  }, {});
  Plotly.newPlot("regime-counts-chart", [{
    type: "bar",
    x: Object.keys(counts),
    y: Object.values(counts),
    marker: { color: Object.keys(counts).map((key) => regimeColors[key] || "#cbd5e1") },
  }], buildLayout({
    yaxisTitle: "Count",
    showLegend: false,
  }), { responsive: true, displayModeBar: false });
}

function renderMethodologyTab() {
  const list = document.getElementById("methodology-config");
  const inputs = state.results.meta.inputs;
  list.innerHTML = "";
  const rows = [
    ["Strategies", inputs.strategies.join(", ")],
    ["Start Date", inputs.start_date],
    ["End Date", inputs.end_date],
    ["Universe Filter", inputs.sector_filter],
    ["Sentiment Window", `${inputs.sentiment_window_days} days`],
    ["Transaction Cost", `${inputs.transaction_cost_bps} bps`],
    ["Initial Capital", formatCurrency(inputs.initial_capital)],
    ["Benchmark", state.options.staticSettings.benchmark],
    ["Rebalance", state.options.staticSettings.rebalance_frequency],
  ];
  rows.forEach(([label, value]) => {
    list.insertAdjacentHTML("beforeend", `<dt>${label}</dt><dd>${value}</dd>`);
  });
}

function getStrategiesInView() {
  const comparisonMode = document.getElementById("comparison-mode").value;
  const focus = document.getElementById("focus-strategy").value;
  const baseline = document.getElementById("baseline-strategy").value;
  if (comparisonMode === "focus-baseline") {
    return Array.from(new Set([baseline, focus]));
  }
  return Array.from(state.selectedStrategies);
}

function getPerformanceMap() {
  return new Map(state.results.performanceMetrics.map((row) => [row.strategy, row]));
}

function getLatestEquityValue(strategy) {
  if (!state.results?.equityCurves?.length) return null;
  const lastRow = state.results.equityCurves[state.results.equityCurves.length - 1];
  return lastRow[strategy];
}

function metricTrace(rows, metric) {
  return {
    type: "bar",
    name: metric,
    x: rows.map((row) => row.strategy),
    y: rows.map((row) => row[metric]),
    hovertemplate: `${metric}<br>%{x}: %{y:.4f}<extra></extra>`,
  };
}

function sortRankingRows(rows, column, direction) {
  const factor = direction === "asc" ? 1 : -1;
  return [...rows].sort((left, right) => {
    const a = left[column];
    const b = right[column];
    if (column === "strategy") {
      return factor * String(a || "").localeCompare(String(b || ""));
    }
    const aNum = toNumber(a);
    const bNum = toNumber(b);
    if (aNum === null && bNum === null) return 0;
    if (aNum === null) return 1;
    if (bNum === null) return -1;
    return factor * (aNum - bNum);
  });
}

function buildLayout(overrides = {}) {
  return {
    margin: { l: 56, r: 24, t: 24, b: 46 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    font: { family: "Segoe UI, Inter, sans-serif", color: "#10233f" },
    xaxis: {
      gridcolor: "rgba(148, 163, 184, 0.16)",
      zeroline: false,
    },
    yaxis: {
      gridcolor: "rgba(148, 163, 184, 0.16)",
      zeroline: false,
    },
    legend: { orientation: "h", y: 1.1 },
    hovermode: "x unified",
    ...overrides,
  };
}

function zeroLine() {
  return {
    type: "line",
    xref: "paper",
    x0: 0,
    x1: 1,
    y0: 0,
    y1: 0,
    line: { color: "rgba(15,39,71,0.35)", width: 1, dash: "dot" },
  };
}

function buildRegimeShapes() {
  const shapes = [];
  const rows = state.results.regimeByRebalance;
  const finalDate = state.results.equityCurves[state.results.equityCurves.length - 1]?.date;
  for (let i = 0; i < rows.length; i += 1) {
    const current = rows[i];
    const next = rows[i + 1];
    shapes.push({
      type: "rect",
      xref: "x",
      yref: "paper",
      x0: current.date,
      x1: next ? next.date : finalDate,
      y0: 0,
      y1: 1,
      fillcolor: regimeColors[current.regime] || "rgba(203, 213, 225, 0.18)",
      line: { width: 0 },
      layer: "below",
    });
  }
  return shapes;
}

function groupWeightsByDate(rows) {
  const grouped = new Map();
  rows.forEach((row) => {
    if (!grouped.has(row.date)) grouped.set(row.date, {});
    grouped.get(row.date)[row.ticker] = row.weight;
  });
  return Array.from(grouped.entries())
    .sort((a, b) => new Date(a[0]) - new Date(b[0]))
    .map(([date, weights]) => ({ date, weights }));
}

function rankTickers(grouped, limit) {
  const totals = new Map();
  grouped.forEach(({ weights }) => {
    Object.entries(weights).forEach(([ticker, value]) => {
      totals.set(ticker, (totals.get(ticker) || 0) + value);
    });
  });
  return Array.from(totals.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([ticker]) => ticker);
}

function buildTop5AreaTraces(grouped, topTickers, strategy) {
  const dates = grouped.map((row) => row.date);
  const traces = topTickers.map((ticker) => ({
    type: "scatter",
    mode: "lines",
    name: ticker,
    x: dates,
    y: grouped.map((row) => row.weights[ticker] || 0),
    stackgroup: "one",
    hovertemplate: "%{x}<br>" + ticker + ": %{y:.2%}<extra></extra>",
  }));
  traces.push({
    type: "scatter",
    mode: "lines",
    name: "Other",
    x: dates,
    y: grouped.map((row) => 1 - topTickers.reduce((sum, ticker) => sum + (row.weights[ticker] || 0), 0)),
    stackgroup: "one",
    line: { color: "rgba(15,39,71,0.40)" },
    hovertemplate: "%{x}<br>Other: %{y:.2%}<extra></extra>",
  });
  return traces;
}

function buildWeightHeatmap(grouped, topCount) {
  const topTickers = rankTickers(grouped, topCount);
  const matrix = topTickers.map((ticker) => grouped.map((row) => row.weights[ticker] || 0));
  return {
    dates: grouped.map((row) => row.date),
    tickers: topTickers,
    matrix,
  };
}

function computeDailyReturns(series) {
  const returns = [null];
  for (let i = 1; i < series.length; i += 1) {
    const prev = series[i - 1];
    const current = series[i];
    if (!Number.isFinite(prev) || !Number.isFinite(current) || prev === 0) {
      returns.push(null);
    } else {
      returns.push((current / prev) - 1);
    }
  }
  return returns;
}

function computeDrawdown(series) {
  let peak = -Infinity;
  return series.map((value) => {
    if (!Number.isFinite(value)) return null;
    peak = Math.max(peak, value);
    return peak > 0 ? (value / peak) - 1 : null;
  });
}

function rollingWindow(series, window, computeFn) {
  return series.map((_, index) => {
    if (index < window - 1) return null;
    const slice = series.slice(index - window + 1, index + 1).filter((value) => value !== null && Number.isFinite(value));
    if (slice.length < window) return null;
    return computeFn(slice);
  });
}

function computeCvar(window, alpha) {
  if (!window.length) return null;
  const sorted = [...window].sort((a, b) => a - b);
  const count = Math.max(1, Math.ceil(alpha * sorted.length));
  return mean(sorted.slice(0, count));
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function std(values) {
  if (values.length < 2) return 0;
  const avg = mean(values);
  const variance = values.reduce((sum, value) => sum + ((value - avg) ** 2), 0) / values.length;
  return Math.sqrt(variance);
}

function formatPercent(value) {
  const numeric = toNumber(value);
  if (!Number.isFinite(numeric)) return "-";
  return `${(numeric * 100).toFixed(1)}%`;
}

function formatNumber(value) {
  const numeric = toNumber(value);
  return Number.isFinite(numeric) ? numeric.toFixed(2) : "-";
}

function formatCurrency(value) {
  const numeric = toNumber(value);
  if (!Number.isFinite(numeric)) return "-";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(numeric);
}

function toNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function formatApiError(detail) {
  if (typeof detail === "string") return detail;
  if (!detail) return "Unknown error";
  if (detail.message) {
    return `${detail.message}\n\n${detail.stderr || detail.stdout || ""}`.trim();
  }
  return JSON.stringify(detail, null, 2);
}

function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((btn) => btn.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
      button.classList.add("active");
      const panel = document.getElementById(`tab-${button.dataset.tab}`);
      panel.classList.add("active");
      window.setTimeout(() => resizePlotsWithin(panel), 40);
    });
  });
}

function scrollToResults() {
  const target = document.querySelector(".top-bar");
  if (target) {
    target.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function estimateRuntimeMs(payload) {
  const previousRunMs = state.results?.meta?.lastRuntimeSeconds
    ? state.results.meta.lastRuntimeSeconds * 1000
    : null;
  if (previousRunMs) {
    return Math.max(previousRunMs, 30000);
  }
  const months = estimateMonthSpan(payload.start_date, payload.end_date);
  const strategyCount = Math.max(1, payload.strategies.length);
  const sentimentFactor = Math.max(0, Number(payload.sentiment_window_days || 15) - 15) * 1200;
  const heuristicMs = 60000 + (months * 2500) + (strategyCount * 35000) + sentimentFactor;
  return Math.max(heuristicMs, 45000);
}

function estimateMonthSpan(startDate, endDate) {
  const start = new Date(startDate);
  const end = new Date(endDate);
  if (!Number.isFinite(start.getTime()) || !Number.isFinite(end.getTime()) || end < start) {
    return 12;
  }
  return Math.max(
    1,
    ((end.getFullYear() - start.getFullYear()) * 12) + (end.getMonth() - start.getMonth()) + 1
  );
}

function estimateProgress(elapsedMs, expectedMs) {
  const raw = elapsedMs / Math.max(expectedMs, 1);
  if (raw <= 1) {
    return 0.1 + (raw * 0.78);
  }
  return 0.88 + (1 - Math.exp(-(raw - 1) * 1.4)) * 0.06;
}

function formatDuration(totalSeconds) {
  const safe = Math.max(0, Math.round(totalSeconds));
  const minutes = Math.floor(safe / 60);
  const seconds = safe % 60;
  if (minutes <= 0) {
    return `${seconds}s`;
  }
  return `${minutes}m ${seconds}s`;
}

function resizeVisiblePlots() {
  document.querySelectorAll(".tab-panel.active .plot").forEach((element) => {
    resizePlotElement(element);
  });
}

function resizePlotsWithin(container) {
  container.querySelectorAll(".plot").forEach((element) => {
    resizePlotElement(element);
  });
}

function resizePlotElement(element) {
  if (element && element.data) {
    Plotly.Plots.resize(element);
  }
}
