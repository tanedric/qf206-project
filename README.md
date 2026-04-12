# Quant Strategy Backtest Dashboard

An interactive local dashboard and backtesting engine for comparing portfolio allocation strategies on a filtered S&P 500 universe.

This project supports:

1. a one-shot script workflow through `Backtest.py`
2. a local web dashboard powered by FastAPI and a static frontend

## Quick Start

### 1. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 2. Create `.env`

Copy `.env.example` to `.env`.

Minimum required variable:

```env
STOCKNEWS_API_KEY=your_stocknews_api_key_here
```

Optional overrides:

```env
BEST_MODEL_PATH=model/best_model.pt
PARAM_DICT_PATH=data/param_dict.json
```

Notes:

- Only `STOCKNEWS_API_KEY` is required for the news sentiment pipeline.
- `BEST_MODEL_PATH` is optional. If omitted, the code uses the default: `model/best_model.pt`.
- `PARAM_DICT_PATH` is optional. If omitted, the code uses the default: `data/param_dict.json`.
- These model-path overrides are only useful if you moved the LSTM files somewhere else.
- If the API key is missing or no news is returned, sentiment views become zero, Black-Litterman falls back to its prior, and the regime defaults to `neutral`.

### 3. Run the dashboard

```powershell
python -m uvicorn dashboard_api:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

### 4. Or run the backtest script directly

Default run:

```powershell
python Backtest.py
```

Example with custom inputs:

```powershell
python Backtest.py --start 2018-01-01 --end 2024-12-31 --sector Financials --sentiment-window-days 20 --transaction-cost-bps 15 --enable-strategies MVO Black-Litterman
```

Currently supported dynamic inputs:

- `--start`
- `--end`
- `--sector`
- `--sentiment-window-days`
- `--transaction-cost-bps`
- `--enable-strategies`
- `--disable-strategies`

Still fixed in the current flow:

- benchmark = `^GSPC`
- rebalance frequency = monthly

### 5. Output files

Backtest outputs are written to `metrics/`.

Main examples:

- `metrics/equity_curves.csv`
- `metrics/performance_metrics.csv`
- `metrics/trading_diagnostics.csv`
- `metrics/prediction_diagnostics.csv`
- `metrics/rebalance_weights.csv`
- `metrics/fig1_equity_curves.png`
- `metrics/fig7_summary_metrics.png`
- `metrics/fig15_equity_curves_by_regime.png`

The dashboard reads from these generated outputs after each run.

---

## What This Project Does

The backtest compares:

- `MVO`
- `Black-Litterman`
- `Equal_Weight`
- `LSTM_MVO`
- `LSTM_BL`

Current default setup:

- Universe: S&P 500 sector / sub-industry filter
- Default filter: `Semiconductor`
- Benchmark: S&P 500 Index (`^GSPC`)
- Rebalancing: Monthly
- Sentiment window: Rolling 15-day news window
- Transaction costs: 10 bps by default

---

## Dashboard Features

The local web UI includes:

- strategy multi-select
- date-range selection
- searchable sector / sub-industry dropdown
- configurable sentiment window
- configurable transaction cost
- configurable initial capital
- focus strategy vs baseline comparison
- KPI cards
- interactive charts with hover tooltips
- drill-down tabs for:
  - Overview
  - Risk
  - Allocation
  - Diagnostics
  - Regime
  - Methodology
- export buttons for:
  - CSV
  - PNG
  - HTML report

---

## Runtime Model

The dashboard uses a long-running local backend process:

- `uvicorn` keeps the API and UI server alive
- when you click **Run Backtest**, the server launches `Backtest.py` as a subprocess
- results are written to `metrics/` and then loaded back into the dashboard

So this is different from running `Backtest.py` once in the terminal.

---

## Repository Layout

```text
.
|-- Backtest.py
|-- dashboard_api.py
|-- webui/
|-- scripts/
|   |-- data_pipeline.py
|   `-- sp500_wikipedia_universe.py
|-- model/
|   `-- best_model.pt
|-- data/
|   |-- param_dict.json
|   |-- news_cache.csv
|   `-- sentiment_cache.csv
`-- metrics/
```

---

## Strategy Notes

### Standard Black-Litterman

- prior = market-implied equilibrium returns
- views = sentiment-derived views
- confidence = omega derived from view uncertainty

### LSTM_BL

- prior = LSTM-predicted returns
- views = sentiment-derived views
- confidence = omega derived from view uncertainty

In the current results, the standard BL variant is the stronger and more robust benchmark.

---

## Recommended Workflow

For the cleanest local usage:

1. install dependencies
2. create `.env`
3. make sure `model/best_model.pt` exists
4. run the dashboard server
5. open the browser UI
6. tweak inputs and run comparisons interactively

---

## Tech Stack

- Python
- FastAPI
- Plotly.js
- PyPortfolioOpt
- PyTorch
- pandas / NumPy / matplotlib
- yfinance
- browser-native `fetch()` for frontend API calls

---

## Notes

- Sector filtering is based on the S&P 500 membership table built from the Wikipedia-derived CSV workflow.
- Benchmark-relative metrics are currently computed against `^GSPC`.
- Monthly rebalancing is the intended operating mode for the current strategy design.
- A video demo can be added later near the top of this README, just below the Quick Start section.
