# Quant Strategy Backtest Dashboard

An interactive local dashboard and backtesting engine for comparing portfolio allocation strategies on an S&P 500 sector universe, with a focus on:

- classical **Mean-Variance Optimisation (MVO)**
- **Black-Litterman (BL)** with sentiment-based views
- **LSTM-enhanced** strategy variants
- rich diagnostics for performance, risk, allocation behaviour, and regime analysis

The project now supports both:

1. a **one-shot script workflow** via `Backtest.py`
2. a **local web dashboard** powered by FastAPI + a static frontend

---

## What This Project Does

The backtest compares the following strategies:

- `MVO`
- `Black-Litterman`
- `Equal_Weight`
- `LSTM_MVO`
- `LSTM_BL`

The current setup uses:

- **Universe**: S&P 500 sector/sub-industry filter
- **Default filter**: `Semiconductor`
- **Benchmark**: S&P 500 Index (`^GSPC`)
- **Rebalancing**: Monthly
- **Sentiment window**: Rolling 15-day news window
- **Transaction costs**: 10 bps by default

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

## Repository Layout

```text
.
├── Backtest.py              # Main backtest engine
├── dashboard_api.py         # Local FastAPI server + UI API
├── webui/                   # Static frontend files
├── scripts/
│   ├── data_pipeline.py
│   └── sp500_wikipedia_universe.py
├── model/
│   └── best_model.pt        # LSTM checkpoint
├── data/
│   ├── param_dict.json
│   ├── news_cache.csv
│   └── sentiment_cache.csv
└── metrics/                 # Generated charts and CSV outputs
```

---

## Setup

### 1. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 2. Configure environment variables

Create a local `.env` file from `.env.example`.

Example:

```env
STOCKNEWS_API_KEY=your_stocknews_api_key_here
BEST_MODEL_PATH=model/best_model.pt
PARAM_DICT_PATH=data/param_dict.json
```

Notes:

- `STOCKNEWS_API_KEY` is needed for the sentiment/news pipeline.
- `BEST_MODEL_PATH` and `PARAM_DICT_PATH` are optional overrides.
- If omitted, the code falls back to the repo defaults.

---

## Running the Dashboard

Start the local server:

```powershell
python -m uvicorn dashboard_api:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

### Runtime model

The dashboard uses a **long-running local backend process**:

- `uvicorn` keeps the API + UI server alive
- when you click **Run Backtest**, the server launches `Backtest.py` as a subprocess
- results are written to `metrics/` and then loaded back into the dashboard

So yes, this is different from running `Backtest.py` once in the terminal.

---

## Running the Backtest Script Directly

You can still run the script without the web UI:

```powershell
python Backtest.py
```

You can also override inputs directly:

```powershell
python Backtest.py --start 2018-01-01 --end 2024-12-31 --sector Financials --sentiment-window-days 20 --transaction-cost-bps 15 --enable-strategies MVO Black-Litterman
```

Current supported dynamic inputs in `Backtest.py`:

- `--start`
- `--end`
- `--sector`
- `--sentiment-window-days`
- `--transaction-cost-bps`
- `--enable-strategies`
- `--disable-strategies`

Still fixed in the current dashboard/backend flow:

- benchmark = `^GSPC`
- rebalance frequency = monthly

---

## Output Files

Backtest outputs are written to `metrics/`.

Key examples:

- `metrics/equity_curves.csv`
- `metrics/performance_metrics.csv`
- `metrics/trading_diagnostics.csv`
- `metrics/prediction_diagnostics.csv`
- `metrics/rebalance_weights.csv`
- `metrics/fig1_equity_curves.png`
- `metrics/fig7_summary_metrics.png`
- `metrics/fig15_equity_curves_by_regime.png`

---

## Strategy Notes

### Standard Black-Litterman

- prior = **market-implied equilibrium returns**
- views = **sentiment-derived views**
- confidence = **omega derived from view uncertainty**

### LSTM_BL

- prior = **LSTM-predicted returns**
- views = **sentiment-derived views**
- confidence = **omega derived from view uncertainty**

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

---

## Notes

- Sector filtering is based on the S&P 500 membership table built from the Wikipedia-derived CSV workflow.
- Benchmark-relative metrics are currently computed against `^GSPC`.
- Monthly rebalancing is the intended operating mode for the current strategy design.

