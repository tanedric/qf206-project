"""Local API + static dashboard wrapper for the backtest UI.

CHANGED: This file adds a lightweight FastAPI server around Backtest.py so the
existing batch backtest can be triggered from a modern local web UI without
rewriting the portfolio logic itself.
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "webui"
METRICS_DIR = ROOT / "metrics"
SCRIPTS_DIR = ROOT / "scripts"
BACKTEST_FILE = ROOT / "Backtest.py"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from sp500_wikipedia_universe import ensure_membership_csv  # type: ignore[import]


DEFAULTS = {
    "start_date": "2015-01-01",
    "end_date": "2026-01-01",
    "sector_filter": "Semiconductor",
    "sentiment_window_days": 15,
    "transaction_cost_bps": 10,
    "benchmark": "^GSPC",
    "rebalance_frequency": "Monthly",
    "initial_capital": 10000,
    "focus_strategy": "Black-Litterman",
    "baseline_strategy": "MVO",
}

STRATEGIES = ["MVO", "Black-Litterman", "Equal_Weight", "LSTM_MVO", "LSTM_BL"]
STRATEGY_COLORS = {
    "MVO": "#2196F3",
    "Black-Litterman": "#FF5722",
    "Equal_Weight": "#4CAF50",
    "LSTM_MVO": "#9C27B0",
    "LSTM_BL": "#FF9800",
}
RUN_LOCK = threading.Lock()
LAST_RUN_CONTEXT: dict[str, Any] = {
    "inputs": DEFAULTS.copy(),
    "generated": False,
    "last_runtime_seconds": None,
}


def _estimate_runtime_seconds(inputs: dict[str, Any]) -> float:
    last_runtime = LAST_RUN_CONTEXT.get("last_runtime_seconds")
    if isinstance(last_runtime, (int, float)) and last_runtime > 0:
        return float(last_runtime)

    start = pd.to_datetime(inputs.get("start_date"), errors="coerce")
    end = pd.to_datetime(inputs.get("end_date"), errors="coerce")
    if pd.isna(start) or pd.isna(end) or end < start:
        months = 12
    else:
        months = max(1, ((end.year - start.year) * 12) + (end.month - start.month) + 1)
    strategies = inputs.get("strategies") or []
    strategy_count = max(1, len(strategies))
    sentiment_days = max(int(inputs.get("sentiment_window_days", 15)), 1)
    # CHANGED: keep ETA estimates simple and backend-local; this is for logging
    # visibility only and does not affect the backtest pipeline itself.
    estimate = 60 + (months * 2.5) + (strategy_count * 35) + max(0, sentiment_days - 15) * 1.2
    return max(estimate, 45.0)


class BacktestRunRequest(BaseModel):
    start_date: str = Field(default=DEFAULTS["start_date"])
    end_date: str = Field(default=DEFAULTS["end_date"])
    sector_filter: str = Field(default=DEFAULTS["sector_filter"])
    strategies: list[str] = Field(default_factory=lambda: STRATEGIES.copy())
    sentiment_window_days: int = Field(default=DEFAULTS["sentiment_window_days"], ge=1)
    transaction_cost_bps: float = Field(default=DEFAULTS["transaction_cost_bps"], ge=0)
    initial_capital: float = Field(default=DEFAULTS["initial_capital"], gt=0)
    focus_strategy: str = Field(default=DEFAULTS["focus_strategy"])
    baseline_strategy: str = Field(default=DEFAULTS["baseline_strategy"])

    @field_validator("strategies")
    @classmethod
    def validate_strategies(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("At least one strategy must be selected.")
        unknown = sorted(set(value) - set(STRATEGIES))
        if unknown:
            raise ValueError(f"Unknown strategies: {unknown}")
        return value

    @field_validator("focus_strategy")
    @classmethod
    def validate_focus(cls, value: str) -> str:
        if value not in STRATEGIES:
            raise ValueError(f"Unknown focus strategy: {value}")
        return value

    @field_validator("baseline_strategy")
    @classmethod
    def validate_baseline(cls, value: str) -> str:
        if value not in STRATEGIES:
            raise ValueError(f"Unknown baseline strategy: {value}")
        return value


app = FastAPI(title="Quant Backtest Dashboard", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

METRICS_DIR.mkdir(exist_ok=True)
WEB_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
app.mount("/metrics", StaticFiles(directory=METRICS_DIR), name="metrics")


def _load_membership_options() -> dict[str, Any]:
    membership = ensure_membership_csv()

    def _clean_unique(column: str) -> list[str]:
        if column not in membership.columns:
            return []
        cleaned = (
            membership[column]
            .dropna()
            .astype(str)
            .str.strip()
        )
        cleaned = cleaned[cleaned.ne("")]
        return sorted(cleaned.unique().tolist())

    sectors = _clean_unique("gics_sector")
    sub_industries = _clean_unique("gics_sub_industry")
    # CHANGED: surface the app's default Semiconductor filter explicitly so the
    # UI can default to it even though the raw CSV stores pluralized variants.
    featured_filters = [
        {"label": "Semiconductor", "value": "Semiconductor", "group": "Recommended"},
    ]
    return {
        "featured": featured_filters,
        "all_option": {"label": "All S&P 500", "value": "All S&P 500", "group": "General"},
        "sectors": [{"label": value, "value": value, "group": "Sector"} for value in sectors],
        "sub_industries": [
            {"label": value, "value": value, "group": "Sub-Industry"}
            for value in sub_industries
        ],
    }


def _read_csv(path: Path, *, parse_dates: bool = False) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if parse_dates:
        return pd.read_csv(path, parse_dates=[0], date_format="%Y-%m-%d")
    return pd.read_csv(path)


def _frame_records(df: pd.DataFrame, *, index_name: str | None = None) -> list[dict[str, Any]]:
    if df.empty:
        return []
    payload = df.copy()
    if index_name is not None:
        payload = payload.reset_index().rename(columns={payload.index.name or "index": index_name})
    for col in payload.columns:
        if pd.api.types.is_datetime64_any_dtype(payload[col]):
            payload[col] = payload[col].dt.strftime("%Y-%m-%d")
    payload = payload.where(pd.notnull(payload), None)
    return payload.to_dict(orient="records")


def _scale_currency_frames(initial_capital: float, equity: pd.DataFrame, gross_equity: pd.DataFrame,
                           turnover_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scale = initial_capital / 100.0
    equity_scaled = equity.copy()
    gross_scaled = gross_equity.copy()
    turnover_scaled = turnover_df.copy()
    if not equity_scaled.empty:
        equity_scaled.loc[:, :] = equity_scaled.loc[:, :] * scale
    if not gross_scaled.empty:
        gross_scaled.loc[:, :] = gross_scaled.loc[:, :] * scale
    if not turnover_scaled.empty:
        for col in ("transaction_cost_paid", "portfolio_value_before_cost"):
            if col in turnover_scaled.columns:
                turnover_scaled[col] = turnover_scaled[col] * scale
    return equity_scaled, gross_scaled, turnover_scaled


def _build_summary_rows(performance_metrics: pd.DataFrame) -> list[dict[str, Any]]:
    if performance_metrics.empty:
        return []
    ranking = performance_metrics.copy()
    if "Sharpe Ratio" in ranking.columns:
        ranking = ranking.sort_values("Sharpe Ratio", ascending=False)
    ranking = ranking.reset_index().rename(columns={ranking.index.name or "index": "strategy"})
    ranking = ranking.where(pd.notnull(ranking), None)
    return ranking.to_dict(orient="records")


def _load_results_bundle(inputs: dict[str, Any]) -> dict[str, Any]:
    performance_metrics = _read_csv(METRICS_DIR / "performance_metrics.csv", parse_dates=False)
    trading_diagnostics = _read_csv(METRICS_DIR / "trading_diagnostics.csv", parse_dates=False)
    concentration_diagnostics = _read_csv(METRICS_DIR / "concentration_diagnostics.csv", parse_dates=False)
    benchmark_diagnostics = _read_csv(METRICS_DIR / "benchmark_diagnostics.csv", parse_dates=False)
    prediction_diagnostics = _read_csv(METRICS_DIR / "prediction_diagnostics.csv", parse_dates=False)
    equity_curves = _read_csv(METRICS_DIR / "equity_curves.csv", parse_dates=True)
    gross_equity_curves = _read_csv(METRICS_DIR / "equity_curves_gross.csv", parse_dates=True)
    turnover_by_rebalance = _read_csv(METRICS_DIR / "turnover_costs_by_rebalance.csv", parse_dates=True)
    rebalance_weights = _read_csv(METRICS_DIR / "rebalance_weights.csv", parse_dates=True)
    prediction_by_rebalance = _read_csv(METRICS_DIR / "prediction_diagnostics_by_rebalance.csv", parse_dates=True)
    regime_by_rebalance = _read_csv(METRICS_DIR / "regime_by_rebalance.csv", parse_dates=True)
    benchmark_equity = _read_csv(METRICS_DIR / "benchmark_equity_curve.csv", parse_dates=True)

    if not performance_metrics.empty:
        performance_metrics = performance_metrics.rename(columns={performance_metrics.columns[0]: "strategy"}).set_index("strategy")
    if not trading_diagnostics.empty:
        trading_diagnostics = trading_diagnostics.rename(columns={trading_diagnostics.columns[0]: "strategy"}).set_index("strategy")
    if not concentration_diagnostics.empty:
        concentration_diagnostics = concentration_diagnostics.rename(columns={concentration_diagnostics.columns[0]: "strategy"}).set_index("strategy")
    if not benchmark_diagnostics.empty:
        benchmark_diagnostics = benchmark_diagnostics.rename(columns={benchmark_diagnostics.columns[0]: "strategy"}).set_index("strategy")
    if not prediction_diagnostics.empty:
        prediction_diagnostics = prediction_diagnostics.rename(columns={prediction_diagnostics.columns[0]: "strategy"}).set_index("strategy")

    if not equity_curves.empty:
        equity_curves = equity_curves.rename(columns={equity_curves.columns[0]: "date"})
        equity_curves["date"] = pd.to_datetime(equity_curves["date"])
        equity_curves = equity_curves.set_index("date")
    if not gross_equity_curves.empty:
        gross_equity_curves = gross_equity_curves.rename(columns={gross_equity_curves.columns[0]: "date"})
        gross_equity_curves["date"] = pd.to_datetime(gross_equity_curves["date"])
        gross_equity_curves = gross_equity_curves.set_index("date")

    if not turnover_by_rebalance.empty:
        turnover_by_rebalance = turnover_by_rebalance.rename(columns={turnover_by_rebalance.columns[0]: "date"})
        turnover_by_rebalance["date"] = pd.to_datetime(turnover_by_rebalance["date"])

    if not rebalance_weights.empty:
        rebalance_weights["date"] = pd.to_datetime(rebalance_weights["date"])

    if not prediction_by_rebalance.empty:
        prediction_by_rebalance["date"] = pd.to_datetime(prediction_by_rebalance["date"])

    if not regime_by_rebalance.empty:
        regime_by_rebalance["date"] = pd.to_datetime(regime_by_rebalance["date"])

    if not benchmark_equity.empty:
        benchmark_equity = benchmark_equity.rename(columns={benchmark_equity.columns[0]: "date"})
        benchmark_equity["date"] = pd.to_datetime(benchmark_equity["date"])

    equity_scaled, gross_scaled, turnover_scaled = _scale_currency_frames(
        float(inputs["initial_capital"]),
        equity_curves,
        gross_equity_curves,
        turnover_by_rebalance,
    )

    benchmark_equity_records: list[dict[str, Any]] = []
    if not benchmark_equity.empty and "benchmark_equity" in benchmark_equity.columns:
        benchmark_copy = benchmark_equity.copy()
        benchmark_copy["benchmark_equity"] = benchmark_copy["benchmark_equity"] * (float(inputs["initial_capital"]) / 100.0)
        benchmark_equity_records = _frame_records(benchmark_copy)

    return {
        "meta": {
            "inputs": inputs,
            "strategyColors": STRATEGY_COLORS,
            "generated": True,
            "lastRuntimeSeconds": LAST_RUN_CONTEXT.get("last_runtime_seconds"),
            "artifacts": {
                "csv": "/api/download/csv",
                "png": "/api/download/png",
                "report": "/api/download/report",
            },
        },
        "performanceMetrics": _frame_records(performance_metrics, index_name="strategy"),
        "summaryRanking": _build_summary_rows(performance_metrics),
        "tradingDiagnostics": _frame_records(trading_diagnostics, index_name="strategy"),
        "concentrationDiagnostics": _frame_records(concentration_diagnostics, index_name="strategy"),
        "benchmarkDiagnostics": _frame_records(benchmark_diagnostics, index_name="strategy"),
        "predictionDiagnostics": _frame_records(prediction_diagnostics, index_name="strategy"),
        # CHANGED: preserve the time index as an explicit `date` field so the
        # frontend can render time-series charts for hero/risk/regime views.
        "equityCurves": _frame_records(equity_scaled, index_name="date"),
        "grossEquityCurves": _frame_records(gross_scaled, index_name="date"),
        "benchmarkEquity": benchmark_equity_records,
        "turnoverByRebalance": _frame_records(turnover_scaled),
        "rebalanceWeights": _frame_records(rebalance_weights),
        "predictionByRebalance": _frame_records(prediction_by_rebalance),
        "regimeByRebalance": _frame_records(regime_by_rebalance),
    }


def _run_backtest_subprocess(request: BacktestRunRequest) -> tuple[int, str, str]:
    sector_value = request.sector_filter.strip() if request.sector_filter else "All S&P 500"
    cmd = [
        sys.executable,
        "-u",
        str(BACKTEST_FILE),
        "--start", request.start_date,
        "--end", request.end_date,
        "--sector", sector_value,
        "--sentiment-window-days", str(request.sentiment_window_days),
        "--transaction-cost-bps", str(request.transaction_cost_bps),
        "--enable-strategies",
        *request.strategies,
    ]
    # CHANGED: force UTF-8 for the subprocess so Windows locale encodings do
    # not crash on Backtest.py's Unicode logging output.
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=1,
    )
    combined_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        combined_lines.append(line)
        print(f"[Backtest.py] {line.rstrip()}", flush=True)
    process.stdout.close()
    return_code = process.wait()
    return return_code, "".join(combined_lines), ""


def _zip_files(pattern: str) -> StreamingResponse:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(METRICS_DIR.glob(pattern)):
            zf.write(path, arcname=path.name)
    buffer.seek(0)
    filename = "metrics_csv.zip" if pattern == "*.csv" else "metrics_png.zip"
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _inline_image(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_report_html() -> str:
    performance = _read_csv(METRICS_DIR / "performance_metrics.csv")
    table_html = performance.to_html(index=False, classes="metrics-table") if not performance.empty else "<p>No metrics generated yet.</p>"
    images = [
        ("Equity Curves", METRICS_DIR / "fig1_equity_curves.png"),
        ("Drawdown", METRICS_DIR / "fig2_drawdown.png"),
        ("Summary Metrics", METRICS_DIR / "fig7_summary_metrics.png"),
        ("Regime Overlay", METRICS_DIR / "fig15_equity_curves_by_regime.png"),
    ]
    image_blocks = []
    for title, path in images:
        src = _inline_image(path)
        if src:
            image_blocks.append(f"<section><h2>{title}</h2><img src=\"{src}\" alt=\"{title}\" /></section>")
    context = LAST_RUN_CONTEXT.get("inputs", DEFAULTS)
    settings_html = "".join(
        f"<li><strong>{key.replace('_', ' ').title()}:</strong> {value}</li>"
        for key, value in context.items()
    )
    return f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Backtest Report</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 32px; color: #10233f; }}
          h1 {{ margin-bottom: 8px; }}
          .grid {{ display: grid; grid-template-columns: 320px 1fr; gap: 24px; }}
          .card {{ border: 1px solid #d6e4f0; border-radius: 14px; padding: 18px; background: #f8fbff; }}
          .metrics-table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
          .metrics-table th, .metrics-table td {{ border: 1px solid #d6e4f0; padding: 8px 10px; text-align: right; }}
          .metrics-table th:first-child, .metrics-table td:first-child {{ text-align: left; }}
          img {{ width: 100%; border: 1px solid #d6e4f0; border-radius: 14px; margin-top: 8px; }}
          ul {{ margin: 0; padding-left: 18px; }}
          section {{ margin-top: 28px; }}
        </style>
      </head>
      <body>
        <h1>Backtest Report</h1>
        <p>Generated from the interactive local dashboard.</p>
        <div class="grid">
          <div class="card">
            <h2>Run Settings</h2>
            <ul>{settings_html}</ul>
          </div>
          <div class="card">
            <h2>Performance Metrics</h2>
            {table_html}
          </div>
        </div>
        {''.join(image_blocks)}
      </body>
    </html>
    """


@app.get("/", response_class=FileResponse)
def serve_index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/options")
def get_options() -> dict[str, Any]:
    return {
        "strategies": STRATEGIES,
        "defaults": DEFAULTS,
        "strategyColors": STRATEGY_COLORS,
        "universeFilters": _load_membership_options(),
        "staticSettings": {
            "benchmark": DEFAULTS["benchmark"],
            "rebalance_frequency": DEFAULTS["rebalance_frequency"],
            "membership_source": "Wikipedia-derived membership CSV",
        },
    }


@app.post("/api/run-backtest")
def run_backtest(request: BacktestRunRequest) -> dict[str, Any]:
    if request.focus_strategy not in request.strategies:
        raise HTTPException(status_code=422, detail="Focus strategy must be one of the selected strategies.")
    if request.baseline_strategy not in request.strategies:
        raise HTTPException(status_code=422, detail="Baseline strategy must be one of the selected strategies.")

    if not RUN_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A backtest is already running. Please wait for it to finish.")

    try:
        started_at = time.perf_counter()
        request_payload = request.model_dump()
        eta_seconds = _estimate_runtime_seconds(request_payload)
        print("[Dashboard API] Received backtest request:", flush=True)
        print(json.dumps(request_payload, indent=2), flush=True)
        print(
            f"[Dashboard API] Starting Backtest.py with estimated runtime ~{eta_seconds:.0f}s.",
            flush=True,
        )
        return_code, stdout, stderr = _run_backtest_subprocess(request)
        if return_code != 0:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Backtest failed.",
                    "stdout": stdout[-4000:],
                    "stderr": stderr[-4000:],
                },
            )
        LAST_RUN_CONTEXT["inputs"] = request_payload
        LAST_RUN_CONTEXT["generated"] = True
        LAST_RUN_CONTEXT["last_runtime_seconds"] = time.perf_counter() - started_at
        print(
            f"[Dashboard API] Backtest completed in {LAST_RUN_CONTEXT['last_runtime_seconds']:.1f}s.",
            flush=True,
        )
        payload = _load_results_bundle(request.model_dump())
        payload["meta"]["stdoutTail"] = stdout[-4000:]
        return payload
    finally:
        RUN_LOCK.release()


@app.get("/api/latest")
def latest_results() -> dict[str, Any]:
    if not (METRICS_DIR / "performance_metrics.csv").exists():
        raise HTTPException(status_code=404, detail="No prior backtest results found.")
    payload = _load_results_bundle(LAST_RUN_CONTEXT.get("inputs", DEFAULTS.copy()))
    return payload


@app.get("/api/download/csv")
def download_csv() -> StreamingResponse:
    return _zip_files("*.csv")


@app.get("/api/download/png")
def download_png() -> StreamingResponse:
    return _zip_files("*.png")


@app.get("/api/download/report")
def download_report() -> HTMLResponse:
    report_html = _build_report_html()
    return HTMLResponse(
        content=report_html,
        headers={"Content-Disposition": 'attachment; filename="backtest_report.html"'},
    )
