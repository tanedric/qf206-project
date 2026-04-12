"""
Backfill, train, and score orchestration for the live-compatible pipeline.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from live_ingestion.feature_formulas import (
    FEATURE_SOURCE_MAP,
    LIVE_ALL_FEATURES,
    FeatureComputationOutput,
    compute_fundamental_feature_panel,
    compute_market_feature_panel,
)
from live_ingestion.fundamentals_adapter import FinnhubFundamentalsAdapter
from live_ingestion.market_adapter import PolygonMarketAdapter
from live_ingestion.normalization import (
    NormalizationState,
    apply_normalization,
    fit_normalization_state,
    save_normalization_state,
)
from src.data.feature_engineering import chronological_split_three_way
from src.models.evaluation_metrics import EvaluationReport, evaluate_predictions
from src.models.return_predictors import build_candidate_models


@dataclass
class LiveBackfillConfig:
    tickers: Sequence[str]
    oap_csv: Path
    output_dir: Path
    as_of_date: pd.Timestamp
    start_month_override: Optional[str] = None
    stale_fundamental_days: int = 540
    max_tickers: Optional[int] = None
    random_state: int = 42
    polygon_min_interval_seconds: float = 12.0
    finnhub_min_interval_seconds: float = 1.0
    http_max_retries: int = 5
    verbose: bool = True


@dataclass
class LivePipelineResult:
    latest_oap_month: Optional[str]
    computed_start_month: Optional[str]
    computed_end_as_of_date: str
    feature_panel_path: str
    missing_report_path: str
    live_predictions_path: Optional[str]
    model_path: Optional[str]
    normalization_path: Optional[str]
    metadata_path: str
    selected_model_name: Optional[str]
    used_features: List[str]
    n_feature_rows: int
    n_missing_rows: int
    n_live_prediction_rows: int
    evaluation: Optional[EvaluationReport] = None


def _normalize_month(value: str | pd.Timestamp) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce", utc=False)
    if pd.isna(parsed):
        raise ValueError(f"Failed to parse month value: {value}")
    return pd.Timestamp(parsed).to_period("M").to_timestamp("M")


def _current_as_of_timestamp() -> pd.Timestamp:
    return pd.Timestamp(date.today()).normalize()


def infer_latest_oap_month(oap_csv: str | Path) -> Optional[pd.Timestamp]:
    path = Path(oap_csv)
    if not path.exists():
        return None
    sample = pd.read_csv(path, usecols=["yyyymm"])
    if sample.empty:
        return None
    yyyymm = sample["yyyymm"].dropna()
    if yyyymm.empty:
        return None
    latest = str(yyyymm.astype(str).str.replace(r"\.0$", "", regex=True).max())
    parsed = pd.to_datetime(latest, format="%Y%m", errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).to_period("M").to_timestamp("M")


def resolve_start_month(config: LiveBackfillConfig) -> Optional[pd.Timestamp]:
    if config.start_month_override:
        return _normalize_month(config.start_month_override)
    latest_oap_month = infer_latest_oap_month(config.oap_csv)
    if latest_oap_month is None:
        return None
    return latest_oap_month + pd.offsets.MonthEnd(1)


def build_processing_dates(start_month: pd.Timestamp, as_of_date: pd.Timestamp) -> List[pd.Timestamp]:
    start_month = _normalize_month(start_month)
    as_of_date = pd.Timestamp(as_of_date).normalize()
    current_month = as_of_date.to_period("M").to_timestamp("M")
    if start_month > current_month:
        return []

    month_ends = pd.date_range(start=start_month, end=current_month, freq="ME")
    processing_dates: List[pd.Timestamp] = []
    for month_end in month_ends:
        processing_dates.append(pd.Timestamp(min(month_end, as_of_date)).normalize())
    return processing_dates


def load_tickers_from_csv(
    csv_path: str | Path,
    *,
    ticker_col: str = "asset_id",
    max_tickers: Optional[int] = None,
) -> List[str]:
    df = pd.read_csv(csv_path, usecols=[ticker_col])
    tickers = sorted(
        {
            str(value).strip().upper()
            for value in df[ticker_col].dropna().tolist()
            if str(value).strip()
        }
    )
    if max_tickers is not None:
        tickers = tickers[: max(0, int(max_tickers))]
    return tickers


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _save_pickle(path: str | Path, obj: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(obj, handle)


def _merge_feature_outputs(outputs: Iterable[FeatureComputationOutput]) -> FeatureComputationOutput:
    panels = [o.panel for o in outputs if o.panel is not None and not o.panel.empty]
    missing_parts = [o.missing for o in outputs if o.missing is not None and not o.missing.empty]
    panel = pd.concat(panels, axis=0, ignore_index=True) if panels else pd.DataFrame()
    missing = pd.concat(missing_parts, axis=0, ignore_index=True) if missing_parts else pd.DataFrame()
    if panel.empty:
        panel = pd.DataFrame(columns=["date", "ticker", *LIVE_ALL_FEATURES, "target_return"])
    if missing.empty:
        missing = pd.DataFrame(columns=["date", "ticker", "feature", "reason", "source"])
    return FeatureComputationOutput(panel=panel, missing=missing)


def _build_panel_for_one_ticker(
    ticker: str,
    *,
    polygon: PolygonMarketAdapter,
    finnhub: FinnhubFundamentalsAdapter,
    evaluation_dates: Sequence[pd.Timestamp],
    as_of_date: pd.Timestamp,
    stale_fundamental_days: int,
) -> FeatureComputationOutput:
    if not evaluation_dates:
        return FeatureComputationOutput(panel=pd.DataFrame(), missing=pd.DataFrame())

    first_month = evaluation_dates[0].to_period("M").to_timestamp("M")
    lookback_start = (first_month - pd.DateOffset(months=36)).to_period("M").to_timestamp(how="start")
    daily_bars = polygon.fetch_daily_bars(
        ticker,
        start_date=lookback_start.date(),
        end_date=as_of_date.date(),
    )

    quarterly = finnhub.fetch_financials_reported(ticker, freq="quarterly")
    annual = finnhub.fetch_financials_reported(ticker, freq="annual")
    profile = finnhub.fetch_company_profile(ticker)

    market_preview = compute_market_feature_panel(
        ticker,
        daily_bars,
        evaluation_dates=evaluation_dates,
        shares_by_date={},
        as_of_date=as_of_date,
    )
    month_close_by_date = {
        pd.Timestamp(row["date"]).normalize(): float(row["month_close"])
        for _, row in market_preview.panel.iterrows()
        if pd.notna(row.get("month_close"))
    }

    fundamentals = compute_fundamental_feature_panel(
        ticker,
        quarterly_filings=quarterly,
        annual_filings=annual,
        evaluation_dates=evaluation_dates,
        month_close_by_date=month_close_by_date,
        current_profile=profile,
        stale_threshold_days=stale_fundamental_days,
        as_of_date=as_of_date,
    )
    shares_by_date = {
        pd.Timestamp(row["date"]).normalize(): float(row["shares_outstanding"])
        for _, row in fundamentals.panel.iterrows()
        if pd.notna(row.get("shares_outstanding"))
    }

    market = compute_market_feature_panel(
        ticker,
        daily_bars,
        evaluation_dates=evaluation_dates,
        shares_by_date=shares_by_date,
        as_of_date=as_of_date,
    )

    merged = market.panel.merge(
        fundamentals.panel,
        on=["date", "ticker"],
        how="outer",
        validate="one_to_one",
    )
    merged = merged.sort_values("date", kind="mergesort").reset_index(drop=True)
    merged["std_turn"] = (
        pd.to_numeric(merged["turn"], errors="coerce")
        .rolling(window=36, min_periods=12)
        .std()
    )

    std_turn_missing_rows: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        date_value = pd.Timestamp(row["date"]).normalize()
        if pd.isna(row["std_turn"]):
            if pd.isna(row["turn"]):
                reason = "turn_missing_so_std_turn_unavailable"
            else:
                reason = "fewer_than_12_months_of_turn_history"
            std_turn_missing_rows.append(
                {
                    "date": date_value,
                    "ticker": ticker.upper(),
                    "feature": "std_turn",
                    "reason": reason,
                    "source": "Polygon+Finnhub",
                }
            )

    missing = _merge_feature_outputs([market, fundamentals]).missing
    if std_turn_missing_rows:
        missing = pd.concat([missing, pd.DataFrame(std_turn_missing_rows)], axis=0, ignore_index=True)
    if missing.empty:
        missing = pd.DataFrame(columns=["date", "ticker", "feature", "reason", "source"])

    return FeatureComputationOutput(panel=merged, missing=missing)


def build_live_feature_panel(
    config: LiveBackfillConfig,
    *,
    polygon_api_key: str,
    finnhub_api_key: str,
) -> FeatureComputationOutput:
    start_month = resolve_start_month(config)
    if start_month is None:
        return FeatureComputationOutput(panel=pd.DataFrame(), missing=pd.DataFrame())

    evaluation_dates = build_processing_dates(start_month, config.as_of_date)
    if not evaluation_dates:
        return FeatureComputationOutput(panel=pd.DataFrame(), missing=pd.DataFrame())

    polygon = PolygonMarketAdapter(
        api_key=polygon_api_key,
        min_interval_seconds=config.polygon_min_interval_seconds,
        max_retries=config.http_max_retries,
    )
    finnhub = FinnhubFundamentalsAdapter(
        api_key=finnhub_api_key,
        min_interval_seconds=config.finnhub_min_interval_seconds,
        max_retries=config.http_max_retries,
    )

    outputs: List[FeatureComputationOutput] = []
    tickers = list(config.tickers)
    if config.max_tickers is not None:
        tickers = tickers[: max(0, int(config.max_tickers))]

    if config.verbose:
        print(
            "Building live feature panel "
            f"for {len(tickers)} ticker(s) across {len(evaluation_dates)} month(s)..."
        )
        print(
            "Request pacing: "
            f"Polygon={config.polygon_min_interval_seconds:.1f}s, "
            f"Finnhub={config.finnhub_min_interval_seconds:.1f}s"
        )

    for idx, ticker in enumerate(tickers, start=1):
        if config.verbose:
            print(f"[{idx}/{len(tickers)}] Fetching vendor data for {ticker}...")
        output = _build_panel_for_one_ticker(
            ticker,
            polygon=polygon,
            finnhub=finnhub,
            evaluation_dates=evaluation_dates,
            as_of_date=config.as_of_date,
            stale_fundamental_days=config.stale_fundamental_days,
        )
        if config.verbose:
            print(
                f"[{idx}/{len(tickers)}] Completed {ticker}: "
                f"{len(output.panel)} panel rows, {len(output.missing)} missing-feature notes"
            )
        outputs.append(output)

    merged = _merge_feature_outputs(outputs)
    panel = merged.panel
    if not panel.empty:
        base_cols = [
            "date",
            "ticker",
            "shares_outstanding",
            "fundamental_report_end_date",
            "fundamental_available_date",
            *LIVE_ALL_FEATURES,
            "target_return",
        ]
        keep_cols = [c for c in base_cols if c in panel.columns]
        panel = panel[keep_cols].sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    return FeatureComputationOutput(panel=panel, missing=merged.missing)


def _evaluate_and_select_model(
    normalized_train: pd.DataFrame,
    normalized_val: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    random_state: int,
) -> tuple[str, Any]:
    models = build_candidate_models(random_state=random_state, feature_set_name="live_vendor_features")
    y_val = normalized_val["target_return"].to_numpy(dtype=float, copy=False)
    best_name = None
    best_model = None
    best_rmse = float("inf")

    for model in models:
        try:
            model.fit(normalized_train, feature_cols, target_col="target_return")
            pred = model.predict(normalized_val)
            y_hat = pred["predicted_return"].to_numpy(dtype=float, copy=False)
            rmse = float(np.sqrt(np.mean((y_val - y_hat) ** 2)))
            if np.isfinite(rmse) and rmse < best_rmse:
                best_name = model.name
                best_model = model
                best_rmse = rmse
        except Exception:
            continue

    if best_model is None or best_name is None:
        raise RuntimeError("No live model trained successfully on the validation split.")
    return best_name, best_model


def train_and_score_live_model(
    panel: pd.DataFrame,
    *,
    output_dir: Path,
    random_state: int,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trainable = panel.dropna(subset=["target_return"]).copy()
    forward_rows = panel[panel["target_return"].isna()].copy()

    if trainable.empty or trainable["date"].nunique() < 3:
        return {
            "selected_model_name": None,
            "used_features": [],
            "evaluation": None,
            "live_predictions": pd.DataFrame(columns=["date", "ticker", "predicted_return"]),
            "model_path": None,
            "normalization_path": None,
            "metadata_path": None,
        }

    trainable = trainable.rename(columns={"ticker": "asset_id"})
    forward_rows_model = forward_rows.rename(columns={"ticker": "asset_id"})

    candidate_features = [c for c in LIVE_ALL_FEATURES if c in trainable.columns]
    train_df, val_df, test_df = chronological_split_three_way(
        trainable,
        date_col="date",
        train_frac=0.6,
        val_frac=0.2,
        test_frac=0.2,
    )

    normalization_state = fit_normalization_state(
        train_df,
        candidate_features,
        log1p_features=("turn", "std_turn", "baspread", "RealizedVol", "Illiquidity", "rd"),
    )
    used_features = list(normalization_state.feature_cols)
    train_n = apply_normalization(train_df, normalization_state)
    val_n = apply_normalization(val_df, normalization_state)
    test_n = apply_normalization(test_df, normalization_state)

    selected_model_name, selected_model = _evaluate_and_select_model(
        train_n,
        val_n,
        feature_cols=used_features,
        random_state=random_state,
    )

    train_val = pd.concat([train_n, val_n], axis=0).sort_values(["asset_id", "date"], kind="mergesort")
    selected_model.fit(train_val, used_features, target_col="target_return")
    test_pred = selected_model.predict(test_n)
    eval_panel = test_pred.merge(
        test_n[["date", "asset_id", "target_return"]],
        on=["date", "asset_id"],
        how="left",
    )
    evaluation = evaluate_predictions(
        y_train=train_val["target_return"].to_numpy(dtype=float, copy=False),
        y_true=test_n["target_return"].to_numpy(dtype=float, copy=False),
        y_pred=test_pred["predicted_return"].to_numpy(dtype=float, copy=False),
        panel=eval_panel.rename(columns={"asset_id": "ticker"}),
        date_col="date",
        pred_col="predicted_return",
        realized_col="target_return",
    )

    final_model = selected_model
    final_train = apply_normalization(trainable, normalization_state)
    final_model.fit(final_train, used_features, target_col="target_return")

    if not forward_rows_model.empty:
        scored_forward = final_model.predict(apply_normalization(forward_rows_model, normalization_state))
        live_predictions = scored_forward.rename(columns={"asset_id": "ticker"})[
            ["date", "ticker", "predicted_return"]
        ].copy()
    else:
        live_predictions = pd.DataFrame(columns=["date", "ticker", "predicted_return"])

    model_path = output_dir / "live_model.pkl"
    normalization_path = output_dir / "normalization_state.json"
    model_metadata_path = output_dir / "model_metadata.json"
    test_predictions_path = output_dir / "test_predictions.csv"

    _save_pickle(model_path, final_model)
    save_normalization_state(normalization_state, normalization_path)
    test_pred.rename(columns={"asset_id": "ticker"})[["date", "ticker", "predicted_return"]].to_csv(
        test_predictions_path,
        index=False,
    )
    _save_json(
        model_metadata_path,
        {
            "selected_model_name": selected_model_name,
            "used_features": used_features,
            "evaluation": asdict(evaluation),
        },
    )

    return {
        "selected_model_name": selected_model_name,
        "used_features": used_features,
        "evaluation": evaluation,
        "live_predictions": live_predictions,
        "model_path": str(model_path),
        "normalization_path": str(normalization_path),
        "metadata_path": str(model_metadata_path),
    }


def run_live_backfill_and_score(
    config: LiveBackfillConfig,
    *,
    polygon_api_key: Optional[str],
    finnhub_api_key: Optional[str],
) -> LivePipelineResult:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latest_oap_month = infer_latest_oap_month(config.oap_csv)
    start_month = resolve_start_month(config)
    feature_panel_path = output_dir / "live_feature_panel.csv"
    missing_report_path = output_dir / "missing_feature_report.csv"
    metadata_path = output_dir / "run_metadata.json"
    live_predictions_path = output_dir / "live_predictions.csv"

    processing_dates = (
        build_processing_dates(start_month, config.as_of_date)
        if start_month is not None
        else []
    )

    if not processing_dates:
        pd.DataFrame(columns=["date", "ticker", *LIVE_ALL_FEATURES, "target_return"]).to_csv(
            feature_panel_path,
            index=False,
        )
        pd.DataFrame(columns=["date", "ticker", "feature", "reason", "source"]).to_csv(
            missing_report_path,
            index=False,
        )
        _save_json(
            metadata_path,
            {
                "latest_oap_month": latest_oap_month,
                "computed_start_month": start_month,
                "as_of_date": config.as_of_date,
                "tickers": list(config.tickers),
                "feature_source_map": FEATURE_SOURCE_MAP,
                "message": "No backfill months to process for the computed window.",
            },
        )
        return LivePipelineResult(
            latest_oap_month=latest_oap_month.strftime("%Y-%m") if latest_oap_month is not None else None,
            computed_start_month=start_month.strftime("%Y-%m") if start_month is not None else None,
            computed_end_as_of_date=config.as_of_date.strftime("%Y-%m-%d"),
            feature_panel_path=str(feature_panel_path),
            missing_report_path=str(missing_report_path),
            live_predictions_path=None,
            model_path=None,
            normalization_path=None,
            metadata_path=str(metadata_path),
            selected_model_name=None,
            used_features=[],
            n_feature_rows=0,
            n_missing_rows=0,
            n_live_prediction_rows=0,
            evaluation=None,
        )

    if not polygon_api_key or not finnhub_api_key:
        raise ValueError(
            "Missing API keys. Set POLYGON_API_KEY / FINNHUB_API_KEY in the environment "
            "or update config/live_api.py placeholders."
        )

    features = build_live_feature_panel(
        config,
        polygon_api_key=polygon_api_key,
        finnhub_api_key=finnhub_api_key,
    )
    panel = features.panel.copy()
    missing = features.missing.copy()
    panel.to_csv(feature_panel_path, index=False)
    missing.to_csv(missing_report_path, index=False)

    model_dir = output_dir / "model_artifacts"
    training = train_and_score_live_model(
        panel,
        output_dir=model_dir,
        random_state=config.random_state,
    )
    live_predictions = training["live_predictions"]
    if live_predictions is not None:
        live_predictions.to_csv(live_predictions_path, index=False)

    _save_json(
        metadata_path,
        {
            "latest_oap_month": latest_oap_month,
            "computed_start_month": start_month,
            "as_of_date": config.as_of_date,
            "tickers": list(config.tickers),
            "feature_source_map": FEATURE_SOURCE_MAP,
            "selected_model_name": training["selected_model_name"],
            "used_features": training["used_features"],
            "evaluation": asdict(training["evaluation"]) if training["evaluation"] is not None else None,
        },
    )

    return LivePipelineResult(
        latest_oap_month=latest_oap_month.strftime("%Y-%m") if latest_oap_month is not None else None,
        computed_start_month=start_month.strftime("%Y-%m") if start_month is not None else None,
        computed_end_as_of_date=config.as_of_date.strftime("%Y-%m-%d"),
        feature_panel_path=str(feature_panel_path),
        missing_report_path=str(missing_report_path),
        live_predictions_path=str(live_predictions_path),
        model_path=training["model_path"],
        normalization_path=training["normalization_path"],
        metadata_path=str(metadata_path),
        selected_model_name=training["selected_model_name"],
        used_features=training["used_features"],
        n_feature_rows=len(panel),
        n_missing_rows=len(missing),
        n_live_prediction_rows=len(live_predictions),
        evaluation=training["evaluation"],
    )
