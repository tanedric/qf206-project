"""
Run the separate live-compatible vendor ingestion, training, and scoring workflow.

This path is intentionally separate from the historical OAP/PERMNO pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.live_api import resolve_api_keys
from live_ingestion.backfill import (
    LiveBackfillConfig,
    load_tickers_from_csv,
    run_live_backfill_and_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill vendor-computable live features and score forward monthly returns."
    )
    parser.add_argument(
        "--ticker_source_csv",
        default="data/raw/returns.csv",
        help="CSV used to source the live ticker universe.",
    )
    parser.add_argument(
        "--ticker_col",
        default="asset_id",
        help="Ticker column inside --ticker_source_csv.",
    )
    parser.add_argument(
        "--oap_csv",
        default="data/raw/signed_predictors_dl_wide.csv",
        help="Historical OAP CSV used only to infer the default post-OAP start month.",
    )
    parser.add_argument(
        "--start_month_override",
        default=None,
        help="Optional override start month (for example 2020-06).",
    )
    parser.add_argument(
        "--as_of_date",
        default=pd.Timestamp.today().normalize().strftime("%Y-%m-%d"),
        help="As-of date for backfill/scoring. Default: today.",
    )
    parser.add_argument(
        "--output_dir",
        default="artifacts/live",
        help="Directory for feature panel, missing report, model artifacts, and predictions.",
    )
    parser.add_argument(
        "--max_tickers",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--polygon_min_interval_seconds",
        type=float,
        default=12.0,
        help="Minimum seconds between Polygon requests. Increase this if you hit HTTP 429.",
    )
    parser.add_argument(
        "--finnhub_min_interval_seconds",
        type=float,
        default=1.0,
        help="Minimum seconds between Finnhub requests. Increase this if you hit HTTP 429.",
    )
    parser.add_argument(
        "--http_max_retries",
        type=int,
        default=5,
        help="Retries for temporary HTTP failures such as 429 or 503.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce progress logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tickers = load_tickers_from_csv(
        args.ticker_source_csv,
        ticker_col=args.ticker_col,
        max_tickers=args.max_tickers,
    )
    if not tickers:
        raise SystemExit(
            f"No tickers found in {args.ticker_source_csv} column '{args.ticker_col}'."
        )

    polygon_api_key, finnhub_api_key = resolve_api_keys()
    config = LiveBackfillConfig(
        tickers=tickers,
        oap_csv=Path(args.oap_csv),
        output_dir=Path(args.output_dir),
        as_of_date=pd.Timestamp(args.as_of_date).normalize(),
        start_month_override=args.start_month_override,
        max_tickers=args.max_tickers,
        polygon_min_interval_seconds=args.polygon_min_interval_seconds,
        finnhub_min_interval_seconds=args.finnhub_min_interval_seconds,
        http_max_retries=args.http_max_retries,
        verbose=not args.quiet,
    )
    result = run_live_backfill_and_score(
        config,
        polygon_api_key=polygon_api_key,
        finnhub_api_key=finnhub_api_key,
    )

    print("=== Live-Compatible Pipeline ===")
    print(f"Tickers: {len(tickers)}")
    if len(tickers) <= 10:
        print(f"Processed tickers: {tickers}")
    print(f"Latest local OAP month: {result.latest_oap_month}")
    print(f"Computed start month: {result.computed_start_month}")
    print(f"As-of date: {result.computed_end_as_of_date}")
    print(f"Feature panel rows: {result.n_feature_rows}")
    print(f"Missing-feature rows: {result.n_missing_rows}")
    print(f"Forward predictions: {result.n_live_prediction_rows}")
    if result.selected_model_name is not None:
        print(f"Selected model: {result.selected_model_name}")
        print(f"Used features: {result.used_features}")
    print(f"Feature panel saved to: {result.feature_panel_path}")
    print(f"Missing report saved to: {result.missing_report_path}")
    print(f"Run metadata saved to: {result.metadata_path}")
    if result.live_predictions_path is not None:
        print(f"Live predictions saved to: {result.live_predictions_path}")
        live_predictions = pd.read_csv(result.live_predictions_path)
        if live_predictions.empty:
            print("Live predictions: none")
        else:
            print("Live predictions:")
            print(live_predictions.to_string(index=False))
    if result.model_path is not None:
        print(f"Model artifact saved to: {result.model_path}")
    if result.normalization_path is not None:
        print(f"Normalization artifact saved to: {result.normalization_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
