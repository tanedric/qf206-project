"""
Minimal runnable demo: ML return-view generator (Phase 4+5).

What it does
- loads firm characteristics CSV + returns CSV
- preprocesses characteristics
- builds supervised dataset with next-period target_return
- selects RIGHT_SIDE_FEATURES (configurable feature groups)
- splits chronologically (train early dates, test later dates)
- fits a baseline ElasticNet model
- outputs standardized predictions: date, asset_id, predicted_return

Example (PowerShell)
python scripts/run_return_prediction_demo.py `
  --chars_csv "data/raw/firm_characteristics.csv" `
  --returns_csv "data/raw/returns.csv" `
  --out_csv "artifacts/predictions/predicted_returns.csv"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running as a script from repo root without installing as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_active_feature_set, get_paths
from src.data.load_data import load_firm_characteristics_csv, load_returns_csv
from src.data.preprocess import preprocess_firm_characteristics
from src.data.feature_engineering import build_supervised_dataset, get_available_features
from src.models.baseline_model import BaselineModel


def chronological_split(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: earliest dates -> train, latest dates -> test."""

    if date_col not in df.columns:
        raise ValueError(f"Missing '{date_col}' column for splitting.")
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must be between 0 and 1.")

    dates = pd.Series(df[date_col].unique()).sort_values()
    if len(dates) < 2:
        raise ValueError("Need at least 2 unique dates for a chronological split.")

    cutoff_idx = max(1, int(round((1.0 - test_fraction) * len(dates)))) - 1
    cutoff_date = dates.iloc[cutoff_idx]

    train_df = df[df[date_col] <= cutoff_date].copy()
    test_df = df[df[date_col] > cutoff_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Chronological split produced empty train/test. cutoff_date={cutoff_date}"
        )
    return train_df, test_df


def run_demo(
    *,
    chars_csv: str,
    returns_csv: str,
    out_csv: str | None,
    date_col: str = "date",
    asset_col: str = "asset_id",
    return_col: str = "return",
    test_fraction: float = 0.2,
) -> None:
    paths = get_paths()

    chars_raw = load_firm_characteristics_csv(chars_csv, date_col=date_col, asset_col=asset_col)
    returns = load_returns_csv(
        returns_csv, date_col=date_col, asset_col=asset_col, return_col=return_col
    )

    # Feature group selection (default: RIGHT_SIDE_FEATURES)
    requested_features = get_active_feature_set()
    available_features = get_available_features(
        chars_raw, requested_features, key_cols=(date_col, asset_col), warn_missing=True
    )

    # Preprocess only the available features (MVP).
    prep = preprocess_firm_characteristics(
        chars_raw,
        date_col=date_col,
        asset_col=asset_col,
        feature_cols=available_features,
        duplicate_mode="drop",
        standardize_cols=True,
    )

    supervised = build_supervised_dataset(
        prep.df,
        returns,
        date_col=date_col,
        asset_col=asset_col,
        return_col=return_col,
        feature_cols=prep.feature_cols,
        target_col="target_return",
        use_available_subset=False,  # already filtered by preprocess step
    )

    train_df, test_df = chronological_split(supervised, date_col=date_col, test_fraction=test_fraction)

    model = BaselineModel(feature_set_name="right")
    model.fit(train_df, prep.feature_cols, target_col="target_return")
    pred_df = model.predict(test_df)

    print("=== Demo: ML return-view generator ===")
    print(f"Requested feature count: {len(requested_features)}")
    print(f"Available feature count: {len(prep.feature_cols)}")
    print(f"Supervised dataset shape: {supervised.shape}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print()
    print("Predictions head:")
    print(pred_df.head(10).to_string(index=False))

    if out_csv:
        out_path = Path(out_csv)
        if not out_path.is_absolute():
            out_path = paths.project_root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(out_path, index=False)
        print()
        print(f"Saved predictions to: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baseline ML return prediction demo.")
    p.add_argument("--chars_csv", required=True, help="Path to firm characteristics CSV.")
    p.add_argument("--returns_csv", required=True, help="Path to realized returns CSV.")
    p.add_argument("--out_csv", default=None, help="Optional output CSV path for predictions.")
    p.add_argument("--test_fraction", type=float, default=0.2, help="Fraction of dates in test.")
    p.add_argument("--date_col", default="date")
    p.add_argument("--asset_col", default="asset_id")
    p.add_argument("--return_col", default="return")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(
        chars_csv=args.chars_csv,
        returns_csv=args.returns_csv,
        out_csv=args.out_csv,
        test_fraction=args.test_fraction,
        date_col=args.date_col,
        asset_col=args.asset_col,
        return_col=args.return_col,
    )

