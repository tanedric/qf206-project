"""
Preprocessing utilities (Phase 2 implementation).

This module provides minimal, safe preprocessing for firm characteristics:
- standardize column names (lightweight)
- sort deterministically by asset and date
- handle duplicates
- fill missing feature values using same-date (cross-sectional) medians, with a
  global median fallback

Critical safety: this module does NOT create targets and does NOT use future
information.

## Data contract: Processed model dataset

Expected as a pandas.DataFrame named `model_df`.

### Required columns (panel format)
- **date**: datetime64[ns]
- **asset_id**: str/int
- **x_001 ... x_094**: float; processed features aligned to `date`
- **y_return_next**: float; realized next-period return (label/target)

### Optional columns
- **weight**: float; sample weight (e.g., market-cap weights) for training
- **is_train / is_valid / is_test**: bool; split indicators (optional)
- **sentiment_score**: float; if/when sentiment is merged in later

### Key invariants
- One row per (date, asset_id).
- Features must be aligned so that using x at `date` predicts y at `date + 1`.
- No look-ahead bias: y uses future info, x uses only contemporaneous/past info.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreprocessResult:
    """Return type for preprocessing that optionally includes metadata."""

    df: pd.DataFrame
    feature_cols: List[str]


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal standardization:
    - strip whitespace
    - lowercase
    - replace spaces with underscores

    This avoids surprising mismatches like 'Asset ID' vs 'asset_id'.
    """

    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def _validate_required_columns(df: pd.DataFrame, required: Iterable[str], *, name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required column(s): {missing}. Found: {list(df.columns)}")


def _handle_duplicates(
    df: pd.DataFrame,
    *,
    key_cols: Sequence[str],
    mode: str = "error",
) -> pd.DataFrame:
    """
    Handle duplicates on key columns.

    - mode='error': raise if any duplicated keys exist
    - mode='drop': drop exact-duplicate rows; raise on conflicting duplicates
    """

    dup_mask = df.duplicated(subset=list(key_cols), keep=False)
    if not bool(dup_mask.any()):
        return df

    if mode not in {"error", "drop"}:
        raise ValueError("duplicate_mode must be one of: {'error', 'drop'}")

    if mode == "error":
        sample = df.loc[dup_mask, list(key_cols)].head(10)
        raise ValueError(
            f"Found duplicate rows on keys {list(key_cols)}. "
            f"Sample duplicated keys:\n{sample}"
        )

    # mode == 'drop'
    # First drop rows that are fully identical.
    deduped = df.drop_duplicates()

    # If keys are still duplicated, we have conflicting duplicates.
    still_dup = deduped.duplicated(subset=list(key_cols), keep=False)
    if bool(still_dup.any()):
        sample = deduped.loc[still_dup, list(key_cols)].head(10)
        raise ValueError(
            "Found conflicting duplicates (same keys, different values) after dropping "
            f"exact duplicates. Keys: {list(key_cols)}. Sample:\n{sample}"
        )

    return deduped


def fill_missing_features_by_date_median(
    df: pd.DataFrame,
    *,
    date_col: str,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Fill missing feature values using cross-sectional median within each date,
    with a global median fallback for dates where a feature is all-NaN.
    """

    out = df.copy()

    # Ensure numeric where possible; non-numeric feature cols become NaN.
    for c in feature_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Same-date median fill.
    by_date_median = out.groupby(date_col, sort=False)[list(feature_cols)].transform("median")
    out[list(feature_cols)] = out[list(feature_cols)].fillna(by_date_median)

    # Global median fallback (per feature).
    global_median = out[list(feature_cols)].median(axis=0, skipna=True)
    out[list(feature_cols)] = out[list(feature_cols)].fillna(global_median)

    # If a feature is entirely missing, median is NaN -> fail fast.
    still_missing = out[list(feature_cols)].isna().any()
    bad_cols = still_missing[still_missing].index.tolist()
    if bad_cols:
        raise ValueError(
            "After median filling, some feature columns are still missing values. "
            "This usually means the entire column is NaN/non-numeric. "
            f"Columns: {bad_cols}"
        )

    return out


def preprocess_firm_characteristics(
    raw_chars_df: pd.DataFrame,
    *,
    date_col: str = "date",
    asset_col: str = "asset_id",
    feature_cols: Optional[Sequence[str]] = None,
    duplicate_mode: str = "drop",
    standardize_cols: bool = True,
) -> PreprocessResult:
    """
    Preprocess firm characteristics into a clean feature panel.

    Returns a `PreprocessResult` containing:
    - `df`: cleaned DataFrame with keys + feature columns
    - `feature_cols`: list of feature column names used
    """

    df = standardize_column_names(raw_chars_df) if standardize_cols else raw_chars_df.copy()
    _validate_required_columns(df, [date_col, asset_col], name="raw firm characteristics")

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in {date_col, asset_col}]
        if not feature_cols:
            raise ValueError(
                "No feature columns detected. Provide `feature_cols` or include non-key columns."
            )
    else:
        _validate_required_columns(df, feature_cols, name="raw firm characteristics (feature_cols)")

    # Deterministic sorting before any group operations.
    df = df.sort_values([asset_col, date_col], kind="mergesort").reset_index(drop=True)

    # Dedupe by (date, asset_id).
    df = _handle_duplicates(df, key_cols=[date_col, asset_col], mode=duplicate_mode)

    # Fill missing feature values (MVP) without touching any targets.
    df = fill_missing_features_by_date_median(df, date_col=date_col, feature_cols=list(feature_cols))

    # Lightweight validation.
    if df[date_col].isna().any():
        raise ValueError(f"Found nulls in '{date_col}' after preprocessing.")
    if df[asset_col].isna().any():
        raise ValueError(f"Found nulls in '{asset_col}' after preprocessing.")

    return PreprocessResult(df=df[[date_col, asset_col, *list(feature_cols)]].copy(), feature_cols=list(feature_cols))


def preprocess_raw_inputs(*args, **kwargs):
    """Backward-compatible alias for earlier scaffold (prefer `preprocess_firm_characteristics`)."""

    return preprocess_firm_characteristics(*args, **kwargs)

