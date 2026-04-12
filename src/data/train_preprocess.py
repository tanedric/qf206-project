"""
Train-only feature pruning and imputation (no target leakage).

Used after chronological split: statistics are fit on the training subset only,
then applied to validation and test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureTransformState:
    """Fitted on train; used to transform val/test."""

    feature_cols: List[str]
    train_global_median: Dict[str, float] = field(default_factory=dict)
    train_date_medians: Dict[str, pd.Series] = field(default_factory=dict)  # col -> Series indexed by date


def coerce_numeric_features(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in feature_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def prune_features_train_only(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    date_col: str,
    max_missing_frac: float = 0.85,
    min_std: float = 1e-12,
) -> List[str]:
    """
    Drop columns with > max_missing_frac NaN in train, or constant / near-constant.
    """
    train_df = coerce_numeric_features(train_df, feature_cols)
    n = len(train_df)
    if n == 0:
        return []

    keep: List[str] = []
    for c in feature_cols:
        col = train_df[c]
        miss_frac = float(col.isna().mean())
        if miss_frac > max_missing_frac:
            continue
        valid = col.dropna()
        if valid.empty:
            continue
        if valid.nunique() <= 1:
            continue
        if float(valid.std()) < min_std:
            continue
        keep.append(c)
    return keep


def fit_imputation_from_train(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    date_col: str,
) -> FeatureTransformState:
    """Cross-sectional median by date (train only), then global median per column."""
    train_df = coerce_numeric_features(train_df, feature_cols)
    state = FeatureTransformState(feature_cols=list(feature_cols))

    for c in feature_cols:
        by_date = train_df.groupby(date_col, sort=False)[c].median()
        state.train_date_medians[c] = by_date
        state.train_global_median[c] = float(train_df[c].median(skipna=True))

    return state


def apply_imputation(
    df: pd.DataFrame,
    state: FeatureTransformState,
    *,
    date_col: str,
) -> pd.DataFrame:
    out = coerce_numeric_features(df, state.feature_cols)
    for c in state.feature_cols:
        if c not in out.columns:
            continue
        # Map each row's date to train date-median; missing dates -> global train median
        med_by_date = state.train_date_medians.get(c)
        if med_by_date is not None and len(med_by_date) > 0:
            mapped = out[date_col].map(med_by_date)
            out[c] = out[c].fillna(mapped)
        g = state.train_global_median.get(c, np.nan)
        if np.isfinite(g):
            out[c] = out[c].fillna(g)
    return out


def preprocess_features_train_val_test(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    date_col: str,
    asset_col: str,
    target_col: str,
    candidate_feature_cols: Sequence[str],
    max_missing_frac: float = 0.85,
    min_std: float = 1e-12,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], FeatureTransformState]:
    """
    Prune features on train; fit imputation on train; apply to val/test.

    Preserves date, asset_id, target_col columns.
    """
    key_and_target = [date_col, asset_col, target_col]
    for k in key_and_target:
        if k not in train_df.columns:
            raise ValueError(f"train_df missing '{k}'")

    kept = prune_features_train_only(
        train_df,
        candidate_feature_cols,
        date_col=date_col,
        max_missing_frac=max_missing_frac,
        min_std=min_std,
    )
    if not kept:
        raise ValueError("No features remain after train-only pruning. Check data / thresholds.")

    state = fit_imputation_from_train(train_df, kept, date_col=date_col)

    def _subset_and_apply(d: pd.DataFrame) -> pd.DataFrame:
        base = d[key_and_target].copy()
        feat_part = apply_imputation(d, state, date_col=date_col)[kept]
        return pd.concat([base, feat_part], axis=1)

    train_out = _subset_and_apply(train_df)
    val_out = _subset_and_apply(val_df)
    test_out = _subset_and_apply(test_df)

    # Final safety: drop rows with non-finite target (should already be clean)
    train_out = train_out.dropna(subset=[target_col]).reset_index(drop=True)
    val_out = val_out.dropna(subset=[target_col]).reset_index(drop=True)
    test_out = test_out.dropna(subset=[target_col]).reset_index(drop=True)

    # Ensure finite features
    for d in (train_out, val_out, test_out):
        X = d[kept].to_numpy(dtype=float, copy=False)
        if not np.isfinite(X).all():
            raise ValueError("Non-finite values remain after imputation.")

    return train_out, val_out, test_out, kept, state
