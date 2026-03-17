"""
Feature engineering / supervised dataset construction (Phase 2 + 3).

Phase 2: Build supervised dataset (features at t, target = return at t+1).
Phase 3: Configurable feature selection; pipeline works with feature subsets
         (e.g. RIGHT_SIDE_FEATURES only) for Black-Litterman–ready ML views.

Critical safety:
- Target is created with per-asset shifting only: groupby(asset_id).shift(-1)
- Features are never computed using future returns
- No global shift is used (prevents cross-asset leakage)
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Set

import pandas as pd

# Non-feature columns that must never be treated as selectable features.
_RESERVED_COLUMNS: Set[str] = {"date", "asset_id", "return", "target_return"}


def get_available_features(
    df: pd.DataFrame,
    requested: Sequence[str],
    *,
    key_cols: Optional[Sequence[str]] = None,
    warn_missing: bool = True,
) -> List[str]:
    """
    Return the subset of requested feature columns that exist in the dataframe.

    Keys (e.g. date, asset_id) and target (target_return) are never included
    in the returned list; only feature columns are considered.

    Parameters
    ----------
    df : DataFrame
        Panel with characteristics (and optionally returns).
    requested : sequence of str
        Configured feature names (e.g. from config.get_active_feature_set()).
    key_cols : optional
        Column names to exclude from the "available" set. Defaults to
        date, asset_id (and target_return is always excluded).
    warn_missing : bool
        If True, emit a warning listing any requested columns not in df.

    Returns
    -------
    List of column names that are in both requested and df.columns, and
    are not in key_cols or reserved (target_return, return).
    """
    if key_cols is None:
        key_cols = ("date", "asset_id")
    excluded = set(key_cols) | _RESERVED_COLUMNS
    available = [c for c in requested if c in df.columns and c not in excluded]
    missing = [c for c in requested if c not in excluded and c not in df.columns]
    if warn_missing and missing:
        warnings.warn(
            f"Feature selection: {len(missing)} configured feature(s) not in dataset: {missing[:10]}{'...' if len(missing) > 10 else ''}. Using {len(available)} available.",
            UserWarning,
            stacklevel=2,
        )
    return available


def validate_requested_features(
    df: pd.DataFrame,
    requested: Sequence[str],
    *,
    key_cols: Optional[Sequence[str]] = None,
) -> None:
    """
    Raise if any requested feature column is missing from the dataframe.

    Use for strict validation when the pipeline must have all requested
    features. Keys (date, asset_id) and target_return are not validated.
    """
    if key_cols is None:
        key_cols = ("date", "asset_id")
    excluded = set(key_cols) | _RESERVED_COLUMNS
    missing = [c for c in requested if c not in excluded and c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required feature column(s): {missing}. "
            f"Available columns: {[c for c in df.columns if c not in excluded]}."
        )


def build_supervised_dataset(
    chars_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    *,
    date_col: str = "date",
    asset_col: str = "asset_id",
    return_col: str = "return",
    feature_cols: Optional[Sequence[str]] = None,
    target_col: str = "target_return",
    use_available_subset: bool = True,
) -> pd.DataFrame:
    """
    Build a supervised learning dataset:
    - join current-period features with next-period realized returns as target

    Steps
    - sort returns by (asset_id, date)
    - compute next-period return per asset via shift(-1)
    - merge target onto characteristics by (date, asset_id)
    - drop rows with missing targets

    When feature_cols is provided and use_available_subset is True (default),
    only columns that exist in chars_df are used; a warning is emitted for
    any configured features missing from the dataset. This keeps the pipeline
    working when only a subset (e.g. RIGHT_SIDE_FEATURES) is present.

    Output columns
    - date
    - asset_id
    - <feature columns...>
    - target_return
    """

    required_chars = {date_col, asset_col}
    required_returns = {date_col, asset_col, return_col}
    missing_chars = [c for c in required_chars if c not in chars_df.columns]
    missing_returns = [c for c in required_returns if c not in returns_df.columns]
    if missing_chars:
        raise ValueError(f"chars_df missing required column(s): {missing_chars}")
    if missing_returns:
        raise ValueError(f"returns_df missing required column(s): {missing_returns}")

    if feature_cols is None:
        feature_cols = [c for c in chars_df.columns if c not in {date_col, asset_col}]
        if not feature_cols:
            raise ValueError("No feature columns found in chars_df.")
        feature_cols = list(feature_cols)
    else:
        feature_cols = list(feature_cols)
        if use_available_subset:
            feature_cols = get_available_features(
                chars_df,
                feature_cols,
                key_cols=(date_col, asset_col),
                warn_missing=True,
            )
            if not feature_cols:
                raise ValueError(
                    "None of the requested feature columns exist in chars_df. "
                    "Check column names and config feature list."
                )
        else:
            missing_feat = [c for c in feature_cols if c not in chars_df.columns]
            if missing_feat:
                raise ValueError(f"chars_df is missing feature column(s): {missing_feat}")

    # Ensure deterministic ordering before shifting.
    r = returns_df[[date_col, asset_col, return_col]].copy()
    r = r.sort_values([asset_col, date_col], kind="mergesort").reset_index(drop=True)

    # Per-asset next-period return target.
    r[target_col] = r.groupby(asset_col, sort=False)[return_col].shift(-1)

    # Merge target onto characteristics at the same date and asset.
    # Interpretation: features at date t predict return at t+1 (stored on row t).
    merged = chars_df[[date_col, asset_col, *list(feature_cols)]].merge(
        r[[date_col, asset_col, target_col]],
        on=[date_col, asset_col],
        how="inner",
        validate="one_to_one",
    )

    merged = merged.sort_values([asset_col, date_col], kind="mergesort").reset_index(drop=True)

    # Drop rows where the next-period return doesn't exist (e.g., last date per asset).
    merged = merged.dropna(subset=[target_col]).reset_index(drop=True)

    if merged[target_col].isna().any():
        raise AssertionError("Sanity check failed: null target_return remains after dropna.")

    return merged[[date_col, asset_col, *list(feature_cols), target_col]].copy()


def build_features(*args, **kwargs):
    """Backward-compatible alias for earlier scaffold (prefer `build_supervised_dataset`)."""

    return build_supervised_dataset(*args, **kwargs)

