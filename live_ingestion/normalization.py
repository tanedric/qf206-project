"""
Train-time feature normalization for the live-compatible pipeline.

The transform is fit on training rows only, persisted, and then reused unchanged
for validation, test, backfill-forward scoring, and live scoring.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class NormalizationState:
    feature_cols: List[str]
    log1p_features: List[str] = field(default_factory=list)
    fill_medians: Dict[str, float] = field(default_factory=dict)
    clip_lower: Dict[str, float] = field(default_factory=dict)
    clip_upper: Dict[str, float] = field(default_factory=dict)
    means: Dict[str, float] = field(default_factory=dict)
    stds: Dict[str, float] = field(default_factory=dict)


def _coerce_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def fit_normalization_state(
    train_df: pd.DataFrame,
    candidate_feature_cols: Sequence[str],
    *,
    log1p_features: Iterable[str] = (),
    max_missing_frac: float = 0.85,
    min_std: float = 1e-12,
    clip_quantiles: tuple[float, float] = (0.01, 0.99),
) -> NormalizationState:
    """
    Fit a per-feature transform on training rows only.

    Steps per feature:
    - drop very sparse or near-constant features
    - optional log1p on non-negative features
    - train-only median fill
    - train-only clipping
    - train-only z-score standardization
    """

    log1p_set = set(log1p_features)
    state = NormalizationState(feature_cols=[])

    for feature in candidate_feature_cols:
        if feature not in train_df.columns:
            continue
        raw = _coerce_series(train_df[feature])
        if float(raw.isna().mean()) > max_missing_frac:
            continue

        transformed = raw.copy()
        use_log = feature in log1p_set
        if use_log:
            transformed = transformed.where(transformed >= 0)
            transformed = np.log1p(transformed)

        valid = transformed.dropna()
        if valid.empty or valid.nunique() <= 1:
            continue

        median = float(valid.median())
        lower = float(valid.quantile(clip_quantiles[0]))
        upper = float(valid.quantile(clip_quantiles[1]))
        filled = transformed.fillna(median).clip(lower=lower, upper=upper)
        mean = float(filled.mean())
        std = float(filled.std())
        if not np.isfinite(std) or std < min_std:
            continue

        state.feature_cols.append(feature)
        if use_log:
            state.log1p_features.append(feature)
        state.fill_medians[feature] = median
        state.clip_lower[feature] = lower
        state.clip_upper[feature] = upper
        state.means[feature] = mean
        state.stds[feature] = std

    if not state.feature_cols:
        raise ValueError("No live features remain after train-only normalization fitting.")
    return state


def apply_normalization(df: pd.DataFrame, state: NormalizationState) -> pd.DataFrame:
    out = df.copy()
    log1p_set = set(state.log1p_features)
    for feature in state.feature_cols:
        values = _coerce_series(out[feature]) if feature in out.columns else pd.Series(index=out.index, dtype=float)
        if feature in log1p_set:
            values = values.where(values >= 0)
            values = np.log1p(values)
        values = values.fillna(state.fill_medians[feature])
        values = values.clip(
            lower=state.clip_lower[feature],
            upper=state.clip_upper[feature],
        )
        values = (values - state.means[feature]) / state.stds[feature]
        out[feature] = values
    return out


def save_normalization_state(state: NormalizationState, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def load_normalization_state(path: str | Path) -> NormalizationState:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return NormalizationState(**payload)
