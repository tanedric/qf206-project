"""
Out-of-sample metrics for return prediction (MVP; not a full backtest).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class EvaluationReport:
    rmse: float
    mae: float
    oos_r2: float
    mean_rank_ic: float
    top_bottom_spread: float


def _oos_r2(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """1 - MSE / MSE of predicting train mean (benchmark)."""
    mse = float(np.mean((y_true - y_pred) ** 2))
    bench = float(np.mean((y_true - float(np.mean(y_train))) ** 2))
    if bench <= 0 or not np.isfinite(bench):
        return float("nan")
    return 1.0 - mse / bench


def mean_spearman_ic_by_date(
    df: pd.DataFrame,
    *,
    date_col: str,
    pred_col: str,
    realized_col: str,
) -> float:
    """Average cross-sectional Spearman correlation per date."""
    ics: list[float] = []
    for _, g in df.groupby(date_col, sort=False):
        if len(g) < 3:
            continue
        sub = g[[pred_col, realized_col]].dropna()
        if len(sub) < 3:
            continue
        if sub[pred_col].nunique(dropna=True) <= 1 or sub[realized_col].nunique(dropna=True) <= 1:
            continue
        r = sub[pred_col].corr(sub[realized_col], method="spearman")
        if np.isfinite(r):
            ics.append(float(r))
    if not ics:
        return float("nan")
    return float(np.mean(ics))


def top_minus_bottom_decile_spread(
    df: pd.DataFrame,
    *,
    date_col: str,
    pred_col: str,
    realized_col: str,
    n_buckets: int = 10,
) -> float:
    """
    Per date: sort by predicted return, top decile mean realized minus bottom decile mean.
    Return average spread across dates with enough names.
    """
    spreads: list[float] = []
    for _, g in df.groupby(date_col, sort=False):
        if len(g) < n_buckets:
            continue
        gg = g[[pred_col, realized_col]].dropna()
        if len(gg) < n_buckets:
            continue
        if gg[pred_col].nunique(dropna=True) <= 1:
            continue
        gg = gg.sort_values(pred_col, kind="mergesort")
        n = len(gg)
        k = max(1, n // n_buckets)
        bottom = gg.iloc[:k][realized_col].mean()
        top = gg.iloc[-k:][realized_col].mean()
        spreads.append(float(top - bottom))
    if not spreads:
        return float("nan")
    return float(np.mean(spreads))


def evaluate_predictions(
    y_train: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    panel: Optional[pd.DataFrame] = None,
    *,
    date_col: str = "date",
    pred_col: str = "predicted_return",
    realized_col: str = "target_return",
) -> EvaluationReport:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    oos_r2 = _oos_r2(y_true, y_pred, y_train)
    if panel is not None and date_col in panel.columns:
        mean_ic = mean_spearman_ic_by_date(
            panel, date_col=date_col, pred_col=pred_col, realized_col=realized_col
        )
        spread = top_minus_bottom_decile_spread(
            panel, date_col=date_col, pred_col=pred_col, realized_col=realized_col
        )
    else:
        mean_ic = float("nan")
        spread = float("nan")
    return EvaluationReport(
        rmse=rmse,
        mae=mae,
        oos_r2=oos_r2,
        mean_rank_ic=mean_ic,
        top_bottom_spread=spread,
    )
