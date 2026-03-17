r"""
Baseline return prediction model (Phase 4 implementation).

This module trains a simple baseline model to generate *ML return views*.
These predictions are intended to be consumed later by a Black-Litterman
aggregator (NOT implemented in this phase).

## Input contract: Processed model dataset
`train_df` / `test_df` must contain:
- date
- asset_id
- selected feature columns (subset allowed; configurable)
- target_return

## Output contract: Predicted returns output

Expected as a pandas.DataFrame named `pred_df`.

### Required columns
- **date**: datetime64[ns]; prediction timestamp (aligned to feature timestamp)
- **asset_id**: str/int
- **predicted_return**: float; expected next-period return (model prediction)

### Optional columns
- **model_name**: str; identifier for model used
- **feature_set_name**: str; e.g. "right", "left", "all"

### Key invariants
- One row per (date, asset_id).
- `mu` must be aligned to the same universe definition used by Σ estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

from src.data.feature_engineering import get_available_features, validate_requested_features


@dataclass
class BaselineModelConfig:
    """
    Minimal configuration for the baseline model.

    ElasticNet is chosen as a safe default for correlated characteristics.
    """

    alpha: float = 1e-3
    l1_ratio: float = 0.1
    fit_intercept: bool = True
    max_iter: int = 10_000
    random_state: int = 42


class BaselineModel:
    """
    Simple baseline return-prediction model producing ML return views.

    API
    - fit(train_df, feature_cols, target_col="target_return")
    - predict(test_df) -> pred_df with [date, asset_id, predicted_return]
    - get_feature_columns()
    """

    def __init__(
        self,
        *,
        config: Optional[BaselineModelConfig] = None,
        model_name: str = "elasticnet",
        feature_set_name: Optional[str] = None,
        strict_features: bool = False,
    ) -> None:
        self.config = config or BaselineModelConfig()
        self.model_name = model_name
        self.feature_set_name = feature_set_name
        self.strict_features = strict_features

        self._feature_cols: List[str] = []
        self._target_col: str = "target_return"
        self._model: Optional[ElasticNet] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: Sequence[str],
        *,
        target_col: str = "target_return",
    ) -> "BaselineModel":
        """Fit the baseline model on a supervised dataset."""

        for c in ("date", "asset_id", target_col):
            if c not in train_df.columns:
                raise ValueError(f"train_df missing required column '{c}'.")

        if self.strict_features:
            validate_requested_features(train_df, feature_cols, key_cols=("date", "asset_id"))
            used_features = [c for c in feature_cols if c in train_df.columns]
        else:
            used_features = get_available_features(
                train_df,
                feature_cols,
                key_cols=("date", "asset_id"),
                warn_missing=True,
            )

        if not used_features:
            raise ValueError("No usable feature columns available to fit the model.")

        # Drop rows with missing target; do not impute target.
        df = train_df.dropna(subset=[target_col]).copy()
        if df.empty:
            raise ValueError("Training data is empty after dropping missing targets.")

        X = df[used_features].to_numpy(dtype=float, copy=False)
        y = df[target_col].to_numpy(dtype=float, copy=False)

        if not np.isfinite(X).all():
            raise ValueError("Non-finite values found in training features. Preprocess features first.")
        if not np.isfinite(y).all():
            raise ValueError("Non-finite values found in training target.")

        self._model = ElasticNet(
            alpha=self.config.alpha,
            l1_ratio=self.config.l1_ratio,
            fit_intercept=self.config.fit_intercept,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
        )
        self._model.fit(X, y)

        self._feature_cols = list(used_features)
        self._target_col = target_col
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Predict expected next-period returns for each (date, asset_id) row."""

        if self._model is None:
            raise RuntimeError("Model is not fit yet. Call fit() first.")

        for c in ("date", "asset_id"):
            if c not in test_df.columns:
                raise ValueError(f"test_df missing required column '{c}'.")

        missing = [c for c in self._feature_cols if c not in test_df.columns]
        if missing:
            raise ValueError(
                "test_df is missing feature column(s) used during training: "
                f"{missing}. Ensure consistent feature selection."
            )

        X = test_df[self._feature_cols].to_numpy(dtype=float, copy=False)
        if not np.isfinite(X).all():
            raise ValueError("Non-finite values found in test features. Preprocess features first.")

        preds = self._model.predict(X).astype(float)

        out = test_df[["date", "asset_id"]].copy()
        out["predicted_return"] = preds
        out["model_name"] = self.model_name
        if self.feature_set_name is not None:
            out["feature_set_name"] = self.feature_set_name
        return out

    def get_feature_columns(self) -> List[str]:
        """Return the feature columns used during fit."""

        return list(self._feature_cols)

