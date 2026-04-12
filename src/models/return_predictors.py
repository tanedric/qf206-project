"""
Sklearn-based return predictors (ElasticNet, RandomForest, optional XGBoost).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

try:
    from xgboost import XGBRegressor

    _HAS_XGB = True
except ImportError:
    XGBRegressor = None  # type: ignore[misc, assignment]
    _HAS_XGB = False


def xgboost_available() -> bool:
    return _HAS_XGB


class SklearnReturnPredictor:
    """Unified fit/predict API for return-view generation."""

    def __init__(
        self,
        name: str,
        estimator,
        *,
        feature_set_name: Optional[str] = None,
    ) -> None:
        self.name = name
        self.estimator = estimator
        self.feature_set_name = feature_set_name
        self._feature_cols: List[str] = []
        self._target_col = "target_return"

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: Sequence[str],
        *,
        target_col: str = "target_return",
    ) -> "SklearnReturnPredictor":
        for c in ("date", "asset_id", target_col):
            if c not in train_df.columns:
                raise ValueError(f"train_df missing '{c}'")
        missing = [c for c in feature_cols if c not in train_df.columns]
        if missing:
            raise ValueError(f"train_df missing feature column(s): {missing}")

        df = train_df.dropna(subset=[target_col]).copy()
        X = df[list(feature_cols)].to_numpy(dtype=float, copy=False)
        y = df[target_col].to_numpy(dtype=float, copy=False)
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Non-finite values in train features or target.")

        self.estimator.fit(X, y)
        self._feature_cols = list(feature_cols)
        self._target_col = target_col
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._feature_cols:
            raise RuntimeError("Model not fit.")
        missing = [c for c in self._feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"df missing feature column(s): {missing}")
        X = df[self._feature_cols].to_numpy(dtype=float, copy=False)
        if not np.isfinite(X).all():
            raise ValueError("Non-finite values in prediction features.")
        preds = self.estimator.predict(X).astype(float)
        out = df[["date", "asset_id"]].copy()
        out["predicted_return"] = preds
        out["model_name"] = self.name
        if self.feature_set_name is not None:
            out["feature_set_name"] = self.feature_set_name
        return out

    def get_feature_columns(self) -> List[str]:
        return list(self._feature_cols)


def build_candidate_models(
    *,
    random_state: int = 42,
    feature_set_name: Optional[str] = None,
) -> List[SklearnReturnPredictor]:
    models: List[SklearnReturnPredictor] = [
        SklearnReturnPredictor(
            "elasticnet",
            ElasticNet(
                alpha=1e-3,
                l1_ratio=0.15,
                max_iter=20_000,
                random_state=random_state,
            ),
            feature_set_name=feature_set_name,
        ),
        SklearnReturnPredictor(
            "random_forest",
            RandomForestRegressor(
                n_estimators=120,
                max_depth=8,
                min_samples_leaf=5,
                min_samples_split=10,
                max_features=0.5,
                random_state=random_state,
                n_jobs=1,
            ),
            feature_set_name=feature_set_name,
        ),
    ]
    if _HAS_XGB:
        models.append(
            SklearnReturnPredictor(
                "xgboost",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    random_state=random_state,
                    n_jobs=-1,
                ),
                feature_set_name=feature_set_name,
            )
        )
    return models
