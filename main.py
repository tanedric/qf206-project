"""
Phase 1 smoke test entrypoint.

This script intentionally does NOT run any real pipeline logic yet.
It only verifies that:
- The project structure is importable.
- All placeholder modules can be imported.
- Settings resolve expected paths/parameters.
"""

from __future__ import annotations

from config.settings import get_parameters, get_paths

from src.data.load_data import load_raw_firm_characteristics
from src.data.preprocess import preprocess_raw_inputs
from src.data.feature_engineering import build_features
from src.models.baseline_model import BaselineModel
from src.portfolio.risk_model import estimate_covariance
from src.portfolio.optimizers import solve_gmvp, solve_msrp
from src.backtest.engine import run_backtest
from src.backtest.metrics import compute_metrics
from src.interfaces.sentiment_interface import SentimentProvider


def main() -> None:
    paths = get_paths()
    params = get_parameters()

    print("Phase 1: project scaffold OK")
    print()
    print("Resolved paths:")
    print(f"- project_root: {paths.project_root}")
    print(f"- data_raw_dir: {paths.data_raw_dir}")
    print(f"- data_processed_dir: {paths.data_processed_dir}")
    print(f"- artifacts_dir: {paths.artifacts_dir}")
    print()
    print("Default parameters:")
    print(f"- n_features: {params.n_features}")
    print(f"- prediction_horizon_periods: {params.prediction_horizon_periods}")
    print(f"- random_seed: {params.random_seed}")
    print()
    print("Pipeline stages (placeholders):")
    stages = [
        ("load_raw_firm_characteristics", load_raw_firm_characteristics),
        ("preprocess_raw_inputs", preprocess_raw_inputs),
        ("build_features", build_features),
        ("BaselineModel.fit / .predict", BaselineModel),
        ("estimate_covariance", estimate_covariance),
        ("solve_gmvp", solve_gmvp),
        ("solve_msrp", solve_msrp),
        ("run_backtest", run_backtest),
        ("compute_metrics", compute_metrics),
        ("SentimentProvider (optional)", SentimentProvider),
    ]
    for name, obj in stages:
        print(f"- {name}: import OK ({obj.__module__})")

    print()
    print("Note: All functions/classes intentionally raise NotImplementedError in Phase 1.")


if __name__ == "__main__":
    main()

