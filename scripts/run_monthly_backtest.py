"""
Standalone monthly backtest runner.

This script intentionally stays separate from model training. It expects a
predictions CSV that has already been generated elsewhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import BacktestConfig, run_monthly_backtest
from src.backtest.metrics import compute_metrics
from src.backtest.providers import FramePredictionProvider, YFinanceAdjustedCloseReturnProvider
from src.backtest.universe import (
    MembershipTableUniverseProvider,
    StaticUniverseProvider,
    WikipediaCurrentSP500UniverseProvider,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the modular monthly backtest.")
    parser.add_argument("--start_date", required=True, help="Backtest window start date.")
    parser.add_argument("--end_date", required=True, help="Backtest window end date.")
    parser.add_argument(
        "--predictions_csv",
        required=True,
        help="CSV with date, asset_id, predicted_return.",
    )
    parser.add_argument("--date_col", default="date", help="Prediction date column.")
    parser.add_argument("--asset_col", default="asset_id", help="Prediction asset column.")
    parser.add_argument("--pred_col", default="predicted_return", help="Prediction value column.")
    parser.add_argument(
        "--rebalance_timing",
        choices=("month_end", "month_start"),
        default="month_end",
        help="Use month_end by default to match the repo's monthly prediction pipeline.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=50,
        help="Hold the top N predicted names each rebalance.",
    )
    parser.add_argument(
        "--weighting_scheme",
        choices=("equal_weight_top_n", "proportional_long_only"),
        default="equal_weight_top_n",
        help="Portfolio construction rule.",
    )
    parser.add_argument(
        "--transaction_cost_bps",
        type=float,
        default=10.0,
        help="One-way transaction cost assumption in basis points per rebalance.",
    )
    parser.add_argument(
        "--initial_capital",
        type=float,
        default=1.0,
        help="Initial portfolio value.",
    )
    parser.add_argument(
        "--membership_csv",
        default=None,
        help="Optional historical S&P 500 membership CSV with ticker,start_date,end_date.",
    )
    parser.add_argument(
        "--universe_source",
        choices=("membership_csv", "current_sp500_wikipedia", "static_from_predictions"),
        default="static_from_predictions",
        help="Universe source. Use membership_csv for proper point-in-time S&P 500 backtests.",
    )
    parser.add_argument(
        "--output_dir",
        default="artifacts/backtests",
        help="Directory for portfolio, holdings, universe, and metrics outputs.",
    )
    return parser.parse_args()


def _build_universe_provider(args: argparse.Namespace, predictions_df: pd.DataFrame):
    if args.universe_source == "membership_csv":
        if not args.membership_csv:
            raise ValueError("--membership_csv is required when --universe_source=membership_csv.")
        return MembershipTableUniverseProvider.from_csv(args.membership_csv)
    if args.universe_source == "current_sp500_wikipedia":
        return WikipediaCurrentSP500UniverseProvider()
    if args.asset_col not in predictions_df.columns:
        raise ValueError(f"Predictions CSV missing asset column: {args.asset_col}")
    tickers = predictions_df[args.asset_col].dropna().astype(str).tolist()
    return StaticUniverseProvider(tickers=tickers)


def main() -> int:
    args = parse_args()
    predictions_df = pd.read_csv(args.predictions_csv)
    prediction_provider = FramePredictionProvider(
        predictions_df,
        date_col=args.date_col,
        asset_col=args.asset_col,
        pred_col=args.pred_col,
    )
    universe_provider = _build_universe_provider(args, predictions_df)
    return_provider = YFinanceAdjustedCloseReturnProvider()

    config = BacktestConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        rebalance_timing=args.rebalance_timing,
        weighting_scheme=args.weighting_scheme,
        top_n=args.top_n,
        initial_capital=args.initial_capital,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    result = run_monthly_backtest(
        config=config,
        universe_provider=universe_provider,
        prediction_provider=prediction_provider,
        return_provider=return_provider,
    )
    metrics = compute_metrics(result.portfolio_df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    portfolio_path = output_dir / "portfolio.csv"
    holdings_path = output_dir / "holdings.csv"
    universe_path = output_dir / "universes.csv"
    metrics_path = output_dir / "metrics.json"
    result.portfolio_df.to_csv(portfolio_path, index=False)
    result.holdings_df.to_csv(holdings_path, index=False)
    result.universe_df.to_csv(universe_path, index=False)
    metrics_path.write_text(json.dumps(metrics.to_dict(), indent=2), encoding="utf-8")

    print("=== Monthly Backtest ===")
    print(f"Backtest window: {args.start_date} -> {args.end_date}")
    print(f"Rebalance timing: {args.rebalance_timing}")
    print(f"Universe source: {args.universe_source}")
    if args.membership_csv:
        print(f"Membership CSV: {args.membership_csv}")
    print(f"Predictions CSV: {args.predictions_csv}")
    print(f"Portfolio rows: {len(result.portfolio_df)}")
    print(f"Holdings rows: {len(result.holdings_df)}")
    print(f"Total return: {metrics.total_return:.6f}")
    print(f"Annualized return: {metrics.annualized_return:.6f}")
    print(f"Annualized volatility: {metrics.annualized_volatility:.6f}")
    print(f"Sharpe ratio: {metrics.sharpe_ratio:.6f}")
    print(f"Max drawdown: {metrics.max_drawdown:.6f}")
    print(f"Saved portfolio to: {portfolio_path}")
    print(f"Saved holdings to: {holdings_path}")
    print(f"Saved universes to: {universe_path}")
    print(f"Saved metrics to: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
