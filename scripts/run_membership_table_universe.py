"""
Standalone runner for MembershipTableUniverseProvider.

This script is intentionally modular: give it a membership table CSV plus a
start/end date, and it will materialize the point-in-time universe for each
monthly rebalance date.

Important:
- this expects a real historical membership table CSV
- yfinance does not provide a clean historical S&P 500 constituent history
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import generate_rebalance_dates
from src.backtest.universe import (
    MembershipTableUniverseProvider,
    build_universe_snapshot_panel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MembershipTableUniverseProvider on its own."
    )
    parser.add_argument(
        "--membership_csv",
        required=True,
        help="CSV with ticker,start_date,end_date columns.",
    )
    parser.add_argument("--start_date", required=True, help="Start date for inspection.")
    parser.add_argument("--end_date", required=True, help="End date for inspection.")
    parser.add_argument(
        "--rebalance_timing",
        choices=("month_end", "month_start"),
        default="month_end",
        help="Month-end is the recommended default for this repo.",
    )
    parser.add_argument("--ticker_col", default="ticker", help="Ticker column name.")
    parser.add_argument("--start_col", default="start_date", help="Membership start-date column.")
    parser.add_argument("--end_col", default="end_date", help="Membership end-date column.")
    parser.add_argument(
        "--out_csv",
        default=None,
        help="Optional path to save the monthly universe snapshot panel.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    provider = MembershipTableUniverseProvider.from_csv(
        args.membership_csv,
        ticker_col=args.ticker_col,
        start_col=args.start_col,
        end_col=args.end_col,
    )
    dates = generate_rebalance_dates(
        args.start_date,
        args.end_date,
        rebalance_timing=args.rebalance_timing,
    )
    panel = build_universe_snapshot_panel(provider, dates=dates)

    print("=== Membership Table Universe ===")
    print(f"Membership CSV: {args.membership_csv}")
    print(f"Window: {args.start_date} -> {args.end_date}")
    print(f"Rebalance timing: {args.rebalance_timing}")
    print(f"Rows: {len(panel)}")
    if not panel.empty:
        print(panel.head(12).to_string(index=False))

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(out_path, index=False)
        print(f"Saved monthly universe snapshots to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
