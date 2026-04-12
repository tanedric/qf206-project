"""
Inspect S&P 500 universe snapshots across monthly rebalance dates.

This is useful for proving whether a provider is truly point-in-time or just
returning the same current constituent list on every date.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import generate_rebalance_dates
from src.backtest.universe import (
    MembershipTableUniverseProvider,
    WikipediaCurrentSP500UniverseProvider,
    build_universe_snapshot_panel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect monthly S&P 500 universe snapshots.")
    parser.add_argument("--start_date", required=True, help="Inspection start date.")
    parser.add_argument("--end_date", required=True, help="Inspection end date.")
    parser.add_argument(
        "--rebalance_timing",
        choices=("month_end", "month_start"),
        default="month_end",
        help="Month-end is the recommended default for this repo.",
    )
    parser.add_argument(
        "--universe_source",
        choices=("membership_csv", "current_sp500_wikipedia"),
        required=True,
        help="Universe source to inspect.",
    )
    parser.add_argument(
        "--membership_csv",
        default=None,
        help="Required when --universe_source=membership_csv.",
    )
    parser.add_argument(
        "--out_csv",
        default=None,
        help="Optional path to save the snapshot panel.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.universe_source == "membership_csv":
        if not args.membership_csv:
            raise ValueError("--membership_csv is required when --universe_source=membership_csv.")
        provider = MembershipTableUniverseProvider.from_csv(args.membership_csv)
    else:
        provider = WikipediaCurrentSP500UniverseProvider()

    rebalance_dates = generate_rebalance_dates(
        args.start_date,
        args.end_date,
        rebalance_timing=args.rebalance_timing,
    )
    panel = build_universe_snapshot_panel(provider, dates=rebalance_dates)

    print("=== S&P 500 Universe Inspection ===")
    print(f"Source: {args.universe_source}")
    if args.membership_csv:
        print(f"Membership CSV: {args.membership_csv}")
    print(f"Dates inspected: {len(panel)}")
    print(panel.head(12).to_string(index=False))

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(out_path, index=False)
        print(f"Saved universe snapshots to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
