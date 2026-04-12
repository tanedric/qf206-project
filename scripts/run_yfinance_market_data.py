"""
Yahoo Finance market-data helper.

Modes:
- history: download historical OHLCV bars
- stream: listen to live quote updates
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.yfinance_adapter import LiveQuoteRecorder, download_history, stream_live_quotes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or stream Yahoo Finance market data.")
    parser.add_argument(
        "--mode",
        choices=("history", "stream"),
        required=True,
        help="history: download OHLCV bars | stream: listen for live quote updates",
    )
    parser.add_argument(
        "--tickers",
        required=True,
        help="Comma-separated or space-separated ticker list, for example 'AAPL,MSFT,NVDA'.",
    )
    parser.add_argument("--start", default=None, help="History mode start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="History mode end date (YYYY-MM-DD).")
    parser.add_argument(
        "--period",
        default="1mo",
        help="History mode period when start/end are not supplied. Example: 5d, 1mo, 1y.",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="History mode interval. Yahoo supports intraday intervals like 1m, 5m, 1h.",
    )
    parser.add_argument(
        "--prepost",
        action="store_true",
        help="Include pre-market and after-hours data in history mode where available.",
    )
    parser.add_argument(
        "--out_csv",
        default=None,
        help="Optional CSV output path. In stream mode, quotes are appended as they arrive.",
    )
    parser.add_argument(
        "--max_messages",
        type=int,
        default=None,
        help="Stream mode only. Stop after this many live quote messages.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "history":
        history = download_history(
            args.tickers,
            start=args.start,
            end=args.end,
            period=args.period,
            interval=args.interval,
            prepost=args.prepost,
        )
        print("=== Yahoo History Download ===")
        print(f"Tickers: {args.tickers}")
        print(f"Rows: {len(history)}")
        if not history.empty:
            print(history.head(10).to_string(index=False))
        if args.out_csv:
            out_path = Path(args.out_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            history.to_csv(out_path, index=False)
            print(f"Saved history to: {out_path}")
        return 0

    print("=== Yahoo Live Quote Stream ===")
    print(f"Tickers: {args.tickers}")
    print("Press Ctrl+C to stop.")
    recorder = LiveQuoteRecorder(
        max_messages=args.max_messages,
        out_csv=args.out_csv,
        verbose=not args.quiet,
    )
    stream_live_quotes(
        args.tickers,
        recorder=recorder,
        verbose=not args.quiet,
    )
    if args.out_csv:
        print(f"Saved live quotes to: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
