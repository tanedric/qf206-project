"""
Build and query an approximate S&P 500 membership table from Wikipedia.

Modes:
- build: scrape Wikipedia and create an approximate membership CSV
- query: read the CSV and return the S&P 500 universe for one YYYYMMDD
- snapshots: read the CSV and emit month-by-month universe snapshots

Important limitations:
- Wikipedia's "Selected changes" table is not an official complete historical source
- this output is approximate and should not be treated as research-grade
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import generate_rebalance_dates
from src.backtest.universe import (
    MembershipTableUniverseProvider,
    build_universe_snapshot_panel,
    get_universe_from_membership_csv,
)

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _normalize_date(value: pd.Timestamp | str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce", utc=False)
    if pd.isna(parsed):
        raise ValueError(f"Failed to parse date value: {value}")
    return pd.Timestamp(parsed).normalize()


def _standardize_ticker(value: object) -> str:
    return str(value).strip().upper().replace(".", "-")


def _table_headers(table) -> List[str]:
    headers: List[str] = []
    for row in table.find_all("tr"):
        ths = row.find_all("th")
        if not ths:
            continue
        row_headers = [" ".join(cell.stripped_strings) for cell in ths]
        headers.extend([header for header in row_headers if header])
        if len(headers) >= 6:
            break
    return headers


def _find_current_constituents_table(soup: BeautifulSoup):
    for table in soup.find_all("table", class_="wikitable"):
        headers = set(_table_headers(table))
        if {"Symbol", "Security", "GICS Sector"}.issubset(headers):
            return table
    raise RuntimeError("Failed to locate the current S&P 500 constituents table on Wikipedia.")


def _find_selected_changes_table(soup: BeautifulSoup):
    for table in soup.find_all("table", class_="wikitable"):
        headers = set(_table_headers(table))
        if {"Effective Date", "Added", "Removed", "Reason"}.issubset(headers):
            return table
    raise RuntimeError("Failed to locate the selected changes table on Wikipedia.")


def _scrape_current_constituents(table) -> pd.DataFrame:
    rows = []
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        ticker = _standardize_ticker(" ".join(cells[0].stripped_strings))
        security = " ".join(cells[1].stripped_strings)
        if ticker:
            rows.append({"ticker": ticker, "security": security})
    if not rows:
        raise RuntimeError("Wikipedia current constituents table produced no ticker rows.")
    return pd.DataFrame(rows).drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def _scrape_selected_changes(table) -> pd.DataFrame:
    rows = []
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 6:
            continue
        effective_date = " ".join(cells[0].stripped_strings)
        added_ticker = _standardize_ticker(" ".join(cells[1].stripped_strings))
        added_security = " ".join(cells[2].stripped_strings)
        removed_ticker = _standardize_ticker(" ".join(cells[3].stripped_strings))
        removed_security = " ".join(cells[4].stripped_strings)
        reason = " ".join(cells[5].stripped_strings)
        parsed_date = pd.to_datetime(effective_date, errors="coerce", utc=False)
        if pd.isna(parsed_date):
            continue
        rows.append(
            {
                "effective_date": pd.Timestamp(parsed_date).normalize(),
                "added_ticker": added_ticker if added_ticker else pd.NA,
                "added_security": added_security if added_security else pd.NA,
                "removed_ticker": removed_ticker if removed_ticker else pd.NA,
                "removed_security": removed_security if removed_security else pd.NA,
                "reason": reason if reason else pd.NA,
            }
        )
    if not rows:
        raise RuntimeError("Wikipedia selected changes table produced no change rows.")
    return pd.DataFrame(rows)


@dataclass
class WikipediaScrapeResult:
    current_constituents: pd.DataFrame
    selected_changes: pd.DataFrame
    fetched_at: pd.Timestamp
    source_url: str


def scrape_wikipedia_sp500_page(url: str = WIKIPEDIA_SP500_URL) -> WikipediaScrapeResult:
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (compatible; sp500-membership-builder/1.0)"},
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    current_table = _find_current_constituents_table(soup)
    changes_table = _find_selected_changes_table(soup)
    return WikipediaScrapeResult(
        current_constituents=_scrape_current_constituents(current_table),
        selected_changes=_scrape_selected_changes(changes_table),
        fetched_at=pd.Timestamp.utcnow().tz_convert(None).normalize(),
        source_url=url,
    )


def build_wikipedia_membership_table(
    *,
    floor_start_date: str = "1957-03-04",
    url: str = WIKIPEDIA_SP500_URL,
) -> pd.DataFrame:
    """
    Build an approximate membership table from the Wikipedia current constituents
    table plus the "Selected changes" history table.

    Approximation logic:
    - start from the current constituent set
    - walk the selected changes backward in time
    - convert effective-date additions/removals into membership intervals
    """

    scraped = scrape_wikipedia_sp500_page(url=url)
    floor_start = _normalize_date(floor_start_date)
    changes = (
        scraped.selected_changes.sort_values(
            ["effective_date", "added_ticker", "removed_ticker"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    active_end_by_ticker: dict[str, pd.Timestamp | pd.NaT] = {
        ticker: pd.NaT for ticker in scraped.current_constituents["ticker"].tolist()
    }
    records: List[dict] = []

    for _, row in changes.iterrows():
        effective_date = _normalize_date(row["effective_date"])
        previous_member_end = effective_date - pd.Timedelta(days=1)

        added_ticker = row["added_ticker"]
        if pd.notna(added_ticker):
            added_ticker = _standardize_ticker(added_ticker)
            current_end = active_end_by_ticker.pop(added_ticker, None)
            records.append(
                {
                    "ticker": added_ticker,
                    "start_date": effective_date,
                    "end_date": current_end,
                    "source": "wikipedia_approx",
                    "note": "Derived from current constituents plus selected-changes reverse walk.",
                }
            )

        removed_ticker = row["removed_ticker"]
        if pd.notna(removed_ticker):
            removed_ticker = _standardize_ticker(removed_ticker)
            active_end_by_ticker[removed_ticker] = previous_member_end

    for ticker, current_end in sorted(active_end_by_ticker.items()):
        records.append(
            {
                "ticker": ticker,
                "start_date": floor_start,
                "end_date": current_end,
                "source": "wikipedia_approx",
                "note": (
                    "Open interval before the oldest selected Wikipedia change. "
                    "This is approximate and may start too early."
                ),
            }
        )

    membership = pd.DataFrame(records)
    membership["ticker"] = membership["ticker"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    membership["start_date"] = pd.to_datetime(membership["start_date"], errors="coerce").dt.normalize()
    membership["end_date"] = pd.to_datetime(membership["end_date"], errors="coerce").dt.normalize()
    membership = membership.dropna(subset=["ticker", "start_date"]).sort_values(
        ["ticker", "start_date", "end_date"], kind="mergesort"
    )
    membership = membership.reset_index(drop=True)
    return membership


def query_membership_csv(
    membership_csv: str | Path,
    *,
    as_of_date: str,
) -> List[str]:
    return get_universe_from_membership_csv(membership_csv, as_of_date=as_of_date)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and query an approximate Wikipedia-based S&P 500 membership CSV."
    )
    parser.add_argument(
        "--mode",
        choices=("build", "query", "snapshots"),
        required=True,
        help="build: scrape and save CSV | query: get universe for one date | snapshots: month-by-month output from CSV",
    )
    parser.add_argument(
        "--membership_csv",
        default="data/raw/sp500_membership_wikipedia_approx.csv",
        help="Membership CSV path to write or read.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Single query date in YYYYMMDD or YYYY-MM-DD format for --mode query.",
    )
    parser.add_argument("--start_date", default=None, help="Start date for --mode snapshots.")
    parser.add_argument("--end_date", default=None, help="End date for --mode snapshots.")
    parser.add_argument(
        "--rebalance_timing",
        choices=("month_end", "month_start"),
        default="month_end",
        help="Monthly snapshot timing for --mode snapshots.",
    )
    parser.add_argument(
        "--floor_start_date",
        default="1957-03-04",
        help="Default start date assigned to names that predate the oldest selected Wikipedia change.",
    )
    parser.add_argument(
        "--out_csv",
        default=None,
        help="Optional output path for snapshot mode. Build mode writes to --membership_csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    membership_path = Path(args.membership_csv)

    if args.mode == "build":
        membership = build_wikipedia_membership_table(
            floor_start_date=args.floor_start_date,
        )
        membership_path.parent.mkdir(parents=True, exist_ok=True)
        membership.to_csv(membership_path, index=False)
        print("=== Wikipedia Approximate S&P 500 Membership Build ===")
        print(f"Rows written: {len(membership)}")
        print(f"Output CSV: {membership_path}")
        print("Warning: This CSV is approximate because Wikipedia only provides selected changes.")
        print(membership.head(12).to_string(index=False))
        return 0

    if args.mode == "query":
        if not args.date:
            raise ValueError("--date is required for --mode query.")
        universe = query_membership_csv(membership_path, as_of_date=args.date)
        print("=== S&P 500 Universe Query ===")
        print(f"Membership CSV: {membership_path}")
        print(f"As-of date: {args.date}")
        print(f"Universe size: {len(universe)}")
        print("|".join(universe))
        return 0

    if not args.start_date or not args.end_date:
        raise ValueError("--start_date and --end_date are required for --mode snapshots.")

    provider = MembershipTableUniverseProvider.from_csv(membership_path)
    dates = generate_rebalance_dates(
        args.start_date,
        args.end_date,
        rebalance_timing=args.rebalance_timing,
    )
    panel = build_universe_snapshot_panel(provider, dates=dates)
    print("=== S&P 500 Monthly Snapshots ===")
    print(f"Membership CSV: {membership_path}")
    print(f"Window: {args.start_date} -> {args.end_date}")
    print(f"Rows: {len(panel)}")
    if not panel.empty:
        print(panel.head(12).to_string(index=False))
    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(out_path, index=False)
        print(f"Saved snapshots to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
