"""
Standalone Wikipedia-backed S&P 500 universe helper.

This module keeps the workflow modular in one place:
1. scrape Wikipedia's current constituents + selected changes tables
2. build or refresh a local approximate membership CSV only when needed
3. query the S&P 500 universe for one date or a monthly date range

Important limitation:
- This is an approximate historical membership source.
- It is suitable for prototyping, not as an authoritative research dataset.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup


WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DEFAULT_HISTORY_FLOOR = "1957-03-04"
DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "sp500_membership_wikipedia_approx.csv"
)
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0 Safari/537.36"
    )
}


@dataclass(frozen=True)
class WikipediaSnapshot:
    current_constituents: pd.DataFrame
    selected_changes: pd.DataFrame
    scraped_at: pd.Timestamp
    latest_effective_date_seen: pd.Timestamp | pd.NaT


def _normalize_ticker(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE"}:
        return None
    return text.replace(".", "-")


def _parse_date(value: object) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"Could not parse date: {value!r}")
    return pd.Timestamp(timestamp).normalize()


def _coerce_optional_date(value: object) -> pd.Timestamp | pd.NaT:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return pd.NaT
    return pd.Timestamp(timestamp).normalize()


def _find_table_by_headers(soup: BeautifulSoup, expected_headers: Iterable[str]) -> object:
    expected = {header.strip().lower() for header in expected_headers}
    for table in soup.find_all("table"):
        headers = {
            cell.get_text(" ", strip=True).strip().lower()
            for cell in table.find_all(["th", "td"])
        }
        if expected.issubset(headers):
            return table
    raise ValueError(f"Could not find Wikipedia table with headers: {sorted(expected)}")


def _table_to_frame(table: object) -> pd.DataFrame:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        rows.append([cell.get_text(" ", strip=True).strip() for cell in cells])
    if not rows:
        raise ValueError("Failed to parse HTML table into rows.")

    header = rows[0]
    data_rows = rows[1:]
    normalized_rows = []
    for row in data_rows:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        normalized_rows.append(row[: len(header)])
    return pd.DataFrame(normalized_rows, columns=header)


def _get_universe_from_frame(
    membership: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    *,
    gics_filter: str | None = None,
) -> list[str]:
    target_date = _parse_date(as_of_date)
    start_ok = membership["start_date"].le(target_date)
    if "end_date" in membership.columns:
        end_ok = membership["end_date"].isna() | membership["end_date"].ge(target_date)
    else:
        end_ok = True
    filtered = membership.loc[start_ok & end_ok].copy()
    if gics_filter is not None:
        gics_filter_lower = gics_filter.strip().lower()
        sector_mask = pd.Series(False, index=filtered.index)
        for col in ("gics_sub_industry", "gics_sector"):
            if col in filtered.columns:
                sector_mask |= filtered[col].astype(str).str.lower().str.contains(
                    gics_filter_lower, na=False
                )
        filtered = filtered.loc[sector_mask]
    universe = filtered["ticker"].dropna().astype(str).sort_values()
    return universe.tolist()


def _scrape_wikipedia_snapshot(timeout_seconds: int = 30) -> WikipediaSnapshot:
    response = requests.get(
        WIKIPEDIA_URL,
        headers=REQUEST_HEADERS,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    current_table = _find_table_by_headers(soup, ["Symbol", "Security", "GICS Sector"])
    changes_table = _find_table_by_headers(
        soup,
        ["Effective Date", "Added", "Removed", "Reason"],
    )

    current = _table_to_frame(current_table)
    changes = _table_to_frame(changes_table)

    current.columns = [str(col).strip() for col in current.columns]
    current = current.rename(columns={"Symbol": "ticker", "Security": "security"})
    current["ticker"] = current["ticker"].map(_normalize_ticker)
    current = current[current["ticker"].notna()].copy()
    current = current.drop_duplicates(subset=["ticker"]).sort_values("ticker", kind="mergesort")

    changes.columns = [str(col).strip() for col in changes.columns]
    rename_map = {}
    for col in changes.columns:
        lowered = col.lower()
        if "effective date" in lowered:
            rename_map[col] = "effective_date"
        elif lowered == "ticker":
            rename_map[col] = "added_ticker"
        elif "ticker.1" in lowered:
            rename_map[col] = "removed_ticker"
        elif lowered == "added ticker":
            rename_map[col] = "added_ticker"
        elif lowered == "removed ticker":
            rename_map[col] = "removed_ticker"
    changes = changes.rename(columns=rename_map)

    if "added_ticker" not in changes.columns or "removed_ticker" not in changes.columns:
        raw_cols = list(changes.columns)
        if len(raw_cols) >= 4:
            changes = changes.rename(
                columns={
                    raw_cols[0]: "effective_date",
                    raw_cols[1]: "added_ticker",
                    raw_cols[3]: "removed_ticker",
                }
            )

    required = {"effective_date", "added_ticker", "removed_ticker"}
    missing = required.difference(changes.columns)
    if missing:
        raise ValueError(f"Wikipedia selected-changes table is missing columns: {sorted(missing)}")

    changes["effective_date"] = pd.to_datetime(
        changes["effective_date"],
        errors="coerce",
        format="mixed",
    ).dt.normalize()
    changes["added_ticker"] = changes["added_ticker"].map(_normalize_ticker)
    changes["removed_ticker"] = changes["removed_ticker"].map(_normalize_ticker)
    changes = changes[changes["effective_date"].notna()].copy()
    changes = changes.sort_values(
        ["effective_date", "added_ticker", "removed_ticker"],
        kind="mergesort",
    ).reset_index(drop=True)

    latest_effective_date_seen = changes["effective_date"].max() if not changes.empty else pd.NaT
    return WikipediaSnapshot(
        current_constituents=current.reset_index(drop=True),
        selected_changes=changes,
        scraped_at=pd.Timestamp.utcnow().tz_localize(None).normalize(),
        latest_effective_date_seen=latest_effective_date_seen,
    )


def build_membership_table(
    snapshot: WikipediaSnapshot,
    *,
    history_floor: str | pd.Timestamp = DEFAULT_HISTORY_FLOOR,
) -> pd.DataFrame:
    floor_date = _parse_date(history_floor)
    current_tickers = sorted(set(snapshot.current_constituents["ticker"]))
    state = set(current_tickers)
    active_interval_end = {ticker: pd.NaT for ticker in state}
    intervals: list[dict[str, object]] = []

    descending_changes = snapshot.selected_changes.sort_values(
        ["effective_date", "added_ticker", "removed_ticker"],
        ascending=[False, True, True],
        kind="mergesort",
    )

    for row in descending_changes.itertuples(index=False):
        effective_date = pd.Timestamp(row.effective_date).normalize()
        prior_end_date = effective_date - pd.Timedelta(days=1)
        added_ticker = _normalize_ticker(getattr(row, "added_ticker", None))
        removed_ticker = _normalize_ticker(getattr(row, "removed_ticker", None))

        if added_ticker:
            end_date = active_interval_end.pop(added_ticker, pd.NaT)
            if added_ticker in state:
                intervals.append(
                    {
                        "ticker": added_ticker,
                        "start_date": effective_date,
                        "end_date": end_date,
                        "source": "wikipedia_approx",
                        "note": "Derived from current constituents plus selected-changes reverse walk.",
                    }
                )
                state.remove(added_ticker)

        if removed_ticker and removed_ticker not in state:
            state.add(removed_ticker)
            active_interval_end[removed_ticker] = prior_end_date

    for ticker in sorted(state):
        intervals.append(
            {
                "ticker": ticker,
                "start_date": floor_date,
                "end_date": active_interval_end.get(ticker, pd.NaT),
                "source": "wikipedia_approx",
                "note": (
                    "Open interval before the oldest selected Wikipedia change. "
                    "This is approximate and may start too early."
                ),
            }
        )

    membership = pd.DataFrame(intervals)
    membership["ticker"] = membership["ticker"].map(_normalize_ticker)
    membership["start_date"] = pd.to_datetime(membership["start_date"], errors="coerce").dt.normalize()
    membership["end_date"] = pd.to_datetime(membership["end_date"], errors="coerce").dt.normalize()
    membership["scraped_at"] = snapshot.scraped_at
    membership["latest_effective_date_seen"] = snapshot.latest_effective_date_seen

    # Join GICS sector columns from current constituents onto all membership rows.
    # Sector classification is applied to all tickers (current and historical) using
    # the latest available snapshot — the best approximation we have from Wikipedia.
    sector_cols = [c for c in snapshot.current_constituents.columns
                   if "gics" in c.lower() or "sector" in c.lower() or "industry" in c.lower()]
    if sector_cols:
        sector_map = snapshot.current_constituents[["ticker"] + sector_cols].copy()
        rename = {}
        for col in sector_cols:
            low = col.lower()
            if "sub" in low:
                rename[col] = "gics_sub_industry"
            elif "sector" in low or "gics" in low:
                rename[col] = "gics_sector"
        sector_map = sector_map.rename(columns=rename).drop_duplicates(subset=["ticker"])
        membership = membership.merge(sector_map, on="ticker", how="left")

    membership = membership.dropna(subset=["ticker", "start_date"]).drop_duplicates(
        subset=["ticker", "start_date", "end_date"],
        keep="last",
    )
    return membership.sort_values(["ticker", "start_date", "end_date"], kind="mergesort").reset_index(drop=True)


def load_membership_csv(csv_path: str | Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Membership CSV not found: {path}")
    frame = pd.read_csv(path)
    for col in ("start_date", "end_date", "scraped_at", "latest_effective_date_seen"):
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce").dt.normalize()
    if "ticker" in frame.columns:
        frame["ticker"] = frame["ticker"].map(_normalize_ticker)
    return frame


def _latest_boundary_in_membership(frame: pd.DataFrame) -> pd.Timestamp | pd.NaT:
    candidates = []
    if "latest_effective_date_seen" in frame.columns:
        latest_seen = pd.to_datetime(frame["latest_effective_date_seen"], errors="coerce").dropna()
        if not latest_seen.empty:
            candidates.append(latest_seen.max())
    for col in ("start_date", "end_date"):
        if col in frame.columns:
            values = pd.to_datetime(frame[col], errors="coerce").dropna()
            if not values.empty:
                candidates.append(values.max())
    if not candidates:
        return pd.NaT
    return pd.Timestamp(max(candidates)).normalize()


def get_sp500_universe_for_date(
    as_of_date: str | pd.Timestamp,
    *,
    csv_path: str | Path = DEFAULT_CSV_PATH,
    auto_update: bool = False,
    gics_filter: str | None = None,
) -> list[str]:
    membership = ensure_membership_csv(csv_path=csv_path, check_for_updates=auto_update)
    return _get_universe_from_frame(membership, as_of_date, gics_filter=gics_filter)


def get_sp500_universe_between_dates(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    *,
    csv_path: str | Path = DEFAULT_CSV_PATH,
    rebalance_timing: str = "month_end",
    auto_update: bool = False,
) -> pd.DataFrame:
    start_ts = _parse_date(start_date)
    end_ts = _parse_date(end_date)
    if end_ts < start_ts:
        raise ValueError("end_date must be on or after start_date.")

    timing = rebalance_timing.strip().lower()
    if timing not in {"month_end", "month_start"}:
        raise ValueError("rebalance_timing must be either 'month_end' or 'month_start'.")

    membership = ensure_membership_csv(csv_path=csv_path, check_for_updates=auto_update)
    freq = "ME" if timing == "month_end" else "MS"
    dates = pd.date_range(start=start_ts, end=end_ts, freq=freq)
    rows: list[dict[str, object]] = []
    for universe_date in dates:
        for ticker in _get_universe_from_frame(membership, universe_date):
            rows.append({"date": universe_date.normalize(), "ticker": ticker})
    return pd.DataFrame(rows, columns=["date", "ticker"])


def update_membership_csv_if_needed(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    *,
    history_floor: str | pd.Timestamp = DEFAULT_HISTORY_FLOOR,
    force_refresh: bool = False,
    timeout_seconds: int = 30,
) -> pd.DataFrame:
    path = Path(csv_path)
    existing = load_membership_csv(path) if path.exists() else None

    if existing is not None and not force_refresh and "scraped_at" in existing.columns:
        scraped_today = pd.to_datetime(existing["scraped_at"], errors="coerce").dropna()
        if not scraped_today.empty and scraped_today.max().normalize() >= pd.Timestamp.today().normalize():
            return existing

    snapshot = _scrape_wikipedia_snapshot(timeout_seconds=timeout_seconds)
    if existing is not None and not force_refresh:
        existing_latest = _latest_boundary_in_membership(existing)
        current_active = set(_get_universe_from_frame(existing, pd.Timestamp.today()))
        scraped_current = set(snapshot.current_constituents["ticker"])
        if (
            (pd.isna(snapshot.latest_effective_date_seen) or existing_latest >= snapshot.latest_effective_date_seen)
            and current_active == scraped_current
        ):
            return existing

    updated = build_membership_table(snapshot, history_floor=history_floor)
    path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(path, index=False)
    return updated


def ensure_membership_csv(
    *,
    csv_path: str | Path = DEFAULT_CSV_PATH,
    check_for_updates: bool = False,
    force_refresh: bool = False,
    timeout_seconds: int = 30,
) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists() or check_for_updates or force_refresh:
        return update_membership_csv_if_needed(
            csv_path=path,
            force_refresh=force_refresh,
            timeout_seconds=timeout_seconds,
        )
    return load_membership_csv(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh/query an approximate Wikipedia-backed S&P 500 membership CSV.",
    )
    parser.add_argument(
        "--mode",
        choices=["update", "query", "range"],
        required=True,
        help="update the CSV, query one date, or return monthly snapshots for a date range.",
    )
    parser.add_argument(
        "--csv_path",
        default=str(DEFAULT_CSV_PATH),
        help="Path to the approximate membership CSV.",
    )
    parser.add_argument(
        "--date",
        help="Single as-of date for query mode. Accepts YYYYMMDD or YYYY-MM-DD.",
    )
    parser.add_argument("--start_date", help="Start date for range mode.")
    parser.add_argument("--end_date", help="End date for range mode.")
    parser.add_argument(
        "--rebalance_timing",
        choices=["month_end", "month_start"],
        default="month_end",
        help="Monthly snapshot timing for range mode.",
    )
    parser.add_argument(
        "--out_csv",
        help="Optional output path for range mode.",
    )
    parser.add_argument(
        "--check_for_updates",
        action="store_true",
        help="If set, compare the local CSV against the live Wikipedia page before querying.",
    )
    parser.add_argument(
        "--force_refresh",
        action="store_true",
        help="If set, rebuild the membership CSV from Wikipedia even if the local file already exists.",
    )
    parser.add_argument(
        "--timeout_seconds",
        type=int,
        default=30,
        help="HTTP timeout for live Wikipedia requests.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.mode == "update":
        membership = update_membership_csv_if_needed(
            csv_path=args.csv_path,
            force_refresh=args.force_refresh,
            timeout_seconds=args.timeout_seconds,
        )
        print(f"Membership CSV ready: {Path(args.csv_path)}")
        print(f"Rows: {len(membership)}")
        latest = _latest_boundary_in_membership(membership)
        print(f"Latest boundary in CSV: {latest.date() if pd.notna(latest) else 'NA'}")
        return 0

    ensure_membership_csv(
        csv_path=args.csv_path,
        check_for_updates=args.check_for_updates,
        force_refresh=args.force_refresh,
        timeout_seconds=args.timeout_seconds,
    )

    if args.mode == "query":
        if not args.date:
            raise SystemExit("--date is required for query mode.")
        universe = get_sp500_universe_for_date(
            args.date,
            csv_path=args.csv_path,
            auto_update=False,
        )
        print(f"As of {pd.Timestamp(_parse_date(args.date)).date()}: {len(universe)} tickers")
        print(",".join(universe))
        return 0

    if not args.start_date or not args.end_date:
        raise SystemExit("--start_date and --end_date are required for range mode.")

    snapshots = get_sp500_universe_between_dates(
        args.start_date,
        args.end_date,
        csv_path=args.csv_path,
        rebalance_timing=args.rebalance_timing,
        auto_update=False,
    )
    if args.out_csv:
        output_path = Path(args.out_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        snapshots.to_csv(output_path, index=False)
        print(f"Monthly universe snapshots saved to: {output_path}")
    else:
        print(snapshots.to_string(index=False))
    print(f"Snapshot rows: {len(snapshots)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
