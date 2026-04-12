"""
Universe providers for the modular backtest workflow.

The key design choice is to keep universe construction separate from both
model training and return realization. This lets us swap in a proper
point-in-time S&P 500 membership table later without rewriting the backtest
engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Protocol, Sequence

import pandas as pd


def _normalize_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce", utc=False)
    if pd.isna(parsed):
        raise ValueError(f"Failed to parse date value: {value}")
    return pd.Timestamp(parsed).normalize()


def _standardize_tickers(values: Sequence[object]) -> List[str]:
    tickers = {
        str(value).strip().upper().replace(".", "-")
        for value in values
        if str(value).strip()
    }
    return sorted(tickers)


class UniverseProvider(Protocol):
    """Point-in-time universe provider."""

    def get_universe(self, as_of_date: pd.Timestamp | str) -> List[str]:
        """Return the asset universe available on one rebalance date."""


def build_universe_snapshot_panel(
    provider: UniverseProvider,
    *,
    dates: Sequence[pd.Timestamp | str],
) -> pd.DataFrame:
    """
    Materialize a universe provider across a set of dates for inspection.

    Returns columns:
    - date
    - universe_size
    - tickers
    """

    rows = []
    for value in dates:
        as_of_date = _normalize_timestamp(value)
        universe = _standardize_tickers(provider.get_universe(as_of_date))
        rows.append(
            {
                "date": as_of_date,
                "universe_size": len(universe),
                "tickers": "|".join(universe),
            }
        )
    return pd.DataFrame(rows)


def get_universe_from_membership_csv(
    csv_path: str | Path,
    *,
    as_of_date: pd.Timestamp | str,
    ticker_col: str = "ticker",
    start_col: str = "start_date",
    end_col: str = "end_date",
) -> List[str]:
    """
    Convenience helper for one-off point-in-time lookups from a membership CSV.
    """

    provider = MembershipTableUniverseProvider.from_csv(
        csv_path,
        ticker_col=ticker_col,
        start_col=start_col,
        end_col=end_col,
    )
    return provider.get_universe(as_of_date)


@dataclass
class StaticUniverseProvider:
    """
    Return the same universe for every date.

    Useful for early backtest scaffolding or for a fixed user-supplied basket.
    """

    tickers: Sequence[str]
    _universe: List[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._universe = _standardize_tickers(self.tickers)
        if not self._universe:
            raise ValueError("StaticUniverseProvider requires at least one ticker.")

    def get_universe(self, as_of_date: pd.Timestamp | str) -> List[str]:
        _ = _normalize_timestamp(as_of_date)
        return list(self._universe)


@dataclass
class MembershipTableUniverseProvider:
    """
    Historical point-in-time universe provider backed by a membership table.

    Expected columns:
    - ticker
    - start_date
    - end_date (optional; null means still active)
    """

    memberships: pd.DataFrame
    ticker_col: str = "ticker"
    start_col: str = "start_date"
    end_col: str = "end_date"
    _memberships: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        df = self.memberships.copy()
        required = {self.ticker_col, self.start_col}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"MembershipTableUniverseProvider missing required column(s): {missing}"
            )
        if self.end_col not in df.columns:
            df[self.end_col] = pd.NaT

        df[self.ticker_col] = (
            df[self.ticker_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace(".", "-", regex=False)
        )
        df = df[df[self.ticker_col] != ""].copy()
        df[self.start_col] = pd.to_datetime(df[self.start_col], errors="coerce").dt.normalize()
        df[self.end_col] = pd.to_datetime(df[self.end_col], errors="coerce").dt.normalize()
        df = df.dropna(subset=[self.start_col]).reset_index(drop=True)
        self._memberships = df

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        *,
        ticker_col: str = "ticker",
        start_col: str = "start_date",
        end_col: str = "end_date",
    ) -> "MembershipTableUniverseProvider":
        df = pd.read_csv(csv_path)
        return cls(
            memberships=df,
            ticker_col=ticker_col,
            start_col=start_col,
            end_col=end_col,
        )

    def get_universe(self, as_of_date: pd.Timestamp | str) -> List[str]:
        date_value = _normalize_timestamp(as_of_date)
        mask = self._memberships[self.start_col] <= date_value
        end_dates = self._memberships[self.end_col]
        mask &= end_dates.isna() | (end_dates >= date_value)
        return _standardize_tickers(self._memberships.loc[mask, self.ticker_col].tolist())


@dataclass
class WikipediaCurrentSP500UniverseProvider:
    """
    Fetch the current S&P 500 constituents from Wikipedia.

    Important limitation:
    - this is *not* a historical constituent source
    - the same current list is returned for every requested date

    Use this only as a prototype fallback when you do not yet have a proper
    point-in-time membership table.
    """

    url: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    _cached_tickers: List[str] | None = field(default=None, init=False, repr=False)

    def _load_current_constituents(self) -> List[str]:
        if self._cached_tickers is not None:
            return list(self._cached_tickers)
        try:
            tables = pd.read_html(self.url)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to download the current S&P 500 constituents from Wikipedia. "
                "Install the HTML parser dependencies or provide a historical membership CSV."
            ) from exc
        if not tables:
            raise RuntimeError("Wikipedia did not return any tables for the S&P 500 constituent page.")
        table = tables[0]
        symbol_col = "Symbol" if "Symbol" in table.columns else table.columns[0]
        self._cached_tickers = _standardize_tickers(table[symbol_col].tolist())
        if not self._cached_tickers:
            raise RuntimeError("Failed to parse any S&P 500 tickers from the Wikipedia table.")
        return list(self._cached_tickers)

    def get_universe(self, as_of_date: pd.Timestamp | str) -> List[str]:
        _ = _normalize_timestamp(as_of_date)
        return self._load_current_constituents()
