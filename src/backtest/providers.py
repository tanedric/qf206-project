"""
Prediction and realized-return providers for the modular backtest engine.

These adapters keep the backtest logic independent from model training and
independent from any one market-data source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Protocol, Sequence

import pandas as pd


def _normalize_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce", utc=False)
    if pd.isna(parsed):
        raise ValueError(f"Failed to parse date value: {value}")
    return pd.Timestamp(parsed).normalize()


def _standardize_assets(values: Iterable[object]) -> List[str]:
    return sorted(
        {
            str(value).strip().upper().replace(".", "-")
            for value in values
            if str(value).strip()
        }
    )


class PredictionProvider(Protocol):
    """Supply predicted returns for one rebalance date and one universe."""

    def get_predictions(
        self,
        as_of_date: pd.Timestamp | str,
        universe: Sequence[str],
    ) -> pd.DataFrame:
        """
        Return columns:
        - asset_id
        - predicted_return
        """


class RealizedReturnProvider(Protocol):
    """Supply realized holding-period returns after a portfolio is formed."""

    def prepare(
        self,
        tickers: Sequence[str],
        *,
        start_date: pd.Timestamp | str,
        end_date: pd.Timestamp | str,
    ) -> None:
        """Optional prefetch hook."""

    def get_period_returns(
        self,
        tickers: Sequence[str],
        *,
        start_date: pd.Timestamp | str,
        end_date: pd.Timestamp | str,
    ) -> pd.DataFrame:
        """
        Return columns:
        - asset_id
        - realized_return
        - start_price_date (optional)
        - end_price_date (optional)
        """


@dataclass
class FramePredictionProvider:
    """
    Serve predictions from a precomputed DataFrame.

    Expected columns:
    - date
    - asset_id (or another supplied asset column)
    - predicted_return
    """

    predictions: pd.DataFrame
    date_col: str = "date"
    asset_col: str = "asset_id"
    pred_col: str = "predicted_return"
    _predictions: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        df = self.predictions.copy()
        required = {self.date_col, self.asset_col, self.pred_col}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"FramePredictionProvider missing required column(s): {missing}")
        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce").dt.normalize()
        df[self.asset_col] = (
            df[self.asset_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace(".", "-", regex=False)
        )
        df[self.pred_col] = pd.to_numeric(df[self.pred_col], errors="coerce")
        df = df.dropna(subset=[self.date_col, self.pred_col])
        self._predictions = (
            df.sort_values([self.date_col, self.asset_col], kind="mergesort")
            .drop_duplicates(subset=[self.date_col, self.asset_col], keep="last")
            .reset_index(drop=True)
        )

    def get_predictions(
        self,
        as_of_date: pd.Timestamp | str,
        universe: Sequence[str],
    ) -> pd.DataFrame:
        date_value = _normalize_timestamp(as_of_date)
        universe_set = set(_standardize_assets(universe))
        out = self._predictions[
            (self._predictions[self.date_col] == date_value)
            & (self._predictions[self.asset_col].isin(universe_set))
        ][[self.asset_col, self.pred_col]].copy()
        return out.rename(
            columns={self.asset_col: "asset_id", self.pred_col: "predicted_return"}
        ).reset_index(drop=True)


@dataclass
class FrameRealizedReturnProvider:
    """
    Serve realized returns from a long-format DataFrame.

    Expected columns:
    - date: period end date
    - asset_id
    - return

    This is useful for synthetic tests or when you already have a clean monthly
    realized-return panel saved locally.
    """

    returns: pd.DataFrame
    date_col: str = "date"
    asset_col: str = "asset_id"
    return_col: str = "return"
    _returns: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        df = self.returns.copy()
        required = {self.date_col, self.asset_col, self.return_col}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"FrameRealizedReturnProvider missing required column(s): {missing}")
        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce").dt.normalize()
        df[self.asset_col] = (
            df[self.asset_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace(".", "-", regex=False)
        )
        df[self.return_col] = pd.to_numeric(df[self.return_col], errors="coerce")
        self._returns = (
            df.dropna(subset=[self.date_col, self.return_col])
            .sort_values([self.date_col, self.asset_col], kind="mergesort")
            .drop_duplicates(subset=[self.date_col, self.asset_col], keep="last")
            .reset_index(drop=True)
        )

    def prepare(
        self,
        tickers: Sequence[str],
        *,
        start_date: pd.Timestamp | str,
        end_date: pd.Timestamp | str,
    ) -> None:
        _ = (_standardize_assets(tickers), _normalize_timestamp(start_date), _normalize_timestamp(end_date))

    def get_period_returns(
        self,
        tickers: Sequence[str],
        *,
        start_date: pd.Timestamp | str,
        end_date: pd.Timestamp | str,
    ) -> pd.DataFrame:
        _ = _normalize_timestamp(start_date)
        end_value = _normalize_timestamp(end_date)
        universe_set = set(_standardize_assets(tickers))
        out = self._returns[
            (self._returns[self.date_col] == end_value)
            & (self._returns[self.asset_col].isin(universe_set))
        ][[self.asset_col, self.return_col]].copy()
        return out.rename(
            columns={self.asset_col: "asset_id", self.return_col: "realized_return"}
        ).reset_index(drop=True)


@dataclass
class YFinanceAdjustedCloseReturnProvider:
    """
    Realized-return provider backed by Yahoo Finance adjusted close prices.

    This provider is intended for realized holding-period returns only.
    It is *not* used for historical S&P 500 constituent membership, because
    yfinance does not provide a clean point-in-time constituent history.
    """

    buffer_days: int = 7
    _history: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _cached_tickers: List[str] = field(default_factory=list, init=False, repr=False)
    _cached_start: pd.Timestamp | None = field(default=None, init=False, repr=False)
    _cached_end: pd.Timestamp | None = field(default=None, init=False, repr=False)

    def prepare(
        self,
        tickers: Sequence[str],
        *,
        start_date: pd.Timestamp | str,
        end_date: pd.Timestamp | str,
    ) -> None:
        asset_list = _standardize_assets(tickers)
        if not asset_list:
            self._history = pd.DataFrame()
            self._cached_tickers = []
            return

        start_value = _normalize_timestamp(start_date) - pd.Timedelta(days=self.buffer_days)
        end_value = _normalize_timestamp(end_date) + pd.Timedelta(days=self.buffer_days)
        if (
            self._history is not None
            and set(asset_list).issubset(set(self._cached_tickers))
            and self._cached_start is not None
            and self._cached_end is not None
            and start_value >= self._cached_start
            and end_value <= self._cached_end
        ):
            return

        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is required for Yahoo-based realized returns. "
                "Install it with: pip install yfinance"
            ) from exc

        raw = yf.download(
            tickers=asset_list,
            start=start_value.strftime("%Y-%m-%d"),
            end=(end_value + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
            actions=False,
            threads=True,
        )
        if raw.empty:
            self._history = pd.DataFrame()
            self._cached_tickers = list(asset_list)
            self._cached_start = start_value
            self._cached_end = end_value
            return

        if isinstance(raw.columns, pd.MultiIndex):
            price_frame = None
            for field_name in ("Adj Close", "Close"):
                if field_name in raw.columns.get_level_values(0):
                    price_frame = raw[field_name].copy()
                    break
            if price_frame is None:
                raise RuntimeError("Yahoo download did not contain Adj Close or Close columns.")
        else:
            column_name = "Adj Close" if "Adj Close" in raw.columns else "Close"
            if column_name not in raw.columns:
                raise RuntimeError("Yahoo download did not contain Adj Close or Close columns.")
            one_ticker = asset_list[0]
            price_frame = raw[[column_name]].rename(columns={column_name: one_ticker})

        price_frame.index = pd.to_datetime(price_frame.index, errors="coerce").normalize()
        price_frame = price_frame.sort_index(kind="mergesort")
        price_frame.columns = [str(col).strip().upper().replace(".", "-") for col in price_frame.columns]

        self._history = price_frame
        self._cached_tickers = list(asset_list)
        self._cached_start = start_value
        self._cached_end = end_value

    def get_period_returns(
        self,
        tickers: Sequence[str],
        *,
        start_date: pd.Timestamp | str,
        end_date: pd.Timestamp | str,
    ) -> pd.DataFrame:
        asset_list = _standardize_assets(tickers)
        start_value = _normalize_timestamp(start_date)
        end_value = _normalize_timestamp(end_date)
        self.prepare(asset_list, start_date=start_value, end_date=end_value)

        if self._history is None or self._history.empty:
            return pd.DataFrame(
                columns=[
                    "asset_id",
                    "realized_return",
                    "start_price_date",
                    "end_price_date",
                    "start_price",
                    "end_price",
                ]
            )

        rows: List[Dict[str, object]] = []
        for asset_id in asset_list:
            if asset_id not in self._history.columns:
                continue
            series = pd.to_numeric(self._history[asset_id], errors="coerce").dropna()
            if series.empty:
                continue
            start_candidates = series.index[series.index <= start_value]
            end_candidates = series.index[series.index <= end_value]
            if len(start_candidates) == 0 or len(end_candidates) == 0:
                continue
            start_obs = pd.Timestamp(start_candidates.max()).normalize()
            end_obs = pd.Timestamp(end_candidates.max()).normalize()
            if end_obs <= start_obs:
                continue

            start_price = float(series.loc[start_obs])
            end_price = float(series.loc[end_obs])
            if start_price == 0.0:
                continue

            rows.append(
                {
                    "asset_id": asset_id,
                    "realized_return": end_price / start_price - 1.0,
                    "start_price_date": start_obs,
                    "end_price_date": end_obs,
                    "start_price": start_price,
                    "end_price": end_price,
                }
            )

        return pd.DataFrame(rows)
