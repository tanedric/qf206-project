"""
Live-compatible feature formulas built from Polygon market data and Finnhub fundamentals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LIVE_REQUIRED_FEATURES: List[str] = [
    "turn",
    "nincr",
    "std_turn",
    "cinvest",
    "baspread",
    "operprof",
    "rd",
]

LIVE_OPTIONAL_FEATURES: List[str] = [
    "Mom6m",
    "Mom12m",
    "RealizedVol",
    "DolVol",
    "Illiquidity",
    "MaxRet",
]

LIVE_ALL_FEATURES: List[str] = LIVE_REQUIRED_FEATURES + LIVE_OPTIONAL_FEATURES

FEATURE_SOURCE_MAP: Dict[str, str] = {
    "turn": "Polygon daily volume + Finnhub point-in-time shares_outstanding",
    "std_turn": "Polygon daily volume + Finnhub point-in-time shares_outstanding",
    "baspread": "Polygon daily bars via Corwin-Schultz high/low spread estimator",
    "Mom6m": "Polygon monthly closes built from daily bars",
    "Mom12m": "Polygon monthly closes built from daily bars",
    "RealizedVol": "Polygon daily close returns within the current month",
    "DolVol": "Polygon daily dollar volume over the prior 2 completed months",
    "Illiquidity": "Polygon daily Amihud-style illiquidity over the prior 12 completed months",
    "MaxRet": "Polygon maximum daily return over the prior completed month",
    "shares_outstanding": "Finnhub reported financials (current-profile fallback only for current month if needed)",
    "rd": "Finnhub reported financials + Polygon month-end close for market capitalization",
    "operprof": "Finnhub reported financials",
    "cinvest": "Finnhub reported financials",
    "nincr": "Finnhub reported financials",
}


@dataclass
class FeatureComputationOutput:
    panel: pd.DataFrame
    missing: pd.DataFrame


def _as_timestamp(value: Any) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce", utc=False)
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed).normalize()


def _month_key(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.to_period("M").dt.to_timestamp("M")


def _previous_month_key(month_end: pd.Timestamp, n: int = 1) -> pd.Timestamp:
    return pd.Timestamp(month_end).to_period("M").to_timestamp("M") - pd.offsets.MonthEnd(n)


def _append_missing(
    rows: List[Dict[str, Any]],
    *,
    date: pd.Timestamp,
    ticker: str,
    feature: str,
    reason: str,
    source: str,
) -> None:
    rows.append(
        {
            "date": pd.Timestamp(date).normalize(),
            "ticker": ticker.upper(),
            "feature": feature,
            "reason": reason,
            "source": source,
        }
    )


def prepare_market_data(daily_bars: pd.DataFrame) -> pd.DataFrame:
    out = daily_bars.copy()
    if out.empty:
        return out
    out = out.sort_values("date", kind="mergesort").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["high"] = pd.to_numeric(out["high"], errors="coerce")
    out["low"] = pd.to_numeric(out["low"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out["daily_return"] = out["close"].pct_change()
    out["dollar_volume"] = out["close"].abs() * out["volume"]
    out["month_key"] = _month_key(out["date"])
    out["cs_spread"] = corwin_schultz_spread(out["high"], out["low"])
    out["illiq_daily"] = np.abs(out["daily_return"]) / out["dollar_volume"].replace(0, np.nan)
    return out


def corwin_schultz_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Corwin-Schultz bid-ask spread estimator based on adjacent daily high/low ranges.
    """

    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    hl = np.log(high / low)
    beta = hl.pow(2) + hl.shift(1).pow(2)
    gamma = np.log(np.maximum(high, high.shift(1)) / np.minimum(low, low.shift(1))).pow(2)
    denom = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
    alpha = alpha.clip(lower=0.0)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return pd.Series(spread, index=high.index, dtype=float)


def build_monthly_market_summary(
    prepared_bars: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    if prepared_bars.empty:
        return pd.DataFrame(
            columns=[
                "month_key",
                "period_end",
                "last_close",
                "last_trade_date",
                "month_volume",
                "baspread_month",
                "realized_vol_month",
                "dollar_volume_mean",
                "max_daily_return_month",
                "is_complete",
            ]
        )

    grouped = prepared_bars.groupby("month_key", sort=True)[
        ["date", "close", "volume", "cs_spread", "daily_return", "dollar_volume"]
    ]
    monthly = grouped.apply(
        lambda g: pd.Series(
            {
                "period_end": pd.Timestamp(min(g["date"].max(), as_of_date)).normalize(),
                "last_close": g["close"].iloc[-1],
                "last_trade_date": g["date"].max(),
                "month_volume": g["volume"].sum(skipna=True),
                "baspread_month": g["cs_spread"].mean(skipna=True),
                "realized_vol_month": g["daily_return"].std(skipna=True),
                "dollar_volume_mean": g["dollar_volume"].mean(skipna=True),
                "max_daily_return_month": g["daily_return"].max(skipna=True),
            }
        )
    )
    monthly = monthly.reset_index()
    current_month_key = as_of_date.to_period("M").to_timestamp("M")
    monthly["is_complete"] = monthly["month_key"] < current_month_key
    if as_of_date == current_month_key:
        monthly.loc[monthly["month_key"] == current_month_key, "is_complete"] = True
    return monthly.sort_values("month_key", kind="mergesort").reset_index(drop=True)


def _month_subset(
    prepared_bars: pd.DataFrame,
    *,
    start_month: pd.Timestamp,
    end_month: pd.Timestamp,
    include_end_month: bool = True,
) -> pd.DataFrame:
    month_key = prepared_bars["month_key"]
    if include_end_month:
        mask = (month_key >= start_month) & (month_key <= end_month)
    else:
        mask = (month_key >= start_month) & (month_key < end_month)
    return prepared_bars.loc[mask].copy()


def compute_market_feature_panel(
    ticker: str,
    daily_bars: pd.DataFrame,
    *,
    evaluation_dates: Sequence[pd.Timestamp],
    shares_by_date: Mapping[pd.Timestamp, float],
    as_of_date: pd.Timestamp,
) -> FeatureComputationOutput:
    prepared = prepare_market_data(daily_bars)
    monthly = build_monthly_market_summary(prepared, as_of_date=as_of_date)
    monthly = monthly.set_index("month_key", drop=False)

    records: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []

    for period_end in evaluation_dates:
        period_end = _as_timestamp(period_end)
        month_key = period_end.to_period("M").to_timestamp("M")
        current = monthly.loc[month_key] if month_key in monthly.index else None

        row: Dict[str, Any] = {
            "date": period_end,
            "ticker": ticker.upper(),
            "turn": np.nan,
            "baspread": np.nan,
            "Mom6m": np.nan,
            "Mom12m": np.nan,
            "RealizedVol": np.nan,
            "DolVol": np.nan,
            "Illiquidity": np.nan,
            "MaxRet": np.nan,
            "target_return": np.nan,
            "month_close": np.nan,
        }

        if current is None or pd.isna(current["last_close"]):
            for feature in (
                "turn",
                "baspread",
                "Mom6m",
                "Mom12m",
                "RealizedVol",
                "DolVol",
                "Illiquidity",
                "MaxRet",
                "target_return",
            ):
                _append_missing(
                    missing_rows,
                    date=period_end,
                    ticker=ticker,
                    feature=feature,
                    reason="no_polygon_market_data_for_month",
                    source="Polygon",
                )
            records.append(row)
            continue

        row["month_close"] = float(current["last_close"])
        row["baspread"] = float(current["baspread_month"]) if pd.notna(current["baspread_month"]) else np.nan
        row["RealizedVol"] = (
            float(current["realized_vol_month"]) if pd.notna(current["realized_vol_month"]) else np.nan
        )

        shares_outstanding = pd.to_numeric(shares_by_date.get(period_end), errors="coerce")
        if pd.notna(shares_outstanding) and float(shares_outstanding) > 0 and pd.notna(current["month_volume"]):
            row["turn"] = float(current["month_volume"]) / float(shares_outstanding)
        else:
            _append_missing(
                missing_rows,
                date=period_end,
                ticker=ticker,
                feature="turn",
                reason="missing_or_nonpositive_shares_outstanding",
                source="Polygon+Finnhub",
            )

        if pd.isna(row["baspread"]):
            _append_missing(
                missing_rows,
                date=period_end,
                ticker=ticker,
                feature="baspread",
                reason="insufficient_high_low_history_for_corwin_schultz",
                source="Polygon",
            )

        if pd.isna(row["RealizedVol"]):
            _append_missing(
                missing_rows,
                date=period_end,
                ticker=ticker,
                feature="RealizedVol",
                reason="insufficient_daily_returns_in_month",
                source="Polygon",
            )

        # Momentum uses completed months only, excluding the current processing month.
        prior_months = monthly.loc[monthly.index < month_key].copy()
        if len(prior_months) >= 6:
            month_returns = prior_months["last_close"].pct_change().dropna()
            if len(month_returns) >= 6:
                row["Mom6m"] = float((1.0 + month_returns.iloc[-6:]).prod() - 1.0)
        if pd.isna(row["Mom6m"]):
            _append_missing(
                missing_rows,
                date=period_end,
                ticker=ticker,
                feature="Mom6m",
                reason="insufficient_completed_months_for_6m_momentum",
                source="Polygon",
            )

        if len(prior_months) >= 12:
            month_returns = prior_months["last_close"].pct_change().dropna()
            if len(month_returns) >= 12:
                row["Mom12m"] = float((1.0 + month_returns.iloc[-12:]).prod() - 1.0)
        if pd.isna(row["Mom12m"]):
            _append_missing(
                missing_rows,
                date=period_end,
                ticker=ticker,
                feature="Mom12m",
                reason="insufficient_completed_months_for_12m_momentum",
                source="Polygon",
            )

        prior_two_months_start = month_key - pd.offsets.MonthEnd(2)
        dolvol_window = _month_subset(
            prepared,
            start_month=prior_two_months_start,
            end_month=month_key,
            include_end_month=False,
        )
        if not dolvol_window.empty and dolvol_window["dollar_volume"].notna().any():
            row["DolVol"] = float(np.log1p(dolvol_window["dollar_volume"].mean(skipna=True)))
        else:
            _append_missing(
                missing_rows,
                date=period_end,
                ticker=ticker,
                feature="DolVol",
                reason="insufficient_prior_two_month_dollar_volume_history",
                source="Polygon",
            )

        illiq_window = _month_subset(
            prepared,
            start_month=month_key - pd.offsets.MonthEnd(12),
            end_month=month_key,
            include_end_month=False,
        )
        if not illiq_window.empty and illiq_window["illiq_daily"].notna().any():
            row["Illiquidity"] = float(illiq_window["illiq_daily"].mean(skipna=True))
        else:
            _append_missing(
                missing_rows,
                date=period_end,
                ticker=ticker,
                feature="Illiquidity",
                reason="insufficient_prior_12_month_illiquidity_history",
                source="Polygon",
            )

        prev_month_key = _previous_month_key(month_key, 1)
        if prev_month_key in monthly.index and pd.notna(monthly.loc[prev_month_key, "max_daily_return_month"]):
            row["MaxRet"] = float(monthly.loc[prev_month_key, "max_daily_return_month"])
        else:
            _append_missing(
                missing_rows,
                date=period_end,
                ticker=ticker,
                feature="MaxRet",
                reason="missing_previous_month_daily_returns",
                source="Polygon",
            )

        next_month_key = month_key + pd.offsets.MonthEnd(1)
        if (
            next_month_key in monthly.index
            and bool(monthly.loc[next_month_key, "is_complete"])
            and pd.notna(monthly.loc[next_month_key, "last_close"])
            and pd.notna(current["last_close"])
            and float(current["last_close"]) != 0.0
        ):
            row["target_return"] = float(monthly.loc[next_month_key, "last_close"] / current["last_close"] - 1.0)
        else:
            _append_missing(
                missing_rows,
                date=period_end,
                ticker=ticker,
                feature="target_return",
                reason="next_month_close_not_yet_fully_realized",
                source="Polygon",
            )

        records.append(row)

    panel = pd.DataFrame.from_records(records).sort_values("date", kind="mergesort").reset_index(drop=True)
    missing = pd.DataFrame.from_records(missing_rows)
    if missing.empty:
        missing = pd.DataFrame(columns=["date", "ticker", "feature", "reason", "source"])
    return FeatureComputationOutput(panel=panel, missing=missing)


def _dedupe_filings(filings: pd.DataFrame) -> pd.DataFrame:
    if filings.empty:
        return filings.copy()
    out = filings.copy()
    out["available_date"] = pd.to_datetime(out["available_date"], errors="coerce").dt.normalize()
    out["report_end_date"] = pd.to_datetime(out["report_end_date"], errors="coerce").dt.normalize()
    return (
        out.sort_values(["available_date", "report_end_date"], kind="mergesort")
        .drop_duplicates(subset=["report_end_date"], keep="last")
        .reset_index(drop=True)
    )


def _available_as_of(filings: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    if filings.empty:
        return filings.copy()
    out = filings.copy()
    out = out[pd.to_datetime(out["available_date"], errors="coerce") <= as_of_date]
    return out.sort_values("report_end_date", kind="mergesort").reset_index(drop=True)


def _latest_with_value(filings: pd.DataFrame, field: str) -> Optional[pd.Series]:
    if filings.empty or field not in filings.columns:
        return None
    sub = filings[filings[field].notna()]
    if sub.empty:
        return None
    return sub.sort_values("report_end_date", kind="mergesort").iloc[-1]


def _last_n_quarters(filings: pd.DataFrame, n: int) -> pd.DataFrame:
    if filings.empty:
        return filings.copy()
    out = filings.sort_values("report_end_date", kind="mergesort").drop_duplicates(
        subset=["report_end_date"], keep="last"
    )
    return out.tail(n).reset_index(drop=True)


def _ttm_sum(filings: pd.DataFrame, field: str) -> Optional[float]:
    last4 = _last_n_quarters(filings, 4)
    if len(last4) < 4 or field not in last4.columns or last4[field].isna().any():
        return None
    return float(last4[field].sum())


def _latest_current_and_lagged_4q(filings: pd.DataFrame, field: str) -> Tuple[Optional[float], Optional[float]]:
    if filings.empty or field not in filings.columns:
        return None, None
    ordered = filings.sort_values("report_end_date", kind="mergesort").drop_duplicates(
        subset=["report_end_date"], keep="last"
    )
    if len(ordered) < 5:
        return None, None
    current = ordered.iloc[-1].get(field)
    lagged = ordered.iloc[-5].get(field)
    current = None if pd.isna(current) else float(current)
    lagged = None if pd.isna(lagged) else float(lagged)
    return current, lagged


def _earnings_increase_streak(filings: pd.DataFrame, *, cap: int = 8) -> Optional[int]:
    if filings.empty or "net_income" not in filings.columns:
        return None
    ordered = filings.sort_values("report_end_date", kind="mergesort").drop_duplicates(
        subset=["report_end_date"], keep="last"
    )
    series = pd.to_numeric(ordered["net_income"], errors="coerce").dropna()
    if len(series) < 2:
        return None
    streak = 0
    values = series.to_numpy(dtype=float)
    for idx in range(len(values) - 1, 0, -1):
        if values[idx] > values[idx - 1]:
            streak += 1
            if streak >= cap:
                return cap
        else:
            break
    return streak


def compute_fundamental_feature_panel(
    ticker: str,
    *,
    quarterly_filings: pd.DataFrame,
    annual_filings: pd.DataFrame,
    evaluation_dates: Sequence[pd.Timestamp],
    month_close_by_date: Mapping[pd.Timestamp, float],
    current_profile: Optional[Mapping[str, Any]] = None,
    stale_threshold_days: int = 540,
    as_of_date: Optional[pd.Timestamp] = None,
) -> FeatureComputationOutput:
    quarterly_filings = _dedupe_filings(quarterly_filings)
    annual_filings = _dedupe_filings(annual_filings)
    profile = dict(current_profile or {})
    live_month = as_of_date.to_period("M").to_timestamp("M") if as_of_date is not None else None

    rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []

    for period_end in evaluation_dates:
        period_end = _as_timestamp(period_end)
        available_q = _available_as_of(quarterly_filings, period_end)
        available_a = _available_as_of(annual_filings, period_end)
        latest_any = pd.concat([available_q, available_a], axis=0, ignore_index=True)
        latest_any = latest_any.sort_values("report_end_date", kind="mergesort").reset_index(drop=True)

        row: Dict[str, Any] = {
            "date": period_end,
            "ticker": ticker.upper(),
            "shares_outstanding": np.nan,
            "rd": np.nan,
            "operprof": np.nan,
            "cinvest": np.nan,
            "nincr": np.nan,
            "fundamental_report_end_date": pd.NaT,
            "fundamental_available_date": pd.NaT,
        }

        latest_shares_row = _latest_with_value(available_q, "shares_outstanding")
        if latest_shares_row is None:
            latest_shares_row = _latest_with_value(available_a, "shares_outstanding")
        if latest_shares_row is not None:
            row["shares_outstanding"] = float(latest_shares_row["shares_outstanding"])
            row["fundamental_report_end_date"] = latest_shares_row["report_end_date"]
            row["fundamental_available_date"] = latest_shares_row["available_date"]
        elif live_month is not None and period_end.to_period("M").to_timestamp("M") == live_month:
            profile_shares = pd.to_numeric(profile.get("shareOutstanding"), errors="coerce")
            if pd.notna(profile_shares) and float(profile_shares) > 0:
                row["shares_outstanding"] = float(profile_shares)
                row["fundamental_available_date"] = period_end
            else:
                _append_missing(
                    missing_rows,
                    date=period_end,
                    ticker=ticker,
                    feature="shares_outstanding",
                    reason="missing_finnhub_reported_and_profile_shares_outstanding",
                    source="Finnhub",
                )
        else:
            _append_missing(
                missing_rows,
                date=period_end,
                ticker=ticker,
                feature="shares_outstanding",
                reason="missing_point_in_time_shares_outstanding",
                source="Finnhub",
            )

        if not latest_any.empty:
            most_recent_available = latest_any.iloc[-1]
            if pd.notna(most_recent_available["available_date"]):
                age_days = int((period_end - pd.Timestamp(most_recent_available["available_date"])).days)
                if age_days > stale_threshold_days:
                    for feature in ("rd", "operprof", "cinvest", "nincr"):
                        _append_missing(
                            missing_rows,
                            date=period_end,
                            ticker=ticker,
                            feature=feature,
                            reason=f"stale_fundamentals_{age_days}d_old",
                            source="Finnhub",
                        )
                    rows.append(row)
                    continue

        month_close = pd.to_numeric(month_close_by_date.get(period_end), errors="coerce")

        # rd: TTM R&D divided by point-in-time market cap.
        rnd_ttm = _ttm_sum(available_q, "rnd")
        if rnd_ttm is None:
            latest_annual = _latest_with_value(available_a, "rnd")
            rnd_ttm = None if latest_annual is None else float(latest_annual["rnd"])
        if (
            rnd_ttm is not None
            and pd.notna(month_close)
            and float(month_close) > 0.0
            and pd.notna(row["shares_outstanding"])
            and float(row["shares_outstanding"]) > 0.0
        ):
            market_cap = float(month_close) * float(row["shares_outstanding"])
            if market_cap > 0:
                row["rd"] = float(rnd_ttm) / market_cap
        if pd.isna(row["rd"]):
            _append_missing(
                missing_rows,
                date=period_end,
                ticker=ticker,
                feature="rd",
                reason="missing_rnd_ttm_or_market_cap_inputs",
                source="Finnhub+Polygon",
            )

        # operprof: TTM (revenue - cogs - sga - interest_expense) / book equity.
        op_numerator = None
        last4 = _last_n_quarters(available_q, 4)
        if len(last4) == 4:
            if last4["operating_income"].notna().all():
                interest = last4["interest_expense"].fillna(0.0)
                op_numerator = float(last4["operating_income"].sum() - interest.sum())
            elif last4["revenue"].notna().all():
                cogs = last4["cogs"].fillna(0.0)
                sga = last4["sga"].fillna(0.0)
                interest = last4["interest_expense"].fillna(0.0)
                op_numerator = float(last4["revenue"].sum() - cogs.sum() - sga.sum() - interest.sum())
        if op_numerator is None:
            annual_base = available_a.sort_values("report_end_date", kind="mergesort").tail(1)
            if not annual_base.empty:
                annual = annual_base.iloc[-1]
                if pd.notna(annual.get("operating_income")):
                    interest_value = pd.to_numeric(annual.get("interest_expense"), errors="coerce")
                    interest_value = 0.0 if pd.isna(interest_value) else float(interest_value)
                    op_numerator = float(annual["operating_income"] - interest_value)
                elif pd.notna(annual.get("revenue")):
                    cogs_value = pd.to_numeric(annual.get("cogs"), errors="coerce")
                    sga_value = pd.to_numeric(annual.get("sga"), errors="coerce")
                    interest_value = pd.to_numeric(annual.get("interest_expense"), errors="coerce")
                    cogs_value = 0.0 if pd.isna(cogs_value) else float(cogs_value)
                    sga_value = 0.0 if pd.isna(sga_value) else float(sga_value)
                    interest_value = 0.0 if pd.isna(interest_value) else float(interest_value)
                    op_numerator = float(
                        annual["revenue"]
                        - cogs_value
                        - sga_value
                        - interest_value
                    )
        latest_equity_row = _latest_with_value(available_q, "equity")
        if latest_equity_row is None:
            latest_equity_row = _latest_with_value(available_a, "equity")
        if latest_equity_row is not None and op_numerator is not None:
            equity = pd.to_numeric(latest_equity_row["equity"], errors="coerce")
            if pd.notna(equity) and float(equity) > 0:
                row["operprof"] = float(op_numerator) / float(equity)
        if pd.isna(row["operprof"]):
            _append_missing(
                missing_rows,
                date=period_end,
                ticker=ticker,
                feature="operprof",
                reason="missing_operating_profitability_inputs",
                source="Finnhub",
            )

        # cinvest: change in (PPE + inventory) over lagged assets, using 4-quarter lag when available.
        current_ppe, lagged_ppe = _latest_current_and_lagged_4q(available_q, "ppe")
        current_inv, lagged_inv = _latest_current_and_lagged_4q(available_q, "inventory")
        _, lagged_assets = _latest_current_and_lagged_4q(available_q, "assets")
        if (
            current_ppe is not None
            and lagged_ppe is not None
            and current_inv is not None
            and lagged_inv is not None
            and lagged_assets is not None
            and lagged_assets != 0.0
        ):
            row["cinvest"] = ((current_ppe + current_inv) - (lagged_ppe + lagged_inv)) / lagged_assets
        else:
            annual_sorted = available_a.sort_values("report_end_date", kind="mergesort")
            if len(annual_sorted) >= 2:
                latest = annual_sorted.iloc[-1]
                prev = annual_sorted.iloc[-2]
                if all(
                    pd.notna(latest.get(col)) and pd.notna(prev.get(col))
                    for col in ("ppe", "inventory", "assets")
                ) and float(prev["assets"]) != 0.0:
                    row["cinvest"] = float(
                        ((latest["ppe"] + latest["inventory"]) - (prev["ppe"] + prev["inventory"]))
                        / prev["assets"]
                    )
        if pd.isna(row["cinvest"]):
            _append_missing(
                missing_rows,
                date=period_end,
                ticker=ticker,
                feature="cinvest",
                reason="missing_capital_investment_inputs",
                source="Finnhub",
            )

        streak = _earnings_increase_streak(available_q, cap=8)
        if streak is not None:
            row["nincr"] = float(streak)
        else:
            _append_missing(
                missing_rows,
                date=period_end,
                ticker=ticker,
                feature="nincr",
                reason="insufficient_quarterly_net_income_history",
                source="Finnhub",
            )

        rows.append(row)

    panel = pd.DataFrame.from_records(rows).sort_values("date", kind="mergesort").reset_index(drop=True)
    missing = pd.DataFrame.from_records(missing_rows)
    if missing.empty:
        missing = pd.DataFrame(columns=["date", "ticker", "feature", "reason", "source"])
    return FeatureComputationOutput(panel=panel, missing=missing)
