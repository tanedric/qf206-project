"""
Resolve returns panels to the canonical permno-based internal identifier.

This module supports:
- ticker-based returns via a historical DATE/TICKER/PERMNO/NAMEENDT mapping file
- permno-based returns directly
- monthly alignment to end-of-month for the OAP characteristics pipeline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.load_data import (
    detect_return_column,
    parse_yyyymm_to_date,
    peek_csv_columns,
)


PathLike = str | Path


@dataclass
class ReturnsResolutionSummary:
    identifier_kind: str
    n_input_rows: int
    n_mapped_rows: int
    n_unmatched_rows: int
    n_ambiguous_rows: int
    n_duplicate_key_rows: int
    n_final_rows: int
    returns_date_samples: list[str] = field(default_factory=list)
    mapping_start_date_samples: list[str] = field(default_factory=list)
    mapping_end_date_samples: list[str] = field(default_factory=list)
    unmatched_examples: list[dict[str, object]] = field(default_factory=list)
    ambiguous_examples: list[dict[str, object]] = field(default_factory=list)


@dataclass
class ResolvedReturnsResult:
    returns_df: pd.DataFrame
    summary: ReturnsResolutionSummary


def _resolve_column_name(columns: Sequence[str], candidates: Iterable[str]) -> Optional[str]:
    lower_map = {str(c).lower(): str(c) for c in columns}
    for cand in candidates:
        found = lower_map.get(str(cand).lower())
        if found is not None:
            return found
    return None


def _sample_dates(series: pd.Series, *, n: int = 5) -> list[str]:
    dates = pd.to_datetime(series, errors="coerce").dropna().head(n)
    return [d.strftime("%Y-%m-%d") for d in dates]


def _parse_direct_dates(series: pd.Series, *, name: str) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=False)
    n_bad = int(parsed.isna().sum())
    if n_bad > 0:
        bad = series.loc[parsed.isna()].head(5).tolist()
        raise ValueError(f"Failed to parse {n_bad} values in '{name}' as dates. Examples: {bad}")
    return parsed


def _standardize_ticker(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip().str.upper()
    return out.mask(out.isin(["", "NAN", "NONE", "<NA>"]))


def _validate_required_columns(df: pd.DataFrame, required: Sequence[str], *, name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required column(s): {missing}. Found: {list(df.columns)}")


def _detect_identifier_kind(values: pd.Series) -> str:
    non_null = values.dropna().astype(str).str.strip()
    if non_null.empty:
        raise ValueError("returns CSV asset identifier column is empty.")
    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.notna().all():
        return "permno"
    if numeric.isna().all():
        return "ticker"
    bad_examples = non_null[numeric.isna()].head(5).tolist()
    raise ValueError(
        "returns CSV mixes numeric and non-numeric asset identifiers. "
        f"Examples of non-numeric values: {bad_examples}"
    )


def load_ticker_permno_map(
    csv_path: PathLike,
    *,
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a historical ticker/PERMNO mapping table with date windows.

    Expected columns when present:
    - DATE (effective start date)
    - NAMEENDDT or NAMEENDT (effective end date)
    - TICKER
    - PERMNO
    - COMNAM (optional)
    """

    available_cols = peek_csv_columns(csv_path, encoding=encoding)
    start_col = _resolve_column_name(available_cols, ["DATE"])
    end_col = _resolve_column_name(available_cols, ["NAMEENDDT", "NAMEENDT"])
    ticker_col = _resolve_column_name(available_cols, ["TICKER"])
    permno_col = _resolve_column_name(available_cols, ["PERMNO"])
    comnam_col = _resolve_column_name(available_cols, ["COMNAM"])

    required = {
        "DATE": start_col,
        "TICKER": ticker_col,
        "PERMNO": permno_col,
        "NAMEENDDT/NAMEENDT": end_col,
    }
    missing = [label for label, actual in required.items() if actual is None]
    if missing:
        raise ValueError(
            "ticker_permno_map.csv is missing required column(s): "
            f"{missing}. Found: {available_cols}"
        )

    usecols = [start_col, ticker_col, permno_col, end_col]
    if comnam_col is not None:
        usecols.append(comnam_col)

    df = pd.read_csv(csv_path, usecols=usecols, encoding=encoding, low_memory=False)
    rename_map = {
        start_col: "mapping_start_date",
        end_col: "mapping_end_date",
        ticker_col: "ticker",
        permno_col: "asset_id",
    }
    if comnam_col is not None:
        rename_map[comnam_col] = "company_name"
    out = df.rename(columns=rename_map).copy()
    out.insert(0, "mapping_row_number", np.arange(len(out), dtype=np.int64) + 2)
    out["mapping_start_date"] = _parse_direct_dates(
        out["mapping_start_date"], name="ticker_permno_map.DATE"
    )
    out["mapping_end_date"] = _parse_direct_dates(
        out["mapping_end_date"], name=f"ticker_permno_map.{end_col}"
    )
    out["ticker"] = _standardize_ticker(out["ticker"])
    asset_numeric = pd.to_numeric(out["asset_id"], errors="coerce")
    if asset_numeric.isna().any():
        bad = out.loc[asset_numeric.isna(), "asset_id"].head(5).tolist()
        raise ValueError(
            "ticker_permno_map.csv contains non-numeric PERMNO values. "
            f"Examples: {bad}"
        )
    out["asset_id"] = asset_numeric.astype("Int64")

    bad_windows = out["mapping_start_date"] > out["mapping_end_date"]
    if bad_windows.any():
        sample = out.loc[
            bad_windows,
            ["ticker", "asset_id", "mapping_start_date", "mapping_end_date"],
        ].head(5)
        raise ValueError(
            "ticker_permno_map.csv contains rows where DATE > NAMEENDDT/NAMEENDT. "
            f"Examples:\n{sample}"
        )

    return out.sort_values(
        ["ticker", "mapping_start_date", "mapping_end_date", "asset_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def load_returns_input_panel(
    csv_path: PathLike,
    *,
    date_col: str = "date",
    asset_col: str = "asset_id",
    return_col: str = "return",
    encoding: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """
    Load returns.csv in one of the supported forms and infer the identifier type.

    Supported forms:
    - ticker, date, return
    - asset_id, date, return (permno or ticker inferred from values)
    - permno, yyyymm, return
    """

    available_cols = peek_csv_columns(csv_path, encoding=encoding)
    actual_date_col = _resolve_column_name(available_cols, [date_col, "yyyymm"])
    actual_asset_col = _resolve_column_name(available_cols, [asset_col, "permno", "ticker"])
    actual_return_col = detect_return_column(available_cols)

    if actual_date_col is None or actual_asset_col is None or actual_return_col is None:
        raise ValueError(
            "returns CSV must contain an identifier column (asset_id/permno/ticker), "
            "a date column (date/yyyymm), and a return column "
            "(return/ret/target_return/future_return). "
            f"Found columns: {available_cols}"
        )

    df = pd.read_csv(
        csv_path,
        usecols=[actual_date_col, actual_asset_col, actual_return_col],
        encoding=encoding,
        low_memory=False,
    ).copy()
    df = df.rename(
        columns={
            actual_date_col: "raw_date",
            actual_asset_col: "raw_identifier",
            actual_return_col: "return",
        }
    )
    df.insert(0, "source_row_number", np.arange(len(df), dtype=np.int64) + 2)

    if actual_date_col.lower() == "yyyymm":
        df["date"] = parse_yyyymm_to_date(df["raw_date"], month_position="end")
    else:
        df["date"] = _parse_direct_dates(df["raw_date"], name=f"returns.{actual_date_col}")

    df["return"] = pd.to_numeric(df["return"], errors="coerce")
    bad_return = df["return"].isna()
    if bad_return.any():
        bad = df.loc[bad_return, "return"].head(5).tolist()
        raise ValueError(f"returns CSV contains non-numeric return values. Examples: {bad}")

    if actual_asset_col.lower() == "permno":
        identifier_kind = "permno"
    elif actual_asset_col.lower() == "ticker":
        identifier_kind = "ticker"
    else:
        identifier_kind = _detect_identifier_kind(df["raw_identifier"])

    if identifier_kind == "permno":
        asset_numeric = pd.to_numeric(df["raw_identifier"], errors="coerce")
        if asset_numeric.isna().any():
            bad = df.loc[asset_numeric.isna(), "raw_identifier"].head(5).tolist()
            raise ValueError(
                "returns CSV was inferred as permno-based but contains non-numeric ids. "
                f"Examples: {bad}"
            )
        out = df[["source_row_number", "date", "return"]].copy()
        out["asset_id"] = asset_numeric.astype("Int64")
        return out[["source_row_number", "date", "asset_id", "return"]], identifier_kind

    out = df[["source_row_number", "date", "return"]].copy()
    out["ticker"] = _standardize_ticker(df["raw_identifier"])
    return out[["source_row_number", "date", "ticker", "return"]], identifier_kind


def map_ticker_returns_to_permno(
    returns_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Map ticker-based returns to permno using a date-aware validity window join."""

    _validate_required_columns(
        returns_df, ["source_row_number", "date", "ticker", "return"], name="ticker returns"
    )
    _validate_required_columns(
        mapping_df,
        ["mapping_row_number", "ticker", "asset_id", "mapping_start_date", "mapping_end_date"],
        name="ticker_permno_map",
    )

    base = returns_df.copy()
    base["ticker"] = _standardize_ticker(base["ticker"])
    relevant_mapping = mapping_df[mapping_df["ticker"].notna()].copy()

    candidates = base.merge(
        relevant_mapping,
        on="ticker",
        how="left",
        suffixes=("_return", "_map"),
    )
    valid = candidates[
        candidates["mapping_start_date"].notna()
        & (candidates["mapping_start_date"] <= candidates["date"])
        & (candidates["date"] <= candidates["mapping_end_date"])
    ].copy()

    match_counts = valid.groupby("source_row_number").size()
    matched_ids = set(match_counts[match_counts == 1].index.tolist())
    ambiguous_ids = set(match_counts[match_counts > 1].index.tolist())
    matched_or_ambiguous_ids = set(match_counts.index.tolist())
    unmatched = base[~base["source_row_number"].isin(matched_or_ambiguous_ids)].copy()
    ambiguous = valid[valid["source_row_number"].isin(ambiguous_ids)].copy()

    mapped = valid[valid["source_row_number"].isin(matched_ids)].copy()
    mapped = mapped[["source_row_number", "date", "asset_id", "return"]].copy()

    ambiguous_examples: list[dict[str, object]] = []
    for row_id, group in ambiguous.groupby("source_row_number", sort=False):
        first = group.iloc[0]
        ambiguous_examples.append(
            {
                "source_row_number": int(row_id),
                "ticker": first["ticker"],
                "date": pd.Timestamp(first["date"]).strftime("%Y-%m-%d"),
                "candidate_permnos": [
                    int(v) for v in group["asset_id"].astype("Int64").dropna().tolist()
                ],
                "candidate_windows": [
                    (
                        pd.Timestamp(r.mapping_start_date).strftime("%Y-%m-%d"),
                        pd.Timestamp(r.mapping_end_date).strftime("%Y-%m-%d"),
                    )
                    for r in group.itertuples(index=False)
                ],
            }
        )
        if len(ambiguous_examples) >= 5:
            break

    summary = {
        "n_mapped_rows": int(len(mapped)),
        "n_unmatched_rows": int(len(unmatched)),
        "n_ambiguous_rows": int(len(ambiguous_ids)),
        "unmatched_examples": unmatched[["source_row_number", "ticker", "date"]]
        .head(5)
        .assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d"))
        .to_dict("records"),
        "ambiguous_examples": ambiguous_examples,
    }
    return mapped.reset_index(drop=True), summary


def resolve_returns_to_canonical_permno(
    returns_csv: PathLike,
    *,
    ticker_permno_map_csv: Optional[PathLike] = None,
    date_col: str = "date",
    asset_col: str = "asset_id",
    return_col: str = "return",
    encoding: Optional[str] = None,
    strict: bool = True,
) -> ResolvedReturnsResult:
    """
    Return a permno-aligned returns panel with columns [date, asset_id, return].

    If returns are ticker-based, a historical ticker_permno_map.csv is required.
    """

    raw_returns, identifier_kind = load_returns_input_panel(
        returns_csv,
        date_col=date_col,
        asset_col=asset_col,
        return_col=return_col,
        encoding=encoding,
    )
    summary = ReturnsResolutionSummary(
        identifier_kind=identifier_kind,
        n_input_rows=int(len(raw_returns)),
        n_mapped_rows=0,
        n_unmatched_rows=0,
        n_ambiguous_rows=0,
        n_duplicate_key_rows=0,
        n_final_rows=0,
        returns_date_samples=_sample_dates(raw_returns["date"]),
    )

    if identifier_kind == "permno":
        final_returns = raw_returns[["date", "asset_id", "return"]].copy()
        summary.n_mapped_rows = int(len(final_returns))
    else:
        if ticker_permno_map_csv is None:
            raise ValueError(
                "returns.csv appears to be ticker-based, so ticker_permno_map.csv is required."
            )
        mapping_df = load_ticker_permno_map(ticker_permno_map_csv, encoding=encoding)
        summary.mapping_start_date_samples = _sample_dates(mapping_df["mapping_start_date"])
        summary.mapping_end_date_samples = _sample_dates(mapping_df["mapping_end_date"])

        mapped_returns, mapping_summary = map_ticker_returns_to_permno(raw_returns, mapping_df)
        summary.n_mapped_rows = int(mapping_summary["n_mapped_rows"])
        summary.n_unmatched_rows = int(mapping_summary["n_unmatched_rows"])
        summary.n_ambiguous_rows = int(mapping_summary["n_ambiguous_rows"])
        summary.unmatched_examples = list(mapping_summary["unmatched_examples"])
        summary.ambiguous_examples = list(mapping_summary["ambiguous_examples"])
        final_returns = mapped_returns[["date", "asset_id", "return"]].copy()

    dup_mask = final_returns.duplicated(subset=["date", "asset_id"], keep=False)
    summary.n_duplicate_key_rows = int(dup_mask.sum())
    summary.n_final_rows = int(len(final_returns))

    if strict and (
        summary.n_unmatched_rows > 0
        or summary.n_ambiguous_rows > 0
        or summary.n_duplicate_key_rows > 0
    ):
        parts: list[str] = []
        if summary.n_unmatched_rows > 0:
            parts.append(
                f"{summary.n_unmatched_rows} unmatched return row(s); examples: {summary.unmatched_examples}"
            )
        if summary.n_ambiguous_rows > 0:
            parts.append(
                f"{summary.n_ambiguous_rows} ambiguous ticker/date mapping row(s); "
                f"examples: {summary.ambiguous_examples}"
            )
        if summary.n_duplicate_key_rows > 0:
            dup_examples = (
                final_returns.loc[dup_mask, ["date", "asset_id"]]
                .head(5)
                .assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d"))
                .to_dict("records")
            )
            parts.append(
                f"{summary.n_duplicate_key_rows} duplicate resolved (date, asset_id) row(s); "
                f"examples: {dup_examples}"
            )
        raise ValueError("Returns-to-permno resolution failed: " + " | ".join(parts))

    final_returns = final_returns.sort_values(["asset_id", "date"], kind="mergesort").reset_index(
        drop=True
    )
    return ResolvedReturnsResult(returns_df=final_returns, summary=summary)
