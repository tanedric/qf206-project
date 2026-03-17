"""
Data loading utilities (Phase 2 implementation).

This module loads two CSVs:
- firm characteristics (features)
- realized returns

It performs only minimal, safe parsing and validation.

## Data contract: Raw firm characteristics input (wide or long allowed)

Expected as a pandas.DataFrame named `raw_chars_df`.

### Option A (recommended): Long / panel format
- **index**: RangeIndex (or any) is acceptable; do not rely on positional order.
- **required columns**:
  - **date**: datetime64[ns] (or string parseable to datetime); observation date
  - **asset_id**: str/int; stable identifier (e.g., PERMNO, ticker+exchange)
  - **feature_001 ... feature_094**: float; firm characteristics at `date`
  - **return_next** (optional at this stage): float; realized next-period return
- **optional columns**:
  - **industry**: str/int (e.g., GICS); used later for constraints/analysis
  - **market_cap**: float; used later for value-weighting/filters

Shape guideline: one row per (date, asset_id).

### Option B: Wide format (less flexible)
- **index**: MultiIndex [date, asset_id] or columns containing those fields.
- **columns**: 94 feature columns + (optional) return column(s).

### Key invariants
- Exactly **94** firm-characteristic feature columns are expected downstream.
- Missing values are allowed in raw input; preprocessing will define rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import pandas as pd

PathLike = Union[str, Path]


def _read_csv(
    csv_path: PathLike,
    *,
    encoding: Optional[str] = None,
    low_memory: bool = False,
) -> pd.DataFrame:
    """Read a CSV with predictable defaults."""

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, encoding=encoding, low_memory=low_memory)


def _parse_date_column(
    df: pd.DataFrame,
    date_col: str,
    *,
    dayfirst: bool = False,
) -> pd.DataFrame:
    """Parse `date_col` into pandas datetime64[ns] with clear failure messages."""

    if date_col not in df.columns:
        raise ValueError(f"Missing required column '{date_col}'. Found: {list(df.columns)}")

    parsed = pd.to_datetime(df[date_col], errors="coerce", dayfirst=dayfirst, utc=False)
    n_bad = int(parsed.isna().sum())
    if n_bad > 0:
        bad_examples = df.loc[parsed.isna(), date_col].head(5).tolist()
        raise ValueError(
            f"Failed to parse {n_bad} values in '{date_col}' as dates. "
            f"Examples: {bad_examples}"
        )

    out = df.copy()
    out[date_col] = parsed
    return out


def _validate_required_columns(df: pd.DataFrame, required: Iterable[str], *, name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required column(s): {missing}. Found: {list(df.columns)}")


def load_firm_characteristics_csv(
    csv_path: PathLike,
    *,
    date_col: str = "date",
    asset_col: str = "asset_id",
    feature_cols: Optional[Sequence[str]] = None,
    dayfirst: bool = False,
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load firm characteristics CSV.

    Minimal expectations:
    - Has columns `date_col` and `asset_col`
    - Has one or more feature columns (either provided via `feature_cols` or inferred)

    Returns
    - DataFrame with at least: [date_col, asset_col, <feature columns...>]
    """

    df = _read_csv(csv_path, encoding=encoding)
    df = _parse_date_column(df, date_col, dayfirst=dayfirst)
    _validate_required_columns(df, [date_col, asset_col], name="firm characteristics CSV")

    if feature_cols is None:
        inferred = [c for c in df.columns if c not in {date_col, asset_col}]
        if not inferred:
            raise ValueError(
                "firm characteristics CSV: could not infer any feature columns. "
                f"Expected at least one non-key column besides '{date_col}' and '{asset_col}'."
            )
        feature_cols = inferred
    else:
        _validate_required_columns(df, feature_cols, name="firm characteristics CSV (feature_cols)")

    return df[[date_col, asset_col, *list(feature_cols)]].copy()


def load_returns_csv(
    csv_path: PathLike,
    *,
    date_col: str = "date",
    asset_col: str = "asset_id",
    return_col: str = "return",
    dayfirst: bool = False,
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load realized returns CSV.

    Minimal expectations:
    - Has columns: date, asset_id, return

    Returns
    - DataFrame with columns: [date_col, asset_col, return_col]
    """

    df = _read_csv(csv_path, encoding=encoding)
    df = _parse_date_column(df, date_col, dayfirst=dayfirst)
    _validate_required_columns(df, [date_col, asset_col, return_col], name="returns CSV")
    return df[[date_col, asset_col, return_col]].copy()


def load_raw_firm_characteristics(*args, **kwargs):
    """Backward-compatible alias for earlier scaffold (prefer `load_firm_characteristics_csv`)."""

    return load_firm_characteristics_csv(*args, **kwargs)

