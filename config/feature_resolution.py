"""
Resolve which OAP / wide column names to use given a feature-set option and
available dataframe columns (case-sensitive match to CSV headers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AbstractSet, Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd

from config.paper_oap_mapping import (
    CANONICAL_PAPER_94_FEATURES,
    DEFAULT_FEATURE_SET,
    FEATURE_SET_OPTIONS,
    OAP_EXTRA_FEATURES,
    PAPER_94_PROXY_MAP,
    PAPER_94_TO_OAP209_MAP,
)

KEY_LIKE_COLS = {"date", "asset_id", "permno", "yyyymm", "return", "ret", "target_return"}


@dataclass
class FeatureResolutionSummary:
    """Human-readable summary of how paper-94 concepts mapped to OAP columns."""

    feature_set: str
    selected_oap_columns: List[str]
    exact_matches: List[tuple[str, str]] = field(default_factory=list)  # (canonical, oap)
    proxy_matches: List[tuple[str, str]] = field(default_factory=list)
    missing_canonical: List[str] = field(default_factory=list)
    extras_added: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "feature_set": self.feature_set,
            "n_selected": len(self.selected_oap_columns),
            "n_exact": len(self.exact_matches),
            "n_proxy": len(self.proxy_matches),
            "n_missing_canonical": len(self.missing_canonical),
            "n_extras": len(self.extras_added),
            "exact_matches": self.exact_matches,
            "proxy_matches": self.proxy_matches,
            "missing_canonical": self.missing_canonical,
            "extras_added": self.extras_added,
        }


def _first_available(cols: AbstractSet[str], candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def list_feature_candidates_from_column_names(
    column_names: Iterable[str],
    *,
    date_col: str,
    asset_col: str,
) -> List[str]:
    """
    Return non-key, non-target columns using header names only.

    This is intentionally header-based rather than dtype-based so large CSVs can be
    inspected without loading the full file into memory first.
    """
    skip = {date_col, asset_col, "return", "ret", "target_return"}
    out: List[str] = []
    for col in column_names:
        name = str(col)
        if name in skip or name in KEY_LIKE_COLS:
            continue
        out.append(name)
    return out


def resolve_oap_columns_for_paper94(
    available_columns: AbstractSet[str],
    *,
    mode: str,
) -> FeatureResolutionSummary:
    """
    Map canonical paper-94 names to OAP column names present in the file.

    mode:
    - paper_94_exact_only: use PAPER_94_TO_OAP209_MAP only when that column exists
    - paper_94_exact_plus_proxy: try exact then proxies
    - paper_94_plus_oap_extras: exact+proxy union with OAP_EXTRA_FEATURES present in file
    """
    if mode not in FEATURE_SET_OPTIONS:
        raise ValueError(f"Unknown feature set '{mode}'. Expected one of {FEATURE_SET_OPTIONS}.")

    selected: List[str] = []
    seen: Set[str] = set()
    exact_matches: List[tuple[str, str]] = []
    proxy_matches: List[tuple[str, str]] = []
    missing: List[str] = []

    for canon in CANONICAL_PAPER_94_FEATURES:
        pref = PAPER_94_TO_OAP209_MAP.get(canon)
        proxies = PAPER_94_PROXY_MAP.get(canon, [])
        chosen: Optional[str] = None
        kind: Optional[str] = None

        if mode == "paper_94_exact_only":
            if pref is not None and pref in available_columns:
                chosen, kind = pref, "exact"
        elif mode in ("paper_94_exact_plus_proxy", "paper_94_plus_oap_extras"):
            if pref is not None and pref in available_columns:
                chosen, kind = pref, "exact"
            else:
                alt = _first_available(available_columns, proxies)
                if alt is not None:
                    chosen, kind = alt, "proxy"

        if chosen is None:
            missing.append(canon)
            continue
        if chosen not in seen:
            selected.append(chosen)
            seen.add(chosen)
        if kind == "exact":
            exact_matches.append((canon, chosen))
        elif kind == "proxy":
            proxy_matches.append((canon, chosen))

    extras_added: List[str] = []
    if mode == "paper_94_plus_oap_extras":
        for c in OAP_EXTRA_FEATURES:
            if c in available_columns and c not in seen:
                selected.append(c)
                seen.add(c)
                extras_added.append(c)

    return FeatureResolutionSummary(
        feature_set=mode,
        selected_oap_columns=selected,
        exact_matches=exact_matches,
        proxy_matches=proxy_matches,
        missing_canonical=missing,
        extras_added=extras_added,
    )


def list_numeric_feature_candidates(df, *, date_col: str, asset_col: str) -> List[str]:
    """All columns that are numeric-like and not key/target-like."""
    header_candidates = set(
        list_feature_candidates_from_column_names(df.columns, date_col=date_col, asset_col=asset_col)
    )
    out: List[str] = []
    for c in df.columns:
        if c not in header_candidates:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(str(c))
        else:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().sum() > 0:
                out.append(str(c))
    return out


def resolve_feature_columns_from_column_names(
    column_names: Iterable[str],
    *,
    feature_set: str = DEFAULT_FEATURE_SET,
    date_col: str = "date",
    asset_col: str = "asset_id",
) -> FeatureResolutionSummary:
    """
    Resolve a feature set using only header names.

    Useful for very large OAP CSVs where we want to decide `usecols` before reading
    the entire dataset into memory.
    """
    cols = {str(c) for c in column_names}
    if feature_set == "oap_all_numeric_pruned":
        return FeatureResolutionSummary(
            feature_set=feature_set,
            selected_oap_columns=list_feature_candidates_from_column_names(
                cols,
                date_col=date_col,
                asset_col=asset_col,
            ),
            exact_matches=[],
            proxy_matches=[],
            missing_canonical=[],
            extras_added=[],
        )
    return resolve_oap_columns_for_paper94(cols, mode=feature_set)


def resolve_feature_columns(
    df: pd.DataFrame,
    *,
    feature_set: str = DEFAULT_FEATURE_SET,
    date_col: str = "date",
    asset_col: str = "asset_id",
) -> FeatureResolutionSummary:
    """
    Full resolution for all supported feature sets.

    oap_all_numeric_pruned: return all numeric columns except keys (pruning of
    high-missing / constant columns happens later on the training split).
    """
    cols = set(df.columns.astype(str))
    if feature_set == "oap_all_numeric_pruned":
        numeric_cols = list_numeric_feature_candidates(df, date_col=date_col, asset_col=asset_col)
        return FeatureResolutionSummary(
            feature_set=feature_set,
            selected_oap_columns=numeric_cols,
            exact_matches=[],
            proxy_matches=[],
            missing_canonical=[],
            extras_added=[],
        )
    return resolve_oap_columns_for_paper94(cols, mode=feature_set)


def compute_paper94_missing_list(summary: FeatureResolutionSummary) -> List[str]:
    """Explicit list of canonical paper features with no resolved OAP column."""
    return list(summary.missing_canonical)
