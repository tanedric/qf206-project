"""
Risk model / covariance estimation (Phase 1 placeholder).

This module will estimate the covariance matrix Σ used by portfolio
optimizers (GMVP / MSRP).

Do NOT implement covariance estimation in Phase 1.

## Intended input
- Realized returns history as a pandas.DataFrame `returns_df` in long format:
  - **date**: datetime64[ns]
  - **asset_id**: str/int
  - **return**: float (realized return for that period)

## Intended output
- Covariance matrix for each rebalance date, e.g.:
  - A dict-like mapping: {date -> numpy.ndarray (n_assets x n_assets)}
  - Or a pandas.DataFrame with a MultiIndex (asset_id x asset_id)

## Key invariants
- Asset ordering must match the corresponding `pred_df` ordering/universe.
- Σ must be symmetric positive semi-definite (PSD) after estimation/shrinkage.
"""

from __future__ import annotations


def estimate_covariance(*args, **kwargs):
    """Placeholder Σ estimator."""

    raise NotImplementedError("Phase 1 scaffold: covariance estimation not implemented.")

