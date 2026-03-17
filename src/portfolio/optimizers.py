"""
Portfolio optimizers (Phase 1 placeholder).

This module will eventually implement:
- Global Minimum Variance Portfolio (GMVP)
- Maximum Sharpe Ratio Portfolio (MSRP)

Do NOT implement optimizers in Phase 1.

## Intended inputs
- `pred_df` (expected returns μ): see `src/models/baseline_model.py`
- Σ (covariance): see `src/portfolio/risk_model.py`
- Optional constraints/metadata (e.g., leverage, long-only, sector caps)

## Intended output
Portfolio weights as a pandas.DataFrame `weights_df` in long format:
- **date**: datetime64[ns]
- **asset_id**: str/int
- **weight**: float

Key invariants:
- Weights sum to 1 per date (unless you explicitly allow leverage later).
- Weights align to the same asset universe as μ and Σ for that date.
"""

from __future__ import annotations


def solve_gmvp(*args, **kwargs):
    """Placeholder GMVP optimizer."""

    raise NotImplementedError("Phase 1 scaffold: GMVP optimizer not implemented.")


def solve_msrp(*args, **kwargs):
    """Placeholder MSRP optimizer."""

    raise NotImplementedError("Phase 1 scaffold: MSRP optimizer not implemented.")

