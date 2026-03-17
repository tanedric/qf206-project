"""
Backtest execution engine (Phase 1 placeholder).

This module will later:
- Combine predicted returns (μ), covariance estimates (Σ), and optimizers
  to generate weights through time.
- Apply trading rules/rebalance schedule.
- Compute portfolio returns from weights and realized returns.

Do NOT implement backtesting in Phase 1.

## Intended inputs
- `weights_df`: see `src/portfolio/optimizers.py`
- `returns_df`: realized returns panel (date, asset_id, return)

## Intended outputs
- `portfolio_df` time series with at least:
  - **date**: datetime64[ns]
  - **portfolio_return**: float
  - **portfolio_value**: float (optional if you track NAV)
"""

from __future__ import annotations


def run_backtest(*args, **kwargs):
    """Placeholder backtest runner."""

    raise NotImplementedError("Phase 1 scaffold: backtest engine not implemented.")

