"""
Performance metrics (Phase 1 placeholder).

This module will later compute performance statistics from `portfolio_df`
produced by the backtest engine.

Do NOT implement metrics in Phase 1.

## Intended inputs
- `portfolio_df` with:
  - date
  - portfolio_return
  - (optional) portfolio_value

## Intended outputs
- A dict-like structure (or DataFrame) with metrics such as:
  - annualized_return, annualized_volatility, sharpe_ratio, max_drawdown
"""

from __future__ import annotations


def compute_metrics(*args, **kwargs):
    """Placeholder metrics calculator."""

    raise NotImplementedError("Phase 1 scaffold: metrics not implemented.")

