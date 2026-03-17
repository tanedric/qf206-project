"""
Centralized project settings for Phase 1 (scaffold + data contracts).

Phase 3: Feature-group configuration for return-prediction ML pipeline.
Feature names align with Table 1 (94 firm characteristics) from the
Forecasting Stock Returns Using Machine Learning research (CBS).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Feature groups (Table 1: 94 firm characteristics)
# Left-side (1–47): built by teammate; placeholder for future integration.
# Right-side (48–94): main active set for ML return prediction.
# ---------------------------------------------------------------------------

LEFT_SIDE_FEATURES: List[str] = [
    "absolute_accruals",
    "working_capital_accruals",
    "abnormal_earnings_announcement_volume",
    "years_since_first_compustat_coverage",
    "asset_growth",
    "bid_ask_spread",
    "beta",
    "beta_squared",
    "book_to_market",
    "industry_adjusted_book_to_market",
    "cash_holdings",
    "cash_flow_to_debt",
    "cash_productivity",
    "cash_flow_to_price_ratio",
    "industry_adjusted_cash_flow_to_price_ratio",
    "industry_adjusted_change_in_asset_turnover",
    "change_in_shares_outstanding",
    "industry_adjusted_change_in_employees",
    "change_in_inventory",
    "change_in_6m_momentum",
    "industry_adjusted_change_in_profit_margin",
    "change_in_tax_expense",
    "corporate_investment",
    "convertible_debt_indicator",
    "current_ratio",
    "depreciation_to_ppe",
    "dividend_initiation",
    "dividend_omission",
    "dollar_trading_volume",
    "dividend_to_price",
    "earnings_announcement_return",
    "growth_in_common_shareholder_equity",
    "earnings_to_price",
    "gross_profitability",
    "growth_in_capital_expenditures",
    "growth_in_long_term_net_operating_assets",
    "industry_sales_concentration",
    "employee_growth_rate",
    "idiosyncratic_return_volatility",
    "illiquidity",
    "industry_momentum",
    "capital_expenditures_and_inventory",
    "leverage",
    "growth_in_long_term_debt",
    "maximum_daily_return",
    "momentum_12m",
    "momentum_1m",
]

RIGHT_SIDE_FEATURES: List[str] = [
    "momentum_36m",
    "momentum_6m",
    "financial_statement_score",
    "size",
    "industry_adjusted_size",
    "number_of_earnings_increases",
    "operating_profitability",
    "organizational_capital",
    "industry_adjusted_pct_change_in_capex",
    "pct_change_in_current_ratio",
    "pct_change_in_depreciation",
    "pct_change_in_gross_margin_minus_pct_change_in_sales",
    "pct_change_in_quick_ratio",
    "pct_change_in_sales_minus_pct_change_in_inventory",
    "pct_change_in_sales_minus_pct_change_in_ar",
    "pct_change_in_sales_minus_pct_change_in_sga",
    "pct_change_sales_to_inventory",
    "percent_accruals",
    "price_delay",
    "financial_statements_score",
    "quick_ratio",
    "rd_increase",
    "rd_to_market_cap",
    "rd_to_sales",
    "real_estate_holdings",
    "return_volatility",
    "return_on_assets",
    "earnings_volatility",
    "return_on_equity",
    "return_on_invested_capital",
    "revenue_surprise",
    "sales_to_cash",
    "sales_to_inventory",
    "sales_to_receivables",
    "secured_debt",
    "secured_debt_indicator",
    "sales_growth",
    "sin_stocks",
    "sales_to_price",
    "volatility_of_liquidity_dollar_trading_volume",
    "volatility_of_liquidity_share_turnover",
    "accrual_volatility",
    "cash_flow_volatility",
    "debt_capacity_firm_tangibility",
    "tax_income_to_book_income",
    "share_turnover",
    "zero_trading_days",
]

ALL_FEATURES: List[str] = LEFT_SIDE_FEATURES + RIGHT_SIDE_FEATURES

# Active feature set for ML return prediction (default: right-side only).
# Change to "left" or "all" when integrating teammate's inputs or full set.
FEATURE_GROUP_CHOICE: str = "right"


def get_active_feature_set() -> List[str]:
    """
    Return the currently active feature list for the return-prediction pipeline.

    - "right" -> RIGHT_SIDE_FEATURES (default)
    - "left"  -> LEFT_SIDE_FEATURES
    - "all"   -> ALL_FEATURES
    """
    if FEATURE_GROUP_CHOICE == "left":
        return list(LEFT_SIDE_FEATURES)
    if FEATURE_GROUP_CHOICE == "all":
        return list(ALL_FEATURES)
    return list(RIGHT_SIDE_FEATURES)


@dataclass(frozen=True)
class Paths:
    """
    Canonical filesystem locations used by the pipeline.

    Adjust these as your team standardizes where data lives.
    """

    project_root: Path
    data_raw_dir: Path
    data_processed_dir: Path
    artifacts_dir: Path


@dataclass(frozen=True)
class Parameters:
    """
    Default parameters for the pipeline.

    Phase 1 note: These are placeholders and can be expanded later.
    """

    n_features: int = 94
    prediction_horizon_periods: int = 1
    random_seed: int = 42


def get_paths() -> Paths:
    """Create path settings relative to the project root."""
    # This file lives in: <root>/config/settings.py
    root = Path(__file__).resolve().parents[1]
    return Paths(
        project_root=root,
        data_raw_dir=root / "data" / "raw",
        data_processed_dir=root / "data" / "processed",
        artifacts_dir=root / "artifacts",
    )


def get_parameters() -> Parameters:
    """Return default pipeline parameters."""
    return Parameters()

