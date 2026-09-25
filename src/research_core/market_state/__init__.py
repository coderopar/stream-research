"""Governed causal market-state primitives."""

from research_core.market_state.asof import AsofSelection, assert_available_by_cutoff, latest_prior_index
from research_core.market_state.guards import assert_feature_columns_safe, future_like_fields, is_governance_field
from research_core.market_state.schemas import (
    FEATURE_ELIGIBLE,
    FUTURE_FORBIDDEN,
    GOVERNANCE_ONLY,
    IDENTITY_ONLY,
    QUALITY_ONLY,
    TRADE_STORAGE_GRID,
)
from research_core.market_state.windows import TradeBucket, relative_bucket_bounds, summarize_trade_bucket

__all__ = [
    "AsofSelection",
    "TradeBucket",
    "FEATURE_ELIGIBLE",
    "FUTURE_FORBIDDEN",
    "GOVERNANCE_ONLY",
    "IDENTITY_ONLY",
    "QUALITY_ONLY",
    "TRADE_STORAGE_GRID",
    "assert_available_by_cutoff",
    "assert_feature_columns_safe",
    "future_like_fields",
    "is_governance_field",
    "latest_prior_index",
    "relative_bucket_bounds",
    "summarize_trade_bucket",
]
