"""Governed research-substrate primitives."""

from .d0 import (
    LEGACY_CANDIDATE_DEFINITION_ID,
    OUTCOME_HORIZONS_SEC,
    OUTCOME_TARGETS_BPS,
    PRIMARY_OBSERVATION_HORIZON_SEC,
    build_scientific_label_clock,
    directional_bps,
    latest_prior_trade,
    summarize_outcome_horizon,
    validate_exact_anchor,
)

__all__ = [
    "LEGACY_CANDIDATE_DEFINITION_ID",
    "OUTCOME_HORIZONS_SEC",
    "OUTCOME_TARGETS_BPS",
    "PRIMARY_OBSERVATION_HORIZON_SEC",
    "build_scientific_label_clock",
    "directional_bps",
    "latest_prior_trade",
    "summarize_outcome_horizon",
    "validate_exact_anchor",
]
