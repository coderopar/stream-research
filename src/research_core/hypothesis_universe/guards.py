"""R1.0 guards that separate idea generation from testing and protected data."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from research_core.registry.models import RegistryValidationError


def assert_no_protected_influence(record: Mapping[str, object]) -> None:
    """Fail closed if an R1 idea/session declares protected P1/D1 influence."""
    sources = record.get("source_refs", [])
    if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)):
        for source in sources:
            if isinstance(source, Mapping):
                zone = str(source.get("source_zone", "")).upper()
                if zone in {"P1", "P1_PROTECTED", "D1", "RESERVE", "PROTECTED"}:
                    raise RegistryValidationError("R1 hypothesis generation cannot use protected P1/D1 sources")
    for key in ("p1_market_data_read", "p1_outcome_read", "protected_performance_read"):
        if key in record and record.get(key) is not False:
            raise RegistryValidationError(f"{key} must remain false during R1 generation")


def assert_generation_not_testing(record: Mapping[str, object]) -> None:
    """Reject fields that turn a universe idea packet into a hidden outcome test."""
    forbidden = {
        "observed_effect",
        "observed_return",
        "p_value",
        "q_value",
        "confidence_interval",
        "backtest_metrics",
        "selected_threshold_from_results",
        "selected_horizon_from_results",
    }
    present = sorted(k for k in forbidden if k in record)
    if present:
        raise RegistryValidationError(
            "R1 universe records cannot embed fresh outcome-test results: " + ", ".join(present)
        )


def assert_testability_separate_from_merit(record: Mapping[str, object]) -> None:
    """Prevent infrastructure availability from masquerading as scientific ranking."""
    infrastructure = record.get("infrastructure_assessment")
    if not isinstance(infrastructure, Mapping):
        raise RegistryValidationError("infrastructure_assessment is required")
    for forbidden in ("scientific_merit_score", "priority_score", "expected_alpha_score"):
        if forbidden in infrastructure:
            raise RegistryValidationError(
                f"{forbidden} is not an infrastructure field; R1.0 forbids using testability as scientific merit"
            )
