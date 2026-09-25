"""Leakage and role guards for R0.5 causal state."""

from __future__ import annotations

from collections.abc import Iterable

from research_core.market_state.schemas import (
    FUTURE_FORBIDDEN_TOKENS,
    GOVERNANCE_FIELD_PREFIXES,
)

# ``outcome_start_ts_ns`` is a governed clock in R0.3 but R0.5 causal state is
# cut at information_available_ts_ns, so it is deliberately not admitted here.


def is_governance_field(name: str) -> bool:
    n = str(name)
    return any(n.startswith(prefix) for prefix in GOVERNANCE_FIELD_PREFIXES)


def future_like_fields(columns: Iterable[str]) -> list[str]:
    bad: list[str] = []
    for col in columns:
        name = str(col)
        lc = name.lower()
        if any(token in lc for token in FUTURE_FORBIDDEN_TOKENS):
            bad.append(name)
    return sorted(set(bad))


def assert_feature_columns_safe(columns: Iterable[str]) -> None:
    cols = [str(c) for c in columns]
    future = future_like_fields(cols)
    governance = sorted(c for c in cols if is_governance_field(c))
    if future or governance:
        raise ValueError(
            "R0.5 feature-eligible state contains forbidden fields: "
            f"future={future}, governance={governance}"
        )
