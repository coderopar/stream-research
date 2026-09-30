"""Executable Protocol v1.2 guardrails for data access and research design."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import RegistryValidationError

ROLES = {
    "THEORY_LITERATURE_AGENT",
    "DATA_STRUCTURE_AGENT",
    "DISCOVERY_AGENT",
    "ADVERSARIAL_REVIEWER",
    "FEASIBILITY_AGENT",
    "PROTOCOL_LIBRARIAN",
    "CONFIRMATION_RUNNER",
    "EXECUTION_RISK_AGENT",
    "LIVE_MONITOR",
}

PROTECTED_ZONES = {"P1_PROTECTED", "D1"}
PROTECTED_CONTENT = {
    "INTEGRITY_METADATA",
    "CAUSAL_FIELDS_FOR_REGISTERED_GATE",
    "DETAILED_FEASIBILITY_DIAGNOSTICS",
    "FUTURE_OUTCOMES",
    "PERFORMANCE_SUMMARY",
}

CONTAMINATION_ORDER = {"CLEAN": 0, "DESIGN_CONTAMINATED": 1, "OUTCOME_CONTAMINATED": 2}


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    reason: str
    contamination_after: str


def _deny(reason: str, contamination: str) -> AccessDecision:
    return AccessDecision(False, reason, contamination)


def authorize_protected_access(
    *,
    role: str,
    zone: str,
    content_class: str,
    current_contamination: str = "CLEAN",
    registered_gate: bool = False,
    batch_frozen: bool = False,
    outcome_access_authorized: bool = False,
    clean_confirmation_context: bool = False,
) -> AccessDecision:
    if role not in ROLES:
        raise RegistryValidationError(f"unknown research role: {role}")
    if zone not in PROTECTED_ZONES:
        raise RegistryValidationError("authorize_protected_access is only for P1_PROTECTED/D1")
    if content_class not in PROTECTED_CONTENT:
        raise RegistryValidationError(f"unknown protected content class: {content_class}")
    if current_contamination not in CONTAMINATION_ORDER:
        raise RegistryValidationError("unknown contamination state")

    if content_class == "INTEGRITY_METADATA":
        if role in {"DATA_STRUCTURE_AGENT", "FEASIBILITY_AGENT", "PROTOCOL_LIBRARIAN", "CONFIRMATION_RUNNER"}:
            return AccessDecision(True, "minimum integrity metadata permitted", current_contamination)
        return _deny("role is not authorized for protected integrity metadata", current_contamination)

    if content_class in {"CAUSAL_FIELDS_FOR_REGISTERED_GATE", "DETAILED_FEASIBILITY_DIAGNOSTICS"}:
        if role != "FEASIBILITY_AGENT":
            return _deny("protected causal feasibility access is limited to the Feasibility Agent", current_contamination)
        if not registered_gate:
            return _deny("D1/P1 causal fields require a pre-registered feasibility gate", current_contamination)
        if content_class == "DETAILED_FEASIBILITY_DIAGNOSTICS":
            return AccessDecision(True, "detailed feasibility access allowed but contaminates design", "DESIGN_CONTAMINATED")
        return AccessDecision(True, "registered feasibility-gate causal access permitted", current_contamination)

    # Outcomes/performance are never available to P1 before it becomes an actual frozen D1 cohort.
    if zone == "P1_PROTECTED":
        return _deny("P1_PROTECTED outcomes remain sealed before immutable D1 cohort freeze", current_contamination)

    if role != "CONFIRMATION_RUNNER":
        return _deny("protected performance access is limited to the frozen Confirmation Runner", current_contamination)
    if current_contamination != "CLEAN":
        return _deny("first protected unblinding requires an uncontaminated confirmation role/context", current_contamination)
    if not batch_frozen or not outcome_access_authorized or not clean_confirmation_context:
        return _deny("D1 outcome access requires frozen batch, explicit authorization, and clean-room context", current_contamination)
    return AccessDecision(True, "registered one-use D1 outcome access permitted", "OUTCOME_CONTAMINATED")


def merge_contamination(current: str, new: str) -> str:
    if current not in CONTAMINATION_ORDER or new not in CONTAMINATION_ORDER:
        raise RegistryValidationError("unknown contamination state")
    return current if CONTAMINATION_ORDER[current] >= CONTAMINATION_ORDER[new] else new


def assert_blind_design_eligible(contamination: str) -> None:
    if contamination != "CLEAN":
        raise RegistryValidationError("contaminated role cannot return to blind design/feasibility/adjudication for the same cohort")


_GOVERNANCE_TOKENS = (
    "overlap_component",
    "episode_id",
    "cluster_id",
    "lineage_id",
    "confirmation_batch",
    "evidence_budget",
    "freeze_hash",
    "outcome_access",
    "contamination",
)
_FUTURE_TOKENS = (
    "future_",
    "forward_return",
    "outcome_",
    "mfe",
    "mae",
    "p35",
    "p50",
    "p75",
    "p100",
    "realized_pnl",
    "future_path",
)


def assert_predictor_columns_safe(columns: Iterable[str]) -> None:
    for column in columns:
        low = str(column).lower()
        if any(token in low for token in _GOVERNANCE_TOKENS):
            raise RegistryValidationError(f"governance metadata is forbidden as a predictor: {column}")
        if any(token in low for token in _FUTURE_TOKENS):
            raise RegistryValidationError(f"future/outcome-derived predictor is forbidden: {column}")


def assert_eligibility_fields_safe(fields: Iterable[str]) -> None:
    for field in fields:
        low = str(field).lower()
        if any(token in low for token in _FUTURE_TOKENS):
            raise RegistryValidationError(f"candidate/trading eligibility cannot depend on future information: {field}")


def assert_time_respecting_validation(scheme: str) -> None:
    normalized = scheme.strip().upper().replace("-", "_").replace(" ", "_")
    forbidden = {"RANDOM", "RANDOM_SPLIT", "KFOLD", "K_FOLD", "STRATIFIED_KFOLD", "SHUFFLE_SPLIT"}
    if normalized in forbidden:
        raise RegistryValidationError("random/ordinary cross-validation is forbidden for dependent market data")
    allowed = {"PURGED_WALK_FORWARD", "WALK_FORWARD_PURGED", "BLOCKED_TIME_SERIES", "PREQUENTIAL"}
    if normalized not in allowed:
        raise RegistryValidationError("validation scheme is not an approved time-respecting design")
