"""Evidence-budget and multiplicity guardrails."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .canonical import verify_freeze_hash
from .models import BUDGET_ID_RE, FAMILY_ID_RE, RegistryValidationError

ALLOWED_CONTROL_METHODS = {
    "FWER",
    "FDR",
    "HIERARCHICAL_CLOSED_TESTING",
    "SEQUENTIAL_VALID",
    "CALIBRATED_BAYESIAN_DECISION",
    "SINGLE_PRIMARY_DECISION",
}


def validate_evidence_budget(record: Mapping[str, Any]) -> None:
    budget_id = record.get("evidence_budget_id")
    if not isinstance(budget_id, str) or not BUDGET_ID_RE.fullmatch(budget_id):
        raise RegistryValidationError("evidence_budget_id must match EB-<stable-id>")
    families = record.get("family_ids")
    if not isinstance(families, list) or not families:
        raise RegistryValidationError("evidence budget must cover at least one hypothesis family")
    if any(not isinstance(x, str) or not FAMILY_ID_RE.fullmatch(x) for x in families):
        raise RegistryValidationError("evidence budget family_ids must use stable FAM- identifiers")
    primaries = record.get("primary_claim_ids")
    if not isinstance(primaries, list) or not primaries:
        raise RegistryValidationError("evidence budget must contain at least one primary claim")
    if len(primaries) != len(set(primaries)):
        raise RegistryValidationError("duplicate primary claims are forbidden")
    secondaries = record.get("secondary_claim_ids", [])
    diagnostics = record.get("robustness_diagnostic_ids", [])
    if not isinstance(secondaries, list) or not isinstance(diagnostics, list):
        raise RegistryValidationError("secondary_claim_ids and robustness_diagnostic_ids must be lists")
    method = record.get("control_method")
    if method not in ALLOWED_CONTROL_METHODS:
        raise RegistryValidationError("control_method must provide an explicit registered operating guarantee")
    guarantee = record.get("operating_guarantee")
    if not isinstance(guarantee, str) or not guarantee.strip():
        raise RegistryValidationError("operating_guarantee must be explicit")
    if method == "SINGLE_PRIMARY_DECISION" and len(primaries) != 1:
        raise RegistryValidationError("SINGLE_PRIMARY_DECISION is allowed only for one primary claim")
    if method == "SINGLE_PRIMARY_DECISION" and record.get("unallocated_alternative_success_paths"):
        raise RegistryValidationError("single-primary nominal inference cannot hide alternative success paths")
    rationale = record.get("anti_family_splitting_rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise RegistryValidationError("anti_family_splitting_rationale is required")
    support = record.get("expected_independent_support")
    if not isinstance(support, dict) or not support:
        raise RegistryValidationError("expected_independent_support is required before D1 spending")
    if not record.get("do_not_spend_threshold"):
        raise RegistryValidationError("do_not_spend_threshold is required")
    verify_freeze_hash(record)
