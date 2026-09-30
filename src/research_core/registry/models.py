"""Validation rules for R0.7 governed registry records."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from .canonical import verify_freeze_hash


class RegistryValidationError(ValueError):
    """Raised when a registry record violates Protocol v1.2 governance."""


HYPOTHESIS_ID_RE = re.compile(r"^HYP-[A-Z0-9][A-Z0-9_-]{2,63}$")
FAMILY_ID_RE = re.compile(r"^FAM-[A-Z0-9][A-Z0-9_-]{2,63}$")
BATCH_ID_RE = re.compile(r"^CB-[A-Z0-9][A-Z0-9_-]{2,63}$")
BUDGET_ID_RE = re.compile(r"^EB-[A-Z0-9][A-Z0-9_-]{2,63}$")
COHORT_ID_RE = re.compile(r"^(D1|RESERVE)-[A-Z0-9][A-Z0-9_-]{2,63}$")
EVENT_ID_RE = re.compile(r"^HE-[A-Z0-9][A-Z0-9_-]{2,63}$")
ACCESS_ID_RE = re.compile(r"^PA-[A-Z0-9][A-Z0-9_-]{2,63}$")
DECISION_ID_RE = re.compile(r"^DEC-[A-Z0-9][A-Z0-9_-]{2,63}$")

RELATIONSHIP_TYPES = {
    "MARGINAL",
    "ADDITIVE",
    "INTERACTION",
    "TEMPORAL_SEQUENCE",
    "CAUSAL_CHAIN",
    "POSITION",
    "PREDICTIVE_MODEL",
}

STAGES = (
    "IDEA",
    "FEASIBLE",
    "EXPLORATORY_CANDIDATE",
    "REGISTERED_CANDIDATE",
    "REGISTERED_HOLDOUT_SURVIVOR",
    "PROSPECTIVE_SHADOW_CANDIDATE",
    "LIVE_CAPITAL_CANDIDATE",
    "SCALABLE_STRATEGY",
)

TERMINAL_CLASSIFICATIONS = {
    "FAIL_INVALIDATED",
    "FAIL_EVIDENTIALLY_UNSUPPORTED",
    "INDETERMINATE",
    "SCIENTIFIC_ONLY",
    "FEASIBILITY_FAILURE",
    "INVALIDATED",
    "D1_DESIGN_CONTAMINATED",
}

REGISTERED_SPEC_REQUIRED = {
    "market_states",
    "information_clocks",
    "eligibility_rules",
    "decision_clock",
    "latency_model",
    "entry_rule",
    "position_state_machine",
    "research_notional",
    "exit_rule",
    "risk_rules",
    "cost_model",
    "primary_outcome",
    "primary_economic_metric",
    "registered_benchmarks",
    "minimum_economically_meaningful_effect",
    "risk_gates",
    "inference_plan",
    "dependence_episode_definition",
    "evidence_budget_id",
    "minimum_support",
    "independent_support_precision_plan",
    "primary_failure_criterion",
    "d1_feasibility_plan",
    "post_unblinding_defect_rule",
    "code_commit",
    "artifact_hashes",
    "confirmation_plan",
}


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistryValidationError(f"{field} must be a non-empty string")
    return value


def _iso_utc(value: Any, field: str) -> None:
    text = _nonempty_string(value, field)
    if not text.endswith("Z"):
        raise RegistryValidationError(f"{field} must be UTC and end with Z")
    try:
        datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise RegistryValidationError(f"{field} is not ISO-8601 UTC") from exc


def _sha256_string(value: Any, field: str) -> str:
    text = _nonempty_string(value, field)
    if len(text) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in text):
        raise RegistryValidationError(f"{field} must be a 64-character SHA-256 hex string")
    return text


def hypothesis_ref(record: Mapping[str, Any]) -> str:
    return f"{record['hypothesis_id']}@{record['version']}"


def validate_hypothesis_version(record: Mapping[str, Any]) -> None:
    hid = _nonempty_string(record.get("hypothesis_id"), "hypothesis_id")
    if not HYPOTHESIS_ID_RE.fullmatch(hid):
        raise RegistryValidationError("hypothesis_id must match HYP-<stable-id>")
    version = record.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise RegistryValidationError("version must be an integer >= 1")
    family = _nonempty_string(record.get("conceptual_family"), "conceptual_family")
    if not FAMILY_ID_RE.fullmatch(family):
        raise RegistryValidationError("conceptual_family must match FAM-<stable-id>")
    _nonempty_string(record.get("creation_source"), "creation_source")
    _iso_utc(record.get("created_at_utc"), "created_at_utc")
    _nonempty_string(record.get("scientific_proposition"), "scientific_proposition")
    relation = record.get("relationship_type")
    if relation not in RELATIONSHIP_TYPES:
        raise RegistryValidationError(f"unsupported relationship_type: {relation!r}")
    for name in ("parent_hypothesis_ids", "derived_from_results", "datasets_that_influenced_design"):
        value = record.get(name)
        if not isinstance(value, list):
            raise RegistryValidationError(f"{name} must be a list")
    spec = record.get("material_spec")
    if not isinstance(spec, dict):
        raise RegistryValidationError("material_spec must be an object")
    verify_freeze_hash(record)


def validate_registered_spec(record: Mapping[str, Any]) -> None:
    validate_hypothesis_version(record)
    spec = record["material_spec"]
    missing = sorted(k for k in REGISTERED_SPEC_REQUIRED if k not in spec or spec[k] in (None, "", [], {}))
    if missing:
        raise RegistryValidationError(f"registered hypothesis is incomplete; missing: {', '.join(missing)}")
    if not isinstance(spec["registered_benchmarks"], list) or not spec["registered_benchmarks"]:
        raise RegistryValidationError("registered_benchmarks must contain the strongest relevant comparator(s)")
    risk = spec["risk_gates"]
    if not isinstance(risk, list) or not risk:
        raise RegistryValidationError("registered trading proposition requires at least one mandatory risk gate")
    review = record.get("adversarial_review")
    if not isinstance(review, dict):
        raise RegistryValidationError("fresh-context adversarial_review is required before registration")
    if review.get("fresh_context") is not True or review.get("outcome_blind") is not True:
        raise RegistryValidationError("adversarial review must be fresh-context and D1-outcome-blind")
    if review.get("participated_in_discovery") is not False:
        raise RegistryValidationError("adversarial reviewer must not inherit the discovery/authoring context")
    for field in ("review_id", "reviewer_context_id"):
        _nonempty_string(review.get(field), f"adversarial_review.{field}")
    _sha256_string(review.get("frozen_packet_hash"), "adversarial_review.frozen_packet_hash")
    unresolved = review.get("unresolved_objections", [])
    if not isinstance(unresolved, list):
        raise RegistryValidationError("unresolved_objections must be a list")
    if unresolved and not review.get("objection_dispositions"):
        raise RegistryValidationError("unresolved reviewer objections require explicit dispositions before D1 spending")

    feasibility = spec["d1_feasibility_plan"]
    if not isinstance(feasibility, dict):
        raise RegistryValidationError("d1_feasibility_plan must be structured")
    if not isinstance(feasibility.get("permitted_gate_outputs"), list) or not feasibility["permitted_gate_outputs"]:
        raise RegistryValidationError("d1_feasibility_plan must freeze permitted_gate_outputs")
    if feasibility.get("same_cohort_redesign_forbidden") is not True:
        raise RegistryValidationError("D1-informed redesign must be forbidden from confirming on the same cohort")
    if "bounded_fallback_tree" not in feasibility:
        raise RegistryValidationError("d1_feasibility_plan must freeze a bounded_fallback_tree, even when empty")

    dependence = spec["dependence_episode_definition"]
    if not isinstance(dependence, dict) or dependence.get("deterministic") is not True:
        raise RegistryValidationError("dependence_episode_definition must be structured and deterministic")
    if not dependence.get("primary_inference_unit") or not dependence.get("governing_conservative_rule"):
        raise RegistryValidationError("dependence plan must freeze the primary unit and conservative governing rule")

    support = spec["independent_support_precision_plan"]
    if not isinstance(support, dict) or not support.get("do_not_spend_threshold"):
        raise RegistryValidationError("independent support/precision plan must include a do-not-spend threshold")

    confirmation = spec["confirmation_plan"]
    if not isinstance(confirmation, dict):
        raise RegistryValidationError("confirmation_plan must be structured")
    for field in ("confirmation_batch_id", "protected_cohort_id", "protocol_version", "immutable_d1_data_version", "unblinding_rule"):
        _nonempty_string(confirmation.get(field), f"confirmation_plan.{field}")
    if confirmation.get("protocol_version") != "1.2":
        raise RegistryValidationError("confirmation_plan must freeze Protocol v1.2")
    hashes = spec["artifact_hashes"]
    if not isinstance(hashes, dict) or not hashes:
        raise RegistryValidationError("artifact_hashes must be a non-empty mapping")
    for name, digest in hashes.items():
        _sha256_string(digest, f"artifact_hashes.{name}")


def validate_material_change(previous: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Prevent same-version mutation and require monotone versions for the same hypothesis ID."""
    if previous.get("hypothesis_id") != current.get("hypothesis_id"):
        return
    old_v = previous.get("version")
    new_v = current.get("version")
    old_hash = previous.get("freeze_hash")
    new_hash = current.get("freeze_hash")
    if new_v == old_v and new_hash != old_hash:
        raise RegistryValidationError("material mutation of an existing hypothesis version is forbidden")
    if isinstance(old_v, int) and isinstance(new_v, int) and new_v < old_v:
        raise RegistryValidationError("hypothesis version cannot move backwards")


def validate_protected_cohort(record: Mapping[str, Any]) -> None:
    cohort_id = _nonempty_string(record.get("cohort_id"), "cohort_id")
    if not COHORT_ID_RE.fullmatch(cohort_id):
        raise RegistryValidationError("cohort_id must be an immutable D1-... or RESERVE-... identifier")
    if record.get("role") not in {"CONFIRMATION", "RESERVE"}:
        raise RegistryValidationError("protected cohort role must be CONFIRMATION or RESERVE")
    if record.get("status") not in {"FROZEN", "SEALED"}:
        raise RegistryValidationError("protected cohort must be FROZEN/SEALED")
    for field in ("start_date_utc", "end_date_utc", "data_version", "canonical_builder_commit", "schema_version", "timezone_contract", "unit_contract", "intended_sequence", "quarantine_status"):
        _nonempty_string(record.get(field), field)
    _iso_utc(record.get("freeze_ts_utc"), "freeze_ts_utc")
    hashes = record.get("source_manifest_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise RegistryValidationError("protected cohort requires source_manifest_hashes")
    for name, digest in hashes.items():
        _sha256_string(digest, f"source_manifest_hashes.{name}")
    if not isinstance(record.get("known_coverage_gaps"), list):
        raise RegistryValidationError("known_coverage_gaps must be a list")
    verify_freeze_hash(record)


def validate_confirmation_batch(record: Mapping[str, Any]) -> None:
    bid = _nonempty_string(record.get("confirmation_batch_id"), "confirmation_batch_id")
    if not BATCH_ID_RE.fullmatch(bid):
        raise RegistryValidationError("confirmation_batch_id must match CB-<stable-id>")
    if record.get("protocol_version") != "1.2":
        raise RegistryValidationError("R0.7 confirmation batches must freeze Protocol v1.2")
    for field in ("cohort_id", "cohort_manifest_ref", "data_version", "freeze_ts_utc", "code_commit", "unblinding_rule"):
        if field == "freeze_ts_utc":
            _iso_utc(record.get(field), field)
        else:
            _nonempty_string(record.get(field), field)
    _sha256_string(record.get("protected_period_architecture_hash"), "protected_period_architecture_hash")
    hypotheses = record.get("hypothesis_versions")
    if not isinstance(hypotheses, list) or not hypotheses:
        raise RegistryValidationError("confirmation batch must contain a finite non-empty hypothesis_versions list")
    if len(hypotheses) != len(set(hypotheses)):
        raise RegistryValidationError("confirmation batch contains duplicate hypothesis versions")
    families = record.get("family_ids")
    if not isinstance(families, list) or not families:
        raise RegistryValidationError("confirmation batch must freeze hypothesis family IDs")
    budgets = record.get("evidence_budget_ids")
    if not isinstance(budgets, list) or not budgets:
        raise RegistryValidationError("confirmation batch must freeze at least one evidence budget")
    roster = record.get("outcome_access_roster")
    if not isinstance(roster, list) or not roster:
        raise RegistryValidationError("confirmation batch must freeze a non-empty outcome_access_roster")
    hashes = record.get("artifact_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise RegistryValidationError("confirmation batch must freeze artifact hashes")
    for name, digest in hashes.items():
        _sha256_string(digest, f"artifact_hashes.{name}")
    if record.get("status") != "FROZEN":
        raise RegistryValidationError("confirmation batch must be FROZEN before D1 unblinding")
    verify_freeze_hash(record)


def validate_event_history(events: Sequence[Mapping[str, Any]]) -> None:
    """Validate monotone lifecycle state and non-erasure of terminal outcomes."""
    current: dict[str, str] = {}
    seen_event_ids: set[str] = set()
    for event in events:
        event_id = _nonempty_string(event.get("event_id"), "event_id")
        if not EVENT_ID_RE.fullmatch(event_id) or event_id in seen_event_ids:
            raise RegistryValidationError("hypothesis lifecycle event_id must be unique and match HE-<stable-id>")
        seen_event_ids.add(event_id)
        _iso_utc(event.get("event_ts_utc"), "event_ts_utc")
        _nonempty_string(event.get("rationale"), "rationale")
        ref = _nonempty_string(event.get("hypothesis_version"), "hypothesis_version")
        event_type = _nonempty_string(event.get("event_type"), "event_type")
        if event_type == "PROMOTE":
            target = event.get("to_stage")
            if target not in STAGES:
                raise RegistryValidationError(f"unknown target stage: {target!r}")
            prior = current.get(ref)
            if prior in TERMINAL_CLASSIFICATIONS:
                raise RegistryValidationError("a terminal version cannot be rescued in place; create a new version")
            if prior in STAGES:
                if STAGES.index(target) != STAGES.index(prior) + 1:
                    raise RegistryValidationError("promotion must advance exactly one governed stage")
            elif target != "IDEA":
                raise RegistryValidationError("first lifecycle event must establish IDEA")
            current[ref] = target
        elif event_type == "CLASSIFY":
            if ref not in current:
                raise RegistryValidationError("first lifecycle event must establish IDEA before classification")
            classification = event.get("classification")
            if classification not in TERMINAL_CLASSIFICATIONS:
                raise RegistryValidationError(f"unknown terminal classification: {classification!r}")
            current[ref] = classification
        else:
            raise RegistryValidationError(f"unknown event_type: {event_type!r}")



def validate_access_event_history(events: Sequence[Mapping[str, Any]]) -> None:
    order = {"CLEAN": 0, "DESIGN_CONTAMINATED": 1, "OUTCOME_CONTAMINATED": 2}
    current: dict[tuple[str, str, str], str] = {}
    seen: set[str] = set()
    for event in events:
        event_id = _nonempty_string(event.get("access_event_id"), "access_event_id")
        if not ACCESS_ID_RE.fullmatch(event_id) or event_id in seen:
            raise RegistryValidationError("access_event_id must be unique and match PA-<stable-id>")
        seen.add(event_id)
        _iso_utc(event.get("event_ts_utc"), "event_ts_utc")
        actor = _nonempty_string(event.get("actor_id"), "actor_id")
        role = _nonempty_string(event.get("role"), "role")
        cohort = _nonempty_string(event.get("cohort_id"), "cohort_id")
        _nonempty_string(event.get("content_class"), "content_class")
        _nonempty_string(event.get("purpose"), "purpose")
        before = event.get("contamination_before")
        after = event.get("contamination_after")
        if before not in order or after not in order:
            raise RegistryValidationError("access event has unknown contamination state")
        key = (actor, role, cohort)
        expected_before = current.get(key, "CLEAN")
        if before != expected_before:
            raise RegistryValidationError("access-event contamination_before does not match prior ledger state")
        if order[after] < order[before]:
            raise RegistryValidationError("protected-access contamination cannot be downgraded")
        if event.get("allowed") is not True and after != before:
            raise RegistryValidationError("denied access must not change contamination state")
        current[key] = after


def validate_decision_log(records: Sequence[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for record in records:
        decision_id = _nonempty_string(record.get("decision_id"), "decision_id")
        if not DECISION_ID_RE.fullmatch(decision_id) or decision_id in seen:
            raise RegistryValidationError("decision_id must be unique and match DEC-<stable-id>")
        seen.add(decision_id)
        _iso_utc(record.get("decision_ts_utc"), "decision_ts_utc")
        _nonempty_string(record.get("decision"), "decision")
        _nonempty_string(record.get("rationale"), "rationale")
        if not isinstance(record.get("relevant_outcomes_seen"), bool):
            raise RegistryValidationError("decision log must explicitly state relevant_outcomes_seen as boolean")
        if not isinstance(record.get("datasets_considered"), list):
            raise RegistryValidationError("decision log datasets_considered must be a list")

def validate_genealogy(records: Sequence[Mapping[str, Any]]) -> None:
    """Require disclosed parent IDs to exist and reject ancestry cycles."""
    ids = {str(r["hypothesis_id"]) for r in records}
    graph: dict[str, set[str]] = {hid: set() for hid in ids}
    for r in records:
        hid = str(r["hypothesis_id"])
        for parent in r.get("parent_hypothesis_ids", []):
            if parent not in ids:
                raise RegistryValidationError(f"unknown parent_hypothesis_id: {parent}")
            graph[hid].add(parent)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise RegistryValidationError("hypothesis genealogy contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for parent in graph[node]:
            visit(parent)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)
