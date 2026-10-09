"""R1.0 hypothesis-universe governance models.

These records sit *before* the permanent R0.7 hypothesis registry.  They make
broad R1 idea generation auditable without pretending that every brainstorm is
already a registered hypothesis or a tested trading proposition.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from research_core.registry import with_freeze_hash, verify_freeze_hash
from research_core.registry.models import RELATIONSHIP_TYPES, RegistryValidationError

IDEA_ID_RE = re.compile(r"^IDEA-R1-[A-Z0-9][A-Z0-9_-]{2,63}$")
FAMILY_ID_RE = re.compile(r"^FAM-[A-Z0-9][A-Z0-9_-]{2,63}$")
SESSION_ID_RE = re.compile(r"^GEN-R1-[A-Z0-9][A-Z0-9_-]{2,63}$")
HYPOTHESIS_ID_RE = re.compile(r"^HYP-[A-Z0-9][A-Z0-9_-]{2,63}$")

GENERATION_MODES = {
    "OUTCOME_BLIND",
    "D0_RESULT_INFORMED",
    "LEGACY_RESULT_INFORMED",
}

SOURCE_TYPES = {
    "THEORY_LITERATURE",
    "MARKET_MECHANISM",
    "USER_PROPOSAL",
    "D0_STRUCTURE",
    "D0_RESULT",
    "LEGACY_LINEAGE",
    "PREDICTIVE_MODEL_DESIGN",
    "ADVERSARIAL_GAP",
}

SOURCE_ZONES = {"NONE", "PUBLIC", "D0", "LEGACY_D0"}

CANDIDATE_DEFINITION_STATUSES = {
    "CONCEPTUAL",
    "LEGACY_FROZEN",
    "TO_BE_DEVELOPED_IN_D0",
}

R2_TESTABILITY = {
    "TESTABLE_NOW",
    "BLOCKED_PENDING_GOVERNED_STATE",
    "BLOCKED_PENDING_EXECUTION_CAPABILITY",
    "BLOCKED_PENDING_STATE_AND_EXECUTION",
    "CONCEPT_ONLY",
}

SESSION_STATUSES = {"FROZEN_PLANNED", "IN_PROGRESS", "COMPLETE"}

DEFAULT_GENERATION_LENSES = (
    "THEORY_LITERATURE",
    "MARKET_MECHANISM",
    "LEGACY_7M_LINEAGE",
    "ORDER_BOOK_LIQUIDITY",
    "PERPS_TRADE_FLOW",
    "DERIVATIVES_POSITIONING",
    "CROSS_VENUE",
    "VOLATILITY_REGIME",
    "PREDICTIVE_MODEL",
)


_FAMILY_FIELDS = {
    "family_id", "family_name", "created_at_utc", "mechanism_class",
    "mechanism_statement", "market_state_domains", "relationship_types",
    "variant_dimensions_kept_in_family", "new_family_boundary_rule",
    "anti_family_splitting_rationale", "legacy_lineage_refs", "freeze_hash",
}
_IDEA_FIELDS = {
    "idea_id", "version", "family_id", "created_at_utc", "title",
    "scientific_proposition", "economic_mechanism", "relationship_type",
    "expected_market_response", "causal_state_requirements", "trigger_concept",
    "candidate_definition_status", "position_concept", "primary_economic_target",
    "risk_concepts", "execution_dependencies", "generation_mode", "source_refs",
    "parent_idea_refs", "parent_hypothesis_ids", "legacy_lineage_refs",
    "derived_from_results", "datasets_that_influenced_design",
    "infrastructure_assessment", "freeze_hash",
}
_SESSION_FIELDS = {
    "session_id", "version", "created_at_utc", "status", "planned_lenses",
    "completed_lenses", "allowed_source_zones", "allowed_generation_modes",
    "p1_market_data_read", "p1_outcome_read", "protected_performance_read",
    "no_outcome_based_pruning_during_generation",
    "retain_failed_duplicate_and_legacy_lineage", "family_splitting_rule",
    "stopping_rule", "freeze_hash",
}


def _reject_unknown_fields(record: Mapping[str, Any], allowed: set[str], kind: str) -> None:
    unknown = sorted(set(record) - allowed)
    if unknown:
        raise RegistryValidationError(f"{kind} contains ungoverned fields: {', '.join(unknown)}")

_FORBIDDEN_RESULT_KEYS = {
    "effect_estimate",
    "p_value",
    "q_value",
    "confidence_interval",
    "ci_low",
    "ci_high",
    "observed_return",
    "observed_win_rate",
    "observed_pnl",
    "observed_mfe",
    "observed_mae",
    "backtest_result",
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


def _string_list(value: Any, field: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
        raise RegistryValidationError(f"{field} must be a list of non-empty strings")
    if nonempty and not value:
        raise RegistryValidationError(f"{field} must not be empty")
    return value


def idea_ref(record: Mapping[str, Any]) -> str:
    return f"{record['idea_id']}@{record['version']}"


def validate_family(record: Mapping[str, Any]) -> None:
    _reject_unknown_fields(record, _FAMILY_FIELDS, "family record")
    family_id = _nonempty_string(record.get("family_id"), "family_id")
    if not FAMILY_ID_RE.fullmatch(family_id):
        raise RegistryValidationError("family_id must match FAM-<stable-id>")
    _nonempty_string(record.get("family_name"), "family_name")
    _iso_utc(record.get("created_at_utc"), "created_at_utc")
    _nonempty_string(record.get("mechanism_class"), "mechanism_class")
    _nonempty_string(record.get("mechanism_statement"), "mechanism_statement")
    _string_list(record.get("market_state_domains"), "market_state_domains", nonempty=True)
    relationships = _string_list(record.get("relationship_types"), "relationship_types", nonempty=True)
    unknown = sorted(set(relationships) - RELATIONSHIP_TYPES)
    if unknown:
        raise RegistryValidationError(f"unsupported family relationship_types: {unknown}")
    _string_list(
        record.get("variant_dimensions_kept_in_family"),
        "variant_dimensions_kept_in_family",
        nonempty=True,
    )
    _nonempty_string(record.get("new_family_boundary_rule"), "new_family_boundary_rule")
    _nonempty_string(record.get("anti_family_splitting_rationale"), "anti_family_splitting_rationale")
    _string_list(record.get("legacy_lineage_refs"), "legacy_lineage_refs")
    verify_freeze_hash(record)


def _validate_source_ref(source: Mapping[str, Any], index: int) -> None:
    _reject_unknown_fields(source, {"source_type", "source_zone", "reference", "result_informed"}, "source record")
    prefix = f"source_refs[{index}]"
    source_type = _nonempty_string(source.get("source_type"), f"{prefix}.source_type")
    if source_type not in SOURCE_TYPES:
        raise RegistryValidationError(f"unsupported {prefix}.source_type: {source_type!r}")
    zone = _nonempty_string(source.get("source_zone"), f"{prefix}.source_zone")
    if zone not in SOURCE_ZONES:
        raise RegistryValidationError(
            f"{prefix}.source_zone must be NONE/PUBLIC/D0/LEGACY_D0; protected P1/D1 sources are forbidden in R1 generation"
        )
    _nonempty_string(source.get("reference"), f"{prefix}.reference")
    if not isinstance(source.get("result_informed"), bool):
        raise RegistryValidationError(f"{prefix}.result_informed must be boolean")
    if source_type == "D0_RESULT" and (zone != "D0" or source.get("result_informed") is not True):
        raise RegistryValidationError("D0_RESULT sources must be D0 and explicitly result_informed")
    if source_type == "D0_STRUCTURE" and source.get("result_informed") is not False:
        raise RegistryValidationError("D0_STRUCTURE is structural/provenance information, not a result-informed source")


def validate_idea(record: Mapping[str, Any]) -> None:
    _reject_unknown_fields(record, _IDEA_FIELDS, "idea record")
    idea_id = _nonempty_string(record.get("idea_id"), "idea_id")
    if not IDEA_ID_RE.fullmatch(idea_id):
        raise RegistryValidationError("idea_id must match IDEA-R1-<stable-id>")
    version = record.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise RegistryValidationError("version must be an integer >= 1")
    family_id = _nonempty_string(record.get("family_id"), "family_id")
    if not FAMILY_ID_RE.fullmatch(family_id):
        raise RegistryValidationError("family_id must match FAM-<stable-id>")
    _iso_utc(record.get("created_at_utc"), "created_at_utc")
    for field in (
        "title",
        "scientific_proposition",
        "economic_mechanism",
        "expected_market_response",
        "trigger_concept",
        "primary_economic_target",
    ):
        _nonempty_string(record.get(field), field)

    relationship = record.get("relationship_type")
    if relationship not in RELATIONSHIP_TYPES:
        raise RegistryValidationError(f"unsupported relationship_type: {relationship!r}")
    if record.get("candidate_definition_status") not in CANDIDATE_DEFINITION_STATUSES:
        raise RegistryValidationError("unknown candidate_definition_status")

    mode = record.get("generation_mode")
    if mode not in GENERATION_MODES:
        raise RegistryValidationError(f"unknown generation_mode: {mode!r}")

    sources = record.get("source_refs")
    if not isinstance(sources, list) or not sources or any(not isinstance(x, Mapping) for x in sources):
        raise RegistryValidationError("source_refs must be a non-empty list of structured source records")
    for i, source in enumerate(sources):
        _validate_source_ref(source, i)

    for field in (
        "parent_idea_refs",
        "parent_hypothesis_ids",
        "legacy_lineage_refs",
        "derived_from_results",
        "datasets_that_influenced_design",
        "risk_concepts",
        "execution_dependencies",
    ):
        _string_list(record.get(field), field)
    for hid in record["parent_hypothesis_ids"]:
        if not HYPOTHESIS_ID_RE.fullmatch(hid):
            raise RegistryValidationError(f"invalid parent_hypothesis_id: {hid}")

    state_reqs = record.get("causal_state_requirements")
    if not isinstance(state_reqs, list) or not state_reqs:
        raise RegistryValidationError("causal_state_requirements must be a non-empty structured list")
    for i, req in enumerate(state_reqs):
        if not isinstance(req, Mapping):
            raise RegistryValidationError("causal_state_requirements entries must be objects")
        _reject_unknown_fields(req, {"state_family", "role", "required"}, "causal-state requirement")
        _nonempty_string(req.get("state_family"), f"causal_state_requirements[{i}].state_family")
        _nonempty_string(req.get("role"), f"causal_state_requirements[{i}].role")
        if not isinstance(req.get("required"), bool):
            raise RegistryValidationError(f"causal_state_requirements[{i}].required must be boolean")

    position = record.get("position_concept")
    if not isinstance(position, Mapping):
        raise RegistryValidationError("position_concept must be structured")
    _reject_unknown_fields(position, {"directionality", "entry_style", "horizon_concept", "exit_concept"}, "position_concept")
    for field in ("directionality", "entry_style", "horizon_concept", "exit_concept"):
        _nonempty_string(position.get(field), f"position_concept.{field}")

    infrastructure = record.get("infrastructure_assessment")
    if not isinstance(infrastructure, Mapping):
        raise RegistryValidationError("infrastructure_assessment must be structured")
    _reject_unknown_fields(infrastructure, {"r2_testability", "available_capabilities", "missing_capabilities", "assessment_basis"}, "infrastructure_assessment")
    if infrastructure.get("r2_testability") not in R2_TESTABILITY:
        raise RegistryValidationError("unknown infrastructure_assessment.r2_testability")
    _string_list(infrastructure.get("available_capabilities"), "infrastructure_assessment.available_capabilities")
    missing = _string_list(infrastructure.get("missing_capabilities"), "infrastructure_assessment.missing_capabilities")
    _nonempty_string(infrastructure.get("assessment_basis"), "infrastructure_assessment.assessment_basis")
    if infrastructure["r2_testability"] == "TESTABLE_NOW" and missing:
        raise RegistryValidationError("TESTABLE_NOW ideas cannot declare missing capabilities")
    if infrastructure["r2_testability"] not in {"TESTABLE_NOW", "CONCEPT_ONLY"} and not missing:
        raise RegistryValidationError("blocked ideas must name the missing governed capability")

    result_informed_sources = [s for s in sources if s["result_informed"]]
    if mode == "OUTCOME_BLIND":
        if result_informed_sources or record["derived_from_results"]:
            raise RegistryValidationError("OUTCOME_BLIND ideas cannot be derived from observed results")
    else:
        if not result_informed_sources:
            raise RegistryValidationError("result-informed generation mode requires at least one result-informed source")
        if not record["derived_from_results"]:
            raise RegistryValidationError("result-informed ideas must retain derived_from_results lineage")
        if not record["datasets_that_influenced_design"]:
            raise RegistryValidationError("result-informed ideas must record datasets_that_influenced_design")
        allowed_result_zones = {"D0"} if mode == "D0_RESULT_INFORMED" else {"D0", "LEGACY_D0"}
        if any(s["source_zone"] not in allowed_result_zones for s in result_informed_sources):
            raise RegistryValidationError("result-informed sources must come only from D0/legacy-D0")

    forbidden = sorted(_FORBIDDEN_RESULT_KEYS.intersection(record.keys()))
    if forbidden:
        raise RegistryValidationError(
            "R1 idea packets store result lineage, not embedded outcome statistics; forbidden fields: " + ", ".join(forbidden)
        )
    verify_freeze_hash(record)


def validate_generation_session(record: Mapping[str, Any]) -> None:
    _reject_unknown_fields(record, _SESSION_FIELDS, "generation-session record")
    sid = _nonempty_string(record.get("session_id"), "session_id")
    if not SESSION_ID_RE.fullmatch(sid):
        raise RegistryValidationError("session_id must match GEN-R1-<stable-id>")
    version = record.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise RegistryValidationError("generation-session version must be an integer >= 1")
    _iso_utc(record.get("created_at_utc"), "created_at_utc")
    if record.get("status") not in SESSION_STATUSES:
        raise RegistryValidationError("unknown generation-session status")
    planned = _string_list(record.get("planned_lenses"), "planned_lenses", nonempty=True)
    completed = _string_list(record.get("completed_lenses"), "completed_lenses")
    if len(planned) != len(set(planned)) or len(completed) != len(set(completed)):
        raise RegistryValidationError("generation lenses must not contain duplicates")
    if any(lens not in planned for lens in completed):
        raise RegistryValidationError("completed_lenses must be a subset of planned_lenses")
    if record.get("status") == "COMPLETE" and set(completed) != set(planned):
        raise RegistryValidationError("COMPLETE generation session must complete every planned lens")
    if record.get("status") == "FROZEN_PLANNED" and completed:
        raise RegistryValidationError("FROZEN_PLANNED session cannot already contain completed lenses")

    allowed_zones = _string_list(record.get("allowed_source_zones"), "allowed_source_zones", nonempty=True)
    if any(zone not in SOURCE_ZONES for zone in allowed_zones):
        raise RegistryValidationError("R1 generation session declares a protected/unknown source zone")
    allowed_modes = _string_list(record.get("allowed_generation_modes"), "allowed_generation_modes", nonempty=True)
    if any(mode not in GENERATION_MODES for mode in allowed_modes):
        raise RegistryValidationError("generation session declares unknown generation mode")

    for field in ("p1_market_data_read", "p1_outcome_read", "protected_performance_read"):
        if record.get(field) is not False:
            raise RegistryValidationError(f"{field} must remain false during R1 universe generation")
    if record.get("no_outcome_based_pruning_during_generation") is not True:
        raise RegistryValidationError("R1 generation cannot prune the universe using observed outcomes")
    if record.get("retain_failed_duplicate_and_legacy_lineage") is not True:
        raise RegistryValidationError("R1 generation must preserve duplicate/failed/legacy ancestry rather than erase it")
    _nonempty_string(record.get("family_splitting_rule"), "family_splitting_rule")
    stop = record.get("stopping_rule")
    if not isinstance(stop, Mapping):
        raise RegistryValidationError("stopping_rule must be structured")
    _reject_unknown_fields(stop, {"type", "required_lenses", "completion_criterion"}, "stopping_rule")
    if stop.get("type") != "PREDECLARED_LENS_COVERAGE":
        raise RegistryValidationError("R1.0 requires PREDECLARED_LENS_COVERAGE stopping")
    required = _string_list(stop.get("required_lenses"), "stopping_rule.required_lenses", nonempty=True)
    if required != list(DEFAULT_GENERATION_LENSES):
        raise RegistryValidationError("R1.0 stopping_rule.required_lenses must equal the frozen default lens order")
    _nonempty_string(stop.get("completion_criterion"), "stopping_rule.completion_criterion")
    verify_freeze_hash(record)


def validate_universe(
    families: Sequence[Mapping[str, Any]],
    ideas: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]],
) -> None:
    family_ids: set[str] = set()
    for family in families:
        validate_family(family)
        fid = family["family_id"]
        if fid in family_ids:
            raise RegistryValidationError(f"duplicate family_id: {fid}")
        family_ids.add(fid)

    idea_keys: set[tuple[str, int]] = set()
    refs: set[str] = set()
    latest_idea_version: dict[str, int] = {}
    for idea in ideas:
        validate_idea(idea)
        key = (idea["idea_id"], idea["version"])
        if key in idea_keys:
            raise RegistryValidationError(f"duplicate idea version: {key[0]}@{key[1]}")
        prior = latest_idea_version.get(idea["idea_id"])
        if prior is None and idea["version"] != 1:
            raise RegistryValidationError("first idea version must be 1")
        if prior is not None:
            if idea["version"] != prior + 1:
                raise RegistryValidationError("idea versions must be appended contiguously")
            expected_parent = f"{idea['idea_id']}@{prior}"
            if expected_parent not in idea["parent_idea_refs"]:
                raise RegistryValidationError("new idea version must retain the immediately prior version as a parent")
        latest_idea_version[idea["idea_id"]] = idea["version"]
        idea_keys.add(key)
        refs.add(idea_ref(idea))
        if idea["family_id"] not in family_ids:
            raise RegistryValidationError(f"idea references unknown family: {idea['family_id']}")

    graph: dict[str, list[str]] = {}
    for idea in ideas:
        ref = idea_ref(idea)
        graph[ref] = list(idea["parent_idea_refs"])
        for parent in idea["parent_idea_refs"]:
            if parent not in refs:
                raise RegistryValidationError(f"idea references unknown parent idea version: {parent}")
            if parent == ref:
                raise RegistryValidationError("idea cannot parent itself")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(ref: str) -> None:
        if ref in visited:
            return
        if ref in visiting:
            raise RegistryValidationError("idea genealogy contains a cycle")
        visiting.add(ref)
        for parent in graph.get(ref, []):
            visit(parent)
        visiting.remove(ref)
        visited.add(ref)

    for ref in graph:
        visit(ref)

    session_keys: set[tuple[str, int]] = set()
    latest_session: dict[str, Mapping[str, Any]] = {}
    status_order = {"FROZEN_PLANNED": 0, "IN_PROGRESS": 1, "COMPLETE": 2}
    for session in sessions:
        validate_generation_session(session)
        key = (session["session_id"], session["version"])
        if key in session_keys:
            raise RegistryValidationError(f"duplicate generation-session version: {key[0]}@{key[1]}")
        session_keys.add(key)
        prior = latest_session.get(session["session_id"])
        if prior is None:
            if session["version"] != 1:
                raise RegistryValidationError("first generation-session version must be 1")
        else:
            if session["version"] != prior["version"] + 1:
                raise RegistryValidationError("generation-session versions must be appended contiguously")
            if session["planned_lenses"] != prior["planned_lenses"]:
                raise RegistryValidationError("frozen planned_lenses cannot change in later session versions")
            if not set(prior["completed_lenses"]).issubset(session["completed_lenses"]):
                raise RegistryValidationError("completed generation lenses cannot be removed")
            if status_order[session["status"]] < status_order[prior["status"]]:
                raise RegistryValidationError("generation-session status cannot move backwards")
        latest_session[session["session_id"]] = session


def promote_idea_to_hypothesis(
    idea: Mapping[str, Any],
    *,
    hypothesis_id: str,
    created_at_utc: str,
) -> dict[str, Any]:
    """Create an R0.7-compatible IDEA-stage hypothesis packet from an R1 idea.

    This does not append to the permanent registry and does not promote beyond
    the IDEA lifecycle stage.  The caller remains responsible for registry
    append, lifecycle events, feasibility, and all later registration gates.
    """
    validate_idea(idea)
    if not HYPOTHESIS_ID_RE.fullmatch(hypothesis_id):
        raise RegistryValidationError("hypothesis_id must match HYP-<stable-id>")
    _iso_utc(created_at_utc, "created_at_utc")
    record = {
        "hypothesis_id": hypothesis_id,
        "version": 1,
        "conceptual_family": idea["family_id"],
        "creation_source": f"R1_UNIVERSE:{idea['generation_mode']}:{idea_ref(idea)}",
        "created_at_utc": created_at_utc,
        "scientific_proposition": idea["scientific_proposition"],
        "relationship_type": idea["relationship_type"],
        "parent_hypothesis_ids": list(idea["parent_hypothesis_ids"]),
        "derived_from_results": list(idea["derived_from_results"]),
        "datasets_that_influenced_design": list(idea["datasets_that_influenced_design"]),
        "material_spec": {
            "r1_idea_ref": idea_ref(idea),
            "economic_mechanism": idea["economic_mechanism"],
            "expected_market_response": idea["expected_market_response"],
            "causal_state_requirements": idea["causal_state_requirements"],
            "trigger_concept": idea["trigger_concept"],
            "candidate_definition_status": idea["candidate_definition_status"],
            "position_concept": idea["position_concept"],
            "primary_economic_target": idea["primary_economic_target"],
            "risk_concepts": idea["risk_concepts"],
            "execution_dependencies": idea["execution_dependencies"],
            "infrastructure_assessment": idea["infrastructure_assessment"],
            "legacy_lineage_refs": idea["legacy_lineage_refs"],
        },
    }
    return with_freeze_hash(record)
