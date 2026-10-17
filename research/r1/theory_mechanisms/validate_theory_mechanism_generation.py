from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_core.hypothesis_universe import (  # noqa: E402
    assert_generation_not_testing,
    assert_no_protected_influence,
    assert_testability_separate_from_merit,
    validate_universe,
)
from research_core.registry import read_jsonl  # noqa: E402
from research_core.registry.models import RegistryValidationError  # noqa: E402

CONTRACT = REPO_ROOT / "research" / "governance" / "r1_theory_mechanism_contract_v1.json"
UNIVERSE = REPO_ROOT / "research" / "governance" / "universe"
SOURCES = REPO_ROOT / "research" / "r1" / "theory_mechanisms" / "literature_sources.json"
PROTECTED = REPO_ROOT / "research" / "governance" / "protected_data_register.yaml"
PERMANENT_HYPOTHESES = REPO_ROOT / "research" / "governance" / "registries" / "hypothesis_versions.jsonl"

EXPECTED_PARENT_COMMIT = "93f954f303f2585dd7de171292dc12075b97cdde"
EXPECTED_PARENT_TAG = "r1-hypothesis-universe-governance-v1"
EXPECTED_COMPLETED = ["THEORY_LITERATURE", "MARKET_MECHANISM"]


def _load_contract() -> dict:
    obj = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if obj.get("protocol_version") != "1.2":
        raise RegistryValidationError("R1.1 contract must remain on Protocol v1.2")
    parent = obj.get("accepted_r1_0_parent", {})
    if parent.get("commit") != EXPECTED_PARENT_COMMIT or parent.get("tag") != EXPECTED_PARENT_TAG:
        raise RegistryValidationError("R1.1 is not anchored to the accepted R1.0 Git freeze")
    scope = obj.get("r1_1_scope", {})
    for key in ("tests_market_hypotheses", "reads_d0_outcomes", "reads_p1", "uses_legacy_7m_results", "mutates_r0_7_hypothesis_registry"):
        if scope.get(key) is not False:
            raise RegistryValidationError(f"R1.1 scope guard is open: {key}")
    source_policy = obj.get("source_policy", {})
    if source_policy.get("allowed_source_zones") != ["PUBLIC", "NONE"]:
        raise RegistryValidationError("R1.1 source zones must remain PUBLIC/NONE only")
    if source_policy.get("d0_results_allowed") is not False or source_policy.get("legacy_results_allowed") is not False:
        raise RegistryValidationError("R1.1 cannot be D0/legacy result informed")
    return obj


def _validate_p1_seal() -> None:
    text = PROTECTED.read_text(encoding="utf-8")
    for token in ("pool_id: P1_PROTECTED", "status: SEALED", "outcome_exposure_status: UNSEEN"):
        if token not in text:
            raise RegistryValidationError(f"P1 seal mismatch: missing {token!r}")


def _load_sources() -> tuple[list[dict], set[str]]:
    rows = json.loads(SOURCES.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise RegistryValidationError("R1.1 literature source catalogue must be a non-empty JSON list")
    ids: set[str] = set()
    required = {"source_id", "title", "authors", "year", "source_kind", "stable_reference", "supported_mechanism_claims", "scope_note"}
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise RegistryValidationError("R1.1 literature source record has missing/extra fields")
        sid = row["source_id"]
        if not isinstance(sid, str) or not sid.startswith("LIT-") or sid in ids:
            raise RegistryValidationError("R1.1 literature source_id must be unique and start LIT-")
        ids.add(sid)
        if not isinstance(row["title"], str) or not row["title"].strip():
            raise RegistryValidationError("literature source title must be non-empty")
        if not isinstance(row["authors"], list) or not row["authors"] or any(not isinstance(x, str) or not x.strip() for x in row["authors"]):
            raise RegistryValidationError("literature source authors must be non-empty strings")
        if not isinstance(row["year"], int) or row["year"] < 1900 or row["year"] > 2026:
            raise RegistryValidationError("literature source year is invalid")
        if not isinstance(row["stable_reference"], str) or not row["stable_reference"].startswith("https://"):
            raise RegistryValidationError("literature source stable_reference must be HTTPS")
        claims = row["supported_mechanism_claims"]
        if not isinstance(claims, list) or not claims or any(not isinstance(x, str) or not x.strip() for x in claims):
            raise RegistryValidationError("literature source claims must be non-empty qualitative strings")
        combined = " ".join(claims).lower()
        for forbidden in ("p_value", "p-value", "project return", "d0 outcome", "p1 outcome"):
            if forbidden in combined:
                raise RegistryValidationError("source catalogue must remain qualitative and project-outcome free")
    return rows, ids


def _validate_r11_universe(contract: dict, source_ids: set[str]) -> tuple[list[dict], list[dict], list[dict]]:
    families = read_jsonl(UNIVERSE / "families.jsonl")
    ideas = read_jsonl(UNIVERSE / "ideas.jsonl")
    sessions = read_jsonl(UNIVERSE / "generation_sessions.jsonl")
    # R1.1's frozen *prefix* remains authoritative after later append-only R1 lenses.
    # Validate the entire growing universe, then evaluate R1.1 assertions on its
    # first eight families, fifteen ideas and two session versions only.
    validate_universe(families, ideas, sessions)
    if len(families) < 8 or len(ideas) < 15 or len(sessions) < 2:
        raise RegistryValidationError("R1.1 frozen prefix was truncated")
    families, ideas, sessions = families[:8], ideas[:15], sessions[:2]

    expected_families = set(contract["family_ids"])
    expected_ideas = set(contract["idea_ids"])
    if {x["family_id"] for x in families} != expected_families:
        raise RegistryValidationError("R1.1 family roster differs from frozen contract")
    if {x["idea_id"] for x in ideas} != expected_ideas:
        raise RegistryValidationError("R1.1 idea roster differs from frozen contract")
    if len(families) != contract["universe_addition"]["family_count"]:
        raise RegistryValidationError("R1.1 family count differs from contract")
    if len(ideas) != contract["universe_addition"]["idea_count"]:
        raise RegistryValidationError("R1.1 idea count differs from contract")

    for idea in ideas:
        assert_no_protected_influence(idea)
        assert_generation_not_testing(idea)
        assert_testability_separate_from_merit(idea)
        if idea["generation_mode"] != "OUTCOME_BLIND":
            raise RegistryValidationError(f"R1.1 idea is not outcome-blind: {idea['idea_id']}")
        if idea["derived_from_results"] or idea["datasets_that_influenced_design"] or idea["legacy_lineage_refs"]:
            raise RegistryValidationError(f"R1.1 idea contains result/legacy lineage: {idea['idea_id']}")
        if idea["parent_hypothesis_ids"] or idea["parent_idea_refs"]:
            raise RegistryValidationError(f"R1.1 seed idea unexpectedly has parent lineage: {idea['idea_id']}")
        lit_refs = []
        for source in idea["source_refs"]:
            if source["source_zone"] not in {"PUBLIC", "NONE"}:
                raise RegistryValidationError("R1.1 idea uses a non-public/non-none source zone")
            if source["source_type"] not in {"THEORY_LITERATURE", "MARKET_MECHANISM"}:
                raise RegistryValidationError("R1.1 idea uses an out-of-scope source type")
            if source["result_informed"] is not False:
                raise RegistryValidationError("R1.1 source cannot be project-result informed")
            if source["source_type"] == "THEORY_LITERATURE":
                lit_refs.append(source["reference"])
                if source["reference"] not in source_ids:
                    raise RegistryValidationError(f"idea references unknown literature source: {source['reference']}")
        if not lit_refs:
            raise RegistryValidationError(f"R1.1 idea lacks theory/literature provenance: {idea['idea_id']}")

    if len(sessions) != 2:
        raise RegistryValidationError("R1.1 must append exactly one generation-session update to the R1.0 plan")
    latest = sessions[-1]
    if latest["session_id"] != "GEN-R1-UNIVERSE-V1" or latest["version"] != 2:
        raise RegistryValidationError("R1.1 must end at GEN-R1-UNIVERSE-V1 version 2")
    if latest["status"] != "IN_PROGRESS":
        raise RegistryValidationError("R1.1 generation session must remain IN_PROGRESS")
    if latest["completed_lenses"] != EXPECTED_COMPLETED:
        raise RegistryValidationError("R1.1 must complete only THEORY_LITERATURE and MARKET_MECHANISM")
    return families, ideas, sessions


def _validate_permanent_registry_untouched() -> int:
    rows = read_jsonl(PERMANENT_HYPOTHESES)
    if rows:
        raise RegistryValidationError("R1.1 must not mutate the permanent R0.7 hypothesis registry")
    return 0


def validate_installation() -> dict:
    contract = _load_contract()
    _validate_p1_seal()
    sources, source_ids = _load_sources()
    families, ideas, sessions = _validate_r11_universe(contract, source_ids)
    permanent_rows = _validate_permanent_registry_untouched()
    testable = sum(x["infrastructure_assessment"]["r2_testability"] == "TESTABLE_NOW" for x in ideas)
    blocked = sum(x["infrastructure_assessment"]["r2_testability"].startswith("BLOCKED_") for x in ideas)
    return {
        "status": "PASS",
        "protocol_version": "1.2",
        "accepted_r1_0_parent": contract["accepted_r1_0_parent"],
        "completed_lenses": EXPECTED_COMPLETED,
        "family_rows": len(families),
        "idea_rows": len(ideas),
        "literature_source_rows": len(sources),
        "testable_now_ideas": testable,
        "blocked_pending_capability_ideas": blocked,
        "generation_session_rows": len(sessions),
        "permanent_hypothesis_rows": permanent_rows,
        "d0_outcomes_read_by_r1_1": False,
        "legacy_7m_results_read_by_r1_1": False,
        "p1_market_data_read": False,
        "p1_outcome_read": False,
        "p1_status": "SEALED_OUTCOME_UNSEEN",
    }


def self_test() -> None:
    summary = validate_installation()
    assert summary["family_rows"] == 8
    assert summary["idea_rows"] == 15
    assert summary["literature_source_rows"] == 10
    assert summary["permanent_hypothesis_rows"] == 0
    assert summary["completed_lenses"] == EXPECTED_COMPLETED


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate R1.1 outcome-blind theory/market-mechanism generation.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    summary = validate_installation()
    if args.self_test:
        self_test()
        summary["self_test"] = "PASS"
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
