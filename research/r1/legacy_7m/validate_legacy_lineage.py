"""Validate R1.2 historical result-informed idea genealogy; never access market/outcome files."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
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

UNIVERSE = REPO_ROOT / "research/governance/universe"
CATALOGUE = REPO_ROOT / "research/r1/legacy_7m/legacy_evidence_catalogue.json"
CONTRACT = REPO_ROOT / "research/governance/r1_legacy_lineage_contract_v1.json"
PROTECTED = REPO_ROOT / "research/governance/protected_data_register.yaml"
PERMANENT = REPO_ROOT / "research/governance/registries/hypothesis_versions.jsonl"
R11 = REPO_ROOT / "research/governance/r1_theory_mechanism_contract_v1.json"
R12_FAMILIES = 5
R12_IDEAS = 20
R11_FAMILIES = 8
R11_IDEAS = 15
EXPECTED_LENSES = ["THEORY_LITERATURE", "MARKET_MECHANISM", "LEGACY_7M_LINEAGE"]
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def validate_evidence(catalogue: dict, *, expected_ids: set[str]) -> dict[str, dict]:
    if set(catalogue) != {"catalogue_version", "source_bundle", "bundle_sha256", "evidence"}:
        raise RegistryValidationError("legacy evidence catalogue has unexpected fields")
    if catalogue["catalogue_version"] != "1.0" or catalogue["source_bundle"] != "full_microstructure_adversarial_review_bundle(1).zip":
        raise RegistryValidationError("unknown historical source bundle identity")
    if not HEX64.fullmatch(str(catalogue["bundle_sha256"])):
        raise RegistryValidationError("historical archive SHA256 is missing or malformed")
    evidence = catalogue["evidence"]
    if not isinstance(evidence, list):
        raise RegistryValidationError("legacy evidence must be a list")
    ids = set()
    fields = {"evidence_id", "phase", "source_bundle", "source_relative_path", "source_sha256", "source_bytes", "record_type", "historical_assessment", "reported_interpretation", "chronology", "parent_evidence_ids"}
    for item in evidence:
        if not isinstance(item, dict) or set(item) != fields:
            raise RegistryValidationError("historical evidence record has missing/extra fields")
        eid = item["evidence_id"]
        if not isinstance(eid, str) or not eid.startswith(("LEG7M-", "LEG7K-", "LEG7I-")) or eid in ids:
            raise RegistryValidationError("duplicate/invalid historical evidence identity")
        ids.add(eid)
        if item["source_bundle"] != catalogue["source_bundle"]:
            raise RegistryValidationError("evidence row source-bundle identity mismatch")
        path = item["source_relative_path"]
        if not isinstance(path, str) or not path.startswith("02_reports_archive_3/") or ".." in Path(path).parts or not path.endswith(".md"):
            raise RegistryValidationError("legacy evidence path must be immutable archived report Markdown")
        if not HEX64.fullmatch(str(item["source_sha256"])) or not isinstance(item["source_bytes"], int) or item["source_bytes"] <= 0:
            raise RegistryValidationError("evidence report hash/size missing")
        if item["record_type"] not in {"OUTCOME_STUDY", "SUBSTRATE", "FEASIBILITY_ONLY", "OUTCOME_INFORMED_DESIGN"}:
            raise RegistryValidationError("unknown historical evidence type")
        if item["historical_assessment"] in {"CONFIRMED", "D1_CONFIRMED", "STRATEGY_PROFITABLE"}:
            raise RegistryValidationError("historical report may not be upgraded to independent confirmation")
        if not isinstance(item["reported_interpretation"], str) or not item["reported_interpretation"].strip():
            raise RegistryValidationError("missing evidence interpretation")
        if item["record_type"] in {"OUTCOME_STUDY", "OUTCOME_INFORMED_DESIGN"} and item["chronology"] != "NOT_PROVEN_OUTCOME_INDEPENDENT_AT_PROGRAM_LEVEL":
            raise RegistryValidationError("legacy outcome studies must retain researcher-choice chronology uncertainty")
        if item["record_type"] not in {"OUTCOME_STUDY", "OUTCOME_INFORMED_DESIGN"} and item["chronology"] != "HISTORICAL_DESIGN_OR_FEASIBILITY_ONLY":
            raise RegistryValidationError("non-outcome evidence cannot be mislabelled as an outcome study")
        if not isinstance(item["parent_evidence_ids"], list):
            raise RegistryValidationError("parent_evidence_ids must be a list")
    for item in evidence:
        for parent in item["parent_evidence_ids"]:
            if parent not in ids or parent == item["evidence_id"]:
                raise RegistryValidationError("unknown or self-referential source-lineage parent")
    if ids != expected_ids:
        raise RegistryValidationError("historical source evidence roster differs from accepted generation contract")
    return {e["evidence_id"]: e for e in evidence}


def assert_append_only_ancestry(families: list, ideas: list, sessions: list, contract: dict) -> None:
    # R1.2 freezes the initial prefix, not the entire expanding R1 universe.
    # R1.3+ may only append; they may not rewrite these original rows.
    if len(families) < R11_FAMILIES + R12_FAMILIES or len(ideas) < R11_IDEAS + R12_IDEAS or len(sessions) < 3:
        raise RegistryValidationError("R1.2 frozen prefix is truncated")
    families = families[:R11_FAMILIES + R12_FAMILIES]
    ideas = ideas[:R11_IDEAS + R12_IDEAS]
    sessions = sessions[:3]
    if [r["family_id"] for r in families[:R11_FAMILIES]] != contract["prior_family_ids"]:
        raise RegistryValidationError("R1.1 family roster was modified or reordered")
    if [r["idea_id"] for r in ideas[:R11_IDEAS]] != contract["prior_idea_ids"]:
        raise RegistryValidationError("R1.1 idea roster was modified or reordered")
    if [r["family_id"] for r in families[R11_FAMILIES:]] != contract["added_family_ids"]:
        raise RegistryValidationError("new family roster differs from R1.2 contract")
    if [r["idea_id"] for r in ideas[R11_IDEAS:]] != contract["added_idea_ids"]:
        raise RegistryValidationError("new idea roster differs from R1.2 contract")
    # Freeze hashes bind all historical records; prefixes protect literal append-only lineage.
    for kind, n in (("families.jsonl", R11_FAMILIES), ("ideas.jsonl", R11_IDEAS), ("generation_sessions.jsonl", 2)):
        data = (UNIVERSE / kind).read_bytes()
        prefix = b"".join(line + b"\n" for line in data.splitlines()[:n])
        if hashlib.sha256(prefix).hexdigest() != contract["prior_prefix_sha256"][kind]:
            raise RegistryValidationError(f"R1.1 historical prefix changed in {kind}")
    if any(x["generation_mode"] != "OUTCOME_BLIND" for x in ideas[:R11_IDEAS]):
        raise RegistryValidationError("R1.1 must remain outcome-blind")


def validate_installation() -> dict:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract["protocol_version"] != "1.2" or contract["accepted_r1_1_parent"] != {"commit_prefix": "c8d57d9", "tag": "r1-theory-mechanisms-v1"}:
        raise RegistryValidationError("R1.2 does not match the accepted R1.1 checkpoint")
    for k in ("new_outcome_tests", "new_causal_feature_computations", "legacy_results_used_as_confirmation", "p1_market_data_read", "p1_outcome_read", "permanent_hypothesis_registry_mutation"):
        if contract["scope"].get(k) is not False:
            raise RegistryValidationError(f"R1.2 scope guard opened: {k}")
    if contract["scope"].get("legacy_archived_reports_reviewed") is not True:
        raise RegistryValidationError("R1.2 must disclose historical result-informed research")
    text = PROTECTED.read_text(encoding="utf-8")
    for token in ("pool_id: P1_PROTECTED", "status: SEALED", "outcome_exposure_status: UNSEEN"):
        if token not in text:
            raise RegistryValidationError("P1 seal invariant not satisfied")
    families, ideas, sessions = (read_jsonl(UNIVERSE / fn) for fn in ("families.jsonl", "ideas.jsonl", "generation_sessions.jsonl"))
    validate_universe(families, ideas, sessions)
    assert_append_only_ancestry(families, ideas, sessions, contract)
    # Restrict R1.2's historical claims to its exact frozen records, while
    # validate_universe above continues to check the *entire* growing registry.
    families = families[:R11_FAMILIES + R12_FAMILIES]
    ideas = ideas[:R11_IDEAS + R12_IDEAS]
    sessions = sessions[:3]
    catalogue = json.loads(CATALOGUE.read_text(encoding="utf-8"))
    sources = validate_evidence(catalogue, expected_ids=set(contract["evidence_ids"]))
    if hashlib.sha256(CATALOGUE.read_bytes()).hexdigest() != contract["evidence_catalogue_sha256"]:
        raise RegistryValidationError("historical evidence catalogue contents changed since contract freeze")
    if catalogue["bundle_sha256"] != contract["source_archive_sha256"]:
        raise RegistryValidationError("historical source bundle hash changed since contract freeze")
    counts = Counter(s["record_type"] for s in sources.values())
    used = set()
    for idea in ideas[R11_IDEAS:]:
        assert_no_protected_influence(idea)
        assert_generation_not_testing(idea)
        assert_testability_separate_from_merit(idea)
        if idea["generation_mode"] != "LEGACY_RESULT_INFORMED" or idea["candidate_definition_status"] != "LEGACY_FROZEN":
            raise RegistryValidationError("R1.2 idea must preserve explicitly result-informed legacy candidate context")
        refs = idea["legacy_lineage_refs"]
        if not refs or any(ref not in sources for ref in refs):
            raise RegistryValidationError("R1.2 idea lacks traceable archive source identity")
        used.update(refs)
        if set(idea["derived_from_results"]) != set(refs):
            raise RegistryValidationError("result-derived ancestry must match legacy evidence references")
        if idea["datasets_that_influenced_design"] != ["LEGACY_D0_2026-05-18_TO_2026-07-18__PHASE_7M_ARCHIVED_REPORTS"]:
            raise RegistryValidationError("legacy historical D0 influence must be explicit and only legacy D0")
        if set(s["reference"] for s in idea["source_refs"]) != set(refs) or any(s != {"source_type": "LEGACY_LINEAGE", "source_zone": "LEGACY_D0", "reference": s["reference"], "result_informed": True} for s in idea["source_refs"]):
            raise RegistryValidationError("legacy idea source records misrepresent zone or result influence")
        if idea["parent_hypothesis_ids"]:
            raise RegistryValidationError("historical hypotheses must not be invented as registered HYP ancestors")
        if idea["infrastructure_assessment"]["r2_testability"] == "TESTABLE_NOW":
            raise RegistryValidationError("original legacy feature reconstruction cannot be treated as governed testability")
    if used | set(contract["standalone_substrate_evidence_ids"]) | set(contract["standalone_policy_evidence_ids"]) != set(sources):
        raise RegistryValidationError("all negative, feasibility, and standalone substrate evidence must be accounted for")
    for eid in contract["standalone_policy_evidence_ids"]:
        if sources[eid]["record_type"] != "OUTCOME_INFORMED_DESIGN":
            raise RegistryValidationError("standalone policy selection must remain outcome-informed design ancestry")
    for eid in contract["standalone_substrate_evidence_ids"]:
        if sources[eid]["record_type"] != "SUBSTRATE":
            raise RegistryValidationError("standalone ancestry must be a substrate, never a dropped outcome study")
    if sessions[-1]["version"] != 3 or sessions[-1]["completed_lenses"] != EXPECTED_LENSES or sessions[-1]["status"] != "IN_PROGRESS":
        raise RegistryValidationError("R1.2 should complete only the legacy lens and leave R1 generation open")
    if read_jsonl(PERMANENT):
        raise RegistryValidationError("R1.2 may not append permanent HYP-* rows")
    return {
        "status": "PASS", "protocol_version": "1.2", "accepted_r1_1_parent": contract["accepted_r1_1_parent"],
        "existing_r1_1_families": R11_FAMILIES, "added_families": R12_FAMILIES,
        "family_rows": len(families), "existing_r1_1_ideas": R11_IDEAS,
        "added_legacy_result_informed_ideas": R12_IDEAS, "idea_rows": len(ideas),
        "legacy_source_records": len(sources), "upstream_policy_design_reports": counts["OUTCOME_INFORMED_DESIGN"], "historical_outcome_study_reports": counts["OUTCOME_STUDY"],
        "historical_feasibility_reports": counts["FEASIBILITY_ONLY"], "historical_substrate_reports": counts["SUBSTRATE"],
        "generation_session_rows": len(sessions), "completed_lenses": EXPECTED_LENSES,
        "permanent_hypothesis_rows": 0,
        "historical_legacy_reports_reviewed": True, "new_d0_outcome_tests": False,
        "p1_market_data_read": False, "p1_outcome_read": False, "p1_status": "SEALED_OUTCOME_UNSEEN",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Validate source-locked R1.2 legacy 7M lineage, without market data.")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    result = validate_installation()
    if args.self_test:
        assert result["added_legacy_result_informed_ideas"] == 20
        assert result["legacy_source_records"] == 19
        result["self_test"] = "PASS"
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
