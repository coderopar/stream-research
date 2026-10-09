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
    DEFAULT_GENERATION_LENSES,
    assert_generation_not_testing,
    assert_no_protected_influence,
    assert_testability_separate_from_merit,
    promote_idea_to_hypothesis,
    validate_universe,
)
from research_core.registry import read_jsonl, validate_hypothesis_version, with_freeze_hash  # noqa: E402
from research_core.registry.models import RegistryValidationError  # noqa: E402

CONTRACT_PATH = REPO_ROOT / "research" / "governance" / "hypothesis_universe_contract_v1.json"
UNIVERSE_DIR = REPO_ROOT / "research" / "governance" / "universe"
PROTECTED_REGISTER = REPO_ROOT / "research" / "governance" / "protected_data_register.yaml"
R09_MANIFEST = REPO_ROOT / "research" / "r0" / "audit" / "R0_9_IMPLEMENTATION_MANIFEST.json"


def _load_contract() -> dict:
    obj = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if obj.get("protocol_version") != "1.2":
        raise RegistryValidationError("R1.0 contract is not bound to Protocol v1.2")
    parent = obj.get("accepted_r0_parent", {})
    if parent.get("commit") != "71192f37e8aa44da4c30326048debf65b1ca5792":
        raise RegistryValidationError("R1.0 is not anchored to the accepted R0 foundation commit")
    if parent.get("tag") != "r0-foundation-accepted-v1":
        raise RegistryValidationError("R1.0 is not anchored to the accepted R0 foundation tag")
    firewall = obj.get("protected_data_firewall", {})
    if firewall.get("p1_market_data_read") is not False or firewall.get("p1_outcome_read") is not False:
        raise RegistryValidationError("R1.0 protected-data firewall is not closed")
    if obj.get("predeclared_generation_lenses") != list(DEFAULT_GENERATION_LENSES):
        raise RegistryValidationError("R1.0 contract lens order differs from executable governance")
    return obj


def _validate_r0_acceptance() -> None:
    obj = json.loads(R09_MANIFEST.read_text(encoding="utf-8"))
    final = obj.get("final_acceptance", {})
    if final.get("overall_status") != "PASS_WITH_ACCEPTED_LIMITATIONS":
        raise RegistryValidationError("R1 cannot open before accepted R0.9 status")
    if final.get("substantive_audit_hash") != "3faab2f04bb4902d19d90a0a9c6df49b42e62b648572c6316110ea869b87accd":
        raise RegistryValidationError("R1.0 R0.9 acceptance hash differs from the frozen parent")
    if final.get("counts", {}).get("blocking_r1") != 0:
        raise RegistryValidationError("R0.9 still records an R1 blocker")


def _validate_p1_seal() -> None:
    text = PROTECTED_REGISTER.read_text(encoding="utf-8")
    for token in ("pool_id: P1_PROTECTED", "status: SEALED", "outcome_exposure_status: UNSEEN"):
        if token not in text:
            raise RegistryValidationError(f"P1 seal no longer matches accepted R0 boundary: missing {token!r}")


def _load_universe() -> tuple[list[dict], list[dict], list[dict]]:
    families = read_jsonl(UNIVERSE_DIR / "families.jsonl")
    ideas = read_jsonl(UNIVERSE_DIR / "ideas.jsonl")
    sessions = read_jsonl(UNIVERSE_DIR / "generation_sessions.jsonl")
    validate_universe(families, ideas, sessions)
    for idea in ideas:
        assert_no_protected_influence(idea)
        assert_generation_not_testing(idea)
        assert_testability_separate_from_merit(idea)
    for session in sessions:
        assert_no_protected_influence(session)
    if not sessions:
        raise RegistryValidationError("R1.0 frozen generation plan is missing")
    first = sessions[0]
    if first.get("session_id") != "GEN-R1-UNIVERSE-V1" or first.get("version") != 1:
        raise RegistryValidationError("first generation session must be the frozen R1.0 universe plan")
    if first.get("status") != "FROZEN_PLANNED":
        raise RegistryValidationError("first generation session must freeze the pre-generation plan")
    return families, ideas, sessions


def validate_installation() -> dict:
    contract = _load_contract()
    _validate_r0_acceptance()
    _validate_p1_seal()
    families, ideas, sessions = _load_universe()
    return {
        "status": "PASS",
        "protocol_version": contract["protocol_version"],
        "accepted_r0_parent": contract["accepted_r0_parent"],
        "p1_status": "SEALED_OUTCOME_UNSEEN",
        "p1_market_data_read": False,
        "p1_outcome_read": False,
        "d0_outcomes_read_by_r1_0": False,
        "family_rows": len(families),
        "idea_rows": len(ideas),
        "generation_session_rows": len(sessions),
        "predeclared_lenses": list(DEFAULT_GENERATION_LENSES),
    }


def self_test() -> None:
    family = with_freeze_hash({
        "family_id": "FAM-SELFTEST",
        "family_name": "Self-test family",
        "created_at_utc": "2026-10-09T00:00:00Z",
        "mechanism_class": "SELF_TEST",
        "mechanism_statement": "Synthetic mechanism used only to exercise governance.",
        "market_state_domains": ["PERPS_TRADES"],
        "relationship_types": ["MARGINAL"],
        "variant_dimensions_kept_in_family": ["threshold", "horizon"],
        "new_family_boundary_rule": "Only a changed mechanism creates a new family.",
        "anti_family_splitting_rationale": "Threshold and horizon variants are not independent mechanisms.",
        "legacy_lineage_refs": [],
    })
    idea = with_freeze_hash({
        "idea_id": "IDEA-R1-SELFTEST",
        "version": 1,
        "family_id": "FAM-SELFTEST",
        "created_at_utc": "2026-10-09T00:01:00Z",
        "title": "Synthetic governance idea",
        "scientific_proposition": "A synthetic state may be associated with a synthetic future response.",
        "economic_mechanism": "Self-test only; no market claim.",
        "relationship_type": "MARGINAL",
        "expected_market_response": "Synthetic directional response",
        "causal_state_requirements": [{"state_family": "PERPS_TRADES", "role": "synthetic input", "required": True}],
        "trigger_concept": "Synthetic trigger",
        "candidate_definition_status": "CONCEPTUAL",
        "position_concept": {
            "directionality": "conditional",
            "entry_style": "conceptual only",
            "horizon_concept": "fixed conceptual horizon",
            "exit_concept": "conceptual time exit",
        },
        "primary_economic_target": "Net return after explicit costs once a real proposition exists",
        "risk_concepts": ["MAE", "tail loss"],
        "execution_dependencies": ["marketable execution model"],
        "generation_mode": "OUTCOME_BLIND",
        "source_refs": [{
            "source_type": "MARKET_MECHANISM",
            "source_zone": "NONE",
            "reference": "R1.0 synthetic self-test",
            "result_informed": False,
        }],
        "parent_idea_refs": [],
        "parent_hypothesis_ids": [],
        "legacy_lineage_refs": [],
        "derived_from_results": [],
        "datasets_that_influenced_design": [],
        "infrastructure_assessment": {
            "r2_testability": "TESTABLE_NOW",
            "available_capabilities": ["PERPS_TRADES"],
            "missing_capabilities": [],
            "assessment_basis": "Synthetic self-test only",
        },
    })
    session = read_jsonl(UNIVERSE_DIR / "generation_sessions.jsonl")[0]
    validate_universe([family], [idea], [session])
    promoted = promote_idea_to_hypothesis(
        idea,
        hypothesis_id="HYP-R1SELFTEST",
        created_at_utc="2026-10-09T00:02:00Z",
    )
    validate_hypothesis_version(promoted)
    assert promoted["material_spec"]["r1_idea_ref"] == "IDEA-R1-SELFTEST@1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate R1.0 hypothesis-universe governance without opening market or outcome data."
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    summary = validate_installation()
    if args.self_test:
        self_test()
        summary["self_test"] = "PASS"
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
