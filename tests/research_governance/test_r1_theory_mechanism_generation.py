from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_core.hypothesis_universe import validate_universe  # noqa: E402
from research_core.registry import read_jsonl  # noqa: E402

CONTRACT = REPO_ROOT / "research/governance/r1_theory_mechanism_contract_v1.json"
UNIVERSE = REPO_ROOT / "research/governance/universe"
SOURCES = REPO_ROOT / "research/r1/theory_mechanisms/literature_sources.json"
MANIFEST = REPO_ROOT / "research/r1/theory_mechanisms/R1_1_IMPLEMENTATION_MANIFEST.json"


def _validator():
    p = REPO_ROOT / "research/r1/theory_mechanisms/validate_theory_mechanism_generation.py"
    spec = importlib.util.spec_from_file_location("r11_validator", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _rows():
    return (
        read_jsonl(UNIVERSE / "families.jsonl"),
        read_jsonl(UNIVERSE / "ideas.jsonl"),
        read_jsonl(UNIVERSE / "generation_sessions.jsonl"),
    )


def test_r11_contract_anchors_to_accepted_r10_freeze():
    obj = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert obj["accepted_r1_0_parent"]["commit"] == "93f954f303f2585dd7de171292dc12075b97cdde"
    assert obj["accepted_r1_0_parent"]["tag"] == "r1-hypothesis-universe-governance-v1"


def test_r11_contract_is_generation_not_testing():
    obj = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert obj["evidentiary_role"] == "OUTCOME_BLIND_HYPOTHESIS_GENERATION_NOT_TESTING"
    assert obj["r1_1_scope"]["creates_market_hypothesis_ideas"] is True
    assert obj["r1_1_scope"]["tests_market_hypotheses"] is False
    assert obj["r1_1_scope"]["selects_thresholds_from_results"] is False
    assert obj["r1_1_scope"]["selects_horizons_from_results"] is False


def test_r11_contract_keeps_d0_legacy_and_p1_results_closed():
    obj = json.loads(CONTRACT.read_text(encoding="utf-8"))
    policy = obj["source_policy"]
    assert policy["allowed_source_zones"] == ["PUBLIC", "NONE"]
    assert policy["d0_results_allowed"] is False
    assert policy["legacy_results_allowed"] is False
    assert policy["d0_outcomes_read"] is False
    assert policy["p1_market_data_read"] is False
    assert policy["p1_outcome_read"] is False


def test_r11_exact_family_and_idea_rosters_match_contract():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    families, ideas, _ = _rows()
    assert {x["family_id"] for x in families} == set(contract["family_ids"])
    assert {x["idea_id"] for x in ideas} == set(contract["idea_ids"])
    assert len(families) == 8
    assert len(ideas) == 15


def test_r11_repository_universe_validates_under_r10_rules():
    families, ideas, sessions = _rows()
    validate_universe(families, ideas, sessions)


def test_all_r11_ideas_are_outcome_blind():
    _, ideas, _ = _rows()
    assert ideas
    assert all(x["generation_mode"] == "OUTCOME_BLIND" for x in ideas)
    assert all(x["derived_from_results"] == [] for x in ideas)
    assert all(x["datasets_that_influenced_design"] == [] for x in ideas)
    assert all(x["legacy_lineage_refs"] == [] for x in ideas)


def test_r11_seed_ideas_have_no_parent_result_or_hypothesis_lineage():
    _, ideas, _ = _rows()
    assert all(x["parent_idea_refs"] == [] for x in ideas)
    assert all(x["parent_hypothesis_ids"] == [] for x in ideas)


def test_r11_sources_are_public_or_first_principles_only():
    _, ideas, _ = _rows()
    for idea in ideas:
        for source in idea["source_refs"]:
            assert source["source_zone"] in {"PUBLIC", "NONE"}
            assert source["source_type"] in {"THEORY_LITERATURE", "MARKET_MECHANISM"}
            assert source["result_informed"] is False


def test_every_r11_idea_has_literature_and_mechanism_provenance():
    _, ideas, _ = _rows()
    for idea in ideas:
        types = {s["source_type"] for s in idea["source_refs"]}
        assert "THEORY_LITERATURE" in types
        assert "MARKET_MECHANISM" in types


def test_literature_catalogue_has_unique_stable_public_sources():
    rows = json.loads(SOURCES.read_text(encoding="utf-8"))
    assert len(rows) == 10
    ids = [x["source_id"] for x in rows]
    assert len(ids) == len(set(ids))
    assert all(x.startswith("LIT-") for x in ids)
    assert all(x["stable_reference"].startswith("https://") for x in rows)


def test_all_literature_references_used_by_ideas_exist_in_catalogue():
    catalog = {x["source_id"] for x in json.loads(SOURCES.read_text(encoding="utf-8"))}
    _, ideas, _ = _rows()
    refs = {
        s["reference"]
        for idea in ideas
        for s in idea["source_refs"]
        if s["source_type"] == "THEORY_LITERATURE"
    }
    assert refs <= catalog
    assert refs


def test_source_catalogue_does_not_embed_project_outcome_statistics():
    text = SOURCES.read_text(encoding="utf-8").lower()
    for forbidden in ("p_value", "p-value", " bps", "% win", "d0 outcome", "p1 outcome"):
        assert forbidden not in text


def test_family_records_have_explicit_anti_splitting_boundaries():
    families, _, _ = _rows()
    for family in families:
        assert family["variant_dimensions_kept_in_family"]
        assert family["new_family_boundary_rule"].strip()
        assert family["anti_family_splitting_rationale"].strip()


def test_r11_does_not_rank_scientific_merit():
    _, ideas, _ = _rows()
    serialized = json.dumps(ideas).lower()
    for forbidden in ("scientific_merit_score", "priority_score", "expected_alpha_score", "rank_score"):
        assert forbidden not in serialized


def test_only_two_ideas_are_currently_testable_now_and_testability_is_not_merit():
    _, ideas, _ = _rows()
    testable = [x for x in ideas if x["infrastructure_assessment"]["r2_testability"] == "TESTABLE_NOW"]
    assert {x["idea_id"] for x in testable} == {
        "IDEA-R1-INFORMED-FLOW-CONTINUATION",
        "IDEA-R1-SIGNED-FLOW-PERSISTENCE",
    }
    assert all(x["infrastructure_assessment"]["missing_capabilities"] == [] for x in testable)


def test_order_book_ideas_are_not_mistaken_for_r06_execution_book_features():
    _, ideas, _ = _rows()
    order_book = [x for x in ideas if any(r["state_family"] == "ORDER_BOOK" for r in x["causal_state_requirements"])]
    assert order_book
    for idea in order_book:
        assert idea["infrastructure_assessment"]["r2_testability"] != "TESTABLE_NOW"
        assert "GOVERNED_ORDER_BOOK_STATE" in idea["infrastructure_assessment"]["missing_capabilities"]


def test_cross_venue_ideas_require_governed_cross_venue_state():
    _, ideas, _ = _rows()
    rows = [x for x in ideas if x["family_id"] == "FAM-CROSS-VENUE-INFORMATION-DIFFUSION"]
    assert len(rows) == 2
    assert all("GOVERNED_SPOT_CROSS_VENUE_STATE" in x["infrastructure_assessment"]["missing_capabilities"] for x in rows)


def test_volatility_feedback_ideas_require_governed_volatility_state():
    _, ideas, _ = _rows()
    rows = [x for x in ideas if x["family_id"] == "FAM-VOLATILITY-LIQUIDITY-FEEDBACK"]
    assert len(rows) == 2
    assert all("GOVERNED_VOLATILITY_STATE" in x["infrastructure_assessment"]["missing_capabilities"] for x in rows)


def test_r11_appends_session_v2_without_changing_frozen_plan():
    _, _, sessions = _rows()
    assert len(sessions) == 2
    assert sessions[0]["version"] == 1 and sessions[0]["status"] == "FROZEN_PLANNED"
    assert sessions[1]["version"] == 2 and sessions[1]["status"] == "IN_PROGRESS"
    assert sessions[1]["planned_lenses"] == sessions[0]["planned_lenses"]
    assert sessions[1]["stopping_rule"] == sessions[0]["stopping_rule"]


def test_r11_completes_only_theory_and_market_mechanism_lenses():
    _, _, sessions = _rows()
    assert sessions[-1]["completed_lenses"] == ["THEORY_LITERATURE", "MARKET_MECHANISM"]


def test_seven_generation_lenses_remain_after_r11():
    _, _, sessions = _rows()
    latest = sessions[-1]
    remaining = [x for x in latest["planned_lenses"] if x not in latest["completed_lenses"]]
    assert remaining == [
        "LEGACY_7M_LINEAGE",
        "ORDER_BOOK_LIQUIDITY",
        "PERPS_TRADE_FLOW",
        "DERIVATIVES_POSITIONING",
        "CROSS_VENUE",
        "VOLATILITY_REGIME",
        "PREDICTIVE_MODEL",
    ]


def test_permanent_r07_hypothesis_registry_remains_empty():
    rows = read_jsonl(REPO_ROOT / "research/governance/registries/hypothesis_versions.jsonl")
    assert rows == []


def test_r11_manifest_declares_protection_and_no_testing():
    obj = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert obj["tests_market_hypotheses"] is False
    assert obj["reads_d0_outcomes"] is False
    assert obj["reads_legacy_7m_results"] is False
    assert obj["p1_market_data_read"] is False
    assert obj["p1_outcome_read"] is False
    assert obj["mutates_r0_7_hypothesis_registry"] is False


def test_r11_validator_source_cannot_open_market_or_outcome_files():
    text = (REPO_ROOT / "research/r1/theory_mechanisms/validate_theory_mechanism_generation.py").read_text(encoding="utf-8").lower()
    assert "data/canonical" not in text
    assert "outcomes.parquet" not in text
    assert "future/outcomes" not in text
    assert "legacy 7m result file" not in text


def test_r11_readme_explicitly_defers_detailed_lenses():
    text = (REPO_ROOT / "research/r1/theory_mechanisms/README.md").read_text(encoding="utf-8")
    for step in ("R1.3", "R1.4", "R1.6", "R1.7"):
        assert step in text
    assert "mechanism families" in text.lower()
    assert "not" in text.lower()


def test_living_research_readme_records_r11_and_frozen_r10_parent():
    text = (REPO_ROOT / "research/README.md").read_text(encoding="utf-8")
    assert "R1.1" in text
    assert "Theory / market-mechanism" in text or "Theory / Market-Mechanism" in text
    assert "93f954f303f2585dd7de171292dc12075b97cdde" in text
    assert "r1-hypothesis-universe-governance-v1" in text


def test_r1_readme_records_r11_lens_progress():
    text = (REPO_ROOT / "research/r1/README.md").read_text(encoding="utf-8")
    assert "R1.1" in text
    assert "THEORY_LITERATURE" in text
    assert "MARKET_MECHANISM" in text
    assert "R1.2" in text


def test_validator_installation_passes_without_market_files():
    summary = _validator().validate_installation()
    assert summary["status"] == "PASS"
    assert summary["family_rows"] == 8
    assert summary["idea_rows"] == 15
    assert summary["literature_source_rows"] == 10
    assert summary["permanent_hypothesis_rows"] == 0
    assert summary["d0_outcomes_read_by_r1_1"] is False
    assert summary["p1_market_data_read"] is False
    assert summary["p1_outcome_read"] is False


def test_validator_self_test_passes():
    _validator().self_test()
