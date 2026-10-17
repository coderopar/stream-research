from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_core.registry import read_jsonl, with_freeze_hash  # noqa: E402
from research_core.registry.models import RegistryValidationError  # noqa: E402
from research_core.hypothesis_universe import validate_universe, validate_idea  # noqa: E402

U = ROOT / "research/governance/universe"
CONTRACT = ROOT / "research/governance/r1_legacy_lineage_contract_v1.json"
CATALOGUE = ROOT / "research/r1/legacy_7m/legacy_evidence_catalogue.json"
DOC = ROOT / "research/r1/legacy_7m/README.md"
MATRIX = ROOT / "research/r1/legacy_7m/lineage_matrix.md"
MANIFEST = ROOT / "research/r1/legacy_7m/R1_2_IMPLEMENTATION_MANIFEST.json"


def m():
    p = ROOT / "research/r1/legacy_7m/validate_legacy_lineage.py"
    spec = importlib.util.spec_from_file_location("r12_validator", p)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rows():
    # Original R1.2 tests remain about their immutable *prefix* after R1.3+.
    f, i, s = (read_jsonl(U / n) for n in ("families.jsonl", "ideas.jsonl", "generation_sessions.jsonl"))
    return f[:13], i[:35], s[:3]


def evidence():
    return json.loads(CATALOGUE.read_text(encoding="utf-8"))


def contract():
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_full_universe_validates_under_accepted_r10_rules():
    validate_universe(*(read_jsonl(U / n) for n in ("families.jsonl", "ideas.jsonl", "generation_sessions.jsonl")))


def test_r12_append_counts_and_parent_freeze():
    families, ideas, sessions = rows()
    m().assert_append_only_ancestry(families, ideas, sessions, contract())
    assert len(families) == 13 and len(ideas) == 35 and len(sessions) == 3
    assert contract()["accepted_r1_1_parent"] == {"commit_prefix":"c8d57d9", "tag":"r1-theory-mechanisms-v1"}


def test_old_r11_ideas_remain_outcome_blind():
    _, ideas, _ = rows()
    assert all(i["generation_mode"] == "OUTCOME_BLIND" for i in ideas[:15])
    assert all(not i["derived_from_results"] and not i["legacy_lineage_refs"] for i in ideas[:15])


def test_all_20_new_ideas_explicitly_legacy_result_informed():
    _, ideas, _ = rows()
    assert len(ideas[15:]) == 20
    for i in ideas[15:]:
        assert i["generation_mode"] == "LEGACY_RESULT_INFORMED"
        assert i["derived_from_results"] == i["legacy_lineage_refs"]
        assert i["datasets_that_influenced_design"] == ["LEGACY_D0_2026-05-18_TO_2026-07-18__PHASE_7M_ARCHIVED_REPORTS"]
        assert all(s["result_informed"] is True and s["source_zone"] == "LEGACY_D0" for s in i["source_refs"])
        assert not i["parent_hypothesis_ids"]


def test_all_legacy_evidence_is_anchored_to_hashed_archived_reports():
    c = evidence()
    assert len(c["evidence"]) == 19
    m().validate_evidence(c,expected_ids=set(contract()["evidence_ids"]))
    assert hashlib.sha256(CATALOGUE.read_bytes()).hexdigest() == contract()["evidence_catalogue_sha256"]
    assert all(x["source_relative_path"].startswith("02_reports_archive_3/") for x in c["evidence"])


def test_pre7m_policy_ancestry_is_recorded_as_result_informed_not_hypothesis_confirmation():
    e={x['evidence_id']:x for x in evidence()['evidence']}
    assert set(contract()['standalone_policy_evidence_ids'])=={'LEG7I-01','LEG7K-01','LEG7K-02','LEG7K-03','LEG7K-04'}
    assert set(e['LEG7M-02']['parent_evidence_ids'])=={'LEG7K-01','LEG7K-02','LEG7K-04'}
    assert all(e[k]['record_type']=='OUTCOME_INFORMED_DESIGN' for k in contract()['standalone_policy_evidence_ids'])


def test_historical_report_types_are_not_rewritten_to_confirmed():
    c=evidence(); assert sum(x["record_type"]=="OUTCOME_STUDY" for x in c["evidence"])==10
    assert sum(x["record_type"]=="FEASIBILITY_ONLY" for x in c["evidence"])==3
    assert sum(x["record_type"]=="SUBSTRATE" for x in c["evidence"])==1
    assert sum(x["record_type"]=="OUTCOME_INFORMED_DESIGN" for x in c["evidence"])==5
    assert all('CONFIRMED' != x['historical_assessment'] for x in c['evidence'])


def test_every_negative_feasibility_and_null_branch_is_preserved():
    _, ideas, _=rows()
    suffixes = {i["idea_id"].replace("IDEA-R1-LEGACY-", "") for i in ideas[15:]}
    for s in ("REFILL-REPLACEMENT-PENETRATION", "AGGRESSIVE-TRADE-CONCENTRATION", "BASIS-CHANGE", "CURRENT-BASIS", "OI-EXPANSION", "FORCE-ORDER-PRESENCE", "BINANCE-SPOT-FLOW", "BINANCE-SPOT-PRICE", "OKX-BBO-IMBALANCE", "OKX-PRICE-CONFIRMATION", "OPTIONS-GAMMA-GEOMETRY"):
        assert s in suffixes


def test_early_35bp_conditioning_and_25bp_legacy_policy_are_not_endorsed():
    text=MATRIX.read_text(encoding='utf8')+DOC.read_text(encoding='utf8')
    assert 'conditioned' in text.lower() and '35bp' in text
    assert '25bp' in text and 'not' in text.lower()
    assert contract()['restrictions']['legacy_candidate_policy_status']=='HISTORICAL_D0_DEVELOPMENT_ONLY_NOT_ACCEPTED_FINAL_EVENT_POLICY'


def test_legacy_clock_overlap_is_explicitly_not_revalidated():
    text=DOC.read_text(encoding='utf8')
    assert 'candidate_second+3s' in text and 'R0.2/R0.3' in text
    assert contract()['restrictions']['original_outcome_clock_status']=='SUPERSEDED_R0_2_R0_3'


def test_unsigned_gamma_cannot_be_reframed_as_signed_dealer_flow():
    _, ideas,_=rows()
    gamma=next(x for x in ideas if x['idea_id']=='IDEA-R1-LEGACY-OPTIONS-GAMMA-GEOMETRY')
    assert 'no dealer sign' in gamma['economic_mechanism']
    assert 'FEASIBILITY_ONLY' in gamma['infrastructure_assessment']['assessment_basis']
    assert 'OPTIONS_OPEN_INTEREST' in [s['state_family'] for s in gamma['causal_state_requirements']]


def test_r11_mechanism_family_is_reused_not_relabelled_from_a_positive_result():
    fam,ideas,_=rows(); known={x['family_id'] for x in fam[:8]}
    assert next(i for i in ideas if i['idea_id']=='IDEA-R1-LEGACY-OPPOSING-WITHDRAWAL')['family_id'] in known
    assert next(i for i in ideas if i['idea_id']=='IDEA-R1-LEGACY-OKX-BBO-IMBALANCE')['family_id'] in known
    assert len(fam[8:]) == 5


def test_ideas_cannot_embed_new_p_values_or_historical_results_as_fields():
    _, ideas,_=rows()
    copy_record=copy.deepcopy(ideas[15]);copy_record['p_value']=0.0001
    with pytest.raises(RegistryValidationError):validate_idea(copy_record)


def test_legacy_idea_without_result_ancestry_is_rejected():
    _, ideas,_=rows();record=copy.deepcopy(ideas[15]);record['derived_from_results']=[]
    record=with_freeze_hash({k:v for k,v in record.items() if k != 'freeze_hash'})
    with pytest.raises(RegistryValidationError,match='derived_from_results'):validate_idea(record)


def test_protected_source_zone_rejected():
    _,ideas,_=rows();x=copy.deepcopy(ideas[15]);x['source_refs'][0]['source_zone']='P1_PROTECTED'
    x=with_freeze_hash({k:v for k,v in x.items() if k != 'freeze_hash'})
    with pytest.raises(RegistryValidationError,match='protected'): validate_idea(x)


def test_report_source_tamper_is_detected(tmp_path, monkeypatch):
    v=m(); c=evidence();c['evidence'][0]['source_sha256']='0'*64
    p=tmp_path/'tampered_catalogue.json'
    p.write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8')
    monkeypatch.setattr(v,'CATALOGUE',p)
    with pytest.raises(RegistryValidationError,match='catalogue contents'):
        v.validate_installation()


def test_unknown_evidence_reference_is_rejected(tmp_path, monkeypatch):
    v=m()
    tmp=tmp_path/'universe';tmp.mkdir()
    for name in ('families.jsonl','generation_sessions.jsonl'):
        (tmp/name).write_bytes((U/name).read_bytes())
    ideas=read_jsonl(U/'ideas.jsonl')
    ideas[15]['legacy_lineage_refs']=['LEG7M-NOT-IN-CATALOGUE']
    ideas[15]=with_freeze_hash({k:x for k,x in ideas[15].items() if k!='freeze_hash'})
    with (tmp/'ideas.jsonl').open('w',encoding='utf8') as f:
        for i in ideas:f.write(json.dumps(i,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n')
    monkeypatch.setattr(v,'UNIVERSE',tmp)
    with pytest.raises(RegistryValidationError,match='traceable'):
        v.validate_installation()


def test_substrate_can_be_standalone_but_no_outcome_report_can_be_silently_dropped():
    c=contract();cat=evidence()
    assert c['standalone_substrate_evidence_ids']==['LEG7M-03']
    assert next(x for x in cat['evidence'] if x['evidence_id']=='LEG7M-03')['record_type']=='SUBSTRATE'
    assert set(c['standalone_substrate_evidence_ids']).isdisjoint({x['evidence_id'] for x in cat['evidence'] if x['record_type']=='OUTCOME_STUDY'})


def test_source_outcome_program_chronology_uncertainty_is_preserved():
    assert all(e['chronology']=='NOT_PROVEN_OUTCOME_INDEPENDENT_AT_PROGRAM_LEVEL' for e in evidence()['evidence'] if e['record_type']=='OUTCOME_STUDY')


def test_testability_is_not_merit_and_all_legacy_features_are_capability_gated():
    _,ideas,_=rows()
    for i in ideas[15:]:
        assert i['infrastructure_assessment']['r2_testability']=='BLOCKED_PENDING_GOVERNED_STATE'
        assert i['infrastructure_assessment']['missing_capabilities']


def test_permanent_registry_remains_empty():
    assert not read_jsonl(ROOT/'research/governance/registries/hypothesis_versions.jsonl')


def test_session_completes_only_legacy_lens_in_addition_to_prior_two():
    _,_,sessions=rows()
    assert sessions[-1]['version']==3
    assert sessions[-1]['completed_lenses']==['THEORY_LITERATURE','MARKET_MECHANISM','LEGACY_7M_LINEAGE']
    assert len(sessions[-1]['planned_lenses'])==9
    assert sessions[-1]['status']=='IN_PROGRESS'


def test_protection_contract_excludes_fresh_outcomes_and_p1():
    scope=contract()['scope']
    assert scope['legacy_archived_reports_reviewed'] is True
    for k in ('new_outcome_tests','new_causal_feature_computations','legacy_results_used_as_confirmation','p1_market_data_read','p1_outcome_read','permanent_hypothesis_registry_mutation'):
        assert scope[k] is False


def test_validator_source_has_no_parquet_market_or_p1_reads():
    text=(ROOT/'research/r1/legacy_7m/validate_legacy_lineage.py').read_text().lower()
    assert 'pq.read_' not in text and 'data/canonical' not in text and 'future/outcomes.parquet' not in text
    assert 'market_events_v1/' not in text


def test_readmes_and_manifest_were_included_with_code():
    assert 'R1.2' in (ROOT/'research/README.md').read_text()
    assert 'R1.2' in (ROOT/'research/r1/README.md').read_text()
    assert 'R1.2' in (ROOT/'research/governance/universe/README.md').read_text()
    assert 'R1.2' in DOC.read_text()
    assert 'LEG7M-12A' in MATRIX.read_text()
    assert json.loads(MANIFEST.read_text())['expected_counts']['new_ideas']==20


def test_validator_passes_without_archived_bundle_or_market_data():
    v=m().validate_installation()
    assert v['status']=='PASS' and v['p1_market_data_read'] is False
    assert v['new_d0_outcome_tests'] is False
    assert v['family_rows']==13 and v['idea_rows']==35


def test_validator_self_test():
    v=m().validate_installation()
    assert v['legacy_source_records']==19
    assert v['generation_session_rows']==3
