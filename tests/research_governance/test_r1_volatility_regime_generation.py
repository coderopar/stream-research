"""Acceptance/adversarial tests for R1.7 volatility-regime generation (no market reads)."""
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[2]
U=ROOT/'research/governance/universe'
C=ROOT/'research/governance/r1_volatility_regime_contract_v1.json'
V=ROOT/'research/r1/volatility_regime/validate_volatility_regime_generation.py'
PUB=ROOT/'research/r1/volatility_regime/new_public_sources.json'

def runner():
    spec=importlib.util.spec_from_file_location('r17_validator',V)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def contract():return json.loads(C.read_text(encoding='utf-8'))
def read(n):return [json.loads(x) for x in (U/n).read_text(encoding='utf-8').splitlines() if x.strip()]
def added():return read('ideas.jsonl')[101:117]
def sources():return {r['source_id'] for r in json.loads(PUB.read_text())}

def test_r17_parent_tag_commit_and_contract():
    c=contract();p=json.loads((ROOT/'research/governance/r1_cross_venue_contract_v1.json').read_text())
    assert c['accepted_parent']=={'tag':'r1-cross-venue-v1','contract_id':p['contract_id'],'commit_prefix':'d5ea875'}

def test_r17_only_hypothesis_generation():
    c=contract();assert c['protocol_version']=='1.2' and c['scientific_boundaries']['new_d0_outcome_tests'] is False

def test_r16_prefix_byte_hash_all_three():runner().assert_parent_prefix(contract())

def test_prefix_truncation_fails(tmp_path,monkeypatch):
    m=runner()
    for name in m.PRIOR:(tmp_path/name).write_bytes((U/name).read_bytes())
    (tmp_path/'ideas.jsonl').write_bytes(b'')
    monkeypatch.setattr(m,'U',tmp_path)
    with pytest.raises(Exception,match='prefix'):m.assert_parent_prefix(contract())

def test_prefix_rewrite_fails(tmp_path,monkeypatch):
    m=runner()
    for name in m.PRIOR:(tmp_path/name).write_bytes((U/name).read_bytes())
    p=tmp_path/'families.jsonl';p.write_bytes(b'X'+p.read_bytes())
    monkeypatch.setattr(m,'U',tmp_path)
    with pytest.raises(Exception,match='prefix'):m.assert_parent_prefix(contract())

def test_full_universe_r10_valid():
    m=runner();m.validate_universe(*(read(name) for name in m.PRIOR))

def test_append_exact_counts_and_versions():
    assert len(read('families.jsonl'))==17 and len(read('ideas.jsonl'))==117 and len(read('generation_sessions.jsonl'))==8
    assert read('generation_sessions.jsonl')[7]['version']==8

def test_one_distinct_family_and_other_reused():
    a=added();c=contract();fs={i['family_id'] for i in a}
    assert fs=={'FAM-VOLATILITY-PERSISTENCE-TRANSITION','FAM-VOLATILITY-LIQUIDITY-FEEDBACK'}
    assert c['new_family_ids']==['FAM-VOLATILITY-PERSISTENCE-TRANSITION']

def test_family_variants_not_independent_evidence():
    f=read('families.jsonl')[16]
    assert all(x in f['new_family_boundary_rule'].lower() for x in ('threshold','horizon'))
    assert 'not independent' in f['anti_family_splitting_rationale'].lower()

def test_exact_idea_ids_and_hashes():
    a=added();c=contract()
    assert len(a)==16 and [x['idea_id'] for x in a]==c['new_idea_ids']
    assert all(i['version']==1 for i in a)
    assert {i['idea_id']:i['freeze_hash'] for i in a}==c['new_idea_freeze_hashes']

def test_source_provenance_all_outcome_blind():
    assert Counter(i['generation_mode'] for i in added())=={'OUTCOME_BLIND':16}
    assert contract()['generation_mode_counts']=={'OUTCOME_BLIND':16,'LEGACY_RESULT_INFORMED':0}

def test_no_hidden_legacy_parent_refs():
    for i in added():
        assert not any(i[x] for x in ('parent_idea_refs','parent_hypothesis_ids','legacy_lineage_refs','derived_from_results','datasets_that_influenced_design'))
        assert all(not r['result_informed'] for r in i['source_refs'])

def test_legacy_7m_unmodified_as_prior_ancestry():
    c=contract();assert c['scientific_boundaries']['retain_prior_legacy_volatility_family_without_hidden_result_ancestry']
    assert 'FAM-VOLATILITY-LIQUIDITY-FEEDBACK' in {i['family_id'] for i in added()}
    assert len(read('ideas.jsonl')[:101])==101

def test_no_preselected_legacy_candidate_policy():
    assert all(i['candidate_definition_status']=='TO_BE_DEVELOPED_IN_D0' for i in added())
    assert all('P75/15m' in i['position_concept']['horizon_concept'] for i in added())

def test_all_real_volatility_requires_price_path():
    for i in added():
        assert 'PERPS_PRICE_PATH' in {r['state_family'] for r in i['causal_state_requirements']}

def test_r05_latest_price_not_rolling_price_history():
    c=contract()['scientific_boundaries']
    assert c['r0_5_latest_trade_price_not_historical_price_path']
    assert c['r0_5_trade_activity_not_price_volatility']

def test_all_sixteen_blocked():
    assert all(i['infrastructure_assessment']['r2_testability'].startswith('BLOCKED_PENDING') for i in added())
    assert all('GOVERNED_PERPS_PRICE_PATH_STATE' in i['infrastructure_assessment']['missing_capabilities'] for i in added())

def test_trade_buckets_do_not_fake_volatility_capability():
    a=next(i for i in added() if i['idea_id']=='IDEA-R1-VOL-FLOW-IMBALANCE')
    assert 'PERPS_TRADES' in {r['state_family'] for r in a['causal_state_requirements']}
    assert a['infrastructure_assessment']['r2_testability']=='BLOCKED_PENDING_GOVERNED_STATE'

def test_missing_book_and_derivatives_require_extra_gates():
    a={i['idea_id']:i for i in added()}
    assert 'GOVERNED_ORDER_BOOK_STATE' in a['IDEA-R1-VOL-LIQUIDITY-FRAGILITY']['infrastructure_assessment']['missing_capabilities']
    assert 'GOVERNED_DERIVATIVES_POSITIONING_STATE' in a['IDEA-R1-VOL-DERIVATIVES-CROWDING']['infrastructure_assessment']['missing_capabilities']

def test_execution_variant_separately_blocked():
    a=next(i for i in added() if i['idea_id']=='IDEA-R1-VOL-CONDITIONED-EXECUTION')
    assert a['infrastructure_assessment']['r2_testability']=='BLOCKED_PENDING_STATE_AND_EXECUTION'
    assert 'GOVERNED_VOLATILITY_CONDITIONED_EXECUTION_RISK' in a['infrastructure_assessment']['missing_capabilities']

def test_volatility_does_not_supply_position_direction():
    assert all('not infer long/short' in i['position_concept']['directionality'].lower() for i in added())

def test_no_hindsight_regime_breakout():
    assert contract()['scientific_boundaries']['future_high_low_bar_and_future_realized_vol_forbidden']
    assert all(any('hindsight' in x.lower() or 'future' in x.lower() for x in i['risk_concepts']) for i in added())

def test_online_detection_delay_explicit():
    assert contract()['scientific_boundaries']['online_regimes_only_after_detection_clock']
    a=next(i for i in added() if i['idea_id']=='IDEA-R1-VOL-TRANSITION-DETECTION')
    assert 'lagged detection' in a['economic_mechanism']

def test_regime_slice_cannot_rescue_failed_whole():
    assert contract()['scientific_boundaries']['post_hoc_regime_slice_not_confirmatory_rescue']
    assert all(any('post-hoc' in x.lower() for x in i['risk_concepts']) for i in added())

def test_sizing_is_separate_strategy_dimension():
    assert contract()['scientific_boundaries']['regime_definition_and_sizing_require_separate_freeze']
    assert 'sizing' in next(i for i in added() if i['idea_id']=='IDEA-R1-VOL-CONDITIONED-EXECUTION')['economic_mechanism']

def test_joint_prediction_not_identified_causality():
    assert contract()['scientific_boundaries']['interaction_is_joint_prediction_not_identified_causal_effect']
    assert 'joint predictive state' in next(i for i in added() if i['idea_id']=='IDEA-R1-VOL-FLOW-IMBALANCE')['economic_mechanism']

def test_literature_roster_and_sha():
    c=contract();assert hashlib.sha256(PUB.read_bytes()).hexdigest()==c['new_public_sources_sha256']
    assert runner().validate_sources(json.loads(PUB.read_text()),c['new_public_source_ids'])==sources()

def test_every_new_source_used():
    assert {r['reference'] for i in added() for r in i['source_refs'] if r['source_type']=='THEORY_LITERATURE'}==sources()

def test_forged_testable_now_rejected():
    m=runner();a=copy.deepcopy(added());a[0]['infrastructure_assessment']['r2_testability']='TESTABLE_NOW';a[0]['infrastructure_assessment']['missing_capabilities']=[]
    with pytest.raises(Exception):m.validate_added_ideas(a,contract(),sources())

def test_hidden_result_source_rejected():
    m=runner();a=copy.deepcopy(added());a[0]['source_refs'].append({'source_type':'LEGACY_LINEAGE','source_zone':'LEGACY_D0','reference':'LEG7M-07','result_informed':True})
    with pytest.raises(Exception):m.validate_added_ideas(a,contract(),sources())

def test_mutated_idea_freeze_rejected():
    m=runner();a=copy.deepcopy(added());a[0]['scientific_proposition']='risk-free alpha'
    with pytest.raises(Exception):m.validate_added_ideas(a,contract(),sources())

def test_unknown_literature_rejected():
    m=runner();a=copy.deepcopy(added());a[0]['source_refs'][0]['reference']='LIT-FAKE'
    with pytest.raises(Exception):m.validate_added_ideas(a,contract(),sources())

def test_catalogue_tamper_rejected(tmp_path,monkeypatch):
    m=runner();p=tmp_path/'new_public_sources.json';p.write_bytes(PUB.read_bytes()+b' ')
    monkeypatch.setattr(m,'SOURCES',p)
    with pytest.raises(Exception,match='tampered'):m.validate_installation()

def test_invalid_public_source_reference_rejected():
    m=runner();a=copy.deepcopy(json.loads(PUB.read_text()));a[0]['stable_reference']='http://unsafe.example'
    with pytest.raises(Exception,match='DOI'):m.validate_sources(a,contract()['new_public_source_ids'])

def test_session_eight_still_open_only_one_lens_left():
    s=read('generation_sessions.jsonl')[7]
    assert s['status']=='IN_PROGRESS' and s['completed_lenses']==s['planned_lenses'][:8]
    assert contract()['remaining_lenses']==['PREDICTIVE_MODEL']

def test_permanent_hypothesis_registry_empty():
    assert (ROOT/'research/governance/registries/hypothesis_versions.jsonl').read_text().strip()==''

def test_no_p1_or_canonical_market_file_read_in_validator():
    t=V.read_text().lower()
    assert 'data/canonical/' not in t and 'outcomes.parquet' not in t
    assert contract()['scientific_boundaries']['p1_market_data_read'] is False
    assert contract()['scientific_boundaries']['p1_outcome_read'] is False

def test_r16_validator_still_passes_after_append():
    p=ROOT/'research/r1/cross_venue/validate_cross_venue_generation.py'
    sp=importlib.util.spec_from_file_location('r16_after_r17',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    assert m.validate_installation()['status']=='PASS'

def test_readme_manifest_and_contract_ship_together():
    for p in ('research/README.md','research/r1/README.md','research/r1/volatility_regime/README.md','research/r1/volatility_regime/R1_7_IMPLEMENTATION_MANIFEST.json'):
        assert (ROOT/p).exists()
    assert 'R1.7' in (ROOT/'research/README.md').read_text()

def test_validator_real_and_self_test_without_market_files():
    assert runner().validate_installation()['status']=='PASS'
    for arg in ([],['--self-test']):
        p=subprocess.run([sys.executable,str(V),*arg],cwd=ROOT,capture_output=True,text=True,timeout=30)
        assert p.returncode==0,p.stderr
        j=json.loads(p.stdout);assert j['total_families']==17 and j['total_ideas']==117
        if arg:assert j['self_test']=='PASS'
