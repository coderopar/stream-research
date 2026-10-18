"""R1.6 outcome-blind/legacy cross-venue generation governance checks."""
from __future__ import annotations
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
C=ROOT/'research/governance/r1_cross_venue_contract_v1.json'
V=ROOT/'research/r1/cross_venue/validate_cross_venue_generation.py'
PUB=ROOT/'research/r1/cross_venue/new_public_sources.json'

def runner():
    sp=importlib.util.spec_from_file_location('r16_validator',V)
    m=importlib.util.module_from_spec(sp)
    assert sp.loader is not None
    sp.loader.exec_module(m)
    return m

def contract(): return json.loads(C.read_text(encoding='utf-8'))
def read(name): return [json.loads(x) for x in (U/name).read_text(encoding='utf-8').splitlines() if x.strip()]
def added(): return read('ideas.jsonl')[85:101]
def sources():
    return {x['source_id'] for x in json.loads(PUB.read_text(encoding='utf-8'))} | {'LIT-MAKAROV-SCHOAR-2020'}
def leg():
    return {x['evidence_id'] for x in json.loads((ROOT/'research/r1/legacy_7m/legacy_evidence_catalogue.json').read_text())['evidence']}


def test_accepted_parent_is_frozen_r15():
    p=json.loads((ROOT/'research/governance/r1_derivatives_positioning_contract_v1.json').read_text())
    assert contract()['accepted_parent']=={'tag':'r1-derivatives-positioning-v1','contract_id':p['contract_id']}

def test_r16_not_outcome_testing():
    c=contract();assert c['protocol_version']=='1.2' and c['scientific_boundaries']['new_d0_outcome_tests'] is False

def test_prior_prefix_sha256_byte_preserved():runner().assert_parent_prefix(contract())

def test_prior_prefix_tamper_fails(tmp_path,monkeypatch):
    m=runner()
    for n in m.PRIOR:(tmp_path/n).write_bytes((U/n).read_bytes())
    p=tmp_path/'ideas.jsonl';p.write_bytes(b'X'+p.read_bytes())
    monkeypatch.setattr(m,'U',tmp_path)
    with pytest.raises(Exception,match='prefix'):m.assert_parent_prefix(contract())

def test_universe_validates_under_r10_rules():
    m=runner();f,i,s=(read(x) for x in m.PRIOR)
    m.validate_universe(f,i,s)

def test_exact_own_family_and_historical_prefix():
    f=read('families.jsonl');c=contract()
    assert len(f)>=16 and c['new_family_ids']==['FAM-CROSS-VENUE-LIQUIDITY-SEGMENTATION']
    assert f[15]['freeze_hash']==c['new_family_freeze_hashes'][f[15]['family_id']]
    assert f[15]['family_id'] not in {x['family_id'] for x in f[:15]}

def test_mechanism_family_split_not_by_horizon_or_venue():
    f=read('families.jsonl')[15]
    text=f['new_family_boundary_rule'].lower()
    assert 'threshold' in text and 'venue' in text and 'horizon' in text
    assert 'independent' in f['anti_family_splitting_rationale']

def test_reuses_information_diffusion_family():
    assert 'FAM-CROSS-VENUE-INFORMATION-DIFFUSION' in {x['family_id'] for x in added()}
    assert set(x['family_id'] for x in added()) == runner().VALID_FAMILIES

def test_exact_idea_roster_versions_and_hashes():
    a=added();c=contract()
    assert len(a)==16 and [x['idea_id'] for x in a]==c['new_idea_ids']
    assert all(x['version']==1 for x in a)
    assert {x['idea_id']:x['freeze_hash'] for x in a}==c['new_idea_freeze_hashes']

def test_r16_twelve_fresh_four_legacy():
    assert Counter(x['generation_mode'] for x in added())=={'OUTCOME_BLIND':12,'LEGACY_RESULT_INFORMED':4}

def test_fresh_ideas_cannot_hide_legacy_results():
    for i in added():
        if i['generation_mode']=='OUTCOME_BLIND':
            assert not i['derived_from_results'] and not i['legacy_lineage_refs']
            assert not i['datasets_that_influenced_design']
            assert all(not x['result_informed'] for x in i['source_refs'])

def test_legacy_idea_ancestry_exact_and_archived():
    c=contract();a={x['idea_id']:x for x in added()}
    for iid,anc in c['legacy_parent_links'].items():
        x=a[iid]
        assert x['generation_mode']=='LEGACY_RESULT_INFORMED'
        assert x['derived_from_results']==[anc['evidence_id']]
        assert x['legacy_lineage_refs']==[anc['evidence_id']]
        assert x['parent_idea_refs']==[anc['parent_idea_ref']]
        assert x['datasets_that_influenced_design']==[runner().DATASET]

def test_legacy_spot_and_okx_null_branches_retained():
    assert {x['derived_from_results'][0] for x in added() if x['generation_mode']=='LEGACY_RESULT_INFORMED'} == {'LEG7M-09','LEG7M-10'}
    text=' '.join(x['economic_mechanism'] for x in added() if x['generation_mode']=='LEGACY_RESULT_INFORMED').lower()
    assert 'did not discriminate' in text and 'failed' in text and ('no convincing confirmation' in text or 'inconclusive' in text)

def test_old_sources_have_hashed_archive_evidence():
    cat=json.loads((ROOT/'research/r1/legacy_7m/legacy_evidence_catalogue.json').read_text())['evidence']
    ids={x['evidence_id']:x for x in cat}
    for key in ('LEG7M-09','LEG7M-10'):
        assert ids[key]['record_type']=='OUTCOME_STUDY'
        assert len(ids[key]['source_sha256'])==64

def test_never_inherits_25bp_or_p75_label():
    assert all(i['candidate_definition_status']=='TO_BE_DEVELOPED_IN_D0' for i in added())
    assert all('P75/15m' in i['position_concept']['horizon_concept'] for i in added())

def test_all_new_ideas_are_capability_blocked():
    assert all(x['infrastructure_assessment']['r2_testability'].startswith('BLOCKED_PENDING') for x in added())
    assert all(x['infrastructure_assessment']['missing_capabilities'] for x in added())

def test_cross_venue_governed_capability_mapping():
    c=contract();rules=c['mandatory_capability_gates']
    for i in added():
        req={x['state_family'] for x in i['causal_state_requirements'] if x['required']}
        expected={rules[k] for k in req if k in rules}
        if i['idea_id']=='IDEA-R1-CROSS-CROSS-VENUE-EXECUTION-BAND':expected.add(c['additional_execution_gate'])
        assert set(i['infrastructure_assessment']['missing_capabilities'])==expected

def test_no_r06_book_claimed_cross_venue_predictor():
    assert contract()['scientific_boundaries']['R0_6_not_multi_venue_execution'] is True
    assert contract()['scientific_boundaries']['spot_and_okx_feeds_not_present_as_governed_r05_features'] is True

def test_two_leg_execution_cost_band_blocked_separately():
    i=next(x for x in added() if x['idea_id']=='IDEA-R1-CROSS-CROSS-VENUE-EXECUTION-BAND')
    assert i['infrastructure_assessment']['r2_testability']=='BLOCKED_PENDING_STATE_AND_EXECUTION'
    assert 'GOVERNED_MULTI_VENUE_EXECUTION_SETTLEMENT' in i['infrastructure_assessment']['missing_capabilities']

def test_mid_or_quote_gap_never_assumed_profit():
    c=contract();assert c['scientific_boundaries']['recorded_quote_gap_not_risk_free_arbitrage']
    assert all('no R1 profitability claim' in x['primary_economic_target'] for x in added())

def test_receive_time_instead_of_future_timestamp():
    assert contract()['scientific_boundaries']['receive_time_and_symbol_normalization_mandatory']
    assert all('receive-time' in ' '.join(x['execution_dependencies']).lower() or 'as-of' in x['causal_state_requirements'][0]['role'].lower() for x in added())

def test_okx_books5_not_full_depth():
    assert contract()['scientific_boundaries']['OKX_books5_not_full_executable_depth']
    assert 'snapshot' in (ROOT/'research/r1/cross_venue/README.md').read_text(encoding='utf-8').lower()

def test_missing_feed_packet_not_treated_as_no_trades():
    assert contract()['scientific_boundaries']['missing_updates_not_zero_market_activity']
    assert any('missing feed partitions' in ' '.join(x['risk_concepts']).lower() for x in added() if x['idea_id'].endswith('SPOT-PAUSE'))

def test_two_new_public_sources_hashed_and_https():
    c=contract();assert hashlib.sha256(PUB.read_bytes()).hexdigest()==c['new_public_sources_sha256']
    assert runner().validate_sources(json.loads(PUB.read_text()),c['new_public_source_ids'])==sources()-{'LIT-MAKAROV-SCHOAR-2020'}

def test_all_public_sources_referenced_and_paper_already_catalogued():
    assert 'LIT-MAKAROV-SCHOAR-2020' in {x['source_id'] for x in json.loads((ROOT/'research/r1/theory_mechanisms/literature_sources.json').read_text())}
    used={r['reference'] for i in added() for r in i['source_refs'] if r['source_type']=='THEORY_LITERATURE'}
    assert used==sources()

def test_unknown_new_source_rejected():
    m=runner();a=copy.deepcopy(added());a[0]['source_refs'][0]['reference']='LIT-FAKE'
    with pytest.raises(Exception):m.validate_added_ideas(a,read('ideas.jsonl')[:85],contract(),sources(),leg())

def test_cross_venue_forged_testability_rejected():
    m=runner();a=copy.deepcopy(added());a[0]['infrastructure_assessment']['missing_capabilities']=[]
    with pytest.raises(Exception):m.validate_added_ideas(a,read('ideas.jsonl')[:85],contract(),sources(),leg())

def test_new_idea_freeze_mutation_rejected():
    m=runner();a=copy.deepcopy(added());a[0]['economic_mechanism']='guaranteed arbitrage'
    with pytest.raises(Exception,match='freeze'):m.validate_added_ideas(a,read('ideas.jsonl')[:85],contract(),sources(),leg())

def test_source_catalogue_tamper_detected(tmp_path,monkeypatch):
    m=runner();p=tmp_path/'new_public_sources.json';p.write_bytes(PUB.read_bytes()+b'\n')
    monkeypatch.setattr(m,'PUBLIC_NEW',p)
    with pytest.raises(Exception,match='tampered'):m.validate_installation()

def test_seventh_session_append_still_open():
    s=read('generation_sessions.jsonl');c=contract()
    assert s[6]['version']==7 and s[6]['status']=='IN_PROGRESS'
    assert s[6]['completed_lenses']==s[0]['planned_lenses'][:7]
    assert s[6]['freeze_hash']==c['session_v7_freeze_hash']
    assert c['remaining_lenses']==['VOLATILITY_REGIME','PREDICTIVE_MODEL']

def test_permanent_registry_unmodified():
    assert (ROOT/'research/governance/registries/hypothesis_versions.jsonl').read_text().strip()==''

def test_no_p1_or_market_outcomes_in_runner():
    t=V.read_text().lower()
    assert 'data/canonical/' not in t and 'future/outcomes.parquet' not in t
    assert contract()['scientific_boundaries']['p1_market_data_read'] is False
    assert contract()['scientific_boundaries']['p1_outcome_read'] is False

def test_r15_validator_compatible_with_append():
    p=ROOT/'research/r1/derivatives/validate_derivatives_generation.py'
    sp=importlib.util.spec_from_file_location('r15_after_r16',p)
    m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    assert m.validate_installation()['status']=='PASS'

def test_r16_living_readme_manifest_contract_present():
    for p in ('research/README.md','research/r1/README.md','research/r1/cross_venue/README.md','research/r1/cross_venue/R1_6_IMPLEMENTATION_MANIFEST.json'):
        assert (ROOT/p).exists()
    assert 'R1.6' in (ROOT/'research/README.md').read_text()

def test_validator_normal_and_selftest():
    assert runner().validate_installation()['status']=='PASS'
    for a in ([],['--self-test']):
        p=subprocess.run([sys.executable,str(V),*a],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=30)
        assert p.returncode==0,p.stderr
        j=json.loads(p.stdout)
        assert j['new_ideas']==16 and j['permanent_hypothesis_rows']==0
        if a:assert j['self_test']=='PASS'
