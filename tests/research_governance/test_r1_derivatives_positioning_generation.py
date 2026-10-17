"""R1.5 append-only derivatives positioning generation governance tests."""
import copy
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
U = ROOT / 'research/governance/universe'
C = ROOT / 'research/governance/r1_derivatives_positioning_contract_v1.json'
V = ROOT / 'research/r1/derivatives/validate_derivatives_generation.py'
S = ROOT / 'research/r1/derivatives/new_public_sources.json'
LEG = ROOT / 'research/r1/legacy_7m/legacy_evidence_catalogue.json'


def runner():
    spec = importlib.util.spec_from_file_location('r15_validator', V)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(name):
    return [json.loads(x) for x in (U / name).read_text(encoding='utf-8').splitlines()]


def contract():
    return json.loads(C.read_text(encoding='utf-8'))


def added():
    return read('ideas.jsonl')[67:85]


def sources():
    return {x['source_id'] for x in json.loads(S.read_text(encoding='utf-8'))}


def leg():
    return {x['evidence_id'] for x in json.loads(LEG.read_text(encoding='utf-8'))['evidence']}


def test_accepted_parent_is_frozen_r14():
    parent = json.loads((ROOT/'research/governance/r1_perps_trade_flow_contract_v1.json').read_text())
    assert contract()['accepted_parent'] == {'tag':'r1-perps-trade-flow-v1','contract_id':parent['contract_id']}


def test_governing_protocol_no_market_research():
    c = contract()
    assert c['protocol_version']=='1.2'
    assert c['scientific_boundaries']['new_d0_outcome_tests'] is False
    assert c['scientific_boundaries']['p1_market_data_read'] is False
    assert c['scientific_boundaries']['p1_outcome_read'] is False


def test_r14_prefix_byte_hashes_frozen():
    runner().assert_parent_prefix(contract())


def test_r14_prefix_tamper_detected(tmp_path,monkeypatch):
    m=runner()
    for n in m.PRIOR:
        (tmp_path/n).write_bytes((U/n).read_bytes())
    p=tmp_path/'ideas.jsonl';p.write_bytes(b' '+p.read_bytes())
    monkeypatch.setattr(m,'U',tmp_path)
    with pytest.raises(Exception,match='prefix'):m.assert_parent_prefix(contract())


def test_legacy_r14_registry_is_prefix_preserved():
    c=contract()
    assert all(len(read(n))>=count for n,count in runner().PRIOR.items())
    for name,n in runner().PRIOR.items():
        assert runner().frozen_prefix_hash(U/name,n)==c['immutable_r1_4_prefix_sha256'][name]


def test_no_new_family_prevents_family_splitting():
    assert len(read('families.jsonl'))==15
    assert contract()['new_family_ids']==[]
    assert {i['family_id'] for i in added()} == runner().VALID_FAMILIES


def test_exact_new_idea_ids_versions_hashes():
    a=added();c=contract()
    assert len(a)==18
    assert [x['idea_id'] for x in a]==c['new_idea_ids']
    assert all(x['version']==1 for x in a)
    assert {x['idea_id']:x['freeze_hash'] for x in a}==c['new_idea_freeze_hashes']


def test_r15_13_fresh_and_5_result_informed():
    assert Counter(i['generation_mode'] for i in added())=={'OUTCOME_BLIND':13,'LEGACY_RESULT_INFORMED':5}


def test_result_informed_ancestors_are_exact_historical_sources():
    for i in added():
        if i['generation_mode']=='LEGACY_RESULT_INFORMED':
            ev,parent=runner().CRITICAL_CONCEPTS[i['idea_id']]
            assert i['derived_from_results']==[ev]
            assert i['legacy_lineage_refs']==[ev]
            assert i['datasets_that_influenced_design']==[runner().DATASET]
            assert i['parent_idea_refs']==[parent]
    assert len(runner().CRITICAL_CONCEPTS)==5


def test_fresh_ideas_do_not_hide_result_influence():
    for i in added():
        if i['generation_mode']=='OUTCOME_BLIND':
            assert not i['derived_from_results'] and not i['legacy_lineage_refs']
            assert not i['datasets_that_influenced_design']
            assert all(not x['result_informed'] for x in i['source_refs'])


def test_legacy_negative_and_feasibility_findings_retained():
    ids={x['idea_id'] for x in added()}
    assert {'IDEA-R1-LEGACY-FORCE-SNAPSHOT-NULL','IDEA-R1-LEGACY-BASIS-CHANGE-REPLICATION-FAIL','IDEA-R1-LEGACY-GAMMA-UNSIGNED-NO-DEALER'}<=ids
    for x in added():
        if x['idea_id'] in ids and x['generation_mode']=='LEGACY_RESULT_INFORMED':
            assert any('LEG7M-' in k for k in x['derived_from_results'])


def test_all_derivative_ideas_blocked_until_governed_asof_state():
    assert all(i['infrastructure_assessment']['r2_testability']=='BLOCKED_PENDING_GOVERNED_STATE' for i in added())
    assert all(i['infrastructure_assessment']['missing_capabilities'] for i in added())


def test_capability_mapping_exact_and_no_r06_book_as_predictor():
    rules=contract()['mandatory_capability_gates']
    for i in added():
        states={s['state_family'] for s in i['causal_state_requirements'] if s['required']}
        assert i['infrastructure_assessment']['missing_capabilities'] == sorted({rules[s] for s in states if s in rules})
        if 'ORDER_BOOK' in states:
            assert 'GOVERNED_ORDER_BOOK_STATE' in i['infrastructure_assessment']['missing_capabilities']
    assert 'PERPS_TRADES' not in rules


def test_oi_cannot_claim_net_long_or_short_from_aggregate_oi():
    assert contract()['scientific_boundaries']['open_interest_does_not_identify_directional_long_short_inventory']
    assert all('long/short' in i['position_concept']['directionality'] for i in added())


def test_force_stream_cannot_be_complete_liquidation_census():
    assert contract()['scientific_boundaries']['force_order_snapshots_not_complete_liquidation_census']
    assert 'snapshot' in S.read_text().lower()
    forced=[i for i in added() if any(x['state_family']=='PERPS_FORCE_ORDERS' for x in i['causal_state_requirements'])]
    assert len(forced)>=3
    assert all('incomplete forced-order snapshots' in i['risk_concepts'][-1] for i in forced)


def test_funding_rate_event_schedule_not_future_published_rate():
    assert contract()['scientific_boundaries']['funding_publication_and_settlement_clocks_distinct']
    x=next(i for i in added() if i['idea_id']=='IDEA-R1-FUNDING-SETTLEMENT-CLOCK-CONTEXT')
    assert 'not realized future rate' in x['trigger_concept']


def test_mark_index_premium_not_guaranteed_trade_profit():
    assert contract()['scientific_boundaries']['mark_index_basis_does_not_imply_risk_free_arbitrage']
    assert all('no R1 profitability claim' in i['primary_economic_target'] for i in added())


def test_unsigned_gamma_not_signed_dealer_hedge():
    assert contract()['scientific_boundaries']['unsigned_OI_weighted_gamma_not_dealer_sign_or_hedging_flow']
    gamma=next(i for i in added() if i['idea_id']=='IDEA-R1-LEGACY-GAMMA-UNSIGNED-NO-DEALER')
    assert gamma['derived_from_results']==['LEG7M-12A']
    assert 'dealer' in ' '.join(gamma['execution_dependencies']).lower()


def test_public_sources_have_stable_verified_roster():
    assert hashlib.sha256(S.read_bytes()).hexdigest()==contract()['new_public_sources_sha256']
    assert runner().validate_sources(json.loads(S.read_text(encoding='utf-8')),contract()['new_public_source_ids'])==sources()


def test_all_new_sources_used_and_legacy_ids_grounded():
    seen={x['reference'] for i in added() for x in i['source_refs'] if x['source_type']=='THEORY_LITERATURE'}
    assert sources()==seen
    assert {'LEG7M-07','LEG7M-08','LEG7M-12A'}<=leg()


def test_public_catalogue_cannot_hide_project_outcomes():
    for x in json.loads(S.read_text(encoding='utf-8')):
        assert 'project outcome' not in x['scope_note'].lower()
        assert x['stable_reference'].startswith('https://')


def test_new_idea_freeze_mismatch_rejected():
    m=runner();records=copy.deepcopy(added());records[0]['expected_market_response']='positive always'
    with pytest.raises(Exception,match='freeze'):
        m.validate_added_ideas(records,read('ideas.jsonl')[:67],contract(),sources(),leg())


def test_cannot_forge_missing_derivative_state_as_testable():
    m=runner();records=copy.deepcopy(added());records[0]['infrastructure_assessment']['missing_capabilities']=[]
    with pytest.raises(Exception):m.validate_added_ideas(records,read('ideas.jsonl')[:67],contract(),sources(),leg())


def test_unknown_legacy_result_id_rejected():
    m=runner();records=copy.deepcopy(added());v=next(x for x in records if x['generation_mode']=='LEGACY_RESULT_INFORMED')
    v['source_refs'][-1]['reference']='LEG7M-FAKE'
    with pytest.raises(Exception):m.validate_added_ideas(records,read('ideas.jsonl')[:67],contract(),sources(),leg())


def test_source_catalogue_tamper_detected(tmp_path,monkeypatch):
    m=runner();p=tmp_path/'new_public_sources.json';p.write_bytes(S.read_bytes()+b'\n')
    monkeypatch.setattr(m,'PUBLIC_NEW',p)
    with pytest.raises(Exception,match='tampered'):m.validate_installation()


def test_universe_unchanged_r0_7_permanent_registry():
    assert (ROOT/'research/governance/registries/hypothesis_versions.jsonl').read_text().strip()==''


def test_session_v6_completes_only_derivatives_and_three_lenses_remain():
    s=read('generation_sessions.jsonl');c=contract()
    assert s[5]['version']==6 and s[5]['completed_lenses']==s[0]['planned_lenses'][:6]
    assert c['remaining_lenses']==['CROSS_VENUE','VOLATILITY_REGIME','PREDICTIVE_MODEL']
    assert s[5]['freeze_hash']==c['session_v6_freeze_hash']


def test_no_legacy_25bp_75bp_event_policy_is_selected():
    assert all(i['candidate_definition_status']=='TO_BE_DEVELOPED_IN_D0' for i in added())
    assert all('P75/15m' in i['position_concept']['horizon_concept'] for i in added())


def test_r15_validator_does_not_open_outcome_or_protected_market_paths():
    t=V.read_text(encoding='utf-8')
    assert 'parquet' not in t.lower()
    assert 'data/canonical' not in t.lower()
    assert 'future/outcomes.parquet' not in t
    assert 'p1_protected/' not in t.lower()


def test_r14_validator_prefix_still_valid_after_append():
    other=ROOT/'research/r1/trade_flow/validate_trade_flow_generation.py'
    spec=importlib.util.spec_from_file_location('r14_post_r15',other)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert m.validate_installation()['status']=='PASS'


def test_living_and_local_readmes_manifest_present():
    for f in ('research/README.md','research/r1/README.md','research/r1/derivatives/README.md','research/r1/derivatives/R1_5_IMPLEMENTATION_MANIFEST.json'):
        assert (ROOT/f).exists()
    assert 'R1.5' in (ROOT/'research/README.md').read_text()


def test_validator_real_and_self_test_pass_without_market_files():
    assert runner().validate_installation()['status']=='PASS'
    import subprocess,sys
    for args in ([],['--self-test']):
        v=subprocess.run([sys.executable,str(V),*args],cwd=ROOT,capture_output=True,text=True,timeout=30)
        assert v.returncode==0,v.stderr
        out=json.loads(v.stdout)
        assert out['new_ideas']==18 and out['permanent_hypothesis_rows']==0
        if args:assert out['self_test']=='PASS'
