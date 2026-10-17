"""Govern R1.4's append-only PERPS_TRADE_FLOW idea generation and causal capability claims."""
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[2]
U=ROOT/'research/governance/universe'
C=ROOT/'research/governance/r1_perps_trade_flow_contract_v1.json'
V=ROOT/'research/r1/trade_flow/validate_trade_flow_generation.py'
S=ROOT/'research/r1/trade_flow/new_public_sources.json'


def runner():
    spec=importlib.util.spec_from_file_location('r14_validator',V)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def read(n):
    return [json.loads(x) for x in (U/n).read_text(encoding='utf-8').splitlines()]


def contract():
    return json.loads(C.read_text(encoding='utf-8'))


def new_ideas():
    return read('ideas.jsonl')[51:67]


def test_r14_parent_contract_and_parent_source_hash():
    c=contract()
    parent=ROOT/'research/governance/r1_order_book_liquidity_contract_v1.json'
    assert c['accepted_parent']['r1_3_tag']=='r1-order-book-liquidity-v1'
    assert c['accepted_parent']['r1_3_contract_sha256']==hashlib.sha256(parent.read_bytes()).hexdigest()


def test_new_lens_cannot_silently_use_market_or_protected_outcomes():
    forbidden=('new_market_outcome_tests','new_market_features_computed','d0_outcome_files_opened','p1_market_data_read',
               'p1_outcome_read','permanent_hypothesis_registry_mutated','outcome_based_merit_ranking','new_strategy_thresholds_frozen')
    assert all(contract()['scope'][x] is False for x in forbidden)


def test_r14_frozen_r13_prefix_byte_hash():
    m=runner()
    m.assert_parent_prefix(contract())


def test_r13_prefix_tamper_detected(tmp_path,monkeypatch):
    m=runner(); tmp_path.mkdir(exist_ok=True)
    for name in m.PRIOR_COUNTS:
        (tmp_path/name).write_bytes((U/name).read_bytes())
    original=(tmp_path/'ideas.jsonl').read_bytes()
    (tmp_path/'ideas.jsonl').write_bytes(b' '+original)
    monkeypatch.setattr(m,'U',tmp_path)
    with pytest.raises(Exception,match='prefix'):
        m.assert_parent_prefix(contract())


def test_r14_own_idea_freeze_hashes_against_contract():
    frozen=contract()['added_idea_freeze_hashes']
    assert {x['idea_id']:x['freeze_hash'] for x in new_ideas()}==frozen


def test_r14_idea_roster_order_and_versions():
    assert [x['idea_id'] for x in new_ideas()]==contract()['added_idea_ids']
    assert all(x['version']==1 for x in new_ideas())


def test_r14_one_genuinely_distinct_family_and_no_variant_splitting():
    fam=read('families.jsonl')
    assert len(fam)>=15
    assert fam[14]['family_id']=='FAM-PERP-TRADE-ARRIVAL-TEMPO'
    assert fam[14]['mechanism_class']=='MARKETABLE_TRADE_ARRIVAL_ACTIVITY'
    assert 'window' in str(fam[14]['variant_dimensions_kept_in_family'])
    assert fam[14]['freeze_hash']==contract()['added_family_freeze_hash']


def test_r14_reuses_prior_metaorder_price_discovery_impact_families():
    family_counts=Counter(x['family_id'] for x in new_ideas())
    assert set(family_counts)=={'FAM-METAORDER-FLOW-PERSISTENCE','FAM-INFORMED-FLOW-PRICE-DISCOVERY',
                                'FAM-IMPACT-EFFICIENCY-ABSORPTION','FAM-PERP-TRADE-ARRIVAL-TEMPO'}


def test_r14_provenance_modes_exact_and_hidden_legacy_forbidden():
    assert Counter(x['generation_mode'] for x in new_ideas())=={'OUTCOME_BLIND':15,'LEGACY_RESULT_INFORMED':1}
    for idea in new_ideas():
        if idea['generation_mode']=='OUTCOME_BLIND':
            assert not idea['derived_from_results'] and not idea['legacy_lineage_refs']
            assert not idea['datasets_that_influenced_design']
            assert all(not s['result_informed'] for s in idea['source_refs'])


def test_r14_failed_replication_retained_not_promoted_to_credible_alpha():
    x=[x for x in new_ideas() if x['generation_mode']=='LEGACY_RESULT_INFORMED'][0]
    assert x['idea_id']=='IDEA-R1-LEGACY-TRADE-SIZE-HHI-FAILED-REPLICATION'
    assert x['derived_from_results']==['LEG7M-07']
    assert x['parent_idea_refs']==['IDEA-R1-LEGACY-AGGRESSIVE-TRADE-CONCENTRATION@1']
    assert 'failed discovery/validation replication' in x['economic_mechanism']
    assert x['infrastructure_assessment']['r2_testability']=='BLOCKED_PENDING_GOVERNED_STATE'


def test_r14_every_idea_uses_perps_trades_not_derivatives_or_book_state():
    assert all(any(y['state_family']=='PERPS_TRADES' and y['required'] for y in x['causal_state_requirements']) for x in new_ideas())
    assert all(not {'ORDER_BOOK','PERPS_DERIVATIVES','SPOT_CROSS_VENUE','OPTIONS'} & {r['state_family'] for r in x['causal_state_requirements']} for x in new_ideas())


def test_r14_twelve_structurally_available_four_blocked():
    assert Counter(x['infrastructure_assessment']['r2_testability'] for x in new_ideas())=={
      'TESTABLE_NOW':12,'BLOCKED_PENDING_GOVERNED_STATE':4}


def test_testable_now_requires_only_governed_bucket_fields():
    for x in new_ideas():
        if x['infrastructure_assessment']['r2_testability']=='TESTABLE_NOW':
            assert not x['infrastructure_assessment']['missing_capabilities']
            assert all(req['state_family']=='PERPS_TRADES' for req in x['causal_state_requirements'])
            assert 'R0.5' in x['infrastructure_assessment']['assessment_basis']


def test_subsecond_arrival_intensity_is_explicitly_blocked():
    x=next(x for x in new_ideas() if x['idea_id']=='IDEA-R1-INTERARRIVAL-SELF-EXCITATION')
    assert x['infrastructure_assessment']['missing_capabilities']==['GOVERNED_PERPS_TRADE_EVENTS']
    assert 'individual' in x['causal_state_requirements'][0]['role']


def test_individual_print_tail_does_not_fake_reconstruction_from_aggregate():
    x=next(x for x in new_ideas() if x['idea_id']=='IDEA-R1-INDIVIDUAL-PRINT-TAIL-SKEW')
    assert x['infrastructure_assessment']['missing_capabilities']==['GOVERNED_PERPS_TRADE_EVENTS']


def test_historical_price_path_is_not_inferred_from_one_latest_trade_price():
    x=next(x for x in new_ideas() if x['idea_id']=='IDEA-R1-PRICE-RESPONSE-FLOW-DECOUPLING')
    assert x['infrastructure_assessment']['missing_capabilities']==['GOVERNED_PERPS_PRICE_PATH_STATE']
    assert any(y['state_family']=='PERPS_PRICE_PATH' for y in x['causal_state_requirements'])


def test_r05_trade_rows_not_unique_order_or_trader_identity():
    assert contract()['state_capability_rules']['bucket_row_count_is_not_unique_orders_or_participants']
    for x in new_ideas():
        assert any('trade_count is canonical feed row count' in v for v in x['execution_dependencies'])


def test_buyer_is_maker_side_not_assumed_unverified_buy_sell():
    assert contract()['state_capability_rules']['raw_maker_flag_mapping_must_be_verified_in_d0']
    for x in new_ideas():
        assert any('buyer_is_maker semantics must be verified' in v for v in x['execution_dependencies'])


def test_r05_bucket_grids_not_pseudoscientific_frozen_windows():
    assert contract()['state_capability_rules']['r0_5_trade_storage_grid_not_scientific_feature_definition']
    assert all(x['candidate_definition_status']=='TO_BE_DEVELOPED_IN_D0' for x in new_ideas())


def test_no_25bp_75bp_or_15min_event_policy_inherited():
    assert contract()['scope']['no_parent_25bp_candidate_policy_endorsement']
    assert all('25bp' not in x['trigger_concept'].lower() for x in new_ideas())


def test_added_public_research_source_sha_and_roster():
    c=contract()
    assert hashlib.sha256(S.read_bytes()).hexdigest()==c['new_public_sources_sha256']
    assert {x['source_id'] for x in json.loads(S.read_text(encoding='utf-8'))}==set(c['added_public_source_ids'])


def test_public_literature_does_not_claim_btcusdt_returns():
    contents=S.read_text(encoding='utf-8').lower()
    assert 'no vpIN implementation'.lower() in contents
    assert 'no fitted hawkes model' in contents


def test_external_source_refs_are_public_and_have_matching_catalogue_ids():
    m=runner()
    records=json.loads(S.read_text(encoding='utf-8'))
    assert m.validate_public_sources(records,contract()['added_public_source_ids'])==set(contract()['added_public_source_ids'])


def test_validator_detects_faked_testable_now_in_fixture():
    m=runner();c=contract(); allideas=read('ideas.jsonl')
    mutated=json.loads(json.dumps(allideas[51:67]))
    x=next(x for x in mutated if x['idea_id']=='IDEA-R1-INTERARRIVAL-SELF-EXCITATION')
    x['infrastructure_assessment']['missing_capabilities']=[]
    x['infrastructure_assessment']['r2_testability']='TESTABLE_NOW'
    payload={k:v for k,v in x.items() if k!='freeze_hash'}
    x['freeze_hash']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')).hexdigest()
    pub={v['source_id'] for v in json.loads(S.read_text(encoding='utf-8'))}
    pub.update({v['source_id'] for v in json.loads((ROOT/'research/r1/theory_mechanisms/literature_sources.json').read_text(encoding='utf-8'))})
    leg={v['evidence_id'] for v in json.loads((ROOT/'research/r1/legacy_7m/legacy_evidence_catalogue.json').read_text(encoding='utf-8'))['evidence']}
    with pytest.raises(Exception,match='frozen idea'):
        m.validate_new_ideas(mutated,allideas,c,pub,leg)


def test_validator_detects_unknown_external_source():
    m=runner();records=json.loads(S.read_text(encoding='utf-8'))
    records[0]['source_id']='LIT-UNKNOWN-2030'
    with pytest.raises(Exception):
        m.validate_public_sources(records,contract()['added_public_source_ids'])


def test_validator_detects_source_catalogue_tamper(monkeypatch,tmp_path):
    m=runner()
    broken=tmp_path/'new_public_sources.json'
    broken.write_bytes(S.read_bytes()+b'\n')
    monkeypatch.setattr(m,'PUBLIC_NEW',broken)
    with pytest.raises(Exception,match='literature source catalogue altered'):
        m.validate_installation()


def test_r14_sessional_append_only_and_remaining_lenses():
    s=read('generation_sessions.jsonl'); c=contract()
    assert len(s)>=5 and s[4]['version']==5 and s[4]['status']=='IN_PROGRESS'
    assert s[4]['completed_lenses']==c['completed_lenses_after']
    assert c['remaining_lenses']==['DERIVATIVES_POSITIONING','CROSS_VENUE','VOLATILITY_REGIME','PREDICTIVE_MODEL']
    assert s[4]['freeze_hash']==c['generation_session_v5_freeze_hash']


def test_r14_full_governance_validates_and_no_permanent_hypotheses():
    m=runner(); result=m.validate_installation()
    assert result['total_families']==15 and result['total_ideas']==67
    assert result['permanent_hypothesis_rows']==0


def test_r13_validator_remains_compatible_with_legitimate_append():
    src=ROOT/'research/r1/order_book/validate_order_book_generation.py'
    spec=importlib.util.spec_from_file_location('r13_validator_post_append',src)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert m.validate_installation()['total_ideas']==51


def test_r14_docs_present_for_all_research_steps():
    assert (ROOT/'research/r1/trade_flow/README.md').exists()
    assert 'R1.4' in (ROOT/'research/README.md').read_text(encoding='utf-8')
    assert 'R1.4' in (ROOT/'research/r1/README.md').read_text(encoding='utf-8')


def test_r14_manifest_guarantees_no_new_outcome_tests_or_p1():
    m=json.loads((ROOT/'research/r1/trade_flow/R1_4_IMPLEMENTATION_MANIFEST.json').read_text())
    assert not m['research_boundary']['new_market_outcome_tests']
    assert not m['research_boundary']['d0_outcome_files_opened']
    assert not m['research_boundary']['p1_market_data_read']
    assert not m['research_boundary']['p1_outcomes_read']


def test_validator_does_not_open_market_parquet_or_p1_sources():
    code=V.read_text(encoding='utf-8').lower()
    assert 'read_parquet' not in code and 'pq.read_table' not in code
    assert 'data/canonical' not in code and 'data/derived/perps/research' not in code


def test_validator_self_test_and_installation_without_market_files():
    assert runner().validate_installation()['status']=='PASS'
    assert runner().validate_installation()['p1_status']=='SEALED_OUTCOME_UNSEEN'
