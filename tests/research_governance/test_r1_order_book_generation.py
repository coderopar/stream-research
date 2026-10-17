"""R1.3 lineage, mechanism and protection regression tests (no market/outcome access)."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_core.hypothesis_universe import validate_universe, validate_idea  # noqa: E402
from research_core.registry import read_jsonl, with_freeze_hash  # noqa: E402
from research_core.registry.models import RegistryValidationError  # noqa: E402

U = ROOT / 'research/governance/universe'
CONTRACT = ROOT / 'research/governance/r1_order_book_liquidity_contract_v1.json'
SOURCES = ROOT / 'research/r1/order_book/new_public_sources.json'
VALIDATOR = ROOT / 'research/r1/order_book/validate_order_book_generation.py'
DOC = ROOT / 'research/r1/order_book/README.md'
MANIFEST = ROOT / 'research/r1/order_book/R1_3_IMPLEMENTATION_MANIFEST.json'


def module():
    spec=importlib.util.spec_from_file_location('r13_validator',VALIDATOR)
    assert spec is not None and spec.loader is not None
    m=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=m
    spec.loader.exec_module(m)
    return m


def rows():
    # R1.3-specific assertions bind to its immutable initial prefix. R1.4+
    # may append records without invalidating this acceptance snapshot.
    f,i,s=(read_jsonl(U / name) for name in ('families.jsonl','ideas.jsonl','generation_sessions.jsonl'))
    return f[:14], i[:51], s[:4]


def contract():
    return json.loads(CONTRACT.read_text(encoding='utf-8'))


def new_ideas():
    return rows()[1][35:]


def test_r13_accepted_parent_contract_is_r12():
    c=contract()
    assert c['accepted_parent']['r1_2_tag']=='r1-legacy-lineage-v1'
    assert c['accepted_parent']['r1_2_contract_id']=='R1_2_LEGACY_7M_LINEAGE_CONTRACT_V1'


def test_r13_is_generation_not_new_market_outcome_testing():
    c=contract()['scope']
    for key in ('new_market_outcome_tests','new_market_features_computed','p1_market_data_read','p1_outcome_read','permanent_hypothesis_registry_mutated','outcome_based_merit_ranking','new_strategy_thresholds_frozen'):
        assert c[key] is False


def test_full_extended_universe_satisfies_original_governance():
    validate_universe(*(read_jsonl(U / f) for f in ('families.jsonl','ideas.jsonl','generation_sessions.jsonl')))


def test_universe_roster_and_counts_are_frozen():
    f,i,s=rows()
    c=contract()
    assert [len(f),len(i),len(s)]==[14,51,4]
    assert [x['family_id'] for x in f[13:]]==c['added_family_ids']
    assert [x['idea_id'] for x in i[35:]]==c['added_idea_ids']


def test_r12_historical_prefix_is_byte_immutable():
    m=module()
    m.assert_prior_prefix(contract())
    assert all(m.frozen_prefix_sha(U/f,n)==contract()['immutable_r1_2_prefix_sha256'][f] for f,n in m.PRIOR.items())


def test_r12_prefix_tamper_is_detected(tmp_path):
    m=module(); data=(U/'ideas.jsonl').read_bytes(); p=tmp_path/'ideas.jsonl'
    p.write_bytes(data.replace(b'IDEA-R1-INFORMED-FLOW-CONTINUATION', b'IDEA-R1-INFORMED-FLOW-ALTERATION',1))
    assert m.frozen_prefix_sha(p,35)!=contract()['immutable_r1_2_prefix_sha256']['ideas.jsonl']


def test_only_new_family_is_spatial_geometry_not_duplicate_resilience():
    f,_,_=rows()
    assert f[13]['mechanism_class']=='SPATIAL_LIQUIDITY_PLACEMENT'
    assert 'depth' in f[13]['anti_family_splitting_rationale'].lower()
    assert set(x['family_id'] for x in new_ideas())<=set(x['family_id'] for x in f)


def test_r13_old_families_are_reused():
    assert len(set(x['family_id'] for x in new_ideas()))>=5
    assert sum(x['family_id']!='FAM-DEPTH-GEOMETRY-FRAGILITY' for x in new_ideas())==13


def test_completed_lens_is_order_book_only_and_generation_stays_open():
    s=rows()[2]
    assert s[-1]['version']==4 and s[-1]['status']=='IN_PROGRESS'
    assert s[-1]['completed_lenses']==contract()['completed_lenses_after']
    assert s[-1]['planned_lenses']==s[0]['planned_lenses']
    assert s[-1]['stopping_rule']==s[0]['stopping_rule']


def test_five_lenses_remain_and_cannot_be_pruned():
    s=rows()[2][-1]
    assert [x for x in s['planned_lenses'] if x not in s['completed_lenses']]==contract()['remaining_lenses']
    assert s['no_outcome_based_pruning_during_generation'] is True


def test_fresh_and_legacy_provenance_modes_are_exact():
    from collections import Counter
    assert dict(Counter(x['generation_mode'] for x in new_ideas()))=={'OUTCOME_BLIND':9,'LEGACY_RESULT_INFORMED':7}


def test_new_outcome_blind_ideas_cannot_embed_legacy_results():
    for x in new_ideas():
        if x['generation_mode']=='OUTCOME_BLIND':
            assert not x['derived_from_results'] and not x['legacy_lineage_refs'] and not x['datasets_that_influenced_design']
            assert all(y['result_informed'] is False for y in x['source_refs'])


def test_result_informed_deepening_cites_legacy_parent_results():
    for x in new_ideas():
        if x['generation_mode']=='LEGACY_RESULT_INFORMED':
            assert x['parent_idea_refs']
            assert x['derived_from_results']==x['legacy_lineage_refs']
            assert x['datasets_that_influenced_design']==[module().LEGACY_DATASET]
            assert any(y['source_type']=='LEGACY_LINEAGE' and y['result_informed'] is True for y in x['source_refs'])


def test_hidden_legacy_parent_under_outcome_blind_mode_is_rejected():
    m=module(); all_ideas=rows()[1]; bad=copy.deepcopy(new_ideas()); i=bad[0]
    i['parent_idea_refs']=['IDEA-R1-LEGACY-OPPOSING-WITHDRAWAL@1']
    bad[0]=with_freeze_hash(i)
    old_lit={x['source_id'] for x in json.loads(m.OLD_SOURCES.read_text())}
    new_lit={x['source_id'] for x in json.loads(SOURCES.read_text())}
    legacy={x['evidence_id'] for x in json.loads(m.LEGACY.read_text())['evidence']}
    with pytest.raises(RegistryValidationError,match='legacy parent ancestry'):
        m.validate_new_ideas(bad,all_ideas,contract(),old_lit|new_lit,legacy)


def test_legacy_evidence_ids_all_source_anchored():
    m=module(); ev={x['evidence_id'] for x in json.loads(m.LEGACY.read_text())['evidence']}
    assert set(contract()['legacy_evidence_ids_used']).issubset(ev)


def test_prior_negative_replacement_penetration_is_not_discarded():
    x=next(x for x in new_ideas() if x['idea_id']=='IDEA-R1-BOOK-REFILL-EXECUTION-BOUND')
    assert 'LEG7M-06' in x['derived_from_results']
    assert 'negative/inconclusive' in x['scientific_proposition'].lower()
    assert 'No presumed continuation advantage' in x['expected_market_response']


def test_all_book_ideas_blocked_and_none_false_testable_now():
    for x in new_ideas():
        assert x['infrastructure_assessment']['r2_testability']=='BLOCKED_PENDING_GOVERNED_STATE'
        assert 'GOVERNED_ORDER_BOOK_STATE' in x['infrastructure_assessment']['missing_capabilities']
        assert any(y['state_family']=='ORDER_BOOK' for y in x['causal_state_requirements'])


def test_false_r2_book_testability_fails_validator():
    x=copy.deepcopy(new_ideas()[0]);x['infrastructure_assessment']['r2_testability']='TESTABLE_NOW'
    with pytest.raises(RegistryValidationError,match='TESTABLE_NOW'):
        validate_idea(with_freeze_hash(x))


def test_no_r06_execution_book_misrepresented_as_predictor_state():
    assert contract()['scope']['r0_6_execution_book_not_predictor_state'] is True
    assert all('GOVERNED_ORDER_BOOK_STATE' in x['infrastructure_assessment']['missing_capabilities'] for x in new_ideas())
    assert 'not a predictive feature' in DOC.read_text().lower()


def test_no_order_identity_or_passive_fill_inference():
    t=DOC.read_text().lower()
    assert 'market-by-price' in t and 'order identity' in t and 'passive' in t
    assert 'queue position' in t


def test_quote_churn_does_not_claim_manipulation():
    x=next(x for x in new_ideas() if x['idea_id']=='IDEA-R1-BOOK-QUOTE-CHURN-RESILIENCE')
    assert 'not proof of spoofing or intent' in x['economic_mechanism']


def test_book_trade_attribution_requires_extra_capability():
    for name in ('CANCEL-VERSUS-TRADE-DRAIN','REFILL-EXECUTION-BOUND'):
        x=next(x for x in new_ideas() if x['idea_id'].endswith(name))
        assert 'GOVERNED_BOOK_TRADE_EVENT_ATTRIBUTION' in x['infrastructure_assessment']['missing_capabilities']


def test_source_catalogue_verifiable_and_has_no_btc_profit_claim():
    m=module()
    source=json.loads(SOURCES.read_text(encoding='utf-8'))
    assert m.validate_source_catalogue(source,contract()['added_public_source_ids'])=={'LIT-HUANG-LEHALLE-ROSENBAUM-2015'}
    assert '1312.0563' in source[0]['stable_reference']


def test_unknown_citation_is_rejected():
    m=module();bad=copy.deepcopy(new_ideas()); bad[0]['source_refs'][0]['reference']='LIT-FAKE'
    bad[0]=with_freeze_hash(bad[0]);old={x['source_id'] for x in json.loads(m.OLD_SOURCES.read_text())}
    new={x['source_id'] for x in json.loads(SOURCES.read_text())}
    lg={x['evidence_id'] for x in json.loads(m.LEGACY.read_text())['evidence']}
    with pytest.raises(RegistryValidationError,match='unverified'):
        m.validate_new_ideas(bad,rows()[1],contract(),old|new,lg)


def test_future_based_eligibility_is_forbidden():
    assert contract()['scope']['no_future_based_book_eligibility'] is True
    assert 'future' in DOC.read_text().lower()


def test_no_25bp_or_75bp_15m_strategy_is_inherited():
    assert contract()['scope']['not_an_endorsement_of_25bp_policy'] is True
    assert all('no 25bp or 75bp/15m inherited' in x['position_concept']['horizon_concept'] for x in new_ideas())


def test_p1_protection_and_no_market_outcome_open_in_validator():
    assert 'pool_id: P1_PROTECTED' in (ROOT/'research/governance/protected_data_register.yaml').read_text()
    src=VALIDATOR.read_text().lower()
    for forbidden in ('pq.read_', 'future/outcomes.parquet', 'market_events_v1/', 'data/canonical/', 'data/derived/perps/research/'):
        assert forbidden not in src


def test_no_permanent_hypothesis_registration():
    assert not read_jsonl(ROOT/'research/governance/registries/hypothesis_versions.jsonl')


def test_living_and_local_readmes_together():
    for path in (ROOT/'research/README.md',ROOT/'research/r1/README.md',ROOT/'research/governance/universe/README.md',DOC):
        assert 'R1.3' in path.read_text(encoding='utf-8')
    assert json.loads(MANIFEST.read_text(encoding='utf-8'))['expected_counts']['new_ideas']==16


def test_r12_validator_still_passes_on_frozen_historical_prefix():
    spec=importlib.util.spec_from_file_location('r12_validator_for_r13',ROOT/'research/r1/legacy_7m/validate_legacy_lineage.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    assert m.validate_installation()['status']=='PASS'


def test_validator_pass_and_self_test_without_market_files():
    assert module().validate_installation()['status']=='PASS'
    from subprocess import run
    p=run([sys.executable,str(VALIDATOR),'--self-test'],capture_output=True,text=True,cwd=ROOT)
    assert p.returncode==0,p.stderr
    assert json.loads(p.stdout)['self_test']=='PASS'
