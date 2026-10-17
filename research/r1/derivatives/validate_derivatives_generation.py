"""Validate R1.5 derivatives and positioning idea generation. No market/outcome reads."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from research_core.hypothesis_universe import (  # noqa: E402
    assert_generation_not_testing, assert_no_protected_influence,
    assert_testability_separate_from_merit, validate_universe,
)
from research_core.registry import read_jsonl, verify_freeze_hash  # noqa: E402
from research_core.registry.models import RegistryValidationError  # noqa: E402

U = ROOT / 'research/governance/universe'
C = ROOT / 'research/governance/r1_derivatives_positioning_contract_v1.json'
PARENT = ROOT / 'research/governance/r1_perps_trade_flow_contract_v1.json'
PUBLIC_NEW = ROOT / 'research/r1/derivatives/new_public_sources.json'
LEGACY = ROOT / 'research/r1/legacy_7m/legacy_evidence_catalogue.json'
PERMANENT = ROOT / 'research/governance/registries/hypothesis_versions.jsonl'
DATASET = 'LEGACY_D0_2026-05-18_TO_2026-07-18__PHASE_7M_ARCHIVED_REPORTS'
PRIOR = {'families.jsonl': 15, 'ideas.jsonl': 67, 'generation_sessions.jsonl': 5}
VALID_FAMILIES = {
    'FAM-PERP-POSITIONING-FORCED-FLOW', 'FAM-PERP-BASIS-DISLOCATION',
    'FAM-FUNDING-CROWDING-REGIME', 'FAM-OPTIONS-SURFACE-RISK-REPRICING',
    'FAM-OPTIONS-UNSIGNED-GAMMA-GEOMETRY',
}
CRITICAL_CONCEPTS = {
    'IDEA-R1-LEGACY-OI-EXPOSURE-REINTERPRETATION': ('LEG7M-07', 'IDEA-R1-LEGACY-OI-EXPANSION@1'),
    'IDEA-R1-LEGACY-FORCE-SNAPSHOT-NULL': ('LEG7M-07', 'IDEA-R1-LEGACY-FORCE-ORDER-PRESENCE@1'),
    'IDEA-R1-LEGACY-COUNTER-FUNDING-UNCERTAIN': ('LEG7M-08', 'IDEA-R1-LEGACY-COUNTER-FUNDING@1'),
    'IDEA-R1-LEGACY-BASIS-CHANGE-REPLICATION-FAIL': ('LEG7M-08', 'IDEA-R1-LEGACY-BASIS-CHANGE@1'),
    'IDEA-R1-LEGACY-GAMMA-UNSIGNED-NO-DEALER': ('LEG7M-12A', 'IDEA-R1-LEGACY-OPTIONS-GAMMA-GEOMETRY@1'),
}


def frozen_prefix_hash(path: Path, count: int) -> str:
    rows = path.read_bytes().splitlines(keepends=True)
    if len(rows) < count:
        raise RegistryValidationError('R1.4 frozen prefix truncated: ' + path.name)
    return hashlib.sha256(b''.join(rows[:count])).hexdigest()


def assert_parent_prefix(contract: dict) -> None:
    for name, count in PRIOR.items():
        if frozen_prefix_hash(U / name, count) != contract['immutable_r1_4_prefix_sha256'][name]:
            raise RegistryValidationError('R1.4 frozen prefix rewritten: ' + name)


def validate_sources(records: list[dict], expected: list[str]) -> set[str]:
    fields = {'source_id','title','authors','year','source_kind','stable_reference','supported_mechanism_claims','scope_note'}
    if not isinstance(records, list) or len(records) != len(expected):
        raise RegistryValidationError('R1.5 public source roster changed')
    ids = set()
    for r in records:
        if not isinstance(r, dict) or set(r) != fields:
            raise RegistryValidationError('public source fields not governed')
        sid = r['source_id']
        if not isinstance(sid, str) or sid in ids or not sid.startswith('LIT-'):
            raise RegistryValidationError('invalid or duplicate source ID')
        ids.add(sid)
        if not isinstance(r['year'], int) or r['year'] < 1900 or r['year'] > 2026:
            raise RegistryValidationError('invalid public source year')
        if not isinstance(r['authors'], list) or not r['authors'] or any(not isinstance(v, str) or not v for v in r['authors']):
            raise RegistryValidationError('public source authors absent')
        if not isinstance(r['stable_reference'], str) or not r['stable_reference'].startswith('https://'):
            raise RegistryValidationError('public source must use HTTPS reference')
        if any(not isinstance(r[v], str) or not r[v].strip() for v in ('title','source_kind','scope_note')):
            raise RegistryValidationError('public source text missing')
        if not isinstance(r['supported_mechanism_claims'], list) or not r['supported_mechanism_claims']:
            raise RegistryValidationError('public source needs qualitative claims')
        if any(not isinstance(c, str) or not c.strip() for c in r['supported_mechanism_claims']):
            raise RegistryValidationError('blank qualitative claim')
        if any(w in json.dumps(r).lower() for w in ('d0 outcome result','p1 outcome','project backtest result','btc alpha demonstrated')):
            raise RegistryValidationError('public catalogue cannot import protected/project outcome claims')
    if ids != set(expected):
        raise RegistryValidationError('public source IDs differ from frozen contract')
    return ids


def validate_added_ideas(new: list[dict], prior: list[dict], contract: dict,
                         sources: set[str], legacy: set[str]) -> None:
    if len(new) != 18 or [i['idea_id'] for i in new] != contract['new_idea_ids']:
        raise RegistryValidationError('R1.5 new idea order/roster changed')
    if len(set(contract['new_idea_ids'])) != 18 or any(i['version'] != 1 for i in new):
        raise RegistryValidationError('R1.5 first-versions/IDs must be unique')
    if dict(Counter(i['generation_mode'] for i in new)) != contract['generation_mode_counts']:
        raise RegistryValidationError('R1.5 outcome-blind versus legacy provenance mix changed')
    previous = {f"{i['idea_id']}@{i['version']}": i for i in prior}
    previous_ids = set(previous)
    covered_sources = set()
    leg_seen = set()
    for i in new:
        verify_freeze_hash(i)
        if i['freeze_hash'] != contract['new_idea_freeze_hashes'][i['idea_id']]:
            raise RegistryValidationError('R1.5 frozen idea changed')
        assert_generation_not_testing(i)
        assert_no_protected_influence(i)
        assert_testability_separate_from_merit(i)
        if i['family_id'] not in VALID_FAMILIES or i['candidate_definition_status'] != 'TO_BE_DEVELOPED_IN_D0':
            raise RegistryValidationError('R1.5 family splitting or inherited legacy candidate policy')
        if i['parent_hypothesis_ids']:
            raise RegistryValidationError('R1.5 cannot register permanent HYP-* records')
        for ref in i['parent_idea_refs']:
            if ref not in previous_ids:
                raise RegistryValidationError('R1.5 genealogy references unknown/nonprior idea')
            if i['generation_mode'] == 'OUTCOME_BLIND' and previous[ref]['generation_mode'] != 'OUTCOME_BLIND':
                raise RegistryValidationError('R1.5 outcome-blind idea conceals result-informed parent')
        missing = i['infrastructure_assessment']['missing_capabilities']
        if i['infrastructure_assessment']['r2_testability'] != 'BLOCKED_PENDING_GOVERNED_STATE' or not missing:
            raise RegistryValidationError('R1.5 derivatives cannot claim structurally TESTABLE_NOW')
        required_states = {r['state_family'] for r in i['causal_state_requirements'] if r['required']}
        if not required_states or not required_states & set(contract['mandatory_capability_gates']):
            raise RegistryValidationError('R1.5 lacks a governed required derivative state')
        expected_missing = sorted({contract['mandatory_capability_gates'][s] for s in required_states if s in contract['mandatory_capability_gates']})
        if missing != expected_missing:
            raise RegistryValidationError('R1.5 missing required derivative causal predictor capabilities')
        refs = i['source_refs']
        if len({(x['source_type'],x['reference']) for x in refs}) != len(refs):
            raise RegistryValidationError('duplicate provenance references')
        result_refs = set()
        for r in refs:
            if r['source_type'] == 'THEORY_LITERATURE':
                if r['source_zone'] != 'PUBLIC' or r['result_informed'] or r['reference'] not in sources:
                    raise RegistryValidationError('unknown/nonpublic source citation')
                covered_sources.add(r['reference'])
            elif r['source_type'] == 'MARKET_MECHANISM':
                if r['source_zone'] != 'NONE' or r['result_informed']:
                    raise RegistryValidationError('false result-informing market mechanism')
            elif r['source_type'] == 'LEGACY_LINEAGE':
                if r['source_zone'] != 'LEGACY_D0' or r['result_informed'] is not True or r['reference'] not in legacy:
                    raise RegistryValidationError('unanchored legacy result influence')
                result_refs.add(r['reference'])
            else:
                raise RegistryValidationError('R1.5 unexpected source type')
        if i['generation_mode'] == 'LEGACY_RESULT_INFORMED':
            if i['idea_id'] not in CRITICAL_CONCEPTS:
                raise RegistryValidationError('undocumented legacy-result-informed mechanism')
            ev, parent = CRITICAL_CONCEPTS[i['idea_id']]
            if result_refs != {ev} or i['legacy_lineage_refs'] != [ev] or i['derived_from_results'] != [ev]:
                raise RegistryValidationError('legacy evidence results silently lost/replaced')
            if i['datasets_that_influenced_design'] != [DATASET] or i['parent_idea_refs'] != [parent]:
                raise RegistryValidationError('legacy dataset or idea ancestry missing')
            leg_seen |= result_refs
        else:
            if result_refs or i['legacy_lineage_refs'] or i['derived_from_results'] or i['datasets_that_influenced_design']:
                raise RegistryValidationError('outcome-blind idea hides result-informed ancestry')
    if set(sources) != covered_sources or leg_seen != {'LEG7M-07','LEG7M-08','LEG7M-12A'}:
        raise RegistryValidationError('R1.5 public/legacy provenance source coverage is incomplete')


def validate_installation() -> dict:
    c = json.loads(C.read_text(encoding='utf-8'))
    parent = json.loads(PARENT.read_text(encoding='utf-8'))
    if c['contract_id'] != 'R1_5_DERIVATIVES_POSITIONING_GENERATION_V1' or c['accepted_parent'] != {
        'tag': 'r1-perps-trade-flow-v1', 'contract_id': parent['contract_id']
    }:
        raise RegistryValidationError('R1.5 parent contract changed')
    if c['new_family_ids'] or c['protocol_version'] != '1.2' or len(c['new_idea_ids']) != 18:
        raise RegistryValidationError('R1.5 forbidden extra family or incorrect research scope')
    assert_parent_prefix(c)
    family, ideas, sessions = (read_jsonl(U / name) for name in PRIOR)
    validate_universe(family, ideas, sessions)
    if len(family) < 15 or len(ideas) < 85 or len(sessions) < 6:
        raise RegistryValidationError('R1.5 full append-only universe truncated')
    # Later R1.6+ stages may append their own families; R1.5 adds none.
    # The byte-frozen first 15 family records are checked above.
    if hashlib.sha256(PUBLIC_NEW.read_bytes()).hexdigest() != c['new_public_sources_sha256']:
        raise RegistryValidationError('R1.5 public source catalogue tampered')
    public = validate_sources(json.loads(PUBLIC_NEW.read_text(encoding='utf-8')),c['new_public_source_ids'])
    leg = {e['evidence_id'] for e in json.loads(LEGACY.read_text(encoding='utf-8'))['evidence']}
    validate_added_ideas(ideas[67:85],ideas[:67],c,public,leg)
    session = sessions[5]
    if session['version'] != 6 or session['status'] != 'IN_PROGRESS' or session['freeze_hash'] != c['session_v6_freeze_hash']:
        raise RegistryValidationError('R1.5 generation session v6 invalid')
    if session['completed_lenses'] != c['completed_lenses_after'] or session['completed_lenses'] != sessions[0]['planned_lenses'][:6]:
        raise RegistryValidationError('DERIVATIVES_POSITIONING lens not completed in the frozen planned order')
    if session['planned_lenses'] != sessions[0]['planned_lenses'] or session['stopping_rule'] != sessions[0]['stopping_rule']:
        raise RegistryValidationError('R1.5 changed frozen research stopping rule')
    if c['remaining_lenses'] != sessions[0]['planned_lenses'][6:]:
        raise RegistryValidationError('R1.5 pruned a future generation lens')
    if any(v is not False for k,v in c['scientific_boundaries'].items() if k in {
        'R0_5_derivatives_predictor_state_exists','new_d0_outcome_tests','p1_market_data_read','p1_outcome_read',
        'permanent_hypothesis_registry_mutated','new_25bp_candidate_or_P75_15m_event_policy_inherited',
    }):
        raise RegistryValidationError('R1.5 protection or missing-capability declaration violated')
    if any(c['scientific_boundaries'][k] is not True for k in (
        'all_new_ideas_blocked_pending_governed_state',
        'open_interest_does_not_identify_directional_long_short_inventory',
        'force_order_snapshots_not_complete_liquidation_census',
        'funding_publication_and_settlement_clocks_distinct',
        'mark_index_basis_does_not_imply_risk_free_arbitrage',
        'unsigned_OI_weighted_gamma_not_dealer_sign_or_hedging_flow',
    )):
        raise RegistryValidationError('R1.5 derivative inference boundary removed')
    if read_jsonl(PERMANENT):
        raise RegistryValidationError('R1.5 cannot mutate permanent hypothesis registry')
    return {
        'status':'PASS','protocol_version':'1.2','accepted_parent':c['accepted_parent'],
        'prior_families':15,'new_families':0,'total_families':15,
        'prior_ideas':67,'new_ideas':18,'total_ideas':85,
        'generation_mode_counts':dict(Counter(i['generation_mode'] for i in ideas[67:85])),
        'generation_session_version':6,'completed_lenses':c['completed_lenses_after'],
        'remaining_lenses':c['remaining_lenses'],'capability_blocked_derivative_ideas':18,
        'public_sources':len(public),'permanent_hypothesis_rows':0,
        'new_d0_outcome_tests':False,'p1_market_data_read':False,'p1_outcome_read':False,
        'p1_status':'SEALED_OUTCOME_UNSEEN',
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='R1.5 derivatives and positioning lens governance validator')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    result = validate_installation()
    if args.self_test:
        assert result['new_ideas'] == 18 and result['capability_blocked_derivative_ideas'] == 18
        result['self_test'] = 'PASS'
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__ == '__main__':
    main()
