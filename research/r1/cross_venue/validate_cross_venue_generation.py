"""Validate R1.6 cross-venue hypothesis generation without opening market/outcome data."""
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
C = ROOT / 'research/governance/r1_cross_venue_contract_v1.json'
PARENT = ROOT / 'research/governance/r1_derivatives_positioning_contract_v1.json'
PUBLIC_NEW = ROOT / 'research/r1/cross_venue/new_public_sources.json'
PREVIOUS_PUBLIC = ROOT / 'research/r1/theory_mechanisms/literature_sources.json'
LEGACY = ROOT / 'research/r1/legacy_7m/legacy_evidence_catalogue.json'
PERMANENT = ROOT / 'research/governance/registries/hypothesis_versions.jsonl'
DATASET = 'LEGACY_D0_2026-05-18_TO_2026-07-18__PHASE_7M_ARCHIVED_REPORTS'
PRIOR = {'families.jsonl': 15, 'ideas.jsonl': 85, 'generation_sessions.jsonl': 6}
VALID_FAMILIES = {'FAM-CROSS-VENUE-INFORMATION-DIFFUSION', 'FAM-CROSS-VENUE-LIQUIDITY-SEGMENTATION'}
SOURCE_FIELDS = {'source_id','title','authors','year','source_kind','stable_reference','supported_mechanism_claims','scope_note'}


def frozen_prefix_hash(path: Path, count: int) -> str:
    lines = path.read_bytes().splitlines(keepends=True)
    if len(lines) < count:
        raise RegistryValidationError('R1.5 frozen prefix truncated: ' + path.name)
    return hashlib.sha256(b''.join(lines[:count])).hexdigest()


def assert_parent_prefix(c: dict) -> None:
    for name, n in PRIOR.items():
        if frozen_prefix_hash(U / name, n) != c['immutable_r1_5_prefix_sha256'][name]:
            raise RegistryValidationError('R1.5 frozen prefix rewritten: ' + name)


def validate_sources(records: list[dict], expected: list[str]) -> set[str]:
    if not isinstance(records,list) or len(records) != len(expected):
        raise RegistryValidationError('R1.6 public source roster changed')
    ids = set()
    for r in records:
        if not isinstance(r,dict) or set(r) != SOURCE_FIELDS:
            raise RegistryValidationError('R1.6 public source fields not governed')
        sid = r['source_id']
        if not isinstance(sid,str) or sid in ids or not sid.startswith('LIT-'):
            raise RegistryValidationError('R1.6 duplicate/invalid public source')
        ids.add(sid)
        if not isinstance(r['stable_reference'],str) or not r['stable_reference'].startswith('https://'):
            raise RegistryValidationError('public source reference must be HTTPS')
        if not isinstance(r['year'],int) or isinstance(r['year'],bool) or not 1900 <= r['year'] <= 2026:
            raise RegistryValidationError('invalid public source year')
        if not isinstance(r['authors'],list) or not r['authors'] or not all(isinstance(x,str) and x for x in r['authors']):
            raise RegistryValidationError('public source authors missing')
        if not all(isinstance(r[x],str) and r[x].strip() for x in ('title','source_kind','scope_note')):
            raise RegistryValidationError('empty source field')
        if not isinstance(r['supported_mechanism_claims'],list) or not r['supported_mechanism_claims'] or not all(isinstance(x,str) and x.strip() for x in r['supported_mechanism_claims']):
            raise RegistryValidationError('public source qualitative claims absent')
        if any(x in json.dumps(r).lower() for x in ('p1 outcome','d0 outcome result','positive btcusdt return','backtested alpha')):
            raise RegistryValidationError('outcome content in purported public source')
    if set(expected) != ids:
        raise RegistryValidationError('public source ID roster mismatch')
    return ids


def validate_added_ideas(rows: list[dict], prior: list[dict], c: dict, public_ids: set[str], legacy_ids: set[str]) -> None:
    if len(rows) != 16 or [x['idea_id'] for x in rows] != c['new_idea_ids']:
        raise RegistryValidationError('R1.6 idea roster mismatch')
    counts = Counter()
    seen_sources, seen_legacy = set(), set()
    known = {f"{x['idea_id']}@{x['version']}" for x in prior}
    for i in rows:
        verify_freeze_hash(i)
        if i['freeze_hash'] != c['new_idea_freeze_hashes'][i['idea_id']] or i['version'] != 1:
            raise RegistryValidationError('R1.6 idea freeze hash or version mismatch')
        if i['family_id'] not in VALID_FAMILIES:
            raise RegistryValidationError('cross-venue idea split into ungoverned family')
        if i['candidate_definition_status'] != 'TO_BE_DEVELOPED_IN_D0':
            raise RegistryValidationError('inherited or prematurely frozen candidate definition')
        assert_generation_not_testing(i)
        assert_no_protected_influence(i)
        assert_testability_separate_from_merit(i)
        states = {x['state_family'] for x in i['causal_state_requirements'] if x['required']}
        if not states or not {'BINANCE_SPOT_TRADES','BINANCE_SPOT_BBO','OKX_SPOT_BBO'}.intersection(states):
            raise RegistryValidationError('cross-venue idea missing observable non-perp state')
        missing = {c['mandatory_capability_gates'][x] for x in states if x in c['mandatory_capability_gates']}
        if i['idea_id'] == 'IDEA-R1-CROSS-CROSS-VENUE-EXECUTION-BAND':
            missing.add(c['additional_execution_gate'])
            if i['infrastructure_assessment']['r2_testability'] != 'BLOCKED_PENDING_STATE_AND_EXECUTION':
                raise RegistryValidationError('multi-venue executable-arbitrage simulation falsely available')
        elif i['infrastructure_assessment']['r2_testability'] != 'BLOCKED_PENDING_GOVERNED_STATE':
            raise RegistryValidationError('R0.6 execution book misrepresented as cross-venue predictor')
        if set(i['infrastructure_assessment']['missing_capabilities']) != missing or not missing:
            raise RegistryValidationError('cross-venue capability map mismatch')
        text = ' '.join([i['scientific_proposition'],i['economic_mechanism'],i['primary_economic_target'],*i['risk_concepts']]).lower()
        if 'no r1 profitability claim' not in text or 'quote' not in text:
            raise RegistryValidationError('idea omitted quote-quality/economic no-claim boundary')
        if 'P75/15m' not in i['position_concept']['horizon_concept']:
            raise RegistryValidationError('legacy horizon not explicitly unselected')
        result_refs = set()
        for ref in i['source_refs']:
            if ref['source_type'] == 'THEORY_LITERATURE':
                if ref['source_zone'] != 'PUBLIC' or ref['result_informed'] or ref['reference'] not in public_ids:
                    raise RegistryValidationError('unknown/false cross-venue public source')
                seen_sources.add(ref['reference'])
            elif ref['source_type'] == 'MARKET_MECHANISM':
                if ref['source_zone'] != 'NONE' or ref['result_informed']:
                    raise RegistryValidationError('first-principles incorrectly result-informed')
            elif ref['source_type'] == 'LEGACY_LINEAGE':
                if ref['source_zone'] != 'LEGACY_D0' or not ref['result_informed'] or ref['reference'] not in legacy_ids:
                    raise RegistryValidationError('ungrounded legacy source')
                result_refs.add(ref['reference'])
            else:
                raise RegistryValidationError('unexpected cross-venue source type')
        if i['generation_mode'] == 'LEGACY_RESULT_INFORMED':
            spec = c['legacy_parent_links'].get(i['idea_id'])
            if not spec or result_refs != {spec['evidence_id']} or i['derived_from_results'] != [spec['evidence_id']] or i['legacy_lineage_refs'] != [spec['evidence_id']]:
                raise RegistryValidationError('legacy cross-venue outcomes hidden or changed')
            if i['datasets_that_influenced_design'] != [DATASET] or i['parent_idea_refs'] != [spec['parent_idea_ref']]:
                raise RegistryValidationError('legacy cross-venue parent/dataset missing')
            if spec['parent_idea_ref'] not in known:
                raise RegistryValidationError('unknown parent idea reference')
            seen_legacy |= result_refs
        else:
            if result_refs or i['derived_from_results'] or i['legacy_lineage_refs'] or i['datasets_that_influenced_design'] or i['parent_idea_refs']:
                raise RegistryValidationError('outcome-blind cross-venue idea hides result influence')
        counts[i['generation_mode']] += 1
    if dict(counts) != c['generation_mode_counts'] or seen_legacy != {'LEG7M-09','LEG7M-10'}:
        raise RegistryValidationError('R1.6 provenance counts or historical null branches changed')
    # All sources in the new catalogue must be used; existing Makarov/Schoar source is also used.
    if seen_sources != public_ids:
        raise RegistryValidationError('incomplete cross-venue literature source coverage')


def validate_installation() -> dict:
    c = json.loads(C.read_text(encoding='utf-8'))
    parent = json.loads(PARENT.read_text(encoding='utf-8'))
    if c['contract_id'] != 'R1_6_CROSS_VENUE_GENERATION_V1' or c['protocol_version'] != '1.2' or c['accepted_parent'] != {'tag':'r1-derivatives-positioning-v1','contract_id':parent['contract_id']}:
        raise RegistryValidationError('R1.6 accepted parent/contract changed')
    if c['prefix_counts'] != {'families':15,'ideas':85,'sessions':6} or c['new_family_ids'] != ['FAM-CROSS-VENUE-LIQUIDITY-SEGMENTATION']:
        raise RegistryValidationError('R1.6 frozen roster/parent changed')
    assert_parent_prefix(c)
    families, ideas, sessions = (read_jsonl(U / name) for name in PRIOR)
    validate_universe(families,ideas,sessions)
    if len(families) < 16 or len(ideas) < 101 or len(sessions) < 7:
        raise RegistryValidationError('R1.6 universe truncated')
    family = families[15]
    verify_freeze_hash(family)
    if family['family_id'] not in c['new_family_ids'] or family['freeze_hash'] != c['new_family_freeze_hashes'][family['family_id']]:
        raise RegistryValidationError('R1.6 family freeze mismatch')
    if not all(x in (family['anti_family_splitting_rationale'] + family['new_family_boundary_rule']).lower() for x in ['threshold','venue']):
        raise RegistryValidationError('family splitting boundary not explicit')
    if hashlib.sha256(PUBLIC_NEW.read_bytes()).hexdigest() != c['new_public_sources_sha256']:
        raise RegistryValidationError('R1.6 public source catalogue tampered')
    new_ids = validate_sources(json.loads(PUBLIC_NEW.read_text(encoding='utf-8')),c['new_public_source_ids'])
    old_lit = {x['source_id'] for x in json.loads(PREVIOUS_PUBLIC.read_text(encoding='utf-8'))}
    assert 'LIT-MAKAROV-SCHOAR-2020' in old_lit
    leg = {x['evidence_id'] for x in json.loads(LEGACY.read_text(encoding='utf-8'))['evidence']}
    validate_added_ideas(ideas[85:101],ideas[:85],c,new_ids|{'LIT-MAKAROV-SCHOAR-2020'},leg)
    s = sessions[6]
    if s['version'] != 7 or s['status'] != 'IN_PROGRESS' or s['freeze_hash'] != c['session_v7_freeze_hash']:
        raise RegistryValidationError('R1.6 session changed')
    if s['completed_lenses'] != c['completed_lenses_after'] or s['completed_lenses'] != sessions[0]['planned_lenses'][:7] or s['planned_lenses'] != sessions[0]['planned_lenses'] or s['stopping_rule'] != sessions[0]['stopping_rule']:
        raise RegistryValidationError('R1.6 changed frozen generation order/stopping rule')
    if c['remaining_lenses'] != ['VOLATILITY_REGIME','PREDICTIVE_MODEL']:
        raise RegistryValidationError('R1.6 silently pruned next generation lens')
    for key,value in c['scientific_boundaries'].items():
        if key in ('new_d0_outcome_tests','p1_market_data_read','p1_outcome_read','permanent_hypothesis_registry_mutated'):
            if value is not False: raise RegistryValidationError('R1.6 forbidden data access or registry mutation')
        elif value is not True: raise RegistryValidationError('R1.6 scientific interpretation boundary removed')
    if read_jsonl(PERMANENT):
        raise RegistryValidationError('R1.6 permanent HYP registry must remain empty')
    return dict(status='PASS',protocol_version='1.2',accepted_parent=c['accepted_parent'],prior_families=15,new_families=1,total_families=16,prior_ideas=85,new_ideas=16,total_ideas=101,generation_mode_counts=c['generation_mode_counts'],public_sources_new=len(new_ids),legacy_evidence_ids=['LEG7M-09','LEG7M-10'],generation_session_version=7,completed_lenses=s['completed_lenses'],remaining_lenses=c['remaining_lenses'],cross_venue_r2_blocked=16,multi_venue_execution_blocked=1,permanent_hypothesis_rows=0,new_d0_outcome_tests=False,p1_market_data_read=False,p1_outcome_read=False,p1_status='SEALED_OUTCOME_UNSEEN')


def main() -> None:
    parser = argparse.ArgumentParser(description='R1.6 cross-venue hypothesis-universe validator')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    result=validate_installation()
    if args.self_test:
        assert result['total_ideas']==101 and result['cross_venue_r2_blocked']==16
        result['self_test']='PASS'
    print(json.dumps(result,sort_keys=True,indent=2))

if __name__ == '__main__': main()
