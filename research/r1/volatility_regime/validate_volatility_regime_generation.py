"""R1.7 pre-outcome volatility/regime universe validation. Never opens market files."""
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
C = ROOT / 'research/governance/r1_volatility_regime_contract_v1.json'
PARENT = ROOT / 'research/governance/r1_cross_venue_contract_v1.json'
SOURCES = ROOT / 'research/r1/volatility_regime/new_public_sources.json'
PERMANENT = ROOT / 'research/governance/registries/hypothesis_versions.jsonl'
PRIOR = {'families.jsonl': 16, 'ideas.jsonl': 101, 'generation_sessions.jsonl': 7}
FAMILIES = {'FAM-VOLATILITY-PERSISTENCE-TRANSITION','FAM-VOLATILITY-LIQUIDITY-FEEDBACK'}
SOURCE_FIELDS = {'source_id','title','authors','year','source_kind','stable_reference','supported_mechanism_claims','scope_note'}
ALLOWED_STATE = {'PERPS_PRICE_PATH','PERPS_TRADES','PERPS_ORDER_BOOK','DERIVATIVES_POSITIONING'}


def frozen_prefix_hash(path: Path, count: int) -> str:
    """Hash raw frozen bytes, not records re-serialized with platform-dependent newlines."""
    lines = path.read_bytes().splitlines(keepends=True)
    if len(lines) < count:
        raise RegistryValidationError('R1.6 frozen prefix truncated: ' + path.name)
    return hashlib.sha256(b''.join(lines[:count])).hexdigest()


def assert_parent_prefix(c: dict) -> None:
    for name, n in PRIOR.items():
        if frozen_prefix_hash(U / name, n) != c['immutable_r1_6_prefix_sha256'][name]:
            raise RegistryValidationError('R1.6 frozen prefix rewritten: ' + name)


def validate_sources(records: list[dict], expected: list[str]) -> set[str]:
    if not isinstance(records, list) or len(records) != len(expected):
        raise RegistryValidationError('R1.7 public-source roster mismatch')
    ids = set()
    for r in records:
        if not isinstance(r, dict) or set(r) != SOURCE_FIELDS:
            raise RegistryValidationError('R1.7 public-source fields changed')
        sid = r['source_id']
        if not isinstance(sid,str) or sid in ids or not sid.startswith('LIT-'):
            raise RegistryValidationError('invalid/duplicate R1.7 source')
        ids.add(sid)
        if not isinstance(r['stable_reference'], str) or not r['stable_reference'].startswith('https://doi.org/'):
            raise RegistryValidationError('R1.7 sources require stable DOI references')
        if not isinstance(r['year'],int) or isinstance(r['year'],bool) or not 1900 <= r['year'] <= 2026:
            raise RegistryValidationError('invalid public source year')
        if not isinstance(r['authors'],list) or not r['authors'] or not all(isinstance(x,str) and x.strip() for x in r['authors']):
            raise RegistryValidationError('source authors missing')
        if not all(isinstance(r[k],str) and r[k].strip() for k in ('title','source_kind','scope_note')):
            raise RegistryValidationError('source missing substantive description')
        if not isinstance(r['supported_mechanism_claims'],list) or not r['supported_mechanism_claims'] or not all(isinstance(x,str) and x.strip() for x in r['supported_mechanism_claims']):
            raise RegistryValidationError('source missing scoped mechanism claims')
        if any(term in json.dumps(r).lower() for term in ('positive btcusdt alpha','p1 return','backtested btcusdt win rate','profitable btcusdt signal')):
            raise RegistryValidationError('outcome claim laundered through public catalogue')
    if ids != set(expected):
        raise RegistryValidationError('R1.7 source identifiers changed')
    return ids


def validate_added_ideas(rows: list[dict], c: dict, public_ids: set[str]) -> None:
    if len(rows) != 16 or [x['idea_id'] for x in rows] != c['new_idea_ids']:
        raise RegistryValidationError('R1.7 idea roster changed')
    used_sources = set()
    counts = Counter()
    for i in rows:
        verify_freeze_hash(i)
        if i['version'] != 1 or i['freeze_hash'] != c['new_idea_freeze_hashes'][i['idea_id']]:
            raise RegistryValidationError('R1.7 frozen idea mutated')
        if i['family_id'] not in FAMILIES:
            raise RegistryValidationError('volatility variation split into new unregistered family')
        if i['candidate_definition_status'] != 'TO_BE_DEVELOPED_IN_D0':
            raise RegistryValidationError('R1.7 adopted historical event policy')
        assert_generation_not_testing(i)
        assert_no_protected_influence(i)
        assert_testability_separate_from_merit(i)
        if i['generation_mode'] != 'OUTCOME_BLIND' or any(i[f] for f in ('parent_idea_refs','parent_hypothesis_ids','legacy_lineage_refs','derived_from_results','datasets_that_influenced_design')):
            raise RegistryValidationError('hidden observed-result genealogy in R1.7 outcome-blind idea')
        states = {r['state_family'] for r in i['causal_state_requirements'] if r['required']}
        if not states or not states <= ALLOWED_STATE or 'PERPS_PRICE_PATH' not in states:
            raise RegistryValidationError('true volatility requires governed historical price path')
        if not all('pre-decision' in r['role'].lower() and 'receive-time' in r['role'].lower() for r in i['causal_state_requirements']):
            raise RegistryValidationError('state requirement omitted predecision receive-time')
        missing = {c['mandatory_capability_gates'][x] for x in states if x in c['mandatory_capability_gates']}
        exec_variant = i['idea_id'] == 'IDEA-R1-VOL-CONDITIONED-EXECUTION'
        if exec_variant:missing.add(c['additional_execution_gate'])
        status = 'BLOCKED_PENDING_STATE_AND_EXECUTION' if exec_variant else 'BLOCKED_PENDING_GOVERNED_STATE'
        if i['infrastructure_assessment']['r2_testability'] != status or set(i['infrastructure_assessment']['missing_capabilities']) != missing or not missing:
            raise RegistryValidationError('R1.7 capability assessment falsified')
        if i['generation_mode'] != 'OUTCOME_BLIND' or i['infrastructure_assessment'].get('scientific_merit_score') is not None:
            raise RegistryValidationError('unjustified merit or provenance status')
        if 'no R1 profitability claim' not in i['primary_economic_target']:
            raise RegistryValidationError('volatility idea misrepresented as demonstrated alpha')
        if 'P75/15m' not in i['position_concept']['horizon_concept']:
            raise RegistryValidationError('legacy horizon not explicitly rejected')
        if 'not infer long/short' not in i['position_concept']['directionality'].lower():
            raise RegistryValidationError('volatility state implicitly asserts direction')
        if not any('post-hoc' in x.lower() for x in i['risk_concepts']):
            raise RegistryValidationError('no regime-slice rescue boundary')
        if not any('future' in x.lower() or 'hindsight' in x.lower() for x in i['risk_concepts']):
            raise RegistryValidationError('future-regime leakage omitted')
        refs = i['source_refs']
        if not any(ref['source_type'] == 'MARKET_MECHANISM' for ref in refs):
            raise RegistryValidationError('first-principles mechanism not identified')
        for ref in refs:
            if ref['source_type'] == 'THEORY_LITERATURE':
                if ref['source_zone'] != 'PUBLIC' or ref['result_informed'] or ref['reference'] not in public_ids:
                    raise RegistryValidationError('unknown or mislabeled literature source')
                used_sources.add(ref['reference'])
            elif ref['source_type'] == 'MARKET_MECHANISM':
                if ref['source_zone'] != 'NONE' or ref['result_informed']:
                    raise RegistryValidationError('mechanism source cannot disguise result influence')
            else:
                raise RegistryValidationError('result source is not permitted in outcome-blind R1.7')
        counts[i['generation_mode']] += 1
    if dict(counts) != {'OUTCOME_BLIND':16} or c['generation_mode_counts'] != {'OUTCOME_BLIND':16,'LEGACY_RESULT_INFORMED':0}:
        raise RegistryValidationError('R1.7 generation-mode counts changed')
    if used_sources != public_ids:
        raise RegistryValidationError('uncited R1.7 public-source record')


def validate_installation() -> dict:
    c = json.loads(C.read_text(encoding='utf-8'))
    parent = json.loads(PARENT.read_text(encoding='utf-8'))
    if c['contract_id'] != 'R1_7_VOLATILITY_REGIME_GENERATION_V1' or c['protocol_version'] != '1.2':
        raise RegistryValidationError('R1.7 contract/protocol changed')
    if c['accepted_parent'] != {'tag':'r1-cross-venue-v1','contract_id':parent['contract_id'],'commit_prefix':'d5ea875'}:
        raise RegistryValidationError('R1.7 accepted parent mismatched')
    if c['prefix_counts'] != {'families':16,'ideas':101,'sessions':7} or len(c['new_idea_ids']) != 16:
        raise RegistryValidationError('R1.7 frozen parent/idea roster invalid')
    assert_parent_prefix(c)
    families,ideas,sessions = (read_jsonl(U / n) for n in PRIOR)
    validate_universe(families,ideas,sessions)
    if len(families) < 17 or len(ideas) < 117 or len(sessions) < 8:
        raise RegistryValidationError('R1.7 append truncated')
    family = families[16]
    verify_freeze_hash(family)
    if family['family_id'] != 'FAM-VOLATILITY-PERSISTENCE-TRANSITION' or c['new_family_ids'] != [family['family_id']] or family['freeze_hash'] != c['new_family_freeze_hashes'][family['family_id']]:
        raise RegistryValidationError('R1.7 family freeze or identity changed')
    if not all(k in (family['new_family_boundary_rule']+' '+family['anti_family_splitting_rationale']).lower() for k in ('threshold','horizon')):
        raise RegistryValidationError('R1.7 family splitting protection missing')
    if hashlib.sha256(SOURCES.read_bytes()).hexdigest() != c['new_public_sources_sha256']:
        raise RegistryValidationError('R1.7 source catalogue tampered')
    public_ids = validate_sources(json.loads(SOURCES.read_text(encoding='utf-8')), c['new_public_source_ids'])
    validate_added_ideas(ideas[101:117],c,public_ids)
    s = sessions[7]
    verify_freeze_hash(s)
    if s['version'] != 8 or s['status'] != 'IN_PROGRESS' or s['freeze_hash'] != c['session_v8_freeze_hash']:
        raise RegistryValidationError('R1.7 frozen generation session changed')
    if s['completed_lenses'] != s['planned_lenses'][:8] or s['completed_lenses'] != c['completed_lenses_after'] or s['planned_lenses'] != sessions[0]['planned_lenses'] or s['stopping_rule'] != sessions[0]['stopping_rule'] or c['remaining_lenses'] != ['PREDICTIVE_MODEL']:
        raise RegistryValidationError('R1.7 prematurely completed or pruned future lens')
    for k,v in c['scientific_boundaries'].items():
        if k in {'new_d0_outcome_tests','p1_market_data_read','p1_outcome_read','permanent_hypothesis_registry_mutated'}:
            if v is not False:raise RegistryValidationError('R1.7 forbidden test, protected read or HYP mutation')
        elif v is not True:raise RegistryValidationError('R1.7 missing scientific interpretation boundary: '+k)
    if read_jsonl(PERMANENT):
        raise RegistryValidationError('R1.7 permanent HYP registry no longer empty')
    return dict(status='PASS',protocol_version='1.2',accepted_parent=c['accepted_parent'],prior_families=16,new_families=1,total_families=17,prior_ideas=101,new_ideas=16,total_ideas=117,generation_mode_counts=c['generation_mode_counts'],new_public_sources=len(public_ids),capability_blocked_volatility_ideas=16,extra_execution_gate_ideas=1,completed_lenses=s['completed_lenses'],remaining_lenses=c['remaining_lenses'],generation_session_version=8,new_d0_outcome_tests=False,p1_market_data_read=False,p1_outcome_read=False,p1_status='SEALED_OUTCOME_UNSEEN',permanent_hypothesis_rows=0)


def main() -> None:
    p = argparse.ArgumentParser(description='R1.7 volatility/regime hypothesis generation validator')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    result=validate_installation()
    if args.self_test:
        assert result['total_ideas'] == 117 and result['capability_blocked_volatility_ideas'] == 16
        result['self_test']='PASS'
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__ == '__main__':main()
