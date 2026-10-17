"""Validate R1.3 append-only, provenance-separated book ideas. No market/outcome access."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_core.hypothesis_universe import (  # noqa: E402
    assert_generation_not_testing,
    assert_no_protected_influence,
    assert_testability_separate_from_merit,
    validate_universe,
)
from research_core.registry import read_jsonl  # noqa: E402
from research_core.registry.models import RegistryValidationError  # noqa: E402

U = ROOT / 'research/governance/universe'
CONTRACT = ROOT / 'research/governance/r1_order_book_liquidity_contract_v1.json'
OLD_CONTRACT = ROOT / 'research/governance/r1_legacy_lineage_contract_v1.json'
OLD_SOURCES = ROOT / 'research/r1/theory_mechanisms/literature_sources.json'
NEW_SOURCES = ROOT / 'research/r1/order_book/new_public_sources.json'
LEGACY = ROOT / 'research/r1/legacy_7m/legacy_evidence_catalogue.json'
PROTECTED = ROOT / 'research/governance/protected_data_register.yaml'
PERMANENT = ROOT / 'research/governance/registries/hypothesis_versions.jsonl'
R0_STATE = ROOT / 'research/governance/market_state_contract_v1.json'
LEGACY_DATASET = 'LEGACY_D0_2026-05-18_TO_2026-07-18__PHASE_7M_ARCHIVED_REPORTS'
PRIOR = {'families.jsonl': 13, 'ideas.jsonl': 35, 'generation_sessions.jsonl': 3}


def frozen_prefix_sha(path: Path, n: int) -> str:
    """Hash exactly the historical JSONL prefix (including its final LF)."""
    original = path.read_bytes()
    lines = original.splitlines(keepends=True)
    if len(lines) < n:
        raise RegistryValidationError(f'append-only prefix truncated: {path.name}')
    return hashlib.sha256(b''.join(lines[:n])).hexdigest()


def assert_prior_prefix(contract: dict) -> None:
    for filename, count in PRIOR.items():
        recorded = contract['immutable_r1_2_prefix_sha256'][filename]
        if frozen_prefix_sha(U / filename, count) != recorded:
            raise RegistryValidationError(f'R1.2 historical append-only prefix was rewritten: {filename}')


def validate_source_catalogue(records: list, expected_ids: list) -> set[str]:
    if not isinstance(records, list) or len(records) != len(expected_ids):
        raise RegistryValidationError('unexpected R1.3 public source roster')
    ids = set()
    required = {'source_id','title','authors','year','stable_reference','source_kind','supported_mechanism_claims','scope_note'}
    for record in records:
        if not isinstance(record, dict) or set(record) != required:
            raise RegistryValidationError('unguarded R1.3 literature field')
        ident = record['source_id']
        if not isinstance(ident,str) or not ident.startswith('LIT-') or ident in ids:
            raise RegistryValidationError('R1.3 source IDs must be unique and stable')
        ids.add(ident)
        if record['source_kind'] != 'PUBLIC_RESEARCH' or not record['stable_reference'].startswith('https://'):
            raise RegistryValidationError('R1.3 new sources must be public and verifiable')
        if not isinstance(record['year'],int) or not (1900 <= record['year'] <= 2026):
            raise RegistryValidationError('invalid source year')
        if not isinstance(record['authors'],list) or not record['authors']:
            raise RegistryValidationError('missing authors')
        if not isinstance(record['supported_mechanism_claims'],list) or not record['supported_mechanism_claims']:
            raise RegistryValidationError('missing qualitative literature claims')
        if any(not isinstance(x,str) or not x.strip() for x in record['supported_mechanism_claims']):
            raise RegistryValidationError('empty claim')
        if any(token in ' '.join(record['supported_mechanism_claims']).lower() for token in ('p_value','backtest','btc return','d0 outcome','p1')):
            raise RegistryValidationError('literature source must not report project outcomes')
    if ids != set(expected_ids):
        raise RegistryValidationError('R1.3 literature references differ from contract')
    return ids


def validate_new_ideas(new_ideas: list[dict], all_ideas: list[dict], contract: dict,
                       literature_ids: set[str], legacy_ids: set[str]) -> None:
    expected = contract['added_idea_ids']
    if [i['idea_id'] for i in new_ideas] != expected or len(expected) != 16:
        raise RegistryValidationError('R1.3 idea roster changed or reordered')
    if len({i['idea_id'] for i in new_ideas}) != len(expected):
        raise RegistryValidationError('R1.3 duplicate idea')
    if [i['version'] for i in new_ideas] != [1] * len(new_ideas):
        raise RegistryValidationError('new ideas must be first versions')
    counts = Counter(i['generation_mode'] for i in new_ideas)
    if counts != contract['added_mode_counts']:
        raise RegistryValidationError('outcome-blind/result-informed balance changed')
    old_refs = {f"{i['idea_id']}@{i['version']}": i for i in all_ideas[:35]}
    source_refs = set()
    results_seen = set()
    for idea in new_ideas:
        assert_generation_not_testing(idea)
        assert_no_protected_influence(idea)
        assert_testability_separate_from_merit(idea)
        if idea['candidate_definition_status'] != 'TO_BE_DEVELOPED_IN_D0':
            raise RegistryValidationError('R1.3 cannot adopt a fixed 25bp or other trading event')
        if idea['parent_hypothesis_ids']:
            raise RegistryValidationError('R1.3 may not create prior permanent HYP lineage')
        if not any(x['state_family']=='ORDER_BOOK' and x['required'] is True for x in idea['causal_state_requirements']):
            raise RegistryValidationError('R1.3 idea must require a governed order-book predictor state')
        infra = idea['infrastructure_assessment']
        if infra['r2_testability'] != 'BLOCKED_PENDING_GOVERNED_STATE' or 'GOVERNED_ORDER_BOOK_STATE' not in infra['missing_capabilities']:
            raise RegistryValidationError('R0.6 execution book is not a governed feature state')
        if len(infra['missing_capabilities']) != len(set(infra['missing_capabilities'])):
            raise RegistryValidationError('duplicate missing capability')
        for p in idea['parent_idea_refs']:
            # Parents must be frozen prior-lens ideas or earlier R1.3 siblings.
            if p not in old_refs:
                if p not in {f"{x['idea_id']}@1" for x in new_ideas[:new_ideas.index(idea)]}:
                    raise RegistryValidationError('R1.3 parent must predate the child')
        refs = [r['reference'] for r in idea['source_refs']]
        if len(refs) != len(set(refs)):
            raise RegistryValidationError('repeated source ref')
        for ref in idea['source_refs']:
            if ref['source_type']=='THEORY_LITERATURE':
                if ref['reference'] not in literature_ids or ref['source_zone']!='PUBLIC' or ref['result_informed'] is not False:
                    raise RegistryValidationError('unverified or falsely result-informed literature')
            elif ref['source_type']=='MARKET_MECHANISM':
                if ref['source_zone']!='NONE' or ref['result_informed'] is not False:
                    raise RegistryValidationError('first-principles source cannot be treated as a result')
            elif ref['source_type']=='LEGACY_LINEAGE':
                if ref['source_zone']!='LEGACY_D0' or ref['result_informed'] is not True or ref['reference'] not in legacy_ids:
                    raise RegistryValidationError('legacy source not grounded in accepted catalogue')
            else:
                raise RegistryValidationError('unexpected R1.3 source type')
            source_refs.add(ref['reference'])
        legacy = [r['reference'] for r in idea['source_refs'] if r['source_type']=='LEGACY_LINEAGE']
        if idea['generation_mode'] == 'LEGACY_RESULT_INFORMED':
            if not legacy or idea['derived_from_results'] != idea['legacy_lineage_refs'] or set(legacy) != set(idea['legacy_lineage_refs']):
                raise RegistryValidationError('legacy influence was not fully disclosed')
            if idea['datasets_that_influenced_design'] != [LEGACY_DATASET]:
                raise RegistryValidationError('legacy D0 historical dataset influence omitted')
            if not idea['parent_idea_refs']:
                raise RegistryValidationError('R1.3 legacy deepening requires a frozen parent idea')
            results_seen.update(legacy)
        else:
            if legacy or idea['derived_from_results'] or idea['legacy_lineage_refs'] or idea['datasets_that_influenced_design']:
                raise RegistryValidationError('OUTCOME_BLIND cannot derive from 7M results')
            for parent in idea['parent_idea_refs']:
                parent_idea = old_refs.get(parent)
                if parent_idea and parent_idea['generation_mode'] != 'OUTCOME_BLIND':
                    raise RegistryValidationError('outcome-blind child cannot hide legacy parent ancestry')
    if sorted(results_seen) != contract['legacy_evidence_ids_used']:
        raise RegistryValidationError('R1.3 historical evidence genealogy differs from frozen contract')
    if not set(contract['added_public_source_ids']).issubset(source_refs):
        raise RegistryValidationError('new R1.3 public source is not used')
    if not any('REFILL-EXECUTION-BOUND' in x['idea_id'] for x in new_ideas):
        raise RegistryValidationError('historical negative replacement penetration must be retained')


def validate_installation() -> dict:
    contract = json.loads(CONTRACT.read_text(encoding='utf-8'))
    old = json.loads(OLD_CONTRACT.read_text(encoding='utf-8'))
    if contract['protocol_version'] != '1.2' or contract['accepted_parent']['r1_2_contract_id'] != old['contract_id']:
        raise RegistryValidationError('not anchored to accepted R1.2 contract')
    if contract['accepted_parent']['r1_2_tag'] != 'r1-legacy-lineage-v1':
        raise RegistryValidationError('R1.2 parent tag unexpected')
    if contract['accepted_parent']['r1_2_legacy_evidence_catalogue_sha256'] != hashlib.sha256(LEGACY.read_bytes()).hexdigest():
        raise RegistryValidationError('R1.2 archived catalogue changed')
    for key in ('new_market_outcome_tests','new_market_features_computed','p1_market_data_read','p1_outcome_read','permanent_hypothesis_registry_mutated','outcome_based_merit_ranking','new_strategy_thresholds_frozen','r0_5_governed_order_book_available','not_an_endorsement_of_25bp_policy'):
        if key == 'not_an_endorsement_of_25bp_policy':
            if contract['scope'][key] is not True:
                raise RegistryValidationError('25bp legacy policy incorrectly endorsed')
        elif contract['scope'][key] is not False:
            raise RegistryValidationError(f'R1.3 boundary opened: {key}')
    for key in ('retains_legacy_failed_inconclusive_ancestry','r0_6_execution_book_not_predictor_state','no_future_based_book_eligibility'):
        if contract['scope'][key] is not True:
            raise RegistryValidationError(f'R1.3 safety guard disabled: {key}')
    seal = PROTECTED.read_text(encoding='utf-8')
    if any(token not in seal for token in ('pool_id: P1_PROTECTED','status: SEALED','outcome_exposure_status: UNSEEN')):
        raise RegistryValidationError('P1 seal failed')
    assert_prior_prefix(contract)
    all_records = [read_jsonl(U / fn) for fn in PRIOR]
    families,ideas,sessions = all_records
    validate_universe(families,ideas,sessions)
    if len(families)<14 or len(ideas)<51 or len(sessions)<4:
        raise RegistryValidationError('R1.3 accepted prefix is truncated')
    # Whole expanding universe was validated above. From here onward R1.3's
    # historical assertion concerns only its own frozen suffix/prefix.
    families, ideas, sessions = families[:14], ideas[:51], sessions[:4]
    if [f['family_id'] for f in families[13:]] != contract['added_family_ids']:
        raise RegistryValidationError('unexpected new mechanism family')
    if families[13]['mechanism_class'] != 'SPATIAL_LIQUIDITY_PLACEMENT':
        raise RegistryValidationError('new family must be spatial geometry, not a duplicate resilience/queue family')
    lens = ['THEORY_LITERATURE','MARKET_MECHANISM','LEGACY_7M_LINEAGE','ORDER_BOOK_LIQUIDITY']
    if sessions[-1]['completed_lenses'] != lens or sessions[-1]['status'] != 'IN_PROGRESS' or sessions[-1]['version'] != 4:
        raise RegistryValidationError('generation lens progress does not match R1.3')
    if sessions[-1]['planned_lenses'] != sessions[0]['planned_lenses'] or sessions[-1]['stopping_rule'] != sessions[0]['stopping_rule']:
        raise RegistryValidationError('R1.3 illegally changed frozen generation plan')
    sources = validate_source_catalogue(json.loads(NEW_SOURCES.read_text(encoding='utf-8')), contract['added_public_source_ids'])
    old_sources = {x['source_id'] for x in json.loads(OLD_SOURCES.read_text(encoding='utf-8'))}
    if sources & old_sources:
        raise RegistryValidationError('duplicate literature source from earlier lens')
    legacy = {x['evidence_id'] for x in json.loads(LEGACY.read_text(encoding='utf-8'))['evidence']}
    validate_new_ideas(ideas[35:], ideas, contract, sources|old_sources, legacy)
    if read_jsonl(PERMANENT):
        raise RegistryValidationError('R1.3 may not promote ideas into permanent HYP registry')
    return {'status':'PASS','protocol_version':'1.2','accepted_r1_2_parent':contract['accepted_parent'],
            'prior_families':13,'new_families':1,'total_families':len(families),
            'prior_ideas':35,'new_ideas':len(ideas)-35,'total_ideas':len(ideas),
            'generation_mode_counts':dict(Counter(i['generation_mode'] for i in ideas[35:])),
            'generation_session_version':4,'completed_lenses':lens,
            'remaining_lenses':[x for x in sessions[-1]['planned_lenses'] if x not in lens],
            'r2_book_ideas_blocked':len(ideas[35:]),'permanent_hypothesis_rows':0,
            'new_d0_outcome_tests':False,'p1_market_data_read':False,'p1_outcome_read':False,
            'p1_status':'SEALED_OUTCOME_UNSEEN'}


def main() -> None:
    p=argparse.ArgumentParser(description='Validate governed R1.3 order-book generation; never touch market data')
    p.add_argument('--self-test',action='store_true'); args=p.parse_args()
    result=validate_installation()
    if args.self_test:
        assert result['new_ideas']==16 and result['r2_book_ideas_blocked']==16
        result['self_test']='PASS'
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__ == '__main__':
    main()
