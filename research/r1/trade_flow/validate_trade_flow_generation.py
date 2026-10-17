"""R1.4: validate the append-only perpetual trade-flow idea lens; no market or outcome access."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_core.hypothesis_universe import (  # noqa: E402
    assert_generation_not_testing,
    assert_no_protected_influence,
    assert_testability_separate_from_merit,
    validate_universe,
)
from research_core.registry import read_jsonl, verify_freeze_hash  # noqa: E402
from research_core.registry.models import RegistryValidationError  # noqa: E402
from research_core.market_state.schemas import BUCKET_FIELDS, TRADE_STORAGE_GRID, FEATURE_ELIGIBLE  # noqa: E402

U = ROOT / 'research/governance/universe'
CONTRACT = ROOT / 'research/governance/r1_perps_trade_flow_contract_v1.json'
PARENT = ROOT / 'research/governance/r1_order_book_liquidity_contract_v1.json'
PUBLIC_NEW = ROOT / 'research/r1/trade_flow/new_public_sources.json'
PUBLIC_OLD = ROOT / 'research/r1/theory_mechanisms/literature_sources.json'
LEGACY = ROOT / 'research/r1/legacy_7m/legacy_evidence_catalogue.json'
PROTECTED = ROOT / 'research/governance/protected_data_register.yaml'
PERMANENT = ROOT / 'research/governance/registries/hypothesis_versions.jsonl'
PRIOR_COUNTS = {'families.jsonl': 14, 'ideas.jsonl': 51, 'generation_sessions.jsonl': 4}
LEGACY_DATASET = 'LEGACY_D0_2026-05-18_TO_2026-07-18__PHASE_7M_ARCHIVED_REPORTS'
NEW_FAMILY_ID = 'FAM-PERP-TRADE-ARRIVAL-TEMPO'
ALLOWED_STATE_FIELDS = {'trade_count','quantity_sum','notional_sum','buyer_maker_true_quantity','buyer_maker_false_quantity'}


def frozen_prefix_sha(path: Path, count: int) -> str:
    rows = path.read_bytes().splitlines(keepends=True)
    if len(rows) < count:
        raise RegistryValidationError('frozen R1.3 prefix truncated: ' + path.name)
    return hashlib.sha256(b''.join(rows[:count])).hexdigest()


def assert_parent_prefix(contract: dict) -> None:
    for name, count in PRIOR_COUNTS.items():
        if frozen_prefix_sha(U / name, count) != contract['immutable_r1_3_prefix_sha256'][name]:
            raise RegistryValidationError('frozen R1.3 prefix rewritten: ' + name)


def validate_public_sources(records: list[dict], expected: list[str]) -> set[str]:
    if not isinstance(records, list) or len(records) != len(expected):
        raise RegistryValidationError('R1.4 public source roster changed')
    ids = set()
    required = {'source_id','title','authors','year','source_kind','stable_reference','supported_mechanism_claims','scope_note'}
    for r in records:
        if not isinstance(r,dict) or set(r) != required:
            raise RegistryValidationError('R1.4 source has extra/missing fields')
        sid = r['source_id']
        if not isinstance(sid,str) or not sid.startswith('LIT-') or sid in ids:
            raise RegistryValidationError('duplicate/invalid R1.4 source ID')
        ids.add(sid)
        if not isinstance(r['year'],int) or not (1900 <= r['year'] <= 2026):
            raise RegistryValidationError('invalid source year')
        if not isinstance(r['authors'],list) or not r['authors']:
            raise RegistryValidationError('missing source authors')
        if not isinstance(r['source_kind'],str) or not r['source_kind'].strip():
            raise RegistryValidationError('source kind empty')
        if not isinstance(r['stable_reference'],str) or not r['stable_reference'].startswith('https://'):
            raise RegistryValidationError('source requires stable HTTPS public link')
        if not isinstance(r['supported_mechanism_claims'],list) or not r['supported_mechanism_claims']:
            raise RegistryValidationError('source lacks qualitative mechanism claims')
        if any(not isinstance(v,str) or not v.strip() for v in r['supported_mechanism_claims']):
            raise RegistryValidationError('empty source claim')
        if any(x in json.dumps(r['supported_mechanism_claims']).lower() for x in ('d0 outcome','p1 outcome','backtest result','btc profitability')):
            raise RegistryValidationError('public references cannot import project outcomes')
    if ids != set(expected):
        raise RegistryValidationError('R1.4 source identities differ from contract')
    return ids


def validate_new_ideas(new: list[dict], all_ideas: list[dict], contract: dict,
                       public: set[str], legacy: set[str]) -> None:
    expected = contract['added_idea_ids']
    if len(new) != 16 or [r['idea_id'] for r in new] != expected:
        raise RegistryValidationError('R1.4 new idea roster differs from contract')
    if len({r['idea_id'] for r in new}) != 16 or any(r['version'] != 1 for r in new):
        raise RegistryValidationError('R1.4 ideas must be unique first versions')
    if dict(Counter(r['generation_mode'] for r in new)) != contract['added_mode_counts']:
        raise RegistryValidationError('R1.4 fresh/legacy provenance mix changed')
    prior_refs = {f"{r['idea_id']}@{r['version']}": r for r in all_ideas[:51]}
    available_refs = set(prior_refs)
    seen_public = set()
    seen_legacy = set()
    for idea in new:
        verify_freeze_hash(idea)
        if idea['freeze_hash'] != contract['added_idea_freeze_hashes'][idea['idea_id']]:
            raise RegistryValidationError('R1.4 frozen idea content or testability was silently changed')
        assert_generation_not_testing(idea)
        assert_no_protected_influence(idea)
        assert_testability_separate_from_merit(idea)
        if idea['candidate_definition_status'] != 'TO_BE_DEVELOPED_IN_D0':
            raise RegistryValidationError('R1.4 cannot adopt legacy or new frozen candidate thresholds')
        if idea['parent_hypothesis_ids']:
            raise RegistryValidationError('R1.4 cannot promote HYP-* records')
        if not idea['parent_idea_refs'] and idea['family_id'] != NEW_FAMILY_ID:
            raise RegistryValidationError('existing mechanism family deepening must retain a parent idea')
        for ref in idea['parent_idea_refs']:
            if ref not in available_refs:
                raise RegistryValidationError('R1.4 idea references future/unknown ancestry')
            parent = prior_refs.get(ref)
            if parent and parent['generation_mode'] != 'OUTCOME_BLIND' and idea['generation_mode'] == 'OUTCOME_BLIND':
                raise RegistryValidationError('outcome-blind R1.4 idea hides result-informed parent')
        if idea['family_id'] not in {
            'FAM-INFORMED-FLOW-PRICE-DISCOVERY','FAM-METAORDER-FLOW-PERSISTENCE',
            'FAM-IMPACT-EFFICIENCY-ABSORPTION',NEW_FAMILY_ID,
        }:
            raise RegistryValidationError('R1.4 trade-flow idea split into an unrelated family')
        requirements = idea['causal_state_requirements']
        if not any(r['state_family'] == 'PERPS_TRADES' and r['required'] for r in requirements):
            raise RegistryValidationError('every R1.4 idea must require PERPS_TRADES')
        infra = idea['infrastructure_assessment']
        missing = infra['missing_capabilities']
        if infra['r2_testability'] == 'TESTABLE_NOW':
            if missing or any(r['state_family'] != 'PERPS_TRADES' for r in requirements):
                raise RegistryValidationError('TESTABLE_NOW cannot use unavailable price/print events')
        else:
            if infra['r2_testability'] != 'BLOCKED_PENDING_GOVERNED_STATE' or not missing:
                raise RegistryValidationError('unsupported R1.4 mechanism must be explicitly state-blocked')
            if set(missing) - {'GOVERNED_PERPS_TRADE_EVENTS','GOVERNED_PERPS_PRICE_PATH_STATE'}:
                raise RegistryValidationError('R1.4 invents ungoverned capability status')
        if len(missing) != len(set(missing)):
            raise RegistryValidationError('duplicated missing capability')
        refs = idea['source_refs']
        if len({r['reference'] for r in refs}) != len(refs):
            raise RegistryValidationError('duplicate R1.4 provenance source')
        source_legacy = set()
        for r in refs:
            if r['source_type'] == 'THEORY_LITERATURE':
                if r['reference'] not in public or r['source_zone'] != 'PUBLIC' or r['result_informed']:
                    raise RegistryValidationError('literature citation is unverified or result-informed')
                seen_public.add(r['reference'])
            elif r['source_type'] == 'MARKET_MECHANISM':
                if r['source_zone'] != 'NONE' or r['result_informed']:
                    raise RegistryValidationError('market mechanism claims cannot be hidden result sources')
            elif r['source_type'] == 'LEGACY_LINEAGE':
                if r['source_zone'] != 'LEGACY_D0' or r['reference'] not in legacy or r['result_informed'] is not True:
                    raise RegistryValidationError('legacy reference ungrounded or mislabeled')
                source_legacy.add(r['reference'])
            else:
                raise RegistryValidationError('unexpected R1.4 source type')
        if idea['generation_mode'] == 'LEGACY_RESULT_INFORMED':
            if source_legacy != set(idea['legacy_lineage_refs']) or source_legacy != set(idea['derived_from_results']):
                raise RegistryValidationError('legacy results have incomplete genealogy')
            if idea['datasets_that_influenced_design'] != [LEGACY_DATASET]:
                raise RegistryValidationError('legacy dataset influence omitted')
            if idea['idea_id'] != 'IDEA-R1-LEGACY-TRADE-SIZE-HHI-FAILED-REPLICATION' or source_legacy != {'LEG7M-07'}:
                raise RegistryValidationError('legacy 7M trade-concentration failure not preserved accurately')
            seen_legacy |= source_legacy
        else:
            if source_legacy or idea['legacy_lineage_refs'] or idea['derived_from_results'] or idea['datasets_that_influenced_design']:
                raise RegistryValidationError('outcome-blind R1.4 idea hid 7M result ancestry')
        text = ' '.join([idea['trigger_concept'],idea['scientific_proposition'],idea['economic_mechanism'],idea['infrastructure_assessment']['assessment_basis']]).lower()
        if 'trade_count' in text and 'unique participants' in text:
            raise RegistryValidationError('trade_count cannot reveal unique participants')
        available_refs.add(f"{idea['idea_id']}@1")
    if seen_legacy != set(contract['legacy_evidence_ids_used']):
        raise RegistryValidationError('LEG7M-07 ancestry not fully accounted for')
    if not set(contract['added_public_source_ids']).issubset(seen_public):
        raise RegistryValidationError('new public literature not linked to any R1.4 idea')


def validate_installation() -> dict:
    c = json.loads(CONTRACT.read_text(encoding='utf-8'))
    parentbytes = PARENT.read_bytes()
    p = json.loads(parentbytes)
    if c['protocol_version'] != '1.2' or c['accepted_parent']['r1_3_contract_id'] != p['contract_id']:
        raise RegistryValidationError('R1.4 not anchored to R1.3')
    if c['accepted_parent']['r1_3_tag'] != 'r1-order-book-liquidity-v1' or c['accepted_parent']['r1_3_contract_sha256'] != hashlib.sha256(parentbytes).hexdigest():
        raise RegistryValidationError('R1.3 contract hash or accepted tag changed')
    for field in ('new_market_outcome_tests','new_market_features_computed','d0_outcome_files_opened','p1_market_data_read','p1_outcome_read','permanent_hypothesis_registry_mutated','outcome_based_merit_ranking','new_strategy_thresholds_frozen'):
        if c['scope'].get(field) is not False:
            raise RegistryValidationError('R1.4 protection contract opened: ' + field)
    for field in ('retains_legacy_trade_concentration_replication_failure','no_parent_25bp_candidate_policy_endorsement','no_future_based_eligibility'):
        if c['scope'].get(field) is not True:
            raise RegistryValidationError('R1.4 protection contract weakened: ' + field)
    status = PROTECTED.read_text(encoding='utf-8')
    for token in ('pool_id: P1_PROTECTED','status: SEALED','outcome_exposure_status: UNSEEN'):
        if token not in status:
            raise RegistryValidationError('P1 seal no longer intact')
    fields = {s.name for s in BUCKET_FIELDS if s.role == FEATURE_ELIGIBLE}
    if not ALLOWED_STATE_FIELDS.issubset(fields) or TRADE_STORAGE_GRID['fine']['bucket_sec'] != 1 or TRADE_STORAGE_GRID['coarse']['bucket_sec'] != 60:
        raise RegistryValidationError('R0.5 bucket capability semantics changed')
    srules = c['state_capability_rules']
    for key in ('testable_now_requires_only_bucket_fields','raw_maker_flag_mapping_must_be_verified_in_d0','bucket_row_count_is_not_unique_orders_or_participants','r0_5_trade_storage_grid_not_scientific_feature_definition'):
        if srules.get(key) is not True:
            raise RegistryValidationError('R1.4 causal trade semantic protection disabled: '+key)
    assert_parent_prefix(c)
    families,ideas,sessions = (read_jsonl(U / name) for name in PRIOR_COUNTS)
    validate_universe(families,ideas,sessions)
    if len(families) < 15 or len(ideas) < 67 or len(sessions) < 5:
        raise RegistryValidationError('R1.4 prefix truncated')
    # R1.5+ may append governed records. Preserve this lens's exact historical prefix.
    families,ideas,sessions = families[:15],ideas[:67],sessions[:5]
    if [x['family_id'] for x in families[14:]] != c['added_family_ids'] or families[14]['mechanism_class'] != 'MARKETABLE_TRADE_ARRIVAL_ACTIVITY' or families[14]['freeze_hash'] != c['added_family_freeze_hash']:
        raise RegistryValidationError('R1.4 new family improperly split or renamed')
    if hashlib.sha256(PUBLIC_NEW.read_bytes()).hexdigest() != c['new_public_sources_sha256']:
        raise RegistryValidationError('R1.4 literature source catalogue altered')
    new_sources = validate_public_sources(json.loads(PUBLIC_NEW.read_text(encoding='utf-8')),c['added_public_source_ids'])
    old_sources = {x['source_id'] for x in json.loads(PUBLIC_OLD.read_text(encoding='utf-8'))}
    if new_sources & old_sources:
        raise RegistryValidationError('R1.4 duplicates existing public research source IDs')
    legacy_ids = {x['evidence_id'] for x in json.loads(LEGACY.read_text(encoding='utf-8'))['evidence']}
    validate_new_ideas(ideas[51:],ideas,c,new_sources|old_sources,legacy_ids)
    planned = sessions[0]['planned_lenses']
    expected = c['completed_lenses_after']
    if sessions[-1]['version'] != 5 or sessions[-1]['status'] != 'IN_PROGRESS' or sessions[-1]['completed_lenses'] != expected or sessions[-1]['freeze_hash'] != c['generation_session_v5_freeze_hash']:
        raise RegistryValidationError('R1.4 session lens advancement invalid')
    if expected != planned[:5] or c['remaining_lenses'] != planned[5:]:
        raise RegistryValidationError('R1.4 cannot prune or reorder the remaining lenses')
    if sessions[-1]['planned_lenses'] != sessions[0]['planned_lenses'] or sessions[-1]['stopping_rule'] != sessions[0]['stopping_rule']:
        raise RegistryValidationError('R1.4 silently changed frozen generation stopping rule')
    if read_jsonl(PERMANENT):
        raise RegistryValidationError('R1.4 may not create permanent HYP-* registrations')
    testable = sum(x['infrastructure_assessment']['r2_testability']=='TESTABLE_NOW' for x in ideas[51:])
    return {'status':'PASS','protocol_version':'1.2','accepted_r1_3_parent':c['accepted_parent'],
        'prior_families':14,'new_families':1,'total_families':15,'prior_ideas':51,'new_ideas':16,'total_ideas':67,
        'generation_mode_counts':dict(Counter(x['generation_mode'] for x in ideas[51:])),
        'testable_now_structurally':testable,'blocked_pending_governed_state':16-testable,
        'generation_session_version':5,'completed_lenses':expected,'remaining_lenses':c['remaining_lenses'],
        'permanent_hypothesis_rows':0,'new_d0_outcome_tests':False,'p1_market_data_read':False,
        'p1_outcome_read':False,'p1_status':'SEALED_OUTCOME_UNSEEN'}


def main() -> None:
    ap=argparse.ArgumentParser(description='R1.4 causal perp trade-flow universe validator (no market/outcome access)')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    result=validate_installation()
    if args.self_test:
        assert result['new_ideas']==16 and result['total_ideas']==67
        assert result['testable_now_structurally']==12 and result['blocked_pending_governed_state']==4
        result['self_test']='PASS'
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=='__main__':
    main()
