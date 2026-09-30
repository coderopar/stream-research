#!/usr/bin/env python3
"""R0.8 — full end-to-end dry run of the governed research machine on D0.

This program deliberately creates a synthetic/non-promotable research proposition
inside an isolated registry namespace.  It never writes to the production R0.7
registries and never reads P1 market data or protected outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_core.dry_run import (  # noqa: E402
    DRY_RUN_HORIZON_SEC,
    add_mechanical_net_proxy,
    apply_flat_only_position_state_machine,
    attach_future_observability,
    build_causal_analysis_population,
    build_fixed_test_state,
    dependence_aware_contrast,
)
from research_core.registry import (  # noqa: E402
    RegistryValidationError,
    append_jsonl,
    assert_eligibility_fields_safe,
    assert_predictor_columns_safe,
    assert_time_respecting_validation,
    authorize_protected_access,
    read_jsonl,
    sha256_canonical,
    validate_access_event_history,
    validate_confirmation_batch,
    validate_decision_log,
    validate_event_history,
    validate_evidence_budget,
    validate_genealogy,
    validate_hypothesis_version,
    validate_material_change,
    validate_protected_cohort,
    validate_registered_spec,
    with_freeze_hash,
)



def implementation_fingerprint() -> dict[str, Any]:
    """Hash the R0.8 implementation surface that can change dry-run semantics.

    This lets idempotence mean same inputs *and same implementation*. A legitimate
    code/contract correction establishes a new baseline instead of being mistaken
    for nondeterminism.
    """
    explicit = [
        Path(__file__).resolve(),
        REPO_ROOT / "src/research_core/dry_run/__init__.py",
        REPO_ROOT / "src/research_core/dry_run/analysis.py",
        REPO_ROOT / "research/governance/r0_dry_run_contract_v1.json",
        REPO_ROOT / "research/governance/research_registry_contract_v1.json",
    ]
    registry_root = REPO_ROOT / "src/research_core/registry"
    explicit.extend(sorted(registry_root.glob("*.py")))
    files = {}
    for path in explicit:
        if not path.exists():
            raise RuntimeError(f"R0.8 implementation fingerprint missing required file: {path}")
        rel = path.relative_to(REPO_ROOT).as_posix()
        files[rel] = sha256_file(path)
    return {"files": files, "sha256": sha256_canonical(files)}


def enforce_same_implementation_idempotence(
    old_manifest: dict[str, Any],
    *,
    current_implementation_sha256: str,
    parent_hashes: dict[str, Any],
    substantive_run_hash: str,
) -> None:
    """Fail only when an identical implementation+parent set changes substantively."""
    old_impl = (old_manifest.get("implementation_fingerprint") or {}).get("sha256")
    same_impl = bool(old_impl) and old_impl == current_implementation_sha256
    same_parents = old_manifest.get("parents") == parent_hashes
    if same_impl and same_parents:
        old_hash = old_manifest.get("substantive_run_hash")
        if old_hash and old_hash != substantive_run_hash:
            raise RuntimeError(
                f"R0.8 same-input same-implementation substantive idempotence failure: "
                f"{old_hash} != {substantive_run_hash}"
            )

SCHEMA_NAME = "r0_full_d0_dry_run_v1"
SCHEMA_VERSION = 1
BUILDER_VERSION = "1.0.1"
DRY_HYP = "HYP-R08DRY001"
DRY_FAM = "FAM-R08DRY001"
DRY_BUDGET = "EB-R08DRY001"
DRY_BATCH = "CB-R08DRY001"
DRY_COHORT = "D1-R08DRY001"
FIXED_TS = "2026-09-30T00:00:00Z"
EXPECTED_CANDIDATES = 7067
EXPECTED_GOVERNED = 7031

PRODUCTION_REGISTRY_FILES = (
    "hypothesis_versions.jsonl", "hypothesis_events.jsonl", "evidence_budgets.jsonl",
    "protected_cohorts.jsonl", "confirmation_batches.jsonl", "access_events.jsonl",
    "decision_log.jsonl",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(value, fh, indent=2, sort_keys=True, allow_nan=False)
            fh.write("\n")
            fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "UNKNOWN"


def load_manifest(root: Path, expected_schema: str) -> tuple[dict[str, Any], Path]:
    p = root / "metadata" / "manifest.json"
    if not p.exists():
        raise RuntimeError(f"missing parent manifest: {p}")
    m = json.loads(p.read_text(encoding="utf-8"))
    if m.get("schema_name") != expected_schema:
        raise RuntimeError(f"unexpected parent schema at {p}: {m.get('schema_name')!r}")
    return m, p


def verify_output(root: Path, manifest: dict[str, Any], key: str, relative_fallback: str) -> Path:
    meta = manifest.get("outputs", {}).get(key, {})
    if not isinstance(meta, dict) or not meta.get("sha256"):
        raise RuntimeError(f"parent manifest missing output hash for {key}")
    p = root / relative_fallback
    if not p.exists():
        raise RuntimeError(f"missing parent output: {p}")
    actual = sha256_file(p)
    if actual != str(meta["sha256"]):
        raise RuntimeError(f"parent output hash mismatch for {key}: {actual} != {meta['sha256']}")
    return p


def snapshot_production_registries(registry_root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in PRODUCTION_REGISTRY_FILES:
        p = registry_root / name
        if not p.exists():
            raise RuntimeError(f"missing production R0.7 registry: {p}")
        out[name] = sha256_bytes(p.read_bytes())
    return out


def assert_p1_sealed(protected_register: Path) -> None:
    text = protected_register.read_text(encoding="utf-8")
    required = ("pool_id: P1_PROTECTED", "status: SEALED", "outcome_exposure_status: UNSEEN")
    missing = [x for x in required if x not in text]
    if missing:
        raise RuntimeError(f"P1 seal contract changed or missing: {missing}")


def _safe_float(x: Any) -> float | None:
    if x is None or pd.isna(x): return None
    v = float(x)
    return v if math.isfinite(v) else None


def dataframe_substantive_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    f = frame[columns].copy().sort_values("candidate_id", kind="mergesort") if "candidate_id" in columns else frame[columns].copy()
    records: list[dict[str, Any]] = []
    for r in f.to_dict("records"):
        clean: dict[str, Any] = {}
        for k, v in r.items():
            if v is None or (isinstance(v, float) and not math.isfinite(v)) or pd.isna(v):
                clean[k] = None
            elif hasattr(v, "item"):
                clean[k] = v.item()
            else:
                clean[k] = v
        records.append(clean)
    return sha256_canonical(records)


def make_dry_run_registry_records(parent_hashes: dict[str, str], code_commit: str) -> dict[str, Any]:
    budget = with_freeze_hash({
        "evidence_budget_id": DRY_BUDGET,
        "family_ids": [DRY_FAM],
        "primary_claim_ids": [f"{DRY_HYP}@1"],
        "secondary_claim_ids": [],
        "robustness_diagnostic_ids": [],
        "control_method": "SINGLE_PRIMARY_DECISION",
        "operating_guarantee": "Exactly one synthetic dry-run primary decision; no alternative success path and no promotable claim.",
        "anti_family_splitting_rationale": "R0.8 contains one fixed plumbing state only; no thresholds, horizons, subgroups, or variants are searched.",
        "expected_independent_support": {"unit": "R0.4 conservative inference clusters", "expected": "D0 dry-run diagnostic only"},
        "do_not_spend_threshold": "No D1 may ever be spent on this synthetic R0.8 hypothesis.",
        "unallocated_alternative_success_paths": False,
    })
    hypothesis = with_freeze_hash({
        "hypothesis_id": DRY_HYP,
        "version": 1,
        "conceptual_family": DRY_FAM,
        "creation_source": "R0_8_INFRASTRUCTURE_DRY_RUN",
        "created_at_utc": FIXED_TS,
        "scientific_proposition": "A fixed causal D0 test state can traverse the governed research workflow end to end.",
        "trading_proposition": "NON_PROMOTABLE DRY RUN ONLY: candidate direction is used solely to exercise execution and position-state plumbing.",
        "relationship_type": "MARGINAL",
        "parent_hypothesis_ids": [],
        "derived_from_results": [],
        "datasets_that_influenced_design": ["D0_INFRASTRUCTURE_ONLY"],
        "material_spec": {
            "market_states": ["raw buyer_is_maker false-vs-true quantity balance over fixed trailing 10s; no directional interpretation"],
            "information_clocks": {"test_state": "R0.5 information_available_ts_ns"},
            "eligibility_rules": {"clock_available": True, "fine_history_status": "AVAILABLE", "book_status": "AVAILABLE", "directional_audit_fill_complete": True, "future_fields_used": False},
            "decision_clock": "R0.5 information_available_ts_ns",
            "latency_model": "R0.6 ZERO_LATENCY_AUDIT_ONLY; infrastructure test, not realistic strategy latency",
            "entry_rule": "R0.6 marketable visible-depth audit fill at causal arrival",
            "position_state_machine": "FLAT-only; enter on state=true; ignore all candidates while fixed 5m position is open",
            "research_notional": "R0.6 1,000 USDT AUDIT-ONLY notional; non-promotable",
            "exit_rule": "fixed 5m D0 label endpoint for plumbing only; no exit-depth claim",
            "risk_rules": ["max one concurrent position", "dry-run classification forced non-promotable"],
            "cost_model": "R0.6 entry crossing cost only; fee/funding/other=0 mechanical test; NOT realistic P&L",
            "primary_outcome": "5m D0 directional-return label, attached only after causal eligibility is frozen",
            "primary_economic_metric": "mechanical return proxy for integration only; not economic evidence",
            "registered_benchmarks": ["state=false within the same fixed D0 plumbing family"],
            "minimum_economically_meaningful_effect": "NOT APPLICABLE FOR PROMOTION; R0.8 is forced SCIENTIFIC_ONLY regardless observed effect",
            "risk_gates": ["No result may promote this synthetic hypothesis; integration invariants must pass"],
            "inference_plan": "equal-weight within-cluster state contrast; deterministic bootstrap over frozen R0.4 conservative clusters",
            "dependence_episode_definition": {"deterministic": True, "primary_inference_unit": "R0.4 conservative_inference_cluster_id", "plausible_alternatives": [], "governing_conservative_rule": "R0.4 conservative cluster governs the dry-run statistic"},
            "evidence_budget_id": DRY_BUDGET,
            "minimum_support": "at least two within-cluster contrasts for an interval; otherwise INDETERMINATE",
            "independent_support_precision_plan": {"effect_assumption": "none; infrastructure only", "do_not_spend_threshold": "D1 spending prohibited for R0.8 synthetic hypothesis"},
            "primary_failure_criterion": "any causal, reproducibility, execution, dependence, registry, or protection invariant fails",
            "d1_feasibility_plan": {"permitted_gate_outputs": ["PASS", "FAIL"], "detailed_access_restrictions": "no real D1 access in R0.8", "bounded_fallback_tree": [], "same_cohort_redesign_forbidden": True},
            "post_unblinding_defect_rule": "not applicable because R0.8 never unblinds real D1; material defects still require new dry-run build/version",
            "code_commit": code_commit,
            "artifact_hashes": parent_hashes,
            "confirmation_plan": {"confirmation_batch_id": DRY_BATCH, "protected_cohort_id": DRY_COHORT, "protocol_version": "1.2", "immutable_d1_data_version": "DRY-RUN-NO-DATA", "unblinding_rule": "No actual unblinding; synthetic registry rehearsal only"},
        },
        "adversarial_review": {
            "review_id": "AR-R08DRY001", "reviewer_context_id": "CTX-R08DRY-FRESH",
            "frozen_packet_hash": sha256_canonical({"hypothesis": DRY_HYP, "purpose": "integration-only"}),
            "fresh_context": True, "outcome_blind": True, "participated_in_discovery": False,
            "unresolved_objections": [], "objection_dispositions": [],
        },
    })
    cohort = with_freeze_hash({
        "cohort_id": DRY_COHORT, "role": "CONFIRMATION", "status": "FROZEN",
        "start_date_utc": "2099-01-01", "end_date_utc": "2099-01-02",
        "data_version": "DRY-RUN-NO-DATA", "canonical_builder_commit": "DRY-RUN-NO-DATA",
        "schema_version": "DRY-RUN-NO-DATA", "source_manifest_hashes": {"synthetic": sha256_canonical("NO_REAL_PROTECTED_DATA")},
        "timezone_contract": "UTC", "unit_contract": "epoch nanoseconds int64",
        "known_coverage_gaps": ["This is a synthetic isolated cohort record; no protected data exists or is accessed."],
        "intended_sequence": "R0.8 registry rehearsal only; never eligible for scientific confirmation",
        "quarantine_status": "SYNTHETIC_NO_DATA", "freeze_ts_utc": FIXED_TS,
    })
    batch = with_freeze_hash({
        "confirmation_batch_id": DRY_BATCH, "protocol_version": "1.2", "cohort_id": DRY_COHORT,
        "cohort_manifest_ref": f"{DRY_COHORT}@DRY-RUN-NO-DATA",
        "protected_period_architecture_hash": cohort["freeze_hash"], "data_version": "DRY-RUN-NO-DATA",
        "freeze_ts_utc": FIXED_TS, "hypothesis_versions": [f"{DRY_HYP}@1"], "family_ids": [DRY_FAM],
        "evidence_budget_ids": [DRY_BUDGET], "outcome_access_roster": ["CONFIRMATION_RUNNER:R08-DRY-RUNNER"],
        "code_commit": code_commit, "artifact_hashes": {"hypothesis": hypothesis["freeze_hash"], "evidence_budget": budget["freeze_hash"]},
        "unblinding_rule": "Never unblind real data; synthetic R0.8 registry rehearsal only", "status": "FROZEN",
    })
    events = []
    for i, stage in enumerate(("IDEA", "FEASIBLE", "EXPLORATORY_CANDIDATE", "REGISTERED_CANDIDATE"), 1):
        events.append({"event_id": f"HE-R08DRY00{i}", "event_ts_utc": FIXED_TS, "hypothesis_version": f"{DRY_HYP}@1", "event_type": "PROMOTE", "to_stage": stage, "rationale": "R0.8 synthetic lifecycle rehearsal; no scientific promotion claim."})
    events.append({"event_id": "HE-R08DRY005", "event_ts_utc": FIXED_TS, "hypothesis_version": f"{DRY_HYP}@1", "event_type": "CLASSIFY", "classification": "SCIENTIFIC_ONLY", "rationale": "R0.8 is infrastructure-only and terminates non-promotable regardless D0 statistic."})
    access = [{
        "access_event_id": "PA-R08DRY001", "event_ts_utc": FIXED_TS, "actor_id": "R08-DRY-RUNNER",
        "role": "CONFIRMATION_RUNNER", "cohort_id": "P1_PROTECTED", "content_class": "FUTURE_OUTCOMES",
        "purpose": "negative-control attempt proving P1 outcomes remain sealed", "allowed": False,
        "contamination_before": "CLEAN", "contamination_after": "CLEAN",
    }]
    decisions = [{
        "decision_id": "DEC-R08DRY001", "decision_ts_utc": FIXED_TS,
        "decision": "Terminate the synthetic R0.8 hypothesis as SCIENTIFIC_ONLY regardless its D0 statistic.",
        "rationale": "The dry run exists to validate machinery, not discover or promote a trading signal.",
        "relevant_outcomes_seen": True, "datasets_considered": ["D0_ONLY"],
    }]
    return {"budget": budget, "hypothesis": hypothesis, "cohort": cohort, "batch": batch, "events": events, "access": access, "decisions": decisions}


def write_isolated_registries(root: Path, records: dict[str, Any]) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    mapping = {
        "evidence_budgets.jsonl": [records["budget"]],
        "hypothesis_versions.jsonl": [records["hypothesis"]],
        "protected_cohorts.jsonl": [records["cohort"]],
        "confirmation_batches.jsonl": [records["batch"]],
        "hypothesis_events.jsonl": records["events"],
        "access_events.jsonl": records["access"],
        "decision_log.jsonl": records["decisions"],
    }
    validators: dict[str, Callable[[Any], None]] = {
        "evidence_budgets.jsonl": validate_evidence_budget,
        "hypothesis_versions.jsonl": validate_registered_spec,
        "protected_cohorts.jsonl": validate_protected_cohort,
        "confirmation_batches.jsonl": validate_confirmation_batch,
    }
    for name, rows in mapping.items():
        p = root / name
        if p.exists(): p.unlink()
        for row in rows:
            if name in validators: validators[name](row)
            append_jsonl(p, row)
    validate_genealogy(read_jsonl(root / "hypothesis_versions.jsonl"))
    validate_event_history(read_jsonl(root / "hypothesis_events.jsonl"))
    validate_access_event_history(read_jsonl(root / "access_events.jsonl"))
    validate_decision_log(read_jsonl(root / "decision_log.jsonl"))
    return {name: sha256_file(root / name) for name in mapping}


def forbidden_action_matrix(records: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    def expect_exception(name: str, fn: Callable[[], Any]) -> None:
        try:
            fn()
        except Exception as exc:
            results.append({"check": name, "blocked": True, "detail": f"{type(exc).__name__}: {exc}"})
        else:
            results.append({"check": name, "blocked": False, "detail": "FORBIDDEN ACTION WAS NOT BLOCKED"})

    d = authorize_protected_access(role="CONFIRMATION_RUNNER", zone="P1_PROTECTED", content_class="FUTURE_OUTCOMES", batch_frozen=True, outcome_access_authorized=True, clean_confirmation_context=True)
    results.append({"check": "P1 outcome access before real D1 freeze", "blocked": not d.allowed, "detail": d.reason})
    d = authorize_protected_access(role="DISCOVERY_AGENT", zone="D1", content_class="INTEGRITY_METADATA")
    results.append({"check": "Discovery Agent protected D1 metadata access", "blocked": not d.allowed, "detail": d.reason})
    expect_exception("future-derived predictor", lambda: assert_predictor_columns_safe(["future_return_5m"]))
    expect_exception("future-derived eligibility", lambda: assert_eligibility_fields_safe(["future_outcome_available"]))
    expect_exception("random cross-validation", lambda: assert_time_respecting_validation("random_split"))
    mutated = dict(records["hypothesis"]); mutated["scientific_proposition"] = "silently changed"
    mutated = with_freeze_hash({k: v for k, v in mutated.items() if k != "freeze_hash"})
    expect_exception("same-version material mutation", lambda: validate_material_change(records["hypothesis"], mutated))
    rescue = list(records["events"]) + [{"event_id": "HE-R08DRY006", "event_ts_utc": FIXED_TS, "hypothesis_version": f"{DRY_HYP}@1", "event_type": "PROMOTE", "to_stage": "REGISTERED_HOLDOUT_SURVIVOR", "rationale": "forbidden rescue attempt"}]
    expect_exception("terminal-version rescue", lambda: validate_event_history(rescue))
    # Positive control: the governed alternative must still be usable.
    assert_time_respecting_validation("PURGED_WALK_FORWARD")
    if not all(bool(r["blocked"]) for r in results):
        raise RuntimeError("one or more R0.8 forbidden-action controls failed open")
    return results


def self_test() -> int:
    core = pd.DataFrame([
        {"candidate_id": "A", "direction": 1, "information_available_ts_ns": 10_000_000_000, "clock_available": True},
        {"candidate_id": "B", "direction": -1, "information_available_ts_ns": 400_000_000_000, "clock_available": True},
        {"candidate_id": "C", "direction": 1, "information_available_ts_ns": 800_000_000_000, "clock_available": True},
    ])
    buckets = []
    for cid, positive in (("A", True), ("B", False), ("C", True)):
        for o in range(-60, 0):
            buckets.append({"candidate_id": cid, "bucket_offset_sec": o, "source_status": "AVAILABLE", "buyer_maker_true_quantity": 1.0 if positive else 2.0, "buyer_maker_false_quantity": 2.0 if positive else 1.0})
    cov = pd.DataFrame([{"candidate_id": x, "fine_history_status": "AVAILABLE"} for x in "ABC"])
    exe = pd.DataFrame([{"candidate_id": x, "book_status": "AVAILABLE", "buy_fill_complete": True, "sell_fill_complete": True, "buy_raw_crossing_cost_bps": 1.0, "sell_raw_crossing_cost_bps": 1.0} for x in "ABC"])
    state = build_fixed_test_state(pd.DataFrame(buckets))
    pop = build_causal_analysis_population(core, cov, exe, state)
    before = pop["trading_eligible"].tolist()
    outcomes = pd.DataFrame([
        {"candidate_id": "A", "outcome_horizon_sec": 300, "outcome_status": "AVAILABLE", "outcome_observable": True, "directional_return_to_last_trade_bps": 5.0, "mae_from_decision_bps": 2.0},
        {"candidate_id": "B", "outcome_horizon_sec": 300, "outcome_status": "D0_BOUNDARY_CENSOR", "outcome_observable": False, "directional_return_to_last_trade_bps": None, "mae_from_decision_bps": None},
        {"candidate_id": "C", "outcome_horizon_sec": 300, "outcome_status": "AVAILABLE", "outcome_observable": True, "directional_return_to_last_trade_bps": 1.0, "mae_from_decision_bps": 3.0},
    ])
    dep = pd.DataFrame([{"candidate_id": x, "conservative_inference_cluster_id": "K1" if x != "C" else "K2"} for x in "ABC"])
    attached = attach_future_observability(pop, outcomes, dep)
    assert before == attached["trading_eligible"].tolist()
    positions = add_mechanical_net_proxy(apply_flat_only_position_state_machine(attached))
    assert positions["position_entered"].sum() == 2
    rec = make_dry_run_registry_records({"parent": "0" * 64}, "SELFTEST")
    validate_evidence_budget(rec["budget"]); validate_registered_spec(rec["hypothesis"]); validate_protected_cohort(rec["cohort"]); validate_confirmation_batch(rec["batch"])
    validate_event_history(rec["events"]); validate_access_event_history(rec["access"]); validate_decision_log(rec["decisions"])
    assert all(r["blocked"] for r in forbidden_action_matrix(rec))
    print("SELF-TEST PASS: causal eligibility remains outcome-blind, position-state plumbing works, isolated registries validate, and forbidden paths fail closed.")
    return 0


def run(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    print("[R0.8] stage 1/6: verifying P1 seal, production registries, and accepted parent hashes", flush=True)
    assert_p1_sealed(args.protected_register)
    prod_before = snapshot_production_registries(args.production_registry_root)

    r03, r03m = load_manifest(args.r03_root, "r0_corrected_d0_substrate_v1")
    r04, r04m = load_manifest(args.r04_root, "r0_dependence_framework_v1")
    r05, r05m = load_manifest(args.r05_root, "r0_causal_market_state_v1")
    r06, r06m = load_manifest(args.r06_root, "r0_execution_simulator_v1")
    if int(r03.get("counts", {}).get("candidate_rows", -1)) != EXPECTED_CANDIDATES:
        raise RuntimeError("R0.3 accepted candidate count changed")
    if int(r05.get("counts", {}).get("governed_clock_rows", -1)) != EXPECTED_GOVERNED:
        raise RuntimeError("R0.5 governed clock count changed")

    p_out = verify_output(args.r03_root, r03, "outcomes", "future/outcomes.parquet")
    p_dep = verify_output(args.r04_root, r04, "candidate_dependence", "dependence/candidate_dependence.parquet")
    p_core = verify_output(args.r05_root, r05, "candidate_core", "causal/candidate_core.parquet")
    p_buckets = verify_output(args.r05_root, r05, "trade_buckets_1s", "causal/trade_buckets_1s.parquet")
    p_cov = verify_output(args.r05_root, r05, "coverage", "quality/trade_coverage.parquet")
    p_exec = verify_output(args.r06_root, r06, "audit_execution_state", "execution/audit_zero_latency_state.parquet")
    parent_hashes = {
        "r03_manifest": sha256_file(r03m), "r03_outcomes": sha256_file(p_out),
        "r04_manifest": sha256_file(r04m), "r04_dependence": sha256_file(p_dep),
        "r05_manifest": sha256_file(r05m), "r05_candidate_core": sha256_file(p_core),
        "r05_trade_buckets_1s": sha256_file(p_buckets), "r05_trade_coverage": sha256_file(p_cov),
        "r06_manifest": sha256_file(r06m), "r06_execution_state": sha256_file(p_exec),
    }
    print("[R0.8] stage 1/6 complete: parent artifacts verified; production registries snapshotted", flush=True)

    print("[R0.8] stage 2/6: building fixed causal test state and freezing eligibility before outcomes", flush=True)
    core = pq.read_table(p_core).to_pandas()
    buckets = pq.read_table(p_buckets).to_pandas()
    coverage = pq.read_table(p_cov).to_pandas()
    execution = pq.read_table(p_exec).to_pandas()
    feature = build_fixed_test_state(buckets)
    causal = build_causal_analysis_population(core, coverage, execution, feature)
    if len(causal) != EXPECTED_CANDIDATES or causal["candidate_id"].duplicated().any():
        raise RuntimeError("R0.8 causal population failed candidate preservation")
    eligible_count = int(causal["trading_eligible"].sum())
    print(f"[R0.8] stage 2/6 complete: {eligible_count:,}/{len(causal):,} causally eligible before outcome attachment", flush=True)

    print("[R0.8] stage 3/6: attaching D0 observability separately and applying dependence-aware descriptive analysis", flush=True)
    outcomes = pq.read_table(p_out, filters=[("outcome_horizon_sec", "=", DRY_RUN_HORIZON_SEC)]).to_pandas()
    dep = pq.read_table(p_dep, columns=["candidate_id", "conservative_inference_cluster_id"]).to_pandas()
    analysis = attach_future_observability(causal, outcomes, dep)
    positions = add_mechanical_net_proxy(apply_flat_only_position_state_machine(analysis))
    contrast = dependence_aware_contrast(positions)
    print(f"[R0.8] stage 3/6 complete: observed={int(positions['analysis_observed'].sum()):,}, positions={int(positions['position_entered'].sum()):,}, independent contrast units={contrast.independent_units}", flush=True)

    print("[R0.8] stage 4/6: rehearsing isolated append-only registries and lifecycle", flush=True)
    code_commit = git_head()
    records = make_dry_run_registry_records(parent_hashes, code_commit)
    registry_hashes = write_isolated_registries(args.output_root / "isolated_registries", records)
    print("[R0.8] stage 4/6 complete: synthetic hypothesis terminates SCIENTIFIC_ONLY; production registries untouched", flush=True)

    print("[R0.8] stage 5/6: red-team negative controls", flush=True)
    forbidden = forbidden_action_matrix(records)
    prod_after = snapshot_production_registries(args.production_registry_root)
    if prod_before != prod_after:
        raise RuntimeError("R0.8 modified production R0.7 registries")
    print(f"[R0.8] stage 5/6 complete: {len(forbidden)}/{len(forbidden)} forbidden-action controls fail closed", flush=True)

    print("[R0.8] stage 6/6: writing deterministic dry-run artifacts and manifest", flush=True)
    candidate_cols = [
        "candidate_id", "direction", "information_available_ts_ns", "clock_available", "feature_available",
        "dry_run_state_score", "dry_run_state_true", "fine_history_status", "book_status", "trading_eligible",
        "outcome_status", "outcome_observable", "analysis_observed", "conservative_inference_cluster_id",
    ]
    position_cols = candidate_cols + [
        "position_action", "position_entered", "planned_exit_ts_ns", "directional_return_to_last_trade_bps",
        "mae_from_decision_bps", "dry_run_entry_cost_bps", "dry_run_mechanical_net_proxy_bps",
    ]
    candidate_path = args.output_root / "analysis" / "candidate_dry_run.parquet"
    position_path = args.output_root / "analysis" / "position_path.parquet"
    atomic_parquet(candidate_path, positions[candidate_cols].sort_values("candidate_id", kind="mergesort"))
    atomic_parquet(position_path, positions[position_cols].sort_values(["information_available_ts_ns", "candidate_id"], kind="mergesort"))
    atomic_csv(args.report_root / "forbidden_actions.csv", pd.DataFrame(forbidden))

    entered_observed = positions[positions["position_entered"].fillna(False).astype(bool) & positions["analysis_observed"].fillna(False).astype(bool)].copy()
    observed_mae = entered_observed["mae_from_decision_bps"].dropna()
    # R0.3 defines MAE as a non-negative adverse-excursion magnitude:
    # max(0, -min(directional_path_return)). Therefore the worst MAE is
    # the maximum magnitude, not the minimum. Fail closed if upstream
    # semantics ever drift to negative values.
    if (observed_mae < 0).any():
        raise RuntimeError("R0.8 expected R0.3 mae_from_decision_bps to be a non-negative magnitude")
    summary = {
        "candidate_rows": int(len(positions)),
        "governed_clock_rows": int(positions["clock_available"].fillna(False).astype(bool).sum()),
        "causally_eligible_rows": eligible_count,
        "outcome_observed_rows": int(positions["analysis_observed"].sum()),
        "state_true_rows": int(positions["dry_run_state_true"].fillna(False).astype(bool).sum()),
        "positions_entered": int(positions["position_entered"].sum()),
        "positions_with_observed_outcome": int(len(entered_observed)),
        "position_mean_mechanical_net_proxy_bps": None if entered_observed.empty else float(entered_observed["dry_run_mechanical_net_proxy_bps"].dropna().mean()),
        "position_worst_mae_bps": None if observed_mae.empty else float(observed_mae.max()),
        "ignored_while_position_open": int(positions["position_action"].eq("IGNORED_WHILE_POSITION_OPEN").sum()),
        "contrast_status": contrast.status,
        "independent_contrast_units": int(contrast.independent_units),
        "contrast_estimate_bps": contrast.estimate_bps,
        "contrast_ci_low_bps": contrast.ci_low_bps,
        "contrast_ci_high_bps": contrast.ci_high_bps,
        "forbidden_controls_passed": int(sum(bool(x["blocked"]) for x in forbidden)),
        "forbidden_controls_total": int(len(forbidden)),
        "production_registry_mutations": 0,
        "p1_read": False,
        "p1_outcome_read": False,
        "classification": "SCIENTIFIC_ONLY",
    }
    candidate_hash = dataframe_substantive_hash(positions, candidate_cols)
    position_hash = dataframe_substantive_hash(positions, position_cols)
    substantive_payload = {
        "schema_name": SCHEMA_NAME, "builder_version": BUILDER_VERSION,
        "parents": parent_hashes, "registry_hashes": registry_hashes,
        "candidate_substantive_hash": candidate_hash, "position_substantive_hash": position_hash,
        "summary": summary,
    }
    substantive_run_hash = sha256_canonical(substantive_payload)
    impl_fingerprint = implementation_fingerprint()
    old_manifest_path = args.output_root / "metadata" / "manifest.json"
    if old_manifest_path.exists():
        old = json.loads(old_manifest_path.read_text(encoding="utf-8"))
        enforce_same_implementation_idempotence(
            old,
            current_implementation_sha256=impl_fingerprint["sha256"],
            parent_hashes=parent_hashes,
            substantive_run_hash=substantive_run_hash,
        )

    manifest = {
        "schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION, "builder_version": BUILDER_VERSION,
        "generated_at_utc": utc_now_iso(), "evidentiary_role": "R0_INTEGRATION_DRY_RUN_D0_ONLY_NON_PROMOTABLE",
        "code_commit_at_run": code_commit, "implementation_fingerprint": impl_fingerprint, "parents": parent_hashes,
        "contracts": {
            "feature": "fixed trailing 10s raw buyer_is_maker balance; threshold 0; no directional/scientific interpretation",
            "eligibility": "causal clock + R0.5 fine-history availability + R0.6 sequence-valid directional audit fill; no future fields",
            "outcome": "R0.3 5m D0 outcome attached only after eligibility; observability tracked separately",
            "dependence": "R0.4 conservative_inference_cluster_id governs descriptive contrast",
            "execution": "R0.6 ZERO_LATENCY_AUDIT_ONLY and audit crossing cost; not realistic strategy execution",
            "position_state_machine": "FLAT-only, fixed 5m; repeated/opposite candidates ignored while open",
            "classification": "forced SCIENTIFIC_ONLY regardless D0 effect",
        },
        "protection": {
            "p1_status_verified": "SEALED_OUTCOME_UNSEEN", "p1_read": False, "p1_outcome_read": False,
            "production_registry_hashes_before": prod_before, "production_registry_hashes_after": prod_after,
            "production_registries_unchanged": prod_before == prod_after,
        },
        "outputs": {
            "candidate_dry_run": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
            "position_path": {"path": str(position_path), "sha256": sha256_file(position_path)},
            "forbidden_actions": {"path": str(args.report_root / "forbidden_actions.csv"), "sha256": sha256_file(args.report_root / "forbidden_actions.csv")},
        },
        "isolated_registry_hashes": registry_hashes,
        "candidate_substantive_hash": candidate_hash, "position_substantive_hash": position_hash,
        "substantive_run_hash": substantive_run_hash, "summary": summary,
        "status": "BUILD_COMPLETE_REVIEW_REQUIRED",
    }
    atomic_json(old_manifest_path, manifest)
    atomic_json(args.report_root / "summary.json", manifest)
    print(json.dumps({"status": manifest["status"], "substantive_run_hash": substantive_run_hash, **summary}, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--r03-root", type=Path, default=REPO_ROOT / "data/derived/perps/research/r0_corrected_d0_substrate_v1")
    p.add_argument("--r04-root", type=Path, default=REPO_ROOT / "data/derived/perps/research/r0_dependence_framework_v1")
    p.add_argument("--r05-root", type=Path, default=REPO_ROOT / "data/derived/perps/research/r0_causal_market_state_v1")
    p.add_argument("--r06-root", type=Path, default=REPO_ROOT / "data/derived/perps/research/r0_execution_simulator_v1")
    p.add_argument("--production-registry-root", type=Path, default=REPO_ROOT / "research/governance/registries")
    p.add_argument("--protected-register", type=Path, default=REPO_ROOT / "research/governance/protected_data_register.yaml")
    p.add_argument("--output-root", type=Path, default=REPO_ROOT / "data/derived/perps/research/r0_full_d0_dry_run_v1")
    p.add_argument("--report-root", type=Path, default=REPO_ROOT / "reports/r0_full_d0_dry_run_v1")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    raise SystemExit(self_test() if a.self_test else run(a))
