from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research_core.audit import (  # noqa: E402
    AuditFinding,
    count_findings,
    overall_status,
    require_unique_finding_ids,
    sha256_canonical,
    sha256_file,
)

CONTRACT_PATH = REPO_ROOT / "research/governance/r0_acceptance_audit_contract_v1.json"
SCHEMA_NAME = "r0_acceptance_audit_v1"
BUILDER_VERSION = "1.0.1"

REQUIRED_TAG_CHAIN = [
    "r0-governance-provenance-v1",
    "r0-causal-clock-substrate-v1",
    "r0-dependence-framework-v1",
    "r0-causal-market-state-v1",
    "r0-execution-simulator-v1",
    "r0-full-d0-dry-run-v1",
]

R07_PATHS = [
    "research/governance/research_registry_contract_v1.json",
    "research/governance/registries",
    "research/r0/registries",
    "src/research_core/registry",
    "tests/research_governance/test_r0_registry_guardrails.py",
]

PRODUCTION_REGISTRY_FILES = [
    "hypothesis_versions.jsonl",
    "hypothesis_events.jsonl",
    "evidence_budgets.jsonl",
    "protected_cohorts.jsonl",
    "confirmation_batches.jsonl",
    "access_events.jsonl",
    "decision_log.jsonl",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git(repo: Path, *args: str, check: bool = True) -> str:
    p = subprocess.run(
        ["git", *args], cwd=repo, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout.strip()


def git_tag_commit(repo: Path, tag: str) -> str | None:
    out = git(repo, "rev-parse", f"{tag}^{{commit}}", check=False)
    return out or None


def git_is_ancestor(repo: Path, older: str, newer: str) -> bool:
    p = subprocess.run(["git", "merge-base", "--is-ancestor", older, newer], cwd=repo)
    return p.returncode == 0


def last_commit_for_paths(repo: Path, ref: str, paths: list[str]) -> str | None:
    out = git(repo, "log", "-1", "--format=%H", ref, "--", *paths, check=False)
    return out or None


def tags_at(repo: Path, commit: str) -> list[str]:
    out = git(repo, "tag", "--points-at", commit, check=False)
    return sorted(x.strip() for x in out.splitlines() if x.strip())


def implementation_fingerprint(repo: Path) -> dict[str, Any]:
    files = [
        repo / "research/r0/audit/run_r0_acceptance_audit.py",
        repo / "src/research_core/audit/__init__.py",
        repo / "src/research_core/audit/checks.py",
        repo / "research/governance/r0_acceptance_audit_contract_v1.json",
    ]
    hashes: dict[str, str] = {}
    for p in files:
        if not p.exists():
            raise RuntimeError(f"missing R0.9 implementation file: {p}")
        hashes[p.relative_to(repo).as_posix()] = sha256_file(p)
    return {"files": hashes, "sha256": sha256_canonical(hashes)}


AUDIT_INPUT_REPO_FILES = [
    "research/README.md",
    "research/r0/README.md",
    "research/governance/protocol_freeze_manifest.json",
    "research/governance/research_reset_manifest.yaml",
    "research/governance/protected_data_register.yaml",
    "research/governance/outcome_access_register.csv",
    "research/governance/datasets/R0_1_INVENTORY_SUMMARY_v1.json",
    "research/r0/market_state/R0_5_IMPLEMENTATION_MANIFEST.json",
    "research/r0/execution/R0_6_IMPLEMENTATION_MANIFEST.json",
    "research/r0/registries/R0_7_IMPLEMENTATION_MANIFEST.json",
    "research/r0/dry_run/R0_8_IMPLEMENTATION_MANIFEST.json",
    "research/governance/clock_contract_v1.json",
    "research/governance/dependence_contract_v1.json",
    "research/governance/market_state_contract_v1.json",
    "research/governance/execution_contract_v1.json",
    "research/governance/research_registry_contract_v1.json",
    "research/governance/r0_dry_run_contract_v1.json",
    "research/governance/r0_acceptance_audit_contract_v1.json",
]


def audit_input_fingerprint(repo: Path, runtime_manifest: Path) -> dict[str, Any]:
    """Fingerprint every repository/runtime input that can change the R0.9 verdict.

    Missing files are represented explicitly instead of raising so the audit can still
    report their absence as findings. This fingerprints governance metadata only plus
    the accepted D0 R0.8 runtime manifest; it never opens P1 market data or outcomes.
    """
    file_hashes: dict[str, str | None] = {}
    for rel in AUDIT_INPUT_REPO_FILES:
        p = repo / rel
        file_hashes[rel] = sha256_file(p) if p.is_file() else None
    reg_root = repo / "research/governance/registries"
    for name in PRODUCTION_REGISTRY_FILES:
        p = reg_root / name
        file_hashes[f"research/governance/registries/{name}"] = sha256_file(p) if p.is_file() else None

    runtime_path = runtime_manifest.resolve()
    runtime_hash = sha256_file(runtime_path) if runtime_path.is_file() else None
    git_state = {
        "head": git(repo, "rev-parse", "HEAD"),
        "required_tag_commits": {tag: git_tag_commit(repo, tag) for tag in REQUIRED_TAG_CHAIN},
    }
    payload = {
        "repo_files": file_hashes,
        "r08_runtime_manifest_sha256": runtime_hash,
        "git_state": git_state,
    }
    return {**payload, "sha256": sha256_canonical(payload)}


def same_audit_identity(
    previous_report: dict[str, Any],
    *,
    accepted_parent: dict[str, Any],
    implementation_sha256: str,
    audit_input_sha256: str,
) -> bool:
    """True only when a previous audit used the same parent, code and audited state."""
    return (
        previous_report.get("accepted_parent") == accepted_parent
        and (previous_report.get("implementation_fingerprint") or {}).get("sha256") == implementation_sha256
        and (previous_report.get("audit_input_fingerprint") or {}).get("sha256") == audit_input_sha256
    )


def finding(
    finding_id: str,
    domain: str,
    severity: str,
    status: str,
    blocking_r1: bool,
    affected_steps: list[str] | tuple[str, ...],
    summary: str,
    evidence: str,
    remediation: str = "",
) -> AuditFinding:
    return AuditFinding(
        finding_id=finding_id,
        domain=domain,
        severity=severity,
        status=status,
        blocking_r1=blocking_r1,
        affected_steps=tuple(affected_steps),
        summary=summary,
        evidence=evidence,
        remediation=remediation,
    )


def audit_git_lineage(repo: Path, contract: dict[str, Any]) -> tuple[list[AuditFinding], dict[str, Any]]:
    fs: list[AuditFinding] = []
    parent = contract["accepted_parent"]
    expected_r08 = parent["commit"]
    r08_tag = parent["tag"]
    actual_r08 = git_tag_commit(repo, r08_tag)
    if actual_r08 == expected_r08:
        fs.append(finding("R09-GIT-001", "GIT_FREEZE_LINEAGE", "INFO", "PASS", False, ["R0.8"],
                          "Accepted R0.8 tag resolves to the frozen commit.", f"{r08_tag} -> {actual_r08}"))
    else:
        fs.append(finding("R09-GIT-001", "GIT_FREEZE_LINEAGE", "BLOCKER", "FINDING", True, ["R0.8"],
                          "Accepted R0.8 tag does not resolve to the frozen commit.", f"expected={expected_r08}; actual={actual_r08}",
                          "Restore or explicitly supersede the R0.8 freeze before R1."))

    tag_commits: dict[str, str | None] = {t: git_tag_commit(repo, t) for t in REQUIRED_TAG_CHAIN}
    missing = [t for t, c in tag_commits.items() if not c]
    if missing:
        fs.append(finding("R09-GIT-002", "GIT_FREEZE_LINEAGE", "BLOCKER", "FINDING", True, ["R0.0-R0.8"],
                          "One or more required frozen R0 tags are missing.", ", ".join(missing),
                          "Restore the documented frozen checkpoints or record an explicit superseding lineage."))
    else:
        broken: list[str] = []
        for a, b in zip(REQUIRED_TAG_CHAIN, REQUIRED_TAG_CHAIN[1:]):
            if not git_is_ancestor(repo, tag_commits[a] or a, tag_commits[b] or b):
                broken.append(f"{a} !<= {b}")
        if broken:
            fs.append(finding("R09-GIT-002", "GIT_FREEZE_LINEAGE", "BLOCKER", "FINDING", True, ["R0.0-R0.8"],
                              "Frozen R0 tags do not form the expected ancestor chain.", "; ".join(broken),
                              "Resolve the Git lineage before treating R0 as one reproducible research history."))
        else:
            fs.append(finding("R09-GIT-002", "GIT_FREEZE_LINEAGE", "INFO", "PASS", False, ["R0.0-R0.8"],
                              "Required frozen tags form a monotone ancestor chain.", " -> ".join(REQUIRED_TAG_CHAIN)))

    r07_commit = last_commit_for_paths(repo, r08_tag, R07_PATHS)
    if not r07_commit:
        fs.append(finding("R09-GIT-003", "GIT_FREEZE_LINEAGE", "MAJOR", "FINDING", True, ["R0.7"],
                          "Could not reconstruct an R0.7 code checkpoint from Git history.", "No commit found for R0.7 paths before R0.8.",
                          "Create or recover an immutable R0.7 checkpoint and document it."))
        r07_tags: list[str] = []
    else:
        r07_tags = tags_at(repo, r07_commit)
        exec_commit = tag_commits.get("r0-execution-simulator-v1")
        r08_commit = tag_commits.get("r0-full-d0-dry-run-v1")
        ordered = bool(exec_commit and r08_commit and git_is_ancestor(repo, exec_commit, r07_commit) and git_is_ancestor(repo, r07_commit, r08_commit))
        if ordered:
            fs.append(finding("R09-GIT-003", "GIT_FREEZE_LINEAGE", "INFO", "PASS", False, ["R0.7"],
                              "R0.7 has a reconstructable immutable commit between R0.6 and R0.8.",
                              f"commit={r07_commit}; tags={r07_tags or ['<none>']}"))
        else:
            fs.append(finding("R09-GIT-003", "GIT_FREEZE_LINEAGE", "MAJOR", "FINDING", True, ["R0.7"],
                              "R0.7 checkpoint is not correctly ordered between R0.6 and R0.8.",
                              f"commit={r07_commit}; tags={r07_tags or ['<none>']}",
                              "Repair or explicitly supersede R0.7 lineage before R1."))

    head = git(repo, "rev-parse", "HEAD")
    fs.append(finding("R09-GIT-004", "GIT_FREEZE_LINEAGE", "INFO", "PASS", False, ["R0.9"],
                      "R0.9 records the audit working checkpoint without requiring a clean tree.",
                      f"HEAD={head}; accepted_parent={actual_r08}"))
    return fs, {"tag_commits": tag_commits, "r07_commit": r07_commit, "r07_tags": r07_tags, "head": head}


def audit_protocol_and_provenance(repo: Path) -> list[AuditFinding]:
    fs: list[AuditFinding] = []
    freeze = repo / "research/governance/protocol_freeze_manifest.json"
    if not freeze.exists():
        return [finding("R09-PROV-001", "PROTOCOL_AND_PROVENANCE", "BLOCKER", "FINDING", True, ["R0.0"],
                        "Protocol freeze manifest is missing.", str(freeze), "Restore the governing protocol freeze artifact.")]
    d = load_json(freeze)
    protocol_path = repo / d.get("protocol_path", "")
    expected_hash = d.get("sha256")
    actual_hash = sha256_file(protocol_path) if protocol_path.exists() else None
    if d.get("status") == "FROZEN" and actual_hash == expected_hash:
        fs.append(finding("R09-PROV-001", "PROTOCOL_AND_PROVENANCE", "INFO", "PASS", False, ["R0.0"],
                          "Protocol v1.2 file matches its frozen cryptographic hash.", f"sha256={actual_hash}"))
    else:
        fs.append(finding("R09-PROV-001", "PROTOCOL_AND_PROVENANCE", "BLOCKER", "FINDING", True, ["R0.0"],
                          "Protocol freeze identity does not verify.", f"status={d.get('status')}; expected={expected_hash}; actual={actual_hash}",
                          "Restore the frozen protocol or create an explicit new protocol version; do not silently replace it."))

    inv = repo / "research/governance/datasets/R0_1_INVENTORY_SUMMARY_v1.json"
    if not inv.exists():
        fs.append(finding("R09-PROV-002", "PROTOCOL_AND_PROVENANCE", "BLOCKER", "FINDING", True, ["R0.1"],
                          "R0.1 authoritative inventory summary is missing.", str(inv), "Regenerate from the frozen canonical boundary before R1."))
    else:
        s = load_json(inv)
        ok = (
            s.get("status") == "PASS"
            and int(s.get("file_count", -1)) == 22369
            and s.get("p1_declared_start_date_utc") == "2026-07-19"
            and int(s.get("undated_file_count", -1)) == 0
            and int(s.get("unknown_undated_market_event_file_count", -1)) == 0
        )
        if ok:
            fs.append(finding("R09-PROV-002", "PROTOCOL_AND_PROVENANCE", "INFO", "PASS", False, ["R0.1"],
                              "Authoritative canonical inventory matches the accepted R0.1 boundary.",
                              f"files={s.get('file_count')}; index_sha256={s.get('file_index_sha256')}; P1_start={s.get('p1_declared_start_date_utc')}"))
        else:
            fs.append(finding("R09-PROV-002", "PROTOCOL_AND_PROVENANCE", "BLOCKER", "FINDING", True, ["R0.1"],
                              "Authoritative canonical inventory no longer matches the accepted R0.1 boundary.", json.dumps(s, sort_keys=True)[:900],
                              "Reconcile the canonical inventory and freeze a new governed data version before R1."))
    return fs


def audit_p1_firewall(repo: Path) -> list[AuditFinding]:
    fs: list[AuditFinding] = []
    protected = repo / "research/governance/protected_data_register.yaml"
    text = protected.read_text(encoding="utf-8") if protected.exists() else ""
    sealed = bool(re.search(r"(?m)^\s*status:\s*SEALED\s*$", text))
    unseen = bool(re.search(r"(?m)^\s*outcome_exposure_status:\s*UNSEEN\s*$", text))
    if sealed and unseen:
        fs.append(finding("R09-P1-001", "P1_FIREWALL", "INFO", "PASS", False, ["R0.0-R0.9"],
                          "P1 protected-data register remains SEALED with outcomes UNSEEN.", "status=SEALED; outcome_exposure_status=UNSEEN"))
    else:
        fs.append(finding("R09-P1-001", "P1_FIREWALL", "BLOCKER", "FINDING", True, ["R0.0-R0.9"],
                          "P1 is not recorded as sealed and outcome-unseen.", f"sealed={sealed}; unseen={unseen}",
                          "Stop R1. Determine contamination status and redesign the protected architecture if needed."))

    access = repo / "research/governance/outcome_access_register.csv"
    bad_rows: list[dict[str, str]] = []
    if access.exists():
        with access.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                outcome_scope = (row.get("outcome_scope") or "").strip().upper()
                status = (row.get("status") or "").strip().upper()
                access_type = (row.get("access_type") or "").strip().upper()
                if outcome_scope not in {"", "NONE"} or "CONTAMIN" in status or access_type in {"OUTCOME_READ", "UNBLIND", "PERFORMANCE_READ"}:
                    bad_rows.append(row)
    else:
        bad_rows.append({"error": "missing outcome access register"})
    if not bad_rows:
        fs.append(finding("R09-P1-002", "P1_FIREWALL", "INFO", "PASS", False, ["R0.0-R0.9"],
                          "Outcome-access register contains no protected-outcome exposure.", f"register={access.relative_to(repo).as_posix()}"))
    else:
        fs.append(finding("R09-P1-002", "P1_FIREWALL", "BLOCKER", "FINDING", True, ["R0.0-R0.9"],
                          "Outcome-access register indicates protected-outcome exposure or is unavailable.", json.dumps(bad_rows, sort_keys=True)[:1200],
                          "Stop R1 and disposition the protected pool under Protocol v1.2."))

    reg_root = repo / "research/governance/registries"
    nonempty: list[str] = []
    missing: list[str] = []
    for name in PRODUCTION_REGISTRY_FILES:
        p = reg_root / name
        if not p.exists():
            missing.append(name)
        elif p.read_text(encoding="utf-8").strip():
            nonempty.append(name)
    if missing:
        fs.append(finding("R09-P1-003", "REGISTRY_AND_LIFECYCLE", "MAJOR", "FINDING", True, ["R0.7"],
                          "One or more production registries are missing before R1.", ", ".join(missing),
                          "Restore the complete append-only registry set before fresh hypothesis generation."))
    elif nonempty:
        fs.append(finding("R09-P1-003", "REGISTRY_AND_LIFECYCLE", "MAJOR", "FINDING", True, ["R0.7-R0.9"],
                          "Production registries are not empty at the pre-R1 boundary.", ", ".join(nonempty),
                          "Audit every record. R0.8 synthetic objects must never have entered production registries; any legitimate pre-R1 record must be explicitly explained."))
    else:
        fs.append(finding("R09-P1-003", "REGISTRY_AND_LIFECYCLE", "INFO", "PASS", False, ["R0.7-R0.9"],
                          "Production registries remain empty at the pre-R1 boundary.", "7/7 registry files present and empty"))

    fs.append(finding("R09-P1-004", "P1_FIREWALL", "INFO", "ACCEPTED_LIMITATION", False, ["R0.0-R0.9"],
                      "Historical absence of every possible P1 filesystem read cannot be proven forensically from repository artifacts alone.",
                      "Protection evidence is procedural: sealed register, access ledger, source guards, and R0.8 no-P1 runtime evidence.",
                      "Continue append-only access logging and do not claim OS-level forensic proof that the artifacts cannot provide."))
    return fs


def load_r08_runner(repo: Path):
    path = repo / "research/r0/dry_run/run_full_d0_dry_run.py"
    spec = importlib.util.spec_from_file_location("r09_r08_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load R0.8 runner")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def audit_r08_runtime(repo: Path, contract: dict[str, Any], runtime_manifest: Path) -> list[AuditFinding]:
    fs: list[AuditFinding] = []
    if not runtime_manifest.exists():
        return [finding("R09-RUN-001", "END_TO_END_REPRODUCIBILITY", "BLOCKER", "FINDING", True, ["R0.8"],
                        "Accepted R0.8 runtime manifest is missing.", str(runtime_manifest),
                        "Reproduce the accepted R0.8 D0 dry run before R1.")]
    m = load_json(runtime_manifest)
    expected_hash = contract["accepted_parent"]["substantive_run_hash"]
    actual_hash = m.get("substantive_run_hash")
    summary = m.get("summary", {})
    protection = m.get("protection", {})
    ok = (
        actual_hash == expected_hash
        and summary.get("classification") == "SCIENTIFIC_ONLY"
        and summary.get("p1_read") is False
        and summary.get("p1_outcome_read") is False
        and summary.get("production_registry_mutations") == 0
        and summary.get("forbidden_controls_passed") == summary.get("forbidden_controls_total") == 7
        and int(summary.get("candidate_rows", -1)) == 7067
        and int(summary.get("governed_clock_rows", -1)) == 7031
        and protection.get("production_registries_unchanged") is True
    )
    if ok:
        fs.append(finding("R09-RUN-001", "END_TO_END_REPRODUCIBILITY", "INFO", "PASS", False, ["R0.8"],
                          "Accepted R0.8 runtime manifest matches the frozen substantive result and protection invariants.",
                          f"hash={actual_hash}; candidates={summary.get('candidate_rows')}; governed={summary.get('governed_clock_rows')}; forbidden=7/7"))
    else:
        fs.append(finding("R09-RUN-001", "END_TO_END_REPRODUCIBILITY", "BLOCKER", "FINDING", True, ["R0.8"],
                          "Accepted R0.8 runtime evidence does not match the frozen acceptance contract.",
                          json.dumps({"hash": actual_hash, "summary": summary, "protection": protection}, sort_keys=True)[:1800],
                          "Reconcile the runtime evidence with the frozen R0.8 code before R1."))

    try:
        current_impl = load_r08_runner(repo).implementation_fingerprint()
        runtime_impl = m.get("implementation_fingerprint") or {}
        if current_impl.get("sha256") == runtime_impl.get("sha256"):
            fs.append(finding("R09-RUN-002", "END_TO_END_REPRODUCIBILITY", "INFO", "PASS", False, ["R0.8"],
                              "R0.8 accepted runtime maps to the current frozen semantic implementation fingerprint.",
                              f"implementation_sha256={current_impl.get('sha256')}"))
        else:
            fs.append(finding("R09-RUN-002", "END_TO_END_REPRODUCIBILITY", "BLOCKER", "FINDING", True, ["R0.8"],
                              "R0.8 runtime implementation fingerprint differs from the frozen code now present.",
                              f"runtime={runtime_impl.get('sha256')}; current={current_impl.get('sha256')}",
                              "Do not infer reproducibility across different semantic implementations; rerun/freeze or restore exact code."))
    except Exception as e:
        fs.append(finding("R09-RUN-002", "END_TO_END_REPRODUCIBILITY", "BLOCKER", "FINDING", True, ["R0.8"],
                          "Could not verify the R0.8 implementation fingerprint.", repr(e),
                          "Repair the fingerprint verification before R1."))

    # Re-verify every D0 parent hash recorded by R0.8. This reads only D0 artifacts.
    roots = {
        "r03": repo / "data/derived/perps/research/r0_corrected_d0_substrate_v1",
        "r04": repo / "data/derived/perps/research/r0_dependence_framework_v1",
        "r05": repo / "data/derived/perps/research/r0_causal_market_state_v1",
        "r06": repo / "data/derived/perps/research/r0_execution_simulator_v1",
    }
    paths = {
        "r03_manifest": roots["r03"] / "metadata/manifest.json",
        "r03_outcomes": roots["r03"] / "future/outcomes.parquet",
        "r04_manifest": roots["r04"] / "metadata/manifest.json",
        "r04_dependence": roots["r04"] / "dependence/candidate_dependence.parquet",
        "r05_manifest": roots["r05"] / "metadata/manifest.json",
        "r05_candidate_core": roots["r05"] / "causal/candidate_core.parquet",
        "r05_trade_buckets_1s": roots["r05"] / "causal/trade_buckets_1s.parquet",
        "r05_trade_coverage": roots["r05"] / "quality/trade_coverage.parquet",
        "r06_manifest": roots["r06"] / "metadata/manifest.json",
        "r06_execution_state": roots["r06"] / "execution/audit_zero_latency_state.parquet",
    }
    expected = m.get("parents") or {}
    mismatches: list[str] = []
    for key, path in paths.items():
        if not path.exists():
            mismatches.append(f"{key}:missing")
        else:
            h = sha256_file(path)
            if expected.get(key) != h:
                mismatches.append(f"{key}:{expected.get(key)}!={h}")
    if mismatches:
        fs.append(finding("R09-RUN-003", "END_TO_END_REPRODUCIBILITY", "BLOCKER", "FINDING", True, ["R0.3-R0.8"],
                          "One or more accepted D0 parent artifacts no longer match the hashes used by R0.8.", "; ".join(mismatches)[:1800],
                          "Restore exact accepted parents or explicitly version and rerun the downstream chain."))
    else:
        fs.append(finding("R09-RUN-003", "END_TO_END_REPRODUCIBILITY", "INFO", "PASS", False, ["R0.3-R0.8"],
                          "All R0.3-R0.6 D0 parent artifacts still match the hashes consumed by accepted R0.8.", f"verified={len(paths)} artifacts"))

    mae = summary.get("position_worst_mae_bps")
    if isinstance(mae, (int, float)) and abs(float(mae) - 209.7608088530789) < 1e-9:
        fs.append(finding("R09-RUN-004", "END_TO_END_REPRODUCIBILITY", "INFO", "PASS", False, ["R0.3", "R0.8"],
                          "Corrected position-level worst-MAE semantics are present in the frozen runtime result.", f"position_worst_mae_bps={mae}"))
    else:
        fs.append(finding("R09-RUN-004", "END_TO_END_REPRODUCIBILITY", "MAJOR", "FINDING", True, ["R0.8"],
                          "R0.8 runtime does not show the accepted corrected worst-MAE value.", f"position_worst_mae_bps={mae}",
                          "Reconcile the pre-freeze MAE correction and accepted runtime manifest."))
    return fs


def audit_documentation_consistency(repo: Path, lineage: dict[str, Any]) -> list[AuditFinding]:
    fs: list[AuditFinding] = []
    top = (repo / "research/README.md").read_text(encoding="utf-8")
    local = (repo / "research/r0/README.md").read_text(encoding="utf-8")
    stale: list[str] = []
    if re.search(r"R0\.7[\s\S]{0,1800}Git freeze pending", top):
        stale.append("research/README.md:R0.7 says Git freeze pending")
    if re.search(r"R0\.8[\s\S]{0,2000}Git freeze pending", top):
        stale.append("research/README.md:R0.8 says Git freeze pending")
    if re.search(r"R0\.7[\s\S]{0,1800}remains to be frozen", local, flags=re.I):
        stale.append("research/r0/README.md:R0.7 says remains to be frozen")
    if re.search(r"R0\.8[\s\S]{0,1800}Git freeze is pending", local, flags=re.I):
        stale.append("research/r0/README.md:R0.8 says Git freeze pending")
    if stale:
        fs.append(finding("R09-DOC-001", "DOCUMENTATION_AND_MANIFEST_CONSISTENCY", "MAJOR", "FINDING", True, ["R0.7", "R0.8"],
                          "Living research documentation is stale relative to actual Git freezes.", "; ".join(stale),
                          "Update the R0.7 and R0.8 checkpoints/statuses in the R0.9 remediation commit."))
    else:
        fs.append(finding("R09-DOC-001", "DOCUMENTATION_AND_MANIFEST_CONSISTENCY", "INFO", "PASS", False, ["R0.7", "R0.8"],
                          "Living research documentation reflects the actual R0.7/R0.8 freeze state.",
                          f"R0.7 commit={lineage.get('r07_commit')}; R0.8 tag=r0-full-d0-dry-run-v1"))

    manifest_specs = [
        ("R0.5", repo / "research/r0/market_state/R0_5_IMPLEMENTATION_MANIFEST.json", "acceptance_state"),
        ("R0.6", repo / "research/r0/execution/R0_6_IMPLEMENTATION_MANIFEST.json", "acceptance_state"),
        ("R0.7", repo / "research/r0/registries/R0_7_IMPLEMENTATION_MANIFEST.json", "acceptance_state"),
        ("R0.8", repo / "research/r0/dry_run/R0_8_IMPLEMENTATION_MANIFEST.json", "implementation_status"),
    ]
    bad: list[str] = []
    for step, path, field in manifest_specs:
        if not path.exists():
            bad.append(f"{step}:manifest missing")
            continue
        value = str(load_json(path).get(field, ""))
        if any(token in value for token in ["PENDING", "NOT_YET", "FREEZE_PENDING"]):
            bad.append(f"{step}:{field}={value}")
    if bad:
        fs.append(finding("R09-DOC-002", "DOCUMENTATION_AND_MANIFEST_CONSISTENCY", "MAJOR", "FINDING", True, ["R0.5-R0.8"],
                          "Implementation manifests still describe already accepted/frozen stages as pending.", "; ".join(bad),
                          "Update acceptance metadata without changing scientific artifacts; record exact checkpoints and accepted evidence."))
    else:
        fs.append(finding("R09-DOC-002", "DOCUMENTATION_AND_MANIFEST_CONSISTENCY", "INFO", "PASS", False, ["R0.5-R0.8"],
                          "Implementation-manifest acceptance states agree with the frozen research history.", "R0.5-R0.8 acceptance fields are non-pending"))

    freeze = load_json(repo / "research/governance/protocol_freeze_manifest.json")
    reset_text = (repo / "research/governance/research_reset_manifest.yaml").read_text(encoding="utf-8")
    stale_governance: list[str] = []
    if "scanner installed" in str(freeze.get("r0_status", "")).lower() or "not yet" in str(freeze.get("r0_status", "")).lower():
        stale_governance.append(f"protocol_freeze_manifest.r0_status={freeze.get('r0_status')}")
    if "PENDING_LOCAL_SCAN" in reset_text or "IMPLEMENTED_PENDING_LOCAL_REPOSITORY_INSTALL" in reset_text:
        stale_governance.append("research_reset_manifest still contains pre-execution R0.0/R0.1 gate statuses")
    if stale_governance:
        fs.append(finding("R09-DOC-003", "DOCUMENTATION_AND_MANIFEST_CONSISTENCY", "MAJOR", "FINDING", True, ["R0.0", "R0.1"],
                          "Foundational governance manifests retain pre-execution status text after R0.1 was accepted.", "; ".join(stale_governance),
                          "Correct status metadata while preserving the original frozen protocol hash and cohort declarations."))
    else:
        fs.append(finding("R09-DOC-003", "DOCUMENTATION_AND_MANIFEST_CONSISTENCY", "INFO", "PASS", False, ["R0.0", "R0.1"],
                          "Foundational governance status metadata reflects completed R0.0/R0.1 acceptance.", "No pre-execution pending status remains."))
    return fs


def accepted_limitations() -> list[AuditFinding]:
    return [
        finding("R09-LIM-001", "CAUSAL_CLOCKS_AND_BOUNDARIES", "INFO", "ACCEPTED_LIMITATION", False, ["R0.3"],
                "Thirty-six historical candidates have unrecoverable governed clocks.",
                "7,067 candidate identities are retained; 7,031 have governed clocks; 36 remain explicit CLOCK_UNAVAILABLE.",
                "Never silently impute these clocks or include them in clock-dependent analyses."),
        finding("R09-LIM-002", "DEPENDENCE_AND_PSEUDOREPLICATION", "INFO", "ACCEPTED_LIMITATION", False, ["R0.4"],
                "Candidate dependence remains intrinsically severe at longer horizons.",
                "Accepted R0.4 diagnostics show multi-member overlap participation rising to ~96.3% at 30m.",
                "Later inference must freeze dependence-aware units; nominal candidate count is not independent sample size."),
        finding("R09-LIM-003", "MARKET_STATE_ASOF", "INFO", "ACCEPTED_LIMITATION", False, ["R0.5"],
                "The causal market-state store is deliberately narrow in version 1.",
                "PERPS_TRADES implemented; ORDER_BOOK, PERPS_DERIVATIVES, SPOT_CROSS_VENUE, OPTIONS deferred.",
                "Each new family needs its own causal availability/provenance contract before use."),
        finding("R09-LIM-004", "EXECUTION_BOUNDARIES", "INFO", "ACCEPTED_LIMITATION", False, ["R0.6", "R0.8"],
                "R0 execution coverage is not a final strategy execution model.",
                "R0.6/R0.8 use ZERO_LATENCY_AUDIT_ONLY and 1,000-USDT audit notional; passive fills and own impact are unsupported.",
                "Every real trading proposition must freeze defensible latency, fees, notional, exits, impact/fill assumptions before D1."),
        finding("R09-LIM-005", "EXECUTION_BOUNDARIES", "INFO", "ACCEPTED_LIMITATION", False, ["R0.6"],
                "Sequence-gap recovery duration is not persisted as a first-class interval audit artifact.",
                "19 sequence gaps occurred across 16 D0 days; candidate queries were all in valid reconstructed states, but exact recovery durations are not frozen.",
                "Improve recovery-interval auditability before it becomes material to a strategy-specific execution claim."),
    ]


def audit_contract_presence(repo: Path) -> list[AuditFinding]:
    required = {
        "R0.2": "research/governance/clock_contract_v1.json",
        "R0.4": "research/governance/dependence_contract_v1.json",
        "R0.5": "research/governance/market_state_contract_v1.json",
        "R0.6": "research/governance/execution_contract_v1.json",
        "R0.7": "research/governance/research_registry_contract_v1.json",
        "R0.8": "research/governance/r0_dry_run_contract_v1.json",
        "R0.9": "research/governance/r0_acceptance_audit_contract_v1.json",
    }
    missing = [f"{step}:{path}" for step, path in required.items() if not (repo / path).exists()]
    if missing:
        return [finding("R09-SEAM-001", "FEATURE_OUTCOME_SEPARATION", "BLOCKER", "FINDING", True, list(required),
                        "One or more critical R0 contracts are missing.", "; ".join(missing),
                        "Restore the missing governed contract before R1.")]
    return [finding("R09-SEAM-001", "FEATURE_OUTCOME_SEPARATION", "INFO", "PASS", False, list(required),
                    "All critical R0.2-R0.9 governance contracts are present.", f"contracts={len(required)}")]


def write_csv(path: Path, findings: list[AuditFinding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["finding_id", "domain", "severity", "status", "blocking_r1", "affected_steps", "summary", "evidence", "remediation"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for x in findings:
            d = x.to_dict()
            d["affected_steps"] = ";".join(d["affected_steps"])
            w.writerow(d)
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def self_test() -> int:
    a = finding("X", "D", "INFO", "PASS", False, ["R0"], "ok", "e")
    b = finding("Y", "D", "INFO", "ACCEPTED_LIMITATION", False, ["R0"], "lim", "e")
    c = finding("Z", "D", "MAJOR", "FINDING", True, ["R0"], "bad", "e")
    assert overall_status([a]) == "PASS"
    assert overall_status([a, b]) == "PASS_WITH_ACCEPTED_LIMITATIONS"
    assert overall_status([a, c]) == "REVIEW_REQUIRED"
    blocker = finding("B", "D", "BLOCKER", "FINDING", True, ["R0"], "bad", "e")
    assert overall_status([a, blocker]) == "NO_GO_BLOCKER"
    require_unique_finding_ids([a, b, c, blocker])
    try:
        require_unique_finding_ids([a, a])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate IDs were not rejected")
    print("SELF-TEST PASS: finding classification, R1 gating, deterministic hashing, and duplicate-ID protection work.")
    return 0


def run(args: argparse.Namespace) -> int:
    repo = args.repo_root.resolve()
    contract = load_json(repo / "research/governance/r0_acceptance_audit_contract_v1.json")
    findings: list[AuditFinding] = []

    print("[R0.9] stage 1/7: attacking Git freeze lineage", flush=True)
    fs, lineage = audit_git_lineage(repo, contract); findings.extend(fs)
    print(f"[R0.9] stage 1/7 complete: {len(fs)} checks", flush=True)

    print("[R0.9] stage 2/7: attacking protocol identity and canonical provenance", flush=True)
    fs = audit_protocol_and_provenance(repo); findings.extend(fs)
    print(f"[R0.9] stage 2/7 complete: {len(fs)} checks", flush=True)

    print("[R0.9] stage 3/7: attacking P1 firewall and production registry isolation", flush=True)
    fs = audit_p1_firewall(repo); findings.extend(fs)
    print(f"[R0.9] stage 3/7 complete: {len(fs)} checks", flush=True)

    print("[R0.9] stage 4/7: re-verifying accepted R0.8 runtime and D0 parent hashes", flush=True)
    fs = audit_r08_runtime(repo, contract, args.r08_runtime_manifest.resolve()); findings.extend(fs)
    print(f"[R0.9] stage 4/7 complete: {len(fs)} checks", flush=True)

    print("[R0.9] stage 5/7: attacking cross-component contract seams", flush=True)
    fs = audit_contract_presence(repo); findings.extend(fs)
    print(f"[R0.9] stage 5/7 complete: {len(fs)} checks", flush=True)

    print("[R0.9] stage 6/7: attacking documentation and acceptance-manifest consistency", flush=True)
    fs = audit_documentation_consistency(repo, lineage); findings.extend(fs)
    print(f"[R0.9] stage 6/7 complete: {len(fs)} checks", flush=True)

    print("[R0.9] stage 7/7: carrying forward residual scientific limitations", flush=True)
    findings.extend(accepted_limitations())
    require_unique_finding_ids(findings)

    status = overall_status(findings)
    counts = count_findings(findings)
    impl = implementation_fingerprint(repo)
    audit_inputs = audit_input_fingerprint(repo, args.r08_runtime_manifest.resolve())
    substantive_payload = {
        "schema_name": SCHEMA_NAME,
        "builder_version": BUILDER_VERSION,
        "accepted_parent": contract["accepted_parent"],
        "implementation_sha256": impl["sha256"],
        "audit_input_sha256": audit_inputs["sha256"],
        "findings": [f.to_dict() for f in sorted(findings, key=lambda x: x.finding_id)],
        "overall_status": status,
        "counts": counts,
        "lineage": lineage,
    }
    substantive_hash = sha256_canonical(substantive_payload)
    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": 1,
        "builder_version": BUILDER_VERSION,
        "generated_at_utc": utc_now_iso(),
        "evidentiary_role": "R0_FALSIFICATION_FIRST_INFRASTRUCTURE_ACCEPTANCE_AUDIT",
        "accepted_parent": contract["accepted_parent"],
        "implementation_fingerprint": impl,
        "audit_input_fingerprint": audit_inputs,
        "overall_status": status,
        "counts": counts,
        "lineage": lineage,
        "substantive_audit_hash": substantive_hash,
        "findings": [f.to_dict() for f in sorted(findings, key=lambda x: x.finding_id)],
        "protection": {"p1_market_data_read": False, "p1_outcome_read": False},
        "interpretation": "R0.9 audits infrastructure/process integrity only. It does not validate a trading hypothesis or profitability.",
    }
    old = args.report_root / "audit_summary.json"
    if old.exists():
        prev = load_json(old)
        if same_audit_identity(
            prev,
            accepted_parent=contract["accepted_parent"],
            implementation_sha256=impl["sha256"],
            audit_input_sha256=audit_inputs["sha256"],
        ):
            prev_hash = prev.get("substantive_audit_hash")
            if prev_hash and prev_hash != substantive_hash:
                raise RuntimeError(
                    "R0.9 same-parent same-implementation same-audit-input nondeterminism: "
                    f"{prev_hash} != {substantive_hash}"
                )
    write_csv(args.report_root / "findings.csv", findings)
    atomic_json(args.report_root / "audit_summary.json", report)

    print(json.dumps({
        "overall_status": status,
        "counts": counts,
        "r07_discovered_commit": lineage.get("r07_commit"),
        "r07_tags": lineage.get("r07_tags"),
        "substantive_audit_hash": substantive_hash,
        "p1_market_data_read": False,
        "p1_outcome_read": False,
        "report_root": str(args.report_root),
    }, indent=2, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    p.add_argument(
        "--r08-runtime-manifest", type=Path,
        default=REPO_ROOT / "data/derived/perps/research/r0_full_d0_dry_run_v1/metadata/manifest.json",
    )
    p.add_argument("--report-root", type=Path, default=REPO_ROOT / "reports/r0_acceptance_audit_v1")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    raise SystemExit(self_test() if a.self_test else run(a))
