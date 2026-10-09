from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.research_core.audit import AuditFinding, count_findings, overall_status, require_unique_finding_ids, sha256_canonical


def runner():
    p = REPO / "research/r0/audit/run_r0_acceptance_audit.py"
    spec = importlib.util.spec_from_file_location("r09_runner_test", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def f(fid="F", severity="INFO", status="PASS", blocking=False):
    return AuditFinding(fid, "D", severity, status, blocking, ("R0",), "s", "e", "")


def test_overall_status_pass():
    assert overall_status([f()]) == "PASS"


def test_overall_status_pass_with_accepted_limitations():
    assert overall_status([f(), f("L", status="ACCEPTED_LIMITATION")]) == "PASS_WITH_ACCEPTED_LIMITATIONS"


def test_overall_status_review_required_for_blocking_major():
    assert overall_status([f("M", severity="MAJOR", status="FINDING", blocking=True)]) == "REVIEW_REQUIRED"


def test_overall_status_no_go_for_blocker():
    assert overall_status([f("B", severity="BLOCKER", status="FINDING", blocking=True)]) == "NO_GO_BLOCKER"


def test_pass_cannot_block_r1():
    with pytest.raises(ValueError):
        f("X", status="PASS", blocking=True)


def test_accepted_limitation_cannot_block_r1():
    with pytest.raises(ValueError):
        f("X", status="ACCEPTED_LIMITATION", blocking=True)


def test_duplicate_finding_ids_rejected():
    with pytest.raises(ValueError):
        require_unique_finding_ids([f("X"), f("X")])


def test_counts_are_explicit():
    rows = [f(), f("L", status="ACCEPTED_LIMITATION"), f("M", severity="MAJOR", status="FINDING", blocking=True)]
    c = count_findings(rows)
    assert c["total"] == 3 and c["passes"] == 1 and c["accepted_limitations"] == 1 and c["majors"] == 1 and c["blocking_r1"] == 1


def test_canonical_hash_is_order_stable_for_mapping_keys():
    assert sha256_canonical({"a": 1, "b": 2}) == sha256_canonical({"b": 2, "a": 1})


def test_contract_freezes_r08_parent_and_prohibits_p1_reads():
    d = json.loads((REPO / "research/governance/r0_acceptance_audit_contract_v1.json").read_text(encoding="utf-8"))
    assert d["accepted_parent"]["commit"] == "53eac195e83b5fe08397ca50ffc1aadbd201c9a1"
    assert d["accepted_parent"]["tag"] == "r0-full-d0-dry-run-v1"
    assert d["p1_market_data_read"] is False and d["p1_outcome_read"] is False
    assert d["no_new_hypothesis_testing"] is True


def test_contract_has_all_required_adversarial_domains():
    d = json.loads((REPO / "research/governance/r0_acceptance_audit_contract_v1.json").read_text(encoding="utf-8"))
    domains = d["required_domains"]
    assert len(domains) == len(set(domains))
    for x in ["P1_FIREWALL", "CAUSAL_CLOCKS_AND_BOUNDARIES", "DEPENDENCE_AND_PSEUDOREPLICATION", "EXECUTION_BOUNDARIES", "DOCUMENTATION_AND_MANIFEST_CONSISTENCY"]:
        assert x in domains


def test_clean_p1_fixture_passes(tmp_path: Path):
    m = runner()
    g = tmp_path / "research/governance"; g.mkdir(parents=True)
    (g / "protected_data_register.yaml").write_text("pool_id: P1_PROTECTED\nstatus: SEALED\noutcome_exposure_status: UNSEEN\n", encoding="utf-8")
    (g / "outcome_access_register.csv").write_text("event_id,access_type,outcome_scope,status\n1,DECLARATION,NONE,CLEAN\n", encoding="utf-8")
    regs = g / "registries"; regs.mkdir()
    for name in m.PRODUCTION_REGISTRY_FILES:
        (regs / name).write_text("", encoding="utf-8")
    rows = m.audit_p1_firewall(tmp_path)
    by = {x.finding_id: x for x in rows}
    assert by["R09-P1-001"].status == "PASS"
    assert by["R09-P1-002"].status == "PASS"
    assert by["R09-P1-003"].status == "PASS"
    assert by["R09-P1-004"].status == "ACCEPTED_LIMITATION"


def test_p1_outcome_exposure_is_blocker(tmp_path: Path):
    m = runner()
    g = tmp_path / "research/governance"; g.mkdir(parents=True)
    (g / "protected_data_register.yaml").write_text("status: SEALED\noutcome_exposure_status: UNSEEN\n", encoding="utf-8")
    (g / "outcome_access_register.csv").write_text("event_id,access_type,outcome_scope,status\n1,OUTCOME_READ,RETURN,CONTAMINATED\n", encoding="utf-8")
    regs = g / "registries"; regs.mkdir()
    for name in m.PRODUCTION_REGISTRY_FILES:
        (regs / name).write_text("", encoding="utf-8")
    rows = m.audit_p1_firewall(tmp_path)
    item = next(x for x in rows if x.finding_id == "R09-P1-002")
    assert item.severity == "BLOCKER" and item.blocking_r1


def test_nonempty_production_registry_blocks_pre_r1_boundary(tmp_path: Path):
    m = runner()
    g = tmp_path / "research/governance"; g.mkdir(parents=True)
    (g / "protected_data_register.yaml").write_text("status: SEALED\noutcome_exposure_status: UNSEEN\n", encoding="utf-8")
    (g / "outcome_access_register.csv").write_text("event_id,access_type,outcome_scope,status\n1,DECLARATION,NONE,CLEAN\n", encoding="utf-8")
    regs = g / "registries"; regs.mkdir()
    for name in m.PRODUCTION_REGISTRY_FILES:
        (regs / name).write_text("{}\n" if name == "hypothesis_versions.jsonl" else "", encoding="utf-8")
    item = next(x for x in m.audit_p1_firewall(tmp_path) if x.finding_id == "R09-P1-003")
    assert item.status == "FINDING" and item.blocking_r1


def test_protocol_hash_fixture_passes(tmp_path: Path):
    m = runner()
    g = tmp_path / "research/governance"; pdir = tmp_path / "research/protocol/v1.2"
    g.mkdir(parents=True); pdir.mkdir(parents=True)
    p = pdir / "x.docx"; p.write_bytes(b"abc")
    h = hashlib.sha256(b"abc").hexdigest()
    (g / "protocol_freeze_manifest.json").write_text(json.dumps({"status":"FROZEN","protocol_path":"research/protocol/v1.2/x.docx","sha256":h}), encoding="utf-8")
    ds = g / "datasets"; ds.mkdir()
    (ds / "R0_1_INVENTORY_SUMMARY_v1.json").write_text(json.dumps({"status":"PASS","file_count":22369,"p1_declared_start_date_utc":"2026-07-19","undated_file_count":0,"unknown_undated_market_event_file_count":0,"file_index_sha256":"x"}), encoding="utf-8")
    rows = m.audit_protocol_and_provenance(tmp_path)
    assert all(x.status == "PASS" for x in rows)


def test_protocol_hash_mismatch_is_blocker(tmp_path: Path):
    m = runner()
    g = tmp_path / "research/governance"; pdir = tmp_path / "research/protocol/v1.2"
    g.mkdir(parents=True); pdir.mkdir(parents=True)
    p = pdir / "x.docx"; p.write_bytes(b"abc")
    (g / "protocol_freeze_manifest.json").write_text(json.dumps({"status":"FROZEN","protocol_path":"research/protocol/v1.2/x.docx","sha256":"0"*64}), encoding="utf-8")
    ds = g / "datasets"; ds.mkdir()
    (ds / "R0_1_INVENTORY_SUMMARY_v1.json").write_text("{}", encoding="utf-8")
    item = m.audit_protocol_and_provenance(tmp_path)[0]
    assert item.severity == "BLOCKER" and item.blocking_r1


def test_stale_documentation_is_a_blocking_major(tmp_path: Path):
    m = runner()
    (tmp_path / "research/r0/market_state").mkdir(parents=True)
    (tmp_path / "research/r0/execution").mkdir(parents=True)
    (tmp_path / "research/r0/registries").mkdir(parents=True)
    (tmp_path / "research/r0/dry_run").mkdir(parents=True)
    (tmp_path / "research/governance").mkdir(parents=True)
    (tmp_path / "research/README.md").write_text("## R0.7\nStatus: Git freeze pending\n## R0.8\nStatus: Git freeze pending\n", encoding="utf-8")
    (tmp_path / "research/r0/README.md").write_text("R0.7 remains to be frozen\nR0.8 Git freeze is pending\n", encoding="utf-8")
    specs = [
        ("market_state/R0_5_IMPLEMENTATION_MANIFEST.json", "acceptance_state"),
        ("execution/R0_6_IMPLEMENTATION_MANIFEST.json", "acceptance_state"),
        ("registries/R0_7_IMPLEMENTATION_MANIFEST.json", "acceptance_state"),
        ("dry_run/R0_8_IMPLEMENTATION_MANIFEST.json", "implementation_status"),
    ]
    for rel, field in specs:
        (tmp_path / "research/r0" / rel).write_text(json.dumps({field:"FREEZE_PENDING"}), encoding="utf-8")
    (tmp_path / "research/governance/protocol_freeze_manifest.json").write_text(json.dumps({"r0_status":"scanner installed but not yet run"}), encoding="utf-8")
    (tmp_path / "research/governance/research_reset_manifest.yaml").write_text("status: PENDING_LOCAL_SCAN\n", encoding="utf-8")
    rows = m.audit_documentation_consistency(tmp_path, {"r07_commit":"abc"})
    assert len(rows) == 3
    assert all(x.status == "FINDING" and x.severity == "MAJOR" and x.blocking_r1 for x in rows)


def test_contract_presence_fixture_fails_closed(tmp_path: Path):
    m = runner()
    rows = m.audit_contract_presence(tmp_path)
    assert rows[0].severity == "BLOCKER"


def test_repository_contract_presence_passes():
    rows = runner().audit_contract_presence(REPO)
    assert rows[0].status == "PASS"


def test_accepted_limitations_are_nonblocking_and_unique():
    rows = runner().accepted_limitations()
    require_unique_finding_ids(rows)
    assert len(rows) >= 5
    assert all(x.status == "ACCEPTED_LIMITATION" and not x.blocking_r1 for x in rows)


def test_runner_source_does_not_open_p1_market_paths():
    text = (REPO / "research/r0/audit/run_r0_acceptance_audit.py").read_text(encoding="utf-8")
    assert "data/canonical/market_events_v1" not in text
    assert "P1_PROTECTED" not in text


def test_r09_readme_is_part_of_living_research_record():
    top = (REPO / "research/README.md").read_text(encoding="utf-8")
    local = (REPO / "research/r0/README.md").read_text(encoding="utf-8")
    assert "R0.9" in top and "Adversarial R0" in top
    assert "R0.9" in local and "audit" in local.lower()


def test_r09_manifest_declares_no_hypothesis_testing_or_p1_reads():
    d = json.loads((REPO / "research/r0/audit/R0_9_IMPLEMENTATION_MANIFEST.json").read_text(encoding="utf-8"))
    assert d["new_hypothesis_testing"] is False
    assert d["p1_market_data_read"] is False
    assert d["p1_outcome_read"] is False



def test_audit_input_fingerprint_changes_when_governance_input_changes(tmp_path: Path, monkeypatch):
    m = runner()
    # The fingerprint must tolerate a partial fixture because missing audited files
    # are themselves meaningful inputs to a fail-closed audit.
    (tmp_path / "research").mkdir()
    readme = tmp_path / "research/README.md"
    readme.write_text("before", encoding="utf-8")
    runtime = tmp_path / "r08_manifest.json"
    runtime.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(m, "git", lambda repo, *args, **kwargs: "HEAD-A")
    monkeypatch.setattr(m, "git_tag_commit", lambda repo, tag: f"commit-{tag}")

    a = m.audit_input_fingerprint(tmp_path, runtime)
    readme.write_text("after", encoding="utf-8")
    b = m.audit_input_fingerprint(tmp_path, runtime)
    assert a["sha256"] != b["sha256"]


def test_idempotence_identity_requires_same_audited_repository_state():
    m = runner()
    parent = {"commit": "P", "tag": "T", "substantive_run_hash": "H"}
    previous = {
        "accepted_parent": parent,
        "implementation_fingerprint": {"sha256": "impl"},
        "audit_input_fingerprint": {"sha256": "inputs-a"},
    }
    assert m.same_audit_identity(
        previous, accepted_parent=parent, implementation_sha256="impl", audit_input_sha256="inputs-a"
    )
    assert not m.same_audit_identity(
        previous, accepted_parent=parent, implementation_sha256="impl", audit_input_sha256="inputs-b"
    )
    # Legacy R0.9 reports predate audit-input fingerprinting and therefore establish
    # no comparable idempotence baseline after a governed remediation.
    legacy = {
        "accepted_parent": parent,
        "implementation_fingerprint": {"sha256": "impl"},
    }
    assert not m.same_audit_identity(
        legacy, accepted_parent=parent, implementation_sha256="impl", audit_input_sha256="inputs-a"
    )

def test_self_test_passes():
    assert runner().self_test() == 0
