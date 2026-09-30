from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_core.registry import (  # noqa: E402
    RegistryValidationError,
    append_jsonl,
    assert_append_only_prefix,
    assert_substantive_equal,
    authorize_protected_access,
    read_jsonl,
    validate_access_event_history,
    validate_confirmation_batch,
    validate_decision_log,
    validate_evidence_budget,
    validate_event_history,
    validate_protected_cohort,
    validate_genealogy,
    validate_hypothesis_version,
    with_freeze_hash,
)

REGISTRY_DIR = REPO_ROOT / "research" / "governance" / "registries"
CONTRACT_PATH = REPO_ROOT / "research" / "governance" / "research_registry_contract_v1.json"
PROTECTED_REGISTER = REPO_ROOT / "research" / "governance" / "protected_data_register.yaml"
OUTCOME_ACCESS_REGISTER = REPO_ROOT / "research" / "governance" / "outcome_access_register.csv"
PROTOCOL_PATH = REPO_ROOT / "research" / "protocol" / "v1.2" / "BTCUSDT_Scientific_Trading_Research_Protocol_v1.2.docx"


def _load_contract() -> dict:
    obj = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if obj.get("protocol_version") != "1.2":
        raise RegistryValidationError("R0.7 contract is not bound to Protocol v1.2")
    firewall = obj.get("outcome_firewall", {})
    if firewall.get("future_outcome_values_read") is not False or firewall.get("p1_read") is not False:
        raise RegistryValidationError("R0.7 outcome/P1 firewall is not closed")
    if obj.get("registry_storage", {}).get("append_only") is not True:
        raise RegistryValidationError("R0.7 registries must be append-only")
    return obj


def _validate_registry_files(contract: dict) -> dict[str, int]:
    mapping = contract["registry_storage"]["registry_files"]
    loaded: dict[str, list[dict]] = {}
    for name, rel in mapping.items():
        path = REPO_ROOT / rel
        if not path.exists():
            raise RegistryValidationError(f"missing governed registry: {rel}")
        loaded[name] = read_jsonl(path)

    for record in loaded["hypothesis_versions"]:
        validate_hypothesis_version(record)
    validate_genealogy(loaded["hypothesis_versions"])
    validate_event_history(loaded["hypothesis_events"])
    for record in loaded["evidence_budgets"]:
        validate_evidence_budget(record)
    for record in loaded["protected_cohorts"]:
        validate_protected_cohort(record)
    for record in loaded["confirmation_batches"]:
        validate_confirmation_batch(record)
    validate_access_event_history(loaded["access_events"])
    validate_decision_log(loaded["decision_log"])

    return {name: len(rows) for name, rows in loaded.items()}


def _validate_current_protection_state() -> None:
    text = PROTECTED_REGISTER.read_text(encoding="utf-8")
    required = (
        "pool_id: P1_PROTECTED",
        "status: SEALED",
        "outcome_exposure_status: UNSEEN",
        'declared_start_date_utc: "2026-07-19"',
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise RegistryValidationError(f"P1 protected register no longer matches frozen R0 state: {missing}")

    with OUTCOME_ACCESS_REGISTER.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    p1 = [r for r in rows if r.get("pool_id") == "P1_PROTECTED"]
    if not p1:
        raise RegistryValidationError("P1 declaration is missing from outcome_access_register.csv")
    exposed = [r for r in p1 if (r.get("outcome_scope") or "").upper() not in {"", "NONE"}]
    if exposed:
        raise RegistryValidationError("P1 protected outcome exposure is recorded; R0.7 cannot claim a clean sealed state")


def validate_installation() -> dict:
    contract = _load_contract()
    if not PROTOCOL_PATH.exists():
        raise RegistryValidationError("Protocol v1.2 artifact is missing")
    counts = _validate_registry_files(contract)
    _validate_current_protection_state()
    return {
        "status": "PASS",
        "protocol_version": "1.2",
        "p1_status": "SEALED_OUTCOME_UNSEEN",
        "future_outcomes_read": False,
        "p1_read": False,
        "registry_counts": counts,
    }


def self_test() -> None:
    # Access firewall: discovery cannot inspect D1; registered feasibility can inspect causal gate fields.
    denied = authorize_protected_access(role="DISCOVERY_AGENT", zone="D1", content_class="INTEGRITY_METADATA")
    assert not denied.allowed
    permitted = authorize_protected_access(
        role="FEASIBILITY_AGENT", zone="D1", content_class="CAUSAL_FIELDS_FOR_REGISTERED_GATE", registered_gate=True
    )
    assert permitted.allowed and permitted.contamination_after == "CLEAN"
    detailed = authorize_protected_access(
        role="FEASIBILITY_AGENT", zone="D1", content_class="DETAILED_FEASIBILITY_DIAGNOSTICS", registered_gate=True
    )
    assert detailed.allowed and detailed.contamination_after == "DESIGN_CONTAMINATED"

    # Append-only and substantive-idempotence primitives.
    assert_append_only_prefix(b'{"a":1}\n', b'{"a":1}\n{"b":2}\n')
    assert_substantive_equal(
        {"rows": 10, "generated_at_utc": "a"},
        {"rows": 10, "generated_at_utc": "b"},
        volatile_fields={"generated_at_utc"},
    )

    # Canonical append rejects duplicate identity when requested.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.jsonl"
        row = with_freeze_hash({
            "hypothesis_id": "HYP-SELFTEST",
            "version": 1,
            "conceptual_family": "FAM-SELFTEST",
            "creation_source": "R0_7_SELF_TEST",
            "created_at_utc": "2026-09-29T00:00:00Z",
            "scientific_proposition": "self-test only",
            "relationship_type": "MARGINAL",
            "parent_hypothesis_ids": [],
            "derived_from_results": [],
            "datasets_that_influenced_design": ["SYNTHETIC_SELF_TEST"],
            "material_spec": {},
        })
        validate_hypothesis_version(row)
        append_jsonl(path, row, unique_key=lambda r: (r["hypothesis_id"], r["version"]))
        try:
            append_jsonl(path, row, unique_key=lambda r: (r["hypothesis_id"], r["version"]))
        except RegistryValidationError:
            pass
        else:
            raise AssertionError("duplicate hypothesis version was not rejected")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate R0.7 registries and research guardrails without opening market/outcome data.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    summary = validate_installation()
    if args.self_test:
        self_test()
        summary["self_test"] = "PASS"
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
