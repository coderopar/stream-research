from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

SEVERITY_ORDER = {"INFO": 0, "MINOR": 1, "MAJOR": 2, "BLOCKER": 3}
VALID_STATUS = {"PASS", "FINDING", "ACCEPTED_LIMITATION", "NOT_TESTABLE"}


@dataclass(frozen=True)
class AuditFinding:
    finding_id: str
    domain: str
    severity: str
    status: str
    blocking_r1: bool
    affected_steps: tuple[str, ...]
    summary: str
    evidence: str
    remediation: str = ""

    def __post_init__(self) -> None:
        if self.severity not in SEVERITY_ORDER:
            raise ValueError(f"invalid severity: {self.severity}")
        if self.status not in VALID_STATUS:
            raise ValueError(f"invalid status: {self.status}")
        if self.status == "PASS" and self.blocking_r1:
            raise ValueError("PASS finding cannot block R1")
        if self.status == "ACCEPTED_LIMITATION" and self.blocking_r1:
            raise ValueError("accepted limitation cannot block R1")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["affected_steps"] = list(self.affected_steps)
        return d


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_canonical(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def overall_status(findings: Iterable[AuditFinding]) -> str:
    findings = list(findings)
    if any(f.status == "FINDING" and f.blocking_r1 and f.severity == "BLOCKER" for f in findings):
        return "NO_GO_BLOCKER"
    if any(f.status == "FINDING" and f.blocking_r1 for f in findings):
        return "REVIEW_REQUIRED"
    if any(f.status == "NOT_TESTABLE" and f.blocking_r1 for f in findings):
        return "REVIEW_REQUIRED"
    if any(f.status == "ACCEPTED_LIMITATION" for f in findings):
        return "PASS_WITH_ACCEPTED_LIMITATIONS"
    return "PASS"


def count_findings(findings: Iterable[AuditFinding]) -> dict[str, int]:
    rows = list(findings)
    return {
        "total": len(rows),
        "passes": sum(f.status == "PASS" for f in rows),
        "open_findings": sum(f.status == "FINDING" for f in rows),
        "accepted_limitations": sum(f.status == "ACCEPTED_LIMITATION" for f in rows),
        "not_testable": sum(f.status == "NOT_TESTABLE" for f in rows),
        "blocking_r1": sum(bool(f.blocking_r1) for f in rows if f.status in {"FINDING", "NOT_TESTABLE"}),
        "blockers": sum(f.status == "FINDING" and f.severity == "BLOCKER" for f in rows),
        "majors": sum(f.status == "FINDING" and f.severity == "MAJOR" for f in rows),
        "minors": sum(f.status == "FINDING" and f.severity == "MINOR" for f in rows),
    }


def require_unique_finding_ids(findings: Iterable[AuditFinding]) -> None:
    ids = [f.finding_id for f in findings]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate R0.9 finding IDs")
