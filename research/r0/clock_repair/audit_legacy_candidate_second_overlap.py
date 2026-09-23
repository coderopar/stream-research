#!/usr/bin/env python3
"""Quantify legacy candidate-second overlap without reading outcome values."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_core.clocks.contract import (  # noqa: E402
    D0_END_EXCLUSIVE_NS,
    D0_START_NS,
    audit_legacy_candidate_second_overlap,
    require_int64_ns,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--candidate-second-column", default="candidate_second")
    p.add_argument("--exact-anchor-column", default="anchor_ts_ns")
    p.add_argument("--legacy-outcome-offset-seconds", type=int, default=3)
    p.add_argument("--exact-feature-offset-seconds", type=int, default=3)
    return p.parse_args()


def _parse_plain_int(raw: str, label: str, row_number: int) -> int:
    if raw is None or raw == "" or raw.strip() != raw or not raw.lstrip("-").isdigit():
        raise ValueError(f"row {row_number}: {label} must be a base-10 integer")
    return int(raw)


def main() -> int:
    args = parse_args()
    if args.legacy_outcome_offset_seconds < 0 or args.exact_feature_offset_seconds < 0:
        raise SystemExit("offsets must be >= 0")

    overlaps: list[int] = []
    row_count = 0
    errors: list[str] = []

    with args.input.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {args.candidate_second_column, args.exact_anchor_column}
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise SystemExit(f"input missing columns: {', '.join(missing)}")

        for row_count, row in enumerate(reader, start=1):
            try:
                candidate_second = _parse_plain_int(
                    row[args.candidate_second_column], args.candidate_second_column, row_count
                )
                anchor = require_int64_ns(
                    _parse_plain_int(row[args.exact_anchor_column], args.exact_anchor_column, row_count),
                    args.exact_anchor_column,
                )
                if not (D0_START_NS <= anchor < D0_END_EXCLUSIVE_NS):
                    raise ValueError(
                        f"row {row_count}: exact anchor is outside D0; P1_PROTECTED access is forbidden"
                    )
                overlap = audit_legacy_candidate_second_overlap(
                    candidate_second=candidate_second,
                    exact_anchor_ts_ns=anchor,
                    legacy_outcome_offset_seconds=args.legacy_outcome_offset_seconds,
                    exact_feature_offset_seconds=args.exact_feature_offset_seconds,
                )
                overlaps.append(overlap)
            except Exception as exc:  # fail closed and report the row
                errors.append(str(exc))

    positive = [x for x in overlaps if x > 0]
    report = {
        "phase": "R0.2",
        "audit": "LEGACY_CANDIDATE_SECOND_OVERLAP",
        "input": str(args.input),
        "row_count": row_count,
        "audited_row_count": len(overlaps),
        "error_count": len(errors),
        "legacy_outcome_offset_seconds": args.legacy_outcome_offset_seconds,
        "exact_feature_offset_seconds": args.exact_feature_offset_seconds,
        "positive_overlap_count": len(positive),
        "positive_overlap_fraction": (len(positive) / len(overlaps)) if overlaps else None,
        "max_overlap_ns": max(overlaps) if overlaps else None,
        "mean_positive_overlap_ns": (sum(positive) / len(positive)) if positive else 0,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors[:100],
        "errors_truncated": max(0, len(errors) - 100),
        "interpretation": (
            "A positive overlap means the legacy candidate-second outcome interval starts before "
            "the exact-anchor feature interval has finished. This is a leakage diagnostic, not a repaired clock."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "errors"}, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
