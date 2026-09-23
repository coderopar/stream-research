#!/usr/bin/env python3
"""Outcome-blind R0.2 causal-clock audit for D0 clock rows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_core.clocks.contract import (  # noqa: E402
    ClockContractError,
    REQUIRED_CLOCK_FIELDS,
    assert_clock_contract,
    assert_d0_only,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="CSV containing governed *_ts_ns clock columns")
    p.add_argument("--report", required=True, type=Path, help="JSON audit report path")
    p.add_argument(
        "--max-outcome-horizon-seconds",
        type=int,
        default=0,
        help="largest outcome horizon that later consumers will request (e.g. 1800 for 30m)",
    )
    return p.parse_args()


def _parse_row(row: dict[str, str], row_number: int) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for field in REQUIRED_CLOCK_FIELDS:
        raw = row.get(field)
        if raw is None or raw == "":
            raise ValueError(f"row {row_number}: missing {field}")
        # Base-10 integer parsing only. Decimal/scientific notation is refused.
        if raw.strip() != raw or not raw.lstrip("-").isdigit():
            raise ValueError(f"row {row_number}: {field} must be a base-10 integer nanosecond timestamp")
        parsed[field] = int(raw)
    return parsed


def main() -> int:
    args = parse_args()
    if args.max_outcome_horizon_seconds < 0:
        raise SystemExit("--max-outcome-horizon-seconds must be >= 0")

    failures: list[dict[str, object]] = []
    codes: Counter[str] = Counter()
    row_count = 0

    with args.input.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [field for field in REQUIRED_CLOCK_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"input is missing governed clock columns: {', '.join(missing)}")

        for row_count, row in enumerate(reader, start=1):
            try:
                parsed = _parse_row(row, row_count)
                assert_clock_contract(parsed)
                assert_d0_only(
                    parsed,
                    max_outcome_horizon_seconds=args.max_outcome_horizon_seconds,
                )
            except ClockContractError as exc:
                for violation in exc.violations:
                    codes[violation.code] += 1
                failures.append(
                    {
                        "row_number": row_count,
                        "violations": [
                            {"code": v.code, "message": v.message, "fields": list(v.fields)}
                            for v in exc.violations
                        ],
                    }
                )
            except ValueError as exc:
                codes["PARSE_ERROR"] += 1
                failures.append(
                    {
                        "row_number": row_count,
                        "violations": [{"code": "PARSE_ERROR", "message": str(exc), "fields": []}],
                    }
                )

    report = {
        "phase": "R0.2",
        "audit": "D0_CAUSAL_CLOCK_CONTRACT",
        "input": str(args.input),
        "max_outcome_horizon_seconds": args.max_outcome_horizon_seconds,
        "row_count": row_count,
        "failed_row_count": len(failures),
        "violation_counts": dict(sorted(codes.items())),
        "status": "PASS" if not failures else "FAIL",
        "failures": failures[:100],
        "failure_rows_truncated": max(0, len(failures) - 100),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"failures"}}, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
