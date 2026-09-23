#!/usr/bin/env python
"""
Audit the canonical market-event store produced by 01_extract_all_market_events.py.

The default audit is designed to be thorough without re-reading every market
payload row. It reconciles the extractor SQLite state with the physical output,
opens every Parquet footer, validates schemas and partition dates, and checks
row counts at source-file, family, and global levels.

An optional deep provenance pass reads only four compact provenance columns,
one source WAL at a time, to prove exact source-record uniqueness and account
for any intentionally skipped or malformed source lines.

Dependencies
------------
    pip install pyarrow

Typical usage
-------------
    python scripts/02_audit_canonical_market_events.py \
      --canonical-root "./data/canonical/market_events_v1"

The input root is normally inferred from the latest extraction run. It can be
provided explicitly when the raw bronze directory has moved:

    python scripts/02_audit_canonical_market_events.py \
      --canonical-root "./data/canonical/market_events_v1" \
      --input-root "/d/stream-data/bronze"

For an exact overnight provenance audit:

    python scripts/02_audit_canonical_market_events.py \
      --canonical-root "./data/canonical/market_events_v1" \
      --deep-provenance
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import sys
import traceback
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "Missing dependency 'pyarrow'. Activate the project virtual environment "
        "and run: pip install pyarrow"
    ) from exc


AUDITOR_VERSION = "1.0.1"
EXPECTED_CANONICAL_SCHEMA_VERSION = 1
DEFAULT_CANONICAL_ROOT = "./data/canonical/market_events_v1"
DEFAULT_REPORT_ROOT = "./reports/canonical_audit"
DEFAULT_SOURCE_GLOB = "**/*.ndjson*.zst"
DATE_PARTITION_RE = re.compile(r"^date=(\d{4}-\d{2}-\d{2})$")
PART_FILENAME_RE = re.compile(r"^part-([0-9a-fA-F]{24})\.parquet$")
COMMON_REQUIRED_COLUMNS = {
    "canonical_schema_version",
    "source_schema",
    "market",
    "event_type",
    "symbol",
    "ts_recv_ns",
    "receive_second",
    "mono_ns",
    "event_id",
    "sequence",
    "source_file_id",
    "source_relative_path",
    "source_record_number",
    "record_extra_json",
    "payload_extra_json",
}
SEVERITY_ORDER = {"INFO": 0, "WARN": 1, "FAIL": 2}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Issue:
    severity: str
    code: str
    scope: str
    message: str
    source_relative_path: str | None = None
    output_path: str | None = None
    family: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "scope": self.scope,
            "message": self.message,
            "source_relative_path": self.source_relative_path,
            "output_path": self.output_path,
            "family": self.family,
        }


@dataclass
class Invariant:
    name: str
    status: str
    issue_count: int
    details: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "issue_count": self.issue_count,
            "details": self.details,
        }


@dataclass
class ShardAudit:
    relative_path: str
    market_root: str | None
    family_name: str | None
    family_key: str | None
    partition_date: str | None
    filename_source_file_id: str | None
    rows: int = 0
    row_groups: int = 0
    size_bytes: int = 0
    schema_signature: str | None = None
    schema_name: str | None = None
    extractor_version: str | None = None
    canonical_schema_metadata: str | None = None
    ts_min_ns: int | None = None
    ts_max_ns: int | None = None
    source_record_min: int | None = None
    source_record_max: int | None = None
    source_file_id_min: str | None = None
    source_file_id_max: str | None = None
    source_path_min: str | None = None
    source_path_max: str | None = None
    canonical_version_min: int | None = None
    canonical_version_max: int | None = None
    null_counts: dict[str, int | None] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "market_root": self.market_root,
            "family_name": self.family_name,
            "family_key": self.family_key,
            "partition_date": self.partition_date,
            "filename_source_file_id": self.filename_source_file_id,
            "rows": self.rows,
            "row_groups": self.row_groups,
            "size_bytes": self.size_bytes,
            "schema_signature": self.schema_signature,
            "schema_name": self.schema_name,
            "extractor_version": self.extractor_version,
            "canonical_schema_metadata": self.canonical_schema_metadata,
            "ts_min_ns": self.ts_min_ns,
            "ts_max_ns": self.ts_max_ns,
            "source_record_min": self.source_record_min,
            "source_record_max": self.source_record_max,
            "source_file_id_min": self.source_file_id_min,
            "source_file_id_max": self.source_file_id_max,
            "source_path_min": self.source_path_min,
            "source_path_max": self.source_path_max,
            "canonical_version_min": self.canonical_version_min,
            "canonical_version_max": self.canonical_version_max,
            "null_counts_json": compact_json(self.null_counts),
            "issue_count": len(self.issues),
        }


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def load_json_text(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def load_json_file(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
    os.replace(temporary, path)


def ns_to_utc_date(value: int | None) -> str | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc).date().isoformat()
    except (ValueError, OSError, OverflowError):
        return None


def bytes_to_gib(value: int) -> float:
    return value / (1024 ** 3)


def quick_fingerprint(path: Path, chunk_bytes: int = 1_048_576) -> str:
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode())
    with path.open("rb") as handle:
        first = handle.read(chunk_bytes)
        digest.update(first)
        if stat.st_size > chunk_bytes:
            handle.seek(max(0, stat.st_size - chunk_bytes))
            digest.update(handle.read(chunk_bytes))
    return digest.hexdigest()


def full_fingerprint(path: Path, chunk_bytes: int = 8 * 1_048_576) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_counter_sum(value: str | None) -> tuple[dict[str, int], int, bool]:
    parsed = load_json_text(value, {})
    if not isinstance(parsed, Mapping):
        return {}, 0, False
    clean: dict[str, int] = {}
    valid = True
    for key, raw in parsed.items():
        try:
            clean[str(key)] = int(raw)
        except (TypeError, ValueError):
            valid = False
    return clean, sum(clean.values()), valid


def canonical_arrow_type_text(value: Any) -> str:
    """Return a semantic Arrow type string stable across Parquet round-trips.

    PyArrow may describe the child of a list created in memory as ``item`` but
    reconstruct the same logical list from Parquet using the child name
    ``element``. Those child labels are representational details, not schema
    differences. Normalize them before manifest comparison and signatures.
    """
    text = str(value)
    text = re.sub(r"(?<=list<)(?:item|element):", "value:", text)
    text = re.sub(r"(?<=large_list<)(?:item|element):", "value:", text)
    return text


def schema_signature(schema: pa.Schema) -> str:
    payload = [
        {
            "name": field.name,
            "type": canonical_arrow_type_text(field.type),
            "nullable": field.nullable,
        }
        for field in schema
    ]
    return hashlib.sha256(compact_json(payload).encode("utf-8")).hexdigest()[:20]


def schema_description(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {
            "name": field.name,
            "type": canonical_arrow_type_text(field.type),
            "nullable": field.nullable,
        }
        for field in schema
    ]


def decode_metadata(metadata: Mapping[bytes, bytes] | None, key: str) -> str | None:
    if not metadata:
        return None
    raw = metadata.get(key.encode("utf-8"))
    if raw is None:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return repr(raw)


def parquet_column_index(parquet_file: pq.ParquetFile, column_name: str) -> int | None:
    parquet_schema = parquet_file.metadata.schema
    for index in range(len(parquet_schema)):
        column = parquet_schema.column(index)
        if column.path == column_name:
            return index
    return None


def parquet_scalar_stats(
    parquet_file: pq.ParquetFile,
    column_name: str,
) -> tuple[Any | None, Any | None, int | None, bool]:
    """Return global min, max, null count, and completeness from row-group stats."""
    index = parquet_column_index(parquet_file, column_name)
    if index is None:
        return None, None, None, False

    minimum: Any | None = None
    maximum: Any | None = None
    null_count = 0
    complete = True
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        column = parquet_file.metadata.row_group(row_group_index).column(index)
        stats = column.statistics
        if stats is None or not stats.has_min_max:
            complete = False
            continue
        current_min = stats.min
        current_max = stats.max
        if isinstance(current_min, bytes):
            current_min = current_min.decode("utf-8", errors="replace")
        if isinstance(current_max, bytes):
            current_max = current_max.decode("utf-8", errors="replace")
        minimum = current_min if minimum is None else min(minimum, current_min)
        maximum = current_max if maximum is None else max(maximum, current_max)
        if stats.null_count is None:
            complete = False
        else:
            null_count += int(stats.null_count)
    return minimum, maximum, (null_count if complete else None), complete


def status_from_issues(issues: Sequence[Issue]) -> str:
    if any(issue.severity == "FAIL" for issue in issues):
        return "FAIL"
    if any(issue.severity == "WARN" for issue in issues):
        return "WARN"
    return "PASS"


def issue_count(issues: Sequence[Issue], codes: set[str] | None = None) -> int:
    if codes is None:
        return len(issues)
    return sum(1 for issue in issues if issue.code in codes)


# ---------------------------------------------------------------------------
# State database and manifest
# ---------------------------------------------------------------------------


def connect_state_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    return connection


def require_state_tables(connection: sqlite3.Connection) -> None:
    required = {"metadata", "extraction_runs", "source_files", "extraction_attempts"}
    actual = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    missing = required - actual
    if missing:
        raise RuntimeError(f"State database is missing required tables: {sorted(missing)}")


def latest_extraction_context(connection: sqlite3.Connection) -> dict[str, Any]:
    row = connection.execute(
        """
        SELECT * FROM extraction_runs
        WHERE status IN ('COMPLETED', 'COMPLETED_WITH_FAILURES')
        ORDER BY COALESCE(completed_at_utc, started_at_utc) DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        row = connection.execute(
            "SELECT * FROM extraction_runs ORDER BY started_at_utc DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row is not None else {}


def expected_schemas_from_manifest(manifest: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    routes = manifest.get("known_routes", [])
    if isinstance(routes, list):
        for route in routes:
            if not isinstance(route, Mapping):
                continue
            output_market = route.get("output_market")
            family = route.get("family")
            schema = route.get("schema")
            if output_market and family and isinstance(schema, list):
                normalized_schema: list[dict[str, Any]] = []
                for raw_field in schema:
                    if not isinstance(raw_field, Mapping):
                        continue
                    field_row = dict(raw_field)
                    if "type" in field_row:
                        field_row["type"] = canonical_arrow_type_text(field_row["type"])
                    normalized_schema.append(field_row)
                result[f"{output_market}/{family}"] = normalized_schema
    return result


# ---------------------------------------------------------------------------
# Source-state audit
# ---------------------------------------------------------------------------


def audit_source_states(
    rows: Sequence[sqlite3.Row],
    canonical_root: Path,
) -> tuple[
    list[Issue],
    dict[str, dict[str, Any]],
    dict[str, str],
    Counter[str],
    Counter[str],
]:
    issues: list[Issue] = []
    sources_by_path: dict[str, dict[str, Any]] = {}
    path_owner: dict[str, str] = {}
    state_family_rows: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    source_ids: dict[str, str] = {}

    for raw_row in rows:
        row = dict(raw_row)
        relative = str(row["source_relative_path"])
        source_id = str(row["source_file_id"])
        sources_by_path[relative] = row
        status_counts[str(row["status"])] += 1

        if source_id in source_ids and source_ids[source_id] != relative:
            issues.append(
                Issue(
                    "FAIL",
                    "DUPLICATE_SOURCE_FILE_ID",
                    "state",
                    f"Source file ID {source_id} belongs to multiple relative paths.",
                    source_relative_path=relative,
                )
            )
        source_ids[source_id] = relative

        if row["status"] != "COMPLETED":
            issues.append(
                Issue(
                    "FAIL",
                    "SOURCE_NOT_COMPLETED",
                    "state",
                    f"Source status is {row['status']!r}, expected 'COMPLETED'.",
                    source_relative_path=relative,
                )
            )

        records_read = int(row.get("records_read") or 0)
        records_written = int(row.get("records_written") or 0)
        malformed = int(row.get("malformed_records") or 0)
        skipped_system = int(row.get("skipped_system_records") or 0)
        unknown = int(row.get("unknown_records") or 0)

        if records_read != records_written + malformed + skipped_system:
            issues.append(
                Issue(
                    "FAIL",
                    "SOURCE_RECORD_ACCOUNTING_MISMATCH",
                    "state",
                    "records_read does not equal records_written + malformed_records "
                    "+ skipped_system_records: "
                    f"{records_read:,} != {records_written:,} + {malformed:,} + {skipped_system:,}.",
                    source_relative_path=relative,
                )
            )

        family_counts, family_sum, family_valid = json_counter_sum(row.get("family_counts_json"))
        if not family_valid:
            issues.append(
                Issue(
                    "FAIL",
                    "INVALID_FAMILY_COUNTS_JSON",
                    "state",
                    "family_counts_json is absent, malformed, or contains non-integer values.",
                    source_relative_path=relative,
                )
            )
        if family_sum != records_written:
            issues.append(
                Issue(
                    "FAIL",
                    "SOURCE_FAMILY_COUNT_MISMATCH",
                    "state",
                    f"Family counts sum to {family_sum:,}, but records_written is {records_written:,}.",
                    source_relative_path=relative,
                )
            )
        state_family_rows.update(family_counts)

        event_counts, event_sum, event_valid = json_counter_sum(row.get("event_type_counts_json"))
        if not event_valid:
            issues.append(
                Issue(
                    "FAIL",
                    "INVALID_EVENT_COUNTS_JSON",
                    "state",
                    "event_type_counts_json is absent, malformed, or contains non-integer values.",
                    source_relative_path=relative,
                )
            )
        if malformed == 0 and event_sum != records_read:
            issues.append(
                Issue(
                    "FAIL",
                    "SOURCE_EVENT_COUNT_MISMATCH",
                    "state",
                    f"Event-type counts sum to {event_sum:,}, but records_read is {records_read:,}.",
                    source_relative_path=relative,
                )
            )

        output_paths = load_json_text(row.get("output_paths_json"), [])
        if not isinstance(output_paths, list):
            output_paths = []
            issues.append(
                Issue(
                    "FAIL",
                    "INVALID_OUTPUT_PATHS_JSON",
                    "state",
                    "output_paths_json is not a JSON list.",
                    source_relative_path=relative,
                )
            )
        if records_written > 0 and not output_paths:
            issues.append(
                Issue(
                    "FAIL",
                    "SOURCE_HAS_NO_OUTPUT_PATHS",
                    "state",
                    "Completed source has written records but no registered output paths.",
                    source_relative_path=relative,
                )
            )

        for raw_output_path in output_paths:
            output_path = str(raw_output_path).replace("\\", "/")
            existing_owner = path_owner.get(output_path)
            if existing_owner is not None and existing_owner != relative:
                issues.append(
                    Issue(
                        "FAIL",
                        "OUTPUT_PATH_MULTIPLE_OWNERS",
                        "state",
                        f"Output path is registered to both {existing_owner!r} and {relative!r}.",
                        source_relative_path=relative,
                        output_path=output_path,
                    )
                )
            path_owner[output_path] = relative
            if not (canonical_root / output_path).exists():
                issues.append(
                    Issue(
                        "FAIL",
                        "REGISTERED_OUTPUT_MISSING",
                        "state",
                        "Registered output path does not exist.",
                        source_relative_path=relative,
                        output_path=output_path,
                    )
                )

        if malformed > 0:
            issues.append(
                Issue(
                    "WARN",
                    "SOURCE_HAS_MALFORMED_RECORDS",
                    "state",
                    f"Source completed with {malformed:,} malformed record(s).",
                    source_relative_path=relative,
                )
            )
        if unknown > 0:
            issues.append(
                Issue(
                    "WARN",
                    "SOURCE_HAS_UNKNOWN_RECORDS",
                    "state",
                    f"Source completed with {unknown:,} unknown routed record(s).",
                    source_relative_path=relative,
                )
            )

    return issues, sources_by_path, path_owner, state_family_rows, status_counts


# ---------------------------------------------------------------------------
# Raw bronze inventory
# ---------------------------------------------------------------------------


def audit_raw_inventory(
    input_root: Path,
    source_glob: str,
    sources_by_path: Mapping[str, Mapping[str, Any]],
    verify_fingerprints: bool,
    progress_every: int,
) -> tuple[list[Issue], list[dict[str, Any]], dict[str, int]]:
    issues: list[Issue] = []
    rows: list[dict[str, Any]] = []
    paths = sorted(path for path in input_root.glob(source_glob) if path.is_file())
    raw_by_relative: dict[str, Path] = {
        path.resolve().relative_to(input_root.resolve()).as_posix(): path.resolve()
        for path in paths
    }

    state_paths = set(sources_by_path)
    raw_paths = set(raw_by_relative)

    for relative in sorted(raw_paths - state_paths):
        issues.append(
            Issue(
                "FAIL",
                "RAW_FILE_MISSING_FROM_STATE",
                "raw_inventory",
                "Raw WAL exists but has no source_files state record.",
                source_relative_path=relative,
            )
        )
        rows.append({
            "source_relative_path": relative,
            "result": "RAW_ONLY",
            "raw_size_bytes": raw_by_relative[relative].stat().st_size,
            "state_size_bytes": None,
            "fingerprint_checked": False,
        })

    for relative in sorted(state_paths - raw_paths):
        issues.append(
            Issue(
                "FAIL",
                "STATE_FILE_MISSING_FROM_RAW",
                "raw_inventory",
                "Source exists in extraction state but the raw WAL is missing.",
                source_relative_path=relative,
            )
        )
        rows.append({
            "source_relative_path": relative,
            "result": "STATE_ONLY",
            "raw_size_bytes": None,
            "state_size_bytes": sources_by_path[relative].get("size_bytes"),
            "fingerprint_checked": False,
        })

    common = sorted(raw_paths & state_paths)
    for index, relative in enumerate(common, start=1):
        path = raw_by_relative[relative]
        state = sources_by_path[relative]
        stat = path.stat()
        result = "MATCH"
        if int(state["size_bytes"]) != stat.st_size:
            result = "SIZE_MISMATCH"
            issues.append(
                Issue(
                    "FAIL",
                    "RAW_SIZE_MISMATCH",
                    "raw_inventory",
                    f"Raw size is {stat.st_size:,}; state size is {int(state['size_bytes']):,}.",
                    source_relative_path=relative,
                )
            )

        fingerprint_checked = False
        if verify_fingerprints and result == "MATCH":
            fingerprint_checked = True
            mode = str(state.get("fingerprint_mode") or "quick")
            current = full_fingerprint(path) if mode == "full" else quick_fingerprint(path)
            if current != state.get("fingerprint"):
                result = "FINGERPRINT_MISMATCH"
                issues.append(
                    Issue(
                        "FAIL",
                        "RAW_FINGERPRINT_MISMATCH",
                        "raw_inventory",
                        f"Current {mode} fingerprint differs from extraction state.",
                        source_relative_path=relative,
                    )
                )

        if result != "MATCH" or verify_fingerprints:
            rows.append({
                "source_relative_path": relative,
                "result": result,
                "raw_size_bytes": stat.st_size,
                "state_size_bytes": state.get("size_bytes"),
                "fingerprint_checked": fingerprint_checked,
            })

        if progress_every > 0 and index % progress_every == 0:
            print(f"  Raw inventory checked {index:,}/{len(common):,} files", flush=True)

    summary = {
        "raw_files": len(raw_paths),
        "state_files": len(state_paths),
        "common_files": len(common),
        "raw_only": len(raw_paths - state_paths),
        "state_only": len(state_paths - raw_paths),
    }
    return issues, rows, summary


# ---------------------------------------------------------------------------
# Parquet shard audit
# ---------------------------------------------------------------------------


def parse_shard_path(relative_path: str) -> tuple[str | None, str | None, str | None, str | None, list[Issue]]:
    issues: list[Issue] = []
    parts = Path(relative_path).as_posix().split("/")
    if len(parts) != 4:
        issues.append(
            Issue(
                "FAIL",
                "INVALID_SHARD_PATH_LAYOUT",
                "parquet",
                "Expected <market>/<family>/date=YYYY-MM-DD/part-<source-id>.parquet.",
                output_path=relative_path,
            )
        )
        return None, None, None, None, issues

    market_root, family_name, date_component, filename = parts
    date_match = DATE_PARTITION_RE.match(date_component)
    partition_date = date_match.group(1) if date_match else None
    if partition_date is None:
        issues.append(
            Issue(
                "FAIL",
                "INVALID_DATE_PARTITION",
                "parquet",
                f"Invalid partition component {date_component!r}.",
                output_path=relative_path,
                family=f"{market_root}/{family_name}",
            )
        )

    filename_match = PART_FILENAME_RE.match(filename)
    source_id = filename_match.group(1).lower() if filename_match else None
    if source_id is None:
        issues.append(
            Issue(
                "FAIL",
                "INVALID_SHARD_FILENAME",
                "parquet",
                "Expected filename part-<24 hex source_file_id>.parquet.",
                output_path=relative_path,
                family=f"{market_root}/{family_name}",
            )
        )
    return market_root, family_name, partition_date, source_id, issues


def inspect_parquet_shard(
    path: Path,
    canonical_root: Path,
    expected_schemas: Mapping[str, list[dict[str, Any]]],
    expected_source_path: str | None,
    expected_source_id: str | None,
    expected_records_read: int | None,
) -> ShardAudit:
    relative = path.resolve().relative_to(canonical_root.resolve()).as_posix()
    market_root, family_name, partition_date, filename_source_id, path_issues = parse_shard_path(relative)
    family_key = f"{market_root}/{family_name}" if market_root and family_name else None
    audit = ShardAudit(
        relative_path=relative,
        market_root=market_root,
        family_name=family_name,
        family_key=family_key,
        partition_date=partition_date,
        filename_source_file_id=filename_source_id,
        size_bytes=path.stat().st_size,
        issues=path_issues,
    )

    try:
        parquet_file = pq.ParquetFile(path)
        audit.rows = int(parquet_file.metadata.num_rows)
        audit.row_groups = int(parquet_file.metadata.num_row_groups)
        schema = parquet_file.schema_arrow
        audit.schema_signature = schema_signature(schema)
        audit.schema_name = decode_metadata(schema.metadata, "canonical_schema_name")
        audit.extractor_version = decode_metadata(schema.metadata, "extractor_version")
        audit.canonical_schema_metadata = decode_metadata(schema.metadata, "canonical_schema_version")

        if audit.rows <= 0:
            audit.issues.append(
                Issue(
                    "FAIL",
                    "EMPTY_PARQUET_SHARD",
                    "parquet",
                    "Parquet shard has zero rows.",
                    output_path=relative,
                    family=family_key,
                )
            )

        actual_columns = set(schema.names)
        missing_common = sorted(COMMON_REQUIRED_COLUMNS - actual_columns)
        if missing_common:
            audit.issues.append(
                Issue(
                    "FAIL",
                    "MISSING_COMMON_COLUMNS",
                    "schema",
                    f"Missing required common columns: {missing_common}.",
                    output_path=relative,
                    family=family_key,
                )
            )

        expected_schema = expected_schemas.get(family_key or "")
        if expected_schema is not None and schema_description(schema) != expected_schema:
            audit.issues.append(
                Issue(
                    "FAIL",
                    "SCHEMA_MISMATCH",
                    "schema",
                    "Physical Arrow schema differs semantically from schema_manifest.json.",
                    output_path=relative,
                    family=family_key,
                )
            )

        if audit.canonical_schema_metadata is None:
            audit.issues.append(
                Issue(
                    "FAIL",
                    "SCHEMA_METADATA_VERSION_MISSING",
                    "schema",
                    "Parquet schema metadata does not declare canonical_schema_version.",
                    output_path=relative,
                    family=family_key,
                )
            )
        elif audit.canonical_schema_metadata != str(EXPECTED_CANONICAL_SCHEMA_VERSION):
            audit.issues.append(
                Issue(
                    "FAIL",
                    "SCHEMA_METADATA_VERSION_MISMATCH",
                    "schema",
                    f"Parquet metadata schema version is {audit.canonical_schema_metadata!r}.",
                    output_path=relative,
                    family=family_key,
                )
            )

        audit.ts_min_ns, audit.ts_max_ns, ts_nulls, ts_complete = parquet_scalar_stats(
            parquet_file, "ts_recv_ns"
        )
        audit.null_counts["ts_recv_ns"] = ts_nulls
        (
            audit.source_record_min,
            audit.source_record_max,
            record_nulls,
            record_complete,
        ) = parquet_scalar_stats(parquet_file, "source_record_number")
        audit.null_counts["source_record_number"] = record_nulls
        (
            audit.source_file_id_min,
            audit.source_file_id_max,
            source_id_nulls,
            source_id_complete,
        ) = parquet_scalar_stats(parquet_file, "source_file_id")
        audit.null_counts["source_file_id"] = source_id_nulls
        (
            audit.source_path_min,
            audit.source_path_max,
            source_path_nulls,
            source_path_complete,
        ) = parquet_scalar_stats(parquet_file, "source_relative_path")
        audit.null_counts["source_relative_path"] = source_path_nulls
        (
            audit.canonical_version_min,
            audit.canonical_version_max,
            version_nulls,
            version_complete,
        ) = parquet_scalar_stats(parquet_file, "canonical_schema_version")
        audit.null_counts["canonical_schema_version"] = version_nulls

        for column, complete in {
            "ts_recv_ns": ts_complete,
            "source_record_number": record_complete,
            "source_file_id": source_id_complete,
            "source_relative_path": source_path_complete,
            "canonical_schema_version": version_complete,
        }.items():
            if not complete:
                audit.issues.append(
                    Issue(
                        "WARN",
                        "INCOMPLETE_PARQUET_STATISTICS",
                        "parquet",
                        f"Row-group statistics are incomplete for {column!r}; standard audit cannot fully prove it.",
                        output_path=relative,
                        family=family_key,
                    )
                )

        if partition_date and audit.ts_min_ns is not None and audit.ts_max_ns is not None:
            min_date = ns_to_utc_date(int(audit.ts_min_ns))
            max_date = ns_to_utc_date(int(audit.ts_max_ns))
            if min_date != partition_date or max_date != partition_date:
                audit.issues.append(
                    Issue(
                        "FAIL",
                        "RECEIVE_DATE_PARTITION_MISMATCH",
                        "partition",
                        f"Partition is {partition_date}, but ts_recv_ns spans {min_date} to {max_date}.",
                        output_path=relative,
                        family=family_key,
                    )
                )

        if filename_source_id and expected_source_id and filename_source_id != expected_source_id.lower():
            audit.issues.append(
                Issue(
                    "FAIL",
                    "FILENAME_SOURCE_ID_MISMATCH",
                    "provenance",
                    f"Filename source ID {filename_source_id} differs from state ID {expected_source_id}.",
                    output_path=relative,
                    source_relative_path=expected_source_path,
                    family=family_key,
                )
            )

        if expected_source_id and source_id_complete:
            if audit.source_file_id_min != expected_source_id or audit.source_file_id_max != expected_source_id:
                audit.issues.append(
                    Issue(
                        "FAIL",
                        "ROW_SOURCE_ID_MISMATCH",
                        "provenance",
                        "source_file_id row-group statistics do not match the source state ID.",
                        output_path=relative,
                        source_relative_path=expected_source_path,
                        family=family_key,
                    )
                )

        if expected_source_path and source_path_complete:
            if audit.source_path_min != expected_source_path or audit.source_path_max != expected_source_path:
                audit.issues.append(
                    Issue(
                        "FAIL",
                        "ROW_SOURCE_PATH_MISMATCH",
                        "provenance",
                        "source_relative_path row-group statistics do not match the registered owner.",
                        output_path=relative,
                        source_relative_path=expected_source_path,
                        family=family_key,
                    )
                )

        if version_complete:
            if (
                audit.canonical_version_min != EXPECTED_CANONICAL_SCHEMA_VERSION
                or audit.canonical_version_max != EXPECTED_CANONICAL_SCHEMA_VERSION
            ):
                audit.issues.append(
                    Issue(
                        "FAIL",
                        "ROW_SCHEMA_VERSION_MISMATCH",
                        "schema",
                        "canonical_schema_version values differ from the expected version.",
                        output_path=relative,
                        family=family_key,
                    )
                )

        if expected_records_read is not None and record_complete:
            if audit.source_record_min is not None and int(audit.source_record_min) < 1:
                audit.issues.append(
                    Issue(
                        "FAIL",
                        "INVALID_SOURCE_RECORD_MIN",
                        "provenance",
                        f"Minimum source record number is {audit.source_record_min}, expected >= 1.",
                        output_path=relative,
                        source_relative_path=expected_source_path,
                        family=family_key,
                    )
                )
            if audit.source_record_max is not None and int(audit.source_record_max) > expected_records_read:
                audit.issues.append(
                    Issue(
                        "FAIL",
                        "SOURCE_RECORD_EXCEEDS_RAW_COUNT",
                        "provenance",
                        f"Maximum source record number {audit.source_record_max} exceeds records_read {expected_records_read}.",
                        output_path=relative,
                        source_relative_path=expected_source_path,
                        family=family_key,
                    )
                )

        targeted_required = {
            "perps/depth_updates": [
                "first_update_id",
                "final_update_id",
                "previous_final_update_id",
            ],
            "perps/book_snapshots": ["last_update_id", "bid_level_count", "ask_level_count"],
            "spot/book_snapshots": ["last_update_id", "bid_level_count", "ask_level_count"],
        }
        for column in targeted_required.get(family_key or "", []):
            _, _, nulls, complete = parquet_scalar_stats(parquet_file, column)
            audit.null_counts[column] = nulls
            if complete and nulls and nulls > 0:
                audit.issues.append(
                    Issue(
                        "FAIL",
                        "REQUIRED_BOOK_FIELD_NULLS",
                        "orderbook",
                        f"Required order-book field {column!r} contains {nulls:,} null value(s).",
                        output_path=relative,
                        family=family_key,
                    )
                )
            elif not complete:
                audit.issues.append(
                    Issue(
                        "WARN",
                        "BOOK_FIELD_STATS_UNAVAILABLE",
                        "orderbook",
                        f"Could not prove null count for {column!r} from Parquet statistics.",
                        output_path=relative,
                        family=family_key,
                    )
                )

    except Exception as exc:
        audit.issues.append(
            Issue(
                "FAIL",
                "PARQUET_UNREADABLE",
                "parquet",
                f"{type(exc).__name__}: {exc}",
                output_path=relative,
                family=family_key,
            )
        )
    return audit


def audit_all_shards(
    canonical_root: Path,
    expected_schemas: Mapping[str, list[dict[str, Any]]],
    path_owner: Mapping[str, str],
    sources_by_path: Mapping[str, Mapping[str, Any]],
    workers: int,
    progress_every: int,
    max_shards: int | None,
) -> tuple[list[ShardAudit], list[Issue], dict[str, int]]:
    physical_paths = sorted(
        path
        for path in canonical_root.rglob("*.parquet")
        if "_extraction_state" not in path.parts and "_staging" not in path.parts
    )
    if max_shards is not None:
        physical_paths = physical_paths[:max_shards]

    expected_parquet_paths = {
        path for path in path_owner if path.lower().endswith(".parquet")
    }
    physical_relative = {
        path.resolve().relative_to(canonical_root.resolve()).as_posix()
        for path in physical_paths
    }

    issues: list[Issue] = []
    for orphan in sorted(physical_relative - expected_parquet_paths):
        issues.append(
            Issue(
                "FAIL",
                "ORPHAN_PARQUET_SHARD",
                "parquet",
                "Physical Parquet shard is not registered to any source file.",
                output_path=orphan,
            )
        )
    if max_shards is None:
        for missing in sorted(expected_parquet_paths - physical_relative):
            issues.append(
                Issue(
                    "FAIL",
                    "REGISTERED_PARQUET_NOT_DISCOVERED",
                    "parquet",
                    "State-registered Parquet shard was not discovered physically.",
                    source_relative_path=path_owner.get(missing),
                    output_path=missing,
                )
            )

    audits: list[ShardAudit] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_to_path = {}
        for path in physical_paths:
            relative = path.resolve().relative_to(canonical_root.resolve()).as_posix()
            owner = path_owner.get(relative)
            state = sources_by_path.get(owner or "")
            future = executor.submit(
                inspect_parquet_shard,
                path,
                canonical_root,
                expected_schemas,
                owner,
                str(state["source_file_id"]) if state else None,
                int(state["records_read"]) if state else None,
            )
            future_to_path[future] = relative

        total = len(future_to_path)
        for index, future in enumerate(as_completed(future_to_path), start=1):
            relative = future_to_path[future]
            try:
                audit = future.result()
            except Exception as exc:  # defensive worker boundary
                audit = ShardAudit(
                    relative_path=relative,
                    market_root=None,
                    family_name=None,
                    family_key=None,
                    partition_date=None,
                    filename_source_file_id=None,
                    issues=[
                        Issue(
                            "FAIL",
                            "SHARD_AUDIT_WORKER_FAILURE",
                            "parquet",
                            f"{type(exc).__name__}: {exc}",
                            output_path=relative,
                        )
                    ],
                )
            audits.append(audit)
            issues.extend(audit.issues)
            if progress_every > 0 and index % progress_every == 0:
                print(f"  Parquet footers checked {index:,}/{total:,}", flush=True)

    audits.sort(key=lambda item: item.relative_path)
    summary = {
        "physical_parquet_files": len(physical_relative),
        "registered_parquet_files": len(expected_parquet_paths),
        "orphan_parquet_files": len(physical_relative - expected_parquet_paths),
        "missing_registered_parquet_files": (
            len(expected_parquet_paths - physical_relative) if max_shards is None else 0
        ),
    }
    return audits, issues, summary


# ---------------------------------------------------------------------------
# Reconciliation and deep provenance
# ---------------------------------------------------------------------------


def reconcile_shards_to_state(
    audits: Sequence[ShardAudit],
    sources_by_path: Mapping[str, Mapping[str, Any]],
    path_owner: Mapping[str, str],
    state_family_rows: Counter[str],
) -> tuple[list[Issue], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    issues: list[Issue] = []
    actual_source_rows: Counter[str] = Counter()
    actual_family_rows: Counter[str] = Counter()
    family_files: Counter[str] = Counter()
    family_sizes: Counter[str] = Counter()
    family_dates: dict[str, set[str]] = defaultdict(set)
    family_ts_min: dict[str, int] = {}
    family_ts_max: dict[str, int] = {}
    coverage: Counter[tuple[str, str]] = Counter()
    coverage_files: Counter[tuple[str, str]] = Counter()
    coverage_ts_min: dict[tuple[str, str], int] = {}
    coverage_ts_max: dict[tuple[str, str], int] = {}

    for audit in audits:
        owner = path_owner.get(audit.relative_path)
        if owner is not None:
            actual_source_rows[owner] += audit.rows
        if audit.family_key:
            actual_family_rows[audit.family_key] += audit.rows
            family_files[audit.family_key] += 1
            family_sizes[audit.family_key] += audit.size_bytes
            if audit.partition_date:
                family_dates[audit.family_key].add(audit.partition_date)
                key = (audit.family_key, audit.partition_date)
                coverage[key] += audit.rows
                coverage_files[key] += 1
                if audit.ts_min_ns is not None:
                    coverage_ts_min[key] = min(
                        coverage_ts_min.get(key, int(audit.ts_min_ns)), int(audit.ts_min_ns)
                    )
                if audit.ts_max_ns is not None:
                    coverage_ts_max[key] = max(
                        coverage_ts_max.get(key, int(audit.ts_max_ns)), int(audit.ts_max_ns)
                    )
            if audit.ts_min_ns is not None:
                family_ts_min[audit.family_key] = min(
                    family_ts_min.get(audit.family_key, int(audit.ts_min_ns)), int(audit.ts_min_ns)
                )
            if audit.ts_max_ns is not None:
                family_ts_max[audit.family_key] = max(
                    family_ts_max.get(audit.family_key, int(audit.ts_max_ns)), int(audit.ts_max_ns)
                )

    source_rows: list[dict[str, Any]] = []
    for relative, state in sorted(sources_by_path.items()):
        expected = int(state.get("records_written") or 0)
        actual = int(actual_source_rows.get(relative, 0))
        result = "PASS" if expected == actual else "FAIL"
        source_rows.append({
            "source_relative_path": relative,
            "source_file_id": state.get("source_file_id"),
            "status": state.get("status"),
            "records_read": state.get("records_read"),
            "records_written_state": expected,
            "records_in_parquet": actual,
            "difference": actual - expected,
            "malformed_records": state.get("malformed_records"),
            "unknown_records": state.get("unknown_records"),
            "skipped_system_records": state.get("skipped_system_records"),
            "result": result,
        })
        if result == "FAIL":
            issues.append(
                Issue(
                    "FAIL",
                    "SOURCE_PARQUET_ROW_COUNT_MISMATCH",
                    "reconciliation",
                    f"State records_written={expected:,}; physical Parquet rows={actual:,}.",
                    source_relative_path=relative,
                )
            )

    all_families = sorted(set(state_family_rows) | set(actual_family_rows))
    family_rows: list[dict[str, Any]] = []
    for family in all_families:
        expected = int(state_family_rows.get(family, 0))
        actual = int(actual_family_rows.get(family, 0))
        result = "PASS" if expected == actual else "FAIL"
        dates = sorted(family_dates.get(family, set()))
        family_rows.append({
            "family": family,
            "state_rows": expected,
            "parquet_rows": actual,
            "difference": actual - expected,
            "parquet_files": family_files.get(family, 0),
            "size_bytes": family_sizes.get(family, 0),
            "size_gib": round(bytes_to_gib(family_sizes.get(family, 0)), 4),
            "first_partition_date": dates[0] if dates else None,
            "last_partition_date": dates[-1] if dates else None,
            "first_ts_recv_ns": family_ts_min.get(family),
            "last_ts_recv_ns": family_ts_max.get(family),
            "result": result,
        })
        if result == "FAIL":
            issues.append(
                Issue(
                    "FAIL",
                    "FAMILY_PARQUET_ROW_COUNT_MISMATCH",
                    "reconciliation",
                    f"State family rows={expected:,}; physical Parquet rows={actual:,}.",
                    family=family,
                )
            )

    coverage_rows = [
        {
            "family": family,
            "date": date,
            "rows": rows,
            "parquet_files": coverage_files[(family, date)],
            "first_ts_recv_ns": coverage_ts_min.get((family, date)),
            "last_ts_recv_ns": coverage_ts_max.get((family, date)),
        }
        for (family, date), rows in sorted(coverage.items())
    ]
    return issues, source_rows, family_rows, coverage_rows


def deep_provenance_audit(
    canonical_root: Path,
    sources_by_path: Mapping[str, Mapping[str, Any]],
    progress_every: int,
) -> tuple[list[Issue], list[dict[str, Any]]]:
    issues: list[Issue] = []
    result_rows: list[dict[str, Any]] = []
    sources = sorted(sources_by_path.items())

    for index, (relative, state) in enumerate(sources, start=1):
        output_paths = load_json_text(state.get("output_paths_json"), [])
        parquet_paths = [
            canonical_root / str(path)
            for path in output_paths
            if str(path).lower().endswith(".parquet")
        ]
        record_arrays: list[np.ndarray] = []
        bad_source_id = 0
        bad_source_path = 0
        bad_schema_version = 0
        rows_scanned = 0
        error_message: str | None = None

        try:
            for path in parquet_paths:
                parquet_file = pq.ParquetFile(path)
                for batch in parquet_file.iter_batches(
                    batch_size=131_072,
                    columns=[
                        "source_file_id",
                        "source_relative_path",
                        "source_record_number",
                        "canonical_schema_version",
                    ],
                    use_threads=False,
                ):
                    rows_scanned += batch.num_rows
                    source_id_array = batch.column(batch.schema.get_field_index("source_file_id"))
                    source_path_array = batch.column(batch.schema.get_field_index("source_relative_path"))
                    record_array = batch.column(batch.schema.get_field_index("source_record_number"))
                    version_array = batch.column(batch.schema.get_field_index("canonical_schema_version"))

                    source_id_matches = pc.sum(
                        pc.cast(pc.equal(source_id_array, state["source_file_id"]), pa.int64())
                    ).as_py()
                    source_path_matches = pc.sum(
                        pc.cast(pc.equal(source_path_array, relative), pa.int64())
                    ).as_py()
                    bad_source_id += batch.num_rows - int(source_id_matches or 0)
                    bad_source_path += batch.num_rows - int(source_path_matches or 0)
                    version_matches = pc.sum(
                        pc.cast(
                            pc.equal(version_array, EXPECTED_CANONICAL_SCHEMA_VERSION),
                            pa.int64(),
                        )
                    ).as_py()
                    bad_schema_version += batch.num_rows - int(version_matches or 0)
                    record_arrays.append(record_array.to_numpy(zero_copy_only=False))

            if record_arrays:
                record_numbers = np.concatenate(record_arrays).astype(np.int64, copy=False)
                sorted_numbers = np.sort(record_numbers)
                duplicate_count = int(np.count_nonzero(sorted_numbers[1:] == sorted_numbers[:-1]))
                unique_count = int(sorted_numbers.size - duplicate_count)
                minimum = int(sorted_numbers[0])
                maximum = int(sorted_numbers[-1])
            else:
                duplicate_count = 0
                unique_count = 0
                minimum = None
                maximum = None

            records_read = int(state.get("records_read") or 0)
            expected_written = int(state.get("records_written") or 0)
            expected_unwritten = int(state.get("malformed_records") or 0) + int(
                state.get("skipped_system_records") or 0
            )
            missing_count = records_read - unique_count

            if rows_scanned != expected_written:
                issues.append(
                    Issue(
                        "FAIL",
                        "DEEP_ROW_COUNT_MISMATCH",
                        "deep_provenance",
                        f"Scanned {rows_scanned:,} rows; state records_written is {expected_written:,}.",
                        source_relative_path=relative,
                    )
                )
            if duplicate_count > 0:
                issues.append(
                    Issue(
                        "FAIL",
                        "DUPLICATE_SOURCE_RECORD_NUMBER",
                        "deep_provenance",
                        f"Found {duplicate_count:,} duplicated source_record_number value(s).",
                        source_relative_path=relative,
                    )
                )
            if missing_count != expected_unwritten:
                issues.append(
                    Issue(
                        "FAIL",
                        "MISSING_SOURCE_RECORD_ACCOUNTING_MISMATCH",
                        "deep_provenance",
                        f"Missing provenance numbers={missing_count:,}; expected skipped+malformed={expected_unwritten:,}.",
                        source_relative_path=relative,
                    )
                )
            if minimum is not None and minimum < 1:
                issues.append(
                    Issue(
                        "FAIL",
                        "DEEP_INVALID_RECORD_MIN",
                        "deep_provenance",
                        f"Minimum source_record_number is {minimum}.",
                        source_relative_path=relative,
                    )
                )
            if maximum is not None and maximum > records_read:
                issues.append(
                    Issue(
                        "FAIL",
                        "DEEP_RECORD_MAX_EXCEEDS_RAW",
                        "deep_provenance",
                        f"Maximum source_record_number is {maximum}; records_read is {records_read}.",
                        source_relative_path=relative,
                    )
                )
            if bad_source_id:
                issues.append(
                    Issue(
                        "FAIL",
                        "DEEP_SOURCE_ID_MISMATCH",
                        "deep_provenance",
                        f"Found {bad_source_id:,} row(s) with the wrong source_file_id.",
                        source_relative_path=relative,
                    )
                )
            if bad_source_path:
                issues.append(
                    Issue(
                        "FAIL",
                        "DEEP_SOURCE_PATH_MISMATCH",
                        "deep_provenance",
                        f"Found {bad_source_path:,} row(s) with the wrong source_relative_path.",
                        source_relative_path=relative,
                    )
                )
            if bad_schema_version:
                issues.append(
                    Issue(
                        "FAIL",
                        "DEEP_SCHEMA_VERSION_MISMATCH",
                        "deep_provenance",
                        f"Found {bad_schema_version:,} row(s) with a different schema version.",
                        source_relative_path=relative,
                    )
                )

            result = "PASS" if not any(
                issue.source_relative_path == relative
                and issue.scope == "deep_provenance"
                and issue.severity == "FAIL"
                for issue in issues
            ) else "FAIL"
            result_rows.append({
                "source_relative_path": relative,
                "source_file_id": state.get("source_file_id"),
                "rows_scanned": rows_scanned,
                "unique_record_numbers": unique_count,
                "duplicate_record_numbers": duplicate_count,
                "missing_record_numbers": missing_count,
                "expected_unwritten_records": expected_unwritten,
                "minimum_record_number": minimum,
                "maximum_record_number": maximum,
                "bad_source_id_rows": bad_source_id,
                "bad_source_path_rows": bad_source_path,
                "bad_schema_version_rows": bad_schema_version,
                "result": result,
                "error_message": None,
            })
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            issues.append(
                Issue(
                    "FAIL",
                    "DEEP_PROVENANCE_SCAN_FAILED",
                    "deep_provenance",
                    error_message,
                    source_relative_path=relative,
                )
            )
            result_rows.append({
                "source_relative_path": relative,
                "source_file_id": state.get("source_file_id"),
                "rows_scanned": rows_scanned,
                "unique_record_numbers": None,
                "duplicate_record_numbers": None,
                "missing_record_numbers": None,
                "expected_unwritten_records": None,
                "minimum_record_number": None,
                "maximum_record_number": None,
                "bad_source_id_rows": bad_source_id,
                "bad_source_path_rows": bad_source_path,
                "bad_schema_version_rows": bad_schema_version,
                "result": "FAIL",
                "error_message": error_message,
            })

        if progress_every > 0 and index % progress_every == 0:
            print(f"  Deep provenance checked {index:,}/{len(sources):,} sources", flush=True)

    return issues, result_rows


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def build_invariants(
    issues: Sequence[Issue],
    status_counts: Mapping[str, int],
    source_count: int,
    shard_count: int,
    unknown_total: int,
    malformed_total: int,
    raw_inventory_performed: bool,
    deep_performed: bool,
) -> list[Invariant]:
    groups: list[tuple[str, set[str], str]] = [
        (
            "All extraction source files completed",
            {"SOURCE_NOT_COMPLETED"},
            f"Sources={source_count:,}; statuses={dict(status_counts)}.",
        ),
        (
            "Source record accounting reconciles",
            {
                "SOURCE_RECORD_ACCOUNTING_MISMATCH",
                "SOURCE_FAMILY_COUNT_MISMATCH",
                "SOURCE_EVENT_COUNT_MISMATCH",
                "INVALID_FAMILY_COUNTS_JSON",
                "INVALID_EVENT_COUNTS_JSON",
            },
            "records_read, records_written, family counts, malformed, and skipped records reconcile per source.",
        ),
        (
            "Registered outputs exist and have one owner",
            {
                "REGISTERED_OUTPUT_MISSING",
                "SOURCE_HAS_NO_OUTPUT_PATHS",
                "OUTPUT_PATH_MULTIPLE_OWNERS",
                "INVALID_OUTPUT_PATHS_JSON",
            },
            "Every state-registered output path must exist and belong to exactly one source WAL.",
        ),
        (
            "Physical Parquet inventory matches state",
            {
                "ORPHAN_PARQUET_SHARD",
                "REGISTERED_PARQUET_NOT_DISCOVERED",
                "PARQUET_UNREADABLE",
                "EMPTY_PARQUET_SHARD",
                "INVALID_SHARD_PATH_LAYOUT",
                "INVALID_DATE_PARTITION",
                "INVALID_SHARD_FILENAME",
            },
            f"Parquet shards inspected={shard_count:,}.",
        ),
        (
            "Canonical schemas are consistent",
            {
                "SCHEMA_MISMATCH",
                "MISSING_COMMON_COLUMNS",
                "SCHEMA_METADATA_VERSION_MISSING",
                "SCHEMA_METADATA_VERSION_MISMATCH",
                "ROW_SCHEMA_VERSION_MISMATCH",
            },
            f"Expected canonical schema version={EXPECTED_CANONICAL_SCHEMA_VERSION}.",
        ),
        (
            "Receive timestamps match date partitions",
            {"RECEIVE_DATE_PARTITION_MISMATCH"},
            "Every shard is partitioned by the UTC date derived from ts_recv_ns.",
        ),
        (
            "Source provenance agrees with state",
            {
                "DUPLICATE_SOURCE_FILE_ID",
                "FILENAME_SOURCE_ID_MISMATCH",
                "ROW_SOURCE_ID_MISMATCH",
                "ROW_SOURCE_PATH_MISMATCH",
                "INVALID_SOURCE_RECORD_MIN",
                "SOURCE_RECORD_EXCEEDS_RAW_COUNT",
            },
            "Filename, row-group statistics, and source state identify the same WAL.",
        ),
        (
            "Physical row counts reconcile to state",
            {
                "SOURCE_PARQUET_ROW_COUNT_MISMATCH",
                "FAMILY_PARQUET_ROW_COUNT_MISMATCH",
            },
            "Parquet footer row counts reconcile per source and per canonical family.",
        ),
        (
            "Order-book sequence and snapshot fields are populated",
            {"REQUIRED_BOOK_FIELD_NULLS"},
            "Depth update IDs and snapshot update/level-count fields must be non-null.",
        ),
    ]

    invariants: list[Invariant] = []
    for name, codes, details in groups:
        relevant = [issue for issue in issues if issue.code in codes]
        invariants.append(
            Invariant(
                name=name,
                status=status_from_issues(relevant),
                issue_count=len(relevant),
                details=details,
            )
        )

    raw_codes = {
        "RAW_FILE_MISSING_FROM_STATE",
        "STATE_FILE_MISSING_FROM_RAW",
        "RAW_SIZE_MISMATCH",
        "RAW_FINGERPRINT_MISMATCH",
    }
    raw_relevant = [issue for issue in issues if issue.code in raw_codes]
    invariants.append(
        Invariant(
            name="Raw bronze inventory matches extraction state",
            status=(status_from_issues(raw_relevant) if raw_inventory_performed else "NOT_RUN"),
            issue_count=len(raw_relevant),
            details=(
                "Raw paths and sizes were compared with source_files state."
                if raw_inventory_performed
                else "No input root was available; pass --input-root to run this check."
            ),
        )
    )

    deep_codes = {
        "DEEP_ROW_COUNT_MISMATCH",
        "DUPLICATE_SOURCE_RECORD_NUMBER",
        "MISSING_SOURCE_RECORD_ACCOUNTING_MISMATCH",
        "DEEP_INVALID_RECORD_MIN",
        "DEEP_RECORD_MAX_EXCEEDS_RAW",
        "DEEP_SOURCE_ID_MISMATCH",
        "DEEP_SOURCE_PATH_MISMATCH",
        "DEEP_SCHEMA_VERSION_MISMATCH",
        "DEEP_PROVENANCE_SCAN_FAILED",
    }
    deep_relevant = [issue for issue in issues if issue.code in deep_codes]
    invariants.append(
        Invariant(
            name="Exact source-record provenance is unique and complete",
            status=(status_from_issues(deep_relevant) if deep_performed else "NOT_RUN"),
            issue_count=len(deep_relevant),
            details=(
                "All source_record_number values were scanned source-by-source."
                if deep_performed
                else "Run with --deep-provenance for this exact, longer audit."
            ),
        )
    )

    unknown_issues = [issue for issue in issues if issue.code == "SOURCE_HAS_UNKNOWN_RECORDS"]
    invariants.append(
        Invariant(
            name="No unresolved unknown market events",
            status="PASS" if unknown_total == 0 else status_from_issues(unknown_issues),
            issue_count=unknown_total,
            details=f"Aggregate unknown_records in source state={unknown_total:,}.",
        )
    )
    malformed_issues = [issue for issue in issues if issue.code == "SOURCE_HAS_MALFORMED_RECORDS"]
    invariants.append(
        Invariant(
            name="No malformed source records",
            status="PASS" if malformed_total == 0 else status_from_issues(malformed_issues),
            issue_count=malformed_total,
            details=f"Aggregate malformed_records in source state={malformed_total:,}.",
        )
    )
    return invariants


def write_findings(
    path: Path,
    summary: Mapping[str, Any],
    invariants: Sequence[Invariant],
    family_rows: Sequence[Mapping[str, Any]],
    issues: Sequence[Issue],
) -> None:
    fail_count = sum(1 for issue in issues if issue.severity == "FAIL")
    warn_count = sum(1 for issue in issues if issue.severity == "WARN")
    lines = [
        "# Canonical Market Events Audit",
        "",
        f"- Auditor version: `{AUDITOR_VERSION}`",
        f"- Run ID: `{summary['run_id']}`",
        f"- Status: **{summary['status']}**",
        f"- Canonical root: `{summary['canonical_root']}`",
        f"- Sources in state: **{summary['source_files']:,}**",
        f"- Parquet shards inspected: **{summary['parquet_files']:,}**",
        f"- Records committed in state: **{summary['records_written_state']:,}**",
        f"- Records counted in Parquet: **{summary['records_in_parquet']:,}**",
        f"- Fail issues: **{fail_count:,}**",
        f"- Warning issues: **{warn_count:,}**",
        "",
        "## Invariants",
        "",
        "| Invariant | Status | Issues | Details |",
        "|---|---:|---:|---|",
    ]
    for invariant in invariants:
        details = invariant.details.replace("|", "\\|")
        lines.append(
            f"| {invariant.name} | {invariant.status} | {invariant.issue_count:,} | {details} |"
        )

    lines.extend(["", "## Canonical families", "", "| Family | Rows | Files | First date | Last date | Result |", "|---|---:|---:|---|---|---|"])
    for row in family_rows:
        lines.append(
            f"| {row['family']} | {int(row['parquet_rows']):,} | {int(row['parquet_files']):,} "
            f"| {row.get('first_partition_date') or ''} | {row.get('last_partition_date') or ''} "
            f"| {row['result']} |"
        )

    issue_counter = Counter((issue.severity, issue.code) for issue in issues)
    lines.extend(["", "## Issue summary", "", "| Severity | Code | Count |", "|---|---|---:|"])
    if issue_counter:
        for (severity, code), count in sorted(
            issue_counter.items(), key=lambda item: (-SEVERITY_ORDER[item[0][0]], item[0][1])
        ):
            lines.append(f"| {severity} | `{code}` | {count:,} |")
    else:
        lines.append("| INFO | `NONE` | 0 |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A `PASS` audit means the standard state, Parquet-footer, schema, partition, and row-count checks found no failures. ",
            "The exact source-record uniqueness check is reported separately and is only performed when `--deep-provenance` is supplied.",
            "",
        ]
    )
    atomic_write_text(path, "\n".join(lines))


# ---------------------------------------------------------------------------
# CLI and orchestration
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Stream canonical market events against extraction state and Parquet output."
    )
    parser.add_argument(
        "--canonical-root",
        default=DEFAULT_CANONICAL_ROOT,
        help="Canonical market_events_v1 root.",
    )
    parser.add_argument(
        "--state-db",
        help="Defaults to CANONICAL_ROOT/_extraction_state/extraction_state.sqlite.",
    )
    parser.add_argument(
        "--input-root",
        help="Raw bronze root. If omitted, inferred from the latest extraction run when available.",
    )
    parser.add_argument(
        "--source-glob",
        default=DEFAULT_SOURCE_GLOB,
        help="Raw WAL discovery glob for the bronze inventory audit.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_REPORT_ROOT,
        help="Directory for audit CSV, JSON, and Markdown reports.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Threads used to inspect independent Parquet footers.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress after this many shards/sources; 0 disables progress messages.",
    )
    parser.add_argument(
        "--verify-fingerprints",
        action="store_true",
        help="Recompute each raw WAL fingerprint. More expensive than the default path/size audit.",
    )
    parser.add_argument(
        "--deep-provenance",
        action="store_true",
        help="Read provenance columns for every canonical row to detect exact duplicates/missing lines.",
    )
    parser.add_argument(
        "--skip-raw-inventory",
        action="store_true",
        help="Do not compare the raw bronze inventory with source state.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        help="Debug only: inspect at most this many physical Parquet shards.",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Return a non-zero exit status when warnings exist even if there are no failures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = utc_now_iso()
    run_id = utc_run_id()
    canonical_root = Path(args.canonical_root).expanduser().resolve()
    state_db = (
        Path(args.state_db).expanduser().resolve()
        if args.state_db
        else canonical_root / "_extraction_state" / "extraction_state.sqlite"
    )
    report_root = Path(args.output_dir).expanduser().resolve()
    run_report_root = report_root / "runs" / run_id
    run_report_root.mkdir(parents=True, exist_ok=True)

    print("Canonical market-events audit")
    print(f"Run ID         : {run_id}")
    print(f"Canonical root : {canonical_root}")
    print(f"State database : {state_db}")
    print(f"Report root    : {run_report_root}")

    if not canonical_root.is_dir():
        print(f"ERROR: Canonical root does not exist: {canonical_root}", file=sys.stderr)
        return 1
    if not state_db.is_file():
        print(f"ERROR: State database does not exist: {state_db}", file=sys.stderr)
        return 1

    all_issues: list[Issue] = []
    connection = connect_state_db(state_db)
    try:
        require_state_tables(connection)
        extraction_context = latest_extraction_context(connection)
        state_metadata = {
            row["key"]: row["value"]
            for row in connection.execute("SELECT key, value FROM metadata").fetchall()
        }
        source_rows_db = connection.execute(
            "SELECT * FROM source_files ORDER BY source_relative_path"
        ).fetchall()
    finally:
        connection.close()

    schema_manifest_path = canonical_root / "_extraction_state" / "schema_manifest.json"
    schema_manifest = load_json_file(schema_manifest_path, {})
    if not schema_manifest:
        all_issues.append(
            Issue(
                "FAIL",
                "SCHEMA_MANIFEST_MISSING_OR_INVALID",
                "state",
                f"Could not load {schema_manifest_path}.",
            )
        )
    expected_schemas = expected_schemas_from_manifest(schema_manifest)
    system_schema = schema_manifest.get("system_schema")
    if isinstance(system_schema, list):
        system_root = canonical_root / "system"
        if system_root.is_dir():
            for child in system_root.iterdir():
                if child.is_dir():
                    expected_schemas[f"system/{child.name}"] = list(system_schema)
    unknown_schema = schema_manifest.get("unknown_schema")
    if isinstance(unknown_schema, list):
        unknown_root = canonical_root / "unknown"
        if unknown_root.is_dir():
            for child in unknown_root.iterdir():
                if child.is_dir():
                    expected_schemas[f"unknown/{child.name}"] = list(unknown_schema)

    print(f"Source states   : {len(source_rows_db):,}")
    print("Auditing extraction state...")
    (
        state_issues,
        sources_by_path,
        path_owner,
        state_family_rows,
        status_counts,
    ) = audit_source_states(source_rows_db, canonical_root)
    all_issues.extend(state_issues)

    input_root: Path | None = None
    if args.input_root:
        input_root = Path(args.input_root).expanduser().resolve()
    elif extraction_context.get("input_root"):
        inferred = Path(str(extraction_context["input_root"]))
        if inferred.exists():
            input_root = inferred.resolve()

    raw_rows: list[dict[str, Any]] = []
    raw_summary: dict[str, int] = {}
    raw_inventory_performed = False
    if not args.skip_raw_inventory and input_root is not None and input_root.is_dir():
        raw_inventory_performed = True
        print(f"Auditing raw inventory: {input_root}")
        raw_issues, raw_rows, raw_summary = audit_raw_inventory(
            input_root,
            args.source_glob,
            sources_by_path,
            args.verify_fingerprints,
            args.progress_every,
        )
        all_issues.extend(raw_issues)
    elif not args.skip_raw_inventory:
        all_issues.append(
            Issue(
                "WARN",
                "RAW_INVENTORY_NOT_RUN",
                "raw_inventory",
                "No accessible input root was supplied or inferred.",
            )
        )

    print("Inspecting Parquet shards...")
    shard_audits, shard_issues, shard_summary = audit_all_shards(
        canonical_root,
        expected_schemas,
        path_owner,
        sources_by_path,
        args.workers,
        args.progress_every,
        args.max_shards,
    )
    all_issues.extend(shard_issues)

    print("Reconciling physical rows to state...")
    reconcile_issues, source_reconciliation, family_summary, coverage_daily = reconcile_shards_to_state(
        shard_audits,
        sources_by_path,
        path_owner,
        state_family_rows,
    )
    all_issues.extend(reconcile_issues)

    actual_family_names = {str(row["family"]) for row in family_summary}
    known_route_families = {
        f"{route.get('output_market')}/{route.get('family')}"
        for route in schema_manifest.get("known_routes", [])
        if isinstance(route, Mapping) and route.get("output_market") and route.get("family")
    }
    for family in sorted(known_route_families - actual_family_names):
        all_issues.append(
            Issue(
                "WARN",
                "EXPECTED_FAMILY_ABSENT",
                "reconciliation",
                "Known extractor route has no rows in the canonical store.",
                family=family,
            )
        )

    deep_rows: list[dict[str, Any]] = []
    if args.deep_provenance:
        print("Running exact deep provenance audit...")
        deep_issues, deep_rows = deep_provenance_audit(
            canonical_root,
            sources_by_path,
            args.progress_every,
        )
        all_issues.extend(deep_issues)

    unknown_total = sum(int(row.get("unknown_records") or 0) for row in sources_by_path.values())
    malformed_total = sum(int(row.get("malformed_records") or 0) for row in sources_by_path.values())
    records_read_state = sum(int(row.get("records_read") or 0) for row in sources_by_path.values())
    records_written_state = sum(int(row.get("records_written") or 0) for row in sources_by_path.values())
    records_in_parquet = sum(audit.rows for audit in shard_audits)
    total_parquet_bytes = sum(audit.size_bytes for audit in shard_audits)

    invariants = build_invariants(
        all_issues,
        status_counts,
        len(sources_by_path),
        len(shard_audits),
        unknown_total,
        malformed_total,
        raw_inventory_performed,
        args.deep_provenance,
    )

    fail_count = sum(1 for issue in all_issues if issue.severity == "FAIL")
    warn_count = sum(1 for issue in all_issues if issue.severity == "WARN")
    status = "FAIL" if fail_count else ("WARN" if warn_count else "PASS")
    completed_at = utc_now_iso()
    summary = {
        "auditor_version": AUDITOR_VERSION,
        "run_id": run_id,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "status": status,
        "canonical_root": str(canonical_root),
        "state_db": str(state_db),
        "input_root": str(input_root) if input_root else None,
        "report_root": str(run_report_root),
        "canonical_schema_version_state": state_metadata.get("canonical_schema_version"),
        "extractor_version_state": state_metadata.get("extractor_version"),
        "schema_manifest_extractor_version": schema_manifest.get("extractor_version"),
        "source_files": len(sources_by_path),
        "source_status_counts": dict(status_counts),
        "parquet_files": len(shard_audits),
        "parquet_size_bytes": total_parquet_bytes,
        "parquet_size_gib": round(bytes_to_gib(total_parquet_bytes), 4),
        "records_read_state": records_read_state,
        "records_written_state": records_written_state,
        "records_in_parquet": records_in_parquet,
        "unknown_records": unknown_total,
        "malformed_records": malformed_total,
        "fail_issues": fail_count,
        "warning_issues": warn_count,
        "raw_inventory_performed": raw_inventory_performed,
        "raw_inventory_summary": raw_summary,
        "deep_provenance_performed": args.deep_provenance,
        "shard_inventory_summary": shard_summary,
        "arguments": vars(args),
    }

    write_csv(
        run_report_root / "canonical_audit_invariants.csv",
        [item.as_dict() for item in invariants],
        ["name", "status", "issue_count", "details"],
    )
    write_csv(
        run_report_root / "canonical_audit_issues.csv",
        [item.as_dict() for item in all_issues],
        [
            "severity",
            "code",
            "scope",
            "message",
            "source_relative_path",
            "output_path",
            "family",
        ],
    )
    write_csv(
        run_report_root / "canonical_source_reconciliation.csv",
        source_reconciliation,
        [
            "source_relative_path",
            "source_file_id",
            "status",
            "records_read",
            "records_written_state",
            "records_in_parquet",
            "difference",
            "malformed_records",
            "unknown_records",
            "skipped_system_records",
            "result",
        ],
    )
    write_csv(
        run_report_root / "canonical_family_summary.csv",
        family_summary,
        [
            "family",
            "state_rows",
            "parquet_rows",
            "difference",
            "parquet_files",
            "size_bytes",
            "size_gib",
            "first_partition_date",
            "last_partition_date",
            "first_ts_recv_ns",
            "last_ts_recv_ns",
            "result",
        ],
    )
    write_csv(
        run_report_root / "canonical_daily_coverage.csv",
        coverage_daily,
        [
            "family",
            "date",
            "rows",
            "parquet_files",
            "first_ts_recv_ns",
            "last_ts_recv_ns",
        ],
    )
    write_csv(
        run_report_root / "canonical_parquet_shards.csv",
        [item.as_dict() for item in shard_audits],
        [
            "relative_path",
            "market_root",
            "family_name",
            "family_key",
            "partition_date",
            "filename_source_file_id",
            "rows",
            "row_groups",
            "size_bytes",
            "schema_signature",
            "schema_name",
            "extractor_version",
            "canonical_schema_metadata",
            "ts_min_ns",
            "ts_max_ns",
            "source_record_min",
            "source_record_max",
            "source_file_id_min",
            "source_file_id_max",
            "source_path_min",
            "source_path_max",
            "canonical_version_min",
            "canonical_version_max",
            "null_counts_json",
            "issue_count",
        ],
    )
    if raw_inventory_performed:
        write_csv(
            run_report_root / "canonical_raw_inventory.csv",
            raw_rows,
            [
                "source_relative_path",
                "result",
                "raw_size_bytes",
                "state_size_bytes",
                "fingerprint_checked",
            ],
        )
    if args.deep_provenance:
        write_csv(
            run_report_root / "canonical_deep_provenance.csv",
            deep_rows,
            [
                "source_relative_path",
                "source_file_id",
                "rows_scanned",
                "unique_record_numbers",
                "duplicate_record_numbers",
                "missing_record_numbers",
                "expected_unwritten_records",
                "minimum_record_number",
                "maximum_record_number",
                "bad_source_id_rows",
                "bad_source_path_rows",
                "bad_schema_version_rows",
                "result",
                "error_message",
            ],
        )

    atomic_write_text(
        run_report_root / "canonical_audit_summary.json",
        json.dumps(summary, indent=2, sort_keys=True, default=str),
    )
    write_findings(
        run_report_root / "canonical_audit_findings.md",
        summary,
        invariants,
        family_summary,
        all_issues,
    )

    # Stable latest copies make downstream workflows independent of run IDs.
    latest_files = [
        "canonical_audit_invariants.csv",
        "canonical_audit_issues.csv",
        "canonical_source_reconciliation.csv",
        "canonical_family_summary.csv",
        "canonical_daily_coverage.csv",
        "canonical_parquet_shards.csv",
        "canonical_audit_summary.json",
        "canonical_audit_findings.md",
    ]
    if raw_inventory_performed:
        latest_files.append("canonical_raw_inventory.csv")
    if args.deep_provenance:
        latest_files.append("canonical_deep_provenance.csv")
    for filename in latest_files:
        source = run_report_root / filename
        destination = report_root / filename
        if source.exists():
            atomic_write_text(destination, source.read_text(encoding="utf-8"))

    print("\nAudit finished.")
    print(f"Status                 : {status}")
    print(f"Source files           : {len(sources_by_path):,}")
    print(f"Parquet files          : {len(shard_audits):,}")
    print(f"Records in state       : {records_written_state:,}")
    print(f"Records in Parquet     : {records_in_parquet:,}")
    print(f"Unknown records        : {unknown_total:,}")
    print(f"Malformed records      : {malformed_total:,}")
    print(f"Failure issues         : {fail_count:,}")
    print(f"Warning issues         : {warn_count:,}")
    print(f"Findings               : {run_report_root / 'canonical_audit_findings.md'}")
    print(f"Summary                : {run_report_root / 'canonical_audit_summary.json'}")

    if fail_count:
        return 1
    if warn_count and args.fail_on_warnings:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nAudit interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nFATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
