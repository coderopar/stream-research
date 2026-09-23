#!/usr/bin/env python3
"""R0.1 canonical dataset inventory and provenance builder.

This tool is intentionally outcome-blind. It is restricted to the repository's
``data/canonical`` tree and only collects file/path metadata, cryptographic hashes,
and (optionally) Parquet footer metadata such as schema and row count.

It MUST NOT be pointed at data/derived, reports, research outcomes, backtests or
strategy-performance artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
D0_START = date(2026, 5, 18)
D0_END = date(2026, 7, 18)
P1_DECLARED_START = date(2026, 7, 19)
DATE_RE = re.compile(r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)")
PARTITION_RE = re.compile(r"^date=(20\d{2}-\d{2}-\d{2})$")
HASH_CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class FileRecord:
    relative_path: str
    feed_key: str
    record_class: str
    file_date_utc: str
    cohort: str
    suffix: str
    size_bytes: int
    mtime_utc: str
    sha256: str
    parquet_row_count: int | None
    parquet_schema_hash: str
    parquet_metadata_status: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_file_date(rel: Path) -> date | None:
    # Prefer an explicit Hive-style date partition because it is least ambiguous.
    for part in rel.parts:
        m = PARTITION_RE.match(part)
        if m:
            try:
                return date.fromisoformat(m.group(1))
            except ValueError:
                return None

    # Fall back to a date token in the filename/path.
    for part in reversed(rel.parts):
        m = DATE_RE.search(part)
        if m:
            try:
                return date.fromisoformat(m.group(1))
            except ValueError:
                continue
    return None


def classify_cohort(d: date | None) -> str:
    if d is None:
        return "UNDATED"
    if d < D0_START:
        return "PRE_D0"
    if D0_START <= d <= D0_END:
        return "D0"
    if d >= P1_DECLARED_START:
        return "P1_PROTECTED"
    return "UNCLASSIFIED"




def classify_record(rel: Path, d: date | None) -> tuple[str, str]:
    """Classify canonical files without forcing non-event metadata into a cohort.

    ``_extraction_state/**`` is provenance metadata for canonicalization itself. It is
    part of the canonical store and is therefore inventoried and hashed, but it is not
    a dated market-event observation and must not be treated as an UNDATED cohort.

    Any other undated file remains ``UNKNOWN_UNDATED`` and blocks R0.1 closure until
    reviewed.
    """
    if rel.parts and rel.parts[0] == "_extraction_state":
        return "CANONICAL_METADATA", "NOT_APPLICABLE"
    if d is None:
        return "UNKNOWN_UNDATED", "UNDATED"
    return "MARKET_EVENT_FILE", classify_cohort(d)


def feed_key_for(rel: Path) -> str:
    parts: list[str] = []
    for part in rel.parts[:-1]:
        if PARTITION_RE.match(part):
            break
        if part.startswith("date="):
            break
        parts.append(part)
    if parts:
        return "/".join(parts)
    return rel.parent.as_posix() if rel.parent.as_posix() != "." else "ROOT"


def parquet_metadata(path: Path, enabled: bool) -> tuple[int | None, str, str, dict[str, Any] | None]:
    if not enabled or path.suffix.lower() != ".parquet":
        return None, "", "NOT_REQUESTED", None
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        return None, "", "PYARROW_UNAVAILABLE", None

    try:
        pf = pq.ParquetFile(path)
        schema = [(field.name, str(field.type), bool(field.nullable)) for field in pf.schema_arrow]
        schema_json = json.dumps(schema, separators=(",", ":"), ensure_ascii=False)
        schema_hash = sha256_text(schema_json)
        meta = {
            "schema_hash": schema_hash,
            "columns": [
                {"name": name, "type": typ, "nullable": nullable}
                for name, typ, nullable in schema
            ],
        }
        return int(pf.metadata.num_rows), schema_hash, "OK", meta
    except Exception as exc:  # provenance scan should report, not hide, malformed files
        return None, "", f"ERROR:{type(exc).__name__}", None


def git_provenance(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            cp = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return cp.stdout.strip()
        except Exception:
            return None

    head = run("rev-parse", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    status = run("status", "--porcelain")
    return {
        "git_head": head,
        "git_branch": branch,
        "git_worktree_dirty": None if status is None else bool(status),
    }


def validate_roots(repo_root: Path, canonical_root: Path, output_root: Path) -> Path:
    """Validate the logical canonical entry-point and return its physical root.

    The repository is allowed to expose canonical data through a symlink located
    under ``<repo>/data/canonical``. This is the intended production topology for
    ``stream-research``: the lightweight repository path points at a large external
    canonical store.

    Security is enforced in two layers:
    1. the *logical* path supplied to the scanner must live under
       ``<repo>/data/canonical``;
    2. after resolving that approved entry-point, nested symlinks inside the
       physical canonical tree are rejected by :func:`enumerate_files`.

    This allows the approved root symlink without turning ``--canonical-root`` into
    a general-purpose external filesystem scanner.
    """
    repo_root = repo_root.resolve()
    canonical_logical = canonical_root.absolute()
    output_root = output_root.resolve()
    canonical_parent_logical = (repo_root / "data" / "canonical").absolute()

    try:
        canonical_logical.relative_to(canonical_parent_logical)
    except ValueError as exc:
        raise SystemExit(
            "REFUSED: --canonical-root must be addressed through "
            "<repo>/data/canonical. R0.1 is not allowed to scan arbitrary "
            "external, derived, outcome, or report trees."
        ) from exc

    try:
        canonical_physical = canonical_logical.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise SystemExit(
            f"Canonical root does not resolve to an existing directory: {canonical_logical}"
        ) from exc

    if not canonical_physical.is_dir():
        raise SystemExit(
            f"Canonical root does not resolve to a directory: {canonical_logical} -> {canonical_physical}"
        )

    # Output must not be under either the logical entry-point or its physical target;
    # provenance artifacts belong in research/governance.
    for forbidden_root in (canonical_logical, canonical_physical):
        try:
            output_root.relative_to(forbidden_root)
        except ValueError:
            continue
        raise SystemExit("REFUSED: output root cannot be inside the canonical data tree.")

    return canonical_physical


def enumerate_files(canonical_root: Path) -> list[Path]:
    """Enumerate the resolved canonical tree without following nested symlinks.

    The canonical *root* may itself have been reached through the approved repository
    symlink. Once inside that physical root, additional symlinks are rejected so a
    malformed partition cannot escape the governed canonical store.
    """
    root_resolved = canonical_root.resolve(strict=True)
    files: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(root_resolved, followlinks=False):
        base = Path(dirpath)

        # Reject nested directory symlinks rather than silently following or skipping
        # them; an R0 provenance scan should make unexpected topology explicit.
        for dirname in list(dirnames):
            child = base / dirname
            if child.is_symlink():
                raise RuntimeError(f"Nested symlink is not allowed in canonical tree: {child} -> {child.resolve()}")

        for filename in filenames:
            p = base / filename
            if p.is_symlink():
                raise RuntimeError(f"Nested symlink is not allowed in canonical tree: {p} -> {p.resolve()}")
            files.append(p)

    return sorted(files)


def write_csv_atomic(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp, path)


def cohort_manifest(
    cohort: str,
    records: list[FileRecord],
    generated_at: str,
    canonical_root_rel: str,
    git_info: dict[str, Any],
    file_index_sha256: str,
) -> dict[str, Any]:
    dated = [date.fromisoformat(r.file_date_utc) for r in records if r.file_date_utc]
    feeds = sorted({r.feed_key for r in records})
    schema_hashes = sorted({r.parquet_schema_hash for r in records if r.parquet_schema_hash})
    return {
        "schema_version": SCHEMA_VERSION,
        "cohort_id": cohort,
        "generated_at_utc": generated_at,
        "canonical_root": canonical_root_rel,
        "file_count": len(records),
        "total_bytes": sum(r.size_bytes for r in records),
        "observed_first_date_utc": min(dated).isoformat() if dated else None,
        "observed_last_date_utc": max(dated).isoformat() if dated else None,
        "feeds": feeds,
        "unique_parquet_schema_hashes": schema_hashes,
        "canonical_file_index_sha256": file_index_sha256,
        "hash_algorithm": "sha256",
        "provenance": git_info,
    }


def build(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    canonical_logical = args.canonical_root.absolute()
    output_root = args.output_root.resolve()
    canonical_root = validate_roots(repo_root, canonical_logical, output_root)

    generated_at = utc_now_iso()
    paths = enumerate_files(canonical_root)
    if not paths:
        raise SystemExit(f"No canonical files found under: {canonical_root}")

    records: list[FileRecord] = []
    schemas: dict[str, dict[str, Any]] = {}

    for i, path in enumerate(paths, start=1):
        rel = path.relative_to(canonical_root)
        d = parse_file_date(rel)
        row_count, schema_hash, schema_status, schema_meta = parquet_metadata(path, args.inspect_parquet_metadata)
        digest = sha256_file(path) if args.hash_mode == "full" else ""
        st = path.stat()
        record_class, cohort = classify_record(rel, d)
        record = FileRecord(
            relative_path=rel.as_posix(),
            feed_key=feed_key_for(rel),
            record_class=record_class,
            file_date_utc=d.isoformat() if d else "",
            cohort=cohort,
            suffix=path.suffix.lower(),
            size_bytes=int(st.st_size),
            mtime_utc=datetime.fromtimestamp(st.st_mtime, timezone.utc).replace(microsecond=0).isoformat(),
            sha256=digest,
            parquet_row_count=row_count,
            parquet_schema_hash=schema_hash,
            parquet_metadata_status=schema_status,
        )
        records.append(record)
        if schema_hash and schema_meta:
            schemas.setdefault(schema_hash, schema_meta)
        if args.progress_every and i % args.progress_every == 0:
            print(f"Inventoried {i}/{len(paths)} files", file=sys.stderr)

    file_index_path = output_root / "CANONICAL_FILE_INDEX_v1.csv"
    fields = list(asdict(records[0]).keys())
    write_csv_atomic(file_index_path, (asdict(r) for r in records), fields)
    file_index_sha = sha256_file(file_index_path)

    # Feed/date coverage is deliberately metadata-only; no feature or outcome values are read.
    coverage: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in records:
        if r.record_class != "MARKET_EVENT_FILE":
            continue
        key = (r.feed_key, r.cohort, r.file_date_utc)
        c = coverage.setdefault(
            key,
            {
                "feed_key": r.feed_key,
                "cohort": r.cohort,
                "date_utc": r.file_date_utc,
                "file_count": 0,
                "total_bytes": 0,
                "parquet_rows_from_metadata": 0,
                "parquet_files_with_row_metadata": 0,
            },
        )
        c["file_count"] += 1
        c["total_bytes"] += r.size_bytes
        if r.parquet_row_count is not None:
            c["parquet_rows_from_metadata"] += r.parquet_row_count
            c["parquet_files_with_row_metadata"] += 1

    coverage_rows = [coverage[k] for k in sorted(coverage)]
    coverage_path = output_root / "FEED_COVERAGE_v1.csv"
    coverage_fields = [
        "feed_key",
        "cohort",
        "date_utc",
        "file_count",
        "total_bytes",
        "parquet_rows_from_metadata",
        "parquet_files_with_row_metadata",
    ]
    write_csv_atomic(coverage_path, coverage_rows, coverage_fields)

    schema_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "inspection_mode": "PARQUET_FOOTER_ONLY" if args.inspect_parquet_metadata else "DISABLED",
        "schemas": schemas,
        "files_with_schema": sum(1 for r in records if r.parquet_schema_hash),
        "parquet_metadata_errors": [
            {"relative_path": r.relative_path, "status": r.parquet_metadata_status}
            for r in records
            if r.suffix == ".parquet" and r.parquet_metadata_status.startswith("ERROR:")
        ],
    }
    atomic_write_json(output_root / "CANONICAL_SCHEMA_INVENTORY_v1.json", schema_payload)

    git_info = git_provenance(repo_root)
    canonical_root_rel = canonical_logical.relative_to(repo_root).as_posix()
    try:
        canonical_root.relative_to(repo_root)
        external_storage = False
    except ValueError:
        external_storage = True

    logical_and_physical_roots_differ = (
        os.path.normcase(os.path.normpath(str(canonical_logical)))
        != os.path.normcase(os.path.normpath(str(canonical_root)))
    )
    os_reported_symlink = canonical_logical.is_symlink()
    canonical_topology = {
        "logical_root": canonical_root_rel,
        "resolved_physical_root": str(canonical_root),
        "logical_and_physical_roots_differ": logical_and_physical_roots_differ,
        "external_storage": external_storage,
        "link_detection": "resolved_path_difference" if logical_and_physical_roots_differ else "same_path",
        "os_reported_is_symlink": os_reported_symlink,
        "os_reported_symlink_target": os.readlink(canonical_logical) if os_reported_symlink else None,
        "note": (
            "On Windows/MSYS, pathlib may report is_symlink=false for a repository redirect. "
            "The authoritative topology indicator is logical_and_physical_roots_differ."
        ),
    }
    by_cohort: dict[str, list[FileRecord]] = defaultdict(list)
    for r in records:
        by_cohort[r.cohort].append(r)

    d0_manifest = cohort_manifest(
        "D0",
        by_cohort.get("D0", []),
        generated_at,
        canonical_root_rel,
        git_info,
        file_index_sha,
    )
    d0_manifest.update(
        {
            "canonical_topology": canonical_topology,
            "declared_start_date_utc": D0_START.isoformat(),
            "declared_end_date_utc": D0_END.isoformat(),
            "role": "DEVELOPMENT",
            "outcome_status": "SPENT",
        }
    )
    atomic_write_json(output_root / "D0_DATASET_MANIFEST_v1.json", d0_manifest)

    p1_records = by_cohort.get("P1_PROTECTED", [])
    p1_manifest = cohort_manifest(
        "P1_PROTECTED",
        p1_records,
        generated_at,
        canonical_root_rel,
        git_info,
        file_index_sha,
    )
    p1_manifest.update(
        {
            "canonical_topology": canonical_topology,
            "declared_start_date_utc": P1_DECLARED_START.isoformat(),
            "role": "PROTECTED_HISTORICAL_POOL",
            "outcome_status": "UNSEEN",
            "confirmation_status": "NOT_YET_D1",
            "allowed_information": [
                "file paths",
                "cryptographic hashes",
                "schema metadata",
                "row-count metadata",
                "date/feed coverage",
            ],
            "prohibited_information": [
                "future returns",
                "P35/P50/P75/P100",
                "MFE/MAE",
                "strategy P&L",
                "signal performance",
            ],
        }
    )
    atomic_write_json(output_root / "P1_PROTECTED_MANIFEST_v1.json", p1_manifest)

    undated_market_event_files = [
        r for r in records if r.record_class == "UNKNOWN_UNDATED"
    ]
    canonical_metadata_files = [
        r for r in records if r.record_class == "CANONICAL_METADATA"
    ]
    p1_dates = sorted(date.fromisoformat(r.file_date_utc) for r in p1_records if r.file_date_utc)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "status": (
            "PASS"
            if p1_records and args.hash_mode == "full" and not undated_market_event_files
            else "INCOMPLETE"
        ),
        "repo_root": str(repo_root),
        "canonical_root": canonical_root_rel,
        "canonical_topology": canonical_topology,
        "hash_mode": args.hash_mode,
        "file_count": len(records),
        "total_bytes": sum(r.size_bytes for r in records),
        "cohort_file_counts": {
            k: len(v)
            for k, v in sorted(by_cohort.items())
            if k not in {"NOT_APPLICABLE"}
        },
        "canonical_metadata_file_count": len(canonical_metadata_files),
        "unknown_undated_market_event_file_count": len(undated_market_event_files),
        "p1_declared_start_date_utc": P1_DECLARED_START.isoformat(),
        "p1_observed_first_date_utc": p1_dates[0].isoformat() if p1_dates else None,
        "p1_observed_last_date_utc": p1_dates[-1].isoformat() if p1_dates else None,
        "undated_file_count": len(undated_market_event_files),
        "file_index_sha256": file_index_sha,
        "r0_1_gate_checks": {
            "canonical_root_exists": canonical_root.exists(),
            "canonical_file_index_generated": file_index_path.exists(),
            "canonical_files_hashed": args.hash_mode == "full" and all(r.sha256 for r in records),
            "D0_manifest_generated": True,
            "P1_manifest_generated": True,
            "feed_coverage_inventory_generated": True,
            "schema_inventory_generated_or_explicitly_unavailable": True,
            "P1_end_date_discovered_from_canonical_partitions": bool(p1_dates),
            "no_noncanonical_paths_scanned": True,
            "no_unreviewed_undated_market_event_files": not undated_market_event_files,
            "canonical_metadata_classified_separately": True,
        },
        "notes": [
            "The inventory reads file bytes only for SHA-256 hashing.",
            "Parquet inspection, when enabled, reads footer metadata only; no market-event values or outcome data are selected.",
            "Canonical extraction-state files are classified as CANONICAL_METADATA and are not assigned to D0/P1 cohorts.",
            "Any other undated canonical file is classified UNKNOWN_UNDATED and blocks R0.1 closure until reviewed.",
        ],
    }
    atomic_write_json(output_root / "R0_1_INVENTORY_SUMMARY_v1.json", summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    script_repo = Path(__file__).resolve().parents[3]
    p.add_argument("--repo-root", type=Path, default=script_repo)
    p.add_argument(
        "--canonical-root",
        type=Path,
        default=script_repo / "data" / "canonical" / "market_events_v1",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=script_repo / "research" / "governance" / "datasets",
    )
    p.add_argument(
        "--hash-mode",
        choices=("full", "none"),
        default="full",
        help="Use 'full' for the R0.1 acceptance run. 'none' is only a quick dry run.",
    )
    p.add_argument(
        "--inspect-parquet-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read only Parquet footer schema/row-count metadata; never row values.",
    )
    p.add_argument("--progress-every", type=int, default=100)
    return p


if __name__ == "__main__":
    raise SystemExit(build(parser().parse_args()))
