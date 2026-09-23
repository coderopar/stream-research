#!/usr/bin/env python3
"""
Orchestrate safe Stream bronze backlog processing in verified batches.

This wrapper deliberately leaves the proven processing scripts untouched:
  - scripts/00_download_bronze_batch.py
  - scripts/01_extract_all_market_events.py
  - scripts/02_audit_canonical_market_events.py

Per batch it performs:
  1. Download next GCS batch through 00_download_bronze_batch.py.
  2. Canonicalize only that source-date window through 01_extract_all_market_events.py.
  3. Verify every manifest WAL against extraction_state.sqlite and physical outputs.
  4. Recompute the extractor's quick fingerprint for every local WAL.
  5. Run the whole canonical auditor with raw inventory skipped.
  6. Re-list GCS and verify every manifest WAL still exists with the same byte size.
  7. Delete only manifest-listed local WALs, after one final local size check.

Fail-closed behavior:
  - Any failed command or verification stops the run immediately.
  - On failure, local raw WALs are retained.
  - Cleanup never targets the whole bronze directory; it targets only paths from
    the successful downloader result manifest.

Default behavior is intentionally conservative: process at most one batch.
Pass --max-batches 0 only after validating the first orchestrated batch; 0 means
continue until no closed-date backlog remains.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_VERSION = "1.1.0"

DEFAULT_REPO = Path(r"C:\Projects\stream-research")
DEFAULT_RAW_ROOT = Path(r"C:\Users\USER\stream-bronze-download\data\bronze")
DEFAULT_MANIFEST_DIR = Path(r"C:\Users\USER\stream-bronze-download\list")
DEFAULT_CANONICAL_ROOT = Path(r"D:\stream-data\canonical\market_events_v1")
DEFAULT_GCS_ROOT = "gs://steve-silas-bucket/bronze"

RESULT_GLOB = "bronze-batch-*-result.json"
LONG_LS_RE = re.compile(r"^\s*(\d+)\s+\S+\s+(gs://\S+)\s*$")


@dataclass
class BatchSummary:
    batch_number: int
    manifest: str
    batch_start: str
    batch_end: str
    wal_count: int
    expected_bytes: int
    records_read: int
    records_written: int
    deleted_files: int
    deleted_bytes: int
    status: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def format_bytes(value: int) -> str:
    if value >= 1024 ** 3:
        return f"{value / (1024 ** 3):.2f} GiB"
    if value >= 1024 ** 2:
        return f"{value / (1024 ** 2):.2f} MiB"
    if value >= 1024:
        return f"{value / 1024:.2f} KiB"
    return f"{value} B"


def fail(message: str, code: int = 2) -> "NoReturn":
    raise SystemExit(f"FAIL: {message}")


def locate_gcloud(explicit: str | None) -> str:
    if explicit:
        candidate = Path(explicit)
        if candidate.exists():
            return str(candidate)
        resolved = shutil.which(explicit)
        if resolved:
            return resolved
        fail(f"Could not locate gcloud executable: {explicit}")

    for name in ("gcloud.cmd", "gcloud"):
        resolved = shutil.which(name)
        if resolved:
            return resolved

    fail("Could not locate gcloud/gcloud.cmd on PATH. Pass --gcloud explicitly.")


def stream_command(cmd: list[str], cwd: Path, label: str) -> tuple[int, str]:
    print()
    print(f"===== {label} =====")
    print("COMMAND:", subprocess.list2cmdline(cmd))
    print()

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)

    return_code = process.wait()
    output = "".join(lines)
    print()
    print(f"{label} exit code: {return_code}")
    return return_code, output


def validate_repo(repo: Path) -> tuple[Path, Path, Path]:
    downloader = repo / "scripts" / "00_download_bronze_batch.py"
    extractor = repo / "scripts" / "01_extract_all_market_events.py"
    auditor = repo / "scripts" / "02_audit_canonical_market_events.py"

    missing = [str(p) for p in (downloader, extractor, auditor) if not p.exists()]
    if missing:
        fail("Required script(s) missing:\n  " + "\n  ".join(missing))

    return downloader, extractor, auditor


def result_manifests(manifest_dir: Path) -> set[Path]:
    if not manifest_dir.exists():
        return set()
    return {p.resolve() for p in manifest_dir.glob(RESULT_GLOB)}


def newest_new_manifest(before: set[Path], manifest_dir: Path) -> Path | None:
    after = result_manifests(manifest_dir)
    new = list(after - before)
    if not new:
        return None
    new.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
    return new[0]


def load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"Could not read downloader result manifest {path}: {exc}")

    rows = payload.get("results")
    if not isinstance(rows, list) or not rows:
        fail(f"Downloader result manifest has no results: {path}")

    required_top = ("batch_start", "batch_end", "file_count", "expected_bytes")
    for name in required_top:
        if name not in payload:
            fail(f"Downloader result manifest missing '{name}': {path}")

    if int(payload["file_count"]) != len(rows):
        fail(
            f"Manifest file_count={payload['file_count']} but results has {len(rows)} rows"
        )

    seen: set[str] = set()
    for item in rows:
        for key in ("relative_path", "expected_bytes", "status"):
            if key not in item:
                fail(f"Manifest row missing '{key}': {item}")
        rel = str(item["relative_path"])
        if rel in seen:
            fail(f"Duplicate relative_path in manifest: {rel}")
        seen.add(rel)
        if str(item["status"]).startswith("FAILED"):
            fail(f"Downloader manifest contains failed row: {rel} [{item['status']}]")

    return payload, rows


def local_path(raw_root: Path, relative_path: str) -> Path:
    parts = relative_path.replace("\\", "/").split("/")
    if any(part in ("", ".", "..") for part in parts):
        fail(f"Unsafe relative path in manifest: {relative_path}")
    path = raw_root.joinpath(*parts).resolve()
    root = raw_root.resolve()
    try:
        path.relative_to(root)
    except ValueError:
        fail(f"Manifest path escapes raw root: {relative_path}")
    return path


def verify_local_manifest(rows: list[dict[str, Any]], raw_root: Path) -> None:
    """Fail closed unless every manifest WAL exists locally at the expected size."""
    problems: list[str] = []
    verified = 0

    for item in rows:
        rel = str(item["relative_path"])
        expected = int(item["expected_bytes"])
        path = local_path(raw_root, rel)

        if not path.exists():
            problems.append(f"{rel}: RAW_FILE_MISSING")
            continue

        actual = path.stat().st_size
        if actual != expected:
            problems.append(f"{rel}: LOCAL_SIZE={actual} EXPECTED={expected}")
            continue

        verified += 1

    print()
    print("===== RESUME LOCAL MANIFEST PREFLIGHT =====")
    print(f"Manifest WALs : {len(rows):,}")
    print(f"Verified      : {verified:,}")
    print(f"Problems      : {len(problems):,}")

    if problems:
        for problem in problems[:50]:
            print("FAIL", problem)
        if len(problems) > 50:
            print(f"... plus {len(problems) - 50:,} more")
        fail("Resume manifest local preflight failed; no extraction or cleanup was run")

    print("PASS: every resume-manifest WAL exists locally with the expected byte size.")


def load_extractor_module(extractor_path: Path):
    spec = importlib.util.spec_from_file_location("stream_extractor_for_backlog", extractor_path)
    if spec is None or spec.loader is None:
        fail(f"Could not load extractor module: {extractor_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "quick_fingerprint"):
        fail("Extractor module has no quick_fingerprint function")
    return module


def verify_batch_ledger_and_fingerprints(
    rows: list[dict[str, Any]],
    raw_root: Path,
    canonical_root: Path,
    extractor_path: Path,
) -> tuple[int, int]:
    db = canonical_root / "_extraction_state" / "extraction_state.sqlite"
    if not db.exists():
        fail(f"Extraction state DB does not exist: {db}")

    extractor = load_extractor_module(extractor_path)
    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row

    problems: list[str] = []
    total_read = 0
    total_written = 0
    verified = 0

    try:
        for item in rows:
            rel = str(item["relative_path"])
            expected_size = int(item["expected_bytes"])
            path = local_path(raw_root, rel)

            record = con.execute(
                """
                SELECT *
                FROM source_files
                WHERE source_relative_path = ?
                """,
                (rel,),
            ).fetchone()

            if record is None:
                problems.append(f"{rel}: MISSING_FROM_LEDGER")
                continue

            cols = set(record.keys())

            def value(name: str, default: Any = None) -> Any:
                return record[name] if name in cols else default

            if value("status") != "COMPLETED":
                problems.append(f"{rel}: STATUS={value('status')}")

            if int(value("size_bytes", -1)) != expected_size:
                problems.append(
                    f"{rel}: LEDGER_SIZE={value('size_bytes')} EXPECTED={expected_size}"
                )

            records_read = int(value("records_read", 0) or 0)
            records_written = int(value("records_written", 0) or 0)
            total_read += records_read
            total_written += records_written

            if records_read != records_written:
                problems.append(
                    f"{rel}: READ_WRITE_MISMATCH={records_read}!={records_written}"
                )

            if int(value("unknown_records", 0) or 0) != 0:
                problems.append(f"{rel}: UNKNOWN={value('unknown_records')}")

            if int(value("malformed_records", 0) or 0) != 0:
                problems.append(f"{rel}: MALFORMED={value('malformed_records')}")

            output_paths_json = value("output_paths_json")
            if not output_paths_json:
                problems.append(f"{rel}: NO_OUTPUT_PATHS")
            else:
                try:
                    output_paths = json.loads(output_paths_json)
                except Exception:
                    problems.append(f"{rel}: INVALID_OUTPUT_PATHS_JSON")
                    output_paths = []

                for output_rel in output_paths:
                    output_path = canonical_root / output_rel
                    if not output_path.exists():
                        problems.append(
                            f"{rel}: MISSING_CANONICAL_OUTPUT={output_rel}"
                        )

            if not path.exists():
                problems.append(f"{rel}: RAW_FILE_MISSING")
                continue

            local_size = path.stat().st_size
            if local_size != expected_size:
                problems.append(
                    f"{rel}: LOCAL_SIZE={local_size} EXPECTED={expected_size}"
                )
                continue

            if value("fingerprint_mode") != "quick":
                problems.append(
                    f"{rel}: UNEXPECTED_FINGERPRINT_MODE={value('fingerprint_mode')}"
                )
                continue

            actual_fingerprint = extractor.quick_fingerprint(path)
            if actual_fingerprint != value("fingerprint"):
                problems.append(
                    f"{rel}: FINGERPRINT_MISMATCH ledger={value('fingerprint')} "
                    f"actual={actual_fingerprint}"
                )
                continue

            verified += 1
    finally:
        con.close()

    print()
    print("===== BATCH LEDGER + FINGERPRINT VERIFICATION =====")
    print(f"Manifest WALs   : {len(rows):,}")
    print(f"Verified WALs   : {verified:,}")
    print(f"Records read    : {total_read:,}")
    print(f"Records written : {total_written:,}")
    print(f"Problems        : {len(problems):,}")

    if problems:
        for problem in problems[:50]:
            print("FAIL", problem)
        if len(problems) > 50:
            print(f"... plus {len(problems) - 50:,} more")
        fail("Batch ledger/fingerprint verification failed")

    if verified != len(rows):
        fail(f"Verified {verified} of {len(rows)} manifest WALs")

    print("PASS: batch ledger, outputs, sizes, and quick fingerprints all match.")
    return total_read, total_written


def fetch_gcs_inventory(gcloud: str, gcs_root: str, timeout_seconds: int) -> dict[str, int]:
    cmd = [gcloud, "storage", "ls", "--long", gcs_root.rstrip("/") + "/**"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        fail(f"GCS inventory timed out after {timeout_seconds}s")

    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        fail(f"GCS inventory failed with exit code {result.returncode}: {message}")

    inventory: dict[str, int] = {}
    prefix = gcs_root.rstrip("/") + "/"

    for line in result.stdout.splitlines():
        match = LONG_LS_RE.match(line)
        if not match:
            continue
        size = int(match.group(1))
        uri = match.group(2)
        if not uri.startswith(prefix):
            continue
        relative = uri[len(prefix):]
        inventory[relative] = size

    return inventory


def verify_gcs(rows: list[dict[str, Any]], inventory: dict[str, int]) -> None:
    problems: list[str] = []
    verified = 0

    for item in rows:
        rel = str(item["relative_path"])
        expected = int(item["expected_bytes"])
        actual = inventory.get(rel)

        if actual is None:
            problems.append(f"{rel}: MISSING_FROM_GCS")
            continue
        if actual != expected:
            problems.append(f"{rel}: GCS_SIZE={actual} EXPECTED={expected}")
            continue
        verified += 1

    print()
    print("===== FINAL GCS VERIFICATION =====")
    print(f"Manifest WALs : {len(rows):,}")
    print(f"Verified      : {verified:,}")
    print(f"Problems      : {len(problems):,}")

    if problems:
        for problem in problems[:50]:
            print("FAIL", problem)
        if len(problems) > 50:
            print(f"... plus {len(problems) - 50:,} more")
        fail("Final GCS verification failed")

    print("PASS: every manifest WAL still exists in GCS with the expected byte size.")


def cleanup_batch(rows: list[dict[str, Any]], raw_root: Path) -> tuple[int, int]:
    # Preflight every path first. Nothing is deleted until the whole manifest is safe.
    problems: list[str] = []
    paths: list[tuple[Path, int, str]] = []

    for item in rows:
        rel = str(item["relative_path"])
        expected = int(item["expected_bytes"])
        path = local_path(raw_root, rel)

        if not path.exists():
            problems.append(f"{rel}: RAW_FILE_MISSING_BEFORE_CLEANUP")
            continue
        actual = path.stat().st_size
        if actual != expected:
            problems.append(
                f"{rel}: SIZE_CHANGED_BEFORE_CLEANUP local={actual} expected={expected}"
            )
            continue
        paths.append((path, actual, rel))

    if problems:
        for problem in problems[:50]:
            print("FAIL", problem)
        fail("Cleanup preflight failed; no batch files were deleted")

    deleted = 0
    deleted_bytes = 0
    for path, size, _rel in paths:
        path.unlink()
        deleted += 1
        deleted_bytes += size

    # Remove only empty descendants; never remove the configured raw root itself.
    if raw_root.exists():
        dirs = [p for p in raw_root.rglob("*") if p.is_dir()]
        for path in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
            try:
                path.rmdir()
            except OSError:
                pass

    print()
    print("===== MANIFEST-SCOPED CLEANUP =====")
    print(f"Deleted WALs  : {deleted:,}")
    print(f"Deleted bytes : {format_bytes(deleted_bytes)}")
    print("PASS: only manifest-listed local raw WALs were deleted.")
    return deleted, deleted_bytes


def write_run_summary(report_root: Path, payload: dict[str, Any]) -> Path:
    report_root.mkdir(parents=True, exist_ok=True)
    path = report_root / "backlog_run_summary.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely process Stream bronze GCS backlog in verified batches."
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--gcs-root", default=DEFAULT_GCS_ROOT)
    parser.add_argument("--gcloud")
    parser.add_argument("--batch-days", type=int, default=7)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=1,
        help="Maximum batches this run. Default 1. Use 0 for all available backlog.",
    )
    parser.add_argument(
        "--download-parallel",
        type=int,
        default=4,
        help="Parallel downloads passed to 00_download_bronze_batch.py.",
    )
    parser.add_argument(
        "--download-max-attempts",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--download-attempt-timeout-minutes",
        type=float,
        default=20.0,
    )
    parser.add_argument("--raw-min-free-space-gb", type=float, default=20.0)
    parser.add_argument("--canonical-min-free-space-gb", type=float, default=30.0)
    parser.add_argument("--auditor-workers", type=int, default=4)
    parser.add_argument("--auditor-progress-every", type=int, default=500)
    parser.add_argument("--gcs-inventory-timeout-minutes", type=float, default=10.0)
    parser.add_argument(
        "--resume-manifest",
        type=Path,
        help=(
            "Resume a previously downloaded batch from its successful result JSON. "
            "The downloader is skipped for the first batch; every manifest WAL must "
            "still exist locally at the expected size. Completed extraction states "
            "are skipped by the extractor, failed/unprocessed WALs are retried, and "
            "cleanup targets the full original manifest only after all checks pass."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the next download batch only; no download/extraction/audit/cleanup.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Run all verification stages but retain successfully processed raw WALs.",
    )

    args = parser.parse_args()

    if args.batch_days < 1:
        parser.error("--batch-days must be at least 1")
    if args.max_batches < 0:
        parser.error("--max-batches cannot be negative")
    if args.download_parallel < 1:
        parser.error("--download-parallel must be at least 1")
    if args.download_max_attempts < 1:
        parser.error("--download-max-attempts must be at least 1")
    if args.download_attempt_timeout_minutes <= 0:
        parser.error("--download-attempt-timeout-minutes must be > 0")
    if args.raw_min_free_space_gb < 0 or args.canonical_min_free_space_gb < 0:
        parser.error("free-space guards cannot be negative")
    if args.auditor_workers < 1:
        parser.error("--auditor-workers must be at least 1")
    if args.gcs_inventory_timeout_minutes <= 0:
        parser.error("--gcs-inventory-timeout-minutes must be > 0")
    if args.dry_run and args.resume_manifest is not None:
        parser.error("--resume-manifest cannot be combined with --dry-run")

    return args


def main() -> int:
    args = parse_args()

    repo = args.repo.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    manifest_dir = args.manifest_dir.expanduser().resolve()
    canonical_root = args.canonical_root.expanduser().resolve()
    downloader, extractor, auditor = validate_repo(repo)
    gcloud = locate_gcloud(args.gcloud)

    state_db = canonical_root / "_extraction_state" / "extraction_state.sqlite"
    if not state_db.exists():
        fail(f"Canonical extraction state DB not found: {state_db}")

    resume_manifest = (
        args.resume_manifest.expanduser().resolve()
        if args.resume_manifest is not None
        else None
    )
    if resume_manifest is not None and not resume_manifest.exists():
        fail(f"Resume manifest does not exist: {resume_manifest}")

    report_root = repo / "reports" / "backlog_runs" / f"{utc_stamp()}-backlog"
    report_root.mkdir(parents=True, exist_ok=True)

    print("Stream bronze backlog orchestrator")
    print("----------------------------------")
    print(f"Version                    : {SCRIPT_VERSION}")
    print(f"Repo                       : {repo}")
    print(f"Raw root                   : {raw_root}")
    print(f"Manifest dir               : {manifest_dir}")
    print(f"Canonical root             : {canonical_root}")
    print(f"GCS root                   : {args.gcs_root}")
    print(f"Batch days                 : {args.batch_days}")
    print(f"Max batches                : {'ALL' if args.max_batches == 0 else args.max_batches}")
    print(f"Keep raw                   : {args.keep_raw}")
    print(f"Dry run                    : {args.dry_run}")
    print(f"Resume manifest            : {resume_manifest or 'None'}")
    print(f"Report root                : {report_root}")

    python_exe = sys.executable

    if args.dry_run:
        cmd = [
            python_exe,
            str(downloader),
            "--gcs-root", args.gcs_root,
            "--raw-root", str(raw_root),
            "--state-db", str(state_db),
            "--manifest-dir", str(manifest_dir),
            "--batch-days", str(args.batch_days),
            "--parallel", str(args.download_parallel),
            "--max-attempts", str(args.download_max_attempts),
            "--attempt-timeout-minutes", str(args.download_attempt_timeout_minutes),
            "--min-free-space-gb", str(args.raw_min_free_space_gb),
            "--gcloud", gcloud,
            "--dry-run",
        ]
        code, _ = stream_command(cmd, repo, "DOWNLOADER DRY RUN")
        return code

    batches: list[BatchSummary] = []
    batch_number = 0
    overall_status = "RUNNING"
    started_at = utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        while args.max_batches == 0 or batch_number < args.max_batches:
            batch_number += 1
            print()
            print("#" * 80)
            print(f"BATCH {batch_number}")
            print("#" * 80)

            if resume_manifest is not None:
                manifest = resume_manifest
                payload, rows = load_manifest(manifest)
                batch_start = str(payload["batch_start"])
                batch_end = str(payload["batch_end"])
                wal_count = len(rows)
                expected_bytes = sum(int(item["expected_bytes"]) for item in rows)

                print()
                print(f"===== BATCH {batch_number} RESUME MANIFEST =====")
                print(f"Manifest       : {manifest}")
                print(f"Batch dates    : {batch_start} -> {batch_end}")
                print(f"WALs           : {wal_count:,}")
                print(f"Expected bytes : {format_bytes(expected_bytes)}")
                print("Downloader     : SKIPPED (resume mode)")

                verify_local_manifest(rows, raw_root)

                # Consume the explicit resume manifest once. Subsequent batches, if
                # requested, return to normal downloader-driven discovery.
                resume_manifest = None
            else:
                before_manifests = result_manifests(manifest_dir)

                download_cmd = [
                    python_exe,
                    str(downloader),
                    "--gcs-root", args.gcs_root,
                    "--raw-root", str(raw_root),
                    "--state-db", str(state_db),
                    "--manifest-dir", str(manifest_dir),
                    "--batch-days", str(args.batch_days),
                    "--parallel", str(args.download_parallel),
                    "--max-attempts", str(args.download_max_attempts),
                    "--attempt-timeout-minutes", str(args.download_attempt_timeout_minutes),
                    "--min-free-space-gb", str(args.raw_min_free_space_gb),
                    "--gcloud", gcloud,
                ]
                code, download_output = stream_command(
                    download_cmd, repo, f"BATCH {batch_number} DOWNLOAD"
                )
                if code != 0:
                    fail(f"Downloader failed for batch {batch_number} with exit code {code}")

                manifest = newest_new_manifest(before_manifests, manifest_dir)
                if manifest is None:
                    if "No new post-frontier WAL batch is available." in download_output:
                        batch_number -= 1
                        overall_status = "CAUGHT_UP"
                        print("No closed-date backlog remains. Processing is caught up.")
                        break
                    fail(
                        "Downloader returned success but did not create a new result manifest. "
                        "Raw files are retained."
                    )

                payload, rows = load_manifest(manifest)
                batch_start = str(payload["batch_start"])
                batch_end = str(payload["batch_end"])
                wal_count = len(rows)
                expected_bytes = sum(int(item["expected_bytes"]) for item in rows)

                print()
                print("===== DOWNLOADER MANIFEST =====")
                print(f"Manifest       : {manifest}")
                print(f"Batch dates    : {batch_start} -> {batch_end}")
                print(f"WALs           : {wal_count:,}")
                print(f"Expected bytes : {format_bytes(expected_bytes)}")

            extract_cmd = [
                python_exe,
                str(extractor),
                "--input-root", str(raw_root),
                "--output-root", str(canonical_root),
                "--source-date-from", batch_start,
                "--source-date-to", batch_end,
                "--workers", "1",
                "--fingerprint-mode", "quick",
                "--compression", "zstd",
                "--compression-level", "3",
                "--include-system-events",
                "--retry-failed",
                "--min-free-space-gb", str(args.canonical_min_free_space_gb),
                "--progress-every", "250000",
            ]
            code, _ = stream_command(
                extract_cmd, repo, f"BATCH {batch_number} CANONICAL EXTRACTION"
            )
            if code != 0:
                fail(
                    f"Canonical extractor failed for batch {batch_number} "
                    f"with exit code {code}. Raw files are retained."
                )

            records_read, records_written = verify_batch_ledger_and_fingerprints(
                rows=rows,
                raw_root=raw_root,
                canonical_root=canonical_root,
                extractor_path=extractor,
            )

            audit_cmd = [
                python_exe,
                str(auditor),
                "--canonical-root", str(canonical_root),
                "--workers", str(args.auditor_workers),
                "--progress-every", str(args.auditor_progress_every),
                "--skip-raw-inventory",
                "--fail-on-warnings",
            ]
            code, _ = stream_command(
                audit_cmd, repo, f"BATCH {batch_number} WHOLE-CANONICAL AUDIT"
            )
            if code != 0:
                fail(
                    f"Canonical auditor failed for batch {batch_number} "
                    f"with exit code {code}. Raw files are retained."
                )

            print()
            print("Fetching final GCS inventory...")
            inventory = fetch_gcs_inventory(
                gcloud,
                args.gcs_root,
                int(args.gcs_inventory_timeout_minutes * 60),
            )
            verify_gcs(rows, inventory)

            if args.keep_raw:
                deleted_files = 0
                deleted_bytes = 0
                print()
                print("KEEP-RAW enabled: verified batch raw files were not deleted.")
            else:
                deleted_files, deleted_bytes = cleanup_batch(rows, raw_root)

            batches.append(
                BatchSummary(
                    batch_number=batch_number,
                    manifest=str(manifest),
                    batch_start=batch_start,
                    batch_end=batch_end,
                    wal_count=wal_count,
                    expected_bytes=expected_bytes,
                    records_read=records_read,
                    records_written=records_written,
                    deleted_files=deleted_files,
                    deleted_bytes=deleted_bytes,
                    status="COMPLETED",
                )
            )

            write_run_summary(
                report_root,
                {
                    "script_version": SCRIPT_VERSION,
                    "started_at_utc": started_at,
                    "updated_at_utc": utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "status": "RUNNING",
                    "keep_raw": args.keep_raw,
                    "batches": [asdict(item) for item in batches],
                },
            )

            print()
            print(
                f"BATCH {batch_number} PASS: {wal_count:,} WALs, "
                f"{records_written:,} canonical records"
                + ("; raw retained." if args.keep_raw else "; raw cleaned safely.")
            )

        if overall_status == "RUNNING":
            overall_status = "COMPLETED_LIMIT"

    except BaseException as exc:
        overall_status = "FAILED"
        summary_path = write_run_summary(
            report_root,
            {
                "script_version": SCRIPT_VERSION,
                "started_at_utc": started_at,
                "completed_at_utc": utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "status": overall_status,
                "keep_raw": args.keep_raw,
                "error": f"{type(exc).__name__}: {exc}",
                "batches": [asdict(item) for item in batches],
            },
        )
        print()
        print(f"Backlog run summary: {summary_path}")
        raise

    summary_path = write_run_summary(
        report_root,
        {
            "script_version": SCRIPT_VERSION,
            "started_at_utc": started_at,
            "completed_at_utc": utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": overall_status,
            "keep_raw": args.keep_raw,
            "batches": [asdict(item) for item in batches],
        },
    )

    print()
    print("=" * 80)
    print("BACKLOG RUN FINISHED")
    print("=" * 80)
    print(f"Status           : {overall_status}")
    print(f"Batches completed: {len(batches):,}")
    print(f"WALs processed   : {sum(b.wal_count for b in batches):,}")
    print(f"Records written  : {sum(b.records_written for b in batches):,}")
    print(f"Raw bytes cleaned: {format_bytes(sum(b.deleted_bytes for b in batches))}")
    print(f"Summary          : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
