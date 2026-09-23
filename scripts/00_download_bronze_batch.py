#!/usr/bin/env python3
"""Download the next unprocessed Stream bronze WAL batch from GCS safely."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sqlite3
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

SCRIPT_VERSION = "1.0.1"
DEFAULT_GCS_ROOT = "gs://steve-silas-bucket/bronze"
DEFAULT_RAW_ROOT = Path(r"C:\Users\USER\stream-bronze-download\data\bronze")
DEFAULT_STATE_DB = Path(
    r"D:\stream-data\canonical\market_events_v1\_extraction_state\extraction_state.sqlite"
)
DEFAULT_MANIFEST_DIR = Path(r"C:\Users\USER\stream-bronze-download\list")
DEFAULT_STREAMS = ("okx_spot", "options", "perps", "spot")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
LONG_LS_RE = re.compile(r"^\s*(\d+)\s+\S+\s+(gs://\S+)\s*$")
PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class GcsObject:
    relative_path: str
    gcs_uri: str
    stream: str
    source_date: date
    name: str
    size_bytes: int


@dataclass
class DownloadResult:
    relative_path: str
    gcs_uri: str
    local_path: str
    expected_bytes: int
    actual_bytes: int | None
    status: str
    attempts: int
    error: str | None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc() -> str:
    return utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def format_bytes(value: int) -> str:
    if value >= 1024 ** 3:
        return f"{value / (1024 ** 3):.2f} GiB"
    if value >= 1024 ** 2:
        return f"{value / (1024 ** 2):.2f} MiB"
    if value >= 1024:
        return f"{value / 1024:.2f} KiB"
    return f"{value} B"


def parse_date(value: str, label: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be YYYY-MM-DD") from exc


def normalize_relative_path(value: str) -> str:
    return value.replace("\\", "/").lstrip("./")


def path_date_from_relative(relative_path: str) -> tuple[str, date] | None:
    parts = normalize_relative_path(relative_path).split("/")
    if len(parts) < 3 or not DATE_RE.match(parts[1]):
        return None
    try:
        return parts[0], date.fromisoformat(parts[1])
    except ValueError:
        return None


def parse_gcs_object(gcs_root: str, uri: str, size_bytes: int, streams: set[str]) -> GcsObject | None:
    prefix = gcs_root.rstrip("/") + "/"
    if not uri.startswith(prefix):
        return None
    relative = uri[len(prefix):]
    parts = relative.split("/")
    if len(parts) != 3:
        return None
    stream, date_text, name = parts
    if stream not in streams or not DATE_RE.match(date_text):
        return None
    if not name.startswith("wal-") or not name.endswith(".zst"):
        return None
    try:
        source_date = date.fromisoformat(date_text)
    except ValueError:
        return None
    return GcsObject(relative, uri, stream, source_date, name, size_bytes)


def locate_gcloud(explicit: str | None) -> str:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return str(p)
        resolved = shutil.which(explicit)
        if resolved:
            return resolved
        raise SystemExit(f"Could not locate gcloud executable: {explicit}")
    for name in ("gcloud.cmd", "gcloud"):
        resolved = shutil.which(name)
        if resolved:
            return resolved
    raise SystemExit("Could not locate gcloud/gcloud.cmd on PATH. Pass --gcloud if needed.")


def run_gcloud_listing(gcloud: str, gcs_root: str, timeout_seconds: int) -> str:
    cmd = [gcloud, "storage", "ls", "--long", gcs_root.rstrip("/") + "/**"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"GCS inventory timed out after {timeout_seconds}s") from exc
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise SystemExit(f"gcloud storage ls failed ({result.returncode}):\n{message}")
    return result.stdout


def parse_listing(text: str, gcs_root: str, streams: Sequence[str]) -> list[GcsObject]:
    allowed = set(streams)
    objects: list[GcsObject] = []
    seen: set[str] = set()
    for line in text.splitlines():
        m = LONG_LS_RE.match(line)
        if not m:
            continue
        obj = parse_gcs_object(gcs_root, m.group(2), int(m.group(1)), allowed)
        if obj is None:
            continue
        if obj.relative_path in seen:
            raise SystemExit(f"Duplicate GCS object in inventory: {obj.relative_path}")
        seen.add(obj.relative_path)
        objects.append(obj)
    objects.sort(key=lambda o: (o.source_date, o.stream, o.relative_path))
    return objects


def load_completed_ledger(
    state_db: Path, streams: Sequence[str]
) -> tuple[dict[str, int], dict[str, date], dict[str, date]]:
    if not state_db.exists():
        raise SystemExit(f"Canonical extraction state DB not found: {state_db}")
    completed: dict[str, int] = {}
    starts: dict[str, date] = {}
    frontiers: dict[str, date] = {}
    allowed = set(streams)
    con = sqlite3.connect(str(state_db))
    con.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "source_files" not in tables:
            raise SystemExit(f"source_files table not found in {state_db}")
        for row in con.execute(
            """
            SELECT source_relative_path, size_bytes
            FROM source_files
            WHERE UPPER(status) = 'COMPLETED'
            """
        ):
            rel = normalize_relative_path(row["source_relative_path"])
            completed[rel] = int(row["size_bytes"])
            parsed = path_date_from_relative(rel)
            if parsed is None:
                continue
            stream, source_date = parsed
            if stream not in allowed:
                continue
            if stream not in starts or source_date < starts[stream]:
                starts[stream] = source_date
            if stream not in frontiers or source_date > frontiers[stream]:
                frontiers[stream] = source_date
    finally:
        con.close()
    return completed, starts, frontiers


def select_batch(
    objects: Sequence[GcsObject],
    completed: dict[str, int],
    starts: dict[str, date],
    frontiers: dict[str, date],
    streams: Sequence[str],
    batch_days: int,
    source_date_from: date | None,
    source_date_to: date | None,
    include_covered_date_extras: bool,
) -> tuple[list[GcsObject], list[GcsObject], list[GcsObject], date | None, date | None]:
    precanonical_history: list[GcsObject] = []
    covered_date_extras: list[GcsObject] = []
    candidates: list[GcsObject] = []
    allowed = set(streams)

    for obj in objects:
        if obj.stream not in allowed or obj.relative_path in completed:
            continue
        if source_date_from and obj.source_date < source_date_from:
            continue
        if source_date_to and obj.source_date > source_date_to:
            continue

        canonical_start = starts.get(obj.stream)
        frontier = frontiers.get(obj.stream)

        if canonical_start is None or frontier is None:
            if source_date_from is None:
                raise SystemExit(
                    f"No COMPLETED canonical history exists for stream '{obj.stream}'. "
                    "Pass --source-date-from explicitly."
                )
            candidates.append(obj)
            continue

        if obj.source_date < canonical_start:
            precanonical_history.append(obj)
            continue

        if obj.source_date <= frontier:
            if include_covered_date_extras:
                candidates.append(obj)
            else:
                covered_date_extras.append(obj)
            continue

        candidates.append(obj)

    if not candidates:
        return [], precanonical_history, covered_date_extras, None, None

    first_date = min(o.source_date for o in candidates)
    batch_end = first_date + timedelta(days=batch_days - 1)
    if source_date_to and batch_end > source_date_to:
        batch_end = source_date_to
    batch = [o for o in candidates if first_date <= o.source_date <= batch_end]
    batch.sort(key=lambda o: (o.source_date, o.stream, o.relative_path))
    return batch, precanonical_history, covered_date_extras, first_date, batch_end


def local_path_for(raw_root: Path, relative_path: str) -> Path:
    parts = normalize_relative_path(relative_path).split("/")
    if len(parts) != 3:
        raise RuntimeError(f"Unexpected WAL relative path: {relative_path}")
    return raw_root.joinpath(*parts)


def check_space(raw_root: Path, required_bytes: int, min_free_space_gb: float) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(raw_root)
    reserve = int(min_free_space_gb * (1024 ** 3))
    projected = usage.free - required_bytes
    print(f"Current raw-volume free    : {format_bytes(usage.free)}")
    print(f"Batch bytes to add         : {format_bytes(required_bytes)}")
    print(f"Projected free after batch : {format_bytes(max(projected, 0))}")
    print(f"Required free reserve      : {min_free_space_gb:.2f} GiB")
    if projected < reserve:
        raise SystemExit(
            f"Refusing download: projected free space would fall below {min_free_space_gb:.2f} GiB."
        )


def print_plan(
    objects: Sequence[GcsObject], completed: dict[str, int],
    starts: dict[str, date], frontiers: dict[str, date],
    batch: Sequence[GcsObject], precanonical_history: Sequence[GcsObject],
    covered_date_extras: Sequence[GcsObject],
    batch_start: date | None, batch_end: date | None, streams: Sequence[str]
) -> None:
    print("===== BRONZE BATCH PLAN =====")
    print(f"Script version            : {SCRIPT_VERSION}")
    print(f"GCS inventory market WALs : {len(objects):,}")
    print(f"Completed ledger WALs      : {len(completed):,}")
    print(f"Pre-canonical GCS history  : {len(precanonical_history):,}")
    if precanonical_history:
        print(f"Pre-canonical history size : {format_bytes(sum(o.size_bytes for o in precanonical_history))}")
    print(f"Covered-date GCS extras    : {len(covered_date_extras):,}")
    if covered_date_extras:
        print(f"Covered-date extra size    : {format_bytes(sum(o.size_bytes for o in covered_date_extras))}")
    print("\nCanonical source-date ranges:")
    for stream in streams:
        start_date = starts.get(stream)
        frontier = frontiers.get(stream)
        if start_date and frontier:
            print(f"  {stream:10s}: {start_date.isoformat()} -> {frontier.isoformat()}")
        else:
            print(f"  {stream:10s}: NONE")
    if not batch:
        print("\nNo new post-frontier WAL batch is available.")
        return
    print(f"\nSelected batch            : {batch_start} -> {batch_end}")
    print(f"Selected WALs             : {len(batch):,}")
    print(f"Selected bytes            : {format_bytes(sum(o.size_bytes for o in batch))}")
    print("\nBy stream:")
    for stream in streams:
        rows = [o for o in batch if o.stream == stream]
        if not rows:
            print(f"  {stream:10s}: files=   0")
            continue
        print(
            f"  {stream:10s}: files={len(rows):4d} "
            f"size={format_bytes(sum(o.size_bytes for o in rows)):>10s} "
            f"first={min(o.source_date for o in rows)} last={max(o.source_date for o in rows)}"
        )


def safe_print(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def download_one(
    obj: GcsObject, raw_root: Path, gcloud: str, max_attempts: int, attempt_timeout_seconds: int
) -> DownloadResult:
    final_path = local_path_for(raw_root, obj.relative_path)
    part_path = final_path.with_name(final_path.name + ".part")
    final_path.parent.mkdir(parents=True, exist_ok=True)

    if final_path.exists():
        actual = final_path.stat().st_size
        if actual == obj.size_bytes:
            safe_print(f"SKIP    {obj.relative_path} ({format_bytes(actual)})")
            return DownloadResult(obj.relative_path, obj.gcs_uri, str(final_path), obj.size_bytes, actual,
                                  "SKIPPED_EXISTING_VERIFIED", 0, None)
        msg = f"Existing final file size {actual} != expected {obj.size_bytes}; refusing overwrite"
        safe_print(f"FAIL    {obj.relative_path} - {msg}")
        return DownloadResult(obj.relative_path, obj.gcs_uri, str(final_path), obj.size_bytes, actual,
                              "FAILED_EXISTING_SIZE_MISMATCH", 0, msg)

    if part_path.exists():
        part_size = part_path.stat().st_size
        if part_size == obj.size_bytes:
            os.replace(part_path, final_path)
            safe_print(f"PROMOTE {obj.relative_path} ({format_bytes(part_size)})")
            return DownloadResult(obj.relative_path, obj.gcs_uri, str(final_path), obj.size_bytes, part_size,
                                  "PROMOTED_VERIFIED_PART", 0, None)
        part_path.unlink()

    error: str | None = None
    for attempt in range(1, max_attempts + 1):
        if part_path.exists():
            part_path.unlink()
        safe_print(f"GET     {obj.relative_path} [attempt {attempt}/{max_attempts}, {format_bytes(obj.size_bytes)}]")
        cmd = [gcloud, "storage", "cp", "--quiet", obj.gcs_uri, str(part_path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=attempt_timeout_seconds, check=False)
        except subprocess.TimeoutExpired:
            error = f"Timed out after {attempt_timeout_seconds}s"
            safe_print(f"RETRY   {obj.relative_path} - {error}")
            continue
        if result.returncode != 0:
            error = (result.stderr or result.stdout or "").strip() or f"gcloud exited {result.returncode}"
            safe_print(f"RETRY   {obj.relative_path} - gcloud exit {result.returncode}")
            continue
        if not part_path.exists():
            error = "gcloud reported success but .part file is missing"
            safe_print(f"RETRY   {obj.relative_path} - {error}")
            continue
        actual = part_path.stat().st_size
        if actual != obj.size_bytes:
            error = f"Downloaded size {actual} != expected {obj.size_bytes}"
            safe_print(f"RETRY   {obj.relative_path} - {error}")
            continue
        os.replace(part_path, final_path)
        safe_print(f"DONE    {obj.relative_path} ({format_bytes(actual)})")
        return DownloadResult(obj.relative_path, obj.gcs_uri, str(final_path), obj.size_bytes, actual,
                              "COMPLETED", attempt, None)

    actual = part_path.stat().st_size if part_path.exists() else None
    safe_print(f"FAIL    {obj.relative_path} - {error or 'unknown error'}")
    return DownloadResult(obj.relative_path, obj.gcs_uri, str(final_path), obj.size_bytes, actual,
                          "FAILED", max_attempts, error or "Unknown failure")


def write_manifest(
    manifest_dir: Path, batch_start: date, batch_end: date, batch: Sequence[GcsObject],
    results: Sequence[DownloadResult] | None, dry_run: bool
) -> tuple[Path, Path]:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    mode = "plan" if dry_run else "result"
    base = f"bronze-batch-{batch_start}-{batch_end}-{stamp}-{mode}"
    csv_path = manifest_dir / f"{base}.csv"
    json_path = manifest_dir / f"{base}.json"
    result_by_rel = {r.relative_path: r for r in (results or [])}
    rows = []
    for obj in batch:
        result = result_by_rel.get(obj.relative_path)
        rows.append({
            "relative_path": obj.relative_path,
            "gcs_uri": obj.gcs_uri,
            "stream": obj.stream,
            "source_date": obj.source_date.isoformat(),
            "expected_bytes": obj.size_bytes,
            "status": result.status if result else "PLANNED",
            "attempts": result.attempts if result else 0,
            "actual_bytes": result.actual_bytes if result else None,
            "error": result.error if result else None,
        })
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": iso_utc(),
        "dry_run": dry_run,
        "batch_start": batch_start.isoformat(),
        "batch_end": batch_end.isoformat(),
        "file_count": len(batch),
        "expected_bytes": sum(o.size_bytes for o in batch),
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, json_path


def verify_downloaded_batch(
    batch: Sequence[GcsObject], results: Sequence[DownloadResult], raw_root: Path
) -> tuple[int, list[str]]:
    failures: list[str] = []
    verified = 0
    result_by_rel = {r.relative_path: r for r in results}
    for obj in batch:
        result = result_by_rel.get(obj.relative_path)
        if result is None:
            failures.append(f"{obj.relative_path}: no result")
            continue
        final_path = local_path_for(raw_root, obj.relative_path)
        if not final_path.exists():
            failures.append(f"{obj.relative_path}: final file missing")
            continue
        actual = final_path.stat().st_size
        if actual != obj.size_bytes:
            failures.append(f"{obj.relative_path}: final size {actual} != {obj.size_bytes}")
            continue
        if result.status.startswith("FAILED"):
            failures.append(f"{obj.relative_path}: result status {result.status}")
            continue
        verified += 1
    return verified, failures


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download the next post-canonical-frontier Stream bronze WAL batch from GCS."
    )
    p.add_argument("--gcs-root", default=DEFAULT_GCS_ROOT)
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    p.add_argument("--state-db", type=Path, default=DEFAULT_STATE_DB)
    p.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    p.add_argument("--streams", default=",".join(DEFAULT_STREAMS))
    p.add_argument("--batch-days", type=int, default=7)
    p.add_argument("--source-date-from")
    p.add_argument("--source-date-to", help="Optional upper source-date bound. Default: yesterday UTC (closed dates only).")
    p.add_argument(
        "--include-covered-date-extras", action="store_true",
        help="Include GCS-only WALs on dates already covered by canonical. Default: report and skip them."
    )
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--max-attempts", type=int, default=2)
    p.add_argument("--attempt-timeout-minutes", type=float, default=20.0)
    p.add_argument("--inventory-timeout-minutes", type=float, default=10.0)
    p.add_argument("--min-free-space-gb", type=float, default=20.0)
    p.add_argument("--max-files", type=int)
    p.add_argument("--gcloud")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.batch_days < 1:
        p.error("--batch-days must be at least 1")
    if args.parallel < 1:
        p.error("--parallel must be at least 1")
    if args.max_attempts < 1:
        p.error("--max-attempts must be at least 1")
    if args.attempt_timeout_minutes <= 0 or args.inventory_timeout_minutes <= 0:
        p.error("timeouts must be greater than 0")
    if args.min_free_space_gb < 0:
        p.error("--min-free-space-gb cannot be negative")
    if args.max_files is not None and args.max_files < 1:
        p.error("--max-files must be at least 1")

    args.source_date_from = parse_date(args.source_date_from, "--source-date-from") if args.source_date_from else None
    args.source_date_to = (
        parse_date(args.source_date_to, "--source-date-to")
        if args.source_date_to
        else (datetime.now(timezone.utc).date() - timedelta(days=1))
    )
    if args.source_date_from and args.source_date_to and args.source_date_from > args.source_date_to:
        p.error("--source-date-from cannot be after --source-date-to")
    args.streams = tuple(x.strip() for x in args.streams.split(",") if x.strip())
    if not args.streams:
        p.error("--streams cannot be empty")
    return args


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    state_db = args.state_db.expanduser().resolve()
    manifest_dir = args.manifest_dir.expanduser().resolve()
    gcloud = locate_gcloud(args.gcloud)

    print("Stream bronze batch downloader")
    print("------------------------------")
    print(f"Version          : {SCRIPT_VERSION}")
    print(f"GCS root         : {args.gcs_root}")
    print(f"Raw root         : {raw_root}")
    print(f"State DB         : {state_db}")
    print(f"Manifest dir     : {manifest_dir}")
    print(f"gcloud           : {gcloud}")
    print(f"Batch days       : {args.batch_days}")
    print(f"Parallel         : {args.parallel}")
    print(f"Max attempts     : {args.max_attempts}")
    print(f"Attempt timeout  : {args.attempt_timeout_minutes:.1f} min")
    print(f"Closed-date cap  : {args.source_date_to}")
    print(f"Dry run          : {args.dry_run}\n")

    completed, starts, frontiers = load_completed_ledger(state_db, args.streams)
    print("Fetching GCS inventory...", flush=True)
    listing = run_gcloud_listing(gcloud, args.gcs_root, int(args.inventory_timeout_minutes * 60))
    objects = parse_listing(listing, args.gcs_root, args.streams)
    batch, precanonical_history, covered_date_extras, batch_start, batch_end = select_batch(
        objects, completed, starts, frontiers, args.streams, args.batch_days,
        args.source_date_from, args.source_date_to, args.include_covered_date_extras,
    )
    if args.max_files is not None and len(batch) > args.max_files:
        batch = batch[:args.max_files]
        if batch:
            batch_start = min(o.source_date for o in batch)
            batch_end = max(o.source_date for o in batch)

    print_plan(
        objects, completed, starts, frontiers, batch,
        precanonical_history, covered_date_extras,
        batch_start, batch_end, args.streams,
    )

    if precanonical_history:
        print("\nNOTE: GCS contains WALs from before the trusted canonical history starts.")
        print("They are historical archive only and are never included in normal backlog processing.")
        for obj in precanonical_history[:5]:
            print(f"  PRE   {obj.relative_path} ({format_bytes(obj.size_bytes)})")
        if len(precanonical_history) > 5:
            print(f"  ... plus {len(precanonical_history) - 5:,} more")

    if covered_date_extras and not args.include_covered_date_extras:
        print("\nNOTE: uncanonicalized GCS objects exist inside already-covered canonical dates.")
        print("They are intentionally excluded from normal backlog processing and require separate review.")
        for obj in covered_date_extras[:10]:
            print(f"  EXTRA {obj.relative_path} ({format_bytes(obj.size_bytes)})")
        if len(covered_date_extras) > 10:
            print(f"  ... plus {len(covered_date_extras) - 10:,} more")

    if not batch:
        return 0
    assert batch_start is not None and batch_end is not None

    plan_csv, plan_json = write_manifest(manifest_dir, batch_start, batch_end, batch, None, True)
    print(f"\nPlan CSV  : {plan_csv}")
    print(f"Plan JSON : {plan_json}")
    if args.dry_run:
        print("\nDRY RUN: no files were downloaded.")
        return 0

    required_bytes = 0
    for obj in batch:
        final_path = local_path_for(raw_root, obj.relative_path)
        if final_path.exists() and final_path.stat().st_size == obj.size_bytes:
            continue
        part_path = final_path.with_name(final_path.name + ".part")
        if part_path.exists() and part_path.stat().st_size == obj.size_bytes:
            continue
        required_bytes += obj.size_bytes

    print("\n===== SPACE GUARD =====")
    check_space(raw_root, required_bytes, args.min_free_space_gb)
    print("\n===== DOWNLOAD =====")

    results: list[DownloadResult] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(download_one, obj, raw_root, gcloud, args.max_attempts,
                        int(args.attempt_timeout_minutes * 60)): obj
            for obj in batch
        }
        for future in as_completed(futures):
            obj = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                safe_print(f"FAIL    {obj.relative_path} - {type(exc).__name__}: {exc}")
                results.append(DownloadResult(
                    obj.relative_path, obj.gcs_uri, str(local_path_for(raw_root, obj.relative_path)),
                    obj.size_bytes, None, "FAILED_EXCEPTION", 0,
                    f"{type(exc).__name__}: {exc}"
                ))

    results.sort(key=lambda r: r.relative_path)
    verified, failures = verify_downloaded_batch(batch, results, raw_root)
    result_csv, result_json = write_manifest(manifest_dir, batch_start, batch_end, batch, results, False)

    print("\n===== FINAL DOWNLOAD AUDIT =====")
    print(f"Planned WALs  : {len(batch):,}")
    print(f"Verified WALs : {verified:,}")
    print(f"Failures      : {len(failures):,}")
    print(f"Result CSV    : {result_csv}")
    print(f"Result JSON   : {result_json}")
    if failures:
        print("\nDownload batch FAILED verification. Raw files are retained.")
        for msg in failures[:20]:
            print(f"  {msg}")
        if len(failures) > 20:
            print(f"  ... plus {len(failures) - 20:,} more")
        return 2

    print("\nPASS: every planned WAL exists locally with the exact GCS-listed byte size.")
    print("No raw files were deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
