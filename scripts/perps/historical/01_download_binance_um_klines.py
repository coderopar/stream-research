#!/usr/bin/env python3
"""Download and verify Binance USD-M monthly kline archives.

Pass 1 of the Stream historical-context pipeline.

Scope is deliberately narrow:
- Binance USD-M futures only.
- BTCUSDT by default.
- 1-minute klines by default.
- Monthly archives only.
- Preserve original ZIP and CHECKSUM files.
- Discover availability from official CHECKSUM objects.
- Verify every archive with Binance's published SHA-256.
- Keep resumable SQLite state and a CSV/JSON manifest.
- Perform no extraction, normalization, resampling, or research feature work.

Official archive pattern:
  https://data.binance.vision/data/futures/um/monthly/klines/
      <SYMBOL>/<INTERVAL>/<SYMBOL>-<INTERVAL>-<YYYY-MM>.zip

Each archive has a sibling .zip.CHECKSUM file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

DOWNLOADER_VERSION = "1.0.0"
BASE_URL = "https://data.binance.vision"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1m"
DEFAULT_START_MONTH = "2019-01"
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\stream-data\historical\binance\futures_um\BTCUSDT\klines_1m"
)
USER_AGENT = f"stream-research-binance-history/{DOWNLOADER_VERSION}"
CHECKSUM_RE = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$")
MONTH_RE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")


@dataclass(frozen=True)
class MonthSpec:
    month: str
    year: int
    month_number: int
    filename: str
    archive_url: str
    checksum_url: str


@dataclass
class DiscoveryResult:
    month: str
    available: bool
    expected_sha256: str = ""
    checksum_filename: str = ""
    checksum_text: str = ""
    http_status: Optional[int] = None
    error: str = ""


@dataclass
class DownloadResult:
    month: str
    status: str
    archive_path: str = ""
    checksum_path: str = ""
    expected_sha256: str = ""
    actual_sha256: str = ""
    compressed_bytes: int = 0
    error: str = ""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def previous_utc_month() -> str:
    now = datetime.now(timezone.utc)
    if now.month == 1:
        return f"{now.year - 1:04d}-12"
    return f"{now.year:04d}-{now.month - 1:02d}"


def parse_month(value: str) -> tuple[int, int]:
    if not MONTH_RE.match(value):
        raise ValueError(f"Invalid month {value!r}; expected YYYY-MM")
    year, month = value.split("-")
    return int(year), int(month)


def month_range(start: str, end: str) -> list[str]:
    sy, sm = parse_month(start)
    ey, em = parse_month(end)
    if (sy, sm) > (ey, em):
        raise ValueError(f"start month {start} is after end month {end}")
    out: list[str] = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y += 1
            m = 1
    return out


def build_month_spec(symbol: str, interval: str, month: str) -> MonthSpec:
    year, month_number = parse_month(month)
    filename = f"{symbol}-{interval}-{month}.zip"
    prefix = f"data/futures/um/monthly/klines/{symbol}/{interval}"
    archive_url = f"{BASE_URL}/{prefix}/{filename}"
    return MonthSpec(
        month=month,
        year=year,
        month_number=month_number,
        filename=filename,
        archive_url=archive_url,
        checksum_url=f"{archive_url}.CHECKSUM",
    )


def build_request(url: str, *, method: str = "GET", range_start: Optional[int] = None) -> urllib.request.Request:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "*/*",
        "Accept-Encoding": "identity",
    }
    if range_start is not None and range_start > 0:
        headers["Range"] = f"bytes={range_start}-"
    return urllib.request.Request(url, headers=headers, method=method)


def sleep_backoff(attempt: int) -> None:
    time.sleep(min(8.0, 0.75 * (2 ** attempt)))


def fetch_small_text(url: str, timeout: float, retries: int) -> tuple[Optional[str], Optional[int], str]:
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(build_request(url), timeout=timeout) as response:
                status = getattr(response, "status", 200)
                payload = response.read(32 * 1024)
                return payload.decode("utf-8", errors="replace"), status, ""
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None, 404, ""
            if exc.code in (408, 425, 429, 500, 502, 503, 504) and attempt < retries:
                sleep_backoff(attempt)
                continue
            return None, exc.code, f"HTTP {exc.code}: {exc.reason}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt < retries:
                sleep_backoff(attempt)
                continue
            return None, None, f"{type(exc).__name__}: {exc}"
    return None, None, "unreachable"


def parse_checksum(text: str, expected_filename: str) -> tuple[str, str]:
    lines = [line.strip() for line in text.replace("\r", "\n").split("\n") if line.strip()]
    if not lines:
        raise ValueError("checksum file is empty")
    match = CHECKSUM_RE.match(lines[0])
    if not match:
        raise ValueError(f"unrecognized checksum format: {lines[0]!r}")
    digest = match.group(1).lower()
    filename = Path(match.group(2).strip()).name
    if filename != expected_filename:
        raise ValueError(
            f"checksum names {filename!r}, expected {expected_filename!r}"
        )
    return digest, filename


def discover_one(spec: MonthSpec, timeout: float, retries: int) -> DiscoveryResult:
    text, status, error = fetch_small_text(spec.checksum_url, timeout, retries)
    if text is None:
        return DiscoveryResult(
            month=spec.month,
            available=False,
            http_status=status,
            error=error,
        )
    try:
        digest, checksum_filename = parse_checksum(text, spec.filename)
    except ValueError as exc:
        return DiscoveryResult(
            month=spec.month,
            available=False,
            http_status=status,
            error=f"CHECKSUM_PARSE_ERROR: {exc}",
        )
    return DiscoveryResult(
        month=spec.month,
        available=True,
        expected_sha256=digest,
        checksum_filename=checksum_filename,
        checksum_text=text,
        http_status=status,
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def short_temp_path(parent: Path, suffix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    return parent / f".tmp-{uuid.uuid4().hex[:8]}{suffix}"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = short_temp_path(path.parent, ".txt")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def atomic_write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = short_temp_path(path.parent, ".csv")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS months (
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            month TEXT NOT NULL,
            archive_url TEXT NOT NULL,
            checksum_url TEXT NOT NULL,
            available INTEGER NOT NULL DEFAULT 0,
            expected_sha256 TEXT,
            actual_sha256 TEXT,
            compressed_bytes INTEGER,
            archive_path TEXT,
            checksum_path TEXT,
            status TEXT NOT NULL,
            http_status INTEGER,
            error TEXT,
            discovered_at TEXT,
            downloaded_at TEXT,
            verified_at TEXT,
            downloader_version TEXT NOT NULL,
            PRIMARY KEY (symbol, interval, month)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL,
            config_json TEXT NOT NULL,
            summary_json TEXT
        )
        """
    )
    connection.commit()


def upsert_discovery(
    connection: sqlite3.Connection,
    symbol: str,
    interval: str,
    spec: MonthSpec,
    result: DiscoveryResult,
) -> None:
    now = utc_now_iso()
    status = "DISCOVERED" if result.available else ("MISSING" if result.http_status == 404 else "DISCOVERY_FAILED")
    connection.execute(
        """
        INSERT INTO months (
            symbol, interval, month, archive_url, checksum_url,
            available, expected_sha256, status, http_status, error,
            discovered_at, downloader_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, interval, month) DO UPDATE SET
            archive_url=excluded.archive_url,
            checksum_url=excluded.checksum_url,
            available=excluded.available,
            expected_sha256=excluded.expected_sha256,
            status=CASE
                WHEN months.status IN ('VERIFIED','VERIFIED_REUSED')
                     AND months.expected_sha256=excluded.expected_sha256
                     AND excluded.available=1
                THEN months.status
                WHEN months.expected_sha256 IS NOT NULL
                     AND excluded.expected_sha256 IS NOT NULL
                     AND months.expected_sha256<>excluded.expected_sha256
                THEN 'CHECKSUM_CHANGED'
                ELSE excluded.status
            END,
            http_status=excluded.http_status,
            error=excluded.error,
            discovered_at=excluded.discovered_at,
            downloader_version=excluded.downloader_version
        """,
        (
            symbol,
            interval,
            spec.month,
            spec.archive_url,
            spec.checksum_url,
            1 if result.available else 0,
            result.expected_sha256 or None,
            status,
            result.http_status,
            result.error or None,
            now,
            DOWNLOADER_VERSION,
        ),
    )
    connection.commit()


def archive_paths(root: Path, spec: MonthSpec) -> tuple[Path, Path]:
    archive = root / "archives" / f"year={spec.year:04d}" / spec.filename
    checksum = root / "checksums" / f"year={spec.year:04d}" / f"{spec.filename}.CHECKSUM"
    return archive, checksum


def preserve_bad_file(path: Path, label: str) -> Optional[Path]:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = path.with_name(f"{path.name}.{label}-{stamp}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.{label}-{stamp}-{counter}")
        counter += 1
    os.replace(path, candidate)
    return candidate


def download_stream_resumable(url: str, destination: Path, timeout: float, retries: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".dl-{hashlib.sha1(destination.name.encode()).hexdigest()[:8]}.part")

    for attempt in range(retries + 1):
        existing = partial.stat().st_size if partial.exists() else 0
        try:
            request = build_request(url, range_start=existing if existing else None)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                content_range = response.headers.get("Content-Range", "")
                range_honored = existing > 0 and (status == 206 or content_range.startswith(f"bytes {existing}-"))

                if existing > 0 and not range_honored:
                    partial.unlink(missing_ok=True)
                    existing = 0

                mode = "ab" if range_honored else "wb"
                with partial.open(mode) as handle:
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            os.replace(partial, destination)
            return
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and partial.exists():
                # Range already complete or server disagrees; restart cleanly.
                partial.unlink(missing_ok=True)
            if exc.code in (408, 425, 429, 500, 502, 503, 504, 416) and attempt < retries:
                sleep_backoff(attempt)
                continue
            raise RuntimeError(f"HTTP {exc.code}: {exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt < retries:
                sleep_backoff(attempt)
                continue
            raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc


def download_one(
    root: Path,
    spec: MonthSpec,
    discovery: DiscoveryResult,
    timeout: float,
    retries: int,
    force: bool,
) -> DownloadResult:
    archive_path, checksum_path = archive_paths(root, spec)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    checksum_path.parent.mkdir(parents=True, exist_ok=True)

    if not discovery.available:
        return DownloadResult(month=spec.month, status="NOT_AVAILABLE", error=discovery.error)

    expected = discovery.expected_sha256
    atomic_write_text(checksum_path, discovery.checksum_text)

    if archive_path.exists() and not force:
        actual = sha256_file(archive_path)
        if actual == expected:
            return DownloadResult(
                month=spec.month,
                status="VERIFIED_REUSED",
                archive_path=str(archive_path),
                checksum_path=str(checksum_path),
                expected_sha256=expected,
                actual_sha256=actual,
                compressed_bytes=archive_path.stat().st_size,
            )
        preserve_bad_file(archive_path, "checksum-mismatch")

    if force and archive_path.exists():
        preserve_bad_file(archive_path, "forced-replace")

    try:
        download_stream_resumable(spec.archive_url, archive_path, timeout, retries)
        actual = sha256_file(archive_path)
        size = archive_path.stat().st_size
        if actual != expected:
            bad = preserve_bad_file(archive_path, "checksum-mismatch")
            return DownloadResult(
                month=spec.month,
                status="CHECKSUM_FAILED",
                archive_path=str(bad or archive_path),
                checksum_path=str(checksum_path),
                expected_sha256=expected,
                actual_sha256=actual,
                compressed_bytes=size,
                error="Downloaded archive SHA-256 did not match Binance CHECKSUM",
            )
        return DownloadResult(
            month=spec.month,
            status="VERIFIED",
            archive_path=str(archive_path),
            checksum_path=str(checksum_path),
            expected_sha256=expected,
            actual_sha256=actual,
            compressed_bytes=size,
        )
    except Exception as exc:  # network/filesystem boundary; recorded for resumability
        return DownloadResult(
            month=spec.month,
            status="DOWNLOAD_FAILED",
            archive_path=str(archive_path),
            checksum_path=str(checksum_path),
            expected_sha256=expected,
            error=f"{type(exc).__name__}: {exc}",
        )


def update_download_state(
    connection: sqlite3.Connection,
    symbol: str,
    interval: str,
    result: DownloadResult,
) -> None:
    verified = result.status in {"VERIFIED", "VERIFIED_REUSED"}
    connection.execute(
        """
        UPDATE months SET
            actual_sha256=?, compressed_bytes=?, archive_path=?, checksum_path=?,
            status=?, error=?, downloaded_at=CASE WHEN ? THEN COALESCE(downloaded_at, ?) ELSE downloaded_at END,
            verified_at=CASE WHEN ? THEN ? ELSE verified_at END,
            downloader_version=?
        WHERE symbol=? AND interval=? AND month=?
        """,
        (
            result.actual_sha256 or None,
            result.compressed_bytes or None,
            result.archive_path or None,
            result.checksum_path or None,
            result.status,
            result.error or None,
            result.status in {"VERIFIED", "DOWNLOAD_FAILED", "CHECKSUM_FAILED"},
            utc_now_iso(),
            verified,
            utc_now_iso(),
            DOWNLOADER_VERSION,
            symbol,
            interval,
            result.month,
        ),
    )
    connection.commit()


def verify_local_month(root: Path, spec: MonthSpec, expected_sha256: str) -> DownloadResult:
    archive_path, checksum_path = archive_paths(root, spec)
    if not archive_path.exists():
        return DownloadResult(
            month=spec.month,
            status="LOCAL_MISSING",
            archive_path=str(archive_path),
            checksum_path=str(checksum_path),
            expected_sha256=expected_sha256,
            error="archive file does not exist",
        )
    actual = sha256_file(archive_path)
    status = "VERIFIED_REUSED" if actual == expected_sha256 else "CHECKSUM_FAILED"
    return DownloadResult(
        month=spec.month,
        status=status,
        archive_path=str(archive_path),
        checksum_path=str(checksum_path),
        expected_sha256=expected_sha256,
        actual_sha256=actual,
        compressed_bytes=archive_path.stat().st_size,
        error="" if status == "VERIFIED_REUSED" else "Local archive SHA-256 does not match expected checksum",
    )


def state_rows(connection: sqlite3.Connection, symbol: str, interval: str) -> list[dict[str, object]]:
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT symbol, interval, month, available, status,
               expected_sha256, actual_sha256, compressed_bytes,
               archive_url, checksum_url, archive_path, checksum_path,
               http_status, error, discovered_at, downloaded_at, verified_at,
               downloader_version
        FROM months
        WHERE symbol=? AND interval=?
        ORDER BY month
        """,
        (symbol, interval),
    ).fetchall()
    return [dict(row) for row in rows]


def write_reports(root: Path, connection: sqlite3.Connection, symbol: str, interval: str, run_id: str) -> dict[str, object]:
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    rows = state_rows(connection, symbol, interval)
    fieldnames = [
        "symbol", "interval", "month", "available", "status",
        "expected_sha256", "actual_sha256", "compressed_bytes",
        "archive_url", "checksum_url", "archive_path", "checksum_path",
        "http_status", "error", "discovered_at", "downloaded_at", "verified_at",
        "downloader_version",
    ]
    atomic_write_csv(metadata / "download_manifest.csv", rows, fieldnames)

    available_rows = [row for row in rows if int(row.get("available") or 0) == 1]
    verified_rows = [row for row in rows if row.get("status") in {"VERIFIED", "VERIFIED_REUSED"}]
    failed_rows = [row for row in rows if row.get("status") in {"DISCOVERY_FAILED", "DOWNLOAD_FAILED", "CHECKSUM_FAILED", "LOCAL_MISSING"}]
    size = sum(int(row.get("compressed_bytes") or 0) for row in verified_rows)

    summary = {
        "downloader_version": DOWNLOADER_VERSION,
        "run_id": run_id,
        "generated_at": utc_now_iso(),
        "symbol": symbol,
        "interval": interval,
        "archive_source": "monthly",
        "market": "Binance USD-M futures",
        "candidate_months_in_state": len(rows),
        "available_months": len(available_rows),
        "verified_months": len(verified_rows),
        "failed_months": len(failed_rows),
        "first_available_month": min((str(row["month"]) for row in available_rows), default=None),
        "last_available_month": max((str(row["month"]) for row in available_rows), default=None),
        "first_verified_month": min((str(row["month"]) for row in verified_rows), default=None),
        "last_verified_month": max((str(row["month"]) for row in verified_rows), default=None),
        "verified_compressed_bytes": size,
        "verified_compressed_gib": round(size / (1024 ** 3), 4),
        "failures": [
            {"month": row["month"], "status": row["status"], "error": row.get("error")}
            for row in failed_rows
        ],
    }
    atomic_write_json(metadata / "latest_download_summary.json", summary)
    return summary


def print_summary(summary: dict[str, object]) -> None:
    print("\nDownload pass summary")
    print("---------------------")
    print(f"Available months : {summary['available_months']}")
    print(f"Verified months  : {summary['verified_months']}")
    print(f"Failed months    : {summary['failed_months']}")
    print(f"First available  : {summary['first_available_month']}")
    print(f"Last available   : {summary['last_available_month']}")
    print(f"Compressed GiB   : {summary['verified_compressed_gib']}")


def self_test() -> int:
    assert month_range("2019-12", "2020-02") == ["2019-12", "2020-01", "2020-02"]
    spec = build_month_spec("BTCUSDT", "1m", "2020-01")
    assert spec.filename == "BTCUSDT-1m-2020-01.zip"
    assert spec.archive_url.endswith("/BTCUSDT/1m/BTCUSDT-1m-2020-01.zip")
    digest = "a" * 64
    parsed, filename = parse_checksum(f"{digest}  BTCUSDT-1m-2020-01.zip\n", spec.filename)
    assert parsed == digest and filename == spec.filename

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = root / ("x" * 80) / ("y" * 80) / "manifest.json"
        atomic_write_json(target, {"ok": True})
        assert json.loads(target.read_text(encoding="utf-8"))["ok"] is True

        db_path = root / "state.sqlite"
        with sqlite3.connect(db_path) as connection:
            ensure_schema(connection)
            discovery = DiscoveryResult(
                month="2020-01",
                available=True,
                expected_sha256=digest,
                checksum_filename=spec.filename,
                checksum_text=f"{digest}  {spec.filename}\n",
                http_status=200,
            )
            upsert_discovery(connection, "BTCUSDT", "1m", spec, discovery)
            rows = state_rows(connection, "BTCUSDT", "1m")
            assert len(rows) == 1 and rows[0]["month"] == "2020-01"

    print("SELF-TEST PASS")
    print("Validated month discovery contracts, Binance checksum parsing, SQLite resumability, and short-path atomic writers.")
    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and SHA-256 verify Binance USD-M monthly kline archives."
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--start-month", default=DEFAULT_START_MONTH, help="YYYY-MM; discovery lower bound")
    parser.add_argument("--end-month", default=previous_utc_month(), help="YYYY-MM; default is previous UTC month")
    parser.add_argument("--months", default="", help="Comma-separated explicit YYYY-MM list; overrides start/end range")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-months", type=int, default=0, help="Limit selected available months after discovery; 0 = no limit")
    parser.add_argument("--discover-only", action="store_true", help="Discover availability/checksums and write manifests, but do not download ZIPs")
    parser.add_argument("--verify-only", action="store_true", help="Do not use network; verify local ZIPs using checksums already recorded in SQLite")
    parser.add_argument("--force", action="store_true", help="Redownload even when a local archive already verifies")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def selected_months(args: argparse.Namespace) -> list[str]:
    if args.months.strip():
        months = [item.strip() for item in args.months.split(",") if item.strip()]
        for month in months:
            parse_month(month)
        return sorted(set(months))
    return month_range(args.start_month, args.end_month)


def run(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    if args.discover_only and args.verify_only:
        raise SystemExit("--discover-only and --verify-only are mutually exclusive")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.retries < 0:
        raise SystemExit("--retries must be >= 0")

    symbol = args.symbol.upper().strip()
    interval = args.interval.strip()
    root = args.output_root.resolve()
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    db_path = metadata / "download_state.sqlite"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]

    months = selected_months(args)
    specs = {month: build_month_spec(symbol, interval, month) for month in months}

    print("Downloading Binance USD-M monthly kline archives...")
    print(f"Downloader version : {DOWNLOADER_VERSION}")
    print(f"Run ID             : {run_id}")
    print(f"Symbol             : {symbol}")
    print(f"Interval           : {interval}")
    print(f"Candidate months   : {len(months)} ({months[0]} to {months[-1]})")
    print(f"Output root        : {root}")
    print(f"Workers            : {args.workers}")
    print("No extraction or normalization will be performed.")

    config = {
        "downloader_version": DOWNLOADER_VERSION,
        "symbol": symbol,
        "interval": interval,
        "months": months,
        "output_root": str(root),
        "workers": args.workers,
        "timeout": args.timeout,
        "retries": args.retries,
        "discover_only": args.discover_only,
        "verify_only": args.verify_only,
        "force": args.force,
        "base_url": BASE_URL,
    }

    with sqlite3.connect(db_path) as connection:
        ensure_schema(connection)
        connection.execute(
            "INSERT INTO runs(run_id, started_at, status, config_json) VALUES (?, ?, ?, ?)",
            (run_id, utc_now_iso(), "RUNNING", json.dumps(config, sort_keys=True)),
        )
        connection.commit()

        try:
            if args.verify_only:
                existing = {
                    row["month"]: row
                    for row in state_rows(connection, symbol, interval)
                    if row.get("expected_sha256")
                }
                if not existing:
                    raise RuntimeError("No stored expected checksums found; run discovery/download first")
                verify_months = [month for month in months if month in existing]
                if not verify_months:
                    verify_months = sorted(existing)
                print(f"Verifying {len(verify_months)} local archives from stored checksums...")
                for month in verify_months:
                    spec = specs.get(month) or build_month_spec(symbol, interval, month)
                    result = verify_local_month(root, spec, str(existing[month]["expected_sha256"]))
                    update_download_state(connection, symbol, interval, result)
                    print(f"[{month}] {result.status}")
            else:
                print("Discovering official monthly CHECKSUM objects...")
                discovery: dict[str, DiscoveryResult] = {}
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    future_map = {
                        executor.submit(discover_one, spec, args.timeout, args.retries): month
                        for month, spec in specs.items()
                    }
                    for future in as_completed(future_map):
                        month = future_map[future]
                        result = future.result()
                        discovery[month] = result
                        upsert_discovery(connection, symbol, interval, specs[month], result)

                available = [month for month in months if discovery[month].available]
                missing = [month for month in months if discovery[month].http_status == 404]
                failed = [
                    month for month in months
                    if not discovery[month].available and discovery[month].http_status != 404
                ]
                print(f"Available archives  : {len(available)}")
                print(f"Unavailable months   : {len(missing)}")
                print(f"Discovery failures   : {len(failed)}")
                if available:
                    print(f"Discovered coverage  : {available[0]} through {available[-1]}")
                if failed:
                    for month in failed:
                        print(f"[{month}] DISCOVERY_FAILED: {discovery[month].error}")

                if not args.discover_only:
                    if args.max_months > 0:
                        available = available[: args.max_months]
                    print(f"Downloading/verifying {len(available)} available monthly archives...")
                    with ThreadPoolExecutor(max_workers=args.workers) as executor:
                        future_map = {
                            executor.submit(
                                download_one,
                                root,
                                specs[month],
                                discovery[month],
                                args.timeout,
                                args.retries,
                                args.force,
                            ): month
                            for month in available
                        }
                        completed = 0
                        for future in as_completed(future_map):
                            month = future_map[future]
                            result = future.result()
                            update_download_state(connection, symbol, interval, result)
                            completed += 1
                            size_mib = result.compressed_bytes / (1024 ** 2) if result.compressed_bytes else 0.0
                            print(f"[{completed:02d}/{len(available):02d}] {month}: {result.status} ({size_mib:.2f} MiB)")
                            if result.error:
                                print(f"    {result.error}")

            summary = write_reports(root, connection, symbol, interval, run_id)
            failed_count = int(summary["failed_months"])
            final_status = "COMPLETED" if failed_count == 0 else "COMPLETED_WITH_FAILURES"
            connection.execute(
                "UPDATE runs SET finished_at=?, status=?, summary_json=? WHERE run_id=?",
                (utc_now_iso(), final_status, json.dumps(summary, sort_keys=True), run_id),
            )
            connection.commit()
            print_summary(summary)
            print(f"Manifest          : {metadata / 'download_manifest.csv'}")
            print(f"Summary           : {metadata / 'latest_download_summary.json'}")
            print(f"SQLite state      : {db_path}")
            return 0 if failed_count == 0 else 1
        except KeyboardInterrupt:
            connection.execute(
                "UPDATE runs SET finished_at=?, status=? WHERE run_id=?",
                (utc_now_iso(), "INTERRUPTED", run_id),
            )
            connection.commit()
            print("\nInterrupted. Verified months remain resumable; rerun the same command.", file=sys.stderr)
            return 130
        except Exception as exc:
            connection.execute(
                "UPDATE runs SET finished_at=?, status=?, summary_json=? WHERE run_id=?",
                (utc_now_iso(), "ERROR", json.dumps({"error": f"{type(exc).__name__}: {exc}"}), run_id),
            )
            connection.commit()
            raise


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
