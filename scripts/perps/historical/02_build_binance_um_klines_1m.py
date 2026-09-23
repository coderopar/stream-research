#!/usr/bin/env python3
"""Normalize verified Binance USD-M BTCUSDT monthly 1m kline archives to Parquet.

Pass 2 of the historical-context pipeline.

Contracts
---------
* Reads only Pass-1 rows whose status is VERIFIED or VERIFIED_REUSED.
* Recomputes SHA-256 of every source ZIP before parsing.
* Never modifies or extracts into the source archive/checksum directories.
* Writes one deterministic Parquet file per verified month.
* Does not fill missing minutes or repair source data.
* Uses SQLite state plus input/config signatures for resumability.
* Uses short same-directory temporary names for Windows path safety.

Default source/output root:
    D:\\stream-data\\historical\\binance\\futures_um\\BTCUSDT\\klines_1m

Expected Pass-1 files:
    metadata/download_manifest.csv
    archives/year=YYYY/BTCUSDT-1m-YYYY-MM.zip

Outputs:
    parquet/symbol=BTCUSDT/year=YYYY/month=MM/part-00000.parquet
    metadata/build_state.sqlite
    metadata/build_manifest.csv
    metadata/latest_build_summary.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

BUILDER_VERSION = "1.0.1"
SCHEMA_VERSION = "binance_um_klines_1m_v1"
DEFAULT_ROOT = Path(r"D:\stream-data\historical\binance\futures_um\BTCUSDT\klines_1m")
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1m"

SOURCE_COLUMNS = [
    "open_time_raw",
    "open",
    "high",
    "low",
    "close",
    "base_volume",
    "close_time_raw",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

OUTPUT_COLUMNS = [
    "symbol",
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "base_volume",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "taker_sell_base_volume",
    "taker_sell_quote_volume",
    "taker_buy_quote_share",
]


@dataclass(frozen=True)
class SourceMonth:
    month: str
    expected_sha256: str
    source_bytes_manifest: int
    archive_path: Path
    source_manifest_row: dict[str, str]


@dataclass
class BuildResult:
    month: str
    status: str
    input_signature: str
    expected_sha256: str
    actual_sha256: str
    source_bytes: int
    rows: int
    first_open_time: str | None
    last_open_time: str | None
    parquet_path: str
    parquet_bytes: int
    processed_at: str
    error: str = ""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def short_tmp_path(final_path: Path, prefix: str) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    return final_path.parent / f".{prefix}-{uuid.uuid4().hex[:8]}.tmp"


def atomic_write_text(path: Path, content: str) -> None:
    tmp = short_tmp_path(path, "tx")
    try:
        tmp.write_text(content, encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def atomic_write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    tmp = short_tmp_path(path, "cs")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def require_libraries():
    try:
        import numpy as np  # noqa: F401
        import pandas as pd  # noqa: F401
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Pass 2 requires numpy, pandas and pyarrow. Install with: "
            "python -m pip install numpy pandas pyarrow"
        ) from exc


def parse_months(value: str | None) -> list[str] | None:
    if not value:
        return None
    months: list[str] = []
    for token in value.split(","):
        token = token.strip()
        try:
            datetime.strptime(token, "%Y-%m")
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid month {token!r}; expected YYYY-MM") from exc
        months.append(token)
    return sorted(set(months))


def load_verified_sources(root: Path, symbol: str, interval: str, selected: list[str] | None) -> list[SourceMonth]:
    manifest_path = root / "metadata" / "download_manifest.csv"
    if not manifest_path.exists():
        raise RuntimeError(f"Pass-1 download manifest was not found: {manifest_path}")

    rows: list[SourceMonth] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"symbol", "interval", "month", "status", "expected_sha256", "archive_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Pass-1 manifest is missing required columns: {sorted(missing)}")
        for row in reader:
            if row.get("symbol") != symbol or row.get("interval") != interval:
                continue
            month = row.get("month", "")
            if selected is not None and month not in selected:
                continue
            if row.get("status") not in {"VERIFIED", "VERIFIED_REUSED"}:
                continue
            archive_path = Path(row["archive_path"])
            if not archive_path.exists():
                raise RuntimeError(f"Manifest says VERIFIED but source archive is missing: {archive_path}")
            expected = (row.get("expected_sha256") or "").strip().lower()
            if len(expected) != 64:
                raise RuntimeError(f"Invalid expected SHA-256 for {month}: {expected!r}")
            source_bytes = int(row.get("source_bytes") or row.get("compressed_bytes") or archive_path.stat().st_size)
            rows.append(
                SourceMonth(
                    month=month,
                    expected_sha256=expected,
                    source_bytes_manifest=source_bytes,
                    archive_path=archive_path,
                    source_manifest_row=dict(row),
                )
            )

    if selected is not None:
        found = {row.month for row in rows}
        missing_selected = sorted(set(selected) - found)
        if missing_selected:
            raise RuntimeError(
                "Requested months are not VERIFIED/VERIFIED_REUSED in the Pass-1 manifest: " + ", ".join(missing_selected)
            )

    if not rows:
        raise RuntimeError("No VERIFIED/VERIFIED_REUSED Pass-1 archives matched the requested symbol/interval/month selection")
    return sorted(rows, key=lambda row: row.month)


def init_state(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS month_build_state (
            month TEXT PRIMARY KEY,
            input_signature TEXT NOT NULL,
            status TEXT NOT NULL,
            expected_sha256 TEXT NOT NULL,
            actual_sha256 TEXT NOT NULL,
            source_bytes INTEGER NOT NULL,
            rows INTEGER NOT NULL,
            first_open_time TEXT,
            last_open_time TEXT,
            parquet_path TEXT NOT NULL,
            parquet_bytes INTEGER NOT NULL,
            processed_at TEXT NOT NULL,
            error TEXT NOT NULL DEFAULT ''
        )
        """
    )
    conn.commit()
    return conn


def state_row(conn: sqlite3.Connection, month: str) -> dict[str, Any] | None:
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM month_build_state WHERE month = ?", (month,)).fetchone()
    return dict(row) if row else None


def save_state(conn: sqlite3.Connection, result: BuildResult) -> None:
    conn.execute(
        """
        INSERT INTO month_build_state (
            month,input_signature,status,expected_sha256,actual_sha256,source_bytes,rows,
            first_open_time,last_open_time,parquet_path,parquet_bytes,processed_at,error
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(month) DO UPDATE SET
            input_signature=excluded.input_signature,
            status=excluded.status,
            expected_sha256=excluded.expected_sha256,
            actual_sha256=excluded.actual_sha256,
            source_bytes=excluded.source_bytes,
            rows=excluded.rows,
            first_open_time=excluded.first_open_time,
            last_open_time=excluded.last_open_time,
            parquet_path=excluded.parquet_path,
            parquet_bytes=excluded.parquet_bytes,
            processed_at=excluded.processed_at,
            error=excluded.error
        """,
        (
            result.month,
            result.input_signature,
            result.status,
            result.expected_sha256,
            result.actual_sha256,
            result.source_bytes,
            result.rows,
            result.first_open_time,
            result.last_open_time,
            result.parquet_path,
            result.parquet_bytes,
            result.processed_at,
            result.error,
        ),
    )
    conn.commit()


def input_signature(source: SourceMonth, symbol: str, interval: str) -> str:
    payload = {
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "symbol": symbol,
        "interval": interval,
        "month": source.month,
        "expected_sha256": source.expected_sha256,
        "source_bytes_manifest": source.source_bytes_manifest,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def parquet_path_for(root: Path, symbol: str, month: str) -> Path:
    year, mon = month.split("-")
    return root / "parquet" / f"symbol={symbol}" / f"year={year}" / f"month={mon}" / "part-00000.parquet"


def _read_csv_member(zip_path: Path):
    import pandas as pd

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [name for name in zf.namelist() if name.lower().endswith(".csv") and not name.endswith("/")]
        if len(members) != 1:
            raise RuntimeError(f"Expected exactly one CSV inside {zip_path.name}, found {len(members)}")
        with zf.open(members[0], "r") as raw:
            payload = raw.read()

    # Binance futures archives are normally headerless, but accept a header row defensively.
    frame = pd.read_csv(io.BytesIO(payload), header=None, dtype=str)
    if frame.shape[1] != len(SOURCE_COLUMNS):
        raise RuntimeError(
            f"Unexpected Binance kline column count in {zip_path.name}: "
            f"expected {len(SOURCE_COLUMNS)}, found {frame.shape[1]}"
        )
    frame.columns = SOURCE_COLUMNS
    if len(frame) and not str(frame.iloc[0]["open_time_raw"]).strip().lstrip("-").isdigit():
        frame = frame.iloc[1:].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"No kline rows found in {zip_path.name}")
    return frame


def normalize_frame(raw, symbol: str, month: str):
    import numpy as np
    import pandas as pd

    frame = raw.copy()
    int_cols = ["open_time_raw", "close_time_raw", "trade_count"]
    float_cols = [
        "open", "high", "low", "close", "base_volume", "quote_volume",
        "taker_buy_base_volume", "taker_buy_quote_volume",
    ]

    for col in int_cols:
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype("int64")
    for col in float_cols:
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype("float64")

    # USD-M futures public archives use millisecond timestamps. Detect obvious unit drift rather than silently guessing.
    median_open = int(frame["open_time_raw"].median())
    if not (946684800000 <= median_open <= 4102444800000):  # 2000-01-01 through 2100-01-01 in ms
        raise RuntimeError(
            f"Unexpected timestamp unit/range for {month}: median open_time_raw={median_open}; "
            "expected Unix milliseconds for USD-M futures klines"
        )

    if frame["open_time_raw"].duplicated().any():
        dup = int(frame["open_time_raw"].duplicated().sum())
        raise RuntimeError(f"Duplicate open_time values in {month}: {dup}")

    if not frame["open_time_raw"].is_monotonic_increasing:
        raise RuntimeError(f"Source rows are not strictly chronological in {month}")

    # Structural validity only. Continuity/gap reconciliation belongs to Pass 3.
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():
        raise RuntimeError(f"OHLC high invariant failed in {month}")
    if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        raise RuntimeError(f"OHLC low invariant failed in {month}")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise RuntimeError(f"Non-positive price encountered in {month}")
    if (frame[["base_volume", "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]] < 0).any().any():
        raise RuntimeError(f"Negative volume encountered in {month}")
    if (frame["trade_count"] < 0).any():
        raise RuntimeError(f"Negative trade_count encountered in {month}")

    tol_base = np.maximum(1e-9, frame["base_volume"].abs().to_numpy() * 1e-10)
    tol_quote = np.maximum(1e-6, frame["quote_volume"].abs().to_numpy() * 1e-10)
    if np.any(frame["taker_buy_base_volume"].to_numpy() - frame["base_volume"].to_numpy() > tol_base):
        raise RuntimeError(f"Taker-buy base volume exceeds total base volume in {month}")
    if np.any(frame["taker_buy_quote_volume"].to_numpy() - frame["quote_volume"].to_numpy() > tol_quote):
        raise RuntimeError(f"Taker-buy quote volume exceeds total quote volume in {month}")

    if (frame["open_time_raw"] % 60_000 != 0).any():
        raise RuntimeError(f"One or more 1m open timestamps are not minute-aligned in {month}")
    expected_close = frame["open_time_raw"] + 59_999
    if not (frame["close_time_raw"] == expected_close).all():
        bad = int((frame["close_time_raw"] != expected_close).sum())
        raise RuntimeError(f"Unexpected 1m close_time convention in {month}: {bad} rows")

    start = pd.Timestamp(f"{month}-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    open_ts = pd.to_datetime(frame["open_time_raw"], unit="ms", utc=True)
    if ((open_ts < start) | (open_ts >= end)).any():
        raise RuntimeError(f"One or more rows fall outside source month {month}")

    out = pd.DataFrame(index=frame.index)
    out["symbol"] = symbol
    out["open_time"] = open_ts
    out["close_time"] = pd.to_datetime(frame["close_time_raw"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "base_volume", "quote_volume"]:
        out[col] = frame[col].astype("float64")
    out["trade_count"] = frame["trade_count"].astype("int64")
    out["taker_buy_base_volume"] = frame["taker_buy_base_volume"].astype("float64")
    out["taker_buy_quote_volume"] = frame["taker_buy_quote_volume"].astype("float64")
    out["taker_sell_base_volume"] = np.maximum(
        0.0, out["base_volume"].to_numpy() - out["taker_buy_base_volume"].to_numpy()
    )
    out["taker_sell_quote_volume"] = np.maximum(
        0.0, out["quote_volume"].to_numpy() - out["taker_buy_quote_volume"].to_numpy()
    )
    out["taker_buy_quote_share"] = np.where(
        out["quote_volume"].to_numpy() > 0,
        out["taker_buy_quote_volume"].to_numpy() / out["quote_volume"].to_numpy(),
        np.nan,
    )
    return out[OUTPUT_COLUMNS]


def atomic_write_parquet(path: Path, frame) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = short_tmp_path(path, "pq")
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(
            table,
            tmp,
            compression="zstd",
            compression_level=6,
            use_dictionary=["symbol"],
            write_statistics=True,
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def build_one(source: SourceMonth, root: Path, symbol: str, interval: str, force: bool) -> BuildResult:
    sig = input_signature(source, symbol, interval)
    final_path = parquet_path_for(root, symbol, source.month)
    actual_sha = ""
    now = utc_now_iso()
    try:
        actual_sha = sha256_file(source.archive_path)
        if actual_sha.lower() != source.expected_sha256.lower():
            raise RuntimeError(
                f"Source checksum mismatch: expected {source.expected_sha256}, actual {actual_sha}"
            )
        source_bytes = source.archive_path.stat().st_size
        if source.source_bytes_manifest and source_bytes != source.source_bytes_manifest:
            raise RuntimeError(
                f"Source byte-size mismatch: manifest={source.source_bytes_manifest}, actual={source_bytes}"
            )

        raw = _read_csv_member(source.archive_path)
        frame = normalize_frame(raw, symbol, source.month)
        atomic_write_parquet(final_path, frame)

        return BuildResult(
            month=source.month,
            status="COMPLETED",
            input_signature=sig,
            expected_sha256=source.expected_sha256,
            actual_sha256=actual_sha,
            source_bytes=source_bytes,
            rows=int(len(frame)),
            first_open_time=frame["open_time"].iloc[0].isoformat(),
            last_open_time=frame["open_time"].iloc[-1].isoformat(),
            parquet_path=str(final_path),
            parquet_bytes=final_path.stat().st_size,
            processed_at=now,
        )
    except Exception as exc:
        # A pending/rebuilt unit must never leave a stale previously-built Parquet
        # that no longer corresponds to the current verified source signature.
        final_path.unlink(missing_ok=True)
        return BuildResult(
            month=source.month,
            status="FAILED",
            input_signature=sig,
            expected_sha256=source.expected_sha256,
            actual_sha256=actual_sha,
            source_bytes=source.archive_path.stat().st_size if source.archive_path.exists() else 0,
            rows=0,
            first_open_time=None,
            last_open_time=None,
            parquet_path=str(final_path),
            parquet_bytes=0,
            processed_at=now,
            error=f"{type(exc).__name__}: {exc}",
        )


def maybe_reuse(conn: sqlite3.Connection, source: SourceMonth, root: Path, symbol: str, interval: str, force: bool) -> BuildResult | None:
    if force:
        return None
    sig = input_signature(source, symbol, interval)
    saved = state_row(conn, source.month)
    final_path = parquet_path_for(root, symbol, source.month)
    if not saved or saved.get("status") != "COMPLETED" or saved.get("input_signature") != sig:
        return None
    if not final_path.exists() or final_path.stat().st_size <= 0:
        return None
    # Revalidate the source checksum before reusing a normalized output.
    actual_sha = sha256_file(source.archive_path)
    if actual_sha.lower() != source.expected_sha256.lower():
        return None
    return BuildResult(
        month=source.month,
        status="COMPLETED_REUSED",
        input_signature=sig,
        expected_sha256=source.expected_sha256,
        actual_sha256=actual_sha,
        source_bytes=source.archive_path.stat().st_size,
        rows=int(saved["rows"]),
        first_open_time=saved.get("first_open_time"),
        last_open_time=saved.get("last_open_time"),
        parquet_path=str(final_path),
        parquet_bytes=final_path.stat().st_size,
        processed_at=utc_now_iso(),
    )


def all_state_results(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    return [dict(row) for row in conn.execute("SELECT * FROM month_build_state ORDER BY month")]


def write_metadata(root: Path, conn: sqlite3.Connection, run_id: str, selected_count: int) -> dict[str, Any]:
    metadata = root / "metadata"
    rows = all_state_results(conn)
    manifest_fields = [
        "month", "status", "expected_sha256", "actual_sha256", "source_bytes", "rows",
        "first_open_time", "last_open_time", "parquet_path", "parquet_bytes", "processed_at",
        "input_signature", "error",
    ]
    atomic_write_csv(metadata / "build_manifest.csv", manifest_fields, rows)

    completed = [row for row in rows if row["status"] == "COMPLETED"]
    failed = [row for row in rows if row["status"] == "FAILED"]
    summary = {
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": utc_now_iso(),
        "selected_months_this_run": selected_count,
        "months_in_state": len(rows),
        "completed_months": len(completed),
        "failed_months": len(failed),
        "first_completed_month": completed[0]["month"] if completed else None,
        "last_completed_month": completed[-1]["month"] if completed else None,
        "rows_written": int(sum(int(row["rows"]) for row in completed)),
        "source_compressed_bytes": int(sum(int(row["source_bytes"]) for row in completed)),
        "parquet_bytes": int(sum(int(row["parquet_bytes"]) for row in completed)),
        "source_compressed_gib": round(sum(int(row["source_bytes"]) for row in completed) / 1024**3, 4),
        "parquet_gib": round(sum(int(row["parquet_bytes"]) for row in completed) / 1024**3, 4),
        "failures": [{"month": row["month"], "error": row["error"]} for row in failed],
        "build_manifest": str(metadata / "build_manifest.csv"),
        "sqlite_state": str(metadata / "build_state.sqlite"),
        "parquet_root": str(root / "parquet"),
        "source_archives_modified": False,
        "missing_minutes_filled": False,
    }
    atomic_write_json(metadata / "latest_build_summary.json", summary)
    atomic_write_json(
        metadata / "dataset_config.json",
        {
            "builder_version": BUILDER_VERSION,
            "schema_version": SCHEMA_VERSION,
            "source_contract": "Pass-1 VERIFIED or VERIFIED_REUSED Binance USD-M monthly kline ZIPs",
            "output_columns": OUTPUT_COLUMNS,
            "compression": "zstd level 6",
            "partitioning": "symbol/year/month",
            "timestamp_unit_source": "Unix milliseconds",
            "timestamp_timezone_output": "UTC",
            "derived_fields": [
                "taker_sell_base_volume",
                "taker_sell_quote_volume",
                "taker_buy_quote_share",
            ],
            "continuity_policy": "Do not fill or repair missing minutes; Pass 3 audits continuity independently",
        },
    )
    return summary


def run_self_test() -> int:
    require_libraries()
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        month = "2020-01"
        zip_path = root / "archives" / "year=2020" / "BTCUSDT-1m-2020-01.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            [1577836800000, "7200", "7210", "7190", "7205", "10", 1577836859999, "72050", 100, "6", "43230", "0"],
            [1577836860000, "7205", "7220", "7200", "7215", "8", 1577836919999, "57720", 90, "5", "36075", "0"],
        ]
        csv_payload = "\n".join(",".join(map(str, row)) for row in rows) + "\n"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("BTCUSDT-1m-2020-01.csv", csv_payload)
        digest = sha256_file(zip_path)
        manifest = root / "metadata" / "download_manifest.csv"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["symbol", "interval", "month", "status", "expected_sha256", "archive_path", "source_bytes"],
            )
            writer.writeheader()
            writer.writerow({
                "symbol": "BTCUSDT", "interval": "1m", "month": month, "status": "VERIFIED_REUSED",
                "expected_sha256": digest, "archive_path": str(zip_path), "source_bytes": zip_path.stat().st_size,
            })

        source = load_verified_sources(root, "BTCUSDT", "1m", [month])[0]
        raw = _read_csv_member(source.archive_path)
        normalized = normalize_frame(raw, "BTCUSDT", month)
        assert len(normalized) == 2
        assert abs(float(normalized.iloc[0]["taker_sell_base_volume"]) - 4.0) < 1e-12
        assert abs(float(normalized.iloc[0]["taker_buy_quote_share"]) - (43230 / 72050)) < 1e-12
        out = parquet_path_for(root, "BTCUSDT", month)
        atomic_write_parquet(out, normalized)
        roundtrip = pd.read_parquet(out)
        assert len(roundtrip) == 2 and list(roundtrip.columns) == OUTPUT_COLUMNS

        deep = root / ("x" * 100) / ("y" * 100) / "metadata.json"
        atomic_write_json(deep, {"ok": True})
        assert json.loads(deep.read_text(encoding="utf-8"))["ok"] is True

    print("SELF-TEST PASS")
    print("Validated VERIFIED/VERIFIED_REUSED source ingestion, checksum enforcement, Binance 1m schema normalization, deterministic sell/share fields, Parquet round-trip and short-path atomic writers.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Pass-1 historical root; Parquet is written beneath this root")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--months", help="Optional comma-separated YYYY-MM subset; every requested month must be VERIFIED in Pass 1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Rebuild selected months even when their signatures already match")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_test()
    require_libraries()
    if args.interval != "1m":
        raise RuntimeError("Pass 2 is intentionally frozen to the 1m interval")
    if args.workers < 1 or args.workers > 16:
        raise RuntimeError("--workers must be between 1 and 16")

    selected = parse_months(args.months)
    root = args.root.resolve()
    sources = load_verified_sources(root, args.symbol, args.interval, selected)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]

    print("Building normalized Binance USD-M 1m kline Parquet...")
    print(f"Builder version  : {BUILDER_VERSION}")
    print(f"Run ID           : {run_id}")
    print(f"Symbol           : {args.symbol}")
    print(f"Interval         : {args.interval}")
    print(f"Verified sources : {len(sources)}")
    print(f"Root             : {root}")
    print(f"Workers          : {args.workers}")
    print("Source ZIPs will be re-hashed and will not be modified.")
    print("Missing minutes will not be filled or repaired.")

    state_path = root / "metadata" / "build_state.sqlite"
    conn = init_state(state_path)
    try:
        pending: list[SourceMonth] = []
        reused: list[BuildResult] = []
        for source in sources:
            result = maybe_reuse(conn, source, root, args.symbol, args.interval, args.force)
            if result is None:
                pending.append(source)
            else:
                reused.append(result)

        completed_this_run = 0
        failed_this_run = 0
        counter = 0
        total = len(sources)
        for result in reused:
            counter += 1
            print(f"[{counter:02d}/{total:02d}] {result.month}: COMPLETED_REUSED ({result.rows:,} rows)")

        if pending:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(build_one, source, root, args.symbol, args.interval, args.force): source
                    for source in pending
                }
                for future in as_completed(futures):
                    result = future.result()
                    # Persist only stable state values; reused is a display status, not a persisted state.
                    if result.status == "COMPLETED":
                        save_state(conn, result)
                        completed_this_run += 1
                    else:
                        save_state(conn, result)
                        failed_this_run += 1
                    counter += 1
                    if result.status == "COMPLETED":
                        print(
                            f"[{counter:02d}/{total:02d}] {result.month}: COMPLETED "
                            f"({result.rows:,} rows, {result.parquet_bytes / 1024**2:.2f} MiB Parquet)"
                        )
                    else:
                        print(f"[{counter:02d}/{total:02d}] {result.month}: FAILED - {result.error}")

        summary = write_metadata(root, conn, run_id, len(sources))
    finally:
        conn.close()

    print("\nBuild pass summary")
    print("------------------")
    print(f"Completed months : {summary['completed_months']}")
    print(f"Failed months    : {summary['failed_months']}")
    print(f"Total rows       : {summary['rows_written']:,}")
    print(f"First completed  : {summary['first_completed_month']}")
    print(f"Last completed   : {summary['last_completed_month']}")
    print(f"Source GiB       : {summary['source_compressed_gib']}")
    print(f"Parquet GiB      : {summary['parquet_gib']}")
    print(f"Manifest         : {summary['build_manifest']}")
    print(f"Summary          : {root / 'metadata' / 'latest_build_summary.json'}")
    print(f"SQLite state     : {summary['sqlite_state']}")

    return 1 if summary["failed_months"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
