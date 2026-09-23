#!/usr/bin/env python3
"""Independently audit normalized Binance USD-M BTCUSDT monthly 1m klines.

Pass 3 of the historical-context pipeline.

Primary contracts
-----------------
* Treat Pass-2 Parquet as an output to verify, not as trusted truth.
* Re-hash every original Pass-1 ZIP and reconcile it to the published checksum.
* Re-read the CSV inside each ZIP and compare source fields with Parquet fields.
* Reconstruct every expected minute from calendar boundaries; never fill gaps.
* Validate schema, timestamps, OHLC, volumes, deterministic sell fields and shares.
* Check cross-month continuity independently.
* Optionally compare the overlapping period with the repository's audited 1s data.
* Historical-data violations are FAILs. Collector-overlap limitations are WARNs.
* Use SQLite state and signatures for resumability.
* Use short same-directory temporary filenames for Windows path safety.

Default historical root:
    D:\\stream-data\\historical\\binance\\futures_um\\BTCUSDT\\klines_1m

Expected Pass-1/Pass-2 inputs:
    metadata/download_manifest.csv
    metadata/build_manifest.csv
    archives/year=YYYY/BTCUSDT-1m-YYYY-MM.zip
    parquet/symbol=BTCUSDT/year=YYYY/month=MM/part-00000.parquet

Default repository overlap candidates:
    data/derived/perps/core_1s_v1
    data/derived/perps/flow_features_1s_v1

Outputs under historical root:
    metadata/audit_state.sqlite
    metadata/audit_units/month=YYYY-MM/result.json
    metadata/latest_audit_summary.json

Reports under repository:
    reports/binance_um_klines_1m_audit/
        binance_um_klines_1m_monthly_audit.csv
        binance_um_klines_1m_gaps.csv
        binance_um_klines_1m_zero_volume.csv
        binance_um_klines_1m_source_reconciliation.csv
        binance_um_klines_1m_overlap_daily.csv
        binance_um_klines_1m_overlap_summary.json
        binance_um_klines_1m_invariants.csv
        binance_um_klines_1m_audit_config.json
        binance_um_klines_1m_findings.md
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

AUDITOR_VERSION = "1.0.1"
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

NUMERIC_SOURCE_TO_OUTPUT = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "base_volume": "base_volume",
    "quote_volume": "quote_volume",
    "taker_buy_base_volume": "taker_buy_base_volume",
    "taker_buy_quote_volume": "taker_buy_quote_volume",
}


@dataclass(frozen=True)
class MonthInput:
    month: str
    expected_sha256: str
    archive_path: Path
    parquet_path: Path
    source_bytes_manifest: int
    build_rows_manifest: int
    build_parquet_bytes_manifest: int


@dataclass
class MonthAudit:
    month: str
    status: str
    input_signature: str
    rows: int
    expected_rows: int
    first_open_time: str
    last_open_time: str
    missing_minutes: int
    duplicate_minutes: int
    unexpected_minutes: int
    non_1m_steps: int
    zero_base_volume_rows: int
    zero_quote_volume_rows: int
    null_share_rows: int
    source_rows: int
    source_parquet_row_match: bool
    source_field_mismatch_rows: int
    source_sha256_match: bool
    parquet_bytes: int
    source_bytes: int
    audited_at: str
    error: str = ""


@dataclass
class Invariant:
    scope: str
    invariant: str
    status: str
    observed: Any
    expected: Any
    detail: str = ""


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


def require_libraries() -> None:
    try:
        import numpy as np  # noqa: F401
        import pandas as pd  # noqa: F401
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Pass 3 requires numpy, pandas and pyarrow. Install with: "
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


def find_repo_root(script_path: Path) -> Path:
    # scripts/perps/historical/03_...py -> repo root is parents[3]
    try:
        candidate = script_path.resolve().parents[3]
        if (candidate / "scripts").exists():
            return candidate
    except IndexError:
        pass
    return Path.cwd().resolve()


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"Required manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_month_inputs(root: Path, symbol: str, interval: str, selected: list[str] | None) -> list[MonthInput]:
    download_rows = _read_manifest(root / "metadata" / "download_manifest.csv")
    build_rows = _read_manifest(root / "metadata" / "build_manifest.csv")

    dmap: dict[str, dict[str, str]] = {}
    for row in download_rows:
        if row.get("symbol") != symbol or row.get("interval") != interval:
            continue
        if row.get("status") not in {"VERIFIED", "VERIFIED_REUSED"}:
            continue
        dmap[row.get("month", "")] = row

    bmap: dict[str, dict[str, str]] = {}
    for row in build_rows:
        month = row.get("month", "")
        status = row.get("status", "")
        if status not in {"COMPLETED", "COMPLETED_REUSED"}:
            continue
        bmap[month] = row

    months = sorted(set(dmap) & set(bmap))
    if selected is not None:
        missing = sorted(set(selected) - set(months))
        if missing:
            raise RuntimeError(
                "Requested months are not simultaneously VERIFIED in Pass 1 and COMPLETED in Pass 2: "
                + ", ".join(missing)
            )
        months = selected
    if not months:
        raise RuntimeError("No months are available for auditing")

    out: list[MonthInput] = []
    for month in months:
        drow = dmap[month]
        brow = bmap[month]
        archive_path = Path(drow.get("archive_path", ""))
        parquet_path = Path(brow.get("parquet_path", ""))
        if not archive_path.exists():
            raise RuntimeError(f"Verified source archive is missing for {month}: {archive_path}")
        if not parquet_path.exists():
            raise RuntimeError(f"Completed Parquet output is missing for {month}: {parquet_path}")
        expected = (drow.get("expected_sha256") or "").strip().lower()
        if len(expected) != 64:
            raise RuntimeError(f"Invalid expected SHA-256 for {month}: {expected!r}")
        out.append(
            MonthInput(
                month=month,
                expected_sha256=expected,
                archive_path=archive_path,
                parquet_path=parquet_path,
                source_bytes_manifest=int(drow.get("source_bytes") or drow.get("compressed_bytes") or 0),
                build_rows_manifest=int(brow.get("rows") or 0),
                build_parquet_bytes_manifest=int(brow.get("parquet_bytes") or 0),
            )
        )
    return out


def init_state(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS month_audit (
            month TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            input_signature TEXT NOT NULL,
            result_json TEXT NOT NULL,
            audited_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def month_signature(item: MonthInput) -> str:
    payload = {
        "auditor_version": AUDITOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "month": item.month,
        "expected_sha256": item.expected_sha256,
        "source_bytes": item.archive_path.stat().st_size,
        "source_mtime_ns": item.archive_path.stat().st_mtime_ns,
        "parquet_bytes": item.parquet_path.stat().st_size,
        "parquet_mtime_ns": item.parquet_path.stat().st_mtime_ns,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def audit_unit_path(root: Path, month: str) -> Path:
    return root / "metadata" / "audit_units" / f"month={month}" / "result.json"


def load_reusable_bundle(
    conn: sqlite3.Connection, root: Path, item: MonthInput, force: bool
) -> tuple[MonthAudit, list[dict[str, Any]], list[dict[str, Any]], list[Invariant]] | None:
    if force:
        return None
    sig = month_signature(item)
    row = conn.execute(
        "SELECT status, input_signature FROM month_audit WHERE month=?", (item.month,)
    ).fetchone()
    unit_path = audit_unit_path(root, item.month)
    if not row or row[0] != "PASS" or row[1] != sig or not unit_path.exists():
        return None
    try:
        payload = json.loads(unit_path.read_text(encoding="utf-8"))
        result = MonthAudit(**payload["result"])
        gaps = list(payload.get("gaps", []))
        zero_rows = list(payload.get("zero_rows", []))
        invariants = [Invariant(**x) for x in payload.get("invariants", [])]
        if result.input_signature != sig or result.status != "PASS":
            return None
        return result, gaps, zero_rows, invariants
    except Exception:
        return None


def save_state(
    conn: sqlite3.Connection,
    root: Path,
    result: MonthAudit,
    gaps: list[dict[str, Any]],
    zero_rows: list[dict[str, Any]],
    invariants: list[Invariant],
) -> None:
    conn.execute(
        """
        INSERT INTO month_audit(month,status,input_signature,result_json,audited_at)
        VALUES(?,?,?,?,?)
        ON CONFLICT(month) DO UPDATE SET
            status=excluded.status,
            input_signature=excluded.input_signature,
            result_json=excluded.result_json,
            audited_at=excluded.audited_at
        """,
        (result.month, result.status, result.input_signature, json.dumps(asdict(result), sort_keys=True), result.audited_at),
    )
    conn.commit()
    atomic_write_json(
        audit_unit_path(root, result.month),
        {
            "auditor_version": AUDITOR_VERSION,
            "month": result.month,
            "input_signature": result.input_signature,
            "result": asdict(result),
            "gaps": gaps,
            "zero_rows": zero_rows,
            "invariants": [asdict(x) for x in invariants],
        },
    )


def _read_source_csv(zip_path: Path):
    import pandas as pd

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv") and not n.endswith("/")]
        if len(members) != 1:
            raise RuntimeError(f"Expected exactly one CSV inside {zip_path.name}, found {len(members)}")
        with zf.open(members[0], "r") as raw:
            payload = raw.read()
    frame = pd.read_csv(io.BytesIO(payload), header=None, dtype=str)
    if frame.shape[1] != len(SOURCE_COLUMNS):
        raise RuntimeError(
            f"Unexpected Binance source column count in {zip_path.name}: "
            f"expected {len(SOURCE_COLUMNS)}, found {frame.shape[1]}"
        )
    frame.columns = SOURCE_COLUMNS
    if len(frame) and not str(frame.iloc[0]["open_time_raw"]).strip().lstrip("-").isdigit():
        frame = frame.iloc[1:].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"No source rows found in {zip_path.name}")
    return frame


def month_bounds(month: str):
    import pandas as pd

    start = pd.Timestamp(f"{month}-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    return start, end


def expected_minute_ms(month: str):
    import numpy as np

    start, end = month_bounds(month)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    return np.arange(start_ms, end_ms, 60_000, dtype="int64")


def _float_mismatch(a, b, atol: float = 1e-9, rtol: float = 1e-11):
    import numpy as np

    aa = np.asarray(a, dtype="float64")
    bb = np.asarray(b, dtype="float64")
    return ~(np.isclose(aa, bb, atol=atol, rtol=rtol, equal_nan=True))


def datetime_series_to_epoch_ms(series):
    """Convert a timezone-aware pandas datetime Series to Unix epoch milliseconds.

    PyArrow can round-trip Parquet timestamps at their stored resolution (for
    example ``datetime64[ms, UTC]``).  ``Series.astype("int64")`` therefore
    returns integers in the Series' own resolution, not always nanoseconds.
    Normalize explicitly from that resolution so continuity checks are correct
    for ns/us/ms/s timestamp dtypes.
    """
    import numpy as np

    dtype = series.dtype
    unit = getattr(dtype, "unit", None)
    raw = series.astype("int64").to_numpy(dtype="int64")
    if unit == "ns":
        return raw // 1_000_000
    if unit == "us":
        return raw // 1_000
    if unit == "ms":
        return raw
    if unit == "s":
        return raw * 1_000
    raise RuntimeError(f"Unsupported datetime resolution for epoch-ms conversion: {dtype!s}")


def audit_month(item: MonthInput, symbol: str) -> tuple[MonthAudit, list[dict[str, Any]], list[dict[str, Any]], list[Invariant]]:
    import numpy as np
    import pandas as pd

    now = utc_now_iso()
    sig = month_signature(item)
    gaps: list[dict[str, Any]] = []
    zero_rows: list[dict[str, Any]] = []
    invariants: list[Invariant] = []
    try:
        actual_sha = sha256_file(item.archive_path)
        sha_ok = actual_sha.lower() == item.expected_sha256.lower()
        invariants.append(Invariant(item.month, "source_sha256_matches_pass1", "PASS" if sha_ok else "FAIL", actual_sha, item.expected_sha256))
        if not sha_ok:
            raise RuntimeError(f"Source SHA-256 mismatch: expected {item.expected_sha256}, got {actual_sha}")

        if item.source_bytes_manifest:
            size_ok = item.archive_path.stat().st_size == item.source_bytes_manifest
            invariants.append(Invariant(item.month, "source_byte_size_matches_manifest", "PASS" if size_ok else "FAIL", item.archive_path.stat().st_size, item.source_bytes_manifest))
            if not size_ok:
                raise RuntimeError("Source byte-size no longer matches Pass-1 manifest")

        pq = pd.read_parquet(item.parquet_path)
        columns_ok = list(pq.columns) == OUTPUT_COLUMNS
        invariants.append(Invariant(item.month, "parquet_schema_columns_exact", "PASS" if columns_ok else "FAIL", list(pq.columns), OUTPUT_COLUMNS))
        if not columns_ok:
            raise RuntimeError(f"Unexpected Parquet columns: {list(pq.columns)}")

        if pq.empty:
            raise RuntimeError("Parquet month is empty")
        symbol_ok = bool((pq["symbol"] == symbol).all())
        invariants.append(Invariant(item.month, "symbol_constant", "PASS" if symbol_ok else "FAIL", sorted(pq["symbol"].astype(str).unique().tolist()), [symbol]))
        if not symbol_ok:
            raise RuntimeError("Unexpected symbol value in Parquet")

        for tcol in ["open_time", "close_time"]:
            if not pd.api.types.is_datetime64_any_dtype(pq[tcol]):
                raise RuntimeError(f"{tcol} is not datetime typed")
            if pq[tcol].dt.tz is None:
                raise RuntimeError(f"{tcol} is timezone-naive")

        open_ms = datetime_series_to_epoch_ms(pq["open_time"])
        close_ms = datetime_series_to_epoch_ms(pq["close_time"])
        expected_ms = expected_minute_ms(item.month)
        expected_rows = int(len(expected_ms))
        rows = int(len(pq))

        duplicate_count = int(pd.Series(open_ms).duplicated().sum())
        if len(open_ms) > 1:
            diffs = np.diff(open_ms)
            non_1m_steps = int(np.sum(diffs != 60_000))
        else:
            non_1m_steps = 0
        actual_unique = np.unique(open_ms)
        missing = np.setdiff1d(expected_ms, actual_unique, assume_unique=True)
        unexpected = np.setdiff1d(actual_unique, expected_ms, assume_unique=True)
        for ts in missing:
            gaps.append({"month": item.month, "kind": "MISSING", "open_time": pd.to_datetime(int(ts), unit="ms", utc=True).isoformat()})
        for ts in unexpected:
            gaps.append({"month": item.month, "kind": "UNEXPECTED", "open_time": pd.to_datetime(int(ts), unit="ms", utc=True).isoformat()})

        checks = [
            ("row_count_matches_calendar", rows == expected_rows, rows, expected_rows),
            ("no_duplicate_minutes", duplicate_count == 0, duplicate_count, 0),
            ("strict_one_minute_spacing", non_1m_steps == 0, non_1m_steps, 0),
            ("no_missing_minutes", len(missing) == 0, len(missing), 0),
            ("no_unexpected_minutes", len(unexpected) == 0, len(unexpected), 0),
            ("first_minute_exact", int(open_ms[0]) == int(expected_ms[0]), int(open_ms[0]), int(expected_ms[0])),
            ("last_minute_exact", int(open_ms[-1]) == int(expected_ms[-1]), int(open_ms[-1]), int(expected_ms[-1])),
            ("close_time_convention", bool(np.all(close_ms == open_ms + 59_999)), int(np.sum(close_ms != open_ms + 59_999)), 0),
        ]
        for name, ok, obs, exp in checks:
            invariants.append(Invariant(item.month, name, "PASS" if ok else "FAIL", obs, exp))
        if not all(x[1] for x in checks):
            raise RuntimeError("Timestamp/continuity invariant failed")

        prices = pq[["open", "high", "low", "close"]]
        ohlc_ok = bool(
            (prices > 0).all().all()
            and (pq["high"] >= pq[["open", "close", "low"]].max(axis=1)).all()
            and (pq["low"] <= pq[["open", "close", "high"]].min(axis=1)).all()
        )
        invariants.append(Invariant(item.month, "ohlc_valid", "PASS" if ohlc_ok else "FAIL", ohlc_ok, True))
        if not ohlc_ok:
            raise RuntimeError("OHLC invariant failed")

        volume_cols = ["base_volume", "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume", "taker_sell_base_volume", "taker_sell_quote_volume"]
        volume_ok = bool((pq[volume_cols] >= -1e-12).all().all())
        trade_ok = bool((pq["trade_count"] >= 0).all())
        buy_base_ok = bool((pq["taker_buy_base_volume"] <= pq["base_volume"] + np.maximum(1e-9, pq["base_volume"].abs() * 1e-10)).all())
        buy_quote_ok = bool((pq["taker_buy_quote_volume"] <= pq["quote_volume"] + np.maximum(1e-6, pq["quote_volume"].abs() * 1e-10)).all())
        for name, ok in [
            ("volumes_nonnegative", volume_ok),
            ("trade_count_nonnegative", trade_ok),
            ("taker_buy_base_not_above_total", buy_base_ok),
            ("taker_buy_quote_not_above_total", buy_quote_ok),
        ]:
            invariants.append(Invariant(item.month, name, "PASS" if ok else "FAIL", ok, True))
        if not (volume_ok and trade_ok and buy_base_ok and buy_quote_ok):
            raise RuntimeError("Volume/trade invariant failed")

        expected_sell_base = pq["base_volume"].to_numpy() - pq["taker_buy_base_volume"].to_numpy()
        expected_sell_quote = pq["quote_volume"].to_numpy() - pq["taker_buy_quote_volume"].to_numpy()
        sell_base_bad = int(_float_mismatch(pq["taker_sell_base_volume"], expected_sell_base, atol=1e-9).sum())
        sell_quote_bad = int(_float_mismatch(pq["taker_sell_quote_volume"], expected_sell_quote, atol=1e-6).sum())
        invariants.append(Invariant(item.month, "derived_sell_base_reconciles", "PASS" if sell_base_bad == 0 else "FAIL", sell_base_bad, 0))
        invariants.append(Invariant(item.month, "derived_sell_quote_reconciles", "PASS" if sell_quote_bad == 0 else "FAIL", sell_quote_bad, 0))
        if sell_base_bad or sell_quote_bad:
            raise RuntimeError("Derived sell-volume reconciliation failed")

        qvol = pq["quote_volume"].to_numpy(dtype="float64")
        qbuy = pq["taker_buy_quote_volume"].to_numpy(dtype="float64")
        share = pq["taker_buy_quote_share"].to_numpy(dtype="float64")
        zero_q = qvol == 0
        expected_share = np.full(len(pq), np.nan, dtype="float64")
        np.divide(qbuy, qvol, out=expected_share, where=~zero_q)
        share_bad = int(_float_mismatch(share, expected_share, atol=1e-12, rtol=1e-11).sum())
        zero_share_nonnull = int(np.sum(zero_q & ~np.isnan(share)))
        nonzero_share_invalid = int(np.sum((~zero_q) & (~np.isfinite(share) | (share < -1e-12) | (share > 1 + 1e-12))))
        invariants.append(Invariant(item.month, "taker_buy_quote_share_reconciles", "PASS" if share_bad == 0 else "FAIL", share_bad, 0))
        invariants.append(Invariant(item.month, "zero_quote_volume_share_is_null", "PASS" if zero_share_nonnull == 0 else "FAIL", zero_share_nonnull, 0))
        invariants.append(Invariant(item.month, "nonzero_quote_share_bounded", "PASS" if nonzero_share_invalid == 0 else "FAIL", nonzero_share_invalid, 0))
        if share_bad or zero_share_nonnull or nonzero_share_invalid:
            raise RuntimeError("Taker-buy quote-share reconciliation failed")

        zero_base = pq["base_volume"].to_numpy(dtype="float64") == 0
        zero_quote_indices = np.flatnonzero(zero_q)
        for idx in zero_quote_indices[:1000]:
            zero_rows.append({
                "month": item.month,
                "open_time": pq.iloc[int(idx)]["open_time"].isoformat(),
                "base_volume": float(pq.iloc[int(idx)]["base_volume"]),
                "quote_volume": float(pq.iloc[int(idx)]["quote_volume"]),
                "trade_count": int(pq.iloc[int(idx)]["trade_count"]),
                "taker_buy_quote_share": None if pd.isna(pq.iloc[int(idx)]["taker_buy_quote_share"]) else float(pq.iloc[int(idx)]["taker_buy_quote_share"]),
            })

        raw = _read_source_csv(item.archive_path)
        source_rows = int(len(raw))
        source_open = pd.to_numeric(raw["open_time_raw"], errors="raise").astype("int64").to_numpy()
        source_close = pd.to_numeric(raw["close_time_raw"], errors="raise").astype("int64").to_numpy()
        source_row_match = source_rows == rows
        mismatch_mask = np.zeros(rows, dtype=bool) if source_row_match else np.ones(max(rows, source_rows), dtype=bool)
        if source_row_match:
            mismatch_mask |= source_open != open_ms
            mismatch_mask |= source_close != close_ms
            src_trade = pd.to_numeric(raw["trade_count"], errors="raise").astype("int64").to_numpy()
            mismatch_mask |= src_trade != pq["trade_count"].to_numpy(dtype="int64")
            for scol, pcol in NUMERIC_SOURCE_TO_OUTPUT.items():
                src = pd.to_numeric(raw[scol], errors="raise").astype("float64").to_numpy()
                mismatch_mask |= _float_mismatch(src, pq[pcol].to_numpy(dtype="float64"), atol=1e-9, rtol=1e-11)
        mismatch_rows = int(mismatch_mask.sum())
        invariants.append(Invariant(item.month, "source_parquet_row_count_match", "PASS" if source_row_match else "FAIL", source_rows, rows))
        invariants.append(Invariant(item.month, "source_fields_reconcile_to_parquet", "PASS" if mismatch_rows == 0 else "FAIL", mismatch_rows, 0))
        if not source_row_match or mismatch_rows:
            raise RuntimeError(f"Source-to-Parquet reconciliation failed: mismatch_rows={mismatch_rows}")

        manifest_rows_ok = item.build_rows_manifest in {0, rows}
        manifest_bytes_ok = item.build_parquet_bytes_manifest in {0, item.parquet_path.stat().st_size}
        invariants.append(Invariant(item.month, "pass2_manifest_row_count_match", "PASS" if manifest_rows_ok else "FAIL", item.build_rows_manifest, rows))
        invariants.append(Invariant(item.month, "pass2_manifest_parquet_bytes_match", "PASS" if manifest_bytes_ok else "FAIL", item.build_parquet_bytes_manifest, item.parquet_path.stat().st_size))
        if not (manifest_rows_ok and manifest_bytes_ok):
            raise RuntimeError("Pass-2 manifest reconciliation failed")

        result = MonthAudit(
            month=item.month,
            status="PASS",
            input_signature=sig,
            rows=rows,
            expected_rows=expected_rows,
            first_open_time=pq["open_time"].iloc[0].isoformat(),
            last_open_time=pq["open_time"].iloc[-1].isoformat(),
            missing_minutes=int(len(missing)),
            duplicate_minutes=duplicate_count,
            unexpected_minutes=int(len(unexpected)),
            non_1m_steps=non_1m_steps,
            zero_base_volume_rows=int(zero_base.sum()),
            zero_quote_volume_rows=int(zero_q.sum()),
            null_share_rows=int(pd.isna(pq["taker_buy_quote_share"]).sum()),
            source_rows=source_rows,
            source_parquet_row_match=source_row_match,
            source_field_mismatch_rows=mismatch_rows,
            source_sha256_match=sha_ok,
            parquet_bytes=item.parquet_path.stat().st_size,
            source_bytes=item.archive_path.stat().st_size,
            audited_at=now,
        )
        return result, gaps, zero_rows, invariants
    except Exception as exc:
        result = MonthAudit(
            month=item.month,
            status="FAIL",
            input_signature=sig,
            rows=0,
            expected_rows=int(len(expected_minute_ms(item.month))),
            first_open_time="",
            last_open_time="",
            missing_minutes=0,
            duplicate_minutes=0,
            unexpected_minutes=0,
            non_1m_steps=0,
            zero_base_volume_rows=0,
            zero_quote_volume_rows=0,
            null_share_rows=0,
            source_rows=0,
            source_parquet_row_match=False,
            source_field_mismatch_rows=0,
            source_sha256_match=False,
            parquet_bytes=item.parquet_path.stat().st_size if item.parquet_path.exists() else 0,
            source_bytes=item.archive_path.stat().st_size if item.archive_path.exists() else 0,
            audited_at=now,
            error=f"{type(exc).__name__}: {exc}",
        )
        invariants.append(Invariant(item.month, "month_audit_completed", "FAIL", result.error, "PASS"))
        return result, gaps, zero_rows, invariants


def parquet_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.parquet") if p.is_file())


def _first_parquet_columns(files: list[Path]) -> list[str]:
    import pyarrow.parquet as pq

    if not files:
        return []
    return pq.ParquetFile(files[0]).schema_arrow.names


def choose_column(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def detect_overlap_source(repo_root: Path, explicit: Path | None):
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend([
        repo_root / "data" / "derived" / "perps" / "core_1s_v1",
        repo_root / "data" / "derived" / "perps" / "flow_features_1s_v1",
    ])
    seen: set[str] = set()
    diagnostics: list[dict[str, Any]] = []
    for root in candidates:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        files = parquet_files(root)
        if not files:
            diagnostics.append({"root": str(root), "status": "NO_PARQUET"})
            continue
        cols = _first_parquet_columns(files)
        time_col = choose_column(cols, ["receive_second", "event_second", "second", "timestamp", "time"])
        # Prefer explicitly trade-derived fields; price_for_features is allowed as a lower-confidence reference.
        strict_price = choose_column(cols, ["last_trade_price", "trade_price", "trade_last_price", "last_price"])
        ref_price = strict_price or choose_column(cols, ["price_for_features"])
        if time_col and ref_price:
            return {
                "root": root,
                "files": files,
                "columns": cols,
                "time_col": time_col,
                "price_col": ref_price,
                "price_semantics": "TRADE_DERIVED" if strict_price else "REFERENCE_PRICE_FOR_FEATURES",
                "trade_count_col": choose_column(cols, ["trade_count", "trades", "n_trades"]),
                "base_volume_col": choose_column(cols, ["base_volume", "trade_base_volume", "volume_base"]),
                "quote_volume_col": choose_column(cols, ["quote_volume", "trade_quote_volume", "volume_quote"]),
            }, diagnostics
        diagnostics.append({"root": str(root), "status": "NO_COMPARABLE_TIME_PRICE", "columns": cols})
    return None, diagnostics


def read_parquet_files(files: list[Path], columns: list[str]):
    import pandas as pd

    frames = []
    for file in files:
        frames.append(pd.read_parquet(file, columns=columns))
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def audit_overlap(hist_root: Path, repo_root: Path, symbol: str, explicit: Path | None):
    import numpy as np
    import pandas as pd
    source, diagnostics = detect_overlap_source(repo_root, explicit)
    if source is None:
        return {
            "status": "WARN",
            "reason": "No repository 1s dataset with a comparable timestamp and trade/reference price column was found",
            "search_diagnostics": diagnostics,
        }, []

    hist_cols = ["open_time", "open", "high", "low", "close", "base_volume", "quote_volume", "trade_count"]
    hist_files = parquet_files(hist_root / "parquet" / f"symbol={symbol}")
    hist = read_parquet_files(hist_files, hist_cols)
    if hist["open_time"].dt.tz is None:
        hist["open_time"] = hist["open_time"].dt.tz_localize("UTC")
    else:
        hist["open_time"] = hist["open_time"].dt.tz_convert("UTC")

    # Use the actual repository overlap rather than assuming May 2026 dates.
    # Read reference once, then intersect time ranges.
    # This dataset is only ~5.3m 1s rows in the current project and is manageable.
    cols = [source["time_col"], source["price_col"]]
    for c in [source.get("trade_count_col"), source.get("base_volume_col"), source.get("quote_volume_col")]:
        if c and c not in cols:
            cols.append(c)
    ref = read_parquet_files(source["files"], cols)
    tcol = source["time_col"]
    if not pd.api.types.is_datetime64_any_dtype(ref[tcol]):
        numeric = pd.to_numeric(ref[tcol], errors="coerce")
        if numeric.notna().all():
            med = float(numeric.median())
            unit = "ms" if med > 1e11 else "s"
            ref[tcol] = pd.to_datetime(numeric, unit=unit, utc=True)
        else:
            ref[tcol] = pd.to_datetime(ref[tcol], utc=True, errors="raise")
    else:
        if ref[tcol].dt.tz is None:
            ref[tcol] = ref[tcol].dt.tz_localize("UTC")
        else:
            ref[tcol] = ref[tcol].dt.tz_convert("UTC")
    ref = ref.dropna(subset=[tcol, source["price_col"]]).sort_values(tcol)
    if ref.empty:
        return {"status": "WARN", "reason": "Comparable repository dataset was empty after timestamp/price filtering", "source_root": str(source["root"])}, []

    start = max(hist["open_time"].min(), ref[tcol].min().floor("min"))
    end = min(hist["open_time"].max() + pd.Timedelta(minutes=1), ref[tcol].max().floor("min") + pd.Timedelta(minutes=1))
    if start >= end:
        return {"status": "WARN", "reason": "No chronological overlap with repository 1s dataset", "source_root": str(source["root"])}, []

    ref = ref[(ref[tcol] >= start) & (ref[tcol] < end)].copy()
    hist = hist[(hist["open_time"] >= start) & (hist["open_time"] < end)].copy()
    ref["minute"] = ref[tcol].dt.floor("min")

    price_col = source["price_col"]
    grouped = ref.groupby("minute", sort=True)
    ref_1m = grouped[price_col].agg(["first", "max", "min", "last"]).rename(columns={"first": "ref_open", "max": "ref_high", "min": "ref_low", "last": "ref_close"}).reset_index()

    # Only aggregate additional metrics when their semantics are present.
    if source.get("trade_count_col"):
        ref_1m["ref_trade_count_sum"] = grouped[source["trade_count_col"]].sum().to_numpy()
    if source.get("base_volume_col"):
        ref_1m["ref_base_volume_sum"] = grouped[source["base_volume_col"]].sum().to_numpy()
    if source.get("quote_volume_col"):
        ref_1m["ref_quote_volume_sum"] = grouped[source["quote_volume_col"]].sum().to_numpy()

    merged = hist.merge(ref_1m, left_on="open_time", right_on="minute", how="inner")
    if merged.empty:
        return {"status": "WARN", "reason": "No exact minute joins with repository 1s dataset", "source_root": str(source["root"])}, []

    for kline, rcol in [("open", "ref_open"), ("high", "ref_high"), ("low", "ref_low"), ("close", "ref_close")]:
        merged[f"{kline}_abs_diff"] = (merged[kline] - merged[rcol]).abs()
        merged[f"{kline}_bps_diff"] = merged[f"{kline}_abs_diff"] / merged[kline].abs().replace(0, np.nan) * 10_000
    merged["max_ohlc_bps_diff"] = merged[[f"{x}_bps_diff" for x in ["open", "high", "low", "close"]]].max(axis=1)
    merged["date"] = merged["open_time"].dt.strftime("%Y-%m-%d")

    daily_rows: list[dict[str, Any]] = []
    for date, g in merged.groupby("date", sort=True):
        daily_rows.append({
            "date": date,
            "matched_minutes": int(len(g)),
            "median_max_ohlc_bps_diff": float(g["max_ohlc_bps_diff"].median()),
            "p95_max_ohlc_bps_diff": float(g["max_ohlc_bps_diff"].quantile(0.95)),
            "max_max_ohlc_bps_diff": float(g["max_ohlc_bps_diff"].max()),
            "exact_ohlc_minutes": int((g["max_ohlc_bps_diff"] <= 1e-9).sum()),
        })

    summary = {
        "status": "PASS" if source["price_semantics"] == "TRADE_DERIVED" else "WARN",
        "source_root": str(source["root"]),
        "timestamp_column": source["time_col"],
        "price_column": price_col,
        "price_semantics": source["price_semantics"],
        "overlap_start": start.isoformat(),
        "overlap_end_exclusive": end.isoformat(),
        "historical_minutes_in_overlap": int(len(hist)),
        "matched_minutes": int(len(merged)),
        "match_rate": float(len(merged) / len(hist)) if len(hist) else None,
        "median_max_ohlc_bps_diff": float(merged["max_ohlc_bps_diff"].median()),
        "p95_max_ohlc_bps_diff": float(merged["max_ohlc_bps_diff"].quantile(0.95)),
        "max_max_ohlc_bps_diff": float(merged["max_ohlc_bps_diff"].max()),
        "note": (
            "REFERENCE_PRICE_FOR_FEATURES is a useful overlap diagnostic but not treated as strict trade-tape reconciliation."
            if source["price_semantics"] != "TRADE_DERIVED"
            else "Trade-derived repository price was used for strict OHLC overlap diagnostics."
        ),
        "search_diagnostics": diagnostics,
    }
    return summary, daily_rows


def build_findings(summary: dict[str, Any], overlap: dict[str, Any]) -> str:
    status = summary["status"]
    lines = [
        "# Binance USD-M BTCUSDT 1m historical audit",
        "",
        f"Audit status: **{status}**",
        "",
        f"Coverage: `{summary['first_open_time']}` through `{summary['last_open_time']}`",
        f"Months audited: **{summary['months_audited']}**",
        f"Rows audited: **{summary['rows_audited']:,}**",
        f"Expected calendar minutes: **{summary['expected_rows']:,}**",
        f"Missing minutes: **{summary['missing_minutes']}**",
        f"Duplicate minutes: **{summary['duplicate_minutes']}**",
        f"Historical invariant failures: **{summary['historical_invariant_failures']}**",
        "",
        "## Source reconciliation",
        "",
        f"Source-to-Parquet mismatch rows: **{summary['source_field_mismatch_rows']}**",
        f"Months with checksum failures: **{summary['checksum_failure_months']}**",
        f"Zero quote-volume rows: **{summary['zero_quote_volume_rows']}**",
        "",
        "## Repository overlap",
        "",
        f"Overlap status: **{overlap.get('status', 'NOT_RUN')}**",
        f"Overlap source: `{overlap.get('source_root', '')}`",
        f"Matched minutes: **{overlap.get('matched_minutes', 0):,}**" if isinstance(overlap.get('matched_minutes'), int) else f"Reason: {overlap.get('reason', '')}",
    ]
    if overlap.get("median_max_ohlc_bps_diff") is not None:
        lines.extend([
            f"Median max OHLC difference: **{overlap['median_max_ohlc_bps_diff']:.6f} bps**",
            f"P95 max OHLC difference: **{overlap['p95_max_ohlc_bps_diff']:.6f} bps**",
            "",
            overlap.get("note", ""),
        ])
    lines.extend([
        "",
        "## Interpretation",
        "",
        "A PASS means the normalized historical series independently reconciled to the verified Binance archives and the expected calendar-minute structure. Collector-overlap warnings do not invalidate clean Binance history when the repository does not expose a strictly comparable trade-price field.",
        "",
    ])
    return "\n".join(lines)


def run_self_test() -> int:
    require_libraries()
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        symbol = "BTCUSDT"
        month = "2020-02"
        start, end = month_bounds(month)
        idx = pd.date_range(start, end - pd.Timedelta(minutes=1), freq="1min", tz="UTC")
        n = len(idx)
        price = 9000.0 + np.arange(n) * 0.01
        base = np.where(np.arange(n) == 10, 0.0, 1.0)
        quote = base * price
        buy_base = base * 0.6
        buy_quote = quote * 0.6
        share = np.full(n, np.nan)
        np.divide(buy_quote, quote, out=share, where=quote > 0)
        # Force millisecond-resolution timestamp dtypes to mirror the actual
        # Pass-2 Parquet files and guard against unit-assumption regressions.
        idx_ms = pd.Series(idx).dt.as_unit("ms")
        close_ms_series = pd.Series(idx + pd.Timedelta(milliseconds=59_999)).dt.as_unit("ms")
        assert str(idx_ms.dtype) == "datetime64[ms, UTC]"
        assert datetime_series_to_epoch_ms(idx_ms)[0] == int(idx[0].timestamp() * 1000)

        frame = pd.DataFrame({
            "symbol": symbol,
            "open_time": idx_ms,
            "close_time": close_ms_series,
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price + 0.1,
            "base_volume": base,
            "quote_volume": quote,
            "trade_count": np.where(base > 0, 10, 0).astype("int64"),
            "taker_buy_base_volume": buy_base,
            "taker_buy_quote_volume": buy_quote,
            "taker_sell_base_volume": base - buy_base,
            "taker_sell_quote_volume": quote - buy_quote,
            "taker_buy_quote_share": share,
        })[OUTPUT_COLUMNS]
        pq_path = root / "parquet" / f"symbol={symbol}" / "year=2020" / "month=02" / "part-00000.parquet"
        pq_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), pq_path)

        zip_path = root / "archives" / "year=2020" / "BTCUSDT-1m-2020-02.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        raw_rows = []
        for i, ts in enumerate(idx):
            raw_rows.append([
                int(ts.timestamp() * 1000), frame.iloc[i]["open"], frame.iloc[i]["high"], frame.iloc[i]["low"], frame.iloc[i]["close"],
                frame.iloc[i]["base_volume"], int(ts.timestamp() * 1000) + 59_999, frame.iloc[i]["quote_volume"],
                frame.iloc[i]["trade_count"], frame.iloc[i]["taker_buy_base_volume"], frame.iloc[i]["taker_buy_quote_volume"], 0,
            ])
        payload = "\n".join(",".join(map(str, r)) for r in raw_rows) + "\n"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("BTCUSDT-1m-2020-02.csv", payload)
        digest = sha256_file(zip_path)

        meta = root / "metadata"
        meta.mkdir(parents=True, exist_ok=True)
        with (meta / "download_manifest.csv").open("w", encoding="utf-8", newline="") as h:
            w = csv.DictWriter(h, fieldnames=["symbol", "interval", "month", "status", "expected_sha256", "archive_path", "source_bytes"])
            w.writeheader(); w.writerow({"symbol": symbol, "interval": "1m", "month": month, "status": "VERIFIED_REUSED", "expected_sha256": digest, "archive_path": str(zip_path), "source_bytes": zip_path.stat().st_size})
        with (meta / "build_manifest.csv").open("w", encoding="utf-8", newline="") as h:
            w = csv.DictWriter(h, fieldnames=["month", "status", "rows", "parquet_path", "parquet_bytes"])
            w.writeheader(); w.writerow({"month": month, "status": "COMPLETED", "rows": n, "parquet_path": str(pq_path), "parquet_bytes": pq_path.stat().st_size})

        item = load_month_inputs(root, symbol, "1m", [month])[0]
        result, gaps, zeros, inv = audit_month(item, symbol)
        assert result.status == "PASS", result.error
        assert result.rows == 41760
        assert result.missing_minutes == 0
        assert result.zero_quote_volume_rows == 1
        assert len(gaps) == 0
        assert zeros and zeros[0]["taker_buy_quote_share"] is None
        assert not [x for x in inv if x.status == "FAIL"]

        deep = root / ("x" * 100) / ("y" * 100) / "summary.json"
        atomic_write_json(deep, {"ok": True})
        assert json.loads(deep.read_text(encoding="utf-8"))["ok"] is True

    print("SELF-TEST PASS")
    print("Validated millisecond-resolution Parquet timestamp conversion, exact calendar-minute reconstruction, source ZIP checksum/source-to-Parquet reconciliation, OHLC/volume rules, zero-volume share handling, deterministic sell fields, and Windows-safe atomic writers.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Historical root created by Pass 1 and Pass 2")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--months", help="Optional comma-separated YYYY-MM subset for smoke testing")
    parser.add_argument("--report-root", type=Path, help="Default: <repo>/reports/binance_um_klines_1m_audit")
    parser.add_argument("--overlap-root", type=Path, help="Optional explicit repository 1s dataset root for overlap diagnostics")
    parser.add_argument("--skip-overlap", action="store_true", help="Skip repository collector-overlap diagnostics")
    parser.add_argument("--force", action="store_true", help="Re-audit selected months even when signatures match")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_test()
    require_libraries()
    if args.interval != "1m":
        raise RuntimeError("Pass 3 is intentionally frozen to the 1m interval")

    repo_root = find_repo_root(Path(__file__))
    root = args.root.resolve()
    report_root = (args.report_root or (repo_root / "reports" / "binance_um_klines_1m_audit")).resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    selected = parse_months(args.months)
    inputs = load_month_inputs(root, args.symbol, args.interval, selected)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]

    print("Auditing normalized Binance USD-M 1m kline history...")
    print(f"Auditor version   : {AUDITOR_VERSION}")
    print(f"Run ID            : {run_id}")
    print(f"Symbol            : {args.symbol}")
    print(f"Interval          : {args.interval}")
    print(f"Months            : {len(inputs)} ({inputs[0].month} to {inputs[-1].month})")
    print(f"Historical root   : {root}")
    print(f"Report root       : {report_root}")
    print("Every source ZIP will be re-hashed and source rows will be reconciled to Parquet.")
    print("Missing minutes will be reported, never filled.")

    state_path = root / "metadata" / "audit_state.sqlite"
    conn = init_state(state_path)
    month_results: list[MonthAudit] = []
    all_gaps: list[dict[str, Any]] = []
    all_zero: list[dict[str, Any]] = []
    all_inv: list[Invariant] = []
    reused_count = 0
    try:
        for idx, item in enumerate(inputs, start=1):
            reusable = load_reusable_bundle(conn, root, item, args.force)
            if reusable is not None:
                result, gaps, zeros, inv = reusable
                reused_count += 1
                display = "PASS_REUSED"
            else:
                result, gaps, zeros, inv = audit_month(item, args.symbol)
                save_state(conn, root, result, gaps, zeros, inv)
                display = result.status
            month_results.append(result)
            all_gaps.extend(gaps)
            all_zero.extend(zeros)
            all_inv.extend(inv)
            if result.status == "PASS":
                print(f"[{idx:02d}/{len(inputs):02d}] {item.month}: {display} ({result.rows:,} rows)")
            else:
                print(f"[{idx:02d}/{len(inputs):02d}] {item.month}: FAIL - {result.error}")
    finally:
        conn.close()

    # Cross-month continuity is evaluated from audited month endpoints, independently of total row count.
    import pandas as pd
    passed = sorted([r for r in month_results if r.status == "PASS"], key=lambda x: x.month)
    cross_month_failures = 0
    for left, right in zip(passed, passed[1:]):
        left_last = pd.Timestamp(left.last_open_time)
        right_first = pd.Timestamp(right.first_open_time)
        expected = left_last + pd.Timedelta(minutes=1)
        ok = right_first == expected
        all_inv.append(Invariant("GLOBAL", f"cross_month_continuity_{left.month}_to_{right.month}", "PASS" if ok else "FAIL", right_first.isoformat(), expected.isoformat()))
        if not ok:
            cross_month_failures += 1

    historical_failures = sum(1 for x in all_inv if x.status == "FAIL")

    if args.skip_overlap:
        overlap_summary = {"status": "SKIPPED", "reason": "--skip-overlap was requested"}
        overlap_daily: list[dict[str, Any]] = []
    else:
        try:
            overlap_summary, overlap_daily = audit_overlap(root, repo_root, args.symbol, args.overlap_root)
        except Exception as exc:
            overlap_summary = {"status": "WARN", "reason": f"Overlap audit could not complete: {type(exc).__name__}: {exc}"}
            overlap_daily = []

    rows_audited = sum(r.rows for r in month_results if r.status == "PASS")
    expected_rows = sum(r.expected_rows for r in month_results)
    status = "FAIL" if historical_failures else ("WARN" if overlap_summary.get("status") == "WARN" else "PASS")
    first_open = passed[0].first_open_time if passed else ""
    last_open = passed[-1].last_open_time if passed else ""
    summary = {
        "auditor_version": AUDITOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": utc_now_iso(),
        "status": status,
        "symbol": args.symbol,
        "interval": args.interval,
        "months_requested": len(inputs),
        "months_audited": len(passed),
        "months_failed": len(month_results) - len(passed),
        "months_reaudited_from_matching_state": reused_count,
        "rows_audited": rows_audited,
        "expected_rows": expected_rows,
        "first_open_time": first_open,
        "last_open_time": last_open,
        "missing_minutes": sum(r.missing_minutes for r in month_results),
        "duplicate_minutes": sum(r.duplicate_minutes for r in month_results),
        "unexpected_minutes": sum(r.unexpected_minutes for r in month_results),
        "cross_month_failures": cross_month_failures,
        "source_field_mismatch_rows": sum(r.source_field_mismatch_rows for r in month_results),
        "checksum_failure_months": sum(1 for r in month_results if not r.source_sha256_match),
        "zero_base_volume_rows": sum(r.zero_base_volume_rows for r in month_results),
        "zero_quote_volume_rows": sum(r.zero_quote_volume_rows for r in month_results),
        "null_share_rows": sum(r.null_share_rows for r in month_results),
        "historical_invariant_failures": historical_failures,
        "overlap_status": overlap_summary.get("status"),
        "historical_root": str(root),
        "report_root": str(report_root),
        "sqlite_state": str(state_path),
    }

    monthly_fields = list(MonthAudit.__dataclass_fields__.keys())
    atomic_write_csv(report_root / "binance_um_klines_1m_monthly_audit.csv", monthly_fields, [asdict(r) for r in month_results])
    atomic_write_csv(report_root / "binance_um_klines_1m_gaps.csv", ["month", "kind", "open_time"], all_gaps)
    atomic_write_csv(report_root / "binance_um_klines_1m_zero_volume.csv", ["month", "open_time", "base_volume", "quote_volume", "trade_count", "taker_buy_quote_share"], all_zero)
    atomic_write_csv(
        report_root / "binance_um_klines_1m_source_reconciliation.csv",
        ["month", "source_rows", "parquet_rows", "row_count_match", "source_field_mismatch_rows", "source_sha256_match", "source_bytes", "parquet_bytes"],
        [{
            "month": r.month,
            "source_rows": r.source_rows,
            "parquet_rows": r.rows,
            "row_count_match": r.source_parquet_row_match,
            "source_field_mismatch_rows": r.source_field_mismatch_rows,
            "source_sha256_match": r.source_sha256_match,
            "source_bytes": r.source_bytes,
            "parquet_bytes": r.parquet_bytes,
        } for r in month_results],
    )
    overlap_fields = ["date", "matched_minutes", "median_max_ohlc_bps_diff", "p95_max_ohlc_bps_diff", "max_max_ohlc_bps_diff", "exact_ohlc_minutes"]
    atomic_write_csv(report_root / "binance_um_klines_1m_overlap_daily.csv", overlap_fields, overlap_daily)
    atomic_write_json(report_root / "binance_um_klines_1m_overlap_summary.json", overlap_summary)
    inv_fields = list(Invariant.__dataclass_fields__.keys())
    atomic_write_csv(report_root / "binance_um_klines_1m_invariants.csv", inv_fields, [asdict(x) for x in all_inv])
    config = {
        "auditor_version": AUDITOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "symbol": args.symbol,
        "interval": args.interval,
        "months": [x.month for x in inputs],
        "continuity_contract": "exact calendar minute reconstruction; no filling or repair",
        "source_reconciliation": "re-hash ZIP; compare source timestamp/trade_count/OHLC/volume/taker-buy fields to Parquet",
        "zero_volume_contract": "taker_buy_quote_share must be null iff quote_volume is zero",
        "overlap_contract": "strict only when repository exposes a trade-derived price; price_for_features is diagnostic WARN",
    }
    atomic_write_json(report_root / "binance_um_klines_1m_audit_config.json", config)
    atomic_write_text(report_root / "binance_um_klines_1m_findings.md", build_findings(summary, overlap_summary))
    atomic_write_json(root / "metadata" / "latest_audit_summary.json", summary)

    print("\nHistorical audit summary")
    print("------------------------")
    print(f"Status                 : {status}")
    print(f"Months audited         : {summary['months_audited']}")
    print(f"Months failed          : {summary['months_failed']}")
    print(f"Rows audited           : {rows_audited:,}")
    print(f"Expected calendar rows : {expected_rows:,}")
    print(f"Missing minutes        : {summary['missing_minutes']}")
    print(f"Duplicate minutes      : {summary['duplicate_minutes']}")
    print(f"Source mismatch rows   : {summary['source_field_mismatch_rows']}")
    print(f"Invariant failures     : {summary['historical_invariant_failures']}")
    print(f"Zero quote-volume rows : {summary['zero_quote_volume_rows']}")
    print(f"Collector overlap      : {overlap_summary.get('status')}")
    if overlap_summary.get("reason"):
        print(f"Overlap note           : {overlap_summary['reason']}")
    elif overlap_summary.get("matched_minutes") is not None:
        print(f"Overlap matched minutes: {overlap_summary['matched_minutes']:,}")
        print(f"Overlap price semantics: {overlap_summary.get('price_semantics')}")
        print(f"P95 OHLC diff (bps)    : {overlap_summary.get('p95_max_ohlc_bps_diff'):.6f}")
    print(f"Summary                : {root / 'metadata' / 'latest_audit_summary.json'}")
    print(f"Findings               : {report_root / 'binance_um_klines_1m_findings.md'}")

    return 1 if historical_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
