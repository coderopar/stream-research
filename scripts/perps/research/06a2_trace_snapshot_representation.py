#!/usr/bin/env python3
"""Trace exact BTCUSDT depth snapshot representation from bronze WAL to canonical Parquet.

Phase 6A2 is a read-only provenance diagnostic.  It selects one canonical
book-snapshot row from each requested 06A1 top-status class (by default:
VALID, EMPTY_BID, EMPTY_ASK, EMPTY_BOTH), then follows the canonical
``source_relative_path`` + ``source_record_number`` back to the exact bronze
WAL record.

For each case it compares three representations:

  1. exact raw bronze ``payload.snapshot`` levels,
  2. canonical ``book_snapshots`` nested ``{price, quantity, extra_json}``
     structs as read with PyArrow,
  3. the same level-pair semantics used by 06A1.

The script's purpose is localization, not repair.  It never changes bronze,
canonical data, or the Phase-6A reconstruction contract.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

TRACE_VERSION = "1.0.0"
SYMBOL_DEFAULT = "BTCUSDT"
DATE_DEFAULT = "2026-06-12"
CASES_DEFAULT = ("VALID", "EMPTY_BID", "EMPTY_ASK", "EMPTY_BOTH")

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_CANONICAL_ROOT = REPO_ROOT / "data" / "canonical" / "market_events_v1"
DEFAULT_BRONZE_ROOT = Path(r"C:\Users\USER\stream-bronze-download\data\bronze")
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_snapshot_representation_trace"

def require_parquet():
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'pyarrow'. Activate the project virtualenv and install pyarrow."
        ) from exc
    return pq


def require_zstandard():
    try:
        import zstandard as zstd
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'zstandard'. Activate the project virtualenv and install zstandard."
        ) from exc
    return zstd


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def safe_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def compact_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".t", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_write_text(path, "")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    fd, tmp = tempfile.mkstemp(prefix=".c", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in fields})
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def iso_from_ns(ts_ns: int | None) -> str | None:
    if ts_ns is None:
        return None
    return datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Canonical-extractor level semantics copied independently from
# 01_extract_all_market_events.py for an exact expected-value comparison.
# ---------------------------------------------------------------------------


def canonical_expected_levels(value: Any) -> list[dict[str, str | None]]:
    if not isinstance(value, list):
        return []
    levels: list[dict[str, str | None]] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            price = safe_str(item[0]) if len(item) >= 1 else None
            quantity = safe_str(item[1]) if len(item) >= 2 else None
            extra = list(item[2:]) if len(item) > 2 else None
        elif isinstance(item, Mapping):
            price = safe_str(item.get("price", item.get("px")))
            quantity = safe_str(item.get("quantity", item.get("qty", item.get("size"))))
            extra_keys = {
                key: val
                for key, val in item.items()
                if key not in {"price", "px", "quantity", "qty", "size"}
            }
            extra = extra_keys or None
        else:
            price = safe_str(item)
            quantity = None
            extra = None
        levels.append(
            {
                "price": price,
                "quantity": quantity,
                "extra_json": compact_json(extra),
            }
        )
    return levels


# ---------------------------------------------------------------------------
# 06A1 level semantics, intentionally matching the prior diagnostic.
# ---------------------------------------------------------------------------


def level_pair_06a1(level: Any) -> tuple[float | None, float | None]:
    if isinstance(level, Mapping):
        p = safe_float(level.get("price"))
        q = safe_float(level.get("quantity"))
        if q is None:
            q = safe_float(level.get("qty"))
        return p, q
    try:
        return safe_float(level[0]), safe_float(level[1])
    except Exception:
        return None, None


def parse_levels_06a1(levels: Any) -> dict[str, Any]:
    raw = list(levels or [])
    usable: list[tuple[float, float]] = []
    malformed = 0
    nonpositive = 0
    for level in raw:
        p, q = level_pair_06a1(level)
        if p is None or q is None:
            malformed += 1
            continue
        if p <= 0 or q <= 0:
            nonpositive += 1
            continue
        usable.append((p, q))
    prices = [p for p, _ in usable]
    return {
        "raw_count": len(raw),
        "usable_count": len(usable),
        "malformed_count": malformed,
        "nonpositive_count": nonpositive,
        "pairs": usable,
        "prices": prices,
        "first_price": usable[0][0] if usable else None,
    }


def classify_top(bid: float | None, ask: float | None) -> str:
    if bid is None and ask is None:
        return "EMPTY_BOTH"
    if bid is None:
        return "EMPTY_BID"
    if ask is None:
        return "EMPTY_ASK"
    if bid <= 0 or ask <= 0:
        return "NON_POSITIVE_TOP"
    if bid < ask:
        return "VALID"
    if bid == ask:
        return "LOCKED"
    return "CROSSED"


def semantics_06a1(bids: Any, asks: Any) -> dict[str, Any]:
    bp = parse_levels_06a1(bids)
    ap = parse_levels_06a1(asks)
    best_bid = max(bp["prices"]) if bp["prices"] else None
    best_ask = min(ap["prices"]) if ap["prices"] else None
    return {
        "status": classify_top(best_bid, best_ask),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_raw_count": bp["raw_count"],
        "ask_raw_count": ap["raw_count"],
        "bid_usable_count": bp["usable_count"],
        "ask_usable_count": ap["usable_count"],
        "bid_malformed_count": bp["malformed_count"],
        "ask_malformed_count": ap["malformed_count"],
        "bid_nonpositive_count": bp["nonpositive_count"],
        "ask_nonpositive_count": ap["nonpositive_count"],
    }


# ---------------------------------------------------------------------------
# Canonical discovery and case selection
# ---------------------------------------------------------------------------

CANONICAL_COLUMNS = [
    "symbol",
    "ts_recv_ns",
    "event_id",
    "sequence",
    "source_file_id",
    "source_relative_path",
    "source_record_number",
    "snapshot_symbol",
    "last_update_id",
    "bids",
    "asks",
    "bid_level_count",
    "ask_level_count",
]


@dataclass
class SelectedCase:
    requested_status: str
    canonical_row: dict[str, Any]
    canonical_semantics: dict[str, Any]


def discover_snapshot_files(root: Path, d: str) -> tuple[Path, ...]:
    base = root / "perps" / "book_snapshots" / f"date={d}"
    if not base.exists():
        return ()
    return tuple(sorted(p for p in base.rglob("*.parquet") if p.is_file()))


def iter_canonical_snapshots(files: Sequence[Path], symbol: str, batch_size: int) -> Iterator[dict[str, Any]]:
    pq = require_parquet()
    for path in files:
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        missing = [c for c in CANONICAL_COLUMNS if c not in available]
        if missing:
            raise RuntimeError(f"{path}: missing required canonical columns: {missing}")
        for batch in pf.iter_batches(batch_size=max(1, batch_size), columns=CANONICAL_COLUMNS):
            for row in batch.to_pylist():
                if safe_str(row.get("symbol")) != symbol:
                    continue
                yield row


def select_cases(
    files: Sequence[Path], symbol: str, requested: Sequence[str], batch_size: int
) -> list[SelectedCase]:
    remaining = {x.upper() for x in requested}
    chosen: dict[str, SelectedCase] = {}
    for row in iter_canonical_snapshots(files, symbol, batch_size):
        sem = semantics_06a1(row.get("bids"), row.get("asks"))
        status = sem["status"]
        if status in remaining and status not in chosen:
            chosen[status] = SelectedCase(status, row, sem)
            if len(chosen) == len(remaining):
                break
    return [chosen[x] for x in requested if x in chosen]


# ---------------------------------------------------------------------------
# Exact bronze provenance resolution and line extraction
# ---------------------------------------------------------------------------


def normalize_rel_path(value: str) -> Path:
    # Canonical paths are stored as source-relative strings; normalize both
    # slash conventions without interpreting a drive letter as absolute here.
    value = value.replace("\\", "/").lstrip("/")
    return Path(*[part for part in value.split("/") if part not in ("", ".")])


def resolve_bronze_path(bronze_root: Path, source_relative_path: str) -> Path:
    rel = normalize_rel_path(source_relative_path)
    candidates: list[Path] = []

    exact = bronze_root / rel
    if exact.is_file():
        return exact

    # Some manifests include an outer bronze/data/bronze prefix.  Strip only
    # known structural prefixes; never silently choose among ambiguous files.
    parts = list(rel.parts)
    prefixes = [
        ("data", "bronze"),
        ("bronze",),
    ]
    for prefix in prefixes:
        if tuple(parts[: len(prefix)]) == prefix:
            p = bronze_root / Path(*parts[len(prefix) :])
            if p.is_file():
                candidates.append(p)

    # Final fallback: same basename, then require the suffix path to agree as
    # far as possible.  Ambiguity is an error rather than a guessed provenance.
    if not candidates and rel.name:
        for p in bronze_root.rglob(rel.name):
            if not p.is_file():
                continue
            pp = p.as_posix().replace("\\", "/")
            rr = rel.as_posix()
            if pp.endswith(rr) or len(rel.parts) == 1:
                candidates.append(p)

    unique = sorted({p.resolve() for p in candidates})
    if len(unique) == 1:
        return unique[0]
    if not unique:
        raise FileNotFoundError(
            f"Could not resolve bronze source_relative_path={source_relative_path!r} under {bronze_root}"
        )
    raise RuntimeError(
        f"Ambiguous bronze provenance for {source_relative_path!r}: "
        + ", ".join(str(p) for p in unique[:10])
    )


def open_text_lines(path: Path) -> Iterator[str]:
    name = path.name.lower()
    if name.endswith(".zst"):
        zstd = require_zstandard()
        dctx = zstd.ZstdDecompressor()
        with path.open("rb") as compressed:
            with dctx.stream_reader(compressed, read_across_frames=True) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8", errors="strict") as text:
                    for line in text:
                        yield line
        return
    # Useful for synthetic self-test and any uncompressed archive copies.
    with path.open("r", encoding="utf-8") as handle:
        yield from handle


def fetch_exact_bronze_records(
    requests: Sequence[tuple[Path, int]]
) -> dict[tuple[Path, int], dict[str, Any]]:
    grouped: dict[Path, set[int]] = defaultdict(set)
    for path, n in requests:
        if n <= 0:
            raise ValueError(f"source_record_number must be positive, got {n}")
        grouped[path].add(n)

    found: dict[tuple[Path, int], dict[str, Any]] = {}
    for path, targets in grouped.items():
        max_target = max(targets)
        outstanding = set(targets)
        for n, line in enumerate(open_text_lines(path), start=1):
            if n in outstanding:
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise RuntimeError(f"{path}:{n}: bronze record is not a JSON object")
                found[(path, n)] = dict(value)
                outstanding.remove(n)
                if not outstanding:
                    break
            if n >= max_target and outstanding:
                break
        if outstanding:
            raise RuntimeError(
                f"{path}: could not retrieve source record(s) {sorted(outstanding)}; "
                "canonical provenance no longer aligns with this bronze file"
            )
    return found


# ---------------------------------------------------------------------------
# Comparison and localization
# ---------------------------------------------------------------------------


def bronze_snapshot(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = record.get("payload")
    payload = dict(payload) if isinstance(payload, Mapping) else {}
    snap = payload.get("snapshot")
    return dict(snap) if isinstance(snap, Mapping) else {}


def normalize_canonical_level(level: Any) -> dict[str, str | None]:
    if isinstance(level, Mapping):
        return {
            "price": safe_str(level.get("price")),
            "quantity": safe_str(level.get("quantity")),
            "extra_json": safe_str(level.get("extra_json")),
        }
    return {"price": None, "quantity": None, "extra_json": None}


def norm_numeric_equal(a: Any, b: Any) -> bool:
    fa = safe_float(a)
    fb = safe_float(b)
    if fa is None or fb is None:
        return safe_str(a) == safe_str(b)
    return math.isclose(fa, fb, rel_tol=0.0, abs_tol=1e-12)


def compare_level_lists(expected: Sequence[Mapping[str, Any]], actual: Sequence[Any]) -> dict[str, Any]:
    actual_norm = [normalize_canonical_level(x) for x in list(actual or [])]
    n = max(len(expected), len(actual_norm))
    price_mismatch = 0
    qty_mismatch = 0
    extra_mismatch = 0
    for i in range(n):
        e = expected[i] if i < len(expected) else {"price": None, "quantity": None, "extra_json": None}
        a = actual_norm[i] if i < len(actual_norm) else {"price": None, "quantity": None, "extra_json": None}
        if not norm_numeric_equal(e.get("price"), a.get("price")):
            price_mismatch += 1
        if not norm_numeric_equal(e.get("quantity"), a.get("quantity")):
            qty_mismatch += 1
        if safe_str(e.get("extra_json")) != safe_str(a.get("extra_json")):
            extra_mismatch += 1
    return {
        "expected_count": len(expected),
        "actual_count": len(actual_norm),
        "price_mismatch_count": price_mismatch,
        "quantity_mismatch_count": qty_mismatch,
        "extra_mismatch_count": extra_mismatch,
        "exact_price_quantity_match": price_mismatch == 0 and qty_mismatch == 0 and len(expected) == len(actual_norm),
    }


def usable_count_from_raw(levels: Any) -> int:
    return parse_levels_06a1(canonical_expected_levels(levels))["usable_count"]


def localize_stage(
    bronze_bids: Any,
    bronze_asks: Any,
    canonical_bids: Any,
    canonical_asks: Any,
    bid_cmp: Mapping[str, Any],
    ask_cmp: Mapping[str, Any],
) -> tuple[str, str]:
    bronze_expected_bids = canonical_expected_levels(bronze_bids)
    bronze_expected_asks = canonical_expected_levels(bronze_asks)
    bronze_sem = semantics_06a1(bronze_expected_bids, bronze_expected_asks)
    canonical_sem = semantics_06a1(canonical_bids, canonical_asks)

    if not bool(bid_cmp["exact_price_quantity_match"]) or not bool(ask_cmp["exact_price_quantity_match"]):
        return (
            "BRONZE_TO_CANONICAL_MISMATCH",
            "The canonical nested price/quantity representation differs from what the extractor should produce from the exact bronze record.",
        )

    # At this point canonical equals extractor-expected bronze.  If the status is
    # already unusable, the anomaly exists upstream of canonical.
    if bronze_sem["status"] != "VALID":
        return (
            "BRONZE_ALREADY_NONUSABLE",
            "Bronze itself produces the same unusable top status under the canonical/06A1 parsing contract; investigate collector/runtime snapshot production next.",
        )

    if canonical_sem["status"] != bronze_sem["status"]:
        # Defensive branch; with exact level equality this should not happen.
        return (
            "CANONICAL_TO_06A1_INTERPRETATION_MISMATCH",
            "Bronze and canonical levels match, but the diagnostic status differs; inspect 06A1 parsing/classification semantics.",
        )

    return (
        "CONSISTENT_END_TO_END",
        "Bronze, canonical representation, and 06A1 parsing agree for this case.",
    )


def build_case_payload(case: SelectedCase, bronze_path: Path, bronze_record: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    crow = case.canonical_row
    snap = bronze_snapshot(bronze_record)
    bronze_bids = snap.get("bids")
    bronze_asks = snap.get("asks")
    expected_bids = canonical_expected_levels(bronze_bids)
    expected_asks = canonical_expected_levels(bronze_asks)
    canonical_bids = list(crow.get("bids") or [])
    canonical_asks = list(crow.get("asks") or [])

    bronze_sem = semantics_06a1(expected_bids, expected_asks)
    canonical_sem = semantics_06a1(canonical_bids, canonical_asks)
    bid_cmp = compare_level_lists(expected_bids, canonical_bids)
    ask_cmp = compare_level_lists(expected_asks, canonical_asks)
    stage, stage_note = localize_stage(
        bronze_bids, bronze_asks, canonical_bids, canonical_asks, bid_cmp, ask_cmp
    )

    raw_ts = safe_int(bronze_record.get("ts_recv_ns"))
    summary = {
        "requested_case": case.requested_status,
        "canonical_status": canonical_sem["status"],
        "bronze_expected_status": bronze_sem["status"],
        "localized_stage": stage,
        "localization_note": stage_note,
        "canonical_ts_recv_ns": safe_int(crow.get("ts_recv_ns")),
        "canonical_time": iso_from_ns(safe_int(crow.get("ts_recv_ns"))),
        "bronze_ts_recv_ns": raw_ts,
        "bronze_time": iso_from_ns(raw_ts),
        "timestamp_match": safe_int(crow.get("ts_recv_ns")) == raw_ts,
        "canonical_event_id": safe_str(crow.get("event_id")),
        "bronze_event_id": safe_str(bronze_record.get("event_id")),
        "event_id_match": safe_str(crow.get("event_id")) == safe_str(bronze_record.get("event_id")),
        "canonical_source_relative_path": safe_str(crow.get("source_relative_path")),
        "canonical_source_record_number": safe_int(crow.get("source_record_number")),
        "resolved_bronze_path": str(bronze_path),
        "canonical_last_update_id": safe_int(crow.get("last_update_id")),
        "bronze_last_update_id": safe_int(snap.get("last_u", snap.get("lastUpdateId"))),
        "last_update_id_match": safe_int(crow.get("last_update_id")) == safe_int(snap.get("last_u", snap.get("lastUpdateId"))),
        "bronze_bid_raw_count": len(bronze_bids) if isinstance(bronze_bids, list) else 0,
        "bronze_ask_raw_count": len(bronze_asks) if isinstance(bronze_asks, list) else 0,
        "bronze_bid_usable_count": bronze_sem["bid_usable_count"],
        "bronze_ask_usable_count": bronze_sem["ask_usable_count"],
        "canonical_bid_raw_count": canonical_sem["bid_raw_count"],
        "canonical_ask_raw_count": canonical_sem["ask_raw_count"],
        "canonical_bid_usable_count": canonical_sem["bid_usable_count"],
        "canonical_ask_usable_count": canonical_sem["ask_usable_count"],
        "canonical_bid_nonpositive_count": canonical_sem["bid_nonpositive_count"],
        "canonical_ask_nonpositive_count": canonical_sem["ask_nonpositive_count"],
        "bid_price_mismatch_count": bid_cmp["price_mismatch_count"],
        "bid_quantity_mismatch_count": bid_cmp["quantity_mismatch_count"],
        "ask_price_mismatch_count": ask_cmp["price_mismatch_count"],
        "ask_quantity_mismatch_count": ask_cmp["quantity_mismatch_count"],
        "bid_exact_price_quantity_match": bid_cmp["exact_price_quantity_match"],
        "ask_exact_price_quantity_match": ask_cmp["exact_price_quantity_match"],
        "bronze_best_bid": bronze_sem["best_bid"],
        "bronze_best_ask": bronze_sem["best_ask"],
        "canonical_best_bid": canonical_sem["best_bid"],
        "canonical_best_ask": canonical_sem["best_ask"],
    }

    level_rows: list[dict[str, Any]] = []
    for side, raw_levels, expected, actual in (
        ("BID", bronze_bids if isinstance(bronze_bids, list) else [], expected_bids, canonical_bids),
        ("ASK", bronze_asks if isinstance(bronze_asks, list) else [], expected_asks, canonical_asks),
    ):
        n = max(len(raw_levels), len(expected), len(actual))
        for i in range(n):
            raw = raw_levels[i] if i < len(raw_levels) else None
            exp = expected[i] if i < len(expected) else {"price": None, "quantity": None, "extra_json": None}
            act = normalize_canonical_level(actual[i]) if i < len(actual) else {"price": None, "quantity": None, "extra_json": None}
            parsed_p, parsed_q = level_pair_06a1(actual[i]) if i < len(actual) else (None, None)
            level_rows.append(
                {
                    "requested_case": case.requested_status,
                    "side": side,
                    "level_index": i,
                    "bronze_raw_json": compact_json(raw),
                    "bronze_expected_price": exp.get("price"),
                    "bronze_expected_quantity": exp.get("quantity"),
                    "bronze_expected_extra_json": exp.get("extra_json"),
                    "canonical_price": act.get("price"),
                    "canonical_quantity": act.get("quantity"),
                    "canonical_extra_json": act.get("extra_json"),
                    "parser_price": parsed_p,
                    "parser_quantity": parsed_q,
                    "price_matches": norm_numeric_equal(exp.get("price"), act.get("price")),
                    "quantity_matches": norm_numeric_equal(exp.get("quantity"), act.get("quantity")),
                    "canonical_parser_price_matches": norm_numeric_equal(act.get("price"), parsed_p),
                    "canonical_parser_quantity_matches": norm_numeric_equal(act.get("quantity"), parsed_q),
                    "parser_usable": bool(parsed_p is not None and parsed_q is not None and parsed_p > 0 and parsed_q > 0),
                }
            )

    full_payload = {
        "trace_version": TRACE_VERSION,
        "summary": summary,
        "bronze_record": bronze_record,
        "bronze_snapshot": snap,
        "canonical_row": crow,
        "canonical_semantics_06a1": canonical_sem,
        "bronze_expected_semantics_06a1": bronze_sem,
        "bronze_to_canonical_bid_comparison": bid_cmp,
        "bronze_to_canonical_ask_comparison": ask_cmp,
        "levels": level_rows,
    }
    return full_payload, level_rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def findings_markdown(
    d: str,
    symbol: str,
    cases: Sequence[Mapping[str, Any]],
    missing_cases: Sequence[str],
) -> str:
    lines = [
        "# Phase 6A2 Snapshot Representation Trace",
        "",
        f"- Trace version: `{TRACE_VERSION}`",
        f"- Date: `{d}`",
        f"- Symbol: `{symbol}`",
        "- Mode: read-only provenance trace; no data or reconstruction rules modified.",
        "",
        "## Case localization",
        "",
    ]
    if cases:
        lines.append("| Case | Canonical status | Bronze status | Localized stage | Bid qty mismatches | Ask qty mismatches |")
        lines.append("|---|---|---|---|---:|---:|")
        for row in cases:
            lines.append(
                f"| {row['requested_case']} | {row['canonical_status']} | {row['bronze_expected_status']} | "
                f"{row['localized_stage']} | {row['bid_quantity_mismatch_count']} | {row['ask_quantity_mismatch_count']} |"
            )
    else:
        lines.append("No cases were traced.")
    if missing_cases:
        lines += ["", "## Missing requested statuses", "", ", ".join(f"`{x}`" for x in missing_cases)]

    stage_counts: dict[str, int] = defaultdict(int)
    for row in cases:
        stage_counts[str(row["localized_stage"])] += 1
    lines += ["", "## Interpretation", ""]
    if stage_counts:
        for stage, n in sorted(stage_counts.items()):
            lines.append(f"- `{stage}`: {n} traced case(s).")
    lines += [
        "",
        "The trace localizes the first observable discrepancy only. If a case is `BRONZE_ALREADY_NONUSABLE`, "
        "the final root cause still lies upstream and requires collector/runtime investigation.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def make_record(ts: int, event_id: str, bids: list[list[str]], asks: list[list[str]], last_u: int) -> dict[str, Any]:
    return {
        "schema": "stream.event.v1",
        "market": "perps",
        "etype": "snapshot",
        "symbol": "BTCUSDT",
        "ts_recv_ns": ts,
        "event_id": event_id,
        "payload": {"snapshot": {"symbol": "BTCUSDT", "last_u": last_u, "bids": bids, "asks": asks}},
    }


def run_self_test() -> None:
    # Validate the three-stage localization logic without relying on local data.
    bronze_good_bids = [["100.00", "2.0"], ["99.99", "1.0"]]
    bronze_good_asks = [["100.01", "3.0"], ["100.02", "1.0"]]
    expected_b = canonical_expected_levels(bronze_good_bids)
    expected_a = canonical_expected_levels(bronze_good_asks)
    assert semantics_06a1(expected_b, expected_a)["status"] == "VALID"

    bronze_zero_bids = [["100.00", "0"], ["99.99", "0"]]
    zero_b = canonical_expected_levels(bronze_zero_bids)
    assert semantics_06a1(zero_b, expected_a)["status"] == "EMPTY_BID"

    cmp_ok = compare_level_lists(expected_b, expected_b)
    assert cmp_ok["exact_price_quantity_match"]
    changed = [dict(x) for x in expected_b]
    changed[0]["quantity"] = "0"
    cmp_bad = compare_level_lists(expected_b, changed)
    assert cmp_bad["quantity_mismatch_count"] == 1

    stage, _ = localize_stage(bronze_good_bids, bronze_good_asks, changed, expected_a, cmp_bad, compare_level_lists(expected_a, expected_a))
    assert stage == "BRONZE_TO_CANONICAL_MISMATCH"

    stage, _ = localize_stage(bronze_zero_bids, bronze_good_asks, zero_b, expected_a, compare_level_lists(zero_b, zero_b), compare_level_lists(expected_a, expected_a))
    assert stage == "BRONZE_ALREADY_NONUSABLE"

    # Verify exact 1-based line retrieval semantics without external dependencies.
    # Real .zst reading uses the project's zstandard dependency during an actual trace.
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.ndjson"
        records = [
            make_record(1_000_000_000 + i, f"e{i}", bronze_good_bids, bronze_good_asks, 100 + i)
            for i in range(3)
        ]
        p.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        got = fetch_exact_bronze_records([(p, 2)])[(p, 2)]
        assert got["event_id"] == "e1"

    print(
        "SELF-TEST PASS\n"
        "Validated extractor-equivalent level normalization, 06A1 parsing, discrepancy localization, and exact 1-based WAL provenance retrieval."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace exact bronze→canonical→06A1 snapshot representation.")
    parser.add_argument("--date", default=DATE_DEFAULT, help="Receive date YYYY-MM-DD (default: 2026-06-12)")
    parser.add_argument("--symbol", default=SYMBOL_DEFAULT)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--bronze-root", type=Path, default=DEFAULT_BRONZE_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument(
        "--cases",
        default=",".join(CASES_DEFAULT),
        help="Comma-separated 06A1 statuses to trace; default VALID,EMPTY_BID,EMPTY_ASK,EMPTY_BOTH",
    )
    parser.add_argument("--canonical-batch-size", type=int, default=256)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0

    try:
        date.fromisoformat(args.date)
    except ValueError as exc:
        raise SystemExit(f"Invalid --date {args.date!r}; expected YYYY-MM-DD") from exc

    requested = [x.strip().upper() for x in str(args.cases).split(",") if x.strip()]
    if not requested:
        raise SystemExit("--cases resolved to an empty list")

    print("Tracing BTCUSDT snapshot representation across bronze → canonical → 06A1...")
    print(f"Trace version     : {TRACE_VERSION}")
    print(f"Date              : {args.date}")
    print(f"Symbol            : {args.symbol}")
    print(f"Canonical root    : {args.canonical_root}")
    print(f"Bronze root       : {args.bronze_root}")
    print(f"Report root       : {args.report_root}")
    print(f"Requested cases   : {', '.join(requested)}")
    print("Mode              : READ ONLY")

    files = discover_snapshot_files(args.canonical_root, args.date)
    if not files:
        raise SystemExit(
            f"No canonical perps/book_snapshots Parquet found for {args.date} under {args.canonical_root}"
        )
    if not args.bronze_root.exists():
        raise SystemExit(
            f"Bronze root does not exist: {args.bronze_root}\n"
            "Override with --bronze-root if your local bronze download moved."
        )

    selected = select_cases(files, args.symbol, requested, args.canonical_batch_size)
    found_statuses = {x.requested_status for x in selected}
    missing = [x for x in requested if x not in found_statuses]
    if not selected:
        raise SystemExit("None of the requested snapshot statuses were found in canonical data")

    provenance_requests: list[tuple[Path, int]] = []
    resolved: dict[str, Path] = {}
    for case in selected:
        rel = safe_str(case.canonical_row.get("source_relative_path"))
        rn = safe_int(case.canonical_row.get("source_record_number"))
        if not rel or rn is None:
            raise RuntimeError(f"{case.requested_status}: missing canonical provenance fields")
        p = resolve_bronze_path(args.bronze_root, rel)
        resolved[case.requested_status] = p
        provenance_requests.append((p, rn))

    bronze_records = fetch_exact_bronze_records(provenance_requests)

    args.report_root.mkdir(parents=True, exist_ok=True)
    case_dir = args.report_root / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    all_level_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []

    for case in selected:
        p = resolved[case.requested_status]
        rn = int(case.canonical_row["source_record_number"])
        record = bronze_records[(p, rn)]
        payload, level_rows = build_case_payload(case, p, record)
        summary_rows.append(payload["summary"])
        all_level_rows.extend(level_rows)
        provenance_rows.append(
            {
                "requested_case": case.requested_status,
                "source_relative_path": case.canonical_row.get("source_relative_path"),
                "source_record_number": rn,
                "resolved_bronze_path": str(p),
                "canonical_ts_recv_ns": case.canonical_row.get("ts_recv_ns"),
                "bronze_ts_recv_ns": record.get("ts_recv_ns"),
                "canonical_event_id": case.canonical_row.get("event_id"),
                "bronze_event_id": record.get("event_id"),
            }
        )
        atomic_write_json(case_dir / f"{case.requested_status.lower()}.json", payload)
        print(
            f"{case.requested_status:10s}: {payload['summary']['localized_stage']} | "
            f"bronze={payload['summary']['bronze_expected_status']} "
            f"canonical={payload['summary']['canonical_status']} | "
            f"source={case.canonical_row.get('source_relative_path')}:{rn}"
        )

    write_csv(args.report_root / "snapshot_representation_trace_summary.csv", summary_rows)
    write_csv(args.report_root / "snapshot_representation_trace_levels.csv", all_level_rows)
    write_csv(args.report_root / "snapshot_representation_trace_provenance.csv", provenance_rows)
    atomic_write_text(
        args.report_root / "snapshot_representation_trace_findings.md",
        findings_markdown(args.date, args.symbol, summary_rows, missing),
    )
    atomic_write_json(
        args.report_root / "snapshot_representation_trace_config.json",
        {
            "trace_version": TRACE_VERSION,
            "date": args.date,
            "symbol": args.symbol,
            "canonical_root": str(args.canonical_root),
            "bronze_root": str(args.bronze_root),
            "requested_cases": requested,
            "found_cases": [x.requested_status for x in selected],
            "missing_cases": missing,
            "canonical_files": [str(p) for p in files],
        },
    )

    print("\nPhase-6A2 snapshot representation trace")
    print("----------------------------------------")
    print(f"Cases traced       : {len(summary_rows)}/{len(requested)}")
    if missing:
        print(f"Cases missing      : {', '.join(missing)}")
    stages: dict[str, int] = defaultdict(int)
    for row in summary_rows:
        stages[str(row["localized_stage"])] += 1
    for stage, n in sorted(stages.items()):
        print(f"{stage:28s}: {n}")
    print(f"Summary            : {args.report_root / 'snapshot_representation_trace_summary.csv'}")
    print(f"Findings           : {args.report_root / 'snapshot_representation_trace_findings.md'}")
    print("No bronze/canonical file was modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
