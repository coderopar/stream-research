#!/usr/bin/env python3
"""R0.3 — build the corrected D0 research substrate.

This is a lineage/infrastructure rebuild only. It preserves the historical
prospective 25-bp candidate population as a named D0 development family,
recovers exact int64 anchors where possible, constructs R0.2-compliant clocks,
and rebuilds future labels directly from canonical perps trades starting at the
exact anchor + 3s decision point.

P1_PROTECTED is never read. Old Phase-7M future outcome files are never used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter, OrderedDict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_core.clocks.contract import (  # noqa: E402
    D0_END_EXCLUSIVE_NS,
    D0_START_NS,
    NS_PER_SECOND,
    REQUIRED_CLOCK_FIELDS,
    assert_clock_contract,
)
from research_core.substrate.d0 import (  # noqa: E402
    CANDIDATE_THRESHOLD_BPS,
    LEGACY_CANDIDATE_DEFINITION_ID,
    MAX_TRADE_STALE_NS,
    OUTCOME_HORIZONS_SEC,
    OUTCOME_TARGETS_BPS,
    PRIMARY_OBSERVATION_HORIZON_SEC,
    assert_causal_columns_future_clean,
    build_scientific_label_clock,
    frozen_candidate_availability_ts_ns,
    latest_prior_trade,
    summarize_outcome_horizon,
    summarize_outcome_horizons,
    validate_exact_anchor,
)

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pd = None
    pa = None
    pq = None

BUILDER_VERSION = "1.0.3"
SCHEMA_VERSION = 1
SCHEMA_NAME = "r0_corrected_d0_substrate_v1"
SYMBOL = "BTCUSDT"
EXPECTED_CANDIDATE_ROWS = 7_067
EXPECTED_CANDIDATE_DAYS = 62
FROZEN_POLICY_ID = "EARLY_25BP_RESET5_RELEASE10_REARM60"
FROZEN_RESET_BPS = 5.0
FROZEN_RELEASE_BPS = 10.0
FROZEN_CANDIDATE_BPS = 25.0
FROZEN_COOLDOWN_SEC = 60

DEFAULT_CANDIDATE_INPUTS = (
    REPO_ROOT
    / "data"
    / "derived"
    / "perps"
    / "research"
    / "prospective_25bp_exact_participant_interaction_v1"
    / "candidate_inputs.parquet"
)
DEFAULT_ANCHOR_ROOT = (
    REPO_ROOT
    / "data"
    / "derived"
    / "perps"
    / "research"
    / "prospective_25bp_refill_redepletion_sequence_v2"
)
DEFAULT_CANONICAL_ROOT = REPO_ROOT / "data" / "canonical" / "market_events_v1"
DEFAULT_EXECUTION_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "execution_foundation_v1"
DEFAULT_DATASET_INVENTORY = REPO_ROOT / "research" / "governance" / "datasets" / "D0_DATASET_MANIFEST_v1.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / SCHEMA_NAME

REQUIRED_CANDIDATE_COLUMNS = (
    "candidate_id",
    "policy_id",
    "candidate_date",
    "direction",
    "direction_label",
    "origin_second",
    "origin_price",
    "release_second",
    "candidate_second",
    "candidate_price",
    "candidate_displacement_bps",
    "reset_threshold_bps",
    "release_threshold_bps",
    "candidate_threshold_bps",
    "cooldown_sec",
    "early_threshold_bps",
)

OPTIONAL_CANDIDATE_COLUMNS = (
    "origin_kind",
    "is_primary_early_threshold",
    "is_reaccel_context",
    "release_to_candidate_sec",
    "seconds_since_prior_same_direction_candidate",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def progress(message: str) -> None:
    print(f"[R0.3 {datetime.now(timezone.utc).strftime('%H:%M:%S')}Z] {message}", flush=True)


def require_stack() -> None:
    if pd is None or pa is None or pq is None:
        raise SystemExit("R0.3 requires pandas and pyarrow in the project .venv")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        Path(tmp).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_csv(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_parquet(path: Path, table: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def parse_date_partition(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("date=") and len(part) == 15:
            try:
                return date.fromisoformat(part[5:]).isoformat()
            except Exception:
                return None
    return None


def d0_date(day: str) -> bool:
    d = date.fromisoformat(day)
    return date(2026, 5, 18) <= d <= date(2026, 7, 18)


def read_candidate_catalog(path: Path) -> Any:
    require_stack()
    if not path.exists():
        raise FileNotFoundError(f"missing frozen D0 candidate inputs: {path}")
    names = pq.ParquetFile(path).schema_arrow.names
    missing = [c for c in REQUIRED_CANDIDATE_COLUMNS if c not in names]
    if missing:
        raise RuntimeError(f"candidate input is missing required columns: {missing}")
    # Match the 7M.3 causal whitelist exactly. ``symbol`` was not persisted in
    # candidate_inputs.parquet; this adapter is BTCUSDT-specific and adds the
    # symbol explicitly in the governed output.
    projected = list(REQUIRED_CANDIDATE_COLUMNS) + [
        c for c in OPTIONAL_CANDIDATE_COLUMNS if c in names
    ]
    frame = pq.read_table(path, columns=projected).to_pandas()
    assert_causal_columns_future_clean(frame.columns)
    if frame["candidate_id"].astype(str).duplicated().any():
        raise RuntimeError("candidate_id must be unique")
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["candidate_date"] = frame["candidate_date"].astype(str)
    bad_dates = sorted(set(d for d in frame["candidate_date"] if not d0_date(d)))
    if bad_dates:
        raise RuntimeError(f"candidate inputs include non-D0 dates: {bad_dates[:10]}")

    # The adapter is deliberately tied to the frozen 7M.3 lineage input, not
    # any arbitrary table that happens to have similarly named columns.
    if len(frame) != EXPECTED_CANDIDATE_ROWS:
        raise RuntimeError(
            f"frozen D0 candidate lineage requires {EXPECTED_CANDIDATE_ROWS:,} rows; "
            f"found {len(frame):,}"
        )
    if frame["candidate_date"].nunique() != EXPECTED_CANDIDATE_DAYS:
        raise RuntimeError(
            f"frozen D0 candidate lineage requires {EXPECTED_CANDIDATE_DAYS} dates; "
            f"found {frame['candidate_date'].nunique()}"
        )
    policy_values = sorted(set(frame["policy_id"].astype(str)))
    if policy_values != [FROZEN_POLICY_ID]:
        raise RuntimeError(f"unexpected frozen policy IDs: {policy_values}")
    expected_numeric = {
        "reset_threshold_bps": FROZEN_RESET_BPS,
        "release_threshold_bps": FROZEN_RELEASE_BPS,
        "candidate_threshold_bps": FROZEN_CANDIDATE_BPS,
        "early_threshold_bps": FROZEN_CANDIDATE_BPS,
        "cooldown_sec": float(FROZEN_COOLDOWN_SEC),
    }
    for col, expected in expected_numeric.items():
        vals = pd.to_numeric(frame[col], errors="raise").astype(float)
        if not bool((vals == float(expected)).all()):
            got = sorted(set(vals.tolist()))[:10]
            raise RuntimeError(f"frozen candidate contract mismatch for {col}: {got}")
    origin = pd.to_numeric(frame["origin_second"], errors="raise").astype("int64")
    release = pd.to_numeric(frame["release_second"], errors="raise").astype("int64")
    candidate = pd.to_numeric(frame["candidate_second"], errors="raise").astype("int64")
    if not bool(((origin <= release) & (release <= candidate)).all()):
        raise RuntimeError("frozen candidate timing order origin <= release <= candidate is violated")
    return frame.sort_values(["candidate_second", "candidate_id"], kind="mergesort").reset_index(drop=True)


def _int64_arrow_field(path: Path, name: str) -> bool:
    if not path.exists():
        return False
    schema = pq.ParquetFile(path).schema_arrow
    if name not in schema.names:
        return False
    return pa.types.is_int64(schema.field(name).type)


def load_exact_anchor_index(
    anchor_root: Path, candidates: Any, *, emit_progress: bool = False
) -> tuple[dict[str, int], dict[str, str], list[dict[str, Any]]]:
    """Recover exact int64 anchors without coercing persisted float epochs."""

    require_stack()
    exact: dict[str, int] = {}
    reasons: dict[str, str] = {}
    provenance: list[dict[str, Any]] = []
    by_day = {d: g.copy() for d, g in candidates.groupby("candidate_date", sort=True)}

    total_days = len(by_day)
    for day_i, (day, day_candidates) in enumerate(by_day.items(), start=1):
        if emit_progress and (day_i == 1 or day_i % 5 == 0 or day_i == total_days):
            progress(f"anchor recovery: D0 day {day_i}/{total_days} ({day})")
        metrics_path = anchor_root / "candidate_metrics" / f"candidate_date={day}" / "part.parquet"
        levels_path = anchor_root / "level_fate" / f"candidate_date={day}" / "part.parquet"
        status_by_id: dict[str, dict[str, Any]] = {}

        if metrics_path.exists():
            names = pq.ParquetFile(metrics_path).schema_arrow.names
            use = [c for c in ("candidate_id", "observation_horizon_sec", "anchor_found", "sequence_valid", "invalid_reason", "anchor_ts_recv_ns") if c in names]
            m = pq.read_table(metrics_path, columns=use).to_pandas()
            if "observation_horizon_sec" in m.columns:
                m = m[pd.to_numeric(m["observation_horizon_sec"], errors="coerce") == PRIMARY_OBSERVATION_HORIZON_SEC].copy()
            if "candidate_id" in m.columns:
                m["candidate_id"] = m["candidate_id"].astype(str)
                if m["candidate_id"].duplicated().any():
                    raise RuntimeError(f"duplicate primary candidate metrics for {day}")
                for r in m.to_dict("records"):
                    status_by_id[str(r["candidate_id"])] = r
            if _int64_arrow_field(metrics_path, "anchor_ts_recv_ns") and "anchor_ts_recv_ns" in m.columns:
                for r in m.to_dict("records"):
                    raw = r.get("anchor_ts_recv_ns")
                    if pd.notna(raw):
                        exact[str(r["candidate_id"])] = int(raw)
            provenance.append({"path": str(metrics_path), "sha256": sha256_file(metrics_path), "role": "anchor_status"})

        # 7M.6F writes level_fate.anchor_ts_recv_ns with an explicit int64 Arrow
        # schema. Use it to recover exact anchors if candidate_metrics was
        # physically float because of nullable pandas materialization.
        if levels_path.exists():
            if not _int64_arrow_field(levels_path, "anchor_ts_recv_ns"):
                raise RuntimeError(f"level_fate anchor_ts_recv_ns is not int64: {levels_path}")
            names = pq.ParquetFile(levels_path).schema_arrow.names
            use = [c for c in ("candidate_id", "anchor_ts_recv_ns", "anchor_found") if c in names]
            lev = pq.read_table(levels_path, columns=use).to_pandas()
            if len(lev):
                lev["candidate_id"] = lev["candidate_id"].astype(str)
                lev = lev[lev["anchor_ts_recv_ns"].notna()].copy()
                for cid, g in lev.groupby("candidate_id", sort=False):
                    vals = sorted(set(int(v) for v in g["anchor_ts_recv_ns"].dropna().tolist()))
                    if len(vals) > 1:
                        raise RuntimeError(f"candidate {cid} has multiple exact anchors in {levels_path}: {vals[:5]}")
                    if len(vals) == 1:
                        if cid in exact and exact[cid] != vals[0]:
                            raise RuntimeError(f"candidate {cid} anchor disagrees between metrics and level_fate")
                        exact[cid] = vals[0]
            provenance.append({"path": str(levels_path), "sha256": sha256_file(levels_path), "role": "exact_anchor_int64"})

        for cid in day_candidates["candidate_id"].astype(str):
            if cid in exact:
                continue
            st = status_by_id.get(cid, {})
            if bool(st.get("anchor_found", False)):
                reasons[cid] = "ANCHOR_FOUND_BUT_NO_EXACT_INT64_PERSISTED_VALUE"
            else:
                reason = str(st.get("invalid_reason") or "EXACT_ANCHOR_UNAVAILABLE")
                reasons[cid] = reason

    return exact, reasons, provenance


def build_frozen_candidate_availability_index(candidates: Any) -> dict[str, int]:
    """Build causal availability clocks for the frozen one-second candidate policy.

    The historical candidate universe was generated from a one-second price
    series.  We therefore use the interval-close boundary as the conservative
    moment the frozen candidate observation is available.  Raw intrasecond
    threshold crossings are not used to redefine candidate identity or timing.
    """

    out: dict[str, int] = {}
    for r in candidates.to_dict("records"):
        out[str(r["candidate_id"])] = frozen_candidate_availability_ts_ns(
            int(r["candidate_second"])
        )
    return out


def build_candidate_and_clock_tables(
    candidates: Any,
    candidate_availability: Mapping[str, int],
    exact_anchors: Mapping[str, int],
    anchor_unavailable: Mapping[str, str],
) -> tuple[Any, Any, Any]:
    require_stack()
    candidate_rows: list[dict[str, Any]] = []
    clock_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for r in candidates.to_dict("records"):
        cid = str(r["candidate_id"])
        csec = int(r["candidate_second"])
        clock_ok = False
        detection = candidate_availability.get(cid)
        anchor = exact_anchors.get(cid)
        if detection is None:
            clock_reason = "FROZEN_CANDIDATE_AVAILABILITY_UNAVAILABLE"
        elif anchor is None:
            clock_reason = anchor_unavailable.get(cid, "EXACT_ANCHOR_UNAVAILABLE")
        else:
            clock_reason = "CLOCK_NOT_BUILT"
            try:
                anchor = validate_exact_anchor(csec, anchor)
                expected_availability = frozen_candidate_availability_ts_ns(csec)
                if int(detection) != expected_availability:
                    raise ValueError("candidate availability does not equal frozen one-second interval close")
                clock = build_scientific_label_clock(
                    anchor, candidate_detection_ts_ns=int(detection)
                )
                # Immediate clock fields must stay in D0. Longer horizon censoring
                # is handled separately so late-D0 candidates are preserved.
                if min(clock.values()) < D0_START_NS or max(clock.values()) >= D0_END_EXCLUSIVE_NS:
                    raise ValueError("governed clock reaches outside D0")
                assert_clock_contract(clock)
                clock_ok = True
                clock_reason = "PASS"
                clock_rows.append(
                    {
                        "candidate_id": cid,
                        "candidate_date": str(r["candidate_date"]),
                        "candidate_definition_id": LEGACY_CANDIDATE_DEFINITION_ID,
                        "clock_mode": "SCIENTIFIC_LABEL_CLOCK_NO_EXECUTION_MODEL",
                        "candidate_detection_semantics": "FROZEN_ONE_SECOND_POLICY_AVAILABLE_AT_INTERVAL_CLOSE",
                        "execution_semantics": "ZERO_LATENCY_PLACEHOLDER_FOR_CLOCK_AUDIT_ONLY",
                        **clock,
                    }
                )
            except Exception as exc:
                clock_reason = f"CLOCK_INVALID:{type(exc).__name__}:{exc}"

        candidate_rows.append(
            {
                "candidate_id": cid,
                "candidate_definition_id": LEGACY_CANDIDATE_DEFINITION_ID,
                "evidentiary_role": "D0_DEVELOPMENT_ONLY",
                "candidate_date": str(r["candidate_date"]),
                "symbol": str(r.get("symbol") or SYMBOL),
                "direction": int(r["direction"]),
                "direction_label": str(r.get("direction_label") or ("UP" if int(r["direction"]) > 0 else "DOWN")),
                "legacy_policy_id": str(r.get("policy_id") or ""),
                "origin_kind": str(r.get("origin_kind") or ""),
                "origin_price": float(r["origin_price"]),
                "candidate_price": float(r["candidate_price"]),
                "candidate_displacement_bps": float(r["candidate_displacement_bps"]),
                "reset_threshold_bps": float(r["reset_threshold_bps"]),
                "release_threshold_bps": float(r["release_threshold_bps"]),
                "candidate_threshold_bps": float(r["candidate_threshold_bps"]),
                "cooldown_sec": int(r["cooldown_sec"]),
                "early_threshold_bps": float(r["early_threshold_bps"]),
                "is_primary_early_threshold": (
                    None if pd.isna(r.get("is_primary_early_threshold"))
                    else bool(r.get("is_primary_early_threshold"))
                ),
                "is_reaccel_context": (
                    None if pd.isna(r.get("is_reaccel_context"))
                    else bool(r.get("is_reaccel_context"))
                ),
                "release_to_candidate_sec": (
                    None if pd.isna(r.get("release_to_candidate_sec"))
                    else float(r.get("release_to_candidate_sec"))
                ),
                "seconds_since_prior_same_direction_candidate": (
                    None if pd.isna(r.get("seconds_since_prior_same_direction_candidate"))
                    else float(r.get("seconds_since_prior_same_direction_candidate"))
                ),
                "legacy_origin_second": int(r["origin_second"]),
                "legacy_release_second": int(r["release_second"]),
                "legacy_candidate_second": csec,
                "candidate_policy_availability_available": bool(detection is not None),
                "exact_anchor_available": bool(anchor is not None),
                "clock_available": bool(clock_ok),
                "clock_unavailable_reason": None if clock_ok else clock_reason,
            }
        )
        audit_rows.append(
            {
                "candidate_id": cid,
                "candidate_date": str(r["candidate_date"]),
                "clock_status": "PASS" if clock_ok else "UNAVAILABLE",
                "reason": clock_reason,
            }
        )

    candidate_df = pd.DataFrame(candidate_rows).sort_values(["candidate_date", "legacy_candidate_second", "candidate_id"], kind="mergesort").reset_index(drop=True)
    clock_df = pd.DataFrame(clock_rows).sort_values(["outcome_start_ts_ns", "candidate_id"], kind="mergesort").reset_index(drop=True) if clock_rows else pd.DataFrame(columns=["candidate_id", *REQUIRED_CLOCK_FIELDS])
    audit_df = pd.DataFrame(audit_rows)
    assert_causal_columns_future_clean(candidate_df.columns)
    assert_causal_columns_future_clean(clock_df.columns)
    return candidate_df, clock_df, audit_df


def canonical_trade_paths(canonical_root: Path, day: str) -> list[Path]:
    if not d0_date(day):
        raise RuntimeError(f"R0.3 refuses non-D0 canonical date: {day}")
    root = canonical_root / "perps" / "trades" / f"date={day}"
    return sorted(root.glob("*.parquet")) if root.exists() else []


def execution_day_path(execution_root: Path, day: str) -> Path:
    if not d0_date(day):
        raise RuntimeError(f"R0.3 refuses non-D0 execution date: {day}")
    return execution_root / f"symbol={SYMBOL}" / f"date={day}" / f"part-{day}.parquet"


def load_day_trade_arrays(canonical_root: Path, days: Sequence[str]) -> tuple[list[int], list[float], list[dict[str, Any]]]:
    require_stack()
    parts = []
    prov: list[dict[str, Any]] = []
    for day in days:
        paths = canonical_trade_paths(canonical_root, day)
        if not paths:
            continue
        for p in paths:
            names = pq.ParquetFile(p).schema_arrow.names
            required = ["ts_recv_ns", "price", "symbol"]
            missing = [c for c in required if c not in names]
            if missing:
                raise RuntimeError(f"canonical trade file missing columns {missing}: {p}")
            tie_cols = [
                c for c in ("source_file_id", "source_record_number", "aggregate_trade_id")
                if c in names
            ]
            f = pq.read_table(p, columns=required + tie_cols).to_pandas()
            f = f[f["symbol"].astype(str) == SYMBOL].copy()
            if len(f):
                f["ts_recv_ns"] = pd.to_numeric(f["ts_recv_ns"], errors="raise").astype("int64")
                f["price"] = pd.to_numeric(f["price"], errors="coerce")
                f = f[f["price"].notna() & (f["price"] > 0)].copy()
                parts.append(f[["ts_recv_ns", "price", *tie_cols]])
            prov.append({"path": str(p), "role": "canonical_perps_trade"})
    if not parts:
        return [], [], prov
    allp = pd.concat(parts, ignore_index=True, sort=False)
    tie_order = [
        c for c in ("source_file_id", "source_record_number", "aggregate_trade_id")
        if c in allp.columns
    ]
    # Never collapse equal receive timestamps: multiple legitimate trades can
    # share one nanosecond. Stable canonical tie fields make the ordering
    # deterministic while preserving every observation.
    allp = allp.sort_values(["ts_recv_ns", *tie_order], kind="mergesort", na_position="last").reset_index(drop=True)
    return allp["ts_recv_ns"].astype("int64").tolist(), allp["price"].astype(float).tolist(), prov


def load_execution_seconds(execution_root: Path, days: Sequence[str]) -> tuple[set[int], list[dict[str, Any]]]:
    require_stack()
    secs: set[int] = set()
    prov: list[dict[str, Any]] = []
    for day in days:
        p = execution_day_path(execution_root, day)
        if not p.exists():
            continue
        names = pq.ParquetFile(p).schema_arrow.names
        if "receive_second" not in names:
            raise RuntimeError(f"execution partition missing receive_second: {p}")
        f = pq.read_table(p, columns=["receive_second"]).to_pandas()
        vals = pd.to_numeric(f["receive_second"], errors="coerce").dropna().astype("int64")
        secs.update(int(x) for x in vals.tolist())
        prov.append({"path": str(p), "role": "execution_1s_observability"})
    return secs, prov


def _days_needed(start_ns: int, end_ns: int) -> list[str]:
    start_day = datetime.fromtimestamp(start_ns // NS_PER_SECOND, tz=timezone.utc).date()
    end_day = datetime.fromtimestamp(end_ns // NS_PER_SECOND, tz=timezone.utc).date()
    out = []
    d = start_day
    while d <= end_day:
        if d >= date(2026, 7, 19):
            break
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def structural_seconds_complete(seconds: set[int], start_ns: int, end_ns: int) -> bool:
    start_sec = int(start_ns // NS_PER_SECOND)
    end_sec = int(end_ns // NS_PER_SECOND)
    return all(s in seconds for s in range(start_sec, end_sec + 1))


def outcome_arrow_schema() -> Any:
    require_stack()
    fields = [
        ("candidate_id", pa.string()),
        ("candidate_date", pa.string()),
        ("candidate_definition_id", pa.string()),
        ("direction", pa.int64()),
        ("outcome_horizon_sec", pa.int64()),
        ("outcome_start_ts_ns", pa.int64()),
        ("outcome_end_ts_ns", pa.int64()),
        ("outcome_status", pa.string()),
        ("outcome_observable", pa.bool_()),
        ("decision_price_valid", pa.bool_()),
        ("decision_price_ts_ns", pa.int64()),
        ("decision_price", pa.float64()),
        ("decision_price_age_ns", pa.int64()),
        ("future_trade_count", pa.int64()),
        ("mfe_from_decision_bps", pa.float64()),
        ("mae_from_decision_bps", pa.float64()),
        ("directional_return_to_last_trade_bps", pa.float64()),
        ("last_trade_ts_ns", pa.int64()),
        ("last_trade_price", pa.float64()),
    ]
    for t in OUTCOME_TARGETS_BPS:
        fields.extend(
            [
                (f"origin_reach_{t}", pa.bool_()),
                (f"origin_hit_{t}_ts_ns", pa.int64()),
                (f"seconds_to_origin_{t}_from_outcome_start", pa.float64()),
            ]
        )
    return pa.schema(fields)


def load_reusable_outcome_rows(
    output_root: Path,
    *,
    d0_manifest_path: Path,
    candidate_inputs_path: Path,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any] | None]:
    """Load prior R0.3 AVAILABLE rows only when lineage inputs are unchanged.

    R0.3.2 changes candidate/anchor clock ordering but does not change the
    anchor+3s outcome semantics for candidates that already had governed clocks.
    Reuse is therefore allowed only when the previous row has the same exact
    ``outcome_start_ts_ns`` at selection time.  Any newly clocked candidate is
    recomputed from canonical D0 data.
    """

    manifest_path = output_root / "metadata" / "manifest.json"
    outcome_path = output_root / "future" / "outcomes.parquet"
    if not (manifest_path.exists() and outcome_path.exists()):
        return {}, None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, None
    if str(manifest.get("schema_name")) != SCHEMA_NAME:
        return {}, None
    inputs = manifest.get("inputs", {}) if isinstance(manifest.get("inputs"), dict) else {}
    prior_d0 = inputs.get("r0_1_d0_manifest", {}) if isinstance(inputs.get("r0_1_d0_manifest"), dict) else {}
    prior_candidates = inputs.get("candidate_inputs", {}) if isinstance(inputs.get("candidate_inputs"), dict) else {}
    if str(prior_d0.get("sha256", "")) != sha256_file(d0_manifest_path):
        return {}, None
    if str(prior_candidates.get("sha256", "")) != sha256_file(candidate_inputs_path):
        return {}, None

    table = pq.read_table(outcome_path)
    required = {
        "candidate_id", "outcome_horizon_sec", "outcome_start_ts_ns",
        "outcome_status",
    }
    if not required.issubset(set(table.schema.names)):
        return {}, None
    rows = table.to_pylist()
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for r in rows:
        if str(r.get("outcome_status")) != "AVAILABLE":
            continue
        key = (str(r["candidate_id"]), int(r["outcome_horizon_sec"]))
        if key in out:
            raise RuntimeError(f"duplicate reusable outcome key: {key}")
        out[key] = r
    return out, {
        "path": str(outcome_path),
        "sha256": sha256_file(outcome_path),
        "prior_builder_version": manifest.get("builder_version"),
        "available_rows_loaded": len(out),
    }


def build_outcomes(
    candidates: Any,
    clocks: Any,
    canonical_root: Path,
    execution_root: Path,
    *,
    progress_every: int = 250,
    reusable_rows: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
) -> tuple[Any, list[dict[str, Any]], Any, dict[str, int]]:
    """Build frozen horizons, reusing unchanged exact-start rows when supplied."""

    require_stack()
    clock_by_id = clocks.set_index("candidate_id", drop=False).to_dict("index") if len(clocks) else {}
    cand_by_id = candidates.set_index("candidate_id", drop=False).to_dict("index")
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    reusable_rows = reusable_rows or {}
    reuse_stats = {"reused_available_rows": 0, "reused_candidates": 0, "recomputed_candidates": 0}

    trade_cache: OrderedDict[tuple[str, ...], tuple[list[int], list[float]]] = OrderedDict()
    exec_cache: OrderedDict[tuple[str, ...], set[int]] = OrderedDict()
    cache_limit = 4
    total_candidates = len(cand_by_id)
    started = time.perf_counter()

    def make_base(cid: str, c: Mapping[str, Any], horizon: int) -> dict[str, Any]:
        base = {
            "candidate_id": cid,
            "candidate_date": str(c["candidate_date"]),
            "candidate_definition_id": LEGACY_CANDIDATE_DEFINITION_ID,
            "direction": int(c["direction"]),
            "outcome_horizon_sec": int(horizon),
            "outcome_start_ts_ns": None,
            "outcome_end_ts_ns": None,
            "outcome_status": "CLOCK_UNAVAILABLE",
            "outcome_observable": False,
            "decision_price_valid": False,
            "decision_price_ts_ns": None,
            "decision_price": None,
            "decision_price_age_ns": None,
            "future_trade_count": 0,
            "mfe_from_decision_bps": None,
            "mae_from_decision_bps": None,
            "directional_return_to_last_trade_bps": None,
            "last_trade_ts_ns": None,
            "last_trade_price": None,
        }
        for t in OUTCOME_TARGETS_BPS:
            base[f"origin_reach_{t}"] = None
            base[f"origin_hit_{t}_ts_ns"] = None
            base[f"seconds_to_origin_{t}_from_outcome_start"] = None
        return base

    for candidate_i, (cid, c) in enumerate(cand_by_id.items(), start=1):
        if progress_every > 0 and (
            candidate_i == 1 or candidate_i % progress_every == 0 or candidate_i == total_candidates
        ):
            elapsed = time.perf_counter() - started
            progress(
                f"outcome rebuild: candidate {candidate_i}/{total_candidates} "
                f"({100.0 * candidate_i / max(1, total_candidates):.1f}%, elapsed {elapsed:.1f}s)"
            )

        bases = {h: make_base(cid, c, h) for h in OUTCOME_HORIZONS_SEC}
        clock = clock_by_id.get(cid)
        if clock is None:
            rows.extend(bases[h] for h in OUTCOME_HORIZONS_SEC)
            continue

        start_ns = int(clock["outcome_start_ts_ns"])
        reusable = []
        for horizon in OUTCOME_HORIZONS_SEC:
            rr = reusable_rows.get((cid, int(horizon)))
            if rr is None or int(rr.get("outcome_start_ts_ns") or -1) != start_ns:
                reusable = []
                break
            reusable.append(dict(rr))
        if len(reusable) == len(OUTCOME_HORIZONS_SEC):
            rows.extend(reusable)
            reuse_stats["reused_available_rows"] += len(reusable)
            reuse_stats["reused_candidates"] += 1
            continue

        reuse_stats["recomputed_candidates"] += 1
        in_d0_horizons: list[int] = []
        for horizon in OUTCOME_HORIZONS_SEC:
            end_ns = start_ns + int(horizon) * NS_PER_SECOND
            bases[horizon]["outcome_start_ts_ns"] = start_ns
            bases[horizon]["outcome_end_ts_ns"] = end_ns
            if end_ns >= D0_END_EXCLUSIVE_NS:
                bases[horizon]["outcome_status"] = "D0_BOUNDARY_CENSOR"
            else:
                in_d0_horizons.append(int(horizon))

        if not in_d0_horizons:
            rows.extend(bases[h] for h in OUTCOME_HORIZONS_SEC)
            continue

        max_end_ns = start_ns + max(in_d0_horizons) * NS_PER_SECOND
        days = tuple(_days_needed(max(D0_START_NS, start_ns - MAX_TRADE_STALE_NS), max_end_ns))
        if not days:
            for horizon in in_d0_horizons:
                bases[horizon]["outcome_status"] = "D0_SOURCE_DAY_UNAVAILABLE"
            rows.extend(bases[h] for h in OUTCOME_HORIZONS_SEC)
            continue

        if days in trade_cache:
            trade_cache.move_to_end(days)
        else:
            tts_new, tpx_new, prov = load_day_trade_arrays(canonical_root, days)
            trade_cache[days] = (tts_new, tpx_new)
            provenance.extend(prov)
            while len(trade_cache) > cache_limit:
                trade_cache.popitem(last=False)
        if days in exec_cache:
            exec_cache.move_to_end(days)
        else:
            secs_new, prov = load_execution_seconds(execution_root, days)
            exec_cache[days] = secs_new
            provenance.extend(prov)
            while len(exec_cache) > cache_limit:
                exec_cache.popitem(last=False)

        tts, tpx = trade_cache[days]
        secs = exec_cache[days]
        if not tts:
            for horizon in in_d0_horizons:
                bases[horizon]["outcome_status"] = "CANONICAL_TRADE_PARTITION_MISSING"
            rows.extend(bases[h] for h in OUTCOME_HORIZONS_SEC)
            continue

        structurally_complete: list[int] = []
        for horizon in in_d0_horizons:
            end_ns = start_ns + horizon * NS_PER_SECOND
            if structural_seconds_complete(secs, start_ns, end_ns):
                structurally_complete.append(horizon)
            else:
                bases[horizon]["outcome_status"] = "EXECUTION_1S_COVERAGE_INCOMPLETE"

        if not structurally_complete:
            rows.extend(bases[h] for h in OUTCOME_HORIZONS_SEC)
            continue

        dpts, dp = latest_prior_trade(tts, tpx, start_ns, max_stale_ns=MAX_TRADE_STALE_NS)
        if dpts is None or dp is None:
            for horizon in structurally_complete:
                bases[horizon]["outcome_status"] = "DECISION_PRICE_MISSING_OR_STALE"
            rows.extend(bases[h] for h in OUTCOME_HORIZONS_SEC)
            continue

        summaries = summarize_outcome_horizons(
            trade_ts_ns=tts,
            trade_prices=tpx,
            outcome_start_ts_ns=start_ns,
            decision_price=float(dp),
            origin_price=float(c["origin_price"]),
            direction=int(c["direction"]),
            horizons_seconds=structurally_complete,
        )
        for horizon in structurally_complete:
            base = bases[horizon]
            base["decision_price_valid"] = True
            base["decision_price_ts_ns"] = int(dpts)
            base["decision_price"] = float(dp)
            base["decision_price_age_ns"] = int(start_ns - dpts)
            base.update(summaries[horizon])
            base["outcome_status"] = "AVAILABLE"
            base["outcome_observable"] = True

        rows.extend(bases[h] for h in OUTCOME_HORIZONS_SEC)

    table = pa.Table.from_pylist(rows, schema=outcome_arrow_schema())
    cov = Counter(r["outcome_status"] for r in rows)
    for status, n in sorted(cov.items()):
        coverage_rows.append({"outcome_status": status, "rows": int(n)})
    return table, provenance, pd.DataFrame(coverage_rows), reuse_stats

def candidate_arrow_table(frame: Any) -> Any:
    require_stack()
    return pa.Table.from_pandas(frame, preserve_index=False)


def clock_arrow_table(frame: Any) -> Any:
    require_stack()
    if frame.empty:
        schema = pa.schema(
            [
                ("candidate_id", pa.string()),
                ("candidate_date", pa.string()),
                ("candidate_definition_id", pa.string()),
                ("clock_mode", pa.string()),
                ("candidate_detection_semantics", pa.string()),
                ("execution_semantics", pa.string()),
                *[(f, pa.int64()) for f in REQUIRED_CLOCK_FIELDS],
            ]
        )
        return pa.Table.from_pylist([], schema=schema)
    for f in REQUIRED_CLOCK_FIELDS:
        frame[f] = pd.to_numeric(frame[f], errors="raise").astype("int64")
    return pa.Table.from_pandas(frame, preserve_index=False)


def load_d0_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"R0.1 D0 manifest is required before R0.3: {path}")
    m = json.loads(path.read_text(encoding="utf-8"))
    if str(m.get("cohort_id", m.get("cohort", "D0"))) not in ("D0", ""):
        raise RuntimeError("unexpected cohort in D0 manifest")
    return m


def run(args: argparse.Namespace) -> int:
    require_stack()
    run_started = time.perf_counter()
    progress("stage 1/6: loading R0.1 D0 manifest and frozen 7M.3 candidate catalog")
    d0_manifest = load_d0_manifest(args.d0_manifest)
    candidates_raw = read_candidate_catalog(args.candidate_inputs)
    progress(f"stage 1/6 complete: {len(candidates_raw):,} frozen D0 candidates")

    progress("stage 2/6: constructing frozen one-second candidate-availability boundaries")
    candidate_availability = build_frozen_candidate_availability_index(candidates_raw)
    progress(f"stage 2/6 complete: {len(candidate_availability):,} candidate availability clocks")

    progress("stage 3/6: recovering exact int64 book anchors")
    anchors, anchor_unavailable, anchor_prov = load_exact_anchor_index(
        args.anchor_root, candidates_raw, emit_progress=True
    )
    progress(f"stage 3/6 complete: {len(anchors):,} exact anchors recovered")

    progress("stage 4/6: constructing governed D0 candidate and clock tables")
    candidate_df, clock_df, clock_audit = build_candidate_and_clock_tables(
        candidates_raw, candidate_availability, anchors, anchor_unavailable
    )
    progress(
        f"stage 4/6 complete: {len(clock_df):,}/{len(candidate_df):,} candidates have governed clocks"
    )

    # Future labels are rebuilt from canonical D0 data only. Historical 7M.3
    # candidate_decision_outcomes.parquet is intentionally not an input.
    reusable_rows: dict[tuple[str, int], dict[str, Any]] = {}
    reuse_provenance = None
    if not args.no_reuse_existing_outcomes:
        reusable_rows, reuse_provenance = load_reusable_outcome_rows(
            args.output_root,
            d0_manifest_path=args.d0_manifest,
            candidate_inputs_path=args.candidate_inputs,
        )
        if reusable_rows:
            progress(
                f"stage 5/6: {len(reusable_rows):,} prior AVAILABLE outcome rows are eligible for exact-start reuse"
            )
        else:
            progress("stage 5/6: no reusable prior outcome rows; performing full D0 outcome rebuild")
    else:
        progress("stage 5/6: prior outcome reuse disabled; performing full D0 outcome rebuild")

    outcome_table, outcome_prov, coverage_df, reuse_stats = build_outcomes(
        candidate_df, clock_df, args.canonical_root, args.execution_root,
        progress_every=args.progress_every,
        reusable_rows=reusable_rows,
    )
    progress(
        f"stage 5/6 complete: {outcome_table.num_rows:,} outcome rows built; "
        f"reused={reuse_stats['reused_available_rows']:,} rows, "
        f"recomputed_candidates={reuse_stats['recomputed_candidates']:,}"
    )
    progress("stage 6/6: writing atomic substrate outputs and manifests")

    causal_root = args.output_root / "causal"
    future_root = args.output_root / "future"
    metadata_root = args.output_root / "metadata"
    candidate_path = causal_root / "candidates.parquet"
    clock_path = causal_root / "clocks.parquet"
    outcome_path = future_root / "outcomes.parquet"
    manifest_path = metadata_root / "manifest.json"

    atomic_parquet(candidate_path, candidate_arrow_table(candidate_df))
    atomic_parquet(clock_path, clock_arrow_table(clock_df))
    atomic_parquet(outcome_path, outcome_table)
    atomic_csv(args.report_root / "clock_coverage.csv", clock_audit)
    atomic_csv(args.report_root / "outcome_coverage.csv", coverage_df)

    outcome_status_counts = {str(r["outcome_status"]): int(r["rows"]) for r in coverage_df.to_dict("records")}
    manifest = {
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "schema_name": SCHEMA_NAME,
        "generated_at_utc": utc_now_iso(),
        "evidentiary_role": "D0_DEVELOPMENT_ONLY",
        "candidate_definition_id": LEGACY_CANDIDATE_DEFINITION_ID,
        "candidate_definition_note": (
            "Historical prospective 25-bp population is preserved for lineage accounting; "
            "R0.3 does not elevate 25bp into a universal or confirmatory candidate definition."
        ),
        "clock_contract": {
            "exact_candidate_detection_required": False,
            "candidate_policy_availability_required": True,
            "exact_anchor_required": True,
            "legacy_candidate_second_promoted_to_market_event_timestamp": False,
            "candidate_detection_semantics": "FROZEN_ONE_SECOND_POLICY_AVAILABLE_AT_INTERVAL_CLOSE",
            "required_ordering": "candidate availability and recovered book anchor are distinct; both must be <= information_available_ts_ns",
            "feature_window": "anchor_ts_ns through anchor_ts_ns + 3s",
            "outcome_start": "anchor_ts_ns + 3s",
            "execution_model_present": False,
            "execution_fields": "zero-latency scientific-label placeholders only; R0.6 must replace for PnL",
        },
        "outcome_contract": {
            "source": "canonical perps trades only",
            "old_7m3_future_outcomes_reused": False,
            "r0_3_prior_rebuilt_outcomes_reused_when_exact_start_unchanged": bool(reuse_stats["reused_available_rows"]),
            "horizons_sec": list(OUTCOME_HORIZONS_SEC),
            "legacy_origin_targets_bps": list(OUTCOME_TARGETS_BPS),
            "decision_price_max_stale_ns": MAX_TRADE_STALE_NS,
            "future_path_begin": "strictly after outcome_start_ts_ns",
            "observability_basis": "D0 boundary + execution-foundation 1s structural coverage + canonical trade availability",
        },
        "protected_data": {
            "p1_start_ns": D0_END_EXCLUSIVE_NS,
            "p1_read": False,
            "non_d0_paths_refused": True,
            "late_d0_horizons": "retained as candidates; affected outcome rows are D0_BOUNDARY_CENSOR and no P1 files are opened",
        },
        "counts": {
            "candidate_rows": int(len(candidate_df)),
            "expected_candidate_rows": EXPECTED_CANDIDATE_ROWS,
            "expected_outcome_rows": EXPECTED_CANDIDATE_ROWS * len(OUTCOME_HORIZONS_SEC),
            "clock_rows": int(len(clock_df)),
            "clock_unavailable_rows": int((~candidate_df["clock_available"].astype(bool)).sum()),
            "outcome_rows": int(outcome_table.num_rows),
            "outcome_status_counts": outcome_status_counts,
            "reused_available_outcome_rows": int(reuse_stats["reused_available_rows"]),
            "reused_outcome_candidates": int(reuse_stats["reused_candidates"]),
            "recomputed_outcome_candidates": int(reuse_stats["recomputed_candidates"]),
        },
        "inputs": {
            "candidate_inputs": {"path": str(args.candidate_inputs), "sha256": sha256_file(args.candidate_inputs)},
            "candidate_availability_source": "frozen 7M.2/7K.1 one-second policy interval-close boundary",
            "anchor_root": str(args.anchor_root),
            "anchor_files": anchor_prov,
            "canonical_root": str(args.canonical_root),
            "execution_root": str(args.execution_root),
            "r0_1_d0_manifest": {"path": str(args.d0_manifest), "sha256": sha256_file(args.d0_manifest), "canonical_file_index_sha256": d0_manifest.get("canonical_file_index_sha256")},
            "outcome_source_file_count_this_run": int(len({x["path"] for x in outcome_prov})),
            "reused_prior_outcome_artifact": reuse_provenance,
        },
        "outputs": {
            "candidates": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
            "clocks": {"path": str(clock_path), "sha256": sha256_file(clock_path)},
            "outcomes": {"path": str(outcome_path), "sha256": sha256_file(outcome_path)},
        },
        "status": "BUILD_COMPLETE_REVIEW_REQUIRED",
    }
    atomic_json(manifest_path, manifest)
    atomic_json(args.report_root / "summary.json", manifest)

    progress(f"stage 6/6 complete: full R0.3 build finished in {time.perf_counter() - run_started:.1f}s")
    print(json.dumps({"status": manifest["status"], **manifest["counts"]}, indent=2, sort_keys=True), flush=True)
    return 0


def self_test() -> int:
    # Pure core tests so --self-test remains useful even before any local D0
    # artifacts are opened.
    anchor = 1_779_235_200_950_000_000
    candidate_available = 1_779_235_201_000_000_000
    clock = build_scientific_label_clock(
        anchor, candidate_detection_ts_ns=candidate_available
    )
    assert clock["outcome_start_ts_ns"] == anchor + 3 * NS_PER_SECOND
    assert clock["candidate_detection_ts_ns"] == candidate_available
    assert clock["candidate_detection_ts_ns"] > clock["anchor_ts_ns"]
    ts = [anchor + 2_900_000_000, anchor + 3_100_000_000, anchor + 3_500_000_000]
    px = [100.0, 100.2, 100.8]
    dts, dp = latest_prior_trade(ts, px, clock["outcome_start_ts_ns"])
    assert dts == ts[0] and abs(float(dp) - 100.0) < 1e-12
    out = summarize_outcome_horizon(
        trade_ts_ns=ts,
        trade_prices=px,
        outcome_start_ts_ns=clock["outcome_start_ts_ns"],
        decision_price=100.0,
        origin_price=100.0,
        direction=1,
        horizon_seconds=300,
        fully_observed=True,
        targets_bps=(50,),
    )
    assert out["origin_reach_50"] is True
    assert out["origin_hit_50_ts_ns"] == ts[2]
    print("SELF-TEST PASS: distinct candidate-availability/book-anchor clocks, anchor+3s outcome start, no pre-start trade leakage, and future first-passage semantics validated.")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-inputs", type=Path, default=DEFAULT_CANDIDATE_INPUTS)
    p.add_argument("--anchor-root", type=Path, default=DEFAULT_ANCHOR_ROOT)
    p.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    p.add_argument("--execution-root", type=Path, default=DEFAULT_EXECUTION_ROOT)
    p.add_argument("--d0-manifest", type=Path, default=DEFAULT_DATASET_INVENTORY)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="emit outcome-rebuild progress every N candidates; 0 disables candidate progress",
    )
    p.add_argument(
        "--no-reuse-existing-outcomes",
        action="store_true",
        help=(
            "force recomputation of every D0 outcome row instead of reusing prior "
            "R0.3 AVAILABLE rows whose exact outcome_start_ts_ns is unchanged"
        ),
    )
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
