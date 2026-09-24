#!/usr/bin/env python3
"""R0.4 — deterministic D0 dependence / episode framework.

This stage reads only the R0.3 causal candidate and clock tables plus their
manifest.  It never opens R0.3 future outcome values and never opens P1 data.

It constructs four complementary dependence structures:

1. exact label-window overlap components at 5m/10m/15m/30m;
2. an outcome-blind 60-second policy-gap temporal episode inherited from the
   frozen candidate policy's cooldown/rearm interval;
3. a direction-partitioned version of that temporal episode;
4. a conservative inference cluster equal to the union of UTC availability-day
   clustering and 30-minute overlap components, preserving cross-midnight
   overlap dependence.

The output is infrastructure, not a hypothesis test and not an inferential
claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import pandas as pd
except Exception:  # pragma: no cover - handled by require_stack
    pd = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - handled by require_stack
    pa = None
    pq = None

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.clocks.contract import (  # noqa: E402
    D0_END_EXCLUSIVE_NS,
    D0_START_NS,
    NS_PER_SECOND,
    REQUIRED_CLOCK_FIELDS,
    assert_clock_contract,
)
from research_core.dependence import (  # noqa: E402
    assert_component_refinement,
    connected_interval_components,
    conservative_day_overlap_clusters,
    exact_lineage_groups,
    gap_connected_episodes,
    partition_episodes_by_value,
)

SCHEMA_NAME = "r0_dependence_framework_v1"
SCHEMA_VERSION = "1.0.0"
BUILDER_VERSION = "1.0.0"
PARENT_SCHEMA_NAME = "r0_corrected_d0_substrate_v1"
EXPECTED_CANDIDATE_ROWS = 7_067
EXPECTED_CLOCK_ROWS = 7_031
EXPECTED_CLOCK_UNAVAILABLE = 36
POLICY_EPISODE_GAP_SEC = 60
HORIZONS_SEC = (300, 600, 900, 1800)
HORIZON_LABELS = {300: "5m", 600: "10m", 900: "15m", 1800: "30m"}

DEFAULT_R03_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" / PARENT_SCHEMA_NAME
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
)
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / SCHEMA_NAME

REQUIRED_CANDIDATE_COLUMNS = (
    "candidate_id",
    "candidate_date",
    "direction",
    "legacy_origin_second",
    "legacy_candidate_second",
    "cooldown_sec",
    "clock_available",
    "clock_unavailable_reason",
)

REQUIRED_CLOCK_COLUMNS = (
    "candidate_id",
    "candidate_date",
    *REQUIRED_CLOCK_FIELDS,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def progress(message: str) -> None:
    print(f"[R0.4 {datetime.now(timezone.utc).strftime('%H:%M:%S')}Z] {message}", flush=True)


def require_stack() -> None:
    if pd is None or pa is None or pq is None:
        raise SystemExit("R0.4 requires pandas and pyarrow in the project .venv")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        Path(tmp).write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
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
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def utc_date_from_ns(ts_ns: int) -> str:
    seconds = int(ts_ns) // NS_PER_SECOND
    return datetime.fromtimestamp(seconds, tz=timezone.utc).date().isoformat()


def _require_columns(frame: Any, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")


def load_parent_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing R0.3 manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_name") != PARENT_SCHEMA_NAME:
        raise RuntimeError(f"unexpected R0.3 schema_name: {manifest.get('schema_name')!r}")
    if bool(manifest.get("protected_data", {}).get("p1_read")):
        raise RuntimeError("R0.3 manifest reports P1 was read; R0.4 refuses this substrate")
    counts = manifest.get("counts", {})
    expected = {
        "candidate_rows": EXPECTED_CANDIDATE_ROWS,
        "clock_rows": EXPECTED_CLOCK_ROWS,
        "clock_unavailable_rows": EXPECTED_CLOCK_UNAVAILABLE,
    }
    for key, value in expected.items():
        if int(counts.get(key, -1)) != value:
            raise RuntimeError(
                f"R0.4 requires accepted corrected R0.3 count {key}={value}; "
                f"found {counts.get(key)!r}"
            )
    return manifest


def load_and_validate_parent(r03_root: Path) -> tuple[Any, Any, dict[str, Any], dict[str, str]]:
    require_stack()
    manifest_path = r03_root / "metadata" / "manifest.json"
    candidate_path = r03_root / "causal" / "candidates.parquet"
    clock_path = r03_root / "causal" / "clocks.parquet"
    for path in (manifest_path, candidate_path, clock_path):
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = load_parent_manifest(manifest_path)
    outputs = manifest.get("outputs", {})
    expected_candidate_hash = str(outputs.get("candidates", {}).get("sha256", ""))
    expected_clock_hash = str(outputs.get("clocks", {}).get("sha256", ""))
    candidate_hash = sha256_file(candidate_path)
    clock_hash = sha256_file(clock_path)
    if not expected_candidate_hash or candidate_hash != expected_candidate_hash:
        raise RuntimeError("R0.3 candidate parquet hash does not match its manifest")
    if not expected_clock_hash or clock_hash != expected_clock_hash:
        raise RuntimeError("R0.3 clock parquet hash does not match its manifest")

    candidates = pq.read_table(candidate_path).to_pandas()
    clocks = pq.read_table(clock_path).to_pandas()
    _require_columns(candidates, REQUIRED_CANDIDATE_COLUMNS, "R0.3 candidates")
    _require_columns(clocks, REQUIRED_CLOCK_COLUMNS, "R0.3 clocks")

    if len(candidates) != EXPECTED_CANDIDATE_ROWS or len(clocks) != EXPECTED_CLOCK_ROWS:
        raise RuntimeError("R0.3 parquet row counts do not match accepted R0.3 counts")
    if candidates["candidate_id"].astype(str).duplicated().any():
        raise RuntimeError("R0.3 candidate_id must be unique")
    if clocks["candidate_id"].astype(str).duplicated().any():
        raise RuntimeError("R0.3 clock candidate_id must be unique")

    candidates = candidates.copy()
    clocks = clocks.copy()
    candidates["candidate_id"] = candidates["candidate_id"].astype(str)
    clocks["candidate_id"] = clocks["candidate_id"].astype(str)
    candidates["candidate_date"] = candidates["candidate_date"].astype(str)
    clocks["candidate_date"] = clocks["candidate_date"].astype(str)

    if int(candidates["clock_available"].astype(bool).sum()) != EXPECTED_CLOCK_ROWS:
        raise RuntimeError("candidate clock_available count disagrees with accepted R0.3 clock count")
    if set(clocks["candidate_id"]) != set(
        candidates.loc[candidates["clock_available"].astype(bool), "candidate_id"]
    ):
        raise RuntimeError("R0.3 clocks do not exactly match candidates flagged clock_available")

    cooldown = pd.to_numeric(candidates["cooldown_sec"], errors="raise")
    if not bool((cooldown == POLICY_EPISODE_GAP_SEC).all()):
        raise RuntimeError(
            f"R0.4 policy-gap episode requires frozen cooldown={POLICY_EPISODE_GAP_SEC}s"
        )
    direction = pd.to_numeric(candidates["direction"], errors="raise").astype("int64")
    if not set(direction.unique()).issubset({-1, 1}):
        raise RuntimeError("candidate direction must be -1 or +1")

    clock_records = clocks.to_dict("records")
    for record in clock_records:
        governed = {field: record[field] for field in REQUIRED_CLOCK_FIELDS}
        # Convert numpy integer scalars to Python ints without accepting floats.
        for field, value in list(governed.items()):
            if isinstance(value, float):
                raise RuntimeError(f"R0.3 clock field {field} unexpectedly materialized as float")
            governed[field] = int(value)
        assert_clock_contract(governed)
        if governed["outcome_start_ts_ns"] + max(HORIZONS_SEC) * NS_PER_SECOND > D0_END_EXCLUSIVE_NS:
            raise RuntimeError(
                "accepted R0.3 governed clock has a 30m label window crossing P1_PROTECTED"
            )

    provenance = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "candidates_path": str(candidate_path),
        "candidates_sha256": candidate_hash,
        "clocks_path": str(clock_path),
        "clocks_sha256": clock_hash,
    }
    return candidates, clocks, manifest, provenance


def _component_stats(assignments: dict[str, Any], id_attr: str) -> dict[str, Any]:
    by_component: dict[str, int] = {}
    for assignment in assignments.values():
        cid = str(getattr(assignment, id_attr))
        if hasattr(assignment, "component_size"):
            size = int(assignment.component_size)
        elif hasattr(assignment, "episode_size"):
            size = int(assignment.episode_size)
        else:
            raise TypeError(f"assignment {type(assignment).__name__} has no size field")
        if cid in by_component and by_component[cid] != size:
            raise AssertionError(f"inconsistent size for {cid}")
        by_component[cid] = size
    sizes = sorted(by_component.values())
    if not sizes:
        return {
            "component_count": 0,
            "singleton_components": 0,
            "candidates_in_multi_member_components": 0,
            "candidate_share_multi_member": None,
            "max_component_size": 0,
            "median_component_size": None,
            "p95_component_size": None,
            "top1_component_candidate_share": None,
            "top5_component_candidate_share": None,
            "top10_component_candidate_share": None,
        }
    s = pd.Series(sizes, dtype="float64")
    total = int(sum(sizes))
    multi = int(sum(x for x in sizes if x > 1))
    desc = sorted(sizes, reverse=True)
    return {
        "component_count": int(len(sizes)),
        "singleton_components": int(sum(x == 1 for x in sizes)),
        "candidates_in_multi_member_components": multi,
        "candidate_share_multi_member": float(multi / total),
        "max_component_size": int(max(sizes)),
        "median_component_size": float(s.median()),
        "p95_component_size": float(s.quantile(0.95)),
        "top1_component_candidate_share": float(sum(desc[:1]) / total),
        "top5_component_candidate_share": float(sum(desc[:5]) / total),
        "top10_component_candidate_share": float(sum(desc[:10]) / total),
    }


def _gap_row(name: str, gaps_ns: list[int]) -> dict[str, Any]:
    gaps_sec = pd.Series([x / NS_PER_SECOND for x in gaps_ns], dtype="float64")
    row: dict[str, Any] = {"gap_type": name, "count": int(len(gaps_ns))}
    for q, label in ((0.10, "p10"), (0.25, "p25"), (0.50, "median"), (0.75, "p75"), (0.90, "p90"), (0.95, "p95"), (0.99, "p99")):
        row[f"{label}_sec"] = None if gaps_sec.empty else float(gaps_sec.quantile(q))
    row["max_sec"] = None if gaps_sec.empty else float(gaps_sec.max())
    return row


def build_gap_summary(rows: list[dict[str, Any]]) -> Any:
    ordered = sorted(rows, key=lambda r: (int(r["candidate_time_ns"]), str(r["candidate_id"])))
    overall = [int(b["candidate_time_ns"]) - int(a["candidate_time_ns"]) for a, b in zip(ordered, ordered[1:])]
    opposite = [
        int(b["candidate_time_ns"]) - int(a["candidate_time_ns"])
        for a, b in zip(ordered, ordered[1:])
        if int(a["direction"]) != int(b["direction"])
    ]
    same: list[int] = []
    for direction in (-1, 1):
        group = [r for r in ordered if int(r["direction"]) == direction]
        same.extend(
            int(b["candidate_time_ns"]) - int(a["candidate_time_ns"])
            for a, b in zip(group, group[1:])
        )
    return pd.DataFrame(
        [
            _gap_row("ALL_CONSECUTIVE", overall),
            _gap_row("SAME_DIRECTION_CONSECUTIVE", same),
            _gap_row("OPPOSITE_DIRECTION_ADJACENT", opposite),
        ]
    )


def build_framework(candidates: Any, clocks: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    clock_by_id = clocks.set_index("candidate_id", drop=False).to_dict("index")
    rows: list[dict[str, Any]] = []
    for c in candidates.to_dict("records"):
        cid = str(c["candidate_id"])
        csec = int(c["legacy_candidate_second"])
        candidate_time_ns = (csec + 1) * NS_PER_SECOND
        if candidate_time_ns < D0_START_NS or candidate_time_ns > D0_END_EXCLUSIVE_NS:
            raise RuntimeError(f"candidate availability outside governed D0 boundary: {cid}")
        rows.append(
            {
                "candidate_id": cid,
                "candidate_date": str(c["candidate_date"]),
                "candidate_availability_date_utc": utc_date_from_ns(candidate_time_ns),
                "direction": int(c["direction"]),
                "legacy_origin_second": int(c["legacy_origin_second"]),
                "legacy_candidate_second": csec,
                "candidate_time_ns": candidate_time_ns,
                "clock_available": bool(c["clock_available"]),
                "clock_unavailable_reason": None if bool(c["clock_available"]) else c.get("clock_unavailable_reason"),
            }
        )

    # Candidate-level spacing fields are structural diagnostics only.  They use
    # the frozen candidate-availability clock and never inspect future outcomes.
    ordered_rows = sorted(rows, key=lambda r: (int(r["candidate_time_ns"]), str(r["candidate_id"])))
    prev_gap: dict[str, float | None] = {str(r["candidate_id"]): None for r in rows}
    next_gap: dict[str, float | None] = {str(r["candidate_id"]): None for r in rows}
    for left, right in zip(ordered_rows, ordered_rows[1:]):
        gap_sec = (int(right["candidate_time_ns"]) - int(left["candidate_time_ns"])) / NS_PER_SECOND
        next_gap[str(left["candidate_id"])] = float(gap_sec)
        prev_gap[str(right["candidate_id"])] = float(gap_sec)

    prev_same: dict[str, float | None] = {str(r["candidate_id"]): None for r in rows}
    next_same: dict[str, float | None] = {str(r["candidate_id"]): None for r in rows}
    for direction in (-1, 1):
        group = [r for r in ordered_rows if int(r["direction"]) == direction]
        for left, right in zip(group, group[1:]):
            gap_sec = (int(right["candidate_time_ns"]) - int(left["candidate_time_ns"])) / NS_PER_SECOND
            next_same[str(left["candidate_id"])] = float(gap_sec)
            prev_same[str(right["candidate_id"])] = float(gap_sec)

    temporal = gap_connected_episodes(
        rows,
        max_gap_ns=POLICY_EPISODE_GAP_SEC * NS_PER_SECOND,
        id_field="candidate_id",
        time_field="candidate_time_ns",
        prefix="POL60",
    )
    direction_by_id = {r["candidate_id"]: int(r["direction"]) for r in rows}
    directional = partition_episodes_by_value(temporal, direction_by_id, prefix="DIR60")
    lineage = exact_lineage_groups(
        rows,
        group_fields=("direction", "legacy_origin_second"),
        id_field="candidate_id",
        time_field="candidate_time_ns",
        prefix="ORIGIN",
    )

    overlap: dict[int, dict[str, Any]] = {}
    for horizon in HORIZONS_SEC:
        intervals = []
        for cid, clock in clock_by_id.items():
            start = int(clock["outcome_start_ts_ns"])
            intervals.append(
                {
                    "candidate_id": cid,
                    "start_ns": start,
                    "end_ns": start + horizon * NS_PER_SECOND,
                }
            )
        overlap[horizon] = connected_interval_components(
            intervals,
            prefix=f"OV{HORIZON_LABELS[horizon].upper()}",
        )

    for fine, coarse in zip(HORIZONS_SEC, HORIZONS_SEC[1:]):
        assert_component_refinement(overlap[fine], overlap[coarse])

    day_by_id = {r["candidate_id"]: str(r["candidate_availability_date_utc"]) for r in rows}
    conservative = conservative_day_overlap_clusters(day_by_id, overlap[1800], prefix="DAY30")
    conservative_sizes = Counter(conservative.values())

    output_rows: list[dict[str, Any]] = []
    for r in rows:
        cid = str(r["candidate_id"])
        t = temporal[cid]
        d = directional[cid]
        lin = lineage[cid]
        out = dict(r)
        out.update(
            {
                "prev_candidate_gap_sec": prev_gap[cid],
                "next_candidate_gap_sec": next_gap[cid],
                "prev_same_direction_gap_sec": prev_same[cid],
                "next_same_direction_gap_sec": next_same[cid],
            }
        )
        out.update(
            {
                "policy_temporal_episode_id": t.episode_id,
                "policy_temporal_episode_size": int(t.episode_size),
                "policy_temporal_episode_start_ts_ns": int(t.episode_start_ns),
                "policy_temporal_episode_end_ts_ns": int(t.episode_end_ns),
                "directional_episode_id": d.episode_id,
                "directional_episode_size": int(d.episode_size),
                "origin_lineage_episode_id": lin.episode_id,
                "origin_lineage_episode_size": int(lin.episode_size),
                "conservative_inference_cluster_id": conservative[cid],
                "conservative_inference_cluster_size": int(conservative_sizes[conservative[cid]]),
            }
        )
        clock = clock_by_id.get(cid)
        out["outcome_start_ts_ns"] = None if clock is None else int(clock["outcome_start_ts_ns"])
        for horizon in HORIZONS_SEC:
            label = HORIZON_LABELS[horizon]
            assignment = overlap[horizon].get(cid)
            out[f"overlap_component_{label}"] = None if assignment is None else assignment.component_id
            out[f"overlap_component_size_{label}"] = None if assignment is None else int(assignment.component_size)
            out[f"overlap_window_end_ts_ns_{label}"] = (
                None if clock is None else int(clock["outcome_start_ts_ns"]) + horizon * NS_PER_SECOND
            )
        output_rows.append(out)

    frame = pd.DataFrame(output_rows).sort_values(
        ["candidate_time_ns", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)

    component_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS_SEC:
        stats = _component_stats(overlap[horizon], "component_id")
        component_rows.append(
            {
                "structure": f"OVERLAP_{HORIZON_LABELS[horizon].upper()}",
                "eligible_candidates": int(len(overlap[horizon])),
                **stats,
            }
        )
    for name, assignments in (
        ("POLICY_TEMPORAL_60S", temporal),
        ("DIRECTIONAL_WITHIN_POLICY_60S", directional),
        ("EXACT_ORIGIN_LINEAGE", lineage),
    ):
        component_rows.append(
            {
                "structure": name,
                "eligible_candidates": int(len(assignments)),
                **_component_stats(assignments, "episode_id"),
            }
        )
    conservative_fake = {
        key: type("A", (), {"episode_id": cluster, "episode_size": conservative_sizes[cluster]})()
        for key, cluster in conservative.items()
    }
    component_rows.append(
        {
            "structure": "CONSERVATIVE_DAY_UNION_30M_OVERLAP",
            "eligible_candidates": int(len(conservative)),
            **_component_stats(conservative_fake, "episode_id"),
        }
    )
    component_summary = pd.DataFrame(component_rows)
    gap_summary = build_gap_summary(rows)
    day_summary = (
        frame.groupby(["candidate_availability_date_utc"], as_index=False)
        .agg(
            candidates=("candidate_id", "size"),
            up_candidates=("direction", lambda s: int((s > 0).sum())),
            down_candidates=("direction", lambda s: int((s < 0).sum())),
            governed_clocks=("clock_available", lambda s: int(s.astype(bool).sum())),
        )
        .sort_values("candidate_availability_date_utc", kind="mergesort")
        .reset_index(drop=True)
    )

    clock_unavailable = frame.loc[~frame["clock_available"].astype(bool), "clock_unavailable_reason"]
    summary = {
        "candidate_rows": int(len(frame)),
        "clock_available_rows": int(frame["clock_available"].astype(bool).sum()),
        "clock_unavailable_rows": int((~frame["clock_available"].astype(bool)).sum()),
        "clock_unavailable_reason_counts": {
            str(k): int(v) for k, v in clock_unavailable.fillna("UNKNOWN").value_counts().items()
        },
        "direction_counts": {str(int(k)): int(v) for k, v in frame["direction"].value_counts().sort_index().items()},
        "availability_day_count": int(frame["candidate_availability_date_utc"].nunique()),
        "max_candidates_on_one_day": int(day_summary["candidates"].max()),
        "policy_episode_gap_sec": POLICY_EPISODE_GAP_SEC,
        "structures": {
            str(r["structure"]): {
                k: (int(v) if isinstance(v, int) else v)
                for k, v in r.items()
                if k != "structure"
            }
            for r in component_rows
        },
    }
    return frame, component_summary, gap_summary, day_summary, summary


def run(args: argparse.Namespace) -> int:
    require_stack()
    progress("stage 1/4: validating frozen R0.3 causal candidate/clock artifacts and hashes")
    candidates, clocks, parent_manifest, provenance = load_and_validate_parent(args.r03_root)
    progress(
        f"stage 1/4 complete: {len(candidates):,} candidates, {len(clocks):,} governed clocks; future outcomes not opened"
    )

    progress("stage 2/4: constructing exact overlap components and outcome-blind episodes")
    frame, component_summary, gap_summary, day_summary, summary = build_framework(candidates, clocks)
    progress("stage 2/4 complete: dependence structures constructed")

    progress("stage 3/4: validating preservation, nesting, and protected-data invariants")
    if len(frame) != EXPECTED_CANDIDATE_ROWS:
        raise AssertionError("candidate preservation failed")
    if int(frame["clock_available"].astype(bool).sum()) != EXPECTED_CLOCK_ROWS:
        raise AssertionError("governed-clock preservation failed")
    if frame["candidate_id"].duplicated().any():
        raise AssertionError("duplicate candidate_id in dependence table")
    if any(str(p).startswith(str(args.r03_root / "future")) for p in (provenance["candidates_path"], provenance["clocks_path"])):
        raise AssertionError("future artifact path entered R0.4 input provenance")
    progress("stage 3/4 complete: all R0.4 invariants pass")

    progress("stage 4/4: writing dependence table, diagnostics, and manifest")
    dependence_path = args.output_root / "dependence" / "candidate_dependence.parquet"
    manifest_path = args.output_root / "metadata" / "manifest.json"
    atomic_parquet(dependence_path, pa.Table.from_pandas(frame, preserve_index=False))
    atomic_csv(args.report_root / "component_summary.csv", component_summary)
    atomic_csv(args.report_root / "gap_summary.csv", gap_summary)
    atomic_csv(args.report_root / "day_concentration.csv", day_summary)

    manifest = {
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "schema_name": SCHEMA_NAME,
        "generated_at_utc": utc_now_iso(),
        "evidentiary_role": "D0_DEPENDENCE_INFRASTRUCTURE_ONLY",
        "parent": {
            "schema_name": PARENT_SCHEMA_NAME,
            "accepted_checkpoint_commit": "e9b3ac699213503348a0549b24e3a9bff02cfa88",
            "accepted_checkpoint_tag": "r0-causal-clock-substrate-v1",
            **provenance,
        },
        "outcome_firewall": {
            "future_outcome_values_read": False,
            "r0_3_future_outcomes_path_opened": False,
            "p1_read": False,
            "dependence_inputs": "R0.3 causal candidates + governed clocks only",
        },
        "dependence_contract": {
            "overlap_interval_semantics": "half-open [outcome_start_ts_ns, outcome_start_ts_ns + horizon)",
            "overlap_horizons_sec": list(HORIZONS_SEC),
            "touching_boundaries_overlap": False,
            "transitive_overlap_components": True,
            "policy_temporal_episode_gap_sec": POLICY_EPISODE_GAP_SEC,
            "policy_temporal_episode_gap_source": "frozen EARLY_25BP_RESET5_RELEASE10_REARM60 cooldown/rearm interval; not outcome tuned",
            "directional_episode": "exact direction partition within each 60s policy temporal episode",
            "origin_lineage_episode": "exact (direction, legacy_origin_second) tuple",
            "conservative_inference_cluster": "union of UTC candidate-availability day and 30m overlap components",
            "primary_claim_rule": (
                "R0.4 freezes dependence structures, not a hypothesis-specific D1 estimator. "
                "A future confirmation plan must pre-register the conservative cluster or an explicit hierarchy/sensitivity rule before D1."
            ),
        },
        "counts": summary,
        "outputs": {
            "candidate_dependence": {
                "path": str(dependence_path),
                "sha256": sha256_file(dependence_path),
            },
            "component_summary": str(args.report_root / "component_summary.csv"),
            "gap_summary": str(args.report_root / "gap_summary.csv"),
            "day_concentration": str(args.report_root / "day_concentration.csv"),
        },
        "status": "BUILD_COMPLETE_REVIEW_REQUIRED",
    }
    atomic_json(manifest_path, manifest)
    atomic_json(args.report_root / "summary.json", manifest)
    progress("stage 4/4 complete")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "candidate_rows": summary["candidate_rows"],
                "clock_available_rows": summary["clock_available_rows"],
                "clock_unavailable_rows": summary["clock_unavailable_rows"],
                "availability_day_count": summary["availability_day_count"],
                "structures": summary["structures"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def self_test() -> int:
    from research_core.dependence import (
        connected_interval_components,
        conservative_day_overlap_clusters,
        gap_connected_episodes,
        partition_episodes_by_value,
    )

    intervals = [
        {"candidate_id": "A", "start_ns": 0, "end_ns": 10},
        {"candidate_id": "B", "start_ns": 9, "end_ns": 20},
        {"candidate_id": "C", "start_ns": 19, "end_ns": 30},
        {"candidate_id": "D", "start_ns": 30, "end_ns": 40},
    ]
    ov = connected_interval_components(intervals, prefix="T")
    assert ov["A"].component_id == ov["B"].component_id == ov["C"].component_id
    assert ov["D"].component_id != ov["C"].component_id  # touching only

    rows = [
        {"candidate_id": "A", "candidate_time_ns": 0},
        {"candidate_id": "B", "candidate_time_ns": 60},
        {"candidate_id": "C", "candidate_time_ns": 121},
    ]
    ep = gap_connected_episodes(rows, max_gap_ns=60, prefix="E")
    assert ep["A"].episode_id == ep["B"].episode_id
    assert ep["C"].episode_id != ep["B"].episode_id
    dep = partition_episodes_by_value(ep, {"A": 1, "B": -1, "C": 1}, prefix="D")
    assert dep["A"].episode_id != dep["B"].episode_id

    day = {"A": "2026-01-01", "B": "2026-01-02", "C": "2026-01-02"}
    day30 = conservative_day_overlap_clusters(day, {"A": ov["A"], "B": ov["B"]}, prefix="X")
    assert day30["A"] == day30["B"] == day30["C"]

    print(
        "SELF-TEST PASS: half-open transitive overlap, frozen-gap episodes, direction partitioning, and conservative day/30m union validated."
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r03-root", type=Path, default=DEFAULT_R03_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
