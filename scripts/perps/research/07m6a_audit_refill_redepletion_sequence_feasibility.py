#!/usr/bin/env python3
"""
Phase 7M.6A — Frozen-substrate refill→re-depletion sequence feasibility audit
=============================================================================

PLAIN-ENGLISH QUESTION
----------------------
Phase 7M.5 showed that, after similar initial opposing-queue destruction,
continuing BTCUSDT 25-bp moves often have MORE opposing refill across MORE
previously depleted levels.  It did not establish that the replacement
liquidity itself is subsequently consumed again.

The next scientific question is therefore:

    After an opposing level is depleted and then refilled, can the frozen
    Phase 7M.3 outputs tell us whether that replacement liquidity is later
    depleted again before the causal decision horizon?

This script does NOT answer the outcome question and does NOT load any future
labels.  It is only a cheap feasibility audit of the already-frozen Phase 7M.3
substrate.

WHY THIS AUDIT EXISTS
---------------------
To detect the physical sequence

    deplete → refill → re-deplete

we need more than aggregate totals such as total depletion, total refill,
refill breadth, or endpoint liquidity.  Ideally we need either:

A) event/update-level causal rows containing price level + quantity state/delta
   + timestamp/elapsed time, so the sequence can be reconstructed directly; or

B) per-level frozen fields that explicitly encode post-refill re-depletion /
   refill consumption after the first refill.

If 7M.3 saved either A or B, the next analysis can remain cheap.  If it saved
only aggregates/first-refill/endpoint state, a true refill-fate sequence cannot
be identified from the frozen substrate and we must decide separately whether
an additional raw replay is scientifically justified.

IMPORTANT SCIENTIFIC RULES
--------------------------
* No raw depth/trade replay.
* No candidate regeneration.
* No future outcome file may be opened.
* No matching and no thresholds are optimized here.
* No ML and no composite signal.
* This audit does not declare any feature predictive.
* A missing capability is a valid scientific result; do not weaken the
  definition of re-depletion merely to avoid replay.

FROZEN INPUT CONTRACT
---------------------
Expected Phase 7M.3 root:

data/derived/perps/research/prospective_25bp_exact_participant_interaction_v1/

Expected frozen metadata:
    schema_name     prospective_25bp_exact_participant_interaction_v1
    builder_version 1.0.0
    policy          EARLY_25BP_RESET5_RELEASE10_REARM60
    candidates      7,067
    horizons        1 / 3 / 5 / 10 sec

KNOWN 7M.3 AGGREGATES
---------------------
The main candidate_mechanics schema already contains useful summaries such as:
    opposing__same_price_refill_after_depletion_quote
    opposing__same_price_refill_to_depletion_ratio
    opposing__refilled_depleted_level_share
    opposing__weighted_median_first_refill_latency_ms
    opposing__durable_same_price_refill_fraction
    opposing__same_price_current_to_initial_ratio

Those are useful mechanics, but by themselves they do NOT prove that the same
replacement quantity was subsequently consumed.  This script therefore audits
all other causal parquet artifacts, especially level_audit, for finer temporal
information.

PASS / FAIL MEANING
-------------------
SUFFICIENT_EVENT_SEQUENCE
    At least one frozen causal dataset contains event/update-level information
    sufficient in principle to reconstruct price-level quantity changes in
    temporal order.

SUFFICIENT_EXPLICIT_REDEP_FIELDS
    No raw event sequence is present, but frozen per-level fields explicitly
    encode post-refill depletion/consumption strongly enough for a cheap next
    phase.

PARTIAL_ONLY
    Frozen data contains depletion/refill timing or endpoint/durability detail,
    but not enough to identify post-refill re-depletion of replacement
    liquidity without making an unverifiable assumption.

INSUFFICIENT_FOR_SEQUENCE
    Only aggregate mechanics are available for the needed concept.

OUTPUTS
-------
data/derived/perps/research/prospective_25bp_refill_redepletion_feasibility_v1/
    metadata/latest_summary.json
    metadata/run_config.json

reports/perps_prospective_25bp_refill_redepletion_feasibility/
    frozen_contract.csv
    causal_parquet_inventory.csv
    dataset_sequence_capability.csv
    analysis_invariants.csv
    findings.md

If the result is SUFFICIENT_*, the next script should first construct the
sequence outcome-blind and run a subgroup/common-support feasibility gate
before future P75 labels are loaded.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = None
    pq = None


BUILDER_VERSION = "1.0.0"
SCHEMA_VERSION = 1
SCHEMA_NAME = "prospective_25bp_refill_redepletion_feasibility_v1"

FROZEN_PHASE7M3_SCHEMA = "prospective_25bp_exact_participant_interaction_v1"
FROZEN_PHASE7M3_BUILDER = "1.0.0"
FROZEN_POLICY = "EARLY_25BP_RESET5_RELEASE10_REARM60"
EXPECTED_CANDIDATES = 7_067
EXPECTED_HORIZONS = (1, 3, 5, 10)

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_M3_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / FROZEN_PHASE7M3_SCHEMA
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_prospective_25bp_refill_redepletion_feasibility"

# Explicitly excluded even from schema inspection.  We do not need future labels
# to answer a substrate capability question.
FUTURE_PATH_TOKENS = (
    "outcome",
    "outcomes",
    "future_label",
    "future_labels",
    "decision_outcome",
    "decision_outcomes",
)

# Aggregate mechanics are useful but do not, on their own, establish the order
# of refill and later re-depletion of replacement liquidity.
KNOWN_AGGREGATE_FIELDS = {
    "opposing__gross_depletion_quote",
    "opposing__same_price_refill_after_depletion_quote",
    "opposing__same_price_refill_to_depletion_ratio",
    "opposing__refilled_depleted_level_share",
    "opposing__weighted_median_first_refill_latency_ms",
    "opposing__durable_same_price_refill_quote",
    "opposing__durable_same_price_refill_fraction",
    "opposing__same_price_current_quote",
    "opposing__same_price_current_to_initial_ratio",
    "opposing__near_restoration_to_depletion_proxy",
}

# Strong evidence that a frozen row directly tells us post-refill fate.
EXPLICIT_REDEP_PATTERNS = (
    r"re[_-]?deplet",
    r"redeplet",
    r"post[_-]?refill.*(?:deplet|consum|remove|loss)",
    r"after[_-]?refill.*(?:deplet|consum|remove|loss)",
    r"subsequent.*refill.*(?:deplet|consum|remove|loss)",
    r"refill.*(?:consumed|removed|redepleted|re_depleted)",
    r"replacement.*(?:consumed|removed|depleted|loss)",
    r"refill.*surviv",
    r"replacement.*surviv",
    r"refill.*cycle",
    r"deplet.*refill.*deplet",
    r"second.*deplet",
)

# Event/update-level sequence can be reconstructed if one causal dataset has a
# price locator, temporal order, and quantity state/delta with repeated rows.
PRICE_PATTERNS = (
    r"^price$",
    r"price_level",
    r"level_price",
    r"book_price",
)
TIME_PATTERNS = (
    r"timestamp",
    r"event_time",
    r"update_time",
    r"exchange_time",
    r"local_time",
    r"elapsed_(?:ms|sec|second)",
    r"event_(?:ms|sec|second)",
    r"time_ms$",
    r"time_ns$",
    r"timestamp_ms$",
    r"timestamp_ns$",
    r"_ts_ms$",
    r"_ts_ns$",
    r"^ts$",
)
QUANTITY_PATTERNS = (
    r"^qty$",
    r"quantity",
    r"new_qty",
    r"prev_qty",
    r"old_qty",
    r"current_qty",
    r"delta_qty",
    r"qty_delta",
    r"size_delta",
    r"new_size",
    r"prev_size",
    r"current_size",
)
SIDE_PATTERNS = (
    r"side_role",
    r"book_side",
    r"^side$",
    r"direction_role",
)
CANDIDATE_PATTERNS = (
    r"candidate_id",
    r"event_id",
    r"episode_id",
)

# Partial timing evidence: useful, but not enough by itself to prove later
# consumption of replacement quantity.
PARTIAL_PATTERNS = (
    r"first.*deplet",
    r"first.*refill",
    r"refill.*latency",
    r"durable.*refill",
    r"same_price_current",
    r"retained",
    r"unrecovered",
    r"initial_qty",
    r"current_qty",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def require_stack() -> None:
    if pa is None or pq is None:
        raise SystemExit("Phase 7M.6A requires pandas and pyarrow in the project .venv.")


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_json(obj: Any, path: Path) -> None:
    atomic_write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", path)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        df.to_csv(tmp_name, index=False)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def norm_name(s: str) -> str:
    return str(s).strip().lower()


def matches_any(name: str, patterns: Iterable[str]) -> bool:
    n = norm_name(name)
    return any(re.search(p, n, flags=re.IGNORECASE) is not None for p in patterns)


def is_future_path(path: Path) -> bool:
    s = str(path).lower().replace("\\", "/")
    return any(tok in s for tok in FUTURE_PATH_TOKENS)


def parquet_row_count(path: Path) -> int:
    pf = pq.ParquetFile(path)
    return int(pf.metadata.num_rows)


def parquet_columns(path: Path) -> list[str]:
    return list(pq.ParquetFile(path).schema_arrow.names)


def parquet_column_types(path: Path) -> dict[str, str]:
    schema = pq.ParquetFile(path).schema_arrow
    return {field.name: str(field.type) for field in schema}


def relative_dataset_name(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except Exception:
        rel = path
    parts = rel.parts
    if not parts:
        return str(rel)
    if len(parts) == 1:
        return parts[0]
    # Partitioned datasets such as level_audit/candidate_date=.../part.parquet
    # are grouped under their top-level dataset directory.
    return parts[0]


def load_metadata(root: Path) -> dict[str, Any]:
    path = root / "metadata" / "latest_summary.json"
    if not path.exists():
        raise RuntimeError(f"Missing frozen Phase 7M.3 metadata: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def audit_contract(meta: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    fixed = tuple(int(x) for x in meta.get("fixed_horizons_sec", []))
    checks = [
        ("schema_name", meta.get("schema_name"), FROZEN_PHASE7M3_SCHEMA),
        ("builder_version", str(meta.get("builder_version")), FROZEN_PHASE7M3_BUILDER),
        ("selected_policy", meta.get("selected_policy"), FROZEN_POLICY),
        ("candidate_count", int(meta.get("candidate_count", -1)), EXPECTED_CANDIDATES),
        ("fixed_horizons_sec", fixed, EXPECTED_HORIZONS),
        ("causal_mechanics_future_labels_separate", bool(meta.get("causal_mechanics_future_labels_separate")), True),
        ("status", meta.get("status"), "PASS"),
    ]
    rows: list[dict[str, Any]] = []
    inv: list[dict[str, Any]] = []
    for field, actual, expected in checks:
        passed = actual == expected
        rows.append({"field": field, "actual": str(actual), "expected": str(expected), "pass": passed})
        inv.append({
            "invariant": f"frozen_contract__{field}",
            "status": "PASS" if passed else "FAIL",
            "detail": f"actual={actual!r}; expected={expected!r}",
        })
    return pd.DataFrame(rows), inv


def inventory_causal_parquets(root: Path) -> tuple[pd.DataFrame, dict[str, list[Path]], list[dict[str, Any]]]:
    all_paths = sorted(root.rglob("*.parquet"))
    future_paths = [p for p in all_paths if is_future_path(p)]
    causal_paths = [p for p in all_paths if not is_future_path(p)]

    grouped: dict[str, list[Path]] = {}
    for p in causal_paths:
        grouped.setdefault(relative_dataset_name(p, root), []).append(p)

    rows: list[dict[str, Any]] = []
    for dataset, paths in sorted(grouped.items()):
        sample = paths[0]
        cols = parquet_columns(sample)
        types = parquet_column_types(sample)
        schema_signatures = set()
        total_rows = 0
        for p in paths:
            c = tuple(parquet_columns(p))
            schema_signatures.add(c)
            total_rows += parquet_row_count(p)
        rows.append({
            "dataset": dataset,
            "file_count": len(paths),
            "row_count": total_rows,
            "sample_path": str(sample.relative_to(root)),
            "column_count": len(cols),
            "schema_variants": len(schema_signatures),
            "columns": "|".join(cols),
            "column_types": "|".join(f"{k}:{v}" for k, v in types.items()),
        })

    inv = [
        {
            "invariant": "future_label_parquets_not_opened",
            "status": "PASS",
            "detail": f"excluded_without_opening={len(future_paths)}; causal_parquets_opened_for_schema={len(causal_paths)}",
        },
        {
            "invariant": "causal_parquet_inventory_nonempty",
            "status": "PASS" if causal_paths else "FAIL",
            "detail": f"causal_parquets={len(causal_paths)}",
        },
    ]
    return pd.DataFrame(rows), grouped, inv


def _candidate_key_columns(cols: list[str]) -> list[str]:
    return [c for c in cols if matches_any(c, CANDIDATE_PATTERNS)]


def classify_dataset_columns(dataset: str, total_rows: int, schema_variants: int, cols: list[str], file_count: int) -> dict[str, Any]:
    cols = sorted(set(cols))

    price_cols = [c for c in cols if matches_any(c, PRICE_PATTERNS)]
    time_cols = [c for c in cols if matches_any(c, TIME_PATTERNS)]
    qty_cols = [c for c in cols if matches_any(c, QUANTITY_PATTERNS)]
    side_cols = [c for c in cols if matches_any(c, SIDE_PATTERNS)]
    candidate_cols = _candidate_key_columns(cols)
    explicit_redep_cols = [c for c in cols if matches_any(c, EXPLICIT_REDEP_PATTERNS)]
    partial_cols = [c for c in cols if matches_any(c, PARTIAL_PATTERNS)]
    known_agg = [c for c in cols if c in KNOWN_AGGREGATE_FIELDS]

    # Schema-level event capability requires structural fields plus event-like
    # multiplicity. This prevents a one-row-per-candidate aggregate table from
    # being mistaken for a sequence merely because it contains elapsed_sec.
    event_structure = bool(price_cols and time_cols and qty_cols and candidate_cols)
    many_rows = total_rows > EXPECTED_CANDIDATES * len(EXPECTED_HORIZONS)
    event_sequence_capable = event_structure and many_rows

    # Explicit post-refill fate can be enough even without update rows, but it
    # must live in candidate/per-level data rather than a candidate aggregate.
    explicit_redep_capable = bool(
        explicit_redep_cols
        and candidate_cols
        and (price_cols or "level_audit" in dataset.lower())
    )

    if event_sequence_capable:
        classification = "SUFFICIENT_EVENT_SEQUENCE"
        reason = "Frozen causal rows have candidate + price + temporal order + quantity state/delta and event-like row multiplicity."
    elif explicit_redep_capable:
        classification = "SUFFICIENT_EXPLICIT_REDEP_FIELDS"
        reason = "Frozen causal per-level data explicitly encodes post-refill depletion/consumption."
    elif partial_cols or known_agg:
        classification = "PARTIAL_ONLY"
        reason = "Frozen data has refill/depletion summaries or timing/end-state fields but not enough to identify post-refill re-depletion directly."
    else:
        classification = "INSUFFICIENT_FOR_SEQUENCE"
        reason = "No event-level sequence or explicit post-refill re-depletion fields detected."

    return {
        "dataset": dataset,
        "file_count": file_count,
        "row_count": total_rows,
        "schema_variants": schema_variants,
        "classification": classification,
        "event_structure_fields_present": event_structure,
        "event_like_row_multiplicity": many_rows,
        "price_columns": "|".join(price_cols),
        "time_columns": "|".join(time_cols),
        "quantity_columns": "|".join(qty_cols),
        "side_columns": "|".join(side_cols),
        "candidate_key_columns": "|".join(candidate_cols),
        "explicit_post_refill_redepletion_columns": "|".join(explicit_redep_cols),
        "partial_refill_fate_columns": "|".join(partial_cols),
        "known_aggregate_columns": "|".join(known_agg),
        "reason": reason,
    }


def inspect_dataset_capability(dataset: str, paths: list[Path], root: Path) -> dict[str, Any]:
    # Use the union of all partition schemas so a field present only in some
    # frozen dates cannot be silently missed.
    union_cols: set[str] = set()
    total_rows = 0
    file_schema_variants: set[tuple[str, ...]] = set()
    for p in paths:
        cols = parquet_columns(p)
        union_cols.update(cols)
        file_schema_variants.add(tuple(cols))
        total_rows += parquet_row_count(p)
    return classify_dataset_columns(
        dataset=dataset,
        total_rows=total_rows,
        schema_variants=len(file_schema_variants),
        cols=sorted(union_cols),
        file_count=len(paths),
    )


def overall_classification(capability: pd.DataFrame) -> tuple[str, str, str | None]:
    if capability.empty:
        return "INSUFFICIENT_FOR_SEQUENCE", "No causal parquet datasets were available to audit.", None

    rank = {
        "INSUFFICIENT_FOR_SEQUENCE": 0,
        "PARTIAL_ONLY": 1,
        "SUFFICIENT_EXPLICIT_REDEP_FIELDS": 2,
        "SUFFICIENT_EVENT_SEQUENCE": 3,
    }
    best_idx = max(capability.index, key=lambda i: rank.get(str(capability.loc[i, "classification"]), -1))
    best = capability.loc[best_idx]
    status = str(best["classification"])
    dataset = str(best["dataset"])

    if status == "SUFFICIENT_EVENT_SEQUENCE":
        explanation = (
            f"Dataset '{dataset}' appears sufficient to reconstruct deplete→refill→re-deplete directly from frozen causal rows. "
            "No raw replay should be proposed before a cheap sequence-builder feasibility phase is attempted."
        )
    elif status == "SUFFICIENT_EXPLICIT_REDEP_FIELDS":
        explanation = (
            f"Dataset '{dataset}' explicitly records post-refill depletion/consumption. "
            "A cheap frozen-substrate analysis appears possible without raw replay."
        )
    elif status == "PARTIAL_ONLY":
        explanation = (
            "The frozen substrate contains useful refill/depletion summaries or timing/end-state information, but the audit did not find "
            "a defensible way to identify subsequent consumption of the replacement liquidity itself."
        )
    else:
        explanation = (
            "The frozen substrate does not expose the temporal/per-level information required to identify deplete→refill→re-deplete."
        )
    return status, explanation, dataset


def render_findings(
    meta: dict[str, Any],
    contract: pd.DataFrame,
    inventory: pd.DataFrame,
    capability: pd.DataFrame,
    overall: str,
    explanation: str,
    best_dataset: str | None,
    invariants: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# Phase 7M.6A — Frozen-substrate refill→re-depletion sequence feasibility audit")
    lines.append("")
    lines.append(f"Generated: {utc_now_iso()}")
    lines.append("")
    lines.append("## Scientific question")
    lines.append("")
    lines.append("Can frozen Phase 7M.3 identify the causal-time sequence **opposing depletion → opposing refill → subsequent re-depletion of replacement liquidity** without another raw replay?")
    lines.append("")
    lines.append("## Overall result")
    lines.append("")
    lines.append(f"**{overall}**")
    lines.append("")
    lines.append(explanation)
    lines.append("")
    lines.append("This audit did not load future outcome labels and did not perform matching or an outcome-effect test.")
    lines.append("")
    lines.append("## Frozen contract")
    lines.append("")
    for _, r in contract.iterrows():
        lines.append(f"- {r['field']}: actual `{r['actual']}`; expected `{r['expected']}`; {'PASS' if bool(r['pass']) else 'FAIL'}")
    lines.append("")
    lines.append("## Dataset capability")
    lines.append("")
    for _, r in capability.sort_values(["classification", "dataset"], ascending=[False, True]).iterrows():
        lines.append(f"### {r['dataset']}")
        lines.append("")
        lines.append(f"- Classification: **{r['classification']}**")
        lines.append(f"- Rows: {int(r['row_count']):,} across {int(r['file_count'])} parquet file(s)")
        if r["price_columns"]:
            lines.append(f"- Price locator(s): `{r['price_columns']}`")
        if r["time_columns"]:
            lines.append(f"- Temporal field(s): `{r['time_columns']}`")
        if r["quantity_columns"]:
            lines.append(f"- Quantity field(s): `{r['quantity_columns']}`")
        if r["explicit_post_refill_redepletion_columns"]:
            lines.append(f"- Explicit post-refill fate field(s): `{r['explicit_post_refill_redepletion_columns']}`")
        if r["partial_refill_fate_columns"]:
            lines.append(f"- Partial refill-fate field(s): `{r['partial_refill_fate_columns']}`")
        lines.append(f"- Interpretation: {r['reason']}")
        lines.append("")
    lines.append("## Decision rule for the next phase")
    lines.append("")
    if overall.startswith("SUFFICIENT_"):
        lines.append("Do **not** replay raw data yet. Build the refill→re-depletion mechanics from the identified frozen dataset, outcome-blind, then run the mandatory subgroup/common-support feasibility gate before loading P75 labels.")
    else:
        lines.append("Do **not** pretend aggregate refill/durability is equivalent to re-depletion of the replacement quantity. First decide whether the missing sequence information is scientifically important enough to justify a targeted additional replay. If replay is considered, run the mandatory pre-compute feasibility gate before spending compute.")
    lines.append("")
    lines.append("## Invariants")
    lines.append("")
    for _, r in invariants.iterrows():
        lines.append(f"- {r['invariant']}: **{r['status']}** — {r['detail']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def build_mock_substrate(root: Path, mode: str) -> None:
    """Create tiny schema-only mock 7M.3 substrates for --self-test."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    meta = {
        "builder_version": FROZEN_PHASE7M3_BUILDER,
        "schema_name": FROZEN_PHASE7M3_SCHEMA,
        "selected_policy": FROZEN_POLICY,
        "candidate_count": EXPECTED_CANDIDATES,
        "fixed_horizons_sec": list(EXPECTED_HORIZONS),
        "causal_mechanics_future_labels_separate": True,
        "status": "PASS",
    }
    atomic_json(meta, root / "metadata" / "latest_summary.json")

    mechanics = pd.DataFrame({
        "candidate_id": ["c1", "c2"],
        "observation_horizon_sec": [3, 3],
        "opposing__same_price_refill_after_depletion_quote": [1.0, 2.0],
        "opposing__durable_same_price_refill_fraction": [0.5, 0.4],
    })
    mechanics.to_parquet(root / "candidate_mechanics.parquet", index=False)

    # A future file exists to prove the audit excludes it without opening.
    pd.DataFrame({"candidate_id": ["c1"], "h900_origin_reach_75": [True]}).to_parquet(
        root / "candidate_decision_outcomes.parquet", index=False
    )

    la = root / "level_audit" / "candidate_date=2026-06-01"
    la.mkdir(parents=True, exist_ok=True)
    if mode == "sufficient":
        n = EXPECTED_CANDIDATES * len(EXPECTED_HORIZONS) + 1
        # Keep memory small while satisfying multiplicity: Arrow can write a
        # repeated scalar table efficiently enough for the self-test.
        df = pd.DataFrame({
            "candidate_id": ["c1"] * n,
            "side_role": ["OPPOSING"] * n,
            "price": [100.0] * n,
            "event_time_ms": list(range(n)),
            "prev_qty": [1.0] * n,
            "new_qty": [0.5] * n,
        })
    elif mode == "explicit":
        df = pd.DataFrame({
            "candidate_id": ["c1"],
            "side_role": ["OPPOSING"],
            "price": [100.0],
            "initial_qty": [1.0],
            "refill_redepletion_quote": [0.4],
        })
    else:
        df = pd.DataFrame({
            "candidate_id": ["c1"],
            "side_role": ["OPPOSING"],
            "price": [100.0],
            "initial_qty": [1.0],
            "first_refill_latency_ms": [200.0],
        })
    df.to_parquet(la / "part.parquet", index=False)


def run_audit(m3_root: Path, output_root: Path, report_root: Path) -> dict[str, Any]:
    require_stack()
    if not m3_root.exists():
        raise RuntimeError(f"Frozen Phase 7M.3 root does not exist: {m3_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    print("[1/5] Verifying frozen Phase 7M.3 contract...", flush=True)
    meta = load_metadata(m3_root)
    contract, inv_contract = audit_contract(meta)
    atomic_csv(contract, report_root / "frozen_contract.csv")

    print("[2/5] Inventorying causal parquet schemas only (future outcomes excluded)...", flush=True)
    inventory, grouped, inv_inventory = inventory_causal_parquets(m3_root)
    atomic_csv(inventory, report_root / "causal_parquet_inventory.csv")

    print("[3/5] Auditing temporal/per-level sequence capability...", flush=True)
    cap_rows = [inspect_dataset_capability(ds, paths, m3_root) for ds, paths in sorted(grouped.items())]
    capability = pd.DataFrame(cap_rows)
    atomic_csv(capability, report_root / "dataset_sequence_capability.csv")

    overall, explanation, best_dataset = overall_classification(capability)

    invariants_rows = inv_contract + inv_inventory
    contract_pass = all(r["status"] == "PASS" for r in inv_contract)
    invariants_rows.append({
        "invariant": "frozen_contract_all_pass",
        "status": "PASS" if contract_pass else "FAIL",
        "detail": f"contract_checks={len(inv_contract)}",
    })
    invariants_rows.append({
        "invariant": "audit_is_outcome_blind",
        "status": "PASS",
        "detail": "Future-label/outcome parquet paths were excluded before ParquetFile/schema access.",
    })
    invariants = pd.DataFrame(invariants_rows)
    atomic_csv(invariants, report_root / "analysis_invariants.csv")

    if not contract_pass:
        overall = "INVALID_FROZEN_CONTRACT"
        explanation = "The expected frozen Phase 7M.3 contract did not verify; do not interpret sequence capability until the substrate identity is resolved."

    print("[4/5] Writing outcome-blind feasibility conclusion...", flush=True)
    summary = {
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "schema_name": SCHEMA_NAME,
        "generated_at": utc_now_iso(),
        "phase7m3_root": str(m3_root),
        "frozen_phase7m3_schema": FROZEN_PHASE7M3_SCHEMA,
        "frozen_phase7m3_builder": FROZEN_PHASE7M3_BUILDER,
        "frozen_policy": FROZEN_POLICY,
        "status": "PASS" if contract_pass else "FAIL",
        "sequence_feasibility": overall,
        "best_capability_dataset": best_dataset,
        "future_outcomes_loaded": False,
        "raw_replay_performed": False,
        "candidate_regeneration_performed": False,
        "interpretation": explanation,
    }
    run_config = {
        "scientific_question": "Can frozen 7M.3 identify opposing depletion→refill→subsequent re-depletion of replacement liquidity?",
        "minimum_sufficient_event_structure": ["candidate key", "price level", "temporal order", "quantity state/delta", "event-like row multiplicity"],
        "alternative_sufficient_structure": "explicit per-level post-refill re-depletion/consumption field",
        "future_path_tokens_excluded": list(FUTURE_PATH_TOKENS),
        "known_aggregate_fields_not_treated_as_direct_sequence_proof": sorted(KNOWN_AGGREGATE_FIELDS),
        "explicit_redepletion_patterns": list(EXPLICIT_REDEP_PATTERNS),
        "raw_replay": False,
        "future_labels": False,
    }
    atomic_json(summary, output_root / "metadata" / "latest_summary.json")
    atomic_json(run_config, output_root / "metadata" / "run_config.json")

    findings = render_findings(meta, contract, inventory, capability, overall, explanation, best_dataset, invariants)
    atomic_write_text(findings, report_root / "findings.md")

    print("[5/5] Complete.", flush=True)
    print(f"Sequence feasibility: {overall}", flush=True)
    if best_dataset:
        print(f"Best frozen dataset: {best_dataset}", flush=True)
    print(explanation, flush=True)
    return summary


def self_test() -> None:
    # Pure schema-classification tests.  They intentionally do not require
    # pyarrow, so the script can validate its scientific decision logic even
    # in an environment without the project parquet stack.
    n_event = EXPECTED_CANDIDATES * len(EXPECTED_HORIZONS) + 1
    cases = [
        (
            "sufficient_event",
            classify_dataset_columns(
                "level_audit", n_event, 1,
                ["candidate_id", "side_role", "price", "event_time_ms", "prev_qty", "new_qty"], 62,
            )["classification"],
            "SUFFICIENT_EVENT_SEQUENCE",
        ),
        (
            "sufficient_explicit",
            classify_dataset_columns(
                "level_audit", 1000, 1,
                ["candidate_id", "side_role", "price", "initial_qty", "refill_redepletion_quote"], 62,
            )["classification"],
            "SUFFICIENT_EXPLICIT_REDEP_FIELDS",
        ),
        (
            "partial",
            classify_dataset_columns(
                "level_audit", 1000, 1,
                ["candidate_id", "side_role", "price", "initial_qty", "first_refill_latency_ms"], 62,
            )["classification"],
            "PARTIAL_ONLY",
        ),
        (
            "aggregate_not_false_explicit",
            classify_dataset_columns(
                "candidate_mechanics.parquet", EXPECTED_CANDIDATES * len(EXPECTED_HORIZONS), 1,
                ["candidate_id", "observation_horizon_sec", "opposing__same_price_refill_to_depletion_ratio"], 1,
            )["classification"],
            "PARTIAL_ONLY",
        ),
    ]
    for name, actual, expected in cases:
        if actual != expected:
            raise AssertionError(f"self-test {name}: expected {expected}, got {actual}")
    print("SELF-TEST PASS", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7M.6A frozen-substrate refill→re-depletion feasibility audit")
    p.add_argument("--m3-root", type=Path, default=DEFAULT_M3_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("Phase 7M.6A — Frozen-substrate refill→re-depletion sequence feasibility audit", flush=True)
    if args.self_test:
        self_test()
        return
    run_audit(args.m3_root, args.output_root, args.report_root)


if __name__ == "__main__":
    main()
