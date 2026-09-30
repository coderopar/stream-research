"""Deterministic analysis primitives for the R0.8 D0 full dry run.

The dry run is intentionally non-promotable.  Its fixed test state exists only
so the complete governed research workflow can be exercised end to end without
claiming a new scientific or trading result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from research_core.execution import CostAssumptions, total_cost_bps

DRY_RUN_FEATURE_WINDOW_SEC = 10
DRY_RUN_HORIZON_SEC = 300
DRY_RUN_FEATURE_THRESHOLD = 0.0
DRY_RUN_COSTS = CostAssumptions(0.0, 0.0, 0.0, "R08_MECHANICAL_TEST_ONLY")
DRY_RUN_BOOTSTRAP_SEED = 20260930
DRY_RUN_BOOTSTRAP_DRAWS = 2000


@dataclass(frozen=True)
class DryRunContrast:
    status: str
    independent_units: int
    estimate_bps: float | None
    ci_low_bps: float | None
    ci_high_bps: float | None


def build_fixed_test_state(fine_buckets: pd.DataFrame) -> pd.DataFrame:
    """Build one fixed causal state from the trailing ten 1-second buckets.

    State definition (fixed before reading outcomes): within [-10s, 0s), compare
    the raw quantities where ``buyer_is_maker`` is false vs true.  No economic
    or directional interpretation is attached to the state.  The state is true
    when the normalized raw balance is > 0.  This is a dry-run plumbing state,
    not a hypothesis proposed for R1 or D1.
    """
    required = {
        "candidate_id", "bucket_offset_sec", "source_status",
        "buyer_maker_true_quantity", "buyer_maker_false_quantity",
    }
    missing = sorted(required - set(fine_buckets.columns))
    if missing:
        raise ValueError(f"fine buckets missing required fields: {missing}")
    w = fine_buckets[
        (fine_buckets["bucket_offset_sec"].astype(int) >= -DRY_RUN_FEATURE_WINDOW_SEC)
        & (fine_buckets["bucket_offset_sec"].astype(int) < 0)
    ].copy()
    rows: list[dict[str, Any]] = []
    for cid, g in w.groupby("candidate_id", sort=False):
        available = bool(len(g) == DRY_RUN_FEATURE_WINDOW_SEC and (g["source_status"] == "AVAILABLE").all())
        if available:
            t = float(g["buyer_maker_true_quantity"].astype(float).sum())
            f = float(g["buyer_maker_false_quantity"].astype(float).sum())
            denom = t + f
            score = 0.0 if denom <= 0 else (f - t) / denom
            state = bool(score > DRY_RUN_FEATURE_THRESHOLD)
        else:
            t = f = score = np.nan
            state = False
        rows.append({
            "candidate_id": str(cid),
            "feature_available": available,
            "raw_buyer_maker_true_qty_10s": t,
            "raw_buyer_maker_false_qty_10s": f,
            "dry_run_state_score": score,
            "dry_run_state_true": state,
        })
    return pd.DataFrame(rows).sort_values("candidate_id", kind="mergesort").reset_index(drop=True)


def build_causal_analysis_population(
    candidate_core: pd.DataFrame,
    trade_coverage: pd.DataFrame,
    execution_state: pd.DataFrame,
    feature_state: pd.DataFrame,
) -> pd.DataFrame:
    """Create eligibility using causal/quality/execution fields only.

    No future outcome status, future completeness, return, MFE or MAE is an
    input to ``trading_eligible``.  Outcome observability is attached later.
    """
    for name, frame in (
        ("candidate_core", candidate_core),
        ("trade_coverage", trade_coverage),
        ("execution_state", execution_state),
        ("feature_state", feature_state),
    ):
        if "candidate_id" not in frame.columns or frame["candidate_id"].astype(str).duplicated().any():
            raise ValueError(f"{name} must contain unique candidate_id")
    core_req = {"candidate_id", "direction", "information_available_ts_ns", "clock_available"}
    cov_req = {"candidate_id", "fine_history_status"}
    exe_req = {"candidate_id", "book_status", "buy_fill_complete", "sell_fill_complete", "buy_raw_crossing_cost_bps", "sell_raw_crossing_cost_bps"}
    if not core_req <= set(candidate_core.columns):
        raise ValueError("candidate_core missing required causal fields")
    if not cov_req <= set(trade_coverage.columns):
        raise ValueError("trade_coverage missing fine_history_status")
    if not exe_req <= set(execution_state.columns):
        raise ValueError("execution_state missing required execution fields")

    frame = candidate_core[list(core_req)].copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame = frame.merge(trade_coverage[["candidate_id", "fine_history_status"]], on="candidate_id", how="left", validate="one_to_one")
    frame = frame.merge(execution_state[list(exe_req)], on="candidate_id", how="left", validate="one_to_one")
    frame = frame.merge(feature_state, on="candidate_id", how="left", validate="one_to_one")
    direction = frame["direction"].astype(int)
    direction_fill = np.where(direction > 0, frame["buy_fill_complete"].fillna(False), frame["sell_fill_complete"].fillna(False))
    frame["trading_eligible"] = (
        frame["clock_available"].fillna(False).astype(bool)
        & frame["feature_available"].fillna(False).astype(bool)
        & frame["fine_history_status"].eq("AVAILABLE")
        & frame["book_status"].eq("AVAILABLE")
        & pd.Series(direction_fill, index=frame.index).astype(bool)
    )
    return frame.sort_values(["information_available_ts_ns", "candidate_id"], kind="mergesort").reset_index(drop=True)


def attach_future_observability(
    causal_population: pd.DataFrame,
    outcomes_5m: pd.DataFrame,
    dependence: pd.DataFrame,
) -> pd.DataFrame:
    """Attach D0 outcomes after eligibility is frozen.

    ``outcome_observable`` and outcome values do not feed back into
    ``trading_eligible``.  This keeps future observability separate from causal
    trading eligibility as required by Protocol v1.2.
    """
    req = {
        "candidate_id", "outcome_horizon_sec", "outcome_status", "outcome_observable",
        "directional_return_to_last_trade_bps", "mae_from_decision_bps",
    }
    if not req <= set(outcomes_5m.columns):
        raise ValueError("outcomes_5m missing required fields")
    if len(outcomes_5m) and not (outcomes_5m["outcome_horizon_sec"].astype(int) == DRY_RUN_HORIZON_SEC).all():
        raise ValueError("dry run requires only the fixed 5-minute outcome horizon")
    dep_req = {"candidate_id", "conservative_inference_cluster_id"}
    if not dep_req <= set(dependence.columns):
        raise ValueError("dependence table missing conservative inference cluster")
    before = causal_population["trading_eligible"].copy()
    out_cols = [
        "candidate_id", "outcome_status", "outcome_observable",
        "directional_return_to_last_trade_bps", "mae_from_decision_bps",
    ]
    merged = causal_population.merge(outcomes_5m[out_cols], on="candidate_id", how="left", validate="one_to_one")
    merged = merged.merge(dependence[["candidate_id", "conservative_inference_cluster_id"]], on="candidate_id", how="left", validate="one_to_one")
    if not before.reset_index(drop=True).equals(merged["trading_eligible"].reset_index(drop=True)):
        raise AssertionError("future-outcome attachment changed causal eligibility")
    merged["analysis_observed"] = merged["trading_eligible"].astype(bool) & merged["outcome_status"].eq("AVAILABLE") & merged["outcome_observable"].fillna(False).astype(bool)
    return merged


def apply_flat_only_position_state_machine(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply a deterministic one-position-at-a-time 5-minute dry-run machine.

    Only state-true, causally eligible candidates can open a position.  While a
    position is open all subsequent candidates are ignored.  This deliberately
    exercises capital-path handling for overlapping/repeated signals.
    """
    ordered = frame.sort_values(["information_available_ts_ns", "candidate_id"], kind="mergesort").copy()
    active_until: int | None = None
    actions: list[str] = []
    entries: list[bool] = []
    exits: list[int | None] = []
    for r in ordered.to_dict("records"):
        ts = int(r["information_available_ts_ns"]) if pd.notna(r["information_available_ts_ns"]) else None
        if not bool(r.get("trading_eligible", False)) or ts is None:
            action, entered, exit_ts = "INELIGIBLE", False, None
        elif not bool(r.get("dry_run_state_true", False)):
            action, entered, exit_ts = "NO_SIGNAL", False, None
        elif active_until is not None and ts < active_until:
            action, entered, exit_ts = "IGNORED_WHILE_POSITION_OPEN", False, None
        else:
            active_until = ts + DRY_RUN_HORIZON_SEC * 1_000_000_000
            action, entered, exit_ts = "ENTER", True, active_until
        actions.append(action)
        entries.append(entered)
        exits.append(exit_ts)
    ordered["position_action"] = actions
    ordered["position_entered"] = entries
    ordered["planned_exit_ts_ns"] = exits
    return ordered


def add_mechanical_net_proxy(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach a non-promotable mechanical net-return proxy.

    R0.3 directional return is reduced by the R0.6 arrival-time entry crossing
    cost.  No exit-depth model is available here, so this quantity is explicitly
    a dry-run plumbing proxy and MUST NOT be interpreted as realistic strategy
    P&L or used for promotion.
    """
    out = frame.copy()
    costs: list[float | None] = []
    net: list[float | None] = []
    for r in out.to_dict("records"):
        crossing = r.get("buy_raw_crossing_cost_bps") if int(r.get("direction", 0)) > 0 else r.get("sell_raw_crossing_cost_bps")
        c = total_cost_bps(None if pd.isna(crossing) else float(crossing), DRY_RUN_COSTS)
        y = r.get("directional_return_to_last_trade_bps")
        yv = None if y is None or pd.isna(y) else float(y)
        costs.append(c)
        net.append(None if c is None or yv is None else yv - c)
    out["dry_run_entry_cost_bps"] = costs
    out["dry_run_mechanical_net_proxy_bps"] = net
    return out


def dependence_aware_contrast(frame: pd.DataFrame) -> DryRunContrast:
    """Equal-weight within-cluster contrast with a deterministic cluster bootstrap."""
    use = frame[frame["analysis_observed"].fillna(False).astype(bool)].copy()
    use = use[pd.notna(use["dry_run_mechanical_net_proxy_bps"]) & pd.notna(use["conservative_inference_cluster_id"])]
    contrasts: list[float] = []
    for _, g in use.groupby("conservative_inference_cluster_id", sort=True):
        a = g.loc[g["dry_run_state_true"].astype(bool), "dry_run_mechanical_net_proxy_bps"].astype(float)
        b = g.loc[~g["dry_run_state_true"].astype(bool), "dry_run_mechanical_net_proxy_bps"].astype(float)
        if len(a) and len(b):
            contrasts.append(float(a.mean() - b.mean()))
    if not contrasts:
        return DryRunContrast("INDETERMINATE_NO_WITHIN_CLUSTER_SUPPORT", 0, None, None, None)
    arr = np.asarray(contrasts, dtype=float)
    estimate = float(arr.mean())
    if len(arr) == 1:
        return DryRunContrast("INDETERMINATE_SINGLE_UNIT", 1, estimate, None, None)
    rng = np.random.default_rng(DRY_RUN_BOOTSTRAP_SEED)
    draws = rng.choice(arr, size=(DRY_RUN_BOOTSTRAP_DRAWS, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return DryRunContrast("DESCRIPTIVE_D0_DRY_RUN_ONLY", len(arr), estimate, float(lo), float(hi))
