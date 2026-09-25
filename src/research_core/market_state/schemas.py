"""Schemas and semantic roles for the R0.5 causal market-state store."""

from __future__ import annotations

from dataclasses import dataclass

FEATURE_ELIGIBLE = "FEATURE_ELIGIBLE"
GOVERNANCE_ONLY = "GOVERNANCE_ONLY"
FUTURE_FORBIDDEN = "FUTURE_FORBIDDEN"
IDENTITY_ONLY = "IDENTITY_ONLY"
QUALITY_ONLY = "QUALITY_ONLY"

TRADE_FINE_LOOKBACK_SEC = 60
TRADE_FINE_BUCKET_SEC = 1
TRADE_COARSE_LOOKBACK_SEC = 30 * 60
TRADE_COARSE_BUCKET_SEC = 60

# These are storage resolutions, not scientific feature windows.  Future R1+
# hypotheses must still pre-register the exact window/transform they use.
TRADE_STORAGE_GRID = {
    "fine": {
        "lookback_sec": TRADE_FINE_LOOKBACK_SEC,
        "bucket_sec": TRADE_FINE_BUCKET_SEC,
    },
    "coarse": {
        "lookback_sec": TRADE_COARSE_LOOKBACK_SEC,
        "bucket_sec": TRADE_COARSE_BUCKET_SEC,
    },
}


@dataclass(frozen=True)
class FieldSpec:
    name: str
    role: str
    information_clock: str
    description: str


CANDIDATE_CORE_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("candidate_id", IDENTITY_ONLY, "R0.3", "Frozen candidate identifier."),
    FieldSpec("candidate_date", IDENTITY_ONLY, "R0.3", "Frozen candidate UTC date."),
    FieldSpec("direction", FEATURE_ELIGIBLE, "candidate_policy", "Frozen candidate direction (+1/-1)."),
    FieldSpec("candidate_policy_availability_ts_ns", QUALITY_ONLY, "candidate_policy", "One-second policy interval-close availability."),
    FieldSpec("anchor_ts_ns", QUALITY_ONLY, "receive_time", "Recovered exact book anchor receive timestamp."),
    FieldSpec("information_available_ts_ns", QUALITY_ONLY, "R0.2 clock contract", "Latest allowed causal cutoff for R0.5 state."),
    FieldSpec("decision_ts_ns", QUALITY_ONLY, "R0.2 clock contract", "Governed scientific decision timestamp."),
    FieldSpec("clock_available", QUALITY_ONLY, "R0.3", "Whether a governed exact clock exists."),
    FieldSpec("clock_unavailable_reason", QUALITY_ONLY, "R0.3", "Explicit reason for missing governed clock."),
)

TRADE_STATE_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("latest_trade_ts_ns", QUALITY_ONLY, "receive_time", "Latest canonical trade at or before cutoff."),
    FieldSpec("latest_trade_price", FEATURE_ELIGIBLE, "receive_time", "Price of latest deterministically ordered trade at or before cutoff."),
    FieldSpec("latest_trade_age_ns", QUALITY_ONLY, "receive_time", "Cutoff minus latest trade receive timestamp."),
    FieldSpec("latest_trade_tie_ambiguous", QUALITY_ONLY, "receive_time", "True when equal-timestamp ordering lacked a canonical tie field."),
    FieldSpec("trade_source_status", QUALITY_ONLY, "partition", "Observed source-partition availability state."),
)

BUCKET_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("trade_count", FEATURE_ELIGIBLE, "receive_time", "Observed canonical trade rows in half-open bucket."),
    FieldSpec("quantity_sum", FEATURE_ELIGIBLE, "receive_time", "Observed base quantity in half-open bucket."),
    FieldSpec("notional_sum", FEATURE_ELIGIBLE, "receive_time", "Observed price*quantity in half-open bucket."),
    FieldSpec("buyer_maker_true_quantity", FEATURE_ELIGIBLE, "receive_time", "Quantity where raw buyer_is_maker is true; no directional reinterpretation."),
    FieldSpec("buyer_maker_false_quantity", FEATURE_ELIGIBLE, "receive_time", "Quantity where raw buyer_is_maker is false; no directional reinterpretation."),
    FieldSpec("buyer_maker_unknown_quantity", QUALITY_ONLY, "receive_time", "Quantity whose raw buyer_is_maker value is missing/unparseable."),
    FieldSpec("first_trade_ts_ns", QUALITY_ONLY, "receive_time", "Earliest observed trade receive timestamp in bucket."),
    FieldSpec("last_trade_ts_ns", QUALITY_ONLY, "receive_time", "Latest observed trade receive timestamp in bucket."),
)

# R0.4 identifiers are useful for resampling/splitting but are explicitly not
# predictors.  Keeping the list here lets tests and downstream loaders reject
# accidental feature use.
GOVERNANCE_FIELD_PREFIXES: tuple[str, ...] = (
    "overlap_component_",
    "policy_temporal_episode_",
    "directional_episode_",
    "origin_lineage_episode_",
    "conservative_inference_cluster",
)

FUTURE_FORBIDDEN_TOKENS: tuple[str, ...] = (
    "outcome",
    "future",
    "mfe",
    "mae",
    "reach_",
    "reaches_",
    "seconds_to_",
    "p35",
    "p50",
    "p75",
    "p100",
    "first_passage",
)
