#!/usr/bin/env python3
"""Build Phase-6B causal liquidity-event primitives and executable markouts.

Purpose
-------
Phase 6B moves from the audited Phase-6A execution foundation into causal
participant/liquidity-event research.  It does not train a predictive model and
it does not make an alpha claim.  Instead it builds four separately auditable
primitive event families:

* DEPTH_WITHDRAWAL      — near-touch visible depth disappears unusually fast.
* AGGRESSIVE_FLOW       — unusually one-sided aggressive trade flow.
* FORCE_ORDER           — observed Binance forceOrder direction/presence.
* MARK_LAST_DISLOCATION — last-trade price departs unusually far from mark.

Open interest is retained as a regime/context variable only.  It is not used as
an event trigger because its sampling cadence is too coarse for sub-second event
mechanics.

Causal timing contract
----------------------
All primitive rows are keyed by ``receive_second``.  A primitive measured in
second ``t`` is available only after that second closes.  Therefore:

    threshold onset second  = first weaker mechanical/observational threshold
    detection second        = first stricter causal detector threshold
    tradeable second        = detection second + 1

The script records the move already realized between onset and detection so a
late detector cannot masquerade as an early signal.

Matched controls
----------------
Detected directional events receive deterministic matched non-event controls.
Controls are primitive-specific: the same primitive family must be neutral at
the anchor and a trailing-only cooldown must have elapsed since the latest
same-family strict detection.  Other primitive activity remains observable
context rather than a global disqualifier.  A configurable embargo is applied
only around the parent event being matched, and bounded candidate reuse avoids
artificially exhausting the 5-second candidate grid.  Matching uses only
contemporaneous causal state: UTC hour/date proximity, 60-second realized
volatility, 60-second quote volume, spread, and visible 5-bps depth.  Future
outcomes are never used in the distance function.

Markouts
--------
For each detected event/control, fixed-horizon markouts are calculated from the
Phase-6A reconstructed book.  The script preserves three different economics:

* ``gross_mid_markout_bps`` — directional mid-to-mid movement.
* ``executable_entry_to_future_mid_bps`` — exact fixed-notional entry VWAP to
  future mid (entry crossing included; no exit crossing).
* ``estimated_roundtrip_net_bps`` — mid markout less Phase-6A fixed-notional
  entry and exit crossing costs and two configurable taker fees.  This is an
  execution-cost estimate, not an exact same-base-quantity round trip.

forceOrder size contract
------------------------
Observed forceOrder size remains selected/censored.  ``force_selected_notional``
is descriptive only and is never interpreted as total liquidation volume or a
calibrated liquidation-intensity measure.

Inputs
------
    data/derived/perps/execution_foundation_v1
    data/derived/perps/flow_features_1s_v1
    data/derived/perps/core_1s_v1

Outputs
-------
    data/derived/perps/research/liquidity_event_primitives_v1/
      primitives/symbol=BTCUSDT/date=YYYY-MM-DD/part-YYYY-MM-DD.parquet
      events/symbol=BTCUSDT/date=YYYY-MM-DD/part-YYYY-MM-DD.parquet
      controls/symbol=BTCUSDT/date=YYYY-MM-DD/part-YYYY-MM-DD.parquet
      markouts/symbol=BTCUSDT/date=YYYY-MM-DD/part-YYYY-MM-DD.parquet
      metadata/build_state.sqlite
      metadata/aggregation_state.sqlite
      metadata/latest_summary.json
      metadata/run_config.json

Reports
-------
    reports/perps_liquidity_event_primitives/
      liquidity_primitive_event_counts.csv
      liquidity_primitive_detection_lag.csv
      liquidity_primitive_control_coverage.csv
      liquidity_primitive_control_balance.csv
      liquidity_primitive_markout_summary.csv
      liquidity_primitive_data_quality.csv
      liquidity_primitive_performance.csv
      liquidity_primitive_invariants.csv
      liquidity_primitive_findings.md

Typical first smoke test
------------------------
    python scripts/perps/research/06b_build_causal_liquidity_event_primitives.py \
      --start-date 2026-06-12 --end-date 2026-06-12
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]


BUILDER_VERSION = "1.0.3"
# Primitive-row semantics are unchanged from v1.0.2; keep its build signature so
# already-completed 62 daily primitive partitions are reused verbatim.
PRIMITIVE_PARTITION_BUILD_VERSION = "1.0.2"
OUTPUT_SCHEMA_VERSION = 3
OUTPUT_SCHEMA_NAME = "perps_liquidity_event_primitives_v1"
DEFAULT_SYMBOL = "BTCUSDT"
SECONDS_PER_DAY = 86_400

DEFAULT_BASELINE_SECONDS = 900
DEFAULT_BASELINE_MIN_SECONDS = 300
DEFAULT_EVENT_BRIDGE_SECONDS = 3
DEFAULT_MAX_DETECTION_LAG_SECONDS = 10
DEFAULT_CONTROL_EMBARGO_SECONDS = 60
# Controls are primitive-specific.  A candidate must be neutral for the same
# primitive at the anchor and must not follow a same-primitive *detection* too
# closely.  The cooldown is trailing-only (causal); unlike v1.0.0 there is no
# future/global calmness requirement.
DEFAULT_CONTROL_TRAILING_COOLDOWN_SECONDS = 10
DEFAULT_CONTROL_CADENCE_SECONDS = 5
DEFAULT_CONTROLS_PER_EVENT = 3
# A control anchor may be reused for multiple parent events, but only up to a
# bounded deterministic cap.  v1.0.0's global no-reuse rule could exhaust the
# 5-second candidate grid before every event received controls.
DEFAULT_CONTROL_MAX_REUSE = 5
DEFAULT_MARKOUT_HORIZONS = (1, 2, 5, 10, 30, 60, 120, 300)
DEFAULT_NOTIONALS = (1_000.0, 5_000.0, 10_000.0, 25_000.0)

# Frozen v1.0.0 detector thresholds.  They are deliberately explicit and
# parameterized on the CLI.  They are event-definition thresholds, not claims
# that these values are economically optimal.
DEFAULT_DEPTH_ONSET_FRAC = 0.10
DEFAULT_DEPTH_DETECT_FRAC = 0.20
DEFAULT_DEPTH_DETECT_Z = 3.0
DEFAULT_FLOW_ONSET_IMBALANCE = 0.60
DEFAULT_FLOW_DETECT_IMBALANCE = 0.75
DEFAULT_FLOW_DETECT_Z = 2.5
DEFAULT_FLOW_VOLUME_Z = 0.0
DEFAULT_DISLOCATION_ONSET_BPS = 1.0
DEFAULT_DISLOCATION_DETECT_BPS = 2.0
DEFAULT_DISLOCATION_DETECT_Z = 2.5

MATCH_COLUMNS = (
    "realized_vol_60s_bps",
    "log_quote_volume_60s",
    "spread_bps",
    "log_total_depth_5bps",
)


def require_arrow() -> None:
    if pa is None or pq is None:
        raise SystemExit(
            "Phase 6B requires numpy, pandas and pyarrow. Activate the project .venv and run:\n"
            "  python -m pip install numpy pandas pyarrow"
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def parse_int_tuple(value: str, name: str) -> tuple[int, ...]:
    try:
        out = tuple(sorted({int(x.strip()) for x in value.split(",") if x.strip()}))
    except ValueError as exc:
        raise SystemExit(f"{name} must be a comma-separated integer list") from exc
    if not out or any(x <= 0 for x in out):
        raise SystemExit(f"{name} must contain positive integers")
    return out


def parse_float_tuple(value: str, name: str) -> tuple[float, ...]:
    try:
        out = tuple(sorted({float(x.strip()) for x in value.split(",") if x.strip()}))
    except ValueError as exc:
        raise SystemExit(f"{name} must be a comma-separated numeric list") from exc
    if not out or any((not math.isfinite(x) or x <= 0) for x in out):
        raise SystemExit(f"{name} must contain positive finite numbers")
    return out


def find_repo_root(script_path: Path) -> Path:
    p = script_path.resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "scripts").exists():
            return parent
    return Path.cwd()


REPO_ROOT = find_repo_root(Path(__file__))
DEFAULT_EXECUTION_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "execution_foundation_v1"
DEFAULT_FLOW_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "flow_features_1s_v1"
DEFAULT_CORE_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "core_1s_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "liquidity_event_primitives_v1"
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_liquidity_event_primitives"


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_id(*parts: Any, length: int = 20) -> str:
    return sha256_text("|".join(str(p) for p in parts))[:length]


def short_tmp(path: Path, prefix: str = "tmp") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.parent / f".{prefix}-{uuid.uuid4().hex[:8]}.tmp"


def atomic_write_text(path: Path, text: str) -> None:
    tmp = short_tmp(path, "tx")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str))


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    tmp = short_tmp(path, "csv")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_parquet(path: Path, frame: pd.DataFrame, metadata: Mapping[str, Any]) -> None:
    require_arrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = short_tmp(path, "pq")
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        schema_meta = dict(table.schema.metadata or {})
        for key, value in metadata.items():
            schema_meta[str(key).encode()] = str(value).encode()
        table = table.replace_schema_metadata(schema_meta)
        pq.write_table(table, tmp, compression="zstd", compression_level=6, use_dictionary=True)
        check = pq.ParquetFile(tmp)
        try:
            if int(check.metadata.num_rows) != len(frame):
                raise RuntimeError(f"Parquet row mismatch for {path}: {check.metadata.num_rows} != {len(frame)}")
        finally:
            check.close()
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def file_signature(paths: Sequence[Path]) -> str:
    rows = []
    for path in sorted(set(paths)):
        st = path.stat()
        rows.append((str(path.resolve()), int(st.st_size), int(st.st_mtime_ns)))
    return sha256_text(compact_json(rows))


def safe_float(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def bps_return(final: float, initial: float, direction: int = 1) -> float:
    if not (math.isfinite(final) and math.isfinite(initial) and initial > 0 and final > 0):
        return math.nan
    raw = (final / initial - 1.0) * 10_000.0
    return raw * int(direction)


@dataclass(frozen=True)
class Config:
    baseline_seconds: int
    baseline_min_seconds: int
    event_bridge_seconds: int
    max_detection_lag_seconds: int
    control_embargo_seconds: int
    control_trailing_cooldown_seconds: int
    control_cadence_seconds: int
    controls_per_event: int
    control_max_reuse: int
    markout_horizons: tuple[int, ...]
    notionals: tuple[float, ...]
    taker_fee_bps: float
    depth_onset_frac: float
    depth_detect_frac: float
    depth_detect_z: float
    flow_onset_imbalance: float
    flow_detect_imbalance: float
    flow_detect_z: float
    flow_volume_z: float
    dislocation_onset_bps: float
    dislocation_detect_bps: float
    dislocation_detect_z: float

    def signature(self) -> str:
        return sha256_text(compact_json(asdict(self)))


class AggregationState:
    """Resumable state for chronological event/control/markout shards.

    Control reuse is global across the requested candidate universe, so reuse is
    valid only for a contiguous completed prefix of dates.  If any date in the
    prefix is missing/stale, that date and every later date are rebuilt in
    chronological order.
    """
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS aggregation_day_state (
                symbol TEXT NOT NULL,
                receive_date TEXT NOT NULL,
                signature TEXT NOT NULL,
                status TEXT NOT NULL,
                event_path TEXT,
                control_path TEXT,
                markout_path TEXT,
                summary_json TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(symbol, receive_date)
            )
            """
        )
        self.conn.commit()

    def reusable(self, symbol: str, day: date, signature: str, paths: Sequence[Path]) -> bool:
        row = self.conn.execute(
            "SELECT signature,status FROM aggregation_day_state WHERE symbol=? AND receive_date=?",
            (symbol, day.isoformat()),
        ).fetchone()
        return bool(row and row[0] == signature and row[1] == "COMPLETED" and all(p.exists() for p in paths))

    def put(self, symbol: str, day: date, signature: str, status: str, paths: Sequence[Path], summary: Mapping[str, Any]) -> None:
        ep, cp, mp = [str(p) for p in paths]
        self.conn.execute(
            """
            INSERT INTO aggregation_day_state(symbol,receive_date,signature,status,event_path,control_path,markout_path,summary_json,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?)
            ON CONFLICT(symbol,receive_date) DO UPDATE SET
              signature=excluded.signature,status=excluded.status,event_path=excluded.event_path,
              control_path=excluded.control_path,markout_path=excluded.markout_path,
              summary_json=excluded.summary_json,updated_at=excluded.updated_at
            """,
            (symbol, day.isoformat(), signature, status, ep, cp, mp, json.dumps(summary, sort_keys=True, default=str), utc_now_iso()),
        )
        self.conn.commit()

    def summary(self, symbol: str, day: date) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT summary_json FROM aggregation_day_state WHERE symbol=? AND receive_date=?",
            (symbol, day.isoformat()),
        ).fetchone()
        if not row or not row[0]:
            return {}
        try:
            return json.loads(row[0])
        except Exception:
            return {}

    def close(self) -> None:
        self.conn.close()


class BuildState:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS day_state (
                symbol TEXT NOT NULL,
                receive_date TEXT NOT NULL,
                signature TEXT NOT NULL,
                status TEXT NOT NULL,
                output_path TEXT,
                summary_json TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(symbol, receive_date)
            )
            """
        )
        self.conn.commit()

    def reusable(self, symbol: str, day: date, signature: str, output: Path) -> bool:
        row = self.conn.execute(
            "SELECT signature,status,output_path FROM day_state WHERE symbol=? AND receive_date=?",
            (symbol, day.isoformat()),
        ).fetchone()
        return bool(row and row[0] == signature and row[1] == "COMPLETED" and output.exists())

    def put(self, symbol: str, day: date, signature: str, status: str, output: Path, summary: Mapping[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO day_state(symbol,receive_date,signature,status,output_path,summary_json,updated_at)
            VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(symbol,receive_date) DO UPDATE SET
              signature=excluded.signature,
              status=excluded.status,
              output_path=excluded.output_path,
              summary_json=excluded.summary_json,
              updated_at=excluded.updated_at
            """,
            (
                symbol,
                day.isoformat(),
                signature,
                status,
                str(output),
                json.dumps(summary, sort_keys=True, default=str),
                utc_now_iso(),
            ),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


@dataclass(frozen=True)
class LayerPaths:
    execution: Path
    flow: Path | None
    core: Path | None

    @property
    def all_files(self) -> tuple[Path, ...]:
        return tuple(p for p in (self.execution, self.flow, self.core) if p is not None)


def partition_path(root: Path, symbol: str, day: date) -> Path | None:
    base = root / f"symbol={symbol}" / f"date={day.isoformat()}"
    if not base.exists():
        return None
    files = sorted(base.glob("*.parquet"))
    if not files:
        return None
    if len(files) > 1:
        # Current Stream derived layers use one file/date.  Multiple shards are
        # supported by read_layer_files but represented here by the first path
        # only to keep the date manifest simple.
        return files[0]
    return files[0]


def partition_files(root: Path, symbol: str, day: date) -> tuple[Path, ...]:
    base = root / f"symbol={symbol}" / f"date={day.isoformat()}"
    return tuple(sorted(base.glob("*.parquet"))) if base.exists() else ()


def discover_execution_days(root: Path, symbol: str, start: date | None, end: date | None) -> list[date]:
    base = root / f"symbol={symbol}"
    if not base.exists():
        raise SystemExit(f"Execution symbol root does not exist: {base}")
    days: list[date] = []
    for child in sorted(base.glob("date=????-??-??")):
        try:
            d = date.fromisoformat(child.name.split("=", 1)[1])
        except Exception:
            continue
        if start and d < start:
            continue
        if end and d > end:
            continue
        if any(child.glob("*.parquet")):
            days.append(d)
    if not days:
        raise SystemExit("No matching Phase-6A execution partitions found")
    return days


def read_files(files: Sequence[Path], columns: Sequence[str], required: bool = True) -> pd.DataFrame:
    require_arrow()
    frames: list[pd.DataFrame] = []
    for path in files:
        pf = pq.ParquetFile(path)
        try:
            names = set(pf.schema_arrow.names)
            missing = [c for c in columns if c not in names]
            if missing:
                if required:
                    raise RuntimeError(f"{path} missing required columns: {missing}")
                use = [c for c in columns if c in names]
            else:
                use = list(columns)
            frames.append(pf.read(columns=use).to_pandas())
        finally:
            pf.close()
    if not frames:
        return pd.DataFrame(columns=list(columns))
    out = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if "receive_second" in out:
        out = out.sort_values("receive_second", kind="mergesort").drop_duplicates("receive_second", keep="last")
    return out.reset_index(drop=True)


EXEC_PRIMITIVE_COLUMNS = (
    "symbol",
    "receive_second",
    "book_valid",
    "state_age_ms",
    "best_bid",
    "best_ask",
    "mid_price",
    "spread_bps",
    "bid_depth_quote_5bps",
    "ask_depth_quote_5bps",
    "bid_depth_quote_10bps",
    "ask_depth_quote_10bps",
    "bid_depth_quote_25bps",
    "ask_depth_quote_25bps",
    "force_order_count",
    "force_sell_count",
    "force_buy_count",
    "force_selected_notional_sum",
)

FLOW_COLUMNS = (
    "receive_second",
    "flow_feature_context_ready",
    "realized_vol_60s_bps",
    "quote_volume_5s",
    "quote_volume_60s",
    "signed_quote_flow_5s",
    "aggressive_flow_imbalance_5s",
    "open_interest_change_60s_pct",
    "open_interest_change_300s_pct",
)

CORE_COLUMNS = (
    "receive_second",
    "core_context_ready",
    "last_trade_price",
    "trade_price_fresh",
    "mark_price",
    "mark_fresh",
)


def read_day_context(
    execution_root: Path,
    flow_root: Path,
    core_root: Path,
    symbol: str,
    day: date,
    lookback_days: int = 1,
) -> tuple[pd.DataFrame, list[Path]]:
    days = [day - timedelta(days=i) for i in range(lookback_days, -1, -1)]
    exec_files: list[Path] = []
    flow_files: list[Path] = []
    core_files: list[Path] = []
    for d in days:
        exec_files.extend(partition_files(execution_root, symbol, d))
        flow_files.extend(partition_files(flow_root, symbol, d))
        core_files.extend(partition_files(core_root, symbol, d))
    if not partition_files(execution_root, symbol, day):
        raise RuntimeError(f"Missing execution partition for {day}")
    if not partition_files(flow_root, symbol, day):
        raise RuntimeError(f"Missing flow-feature partition for {day}")
    if not partition_files(core_root, symbol, day):
        raise RuntimeError(f"Missing core partition for {day}")

    execution = read_files(exec_files, EXEC_PRIMITIVE_COLUMNS)
    flow = read_files(flow_files, FLOW_COLUMNS)
    core = read_files(core_files, CORE_COLUMNS)
    frame = execution.merge(flow, on="receive_second", how="left", validate="one_to_one")
    frame = frame.merge(core, on="receive_second", how="left", validate="one_to_one")
    frame = frame.sort_values("receive_second", kind="mergesort").reset_index(drop=True)
    return frame, [*exec_files, *flow_files, *core_files]


def exact_prior_window_ready(seconds: np.ndarray, window: int, min_rows: int) -> np.ndarray:
    n = len(seconds)
    ready = np.zeros(n, dtype=bool)
    if window <= 0 or n == 0:
        return ready
    # For row i, the baseline is up to ``window`` prior rows and never includes
    # the current row.  Before a full window exists we allow the explicitly
    # configured minimum, but only when the receive-second path is exact.
    for i in range(min_rows, n):
        start = max(0, i - window)
        count = i - start
        if count < min_rows:
            continue
        if seconds[i - 1] - seconds[start] == count - 1:
            ready[i] = True
    return ready


def causal_zscore(series: pd.Series, seconds: np.ndarray, window: int, min_rows: int) -> tuple[pd.Series, pd.Series]:
    values = pd.to_numeric(series, errors="coerce")
    shifted = values.shift(1)
    mean = shifted.rolling(window, min_periods=min_rows).mean()
    std = shifted.rolling(window, min_periods=min_rows).std(ddof=0)
    exact_ready = exact_prior_window_ready(seconds, window, min_rows)
    z = (values - mean) / std.replace(0.0, np.nan)
    z = z.where(exact_ready)
    ready = pd.Series(exact_ready, index=series.index) & mean.notna() & std.gt(0)
    return z, ready


def build_primitive_frame(context: pd.DataFrame, day: date, cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    f = context.copy()
    numeric = [
        c for c in f.columns if c not in {
            "symbol", "book_valid", "flow_feature_context_ready", "core_context_ready",
            "trade_price_fresh", "mark_fresh"
        }
    ]
    for c in numeric:
        if c in f:
            f[c] = pd.to_numeric(f[c], errors="coerce")

    seconds = pd.to_numeric(f["receive_second"], errors="raise").to_numpy(np.int64)
    prev_exact = pd.Series(False, index=f.index)
    if len(f) > 1:
        prev_exact.iloc[1:] = np.diff(seconds) == 1
    prev_book_valid = f["book_valid"].shift(1).fillna(False).astype(bool)

    input_ready = (
        f["book_valid"].fillna(False).astype(bool)
        & f["flow_feature_context_ready"].fillna(False).astype(bool)
        & f["core_context_ready"].fillna(False).astype(bool)
    )
    f["primitive_input_ready"] = input_ready

    bid = pd.to_numeric(f["bid_depth_quote_5bps"], errors="coerce")
    ask = pd.to_numeric(f["ask_depth_quote_5bps"], errors="coerce")
    prev_bid = bid.shift(1)
    prev_ask = ask.shift(1)
    depth_pair_ready = input_ready & prev_exact & prev_book_valid & prev_bid.gt(0) & prev_ask.gt(0) & bid.ge(0) & ask.ge(0)
    bid_withdraw = (1.0 - bid / prev_bid).clip(lower=0.0, upper=1.0).where(depth_pair_ready)
    ask_withdraw = (1.0 - ask / prev_ask).clip(lower=0.0, upper=1.0).where(depth_pair_ready)
    f["bid_withdrawal_frac_1s"] = bid_withdraw
    f["ask_withdrawal_frac_1s"] = ask_withdraw
    f["bid_withdrawal_z_900s"], f["bid_withdrawal_baseline_ready"] = causal_zscore(
        bid_withdraw, seconds, cfg.baseline_seconds, cfg.baseline_min_seconds
    )
    f["ask_withdrawal_z_900s"], f["ask_withdrawal_baseline_ready"] = causal_zscore(
        ask_withdraw, seconds, cfg.baseline_seconds, cfg.baseline_min_seconds
    )
    total_depth = bid + ask
    f["depth_imbalance_5bps"] = ((bid - ask) / total_depth.replace(0.0, np.nan)).where(input_ready)
    f["total_depth_5bps"] = total_depth.where(input_ready)
    f["log_total_depth_5bps"] = np.log1p(f["total_depth_5bps"].clip(lower=0))

    signed5 = pd.to_numeric(f["signed_quote_flow_5s"], errors="coerce")
    imb5 = pd.to_numeric(f["aggressive_flow_imbalance_5s"], errors="coerce")
    qv5 = pd.to_numeric(f["quote_volume_5s"], errors="coerce")
    f["signed_quote_flow_5s_z_900s"], f["flow_baseline_ready"] = causal_zscore(
        signed5, seconds, cfg.baseline_seconds, cfg.baseline_min_seconds
    )
    log_qv5 = np.log1p(qv5.clip(lower=0.0))
    f["quote_volume_5s_z_900s"], f["flow_volume_baseline_ready"] = causal_zscore(
        log_qv5, seconds, cfg.baseline_seconds, cfg.baseline_min_seconds
    )

    trade = pd.to_numeric(f["last_trade_price"], errors="coerce")
    mark = pd.to_numeric(f["mark_price"], errors="coerce")
    fresh_pair = (
        input_ready
        & f["trade_price_fresh"].fillna(False).astype(bool)
        & f["mark_fresh"].fillna(False).astype(bool)
        & trade.gt(0)
        & mark.gt(0)
    )
    dislocation = ((trade / mark) - 1.0) * 10_000.0
    f["mark_last_dislocation_bps"] = dislocation.where(fresh_pair)
    f["mark_last_dislocation_z_900s"], f["dislocation_baseline_ready"] = causal_zscore(
        f["mark_last_dislocation_bps"], seconds, cfg.baseline_seconds, cfg.baseline_min_seconds
    )

    # Depth: ask withdrawal is upward vulnerability (+1), bid withdrawal is
    # downward vulnerability (-1).
    f["depth_ask_onset"] = input_ready & ask_withdraw.ge(cfg.depth_onset_frac)
    f["depth_bid_onset"] = input_ready & bid_withdraw.ge(cfg.depth_onset_frac)
    f["depth_ask_detect"] = (
        f["depth_ask_onset"]
        & ask_withdraw.ge(cfg.depth_detect_frac)
        & f["ask_withdrawal_z_900s"].ge(cfg.depth_detect_z)
    )
    f["depth_bid_detect"] = (
        f["depth_bid_onset"]
        & bid_withdraw.ge(cfg.depth_detect_frac)
        & f["bid_withdrawal_z_900s"].ge(cfg.depth_detect_z)
    )

    # Aggressive flow: use both directional imbalance and a causal surprise in
    # 5-second signed flow.  A non-negative volume z guard avoids classifying a
    # tiny one-sided print as a major flow event by imbalance alone.
    abs_imb = imb5.abs()
    flow_z = pd.to_numeric(f["signed_quote_flow_5s_z_900s"], errors="coerce")
    vol_z = pd.to_numeric(f["quote_volume_5s_z_900s"], errors="coerce")
    f["flow_buy_onset"] = input_ready & imb5.ge(cfg.flow_onset_imbalance)
    f["flow_sell_onset"] = input_ready & imb5.le(-cfg.flow_onset_imbalance)
    f["flow_buy_detect"] = (
        f["flow_buy_onset"]
        & abs_imb.ge(cfg.flow_detect_imbalance)
        & flow_z.ge(cfg.flow_detect_z)
        & vol_z.ge(cfg.flow_volume_z)
    )
    f["flow_sell_detect"] = (
        f["flow_sell_onset"]
        & abs_imb.ge(cfg.flow_detect_imbalance)
        & flow_z.le(-cfg.flow_detect_z)
        & vol_z.ge(cfg.flow_volume_z)
    )

    # Mark-last dislocation. Positive means the observed last trade is above
    # mark; negative means below mark.
    d = pd.to_numeric(f["mark_last_dislocation_bps"], errors="coerce")
    dz = pd.to_numeric(f["mark_last_dislocation_z_900s"], errors="coerce")
    f["dislocation_up_onset"] = fresh_pair & d.ge(cfg.dislocation_onset_bps)
    f["dislocation_down_onset"] = fresh_pair & d.le(-cfg.dislocation_onset_bps)
    f["dislocation_up_detect"] = (
        f["dislocation_up_onset"]
        & d.ge(cfg.dislocation_detect_bps)
        & dz.ge(cfg.dislocation_detect_z)
    )
    f["dislocation_down_detect"] = (
        f["dislocation_down_onset"]
        & d.le(-cfg.dislocation_detect_bps)
        & dz.le(-cfg.dislocation_detect_z)
    )

    force_count = pd.to_numeric(f["force_order_count"], errors="coerce").fillna(0)
    force_sell = pd.to_numeric(f["force_sell_count"], errors="coerce").fillna(0)
    force_buy = pd.to_numeric(f["force_buy_count"], errors="coerce").fillna(0)
    force_ready = f["book_valid"].fillna(False).astype(bool) & force_count.gt(0)
    f["force_buy_onset"] = force_ready & force_buy.gt(force_sell)
    f["force_sell_onset"] = force_ready & force_sell.gt(force_buy)
    f["force_mixed_onset"] = force_ready & force_sell.eq(force_buy) & force_count.gt(0)
    f["force_buy_detect"] = f["force_buy_onset"]
    f["force_sell_detect"] = f["force_sell_onset"]
    f["force_mixed_detect"] = f["force_mixed_onset"]

    onset_cols = [
        "depth_ask_onset", "depth_bid_onset", "flow_buy_onset", "flow_sell_onset",
        "dislocation_up_onset", "dislocation_down_onset", "force_buy_onset",
        "force_sell_onset", "force_mixed_onset",
    ]
    detect_cols = [c.replace("_onset", "_detect") for c in onset_cols]
    f["primitive_any_onset"] = f[onset_cols].any(axis=1)
    f["primitive_any_detect"] = f[detect_cols].any(axis=1)

    f["log_quote_volume_60s"] = np.log1p(pd.to_numeric(f["quote_volume_60s"], errors="coerce").clip(lower=0.0))
    f["receive_date"] = pd.to_datetime(f["receive_second"], unit="s", utc=True).dt.date.astype(str)
    f["utc_hour"] = pd.to_datetime(f["receive_second"], unit="s", utc=True).dt.hour.astype(np.int8)

    day_start = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())
    current = f.loc[(f["receive_second"] >= day_start) & (f["receive_second"] < day_start + SECONDS_PER_DAY)].copy()
    current = current.reset_index(drop=True)
    summary = {
        "receive_date": day.isoformat(),
        "rows": int(len(current)),
        "input_ready_rows": int(current["primitive_input_ready"].sum()),
        "book_valid_rows": int(current["book_valid"].fillna(False).sum()),
        "any_onset_rows": int(current["primitive_any_onset"].sum()),
        "any_detect_rows": int(current["primitive_any_detect"].sum()),
        "depth_ask_detect_rows": int(current["depth_ask_detect"].sum()),
        "depth_bid_detect_rows": int(current["depth_bid_detect"].sum()),
        "flow_buy_detect_rows": int(current["flow_buy_detect"].sum()),
        "flow_sell_detect_rows": int(current["flow_sell_detect"].sum()),
        "dislocation_up_detect_rows": int(current["dislocation_up_detect"].sum()),
        "dislocation_down_detect_rows": int(current["dislocation_down_detect"].sum()),
        "force_detect_rows": int(current[["force_buy_detect", "force_sell_detect", "force_mixed_detect"]].any(axis=1).sum()),
    }
    return current, summary


PRIMITIVE_OUTPUT_COLUMNS = (
    "symbol", "receive_second", "receive_date", "utc_hour",
    "primitive_input_ready", "book_valid", "state_age_ms", "mid_price", "spread_bps",
    "bid_depth_quote_5bps", "ask_depth_quote_5bps", "bid_depth_quote_10bps", "ask_depth_quote_10bps",
    "bid_depth_quote_25bps", "ask_depth_quote_25bps", "total_depth_5bps", "log_total_depth_5bps",
    "depth_imbalance_5bps", "bid_withdrawal_frac_1s", "ask_withdrawal_frac_1s",
    "bid_withdrawal_z_900s", "ask_withdrawal_z_900s",
    "realized_vol_60s_bps", "quote_volume_5s", "quote_volume_60s", "log_quote_volume_60s",
    "signed_quote_flow_5s", "aggressive_flow_imbalance_5s", "signed_quote_flow_5s_z_900s",
    "quote_volume_5s_z_900s", "open_interest_change_60s_pct", "open_interest_change_300s_pct",
    "last_trade_price", "trade_price_fresh", "mark_price", "mark_fresh",
    "mark_last_dislocation_bps", "mark_last_dislocation_z_900s",
    "force_order_count", "force_sell_count", "force_buy_count", "force_selected_notional_sum",
    "depth_ask_onset", "depth_ask_detect", "depth_bid_onset", "depth_bid_detect",
    "flow_buy_onset", "flow_buy_detect", "flow_sell_onset", "flow_sell_detect",
    "dislocation_up_onset", "dislocation_up_detect", "dislocation_down_onset", "dislocation_down_detect",
    "force_buy_onset", "force_buy_detect", "force_sell_onset", "force_sell_detect",
    "force_mixed_onset", "force_mixed_detect", "primitive_any_onset", "primitive_any_detect",
)


def primitive_output_path(root: Path, symbol: str, day: date) -> Path:
    return root / "primitives" / f"symbol={symbol}" / f"date={day.isoformat()}" / f"part-{day.isoformat()}.parquet"


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    direction: int
    onset_col: str
    detect_col: str
    intensity_col: str
    surprise_col: str | None


SPECS = (
    PrimitiveSpec("DEPTH_WITHDRAWAL", +1, "depth_ask_onset", "depth_ask_detect", "ask_withdrawal_frac_1s", "ask_withdrawal_z_900s"),
    PrimitiveSpec("DEPTH_WITHDRAWAL", -1, "depth_bid_onset", "depth_bid_detect", "bid_withdrawal_frac_1s", "bid_withdrawal_z_900s"),
    PrimitiveSpec("AGGRESSIVE_FLOW", +1, "flow_buy_onset", "flow_buy_detect", "aggressive_flow_imbalance_5s", "signed_quote_flow_5s_z_900s"),
    PrimitiveSpec("AGGRESSIVE_FLOW", -1, "flow_sell_onset", "flow_sell_detect", "aggressive_flow_imbalance_5s", "signed_quote_flow_5s_z_900s"),
    PrimitiveSpec("MARK_LAST_DISLOCATION", +1, "dislocation_up_onset", "dislocation_up_detect", "mark_last_dislocation_bps", "mark_last_dislocation_z_900s"),
    PrimitiveSpec("MARK_LAST_DISLOCATION", -1, "dislocation_down_onset", "dislocation_down_detect", "mark_last_dislocation_bps", "mark_last_dislocation_z_900s"),
    PrimitiveSpec("FORCE_ORDER", +1, "force_buy_onset", "force_buy_detect", "force_selected_notional_sum", None),
    PrimitiveSpec("FORCE_ORDER", -1, "force_sell_onset", "force_sell_detect", "force_selected_notional_sum", None),
    PrimitiveSpec("FORCE_ORDER", 0, "force_mixed_onset", "force_mixed_detect", "force_selected_notional_sum", None),
)

EVENT_STATE_COLUMNS = (
    "receive_second", "receive_date", "utc_hour", "mid_price", "spread_bps",
    "realized_vol_60s_bps", "quote_volume_60s", "log_quote_volume_60s",
    "total_depth_5bps", "log_total_depth_5bps", "depth_imbalance_5bps",
    "open_interest_change_60s_pct", "open_interest_change_300s_pct",
    "force_order_count", "force_selected_notional_sum",
)


def cluster_seconds(values: Sequence[int], bridge_seconds: int) -> list[tuple[int, int, list[int]]]:
    if not values:
        return []
    vals = sorted(set(int(v) for v in values))
    groups: list[tuple[int, int, list[int]]] = []
    cur = [vals[0]]
    for value in vals[1:]:
        if value - cur[-1] <= bridge_seconds:
            cur.append(value)
        else:
            groups.append((cur[0], cur[-1], cur))
            cur = [value]
    groups.append((cur[0], cur[-1], cur))
    return groups


def control_family_slug(primitive: str) -> str:
    return primitive.lower().replace("_", "")


def family_specs() -> dict[str, list[PrimitiveSpec]]:
    out: dict[str, list[PrimitiveSpec]] = defaultdict(list)
    for spec in SPECS:
        out[spec.name].append(spec)
    return dict(out)


def collect_hits_and_candidates(paths: Sequence[Path], cfg: Config) -> tuple[dict[tuple[str, int], pd.DataFrame], pd.DataFrame, list[dict[str, Any]]]:
    """Collect event hits and a causal, primitive-specific control candidate grid.

    v1.0.3 preserves the v1.0.1 primitive-specific causal control contract while
    collecting candidates in a form that can be matched with vectorized NumPy
    kernels rather than per-candidate Pandas loops.
    For each primitive family we retain two causal eligibility fields:

    - ``control_neutral_<family>``: no same-family weak onset at the anchor.
    - ``seconds_since_<family>_detect``: elapsed seconds since the latest
      same-family strict detection at or before the anchor.

    The latter is constructed using only present/past rows and carries state
    across day partitions, so it introduces no future-event embargo.
    """
    require_arrow()
    hits: dict[tuple[str, int], list[pd.DataFrame]] = defaultdict(list)
    candidates: list[pd.DataFrame] = []
    quality_rows: list[dict[str, Any]] = []
    families = family_specs()
    last_detect_by_family: dict[str, int | None] = {name: None for name in families}

    read_cols = sorted(set(PRIMITIVE_OUTPUT_COLUMNS))
    for path in sorted(paths):
        frame = read_files([path], read_cols)
        if frame.empty:
            continue
        frame = frame.sort_values("receive_second", kind="mergesort").reset_index(drop=True)
        day_text = str(frame["receive_date"].iloc[0])
        quality_rows.append({
            "receive_date": day_text,
            "rows": len(frame),
            "primitive_input_ready_rows": int(frame["primitive_input_ready"].fillna(False).sum()),
            "book_valid_rows": int(frame["book_valid"].fillna(False).sum()),
            "any_onset_rows": int(frame["primitive_any_onset"].fillna(False).sum()),
            "any_detect_rows": int(frame["primitive_any_detect"].fillna(False).sum()),
        })
        for spec in SPECS:
            mask = frame[spec.onset_col].fillna(False).astype(bool) | frame[spec.detect_col].fillna(False).astype(bool)
            if not mask.any():
                continue
            cols = list(EVENT_STATE_COLUMNS) + [spec.onset_col, spec.detect_col, spec.intensity_col]
            if spec.surprise_col:
                cols.append(spec.surprise_col)
            part = frame.loc[mask, list(dict.fromkeys(cols))].copy()
            hits[(spec.name, spec.direction)].append(part)

        sec = pd.to_numeric(frame["receive_second"], errors="coerce").fillna(-1).astype(np.int64)
        cadence = (sec % cfg.control_cadence_seconds) == 0
        cmask = frame["primitive_input_ready"].fillna(False).astype(bool) & cadence
        ccols = list(EVENT_STATE_COLUMNS) + ["primitive_any_onset", "primitive_any_detect"]
        c = frame.loc[cmask, ccols].copy()
        candidate_seconds = pd.to_numeric(c["receive_second"], errors="coerce").to_numpy(np.int64)

        for primitive, specs in families.items():
            slug = control_family_slug(primitive)
            onset_cols = [spec.onset_col for spec in specs]
            detect_cols = [spec.detect_col for spec in specs]
            family_onset = frame[onset_cols].fillna(False).astype(bool).any(axis=1)
            family_detect = frame[detect_cols].fillna(False).astype(bool).any(axis=1)
            c[f"control_neutral_{slug}"] = (~family_onset.loc[c.index]).to_numpy(bool)

            detect_seconds = pd.to_numeric(
                frame.loc[family_detect, "receive_second"], errors="coerce"
            ).dropna().astype(np.int64).to_numpy()
            prev_before_day = last_detect_by_family.get(primitive)
            since = np.full(len(candidate_seconds), np.inf, dtype=float)
            if len(candidate_seconds):
                if len(detect_seconds):
                    pos = np.searchsorted(detect_seconds, candidate_seconds, side="right") - 1
                    has_current = pos >= 0
                    if has_current.any():
                        since[has_current] = candidate_seconds[has_current] - detect_seconds[pos[has_current]]
                    if prev_before_day is not None:
                        missing = ~has_current
                        since[missing] = candidate_seconds[missing] - int(prev_before_day)
                elif prev_before_day is not None:
                    since[:] = candidate_seconds - int(prev_before_day)
            c[f"seconds_since_{slug}_detect"] = since
            if len(detect_seconds):
                last_detect_by_family[primitive] = int(detect_seconds[-1])

        candidates.append(c)

    hit_frames = {
        key: (pd.concat(parts, ignore_index=True).sort_values("receive_second", kind="mergesort").reset_index(drop=True))
        for key, parts in hits.items()
    }
    controls = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    if not controls.empty:
        controls = controls.sort_values("receive_second", kind="mergesort").drop_duplicates("receive_second").reset_index(drop=True)
    return hit_frames, controls, quality_rows


def collect_candidates_and_quality(paths: Sequence[Path], cfg: Config) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Scan all primitive partitions once for the global v1.0.2 candidate universe.

    Unlike ``collect_hits_and_candidates`` this deliberately does not retain
    event-hit frames.  The candidate universe is the only range-global object
    needed to preserve ANY_DATE_SAME_HOUR matching and robust scales.
    """
    require_arrow()
    candidates: list[pd.DataFrame] = []
    quality_rows: list[dict[str, Any]] = []
    families = family_specs()
    last_detect_by_family: dict[str, int | None] = {name: None for name in families}
    read_cols = sorted(set(PRIMITIVE_OUTPUT_COLUMNS))
    for i, path in enumerate(sorted(paths), 1):
        frame = read_files([path], read_cols)
        if frame.empty:
            continue
        frame = frame.sort_values("receive_second", kind="mergesort").reset_index(drop=True)
        day_text = str(frame["receive_date"].iloc[0])
        quality_rows.append({
            "receive_date": day_text,
            "rows": len(frame),
            "primitive_input_ready_rows": int(frame["primitive_input_ready"].fillna(False).sum()),
            "book_valid_rows": int(frame["book_valid"].fillna(False).sum()),
            "any_onset_rows": int(frame["primitive_any_onset"].fillna(False).sum()),
            "any_detect_rows": int(frame["primitive_any_detect"].fillna(False).sum()),
        })
        sec = pd.to_numeric(frame["receive_second"], errors="coerce").fillna(-1).astype(np.int64)
        cadence = (sec % cfg.control_cadence_seconds) == 0
        cmask = frame["primitive_input_ready"].fillna(False).astype(bool) & cadence
        ccols = list(EVENT_STATE_COLUMNS) + ["primitive_any_onset", "primitive_any_detect"]
        c = frame.loc[cmask, ccols].copy()
        candidate_seconds = pd.to_numeric(c["receive_second"], errors="coerce").to_numpy(np.int64)
        for primitive, specs in families.items():
            slug = control_family_slug(primitive)
            onset_cols = [spec.onset_col for spec in specs]
            detect_cols = [spec.detect_col for spec in specs]
            family_onset = frame[onset_cols].fillna(False).astype(bool).any(axis=1)
            family_detect = frame[detect_cols].fillna(False).astype(bool).any(axis=1)
            c[f"control_neutral_{slug}"] = (~family_onset.loc[c.index]).to_numpy(bool)
            detect_seconds = pd.to_numeric(frame.loc[family_detect, "receive_second"], errors="coerce").dropna().astype(np.int64).to_numpy()
            prev_before_day = last_detect_by_family.get(primitive)
            since = np.full(len(candidate_seconds), np.inf, dtype=float)
            if len(candidate_seconds):
                if len(detect_seconds):
                    dpos = np.searchsorted(detect_seconds, candidate_seconds, side="right") - 1
                    has_current = dpos >= 0
                    if has_current.any():
                        since[has_current] = candidate_seconds[has_current] - detect_seconds[dpos[has_current]]
                    if prev_before_day is not None:
                        missing = ~has_current
                        since[missing] = candidate_seconds[missing] - int(prev_before_day)
                elif prev_before_day is not None:
                    since[:] = candidate_seconds - int(prev_before_day)
            c[f"seconds_since_{slug}_detect"] = since
            if len(detect_seconds):
                last_detect_by_family[primitive] = int(detect_seconds[-1])
        candidates.append(c)
        if i == 1 or i % 10 == 0 or i == len(paths):
            print(f"    candidate scan [{i:02d}/{len(paths):02d}] {day_text}: {len(c):,} anchors", flush=True)
    out = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    if not out.empty:
        out = out.sort_values("receive_second", kind="mergesort").drop_duplicates("receive_second").reset_index(drop=True)
    return out, quality_rows


def collect_hits_only(paths: Sequence[Path]) -> dict[tuple[str, int], pd.DataFrame]:
    hits: dict[tuple[str, int], list[pd.DataFrame]] = defaultdict(list)
    read_cols = sorted(set(PRIMITIVE_OUTPUT_COLUMNS))
    for path in sorted(paths):
        frame = read_files([path], read_cols)
        if frame.empty:
            continue
        frame = frame.sort_values("receive_second", kind="mergesort").reset_index(drop=True)
        for spec in SPECS:
            mask = frame[spec.onset_col].fillna(False).astype(bool) | frame[spec.detect_col].fillna(False).astype(bool)
            if not mask.any():
                continue
            cols = list(EVENT_STATE_COLUMNS) + [spec.onset_col, spec.detect_col, spec.intensity_col]
            if spec.surprise_col:
                cols.append(spec.surprise_col)
            hits[(spec.name, spec.direction)].append(frame.loc[mask, list(dict.fromkeys(cols))].copy())
    return {
        key: pd.concat(parts, ignore_index=True).sort_values("receive_second", kind="mergesort").reset_index(drop=True)
        for key, parts in hits.items()
    }


def aggregation_output_path(root: Path, dataset: str, symbol: str, day: date) -> Path:
    return root / dataset / f"symbol={symbol}" / f"date={day.isoformat()}" / f"part-{day.isoformat()}.parquet"


def aggregation_report_dir(report_root: Path, day: date) -> Path:
    return report_root / "daily" / f"date={day.isoformat()}"


def prepare_control_matcher(candidates: pd.DataFrame, cfg: Config) -> dict[str, dict[str, Any]]:
    """Prepare the exact v1.0.2 candidate pools once for chronological shards."""
    if candidates.empty:
        return {}
    cand = candidates.copy().dropna(subset=list(MATCH_COLUMNS)).reset_index(drop=True)
    primitive_data: dict[str, dict[str, Any]] = {}
    for primitive in sorted(family_specs()):
        slug = control_family_slug(primitive)
        neutral_col = f"control_neutral_{slug}"
        since_col = f"seconds_since_{slug}_detect"
        # Synthetic/unit callers may provide candidates for only the primitive
        # under test. Production candidate scans always contain all families.
        if neutral_col not in cand.columns or since_col not in cand.columns:
            continue
        neutral = cand[neutral_col].fillna(False).astype(bool).to_numpy(bool)
        since = pd.to_numeric(cand[since_col], errors="coerce").to_numpy(float)
        eligible = neutral & (since > cfg.control_trailing_cooldown_seconds)
        pool = cand.loc[eligible].copy().reset_index(drop=True)
        scales = np.asarray([robust_scale(pool[c]) for c in MATCH_COLUMNS], dtype=float) if not pool.empty else np.ones(len(MATCH_COLUMNS), dtype=float)
        groups: dict[tuple[str, int], np.ndarray] = {}
        hours: dict[int, np.ndarray] = {}
        if not pool.empty:
            for key, idx in pool.groupby(["receive_date", "utc_hour"], sort=False).groups.items():
                groups[(str(key[0]), int(key[1]))] = np.asarray(list(idx), dtype=np.int64)
            hours = {int(h): np.asarray(list(idx), dtype=np.int64) for h, idx in pool.groupby("utc_hour", sort=False).groups.items()}
        seconds = pd.to_numeric(pool.get("receive_second", pd.Series(dtype=float)), errors="coerce").to_numpy(np.int64)
        primitive_data[primitive] = {
            "pool": pool,
            "seconds": seconds,
            "features": pool.loc[:, list(MATCH_COLUMNS)].apply(pd.to_numeric, errors="coerce").to_numpy(float) if not pool.empty else np.empty((0, len(MATCH_COLUMNS))),
            "scales": scales,
            "groups": groups,
            "hours": hours,
            "usage": np.zeros(len(pool), dtype=np.int16),
            "since_col": since_col,
        }
    return primitive_data


def restore_control_usage(primitive_data: Mapping[str, dict[str, Any]], control_paths: Sequence[Path]) -> None:
    """Restore global candidate reuse counts from a contiguous completed prefix."""
    for path in control_paths:
        if not path.exists():
            continue
        frame = read_files([path], ["primitive", "anchor_second", "candidate_reuse_ordinal"])
        if frame.empty:
            continue
        for primitive, g in frame.groupby("primitive", sort=False):
            pdata = primitive_data.get(str(primitive))
            if not pdata:
                continue
            seconds: np.ndarray = pdata["seconds"]
            usage: np.ndarray = pdata["usage"]
            anchors = pd.to_numeric(g["anchor_second"], errors="coerce").to_numpy(np.int64)
            ords = pd.to_numeric(g["candidate_reuse_ordinal"], errors="coerce").fillna(0).to_numpy(np.int16)
            pos = np.searchsorted(seconds, anchors, side="left")
            valid = (pos < len(seconds)) & (seconds[np.clip(pos, 0, max(len(seconds)-1, 0))] == anchors) if len(seconds) else np.zeros(len(anchors), dtype=bool)
            for idx, ordinal in zip(pos[valid], ords[valid]):
                if ordinal > usage[int(idx)]:
                    usage[int(idx)] = ordinal


def build_controls_prepared(events: pd.DataFrame, primitive_data: Mapping[str, dict[str, Any]], cfg: Config, symbol: str) -> pd.DataFrame:
    """Match one chronological event shard using persistent global usage state."""
    if events.empty or not primitive_data:
        return pd.DataFrame()
    detected = events.loc[events["detected"].fillna(False).astype(bool) & events["direction"].ne(0)].copy()
    if detected.empty:
        return pd.DataFrame()
    out: list[dict[str, Any]] = []
    event_records = detected.sort_values("detection_second", kind="mergesort").to_dict("records")
    for ev in event_records:
        primitive = str(ev["primitive"])
        pdata = primitive_data.get(primitive)
        if not pdata or len(pdata["pool"]) == 0:
            continue
        pool: pd.DataFrame = pdata["pool"]
        seconds: np.ndarray = pdata["seconds"]
        features: np.ndarray = pdata["features"]
        scales: np.ndarray = pdata["scales"]
        usage: np.ndarray = pdata["usage"]
        parent_second = int(ev["detection_second"])
        ev_features = np.asarray([safe_float(ev.get(f"anchor_{c}")) for c in MATCH_COLUMNS], dtype=float)
        if not np.isfinite(ev_features).all():
            continue
        needed = cfg.controls_per_event
        selected: list[tuple[int, str, float, int]] = []
        selected_indices: set[int] = set()
        for tier_name, idxs in candidate_tiers(ev, pdata["groups"], pdata["hours"]):
            if needed <= 0:
                break
            idxs = np.asarray(idxs, dtype=np.int64)
            if len(idxs) == 0:
                continue
            valid = (np.abs(seconds[idxs] - parent_second) > cfg.control_embargo_seconds) & (usage[idxs] < cfg.control_max_reuse)
            if selected_indices:
                valid &= ~np.isin(idxs, np.fromiter(selected_indices, dtype=np.int64))
            if not valid.any():
                continue
            tier_idxs = idxs[valid]
            vals = features[tier_idxs]
            dist = np.sum(np.abs(vals - ev_features) / scales, axis=1)
            dist += usage[tier_idxs].astype(float) * 1e-6
            dist += np.abs(seconds[tier_idxs] - parent_second) / (86400.0 * 1000.0)
            dist[~np.isfinite(dist)] = np.inf
            chosen_local = _stable_k_smallest_positions(dist, tier_idxs, needed)
            for p in chosen_local:
                idx = int(tier_idxs[int(p)])
                if idx in selected_indices or usage[idx] >= cfg.control_max_reuse:
                    continue
                reuse_ordinal = int(usage[idx]) + 1
                selected.append((idx, tier_name, float(dist[int(p)]), reuse_ordinal))
                selected_indices.add(idx)
                usage[idx] = reuse_ordinal
                needed -= 1
                if needed <= 0:
                    break
        slug = control_family_slug(primitive)
        since_col = f"seconds_since_{slug}_detect"
        for ordinal, (idx, tier, distance, reuse_ordinal) in enumerate(selected, 1):
            c = pool.iloc[idx]
            sec = int(seconds[idx])
            row = {
                "control_id": f"CT-{stable_id(ev['event_id'], sec, ordinal)}",
                "parent_event_id": ev["event_id"], "symbol": symbol,
                "primitive": ev["primitive"], "direction": int(ev["direction"]),
                "direction_label": ev["direction_label"], "control_ordinal": ordinal,
                "anchor_second": sec, "anchor_time_utc": datetime.fromtimestamp(sec, tz=timezone.utc),
                "tradeable_second": sec + 1, "match_tier": tier, "match_distance": distance,
                "control_target_event_embargo_seconds": cfg.control_embargo_seconds,
                "control_trailing_cooldown_seconds": cfg.control_trailing_cooldown_seconds,
                "control_max_reuse": cfg.control_max_reuse, "candidate_reuse_ordinal": reuse_ordinal,
                "seconds_since_same_primitive_detect": safe_float(c.get(since_col)),
            }
            for col in EVENT_STATE_COLUMNS:
                if col != "receive_second":
                    row[f"anchor_{col}"] = c.get(col, math.nan)
            out.append(row)
    return pd.DataFrame(out)


def concat_daily_reports(report_root: Path, filename: str, days: Sequence[date]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for day in days:
        path = aggregation_report_dir(report_root, day) / filename
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty:
            continue
        if "receive_date" not in frame.columns:
            frame.insert(0, "receive_date", day.isoformat())
        rows.extend(frame.to_dict("records"))
    return rows


def build_partitioned_findings(summary: Mapping[str, Any]) -> str:
    return "\n".join([
        "# Phase-6B causal liquidity-event primitive findings",
        "",
        f"- Builder: `{BUILDER_VERSION}`",
        f"- Status: **{summary.get('status')}**",
        f"- Dates completed: **{summary.get('dates_completed')}/{summary.get('dates_requested')}**",
        f"- Primitive rows: **{int(summary.get('primitive_rows', 0)):,}**",
        f"- Events: **{int(summary.get('events', 0)):,}**",
        f"- Detected events: **{int(summary.get('detected_events', 0)):,}**",
        f"- Matched controls: **{int(summary.get('controls', 0)):,}**",
        f"- Markout rows: **{int(summary.get('markout_rows', 0)):,}**",
        f"- Failed daily invariants: **{int(summary.get('failed_invariants', 0))}**",
        "",
        "## Aggregation contract",
        "Events, controls and fixed-horizon markouts are materialized as independent UTC-day Parquet shards. "
        "The global candidate universe and robust scales are frozen for the requested date range; controls are then matched in chronological event order with candidate reuse state carried across days. "
        "This preserves the v1.0.2 SAME_DATE/ANY_DATE matching and bounded-reuse semantics without a monolithic event/control/markout build.",
        "",
        "Daily report rows are concatenated into the top-level CSV reports. Pooled bootstrap markout inference is intentionally not recomputed in one monolithic dataframe; Phase 6C should consume the partitioned event/control datasets directly.",
        "",
        "No alpha or strategy claim is made by Phase 6B.",
    ]) + "\n"


def row_at_or_none(frame: pd.DataFrame, second: int) -> Mapping[str, Any] | None:
    if frame.empty:
        return None
    seconds = frame["receive_second"].to_numpy(np.int64)
    pos = int(np.searchsorted(seconds, int(second), side="left"))
    if pos < len(seconds) and int(seconds[pos]) == int(second):
        return frame.iloc[pos].to_dict()
    return None


def build_events(hit_frames: Mapping[tuple[str, int], pd.DataFrame], cfg: Config, symbol: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in SPECS:
        key = (spec.name, spec.direction)
        frame = hit_frames.get(key)
        if frame is None or frame.empty:
            continue
        onset_rows = frame.loc[frame[spec.onset_col].fillna(False).astype(bool)]
        onset_seconds = onset_rows["receive_second"].astype(np.int64).tolist()
        detect_seconds = frame.loc[frame[spec.detect_col].fillna(False).astype(bool), "receive_second"].astype(np.int64).to_numpy()
        for onset, end, cluster in cluster_seconds(onset_seconds, cfg.event_bridge_seconds):
            # Detection thresholds are strict subsets of onset thresholds, so
            # a detector hit must belong to this same onset episode.  Never let
            # a later episode's detector retroactively "detect" an earlier one.
            hi = min(end, onset + cfg.max_detection_lag_seconds)
            left = int(np.searchsorted(detect_seconds, onset, side="left"))
            detection: int | None = None
            if left < len(detect_seconds) and int(detect_seconds[left]) <= hi:
                detection = int(detect_seconds[left])
            onset_row = row_at_or_none(frame, onset) or {}
            detect_row = row_at_or_none(frame, detection) if detection is not None else None
            anchor = detect_row or onset_row
            onset_mid = safe_float(onset_row.get("mid_price"))
            detect_mid = safe_float((detect_row or {}).get("mid_price")) if detection is not None else math.nan
            move_before = bps_return(detect_mid, onset_mid, spec.direction) if spec.direction else math.nan
            eid = f"EV-{stable_id(symbol, spec.name, spec.direction, onset)}"
            row: dict[str, Any] = {
                "event_id": eid,
                "symbol": symbol,
                "primitive": spec.name,
                "direction": spec.direction,
                "direction_label": "UP" if spec.direction > 0 else ("DOWN" if spec.direction < 0 else "MIXED"),
                "onset_second": int(onset),
                "onset_time_utc": datetime.fromtimestamp(onset, tz=timezone.utc),
                "onset_date": datetime.fromtimestamp(onset, tz=timezone.utc).date().isoformat(),
                "episode_last_onset_second": int(end),
                "onset_hit_count": int(len(cluster)),
                "detected": bool(detection is not None),
                "detection_second": detection,
                "detection_time_utc": datetime.fromtimestamp(detection, tz=timezone.utc) if detection is not None else pd.NaT,
                "detection_lag_seconds": int(detection - onset) if detection is not None else math.nan,
                "tradeable_second": int(detection + 1) if detection is not None else math.nan,
                "onset_mid_price": onset_mid,
                "detection_mid_price": detect_mid,
                "move_realized_before_detection_bps": move_before,
                "onset_intensity": safe_float(onset_row.get(spec.intensity_col)),
                "detection_intensity": safe_float((detect_row or {}).get(spec.intensity_col)) if detection is not None else math.nan,
                "onset_surprise_z": safe_float(onset_row.get(spec.surprise_col)) if spec.surprise_col else math.nan,
                "detection_surprise_z": safe_float((detect_row or {}).get(spec.surprise_col)) if detection is not None and spec.surprise_col else math.nan,
                "force_size_is_censored_selected": spec.name == "FORCE_ORDER",
            }
            for c in EVENT_STATE_COLUMNS:
                if c == "receive_second":
                    continue
                row[f"anchor_{c}"] = anchor.get(c, math.nan)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["onset_second", "primitive", "direction"], kind="mergesort").reset_index(drop=True)


def near_any_event_mask(seconds: np.ndarray, event_seconds: np.ndarray, embargo: int) -> np.ndarray:
    if len(event_seconds) == 0:
        return np.zeros(len(seconds), dtype=bool)
    ev = np.sort(np.asarray(event_seconds, dtype=np.int64))
    pos = np.searchsorted(ev, seconds, side="left")
    out = np.zeros(len(seconds), dtype=bool)
    for delta in (0, -1):
        idx = pos + delta
        valid = (idx >= 0) & (idx < len(ev))
        safe = np.clip(idx, 0, len(ev) - 1)
        out |= valid & (np.abs(ev[safe] - seconds) <= embargo)
    return out




def near_any_interval_mask(seconds: np.ndarray, intervals: np.ndarray, embargo: int) -> np.ndarray:
    if intervals.size == 0:
        return np.zeros(len(seconds), dtype=bool)
    raw = sorted((int(a) - embargo, int(b) + embargo) for a, b in intervals)
    merged: list[list[int]] = []
    for a, b in raw:
        if not merged or a > merged[-1][1] + 1:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    starts = np.asarray([x[0] for x in merged], dtype=np.int64)
    ends = np.asarray([x[1] for x in merged], dtype=np.int64)
    pos = np.searchsorted(starts, seconds, side="right") - 1
    valid = pos >= 0
    safe = np.clip(pos, 0, len(ends) - 1)
    return valid & (seconds <= ends[safe])

def robust_scale(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    if len(x) == 0:
        return 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    if math.isfinite(mad) and mad > 1e-12:
        return mad
    std = float(np.std(x))
    return std if math.isfinite(std) and std > 1e-12 else 1.0


def candidate_tiers(event_row: Mapping[str, Any], groups: Mapping[tuple[str, int], np.ndarray], all_by_hour: Mapping[int, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    d = str(event_row.get("anchor_receive_date") or event_row.get("onset_date"))
    h = int(safe_float(event_row.get("anchor_utc_hour"))) if math.isfinite(safe_float(event_row.get("anchor_utc_hour"))) else datetime.fromtimestamp(int(event_row["detection_second"]), tz=timezone.utc).hour
    tiers: list[tuple[str, np.ndarray]] = []
    tiers.append(("SAME_DATE_SAME_HOUR", groups.get((d, h), np.array([], dtype=np.int64))))
    adjacent = [groups.get((d, hh), np.array([], dtype=np.int64)) for hh in {max(0, h - 1), min(23, h + 1)}]
    tiers.append(("SAME_DATE_ADJACENT_HOUR", np.unique(np.concatenate(adjacent)) if adjacent else np.array([], dtype=np.int64)))
    tiers.append(("ANY_DATE_SAME_HOUR", all_by_hour.get(h, np.array([], dtype=np.int64))))
    return tiers


def _stable_k_smallest_positions(dist: np.ndarray, global_indices: np.ndarray, k: int) -> np.ndarray:
    """Return up to k finite local positions by (distance, global index).

    ``argpartition`` avoids sorting an entire tier.  We then include every item
    tied at the kth distance and apply the same deterministic global-index
    tiebreak that the previous stable full sort implied.
    """
    if k <= 0 or len(dist) == 0:
        return np.array([], dtype=np.int64)
    finite = np.flatnonzero(np.isfinite(dist))
    if len(finite) == 0:
        return np.array([], dtype=np.int64)
    if len(finite) <= k:
        order = np.lexsort((global_indices[finite], dist[finite]))
        return finite[order].astype(np.int64, copy=False)
    vals = dist[finite]
    kth = float(np.partition(vals, k - 1)[k - 1])
    tied = finite[vals <= kth]
    order = np.lexsort((global_indices[tied], dist[tied]))
    return tied[order[:k]].astype(np.int64, copy=False)


def build_controls(events: pd.DataFrame, candidates: pd.DataFrame, cfg: Config, symbol: str) -> pd.DataFrame:
    """Compatibility wrapper for one-shot matching; v1.0.3 shards use prepared state."""
    pdata = prepare_control_matcher(candidates, cfg)
    return build_controls_prepared(events, pdata, cfg, symbol)


def markout_exec_columns(notionals: Sequence[float]) -> list[str]:
    cols = ["receive_second", "book_valid", "mid_price"]
    for n in notionals:
        tag = str(int(round(n)))
        for side in ("buy", "sell"):
            cols.extend([
                f"{side}_vwap_{tag}",
                f"{side}_complete_{tag}",
                f"{side}_crossing_cost_bps_{tag}",
            ])
    return cols


def read_execution_for_days(root: Path, symbol: str, days: Sequence[date], columns: Sequence[str]) -> pd.DataFrame:
    files: list[Path] = []
    for d in sorted(set(days)):
        files.extend(partition_files(root, symbol, d))
    return read_files(files, columns) if files else pd.DataFrame(columns=list(columns))


def _lookup_positions(sorted_seconds: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(sorted_seconds) == 0:
        return np.zeros(len(targets), dtype=np.int64), np.zeros(len(targets), dtype=bool)
    pos = np.searchsorted(sorted_seconds, targets, side="left")
    safe = np.clip(pos, 0, len(sorted_seconds) - 1)
    found = (pos < len(sorted_seconds)) & (sorted_seconds[safe] == targets)
    return safe.astype(np.int64, copy=False), found


def _numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(float)


def _bool_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame[column].fillna(False).astype(bool).to_numpy(bool)


def build_markouts(
    events: pd.DataFrame,
    controls: pd.DataFrame,
    execution_root: Path,
    symbol: str,
    cfg: Config,
) -> pd.DataFrame:
    """Build fixed-horizon event/control markouts with vectorized time lookup.

    The output contract and row order remain anchor -> horizon -> notional, as
    in v1.0.1, but Pandas ``.loc`` is no longer called inside the nested loops.
    """
    parts: list[pd.DataFrame] = []
    if not events.empty:
        e = events.loc[events["detected"].fillna(False).astype(bool) & events["direction"].ne(0),
                       ["event_id", "primitive", "direction", "detection_second", "tradeable_second"]].copy()
        if not e.empty:
            parts.append(pd.DataFrame({
                "anchor_id": e["event_id"].astype(str).to_numpy(),
                "parent_event_id": e["event_id"].astype(str).to_numpy(),
                "sample_type": np.repeat("EVENT", len(e)),
                "primitive": e["primitive"].astype(str).to_numpy(),
                "direction": pd.to_numeric(e["direction"], errors="raise").to_numpy(np.int8),
                "anchor_second": pd.to_numeric(e["detection_second"], errors="raise").to_numpy(np.int64),
                "tradeable_second": pd.to_numeric(e["tradeable_second"], errors="raise").to_numpy(np.int64),
            }))
    if not controls.empty:
        c = controls.loc[:, ["control_id", "parent_event_id", "primitive", "direction", "anchor_second", "tradeable_second"]].copy()
        if not c.empty:
            parts.append(pd.DataFrame({
                "anchor_id": c["control_id"].astype(str).to_numpy(),
                "parent_event_id": c["parent_event_id"].astype(str).to_numpy(),
                "sample_type": np.repeat("CONTROL", len(c)),
                "primitive": c["primitive"].astype(str).to_numpy(),
                "direction": pd.to_numeric(c["direction"], errors="raise").to_numpy(np.int8),
                "anchor_second": pd.to_numeric(c["anchor_second"], errors="raise").to_numpy(np.int64),
                "tradeable_second": pd.to_numeric(c["tradeable_second"], errors="raise").to_numpy(np.int64),
            }))
    if not parts:
        return pd.DataFrame()

    anchor_df = pd.concat(parts, ignore_index=True)
    anchor_df["entry_date"] = pd.to_datetime(anchor_df["tradeable_second"], unit="s", utc=True).dt.date
    columns = markout_exec_columns(cfg.notionals)
    horizons = np.asarray(cfg.markout_horizons, dtype=np.int64)
    notionals = np.asarray(cfg.notionals, dtype=float)
    H, N = len(horizons), len(notionals)
    out_frames: list[pd.DataFrame] = []

    for entry_date, group in anchor_df.groupby("entry_date", sort=True):
        group = group.reset_index(drop=True)
        execution = read_execution_for_days(execution_root, symbol, [entry_date, entry_date + timedelta(days=1)], columns)
        if execution.empty:
            continue
        execution = execution.sort_values("receive_second", kind="mergesort").drop_duplicates("receive_second", keep="last").reset_index(drop=True)
        ex_seconds = pd.to_numeric(execution["receive_second"], errors="raise").to_numpy(np.int64)
        ex_book_valid = _bool_array(execution, "book_valid")
        ex_mid = _numeric_array(execution, "mid_price")

        m = len(group)
        aidx = np.repeat(np.arange(m, dtype=np.int64), H * N)
        hidx = np.tile(np.repeat(np.arange(H, dtype=np.int64), N), m)
        nidx = np.tile(np.arange(N, dtype=np.int64), m * H)
        entry_secs_anchor = pd.to_numeric(group["tradeable_second"], errors="raise").to_numpy(np.int64)
        entry_pos_anchor, entry_found_anchor = _lookup_positions(ex_seconds, entry_secs_anchor)
        exit_secs = entry_secs_anchor[aidx] + horizons[hidx]
        exit_pos, exit_found = _lookup_positions(ex_seconds, exit_secs)

        entry_valid_anchor = entry_found_anchor & ex_book_valid[entry_pos_anchor]
        entry_mid_anchor = ex_mid[entry_pos_anchor].copy()
        entry_mid_anchor[~entry_found_anchor] = np.nan
        entry_valid = entry_valid_anchor[aidx]
        entry_mid = entry_mid_anchor[aidx]
        exit_valid = exit_found & ex_book_valid[exit_pos]
        exit_mid = ex_mid[exit_pos].copy()
        exit_mid[~exit_found] = np.nan
        directions_anchor = pd.to_numeric(group["direction"], errors="raise").to_numpy(np.int8)
        directions = directions_anchor[aidx]
        gross = np.full(len(aidx), np.nan, dtype=float)
        gross_ok = entry_valid & exit_valid & np.isfinite(entry_mid) & np.isfinite(exit_mid) & (entry_mid > 0) & (exit_mid > 0)
        gross[gross_ok] = (exit_mid[gross_ok] / entry_mid[gross_ok] - 1.0) * 10_000.0 * directions[gross_ok]

        buy_vwap = np.column_stack([_numeric_array(execution, f"buy_vwap_{int(round(n))}") for n in notionals])
        sell_vwap = np.column_stack([_numeric_array(execution, f"sell_vwap_{int(round(n))}") for n in notionals])
        buy_complete = np.column_stack([_bool_array(execution, f"buy_complete_{int(round(n))}") for n in notionals])
        sell_complete = np.column_stack([_bool_array(execution, f"sell_complete_{int(round(n))}") for n in notionals])
        buy_cross = np.column_stack([_numeric_array(execution, f"buy_crossing_cost_bps_{int(round(n))}") for n in notionals])
        sell_cross = np.column_stack([_numeric_array(execution, f"sell_crossing_cost_bps_{int(round(n))}") for n in notionals])

        ep = entry_pos_anchor[aidx]
        long = directions > 0
        entry_complete = np.where(long, buy_complete[ep, nidx], sell_complete[ep, nidx]) & entry_valid
        exit_complete = np.where(long, sell_complete[exit_pos, nidx], buy_complete[exit_pos, nidx]) & exit_valid
        entry_vwap = np.where(long, buy_vwap[ep, nidx], sell_vwap[ep, nidx])
        entry_cross = np.where(long, buy_cross[ep, nidx], sell_cross[ep, nidx])
        exit_cross = np.where(long, sell_cross[exit_pos, nidx], buy_cross[exit_pos, nidx])
        entry_vwap[~entry_found_anchor[aidx]] = np.nan
        entry_cross[~entry_found_anchor[aidx]] = np.nan
        exit_cross[~exit_found] = np.nan

        executable_to_mid = np.full(len(aidx), np.nan, dtype=float)
        exec_ok = entry_complete & exit_valid & np.isfinite(entry_vwap) & np.isfinite(exit_mid) & (entry_vwap > 0) & (exit_mid > 0)
        long_ok = exec_ok & long
        short_ok = exec_ok & ~long
        executable_to_mid[long_ok] = (exit_mid[long_ok] / entry_vwap[long_ok] - 1.0) * 10_000.0
        executable_to_mid[short_ok] = (entry_vwap[short_ok] / exit_mid[short_ok] - 1.0) * 10_000.0

        entry_adjusted = np.full(len(aidx), np.nan, dtype=float)
        ea_ok = np.isfinite(gross) & np.isfinite(entry_cross) & entry_complete
        entry_adjusted[ea_ok] = gross[ea_ok] - entry_cross[ea_ok] - cfg.taker_fee_bps
        estimated_roundtrip = np.full(len(aidx), np.nan, dtype=float)
        rt_ok = np.isfinite(gross) & np.isfinite(entry_cross) & np.isfinite(exit_cross) & entry_complete & exit_complete
        estimated_roundtrip[rt_ok] = gross[rt_ok] - entry_cross[rt_ok] - exit_cross[rt_ok] - 2.0 * cfg.taker_fee_bps

        out_frames.append(pd.DataFrame({
            "anchor_id": group["anchor_id"].astype(str).to_numpy()[aidx],
            "parent_event_id": group["parent_event_id"].astype(str).to_numpy()[aidx],
            "sample_type": group["sample_type"].astype(str).to_numpy()[aidx],
            "primitive": group["primitive"].astype(str).to_numpy()[aidx],
            "direction": directions.astype(np.int8),
            "anchor_second": pd.to_numeric(group["anchor_second"], errors="raise").to_numpy(np.int64)[aidx],
            "tradeable_second": entry_secs_anchor[aidx],
            "horizon_seconds": horizons[hidx],
            "quote_notional": notionals[nidx],
            "entry_book_valid": entry_valid,
            "exit_book_valid": exit_valid,
            "entry_mid_price": entry_mid,
            "exit_mid_price": exit_mid,
            "gross_mid_markout_bps": gross,
            "entry_side": np.where(long, "BUY", "SELL"),
            "exit_side": np.where(long, "SELL", "BUY"),
            "entry_complete": entry_complete,
            "exit_complete": exit_complete,
            "entry_crossing_cost_bps": entry_cross,
            "exit_crossing_cost_bps": exit_cross,
            "executable_entry_to_future_mid_bps": executable_to_mid,
            "entry_cost_adjusted_markout_bps": entry_adjusted,
            "estimated_roundtrip_net_bps": estimated_roundtrip,
            "taker_fee_bps_per_leg": np.full(len(aidx), cfg.taker_fee_bps, dtype=float),
            "roundtrip_cost_contract": np.repeat("MID_MARKOUT_MINUS_FIXED_QUOTE_ENTRY_EXIT_CROSSING_COSTS_MINUS_TWO_FEES", len(aidx)),
        }))

    return pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()


def standardized_mean_difference(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray([v for v in a if math.isfinite(float(v))], dtype=float)
    y = np.asarray([v for v in b if math.isfinite(float(v))], dtype=float)
    if len(x) < 2 or len(y) < 2:
        return math.nan
    pooled = math.sqrt((float(np.var(x, ddof=1)) + float(np.var(y, ddof=1))) / 2.0)
    return (float(np.mean(x)) - float(np.mean(y))) / pooled if pooled > 0 else 0.0


def bootstrap_mean_ci(values: Sequence[float], seed: int = 7, reps: int = 1000) -> tuple[float, float]:
    x = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
    if len(x) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=float)
    # Same non-parametric iid bootstrap, vectorized in bounded chunks to avoid
    # a Python loop per replicate without allocating an unbounded index matrix.
    chunk = 64
    for start in range(0, reps, chunk):
        stop = min(reps, start + chunk)
        idx = rng.integers(0, len(x), size=(stop - start, len(x)))
        means[start:stop] = x[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def event_count_reports(events: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    counts: list[dict[str, Any]] = []
    lags: list[dict[str, Any]] = []
    if events.empty:
        return counts, lags
    for (primitive, direction), g in events.groupby(["primitive", "direction"], dropna=False):
        detected = g.loc[g["detected"].fillna(False).astype(bool)]
        l = pd.to_numeric(detected["detection_lag_seconds"], errors="coerce").dropna()
        move = pd.to_numeric(detected["move_realized_before_detection_bps"], errors="coerce").dropna()
        counts.append({
            "primitive": primitive,
            "direction": int(direction),
            "events": len(g),
            "detected_events": len(detected),
            "detection_rate": len(detected) / len(g) if len(g) else math.nan,
        })
        lags.append({
            "primitive": primitive,
            "direction": int(direction),
            "detected_events": len(detected),
            "median_detection_lag_seconds": float(l.median()) if len(l) else math.nan,
            "p90_detection_lag_seconds": float(l.quantile(0.90)) if len(l) else math.nan,
            "p95_detection_lag_seconds": float(l.quantile(0.95)) if len(l) else math.nan,
            "median_move_realized_before_detection_bps": float(move.median()) if len(move) else math.nan,
            "p90_move_realized_before_detection_bps": float(move.quantile(0.90)) if len(move) else math.nan,
        })
    return counts, lags


def control_coverage_report(events: pd.DataFrame, controls: pd.DataFrame, cfg: Config) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if events.empty:
        return rows
    detected = events.loc[events["detected"].fillna(False).astype(bool) & events["direction"].ne(0)].copy()
    if detected.empty:
        return rows

    group_specs: list[tuple[str, Any]] = [("ALL", detected)]
    group_specs.extend((f"{primitive}|{int(direction)}", g) for (primitive, direction), g in detected.groupby(["primitive", "direction"], dropna=False))
    for label, eg in group_specs:
        event_ids = set(eg["event_id"].astype(str))
        cg = controls.loc[controls["parent_event_id"].astype(str).isin(event_ids)] if not controls.empty else pd.DataFrame()
        control_counts = cg.groupby("parent_event_id").size() if not cg.empty else pd.Series(dtype=int)
        with_any = sum(1 for eid in event_ids if int(control_counts.get(eid, 0)) > 0)
        with_full = sum(1 for eid in event_ids if int(control_counts.get(eid, 0)) >= cfg.controls_per_event)
        reuse_max = int(pd.to_numeric(cg.get("candidate_reuse_ordinal", pd.Series(dtype=float)), errors="coerce").max()) if not cg.empty else 0
        rows.append({
            "group": label,
            "detected_events": len(event_ids),
            "events_with_any_control": with_any,
            "events_with_full_controls": with_full,
            "event_control_coverage_rate": with_any / len(event_ids) if event_ids else math.nan,
            "full_control_coverage_rate": with_full / len(event_ids) if event_ids else math.nan,
            "controls": int(len(cg)),
            "mean_controls_per_event": len(cg) / len(event_ids) if event_ids else math.nan,
            "max_candidate_reuse_ordinal": reuse_max,
        })
    return rows


def control_balance_report(events: pd.DataFrame, controls: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if events.empty or controls.empty:
        return rows
    detected = events.loc[events["detected"].fillna(False).astype(bool) & events["direction"].ne(0)]
    for primitive, eg in detected.groupby("primitive"):
        cg = controls.loc[controls["primitive"] == primitive]
        for col in MATCH_COLUMNS:
            ec = f"anchor_{col}"
            rows.append({
                "primitive": primitive,
                "feature": col,
                "event_n": int(pd.to_numeric(eg[ec], errors="coerce").notna().sum()),
                "control_n": int(pd.to_numeric(cg[ec], errors="coerce").notna().sum()),
                "event_mean": float(pd.to_numeric(eg[ec], errors="coerce").mean()),
                "control_mean": float(pd.to_numeric(cg[ec], errors="coerce").mean()),
                "standardized_mean_difference": standardized_mean_difference(
                    pd.to_numeric(eg[ec], errors="coerce").dropna().tolist(),
                    pd.to_numeric(cg[ec], errors="coerce").dropna().tolist(),
                ),
            })
    return rows


def markout_summary_report(markouts: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if markouts.empty:
        return rows
    metrics = ("gross_mid_markout_bps", "entry_cost_adjusted_markout_bps", "estimated_roundtrip_net_bps")
    keys = ["primitive", "horizon_seconds", "quote_notional"]
    for group_key, g in markouts.groupby(keys, dropna=False, sort=True):
        primitive, horizon, notional = group_key
        ev = g.loc[g["sample_type"] == "EVENT"]
        co = g.loc[g["sample_type"] == "CONTROL"]
        for metric in metrics:
            ev_values = pd.to_numeric(ev[metric], errors="coerce")
            co_values = pd.to_numeric(co[metric], errors="coerce")
            if not co.empty:
                parent_means = (
                    co.assign(_metric=co_values)
                    .groupby("parent_event_id", dropna=False)["_metric"]
                    .mean()
                )
                matched_control = ev["parent_event_id"].map(parent_means)
                paired_series = ev_values - pd.to_numeric(matched_control, errors="coerce")
                paired = paired_series[np.isfinite(paired_series.to_numpy(float))].to_numpy(float)
            else:
                paired = np.array([], dtype=float)
            lo, hi = bootstrap_mean_ci(paired, seed=7 + int(horizon) + int(round(float(notional))))
            rows.append({
                "primitive": primitive,
                "horizon_seconds": int(horizon),
                "quote_notional": float(notional),
                "metric": metric,
                "event_n": int(ev_values.notna().sum()),
                "control_n": int(co_values.notna().sum()),
                "event_mean": float(ev_values.mean()) if ev_values.notna().any() else math.nan,
                "event_median": float(ev_values.median()) if ev_values.notna().any() else math.nan,
                "control_mean": float(co_values.mean()) if co_values.notna().any() else math.nan,
                "control_median": float(co_values.median()) if co_values.notna().any() else math.nan,
                "paired_event_minus_control_n": int(len(paired)),
                "paired_mean_difference": float(np.mean(paired)) if len(paired) else math.nan,
                "paired_median_difference": float(np.median(paired)) if len(paired) else math.nan,
                "paired_mean_bootstrap_ci95_low": lo,
                "paired_mean_bootstrap_ci95_high": hi,
            })
    return rows


def invariant_report(
    primitive_paths: Sequence[Path],
    events: pd.DataFrame,
    controls: pd.DataFrame,
    markouts: pd.DataFrame,
    cfg: Config,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    def add(name: str, ok: bool, observed: Any, expected: Any, detail: str = "", warn_only: bool = False) -> None:
        status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
        rows.append({"check": name, "status": status, "observed": observed, "expected": expected, "detail": detail})

    add("primitive_partitions_exist", len(primitive_paths) > 0, len(primitive_paths), ">0")
    detected = pd.DataFrame()
    if not events.empty:
        add("event_ids_unique", not events["event_id"].duplicated().any(), int(events["event_id"].duplicated().sum()), 0)
        detected = events.loc[events["detected"].fillna(False).astype(bool) & events["direction"].ne(0)].copy()
        bad_lag = (pd.to_numeric(detected["detection_lag_seconds"], errors="coerce") < 0).sum()
        add("detection_lag_nonnegative", int(bad_lag) == 0, int(bad_lag), 0)
        bad_trade = (
            pd.to_numeric(detected["tradeable_second"], errors="coerce")
            != pd.to_numeric(detected["detection_second"], errors="coerce") + 1
        ).sum()
        add("tradeable_is_detection_plus_one", int(bad_trade) == 0, int(bad_trade), 0)

    if not detected.empty:
        matched_parent_ids = set(controls["parent_event_id"].astype(str)) if not controls.empty else set()
        detected_ids = set(detected["event_id"].astype(str))
        coverage = len(detected_ids & matched_parent_ids) / len(detected_ids) if detected_ids else math.nan
        add("matched_controls_nonzero", len(controls) > 0, len(controls), ">0", "Detected directional events require a usable matched-control layer")
        add(
            "detected_event_control_coverage_at_least_80pct",
            math.isfinite(coverage) and coverage >= 0.80,
            coverage,
            ">=0.80",
            "Coverage below target is a research-utility warning, not a mechanical pipeline failure",
            warn_only=True,
        )

    if not controls.empty:
        add("control_ids_unique", not controls["control_id"].duplicated().any(), int(controls["control_id"].duplicated().sum()), 0)
        bad = (pd.to_numeric(controls["tradeable_second"], errors="coerce") != pd.to_numeric(controls["anchor_second"], errors="coerce") + 1).sum()
        add("control_tradeable_is_anchor_plus_one", int(bad) == 0, int(bad), 0)
        max_reuse = int(pd.to_numeric(controls.get("candidate_reuse_ordinal", pd.Series(dtype=float)), errors="coerce").max())
        add("control_reuse_within_cap", max_reuse <= cfg.control_max_reuse, max_reuse, f"<={cfg.control_max_reuse}")
        bad_cooldown = (pd.to_numeric(controls.get("seconds_since_same_primitive_detect"), errors="coerce") <= cfg.control_trailing_cooldown_seconds).sum()
        add("control_same_primitive_trailing_cooldown", int(bad_cooldown) == 0, int(bad_cooldown), 0)

    if not markouts.empty:
        bad_h = ~markouts["horizon_seconds"].isin(cfg.markout_horizons)
        add("markout_horizons_frozen", int(bad_h.sum()) == 0, int(bad_h.sum()), 0)
        bad_n = ~markouts["quote_notional"].isin(cfg.notionals)
        add("markout_notionals_frozen", int(bad_n.sum()) == 0, int(bad_n.sum()), 0)
    return rows


def build_findings(
    summary: Mapping[str, Any],
    counts: Sequence[Mapping[str, Any]],
    lags: Sequence[Mapping[str, Any]],
    balance: Sequence[Mapping[str, Any]],
    coverage: Sequence[Mapping[str, Any]],
    checks: Sequence[Mapping[str, Any]],
) -> str:
    failed = [r for r in checks if r.get("status") == "FAIL"]
    lines = [
        "# Phase 6B — Causal Liquidity-Event Primitives",
        "",
        f"Builder version: `{BUILDER_VERSION}`",
        "",
        "Phase 6B is an event-mechanics and matched-control research layer. It does **not** make an alpha or strategy claim.",
        "",
        "## Causal contract",
        "",
        "- Primitive state is measured in receive second `t`.",
        "- Detection is only available after `t` closes.",
        "- Earliest hypothetical trade is `t+1`.",
        "- The weaker onset threshold and stricter detector are retained separately.",
        "- The move already realized before detection is explicitly measured.",
        "- forceOrder size is selected/censored and is never interpreted as total liquidation volume.",
        "- Open interest is context/regime information only, not a trigger.",
        "- Controls are primitive-specific: same-family neutral at the anchor, trailing-only same-family detection cooldown, and no global future calmness requirement.",
        "",
        "## Build summary",
        "",
        f"- Dates: **{summary.get('dates_completed', 0)} / {summary.get('dates_requested', 0)}**",
        f"- Primitive rows: **{int(summary.get('primitive_rows', 0)):,}**",
        f"- Events: **{int(summary.get('events', 0)):,}**",
        f"- Detected events: **{int(summary.get('detected_events', 0)):,}**",
        f"- Controls: **{int(summary.get('controls', 0)):,}**",
        f"- Markout rows: **{int(summary.get('markout_rows', 0)):,}**",
        f"- Failed invariants: **{len(failed)}**",
        "",
        "## Event families",
        "",
    ]
    for row in counts:
        lines.append(
            f"- {row['primitive']} direction={row['direction']}: {int(row['events']):,} onset episodes; "
            f"{int(row['detected_events']):,} detected ({float(row['detection_rate']):.1%})."
        )
    lines += ["", "## Detection lag", ""]
    for row in lags:
        med = row.get("median_detection_lag_seconds")
        move = row.get("median_move_realized_before_detection_bps")
        lines.append(
            f"- {row['primitive']} direction={row['direction']}: median lag={med if math.isfinite(safe_float(med)) else 'NA'}s; "
            f"median directional move before detection={move if math.isfinite(safe_float(move)) else 'NA'} bps."
        )
    if coverage:
        overall = next((r for r in coverage if r.get("group") == "ALL"), None)
        if overall is not None:
            lines += [
                "", "## Control coverage", "",
                f"- Detected events with >=1 control: **{safe_float(overall.get('event_control_coverage_rate')):.1%}**.",
                f"- Detected events with all requested controls: **{safe_float(overall.get('full_control_coverage_rate')):.1%}**.",
                f"- Mean controls per detected event: **{safe_float(overall.get('mean_controls_per_event')):.2f}**.",
            ]
    if balance:
        max_abs = max((abs(safe_float(r.get("standardized_mean_difference"))) for r in balance if math.isfinite(safe_float(r.get("standardized_mean_difference")))), default=math.nan)
        lines += ["", "## Control matching", "", f"- Maximum absolute matching-feature SMD: **{max_abs:.3f}**" if math.isfinite(max_abs) else "- Control-balance SMD unavailable."]
    lines += [
        "",
        "## Markout interpretation",
        "",
        "`gross_mid_markout_bps` is mid-to-mid. `executable_entry_to_future_mid_bps` includes exact Phase-6A fixed-notional entry crossing. "
        "`estimated_roundtrip_net_bps` subtracts fixed-quote entry/exit crossing-cost estimates and two configured taker fees; it is not an exact same-base-quantity round trip.",
        "",
        "Matched-control differences are descriptive event-study statistics. They are not evidence of a deployable edge until robustness, detection-lag, interaction, and out-of-sample tests are completed.",
    ]
    return "\n".join(lines) + "\n"


def run_self_test() -> None:
    # Exact baseline readiness across a gap.
    seconds = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)
    ready = exact_prior_window_ready(seconds, 3, 3)
    if not ready[3] or ready[7]:
        raise AssertionError(f"exact prior-window readiness failed: {ready}")

    # Causal z excludes current row.
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 10.0])
    z, _ = causal_zscore(s, np.arange(5, dtype=np.int64), 3, 3)
    if not math.isfinite(float(z.iloc[-1])) or float(z.iloc[-1]) <= 0:
        raise AssertionError(z)

    # Event grouping and detection delay.
    groups = cluster_seconds([100, 101, 104, 110], 3)
    if groups[0][0] != 100 or groups[0][1] != 104 or groups[1][0] != 110:
        raise AssertionError(groups)

    # Directional markout sign.
    if not math.isclose(bps_return(101.0, 100.0, +1), 100.0, rel_tol=1e-12):
        raise AssertionError("long bps")
    if not math.isclose(bps_return(99.0, 100.0, -1), 100.0, rel_tol=1e-12):
        raise AssertionError("short bps")

    # Embargo marks nearest event on either side.
    mask = near_any_event_mask(np.array([90, 95, 100, 105, 110]), np.array([100]), 5)
    expected = np.array([False, True, True, True, False])
    if not np.array_equal(mask, expected):
        raise AssertionError((mask, expected))
    imask = near_any_interval_mask(
        np.array([90, 94, 95, 100, 109, 110, 111]),
        np.array([[100, 105]], dtype=np.int64),
        5,
    )
    iexpected = np.array([False, False, True, True, True, True, False])
    if not np.array_equal(imask, iexpected):
        raise AssertionError((imask, iexpected))

    # v1.0.2 optimized implementation of the frozen v1.0.1 control contract: unrelated primitive activity must not globally
    # embargo a candidate; same-primitive recent detections and the parent-event
    # neighborhood must still be excluded.
    class _Cfg:
        control_trailing_cooldown_seconds = 10
        control_embargo_seconds = 60
        controls_per_event = 2
        control_max_reuse = 2

    event = pd.DataFrame([{
        "event_id": "EV-T", "detected": True, "direction": 1,
        "primitive": "DEPTH_WITHDRAWAL", "direction_label": "UP",
        "detection_second": 1000, "anchor_receive_date": "1970-01-01",
        "onset_date": "1970-01-01", "anchor_utc_hour": 0,
        **{f"anchor_{c}": 1.0 for c in MATCH_COLUMNS},
    }])
    candidate_rows = []
    for sec, since in [(100, 100.0), (200, 100.0), (300, 5.0), (950, 100.0), (1100, 100.0)]:
        row = {
            "receive_second": sec, "receive_date": "1970-01-01", "utc_hour": 0,
            "control_neutral_depthwithdrawal": True,
            "seconds_since_depthwithdrawal_detect": since,
            # Deliberately mark unrelated global primitive activity true.
            "primitive_any_onset": True, "primitive_any_detect": True,
        }
        for c in MATCH_COLUMNS:
            row[c] = 1.0
        candidate_rows.append(row)
    test_controls = build_controls(event, pd.DataFrame(candidate_rows), _Cfg(), "BTCUSDT")
    if len(test_controls) != 2:
        raise AssertionError(f"primitive-specific control matching failed: {test_controls}")
    chosen = set(test_controls["anchor_second"].astype(int))
    if 300 in chosen or 950 in chosen:
        raise AssertionError(f"cooldown/parent embargo failed: {chosen}")

    # v1.0.3 chronological sharding must reproduce one-shot candidate reuse.
    event2 = event.copy()
    event2.loc[:, "event_id"] = "EV-T2"
    event2.loc[:, "detection_second"] = 2000
    event2.loc[:, "anchor_receive_date"] = "1970-01-01"
    all_events = pd.concat([event, event2], ignore_index=True)
    one_shot = build_controls(all_events, pd.DataFrame(candidate_rows), _Cfg(), "BTCUSDT")
    pdata = prepare_control_matcher(pd.DataFrame(candidate_rows), _Cfg())
    shard_a = build_controls_prepared(event, pdata, _Cfg(), "BTCUSDT")
    shard_b = build_controls_prepared(event2, pdata, _Cfg(), "BTCUSDT")
    sharded = pd.concat([shard_a, shard_b], ignore_index=True)
    cols = ["parent_event_id", "anchor_second", "control_ordinal", "candidate_reuse_ordinal", "match_tier"]
    if one_shot.loc[:, cols].reset_index(drop=True).to_dict("records") != sharded.loc[:, cols].reset_index(drop=True).to_dict("records"):
        raise AssertionError("partitioned prepared-control matching diverged from one-shot matching")

    # Partial-selection kernel must reproduce stable (distance, global-index) ordering.
    dtest = np.array([3.0, 1.0, 1.0, 2.0, np.inf])
    itest = np.array([9, 7, 5, 3, 1], dtype=np.int64)
    got = _stable_k_smallest_positions(dtest, itest, 3)
    if not np.array_equal(got, np.array([2, 1, 3], dtype=np.int64)):
        raise AssertionError(f"stable partial selection failed: {got}")

    print("SELF-TEST PASS")
    print(
        "Validated exact causal baseline readiness, current-row-excluded z-scores, "
        "event bridging, directional markouts, vectorized primitive-specific causal controls, "
        "parent-event embargoes, bounded reuse, chronological partitioned control parity, and partial nearest-neighbor selection."
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Phase-6B causal liquidity-event primitives and matched executable markouts.")
    p.add_argument("--execution-root", type=Path, default=DEFAULT_EXECUTION_ROOT)
    p.add_argument("--flow-root", type=Path, default=DEFAULT_FLOW_ROOT)
    p.add_argument("--core-root", type=Path, default=DEFAULT_CORE_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--start-date")
    p.add_argument("--end-date")
    p.add_argument("--baseline-seconds", type=int, default=DEFAULT_BASELINE_SECONDS)
    p.add_argument("--baseline-min-seconds", type=int, default=DEFAULT_BASELINE_MIN_SECONDS)
    p.add_argument("--event-bridge-seconds", type=int, default=DEFAULT_EVENT_BRIDGE_SECONDS)
    p.add_argument("--max-detection-lag-seconds", type=int, default=DEFAULT_MAX_DETECTION_LAG_SECONDS)
    p.add_argument("--control-embargo-seconds", type=int, default=DEFAULT_CONTROL_EMBARGO_SECONDS,
                   help="Exclude candidate anchors within +/- this many seconds of their parent event only")
    p.add_argument("--control-trailing-cooldown-seconds", type=int, default=DEFAULT_CONTROL_TRAILING_COOLDOWN_SECONDS,
                   help="Require this many seconds since the latest same-primitive strict detection; trailing-only/causal")
    p.add_argument("--control-cadence-seconds", type=int, default=DEFAULT_CONTROL_CADENCE_SECONDS)
    p.add_argument("--controls-per-event", type=int, default=DEFAULT_CONTROLS_PER_EVENT)
    p.add_argument("--control-max-reuse", type=int, default=DEFAULT_CONTROL_MAX_REUSE,
                   help="Maximum deterministic reuse count for one candidate anchor within a primitive family")
    p.add_argument("--markout-horizons", default=",".join(map(str, DEFAULT_MARKOUT_HORIZONS)))
    p.add_argument("--notionals", default=",".join(str(int(x)) for x in DEFAULT_NOTIONALS))
    p.add_argument("--taker-fee-bps", type=float, default=0.0)
    p.add_argument("--depth-onset-frac", type=float, default=DEFAULT_DEPTH_ONSET_FRAC)
    p.add_argument("--depth-detect-frac", type=float, default=DEFAULT_DEPTH_DETECT_FRAC)
    p.add_argument("--depth-detect-z", type=float, default=DEFAULT_DEPTH_DETECT_Z)
    p.add_argument("--flow-onset-imbalance", type=float, default=DEFAULT_FLOW_ONSET_IMBALANCE)
    p.add_argument("--flow-detect-imbalance", type=float, default=DEFAULT_FLOW_DETECT_IMBALANCE)
    p.add_argument("--flow-detect-z", type=float, default=DEFAULT_FLOW_DETECT_Z)
    p.add_argument("--flow-volume-z", type=float, default=DEFAULT_FLOW_VOLUME_Z)
    p.add_argument("--dislocation-onset-bps", type=float, default=DEFAULT_DISLOCATION_ONSET_BPS)
    p.add_argument("--dislocation-detect-bps", type=float, default=DEFAULT_DISLOCATION_DETECT_BPS)
    p.add_argument("--dislocation-detect-z", type=float, default=DEFAULT_DISLOCATION_DETECT_Z)
    p.add_argument("--force", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    require_arrow()

    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    if start and end and start > end:
        raise SystemExit("--start-date must be <= --end-date")
    if args.baseline_min_seconds <= 0 or args.baseline_min_seconds > args.baseline_seconds:
        raise SystemExit("baseline min seconds must be >0 and <= baseline seconds")
    if args.control_embargo_seconds < 0 or args.control_trailing_cooldown_seconds < 0:
        raise SystemExit("control embargo/cooldown seconds must be >= 0")
    if args.control_cadence_seconds <= 0 or args.controls_per_event <= 0 or args.control_max_reuse <= 0:
        raise SystemExit("control cadence, controls/event and max reuse must be > 0")

    cfg = Config(
        baseline_seconds=args.baseline_seconds,
        baseline_min_seconds=args.baseline_min_seconds,
        event_bridge_seconds=args.event_bridge_seconds,
        max_detection_lag_seconds=args.max_detection_lag_seconds,
        control_embargo_seconds=args.control_embargo_seconds,
        control_trailing_cooldown_seconds=args.control_trailing_cooldown_seconds,
        control_cadence_seconds=args.control_cadence_seconds,
        controls_per_event=args.controls_per_event,
        control_max_reuse=args.control_max_reuse,
        markout_horizons=parse_int_tuple(args.markout_horizons, "--markout-horizons"),
        notionals=parse_float_tuple(args.notionals, "--notionals"),
        taker_fee_bps=float(args.taker_fee_bps),
        depth_onset_frac=float(args.depth_onset_frac),
        depth_detect_frac=float(args.depth_detect_frac),
        depth_detect_z=float(args.depth_detect_z),
        flow_onset_imbalance=float(args.flow_onset_imbalance),
        flow_detect_imbalance=float(args.flow_detect_imbalance),
        flow_detect_z=float(args.flow_detect_z),
        flow_volume_z=float(args.flow_volume_z),
        dislocation_onset_bps=float(args.dislocation_onset_bps),
        dislocation_detect_bps=float(args.dislocation_detect_bps),
        dislocation_detect_z=float(args.dislocation_detect_z),
    )

    days = discover_execution_days(args.execution_root, args.symbol, start, end)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    metadata_root = args.output_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    state = BuildState(metadata_root / "build_state.sqlite")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]

    config_payload = {
        "builder_version": BUILDER_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "output_schema_name": OUTPUT_SCHEMA_NAME,
        "run_id": run_id,
        "symbol": args.symbol,
        "start_date": days[0].isoformat(),
        "end_date": days[-1].isoformat(),
        "config": asdict(cfg),
        "causal_contract": "primitive second t available after t closes; earliest tradeable second=t+1",
        "force_order_size_contract": "selected/censored descriptive observation; not total liquidation volume",
        "open_interest_contract": "regime/context only; not a Phase-6B event trigger",
        "control_contract": "primitive-specific neutral anchor + trailing-only same-primitive detection cooldown + parent-event-only embargo + bounded deterministic reuse",
        "performance_contract": "v1.0.2 scientific semantics; existing primitive shards reused; global candidate scan + chronological partitioned/resumable event/control/markout aggregation",
        "roundtrip_cost_contract": "gross mid markout minus fixed-quote Phase-6A entry/exit crossing costs minus two configured fees; not exact same-base roundtrip",
        "created_at_utc": utc_now_iso(),
    }
    atomic_write_json(metadata_root / "run_config.json", config_payload)

    print("Building Phase-6B causal liquidity-event primitives...")
    print(f"Builder version    : {BUILDER_VERSION}")
    print(f"Run ID             : {run_id}")
    print(f"Symbol             : {args.symbol}")
    print(f"Dates              : {len(days)} ({days[0]} to {days[-1]})")
    print(f"Execution root     : {args.execution_root}")
    print(f"Flow root          : {args.flow_root}")
    print(f"Core root          : {args.core_root}")
    print(f"Output root        : {args.output_root}")
    print(f"Baseline           : {cfg.baseline_seconds}s causal, current second excluded")
    print(
        f"Controls/event     : {cfg.controls_per_event} "
        f"(cadence={cfg.control_cadence_seconds}s, parent embargo=±{cfg.control_embargo_seconds}s, "
        f"same-primitive trailing cooldown={cfg.control_trailing_cooldown_seconds}s, max reuse={cfg.control_max_reuse})"
    )
    print(f"Markout horizons   : {', '.join(map(str, cfg.markout_horizons))}s")
    print(f"Quote notionals    : {', '.join(f'${x:,.0f}' for x in cfg.notionals)}")
    print("forceOrder size    : selected/censored; NOT total liquidation volume")
    print("Tradeable timing   : detection second + 1")
    print("Performance        : partitioned/resumable aggregation; global candidate universe preserved", flush=True)

    primitive_paths: list[Path] = []
    daily_quality: list[dict[str, Any]] = []
    failed_days: list[dict[str, Any]] = []
    reused = built = 0
    started = time.perf_counter()

    try:
        for i, day in enumerate(days, 1):
            out = primitive_output_path(args.output_root, args.symbol, day)
            # Signature includes current + previous-day context because causal
            # baseline z-scores depend on that history.
            context_files: list[Path] = []
            for d in (day - timedelta(days=1), day):
                context_files.extend(partition_files(args.execution_root, args.symbol, d))
                context_files.extend(partition_files(args.flow_root, args.symbol, d))
                context_files.extend(partition_files(args.core_root, args.symbol, d))
            if not partition_files(args.flow_root, args.symbol, day) or not partition_files(args.core_root, args.symbol, day):
                failed_days.append({"receive_date": day.isoformat(), "error": "missing current flow/core partition"})
                print(f"[{i:02d}/{len(days):02d}] {day}: FAILED missing flow/core partition")
                continue
            sig = sha256_text(compact_json({
                "files": file_signature(context_files),
                "config": cfg.signature(),
                "version": PRIMITIVE_PARTITION_BUILD_VERSION,
            }))
            if not args.force and state.reusable(args.symbol, day, sig, out):
                primitive_paths.append(out)
                reused += 1
                pf = pq.ParquetFile(out)
                try:
                    daily_quality.append({"receive_date": day.isoformat(), "rows": int(pf.metadata.num_rows), "status": "REUSED"})
                finally:
                    pf.close()
                print(f"[{i:02d}/{len(days):02d}] {day}: COMPLETED_REUSED")
                continue
            t0 = time.perf_counter()
            try:
                context, actual_files = read_day_context(args.execution_root, args.flow_root, args.core_root, args.symbol, day)
                frame, summary = build_primitive_frame(context, day, cfg)
                missing = [c for c in PRIMITIVE_OUTPUT_COLUMNS if c not in frame.columns]
                if missing:
                    raise RuntimeError(f"Primitive output columns missing: {missing}")
                frame = frame.loc[:, list(PRIMITIVE_OUTPUT_COLUMNS)].copy()
                if frame["receive_second"].duplicated().any():
                    raise RuntimeError("primitive receive_second duplicates")
                metadata = {
                    "builder_version": BUILDER_VERSION,
                    "schema_version": OUTPUT_SCHEMA_VERSION,
                    "symbol": args.symbol,
                    "receive_date": day.isoformat(),
                    "input_signature": sig,
                    "config_signature": cfg.signature(),
                    "availability_rule": "available_after_second=receive_second+1",
                }
                atomic_write_parquet(out, frame, metadata)
                summary["build_seconds"] = time.perf_counter() - t0
                state.put(args.symbol, day, sig, "COMPLETED", out, summary)
                primitive_paths.append(out)
                daily_quality.append(summary)
                built += 1
                print(
                    f"[{i:02d}/{len(days):02d}] {day}: COMPLETED "
                    f"({len(frame):,} rows, ready={summary['input_ready_rows']:,}, detects={summary['any_detect_rows']:,}, build={summary['build_seconds']:.1f}s)"
                )
            except Exception as exc:
                state.put(args.symbol, day, sig, "FAILED", out, {"error": repr(exc)})
                failed_days.append({"receive_date": day.isoformat(), "error": repr(exc)})
                print(f"[{i:02d}/{len(days):02d}] {day}: FAILED {exc}")

        if not primitive_paths:
            raise RuntimeError("No Phase-6B primitive partitions are available")

        print("Building partitioned event/control/markout shards...", flush=True)
        stage_seconds: dict[str, float] = {}
        _t = time.perf_counter()
        candidates, quality_from_outputs = collect_candidates_and_quality(primitive_paths, cfg)
        stage_seconds["build_global_candidate_universe"] = time.perf_counter() - _t
        print(f"  global candidates      : {len(candidates):,} in {stage_seconds['build_global_candidate_universe']:.2f}s", flush=True)
        _t = time.perf_counter()
        primitive_data = prepare_control_matcher(candidates, cfg)
        stage_seconds["prepare_global_control_matcher"] = time.perf_counter() - _t
        print(f"  control matcher        : {stage_seconds['prepare_global_control_matcher']:.2f}s", flush=True)

        primitive_by_day = {date.fromisoformat(p.parent.name.split("=",1)[1]): p for p in primitive_paths}
        candidate_universe_signature = sha256_text(compact_json({
            "primitive_files": file_signature(primitive_paths),
            "config": cfg.signature(),
            "builder_version": BUILDER_VERSION,
            "control_contract": "primitive_specific_causal_v1_global_candidate_universe",
        }))
        agg_state = AggregationState(metadata_root / "aggregation_state.sqlite")
        agg_summaries: list[dict[str, Any]] = []
        rebuilt_suffix = False
        reusable_control_prefix: list[Path] = []
        agg_failed_days: list[dict[str, Any]] = []
        agg_started = time.perf_counter()
        try:
            # Reuse only a contiguous prefix: candidate reuse affects every later day.
            prefix_len = 0
            for day in days:
                pcur = primitive_by_day.get(day)
                if pcur is None:
                    break
                context_paths = [p for d in (day - timedelta(days=1), day, day + timedelta(days=1)) if (p := primitive_by_day.get(d)) is not None]
                exec_files = list(partition_files(args.execution_root, args.symbol, day)) + list(partition_files(args.execution_root, args.symbol, day + timedelta(days=1)))
                agg_sig = sha256_text(compact_json({
                    "candidate_universe": candidate_universe_signature,
                    "event_context": file_signature(context_paths),
                    "execution": file_signature(exec_files) if exec_files else "",
                    "config": cfg.signature(), "version": BUILDER_VERSION,
                }))
                paths = [aggregation_output_path(args.output_root, ds, args.symbol, day) for ds in ("events","controls","markouts")]
                if not args.force and agg_state.reusable(args.symbol, day, agg_sig, paths):
                    prefix_len += 1
                    reusable_control_prefix.append(paths[1])
                    agg_summaries.append(agg_state.summary(args.symbol, day))
                else:
                    break
            if prefix_len:
                print(f"  aggregation prefix     : {prefix_len}/{len(days)} days reusable", flush=True)
                restore_control_usage(primitive_data, reusable_control_prefix)
            else:
                print("  aggregation prefix     : 0 days reusable", flush=True)

            for i, day in enumerate(days, 1):
                if i <= prefix_len:
                    sm = agg_summaries[i-1] if i-1 < len(agg_summaries) else {}
                    print(f"  [{i:02d}/{len(days):02d}] {day}: AGGREGATION_REUSED events={int(sm.get('events',0)):,} controls={int(sm.get('controls',0)):,}", flush=True)
                    continue
                pcur = primitive_by_day.get(day)
                if pcur is None:
                    agg_failed_days.append({"receive_date": day.isoformat(), "error": "missing primitive partition"})
                    print(f"  [{i:02d}/{len(days):02d}] {day}: AGGREGATION_FAILED missing primitive partition", flush=True)
                    continue
                context_paths = [p for d in (day - timedelta(days=1), day, day + timedelta(days=1)) if (p := primitive_by_day.get(d)) is not None]
                exec_files = list(partition_files(args.execution_root, args.symbol, day)) + list(partition_files(args.execution_root, args.symbol, day + timedelta(days=1)))
                agg_sig = sha256_text(compact_json({
                    "candidate_universe": candidate_universe_signature,
                    "event_context": file_signature(context_paths),
                    "execution": file_signature(exec_files) if exec_files else "",
                    "config": cfg.signature(), "version": BUILDER_VERSION,
                }))
                event_path = aggregation_output_path(args.output_root, "events", args.symbol, day)
                control_path = aggregation_output_path(args.output_root, "controls", args.symbol, day)
                markout_path = aggregation_output_path(args.output_root, "markouts", args.symbol, day)
                paths = [event_path, control_path, markout_path]
                tday = time.perf_counter()
                try:
                    th = time.perf_counter()
                    hit_frames = collect_hits_only(context_paths)
                    events_context = build_events(hit_frames, cfg, args.symbol)
                    events = events_context.loc[events_context["onset_date"].astype(str) == day.isoformat()].copy().reset_index(drop=True) if not events_context.empty else pd.DataFrame()
                    event_seconds = time.perf_counter() - th
                    tc = time.perf_counter()
                    controls = build_controls_prepared(events, primitive_data, cfg, args.symbol)
                    control_seconds = time.perf_counter() - tc
                    tm = time.perf_counter()
                    markouts = build_markouts(events, controls, args.execution_root, args.symbol, cfg)
                    markout_seconds = time.perf_counter() - tm
                    common_meta = {
                        "builder_version": BUILDER_VERSION, "schema_version": OUTPUT_SCHEMA_VERSION,
                        "config_signature": cfg.signature(), "run_id": run_id,
                        "receive_date": day.isoformat(),
                        "candidate_universe_signature": candidate_universe_signature,
                        "control_contract": "primitive_specific_causal_v1_global_candidate_universe",
                    }
                    atomic_write_parquet(event_path, events, {**common_meta, "dataset": "events"})
                    atomic_write_parquet(control_path, controls, {**common_meta, "dataset": "controls"})
                    atomic_write_parquet(markout_path, markouts, {**common_meta, "dataset": "markouts"})

                    # Daily reports are bounded and make the final report merge cheap.
                    counts, lags = event_count_reports(events)
                    coverage = control_coverage_report(events, controls, cfg)
                    balance = control_balance_report(events, controls)
                    markout_summary = markout_summary_report(markouts)
                    checks = invariant_report([pcur], events, controls, markouts, cfg)
                    rdir = aggregation_report_dir(args.report_root, day)
                    atomic_write_csv(rdir / "event_counts.csv", counts)
                    atomic_write_csv(rdir / "detection_lag.csv", lags)
                    atomic_write_csv(rdir / "control_coverage.csv", coverage)
                    atomic_write_csv(rdir / "control_balance.csv", balance)
                    atomic_write_csv(rdir / "markout_summary.csv", markout_summary)
                    atomic_write_csv(rdir / "invariants.csv", checks)
                    perf = [
                        {"stage":"build_events","seconds":event_seconds},
                        {"stage":"build_controls","seconds":control_seconds},
                        {"stage":"build_markouts","seconds":markout_seconds},
                    ]
                    atomic_write_csv(rdir / "performance.csv", perf)
                    detected_n = int(events["detected"].fillna(False).sum()) if not events.empty else 0
                    directional_n = int((events["detected"].fillna(False) & events["direction"].ne(0)).sum()) if not events.empty else 0
                    failed_daily = sum(1 for r in checks if r.get("status") == "FAIL")
                    day_summary = {
                        "receive_date": day.isoformat(), "events": int(len(events)), "detected_events": detected_n,
                        "directional_detected_events": directional_n, "controls": int(len(controls)),
                        "markout_rows": int(len(markouts)), "failed_invariants": int(failed_daily),
                        "event_seconds": event_seconds, "control_seconds": control_seconds,
                        "markout_seconds": markout_seconds, "runtime_seconds": time.perf_counter() - tday,
                    }
                    agg_state.put(args.symbol, day, agg_sig, "COMPLETED", paths, day_summary)
                    agg_summaries.append(day_summary)
                    print(
                        f"  [{i:02d}/{len(days):02d}] {day}: AGGREGATED "
                        f"events={len(events):,} detected={detected_n:,} controls={len(controls):,} "
                        f"markouts={len(markouts):,} runtime={day_summary['runtime_seconds']:.1f}s", flush=True
                    )
                except Exception as exc:
                    agg_state.put(args.symbol, day, agg_sig, "FAILED", paths, {"error": repr(exc)})
                    agg_failed_days.append({"receive_date": day.isoformat(), "error": repr(exc)})
                    print(f"  [{i:02d}/{len(days):02d}] {day}: AGGREGATION_FAILED {exc}", flush=True)
                    # Stop here: later candidate reuse would no longer be equivalent.
                    break
        finally:
            agg_state.close()
        stage_seconds["partitioned_aggregation"] = time.perf_counter() - agg_started

        # Concatenate bounded daily report shards; no monolithic raw dataframe.
        completed_days = []
        for day in days:
            ep = aggregation_output_path(args.output_root, "events", args.symbol, day)
            cp = aggregation_output_path(args.output_root, "controls", args.symbol, day)
            mp = aggregation_output_path(args.output_root, "markouts", args.symbol, day)
            if ep.exists() and cp.exists() and mp.exists():
                completed_days.append(day)
            else:
                break
        counts = concat_daily_reports(args.report_root, "event_counts.csv", completed_days)
        lags = concat_daily_reports(args.report_root, "detection_lag.csv", completed_days)
        coverage = concat_daily_reports(args.report_root, "control_coverage.csv", completed_days)
        balance = concat_daily_reports(args.report_root, "control_balance.csv", completed_days)
        markout_summary = concat_daily_reports(args.report_root, "markout_summary.csv", completed_days)
        checks = concat_daily_reports(args.report_root, "invariants.csv", completed_days)
        performance_rows = concat_daily_reports(args.report_root, "performance.csv", completed_days)
        atomic_write_csv(args.report_root / "liquidity_primitive_event_counts.csv", counts)
        atomic_write_csv(args.report_root / "liquidity_primitive_detection_lag.csv", lags)
        atomic_write_csv(args.report_root / "liquidity_primitive_control_coverage.csv", coverage)
        atomic_write_csv(args.report_root / "liquidity_primitive_control_balance.csv", balance)
        atomic_write_csv(args.report_root / "liquidity_primitive_markout_summary.csv", markout_summary)
        atomic_write_csv(args.report_root / "liquidity_primitive_data_quality.csv", quality_from_outputs)
        atomic_write_csv(args.report_root / "liquidity_primitive_performance.csv", performance_rows)
        atomic_write_csv(args.report_root / "liquidity_primitive_invariants.csv", checks)

        # Reload tiny aggregation summaries from state for a deterministic total.
        state2 = AggregationState(metadata_root / "aggregation_state.sqlite")
        try:
            all_summaries = [state2.summary(args.symbol, d) for d in completed_days]
        finally:
            state2.close()
        failed_checks_n = int(sum(int(x.get("failed_invariants", 0)) for x in all_summaries))
        status = "PASS" if len(completed_days) == len(days) and not failed_days and not agg_failed_days and failed_checks_n == 0 else ("PARTIAL" if completed_days else "FAIL")
        summary = {
            "status": status, "builder_version": BUILDER_VERSION, "run_id": run_id, "symbol": args.symbol,
            "start_date": days[0].isoformat(), "end_date": days[-1].isoformat(),
            "dates_requested": len(days), "dates_completed": len(completed_days),
            "dates_built": built, "dates_reused": reused, "dates_failed": len(failed_days) + len(agg_failed_days),
            "primitive_rows": int(sum(int(r.get("rows",0)) for r in quality_from_outputs)),
            "events": int(sum(int(x.get("events",0)) for x in all_summaries)),
            "detected_events": int(sum(int(x.get("detected_events",0)) for x in all_summaries)),
            "directional_detected_events": int(sum(int(x.get("directional_detected_events",0)) for x in all_summaries)),
            "controls": int(sum(int(x.get("controls",0)) for x in all_summaries)),
            "markout_rows": int(sum(int(x.get("markout_rows",0)) for x in all_summaries)),
            "performance_stage_seconds": {k: float(v) for k,v in stage_seconds.items()},
            "failed_invariants": failed_checks_n, "failed_days": failed_days + agg_failed_days,
            "runtime_seconds": time.perf_counter() - started,
            "events_root": str(args.output_root / "events" / f"symbol={args.symbol}"),
            "controls_root": str(args.output_root / "controls" / f"symbol={args.symbol}"),
            "markouts_root": str(args.output_root / "markouts" / f"symbol={args.symbol}"),
            "aggregation_contract": "partitioned_chronological_global_candidate_universe_v1",
        }
        atomic_write_json(metadata_root / "latest_summary.json", summary)
        atomic_write_text(args.report_root / "liquidity_primitive_findings.md", build_partitioned_findings(summary))
        print("\nPhase-6B causal liquidity-event primitive summary", flush=True)
        print("---------------------------------------------------", flush=True)
        print(f"Status                  : {summary['status']}", flush=True)
        print(f"Dates completed         : {summary['dates_completed']}/{summary['dates_requested']}", flush=True)
        print(f"Primitive rows          : {summary['primitive_rows']:,}", flush=True)
        print(f"Events                  : {summary['events']:,}", flush=True)
        print(f"Detected events         : {summary['detected_events']:,}", flush=True)
        print(f"Matched controls        : {summary['controls']:,}", flush=True)
        print(f"Markout rows            : {summary['markout_rows']:,}", flush=True)
        print(f"Failed invariants       : {summary['failed_invariants']}", flush=True)
        print(f"Runtime                 : {summary['runtime_seconds']:.1f}s", flush=True)
        print(f"Summary                 : {metadata_root / 'latest_summary.json'}", flush=True)
        print(f"Findings                : {args.report_root / 'liquidity_primitive_findings.md'}", flush=True)
        print("No alpha or strategy claim is made by Phase 6B.", flush=True)
        return 0 if summary["status"] == "PASS" else 1
    finally:
        state.close()


if __name__ == "__main__":
    raise SystemExit(main())
