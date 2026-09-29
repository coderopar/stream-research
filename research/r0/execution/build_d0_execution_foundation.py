#!/usr/bin/env python3
"""R0.6 — build an outcome-blind D0 execution feasibility artifact.

The production build replays canonical BTCUSDT snapshots + diff depth in exact
receive-time order and samples the reconstructed book at each R0.5 governed
information timestamp under the fixed ZERO_LATENCY_AUDIT_ONLY profile.

No future outcome artifact is opened.  No P1 partition is read.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from research_core.clocks.contract import D0_END_EXCLUSIVE_NS, D0_START_NS, NS_PER_SECOND, require_int64_ns
from research_core.execution import ZERO_LATENCY_AUDIT, ReplayBook, acceleration_available, build_execution_timeline, simulate_marketable_fill

SCHEMA_NAME = "r0_execution_simulator_v1"
SCHEMA_VERSION = 1
BUILDER_VERSION = "1.0.0"
EXPECTED_CANDIDATES = 7067
EXPECTED_CLOCKS = 7031
EXPECTED_UNAVAILABLE = 36
AUDIT_NOTIONAL_QUOTE = 1000.0
SYMBOL = "BTCUSDT"
PRICE_SCALE = 10.0
D0_START_DATE = date(2026, 5, 18)
D0_END_DATE = date(2026, 7, 18)
P1_START_DATE = date(2026, 7, 19)

DEFAULT_CANONICAL_ROOT = REPO_ROOT / "data" / "canonical" / "market_events_v1"
DEFAULT_R05_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "r0_causal_market_state_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / SCHEMA_NAME


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def progress(msg: str) -> None:
    print(f"[R0.6 {datetime.now(timezone.utc).strftime('%H:%M:%SZ')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex[:8]}.tmp"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str))


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex[:8]}.tmp"
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex[:8]}.tmp"
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), tmp, compression="zstd", write_statistics=True)
    check = pq.ParquetFile(tmp)
    try:
        if int(check.metadata.num_rows) != len(frame):
            raise RuntimeError("atomic parquet row-count validation failed")
    finally:
        close = getattr(check, "close", None)
        if callable(close): close()
    os.replace(tmp, path)


def utc_date_from_ns(ts_ns: int) -> date:
    return datetime.fromtimestamp(int(ts_ns) / 1e9, tz=timezone.utc).date()


def d0_dates() -> list[date]:
    out=[]; d=D0_START_DATE
    while d <= D0_END_DATE:
        out.append(d); d += timedelta(days=1)
    return out


def canonical_family_paths(root: Path, family: str, d: date) -> tuple[Path, ...]:
    if d >= P1_START_DATE:
        raise RuntimeError(f"non-D0 canonical partition refused: {d}")
    base = root / "perps" / family / f"date={d.isoformat()}"
    return tuple(sorted(base.glob("*.parquet"))) if base.exists() else ()


def load_r05_candidate_core(r05_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    import pyarrow.parquet as pq
    manifest_path = r05_root / "metadata" / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"missing accepted R0.5 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_name") != "r0_causal_market_state_v1":
        raise RuntimeError("unexpected R0.5 schema")
    counts = manifest.get("counts", {})
    if int(counts.get("candidate_rows", -1)) != EXPECTED_CANDIDATES or int(counts.get("governed_clock_rows", -1)) != EXPECTED_CLOCKS:
        raise RuntimeError("R0.5 accepted counts changed")
    p = r05_root / "causal" / "candidate_core.parquet"
    expected = str(manifest.get("outputs", {}).get("candidate_core", {}).get("sha256", ""))
    if not p.exists() or not expected or sha256_file(p) != expected:
        raise RuntimeError("R0.5 candidate_core hash mismatch")
    core = pq.read_table(p).to_pandas()
    req = {"candidate_id","candidate_date","direction","information_available_ts_ns","clock_available","clock_unavailable_reason"}
    if not req <= set(core.columns): raise RuntimeError(f"R0.5 candidate_core missing fields: {sorted(req-set(core.columns))}")
    if len(core) != EXPECTED_CANDIDATES or core["candidate_id"].astype(str).duplicated().any(): raise RuntimeError("R0.5 candidate_core identity invariant failed")
    return core, {"manifest": str(manifest_path), "candidate_core": {"path": str(p), "sha256": expected}}


def _require_exact_int64_schema(pf: Any, path: Path, names: Sequence[str]) -> None:
    import pyarrow as pa
    fields = {f.name:f.type for f in pf.schema_arrow}
    for name in names:
        if name not in fields: raise RuntimeError(f"{path}: missing required field {name}")
        if not pa.types.is_int64(fields[name]): raise RuntimeError(f"{path}: {name} must be exact int64, got {fields[name]}")


def _iter_snapshot_file(path: Path) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq
    cols=["ts_recv_ns","symbol","last_update_id","bids","asks","source_relative_path","source_record_number"]
    pf=pq.ParquetFile(path)
    try:
        _require_exact_int64_schema(pf,path,["ts_recv_ns","last_update_id","source_record_number"])
        use=[c for c in cols if c in pf.schema_arrow.names]
        last=None
        for batch in pf.iter_batches(batch_size=64,columns=use):
            for r in batch.to_pylist():
                if str(r.get("symbol","")).upper()!=SYMBOL: continue
                ts=int(r["ts_recv_ns"]); sr=int(r.get("source_record_number") or 0); sp=str(r.get("source_relative_path") or path.as_posix())
                key=(ts,sp,sr)
                if last is not None and key<last: raise RuntimeError(f"canonical snapshot shard is not receive-time sorted: {path}")
                last=key; r["ts_recv_ns"]=ts; r["source_record_number"]=sr; r["_source_path"]=sp
                yield r
    finally:
        close=getattr(pf,"close",None)
        if callable(close): close()


def _arrow_levels(array: Any, price_scale: float, path: Path, name: str):
    import pyarrow as pa
    import pyarrow.compute as pc
    if not (pa.types.is_list(array.type) or pa.types.is_large_list(array.type)):
        raise RuntimeError(f"{path}: {name} must be list<struct>")
    off_abs=array.offsets.to_numpy(zero_copy_only=False).astype("int64",copy=False)
    if len(off_abs)==0:return np.asarray([0],dtype=np.int64),np.empty(0,dtype=np.int64),np.empty(0,dtype=np.float64)
    base=int(off_abs[0]); stop=int(off_abs[-1]); offsets=off_abs-base; values=array.values.slice(base,max(0,stop-base))
    price=pc.cast(values.field("price"),pa.float64(),safe=False).to_numpy(zero_copy_only=False)
    qty=pc.cast(values.field("quantity"),pa.float64(),safe=False).to_numpy(zero_copy_only=False)
    price=np.asarray(price,dtype=np.float64); qty=np.asarray(qty,dtype=np.float64)
    finite=np.isfinite(price)&(price>0); scaled=price[finite]*float(price_scale)
    if len(scaled):
        rounded=np.rint(scaled); bad=np.abs(scaled-rounded)>1e-6
        if np.any(bad): raise RuntimeError(f"{path}: {name} contains price not aligned to price_scale={price_scale:g}")
    ticks=np.zeros(len(price),dtype=np.int64); ticks[finite]=np.rint(price[finite]*float(price_scale)).astype(np.int64)
    return offsets.astype(np.int64,copy=False),ticks,qty


def _iter_depth_file(path: Path, price_scale: float) -> Iterator[dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    cols=["ts_recv_ns","symbol","first_update_id","final_update_id","previous_final_update_id","bid_updates","ask_updates","source_relative_path","source_record_number"]
    pf=pq.ParquetFile(path)
    try:
        _require_exact_int64_schema(pf,path,["ts_recv_ns","first_update_id","final_update_id","previous_final_update_id","source_record_number"])
        use=[c for c in cols if c in pf.schema_arrow.names]; last=None
        for batch in pf.iter_batches(batch_size=8192,columns=use):
            if "symbol" in batch.schema.names:
                sym=batch.column(batch.schema.get_field_index("symbol")); mask=pc.fill_null(pc.equal(sym,pa.scalar(SYMBOL)),False)
                if not bool(pc.all(mask).as_py()): batch=batch.filter(mask)
            if batch.num_rows==0: continue
            def ints(name:str):
                arr=batch.column(batch.schema.get_field_index(name)); return np.asarray(pc.cast(arr,pa.int64(),safe=False).to_numpy(zero_copy_only=False),dtype=np.int64)
            ts=ints("ts_recv_ns"); U=ints("first_update_id"); u=ints("final_update_id"); pu=ints("previous_final_update_id"); sr=ints("source_record_number")
            b=batch.column(batch.schema.get_field_index("bid_updates")); a=batch.column(batch.schema.get_field_index("ask_updates"))
            bo,bt,bq=_arrow_levels(b,price_scale,path,"bid_updates"); ao,at,aq=_arrow_levels(a,price_scale,path,"ask_updates")
            if "source_relative_path" in batch.schema.names: sps=batch.column(batch.schema.get_field_index("source_relative_path")).to_pylist()
            else:sps=[path.as_posix()]*batch.num_rows
            for i in range(batch.num_rows):
                sp=str(sps[i] or path.as_posix()); key=(int(ts[i]),sp,int(sr[i]))
                if last is not None and key<last: raise RuntimeError(f"canonical depth shard is not receive-time sorted: {path}")
                last=key; b0,b1=int(bo[i]),int(bo[i+1]); a0,a1=int(ao[i]),int(ao[i+1])
                yield {"ts_recv_ns":int(ts[i]),"first_update_id":int(U[i]),"final_update_id":int(u[i]),"previous_final_update_id":int(pu[i]),"source_record_number":int(sr[i]),"_source_path":sp,"_fast_bid_ticks":bt[b0:b1],"_fast_bid_qty":bq[b0:b1],"_fast_ask_ticks":at[a0:a1],"_fast_ask_qty":aq[a0:a1]}
    finally:
        close=getattr(pf,"close",None)
        if callable(close): close()


def _merge_files(paths: Sequence[Path], maker, priority: int, kind: str) -> Iterator[dict[str, Any]]:
    heap=[]
    for idx,p in enumerate(paths):
        it=maker(p)
        try:r=next(it)
        except StopIteration:continue
        heap.append((int(r["ts_recv_ns"]),str(r.get("_source_path","")),int(r.get("source_record_number",0)),idx,r,it))
    heapq.heapify(heap)
    while heap:
        _,_,_,idx,r,it=heapq.heappop(heap); r["_priority"]=priority; r["_kind"]=kind; yield r
        try:n=next(it)
        except StopIteration:continue
        heapq.heappush(heap,(int(n["ts_recv_ns"]),str(n.get("_source_path","")),int(n.get("source_record_number",0)),idx,n,it))


def merged_book_events(snapshot_paths: Sequence[Path], depth_paths: Sequence[Path], price_scale: float) -> Iterator[dict[str, Any]]:
    snaps=_merge_files(snapshot_paths,_iter_snapshot_file,0,"SNAPSHOT")
    depth=_merge_files(depth_paths,lambda p:_iter_depth_file(p,price_scale),1,"DEPTH")
    heap=[]
    for idx,it in enumerate((snaps,depth)):
        try:r=next(it)
        except StopIteration:continue
        heap.append((int(r["ts_recv_ns"]),int(r["_priority"]),str(r.get("_source_path","")),int(r.get("source_record_number",0)),idx,r,it))
    heapq.heapify(heap)
    while heap:
        _,_,_,_,idx,r,it=heapq.heappop(heap); yield r
        try:n=next(it)
        except StopIteration:continue
        heapq.heappush(heap,(int(n["ts_recv_ns"]),int(n["_priority"]),str(n.get("_source_path","")),int(n.get("source_record_number",0)),idx,n,it))


def _audit_row(candidate: Mapping[str, Any], book: ReplayBook) -> dict[str, Any]:
    cid=str(candidate["candidate_id"])
    if not bool(candidate["clock_available"]):
        return {"candidate_id":cid,"candidate_date":str(candidate["candidate_date"]),"direction":int(candidate["direction"]),"clock_available":False,"arrival_profile_id":"ZERO_LATENCY_AUDIT_ONLY","arrival_ts_ns":None,"book_status":"CLOCK_UNAVAILABLE","sequence_valid":False,"state_age_ns":None,"best_bid":None,"best_ask":None,"mid":None,"spread_bps":None,"audit_notional_quote":AUDIT_NOTIONAL_QUOTE,"buy_fill_complete":False,"buy_vwap":None,"buy_raw_crossing_cost_bps":None,"sell_fill_complete":False,"sell_vwap":None,"sell_raw_crossing_cost_bps":None}
    info=require_int64_ns(int(candidate["information_available_ts_ns"]),"information_available_ts_ns")
    tl=build_execution_timeline(info,ZERO_LATENCY_AUDIT); arrival=int(tl["estimated_exchange_arrival_ts_ns"])
    if arrival < D0_START_NS or arrival >= D0_END_EXCLUSIVE_NS: raise RuntimeError(f"{cid}: audit arrival leaves D0")
    state=book.state(arrival)
    row={"candidate_id":cid,"candidate_date":str(candidate["candidate_date"]),"direction":int(candidate["direction"]),"clock_available":True,"arrival_profile_id":str(tl["latency_profile_id"]),"arrival_ts_ns":arrival,"book_status":state.status,"sequence_valid":bool(state.sequence_valid),"bridge_pending":bool(state.bridge_pending),"last_state_ts_ns":state.last_state_ts_ns,"state_age_ns":state.state_age_ns,"best_bid":state.best_bid,"best_ask":state.best_ask,"best_bid_qty":state.best_bid_qty,"best_ask_qty":state.best_ask_qty,"mid":state.mid,"spread_bps":state.spread_bps,"last_update_id":state.last_update_id,"bootstrap_source":state.bootstrap_source,"last_invalid_reason":state.last_invalid_reason,"audit_notional_quote":AUDIT_NOTIONAL_QUOTE,"buy_fill_complete":False,"buy_vwap":None,"buy_filled_quote":0.0,"buy_raw_crossing_cost_bps":None,"sell_fill_complete":False,"sell_vwap":None,"sell_filled_quote":0.0,"sell_raw_crossing_cost_bps":None}
    if state.sequence_valid and state.mid is not None:
        buy=simulate_marketable_fill(side="BUY",quote_notional=AUDIT_NOTIONAL_QUOTE,levels=book.levels_for("BUY"),arrival_mid=state.mid)
        sell=simulate_marketable_fill(side="SELL",quote_notional=AUDIT_NOTIONAL_QUOTE,levels=book.levels_for("SELL"),arrival_mid=state.mid)
        row.update({"buy_fill_complete":buy.complete,"buy_vwap":buy.vwap,"buy_filled_quote":buy.filled_quote,"buy_raw_crossing_cost_bps":buy.raw_crossing_cost_bps,"sell_fill_complete":sell.complete,"sell_vwap":sell.vwap,"sell_filled_quote":sell.filled_quote,"sell_raw_crossing_cost_bps":sell.raw_crossing_cost_bps})
    return row


def build(args: argparse.Namespace) -> int:
    started=time.perf_counter()
    progress("stage 1/5: validating accepted R0.5 candidate core and execution contract")
    core,parent=load_r05_candidate_core(args.r05_root)
    if int(core["clock_available"].astype(bool).sum())!=EXPECTED_CLOCKS or int((~core["clock_available"].astype(bool)).sum())!=EXPECTED_UNAVAILABLE: raise RuntimeError("R0.5 clock counts changed")
    governed=core[core["clock_available"].astype(bool)].copy()
    governed["arrival_date"]=governed["information_available_ts_ns"].map(lambda x:utc_date_from_ns(int(x)))
    by_day={d:g.sort_values(["information_available_ts_ns","candidate_id"],kind="mergesort").to_dict("records") for d,g in governed.groupby("arrival_date")}
    unavailable=core[~core["clock_available"].astype(bool)].to_dict("records")
    progress(f"stage 1/5 complete: {len(core):,} candidates, {len(governed):,} governed clocks")

    if not acceleration_available() and not args.allow_slow_python:
        raise RuntimeError("full R0.6 replay requires numba acceleration; install numba or pass --allow-slow-python only for diagnostics")
    book=ReplayBook(price_scale=args.price_scale,accelerated=acceleration_available() and not args.allow_slow_python)
    # warm the compiled update kernel before timing the production replay
    if acceleration_available() and not args.allow_slow_python:
        book.bids.update_arrays(np.asarray([1000],dtype=np.int64),np.asarray([1.0],dtype=np.float64)); book.bids.update_arrays(np.asarray([1000],dtype=np.int64),np.asarray([0.0],dtype=np.float64))

    progress("stage 2/5: replaying canonical snapshots+diffs and sampling zero-latency audit execution state")
    rows=[]; daily=[]; gaps=[]; total_depth=0; source_prov=[]
    # preserve the 36 unavailable candidates without touching the book
    rows.extend(_audit_row(r,book) for r in unavailable)
    for di,d in enumerate(d0_dates(),start=1):
        snaps=canonical_family_paths(args.canonical_root,"book_snapshots",d); depths=canonical_family_paths(args.canonical_root,"depth_updates",d)
        queries=by_day.get(d,[]); qidx=0; day_available=0; day_depth0=book.depth_count; day_snap0=book.snapshot_count; gap0=book.sequence_gap_count+book.crossed_count+book.bridge_failure_count
        if args.progress_every_day and (di==1 or di%args.progress_every_day==0 or di==len(d0_dates())):
            progress(f"execution replay: D0 day {di}/62 ({d}); queries={len(queries):,}; snapshots={len(snaps)}, depth_files={len(depths)}")
        for role, paths in (("book_snapshots", snaps), ("depth_updates", depths)):
            for sp in paths:
                st=sp.stat(); source_prov.append({"date":d.isoformat(),"role":role,"path":str(sp),"size_bytes":int(st.st_size)})
        if not snaps or not depths:
            book.invalidate("SOURCE_PARTITION_MISSING")
            for q in queries:
                rows.append(_audit_row(q,book))
            daily.append({"date":d.isoformat(),"query_count":len(queries),"available_count":0,"snapshot_files":len(snaps),"depth_files":len(depths),"depth_messages":0,"snapshot_messages":0,"gap_events":0,"source_status":"SOURCE_PARTITION_MISSING"})
            continue
        current_ts=None
        for ev in merged_book_events(snaps,depths,args.price_scale):
            ts=int(ev["ts_recv_ns"])
            if current_ts is None: current_ts=ts
            if ts!=current_ts:
                while qidx<len(queries) and int(queries[qidx]["information_available_ts_ns"]) < ts:
                    r=_audit_row(queries[qidx],book); rows.append(r); day_available += int(r["book_status"]=="AVAILABLE"); qidx+=1
                current_ts=ts
            before=(book.sequence_gap_count,book.crossed_count,book.bridge_failure_count,book.last_invalid_reason)
            if ev["_kind"]=="SNAPSHOT": book.apply_snapshot(ev)
            else:
                book.apply_depth(ev); total_depth+=1
                if args.progress_every_depth and total_depth%args.progress_every_depth==0:
                    progress(f"  depth messages={total_depth:,}; valid={'Y' if book.valid else 'N'}; pending={'Y' if book.bridge_pending else 'N'}")
            after=(book.sequence_gap_count,book.crossed_count,book.bridge_failure_count,book.last_invalid_reason)
            if after[:3]!=before[:3]: gaps.append({"date":d.isoformat(),"ts_recv_ns":ts,"reason":book.last_invalid_reason})
        while qidx<len(queries):
            r=_audit_row(queries[qidx],book); rows.append(r); day_available += int(r["book_status"]=="AVAILABLE"); qidx+=1
        daily.append({"date":d.isoformat(),"query_count":len(queries),"available_count":day_available,"snapshot_files":len(snaps),"depth_files":len(depths),"depth_messages":book.depth_count-day_depth0,"snapshot_messages":book.snapshot_count-day_snap0,"gap_events":book.sequence_gap_count+book.crossed_count+book.bridge_failure_count-gap0,"source_status":"AVAILABLE"})
    frame=pd.DataFrame(rows)
    if len(frame)!=EXPECTED_CANDIDATES or frame["candidate_id"].astype(str).duplicated().any(): raise RuntimeError("R0.6 candidate preservation failed")
    progress(f"stage 2/5 complete: {len(frame):,} candidate execution rows")

    progress("stage 3/5: validating execution and protection invariants")
    avail=frame[frame["book_status"]=="AVAILABLE"]
    if len(avail) and bool((avail["arrival_ts_ns"].astype("int64")>=D0_END_EXCLUSIVE_NS).any()): raise RuntimeError("P1 arrival detected")
    if len(avail) and bool((avail["last_state_ts_ns"].astype("int64")>avail["arrival_ts_ns"].astype("int64")).any()): raise RuntimeError("future book state selected")
    if len(avail) and bool((avail["best_bid"]>=avail["best_ask"]).any()): raise RuntimeError("crossed available book")
    progress("stage 3/5 complete: no future state, crossed available book, or P1 access")

    progress("stage 4/5: writing execution, quality and replay diagnostics")
    coverage=frame[["candidate_id","clock_available","book_status","sequence_valid","state_age_ns","buy_fill_complete","sell_fill_complete"]].copy()
    out_exec=args.output_root/"execution"/"audit_zero_latency_state.parquet"; out_cov=args.output_root/"quality"/"candidate_execution_coverage.parquet"
    atomic_parquet(out_exec,frame.sort_values(["candidate_date","candidate_id"],kind="mergesort").reset_index(drop=True)); atomic_parquet(out_cov,coverage)
    daily_df=pd.DataFrame(daily); gaps_df=pd.DataFrame(gaps,columns=["date","ts_recv_ns","reason"]); source_df=pd.DataFrame(source_prov,columns=["date","role","path","size_bytes"]).drop_duplicates(subset=["path"],keep="first")
    atomic_csv(args.report_root/"daily_replay_summary.csv",daily_df); atomic_csv(args.report_root/"sequence_gap_summary.csv",gaps_df); atomic_csv(args.report_root/"source_provenance.csv",source_df)
    progress("stage 4/5 complete")

    progress("stage 5/5: writing manifest and outcome-blind summary")
    status_counts={str(k):int(v) for k,v in frame["book_status"].fillna("UNKNOWN").value_counts().items()}
    ages=pd.to_numeric(avail["state_age_ns"],errors="coerce").dropna()/1e6 if len(avail) else pd.Series(dtype=float)
    summary={
        "candidate_rows":int(len(frame)),"governed_clock_rows":EXPECTED_CLOCKS,"clock_unavailable_rows":EXPECTED_UNAVAILABLE,
        "book_status_counts":status_counts,"book_available_rows":int(len(avail)),
        "audit_buy_complete_rows":int(avail["buy_fill_complete"].fillna(False).astype(bool).sum()) if len(avail) else 0,
        "audit_sell_complete_rows":int(avail["sell_fill_complete"].fillna(False).astype(bool).sum()) if len(avail) else 0,
        "state_age_ms_median":float(ages.median()) if len(ages) else None,"state_age_ms_p95":float(ages.quantile(.95)) if len(ages) else None,"state_age_ms_max":float(ages.max()) if len(ages) else None,
        "sequence_gap_events":int(book.sequence_gap_count),"crossed_book_events":int(book.crossed_count),"bootstrap_bridge_failures":int(book.bridge_failure_count),
        "status":"BUILD_COMPLETE_REVIEW_REQUIRED"
    }
    manifest={"schema_name":SCHEMA_NAME,"schema_version":SCHEMA_VERSION,"builder_version":BUILDER_VERSION,"generated_at_utc":utc_now_iso(),"evidentiary_role":"D0_EXECUTION_INFRASTRUCTURE_ONLY","accepted_parent_checkpoint":{"commit":"62c99c0","tag":"r0-causal-market-state-v1"},"parents":parent,"canonical":{"logical_root":str(args.canonical_root),"families_read":["perps/book_snapshots","perps/depth_updates"],"source_files_observed":int(len(source_df)),"source_full_content_rehash":False,"r0_1_lineage_tag":"r0-governance-provenance-v1"},"outcome_firewall":{"future_outcomes_opened":False,"returns_mfe_mae_threshold_labels_used":False,"p1_read":False},"audit_contract":{"profile":"ZERO_LATENCY_AUDIT_ONLY","quote_notional_usdt":AUDIT_NOTIONAL_QUOTE,"scientific_strategy_definition":False,"fees_applied":False,"passive_fill_model":False,"own_market_impact_model":False},"book_contract":{"snapshot_bootstrap":"complete checkpoint + exact first-following pu==last_update_id handshake","healthy_state":"continuous pu->u diff replay","gap":"invalidate until later valid handshake","freshness":"exact state_age_ns recorded; no universal max-age threshold frozen"},"counts":summary,"outputs":{"audit_execution_state":{"path":str(out_exec),"sha256":sha256_file(out_exec)},"candidate_execution_coverage":{"path":str(out_cov),"sha256":sha256_file(out_cov)}},"reports":{"daily_replay_summary":str(args.report_root/"daily_replay_summary.csv"),"sequence_gap_summary":str(args.report_root/"sequence_gap_summary.csv"),"source_provenance":str(args.report_root/"source_provenance.csv")},"status":"BUILD_COMPLETE_REVIEW_REQUIRED"}
    atomic_json(args.output_root/"metadata"/"manifest.json",manifest); atomic_json(args.report_root/"summary.json",manifest)
    progress(f"stage 5/5 complete: full R0.6 build finished in {time.perf_counter()-started:.1f}s")
    print(json.dumps(summary,indent=2,sort_keys=True),flush=True)
    return 0


def self_test() -> int:
    from research_core.execution import CostAssumptions, LatencyAssumptions, ReplayBook, build_execution_timeline, simulate_marketable_fill, total_cost_bps
    info=D0_START_NS+100*NS_PER_SECOND
    t=build_execution_timeline(info,LatencyAssumptions(1,2,3,4,5,6,"SYNTH"))
    assert t["decision_ts_ns"]==info+3 and t["estimated_exchange_arrival_ts_ns"]==info+10 and t["fill_complete_ts_ns"]==info+21
    b=ReplayBook(price_scale=10.0,accelerated=False)
    b.apply_snapshot({"ts_recv_ns":info-2,"last_update_id":10,"bids":[{"price":"99.9","quantity":"20"}],"asks":[{"price":"100.1","quantity":"20"}]})
    assert not b.state(info-2).sequence_valid
    b.apply_depth({"ts_recv_ns":info-1,"first_update_id":11,"final_update_id":11,"previous_final_update_id":10,"bid_updates":[],"ask_updates":[]})
    s=b.state(info); assert s.sequence_valid and s.best_bid==99.9 and s.best_ask==100.1
    f=simulate_marketable_fill(side="BUY",quote_notional=1000,levels=b.levels_for("BUY"),arrival_mid=s.mid)
    assert f.complete and f.vwap==100.1 and f.raw_crossing_cost_bps is not None
    assert total_cost_bps(f.raw_crossing_cost_bps,CostAssumptions(4.0,0.0,0.0,"SYNTH"))>f.raw_crossing_cost_bps
    b.apply_depth({"ts_recv_ns":info+1,"first_update_id":12,"final_update_id":12,"previous_final_update_id":999,"bid_updates":[],"ask_updates":[]})
    assert not b.state(info+1).sequence_valid
    print("SELF-TEST PASS: explicit latency timeline, checkpoint+diff handshake, marketable depth walk, explicit costs, and gap invalidation validated.")
    return 0


def parse_args(argv: Sequence[str] | None=None) -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--canonical-root",type=Path,default=DEFAULT_CANONICAL_ROOT)
    p.add_argument("--r05-root",type=Path,default=DEFAULT_R05_ROOT)
    p.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root",type=Path,default=DEFAULT_REPORT_ROOT)
    p.add_argument("--price-scale",type=float,default=PRICE_SCALE)
    p.add_argument("--progress-every-day",type=int,default=5)
    p.add_argument("--progress-every-depth",type=int,default=500000)
    p.add_argument("--allow-slow-python",action="store_true")
    p.add_argument("--self-test",action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None=None) -> int:
    args=parse_args(argv)
    if args.self_test:return self_test()
    return build(args)

if __name__=="__main__": raise SystemExit(main())
