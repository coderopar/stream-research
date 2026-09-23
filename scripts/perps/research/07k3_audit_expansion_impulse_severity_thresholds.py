#!/usr/bin/env python3
"""
Phase 7K.3 — Expansion-impulse severity threshold audit
======================================================

Build ONE common retrospective impulse universe from 50-bp first-passage
expansions, then label the SAME impulses by whether they reach 50/75/100 bps
within 5/10/15 minutes. Reuse frozen 7K.2 causal candidate policies and old
7I.1 only for target-aware coverage auditing. No raw depth/trade replay, ML,
options, or upstream mutation.

Core rules
----------
* Phase-6A one-second execution price.
* Fixed 10-second audit anchors.
* Common parent: first passage to 50 bps within any fully observable 5/10/15m.
* Same-direction parent detections cluster by 50-bp hit time, bounded to 180s
  from the cluster's FIRST hit (non-transitive).
* 75/100 are nested labels on the same impulse, never independently clustered.
* Re-acceleration = second-or-later common 50-bp impulse inside the same frozen
  Phase-7A physical leg.
* Coverage is target-aware: a candidate can miss 50 but still cover 75/100 if
  it fires before that later target is reached.
* For later impulses, freshness requires previous common impulse 50-bp hit <
  candidate origin <= candidate <= current target hit; if release_second exists,
  origin <= release <= candidate must also hold.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from bisect import bisect_right
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

BUILDER_VERSION = "1.0.0"
SCHEMA_VERSION = 1
SCHEMA_NAME = "expansion_impulse_severity_threshold_audit_v1"
DEFAULT_SYMBOL = "BTCUSDT"

FROZEN_PHASE7A_VERSION = "1.0.0"
FROZEN_PHASE7K1_VERSION = "1.0.2"
FROZEN_PHASE7K2_VERSION = "1.0.0"

ANCHOR_CADENCE_SEC = 10
COMMON_PARENT_BPS = 50.0
SEVERITY_THRESHOLDS_BPS = (50.0, 75.0, 100.0)
HORIZONS_SEC = (300, 600, 900)
COMMON_CLUSTER_HIT_SPAN_SEC = 180
CANDIDATE_MAPPING_LOOKBACK_SEC = 900
DEFAULT_MAX_TRADE_STALE_SEC = 5

K2_POLICIES = (
    "TEST_COOLDOWN_0S",
    "TEST_COOLDOWN_60S",
    "TEST_COOLDOWN_120S",
    "BENCHMARK_COOLDOWN_180S",
)
OLD_POLICY = "OLD_7I1_FROZEN"
ALL_POLICIES = (*K2_POLICIES, OLD_POLICY)

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_PHASE7A_SCRIPT = REPO_ROOT / "scripts/perps/research/07a_build_large_move_event_catalog.py"
DEFAULT_PHASE7K1_SCRIPT = REPO_ROOT / "scripts/perps/research/07k1_build_causal_repeatable_expansion_impulse_candidate_catalog.py"
DEFAULT_EXECUTION_ROOT = REPO_ROOT / "data/derived/perps/execution_foundation_v1"
DEFAULT_PHASE7A_ROOT = REPO_ROOT / "data/derived/perps/research/large_move_event_catalog_v1"
DEFAULT_PHASE7I1_ROOT = REPO_ROOT / "data/derived/perps/research/causal_large_move_candidate_catalog_v1"
DEFAULT_PHASE7K2_ROOT = REPO_ROOT / "data/derived/perps/research/causal_rearm_timing_audit_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/derived/perps/research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports/perps_expansion_impulse_severity_threshold_audit"


def stack(parquet=False):
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise SystemExit("Phase 7K.3 requires numpy and pandas; activate .venv first.") from exc
    if not parquet:
        return np, pd, None, None
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit("Phase 7K.3 parquet I/O requires pyarrow.") from exc
    return np, pd, pa, pq


def load_module(path: Path, name: str):
    import sys
    if not path.exists():
        raise RuntimeError(f"Missing required frozen script: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except Exception:
        return math.nan


def iso_time(sec: Any) -> str | None:
    v = safe_float(sec)
    if not math.isfinite(v):
        return None
    return datetime.fromtimestamp(int(v), tz=timezone.utc).replace(microsecond=0).isoformat()


def stable_id(*parts: Any, length=24):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:length]


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7k3-", suffix=path.suffix, dir=str(path.parent)); os.close(fd)
    try:
        Path(tmp).write_text(text, encoding="utf-8"); os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_json(path: Path, obj):
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def atomic_csv(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7k3-", suffix=".csv", dir=str(path.parent)); os.close(fd)
    try:
        df.to_csv(tmp, index=False); os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_parquet(df, path: Path):
    _, _, pa, pq = stack(parquet=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7k3-", suffix=".parquet", dir=str(path.parent)); os.close(fd)
    try:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd"); os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def contiguous_segments(sec, price):
    np, _, _, _ = stack()
    valid = np.isfinite(price) & (price > 0)
    start = prev = None
    for i in range(len(sec)):
        if not valid[i]:
            if start is not None: yield start, prev
            start = prev = None; continue
        if start is None: start = i
        elif prev is None or int(sec[i]) != int(sec[prev]) + 1:
            yield start, prev; start = i
        prev = i
    if start is not None: yield start, prev


def forward_extreme(sec, price, horizon):
    np, pd, _, _ = stack()
    fmax = np.full(len(price), np.nan); fmin = np.full(len(price), np.nan); w = horizon + 1
    for a, b in contiguous_segments(sec, price):
        if b - a + 1 < w: continue
        arr = price[a:b+1]; rev = pd.Series(arr[::-1])
        fmax[a:b+1] = rev.rolling(w, min_periods=w).max().to_numpy()[::-1]
        fmin[a:b+1] = rev.rolling(w, min_periods=w).min().to_numpy()[::-1]
    return fmax, fmin


def full_horizon(sec, price, i, h):
    j = i + h
    return j < len(sec) and math.isfinite(float(price[i])) and math.isfinite(float(price[j])) and int(sec[j]) == int(sec[i]) + h


def first_hit(price, i, j, anchor, direction, target):
    np, _, _, _ = stack()
    path = price[i:j+1]
    dr = ((path / anchor - 1.0) if direction > 0 else (1.0 - path / anchor)) * 10000.0
    hits = np.flatnonzero(dr + 1e-12 >= target)
    return int(i + hits[0]) if len(hits) else None


def build_parent_detections(sec, price, start_date, end_date):
    np, pd, _, _ = stack()
    print("      computing 5/10/15m forward extrema...")
    extrema = {h: forward_extreme(sec, price, h) for h in HORIZONS_SEC}
    lo = int(datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc).timestamp())
    hi = int(datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp())
    anchors = np.flatnonzero((sec % ANCHOR_CADENCE_SEC == 0) & (sec >= lo) & (sec < hi) & np.isfinite(price) & (price > 0))
    rows = []
    for k, i in enumerate(anchors):
        if k and k % 50000 == 0:
            print(f"      scanned anchors={k:,}/{len(anchors):,} parent detections={len(rows):,}")
        ap = float(price[i]); available = {h: full_horizon(sec, price, i, h) for h in HORIZONS_SEC}
        hs = [h for h in HORIZONS_SEC if available[h]]
        if not hs: continue
        max_h = max(hs)
        for direction in (+1, -1):
            qualifies = False
            for h in hs:
                mx, mn = extrema[h]
                if not (math.isfinite(float(mx[i])) and math.isfinite(float(mn[i]))): continue
                mfe = ((float(mx[i]) / ap - 1.0) if direction > 0 else (1.0 - float(mn[i]) / ap)) * 10000.0
                if mfe + 1e-12 >= COMMON_PARENT_BPS:
                    qualifies = True; break
            if not qualifies: continue
            end_i = i + max_h
            hits = {int(t): first_hit(price, i, end_i, ap, direction, t) for t in SEVERITY_THRESHOLDS_BPS}
            if hits[50] is None: raise RuntimeError("50bp extrema qualified but first hit missing")
            r = {
                "parent_detection_id": stable_id("7K3", int(sec[i]), direction, int(sec[hits[50]])),
                "direction": direction, "direction_label": "UP" if direction > 0 else "DOWN",
                "anchor_second": int(sec[i]), "anchor_time": iso_time(sec[i]), "anchor_price": ap,
                "max_observable_horizon_sec": max_h,
            }
            for h in HORIZONS_SEC: r[f"available_{h//60}m"] = bool(available[h])
            for target in (50, 75, 100):
                idx = hits[target]; hsx = int(sec[idx]) if idx is not None else None
                r[f"hit_{target}_second"] = hsx if hsx is not None else math.nan
                r[f"hit_{target}_time"] = iso_time(hsx)
                r[f"seconds_to_{target}"] = hsx - int(sec[i]) if hsx is not None else math.nan
                for h in HORIZONS_SEC:
                    pos = hsx is not None and hsx - int(sec[i]) <= h
                    r[f"reach_{target}_{h//60}m"] = bool(pos)
                    r[f"observable_{target}_{h//60}m"] = bool(pos or available[h])
            rows.append(r)
    out = pd.DataFrame(rows)
    if len(out): out = out.sort_values(["hit_50_second","direction","anchor_second"], kind="mergesort").reset_index(drop=True)
    return out, len(anchors)


def cluster_common_impulses(detections):
    _, pd, _, _ = stack()
    rows, idmap = [], {}
    for direction, g in detections.groupby("direction", sort=False):
        recs = sorted(g.to_dict("records"), key=lambda r:(int(r["hit_50_second"]), int(r["anchor_second"])))
        cluster, first50 = [], None
        def flush(items):
            if not items: return
            hits50 = [int(x["hit_50_second"]) for x in items]; anchors = [int(x["anchor_second"]) for x in items]
            rep = min(items, key=lambda x:(int(x["hit_50_second"]), -int(x["anchor_second"])))
            iid = stable_id("7K3_COMMON50", direction, min(hits50), max(hits50), min(anchors), max(anchors))
            o = {
                "impulse_id": iid, "direction": int(direction), "direction_label": "UP" if int(direction)>0 else "DOWN",
                "representative_anchor_second": int(rep["anchor_second"]), "representative_anchor_time": rep["anchor_time"],
                "representative_anchor_price": float(rep["anchor_price"]), "hit_50_second": int(rep["hit_50_second"]),
                "hit_50_time": rep["hit_50_time"], "parent_detection_count": len(items),
                "parent_hit_span_sec": max(hits50)-min(hits50), "parent_anchor_span_sec": max(anchors)-min(anchors),
            }
            for target in (50,75,100):
                thits = [int(x[f"hit_{target}_second"]) for x in items if math.isfinite(safe_float(x.get(f"hit_{target}_second")))]
                earliest = min(thits) if thits else math.nan
                o[f"hit_{target}_second"] = earliest; o[f"hit_{target}_time"] = iso_time(earliest)
                for h in (5,10,15):
                    o[f"reach_{target}_{h}m"] = any(bool(x[f"reach_{target}_{h}m"]) for x in items)
                    o[f"observable_{target}_{h}m"] = any(bool(x[f"observable_{target}_{h}m"]) for x in items)
            o["severity_15m"] = "100_PLUS" if o["reach_100_15m"] else "75_ONLY" if o["reach_75_15m"] else "50_ONLY" if o["observable_75_15m"] else "CENSORED"
            rows.append(o)
            for x in items: idmap[str(x["parent_detection_id"])] = iid
        for r in recs:
            h50 = int(r["hit_50_second"])
            if not cluster: cluster=[r]; first50=h50
            elif h50 - first50 <= COMMON_CLUSTER_HIT_SPAN_SEC: cluster.append(r)
            else: flush(cluster); cluster=[r]; first50=h50
        flush(cluster)
    impulses = pd.DataFrame(rows).sort_values(["hit_50_second","direction"], kind="mergesort").reset_index(drop=True)
    detections = detections.copy(); detections["impulse_id"] = detections["parent_detection_id"].astype(str).map(idmap)
    return impulses, detections


def load_physical(root):
    _, _, _, pq = stack(parquet=True); p = root / "physical_events.parquet"
    if not p.exists(): raise RuntimeError(f"Missing Phase-7A physical events: {p}")
    return pq.read_table(p).to_pandas()


def interval_end(r):
    for c in ("reversal_confirm_second","extreme_second"):
        v=safe_float(r.get(c))
        if math.isfinite(v): return int(v)
    return int(r["onset_second"])


def map_to_physical(impulses, physical):
    _, pd, _, _ = stack(); bydir={}
    for d,g in physical.groupby(pd.to_numeric(physical["direction"], errors="coerce")):
        recs=g.sort_values("onset_second").to_dict("records"); bydir[int(d)]=([int(r["onset_second"]) for r in recs],[interval_end(r) for r in recs],recs)
    out=[]
    for imp in impulses.to_dict("records"):
        hit=int(imp["hit_50_second"]); pack=bydir.get(int(imp["direction"])); ph=None
        if pack:
            starts,ends,recs=pack; j=bisect_right(starts,hit)-1
            for k in range(j,max(-1,j-4),-1):
                if k>=0 and starts[k]<=hit<=ends[k]: ph=recs[k]; break
        r=dict(imp); r["mapped_phase7a_event"]=ph is not None; r["phase7a_event_id"]=str(ph["event_id"]) if ph else None; out.append(r)
    out=pd.DataFrame(out); mapped=out["mapped_phase7a_event"].fillna(False).astype(bool)
    out["impulse_ordinal_in_physical_leg"]=math.nan; out["previous_impulse_50_hit_second"]=math.nan
    if mapped.any():
        m=out.loc[mapped].sort_values(["phase7a_event_id","hit_50_second","direction"],kind="mergesort").copy()
        m["impulse_ordinal_in_physical_leg"]=m.groupby("phase7a_event_id").cumcount()+1
        m["previous_impulse_50_hit_second"]=m.groupby("phase7a_event_id")["hit_50_second"].shift(1)
        out.loc[m.index,"impulse_ordinal_in_physical_leg"]=m["impulse_ordinal_in_physical_leg"]
        out.loc[m.index,"previous_impulse_50_hit_second"]=m["previous_impulse_50_hit_second"]
    out["is_reacceleration"]=pd.to_numeric(out["impulse_ordinal_in_physical_leg"],errors="coerce")>1
    return out.sort_values(["hit_50_second","direction"],kind="mergesort").reset_index(drop=True)


def load_k2(root):
    _, _, _, pq=stack(parquet=True); meta=root/"metadata/latest_summary.json"; cp=root/"candidates.parquet"
    if not meta.exists() or not cp.exists(): raise RuntimeError(f"Missing frozen 7K.2 outputs under {root}")
    md=json.loads(meta.read_text());
    if str(md.get("builder_version"))!=FROZEN_PHASE7K2_VERSION: raise RuntimeError(f"Need 7K.2 {FROZEN_PHASE7K2_VERSION}; found {md.get('builder_version')}")
    return md,pq.read_table(cp).to_pandas()


def load_old(root):
    _, pd, _, pq=stack(parquet=True); p=root/"candidates.parquet"
    if not p.exists(): raise RuntimeError(f"Missing old 7I.1 candidates: {p}")
    d=pq.read_table(p).to_pandas(); req={"candidate_id","candidate_second","origin_second","direction"}; miss=req-set(d.columns)
    if miss: raise RuntimeError(f"Old 7I.1 candidates missing {sorted(miss)}")
    d=d.copy(); d["policy_id"]=OLD_POLICY; d["candidate_id"]=d["candidate_id"].astype(str)
    if "release_second" not in d: d["release_second"]=math.nan
    return d


def map_candidates(candidates, impulses, policy, target, horizon):
    _, pd, _, _=stack(); col=f"reach_{target}_{horizon}m"; hitcol=f"hit_{target}_second"
    elig=impulses[impulses[col].fillna(False).astype(bool)].copy(); c=candidates[candidates["policy_id"].astype(str)==policy].copy()
    if elig.empty or c.empty: return pd.DataFrame()
    for x in ("candidate_second","origin_second","release_second"):
        if x not in c: c[x]=math.nan
        c[x]=pd.to_numeric(c[x],errors="coerce")
    bydir={int(d):g.sort_values("candidate_second").reset_index(drop=True) for d,g in c.groupby(pd.to_numeric(c["direction"],errors="coerce"))}
    used=set(); rows=[]
    for imp in elig.sort_values([hitcol,"direction"]).to_dict("records"):
        hit=int(imp[hitcol]); ordv=safe_float(imp.get("impulse_ordinal_in_physical_leg")); ordinal=int(ordv) if math.isfinite(ordv) else None
        pv=safe_float(imp.get("previous_impulse_50_hit_second")); prev=int(pv) if math.isfinite(pv) else None; chosen=None; g=bydir.get(int(imp["direction"]))
        if g is not None and len(g):
            secs=g["candidate_second"].astype(int).tolist(); j=bisect_right(secs,hit)-1
            while j>=0:
                r=g.iloc[j]; cid=str(r["candidate_id"]); cs=int(r["candidate_second"]); os_=int(r["origin_second"]); rv=safe_float(r.get("release_second")); rel=int(rv) if math.isfinite(rv) else None
                if hit-cs>CANDIDATE_MAPPING_LOOKBACK_SEC: break
                if cid in used: j-=1; continue
                ok=os_<=cs<=hit and (rel is None or os_<=rel<=cs)
                if ordinal is not None and ordinal>1: ok=ok and prev is not None and prev<os_
                if ok: chosen=r; break
                j-=1
        base={"policy_id":policy,"target_bps":target,"horizon_min":horizon,"impulse_id":str(imp["impulse_id"]),"phase7a_event_id":imp.get("phase7a_event_id"),"impulse_ordinal_in_physical_leg":ordinal if ordinal is not None else math.nan,"is_reacceleration":bool(ordinal is not None and ordinal>1),"previous_impulse_50_hit_second":prev if prev is not None else math.nan,"target_hit_second":hit,"target_hit_time":iso_time(hit)}
        if chosen is None:
            rows.append({**base,"candidate_id":None,"candidate_origin_second":math.nan,"candidate_release_second":math.nan,"candidate_release_available":False,"candidate_second":math.nan,"candidate_lead_to_target_sec":math.nan,"mapped":False})
        else:
            cid=str(chosen["candidate_id"]); used.add(cid); rv=safe_float(chosen.get("release_second")); relok=math.isfinite(rv)
            rows.append({**base,"candidate_id":cid,"candidate_origin_second":int(chosen["origin_second"]),"candidate_release_second":int(rv) if relok else math.nan,"candidate_release_available":relok,"candidate_second":int(chosen["candidate_second"]),"candidate_lead_to_target_sec":hit-int(chosen["candidate_second"]),"mapped":True})
    return pd.DataFrame(rows)


def definition_summary(imp):
    _, pd, _, _=stack(); rows=[]
    for t in (50,75,100):
        for h in (5,10,15):
            pos=imp[f"reach_{t}_{h}m"].fillna(False).astype(bool); obs=imp[f"observable_{t}_{h}m"].fillna(False).astype(bool)
            rows.append({"target_bps":t,"horizon_min":h,"common_impulses":len(imp),"observable_impulses":int(obs.sum()),"qualifying_impulses":int(pos.sum()),"qualifying_rate_of_observable":float(pos.sum()/obs.sum()) if obs.sum() else math.nan})
    return pd.DataFrame(rows)


def reacc_summary(imp):
    _, pd, _, _=stack(); g0=imp[imp["mapped_phase7a_event"].fillna(False).astype(bool)]; rows=[]
    for t in (50,75,100):
        for h in (5,10,15):
            g=g0[g0[f"reach_{t}_{h}m"].fillna(False).astype(bool)]; re=g[g["is_reacceleration"].fillna(False).astype(bool)]
            rows.append({"target_bps":t,"horizon_min":h,"mapped_qualifying_impulses":len(g),"first_impulses":len(g)-len(re),"reacceleration_impulses":len(re),"reacceleration_share":float(len(re)/len(g)) if len(g) else math.nan})
    return pd.DataFrame(rows)


def coverage_reports(mapping):
    _, pd, _, _=stack(); rows=[]; ordrows=[]; late=[]
    for (p,t,h),g in mapping.groupby(["policy_id","target_bps","horizon_min"],sort=False):
        re=g["is_reacceleration"].fillna(False).astype(bool); mp=g["mapped"].fillna(False).astype(bool)
        rows.append({"policy_id":p,"target_bps":int(t),"horizon_min":int(h),"eligible_impulses":len(g),"covered_impulses":int(mp.sum()),"coverage_rate":float(mp.mean()),"eligible_reaccelerations":int(re.sum()),"covered_reaccelerations":int((mp&re).sum()),"reacceleration_coverage_rate":float((mp&re).sum()/re.sum()) if re.sum() else math.nan})
        for o,og in g.groupby("impulse_ordinal_in_physical_leg",dropna=False):
            mm=og["mapped"].fillna(False).astype(bool); ordrows.append({"policy_id":p,"target_bps":int(t),"horizon_min":int(h),"impulse_ordinal_in_physical_leg":o,"eligible_impulses":len(og),"covered_impulses":int(mm.sum()),"coverage_rate":float(mm.mean())})
        mg=g[mp]
        if len(mg):
            lead=pd.to_numeric(mg["candidate_lead_to_target_sec"],errors="coerce"); late.append({"policy_id":p,"target_bps":int(t),"horizon_min":int(h),"mapped_impulses":len(mg),"median_candidate_lead_sec":float(lead.median()),"p25_candidate_lead_sec":float(lead.quantile(.25)),"p75_candidate_lead_sec":float(lead.quantile(.75)),"share_within_30s_of_target":float((lead<=30).mean()),"share_within_60s_of_target":float((lead<=60).mean())})
    return pd.DataFrame(rows),pd.DataFrame(ordrows),pd.DataFrame(late)


def nesting(imp):
    _, pd, _, _=stack(); rows=[]; checks=[]
    for h in (5,10,15): checks += [(f"100_{h}_subset_75",f"reach_100_{h}m",f"reach_75_{h}m"),(f"75_{h}_subset_50",f"reach_75_{h}m",f"reach_50_{h}m")]
    for t in (50,75,100): checks += [(f"{t}_5_subset_10",f"reach_{t}_5m",f"reach_{t}_10m"),(f"{t}_10_subset_15",f"reach_{t}_10m",f"reach_{t}_15m")]
    for name,ch,par in checks:
        c=imp[ch].fillna(False).astype(bool); p=imp[par].fillna(False).astype(bool); v=int((c&~p).sum()); rows.append({"check":name,"child":ch,"parent":par,"subset_violation_count":v,"status":"PASS" if v==0 else "FAIL"})
    return pd.DataFrame(rows)


def multiplicity(imp):
    _, pd, _, _=stack(); g=imp[imp["mapped_phase7a_event"].fillna(False).astype(bool)]; rows=[]
    for eid,x in g.groupby("phase7a_event_id"):
        rows.append({"phase7a_event_id":str(eid),"common_50_impulses":len(x),"reaccelerations_50":int(x["is_reacceleration"].sum()),"impulses_reaching_75_15m":int(x["reach_75_15m"].sum()),"impulses_reaching_100_15m":int(x["reach_100_15m"].sum())})
    return pd.DataFrame(rows).sort_values("common_50_impulses",ascending=False) if rows else pd.DataFrame()


def invariants(det,imp,mapping,nest,anchors):
    _, pd, _, _=stack(); rows=[]
    def add(n,ok,o,e): rows.append({"invariant":n,"status":"PASS" if ok else "FAIL","observed":o,"expected":e})
    add("anchor_grid_nonempty",anchors>0,anchors,">0"); add("parent_detections_nonempty",len(det)>0,len(det),">0"); add("common_impulses_nonempty",len(imp)>0,len(imp),">0")
    add("parent_ids_unique",not det["parent_detection_id"].duplicated().any(),int(det["parent_detection_id"].duplicated().sum()),0); add("impulse_ids_unique",not imp["impulse_id"].duplicated().any(),int(imp["impulse_id"].duplicated().sum()),0)
    add("all_parent_detections_mapped",det["impulse_id"].notna().all(),int(det["impulse_id"].isna().sum()),0); nf=int((nest["status"]=="FAIL").sum()); add("severity_nesting",nf==0,nf,0)
    mg=mapping[mapping["mapped"].fillna(False).astype(bool)]; dup=int(mg.duplicated(["policy_id","target_bps","horizon_min","candidate_id"]).sum()); add("candidate_one_to_one_per_policy_target_horizon",dup==0,dup,0)
    re=mg[mg["is_reacceleration"].fillna(False).astype(bool)]
    if len(re):
        prev=pd.to_numeric(re["previous_impulse_50_hit_second"],errors="coerce"); origin=pd.to_numeric(re["candidate_origin_second"],errors="coerce"); cand=pd.to_numeric(re["candidate_second"],errors="coerce"); hit=pd.to_numeric(re["target_hit_second"],errors="coerce"); rel=pd.to_numeric(re["candidate_release_second"],errors="coerce"); ra=re["candidate_release_available"].fillna(False).astype(bool); ok=(prev<origin)&(origin<=cand)&(cand<=hit)&(~ra|((origin<=rel)&(rel<=cand))); add("fresh_reaccel_mapping",bool(ok.all()),int((~ok).sum()),0)
    return pd.DataFrame(rows)


def findings(status, defs, reacc, cov):
    lines=["# Phase 7K.3 — Expansion-impulse severity threshold audit","",f"- Builder: **{BUILDER_VERSION}**",f"- Status: **{status}**","- Common parent: **50 bps**","- Severity labels: **50 / 75 / 100 bps**","- Horizons: **5 / 10 / 15 minutes**","","## 15-minute impulse counts",""]
    for t in (50,75,100):
        d=defs[(defs.target_bps==t)&(defs.horizon_min==15)].iloc[0]; r=reacc[(reacc.target_bps==t)&(reacc.horizon_min==15)].iloc[0]; lines.append(f"- {t}bp: {int(d.qualifying_impulses):,} impulses; {int(r.reacceleration_impulses):,} re-accelerations.")
    lines += ["","## Target-aware re-acceleration coverage (15m)",""]
    for t in (50,75,100):
        for p in ("TEST_COOLDOWN_0S",OLD_POLICY):
            x=cov[(cov.policy_id==p)&(cov.target_bps==t)&(cov.horizon_min==15)]
            if len(x): lines.append(f"- {p} @ {t}bp: {100*safe_float(x.iloc[0].reacceleration_coverage_rate):.1f}% re-acceleration coverage.")
    lines += ["","## Interpretation discipline","","- 75/100 are labels on the common 50-bp impulse; they are not separately clustered events.","- Higher-target coverage can be higher because the detector has more time to fire before 75/100.","- This phase does not automatically choose the final severity threshold.","- Do not run another raw-depth replay until severity and causal-candidate ontology are frozen.",""]
    return "\n".join(lines)


def self_test():
    _, pd, _, _=stack(); rows=[]
    def rec(pid,a,h50,h75,h100):
        r={"parent_detection_id":pid,"direction":1,"direction_label":"UP","anchor_second":a,"anchor_time":iso_time(a),"anchor_price":100.0,"max_observable_horizon_sec":900,"hit_50_second":h50,"hit_50_time":iso_time(h50),"seconds_to_50":h50-a,"hit_75_second":h75 if h75 else math.nan,"hit_75_time":iso_time(h75),"seconds_to_75":h75-a if h75 else math.nan,"hit_100_second":h100 if h100 else math.nan,"hit_100_time":iso_time(h100),"seconds_to_100":h100-a if h100 else math.nan}
        for h in (5,10,15):
            r[f"available_{h}m"]=True
            for t,hit in ((50,h50),(75,h75),(100,h100)):
                pos=hit is not None and hit-a<=h*60; r[f"reach_{t}_{h}m"]=pos; r[f"observable_{t}_{h}m"]=True
        return r
    rows=[rec("a1",100,160,200,240),rec("a2",110,170,210,250),rec("b1",500,600,None,None)]
    imp,lab=cluster_common_impulses(pd.DataFrame(rows)); assert len(imp)==2; assert int(imp.reach_50_15m.sum())==2; assert int(imp.reach_75_15m.sum())==1; assert int(imp.reach_100_15m.sum())==1; n=nesting(imp); assert int(n.subset_violation_count.sum())==0; assert lab.impulse_id.notna().all()
    print("SELF-TEST PASS"); print("Validated one common 50-bp parent, bounded clustering, and nested 50/75/100 labels across 5/10/15m.")


def parse_args(argv: Sequence[str] | None=None):
    p=argparse.ArgumentParser(description="Phase 7K.3 expansion-impulse severity threshold audit")
    p.add_argument("--symbol",default=DEFAULT_SYMBOL); p.add_argument("--start-date",default="2026-05-18"); p.add_argument("--end-date",default="2026-07-18")
    p.add_argument("--phase7a-script",type=Path,default=DEFAULT_PHASE7A_SCRIPT); p.add_argument("--phase7k1-script",type=Path,default=DEFAULT_PHASE7K1_SCRIPT); p.add_argument("--execution-root",type=Path,default=DEFAULT_EXECUTION_ROOT); p.add_argument("--phase7a-root",type=Path,default=DEFAULT_PHASE7A_ROOT); p.add_argument("--phase7i1-root",type=Path,default=DEFAULT_PHASE7I1_ROOT); p.add_argument("--phase7k2-root",type=Path,default=DEFAULT_PHASE7K2_ROOT); p.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT_ROOT); p.add_argument("--report-root",type=Path,default=DEFAULT_REPORT_ROOT); p.add_argument("--max-trade-stale-sec",type=int,default=DEFAULT_MAX_TRADE_STALE_SEC); p.add_argument("--self-test",action="store_true"); return p.parse_args(argv)


def main(argv: Sequence[str] | None=None):
    args=parse_args(argv)
    if args.self_test: self_test(); return 0
    np,pd,_,_=stack(); start=date.fromisoformat(args.start_date); end=date.fromisoformat(args.end_date)
    if start>end: raise SystemExit("--start-date must be <= --end-date")
    if args.symbol!=DEFAULT_SYMBOL: raise SystemExit(f"Phase 7K.3 frozen to {DEFAULT_SYMBOL}")
    if args.max_trade_stale_sec!=DEFAULT_MAX_TRADE_STALE_SEC: raise SystemExit(f"Phase 7K.3 frozen max_trade_stale_sec={DEFAULT_MAX_TRADE_STALE_SEC}")
    p7a=load_module(args.phase7a_script,"phase7a_for_7k3"); p7k1=load_module(args.phase7k1_script,"phase7k1_for_7k3")
    if str(getattr(p7a,"BUILDER_VERSION",""))!=FROZEN_PHASE7A_VERSION: raise RuntimeError("Wrong frozen Phase-7A version")
    if str(getattr(p7k1,"BUILDER_VERSION",""))!=FROZEN_PHASE7K1_VERSION: raise RuntimeError("Wrong frozen Phase-7K.1 version")
    print("Building Phase 7K.3 expansion-impulse severity threshold audit..."); print(f"Builder version       : {BUILDER_VERSION}"); print(f"Dates                 : {start} to {end}"); print("Common parent         : 50 bp"); print("Severity              : 50 / 75 / 100 bp"); print("Horizons              : 5 / 10 / 15m"); print("Raw replay            : NONE")
    print("[1/9] Loading Phase-6A execution prices..."); days=[]; d=start-timedelta(days=1)
    while d<=end+timedelta(days=1):
        p=p7a.execution_day_path(args.execution_root,args.symbol,d)
        if p.exists(): days.append(d)
        d+=timedelta(days=1)
    frame=p7a.load_execution_range(args.execution_root,args.symbol,days); sec,price,source=p7a.build_catalog_price(frame,args.max_trade_stale_sec,"last"); print(f"      rows={len(sec):,} source={source}")
    print("[2/9] Building independent 50-bp parent detections..."); det,anchor_count=build_parent_detections(sec,price,start,end); print(f"      anchors={anchor_count:,} detections={len(det):,}")
    print("[3/9] Clustering common 50-bp impulses..."); imp,det=cluster_common_impulses(det); print(f"      common impulses={len(imp):,}")
    print("[4/9] Mapping to Phase-7A physical legs..."); imp=map_to_physical(imp,load_physical(args.phase7a_root)); print(f"      mapped={int(imp.mapped_phase7a_event.sum()):,}/{len(imp):,} reaccels={int(imp.is_reacceleration.sum()):,}")
    print("[5/9] Loading frozen candidate universes..."); k2meta,k2=load_k2(args.phase7k2_root); old=load_old(args.phase7i1_root); print(f"      7K.2 rows={len(k2):,} old7I.1={len(old):,}")
    print("[6/9] Target-aware mapping..."); maps=[]
    for t in (50,75,100):
        for h in (5,10,15):
            for p in K2_POLICIES: maps.append(map_candidates(k2,imp,p,t,h))
            maps.append(map_candidates(old,imp,OLD_POLICY,t,h))
    mapping=pd.concat([m for m in maps if len(m)],ignore_index=True)
    print("[7/9] Building reports..."); defs=definition_summary(imp); reacc=reacc_summary(imp); cov,ordcov,late=coverage_reports(mapping); nest=nesting(imp); multi=multiplicity(imp)
    print("[8/9] Running invariants..."); inv=invariants(det,imp,mapping,nest,anchor_count); failed=inv[inv.status=="FAIL"]; status="PASS" if failed.empty else "FAIL"
    summary={"builder_version":BUILDER_VERSION,"schema_version":SCHEMA_VERSION,"schema_name":SCHEMA_NAME,"status":status,"built_at_utc":datetime.now(timezone.utc).replace(microsecond=0).isoformat(),"symbol":args.symbol,"start_date":str(start),"end_date":str(end),"price_source":source,"common_parent_bps":COMMON_PARENT_BPS,"severity_thresholds_bps":list(SEVERITY_THRESHOLDS_BPS),"horizons_sec":list(HORIZONS_SEC),"anchor_count":anchor_count,"parent_detection_count":len(det),"common_impulse_count":len(imp),"mapped_phase7a_impulses":int(imp.mapped_phase7a_event.sum()),"common_50_reaccelerations":int(imp.is_reacceleration.sum()),"frozen_phase7k2_builder_version":k2meta.get("builder_version"),"scientific_state":"SEVERITY_AUDIT_NOT_YET_FROZEN","failed_invariants":len(failed)}
    config={**summary,"anchor_cadence_sec":ANCHOR_CADENCE_SEC,"cluster_hit_span_sec":COMMON_CLUSTER_HIT_SPAN_SEC,"candidate_mapping_lookback_sec":CANDIDATE_MAPPING_LOOKBACK_SEC,"candidate_policies":list(ALL_POLICIES),"reacceleration_rule":"second-or-later common 50bp impulse inside same Phase-7A physical leg","fresh_mapping_rule":"previous common 50 hit < candidate origin <= candidate <= current target hit; release ordering when available"}
    print("[9/9] Writing outputs..."); atomic_parquet(det,args.output_root/"parent_detections.parquet"); atomic_parquet(imp,args.output_root/"common_impulses.parquet"); atomic_parquet(mapping,args.output_root/"candidate_severity_mapping.parquet"); atomic_json(args.output_root/"metadata/latest_summary.json",summary); atomic_json(args.output_root/"metadata/run_config.json",config); atomic_csv(defs,args.report_root/"severity_definition_summary.csv"); atomic_csv(reacc,args.report_root/"severity_reacceleration_summary.csv"); atomic_csv(cov,args.report_root/"severity_candidate_coverage.csv"); atomic_csv(ordcov,args.report_root/"severity_candidate_coverage_by_ordinal.csv"); atomic_csv(late,args.report_root/"severity_candidate_lateness.csv"); atomic_csv(nest,args.report_root/"severity_nesting_consistency.csv"); atomic_csv(multi,args.report_root/"severity_physical_leg_multiplicity.csv"); atomic_csv(inv,args.report_root/"severity_invariants.csv"); atomic_text(args.report_root/"severity_findings.md",findings(status,defs,reacc,cov))
    print(); print(f"Phase 7K.3 status: {status}"); print(f"Common 50bp impulses: {len(imp):,}")
    for t in (50,75,100):
        d=defs[(defs.target_bps==t)&(defs.horizon_min==15)].iloc[0]; r=reacc[(reacc.target_bps==t)&(reacc.horizon_min==15)].iloc[0]; print(f"{t:3d}bp <=15m: {int(d.qualifying_impulses):,} impulses; {int(r.reacceleration_impulses):,} reaccels")
    print(f"Failed invariants: {len(failed)}"); print(f"Reports: {args.report_root}")
    if len(failed): print(failed.to_string(index=False)); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
