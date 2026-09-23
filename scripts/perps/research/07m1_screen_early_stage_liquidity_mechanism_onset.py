#!/usr/bin/env python3
"""
Phase 7M.1 — Early-stage liquidity-mechanism onset screen
=========================================================

Question
--------
Inside the frozen causal opportunities that eventually reached the 35-bp
candidate point, was the opposing-liquidity-failure signature already visible
at 10/15/20/25/30 bps, and how much movement/time remained then?

IMPORTANT LIMITATION
--------------------
Every row in this phase belongs to a candidate that later reached 35 bps.
Therefore this is a retrospective MECHANISM-ONSET diagnostic, NOT a prospective
early-entry test. If an earlier stage looks promising, a later phase must build
an early-stage universe containing attempts that fail before 35 bps.

Price geometry
--------------
Stage first passages are reconstructed from the same frozen Phase-6A
LAST_TRADE_FFILL_MAX_5S price source used to generate Phase-7K.2 candidates.
Required invariants:
  HIT_10 == frozen release_second
  HIT_35 == frozen candidate_second

Cheap causal proxy mechanics
----------------------------
At each stage t, use Phase-6B rows t..t+3 (available after t+3 closes):
  1) opposing aggregate-5bp peak depth loss over 3s
  2) opposing aggregate-5bp retained depth ratio at +3s (oriented low=failure)
  3) opposing peak 1s withdrawal fraction over t+1..t+3
  4) support-minus-opposing aggregate retained ratio (diagnostic)

These are aggregate near-touch screening proxies, NOT the exact frozen-level
queue-survival mechanics measured by 7E/7L.1.

Controls / feasibility
----------------------
For each stage and feature separately:
  exposure = high vs low oriented proxy tercile, defined outcome-blind
  exact match = candidate_date x direction
  continuous controls = 3s directional momentum + log directional aggressive
                        quote proxy
  greedy 1:1 matching without replacement
  caliper <=0.75 z each control; <=1.00 z Euclidean
  outcome-blind trimming until both |SMD| <=0.20

Full-population feature feasibility:
  balanced pairs >=150 AND both |SMD|<=0.20
Stage exact-replay feasibility:
  >=2 of 3 primary proxy features feasible
Re-acceleration diagnostic feature feasibility:
  balanced pairs >=50 AND both |SMD|<=0.20

Only AFTER stage inputs, pair manifests, balance and feasibility are frozen are
future 75/100 outcomes loaded from Phase-7L.1.

No raw depth replay. No raw trade replay. No ML. No composite score.
"""
from __future__ import annotations

import argparse, gc, importlib.util, json, math, os, sys, tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = pq = None

BUILDER_VERSION = "1.0.1"
SCHEMA_VERSION = 1
SCHEMA_NAME = "early_stage_liquidity_onset_screen_v1"
SYMBOL = "BTCUSDT"
FROZEN_PHASE7A_VERSION = "1.0.0"
FROZEN_PHASE7K2_VERSION = "1.0.0"
FROZEN_PHASE6B_VERSION = "1.0.3"
FROZEN_PHASE7L1_VERSION = "1.0.1"
FROZEN_PHASE7L3_VERSION = "1.0.0"
POLICY = "TEST_COOLDOWN_60S"
EXPECTED_CANDIDATES = 4037
EXPECTED_DAYS = 62
EXPECTED_REACCEL = 1336
MAX_TRADE_STALE_SEC = 5
STAGES = (10, 15, 20, 25, 30, 35)
OBS_SEC = 3
OUTCOME_HORIZONS = (300, 600, 900, 1800)
PRIMARY_HORIZON = 900
DISCOVERY_END = "2026-06-17"
VALIDATION_START = "2026-06-18"
MATCH_Z = 0.75
MATCH_DIST = 1.00
BALANCE_SMD = 0.20
MIN_FULL_PAIRS = 150
MIN_REACCEL_PAIRS = 50
MIN_SPLIT_PAIRS = 60
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 20260831
REACCEL_FLAG = "is_reaccel__REARM_ANY_PRIOR50_30M_NO_OPPOSITE50"

PRIMARY_FEATURES = (
    ("opposing__aggregate5bp_peak_loss_share_3s", +1, "OPPOSING_AGG5_PEAK_LOSS_3S"),
    ("opposing__aggregate5bp_retained_ratio_3s", -1, "OPPOSING_AGG5_LOW_RETAINED_3S"),
    ("opposing__peak_1s_withdrawal_frac_3s", +1, "OPPOSING_PEAK_1S_WITHDRAWAL_3S"),
)
DIAG_FEATURES = (
    ("support_minus_opposing__aggregate5bp_retained_ratio_3s", +1, "SUPPORT_MINUS_OPPOSING_AGG5_RETAINED_3S"),
)
FEATURES = PRIMARY_FEATURES + DIAG_FEATURES
P6B_COLS = (
    "symbol", "receive_second", "receive_date", "primitive_input_ready", "book_valid",
    "bid_depth_quote_5bps", "ask_depth_quote_5bps",
    "bid_withdrawal_frac_1s", "ask_withdrawal_frac_1s",
    "quote_volume_5s", "signed_quote_flow_5s", "aggressive_flow_imbalance_5s",
)

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[3] if len(SCRIPT.parents) >= 4 else Path.cwd()
DEF_7A = ROOT / "scripts/perps/research/07a_build_large_move_event_catalog.py"
DEF_EXEC = ROOT / "data/derived/perps/execution_foundation_v1"
DEF_K2 = ROOT / "data/derived/perps/research/causal_rearm_timing_audit_v1"
DEF_6B = ROOT / "data/derived/perps/research/liquidity_event_primitives_v1"
DEF_L1 = ROOT / "data/derived/perps/research/repeatable_candidate_participant_interaction_v1"
DEF_L3 = ROOT / "data/derived/perps/research/causal_reacceleration_context_audit_v1"
DEF_OUT = ROOT / "data/derived/perps/research" / SCHEMA_NAME
DEF_REPORT = ROOT / "reports/perps_early_stage_liquidity_onset_screen"


def arrow():
    if pa is None or pq is None:
        raise SystemExit("Phase 7M.1 requires pyarrow in the project .venv")


def now(): return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
def sf(x):
    try: v=float(x); return v if math.isfinite(v) else math.nan
    except Exception: return math.nan


def bs(s):
    if pd.api.types.is_bool_dtype(s): return s.fillna(False).astype(bool)
    return s.astype(str).str.lower().map({"true":True,"false":False,"1":True,"0":False}).fillna(False).astype(bool)


def req(df, cols, name):
    miss=[c for c in cols if c not in df.columns]
    if miss: raise RuntimeError(f"{name} missing columns: {miss}")


def module(path, name):
    if not path.exists(): raise RuntimeError(f"Missing frozen script: {path}")
    spec=importlib.util.spec_from_file_location(name,str(path))
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot import {path}")
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m


def meta(root, builder, label):
    p=root/"metadata/latest_summary.json"
    if not p.exists(): raise RuntimeError(f"Missing {label} metadata: {p}")
    m=json.loads(p.read_text(encoding="utf-8"))
    if str(m.get("builder_version"))!=builder:
        raise RuntimeError(
            f"{label} builder expected {builder}; found {m.get('builder_version')!r}. "
            "This phase also validates the concrete parquet columns it consumes."
        )
    if str(m.get("status"))!="PASS": raise RuntimeError(f"{label} status must PASS")
    return m


def atomic_text(path,text):
    path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix="tmp7m1-",suffix=path.suffix,dir=str(path.parent)); os.close(fd)
    try: Path(tmp).write_text(text,encoding="utf-8"); os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_json(path,obj): atomic_text(path,json.dumps(obj,indent=2,sort_keys=True,default=str)+"\n")
def atomic_csv(df,path):
    path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix="tmp7m1-",suffix=".csv",dir=str(path.parent)); os.close(fd)
    try: df.to_csv(tmp,index=False); os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_parquet(df,path):
    arrow(); path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix="tmp7m1-",suffix=".parquet",dir=str(path.parent)); os.close(fd)
    try: pq.write_table(pa.Table.from_pandas(df,preserve_index=False),tmp,compression="zstd"); os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def dbps(price,anchor,direction):
    p=np.asarray(price,dtype=float)
    return (p/anchor-1)*10000 if direction>0 else (1-p/anchor)*10000


def qbin(s,n=3):
    x=pd.to_numeric(s,errors="coerce").replace([np.inf,-np.inf],np.nan); out=pd.Series(pd.NA,index=s.index,dtype="Int64"); mask=x.notna()
    if mask.any(): out.loc[mask]=np.ceil(x.loc[mask].rank(method="first",pct=True)*n).clip(1,n).astype(int).astype("Int64")
    return out


def zscore(s):
    x=pd.to_numeric(s,errors="coerce").astype(float); mu=float(x.mean()); sd=float(x.std(ddof=0))
    return (x-mu)/sd if math.isfinite(sd) and sd>0 else pd.Series(np.nan,index=s.index,dtype=float)


def smd(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
    if not len(a) or not len(b): return math.nan
    va=float(np.var(a,ddof=1)) if len(a)>1 else 0; vb=float(np.var(b,ddof=1)) if len(b)>1 else 0; den=math.sqrt(max(0,.5*(va+vb)))
    if den<=0: return 0.0 if math.isclose(float(a.mean()),float(b.mean())) else math.nan
    return float((a.mean()-b.mean())/den)


def load_candidates(root):
    m=meta(root,FROZEN_PHASE7K2_VERSION,"Phase 7K.2"); p=root/"candidates.parquet"; d=pd.read_parquet(p); d=d[d.policy_id.astype(str)==POLICY].copy()
    req(d,["candidate_id","policy_id","candidate_date","direction","direction_label","origin_kind","origin_second","origin_price","release_second","candidate_second","candidate_price","candidate_displacement_bps","reset_threshold_bps","release_threshold_bps","candidate_threshold_bps","cooldown_sec"],"7K.2 candidates")
    if len(d)!=EXPECTED_CANDIDATES or d.candidate_id.duplicated().any(): raise RuntimeError(f"Frozen candidate contract mismatch: rows={len(d)}")
    if d.candidate_date.astype(str).nunique()!=EXPECTED_DAYS: raise RuntimeError("Frozen candidate day count mismatch")
    return d.reset_index(drop=True),m


def stage_geometry(cand,p7a,exec_root,symbol):
    start=datetime.fromtimestamp(int(cand.origin_second.min()),tz=timezone.utc).date(); end=datetime.fromtimestamp(int(cand.candidate_second.max()),tz=timezone.utc).date(); days=[]; d=start-timedelta(days=1)
    while d<=end+timedelta(days=1):
        p=p7a.execution_day_path(exec_root,symbol,d)
        if p.exists(): days.append(d)
        d+=timedelta(days=1)
    frame=p7a.load_execution_range(exec_root,symbol,days); sec,price,source=p7a.build_catalog_price(frame,MAX_TRADE_STALE_SEC,"last")
    rows=[]; bad10=bad35=missing=nonmono=0
    for r in cand.to_dict("records"):
        os_=int(r["origin_second"]); cs=int(r["candidate_second"]); op=float(r["origin_price"]); direction=int(r["direction"]); li=int(np.searchsorted(sec,os_)); ri=int(np.searchsorted(sec,cs,side="right"))
        if li>=len(sec) or int(sec[li])!=os_: raise RuntimeError(f"Missing exact origin second {r['candidate_id']}")
        ps=sec[li:ri]; pp=price[li:ri]; mv=dbps(pp,op,direction); hits={}
        for st in STAGES:
            ix=np.flatnonzero(np.isfinite(mv)&(mv+1e-12>=st)); hits[st]=int(ps[int(ix[0])]) if len(ix) else None; missing += int(hits[st] is None)
        bad10+=int(hits[10]!=int(r["release_second"])); bad35+=int(hits[35]!=cs)
        if any(v is None for v in hits.values()): continue
        nonmono += int(any(hits[STAGES[i]]>hits[STAGES[i+1]] for i in range(len(STAGES)-1)))
        for st in STAGES:
            ss=hits[st]; ds=ss+OBS_SEC; si=int(np.searchsorted(sec,ss)); di=int(np.searchsorted(sec,ds)); valid=si<len(sec) and di<len(sec) and int(sec[si])==ss and int(sec[di])==ds and np.isfinite(price[si]) and np.isfinite(price[di]) and price[si]>0 and price[di]>0
            sp=float(price[si]) if valid else math.nan; dp=float(price[di]) if valid else math.nan; mom=float(dbps(dp,sp,direction)) if valid else math.nan; disp=float(dbps(dp,op,direction)) if valid else math.nan
            rows.append({"candidate_id":str(r["candidate_id"]),"policy_id":str(r["policy_id"]),"candidate_date":str(r["candidate_date"]),"direction":direction,"direction_label":str(r["direction_label"]),"origin_kind":str(r["origin_kind"]),"origin_second":os_,"origin_price":op,"release_second":int(r["release_second"]),"candidate_second":cs,"stage_bps":st,"stage_second":ss,"stage_date":datetime.fromtimestamp(ss,tz=timezone.utc).date().isoformat(),"decision_second":ds,"proxy_available_after_second":ds+1,"stage_price":sp,"decision_price":dp,"decision_price_valid":valid,"directional_mid_move_bps_3s_proxy":mom,"decision_displacement_from_origin_bps":disp,"seconds_stage_to_35":cs-ss,"nominal_remaining_stage_to_35_bps":35-st,"nominal_remaining_stage_to_75_bps":75-st,"nominal_remaining_stage_to_100_bps":100-st,"price_source":source,"future35_conditioned_universe":True})
    if bad10 or bad35 or missing or nonmono: raise RuntimeError(f"Stage geometry failure: hit10={bad10} hit35={bad35} missing={missing} nonmono={nonmono}")
    del frame; gc.collect(); return pd.DataFrame(rows),source


def p6b_path(root,symbol,d): return root/"primitives"/f"symbol={symbol}"/f"date={d.isoformat()}"/f"part-{d.isoformat()}.parquet"
def read_p6b(root,symbol,days):
    arrow(); parts=[]
    for d in sorted(set(days)):
        p=p6b_path(root,symbol,d)
        if not p.exists(): continue
        names=set(pq.read_schema(p).names); miss=[c for c in P6B_COLS if c not in names]
        if miss: raise RuntimeError(f"Phase6B missing {miss}: {p}")
        parts.append(pq.read_table(p,columns=list(P6B_COLS)).to_pandas())
    if not parts: return pd.DataFrame(columns=P6B_COLS)
    x=pd.concat(parts,ignore_index=True); x=x[x.symbol.astype(str).str.upper()==symbol.upper()].copy(); x.receive_second=pd.to_numeric(x.receive_second,errors="raise").astype(np.int64); return x.sort_values("receive_second",kind="mergesort").drop_duplicates("receive_second",keep="last").reset_index(drop=True)


def stage_proxies(geom,p6b_root,symbol):
    parts=[]; groups=list(geom.groupby("stage_date",sort=True))
    for n,(dtxt,g) in enumerate(groups,1):
        d=date.fromisoformat(str(dtxt)); prim=read_p6b(p6b_root,symbol,[d,d+timedelta(days=1)]); lookup={int(s):i for i,s in enumerate(prim.receive_second.tolist())}; rows=[]
        for r in g.to_dict("records"):
            out=dict(r); ss=int(r["stage_second"]); need=[ss,ss+1,ss+2,ss+3]
            if any(s not in lookup for s in need): out.update(proxy_valid=False,proxy_invalid_reason="MISSING_EXACT_PHASE6B_SECOND"); rows.append(out); continue
            w=prim.iloc[[lookup[s] for s in need]].copy()
            if not (bs(w.primitive_input_ready).all() and bs(w.book_valid).all()): out.update(proxy_valid=False,proxy_invalid_reason="PHASE6B_NOT_READY_OR_BOOK_INVALID"); rows.append(out); continue
            up=int(r["direction"])>0; oc="ask_depth_quote_5bps" if up else "bid_depth_quote_5bps"; sc="bid_depth_quote_5bps" if up else "ask_depth_quote_5bps"; wc="ask_withdrawal_frac_1s" if up else "bid_withdrawal_frac_1s"; opp=pd.to_numeric(w[oc],errors="coerce").to_numpy(float); sup=pd.to_numeric(w[sc],errors="coerce").to_numpy(float)
            if not (np.isfinite(opp).all() and np.isfinite(sup).all() and opp[0]>0 and sup[0]>0 and (opp>=0).all() and (sup>=0).all()): out.update(proxy_valid=False,proxy_invalid_reason="INVALID_AGGREGATE_DEPTH"); rows.append(out); continue
            oret=float(opp[-1]/opp[0]); sret=float(sup[-1]/sup[0]); peak=float(max(0,1-float(np.min(opp[1:]))/opp[0])); wd=pd.to_numeric(w.iloc[1:][wc],errors="coerce").to_numpy(float); pwithdraw=float(np.nanmax(wd)) if np.isfinite(wd).any() else math.nan; qv=sf(w.iloc[-1].quote_volume_5s); signed=sf(w.iloc[-1].signed_quote_flow_5s); direction=int(r["direction"]); aggr=max(0,.5*(qv+direction*signed)) if math.isfinite(qv) and math.isfinite(signed) else math.nan; imb=sf(w.iloc[-1].aggressive_flow_imbalance_5s)
            out.update(proxy_valid=True,proxy_invalid_reason=None,opposing__aggregate5bp_start_quote=float(opp[0]),opposing__aggregate5bp_end_quote=float(opp[-1]),opposing__aggregate5bp_peak_loss_share_3s=peak,opposing__aggregate5bp_retained_ratio_3s=oret,opposing__peak_1s_withdrawal_frac_3s=pwithdraw,support__aggregate5bp_retained_ratio_3s=sret,support_minus_opposing__aggregate5bp_retained_ratio_3s=sret-oret,directional_aggressive_quote_5s_proxy=aggr,directional_aggressive_flow_imbalance_5s_proxy=direction*imb if math.isfinite(imb) else math.nan)
            rows.append(out)
        parts.append(pd.DataFrame(rows)); print(f"      Phase6B [{n:02d}/{len(groups):02d}] {dtxt}: stage rows={len(g):,}"); del prim; gc.collect()
        if pa is not None:
            try: pa.default_memory_pool().release_unused()
            except Exception: pass
    out=pd.concat(parts,ignore_index=True); out["__log_aggression"]=np.log1p(pd.to_numeric(out.directional_aggressive_quote_5s_proxy,errors="coerce").clip(lower=0))
    for st in STAGES:
        mask=out.stage_bps.astype(int)==st; out.loc[mask,"__momentum_z"]=zscore(out.loc[mask,"directional_mid_move_bps_3s_proxy"]); out.loc[mask,"__aggression_z"]=zscore(out.loc[mask,"__log_aggression"])
        for f,sign,_ in FEATURES:
            ori=pd.to_numeric(out.loc[mask,f],errors="coerce")*sign; out.loc[mask,f"__oriented__{f}"]=ori; out.loc[mask,f"__tercile__{f}"]=qbin(ori).astype(float)
    return out


def attach_reaccel(x,root):
    m=meta(root,FROZEN_PHASE7L3_VERSION,"Phase 7L.3"); lab=pd.read_parquet(root/"candidate_context_labels.parquet"); req(lab,["candidate_id",REACCEL_FLAG],"7L.3 labels"); lab.candidate_id=lab.candidate_id.astype(str); n=int(bs(lab[REACCEL_FLAG]).sum())
    if n!=EXPECTED_REACCEL: raise RuntimeError(f"Expected {EXPECTED_REACCEL} reaccel labels; found {n}")
    y=x.merge(lab[["candidate_id",REACCEL_FLAG]],on="candidate_id",how="left",validate="many_to_one"); y[REACCEL_FLAG]=bs(y[REACCEL_FLAG]); return y,m


def greedy(high,low,feature,st,pop):
    if high.empty or low.empty: return []
    hi=high.reset_index(drop=True); lo=low.reset_index(drop=True); hm=hi.__momentum_z.to_numpy(float); ha=hi.__aggression_z.to_numpy(float); lm=lo.__momentum_z.to_numpy(float); la=lo.__aggression_z.to_numpy(float); choices=[]
    for i in range(len(hi)):
        dm=np.abs(lm-hm[i]); da=np.abs(la-ha[i]); dist=np.sqrt(dm*dm+da*da); js=np.where((dm<=MATCH_Z)&(da<=MATCH_Z)&(dist<=MATCH_DIST))[0]
        for j in js: choices.append((float(dist[j]),float(dm[j]),float(da[j]),str(hi.iloc[i].candidate_id),str(lo.iloc[j].candidate_id),i,int(j)))
    choices.sort(key=lambda z:z[:5]); uh=set(); ul=set(); rows=[]
    for dist,dm,da,hid,lid,i,j in choices:
        if i in uh or j in ul: continue
        uh.add(i); ul.add(j); h=hi.iloc[i]; l=lo.iloc[j]; rows.append({"population":pop,"stage_bps":st,"feature":feature,"candidate_date":str(h.candidate_date),"direction":int(h.direction),"direction_label":str(h.direction_label),"high_candidate_id":hid,"low_candidate_id":lid,"high_momentum_z":float(h.__momentum_z),"low_momentum_z":float(l.__momentum_z),"high_aggression_z":float(h.__aggression_z),"low_aggression_z":float(l.__aggression_z),"momentum_abs_diff_z":dm,"aggression_abs_diff_z":da,"control_distance_z":dist,"high_feature_oriented":sf(h[f"__oriented__{feature}"]),"low_feature_oriented":sf(l[f"__oriented__{feature}"])})
    return rows


def balance(p):
    if p.empty: return {"momentum_smd":math.nan,"aggression_smd":math.nan,"max_abs_smd":math.nan,"sum_abs_smd":math.nan}
    m=smd(p.high_momentum_z,p.low_momentum_z); a=smd(p.high_aggression_z,p.low_aggression_z); vals=[abs(v) for v in (m,a) if math.isfinite(v)]; return {"momentum_smd":m,"aggression_smd":a,"max_abs_smd":max(vals) if vals else math.nan,"sum_abs_smd":sum(vals) if vals else math.nan}

def bal_ok(m): return math.isfinite(m["momentum_smd"]) and math.isfinite(m["aggression_smd"]) and abs(m["momentum_smd"])<=BALANCE_SMD and abs(m["aggression_smd"])<=BALANCE_SMD


def repair(p,min_pairs):
    cur=p.copy().reset_index(drop=True)
    while len(cur)>min_pairs and not bal_ok(balance(cur)):
        choices=[]
        for idx,r in cur.iterrows():
            cm=balance(cur.drop(index=idx)); choices.append((sf(cm["max_abs_smd"]) if math.isfinite(sf(cm["max_abs_smd"])) else float("inf"),sf(cm["sum_abs_smd"]) if math.isfinite(sf(cm["sum_abs_smd"])) else float("inf"),-sf(r.control_distance_z),str(r.get("pair_id","")),int(idx)))
        choices.sort(); cur=cur.drop(index=choices[0][-1]).reset_index(drop=True)
    return cur


def build_pairs(inp):
    parts=[]; cov=[]; bals=[]; counter=0; primset={f for f,_,_ in PRIMARY_FEATURES}
    pops={"ALL35":pd.Series(True,index=inp.index),"REACCEL35":bs(inp[REACCEL_FLAG])}
    for pop,pmask in pops.items():
        minimum=MIN_FULL_PAIRS if pop=="ALL35" else MIN_REACCEL_PAIRS
        for st in STAGES:
            allmask=pmask&(inp.stage_bps.astype(int)==st); valid=allmask&bs(inp.proxy_valid)&inp.__momentum_z.notna()&inp.__aggression_z.notna(); base=inp.loc[valid].copy(); cov.append({"population":pop,"stage_bps":st,"population_stage_rows":int(allmask.sum()),"valid_proxy_control_rows":len(base),"valid_share":len(base)/max(1,int(allmask.sum()))})
            for f,_,label in FEATURES:
                tc=f"__tercile__{f}"; x=base[pd.to_numeric(base[tc],errors="coerce").isin([1,3])].copy(); high=x[pd.to_numeric(x[tc],errors="coerce")==3]; low=x[pd.to_numeric(x[tc],errors="coerce")==1]; rows=[]
                for _,b in x.groupby(["candidate_date","direction"],sort=True): rows+=greedy(b[pd.to_numeric(b[tc],errors="coerce")==3],b[pd.to_numeric(b[tc],errors="coerce")==1],f,st,pop)
                raw=pd.DataFrame(rows)
                if len(raw): raw["pair_id"]=[f"7M1-{counter+i+1:08d}" for i in range(len(raw))]; counter+=len(raw)
                pre=balance(raw); rep=repair(raw,minimum); post=balance(rep); feasible=len(rep)>=minimum and bal_ok(post)
                if len(rep): rep["mechanism_label"]=label; rep["pairs_before_repair"]=len(raw); rep["pairs_after_repair"]=len(rep); rep["final_momentum_smd"]=post["momentum_smd"]; rep["final_aggression_smd"]=post["aggression_smd"]; rep["final_balance_pass"]=bal_ok(post); parts.append(rep)
                bals.append({"population":pop,"stage_bps":st,"feature":f,"mechanism_label":label,"is_primary_feature":f in primset,"high_tercile_candidates":len(high),"low_tercile_candidates":len(low),"matched_pairs_before_repair":len(raw),"matched_pairs_after_repair":len(rep),"pairs_trimmed":len(raw)-len(rep),"pre_repair_momentum_smd":pre["momentum_smd"],"pre_repair_aggression_smd":pre["aggression_smd"],"post_repair_momentum_smd":post["momentum_smd"],"post_repair_aggression_smd":post["aggression_smd"],"post_repair_max_abs_smd":post["max_abs_smd"],"min_pairs_gate":minimum,"feasible":feasible})
    return (pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(),pd.DataFrame(cov),pd.DataFrame(bals))


def feasibility(bal):
    rows=[]
    for pop in ("ALL35","REACCEL35"):
        for st in STAGES:
            x=bal[(bal.population==pop)&(bal.stage_bps.astype(int)==st)&bal.is_primary_feature.astype(bool)]; n=int(x.feasible.astype(bool).sum()); rows.append({"population":pop,"stage_bps":st,"primary_features":len(x),"primary_features_feasible":n,"min_primary_features_for_exact_replay":2,"exact_replay_feasible":n>=2,"minimum_balanced_pairs_observed":int(x.matched_pairs_after_repair.min()) if len(x) else 0,"median_balanced_pairs_observed":float(x.matched_pairs_after_repair.median()) if len(x) else math.nan})
    return pd.DataFrame(rows)


def outcomes(root):
    m=meta(root,FROZEN_PHASE7L1_VERSION,"Phase 7L.1"); d=pd.read_parquet(root/"candidate_decision_outcomes.parquet"); d=d[pd.to_numeric(d.observation_horizon_sec,errors="coerce")==OBS_SEC].copy(); d.candidate_id=d.candidate_id.astype(str)
    if d.candidate_id.duplicated().any(): raise RuntimeError("Duplicate 3s outcomes")
    keep=["candidate_id","decision_price_valid"]
    for h in OUTCOME_HORIZONS:
        cs=[f"h{h}_fully_observed",f"h{h}_origin_reach_75",f"h{h}_origin_reach_100"]; req(d,cs,"7L.1 outcomes"); keep+=cs
    x=d[list(dict.fromkeys(keep))].copy()
    for h in OUTCOME_HORIZONS:
        obs=bs(x.decision_price_valid)&bs(x[f"h{h}_fully_observed"]); x[f"__obs_{h}"]=obs
        for t in (75,100): x[f"__y_{t}_{h}"]=x[f"h{h}_origin_reach_{t}"].where(obs,pd.NA).astype("boolean")
    return x,m


def join_out(p,o):
    return p.merge(o.add_prefix("high__"),left_on="high_candidate_id",right_on="high__candidate_id",how="left",validate="many_to_one").merge(o.add_prefix("low__"),left_on="low_candidate_id",right_on="low__candidate_id",how="left",validate="many_to_one")


def effect(p,t,h):
    if p.empty: return {"pairs":0,"high_rate":math.nan,"low_rate":math.nan,"effect_pp":math.nan,"high_wins":0,"low_wins":0}
    x=p[bs(p[f"high____obs_{h}"])&bs(p[f"low____obs_{h}"])].copy(); hy=x[f"high____y_{t}_{h}"].astype("boolean"); ly=x[f"low____y_{t}_{h}"].astype("boolean"); v=hy.notna()&ly.notna(); hy=hy[v].astype(bool); ly=ly[v].astype(bool)
    if not len(hy): return {"pairs":0,"high_rate":math.nan,"low_rate":math.nan,"effect_pp":math.nan,"high_wins":0,"low_wins":0}
    return {"pairs":len(hy),"high_rate":float(hy.mean()),"low_rate":float(ly.mean()),"effect_pp":100*float((hy.astype(int)-ly.astype(int)).mean()),"high_wins":int((hy&~ly).sum()),"low_wins":int((~hy&ly).sum())}


def bootstrap(p,t,h,reps,seed):
    x=p[bs(p[f"high____obs_{h}"])&bs(p[f"low____obs_{h}"])].copy(); hy=x[f"high____y_{t}_{h}"].astype("boolean"); ly=x[f"low____y_{t}_{h}"].astype("boolean"); v=hy.notna()&ly.notna(); x=x.loc[v].copy()
    if not len(x): return math.nan,math.nan,0
    x["__d"]=x[f"high____y_{t}_{h}"].astype(bool).astype(int)-x[f"low____y_{t}_{h}"].astype(bool).astype(int); g=x.groupby(x.candidate_date.astype(str)).__d.agg(["sum","count"])
    if len(g)<2: return math.nan,math.nan,0
    n=len(g); rng=np.random.default_rng(seed); mult=rng.multinomial(n,np.full(n,1/n),size=reps); den=mult@g["count"].to_numpy(float); vals=np.divide(mult@g["sum"].to_numpy(float),den,out=np.full(reps,np.nan),where=den>0)*100; vals=vals[np.isfinite(vals)]
    if len(vals)<250: return math.nan,math.nan,len(vals)
    lo,hi=np.percentile(vals,[2.5,97.5]); return float(lo),float(hi),len(vals)


def analyze(po,bal,reps):
    feasible={(str(r.population),int(r.stage_bps),str(r.feature)):bool(r.feasible) for r in bal.itertuples(index=False)}; rows=[]; splits=[]; reacc=[]
    for pop in ("ALL35","REACCEL35"):
        for st in STAGES:
            for f,_,label in FEATURES:
                p=po[(po.population==pop)&(po.stage_bps.astype(int)==st)&(po.feature==f)].copy(); ok=feasible.get((pop,st,f),False)
                for h in OUTCOME_HORIZONS:
                    for t in (75,100):
                        r=effect(p,t,h); lo=hi=math.nan; nb=0
                        if h==PRIMARY_HORIZON and ok: lo,hi,nb=bootstrap(p,t,h,reps,BOOTSTRAP_SEED+st*1000+t*10+sum(map(ord,f))%101+(50000 if pop=="REACCEL35" else 0))
                        row={"population":pop,"stage_bps":st,"feature":f,"mechanism_label":label,"outcome_horizon_sec":h,"target_bps":t,"feasible_before_outcomes":ok,**r,"ci_low_pp":lo,"ci_high_pp":hi,"bootstrap_valid_reps":nb,"future35_conditioned_interpretation_only":True}; rows.append(row); reacc.append(row) if pop=="REACCEL35" else None
                if pop=="ALL35":
                    for split,mask in (("DISCOVERY",p.candidate_date.astype(str)<=DISCOVERY_END),("VALIDATION",p.candidate_date.astype(str)>=VALIDATION_START)):
                        sp=p[mask].copy(); r=effect(sp,75,PRIMARY_HORIZON); sok=ok and len(sp)>=MIN_SPLIT_PAIRS; lo=hi=math.nan; nb=0
                        if sok: lo,hi,nb=bootstrap(sp,75,PRIMARY_HORIZON,reps,BOOTSTRAP_SEED+st*100+sum(map(ord,f+split))%211)
                        splits.append({"stage_bps":st,"feature":f,"mechanism_label":label,"split":split,"pairs_available":len(sp),"split_pair_gate":MIN_SPLIT_PAIRS,"split_feasible":sok,**r,"ci_low_pp":lo,"ci_high_pp":hi,"bootstrap_valid_reps":nb})
    return pd.DataFrame(rows),pd.DataFrame(splits),pd.DataFrame(reacc)


def timing(inp):
    rows=[]
    for st in STAGES:
        x=inp[inp.stage_bps.astype(int)==st]; lag=pd.to_numeric(x.seconds_stage_to_35,errors="coerce"); disp=pd.to_numeric(x.decision_displacement_from_origin_bps,errors="coerce")
        rows.append({"stage_bps":st,"candidates":len(x),"median_seconds_stage_to_35":float(lag.median()),"p25_seconds_stage_to_35":float(lag.quantile(.25)),"p75_seconds_stage_to_35":float(lag.quantile(.75)),"nominal_remaining_stage_to_75_bps":75-st,"nominal_remaining_stage_to_100_bps":100-st,"median_decision_displacement_after_3s_bps":float(disp.median()),"median_actual_remaining_decision_to_75_bps":float((75-disp).median()),"median_actual_remaining_decision_to_100_bps":float((100-disp).median())})
    return pd.DataFrame(rows)


def checks(cand,inp,pairs,bal,po,source,metas):
    rows=[]
    def add(n,ok,o,e): rows.append({"invariant":n,"status":"PASS" if ok else "FAIL","observed":o,"expected":e})
    add("candidate_count",len(cand)==EXPECTED_CANDIDATES,len(cand),EXPECTED_CANDIDATES); add("stage_rows_exact",len(inp)==EXPECTED_CANDIDATES*len(STAGES),len(inp),EXPECTED_CANDIDATES*len(STAGES)); add("frozen_price_source",source=="LAST_TRADE_FFILL_MAX_5S",source,"LAST_TRADE_FFILL_MAX_5S")
    x10=inp[inp.stage_bps.astype(int)==10]; x35=inp[inp.stage_bps.astype(int)==35]; add("hit10_equals_release",bool((x10.stage_second.astype(int).to_numpy()==x10.release_second.astype(int).to_numpy()).all()),"checked",True); add("hit35_equals_candidate",bool((x35.stage_second.astype(int).to_numpy()==x35.candidate_second.astype(int).to_numpy()).all()),"checked",True)
    leaks=[c for c in inp.columns if any(k in c.lower() for k in ("reach_75","reach_100","outcome","h900_"))]; add("stage_inputs_future_clean",not leaks,leaks,[]); pleaks=[c for c in pairs.columns if any(k in c.lower() for k in ("reach_","outcome","fully_observed"))]; add("pair_manifest_future_clean",not pleaks,pleaks,[]); add("outcome_join_preserves_pairs",len(po)==len(pairs),len(po),len(pairs)); bad=int((bal.feasible.astype(bool)&(pd.to_numeric(bal.post_repair_max_abs_smd,errors="coerce")>BALANCE_SMD+1e-12)).sum()); add("feasible_rows_pass_balance",bad==0,bad,0)
    for label,(m,want) in metas.items(): add(label,str(m.get("builder_version"))==want,m.get("builder_version"),want)
    return pd.DataFrame(rows)


def findings(status,feas,tim,res):
    lines=["# Phase 7M.1 — Early-stage liquidity-mechanism onset screen","",f"- Builder: **{BUILDER_VERSION}**",f"- Status: **{status}**","- Universe: **4,037 frozen candidates that later reached 35bp**","- Prospective early-entry claim: **NO**","- Raw depth/trade replay: **NO**","","## Critical limitation","","All early stages are conditioned on later reaching 35bp. Results say when a mechanism appears inside known 35bp opportunities; they do not yet say an early stage can prospectively distinguish attempts that will fail before 35.","","## Pre-compute exact-replay feasibility"]
    for _,r in feas[feas.population=="ALL35"].iterrows(): lines.append(f"- {int(r.stage_bps)}bp: {int(r.primary_features_feasible)}/{int(r.primary_features)} primary proxies feasible; median balanced pairs={sf(r.median_balanced_pairs_observed):.0f}; exact-replay-feasible={bool(r.exact_replay_feasible)}.")
    lines += ["","## Remaining opportunity"]
    for _,r in tim.iterrows(): lines.append(f"- {int(r.stage_bps)}bp: median {sf(r.median_seconds_stage_to_35):.0f}s before 35; nominal {int(r.nominal_remaining_stage_to_75_bps)}bp left to 75; after the +3s observation median actual remaining={sf(r.median_actual_remaining_decision_to_75_bps):.1f}bp.")
    lines += ["","## 75bp / 15m full-population onset screen"]
    p=res[(res.population=="ALL35")&(res.target_bps==75)&(res.outcome_horizon_sec==PRIMARY_HORIZON)]
    for st in STAGES:
        lines += ["",f"### {st}bp"]
        for _,r in p[p.stage_bps.astype(int)==st].iterrows(): lines.append(f"- {r.mechanism_label}: " + (f"effect={sf(r.effect_pp):+.2f}pp, CI=[{sf(r.ci_low_pp):+.2f},{sf(r.ci_high_pp):+.2f}]pp, pairs={int(r.pairs)}." if bool(r.feasible_before_outcomes) else "NOT INTERPRETABLE — failed the pre-outcome feasibility/balance gate."))
    lines += ["","## Next-step discipline","","No early stage is automatically frozen from significance. If an earlier stage has adequate pre-compute support and the mechanism direction reproduces in validation, only then consider exact queue replay. Before any trading claim, build an early-stage candidate universe containing failures before 35bp.",""]
    return "\n".join(lines)


def self_test():
    sec=np.arange(100,151); price=np.linspace(100,100.4,len(sec)); mv=dbps(price,100,+1); hits={st:int(sec[np.flatnonzero(mv>=st)[0]]) for st in STAGES}; assert all(hits[STAGES[i]]<=hits[STAGES[i+1]] for i in range(len(STAGES)-1))
    f=PRIMARY_FEATURES[0][0]; rows=[]
    for di in range(12):
        day=f"2026-06-{di+1:02d}"
        for i in range(24):
            common={"candidate_date":day,"direction":1 if i%2==0 else -1,"direction_label":"UP" if i%2==0 else "DOWN"}; rows.append({**common,"candidate_id":f"L-{di}-{i}","__momentum_z":-.5+i*.02,"__aggression_z":-.3+i*.015,f"__tercile__{f}":1.,f"__oriented__{f}":.1}); rows.append({**common,"candidate_id":f"H-{di}-{i}","__momentum_z":-.49+i*.02,"__aggression_z":-.29+i*.015,f"__tercile__{f}":3.,f"__oriented__{f}":.8})
    d=pd.DataFrame(rows); pr=[]
    for _,b in d.groupby(["candidate_date","direction"]): pr+=greedy(b[b[f"__tercile__{f}"]==3],b[b[f"__tercile__{f}"]==1],f,20,"ALL35")
    p=pd.DataFrame(pr); p["pair_id"]=[f"P{i}" for i in range(len(p))]; assert len(p)==288 and bal_ok(balance(p)); rep=repair(p,MIN_FULL_PAIRS); assert len(rep)>=MIN_FULL_PAIRS and bal_ok(balance(rep)); assert not any("reach_" in c for c in rep.columns)
    print("SELF-TEST PASS"); print("Validated monotonic stage geometry, outcome-blind exact-date/direction matching, SMD balance gate, minimum-pair feasibility, and future-outcome separation."); return 0


def args(argv=None):
    p=argparse.ArgumentParser(description="Phase 7M.1 early-stage liquidity-mechanism onset screen"); p.add_argument("--symbol",default=SYMBOL); p.add_argument("--phase7a-script",type=Path,default=DEF_7A); p.add_argument("--execution-root",type=Path,default=DEF_EXEC); p.add_argument("--phase7k2-root",type=Path,default=DEF_K2); p.add_argument("--phase6b-root",type=Path,default=DEF_6B); p.add_argument("--phase7l1-root",type=Path,default=DEF_L1); p.add_argument("--phase7l3-root",type=Path,default=DEF_L3); p.add_argument("--output-root",type=Path,default=DEF_OUT); p.add_argument("--report-root",type=Path,default=DEF_REPORT); p.add_argument("--bootstrap-reps",type=int,default=BOOTSTRAP_REPS); p.add_argument("--self-test",action="store_true"); return p.parse_args(argv)


def main(argv=None):
    a=args(argv)
    if a.self_test: return self_test()
    arrow()
    if a.symbol!=SYMBOL: raise SystemExit(f"Frozen to {SYMBOL}")
    print("Building Phase 7M.1 early-stage liquidity-mechanism onset screen..."); print(f"Builder version        : {BUILDER_VERSION}"); print("Stages                 : 10 / 15 / 20 / 25 / 30 / 35 bp"); print("Universe caveat        : future-35-conditioned diagnostic"); print("Raw depth/trade replay : NONE")
    print("[1/9] Loading frozen contracts..."); cand,k2m=load_candidates(a.phase7k2_root); p6m=meta(a.phase6b_root,FROZEN_PHASE6B_VERSION,"Phase 6B"); p7a=module(a.phase7a_script,"phase7a_for_7m1")
    if str(getattr(p7a,"BUILDER_VERSION",""))!=FROZEN_PHASE7A_VERSION: raise RuntimeError("Wrong Phase7A version")
    print("[2/9] Reconstructing frozen-price stage geometry..."); geom,source=stage_geometry(cand,p7a,a.execution_root,a.symbol); print(f"      rows={len(geom):,} source={source}")
    print("[3/9] Extracting cheap Phase6B proxies..."); inp=stage_proxies(geom,a.phase6b_root,a.symbol)
    print("[4/9] Attaching strict causal reacceleration context..."); inp,l3m=attach_reaccel(inp,a.phase7l3_root); print(f"      valid proxy rows={int(bs(inp.proxy_valid).sum()):,}/{len(inp):,}")
    print("[5/9] Freezing outcome-blind matches and PRE-COMPUTE gate..."); pairs,cov,bal=build_pairs(inp); feas=feasibility(bal)
    for _,r in feas[feas.population=="ALL35"].iterrows(): print(f"      {int(r.stage_bps):2d}bp: primary={int(r.primary_features_feasible)}/{int(r.primary_features)} median_pairs={sf(r.median_balanced_pairs_observed):.0f} EXACT_REPLAY_FEASIBLE={bool(r.exact_replay_feasible)}")
    a.output_root.mkdir(parents=True,exist_ok=True); a.report_root.mkdir(parents=True,exist_ok=True); atomic_parquet(inp,a.output_root/"stage_inputs.parquet"); atomic_parquet(pairs,a.output_root/"matched_pairs.parquet"); atomic_csv(cov,a.report_root/"stage_coverage.csv"); atomic_csv(bal,a.report_root/"matching_balance.csv"); atomic_csv(feas,a.report_root/"feasibility_gate.csv")
    print("[6/9] Loading future outcomes AFTER gate freeze..."); out,l1m=outcomes(a.phase7l1_root); po=join_out(pairs,out)
    print("[7/9] Computing 75/100 onset contrasts..."); res,splits,reacc=analyze(po,bal,a.bootstrap_reps); tim=timing(inp)
    print("[8/9] Running invariants..."); inv=checks(cand,inp,pairs,bal,po,source,{"phase7k2_builder":(k2m,FROZEN_PHASE7K2_VERSION),"phase6b_builder":(p6m,FROZEN_PHASE6B_VERSION),"phase7l1_builder":(l1m,FROZEN_PHASE7L1_VERSION),"phase7l3_builder":(l3m,FROZEN_PHASE7L3_VERSION)}); failed=inv[inv.status=="FAIL"]; status="PASS" if failed.empty else "FAIL"
    print("[9/9] Writing reports..."); atomic_csv(tim,a.report_root/"timing_opportunity.csv"); atomic_csv(res,a.report_root/"matched_outcome_contrasts.csv"); atomic_csv(splits,a.report_root/"discovery_validation.csv"); atomic_csv(reacc,a.report_root/"reacceleration_diagnostic.csv"); atomic_csv(inv,a.report_root/"analysis_invariants.csv")
    early=feas[(feas.population=="ALL35")&(feas.stage_bps.astype(int)<35)&feas.exact_replay_feasible.astype(bool)].stage_bps.astype(int).tolist(); summary={"builder_version":BUILDER_VERSION,"schema_version":SCHEMA_VERSION,"schema_name":SCHEMA_NAME,"status":status,"built_at_utc":now(),"symbol":a.symbol,"candidate_policy":POLICY,"candidate_count":len(cand),"stages_bps":list(STAGES),"observation_horizon_sec":OBS_SEC,"price_source":source,"future35_conditioned_universe":True,"prospective_early_entry_claim":False,"balance_gate_abs_smd":BALANCE_SMD,"min_full_balanced_pairs":MIN_FULL_PAIRS,"min_reaccel_balanced_pairs":MIN_REACCEL_PAIRS,"early_stages_feasible_for_possible_exact_replay":early,"raw_depth_replay":False,"raw_trade_replay":False,"failed_invariants":len(failed),"scientific_state":"EARLY_STAGE_MECHANISM_ONSET_SCREEN_NOT_PROSPECTIVE_ENTRY"}; config={**summary,"phase7k2_root":str(a.phase7k2_root),"execution_root":str(a.execution_root),"phase6b_root":str(a.phase6b_root),"phase7l1_root":str(a.phase7l1_root),"phase7l3_root":str(a.phase7l3_root),"matching":{"exact":["candidate_date","direction"],"per_control_z_caliper":MATCH_Z,"euclidean_z_caliper":MATCH_DIST,"balance_abs_smd_gate":BALANCE_SMD},"discovery_end_date":DISCOVERY_END,"validation_start_date":VALIDATION_START,"future_join_order":"stage inputs, exposure bins, matches, balance repair and feasibility frozen before outcomes","proxy_warning":"aggregate 5bp proxies are screening variables, not exact frozen-level queue mechanics"}; atomic_json(a.output_root/"metadata/latest_summary.json",summary); atomic_json(a.output_root/"metadata/run_config.json",config); atomic_text(a.report_root/"findings.md",findings(status,feas,tim,res))
    print(); print(f"Phase 7M.1 status: {status}"); print(f"Early stages passing pre-compute exact-replay feasibility: {early if early else 'NONE'}"); print("Interpretation: mechanism-onset diagnostic ONLY; not prospective early entry."); print(f"Reports: {a.report_root}")
    if len(failed): print(failed.to_string(index=False)); return 2
    return 0

if __name__=="__main__": raise SystemExit(main())
