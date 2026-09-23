#!/usr/bin/env python3
"""
Phase 7L.4 — Re-acceleration liquidity-mechanism matched analysis
================================================================

Plain-English question
----------------------
Within the strict causal re-acceleration context frozen after Phase 7L.3,
does opposing-liquidity failure still distinguish 75/100-bp continuation when
we compare candidates from the same day and direction with similar continuous
3-second momentum and directional aggressive notional?

Frozen re-acceleration context
------------------------------
REARM_ANY_PRIOR50_30M_NO_OPPOSITE50:
* current 35-bp candidate came from a causal REARM_RESET;
* a same-direction 50-bp expansion had already been confirmed within 30m;
* no opposite-direction 50-bp expansion occurred between that prior expansion
  and the current reset/origin.

The current candidate's own future is not used to define the context.

Primary design
--------------
* observation point: candidate + 3 seconds;
* book exposure: top vs bottom tercile of ONE book mechanism at a time;
* exact matching block: candidate_date x direction;
* continuous matching controls:
    directional_mid_move_bps
    log1p(directional_aggressive_quote)
* deterministic 1:1 greedy minimum-distance matching without replacement;
* primary caliper: <=0.75 SD on each control and <=1.00 SD Euclidean distance;
* primary outcome: origin reaches 75bp within 15m of decision timestamp;
* secondary: origin reaches 100bp within 15m;
* robustness: 5/10/15/30m horizons and tight/primary/relaxed calipers;
* uncertainty: candidate-date cluster bootstrap (1,000 reps by default).

No raw depth/trade replay, no ML, no composite book-failure score, no profit
claim, and no earlier-than-35 entry optimization.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

BUILDER_VERSION = "1.0.0"
SCHEMA_VERSION = 1
SCHEMA_NAME = "reacceleration_liquidity_matched_analysis_v1"

FROZEN_PHASE7L3_BUILDER = "1.0.0"
FROZEN_PHASE7L1_BUILDER = "1.0.1"
REACCEL_DEFINITION = "REARM_ANY_PRIOR50_30M_NO_OPPOSITE50"
REACCEL_FLAG = f"is_reaccel__{REACCEL_DEFINITION}"

EXPECTED_REACCEL_CANDIDATES = 1336
PRIMARY_OBSERVATION_HORIZON_SEC = 3
OUTCOME_HORIZONS_SEC = (300, 600, 900, 1800)
PRIMARY_OUTCOME_HORIZON_SEC = 900
PRIMARY_TARGET_BPS = 75
SECONDARY_TARGET_BPS = 100

MOMENTUM = "directional_mid_move_bps"
AGGRESSION = "directional_aggressive_quote"
BOOK_BINS = 3

CORE_BOOK_FEATURES = (
    ("opposing__durable_unrecovered_loss_share_initial", +1, "DURABLE_UNRECOVERED_LOSS"),
    ("opposing__peak_initial_queue_loss_share", +1, "PEAK_INITIAL_QUEUE_LOSS"),
    ("opposing__retained_initial_quote_ratio", -1, "LOW_RETAINED_OPPOSING_LIQUIDITY"),
    ("support_minus_opposing_retained_ratio", +1, "SUPPORT_VS_OPPOSING_ASYMMETRY"),
)

CALIPERS = {
    "TIGHT":   {"per_control_z": 0.50, "euclidean_z": 0.75},
    "PRIMARY": {"per_control_z": 0.75, "euclidean_z": 1.00},
    "RELAXED": {"per_control_z": 1.00, "euclidean_z": 1.50},
}
PRIMARY_CALIPER = "PRIMARY"
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 20260830
MIN_BOOTSTRAP_VALID_REPS = 250

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_L3_ROOT = REPO_ROOT / "data/derived/perps/research/causal_reacceleration_context_audit_v1"
DEFAULT_L1_ROOT = REPO_ROOT / "data/derived/perps/research/repeatable_candidate_participant_interaction_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/derived/perps/research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports/perps_reacceleration_liquidity_matched_analysis"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return math.nan
    return v if math.isfinite(v) else math.nan


def bools(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return s.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False}).fillna(False).astype(bool)


def require(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7l4-", suffix=path.suffix, dir=str(path.parent))
    os.close(fd)
    try:
        Path(tmp).write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_json(path: Path, obj: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7l4-", suffix=".csv", dir=str(path.parent))
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7l4-", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def load_meta(root: Path, builder: str, label: str) -> dict[str, Any]:
    p = root / "metadata/latest_summary.json"
    if not p.exists():
        raise RuntimeError(f"Missing {label} metadata: {p}")
    m = json.loads(p.read_text(encoding="utf-8"))
    if str(m.get("builder_version")) != builder:
        raise RuntimeError(f"{label} builder mismatch: expected {builder}, found {m.get('builder_version')!r}")
    if str(m.get("status")) != "PASS":
        raise RuntimeError(f"{label} must be PASS; found {m.get('status')!r}")
    return m


def percentile_bin(s: pd.Series, n: int) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    mask = x.notna()
    if mask.any():
        r = x.loc[mask].rank(method="first", pct=True)
        out.loc[mask] = np.ceil(r * n).clip(1, n).astype(int).astype("Int64")
    return out


def zscore(s: pd.Series) -> tuple[pd.Series, float, float]:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    mu = float(x.mean()); sd = float(x.std(ddof=0))
    if not math.isfinite(sd) or sd <= 0:
        raise RuntimeError("Invalid constant control for z-scoring")
    return (x - mu) / sd, mu, sd


def build_causal_frame(labels: pd.DataFrame, mechanics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    require(labels, ["candidate_id", "candidate_date", "direction", "direction_label", REACCEL_FLAG], "7L.3 labels")
    require(mechanics, ["candidate_id", "candidate_date", "direction", "direction_label", "observation_horizon_sec", "window_valid", "decision_price_valid", MOMENTUM, AGGRESSION] + [f for f,_,_ in CORE_BOOK_FEATURES], "7L.1 mechanics")

    lab = labels[bools(labels[REACCEL_FLAG])].copy()
    if len(lab) != EXPECTED_REACCEL_CANDIDATES:
        raise RuntimeError(f"Expected {EXPECTED_REACCEL_CANDIDATES} strict re-acceleration candidates; found {len(lab)}")
    lab["candidate_id"] = lab["candidate_id"].astype(str)

    m = mechanics[pd.to_numeric(mechanics["observation_horizon_sec"], errors="coerce") == PRIMARY_OBSERVATION_HORIZON_SEC].copy()
    m["candidate_id"] = m["candidate_id"].astype(str)
    if m["candidate_id"].duplicated().any():
        raise RuntimeError("Duplicate primary-horizon mechanics")

    keep = [c for c in ["candidate_id", "candidate_date", "direction", "direction_label", "origin_kind", "same_direction_candidate_ordinal", "seconds_since_latest_same_direction_50_hit", REACCEL_FLAG] if c in lab.columns]
    d = lab[keep].merge(m, on=["candidate_id", "candidate_date", "direction", "direction_label"], how="left", validate="one_to_one")
    d = d[bools(d["window_valid"]) & bools(d["decision_price_valid"])].copy()

    d["__mom_raw"] = pd.to_numeric(d[MOMENTUM], errors="coerce")
    d["__agg_raw"] = np.log1p(pd.to_numeric(d[AGGRESSION], errors="coerce").clip(lower=0.0))
    d = d[d["__mom_raw"].notna() & d["__agg_raw"].notna()].copy()
    d["__mom_z"], mm, ms = zscore(d["__mom_raw"])
    d["__agg_z"], am, astd = zscore(d["__agg_raw"])

    for f, sign, _ in CORE_BOOK_FEATURES:
        d[f"__oriented__{f}"] = pd.to_numeric(d[f], errors="coerce") * sign
        d[f"__tercile__{f}"] = percentile_bin(d[f"__oriented__{f}"], BOOK_BINS).astype(float)

    dates = sorted(d["candidate_date"].astype(str).unique())
    rank = {x:i for i,x in enumerate(dates)}
    denom = max(1, len(dates))
    d["era4"] = d["candidate_date"].astype(str).map(lambda x: min(4, 1 + int(4 * rank[x] / denom)))
    return d.reset_index(drop=True), {"momentum_mean": mm, "momentum_std": ms, "log_aggression_mean": am, "log_aggression_std": astd}


def greedy_block(high: pd.DataFrame, low: pd.DataFrame, feature: str, caliper_name: str) -> list[dict[str, Any]]:
    if high.empty or low.empty: return []
    cal = CALIPERS[caliper_name]
    hi = high.reset_index(drop=True); lo = low.reset_index(drop=True)
    hm = hi["__mom_z"].to_numpy(float); ha = hi["__agg_z"].to_numpy(float)
    lm = lo["__mom_z"].to_numpy(float); la = lo["__agg_z"].to_numpy(float)
    choices = []
    for i in range(len(hi)):
        dm = np.abs(lm - hm[i]); da = np.abs(la - ha[i]); dist = np.sqrt(dm*dm + da*da)
        js = np.where((dm <= cal["per_control_z"]) & (da <= cal["per_control_z"]) & (dist <= cal["euclidean_z"]))[0]
        for j in js:
            choices.append((float(dist[j]), float(dm[j]), float(da[j]), str(hi.iloc[i]["candidate_id"]), str(lo.iloc[j]["candidate_id"]), i, int(j)))
    choices.sort(key=lambda x: x[:5])
    used_h, used_l, rows = set(), set(), []
    for dist, dm, da, hid, lid, i, j in choices:
        if i in used_h or j in used_l: continue
        used_h.add(i); used_l.add(j)
        h, l = hi.iloc[i], lo.iloc[j]
        rows.append({
            "feature": feature, "caliper": caliper_name,
            "candidate_date": str(h["candidate_date"]), "direction": int(h["direction"]), "direction_label": str(h["direction_label"]),
            "high_candidate_id": hid, "low_candidate_id": lid,
            "control_distance_z": dist, "momentum_abs_diff_z": dm, "aggression_abs_diff_z": da,
            "high_mom_z": float(h["__mom_z"]), "low_mom_z": float(l["__mom_z"]),
            "high_agg_z": float(h["__agg_z"]), "low_agg_z": float(l["__agg_z"]),
            "high_book_oriented": safe_float(h[f"__oriented__{feature}"]), "low_book_oriented": safe_float(l[f"__oriented__{feature}"]),
            "high_era4": int(h["era4"]), "low_era4": int(l["era4"]),
        })
    return rows


def build_pairs(causal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f, _, label in CORE_BOOK_FEATURES:
        t = f"__tercile__{f}"
        e = causal[pd.to_numeric(causal[t], errors="coerce").isin([1, BOOK_BINS])].copy()
        for cal in CALIPERS:
            for (_, _), block in e.groupby(["candidate_date", "direction"], sort=True):
                high = block[pd.to_numeric(block[t], errors="coerce") == BOOK_BINS]
                low = block[pd.to_numeric(block[t], errors="coerce") == 1]
                for r in greedy_block(high, low, f, cal):
                    r["mechanism_label"] = label; rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty: return out
    out = out.sort_values(["feature","caliper","candidate_date","direction","control_distance_z","high_candidate_id","low_candidate_id"], kind="mergesort").reset_index(drop=True)
    out["pair_id"] = [f"PAIR-{i:07d}" for i in range(1, len(out)+1)]
    return out


def smd(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a,float); b = np.asarray(b,float); a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
    if not len(a) or not len(b): return math.nan
    va = np.var(a, ddof=1) if len(a)>1 else 0.0; vb = np.var(b, ddof=1) if len(b)>1 else 0.0
    den = math.sqrt(max(0.0, 0.5*(va+vb)))
    return 0.0 if den == 0 and math.isclose(float(a.mean()), float(b.mean())) else (math.nan if den == 0 else float((a.mean()-b.mean())/den))


def audit_matching(causal: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    by_id = causal.set_index("candidate_id")
    sr, br = [], []
    for f, _, label in CORE_BOOK_FEATURES:
        t = f"__tercile__{f}"; raw = causal[pd.to_numeric(causal[t], errors="coerce").isin([1,BOOK_BINS])]
        rhi = raw[pd.to_numeric(raw[t], errors="coerce") == BOOK_BINS]; rlo = raw[pd.to_numeric(raw[t], errors="coerce") == 1]
        for cal in CALIPERS:
            p = pairs[(pairs.feature == f) & (pairs.caliper == cal)]
            hids = p.high_candidate_id.astype(str).tolist(); lids = p.low_candidate_id.astype(str).tolist()
            mhi = by_id.loc[hids] if len(p) else causal.iloc[0:0]; mlo = by_id.loc[lids] if len(p) else causal.iloc[0:0]
            sr.append({"feature":f,"mechanism_label":label,"caliper":cal,"high_candidates":len(rhi),"low_candidates":len(rlo),"matched_pairs":len(p),"median_distance_z":safe_float(p.control_distance_z.median()) if len(p) else math.nan,"p90_distance_z":safe_float(p.control_distance_z.quantile(.9)) if len(p) else math.nan})
            for name,col in (("MOMENTUM","__mom_z"),("AGGRESSION","__agg_z")):
                br.append({"feature":f,"caliper":cal,"control":name,"pre_match_smd":smd(rhi[col].to_numpy(),rlo[col].to_numpy()),"post_match_smd":smd(mhi[col].to_numpy(),mlo[col].to_numpy()) if len(p) else math.nan,"matched_pairs":len(p)})
    return pd.DataFrame(sr), pd.DataFrame(br)


def prepare_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    d = outcomes[pd.to_numeric(outcomes["observation_horizon_sec"], errors="coerce") == PRIMARY_OBSERVATION_HORIZON_SEC].copy()
    d["candidate_id"] = d["candidate_id"].astype(str)
    if d.candidate_id.duplicated().any(): raise RuntimeError("Duplicate 3s outcome rows")
    keep = ["candidate_id","decision_price_valid"]
    for h in OUTCOME_HORIZONS_SEC:
        cols = [f"h{h}_fully_observed",f"h{h}_origin_reach_75",f"h{h}_origin_reach_100"]
        require(d, cols, "7L.1 outcomes"); keep += cols
    x = d[list(dict.fromkeys(keep))].copy()
    for h in OUTCOME_HORIZONS_SEC:
        obs = bools(x.decision_price_valid) & bools(x[f"h{h}_fully_observed"])
        x[f"__obs_{h}"] = obs
        for t in (75,100): x[f"__y_{t}_{h}"] = x[f"h{h}_origin_reach_{t}"].where(obs,pd.NA).astype("boolean")
    return x


def join_outcomes(pairs: pd.DataFrame, o: pd.DataFrame) -> pd.DataFrame:
    hi=o.add_prefix("high__"); lo=o.add_prefix("low__")
    return pairs.merge(hi,left_on="high_candidate_id",right_on="high__candidate_id",how="left",validate="many_to_one").merge(lo,left_on="low_candidate_id",right_on="low__candidate_id",how="left",validate="many_to_one")


def effect(p: pd.DataFrame, target: int, horizon: int) -> dict[str,Any]:
    x = p[bools(p[f"high____obs_{horizon}"]) & bools(p[f"low____obs_{horizon}"])].copy()
    hy=x[f"high____y_{target}_{horizon}"].astype("boolean"); ly=x[f"low____y_{target}_{horizon}"].astype("boolean"); v=hy.notna()&ly.notna(); x=x.loc[v]; hy=hy.loc[v].astype(bool); ly=ly.loc[v].astype(bool)
    if not len(x): return {"pairs":0,"high_rate":math.nan,"low_rate":math.nan,"effect_pp":math.nan,"high_wins":0,"low_wins":0}
    return {"pairs":len(x),"high_rate":float(hy.mean()),"low_rate":float(ly.mean()),"effect_pp":100*float((hy.astype(int)-ly.astype(int)).mean()),"high_wins":int((hy&~ly).sum()),"low_wins":int((~hy&ly).sum())}


def bootstrap_day(p: pd.DataFrame, target:int, horizon:int, reps:int, seed:int) -> tuple[float,float,int]:
    x = p[bools(p[f"high____obs_{horizon}"]) & bools(p[f"low____obs_{horizon}"])].copy()
    hy=x[f"high____y_{target}_{horizon}"].astype("boolean"); ly=x[f"low____y_{target}_{horizon}"].astype("boolean"); v=hy.notna()&ly.notna(); x=x.loc[v]
    if not len(x): return math.nan,math.nan,0
    x["__d"] = x[f"high____y_{target}_{horizon}"].astype(bool).astype(int)-x[f"low____y_{target}_{horizon}"].astype(bool).astype(int)
    g=x.groupby(x.candidate_date.astype(str)).__d.agg(["sum","count"])
    if len(g)<2: return math.nan,math.nan,0
    sums=g["sum"].to_numpy(float); counts=g["count"].to_numpy(float); n=len(g); rng=np.random.default_rng(seed)
    mult=rng.multinomial(n,np.full(n,1/n),size=reps); den=mult@counts; vals=np.divide(mult@sums,den,out=np.full(reps,np.nan),where=den>0)*100; vals=vals[np.isfinite(vals)]
    if len(vals)<MIN_BOOTSTRAP_VALID_REPS: return math.nan,math.nan,len(vals)
    lo,hi=np.percentile(vals,[2.5,97.5]); return float(lo),float(hi),len(vals)


def label_effect(e:float,lo:float)->str:
    if not math.isfinite(e): return "NOT_TESTED"
    if e<=0: return "NOT_SUPPORTED"
    return "SUPPORTED" if math.isfinite(lo) and lo>0 else "TENTATIVE"


def analyze(po: pd.DataFrame, reps:int) -> pd.DataFrame:
    rows=[]
    for f,_,label in CORE_BOOK_FEATURES:
        for cal in CALIPERS:
            p=po[(po.feature==f)&(po.caliper==cal)]
            for h in OUTCOME_HORIZONS_SEC:
                for t in (75,100):
                    r=effect(p,t,h)
                    if cal==PRIMARY_CALIPER and h==PRIMARY_OUTCOME_HORIZON_SEC:
                        lo,hi,n=bootstrap_day(p,t,h,reps,BOOTSTRAP_SEED+t*100+sum(map(ord,f))%97)
                    else: lo=hi=math.nan; n=0
                    rows.append({"feature":f,"mechanism_label":label,"caliper":cal,"outcome_horizon_sec":h,"target_bps":t,**r,"ci_low_pp":lo,"ci_high_pp":hi,"bootstrap_valid_reps":n,"interpretation":label_effect(safe_float(r["effect_pp"]),lo) if cal==PRIMARY_CALIPER and h==PRIMARY_OUTCOME_HORIZON_SEC else "ROBUSTNESS_ONLY"})
    return pd.DataFrame(rows)


def subgroup(po:pd.DataFrame, kind:str, reps:int)->pd.DataFrame:
    p0=po[po.caliper==PRIMARY_CALIPER]; rows=[]
    specs = [("UP",p0.direction==1),("DOWN",p0.direction==-1)] if kind=="DIRECTION" else [(f"ERA_{e}",p0.high_era4==e) for e in (1,2,3,4)]
    for seg,mask in specs:
        d=p0.loc[mask]
        for f,_,label in CORE_BOOK_FEATURES:
            p=d[d.feature==f]
            for t in (75,100):
                r=effect(p,t,PRIMARY_OUTCOME_HORIZON_SEC)
                if kind=="DIRECTION": lo,hi,n=bootstrap_day(p,t,PRIMARY_OUTCOME_HORIZON_SEC,reps,BOOTSTRAP_SEED+t*1000+sum(map(ord,f+seg))%193)
                else: lo=hi=math.nan; n=0
                rows.append({"subgroup_type":kind,"segment":seg,"feature":f,"mechanism_label":label,"target_bps":t,**r,"ci_low_pp":lo,"ci_high_pp":hi,"bootstrap_valid_reps":n})
    return pd.DataFrame(rows)


def invariants(labels:pd.DataFrame, causal:pd.DataFrame, pairs:pd.DataFrame, po:pd.DataFrame, l3:dict[str,Any])->pd.DataFrame:
    rows=[]
    def add(name,ok,obs,exp): rows.append({"invariant":name,"status":"PASS" if ok else "FAIL","observed":obs,"expected":exp})
    add("strict_context_flag_present",REACCEL_FLAG in labels.columns,REACCEL_FLAG in labels.columns,True)
    n=int(bools(labels[REACCEL_FLAG]).sum()); add("strict_context_count",n==EXPECTED_REACCEL_CANDIDATES,n,EXPECTED_REACCEL_CANDIDATES)
    add("causal_ids_unique",not causal.candidate_id.duplicated().any(),int(causal.candidate_id.duplicated().sum()),0)
    duph=int(pairs.duplicated(["feature","caliper","high_candidate_id"]).sum()); dupl=int(pairs.duplicated(["feature","caliper","low_candidate_id"]).sum()); add("without_replacement_high",duph==0,duph,0); add("without_replacement_low",dupl==0,dupl,0)
    leaks=[c for c in pairs.columns if "reach_" in c.lower() or "outcome" in c.lower() or "fully_observed" in c.lower()]; add("pair_manifest_future_clean",not leaks,leaks,[])
    add("outcome_join_preserves_pairs",len(po)==len(pairs),len(po),len(pairs))
    bad=0
    for name,cal in CALIPERS.items():
        p=pairs[pairs.caliper==name]; bad += int((p.momentum_abs_diff_z>cal["per_control_z"]+1e-12).sum()+(p.aggression_abs_diff_z>cal["per_control_z"]+1e-12).sum()+(p.control_distance_z>cal["euclidean_z"]+1e-12).sum())
    add("calipers_respected",bad==0,bad,0)
    add("7l3_context_selection_was_outcome_blind",not bool(l3.get("definition_selection_used_current_75_100_outcomes",True)),l3.get("definition_selection_used_current_75_100_outcomes"),False)
    return pd.DataFrame(rows)


def findings(status:str, ms:pd.DataFrame, bal:pd.DataFrame, res:pd.DataFrame)->str:
    lines=["# Phase 7L.4 — Re-acceleration liquidity matched analysis","",f"- Builder: **{BUILDER_VERSION}**",f"- Status: **{status}**",f"- Strict causal context: **{REACCEL_DEFINITION}**",f"- Raw context N: **{EXPECTED_REACCEL_CANDIDATES:,}**","- Raw depth/trade replay: **NO**","- Pairs built before outcomes: **YES**","","## Primary matching support"]
    for _,r in ms[ms.caliper==PRIMARY_CALIPER].iterrows():
        b=bal[(bal.feature==r.feature)&(bal.caliper==PRIMARY_CALIPER)]; bm=b[b.control=="MOMENTUM"]; ba=b[b.control=="AGGRESSION"]
        lines.append(f"- {r.mechanism_label}: pairs={int(r.matched_pairs)}, median distance={safe_float(r.median_distance_z):.3f}, post |SMD| momentum={abs(safe_float(bm.post_match_smd.iloc[0])) if len(bm) else math.nan:.3f}, aggression={abs(safe_float(ba.post_match_smd.iloc[0])) if len(ba) else math.nan:.3f}.")
    for target in (75,100):
        lines += ["",f"## Primary {target}bp results"]
        x=res[(res.caliper==PRIMARY_CALIPER)&(res.outcome_horizon_sec==PRIMARY_OUTCOME_HORIZON_SEC)&(res.target_bps==target)].sort_values("effect_pp",ascending=False)
        for _,r in x.iterrows():
            lines.append(f"- {r.mechanism_label}: high={100*safe_float(r.high_rate):.1f}%, low={100*safe_float(r.low_rate):.1f}%, effect={safe_float(r.effect_pp):+.2f}pp, CI=[{safe_float(r.ci_low_pp):+.2f},{safe_float(r.ci_high_pp):+.2f}]pp, pairs={int(r.pairs)}, **{r.interpretation}**.")
    lines += ["","## Discipline","- Same day and same direction are exact matching blocks.","- Momentum/aggression are continuous controls, not 3x3 bins.","- Each book mechanism is tested separately.","- This tests the re-acceleration mechanism at the 35bp examination point; it does not search for an earlier entry.",""]
    return "\n".join(lines)


def self_test()->int:
    rows=[]; f="opposing__peak_initial_queue_loss_share"
    for day in ("2026-06-01","2026-06-02"):
        for direction in (1,-1):
            for i in range(12):
                common={"candidate_date":day,"direction":direction,"direction_label":"UP" if direction>0 else "DOWN","era4":1}
                rows.append({**common,"candidate_id":f"L-{day}-{direction}-{i}","__mom_z":-.2+.02*i,"__agg_z":-.1+.02*i,f"__oriented__{f}":.1+i*.001,f"__tercile__{f}":1.0})
                rows.append({**common,"candidate_id":f"H-{day}-{direction}-{i}","__mom_z":-.19+.02*i,"__agg_z":-.09+.02*i,f"__oriented__{f}":.8+i*.001,f"__tercile__{f}":3.0})
    df=pd.DataFrame(rows); pairs=[]
    for _,b in df.groupby(["candidate_date","direction"]): pairs += greedy_block(b[b[f"__tercile__{f}"]==3],b[b[f"__tercile__{f}"]==1],f,"PRIMARY")
    p=pd.DataFrame(pairs); assert len(p)==48 and not p.high_candidate_id.duplicated().any() and not p.low_candidate_id.duplicated().any()
    p["high____obs_900"]=True; p["low____obs_900"]=True; p["high____y_75_900"]=[i%4!=0 for i in range(len(p))]; p["low____y_75_900"]=[i%4==0 for i in range(len(p))]
    r=effect(p,75,900); assert r["effect_pp"]>0
    lo,hi,n=bootstrap_day(p,75,900,1000,123); assert n==1000 and lo>0 and hi>0
    print("SELF-TEST PASS")
    print("Validated same-date/same-direction continuous matching, calipers, without-replacement pairs, paired effects, and date-cluster bootstrap.")
    return 0


def parse_args(argv:Sequence[str]|None=None):
    p=argparse.ArgumentParser(description="Phase 7L.4 re-acceleration liquidity matched analysis")
    p.add_argument("--phase7l3-root",type=Path,default=DEFAULT_L3_ROOT); p.add_argument("--phase7l1-root",type=Path,default=DEFAULT_L1_ROOT); p.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT_ROOT); p.add_argument("--report-root",type=Path,default=DEFAULT_REPORT_ROOT); p.add_argument("--bootstrap-reps",type=int,default=BOOTSTRAP_REPS); p.add_argument("--self-test",action="store_true"); return p.parse_args(argv)


def main(argv:Sequence[str]|None=None)->int:
    args=parse_args(argv)
    if args.self_test: return self_test()
    print("Building Phase 7L.4 re-acceleration liquidity matched analysis...")
    print(f"Builder version        : {BUILDER_VERSION}")
    print(f"Strict reaccel context : {REACCEL_DEFINITION}")
    print(f"Expected context N     : {EXPECTED_REACCEL_CANDIDATES:,}")
    print("Matching               : same date + same direction + continuous momentum/aggression")
    print("Raw depth/trade replay : NONE")

    print("[1/8] Loading frozen inputs...")
    l3=load_meta(args.phase7l3_root,FROZEN_PHASE7L3_BUILDER,"Phase 7L.3"); l1=load_meta(args.phase7l1_root,FROZEN_PHASE7L1_BUILDER,"Phase 7L.1")
    labels=pd.read_parquet(args.phase7l3_root/"candidate_context_labels.parquet"); mechanics=pd.read_parquet(args.phase7l1_root/"candidate_mechanics.parquet"); outcomes=pd.read_parquet(args.phase7l1_root/"candidate_decision_outcomes.parquet")
    print(f"      labels={len(labels):,} mechanics={len(mechanics):,} outcomes={len(outcomes):,}")

    print("[2/8] Building outcome-blind strict re-acceleration frame...")
    causal,scaling=build_causal_frame(labels,mechanics); print(f"      valid strict-context 3s rows={len(causal):,}")

    print("[3/8] Building outcome-blind matched pairs...")
    pairs=build_pairs(causal)
    if pairs.empty: raise RuntimeError("No pairs produced")
    print(f"      pair rows across features/calipers={len(pairs):,}")

    print("[4/8] Auditing matching support/balance before outcomes...")
    ms,bal=audit_matching(causal,pairs)
    for _,r in ms[ms.caliper==PRIMARY_CALIPER].iterrows(): print(f"      {r.mechanism_label:34s} pairs={int(r.matched_pairs):4d} medianD={safe_float(r.median_distance_z):.3f}")

    print("[5/8] Joining future outcomes after pair manifest is frozen...")
    ot=prepare_outcomes(outcomes); po=join_outcomes(pairs,ot)

    print("[6/8] Computing matched 75/100 contrasts...")
    res=analyze(po,args.bootstrap_reps); direc=subgroup(po,"DIRECTION",args.bootstrap_reps); era=subgroup(po,"ERA",args.bootstrap_reps)

    print("[7/8] Running invariants...")
    inv=invariants(labels,causal,pairs,po,l3); failed=inv[inv.status=="FAIL"]; status="PASS" if failed.empty else "FAIL"

    print("[8/8] Writing outputs...")
    args.output_root.mkdir(parents=True,exist_ok=True); args.report_root.mkdir(parents=True,exist_ok=True)
    atomic_parquet(pairs,args.output_root/"matched_pairs.parquet"); atomic_csv(ms,args.report_root/"matching_summary.csv"); atomic_csv(bal,args.report_root/"matching_balance.csv"); atomic_csv(res,args.report_root/"matched_outcome_contrasts.csv"); atomic_csv(res[res.caliper==PRIMARY_CALIPER],args.report_root/"horizon_robustness.csv"); atomic_csv(res[res.outcome_horizon_sec==PRIMARY_OUTCOME_HORIZON_SEC],args.report_root/"caliper_robustness.csv"); atomic_csv(direc,args.report_root/"direction_robustness.csv"); atomic_csv(era,args.report_root/"era_robustness.csv"); atomic_csv(inv,args.report_root/"analysis_invariants.csv")
    primary=res[(res.caliper==PRIMARY_CALIPER)&(res.outcome_horizon_sec==PRIMARY_OUTCOME_HORIZON_SEC)]
    s75=int(((primary.target_bps==75)&(primary.interpretation=="SUPPORTED")).sum()); s100=int(((primary.target_bps==100)&(primary.interpretation=="SUPPORTED")).sum())
    summary={"builder_version":BUILDER_VERSION,"schema_version":SCHEMA_VERSION,"schema_name":SCHEMA_NAME,"status":status,"built_at_utc":now_iso(),"reaccel_definition":REACCEL_DEFINITION,"raw_reaccel_candidates":EXPECTED_REACCEL_CANDIDATES,"valid_causal_reaccel_candidates":len(causal),"primary_caliper":PRIMARY_CALIPER,"supported_primary_75_features":s75,"supported_primary_100_features":s100,"pair_manifest_built_before_outcomes":True,"raw_depth_replay":False,"raw_trade_replay":False,"failed_invariants":len(failed),"scientific_state":"REACCELERATION_MECHANISM_MATCHED_ANALYSIS_COMPLETE_NOT_ALPHA"}
    atomic_json(args.output_root/"metadata/latest_summary.json",summary); atomic_json(args.output_root/"metadata/run_config.json",{**summary,"phase7l3_root":str(args.phase7l3_root),"phase7l1_root":str(args.phase7l1_root),"matching_exact_blocks":["candidate_date","direction"],"matching_controls":[MOMENTUM,f"log1p({AGGRESSION})"],"control_scaling":scaling,"calipers":CALIPERS,"bootstrap_reps":args.bootstrap_reps,"outcome_join_order":"pairs frozen before outcomes"}); atomic_text(args.report_root/"findings.md",findings(status,ms,bal,res))
    print(); print(f"Phase 7L.4 status: {status}"); print(f"Primary 75bp supported mechanisms: {s75}/{len(CORE_BOOK_FEATURES)}"); print(f"Primary 100bp supported mechanisms: {s100}/{len(CORE_BOOK_FEATURES)}"); print(f"Reports: {args.report_root}")
    if len(failed): print(failed.to_string(index=False)); return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
