"""Authoritative receive-time order-book replay for R0.6.

Semantics are intentionally conservative and mirror the source evidence:
complete collector-local snapshots may bootstrap/recover only when the first
following diff proves exact ``pu == snapshot_last_update_id`` continuation.
Once a diff chain is valid, later snapshots are audit-only.  A real ``pu`` gap
or crossed book invalidates execution state until a later valid handshake.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

try:  # optional acceleration for the full production replay
    import numba as _numba  # type: ignore
    from numba import njit as _njit, types as _types  # type: ignore
    from numba.typed import Dict as _NumbaDict  # type: ignore
except Exception:  # pragma: no cover
    _numba = None
    _njit = None
    _types = None
    _NumbaDict = None


if _njit is not None:
    @_njit(cache=True)
    def _apply_kernel(levels, ticks, quantities, min_tick, max_tick, active_count):
        for i in range(ticks.size):
            tick = int(ticks[i]); q = float(quantities[i])
            if tick <= 0 or not math.isfinite(q) or q < 0.0:
                continue
            exists = tick in levels
            if q == 0.0:
                if exists:
                    del levels[tick]; active_count -= 1
                    if active_count == 0:
                        min_tick = 0; max_tick = 0
                    else:
                        if tick == min_tick:
                            t = min_tick + 1
                            while t <= max_tick and t not in levels: t += 1
                            min_tick = t
                        if tick == max_tick:
                            t = max_tick - 1
                            while t >= min_tick and t not in levels: t -= 1
                            max_tick = t
            elif exists:
                levels[tick] = q
            else:
                levels[tick] = q; active_count += 1
                if active_count == 1:
                    min_tick = tick; max_tick = tick
                else:
                    if tick < min_tick: min_tick = tick
                    if tick > max_tick: max_tick = tick
        return min_tick, max_tick, active_count


class _PythonSide:
    def __init__(self, is_bid: bool, price_scale: float):
        self.is_bid = bool(is_bid); self.price_scale = float(price_scale); self.qty: dict[int, float] = {}
    def clear(self): self.qty.clear()
    def _tick(self, price: float) -> int:
        s = float(price) * self.price_scale; t = int(round(s))
        if abs(s - t) > 1e-6: raise RuntimeError(f"price {price} not aligned to price_scale={self.price_scale:g}")
        return t
    def reset(self, levels: Sequence[Any]):
        self.clear()
        for level in levels or []:
            try:
                p = float(level.get("price")); q = float(level.get("quantity")) if isinstance(level, Mapping) else None
            except Exception:
                try: p = float(level[0]); q = float(level[1])
                except Exception: continue
            if p > 0 and q is not None and q > 0: self.qty[self._tick(p)] = q
    def update_arrays(self, ticks, quantities):
        for t, q in zip(ticks, quantities):
            ti = int(t); qf = float(q)
            if ti <= 0 or not math.isfinite(qf) or qf < 0: continue
            if qf == 0: self.qty.pop(ti, None)
            else: self.qty[ti] = qf
    def best(self):
        if not self.qty: return None, None
        t = max(self.qty) if self.is_bid else min(self.qty)
        return t / self.price_scale, self.qty[t]
    def iter_best(self) -> Iterator[tuple[float,float]]:
        for t in sorted(self.qty, reverse=self.is_bid):
            q = self.qty[t]
            if q > 0: yield t / self.price_scale, q


class _CompiledSide:
    def __init__(self, is_bid: bool, price_scale: float):
        if _NumbaDict is None or _types is None: raise RuntimeError("numba is required for accelerated replay")
        self.is_bid=bool(is_bid); self.price_scale=float(price_scale)
        self.qty=_NumbaDict.empty(key_type=_types.int64,value_type=_types.float64); self.min_tick=0; self.max_tick=0; self.active_count=0
    def clear(self):
        self.qty=_NumbaDict.empty(key_type=_types.int64,value_type=_types.float64); self.min_tick=0; self.max_tick=0; self.active_count=0
    def _tick(self, price: float)->int:
        s=float(price)*self.price_scale; t=int(round(s))
        if abs(s-t)>1e-6: raise RuntimeError(f"price {price} not aligned to price_scale={self.price_scale:g}")
        return t
    def reset(self, levels: Sequence[Any]):
        self.clear(); ticks=[]; qty=[]
        for level in levels or []:
            try:
                if isinstance(level, Mapping): p=float(level.get('price')); q=float(level.get('quantity'))
                else: p=float(level[0]); q=float(level[1])
            except Exception: continue
            if p>0 and q>0: ticks.append(self._tick(p)); qty.append(q)
        self.update_arrays(np.asarray(ticks,dtype=np.int64),np.asarray(qty,dtype=np.float64))
    def update_arrays(self,ticks,quantities):
        self.min_tick,self.max_tick,self.active_count=_apply_kernel(self.qty,np.asarray(ticks,dtype=np.int64),np.asarray(quantities,dtype=np.float64),int(self.min_tick),int(self.max_tick),int(self.active_count))
    def best(self):
        if self.active_count<=0:return None,None
        t=int(self.max_tick if self.is_bid else self.min_tick)
        try:q=float(self.qty[t])
        except KeyError:return None,None
        return t/self.price_scale,q
    def iter_best(self)->Iterator[tuple[float,float]]:
        if self.active_count<=0:return
        t=int(self.max_tick if self.is_bid else self.min_tick); stop=int(self.min_tick if self.is_bid else self.max_tick); step=-1 if self.is_bid else 1
        while t>=stop if self.is_bid else t<=stop:
            if t in self.qty:
                q=float(self.qty[t])
                if q>0: yield t/self.price_scale,q
            t+=step


def acceleration_available() -> bool:
    return _numba is not None


@dataclass(frozen=True)
class BookState:
    query_ts_ns: int
    status: str
    sequence_valid: bool
    bridge_pending: bool
    last_state_ts_ns: int | None
    state_age_ns: int | None
    best_bid: float | None
    best_ask: float | None
    best_bid_qty: float | None
    best_ask_qty: float | None
    mid: float | None
    spread_bps: float | None
    last_update_id: int | None
    bootstrap_source: str
    last_invalid_reason: str | None


class ReplayBook:
    def __init__(self, *, price_scale: float = 10.0, accelerated: bool = False):
        side = _CompiledSide if accelerated else _PythonSide
        self.price_scale=float(price_scale); self.bids=side(True,self.price_scale); self.asks=side(False,self.price_scale)
        self.valid=False; self.bridge_pending=False; self.snapshot_last_update_id=None; self.previous_u=None
        self.last_state_ts_ns=None; self.last_snapshot_ts_ns=None; self.bootstrap_source='NONE'; self.last_invalid_reason='UNBOOTSTRAPPED'
        self.snapshot_count=0; self.depth_count=0; self.bootstrap_count=0; self.sequence_gap_count=0; self.crossed_count=0; self.bridge_failure_count=0
    def _top_ok(self):
        b,_=self.bids.best(); a,_=self.asks.best(); return bool(b is not None and a is not None and b>0 and a>0 and b<a)
    @staticmethod
    def _snapshot_top(levels: Sequence[Any], is_bid: bool):
        bp=None; bq=None
        for x in levels or []:
            try:
                if isinstance(x,Mapping): p=float(x.get('price')); q=float(x.get('quantity'))
                else:p=float(x[0]);q=float(x[1])
            except Exception:continue
            if p<=0 or q<=0:continue
            if bp is None or (p>bp if is_bid else p<bp): bp,bq=p,q
        return bp,bq
    def invalidate(self, reason: str):
        self.valid=False; self.bridge_pending=False; self.snapshot_last_update_id=None; self.previous_u=None; self.last_invalid_reason=str(reason)
        if reason=='PU_SEQUENCE_GAP': self.sequence_gap_count+=1
        elif reason=='CROSSED_BOOK': self.crossed_count+=1
        elif reason=='BOOTSTRAP_LOCAL_BRIDGE_FAILED': self.bridge_failure_count+=1
    def apply_snapshot(self,row: Mapping[str,Any]):
        ts=int(row['ts_recv_ns']); self.snapshot_count+=1; self.last_snapshot_ts_ns=ts
        bid=self._snapshot_top(row.get('bids') or [],True); ask=self._snapshot_top(row.get('asks') or [],False)
        try:last_u=int(row.get('last_update_id')) if row.get('last_update_id') is not None else None
        except Exception:last_u=None
        complete=bid[0] is not None and ask[0] is not None and bid[0]<ask[0] and last_u is not None
        if self.valid or self.bridge_pending: return
        if not complete:return
        self.bids.reset(row.get('bids') or []); self.asks.reset(row.get('asks') or [])
        self.snapshot_last_update_id=last_u; self.previous_u=None; self.bridge_pending=True; self.valid=False; self.last_state_ts_ns=ts; self.last_invalid_reason='BOOTSTRAP_PENDING'
    def apply_depth(self,row: Mapping[str,Any]):
        ts=int(row['ts_recv_ns']); self.depth_count+=1
        U=int(row['first_update_id']); u=int(row['final_update_id']); pu=int(row['previous_final_update_id'])
        if self.bridge_pending:
            L=self.snapshot_last_update_id
            if L is None: self.invalidate('BOOTSTRAP_MISSING_LAST_UPDATE_ID'); return
            if u<=L:return
            if pu!=L:self.invalidate('BOOTSTRAP_LOCAL_BRIDGE_FAILED');return
            self._apply(row); self.previous_u=u; self.bridge_pending=False; self.last_state_ts_ns=ts
            if not self._top_ok(): self.invalidate('CROSSED_BOOK');return
            self.valid=True; self.bootstrap_count+=1; self.bootstrap_source='COMPLETE_LOCALBOOK_CHECKPOINT'; self.last_invalid_reason=None; return
        if not self.valid:return
        if self.previous_u is None or pu!=self.previous_u:self.invalidate('PU_SEQUENCE_GAP');return
        self._apply(row); self.previous_u=u; self.last_state_ts_ns=ts
        if not self._top_ok():self.invalidate('CROSSED_BOOK')
    def _apply(self,row: Mapping[str,Any]):
        if '_fast_bid_ticks' in row:
            self.bids.update_arrays(row.get('_fast_bid_ticks'),row.get('_fast_bid_qty')); self.asks.update_arrays(row.get('_fast_ask_ticks'),row.get('_fast_ask_qty')); return
        for side,key in ((self.bids,'bid_updates'),(self.asks,'ask_updates')):
            ticks=[]; qty=[]
            for x in row.get(key) or []:
                try:
                    if isinstance(x,Mapping):p=float(x.get('price'));q=float(x.get('quantity'))
                    else:p=float(x[0]);q=float(x[1])
                except Exception:continue
                ticks.append(side._tick(p)); qty.append(q)
            side.update_arrays(np.asarray(ticks,dtype=np.int64),np.asarray(qty,dtype=np.float64))
    def state(self,query_ts_ns:int)->BookState:
        q=int(query_ts_ns); b,bq=self.bids.best(); a,aq=self.asks.best(); top_ok=self._top_ok()
        sequence=bool(self.valid and not self.bridge_pending and top_ok)
        age=None if self.last_state_ts_ns is None else q-int(self.last_state_ts_ns)
        if age is not None and age<0: raise ValueError('book state is from the future relative to query')
        status='AVAILABLE' if sequence else ('BRIDGE_PENDING' if self.bridge_pending else str(self.last_invalid_reason or 'BOOK_INVALID'))
        mid=(b+a)/2 if sequence and b is not None and a is not None else None
        spread=((a-b)/mid*10000.0) if mid else None
        return BookState(q,status,sequence,bool(self.bridge_pending),self.last_state_ts_ns,age,b if sequence else None,a if sequence else None,bq if sequence else None,aq if sequence else None,mid,spread,self.previous_u if self.previous_u is not None else self.snapshot_last_update_id,self.bootstrap_source,self.last_invalid_reason)
    def levels_for(self,side:str)->Iterable[tuple[float,float]]:
        s=str(side).upper()
        if s=='BUY':return self.asks.iter_best()
        if s=='SELL':return self.bids.iter_best()
        raise ValueError('side must be BUY or SELL')
