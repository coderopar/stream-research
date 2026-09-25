# R0.5 — Causal Market-State Store

R0.5 creates reusable **outcome-blind causal state primitives** at the R0.3
governed information cutoff. It is infrastructure, not feature selection and
not a hypothesis test.

## v1 scope

The first governed family is canonical Binance BTCUSDT perps trades. The
implementation deliberately does **not** import the historical Phase-7M feature
zoo. Order-book, OI/funding/basis, spot/cross-venue, and options state are
recorded as deferred until each family has a source-specific causal availability
contract.

This is intentional. Unsupported semantics are safer as `DEFERRED` than as a
plausible-looking but unaudited feature.

## Causal cutoff

For every governed candidate, R0.5 uses:

`R0.3 information_available_ts_ns`

Rules:

- as-of state may use only `ts_recv_ns <= cutoff`;
- trailing buckets are exact half-open `[start, cutoff)` intervals;
- a trade exactly at the cutoff may be the latest-prior state but is excluded
  from the trailing bucket ending at that cutoff;
- nanosecond timestamps remain integer-valued;
- no future interpolation or nearest-neighbor join is allowed.

## Trade primitive storage grid

The store writes candidate-relative additive buckets at two resolutions:

- 1-second buckets over the trailing 60 seconds;
- 1-minute buckets over the trailing 30 minutes.

These are **storage resolutions, not scientific feature definitions**. A later
R1+ hypothesis must still register its exact window and transform. The grid
exists to avoid repeatedly scanning raw trades for common causal history while
not choosing windows from continuation outcomes.

Raw `buyer_is_maker` states are retained as true/false/unknown quantity totals.
R0.5 does not relabel them into a directional trading hypothesis.

## Physical separation

Outputs:

```text
data/derived/perps/research/r0_causal_market_state_v1/
  causal/
    candidate_core.parquet
    trade_state.parquet
    trade_buckets_1s.parquet
    trade_buckets_1m.parquet
  governance/
    dependence_linkage.parquet
  quality/
    trade_coverage.parquet
  metadata/
    manifest.json
```

R0.4 overlap/episode/cluster fields live only in `governance/` and are marked
`GOVERNANCE_ONLY`; they are not eligible predictive features.

The 36 R0.3 clock-unavailable candidates remain in `candidate_core` and
`trade_state` with explicit `CLOCK_UNAVAILABLE` state. They do not silently
disappear.

## Runtime I/O strategy (v1.0.1)

The builder does not scan complete current/previous UTC trade days for every
candidate day. It indexes Parquet row-group `ts_recv_ns` min/max statistics and
reads only row groups intersecting the union of required trailing 30-minute
causal histories. Exact bucket boundaries are still applied after loading.

This is an I/O optimization only, not a scientific-semantic change. If a
candidate has no BTCUSDT trade inside its pruned 30-minute history, the builder
falls back to the original full current/previous-D0-day loader for that day so
latest-prior state remains exact. Row groups without usable timestamp statistics
are read conservatively.

The output manifest records files/row-groups indexed and read plus the number of
full-day fallback groups.

## Run

```bash
python research/r0/market_state/build_d0_market_state_store.py --self-test
python research/r0/market_state/build_d0_market_state_store.py
```

The full builder must finish with `BUILD_COMPLETE_REVIEW_REQUIRED`. Review
coverage, source status, tie ambiguity and row counts before declaring R0.5
accepted.

## Acceptance requirements

- 7,067 candidate identities preserved;
- 7,031 governed clocks and 36 explicit unavailable clocks preserved;
- no P1 reads;
- no R0.3 future outcome artifact opened;
- exact int64 receive timestamps only;
- as-of selection never chooses a future record;
- trailing windows are exact half-open intervals;
- missing source partitions/history are explicit, never converted to zeros;
- equal-timestamp ambiguity is explicit when no canonical tie key exists;
- R0.4 governance fields remain physically non-feature state;
- output hashes and source-path/schema provenance are emitted.
