# R0.6 — Execution + Cost Simulator

R0.6 establishes the execution machinery required by Protocol v1.2 before any
fresh hypothesis is allowed to claim economic viability.

It does **not** test whether the historical 7M candidates make money.

## Canonical executable book

R0.6 reconstructs BTCUSDT USD-M perpetual depth directly from canonical:

- `perps/book_snapshots`
- `perps/depth_updates`

ordered by exact integer `ts_recv_ns`.

Stored snapshots are collector-local checkpoints, not blindly trusted full-book
truth. A complete two-sided checkpoint can bootstrap/recover an invalid replay
only when its first later non-stale diff proves exact local continuation:

`pu == snapshot_last_update_id`

Once that handshake succeeds, the diff stream is authoritative. Later snapshots
are audit-only. A real `pu` gap or crossed reconstructed book invalidates
execution state until a later checkpoint successfully handshakes.

Depth updates are absolute quantities; zero deletes the level.

## Marketable fill semantics

A marketable BUY walks visible asks. A marketable SELL walks visible bids.
Insufficient visible depth produces an incomplete fill. R0.6 never extrapolates
beyond visible reconstructed depth.

Raw crossing cost is measured against the reconstructed arrival-time mid. Fees,
funding and other costs are separate explicit assumptions.

Passive-limit execution is deliberately unsupported in v1. A later passive
strategy would require a documented queue/fill model or prospective shadow/live
evidence.

## Latency

`LatencyAssumptions` decomposes:

1. computation;
2. decision;
3. order submission;
4. network transmission;
5. exchange processing;
6. fill completion.

Every component is an explicit nonnegative integer number of nanoseconds. R0.6
freezes no claim that a particular latency profile is realistic.

The full D0 builder uses `ZERO_LATENCY_AUDIT_ONLY` solely to audit whether a
causal executable book exists at the parent R0.5 information timestamp. That is
an idealized lower-bound infrastructure diagnostic and **not** a strategy entry
assumption.

## Audit notional

The builder simulates both BUY and SELL at 1,000 USDT quote notional only to
exercise visible-depth fill mechanics and report coverage. It is not a frozen
research position size. A later hypothesis/position specification must register
its own standardized notional before protected confirmation.

## Freshness

R0.6 records exact `state_age_ns`. It does not invent a universal maximum age.
A future registered strategy must freeze its acceptable book-age rule before
protected outcomes are opened.

## Outputs

```text
data/derived/perps/research/r0_execution_simulator_v1/
  execution/
    audit_zero_latency_state.parquet
  quality/
    candidate_execution_coverage.parquet
  metadata/
    manifest.json

reports/r0_execution_simulator_v1/
  daily_replay_summary.csv
  sequence_gap_summary.csv
  summary.json
```

## Run

```bash
python research/r0/execution/build_d0_execution_foundation.py --self-test
python research/r0/execution/build_d0_execution_foundation.py
```

The full build must finish `BUILD_COMPLETE_REVIEW_REQUIRED`. Review sequence-
valid coverage, state-age distribution and audit-fill completeness before R0.6
is accepted.

## Relationship to R0.3 labels

R0.3 `anchor + 3s` outcomes are corrected legacy scientific labels. They are not
a generic executable entry clock. Once a future strategy freezes nonzero
latency/fill assumptions, its economic return path must begin no earlier than
R0.6 `executable_entry_ts_ns`.
