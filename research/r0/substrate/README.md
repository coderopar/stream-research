# R0.3 — Corrected D0 Research Substrate

R0.3 rebuilds the reusable D0 candidate/clock/outcome substrate after the R0.2
causal-clock repair. It is **not** a new hypothesis test and it does not inspect
`P1_PROTECTED`.

## Scientific scope

The R0.3 adapter preserves the historical prospective 25-bp population as one
explicitly named **D0 development candidate family**. This preserves lineage
without declaring 25bp to be uniquely correct or confirmatory.

Historical source review established that two different clocks had previously
been conflated:

1. the **frozen one-second candidate-policy availability clock**; and
2. the **recovered exact book-state anchor clock**.

The Phase 7M.2/7K.1 candidate population was generated from a one-second price
series. A candidate can therefore be treated as causally available only when its
one-second observation is complete. The governed candidate-availability clock is
thus the deterministic interval-close boundary:

```text
candidate_availability_ts = (candidate_second + 1 second)
```

This boundary is a policy-observability clock. It is **not** presented as an
exact raw-trade timestamp and it is never substituted for the exact book anchor.

The exact book anchor is recovered independently from physically int64 receive
timestamps. It may occur before the candidate-policy availability boundary within
the same second. That ordering is valid and must not be rejected.

The governed scientific-label clock is therefore:

```text
candidate_detection_ts   = frozen one-second policy interval close
anchor_ts                 = recovered exact frozen-book state
feature_window_start      = anchor_ts
feature_window_end        = anchor_ts + 3s
information_available     = anchor_ts + 3s
decision_ts               = anchor_ts + 3s
outcome_start_ts           = anchor_ts + 3s

required:
  candidate_detection_ts <= information_available_ts
  anchor_ts              <= information_available_ts

not required:
  candidate_detection_ts <= anchor_ts
```

The execution-related R0.2 fields are set equal to decision time **only to audit
scientific label timing**. They are not a fill model and must not be used for
P&L. R0.6 supplies the execution simulator.

## Why R0.3.2 changed the earlier draft

The first full R0.3 build retained 7,067 candidates and rebuilt all 28,268
candidate-horizon rows, but 403 candidates were rejected solely because a
reconstructed intrasecond 25bp trade occurred after the recovered book anchor.
Review of the actual 7M.3/7M.6 source showed that this ordering was not a valid
invariant: the book anchor and the candidate-policy clock represent different
objects. The historical 7K.1 state machine also explicitly permits same-second
release/candidate transitions for large one-second jumps, so R0.3 must preserve
`origin_second <= release_second <= candidate_second` rather than strengthen it
to strict inequalities.

R0.3.2 corrects those assumptions instead of treating the 403/9 exclusions as
coverage loss.

## Exact-anchor source

Preferred source:

`data/derived/perps/research/prospective_25bp_refill_redepletion_sequence_v2/`

R0.3 refuses to recover an epoch-nanosecond anchor by casting a historical
floating-point timestamp. It prefers physically int64 persisted anchors. The
7M.6F `level_fate` checkpoints have an explicit int64 `anchor_ts_recv_ns`
schema and are used as the fallback when nullable candidate metrics were
materialized as float.

Candidates without an exact recoverable anchor remain in
`causal/candidates.parquet` with `clock_available=false`; they are not silently
deleted.

## Future-label rebuild

Historical `candidate_decision_outcomes.parquet` is **not reused**.

Future labels are rebuilt from canonical BTCUSDT perp trades using receive-time
int64 nanoseconds. The path begins strictly after the exact
`outcome_start_ts_ns`. The decision price is the latest canonical trade at or
before the decision time, subject to the frozen five-second stale-price guard.

R0.3 emits the fixed lineage matrix:

- horizons: 5m, 10m, 15m, 30m;
- legacy origin targets: 35bp, 50bp, 75bp, 100bp;
- MFE/MAE and directional return from the corrected decision point.

These are D0 development labels, not confirmation claims.

### Safe reuse after the clock-semantics correction

If a prior R0.3 build already exists, builder v1.0.3 may reuse only prior
`AVAILABLE` outcome rows whose exact `outcome_start_ts_ns` is unchanged and
whose frozen candidate input hash and R0.1 D0 manifest hash are unchanged.
Newly admitted clocks are recomputed from canonical D0 data. This is reuse of
already rebuilt R0.3 outputs, **not** reuse of historical Phase-7M outcome files.

Use `--no-reuse-existing-outcomes` to force a full rebuild.

## P1 protection

The builder will not open canonical or execution partitions dated 2026-07-19 or
later. Late-D0 candidates are retained, but any horizon reaching the P1 boundary
is emitted as `D0_BOUNDARY_CENSOR` with null outcome values.

## Outputs

```text
data/derived/perps/research/r0_corrected_d0_substrate_v1/
  causal/
    candidates.parquet
    clocks.parquet
  future/
    outcomes.parquet
  metadata/
    manifest.json

reports/r0_corrected_d0_substrate_v1/
  clock_coverage.csv
  outcome_coverage.csv
  summary.json
```

The physical separation between `causal/` and `future/` is deliberate.

## Run

First run the tests:

```bash
python -m pytest \
  tests/research_governance/test_r0_clock_contract.py \
  tests/research_governance/test_r0_d0_substrate.py \
  -v
```

Then build the corrected D0 substrate:

```bash
python research/r0/substrate/build_corrected_d0_substrate.py
```

With an existing v1.0.2 R0.3 build, unchanged AVAILABLE outcomes are reused only
under the exact-start/hash guards above, so the correction does not require
recomputing the already-valid 6,619 candidate paths.

## Acceptance intent

R0.3 passes only if:

1. every original D0 candidate is retained in the candidate catalog;
2. the frozen candidate policy availability boundary is represented separately from the exact book anchor;
3. same-second release/candidate transitions allowed by the frozen state machine are preserved;
4. exact anchors are never reconstructed from integer `candidate_second`;
5. float epoch-nanosecond anchor coercion is refused;
6. every emitted governed clock passes the revised R0.2 contract;
7. future labels start at exact anchor + 3s;
8. no historical Phase-7M future-outcome file is used;
9. no P1 partition is opened;
10. unavailable clocks/outcomes are represented explicitly rather than silently dropped;
11. the full frozen lineage contains exactly 7,067 candidates and 28,268 candidate-horizon outcome rows.
