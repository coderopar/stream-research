# R0.2 — Causal Clock Repair

R0.2 establishes the causal clock contract that all later research must obey.
It does **not** rerun the old hypothesis program and it does **not** access
`P1_PROTECTED` outcomes.

## Why this exists

The historical Phase 7M lineage used an integer `candidate_second` in at least
one outcome construction path. Later exact replay showed that the true anchor
could occur materially inside that second. When a feature window was measured
from the exact anchor but an outcome window was measured from
`candidate_second + offset`, the outcome could begin before the feature window
ended. That is causal leakage.

R0.2 therefore makes exact clocks a governed primitive rather than an
analysis-specific convention.

## Contract

The authoritative machine-readable contract is:

`research/governance/clock_contract_v1.json`

All governed timestamps are signed int64 Unix nanoseconds. Floats are rejected.
The minimum execution/outcome ordering is:

```text
information_available_ts_ns
    <= decision_ts_ns
    <= order_send_ts_ns
    <= estimated_exchange_arrival_ts_ns
    <= executable_entry_ts_ns
    <= outcome_start_ts_ns
```

The feature window must also end no later than `outcome_start_ts_ns`.

## D0-only rule

R0.2 is restricted to D0:

- start: `2026-05-18T00:00:00Z`
- end exclusive / P1 start: `2026-07-19T00:00:00Z`

The guard checks the requested maximum outcome horizon as well as the stored
clock fields. A late-D0 candidate whose horizon would cross into P1 is refused.

## Legacy candidate-second rule

`candidate_second` is accepted only by the diagnostic overlap function. It is
never promoted into a repaired governed timestamp. Repair requires exact
source timestamps.

## Audit a D0 clock table

The CLI expects a CSV with the ten governed `*_ts_ns` columns. It validates
clock type/order, refuses P1, and writes an audit summary without reading
market values or outcomes.

```bash
python research/r0/clock_repair/audit_d0_clock_contract.py \
  --input path/to/d0_clock_rows.csv \
  --max-outcome-horizon-seconds 1800 \
  --report outputs/r0_clock_audit/report.json
```

For R0.2, do not point this command at a P1-derived table. The command fails
closed if any row or requested horizon reaches P1.

## Acceptance tests

```bash
python -m pytest tests/research_governance/test_r0_clock_contract.py -v
```

The regression suite includes a 950 ms candidate-second overlap case to prove
that the historical failure mode is detected.

## Quantify the historical candidate-second overlap

If a D0-only table contains both the old integer `candidate_second` and an exact
`anchor_ts_ns`, quantify the timing defect without reading outcomes:

```bash
python research/r0/clock_repair/audit_legacy_candidate_second_overlap.py \
  --input path/to/d0_candidates.csv \
  --report outputs/r0_clock_audit/legacy_overlap.json \
  --legacy-outcome-offset-seconds 3 \
  --exact-feature-offset-seconds 3
```

A positive overlap is evidence that the legacy outcome interval begins before
the exact-anchor feature interval has finished. The utility never uses
`candidate_second` to construct a repaired clock.


## Semantic correction discovered during R0.3 acceptance

Review of the actual Phase 7M.2/7M.3/7M.6 source showed that
`candidate_detection_ts_ns` and `anchor_ts_ns` represent different clocks.
The candidate population was generated from a one-second price policy, while
the recovered anchor is an exact book-state receive timestamp inside that
second. The recovered book state may therefore precede the point at which the
frozen one-second candidate observation is causally available.

R0 clock contract revision 1.1 removes the unsupported invariant
`candidate_detection_ts_ns <= anchor_ts_ns` and instead requires both candidate
availability and the exact anchor to be known by `information_available_ts_ns`.
For the frozen one-second policy, the conservative candidate-availability clock
is the interval-close boundary `(candidate_second + 1 second)`. This boundary
is not an exact raw market-event timestamp and must never replace the exact
book anchor.
