# R0.4 — Dependence / Episode Framework

R0.4 makes candidate dependence explicit before any new hypothesis research.
Protocol v1.2 states that raw candidate count is not independent evidence and
requires the episode/dependence construction used for confirmation to be
deterministic, documented, and frozen in D0 rather than selected after seeing
which definition gives the most favorable inference.

R0.4 is therefore infrastructure. It does not test whether any state predicts
continuation.

## Inputs and firewall

The builder opens only:

```text
data/derived/perps/research/r0_corrected_d0_substrate_v1/
  causal/candidates.parquet
  causal/clocks.parquet
  metadata/manifest.json
```

It verifies the R0.3 manifest hashes and accepted corrected counts:

- 7,067 candidates;
- 7,031 governed clocks;
- 36 explicit clock-unavailable candidates.

It does **not** open `future/outcomes.parquet`. It does not read return, MFE,
MAE, P35/P50/P75/P100, or other future outcome values. It does not open
`P1_PROTECTED`.

## Frozen dependence structures

### 1. Exact forward-window overlap

For each governed R0.3 clock, construct four theoretical label intervals from
the already-governed `outcome_start_ts_ns`:

```text
5m  = [outcome_start, outcome_start + 5m)
10m = [outcome_start, outcome_start + 10m)
15m = [outcome_start, outcome_start + 15m)
30m = [outcome_start, outcome_start + 30m)
```

Intervals are **half-open**. If one interval ends exactly when another begins,
they do not overlap. Components are transitive: A-B and B-C overlap puts A, B,
and C in one connected component even if A and C do not directly overlap.

The 36 candidates without governed R0.3 clocks remain in the output but have
null overlap-component fields.

### 2. Policy temporal episodes

A horizon-independent temporal episode connects consecutive frozen candidate
availability timestamps when their gap is at most **60 seconds**.

The 60-second threshold is not inferred from D0 performance. It is inherited
from the already-frozen `EARLY_25BP_RESET5_RELEASE10_REARM60` candidate policy's
cooldown/rearm interval.

### 3. Directional episodes

Within each 60-second policy temporal episode, candidates are partitioned by
exact direction (`UP/+1` vs `DOWN/-1`). This is an exact categorical partition,
not a globally ordered run rule, so simultaneous opposite-direction candidates
do not receive an arbitrary causal ordering.

### 4. Exact origin-lineage groups

Candidates sharing the exact tuple:

```text
(direction, legacy_origin_second)
```

receive the same origin-lineage episode. This preserves already-recorded move
lineage without inventing a new distance or volatility threshold.

### 5. Conservative inference cluster

R0.4 additionally creates a conservative clustering unit equal to the union of:

1. all candidates sharing the same UTC **candidate-availability date**; and
2. any dates joined by a 30-minute overlap component crossing midnight.

This preserves plausible same-day latent-regime dependence while preventing an
overlapping forward path from being split merely because UTC midnight occurs in
the middle of it.

R0.4 freezes the structure, not a universal hypothesis-specific estimator. A
future D1 confirmation plan must pre-register either this conservative unit or
an explicit hierarchy/sensitivity rule in which the conservative registered
result governs the primary claim.

## Outputs

```text
data/derived/perps/research/r0_dependence_framework_v1/
  dependence/
    candidate_dependence.parquet
  metadata/
    manifest.json

reports/r0_dependence_framework_v1/
  component_summary.csv
  gap_summary.csv
  day_concentration.csv
  summary.json
```

Candidate-level output includes:

- candidate availability timestamp/date and direction;
- R0.3 clock availability/reason;
- 5m/10m/15m/30m overlap component IDs and sizes;
- 60s policy temporal episode ID/size;
- direction-partitioned episode ID/size;
- exact origin-lineage episode ID/size;
- conservative day/30m inference-cluster ID/size.

## Run

```bash
python -m pytest \
  tests/research_governance/test_r0_clock_contract.py \
  tests/research_governance/test_r0_d0_substrate.py \
  tests/research_governance/test_r0_dependence.py \
  -v

python research/r0/dependence/build_d0_dependence_framework.py --self-test
python research/r0/dependence/build_d0_dependence_framework.py
```

## Acceptance intent

R0.4 is acceptable only if:

1. all 7,067 candidates are preserved exactly once;
2. exactly the accepted 7,031 candidates receive overlap components;
3. the 36 clock-unavailable candidates remain explicit;
4. no future outcome values or P1 data are read;
5. interval touching is non-overlap and transitive chains are represented by one component;
6. 5m components refine 10m, which refine 15m, which refine 30m;
7. the 60-second episode threshold is inherited from the frozen candidate policy, not selected from outcomes;
8. simultaneous candidates do not rely on arbitrary row ordering for direction partitioning;
9. cross-midnight 30m overlaps union the corresponding UTC-day inference clusters;
10. component, gap, and day concentration diagnostics are emitted for review before the phase is frozen.
