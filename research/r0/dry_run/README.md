# R0.8 — Full dry run on D0

R0.8 is the integration gate for the scientific reset. R0.1-R0.7 proved the
major pieces separately; R0.8 asks whether those pieces can operate together as
one governed research workflow before we begin fresh hypothesis generation.

R0.8 is **not hypothesis discovery** and must not create a promotable finding.
The fixed dry-run proposition is permanently classified `SCIENTIFIC_ONLY`
regardless of whether its D0 descriptive statistic looks favorable.

## What the dry run exercises

The runner consumes the already-built and hash-verified D0 artifacts:

1. R0.3 corrected candidates/outcomes;
2. R0.4 conservative dependence structure;
3. R0.5 causal trade-state storage;
4. R0.6 executable-book audit state;
5. R0.7 registry and guardrail APIs.

It does not rebuild the expensive R0.5/R0.6 parents and it does not read raw P1
market files.

The flow is:

```text
verify accepted parent hashes + P1 seal
        ↓
construct one fixed causal 10s plumbing state from R0.5
        ↓
freeze candidate/trading eligibility using causal information only
        ↓
attach D0 outcome observability separately
        ↓
apply R0.4 conservative dependence unit
        ↓
require R0.6 executable entry coverage
        ↓
apply one-position-at-a-time 5m state machine
        ↓
compute a mechanical non-promotable D0 return proxy
        ↓
rehearse R0.7 append-only registration in an isolated namespace
        ↓
force terminal classification SCIENTIFIC_ONLY
        ↓
run forbidden-action negative controls
        ↓
prove production registries did not change
```

## Fixed test state

The plumbing state is deliberately simple and frozen before outcomes:

- source: R0.5 1-second trade buckets;
- window: exact trailing `[-10s, 0s)`;
- quantity: raw `buyer_is_maker=false` versus `buyer_is_maker=true` volume;
- score: `(false_qty - true_qty) / (false_qty + true_qty)`;
- state true: score `> 0`.

This does **not** assign a directional market interpretation to
`buyer_is_maker`. It exists only to make the full machinery process a real
causal field. The threshold is not searched or tuned from D0 outcomes.

## Eligibility versus outcome observability

This separation is a central R0.8 acceptance requirement.

Trading eligibility may depend only on information available by the governed
R0.5/R0.6 decision/execution clock:

- governed clock available;
- R0.5 fine history `AVAILABLE`;
- fixed feature available;
- R0.6 book `AVAILABLE` and sequence valid;
- direction-specific audit fill complete.

Only after that eligibility vector is frozen does R0.8 open the R0.3 5-minute
D0 outcome table. `D0_BOUNDARY_CENSOR`, missing future paths, or another future
outcome status may change **outcome observability**, but may never retroactively
change `trading_eligible`.

## Dependence

The descriptive integration statistic uses R0.4
`conservative_inference_cluster_id`. It computes state-true minus state-false
means within clusters that contain both states, then averages those cluster
contrasts equally. A deterministic cluster bootstrap is used only to exercise
the inference machinery.

No significance or economic claim from this synthetic state is promotable.

## Execution and position path

R0.8 reuses the accepted R0.6 `ZERO_LATENCY_AUDIT_ONLY` executable-book state.
That profile and the 1,000-USDT notional remain infrastructure diagnostics, not
realistic strategy assumptions.

A simple state machine enforces realizable capital paths:

```text
FLAT + eligible state=true → ENTER for fixed 5m
OPEN + any new candidate   → IGNORE until exit
```

The mechanical net-return proxy subtracts R0.6 entry crossing cost from the
R0.3 5-minute directional return. It does not simulate exit depth, and therefore
must not be presented as realistic strategy P&L.

## Registry isolation

The synthetic hypothesis, evidence budget, lifecycle, cohort, batch, access
ledger and decision log are created under:

```text
data/derived/perps/research/r0_full_d0_dry_run_v1/isolated_registries/
```

The production R0.7 registries under `research/governance/registries/` are
hashed before and after the run and must remain byte-for-byte unchanged.

The synthetic D1/cohort/batch objects are test objects only. They do not refer to
or authorize access to real P1/D1 data.

## Negative controls

R0.8 deliberately attempts forbidden operations and requires them to fail
closed, including:

- P1 outcome access;
- Discovery Agent D1 access;
- future-derived predictors;
- future-derived eligibility;
- random cross-validation;
- silent same-version material mutation;
- rescuing a terminal hypothesis version in place.

## Idempotence

The output manifest stores a substantive run hash over parent hashes, isolated
registry hashes, candidate/position content, and the summary. `generated_at_utc`
is intentionally volatile. If the same accepted parent hashes and the same implementation fingerprint are
rerun but substantive output changes, R0.8 fails. A legitimate pre-freeze code or
contract change establishes a new baseline instead of being misclassified as
nondeterminism.

## Outputs

```text
data/derived/perps/research/r0_full_d0_dry_run_v1/
  analysis/
    candidate_dry_run.parquet
    position_path.parquet
  isolated_registries/
    *.jsonl
  metadata/
    manifest.json

reports/r0_full_d0_dry_run_v1/
  forbidden_actions.csv
  summary.json
```

## Run

```bash
python -m pytest tests/research_governance/test_r0_full_dry_run.py -v
python research/r0/dry_run/run_full_d0_dry_run.py --self-test
python research/r0/dry_run/run_full_d0_dry_run.py
```

Then rerun the full governance suite:

```bash
python -m pytest tests/research_governance -q -rs
```

## Acceptance

### Acceptance-run correction

The first full-D0 acceptance run exposed one integration-summary defect: R0.3
`mae_from_decision_bps` is a **non-negative adverse-excursion magnitude**, but
R0.8 initially summarized `position_worst_mae_bps` with the minimum. That
reported `0.0` even though the entered-position distribution had real adverse
excursions up to roughly 210 bps. The dry-run analysis population, eligibility,
position path, contrast, registry behavior, P1 firewall, and substantive-run
hash were unaffected; only the summary risk aggregation was wrong. The builder
now fails closed on negative MAE values and defines worst MAE as the maximum
non-negative magnitude. Acceptance must be rerun after this correction before
R0.8 is frozen.

R0.8 can pass only if:

1. all accepted parent hashes are verified before analysis;
2. P1 remains `SEALED` / `UNSEEN` and no P1 market/outcome file is opened;
3. causal eligibility is frozen before future observability is attached;
4. all 7,067 candidate identities remain explicit;
5. dependence uses the frozen R0.4 structure rather than independent rows;
6. execution availability comes from R0.6 rather than a synthetic mid-price;
7. repeated signals are translated through a position state machine;
8. the synthetic registry lifecycle validates and terminates `SCIENTIFIC_ONLY`;
9. all forbidden-action negative controls fail closed;
10. production R0.7 registries are unchanged;
11. same-input, same-implementation substantive reruns are idempotent; legitimate code/contract corrections establish a new implementation-fingerprinted baseline;
12. `research/README.md` is updated in the same research-step commit.

After acceptance, the next phase is R0.9 — the adversarial acceptance audit of
the complete R0 system.


### Pre-freeze correction: implementation-aware idempotence

During R0.8 acceptance, the MAE summary correction legitimately changed the substantive run hash. The original idempotence check compared the corrected run with a manifest produced by the pre-correction implementation and therefore failed even though the corrected implementation itself was deterministic. R0.8 now records a hash fingerprint of the dry-run runner, dry-run core, registry core, and governing contracts. A prior manifest is required to match substantively only when both the accepted parent hashes and this implementation fingerprint are unchanged. A changed implementation establishes a new baseline; a changed substantive result under the same implementation still fails closed.


## Final local acceptance

R0.8 passed local acceptance before Git freeze. The corrected full-D0 implementation:

- ran twice with identical substantive hash `d7024f938942fd35241ce8c37148622032db24ba929a513f754167cf343066fd`;
- preserved 7,067 candidate rows and 7,031 governed/causally eligible rows;
- entered 2,836 positions from 3,453 state-true rows while ignoring 617 signals during existing exposure;
- reported final `position_worst_mae_bps = 209.7608088530789`;
- blocked 7/7 forbidden-action controls;
- made zero production-registry mutations;
- read neither P1 market data nor P1 outcomes;
- passed 22/22 R0.8-specific tests; and
- passed the full research-governance suite with 168 passed, 3 known Windows symlink-capability skips, and 0 failures.

The synthetic dry-run result remains permanently `SCIENTIFIC_ONLY` and is not a trading or hypothesis-discovery result.
