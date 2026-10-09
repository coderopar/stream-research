# R0.9 — Adversarial R0 Acceptance Audit

R0.9 is the final gate of the scientific-infrastructure reset. It does **not** search for a trading signal, optimize a threshold, inspect P1 market data, or inspect P1 outcomes. Its job is to assume that the R0 machinery may still contain hidden scientific/process defects and try to falsify its readiness for R1.

## Core question

> Is the complete R0 research machine trustworthy enough that fresh D0 hypothesis generation can begin without silently inheriting the defects that invalidated the old research process?

This is stricter than asking whether the code runs. R0.9 treats every earlier `PASS`, `accepted`, or `frozen` label as a researcher claim that must be checked against Git history, governance contracts, runtime hashes, production registries, accepted limitations, and cross-component consistency.

## What R0.9 attacks

The machine audit covers:

1. **Git freeze lineage** — accepted checkpoints must exist, form a monotone ancestor chain, and map to the scientific record.
2. **Protocol and provenance** — Protocol v1.2 must still match its frozen SHA-256 identity, and the authoritative R0.1 inventory must still describe the accepted canonical boundary.
3. **P1 firewall** — P1 must remain `SEALED` and outcome-unseen; the outcome-access register must show no protected-outcome exposure; production research registries must remain clean before R1.
4. **R0.8 reproducibility** — the accepted full-D0 dry-run hash, no-P1/no-registry-mutation invariants, semantic implementation fingerprint, and all R0.3–R0.6 parent hashes are reverified.
5. **Cross-component contracts** — all causal-clock, dependence, market-state, execution, registry, dry-run, and audit contracts must still be present.
6. **Documentation / manifest consistency** — README and implementation-manifest claims must agree with what was actually frozen. Stale “pending” metadata is a finding, not silently repaired before inspection.
7. **Residual limitations** — accepted weaknesses are carried forward explicitly rather than disappearing when R0 is declared complete.

## Finding model

Every audit row receives:

- stable `finding_id`;
- domain;
- severity: `INFO`, `MINOR`, `MAJOR`, or `BLOCKER`;
- status: `PASS`, `FINDING`, `ACCEPTED_LIMITATION`, or `NOT_TESTABLE`;
- affected R0 stages;
- evidence;
- remediation;
- explicit `blocking_r1` flag.

The R1 gate is deterministic:

```text
open BLOCKER finding            -> NO_GO_BLOCKER
other R1-blocking finding       -> REVIEW_REQUIRED
accepted limitations only       -> PASS_WITH_ACCEPTED_LIMITATIONS
no open findings                -> PASS
```

A first R0.9 run is allowed to return `REVIEW_REQUIRED`. That is not an audit failure. It means the audit found something that must be reconciled before R1.

## Important distinction: defect versus accepted limitation

R0.9 does not pretend every imperfection is a blocker.

Examples of **accepted limitations** that must remain explicit:

- 36/7,067 historical candidates have unrecoverable governed clocks;
- candidate/outcome dependence is severe at longer horizons;
- R0.5 v1 only implements the `PERPS_TRADES` state family;
- R0.6/R0.8 use an audit-only zero-added-latency and 1,000-USDT execution profile rather than a final strategy profile;
- R0.6 does not persist exact sequence-gap recovery-duration intervals;
- repository artifacts cannot forensically prove that no process outside the governed workflow ever read a P1 file, so P1 cleanliness remains a procedural evidence claim backed by the sealed register, access ledger, source guards, and dry-run evidence.

These limitations do not block R1 because R1 is D0-only idea generation, but they constrain what later stages may claim.

## What R0.9 does not establish

Even a clean R0.9 result does **not** establish:

- that the old 25bp candidate definition is correct;
- that any old 7M association is confirmatory;
- that a profitable strategy exists;
- that zero latency or 1,000 USDT is a realistic production execution model;
- that P1 should be one D1 cohort or a D1/reserve partition;
- that a future hypothesis will survive D1, D2, or D3.

It establishes only that the **research infrastructure and governance process are fit to begin R1 on D0**.

## Manual adversarial review requirement

The automated runner is deliberately strong on reproducibility, lineage, protection, and cross-component seams, but it cannot replace scientific judgment. Before the final R0.9 verdict, the finding ledger should also be reviewed for:

- whether any R0 correction could have left downstream contamination;
- whether any component silently relies on a stronger semantic claim than its source supports;
- whether a deferred limitation has accidentally become an assumed capability;
- whether a PASS label is based only on tests rather than evidence at the required scientific level;
- whether a known limitation should be upgraded to a blocker before R1.

No new hypothesis or outcome search should be introduced as part of that review.

## Run

Self-test:

```bash
python research/r0/audit/run_r0_acceptance_audit.py --self-test
```

R0.9-specific tests:

```bash
python -m pytest tests/research_governance/test_r0_acceptance_audit.py -v
```

Real audit:

```bash
python research/r0/audit/run_r0_acceptance_audit.py
```

Outputs are written to:

```text
reports/r0_acceptance_audit_v1/findings.csv
reports/r0_acceptance_audit_v1/audit_summary.json
```

The runtime audit reads governance metadata and accepted **D0** artifacts only. It does not open P1 market partitions or P1 outcomes.

## Acceptance workflow

1. Run the audit once without pre-cleaning expected metadata drift.
2. Inspect every `FINDING` and `NOT_TESTABLE` row.
3. Fix only the defects necessary to restore scientific/process integrity; do not use R0.9 to optimize market hypotheses.
4. Update `research/README.md` with the finding and resolution history.
5. Rerun R0.9.
6. Run the full `tests/research_governance` suite.
7. Only when zero R1-blocking findings remain may R0.9 be frozen and R1 opened.

## First real audit and remediation record

The first real R0.9 audit returned `REVIEW_REQUIRED` with substantive audit hash `a85ea4c119c9e669191edd8e2f5fdacfe8f73e4e0c6045f0c4dca6fcf63bffdf`. It produced 23 ledger rows: 14 `PASS`, 6 `ACCEPTED_LIMITATION`, 3 `FINDING`, 0 `BLOCKER`, and 0 `NOT_TESTABLE`.

The three findings were all documentation/acceptance-metadata drift:

- `R09-DOC-001` — living READMEs still described R0.7/R0.8 Git freezes as pending;
- `R09-DOC-002` — R0.5-R0.8 implementation manifests still carried pre-acceptance status values;
- `R09-DOC-003` — R0.0/R0.1 governance manifests still carried pre-execution status text.

The remediation is metadata-only. It does not modify market data, D0 research artifacts, P1 declarations, protected-data access history, production registries, hypothesis definitions, causal clocks, dependence structures, market-state data, execution results, the R0.8 substantive run, or the frozen Protocol-v1.2 document SHA-256. R1 remains closed until a fresh R0.9 run verifies these findings are closed.

## Audit idempotence identity

R0.9 substantive idempotence is scoped to the same accepted R0.8 parent, the same R0.9 implementation fingerprint, and the same audit-input fingerprint. The audit-input fingerprint covers the governance/documentation files and production-registry state that can change audit findings, the accepted R0.8 runtime manifest, Git HEAD, and required frozen-tag resolutions. A deliberate remediation to audited governance metadata therefore establishes a new baseline; a repeated audit of an unchanged state must reproduce the same substantive audit hash or fail closed.


## Final acceptance record

After the metadata-only remediation and audit-identity hardening, R0.9 was rerun twice against the same accepted parent, implementation, and audited repository state. Both runs returned `PASS_WITH_ACCEPTED_LIMITATIONS` with identical substantive audit hash:

```text
3faab2f04bb4902d19d90a0a9c6df49b42e62b648572c6316110ea869b87accd
```

Final ledger: 23 rows total, 17 `PASS`, 6 `ACCEPTED_LIMITATION`, 0 open findings, 0 majors, 0 blockers, 0 `NOT_TESTABLE`, and 0 R1-blocking rows. P1 market data and P1 outcomes were not read.

Acceptance regression on the project `.venv`:

- R0.9-specific suite: 26 passed, 0 failed, 0 skipped;
- full `tests/research_governance` suite: 194 passed, 3 skipped, 0 failed;
- the three skips are the known Windows symlink-capability checks from R0.1.

The six carried limitations were reviewed explicitly and accepted as non-blocking for D0-only R1 hypothesis generation. They remain binding constraints on later research: unavailable governed clocks for 36 historical candidates, long-horizon dependence, limited R0.5 state-family coverage, audit-only execution latency/notional, missing exact sequence-gap recovery duration, and the epistemic limit that repository evidence cannot prove the absence of every possible external P1 read.

R0.9 is therefore accepted as `PASS_WITH_ACCEPTED_LIMITATIONS`. R1 may open only after this accepted state is frozen in Git.
