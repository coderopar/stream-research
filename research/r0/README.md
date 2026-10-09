# R0 — Scientific reset and substrate repair

This directory tracks the governed R0 scientific-reset work packages under Protocol v1.2.

For the cross-phase scientific narrative, plain-language interpretation, acceptance evidence, known limitations, and the rule that every research-step commit must update the research record, see `research/README.md`. This R0 README remains the phase-local implementation index.

## R0.0 — Governance freeze

Implemented by:

- `research/governance/research_reset_manifest.yaml`
- `research/governance/protected_data_register.yaml`
- `research/governance/outcome_access_register.csv`
- `research/governance/protocol_freeze_manifest.json`
- `research/governance/research_lineage_reset.md`
- `research/protocol/v1.2/BTCUSDT_Scientific_Trading_Research_Protocol_v1.2.docx`

The reset declares the old 2026-05-18 through 2026-07-18 period as D0 development data
and the newly added post-2026-07-18 extension as `P1_PROTECTED` with outcomes unseen.

## R0.1 — Dataset inventory and provenance

Implemented by:

- `research/r0/provenance/build_dataset_inventory.py`
- `research/r0/provenance/README.md`
- `research/governance/datasets/README.md`

Run the scanner in the actual repository. The repository's canonical entry-point may
be a symlink to the external canonical store; R0.1 explicitly supports that topology
while refusing derived/outcome/report roots.

R0.1 is frozen at tag `r0-governance-provenance-v1`.

## R0.2 — Causal clock repair

Implemented by:

- `src/research_core/clocks/contract.py`
- `research/r0/clock_repair/`
- `research/governance/clock_contract_v1.json`
- `tests/research_governance/test_r0_clock_contract.py`

R0.2 replaces integer-second timing assumptions with an explicit int64-nanosecond
clock contract and regression tests for the historical feature/outcome overlap.

## R0.3 — Corrected D0 research substrate

Implemented by:

- `src/research_core/substrate/d0.py`
- `research/r0/substrate/build_corrected_d0_substrate.py`
- `research/r0/substrate/README.md`
- `research/r0/substrate/R0_3_IMPLEMENTATION_MANIFEST.json`
- `tests/research_governance/test_r0_d0_substrate.py`

R0.3 retains the historical 25-bp population as an explicitly labeled D0
lineage family. It represents the frozen one-second candidate policy as available
at interval close, keeps that policy clock distinct from the recovered exact
book-state anchor, preserves same-second release/candidate transitions permitted
by the historical state machine, rebuilds the +3s scientific label clock under
R0.2, and reconstructs future labels from canonical D0 trades without using the
old Phase-7M outcome tables or opening P1. The accepted build preserves all
7,067 candidates, governs 7,031 exact clocks, and retains the 36 upstream
level-audit omissions explicitly as clock-unavailable.

## R0.4 — Dependence / episode framework

Implemented by:

- `src/research_core/dependence/`
- `research/governance/dependence_contract_v1.json`
- `research/r0/dependence/`
- `tests/research_governance/test_r0_dependence.py`

R0.4 freezes outcome-blind dependence structures before new research: exact
5m/10m/15m/30m forward-window overlap components, a 60-second temporal episode
inherited from the frozen candidate-policy cooldown, direction and origin-lineage
partitions, and a conservative inference cluster that unions UTC availability-day
clustering with cross-midnight 30-minute overlap. It reads R0.3 causal candidates
and clocks only; future outcome values and P1 remain unopened.


## R0.5 — Causal market-state store

Implemented by:

- `src/research_core/market_state/`
- `research/governance/market_state_contract_v1.json`
- `research/r0/market_state/`
- `tests/research_governance/test_r0_market_state.py`

R0.5 freezes reusable outcome-blind causal state interfaces at the R0.3
`information_available_ts_ns` cutoff. Version 1 implements canonical BTCUSDT
perps-trade primitives plus explicit freshness/coverage/provenance, while R0.4
dependence identifiers remain physically separated as governance-only metadata.
Order-book, derivatives, cross-venue and options families are deferred until
their source-specific causal availability semantics are independently governed.

## R0.6 — Execution + cost simulator

Implemented by:

- `src/research_core/execution/`
- `research/governance/execution_contract_v1.json`
- `research/r0/execution/`
- `tests/research_governance/test_r0_execution.py`

R0.6 reconstructs canonical BTCUSDT perps order-book state from collector-local
checkpoints plus continuous diff-depth updates and provides explicit marketable
fill, latency and friction primitives. The full D0 build uses a zero-latency,
1,000-USDT **audit-only** profile to measure replay/fill coverage; those audit
values are not a registered trading strategy assumption. Passive fills and own-
market impact remain unsupported until separately justified.

## R0.7 — Research registries + guardrails

Implemented by:

- `src/research_core/registry/`
- `research/governance/research_registry_contract_v1.json`
- `research/governance/registries/`
- `research/r0/registries/`
- `tests/research_governance/test_r0_registry_guardrails.py`

R0.7 installs append-only hypothesis/genealogy, lifecycle, evidence-budget,
confirmation-batch, access-contamination and decision registries plus executable
Protocol-v1.2 guardrails. It creates no hypotheses, does not partition P1, and
opens neither protected causal data nor future outcomes. Registration is blocked
until the position, execution, benchmark/MEE, dependence, risk, evidence-control,
D1-feasibility and fresh-context adversarial-review requirements are complete.

Acceptance evidence on the Windows development environment: R0.7-specific tests 37/37 passed; the full `tests/research_governance` suite passed 146 tests with 3 R0.1 symlink-capability skips and zero failures. R0.7 is frozen at commit `c02364e86aee2ec8d6d7eb9821f65e2a5f30d0c8`, tag `r0-research-guardrails-v1`.

## R0.8 — Full dry run on D0

Implemented by:

- `src/research_core/dry_run/`
- `research/governance/r0_dry_run_contract_v1.json`
- `research/r0/dry_run/`
- `tests/research_governance/test_r0_full_dry_run.py`

R0.8 integrates the accepted R0.3-R0.7 machinery using one deliberately
non-promotable D0 plumbing hypothesis. It verifies parent hashes, builds a fixed
causal test state from R0.5, freezes eligibility before opening D0 outcomes,
uses the R0.4 conservative dependence unit, requires R0.6 executable-book
coverage, applies a one-position-at-a-time state machine, rehearses R0.7
registration in an isolated namespace, and proves forbidden research paths fail
closed. The synthetic hypothesis is forced to terminate `SCIENTIFIC_ONLY`
regardless of its D0 descriptive result. Production registries and P1 remain
untouched.

R0.8 acceptance is PASS and frozen at commit `53eac195e83b5fe08397ca50ffc1aadbd201c9a1`, tag `r0-full-d0-dry-run-v1`. The corrected full-D0 build preserved all 7,067 candidates, had 7,031 governed/causally eligible rows, entered 2,836 non-overlapping positions from 3,453 state-true rows, blocked all 7/7 forbidden-action controls, made zero production-registry mutations, and read neither P1 market data nor P1 outcomes. The MAE summary-only defect was corrected before freeze; the final position-level worst MAE is 209.7608088530789 bps.

The corrected implementation ran twice with identical substantive hash `d7024f938942fd35241ce8c37148622032db24ba929a513f754167cf343066fd`. R0.8 acceptance also hardened idempotence: substantive equality is required only for the same accepted parents and the same implementation fingerprint; a legitimate code/contract correction creates a new baseline while true same-implementation nondeterminism still fails closed.

Acceptance evidence on the Windows development environment: R0.8-specific tests 22/22 passed; the full `tests/research_governance` suite passed 168 tests with 3 known R0.1 symlink-capability skips and zero failures.

## R0.9 — Adversarial R0 acceptance audit

Implemented by:

- `src/research_core/audit/`
- `research/governance/r0_acceptance_audit_contract_v1.json`
- `research/r0/audit/`
- `tests/research_governance/test_r0_acceptance_audit.py`

R0.9 is the final falsification-first infrastructure gate before R1. It does not search for market hypotheses or inspect P1. It re-verifies the R0 freeze lineage, Protocol-v1.2 identity, canonical provenance, P1 seal/access ledger, production-registry cleanliness, accepted R0.8 runtime and implementation fingerprint, R0.3-R0.6 parent hashes, critical contracts, documentation/manifest consistency, and all carried-forward scientific limitations.

The first real audit is intentionally permitted to return `REVIEW_REQUIRED`. Findings are recorded with stable IDs, severity, evidence, remediation and an explicit `blocking_r1` flag. R0 is not complete until the audit is rerun with zero R1-blocking findings and the full research-governance regression remains clean.

The first real R0.9 audit returned `REVIEW_REQUIRED`: 14 PASS rows, 6 accepted limitations, 3 MAJOR R1-blocking findings, 0 blockers and 0 `NOT_TESTABLE`. The three findings were exclusively stale documentation/acceptance metadata (`R09-DOC-001`, `R09-DOC-002`, `R09-DOC-003`). The remediation changed metadata only and left scientific artifacts, Protocol-v1.2 SHA-256, D0 parent artifacts, P1 status, registries and runtime results unchanged.

R0.9 idempotence was hardened after the metadata-remediation rerun: a changed audited governance/documentation state establishes a new audit baseline, while a repeated audit of the same accepted parent + R0.9 implementation + audit-input fingerprint must reproduce the same substantive audit hash or fail closed.

Final acceptance is `PASS_WITH_ACCEPTED_LIMITATIONS`. Two consecutive post-remediation audits produced identical substantive hash `3faab2f04bb4902d19d90a0a9c6df49b42e62b648572c6316110ea869b87accd`. The final ledger contains 17 PASS rows, 6 accepted limitations, 0 open findings, 0 majors, 0 blockers, 0 `NOT_TESTABLE`, and 0 R1-blocking rows. R0.9-specific tests passed 26/26; the full research-governance suite passed 194 tests with the same 3 known Windows symlink-capability skips and zero failures. P1 market data and P1 outcomes remained unread.

The six carried limitations were reviewed explicitly and accepted as non-blocking for D0-only R1 hypothesis generation. They remain binding constraints where relevant in later stages rather than being treated as resolved. R0 is scientifically complete pending only the Git freeze of this R0.9 acceptance state.
