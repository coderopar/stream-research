# BTCUSDT Scientific Research Program

This file is the living scientific map of the repository. It explains, in plain language, what each research step is trying to accomplish, what was learned, what remains unresolved, and why the next step is justified.

The governing research constitution is Protocol v1.2 in `research/protocol/v1.2/`.

## Research-step documentation rule

Every commit that advances, changes, repairs, invalidates, freezes, or materially reinterprets a scientific research step **MUST update this README in the same commit**.

A research-step update must record, at minimum:

- **Step and question** — what scientific or research-infrastructure question was being addressed.
- **What changed** — the machinery, data contract, analysis, or decision introduced by the step.
- **Why it matters** — the failure mode or uncertainty the step removes or exposes.
- **Evidence / acceptance status** — tests, build evidence, coverage, or review outcome supporting the status.
- **Known limitations** — what the step does not establish and what remains deferred.
- **Implication for later research** — what downstream work may now rely on, and what it still may not assume.
- **Next step** — the next governed research action.
- **Commit / tag** — once frozen, record the accepted code checkpoint and tag where applicable.

Phase-specific READMEs remain the detailed implementation record. This file is the cross-phase scientific narrative and must stay understandable without reading source code.

Pure refactors, formatting-only changes, dependency maintenance, or other changes that do not alter the scientific process or interpretation do not require a new research-step entry. If a bug fix can change candidate membership, clocks, features, execution, outcomes, inference, eligibility, lineage, or scientific interpretation, it is a research change and must be documented here.

## Program sequence

The current governed sequence is:

```text
R0  Scientific infrastructure reset
  -> R1  Fresh hypothesis-universe generation
  -> R2  D0 exploratory testing
  -> R3  Economic translation
  -> R4  Adversarial reduction and pre-confirmation freeze
  -> D1  One-use protected historical confirmation
  -> D2  Prospective shadow trading
  -> D3  Controlled small live capital
```

The protected extension remains sealed while R0-R4 use D0 for development.

---

# R0 — Scientific infrastructure reset

R0 is not intended to prove a trading strategy. Its purpose is to make the research machine trustworthy before fresh hypothesis generation and before any protected confirmation data is spent.

## R0.0 — Governance freeze

**Question:** What rules govern the reset before any further research can influence protected data?

**What changed:** The old May-July research period was declared D0 development data. The newly added post-2026-07-18 extension was declared `P1_PROTECTED` with outcomes unseen. Protocol v1.2, protected-data registers, outcome-access registers, and lineage-reset artifacts were frozen.

**Why it matters:** A holdout is only valuable if its independence is preserved before outcomes or design-relevant protected information influence research choices.

**Status:** PASS / frozen.

**Known limitation:** This step does not decide whether P1 will later be one D1 cohort or be pre-partitioned into D1 plus reserve. That decision must be justified from D0 support/precision needs before protected unblinding.

**Implication:** Later R0 work must remain outcome-blind with respect to P1.

**Checkpoint:** commit `42c7c89`; tag `r0-governance-provenance-v1` also freezes R0.1.

## R0.1 — Dataset inventory and provenance

**Question:** Do we know exactly what canonical data exists, where it lives, which dates belong to D0 versus P1, and how to reproduce that boundary?

**What changed:** Canonical files, schemas, feed coverage, date boundaries, hashes, logical/physical storage roots, and the external canonical-store topology were inventoried and frozen.

**Why it matters:** Research results are not reproducible if different runs can silently use different data versions or if derived/outcome folders can be mistaken for canonical inputs.

**Status:** PASS / frozen.

**Evidence:** 22,369 canonical files indexed; D0 and P1 boundaries identified without unknown dated market-event files; full file-index hash frozen.

**Known limitation:** Three symlink-specific tests are skipped on the current Windows platform because the environment does not provide directory-symlink capability. This is an environment limitation, not a failed provenance invariant; logical-vs-resolved-path topology is still governed.

**Implication:** All later stages can refer to a reproducible canonical dataset boundary and provenance chain.

**Checkpoint:** commit `42c7c89`; tag `r0-governance-provenance-v1`.

## R0.2 — Causal clock repair

**Question:** Can every research input, decision, execution event, and outcome be placed on a causally valid timeline without integer-second shortcuts or future overlap?

**What changed:** Integer-second timing assumptions were replaced by explicit int64-nanosecond clocks for candidate availability, information availability, decision, order send, exchange arrival, executable entry, and outcome start. The historical candidate-second overlap defect was turned into a regression test.

**Why it matters:** The old research path could overlap feature information and future outcome measurement by roughly 0.95 seconds. In microstructure research that can manufacture apparent predictability.

**Status:** PASS / frozen.

**Known limitation:** Correct timing cannot reconstruct information that the historical pipeline never preserved; missing upstream lineage is surfaced rather than guessed.

**Implication:** Later research must derive eligibility and features only from information available by the governed decision clock, and economic outcomes cannot begin before executable entry.

**Checkpoint:** included in commit `e9b3ac699213503348a0549b24e3a9bff02cfa88`; tag `r0-causal-clock-substrate-v1`.

## R0.3 — Corrected D0 research substrate

**Question:** Can the heavily explored 7M candidate lineage be rebuilt under the corrected clock contract without changing which historical events were candidates?

**What changed:** All 7,067 frozen 7M candidates were preserved. Exact governed clocks were recovered for 7,031 candidates; 36 missing historical level-audit cases remain explicitly `CLOCK_UNAVAILABLE`. Corrected 5m/10m/15m/30m outcomes were rebuilt without opening P1 or blindly reusing defective legacy outcome timing.

**Why it matters:** This lets us study the old research direction honestly after repairing its defects without pretending the inherited 25bp candidate policy has been independently validated or rediscovered.

**Status:** PASS WITH EXPLICIT COVERAGE LIMITATION / frozen.

**Evidence:** 7,067/7,067 candidate identities preserved; 28,268/28,268 expected horizon rows produced; 28,124 available and 144 clock-unavailable outcome rows.

**Known limitation:** The 36 candidates without recoverable governed clocks cannot support corrected clock-dependent analyses. The inherited candidate-generation policy remains historical D0 lineage, not an accepted final trading-event definition.

**Implication:** Old 7M hypotheses may be re-examined as development evidence, but R1 remains free to generate completely fresh event/candidate definitions.

**Checkpoint:** commit `e9b3ac699213503348a0549b24e3a9bff02cfa88`; tag `r0-causal-clock-substrate-v1`.

## R0.4 — Dependence / episode framework

**Question:** Are the candidate rows independent pieces of evidence, and what dependence structure should later inference respect?

**What changed:** Outcome-window overlap components were frozen at 5m/10m/15m/30m together with 60-second policy episodes, directional episodes, exact-origin lineage, and a conservative day/30m union structure.

**Why it matters:** Nominal candidate count is not independent sample size. At longer horizons many candidates share the same future market path; treating them as independent would overstate precision and significance.

**Status:** PASS WITH EXPLICIT DEPENDENCE LIMITATION / frozen.

**Evidence:** Multi-member overlap participation rises from about 54.2% at 5m to about 96.3% at 30m; 15m has 7,031 governed candidates but only 1,675 overlap components.

**Known limitation:** No single clustering definition magically reveals a true effective sample size. Later inference must pre-specify conservative dependence treatment and concentration sensitivity.

**Implication:** Random row-level train/test splits and naive independent-candidate confidence intervals are not valid defaults for this research program.

**Checkpoint:** commit `9bc5fd1`; tag `r0-dependence-framework-v1`.

## R0.5 — Causal market-state store

**Question:** Can later hypotheses consume a common, reproducible market-state history instead of each reconstructing causal inputs differently?

**What changed:** A reusable causal state store was frozen at each candidate's governed `information_available_ts_ns`. Version 1 implements canonical BTCUSDT perpetual-trade state and 1-second/1-minute storage buckets with freshness, coverage, source provenance, deterministic tie handling, and explicit missing-partition behavior.

**Why it matters:** If each hypothesis independently rebuilds history, tiny differences in as-of joins, cutoffs, missingness, or timestamp handling can become hidden researcher degrees of freedom or leakage paths.

**Status:** PASS / frozen.

**Evidence:** 7,031/7,031 governed candidates have available trade state, 60-second history, and 30-minute history; no latest-trade timestamp ambiguity was observed. The governed suite passed before freeze.

**Known limitation:** Version 1 intentionally implements `PERPS_TRADES` only. `ORDER_BOOK`, `PERPS_DERIVATIVES`, `SPOT_CROSS_VENUE`, and `OPTIONS` state families remain deferred until each has its own causal availability contract. A cold rebuild is operationally expensive.

**Implication:** Future hypotheses can safely reuse the implemented trade-state substrate. Hypotheses requiring deferred data families must first extend the governed state store rather than bypass it.

**Checkpoint:** commit `62c99c09f5caef61cdfa1d9f1a70fc0236b28579`; tag `r0-causal-market-state-v1`.

## R0.6 — Execution + cost simulator

**Question:** If a signal is causally available, can we reconstruct a price and fill that could actually have been obtained rather than evaluating against a mid-price fantasy?

**What changed:** Canonical BTCUSDT perpetual order books are reconstructed from checkpoints plus diff-depth updates with sequence validation, gap invalidation, crossed-book protection, and recovery handshakes. Marketable orders walk visible depth. Latency and costs are explicit model inputs rather than hidden constants.

**Why it matters:** Predictive edge is not trading edge unless it survives the delay, bid/ask crossing, depth, fees, funding, and other frictions that occur after the information becomes available.

**Status:** PASS / frozen as an execution foundation.

**Evidence:** 7,031/7,031 governed candidate clocks had valid reconstructed executable books in the D0 audit; technical BUY and SELL depth walks completed for all 7,031. Nineteen sequence gaps were detected and invalidated/recovered without leaving any candidate query in an invalid state.

**Known limitation:** The production coverage audit uses a zero-added-latency and 1,000-USDT **audit-only** profile. It does not claim those are realistic strategy parameters. Passive-limit fills, own-market impact, final fee schedules, final standardized research notional, and live latency calibration remain to be registered later where applicable.

**Implication:** We now have a trustworthy execution engine, but each trading hypothesis must still freeze a defensible realistic execution/cost profile before protected confirmation.

**Checkpoint:** commit `6e742f8`; tag `r0-execution-simulator-v1`.

## R0.7 — Research registries + guardrails

**Question:** Now that the machinery is trustworthy, how do we prevent future research from quietly changing hypotheses, reusing protected information, hiding failures, splitting families, or overfitting through process freedom?

**What changed:** Append-only registries and executable guardrails were added for hypothesis versions and genealogy, lifecycle transitions, evidence budgets, protected cohorts, confirmation batches, access/contamination events, decision logs, predictor/eligibility restrictions, and time-respecting validation rules.

**Why it matters:** Scientific discipline must be encoded in the workflow rather than depend only on memory. A failed hypothesis must remain failed in that version; a material redesign must create a descendant version; D1 access and contamination must be auditable.

**Status:** IMPLEMENTED; local acceptance tests PASS; Git freeze pending.

**Evidence:** R0.7-specific suite: 37/37 passed. Full `tests/research_governance` suite: 146 passed, 3 skipped, 0 failed. The three skips are R0.1 symlink tests unavailable on the current Windows platform. No hypotheses, confirmation batches, protected cohorts, P1 reads, or protected outcome reads were created by R0.7.

**Known limitation:** R0.7 intentionally does not decide P1 cohort partitioning and does not itself prove that future researchers will make good hypotheses. It constrains and records how research is conducted.

**Implication:** R1 can generate ideas broadly in D0 while preserving permanent ancestry and preventing those exploratory choices from masquerading as independent D1 confirmation.

**Next step:** Freeze R0.7 in Git, then run R0.8 as an end-to-end dry run of the research machine on D0.

## R0.8 — Full dry run on D0

**Question:** Do the individually accepted R0 components actually work together as one governed research workflow before we start generating fresh hypotheses?

**What changed:** An end-to-end D0 integration runner was added. It verifies accepted parent hashes, creates one fixed non-promotable causal test state from R0.5 trade buckets, freezes eligibility before attaching future observability, uses the R0.4 conservative dependence unit, requires R0.6 executable-book coverage, applies a one-position-at-a-time 5-minute state machine, rehearses R0.7 registration in an isolated namespace, and red-teams the main forbidden paths.

**Why it matters:** Passing unit tests for clocks, state, execution, dependence, and governance separately is not enough. Research can still fail at the seams—for example if outcome availability silently changes eligibility, if overlapping signals become independent trades, if a dry-run hypothesis mutates production registries, or if protected-data guards are bypassed by the integrated workflow.

**Status:** PASS — local acceptance complete; Git freeze pending. The corrected full-D0 implementation ran twice with identical substantive hash `d7024f938942fd35241ce8c37148622032db24ba929a513f754167cf343066fd`; the R0.8-specific suite passed 22/22 tests, and the full research-governance suite passed 168 tests with 3 known Windows symlink-capability skips and zero failures.

**Scientific interpretation:** The R0.8 state is intentionally synthetic plumbing, not a newly proposed market mechanism or trading signal. Its D0 result is permanently non-promotable and the lifecycle terminates `SCIENTIFIC_ONLY` regardless of effect direction or magnitude.

**Acceptance correction:** The first real D0 run exposed a summary-only risk bug: R0.3 stores MAE as a non-negative adverse-excursion magnitude, but R0.8 initially took the minimum when reporting `position_worst_mae_bps`. The underlying 2,836 entered positions had valid MAE variation (median about 9.60 bps; maximum about 209.76 bps). The analysis population, position path, P1 firewall, registry isolation, negative controls, and deterministic substantive hash were unaffected. The builder now uses the maximum non-negative MAE and rejects negative values. The corrected full-D0 rerun completed twice with `position_worst_mae_bps = 209.7608088530789` and identical substantive hashes, so the defect is closed before freeze.

**Known limitation:** R0.8 deliberately reuses the R0.6 zero-latency/1,000-USDT audit execution profile and a mechanical entry-cost proxy. This exercises execution integration but does not create a realistic strategy P&L model. R0.8 also does not decide the real P1/D1 cohort architecture.

**Implication:** The integrated D0 research machine passed its R0.8 gate: causal eligibility remained outcome-blind, P1 was untouched, production registries were unchanged, overlapping exposure was controlled, 7/7 negative controls failed closed, and corrected same-implementation reruns were deterministic. This does not validate any trading hypothesis; it permits the infrastructure to proceed to R0.9 adversarial acceptance review.

**Next step:** Freeze the accepted R0.8 step in Git, then perform R0.9 — Adversarial R0 Acceptance Audit.


---

## R0 remaining gates

```text
R0.8  Full dry run on D0
R0.9  Adversarial R0 acceptance audit
```

R0 is complete only after these integration/adversarial gates pass. Passing isolated component tests is necessary but not sufficient.

## After R0

R1 begins fresh hypothesis-universe generation. The repaired 7M direction remains one historical D0 lineage branch, not the mandatory path for new research. Fresh hypothesis generation may introduce new candidate definitions, causal market states, interaction structures, data families, predictive models, and economic mechanisms, provided they remain inside the governed D0 research process until frozen for protected confirmation.


R0.8 acceptance also hardened idempotence after the MAE summary correction: substantive equality is now required only for the same accepted parents and the same implementation fingerprint; a legitimate code/contract correction creates a new baseline while true same-implementation nondeterminism still fails closed.
