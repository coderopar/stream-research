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

**Status:** PASS / frozen at commit `c02364e86aee2ec8d6d7eb9821f65e2a5f30d0c8`, tag `r0-research-guardrails-v1`.

**Evidence:** R0.7-specific suite: 37/37 passed. Full `tests/research_governance` suite: 146 passed, 3 skipped, 0 failed. The three skips are R0.1 symlink tests unavailable on the current Windows platform. No hypotheses, confirmation batches, protected cohorts, P1 reads, or protected outcome reads were created by R0.7.

**Known limitation:** R0.7 intentionally does not decide P1 cohort partitioning and does not itself prove that future researchers will make good hypotheses. It constrains and records how research is conducted.

**Implication:** R1 can generate ideas broadly in D0 while preserving permanent ancestry and preventing those exploratory choices from masquerading as independent D1 confirmation.

**Next step:** R0.7 was frozen successfully; proceed through R0.8 integration and R0.9 adversarial acceptance.

## R0.8 — Full dry run on D0

**Question:** Do the individually accepted R0 components actually work together as one governed research workflow before we start generating fresh hypotheses?

**What changed:** An end-to-end D0 integration runner was added. It verifies accepted parent hashes, creates one fixed non-promotable causal test state from R0.5 trade buckets, freezes eligibility before attaching future observability, uses the R0.4 conservative dependence unit, requires R0.6 executable-book coverage, applies a one-position-at-a-time 5-minute state machine, rehearses R0.7 registration in an isolated namespace, and red-teams the main forbidden paths.

**Why it matters:** Passing unit tests for clocks, state, execution, dependence, and governance separately is not enough. Research can still fail at the seams—for example if outcome availability silently changes eligibility, if overlapping signals become independent trades, if a dry-run hypothesis mutates production registries, or if protected-data guards are bypassed by the integrated workflow.

**Status:** PASS / frozen at commit `53eac195e83b5fe08397ca50ffc1aadbd201c9a1`, tag `r0-full-d0-dry-run-v1`. The corrected full-D0 implementation ran twice with identical substantive hash `d7024f938942fd35241ce8c37148622032db24ba929a513f754167cf343066fd`; the R0.8-specific suite passed 22/22 tests, and the full research-governance suite passed 168 tests with 3 known Windows symlink-capability skips and zero failures.

**Scientific interpretation:** The R0.8 state is intentionally synthetic plumbing, not a newly proposed market mechanism or trading signal. Its D0 result is permanently non-promotable and the lifecycle terminates `SCIENTIFIC_ONLY` regardless of effect direction or magnitude.

**Acceptance correction:** The first real D0 run exposed a summary-only risk bug: R0.3 stores MAE as a non-negative adverse-excursion magnitude, but R0.8 initially took the minimum when reporting `position_worst_mae_bps`. The underlying 2,836 entered positions had valid MAE variation (median about 9.60 bps; maximum about 209.76 bps). The analysis population, position path, P1 firewall, registry isolation, negative controls, and deterministic substantive hash were unaffected. The builder now uses the maximum non-negative MAE and rejects negative values. The corrected full-D0 rerun completed twice with `position_worst_mae_bps = 209.7608088530789` and identical substantive hashes, so the defect is closed before freeze.

**Known limitation:** R0.8 deliberately reuses the R0.6 zero-latency/1,000-USDT audit execution profile and a mechanical entry-cost proxy. This exercises execution integration but does not create a realistic strategy P&L model. R0.8 also does not decide the real P1/D1 cohort architecture.

**Implication:** The integrated D0 research machine passed its R0.8 gate: causal eligibility remained outcome-blind, P1 was untouched, production registries were unchanged, overlapping exposure was controlled, 7/7 negative controls failed closed, and corrected same-implementation reruns were deterministic. This does not validate any trading hypothesis; it permits the infrastructure to proceed to R0.9 adversarial acceptance review.

**Next step:** R0.8 was frozen successfully; perform R0.9 — Adversarial R0 Acceptance Audit.


## R0.9 — Adversarial R0 acceptance audit

**Question:** After all R0 components have been built and the end-to-end dry run has passed, does the complete scientific infrastructure actually survive a falsification-first review strongly enough to open fresh D0 hypothesis generation in R1?

**What changed:** R0.9 installs a finding-ledger audit that independently rechecks Git freeze lineage, the frozen Protocol-v1.2 hash, canonical provenance identity, the P1 firewall and access ledger, production-registry cleanliness, the accepted R0.8 runtime hash and implementation fingerprint, R0.3–R0.6 parent hashes, critical cross-component governance contracts, and consistency between the living research record and implementation manifests. Known scientific limitations are carried forward explicitly rather than disappearing behind a final PASS label.

**Why it matters:** R0.8 proved the components can work together. R0.9 asks a different question: whether the entire reconstructed research process can withstand an attempt to falsify its readiness. A technically green test suite is not sufficient if freeze provenance is ambiguous, documentation contradicts Git, protected-data status is unclear, parent artifacts drifted, or accepted limitations were silently forgotten.

**Status:** `PASS_WITH_ACCEPTED_LIMITATIONS`. The first audit correctly returned `REVIEW_REQUIRED` because it found three MAJOR documentation/acceptance-metadata inconsistencies (`R09-DOC-001` through `R09-DOC-003`). Those findings were remediated without changing scientific artifacts. Two consecutive post-remediation audits then produced the same substantive hash `3faab2f04bb4902d19d90a0a9c6df49b42e62b648572c6316110ea869b87accd` and a final ledger of 17 PASS, 6 accepted limitations, 0 open findings, 0 majors, 0 blockers, 0 `NOT_TESTABLE`, and 0 R1-blocking rows.

**Protection:** R0.9 performed no new hypothesis testing and read neither P1 market partitions nor P1 outcomes. It audited governance metadata and accepted D0 artifacts only.

**Acceptance evidence:** R0.9-specific tests passed 26/26. The full `tests/research_governance` suite passed 194 tests with 3 known Windows symlink-capability skips and zero failures. The six carried limitations were reviewed explicitly and accepted as non-blocking for D0-only R1 hypothesis generation; they remain binding constraints for later stages where relevant.

**Implication:** The R0 research foundation has survived its final falsification-first gate. This does not validate a trading hypothesis; it establishes that the governed research machine is fit to begin fresh D0 hypothesis generation.

**Next step:** Freeze this accepted R0.9 state in Git, then open R1 — Fresh Hypothesis Universe Generation.

---

## R0 status

R0 is complete with accepted limitations and frozen at commit `71192f37e8aa44da4c30326048debf65b1ca5792`, tag `r0-foundation-accepted-v1`. No scientific R1 blocker remains.

# R1 — Fresh hypothesis-universe generation

R1 uses the accepted R0 machinery to generate a broad, auditable universe of BTCUSDT research ideas before exploratory outcome testing begins in R2. The repaired 7M path is one lineage branch, not the default or mandatory research direction.

## R1.0 — Hypothesis-generation governance + universe schema

**Question:** Before generating fresh hypotheses, what record structure and stopping rules prevent brainstorming, legacy-result influence, family splitting, implementation convenience, or hidden outcome use from becoming undocumented researcher degrees of freedom?

**What changed:** R1.0 adds append-only pre-hypothesis registries for generation sessions, conceptual families, and idea packets; explicit `OUTCOME_BLIND`, `D0_RESULT_INFORMED`, and `LEGACY_RESULT_INFORMED` provenance; anti-family-splitting rules; causal-state / trigger / position / risk / execution concepts; infrastructure-testability status separate from scientific merit; and an explicit adapter into the permanent R0.7 `HYP-*` registry. A frozen generation plan predeclares nine lenses that must be addressed before R1.9 can freeze the universe.

**Why it matters:** R1 should be broad enough to generate unsupported or unconventional ideas without allowing favorable D0 results, existing code, or threshold/horizon variants to masquerade as independent mechanisms. A raw idea must not receive the status of a registered hypothesis merely because it was written down.

**Protection:** R1.0 reads no D0 outcomes, P1 market data, P1 outcomes, or protected performance. It creates no BTCUSDT hypothesis families or market ideas and does not mutate the permanent R0.7 hypothesis registry. P1 remains sealed.

**Known limitation:** R1.0 governs how ideas are recorded; it does not guarantee that later R1 ideas are insightful. Conceptual family boundaries still require scientific judgment and are therefore reviewed again at R1.9.

**Implication:** R1.1-R1.8 can generate ideas broadly while preserving whether an idea was genuinely outcome-blind or inherited from D0/legacy results. Missing state or execution capability may block later R2 testing but may not be used as a scientific-merit score.

**Status:** PASS / frozen at commit `93f954f303f2585dd7de171292dc12075b97cdde`, tag `r1-hypothesis-universe-governance-v1`. R1.0-specific validation passed 29/29 tests. The accepted project-environment research-governance suite passed 223 tests with the same 3 known Windows symlink-capability skips and zero failures. The validator returned `PASS` with 0 family rows, 0 idea rows, 1 frozen generation session, no D0 outcome read, no P1 market-data read, no P1 outcome read, and `P1_PROTECTED` still `SEALED_OUTCOME_UNSEEN`.

**Next step:** R1.0 is frozen. Proceed to R1.1 theory / market-mechanism family generation.


## R1.1 — Theory / market-mechanism family generation

**Question:** Before opening D0 exploratory testing, what economically plausible BTCUSDT mechanisms should enter the governed universe purely from public market-microstructure theory/literature and first-principles reasoning?

**What changed:** R1.1 appends eight conceptual families and fifteen immutable idea records to the R1 universe. The families cover information-bearing flow/price discovery, metaorder flow persistence, liquidity resilience, impact efficiency versus absorption, queue pressure, transient impact decay, cross-venue information diffusion, and volatility-liquidity feedback. A frozen ten-source public literature catalogue records the qualitative provenance used to motivate those mechanisms. Generation session `GEN-R1-UNIVERSE-V1` advances to version 2 / `IN_PROGRESS`, with only `THEORY_LITERATURE` and `MARKET_MECHANISM` marked complete.

**Why it matters:** This is the first point where the reset research program actually creates new market ideas. By doing it before fresh D0 outcome testing, the universe is not constructed by looking for whatever happened to work historically. Family boundaries also prevent threshold, horizon, window, queue-level, or venue-pair variants from masquerading as independent mechanisms.

**Source discipline:** R1.1 uses `PUBLIC` literature and `NONE` first-principles mechanism sources only. All fifteen ideas are `OUTCOME_BLIND`: no D0 outcome, legacy 7M result, P1 market data, P1 outcome, fitted project statistic, or observed strategy return influences admission. Public empirical papers are background literature, not project outcomes, and their fitted coefficients/statistics are not copied into idea packets.

**Infrastructure boundary:** Scientific admissibility is separate from current testability. Two ideas can currently be developed using the governed `PERPS_TRADES` state; thirteen require one or more deferred governed state capabilities such as `ORDER_BOOK`, cross-venue state, volatility state, or causal intrawindow price-path state. Those ideas remain in the universe, but R2 may not reconstruct those states ad hoc. R0.6 order-book replay remains an execution foundation, not an approved order-book predictor store.

**Known limitation:** R1.1 is intentionally broad and qualitative. It does not claim these mechanisms transfer to BTCUSDT, does not rank their merit, and does not choose thresholds, horizons, or interactions. Detailed generation for order-book, trade-flow, cross-venue, and volatility lenses remains reserved for later R1 stages.

**Implication:** The research program now has a first outcome-blind mechanism layer. The next stage can add the old 7M lineage transparently as result-informed ancestry rather than allowing it to silently shape the fresh theory universe.

**Acceptance boundary:** This README and all R1.1 artifacts ship in one commit candidate. The step is committed only after the R1.1 validator/tests and full `tests/research_governance` regression pass in the project environment; no separate post-test metadata patch is required.

**Next step:** R1.2 — Legacy 7M lineage generation under explicit `LEGACY_RESULT_INFORMED` provenance.

## After R0

R1 begins fresh hypothesis-universe generation. The repaired 7M direction remains one historical D0 lineage branch, not the mandatory path for new research. Fresh hypothesis generation may introduce new candidate definitions, causal market states, interaction structures, data families, predictive models, and economic mechanisms, provided they remain inside the governed D0 research process until frozen for protected confirmation. A hypothesis that depends on a market-state family not yet implemented by the governed state store may be generated in R1, but it cannot be tested ad hoc in R2; the corresponding causal state capability must first be added under governance.


R0.8 acceptance also hardened idempotence after the MAE summary correction: substantive equality is now required only for the same accepted parents and the same implementation fingerprint; a legitimate code/contract correction creates a new baseline while true same-implementation nondeterminism still fails closed.

**R0.9 rerun hardening:** The first post-remediation rerun exposed an audit-identity defect rather than scientific nondeterminism. The pre-remediation audit hash was being compared with a repository whose audited governance/documentation state had intentionally changed. R0.9 now fingerprints the audited governance inputs, production-registry state, accepted R0.8 runtime manifest, Git HEAD, and frozen tag resolutions. Idempotence is enforced only when accepted parent, audit implementation, and audited repository state are all unchanged. This change does not read P1 market data or outcomes and does not alter any R0 scientific artifact.

## R1.2 — Recovering the legacy 7M research ancestry

**Question:** Can we retain what the previous, heavily inspected 7M program taught us, including what failed, without pretending that historical results are independent validation or automatically carrying its candidate policy forward?

**What changed:** A source-hashed, nineteen-record historical evidence catalogue (including five pre-7M policy-selection precursors) and human-readable 7M.1–7M.12A chronology are paired with twenty explicit `LEGACY_RESULT_INFORMED` idea records. Five distinct mechanism families were added; historical liquidity, flow and cross-venue ideas reuse existing R1.1 families instead of splitting families to increase statistical testing opportunities. The frozen generation plan advances to session v3, completing `LEGACY_7M_LINEAGE` only.

**Why it matters:** The old work explored many related outcomes, horizons and features with repeatedly inspected dates, and the pre-R0 candidate-second horizon was later found to overlap information from the exact book anchor. It is scientifically useful development lineage, not an independent confirmation dataset. Null results, failed replication and feasibility-only investigations remain visible, rather than silently removed.

**Evidence/acceptance:** Original archived report paths and SHA256 fingerprints, lineage validation, and full governance regression are shipped in the same proposed commit package. Local test/validator acceptance is required before committing; no separate metadata patch is needed.

**Limitations:** The archive is a historical *report* record, not a fresh audit of underlying raw events or a way to prove all historical analysis choices were preregistered. No economic profitability, causal effect or signed dealer hedge inference is established. The original 25bp/reset/rearm candidate definition remains legacy and optional for future D0 designs. P1 is not accessed.

**Implication:** R1.3–R1.8 can develop mechanism-specific variants with accurate genealogy but must not mine old outcome tables or treat a favorable old effect as a new confirmation. Existing unsupported causal state families still need governed implementations before R2.

**Next:** R1.3 — deeper order-book/liquidity hypothesis generation. R1.9 must still audit the complete universe and family boundaries.

## R1.3 — Order-book and liquidity generation (commit candidate)

R1.3 deepens the first-principles and historical liquidity mechanisms without running a new market/outcome study. It adds one spatial depth-geometry family and sixteen idea packets (nine `OUTCOME_BLIND`, seven `LEGACY_RESULT_INFORMED`) to the existing governed universe, yielding **14 families and 51 ideas**. The first 13 families, 35 ideas and three session versions are preserved byte-for-byte. The new `GEN-R1-UNIVERSE-V1@4` completes the `ORDER_BOOK_LIQUIDITY` lens; five lenses remain.

Why it matters: we now have auditable hypotheses about depth location, queue pressure, replenishment, cancellation/execution ambiguity, transient impact, and flow-conditioned liquidity, without treating the earlier 7M correlations or old 25bp selection policy as proven trading rules. Null/inconclusive legacy refill-penetration evidence remains explicit. All 16 book ideas remain **blocked** for R2 until a proper receive-time-governed causal `ORDER_BOOK` predictor state exists: R0.6 execution replay is not a book feature store. Public market-by-price updates do not reveal passive queue position, individual order identity or hidden liquidity; no claims of spoofing or realized fills can be inferred from them.

No fresh D0 outcomes were opened, no P1 data or outcomes were read, and no R0.7 permanent hypothesis was registered. The package includes its initial research narrative, contract, tests and implementation manifest from the outset; local acceptance decides whether to freeze it, without a post-testing README patch. See `research/r1/order_book/README.md` for the full theory/legacy distinction and future capability gates. **Next: R1.4 — perpetual trade-flow generation.**

## R1.4 — Perpetual trade-flow hypotheses

**Question:** What pre-decision, causally observable patterns of trade-side activity, persistence, and arrival tempo might matter for executable BTCUSDT price response—without mistaking activity for information or selecting by returns?

**What changed:** Reused the existing informed-flow, metaorder-persistence and impact-efficiency families; added one independent trade-arrival-tempo family and sixteen ideas (fifteen outcome-blind, one legacy-informed with 7M.7C failed replication). The append-only universe now has **15 families, 67 ideas, and generation-session version 5**. Four required lenses remain. Three public literature sources are explicitly catalogued.

**Why it matters:** R0.5 makes causal trade-count, quantity, notional and raw maker-flag category buckets available now, but not per-print size/interarrival streams or an entire historical price path. Twelve R1.4 ideas are *structurally* testable with existing candidate-indexed D0 buckets, while four are capability-blocked. Any new candidate definition needs a governed rebuild of its causal state, not a free pass from the original frozen-candidate store. This does **not** mean any is profitable, that aggressor sign has been verified, or that latency/fees have been calibrated.

**Evidence/acceptance:** The R1.4 validator hashes frozen R1.3 prefix bytes, validates each idea's immutable genealogy, literature refs, infrastructure gating, sealed P1 boundary, and no permanent HYP registry mutation. Tests, validator, manuscript catalogue and manifest ship in the **same initial commit-candidate ZIP**; local acceptance is required before Git freeze.

**Known limitations:** Trade rows are not unique orders or traders; `buyer_is_maker` is raw until its semantics are verified; storage grids are not scientific feature windows; unobserved print distributions/interarrival/price histories are not fabricated; historical trade concentration failed replication and remains unconfirmed. No new D0 outcomes or P1 reads occurred.

**Implication:** Later R2 may use existing bucket families after D0 feature/candidate design and proper evidence/return/risk gates. It may not use blocked capabilities until governed state exists. No old 25-bp candidate policy is compulsory. See `research/r1/trade_flow/README.md`.

**Next:** R1.5 — derivatives and positioning lens. R1.4 commit/tag are established only after local verification.

## R1.5 — Derivatives and positioning hypothesis generation

**Question:** Which causally observed changes in outstanding perp
contracts, forced-order snapshots, funding regimes, mark/index premium,
and option implied-risk geometry might eventually support executable
BTCUSDT positions?

**What changed:** On the immutable R1.4 15-family/67-idea prefix, R1.5
appends 18 governed idea records and generation-session v6; five
`LEGACY_RESULT_INFORMED` ideas retain 7M.7 / 7M.8 / 7M.12A source
ancestry and failed/null/feasibility qualifiers. Thirteen ideas have
independent theoretical/mechanism provenance. No new family was justified.

**Why it matters:** Derivatives observations have distinct semantics. OI
is not directional long/short exposure; force-order snapshots are not the
full liquidation tape; funding has causal publication and settlement
clocks; mark/index basis is not risk-free tradable P&L; and unsigned
OI-weighted gamma cannot identify dealer hedging direction. Confusing
these would create spurious signals.

**Status:** Complete R1.5 commit candidate, local acceptance pending.

**Evidence and limits:** 15 families / 85 ideas / six generation-session
records; every new idea is explicitly capability-blocked until causal
derivatives states exist. R0.5 still provides only historical-D0-candidate
perp trade buckets; the R0.6 audit execution profile is not a strategy.
No new D0 or P1 outcomes were opened or tested and the permanent `HYP-*`
registry remains untouched. See `research/r1/derivatives/README.md`.

**Next:** R1.6 cross-venue hypothesis generation.
