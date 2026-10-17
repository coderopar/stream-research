# R1.2 — Legacy Phase 7M lineage reconstruction

**Purpose:** retain the historical *design ancestry*, including apparently positive, null, failed, feasibility-rejected, and outcome-unopened branches, without mistaking old D0 results for independent confirmation. This is the `LEGACY_7M_LINEAGE` lens of the frozen R1.0 universe-generation plan.

**Evidence basis:** archived Phase 7M report `findings.md` files preserved in `full_microstructure_adversarial_review_bundle(1).zip`, and the already accepted R0.2–R0.9 scientific reset. The source bundle is **not** included in the R1.2 ZIP. `legacy_evidence_catalogue.json` supplies the bundle SHA256 plus *exact report-relative paths and report SHA256 hashes*, so historical claims can be independently traced if the archive is available. Do not mistake a file hash for a new scientific replication: this stage catalogs existing reports and does **not** execute them, independently verify their raw underlying events, or reconstruct chronology that their records do not establish.

## What we preserve

- **19 archived source records**: ten 7M outcome-analysis reports, three 7M feasibility reports, the 7M.3 historical measurement substrate, and five upstream 7I/7K outcome-informed policy-selection reports.
- **20 `LEGACY_RESULT_INFORMED` idea records**, including unsuccessful and untested branches. All retain `derived_from_results`, `datasets_that_influenced_design`, `legacy_lineage_refs`, and individual `source_refs` pointing to the catalogue; none is a permanent `HYP-*` registration.
- **5 genuinely separate mechanism families** (positioning/force orders, mark/index basis, funding crowding, options IV surface, unsigned gamma geometry). Other legacy ideas reuse relevant existing R1.1 families rather than splitting families along result or window boundaries.
- **One append-only generation-session record**, `GEN-R1-UNIVERSE-V1@3`, completing only `LEGACY_7M_LINEAGE` and leaving six other lenses uncompleted. R1.1's 8 families, 15 outcome-blind ideas and the first two sessions remain exactly unchanged.

## Scientific interpretation rules

1. **This is already-inspected D0 development evidence.** Result-informed hypotheses are not as strong as independent confirmation, even when older phase reports describe matched D/V replication or passed pre-outcome gates. D/V periods were inspected repeatedly and the program explored many related families.
2. **The old P75/15m event outcome is not a trade.** No legacy candidate event, matching result, or apparent continuation association proves positive return net of spread, fees, latency, funding, slippage, capital constraints or risk. R1.2 freezes **no new trading threshold, entry/exit clock, holding horizon, sizing rule, or hypothesis ranking**.
3. **The causal clock defect matters.** The original 7M.3 future label timing used integer `candidate_second+3s` although the recovered exact book anchor was commonly around +0.95s within that second. Features and labels could overlap; R0.2/R0.3 repaired D0 clocks/outcomes. Historical Phase 7M outcome claims remain historical references, not recalculated confirmations. Merely referencing the old 7M.3 positive report must not silently restore the faulty outcome clock.
4. **7M.1 conditioned on future 35bp success.** It cannot directly identify a prospective early-trade opportunity. 7M.2 built a prospective early-25bp population including failures before 35bp, but its 25bp/reset/release/rearm policy is a *legacy choice*, not an accepted final event universe for R1/R2.
5. **Mechanism identification is narrower than the stories.** Opposing refill does not prove absorption; matched associations do not establish causality. ForceOrder observations are censored, OI does not reveal traders' long/short identity, basis does not reveal flow origin, funding is a slow regime, option IV is not hedge flow, and `OI×|gamma|` measures tracked **unsigned** sensitivity rather than dealer net gamma or signed hedging.
6. **Failure is ancestry, not an invitation to retune.** Replacement penetration had a negative/inconclusive matched estimate; trade-size concentration failed D/V replication; spot aggressive-flow and OKX BBO confirmation were null/inconclusive; some price-confirmation branches failed feasibility and never had valid outcome testing. R1.2 retains them instead of silently dropping them or choosing new cutoffs after the fact.
7. **Timestamp-history limitations persist.** The 7M.11B matching step experienced nanosecond-to-float precision loss before repair; the catalogue records the repaired historical report, not a new validation. Options OI/Greeks overlap covers a limited tracked surface; 7M.12A is feasibility only.
8. **No premature capability promotions.** R0.5 provides governed `PERPS_TRADES` aggregate state only. Legacy raw features, governed books, force-order/OI, basis/funding, cross-venue and options state must be implemented under current causal contracts before R2 testing. R0.6 executable-book replay is not a book-predictor state store.

## Chronology and confidence

Read `lineage_matrix.md` and the hashed evidence catalogue. The matrix keeps 7M.1–7M.12A in order and explicitly distinguishes historical source claims from the modern validity judgment. The supplied reports establish a sequence of archived phases; **they do not establish program-wide independence of threshold selection, model-search decisions, or repeated reuse of the D/V dates**. Where that independence cannot be reconstructed, the register says so instead of inventing preregistration chronology.

## Scope and implementation

This is **source interpretation and data-registration work**, not a computational experiment. Only the archived source reports were read to author the catalogue and new ideas. The installed validator reads contracts, the catalogue, registries, and the P1 seal—not canonical market files, legacy outcome parquet, current D0 outcome parquet, or P1. R1.2 produces no result-based ranking, no new outcome observations, no fresh matched pairs, and no permanent registry mutations. The evidence archive itself remains outside the patch.

## Acceptance gate before commit

```bash
python -m pytest tests/research_governance/test_r1_legacy_lineage.py -v
python research/r1/legacy_7m/validate_legacy_lineage.py
python research/r1/legacy_7m/validate_legacy_lineage.py --self-test
python -m pytest tests/research_governance -q -rs
```

These are required local checks; the ZIP includes the entire proposed commit state (code, records, README, contract and tests) in one package. No separate acceptance-metadata patch is planned.

**Next:** R1.3 — deepen order-book / liquidity hypotheses on a governed causal-state plan without using observed outcomes to prune this R1 universe. Remaining lenses R1.3–R1.8 must still be addressed before R1.9.
