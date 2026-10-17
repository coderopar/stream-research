# R1.5 — Derivatives and positioning hypothesis generation

## Scientific question

What mechanisms involving **open interest, observed forced orders, perpetual funding, mark/index basis, options implied-risk surface, and unsigned gamma geometry** might condition economically viable BTCUSDT positions? This is idea generation only. No new D0 outcomes or market feature computations are performed, and P1 remains sealed.

**Accepted parent:** `r1-perps-trade-flow-v1`; R1.4 first 15 families / 67 ideas / 5 generation sessions are byte-frozen in `research/governance/r1_derivatives_positioning_contract_v1.json`.

## Family discipline and ideas

**No new family is created.** The five existing R1.2 families already span positioning/forced flow, basis, funding crowding, options-surface repricing and unsigned gamma geometry. The windows, thresholds, directions, tenor filters and outcome horizons remain **within** those families; splitting them for more confirmatory shots is prohibited.

| Idea | Mechanism family | Provenance | R2 status |
|---|---|---|---|
| `OI-FLOW-CHURN-DIVERGENCE` | `PERP-POSITIONING-FORCED-FLOW` | OUTCOME_BLIND | Blocked: governed derivative state |
| `OI-PRICE-RESPONSE-DECOUPLING` | `PERP-POSITIONING-FORCED-FLOW` | OUTCOME_BLIND | Blocked: governed derivative state |
| `OI-EXPANSION-PERSISTENCE` | `PERP-POSITIONING-FORCED-FLOW` | OUTCOME_BLIND | Blocked: governed derivative state |
| `OI-CONTRACTION-AFTER-FLOW-SHOCK` | `PERP-POSITIONING-FORCED-FLOW` | OUTCOME_BLIND | Blocked: governed derivative state |
| `OBSERVED-FORCE-FLOW-CLUSTERING` | `PERP-POSITIONING-FORCED-FLOW` | OUTCOME_BLIND | Blocked: governed derivative state |
| `FORCE-OI-CONTRACTION-JOINT-STATE` | `PERP-POSITIONING-FORCED-FLOW` | OUTCOME_BLIND | Blocked: governed derivative state |
| `FORCED-FLOW-DEPTH-FEEDBACK` | `PERP-POSITIONING-FORCED-FLOW` | OUTCOME_BLIND | Blocked: governed derivative state |
| `LEGACY-OI-EXPOSURE-REINTERPRETATION` | `PERP-POSITIONING-FORCED-FLOW` | LEGACY_RESULT_INFORMED | Blocked: governed derivative state |
| `LEGACY-FORCE-SNAPSHOT-NULL` | `PERP-POSITIONING-FORCED-FLOW` | LEGACY_RESULT_INFORMED | Blocked: governed derivative state |
| `FUNDING-FLOW-OPPOSITION` | `FUNDING-CROWDING-REGIME` | OUTCOME_BLIND | Blocked: governed derivative state |
| `FUNDING-SETTLEMENT-CLOCK-CONTEXT` | `FUNDING-CROWDING-REGIME` | OUTCOME_BLIND | Blocked: governed derivative state |
| `LEGACY-COUNTER-FUNDING-UNCERTAIN` | `FUNDING-CROWDING-REGIME` | LEGACY_RESULT_INFORMED | Blocked: governed derivative state |
| `BASIS-PREMIUM-DRIFT-OI` | `PERP-BASIS-DISLOCATION` | OUTCOME_BLIND | Blocked: governed derivative state |
| `BASIS-PREMIUM-TRADE-TENSION` | `PERP-BASIS-DISLOCATION` | OUTCOME_BLIND | Blocked: governed derivative state |
| `LEGACY-BASIS-CHANGE-REPLICATION-FAIL` | `PERP-BASIS-DISLOCATION` | LEGACY_RESULT_INFORMED | Blocked: governed derivative state |
| `OPTIONS-VOL-SURFACE-FUNDING-CONTEXT` | `OPTIONS-SURFACE-RISK-REPRICING` | OUTCOME_BLIND | Blocked: governed derivative state |
| `LEGACY-GAMMA-UNSIGNED-NO-DEALER` | `OPTIONS-UNSIGNED-GAMMA-GEOMETRY` | LEGACY_RESULT_INFORMED | Blocked: governed derivative state |
| `FUNDING-BASIS-DIVERGENCE` | `PERP-BASIS-DISLOCATION` | OUTCOME_BLIND | Blocked: governed derivative state |

## Scientific interpretation and safeguards

1. **Open interest is unsigned outstanding-contract exposure.** Every matched contract has both a long and a short. A rise in OI is **not** direct observation of new longs or proof of bullish positioning; distinguish stock (OI) from flow (volume). OI price/volume interactions are hypotheses, not identities.
2. **Observed force orders are not the full liquidation tape.** The public Binance force-order stream can publish at most the latest liquidation-order snapshot within a 1000-ms interval per symbol. Missing messages and undercounting must not be coded as zero liquidation. Causal coverage, exchange scope and censored-event semantics require independent D0 validation.
3. **Funding has multiple clocks.** The latest published funding indicator is not necessarily the realized future settlement rate. A scheduled settlement time is ex-ante known, but future funding/payment must not be used to form prior eligibility.
4. **Mark and index prices have distinct roles.** Mark/index premium is a market-state variable; it is not a directly executable risk-free convergence trade. Actual marketable entry, funding carry, basis risk, venue constraints, and fees must be modeled later.
5. **Options gamma remains unsigned.** Open-interest-weighted gamma concentration cannot identify a dealer’s net inventory, customer sign, or hedge transactions. The 7M.12A source was feasibility-only with incomplete option OI/Greek overlap.
6. **Data existence is not a causal feature store.** R0.5 supports only governed PERPS_TRADES buckets for the historic D0 candidates. None of these derivatives ideas is R2-testable until each required historical feed is transformed into an exact as-of, receive-time-available, quality-checked and lineage-governed state store. R0.6 executable-book reconstruction is not a source of predictors.
7. **No inherited selection policy.** The legacy 25-bp trigger and 75-bp/15-min label are historical D0 exploration, not imposed on fresh event definitions or trade exits.

## Provenance

Thirteen ideas are independently motivated by public exchange/CME documentation and first-principles market mechanisms (`OUTCOME_BLIND`). Five legacy-result-informed descendants explicitly preserve `LEG7M-07` (OI/force-order null), `LEG7M-08` (funding/basis mixed and failed replication) and `LEG7M-12A` (unsigned gamma feasibility) with parent idea references. The new public source catalogue is `new_public_sources.json`, with stable HTTPS source identities. All prior 67 ideas and 15 families are untouched.

## Guardrails and acceptance

- 18 new ideas, zero new families, generation session v6 (`DERIVATIVES_POSITIONING` complete), total 85 ideas / 15 families. Three lenses remain: `CROSS_VENUE`, `VOLATILITY_REGIME`, `PREDICTIVE_MODEL`.
- All 18 ideas are `BLOCKED_PENDING_GOVERNED_STATE` because historical OI, force-order, funding, mark/index, and options feature families are not governed in R0.5. The testability label expresses infrastructure only, not relative scientific merit.
- No new outcome tests, no D0/P1 market or protected outcomes, no evidence-based ranking/pruning, no permanent `HYP-*` mutations.
- Future stages may append beyond 85 ideas; the R1.5 validator freezes its own **historical prefix** rather than rejecting legitimate R1.6+ expansion.

Run locally after extracting the package at repo root:

```bash
python -m pytest tests/research_governance/test_r1_derivatives_positioning_generation.py -v
python research/r1/derivatives/validate_derivatives_generation.py
python research/r1/derivatives/validate_derivatives_generation.py --self-test
python research/r1/trade_flow/validate_trade_flow_generation.py
python -m pytest tests/research_governance -q -rs
```

**Next:** R1.6 — cross-venue hypothesis generation, without protected-data access.
