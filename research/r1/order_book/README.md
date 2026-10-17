# R1.3 — Order-book and liquidity mechanism generation

**Purpose.** Expand the mechanism universe with causally plausible book/liquidity hypotheses, not a backtest and not a proposal to inherit legacy Phase 7M's event selection. This step completes only the frozen `ORDER_BOOK_LIQUIDITY` generation lens under Protocol v1.2.

## What we built, and why

R1.3 appended one genuinely distinct family, **multi-level depth geometry and fragility**, plus sixteen versioned ideas. It reuses the earlier R1.1 price-discovery, queue-pressure, resilience, transient-impact and absorption families for every other mechanism. Different thresholds, ladder bands, event triggers, directions, holding periods, book sides and execution assumptions are **within-family variants**, never independently budgeted families simply because a variant eventually looks favorable.

The 16 ideas are qualitative, mutually non-exclusive starting mechanisms. Their exact candidate definitions, lookbacks, cutoffs, side, latency, exit and costs have not been frozen; no measured BTCUSDT return or direction-hit claim was generated. An observable book *configuration* is not the same thing as an actionable position. In R2/R3 the primary objective must remain **positive net executable return with risk gates and a meaningful benchmark**, not impressive conditional event frequencies.

| Family / mechanism | Added ideas | Provenance |
|---|---|---|
| Multi-level depth geometry | Depth gaps/fragility; depth concentration; depth migrating away from the touch | Fresh public theory + first-principles |
| Queue pressure | Persistent queue asymmetry; imbalance pressure failure | Fresh public theory + first-principles |
| Information / book event flow | Depth-conditioned order-book-event OFI | Fresh public theory + first-principles |
| Resilience and transient impact | Quote churn; spread/depth recovery after shock | Fresh public theory + first-principles |
| Opposing liquidity attrition | Opposing attrition; durable queue loss | **Legacy-result-informed**, 7M lineage |
| Replenishment dynamics | Opposing refill breadth; near-restoration failure; supporting-vs-opposing refill race | **Legacy-result-informed**, 7M lineage |
| Refill execution / book drain | Replacement execution bounds; cancellation-vs-execution drain | **Legacy-result-informed**, 7M lineage |
| Joint book-flow response | Book recovery with flow decay | Fresh public theory + first-principles |

The result is **9 `OUTCOME_BLIND`** plus **7 `LEGACY_RESULT_INFORMED`** ideas. A fresh idea means its stated design uses the cited public model or first-principles mechanism and outcome-blind parent only; it does **not** mean it was empirically validated. A legacy-informed idea exposes `legacy_lineage_refs`, `derived_from_results`, the pre-reset D0 dataset, and its historical parent ideas, including null/negative ones. The negative/inconclusive legacy replacement-penetration branch is preserved as an explicit competitor, **not** inverted or threshold-rescued. The pre-7M 25bp/60-second development policy and the 75bp/15m research outcome are historical ancestors, **not** required candidate rules, endpoints or holding periods.

## Sources: what they do and do not support

We reuse literature already catalogued in R1.1: Cont–Kukanov–Stoikov (order-book-event OFI and depth), Gould–Bonart (queue imbalance), Obizhaeva–Wang (book resilience), and Gould et al. (book structure). R1.3 adds one **public** source, Huang–Lehalle–Rosenbaum (2015), *Simulating and Analyzing Order Book Data: The Queue-Reactive Model*, https://arxiv.org/abs/1312.0563. Its queue-state-dependent event intensities are a conceptual starting point, not a BTCUSDT trading result. Exact added-source metadata live in `new_public_sources.json`; older source identities remain in `research/r1/theory_mechanisms/literature_sources.json`.

Specific conjectures such as a depth-gap continuation effect, queue-pressure-failure reversal, or a support/opposition refill race are **our candidate mechanisms**; the cited sources do not establish those directions or net profits on Binance BTCUSDT. Historical 7M observations are acknowledged separately and retain their exploratory, dependent, sometimes clock-overlapped evidentiary status. See `research/r1/legacy_7m/lineage_matrix.md` for their limitations.

## Hard causal and market-data constraints

All sixteen ideas remain `BLOCKED_PENDING_GOVERNED_STATE` for R2. **R0.5 v1 provides causal `PERPS_TRADES` only.** The R0.6 order-book replay validates *marketable executable-book coverage and depth-walking at arrival*, but it is **not a predictive feature store**. A new governed `ORDER_BOOK` as-of state must be built before book indicators can enter R2, with explicit input provenance, int64 nanosecond receive-time clocks, Binance snapshot + `U/u/pu` bridging, gap/cross-book invalidation, no future interpolation, freshness, missingness and deterministic tie ordering. No accidental use of execution-arrival book as pre-decision evidence. No time windows may reach into protected P1. A causal book-state failure makes the feature unavailable rather than backward-filling from future states.

The order-book feeds here are market-by-price (aggregated depth). They cannot reconstruct **order identity**, hidden/iceberg liquidity, trader intentions, spoofing, or an individual passive order's **queue position** or fill probability. Snapshot/update quantity loss does not uniquely tell cancellation, relocation and execution apart. Joint book/trade hypotheses require a separate causally aligned attribution layer with explicit unknown/ambiguous cases, **not** fabricated causal certainty. Staleness, replay gaps, receive-time ordering and exchange-to-receive latency can all matter materially. R0.6's zero-added-latency / 1,000-USDT audit profile is **not** a realistic strategy latency/fee/size profile. Passive fills are not simulated from it, and depth beyond visible levels is not extrapolated.

`GOVERNED_BOOK_EVENT_OFI`, `GOVERNED_BOOK_TRADE_EVENT_ATTRIBUTION`, and `GOVERNED_DEPTH_LEVEL_EVENT_HISTORY` identify more specific deferred capabilities, in addition to baseline `GOVERNED_ORDER_BOOK_STATE`.

**Eligibility** must be frozen using information actually observable no later than the decision clock. A later move, refill, reversal, resting-order execution, or complete future path may **not** retroactively select the tradeable cohort. R0.2 exact clocks, R0.4 dependence, and R0.7 registration/evidence budgets remain mandatory. Strong inference later must account for overlapping outcome windows rather than counting every candidate independently.

## Appended and protected records

- Frozen R1.2 prefix: 13 families / 35 ideas / 3 generation sessions. Original bytes and per-row `freeze_hash` values are retained.
- New R1.3 suffix: 1 family / 16 ideas / generation-session v4, which completes `ORDER_BOOK_LIQUIDITY` and leaves the next five lenses pending.
- Resulting universe: **14 families / 51 ideas / 4 sessions**, with **zero permanent `HYP-*` registrations**.
- No new outcome testing, candidate-return inspection, feature extraction or P1 reading occurred during this generation step.

`r1_order_book_liquidity_contract_v1.json` fingerprints the original R1.2 prefixes, fixes the new IDs/provenance counts, and documents scientific/infrastructure boundaries. `validate_order_book_generation.py` checks the complete expanded universe, ancestor identity, source lineage and book capability gates. The R1.2 validator now revalidates its own immutable historical prefix rather than requiring the growing universe to remain stuck at its previous count.

## Acceptance for the one-package commit candidate

```bash
python -m pytest tests/research_governance/test_r1_order_book_generation.py -v
python research/r1/order_book/validate_order_book_generation.py
python research/r1/order_book/validate_order_book_generation.py --self-test
python research/r1/legacy_7m/validate_legacy_lineage.py
python -m pytest tests/research_governance -q -rs
```

The delivered ZIP includes research README changes, manifest, contracts, registry appendices, validator and tests together. Acceptance results are assessed before Git commit; **no separate metadata-only patch is part of this step**. Proceed next to R1.4 — perpetual trade-flow generation — after accepting/freeze-tagging R1.3. P1 remains sealed throughout.
