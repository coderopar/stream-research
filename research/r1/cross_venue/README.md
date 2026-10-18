# R1.6 — Cross-venue hypothesis generation

## Scientific question

What observable information-transmission or venue-liquidity segmentation mechanisms could change the distribution of **executable BTCUSDT perpetual positions**? R1.6 generates ideas only. It does not construct a new trading policy, run D0 outcome tests, re-evaluate old 7M effects, rank ideas, or read P1.

**Accepted parent:** `r1-derivatives-positioning-v1` / `R1_5_DERIVATIVES_POSITIONING_GENERATION_V1`. The first 15 families, 85 ideas and six generation sessions are byte-frozen in `research/governance/r1_cross_venue_contract_v1.json`.

## Mechanism families and added ideas

R1.6 reuses **cross-venue information diffusion** for lead/lag and multi-venue information transmission, and creates **one** genuinely distinct family, `FAM-CROSS-VENUE-LIQUIDITY-SEGMENTATION`, for different *available venue-specific liquidity and executable price dislocations*. Venue choice, threshold, filter, horizon, spread and quote definition are variants **inside** these families; they do not create independent evidence budgets.

| Mechanism | Idea | Provenance |
|---|---|---|
| Information diffusion | Spot trade flow precedes perpetual repricing | OUTCOME_BLIND |
| Information diffusion | Perpetual flow precedes spot repricing | OUTCOME_BLIND |
| Information diffusion | Spot/perpetual trade-flow disagreement | OUTCOME_BLIND |
| Information diffusion | OKX spot BBO versus Binance perp lead/lag | OUTCOME_BLIND |
| Information diffusion | Binance versus OKX spot quote disagreement | OUTCOME_BLIND |
| Information diffusion | Common versus venue-local information shocks | OUTCOME_BLIND |
| Information diffusion | Apparent sequence versus receive-time sequence | OUTCOME_BLIND |
| Information diffusion | Venue quote response during spot quiet intervals | OUTCOME_BLIND |
| Liquidity segmentation | Cross-venue displayed top-of-book quantity | OUTCOME_BLIND |
| Liquidity segmentation | One-venue spread-widening asymmetry | OUTCOME_BLIND |
| Liquidity segmentation | Different venue-local liquidity recovery | OUTCOME_BLIND |
| Liquidity segmentation | Multi-venue apparent arbitrage versus executable cost band | OUTCOME_BLIND |
| Information diffusion | Binance spot-flow null/no discrimination | LEGACY_RESULT_INFORMED |
| Information diffusion | Binance spot-price failed matching | LEGACY_RESULT_INFORMED |
| Information diffusion | OKX BBO imbalance null/inconclusive | LEGACY_RESULT_INFORMED |
| Information diffusion | OKX spot-price failed matching | LEGACY_RESULT_INFORMED |

**R1.6 total:** 16 ideas (12 `OUTCOME_BLIND`, 4 `LEGACY_RESULT_INFORMED`), one new family. Universe cumulative: **16 families / 101 ideas**, generation session v7; remaining lenses `VOLATILITY_REGIME` and `PREDICTIVE_MODEL`.

## Evidence provenance

- Existing theory source: Makarov–Schoar (2020), `LIT-MAKAROV-SCHOAR-2020`, motivates common and venue-specific shocks and price dislocations without importing paper statistics as BTCUSDT results.
- Two new documented sources: Binance Spot WebSocket market streams and OKX public WebSocket book channels, frozen in `new_public_sources.json`. API documentation is **not proof of 2026 historical feed completeness or profitable arbitrage**.
- `LEG7M-09` (Binance spot) and `LEG7M-10` (OKX) remain hashed, archived D0 *result-informed* ancestry. The historical spot-flow non-discrimination, OKX BBO null/inconclusive branch and failed spot/OKX price matching gates are retained, not rescued by relabeling or treated as independent confirmation. The original 25-bp candidate policy and P75/15m label are historical development choices, not mandated for new research.

## Non-negotiable scientific boundaries

1. **Receive time governs causality.** Exchange timestamps, local receipt clocks and book snapshot/reporting cadence are distinct; do not infer cross-venue lead/lag by nearest/future-timestamp joins, fill missing packets with invented quotes, or condition on later feed completeness. Require exact as-of state, freshness, clock contract and asset/instrument/quote-unit normalization for every venue.
2. **Compare comparable instruments.** BTCUSDT Binance perpetual, Binance BTCUSDT spot and OKX BTC-USDT spot have different contract, tick, spread, depth, liquidity and fee conventions. A numerical quote difference is not automatically an economic mispricing.
3. **No risk-free arbitrage conclusion.** A displayed BBO/midprice gap is not a locked profit. Two-leg exchange-arrival execution, order-book depth at both venues, independent fees, latency, fills, failed-leg inventory risk, capital access, venue credit/transfer constraints and settlement/funding must be modeled and frozen before such a claim.
4. **OKX snapshots are not continuous full depth.** The documented `books5` stream provides top-five **snapshots**, not full depth or a complete trade tape; top-of-book imbalance does not prove queue identities or passive order fills.
5. **No unguided source promotion.** Canonical Binance spot/OKX files may exist, but R0.5 currently governs only historical D0 PERPS_TRADES buckets. R0.6 reconstructs one-venue execution books, not a causal multi-venue predictor store or a live two-leg execution simulator. All 16 ideas remain **blocked for R2** until governed as-of source states are built; the two-leg cost-band idea additionally needs multi-venue execution/settlement infrastructure.
6. **Hypothesis generation, not testing.** The position side, state thresholds, decision moment, horizon, benchmark, minimum economic effect, risk and cost plan are left for D0. P1 stays sealed; the permanent R0.7 `HYP-*` registry remains empty.

## Validation

```bash
python -m pytest tests/research_governance/test_r1_cross_venue_generation.py -v
python research/r1/cross_venue/validate_cross_venue_generation.py
python research/r1/cross_venue/validate_cross_venue_generation.py --self-test
python research/r1/derivatives/validate_derivatives_generation.py
python -m pytest tests/research_governance -q -rs
```

Validators operate on governance/source text only, not on market datasets, D0 outcomes or P1 market/outcome files. Historical R1.5 prefix checks intentionally tolerate later append-only R1.7+ growth.

**Next:** R1.7 — volatility/regime hypothesis generation, followed by predictive-model lens R1.8 and R1.9 universe freeze/adversarial review.
