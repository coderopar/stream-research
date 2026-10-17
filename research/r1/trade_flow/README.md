# R1.4 — Perpetual trade-flow hypothesis-generation pass

## Question and scope

Which **causally observable** properties of perpetual-futures trade flow might distinguish price discovery, continued execution, exhaustion, or purely two-sided turnover? This step **generates** hypotheses; it does not test returns, compute new trade features, optimize events/horizons, run a forecast, or access P1.

**Accepted parent:** `r1-order-book-liquidity-v1` and the R1.3 contract SHA recorded in `research/governance/r1_perps_trade_flow_contract_v1.json`. R1.3 remains an immutable 14-family/51-idea/4-session prefix. R1.4 appends **one** independent mechanism family, **16** ideas, and session `GEN-R1-UNIVERSE-V1` version **5**.

### The one new family

**Trade-arrival tempo and clustering** (`FAM-PERP-TRADE-ARRIVAL-TEMPO`). A burst/quiet period or interarrival clustering concerns the *timing of observed demand*, not necessarily its sign, parent-order identity, or realized price impact. Variants in tempo windows, intensity thresholds, side condition and point-process fit remain one conceptual family. Price discovery, flow persistence, and impact-efficiency deepening reuse the existing R1.1 families.

### Sixteen ideas

| Group | Governed idea | R2 structural status |
|---|---|---|
| Flow persistence | `SIDE-DOMINANCE-PERSISTENCE` | Testable with current bucket state |
| Flow persistence | `SIDE-PRESSURE-ACCELERATION` | Testable with current bucket state |
| Flow persistence | `SIDE-DOMINANCE-REVERSAL` | Testable with current bucket state |
| Price discovery | `IMBALANCE-ACTIVITY-NORMALIZATION` | Testable with current bucket state |
| Price discovery | `COUNT-QUANTITY-DIVERGENCE` | Testable with current bucket state |
| Price discovery | `NOTIONAL-SIDE-ASYMMETRY` | Testable with current bucket state |
| Arrival tempo | `FLOW-BURST-CONTINUATION` | Testable with current bucket state |
| Arrival tempo | `FLOW-BURST-DISSIPATION` | Testable with current bucket state |
| Arrival tempo | `FLOW-SIDE-BURST-COUPLING` | Testable with current bucket state |
| Arrival tempo | `FLOW-QUIET-TO-ACTIVE-TRANSITION` | Testable with current bucket state |
| Flow persistence | `FLOW-BIDIRECTIONAL-CHURN` | Testable with current bucket state |
| Flow persistence | `FLOW-SHORT-LONG-REGIME-CONTRAST` | Testable with current bucket state |
| Arrival tempo | `INTERARRIVAL-SELF-EXCITATION` | **Blocked:** individual trade events and timestamps |
| Flow persistence | `INDIVIDUAL-PRINT-TAIL-SKEW` | **Blocked:** individual trade sizes |
| Impact efficiency | `PRICE-RESPONSE-FLOW-DECOUPLING` | **Blocked:** historical causal price path |
| Legacy failed-branch retention | `LEGACY-TRADE-SIZE-HHI-FAILED-REPLICATION` | **Blocked:** individual trade sizes, **LEGACY_RESULT_INFORMED** |

Prefix every short code with `IDEA-R1-` for its permanent universe identifier. All sixteen are *pre-hypothesis seeds*, not `HYP-*` registrations, and none is evidence of positive net expectancy. The legacy idea retains `LEG7M-07` and its previously failed discovery/validation replication; it cannot be re-described as a successful signal or silently changed into outcome-blind status.

### R0.5 causal input capabilities — explicit limits

The accepted `PERPS_TRADES` feature store has receive-time-governed **1-second buckets over a trailing minute** and **1-minute buckets over a trailing 30 minutes**, including `trade_count`, `quantity_sum`, `notional_sum`, and raw `buyer_maker_true_quantity` / `buyer_maker_false_quantity`. Those are *storage grids*, not fixed research feature windows. The current R0.5 production store is **indexed by the frozen legacy D0 candidates**, not a universal queryable state for arbitrary new event definitions. `TESTABLE_NOW` therefore means these fields are available for that existing D0 substrate; a fresh candidate policy will first need a governed candidate-state rebuild or equivalent causal capability. Every later test must choose its transformation/window/candidate/latency on D0 and account for the exact information-availability clock.

- `buyer_is_maker` is retained as the raw observed field. Its marketable buy/sell aggressor semantics **must be verified against the exchange/feed definition** before deriving a signed signal. Do not silently treat `true` as buy aggression. Unknown/missing categories are quality flags, not zero-valued inferred trades.
- `trade_count` is the count of **canonical feed trade rows**, not the count of distinct people, parent orders, independently initiated market orders, or unique counterparties.
- Quantity/notional sums do not recover the size distribution of individual prints. Aggregated buckets do not recover **sub-second trade interarrival times** or a continuous historical trade-price path. Although the trade-state record carries one `latest_trade_price`, it does not provide the sequence needed for a causal impact curve.
- These trade-flow ideas do not get to use R0.6 arrival/executable book states as predictors, and R0.6's zero-latency/1,000-USDT profile is **audit-only**. Position return gates require actual D0-developed latency/fees/depth/size/funding assumptions.
- `TESTABLE_NOW` describes only the existence of the required *causal state family*. It does not mean the complete position, execution assumptions, evidence-control, return/risk gates, or scientific confirmation are already registered.

### Mechanism literature and why it is not confirmation

The existing R1.1 literature is reused. R1.4 adds three public sources in `new_public_sources.json`: Lillo–Farmer (2004) on long-memory of order signs; Bacry–Mastromatteo–Muzy (2015) on Hawkes/point-process modeling; and Easley–López de Prado–O'Hara (2012) on flow toxicity as adverse-selection framing. These sources support **candidate mechanisms**, not verified BTCUSDT predictive profits; no empirical result from them or our archived D0 studies sets a threshold or a performance claim here. In particular, we **do not** implement VPIN or fit a Hawkes process in R1.4.

### Protection and genealogy

- 15 new ideas `OUTCOME_BLIND` (public theory and first principles); **one** `LEGACY_RESULT_INFORMED` (the original failed 7M trade concentration branch).
- No new D0 outcomes opened, no historical 7M result recalculation, no P1 market/outcome access, and no outcome-based pruning.
- Frozen R1.3 JSONL prefix bytes are preserved exactly. R1.4 records/families/session are hash-frozen and future R1.5+ append is allowed without rewriting these records.
- Original 25-bp / 60-s policy and P75/15m label stay historical ancestry, not required future candidate/event definitions.
- All 4 remaining lenses are still mandatory: `DERIVATIVES_POSITIONING`, `CROSS_VENUE`, `VOLATILITY_REGIME`, `PREDICTIVE_MODEL`.

### Local acceptance

The complete commit candidate includes the contract, literature catalogue, 16 idea records, family/session records, validator, tests and both README updates; there is **no separate final metadata patch**.

```bash
python -m pytest tests/research_governance/test_r1_perps_trade_flow_generation.py -v
python research/r1/trade_flow/validate_trade_flow_generation.py
python research/r1/trade_flow/validate_trade_flow_generation.py --self-test
python research/r1/order_book/validate_order_book_generation.py
python -m pytest tests/research_governance -q -rs
```

**Next:** R1.5 derivatives and positioning generation. No R1.4 idea is promoted/tested as a trading strategy in this step.
