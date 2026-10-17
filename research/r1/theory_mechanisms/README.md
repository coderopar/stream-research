# R1.1 — Theory / Market-Mechanism Family Generation

R1.1 is the first stage that adds actual BTCUSDT research ideas to the governed
R1 universe. It is deliberately **outcome-blind** with respect to this project:
no D0 outcome table, legacy 7M result, P1 market partition, P1 outcome, backtest,
p-value, hit rate, or observed strategy return is used to decide which ideas are
admitted.

The purpose is not to identify the "best" signal. It is to create a broad,
auditable first-principles mechanism layer before R1.2-R1.8 deepen the remaining
predeclared generation lenses.

## Inputs

R1.1 uses two permitted source classes:

1. **Public market-microstructure literature.** The frozen qualitative source
   catalogue is `literature_sources.json`. Empirical findings from public papers
   are background evidence, not project D0/P1 results, and no fitted coefficient,
   p-value, return, or project threshold is copied into an idea packet.
2. **First-principles market-mechanism synthesis.** These are explicitly labelled
   `MARKET_MECHANISM` / `NONE` sources. Where a proposition goes beyond what a
   cited paper directly establishes, the idea text calls it a hypothesis or
   inference rather than attributing it to the literature.

The literature catalogue includes classic work on information-bearing trades,
adverse selection, delayed trade impact, persistent order flow, order-flow
imbalance and depth, liquidity resilience, limit-order-book dynamics, queue
imbalance, transient impact, and crypto cross-venue fragmentation.

## Families admitted in R1.1

R1.1 adds eight mechanism families:

1. `FAM-INFORMED-FLOW-PRICE-DISCOVERY` — directional trades can carry information
   that prices incorporate progressively; liquidity conditions the response.
2. `FAM-METAORDER-FLOW-PERSISTENCE` — split parent demand can create persistent
   same-sign flow, while liquidity adapts to it.
3. `FAM-LIQUIDITY-RESILIENCE` — the speed of liquidity recovery after a shock may
   distinguish impact decay from impact persistence.
4. `FAM-IMPACT-EFFICIENCY-ABSORPTION` — price movement per unit aggressive flow
   may distinguish effective penetration from absorption/exhaustion.
5. `FAM-QUEUE-PRESSURE` — near-touch queue asymmetry can alter immediate price
   pressure; failure of that pressure may itself be informative.
6. `FAM-TRANSIENT-IMPACT-DECAY` — unreinforced shocks may decay as liquidity
   recovers.
7. `FAM-CROSS-VENUE-INFORMATION-DIFFUSION` — fragmented BTC venues can incorporate
   common information at different speeds.
8. `FAM-VOLATILITY-LIQUIDITY-FEEDBACK` — thin liquidity and volatility can amplify
   impact, while liquidity recovery can damp that feedback.

These are **mechanism families**, not evidence-budget-independent variants.
Thresholds, horizons, windows, queue levels, venue pairs, and interaction cutoffs
remain declared within-family variant dimensions unless the economic mechanism
itself changes.

## Fifteen idea records

The authoritative ideas are in
`research/governance/universe/ideas.jsonl`. They cover:

- information-bearing flow continuation;
- flow impact conditioned by depth;
- signed-flow persistence;
- persistent flow conditioned by adaptive liquidity;
- fast-resilience reversion;
- slow-resilience continuation;
- efficient flow-to-price transmission continuation;
- aggressive-flow absorption/exhaustion reversal;
- near-touch queue imbalance direction;
- queue-pressure failure;
- isolated-shock decay;
- cross-venue lead-lag;
- common cross-venue flow confirmation;
- thin-liquidity amplification during volatile states;
- liquidity-recovery damping after stress.

No idea is ranked by scientific merit or historical profitability. Two ideas are
currently marked `TESTABLE_NOW` only because R0.5 already exposes the required
`PERPS_TRADES` state. The others remain scientifically admissible but are blocked
until the corresponding governed state capability exists. This distinction is
about infrastructure, not about whether an idea is "good."

## Boundary with later R1 stages

R1.1 intentionally does **not** exhaust every detailed order-book, trade-flow,
cross-venue, or volatility hypothesis. Those deeper passes remain predeclared:

- R1.3 — order-book / liquidity;
- R1.4 — perpetual trade flow;
- R1.6 — cross-venue;
- R1.7 — volatility / regime.

R1.1 seeds the mechanisms those stages may refine without using outcomes to prune
or privilege them. R1.2 separately handles legacy 7M result lineage; R1.1 uses no
legacy result evidence.

## Generation-session progress

R1.1 appends generation-session version 2 and marks only these lenses complete:

- `THEORY_LITERATURE`
- `MARKET_MECHANISM`

The session remains `IN_PROGRESS`. The seven remaining lenses must still be
addressed before R1.9 may freeze the universe.

## Validation

```bash
python research/r1/theory_mechanisms/validate_theory_mechanism_generation.py
python research/r1/theory_mechanisms/validate_theory_mechanism_generation.py --self-test
python -m pytest tests/research_governance/test_r1_theory_mechanism_generation.py -v
python -m pytest tests/research_governance -q -rs
```

## Acceptance boundary

Implementation is complete. Packaging validation passed 29/29 R1.1-specific tests and the R1.1 validator/self-test; the packaging environment full governance run passed 246 tests with 9 PyArrow-only skips and zero failures.

The authoritative acceptance boundary is the Git commit: this package includes registries, contracts, tests, manifests, and README changes together, and it must not be committed unless the project-environment R1.1 tests/validator and full research-governance regression pass. No separate post-test metadata patch is required.
