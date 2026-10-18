# R1.7 — Volatility and Regime Hypothesis Generation

## Scope and economic question

Can **pre-decision, causally observable BTCUSDT price variation** identify states of changing uncertainty or modify the relationship between liquidity, trade flow, derivatives positioning, and *net executable* position outcomes? This step **generates ideas only**. No D0 outcome research, return scoring, parameter optimization, position tests, model fitting, or protected P1 access occurs here.

**Accepted parent:** `r1-cross-venue-v1` (`d5ea875`, frozen R1.6 16 families / 101 ideas / seven generation sessions). R1.7 adds one distinct family, sixteen outcome-blind ideas, generation session v8, three verifiable literature records, a contract, validator and adversarial tests. Universe now has **17 families / 117 ideas**. Only the predeclared `PREDICTIVE_MODEL` lens remains before R1.9 review.

## Business and scientific meanings

- **Realized variation**: a backward-looking estimator of price variation computed only from prices that had been received by the decision time. It is not a guarantee of future risk and not a long/short signal.
- **Volatility clustering / memory**: sustained high or low *conditional uncertainty* may persist even when price direction is unpredictable. Conditional variance is different from conditional mean.
- **Regime**: a label assigned *as of* the decision using a predeclared historical measurement, model, update frequency, and cutoff. A hindsight interval identified using its complete later price path is not a causal regime.
- **Regime transition**: a state change that becomes actionable only after the online detection delay. Do not backdate it to the first tick of a later-identified breakout.
- **Liquidity–volatility feedback**: a potentially joint or mediated relationship among volatility, withdrawal/replenishment and trading pressure. Predictive interactions are **not** a demonstrated causal effect.
- **Regime robustness**: a preregistered full-sample hypothesis can be stress-tested across ex-ante regimes. A favorable subgroup noticed after a failure cannot retroactively confirm the failed parent strategy.
- **Volatility-based sizing**: portfolio/position risk adjustment. This is a **new strategy dimension** needing its own freeze and execution/risk assumptions, not a correction to a previously tested position.

## Sixteen governed seeds

| Topic | Idea ID suffix | Conceptual family |
|---|---|---|
| Conditional variation persistence | `VOL-PERSISTENCE` | New volatility memory/transition |
| Accelerating realized variation | `VOL-ACCELERATION` | New |
| Previously compressed variation | `VOL-COMPRESSION` | New |
| Volatility-shock decay | `VOL-SHOCK-DECAY` | New |
| Signed shock asymmetry | `VOL-SIGNED-ASYMMETRY` | New |
| Discrete jump vs diffuse variation | `VOL-JUMP-CONTINUOUS` | New |
| Sparse/stale print sensitivity | `VOL-SAMPLING-SENSITIVITY` | New |
| Online transition-detection delay | `VOL-TRANSITION-DETECTION` | New |
| Historical estimator disagreement | `VOL-UNCERTAINTY-PROXY` | New |
| Prior volatility × trade imbalance | `VOL-FLOW-IMBALANCE` | Existing volatility–liquidity feedback |
| Prior volatility × activity burst | `VOL-FLOW-BURST` | Existing |
| Prior volatility × order-book fragility | `VOL-LIQUIDITY-FRAGILITY` | Existing |
| Prior volatility × refill resilience | `VOL-REFILL-RESILIENCE` | Existing |
| Prior volatility × derivatives crowding | `VOL-DERIVATIVES-CROWDING` | Existing |
| Prior volatility × executable costs | `VOL-CONDITIONED-EXECUTION` | Existing; extra execution gate |
| Preregistered regime robustness | `VOL-ROBUSTNESS-DIAGNOSTIC` | New |

All short names are prefixed `IDEA-R1-`. They are *ideas*, not `HYP-*` registrations, independent strategies, or confirmed performance effects. Volatility estimator, threshold, regime count, sampling cadence, direction and horizon variants do **not** create new statistical families. The existing R1.1 volatility–liquidity feedback family is reused for the six interactions rather than relabeling the same economic mechanism.

## Strict infrastructure gate

**All 16 ideas are `BLOCKED_PENDING_GOVERNED_STATE`** except `VOL-CONDITIONED-EXECUTION` (`BLOCKED_PENDING_STATE_AND_EXECUTION`). R0.5 currently contains receive-time causal `PERPS_TRADES` count/quantity/notional/maker-flag buckets for the historical D0 candidate population. It does **not** supply a governed historical price-return path with quality, as-of timestamps and causal realized-volatility transforms. `latest_trade_price` in one trade-state record is not a series. Trade intensity alone **must not** be labeled price volatility. Historical data physically present in the canonical archive is not automatically an approved predictor state.

R0.6 marketable-order replay can audit executions, but its zero-added-latency 1,000-USDT assumption is not a realistic economic strategy model and its order book is not an approved reusable predictor. Any volatility–book or volatility–positioning interaction needs the separately governed book or derivative predictor family. Volatility-conditioned execution additionally needs a documented latency, spread/depth/slippage, funding/fees and tail-risk cost model. A new candidate definition requires its own causal feature build; R0.5 is candidate-indexed rather than universal.

## Causal ordering and falsification hazards

1. Every measurement must be formed from **integer-nanosecond exchange/receive-clock-correct** observations available before entry, with publication delay, missingness and late updates handled explicitly. Outcome start is no earlier than executable entry.
2. A complete current-day high/low, full-day realized volatility, symmetric smoothing filter, future-return quartile, or retrospective “breakout onset” is **not** available before it occurs. A change-point regime must carry detection latency.
3. Volatility sampling at sparse trade prices is vulnerable to staleness and microstructure artifacts. Fix observation-quality rules and investigate estimator sensitivity without selecting the most favorable outcome-defined threshold.
4. A+ B predicting returns better than A or B defines a **joint predictive state**, not an identified A→B causal mechanism. If book depletion mediates volatility, controls must respect the proposed estimand and time ordering.
5. Related regimes share market episodes and R0.4 overlap clusters. Multiple regime cutoffs, windows, feature combinations, and horizons count toward one family’s inference/multiplicity budget; long forward returns are not IID.
6. A high-volatility slice found after seeing bad returns must be *new future research*; it cannot rescue a failed original confirmatory version. D1 or P1 covariate distributions also cannot be mined to tune regime feasibility.
7. Any alleged net position advantage later requires a D0-designed entry/exit clock, benchmark, minimum economically meaningful effect, stress losses/MAE and feasible execution/cost accounting. Even perfectly forecast volatility may have **zero directional edge**.

## Literature and provenance

Three public peer-reviewed sources in `new_public_sources.json` support conceptual mechanisms only: Engle (1982) conditional variance clustering; Andersen–Bollerslev–Diebold–Labys (2003) realized variation; Moreira–Muir (2017) volatility-managed portfolio sizing. They do not validate BTCUSDT outcomes, strategy rules, empirical thresholds, or signed market direction. **All sixteen new ideas are `OUTCOME_BLIND`**. No historical 7M volatility outcome is being silently imported as an independent finding; previous 7M result-informed families and ideas remain in their frozen prefix with unchanged ancestry.

## Local acceptance and freeze

Run after applying the **single changed-files ZIP** at repository root:

```bash
python -m pytest tests/research_governance/test_r1_volatility_regime_generation.py -v
python research/r1/volatility_regime/validate_volatility_regime_generation.py
python research/r1/volatility_regime/validate_volatility_regime_generation.py --self-test
python research/r1/cross_venue/validate_cross_venue_generation.py
python -m pytest tests/research_governance -q -rs
```

The validator checks immutable R1.6 parent **raw JSONL prefix bytes**, hashed new records, provenance and literature roster, capability gates, regime leakage, P1 firewall, original stopping rule and empty permanent registry. It accepts legitimate future append-only sessions. Manifest, both living README updates and tests ship in the same commit-candidate ZIP; no separate acceptance-metadata patch.

**Next:** R1.8 predictive-model lens, followed by R1.9 universe freeze and fresh-context adversarial review, before any new D0 outcome testing.
