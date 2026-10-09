# R1.0 — Hypothesis-Generation Governance + Universe Schema

R1.0 freezes how the hypothesis universe will be generated before R1.1 begins.
It creates no BTCUSDT hypothesis families, no market ideas, and no outcome tests.

## Installed controls

- append-only pre-hypothesis registries for generation sessions, families, and ideas;
- explicit `OUTCOME_BLIND`, `D0_RESULT_INFORMED`, and `LEGACY_RESULT_INFORMED` provenance;
- a P1/D1 source firewall;
- family definitions with anti-family-splitting and new-family boundary rules;
- required mechanism, causal-state, trigger, position, risk, and execution concepts for every idea;
- separate infrastructure/testability assessment that cannot masquerade as scientific ranking;
- explicit lineage for parent ideas, parent hypotheses, legacy research, observed D0 results, and influencing datasets;
- a promotion adapter that produces an R0.7-compatible `HYP-*` IDEA packet without bypassing later R0.7 gates;
- a frozen generation plan covering nine predeclared lenses before R1.9 may freeze the universe.

## Frozen generation lenses

1. `THEORY_LITERATURE`
2. `MARKET_MECHANISM`
3. `LEGACY_7M_LINEAGE`
4. `ORDER_BOOK_LIQUIDITY`
5. `PERPS_TRADE_FLOW`
6. `DERIVATIVES_POSITIONING`
7. `CROSS_VENUE`
8. `VOLATILITY_REGIME`
9. `PREDICTIVE_MODEL`

R1.9 cannot silently omit a lens because another branch looked better. Each lens
must have a documented pass or an explicit non-outcome-based waiver.

## What R1.0 does not do

- no D0 outcome scan;
- no P1/P1-outcome read;
- no threshold, horizon, or interaction selection from fresh outcomes;
- no scientific-merit scoring;
- no hypothesis testing;
- no permanent `HYP-*` registry mutation;
- no D1 cohort design or unblinding.

## Validation

```bash
python research/r1/universe/validate_hypothesis_universe_governance.py
python research/r1/universe/validate_hypothesis_universe_governance.py --self-test
python -m pytest tests/research_governance/test_r1_hypothesis_universe_governance.py -v
```

## Acceptance status

`LOCAL_ACCEPTANCE_PASS_GIT_FREEZE_PENDING`

Accepted project-environment evidence:

- R1.0-specific tests: 29 passed, 0 failed, 0 skipped;
- full research-governance suite: 223 passed, 3 known Windows symlink-capability skips, 0 failed;
- validator: `PASS`;
- registry state: 0 families, 0 ideas, 1 frozen generation session;
- D0 outcomes read: false;
- P1 market data read: false;
- P1 outcomes read: false;
- P1 status: `SEALED_OUTCOME_UNSEEN`.

The remaining acceptance action is the Git freeze of this R1.0 checkpoint.
