# R1 — Fresh Hypothesis Universe Generation

R1 begins only after the R0 research foundation passed its adversarial acceptance
gate and was frozen at commit `71192f37e8aa44da4c30326048debf65b1ca5792`,
tag `r0-foundation-accepted-v1`.

R1 asks a different question from R0:

> What plausible, economically meaningful BTCUSDT market states or mechanisms
> should the governed D0 research program investigate?

R1 is hypothesis generation, not protected confirmation and not yet R2 outcome
testing. The untouched `P1_PROTECTED` extension remains sealed.

## Governed sequence

```text
R1.0  Hypothesis-generation governance + universe schema
R1.1  Theory / market-mechanism families
R1.2  Legacy 7M lineage
R1.3  Order-book / liquidity hypotheses
R1.4  Perpetual trade-flow hypotheses
R1.5  Derivatives / positioning hypotheses
R1.6  Cross-venue hypotheses
R1.7  Volatility / regime hypotheses
R1.8  Predictive-model hypothesis track
R1.9  Universe freeze + adversarial review
```

The predeclared generation plan also requires every lens above to be addressed
before R1.9 can freeze the universe. Missing lenses need an explicit
non-outcome-based waiver; they cannot simply disappear because another branch
looks more promising.

## R1.0 boundary

R1.0 installs the record formats and guardrails before R1.1 generates the first
fresh families or ideas.

Three identities must remain distinct:

1. **Universe idea** — a broad R1 proposition with mechanism, provenance,
   required causal state, position concept, risks, and infrastructure needs.
2. **Permanent hypothesis** — an idea deliberately promoted into the R0.7
   `HYP-*` registry with permanent genealogy and versioning.
3. **Registered candidate** — a much later state requiring the full position,
   execution, benchmark/MEE, inference, risk, evidence-budget, D1-feasibility,
   and adversarial-review specification.

R1.0 therefore prevents brainstorming records from receiving scientific status
that they have not earned.

## Result provenance

R1 permits three generation modes:

- `OUTCOME_BLIND` — no observed result influenced the idea;
- `D0_RESULT_INFORMED` — an observed D0 result influenced the design and must be
  recorded in `derived_from_results` and `datasets_that_influenced_design`;
- `LEGACY_RESULT_INFORMED` — old/pre-reset D0 work influenced the idea and its
  ancestry remains visible.

A result-informed idea is not disallowed. Hiding that influence is disallowed.
R1 idea packets store result *lineage references*, not embedded p-values,
backtest returns, confidence intervals, or newly selected thresholds/horizons.
Fresh outcome testing belongs to R2.

## Family discipline

Families are defined by economic mechanism / causal relationship, not by which
variant produced a favorable number. Each family must declare the threshold,
horizon, window, filter, interaction, or execution dimensions that remain in
that family and a written rule for what would constitute a genuinely new
family.

This is the first anti-family-splitting barrier. R1.9 will review it again before
the universe is frozen.

## Infrastructure is not scientific merit

R1 may generate hypotheses that current R0 state machinery cannot yet test.
For example, an options or order-book research-state hypothesis can be a valid
scientific idea even though R0.5 v1 currently exposes only `PERPS_TRADES` as a
governed reusable state family.

Every idea therefore carries a separate infrastructure assessment. Missing
capability may block R2 testing, but it must not lower or raise the idea's
scientific merit simply because implementation happens to exist.

No unsupported state family may be reconstructed ad hoc in R2 merely to test a
promising idea. The missing governed capability must be added under the same
causal/provenance standards first.

## Protected-data rule

R1 uses `NONE`, `PUBLIC`, `D0`, and `LEGACY_D0` source zones only.
`P1_PROTECTED`, D1, reserve-cohort data, protected performance, and protected
outcomes are forbidden generation inputs.

## R1.0 acceptance checkpoint

R1.0 passed local project-environment acceptance on 2026-10-09 and is awaiting only its Git freeze. R1.0-specific tests passed 29/29; the full research-governance suite passed 223 tests with 3 known Windows symlink-capability skips and zero failures. The validator returned `PASS` with zero family rows, zero idea rows, one frozen generation session, no D0 outcome read, no P1 market-data read, and no P1 outcome read. `P1_PROTECTED` remains `SEALED_OUTCOME_UNSEEN`.

This acceptance means the *rules for generating the universe* are ready. It does not endorse any BTCUSDT hypothesis because R1.0 deliberately created none. R1.1 must begin only after this checkpoint is frozen in Git.
