# R0.7 — Research Registries + Guardrails

R0.7 turns Protocol v1.2 from prose into executable research controls before fresh hypothesis generation begins.
It does not inspect D0 outcomes, P1 causal fields, or P1 outcomes and it creates no hypotheses.

## What is installed

- append-only, diff-friendly JSONL registries;
- deterministic freeze hashes for immutable hypothesis/budget/batch records;
- hypothesis genealogy and cycle validation;
- immutable protected-cohort architecture records for future named D1/reserve cohorts;
- governed stage transitions with no in-place rescue of terminal versions;
- a complete-registration gate for D1-bound hypotheses;
- evidence-budget validation that requires an actual operating guarantee;
- protected-role/cohort access rules and monotone design/outcome contamination;
- predictor/eligibility guards against governance or future-derived information;
- a time-respecting validation guard that rejects random/K-fold validation for dependent market data;
- substantive-idempotence and append-only-prefix primitives;
- an R0.7 validator/self-test that reads governance metadata only.

## Important boundary

R0.7 deliberately does **not** decide whether the untouched protected extension becomes one D1 cohort or a D1/reserve partition. Protocol v1.2 requires that decision to be justified from D0 independent-support/precision needs and frozen before protected performance unblinding. That decision belongs later in the pre-D1 reduction/freeze path, not in this infrastructure phase.

## Run

```bash
python research/r0/registries/validate_research_governance.py
python research/r0/registries/validate_research_governance.py --self-test
```

The normal validator checks registry syntax/invariants, Protocol v1.2 linkage, current P1 sealed status, the historical outcome-access declaration, and the R0.7 contract. It does not open market data or outcome artifacts.
