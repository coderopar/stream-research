# Governed research registries

These files are the append-only research-control surface introduced in R0.7 under Protocol v1.2.
They are intentionally empty at installation: R0.7 creates governance machinery, not hypotheses.

## Files

- `hypothesis_versions.jsonl` — one immutable frozen record per hypothesis version.
- `hypothesis_events.jsonl` — append-only lifecycle promotion/classification events. Negative and invalidated results remain visible.
- `evidence_budgets.jsonl` — frozen hypothesis-family evidence/multiplicity budgets with an explicit operating guarantee.
- `protected_cohorts.jsonl` — immutable named D1/reserve cohort architecture with boundaries, data version, source hashes, timezone/unit contracts, coverage gaps, and quarantine status.
- `confirmation_batches.jsonl` — finite immutable D1 batch manifests. A batch must be frozen before protected performance access.
- `access_events.jsonl` — future per-role/cohort protected access and contamination events.
- `decision_log.jsonl` — material research/governance decisions and whether relevant outcomes were already seen.

The historical R0.0 declaration remains in `research/governance/outcome_access_register.csv`; R0.7 does not rewrite that accepted artifact.

## Append-only rule

Never edit or delete an existing JSONL row to make history cleaner. A material hypothesis change creates a new hypothesis version. A status change is a new lifecycle event. A failed/invalidated result stays present. If a protected-data exposure occurs, append an access event immediately.

All immutable version/budget/batch records carry a `freeze_hash` computed from deterministic canonical JSON. The library code rejects a record whose contents no longer match its freeze hash.

## Confirmation boundary

An empty registry is expected during R0.7. Before any future D1 unblinding, the program must have, at minimum:

1. complete hypothesis versions and genealogy;
2. fresh-context adversarial review;
3. frozen evidence budget with a real operating guarantee;
4. deterministic dependence/episode rule;
5. strongest relevant benchmark, MEE, and mandatory risk gates;
6. frozen D1 feasibility outputs/fallback rules;
7. immutable confirmation batch, protocol/data/code identifiers, artifact hashes, and access roster.

Detailed D1 feasibility access is itself design-relevant information. R0.7 therefore tracks `DESIGN_CONTAMINATED` separately from `OUTCOME_CONTAMINATED` and never allows contamination to be downgraded.
