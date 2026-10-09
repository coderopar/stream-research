# R1 Hypothesis-Universe Registries

These append-only registries sit before the permanent R0.7 hypothesis registry.
They preserve broad R1 idea generation without pretending that every brainstorm
is already a registered hypothesis or a tested trading proposition.

Files:

- `generation_sessions.jsonl` — frozen generation plans and later append-only session updates;
- `families.jsonl` — conceptual mechanism families used to prevent family splitting;
- `ideas.jsonl` — immutable/versioned R1 idea packets with source and result lineage.

R1.0 freezes the first generation plan but creates no market hypothesis families
and no market ideas. R1.1+ may append records under the contract in
`research/governance/hypothesis_universe_contract_v1.json`.

A universe idea is **not** yet a permanent hypothesis. Promotion is explicit and
must preserve family, result, dataset, and ancestry fields into the R0.7
`hypothesis_versions.jsonl` registry. R0.7 lifecycle, feasibility, evidence
budget, adversarial-review, and D1 registration gates continue to govern after
promotion.

P1/D1 data are not permitted sources for R1 generation. D0-result-informed and
legacy-result-informed ideas are allowed only when their influence is recorded
explicitly; outcome-blind ideas may not claim result independence if observed
results actually influenced their design.
