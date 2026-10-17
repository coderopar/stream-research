# R1 Hypothesis-Universe Registries

These append-only registries sit before the permanent R0.7 hypothesis registry.
They preserve broad R1 idea generation without pretending that every brainstorm
is already a registered hypothesis or a tested trading proposition.

Files:

- `generation_sessions.jsonl` — frozen generation plans and later append-only session updates;
- `families.jsonl` — conceptual mechanism families used to prevent family splitting;
- `ideas.jsonl` — immutable/versioned R1 idea packets with source and result lineage.

R1.0 froze the first generation plan without creating market ideas. R1.1 has now appended the first eight outcome-blind mechanism families, fifteen outcome-blind idea records, and generation-session version 2 under the same append-only contract. Later R1 stages must continue to append rather than rewrite these records.

A universe idea is **not** yet a permanent hypothesis. Promotion is explicit and
must preserve family, result, dataset, and ancestry fields into the R0.7
`hypothesis_versions.jsonl` registry. R0.7 lifecycle, feasibility, evidence
budget, adversarial-review, and D1 registration gates continue to govern after
promotion.

P1/D1 data are not permitted sources for R1 generation. D0-result-informed and
legacy-result-informed ideas are allowed only when their influence is recorded
explicitly; outcome-blind ideas may not claim result independence if observed
results actually influenced their design.


R1.1 source catalogue and mechanism rationale live under `research/r1/theory_mechanisms/`. Public literature references are qualitative provenance only; no project D0/P1 outcomes are embedded in the R1.1 idea packets.

## R1.2 legacy ancestry addition

Following R1.1, append-only session v3 completes `LEGACY_7M_LINEAGE`. Five new families and twenty `LEGACY_RESULT_INFORMED` ideas are admitted; all cite hashed archived-report evidence IDs in `research/r1/legacy_7m/legacy_evidence_catalogue.json`. Failed and feasibility-only branches are retained. Existing R1.1 family/idea/session JSONL prefixes remain byte-identical. The legacy candidate policy and older event outcomes are not new confirmatory evidence, and P1 stays sealed.
