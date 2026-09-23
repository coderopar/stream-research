# R0 — Scientific reset and substrate repair

This patch implements the first two R0 work packages under Protocol v1.2.

## R0.0 — Governance freeze

Implemented by:

- `research/governance/research_reset_manifest.yaml`
- `research/governance/protected_data_register.yaml`
- `research/governance/outcome_access_register.csv`
- `research/governance/protocol_freeze_manifest.json`
- `research/governance/research_lineage_reset.md`
- `research/protocol/v1.2/BTCUSDT_Scientific_Trading_Research_Protocol_v1.2.docx`

The reset declares the old 2026-05-18 through 2026-07-18 period as D0 development data
and the newly added post-2026-07-18 extension as `P1_PROTECTED` with outcomes unseen.

## R0.1 — Dataset inventory and provenance

Implemented by:

- `research/r0/provenance/build_dataset_inventory.py`
- `research/r0/provenance/README.md`
- `research/governance/datasets/README.md`

Run the scanner in the actual repository. The repository's canonical entry-point may
be a symlink to the external canonical store; R0.1 explicitly supports that topology
while refusing derived/outcome/report roots.

R0.2 clock repair must not begin until the R0.1 generated manifests have been reviewed and
the R0.1 gate is explicitly marked complete.
