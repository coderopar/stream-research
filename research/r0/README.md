# R0 — Scientific reset and substrate repair

This directory tracks the governed R0 scientific-reset work packages under Protocol v1.2.

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

R0.1 is frozen at tag `r0-governance-provenance-v1`.

## R0.2 — Causal clock repair

Implemented by:

- `src/research_core/clocks/contract.py`
- `research/r0/clock_repair/`
- `research/governance/clock_contract_v1.json`
- `tests/research_governance/test_r0_clock_contract.py`

R0.2 replaces integer-second timing assumptions with an explicit int64-nanosecond
clock contract and regression tests for the historical feature/outcome overlap.

## R0.3 — Corrected D0 research substrate

Implemented by:

- `src/research_core/substrate/d0.py`
- `research/r0/substrate/build_corrected_d0_substrate.py`
- `research/r0/substrate/README.md`
- `research/r0/substrate/R0_3_IMPLEMENTATION_MANIFEST.json`
- `tests/research_governance/test_r0_d0_substrate.py`

R0.3 retains the historical 25-bp population as an explicitly labeled D0
lineage family. It reconstructs exact candidate observability only when the
legacy origin/release preconditions are already known from prior seconds,
recovers exact int64 anchors where available, rebuilds the +3s scientific label
clock under R0.2, and reconstructs future labels from canonical D0 trades without
using the old Phase-7M outcome tables or opening P1.
