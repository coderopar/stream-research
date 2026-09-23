# R0.1 — Dataset inventory and immutable provenance

This component inventories the real canonical market-event tree in the local
`stream-research` repository without touching derived outcomes.

## Safety boundary

`build_dataset_inventory.py` requires the *logical entry-point* to live under
`<repo>/data/canonical`. The entry-point may itself be a symlink to an external
canonical store (the current repository uses
`data/canonical/market_events_v1 -> D:/stream-data/canonical/market_events_v1`).
After that approved root is resolved, nested symlinks are rejected. The scanner
therefore cannot be repointed at `data/derived`, reports, backtests, hypothesis
outputs, or arbitrary external directories.

For Parquet files it may read footer metadata (schema and row count). It does not select
or aggregate market-event columns. Full-file reads are performed only to compute SHA-256.

## Expected historical partition

- D0: `2026-05-18` through `2026-07-18` — development / outcome-spent.
- P1_PROTECTED: begins `2026-07-19` — protected / outcome-unseen.
- The P1 terminal date is **discovered from the actual canonical paths**. It is not guessed.

## Run

From the repository root, with the project virtual environment activated:

```bash
python research/r0/provenance/build_dataset_inventory.py
```

A quick no-hash dry run is available for debugging only:

```bash
python research/r0/provenance/build_dataset_inventory.py --hash-mode none
```

The R0.1 acceptance run MUST use the default `--hash-mode full`.

The generated manifests record both the repository-relative logical entry-point and
the resolved physical canonical root so the external-store topology is explicit. On
Windows/MSYS the authoritative redirect indicator is whether the logical and resolved
paths differ; ``Path.is_symlink()`` is retained only as an OS diagnostic.

Canonical extraction bookkeeping under ``_extraction_state/**`` is inventoried and
hashed as ``CANONICAL_METADATA`` but is not assigned to D0 or P1. Any other undated
canonical file is classified ``UNKNOWN_UNDATED`` and blocks R0.1 closure until reviewed.

## Generated outputs

The script writes to `research/governance/datasets/`:

- `CANONICAL_FILE_INDEX_v1.csv`
- `FEED_COVERAGE_v1.csv`
- `CANONICAL_SCHEMA_INVENTORY_v1.json`
- `D0_DATASET_MANIFEST_v1.json`
- `P1_PROTECTED_MANIFEST_v1.json`
- `R0_1_INVENTORY_SUMMARY_v1.json`

## Closing R0.1

R0.1 is not closed merely because the script runs. Review
`R0_1_INVENTORY_SUMMARY_v1.json` and resolve any `UNKNOWN_UNDATED` or `UNCLASSIFIED`
market-event files. `_extraction_state/**` metadata is explicitly non-cohort provenance
and does not block closure. The acceptance run should report:

- canonical files hashed;
- a discovered P1 last date;
- no scan outside the canonical tree;
- D0 and P1 manifests generated;
- schema inspection completed or explicitly reported unavailable.
