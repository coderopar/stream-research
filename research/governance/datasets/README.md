# Dataset governance artifacts

This directory receives generated R0.1 provenance outputs.

Expected files after running the inventory builder:

- `CANONICAL_FILE_INDEX_v1.csv`
- `FEED_COVERAGE_v1.csv`
- `CANONICAL_SCHEMA_INVENTORY_v1.json`
- `D0_DATASET_MANIFEST_v1.json`
- `P1_PROTECTED_MANIFEST_v1.json`
- `R0_1_INVENTORY_SUMMARY_v1.json`

These outputs are intentionally not pre-populated in the patch because the supplied
adversarial-review bundle omits the actual canonical market-event files.
