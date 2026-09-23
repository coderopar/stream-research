# BTCUSDT research lineage reset

**Reset ID:** `R0_RESET_2026-09-22`
**Governing protocol:** BTCUSDT Scientific Trading Research Protocol v1.2
**Status:** Frozen for the R0/R1 research cycle.

## Purpose

This reset separates the historical research lineage from the next evidentiary cycle.
The previous work is retained as valuable development history, engineering infrastructure,
negative evidence, and hypothesis ancestry. It is not treated as untouched confirmation.

## D0 — development data

The established historical study period is **2026-05-18 through 2026-07-18**.
The period has been repeatedly inspected across the research lineage and is therefore
classified `OUTCOME_SPENT`.

D0 may be used without restriction for:

- feature engineering;
- hypothesis generation;
- threshold and horizon exploration;
- interaction discovery;
- model development;
- execution simulation development;
- falsification and adversarial analysis;
- debugging and clock repair;
- power/support estimation for future registered tests.

No result discovered or tuned on D0 is an independent confirmation result.

## P1_PROTECTED — untouched historical pool

The post-D0 canonical extension begins after **2026-07-18** and is registered as
`P1_PROTECTED`. Its exact terminal date must be discovered from the local canonical
partition inventory rather than guessed from the review bundle.

The user has declared that the extension was only downloaded and canonicalized. No future
returns, P35/P50/P75/P100 labels, strategy P&L, backtests, signal performance, or candidate
future paths have been inspected.

P1 is deliberately **not yet called D1**. The decision to use it as one confirmation cohort
or as pre-frozen confirmation/reserve cohorts must be made from D0 support/power analysis
before any protected outcome is opened.

## Historical result classifications

The reset does not erase the old research lineage. Historical artifacts should be treated as
one of four categories:

1. **Retained infrastructure** — canonical ingestion, exact replay components, causal feature
   construction and other engineering that survives audit.
2. **Known-invalid result** — analyses invalidated by a demonstrated implementation,
   timing, matching or inference defect.
3. **Exploratory evidence** — associations discovered within the repeatedly inspected D0
   lineage, including results that previously used discovery/validation language.
4. **Hypothesis ancestry** — ideas worth revisiting scientifically, with their historical
   parentage explicitly preserved.

## R0.0 freeze rules

- Protocol v1.2 is the governing protocol for this cycle.
- P1 protected outcomes remain sealed through R0.
- Access to protected outcomes must be logged in `outcome_access_register.csv`.
- Canonicalization, path inventory, hashing, schema inspection and coverage checks do not by
  themselves consume P1 outcome evidence.
- No post-result partitioning of the already collected P1 period is permitted.

## R0.1 provenance rule

The adversarial-review bundle intentionally omitted raw market events and large canonical
substrates. Therefore this patch does **not** fabricate P1 dates, file counts, hashes or feed
coverage. `research/r0/provenance/build_dataset_inventory.py` must be run inside the actual
`stream-research` repository against `data/canonical/market_events_v1`.

The generated manifests become the authoritative R0.1 inventory artifacts.
