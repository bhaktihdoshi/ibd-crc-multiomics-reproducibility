# CRC Stage 2 Output Specification

Key Stage 2 outputs will include:

- contract validation tables
- canonical metabolite crosswalk and feature-order contract
- shared clinical feature contract
- nested CV results and sample-level OOF predictions
- selected-model table by training cohort and feature set
- frozen-threshold table from nested OOF predictions
- external predictions with frozen threshold and 0.5 labels
- external performance metrics and bootstrap confidence intervals
- post-run diagnostics
- Yachida-only stage/subtype analysis plans
- differential-abundance outputs
- pathway/enrichment status or results
- CRC integration package manifest
- freeze-readiness checks

Fast-test outputs must be written under `stage2/fast_test/` and labelled non-final.

Fast-test validation and fast-test primary smoke modelling are separate commands. The primary smoke command will perform a reduced model fit and must remain non-final.
