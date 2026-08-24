# CRC Stage 2 Transformation Policy

The primary bidirectional CRC-vs-Control external-validation analysis will retain the supplied aligned metabolomics values for both cohorts.

Primary policy:

- no log1p transformation by default
- no clipping of negative values
- no combined-cohort normalization
- no ComBat, SVA or Harmony before external validation
- missingness filtering, constant-feature filtering, imputation and scaling will be fitted inside training folds only
- the frozen fitted preprocessing pipeline will be applied unchanged to the external cohort

A log1p sensitivity analysis may be scaffolded later only if it is disabled by default, applied only to mathematically valid non-negative inputs, written to a separate sensitivity output directory, and never selected using external performance.
