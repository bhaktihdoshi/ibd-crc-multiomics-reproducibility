# sCCA Nested Preflight Report

Current implementation status before this pass: `placeholder/not executed`.

Corrections made before any full nested run: fold-local median imputation, zero/near-zero-SD filtering after imputation, fold-local scaling, frozen transformation of validation/test folds, single PMA fit helper, checkpoint-success validation and schema-valid output writing.

PMA::CCA is kept with `typex="standard"` and `typez="standard"`; therefore no zero/near-zero-SD column is allowed to reach PMA even though inputs are already externally standardised.

Selection rule: maximise mean absolute inner-fold cross-view canonical correlation; break ties by lower inner-fold SD; then fewer selected features; then simpler parameter order.

Public fold diagnostics use anonymous row keys only; participant IDs are not exported.

| test                              | status   | observed   | expected   |
|:----------------------------------|:---------|:-----------|:-----------|
| implementation_complete           | pass     | True       | True       |
| fold_local_preprocessing_verified | pass     | True       | True       |
| zero_sd_filtering_verified        | pass     | True       | True       |
| pma_standardisation_verified      | pass     | True       | True       |
| manual_projection_verified        | pass     | True       | True       |
| inner_tuning_verified             | pass     | True       | True       |
| fold_integrity_verified           | pass     | True       | True       |
| component_alignment_verified      | pass     | True       | True       |
| resume_logic_verified             | pass     | True       | True       |
| output_serialisation_verified     | pass     | True       | True       |
| unit_tests_passed                 | pass     | True       | True       |
| single_real_fold_passed           | pass     | True       | True       |
| original_hashes_unchanged         | pass     | True       | True       |
| prior_focused_outputs_unchanged   | pass     | True       | True       |
