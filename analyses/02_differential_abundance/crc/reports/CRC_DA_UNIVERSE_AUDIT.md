# CRC DA Universe Audit

## Finding

Current CRC DA is restricted to the Sinha/Yachida shared modelling feature space. Evidence: `run_differential_abundance()` reads `stage2/data_processed/crc_stage2_primary_metabolomics_canonical.csv`; that matrix is created by `canonical_matrix()`, which iterates only over `crc_stage2_shared_feature_space.csv`.

## Current trace

Raw metabolomics and maps → `build_canonical_feature_space()` → `crc_stage2_shared_feature_space.csv` → `canonical_matrix()` → `crc_stage2_primary_metabolomics_canonical.csv` → `run_differential_abundance()` → `crc_stage2_differential_abundance_results.csv`.

## Current counts

| disease   | cohort   | comparison     |   raw_metabolite_count |   identifier_valid_count |   valid_hmdb_feature_count |   valid_kegg_feature_count |   deterministic_high_confidence_count |   eligible_cohort_specific_count |   eligible_unique_hmdb_count |   eligible_unique_kegg_count |   missingness_ok_feature_count |   nonconstant_after_missingness_count |   unique_deterministic_canonical_metabolite_count |   pairwise_shared_model_count |   actual_da_tested_count |   fdr_significant_count |   kegg_mapped_tested_count |   ora_background_count | da_feature_source                                                                                                                     | restricted_to_model_space   | notes                                                                                                                             |
|:----------|:---------|:---------------|-----------------------:|-------------------------:|---------------------------:|---------------------------:|--------------------------------------:|---------------------------------:|-----------------------------:|-----------------------------:|-------------------------------:|--------------------------------------:|--------------------------------------------------:|------------------------------:|-------------------------:|------------------------:|---------------------------:|-----------------------:|:--------------------------------------------------------------------------------------------------------------------------------------|:----------------------------|:----------------------------------------------------------------------------------------------------------------------------------|
| CRC       | Sinha    | CRC vs Control |                    530 |                      354 |                        353 |                        189 |                                   353 |                              353 |                          352 |                          188 |                            353 |                                   353 |                                               354 |                           110 |                      110 |                       0 |                        107 |                    107 | stage2/data_processed/crc_stage2_primary_metabolomics_canonical.csv built from crc_stage2_shared_feature_space via canonical_matrix() | True                        | Current CRC DA reads processed primary canonical matrix; canonical_matrix iterates only over shared_space canonical_feature_name. |
| CRC       | Yachida  | CRC vs Healthy |                    450 |                      445 |                        419 |                        432 |                                   434 |                              426 |                          401 |                          413 |                            434 |                                   426 |                                               444 |                           110 |                      110 |                       2 |                        107 |                    107 | stage2/data_processed/crc_stage2_primary_metabolomics_canonical.csv built from crc_stage2_shared_feature_space via canonical_matrix() | True                        | Current CRC DA reads processed primary canonical matrix; canonical_matrix iterates only over shared_space canonical_feature_name. |

## Transformation and effect comparability

- Source scale: supplied aligned values in the Stage 2 primary canonical matrix.
- Current DA transformation: each metabolite is cohort-specifically z-standardised before regression.
- Formula: `metabolite_z ~ crc_status + age_years`.
- Robust SE method: HC3 robust OLS with t-distribution p-values.
- Missing values: complete-case within each univariate feature/design fit.
- FDR: BH within cohort in the current restricted DA table.
- Positive coefficient: higher metabolite abundance in CRC relative to control/healthy under the fitted model.

## Decision

CRC DA classification: **EXPANDED_COHORT_SPECIFIC_DA_RECOMMENDED**. The current 110-feature DA should be retained as harmonised sensitivity/replication-focused output, but not treated as the full biological discovery universe if a controlled rerun is approved.
