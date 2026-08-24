# Corrected-v2 Stage 4C Completion Report

Stage 4C was executed under Branch B as unsupervised/descriptive integration only.

Authoritative outputs are the files prefixed `stage4c_corrected_v2_` inside the corrected-v2 folder.

## Validation

| test                               | status   | notes                                                              |
|:-----------------------------------|:---------|:-------------------------------------------------------------------|
| input_integrity                    | pass     | rows=8                                                             |
| mofa_scores_written                | pass     | rows=4888                                                          |
| mofa_variance_written              | pass     | rows=72                                                            |
| mofa_factor_classification_written | pass     | rows=16                                                            |
| scca_summary_written               | pass     | rows=7                                                             |
| scca_reproducibility_written       | pass     | rows=6                                                             |
| no_subtype_predictive_tests        | pass     | only pooled IBD-vs-control and CRC-vs-control post-hoc labels used |

## View Dimensions And Transformations

| disease   | cohort   | view       |   n_participants |   n_features | transformation                                       | scaling_scope                                | disease_label_used   | clinical_variables_included   | finite_values   |
|:----------|:---------|:-----------|-----------------:|-------------:|:-----------------------------------------------------|:---------------------------------------------|:---------------------|:------------------------------|:----------------|
| IBD       | Franzosa | microbiome |              137 |           50 | closure_zero_handling_CLR_then_standard_scaling      | cohort_local_for_stage4c_structural_analysis | False                | False                         | True            |
| IBD       | Franzosa | metabolome |              137 |          166 | cohort_local_median_imputation_then_standard_scaling | cohort_local_for_stage4c_structural_analysis | False                | False                         | True            |
| IBD       | iHMP     | microbiome |               91 |           50 | closure_zero_handling_CLR_then_standard_scaling      | cohort_local_for_stage4c_structural_analysis | False                | False                         | True            |
| IBD       | iHMP     | metabolome |               91 |          166 | cohort_local_median_imputation_then_standard_scaling | cohort_local_for_stage4c_structural_analysis | False                | False                         | True            |
| CRC       | Sinha    | microbiome |              131 |           52 | closure_zero_handling_CLR_then_standard_scaling      | cohort_local_for_stage4c_structural_analysis | False                | False                         | True            |
| CRC       | Sinha    | metabolome |              131 |          109 | cohort_local_median_imputation_then_standard_scaling | cohort_local_for_stage4c_structural_analysis | False                | False                         | True            |
| CRC       | Yachida  | microbiome |              252 |           52 | closure_zero_handling_CLR_then_standard_scaling      | cohort_local_for_stage4c_structural_analysis | False                | False                         | True            |
| CRC       | Yachida  | metabolome |              252 |          109 | cohort_local_median_imputation_then_standard_scaling | cohort_local_for_stage4c_structural_analysis | False                | False                         | True            |

## MOFA Factor Classification

| disease   | factor   |   mean_metabolome_r2_percent |   mean_microbiome_r2_percent |   mean_total_r2_percent | factor_classification   | classification_rule                                                                              |
|:----------|:---------|-----------------------------:|-----------------------------:|------------------------:|:------------------------|:-------------------------------------------------------------------------------------------------|
| CRC       | Factor1  |                     15.5257  |                    1.53979   |                17.0655  | metabolome_dominant     | shared if each view contributes >= 20% of mean two-view R2; otherwise dominant view is larger R2 |
| CRC       | Factor2  |                      6.06726 |                    2.67842   |                 8.74568 | shared_multiview        | shared if each view contributes >= 20% of mean two-view R2; otherwise dominant view is larger R2 |
| CRC       | Factor3  |                      7.26855 |                    0.0101074 |                 7.27865 | metabolome_dominant     | shared if each view contributes >= 20% of mean two-view R2; otherwise dominant view is larger R2 |
| CRC       | Factor4  |                      6.15538 |                    0.558605  |                 6.71399 | metabolome_dominant     | shared if each view contributes >= 20% of mean two-view R2; otherwise dominant view is larger R2 |
| CRC       | Factor5  |                      0       |                    0         |                 0       | low_information         | shared if each view contributes >= 20% of mean two-view R2; otherwise dominant view is larger R2 |
| CRC       | Factor6  |                      0       |                    0         |                 0       | low_information         | shared if each view contributes >= 20% of mean two-view R2; otherwise dominant view is larger R2 |
| CRC       | Factor7  |                      0       |                    0         |                 0       | low_information         | shared if each view contributes >= 20% of mean two-view R2; otherwise dominant view is larger R2 |
| CRC       | Factor8  |                      0       |                    0         |                 0       | low_information         | shared if each view contributes >= 20% of mean two-view R2; otherwise dominant view is larger R2 |
| IBD       | Factor1  |                     10.268   |                    9.18465   |                19.4526  | shared_multiview        | shared if each view contributes >= 20% of mean two-view R2; otherwise dominant view is larger R2 |
| IBD       | Factor2  |                      8.74741 |                    1.15468   |                 9.90209 | metabolome_dominant     | shared if each view contributes >= 20% of mean two-view R2; otherwise dominant view is larger R2 |
| IBD       | Factor3  |                      4.37276 |                    2.98321   |                 7.35597 | shared_multiview        | shared if each view contributes >= 20% of mean two-view R2; otherwise dominant view is larger R2 |
| IBD       | Factor4  |                      3.60806 |                    2.2024    |                 5.81046 | shared_multiview        | shared if each view contributes >= 20% of mean two-view R2; otherwise dominant view is larger R2 |
| IBD       | Factor5  |                      2.71087 |                    0.323739  |                 3.03461 | metabolome_dominant     | shared if each view contributes >= 20% of mean two-view R2; otherwise dominant view is larger R2 |
| IBD       | Factor6  |                      0       |                    0         |                 0       | low_information         | shared if each view contributes >= 20% of mean two-view R2; otherwise dominant view is larger R2 |
| IBD       | Factor7  |                      0       |                    0         |                 0       | low_information         | shared if each view contributes >= 20% of mean two-view R2; otherwise dominant view is larger R2 |
| IBD       | Factor8  |                      0       |                    0         |                 0       | low_information         | shared if each view contributes >= 20% of mean two-view R2; otherwise dominant view is larger R2 |

## sCCA Component Summary

| disease   | cohort   |   component |   apparent_full_data_canonical_correlation |   n_samples |   microbiome_features |   metabolome_features |   penaltyx |   penaltyz | pma_function   | phenotype_used_for_fit   |
|:----------|:---------|------------:|-------------------------------------------:|------------:|----------------------:|----------------------:|-----------:|-----------:|:---------------|:-------------------------|
| IBD       | Franzosa |           1 |                                   0.831511 |         137 |                    50 |                   166 |       0.4  |       0.4  | PMA::CCA       | False                    |
| IBD       | iHMP     |           1 |                                   0.78639  |          91 |                    50 |                   166 |       0.4  |       0.4  | PMA::CCA       | False                    |
| IBD       | iHMP     |           2 |                                   0.746078 |          91 |                    50 |                   166 |       0.4  |       0.4  | PMA::CCA       | False                    |
| CRC       | Sinha    |           1 |                                   0.524271 |         131 |                    52 |                   109 |       0.25 |       0.25 | PMA::CCA       | False                    |
| CRC       | Sinha    |           2 |                                   0.540311 |         131 |                    52 |                   109 |       0.25 |       0.25 | PMA::CCA       | False                    |
| CRC       | Yachida  |           1 |                                   0.707656 |         252 |                    52 |                   109 |       0.4  |       0.25 | PMA::CCA       | False                    |
| CRC       | Yachida  |           2 |                                   0.693315 |         252 |                    52 |                   109 |       0.4  |       0.25 | PMA::CCA       | False                    |

## sCCA Cross-Cohort Reproducibility

| disease   | cohort_a   | cohort_b   |   component_a |   component_b | matched_by                                          | view       |   n_common |   spearman |   cosine_abs |   sign_agreement_fraction |   selected_overlap_n |   selected_union_n |   selected_jaccard | component_matching_ambiguity   |
|:----------|:-----------|:-----------|--------------:|--------------:|:----------------------------------------------------|:-----------|-----------:|-----------:|-------------:|--------------------------:|---------------------:|-------------------:|-------------------:|:-------------------------------|
| IBD       | Franzosa   | iHMP       |             1 |             2 | maximum_combined_absolute_loading_cosine_similarity | microbiome |         50 |   0.36508  |    0.388732  |                  0.66     |                    4 |                 21 |          0.190476  | none_flagged                   |
| IBD       | Franzosa   | iHMP       |             1 |             2 | maximum_combined_absolute_loading_cosine_similarity | metabolome |        166 |   0.499821 |    0.525007  |                  0.76506  |                   19 |                 58 |          0.327586  | none_flagged                   |
| CRC       | Sinha      | Yachida    |             1 |             1 | maximum_combined_absolute_loading_cosine_similarity | microbiome |         52 |   0.127347 |    0.16942   |                  0.634615 |                    1 |                 20 |          0.05      | none_flagged                   |
| CRC       | Sinha      | Yachida    |             1 |             1 | maximum_combined_absolute_loading_cosine_similarity | metabolome |        109 |   0.150064 |    0.0926757 |                  0.844037 |                    2 |                 19 |          0.105263  | none_flagged                   |
| CRC       | Sinha      | Yachida    |             2 |             2 | maximum_combined_absolute_loading_cosine_similarity | microbiome |         52 |   0.357707 |    0.102321  |                  0.711538 |                    3 |                 18 |          0.166667  | none_flagged                   |
| CRC       | Sinha      | Yachida    |             2 |             2 | maximum_combined_absolute_loading_cosine_similarity | metabolome |        109 |  -0.132443 |    0.0258776 |                  0.807339 |                    2 |                 21 |          0.0952381 | none_flagged                   |
