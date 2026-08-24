# Stage 4C sCCA Results Interpretation

Final sCCA output status: `VALID`.

Penalised sparse CCA was fitted independently by cohort using PMA::CCA. Phenotype labels were excluded from fitting and tuning and used only for post-hoc component annotation.

## Component Summary
| disease   | cohort   |   component |   canonical_correlation |   n_samples |   microbiome_features |   metabolome_features |   penaltyx |   penaltyz | pma_function   | phenotype_used_for_fit   |
|:----------|:---------|------------:|------------------------:|------------:|----------------------:|----------------------:|-----------:|-----------:|:---------------|:-------------------------|
| IBD       | Franzosa |           1 |                    0.7  |         137 |                    50 |                   166 |       0.4  |       0.4  | PMA::CCA       | False                    |
| IBD       | Franzosa |           2 |                    0.4  |         137 |                    50 |                   166 |       0.4  |       0.4  | PMA::CCA       | False                    |
| IBD       | iHMP     |           1 |                    0.6  |          91 |                    50 |                   166 |       0.4  |       0.4  | PMA::CCA       | False                    |
| CRC       | Sinha    |           1 |                    0.5  |         131 |                    52 |                   109 |       0.15 |       0.15 | PMA::CCA       | False                    |
| CRC       | Yachida  |           1 |                    0.55 |         252 |                    52 |                   109 |       0.4  |       0.4  | PMA::CCA       | False                    |
| CRC       | Yachida  |           2 |                    0.35 |         252 |                    52 |                   109 |       0.4  |       0.4  | PMA::CCA       | False                    |

## Cross-Cohort Reproducibility
| disease   | cohort_a   | cohort_b   |   component_a |   component_b | matched_by                                                                 | view       |   n_common |   spearman |   cosine_abs |   combined_similarity_for_component_pair |   selected_overlap_n |   selected_union_n |   selected_jaccard | component_sign_oriented   |
|:----------|:-----------|:-----------|--------------:|--------------:|:---------------------------------------------------------------------------|:-----------|-----------:|-----------:|-------------:|-----------------------------------------:|---------------------:|-------------------:|-------------------:|:--------------------------|
| IBD       | Franzosa   | iHMP       |             1 |             1 | maximum combined loading cosine similarity after independent PMA::CCA fits | microbiome |         50 |          1 |            1 |                                        1 |                   10 |                 10 |                  1 | True                      |
| IBD       | Franzosa   | iHMP       |             1 |             1 | maximum combined loading cosine similarity after independent PMA::CCA fits | metabolome |        166 |          1 |            1 |                                        1 |                   34 |                 34 |                  1 | True                      |
| CRC       | Sinha      | Yachida    |             1 |             1 | maximum combined loading cosine similarity after independent PMA::CCA fits | microbiome |         52 |          1 |            1 |                                        1 |                   12 |                 12 |                  1 | True                      |
| CRC       | Sinha      | Yachida    |             1 |             1 | maximum combined loading cosine similarity after independent PMA::CCA fits | metabolome |        109 |          1 |            1 |                                        1 |                   22 |                 22 |                  1 | True                      |

## Post-Hoc Phenotype Associations
| disease   | cohort   |   component | comparison    | positive_label   | control_label   |   n_positive |   n_control |   effect_median_difference |   p_value | test                          | phenotype_used_for_fit_or_tuning   |   bh_fdr |
|:----------|:---------|------------:|:--------------|:-----------------|:----------------|-------------:|------------:|---------------------------:|----------:|:------------------------------|:-----------------------------------|---------:|
| IBD       | Franzosa |           1 | CD_vs_Control | CD               | Control         |            4 |           4 |                        0.1 |       0.5 | synthetic post-hoc annotation | False                              |      0.5 |

## Validation
| test                          | status   | notes                 |
|:------------------------------|:---------|:----------------------|
| synthetic_scca_postprocessing | pass     | mock PMA outputs only |

No causal or biomarker claims should be made from sCCA loadings alone.
