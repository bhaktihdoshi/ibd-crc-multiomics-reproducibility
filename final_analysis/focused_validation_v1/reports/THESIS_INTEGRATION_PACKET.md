# Thesis Integration Packet

Final focused-validation status: `COMPLETE_ADDITIVE_VALIDATION_READY_FOR_THESIS_INTEGRATION`.
No new modelling was run to create this packet; values are copied from completed focused-validation outputs.

## 1. Exact H3 Paired Results
| disease   | transfer_direction   | comparator   |   early_integration_roc_auc |   comparator_roc_auc |   delta_auc_early_minus_comparator |   paired_bootstrap_ci_low |   paired_bootstrap_ci_high |   raw_p_value |   bh_adjusted_q_value | final_interpretation   | stored_contrast                    |   valid_bootstrap_n |
|:----------|:---------------------|:-------------|----------------------------:|---------------------:|-----------------------------------:|--------------------------:|---------------------------:|--------------:|----------------------:|:-----------------------|:-----------------------------------|--------------------:|
| CRC       | Sinha→Yachida        | MICROBIOME   |                    0.50948  |             0.562961 |                        -0.0534803  |               -0.141865   |                 0.0393102  |     0.249875  |              0.333167 | inconclusive           | EARLY_INTEGRATION_minus_MICROBIOME |                2000 |
| CRC       | Sinha→Yachida        | METABOLOME   |                    0.50948  |             0.505512 |                         0.0039685  |               -0.00100787 |                 0.00907087 |     0.129935  |              0.207896 | inconclusive           | EARLY_INTEGRATION_minus_METABOLOME |                2000 |
| CRC       | Yachida→Sinha        | MICROBIOME   |                    0.588818 |             0.581862 |                         0.00695559 |               -0.0722445  |                 0.0901552  |     0.868566  |              0.868566 | inconclusive           | EARLY_INTEGRATION_minus_MICROBIOME |                2000 |
| CRC       | Yachida→Sinha        | METABOLOME   |                    0.588818 |             0.411985 |                         0.176833   |                0.0299559  |                 0.322659   |     0.017991  |              0.047976 | significantly higher   | EARLY_INTEGRATION_minus_METABOLOME |                2000 |
| IBD       | Franzosa→iHMP        | MICROBIOME   |                    0.742604 |             0.656805 |                         0.0857988  |               -0.0272337  |                 0.201198   |     0.122939  |              0.207896 | inconclusive           | EARLY_INTEGRATION_minus_MICROBIOME |                2000 |
| IBD       | Franzosa→iHMP        | METABOLOME   |                    0.742604 |             0.737278 |                         0.00532544 |               -0.0124408  |                 0.0248521  |     0.62069   |              0.70936  | inconclusive           | EARLY_INTEGRATION_minus_METABOLOME |                2000 |
| IBD       | iHMP→Franzosa        | MICROBIOME   |                    0.386905 |             0.540564 |                        -0.15366    |               -0.206349   |                -0.102728   |     0.0009995 |              0.003998 | significantly lower    | EARLY_INTEGRATION_minus_MICROBIOME |                2000 |
| IBD       | iHMP→Franzosa        | METABOLOME   |                    0.386905 |             0.845899 |                        -0.458995   |               -0.574735   |                -0.33421    |     0.0009995 |              0.003998 | significantly lower    | EARLY_INTEGRATION_minus_METABOLOME |                2000 |

BH-significant iHMP→Franzosa contrasts:
| disease   | transfer_direction   | comparator   |   early_integration_roc_auc |   comparator_roc_auc |   delta_auc_early_minus_comparator |   paired_bootstrap_ci_low |   paired_bootstrap_ci_high |   raw_p_value |   bh_adjusted_q_value | final_interpretation   | stored_contrast                    |   valid_bootstrap_n |
|:----------|:---------------------|:-------------|----------------------------:|---------------------:|-----------------------------------:|--------------------------:|---------------------------:|--------------:|----------------------:|:-----------------------|:-----------------------------------|--------------------:|
| IBD       | iHMP→Franzosa        | MICROBIOME   |                    0.386905 |             0.540564 |                          -0.15366  |                 -0.206349 |                  -0.102728 |     0.0009995 |              0.003998 | significantly lower    | EARLY_INTEGRATION_minus_MICROBIOME |                2000 |
| IBD       | iHMP→Franzosa        | METABOLOME   |                    0.386905 |             0.845899 |                          -0.458995 |                 -0.574735 |                  -0.33421  |     0.0009995 |              0.003998 | significantly lower    | EARLY_INTEGRATION_minus_METABOLOME |                2000 |

## 2. Final Main Table 2
Source: `tables/thesis_table2_complete_external_metrics.csv`; display: `tables/thesis_table2_display.csv`.
PR-AUC depends on external-cohort prevalence. Brier score is probability prediction error; it is not threshold-dependent and is not a complete calibration assessment.
| Disease   | Transfer      | Representation    | External ROC-AUC (95% CI)   |   PR-AUC |    MCC |   Sensitivity |   Specificity |   Brier score |   Frozen source threshold |
|:----------|:--------------|:------------------|:----------------------------|---------:|-------:|--------------:|--------------:|--------------:|--------------------------:|
| CRC       | Sinha→Yachida | EARLY_INTEGRATION | 0.509 (0.439-0.584)         |    0.53  |  0.024 |         0.536 |         0.488 |         0.486 |                    -0.028 |
| CRC       | Sinha→Yachida | METABOLOME        | 0.506 (0.435-0.581)         |    0.521 |  0.04  |         0.536 |         0.504 |         0.48  |                     0.255 |
| CRC       | Sinha→Yachida | MICROBIOME        | 0.563 (0.489-0.631)         |    0.546 |  0.051 |         0.984 |         0.031 |         0.26  |                    -0.073 |
| CRC       | Yachida→Sinha | EARLY_INTEGRATION | 0.589 (0.474-0.699)         |    0.462 |  0.001 |         0.81  |         0.191 |         0.549 |                     1.181 |
| CRC       | Yachida→Sinha | METABOLOME        | 0.412 (0.307-0.522)         |    0.299 | -0.051 |         0.286 |         0.663 |         0.253 |                     0.03  |
| CRC       | Yachida→Sinha | MICROBIOME        | 0.582 (0.473-0.697)         |    0.46  |  0.018 |         0.952 |         0.056 |         0.371 |                    -0.276 |
| IBD       | Franzosa→iHMP | EARLY_INTEGRATION | 0.743 (0.631-0.847)         |    0.869 |  0.371 |         0.677 |         0.731 |         0.308 |                    -0.444 |
| IBD       | Franzosa→iHMP | METABOLOME        | 0.737 (0.625-0.844)         |    0.868 |  0.298 |         0.738 |         0.577 |         0.308 |                    -0.14  |
| IBD       | Franzosa→iHMP | MICROBIOME        | 0.657 (0.526-0.780)         |    0.793 |  0.089 |         0.877 |         0.192 |         0.201 |                     0.057 |
| IBD       | iHMP→Franzosa | EARLY_INTEGRATION | 0.387 (0.287-0.487)         |    0.534 | -0.217 |         0.21  |         0.589 |         0.254 |                     0.314 |
| IBD       | iHMP→Franzosa | METABOLOME        | 0.846 (0.775-0.905)         |    0.907 |  0     |         0     |         1     |         0.245 |                     0.131 |
| IBD       | iHMP→Franzosa | MICROBIOME        | 0.541 (0.439-0.642)         |    0.654 |  0.013 |         0.531 |         0.482 |         0.247 |                     0.027 |

## 3. Rank-Metabolomics Exact Results
| disease   | transfer      | comparison                                                         |   original_auc |   rank_auc |   delta_rank_minus_original_auc |     ci_low |   ci_high |         p |        q | interpretation   | wording_boundary                                                                                              |
|:----------|:--------------|:-------------------------------------------------------------------|---------------:|-----------:|--------------------------------:|-----------:|----------:|----------:|---------:|:-----------------|:--------------------------------------------------------------------------------------------------------------|
| IBD       | Franzosa→iHMP | RANK_METABOLOME_minus_original_METABOLOME                          |       0.737278 |   0.716568 |                      -0.0207101 | -0.17574   | 0.127219  | 0.7996    | 0.7996   | inconclusive     | representation-dependent; does not prove original failure was caused by cross-assay numerical incomparability |
| IBD       | Franzosa→iHMP | EARLY_INTEGRATION_RANK_METABOLOME_minus_original_EARLY_INTEGRATION |       0.742604 |   0.767456 |                       0.0248521 | -0.0633284 | 0.110074  | 0.557721  | 0.743628 | inconclusive     | representation-dependent; does not prove original failure was caused by cross-assay numerical incomparability |
| IBD       | iHMP→Franzosa | RANK_METABOLOME_minus_original_METABOLOME                          |       0.845899 |   0.857584 |                       0.0116843 | -0.0498347 | 0.0791501 | 0.727636  | 0.7996   | inconclusive     | representation-dependent; does not prove original failure was caused by cross-assay numerical incomparability |
| IBD       | iHMP→Franzosa | EARLY_INTEGRATION_RANK_METABOLOME_minus_original_EARLY_INTEGRATION |       0.386905 |   0.832451 |                       0.445547  |  0.338619  | 0.546522  | 0.0009995 | 0.007996 | improved         | representation-dependent; does not prove original failure was caused by cross-assay numerical incomparability |
| CRC       | Sinha→Yachida | RANK_METABOLOME_minus_original_METABOLOME                          |       0.505512 |   0.567685 |                       0.0621732 | -0.0225559 | 0.142049  | 0.141929  | 0.283858 | inconclusive     | representation-dependent; does not prove original failure was caused by cross-assay numerical incomparability |
| CRC       | Sinha→Yachida | EARLY_INTEGRATION_RANK_METABOLOME_minus_original_EARLY_INTEGRATION |       0.50948  |   0.580535 |                       0.0710551 | -0.0132315 | 0.150617  | 0.093953  | 0.250541 | inconclusive     | representation-dependent; does not prove original failure was caused by cross-assay numerical incomparability |
| CRC       | Yachida→Sinha | RANK_METABOLOME_minus_original_METABOLOME                          |       0.411985 |   0.515249 |                       0.103264  | -0.0101726 | 0.219643  | 0.0709645 | 0.250541 | inconclusive     | representation-dependent; does not prove original failure was caused by cross-assay numerical incomparability |
| CRC       | Yachida→Sinha | EARLY_INTEGRATION_RANK_METABOLOME_minus_original_EARLY_INTEGRATION |       0.588818 |   0.531568 |                      -0.0572499 | -0.18219   | 0.0679642 | 0.368816  | 0.590105 | inconclusive     | representation-dependent; does not prove original failure was caused by cross-assay numerical incomparability |

Exact representation-dependent iHMP→Franzosa early-integration rank contrast: delta=0.445546737213, CI=(0.338618827160, 0.546522266314), p=0.000999500250, q=0.007996001999.

## 4. MOFA Final Thesis Values
Primary shared factors: IBD=3; CRC=1.
K values tested: [5, 8, 10]; seeds=10; thresholds=[0.15, 0.2, 0.25, 0.3]; successful configurations=58; paired IBD/CRC configurations=30; IBD>CRC matched count=30 (1.000).
Primary IBD Factor 3 top absolute metabolome loadings:
| canonical_display_name   | HMDB_identifier   | KEGG_identifier   |   signed_loading |   absolute_loading |   within_metabolome_view_rank |   rank_across_all_Factor3_features |
|:-------------------------|:------------------|:------------------|-----------------:|-------------------:|------------------------------:|-----------------------------------:|
| N-acetylputrescine       | HMDB0002064       | C02714            |        -0.429478 |           0.429478 |                             1 |                                  1 |
| eicosatrienoic acid      | HMDB0002925       | C03242            |        -0.422184 |           0.422184 |                             2 |                                  2 |
| putrescine               | HMDB0001414       | C00134            |        -0.384912 |           0.384912 |                             3 |                                  3 |
Global shared-factor structure was highly stable. Individual factor identity/alignment was less stable. Primary Factor 3 overlap is internal convergence, not independent feature-level replication across every MOFA configuration.

## 5. Metabolite Robustness Final Values
Seven replicated metabolites: N-acetylputrescine, adrenic acid, arachidonic acid, eicosatrienoic acid, ketodeoxycholate, putrescine, taurine.
Cohort-metabolite combinations=14; minimum sign retention=100.0%; sign reversals=0; nominal p<0.05 status changes=0; magnitude classes={'direction_stable_and_magnitude_stable': 14}.
Complete-family BH-FDR was not recomputed under each omission; report is limited to coefficient, interval and raw-p-value stability.

## 6. sCCA Final Thesis Values
A. Apparent full-data correlations:
| disease   | cohort   |   component |   apparent_full_data_canonical_correlation |
|:----------|:---------|------------:|-------------------------------------------:|
| IBD       | Franzosa |           1 |                                   0.831511 |
| IBD       | iHMP     |           1 |                                   0.78639  |
| IBD       | iHMP     |           2 |                                   0.746078 |
| CRC       | Sinha    |           1 |                                   0.524271 |
| CRC       | Sinha    |           2 |                                   0.540311 |
| CRC       | Yachida  |           1 |                                   0.707656 |
| CRC       | Yachida  |           2 |                                   0.693315 |
B. Cross-cohort loading similarity / selected-set overlap summaries:
| disease   |   mean_bootstrap_combined_cosine |
|:----------|---------------------------------:|
| CRC       |                         0.535709 |
| IBD       |                         0.686924 |
| disease   | view       |   mean_median_jaccard |
|:----------|:-----------|----------------------:|
| CRC       | metabolome |              0.269219 |
| CRC       | microbiome |              0.396642 |
| IBD       | metabolome |              0.523774 |
| IBD       | microbiome |              0.483069 |
Best IBD component match from nested alignment:
|   fitted_component |   primary_component |   microbial_cosine |   metabolite_cosine |   combined_absolute_cosine |   sign_orientation |   best_minus_second_best_margin | status        | disease   | cohort   |   outer_fold |
|-------------------:|--------------------:|-------------------:|--------------------:|---------------------------:|-------------------:|--------------------------------:|:--------------|:----------|:---------|-------------:|
|                  1 |                   1 |           0.962344 |            0.964508 |                   0.963426 |                  1 |                             nan | matched_valid | IBD       | Franzosa |            2 |
C. Bootstrap component recovery:
| disease   | cohort   |   primary_component |   requested_bootstrap_n |   successful_model_fits |   valid_aligned_component_count |   primary_component_recovery_frequency_all_requested |   primary_component_recovery_frequency_successful_fits |   median_combined_cosine_similarity |
|:----------|:---------|--------------------:|------------------------:|------------------------:|--------------------------------:|-----------------------------------------------------:|-------------------------------------------------------:|------------------------------------:|
| IBD       | Franzosa |                   1 |                     200 |                     200 |                             200 |                                                1     |                                                  1     |                            0.786537 |
| IBD       | iHMP     |                   1 |                     200 |                     200 |                             185 |                                                0.925 |                                                  0.925 |                            0.725188 |
| IBD       | iHMP     |                   2 |                     200 |                     200 |                              42 |                                                0.21  |                                                  0.21  |                            0.549045 |
| CRC       | Sinha    |                   1 |                     200 |                     200 |                             143 |                                                0.715 |                                                  0.715 |                            0.408554 |
| CRC       | Sinha    |                   2 |                     200 |                     200 |                              86 |                                                0.43  |                                                  0.43  |                            0.463016 |
| CRC       | Yachida  |                   1 |                     200 |                     200 |                             180 |                                                0.9   |                                                  0.9   |                            0.806775 |
| CRC       | Yachida  |                   2 |                     200 |                     200 |                              27 |                                                0.135 |                                                  0.135 |                            0.464491 |
D. Nested outer C1:
| disease   | cohort   |   component |   successful_outer_folds |   median_abs_outer_test_correlation |   iqr_abs_outer_test_correlation |   min_abs_outer_test_correlation |   max_abs_outer_test_correlation |   median_training_apparent_abs_correlation |   median_apparent_minus_test_optimism |
|:----------|:---------|------------:|-------------------------:|------------------------------------:|---------------------------------:|---------------------------------:|---------------------------------:|-------------------------------------------:|--------------------------------------:|
| CRC       | Sinha    |           1 |                        5 |                            0.224327 |                        0.145878  |                       0.00331642 |                         0.358646 |                                   0.521513 |                              0.263655 |
| CRC       | Yachida  |           1 |                        5 |                            0.609429 |                        0.0596499 |                       0.480949   |                         0.690895 |                                   0.70545  |                              0.108179 |
| IBD       | Franzosa |           1 |                        5 |                            0.711221 |                        0.0687801 |                       0.687397   |                         0.820895 |                                   0.840963 |                              0.121001 |
| IBD       | iHMP     |           1 |                        5 |                            0.626196 |                        0.147398  |                       0.250581   |                         0.771028 |                                   0.832069 |                              0.201004 |
Individual C1 outer-fold absolute correlations:
- CRC|Sinha: 0.003316, 0.247223, 0.101345, 0.358646, 0.224327
- CRC|Yachida: 0.609111, 0.609429, 0.690895, 0.668761, 0.480949
- IBD|Franzosa: 0.711221, 0.757799, 0.687397, 0.820895, 0.689019
- IBD|iHMP: 0.581778, 0.771028, 0.729177, 0.626196, 0.250581
C2 does not have comparable five-fold support and should not be presented as an equivalent nested summary.

## 7. Covariate Audit
Final decision: `NOT_FEASIBLE_AFTER_PRIVATE_METADATA_AUDIT`. Clinical covariates were insufficiently comparable and complete for defensible paired cross-cohort adjustment across the focused validation design.

## 8. Figure/Table Inventory
| display_id              | title                                          | placement     | source                                                                                              | role                                          |
|:------------------------|:-----------------------------------------------|:--------------|:----------------------------------------------------------------------------------------------------|:----------------------------------------------|
| Figure 1                | study design                                   | main          | planned thesis display                                                                              | study design overview                         |
| Figure 2                | participant eligibility/matching               | main          | planned thesis display                                                                              | cohort/sample eligibility                     |
| Table 1                 | cohorts and harmonised feature spaces          | main          | planned thesis display                                                                              | cohort and feature-space overview             |
| Figure 3                | metabolite association/replication/concordance | main          | planned thesis display                                                                              | frozen Stage 2/3 metabolite evidence          |
| Figure 4                | external transportability + ROC curves         | main          | final_analysis/focused_validation_v1/figures/figure4_external_transportability_composite.png        | frozen Stage 4B prediction transportability   |
| Table 2                 | complete external performance metrics          | main          | final_analysis/focused_validation_v1/tables/focused_table2_complete_external_metrics_display.csv    | complete frozen external prediction metrics   |
| Figure 5                | internal/external + threshold diagnostic       | main          | final_analysis/focused_validation_v1/figures/figure5_internal_external_threshold_diagnostic.png     | threshold/decision-score diagnostic           |
| Figure 6                | latent structure and stability                 | main          | final_analysis/focused_validation_v1/figures/figure6_latent_structure_and_stability.png             | MOFA and sCCA stability synthesis             |
| Supplementary Figure S1 | replicated-metabolite distributions            | supplementary | final_analysis/focused_validation_v1/figures/supp_figure_s1_replicated_metabolite_distributions.png | seven replicated IBD metabolite distributions |
| Supplementary Table S1  | software/versions/functions                    | supplementary | final_analysis/focused_validation_v1/manifests/baseline_integrity.json                              | software and provenance                       |
No imputation/KS Figure, CRC ordered-state display, graph/GCN display, Supplementary Table S2, or Supplementary Table S3 is required for the final focused thesis display plan.

## 9. Dangling-Reference Audit
| reference_to_check                                 | recommended_resolution                                                     | status                           |
|:---------------------------------------------------|:---------------------------------------------------------------------------|:---------------------------------|
| Supplementary Table S2                             | Remove unless independently created; not required in final display plan.   | requires_manual_manuscript_check |
| Supplementary Table S3                             | Remove if complete metrics are represented in main Table 2.                | requires_manual_manuscript_check |
| Figures 7 or 8                                     | Remove from focused thesis display plan.                                   | requires_manual_manuscript_check |
| ordered-state CRC analysis                         | Do not reference in focused final results.                                 | requires_manual_manuscript_check |
| graph/GCN                                          | Do not reference as a final focused-thesis figure/table.                   | requires_manual_manuscript_check |
| KS imputation validation                           | No imputation/KS figure is part of final analysis.                         | requires_manual_manuscript_check |
| prespecified sensitivity analyses                  | Use post-freeze/additive validation wording unless truly prespecified.     | requires_manual_manuscript_check |
| co-primary language                                | Avoid language implying prospective prespecification of additive analyses. | requires_manual_manuscript_check |
| causal attribution of rank-metabolomics            | Use representation-dependent; no causal explanation.                       | requires_manual_manuscript_check |
| sCCA correlations only apparent                    | Distinguish apparent full-data from nested outer C1 estimates.             | requires_manual_manuscript_check |
| C2 nested estimates comparable five-fold summaries | Do not present C2 as equivalent to C1; fold support is limited.            | requires_manual_manuscript_check |
