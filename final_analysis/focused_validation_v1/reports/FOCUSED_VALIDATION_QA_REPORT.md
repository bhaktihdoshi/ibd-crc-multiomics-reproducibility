# Focused Validation QA Report

| test                                  | status   | observed                             | expected                               |
|:--------------------------------------|:---------|:-------------------------------------|:---------------------------------------|
| recomputed_12_external_roc_auc_values | pass     | 5.551115123125783e-17                | <=1e-6                                 |
| no_cd_uc_predictive_tasks             | pass     | ['CRC_vs_Control', 'IBD_vs_Control'] | IBD_vs_Control and CRC_vs_Control only |
| eight_h3_comparisons                  | pass     | 8                                    | 8                                      |
| decision_scores_used_for_ranking      | pass     | decision_score                       | decision_score                         |
| brier_score_from_probability          | pass     | brier_score column                   | present                                |
| frozen_thresholds_present             | pass     | nonmissing                           | true                                   |
| figure_values_have_source_manifest    | pass     | exists                               | exists                                 |
| authoritative_hashes_unchanged        | pass     | all matched                          | all matched                            |
| mofa_sensitivity_tests                | pass     | {'pass': 6}                          | no failures                            |
| mofa_sensitivity_60_records           | pass     | 60                                   | 60                                     |
| mofa_postrun_audit_tests              | pass     | {'pass': 10}                         | no failures                            |
| metabolite_robustness_tests           | pass     | {'pass': 8}                          | no failures                            |
| rank_metabolomics_tests               | pass     | {'pass': 21}                         | no failures                            |
| scca_stability_tests                  | pass     | {'pass': 23}                         | no failures                            |
| scca_nested_tests                     | pass     | {'pass': 5}                          | no failures                            |
| scca_nested_preflight_tests           | pass     | {'pass': 14}                         | no failures                            |
| final_integrated_display_tests        | pass     | {'pass': 19}                         | no failures                            |
