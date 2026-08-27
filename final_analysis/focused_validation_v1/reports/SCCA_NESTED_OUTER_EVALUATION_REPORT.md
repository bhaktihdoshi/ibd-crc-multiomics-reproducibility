# sCCA Nested Outer Evaluation Report

Nested outer sCCA was run independently within each cohort using diagnosis-independent KFold splits. All imputation, zero/near-zero variance filtering, scaling and PMA fitting were fold-local.

## Summary

| disease   | cohort   |   component |   successful_outer_folds |   median_abs_outer_test_correlation |   iqr_abs_outer_test_correlation |   min_abs_outer_test_correlation |   max_abs_outer_test_correlation |   median_training_apparent_abs_correlation |   median_apparent_minus_test_optimism |
|:----------|:---------|------------:|-------------------------:|------------------------------------:|---------------------------------:|---------------------------------:|---------------------------------:|-------------------------------------------:|--------------------------------------:|
| CRC       | Sinha    |           1 |                        5 |                            0.224327 |                        0.145878  |                       0.00331642 |                         0.358646 |                                   0.521513 |                              0.263655 |
| CRC       | Sinha    |           2 |                        2 |                            0.352814 |                        0.0447953 |                       0.308019   |                         0.397609 |                                   0.574276 |                              0.221463 |
| CRC       | Yachida  |           1 |                        5 |                            0.609429 |                        0.0596499 |                       0.480949   |                         0.690895 |                                   0.70545  |                              0.108179 |
| CRC       | Yachida  |           2 |                        1 |                            0.451582 |                        0         |                       0.451582   |                         0.451582 |                                   0.610687 |                              0.159105 |
| IBD       | Franzosa |           1 |                        5 |                            0.711221 |                        0.0687801 |                       0.687397   |                         0.820895 |                                   0.840963 |                              0.121001 |
| IBD       | iHMP     |           1 |                        5 |                            0.626196 |                        0.147398  |                       0.250581   |                         0.771028 |                                   0.832069 |                              0.201004 |
| IBD       | iHMP     |           2 |                        1 |                            0.69651  |                        0         |                       0.69651    |                         0.69651  |                                   0.798522 |                              0.102012 |

## Parameter Selection

| disease   | cohort   |   n_components |   penaltyx |   penaltyz |   outer_folds |
|:----------|:---------|---------------:|-----------:|-----------:|--------------:|
| CRC       | Sinha    |              1 |       0.15 |       0.15 |             2 |
| CRC       | Sinha    |              1 |       0.25 |       0.15 |             1 |
| CRC       | Sinha    |              2 |       0.25 |       0.25 |             1 |
| CRC       | Sinha    |              2 |       0.4  |       0.15 |             1 |
| CRC       | Yachida  |              1 |       0.4  |       0.25 |             1 |
| CRC       | Yachida  |              1 |       0.4  |       0.4  |             3 |
| CRC       | Yachida  |              2 |       0.4  |       0.4  |             1 |
| IBD       | Franzosa |              1 |       0.25 |       0.4  |             1 |
| IBD       | Franzosa |              1 |       0.4  |       0.4  |             4 |
| IBD       | iHMP     |              1 |       0.4  |       0.4  |             4 |
| IBD       | iHMP     |              2 |       0.4  |       0.4  |             1 |

## Failures

No outer-fold failures.
