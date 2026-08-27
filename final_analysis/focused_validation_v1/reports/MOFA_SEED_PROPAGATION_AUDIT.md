# MOFA Seed Propagation Audit

Seed values were recorded and passed by the additive sensitivity runner. The saved outputs indicate whether different seeds produced distinct model hashes/loading matrices or deterministic-identical results under the current MOFA configuration.

| disease   |   K |   unique_model_hashes_for_disease_K |   unique_final_elbo_values_for_disease_K |   loading_matrix_l2_difference_vs_min_seed |
|:----------|----:|------------------------------------:|-----------------------------------------:|-------------------------------------------:|
| CRC       |   5 |                                  10 |                                        1 |                                          0 |
| CRC       |   8 |                                  10 |                                        1 |                                          0 |
| CRC       |  10 |                                  10 |                                        1 |                                          0 |
| IBD       |   5 |                                  10 |                                        1 |                                          0 |
| IBD       |   8 |                                  10 |                                        1 |                                          0 |
| IBD       |  10 |                                  10 |                                        1 |                                          0 |
