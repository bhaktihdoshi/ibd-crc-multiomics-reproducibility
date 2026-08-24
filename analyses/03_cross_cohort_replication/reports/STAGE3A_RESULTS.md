# Stage 3A Results

Shared metabolite space: 56 HMDB metabolites; primary complete-case set: 55.

## Within-Disease Concordance
| disease   | comparison                              | phenotype   |   n_complete_metabolites |   same_sign_count |   opposite_sign_count |   same_sign_proportion |   spearman_rho |   spearman_p_value |   pearson_r_secondary |   pearson_p_value_secondary |   strict_FDR_replication_events_full_expanded |   strict_FDR_unique_metabolites_full_expanded |   strict_FDR_replication_events_in_stage3_56 |   strict_FDR_unique_metabolites_in_stage3_56 | replication_count_note                                                                                                                                                                                   |
|:----------|:----------------------------------------|:------------|-------------------------:|------------------:|----------------------:|-----------------------:|---------------:|-------------------:|----------------------:|----------------------------:|----------------------------------------------:|----------------------------------------------:|---------------------------------------------:|---------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| IBD       | Franzosa vs iHMP IBD vs Control         | ibd         |                       55 |                41 |                    14 |               0.745455 |       0.782179 |        1.77593e-12 |              0.789111 |                 8.28484e-13 |                                             7 |                                             7 |                                            3 |                                            3 | Full expanded IBD replication count is reproduced from the validated post-reopening replication audit; Stage 3A-subset count is lower when replicated metabolites are outside the strict four-cohort 56. |
| IBD       | Franzosa vs iHMP CD vs Control          | cd          |                       55 |                43 |                    12 |               0.781818 |       0.744012 |        7.56682e-11 |              0.745427 |                 6.66312e-11 |                                             3 |                                             3 |                                            1 |                                            1 | Full expanded IBD replication count is reproduced from the validated post-reopening replication audit; Stage 3A-subset count is lower when replicated metabolites are outside the strict four-cohort 56. |
| IBD       | Franzosa vs iHMP UC vs Control          | uc          |                       55 |                40 |                    15 |               0.727273 |       0.776263 |        3.33189e-12 |              0.795152 |                 4.16361e-13 |                                             0 |                                             0 |                                            0 |                                            0 | Full expanded IBD replication count is reproduced from the validated post-reopening replication audit; Stage 3A-subset count is lower when replicated metabolites are outside the strict four-cohort 56. |
| CRC       | Sinha vs Yachida CRC vs Control/Healthy | crc         |                       56 |                30 |                    26 |               0.535714 |       0.134313 |        0.323683    |              0.189975 |                 0.1608      |                                             0 |                                             0 |                                            0 |                                            0 | CRC strict replication is 0 in the validated expanded DA audit and in the Stage 3A strict subset.                                                                                                        |

## Cross-Disease Spearman Correlations
| ibd_phenotype   | ibd_cohort   | crc_cohort   |   n | method   |    estimate |   p_value | interpretation                                                                    |   BH_FDR |
|:----------------|:-------------|:-------------|----:|:---------|------------:|----------:|:----------------------------------------------------------------------------------|---------:|
| ibd             | Franzosa     | Sinha        |  56 | Spearman |  0.192755   | 0.15465   | Primary rank-based cross-disease effect association; coefficients are not pooled. | 0.37116  |
| ibd             | Franzosa     | Yachida      |  56 | Spearman | -0.0164046  | 0.904483  | Primary rank-based cross-disease effect association; coefficients are not pooled. | 0.952556 |
| ibd             | iHMP         | Sinha        |  55 | Spearman |  0.290548   | 0.0314027 | Primary rank-based cross-disease effect association; coefficients are not pooled. | 0.125611 |
| ibd             | iHMP         | Yachida      |  55 | Spearman |  0.0685426  | 0.619023  | Primary rank-based cross-disease effect association; coefficients are not pooled. | 0.881839 |
| cd              | Franzosa     | Sinha        |  56 | Spearman |  0.312372   | 0.0190865 | Primary rank-based cross-disease effect association; coefficients are not pooled. | 0.125611 |
| cd              | Franzosa     | Yachida      |  56 | Spearman |  0.0462748  | 0.734866  | Primary rank-based cross-disease effect association; coefficients are not pooled. | 0.881839 |
| cd              | iHMP         | Sinha        |  55 | Spearman |  0.304834   | 0.0236395 | Primary rank-based cross-disease effect association; coefficients are not pooled. | 0.125611 |
| cd              | iHMP         | Yachida      |  55 | Spearman |  0.103175   | 0.4535    | Primary rank-based cross-disease effect association; coefficients are not pooled. | 0.777429 |
| uc              | Franzosa     | Sinha        |  56 | Spearman |  0.00813397 | 0.952556  | Primary rank-based cross-disease effect association; coefficients are not pooled. | 0.952556 |
| uc              | Franzosa     | Yachida      |  56 | Spearman | -0.13527    | 0.320216  | Primary rank-based cross-disease effect association; coefficients are not pooled. | 0.640431 |
| uc              | iHMP         | Sinha        |  55 | Spearman |  0.260173   | 0.0550657 | Primary rank-based cross-disease effect association; coefficients are not pooled. | 0.165197 |
| uc              | iHMP         | Yachida      |  55 | Spearman |  0.0558442  | 0.685511  | Primary rank-based cross-disease effect association; coefficients are not pooled. | 0.881839 |

BH-FDR-significant cross-disease Spearman correlations: 0.

## Directional Concordance Counts
| comparison   | direction_class                    |   n |
|:-------------|:-----------------------------------|----:|
| CD vs CRC    | BOTH_DISCORDANT                    |   6 |
| CD vs CRC    | CONCORDANT_NEGATIVE                |   5 |
| CD vs CRC    | CONCORDANT_POSITIVE                |  13 |
| CD vs CRC    | CRC_DISCORDANT                     |  19 |
| CD vs CRC    | IBD_DISCORDANT                     |   6 |
| CD vs CRC    | INCOMPLETE                         |   1 |
| CD vs CRC    | OPPOSITE_IBD_NEGATIVE_CRC_POSITIVE |   3 |
| CD vs CRC    | OPPOSITE_IBD_POSITIVE_CRC_NEGATIVE |   3 |
| IBD vs CRC   | BOTH_DISCORDANT                    |   4 |
| IBD vs CRC   | CONCORDANT_NEGATIVE                |   4 |
| IBD vs CRC   | CONCORDANT_POSITIVE                |   9 |
| IBD vs CRC   | CRC_DISCORDANT                     |  21 |
| IBD vs CRC   | IBD_DISCORDANT                     |  10 |
| IBD vs CRC   | INCOMPLETE                         |   1 |
| IBD vs CRC   | OPPOSITE_IBD_NEGATIVE_CRC_POSITIVE |   4 |
| IBD vs CRC   | OPPOSITE_IBD_POSITIVE_CRC_NEGATIVE |   3 |
| UC vs CRC    | BOTH_DISCORDANT                    |   4 |
| UC vs CRC    | CONCORDANT_NEGATIVE                |   5 |
| UC vs CRC    | CONCORDANT_POSITIVE                |   6 |
| UC vs CRC    | CRC_DISCORDANT                     |  21 |
| UC vs CRC    | IBD_DISCORDANT                     |  11 |
| UC vs CRC    | INCOMPLETE                         |   1 |
| UC vs CRC    | OPPOSITE_IBD_NEGATIVE_CRC_POSITIVE |   6 |
| UC vs CRC    | OPPOSITE_IBD_POSITIVE_CRC_NEGATIVE |   2 |

## Replication-Aware Interpretation
Directional concordance is not the same as cross-disease replication. CRC strict replication is 0, so no metabolite is described as a validated shared IBD-CRC biomarker.

## Pathway Context
IBD expanded ORA has 18 significant result rows representing 7 unique pathways, with 0 strict pathway replication. CRC expanded ORA has no defensible significant pathway result. Stage 3A uses pathway context descriptively only.
