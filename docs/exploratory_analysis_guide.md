# Exploratory Analysis Guide

See `exploratory/README.md` for the full warning and folder-by-folder description. This file adds a quick index of what negative/null results are preserved and where (per the instruction to keep scientifically important negative findings, not just positive ones):

| Negative/null finding | Where |
|---|---|
| Zero strict CRC metabolite replication (Sinha vs. Yachida) | `analyses/03_cross_cohort_replication/tables/stage3a_within_disease_concordance.csv` (CRC row) |
| Zero BH-FDR-significant cross-disease Spearman correlations (0/24) | `analyses/03_cross_cohort_replication/tables/stage3a_cross_disease_correlation_results.csv` |
| Zero directionally-coherent Stage3A+3B FDR candidates | `analyses/03_cross_cohort_replication/tables/stage3b_directionally_coherent_fdr_significant_candidates.csv` (genuinely empty, 0 rows, by construction) |
| Limited Yachida ordered-state overlap (only 2/56 HC3-significant, neither monotonic) | `analyses/03_cross_cohort_replication/tables/stage3b_yachida_ordered_state_trend_results.csv` |
| Weak/inconsistent CRC microbiome transfer | `final_analysis/tables/stage4b_corrected_v2b_external_performance.csv` (CRC rows, ROC-AUC ≈0.56–0.58) |
| CRC microbiome sensitivity showing representation dependence | `results/sensitivity/sensitivity_crc_microbiome_paired_comparisons_bh_fdr.csv` |
| GCN topology not clearly improving over identity/ridge comparators | `final_analysis/tables/final_stage4/table5_stage4e_paired_topology_findings.csv`, `supp_stage4e_primary_24_comparisons.csv` |
| No defensible common-scale metabolomics matrix | `final_analysis/code/audit_stage4b_v2c_common_scale_feasibility.py` and its output report |

These are treated as part of the thesis's evidentiary record, not omissions.
