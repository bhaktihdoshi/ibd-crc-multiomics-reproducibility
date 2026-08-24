# Stage 4B v2c Final Interpretation Freeze

This file freezes the interpretation of corrected-v2b Stage 4B after the bounded v2c common-scale metabolomics feasibility audit.

Final feasibility decision: **NO DEFENSIBLE COMMON-SCALE METABOLOMICS MATRIX AVAILABLE LOCALLY.**

Adopted branch: **B. Freeze v2b as the as-supplied cross-platform stress-test run and restructure Stage 4C-E accordingly.**

No historical corrected-v2 or v2b outputs are overwritten by this freeze.

## Preserved Outputs

- `tables/stage4b_corrected_v2b_external_performance.csv`
- `tables/stage4b_corrected_v2b_paired_model_comparisons.csv`
- `tables/stage4b_corrected_v2b_probability_diagnostics.csv`
- `tables/stage4b_corrected_v2b_transformation_domain_shift_diagnostics.csv`
- `reports/STAGE4B_V2C_FINAL_METABOLOMICS_FEASIBILITY_DECISION.md`

## Interpretation Rules

| rule_id                                         | interpretation                                                                                                      | qualification                                                                                       |
|:------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------|
| stage4b_principal_transportable_analysis        | Microbiome-only reciprocal validation is the principal directly transportable molecular analysis.                   | Retain the Sinha 16S-versus-Yachida shotgun limitation.                                             |
| metabolome_stress_test                          | Metabolome-only transfer is an as-supplied-scale stress test.                                                       | Paired cohorts lack verified common metabolomics measurement semantics and units.                   |
| early_integration_stress_test                   | Early integration is also an as-supplied-scale stress test.                                                         | It inherits the metabolomics incompatibility.                                                       |
| decision_score_auc                              | Decision-score ROC-AUC measures ranking under domain shift.                                                         | It is not evidence of deployable calibrated probability transport.                                  |
| classification_and_calibration_not_claimed      | Saturated/compressed probabilities and one-class predictions prevent clinical classification or calibration claims. | Thresholded outputs are diagnostic only when transfer probabilities fail.                           |
| high_auc_not_operational_transportability       | High AUC must not be described as operational transportability where thresholds or probabilities fail.              | AUC and deployable clinical classification answer different questions.                              |
| crc_biology_not_intrinsically_less_reproducible | No claim should be made that CRC biology is intrinsically less reproducible.                                        | Observed performance reflects these cohorts, platforms, feature spaces and stress-test assumptions. |
| paired_comparisons_conditional                  | Paired comparisons remain conditional on the stress-test representation.                                            | They should not be used as proof of common-scale metabolomics transport.                            |

## Restructured Stage Dependency Map

| stage   | artifact_scope                      | status                                              | interpretation_constraint                                                            |
|:--------|:------------------------------------|:----------------------------------------------------|:-------------------------------------------------------------------------------------|
| 4B      | microbiome_only_external_transfer   | principal_transportable_molecular_analysis          | retain Sinha 16S versus Yachida shotgun caveat                                       |
| 4B      | metabolome_only_external_transfer   | stress_test_only                                    | do not present as common-scale external metabolomics validation                      |
| 4B      | early_integration_external_transfer | stress_test_only                                    | inherits metabolomics measurement-scale incompatibility                              |
| 4C_MOFA | latent_factor_analysis              | descriptive_structural_analysis_to_be_rerun         | cohort-local preprocessing; no external projection; pooled phenotype annotation only |
| 4C_sCCA | cross_view_association              | independent_cohort_fits_to_be_rerun                 | held-out correlations separate from final refit; no phenotype in fitting/tuning      |
| 4D      | prior_graph                         | three_modes_to_be_constructed                       | identity, KEGG-only, KEGG+NJC19; unsigned undirected binary graph                    |
| 4E      | GCN                                 | primary_within_cohort_topology_evaluation_to_be_run | external mode secondary stress test only                                             |

## Figure Policy

| figure_intent                                           | required_label                                                                      | generate_now   |
|:--------------------------------------------------------|:------------------------------------------------------------------------------------|:---------------|
| External Stage 4B ROC curves/forest estimates           | bootstrap CIs; microbiome principal, metabolome/integration stress-test where shown | False          |
| Paired early-integration delta AUC forest               | stress-test comparisons                                                             | False          |
| MOFA variance heatmap and factor-score distributions    | cohort-by-view structural analysis                                                  | False          |
| sCCA held-out distributions and loading reproducibility | independently fitted cohort-local sCCA                                              | False          |
| Identity to KEGG to KEGG+NJC19 connectivity             | connectivity not biological activity                                                | False          |
| GCN paired fold/seed plots                              | within-cohort topology ablation                                                     | False          |
| External probability distributions                      | saturation/compression diagnostics                                                  | False          |
| Cross-method feature-evidence matrix                    | evidence synthesis, not causal proof                                                | False          |

## Thesis Wording

- Metabolite identifiers could be harmonised, but assay measurement scales could not be verified as quantitatively harmonised.
- Identifier harmonisation is not equivalent to quantitative harmonisation.
- Source-fitted scaling was leakage-safe, but it cannot repair incompatible upstream LC-MS response scales.
- v2b metabolome-only and early-integration external results should be described as stress tests under domain shift.
- Microbiome-only reciprocal validation remains the principal directly transportable molecular Stage 4B analysis, with the Sinha/Yachida platform caveat.
- Corrected numerical Results should not be rewritten until restructured Stage 4C-E analyses have been run by the user.
