# CRC Stage 2 Final Closure

## Status

**CRC_STAGE2_FROZEN_FOR_WRITEUP**

Stage 2 is frozen for thesis write-up. If a later Phase 3 analysis reveals evidence of a Stage 2 preprocessing or interpretation issue, any reopening must be explicitly documented rather than silently modifying frozen outputs.

## Primary Analysis Status

The primary CRC Stage 2 analysis is complete. The primary task remains CRC vs Control with bidirectional Sinha-to-Yachida and Yachida-to-Sinha external validation. No Phase 3 IBD-CRC integration has been started.

## Reporting Status

The reporting layer has been updated to include the final diagnostic findings. The corrected top-feature reporting output is authoritative for presentation/write-up.

## Diagnostic Status

Final diagnostic classification: **D. MIXED: reporting defect existed, but primary model probabilities and metrics are valid.**

The clinical top-feature reporting bug was real and affected mainly `Yachida -> Sinha | clinical | extra_trees`. Metabolite rows were incorrectly included/padded in the clinical top-feature display. This was reporting-only: model inputs, fitted models, predictions, thresholds and performance metrics were unaffected. The reporting filter was corrected, and clinical top-feature outputs now contain only clinical variables or encoded clinical variables.

## Model Validity Summary

- All six saved pipelines loaded successfully.
- Feature routing passed: clinical = clinical only; metabolomics = metabolomics only; integrated = both.
- Frozen preprocessing was applied correctly.
- Saved models reproduced stored external probabilities.
- Maximum absolute probability discrepancy was approximately `1.11e-16`.
- Positive-class orientation was verified: 0 = Control/Healthy, 1 = CRC, and `predict_proba[:, 1]` = CRC probability.

## Probability Pattern Interpretation

- `Sinha -> Yachida | clinical | ridge_logistic`: narrow external probabilities around 0.492-0.505 reflect weak clinical signal, strong ridge shrinkage/small coefficients and near-zero linear predictors, not a preprocessing defect.
- `Sinha -> Yachida | integrated | ridge_logistic`: saturated external probabilities near 0 and 1 reflect extreme external transformed metabolite values under the frozen Sinha-trained scaler and substantial cohort/domain shift. This was not a feature-order defect, class-orientation defect or plotting artifact.
- External transformed features are not expected to have mean 0 and SD 1 because the scaler was fitted on the training cohort.

## Internal Instability

Sinha-trained metabolomics/integrated models already showed substantial outer-fold variability internally, supporting model instability/cohort sensitivity and consistency with weak external transportability. Low external AUC is not interpreted as a software failure.

## Final Selected Models

| Train -> External | Feature set | Model | Internal ROC-AUC | External ROC-AUC | External PR-AUC | Threshold collapse |
|---|---|---|---:|---:|---:|---|
| Sinha -> Yachida | clinical | ridge_logistic | 0.559 | 0.427 | 0.474 | False |
| Sinha -> Yachida | integrated | ridge_logistic | 0.683 | 0.505 | 0.552 | False |
| Sinha -> Yachida | metabolomics | ridge_logistic | 0.661 | 0.478 | 0.531 | False |
| Yachida -> Sinha | clinical | extra_trees | 0.562 | 0.441 | 0.307 | False |
| Yachida -> Sinha | integrated | ridge_logistic | 0.700 | 0.380 | 0.292 | False |
| Yachida -> Sinha | metabolomics | ridge_logistic | 0.707 | 0.413 | 0.296 | True |

## Authoritative Final Output Tables

See `reports/CRC_STAGE2_AUTHORITATIVE_OUTPUT_MANIFEST.csv`. Key authoritative tables include:

- `tables/main_text/crc_stage2_selected_model_performance_summary.csv`
- `predictions/crc_stage2_external_predictions.csv`
- `tables/crc_stage2_threshold_selection.csv`
- `tables/crc_stage2_external_calibration_assessment.csv`
- `tables/main_text/crc_stage2_final_model_summary.csv`
- `tables/crc_stage2_differential_abundance_results.csv`
- `tables/crc_stage2_pathway_enrichment_status.csv`
- `tables/main_text/crc_stage2_top_predictive_features.csv`
- diagnostic audit tables listed in the authoritative manifest

## Authoritative Final Figures

Authoritative figures are listed in `reports/CRC_STAGE2_AUTHORITATIVE_OUTPUT_MANIFEST.csv` and indexed in `tables/main_text/crc_stage2_figure_index.csv`. Use corrected `crc_stage2_top_features_*` figures for feature interpretation. Earlier basic ROC/PR figures without CI are retained as historical/non-authoritative when CI versions exist.

## Provenance Hashes

- `tables/main_text/crc_stage2_selected_model_performance_summary.csv`: `80efc5f42b28ab1d7854914cdafcd4c249cb7cbe19c336e99e05298571556f10`
- `predictions/crc_stage2_external_predictions.csv`: `9ad1bca10164a878229b7dbead42f63fb77b000d3a8f1c1ceea763c75dc05a44`
- `tables/crc_stage2_threshold_selection.csv`: `fd76ee9537d1ffebe266fce645829167f7bb2dd055d97b2c50eecf206f4c3112`
- `tables/main_text/crc_stage2_top_predictive_features.csv`: `2ad793b670c8f7986b39bec81ba8319f08f3ac8c78c97f7725416137be33a2f7`
- `reports/CRC_STAGE2_EXTERNAL_PERFORMANCE_DIAGNOSTIC.md`: `aa2d71d0d88330e3a8742c4ac4ef570a359392f0b8c54375f15fa995e7f2bc69`

## Phase 3 Not Started

This closure pass did not compute the four-cohort metabolite intersection, compare IBD and CRC effect sizes, build a shared metabolite matrix, perform PCA/UMAP, build kNN graphs, run Leiden/Louvain, perform disease-trajectory integration, or modify IBD outputs.
