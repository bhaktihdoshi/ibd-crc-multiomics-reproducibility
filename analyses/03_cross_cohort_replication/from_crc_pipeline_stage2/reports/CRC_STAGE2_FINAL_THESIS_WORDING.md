# CRC Stage 2 Final Thesis Wording

## Methods / QC Note

A reporting-only defect affecting the Yachida-to-Sinha clinical Extra Trees feature-importance display was identified during post-run audit. The defect incorrectly included/padded metabolite rows in a clinical top-feature display. Model inputs, fitted pipelines, predictions, thresholds and performance estimates were unaffected, and the reporting output was corrected before final presentation. The corrected clinical top-feature outputs now contain only clinical variables or encoded clinical variables.

## Model Validity / Provenance Note

All six saved CRC Stage 2 pipelines loaded successfully. Feature routing passed for all selected contexts: clinical models routed only clinical variables, metabolomics models routed only metabolomics variables, and integrated models routed both intended branches. Frozen preprocessing was applied correctly, and saved pipelines reproduced the stored external predicted probabilities with a maximum absolute discrepancy of approximately 1.11e-16. Positive-class orientation was verified as 0 = Control/Healthy, 1 = CRC, with `predict_proba[:, 1]` representing CRC probability.

## Limitations / Interpretation

Moderate internal CRC discrimination did not generalise reliably between Sinha and Yachida. Post-run audit confirmed correct feature routing, positive-class orientation, frozen preprocessing and numerical reproduction of saved external predictions, supporting interpretation of the poor external performance as cohort/domain shift and model instability rather than implementation error.

The Sinha-to-Yachida clinical Ridge Logistic model produced narrow external probabilities around 0.492-0.505. This is interpreted as weak clinical signal with strong ridge shrinkage/small coefficients and near-zero linear predictors, not a preprocessing defect.

The Sinha-to-Yachida integrated Ridge Logistic model produced strongly saturated external probabilities near 0 and 1. This is interpreted as extreme external transformed metabolite values under the frozen Sinha-trained scaler and substantial cohort/domain shift, not a feature-order defect, class-orientation defect or plotting artifact. External transformed features are not expected to have mean 0 and SD 1 because the scaler was fitted on the training cohort.

Sinha-trained metabolomics and integrated models already showed substantial outer-fold variability internally, consistent with model instability/cohort sensitivity and weak external transportability. Predictive feature importance should be interpreted as predictive association only, not causal evidence or differential abundance.
