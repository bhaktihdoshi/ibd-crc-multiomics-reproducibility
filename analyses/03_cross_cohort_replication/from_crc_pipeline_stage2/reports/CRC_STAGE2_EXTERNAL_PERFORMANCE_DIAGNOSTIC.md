# CRC Stage 2 External Performance Diagnostic

Final classification: **D. MIXED: reporting defect exists but primary model probabilities/metrics are valid**

## Diagnostic conclusions

1. Clinical feature-importance issue real: **True**. The source importance table contained zero-importance metabolite padding in the Yachida→Sinha clinical Extra Trees context.
2. Presentation top-feature table/plots currently contain clinical-context metabolite rows: **False**. The reporting filter was corrected and the top-feature table/plots were regenerated only from valid context features.
3. Saved pipeline routing critical defects: **False**. Clinical pipelines route only age/BMI/sex; metabolomics pipelines route only metabolite columns; integrated pipelines route both.
4. Frozen preprocessing applied: **True**. External transformed distributions are shifted in Sinha→Yachida metabolomics/integrated models, but no non-finite transformed values were found.
5. Saved models reproduce stored external probabilities: **True**. Maximum probability difference was 1.11e-16.
6. Positive-class orientation correct: **True**. All selected estimators have classes_ = [0, 1], with class 1 used as CRC probability.

## Specific probability patterns

- **Sinha→Yachida | clinical | ridge_logistic**: intercept -0.0000; L2 norm 0.0123; linear predictor -0.031 to 0.019; probability 0.492296 to 0.504780; AUC 0.427. The narrow 0.492–0.505 range is explained by tiny ridge coefficients and a near-zero linear predictor, consistent with strongly limited clinical signal rather than a scaling or label bug.
- **Sinha→Yachida | integrated | ridge_logistic**: intercept -0.0322; L2 norm 0.4130; linear predictor -15068.062 to 8935.192; probability 0.000000 to 1.000000; AUC 0.505. The near-0/1 saturation is explained by extreme external transformed metabolite values under the frozen Sinha-trained scaler, producing huge linear predictors; this is domain shift/saturation, not a feature-order or orientation defect.

## Internal stability

- Sinha→Yachida | clinical: outer ROC-AUC values 0.697531;0.477124;0.361111;0.500000;0.756944; mean 0.559, SD 0.164, min 0.361, max 0.757.
- Sinha→Yachida | integrated: outer ROC-AUC values 0.672840;0.712418;0.506944;0.881944;0.638889; mean 0.683, SD 0.136, min 0.507, max 0.882.
- Sinha→Yachida | metabolomics: outer ROC-AUC values 0.635802;0.712418;0.486111;0.833333;0.638889; mean 0.661, SD 0.127, min 0.486, max 0.833.

## Model-rerun decision

No CRC model rerun is required. The reporting defect affected feature-importance display/interpretation only, not primary fitted models, saved probabilities, thresholds, labels or external metrics.

## Generated audit files

- `tables/crc_stage2_feature_importance_context_audit.csv`
- `tables/crc_stage2_saved_pipeline_structure_audit.csv`
- `tables/crc_stage2_external_transformed_feature_audit.csv`
- `tables/crc_stage2_external_transformation_summary.csv`
- `tables/crc_stage2_saved_model_prediction_reproduction_audit.csv`
- `tables/crc_stage2_specific_probability_pattern_diagnostic.csv`
- `tables/crc_stage2_selected_model_instability_diagnostic.csv`
- `tables/crc_stage2_positive_class_orientation_audit.csv`
