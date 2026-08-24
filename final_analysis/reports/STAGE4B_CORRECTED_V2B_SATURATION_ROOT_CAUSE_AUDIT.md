# Stage 4B Corrected-v2b Saturation / Root-Cause Audit

Stage 4C-E remain paused. The completed corrected-v2 Stage 4B outputs are preserved and should be treated as the probability-metric run.

## Finding

The completed corrected-v2 Stage 4B runner calculated external ROC-AUC, PR-AUC, bootstrap CIs and paired model comparisons from `predict_proba` probabilities. This is numerically vulnerable for logistic regression when external logits are extremely large or tightly compressed, because sigmoid probabilities can saturate at exactly 0/1 or lose resolution through floating-point rounding.

Corrected-v2b patches the Stage 4B runner so that:

- `decision_score` is exported for internal OOF, source-threshold OOF and external predictions;
- ROC-AUC and average precision are calculated from `decision_score`;
- paired external bootstrap ΔROC-AUC comparisons use `decision_score`;
- probabilities are still exported and retained for diagnostics;
- classification uses a frozen source-derived Youden threshold on the `decision_score` scale;
- a probability logit-equivalent threshold is exported only for interpretation.

## Current Corrected-v2 Probability Resolution Problems

The following values are from the frozen corrected-v2 external prediction table and will become stale once corrected-v2b is rerun:

| disease | train -> external | representation | unique probabilities | exact zeros | exact ones | probability range | probability SD | ROC-AUC currently from probability |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| IBD | Franzosa -> iHMP | METABOLOME | 2 | 32 | 59 | 1.000000 | 0.477485 | 0.657692 |
| IBD | Franzosa -> iHMP | EARLY_INTEGRATION | 2 | 40 | 51 | 1.000000 | 0.496334 | 0.703846 |
| CRC | Sinha -> Yachida | METABOLOME | 6 | 121 | 123 | 1.000000 | 0.499748 | 0.507969 |
| CRC | Sinha -> Yachida | EARLY_INTEGRATION | 8 | 119 | 126 | 1.000000 | 0.498561 | 0.514362 |
| IBD | iHMP -> Franzosa | METABOLOME | 137 | 0 | 0 | 0.007237 | 0.000928 | 0.845899 |

The iHMP -> Franzosa metabolome model is especially important: probability values are compressed around 0.525-0.532 and yield a one-class external classification under the frozen probability threshold, while ROC-AUC remains high. This is possible because ROC-AUC measures ranking, not thresholded classification or calibration.

## Root-Cause Hypotheses To Audit

Corrected-v2b adds a transformation-domain-shift diagnostic script. It does not change the primary pipeline. It audits:

- source and external raw feature ranges;
- whether blocks are metabolome or microbiome and what preprocessing is applied;
- source-fitted fold/final transformers;
- external transformed quantiles and extreme z-score fractions;
- maximum absolute model logits;
- nonfinite values;
- exact probability zeros/ones;
- unique decision scores versus unique probabilities.

Specific questions to check after running diagnostics:

- Are Franzosa and iHMP metabolomics matrices on compatible upstream transformation/unit scales?
- Are Sinha and Yachida metabolomics matrices on compatible upstream transformation/unit scales?
- Do shared HMDB IDs hide incompatible measurement units or transformations?
- Does early integration inherit extreme metabolite logits and dominate the microbiome block?
- Is block-specific standardisation fitted on source data only? It should be.

## Leakage-Safe Sensitivities Prepared For Later Approval Only

If the diagnostic confirms incompatible upstream transformations, do not silently change the primary pipeline. Possible future sensitivity analyses include:

- within-sample metabolite ranks;
- source-defined clipping/winsorisation thresholds;
- consistently applied log1p where scientifically valid for all cohorts;
- robust source-fitted scaling;
- presence/absence or within-sample ranks for CRC microbiome.

These are not run by the corrected-v2b patch.

## Statistical Wording Correction

The v2b paired table reports p-values less than 0.001 as `p < 0.001`, not `p = 0`. It also adds BH-FDR across the eight prespecified paired comparisons. Interpretive wording is conservative:

- `unadjusted interval excluded zero`;
- `inconclusive`;
- `early integration was lower`.

The post-hoc best-single-omic comparison remains explicitly labelled as post-hoc because selecting that comparator uses external outcomes.

## Exact Deterministic Rerun Commands

```bash
python thesis_preprocessing_workflow/stage4_multiomics_integration/corrected_v2/scripts/run_stage4b_corrected_v2b.py
python thesis_preprocessing_workflow/stage4_multiomics_integration/corrected_v2/scripts/audit_stage4b_corrected_v2b_postrun.py
python thesis_preprocessing_workflow/stage4_multiomics_integration/corrected_v2/scripts/audit_stage4b_corrected_v2b_transform_shift.py --run
```

## Expected Corrected-v2b Outputs

- `tables/stage4b_corrected_v2b_external_performance.csv`
- `tables/stage4b_corrected_v2b_internal_performance.csv`
- `tables/stage4b_corrected_v2b_thresholds.csv`
- `tables/stage4b_corrected_v2b_paired_model_comparisons.csv`
- `tables/stage4b_corrected_v2b_probability_diagnostics.csv`
- `tables/stage4b_corrected_v2b_transformation_domain_shift_diagnostics.csv`
- `predictions/stage4b_corrected_v2b_external_predictions.csv`
- `predictions/stage4b_corrected_v2b_internal_oof_predictions.csv`
- `predictions/stage4b_corrected_v2b_source_oof_threshold_predictions.csv`
- `reports/STAGE4B_CORRECTED_V2B_COMPLETED_RUN_AUDIT.md`

## Blocking Criteria Before Stage 4C-E

Do not proceed to Stage 4C-E until corrected-v2b has been run and reviewed if:

- decision-score metrics materially differ from probability metrics for saturated models;
- transformation diagnostics show extreme external z-score shifts;
- external logits are nonfinite or produce complete probability saturation;
- source/external metabolomics transformations are not scientifically comparable.
