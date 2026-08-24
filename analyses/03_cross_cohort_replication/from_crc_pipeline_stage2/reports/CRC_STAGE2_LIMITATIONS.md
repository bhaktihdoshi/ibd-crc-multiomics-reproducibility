# CRC Stage 2 Limitations

- Marked internal-to-external performance degradation is observed in some selected contexts.
- Cohort, platform and preprocessing heterogeneity limit transportability between Sinha and Yachida.
- Transportability is direction-dependent; Sinha→Yachida and Yachida→Sinha should be interpreted separately.
- Below-chance external ROC-AUC values, where observed, indicate poor cross-cohort transportability under the frozen model orientation rather than a reason to invert predictions.
- Frozen-threshold predictions may collapse to one class in some contexts; this is retained and flagged rather than corrected using external labels.
- Calibration is assessed externally but not corrected or recalibrated on external data.
- External data were never used for model-family selection, hyperparameter tuning, preprocessing, threshold selection or feature-importance fitting.
- Predictive feature importance is not causal evidence and is not equivalent to differential abundance.
