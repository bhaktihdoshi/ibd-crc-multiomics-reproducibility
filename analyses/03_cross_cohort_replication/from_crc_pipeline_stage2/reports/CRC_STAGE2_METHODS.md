# CRC Stage 2 Methods

CRC Stage 2 reporting consumes frozen predictions, selected thresholds, internal nested-CV results, selected model metadata and saved joblib pipelines. It does not call primary modelling, hyperparameter search, threshold optimisation, final fitting, differential abundance, pathway analysis or Yachida stage analysis.

External ROC, precision-recall, transportability, calibration and probability-distribution plots use the observed external probabilities exactly as saved. Frozen thresholds were selected from training-cohort out-of-fold predictions using Youden J. Feature interpretation uses the existing training-only feature-importance table.
