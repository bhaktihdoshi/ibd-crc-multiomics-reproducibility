# CRC Stage 2 Results Summary

CRC Stage 2 reporting was generated from completed primary outputs only. No models were refitted, no thresholds were reselected and no external labels were used for model selection.

## Selected External Performance

| Transfer | Feature set | Model | Internal ROC-AUC mean ± SD | External ROC-AUC (95% CI) | External PR-AUC (95% CI) | Frozen-threshold collapse |
|---|---|---|---|---|---|---|
| Sinha→Yachida | clinical | ridge_logistic | 0.559 ± 0.164 | 0.427 [0.360, 0.491] | 0.474 [0.415, 0.550] | False |
| Sinha→Yachida | integrated | ridge_logistic | 0.683 ± 0.136 | 0.505 [0.443, 0.567] | 0.552 [0.486, 0.623] | False |
| Sinha→Yachida | metabolomics | ridge_logistic | 0.661 ± 0.127 | 0.478 [0.422, 0.537] | 0.531 [0.470, 0.597] | False |
| Yachida→Sinha | clinical | extra_trees | 0.562 ± 0.064 | 0.441 [0.341, 0.549] | 0.307 [0.221, 0.424] | False |
| Yachida→Sinha | integrated | ridge_logistic | 0.700 ± 0.072 | 0.380 [0.276, 0.482] | 0.292 [0.208, 0.404] | False |
| Yachida→Sinha | metabolomics | ridge_logistic | 0.707 ± 0.071 | 0.413 [0.303, 0.520] | 0.296 [0.210, 0.430] | True |

Below-chance external ROC-AUC values are reported as poor cross-cohort transportability, not as software failures or successful diagnostic classification.
Calibration is assessed externally using frozen predictions only; no recalibration is applied. Predictive feature importance is not causal evidence.
