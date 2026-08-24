# Stage 4B Methods

Stage 4B compares matched-subject metabolomics-only ridge, microbiome-only ridge and early-concatenation ridge. The three representations use identical matched subjects within each cohort/direction. All filtering, CLR/rCLR, imputation and scaling must be fitted on training folds only.

Stage 4B uses the final frozen Stage 4A design only. Earlier acquisition-stage feasibility reports are intentionally ignored.

Ridge implementation: L2 logistic regression, class_weight='balanced', solver='liblinear', C grid [0.01, 0.1, 1.0, 10.0]. Thresholds are selected from training-only predictions and frozen before external validation.
