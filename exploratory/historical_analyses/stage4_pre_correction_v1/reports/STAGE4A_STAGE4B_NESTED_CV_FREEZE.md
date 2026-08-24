# Stage 4B Nested-CV Freeze

Internal performance will use stratified nested CV in the training cohort.
Outer CV: 5 folds where class counts permit.
Inner CV: 5 folds where class counts permit.
Ridge C grid: [0.01, 0.1, 1.0, 10.0].
All empirical preprocessing and C selection must occur inside training folds only.
Thresholds for MCC/balanced accuracy diagnostics are selected from training-only out-of-fold predictions and frozen before external validation.
