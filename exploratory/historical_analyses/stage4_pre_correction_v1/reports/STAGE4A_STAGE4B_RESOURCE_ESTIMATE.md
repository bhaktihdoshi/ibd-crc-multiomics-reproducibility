# Stage 4B Resource Estimate

Representations: 3. C values: 4. Outer folds: 5. Inner folds: 5.
Approximate fits per representation per direction: outer_folds x inner_folds x C_values + final refit = 5 x 5 x 4 + 1 = 101.
Approximate total supervised fits across eligible directions/tasks/representations: 2424.
Expected runtime: minutes to low hours on a laptop because Stage 4B is restricted to ridge/logistic models and modest feature spaces.
Expected RAM: <8 GB for dense matrices of the frozen dimensions; disk use is dominated by saved tables/plots rather than models.
