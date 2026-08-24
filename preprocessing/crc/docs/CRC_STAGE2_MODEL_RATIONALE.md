# CRC Stage 2 Model Rationale

The primary CRC analysis will use the same five model families as the frozen IBD analysis to maintain methodological parity across disease streams.

1. Ridge logistic regression will provide a stable L2-regularized linear baseline for correlated metabolites.
2. Elastic-net logistic regression will combine L1/L2 shrinkage, allowing sparse predictors while retaining correlated feature groups.
3. Random forest will provide a nonlinear bagged-tree ensemble capable of modelling interactions.
4. Extra Trees will provide a more highly randomized tree ensemble that may reduce variance and improve robustness.
5. sklearn GradientBoostingClassifier will provide a sequential boosted-tree model for nonlinearities and interactions.

XGBoost will be omitted from the primary analysis because sklearn gradient boosting already represents the boosted-tree modelling class, XGBoost is an incremental implementation rather than a fundamentally distinct paradigm, and identical IBD/CRC model-family parity is methodologically useful. The thesis contribution is external validation and later IBD-to-CRC integration, not exhaustive algorithm benchmarking. XGBoost may be added later only as a separately labelled sensitivity analysis.

LSTM models will not be enabled for the primary CRC metabolomics analysis because both Sinha and Yachida inputs are cross-sectional sample-level metabolomics tables rather than participant-level longitudinal sequences.
