# IBD Pipeline Final Status

Ready for thesis: supplied-data reconstruction, first-visit iHMP primary handling, original-sample Franzosa metabolite repair, annotation audit, bidirectional external validation tables, Brier-score reporting, formal differential-abundance tables, calprotectin-adjusted analyses, and predictive feature-importance tables.

Ready with caveats: model outputs are defensible for cohort-transfer evaluation because internal model development now uses nested cross-validation, but external results must still not be used for model selection.

Hyperparameter tuning note: Grid search was retained where the candidate space was computationally manageable because it provides exhaustive and reproducible evaluation of predefined hyperparameter values. Broader randomised searches were used for larger tree-based parameter spaces to improve coverage of plausible model complexity without requiring evaluation of every Cartesian combination.

Class-imbalance note: logistic regression, random forest and extra trees use estimator-level balanced class weights. Gradient boosting does not support class_weight directly, so fold-local balanced sample weights are passed as clf__sample_weight during training-only nested CV, final refit, threshold cross-fitting and training-only permutation importance. No SMOTE or oversampling is used.

Reporting completeness note: representative PCA/EDA, model comparison, external transportability, feature-interpretation, stable-metabolite, pathway and sensitivity-status outputs are indexed in tables/thesis_reporting_output_index.csv and reports/thesis_reporting_output_index.md.

Calibration note: calibration curves, Brier score, expected calibration error and calibration intercept/slope are assessment-only outputs generated after frozen external prediction. They are not used for model selection, hyperparameter tuning or threshold optimisation.

Caveats: pathway/class enrichment is limited to reproducibly available HMDB/KEGG/SMPDB annotation and excludes terms with fewer than three mapped metabolites; predictive feature importance and differential abundance are not causal biomarker evidence.

Not recommended: CD-versus-UC primary modelling and deep learning/GNNs at this stage.
