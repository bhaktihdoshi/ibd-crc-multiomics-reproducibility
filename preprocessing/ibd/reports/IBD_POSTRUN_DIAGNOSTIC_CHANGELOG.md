# IBD Post-run Diagnostic Changelog

- Created timestamped backup of current completed IBD run.
- Recorded pre-remediation hashes in `logs/ibd_preremediation_manifest.json`.
- Added post-run diagnostic tables for thresholds, positive class, feature order, Brier score, bidirectional asymmetry, clinical comparator and automated tests.
- Added post-run threshold probability distribution plots.
- Moved custom preprocessing filters to `ibd_pipeline_transformers.py` and migrated all selected joblib pipelines away from `__main__` transformer references.
- Verified all selected joblib pipelines load successfully and fitted missingness/constant-feature masks are directly inspectable.
- Verified selected logistic final fits converged by checking fitted `n_iter_ < max_iter`.
- Fixed Matplotlib `boxplot(labels=...)` deprecation in `run_pipeline.py`.
- Did not rerun nested CV or alter model selection, cohort definitions, external validation design, labels or thresholds.
