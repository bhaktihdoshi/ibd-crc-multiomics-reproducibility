# IBD preprocessing, differential abundance & cross-cohort replication

This directory (together with `analyses/02_differential_abundance/`,
`analyses/03_cross_cohort_replication/`, and `exploratory/historical_analyses/`)
holds the curated contents of the original `thesis_preprocessing_workflow/ibd_pipeline`
project — the IBD-specific preprocessing, differential-abundance (DA), and
bidirectional cross-cohort replication stage of the thesis. Full provenance for
every file (copied or excluded) is recorded in
`docs/_provenance_fragment_ibd.csv`; files over 10MB that were found but not
copied are recorded in `docs/_large_file_fragment_ibd.csv`.

## What this pipeline is (per its own README/config)

Per the source `ibd_pipeline/README.md`: *"IBD-only externally validated
pipeline... uses only supplied Franzosa and iHMP sample IDs, repairs Franzosa
metabolite access from the published supplement without adding samples,
selects first eligible iHMP visits for the primary analysis, and runs
bidirectional external validation."* It is driven by a single entry point,
`run_pipeline.py` (copied here), configured via `config.json` (copied here),
and covers cohort loading/QC, metabolite-ID harmonisation (HMDB/KEGG),
differential abundance + pathway enrichment, cross-cohort (Franzosa <-> iHMP)
biological replication, AND IBD-only predictive modelling (nested CV,
hyperparameter search, ridge/elastic-net/random-forest/extra-trees/
gradient-boosting classifiers).

**Important — this is an upstream feeder stage, not the thesis's final
predictive/integration analysis.** The FINAL multi-omics predictive/
integration work lives in a separate `stage4_multiomics_integration/
corrected_v2` pipeline, curated separately into this repository under
`final_analysis/`. `ibd_pipeline` feeds that later work (see
`final_feature_package_for_crc_integration.csv` and
`ibd_graph_node_candidates.csv`, both routed to
`analyses/03_cross_cohort_replication/tables/`) but its own predictive
modelling should not be read as the thesis's final predictive claims.

## What is here vs. what was routed elsewhere

- **preprocessing/ibd/** (this directory): cohort loading, sample
  matching/exclusion, QC, metabolite-ID harmonisation (HMDB/KEGG annotation
  resources), processed feature matrices (`data_processed/`), pipeline
  code/config/logs, and pipeline-level documentation/audit reports.
- **analyses/02_differential_abundance/**: differential-abundance results,
  volcano/pathway-enrichment figures and tables, annotation/matching-confidence
  tables, and the `biological_sensitivity_shared_space` sensitivity-analysis
  subset (DA recomputed on the shared-feature-space subset).
- **analyses/03_cross_cohort_replication/**: the `biological_primary_expanded`
  figures/tables — explicitly the authoritative expanded DA/replication
  analysis underlying the thesis's cross-cohort biological-replication
  findings (e.g. the strictly-replicated IBD metabolites) — plus the two
  feature-package tables that feed Stage 4.
- **exploratory/historical_analyses/**: everything classified
  `HISTORICAL_SUPERSEDED` (see below).
- **exploratory/diagnostic_analyses/**: `run_postrun_diagnostics.py` and the
  post-run diagnostic changelog report.
- **tests/ibd/**: `test_ibd_crc_parity_transformers.py`.

## HISTORICAL_SUPERSEDED material and why

All IBD-only **predictive/classifier** material was classified
`HISTORICAL_SUPERSEDED` and routed to `exploratory/historical_analyses/`,
because prediction is not this pipeline's authoritative contribution — that
role belongs to the later Stage 4 (`stage4_multiomics_integration/
corrected_v2`) pipeline curated elsewhere in this repository. Nothing in the
source `ibd_pipeline/README.md` or its manifests claims the IBD-only
classifiers are the thesis's final predictive result, so the default
(supersede) rule from the curation brief applies. This includes:

- **`exploratory/historical_analyses/ibd_models_not_copied`**: 111 serialized
  `.joblib` classifier objects (ridge/elastic-net logistic, random forest,
  extra trees, gradient boosting, across `ibd_vs_control`/`cd_vs_control`/
  `uc_vs_control` x clinical/metabolomics/integrated feature sets x
  Franzosa<->iHMP directions). **Binaries were NOT copied** (see curation
  rule on model binaries); each is logged individually as an excluded row in
  `docs/_provenance_fragment_ibd.csv`.
- **`exploratory/historical_analyses/ibd_predictions/`**: raw
  bidirectional external prediction / OOF-threshold CSVs from those
  classifiers.
- **`exploratory/historical_analyses/ibd_predictive_tables/`**:
  hyperparameter-search results/audit/grid/summary, nested-CV performance,
  internal/external model-comparison, calibration, threshold, decision-curve,
  permutation-test, and presentation-slide tables for the classifiers.
- **`exploratory/historical_analyses/ibd_predictive_figures/`**: ROC/PR
  curves, calibration plots, confusion matrices, decision curves, threshold
  visualisations, and per-model top-feature-importance plots.
- **`exploratory/historical_analyses/ibd_predictive_reports/`**: calibration
  policy, threshold-transportability diagnostic, and external-
  transportability interpretation reports.

Two files at the top of `tables/hyperparameter_search_results.csv` and
`tables/hyperparameter_search_audit.csv` (~16MB each) additionally exceeded
the 10MB size limit and were excluded on that basis as well as being
predictive/superseded material — see `docs/_large_file_fragment_ibd.csv`.

## Excluded duplicates

`figures/*.png` at the top level of `ibd_pipeline/figures/` (125 files) were
found to be byte-identical loose re-exports of figures already organised
under `figures/{main_text,supplementary,postrun,biological_primary_expanded,
biological_sensitivity_shared_space}/`; 115 of these were excluded as
duplicates (10 had no categorized-subfolder counterpart and were individually
classified and copied). The `postrun_backups/` directory (156MB, 890 files)
was confirmed to be a pure duplicate backup of `tables/` and `figures/` and
was excluded entirely per the curation brief's hard exclusion rule — it is
not represented in the provenance fragment at all (never enumerated).

## Reproduction notes

- `run_pipeline.py` is copied as-is (monolithic script covering
  preprocessing + DA + replication + the superseded predictive modelling);
  it is the authoritative source of the preprocessing/DA/replication logic
  even though parts of it are no longer the prediction reference.
- `ibd_pipeline_transformers.py`, `config.json`, and `requirements.txt` are
  needed alongside `run_pipeline.py` to reproduce a run.
- `tables/franzosa_annotation_long.csv` is retained as a metabolite-ID
  harmonisation reference; its iHMP counterpart, `ihmp_annotation_long.csv`
  (~14MB), exceeded the size limit and was not copied (see
  `docs/_large_file_fragment_ibd.csv`).
- No raw third-party source data (e.g. untouched journal supplementary files
  or patient-level raw sequencing/metabolomics files) was found inside
  `ibd_pipeline` itself — all data files here are processed/derived outputs.
