# CRC Preprocessing (Stage 1) and Stage 2 Methods Documentation

This directory curates the **CRC-specific data-acquisition, provenance, and
preprocessing (Stage 1)** work from `thesis_preprocessing_workflow/crc_pipeline`,
plus every Stage 1/Stage 2 methods/policy document from that pipeline. It is
the CRC-side counterpart to `preprocessing/ibd/`. The CRC pipeline's own
*Stage 2* differential-abundance, cross-cohort external-validation, and
predictive-modelling *outputs* are curated separately, in
`analyses/02_differential_abundance/crc/`,
`analyses/03_cross_cohort_replication/from_crc_pipeline_stage2/`, and
`analyses/04_predictive_baselines/crc/` respectively (see "Where Stage 2
outputs live" below) — this folder holds Stage 1 data/code plus all the
shared Stage 1 and Stage 2 markdown methods documentation.

## What Stage 1 covers

Two stool/fecal CRC metabolomics cohorts (Sinha et al. 2016, 131 samples;
Yachida et al. 2019, 347 samples) were audited, harmonised, and QC'd, without
any modelling, imputation, scaling, feature selection, or differential
abundance:

- Local metadata, metabolomics matrices, and metabolite mapping files were
  inventoried and checked for duplication/missingness.
- Individual Yachida Stage 0–IV labels were recovered from the official
  Supplementary Table S2-2 (Sinha has no deterministic participant-level
  stage field and remains binary-only, Control vs CRC).
- HMDB/KEGG metabolite identifiers were harmonised, yielding 110 shared HMDB
  and 107 shared KEGG identifiers (217 unified) across cohorts — the feature
  space used by all later cross-cohort work.
- Raw, non-imputed, non-scaled Stage 2 model-preparation tables were produced
  for the primary CRC-vs-Control task and nine Yachida-only secondary
  stage/subtype tasks.

Stage 1 is frozen (`RELEASE_NOTES.md`, status: Frozen) and must not be
modified by any downstream stage. `crc_pipeline.py` is the single
user-facing CRC entry point; `crc_pipeline_transformers.py` provides the
sklearn-compatible transformer classes used by the frozen pipeline.

## Contents

- `crc_pipeline.py`, `crc_pipeline_transformers.py` — Stage 1 code (current
  version, includes the Stage 2 module loader).
- `docs/` — all `CRC_STAGE1_*`, `CRC_STAGE2_*` markdown methods/policy
  documents (methods summary/appendix, run order, execution guide, input
  and output specifications, modelling checklist, model rationale,
  transformation/calibration/stage-subtype policies, limitations draft) plus
  `CRC_IBD_PARITY_CROSSWALK.csv`, a thesis-authored crosswalk documenting
  IBD-vs-CRC Stage 2 methodological-parity decisions. The `CRC_STAGE2_*`
  documents here are the **pre-registered** policy/plan versions; the
  **post-execution** final Stage 2 methods/limitations/results reports live
  with the Stage 2 outputs they describe (see below).
- `tables/` — Stage 1 audit, QC, provenance, and Stage 2-input-contract
  tables, including `tables/stage2_model_prep/*.csv` (the ten raw model-prep
  files: primary CRC-vs-Control plus nine Yachida-only secondary tasks).
- `figures/` — Stage 1 QC figures (cohort PCA, zero-fraction-by-cohort,
  Yachida stage-distribution figures, Stage 2 task-eligibility figure).
- `reports/crc_stage1_data_audit.md` — full narrative Stage 1 audit report.
- `manifests/` — Stage 1 release manifest, build manifest, freeze-integrity
  checks, complete data dictionary, and data-provenance table/diagram.
- `external_metadata/` — small, thesis-team-curated cohort metadata
  crosswalks (`SINHA_CRC_2016_metadata_curated.tsv`,
  `YACHIDA_CRC_2019_metadata_curated.tsv`) used as Stage 1 pipeline input.
  **Not included**: the raw official journal supplementary files (Sinha
  S1–S5, and the 82MB Yachida Supplementary Tables 1–15 workbook) — these are
  third-party redistributions of published journal supplementary material
  and were excluded per the repository's data-inclusion policy. See
  `data/README.md` (or `docs/_large_file_fragment_crc.csv` /
  `docs/_provenance_fragment_crc.csv`) for what they are and where to obtain
  them from the original publications (Sinha et al. 2016, *PLOS ONE*,
  doi:10.1371/journal.pone.0152126; Yachida et al. 2019, *Nature Medicine*).

Unit tests for this pipeline (KEGG ORA outputs, model selection/balanced GB,
transformer joblib round-trip, Yachida stage-label recovery) are curated in
`tests/crc/`.

## Where Stage 2 outputs live (and what is HISTORICAL_SUPERSEDED)

Stage 2 consumes the frozen Stage 1 contract above to build CRC-vs-Control
predictive models with bidirectional Sinha↔Yachida external validation, run
differential abundance/KEGG ORA, and prepare Yachida-only secondary
stage/subtype analyses. Its outputs are split three ways in this repository,
by scientific role, all implemented by a single script,
`analyses/04_predictive_baselines/crc/code/run_stage2_modelling.py`:

1. **`analyses/02_differential_abundance/crc/`** — the primary expanded
   cohort-specific differential-abundance and KEGG ORA results
   (`EXPANDED_COHORT_SPECIFIC_DA_AND_ORA`), classified
   `FINAL_AUTHORITATIVE`.
2. **`analyses/03_cross_cohort_replication/from_crc_pipeline_stage2/`** —
   everything that documents or feeds cross-cohort replication findings:
   the bidirectional Sinha↔Yachida external-validation performance,
   predictions, calibration, thresholds, and diagnostics (the direct source
   of the below-chance / "CRC strict replication" external-transportability
   finding); the harmonised shared-feature-space DA/ORA **sensitivity**
   analysis (`HARMONISED_SHARED_FEATURE_SENSITIVITY_ANALYSIS`, classified
   `SENSITIVITY`) used for Sinha-vs-Yachida within-disease concordance
   checks; and the Yachida-only ordered-state/stage-subtype manifests and
   task plan. Classified `FINAL_AUTHORITATIVE` (per
   `CRC_STAGE2_AUTHORITATIVE_OUTPUT_MANIFEST.csv`, copied into that
   directory's `reports/`) unless noted otherwise.
3. **`analyses/04_predictive_baselines/crc/`** — the CRC-vs-Control
   classifier-building infrastructure itself: canonical feature-space
   construction, clinical-covariate recovery, nested-CV design,
   hyperparameter search, model selection, and leakage-safety checks.
   **Classified `HISTORICAL_SUPERSEDED`**: the thesis's final,
   authoritative CRC-prediction analysis is the separately curated Stage 4
   `corrected_v2` multi-omics integration pipeline
   (`final_analysis/` / `exploratory/historical_analyses/stage4_pre_correction_v1/`
   in this repository); this Stage 2 CRC-only classifier work remains the
   historical record of how those earlier CRC-only baselines were built and
   externally validated, and is retained for transparency and reproducibility,
   not as the thesis's final CRC-prediction claim.

Fitted model binaries (`*.joblib`) and `stage2/fast_test/` smoke-test
duplicates from the source pipeline were **not copied** (binary-artifact and
non-final-output exclusion rules); their non-binary configuration/
reproducibility metadata was copied instead. See
`docs/_provenance_fragment_crc.csv` for the full file-by-file rationale.
