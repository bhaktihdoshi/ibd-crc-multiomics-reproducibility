# Analysis Timeline / Development Log

Dates below are drawn from filesystem timestamps and manifest/report timestamps found in the original project where defensible; phases without a confident date are listed in likely sequence without a specific date. Do not treat undated phases as precisely ordered relative to each other beyond the general flow described.

| Phase | What happened | Evidence |
|---|---|---|
| Initial cohort-specific preprocessing | IBD (Franzosa, iHMP) and CRC (Sinha, Yachida) cohorts acquired, QC'd, and matched independently | `preprocessing/ibd/`, `preprocessing/crc/` |
| Early IBD modelling | Cohort-local predictive classifiers, hyperparameter search, calibration work | `exploratory/historical_analyses/` (IBD predictive material) |
| CRC parity work | CRC Stage 1/Stage 2 pipeline built to methodological parity with the IBD pipeline | `preprocessing/crc/CRC_IBD_PARITY_CROSSWALK.csv`, `CRC_PIPELINE_RUN_ORDER.md` |
| Differential abundance | Within-cohort DA for both diseases | `analyses/02_differential_abundance/` |
| Expanded DA reopening | A later, expanded/corrected DA pass — see `ibd_expanded_da_replication_summary.csv` and `ibd_expanded_replication_final_audit.csv` naming | `preprocessing/ibd/`, top-level `tables/` in original project |
| Stage 3 cross-cohort biological analysis | Within-disease concordance, cross-disease concordance, Yachida ordered-state analysis | `analyses/03_cross_cohort_replication/` |
| Initial Stage 4 integration (pre-correction) | Original Stage 4A–4E pipeline (genus reconstruction, Stage 4B/C/D/E) | `exploratory/historical_analyses/stage4_pre_correction_v1/` |
| Corrected-v2 | Hardened re-implementation: genus double-counting fix, decision-score ranking metrics, corrected threshold selection, block-aware early integration | `final_analysis/code/stage4_corrected_v2_common.py` and siblings |
| Stage 4B decision-score correction (v2 → v2b) | An intermediate `run_stage4b_corrected_v2.py` (still `predict_proba`-based) was itself superseded within corrected-v2 by `run_stage4b_corrected_v2b.py` | `final_analysis/code/legacy_within_corrected_v2/run_stage4b_corrected_v2.py` vs. `final_analysis/code/run_stage4b_corrected_v2b.py` |
| Stage 4C | MOFA + sCCA structural integration, with explicit sCCA tuning-vs-held-out terminology correction | `final_analysis/code/run_stage4c_corrected_v2.py` |
| Stage 4D/E | KEGG/NJC19 graph construction (identity, KEGG-only, broad, conservative) and GCN vs. identity/ridge comparison. Authority freeze dated **20 August 2026** | `final_analysis/manifests/FINAL_STAGE4_AUTHORITY_MANIFEST.json` |
| Post-freeze hardening | Microbiome missingness NaN/zero semantics hardened; strict unresolved-missing guard added | `final_analysis/manifests/POST_FREEZE_CODE_PATCH_MANIFEST.json` |
| CRC microbiome representation sensitivity | Rank and presence/absence representations added and run as a post-freeze sensitivity branch | `final_analysis/code/run_stage4b_crc_microbiome_sensitivity.py`, `final_analysis/tables/sensitivity_crc_microbiome_*` |
| Equivalence verification | Post-freeze code changes proven not to alter any frozen number | `final_analysis/manifests/stage4_post_freeze_equivalence_check.json` |
| Final QA | Consolidated 10-check QA suite | `final_analysis/reports/final_stage4_qa_report.json` |
| Co-primary reporting addendum | Microbiome and metabolome reframed as co-primary (not principal/stress-test) for final thesis reporting, without changing any number | `final_analysis/manifests/POST_FREEZE_ANALYSIS_ROLE_ADDENDUM.json` |
| Stage 2/3 claim provenance map | Machine-readable claim→source-CSV map built and verified for the biological-replication findings | `analyses/03_cross_cohort_replication/manifests/stage2_3_manuscript_claim_provenance.csv` |
| GitHub repository curation | This repository assembled from the frozen, verified project state | `docs/file_provenance_map.csv`, `REPOSITORY_RELEASE_MANIFEST.json` |
