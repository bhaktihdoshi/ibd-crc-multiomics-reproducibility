# Preprocessing Overview

## Microbiome

1. **Acquisition/taxonomic profiling** (`preprocessing/microbiome/stage4a_acquisition/`): raw metagenomic/16S data profiled to per-cohort taxonomic tables (e.g. MetaPhlAn for iHMP).
2. **Genus reconstruction** (`final_analysis/code/stage4_corrected_v2_common.py::authoritative_genus_matrix`): one coherent taxonomic source per genus per cohort (no parent+descendant double counting), then re-closure so each sample sums to 1.
3. **Missingness audit**: explicit `NaN` (unresolved/never-measured) is distinguished from an observed `0` (detected-absent); a feature that is `NaN` for every sample in a cohort is excluded rather than zero-filled. See `final_analysis/code/audit_stage4a_microbiome_missingness.py`.
4. **Training-fold prevalence filtering** (≥10% of training-fold samples must have non-zero abundance).
5. **Re-closure** of the retained feature subset so each sample sums to 1 again.
6. **Fixed pseudocount** (1e-6) added before log transform.
7. **Centred log-ratio (CLR) transform**.
8. **Feature-wise z-standardisation** using training-fold mean/SD only.
9. **Frozen transform applied** to validation/external samples (no re-fitting).

Finite-value assertions are enforced after every step (`_require_no_unresolved_missing`, `_clr`, `_reclose` all raise on NaN/inf).

## Metabolomics

1. **Metabolite identifier harmonisation** to HMDB IDs across cohorts (`preprocessing/ibd/`, `preprocessing/crc/`).
2. **Training-fold missingness filtering** (>30% missing in the training fold → feature removed).
3. **Training-fold zero-variance filtering**.
4. **Training-fold median imputation**.
5. **Training-fold z-standardisation**.
6. **Frozen transform applied** to validation/external samples.

This prevents leakage but does **not** establish cross-assay quantitative equivalence — see `docs/limitations.md`.

## Feature-space construction and matching

- Participant matching requires exact microbiome/metabolome sample alignment (`aligned_train_external()` asserts identical feature IDs and order between train and external sets; a mismatch is a hard failure, not a silent fill).
- One row per participant is enforced (duplicate sample IDs raise an error).
- Shared genus feature spaces (50 genera for IBD, 52 for CRC) are constructed once and reused across all IBD/CRC tasks respectively.

## QC and leakage prevention

All trainable preprocessing steps above are fit exclusively within the training partition of each CV fold — see `docs/reproducibility.md` for the full leakage-prevention design and how it is tested.
