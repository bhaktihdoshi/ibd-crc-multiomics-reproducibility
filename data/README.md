# Data Availability

**No raw or third-party-source data is redistributed in this repository.** This folder documents where to obtain each cohort's source data and how it maps onto the processing code in `preprocessing/`. See `docs/large_file_audit.csv` and `docs/file_provenance_map.csv` for the complete list of excluded source files with reasons.

## Cohorts and sources

| Cohort | Disease | Data type | How to obtain |
|---|---|---|---|
| Franzosa et al. | IBD | Metagenomics + metabolomics | Original publication supplementary data (Nature Microbiology, 2019 — "Gut microbiome structure and metabolic activity in inflammatory bowel disease"). Excluded from this repo: `41564_2018_306_MOESM4_ESM.xlsx` (~16MB supplementary table). |
| iHMP / IBDMDB | IBD | Metagenomics + metabolomics | [IBDMDB data portal](https://ibdmdb.org/) (Integrative Human Microbiome Project). Raw/processed downloads obtained directly from the portal per its own terms. |
| Sinha et al. | CRC | 16S rRNA amplicon | Original publication supplementary tables (excluded from this repo). |
| Yachida et al. | CRC | Shotgun metagenomics + metabolomics, staged | Original publication supplementary data (Nature Medicine, 2019 — "Metagenomic and metabolomic analyses reveal distinct stage-specific phenotypes of the gut microbiota in colorectal cancer"). Excluded from this repo: `41591_2019_458_MOESM3_ESM.xlsx` (~82MB) and `yachida_supplementary_tables_1_15.xlsx` (~82MB, a second copy used at a different pipeline stage). |
| MiMeDB | Prior-knowledge reference (Stage 4D graph construction) | Metabolite reference database | [MiMeDB](https://mimedb.org/). Excluded from this repo: `mimedb_metabolites_v2.csv` (~51MB). |

None of these third-party files are copied into this repository, regardless of size, because they are original journal/database supplementary material with their own copyright and redistribution terms, not thesis-team-authored content.

## What IS included

Small (<10MB), thesis-team-derived/harmonised files that are transformations of the above (crosswalks, canonical taxon matrices below the size threshold, per-participant harmonised metadata) are included where their scientific role clearly requires it for reproducibility, subject to the size cap and the clinical-metadata redistribution caveat in `docs/privacy_security_audit.md`. Several derived microbiome matrices exceed 10MB (e.g. `yachida_microbiome_canonical_unfiltered.csv`, ~16MB) and are excluded for size reasons even though they are thesis-derived, not raw third-party data — see `docs/large_file_audit.csv` for the recommended action (Git LFS / data release) for each.

## Reconstructing the analysis-ready data

1. Obtain each cohort's source data from the links above.
2. Place raw files under a local `data/raw/<cohort>/` directory (not tracked by this repository).
3. Run the corresponding preprocessing scripts in `preprocessing/` (see `preprocessing/ibd/README.md`, `preprocessing/crc/README.md`, and `preprocessing/microbiome/stage4a_acquisition/`) to regenerate the processed matrices.
4. Set `THESIS_PROJECT_ROOT` (see `docs/reproducibility.md`) so the scripts resolve paths relative to your local checkout rather than the original author's machine.

## Participant-level clinical metadata — withheld (conservative decision)

**33 files containing participant-level clinical metadata were removed from this repository on 2026-08-25**, following author review. This includes per-participant age, BMI, sex, smoking status, fecal calprotectin, medication flags (antibiotic/steroid/immunosuppressant/mesalamine), visit/collection dates, and CRC disease-stage/trajectory labels, across both the IBD and CRC pipelines (`preprocessing/ibd/tables/included_samples.csv` and `excluded_samples.csv`, several `preprocessing/ibd/data_processed/` and `preprocessing/crc/` tables, and all `crc_stage2_yachida_*_model_prep.csv` files, among others — the full list is in `docs/file_provenance_map.csv`, rows with `status = WITHDRAWN_PARTICIPANT_LEVEL_METADATA`).

**Rationale**: even with de-identified sample IDs and even though the underlying cohorts are publicly studied, redistributing a reformatted, combined per-participant clinical table is a different act than linking to the original public deposition, and the specific redistribution terms of each cohort's original data-use agreement were not independently confirmed. The conservative default is to withhold rather than assume permission.

**What was kept instead**:
- All processing/ingestion **scripts** that read and harmonise this metadata remain in the repository (see `preprocessing/ibd/`, `preprocessing/crc/`) — the code that shows *how* the cohorts were processed is intact.
- Aggregate/non-participant-level tables (cohort counts, missingness rates, feature-metadata crosswalks, sample-flow diagrams) were kept — these do not carry per-participant rows.
- Molecular (microbiome/metabolome abundance) data without attached clinical covariates was kept where it was already below the size threshold.

**How to regenerate the withheld tables**: obtain each cohort's original metadata from the sources listed above (or, for iHMP/IBDMDB, directly from the [data portal](https://ibdmdb.org/); for Sinha/Yachida, from their published supplementary tables), then run the corresponding preprocessing script. The scripts document the exact expected input format.

This decision can be revisited if the author confirms redistribution is explicitly permitted under a given cohort's terms — see `docs/privacy_security_audit.md` for the full audit trail.
