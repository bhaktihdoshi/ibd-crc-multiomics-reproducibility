# Cohort Overview

| Cohort | Disease | Microbiome assay | Metabolomics assay | Case n | Control n | Special notes |
|---|---|---|---|---|---|---|
| Franzosa | IBD (CD/UC/Control) | Shotgun metagenomics | LC-MS | 81 | 56 | Species-level profiles aggregated to genus (no explicit genus-level profile available) |
| iHMP / IBDMDB | IBD (CD/UC/Control) | Shotgun metagenomics | LC-MS, multi-method | 65 | 26 | First available/one-visit-matched representation used per participant |
| Sinha | CRC (CRC/Control) | 16S rRNA amplicon (V3–V4) | Non-targeted LC/GC-MS/MS (Metabolon) | 42 | 89 | Contributes **both** microbiome and metabolomics; used in the CRC microbiome-only, metabolome-only, and early-integration representations |
| Yachida | CRC (CRC/Control), staged | Shotgun metagenomics | CE-TOFMS | 125 | 127 | Disease-stage-ordered (Healthy→MP→Stage 0→I→II→III→IV) used in the Stage 3B ordered-state analysis |

Case/control counts above are from `final_analysis/tables/final_stage4/table1_cohort_and_assay_overview.csv` (authoritative) — consult that table directly for per-representation/per-task counts, which can differ slightly from the headline numbers above depending on matching requirements.

## Assay heterogeneity

- **IBD pair (Franzosa, iHMP)**: both shotgun metagenomics, but taxonomic database/processing heterogeneity between the two source pipelines remains — not a perfectly harmonised representation even within-platform.
- **CRC pair (Sinha, Yachida) — microbiome**: 16S rRNA amplicon vs. shotgun metagenomics — platform is confounded with cohort. See `docs/limitations.md`.
- **CRC pair (Sinha, Yachida) — metabolomics**: LC/GC-MS/MS (Metabolon) vs. CE-TOFMS — different platforms; both cohorts have metabolomics data, but cross-cohort quantitative-scale equivalence was not established, same as the IBD pair (see `docs/limitations.md`).
- **Metabolomics across all pairs**: different assay platforms/methods per cohort; cross-cohort quantitative-scale equivalence was not established (see `docs/limitations.md`).

## Genus harmonisation policy

Each cohort follows one coherent per-cohort policy to avoid parent+descendant double counting: `explicit_genus_only` (use only rows already labelled at genus rank) for iHMP, Sinha, and Yachida; `aggregate_species_to_genus` (sum species-level rows up to their parent genus) for Franzosa, which lacks an explicit genus-level profile. See `final_analysis/code/stage4_corrected_v2_common.py`'s `authoritative_genus_matrix()`.
