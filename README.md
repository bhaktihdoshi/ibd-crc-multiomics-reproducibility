# IBD and CRC Multi-Omics Transportability — MSc Thesis Analysis Repository

## At a glance

This thesis tests whether **microbiome** and **metabolome** case-control signatures for **inflammatory bowel disease (IBD)** and **colorectal cancer (CRC)** transport across independently processed cohorts: **Franzosa**, **iHMP/IBDMDB** (IBD, both shotgun metagenomics), **Sinha** (CRC, 16S + metabolomics), **Yachida** (CRC, shotgun + metabolomics, disease-stage-ordered). **All four cohorts contribute both microbiome and metabolome data.**

- **Final analysis**: [`final_analysis/`](final_analysis/) (Stage 4 predictive/integration) + [`analyses/03_cross_cohort_replication/`](analyses/03_cross_cohort_replication/) (Stage 2/3 biological replication). Start with [`final_analysis/README.md`](final_analysis/README.md) for what's authoritative.
- **Microbiome-only and metabolome-only are the two co-primary single-omic analyses** — neither is secondary to the other (early integration is a separate comparator; see §9 below).
- **Sensitivity / exploratory / historical work is clearly segregated**: [`results/sensitivity/`](results/sensitivity/) (post-freeze CRC representation checks) vs. [`exploratory/`](exploratory/) (alternative approaches and the pre-correction pipeline — **not final results**, see its warning README).
- **Data**: no raw/third-party cohort data is redistributed here — see [`data/README.md`](data/README.md) for exact sources, accessions, and retrieval instructions. **Participant-level clinical metadata (age, BMI, medication flags, disease stage, etc.) is deliberately withheld** even though sample IDs are de-identified — see `data/README.md` for what's included vs. withheld and how to regenerate it locally.
- **Reproduce it**: see [`docs/reproducibility.md`](docs/reproducibility.md) for exact commands.
- **Limitations**: cross-cohort metabolomics quantitative-scale equivalence unverified; CRC microbiome platform (16S vs. shotgun) confounded with cohort — see [`docs/limitations.md`](docs/limitations.md).
- **Every number's source**: [`docs/file_provenance_map.csv`](docs/file_provenance_map.csv) and, for the thesis's exact figures/tables, [`final_analysis/tables/final_stage4_figure_source_values.csv`](final_analysis/tables/final_stage4_figure_source_values.csv) / [`analyses/03_cross_cohort_replication/manifests/stage2_3_manuscript_claim_provenance.csv`](analyses/03_cross_cohort_replication/manifests/stage2_3_manuscript_claim_provenance.csv).

Full detail in the 10 sections below.

---

## 1. Project title

Cross-cohort transportability of microbiome and metabolome signatures in inflammatory bowel disease (IBD) and colorectal cancer (CRC): a multi-omics integration analysis.

## 2. Scientific background

Microbiome and metabolome case-control signals are frequently reported within a single cohort but rarely tested for whether they generalise to an independent cohort processed by a different group, with a different assay batch, and sometimes a different measurement platform. This project assembles four public/consortium cohorts spanning IBD and CRC, harmonises their microbiome and metabolome representations as far as is defensible, and tests reciprocal external transportability of single-omic and multi-omic predictive models, alongside descriptive structural integration (MOFA, sparse CCA) and a prior-knowledge-graph-informed GCN.

## 3. Research question

Do microbiome-only and metabolome-only case-control signatures, and their combination, transport across independently processed cohorts of the same disease — and does incorporating curated biological prior knowledge (KEGG pathways, NJC19 microbe–metabolite relations) via a graph neural network improve on simpler comparators?

## 4. Cohorts

| Cohort | Disease | Microbiome assay | Metabolomics assay |
|---|---|---|---|
| Franzosa et al. | IBD (CD/UC/Control) | Shotgun metagenomics | LC-MS |
| iHMP / IBDMDB | IBD (CD/UC/Control) | Shotgun metagenomics | LC-MS, multi-method |
| Sinha et al. | CRC (CRC/Control) | 16S rRNA amplicon (V3–V4) | Non-targeted LC/GC-MS/MS (Metabolon) |
| Yachida et al. | CRC (CRC/Control), staged | Shotgun metagenomics | CE-TOFMS |

Sinha contributes **both** microbiome and metabolomics data to this analysis and is used in the CRC microbiome-only, metabolome-only, and early-integration representations (see `final_analysis/tables/final_stage4/table1_cohort_and_assay_overview.csv` for the authoritative per-cohort assay/sample counts). Sinha (16S) vs. Yachida (shotgun) remains platform-confounded with cohort for CRC **microbiome** transfer specifically — this limitation is carried through the analysis and reporting, not resolved by scaling. The two cohorts also use different metabolomics platforms (LC/GC-MS/MS vs. CE-TOFMS); cross-cohort metabolomics quantitative-scale equivalence was not established for either disease pair (see `docs/limitations.md`).

## 5. IBD and CRC study design

Both diseases follow the same overall design: cohort-specific preprocessing and QC → harmonised feature-space construction (genus-level microbiome composition; HMDB-matched metabolite panel) → within-cohort differential abundance and cross-cohort replication (Stage 2/3) → matched multi-omic predictive and structural integration modelling with reciprocal external validation (Stage 4). The current Stage 4 predictive scope is deliberately restricted to pooled `IBD_vs_Control` and `CRC_vs_Control`; CD/UC subtype analyses are retained as descriptive/historical work, not current predictive tasks.

## 6. Data modalities

- **Microbiome**: taxonomic relative abundance, reconstructed to a single authoritative genus-level representation per cohort (no parent+descendant double counting), fold-local prevalence filtering, re-closure, pseudocount, centered log-ratio (CLR) transform, training-fold z-standardisation.
- **Metabolomics**: HMDB-matched metabolite panels, fold-local missingness/zero-variance filtering, training-fold median imputation and z-standardisation. Cross-cohort quantitative-scale equivalence of the metabolomics assays was **not** established — see `docs/limitations.md`.
- **Clinical/phenotype**: case-control labels only for the current predictive scope; a fully harmonised four-cohort clinical covariate set does not exist (see limitations).

## 7. Repository structure

```
docs/            Analysis overview, timelines, reproducibility, limitations, provenance/audit reports
data/            Data availability documentation (raw data is NOT redistributed — see data/README.md)
preprocessing/   Cohort acquisition, QC, matching, microbiome/metabolomics processing
analyses/        Analyses grouped by stage (Stage 2/3 replication is fully here; Stage 4 is a pointer to final_analysis/)
final_analysis/  THE authoritative Stage 4 code, results, manifests, and provenance — start here for final numbers
exploratory/     Alternative models, alternative preprocessing, historical/superseded pipeline versions
notebooks/       Exploratory and preprocessing notebooks
tests/           Test suites (see tests/README.md)
results/         Thesis-facing final figures/tables, sensitivity results, exploratory results
thesis/          Placeholder for the final submitted PDF (not yet added — see thesis/README.md)
```

## 8. Analysis stages

1. Cohort-specific preprocessing (`preprocessing/`)
2. Stage 2/3: within-cohort differential abundance and cross-cohort biological replication (`analyses/03_cross_cohort_replication/`)
3. Stage 4A: matched multi-omics feature-space construction, authoritative genus harmonisation (`final_analysis/`)
4. Stage 4B: microbiome-only and metabolome-only co-primary predictive models, early-integration comparator
5. Stage 4C: MOFA and sparse CCA structural integration (descriptive)
6. Stage 4D: KEGG/NJC19 prior-knowledge graph construction (identity, KEGG-only, broad KEGG+NJC19, conservative KEGG+NJC19)
7. Stage 4E: graph convolutional network vs. identity ablation vs. matched ridge comparator
8. Post-freeze: sensitivity analyses, QA, equivalence verification, co-primary reporting addendum

## 9. Final analytical hierarchy

```
CO-PRIMARY SINGLE-OMIC ANALYSES:      microbiome-only, metabolome-only
INTEGRATION COMPARISON:               early integration
LATENT / STRUCTURAL MULTI-OMICS:      MOFA, sCCA (descriptive; sCCA fold correlations are
                                       tuning-CV diagnostics, not unbiased held-out estimates)
PRIOR-KNOWLEDGE / GRAPH ANALYSIS:     KEGG/NJC19 graph, GCN vs. identity vs. ridge
POST-FREEZE SENSITIVITY:              CRC microbiome CLR / rank / presence-absence representations
```

Metabolome-only is **not** a secondary analysis or "stress test" in the final reporting framework — it is co-primary with microbiome-only, run through the identical protocol. It carries its own, explicit limitation (cross-cohort quantitative-scale equivalence unverified) rather than being demoted. See `final_analysis/README.md` for full detail.

## 10. Reproducibility design

Fixed random seeds; nested cross-validation with all trainable preprocessing (missingness/prevalence filtering, imputation, scaling, hyperparameter selection) fit only on the training partition; decision-score (not saturated-probability) ranking metrics; Youden thresholds selected on source cross-fitted out-of-fold decision scores and frozen before external evaluation; 2,000-resample stratified participant bootstrap CIs with BH-FDR across prespecified comparison families; sha256-hash-tracked authoritative output registry; a consolidated automated QA suite; a post-freeze code-change/equivalence-verification layer. See `docs/reproducibility.md`.
