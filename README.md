# IBD and CRC Multi-Omics Transportability — MSc Thesis Analysis Repository

## At a glance

This thesis tests whether **microbiome** and **metabolome** case-control signatures for **inflammatory bowel disease (IBD)** and **colorectal cancer (CRC)** transport across independently processed cohorts: **Franzosa**, **iHMP/IBDMDB** (IBD), **Sinha**, **Yachida** (CRC).

- **Final analysis**: [`final_analysis/`](final_analysis/) (Stage 4 predictive/integration) + [`analyses/03_cross_cohort_replication/`](analyses/03_cross_cohort_replication/) (Stage 2/3 biological replication). Start with [`final_analysis/README.md`](final_analysis/README.md) for what's authoritative.
- **Microbiome-only and metabolome-only are the two co-primary single-omic analyses** — neither is secondary to the other (early integration is a separate comparator; see §9 below).
- **Sensitivity / exploratory / historical work is clearly segregated**: [`results/sensitivity/`](results/sensitivity/) (post-freeze CRC representation checks) vs. [`exploratory/`](exploratory/) (alternative approaches and the pre-correction pipeline — **not final results**, see its warning README).
- **Data**: no raw/third-party cohort data is redistributed here — see [`data/README.md`](data/README.md) for exact sources, accessions, and retrieval instructions.
- **Reproduce it**: see [`docs/reproducibility.md`](docs/reproducibility.md) for exact commands.
- **Every number's source**: [`docs/file_provenance_map.csv`](docs/file_provenance_map.csv) and, for the thesis's exact figures/tables, [`final_analysis/tables/final_stage4_figure_source_values.csv`](final_analysis/tables/final_stage4_figure_source_values.csv) / [`analyses/03_cross_cohort_replication/manifests/stage2_3_manuscript_claim_provenance.csv`](analyses/03_cross_cohort_replication/manifests/stage2_3_manuscript_claim_provenance.csv).

Full detail in the 20 sections below.

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
| Sinha et al. | CRC (CRC/Control) | 16S rRNA amplicon | — (microbiome only in this analysis) |
| Yachida et al. | CRC (CRC/Control), staged | Shotgun metagenomics | CE-TOFMS / LC-MS |

Sinha (16S) vs. Yachida (shotgun) is platform-confounded with cohort for CRC — this limitation is carried through the analysis and reporting, not resolved by scaling.

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

## 11. How to reproduce analyses

See `docs/reproducibility.md` and `final_analysis/README.md` for exact commands. In brief: the Stage 4 pipeline is run via the scripts in `final_analysis/code/` (`run_stage4a_corrected_v2.py` → `run_stage4e_gcn_corrected_v2.py`, then `freeze_final_stage4_results.py`); QA via `run_final_stage4_qa.py`; the CRC sensitivity branch via `run_stage4b_crc_microbiome_sensitivity.py`. Raw data must first be obtained per `data/README.md` — it is not redistributed in this repository.

## 12. Where final thesis outputs live

`final_analysis/` (Stage 4) and `analyses/03_cross_cohort_replication/` (Stage 2/3). Thesis-facing copies of the final figures/tables are also collected under `results/final/`.

## 13. Where exploratory/historical analyses live

`exploratory/` — alternative model families, alternative preprocessing, diagnostic analyses, discarded/infeasible ideas, and the full pre-correction Stage 4 pipeline (`exploratory/historical_analyses/stage4_pre_correction_v1/`) and IBD-only predictive-modelling history. **These are not final results** — see the warning in `exploratory/README.md`.

## 14. Data availability / accessions

No raw participant-level data is redistributed in this repository. See `data/README.md` for each cohort's original publication, accession/repository, and how to obtain it.

## 15. Software environment

See `docs/software_environment.md` (general) and `docs/software_environment_stage4.md` (exact Stage 4 package versions).

## 16. Known limitations

See `docs/limitations.md`. Headline items: metabolomics cross-cohort quantitative-scale equivalence unverified; CRC microbiome platform (16S vs. shotgun) confounded with cohort; sCCA fold correlations are tuning diagnostics, not unbiased estimates; GCN results are specific to the tested graph policies/architectures, not general evidence about GNN capability; no fully harmonised four-cohort clinical covariate set exists.

## 17. Citation instructions

See `CITATION.cff`.

## 18. Thesis citation placeholder

> [Author]. (2026). *[Thesis title]*. MSc thesis, [Institution]. [DOI/URL once available.]

## 19. Contact / author

[Author name / contact email — to be filled in by the author before publication.]

## 20. Licence / data-licence disclaimer

**The code in this repository is released under the MIT Licence** (see `LICENSE`). **This applies only to the original analysis code, scripts, and documentation authored for this project.** It does **not** grant any redistribution rights over, and does not apply to:

- the third-party cohort datasets (Franzosa, iHMP/IBDMDB, Sinha, Yachida) — not redistributed here, see `data/README.md`;
- third-party reference resources (KEGG, HMDB, NJC19, MiMeDB) — not redistributed here;
- the publications cited in `docs/references_and_data_sources.md`;
- the dissertation document itself, once added (subject to the author's institution's own policies).

See `docs/data_provenance.md` for the full distinction between the code licence, third-party data licences, and third-party resources.

---

**For examiners**: final thesis claims should be traced through the authority/provenance files described in `final_analysis/README.md` and `analyses/03_cross_cohort_replication/README.md`, not inferred from file location alone. Historical results in `exploratory/` may differ from the final analysis and must not be cited as thesis evidence.
