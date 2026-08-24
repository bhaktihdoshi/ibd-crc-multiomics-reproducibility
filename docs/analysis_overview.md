# Analysis Overview

## Main inferential pipeline

```
RAW/SOURCE DATA
        │  (Franzosa, iHMP/IBDMDB, Sinha, Yachida — see data/README.md)
        ▼
COHORT-SPECIFIC PREPROCESSING
        │  (preprocessing/ibd/, preprocessing/crc/, preprocessing/microbiome/)
        ▼
QC + MATCHING + HARMONISATION
        │  (participant/sample matching, exclusions, genus/metabolite ID harmonisation)
        ▼
STAGE 2/3 BIOLOGICAL REPLICATION
        │  (analyses/03_cross_cohort_replication/ — differential abundance,
        │   within-disease concordance, cross-disease concordance,
        │   Yachida ordered-state trajectory analysis)
        ▼
STAGE 4A — MATCHED MULTI-OMICS FEATURE SPACE
        │  (authoritative genus harmonisation, shared genus space,
        │   metabolite HMDB matching; final_analysis/)
        ▼
STAGE 4B ────────────────────────────────────────┐
  metabolome-only ─┐                              │
                    ├─ CO-PRIMARY                  │
  microbiome-only ─┘                              │
        │                                          │
        └─ early integration (comparator) ─────────┘
        ▼
STAGE 4C
  MOFA (descriptive structural integration)
  sCCA (descriptive; tuning-CV diagnostics, not held-out estimates)
        ▼
STAGE 4D
  KEGG / NJC19 prior-knowledge graph construction
  (identity, KEGG-only, broad KEGG+NJC19, conservative KEGG+NJC19)
        ▼
STAGE 4E
  GCN vs. identity ablation vs. matched ridge comparator
  (5 outer folds × 5 seeds, participant-level OOF, paired bootstrap + BH-FDR)
        ▼
POST-FREEZE SENSITIVITY + QA + PROVENANCE
  (CRC microbiome CLR / rank / presence-absence sensitivity;
   consolidated QA suite; code-patch and equivalence verification;
   co-primary reporting addendum)
        ▼
FINAL THESIS FIGURES/TABLES
  (results/final/, final_analysis/tables, final_analysis/figures)
```

## Side branch — NOT part of the final inferential pipeline

```
                    ┌───────────────────────────────────────────┐
RAW/SOURCE DATA ────┤  EXPLORATORY / HISTORICAL WORK              │
                    │  - alternative model families               │
                    │    (ridge/elastic-net/random-forest/GBM)     │
                    │  - earlier IBD-only / CRC-only classifiers   │
                    │  - pre-correction Stage 4 pipeline (v1)      │
                    │  - alternative preprocessing approaches      │
                    │  - discarded/infeasible ideas                │
                    │  - clustering/PCA/EDA                        │
                    │                                               │
                    │  see exploratory/README.md                   │
                    └───────────────────────────────────────────┘
```

Material in the side branch informed the final design or was tried and superseded — it is preserved for scientific transparency but **must not be read as thesis evidence**. See `exploratory/README.md` and the "historical files NOT to use" section of `final_analysis/README.md`.

## Two co-primary single-omic analyses, not a principal/secondary pair

A note on why Stage 4B is drawn with microbiome and metabolome side-by-side above: both representations were evaluated under the identical nested-CV / decision-score / bootstrap protocol on matched participants with reciprocal external validation. The final thesis reporting framework treats them as co-primary. Early integration sits below them as a separate comparator answering a different question (does combining the two co-primary layers help?), not as a third co-primary analysis. See `final_analysis/README.md` §2 for the full interpretation rules.
