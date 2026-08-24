# Stage 4A Final Frozen Design

## IBD
- Franzosa matched n: 137 with class counts {'Control': 56, 'CD': 45, 'UC': 36}.
- iHMP matched n: 91 with class counts {'CD': 39, 'UC': 26, 'Control': 26}.
- Microbial rank: genus.
- Shared microbial p: 50.
- Shared metabolite p: 166.
- External directions: Franzosa -> iHMP; iHMP -> Franzosa.
- Stage 4B status: STAGE4B_IBD_GO.

## CRC
- Sinha matched n: 131 with class counts {'Control/Healthy': 89, 'CRC': 42}.
- Yachida matched n: 252 with class counts {'Control/Healthy': 127, 'CRC': 125}.
- Microbial rank: genus.
- Shared microbial p: 52.
- Shared metabolite p: 110.
- External directions: Sinha -> Yachida; Yachida -> Sinha.
- Stage 4B status: STAGE4B_CRC_GO.

## Methods
- Microbiome transformation: training-fold prevalence, zero handling, CLR and standardisation.
- Metabolome preprocessing: training-fold missingness filtering, constant removal, imputation and standardisation.
- Nested CV: 5x5 stratified where class counts permit, ridge C-grid only.
- Metrics: primary ROC-AUC; secondary PR-AUC; MCC threshold diagnostic; balanced accuracy supplementary.
- Threshold rule: training-only out-of-fold Youden/MCC-compatible threshold frozen before external validation.

## Stage 4C
- MOFA/MOFA2: unsupervised, group-aware/separate-cohort reproducibility design only.
- sCCA: optional secondary cross-view sensitivity with 1-2 components.

## Stage 4D
- Graph design: prior-knowledge topology independent of disease labels.
- GCN formulation: later exploratory graph model only after Stage 4B ridge benchmarks.

## Final Feasibility Class
A. FOUR_COHORT_MULTIOMICS_READY
