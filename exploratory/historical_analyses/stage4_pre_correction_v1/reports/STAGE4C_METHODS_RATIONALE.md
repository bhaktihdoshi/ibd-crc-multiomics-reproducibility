# Stage 4C Methods Rationale

Stage 4C is designed as biological integration rather than another classifier benchmark. Stage 4B remains the frozen predictive baseline.

## Primary method: MOFA / MOFA2

MOFA is primary because it models shared and view-specific latent variation across molecular layers without using disease labels. The molecular views are microbiome and metabolome. Clinical variables and disease labels are annotations only, used after unsupervised factor learning for descriptive association tests.

MOFA does not need to outperform ridge ROC-AUC to add value. Its purpose is to test whether microbiome-metabolome latent structure is reproducible across independent cohorts and whether biologically interpretable factors align with IBD/CRC phenotype annotations.

## Secondary method: penalised sparse CCA

Penalised sparse canonical correlation analysis is a secondary sensitivity analysis. It asks whether sparse linear combinations of microbial and metabolite features maximise cross-view association within each cohort and whether those independently fitted sparse loading structures reproduce across cohorts. The implementation uses `PMA::CCA`, where `penaltyx` and `penaltyz` impose sparsity constraints inside the canonical-correlation optimisation objective.

Reference: Witten DM, Tibshirani R, Hastie T. A penalized matrix decomposition, with applications to sparse principal components and canonical correlation analysis. Biostatistics. 2009;10(3):515-534. doi:10.1093/biostatistics/kxp008

Post-hoc thresholded dense sklearn CCA is not used as the thesis sCCA analysis.

## Why autoencoders are not primary

Autoencoders are flexible but less stable and less interpretable for the current n/p setting. They would require additional architecture choices and validation that are not justified before establishing simpler, interpretable latent integration.

## Why MEFISTO is not primary

MEFISTO is designed for structured temporal or spatial variation. The current central Stage 4C comparison is cross-sectional cross-cohort reproducibility rather than a shared longitudinal trajectory. iHMP longitudinality was deliberately not used as the primary Stage 4 matched multi-omics design.

## Leakage policy

No disease-label feature selection is performed before MOFA or sCCA. Disease labels are used only after latent factors/components are learned to annotate the unsupervised structure. No external prediction/projection is implemented unless the actual MOFA2 implementation supports it defensibly.
