# Stage 4A Final Preprocessing Policy

Microbiome primary pipeline: training-fold prevalence rule, zero handling with a fixed pseudocount or training-derived compatible policy, CLR transformation, then training-fold standardisation.
Metabolome pipeline: training-fold missingness filtering, constant-feature removal, imputation and standardisation, consistent with the validated metabolomics analyses.
Early integration: process microbiome and metabolome views separately, then concatenate transformed views inside the same training-fold pipeline.
No global prevalence filtering, missingness filtering, imputation, scaling or feature selection is allowed before Stage 4B cross-validation.
