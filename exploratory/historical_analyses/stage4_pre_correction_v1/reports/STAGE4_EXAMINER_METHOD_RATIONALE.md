# Stage 4 Examiner Method Rationale

Metabolomics baseline: establishes a single-omic reference and transportability.
Microbiome baseline: tests whether the second stool-derived layer independently carries reproducible disease information.
Early integration: simplest fair multi-omics comparator.
MOFA: evaluates shared and view-specific latent multi-omic variation.
sCCA: sensitivity analysis for sparse cross-view covariance.
GCN: tests whether prior biological topology adds interpretability beyond unstructured and latent integration.
Autoencoder is not prioritised because it sacrifices interpretability without answering an additional necessary question. MEFISTO is not prioritised because the CRC state ordering is cross-sectional, not longitudinal time. Patient graph is not preferred because domain shift is high and it is less biologically direct than a fixed molecular graph. Causal GNN is not justified because observational cross-sectional datasets do not identify causal direction.
