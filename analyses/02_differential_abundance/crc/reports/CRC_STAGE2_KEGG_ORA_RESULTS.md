# CRC Stage 2 KEGG ORA Results

## Methods

CRC KEGG pathway over-representation analysis was run as a controlled pathway-only reopening of Stage 2. Existing CRC differential-abundance results were consumed read-only. No model fitting, nested cross-validation, external prediction generation, threshold optimisation, calibration, feature importance, DA refitting, PCA/UMAP, graph construction, clustering or Stage 3 integration was performed.

## Background Definition

For each cohort/contrast, the ORA background was all DA-tested metabolites with deterministic KEGG mapping from the existing CRC HMDB/KEGG crosswalk. This is cohort-specific and contrast-specific, not all KEGG compounds, not all human metabolites, not the Stage 3 56-metabolite set, and not only significant metabolites.

## Statistical Test

The test is a one-sided hypergeometric right-tail over-representation test. Fold enrichment is `(k / n) / (K / N)`. BH correction is applied separately within cohort × comparison × direction × annotation source. Pathways require at least 3 background compounds and at least 2 significant compounds.

## Main Results

- Sinha BH-FDR-significant pathways: **0**
- Yachida BH-FDR-significant pathways: **0**
- Replicated CRC pathways by strict definition: **0**

## Interpretation Cautions

Pathways are described as enriched among CRC-associated metabolites or over-represented in the significant metabolite set. These results do not prove pathway activation, causal mechanism or clinical biomarker validity.
