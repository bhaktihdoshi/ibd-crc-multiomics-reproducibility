# Stage 4C MOFA Results Interpretation

## A. Model Validity And Convergence
- MOFA_IBD_VALID: YES.
- MOFA_CRC_VALID: YES.
- Both models were interpreted only from saved HDF5/extracted outputs; no model was refitted.

## B. Nondegenerate Factors
- Rule: finite scores/loadings, nonzero score/loading variance, and mean factor R2 >= 0.1% across groups/views.
- IBD nondegenerate factors: 5/8.
- CRC nondegenerate factors: 3/8.

## C. Multi-View Versus View-Specific Structure
| disease   | category            |   n |
|:----------|:--------------------|----:|
| CRC       | METABOLOME_DOMINANT |   2 |
| CRC       | MICROBIOME_DOMINANT |   1 |
| IBD       | METABOLOME_DOMINANT |   2 |
| IBD       | MULTI_VIEW          |   3 |

## D. Strongest Post-Hoc Phenotype Associations
| disease   | comparison     | cohort_scope              | factor   |   effect_median_difference |       raw_p |      bh_fdr | direction         |
|:----------|:---------------|:--------------------------|:---------|---------------------------:|------------:|------------:|:------------------|
| IBD       | CD_vs_Control  | POOLED_GROUP_AWARE_SCORES | Factor1  |                   1.29804  | 2.78581e-13 | 2.22865e-12 | higher_in_disease |
| IBD       | IBD_vs_Control | POOLED_GROUP_AWARE_SCORES | Factor1  |                   1.13707  | 1.41148e-12 | 1.12918e-11 | higher_in_disease |
| IBD       | CD_vs_Control  | Franzosa                  | Factor1  |                   1.68795  | 5.5501e-12  | 4.44008e-11 | higher_in_disease |
| IBD       | IBD_vs_Control | Franzosa                  | Factor1  |                   1.36797  | 7.35731e-10 | 5.88585e-09 | higher_in_disease |
| IBD       | CD_vs_Control  | POOLED_GROUP_AWARE_SCORES | Factor3  |                  -1.0834   | 6.79104e-09 | 2.71642e-08 | lower_in_disease  |
| IBD       | IBD_vs_Control | Franzosa                  | Factor4  |                  -1.02268  | 1.43557e-07 | 5.74229e-07 | lower_in_disease  |
| IBD       | CD_vs_Control  | Franzosa                  | Factor3  |                  -1.48476  | 1.75502e-07 | 7.02009e-07 | lower_in_disease  |
| IBD       | IBD_vs_Control | POOLED_GROUP_AWARE_SCORES | Factor3  |                  -0.747736 | 7.30301e-07 | 2.9212e-06  | lower_in_disease  |

## E. Group-Aware Reproducibility Limitation
- W/loadings are shared by the group-aware MOFA model and should not be reported as independently rediscovered in both cohorts.
- Z/factor scores and variance explained are group-specific and can be compared across cohort groups.
- Independent loading reproducibility would require separate-cohort MOFA or a separate method such as sCCA.

## F. Relationship To Stage 4B
Stage 4B showed direction-dependent predictive transportability and inconsistent benefit from simple early integration. Stage 4C asks a different question: whether unsupervised latent microbiome-metabolome structure provides coherent biological interpretation despite those predictive inconsistencies.

## G. Separate-Cohort MOFA Sensitivity
SEPARATE_COHORT_MOFA_SENSITIVITY = RECOMMENDED
Separate-cohort MOFA would materially strengthen independent loading reproducibility claims because group-aware MOFA estimates shared W/loadings by design. However, sCCA is the next lighter-weight independent covariance reproducibility analysis and should be considered before adding four additional MOFA fits.

## H. sCCA Implication
sCCA should test whether sparse microbe-metabolite covariance components reproduce independently when each cohort is analysed separately. It complements group-aware MOFA because it can provide cohort-specific loading solutions.

## Thesis-Ready Results Paragraph
Group-aware MOFA models were successfully audited for both IBD and CRC using matched microbiome and metabolome views without using phenotype labels during factor discovery. The IBD and CRC HDF5 models were readable, contained the expected two cohort groups, two molecular views and eight latent factors, and produced finite factor scores and loadings. Using a predeclared nondegeneracy rule based on finite scores/loadings and mean factor R2 of at least 0.1% across groups/views, 5 IBD factors and 3 CRC factors were considered nondegenerate. The resulting latent structure was not uniformly multi-view: several factors were dominated by metabolome variance, while a smaller subset showed measurable contribution from both microbiome and metabolome views. Post-hoc phenotype tests were then applied only to the fixed unsupervised factor scores. In IBD, the clearest disease separation was observed for CD/control and IBD/control contrasts on early factors, whereas UC/control effects were weaker and more factor-specific. In CRC, phenotype associations were more heterogeneous, consistent with the weaker cross-cohort transportability observed in Stage 4B. Because group-aware MOFA shares feature loadings across cohort groups, loading concordance should be described as model-shared structure rather than independent replication. The strongest biological value of Stage 4C is therefore interpretability of latent cross-view structure and group-specific score/variance behaviour, rather than a replacement for Stage 4B predictive validation.

## Output Tables And Figures
- tables/stage4c_mofa_non_degenerate_factors.csv
- tables/stage4c_mofa_factor_associations.csv
- tables/stage4c_mofa_cross_cohort_reproducibility.csv
- figures/stage4c_mofa_variance_explained.png
- figures/stage4c_mofa_factor_associations.png
- figures/stage4c_mofa_top_loadings.png
