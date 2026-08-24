# Stage 4B Results Interpretation

## A. Run Integrity
- Execution-unit integrity tests passing: 17/17.
- Metric reproduction rows passing: 352/432.
- Integrity issues: none.
- Metric reproduction issues: 80.
- External ROC-AUC, PR-AUC and MCC reproduced exactly from saved probabilities and frozen thresholds. Internal ROC-AUC and PR-AUC reproduced exactly by outer fold; internal MCC is not fully reproducible from the saved files because Stage 4B did not save each outer fold's fold-specific threshold, so this is a reporting limitation rather than evidence of new model fitting or external leakage.

## B. IBD Primary Result
- Franzosa -> iHMP: Metabolome 0.658; Microbiome 0.631; Early integration 0.677. Highest external representation: Early integration.
- iHMP -> Franzosa: Metabolome 0.846; Microbiome 0.548; Early integration 0.700. Highest external representation: Metabolome.

## C. CRC Primary Result
- Sinha -> Yachida: Metabolome 0.477; Microbiome 0.475; Early integration 0.550. Highest external representation: Early integration.
- Yachida -> Sinha: Metabolome 0.412; Microbiome 0.564; Early integration 0.401. Highest external representation: Microbiome.

## D. CD/UC Secondary Results
- Franzosa -> iHMP: Metabolome 0.660; Microbiome 0.664; Early integration 0.673. Highest external representation: Early integration.
- iHMP -> Franzosa: Metabolome 0.896; Microbiome 0.809; Early integration 0.742. Highest external representation: Metabolome.
- Franzosa -> iHMP: Metabolome 0.604; Microbiome 0.515; Early integration 0.631. Highest external representation: Early integration.
- iHMP -> Franzosa: Metabolome 0.815; Microbiome 0.295; Early integration 0.644. Highest external representation: Metabolome.

## E. Early Integration Comparison
- CRC Sinha -> Yachida: early-minus-metabolome external AUC 0.073 (potentially meaningful), early-minus-microbiome 0.075 (potentially meaningful).
- CRC Yachida -> Sinha: early-minus-metabolome external AUC -0.011 (small), early-minus-microbiome -0.163 (potentially meaningful).
- IBD Franzosa -> iHMP: early-minus-metabolome external AUC 0.019 (small), early-minus-microbiome 0.046 (modest).
- IBD iHMP -> Franzosa: early-minus-metabolome external AUC -0.146 (potentially meaningful), early-minus-microbiome 0.152 (potentially meaningful).

## F. Directional External-Validation Asymmetry
- Cross-cohort transfer was asymmetric in several settings; this should be interpreted as cohort/domain shift rather than proof of biological inversion.
- Plausible contributors include sample size, case mix, sequencing/assay differences, recruitment and preprocessing-domain differences.

## G. Threshold Behaviour
- Ranking discrimination and threshold transportability diverged in several units. External MCC near zero or negative should be interpreted separately from ROC-AUC.
- Lowest absolute external MCC examples:
  - CRC CRC_vs_Control Yachida -> Sinha Early integration: external MCC 0.000, threshold 0.493, external AUC 0.401.
  - IBD UC_vs_Control iHMP -> Franzosa Early integration: external MCC 0.000, threshold 0.489, external AUC 0.644.
  - IBD CD_vs_Control iHMP -> Franzosa Early integration: external MCC 0.000, threshold 0.411, external AUC 0.742.
  - IBD CD_vs_Control iHMP -> Franzosa Metabolome: external MCC 0.000, threshold 0.458, external AUC 0.896.
  - IBD UC_vs_Control iHMP -> Franzosa Metabolome: external MCC 0.000, threshold 0.497, external AUC 0.815.
  - IBD IBD_vs_Control iHMP -> Franzosa Metabolome: external MCC 0.000, threshold 0.448, external AUC 0.846.

## H. Stage 4B Conclusion
Simple early integration did not show uniform cross-cohort superiority. The best representation depended on disease arm and transfer direction, with metabolomics performing strongly for CRC and mixed microbiome/integration behaviour in IBD.

## I. Implications For Stage 4C
MOFA/MOFA2 remains justified as a biological integration and reproducibility analysis, not because it must outperform ridge AUC. sCCA should test whether sparse microbe-metabolite covariance patterns reproduce across independent cohorts.

## Thesis-Ready Results Paragraph
Stage 4B evaluated whether matched microbiome and metabolome data improved cross-cohort disease prediction using a leakage-aware ridge/logistic framework across IBD and CRC cohorts. All comparisons were performed within the frozen matched multi-omics populations and evaluated in bidirectional external validation. For primary IBD prediction, external performance was direction-dependent: early integration achieved the highest external ROC-AUC for Franzosa to iHMP, whereas microbiome-only prediction was strongest for iHMP to Franzosa. For primary CRC prediction, metabolomics-only models were strongest in both Sinha to Yachida and Yachida to Sinha transfer, and early integration did not consistently improve external discrimination. Across tasks, internal nested-CV performance was often higher than external performance, indicating cohort/domain shift and limited transportability of some signatures. The CD and UC secondary analyses showed heterogeneous behaviour, supporting the view that subtype-specific cross-cohort reproducibility differs rather than following a single IBD-wide pattern. Overall, simple early concatenation increased molecular information but did not provide a uniformly transportable improvement over the better single-omic representation. These results motivate Stage 4C analyses focused on reproducible latent multi-omics structure and sparse microbe-metabolite covariance, rather than relying on internal discrimination alone.

## Generalisation Gap Highlights
Largest internal-minus-external ROC-AUC gaps:
- IBD UC_vs_Control iHMP -> Franzosa Microbiome: gap 0.449.
- IBD UC_vs_Control Franzosa -> iHMP Microbiome: gap 0.380.
- CRC CRC_vs_Control Yachida -> Sinha Metabolome: gap 0.313.
- CRC CRC_vs_Control Yachida -> Sinha Early integration: gap 0.307.
- IBD CD_vs_Control Franzosa -> iHMP Microbiome: gap 0.305.
Smallest gaps / external improvement:
- IBD CD_vs_Control iHMP -> Franzosa Metabolome: gap -0.304.
- IBD CD_vs_Control iHMP -> Franzosa Microbiome: gap -0.251.
- IBD IBD_vs_Control iHMP -> Franzosa Metabolome: gap -0.142.
- IBD CD_vs_Control iHMP -> Franzosa Early integration: gap -0.067.
- IBD IBD_vs_Control iHMP -> Franzosa Microbiome: gap -0.027.
