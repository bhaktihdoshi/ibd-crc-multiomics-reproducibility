# Stage 3 Cross-Disease Analysis Plan

Terminology: Stage 3 will use a cross-sectional disease-state framework, not observed longitudinal progression.

## Planned primary analyses, future execution only

1. Effect-size correlation using complete shared HMDB metabolites only:
   - IBD vs CRC
   - CD vs CRC
   - UC vs CRC
   - Spearman correlation, with confidence intervals if sample size supports it.

2. Directional concordance:
   - positive in both IBD and CRC
   - negative in both
   - opposite direction
   - IBD-specific
   - CRC-specific
   - non-significant in both

3. Replication-aware subset:
   - Franzosa/iHMP replicated IBD evidence
   - Sinha/Yachida replicated CRC evidence
   - prioritize four-way shared metabolites with consistent cohort-level evidence.

4. Predictive/DA concordance:
   - compare signed logistic coefficients and DA directions only where both are defined on the same HMDB feature and direction is scientifically interpretable.

No new regression, machine learning, dimensionality reduction, pathway fitting or graph analysis is executed by this setup pass.
