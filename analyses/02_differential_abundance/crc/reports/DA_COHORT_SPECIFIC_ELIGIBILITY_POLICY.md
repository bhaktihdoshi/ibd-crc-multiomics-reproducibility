# DA Cohort-Specific Eligibility Policy

This policy separates biological association testing from predictive external-validation feature constraints. It is an audit proposal only; no expanded DA has been run.

## Eligible for cohort-specific DA

A feature is eligible for cohort-specific DA if all conditions are met:

1. The source assay feature is present in the cohort metabolomics matrix.
2. The sample can be linked to the cohort metadata and current disease/control analysis labels.
3. The feature has a deterministic, source-supported identity suitable for reporting: valid HMDB or valid KEGG, with high-confidence/source-supported annotation where available.
4. The feature is not an unresolved one-feature-to-multiple-canonical-entity ambiguity. If multiple assay features map to one canonical entity, use the existing deterministic assay-feature selection rule or report them separately only with clear provenance.
5. The feature has <=30% missingness in the analysis sample set for that cohort.
6. The feature is non-constant among observed values in that cohort analysis set.
7. The current cohort-specific transformation/effect model can be applied without changing the interpretation.

The feature does **not** need to be present in the partner cohort to be eligible for cohort-specific DA.

## Eligible for cross-cohort replication

A DA feature is eligible for within-disease replication only if the same canonical identifier was actually testable in both disease cohorts. Replication should be reported on raw p-value/FDR significance and effect-direction concordance, using the predeclared primary definition: FDR-significant in both cohorts and same effect direction.

## Eligible for KEGG ORA

A feature contributes to KEGG ORA only if it was actually tested in the relevant DA family and has deterministic KEGG pathway mapping. The ORA background is all tested DA features with deterministic KEGG mapping in that cohort/contrast.

## Eligible for Stage 3

Stage 3 primary integration remains restricted to the approved four-cohort comparable feature universe, currently the strict 56 HMDB metabolites. Expanded cohort-specific DA should later be subset back onto the Stage 3 common space where needed.
