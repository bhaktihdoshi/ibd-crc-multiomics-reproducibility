# Corrected-v2 Stage 4C-D-E Scope Policy

This policy prevents downstream corrected-v2 reruns from reintroducing obsolete or overbroad scope.

## Predictive Tasks

Corrected-v2 predictive modelling is restricted to:

- pooled IBD versus control, where CD and UC are both coded as disease-positive;
- CRC versus control.

CD and UC counts are retained only for cohort description and class-balance context. CD-vs-control and UC-vs-control are not corrected-v2 predictive tasks.

## MOFA

- Use corrected-v2 microbial genus matrices.
- Use microbiome and metabolome as omic views.
- Do not include clinical variables as MOFA views.
- Phenotype annotation should use pooled IBD-versus-control and CRC-versus-control only.
- Remove subtype-specific MOFA phenotype tests from the corrected-v2 plan.

## sCCA

- Use corrected-v2 microbial genus matrices.
- Use microbiome and metabolome as the two cross-view matrices.
- Do not include clinical variables as sCCA views.
- Fold-local centring/scaling must be verified before rerun.
- Disease labels must remain outside fitting and tuning.

## Clinical Variables

Clinical variables are not used as corrected-v2 common predictive baselines, MOFA/sCCA views or biological graph nodes. The reason is methodological rather than biological dismissal: clinical variables were not sufficiently complete and harmonised across the four public cohorts for a fair common cross-cohort baseline.

## NJC19 Terminology

NJC19 edges should be described as curated production, consumption or transport relations. Preserve `relation_type` or equivalent provenance fields wherever available.

Do not describe NJC19 edges as disease associations. Consumption/import relations should not be converted into negative adjacency weights.

## Primary Graph Policy

The corrected-v2 primary graph is:

- unsigned;
- undirected;
- feature-level, not participant-level;
- composed of metabolite-metabolite KEGG pathway co-membership plus optional NJC19 microbe-metabolite relations;
- free of phenotype-derived, prediction-derived, coefficient-derived or correlation-derived edges.

Do not add microbe-microbe edges to the primary graph.

Required graph modes:

1. identity;
2. KEGG-only;
3. KEGG + NJC19.

Connectivity is a graph-coverage result, not proof that an edge is active in the study participants.
