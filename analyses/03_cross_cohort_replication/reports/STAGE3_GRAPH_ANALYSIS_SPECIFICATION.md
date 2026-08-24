# Stage 3 Graph Analysis Specification

No graph is constructed in this setup pass.

## Candidate graph designs

Preferred future graph, if justified: disease-state effect-profile graph, where nodes are disease contrasts/states and features are shared HMDB effect profiles.

Secondary exploratory graph: patient-similarity graph using within-cohort standardised four-way shared metabolites, only if feature count and cohort-confounding checks pass.

## Future methods

- Feature space: four-way shared HMDB metabolites, optionally filtered by annotation confidence and missingness.
- Distance: correlation distance for effect profiles; Euclidean/correlation for standardised patient features.
- Scaling: within-cohort only for patient-level analysis; no pooled raw scaling.
- k selection: pre-specified grid with stability assessment; no cherry-picking.
- Community detection: Leiden or Louvain only after kNN graph construction is justified.
- Stability: repeat across seeds and k; assess adjusted Rand/index stability.
- Confounding: test whether clusters are driven by cohort/platform before disease interpretation.
- Enrichment: disease/stage enrichment by permutation or exact tests where valid.
- Clinical profiling: age/sex/BMI summaries, with cautious interpretation.
- Pathway interpretation: attach existing HMDB/KEGG/pathway annotations only after feature-level results are stable.

## Stop conditions

Do not proceed to patient graph modelling if:
- four-way shared metabolite space is too small
- graph separates primarily by cohort/platform
- shared features show extreme cross-cohort distribution shift
- sample-level normalization is not comparable
- HS is required to create an ordered CRC stage axis
- graph communities cannot be interpreted without overclaiming progression
