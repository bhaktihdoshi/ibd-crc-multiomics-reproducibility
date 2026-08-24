# Stage 3 Setup and Feasibility Report

## 1. Authoritative upstream inputs found

Stage 3 input manifest: `manifests/stage3_upstream_input_manifest.csv`.

CRC has an authoritative output manifest at `thesis_preprocessing_workflow/crc_pipeline/stage2/reports/CRC_STAGE2_AUTHORITATIVE_OUTPUT_MANIFEST.csv`. IBD live final outputs were identified from the current IBD pipeline directories and logged as live final outputs where present.

## 2. Four-cohort metabolite overlap counts

- Franzosa: 166
- iHMP: 166
- Sinha: 353
- Yachida: 418
- Franzosa ∩ iHMP: 166
- Sinha ∩ Yachida: 110
- Four-way Franzosa ∩ iHMP ∩ Sinha ∩ Yachida: 56

Tables: `stage3_metabolite_overlap_summary.csv`, `stage3_pairwise_overlap_matrix.csv`, `stage3_four_cohort_shared_metabolites.csv`.

## 3. Annotation compatibility

- Four-way HMDB count: 56
- KEGG mapped count from existing deterministic mappings: 56
- KEGG unmapped count: 0
- Ambiguous count: 2

No internet/API annotation enrichment was performed.

## 4. Raw-integration risk assessment

Raw patient-level pooling is classified as **HIGH_DOMAIN_SHIFT_RISK** and is not recommended as the primary Stage 3 representation.

## 5. Recommended integration representation

Primary: disease-level metabolite effect profiles.

Secondary/sensitivity: within-cohort standardised patient-level abundance, only after strict GO checks.

## 6. Proposed primary analyses

1. Four-cohort metabolite intersection.
2. Cross-disease effect-size integration.
3. Directional concordance.
4. Replicated shared metabolite identification.

## 7. Proposed secondary analyses

5. Pathway/module comparison.
6. Yachida ordered stage exploration as cross-sectional disease states.
7. Latent-space visualisation if justified.

## 8. Graph-analysis feasibility

Patient-level graph analysis is **NO_GO_FOR_PATIENT_LEVEL_GRAPH at setup stage**. A disease-state effect-profile graph is more defensible if the shared effect table is sufficiently populated.

## 9. Major scientific risks

- Cohort/platform confounding.
- Small four-way shared metabolite set.
- Heterogeneous preprocessing history.
- Cross-sectional design cannot prove progression.
- Annotation/pathway coverage may be incomplete.

## 10. Go/no-go recommendations

See `reports/STAGE3_GO_NO_GO_CRITERIA.md`.

## 11. Future heavy commands, not run now

```bash
python scripts/run_stage3.py effect-integration --allow-heavy
python scripts/run_stage3.py latent --allow-heavy
python scripts/run_stage3.py graph --allow-heavy
python scripts/run_stage3.py clustering --allow-heavy
```

These commands are scaffolded for future use only and should not be run until supervisor approval and GO criteria are satisfied.
