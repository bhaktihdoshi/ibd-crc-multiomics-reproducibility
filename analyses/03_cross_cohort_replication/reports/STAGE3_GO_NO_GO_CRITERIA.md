# Stage 3 GO / NO-GO Criteria

## Current lightweight setup assessment

- Four-way HMDB overlap: **56** metabolites.
- KEGG mapped among four-way shared: **56**.
- Ambiguous among four-way shared: **2**.

## Analysis-family decisions

| Analysis family | Current decision | Rationale |
|---|---|---|
| Four-cohort identifier overlap | GO | Lightweight deterministic HMDB set operation already completed. |
| Disease-level effect integration | GO_WITH_RESTRICTIONS | Use existing/future standardized effects only; no name-only matching. |
| Directional concordance | GO_WITH_RESTRICTIONS | Must use complete shared HMDB pairs and FDR-aware labels. |
| Pathway/module comparison | GO_WITH_RESTRICTIONS | Only if annotation coverage is adequate; avoid overclaiming if KEGG coverage is sparse. |
| Yachida ordered stage exploration | GO_WITH_RESTRICTIONS | Cross-sectional state ordering only; HS excluded. |
| Latent-space visualisation | GO_WITH_RESTRICTIONS | Only after feature-count/domain-shift checks. |
| Patient-level graph/clustering | NO_GO_FOR_PATIENT_LEVEL_GRAPH at setup stage | Known CRC domain shift and heterogeneous preprocessing make direct graph modelling high risk before further diagnostics. |

## Required questions before heavy execution

- Is the four-way metabolite intersection sufficiently large for the proposed method?
- Are canonical identifiers reliable and non-ambiguous?
- Are effect-size definitions comparable across disease pipelines?
- Is direct sample pooling defensible, or should it remain avoided?
- Is cohort/platform confounding likely to dominate?
- Are pathway resources adequate?
- Can Yachida stage analyses be interpreted safely as cross-sectional state comparisons?
- Is a graph biologically justified rather than visually attractive only?

## Mapping Expansion Audit Update

The mapping-expansion audit retained the strict four-way HMDB count of 56 and found 0 additional high-confidence rescued entities. This does not materially change the prior GO/NO-GO recommendation: strict HMDB effect-profile integration remains primary, with expanded high-confidence mapping as a sensitivity option if future deterministic rescue increases the feature set.
