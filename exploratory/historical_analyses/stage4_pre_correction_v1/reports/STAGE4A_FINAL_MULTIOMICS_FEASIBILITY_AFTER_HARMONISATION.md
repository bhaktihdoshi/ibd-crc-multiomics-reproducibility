# Stage 4A Final Multi-Omics Feasibility After Harmonisation

## Franzosa
Final microbial source: official Nature Supplementary Dataset 4 species relative abundance. Final matched n=137. Raw species p=202; canonical included species p=155 and canonical genus p=23 from deterministic label parsing. Genus aggregation can be derived later from species labels if iHMP becomes available.

## iHMP
Official metadata was acquired, but no valid abundance matrix was acquired. Final matched n=0. Species p=0. Genus p=0.

## Sinha
Final representation: explicit taxonomic tree nodes from existing S3 microbial table. Final n=131. Parsed tree features include 220 total labels; canonical included species p=0, genus p=57 and family p=43 after excluding ambiguous/unclassified labels.

## Yachida
Official supplementary species/genus profiles acquired. Duplicate conflicting subject IDs are excluded from the clean handoff. Final unambiguous matched n=252. Canonical species p=8367. Canonical genus p=1941.

## IBD Cross-Cohort
Species overlap=0; genus overlap unavailable because iHMP abundance was not acquired. Chosen rank: none. Shared p=0.

## CRC Cross-Cohort
Species overlap=0 using deterministic canonical labels; genus overlap=52. Chosen rank: genus if Stage 4B proceeds. Shared p=52.

## Final
Feasibility classification: D. CRC_ONLY_CROSS_COHORT_MULTIOMICS_READY.
IBD: STAGE4B_IBD_NO_GO.
CRC: STAGE4B_CRC_GO if the supervisor accepts genus-level 16S-shotgun harmonisation and Yachida ambiguous-subject exclusion.
