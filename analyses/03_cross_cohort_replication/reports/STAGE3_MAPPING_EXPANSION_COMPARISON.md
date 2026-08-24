# Stage 3 Mapping Expansion Comparison

## Strict Set

- n = 56
- Matching rule: canonical HMDB present in Franzosa, iHMP, Sinha and Yachida.
- Strengths: highest specificity; no cross-namespace assumptions; directly traceable to the existing strict Stage 3 file.
- Weaknesses: may miss metabolites whose identifiers differ across source mappings.

## Expanded High-Confidence Set

- n = 56
- HMDB alias/canonicalisation rescues: 0
- Exact KEGG rescues: 0
- Exact structural identifier rescues: 0
- Strengths: allows deterministic exact-identifier rescue while preserving provenance.
- Additional assumptions: exact KEGG rescue is allowed only when one-to-one and non-conflicting in local mappings. No name-only candidate is admitted.

## Ambiguous Set

- ambiguous candidate count = 0
- excluded one-to-many/many-to-one mapping rows = 13
- These were excluded because name-only, one-to-many, many-to-one, class-level or conflicting mappings are not reliable enough for primary Stage 3 integration.

## Recommendation

Use **strict 56 HMDB metabolites as the primary Stage 3 set**. Use the expanded high-confidence set as a sensitivity set only. In this audit, the expanded set does not materially increase the feature count, so it should not replace the strict set.
