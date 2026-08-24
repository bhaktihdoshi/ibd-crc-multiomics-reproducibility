# Stage 3 Mapping Expansion Provenance

## Sources used

- `thesis_preprocessing_workflow/ibd_pipeline/tables/hmdb_to_kegg_mapping_audit.csv` sha256 `bbabb322d73f1b2bcef2f6e73960d8b01ee270ef68e77b3521a5225d97ebdf76`
- `thesis_preprocessing_workflow/ibd_pipeline/tables/differential_abundance_standardized_results.csv` sha256 `8b66844bc6ea3f9411dd83a8472537a3a987bbb4c61c7d146063f58aeff4e41d`
- `thesis_preprocessing_workflow/crc_pipeline/stage2/tables/crc_stage2_canonical_metabolite_manifest.csv` sha256 `f5792a7818d946a17e6ff756b02ce19e3a83eeaff30a2ea0bdafa9dedde718e4`
- `thesis_preprocessing_workflow/crc_pipeline/stage2/tables/crc_stage2_hmdb_kegg_crosswalk.csv` sha256 `08e8275db20201df7dec07ca68fde994a41bc78cb1518e5d39a343c7ca36b366`
- `thesis_preprocessing_workflow/stage3_ibd_crc_integration/tables/stage3_four_cohort_shared_metabolites.csv` sha256 `10f5a8b200b21ad7e2efcf357a3ca082e795ce26d8f46c67a22a8e4e2b51f3b9`

## Deterministic rules

- HMDB IDs were normalised for prefix, whitespace and case only.
- Exact KEGG rescue required the same KEGG compound ID in all four cohorts and no conflicting HMDB assignment.
- Structural rescue was not performed because no InChIKey, ChEBI, PubChem CID or SMILES identifiers were present in the local authoritative inputs.
- Name matching was used for candidate generation only and was not admitted into the primary expanded set.
- One-to-many, many-to-one, class-level and conflicting annotations were excluded.
- No internet/API annotation enrichment was performed.
- No frozen IBD or CRC file was modified.
