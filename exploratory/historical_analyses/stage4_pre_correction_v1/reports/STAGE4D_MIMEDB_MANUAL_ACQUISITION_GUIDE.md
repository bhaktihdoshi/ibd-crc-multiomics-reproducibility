# Stage 4D MiMeDB Manual Relation Export Guide

Purpose: acquire explicit MiMeDB MicrobeCard `Related Metabolites` exports for Stage 4D cross-omic graph augmentation. This is manual data acquisition only. Do not fit GCN/GAT or construct phenotype-derived edges.

## Input Manifest

Use:
`/Users/bhaktidoshi/Downloads/Thesis Data/thesis_preprocessing_workflow/stage4_multiomics_integration/tables/stage4d_mimedb_manual_export_manifest.csv`

Rows with `selection_status = READY_FOR_MANUAL_EXPORT` can be exported directly.
Rows with `selection_status = MULTIPLE_RECORDS_REVIEW_REQUIRED` indicate multiple MiMeDB species/strain records for the same Stage 4 genus. Do not arbitrarily select one record. Either export all listed records or document a biological/manual restriction.

## Destination Folder

Save all downloaded relation files here:
`/Users/bhaktidoshi/Downloads/Thesis Data/thesis_preprocessing_workflow/stage4_multiomics_integration/data/stage4d_prior_sources/mimedb/relations`

## Browser Steps For Each Manifest Row

1. Open the `microbecard_url` in a browser.
2. On the MicrobeCard page, locate the `Related Metabolites` section/table.
3. Click `Download metabolites`.
4. Save the file using the exact path in `local_relation_file`.
5. Keep the file as CSV. Do not rename columns manually.
6. After exports are complete, rerun only the Stage 4D cross-omic dry-run/parser audit before any graph augmentation.

## Deterministic File Naming

`<canonical_genus>__<microbe_id>__related_metabolites.csv`

Example:
`Acidaminococcus__MMDBm0000003__related_metabolites.csv`

## Current Workload Summary

- Unique Stage 4 genera: 64
- Genera with MiMeDB matches: 60
- Total matched MiMeDB records: 846
- Genera with exactly one MiMeDB record: 2
- Genera with multiple species/strain records: 58
- Required manual exports if exporting all matched records: 846
- Feasibility call: `MANUAL_EXPORT_TOO_BURDENSOME`

## Notes

- Prefer records with explicit human/gut relevance only where this is visible in MiMeDB metadata; do not invent relevance.
- The manifest flags records with explicit human/gut wording in `selection_reason`, but does not automatically remove other species/strain records.
- No disease labels, model coefficients, correlations, MOFA, sCCA or validation outcomes should be used to decide graph edges.
