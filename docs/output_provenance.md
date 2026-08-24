# Output Provenance

Every scientifically meaningful file copied into this repository (or deliberately excluded) is logged in `docs/file_provenance_map.csv`, one row per file, with columns: `original_path` (location in the original working project), `github_repository_path` (location here, blank if excluded), `category`, `status` (COPIED / EXCLUDED_* variants), `reason_for_inclusion`, `file_type`, `analysis_stage`, `classification`, `authoritative_status`, `included` (True/False), `notes`.

For a numbers-level provenance map (which CSV a specific thesis claim/figure/table number comes from), see:

- `final_analysis/tables/final_stage4_figure_source_values.csv` — Stage 4 frozen figure → source CSV map.
- `analyses/03_cross_cohort_replication/manifests/stage2_3_manuscript_claim_provenance.csv` / `.json` — Stage 2/3 claim → source CSV map, with every value asserted against its live source at generation time.

For a file-classification-only view (no copy/exclusion status), see `docs/analysis_file_classification.csv`.
