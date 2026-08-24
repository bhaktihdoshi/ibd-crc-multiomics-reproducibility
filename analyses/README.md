# Analyses — Index

This folder groups the project's analyses by scientific stage. For Stage 4 (multi-omics integration: early integration, MOFA, sCCA, graph construction, GCN, sensitivity), the actual authoritative code/results are consolidated in one place — `final_analysis/` — rather than duplicated per-stage here, to avoid maintaining the same files in two locations. Each `05_*`–`10_*` subfolder below is a short pointer explaining what that stage covers and exactly where its code/results live.

| Folder | Stage | Where the real content is |
|---|---|---|
| `01_initial_single_cohort_work/` | Early, cohort-specific exploratory work (IBD/CRC) | See `preprocessing/`, `exploratory/historical_analyses/` |
| `02_differential_abundance/` | Stage 2 differential abundance | See `preprocessing/ibd/`, `preprocessing/crc/`, and `analyses/03_cross_cohort_replication/` |
| `03_cross_cohort_replication/` | Stage 3 biological replication | **Full content here** — this is a complete, mostly-authoritative pipeline (see its own README) |
| `04_predictive_baselines/` | Early/cohort-local predictive baselines superseded by Stage 4 | See `exploratory/historical_analyses/` |
| `05_early_integration/` – `09_gcn/` | Stage 4B (early integration) – Stage 4E (GCN) | **Pointer only** — see `final_analysis/code/` and `final_analysis/README.md` |
| `10_sensitivity_analyses/` | Post-freeze sensitivity (CRC microbiome CLR/rank/presence-absence) | **Pointer only** — see `final_analysis/code/run_stage4b_crc_microbiome_sensitivity.py` and `final_analysis/tables/sensitivity_crc_microbiome_*` |

`analyses/03_cross_cohort_replication/` is the one exception that holds its full content directly (Stage 3 is a self-contained pipeline, not consolidated elsewhere).
