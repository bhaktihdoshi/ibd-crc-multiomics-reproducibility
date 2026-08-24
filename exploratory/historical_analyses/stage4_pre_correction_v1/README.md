# Stage 4 — Pre-Correction (v1) Pipeline — HISTORICAL, NOT AUTHORITATIVE

**Do not cite any number from this folder in the thesis.** This is the original Stage 4 multi-omics pipeline before a hardening/correction pass produced the `corrected_v2` tree that is actually authoritative (see `final_analysis/README.md`).

## Why this version was superseded

An audit of this original pipeline found several issues that were fixed in `corrected_v2`, including:

- Genus-reconstruction double counting (explicit genus rows summed together with their own descendant species rows in some cohorts).
- Stage 4B ranking metrics (ROC-AUC/PR-AUC) computed from `predict_proba` outputs, which saturate/compress under class-weighted logistic regression, rather than from `decision_function` scores.
- Youden threshold selection using fitted full-source probabilities rather than source cross-fitted out-of-fold decision scores, risking resubstitution bias.
- A denser, less interpretable KEGG-only graph topology later supplemented with curated NJC19 relations and a conservative variant.

None of these issues were catastrophic to the analysis's validity in a way that invalidates the project, but they were judged significant enough to warrant a corrected rerun, which became `corrected_v2`. This folder is kept so the development history and the reasoning behind the correction are visible — not because its numbers are usable.

## What's here

- `scripts/` — the original `run_stage4a.py` … `run_stage4e_gcn.py`, graph-construction scripts, and audit scripts.
- `tables/`, `figures/`, `predictions/`, `manifests/`, `reports/`, `logs/` — their outputs.
- `models/` — the original (smaller) MOFA model objects.
- `tests/`, `configs/` — associated test/config files.

Six of the acquisition/taxonomic-profiling scripts here (`acquire_stage4a_microbiome.py`, `assemble_ihmp_metaphlan_profiles.py`, `prepare_ihmp_raw_metagenomics.py`, `recover_ihmp_microbiome.py`, `run_ihmp_metaphlan.sh`, `finalize_stage4a_freeze.py`) are a partial exception: their **output** (the canonical per-cohort microbiome relative-abundance matrices) is still consumed as an input by the corrected_v2 pipeline, even though corrected_v2 does not re-run these scripts themselves. They are duplicated under `preprocessing/microbiome/stage4a_acquisition/` for that reason — see that folder for the preprocessing narrative.

## Relationship to corrected_v2

`corrected_v2` is a from-scratch-hardened re-implementation, not a patch applied to these files. Compare `final_analysis/code/run_stage4b_corrected_v2b.py` against `scripts/run_stage4b.py` here if you want to see the specific ranking-metric and threshold-selection fix directly.
