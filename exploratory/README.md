# Exploratory and Historical Material — READ THIS BEFORE CITING ANYTHING HERE

**Nothing in this folder is part of the final thesis inferential analysis.** It is preserved to show the analytical development of the project — what was tried, what worked, what didn't, and what was superseded — not as competing valid results. If you are looking for the final thesis numbers, go to `final_analysis/` and `analyses/03_cross_cohort_replication/` instead.

## Why this material is included

An MSc thesis analysis is not just its final pipeline — it is the product of iteration, dead ends, and corrections. This repository keeps that history visible rather than deleting it, because:

- Some of it directly explains *why* the final design looks the way it does (e.g. the pre-correction Stage 4 pipeline shows exactly what was wrong before the correction).
- Some of it is genuinely informative negative/null results that remain scientifically relevant even though they didn't make the final cut for the primary pipeline.
- Hiding this material would make the project look more linear and less rigorously developed than it actually was.

## What's here

- `alternative_models/` — alternative predictive model families explored (ridge/elastic-net/random-forest/gradient-boosting variants where they exist).
- `alternative_preprocessing/` — preprocessing approaches that were tried and not adopted.
- `diagnostic_analyses/` — model diagnostics, probability-compression checks, and similar investigative work.
- `discarded_or_infeasible_ideas/` — analyses considered but judged infeasible (or documented as such) before execution.
- `historical_analyses/` — the largest bucket, containing:
  - **`stage4_pre_correction_v1/`** — the complete original Stage 4 pipeline before the corrected-v2 hardening pass. See its own `README.md` for exactly what was fixed and why.
  - IBD-only and CRC-only predictive/classifier development (hyperparameter search, calibration, nested-CV model selection, ROC/PR/confusion-matrix/decision-curve diagnostics) that predates and is superseded by the Stage 4 multi-omics predictive pipeline for final prediction claims.

## Warning for readers

- Numbers here may **differ** from the final analysis — sometimes substantially, since some of this material predates specific bug fixes (see `stage4_pre_correction_v1/README.md`).
- Do not average, pool, or otherwise combine results from here with `final_analysis/` results.
- If you want to cite historical development for methodological-narrative purposes (e.g. "an earlier pipeline version exhibited X, which motivated the correction described in Y"), that is an appropriate use — citing a number from here as a thesis *finding* is not.

## Classification

Every file here is logged in `docs/file_provenance_map.csv` with `classification = HISTORICAL_SUPERSEDED`, `EXPLORATORY`, or a related non-authoritative label, and an `authoritative_status` field explaining its relationship to the final analysis.
