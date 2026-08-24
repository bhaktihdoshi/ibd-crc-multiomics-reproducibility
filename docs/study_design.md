# Study Design

## Overview

Two disease areas (IBD, CRC), each with two independently-processed cohorts, are used to test cross-cohort transportability of microbiome and metabolome case-control signatures. See `README.md` §4–5 for the cohort table and high-level design, and `docs/cohort_overview.md` for per-cohort detail.

## Predictive design (Stage 4B)

For each disease, both transfer directions are evaluated reciprocally (e.g. Franzosa→iHMP and iHMP→Franzosa), for three representations (microbiome-only, metabolome-only, early integration), giving 4 tasks × 3 representations = 12 train/external combinations. Each combination uses nested cross-validation on the training cohort (outer 5-fold for performance estimation, inner 5-fold for hyperparameter selection), followed by a single final refit on the full training cohort evaluated once on the external cohort, with a threshold frozen from source cross-fitted out-of-fold decision scores before external evaluation.

## Structural integration design (Stage 4C)

MOFA and sparse CCA are fit within-cohort (not pooled across cohorts) on the harmonised molecular feature spaces, descriptively — no phenotype label is used in fitting or tuning. Post-hoc phenotype annotation (pooled IBD-vs-Control or CRC-vs-Control) is applied only after fitting, for interpretation.

## Graph/GCN design (Stage 4D/E)

Four graph topologies (identity, KEGG-only, broad KEGG+NJC19, conservative KEGG+NJC19) are each evaluated with a one-layer mean-pooling GCN using 5 outer folds × 5 seeds, participant-level out-of-fold ensemble evaluation, compared against an identity-graph ablation and a matched ridge-regression comparator using paired participant bootstrap and BH-FDR-corrected comparisons within two prespecified families (24 primary comparisons, 48 sensitivity comparisons).

## Current predictive scope

Only pooled `IBD_vs_Control` and `CRC_vs_Control` tasks are part of the current Stage 4 predictive scope, graph evaluation, and multiplicity-correction families. CD-vs-Control and UC-vs-Control classifiers exist as descriptive/historical analyses only (see `exploratory/historical_analyses/`).
