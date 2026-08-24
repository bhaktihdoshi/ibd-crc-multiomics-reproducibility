# Corrected-v2 Implementation Audit

Thesis: `Cross-cohort reproducibility of gut microbiome–metabolome integration in inflammatory bowel disease and colorectal cancer`

Scope: Stage 4 corrected-v2 pre-execution correction and reorganisation. Historical Stage 4 outputs were preserved. Corrected outputs are written only under:

`thesis_preprocessing_workflow/stage4_multiomics_integration/corrected_v2/`

No complete or computationally expensive modelling analysis was run during this pass.

## Severity-Ranked Audit

| Severity | Suspected issue | Status | Evidence | Corrected-v2 action |
|---|---|---|---|---|
| Critical | Explicit genus and descendant species could be summed into one genus feature | Confirmed | Historical Stage 4B `to_genus_matrix()` accepted both genus and species ranks and grouped by genus name. Source matrices contain mixed ranks for iHMP and Yachida. | Added authoritative cohort-specific genus construction: Franzosa species-to-genus aggregation; iHMP/Sinha/Yachida explicit-genus only. No parent-child double counting. |
| Critical | Early integration used a generic preprocessing pipeline on the concatenated raw matrix | Confirmed | Historical Stage 4B used one missingness/imputation/scaling pipeline for `EARLY_INTEGRATION`. | Added block-aware `ColumnTransformer`: metabolome block gets missingness/constant/median/scaling; microbiome block gets prevalence/re-closure/CLR/scaling. |
| Critical | Final Youden threshold selected from full-training fitted probabilities | Confirmed | Historical `final_fit_external()` selected threshold from `final_pipe.predict_proba(X_train)` after fitting on all source data. | Added final threshold selection from full-source cross-fitted OOF probabilities using the selected ridge `C`; final pipeline is then refit and frozen before external prediction. |
| High | Microbiome CLR was applied without explicit re-closure after fold-local prevalence filtering | Confirmed | Historical `MicrobiomePreprocessor._clr()` used filtered values directly after adding pseudocount. | Added re-closure inside the fold-local microbiome transformer before CLR. |
| High | External ROC/PR uncertainty missing from corrected Stage 4B design | Confirmed for historical Stage 4B | Historical Stage 4B exported point estimates only. | Added 2,000 stratified participant-level bootstrap CI fields for external ROC-AUC and PR-AUC in corrected-v2. |
| High | Identity GCN may not retain enough node identity | Confirmed as methodological limitation, not patched in frozen output | Current GCN node tensor contains abundance and node-type, with identity adjacency. Named-node identity is encoded mostly through fixed node order and learned weights after pooling. | Documented as a corrected-v2 GCN requirement. A future corrected-v2 GCN script should add low-complexity node identity embeddings and multi-seed comparisons before final thesis GCN claims. |
| Medium | sCCA held-out scaling may require re-verification | Not yet confirmed in code during this scoped pass | Requires deeper inspection of PMA input generation and R tuning path. | Listed as a blocking check before corrected sCCA rerun. |
| Medium | MOFA requires corrected microbial matrices before refit | Confirmed dependency | MOFA consumes Stage 4C inputs derived from previous Stage 4A feature space. | Listed as downstream rerun dependency after corrected Stage 4A/B. |
| Medium | Corrected-v2 modelling scope still included CD-vs-control and UC-vs-control | Confirmed | Corrected-v2 `TASK_SPECS` initially retained subtype tasks from the earlier IBD-only model plan. | Removed subtype prediction tasks. CD and UC are retained only as descriptive cohort counts; pooled IBD versus control is the only IBD predictive task. |

## Patched Scripts

| Script | Purpose |
|---|---|
| `corrected_v2/scripts/stage4_corrected_v2_common.py` | Shared corrected-v2 utilities, authoritative genus construction, fold-local preprocessing, ridge pipelines, cross-fitted thresholds and bootstrap CIs. |
| `corrected_v2/scripts/run_stage4a_corrected_v2.py` | Lightweight authoritative genus harmonisation and before/after microbiome audits. |
| `corrected_v2/scripts/run_stage4b_corrected_v2.py` | Corrected leakage-safe Stage 4B ridge models with block-aware early integration. |
| `corrected_v2/scripts/make_corrected_v2_thesis_tables.py` | Strict post-run thesis table generator. Fails if corrected-v2 result tables are missing. |
| `corrected_v2/scripts/make_corrected_v2_thesis_figures.py` | Strict post-run thesis figure generator. Fails if corrected-v2 result tables are missing. |
| `corrected_v2/tests/test_corrected_v2_integrity.py` | Lightweight unit tests for genus double-counting, re-closure/CLR, block-aware early integration, OOF threshold logic, primary-task scope and pooled IBD label coding. |

## Lightweight Checks Run

| Check | Status |
|---|---|
| Corrected Stage 4A authoritative genus audit/harmonisation | Passed |
| Corrected Stage 4B dry-run schema construction | Passed |
| Corrected-v2 script syntax compilation | Passed |
| Corrected-v2 unit tests | Passed: 5 tests |

Only lightweight checks were run. No corrected full Stage 4B modelling, MOFA, sCCA, graph reconstruction, multi-seed GCN or bootstrap-only final inference job was run beyond code paths already included in the Stage 4B script.

## Current Corrected Stage 4A Outputs

| Output | Meaning |
|---|---|
| `tables/stage4a_corrected_v2_microbiome_before_audit.csv` | Source rank composition, row sums, sparsity and duplicated IDs. |
| `tables/stage4a_corrected_v2_authoritative_genus_after_audit.csv` | Corrected genus tables after one-rank selection and re-closure. |
| `tables/stage4a_corrected_v2_taxon_crosswalk.csv` | Original taxon label, rank, canonical genus, inclusion status and exclusion reason. |
| `tables/stage4a_corrected_v2_ibd_shared_genera.csv` | Corrected shared IBD genus schema. |
| `tables/stage4a_corrected_v2_crc_shared_genera.csv` | Corrected shared CRC genus schema. |
| `tables/stage4a_corrected_v2_shared_genus_feature_space.csv` | Shared feature counts for IBD and CRC transfer pairs. |
| `tables/stage4a_corrected_v2_validation_tests.csv` | Genus harmonisation integrity checks. |

## Dependency Map: Stale Until Corrected Rerun

The following historical outputs should be considered obsolete for final thesis numerical reporting after these corrections:

| Area | Historical outputs affected | Why stale |
|---|---|---|
| Stage 4A microbiome feature space | `stage4a_*shared_microbiome_features.csv`, `stage4a_final_multiomics_feature_space.csv` | Genus/species harmonisation policy changed. |
| Stage 4B prediction | `stage4b_external_performance.csv`, `stage4b_internal_performance.csv`, `stage4b_threshold_diagnostics.csv`, `stage4b_external_predictions.csv` | Microbiome preprocessing, early integration and threshold selection changed. |
| Stage 4C MOFA | `stage4c_mofa_*` | MOFA inputs must be rebuilt from corrected microbial matrices. |
| Stage 4C sCCA | `stage4c_scca_*` | sCCA inputs and fold-local scaling need corrected matrices and verification. |
| Stage 4D graph | `stage4d_graph_*` | Graph node set may change if corrected shared genera differ; graph modes must be identity, KEGG-only and KEGG+NJC19. |
| Stage 4E GCN | `stage4e_gcn_*` | GCN node features and fair ablations depend on corrected Stage 4A/B/D; graph topology must remain unsigned/undirected and no microbe-microbe edges should be used in the primary graph. |
| Final figures/tables | Any figure/table using historical Stage 4B-E numbers | Must be regenerated from corrected-v2 outputs after manual execution. |

Do not update dissertation numerical claims with corrected-v2 values until the full corrected pipeline has been executed and validated.

## Proposed Final Display Items

Maximum planned display items: 9, leaving one spare.

| Display | Scientific purpose |
|---:|---|
| 1 | Show the whole study workflow: four stool cohorts, harmonisation, reciprocal validation and analysis progression. |
| 2 | Summarise cohort assays, matched participants and corrected shared features. |
| 3 | Show corrected external predictive transportability for metabolome, microbiome and early integration in pooled IBD-vs-control and CRC-vs-control only. |
| 4 | Quantify whether early integration improves over each single omic using paired ΔAUC for pooled IBD and CRC only. |
| 5 | Show MOFA latent structure: variance explained, pooled disease-control factor-score distributions and interpretable loadings. |
| 6 | Show sCCA sparse cross-view covariance and cross-cohort loading reproducibility. |
| 7 | Show prior graph construction, KEGG/NJC19 edge types and connectivity improvement. |
| 8 | Show corrected GCN topology evaluation with fair graph-mode comparisons and probability diagnostics. |
| 9 | Synthesize cross-method biological evidence using a feature-by-method evidence matrix. |

Demote to supplement: repeated AUC bars, internal-only AUC plots, graph hashes, raw hyperparameter traces, full-network hairballs, degenerate confusion matrices and duplicate MOFA/sCCA panels.

## Results Outline

1. Cohort overview, matching and QC: eligible participants, class counts, CD/UC descriptive counts, selected iHMP visit rule, exclusions, shared genera/metabolites, sparsity, missingness and assay comparability.
2. Single-omic and early integration prediction: pooled IBD-vs-control and CRC-vs-control external results first, internal CV secondary, each reciprocal direction reported separately.
3. Latent integration: MOFA for shared/view-specific latent variation; sCCA for sparse direct cross-view covariance. Phenotype annotation should use pooled IBD-vs-control and CRC-vs-control only, not CD/UC subtype-specific tests.
4. Prior graph: node counts, KEGG coverage, NJC19 coverage, isolated nodes before/after and graph limitations. NJC19 edges are curated production, consumption or transport relations, not evidence of disease association.
5. GCN topology evaluation: identity, KEGG-only and KEGG+NJC19; multi-seed stability; probability diagnostics; comparison with corrected ridge.
6. Cross-method biological synthesis: features supported by more than one independent analytical approach.

Each paragraph should state: analytical question, exact input, exact result with uncertainty, comparator, conservative interpretation and transition.

## Discussion Evidence Map

1. Corrected principal findings with exact values.
2. Whether early integration improved over the best single omic.
3. What MOFA and sCCA added beyond prediction.
4. Whether topology improved GCN relative to fair ablations.
5. Comparison with Franzosa, Lloyd-Price/iHMP, Sinha, Yachida and independent literature.
6. Cohort/platform explanations for heterogeneity.
7. Transportability versus intrinsic disease biology.
8. Strengths.
9. Limitations: 16S versus shotgun confounding, metabolomics platform differences, target-schema-aware harmonisation, cross-sectional design, one iHMP visit, no covariate residualisation for MOFA/sCCA, clinical variables not sufficiently complete/harmonised for a fair common cross-cohort predictive baseline, stool as snapshot, reverse causation, NJC19 production/consumption/transport relations are curated priors rather than cohort-specific activity, species-to-genus loss, small n, neural-network instability, no mechanistic experiments.
10. Proportionate future work.

## Reproducibility Supplement Template

Include:

- Script-to-output manifest.
- Package versions, OS, seeds and Python/R versions.
- Input and output checksums.
- Participant manifests and selected iHMP visit manifest.
- Corrected feature lists.
- Exact commands and runtime logs.
- Validation tests.
- Archive/repository link placeholder.

Keep graph hashes and code internals here, not in the main thesis Results.

## Reduced Test Commands

Run these before full analysis:

```bash
cd "/Users/bhaktidoshi/Downloads/Thesis Data"
/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/corrected_v2/scripts/run_stage4a_corrected_v2.py
/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/corrected_v2/scripts/run_stage4b_corrected_v2.py --dry-run
/opt/anaconda3/bin/python -m pytest thesis_preprocessing_workflow/stage4_multiomics_integration/corrected_v2/tests -q
```

Expected outputs:

- Stage 4A corrected-v2 audit tables.
- `stage4b_corrected_v2_dry_run.csv`.
- All corrected-v2 unit tests pass.

## Full Manual Execution Order

1. Corrected taxonomic harmonisation:

```bash
cd "/Users/bhaktidoshi/Downloads/Thesis Data"
/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/corrected_v2/scripts/run_stage4a_corrected_v2.py
```

2. Corrected Stage 4B ridge models:

```bash
cd "/Users/bhaktidoshi/Downloads/Thesis Data"
/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/corrected_v2/scripts/run_stage4b_corrected_v2.py
```

3. Corrected MOFA inputs and fit:

```bash
# To be enabled after Stage 4C input builder is pointed at corrected_v2 authoritative genus matrices.
# Phenotype annotations should be pooled IBD-vs-control and CRC-vs-control only.
# Do not include clinical variables as MOFA views.
```

4. Corrected sCCA:

```bash
# To be enabled after fold-local centring/scaling is verified in the PMA preparation/tuning path.
# Do not include clinical variables as sCCA views.
```

5. Graph reconstruction:

```bash
# To be enabled after corrected shared genus node set is frozen.
# Required modes: identity, KEGG-only, KEGG+NJC19.
# Primary graph: unsigned, undirected, no microbe-microbe edges.
# Preserve NJC19 relation_type as production/consumption/transport provenance.
```

6. Multi-seed GCN:

```bash
# To be enabled after corrected graph modes include identity, KEGG-only and KEGG+NJC19 with node-aware identity ablation.
```

7. Final thesis tables:

```bash
cd "/Users/bhaktidoshi/Downloads/Thesis Data"
/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/corrected_v2/scripts/make_corrected_v2_thesis_tables.py
```

8. Final thesis figures:

```bash
cd "/Users/bhaktidoshi/Downloads/Thesis Data"
/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/corrected_v2/scripts/make_corrected_v2_thesis_figures.py
```

9. Thesis-number consistency audit:

```bash
# Compare all thesis numbers against corrected_v2 tables before submission.
```

## Validation Criteria

Stage passes only if:

- One authoritative genus representation exists per cohort.
- Pre-CLR row sums are approximately one.
- No parent genus and descendant species are double-counted.
- Train/external feature order is identical.
- Corrected-v2 predictive tasks are only pooled IBD-vs-control and CRC-vs-control.
- In pooled IBD, CD and UC participants are coded y=1 and controls are coded y=0.
- CD and UC counts are retained only for cohort description.
- Microbiome-only and early-integration microbiome preprocessing share the same transformer logic.
- Missingness, prevalence, imputation, CLR and scaling are fitted inside training folds.
- Youden thresholds come from cross-fitted source probabilities.
- External data are used only for final prediction and metric calculation.
- External ROC/PR CIs are present.
- Degenerate probabilities and one-class classifications are flagged.
- Table/figure scripts use corrected-v2 outputs only.
- Clinical variables are not used as MOFA/sCCA views or graph nodes, and are not included as a common predictive baseline unless a later audit proves they are sufficiently complete and harmonised across cohorts.

## Current Status

`CORRECTED_V2_PRE_EXECUTION_READY_FOR_STAGE4A_STAGE4B`

Stage 4A and Stage 4B corrected-v2 code paths are implemented and lightweight-validated. Stage 4C-E correction scripts still need targeted follow-up before those downstream stages should be rerun.
