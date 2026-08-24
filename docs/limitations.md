# Known Limitations

These are the limitations the final analysis explicitly carries forward rather than resolves. They are stated here so a reader does not have to reconstruct them from scattered audit files.

## Metabolomics cross-cohort quantitative-scale equivalence was not established

HMDB ID matching establishes **feature identity** (the same metabolite is being referred to across cohorts) but does not establish common units, response factors, or upstream instrument/processing transformations. Training-fold z-standardisation prevents leakage but does not create cross-assay comparability. This is why metabolome-only, while co-primary in the reporting framework, retains this specific caveat that microbiome does not. A feasibility investigation into a common-scale metabolomics matrix (`final_analysis/code/audit_stage4b_v2c_common_scale_feasibility.py` and `audit_stage4b_v2c_metabolomics_comparability.py`) found no defensible common-scale representation was locally available. Joint source+target normalisation (ComBat, ranks fit across both cohorts, pooled centring) was deliberately not applied to the primary pipeline, because it would leak target-distribution information into the external-validation question.

## CRC microbiome platform is confounded with cohort

Sinha (16S rRNA amplicon) and Yachida (shotgun metagenomics) differ in assay platform, and platform is perfectly confounded with cohort for the CRC pair — there is no way to isolate a platform effect from a cohort effect with only two CRC cohorts. Post-freeze rank and presence/absence sensitivity representations (`final_analysis/tables/sensitivity_crc_microbiome_*`) test whether the primary CLR-based result depends on abundance-scale representation; they cannot resolve the platform-vs-cohort confound itself.

## sCCA fold correlations are tuning diagnostics, not unbiased estimates

The current sCCA implementation performs a single-layer cross-validation to select PMA sparsity penalties; the resulting fold correlations are internal tuning-CV diagnostics, not out-of-sample generalisation estimates from a nested design. Full-cohort sCCA correlations are apparent/descriptive (computed on the same data used to select the sparsity penalty), not held-out. This is a genuine, documented limitation — not fixed in the current thesis — and is tracked as future work (a nested outer-CV sCCA evaluation) rather than resolved. Do not describe current sCCA correlations as "validated," "held-out," or "external."

## MOFA and sCCA are descriptive, not causal

Both are unsupervised/semi-supervised structural integration methods run on non-covariate-residualised, transformed omics data. Cross-omic covariance may partly reflect shared disease association rather than a specific mechanistic coupling. No causal claim is supported by either method as currently run.

## GCN results are architecture/policy-specific

The Stage 4E GCN vs. identity-ablation vs. ridge comparison is specific to the tested graph topologies (identity, KEGG-only, broad KEGG+NJC19, conservative KEGG+NJC19), the one-layer mean-pooling architecture, and the training/evaluation protocol used. It is not general evidence that graph neural networks fail or succeed on microbiome-metabolome integration tasks broadly — a different architecture, a different graph construction, or a different task could behave differently.

## No fully harmonised four-cohort clinical covariate set exists

Age, sex, BMI, medication, disease activity, calprotectin, smoking, and tumour-stage variables are not uniformly available/harmonised across all four cohorts. No common four-cohort clinical-only model was built. Where a clinically sensible covariate set exists within one disease pair, it is documented explicitly as a pair-specific sensitivity, not generalised.

## Current predictive scope is pooled IBD-vs-Control and CRC-vs-Control only

CD-vs-Control and UC-vs-Control subtype classifiers exist as descriptive/historical analyses (see `exploratory/historical_analyses/`) but are not part of the current Stage 4 predictive scope, graph evaluation, or multiplicity-correction families.

## Yachida ordered-state findings are linear associations, not proof of monotonic trajectories

Of 56 tested metabolites, 2 showed significant HC3-robust **ordered-state linear associations** after BH correction (citrulline, N6,N6,N6-trimethyllysine) — neither has a monotonic abundance profile across disease stages (`profile_pattern` = "irregular" and "early_peak" respectively). A separate rank-based Spearman sensitivity found 6 significant associations. These are two distinct tests; do not conflate them into a single "monotonic progression" claim. See `analyses/03_cross_cohort_replication/manifests/stage2_3_manuscript_claim_provenance.csv`.

## Historical work does not represent final results

The pre-correction Stage 4 pipeline (`exploratory/historical_analyses/stage4_pre_correction_v1/`) and IBD/CRC-only predictive-modelling history contain real limitations that were specifically fixed in the corrected/final pipeline (see `exploratory/historical_analyses/stage4_pre_correction_v1/README.md`). They are preserved for transparency, not as competing valid results.
