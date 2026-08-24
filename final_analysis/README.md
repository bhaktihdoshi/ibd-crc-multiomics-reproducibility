# Final Analysis — Authoritative Thesis Results

**If you are an examiner or reader who wants to know "what actually produced the numbers in this thesis," start here.**

This folder is the authoritative source for every Stage 4 (multi-omics integration) numerical claim in the thesis. Everything outside `final_analysis/`, `analyses/03_cross_cohort_replication/` (Stage 2/3 biology), and `preprocessing/` is either supporting preprocessing, a labelled sensitivity analysis, or historical/exploratory material that must **not** be cited as a final result.

## 1. The authority chain, in order

1. **`manifests/FINAL_STAGE4_AUTHORITY_MANIFEST.json`** — the original freeze, dated 20 August. Declares which files are authoritative for Stage 4A–4E and lists a 29-file authoritative registry (`tables/final_stage4_authoritative_files.csv`) with sha256 hashes.
2. **`manifests/final_stage4_claim_policy.csv`** — the original frozen claim policy (what is/isn't allowed to be claimed from each stage). Kept unmodified for historical transparency.
3. **`manifests/POST_FREEZE_CODE_PATCH_MANIFEST.json`** — records every code change made *after* the 20 August freeze (hardening, new sensitivity branch, QA tooling), with an explicit assessment of whether each change could affect any frozen number (answer: no, in every case, with the equivalence evidence cited).
4. **`manifests/POST_FREEZE_ANALYSIS_ROLE_ADDENDUM.json`** + **`manifests/post_freeze_claim_policy_addendum.csv`** — the final-thesis-reporting decision to treat microbiome and metabolome as **co-primary single-omic analyses** (see §2). This supersedes the original claim policy's "microbiome principal / metabolome stress test" framing *for reporting purposes only* — no numbers changed.
5. **`reports/final_stage4_qa_report.json`** — the final QA suite result (10/10 checks passing at time of freeze: authoritative-file hash integrity, no duplicate participants, correct cohort counts, no non-finite values, feature-order-at-transfer, fixed-seed reproducibility, figure/claim provenance, reporting-validation pass).
6. **`manifests/stage4_post_freeze_equivalence_check.json`** — direct numerical proof that the post-freeze-patched code still reconstructs the frozen Stage 4A genus matrices and reproduces the frozen Stage 4B MICROBIOME AUCs exactly (within 1e-6).
7. **`manifests/FINAL_REPRODUCIBILITY_BUNDLE.json`** — one file referencing and hash-checking all of the above together.

**None of items 3–7 change any frozen number.** They are provenance/verification layers added after the freeze. If you only read one file, read `FINAL_REPRODUCIBILITY_BUNDLE.json`.

## 2. Final analytical hierarchy

```
CO-PRIMARY SINGLE-OMIC ANALYSES        microbiome-only   metabolome-only
INTEGRATION COMPARISON                 early integration (tests whether combining the two co-primary
                                        layers improves on either alone — NOT itself a third co-primary analysis)
LATENT / STRUCTURAL MULTI-OMICS        MOFA, sCCA (descriptive; sCCA fold correlations are tuning-CV
                                        diagnostics, not unbiased held-out estimates; full-cohort correlations
                                        are apparent/descriptive, not causal)
PRIOR-KNOWLEDGE / GRAPH ANALYSIS       KEGG/NJC19 graph construction, GCN vs. identity ablation vs.
                                        ridge comparator (specific to the tested graph policies/architectures —
                                        not general evidence about GNNs)
```

Microbiome and metabolome are **co-primary**: both were run through the identical prespecified nested-CV / decision-score / bootstrap protocol on matched participants with reciprocal external validation. Metabolome is **not** a secondary analysis or "stress test" — but it does retain an additional limitation that microbiome does not: cross-cohort quantitative-scale equivalence of the metabolomics assays was never established (HMDB ID matching gives feature identity, not common units, response factors, or transformations). Microbiome retains its own, different caveat: for the CRC pair specifically, Sinha (16S) vs. Yachida (shotgun) is platform-confounded with cohort.

**Use `generate_post_freeze_co_primary_display_layer.py`'s output** (`tables/post_freeze_table2_stage4b_transportability_co_primary.csv` and `figures/post_freeze_figure_B_stage4b_transportability_co_primary.png`) for any thesis-facing Stage 4B table/figure — not the original frozen `table2_stage4b_transportability.csv` / `figure_B_...png`, which still carry the superseded "principal / stress test" wording (kept unmodified for provenance, not for citation).

## 3. What supports the thesis vs. what is post-freeze addition

| Category | Examples | Use for thesis numbers? |
|---|---|---|
| Frozen authoritative (20 Aug) | `tables/stage4b_corrected_v2b_*`, `tables/stage4e_corrected_v2_ensemble_*`, `tables/final_stage4/table1-5_*` | **Yes** |
| Post-freeze display (relabelled, same numbers) | `tables/post_freeze_table2_*`, `figures/post_freeze_figure_B_*` | **Yes** — prefer these for co-primary framing |
| Post-freeze sensitivity | `tables/sensitivity_crc_microbiome_*` | Cite explicitly as post-freeze sensitivity evidence, not primary |
| Post-freeze provenance/QA tooling | everything under `manifests/POST_FREEZE_*`, `reports/final_stage4_qa_report.json` | Use for methods/reproducibility text, not as a results source |

## 4. Historical files NOT to use for final numerical claims

`exploratory/historical_analyses/stage4_pre_correction_v1/` contains the **original, pre-correction** Stage 4 pipeline (its own `run_stage4a.py`...`run_stage4e_gcn.py`, tables, figures, models). This was the version before the corrected-v2 hardening pass (leakage fixes, decision-score ranking metrics, threshold-freezing correction, genus double-counting fix). It is kept for provenance/development history only. **Do not cite any number from that folder in the thesis.** The `stage4_multiomics_integration/scripts/run_stage4b_corrected_v2.py` file inside `final_analysis/code/legacy_within_corrected_v2/` is a similar case one level up — an early corrected-v2 iteration (still `predict_proba`-based) superseded by `run_stage4b_corrected_v2b.py` within the same corrected-v2 tree.

## 5. Where the final numbers actually come from

Every number in the thesis Stage 4 results traces back through the chain in §1. If you need the source CSV for a specific figure/table, consult `tables/final_stage4_figure_source_values.csv` (original frozen figure→source map) or the post-freeze equivalents referenced above. For Stage 2/3 biological claims (replicated metabolites, within-disease concordance, Yachida ordered-state), see `analyses/03_cross_cohort_replication/manifests/stage2_3_manuscript_claim_provenance.csv`/`.json` instead — that is the authoritative claim→source map for the biological-replication half of the thesis.
