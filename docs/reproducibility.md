# Reproducibility

## Random seeds

A single project-level random seed (`RANDOM_SEED = 20260813` in `final_analysis/code/stage4_corrected_v2_common.py`) is used as the base for all `StratifiedKFold` splits and model fitting across Stage 4B–E, with deterministic per-fold/per-seed offsets (e.g. `RANDOM_SEED + fold`, `RANDOM_SEED + 1000 + fold`) so that re-running the pipeline reproduces bit-identical fold assignments and decision scores. This is directly verified by `final_analysis/reports/final_stage4_qa_report.json`'s `fixed_seed_reproducibility` check, which refits the same inner-CV loop twice with the same seed and asserts bit-identical output.

## Cross-validation structure

Stage 4B: 5 outer folds × inner 5-fold grid search over `C ∈ {0.01, 0.1, 1.0, 10.0}` for L2-penalised logistic regression, with automatic fold-count reduction (`fold_count()`) when a class has fewer than 5 members. Stage 4E: 5 outer folds × 5 seeds, participant-level out-of-fold ensemble evaluation.

## Training-fold-only preprocessing (leakage prevention)

All trainable preprocessing — microbiome prevalence filtering, re-closure, CLR fitting, z-scaling; metabolomics missingness/zero-variance filtering, median imputation, z-scaling; hyperparameter selection — is fit exclusively on the training partition inside each CV fold and only ever *applied* (not re-fit) to the validation/external partition. This is enforced by sklearn `Pipeline`/`ColumnTransformer` composition (see `MicrobiomeCLRPreprocessor`, `metabolome_pipeline()` in `stage4_corrected_v2_common.py`) and independently checked by the QA suite's leakage-adjacent tests.

## Source/external separation

Each transfer direction trains on one cohort and evaluates on the reciprocal cohort (e.g. Franzosa→iHMP and iHMP→Franzosa). Feature order and identity are asserted identical between train and external sets before any transfer (`aligned_train_external()`); a mismatch raises a hard error rather than silently reindexing or filling.

## Frozen decision thresholds

The Youden-J threshold is selected from **source cross-fitted out-of-fold decision scores only** (never from resubstitution on the fitted model, and never from external labels), then frozen and applied unchanged to the external evaluation set. `external_data_used_for_threshold: False` is recorded explicitly in the threshold manifest for every task/representation.

## Bootstrap procedure

2,000 stratified-by-class participant bootstrap resamples. For paired model comparisons, the same resample index sequence is reused across both arms of the comparison so the paired difference is computed on matched resamples, not independently redrawn ones.

## BH-FDR use

Benjamini-Hochberg FDR correction is applied within each pre-specified comparison family separately (e.g. the Stage 4E 24-comparison primary family and 48-comparison sensitivity family are corrected independently; the CRC microbiome sensitivity runner corrects its own 4-comparison family separately). Comparison families are not pooled across unrelated hypotheses.

## Authority manifests and hash checks

`final_analysis/manifests/final_stage4_authoritative_files.csv` records a sha256 hash for every one of the 29 files frozen as authoritative on 20 August. `run_final_stage4_qa.py`'s `authoritative_files_unmutated` check recomputes and compares these hashes on every run — any post-freeze code change that accidentally altered a frozen output would be caught immediately.

## Post-freeze equivalence verification

`final_analysis/code/run_stage4_post_freeze_equivalence_check.py` independently reconstructs the four authoritative genus-proportion matrices and the shared 50-genus (IBD) / 52-genus (CRC) feature spaces from raw source data using the current (post-freeze-patched) code and diffs them numerically against the frozen outputs. Run with `--run-clr-replication` it additionally refits the Stage 4B MICROBIOME nested-CV protocol for all 4 tasks and confirms the resulting external ROC-AUC/PR-AUC match the frozen values within 1e-6.

## Test suite

`final_analysis/tests/test_corrected_v2_integrity.py` (50 tests) covers: no genus double-counting, fold-local preprocessing, pooled-task-only scope, decision-score threshold/metrics correctness, sCCA label-freeness and non-nested-CV honesty, graph mode/edge-type validation, ensemble averaging, BH-FDR family sizes, seed-replay reproducibility, and authoritative-source freeze-script checks. Run with `python3 -m pytest final_analysis/tests/test_corrected_v2_integrity.py -q`.

## Final QA

`python3 final_analysis/code/run_final_stage4_qa.py` runs the full test suite plus 10 additional direct checks (see `final_analysis/README.md` §1 item 5) and writes one consolidated PASS/FAIL verdict to `final_analysis/reports/final_stage4_qa_report.json`.

## Environment/version records

See `docs/software_environment.md` and `docs/software_environment_stage4.md` for exact package versions used.

## Exact commands

```bash
# from final_analysis/code/, with THESIS_PROJECT_ROOT set if not running from the original layout
export THESIS_PROJECT_ROOT=/path/to/repository/parent

# Stage 4 pipeline (each stage writes to final_analysis/{tables,predictions,figures,manifests}/)
python3 run_stage4a_corrected_v2.py
python3 run_stage4b_corrected_v2b.py
python3 run_stage4c_corrected_v2.py --analysis all
python3 run_stage4d_corrected_v2_graph.py
python3 run_stage4e_gcn_corrected_v2.py
python3 freeze_final_stage4_results.py

# Post-freeze sensitivity, QA, equivalence (safe to re-run; do not overwrite the 20-Aug freeze)
python3 run_stage4b_crc_microbiome_sensitivity.py
python3 run_stage4_post_freeze_equivalence_check.py --run-clr-replication
python3 run_final_stage4_qa.py

# Stage 2/3 biological replication and claim provenance
cd ../../analyses/03_cross_cohort_replication/scripts
python3 run_stage3.py
python3 generate_stage2_3_manuscript_claim_provenance.py
```

Note: several Stage 4 steps require raw/processed data not redistributed in this repository — see `data/README.md` to obtain and place source data before running.
