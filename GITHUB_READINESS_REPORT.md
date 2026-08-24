# GitHub Readiness Report

Curated repository: `github_ibd_crc_multiomics/` (152MB, 1,338 files). Source project (`thesis_preprocessing_workflow/`, 788MB, ~2,946 files) was read-only throughout — verified untouched at the end of curation by each contributing agent and by direct spot-check.

## A. Repository structure created

Full target structure built: `docs/`, `data/`, `preprocessing/{ibd,crc,harmonisation,microbiome,metabolomics,qc}/`, `analyses/01–10/`, `final_analysis/{code,configs,manifests,provenance,tables,predictions,figures,reports,tests}/`, `exploratory/{alternative_models,alternative_preprocessing,diagnostic_analyses,discarded_or_infeasible_ideas,historical_analyses}/`, `notebooks/{preprocessing,exploration,modelling}/`, `tests/`, `results/{final,sensitivity,exploratory}/`, `thesis/{figure_provenance,table_provenance}/`. Stage 4 (MOFA/sCCA/graph/GCN/sensitivity) code is consolidated once under `final_analysis/` rather than duplicated per-stage under `analyses/06–10`, which instead hold pointer READMEs — this avoids maintaining the same files in two places while preserving the conceptual stage grouping (see `analyses/README.md`).

## B. Number of files reviewed

1,945 files logged in `docs/file_provenance_map.csv` (all four pipeline slices: IBD 625, CRC 418, Stage 3 111 + duplicated provenance files, Stage 4 789 + 6 dual-placed preprocessing scripts).

## C. Number copied

1,417 files copied (per the provenance map's `included=True` count), ~152MB total.

## D. Number excluded

528 files excluded per the provenance map, breaking down as: `postrun_backups/` duplicate tree (confirmed byte-size-identical to already-copied tables, IBD pipeline — not individually logged, whole tree skipped), 111 IBD `.joblib` classifier binaries, ~171 CRC pipeline files (raw third-party supplementary data, duplicate/stale snapshots, smoke-test outputs, 6 `.joblib` binaries), 4 `.DS_Store` files (Stage 3), and Stage 4 raw third-party data + `Archive.zip` + oversized derived tables.

## E. Number classified FINAL_AUTHORITATIVE

479 files.

## F. Number classified SENSITIVITY

55 files.

## G. Number classified EXPLORATORY / HISTORICAL_SUPERSEDED

625 files (`HISTORICAL_SUPERSEDED`) — the largest single category, dominated by the pre-correction Stage 4 pipeline (296 files) and IBD-only predictive-modelling history (294 files). No separate `EXPLORATORY`-labelled rows were found necessary beyond this by the contributing agents; genuinely exploratory (as opposed to definitively superseded) material is smaller in volume and folded into the same `exploratory/` tree.

## H. Large files requiring attention

12 files >10MB found in the source project, **none copied into the repository** (all flagged in `docs/large_file_audit.csv`): two ~82MB raw third-party Yachida supplementary workbooks, one 51MB MiMeDB reference dump, one 16MB Franzosa supplementary workbook (all third-party — do not redistribute, obtain from source per `data/README.md`), plus 7 thesis-derived-but-oversized files (hyperparameter search tables, iHMP annotation table, Yachida processed microbiome matrices, two Stage 4E loss-convergence diagnostic tables) recommended for Git LFS or a data/artifact release rather than direct git storage, and one 3.8MB nested `Archive.zip` (contents not opened or audited — excluded per policy). **No file exceeds 100MB** — no GitHub hard-limit violation.

## I. Data files withheld for licensing/privacy

All raw cohort/reference data (Franzosa, iHMP, Sinha, Yachida, MiMeDB) withheld — see `data/README.md` for sourcing instructions. Additionally flagged for **author review** (not withheld, but not cleared either): `preprocessing/ibd/tables/included_samples.csv`/`excluded_samples.csv` and CRC analogues contain per-participant clinical metadata (age, BMI, calprotectin, medication flags) alongside de-identified sample IDs — see `docs/privacy_security_audit.md` for the specific recommendation.

## J. Duplicate/backups removed

`ibd_pipeline/postrun_backups/` (156MB, ~890 files, confirmed pure duplicate of `ibd_pipeline/tables/` and `figures/`) — entirely excluded. `crc_pipeline/stage1_release/` (10MB, confirmed byte-identical older snapshot) and `crc_pipeline/stage2/tables/pathway/` (confirmed duplicate of `biological_sensitivity_shared_space/`) — excluded. 115 loose top-level duplicate IBD figures (already organized under subfolders) — excluded, only the subfolder copies kept. All `__pycache__/`, `.pytest_cache/`, `.ipynb_checkpoints/`, `.DS_Store` — excluded throughout.

## K. Absolute paths still present

37 occurrences logged in `docs/portability_audit.csv`. One was in genuinely active code (`final_analysis/code/run_stage4e_gcn_corrected_v2.py`, a string-literal example command) — **fixed** to a repository-relative example. The remaining 36 are inside historical/test/dual-placed-preprocessing files, deliberately left unmodified per the instruction not to alter historical evidence files merely to hide provenance; the two live path-resolution modules that matter for actually running the pipeline (`stage4_corrected_v2_common.py`, `stage3_config.py`) already support a `THESIS_PROJECT_ROOT` environment-variable override and contain no hardcoded paths.

## L. Privacy/security findings

**PASS** on secrets/credentials/API-keys/tokens/private-keys (zero matches, full-repository regex scan) and on email addresses/direct identifiers (zero matches). **One REVIEW item**: participant-level clinical metadata redistribution terms (see I above and `docs/privacy_security_audit.md`) — not a violation, but not independently verifiable by this process, so flagged rather than assumed clear.

## M. Missing documentation

None identified against the requested structure — all 22 requested `docs/` files plus the top-level README, LICENSE, CITATION.cff, `.gitignore`, `requirements.txt`, and folder-level READMEs (`final_analysis/`, `exploratory/`, `tests/`, `data/`, `results/`, `thesis/`, `analyses/`, plus per-pipeline preprocessing READMEs written by the curating agents) are present.

## N. Potentially misleading stale results

The clearest risk is the **original frozen Stage 4B table/figure** (`final_analysis/tables/final_stage4/table2_stage4b_transportability.csv`, `final_analysis/figures/final_stage4/figure_B_stage4b_transportability.png`), which still say "Microbiome principal / Metabolome stress test." These are deliberately preserved unmodified for provenance (see `final_analysis/README.md` for why) but are **not** what `results/final/` links to — the co-primary-relabelled versions are used there instead. A reader who browses `final_analysis/` directly rather than starting from its README could still stumble on the old wording; the README is written to make this explicit, but it relies on the reader actually reading it. The entire `exploratory/historical_analyses/stage4_pre_correction_v1/` tree is the other major stale-result risk, mitigated by its own prominent README warning.

## O. GitHub files >100MB

None. Largest file in the repository is well under 25MB.

## P. Safe to initialise/upload?

**READY AFTER MANUAL REVIEW.** The repository is structurally complete, safety-audited (no secrets, no raw restricted data, no files exceeding GitHub's size limits), and the final analysis is clearly distinguished from historical/exploratory material throughout. It is not marked fully READY because of the two items in §Q below that require a decision only you can make.

## Q. Exact remaining manual actions for you

1. **Choose a code licence.** `LICENSE` is currently a placeholder explaining the options — pick one, replace the file, update `CITATION.cff` and README §20.
2. **Resolve the participant-metadata review item** in `docs/privacy_security_audit.md` — confirm each cohort's data-use terms permit redistributing the per-participant clinical metadata tables as currently included, or replace them with a summary/aggregate version.
3. **Fill in author/institution placeholders** in `CITATION.cff`, `README.md` §18–19, and the thesis citation placeholder.
4. **Decide on the 7 oversized-but-thesis-derived files** flagged in `docs/large_file_audit.csv` (Git LFS vs. a Zenodo/OSF data release vs. omission) — none were added to git by default, per instruction not to auto-enroll in LFS without asking.
5. **Add the final thesis PDF** to `thesis/final_thesis.pdf` once you've identified and approved the submitted version (currently absent by design).
6. **Review the two `EXCLUDED_LARGE_FILE`... vs `EXCLUDE_FROM_GITHUB`... status label inconsistencies** if you want a single canonical status vocabulary — the three contributing curation agents used slightly different literal status strings (`EXCLUDED_LARGE_FILE`, `EXCLUDED`, `excluded`) for conceptually the same outcome; `classification` (the controlled taxonomy) is consistent throughout, but `status` (free text) is not. This does not affect correctness, only cosmetic consistency of `docs/file_provenance_map.csv`.
7. Only after 1–5 above: initialise git (`git init`), review `git status`/`.gitignore` once more, and push.

---

## READY AFTER MANUAL REVIEW
