# GitHub Readiness Report

Curated repository: `github_ibd_crc_multiomics/` (142MB working tree + 44MB local `.git`, 1,306 tracked files). Source project (`thesis_preprocessing_workflow/`, 788MB) confirmed untouched throughout — zero files modified during the entire curation process.

**Update (2026-08-24): the four pre-publication review items have been resolved.** This supersedes the previous version of this report, which left two manual-review items open.

## Resolution of the four review items

### 1. Participant-level clinical metadata — RESOLVED (withheld)

An initial spot-check found 2 files with participant-level clinical metadata. A **systematic full-repository re-scan** (every CSV, searching for a sample/participant-ID column co-occurring with any clinical-covariate column) found this was far more widespread: **33 files total**, across both IBD and CRC pipelines, including per-participant age/BMI/sex/smoking/calprotectin/medication flags and CRC disease-stage/trajectory labels, and the complete set of `crc_stage2_yachida_*_model_prep.csv` training tables. **All 33 were removed** — the conservative route, per instruction: de-identified IDs and public-cohort status do not by themselves establish a right to redistribute a reformatted per-participant table. Processing **scripts**, aggregate/summary tables, and molecular-only data were retained. Full list and regeneration instructions: `docs/file_provenance_map.csv` (`status = WITHDRAWN_PARTICIPANT_LEVEL_METADATA`) and `data/README.md`. Verified clean by re-running the same scan after removal.

### 2. Code licence — RESOLVED (MIT)

Audited the entire codebase for copyright headers, licence notices, or "adapted from"/attribution comments indicating copied third-party code — **none found**. MIT Licence adopted (`LICENSE`), with explicit scope language in `LICENSE`, `README.md` §20, `CITATION.cff`, and `docs/data_provenance.md` stating it applies only to the original code and does not extend to third-party cohort data, KEGG/HMDB/NJC19/MiMeDB resources, cited publications, or the dissertation itself.

### 3. Git staging dry run — DONE

`git init` + `git add -A` performed inside the curated directory (no remote configured, nothing pushed). Findings:
- **1,306 files staged** (33 fewer than the 1,338 pre-review total, matching the withdrawn participant-metadata files exactly).
- **Zero staged files exceed 10MB** — verified directly against `git diff --cached --name-only`, not just the working-tree listing. Largest staged file is ~6.97MB (`crc_stage2_hyperparameter_search_results.csv`).
- **Secret/credential scan re-run against exactly the staged file list** (not just the working tree) — zero matches; the two regex hits found were confirmed as this report and the privacy audit *describing* the scan methodology, not actual secrets.
- **Participant-metadata pattern re-scan against the staged list** — zero matches, confirming the withdrawal in item 1 is reflected in what would actually be pushed.
- A local commit was made (`90cd3f0`, "Initial curated repository (local dry-run commit, not pushed)") to complete the dry run. **No remote was added and nothing was pushed.**

### 4. README / examiner first-look audit — DONE

Added an "At a glance" block at the very top of `README.md` (before the 20 numbered sections) explicitly covering all 8 required points within one short skim: what the thesis studied, the four cohorts, where the final analysis is, microbiome/metabolome co-primary status, sensitivity/exploratory/historical segregation, data-availability pointer, reproduction pointer, and the exact figure/table provenance files — each as a direct clickable link. Verified `final_analysis/README.md` and `analyses/03_cross_cohort_replication/README.md` (both referenced at the end of the top README) exist and are populated.

## Everything else from the original audit (unchanged, still valid)

## A. Repository structure created

Full target structure built: `docs/`, `data/`, `preprocessing/{ibd,crc,harmonisation,microbiome,metabolomics,qc}/`, `analyses/01–10/`, `final_analysis/{code,configs,manifests,provenance,tables,predictions,figures,reports,tests}/`, `exploratory/{alternative_models,alternative_preprocessing,diagnostic_analyses,discarded_or_infeasible_ideas,historical_analyses}/`, `notebooks/{preprocessing,exploration,modelling}/`, `tests/`, `results/{final,sensitivity,exploratory}/`, `thesis/{figure_provenance,table_provenance}/`.

## B–G. File counts (updated post-withdrawal)

- Files reviewed: 1,945 (logged in `docs/file_provenance_map.csv`)
- Files copied/tracked in git: **1,306**
- Files excluded: 639 (528 original exclusions + 33 withdrawn participant-metadata files re-classified, net of 5 rows that were duplicate-logged and consolidated)
- `FINAL_AUTHORITATIVE`: 479 · `SENSITIVITY`: 55 · `HISTORICAL_SUPERSEDED`: 625

## H. Large files requiring attention

12 files >10MB in the source project, none copied/staged (verified twice — working tree and git index). See `docs/large_file_audit.csv`. No file exceeds 100MB.

## I. Data files withheld for licensing/privacy

All raw cohort/reference data (unchanged from original audit) **plus** the 33 participant-level clinical metadata files (new, see item 1 above). Nothing remains flagged as an open review item — both categories are now definitively withheld, not pending.

## J. Duplicate/backups removed

Unchanged from original audit: `postrun_backups/` (156MB), `stage1_release/` duplicate snapshot, duplicate pathway tables, 115 loose duplicate figures, all cache directories.

## K. Absolute paths still present

37 occurrences in historical/test files (unchanged); one active-code occurrence fixed. See `docs/portability_audit.csv`.

## L. Privacy/security findings

**PASS**, no remaining review items (see item 1 and item 3 above — resolved and re-verified against the staged git index specifically).

## M. Missing documentation

None.

## N. Potentially misleading stale results

Unchanged: the original frozen Stage 4B table/figure and the pre-correction Stage 4 pipeline remain, deliberately, with prominent warnings.

## O. GitHub files >100MB

None (double-verified: working tree and git index).

## P. Safe to initialise/upload?

**READY.**

## Q. Remaining manual actions for you

1. Fill in the author/institution placeholders in `CITATION.cff`, `README.md` §18–19, and `LICENSE`'s copyright line.
2. Decide on the 7 oversized-but-thesis-derived files in `docs/large_file_audit.csv` (Git LFS vs. Zenodo/OSF vs. omission) — not blocking, since none are currently tracked.
3. Add the final thesis PDF to `thesis/final_thesis.pdf` once approved (absent by design — do not add the draft).
4. Create the GitHub account and an **empty** public repository (no auto-generated README/licence/`.gitignore`) under a clean name (e.g. `ibd-crc-multiomics-reproducibility`) — this is a step only you can take; I cannot create GitHub accounts or repositories.
5. Once the empty repository exists, tell me its URL and I can add it as a remote and push the already-committed local repository (`git remote add origin <url>` + `git push -u origin main`) — I will ask for your explicit confirmation before pushing, per standard practice for actions visible to others.
6. Finalise the thesis's Code Availability sentence with the real repository URL once step 4 is done.

---

## READY
