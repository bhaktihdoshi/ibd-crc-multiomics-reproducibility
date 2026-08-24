# Privacy & Security Audit

## Method

1. Regex scan across every copied `.py`/`.R`/`.sh`/`.json`/`.md`/`.txt`/`.csv`/`.yml`/`.cfg`/`.ini` file in the curated repository for credential/secret patterns: API keys, `password=`, `token=`, AWS access keys, PEM/SSH private-key headers.
2. Regex scan for email addresses across all `.py`/`.md`/`.json` files.
3. Filename scan for `.env`, `*credential*`, `*secret*`, `*.pem`, `*.key`.
4. Manual spot-check of participant-metadata tables for names, dates of birth, addresses, or other direct identifiers.
5. Each curation agent (IBD, CRC, Stage 3) independently ran the same class of scan on its slice before copying; findings below also incorporate their reports.

## Result: secrets/credentials — **PASS**

Zero matches for API keys, passwords, tokens, AWS credentials, or private-key material anywhere in the curated repository. No `.env`/credential/secret/key files present.

## Result: email addresses — **PASS**

Zero email addresses found in any copied code or documentation file.

## Result: direct identifiers (names, DOB, addresses) — **PASS**

No participant names, dates of birth, or addresses were found. All participant-level records use de-identified sample/participant codes only (e.g. `PRISM.7122`, `CRC.222`-style codes), consistent with the identifiers already used in the original public cohort depositions this project draws on.

## Result: quasi-identifying clinical metadata — **RESOLVED (withheld)**

An initial pass found `preprocessing/ibd/tables/included_samples.csv`/`excluded_samples.csv` containing per-participant clinical covariates (age, sex, BMI, smoking status, fecal calprotectin, medication flags) alongside de-identified sample IDs. A **second, systematic pass** across the entire repository (searching every CSV for a sample/participant-ID column co-occurring with any clinical-covariate column) found this pattern was far more widespread than the initial spot-check — **33 files total**, spanning both the IBD and CRC pipelines, including per-participant CRC disease-stage/trajectory labels and the full set of `crc_stage2_yachida_*_model_prep.csv` training tables.

**Decision (author, 2026-08-24): conservative route taken.** All 33 files were removed from the repository. De-identified sample IDs and public-cohort status do not by themselves establish a right to redistribute a reformatted, combined per-participant clinical table — that is a different act from linking to the original deposition. Rather than assess each cohort's licence terms individually under time pressure, every participant-level clinical file was withdrawn; only processing **scripts**, aggregate/summary tables, and molecular-only (no clinical covariate) data were retained. See `data/README.md` for the full list and how to regenerate these tables locally from the original sources.

This closes the review item — verified by a full re-scan of the repository after removal (see `docs/file_provenance_map.csv`, `status = WITHDRAWN_PARTICIPANT_LEVEL_METADATA`, 33 rows) confirming no remaining file combines a participant identifier with a clinical covariate.

## Result: filesystem/machine-specific information — **PASS (after action)**

Absolute local filesystem paths (`/Users/bhaktidoshi/...`) were found in 37 locations, all inside historical/test files or as one string-literal example command in an active script (`final_analysis/code/run_stage4e_gcn_corrected_v2.py`), which has been fixed to a repository-relative example. See `docs/portability_audit.csv` for the full list and disposition of each. None of these paths reveal sensitive personal information beyond the local username already visible in this project's own directory naming — no separate remediation was required for privacy purposes (this is a portability concern, addressed separately, not a privacy leak).

## Overall verdict

**PASS.** No secrets, credentials, emails, or direct identifiers were found anywhere in the curated repository. Participant-level clinical metadata (33 files) was identified via a systematic full-repository re-scan and withheld by author decision — see above.
