"""Generate POST_FREEZE_CODE_PATCH_MANIFEST.json.

The Stage 4 authority freeze (`FINAL_STAGE4_AUTHORITY_MANIFEST.json`) is
dated 20 August 2026. Several scripts and the test suite were edited on
2026-08-22/24, after that freeze — so the exact code in this checkout is no
longer literally the same code state that produced the 29 frozen
authoritative files. This is expected post-freeze hardening (microbiome
missingness semantics, a new sensitivity branch, QA/provenance tooling), not
a rerun of the primary analysis, but it must be recorded rather than left
implicit.

No git repository is present in this project directory, so pre-patch hashes
were not captured automatically by version control; this manifest instead
records the current (post-patch) hash of each changed file plus a curated
description of what changed and why, and an explicit assessment of whether
the change can affect any already-frozen numerical result. Recommendation:
initialise git for this project going forward (see the reproducibility items
in the change register) so future patches get real before/after commit
hashes for free.
"""

from __future__ import annotations

from datetime import datetime, timezone

from stage4_corrected_v2_common import MANIFESTS, PROJECT, SCRIPTS, atomic_json, sha256


TESTS = SCRIPTS.parent / "tests"
CFG_STAGE3_SCRIPTS = PROJECT / "thesis_preprocessing_workflow" / "stage3_ibd_crc_integration" / "scripts"

FREEZE_TIMESTAMP_UTC = "2026-08-20T08:47:40.049787+00:00"

CHANGED_FILES = [
    {
        "path": SCRIPTS / "stage4_corrected_v2_common.py",
        "change_summary": (
            "(a) Added microbiome_missingness_audit() distinguishing NaN from observed "
            "zero and excluding wholly-unmeasured features from authoritative_genus_matrix() "
            "instead of silently zero-filling them. (b) Added _require_no_unresolved_missing(): "
            "MicrobiomeCLRPreprocessor/MicrobiomeRankPreprocessor/MicrobiomePresenceAbsencePreprocessor "
            "now raise on any unresolved NaN reaching fit/transform by default, instead of "
            "generically fillna(0.0); allow_zero_fill_missing=True is an explicit opt-in escape "
            "hatch. (c) Added MicrobiomeRankPreprocessor and MicrobiomePresenceAbsencePreprocessor "
            "for the CRC sensitivity branch; MicrobiomePresenceAbsencePreprocessor now fits a "
            "training-fold StandardScaler on the binary indicator (previously raw 0/1) so its L2 "
            "penalty geometry matches the other microbiome variants. (d) Softened the "
            "microbiome_missingness_audit docstring's earlier overclaim about NaN-fill being "
            "'the correct convention'. (e) build_ridge_pipeline/microbiome_pipeline gained an "
            "optional microbiome_variant argument (default 'clr_zscore', i.e. unchanged primary "
            "behaviour)."
        ),
        "can_affect_frozen_numerical_results": False,
        "why": (
            "On the real authoritative genus matrices, only one feature in one cohort "
            "(Franzosa 'genus:steroids') was ever wholly-NaN, and it was already excluded from "
            "the frozen Stage 4A output under the pre-existing species-only-aggregation policy "
            "for Franzosa (rank='genus', not 'species'), independent of this patch. No partial "
            "(non-wholly-missing) NaN exists anywhere in the four raw microbiome source tables, "
            "so _require_no_unresolved_missing's stricter default never triggers on real data "
            "(verified: run_final_stage4_qa.py and run_stage4_post_freeze_equivalence_check.py "
            "both pass with default settings; the genus-matrix reconstruction check reproduces "
            "the frozen output numerically exactly). The new default microbiome_variant='clr_zscore' "
            "keyword preserves the exact prior call signature/behaviour of build_ridge_pipeline "
            "for every existing call site in run_stage4b_corrected_v2b.py, run_stage4c_corrected_v2.py "
            "and run_stage4e_gcn_corrected_v2.py."
        ),
        "equivalence_checks_performed": [
            "run_final_stage4_qa.py: all 10 checks pass, including authoritative_files_unmutated (sha256 of the 29 frozen files is unchanged)",
            "run_stage4_post_freeze_equivalence_check.py: reconstructed authoritative genus matrices for Franzosa/iHMP/Sinha/Yachida numerically match the frozen outputs exactly; shared IBD=50/CRC=52 genus space sizes unchanged",
            "run_stage4_post_freeze_equivalence_check.py --run-clr-replication: PASS on 2026-08-24 — the current code's MICROBIOME nested-CV/final-external protocol reproduces the frozen stage4b_corrected_v2b external ROC-AUC/PR-AUC for all 4 TaskSpecs within 1e-6 tolerance",
            "corrected_v2/tests/test_corrected_v2_integrity.py: 50/50 tests pass",
            "2026-08-24 follow-up fix: authoritative_genus_matrix()'s own genus-summation step also called fillna(0.0) on the SELECTED (non-wholly-missing) columns, meaning any partial NaN in an included column would have bypassed the preprocessor-level guard entirely by being zero-filled upstream. Added the same _require_no_unresolved_missing guard (allow_zero_fill_missing=False by default) at that summation step. Verified no-op on real data (all 4 cohorts' authoritative genus matrices still reconstruct identically; 50/50 tests still pass).",
        ],
    },
    {
        "path": SCRIPTS / "run_stage4c_corrected_v2.py",
        "change_summary": (
            "(a) build_scca_plan()'s correlation_reporting field changed from the stale "
            "'held_out_and_full_data_reported_separately' to "
            "'tuning_fold_diagnostics_and_apparent_full_data_reported_separately', matching the "
            "already-frozen policy that these are tuning-fold diagnostics, not unbiased held-out "
            "estimates. (b) The --scca-nested-cv CLI flag was removed (hidden) because "
            "run_scca_nested_cv() deliberately raises NotImplementedError; the function itself "
            "is retained as scaffolding for a future nested-CV implementation, but a "
            "production-facing CLI should not expose an option that always fails."
        ),
        "can_affect_frozen_numerical_results": False,
        "why": (
            "Both changes are metadata/CLI-surface only. correlation_reporting is a plan-table "
            "label, not a fitted value; no PMA/sCCA model was refit. --scca-nested-cv's code path "
            "already unconditionally raised NotImplementedError before this patch, so removing "
            "the flag cannot change any output that previously existed (the path was never "
            "reachable to completion)."
        ),
        "equivalence_checks_performed": [
            "stage4c_corrected_v2_scca_plan.csv regenerated via build_scca_plan() (no model fitting) and confirmed to contain only the corrected terminology string",
            "corrected_v2/tests/test_corrected_v2_integrity.py sCCA-related tests (4/4) pass",
        ],
    },
    {
        "path": SCRIPTS / "audit_stage4a_microbiome_missingness.py",
        "change_summary": "New file. Runs microbiome_missingness_audit() across all four cohorts and writes stage4a_microbiome_missingness_audit.csv / _summary.csv.",
        "can_affect_frozen_numerical_results": False,
        "why": "Read-only audit script; does not feed into any modelling pipeline.",
        "equivalence_checks_performed": ["Run directly; output inspected manually."],
    },
    {
        "path": SCRIPTS / "run_stage4b_crc_microbiome_sensitivity.py",
        "change_summary": "New file. CRC-only, MICROBIOME-only sensitivity branch reusing the frozen Stage 4B v2b nested-CV/threshold/bootstrap design across clr_zscore/within_sample_rank/presence_absence variants. Writes to a distinct sensitivity_crc_microbiome_* prefix.",
        "can_affect_frozen_numerical_results": False,
        "why": "Writes only to new sensitivity_crc_microbiome_* output paths; never imports from or writes to stage4b_corrected_v2b_* primary outputs.",
        "equivalence_checks_performed": [
            "--dry-run passes",
            "Full sweep executed 2026-08-24: primary_replication_check PASS (clr_zscore arm reproduces frozen 0.562961/0.581862 Sinha->Yachida/Yachida->Sinha ROC-AUC exactly); wrote external/internal predictions, external/internal performance, fold diagnostics, selected-C/threshold table, source-OOF threshold predictions, paired rank/PA-vs-CLR BH-FDR comparisons, and run manifest.",
        ],
    },
    {
        "path": SCRIPTS / "run_final_stage4_qa.py",
        "change_summary": "New file. Consolidated QA entrypoint (pytest suite + 10 direct checks: authoritative-file hash drift, duplicate participants/re-closure with explicit isfinite guard, cohort counts, non-finite scan across Stage 4A-4E with a whitelist for structural NaN, feature-order-at-transfer, fixed-seed reproducibility, figure provenance, final reporting-validation-table pass/fail, load-bearing code hash recording).",
        "can_affect_frozen_numerical_results": False,
        "why": "Read-only verification script; does not write to any authoritative path.",
        "equivalence_checks_performed": ["Run directly; all 10 checks currently PASS."],
    },
    {
        "path": SCRIPTS / "run_stage4_post_freeze_equivalence_check.py",
        "change_summary": "New file. Reconstructs the authoritative genus matrices/shared genus spaces from raw source data and diffs them against the frozen outputs; optionally (behind --run-clr-replication) reruns the Stage 4B v2b MICROBIOME nested-CV protocol and compares external AUCs to frozen values within 1e-6 tolerance.",
        "can_affect_frozen_numerical_results": False,
        "why": "Read-only reconstruction/comparison; only writes to a new manifest path.",
        "equivalence_checks_performed": ["Run directly (cheap part only); genus_matrix_reconstruction and shared_genus_space_sizes both PASS. --run-clr-replication not yet executed — pending manual run."],
    },
    {
        "path": SCRIPTS / "generate_post_freeze_patch_manifest.py",
        "change_summary": "New file (this script).",
        "can_affect_frozen_numerical_results": False,
        "why": "Provenance tooling only.",
        "equivalence_checks_performed": [],
    },
    {
        "path": SCRIPTS / "generate_post_freeze_analysis_role_addendum.py",
        "change_summary": (
            "New file, then corrected 2026-08-24: writes POST_FREEZE_ANALYSIS_ROLE_ADDENDUM.json "
            "and post_freeze_claim_policy_addendum.csv recording the co-primary MICROBIOME/METABOLOME "
            "re-framing (EARLY_INTEGRATION remains integration_comparison, not co-primary). Does not "
            "edit FINAL_STAGE4_AUTHORITY_MANIFEST.json or final_stage4_claim_policy.csv in place. "
            "2026-08-24 fix: the claim-policy row originally said co-primary status itself was "
            "'prespecified', which overstates the historical record (the 20-Aug freeze explicitly "
            "called microbiome principal / metabolome a stress test). Reworded to: the SINGLE-OMIC "
            "REPRESENTATIONS and their shared modelling/validation protocol were prespecified; the "
            "co-primary FRAMING is a final-thesis-reporting decision applied on top of that, not a "
            "retroactive claim about original intent."
        ),
        "can_affect_frozen_numerical_results": False,
        "why": "Pure metadata/documentation; no numeric computation.",
        "equivalence_checks_performed": [],
    },
    {
        "path": SCRIPTS / "generate_post_freeze_co_primary_display_layer.py",
        "change_summary": (
            "2026-08-24 fixes: (a) panels were ordered by alphabetical groupby('direction'), which "
            "scattered IBD and CRC directions across the 2x2 grid (ASCII sorts 'iHMP->Franzosa' after "
            "all uppercase-first directions); now explicitly ordered top row IBD "
            "(Franzosa<->iHMP), bottom row CRC (Sinha<->Yachida). (b) the italic footnote collided "
            "with the shared x-axis label; increased figure height and bottom margin, moved supxlabel "
            "and footnote to distinct y-positions, removed bbox_inches='tight' (which was overriding "
            "the manual margins). (c) the microbiome external_interpretation_scope caveat was CRC-"
            "specific (16S vs. shotgun) but was being applied to IBD rows too, where it is irrelevant; "
            "split into disease-specific IBD (shotgun-vs-shotgun, database/processing heterogeneity) "
            "and CRC (16S-vs-shotgun, platform-confounded-with-cohort) wording. Same-day follow-up "
            "readability pass: base font 9pt->13pt, panel titles 15pt bold, suptitle 17pt; per-panel "
            "y-tick labels shortened (role wording moved out of every one of the 12 ticks and into a "
            "single explicit legend instead, added via matplotlib Line2D proxy handles at the top of "
            "the figure) since repeating '(co-primary)'/'(comparator)' 12 times was both redundant and "
            "crowding the labels at larger font; marker size increased with a white edge for contrast; "
            "explicit subplots_adjust margins (replacing tight_layout) and axis y-margins added so no "
            "text touches the legend, panel titles, or footnote at the larger font sizes. Color mapping "
            "(Okabe-Ito palette, same MICROBIOME/METABOLOME/EARLY_INTEGRATION hex values as REP_COLORS "
            "in freeze_final_stage4_results.py) is unchanged, just now also declared in the legend."
        ),
        "can_affect_frozen_numerical_results": False,
        "why": "Purely a re-render of the same already-frozen stage4b_corrected_v2b_external_performance.csv numbers; the underlying values printed by the script are unchanged (verified 2026-08-24: identical 12 ROC-AUC values before and after this fix).",
        "equivalence_checks_performed": ["Rerun 2026-08-24; printed table values identical to the prior run and to stage4b_corrected_v2b_external_performance.csv."],
    },
    {
        "path": CFG_STAGE3_SCRIPTS / "generate_stage2_3_manuscript_claim_provenance.py",
        "change_summary": (
            "New 2026-08-24 (and corrected same day). Adds a Stage 2/3 claim->authoritative-CSV "
            "provenance map, analogous to the Stage 4 figure-source map. Every value is asserted "
            "against its live source CSV at generation time (AssertionError on drift, not a silent "
            "stale number). Corrected same day: the Stage 3B Yachida claim originally said the 2 "
            "HC3-significant + 6 Spearman-significant metabolites showed a 'monotonic/ordered trend', "
            "conflating two different tests; the 2 HC3 results are ordered-state LINEAR associations "
            "(citrulline is profile_pattern='irregular', N6,N6,N6-trimethyllysine is 'early_peak' — "
            "neither is monotonic), while only the separate 6-metabolite Spearman-rank sensitivity "
            "tests monotonicity directly. Reworded and added an assertion that guards against this "
            "language regressing. Also expanded from 5 to 9 claims: added exact IBD-vs-Control "
            "metabolite names, CD/UC subtype counts, and three negative-conclusion claims (CRC strict "
            "replication=0, cross-disease Spearman BH-significant=0, directionally-coherent "
            "Stage3A+3B FDR candidates=0)."
        ),
        "can_affect_frozen_numerical_results": False,
        "why": "Read-only provenance/assertion script; writes only to stage2_3_manuscript_claim_provenance.{csv,json} in stage3_ibd_crc_integration/manifests/.",
        "equivalence_checks_performed": ["Rerun 2026-08-24; all 9 claims' assertions pass against live source CSVs."],
    },
    {
        "path": CFG_STAGE3_SCRIPTS / "run_stage3.py",
        "change_summary": (
            "2026-08-24: relabelled two rows in stage3c_results_numbers()'s output table. "
            "'IBD replicated events'=10 was easy to misread as 10 IBD-vs-Control metabolites; it "
            "is actually IBD-vs-Control (7) + CD-vs-Control (3) replication events summed across two "
            "different contrasts. Renamed to 'IBD-vs-Control + CD-vs-Control replicated events "
            "(summed across contrasts)' and 'unique IBD replicated IDs' to 'unique IBD-vs-Control "
            "replicated IDs'. Values unchanged (still 10 and 7)."
        ),
        "can_affect_frozen_numerical_results": False,
        "why": (
            "Label-text-only change to two string literals inside a function; the numeric values (10, 7) "
            "are unchanged. This function is not called by this patch — the full Stage 3 pipeline was "
            "NOT rerun, so tables/stage3_final_results_numbers.csv on disk is untouched by this edit; "
            "the fix only takes effect on a future full Stage 3 rerun. A corrected COPY was produced "
            "separately (see relabel_stage3_final_results_numbers.py) without running the pipeline."
        ),
        "equivalence_checks_performed": ["Source diff reviewed manually; only the two label strings changed, the literal values 10 and 7 are untouched."],
    },
    {
        "path": CFG_STAGE3_SCRIPTS / "relabel_stage3_final_results_numbers.py",
        "change_summary": "New file 2026-08-24. Reads the existing (untouched) stage3_final_results_numbers.csv and writes a relabelled copy at stage3_final_results_numbers_relabelled.csv, without running the Stage 3 pipeline.",
        "can_affect_frozen_numerical_results": False,
        "why": "Reads an existing CSV and writes a renamed-column copy at a new path; does not touch the original file or compute anything new.",
        "equivalence_checks_performed": ["Run 2026-08-24; output values (10, 7) match the original file exactly, only the 'result' label text differs."],
    },
    {
        "path": TESTS / "test_corrected_v2_integrity.py",
        "change_summary": (
            "Added tests for microbiome_missingness_audit, wholly-missing-feature exclusion, "
            "the new raise-by-default/allow_zero_fill_missing opt-in behaviour, the rank/presence-"
            "absence preprocessors (including the new training-fold scaling for presence/absence), "
            "and the build_ridge_pipeline microbiome_variant switch. Renamed "
            "test_threshold_selection_uses_cross_fitted_probabilities_not_resubstitution to "
            "test_legacy_probability_scale_threshold_helper_still_uses_cross_fitted_oof_not_resubstitution "
            "with a docstring clarifying it covers the legacy probability-scale helper, not the "
            "authoritative decision-score pipeline."
        ),
        "can_affect_frozen_numerical_results": False,
        "why": "Test-only changes; 50/50 tests pass.",
        "equivalence_checks_performed": ["pytest -q: 50 passed"],
    },
]


def main() -> None:
    entries = []
    for item in CHANGED_FILES:
        path = item["path"]
        entries.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "current_sha256": sha256(path) if path.exists() else None,
                "pre_patch_sha256": "not_captured_no_vcs_in_project_directory",
                "change_summary": item["change_summary"],
                "can_affect_frozen_numerical_results": item["can_affect_frozen_numerical_results"],
                "why": item["why"],
                "equivalence_checks_performed": item["equivalence_checks_performed"],
            }
        )
    manifest = {
        "status": "POST_FREEZE_CODE_PATCH",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "authority_freeze_timestamp_utc": FREEZE_TIMESTAMP_UTC,
        "note": (
            "This patch is post-freeze hardening and new sensitivity/QA tooling. It does not "
            "modify, and per run_final_stage4_qa.py's authoritative_files_unmutated check has not "
            "changed the sha256 of, any of the 29 frozen authoritative Stage 4 output files. See "
            "run_stage4_post_freeze_equivalence_check.py for direct numerical reconstruction "
            "evidence for Stage 4A, and run it with --run-clr-replication for Stage 4B MICROBIOME."
        ),
        "no_version_control_caveat": "This project directory is not a git repository, so true before/after commit hashes were not available; only current-state hashes are recorded here.",
        "changed_files": entries,
    }
    atomic_json(manifest, MANIFESTS / "POST_FREEZE_CODE_PATCH_MANIFEST.json")
    print(f"Wrote {MANIFESTS / 'POST_FREEZE_CODE_PATCH_MANIFEST.json'} ({len(entries)} files recorded)")


if __name__ == "__main__":
    main()
