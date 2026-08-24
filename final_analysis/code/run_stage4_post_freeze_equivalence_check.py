"""Post-freeze equivalence check: does the current (patched) code still
reconstruct the frozen 20-August Stage 4A/4B authoritative outputs?

`run_final_stage4_qa.py`'s `authoritative_files_unmutated` check only proves
the frozen *files on disk* have not changed. It says nothing about whether
the code that would currently generate them still produces the same numbers,
because `stage4_corrected_v2_common.py` has been edited since the freeze
(see `POST_FREEZE_CODE_PATCH_MANIFEST.json`). This script closes that gap
with two checks:

  1. [cheap, run automatically by `main()`] Reconstruct the four authoritative
     genus-proportion matrices and the shared IBD (50-genus) / CRC (52-genus)
     feature spaces from the raw source tables using the current
     `authoritative_genus_matrix()`, and diff them numerically against the
     frozen `*_authoritative_genus_proportions.csv` files and shared-genera
     lists. Pure data reconstruction, no model fitting — safe to run here.

  2. [real model fitting, NOT run automatically] `microbiome_clr_zscore_
     replication_check()` refits the exact frozen Stage 4B v2b nested-CV +
     final-external protocol for the MICROBIOME representation on all four
     TaskSpecs and compares the resulting external ROC-AUC/PR-AUC against
     the frozen `stage4b_corrected_v2b_external_performance.csv` values
     within a prespecified tolerance. This is real (if small/fast) logistic
     regression fitting across 4 tasks x nested outer/inner CV, so it is left
     for manual execution rather than being run as part of this script's
     default invocation:

         python3 run_stage4_post_freeze_equivalence_check.py --run-clr-replication

Tolerance: ROC-AUC/PR-AUC agreement within 1e-6 is expected if nothing
numerically relevant changed (StratifiedKFold with a fixed random_state is
deterministic), since none of the post-freeze edits altered the fold-local
prevalence filter, reclosure, pseudocount, CLR formula, or scaler fitting
for the primary (allow_zero_fill_missing=False, and no NaN reaches this path
on real data) code path.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from stage4_corrected_v2_common import (
    MANIFESTS,
    MICROBIOME_INPUTS,
    TABLES,
    TASK_SPECS,
    V2,
    atomic_json,
    authoritative_genus_matrix,
    read_csv,
)


ROC_PR_TOLERANCE = 1e-6


def check_genus_matrix_reconstruction() -> dict:
    issues = []
    per_cohort = {}
    for cohort, path in MICROBIOME_INPUTS.items():
        raw = read_csv(path)
        live_genus, _ = authoritative_genus_matrix(cohort, raw)
        frozen_path = V2 / "data" / "processed" / f"{cohort.lower()}_authoritative_genus_proportions.csv"
        if not frozen_path.exists():
            issues.append(f"{cohort}: frozen authoritative genus table not found at {frozen_path}")
            continue
        frozen = read_csv(frozen_path)
        frozen["sample_id"] = frozen["sample_id"].astype(str)
        live_sorted = live_genus.set_index("sample_id").sort_index(axis=0).sort_index(axis=1)
        frozen_sorted = frozen.set_index("sample_id").sort_index(axis=0).sort_index(axis=1)
        same_shape = live_sorted.shape == frozen_sorted.shape
        same_columns = list(live_sorted.columns) == list(frozen_sorted.columns)
        same_index = list(live_sorted.index) == list(frozen_sorted.index)
        numerically_equal = same_shape and same_columns and same_index and np.allclose(live_sorted.to_numpy(), frozen_sorted.to_numpy(), atol=1e-9)
        per_cohort[cohort] = {
            "live_shape": list(live_sorted.shape),
            "frozen_shape": list(frozen_sorted.shape),
            "same_columns": same_columns,
            "same_index": same_index,
            "numerically_equal": bool(numerically_equal),
        }
        if not numerically_equal:
            issues.append(f"{cohort}: reconstructed authoritative genus matrix differs from frozen output")
    return {"ok": not issues, "per_cohort": per_cohort, "issues": issues}


def check_shared_genus_space_sizes() -> dict:
    ibd_shared = read_csv(TABLES / "stage4a_corrected_v2_ibd_shared_genera.csv")
    crc_shared = read_csv(TABLES / "stage4a_corrected_v2_crc_shared_genera.csv")
    ok = len(ibd_shared) == 50 and len(crc_shared) == 52
    return {"ok": ok, "ibd_shared_n": len(ibd_shared), "crc_shared_n": len(crc_shared), "expected": {"ibd": 50, "crc": 52}}


def microbiome_clr_zscore_replication_check() -> dict:
    """Real model fitting: reruns the frozen Stage 4B v2b MICROBIOME nested-CV
    + final-external protocol and compares against frozen external performance.
    Not run by default from main() — pass --run-clr-replication."""
    from run_stage4b_corrected_v2b import CorrectedDataBundle, aligned_train_external, final_external, nested_cv

    frozen = read_csv(TABLES / "stage4b_corrected_v2b_external_performance.csv")
    frozen = frozen[frozen["representation"].eq("MICROBIOME")]
    bundle = CorrectedDataBundle()
    rows = []
    for spec in TASK_SPECS:
        X_train, y_train, train_ids, X_ext, y_ext, ext_ids = aligned_train_external(bundle, spec, "MICROBIOME")
        _, _, _, _ = nested_cv(X_train, y_train, train_ids, "MICROBIOME", spec)
        _, _, _, metrics = final_external(X_train, y_train, train_ids, X_ext, y_ext, ext_ids, "MICROBIOME")
        frozen_row = frozen[(frozen["task"] == spec.task) & (frozen["training_cohort"] == spec.train_cohort) & (frozen["external_cohort"] == spec.external_cohort)]
        frozen_roc = float(frozen_row["roc_auc"].iloc[0]) if len(frozen_row) else float("nan")
        frozen_pr = float(frozen_row["pr_auc"].iloc[0]) if len(frozen_row) else float("nan")
        rows.append(
            {
                "task": spec.task,
                "training_cohort": spec.train_cohort,
                "external_cohort": spec.external_cohort,
                "live_roc_auc": metrics["roc_auc"],
                "frozen_roc_auc": frozen_roc,
                "roc_auc_match": abs(metrics["roc_auc"] - frozen_roc) <= ROC_PR_TOLERANCE,
                "live_pr_auc": metrics["pr_auc"],
                "frozen_pr_auc": frozen_pr,
                "pr_auc_match": abs(metrics["pr_auc"] - frozen_pr) <= ROC_PR_TOLERANCE,
            }
        )
    result = pd.DataFrame(rows)
    ok = bool(result["roc_auc_match"].all() and result["pr_auc_match"].all())
    return {"ok": ok, "tolerance": ROC_PR_TOLERANCE, "comparisons": result.to_dict(orient="records")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-freeze equivalence check for Stage 4A/4B.")
    parser.add_argument("--run-clr-replication", action="store_true", help="Also rerun the real Stage 4B v2b MICROBIOME nested-CV protocol and compare to frozen AUCs (real model fitting, several minutes).")
    args = parser.parse_args()

    checks = {
        "genus_matrix_reconstruction": check_genus_matrix_reconstruction(),
        "shared_genus_space_sizes": check_shared_genus_space_sizes(),
    }
    if args.run_clr_replication:
        checks["microbiome_clr_zscore_replication"] = microbiome_clr_zscore_replication_check()
    else:
        checks["microbiome_clr_zscore_replication"] = {"ok": None, "note": "not run; pass --run-clr-replication to execute (real model fitting)"}

    overall_ok = all(v.get("ok") is not False for v in checks.values())
    report = {"overall_status": "PASS" if overall_ok else "FAIL", "checks": checks}
    atomic_json(report, MANIFESTS / "stage4_post_freeze_equivalence_check.json")
    for name, result in checks.items():
        status = "PASS" if result.get("ok") else ("SKIPPED" if result.get("ok") is None else "FAIL")
        print(f"[{status}] {name}")
    print(f"\nOverall: {'PASS' if overall_ok else 'FAIL'}")
    if not overall_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
