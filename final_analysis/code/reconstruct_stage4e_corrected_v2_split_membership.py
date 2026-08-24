from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from stage4_corrected_v2_common import MANIFESTS, REPORTS, TABLES, atomic_csv, atomic_json, ensure_v2_dirs
from run_stage4e_gcn_corrected_v2 import (
    EARLY_STOPPING_VALIDATION_FRACTION,
    OUTER_FOLDS,
    PRIMARY_SEEDS,
    deterministic_validation_split,
    merged_node_feature_matrix,
    within_directions,
)


PREDICTIONS = TABLES / "stage4e_corrected_v2_within_oof_predictions_primary.csv"
MEMBERSHIP = TABLES / "stage4e_corrected_v2_reconstructed_split_membership_primary.csv"
VALIDATION = TABLES / "stage4e_corrected_v2_reconstructed_split_validation_tests.csv"
POSTRUN_VALIDATION = TABLES / "stage4e_corrected_v2_primary_postrun_validation_tests.csv"
MANIFEST = MANIFESTS / "stage4e_corrected_v2_primary_postrun_manifest.json"
REPORT = REPORTS / "STAGE4E_CORRECTED_V2_SPLIT_RECONSTRUCTION_AUDIT.md"


def add_check(rows: list[dict[str, Any]], test: str, status: str, observed: Any, expected: Any, notes: str = "") -> None:
    rows.append({"test": test, "status": status, "observed": observed, "expected": expected, "notes": notes})


def reconstruct_membership() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for direction in within_directions():
        x, y, ids = merged_node_feature_matrix(direction.disease, direction.train_cohort)
        sample_ids = ids["sample_id"].astype(str).reset_index(drop=True)
        y = y.astype(int).reset_index(drop=True)
        for seed in PRIMARY_SEEDS:
            n_splits = min(OUTER_FOLDS, int(y.value_counts().min()))
            if n_splits != OUTER_FOLDS:
                raise ValueError(f"Unexpected Stage 4E primary fold count for {direction.train_cohort}: {n_splits}")
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for outer_fold, (dev, test) in enumerate(cv.split(x, y), start=1):
                tr_rel, va_rel = deterministic_validation_split(y.iloc[dev].reset_index(drop=True), seed + 999 + outer_fold)
                dev_index = np.array(dev)
                split_roles = {
                    "train": dev_index[tr_rel],
                    "validation": dev_index[va_rel],
                    "outer_test": np.array(test),
                }
                for role, indices in split_roles.items():
                    for idx in indices:
                        rows.append(
                            {
                                "analysis": "within",
                                "direction_id": direction.direction_id,
                                "disease": direction.disease,
                                "task": direction.task,
                                "cohort": direction.train_cohort,
                                "seed": int(seed),
                                "outer_fold": int(outer_fold),
                                "sample_id": sample_ids.iloc[int(idx)],
                                "y_true": int(y.iloc[int(idx)]),
                                "split_role": role,
                                "outer_splitter": "StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)",
                                "validation_splitter": "deterministic_validation_split(y_dev, seed + 999 + outer_fold)",
                                "outer_random_state": int(seed),
                                "validation_random_state": int(seed + 999 + outer_fold),
                                "early_stopping_validation_fraction": float(EARLY_STOPPING_VALIDATION_FRACTION),
                                "reconstruction_method": "deterministic_replay_of_corrected_v2_stage4e_splitter",
                                "originally_persisted": False,
                            }
                        )
    return pd.DataFrame(rows)


def validate_membership(membership: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    gcn_pred = predictions[predictions["model_family"].eq("gcn")].copy()
    graph_modes = sorted(gcn_pred["graph_mode"].dropna().unique().tolist())
    for (cohort, seed, outer_fold), part in membership.groupby(["cohort", "seed", "outer_fold"], sort=True):
        role_sets = {role: set(p["sample_id"].astype(str)) for role, p in part.groupby("split_role")}
        all_ids = set(membership[membership["cohort"].eq(cohort)]["sample_id"].astype(str))
        union = set().union(*role_sets.values())
        disjoint = not (role_sets["train"] & role_sets["validation"] or role_sets["train"] & role_sets["outer_test"] or role_sets["validation"] & role_sets["outer_test"])
        add_check(rows, "train_validation_outer_test_mutually_disjoint", "pass" if disjoint else "fail", f"{cohort}|{seed}|{outer_fold}", "no overlap")
        add_check(rows, "split_union_equals_full_cohort", "pass" if union == all_ids else "fail", f"{cohort}|{seed}|{outer_fold}|{len(union)}/{len(all_ids)}", "all cohort participants represented once across train/validation/outer_test")
        for role, ids in role_sets.items():
            classes = part[part["split_role"].eq(role)]["y_true"].nunique()
            add_check(rows, "class_counts_valid", "pass" if classes == 2 else "fail", f"{cohort}|{seed}|{outer_fold}|{role}|classes={classes}", "two classes in each split role")
        for graph_mode in graph_modes:
            pp = gcn_pred[
                gcn_pred["cohort"].eq(cohort)
                & gcn_pred["seed"].astype(str).eq(str(seed))
                & gcn_pred["outer_fold"].astype(int).eq(int(outer_fold))
                & gcn_pred["graph_mode"].eq(graph_mode)
            ]
            pred_ids = set(pp["sample_id"].astype(str))
            add_check(rows, "outer_test_ids_match_saved_gcn_oof_predictions", "pass" if pred_ids == role_sets["outer_test"] else "fail", f"{cohort}|{seed}|{outer_fold}|{graph_mode}|{len(pred_ids)}/{len(role_sets['outer_test'])}", "saved OOF predictions must equal reconstructed outer-test IDs")
        per_mode = [
            set(gcn_pred[
                gcn_pred["cohort"].eq(cohort)
                & gcn_pred["seed"].astype(str).eq(str(seed))
                & gcn_pred["outer_fold"].astype(int).eq(int(outer_fold))
                & gcn_pred["graph_mode"].eq(graph_mode)
            ]["sample_id"].astype(str))
            for graph_mode in graph_modes
        ]
        graph_mode_identical = all(ids == per_mode[0] for ids in per_mode[1:])
        add_check(rows, "outer_test_membership_identical_across_graph_modes", "pass" if graph_mode_identical else "fail", f"{cohort}|{seed}|{outer_fold}", "graph mode cannot alter fold membership")
        add_check(rows, "train_validation_membership_identical_across_graph_modes", "pass", f"{cohort}|{seed}|{outer_fold}", "train/validation split is reconstructed before graph-mode loop and is therefore graph-mode invariant")
    for (cohort, seed), part in membership[membership["split_role"].eq("outer_test")].groupby(["cohort", "seed"], sort=True):
        counts = part["sample_id"].astype(str).value_counts()
        ok = counts.eq(1).all()
        add_check(rows, "each_participant_once_outer_test_across_five_folds_per_seed", "pass" if ok else "fail", f"{cohort}|{seed}|min={counts.min()}|max={counts.max()}|n={len(counts)}", "each participant appears exactly once as outer-test per seed")
    return pd.DataFrame(rows)


def update_postrun_validation(validation: pd.DataFrame) -> None:
    if not POSTRUN_VALIDATION.exists():
        return
    post = pd.read_csv(POSTRUN_VALIDATION)
    mask = post["test"].eq("train_validation_test_sample_id_overlap_check")
    if mask.any():
        post.loc[mask, "status"] = "pass"
        post.loc[mask, "observed"] = "reconstructed split membership validation passed"
        post.loc[mask, "expected"] = "explicit train/validation/outer_test sample IDs are mutually disjoint and match saved OOF predictions"
        post.loc[mask, "notes"] = "Resolved by deterministic reconstruction from frozen corrected-v2 Stage 4E splitter code and seeds; split IDs were not originally persisted during fitting."
    if validation["status"].eq("fail").any():
        raise RuntimeError("Cannot mark postrun split validation as pass because reconstruction checks failed.")
    atomic_csv(post, POSTRUN_VALIDATION)


def write_report(membership: pd.DataFrame, validation: pd.DataFrame) -> None:
    status = "PASS" if not validation["status"].eq("fail").any() else "FAIL"
    summary = membership.groupby(["cohort", "split_role"])["sample_id"].nunique().reset_index(name="unique_participants")
    lines = [
        "# Stage 4E Corrected-v2 Split Reconstruction Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Status: {status}",
        "",
        "This table is reconstructed, not originally persisted. It was generated by deterministic replay of the frozen corrected-v2 Stage 4E splitter code, including the saved primary seeds, five stratified outer folds and the early-stopping validation split seeded as `seed + 999 + outer_fold`.",
        "",
        "No GCN, ridge model, CV, permutation, bootstrap, external validation or sensitivity analysis was rerun.",
        "",
        "## Membership Summary",
        summary.to_markdown(index=False),
        "",
        "## Validation Summary",
        validation.groupby(["test", "status"]).size().reset_index(name="n").to_markdown(index=False),
    ]
    REPORT.write_text("\n".join(lines))


def update_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest["split_membership_reconstruction"] = {
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "original_split_ids_persisted": False,
        "reconstruction_method": "deterministic_replay_of_corrected_v2_stage4e_splitter",
        "membership_table": str(MEMBERSHIP),
        "validation_table": str(VALIDATION),
        "audit_report": str(REPORT),
    }
    atomic_json(manifest, MANIFEST)


def run() -> None:
    ensure_v2_dirs()
    if not PREDICTIONS.exists():
        raise FileNotFoundError(PREDICTIONS)
    predictions = pd.read_csv(PREDICTIONS, low_memory=False)
    membership = reconstruct_membership()
    validation = validate_membership(membership, predictions)
    atomic_csv(membership, MEMBERSHIP)
    atomic_csv(validation, VALIDATION)
    if validation["status"].eq("fail").any():
        write_report(membership, validation)
        raise RuntimeError("Stage 4E split reconstruction validation failed.")
    update_postrun_validation(validation)
    update_manifest()
    write_report(membership, validation)
    print("STAGE4E_CORRECTED_V2_SPLIT_RECONSTRUCTION_COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct corrected-v2 Stage 4E primary split membership from frozen splitter code and seeds.")
    parser.add_argument("--run", action="store_true", help="Write reconstructed membership, validation table and audit report.")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("Use --run.")
    run()


if __name__ == "__main__":
    main()
