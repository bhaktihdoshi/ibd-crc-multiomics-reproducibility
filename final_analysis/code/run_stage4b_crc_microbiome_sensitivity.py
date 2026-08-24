"""Sensitivity branch: CRC microbiome rank / presence-absence transportability.

Sinha (16S) and Yachida (shotgun) are quantified on different platforms, so
Stage 4B's headline CRC microbiome AUCs (~0.563/0.582) could in principle be
partly an artefact of platform-specific abundance-scale differences rather
than a genuine, transferable microbial signal. This script re-runs the exact
frozen Stage 4B v2b design (same nested outer/inner CV, same source
cross-fitted-OOF Youden threshold, same decision-score ranking metrics, same
2,000-resample stratified bootstrap) for the CRC_vs_Control task only, on the
MICROBIOME representation only, swapping in two scale-robust preprocessing
variants:

  - within_sample_rank: within-sample rank of retained genera (invariant to
    any constant multiplicative rescaling of abundance).
  - presence_absence: binary detected/not-detected (invariant to abundance
    magnitude entirely).

The primary clr_zscore variant is also re-run here as a same-code-path
replica so all three numbers are directly comparable; it is NOT a
replacement for the frozen `stage4b_corrected_v2b_*` primary outputs, which
remain authoritative. All outputs are written under a distinct
`sensitivity_crc_microbiome_*` prefix and never overwrite primary tables.

This script performs real model fitting (nested nested-CV logistic
regression) and is intended to be run manually from the terminal, not as
part of an automated/heavy batch invocation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stage4_corrected_v2_common import (
    C_GRID,
    INNER_FOLDS,
    MANIFESTS,
    MICROBIOME_SENSITIVITY_VARIANTS,
    OUTER_FOLDS,
    PREDICTIONS,
    RANDOM_SEED,
    TABLES,
    TASK_SPECS,
    atomic_csv,
    atomic_json,
    ensure_v2_dirs,
    fold_count,
    read_csv,
    sha256,
)
from run_stage4b_corrected_v2b import (
    CorrectedDataBundle,
    aligned_train_external,
    add_context,
    binary_metrics_from_scores,
    build_ridge_pipeline,
    cross_fitted_decision_scores_and_probs,
    pipeline_scores,
    probability_threshold_from_decision_threshold,
    select_youden_threshold_from_scores,
    stratified_bootstrap_score_ci,
)
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


OUTPUT_PREFIX = "sensitivity_crc_microbiome"
CRC_SPECS = [spec for spec in TASK_SPECS if spec.disease == "CRC"]
REPRESENTATION = "MICROBIOME"


def selected_c_by_inner_oof(X: pd.DataFrame, y: pd.Series, variant: str, seed: int) -> tuple[float, pd.DataFrame]:
    inner_n = fold_count(y, INNER_FOLDS)
    rows = []
    for c in C_GRID:
        scores = np.full(len(y), np.nan)
        cv = StratifiedKFold(n_splits=inner_n, shuffle=True, random_state=seed)
        for train_idx, val_idx in cv.split(X, y):
            pipe = build_ridge_pipeline(REPRESENTATION, c, microbiome_variant=variant)
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            s, _ = pipeline_scores(pipe, X.iloc[val_idx])
            scores[val_idx] = s
        if not np.isfinite(scores).all():
            raise ValueError("Inner cross-fitted decision scores contain non-finite values.")
        rows.append({"C": c, "inner_cv_roc_auc": roc_auc_score(y, scores), "inner_cv_pr_auc": average_precision_score(y, scores), "inner_folds": inner_n})
    search = pd.DataFrame(rows).sort_values(["inner_cv_roc_auc", "inner_cv_pr_auc", "C"], ascending=[False, False, True])
    return float(search.iloc[0]["C"]), search


def nested_cv_variant(X: pd.DataFrame, y: pd.Series, ids: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer_n = fold_count(y, OUTER_FOLDS)
    outer = StratifiedKFold(n_splits=outer_n, shuffle=True, random_state=RANDOM_SEED)
    pred_rows, perf_rows, fold_rows, threshold_rows = [], [], [], []
    for fold, (dev_idx, val_idx) in enumerate(outer.split(X, y), start=1):
        X_dev, y_dev = X.iloc[dev_idx], y.iloc[dev_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        best_c, _ = selected_c_by_inner_oof(X_dev, y_dev, variant, RANDOM_SEED + fold)
        source_oof_scores = np.full(len(y_dev), np.nan)
        cv = StratifiedKFold(n_splits=fold_count(y_dev, INNER_FOLDS), shuffle=True, random_state=RANDOM_SEED + 1000 + fold)
        for tr, va in cv.split(X_dev, y_dev):
            p = build_ridge_pipeline(REPRESENTATION, best_c, microbiome_variant=variant)
            p.fit(X_dev.iloc[tr], y_dev.iloc[tr])
            s, _ = pipeline_scores(p, X_dev.iloc[va])
            source_oof_scores[va] = s
        threshold = select_youden_threshold_from_scores(y_dev, source_oof_scores)
        pipe = build_ridge_pipeline(REPRESENTATION, best_c, microbiome_variant=variant)
        pipe.fit(X_dev, y_dev)
        score, prob = pipeline_scores(pipe, X_val)
        perf_rows.append({"outer_fold": fold, "selected_C": best_c, **binary_metrics_from_scores(y_val, score, prob, threshold["decision_score_threshold"])})
        fold_rows.append(
            {
                "outer_fold": fold,
                "outer_train_n": len(dev_idx),
                "outer_validation_n": len(val_idx),
                "outer_train_class_counts": y_dev.value_counts().to_dict(),
                "outer_validation_class_counts": y_val.value_counts().to_dict(),
            }
        )
        threshold_rows.append({"outer_fold": fold, "selected_C": best_c, "threshold_source": "outer_dev_cross_fitted_decision_scores", **threshold})
        for row_i, s, p in zip(val_idx, score, prob):
            pred_rows.append({"sample_id": ids.iloc[row_i]["sample_id"], "y_true": int(y.iloc[row_i]), "decision_score": float(s), "probability": float(p), "outer_fold": fold})
    return pd.DataFrame(pred_rows), pd.DataFrame(perf_rows), pd.DataFrame(fold_rows), pd.DataFrame(threshold_rows)


def final_external_variant(X_train, y_train, train_ids, X_ext, y_ext, ext_ids, variant: str):
    best_c, _ = selected_c_by_inner_oof(X_train, y_train, variant, RANDOM_SEED + 9000)
    threshold_oof_scores, _ = cross_fitted_decision_scores_and_probs(X_train, y_train, REPRESENTATION, best_c, fold_count(y_train, INNER_FOLDS), RANDOM_SEED + 9100) if variant == "clr_zscore" else (None, None)
    if threshold_oof_scores is None:
        threshold_oof_scores = np.full(len(y_train), np.nan)
        cv = StratifiedKFold(n_splits=fold_count(y_train, INNER_FOLDS), shuffle=True, random_state=RANDOM_SEED + 9100)
        for tr, va in cv.split(X_train, y_train):
            p = build_ridge_pipeline(REPRESENTATION, best_c, microbiome_variant=variant)
            p.fit(X_train.iloc[tr], y_train.iloc[tr])
            s, _ = pipeline_scores(p, X_train.iloc[va])
            threshold_oof_scores[va] = s
        if not np.isfinite(threshold_oof_scores).all():
            raise ValueError("Source cross-fitted OOF decision scores contain non-finite values.")
    threshold = select_youden_threshold_from_scores(y_train, threshold_oof_scores)
    threshold["probability_threshold_logit_equivalent"] = probability_threshold_from_decision_threshold(threshold["decision_score_threshold"])
    final_pipe = build_ridge_pipeline(REPRESENTATION, best_c, microbiome_variant=variant)
    final_pipe.fit(X_train, y_train)
    external_score, external_prob = pipeline_scores(final_pipe, X_ext)
    metrics = binary_metrics_from_scores(y_ext, external_score, external_prob, threshold["decision_score_threshold"])
    metrics.update(stratified_bootstrap_score_ci(y_ext, external_score, "roc_auc"))
    metrics.update(stratified_bootstrap_score_ci(y_ext, external_score, "pr_auc"))
    metrics["selected_C"] = best_c
    metrics["external_data_used_for_threshold"] = False
    ext_pred = pd.DataFrame(
        {
            "external_sample_id": ext_ids["sample_id"].astype(str),
            "y_true": y_ext.astype(int),
            "decision_score": external_score,
            "probability": external_prob,
            "frozen_decision_score_threshold": threshold["decision_score_threshold"],
        }
    )
    source_oof_pred = pd.DataFrame(
        {
            "sample_id": train_ids["sample_id"].astype(str),
            "y_true": y_train.astype(int),
            "decision_score": threshold_oof_scores,
            "threshold_role": "source_cross_fitted_threshold_selection",
        }
    )
    threshold_row = {"selected_C": best_c, "threshold_source": "full_source_cross_fitted_oof", "external_data_used_for_threshold": False, **threshold}
    return ext_pred, source_oof_pred, threshold_row, metrics


def dry_run() -> None:
    ensure_v2_dirs()
    bundle = CorrectedDataBundle()
    rows = []
    for spec in CRC_SPECS:
        X_train, y_train, _, X_ext, y_ext, _ = aligned_train_external(bundle, spec, REPRESENTATION)
        for variant in MICROBIOME_SENSITIVITY_VARIANTS:
            pipe = build_ridge_pipeline(REPRESENTATION, C_GRID[0], microbiome_variant=variant)
            rows.append(
                {
                    "task": spec.task,
                    "training_cohort": spec.train_cohort,
                    "external_cohort": spec.external_cohort,
                    "microbiome_variant": variant,
                    "train_n": len(y_train),
                    "external_n": len(y_ext),
                    "train_p": X_train.shape[1],
                    "feature_order_identical": list(X_train.columns) == list(X_ext.columns),
                    "pipeline_steps": list(pipe.named_steps),
                    "status": "pass",
                }
            )
    atomic_csv(pd.DataFrame(rows), TABLES / f"{OUTPUT_PREFIX}_dry_run.csv")
    print("SENSITIVITY_CRC_MICROBIOME_DRY_RUN_PASSED")


REPLICATION_TOLERANCE = 1e-6


def primary_replication_check(external_perf_df: pd.DataFrame) -> pd.DataFrame:
    """Item-12 acceptance check: the clr_zscore arm run through THIS sensitivity
    runner must reproduce the frozen stage4b_corrected_v2b MICROBIOME external
    AUCs, because it is supposed to be the same pipeline executed a second time.
    If it does not, the rank/presence-absence results below cannot be trusted
    and this function raises before they are interpreted."""
    frozen = read_csv(TABLES / "stage4b_corrected_v2b_external_performance.csv")
    frozen = frozen[frozen["representation"].eq(REPRESENTATION)]
    rows = []
    for _, row in external_perf_df[external_perf_df["microbiome_variant"].eq("clr_zscore")].iterrows():
        frow = frozen[(frozen["task"] == row["task"]) & (frozen["training_cohort"] == row["training_cohort"]) & (frozen["external_cohort"] == row["external_cohort"])]
        frozen_roc = float(frow["roc_auc"].iloc[0]) if len(frow) else float("nan")
        frozen_pr = float(frow["pr_auc"].iloc[0]) if len(frow) else float("nan")
        rows.append(
            {
                "task": row["task"],
                "training_cohort": row["training_cohort"],
                "external_cohort": row["external_cohort"],
                "sensitivity_runner_roc_auc": row["roc_auc"],
                "frozen_v2b_roc_auc": frozen_roc,
                "roc_auc_match": abs(row["roc_auc"] - frozen_roc) <= REPLICATION_TOLERANCE,
                "sensitivity_runner_pr_auc": row["pr_auc"],
                "frozen_v2b_pr_auc": frozen_pr,
                "pr_auc_match": abs(row["pr_auc"] - frozen_pr) <= REPLICATION_TOLERANCE,
            }
        )
    result = pd.DataFrame(rows)
    if not (result["roc_auc_match"].all() and result["pr_auc_match"].all()):
        raise RuntimeError(
            "Primary-replication acceptance check FAILED: the clr_zscore arm of the "
            "sensitivity runner does not reproduce the frozen stage4b_corrected_v2b "
            "MICROBIOME external AUCs within tolerance. Stopping before the rank/"
            "presence-absence results are written or interpreted:\n" + result.to_string(index=False)
        )
    return result


def paired_sensitivity_comparisons(external_pred_df: pd.DataFrame) -> pd.DataFrame:
    """Item-13: rank-CLR and presence_absence-CLR paired comparisons per CRC
    transfer direction, using identical participant bootstrap resamples per
    direction (paired_stratified_auc_delta reuses one resample loop per
    call), then BH-FDR across this separate 4-comparison sensitivity family."""
    from audit_stage4b_corrected_v2b_postrun import bh_fdr, paired_stratified_auc_delta

    rows = []
    for (task, train_cohort, ext_cohort), group in external_pred_df.groupby(["task", "training_cohort", "external_cohort"]):
        by_variant = {v: group[group["microbiome_variant"].eq(v)].sort_values("external_sample_id").reset_index(drop=True) for v in MICROBIOME_SENSITIVITY_VARIANTS}
        clr = by_variant["clr_zscore"]
        if not (clr["external_sample_id"].tolist() == by_variant["within_sample_rank"]["external_sample_id"].tolist() == by_variant["presence_absence"]["external_sample_id"].tolist()):
            raise ValueError(f"{task} {train_cohort}->{ext_cohort}: participant alignment mismatch across microbiome_variant arms")
        y = clr["y_true"].to_numpy()
        for comparator_variant in ["within_sample_rank", "presence_absence"]:
            comparator = by_variant[comparator_variant]
            # paired_stratified_auc_delta(y, early_score, comparator_score) computes
            # delta = auc(early_score) - auc(comparator_score) on identical bootstrap
            # resamples. Passing (comparator_variant, clr) as (early, comparator) makes
            # its "early_integration_auc" actually hold the comparator_variant's AUC and
            # its "comparator_auc" actually hold the clr_zscore AUC — rename both
            # explicitly below so the output columns are not misleading.
            result = paired_stratified_auc_delta(y, comparator["decision_score"].to_numpy(), clr["decision_score"].to_numpy())
            rename = {"early_integration_auc": f"{comparator_variant}_auc", "comparator_auc": "clr_zscore_auc"}
            rows.append(
                {
                    "task": task,
                    "training_cohort": train_cohort,
                    "external_cohort": ext_cohort,
                    "comparison": f"{comparator_variant}_minus_clr_zscore",
                    **{rename.get(k, k): v for k, v in result.items()},
                }
            )
    out = pd.DataFrame(rows)
    out["bh_fdr_q_value"] = bh_fdr(out["bootstrap_two_sided_p"])
    return out


def run_full() -> None:
    ensure_v2_dirs()
    start = time.time()
    bundle = CorrectedDataBundle()
    internal_pred, internal_perf, internal_fold_diag, internal_thresholds = [], [], [], []
    external_pred, external_perf, source_oof_pred = [], [], []
    for spec in CRC_SPECS:
        X_train, y_train, train_ids, X_ext, y_ext, ext_ids = aligned_train_external(bundle, spec, REPRESENTATION)
        for variant in MICROBIOME_SENSITIVITY_VARIANTS:
            pred, perf, fold_diag, thresholds = nested_cv_variant(X_train, y_train, train_ids, variant)
            ext, oof, threshold_row, metrics = final_external_variant(X_train, y_train, train_ids, X_ext, y_ext, ext_ids, variant)
            for df in (pred, perf, fold_diag, thresholds, ext, oof):
                df["microbiome_variant"] = variant
            metrics["microbiome_variant"] = variant
            internal_pred.append(add_context(pred, spec, REPRESENTATION))
            internal_perf.append(add_context(perf, spec, REPRESENTATION))
            internal_fold_diag.append(add_context(fold_diag, spec, REPRESENTATION))
            internal_thresholds.append(add_context(thresholds, spec, REPRESENTATION))
            external_pred.append(add_context(ext, spec, REPRESENTATION))
            external_perf.append(add_context(pd.DataFrame([metrics]), spec, REPRESENTATION))
            source_oof_pred.append(add_context(oof, spec, REPRESENTATION))
    external_pred_df = pd.concat(external_pred, ignore_index=True)
    external_perf_df = pd.concat(external_perf, ignore_index=True)
    atomic_csv(pd.concat(internal_pred, ignore_index=True), PREDICTIONS / f"{OUTPUT_PREFIX}_internal_oof_predictions.csv")
    atomic_csv(external_pred_df, PREDICTIONS / f"{OUTPUT_PREFIX}_external_predictions.csv")
    atomic_csv(pd.concat(source_oof_pred, ignore_index=True), PREDICTIONS / f"{OUTPUT_PREFIX}_source_oof_threshold_predictions.csv")
    atomic_csv(pd.concat(internal_perf, ignore_index=True), TABLES / f"{OUTPUT_PREFIX}_internal_performance.csv")
    atomic_csv(pd.concat(internal_fold_diag, ignore_index=True), TABLES / f"{OUTPUT_PREFIX}_fold_diagnostics.csv")
    atomic_csv(pd.concat(internal_thresholds, ignore_index=True), TABLES / f"{OUTPUT_PREFIX}_selected_params_by_outer_fold.csv")
    atomic_csv(external_perf_df, TABLES / f"{OUTPUT_PREFIX}_external_performance.csv")

    replication = primary_replication_check(external_perf_df)  # raises if the clr_zscore arm drifted from frozen v2b
    atomic_csv(replication, TABLES / f"{OUTPUT_PREFIX}_primary_replication_check.csv")

    paired = paired_sensitivity_comparisons(external_pred_df)
    atomic_csv(paired, TABLES / f"{OUTPUT_PREFIX}_paired_comparisons_bh_fdr.csv")

    atomic_json(
        {
            "status": "complete",
            "runtime_seconds": round(time.time() - start, 2),
            "script_sha256": sha256(Path(__file__)),
            "variants": list(MICROBIOME_SENSITIVITY_VARIANTS),
            "primary_replication_check": "PASS (see stage4b_corrected_v2b_primary_replication_check equivalent above; run raised RuntimeError and aborted if it had failed)",
            "isolation_note": (
                "This is a post-freeze sensitivity branch. It writes only to sensitivity_crc_microbiome_* "
                "paths, never to stage4b_corrected_v2b_* primary outputs, and is NOT added to "
                "FINAL_STAGE4_AUTHORITY_MANIFEST.json or final_stage4_authoritative_files.csv. If these "
                "results are used in the thesis they must be cited explicitly as post-freeze sensitivity "
                "evidence, not as part of the frozen Stage 4B primary analysis."
            ),
        },
        MANIFESTS / f"{OUTPUT_PREFIX}_run_manifest.json",
    )
    print(external_perf_df[["task", "external_cohort", "microbiome_variant", "roc_auc", "pr_auc"]].to_string(index=False))
    print("\nPaired sensitivity comparisons (BH-FDR across this 4-comparison family):")
    print(paired[["task", "external_cohort", "comparison", "delta_roc_auc", "bootstrap_two_sided_p", "bh_fdr_q_value", "interpretation"]].to_string(index=False))
    print("\nSENSITIVITY_CRC_MICROBIOME_COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser(description="CRC microbiome rank / presence-absence transportability sensitivity (Stage 4B v2b design, sensitivity branch).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    else:
        run_full()


if __name__ == "__main__":
    main()
