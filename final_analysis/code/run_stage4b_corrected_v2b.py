from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from stage4_corrected_v2_common import (
    C_GRID,
    INNER_FOLDS,
    MANIFESTS,
    METABOLOME_INPUTS,
    METADATA_INPUTS,
    OUTER_FOLDS,
    PREDICTIONS,
    RANDOM_SEED,
    TABLES,
    TASK_SPECS,
    V2,
    TaskSpec,
    atomic_csv,
    atomic_json,
    build_ridge_pipeline,
    ensure_v2_dirs,
    fold_count,
    read_csv,
    sha256,
    stratified_bootstrap_metric_ci,
)


REPRESENTATIONS = ["METABOLOME", "MICROBIOME", "EARLY_INTEGRATION"]
OUTPUT_PREFIX = "stage4b_corrected_v2b"


def pipeline_scores(pipe, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    probability = pipe.predict_proba(X)[:, 1]
    if hasattr(pipe, "decision_function"):
        decision_score = pipe.decision_function(X)
    else:
        clipped = np.clip(probability, 1e-15, 1 - 1e-15)
        decision_score = np.log(clipped / (1 - clipped))
    return np.asarray(decision_score, dtype=float), np.asarray(probability, dtype=float)


def cross_fitted_decision_scores_and_probs(X: pd.DataFrame, y: pd.Series, representation: str, c_value: float, folds: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    scores = np.full(len(y), np.nan)
    probs = np.full(len(y), np.nan)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, val_idx in cv.split(X, y):
        pipe = build_ridge_pipeline(representation, c_value)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        s, p = pipeline_scores(pipe, X.iloc[val_idx])
        scores[val_idx] = s
        probs[val_idx] = p
    if np.isnan(scores).any() or np.isnan(probs).any():
        raise ValueError("Cross-fitted decision scores/probabilities contain NaN.")
    if not np.isfinite(scores).all() or not np.isfinite(probs).all():
        raise ValueError("Cross-fitted decision scores/probabilities contain non-finite values.")
    return scores, probs


def select_youden_threshold_from_scores(y: pd.Series, decision_score: np.ndarray) -> dict[str, float]:
    if y.nunique() != 2 or len(np.unique(decision_score)) < 2:
        return {"decision_score_threshold": 0.0, "youden_j": 0.0, "sensitivity": np.nan, "specificity": np.nan, "roc_auc": np.nan}
    fpr, tpr, thresholds = roc_curve(y, decision_score)
    j = tpr - fpr
    idx = int(np.nanargmax(j))
    return {
        "decision_score_threshold": float(thresholds[idx]),
        "youden_j": float(j[idx]),
        "sensitivity": float(tpr[idx]),
        "specificity": float(1 - fpr[idx]),
        "roc_auc": float(roc_auc_score(y, decision_score)),
    }


def probability_threshold_from_decision_threshold(decision_score_threshold: float) -> float:
    if not np.isfinite(decision_score_threshold):
        return float("nan")
    return float(1.0 / (1.0 + np.exp(-decision_score_threshold)))


def binary_metrics_from_scores(y: pd.Series, decision_score: np.ndarray, probability: np.ndarray, decision_score_threshold: float) -> dict[str, float]:
    pred = (decision_score >= decision_score_threshold).astype(int)
    y_arr = y.to_numpy(dtype=int)
    tn = int(((pred == 0) & (y_arr == 0)).sum())
    fp = int(((pred == 1) & (y_arr == 0)).sum())
    fn = int(((pred == 0) & (y_arr == 1)).sum())
    tp = int(((pred == 1) & (y_arr == 1)).sum())
    return {
        "roc_auc": float(roc_auc_score(y, decision_score)) if y.nunique() == 2 and len(np.unique(decision_score)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(y, decision_score)) if y.nunique() == 2 else np.nan,
        "probability_roc_auc": float(roc_auc_score(y, probability)) if y.nunique() == 2 and len(np.unique(probability)) > 1 else np.nan,
        "probability_pr_auc": float(average_precision_score(y, probability)) if y.nunique() == 2 else np.nan,
        "ranking_metric_score_column": "decision_score",
        "evaluation_positive_prevalence": float(y.mean()),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "decision_score_mean": float(np.mean(decision_score)),
        "decision_score_min": float(np.min(decision_score)),
        "decision_score_max": float(np.max(decision_score)),
        "decision_score_sd": float(np.std(decision_score)),
        "decision_score_unique_n": int(pd.Series(np.round(decision_score, 12)).nunique(dropna=True)),
        "probability_mean": float(np.mean(probability)),
        "probability_min": float(np.min(probability)),
        "probability_max": float(np.max(probability)),
        "probability_sd": float(np.std(probability)),
        "probability_unique_n": int(pd.Series(np.round(probability, 12)).nunique(dropna=True)),
        "probability_saturation_flag": bool(np.any(probability <= 1e-6) or np.any(probability >= 1 - 1e-6)),
        "probability_resolution_loss_flag": bool(pd.Series(np.round(probability, 12)).nunique(dropna=True) < pd.Series(np.round(decision_score, 12)).nunique(dropna=True)),
        "one_class_prediction_flag": bool(len(np.unique(pred)) <= 1),
    }


def stratified_bootstrap_score_ci(y: pd.Series, score: np.ndarray, metric: str, n_boot: int = 2000, seed: int = 20260813) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y_arr = y.to_numpy(dtype=int)
    score = np.asarray(score, dtype=float)
    pos = np.where(y_arr == 1)[0]
    neg = np.where(y_arr == 0)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, size=len(pos), replace=True), rng.choice(neg, size=len(neg), replace=True)])
        yy = y_arr[idx]
        ss = score[idx]
        if len(np.unique(yy)) < 2:
            continue
        if metric == "roc_auc" and len(np.unique(ss)) > 1:
            vals.append(roc_auc_score(yy, ss))
        elif metric == "pr_auc":
            vals.append(average_precision_score(yy, ss))
    if not vals:
        return {f"{metric}_ci_low": np.nan, f"{metric}_ci_high": np.nan, f"{metric}_bootstrap_n": 0}
    return {f"{metric}_ci_low": float(np.percentile(vals, 2.5)), f"{metric}_ci_high": float(np.percentile(vals, 97.5)), f"{metric}_bootstrap_n": len(vals)}


class CorrectedDataBundle:
    def __init__(self) -> None:
        self.ibd_samples = read_csv(METADATA_INPUTS["IBD"])
        self.crc_meta = read_csv(METADATA_INPUTS["CRC"])
        self.met = {
            "Franzosa": self._ibd_met(read_csv(METABOLOME_INPUTS["Franzosa"])),
            "iHMP": self._ibd_met(read_csv(METABOLOME_INPUTS["iHMP"])),
            "Sinha": self._crc_met(read_csv(METABOLOME_INPUTS["CRC"]), "Sinha"),
            "Yachida": self._crc_met(read_csv(METABOLOME_INPUTS["CRC"]), "Yachida"),
        }
        self.micro = {
            cohort: read_csv(V2 / "data" / "processed" / f"{cohort.lower()}_authoritative_genus_proportions.csv")
            for cohort in ["Franzosa", "iHMP", "Sinha", "Yachida"]
        }
        for cohort in self.met:
            self.met[cohort]["sample_id"] = self.met[cohort]["sample_id"].astype(str)
            self.micro[cohort]["sample_id"] = self.micro[cohort]["sample_id"].astype(str)
        self.matched_ids = {
            cohort: set(self.met[cohort]["sample_id"]) & set(self.micro[cohort]["sample_id"])
            for cohort in self.met
        }
        self.shared_micro = {
            "IBD": read_csv(TABLES / "stage4a_corrected_v2_ibd_shared_genera.csv")["feature_name"].astype(str).tolist(),
            "CRC": read_csv(TABLES / "stage4a_corrected_v2_crc_shared_genera.csv")["feature_name"].astype(str).tolist(),
        }

    @staticmethod
    def _ibd_met(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["sample_id"] = df["sample_id"].astype(str)
        return df.rename(columns={c: f"met__{c}" for c in df.columns if c != "sample_id"})

    @staticmethod
    def _crc_met(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
        sub = df[df["cohort"].eq(cohort)].copy()
        sub["sample_id"] = sub["sample_id"].astype(str)
        cols = [c for c in sub.columns if str(c).startswith("met__")]
        return sub[["sample_id"] + cols]

    def labels(self, spec: TaskSpec, cohort: str) -> pd.DataFrame:
        if spec.disease == "IBD":
            sub = self.ibd_samples[self.ibd_samples["cohort"].eq(cohort)].copy()
            sub = sub[sub["sample_id"].astype(str).isin(self.matched_ids[cohort])]
            labels = sub["harmonised_label"].replace({"IBD_control": "Control"}).astype(str)
            return pd.DataFrame({"sample_id": sub["sample_id"].astype(str), "raw_label": labels})
        sub = self.crc_meta[self.crc_meta["cohort"].eq(cohort)].copy()
        sub = sub[sub["sample_id"].astype(str).isin(self.matched_ids[cohort])]
        labels = sub["crc_binary"].map({0: "Control/Healthy", 1: "CRC"}).astype(str)
        return pd.DataFrame({"sample_id": sub["sample_id"].astype(str), "raw_label": labels})


def build_xy(bundle: CorrectedDataBundle, spec: TaskSpec, cohort: str, representation: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    labels = bundle.labels(spec, cohort)
    if spec.task == "IBD_vs_Control":
        labels["label"] = np.where(labels["raw_label"].eq("Control"), "Control", "IBD")
        keep = {"Control", "IBD"}
    else:
        labels["label"] = labels["raw_label"]
        keep = {spec.negative_label, spec.positive_label}
    labels = labels[labels["label"].isin(keep)].copy()
    labels["y"] = labels["label"].eq(spec.positive_label).astype(int)
    met = bundle.met[cohort].copy()
    mic = bundle.micro[cohort][["sample_id"] + bundle.shared_micro[spec.disease]].copy()
    if representation == "METABOLOME":
        merged = labels.merge(met, on="sample_id", how="inner")
    elif representation == "MICROBIOME":
        merged = labels.merge(mic, on="sample_id", how="inner")
    elif representation == "EARLY_INTEGRATION":
        merged = labels.merge(met, on="sample_id", how="inner").merge(mic, on="sample_id", how="inner")
    else:
        raise ValueError(representation)
    if merged["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {spec} {cohort} {representation}")
    y = merged["y"].astype(int).reset_index(drop=True)
    feature_cols = [c for c in merged.columns if c not in {"sample_id", "raw_label", "label", "y"}]
    if not feature_cols:
        raise ValueError(f"No features in {spec} {cohort} {representation}")
    X = merged[feature_cols].apply(pd.to_numeric, errors="raise").reset_index(drop=True)
    ids = merged[["sample_id", "label", "y"]].reset_index(drop=True)
    if y.nunique() != 2:
        raise ValueError(f"One-class outcome in {spec} {cohort} {representation}: {y.value_counts().to_dict()}")
    return X, y, ids


def aligned_train_external(bundle: CorrectedDataBundle, spec: TaskSpec, representation: str):
    X_train, y_train, train_ids = build_xy(bundle, spec, spec.train_cohort, representation)
    X_ext, y_ext, ext_ids = build_xy(bundle, spec, spec.external_cohort, representation)
    if list(X_train.columns) != list(X_ext.columns):
        raise ValueError(f"Train/external feature order mismatch for {spec} {representation}")
    return X_train, y_train, train_ids, X_ext, y_ext, ext_ids


def selected_c_by_inner_oof(X: pd.DataFrame, y: pd.Series, representation: str, seed: int) -> tuple[float, pd.DataFrame]:
    inner_n = fold_count(y, INNER_FOLDS)
    rows = []
    for c in C_GRID:
        scores, probs = cross_fitted_decision_scores_and_probs(X, y, representation, c, inner_n, seed)
        rows.append(
            {
                "C": c,
                "inner_cv_roc_auc": roc_auc_score(y, scores),
                "inner_cv_pr_auc": average_precision_score(y, scores),
                "inner_cv_probability_roc_auc": roc_auc_score(y, probs),
                "inner_cv_probability_pr_auc": average_precision_score(y, probs),
                "ranking_metric_score_column": "decision_score",
                "inner_folds": inner_n,
            }
        )
    scores = pd.DataFrame(rows).sort_values(["inner_cv_roc_auc", "inner_cv_pr_auc", "C"], ascending=[False, False, True])
    return float(scores.iloc[0]["C"]), scores


def nested_cv(X: pd.DataFrame, y: pd.Series, ids: pd.DataFrame, representation: str, spec: TaskSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer_n = fold_count(y, OUTER_FOLDS)
    outer = StratifiedKFold(n_splits=outer_n, shuffle=True, random_state=RANDOM_SEED)
    pred_rows = []
    perf_rows = []
    param_rows = []
    fold_rows = []
    for fold, (dev_idx, val_idx) in enumerate(outer.split(X, y), start=1):
        X_dev, y_dev = X.iloc[dev_idx], y.iloc[dev_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        best_c, search = selected_c_by_inner_oof(X_dev, y_dev, representation, RANDOM_SEED + fold)
        source_oof_scores_for_threshold, source_oof_probs_for_threshold = cross_fitted_decision_scores_and_probs(X_dev, y_dev, representation, best_c, fold_count(y_dev, INNER_FOLDS), RANDOM_SEED + 1000 + fold)
        threshold = select_youden_threshold_from_scores(y_dev, source_oof_scores_for_threshold)
        threshold["probability_threshold_logit_equivalent"] = probability_threshold_from_decision_threshold(threshold["decision_score_threshold"])
        threshold["threshold_scale"] = "decision_score"
        pipe = build_ridge_pipeline(representation, best_c)
        pipe.fit(X_dev, y_dev)
        score, prob = pipeline_scores(pipe, X_val)
        perf_rows.append({"outer_fold": fold, **binary_metrics_from_scores(y_val, score, prob, threshold["decision_score_threshold"])})
        param_rows.append({"outer_fold": fold, "selected_C": best_c, "search_rows": len(search), "threshold_source": "outer_dev_cross_fitted_decision_scores"})
        fold_rows.append(
            {
                "outer_fold": fold,
                "outer_train_n": len(dev_idx),
                "outer_validation_n": len(val_idx),
                "outer_train_class_counts": y_dev.value_counts().to_dict(),
                "outer_validation_class_counts": y_val.value_counts().to_dict(),
            }
        )
        for row_i, s, p in zip(val_idx, score, prob):
            pred_rows.append({"sample_id": ids.iloc[row_i]["sample_id"], "y_true": int(y.iloc[row_i]), "decision_score": float(s), "probability": float(p), "outer_fold": fold})
    return pd.DataFrame(pred_rows), pd.DataFrame(perf_rows), pd.DataFrame(param_rows), pd.DataFrame(fold_rows)


def final_external(X_train: pd.DataFrame, y_train: pd.Series, train_ids: pd.DataFrame, X_ext: pd.DataFrame, y_ext: pd.Series, ext_ids: pd.DataFrame, representation: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    best_c, search = selected_c_by_inner_oof(X_train, y_train, representation, RANDOM_SEED + 9000)
    threshold_oof_scores, threshold_oof_probs = cross_fitted_decision_scores_and_probs(X_train, y_train, representation, best_c, fold_count(y_train, INNER_FOLDS), RANDOM_SEED + 9100)
    threshold = select_youden_threshold_from_scores(y_train, threshold_oof_scores)
    threshold["probability_threshold_logit_equivalent"] = probability_threshold_from_decision_threshold(threshold["decision_score_threshold"])
    threshold["threshold_scale"] = "decision_score"
    final_pipe = build_ridge_pipeline(representation, best_c)
    final_pipe.fit(X_train, y_train)
    external_score, external_prob = pipeline_scores(final_pipe, X_ext)
    external_pred = pd.DataFrame(
        {
            "external_sample_id": ext_ids["sample_id"].astype(str),
            "y_true": y_ext.astype(int),
            "decision_score": external_score,
            "probability": external_prob,
            "frozen_decision_score_threshold": threshold["decision_score_threshold"],
            "frozen_probability_threshold_logit_equivalent": threshold["probability_threshold_logit_equivalent"],
            "predicted_label": (external_score >= threshold["decision_score_threshold"]).astype(int),
        }
    )
    oof_pred = pd.DataFrame(
        {
            "sample_id": train_ids["sample_id"].astype(str),
            "y_true": y_train.astype(int),
            "decision_score": threshold_oof_scores,
            "probability": threshold_oof_probs,
            "threshold_role": "source_cross_fitted_threshold_selection",
            "threshold_scale": "decision_score",
        }
    )
    metrics = binary_metrics_from_scores(y_ext, external_score, external_prob, threshold["decision_score_threshold"])
    metrics.update(stratified_bootstrap_score_ci(y_ext, external_score, "roc_auc"))
    metrics.update(stratified_bootstrap_score_ci(y_ext, external_score, "pr_auc"))
    return external_pred, oof_pred, {"selected_C": best_c, **threshold, "threshold_source": "full_source_cross_fitted_oof", "external_data_used_for_threshold": False}, metrics


def add_context(df: pd.DataFrame, spec: TaskSpec, representation: str) -> pd.DataFrame:
    out = df.copy()
    for col, value in {
        "representation": representation,
        "external_cohort": spec.external_cohort,
        "training_cohort": spec.train_cohort,
        "task": spec.task,
        "disease": spec.disease,
    }.items():
        out.insert(0, col, value)
    return out


def dry_run() -> None:
    ensure_v2_dirs()
    bundle = CorrectedDataBundle()
    rows = []
    for spec in TASK_SPECS:
        for rep in REPRESENTATIONS:
            X_train, y_train, _, X_ext, y_ext, _ = aligned_train_external(bundle, spec, rep)
            pipe = build_ridge_pipeline(rep, C_GRID[0])
            rows.append(
                {
                    "disease": spec.disease,
                    "task": spec.task,
                    "training_cohort": spec.train_cohort,
                    "external_cohort": spec.external_cohort,
                    "representation": rep,
                    "train_n": len(y_train),
                    "external_n": len(y_ext),
                    "train_p": X_train.shape[1],
                    "external_p": X_ext.shape[1],
                    "train_class_counts": y_train.value_counts().to_dict(),
                    "external_class_counts": y_ext.value_counts().to_dict(),
                    "feature_order_identical": list(X_train.columns) == list(X_ext.columns),
                    "pipeline_steps": list(pipe.named_steps),
                    "status": "pass",
                }
            )
    atomic_csv(pd.DataFrame(rows), TABLES / f"{OUTPUT_PREFIX}_dry_run.csv")
    print("STAGE4B_CORRECTED_V2B_DRY_RUN_PASSED")


def run_full() -> None:
    ensure_v2_dirs()
    start = time.time()
    bundle = CorrectedDataBundle()
    internal_pred = []
    threshold_oof_pred = []
    external_pred = []
    internal_perf = []
    external_perf = []
    params = []
    thresholds = []
    folds = []
    for spec in TASK_SPECS:
        for rep in REPRESENTATIONS:
            X_train, y_train, train_ids, X_ext, y_ext, ext_ids = aligned_train_external(bundle, spec, rep)
            pred, perf, fold_params, fold_diag = nested_cv(X_train, y_train, train_ids, rep, spec)
            ext, source_oof, threshold, metrics = final_external(X_train, y_train, train_ids, X_ext, y_ext, ext_ids, rep)
            internal_pred.append(add_context(pred, spec, rep))
            internal_perf.append(add_context(perf, spec, rep))
            params.append(add_context(fold_params, spec, rep))
            folds.append(add_context(fold_diag, spec, rep))
            thresholds.append(add_context(pd.DataFrame([threshold]), spec, rep))
            threshold_oof_pred.append(add_context(source_oof, spec, rep))
            external_pred.append(add_context(ext, spec, rep))
            external_perf.append(add_context(pd.DataFrame([metrics]), spec, rep))
    atomic_csv(pd.concat(internal_pred, ignore_index=True), PREDICTIONS / f"{OUTPUT_PREFIX}_internal_oof_predictions.csv")
    atomic_csv(pd.concat(threshold_oof_pred, ignore_index=True), PREDICTIONS / f"{OUTPUT_PREFIX}_source_oof_threshold_predictions.csv")
    atomic_csv(pd.concat(external_pred, ignore_index=True), PREDICTIONS / f"{OUTPUT_PREFIX}_external_predictions.csv")
    atomic_csv(pd.concat(internal_perf, ignore_index=True), TABLES / f"{OUTPUT_PREFIX}_internal_performance.csv")
    atomic_csv(pd.concat(external_perf, ignore_index=True), TABLES / f"{OUTPUT_PREFIX}_external_performance.csv")
    atomic_csv(pd.concat(params, ignore_index=True), TABLES / f"{OUTPUT_PREFIX}_selected_params_by_outer_fold.csv")
    atomic_csv(pd.concat(thresholds, ignore_index=True), TABLES / f"{OUTPUT_PREFIX}_thresholds.csv")
    atomic_csv(pd.concat(folds, ignore_index=True), TABLES / f"{OUTPUT_PREFIX}_fold_diagnostics.csv")
    atomic_json({"status": "complete", "runtime_seconds": round(time.time() - start, 2), "script_sha256": sha256(Path(__file__)), "ranking_metric_score_column": "decision_score"}, MANIFESTS / f"{OUTPUT_PREFIX}_run_manifest.json")
    print("STAGE4B_CORRECTED_V2B_COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrected-v2b Stage 4B leakage-safe ridge baselines with decision-score ranking metrics.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    else:
        run_full()


if __name__ == "__main__":
    main()
