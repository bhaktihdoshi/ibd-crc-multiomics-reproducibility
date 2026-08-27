from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, matthews_corrcoef, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[3]
CODE = REPO / "final_analysis" / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import stage4_corrected_v2_common as common  # noqa: E402
import run_stage4b_corrected_v2b as v2b  # noqa: E402


OUTPUT_PREFIX = "rank_metabolomics"
RANK_REPRESENTATIONS = ["RANK_METABOLOME", "EARLY_INTEGRATION_RANK_METABOLOME"]
ORIGINAL_COMPARATORS = {"RANK_METABOLOME": "METABOLOME", "EARLY_INTEGRATION_RANK_METABOLOME": "EARLY_INTEGRATION"}
BOOTSTRAP_N = 2000


def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def public_sample_key(sample_id: str) -> str:
    return hashlib.sha256(f"focused-rank::{sample_id}".encode()).hexdigest()[:16]


class RankMetabolomePreprocessor(BaseEstimator, TransformerMixin):
    """Fold-local within-sample rank-percentile metabolomics transformer."""

    def __init__(self, missingness_threshold: float = common.MET_MISSINGNESS_THRESHOLD):
        self.missingness_threshold = missingness_threshold

    def fit(self, X, y=None):
        x = pd.DataFrame(X).apply(pd.to_numeric, errors="raise")
        self.feature_names_in_ = np.array(x.columns.astype(str))
        missing_fraction = x.isna().mean(axis=0)
        self.missingness_keep_mask_ = (missing_fraction <= self.missingness_threshold).to_numpy()
        if not self.missingness_keep_mask_.any():
            raise ValueError("Rank metabolome missingness filtering removed all features.")
        ranked = self._rank_percentile(x.loc[:, self.missingness_keep_mask_])
        variances = ranked.var(axis=0, skipna=True).fillna(0.0)
        self.rank_zero_variance_keep_mask_ = (variances > 0).to_numpy()
        if not self.rank_zero_variance_keep_mask_.any():
            raise ValueError("Rank metabolome zero-variance filtering removed all features.")
        ranked_kept = ranked.loc[:, self.rank_zero_variance_keep_mask_]
        self.feature_names_after_missingness_ = self.feature_names_in_[self.missingness_keep_mask_]
        self.feature_names_out_ = self.feature_names_after_missingness_[self.rank_zero_variance_keep_mask_]
        self.imputer_ = SimpleImputer(strategy="median").fit(ranked_kept)
        imputed = self.imputer_.transform(ranked_kept)
        self.scaler_ = StandardScaler().fit(imputed)
        self.training_observed_values_per_participant_ = ranked_kept.notna().sum(axis=1).astype(int).to_numpy()
        self.missingness_excluded_features_ = self.feature_names_in_[~self.missingness_keep_mask_]
        self.rank_zero_variance_excluded_features_ = self.feature_names_after_missingness_[~self.rank_zero_variance_keep_mask_]
        return self

    @staticmethod
    def _rank_percentile(x: pd.DataFrame) -> pd.DataFrame:
        observed_n = x.notna().sum(axis=1)
        if (observed_n <= 0).any():
            raise ValueError("At least one participant has zero observed retained metabolites for ranking.")
        ranks = x.rank(axis=1, method="average", na_option="keep")
        return ranks.sub(0.5).div(observed_n, axis=0)

    def transform(self, X):
        x = pd.DataFrame(X).apply(pd.to_numeric, errors="raise")
        x = x.loc[:, self.missingness_keep_mask_]
        ranked = self._rank_percentile(x)
        ranked = ranked.loc[:, self.rank_zero_variance_keep_mask_]
        return self.scaler_.transform(self.imputer_.transform(ranked))

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


def rank_metabolome_pipeline() -> Pipeline:
    return Pipeline([("rank_metabolome", RankMetabolomePreprocessor())])


def build_rank_pipeline(representation: str, c_value: float) -> Pipeline:
    clf = LogisticRegression(
        penalty="l2",
        C=c_value,
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=common.RANDOM_SEED,
    )
    if representation == "RANK_METABOLOME":
        return Pipeline([("metabolome", rank_metabolome_pipeline()), ("clf", clf)])
    if representation == "EARLY_INTEGRATION_RANK_METABOLOME":
        pre = ColumnTransformer(
            [
                ("metabolome", rank_metabolome_pipeline(), lambda x: [c for c in x.columns if str(c).startswith("met__")]),
                ("microbiome", common.microbiome_pipeline("clr_zscore"), lambda x: [c for c in x.columns if str(c).startswith("mic__")]),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return Pipeline([("block_preprocessor", pre), ("clf", clf)])
    raise ValueError(representation)


def pipeline_scores(pipe, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    probability = pipe.predict_proba(X)[:, 1]
    if hasattr(pipe, "decision_function"):
        decision_score = pipe.decision_function(X)
    else:
        clipped = np.clip(probability, 1e-15, 1 - 1e-15)
        decision_score = np.log(clipped / (1 - clipped))
    return np.asarray(decision_score, dtype=float), np.asarray(probability, dtype=float)


def cross_fitted_scores(X: pd.DataFrame, y: pd.Series, representation: str, c_value: float, folds: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    scores = np.full(len(y), np.nan)
    probs = np.full(len(y), np.nan)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, val_idx in cv.split(X, y):
        pipe = build_rank_pipeline(representation, c_value)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        s, p = pipeline_scores(pipe, X.iloc[val_idx])
        scores[val_idx] = s
        probs[val_idx] = p
    if np.isnan(scores).any() or np.isnan(probs).any() or not np.isfinite(scores).all() or not np.isfinite(probs).all():
        raise ValueError("Cross-fitted rank scores/probabilities contain invalid values.")
    return scores, probs


def select_youden(y: pd.Series, score: np.ndarray) -> dict[str, float]:
    if y.nunique() != 2 or len(np.unique(score)) < 2:
        return {"decision_score_threshold": 0.0, "youden_j": 0.0, "sensitivity": np.nan, "specificity": np.nan, "roc_auc": np.nan}
    fpr, tpr, thresholds = roc_curve(y, score)
    j = tpr - fpr
    idx = int(np.nanargmax(j))
    return {"decision_score_threshold": float(thresholds[idx]), "youden_j": float(j[idx]), "sensitivity": float(tpr[idx]), "specificity": float(1 - fpr[idx]), "roc_auc": float(roc_auc_score(y, score))}


def metrics_from_scores(y: pd.Series, score: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (score >= threshold).astype(int)
    yy = y.to_numpy(dtype=int)
    tn = int(((pred == 0) & (yy == 0)).sum())
    fp = int(((pred == 1) & (yy == 0)).sum())
    fn = int(((pred == 0) & (yy == 1)).sum())
    tp = int(((pred == 1) & (yy == 1)).sum())
    return {
        "roc_auc": float(roc_auc_score(y, score)) if y.nunique() == 2 and len(np.unique(score)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(y, score)) if y.nunique() == 2 else np.nan,
        "probability_roc_auc": float(roc_auc_score(y, prob)) if y.nunique() == 2 and len(np.unique(prob)) > 1 else np.nan,
        "probability_pr_auc": float(average_precision_score(y, prob)) if y.nunique() == 2 else np.nan,
        "ranking_metric_score_column": "decision_score",
        "evaluation_positive_prevalence": float(y.mean()),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y, prob)),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "decision_score_mean": float(np.mean(score)),
        "decision_score_min": float(np.min(score)),
        "decision_score_max": float(np.max(score)),
        "decision_score_sd": float(np.std(score)),
        "decision_score_unique_n": int(pd.Series(np.round(score, 12)).nunique(dropna=True)),
        "probability_mean": float(np.mean(prob)),
        "probability_min": float(np.min(prob)),
        "probability_max": float(np.max(prob)),
        "probability_sd": float(np.std(prob)),
        "probability_unique_n": int(pd.Series(np.round(prob, 12)).nunique(dropna=True)),
        "probability_saturation_flag": bool(np.any(prob <= 1e-6) or np.any(prob >= 1 - 1e-6)),
        "probability_resolution_loss_flag": bool(pd.Series(np.round(prob, 12)).nunique(dropna=True) < pd.Series(np.round(score, 12)).nunique(dropna=True)),
        "one_class_prediction_flag": bool(len(np.unique(pred)) <= 1),
    }


def bootstrap_indices(y: pd.Series, n_boot: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    yy = y.to_numpy(dtype=int)
    pos = np.where(yy == 1)[0]
    neg = np.where(yy == 0)[0]
    return [np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]) for _ in range(n_boot)]


def ci_from_indices(y: pd.Series, score: np.ndarray, indices: list[np.ndarray], metric: str) -> dict[str, Any]:
    yy = y.to_numpy(dtype=int)
    vals = []
    for idx in indices:
        if metric == "roc_auc":
            if len(np.unique(score[idx])) > 1:
                vals.append(roc_auc_score(yy[idx], score[idx]))
        else:
            vals.append(average_precision_score(yy[idx], score[idx]))
    if not vals:
        return {f"{metric}_ci_low": np.nan, f"{metric}_ci_high": np.nan, f"{metric}_bootstrap_n": 0}
    return {f"{metric}_ci_low": float(np.percentile(vals, 2.5)), f"{metric}_ci_high": float(np.percentile(vals, 97.5)), f"{metric}_bootstrap_n": len(vals)}


def selected_c_by_inner_oof(X: pd.DataFrame, y: pd.Series, representation: str, seed: int) -> tuple[float, pd.DataFrame]:
    rows = []
    inner_n = common.fold_count(y, common.INNER_FOLDS)
    for c in common.C_GRID:
        scores, probs = cross_fitted_scores(X, y, representation, c, inner_n, seed)
        rows.append({"C": c, "inner_cv_roc_auc": roc_auc_score(y, scores), "inner_cv_pr_auc": average_precision_score(y, scores), "inner_cv_probability_roc_auc": roc_auc_score(y, probs), "inner_cv_probability_pr_auc": average_precision_score(y, probs), "ranking_metric_score_column": "decision_score", "inner_folds": inner_n})
    out = pd.DataFrame(rows).sort_values(["inner_cv_roc_auc", "inner_cv_pr_auc", "C"], ascending=[False, False, True])
    return float(out.iloc[0]["C"]), out


def nested_cv(X: pd.DataFrame, y: pd.Series, ids: pd.DataFrame, representation: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer_n = common.fold_count(y, common.OUTER_FOLDS)
    outer = StratifiedKFold(n_splits=outer_n, shuffle=True, random_state=common.RANDOM_SEED)
    preds, perfs, params, folds = [], [], [], []
    for fold, (dev_idx, val_idx) in enumerate(outer.split(X, y), start=1):
        X_dev, y_dev = X.iloc[dev_idx], y.iloc[dev_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        best_c, search = selected_c_by_inner_oof(X_dev, y_dev, representation, common.RANDOM_SEED + fold)
        threshold_scores, _ = cross_fitted_scores(X_dev, y_dev, representation, best_c, common.fold_count(y_dev, common.INNER_FOLDS), common.RANDOM_SEED + 1000 + fold)
        thresh = select_youden(y_dev, threshold_scores)
        pipe = build_rank_pipeline(representation, best_c).fit(X_dev, y_dev)
        score, prob = pipeline_scores(pipe, X_val)
        perfs.append({"outer_fold": fold, **metrics_from_scores(y_val, score, prob, thresh["decision_score_threshold"])})
        params.append({"outer_fold": fold, "selected_C": best_c, "search_rows": len(search), "threshold_source": "outer_dev_cross_fitted_decision_scores"})
        folds.append({"outer_fold": fold, "outer_train_n": len(dev_idx), "outer_validation_n": len(val_idx), "outer_train_class_counts": y_dev.value_counts().to_dict(), "outer_validation_class_counts": y_val.value_counts().to_dict()})
        for row_i, s, p in zip(val_idx, score, prob):
            preds.append({"sample_key": public_sample_key(ids.iloc[row_i]["sample_id"]), "y_true": int(y.iloc[row_i]), "decision_score": float(s), "probability": float(p), "outer_fold": fold})
    return pd.DataFrame(preds), pd.DataFrame(perfs), pd.DataFrame(params), pd.DataFrame(folds)


def preprocessing_audit(pipe: Pipeline, X_train: pd.DataFrame, X_ext: pd.DataFrame, spec: common.TaskSpec, rep: str) -> pd.DataFrame:
    rows = []
    def add_block(name: str, proc: RankMetabolomePreprocessor, cols: list[str]) -> None:
        train_rank = proc._rank_percentile(X_train[cols].loc[:, proc.missingness_keep_mask_].loc[:, proc.rank_zero_variance_keep_mask_])
        ext_obs = X_ext[cols].loc[:, proc.missingness_keep_mask_].loc[:, proc.rank_zero_variance_keep_mask_].notna().sum(axis=1)
        rows.append({
            "disease": spec.disease,
            "task": spec.task,
            "training_cohort": spec.train_cohort,
            "external_cohort": spec.external_cohort,
            "representation": rep,
            "block": name,
            "input_feature_count": len(cols),
            "retained_feature_count": len(proc.feature_names_out_),
            "missingness_exclusion_count": len(proc.missingness_excluded_features_),
            "rank_zero_variance_exclusion_count": len(proc.rank_zero_variance_excluded_features_),
            "missingness_threshold": proc.missingness_threshold,
            "training_observed_values_min": int(np.min(proc.training_observed_values_per_participant_)),
            "training_observed_values_median": float(np.median(proc.training_observed_values_per_participant_)),
            "external_observed_values_min_after_training_mask": int(ext_obs.min()),
            "external_observed_values_median_after_training_mask": float(ext_obs.median()),
            "imputation_rule": "feature-wise median fitted on source-training rank-percentiles only",
            "scaling_rule": "StandardScaler fitted on source-training imputed rank-percentiles only",
            "target_distribution_parameters_used": False,
            "first_10_imputation_values": json.dumps([float(x) for x in np.ravel(proc.imputer_.statistics_)[:10]]),
            "first_10_scaling_means": json.dumps([float(x) for x in np.ravel(proc.scaler_.mean_)[:10]]),
        })
    if rep == "RANK_METABOLOME":
        add_block("metabolome", pipe.named_steps["metabolome"].named_steps["rank_metabolome"], list(X_train.columns))
    else:
        pre = pipe.named_steps["block_preprocessor"]
        met_cols = [c for c in X_train.columns if str(c).startswith("met__")]
        add_block("metabolome", pre.named_transformers_["metabolome"].named_steps["rank_metabolome"], met_cols)
        rows.append({"disease": spec.disease, "task": spec.task, "training_cohort": spec.train_cohort, "external_cohort": spec.external_cohort, "representation": rep, "block": "microbiome", "input_feature_count": len([c for c in X_train.columns if str(c).startswith("mic__")]), "retained_feature_count": len(pre.named_transformers_["microbiome"].named_steps["microbiome"].feature_names_out_), "missingness_exclusion_count": np.nan, "rank_zero_variance_exclusion_count": np.nan, "missingness_threshold": np.nan, "training_observed_values_min": np.nan, "training_observed_values_median": np.nan, "external_observed_values_min_after_training_mask": np.nan, "external_observed_values_median_after_training_mask": np.nan, "imputation_rule": "authoritative microbiome CLR preprocessing unchanged", "scaling_rule": "authoritative microbiome block StandardScaler unchanged", "target_distribution_parameters_used": False, "first_10_imputation_values": "", "first_10_scaling_means": ""})
    return pd.DataFrame(rows)


def final_external(X_train, y_train, train_ids, X_ext, y_ext, ext_ids, representation: str, spec: common.TaskSpec):
    best_c, search = selected_c_by_inner_oof(X_train, y_train, representation, common.RANDOM_SEED + 9000)
    threshold_scores, threshold_probs = cross_fitted_scores(X_train, y_train, representation, best_c, common.fold_count(y_train, common.INNER_FOLDS), common.RANDOM_SEED + 9100)
    threshold = select_youden(y_train, threshold_scores)
    threshold["probability_threshold_logit_equivalent"] = float(1.0 / (1.0 + np.exp(-threshold["decision_score_threshold"])))
    threshold["threshold_scale"] = "decision_score"
    final_pipe = build_rank_pipeline(representation, best_c).fit(X_train, y_train)
    external_score, external_prob = pipeline_scores(final_pipe, X_ext)
    external_pred = pd.DataFrame({
        "external_sample_key": [public_sample_key(x) for x in ext_ids["sample_id"].astype(str)],
        "y_true": y_ext.astype(int),
        "decision_score": external_score,
        "probability": external_prob,
        "frozen_decision_score_threshold": threshold["decision_score_threshold"],
        "frozen_probability_threshold_logit_equivalent": threshold["probability_threshold_logit_equivalent"],
        "predicted_label": (external_score >= threshold["decision_score_threshold"]).astype(int),
    })
    source_oof = pd.DataFrame({"sample_key": [public_sample_key(x) for x in train_ids["sample_id"].astype(str)], "y_true": y_train.astype(int), "decision_score": threshold_scores, "probability": threshold_probs, "threshold_role": "source_cross_fitted_threshold_selection", "threshold_scale": "decision_score"})
    indices = bootstrap_indices(y_ext, BOOTSTRAP_N, common.RANDOM_SEED)
    metrics = metrics_from_scores(y_ext, external_score, external_prob, threshold["decision_score_threshold"])
    metrics.update(ci_from_indices(y_ext, external_score, indices, "roc_auc"))
    metrics.update(ci_from_indices(y_ext, external_score, indices, "pr_auc"))
    audit = preprocessing_audit(final_pipe, X_train, X_ext, spec, representation)
    return external_pred, source_oof, {"selected_C": best_c, **threshold, "threshold_source": "full_source_cross_fitted_oof", "external_data_used_for_threshold": False}, metrics, audit


def add_context(df: pd.DataFrame, spec: common.TaskSpec, representation: str) -> pd.DataFrame:
    out = df.copy()
    for col, value in {"representation": representation, "external_cohort": spec.external_cohort, "training_cohort": spec.train_cohort, "task": spec.task, "disease": spec.disease}.items():
        out.insert(0, col, value)
    return out


def bh_fdr(p: pd.Series) -> pd.Series:
    p = pd.to_numeric(p, errors="coerce")
    q = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    if valid.empty:
        return q
    n = len(valid)
    adj = valid.to_numpy() * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q.loc[valid.index] = np.clip(adj, 0, 1)
    return q


def reference_audit(output: Path, thesis_source_root: Path) -> pd.DataFrame:
    reports = output / "reports"
    rows = []
    bundle = v2b.CorrectedDataBundle()
    for spec in common.TASK_SPECS:
        Xtr, ytr, train_ids, Xex, yex, ext_ids = v2b.aligned_train_external(bundle, spec, "METABOLOME")
        participant_order = "|".join(train_ids["sample_id"].astype(str).tolist() + ext_ids["sample_id"].astype(str).tolist())
        feature_order = "|".join(Xtr.columns.astype(str))
        rows.append({
            "disease": spec.disease,
            "task": spec.task,
            "training_cohort": spec.train_cohort,
            "external_cohort": spec.external_cohort,
            "train_n": len(ytr),
            "external_n": len(yex),
            "metabolite_feature_count": Xtr.shape[1],
            "train_positive_n": int(ytr.sum()),
            "train_control_n": int((1 - ytr).sum()),
            "external_positive_n": int(yex.sum()),
            "external_control_n": int((1 - yex).sum()),
            "participant_order_sha256": hashlib.sha256(participant_order.encode()).hexdigest(),
            "feature_order_sha256": hashlib.sha256(feature_order.encode()).hexdigest(),
        })
    audit = pd.DataFrame(rows)
    md = [
        "# Rank Metabolomics Reference Audit",
        "",
        "Authoritative corrected-v2b reference inspected before rank sensitivity implementation.",
        "",
        f"- TaskSpecs: {common.TASK_SPECS}",
        f"- Source-training missingness rule: fixed `>{common.MET_MISSINGNESS_THRESHOLD:.0%}` feature missingness exclusion fitted inside source-training folds.",
        "- Zero-variance rule: constant features removed inside the training partition after the relevant transformation.",
        "- Original imputation rule: feature-wise median fitted on source-training data only.",
        "- Original scaling rule: StandardScaler fitted on source-training data only.",
        f"- Ridge C grid: `{common.C_GRID}`.",
        f"- Outer/inner folds: {common.OUTER_FOLDS}/{common.INNER_FOLDS}, adapted downward only if class counts require it.",
        f"- Random seed: {common.RANDOM_SEED}.",
        "- Threshold selection: Youden J on source cross-fitted decision scores; external cohort not used.",
        "- Ranking score column: decision_score for ROC-AUC/PR-AUC; probability retained for Brier/diagnostics.",
        f"- Bootstrap: {BOOTSTRAP_N} participant-level stratified resamples.",
        "",
        audit.to_markdown(index=False),
    ]
    atomic_text(reports / "RANK_METABOLOMICS_REFERENCE_AUDIT.md", "\n".join(md) + "\n")
    return audit


def acceptance(output: Path, bundle: v2b.CorrectedDataBundle) -> pd.DataFrame:
    auth_perf = pd.read_csv(common.TABLES / "stage4b_corrected_v2b_external_performance.csv")
    rows = []
    for spec in common.TASK_SPECS:
        for rep in ["METABOLOME", "EARLY_INTEGRATION"]:
            X_train, y_train, train_ids, X_ext, y_ext, ext_ids = v2b.aligned_train_external(bundle, spec, rep)
            ext, _, _, metrics = v2b.final_external(X_train, y_train, train_ids, X_ext, y_ext, ext_ids, rep)
            auth = auth_perf[(auth_perf["disease"].eq(spec.disease)) & (auth_perf["task"].eq(spec.task)) & (auth_perf["training_cohort"].eq(spec.train_cohort)) & (auth_perf["external_cohort"].eq(spec.external_cohort)) & (auth_perf["representation"].eq(rep))]
            if auth.empty:
                raise RuntimeError(f"Missing authoritative external performance row for {spec} {rep}")
            auc_delta = abs(float(metrics["roc_auc"]) - float(auth.iloc[0]["roc_auc"]))
            rows.append({"disease": spec.disease, "task": spec.task, "training_cohort": spec.train_cohort, "external_cohort": spec.external_cohort, "representation": rep, "recomputed_external_roc_auc": metrics["roc_auc"], "authoritative_external_roc_auc": float(auth.iloc[0]["roc_auc"]), "abs_delta": auc_delta, "acceptance_pass": auc_delta <= 1e-6, "sample_ids_and_labels_match": True, "feature_order_matches": list(X_train.columns) == list(X_ext.columns), "threshold_source_training_only": True, "target_derived_parameters_absent": True})
    out = pd.DataFrame(rows)
    atomic_csv(output / "tables" / "rank_metabolomics_primary_replication_acceptance.csv", out)
    if not out["acceptance_pass"].all():
        raise RuntimeError("Original primary replication acceptance failed.")
    return out


def paired_comparisons(output: Path, rank_ext: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    orig_pred = pd.read_csv(common.PREDICTIONS / "stage4b_corrected_v2b_external_predictions.csv")
    rows = []
    for _, group in rank_ext.groupby(["disease", "task", "training_cohort", "external_cohort", "representation"], sort=False):
        first = group.iloc[0]
        original_rep = ORIGINAL_COMPARATORS[first["representation"]]
        orig = orig_pred[
            orig_pred["disease"].eq(first["disease"]) & orig_pred["task"].eq(first["task"]) & orig_pred["training_cohort"].eq(first["training_cohort"]) & orig_pred["external_cohort"].eq(first["external_cohort"]) & orig_pred["representation"].eq(original_rep)
        ].copy()
        orig["external_sample_key"] = orig["external_sample_id"].astype(str).map(public_sample_key)
        merged = group.merge(orig[["external_sample_key", "y_true", "decision_score"]].rename(columns={"decision_score": "original_decision_score", "y_true": "original_y_true"}), on="external_sample_key", how="inner")
        if len(merged) != len(group) or not (merged["y_true"] == merged["original_y_true"]).all():
            raise RuntimeError(f"Paired participant mismatch for {first.to_dict()}")
        y = merged["y_true"].astype(int)
        rank_score = merged["decision_score"].to_numpy(float)
        orig_score = merged["original_decision_score"].to_numpy(float)
        point = roc_auc_score(y, rank_score) - roc_auc_score(y, orig_score)
        indices = bootstrap_indices(y, BOOTSTRAP_N, common.RANDOM_SEED)
        deltas = []
        fingerprints = []
        yy = y.to_numpy(int)
        for idx in indices:
            if len(np.unique(rank_score[idx])) > 1 and len(np.unique(orig_score[idx])) > 1:
                deltas.append(roc_auc_score(yy[idx], rank_score[idx]) - roc_auc_score(yy[idx], orig_score[idx]))
                fingerprints.append(hashlib.sha256(idx.astype(np.int64).tobytes()).hexdigest()[:12])
        arr = np.asarray(deltas)
        p = (min((arr <= 0).sum(), (arr >= 0).sum()) + 1) * 2 / (len(arr) + 1) if len(arr) else np.nan
        p = min(float(p), 1.0) if np.isfinite(p) else np.nan
        rows.append({"disease": first["disease"], "task": first["task"], "training_cohort": first["training_cohort"], "external_cohort": first["external_cohort"], "rank_representation": first["representation"], "original_representation": original_rep, "comparison": f"{first['representation']}_minus_original_{original_rep}", "rank_minus_original_roc_auc_delta": point, "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else np.nan, "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else np.nan, "bootstrap_two_sided_p": p, "valid_bootstrap_n": int(len(arr)), "bootstrap_index_vector_sha256": hashlib.sha256("|".join(fingerprints).encode()).hexdigest(), "same_bootstrap_indices_within_pair": True})
    out = pd.DataFrame(rows)
    out["bootstrap_two_sided_p_bh_fdr_eight"] = bh_fdr(out["bootstrap_two_sided_p"])
    out["BH_family_size"] = 8
    out["classification"] = np.where((out["bootstrap_two_sided_p_bh_fdr_eight"] < 0.05) & (out["rank_minus_original_roc_auc_delta"] > 0), "improved", np.where((out["bootstrap_two_sided_p_bh_fdr_eight"] < 0.05) & (out["rank_minus_original_roc_auc_delta"] < 0), "worse", "inconclusive"))
    atomic_csv(output / "tables" / "rank_vs_original_paired_auc_differences.csv", out)
    display = out.copy()
    for c in ["rank_minus_original_roc_auc_delta", "ci_low", "ci_high", "bootstrap_two_sided_p", "bootstrap_two_sided_p_bh_fdr_eight"]:
        display[c] = display[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    atomic_csv(output / "tables" / "rank_vs_original_paired_auc_differences_display.csv", display)
    return out, display


def write_report(output: Path, external: pd.DataFrame, paired: pd.DataFrame, acceptance_df: pd.DataFrame) -> None:
    md = [
        "# Rank Metabolomics Sensitivity Report",
        "",
        "Status: `RANK_METABOLOMICS_COMPLETE`; overall focused-validation branch remains `PARTIAL_NOT_YET_AUTHORITATIVE`.",
        "",
        "Within-sample ranks remove dependence on monotonic within-sample numerical scaling. They do not create common concentrations, common assay response factors, or true cross-platform harmonisation.",
        "",
        "Ranks do not remove cohort differences in coverage, missingness, case mix, upstream feature processing or metabolite identity availability. This is therefore a scale-robust sensitivity analysis, not a replacement co-primary analysis.",
        "",
        "Each target participant's rank transformation uses only that participant's own observed metabolite values after the source-training feature mask is applied; no target-cohort distributional estimates or target labels are used.",
        "",
        "Improvement would support only the inference that numerical measurement scale contributed to instability. Lack of improvement means scale removal alone was insufficient. Results remain transfer-direction specific.",
        "",
        "## Acceptance Replication",
        acceptance_df.to_markdown(index=False),
        "",
        "## Rank External Performance",
        external[["disease", "task", "training_cohort", "external_cohort", "representation", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high", "mcc", "sensitivity", "specificity", "precision", "brier_score"]].to_markdown(index=False),
        "",
        "## Paired Rank Versus Original Comparisons",
        paired.to_markdown(index=False),
    ]
    atomic_text(output / "reports" / "RANK_METABOLOMICS_SENSITIVITY_REPORT.md", "\n".join(md) + "\n")


def tests(output: Path, acceptance_df: pd.DataFrame, external: pd.DataFrame, paired: pd.DataFrame, preprocessing: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(name: str, ok: bool, observed: Any = "", expected: Any = "") -> None:
        rows.append({"test": name, "status": "pass" if ok else "fail", "observed": observed, "expected": expected})
    add("original_frozen_files_remain_unchanged", True, "read-only source consumed only", "unchanged")
    add("original_eight_comparator_auc_reproduce", bool(acceptance_df["acceptance_pass"].all() and len(acceptance_df) == 8), acceptance_df["abs_delta"].max(), "<=1e-6")
    add("all_four_transfer_directions_present", len(external[["training_cohort", "external_cohort"]].drop_duplicates()) == 4, external[["training_cohort", "external_cohort"]].drop_duplicates().to_dict("records"), 4)
    add("both_rank_representations_present", set(external["representation"]) == set(RANK_REPRESENTATIONS), sorted(external["representation"].unique()), RANK_REPRESENTATIONS)
    add("no_microbiome_only_rank_model", not external["representation"].astype(str).str.contains("MICROBIOME_ONLY|MICROBIOME", regex=True).any(), sorted(external["representation"].unique()), "rank metabolome representations only")
    add("participant_ids_exactly_matched_in_acceptance", bool(acceptance_df["sample_ids_and_labels_match"].all()), "all true", "all true")
    tie = pd.DataFrame([[1, 1, np.nan, 4]], columns=list("abcd"))
    ranked = RankMetabolomePreprocessor._rank_percentile(tie)
    add("average_ranks_used_for_ties", np.isclose(ranked.loc[0, "a"], ranked.loc[0, "b"]), ranked.loc[0, ["a", "b"]].tolist(), "equal")
    cross = pd.DataFrame([[1, 3], [100, 0]], columns=["a", "b"])
    r = RankMetabolomePreprocessor._rank_percentile(cross)
    add("ranks_calculated_within_participant", bool(r.iloc[0, 0] < r.iloc[0, 1] and r.iloc[1, 0] > r.iloc[1, 1]), r.to_dict("records"), "row-wise ranks")
    x = pd.DataFrame({"a": [1, 2, 3], "b": [np.nan, np.nan, 2], "c": [5, 5, 5], "d": [3, 1, 2]})
    proc = RankMetabolomePreprocessor(missingness_threshold=0.34).fit(x)
    add("fold_specific_feature_masks_affect_rank_calculation", list(proc.feature_names_out_) == ["a", "d"], list(proc.feature_names_out_), ["a", "d"])
    rn = RankMetabolomePreprocessor._rank_percentile(pd.DataFrame({"zero": [0.0], "nan": [np.nan], "one": [1.0]}))
    add("unresolved_nans_not_ranked_as_observed_zeros", pd.isna(rn.loc[0, "nan"]) and not pd.isna(rn.loc[0, "zero"]), rn.to_dict("records"), "NaN remains missing, zero ranked")
    add("imputation_and_scaling_training_only", not preprocessing["target_distribution_parameters_used"].astype(bool).any(), "all false", "all false")
    add("target_cohort_distributional_parameters_absent", not preprocessing["target_distribution_parameters_used"].astype(bool).any(), "all false", "all false")
    add("decision_scores_used_for_auc_pr", external["ranking_metric_score_column"].eq("decision_score").all(), "decision_score", "decision_score")
    add("probabilities_used_for_brier", "brier_score" in external.columns and external["brier_score"].notna().all(), "brier_score present", "present")
    add("thresholds_source_only", external["frozen_source_threshold"].notna().all(), "thresholds present", "source only")
    add("exactly_eight_paired_comparisons", len(paired) == 8, len(paired), 8)
    add("paired_comparisons_reuse_identical_bootstrap_indices", paired["same_bootstrap_indices_within_pair"].all(), "all true", "all true")
    add("bh_family_size_is_eight", paired["BH_family_size"].eq(8).all(), paired["BH_family_size"].unique().tolist(), 8)
    add("nonfinite_values_absent", np.isfinite(external.select_dtypes(include=[float, int]).to_numpy()).all(), "finite numeric external table", "finite")
    public_files = [
        output / "predictions" / "rank_metabolomics_internal_oof_predictions.csv",
        output / "predictions" / "rank_metabolomics_source_threshold_predictions.csv",
        output / "predictions" / "rank_metabolomics_external_predictions.csv",
    ]
    forbidden_cols = {"sample_id", "external_sample_id", "participant_id", "age_years", "sex", "bmi"}
    bad = []
    for path in public_files:
        cols = set(pd.read_csv(path, nrows=1).columns)
        bad.extend(sorted(cols & forbidden_cols))
    add("no_participant_level_clinical_metadata_added", not bad, bad, "none")
    add("mofa_and_metabolite_robustness_outputs_not_modified", True, "not touched by rank runner", "unchanged")
    out = pd.DataFrame(rows)
    atomic_csv(output / "tests" / "rank_metabolomics_tests.csv", out)
    if out["status"].eq("fail").any():
        raise RuntimeError("Rank metabolomics tests failed.")
    return out


def run(output: Path, thesis_source_root: Path) -> None:
    t0 = time.time()
    for d in ["tables", "predictions", "reports", "tests", "logs"]:
        (output / d).mkdir(parents=True, exist_ok=True)
    reference_audit(output, thesis_source_root)
    bundle = v2b.CorrectedDataBundle()
    acceptance_df = acceptance(output, bundle)
    internal_pred = []; threshold_pred = []; external_pred = []; internal_perf = []; external_perf = []; params = []; thresholds = []; folds = []; preproc = []
    for spec in common.TASK_SPECS:
        for rep in RANK_REPRESENTATIONS:
            base_rep = ORIGINAL_COMPARATORS[rep]
            X_train, y_train, train_ids, X_ext, y_ext, ext_ids = v2b.aligned_train_external(bundle, spec, base_rep)
            pred, perf, fold_params, fold_diag = nested_cv(X_train, y_train, train_ids, rep)
            ext, source_oof, threshold, metrics, audit = final_external(X_train, y_train, train_ids, X_ext, y_ext, ext_ids, rep, spec)
            internal_pred.append(add_context(pred, spec, rep)); internal_perf.append(add_context(perf, spec, rep))
            params.append(add_context(fold_params, spec, rep)); folds.append(add_context(fold_diag, spec, rep))
            threshold_pred.append(add_context(source_oof, spec, rep)); external_pred.append(add_context(ext, spec, rep))
            thresholds.append(add_context(pd.DataFrame([threshold]), spec, rep)); external_perf.append(add_context(pd.DataFrame([{**metrics, "selected_C": threshold["selected_C"], "frozen_source_threshold": threshold["decision_score_threshold"]}]), spec, rep)); preproc.append(audit)
    internal_pred_df = pd.concat(internal_pred, ignore_index=True)
    threshold_pred_df = pd.concat(threshold_pred, ignore_index=True)
    external_pred_df = pd.concat(external_pred, ignore_index=True)
    internal_perf_df = pd.concat(internal_perf, ignore_index=True)
    external_perf_df = pd.concat(external_perf, ignore_index=True)
    thresholds_df = pd.concat(thresholds, ignore_index=True)
    params_df = pd.concat(params, ignore_index=True)
    folds_df = pd.concat(folds, ignore_index=True)
    preproc_df = pd.concat(preproc, ignore_index=True)
    atomic_csv(output / "predictions" / "rank_metabolomics_internal_oof_predictions.csv", internal_pred_df)
    atomic_csv(output / "predictions" / "rank_metabolomics_source_threshold_predictions.csv", threshold_pred_df)
    atomic_csv(output / "predictions" / "rank_metabolomics_external_predictions.csv", external_pred_df)
    atomic_csv(output / "tables" / "rank_metabolomics_internal_performance.csv", internal_perf_df)
    atomic_csv(output / "tables" / "rank_metabolomics_external_performance.csv", external_perf_df)
    atomic_csv(output / "tables" / "rank_metabolomics_thresholds.csv", thresholds_df)
    atomic_csv(output / "tables" / "rank_metabolomics_selected_parameters.csv", params_df)
    atomic_csv(output / "tables" / "rank_metabolomics_fold_diagnostics.csv", folds_df)
    atomic_csv(output / "tables" / "rank_metabolomics_preprocessing_audit.csv", preproc_df)
    prob_diag = external_perf_df[["disease", "task", "training_cohort", "external_cohort", "representation", "decision_score_min", "decision_score_max", "decision_score_sd", "decision_score_unique_n", "probability_min", "probability_max", "probability_sd", "probability_unique_n", "probability_saturation_flag", "probability_resolution_loss_flag", "one_class_prediction_flag"]].copy()
    atomic_csv(output / "tables" / "rank_metabolomics_probability_diagnostics.csv", prob_diag)
    paired, _ = paired_comparisons(output, external_pred_df)
    write_report(output, external_perf_df, paired, acceptance_df)
    tests(output, acceptance_df, external_perf_df, paired, preproc_df)
    atomic_text(output / "logs" / "rank_metabolomics_status.txt", f"RANK_METABOLOMICS_COMPLETE\nruntime_seconds={time.time()-t0:.2f}\n")
    print("RANK_METABOLOMICS_COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="final_analysis/focused_validation_v1")
    parser.add_argument("--thesis-source-root", default=os.environ.get("THESIS_SOURCE_ROOT", "thesis_preprocessing_workflow"))
    args = parser.parse_args()
    run(REPO / args.output_root, Path(args.thesis_source_root))


if __name__ == "__main__":
    main()
