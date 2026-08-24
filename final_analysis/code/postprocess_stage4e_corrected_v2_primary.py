from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, matthews_corrcoef, roc_auc_score

from stage4_corrected_v2_common import MANIFESTS, REPORTS, TABLES, atomic_csv, atomic_json, ensure_v2_dirs


BOOTSTRAP_SEED = 20260820
BOOTSTRAP_N = 2000
EXPECTED_PARTICIPANTS = {"Franzosa": 137, "iHMP": 91, "Sinha": 131, "Yachida": 252}
GRAPH_MODES = ["identity", "kegg_only", "kegg_njc19", "conservative_kegg_njc19"]
REPRESENTATIONS = GRAPH_MODES + ["ridge_early_integration"]
PRIMARY_ARCHITECTURE = {
    "architecture_id": "primary_one_layer_mean",
    "layers": 1,
    "hidden_dim": 16,
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "readout": "mean",
}

INPUTS = {
    "performance": TABLES / "stage4e_corrected_v2_within_fold_seed_performance_primary.csv",
    "predictions": TABLES / "stage4e_corrected_v2_within_oof_predictions_primary.csv",
    "thresholds": TABLES / "stage4e_corrected_v2_within_thresholds_primary.csv",
    "loss": TABLES / "stage4e_corrected_v2_within_loss_convergence_primary.csv",
    "masking": TABLES / "stage4e_corrected_v2_preprocessing_masking_diagnostics_primary.csv",
    "architecture_records_old_name": TABLES / "stage4e_corrected_v2_selected_hyperparameters_primary.csv",
    "split_manifest": TABLES / "stage4e_corrected_v2_split_fold_manifest.csv",
    "graph_hash": MANIFESTS / "stage4d_corrected_v2_graph_hash.txt",
}

OUTPUTS = {
    "validation": TABLES / "stage4e_corrected_v2_primary_postrun_validation_tests.csv",
    "architecture_records": TABLES / "stage4e_corrected_v2_prespecified_architecture_records_primary.csv",
    "seed_oof_performance": TABLES / "stage4e_corrected_v2_seed_level_oof_performance_primary.csv",
    "seed_summary": TABLES / "stage4e_corrected_v2_seed_variability_summary_primary.csv",
    "ensemble_predictions": TABLES / "stage4e_corrected_v2_ensemble_oof_predictions_primary.csv",
    "ensemble_performance": TABLES / "stage4e_corrected_v2_ensemble_oof_performance_primary.csv",
    "bootstrap_ci": TABLES / "stage4e_corrected_v2_ensemble_bootstrap_ci_primary.csv",
    "paired_comparisons": TABLES / "stage4e_corrected_v2_paired_model_comparisons_primary.csv",
    "diagnostics": TABLES / "stage4e_corrected_v2_probability_convergence_diagnostics_primary.csv",
    "one_class_rows": TABLES / "stage4e_corrected_v2_one_class_prediction_rows_primary.csv",
    "saturation_rows": TABLES / "stage4e_corrected_v2_saturation_rows_primary.csv",
    "interpretation": TABLES / "stage4e_corrected_v2_graph_topology_interpretation_primary.csv",
    "manifest": MANIFESTS / "stage4e_corrected_v2_primary_postrun_manifest.json",
    "report": REPORTS / "STAGE4E_CORRECTED_V2_PRIMARY_POSTRUN_AUDIT.md",
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_input(name: str) -> pd.DataFrame:
    path = INPUTS[name]
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise ValueError(f"Empty Stage 4E primary input: {path}")
    return df


def status_rows() -> list[dict[str, Any]]:
    return []


def add_check(rows: list[dict[str, Any]], test: str, status: str, observed: Any = "", expected: Any = "", notes: str = "") -> None:
    rows.append({"test": test, "status": status, "observed": observed, "expected": expected, "notes": notes})


def representation_label(row: pd.Series | dict[str, Any]) -> str:
    model_family = str(row.get("model_family", ""))
    if model_family == "ridge_early_integration":
        return "ridge_early_integration"
    return str(row.get("graph_mode", ""))


def validate_sources(perf: pd.DataFrame, pred: pd.DataFrame, arch: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    gcn_perf = perf[perf["model_family"].eq("gcn")]
    ridge_perf = perf[perf["model_family"].eq("ridge_early_integration")]
    add_check(rows, "gcn_fold_seed_performance_rows", "pass" if len(gcn_perf) == 400 else "fail", len(gcn_perf), 400)
    add_check(rows, "ridge_fold_performance_rows", "pass" if len(ridge_perf) == 20 else "fail", len(ridge_perf), 20)
    add_check(rows, "total_fold_performance_rows", "pass" if len(perf) == 420 else "fail", len(perf), 420)
    add_check(rows, "oof_prediction_rows", "pass" if len(pred) == 12831 else "fail", len(pred), 12831)

    gcn_counts = gcn_perf.groupby(["cohort", "graph_mode"]).size()
    add_check(rows, "gcn_25_fold_seed_rows_per_cohort_graph_mode", "pass" if (len(gcn_counts) == 16 and gcn_counts.eq(25).all()) else "fail", gcn_counts.to_dict(), "16 groups all n=25")
    ridge_counts = ridge_perf.groupby(["cohort", "graph_mode"]).size()
    add_check(rows, "ridge_not_applicable_graph_mode", "pass" if set(ridge_perf["graph_mode"]) == {"not_applicable"} else "fail", sorted(ridge_perf["graph_mode"].unique()), "not_applicable")
    add_check(rows, "four_cohorts_present", "pass" if set(perf["cohort"]) == set(EXPECTED_PARTICIPANTS) else "fail", sorted(perf["cohort"].unique()), sorted(EXPECTED_PARTICIPANTS))
    add_check(rows, "four_gcn_graph_modes_present", "pass" if set(gcn_perf["graph_mode"]) == set(GRAPH_MODES) else "fail", sorted(gcn_perf["graph_mode"].unique()), GRAPH_MODES)
    add_check(rows, "five_seeds_present_for_gcn", "pass" if gcn_perf.groupby(["cohort", "graph_mode"])["seed"].nunique().eq(5).all() else "fail", gcn_perf.groupby(["cohort", "graph_mode"])["seed"].nunique().to_dict(), 5)
    add_check(rows, "five_outer_folds_present", "pass" if perf.groupby(["cohort", "model_family"])["outer_fold"].nunique().min() == 5 else "fail", perf.groupby(["cohort", "model_family"])["outer_fold"].nunique().to_dict(), 5)

    participant_counts = pred.groupby("cohort")["sample_id"].nunique().to_dict()
    add_check(rows, "participant_counts", "pass" if participant_counts == EXPECTED_PARTICIPANTS else "fail", participant_counts, EXPECTED_PARTICIPANTS)
    gcn_sets_ok = True
    for (cohort, seed), part in pred[pred["model_family"].eq("gcn")].groupby(["cohort", "seed"]):
        sets = [set(g["sample_id"]) for _, g in part.groupby("graph_mode")]
        gcn_sets_ok = gcn_sets_ok and all(s == sets[0] for s in sets)
    add_check(rows, "identical_gcn_sample_sets_across_graph_modes_by_seed", "pass" if gcn_sets_ok else "fail")
    ridge_ok = True
    for cohort, part in pred.groupby("cohort"):
        ridge_set = set(part.loc[part["model_family"].eq("ridge_early_integration"), "sample_id"])
        gcn_set = set(part.loc[part["model_family"].eq("gcn"), "sample_id"])
        ridge_ok = ridge_ok and ridge_set == gcn_set
    add_check(rows, "ridge_same_participant_set_as_gcn", "pass" if ridge_ok else "fail")

    dup_cols = ["cohort", "model_family", "graph_mode", "architecture_id", "seed", "sample_id"]
    add_check(rows, "no_duplicate_model_sample_records", "pass" if not pred.duplicated(dup_cols).any() else "fail", int(pred.duplicated(dup_cols).sum()), 0)
    add_check(rows, "finite_decision_scores", "pass" if np.isfinite(pred["decision_score"]).all() else "fail")
    add_check(rows, "finite_probabilities", "pass" if np.isfinite(pred["probability"]).all() else "fail")
    add_check(rows, "probabilities_in_unit_interval", "pass" if pred["probability"].between(0, 1).all() else "fail")
    add_check(rows, "binary_outcomes", "pass" if set(pred["y_true"].unique()).issubset({0, 1}) else "fail", sorted(pred["y_true"].unique()), [0, 1])
    add_check(rows, "ranking_metrics_from_decision_scores", "pass" if perf["ranking_metrics_use"].eq("continuous_logit_or_decision_score").all() else "fail")

    arch_ok = (
        set(arch["architecture_id"]) == {PRIMARY_ARCHITECTURE["architecture_id"]}
        and arch["layers"].eq(PRIMARY_ARCHITECTURE["layers"]).all()
        and arch["hidden_dim"].eq(PRIMARY_ARCHITECTURE["hidden_dim"]).all()
        and np.allclose(arch["dropout"], PRIMARY_ARCHITECTURE["dropout"])
        and np.allclose(arch["learning_rate"], PRIMARY_ARCHITECTURE["learning_rate"])
        and arch["readout"].eq(PRIMARY_ARCHITECTURE["readout"]).all()
        and arch["hpo_candidate"].eq(False).all()
    )
    add_check(rows, "primary_architecture_integrity", "pass" if arch_ok else "fail", "primary_one_layer_mean records", PRIMARY_ARCHITECTURE)

    add_check(
        rows,
        "train_validation_test_sample_id_overlap_check",
        "unresolved",
        "split manifest contains counts only",
        "explicit train/validation/test sample IDs",
        "Cannot verify non-overlap or identical early-stopping membership from saved outputs; future reruns should export sample IDs.",
    )
    add_check(
        rows,
        "identical_outer_test_membership_across_graph_modes_and_ridge",
        "pass" if split["same_fold_manifest_for_all_graph_modes"].eq(True).all() else "fail",
        bool(split["same_fold_manifest_for_all_graph_modes"].eq(True).all()),
        True,
        "Supported by split manifest flag and OOF sample-set checks.",
    )
    return pd.DataFrame(rows)


def metrics_from_predictions(df: pd.DataFrame, class_column: str = "predicted_class") -> dict[str, Any]:
    y = df["y_true"].astype(int)
    score = df["decision_score"].astype(float)
    prob = df["probability"].astype(float)
    pred = df[class_column].astype(int)
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    return {
        "n": int(len(df)),
        "positive_n": int(y.sum()),
        "negative_n": int((1 - y).sum()),
        "roc_auc": float(roc_auc_score(y, score)) if y.nunique() == 2 and score.nunique() > 1 else np.nan,
        "pr_auc": float(average_precision_score(y, score)) if y.nunique() == 2 else np.nan,
        "brier_score": float(brier_score_loss(y, prob)) if y.nunique() == 2 else np.nan,
        "mcc": float(matthews_corrcoef(y, pred)) if pred.nunique() > 1 else 0.0,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "predicted_positive_n": int(pred.sum()),
        "predicted_negative_n": int((1 - pred).sum()),
        "decision_score_min": float(score.min()),
        "decision_score_max": float(score.max()),
        "decision_score_sd": float(score.std(ddof=0)),
        "decision_score_iqr": float(score.quantile(0.75) - score.quantile(0.25)),
        "probability_min": float(prob.min()),
        "probability_max": float(prob.max()),
        "probability_sd": float(prob.std(ddof=0)),
        "probability_iqr": float(prob.quantile(0.75) - prob.quantile(0.25)),
        "unique_decision_scores_rounded_12": int(score.round(12).nunique()),
        "unique_probabilities_rounded_12": int(prob.round(12).nunique()),
        "compressed_probability_flag": bool(prob.std(ddof=0) < 0.01),
        "saturation_flag": bool((prob <= 1e-6).any() or (prob >= 1 - 1e-6).any()),
        "constant_logit_failure": bool(score.round(12).nunique() <= 1),
        "one_class_prediction_flag": bool(pred.nunique() <= 1),
    }


def seed_level_oof_performance(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    gcn = pred[pred["model_family"].eq("gcn")].copy()
    for keys, group in gcn.groupby(["cohort", "disease", "task", "graph_mode", "seed"], sort=True):
        cohort, disease, task, graph_mode, seed = keys
        if group["sample_id"].nunique() != EXPECTED_PARTICIPANTS[cohort]:
            raise ValueError(f"Incomplete GCN OOF participant coverage for {keys}")
        rows.append({"cohort": cohort, "disease": disease, "task": task, "model_family": "gcn", "representation": graph_mode, "graph_mode": graph_mode, "seed": seed, **metrics_from_predictions(group)})
    ridge = pred[pred["model_family"].eq("ridge_early_integration")].copy()
    for keys, group in ridge.groupby(["cohort", "disease", "task"], sort=True):
        cohort, disease, task = keys
        rows.append({"cohort": cohort, "disease": disease, "task": task, "model_family": "ridge_early_integration", "representation": "ridge_early_integration", "graph_mode": "not_applicable", "seed": "not_applicable", **metrics_from_predictions(group)})
    return pd.DataFrame(rows)


def seed_variability(seed_perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    gcn = seed_perf[seed_perf["model_family"].eq("gcn")]
    for keys, group in gcn.groupby(["cohort", "representation"], sort=True):
        cohort, rep = keys
        for metric in ["roc_auc", "pr_auc", "brier_score", "mcc", "sensitivity", "specificity"]:
            vals = group[metric].dropna().astype(float)
            rows.append({"cohort": cohort, "representation": rep, "metric": metric, "seed_n": len(vals), "mean": vals.mean(), "median": vals.median(), "sd": vals.std(ddof=1), "min": vals.min(), "max": vals.max(), "iqr": vals.quantile(0.75) - vals.quantile(0.25)})
    return pd.DataFrame(rows)


def ensemble_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    gcn = pred[pred["model_family"].eq("gcn")]
    for keys, group in gcn.groupby(["cohort", "disease", "task", "graph_mode", "sample_id"], sort=True):
        cohort, disease, task, graph_mode, sample_id = keys
        if group["seed"].nunique() != 5:
            raise ValueError(f"Expected five seed predictions for {keys}")
        y_values = group["y_true"].unique()
        if len(y_values) != 1:
            raise ValueError(f"Conflicting y_true for {keys}")
        votes = group["predicted_class"].astype(int)
        vote_sum = int(votes.sum())
        rows.append({
            "cohort": cohort,
            "disease": disease,
            "task": task,
            "model_family": "gcn",
            "representation": graph_mode,
            "graph_mode": graph_mode,
            "sample_id": sample_id,
            "y_true": int(y_values[0]),
            "decision_score": float(group["decision_score"].mean()),
            "probability": float(group["probability"].mean()),
            "predicted_class": int(vote_sum >= 3),
            "class_decision_rule": "seed_majority_vote",
            "positive_votes": vote_sum,
            "negative_votes": int(5 - vote_sum),
            "vote_tie": False,
            "seed_n": 5,
        })
    ridge = pred[pred["model_family"].eq("ridge_early_integration")].copy()
    ridge_rows = ridge.assign(
        representation="ridge_early_integration",
        class_decision_rule="fold_specific_frozen_threshold",
        positive_votes=np.nan,
        negative_votes=np.nan,
        vote_tie=False,
        seed_n=1,
    )
    rows_df = pd.DataFrame(rows)
    keep = list(rows_df.columns)
    return pd.concat([rows_df, ridge_rows[keep]], ignore_index=True)


def ensemble_performance(ensemble: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in ensemble.groupby(["cohort", "disease", "task", "model_family", "representation", "graph_mode"], sort=True):
        cohort, disease, task, family, rep, mode = keys
        rows.append({"cohort": cohort, "disease": disease, "task": task, "model_family": family, "representation": rep, "graph_mode": mode, **metrics_from_predictions(group)})
    return pd.DataFrame(rows)


def bootstrap_metric_ci(ensemble: pd.DataFrame, n_bootstrap: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cohort, cdf in ensemble.groupby("cohort", sort=True):
        ids = cdf[["sample_id", "y_true"]].drop_duplicates()
        pos = ids.loc[ids["y_true"].eq(1), "sample_id"].to_numpy()
        neg = ids.loc[ids["y_true"].eq(0), "sample_id"].to_numpy()
        reps = sorted(cdf["representation"].unique())
        values = {rep: {"roc_auc": [], "pr_auc": [], "brier_score": []} for rep in reps}
        for _ in range(n_bootstrap):
            sampled = np.concatenate([rng.choice(pos, size=len(pos), replace=True), rng.choice(neg, size=len(neg), replace=True)])
            for rep in reps:
                sub = cdf[cdf["representation"].eq(rep)].set_index("sample_id").loc[sampled].reset_index()
                m = metrics_from_predictions(sub)
                for metric in values[rep]:
                    values[rep][metric].append(m[metric])
        for rep in reps:
            for metric, vals in values[rep].items():
                arr = np.asarray(vals, dtype=float)
                rows.append({"cohort": cohort, "representation": rep, "metric": metric, "bootstrap_n": n_bootstrap, "bootstrap_seed": seed, "estimate": float(metrics_from_predictions(cdf[cdf["representation"].eq(rep)])[metric]), "ci_low": float(np.nanpercentile(arr, 2.5)), "ci_high": float(np.nanpercentile(arr, 97.5))})
    return pd.DataFrame(rows)


def bh_fdr(p_values: pd.Series) -> pd.Series:
    p = p_values.astype(float).to_numpy()
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    cumulative = 1.0
    for i in range(n - 1, -1, -1):
        cumulative = min(cumulative, ranked[i] * n / (i + 1))
        adjusted[order[i]] = cumulative
    return pd.Series(np.minimum(adjusted, 1.0), index=p_values.index)


def paired_comparisons(ensemble: pd.DataFrame, n_bootstrap: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    contrasts = [
        ("kegg_njc19", "identity"),
        ("conservative_kegg_njc19", "identity"),
        ("kegg_njc19", "conservative_kegg_njc19"),
        ("kegg_only", "identity"),
        ("kegg_njc19", "ridge_early_integration"),
        ("conservative_kegg_njc19", "ridge_early_integration"),
    ]
    rng = np.random.default_rng(seed)
    rows = []
    for cohort, cdf in ensemble.groupby("cohort", sort=True):
        ids = cdf[["sample_id", "y_true"]].drop_duplicates()
        pos = ids.loc[ids["y_true"].eq(1), "sample_id"].to_numpy()
        neg = ids.loc[ids["y_true"].eq(0), "sample_id"].to_numpy()
        by_rep = {rep: part.set_index("sample_id") for rep, part in cdf.groupby("representation")}
        for model_a, model_b in contrasts:
            a = by_rep[model_a].loc[ids["sample_id"]]
            b = by_rep[model_b].loc[ids["sample_id"]]
            obs = roc_auc_score(ids["y_true"], a["decision_score"]) - roc_auc_score(ids["y_true"], b["decision_score"])
            diffs = []
            for _ in range(n_bootstrap):
                sampled = np.concatenate([rng.choice(pos, size=len(pos), replace=True), rng.choice(neg, size=len(neg), replace=True)])
                yy = ids.set_index("sample_id").loc[sampled, "y_true"]
                aa = by_rep[model_a].loc[sampled, "decision_score"]
                bb = by_rep[model_b].loc[sampled, "decision_score"]
                diffs.append(roc_auc_score(yy, aa) - roc_auc_score(yy, bb))
            arr = np.asarray(diffs, dtype=float)
            le0 = int((arr <= 0).sum())
            ge0 = int((arr >= 0).sum())
            p = 2 * (min(le0, ge0) + 1) / (len(arr) + 1)
            lo, hi = np.nanpercentile(arr, [2.5, 97.5])
            rows.append({
                "cohort": cohort,
                "model_a": model_a,
                "model_b": model_b,
                "comparison": f"{model_a}_minus_{model_b}",
                "observed_delta_roc_auc": float(obs),
                "bootstrap_median_delta_roc_auc": float(np.nanmedian(arr)),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "bootstrap_n": n_bootstrap,
                "bootstrap_seed": seed,
                "proportion_delta_le_zero": float(le0 / len(arr)),
                "proportion_delta_ge_zero": float(ge0 / len(arr)),
                "finite_resample_two_sided_p": float(min(p, 1.0)),
                "interpretation": "CI excludes zero in favour of A" if lo > 0 else ("CI excludes zero in favour of B" if hi < 0 else "inconclusive"),
            })
    out = pd.DataFrame(rows)
    if len(out) != 24:
        raise ValueError(f"Expected 24 paired comparisons, observed {len(out)}")
    out["bh_fdr_p"] = bh_fdr(out["finite_resample_two_sided_p"])
    return out


def diagnostics_table(perf: pd.DataFrame, loss: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final_loss = loss.sort_values("epoch").groupby(["cohort", "model_family", "graph_mode", "seed", "outer_fold"], dropna=False).tail(1)
    convergence = final_loss.groupby(["cohort", "model_family", "graph_mode"], dropna=False).agg(
        model_rows=("epoch", "size"),
        median_epochs_run=("epoch", "median"),
        max_epochs_run=("epoch", "max"),
        median_final_train_loss=("train_loss", "median"),
        median_final_validation_loss=("validation_loss", "median"),
    ).reset_index()
    diag = perf.groupby(["cohort", "model_family", "graph_mode"], dropna=False).agg(
        rows=("roc_auc", "size"),
        compressed_probability_rows=("compressed_probability_flag", "sum"),
        saturation_rows=("saturation_flag", "sum"),
        one_class_prediction_rows=("one_class_prediction_flag", "sum"),
        constant_logit_failure_rows=("constant_logit_failure", "sum"),
    ).reset_index()
    for col in ["compressed_probability_rows", "saturation_rows", "one_class_prediction_rows", "constant_logit_failure_rows"]:
        diag[col.replace("_rows", "_proportion")] = diag[col] / diag["rows"]
    diag = diag.merge(convergence, on=["cohort", "model_family", "graph_mode"], how="left")
    return diag, perf[perf["one_class_prediction_flag"].astype(bool)].copy(), perf[perf["saturation_flag"].astype(bool)].copy()


def model_a_significantly_better(row: pd.Series) -> bool:
    """Directional paired-bootstrap decision after the prespecified 24-test BH-FDR family."""
    return bool(
        float(row["observed_delta_roc_auc"]) > 0
        and float(row["ci_low"]) > 0
        and float(row["bh_fdr_p"]) < 0.05
    )


def model_a_significantly_worse(row: pd.Series) -> bool:
    """Directional paired-bootstrap decision after the prespecified 24-test BH-FDR family."""
    return bool(
        float(row["observed_delta_roc_auc"]) < 0
        and float(row["ci_high"]) < 0
        and float(row["bh_fdr_p"]) < 0.05
    )


def directional_comparison_summary(paired: pd.DataFrame, comparisons: list[str]) -> dict[str, Any]:
    part = paired[paired["comparison"].isin(comparisons)].copy()
    if part.empty:
        return {
            "any_better": False,
            "any_worse": False,
            "better_models": "",
            "worse_models": "",
        }
    better, worse = [], []
    for _, row in part.iterrows():
        label = f"{row['model_a']} vs {row['model_b']}"
        if model_a_significantly_better(row):
            better.append(f"{label}: model_a higher ROC-AUC")
        if model_a_significantly_worse(row):
            worse.append(f"{label}: model_a lower ROC-AUC")
    return {
        "any_better": bool(better),
        "any_worse": bool(worse),
        "better_models": "; ".join(better),
        "worse_models": "; ".join(worse),
    }


def interpretation_table(performance: pd.DataFrame, paired: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for cohort, part in performance.groupby("cohort", sort=True):
        roc = dict(zip(part["representation"], part["roc_auc"]))
        gcn_modes = [m for m in GRAPH_MODES if m in roc]
        best = max(gcn_modes, key=lambda m: roc.get(m, -np.inf))
        identity_summary = {"any_better": False, "any_worse": False, "better_models": "", "worse_models": ""}
        ridge_summary = {"any_better": False, "any_worse": False, "better_models": "", "worse_models": ""}
        if paired is not None and not paired.empty:
            cp = paired[paired["cohort"].eq(cohort)]
            identity_summary = directional_comparison_summary(
                cp,
                ["kegg_njc19_minus_identity", "conservative_kegg_njc19_minus_identity", "kegg_only_minus_identity"],
            )
            ridge_summary = directional_comparison_summary(
                cp,
                ["kegg_njc19_minus_ridge_early_integration", "conservative_kegg_njc19_minus_ridge_early_integration"],
            )
        rows.append({
            "cohort": cohort,
            "identity_roc_auc": roc.get("identity", np.nan),
            "kegg_only_roc_auc": roc.get("kegg_only", np.nan),
            "kegg_njc19_roc_auc": roc.get("kegg_njc19", np.nan),
            "conservative_kegg_njc19_roc_auc": roc.get("conservative_kegg_njc19", np.nan),
            "ridge_early_integration_roc_auc": roc.get("ridge_early_integration", np.nan),
            "best_gcn_mode": best,
            "any_biological_graph_significantly_better_than_identity": identity_summary["any_better"],
            "any_biological_graph_significantly_worse_than_identity": identity_summary["any_worse"],
            "any_tested_graph_gcn_significantly_better_than_ridge": ridge_summary["any_better"],
            "any_tested_graph_gcn_significantly_worse_than_ridge": ridge_summary["any_worse"],
            "significant_better_models": "; ".join(v for v in [identity_summary["better_models"], ridge_summary["better_models"]] if v),
            "significant_worse_models": "; ".join(v for v in [identity_summary["worse_models"], ridge_summary["worse_models"]] if v),
            "interpretation_basis": "Directional decision requires observed_delta_roc_auc sign, 95% bootstrap CI excluding zero in the same direction, and BH-FDR < 0.05 across the prespecified 24-comparison family.",
            "conservative_wording": "Restricted to corrected-v2 cohorts, fixed molecular inputs, fixed graph policies and low-complexity primary GCN architecture; not evidence that GCNs or biological topology fail generally.",
        })
    return pd.DataFrame(rows)


def write_report(validation: pd.DataFrame, ensemble_perf: pd.DataFrame | None, paired: pd.DataFrame | None, bootstrap_done: bool) -> None:
    lines = [
        "# Stage 4E Corrected-v2 Primary Post-run Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "No fitted GCN or ridge model was rerun during this post-processing pass.",
        "",
        "## Validation",
        validation.to_markdown(index=False),
    ]
    if ensemble_perf is not None:
        lines += ["", "## Authoritative Ensemble OOF Performance", ensemble_perf.to_markdown(index=False)]
    if paired is not None:
        lines += ["", "## Prespecified Paired Comparisons", paired.to_markdown(index=False)]
    lines += [
        "",
        "## Directional Topology Interpretation",
        "Under the fixed one-layer mean-pooling GCN, biological topology did not significantly improve ROC-AUC over identity in any cohort. In Franzosa, KEGG-only, broad KEGG+NJC19 and conservative KEGG+NJC19 graphs significantly reduced ROC-AUC relative to identity after BH-FDR. Broad and conservative graph GCNs were significantly inferior to the matched early-integration ridge comparator in all four cohorts. These findings are architecture- and graph-policy-specific and do not establish that GCNs or biological topology fail generally.",
        "",
        "## Authority",
        "- Authoritative: ensemble OOF performance, seed variability, participant-level bootstrap CIs, paired topology comparisons and diagnostics from the primary within-cohort run.",
        "- Non-authoritative: smoke outputs, historical Stage 4E outputs, fold-mean tables as the main performance estimate, unexecuted sensitivities and unexecuted external stress tests.",
        f"- Bootstrap executed: {bootstrap_done}.",
        "",
    ]
    OUTPUTS["report"].write_text("\n".join(lines))


def postprocess(structural_dry_run: bool = False, n_bootstrap: int = BOOTSTRAP_N) -> None:
    ensure_v2_dirs()
    perf = read_input("performance")
    pred = read_input("predictions")
    thresholds = read_input("thresholds")
    loss = read_input("loss")
    masking = read_input("masking")
    arch = read_input("architecture_records_old_name")
    split = read_input("split_manifest")
    validation = validate_sources(perf, pred, arch, split)
    atomic_csv(validation, OUTPUTS["validation"])
    atomic_csv(arch, OUTPUTS["architecture_records"])
    hard_fail = validation["status"].eq("fail").any()
    if hard_fail:
        raise RuntimeError("Stage 4E primary post-run validation failed.")

    seed_perf = seed_level_oof_performance(pred)
    seed_summary = seed_variability(seed_perf)
    ensemble = ensemble_predictions(pred)
    ensemble_perf = ensemble_performance(ensemble)
    diag, one_class, saturation = diagnostics_table(perf, loss)

    atomic_csv(seed_perf, OUTPUTS["seed_oof_performance"])
    atomic_csv(seed_summary, OUTPUTS["seed_summary"])
    atomic_csv(ensemble, OUTPUTS["ensemble_predictions"])
    atomic_csv(ensemble_perf, OUTPUTS["ensemble_performance"])
    atomic_csv(diag, OUTPUTS["diagnostics"])
    atomic_csv(one_class, OUTPUTS["one_class_rows"])
    atomic_csv(saturation, OUTPUTS["saturation_rows"])

    bootstrap_ci = None
    paired = None
    if not structural_dry_run:
        bootstrap_ci = bootstrap_metric_ci(ensemble, n_bootstrap=n_bootstrap)
        paired = paired_comparisons(ensemble, n_bootstrap=n_bootstrap)
        atomic_csv(bootstrap_ci, OUTPUTS["bootstrap_ci"])
        atomic_csv(paired, OUTPUTS["paired_comparisons"])
        atomic_csv(interpretation_table(ensemble_perf, paired), OUTPUTS["interpretation"])
    else:
        atomic_csv(interpretation_table(ensemble_perf, None), OUTPUTS["interpretation"])

    input_hashes = {name: file_sha256(path) for name, path in INPUTS.items() if path.exists() and path.is_file()}
    output_hashes = {name: file_sha256(path) for name, path in OUTPUTS.items() if path.exists() and path.is_file() and path.suffix != ".json"}
    manifest = {
        "status": "structural_dry_run_complete" if structural_dry_run else "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "stage4d_graph_hash": INPUTS["graph_hash"].read_text().strip() if INPUTS["graph_hash"].exists() else "",
        "architecture": PRIMARY_ARCHITECTURE,
        "seeds": sorted(pred.loc[pred["model_family"].eq("gcn"), "seed"].astype(int).unique().tolist()),
        "outer_folds": sorted(pred["outer_fold"].astype(int).unique().tolist()),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_count": 0 if structural_dry_run else n_bootstrap,
        "participant_counts": pred.groupby("cohort")["sample_id"].nunique().to_dict(),
        "external_analysis_executed": False,
        "sensitivity_analysis_executed": False,
        "model_refitting_during_postprocessing": False,
        "authoritative_outputs": ["ensemble_oof_performance", "seed_variability", "bootstrap_ci", "paired_comparisons", "diagnostics"],
        "non_authoritative_outputs": ["smoke_outputs", "historical_stage4e_outputs", "fold_mean_as_main_performance", "unexecuted_sensitivities", "unexecuted_external_stress_tests"],
    }
    atomic_json(manifest, OUTPUTS["manifest"])
    write_report(validation, ensemble_perf, paired, bootstrap_done=not structural_dry_run)
    print("STAGE4E_CORRECTED_V2_PRIMARY_POSTPROCESS_STRUCTURAL_DRY_RUN_COMPLETE" if structural_dry_run else "STAGE4E_CORRECTED_V2_PRIMARY_POSTPROCESS_COMPLETE")


def correct_interpretation_only() -> None:
    ensure_v2_dirs()
    if not OUTPUTS["ensemble_performance"].exists():
        raise FileNotFoundError(OUTPUTS["ensemble_performance"])
    if not OUTPUTS["paired_comparisons"].exists():
        raise FileNotFoundError(OUTPUTS["paired_comparisons"])
    ensemble_perf = pd.read_csv(OUTPUTS["ensemble_performance"])
    paired = pd.read_csv(OUTPUTS["paired_comparisons"])
    validation = pd.read_csv(OUTPUTS["validation"]) if OUTPUTS["validation"].exists() else pd.DataFrame()
    atomic_csv(interpretation_table(ensemble_perf, paired), OUTPUTS["interpretation"])
    manifest = {}
    if OUTPUTS["manifest"].exists():
        manifest = json.loads(OUTPUTS["manifest"].read_text())
    manifest.update({
        "directional_graph_topology_interpretation_corrected": True,
        "directional_graph_topology_interpretation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "directional_interpretation_basis": "observed_delta_roc_auc sign plus CI direction plus BH-FDR < 0.05; not significance alone",
    })
    manifest.setdefault("output_hashes", {})
    manifest["output_hashes"]["interpretation"] = file_sha256(OUTPUTS["interpretation"])
    atomic_json(manifest, OUTPUTS["manifest"])
    write_report(validation, ensemble_perf, paired, bootstrap_done=True)
    print("STAGE4E_CORRECTED_V2_DIRECTIONAL_INTERPRETATION_CORRECTED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process completed corrected-v2 Stage 4E primary within-cohort outputs without model fitting.")
    parser.add_argument("--run", action="store_true", help="Run full post-processing including 2,000 participant-level bootstrap resamples.")
    parser.add_argument("--structural-dry-run", action="store_true", help="Run non-bootstrap structural post-processing and validation.")
    parser.add_argument("--correct-interpretation-only", action="store_true", help="Rewrite only the directional graph interpretation/report from existing frozen paired-comparison outputs.")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_N)
    args = parser.parse_args()
    if args.correct_interpretation_only:
        correct_interpretation_only()
        return
    if args.structural_dry_run:
        postprocess(structural_dry_run=True, n_bootstrap=0)
        return
    if args.run:
        postprocess(structural_dry_run=False, n_bootstrap=args.bootstrap)
        return
    raise SystemExit("Use --structural-dry-run, --correct-interpretation-only or --run.")


if __name__ == "__main__":
    main()
