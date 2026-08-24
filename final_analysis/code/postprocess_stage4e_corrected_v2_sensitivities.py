from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from stage4_corrected_v2_common import MANIFESTS, REPORTS, TABLES, atomic_csv, atomic_json, ensure_v2_dirs
from postprocess_stage4e_corrected_v2_primary import (
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    EXPECTED_PARTICIPANTS,
    GRAPH_MODES,
    bh_fdr,
    metrics_from_predictions,
    model_a_significantly_better,
    model_a_significantly_worse,
)


ARCHITECTURES = ["sensitivity_two_layer_mean", "sensitivity_one_layer_gated"]
COMPARISONS = [
    ("kegg_njc19", "identity"),
    ("conservative_kegg_njc19", "identity"),
    ("kegg_njc19", "conservative_kegg_njc19"),
    ("kegg_only", "identity"),
    ("kegg_njc19", "ridge_early_integration"),
    ("conservative_kegg_njc19", "ridge_early_integration"),
]

INPUTS = {
    "performance": TABLES / "stage4e_corrected_v2_within_fold_seed_performance_sensitivities.csv",
    "predictions": TABLES / "stage4e_corrected_v2_within_oof_predictions_sensitivities.csv",
    "thresholds": TABLES / "stage4e_corrected_v2_within_thresholds_sensitivities.csv",
    "loss": TABLES / "stage4e_corrected_v2_within_loss_convergence_sensitivities.csv",
    "masking": TABLES / "stage4e_corrected_v2_preprocessing_masking_diagnostics_sensitivities.csv",
    "architecture_records_old_name": TABLES / "stage4e_corrected_v2_selected_hyperparameters_sensitivities.csv",
    "primary_predictions": TABLES / "stage4e_corrected_v2_within_oof_predictions_primary.csv",
    "primary_thresholds": TABLES / "stage4e_corrected_v2_within_thresholds_primary.csv",
    "primary_membership": TABLES / "stage4e_corrected_v2_reconstructed_split_membership_primary.csv",
    "primary_ensemble_performance": TABLES / "stage4e_corrected_v2_ensemble_oof_performance_primary.csv",
    "primary_interpretation": TABLES / "stage4e_corrected_v2_graph_topology_interpretation_primary.csv",
}

OUTPUTS = {
    "validation": TABLES / "stage4e_corrected_v2_sensitivity_postrun_validation_tests.csv",
    "architecture_records": TABLES / "stage4e_corrected_v2_prespecified_architecture_records_sensitivities.csv",
    "seed_oof_performance": TABLES / "stage4e_corrected_v2_sensitivity_seed_level_oof_performance.csv",
    "seed_summary": TABLES / "stage4e_corrected_v2_sensitivity_seed_variability_summary.csv",
    "ensemble_predictions": TABLES / "stage4e_corrected_v2_sensitivity_ensemble_oof_predictions.csv",
    "ensemble_performance": TABLES / "stage4e_corrected_v2_sensitivity_ensemble_oof_performance.csv",
    "bootstrap_ci": TABLES / "stage4e_corrected_v2_sensitivity_bootstrap_ci.csv",
    "paired_comparisons": TABLES / "stage4e_corrected_v2_sensitivity_paired_comparisons.csv",
    "diagnostics": TABLES / "stage4e_corrected_v2_sensitivity_probability_convergence_diagnostics.csv",
    "robustness": TABLES / "stage4e_corrected_v2_architecture_robustness_summary.csv",
    "manifest": MANIFESTS / "stage4e_corrected_v2_sensitivity_postrun_manifest.json",
    "report": REPORTS / "STAGE4E_CORRECTED_V2_SENSITIVITY_POSTRUN_AUDIT.md",
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise ValueError(f"Empty required Stage 4E sensitivity file: {path}")
    return df


def add_check(rows: list[dict[str, Any]], test: str, status: str, observed: Any = "", expected: Any = "", notes: str = "") -> None:
    rows.append({"test": test, "status": status, "observed": observed, "expected": expected, "notes": notes})


def validate_sources(perf: pd.DataFrame, pred: pd.DataFrame, arch: pd.DataFrame, primary_pred: pd.DataFrame, primary_thresholds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    gcn_perf = perf[perf["model_family"].eq("gcn")]
    ridge_perf = perf[perf["model_family"].eq("ridge_early_integration")]
    add_check(rows, "gcn_fold_seed_performance_rows", "pass" if len(gcn_perf) == 800 else "fail", len(gcn_perf), 800)
    add_check(rows, "ridge_fold_performance_rows", "pass" if len(ridge_perf) == 20 else "fail", len(ridge_perf), 20)
    add_check(rows, "total_fold_performance_rows", "pass" if len(perf) == 820 else "fail", len(perf), 820)
    add_check(rows, "four_cohorts_present", "pass" if set(perf["cohort"]) == set(EXPECTED_PARTICIPANTS) else "fail", sorted(perf["cohort"].unique()), sorted(EXPECTED_PARTICIPANTS))
    add_check(rows, "two_sensitivity_architectures_present", "pass" if set(gcn_perf["architecture_id"]) == set(ARCHITECTURES) else "fail", sorted(gcn_perf["architecture_id"].unique()), ARCHITECTURES)
    add_check(rows, "four_gcn_graph_modes_present", "pass" if set(gcn_perf["graph_mode"]) == set(GRAPH_MODES) else "fail", sorted(gcn_perf["graph_mode"].unique()), GRAPH_MODES)
    add_check(rows, "five_outer_folds_present", "pass" if gcn_perf.groupby(["cohort", "architecture_id", "graph_mode"])["outer_fold"].nunique().eq(5).all() else "fail", gcn_perf.groupby(["cohort", "architecture_id", "graph_mode"])["outer_fold"].nunique().to_dict(), 5)
    add_check(rows, "five_seeds_present", "pass" if gcn_perf.groupby(["cohort", "architecture_id", "graph_mode"])["seed"].nunique().eq(5).all() else "fail", gcn_perf.groupby(["cohort", "architecture_id", "graph_mode"])["seed"].nunique().to_dict(), 5)
    counts = gcn_perf.groupby(["cohort", "architecture_id", "graph_mode"]).size()
    add_check(rows, "twenty_five_rows_per_cohort_architecture_graph_mode", "pass" if len(counts) == 32 and counts.eq(25).all() else "fail", counts.to_dict(), "32 groups all n=25")

    gcn_pred = pred[pred["model_family"].eq("gcn")].copy()
    participant_counts = pred.groupby("cohort")["sample_id"].nunique().to_dict()
    add_check(rows, "participant_counts", "pass" if participant_counts == EXPECTED_PARTICIPANTS else "fail", participant_counts, EXPECTED_PARTICIPANTS)
    coverage = gcn_pred.groupby(["cohort", "architecture_id", "graph_mode", "seed"])["sample_id"].nunique()
    expected_by_cohort = coverage.reset_index().assign(expected=lambda d: d["cohort"].map(EXPECTED_PARTICIPANTS))
    add_check(rows, "one_oof_prediction_per_participant_per_architecture_graph_seed", "pass" if expected_by_cohort["sample_id"].eq(expected_by_cohort["expected"]).all() else "fail", expected_by_cohort[~expected_by_cohort["sample_id"].eq(expected_by_cohort["expected"])].to_dict("records"), "cohort participant count")
    dup_cols = ["cohort", "model_family", "architecture_id", "graph_mode", "seed", "sample_id"]
    add_check(rows, "no_duplicate_model_sample_architecture_seed_records", "pass" if not pred.duplicated(dup_cols).any() else "fail", int(pred.duplicated(dup_cols).sum()), 0)
    add_check(rows, "finite_decision_scores", "pass" if np.isfinite(pred["decision_score"]).all() else "fail")
    add_check(rows, "finite_probabilities", "pass" if np.isfinite(pred["probability"]).all() else "fail")
    add_check(rows, "probabilities_in_unit_interval", "pass" if pred["probability"].between(0, 1).all() else "fail")
    add_check(rows, "ranking_metrics_from_decision_scores", "pass" if perf["ranking_metrics_use"].eq("continuous_logit_or_decision_score").all() else "fail")
    add_check(rows, "architecture_ids_are_prespecified_sensitivities", "pass" if set(gcn_pred["architecture_id"]) == set(ARCHITECTURES) else "fail", sorted(gcn_pred["architecture_id"].unique()), ARCHITECTURES)
    hpo_ok = arch["selection_scope"].eq("prespecified_architecture_no_hpo").all() and arch["hpo_candidate"].eq(False).all()
    add_check(rows, "no_hpo_occurred", "pass" if hpo_ok else "fail", arch[["selection_scope", "hpo_candidate"]].drop_duplicates().to_dict("records"), "prespecified_architecture_no_hpo and hpo_candidate=False")

    sets_ok = True
    for (cohort, seed), part in gcn_pred.groupby(["cohort", "seed"]):
        sets = [set(g["sample_id"].astype(str)) for _, g in part.groupby(["architecture_id", "graph_mode"])]
        sets_ok = sets_ok and bool(sets) and all(s == sets[0] for s in sets)
    add_check(rows, "identical_participants_across_graph_modes_and_architectures", "pass" if sets_ok else "fail")

    if INPUTS["primary_membership"].exists():
        membership = read_csv(INPUTS["primary_membership"])
        outer = membership[membership["split_role"].eq("outer_test")]
        ok = True
        for (cohort, seed, outer_fold), pp in gcn_pred.groupby(["cohort", "seed", "outer_fold"]):
            expected = set(outer[(outer["cohort"].eq(cohort)) & (outer["seed"].astype(str).eq(str(seed))) & (outer["outer_fold"].astype(int).eq(int(outer_fold)))]["sample_id"].astype(str))
            observed_sets = [set(g["sample_id"].astype(str)) for _, g in pp.groupby(["architecture_id", "graph_mode"])]
            ok = ok and all(s == expected for s in observed_sets)
        add_check(rows, "sensitivity_outer_test_membership_matches_reconstructed_primary", "pass" if ok else "fail", "checked cohort/seed/fold/architecture/graph_mode", "identical sample IDs")
    else:
        add_check(rows, "sensitivity_outer_test_membership_matches_reconstructed_primary", "unresolved", "primary reconstructed split membership absent", INPUTS["primary_membership"], "Cannot directly prove identical folds without reconstructed primary membership.")

    sens_ridge = pred[pred["model_family"].eq("ridge_early_integration")].copy()
    prim_ridge = primary_pred[primary_pred["model_family"].eq("ridge_early_integration")].copy()
    keys = ["cohort", "outer_fold", "sample_id"]
    merged = sens_ridge.merge(prim_ridge, on=keys, suffixes=("_sens", "_primary"), how="outer", indicator=True)
    same_rows = merged["_merge"].eq("both").all() and len(sens_ridge) == len(prim_ridge)
    numeric_equal = same_rows and np.allclose(merged["decision_score_sens"], merged["decision_score_primary"], rtol=0, atol=1e-12) and np.allclose(merged["probability_sens"], merged["probability_primary"], rtol=0, atol=1e-12)
    class_equal = same_rows and merged["predicted_class_sens"].astype(int).equals(merged["predicted_class_primary"].astype(int))
    add_check(rows, "sensitivity_ridge_predictions_equal_frozen_primary_ridge", "pass" if (same_rows and numeric_equal and class_equal) else "fail", {"rows": len(merged), "same_rows": same_rows, "numeric_equal_atol_1e_12": numeric_equal, "class_equal": class_equal}, "exact sample/fold IDs and floating-point tolerance 1e-12")

    sens_thr = read_csv(INPUTS["thresholds"])
    sens_ridge_thr = sens_thr[sens_thr["model_family"].eq("ridge_early_integration")].copy()
    prim_ridge_thr = primary_thresholds[primary_thresholds["model_family"].eq("ridge_early_integration")].copy()
    thr_merge = sens_ridge_thr.merge(prim_ridge_thr, on=["cohort", "outer_fold"], suffixes=("_sens", "_primary"), how="outer", indicator=True)
    thr_equal = thr_merge["_merge"].eq("both").all() and np.allclose(thr_merge["threshold_sens"], thr_merge["threshold_primary"], rtol=0, atol=1e-12)
    add_check(rows, "sensitivity_ridge_thresholds_equal_frozen_primary_ridge", "pass" if thr_equal else "fail", {"rows": len(thr_merge), "threshold_equal_atol_1e_12": bool(thr_equal)}, "same ridge threshold per cohort/fold")
    return pd.DataFrame(rows)


def seed_level_oof_performance(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    gcn = pred[pred["model_family"].eq("gcn")]
    for keys, group in gcn.groupby(["cohort", "disease", "task", "architecture_id", "graph_mode", "seed"], sort=True):
        cohort, disease, task, arch, graph_mode, seed = keys
        if group["sample_id"].nunique() != EXPECTED_PARTICIPANTS[cohort]:
            raise ValueError(f"Incomplete sensitivity OOF participant coverage for {keys}")
        rows.append({"cohort": cohort, "disease": disease, "task": task, "model_family": "gcn", "architecture_id": arch, "representation": graph_mode, "graph_mode": graph_mode, "seed": seed, **metrics_from_predictions(group)})
    return pd.DataFrame(rows)


def seed_variability(seed_perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in seed_perf.groupby(["cohort", "architecture_id", "representation"], sort=True):
        cohort, arch, rep = keys
        for metric in ["roc_auc", "pr_auc", "brier_score", "mcc", "sensitivity", "specificity"]:
            vals = group[metric].dropna().astype(float)
            rows.append({"cohort": cohort, "architecture_id": arch, "representation": rep, "metric": metric, "seed_n": len(vals), "mean": vals.mean(), "median": vals.median(), "sd": vals.std(ddof=1), "iqr": vals.quantile(0.75) - vals.quantile(0.25), "min": vals.min(), "max": vals.max()})
    return pd.DataFrame(rows)


def ensemble_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    gcn = pred[pred["model_family"].eq("gcn")]
    for keys, group in gcn.groupby(["cohort", "disease", "task", "architecture_id", "graph_mode", "sample_id"], sort=True):
        cohort, disease, task, arch, graph_mode, sample_id = keys
        if group["seed"].nunique() != 5:
            raise ValueError(f"Expected five seed predictions for {keys}")
        y_values = group["y_true"].unique()
        if len(y_values) != 1:
            raise ValueError(f"Conflicting y_true for {keys}")
        votes = group["predicted_class"].astype(int)
        vote_sum = int(votes.sum())
        rows.append({"cohort": cohort, "disease": disease, "task": task, "model_family": "gcn", "architecture_id": arch, "representation": graph_mode, "graph_mode": graph_mode, "sample_id": sample_id, "y_true": int(y_values[0]), "decision_score": float(group["decision_score"].mean()), "probability": float(group["probability"].mean()), "predicted_class": int(vote_sum >= 3), "class_decision_rule": "seed_majority_vote", "positive_votes": vote_sum, "negative_votes": int(5 - vote_sum), "vote_tie": False, "seed_n": 5})
    ridge = pred[pred["model_family"].eq("ridge_early_integration")].copy()
    ridge = ridge.assign(architecture_id="frozen_primary_ridge_once", representation="ridge_early_integration", class_decision_rule="fold_specific_frozen_threshold", positive_votes=np.nan, negative_votes=np.nan, vote_tie=False, seed_n=1)
    out = pd.DataFrame(rows)
    return pd.concat([out, ridge[out.columns]], ignore_index=True)


def ensemble_performance(ensemble: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in ensemble.groupby(["cohort", "disease", "task", "model_family", "architecture_id", "representation", "graph_mode"], sort=True):
        cohort, disease, task, family, arch, rep, mode = keys
        rows.append({"cohort": cohort, "disease": disease, "task": task, "model_family": family, "architecture_id": arch, "representation": rep, "graph_mode": mode, **metrics_from_predictions(group)})
    return pd.DataFrame(rows)


def bootstrap_metric_ci(ensemble: pd.DataFrame, n_bootstrap: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cohort, cdf in ensemble.groupby("cohort", sort=True):
        ids = cdf[["sample_id", "y_true"]].drop_duplicates()
        pos = ids.loc[ids["y_true"].eq(1), "sample_id"].to_numpy()
        neg = ids.loc[ids["y_true"].eq(0), "sample_id"].to_numpy()
        sampled_sets = [np.concatenate([rng.choice(pos, size=len(pos), replace=True), rng.choice(neg, size=len(neg), replace=True)]) for _ in range(n_bootstrap)]
        for (arch, rep), part in cdf.groupby(["architecture_id", "representation"], sort=True):
            by_id = part.set_index("sample_id")
            vals = {"roc_auc": [], "pr_auc": [], "brier_score": []}
            for sampled in sampled_sets:
                sub = by_id.loc[sampled].reset_index()
                metric = metrics_from_predictions(sub)
                for name in vals:
                    vals[name].append(metric[name])
            for metric_name, metric_vals in vals.items():
                arr = np.asarray(metric_vals, dtype=float)
                rows.append({"cohort": cohort, "architecture_id": arch, "representation": rep, "metric": metric_name, "estimate": float(metrics_from_predictions(part)[metric_name]), "bootstrap_n": n_bootstrap, "bootstrap_seed": seed, "ci_low": float(np.nanpercentile(arr, 2.5)), "ci_high": float(np.nanpercentile(arr, 97.5))})
    return pd.DataFrame(rows)


def paired_comparisons(ensemble: pd.DataFrame, n_bootstrap: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cohort, cdf in ensemble.groupby("cohort", sort=True):
        ids = cdf[["sample_id", "y_true"]].drop_duplicates()
        pos = ids.loc[ids["y_true"].eq(1), "sample_id"].to_numpy()
        neg = ids.loc[ids["y_true"].eq(0), "sample_id"].to_numpy()
        sampled_sets = [np.concatenate([rng.choice(pos, size=len(pos), replace=True), rng.choice(neg, size=len(neg), replace=True)]) for _ in range(n_bootstrap)]
        ridge = cdf[cdf["representation"].eq("ridge_early_integration")].set_index("sample_id")
        for arch in ARCHITECTURES:
            arch_df = cdf[(cdf["architecture_id"].eq(arch)) | (cdf["representation"].eq("ridge_early_integration"))]
            by_rep = {rep: part.set_index("sample_id") for rep, part in arch_df.groupby("representation")}
            by_rep["ridge_early_integration"] = ridge
            for model_a, model_b in COMPARISONS:
                obs = roc_auc_score(ids["y_true"], by_rep[model_a].loc[ids["sample_id"], "decision_score"]) - roc_auc_score(ids["y_true"], by_rep[model_b].loc[ids["sample_id"], "decision_score"])
                diffs = []
                for sampled in sampled_sets:
                    yy = ids.set_index("sample_id").loc[sampled, "y_true"]
                    aa = by_rep[model_a].loc[sampled, "decision_score"]
                    bb = by_rep[model_b].loc[sampled, "decision_score"]
                    diffs.append(roc_auc_score(yy, aa) - roc_auc_score(yy, bb))
                arr = np.asarray(diffs, dtype=float)
                le0 = int((arr <= 0).sum())
                ge0 = int((arr >= 0).sum())
                p = 2 * (min(le0, ge0) + 1) / (len(arr) + 1)
                lo, hi = np.nanpercentile(arr, [2.5, 97.5])
                rows.append({"cohort": cohort, "architecture_id": arch, "model_a": model_a, "model_b": model_b, "comparison": f"{model_a}_minus_{model_b}", "observed_delta_roc_auc": float(obs), "bootstrap_median_delta_roc_auc": float(np.nanmedian(arr)), "ci_low": float(lo), "ci_high": float(hi), "bootstrap_n": n_bootstrap, "bootstrap_seed": seed, "proportion_delta_le_zero": float(le0 / len(arr)), "proportion_delta_ge_zero": float(ge0 / len(arr)), "finite_resample_two_sided_p": float(min(p, 1.0))})
    out = pd.DataFrame(rows)
    if len(out) != 48:
        raise ValueError(f"Expected 48 sensitivity paired comparisons, observed {len(out)}")
    out["bh_fdr_p"] = bh_fdr(out["finite_resample_two_sided_p"])
    decisions = []
    for _, row in out.iterrows():
        if model_a_significantly_better(row):
            decisions.append("model_a_significantly_better_after_48_test_bh_fdr")
        elif model_a_significantly_worse(row):
            decisions.append("model_a_significantly_worse_after_48_test_bh_fdr")
        else:
            decisions.append("inconclusive_after_48_test_bh_fdr")
    out["direction_specific_interpretation"] = decisions
    out["comparison_family"] = "stage4e_sensitivity_48_prespecified_architecture_comparisons"
    return out


def probability_convergence_diagnostics(perf: pd.DataFrame, loss: pd.DataFrame) -> pd.DataFrame:
    final_loss = loss.sort_values("epoch").groupby(["cohort", "architecture_id", "graph_mode", "seed", "outer_fold"], dropna=False).tail(1)
    loss_summary = final_loss.groupby(["cohort", "architecture_id", "graph_mode"], dropna=False).agg(rows=("epoch", "size"), median_epochs_run=("epoch", "median"), max_epochs_run=("epoch", "max"), completed_rows=("convergence_status", lambda s: int(s.eq("completed").sum()))).reset_index()
    diag = perf.groupby(["cohort", "model_family", "architecture_id", "graph_mode"], dropna=False).agg(rows=("roc_auc", "size"), compressed_probability_rows=("compressed_probability_flag", "sum"), saturation_rows=("saturation_flag", "sum"), one_class_prediction_rows=("one_class_prediction_flag", "sum"), constant_logit_failure_rows=("constant_logit_failure", "sum")).reset_index()
    for col in ["compressed_probability_rows", "saturation_rows", "one_class_prediction_rows", "constant_logit_failure_rows"]:
        diag[col.replace("_rows", "_proportion")] = diag[col] / diag["rows"]
    return diag.merge(loss_summary, on=["cohort", "architecture_id", "graph_mode"], how="left").assign(embedding_diagnostics_available=False, oversmoothing_claim_supported=False)


def robustness_summary(ensemble_perf: pd.DataFrame, paired: pd.DataFrame | None) -> pd.DataFrame:
    primary_perf = read_csv(INPUTS["primary_ensemble_performance"])
    rows = []
    for _, row in ensemble_perf[ensemble_perf["model_family"].eq("gcn")].iterrows():
        cohort, arch, rep = row["cohort"], row["architecture_id"], row["representation"]
        primary = primary_perf[(primary_perf["cohort"].eq(cohort)) & (primary_perf["representation"].eq(rep))]
        primary_auc = float(primary["roc_auc"].iloc[0]) if not primary.empty else np.nan
        bio_better = False
        gcn_better_ridge = False
        if paired is not None and rep in {"kegg_only", "kegg_njc19", "conservative_kegg_njc19"}:
            arch_p = paired[(paired["cohort"].eq(cohort)) & (paired["architecture_id"].eq(arch))]
            bio_better = bool(arch_p[arch_p["comparison"].isin(["kegg_njc19_minus_identity", "conservative_kegg_njc19_minus_identity", "kegg_only_minus_identity"])]["direction_specific_interpretation"].eq("model_a_significantly_better_after_48_test_bh_fdr").any())
            gcn_better_ridge = bool(arch_p[arch_p["comparison"].isin(["kegg_njc19_minus_ridge_early_integration", "conservative_kegg_njc19_minus_ridge_early_integration"])]["direction_specific_interpretation"].eq("model_a_significantly_better_after_48_test_bh_fdr").any())
        supports = "supports_primary_no_biological_topology_superiority" if not bio_better and not gcn_better_ridge else "contradicts_primary_qualitative_conclusion_exploratory"
        rows.append({"cohort": cohort, "architecture_id": arch, "graph_mode": rep, "primary_ensemble_auc": primary_auc, "sensitivity_ensemble_auc": row["roc_auc"], "descriptive_auc_difference_sensitivity_minus_primary": float(row["roc_auc"] - primary_auc) if np.isfinite(primary_auc) else np.nan, "biological_topology_significantly_exceeded_identity_within_sensitivity": bio_better, "graph_gcn_significantly_exceeded_ridge_within_sensitivity": gcn_better_ridge, "qualitative_robustness_interpretation": supports, "selection_warning": "descriptive robustness only; architecture is not selected from these differences"})
    return pd.DataFrame(rows)


def write_report(validation: pd.DataFrame, ensemble_perf: pd.DataFrame, diagnostics: pd.DataFrame, bootstrap_done: bool, paired: pd.DataFrame | None) -> None:
    lines = [
        "# Stage 4E Corrected-v2 Sensitivity Post-run Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "No GCN, ridge, CV, primary model, external transfer, permutation or sensitivity model fitting was rerun during this post-processing pass.",
        "",
        "Sensitivity architectures are robustness analyses only and remain separate from the frozen 24-comparison primary family.",
        "",
        "## Validation",
        validation.to_markdown(index=False),
        "",
        "## Sensitivity Ensemble Performance",
        ensemble_perf.to_markdown(index=False),
        "",
        "## Probability and Convergence Diagnostics",
        diagnostics.to_markdown(index=False),
        "",
        "## Bootstrap Status",
        f"- Bootstrap executed: {bootstrap_done}.",
        "- Full 2,000-resample sensitivity bootstrap is available through `--run --bootstrap 2000`.",
        "",
        "## Authority",
        "- Primary Stage 4E remains the main result.",
        "- Sensitivity outputs are robustness analyses only.",
        "- Smoke outputs remain non-authoritative.",
        "- External Stage 4E remains unexecuted.",
        "- Explicit layer-wise embedding variance/similarity diagnostics were not available in saved outputs; over-smoothing is therefore not claimed.",
    ]
    if paired is not None:
        lines += ["", "## Sensitivity Paired Comparisons", paired.to_markdown(index=False)]
    OUTPUTS["report"].write_text("\n".join(lines))


def postprocess(structural_dry_run: bool = False, n_bootstrap: int = BOOTSTRAP_N) -> None:
    ensure_v2_dirs()
    perf = read_csv(INPUTS["performance"])
    pred = read_csv(INPUTS["predictions"])
    arch = read_csv(INPUTS["architecture_records_old_name"])
    loss = read_csv(INPUTS["loss"])
    primary_pred = read_csv(INPUTS["primary_predictions"])
    primary_thresholds = read_csv(INPUTS["primary_thresholds"])
    validation = validate_sources(perf, pred, arch, primary_pred, primary_thresholds)
    atomic_csv(validation, OUTPUTS["validation"])
    if validation["status"].eq("fail").any():
        raise RuntimeError("Stage 4E sensitivity post-run validation failed.")

    arch_copy = arch.copy()
    arch_copy["record_role"] = "prespecified_sensitivity_architecture_no_hpo"
    atomic_csv(arch_copy, OUTPUTS["architecture_records"])
    seed_perf = seed_level_oof_performance(pred)
    seed_summary = seed_variability(seed_perf)
    ensemble = ensemble_predictions(pred)
    ensemble_perf = ensemble_performance(ensemble)
    diagnostics = probability_convergence_diagnostics(perf, loss)
    atomic_csv(seed_perf, OUTPUTS["seed_oof_performance"])
    atomic_csv(seed_summary, OUTPUTS["seed_summary"])
    atomic_csv(ensemble, OUTPUTS["ensemble_predictions"])
    atomic_csv(ensemble_perf, OUTPUTS["ensemble_performance"])
    atomic_csv(diagnostics, OUTPUTS["diagnostics"])

    paired = None
    if not structural_dry_run:
        bootstrap_ci = bootstrap_metric_ci(ensemble, n_bootstrap=n_bootstrap)
        paired = paired_comparisons(ensemble, n_bootstrap=n_bootstrap)
        atomic_csv(bootstrap_ci, OUTPUTS["bootstrap_ci"])
        atomic_csv(paired, OUTPUTS["paired_comparisons"])
    atomic_csv(robustness_summary(ensemble_perf, paired), OUTPUTS["robustness"])

    manifest = {
        "status": "structural_dry_run_complete" if structural_dry_run else "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sensitivity_architectures": ARCHITECTURES,
        "graph_modes": GRAPH_MODES,
        "comparison_family": "separate_sensitivity_family_48_comparisons" if not structural_dry_run else "not_executed_in_structural_dry_run",
        "bootstrap_count": 0 if structural_dry_run else n_bootstrap,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "model_refitting_during_postprocessing": False,
        "external_analysis_executed": False,
        "primary_results_modified": False,
        "ridge_result_policy": "frozen_primary_ridge_used_once_after_equality_check",
        "input_hashes": {name: file_sha256(path) for name, path in INPUTS.items() if path.exists() and path.is_file()},
        "output_hashes": {name: file_sha256(path) for name, path in OUTPUTS.items() if path.exists() and path.is_file() and path.suffix != ".json"},
    }
    atomic_json(manifest, OUTPUTS["manifest"])
    write_report(validation, ensemble_perf, diagnostics, bootstrap_done=not structural_dry_run, paired=paired)
    print("STAGE4E_CORRECTED_V2_SENSITIVITY_POSTPROCESS_STRUCTURAL_DRY_RUN_COMPLETE" if structural_dry_run else "STAGE4E_CORRECTED_V2_SENSITIVITY_POSTPROCESS_COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser(description="No-refit post-processing for completed corrected-v2 Stage 4E sensitivity runs.")
    parser.add_argument("--structural-dry-run", action="store_true", help="Validate and build non-bootstrap sensitivity summaries only.")
    parser.add_argument("--run", action="store_true", help="Run full no-refit sensitivity post-processing including participant bootstrap.")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_N)
    args = parser.parse_args()
    if args.structural_dry_run:
        postprocess(structural_dry_run=True, n_bootstrap=0)
        return
    if args.run:
        postprocess(structural_dry_run=False, n_bootstrap=args.bootstrap)
        return
    raise SystemExit("Use --structural-dry-run or --run.")


if __name__ == "__main__":
    main()
