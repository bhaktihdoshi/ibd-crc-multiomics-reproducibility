from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from stage4_corrected_v2_common import (
    MANIFESTS,
    PREDICTIONS,
    RANDOM_SEED,
    REPORTS,
    TABLES,
    atomic_csv,
    ensure_v2_dirs,
)


KEY = ["disease", "task", "training_cohort", "external_cohort", "representation"]
TRANSFER_KEY = ["disease", "task", "training_cohort", "external_cohort"]
REPRESENTATIONS = ["METABOLOME", "MICROBIOME", "EARLY_INTEGRATION"]
PRIMARY_COMPARISONS = [
    ("EARLY_INTEGRATION", "METABOLOME"),
    ("EARLY_INTEGRATION", "MICROBIOME"),
]
BOOTSTRAPS = 2000
PROBABILITY_EPS = 1e-6
COMPRESSED_SD_THRESHOLD = 0.01
COMPRESSED_RANGE_THRESHOLD = 0.05


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def auc_or_nan(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or len(np.unique(p)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def paired_stratified_auc_delta(
    y: np.ndarray,
    early_prob: np.ndarray,
    comparator_prob: np.ndarray,
    n_boot: int = BOOTSTRAPS,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    early_prob = np.asarray(early_prob, dtype=float)
    comparator_prob = np.asarray(comparator_prob, dtype=float)
    early_auc = auc_or_nan(y, early_prob)
    comparator_auc = auc_or_nan(y, comparator_prob)
    observed_delta = early_auc - comparator_auc
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    deltas: list[float] = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(pos, size=len(pos), replace=True),
                rng.choice(neg, size=len(neg), replace=True),
            ]
        )
        if len(np.unique(y[idx])) < 2:
            continue
        early_auc_b = auc_or_nan(y[idx], early_prob[idx])
        comparator_auc_b = auc_or_nan(y[idx], comparator_prob[idx])
        if np.isfinite(early_auc_b) and np.isfinite(comparator_auc_b):
            deltas.append(early_auc_b - comparator_auc_b)
    if not deltas or not np.isfinite(observed_delta):
        return {
            "comparator_auc": comparator_auc,
            "early_integration_auc": early_auc,
            "delta_roc_auc": observed_delta,
            "delta_ci_low": np.nan,
            "delta_ci_high": np.nan,
            "valid_bootstrap_n": len(deltas),
            "bootstrap_two_sided_p": np.nan,
            "interpretation": "unresolved",
        }
    arr = np.asarray(deltas, dtype=float)
    if observed_delta >= 0:
        p_value = min(1.0, 2.0 * float(np.mean(arr <= 0)))
    else:
        p_value = min(1.0, 2.0 * float(np.mean(arr >= 0)))
    ci_low = float(np.percentile(arr, 2.5))
    ci_high = float(np.percentile(arr, 97.5))
    if ci_low > 0:
        interpretation = "supported"
    elif ci_high < 0:
        interpretation = "worse"
    else:
        interpretation = "inconclusive"
    return {
        "comparator_auc": comparator_auc,
        "early_integration_auc": early_auc,
        "delta_roc_auc": float(observed_delta),
        "delta_ci_low": ci_low,
        "delta_ci_high": ci_high,
        "valid_bootstrap_n": int(len(arr)),
        "bootstrap_two_sided_p": float(p_value),
        "interpretation": interpretation,
    }


def build_paired_comparisons(external_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for transfer_values, group in external_predictions.groupby(TRANSFER_KEY, dropna=False):
        transfer = dict(zip(TRANSFER_KEY, transfer_values, strict=True))
        for early_rep, comparator_rep in PRIMARY_COMPARISONS:
            early = group[group["representation"].eq(early_rep)].copy()
            comp = group[group["representation"].eq(comparator_rep)].copy()
            merged = early.merge(
                comp,
                on=["external_sample_id", "y_true"],
                suffixes=("_early", "_comparator"),
                validate="one_to_one",
            )
            if merged.empty:
                raise ValueError(f"No paired predictions for {transfer} {early_rep} vs {comparator_rep}")
            stats = paired_stratified_auc_delta(
                merged["y_true"].to_numpy(dtype=int),
                merged["probability_early"].to_numpy(dtype=float),
                merged["probability_comparator"].to_numpy(dtype=float),
                seed=RANDOM_SEED + len(rows),
            )
            rows.append(
                {
                    **transfer,
                    "primary_comparison": f"{early_rep}_minus_{comparator_rep}",
                    "comparison_type": "prespecified_paired_external_bootstrap",
                    "early_integration_representation": early_rep,
                    "comparator_representation": comparator_rep,
                    "external_n": int(len(merged)),
                    "external_positive_n": int(merged["y_true"].sum()),
                    "external_negative_n": int((1 - merged["y_true"]).sum()),
                    **stats,
                }
            )
        # Explicitly label the best-single-omic comparison as post-hoc.
        perf_by_rep = {
            rep: auc_or_nan(
                group[group["representation"].eq(rep)]["y_true"].to_numpy(dtype=int),
                group[group["representation"].eq(rep)]["probability"].to_numpy(dtype=float),
            )
            for rep in ["METABOLOME", "MICROBIOME"]
        }
        best_single = max(perf_by_rep, key=lambda rep: (-np.inf if not np.isfinite(perf_by_rep[rep]) else perf_by_rep[rep]))
        early = group[group["representation"].eq("EARLY_INTEGRATION")]
        comp = group[group["representation"].eq(best_single)]
        merged = early.merge(comp, on=["external_sample_id", "y_true"], suffixes=("_early", "_comparator"), validate="one_to_one")
        stats = paired_stratified_auc_delta(
            merged["y_true"].to_numpy(dtype=int),
            merged["probability_early"].to_numpy(dtype=float),
            merged["probability_comparator"].to_numpy(dtype=float),
            seed=RANDOM_SEED + 100 + len(rows),
        )
        rows.append(
            {
                **transfer,
                "primary_comparison": f"EARLY_INTEGRATION_minus_best_single_omic_{best_single}",
                "comparison_type": "post_hoc_best_single_omic_uses_external_outcomes",
                "early_integration_representation": "EARLY_INTEGRATION",
                "comparator_representation": best_single,
                "external_n": int(len(merged)),
                "external_positive_n": int(merged["y_true"].sum()),
                "external_negative_n": int((1 - merged["y_true"]).sum()),
                **stats,
            }
        )
    return pd.DataFrame(rows)


def build_probability_diagnostics(external_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for values, group in external_predictions.groupby(KEY, dropna=False):
        context = dict(zip(KEY, values, strict=True))
        prob = group["probability"].to_numpy(dtype=float)
        threshold = float(group["frozen_threshold"].iloc[0])
        pred = (prob >= threshold).astype(int)
        prob_range = float(np.nanmax(prob) - np.nanmin(prob))
        prob_sd = float(np.nanstd(prob))
        rows.append(
            {
                **context,
                "external_n": int(len(group)),
                "external_positive_n": int(group["y_true"].sum()),
                "frozen_threshold": threshold,
                "probability_mean": float(np.nanmean(prob)),
                "probability_min": float(np.nanmin(prob)),
                "probability_max": float(np.nanmax(prob)),
                "probability_range": prob_range,
                "probability_sd": prob_sd,
                "fraction_probability_le_1e_minus_6": float(np.mean(prob <= PROBABILITY_EPS)),
                "fraction_probability_ge_1_minus_1e_minus_6": float(np.mean(prob >= 1 - PROBABILITY_EPS)),
                "unique_probability_n": int(pd.Series(np.round(prob, 12)).nunique(dropna=True)),
                "predicted_positive_proportion": float(np.mean(pred)),
                "constant_probability_flag": bool(pd.Series(np.round(prob, 12)).nunique(dropna=True) <= 1),
                "compressed_probability_flag": bool((prob_sd < COMPRESSED_SD_THRESHOLD) or (prob_range < COMPRESSED_RANGE_THRESHOLD)),
                "saturation_flag": bool(np.any(prob <= PROBABILITY_EPS) or np.any(prob >= 1 - PROBABILITY_EPS)),
                "one_class_prediction_flag": bool(len(np.unique(pred)) <= 1),
                "diagnostic_rule": (
                    f"descriptive only: compressed if SD<{COMPRESSED_SD_THRESHOLD} "
                    f"or range<{COMPRESSED_RANGE_THRESHOLD}; saturated if p<=1e-6 or p>=1-1e-6"
                ),
            }
        )
    return pd.DataFrame(rows)


def structural_audit(
    external_perf: pd.DataFrame,
    thresholds: pd.DataFrame,
    external_predictions: pd.DataFrame,
    threshold_predictions: pd.DataFrame,
    manifest_path: Path,
) -> pd.DataFrame:
    expected_tasks = {"IBD_vs_Control", "CRC_vs_Control"}
    expected_reps = set(REPRESENTATIONS)
    expected_transfers = {
        ("IBD", "IBD_vs_Control", "Franzosa", "iHMP"),
        ("IBD", "IBD_vs_Control", "iHMP", "Franzosa"),
        ("CRC", "CRC_vs_Control", "Sinha", "Yachida"),
        ("CRC", "CRC_vs_Control", "Yachida", "Sinha"),
    }
    rows = []

    def add(check: str, passed: bool, detail: str) -> None:
        rows.append({"check": check, "status": "PASS" if passed else "FAIL", "detail": detail})

    transfer_set = set(external_perf[TRANSFER_KEY].drop_duplicates().itertuples(index=False, name=None))
    add("external_performance_shape", external_perf.shape == (12, 28), f"observed_shape={external_perf.shape}")
    add("tasks_pooled_only", set(external_perf["task"]) == expected_tasks, f"tasks={sorted(external_perf['task'].unique())}")
    add("four_transfer_directions", transfer_set == expected_transfers, f"transfers={sorted(transfer_set)}")
    add("three_representations_each", external_perf.groupby(TRANSFER_KEY)["representation"].nunique().eq(3).all(), "representation count by transfer direction checked")
    add("representations_expected", set(external_perf["representation"]) == expected_reps, f"representations={sorted(external_perf['representation'].unique())}")
    add("bootstrap_ci_complete", external_perf[["roc_auc_ci_low", "roc_auc_ci_high", "pr_auc_ci_low", "pr_auc_ci_high"]].notna().all().all(), "ROC-AUC and PR-AUC CI columns non-missing")
    add("bootstrap_count_2000", external_perf[["roc_auc_bootstrap_n", "pr_auc_bootstrap_n"]].eq(2000).all().all(), "all bootstrap counts should be 2000")
    add("threshold_rows", len(thresholds) == 12, f"threshold_rows={len(thresholds)}")
    add("threshold_source_training_oof", thresholds["threshold_source"].eq("full_source_cross_fitted_oof").all(), f"sources={thresholds['threshold_source'].unique().tolist()}")
    add("external_not_used_for_threshold", thresholds["external_data_used_for_threshold"].eq(False).all(), "external_data_used_for_threshold all False")
    dup = threshold_predictions.duplicated(KEY + ["sample_id"]).sum()
    add("threshold_oof_unique_per_source_sample", dup == 0, f"duplicate context/sample rows={int(dup)}")
    add("external_predictions_have_probabilities", external_predictions["probability"].notna().all(), "all external probability fields non-missing")
    add("run_manifest_exists", manifest_path.exists(), str(manifest_path))
    return pd.DataFrame(rows)


def reporting_internal_performance(internal_perf: pd.DataFrame) -> pd.DataFrame:
    out = internal_perf.copy()
    if "external_positive_prevalence" in out.columns:
        out = out.rename(columns={"external_positive_prevalence": "evaluation_positive_prevalence"})
    return out


def markdown_report(
    audit: pd.DataFrame,
    external_perf: pd.DataFrame,
    paired: pd.DataFrame,
    diagnostics: pd.DataFrame,
    manifest_path: Path,
) -> str:
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    failures = audit[audit["status"].ne("PASS")]
    one_class = diagnostics[diagnostics["one_class_prediction_flag"]].copy()
    compressed = diagnostics[diagnostics["compressed_probability_flag"]].copy()
    saturated = diagnostics[diagnostics["saturation_flag"]].copy()
    lines = [
        "# Stage 4B Corrected-v2 Completed-Run Audit",
        "",
        "This report audits the completed corrected-v2 Stage 4B outputs without refitting models or changing frozen predictions.",
        "",
        "## Structural Integrity",
        "",
        f"Overall status: {'PASS' if failures.empty else 'FAIL'}",
        "",
        audit.to_markdown(index=False),
        "",
        "## External Performance",
        "",
        external_perf[
            [
                "disease",
                "task",
                "training_cohort",
                "external_cohort",
                "representation",
                "roc_auc",
                "roc_auc_ci_low",
                "roc_auc_ci_high",
                "pr_auc",
                "pr_auc_ci_low",
                "pr_auc_ci_high",
                "sensitivity",
                "specificity",
                "precision",
                "mcc",
            ]
        ].to_markdown(index=False),
        "",
        "## Paired External Model Comparisons",
        "",
        "Primary comparisons are prespecified paired stratified-bootstrap differences: early integration minus metabolome, and early integration minus microbiome. The same external participants are resampled for both models. Best-single-omic rows are post-hoc and should not be presented as prespecified evidence.",
        "",
        paired.to_markdown(index=False),
        "",
        "## Probability Diagnostics",
        "",
        "Compression and saturation flags are descriptive diagnostic rules, not inferential thresholds. ROC-AUC can remain high when ranking is preserved even if transferred probabilities are compressed, saturated, poorly calibrated, or produce one-class predictions under the frozen training-derived threshold.",
        "",
        diagnostics[
            [
                "disease",
                "task",
                "training_cohort",
                "external_cohort",
                "representation",
                "probability_min",
                "probability_max",
                "probability_range",
                "probability_sd",
                "predicted_positive_proportion",
                "constant_probability_flag",
                "compressed_probability_flag",
                "saturation_flag",
                "one_class_prediction_flag",
            ]
        ].to_markdown(index=False),
        "",
        "## One-Class External Predictions",
        "",
        one_class.to_markdown(index=False) if not one_class.empty else "No one-class external predictions detected.",
        "",
        "## Differences From Historical Stage 4B",
        "",
        "- Corrected-v2 uses only pooled IBD-vs-control and CRC-vs-control predictive tasks.",
        "- CD and UC remain descriptive subclasses only; subtype-specific predictive tasks are stale for Stage 4 corrected-v2.",
        "- Corrected-v2 evaluates metabolome, microbiome, and early-integration representations across four transfer directions.",
        "- Frozen training-derived Youden thresholds are retained for external evaluation; external outcomes are not used for threshold selection.",
        "",
        "## Stale Thesis Figures And Claims",
        "",
        "- Any figure claiming CD-vs-control or UC-vs-control Stage 4 corrected-v2 model performance is stale.",
        "- Any Stage 4 figure comparing separate old IBD subtype GCN/MOFA/sCCA outputs should be replaced or labelled historical.",
        "- Any statement that early integration is superior based on overlap of separate CIs is stale; use the paired comparison table.",
        "- Any statement that high ROC-AUC implies calibrated or clinically usable probabilities should be qualified using the probability diagnostics.",
        "",
        "## Manifest",
        "",
        f"`{manifest_path}`",
        "",
        "```json",
        json.dumps(manifest, indent=2, default=str),
        "```",
    ]
    return "\n".join(lines) + "\n"


def build_stale_dependency_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "artifact_or_claim": "CD_vs_Control / UC_vs_Control Stage 4 predictive figures",
                "status": "stale_for_corrected_v2",
                "reason": "Corrected-v2 predictive scope is pooled IBD vs control and CRC vs control only.",
                "replacement": "Use corrected-v2 pooled IBD and CRC Stage 4B outputs.",
            },
            {
                "artifact_or_claim": "Clinical-variable Stage 4 multi-omics views or graph nodes",
                "status": "exclude_from_corrected_v2",
                "reason": "Clinical variables were not sufficiently complete and harmonised for fair common cross-cohort prediction.",
                "replacement": "Use molecular views only; clinical variables may be descriptive context.",
            },
            {
                "artifact_or_claim": "NJC19 positive association wording",
                "status": "stale_wording",
                "reason": "NJC19 edges are curated production, consumption, or transport relations, not disease-derived positive statistical associations.",
                "replacement": "Use unsigned undirected curated relation wording and preserve relation_type/provenance.",
            },
            {
                "artifact_or_claim": "Separate-CI overlap as model comparison evidence",
                "status": "methodologically_stale",
                "reason": "Model comparisons require paired external resampling, not overlap of independent confidence intervals.",
                "replacement": "Use stage4b_corrected_v2_paired_model_comparisons.csv.",
            },
        ]
    )


def main() -> None:
    ensure_v2_dirs()
    external_perf = read_required_csv(TABLES / "stage4b_corrected_v2_external_performance.csv")
    internal_perf = read_required_csv(TABLES / "stage4b_corrected_v2_internal_performance.csv")
    thresholds = read_required_csv(TABLES / "stage4b_corrected_v2_thresholds.csv")
    external_predictions = read_required_csv(PREDICTIONS / "stage4b_corrected_v2_external_predictions.csv")
    threshold_predictions = read_required_csv(PREDICTIONS / "stage4b_corrected_v2_source_oof_threshold_predictions.csv")
    manifest_path = MANIFESTS / "stage4b_corrected_v2_run_manifest.json"

    audit = structural_audit(external_perf, thresholds, external_predictions, threshold_predictions, manifest_path)
    paired = build_paired_comparisons(external_predictions)
    diagnostics = build_probability_diagnostics(external_predictions)
    internal_reporting = reporting_internal_performance(internal_perf)
    stale_map = build_stale_dependency_map()

    atomic_csv(audit, TABLES / "stage4b_corrected_v2_completed_run_structural_audit.csv")
    atomic_csv(paired, TABLES / "stage4b_corrected_v2_paired_model_comparisons.csv")
    atomic_csv(diagnostics, TABLES / "stage4b_corrected_v2_probability_diagnostics.csv")
    atomic_csv(internal_reporting, TABLES / "stage4b_corrected_v2_internal_performance_for_reporting.csv")
    atomic_csv(stale_map, TABLES / "stage4b_corrected_v2_stale_output_dependency_map.csv")
    (REPORTS / "STAGE4B_CORRECTED_V2_COMPLETED_RUN_AUDIT.md").write_text(
        markdown_report(audit, external_perf, paired, diagnostics, manifest_path)
    )
    print("STAGE4B_CORRECTED_V2_POSTRUN_AUDIT_COMPLETE")


if __name__ == "__main__":
    main()
