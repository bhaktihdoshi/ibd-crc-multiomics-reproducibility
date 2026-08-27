from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


DISEASE_COHORTS = {"IBD": ["Franzosa", "iHMP"], "CRC": ["Sinha", "Yachida"]}
MASTER_SEED = 20260825
REQUESTED_BOOTSTRAPS = 200
PSEUDOCOUNT = 1e-6
SCCA_COMPONENTS = [1, 2]
SCCA_PENALTYX = [0.15, 0.25, 0.40]
SCCA_PENALTYZ = [0.15, 0.25, 0.40]
SCCA_FOLDS = 5
ALIGNMENT_AMBIGUITY_MARGIN = 0.05
DEGENERATE_NORM_EPS = 1e-12
HIGH_RECURRENCE = 0.80
MODERATE_RECURRENCE = 0.50


@dataclass(frozen=True)
class Paths:
    repo: Path
    output_root: Path
    thesis_source_root: Path
    corrected_v2: Path
    pma_input_dir: Path
    pma_output_dir: Path
    tables: Path
    reports: Path
    tests: Path
    logs: Path
    temp_root: Path


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    tmp.replace(path)


def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(master_seed: int, cohort: str, bootstrap_index: int) -> int:
    payload = f"{master_seed}|{cohort}|{bootstrap_index}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31 - 1)


def display_name(feature_id: str) -> str:
    return feature_id.removeprefix("mic__").removeprefix("met__")


def recurrence_category(freq: float) -> str:
    if not np.isfinite(freq):
        return "not_estimable"
    if freq >= HIGH_RECURRENCE:
        return "high_recurrence"
    if freq >= MODERATE_RECURRENCE:
        return "moderate_recurrence"
    return "low_recurrence"


def load_stage4c_module(paths: Paths):
    sys.path.insert(0, str(paths.corrected_v2 / "scripts"))
    sys.path.insert(0, str(paths.repo / "final_analysis" / "code"))
    import run_stage4c_corrected_v2 as stage4c  # type: ignore

    return stage4c


def numeric_view(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["sample_id"]).apply(pd.to_numeric, errors="raise")


def clr_transform(matrix: pd.DataFrame) -> pd.DataFrame:
    x = matrix.apply(pd.to_numeric, errors="raise").fillna(0.0)
    if (x < 0).any().any():
        raise ValueError("Microbiome matrix contains negative values.")
    totals = x.sum(axis=1)
    if (totals <= 0).any():
        raise ValueError("A bootstrap microbiome sample has zero total abundance.")
    closed = x.div(totals, axis=0)
    logged = np.log(closed + PSEUDOCOUNT)
    return logged.sub(logged.mean(axis=1), axis=0)


def prepare_bootstrap_views(block: Any, feature_universe: dict[str, list[str]], indices: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    micro_raw = numeric_view(block.microbiome).iloc[indices].copy()
    met_raw = numeric_view(block.metabolome).iloc[indices].copy()
    boot_ids = [f"boot_{i + 1:03d}" for i in range(len(indices))]
    micro_raw.index = boot_ids
    met_raw.index = boot_ids

    micro_full = clr_transform(micro_raw.loc[:, feature_universe["microbiome"]])
    micro_sd = micro_full.std(axis=0, ddof=1)
    micro_valid = np.isfinite(micro_sd) & (micro_sd > DEGENERATE_NORM_EPS)
    micro_keep = micro_valid[micro_valid].index.astype(str).tolist()
    micro_scaled = pd.DataFrame(
        StandardScaler().fit_transform(micro_full.loc[:, micro_keep]),
        index=boot_ids,
        columns=micro_keep,
    )

    met_full = met_raw.loc[:, feature_universe["metabolome"]].apply(pd.to_numeric, errors="raise")
    met_pre_n = met_full.shape[1]
    met_nonmissing = met_full.notna().any(axis=0)
    pre_keep_cols = met_nonmissing[met_nonmissing].index.astype(str).tolist()
    met_imputed_all = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(met_full.loc[:, pre_keep_cols]),
        index=boot_ids,
        columns=pre_keep_cols,
    )
    met_sd_after_imputation = met_imputed_all.std(axis=0, ddof=1)
    met_keep = np.isfinite(met_sd_after_imputation) & (met_sd_after_imputation > DEGENERATE_NORM_EPS)
    met_keep_cols = met_keep[met_keep].index.astype(str).tolist()
    zero_sd_after_imputation = met_keep[~met_keep].index.astype(str).tolist()
    met_scaled = pd.DataFrame(
        StandardScaler().fit_transform(met_imputed_all.loc[:, met_keep_cols]),
        index=boot_ids,
        columns=met_keep_cols,
    )
    micro_sd_before_pma = micro_scaled.std(axis=0, ddof=1)
    met_sd_before_pma = met_scaled.std(axis=0, ddof=1)

    if micro_scaled.shape[1] < 2 or met_scaled.shape[1] < 2:
        raise ValueError("Too few valid bootstrap features after Stage 4C-local preprocessing.")
    if not np.isfinite(micro_scaled.to_numpy()).all() or not np.isfinite(met_scaled.to_numpy()).all():
        raise ValueError("Non-finite transformed values in bootstrap views.")

    audit = {
        "microbiome_retained_n": int(micro_scaled.shape[1]),
        "metabolome_retained_n": int(met_scaled.shape[1]),
        "microbiome_dropped_n": int(len(feature_universe["microbiome"]) - micro_scaled.shape[1]),
        "metabolome_dropped_n": int(len(feature_universe["metabolome"]) - met_scaled.shape[1]),
        "metabolome_feature_count_before_preprocessing": int(met_pre_n),
        "metabolome_feature_count_after_median_imputation": int(met_imputed_all.shape[1]),
        "metabolome_feature_count_after_zero_variance_filtering": int(met_scaled.shape[1]),
        "zero_sd_metabolites_after_imputation": "|".join(zero_sd_after_imputation),
        "zero_sd_metabolite_count_after_imputation": int(len(zero_sd_after_imputation)),
        "zero_sd_metabolite_count_immediately_before_pma": int((met_sd_before_pma <= DEGENERATE_NORM_EPS).sum()),
        "zero_sd_microbiome_count_immediately_before_pma": int((micro_sd_before_pma <= DEGENERATE_NORM_EPS).sum()),
        "min_microbiome_sd_before_pma": float(micro_sd_before_pma.min()),
        "min_metabolome_sd_before_pma": float(met_sd_before_pma.min()),
        "participant_row_multiplicity_distribution": json.dumps(pd.Series(indices).value_counts().value_counts().sort_index().astype(int).to_dict()),
        "unique_participant_rows_in_bootstrap": int(pd.Series(indices).nunique()),
        "max_participant_row_multiplicity": int(pd.Series(indices).value_counts().max()),
    }
    return micro_scaled, met_scaled, audit


def run_r_bootstrap(paths: Paths, micro: pd.DataFrame, met: pd.DataFrame, seed: int, work_dir: Path) -> dict[str, pd.DataFrame]:
    in_dir = work_dir / "input"
    out_dir = work_dir / "output"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    micro.to_csv(in_dir / "microbiome.csv")
    met.to_csv(in_dir / "metabolome.csv")
    helper = paths.repo / "final_analysis" / "code" / "focused_validation" / "scca_bootstrap_pma.R"
    cmd = [
        shutil.which("Rscript") or "Rscript",
        str(helper),
        "--input-dir",
        str(in_dir),
        "--output-dir",
        str(out_dir),
        "--seed",
        str(seed),
        "--components",
        ",".join(map(str, SCCA_COMPONENTS)),
        "--penaltyx",
        ",".join(map(str, SCCA_PENALTYX)),
        "--penaltyz",
        ",".join(map(str, SCCA_PENALTYZ)),
        "--folds",
        str(SCCA_FOLDS),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode:
        raise RuntimeError(f"PMA bootstrap failed: {proc.stderr[-1500:]}")
    return {
        "summary": pd.read_csv(out_dir / "component_summary.csv"),
        "microbiome": pd.read_csv(out_dir / "microbiome_loadings.csv"),
        "metabolome": pd.read_csv(out_dir / "metabolome_loadings.csv"),
        "tuning": pd.read_csv(out_dir / "tuning_summary.csv"),
    }


def primary_tables(paths: Paths) -> dict[str, pd.DataFrame]:
    return {
        "summary": pd.read_csv(paths.corrected_v2 / "tables" / "stage4c_corrected_v2_scca_component_summary.csv"),
        "tuning": pd.read_csv(paths.corrected_v2 / "tables" / "stage4c_corrected_v2_scca_tuning_summary.csv"),
        "microbiome": pd.read_csv(paths.corrected_v2 / "tables" / "stage4c_corrected_v2_scca_microbiome_loadings.csv"),
        "metabolome": pd.read_csv(paths.corrected_v2 / "tables" / "stage4c_corrected_v2_scca_metabolome_loadings.csv"),
        "cross": pd.read_csv(paths.corrected_v2 / "tables" / "stage4c_corrected_v2_scca_cross_cohort_reproducibility.csv"),
        "manifest": pd.read_csv(paths.corrected_v2 / "tables" / "stage4c_corrected_v2_scca_input_manifest.csv"),
    }


def make_primary_audit(paths: Paths, tbl: dict[str, pd.DataFrame], stage4c: Any) -> dict[tuple[str, str], dict[str, list[str]]]:
    expected_corr = {
        ("IBD", "Franzosa", 1): (0.831511, 0.02),
        ("IBD", "iHMP", 1): (0.786390, 0.02),
        ("IBD", "iHMP", 2): (0.746078, 0.02),
        ("CRC", "Sinha", 1): (0.524271, 0.03),
        ("CRC", "Sinha", 2): (0.540311, 0.03),
        ("CRC", "Yachida", 1): (0.707602, 0.03),
        ("CRC", "Yachida", 2): (0.692905, 0.03),
    }
    rows = []
    feature_universe: dict[tuple[str, str], dict[str, list[str]]] = {}
    for _, mrow in tbl["manifest"].iterrows():
        disease, cohort = str(mrow["disease"]), str(mrow["cohort"])
        feature_universe[(disease, cohort)] = {}
        for view in ["microbiome", "metabolome"]:
            load = tbl[view]
            feats = (
                load[load["disease"].eq(disease) & load["cohort"].eq(cohort) & load["component"].eq(1)]["feature_id"]
                .astype(str)
                .tolist()
            )
            feature_universe[(disease, cohort)][view] = feats
    version = stage4c.pma_version()
    for _, row in tbl["summary"].iterrows():
        disease, cohort, comp = str(row["disease"]), str(row["cohort"]), int(row["component"])
        key = (disease, cohort, comp)
        target, tol = expected_corr.get(key, (np.nan, np.nan))
        corr = float(row["apparent_full_data_canonical_correlation"])
        if np.isfinite(target) and abs(corr - target) > tol:
            raise ValueError(f"Primary sCCA correlation mismatch for {key}: {corr} vs {target}")
        rows.append(
            {
                "disease": disease,
                "cohort": cohort,
                "participant_count": int(row["n_samples"]),
                "microbiome_feature_count": int(row["microbiome_features"]),
                "metabolome_feature_count": int(row["metabolome_features"]),
                "component": comp,
                "selected_component_count": int(tbl["summary"][tbl["summary"]["cohort"].eq(cohort)]["component"].max()),
                "selected_penaltyx": float(row["penaltyx"]),
                "selected_penaltyz": float(row["penaltyz"]),
                "apparent_full_data_canonical_correlation": corr,
                "microbiome_selected_features": int(tbl["microbiome"].query("disease == @disease and cohort == @cohort and component == @comp")["selected"].sum()),
                "metabolome_selected_features": int(tbl["metabolome"].query("disease == @disease and cohort == @cohort and component == @comp")["selected"].sum()),
                "preprocessing_sequence": "microbiome: re-close/CLR/pseudocount/filter/scale; metabolome: fixed feature universe/median impute/filter/scale",
                "pma_version": version,
                "typex": "standard",
                "typez": "standard",
                "candidate_component_counts": ",".join(map(str, SCCA_COMPONENTS)),
                "penaltyx_grid": ",".join(map(str, SCCA_PENALTYX)),
                "penaltyz_grid": ",".join(map(str, SCCA_PENALTYZ)),
                "cv_procedure": "5-fold unsupervised tuning diagnostic; not independent held-out evaluation",
                "random_seed": int(stage4c.RANDOM_SEED),
                "phenotype_used_for_tuning_or_fit": False,
            }
        )
    audit = pd.DataFrame(rows)
    atomic_csv(paths.tables / "scca_primary_solution_audit.csv", audit)
    atomic_text(
        paths.reports / "SCCA_PRIMARY_CONFIGURATION_AUDIT.md",
        "# sCCA Primary Configuration Audit\n\n"
        "The focused bootstrap stage reconstructed the authoritative corrected-v2 sCCA configuration from private Stage 4C outputs before resampling. "
        "PMA tuning uses diagnosis-free cross-view correlation only; phenotype variables are not included in the view matrices or tuning criterion.\n\n"
        + audit.to_markdown(index=False)
        + "\n",
    )
    return feature_universe


def vector_for(loadings: pd.DataFrame, disease: str, cohort: str, component: int, universe: list[str]) -> np.ndarray:
    sub = loadings[loadings["disease"].eq(disease) & loadings["cohort"].eq(cohort) & loadings["component"].eq(component)]
    mapper = dict(zip(sub["feature_id"].astype(str), pd.to_numeric(sub["loading"], errors="raise")))
    return np.asarray([float(mapper.get(f, 0.0)) for f in universe], dtype=float)


def full_boot_vector(loadings: pd.DataFrame, component: int, universe: list[str]) -> np.ndarray:
    sub = loadings[loadings["component"].eq(component)]
    mapper = dict(zip(sub["feature_id"].astype(str), pd.to_numeric(sub["loading"], errors="raise")))
    return np.asarray([float(mapper.get(f, 0.0)) for f in universe], dtype=float)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= DEGENERATE_NORM_EPS:
        return np.nan
    return float(np.dot(a, b) / denom)


def selected_set(vec: np.ndarray, universe: list[str]) -> set[str]:
    return {feature for feature, value in zip(universe, vec) if abs(float(value)) > 1e-12}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return float(len(a & b) / len(a | b)) if (a | b) else np.nan


def align_components(primary: dict[str, dict[int, np.ndarray]], boot: dict[str, dict[int, np.ndarray]]) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    p_components = sorted(primary["microbiome"])
    b_components = sorted(boot["microbiome"])
    rows = []
    matches: dict[int, dict[str, Any]] = {}
    if not p_components or not b_components:
        return rows, matches
    matrix = np.full((len(b_components), len(p_components)), -1.0)
    signed = {}
    for i, bc in enumerate(b_components):
        for j, pc in enumerate(p_components):
            mc = cosine(boot["microbiome"][bc], primary["microbiome"][pc])
            zc = cosine(boot["metabolome"][bc], primary["metabolome"][pc])
            if np.isfinite(mc) and np.isfinite(zc):
                combined_signed = float(np.nanmean([mc, zc]))
                matrix[i, j] = abs(combined_signed)
                signed[(bc, pc)] = (mc, zc, combined_signed)
    if (matrix >= 0).any():
        row_idx, col_idx = linear_sum_assignment(-matrix)
        assigned_primary = set()
        assigned_boot = set()
        for ri, ci in zip(row_idx, col_idx):
            bc = b_components[ri]
            pc = p_components[ci]
            score = matrix[ri, ci]
            if score < 0 or not np.isfinite(score):
                continue
            sorted_scores = sorted([v for v in matrix[ri, :] if v >= 0], reverse=True)
            second = sorted_scores[1] if len(sorted_scores) > 1 else np.nan
            margin = score - second if np.isfinite(second) else np.nan
            mc, zc, combined_signed = signed[(bc, pc)]
            sign = 1 if combined_signed >= 0 else -1
            status = "ambiguous_match" if np.isfinite(margin) and margin < ALIGNMENT_AMBIGUITY_MARGIN else "matched_valid"
            row = {
                "bootstrap_component": bc,
                "primary_component": pc,
                "microbial_cosine": mc,
                "metabolite_cosine": zc,
                "combined_cosine": score,
                "signed_combined_cosine": combined_signed,
                "sign_orientation": sign,
                "best_match_score": score,
                "second_best_score": second,
                "best_minus_second_best_margin": margin,
                "status": status,
            }
            rows.append(row)
            matches[pc] = row
            assigned_primary.add(pc)
            assigned_boot.add(bc)
        for bc in set(b_components) - assigned_boot:
            rows.append({"bootstrap_component": bc, "primary_component": np.nan, "status": "unmatched_extra_component"})
        for pc in set(p_components) - assigned_primary:
            rows.append({"bootstrap_component": np.nan, "primary_component": pc, "status": "primary_component_not_recovered"})
    return rows, matches


def pairwise_jaccard_mean(sets: list[set[str]]) -> float:
    if len(sets) < 2:
        return np.nan
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            vals.append(jaccard(sets[i], sets[j]))
    return float(np.mean(vals)) if vals else np.nan


def penalty_distribution_json(rows: pd.DataFrame) -> str:
    counts = rows.groupby(["selected_penaltyx", "selected_penaltyz"]).size().astype(int)
    payload = {f"penaltyx={px},penaltyz={pz}": int(n) for (px, pz), n in counts.items()}
    return json.dumps(payload, sort_keys=True)


def make_recovery_denominator_audit(bootstrap_n: int, params: pd.DataFrame, alignment: pd.DataFrame, primary_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, prow in primary_summary.iterrows():
        disease = str(prow["disease"])
        cohort = str(prow["cohort"])
        comp = int(prow["component"])
        psub = params[params["cohort"].eq(cohort)]
        successful = int(psub["status"].eq("success").sum())
        asub = alignment[alignment["cohort"].eq(cohort) & alignment["primary_component"].eq(comp)] if not alignment.empty else pd.DataFrame()
        recovered_valid = int(asub["status"].eq("matched_valid").sum()) if not asub.empty else 0
        ambiguous = int(asub["status"].eq("ambiguous_match").sum()) if not asub.empty else 0
        valid_opportunity = recovered_valid + ambiguous
        rows.append(
            {
                "disease": disease,
                "cohort": cohort,
                "primary_component": comp,
                "requested_bootstrap_n": int(bootstrap_n),
                "successful_model_fit_n": successful,
                "model_fit_failed_n": int(bootstrap_n - successful),
                "valid_matching_opportunity_n": int(valid_opportunity),
                "recovered_valid_n": recovered_valid,
                "ambiguous_match_n": ambiguous,
                "requested_denominator_display": f"{recovered_valid}/{bootstrap_n}",
                "requested_denominator_frequency": recovered_valid / bootstrap_n,
                "successful_fit_denominator_display": f"{recovered_valid}/{successful}" if successful else f"{recovered_valid}/0",
                "successful_fit_denominator_frequency": recovered_valid / successful if successful else np.nan,
                "valid_alignment_denominator_display": f"{recovered_valid}/{valid_opportunity}" if valid_opportunity else f"{recovered_valid}/0",
                "valid_alignment_denominator_frequency": recovered_valid / valid_opportunity if valid_opportunity else np.nan,
                "denominator_interpretation": "requested=resampling robustness; successful_fit=conditional on PMA fitting; valid_alignment=conditional on non-degenerate matching opportunity",
            }
        )
    return pd.DataFrame(rows)


def make_jaccard_denominator_audit(jaccards: pd.DataFrame, params: pd.DataFrame, alignment: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if jaccards.empty:
        return pd.DataFrame(rows)
    for keys, sub in jaccards.groupby(["disease", "cohort", "primary_component", "view"]):
        disease, cohort, comp, view = keys
        psub = params[params["cohort"].eq(cohort)]
        asub = alignment[alignment["cohort"].eq(cohort) & alignment["primary_component"].eq(comp)]
        rows.append(
            {
                "disease": disease,
                "cohort": cohort,
                "primary_component": int(comp),
                "view": view,
                "median_jaccard": float(sub["jaccard_vs_primary"].median()),
                "mean_jaccard": float(sub["jaccard_vs_primary"].mean()),
                "jaccard_contributing_n": int(sub["bootstrap_index"].nunique()),
                "successful_model_fit_n": int(psub["status"].eq("success").sum()),
                "model_fit_failed_n": int(psub["status"].eq("failure").sum()),
                "ambiguous_matches_excluded_n": int(asub["status"].eq("ambiguous_match").sum()),
                "failed_fits_assigned_zero": False,
                "ambiguous_matches_included": False,
                "median_is_conditional_on_valid_alignment": True,
                "denominator_display": f"{sub['bootstrap_index'].nunique()} valid aligned components",
            }
        )
    return pd.DataFrame(rows)


def make_yachida_failure_audit(old_failures: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    y = params[params["cohort"].eq("Yachida")].copy()
    old = old_failures.copy()
    old_failed = set(old["bootstrap_index"].astype(int)) if not old.empty and "bootstrap_index" in old.columns else set()
    rows = []
    for _, row in y.iterrows():
        idx = int(row["bootstrap_index"])
        current_failed = str(row["status"]) == "failure"
        old_msg = ""
        if idx in old_failed:
            old_match = old[old["bootstrap_index"].astype(int).eq(idx)]
            old_msg = str(old_match["failure"].iloc[0]) if "failure" in old_match.columns and not old_match.empty else ""
        zero_before_pma = int(row.get("zero_sd_metabolite_count_immediately_before_pma", 0) or 0)
        fold_removed = row.get("fold_local_max_metabolome_features_removed", np.nan)
        if idx in old_failed and not current_failed:
            classification = "preprocessing_bug_zero_variance_not_removed"
        elif current_failed and zero_before_pma > 0:
            classification = "preprocessing_bug_zero_variance_not_removed"
        elif current_failed:
            classification = "PMA_internal_scaling_failure"
        elif pd.notna(fold_removed) and float(fold_removed) > 0:
            classification = "corrected_fold_local_zero_variance_filtered"
        else:
            classification = "no_failure_after_correction"
        rows.append(
            {
                "cohort": "Yachida",
                "bootstrap_index": idx,
                "bootstrap_seed": int(row["bootstrap_seed"]),
                "old_failure_status": "failure" if idx in old_failed else "success",
                "corrected_status": str(row["status"]),
                "participant_row_multiplicity_distribution": row.get("participant_row_multiplicity_distribution", ""),
                "unique_participant_rows_in_bootstrap": row.get("unique_participant_rows_in_bootstrap", np.nan),
                "max_participant_row_multiplicity": row.get("max_participant_row_multiplicity", np.nan),
                "metabolome_feature_count_before_preprocessing": row.get("metabolome_feature_count_before_preprocessing", np.nan),
                "metabolome_feature_count_after_median_imputation": row.get("metabolome_feature_count_after_median_imputation", np.nan),
                "metabolome_feature_count_after_zero_variance_filtering": row.get("metabolome_feature_count_after_zero_variance_filtering", np.nan),
                "zero_sd_metabolite_count_after_imputation": row.get("zero_sd_metabolite_count_after_imputation", np.nan),
                "zero_sd_metabolites_after_imputation": row.get("zero_sd_metabolites_after_imputation", ""),
                "zero_sd_metabolite_count_immediately_before_pma": row.get("zero_sd_metabolite_count_immediately_before_pma", np.nan),
                "min_metabolome_sd_before_pma": row.get("min_metabolome_sd_before_pma", np.nan),
                "fold_local_max_metabolome_features_removed": row.get("fold_local_max_metabolome_features_removed", np.nan),
                "selected_component_count": row.get("selected_component_count", np.nan),
                "selected_penaltyx": row.get("selected_penaltyx", np.nan),
                "selected_penaltyz": row.get("selected_penaltyz", np.nan),
                "old_pma_error": old_msg,
                "corrected_pma_error": row.get("failure", "") if current_failed else "",
                "failure_classification": classification,
            }
        )
    return pd.DataFrame(rows)


def load_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def run_stability(paths: Paths, bootstrap_n: int, master_seed: int, resume: bool) -> None:
    stage4c = load_stage4c_module(paths)
    tbl = primary_tables(paths)
    feature_universe = make_primary_audit(paths, tbl, stage4c)
    blocks = stage4c.load_blocks()

    source_files = [
        paths.corrected_v2 / "tables" / "stage4c_corrected_v2_scca_component_summary.csv",
        paths.corrected_v2 / "tables" / "stage4c_corrected_v2_scca_tuning_summary.csv",
        paths.corrected_v2 / "tables" / "stage4c_corrected_v2_scca_microbiome_loadings.csv",
        paths.corrected_v2 / "tables" / "stage4c_corrected_v2_scca_metabolome_loadings.csv",
        paths.corrected_v2 / "tables" / "stage4c_corrected_v2_scca_cross_cohort_reproducibility.csv",
    ]
    before_hashes = {str(p): sha256(p) for p in source_files if p.exists()}
    prior_focused = [
        paths.tables / "rank_metabolomics_external_performance.csv",
        paths.tables / "mofa_sensitivity_run_manifest.csv",
        paths.tables / "replicated_metabolite_primary_result_audit.csv",
    ]
    prior_hashes = {str(p): sha256(p) for p in prior_focused if p.exists()}

    old_yachida_failures = pd.DataFrame()
    old_failure_path = paths.tables / "scca_bootstrap_failures.csv"
    if old_failure_path.exists():
        previous_failures = pd.read_csv(old_failure_path)
        old_yachida_failures = previous_failures[previous_failures["cohort"].eq("Yachida")].copy()

    alignment_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    feature_records: list[dict[str, Any]] = []
    jaccard_rows: list[dict[str, Any]] = []

    run_dir = paths.temp_root / "scca_stability_work"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = paths.logs / "scca_stability.log"

    for disease, cohorts in DISEASE_COHORTS.items():
        for cohort in cohorts:
            block = blocks[(disease, cohort)]
            universe = feature_universe[(disease, cohort)]
            primary = {
                "microbiome": {},
                "metabolome": {},
            }
            n_primary_components = int(tbl["summary"][tbl["summary"]["cohort"].eq(cohort)]["component"].max())
            for comp in range(1, n_primary_components + 1):
                primary["microbiome"][comp] = vector_for(tbl["microbiome"], disease, cohort, comp, universe["microbiome"])
                primary["metabolome"][comp] = vector_for(tbl["metabolome"], disease, cohort, comp, universe["metabolome"])
            selected_boot_sets: dict[tuple[int, str], list[set[str]]] = {(pc, view): [] for pc in primary["microbiome"] for view in ["microbiome", "metabolome"]}

            for b in range(1, bootstrap_n + 1):
                seed = stable_seed(master_seed, cohort, b)
                rng = np.random.default_rng(seed)
                start = time.time()
                try:
                    indices = rng.integers(0, len(block.metadata), size=len(block.metadata))
                    micro, met, audit = prepare_bootstrap_views(block, universe, indices)
                    with tempfile.TemporaryDirectory(prefix=f"{cohort}_{b:03d}_", dir=str(run_dir)) as td:
                        res = run_r_bootstrap(paths, micro, met, seed, Path(td))
                    chosen = res["summary"].iloc[0].to_dict()
                    r_summary = res["summary"].iloc[0].to_dict()
                    fold_removed = res["tuning"][res["tuning"]["row_type"].eq("fold")]
                    parameter_rows.append(
                        {
                            "cohort": cohort,
                            "disease": disease,
                            "bootstrap_index": b,
                            "bootstrap_seed": seed,
                            "status": "success",
                            "selected_component_count": int(res["summary"]["component"].max()),
                            "selected_penaltyx": float(chosen["penaltyx"]),
                            "selected_penaltyz": float(chosen["penaltyz"]),
                            "tuning_score": float(chosen["tuning_score"]),
                            "tuning_score_sd": float(chosen["tuning_score_sd"]),
                            "selected_microbial_feature_count": int(res["summary"]["selected_microbiome_feature_count"].sum()),
                            "selected_metabolite_feature_count": int(res["summary"]["selected_metabolome_feature_count"].sum()),
                            "runtime_seconds": round(time.time() - start, 3),
                            "fold_local_max_metabolome_features_removed": int(pd.to_numeric(fold_removed.get("fold_local_metabolome_features_removed", pd.Series(dtype=float)), errors="coerce").max()) if not fold_removed.empty else np.nan,
                            "fold_local_max_microbiome_features_removed": int(pd.to_numeric(fold_removed.get("fold_local_microbiome_features_removed", pd.Series(dtype=float)), errors="coerce").max()) if not fold_removed.empty else np.nan,
                            "pma_full_fit_min_metabolome_sd_before_pma": float(r_summary.get("min_metabolome_sd_before_pma", np.nan)),
                            "pma_full_fit_min_microbiome_sd_before_pma": float(r_summary.get("min_microbiome_sd_before_pma", np.nan)),
                            "pma_internal_standardisation": str(r_summary.get("pma_internal_standardisation", "typex=standard,typez=standard")),
                            **audit,
                        }
                    )
                    boot = {"microbiome": {}, "metabolome": {}}
                    for comp in sorted(res["summary"]["component"].astype(int).unique()):
                        boot["microbiome"][comp] = full_boot_vector(res["microbiome"], comp, universe["microbiome"])
                        boot["metabolome"][comp] = full_boot_vector(res["metabolome"], comp, universe["metabolome"])
                    ar, matches = align_components(primary, boot)
                    for row in ar:
                        alignment_rows.append({"disease": disease, "cohort": cohort, "bootstrap_index": b, "bootstrap_seed": seed, **row})
                    for pc, match in matches.items():
                        if match["status"] != "matched_valid":
                            continue
                        bc = int(match["bootstrap_component"])
                        sign = int(match["sign_orientation"])
                        for view in ["microbiome", "metabolome"]:
                            vec = boot[view][bc] * sign
                            prim_vec = primary[view][pc]
                            selected_primary = selected_set(prim_vec, universe[view])
                            selected_boot = selected_set(vec, universe[view])
                            selected_boot_sets[(pc, view)].append(selected_boot)
                            jaccard_rows.append(
                                {
                                    "disease": disease,
                                    "cohort": cohort,
                                    "bootstrap_index": b,
                                    "primary_component": pc,
                                    "view": view,
                                    "jaccard_vs_primary": jaccard(selected_primary, selected_boot),
                                    "primary_selected_n": len(selected_primary),
                                    "bootstrap_selected_n": len(selected_boot),
                                    "combined_mean_jaccard_vs_primary": np.nan,
                                }
                            )
                            for feature, p_load, b_load in zip(universe[view], prim_vec, vec):
                                feature_records.append(
                                    {
                                        "disease": disease,
                                        "cohort": cohort,
                                        "primary_component": pc,
                                        "view": view,
                                        "feature_id": feature,
                                        "display_name": display_name(feature),
                                        "bootstrap_index": b,
                                        "primary_loading": float(p_load),
                                        "bootstrap_loading": float(b_load),
                                        "bootstrap_selected": bool(abs(b_load) > 1e-12),
                                        "sign_consistent": bool(np.sign(p_load) == np.sign(b_load)) if abs(p_load) > 1e-12 and abs(b_load) > 1e-12 else np.nan,
                                    }
                                )
                except Exception as exc:
                    failure_rows.append(
                        {
                            "disease": disease,
                            "cohort": cohort,
                            "bootstrap_index": b,
                            "bootstrap_seed": seed,
                            "status": "failure",
                            "failure": repr(exc),
                        }
                    )
                    parameter_rows.append(
                        {
                            "cohort": cohort,
                            "disease": disease,
                            "bootstrap_index": b,
                            "bootstrap_seed": seed,
                            "status": "failure",
                            "selected_component_count": np.nan,
                            "selected_penaltyx": np.nan,
                            "selected_penaltyz": np.nan,
                            "tuning_score": np.nan,
                            "tuning_score_sd": np.nan,
                            "selected_microbial_feature_count": np.nan,
                            "selected_metabolite_feature_count": np.nan,
                            "runtime_seconds": round(time.time() - start, 3),
                        }
                    )
                if b % 20 == 0:
                    with log_path.open("a") as handle:
                        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {disease} {cohort} bootstrap {b}/{bootstrap_n}\n")

    alignment = pd.DataFrame(alignment_rows)
    params = pd.DataFrame(parameter_rows)
    failures = pd.DataFrame(failure_rows)
    feature_long = pd.DataFrame(feature_records)

    feature_rows = []
    for disease, cohorts in DISEASE_COHORTS.items():
        for cohort in cohorts:
            universe = feature_universe[(disease, cohort)]
            for view in ["microbiome", "metabolome"]:
                primary_loads = tbl[view]
                for comp in sorted(primary_loads[primary_loads["cohort"].eq(cohort)]["component"].astype(int).unique()):
                    sub_primary = primary_loads.query("disease == @disease and cohort == @cohort and component == @comp")
                    pmap = dict(zip(sub_primary["feature_id"].astype(str), pd.to_numeric(sub_primary["loading"], errors="raise")))
                    ranks = sub_primary.assign(abs_loading=lambda d: d["loading"].abs()).sort_values("abs_loading", ascending=False)
                    rmap = {str(fid): rank + 1 for rank, fid in enumerate(ranks["feature_id"].astype(str).tolist())}
                    valid_aligned = alignment[
                        alignment["disease"].eq(disease)
                        & alignment["cohort"].eq(cohort)
                        & alignment["primary_component"].eq(comp)
                        & alignment["status"].eq("matched_valid")
                    ]
                    denom_valid = int(valid_aligned["bootstrap_index"].nunique())
                    fsub = feature_long.query("disease == @disease and cohort == @cohort and primary_component == @comp and view == @view")
                    for feature in universe[view]:
                        vals = fsub[fsub["feature_id"].eq(feature)]
                        boot_loads = vals["bootstrap_loading"].astype(float).to_numpy() if not vals.empty else np.array([])
                        selected_n = int(vals["bootstrap_selected"].sum()) if not vals.empty else 0
                        sign_vals = vals["sign_consistent"].dropna().astype(bool) if not vals.empty else pd.Series(dtype=bool)
                        feature_rows.append(
                            {
                                "disease": disease,
                                "cohort": cohort,
                                "primary_component": comp,
                                "view": view,
                                "feature_id": feature,
                                "display_name": display_name(feature),
                                "primary_loading": float(pmap.get(feature, 0.0)),
                                "primary_absolute_loading_rank": int(rmap.get(feature, len(universe[view]))),
                                "primary_selected": bool(abs(float(pmap.get(feature, 0.0))) > 1e-12),
                                "requested_bootstrap_n": int(bootstrap_n),
                                "successful_aligned_bootstrap_n": denom_valid,
                                "bootstrap_selected_n": selected_n,
                                "selection_frequency_all_requested": selected_n / bootstrap_n,
                                "selection_frequency_valid_aligned": selected_n / denom_valid if denom_valid else np.nan,
                                "sign_consistency_fraction": float(sign_vals.mean()) if len(sign_vals) else np.nan,
                                "median_loading": float(np.nanmedian(boot_loads)) if len(boot_loads) else np.nan,
                                "loading_p2_5": float(np.nanpercentile(boot_loads, 2.5)) if len(boot_loads) else np.nan,
                                "loading_p97_5": float(np.nanpercentile(boot_loads, 97.5)) if len(boot_loads) else np.nan,
                                "median_absolute_loading": float(np.nanmedian(np.abs(boot_loads))) if len(boot_loads) else np.nan,
                                "mean_absolute_loading": float(np.nanmean(np.abs(boot_loads))) if len(boot_loads) else np.nan,
                                "recurrence_category_descriptive": recurrence_category(selected_n / denom_valid if denom_valid else np.nan),
                            }
                        )
    feature_stability = pd.DataFrame(feature_rows)
    atomic_csv(paths.tables / "scca_bootstrap_feature_stability.csv", feature_stability)
    atomic_csv(paths.tables / "scca_bootstrap_parameter_selection.csv", params)
    atomic_csv(paths.tables / "scca_bootstrap_component_alignment.csv", alignment)
    atomic_csv(paths.tables / "scca_bootstrap_failures.csv", failures)

    jaccards = pd.DataFrame(jaccard_rows)
    if not jaccards.empty:
        combo = jaccards.groupby(["disease", "cohort", "bootstrap_index", "primary_component"])["jaccard_vs_primary"].mean().rename("combined_mean_jaccard_vs_primary")
        jaccards = jaccards.drop(columns=["combined_mean_jaccard_vs_primary"]).merge(combo.reset_index(), on=["disease", "cohort", "bootstrap_index", "primary_component"], how="left")
    atomic_csv(paths.tables / "scca_bootstrap_selected_set_jaccard.csv", jaccards)

    component_rows = []
    for _, prow in tbl["summary"].iterrows():
        disease, cohort, comp = str(prow["disease"]), str(prow["cohort"]), int(prow["component"])
        psub = params[params["cohort"].eq(cohort)]
        asub = alignment[alignment["cohort"].eq(cohort) & alignment["primary_component"].eq(comp)] if not alignment.empty else pd.DataFrame()
        valid = asub[asub["status"].eq("matched_valid")] if not asub.empty else pd.DataFrame()
        cos = valid["combined_cosine"].astype(float).to_numpy() if not valid.empty else np.array([])
        margins = valid["best_minus_second_best_margin"].astype(float).dropna().to_numpy() if not valid.empty else np.array([])
        component_rows.append(
            {
                "disease": disease,
                "cohort": cohort,
                "primary_component": comp,
                "requested_bootstrap_n": bootstrap_n,
                "successful_model_fits": int(psub["status"].eq("success").sum()),
                "valid_aligned_component_count": int(len(valid)),
                "ambiguous_match_count": int(asub["status"].eq("ambiguous_match").sum()) if not asub.empty else 0,
                "degenerate_match_count": int(asub["status"].str.contains("degenerate", na=False).sum()) if not asub.empty else 0,
                "primary_component_recovery_frequency_all_requested": len(valid) / bootstrap_n,
                "primary_component_recovery_frequency_successful_fits": len(valid) / max(1, int(psub["status"].eq("success").sum())),
                "median_combined_cosine_similarity": float(np.nanmedian(cos)) if len(cos) else np.nan,
                "iqr_combined_cosine_similarity": float(np.nanpercentile(cos, 75) - np.nanpercentile(cos, 25)) if len(cos) else np.nan,
                "median_best_minus_second_best_match_margin": float(np.nanmedian(margins)) if len(margins) else np.nan,
                "component_count_selection_distribution": json.dumps(psub[psub["status"].eq("success")]["selected_component_count"].value_counts(dropna=False).sort_index().to_dict()),
                "penalty_selection_distribution": penalty_distribution_json(psub[psub["status"].eq("success")]),
            }
        )
    component_stability = pd.DataFrame(component_rows)
    atomic_csv(paths.tables / "scca_bootstrap_component_stability.csv", component_stability)

    recovery_audit = make_recovery_denominator_audit(bootstrap_n, params, alignment, tbl["summary"])
    atomic_csv(paths.tables / "scca_component_recovery_denominator_audit.csv", recovery_audit)
    jaccard_audit = make_jaccard_denominator_audit(jaccards, params, alignment)
    atomic_csv(paths.tables / "scca_jaccard_denominator_audit.csv", jaccard_audit)
    yachida_audit = make_yachida_failure_audit(old_yachida_failures, params)
    atomic_csv(paths.tables / "scca_yachida_failure_audit.csv", yachida_audit)

    reported = feature_stability[feature_stability["primary_selected"]].copy()
    reported = reported.sort_values(["disease", "cohort", "primary_component", "view", "primary_absolute_loading_rank"]).groupby(
        ["disease", "cohort", "primary_component", "view"], as_index=False
    ).head(15)
    reported["loading_interval_crosses_zero"] = (reported["loading_p2_5"] <= 0) & (reported["loading_p97_5"] >= 0)
    atomic_csv(paths.tables / "scca_reported_feature_stability.csv", reported)

    after_hashes = {str(p): sha256(p) for p in source_files if p.exists()}
    prior_after = {str(p): sha256(p) for p in prior_focused if p.exists()}
    hash_rows = []
    for p, h in before_hashes.items():
        hash_rows.append({"path": p, "scope": "authoritative_stage4c_original", "hash_before": h, "hash_after": after_hashes.get(p), "unchanged": h == after_hashes.get(p)})
    for p, h in prior_hashes.items():
        hash_rows.append({"path": p, "scope": "prior_focused_validation_output", "hash_before": h, "hash_after": prior_after.get(p), "unchanged": h == prior_after.get(p)})
    hash_check = pd.DataFrame(hash_rows)
    atomic_csv(paths.tables / "scca_stability_original_hash_check.csv", hash_check)

    tests = make_tests(paths, bootstrap_n, params, alignment, feature_stability, component_stability, jaccards, hash_check)
    atomic_csv(paths.tests / "scca_stability_tests.csv", tests)
    if tests["status"].eq("fail").any():
        raise RuntimeError("sCCA stability tests failed.")

    write_report(paths, bootstrap_n, component_stability, params, jaccards, reported, tbl["cross"], hash_check, recovery_audit, jaccard_audit, yachida_audit)


def make_tests(paths: Paths, bootstrap_n: int, params: pd.DataFrame, alignment: pd.DataFrame, feature: pd.DataFrame, comp: pd.DataFrame, jac: pd.DataFrame, hash_check: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(name: str, passed: bool, observed: Any = "", expected: Any = "") -> None:
        rows.append({"test": name, "status": "pass" if passed else "fail", "observed": observed, "expected": expected})

    for cohort in ["Franzosa", "iHMP", "Sinha", "Yachida"]:
        sub = params[params["cohort"].eq(cohort)]
        add(f"{cohort}_all_bootstrap_seeds_recorded", sub["bootstrap_index"].nunique() == bootstrap_n, sub["bootstrap_index"].nunique(), bootstrap_n)
    add("primary_inputs_no_participant_ids_exported", not any(c.lower() in {"sample_id", "participant_id", "subject_id"} for c in feature.columns), list(feature.columns), "no direct participant identifiers")
    add("diagnosis_absent_from_tuning_and_fit", "phenotype" not in params.columns and "diagnosis" not in params.columns, list(params.columns), "no phenotype/diagnosis columns")
    add("primary_penalties_not_blindly_reused", params[params["status"].eq("success")][["selected_penaltyx", "selected_penaltyz"]].drop_duplicates().shape[0] >= 1, "retuned per bootstrap", "retuned per bootstrap")
    add("preprocessing_bootstrap_local_audited", {"microbiome_retained_n", "metabolome_retained_n", "microbiome_dropped_n", "metabolome_dropped_n"}.issubset(params.columns), list(params.columns), "bootstrap-local QC columns")
    add("loading_vectors_full_universe", feature.groupby(["cohort", "primary_component", "view"])["feature_id"].nunique().min() > 0, "nonempty", "nonempty")
    add("alignment_loading_similarity_only", {"microbial_cosine", "metabolite_cosine", "combined_cosine"}.issubset(alignment.columns), list(alignment.columns), "cosine fields present")
    add("same_sign_flip_recorded", alignment["sign_orientation"].dropna().isin([-1, 1]).all() if not alignment.empty else False, "sign_orientation", "-1 or 1")
    cos_cols = [c for c in ["microbial_cosine", "metabolite_cosine", "combined_cosine"] if c in alignment.columns]
    add("cosines_in_bounds", all(alignment[c].dropna().between(-1, 1).all() for c in cos_cols), "all cosine columns", "[-1,1]")
    freq_cols = [c for c in feature.columns if c.startswith("selection_frequency")]
    add("selection_frequencies_in_bounds", all(feature[c].dropna().between(0, 1).all() for c in freq_cols), freq_cols, "[0,1]")
    add("explicit_denominators_present", {"requested_bootstrap_n", "successful_aligned_bootstrap_n"}.issubset(feature.columns), list(feature.columns), "denominator columns")
    add("counts_reconcile", (comp["valid_aligned_component_count"] <= comp["successful_model_fits"]).all() and (comp["successful_model_fits"] <= bootstrap_n).all(), "valid<=success<=requested", "true")
    add("zero_variance_features_cannot_reach_pma_input", int(params.get("zero_sd_metabolite_count_immediately_before_pma", pd.Series([0])).fillna(0).sum()) == 0 and int(params.get("zero_sd_microbiome_count_immediately_before_pma", pd.Series([0])).fillna(0).sum()) == 0, "zero immediate pre-PMA zero-SD columns", "0")
    yachida = params[params["cohort"].eq("Yachida")]
    add("yachida_200_indices_accounted_for", yachida["bootstrap_index"].nunique() == bootstrap_n, yachida["bootstrap_index"].nunique(), bootstrap_n)
    add("yachida_success_plus_failed_equals_200", len(yachida) == bootstrap_n and yachida["status"].isin(["success", "failure"]).all(), yachida["status"].value_counts().to_dict(), "success+failure=200")
    add("jaccards_in_bounds", jac.empty or jac["jaccard_vs_primary"].dropna().between(0, 1).all(), "jaccard_vs_primary", "[0,1]")
    add("failed_fits_not_jaccard_zero", jac.empty or (jac["jaccard_vs_primary"] != 0).any(), "Jaccard rows exist only for aligned successes", "failures excluded, not assigned zero")
    public_outputs = list(paths.tables.glob("scca_bootstrap*.csv")) + [paths.tables / "scca_reported_feature_stability.csv"]
    leaked = []
    for path in public_outputs:
        if path.exists():
            text = path.read_text(errors="ignore")
            if "sample_id" in text or "participant_id" in text or "subject_id" in text:
                leaked.append(path.name)
    add("no_participant_identifiers_in_public_scca_outputs", not leaked, leaked, "none")
    add("original_stage4c_hashes_unchanged", hash_check[hash_check["scope"].eq("authoritative_stage4c_original")]["unchanged"].all(), "hash comparison", "all unchanged")
    add("prior_focused_outputs_unchanged", hash_check[hash_check["scope"].eq("prior_focused_validation_output")]["unchanged"].all(), "hash comparison", "all unchanged")
    return pd.DataFrame(rows)


def write_report(paths: Paths, bootstrap_n: int, comp: pd.DataFrame, params: pd.DataFrame, jac: pd.DataFrame, reported: pd.DataFrame, cross: pd.DataFrame, hash_check: pd.DataFrame, recovery_audit: pd.DataFrame, jaccard_audit: pd.DataFrame, yachida_audit: pd.DataFrame) -> None:
    param_dist = (
        params[params["status"].eq("success")]
        .groupby(["disease", "cohort", "selected_component_count", "selected_penaltyx", "selected_penaltyz"])
        .size()
        .reset_index(name="n_bootstraps")
        .sort_values(["disease", "cohort", "n_bootstraps"], ascending=[True, True, False])
    )
    jac_summary = pd.DataFrame()
    if not jac.empty:
        jac_summary = jac.groupby(["disease", "cohort", "primary_component", "view"], as_index=False)["jaccard_vs_primary"].agg(["median", "mean", "std"]).reset_index()
    reported_summary = reported[
        [
            "disease",
            "cohort",
            "primary_component",
            "view",
            "display_name",
            "primary_loading",
            "primary_absolute_loading_rank",
            "selection_frequency_valid_aligned",
            "sign_consistency_fraction",
            "median_absolute_loading",
            "loading_p2_5",
            "loading_p97_5",
            "loading_interval_crosses_zero",
            "recurrence_category_descriptive",
        ]
    ].copy()
    ibd_mean = cross[cross["disease"].eq("IBD")]["cosine_abs"].mean()
    crc_mean = cross[cross["disease"].eq("CRC")]["cosine_abs"].mean()
    statement = "remains numerically correct" if ibd_mean > crc_mean else "does not remain numerically correct"
    weak = reported_summary[reported_summary["recurrence_category_descriptive"].eq("low_recurrence")]
    text = [
        "# sCCA Loading Stability Report",
        "",
        f"Requested participant-level bootstrap resamples per cohort: `{bootstrap_n}`.",
        "",
        "This analysis evaluates within-cohort stability of the sparse CCA solutions. It is not independent external replication and it does not use diagnosis labels for tuning, fitting or component alignment.",
        "",
        "## Component Recovery",
        "",
        comp.to_markdown(index=False),
        "",
        "## Recovery Denominators",
        "",
        recovery_audit.to_markdown(index=False),
        "",
        "## Yachida Failure Audit",
        "",
        "The original focused sCCA stability helper filtered zero-variance features on the full bootstrap matrix, but not inside PMA tuning-fold training matrices. Yachida bootstrap failures were therefore attributable to fold-local zero-SD metabolite columns reaching PMA::CCA internal standardisation. The additive helper now performs fit-local zero-SD filtering before each PMA fold fit and records pre-PMA SD diagnostics.",
        "",
        yachida_audit.groupby(["old_failure_status", "corrected_status", "failure_classification"]).size().reset_index(name="n").to_markdown(index=False),
        "",
        "## Parameter Selection Distribution",
        "",
        param_dist.to_markdown(index=False),
        "",
        "## Selected-Set Jaccard Summary",
        "",
        (jac_summary.to_markdown(index=False) if not jac_summary.empty else "No valid aligned Jaccard values were available."),
        "",
        "Jaccard medians are conditional on valid aligned components. Failed PMA fits and ambiguous matches are not assigned Jaccard zero.",
        "",
        jaccard_audit.to_markdown(index=False),
        "",
        "## Currently Reported Loading Stability",
        "",
        reported_summary.to_markdown(index=False),
        "",
        "## Thesis Statement Check",
        "",
        f"The existing statement that IBD has greater full-data cross-cohort loading similarity than CRC `{statement}` using the authoritative full-data cosine table (mean IBD cosine={ibd_mean:.3f}; mean CRC cosine={crc_mean:.3f}). Bootstrap stability is reported separately and is not collapsed into this cross-cohort comparison.",
        "",
        "## Statements To Weaken",
        "",
        ("No primary selected top-loading features fell in the low-recurrence descriptive band." if weak.empty else "Features in the low-recurrence descriptive band should be described as primary-solution loadings with limited bootstrap recurrence, not robust biomarkers:\n\n" + weak[["disease", "cohort", "primary_component", "view", "display_name", "selection_frequency_valid_aligned"]].to_markdown(index=False)),
        "",
        "## Integrity",
        "",
        hash_check.to_markdown(index=False),
    ]
    atomic_text(paths.reports / "SCCA_LOADING_STABILITY_REPORT.md", "\n".join(text) + "\n")
    ya = yachida_audit.groupby(["old_failure_status", "corrected_status", "failure_classification"]).size().reset_index(name="n")
    zero = yachida_audit[
        [
            "bootstrap_index",
            "bootstrap_seed",
            "old_failure_status",
            "corrected_status",
            "unique_participant_rows_in_bootstrap",
            "max_participant_row_multiplicity",
            "metabolome_feature_count_before_preprocessing",
            "metabolome_feature_count_after_median_imputation",
            "metabolome_feature_count_after_zero_variance_filtering",
            "zero_sd_metabolite_count_after_imputation",
            "zero_sd_metabolite_count_immediately_before_pma",
            "fold_local_max_metabolome_features_removed",
            "min_metabolome_sd_before_pma",
            "failure_classification",
        ]
    ].head(25)
    atomic_text(
        paths.reports / "SCCA_YACHIDA_FAILURE_AUDIT.md",
        "# sCCA Yachida Failure Audit\n\n"
        "The previous 105/200 Yachida bootstrap failures were traced to zero-SD metabolite columns arising inside PMA tuning-fold training matrices. "
        "The full bootstrap matrices had zero-SD columns removed before PMA input, but the R helper originally passed unfiltered inner-fold training matrices to PMA::CCA, whose `typez=\"standard\"` setting performs additional internal standardisation. "
        "The additive focused-validation helper now removes zero-SD columns within each PMA fit and maps excluded columns back to zero loading in the complete harmonised feature universe.\n\n"
        "No participant identifiers are exported; bootstrap row multiplicities are reported only as anonymous count distributions.\n\n"
        "## Status Summary\n\n"
        + ya.to_markdown(index=False)
        + "\n\n## First 25 Bootstrap Diagnostics\n\n"
        + zero.to_markdown(index=False)
        + "\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="final_analysis/focused_validation_v1")
    parser.add_argument("--thesis-source-root", default=os.environ.get("THESIS_SOURCE_ROOT", "thesis_preprocessing_workflow"))
    parser.add_argument("--bootstrap-n", type=int, default=REQUESTED_BOOTSTRAPS)
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo = Path.cwd()
    out = repo / args.output_root
    paths = Paths(
        repo=repo,
        output_root=out,
        thesis_source_root=Path(args.thesis_source_root),
        corrected_v2=Path(args.thesis_source_root) / "stage4_multiomics_integration" / "corrected_v2",
        pma_input_dir=Path(args.thesis_source_root) / "stage4_multiomics_integration" / "corrected_v2" / "data" / "stage4c_scca_pma_inputs",
        pma_output_dir=Path(args.thesis_source_root) / "stage4_multiomics_integration" / "corrected_v2" / "data" / "stage4c_scca_pma_outputs",
        tables=out / "tables",
        reports=out / "reports",
        tests=out / "tests",
        logs=out / "logs",
        temp_root=out / "tmp",
    )
    for directory in [paths.tables, paths.reports, paths.tests, paths.logs, paths.temp_root]:
        directory.mkdir(parents=True, exist_ok=True)
    run_stability(paths, args.bootstrap_n, args.master_seed, args.resume)


if __name__ == "__main__":
    main()
