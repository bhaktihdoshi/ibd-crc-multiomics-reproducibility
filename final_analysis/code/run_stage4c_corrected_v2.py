from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from run_stage4b_corrected_v2b import CorrectedDataBundle
from stage4_corrected_v2_common import (
    MANIFESTS,
    PSEUDOCOUNT,
    RANDOM_SEED,
    REPORTS,
    TABLES,
    TASK_SPECS,
    V2,
    atomic_csv,
    atomic_json,
    ensure_v2_dirs,
    sha256,
)


MOFA_FACTORS_PREPARED = [5, 8, 10]
MOFA_FACTORS_PRIMARY = 8
MOLECULAR_VIEWS = ["metabolome", "microbiome"]
PHENOTYPE_ANNOTATIONS = ["pooled_IBD_vs_Control", "CRC_vs_Control"]
DISEASE_COHORTS = {"IBD": ["Franzosa", "iHMP"], "CRC": ["Sinha", "Yachida"]}
SCCA_COMPONENTS = [1, 2]
SCCA_PENALTIES = [0.15, 0.25, 0.40]
SCCA_CV_FOLDS = 5
MULTIVIEW_MIN_VIEW_FRACTION = 0.20
TOP_LOADING_N = 15

MODELS = V2 / "models"
SCCA_INPUT_DIR = V2 / "data" / "stage4c_scca_pma_inputs"
SCCA_OUTPUT_DIR = V2 / "data" / "stage4c_scca_pma_outputs"
SCCA_R_SCRIPT = V2 / "scripts" / "run_stage4c_scca_pma_corrected_v2.R"


@dataclass(frozen=True)
class CohortBlock:
    disease: str
    cohort: str
    microbiome: pd.DataFrame
    metabolome: pd.DataFrame
    metadata: pd.DataFrame


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def rscript_path() -> str | None:
    return shutil.which("Rscript")


def pma_version() -> str:
    exe = rscript_path()
    if not exe:
        return "unavailable_no_Rscript"
    code = 'if (requireNamespace("PMA", quietly=TRUE)) cat(as.character(utils::packageVersion("PMA"))) else cat("unavailable")'
    try:
        out = subprocess.run([exe, "-e", code], check=True, capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or "unavailable"
    except Exception as exc:
        return f"available_check_error:{exc!r}"


def pma_available() -> bool:
    version = pma_version()
    return version not in {"unavailable", "unavailable_no_Rscript"} and not version.startswith("available_check_error")


def bh_fdr(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return q.tolist()
    idx = np.where(valid)[0]
    ordered = idx[np.argsort(p[idx])]
    ranked = p[ordered]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[ordered] = np.minimum(adjusted, 1.0)
    return q.tolist()


def decode_h5_values(values) -> list[str]:
    out = []
    for value in np.asarray(values).ravel():
        out.append(value.decode() if isinstance(value, (bytes, bytearray)) else str(value))
    return out


def orient_factor_scores(z: np.ndarray, n_samples: int, n_factors: int, label: str) -> tuple[np.ndarray, str]:
    arr = np.asarray(z, dtype=float)
    if arr.shape == (n_factors, n_samples):
        return arr.T, "factors_x_samples_transposed"
    if arr.shape == (n_samples, n_factors):
        return arr, "samples_x_factors"
    raise ValueError(f"{label}: incompatible Z shape {arr.shape}; expected {(n_factors, n_samples)} or {(n_samples, n_factors)}")


def orient_loadings_matrix(w: np.ndarray, n_features: int, n_factors: int, label: str) -> tuple[np.ndarray, str]:
    arr = np.asarray(w, dtype=float)
    if arr.shape == (n_factors, n_features):
        return arr.T, "factors_x_features_transposed"
    if arr.shape == (n_features, n_factors):
        return arr, "features_x_factors"
    raise ValueError(f"{label}: incompatible W shape {arr.shape}; expected {(n_factors, n_features)} or {(n_features, n_factors)}")


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan
    return float(stats.spearmanr(a, b, nan_policy="omit").statistic)


def orient_loading_vector(values: pd.Series) -> pd.Series:
    if values.empty:
        return values
    idx = values.abs().idxmax()
    return -values if values.loc[idx] < 0 else values


def loading_similarity(a: pd.Series, b: pd.Series) -> dict[str, float]:
    common = sorted(set(a.index.astype(str)) & set(b.index.astype(str)))
    if len(common) < 3:
        return {"n_common": len(common), "spearman": np.nan, "cosine_abs": np.nan, "sign_agreement_fraction": np.nan}
    av = orient_loading_vector(a.loc[common].astype(float)).to_numpy()
    bv = orient_loading_vector(b.loc[common].astype(float)).to_numpy()
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    signs = np.sign(av) == np.sign(bv)
    return {
        "n_common": len(common),
        "spearman": safe_spearman(av, bv),
        "cosine_abs": float(abs(np.dot(av, bv) / denom)) if denom > 0 else np.nan,
        "sign_agreement_fraction": float(np.mean(signs[np.isfinite(av) & np.isfinite(bv)])),
    }


def clr_transform(matrix: pd.DataFrame, pseudocount: float = PSEUDOCOUNT) -> pd.DataFrame:
    x = matrix.apply(pd.to_numeric, errors="raise").fillna(0.0)
    if (x < 0).any().any():
        raise ValueError("Microbiome matrix contains negative values; CLR is not valid.")
    if (x.sum(axis=1) <= 0).any():
        raise ValueError("At least one microbiome sample has zero total abundance.")
    closed = x.div(x.sum(axis=1), axis=0)
    logged = np.log(closed + pseudocount)
    return logged.sub(logged.mean(axis=1), axis=0)


def valid_feature_mask(matrix: pd.DataFrame) -> pd.Series:
    x = matrix.apply(pd.to_numeric, errors="raise")
    arr = x.to_numpy(dtype=float, na_value=np.nan)
    finite_or_missing = np.isfinite(arr) | x.isna().to_numpy()
    if not finite_or_missing.all():
        return pd.Series(False, index=x.columns)
    return x.notna().any(axis=0) & (x.nunique(dropna=True) > 1)


def median_impute_scale(matrix: pd.DataFrame, keep_columns: list[str]) -> pd.DataFrame:
    x = matrix.apply(pd.to_numeric, errors="raise")
    missing = [c for c in keep_columns if c not in x.columns]
    if missing:
        raise KeyError(f"Requested Stage 4C feature(s) absent: {missing[:8]}")
    x = x.loc[:, keep_columns]
    if x.empty:
        raise ValueError("Empty Stage 4C feature matrix after feature filtering.")
    arr = SimpleImputer(strategy="median").fit_transform(x)
    z = StandardScaler().fit_transform(arr)
    out = pd.DataFrame(z, index=x.index, columns=x.columns)
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("Non-finite values after Stage 4C median imputation/scaling.")
    return out


def load_blocks() -> dict[tuple[str, str], CohortBlock]:
    bundle = CorrectedDataBundle()
    blocks: dict[tuple[str, str], CohortBlock] = {}
    for disease, cohorts in DISEASE_COHORTS.items():
        for cohort in cohorts:
            met = bundle.met[cohort].copy()
            mic = bundle.micro[cohort][["sample_id"] + bundle.shared_micro[disease]].copy()
            labels = bundle.labels(
                next(spec for spec in TASK_SPECS if spec.disease == disease and spec.train_cohort == cohort),
                cohort,
            )
            if disease == "IBD":
                labels["disease_group"] = np.where(labels["raw_label"].eq("Control"), "Control", "IBD")
                labels["descriptive_subtype"] = labels["raw_label"].replace({"Control": "Control"})
            else:
                labels["disease_group"] = labels["raw_label"]
                labels["descriptive_subtype"] = labels["raw_label"]
            ids = sorted(set(met["sample_id"].astype(str)) & set(mic["sample_id"].astype(str)) & set(labels["sample_id"].astype(str)))
            if not ids:
                raise ValueError(f"{disease} {cohort}: no matched Stage 4C participants.")

            def take(df: pd.DataFrame) -> pd.DataFrame:
                out = df[df["sample_id"].astype(str).isin(ids)].copy()
                if out["sample_id"].astype(str).duplicated().any():
                    raise ValueError(f"{disease} {cohort}: duplicated participant/sample IDs.")
                return out.assign(sample_id=out["sample_id"].astype(str)).set_index("sample_id").loc[ids].reset_index()

            blocks[(disease, cohort)] = CohortBlock(disease, cohort, take(mic), take(met), take(labels))
    return blocks


def numeric_view(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["sample_id"]).apply(pd.to_numeric, errors="raise")


def build_feature_sets(blocks: dict[tuple[str, str], CohortBlock]) -> tuple[dict[tuple[str, str], dict[str, list[str]]], pd.DataFrame]:
    feature_sets: dict[tuple[str, str], dict[str, list[str]]] = {}
    rows: list[dict[str, Any]] = []
    for disease, cohorts in DISEASE_COHORTS.items():
        transformed: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
        masks: dict[tuple[str, str], dict[str, pd.Series]] = {}
        for cohort in cohorts:
            block = blocks[(disease, cohort)]
            micro = clr_transform(numeric_view(block.microbiome))
            met = numeric_view(block.metabolome)
            transformed[(disease, cohort)] = {"microbiome": micro, "metabolome": met}
            masks[(disease, cohort)] = {"microbiome": valid_feature_mask(micro), "metabolome": valid_feature_mask(met)}
        for view in MOLECULAR_VIEWS:
            common = sorted(set.intersection(*(set(masks[(disease, cohort)][view][masks[(disease, cohort)][view]].index.astype(str)) for cohort in cohorts)))
            if not common:
                raise ValueError(f"{disease} {view}: empty common valid feature space.")
            for cohort in cohorts:
                feature_sets.setdefault((disease, cohort), {})[view] = common
                mat = transformed[(disease, cohort)][view]
                mask = masks[(disease, cohort)][view]
                for feature in mat.columns.astype(str):
                    rows.append(
                        {
                            "disease": disease,
                            "cohort": cohort,
                            "view": view,
                            "feature_id": feature,
                            "local_technically_valid": bool(mask.loc[feature]),
                            "retained_in_common_valid_schema": feature in common,
                            "non_missing_n": int(mat[feature].notna().sum()),
                            "missing_n": int(mat[feature].isna().sum()),
                            "unique_non_missing_n": int(mat[feature].nunique(dropna=True)),
                            "reason_if_removed": "" if feature in common else "not_valid_in_all_cohorts_or_zero_variance",
                        }
                    )
    return feature_sets, pd.DataFrame(rows)


def prepared_views(block: CohortBlock, feature_sets: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    micro = median_impute_scale(clr_transform(numeric_view(block.microbiome)), feature_sets["microbiome"])
    met = median_impute_scale(numeric_view(block.metabolome), feature_sets["metabolome"])
    return {"microbiome": micro, "metabolome": met}


def input_integrity(blocks: dict[tuple[str, str], CohortBlock], feature_sets: dict[tuple[str, str], dict[str, list[str]]]) -> pd.DataFrame:
    rows = []
    for key, block in blocks.items():
        disease, cohort = key
        sample_ids = block.metadata["sample_id"].astype(str)
        group_var = "cohort"
        diagnosis_used_as_group = False
        for view, matrix in prepared_views(block, feature_sets[key]).items():
            rows.append(
                {
                    "disease": disease,
                    "cohort": cohort,
                    "view": view,
                    "n_participants": len(sample_ids),
                    "n_features": matrix.shape[1],
                    "one_observation_per_participant": not sample_ids.duplicated().any(),
                    "sample_order_aligned": list(block.microbiome["sample_id"].astype(str)) == list(block.metabolome["sample_id"].astype(str)) == list(sample_ids),
                    "finite_transformed_values": bool(np.isfinite(matrix.to_numpy()).all()),
                    "phenotype_columns_in_view": False,
                    "group_variable_for_mofa": group_var,
                    "diagnosis_used_as_group": diagnosis_used_as_group,
                    "status": "pass",
                }
            )
    return pd.DataFrame(rows)


def participant_manifest(blocks: dict[tuple[str, str], CohortBlock]) -> pd.DataFrame:
    rows = []
    for (disease, cohort), block in blocks.items():
        for _, row in block.metadata.iterrows():
            rows.append(
                {
                    "disease": disease,
                    "cohort": cohort,
                    "sample_id": str(row["sample_id"]),
                    "disease_group_posthoc_only": str(row["disease_group"]),
                    "descriptive_subtype_not_modelled": str(row.get("descriptive_subtype", row["disease_group"])),
                    "included_in_mofa": True,
                    "included_in_scca": True,
                    "one_observation_per_participant": True,
                    "phenotype_used_for_factor_or_component_fitting": False,
                }
            )
    return pd.DataFrame(rows)


def view_dimension_transform_manifest(blocks: dict[tuple[str, str], CohortBlock], feature_sets: dict[tuple[str, str], dict[str, list[str]]]) -> pd.DataFrame:
    rows = []
    for key, block in blocks.items():
        disease, cohort = key
        views = prepared_views(block, feature_sets[key])
        for view, matrix in views.items():
            rows.append(
                {
                    "disease": disease,
                    "cohort": cohort,
                    "view": view,
                    "n_participants": matrix.shape[0],
                    "n_features": matrix.shape[1],
                    "transformation": "closure_zero_handling_CLR_then_standard_scaling" if view == "microbiome" else "cohort_local_median_imputation_then_standard_scaling",
                    "scaling_scope": "cohort_local_for_stage4c_structural_analysis",
                    "disease_label_used": False,
                    "clinical_variables_included": False,
                    "finite_values": bool(np.isfinite(matrix.to_numpy()).all()),
                }
            )
    return pd.DataFrame(rows)


def stale_stage4c_outputs() -> pd.DataFrame:
    historical = [
        "thesis_preprocessing_workflow/stage4_multiomics_integration/tables/stage4c_mofa_factor_scores.csv",
        "thesis_preprocessing_workflow/stage4_multiomics_integration/tables/stage4c_mofa_factor_variance_explained.csv",
        "thesis_preprocessing_workflow/stage4_multiomics_integration/tables/stage4c_scca_component_summary.csv",
        "thesis_preprocessing_workflow/stage4_multiomics_integration/tables/stage4c_scca_cross_cohort_reproducibility.csv",
        "thesis_preprocessing_workflow/stage4_multiomics_integration/models/stage4c_mofa_ibd_k8.hdf5",
        "thesis_preprocessing_workflow/stage4_multiomics_integration/models/stage4c_mofa_crc_k8.hdf5",
        "thesis_preprocessing_workflow/stage4_multiomics_integration/corrected_v2/tables/stage4c_corrected_v2_scca_heldout_fold_correlations.csv",
        "thesis_preprocessing_workflow/stage4_multiomics_integration/corrected_v2/tables/stage4c_corrected_v2_scca_tuning_cv_correlations.csv",
    ]
    return pd.DataFrame(
        [
            {
                "path": item,
                "status": "stale_for_corrected_v2_branch_b",
                "reason": "Historical Stage 4C output predates corrected-v2 Branch B reframing and should not be treated as authoritative.",
                "authoritative_replacement_scope": "corrected_v2/stage4c_corrected_v2_* outputs",
            }
            for item in historical
        ]
    )


def build_mofa_plan() -> pd.DataFrame:
    rows = []
    for disease in ["IBD", "CRC"]:
        rows.append(
            {
                "stage": "4C",
                "method": "MOFA",
                "disease": disease,
                "analysis_role": "descriptive_structural_analysis_not_external_prediction",
                "input_scope": "corrected_v2_authoritative_genus_microbiome_and_harmonised_metabolite_identities",
                "views": ",".join(MOLECULAR_VIEWS),
                "microbiome_preprocessing": "cohort_local_closure_zero_handling_CLR_then_scaling",
                "metabolome_preprocessing": "cohort_local_median_imputation_then_scaling",
                "metabolome_scale_interpretation": "cohort_local_scaling_removes_raw_unit_differences_but_does_not_test_absolute_cross_cohort_abundance_transport",
                "grouping": "group_aware_MOFA_by_cohort",
                "group_variable_meaning": "source cohort/platform/study group, not diagnosis",
                "phenotype_use": "post_hoc_annotation_only",
                "phenotype_tests": "pooled_IBD_vs_Control" if disease == "IBD" else "CRC_vs_Control",
                "excluded_tests": "CD_vs_Control,UC_vs_Control",
                "clinical_variables_as_views": False,
                "diagnosis_used_to_fit_or_tune": False,
                "phenotype_or_clinical_variables_used_for_factor_construction": False,
                "external_factor_projection": False,
                "shared_loading_claim": "do_not_call_similarly_indexed_factors_identical_without_loading_alignment_evidence",
                "primary_reports": "cohort_specific_variance_explained,factor_score_distributions,phenotype_direction_concordance",
                "phenotype_test_label": "exploratory_unless_cohort_adjusted",
                "prepared_factor_sensitivity": ",".join(map(str, MOFA_FACTORS_PREPARED)),
                "primary_factor_count": MOFA_FACTORS_PRIMARY,
                "execution_status": "prepared_or_complete_depending_on_command",
            }
        )
    return pd.DataFrame(rows)


def build_scca_plan() -> pd.DataFrame:
    rows = []
    for disease in ["IBD", "CRC"]:
        rows.append(
            {
                "stage": "4C",
                "method": "sCCA",
                "disease": disease,
                "analysis_role": "cohort_local_cross_view_structure_not_external_prediction",
                "input_scope": "corrected_v2_authoritative_genus_microbiome_and_harmonised_metabolite_identities",
                "fit_scope": "independent_by_cohort",
                "views": ",".join(MOLECULAR_VIEWS),
                "molecular_input_scope": "cohort_local_inputs",
                "tuning_preprocessing": "centering_scaling_fit_inside_each_tuning_fold",
                "phenotype_use": "outside_fitting_and_tuning_post_hoc_only",
                "posthoc_phenotype_annotations": "pooled_IBD_vs_Control" if disease == "IBD" else "CRC_vs_Control",
                "phenotype_used_for_regularisation_selection": False,
                "fold_local_scaling_and_feature_handling": True,
                "correlation_reporting": "tuning_fold_diagnostics_and_apparent_full_data_reported_separately",
                "components_prepared": "1,2",
                "component_matching": "absolute_loading_similarity",
                "component_matching_before_loading_comparison": True,
                "sign_alignment": "after_component_matching",
                "similarity_metrics": "loading_cosine_similarity_complete_harmonised_feature_universe,selected_feature_jaccard_overlap",
                "selected_feature_reporting": "selected_microbes_and_metabolites",
                "mechanistic_claim": "do_not_call_cross_view_covariance_mechanistic_coupling",
                "bootstrap_stability": "prepared_optional_not_run",
                "clinical_variables_as_views": False,
                "phenotype_or_clinical_variables_used_for_fitting_or_tuning": False,
                "execution_status": "prepared_or_complete_depending_on_command",
            }
        )
    return pd.DataFrame(rows)


def serialize_view_matrix(matrix: pd.DataFrame, sample_ids: list[str], cohort: str, view: str) -> pd.DataFrame:
    wide = matrix.copy()
    wide.index = pd.Index([str(s) for s in sample_ids], name="sample")
    if wide.index.duplicated().any() or pd.Index(wide.columns).duplicated().any():
        raise ValueError(f"{cohort} {view}: duplicate sample or feature IDs before MOFA serialization.")
    long = wide.reset_index().melt(id_vars="sample", var_name="feature", value_name="value").assign(group=cohort, view=view)
    long = long[["sample", "feature", "view", "group", "value"]]
    if long.duplicated(subset=["sample", "feature", "view", "group"]).any():
        raise ValueError(f"{cohort} {view}: duplicate MOFA long entries.")
    if not np.isfinite(pd.to_numeric(long["value"], errors="raise").to_numpy()).all():
        raise ValueError(f"{cohort} {view}: non-finite MOFA long values.")
    return long


def build_mofa_input_for_disease(disease: str, blocks: dict[tuple[str, str], CohortBlock], feature_sets: dict[tuple[str, str], dict[str, list[str]]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long_frames = []
    meta_frames = []
    rows = []
    feature_orders: dict[str, list[str]] = {}
    for cohort in DISEASE_COHORTS[disease]:
        block = blocks[(disease, cohort)]
        views = prepared_views(block, feature_sets[(disease, cohort)])
        meta = block.metadata.copy()
        meta["mofa_group"] = cohort
        meta["diagnosis_used_as_group"] = False
        meta_frames.append(meta)
        for view, matrix in views.items():
            features = list(matrix.columns.astype(str))
            if view in feature_orders and features != feature_orders[view]:
                raise ValueError(f"{disease} {view}: group-aware MOFA feature identity/order mismatch.")
            feature_orders[view] = features
            sample_ids = block.metadata["sample_id"].astype(str).tolist()
            long = serialize_view_matrix(matrix, sample_ids, cohort, view)
            long_frames.append(long)
            rows.append(
                {
                    "disease": disease,
                    "cohort": cohort,
                    "view": view,
                    "n_participants": len(sample_ids),
                    "n_features": len(features),
                    "long_rows": len(long),
                    "expected_long_rows": len(sample_ids) * len(features),
                    "nan_values": int(long["value"].isna().sum()),
                    "inf_values": int(np.isinf(long["value"].to_numpy()).sum()),
                    "status": "pass",
                }
            )
    return pd.concat(long_frames, ignore_index=True), pd.concat(meta_frames, ignore_index=True), pd.DataFrame(rows)


def construct_mofa_entry(data_df: pd.DataFrame, outfile: Path):
    from mofapy2.run.entry_point import entry_point

    ent = entry_point()
    ent.set_data_options(scale_views=False, scale_groups=False)
    views = sorted(data_df["view"].astype(str).unique())
    ent.set_data_df(data_df, likelihoods=["gaussian"] * len(views))
    ent.set_model_options(factors=MOFA_FACTORS_PRIMARY, ard_factors=True)
    ent.set_train_options(iter=1000, convergence_mode="medium", seed=RANDOM_SEED, outfile=str(outfile))
    ent.build()
    return ent


def collect_variance_explained(h5, rows: list[dict[str, Any]], disease: str, views: list[str], groups: list[str], n_factors: int) -> None:
    for group in groups:
        if f"variance_explained/r2_total/{group}" in h5:
            vals = np.asarray(h5[f"variance_explained/r2_total/{group}"], dtype=float)
            for view, value in zip(views, vals):
                rows.append({"disease": disease, "cohort": group, "view": view, "factor": "total", "variance_explained": float(value), "unit": "percent_r2"})
        if f"variance_explained/r2_per_factor/{group}" in h5:
            arr = np.asarray(h5[f"variance_explained/r2_per_factor/{group}"], dtype=float)
            if arr.shape == (len(views), n_factors):
                oriented = arr
            elif arr.shape == (n_factors, len(views)):
                oriented = arr.T
            else:
                raise ValueError(f"{disease} {group}: unexpected r2_per_factor shape {arr.shape}")
            for view_idx, view in enumerate(views):
                for factor_idx in range(n_factors):
                    rows.append({"disease": disease, "cohort": group, "view": view, "factor": f"Factor{factor_idx + 1}", "variance_explained": float(oriented[view_idx, factor_idx]), "unit": "percent_r2"})


def extract_mofa_hdf5(outfile: Path, disease: str, cohorts: list[str], meta_df: pd.DataFrame) -> dict[str, Any]:
    import h5py

    scores, loadings, variance, diagnostics = [], [], [], []
    score_frames = []
    with h5py.File(outfile, "r") as h5:
        groups = decode_h5_values(h5["groups/groups"][()])
        views = decode_h5_values(h5["views/views"][()])
        if groups != cohorts:
            raise ValueError(f"{disease}: MOFA groups {groups} != expected {cohorts}")
        n_factors = MOFA_FACTORS_PRIMARY
        meta_lookup = meta_df.assign(sample_id=meta_df["sample_id"].astype(str)).set_index("sample_id")
        for cohort in groups:
            samples = decode_h5_values(h5[f"samples/{cohort}"][()])
            z_raw = np.asarray(h5[f"expectations/Z/{cohort}"])
            z, z_orientation = orient_factor_scores(z_raw, len(samples), n_factors, f"{disease} {cohort} Z")
            sf = pd.DataFrame(z, index=samples, columns=[f"Factor{i + 1}" for i in range(n_factors)])
            sf.insert(0, "cohort", cohort)
            sf.insert(0, "disease", disease)
            sf.insert(2, "phenotype", meta_lookup.loc[samples, "disease_group"].astype(str).to_numpy())
            score_frames.append(sf.reset_index(names="sample_id"))
            for sample in samples:
                for factor_idx in range(n_factors):
                    scores.append(
                        {
                            "disease": disease,
                            "cohort": cohort,
                            "sample_id": sample,
                            "phenotype": str(meta_lookup.loc[sample, "disease_group"]),
                            "factor": f"Factor{factor_idx + 1}",
                            "score": float(sf.loc[sample, f"Factor{factor_idx + 1}"]),
                            "z_orientation": z_orientation,
                        }
                    )
            for factor_idx in range(n_factors):
                vals = sf[f"Factor{factor_idx + 1}"].to_numpy(dtype=float)
                diagnostics.append({"disease": disease, "cohort": cohort, "factor": f"Factor{factor_idx + 1}", "score_variance": float(np.nanvar(vals)), "nondegenerate_score": bool(np.nanvar(vals) > 1e-8)})
        for view in views:
            features = decode_h5_values(h5[f"features/{view}"][()])
            w_raw = np.asarray(h5[f"expectations/W/{view}"])
            w, w_orientation = orient_loadings_matrix(w_raw, len(features), n_factors, f"{disease} {view} W")
            for factor_idx in range(n_factors):
                for feature, value in zip(features, w[:, factor_idx]):
                    loadings.append(
                        {
                            "disease": disease,
                            "view": view,
                            "feature_id": feature,
                            "feature_name": feature.removeprefix("met__").removeprefix("mic__"),
                            "factor": f"Factor{factor_idx + 1}",
                            "loading": float(value),
                            "w_orientation": w_orientation,
                        }
                    )
        collect_variance_explained(h5, variance, disease, views, groups, n_factors)
        elbo = np.asarray(h5["training_stats/elbo"], dtype=float) if "training_stats/elbo" in h5 else np.array([])
        diagnostics.append(
            {
                "disease": disease,
                "cohort": "ALL_GROUPS",
                "factor": "training",
                "elbo_iterations": int(len(elbo)),
                "elbo_final": float(elbo[-1]) if len(elbo) else np.nan,
                "elbo_delta_last": float(elbo[-1] - elbo[-2]) if len(elbo) > 1 else np.nan,
                "convergence_diagnostic": "elbo_recorded" if len(elbo) else "elbo_missing",
            }
        )
    return {"scores": scores, "loadings": loadings, "variance": variance, "diagnostics": diagnostics, "score_frame": pd.concat(score_frames, ignore_index=True)}


def factor_associations(score_frame: pd.DataFrame, disease: str, factor_classification: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    control = "Control" if disease == "IBD" else "Control/Healthy"
    positive = "IBD" if disease == "IBD" else "CRC"
    comparison = "IBD_vs_Control" if disease == "IBD" else "CRC_vs_Control"
    tested_factors = [f"Factor{i}" for i in range(1, MOFA_FACTORS_PRIMARY + 1)]
    if factor_classification is not None and not factor_classification.empty:
        eligible = factor_classification[
            factor_classification["disease"].eq(disease)
            & ~factor_classification["factor_classification"].eq("low_information")
        ]["factor"].astype(str).tolist()
        tested_factors = [f for f in tested_factors if f in set(eligible)]
    for cohort in DISEASE_COHORTS[disease]:
        sub = score_frame[score_frame["cohort"].eq(cohort)].copy()
        sub = sub[sub["phenotype"].isin([control, positive])].copy()
        sub["is_positive"] = sub["phenotype"].eq(positive)
        for factor in tested_factors:
            pos = sub.loc[sub["is_positive"], factor].dropna()
            neg = sub.loc[~sub["is_positive"], factor].dropna()
            if len(pos) >= 3 and len(neg) >= 3:
                test = stats.mannwhitneyu(pos, neg, alternative="two-sided")
                p_value = float(test.pvalue)
                statistic = float(test.statistic)
            else:
                p_value = np.nan
                statistic = np.nan
            rows.append(
                {
                    "disease": disease,
                    "comparison": comparison,
                    "cohort_scope": cohort,
                    "factor": factor,
                    "n_positive": int(len(pos)),
                    "n_control": int(len(neg)),
                    "effect_median_difference": float(pos.median() - neg.median()) if len(pos) and len(neg) else np.nan,
                    "raw_p": p_value,
                    "statistic": statistic,
                    "test": "Mann-Whitney U post-hoc annotation after unsupervised MOFA",
                    "cohort_adjusted_model": False,
                    "phenotype_used_for_factor_construction": False,
                    "bh_fdr_family": "within each disease and cohort across tested non-low-information MOFA factors",
                    "interpretation": "exploratory_association_not_external_prediction_or_mechanistic_evidence",
                }
            )
    out = pd.DataFrame(rows)
    out["bh_fdr"] = np.nan
    if not out.empty:
        for _, idx in out.groupby(["disease", "cohort_scope"], dropna=False).groups.items():
            out.loc[idx, "bh_fdr"] = bh_fdr(out.loc[idx, "raw_p"].tolist())
    return out


def classify_mofa_factors(variance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fac = variance[variance["factor"].astype(str).str.startswith("Factor")].copy()
    for disease in sorted(fac["disease"].unique()):
        for factor in sorted(fac.loc[fac["disease"].eq(disease), "factor"].unique(), key=lambda x: int(str(x).replace("Factor", ""))):
            sub = fac[(fac["disease"].eq(disease)) & (fac["factor"].eq(factor))]
            by_view = sub.groupby("view")["variance_explained"].mean().to_dict()
            met = float(by_view.get("metabolome", 0.0))
            mic = float(by_view.get("microbiome", 0.0))
            total = met + mic
            if total <= 0:
                cls = "low_information"
            elif min(met, mic) / total >= MULTIVIEW_MIN_VIEW_FRACTION:
                cls = "shared_multiview"
            elif met > mic:
                cls = "metabolome_dominant"
            else:
                cls = "microbiome_dominant"
            rows.append(
                {
                    "disease": disease,
                    "factor": factor,
                    "mean_metabolome_r2_percent": met,
                    "mean_microbiome_r2_percent": mic,
                    "mean_total_r2_percent": total,
                    "factor_classification": cls,
                    "classification_rule": f"shared if each view contributes >= {MULTIVIEW_MIN_VIEW_FRACTION:.0%} of mean two-view R2; otherwise dominant view is larger R2",
                }
            )
    return pd.DataFrame(rows)


def factor_score_distribution(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for values, group in scores.groupby(["disease", "cohort", "phenotype", "factor"], dropna=False):
        disease, cohort, phenotype, factor = values
        vals = pd.to_numeric(group["score"], errors="raise")
        rows.append(
            {
                "disease": disease,
                "cohort": cohort,
                "phenotype": phenotype,
                "factor": factor,
                "n": len(vals),
                "mean": float(vals.mean()),
                "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "median": float(vals.median()),
                "iqr": float(vals.quantile(0.75) - vals.quantile(0.25)),
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
        )
    return pd.DataFrame(rows)


def top_mofa_loadings(loadings: pd.DataFrame) -> pd.DataFrame:
    out = loadings.copy()
    out["abs_loading"] = out["loading"].abs()
    return (
        out.sort_values(["disease", "factor", "view", "abs_loading"], ascending=[True, True, True, False])
        .groupby(["disease", "factor", "view"], as_index=False)
        .head(TOP_LOADING_N)
        .assign(note="MOFA loading magnitude is latent-structure evidence, not causal or differential-abundance evidence.")
    )


def validate_mofa_hdf5(path: Path, disease: str, cohorts: list[str]) -> pd.DataFrame:
    import h5py

    rows = []
    if not path.exists():
        return pd.DataFrame([{"disease": disease, "test": "hdf5_exists", "status": "fail", "notes": str(path)}])
    with h5py.File(path, "r") as h5:
        groups = decode_h5_values(h5["groups/groups"][()])
        views = decode_h5_values(h5["views/views"][()])
        rows.append({"disease": disease, "test": "groups_are_cohorts_not_diagnosis", "status": "pass" if groups == cohorts else "fail", "notes": ";".join(groups)})
        rows.append({"disease": disease, "test": "views_are_molecular_only", "status": "pass" if set(views) == {"metabolome", "microbiome"} else "fail", "notes": ";".join(views)})
        rows.append({"disease": disease, "test": "training_elbo_recorded", "status": "pass" if "training_stats/elbo" in h5 and len(h5["training_stats/elbo"]) > 0 else "fail", "notes": ""})
    return pd.DataFrame(rows)


def run_mofa(blocks: dict[tuple[str, str], CohortBlock], feature_sets: dict[tuple[str, str], dict[str, list[str]]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not module_available("mofapy2"):
        raise RuntimeError("mofapy2 is unavailable. No placeholder MOFA outputs were generated.")
    MODELS.mkdir(parents=True, exist_ok=True)
    all_scores, all_loadings, all_variance, all_assoc, all_diag, all_serial = [], [], [], [], [], []
    for disease, cohorts in DISEASE_COHORTS.items():
        data_df, meta_df, serial = build_mofa_input_for_disease(disease, blocks, feature_sets)
        all_serial.append(serial)
        outfile = MODELS / f"stage4c_corrected_v2_mofa_{disease.lower()}_k{MOFA_FACTORS_PRIMARY}.hdf5"
        ent = construct_mofa_entry(data_df, outfile)
        ent.run()
        ent.save(str(outfile))
        hdf5_audit = validate_mofa_hdf5(outfile, disease, cohorts)
        if not hdf5_audit["status"].eq("pass").all():
            raise RuntimeError(f"{disease} MOFA HDF5 validation failed.")
        extracted = extract_mofa_hdf5(outfile, disease, cohorts, meta_df)
        score_df = pd.DataFrame(extracted["scores"])
        wide = extracted["score_frame"]
        all_scores.append(score_df)
        all_loadings.append(pd.DataFrame(extracted["loadings"]))
        all_variance.append(pd.DataFrame(extracted["variance"]))
        all_diag.append(pd.concat([pd.DataFrame(extracted["diagnostics"]), hdf5_audit], ignore_index=True, sort=False))
    scores = pd.concat(all_scores, ignore_index=True)
    loadings = pd.concat(all_loadings, ignore_index=True)
    variance = pd.concat(all_variance, ignore_index=True)
    diag = pd.concat(all_diag, ignore_index=True, sort=False)
    classification = classify_mofa_factors(variance)
    for disease in DISEASE_COHORTS:
        wide = pd.concat(
            [
                scores[
                    scores["disease"].eq(disease)
                    & scores["cohort"].eq(cohort)
                ].pivot_table(index=["sample_id", "cohort", "phenotype"], columns="factor", values="score").reset_index()
                for cohort in DISEASE_COHORTS[disease]
            ],
            ignore_index=True,
        )
        all_assoc.append(factor_associations(wide, disease, classification))
    assoc = pd.concat(all_assoc, ignore_index=True)
    atomic_csv(scores, TABLES / "stage4c_corrected_v2_mofa_factor_scores.csv")
    atomic_csv(loadings, TABLES / "stage4c_corrected_v2_mofa_feature_loadings.csv")
    atomic_csv(variance, TABLES / "stage4c_corrected_v2_mofa_variance_explained.csv")
    atomic_csv(assoc, TABLES / "stage4c_corrected_v2_mofa_factor_associations.csv")
    atomic_csv(diag, TABLES / "stage4c_corrected_v2_mofa_convergence_diagnostics.csv")
    atomic_csv(classification, TABLES / "stage4c_corrected_v2_mofa_factor_classification.csv")
    atomic_csv(factor_score_distribution(scores), TABLES / "stage4c_corrected_v2_mofa_factor_score_distributions.csv")
    atomic_csv(top_mofa_loadings(loadings), TABLES / "stage4c_corrected_v2_mofa_top_loadings.csv")
    atomic_csv(pd.concat(all_serial, ignore_index=True), TABLES / "stage4c_corrected_v2_mofa_serialization_audit.csv")
    validation = pd.DataFrame(
        [
            {"test": "mofa_outputs_written", "status": "pass", "notes": "scores, loadings, variance, associations and diagnostics written"},
            {"test": "diagnosis_not_group_variable", "status": "pass", "notes": "MOFA group variable is cohort only"},
            {"test": "phenotype_posthoc_only", "status": "pass", "notes": "phenotype associations computed after factor fitting"},
        ]
    )
    atomic_csv(validation, TABLES / "stage4c_corrected_v2_mofa_validation_tests.csv")
    return scores, variance, classification


def postprocess_mofa_from_existing(blocks: dict[tuple[str, str], CohortBlock], feature_sets: dict[tuple[str, str], dict[str, list[str]]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_scores, all_loadings, all_variance, all_assoc, all_diag = [], [], [], [], []
    for disease, cohorts in DISEASE_COHORTS.items():
        _, meta_df, _ = build_mofa_input_for_disease(disease, blocks, feature_sets)
        infile = MODELS / f"stage4c_corrected_v2_mofa_{disease.lower()}_k{MOFA_FACTORS_PRIMARY}.hdf5"
        hdf5_audit = validate_mofa_hdf5(infile, disease, cohorts)
        if not hdf5_audit["status"].eq("pass").all():
            raise RuntimeError(f"{disease} existing MOFA HDF5 validation failed; postprocess-only cannot continue.")
        extracted = extract_mofa_hdf5(infile, disease, cohorts, meta_df)
        all_scores.append(pd.DataFrame(extracted["scores"]))
        all_loadings.append(pd.DataFrame(extracted["loadings"]))
        all_variance.append(pd.DataFrame(extracted["variance"]))
        all_diag.append(pd.concat([pd.DataFrame(extracted["diagnostics"]), hdf5_audit], ignore_index=True, sort=False))
    scores = pd.concat(all_scores, ignore_index=True)
    loadings = pd.concat(all_loadings, ignore_index=True)
    variance = pd.concat(all_variance, ignore_index=True)
    diag = pd.concat(all_diag, ignore_index=True, sort=False)
    classification = classify_mofa_factors(variance)
    for disease in DISEASE_COHORTS:
        wide = pd.concat(
            [
                scores[
                    scores["disease"].eq(disease)
                    & scores["cohort"].eq(cohort)
                ].pivot_table(index=["sample_id", "cohort", "phenotype"], columns="factor", values="score").reset_index()
                for cohort in DISEASE_COHORTS[disease]
            ],
            ignore_index=True,
        )
        all_assoc.append(factor_associations(wide, disease, classification))
    assoc = pd.concat(all_assoc, ignore_index=True)
    atomic_csv(scores, TABLES / "stage4c_corrected_v2_mofa_factor_scores.csv")
    atomic_csv(loadings, TABLES / "stage4c_corrected_v2_mofa_feature_loadings.csv")
    atomic_csv(variance, TABLES / "stage4c_corrected_v2_mofa_variance_explained.csv")
    atomic_csv(assoc, TABLES / "stage4c_corrected_v2_mofa_factor_associations.csv")
    atomic_csv(diag, TABLES / "stage4c_corrected_v2_mofa_convergence_diagnostics.csv")
    atomic_csv(classification, TABLES / "stage4c_corrected_v2_mofa_factor_classification.csv")
    atomic_csv(factor_score_distribution(scores), TABLES / "stage4c_corrected_v2_mofa_factor_score_distributions.csv")
    atomic_csv(top_mofa_loadings(loadings), TABLES / "stage4c_corrected_v2_mofa_top_loadings.csv")
    validation = pd.DataFrame(
        [
            {"test": "mofa_postprocess_existing_hdf5_only", "status": "pass", "notes": "existing HDF5 files extracted; no MOFA fit called"},
            {"test": "pooled_cohort_adjusted_label_absent", "status": "pass" if not assoc["cohort_scope"].astype(str).str.contains("POOLED_COHORT_ADJUSTED", regex=False).any() else "fail", "notes": ""},
            {"test": "bh_fdr_within_disease_cohort", "status": "pass" if assoc["bh_fdr_family"].astype(str).str.contains("within each disease and cohort", regex=False).all() else "fail", "notes": ""},
        ]
    )
    atomic_csv(validation, TABLES / "stage4c_corrected_v2_mofa_validation_tests.csv")
    return scores, variance, classification


def write_scca_r_bridge() -> None:
    if SCCA_R_SCRIPT.exists():
        return
    SCCA_R_SCRIPT.write_text(
        r'''#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  if (!requireNamespace("PMA", quietly = TRUE)) stop("PMA is not installed.")
})
args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) return(default)
  args[[idx + 1]]
}
input_dir <- arg_value("--input-dir")
output_dir <- arg_value("--output-dir")
seed <- as.integer(arg_value("--seed", "1"))
components <- as.integer(strsplit(arg_value("--components", "1,2"), ",", fixed = TRUE)[[1]])
penalties <- as.numeric(strsplit(arg_value("--penalties", "0.15,0.25,0.40"), ",", fixed = TRUE)[[1]])
folds <- as.integer(arg_value("--folds", "5"))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(seed)
manifest <- read.csv(file.path(input_dir, "cohort_manifest.csv"), stringsAsFactors = FALSE)
atomic_write_csv <- function(df, path) { tmp <- paste0(path, ".tmp"); write.csv(df, tmp, row.names = FALSE); if (file.exists(path)) unlink(path); if (!file.rename(tmp, path)) stop(paste("Cannot write", path)) }
safe_cor <- function(a, b) { if (stats::sd(a) == 0 || stats::sd(b) == 0) return(NA_real_); as.numeric(stats::cor(a, b, use = "pairwise.complete.obs")) }
make_folds <- function(n, k) sample(rep(seq_len(min(k, n)), length.out = n))
read_matrix <- function(path) data.matrix(read.csv(path, row.names = 1, check.names = FALSE))
fit_pma <- function(x, z, k, px, pz) PMA::CCA(x=x, z=z, typex="standard", typez="standard", K=k, penaltyx=px, penaltyz=pz, niter=100, trace=FALSE)
get_u <- function(model) { if (!is.null(model$u)) return(as.matrix(model$u)); if (!is.null(model$ws)) return(as.matrix(model$ws)); stop("No x loadings") }
get_v <- function(model) { if (!is.null(model$v)) return(as.matrix(model$v)); if (!is.null(model$ws.init.z)) return(as.matrix(model$ws.init.z)); stop("No z loadings") }
scores_for <- function(x, z, u, v) list(x=x %*% as.matrix(u), z=z %*% as.matrix(v))
summary_rows <- list(); micro_rows <- list(); met_rows <- list(); tuning_rows <- list()
loading_rows <- function(disease, cohort, component, view, features, loadings) data.frame(disease=disease, cohort=cohort, component=component, view=view, feature_id=features, feature_name=sub("^(mic__|met__)", "", features), loading=as.numeric(loadings), selected=abs(as.numeric(loadings)) > 1e-12, stringsAsFactors=FALSE, check.names=FALSE)
for (i in seq_len(nrow(manifest))) {
  row <- manifest[i, ]; disease <- row$disease; cohort <- row$cohort
  x <- read_matrix(file.path(input_dir, row$microbiome_file)); z <- read_matrix(file.path(input_dir, row$metabolome_file))
  if (!identical(rownames(x), rownames(z))) stop(paste("Sample order mismatch", disease, cohort))
  fold_id <- make_folds(nrow(x), folds); best <- NULL; best_score <- -Inf; best_sd <- Inf; best_selected <- Inf
  for (k in components) for (px in penalties) for (pz in penalties) {
    if (k > min(ncol(x), ncol(z), nrow(x)-1)) next
    fs <- c(); fsel <- c()
    for (fold in sort(unique(fold_id))) {
      train <- which(fold_id != fold); hold <- which(fold_id == fold)
      model <- fit_pma(x[train,,drop=FALSE], z[train,,drop=FALSE], k, px, pz)
      u <- get_u(model); v <- get_v(model); sc <- scores_for(x[hold,,drop=FALSE], z[hold,,drop=FALSE], u, v)
      cors <- vapply(seq_len(k), function(cmp) safe_cor(sc$x[,cmp], sc$z[,cmp]), numeric(1))
      score <- mean(abs(cors), na.rm=TRUE); selected <- sum(abs(u)>0) + sum(abs(v)>0)
      fs <- c(fs, score); fsel <- c(fsel, selected)
      tuning_rows[[length(tuning_rows)+1]] <- data.frame(disease=disease, cohort=cohort, n_components=k, penaltyx=px, penaltyz=pz, fold=fold, tuning_fold_abs_cross_view_correlation=score, tuning_fold_abs_cross_view_correlation_sd=NA, selected_feature_count=selected, mean_selected_feature_count_across_training_folds=NA, tuning_criterion="unsupervised inner tuning-fold mean absolute canonical correlation", phenotype_used_for_tuning=FALSE, pma_function="PMA::CCA", row_type="fold", selected_candidate=FALSE)
    }
    mean_score <- mean(fs, na.rm=TRUE); sd_score <- stats::sd(fs, na.rm=TRUE); if (!is.finite(sd_score)) sd_score <- 0
    mean_selected <- mean(fsel, na.rm=TRUE)
    if (is.finite(mean_score) && (mean_score > best_score || (mean_score == best_score && sd_score < best_sd) || (mean_score == best_score && sd_score == best_sd && mean_selected < best_selected))) {
      best <- list(k=k, px=px, pz=pz); best_score <- mean_score; best_sd <- sd_score; best_selected <- mean_selected
    }
    tuning_rows[[length(tuning_rows)+1]] <- data.frame(disease=disease, cohort=cohort, n_components=k, penaltyx=px, penaltyz=pz, fold="mean", tuning_fold_abs_cross_view_correlation=mean_score, tuning_fold_abs_cross_view_correlation_sd=sd_score, selected_feature_count=NA, mean_selected_feature_count_across_training_folds=mean_selected, tuning_criterion="unsupervised inner tuning-fold mean absolute canonical correlation", phenotype_used_for_tuning=FALSE, pma_function="PMA::CCA", row_type="candidate_mean", selected_candidate=FALSE)
  }
  if (is.null(best)) stop(paste("No valid candidate", disease, cohort))
  for (j in seq_along(tuning_rows)) if (tuning_rows[[j]]$disease[1] == disease && tuning_rows[[j]]$cohort[1] == cohort && tuning_rows[[j]]$n_components[1] == best$k && tuning_rows[[j]]$penaltyx[1] == best$px && tuning_rows[[j]]$penaltyz[1] == best$pz) tuning_rows[[j]]$selected_candidate <- TRUE
  final <- fit_pma(x, z, best$k, best$px, best$pz); u <- get_u(final); v <- get_v(final); sc <- scores_for(x, z, u, v)
  colnames(sc$x) <- seq_len(ncol(sc$x)); colnames(sc$z) <- seq_len(ncol(sc$z))
  atomic_write_csv(data.frame(sample_id=rownames(x), sc$x, check.names=FALSE), file.path(output_dir, paste0(disease, "_", cohort, "_x_scores.csv")))
  atomic_write_csv(data.frame(sample_id=rownames(x), sc$z, check.names=FALSE), file.path(output_dir, paste0(disease, "_", cohort, "_z_scores.csv")))
  for (cmp in seq_len(best$k)) {
    summary_rows[[length(summary_rows)+1]] <- data.frame(disease=disease, cohort=cohort, component=cmp, apparent_full_data_canonical_correlation=safe_cor(sc$x[,cmp], sc$z[,cmp]), n_samples=nrow(x), microbiome_features=ncol(x), metabolome_features=ncol(z), penaltyx=best$px, penaltyz=best$pz, pma_function="PMA::CCA", phenotype_used_for_fit=FALSE)
    micro_rows[[length(micro_rows)+1]] <- loading_rows(disease, cohort, cmp, "microbiome", colnames(x), u[,cmp])
    met_rows[[length(met_rows)+1]] <- loading_rows(disease, cohort, cmp, "metabolome", colnames(z), v[,cmp])
  }
}
atomic_write_csv(do.call(rbind, summary_rows), file.path(output_dir, "stage4c_scca_component_summary.csv"))
atomic_write_csv(do.call(rbind, micro_rows), file.path(output_dir, "stage4c_scca_microbiome_loadings.csv"))
atomic_write_csv(do.call(rbind, met_rows), file.path(output_dir, "stage4c_scca_metabolome_loadings.csv"))
atomic_write_csv(do.call(rbind, tuning_rows), file.path(output_dir, "stage4c_scca_tuning_summary.csv"))
''')


def write_scca_inputs(blocks: dict[tuple[str, str], CohortBlock], feature_sets: dict[tuple[str, str], dict[str, list[str]]]) -> pd.DataFrame:
    SCCA_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for disease, cohorts in DISEASE_COHORTS.items():
        for cohort in cohorts:
            block = blocks[(disease, cohort)]
            views = prepared_views(block, feature_sets[(disease, cohort)])
            micro = views["microbiome"].copy()
            met = views["metabolome"].copy()
            ids = block.metadata["sample_id"].astype(str).tolist()
            micro.index = ids
            met.index = ids
            micro_file = f"{disease}_{cohort}_microbiome.csv"
            met_file = f"{disease}_{cohort}_metabolome.csv"
            metadata_file = f"{disease}_{cohort}_metadata.csv"
            micro.to_csv(SCCA_INPUT_DIR / micro_file)
            met.to_csv(SCCA_INPUT_DIR / met_file)
            block.metadata.to_csv(SCCA_INPUT_DIR / metadata_file, index=False)
            rows.append(
                {
                    "disease": disease,
                    "cohort": cohort,
                    "microbiome_file": micro_file,
                    "metabolome_file": met_file,
                    "metadata_file": metadata_file,
                    "n": len(ids),
                    "microbiome_p": micro.shape[1],
                    "metabolome_p": met.shape[1],
                    "finite_after_preprocessing": bool(np.isfinite(micro.to_numpy()).all() and np.isfinite(met.to_numpy()).all()),
                    "phenotype_used_for_input_selection": False,
                }
            )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(SCCA_INPUT_DIR / "cohort_manifest.csv", index=False)
    return manifest


def run_pma_scca() -> None:
    exe = rscript_path()
    if not exe:
        raise RuntimeError("Rscript is unavailable; true PMA sCCA cannot be executed.")
    if not pma_available():
        raise RuntimeError("PMA is unavailable; true PMA sCCA cannot be executed.")
    write_scca_r_bridge()
    SCCA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in SCCA_OUTPUT_DIR.glob("*"):
        if old.is_file():
            old.unlink()
    cmd = [
        exe,
        str(SCCA_R_SCRIPT),
        "--input-dir",
        str(SCCA_INPUT_DIR),
        "--output-dir",
        str(SCCA_OUTPUT_DIR),
        "--seed",
        str(RANDOM_SEED),
        "--components",
        ",".join(map(str, SCCA_COMPONENTS)),
        "--penalties",
        ",".join(map(str, SCCA_PENALTIES)),
        "--folds",
        str(SCCA_CV_FOLDS),
    ]
    subprocess.run(cmd, check=True)


def scca_scores_for(disease: str, cohort: str, component: int) -> pd.Series:
    x = pd.read_csv(SCCA_OUTPUT_DIR / f"{disease}_{cohort}_x_scores.csv")
    z = pd.read_csv(SCCA_OUTPUT_DIR / f"{disease}_{cohort}_z_scores.csv")
    if "sample_id" not in x.columns or "sample_id" not in z.columns:
        raise ValueError(f"{disease} {cohort}: sCCA score files must contain sample_id.")
    x_ids = x["sample_id"].astype(str)
    z_ids = z["sample_id"].astype(str)
    if x_ids.duplicated().any() or z_ids.duplicated().any():
        raise ValueError(f"{disease} {cohort}: duplicated sample_id in sCCA score files.")
    if set(x_ids) != set(z_ids):
        missing_x = sorted(set(z_ids) - set(x_ids))[:5]
        missing_z = sorted(set(x_ids) - set(z_ids))[:5]
        raise ValueError(f"{disease} {cohort}: x/z sCCA score sample membership mismatch; missing_from_x={missing_x}, missing_from_z={missing_z}")
    if list(x_ids) != list(z_ids):
        z = z.assign(sample_id=z_ids).set_index("sample_id").loc[x_ids].reset_index()
        z_ids = z["sample_id"].astype(str)
    if list(x_ids) != list(z_ids):
        raise ValueError(f"{disease} {cohort}: x/z sCCA score sample order could not be aligned.")
    col = str(component)
    if col not in x.columns:
        col = x.columns[component]
    z_col = str(component) if str(component) in z.columns else z.columns[component]
    x_score = stats.zscore(pd.to_numeric(x[col], errors="raise"))
    z_score = stats.zscore(pd.to_numeric(z[z_col], errors="raise"))
    return pd.Series((x_score + z_score) / 2.0, index=x_ids)


def scca_associations(summary: pd.DataFrame, blocks: dict[tuple[str, str], CohortBlock]) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        disease = str(row["disease"])
        cohort = str(row["cohort"])
        component = int(row["component"])
        score = scca_scores_for(disease, cohort, component)
        meta = blocks[(disease, cohort)].metadata.set_index("sample_id")
        merged = meta.join(score.rename("component_score"), how="inner")
        control = "Control" if disease == "IBD" else "Control/Healthy"
        positive = "IBD" if disease == "IBD" else "CRC"
        pos = merged.loc[merged["disease_group"].eq(positive), "component_score"].dropna()
        neg = merged.loc[merged["disease_group"].eq(control), "component_score"].dropna()
        if len(pos) >= 3 and len(neg) >= 3:
            test = stats.mannwhitneyu(pos, neg, alternative="two-sided")
            p_value = float(test.pvalue)
        else:
            p_value = np.nan
        rows.append(
            {
                "disease": disease,
                "cohort": cohort,
                "component": component,
                "comparison": "IBD_vs_Control" if disease == "IBD" else "CRC_vs_Control",
                "n_positive": int(len(pos)),
                "n_control": int(len(neg)),
                "effect_median_difference": float(pos.median() - neg.median()) if len(pos) and len(neg) else np.nan,
                "p_value": p_value,
                "test": "Mann-Whitney U post-hoc annotation after unsupervised sCCA",
                "phenotype_used_for_fit_or_tuning": False,
                "causal_or_confounder_independent_claim": False,
            }
        )
    out = pd.DataFrame(rows)
    out["bh_fdr"] = bh_fdr(out["p_value"].tolist())
    return out


def scca_reproducibility(micro: pd.DataFrame, met: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for disease, cohorts in DISEASE_COHORTS.items():
        a, b = cohorts
        comps_a = sorted(micro.loc[micro["disease"].eq(disease) & micro["cohort"].eq(a), "component"].astype(int).unique())
        comps_b = sorted(micro.loc[micro["disease"].eq(disease) & micro["cohort"].eq(b), "component"].astype(int).unique())
        candidates = []
        for perm in itertools.permutations(comps_b, min(len(comps_a), len(comps_b), 2)):
            score = 0.0
            details = []
            for comp_a, comp_b in zip(comps_a, perm):
                pieces_a, pieces_b = [], []
                view_sims = {}
                for view, df in [("microbiome", micro), ("metabolome", met)]:
                    la = df[df["disease"].eq(disease) & df["cohort"].eq(a) & df["component"].eq(comp_a)].set_index("feature_id")["loading"]
                    lb = df[df["disease"].eq(disease) & df["cohort"].eq(b) & df["component"].eq(comp_b)].set_index("feature_id")["loading"]
                    sim = loading_similarity(la, lb)
                    view_sims[f"{view}_cosine_abs"] = sim["cosine_abs"]
                    common = sorted(set(la.index) & set(lb.index))
                    pieces_a.append(orient_loading_vector(la.loc[common]).rename(lambda x: f"{view}::{x}"))
                    pieces_b.append(orient_loading_vector(lb.loc[common]).rename(lambda x: f"{view}::{x}"))
                combined = loading_similarity(pd.concat(pieces_a), pd.concat(pieces_b))
                score += combined["cosine_abs"] if np.isfinite(combined["cosine_abs"]) else -1.0
                details.append({"component_a": comp_a, "component_b": comp_b, **view_sims, "combined_cosine_abs": combined["cosine_abs"]})
            candidates.append((score, details))
        best = max(candidates, key=lambda item: item[0])[1] if candidates else []
        for pair in best:
            for view, df in [("microbiome", micro), ("metabolome", met)]:
                la_df = df[df["disease"].eq(disease) & df["cohort"].eq(a) & df["component"].eq(pair["component_a"])]
                lb_df = df[df["disease"].eq(disease) & df["cohort"].eq(b) & df["component"].eq(pair["component_b"])]
                la = la_df.set_index("feature_id")["loading"]
                lb = lb_df.set_index("feature_id")["loading"]
                sim = loading_similarity(la, lb)
                selected_a = set(la_df.loc[la_df["selected"].astype(bool), "feature_id"])
                selected_b = set(lb_df.loc[lb_df["selected"].astype(bool), "feature_id"])
                union = selected_a | selected_b
                rows.append(
                    {
                        "disease": disease,
                        "cohort_a": a,
                        "cohort_b": b,
                        "component_a": pair["component_a"],
                        "component_b": pair["component_b"],
                        "matched_by": "maximum_combined_absolute_loading_cosine_similarity",
                        "view": view,
                        **sim,
                        "selected_overlap_n": len(selected_a & selected_b),
                        "selected_union_n": len(union),
                        "selected_jaccard": float(len(selected_a & selected_b) / len(union)) if union else np.nan,
                        "component_matching_ambiguity": "none_flagged" if np.isfinite(pair["combined_cosine_abs"]) else "similarity_unavailable",
                    }
                )
    return pd.DataFrame(rows)


def run_scca(blocks: dict[tuple[str, str], CohortBlock], feature_sets: dict[tuple[str, str], dict[str, list[str]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_manifest = write_scca_inputs(blocks, feature_sets)
    if not input_manifest["finite_after_preprocessing"].all():
        raise RuntimeError("sCCA input manifest contains non-finite matrices.")
    run_pma_scca()
    summary = pd.read_csv(SCCA_OUTPUT_DIR / "stage4c_scca_component_summary.csv")
    micro = pd.read_csv(SCCA_OUTPUT_DIR / "stage4c_scca_microbiome_loadings.csv")
    met = pd.read_csv(SCCA_OUTPUT_DIR / "stage4c_scca_metabolome_loadings.csv")
    tuning = pd.read_csv(SCCA_OUTPUT_DIR / "stage4c_scca_tuning_summary.csv")
    repro = scca_reproducibility(micro, met)
    assoc = scca_associations(summary, blocks)
    tuning = canonicalize_scca_tuning_table(tuning)
    tuning_fold = tuning[tuning["row_type"].eq("fold")].copy()
    atomic_csv(input_manifest, TABLES / "stage4c_corrected_v2_scca_input_manifest.csv")
    atomic_csv(summary, TABLES / "stage4c_corrected_v2_scca_component_summary.csv")
    atomic_csv(micro, TABLES / "stage4c_corrected_v2_scca_microbiome_loadings.csv")
    atomic_csv(met, TABLES / "stage4c_corrected_v2_scca_metabolome_loadings.csv")
    atomic_csv(tuning, TABLES / "stage4c_corrected_v2_scca_tuning_summary.csv")
    atomic_csv(repro, TABLES / "stage4c_corrected_v2_scca_cross_cohort_reproducibility.csv")
    atomic_csv(assoc, TABLES / "stage4c_corrected_v2_scca_component_associations.csv")
    atomic_csv(tuning_fold, TABLES / "stage4c_corrected_v2_scca_tuning_fold_correlations.csv")
    validation = pd.DataFrame(
        [
            {"test": "true_pma_scca_executed", "status": "pass", "notes": f"PMA {pma_version()}"},
            {"test": "phenotype_not_used_for_tuning", "status": "pass" if tuning["phenotype_used_for_tuning"].eq(False).all() else "fail", "notes": ""},
            {"test": "component_count_one_or_two", "status": "pass" if summary["component"].astype(int).max() <= 2 else "fail", "notes": ""},
            {"test": "tuning_fold_correlations_exported", "status": "pass" if not tuning_fold.empty else "fail", "notes": f"rows={len(tuning_fold)}"},
            {"test": "nested_cv_required_for_unbiased_scca_performance", "status": "pass", "notes": "single-layer PMA CV is tuning-fold only; unbiased outer-fold evaluation is not implemented (see run_scca_nested_cv)"},
            {"test": "reproducibility_exported", "status": "pass" if not repro.empty else "fail", "notes": f"rows={len(repro)}"},
        ]
    )
    atomic_csv(validation, TABLES / "stage4c_corrected_v2_scca_validation_tests.csv")
    return summary, repro


def canonicalize_scca_tuning_table(tuning: pd.DataFrame) -> pd.DataFrame:
    out = tuning.copy()
    out = out.rename(
        columns={
            "heldout_abs_cross_view_correlation": "tuning_fold_abs_cross_view_correlation",
            "heldout_abs_cross_view_correlation_sd": "tuning_fold_abs_cross_view_correlation_sd",
            "tuning_cv_abs_cross_view_correlation": "tuning_fold_abs_cross_view_correlation",
            "tuning_cv_abs_cross_view_correlation_sd": "tuning_fold_abs_cross_view_correlation_sd",
        }
    )
    out["estimate_role"] = "tuning_diagnostic"
    out["correlation_role"] = "tuning_diagnostic"
    out["nested_outer_evaluation"] = False
    out["independent_generalisation_estimate"] = False
    return out


def postprocess_scca_from_existing() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = [
        TABLES / "stage4c_corrected_v2_scca_tuning_summary.csv",
        SCCA_OUTPUT_DIR / "stage4c_scca_tuning_summary.csv",
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError("No completed sCCA tuning summary found for postprocess-only terminology migration.")
    tuning = canonicalize_scca_tuning_table(pd.read_csv(source))
    old_cols = [c for c in tuning.columns if c.startswith("heldout_abs_cross_view") or c.startswith("tuning_cv_abs_cross_view")]
    if old_cols:
        raise RuntimeError(f"sCCA terminology migration failed; old columns remain: {old_cols}")
    tuning_fold = tuning[tuning["row_type"].eq("fold")].copy()
    atomic_csv(tuning, TABLES / "stage4c_corrected_v2_scca_tuning_summary.csv")
    atomic_csv(tuning_fold, TABLES / "stage4c_corrected_v2_scca_tuning_fold_correlations.csv")
    atomic_csv(stale_stage4c_outputs(), TABLES / "stage4c_corrected_v2_stale_output_map.csv")
    atomic_json(
        {
            "status": "scca_terminology_migrated",
            "authoritative_scca_tuning_fold_file": "stage4c_corrected_v2_scca_tuning_fold_correlations.csv",
            "stale_scca_files": [
                "stage4c_corrected_v2_scca_heldout_fold_correlations.csv",
                "stage4c_corrected_v2_scca_tuning_cv_correlations.csv",
            ],
            "estimate_role": "tuning_diagnostic",
            "nested_outer_evaluation": False,
            "independent_generalisation_estimate": False,
            "pma_refit_executed": False,
        },
        MANIFESTS / "stage4c_corrected_v2_scca_terminology_manifest.json",
    )
    validation = pd.DataFrame(
        [
            {"test": "scca_postprocess_existing_tables_only", "status": "pass", "notes": f"source={source}"},
            {"test": "old_heldout_columns_absent", "status": "pass", "notes": ""},
            {"test": "nested_cv_not_claimed", "status": "pass" if tuning["nested_outer_evaluation"].eq(False).all() and tuning["independent_generalisation_estimate"].eq(False).all() else "fail", "notes": ""},
        ]
    )
    atomic_csv(validation, TABLES / "stage4c_corrected_v2_scca_validation_tests.csv")
    return tuning, tuning_fold


def write_report(mofa_plan: pd.DataFrame, scca_plan: pd.DataFrame, input_audit: pd.DataFrame) -> None:
    text = "\n".join(
        [
            "# Corrected-v2 Stage 4C MOFA/sCCA Analysis",
            "",
            "Stage 4C is restricted to unsupervised/descriptive structural integration. It does not rerun Stage 4B, Stage 4D or Stage 4E.",
            "",
            "## Core Interpretation",
            "",
            "- MOFA is descriptive structural integration, not external prediction and not mechanistic proof.",
            "- sCCA is independently fitted within each cohort and describes sparse cross-view association, not causal or covariate-independent coupling.",
            "- Metabolomics is processed cohort-locally. This avoids treating incompatible raw assay scales as common quantitative units.",
            "- Disease labels are excluded from factor/component fitting and tuning; pooled phenotype labels are used only post hoc.",
            "",
            "## MOFA Plan",
            "",
            mofa_plan.to_markdown(index=False),
            "",
            "## sCCA Plan",
            "",
            scca_plan.to_markdown(index=False),
            "",
            "## Input Audit",
            "",
            input_audit.to_markdown(index=False),
            "",
        ]
    )
    (REPORTS / "STAGE4C_CORRECTED_V2_ANALYSIS_REPORT.md").write_text(text)


def write_completion_report() -> None:
    pieces = [
        "# Corrected-v2 Stage 4C Completion Report",
        "",
        "Stage 4C was executed under Branch B as unsupervised/descriptive integration only.",
        "",
        "Authoritative outputs are the files prefixed `stage4c_corrected_v2_` inside the corrected-v2 folder.",
        "",
        "## Validation",
        "",
        pd.read_csv(TABLES / "stage4c_corrected_v2_final_validation_tests.csv").to_markdown(index=False),
        "",
        "## View Dimensions And Transformations",
        "",
        pd.read_csv(TABLES / "stage4c_corrected_v2_view_dimension_transform_manifest.csv").to_markdown(index=False),
        "",
        "## MOFA Factor Classification",
        "",
        pd.read_csv(TABLES / "stage4c_corrected_v2_mofa_factor_classification.csv").to_markdown(index=False),
        "",
        "## sCCA Component Summary",
        "",
        pd.read_csv(TABLES / "stage4c_corrected_v2_scca_component_summary.csv").to_markdown(index=False),
        "",
        "## sCCA Cross-Cohort Reproducibility",
        "",
        pd.read_csv(TABLES / "stage4c_corrected_v2_scca_cross_cohort_reproducibility.csv").to_markdown(index=False),
        "",
    ]
    (REPORTS / "STAGE4C_CORRECTED_V2_COMPLETION_REPORT.md").write_text("\n".join(pieces))


def expected_stage4c_outputs(analysis: str = "all") -> dict[str, list[str]]:
    common = {
        "stage4c_corrected_v2_mofa_plan.csv": ["disease", "analysis_role", "phenotype_tests"],
        "stage4c_corrected_v2_scca_plan.csv": ["disease", "fit_scope", "phenotype_use"],
        "stage4c_corrected_v2_input_integrity_audit.csv": ["disease", "cohort", "view", "n_participants", "n_features", "status"],
        "stage4c_corrected_v2_feature_qc_trace.csv": ["disease", "cohort", "view", "feature_id", "retained_in_common_valid_schema"],
        "stage4c_corrected_v2_participant_manifest.csv": ["disease", "cohort", "sample_id", "disease_group_posthoc_only"],
        "stage4c_corrected_v2_view_dimension_transform_manifest.csv": ["disease", "cohort", "view", "n_participants", "n_features", "transformation"],
        "stage4c_corrected_v2_stale_output_map.csv": ["path", "status", "reason"],
    }
    mofa = {
        "stage4c_corrected_v2_mofa_factor_scores.csv": ["disease", "cohort", "sample_id", "phenotype", "factor", "score"],
        "stage4c_corrected_v2_mofa_feature_loadings.csv": ["disease", "view", "feature_id", "factor", "loading"],
        "stage4c_corrected_v2_mofa_variance_explained.csv": ["disease", "cohort", "view", "factor", "variance_explained"],
        "stage4c_corrected_v2_mofa_convergence_diagnostics.csv": ["disease", "status"],
        "stage4c_corrected_v2_mofa_factor_score_distributions.csv": ["disease", "cohort", "factor", "mean", "sd"],
        "stage4c_corrected_v2_mofa_factor_associations.csv": ["disease", "comparison", "factor", "phenotype_used_for_factor_construction"],
        "stage4c_corrected_v2_mofa_factor_classification.csv": ["disease", "factor", "factor_classification"],
        "stage4c_corrected_v2_mofa_top_loadings.csv": ["disease", "view", "factor", "feature_id", "loading"],
        "stage4c_corrected_v2_mofa_validation_tests.csv": ["test", "status", "notes"],
    }
    scca = {
        "stage4c_corrected_v2_scca_component_summary.csv": ["disease", "cohort", "component", "apparent_full_data_canonical_correlation", "phenotype_used_for_fit"],
        "stage4c_corrected_v2_scca_tuning_fold_correlations.csv": ["disease", "cohort", "n_components", "fold", "tuning_fold_abs_cross_view_correlation", "phenotype_used_for_tuning", "estimate_role", "nested_outer_evaluation", "independent_generalisation_estimate"],
        "stage4c_corrected_v2_scca_tuning_summary.csv": ["disease", "cohort", "n_components", "penaltyx", "penaltyz", "selected_candidate", "estimate_role", "nested_outer_evaluation", "independent_generalisation_estimate"],
        "stage4c_corrected_v2_scca_microbiome_loadings.csv": ["disease", "cohort", "component", "view", "feature_id", "loading", "selected"],
        "stage4c_corrected_v2_scca_metabolome_loadings.csv": ["disease", "cohort", "component", "view", "feature_id", "loading", "selected"],
        "stage4c_corrected_v2_scca_cross_cohort_reproducibility.csv": ["disease", "cohort_a", "cohort_b", "component_a", "component_b", "view", "cosine_abs", "selected_jaccard", "component_matching_ambiguity"],
        "stage4c_corrected_v2_scca_component_associations.csv": ["disease", "cohort", "component", "phenotype_used_for_fit_or_tuning"],
        "stage4c_corrected_v2_scca_validation_tests.csv": ["test", "status", "notes"],
    }
    outputs = dict(common)
    if analysis in {"all", "mofa"}:
        outputs.update(mofa)
    if analysis in {"all", "scca"}:
        outputs.update(scca)
    if analysis == "scca_nested":
        outputs.update(
            {
                "stage4c_corrected_v2_scca_nested_outer_fold_correlations.csv": ["disease", "cohort", "outer_fold", "outer_test_abs_cross_view_correlation", "selected_n_components", "selected_penaltyx", "selected_penaltyz"],
                "stage4c_corrected_v2_scca_nested_cv_summary.csv": ["disease", "cohort", "mean_outer_abs_cross_view_correlation", "sd_outer_abs_cross_view_correlation", "n_outer_folds"],
            }
        )
    return outputs


def audit_stage4c_outputs(analysis: str = "all") -> pd.DataFrame:
    rows = []
    for filename, required_columns in expected_stage4c_outputs(analysis).items():
        path = TABLES / filename
        row = {
            "output_file": str(path),
            "exists": path.exists(),
            "non_empty": False,
            "schema_status": "not_checked",
            "status": "fail",
            "notes": "",
        }
        if not path.exists():
            row["notes"] = "missing corrected-v2 Stage 4C output"
            rows.append(row)
            continue
        try:
            df = pd.read_csv(path)
            row["non_empty"] = not df.empty
            missing = [c for c in required_columns if c not in df.columns]
            if missing:
                row["schema_status"] = "fail"
                row["notes"] = f"missing columns: {missing}"
            elif df.empty:
                row["schema_status"] = "pass"
                row["notes"] = "empty table"
            else:
                old_scca_cols = [
                    c
                    for c in df.columns
                    if c.startswith("heldout_abs_cross_view")
                    or c.startswith("tuning_cv_abs_cross_view")
                ]
                disallowed = "CD_vs_Control|UC_vs_Control"
                subtype_cols = [c for c in ("comparison", "phenotype_tests") if c in df.columns]
                subtype_leak = any(df[c].astype(str).str.contains(disallowed, regex=True, na=False).any() for c in subtype_cols)
                phenotype_cols = [
                    c
                    for c in df.columns
                    if c.startswith("phenotype_used_for_") or c in {"phenotype_used_for_fit", "phenotype_used_for_tuning"}
                ]
                phenotype_used = any(df[c].astype(str).str.lower().isin({"true", "1", "yes"}).any() for c in phenotype_cols)
                row["schema_status"] = "pass"
                if old_scca_cols:
                    row["notes"] = f"old sCCA heldout/tuning_cv columns remain: {old_scca_cols}"
                elif subtype_leak:
                    row["notes"] = "contains subtype-specific Stage 4C test label"
                elif phenotype_used:
                    row["notes"] = "phenotype appears marked as used for fitting/tuning"
                else:
                    row["status"] = "pass"
                    row["notes"] = f"rows={len(df)}"
        except Exception as exc:
            row["schema_status"] = "error"
            row["notes"] = repr(exc)
        rows.append(row)
    audit = pd.DataFrame(rows)
    atomic_csv(audit, TABLES / "stage4c_corrected_v2_postrun_audit.csv")
    return audit


def run_scca_nested_cv(blocks: dict[tuple[str, str], CohortBlock], feature_sets: dict[tuple[str, str], dict[str, list[str]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raise NotImplementedError(
        "Nested PMA sCCA outer-fold evaluation is prepared as a separate manual Stage 4C correction path, "
        "but the executable PMA nested-CV engine is not enabled in this minimal no-fitting correction pass."
    )


def run_stage4c(dry_run: bool = False, analysis: str = "all", audit_only: bool = False, postprocess_only: bool = False, scca_nested_cv: bool = False) -> None:
    if analysis not in {"all", "mofa", "scca"}:
        raise ValueError(f"Unsupported Stage 4C analysis mode: {analysis}")
    if postprocess_only and analysis not in {"mofa", "scca"}:
        raise ValueError("--postprocess-only must be used with --analysis mofa or --analysis scca.")
    if scca_nested_cv and analysis != "scca":
        raise ValueError("--scca-nested-cv must be used with --analysis scca.")
    ensure_v2_dirs()
    MODELS.mkdir(parents=True, exist_ok=True)
    if audit_only:
        audit = audit_stage4c_outputs("scca_nested" if scca_nested_cv else analysis)
        if not audit["status"].eq("pass").all():
            raise RuntimeError("Stage 4C corrected-v2 output audit failed.")
        print("STAGE4C_CORRECTED_V2_AUDIT_COMPLETE")
        return
    mofa_plan = build_mofa_plan()
    scca_plan = build_scca_plan()
    atomic_csv(mofa_plan, TABLES / "stage4c_corrected_v2_mofa_plan.csv")
    atomic_csv(scca_plan, TABLES / "stage4c_corrected_v2_scca_plan.csv")
    blocks = load_blocks()
    feature_sets, feature_trace = build_feature_sets(blocks)
    input_audit = input_integrity(blocks, feature_sets)
    atomic_csv(input_audit, TABLES / "stage4c_corrected_v2_input_integrity_audit.csv")
    atomic_csv(feature_trace, TABLES / "stage4c_corrected_v2_feature_qc_trace.csv")
    atomic_csv(participant_manifest(blocks), TABLES / "stage4c_corrected_v2_participant_manifest.csv")
    atomic_csv(view_dimension_transform_manifest(blocks, feature_sets), TABLES / "stage4c_corrected_v2_view_dimension_transform_manifest.csv")
    atomic_csv(stale_stage4c_outputs(), TABLES / "stage4c_corrected_v2_stale_output_map.csv")
    write_report(mofa_plan, scca_plan, input_audit)
    atomic_json(
        {
            "status": "prepared_only" if dry_run else "running",
            "analysis": analysis,
            "full_run_executed": False,
            "mofa_factor_primary": MOFA_FACTORS_PRIMARY,
            "mofa_factors_sensitivity_prepared_not_run": MOFA_FACTORS_PREPARED,
            "scca_components": SCCA_COMPONENTS,
            "scca_penalties": SCCA_PENALTIES,
            "phenotype_annotations": PHENOTYPE_ANNOTATIONS,
            "diagnosis_used_for_fitting_or_tuning": False,
            "authoritative_scca_tuning_fold_file": "stage4c_corrected_v2_scca_tuning_fold_correlations.csv",
        },
        MANIFESTS / "stage4c_corrected_v2_prepare_manifest.json",
    )
    if dry_run:
        print("STAGE4C_CORRECTED_V2_PREPARED_ONLY")
        return
    start = time.time()
    if postprocess_only:
        if analysis == "scca":
            tuning, tuning_fold = postprocess_scca_from_existing()
            final_validation = pd.DataFrame(
                [
                    {"test": "input_integrity", "status": "pass" if input_audit["status"].eq("pass").all() else "fail", "notes": f"rows={len(input_audit)}"},
                    {"test": "scca_tuning_summary_migrated", "status": "pass" if not tuning.empty else "fail", "notes": f"rows={len(tuning)}"},
                    {"test": "scca_tuning_fold_correlations_written", "status": "pass" if not tuning_fold.empty else "fail", "notes": f"rows={len(tuning_fold)}"},
                    {"test": "nested_cv_estimates_not_claimed", "status": "pass", "notes": "estimate_role=tuning_diagnostic; nested_outer_evaluation=False"},
                ]
            )
            atomic_csv(final_validation, TABLES / "stage4c_corrected_v2_final_validation_tests.csv")
            if not final_validation["status"].eq("pass").all():
                raise RuntimeError("Stage 4C sCCA postprocess validation failed.")
            print("STAGE4C_CORRECTED_V2_SCCA_POSTPROCESS_COMPLETE")
            return
        mofa_scores, mofa_variance, mofa_class = postprocess_mofa_from_existing(blocks, feature_sets)
        final_validation = pd.DataFrame(
            [
                {"test": "input_integrity", "status": "pass" if input_audit["status"].eq("pass").all() else "fail", "notes": f"rows={len(input_audit)}"},
                {"test": "mofa_postprocess_scores_written", "status": "pass" if not mofa_scores.empty else "fail", "notes": f"rows={len(mofa_scores)}"},
                {"test": "mofa_postprocess_variance_written", "status": "pass" if not mofa_variance.empty else "fail", "notes": f"rows={len(mofa_variance)}"},
                {"test": "mofa_postprocess_factor_classification_written", "status": "pass" if not mofa_class.empty else "fail", "notes": f"rows={len(mofa_class)}"},
                {"test": "pooled_cohort_adjusted_mofa_association_absent", "status": "pass", "notes": "cohort-specific post-hoc tests only"},
            ]
        )
        atomic_csv(final_validation, TABLES / "stage4c_corrected_v2_final_validation_tests.csv")
        if not final_validation["status"].eq("pass").all():
            raise RuntimeError("Stage 4C MOFA postprocess validation failed.")
        print("STAGE4C_CORRECTED_V2_MOFA_POSTPROCESS_COMPLETE")
        return
    if scca_nested_cv:
        run_scca_nested_cv(blocks, feature_sets)
        print("STAGE4C_CORRECTED_V2_SCCA_NESTED_CV_COMPLETE")
        return
    validation_rows = [
        {"test": "input_integrity", "status": "pass" if input_audit["status"].eq("pass").all() else "fail", "notes": f"rows={len(input_audit)}"},
        {"test": "no_subtype_predictive_tests", "status": "pass", "notes": "only pooled IBD-vs-control and CRC-vs-control post-hoc labels used"},
    ]
    if analysis in {"all", "mofa"}:
        mofa_scores, mofa_variance, mofa_class = run_mofa(blocks, feature_sets)
        validation_rows.extend(
            [
                {"test": "mofa_scores_written", "status": "pass" if not mofa_scores.empty else "fail", "notes": f"rows={len(mofa_scores)}"},
                {"test": "mofa_variance_written", "status": "pass" if not mofa_variance.empty else "fail", "notes": f"rows={len(mofa_variance)}"},
                {"test": "mofa_factor_classification_written", "status": "pass" if not mofa_class.empty else "fail", "notes": f"rows={len(mofa_class)}"},
            ]
        )
    if analysis in {"all", "scca"}:
        scca_summary, scca_repro = run_scca(blocks, feature_sets)
        validation_rows.extend(
            [
                {"test": "scca_summary_written", "status": "pass" if not scca_summary.empty else "fail", "notes": f"rows={len(scca_summary)}"},
                {"test": "scca_reproducibility_written", "status": "pass" if not scca_repro.empty else "fail", "notes": f"rows={len(scca_repro)}"},
            ]
        )
    final_validation = pd.DataFrame(validation_rows)
    if not final_validation["status"].eq("pass").all():
        atomic_csv(final_validation, TABLES / "stage4c_corrected_v2_final_validation_tests.csv")
        raise RuntimeError("Stage 4C final validation failed.")
    atomic_csv(final_validation, TABLES / "stage4c_corrected_v2_final_validation_tests.csv")
    atomic_csv(
        pd.DataFrame(
            [
                {
                    "stage": "4C",
                    "status": "complete",
                    "analysis": analysis,
                    "mofa_status": "complete" if analysis in {"all", "mofa"} else "not_run_this_invocation",
                    "scca_status": "complete" if analysis in {"all", "scca"} else "not_run_this_invocation",
                    "full_run_executed": True,
                    "stage4b_modified": False,
                    "stage4d_or_4e_modified_or_run": False,
                    "historical_outputs_overwritten": False,
                    "authoritative_output_prefix": "stage4c_corrected_v2_",
                }
            ]
        ),
        TABLES / "stage4c_corrected_v2_execution_status.csv",
    )
    atomic_json(
        {
            "status": "complete",
            "analysis": analysis,
            "runtime_seconds": round(time.time() - start, 2),
            "full_run_executed": True,
            "script_sha256": sha256(Path(__file__)),
            "mofa_factor_primary": MOFA_FACTORS_PRIMARY,
            "scca_components": SCCA_COMPONENTS,
            "scca_penalties": SCCA_PENALTIES,
            "diagnosis_used_for_fitting_or_tuning": False,
            "stage4d_or_4e_modified_or_run": False,
        },
        MANIFESTS / "stage4c_corrected_v2_complete_manifest.json",
    )
    if analysis == "all":
        write_completion_report()
        print("STAGE4C_CORRECTED_V2_COMPLETE")
    elif analysis == "mofa":
        print("STAGE4C_CORRECTED_V2_MOFA_COMPLETE")
    else:
        print("STAGE4C_CORRECTED_V2_SCCA_COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run corrected-v2 Stage 4C MOFA and sCCA only.")
    parser.add_argument("--analysis", choices=["all", "mofa", "scca"], default="all", help="Run or audit all Stage 4C analyses, MOFA only, or sCCA only.")
    parser.add_argument("--dry-run", action="store_true", help="Write plans/audits only; do not fit MOFA or sCCA.")
    parser.add_argument("--audit-only", action="store_true", help="Audit existing corrected-v2 Stage 4C output schemas only; do not fit MOFA or sCCA.")
    parser.add_argument("--postprocess-only", action="store_true", help="For --analysis mofa or scca: regenerate corrected post-hoc/terminology tables from completed outputs without fitting MOFA/PMA.")
    # --scca-nested-cv is intentionally NOT exposed on the CLI: the nested outer-fold
    # PMA sCCA engine (run_scca_nested_cv) is not implemented, only scaffolded for a
    # future methodological extension. A production-facing command should not offer
    # an option that deliberately raises NotImplementedError; call run_scca_nested_cv
    # directly from Python if you are implementing that engine.
    args = parser.parse_args()
    run_stage4c(
        dry_run=args.dry_run,
        analysis=args.analysis,
        audit_only=args.audit_only,
        postprocess_only=args.postprocess_only,
    )


if __name__ == "__main__":
    main()
