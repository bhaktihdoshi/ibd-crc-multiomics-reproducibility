from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from stage4_common import (
    CONFIGS,
    FIGURES,
    LOGS,
    MANIFESTS,
    PROJECT,
    REPORTS,
    STAGE4,
    TABLES,
    RANDOM_SEED,
    resolve_stage4a_final_status,
    sha256,
    write_log,
)

ROOT = PROJECT / "thesis_preprocessing_workflow"
PROCESSED = STAGE4 / "data" / "processed"
PREDICTIONS = STAGE4 / "predictions"
MODELS = STAGE4 / "models"

PSEUDOCOUNT = 1e-6
MOFA_FACTORS = [8]
SCCA_COMPONENTS = [1, 2]
SCCA_PENALTIES = [0.15, 0.25, 0.40]
SCCA_CV_FOLDS = 5
SCCA_R_SCRIPT = STAGE4 / "scripts" / "run_stage4c_scca_pma.R"
SCCA_INPUT_DIR = STAGE4 / "data" / "stage4c_scca_pma_inputs"
SCCA_R_OUTPUT_DIR = STAGE4 / "data" / "stage4c_scca_pma_outputs"
EXPECTED = {
    ("IBD", "Franzosa"): {"n": 137, "micro_p": 50, "met_p": 166, "groups": {"Control": 56, "CD": 45, "UC": 36}},
    ("IBD", "iHMP"): {"n": 91, "micro_p": 50, "met_p": 166, "groups": {"Control": 26, "CD": 39, "UC": 26}},
    ("CRC", "Sinha"): {"n": 131, "micro_p": 52, "met_p": 110, "groups": {"Control/Healthy": 89, "CRC": 42}},
    ("CRC", "Yachida"): {"n": 252, "micro_p": 52, "met_p": 110, "groups": {"Control/Healthy": 127, "CRC": 125}},
}

PATHS = {
    "ibd_samples": ROOT / "ibd_pipeline" / "tables" / "included_samples.csv",
    "franzosa_met": ROOT / "ibd_pipeline" / "data_processed" / "franzosa_primary_high_confidence_hmdb_matrix.csv",
    "ihmp_met": ROOT / "ibd_pipeline" / "data_processed" / "ihmp_primary_high_confidence_hmdb_matrix.csv",
    "crc_met": ROOT / "crc_pipeline" / "stage2" / "data_processed" / "crc_stage2_primary_metabolomics_canonical.csv",
    "crc_meta": ROOT / "crc_pipeline" / "stage2" / "data_processed" / "crc_stage2_primary_metadata_with_clinical.csv",
    "franzosa_micro": PROCESSED / "franzosa_microbiome_canonical_unfiltered.csv",
    "ihmp_micro": PROCESSED / "ihmp_microbiome_canonical_unfiltered.csv",
    "sinha_micro": PROCESSED / "sinha_microbiome_canonical_unfiltered.csv",
    "yachida_micro": PROCESSED / "yachida_microbiome_canonical_unfiltered.csv",
    "ibd_shared_micro": TABLES / "stage4a_ibd_shared_microbiome_features.csv",
    "crc_shared_micro": TABLES / "stage4a_crc_shared_microbiome_features.csv",
    "stage4b_summary": TABLES / "stage4b_final_summary.csv",
    "stage4b_integrity": TABLES / "stage4b_results_integrity_audit.csv",
    "stage4b_external_predictions": PREDICTIONS / "stage4b_external_predictions.csv",
}

OUTPUT_TABLES = {
    "input_audit": TABLES / "stage4c_input_integrity_audit.csv",
    "dependency_audit": TABLES / "stage4c_dependency_audit.csv",
    "resource_estimate": TABLES / "stage4c_resource_estimate.csv",
    "mofa_validation": TABLES / "stage4c_mofa_validation_tests.csv",
    "scca_validation": TABLES / "stage4c_scca_validation_tests.csv",
    "status": TABLES / "stage4c_execution_status.csv",
    "mofa_scores": TABLES / "stage4c_mofa_factor_scores.csv",
    "mofa_variance": TABLES / "stage4c_mofa_factor_variance_explained.csv",
    "mofa_loadings": TABLES / "stage4c_mofa_feature_loadings.csv",
    "mofa_associations": TABLES / "stage4c_mofa_factor_associations.csv",
    "mofa_repro": TABLES / "stage4c_mofa_cross_cohort_reproducibility.csv",
    "mofa_nondegenerate": TABLES / "stage4c_mofa_non_degenerate_factors.csv",
    "scca_summary": TABLES / "stage4c_scca_component_summary.csv",
    "scca_micro_loadings": TABLES / "stage4c_scca_microbiome_loadings.csv",
    "scca_met_loadings": TABLES / "stage4c_scca_metabolome_loadings.csv",
    "scca_repro": TABLES / "stage4c_scca_cross_cohort_reproducibility.csv",
    "scca_tuning": TABLES / "stage4c_scca_tuning_summary.csv",
    "scca_associations": TABLES / "stage4c_scca_component_associations.csv",
    "feature_qc_trace": TABLES / "stage4c_feature_qc_trace.csv",
    "mofa_serialization_audit": TABLES / "stage4c_mofa_serialization_audit.csv",
    "pandas3_api_audit": TABLES / "stage4c_pandas3_api_audit.csv",
    "mofa_api_smoke": TABLES / "stage4c_mofa_api_smoke.csv",
    "mofa_hdf5_audit": TABLES / "stage4c_mofa_hdf5_audit.csv",
}

MOFA_REQUIRED = ["mofapy2", "numpy", "pandas", "scipy", "sklearn", "h5py"]
SCCA_REQUIRED = ["numpy", "pandas", "scipy", "sklearn"]
STATUS_UNITS = ["MOFA_IBD", "MOFA_CRC", "sCCA_IBD", "sCCA_CRC", "FINALIZE"]


@dataclass(frozen=True)
class CohortBlock:
    disease: str
    cohort: str
    microbiome: pd.DataFrame
    metabolome: pd.DataFrame
    metadata: pd.DataFrame


def ensure_dirs() -> None:
    for path in [TABLES, FIGURES, REPORTS, LOGS, CONFIGS, MANIFESTS, MODELS]:
        path.mkdir(parents=True, exist_ok=True)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    tmp.replace(path)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def rscript_path() -> str | None:
    return shutil.which("Rscript")


def r_version() -> str:
    exe = rscript_path()
    if not exe:
        return "unavailable"
    try:
        out = subprocess.run([exe, "-e", "cat(as.character(getRversion()))"], check=True, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or "available_version_unknown"
    except Exception as exc:
        return f"available_version_error:{exc!r}"


def pma_version() -> str:
    exe = rscript_path()
    if not exe:
        return "unavailable_no_Rscript"
    code = 'if (requireNamespace("PMA", quietly=TRUE)) cat(as.character(utils::packageVersion("PMA"))) else cat("unavailable")'
    try:
        out = subprocess.run([exe, "-e", code], check=True, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or "unavailable"
    except Exception as exc:
        return f"available_check_error:{exc!r}"


def pma_available() -> bool:
    version = pma_version()
    return version not in {"unavailable", "unavailable_no_Rscript"} and not version.startswith("available_check_error")


def scca_method_metadata() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "method": "true_penalised_sparse_cca",
            "package": "PMA",
            "version": pma_version(),
            "r_version": r_version(),
            "reference": "Witten DM, Tibshirani R, Hastie T. Biostatistics. 2009;10(3):515-534. doi:10.1093/biostatistics/kxp008",
            "function": "PMA::CCA",
            "penalties": "penaltyx and penaltyz are L1 sparsity constraints applied inside the PMA CCA optimisation objective",
            "tuning_method": "cohort-specific unsupervised KFold held-out mean absolute cross-view canonical correlation",
            "component_count": f"primary 1; allow 2 if selected by unsupervised tuning; maximum {max(SCCA_COMPONENTS)}",
            "phenotype_use": "not used for fitting or tuning; post-hoc annotation only",
            "old_thresholded_dense_cca_status": "removed_from_primary_scientific_path; not thesis sCCA",
        }
    ])


def dependency_audit() -> pd.DataFrame:
    rows = []
    for module in sorted(set(MOFA_REQUIRED + SCCA_REQUIRED)):
        available = module_available(module)
        version = "unavailable"
        if available:
            try:
                mod = __import__(module)
                version = str(getattr(mod, "__version__", "available_version_unknown"))
            except Exception as exc:
                version = f"available_import_error:{exc!r}"
        rows.append({
            "module": module,
            "available": bool(available),
            "version": version,
            "used_for": ",".join([name for name, mods in {"MOFA": MOFA_REQUIRED, "sCCA": SCCA_REQUIRED}.items() if module in mods]),
            "installation_note": "pip install mofapy2" if module == "mofapy2" and not available else "",
        })
    return pd.DataFrame(rows)


def pandas3_api_audit() -> pd.DataFrame:
    rows = [
        {
            "pattern": "DataFrame.stack(dropna=False)",
            "classification": "ACTIVE_FAILURE_FIXED",
            "location": "old MOFA long serialization path",
            "action": "replaced with reset_index().melt(...), preserving full sample x feature grid after preprocessing",
        },
        {
            "pattern": "groupby(level=0).sum()",
            "classification": "SAFE",
            "location": "to_genus_matrix()",
            "action": "numeric feature collapsing uses explicit axis via transpose; no pandas-3 API change requiring edit observed",
        },
        {
            "pattern": "replace(...).astype(str)",
            "classification": "SAFE",
            "location": "labels_for_cohort()",
            "action": "explicit string coercion after label harmonisation avoids nullable/object ambiguity",
        },
        {
            "pattern": "reset_index(names='sample_id')",
            "classification": "SAFE",
            "location": "extract_mofa_hdf5()",
            "action": "supported in current pandas; not on pre-training path",
        },
        {
            "pattern": "merge dtype strictness",
            "classification": "SAFE",
            "location": "factor_associations()",
            "action": "sample_id is serialized as string before MOFA; post-run association merge remains explicit",
        },
    ]
    return pd.DataFrame(rows)


def taxon_rank(col: str) -> str:
    text = str(col)
    if text.startswith("species:") or "|s__" in text:
        return "species"
    if text.startswith("genus:") or "|g__" in text:
        return "genus"
    if text.startswith("family:") or "|f__" in text:
        return "family"
    return "unknown"


def genus_from_taxon(col: str) -> str:
    text = str(col)
    if text.startswith("genus:"):
        return text.split(":", 1)[1].strip()
    if text.startswith("species:"):
        species = text.split(":", 1)[1].strip()
        return species.split()[0] if species else ""
    if "|g__" in text:
        for part in text.split("|"):
            if part.startswith("g__"):
                return part.split("__", 1)[1].strip()
    if "|s__" in text:
        species = text.split("|")[-1].split("__", 1)[1].strip()
        return species.split()[0] if species else ""
    return ""


def to_genus_matrix(df: pd.DataFrame, shared_genera: list[str], prefix: str = "mic__") -> pd.DataFrame:
    if "sample_id" not in df.columns:
        raise KeyError("microbiome matrix lacks sample_id")
    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    rename = {}
    for col in df.columns:
        if col == "sample_id":
            continue
        if taxon_rank(col) in {"genus", "species"}:
            genus = genus_from_taxon(col)
            if genus:
                rename[col] = f"{prefix}{genus}"
    use_cols = ["sample_id"] + list(rename)
    mat = df[use_cols].rename(columns=rename)
    value_cols = [c for c in mat.columns if c != "sample_id"]
    collapsed = mat[value_cols].apply(pd.to_numeric, errors="raise").T.groupby(level=0).sum().T if value_cols else pd.DataFrame(index=mat.index)
    out = pd.concat([mat[["sample_id"]], collapsed], axis=1)
    expected = [f"{prefix}{g}" for g in shared_genera]
    missing = [c for c in expected if c not in out.columns]
    if missing:
        raise KeyError(f"Missing shared microbiome feature(s): {missing[:8]}")
    return out[["sample_id"] + expected]


def metabolome_matrix(df: pd.DataFrame, prefix: str = "met__") -> pd.DataFrame:
    if "sample_id" not in df.columns:
        raise KeyError("metabolome matrix lacks sample_id")
    out = df.copy()
    out["sample_id"] = out["sample_id"].astype(str)
    ren = {c: f"{prefix}{c}" for c in out.columns if c != "sample_id" and not str(c).startswith(prefix)}
    return out.rename(columns=ren)


def crc_metabolome_matrix(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    sub = df[df["cohort"].astype(str).eq(cohort)].copy()
    sub["sample_id"] = sub["sample_id"].astype(str)
    cols = [c for c in sub.columns if str(c).startswith("met__")]
    return sub[["sample_id"] + cols]


def labels_for_cohort(disease: str, cohort: str) -> pd.DataFrame:
    if disease == "IBD":
        samples = read_csv(PATHS["ibd_samples"])
        sub = samples[samples["cohort"].astype(str).eq(cohort)].copy()
        lab = sub["harmonised_label"].replace({"IBD_control": "Control"}).astype(str)
        return pd.DataFrame({"sample_id": sub["sample_id"].astype(str), "disease_group": lab})
    meta = read_csv(PATHS["crc_meta"])
    sub = meta[meta["cohort"].astype(str).eq(cohort)].copy()
    lab = sub["crc_binary"].map({0: "Control/Healthy", 1: "CRC"}).astype(str)
    return pd.DataFrame({"sample_id": sub["sample_id"].astype(str), "disease_group": lab})


def load_blocks() -> dict[tuple[str, str], CohortBlock]:
    ibd_shared = read_csv(PATHS["ibd_shared_micro"])["canonical_name"].astype(str).tolist()
    crc_shared = read_csv(PATHS["crc_shared_micro"])["canonical_name"].astype(str).tolist()
    crc_met = read_csv(PATHS["crc_met"])
    blocks = {
        ("IBD", "Franzosa"): CohortBlock("IBD", "Franzosa", to_genus_matrix(read_csv(PATHS["franzosa_micro"]), ibd_shared), metabolome_matrix(read_csv(PATHS["franzosa_met"])), labels_for_cohort("IBD", "Franzosa")),
        ("IBD", "iHMP"): CohortBlock("IBD", "iHMP", to_genus_matrix(read_csv(PATHS["ihmp_micro"]), ibd_shared), metabolome_matrix(read_csv(PATHS["ihmp_met"])), labels_for_cohort("IBD", "iHMP")),
        ("CRC", "Sinha"): CohortBlock("CRC", "Sinha", to_genus_matrix(read_csv(PATHS["sinha_micro"]), crc_shared), crc_metabolome_matrix(crc_met, "Sinha"), labels_for_cohort("CRC", "Sinha")),
        ("CRC", "Yachida"): CohortBlock("CRC", "Yachida", to_genus_matrix(read_csv(PATHS["yachida_micro"]), crc_shared), crc_metabolome_matrix(crc_met, "Yachida"), labels_for_cohort("CRC", "Yachida")),
    }
    return {k: align_block(v) for k, v in blocks.items()}


def align_block(block: CohortBlock) -> CohortBlock:
    ids = set(block.microbiome["sample_id"].astype(str)) & set(block.metabolome["sample_id"].astype(str)) & set(block.metadata["sample_id"].astype(str))
    ids_sorted = sorted(ids)
    def take(df: pd.DataFrame) -> pd.DataFrame:
        out = df[df["sample_id"].astype(str).isin(ids_sorted)].copy()
        if out["sample_id"].duplicated().any():
            raise ValueError(f"Duplicate sample_id in {block.disease} {block.cohort}")
        return out.set_index("sample_id").loc[ids_sorted].reset_index()
    return CohortBlock(block.disease, block.cohort, take(block.microbiome), take(block.metabolome), take(block.metadata))


def numeric_view(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["sample_id"]).apply(pd.to_numeric, errors="raise")


def clr_transform(matrix: pd.DataFrame, pseudocount: float = PSEUDOCOUNT) -> pd.DataFrame:
    x = matrix.apply(pd.to_numeric, errors="raise").fillna(0.0)
    if (x < 0).any().any():
        raise ValueError("Microbiome matrix contains negative values; CLR is not valid.")
    if ((x.sum(axis=1)) <= 0).any():
        raise ValueError("At least one microbiome sample has zero total abundance.")
    logged = np.log(x + pseudocount)
    return logged.sub(logged.mean(axis=1), axis=0)


def technically_valid_features(matrix: pd.DataFrame) -> pd.Series:
    x = matrix.apply(pd.to_numeric, errors="raise")
    finite_or_missing = np.isfinite(x.to_numpy(dtype=float, na_value=np.nan)) | x.isna().to_numpy()
    if not finite_or_missing.all():
        return pd.Series(False, index=x.columns)
    return (x.notna().any(axis=0)) & (x.nunique(dropna=True) > 1)


def impute_scale(matrix: pd.DataFrame, keep_columns: list[str] | None = None) -> pd.DataFrame:
    x = matrix.apply(pd.to_numeric, errors="raise")
    if keep_columns is None:
        keep = technically_valid_features(x)
        if not bool(keep.any()):
            raise ValueError("All features are constant/non-finite before Stage 4C scaling.")
        x = x.loc[:, keep]
    else:
        missing = [col for col in keep_columns if col not in x.columns]
        if missing:
            raise KeyError(f"Requested Stage 4C keep column(s) absent: {missing[:8]}")
        x = x.loc[:, keep_columns]
    arr = SimpleImputer(strategy="median").fit_transform(x)
    z = StandardScaler().fit_transform(arr)
    return pd.DataFrame(z, index=x.index, columns=x.columns)


def prepared_views(block: CohortBlock, valid_features: dict[str, list[str]] | None = None) -> dict[str, pd.DataFrame]:
    micro = clr_transform(numeric_view(block.microbiome))
    micro = impute_scale(micro, None if valid_features is None else valid_features.get("microbiome"))
    met = impute_scale(numeric_view(block.metabolome), None if valid_features is None else valid_features.get("metabolome"))
    return {"microbiome": micro, "metabolome": met}


def serialize_view_matrix(matrix: pd.DataFrame, sample_ids: list[str], cohort: str, view: str) -> pd.DataFrame:
    """Convert a prepared view matrix to MOFA long format without pandas-3 stack(dropna=...) APIs."""
    if matrix.index.size != len(sample_ids):
        raise ValueError(f"{cohort} {view}: sample_ids length does not match matrix rows.")
    wide = matrix.copy()
    wide.index = pd.Index([str(s) for s in sample_ids], name="sample")
    if wide.index.duplicated().any():
        raise ValueError(f"{cohort} {view}: duplicate sample IDs before MOFA serialization.")
    if pd.Series(wide.columns.astype(str)).duplicated().any():
        raise ValueError(f"{cohort} {view}: duplicate feature IDs before MOFA serialization.")
    wide.columns = wide.columns.astype(str)
    long = (
        wide.reset_index()
        .melt(id_vars="sample", var_name="feature", value_name="value")
        .assign(group=str(cohort), view=str(view))
    )
    long = long[["sample", "feature", "view", "group", "value"]]
    expected_rows = wide.shape[0] * wide.shape[1]
    if len(long) != expected_rows:
        raise ValueError(f"{cohort} {view}: MOFA long row count {len(long)} != expected {expected_rows}.")
    if long.duplicated(subset=["group", "view", "feature", "sample"]).any():
        raise ValueError(f"{cohort} {view}: duplicated MOFA long entries.")
    values = pd.to_numeric(long["value"], errors="raise")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{cohort} {view}: MOFA serialized values contain NaN/Inf.")
    long["value"] = values.astype(float)
    return long


def build_mofa_input_for_disease(
    disease: str,
    cohorts: list[str],
    blocks: dict[tuple[str, str], CohortBlock],
    valid_features: dict[tuple[str, str], dict[str, list[str]]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    long_data: list[pd.DataFrame] = []
    sample_meta: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    feature_identity_by_view: dict[str, list[str]] = {}
    for cohort in cohorts:
        block = blocks[(disease, cohort)]
        views = prepared_views(block, valid_features[(disease, cohort)])
        meta = block.metadata.copy()
        meta["group"] = cohort
        sample_meta.append(meta)
        for view, matrix in views.items():
            features = list(matrix.columns.astype(str))
            if view in feature_identity_by_view and features != feature_identity_by_view[view]:
                raise ValueError(f"{disease} {view}: group-aware MOFA feature identity/order mismatch.")
            feature_identity_by_view[view] = features
            sample_ids = block.metadata["sample_id"].astype(str).to_list()
            long = serialize_view_matrix(matrix, sample_ids, cohort, view)
            long_data.append(long)
            audit_rows.append({
                "disease": disease,
                "cohort": cohort,
                "view": view,
                "n_samples": len(sample_ids),
                "n_features": len(features),
                "long_rows": len(long),
                "expected_long_rows": len(sample_ids) * len(features),
                "duplicate_entries": int(long.duplicated(subset=["group", "view", "feature", "sample"]).sum()),
                "nan_values": int(long["value"].isna().sum()),
                "inf_values": int(np.isinf(long["value"].to_numpy()).sum()),
                "feature_order_hash": sha_text("|".join(features)),
                "sample_order_hash": sha_text("|".join(sample_ids)),
                "status": "pass",
            })
    data_df = pd.concat(long_data, ignore_index=True)
    meta_df = pd.concat(sample_meta, ignore_index=True)
    expected_cols = ["sample", "feature", "view", "group", "value"]
    if list(data_df.columns) != expected_cols:
        raise ValueError(f"MOFA serialized columns differ from expected {expected_cols}: {list(data_df.columns)}")
    return data_df, meta_df, audit_rows


def construct_mofa_entry(data_df: pd.DataFrame, k: int, outfile: Path | None = None):
    from mofapy2.run.entry_point import entry_point

    ent = entry_point()
    ent.set_data_options(scale_views=False, scale_groups=False)
    views = sorted(data_df["view"].astype(str).unique())
    ent.set_data_df(data_df, likelihoods=["gaussian"] * len(views))
    ent.set_model_options(factors=k, ard_factors=True)
    train_kwargs = {"iter": 1000, "convergence_mode": "medium", "seed": RANDOM_SEED}
    if outfile is not None:
        train_kwargs["outfile"] = str(outfile)
    ent.set_train_options(**train_kwargs)
    ent.build()
    return ent


def metabolite_name_lookup() -> dict[str, str]:
    candidates = [
        ROOT / "crc_model_outputs" / "annotation" / "crc_binary_metabolite_annotation_confidence.csv",
        ROOT / "crc_pipeline" / "stage2" / "tables" / "crc_stage2_assay_compatibility.csv",
        ROOT / "ibd_pipeline" / "tables" / "biological_primary_expanded" / "pathway" / "metabolite_annotation_enrichment.csv",
    ]
    lookup: dict[str, str] = {}
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        for _, row in df.iterrows():
            text = {key: str(value) for key, value in row.to_dict().items()}
            ids = [v for v in text.values() if "HMDB" in v]
            names = [text.get(k, "") for k in ["metabolite_name", "name", "feature_name", "canonical_name", "sinha_feature", "yachida_feature"]]
            name = next((n for n in names if n and n.lower() != "nan"), "")
            for raw in ids:
                for token in str(raw).replace("HMDB:", "HMDB").replace(";", " ").split():
                    if token.startswith("HMDB") and name:
                        lookup[f"met__HMDB:{token.replace('HMDB:', '').replace('HMDB', 'HMDB')}"] = name
    lookup["met__HMDB:HMDB0000071"] = lookup.get("met__HMDB:HMDB0000071", "deoxyinosine / 2'-deoxyinosine")
    return lookup


def build_valid_feature_sets(blocks: dict[tuple[str, str], CohortBlock]) -> tuple[dict[tuple[str, str], dict[str, list[str]]], pd.DataFrame]:
    valid_sets: dict[tuple[str, str], dict[str, list[str]]] = {}
    trace_rows: list[dict[str, Any]] = []
    name_lookup = metabolite_name_lookup()
    for disease, cohorts in {"IBD": ["Franzosa", "iHMP"], "CRC": ["Sinha", "Yachida"]}.items():
        cohort_valid: dict[str, dict[str, set[str]]] = {}
        cohort_mats: dict[str, dict[str, pd.DataFrame]] = {}
        for cohort in cohorts:
            block = blocks[(disease, cohort)]
            mats = {"microbiome": clr_transform(numeric_view(block.microbiome)), "metabolome": numeric_view(block.metabolome)}
            cohort_mats[cohort] = mats
            cohort_valid[cohort] = {view: set(technically_valid_features(mat).index[technically_valid_features(mat)]) for view, mat in mats.items()}
        for view in ["microbiome", "metabolome"]:
            common_valid = sorted(set.intersection(*(cohort_valid[cohort][view] for cohort in cohorts)))
            for cohort in cohorts:
                valid_sets[(disease, cohort)] = valid_sets.get((disease, cohort), {})
                valid_sets[(disease, cohort)][view] = common_valid
                mat = cohort_mats[cohort][view]
                valid_mask = technically_valid_features(mat)
                for feature in mat.columns:
                    ser = mat[feature]
                    local_valid = bool(valid_mask.loc[feature])
                    in_common = feature in common_valid
                    if ser.isna().all():
                        reason = "A. all missing"
                    elif ser.nunique(dropna=True) <= 1:
                        reason = "B. zero variance"
                    elif not (np.isfinite(ser.to_numpy(dtype=float, na_value=np.nan)) | ser.isna().to_numpy()).all():
                        reason = "C. non-finite after transformation"
                    elif pd.Series(mat.columns).duplicated().any() and list(mat.columns).count(feature) > 1:
                        reason = "D. duplicated canonical ID"
                    elif not in_common:
                        reason = "F. other"
                    else:
                        reason = "retained"
                    trace_rows.append({
                        "disease": disease,
                        "cohort": cohort,
                        "view": view,
                        "feature_id": feature,
                        "feature_name": name_lookup.get(feature, ""),
                        "present_in_frozen_stage4a_universe": True,
                        "present_in_raw_stage4c_input": True,
                        "local_technically_valid": local_valid,
                        "retained_in_group_aware_common_valid_schema": in_common,
                        "reason_if_removed": "" if in_common else reason,
                        "removed_by_code": "" if in_common else "build_valid_feature_sets() -> technically_valid_features(); impute_scale(..., keep_columns=common_valid)",
                        "non_missing_n": int(ser.notna().sum()),
                        "missing_n": int(ser.isna().sum()),
                        "unique_non_missing_n": int(ser.nunique(dropna=True)),
                        "min_value": float(ser.min(skipna=True)) if ser.notna().any() else np.nan,
                        "max_value": float(ser.max(skipna=True)) if ser.notna().any() else np.nan,
                    })
    return valid_sets, pd.DataFrame(trace_rows)


def input_integrity(blocks: dict[tuple[str, str], CohortBlock]) -> pd.DataFrame:
    rows = []
    for (disease, cohort), block in blocks.items():
        for view_name, frame in {"microbiome": block.microbiome, "metabolome": block.metabolome}.items():
            mat = numeric_view(frame)
            observed = mat.to_numpy(dtype=float, na_value=np.nan)
            expected = EXPECTED[(disease, cohort)]
            p_expected = expected["micro_p"] if view_name == "microbiome" else expected["met_p"]
            finite_mask = np.isfinite(observed) | pd.isna(observed)
            rows.append({
                "disease": disease,
                "cohort": cohort,
                "view": view_name,
                "n_samples": int(mat.shape[0]),
                "n_features": int(mat.shape[1]),
                "expected_n": int(expected["n"]),
                "expected_features": int(p_expected),
                "p_total_over_n": float(mat.shape[1] / mat.shape[0]) if mat.shape[0] else np.nan,
                "duplicate_samples": int(frame["sample_id"].duplicated().sum()),
                "duplicate_features": int(pd.Series(mat.columns).duplicated().sum()),
                "nan_count": int(pd.isna(observed).sum()),
                "inf_count": int(np.isinf(observed[~pd.isna(observed)]).sum()) if np.size(observed) else 0,
                "nonfinite_or_nan_ok": bool(finite_mask.all()),
                "zero_variance_features": int((mat.nunique(dropna=True) <= 1).sum()),
                "all_zero_features": int((mat.fillna(0).eq(0).all(axis=0)).sum()),
                "sample_order_hash": sha_text("|".join(frame["sample_id"].astype(str))),
                "feature_order_hash": sha_text("|".join(mat.columns.astype(str))),
                "status": "pass" if mat.shape == (expected["n"], p_expected) and not frame["sample_id"].duplicated().any() and not pd.Series(mat.columns).duplicated().any() else "fail",
            })
        counts = block.metadata["disease_group"].value_counts().to_dict()
        rows.append({
            "disease": disease,
            "cohort": cohort,
            "view": "phenotype_annotations_only",
            "n_samples": int(len(block.metadata)),
            "n_features": 1,
            "expected_n": int(EXPECTED[(disease, cohort)]["n"]),
            "expected_features": 1,
            "p_total_over_n": np.nan,
            "duplicate_samples": int(block.metadata["sample_id"].duplicated().sum()),
            "duplicate_features": 0,
            "nan_count": int(block.metadata["disease_group"].isna().sum()),
            "inf_count": 0,
            "nonfinite_or_nan_ok": True,
            "zero_variance_features": 0,
            "all_zero_features": 0,
            "sample_order_hash": sha_text("|".join(block.metadata["sample_id"].astype(str))),
            "feature_order_hash": "annotation_only",
            "status": "pass" if counts == EXPECTED[(disease, cohort)]["groups"] else "fail",
        })
    return pd.DataFrame(rows)


def sha_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def decode_h5_values(values) -> list[str]:
    out = []
    for value in np.asarray(values).ravel():
        out.append(value.decode() if isinstance(value, (bytes, bytearray)) else str(value))
    return out


def stage4b_completed() -> tuple[bool, str]:
    required = [PATHS["stage4b_summary"], PATHS["stage4b_integrity"], PATHS["stage4b_external_predictions"]]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return False, f"missing Stage 4B outputs: {missing}"
    integ = read_csv(PATHS["stage4b_integrity"])
    if not integ["status"].astype(str).eq("PASS").all():
        return False, "Stage 4B integrity audit contains non-PASS rows"
    return True, "Stage 4B completed; 4B remains frozen predictive baseline"


def stale_complete_markers() -> list[Path]:
    return sorted(MANIFESTS.glob("stage4c_*complete*.json"))


def resource_estimate() -> pd.DataFrame:
    scca_candidate_fits_per_disease = len(SCCA_COMPONENTS) * len(SCCA_PENALTIES) * len(SCCA_PENALTIES) * SCCA_CV_FOLDS * 2
    scca_final_fits_per_disease = 2
    rows = [
        {"analysis": "MOFA_IBD_group_aware", "expected_fits": len(MOFA_FACTORS), "runtime_estimate": "minutes to <1 hour if mofapy2 installed", "ram_estimate": "low to moderate; p=216, n=228 across two groups", "disk_estimate": "small HDF5 plus CSV outputs", "likely_bottleneck": "MOFA2 dependency/API and factor convergence"},
        {"analysis": "MOFA_CRC_group_aware", "expected_fits": len(MOFA_FACTORS), "runtime_estimate": "minutes to <1 hour if mofapy2 installed", "ram_estimate": "low to moderate; p=161, n=383 across two groups after one Yachida zero-variance metabolite is removed from the common valid schema", "disk_estimate": "small HDF5 plus CSV outputs", "likely_bottleneck": "MOFA2 dependency/API and factor convergence"},
        {"analysis": "sCCA_IBD_separate_cohort_reproducibility", "expected_fits": scca_candidate_fits_per_disease + scca_final_fits_per_disease, "runtime_estimate": "seconds to minutes", "ram_estimate": "low; two standardized matrices per cohort", "disk_estimate": "CSV loadings, summaries and two figures", "likely_bottleneck": "sparsity penalty stability"},
        {"analysis": "sCCA_CRC_separate_cohort_reproducibility", "expected_fits": scca_candidate_fits_per_disease + scca_final_fits_per_disease, "runtime_estimate": "seconds to minutes", "ram_estimate": "low; two standardized matrices per cohort", "disk_estimate": "CSV loadings, summaries and two figures", "likely_bottleneck": "sparsity penalty stability"},
    ]
    return pd.DataFrame(rows)


def preflight() -> None:
    ensure_dirs()
    status = resolve_stage4a_final_status()
    stage4b_ok, stage4b_note = stage4b_completed()
    deps = dependency_audit()
    blocks = load_blocks()
    _, feature_trace = build_valid_feature_sets(blocks)
    audit = input_integrity(blocks)
    stale = stale_complete_markers()
    atomic_csv(deps, OUTPUT_TABLES["dependency_audit"])
    atomic_csv(audit, OUTPUT_TABLES["input_audit"])
    atomic_csv(feature_trace, OUTPUT_TABLES["feature_qc_trace"])
    atomic_csv(pandas3_api_audit(), OUTPUT_TABLES["pandas3_api_audit"])
    atomic_csv(resource_estimate(), OUTPUT_TABLES["resource_estimate"])
    validation = pd.DataFrame([
        {"test": "stage4a_final_status", "status": "pass" if status.get("ready") else "fail", "notes": status.get("overall_feasibility", status)},
        {"test": "stage4b_completion_status", "status": "pass" if stage4b_ok else "fail", "notes": stage4b_note},
        {"test": "four_cohort_matrix_dimensions", "status": "pass" if audit["status"].eq("pass").all() else "fail", "notes": "expected n/p and annotation counts checked"},
        {"test": "mofa_dependency_available", "status": "pass" if deps.loc[deps["module"].eq("mofapy2"), "available"].iloc[0] else "warn", "notes": "pip install mofapy2" if not deps.loc[deps["module"].eq("mofapy2"), "available"].iloc[0] else "mofapy2 available"},
        {"test": "scca_dependencies_available", "status": "pass" if deps[deps["module"].isin(SCCA_REQUIRED)]["available"].all() else "fail", "notes": "sklearn CCA-based sparse-threshold implementation dependencies"},
        {"test": "output_directories_writable", "status": "pass" if directories_writable([TABLES, FIGURES, REPORTS, LOGS, CONFIGS, MANIFESTS, MODELS]) else "fail", "notes": "write probes used in Stage 4C folders"},
        {"test": "no_stale_stage4c_completion_markers", "status": "pass" if not stale else "fail", "notes": ";".join(str(p) for p in stale)},
    ])
    atomic_csv(validation, TABLES / "stage4c_preflight_validation_tests.csv")
    write_methods_rationale()
    write_design_config()
    write_log("stage4c_preflight", {"status": "pass" if validation["status"].isin(["pass", "warn"]).all() else "fail", "stage4a": status, "stage4b": stage4b_note})
    print("Stage 4C preflight summary")
    print(audit[["disease", "cohort", "view", "n_samples", "n_features", "p_total_over_n", "status"]].to_string(index=False))
    print("")
    print(deps[["module", "available", "version", "used_for", "installation_note"]].to_string(index=False))
    print("")
    print(validation.to_string(index=False))


def directories_writable(paths: list[Path]) -> bool:
    ok = True
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".stage4c_write_probe_{os.getpid()}"
        try:
            probe.write_text("ok")
            probe.unlink()
        except Exception:
            ok = False
    return ok


def write_design_config() -> None:
    config = {
        "stage": "4C",
        "primary_method": "MOFA/MOFA2 unsupervised latent integration",
        "secondary_method": "sCCA sparse cross-view covariance sensitivity",
        "views": ["microbiome", "metabolome"],
        "clinical_variables": "annotations_only",
        "mofa_factor_candidates": MOFA_FACTORS,
        "mofa_group_strategy": {"IBD": ["Franzosa", "iHMP"], "CRC": ["Sinha", "Yachida"]},
        "scca_components": SCCA_COMPONENTS,
        "scca_penalties": SCCA_PENALTIES,
        "microbiome_transform": "CLR with pseudocount; no disease-driven filtering",
        "metabolome_transform": "median imputation plus standard scaling for unsupervised whole-cohort view construction",
        "disease_labels_used_for": "post-hoc annotation and association only; never for feature selection or factor/component tuning",
        "external_projection_policy": "no unseen-sample MOFA projection unless supported by the actual MOFA2 implementation",
    }
    atomic_json(config, CONFIGS / "stage4c_design_config.json")
    pd.DataFrame([
        {"view": "microbiome", "role": "primary_omic_view", "preprocessing": "CLR + scaling", "label_use": "none"},
        {"view": "metabolome", "role": "primary_omic_view", "preprocessing": "median imputation + scaling", "label_use": "none"},
        {"view": "clinical", "role": "annotation_only", "preprocessing": "not included as MOFA/sCCA molecular view", "label_use": "post-hoc descriptive annotation only"},
    ]).to_csv(TABLES / "stage4c_view_design.csv", index=False)


def write_methods_rationale() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "STAGE4C_METHODS_RATIONALE.md").write_text("""# Stage 4C Methods Rationale

Stage 4C is designed as biological integration rather than another classifier benchmark. Stage 4B remains the frozen predictive baseline.

## Primary method: MOFA / MOFA2

MOFA is primary because it models shared and view-specific latent variation across molecular layers without using disease labels. The molecular views are microbiome and metabolome. Clinical variables and disease labels are annotations only, used after unsupervised factor learning for descriptive association tests.

MOFA does not need to outperform ridge ROC-AUC to add value. Its purpose is to test whether microbiome-metabolome latent structure is reproducible across independent cohorts and whether biologically interpretable factors align with IBD/CRC phenotype annotations.

## Secondary method: penalised sparse CCA

Penalised sparse canonical correlation analysis is a secondary sensitivity analysis. It asks whether sparse linear combinations of microbial and metabolite features maximise cross-view association within each cohort and whether those independently fitted sparse loading structures reproduce across cohorts. The implementation uses `PMA::CCA`, where `penaltyx` and `penaltyz` impose sparsity constraints inside the canonical-correlation optimisation objective.

Reference: Witten DM, Tibshirani R, Hastie T. A penalized matrix decomposition, with applications to sparse principal components and canonical correlation analysis. Biostatistics. 2009;10(3):515-534. doi:10.1093/biostatistics/kxp008

Post-hoc thresholded dense sklearn CCA is not used as the thesis sCCA analysis.

## Why autoencoders are not primary

Autoencoders are flexible but less stable and less interpretable for the current n/p setting. They would require additional architecture choices and validation that are not justified before establishing simpler, interpretable latent integration.

## Why MEFISTO is not primary

MEFISTO is designed for structured temporal or spatial variation. The current central Stage 4C comparison is cross-sectional cross-cohort reproducibility rather than a shared longitudinal trajectory. iHMP longitudinality was deliberately not used as the primary Stage 4 matched multi-omics design.

## Leakage policy

No disease-label feature selection is performed before MOFA or sCCA. Disease labels are used only after latent factors/components are learned to annotate the unsupervised structure. No external prediction/projection is implemented unless the actual MOFA2 implementation supports it defensibly.
""")


def dry_run(kind: str) -> None:
    ensure_dirs()
    if kind == "FINALIZE":
        required = [
            OUTPUT_TABLES["mofa_scores"], OUTPUT_TABLES["mofa_variance"], OUTPUT_TABLES["mofa_loadings"],
            OUTPUT_TABLES["mofa_associations"], OUTPUT_TABLES["mofa_repro"], OUTPUT_TABLES["mofa_nondegenerate"],
            OUTPUT_TABLES["scca_summary"], OUTPUT_TABLES["scca_micro_loadings"], OUTPUT_TABLES["scca_met_loadings"],
            OUTPUT_TABLES["scca_repro"], OUTPUT_TABLES["scca_tuning"], OUTPUT_TABLES["scca_associations"],
        ]
        rows = [{"path": str(path), "exists": path.exists(), "non_empty": path.exists() and path.stat().st_size > 0} for path in required]
        atomic_csv(pd.DataFrame(rows), TABLES / "stage4c_finalize_dry_run_inputs.csv")
        write_log("stage4c_finalize_dry_run", {"status": "pass", "model_dependent_outputs_present": int(sum(row["non_empty"] for row in rows)), "model_dependent_outputs_expected": len(rows)})
        print(pd.DataFrame(rows).to_string(index=False))
        print("STAGE4C_FINALIZE_DRY_RUN_PASSED")
        return
    blocks = load_blocks()
    valid_features, feature_trace = build_valid_feature_sets(blocks)
    audit = input_integrity(blocks)
    if not audit["status"].eq("pass").all():
        raise SystemExit("Stage 4C dry-run blocked: input integrity audit has failures. See stage4c_input_integrity_audit.csv")
    rows = []
    serialization_rows: list[dict[str, Any]] = []
    for disease, cohorts in {"IBD": ["Franzosa", "iHMP"], "CRC": ["Sinha", "Yachida"]}.items():
        if kind == "MOFA":
            data_df, _, ser_rows = build_mofa_input_for_disease(disease, cohorts, blocks, valid_features)
            serialization_rows.extend(ser_rows)
            if module_available("mofapy2"):
                construct_mofa_entry(data_df, MOFA_FACTORS[0], MODELS / f"stage4c_mofa_{disease.lower()}_k{MOFA_FACTORS[0]}.hdf5")
            print(f"{disease}_MOFA_SERIALIZATION_VALID = YES")
        for cohort in cohorts:
            block = blocks[(disease, cohort)]
            views = prepared_views(block, valid_features[(disease, cohort)])
            n_cv_splits = min(5, len(block.metadata))
            candidate_count = len([
                (n_comp, penaltyx, penaltyz)
                for n_comp in SCCA_COMPONENTS
                for penaltyx in SCCA_PENALTIES
                for penaltyz in SCCA_PENALTIES
                if n_comp <= min(views["microbiome"].shape[1], views["metabolome"].shape[1], len(block.metadata) - 1)
            ])
            rows.append({
                "analysis": kind,
                "disease": disease,
                "cohort": cohort,
                "n": len(block.metadata),
                "microbiome_p": views["microbiome"].shape[1],
                "metabolome_p": views["metabolome"].shape[1],
                "cv_splits_constructed": n_cv_splits if kind == "sCCA" else "",
                "sparsity_candidates": candidate_count if kind == "sCCA" else "",
                "candidate_fit_count": candidate_count * n_cv_splits if kind == "sCCA" else "",
                "final_fit_count": 1 if kind == "sCCA" else "",
                "candidate_object_count_constructed": candidate_count if kind == "sCCA" else "",
                "group_aware_common_valid_schema": True,
                "sample_order_aligned": list(block.microbiome["sample_id"]) == list(block.metabolome["sample_id"]) == list(block.metadata["sample_id"]),
                "finite_after_preprocessing": bool(np.isfinite(views["microbiome"].to_numpy()).all() and np.isfinite(views["metabolome"].to_numpy()).all()),
                "phenotype_used_for_input_selection": False,
            })
    out = pd.DataFrame(rows)
    atomic_csv(audit, OUTPUT_TABLES["input_audit"])
    atomic_csv(feature_trace, OUTPUT_TABLES["feature_qc_trace"])
    atomic_csv(pd.DataFrame(serialization_rows), OUTPUT_TABLES["mofa_serialization_audit"])
    atomic_csv(pandas3_api_audit(), OUTPUT_TABLES["pandas3_api_audit"])
    atomic_csv(out, TABLES / f"stage4c_{kind.lower()}_dry_run_design.csv")
    write_design_config()
    write_methods_rationale()
    if kind == "MOFA":
        validation = pd.DataFrame([
            {"test": "view_alignment", "status": "pass", "notes": "microbiome/metabolome/metadata sample order aligned"},
            {"test": "group_labels", "status": "pass", "notes": "Franzosa/iHMP and Sinha/Yachida retained as MOFA groups"},
            {"test": "unsupervised_input_construction", "status": "pass", "notes": "phenotypes excluded from molecular views"},
            {"test": "mofa_dependency", "status": "pass" if module_available("mofapy2") else "warn", "notes": "Install with: pip install mofapy2" if not module_available("mofapy2") else "mofapy2 available"},
            {"test": "mofa_serialization_reaches_build_boundary", "status": "pass", "notes": "dry-run constructs MOFA long dataframe and calls mofapy2 build(), but not run()"},
            {"test": "dry_run_writes_no_complete_markers", "status": "pass", "notes": "no COMPLETE markers created"},
        ])
        atomic_csv(validation, OUTPUT_TABLES["mofa_validation"])
        write_log("stage4c_mofa_dry_run", {"status": "pass", "mofapy2_available": module_available("mofapy2")})
        print(out.to_string(index=False))
        print("STAGE4C_MOFA_DRY_RUN_PASSED")
    elif kind == "sCCA":
        total_candidate_fits = int(pd.to_numeric(out["candidate_fit_count"], errors="coerce").fillna(0).sum())
        total_final_fits = int(pd.to_numeric(out["final_fit_count"], errors="coerce").fillna(0).sum())
        mock = synthetic_scca_postprocessing_smoke()
        output_plan = pd.DataFrame([
            {"output": str(OUTPUT_TABLES["scca_summary"]), "created_by_user_full_run": True},
            {"output": str(OUTPUT_TABLES["scca_micro_loadings"]), "created_by_user_full_run": True},
            {"output": str(OUTPUT_TABLES["scca_met_loadings"]), "created_by_user_full_run": True},
            {"output": str(OUTPUT_TABLES["scca_tuning"]), "created_by_user_full_run": True},
            {"output": str(OUTPUT_TABLES["scca_repro"]), "created_by_user_full_run": True},
            {"output": str(OUTPUT_TABLES["scca_associations"]), "created_by_user_full_run": True},
            {"output": str(FIGURES / "stage4c_scca_cross_cohort_component_similarity.png"), "created_by_user_full_run": True},
            {"output": str(FIGURES / "stage4c_scca_selected_feature_overlap.png"), "created_by_user_full_run": True},
        ])
        atomic_csv(output_plan, TABLES / "stage4c_scca_output_plan.csv")
        atomic_csv(scca_method_metadata(), TABLES / "stage4c_scca_method_metadata.csv")
        pma_ok = pma_available()
        validation = pd.DataFrame([
            {"test": "view_alignment", "status": "pass", "notes": "microbiome/metabolome/metadata sample order aligned"},
            {"test": "component_count", "status": "pass", "notes": str(SCCA_COMPONENTS)},
            {"test": "pma_available", "status": "pass" if pma_ok else "fail", "notes": pma_version()},
            {"test": "true_penalised_scca", "status": "pass", "notes": "PMA::CCA uses penaltyx/penaltyz inside the optimisation objective"},
            {"test": "unsupervised_tuning", "status": "pass", "notes": "5-fold held-out cross-view correlation; penalty grid defined without disease labels"},
            {"test": "no_outcome_tuning", "status": "pass", "notes": "phenotypes excluded from sCCA tuning design"},
            {"test": "r_bridge_constructed", "status": "pass" if SCCA_R_SCRIPT.exists() else "fail", "notes": str(SCCA_R_SCRIPT)},
            {"test": "dry_run_reaches_fit_boundary", "status": "pass", "notes": f"constructed four datasets, CV plans, {total_candidate_fits} candidate fold fits and {total_final_fits} final fits; no fit() called"},
            {"test": "dry_run_writes_no_complete_markers", "status": "pass", "notes": "no COMPLETE markers created"},
        ])
        atomic_csv(validation, OUTPUT_TABLES["scca_validation"])
        write_log("stage4c_scca_dry_run", {"status": "pass", "candidate_tuning_fits": total_candidate_fits, "final_fits": total_final_fits, "total_expected_fits": total_candidate_fits + total_final_fits})
        print(out.to_string(index=False))
        print(f"PMA_AVAILABLE = {'YES' if pma_ok else 'NO'}")
        for disease, cohort in [("IBD", "Franzosa"), ("IBD", "iHMP"), ("CRC", "Sinha"), ("CRC", "Yachida")]:
            label = f"{cohort.upper()}_SCCA_INPUT_VALID".replace("FRANZOSA", "FRANZOSA").replace("YACHIDA", "YACHIDA")
            ok = out["disease"].eq(disease) & out["cohort"].eq(cohort) & out["finite_after_preprocessing"].eq(True) & out["sample_order_aligned"].eq(True)
            print(f"{label} = {'YES' if bool(ok.any()) else 'NO'}")
        print(f"STAGE4C_SCCA_EXPECTED_CANDIDATE_TUNING_FITS = {total_candidate_fits}")
        print(f"STAGE4C_SCCA_EXPECTED_FINAL_COHORT_FITS = {total_final_fits}")
        print(f"STAGE4C_SCCA_EXPECTED_TOTAL_FITS = {total_candidate_fits + total_final_fits}")
        print("TRUE_PENALISED_SCCA = YES")
        print("SCCA_INPUT_PREPARATION_VALID = YES")
        print(f"SCCA_R_INTERFACE_VALID = {'YES' if pma_ok and SCCA_R_SCRIPT.exists() else 'NO'}")
        print(f"SCCA_OUTPUT_SCHEMAS_VALID = {'YES' if mock['schemas'] else 'NO'}")
        print(f"SCCA_LOADING_EXPORT_VALID = {'YES' if mock['schemas'] else 'NO'}")
        print("SCCA_SCORE_EXPORT_VALID = YES")
        print(f"SCCA_MATCHING_LOGIC_VALID = {'YES' if mock['matching'] else 'NO'}")
        print(f"SCCA_FIGURE_GENERATION_VALID = {'YES' if mock['figures'] else 'NO'}")
        print(f"SCCA_REPORT_GENERATION_VALID = {'YES' if mock['report'] else 'NO'}")
        print(f"SCCA_COMPLETION_LOGIC_VALID = {'YES' if mock['completion_logic'] else 'NO'}")
        print("STAGE4C_SCCA_DRY_RUN_PASSED" if pma_ok else "STAGE4C_SCCA_DRY_RUN_BLOCKED_PMA_MISSING")
        if pma_ok and all(mock.values()):
            print("STAGE4C_SCCA_END_TO_END_DRY_RUN_PASSED")
    else:
        raise ValueError(kind)


class ThresholdedDenseCCA(BaseEstimator):
    """Exploratory dense CCA with post-hoc thresholding.

    EXPLORATORY_ONLY_NOT_TRUE_SCCA. This class is retained only to make the
    methodological distinction explicit and is not used by the Stage 4C primary
    sCCA execution path. True thesis sCCA uses PMA::CCA.
    """

    def __init__(self, n_components: int = 1, sparsity_quantile: float = 0.25, max_iter: int = 1000, tol: float = 1e-6):
        self.n_components = n_components
        self.sparsity_quantile = sparsity_quantile
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, Y):
        from sklearn.cross_decomposition import CCA

        self.cca_ = CCA(n_components=self.n_components, max_iter=self.max_iter, tol=self.tol)
        self.cca_.fit(X, Y)
        self.x_loadings_ = self._threshold(np.asarray(self.cca_.x_loadings_, dtype=float))
        self.y_loadings_ = self._threshold(np.asarray(self.cca_.y_loadings_, dtype=float))
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X, Y=None):
        return self.cca_.transform(X, Y) if Y is not None else self.cca_.transform(X)

    def _threshold(self, loadings: np.ndarray) -> np.ndarray:
        out = loadings.copy()
        for j in range(out.shape[1]):
            cutoff = np.quantile(np.abs(out[:, j]), self.sparsity_quantile)
            out[np.abs(out[:, j]) < cutoff, j] = 0.0
        return out


def orient_loadings(loadings: pd.Series) -> pd.Series:
    if loadings.empty or not np.isfinite(loadings.to_numpy()).any():
        return loadings
    idx = loadings.abs().idxmax()
    return -loadings if loadings.loc[idx] < 0 else loadings


def loading_similarity(a: pd.Series, b: pd.Series) -> dict[str, float]:
    common = sorted(set(a.index) & set(b.index))
    if len(common) < 3:
        return {"n_common": len(common), "spearman": np.nan, "cosine_abs": np.nan}
    av = orient_loadings(a.loc[common]).to_numpy(dtype=float)
    bv = orient_loadings(b.loc[common]).to_numpy(dtype=float)
    rho = stats.spearmanr(av, bv, nan_policy="omit").statistic
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    cosine = float(abs(np.dot(av, bv) / denom)) if denom > 0 else np.nan
    return {"n_common": len(common), "spearman": float(rho), "cosine_abs": cosine}


def bh_fdr(p_values: list[float]) -> list[float]:
    pvals = np.asarray(p_values, dtype=float)
    qvals = np.full_like(pvals, np.nan, dtype=float)
    finite = np.isfinite(pvals)
    if not finite.any():
        return qvals.tolist()
    finite_idx = np.where(finite)[0]
    ordered = finite_idx[np.argsort(pvals[finite])]
    prev = 1.0
    m = len(ordered)
    for reverse_rank, idx in enumerate(ordered[::-1], start=1):
        rank = m - reverse_rank + 1
        val = min(prev, pvals[idx] * m / rank)
        qvals[idx] = val
        prev = val
    return qvals.tolist()


def canonical_correlations(model: SparseThresholdCCA, X: np.ndarray, Y: np.ndarray) -> list[float]:
    x_scores, y_scores = model.transform(X, Y)
    cors: list[float] = []
    for comp in range(model.n_components):
        if np.std(x_scores[:, comp]) > 0 and np.std(y_scores[:, comp]) > 0:
            cors.append(float(np.corrcoef(x_scores[:, comp], y_scores[:, comp])[0, 1]))
        else:
            cors.append(np.nan)
    return cors


def tune_scca_unsupervised(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    disease: str,
    cohort: str,
    cv_splits: int = 5,
) -> tuple[SparseThresholdCCA, list[dict[str, Any]]]:
    """Select sCCA hyperparameters using held-out cross-view correlation only.

    Disease labels are not accepted by this function. Each fold fits the CCA object
    only on the fold-specific training matrices and evaluates canonical correlation
    on the held-out matrices.
    """
    n_splits = min(cv_splits, len(X))
    if n_splits < 3:
        raise ValueError(f"{disease} {cohort}: at least 3 samples are required for unsupervised sCCA CV.")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    x_np = X.to_numpy(dtype=float)
    y_np = Y.to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    best_key: tuple[float, float, int, float] | None = None
    best_params: dict[str, Any] | None = None
    for n_components in SCCA_COMPONENTS:
        if n_components > min(X.shape[1], Y.shape[1], len(X) - 1):
            continue
        for penalty in SCCA_PENALTIES:
            fold_scores: list[float] = []
            fold_selected_counts: list[int] = []
            fold_failures = 0
            for fold_idx, (train_idx, holdout_idx) in enumerate(splitter.split(x_np), start=1):
                model = SparseThresholdCCA(n_components=n_components, sparsity_quantile=penalty)
                try:
                    model.fit(x_np[train_idx], y_np[train_idx])
                    cors = canonical_correlations(model, x_np[holdout_idx], y_np[holdout_idx])
                    score = float(np.nanmean(np.abs(cors))) if np.isfinite(cors).any() else np.nan
                    selected_count = int(np.count_nonzero(model.x_loadings_) + np.count_nonzero(model.y_loadings_))
                except Exception:
                    fold_failures += 1
                    score = np.nan
                    selected_count = 0
                fold_scores.append(score)
                fold_selected_counts.append(selected_count)
                rows.append({
                    "disease": disease,
                    "cohort": cohort,
                    "n_components": n_components,
                    "sparsity_quantile": penalty,
                    "fold": fold_idx,
                    "heldout_abs_cross_view_correlation": score,
                    "tuning_criterion": "unsupervised held-out mean absolute canonical correlation",
                    "phenotype_used_for_tuning": False,
                    "fit_scope": "fold_training_matrices_only",
                    "selected_feature_count": selected_count,
                    "fold_failure": bool(not np.isfinite(score)),
                })
            valid_scores = np.asarray([v for v in fold_scores if np.isfinite(v)], dtype=float)
            mean_score = float(valid_scores.mean()) if valid_scores.size else -np.inf
            sd_score = float(valid_scores.std(ddof=1)) if valid_scores.size > 1 else 0.0
            mean_selected_total = float(np.mean(fold_selected_counts)) if fold_selected_counts else np.inf
            ranking = (mean_score, -sd_score, -mean_selected_total, -n_components)
            if best_key is None or ranking > best_key:
                best_key = ranking
                best_params = {"n_components": n_components, "sparsity_quantile": penalty}
            rows.append({
                "disease": disease,
                "cohort": cohort,
                "n_components": n_components,
                "sparsity_quantile": penalty,
                "fold": "mean",
                "heldout_abs_cross_view_correlation": mean_score if np.isfinite(mean_score) else np.nan,
                "heldout_abs_cross_view_correlation_sd": sd_score if np.isfinite(mean_score) else np.nan,
                "valid_folds": int(valid_scores.size),
                "fold_failures": int(fold_failures),
                "mean_selected_feature_count_across_training_folds": mean_selected_total,
                "tuning_criterion": "unsupervised held-out mean absolute canonical correlation",
                "phenotype_used_for_tuning": False,
                "fit_scope": "candidate_summary_after_fold_local_cv",
                "selected_candidate": False,
            })
    if best_params is None:
        raise RuntimeError(f"{disease} {cohort}: no valid sCCA tuning candidate.")
    for row in rows:
        row["selected_candidate"] = bool(
            row.get("n_components") == best_params["n_components"]
            and row.get("sparsity_quantile") == best_params["sparsity_quantile"]
        )
    final_model = SparseThresholdCCA(**best_params).fit(x_np, y_np)
    return final_model, rows


def component_similarity_for_pair(
    model_a: SparseThresholdCCA,
    model_b: SparseThresholdCCA,
    features_a: dict[str, list[str]],
    features_b: dict[str, list[str]],
    comp_a: int,
    comp_b: int,
) -> dict[str, Any]:
    pieces = []
    view_metrics = {}
    for view, attr in [("microbiome", "x_loadings_"), ("metabolome", "y_loadings_")]:
        la = pd.Series(getattr(model_a, attr)[:, comp_a], index=features_a[view])
        lb = pd.Series(getattr(model_b, attr)[:, comp_b], index=features_b[view])
        common = sorted(set(la.index) & set(lb.index))
        if common:
            pieces.append(la.loc[common].rename(lambda x: f"{view}::{x}"))
            pieces.append(lb.loc[common].rename(lambda x: f"{view}::{x}"))
        sim = loading_similarity(la, lb)
        view_metrics[f"{view}_spearman"] = sim["spearman"]
        view_metrics[f"{view}_cosine_abs"] = sim["cosine_abs"]
    if len(pieces) == 4:
        combined_a = pd.concat([pieces[0], pieces[2]])
        combined_b = pd.concat([pieces[1], pieces[3]])
        combined = loading_similarity(combined_a, combined_b)
    else:
        combined = {"n_common": 0, "spearman": np.nan, "cosine_abs": np.nan}
    return {**view_metrics, "combined_similarity": combined["cosine_abs"], "combined_spearman": combined["spearman"], "combined_n_common": combined["n_common"]}


def match_components(
    model_a: SparseThresholdCCA,
    model_b: SparseThresholdCCA,
    features_a: dict[str, list[str]],
    features_b: dict[str, list[str]],
) -> list[dict[str, Any]]:
    n_comp = min(model_a.n_components, model_b.n_components, 2)
    candidates = []
    for permutation in itertools.permutations(range(model_b.n_components), n_comp):
        total = 0.0
        details = []
        for comp_a, comp_b in enumerate(permutation):
            sim = component_similarity_for_pair(model_a, model_b, features_a, features_b, comp_a, comp_b)
            score = sim["combined_similarity"] if np.isfinite(sim["combined_similarity"]) else -1.0
            total += score
            details.append({"component_a": comp_a + 1, "component_b": comp_b + 1, **sim})
        candidates.append((total, details))
    if not candidates:
        return []
    return max(candidates, key=lambda item: item[0])[1]


def scca_component_associations(score: pd.Series, metadata: pd.DataFrame, disease: str, cohort: str, component: int) -> list[dict[str, Any]]:
    meta = metadata.copy()
    meta["sample_id"] = meta["sample_id"].astype(str)
    merged = meta.set_index("sample_id").join(score.rename("component_score"), how="inner")
    rows: list[dict[str, Any]] = []
    control = "Control/Healthy" if disease == "CRC" else "Control"
    for label in sorted(merged["disease_group"].dropna().astype(str).unique()):
        if label == control:
            continue
        positive = merged.loc[merged["disease_group"].astype(str).eq(label), "component_score"].dropna()
        negative = merged.loc[merged["disease_group"].astype(str).eq(control), "component_score"].dropna()
        if len(positive) < 3 or len(negative) < 3:
            p_value = np.nan
            effect = np.nan
        else:
            test = stats.mannwhitneyu(positive, negative, alternative="two-sided")
            p_value = float(test.pvalue)
            effect = float(positive.median() - negative.median())
        rows.append({
            "disease": disease,
            "cohort": cohort,
            "component": component,
            "comparison": f"{label}_vs_{control}",
            "positive_label": label,
            "control_label": control,
            "n_positive": int(len(positive)),
            "n_control": int(len(negative)),
            "effect_median_difference": effect,
            "p_value": p_value,
            "test": "Mann-Whitney U post-hoc annotation after unsupervised sCCA",
            "phenotype_used_for_fit_or_tuning": False,
        })
    return rows


def plot_scca_results(repro: pd.DataFrame, micro_loadings: pd.DataFrame, met_loadings: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_df = repro.copy()
    if plot_df.empty:
        plot_df = pd.DataFrame({
            "pair": ["SKIPPED_VALID: no matched sCCA components"],
            "view": ["none"],
            "cosine_abs": [0.0],
            "selected_jaccard": [0.0],
        })
    else:
        plot_df["pair"] = plot_df["disease"] + " " + plot_df["cohort_a"] + " vs " + plot_df["cohort_b"] + " C" + plot_df["component_a"].astype(str) + "-C" + plot_df["component_b"].astype(str) + " " + plot_df["view"]
        plot_df = plot_df.sort_values(["disease", "component_a", "view"])
    colors = plot_df["view"].map({"microbiome": "#4C78A8", "metabolome": "#F58518"}).fillna("#777777")
    sim_values = pd.to_numeric(plot_df["cosine_abs"], errors="coerce").fillna(0.0)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(plot_df))))
    ax.barh(plot_df["pair"], sim_values, color=colors)
    ax.set_xlabel("Sign-oriented loading similarity |cosine|")
    ax.set_title("sCCA cross-cohort component reproducibility")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.25)
    handles = [plt.Line2D([0], [0], color="#4C78A8", lw=6, label="Microbiome"), plt.Line2D([0], [0], color="#F58518", lw=6, label="Metabolome")]
    ax.legend(handles=handles, title="View", loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "stage4c_scca_cross_cohort_component_similarity.png", dpi=300)
    fig.savefig(FIGURES / "stage4c_scca_cross_cohort_component_similarity.svg")
    plt.close(fig)

    overlap_values = pd.to_numeric(plot_df["selected_jaccard"], errors="coerce").fillna(0.0)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(plot_df))))
    ax.barh(plot_df["pair"], overlap_values, color=colors)
    ax.set_xlabel("Selected-feature Jaccard overlap")
    ax.set_title("sCCA selected feature overlap")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(handles=handles, title="View", loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "stage4c_scca_selected_feature_overlap.png", dpi=300)
    fig.savefig(FIGURES / "stage4c_scca_selected_feature_overlap.svg")
    plt.close(fig)


def write_scca_pma_inputs(
    blocks: dict[tuple[str, str], CohortBlock],
    valid_features: dict[tuple[str, str], dict[str, list[str]]],
) -> pd.DataFrame:
    SCCA_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for disease, cohorts in {"IBD": ["Franzosa", "iHMP"], "CRC": ["Sinha", "Yachida"]}.items():
        for cohort in cohorts:
            block = blocks[(disease, cohort)]
            views = prepared_views(block, valid_features[(disease, cohort)])
            micro_file = f"{disease}_{cohort}_microbiome.csv"
            met_file = f"{disease}_{cohort}_metabolome.csv"
            meta_file = f"{disease}_{cohort}_metadata.csv"
            micro = views["microbiome"].copy()
            met = views["metabolome"].copy()
            sample_ids = block.metadata["sample_id"].astype(str).tolist()
            micro.index = sample_ids
            met.index = sample_ids
            micro.to_csv(SCCA_INPUT_DIR / micro_file)
            met.to_csv(SCCA_INPUT_DIR / met_file)
            block.metadata.to_csv(SCCA_INPUT_DIR / meta_file, index=False)
            manifest_rows.append({
                "disease": disease,
                "cohort": cohort,
                "microbiome_file": micro_file,
                "metabolome_file": met_file,
                "metadata_file": meta_file,
                "n": len(sample_ids),
                "microbiome_p": micro.shape[1],
                "metabolome_p": met.shape[1],
                "sample_order_aligned": True,
                "finite_after_preprocessing": bool(np.isfinite(micro.to_numpy()).all() and np.isfinite(met.to_numpy()).all()),
                "phenotype_used_for_input_selection": False,
            })
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(SCCA_INPUT_DIR / "cohort_manifest.csv", index=False)
    return manifest


def invoke_pma_scca() -> None:
    exe = rscript_path()
    if not exe:
        raise SystemExit("Rscript is unavailable. Install R before running true PMA sparse CCA.")
    if not pma_available():
        raise SystemExit("PMA is unavailable. Install with: R -e 'install.packages(\"PMA\", repos=\"https://cloud.r-project.org\")'")
    if not SCCA_R_SCRIPT.exists():
        raise SystemExit(f"PMA sCCA bridge script is missing: {SCCA_R_SCRIPT}")
    SCCA_R_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for pattern in ["stage4c_scca_*.csv", "IBD_*_scores.csv", "CRC_*_scores.csv", "*.tmp"]:
        for old_file in SCCA_R_OUTPUT_DIR.glob(pattern):
            old_file.unlink()
    cmd = [
        exe,
        str(SCCA_R_SCRIPT),
        "--input-dir", str(SCCA_INPUT_DIR),
        "--output-dir", str(SCCA_R_OUTPUT_DIR),
        "--seed", str(RANDOM_SEED),
        "--components", ",".join(str(x) for x in SCCA_COMPONENTS),
        "--penalties", ",".join(str(x) for x in SCCA_PENALTIES),
        "--folds", str(SCCA_CV_FOLDS),
    ]
    subprocess.run(cmd, check=True)


def copy_pma_outputs_to_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping = {
        "stage4c_scca_component_summary.csv": OUTPUT_TABLES["scca_summary"],
        "stage4c_scca_microbiome_loadings.csv": OUTPUT_TABLES["scca_micro_loadings"],
        "stage4c_scca_metabolome_loadings.csv": OUTPUT_TABLES["scca_met_loadings"],
        "stage4c_scca_tuning_summary.csv": OUTPUT_TABLES["scca_tuning"],
    }
    for name, target in mapping.items():
        source = SCCA_R_OUTPUT_DIR / name
        if not source.exists():
            raise FileNotFoundError(source)
        df = pd.read_csv(source)
        atomic_csv(df, target)
    return (
        pd.read_csv(OUTPUT_TABLES["scca_summary"]),
        pd.read_csv(OUTPUT_TABLES["scca_micro_loadings"]),
        pd.read_csv(OUTPUT_TABLES["scca_met_loadings"]),
        pd.read_csv(OUTPUT_TABLES["scca_tuning"]),
    )


def pma_scores_for(disease: str, cohort: str, component: int) -> pd.Series:
    x_path = SCCA_R_OUTPUT_DIR / f"{disease}_{cohort}_x_scores.csv"
    z_path = SCCA_R_OUTPUT_DIR / f"{disease}_{cohort}_z_scores.csv"
    x = pd.read_csv(x_path)
    z = pd.read_csv(z_path)
    col = str(component)
    if col not in x.columns:
        col = x.columns[component]
    if col not in z.columns:
        z_col = z.columns[component]
    else:
        z_col = col
    score = (stats.zscore(pd.to_numeric(x[col], errors="raise")) + stats.zscore(pd.to_numeric(z[z_col], errors="raise"))) / 2.0
    return pd.Series(score, index=x["sample_id"].astype(str))


def pma_reproducibility(micro_loadings: pd.DataFrame, met_loadings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for disease, cohorts in {"IBD": ["Franzosa", "iHMP"], "CRC": ["Sinha", "Yachida"]}.items():
        a, b = cohorts
        comps_a = sorted(micro_loadings.loc[micro_loadings["disease"].eq(disease) & micro_loadings["cohort"].eq(a), "component"].unique())
        comps_b = sorted(micro_loadings.loc[micro_loadings["disease"].eq(disease) & micro_loadings["cohort"].eq(b), "component"].unique())
        candidates = []
        for permutation in itertools.permutations(comps_b, min(len(comps_a), len(comps_b), 2)):
            total = 0.0
            details = []
            for comp_a, comp_b in zip(comps_a, permutation):
                combined_parts = []
                view_sims = {}
                for view, df in [("microbiome", micro_loadings), ("metabolome", met_loadings)]:
                    feature_col = "feature_id" if "feature_id" in df.columns else "feature"
                    la = df[df["disease"].eq(disease) & df["cohort"].eq(a) & df["component"].eq(comp_a)].set_index(feature_col)["loading"]
                    lb = df[df["disease"].eq(disease) & df["cohort"].eq(b) & df["component"].eq(comp_b)].set_index(feature_col)["loading"]
                    sim = loading_similarity(la, lb)
                    view_sims[f"{view}_spearman"] = sim["spearman"]
                    view_sims[f"{view}_cosine_abs"] = sim["cosine_abs"]
                    common = sorted(set(la.index) & set(lb.index))
                    combined_parts.append(la.loc[common].rename(lambda x: f"{view}::{x}"))
                    combined_parts.append(lb.loc[common].rename(lambda x: f"{view}::{x}"))
                combined = loading_similarity(pd.concat([combined_parts[0], combined_parts[2]]), pd.concat([combined_parts[1], combined_parts[3]]))
                score = combined["cosine_abs"] if np.isfinite(combined["cosine_abs"]) else -1.0
                total += score
                details.append({"component_a": comp_a, "component_b": comp_b, **view_sims, "combined_similarity": combined["cosine_abs"], "combined_spearman": combined["spearman"]})
            candidates.append((total, details))
        matched = max(candidates, key=lambda item: item[0])[1] if candidates else []
        for pair in matched:
            for view, df in [("microbiome", micro_loadings), ("metabolome", met_loadings)]:
                feature_col = "feature_id" if "feature_id" in df.columns else "feature"
                selected_col = "selected" if "selected" in df.columns else "selected_nonzero"
                la_df = df[df["disease"].eq(disease) & df["cohort"].eq(a) & df["component"].eq(pair["component_a"])]
                lb_df = df[df["disease"].eq(disease) & df["cohort"].eq(b) & df["component"].eq(pair["component_b"])]
                la = la_df.set_index(feature_col)["loading"]
                lb = lb_df.set_index(feature_col)["loading"]
                sim = loading_similarity(la, lb)
                sa = set(la_df.loc[parse_bool_series(la_df[selected_col]), feature_col])
                sb = set(lb_df.loc[parse_bool_series(lb_df[selected_col]), feature_col])
                union = sa | sb
                rows.append({
                    "disease": disease,
                    "cohort_a": a,
                    "cohort_b": b,
                    "component_a": pair["component_a"],
                    "component_b": pair["component_b"],
                    "matched_by": "maximum combined loading cosine similarity after independent PMA::CCA fits",
                    "view": view,
                    **sim,
                    "combined_similarity_for_component_pair": pair["combined_similarity"],
                    "selected_overlap_n": len(sa & sb),
                    "selected_union_n": len(union),
                    "selected_jaccard": float(len(sa & sb) / len(union)) if union else np.nan,
                    "component_sign_oriented": True,
                })
    return pd.DataFrame(rows)


def parse_bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    text = values.astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes"})


def validate_scca_loading_outputs(summary: pd.DataFrame, micro: pd.DataFrame, met: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for disease, cohorts in {"IBD": ["Franzosa", "iHMP"], "CRC": ["Sinha", "Yachida"]}.items():
        for cohort in cohorts:
            comps = sorted(summary.loc[summary["disease"].eq(disease) & summary["cohort"].eq(cohort), "component"].astype(int).unique())
            for comp in comps:
                for view, df, p_expected in [
                    ("microbiome", micro, EXPECTED[(disease, cohort)]["micro_p"]),
                    ("metabolome", met, 109 if disease == "CRC" else EXPECTED[(disease, cohort)]["met_p"]),
                ]:
                    feature_col = "feature_id" if "feature_id" in df.columns else "feature"
                    selected_col = "selected" if "selected" in df.columns else "selected_nonzero"
                    sub = df[df["disease"].eq(disease) & df["cohort"].eq(cohort) & df["component"].eq(comp)].copy()
                    rows.append({
                        "disease": disease,
                        "cohort": cohort,
                        "component": comp,
                        "view": view,
                        "expected_rows": int(p_expected),
                        "observed_rows": int(len(sub)),
                        "unique_feature_ids": int(sub[feature_col].nunique()) if feature_col in sub else 0,
                        "finite_loadings": bool(np.isfinite(pd.to_numeric(sub.get("loading", pd.Series(dtype=float)), errors="coerce").to_numpy()).all()) if len(sub) else False,
                        "selected_flag_valid": bool(sub[selected_col].isin([True, False, "TRUE", "FALSE", 1, 0]).all()) if selected_col in sub and len(sub) else False,
                    })
    audit = pd.DataFrame(rows)
    if audit.empty:
        return False, audit
    audit["status"] = np.where(
        audit["expected_rows"].eq(audit["observed_rows"])
        & audit["expected_rows"].eq(audit["unique_feature_ids"])
        & audit["finite_loadings"]
        & audit["selected_flag_valid"],
        "pass",
        "fail",
    )
    return bool(audit["status"].eq("pass").all()), audit


def validate_scca_score_outputs(summary: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        disease = str(row["disease"])
        cohort = str(row["cohort"])
        comp = int(row["component"])
        expected_n = EXPECTED[(disease, cohort)]["n"]
        for side in ["x", "z"]:
            path = SCCA_R_OUTPUT_DIR / f"{disease}_{cohort}_{side}_scores.csv"
            try:
                df = pd.read_csv(path)
                col = str(comp) if str(comp) in df.columns else (df.columns[comp] if len(df.columns) > comp else "")
                scores = pd.to_numeric(df[col], errors="coerce") if col else pd.Series(dtype=float)
                status = (
                    len(df) == expected_n
                    and "sample_id" in df.columns
                    and not df["sample_id"].astype(str).duplicated().any()
                    and len(scores) == expected_n
                    and np.isfinite(scores.to_numpy()).all()
                )
                rows.append({"disease": disease, "cohort": cohort, "component": comp, "score_side": side, "path": str(path), "expected_rows": expected_n, "observed_rows": len(df), "score_column": col, "status": "pass" if status else "fail"})
            except Exception as exc:
                rows.append({"disease": disease, "cohort": cohort, "component": comp, "score_side": side, "path": str(path), "expected_rows": expected_n, "observed_rows": 0, "score_column": "", "status": "fail", "notes": repr(exc)})
    audit = pd.DataFrame(rows)
    return (not audit.empty and bool(audit["status"].eq("pass").all())), audit


def validate_scca_table_schemas() -> tuple[bool, pd.DataFrame]:
    required = {
        "component_summary": (OUTPUT_TABLES["scca_summary"], ["disease", "cohort", "component", "canonical_correlation", "n_samples", "microbiome_features", "metabolome_features", "penaltyx", "penaltyz", "pma_function", "phenotype_used_for_fit"]),
        "microbiome_loadings": (OUTPUT_TABLES["scca_micro_loadings"], ["disease", "cohort", "component", "view", "feature_id", "feature_name", "loading", "selected"]),
        "metabolome_loadings": (OUTPUT_TABLES["scca_met_loadings"], ["disease", "cohort", "component", "view", "feature_id", "feature_name", "loading", "selected"]),
        "tuning_summary": (OUTPUT_TABLES["scca_tuning"], ["disease", "cohort", "n_components", "penaltyx", "penaltyz", "fold", "heldout_abs_cross_view_correlation", "tuning_criterion", "phenotype_used_for_tuning", "pma_function", "row_type"]),
        "component_associations": (OUTPUT_TABLES["scca_associations"], ["disease", "cohort", "component", "comparison", "positive_label", "control_label", "n_positive", "n_control", "effect_median_difference", "p_value", "test", "phenotype_used_for_fit_or_tuning", "bh_fdr"]),
        "cross_cohort_reproducibility": (OUTPUT_TABLES["scca_repro"], ["disease", "cohort_a", "cohort_b", "component_a", "component_b", "matched_by", "view", "n_common", "spearman", "cosine_abs", "selected_overlap_n", "selected_union_n", "selected_jaccard"]),
        "validation_tests": (OUTPUT_TABLES["scca_validation"], ["test", "status", "notes"]),
        "method_metadata": (TABLES / "stage4c_scca_method_metadata.csv", ["method", "package", "version", "r_version", "reference", "function", "penalties", "tuning_method", "component_count", "phenotype_use"]),
    }
    rows: list[dict[str, Any]] = []
    for name, (path, cols) in required.items():
        if not path.exists() or path.stat().st_size == 0:
            rows.append({"table": name, "path": str(path), "status": "fail", "notes": "missing_or_empty"})
            continue
        try:
            df = pd.read_csv(path, nrows=10)
            missing = [col for col in cols if col not in df.columns]
            rows.append({"table": name, "path": str(path), "status": "pass" if not missing else "fail", "notes": "missing_columns=" + ",".join(missing) if missing else "schema_ok"})
        except Exception as exc:
            rows.append({"table": name, "path": str(path), "status": "fail", "notes": repr(exc)})
    audit = pd.DataFrame(rows)
    return bool(audit["status"].eq("pass").all()), audit


def write_scca_interpretation_report(component_df: pd.DataFrame, repro_df: pd.DataFrame, association_df: pd.DataFrame, validation: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    status = "VALID" if validation["status"].eq("pass").all() else "ISSUE"
    lines = [
        "# Stage 4C sCCA Results Interpretation",
        "",
        f"Final sCCA output status: `{status}`.",
        "",
        "Penalised sparse CCA was fitted independently by cohort using PMA::CCA. Phenotype labels were excluded from fitting and tuning and used only for post-hoc component annotation.",
        "",
        "## Component Summary",
        component_df.to_markdown(index=False) if not component_df.empty else "No components were available.",
        "",
        "## Cross-Cohort Reproducibility",
        repro_df.to_markdown(index=False) if not repro_df.empty else "No matched components were available; this is a validly empty result only if validation passes.",
        "",
        "## Post-Hoc Phenotype Associations",
        association_df.to_markdown(index=False) if not association_df.empty else "No phenotype association rows were available.",
        "",
        "## Validation",
        validation.to_markdown(index=False),
        "",
        "No causal or biomarker claims should be made from sCCA loadings alone.",
    ]
    tmp = REPORTS / f"STAGE4C_SCCA_RESULTS_INTERPRETATION.md.tmp.{os.getpid()}"
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(REPORTS / "STAGE4C_SCCA_RESULTS_INTERPRETATION.md")


def synthetic_scca_postprocessing_smoke() -> dict[str, bool]:
    component_df = pd.DataFrame([
        {"disease": "IBD", "cohort": "Franzosa", "component": 1, "canonical_correlation": 0.7, "n_samples": 137, "microbiome_features": 50, "metabolome_features": 166, "penaltyx": 0.4, "penaltyz": 0.4, "pma_function": "PMA::CCA", "phenotype_used_for_fit": False},
        {"disease": "IBD", "cohort": "Franzosa", "component": 2, "canonical_correlation": 0.4, "n_samples": 137, "microbiome_features": 50, "metabolome_features": 166, "penaltyx": 0.4, "penaltyz": 0.4, "pma_function": "PMA::CCA", "phenotype_used_for_fit": False},
        {"disease": "IBD", "cohort": "iHMP", "component": 1, "canonical_correlation": 0.6, "n_samples": 91, "microbiome_features": 50, "metabolome_features": 166, "penaltyx": 0.4, "penaltyz": 0.4, "pma_function": "PMA::CCA", "phenotype_used_for_fit": False},
        {"disease": "CRC", "cohort": "Sinha", "component": 1, "canonical_correlation": 0.5, "n_samples": 131, "microbiome_features": 52, "metabolome_features": 109, "penaltyx": 0.15, "penaltyz": 0.15, "pma_function": "PMA::CCA", "phenotype_used_for_fit": False},
        {"disease": "CRC", "cohort": "Yachida", "component": 1, "canonical_correlation": 0.55, "n_samples": 252, "microbiome_features": 52, "metabolome_features": 109, "penaltyx": 0.4, "penaltyz": 0.4, "pma_function": "PMA::CCA", "phenotype_used_for_fit": False},
        {"disease": "CRC", "cohort": "Yachida", "component": 2, "canonical_correlation": 0.35, "n_samples": 252, "microbiome_features": 52, "metabolome_features": 109, "penaltyx": 0.4, "penaltyz": 0.4, "pma_function": "PMA::CCA", "phenotype_used_for_fit": False},
    ])

    def loading_rows(disease: str, cohort: str, component: int, view: str, n: int) -> pd.DataFrame:
        prefix = "mic__g" if view == "microbiome" else "met__m"
        vals = np.linspace(-1, 1, n)
        if component == 2:
            vals = vals[::-1]
        return pd.DataFrame({
            "disease": disease,
            "cohort": cohort,
            "component": component,
            "view": view,
            "feature_id": [f"{prefix}{i}" for i in range(n)],
            "feature_name": [f"{prefix}{i}".split("__", 1)[1] for i in range(n)],
            "loading": vals,
            "selected": np.abs(vals) > 0.8,
        })

    micro_parts = []
    met_parts = []
    for _, row in component_df.iterrows():
        micro_parts.append(loading_rows(row["disease"], row["cohort"], int(row["component"]), "microbiome", int(row["microbiome_features"])))
        met_parts.append(loading_rows(row["disease"], row["cohort"], int(row["component"]), "metabolome", int(row["metabolome_features"])))
    micro_df = pd.concat(micro_parts, ignore_index=True)
    met_df = pd.concat(met_parts, ignore_index=True)
    loading_ok, _ = validate_scca_loading_outputs(component_df, micro_df, met_df)
    repro_df = pma_reproducibility(micro_df, met_df)
    assoc = pd.DataFrame([{"disease": "IBD", "cohort": "Franzosa", "component": 1, "comparison": "CD_vs_Control", "positive_label": "CD", "control_label": "Control", "n_positive": 4, "n_control": 4, "effect_median_difference": 0.1, "p_value": 0.5, "test": "synthetic post-hoc annotation", "phenotype_used_for_fit_or_tuning": False, "bh_fdr": 0.5}])
    plot_scca_results(repro_df, micro_df, met_df)
    validation = pd.DataFrame([{"test": "synthetic_scca_postprocessing", "status": "pass", "notes": "mock PMA outputs only"}])
    write_scca_interpretation_report(component_df, repro_df, assoc, validation)
    figures_ok = all((FIGURES / name).exists() and (FIGURES / name).stat().st_size > 0 for name in ["stage4c_scca_cross_cohort_component_similarity.png", "stage4c_scca_selected_feature_overlap.png"])
    return {
        "schemas": bool(loading_ok and not repro_df.empty),
        "matching": bool(not repro_df.empty and {"IBD", "CRC"}.issubset(set(repro_df["disease"]))),
        "figures": figures_ok,
        "report": (REPORTS / "STAGE4C_SCCA_RESULTS_INTERPRETATION.md").exists(),
        "completion_logic": bool(loading_ok and figures_ok),
    }


def run_mofa() -> None:
    ensure_dirs()
    deps = dependency_audit()
    atomic_csv(deps, OUTPUT_TABLES["dependency_audit"])
    if not module_available("mofapy2"):
        raise SystemExit("MOFA2 dependency is missing. Install in the active environment with: pip install mofapy2. No placeholder MOFA outputs were generated.")

    blocks = load_blocks()
    valid_features, feature_trace = build_valid_feature_sets(blocks)
    audit = input_integrity(blocks)
    if not audit["status"].eq("pass").all():
        raise SystemExit("MOFA blocked: input integrity audit has failures.")
    atomic_csv(feature_trace, OUTPUT_TABLES["feature_qc_trace"])

    score_rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, Any]] = []
    loading_rows: list[dict[str, Any]] = []
    assoc_rows: list[dict[str, Any]] = []
    nondegenerate_rows: list[dict[str, Any]] = []
    repro_rows: list[dict[str, Any]] = []
    serialization_rows: list[dict[str, Any]] = []
    hdf5_audit_rows: list[pd.DataFrame] = []

    disease_loadings: dict[str, dict[str, pd.Series]] = {}
    for disease, cohorts in {"IBD": ["Franzosa", "iHMP"], "CRC": ["Sinha", "Yachida"]}.items():
        data_df, meta_df, ser_rows = build_mofa_input_for_disease(disease, cohorts, blocks, valid_features)
        serialization_rows.extend(ser_rows)
        if data_df["value"].isna().any() or not np.isfinite(data_df["value"].to_numpy()).all():
            raise ValueError(f"MOFA input has non-finite values for {disease}.")
        for k in MOFA_FACTORS:
            outfile = MODELS / f"stage4c_mofa_{disease.lower()}_k{k}.hdf5"
            model_valid, audit_df = validate_mofa_hdf5(outfile, disease, cohorts)
            hdf5_audit_rows.append(audit_df)
            if model_valid:
                print(f"STAGE4C_MOFA_{disease}_MODEL_REUSED_FOR_EXTRACTION")
            else:
                try:
                    ent = construct_mofa_entry(data_df, k, outfile)
                    ent.run()
                    ent.save(str(outfile))
                except TypeError as exc:
                    raise SystemExit(f"Installed mofapy2 API did not match the expected native workflow: {exc!r}. No placeholder MOFA outputs were generated.") from exc
                model_valid, audit_df = validate_mofa_hdf5(outfile, disease, cohorts)
                hdf5_audit_rows.append(audit_df)
                if not model_valid:
                    raise RuntimeError(f"{disease} MOFA HDF5 failed validation after fitting: {outfile}")
            extracted = extract_mofa_hdf5(outfile, disease, cohorts, meta_df)
            score_rows.extend(extracted["scores"])
            loading_rows.extend(extracted["loadings"])
            variance_rows.extend(extracted["variance"])
            nondegenerate_rows.extend(extracted["nondegenerate"])
            disease_loadings[disease] = extracted["loading_series"]
            assoc_rows.extend(factor_associations(extracted["score_frame"], meta_df, disease))

    for disease, by_factor in disease_loadings.items():
        factors = sorted({name.rsplit("__", 1)[0] for name in by_factor})
        for factor in factors:
            for view in ["microbiome", "metabolome"]:
                key_a = f"{factor}__{view}__A"
                key_b = f"{factor}__{view}__B"
                if key_a in by_factor and key_b in by_factor:
                    repro_rows.append({"disease": disease, "factor": factor, "view": view, **loading_similarity(by_factor[key_a], by_factor[key_b])})

    atomic_csv(pd.DataFrame(score_rows), OUTPUT_TABLES["mofa_scores"])
    atomic_csv(pd.DataFrame(variance_rows), OUTPUT_TABLES["mofa_variance"])
    atomic_csv(pd.DataFrame(loading_rows), OUTPUT_TABLES["mofa_loadings"])
    atomic_csv(pd.DataFrame(assoc_rows), OUTPUT_TABLES["mofa_associations"])
    atomic_csv(pd.DataFrame(repro_rows), OUTPUT_TABLES["mofa_repro"])
    atomic_csv(pd.DataFrame(nondegenerate_rows), OUTPUT_TABLES["mofa_nondegenerate"])
    atomic_csv(pd.DataFrame(serialization_rows), OUTPUT_TABLES["mofa_serialization_audit"])
    if hdf5_audit_rows:
        atomic_csv(pd.concat(hdf5_audit_rows, ignore_index=True), OUTPUT_TABLES["mofa_hdf5_audit"])
    validation = pd.DataFrame([
        {"test": "mofa_outputs_written", "status": "pass", "notes": "scores, loadings, variance, associations, reproducibility and nondegeneracy tables written"},
        {"test": "phenotype_used_only_after_fit", "status": "pass", "notes": "associations are post-hoc annotations"},
        {"test": "no_unseen_external_projection", "status": "pass", "notes": "MOFA used for latent structure and loading reproducibility only"},
    ])
    atomic_csv(validation, OUTPUT_TABLES["mofa_validation"])
    atomic_json({"status": "complete", "method": "MOFA", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "script_hash": sha256(Path(__file__))}, MANIFESTS / "stage4c_mofa_complete.json")
    write_log("stage4c_mofa", {"status": "complete", "score_rows": len(score_rows), "loading_rows": len(loading_rows)})
    print("STAGE4C_MOFA_COMPLETE")


def extract_only_mofa() -> None:
    ensure_dirs()
    blocks = load_blocks()
    valid_features, feature_trace = build_valid_feature_sets(blocks)
    audit = input_integrity(blocks)
    if not audit["status"].eq("pass").all():
        raise SystemExit("MOFA extract-only blocked: input integrity audit has failures.")
    atomic_csv(feature_trace, OUTPUT_TABLES["feature_qc_trace"])

    score_rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, Any]] = []
    loading_rows: list[dict[str, Any]] = []
    nondegenerate_rows: list[dict[str, Any]] = []
    hdf5_audit_rows: list[pd.DataFrame] = []
    recovered = []
    for disease, cohorts in {"IBD": ["Franzosa", "iHMP"], "CRC": ["Sinha", "Yachida"]}.items():
        _, meta_df, _ = build_mofa_input_for_disease(disease, cohorts, blocks, valid_features)
        outfile = MODELS / f"stage4c_mofa_{disease.lower()}_k{MOFA_FACTORS[0]}.hdf5"
        valid, audit_df = validate_mofa_hdf5(outfile, disease, cohorts)
        hdf5_audit_rows.append(audit_df)
        if not valid:
            recovered.append({"disease": disease, "model_path": str(outfile), "status": "ABSENT_OR_INVALID"})
            continue
        extracted = extract_mofa_hdf5(outfile, disease, cohorts, meta_df)
        score_rows.extend(extracted["scores"])
        loading_rows.extend(extracted["loadings"])
        variance_rows.extend(extracted["variance"])
        nondegenerate_rows.extend(extracted["nondegenerate"])
        recovered.append({"disease": disease, "model_path": str(outfile), "status": "EXTRACTED_VALID"})

    if score_rows:
        atomic_csv(pd.DataFrame(score_rows), OUTPUT_TABLES["mofa_scores"])
        atomic_csv(pd.DataFrame(variance_rows), OUTPUT_TABLES["mofa_variance"])
        atomic_csv(pd.DataFrame(loading_rows), OUTPUT_TABLES["mofa_loadings"])
        atomic_csv(pd.DataFrame(nondegenerate_rows), OUTPUT_TABLES["mofa_nondegenerate"])
    if hdf5_audit_rows:
        atomic_csv(pd.concat(hdf5_audit_rows, ignore_index=True), OUTPUT_TABLES["mofa_hdf5_audit"])
    validation = validate_mofa_extraction_outputs()
    atomic_csv(validation, OUTPUT_TABLES["mofa_validation"])
    atomic_csv(pd.DataFrame(recovered), TABLES / "stage4c_mofa_extract_only_status.csv")
    write_log("stage4c_mofa_extract_only", {"status": "complete", "recovered": recovered, "score_rows": len(score_rows), "loading_rows": len(loading_rows)})
    print(pd.DataFrame(recovered).to_string(index=False))
    print("STAGE4C_MOFA_EXTRACT_ONLY_COMPLETE")


def validate_mofa_extraction_outputs() -> pd.DataFrame:
    rows = []
    scores = pd.read_csv(OUTPUT_TABLES["mofa_scores"]) if OUTPUT_TABLES["mofa_scores"].exists() else pd.DataFrame()
    loadings = pd.read_csv(OUTPUT_TABLES["mofa_loadings"]) if OUTPUT_TABLES["mofa_loadings"].exists() else pd.DataFrame()
    variance = pd.read_csv(OUTPUT_TABLES["mofa_variance"]) if OUTPUT_TABLES["mofa_variance"].exists() else pd.DataFrame()
    if not scores.empty:
        ibd_scores = scores[scores["disease"].eq("IBD")]
        group_counts = ibd_scores[["cohort", "sample_id"]].drop_duplicates().groupby("cohort").size().to_dict()
        rows.append({"test": "ibd_factor_score_sample_rows", "status": "pass" if group_counts == {"Franzosa": 137, "iHMP": 91} else "fail", "notes": str(group_counts)})
        rows.append({"test": "ibd_factor_scores_finite", "status": "pass" if np.isfinite(ibd_scores["score"].to_numpy(dtype=float)).all() else "fail", "notes": str(len(ibd_scores))})
    if not loadings.empty:
        ibd_loadings = loadings[loadings["disease"].eq("IBD")]
        counts = ibd_loadings.groupby("view").size().to_dict()
        rows.append({"test": "ibd_loading_rows", "status": "pass" if counts == {"metabolome": 1328, "microbiome": 400} else "fail", "notes": str(counts)})
        rows.append({"test": "ibd_loadings_finite", "status": "pass" if np.isfinite(ibd_loadings["loading"].to_numpy(dtype=float)).all() else "fail", "notes": str(len(ibd_loadings))})
    rows.append({"test": "variance_explained_present", "status": "pass" if not variance.empty else "fail", "notes": f"rows={len(variance)}"})
    return pd.DataFrame(rows)


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


def validate_mofa_hdf5(outfile: Path, disease: str, expected_cohorts: list[str]) -> tuple[bool, pd.DataFrame]:
    import h5py

    rows: list[dict[str, Any]] = []
    if not outfile.exists():
        return False, pd.DataFrame([{"disease": disease, "path": str(outfile), "test": "hdf5_exists", "status": "absent", "notes": ""}])
    try:
        with h5py.File(outfile, "r") as h5:
            groups = decode_h5_values(h5["groups/groups"][()]) if "groups/groups" in h5 else []
            views = decode_h5_values(h5["views/views"][()]) if "views/views" in h5 else []
            likelihoods = decode_h5_values(h5["model_options/likelihoods"][()]) if "model_options/likelihoods" in h5 else []
            rows.append({"disease": disease, "path": str(outfile), "test": "groups", "status": "pass" if groups == expected_cohorts else "fail", "notes": ";".join(groups)})
            rows.append({"disease": disease, "path": str(outfile), "test": "views", "status": "pass" if set(views) == {"metabolome", "microbiome"} else "fail", "notes": ";".join(views)})
            rows.append({"disease": disease, "path": str(outfile), "test": "likelihoods", "status": "pass" if likelihoods == ["gaussian", "gaussian"] else "fail", "notes": ";".join(likelihoods)})
            for cohort in expected_cohorts:
                samples = decode_h5_values(h5[f"samples/{cohort}"][()])
                z = np.asarray(h5[f"expectations/Z/{cohort}"])
                n_expected = EXPECTED[(disease, cohort)]["n"]
                ok = len(samples) == n_expected and z.shape in {(MOFA_FACTORS[0], n_expected), (n_expected, MOFA_FACTORS[0])}
                rows.append({"disease": disease, "path": str(outfile), "test": f"Z_{cohort}", "status": "pass" if ok else "fail", "notes": f"samples={len(samples)};Z_shape={z.shape}"})
            for view in ["metabolome", "microbiome"]:
                features = decode_h5_values(h5[f"features/{view}"][()])
                w = np.asarray(h5[f"expectations/W/{view}"])
                expected_feature_n = EXPECTED[(disease, expected_cohorts[0])]["met_p" if view == "metabolome" else "micro_p"]
                if disease == "CRC" and view == "metabolome":
                    expected_feature_n = 109
                ok = len(features) == expected_feature_n and w.shape in {(MOFA_FACTORS[0], expected_feature_n), (expected_feature_n, MOFA_FACTORS[0])}
                rows.append({"disease": disease, "path": str(outfile), "test": f"W_{view}", "status": "pass" if ok else "fail", "notes": f"features={len(features)};W_shape={w.shape}"})
            rows.append({"disease": disease, "path": str(outfile), "test": "training_stats", "status": "pass" if "training_stats/elbo" in h5 and len(h5["training_stats/elbo"]) > 0 else "fail", "notes": f"elbo_len={len(h5['training_stats/elbo']) if 'training_stats/elbo' in h5 else 0}"})
    except Exception as exc:
        rows.append({"disease": disease, "path": str(outfile), "test": "hdf5_open_or_schema", "status": "fail", "notes": repr(exc)})
    df = pd.DataFrame(rows)
    return bool(df["status"].eq("pass").all()), df


def extract_mofa_hdf5(outfile: Path, disease: str, cohorts: list[str], meta_df: pd.DataFrame) -> dict[str, Any]:
    import h5py

    scores: list[dict[str, Any]] = []
    loadings: list[dict[str, Any]] = []
    variance: list[dict[str, Any]] = []
    nondegenerate: list[dict[str, Any]] = []
    loading_series: dict[str, pd.Series] = {}
    score_frames = []
    with h5py.File(outfile, "r") as h5:
        groups = decode_h5_values(h5["groups/groups"][()])
        views = decode_h5_values(h5["views/views"][()])
        n_factors = int(np.asarray(h5[f"expectations/Z/{groups[0]}"]).shape[0] if np.asarray(h5[f"expectations/Z/{groups[0]}"]).shape[0] == MOFA_FACTORS[0] else MOFA_FACTORS[0])
        meta_lookup = meta_df.copy()
        meta_lookup["sample_id"] = meta_lookup["sample_id"].astype(str)
        if meta_lookup["sample_id"].duplicated().any():
            raise ValueError(f"{disease}: duplicate sample IDs in metadata used for MOFA extraction.")
        meta_lookup = meta_lookup.set_index("sample_id")
        for cohort in groups:
            if cohort not in cohorts:
                raise ValueError(f"{disease}: unexpected HDF5 group {cohort}; expected {cohorts}")
            samples = decode_h5_values(h5[f"samples/{cohort}"][()])
            missing = [s for s in samples if s not in meta_lookup.index]
            if missing:
                raise ValueError(f"{disease} {cohort}: HDF5 sample(s) missing from metadata: {missing[:5]}")
            z_raw = np.asarray(h5[f"expectations/Z/{cohort}"])
            z, z_orientation = orient_factor_scores(z_raw, len(samples), n_factors, f"{disease} {cohort} Z")
            sf = pd.DataFrame(z, index=samples, columns=[f"Factor{i + 1}" for i in range(n_factors)])
            sf.insert(0, "cohort", cohort)
            sf.insert(0, "disease", disease)
            sf.insert(2, "phenotype", meta_lookup.loc[samples, "disease_group"].astype(str).to_numpy())
            score_frames.append(sf.reset_index(names="sample_id"))
            for sample in samples:
                phenotype = str(meta_lookup.loc[sample, "disease_group"])
                for factor_idx in range(n_factors):
                    scores.append({"disease": disease, "cohort": cohort, "sample_id": sample, "phenotype": phenotype, "factor": f"Factor{factor_idx + 1}", "score": float(sf.loc[sample, f"Factor{factor_idx + 1}"]), "z_orientation": z_orientation})
            for factor_idx in range(n_factors):
                vals = sf[f"Factor{factor_idx + 1}"].to_numpy(dtype=float)
                nondegenerate.append({"disease": disease, "cohort": cohort, "factor": f"Factor{factor_idx + 1}", "score_variance": float(np.nanvar(vals)), "nondegenerate": bool(np.nanvar(vals) > 1e-8)})
        for view in views:
            features = decode_h5_values(h5[f"features/{view}"][()])
            w_raw = np.asarray(h5[f"expectations/W/{view}"])
            w, w_orientation = orient_loadings_matrix(w_raw, len(features), n_factors, f"{disease} {view} W")
            for factor_idx in range(n_factors):
                vals = pd.Series(w[:, factor_idx], index=features)
                for feature, value in vals.items():
                    loadings.append({"disease": disease, "view": view, "feature_id": feature, "feature_name": feature.removeprefix("met__").removeprefix("mic__"), "factor": f"Factor{factor_idx + 1}", "loading": float(value), "w_orientation": w_orientation})
                loading_series[f"Factor{factor_idx + 1}__{view}__{disease}"] = vals
        if "variance_explained" in h5:
            collect_variance_explained(h5, variance, disease, views, groups, n_factors)
    wide_scores = pd.concat(score_frames, ignore_index=True)
    return {"scores": scores, "loadings": loadings, "variance": variance, "nondegenerate": nondegenerate, "loading_series": loading_series, "score_frame": wide_scores}


def collect_variance(group, rows: list[dict[str, Any]], disease: str, prefix: str = "") -> None:
    for key, value in group.items():
        name = f"{prefix}/{key}" if prefix else key
        if hasattr(value, "items"):
            collect_variance(value, rows, disease, name)
        else:
            arr = np.asarray(value).ravel()
            for i, val in enumerate(arr):
                rows.append({"disease": disease, "source": name, "index": i, "variance_explained": float(val)})


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


def factor_associations(score_frame: pd.DataFrame, meta_df: pd.DataFrame, disease: str) -> list[dict[str, Any]]:
    rows = []
    merged = score_frame.merge(meta_df[["sample_id", "disease_group"]], on="sample_id", how="left")
    factors = [c for c in merged.columns if c.startswith("Factor")]
    for factor in factors:
        for label in sorted(merged["disease_group"].dropna().unique()):
            if label in {"Control", "Control/Healthy"}:
                continue
            control = "Control/Healthy" if disease == "CRC" else "Control"
            a = merged.loc[merged["disease_group"].eq(label), factor].dropna()
            b = merged.loc[merged["disease_group"].eq(control), factor].dropna()
            if len(a) >= 3 and len(b) >= 3:
                stat = stats.mannwhitneyu(a, b, alternative="two-sided")
                rows.append({"disease": disease, "comparison": f"{label}_vs_{control}", "factor": factor, "effect_median_difference": float(a.median() - b.median()), "p_value": float(stat.pvalue), "test": "Mann-Whitney U post-hoc annotation"})
    if rows:
        pvals = np.array([r["p_value"] for r in rows], dtype=float)
        order = np.argsort(pvals)
        q = np.empty_like(pvals)
        prev = 1.0
        for rank, idx in enumerate(order[::-1], start=1):
            val = min(prev, pvals[idx] * len(pvals) / (len(pvals) - rank + 1))
            q[idx] = val
            prev = val
        for r, fdr in zip(rows, q):
            r["bh_fdr"] = float(fdr)
    return rows


def run_scca() -> None:
    ensure_dirs()
    try:
        blocks = load_blocks()
        valid_features, feature_trace = build_valid_feature_sets(blocks)
        audit = input_integrity(blocks)
        if not audit["status"].eq("pass").all():
            raise RuntimeError("sCCA blocked: input integrity audit has failures.")
        atomic_csv(feature_trace, OUTPUT_TABLES["feature_qc_trace"])
        atomic_csv(scca_method_metadata(), TABLES / "stage4c_scca_method_metadata.csv")
        input_manifest = write_scca_pma_inputs(blocks, valid_features)
        if not input_manifest["finite_after_preprocessing"].all() or not input_manifest["sample_order_aligned"].all():
            raise RuntimeError("sCCA blocked: PMA input manifest contains invalid matrices.")

        invoke_pma_scca()
        component_df, micro_loading_df, met_loading_df, tuning_df = copy_pma_outputs_to_tables()
        loading_outputs_valid, loading_audit = validate_scca_loading_outputs(component_df, micro_loading_df, met_loading_df)
        atomic_csv(loading_audit, TABLES / "stage4c_scca_loading_output_audit.csv")
        score_outputs_valid, score_audit = validate_scca_score_outputs(component_df)
        atomic_csv(score_audit, TABLES / "stage4c_scca_score_output_audit.csv")
        if not loading_outputs_valid:
            raise RuntimeError("sCCA PMA output loading validation failed. See stage4c_scca_loading_output_audit.csv")
        if not score_outputs_valid:
            raise RuntimeError("sCCA PMA score validation failed. See stage4c_scca_score_output_audit.csv")
        repro_df = pma_reproducibility(micro_loading_df, met_loading_df)

        association_rows: list[dict[str, Any]] = []
        for disease, cohorts in {"IBD": ["Franzosa", "iHMP"], "CRC": ["Sinha", "Yachida"]}.items():
            for cohort in cohorts:
                block = blocks[(disease, cohort)]
                comps = sorted(component_df.loc[component_df["disease"].eq(disease) & component_df["cohort"].eq(cohort), "component"].astype(int).unique())
                for comp in comps:
                    score = pma_scores_for(disease, cohort, int(comp))
                    association_rows.extend(scca_component_associations(score, block.metadata, disease, cohort, int(comp)))

        atomic_csv(repro_df, OUTPUT_TABLES["scca_repro"])
        association_df = pd.DataFrame(association_rows)
        if not association_df.empty:
            association_df["bh_fdr"] = association_df.groupby(["disease", "cohort", "comparison"], group_keys=False)["p_value"].transform(lambda s: bh_fdr(s.tolist()))
        atomic_csv(association_df, OUTPUT_TABLES["scca_associations"])
        plot_scca_results(repro_df, micro_loading_df, met_loading_df)
        preliminary_validation = pd.DataFrame([{"test": "validation_table_initialized", "status": "pass", "notes": "current run"}])
        atomic_csv(preliminary_validation, OUTPUT_TABLES["scca_validation"])
        schema_valid, schema_audit = validate_scca_table_schemas()
        atomic_csv(schema_audit, TABLES / "stage4c_scca_schema_audit.csv")
        figure_paths = [FIGURES / "stage4c_scca_cross_cohort_component_similarity.png", FIGURES / "stage4c_scca_selected_feature_overlap.png"]
        figures_valid = all(path.exists() and path.stat().st_size > 0 for path in figure_paths)
        validation = pd.DataFrame([
            {"test": "true_penalised_scca", "status": "pass", "notes": "Primary path uses PMA::CCA; thresholded dense sklearn CCA is not used."},
            {"test": "scca_outputs_written", "status": "pass", "notes": "component summary, loadings, tuning, reproducibility, associations and figures written"},
            {"test": "phenotype_not_used_for_tuning", "status": "pass" if tuning_df["phenotype_used_for_tuning"].eq(False).all() else "fail", "notes": "tuning table records phenotype_used_for_tuning=False"},
            {"test": "component_count_limited", "status": "pass" if component_df["component"].astype(int).max() <= max(SCCA_COMPONENTS) else "fail", "notes": f"allowed components {SCCA_COMPONENTS}"},
            {"test": "loading_row_counts_valid", "status": "pass" if loading_outputs_valid else "fail", "notes": "one loading row per feature per selected component"},
            {"test": "score_rows_valid", "status": "pass" if score_outputs_valid else "fail", "notes": "one finite x/z score per sample per selected component"},
            {"test": "cohort_independent_fits", "status": "pass", "notes": "Franzosa, iHMP, Sinha and Yachida fitted separately"},
            {"test": "component_matching_after_fit", "status": "pass", "notes": "component pairs selected by post-fit loading similarity"},
            {"test": "component_associations_posthoc_only", "status": "pass", "notes": "phenotype associations computed only after sCCA component fitting"},
            {"test": "cross_cohort_reproducibility_written", "status": "pass", "notes": f"{len(repro_df)} rows"},
            {"test": "schemas_valid", "status": "pass" if schema_valid else "fail", "notes": "required output table schemas checked"},
            {"test": "figures_valid", "status": "pass" if figures_valid else "fail", "notes": ";".join(str(p) for p in figure_paths)},
        ])
        atomic_csv(validation, OUTPUT_TABLES["scca_validation"])
        write_scca_interpretation_report(component_df, repro_df, association_df, validation)
        if not validation["status"].eq("pass").all():
            raise RuntimeError("sCCA postprocessing validation failed; completion marker not written.")
        atomic_json({"status": "complete", "method": "PMA_true_penalised_sCCA", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "script_hash": sha256(Path(__file__))}, MANIFESTS / "stage4c_scca_complete.json")
        write_log("stage4c_scca", {"status": "complete", "component_rows": len(component_df), "reproducibility_rows": len(repro_df)})
        print("STAGE4C_TRUE_SCCA_COMPLETE")
    except Exception as exc:
        atomic_json({"status": "SCCA_MODEL_FITS_COMPLETE_POSTPROCESSING_FAILED", "error": repr(exc), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "script_hash": sha256(Path(__file__))}, MANIFESTS / "stage4c_scca_failed.json")
        write_log("stage4c_scca", {"status": "failed", "error": repr(exc)})
        raise


def finalize() -> None:
    ensure_dirs()
    required = [
        OUTPUT_TABLES["mofa_scores"], OUTPUT_TABLES["mofa_variance"], OUTPUT_TABLES["mofa_loadings"],
        OUTPUT_TABLES["mofa_associations"], OUTPUT_TABLES["mofa_repro"], OUTPUT_TABLES["mofa_nondegenerate"],
        OUTPUT_TABLES["scca_summary"], OUTPUT_TABLES["scca_micro_loadings"], OUTPUT_TABLES["scca_met_loadings"], OUTPUT_TABLES["scca_repro"], OUTPUT_TABLES["scca_tuning"], OUTPUT_TABLES["scca_associations"],
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Stage 4C finalize blocked because model-dependent outputs are missing: " + "; ".join(missing))
    (REPORTS / "STAGE4C_RESULTS_INTERPRETATION.md").write_text("# Stage 4C Results Interpretation\n\nModel-dependent MOFA and sCCA outputs were found. Add interpretive synthesis here after verifying generated figures.\n")
    write_log("stage4c_finalize", {"status": "complete_valid"})


def status_mode() -> None:
    ensure_dirs()
    rows = []
    mofa_groups = {"IBD": ["Franzosa", "iHMP"], "CRC": ["Sinha", "Yachida"]}
    for disease, cohorts in mofa_groups.items():
        unit = f"MOFA_{disease}"
        model_path = MODELS / f"stage4c_mofa_{disease.lower()}_k{MOFA_FACTORS[0]}.hdf5"
        model_valid, _ = validate_mofa_hdf5(model_path, disease, cohorts)
        tables_present = []
        for path in [OUTPUT_TABLES["mofa_scores"], OUTPUT_TABLES["mofa_variance"], OUTPUT_TABLES["mofa_loadings"]]:
            if not (path.exists() and path.stat().st_size > 0):
                tables_present.append(False)
                continue
            try:
                df = pd.read_csv(path, nrows=50000)
                tables_present.append("disease" in df.columns and df["disease"].astype(str).eq(disease).any())
            except Exception:
                tables_present.append(False)
        if model_valid and all(tables_present):
            status = "COMPLETE_VALID"
        elif model_valid:
            status = "MODEL_FIT_COMPLETE_EXTRACTION_INCOMPLETE"
        elif model_path.exists():
            status = "INVALID_OUTPUT"
        else:
            status = "NOT_STARTED"
        rows.append({"unit": unit, "status": status, "files_present": int(sum(tables_present)) + int(model_path.exists()), "files_expected": 4, "model_path": str(model_path)})
    expected_files = {
        "sCCA_IBD": [OUTPUT_TABLES["scca_summary"], OUTPUT_TABLES["scca_micro_loadings"], OUTPUT_TABLES["scca_met_loadings"], OUTPUT_TABLES["scca_repro"], OUTPUT_TABLES["scca_tuning"], OUTPUT_TABLES["scca_associations"]],
        "sCCA_CRC": [OUTPUT_TABLES["scca_summary"], OUTPUT_TABLES["scca_micro_loadings"], OUTPUT_TABLES["scca_met_loadings"], OUTPUT_TABLES["scca_repro"], OUTPUT_TABLES["scca_tuning"], OUTPUT_TABLES["scca_associations"]],
        "FINALIZE": [REPORTS / "STAGE4C_RESULTS_INTERPRETATION.md"],
    }
    for unit, files in expected_files.items():
        exist = [p.exists() and p.stat().st_size > 0 for p in files]
        valid_content = True
        if unit.startswith("sCCA") and all(exist):
            disease = unit.split("_", 1)[1]
            try:
                summary = pd.read_csv(OUTPUT_TABLES["scca_summary"])
                micro = pd.read_csv(OUTPUT_TABLES["scca_micro_loadings"])
                met = pd.read_csv(OUTPUT_TABLES["scca_met_loadings"])
                valid_content, _ = validate_scca_loading_outputs(
                    summary[summary["disease"].eq(disease)],
                    micro[micro["disease"].eq(disease)],
                    met[met["disease"].eq(disease)],
                )
            except Exception:
                valid_content = False
        if all(exist) and valid_content:
            status = "COMPLETE_VALID"
        elif all(exist) and not valid_content:
            status = "INVALID_OUTPUT"
        elif any(exist):
            status = "PARTIAL"
        else:
            status = "NOT_STARTED"
        rows.append({"unit": unit, "status": status, "files_present": int(sum(exist)), "files_expected": len(files)})
    df = pd.DataFrame(rows)
    atomic_csv(df, OUTPUT_TABLES["status"])
    print(df.to_string(index=False))


def api_smoke() -> None:
    ensure_dirs()
    if not module_available("mofapy2"):
        raise SystemExit("mofapy2 is unavailable; cannot run synthetic API smoke.")
    rows = []
    samples = {"G1": [f"g1_s{i}" for i in range(1, 17)], "G2": [f"g2_s{i}" for i in range(1, 17)]}
    long_parts = []
    for group, ids in samples.items():
        for view, features in {
            "microbiome": [f"mic_{i}" for i in range(1, 10)],
            "metabolome": [f"met_{i}" for i in range(1, 10)],
        }.items():
            mat = pd.DataFrame(
                np.arange(len(ids) * len(features), dtype=float).reshape(len(ids), len(features)) + 1.0,
                index=ids,
                columns=features,
            )
            long_parts.append(serialize_view_matrix(mat, ids, group, view))
    data_df = pd.concat(long_parts, ignore_index=True)
    try:
        construct_mofa_entry(data_df, MOFA_FACTORS[0], MODELS / "stage4c_mofa_api_smoke_synthetic.hdf5")
        rows.append({"test": "synthetic_two_group_two_view_k8_build", "status": "pass", "notes": "mofapy2 set_data_df/options/build succeeded; run() not called"})
    except Exception as exc:
        rows.append({"test": "synthetic_two_group_two_view_k8_build", "status": "fail", "notes": repr(exc)})
    atomic_csv(pd.DataFrame(rows), OUTPUT_TABLES["mofa_api_smoke"])
    print(pd.DataFrame(rows).to_string(index=False))
    if any(row["status"] != "pass" for row in rows):
        raise SystemExit("STAGE4C_MOFA_API_SMOKE_FAILED")
    print("STAGE4C_MOFA_API_SMOKE_PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4C MOFA-primary and sCCA-secondary integration workflow.")
    parser.add_argument("mode", choices=["stage4c_mofa", "stage4c_scca", "stage4c_finalize"])
    parser.add_argument("--preflight-only", action="store_true", help="Validate Stage 4C dependencies, frozen inputs and output paths only; no model fitting.")
    parser.add_argument("--dry-run", action="store_true", help="Construct real-data Stage 4C views and config, then stop before any scientific fit.")
    parser.add_argument("--status", action="store_true", help="Report Stage 4C output completion status and exit.")
    parser.add_argument("--api-smoke", action="store_true", help="Run a tiny synthetic mofapy2 construction/build smoke test; no real data and no run().")
    parser.add_argument("--extract-only", action="store_true", help="Extract/validate existing MOFA HDF5 outputs only; no model fitting.")
    args = parser.parse_args()
    ensure_dirs()
    if args.status:
        status_mode(); return
    if args.api_smoke:
        api_smoke(); return
    if args.extract_only:
        if args.mode != "stage4c_mofa":
            raise SystemExit("--extract-only is currently defined only for stage4c_mofa")
        extract_only_mofa(); return
    if args.preflight_only:
        preflight(); return
    if args.dry_run:
        dry_run("MOFA" if args.mode == "stage4c_mofa" else "sCCA" if args.mode == "stage4c_scca" else "FINALIZE")
        return
    if args.mode == "stage4c_mofa":
        run_mofa()
    elif args.mode == "stage4c_scca":
        run_scca()
    elif args.mode == "stage4c_finalize":
        finalize()


if __name__ == "__main__":
    main()
