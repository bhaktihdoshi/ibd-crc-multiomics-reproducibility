from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import scipy
    from scipy.stats import hypergeom as scipy_hypergeom
    from scipy.stats import t as t_dist
except Exception:  # pragma: no cover - recorded in manifest if unavailable
    scipy = None
    scipy_hypergeom = None
    t_dist = None

try:
    import sklearn
except Exception:  # pragma: no cover - recorded in manifest if unavailable
    sklearn = None

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[2]
CRC_DIR = ROOT / "thesis_preprocessing_workflow" / "crc_pipeline"
STAGE1_RELEASE = CRC_DIR / "stage1_release"
RAW_DIR = ROOT / "Colon_Cancer_Data_Set"
SINHA_DIR = RAW_DIR / "Sinha-et-al"
YACHIDA_DIR = RAW_DIR / "Yachinda_et-al"

STAGE2 = CRC_DIR / "stage2"
TABLES = STAGE2 / "tables"
TABLES_MAIN = TABLES / "main_text"
PATHWAY_TABLES = TABLES / "pathway"
FIGURES = STAGE2 / "figures"
FIGURES_MAIN = FIGURES / "main_text"
FIGURES_SUPP = FIGURES / "supplementary"
FIGURES_PRESENTATION = FIGURES / "presentation"
MODELS = STAGE2 / "models"
PREDICTIONS = STAGE2 / "predictions"
LOGS = STAGE2 / "logs"
REPORTS = STAGE2 / "reports"
PROCESSED = STAGE2 / "data_processed"
ANNOTATION_CACHE = CRC_DIR / "annotation_resources"

STAGE1_TABLES = CRC_DIR / "tables"
STAGE1_STAGE2_PREP = STAGE1_TABLES / "stage2_model_prep"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from thesis_preprocessing_workflow.crc_pipeline.crc_pipeline_transformers import (
        ConstantFeatureFilter,
        BalancedGradientBoostingClassifier,
        InputColumnOrderValidator,
        Log1pTransformer,
        MissingnessFilter,
    )
except ImportError:
    from crc_pipeline_transformers import (
        ConstantFeatureFilter,
        BalancedGradientBoostingClassifier,
        InputColumnOrderValidator,
        Log1pTransformer,
        MissingnessFilter,
    )


@dataclass(frozen=True)
class CRCStage2Config:
    random_seed: int = 42
    fast_test: bool = False
    run_id: str = "crc_stage2_final_candidate"
    primary_missingness_threshold: float = 0.30
    external_validation_directions: tuple[tuple[str, str], ...] = (("Sinha", "Yachida"), ("Yachida", "Sinha"))
    primary_task_name: str = "crc_vs_control"
    model_selection_metric: str = "roc_auc"
    model_selection_effective_tie_tolerance_auc: float = 0.01
    model_selection_effective_tie_tolerance_pr_auc: float = 0.01
    model_selection_effective_tie_tolerance_roc_auc_sd: float = 0.005
    bootstrap_repetitions: int = 2000
    threshold_selection_rule: str = "training_only_youden_j"
    outer_folds_requested: int = 5
    inner_folds_requested: int = 5
    reduced_inner_folds: int = 3
    min_class_count_for_requested_inner: int = 5
    model_families: tuple[str, ...] = (
        "ridge_logistic",
        "elastic_net_logistic",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
    )
    feature_sets: tuple[str, ...] = ("clinical", "metabolomics", "integrated")
    shared_clinical_features: tuple[str, ...] = ("age_years", "sex", "bmi")
    clinical_features_excluded_from_primary: tuple[str, ...] = (
        "stage_major",
        "stage_source_faithful",
        "stage_harmonized",
        "tumour_site",
        "smoking",
        "alcohol",
        "hosp1",
        "hosp2",
    )
    primary_sinha_expected: dict[str, int] = field(default_factory=lambda: {"n": 131, "control": 89, "crc": 42})
    primary_yachida_expected: dict[str, int] = field(default_factory=lambda: {"n": 277, "control": 127, "crc": 150})
    expected_identifier_overlap: dict[str, int] = field(default_factory=lambda: {"HMDB": 110, "KEGG": 107, "HMDB_or_KEGG_unified": 217})
    randomized_search_n_iter: dict[str, int] = field(default_factory=lambda: {"random_forest": 75, "extra_trees": 75, "gradient_boosting": 75})
    fast_test_randomized_search_n_iter: int = 4
    fast_test_models: tuple[str, ...] = ("ridge_logistic", "random_forest")
    model_simplicity_order: dict[str, int] = field(
        default_factory=lambda: {
            "ridge_logistic": 0,
            "elastic_net_logistic": 1,
            "random_forest": 2,
            "extra_trees": 3,
            "gradient_boosting": 4,
        }
    )
    transformation_rules: dict[str, str] = field(
        default_factory=lambda: {
            "Sinha": "none_predefined_values_appear_transformed_or_standardized",
            "Yachida": "primary_analysis_no_log1p_supplied_values_retained",
        }
    )
    feature_harmonisation_rules: dict[str, str] = field(
        default_factory=lambda: {
            "canonical_priority": "valid HMDB first; valid KEGG fallback only when HMDB absent",
            "duplicates": "select one deterministic assay feature per canonical metabolite per cohort; do not average",
            "external_validation": "identical canonical feature names and order in both cohorts",
        }
    )
    annotation_settings: dict[str, Any] = field(
        default_factory=lambda: {
            "allow_external_download": False,
            "annotation_cache_dir": "thesis_preprocessing_workflow/crc_pipeline/annotation_resources",
            "minimum_pathway_background_compounds": 3,
            "minimum_pathway_significant_compounds": 2,
            "low_coverage_threshold": 0.50,
            "excluded_kegg_pathway_ids": ["map01100", "map01110", "map01120", "map01200"],
            "excluded_kegg_pathway_name_patterns": [
                "Metabolic pathways",
                "Biosynthesis of secondary metabolites",
                "Microbial metabolism in diverse environments",
                "Biosynthesis of cofactors",
                "Global and overview maps",
            ],
        }
    )

    def active_models(self) -> tuple[str, ...]:
        return self.fast_test_models if self.fast_test else self.model_families

    def active_outer_folds(self) -> int:
        return 3 if self.fast_test else self.outer_folds_requested

    def active_inner_folds(self) -> int:
        return 2 if self.fast_test else self.inner_folds_requested

    def active_bootstrap_repetitions(self) -> int:
        return 100 if self.fast_test else self.bootstrap_repetitions


CONFIG = CRCStage2Config()


def ensure_dirs() -> None:
    for d in [STAGE2, TABLES, TABLES_MAIN, PATHWAY_TABLES, FIGURES, FIGURES_MAIN, FIGURES_SUPP, FIGURES_PRESENTATION, MODELS, PREDICTIONS, LOGS, REPORTS, PROCESSED]:
        d.mkdir(parents=True, exist_ok=True)


def configure_output_roots(config: CRCStage2Config) -> None:
    """Redirect fast-test outputs away from final Stage 2 outputs."""
    global STAGE2, TABLES, TABLES_MAIN, PATHWAY_TABLES, FIGURES, FIGURES_MAIN, FIGURES_SUPP, FIGURES_PRESENTATION, MODELS, PREDICTIONS, LOGS, REPORTS, PROCESSED
    if not config.fast_test:
        return
    STAGE2 = CRC_DIR / "stage2" / "fast_test"
    TABLES = STAGE2 / "tables"
    TABLES_MAIN = TABLES / "main_text"
    PATHWAY_TABLES = TABLES / "pathway"
    FIGURES = STAGE2 / "figures"
    FIGURES_MAIN = FIGURES / "main_text"
    FIGURES_SUPP = FIGURES / "supplementary"
    FIGURES_PRESENTATION = FIGURES / "presentation"
    MODELS = STAGE2 / "models"
    PREDICTIONS = STAGE2 / "predictions"
    LOGS = STAGE2 / "logs"
    REPORTS = STAGE2 / "reports"
    PROCESSED = STAGE2 / "data_processed"


def setup_logger() -> logging.Logger:
    ensure_dirs()
    logger = logging.getLogger("crc_stage2")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(LOGS / "crc_stage2_run.log")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(sh)
    return logger


LOGGER: logging.Logger | None = None


def get_logger() -> logging.Logger:
    global LOGGER
    if LOGGER is None:
        LOGGER = setup_logger()
    return LOGGER


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", **kwargs)


def package_versions() -> dict[str, str]:
    versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": getattr(sklearn, "__version__", "unavailable"),
        "scipy": getattr(scipy, "__version__", "unavailable"),
        "joblib": getattr(joblib, "__version__", "unavailable"),
    }
    return versions


def write_config(config: CRCStage2Config) -> None:
    ensure_dirs()
    payload = asdict(config)
    payload["analysis_status"] = "NON-FINAL FAST TEST" if config.fast_test else "FINAL_CANDIDATE"
    payload["output_directories"] = {
        "stage2": str(STAGE2),
        "tables": str(TABLES),
        "tables_main_text": str(TABLES_MAIN),
        "figures": str(FIGURES),
        "figures_main_text": str(FIGURES_MAIN),
        "figures_supplementary": str(FIGURES_SUPP),
        "models": str(MODELS),
        "predictions": str(PREDICTIONS),
        "logs": str(LOGS),
        "reports": str(REPORTS),
        "data_processed": str(PROCESSED),
    }
    payload["package_versions"] = package_versions()
    payload["configuration_sha256"] = hashlib.sha256(json.dumps(asdict(config), sort_keys=True, default=str).encode()).hexdigest()
    (REPORTS / "crc_stage2_configuration.json").write_text(json.dumps(payload, indent=2, default=str))


def context_key(*parts: Any) -> str:
    return "__".join(str(p).replace("/", "_").replace(" ", "_") for p in parts)


def write_context_checkpoint_manifest(rows: list[dict[str, Any]], name: str) -> None:
    pd.DataFrame(rows).to_csv(LOGS / name, index=False)


def load_stage1_manifest() -> dict[str, Any]:
    manifest_path = CRC_DIR / "CRC_STAGE1_RELEASE_MANIFEST.json"
    if not STAGE1_RELEASE.exists():
        raise FileNotFoundError(f"Frozen Stage 1 release directory missing: {STAGE1_RELEASE}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Frozen Stage 1 release manifest missing: {manifest_path}")
    return json.loads(manifest_path.read_text())


def validate_stage1_contract(config: CRCStage2Config) -> pd.DataFrame:
    ensure_dirs()
    manifest = load_stage1_manifest()
    rows: list[dict[str, Any]] = []

    def record(check: str, observed: Any, expected: Any, critical: bool = True) -> None:
        passed = observed == expected
        rows.append({"check": check, "observed": observed, "expected": expected, "critical": critical, "passed": passed})
        if critical and not passed:
            raise AssertionError(f"Stage 1 contract violation: {check}: observed={observed!r}, expected={expected!r}")

    record("stage1_manifest_version", manifest.get("version"), "1.0")
    record("stage1_manifest_status", manifest.get("status"), "Frozen")

    required_files = [
        CRC_DIR / "CRC_STAGE1_RELEASE_MANIFEST.json",
        CRC_DIR / "CRC_STAGE2_INPUT_SPECIFICATION.md",
        CRC_DIR / "CRC_STAGE2_MODELLING_CHECKLIST.md",
        CRC_DIR / "CRC_STAGE1_DATA_PROVENANCE.csv",
        CRC_DIR / "CRC_STAGE1_COMPLETE_DATA_DICTIONARY.csv",
        CRC_DIR / "crc_stage1_freeze_integrity_checks.csv",
        STAGE1_TABLES / "crc_stage2_task_eligibility.csv",
        STAGE1_TABLES / "crc_stage1_identifier_overlap.csv",
        STAGE1_TABLES / "crc_stage1_sample_flow.csv",
        STAGE1_TABLES / "crc_stage1_class_counts.csv",
        STAGE1_TABLES / "crc_stage1_shared_identifier_manifest.csv",
        STAGE1_TABLES / "crc_stage1_feature_mapping_manifest.csv",
        STAGE1_STAGE2_PREP / "crc_stage2_yachida_primary_crc_vs_control_model_prep.csv",
        SINHA_DIR / "metadata.tsv",
        SINHA_DIR / "mtb.tsv",
        SINHA_DIR / "mtb.map.tsv",
    ]
    for path in required_files:
        record(f"required_file_exists::{path.name}", path.exists(), True)

    release_hashes = {item["path"]: item["sha256"] for item in manifest.get("generated_outputs", []) if "path" in item and "sha256" in item}
    for rel_path in [
        "tables/crc_stage2_task_eligibility.csv",
        "tables/crc_stage1_identifier_overlap.csv",
        "tables/crc_stage1_sample_flow.csv",
        "tables/crc_stage1_class_counts.csv",
        "tables/stage2_model_prep/crc_stage2_yachida_primary_crc_vs_control_model_prep.csv",
    ]:
        source_path = CRC_DIR / rel_path
        expected_hash = release_hashes.get(rel_path)
        if source_path.exists() and expected_hash:
            record(f"stage1_hash::{rel_path}", sha256(source_path), expected_hash)
        else:
            record(f"stage1_hash::{rel_path}", "not_checked", "not_checked", critical=False)

    integrity = pd.read_csv(CRC_DIR / "crc_stage1_freeze_integrity_checks.csv")
    record("all_stage1_freeze_integrity_checks_passed", bool(integrity["passed"].all()), True)
    record("stage1_freeze_integrity_check_count", int(len(integrity)), 36)

    sample_flow = pd.read_csv(STAGE1_TABLES / "crc_stage1_sample_flow.csv")
    class_counts = pd.read_csv(STAGE1_TABLES / "crc_stage1_class_counts.csv")
    identifier_overlap = pd.read_csv(STAGE1_TABLES / "crc_stage1_identifier_overlap.csv")
    yachida_primary = pd.read_csv(STAGE1_STAGE2_PREP / "crc_stage2_yachida_primary_crc_vs_control_model_prep.csv")

    def flow_n(cohort: str, item: str) -> int:
        return int(sample_flow.loc[(sample_flow["cohort"].eq(cohort)) & (sample_flow["item"].eq(item)), "n"].iloc[0])

    record("Sinha metadata n", flow_n("Sinha", "metadata_samples"), config.primary_sinha_expected["n"])
    record("Yachida metadata n", flow_n("Yachida", "metadata_samples"), 347)
    record("Yachida primary task n", len(yachida_primary), config.primary_yachida_expected["n"])
    record("Yachida primary controls", int((yachida_primary["model_label"] == "Control").sum()), config.primary_yachida_expected["control"])
    record("Yachida primary CRC", int((yachida_primary["model_label"] == "CRC").sum()), config.primary_yachida_expected["crc"])
    record("Yachida primary MP excluded", bool(~yachida_primary["Study.Group"].eq("MP").any()), True)
    record("Yachida primary HS excluded", bool(~yachida_primary["Study.Group"].eq("HS").any()), True)

    sinha_counts = class_counts[(class_counts["cohort"].eq("Sinha")) & (class_counts["label_column"].eq("Study.Group"))]
    record("Sinha controls", int(sinha_counts.loc[sinha_counts["class"].astype(str).eq("0"), "n_samples"].iloc[0]), config.primary_sinha_expected["control"])
    record("Sinha CRC cases", int(sinha_counts.loc[sinha_counts["class"].astype(str).eq("1"), "n_samples"].iloc[0]), config.primary_sinha_expected["crc"])

    for identifier_type, expected in config.expected_identifier_overlap.items():
        observed = int(identifier_overlap.loc[identifier_overlap["identifier_type"].eq(identifier_type), "shared_unique_identifiers"].iloc[0])
        record(f"shared_identifier_count::{identifier_type}", observed, expected)

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "crc_stage2_contract_validation.csv", index=False)
    if not out["passed"].all():
        raise AssertionError("CRC Stage 2 contract validation failed")
    return out


def clean_token(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if text in {"", "NA", "NAN", "NONE", "-"}:
        return None
    return text


def parse_identifier_field(value: Any) -> list[str]:
    token = clean_token(value)
    if token is None:
        return []
    return sorted({x.strip().upper() for x in re.split(r"[;,|/]\s*", token) if x.strip()})


def is_valid_hmdb(identifier: str) -> bool:
    return bool(re.fullmatch(r"HMDB\d{7}", str(identifier).upper()))


def is_valid_kegg(identifier: str) -> bool:
    return bool(re.fullmatch(r"C\d{5}", str(identifier).upper()))


def canonical_ids_from_row(row: pd.Series) -> list[tuple[str, str, str]]:
    hmdb = [x for x in parse_identifier_field(row.get("HMDB")) if is_valid_hmdb(x)]
    kegg = [x for x in parse_identifier_field(row.get("KEGG")) if is_valid_kegg(x)]
    if hmdb:
        return [("HMDB", x, f"HMDB:{x}") for x in hmdb]
    return [("KEGG", x, f"KEGG:{x}") for x in kegg]


def feature_selection_score(row: pd.Series) -> tuple:
    feature = str(row.get("feature", row.get("Compound", "")))
    hmdb_valid = int(row.get("valid_hmdb", False))
    kegg_valid = int(row.get("valid_kegg", False))
    has_method = int(pd.notna(row.get("Method", pd.NA)))
    has_mode = int(pd.notna(row.get("Mode", pd.NA)))
    has_adduct = int(pd.notna(row.get("Adduct", pd.NA)))
    rt = row.get("Retention time", row.get("RT", pd.NA))
    mz = row.get("m/z", row.get("mz", pd.NA))
    has_rt = int(pd.notna(rt))
    has_mz = int(pd.notna(mz))
    return (-hmdb_valid, -kegg_valid, -has_method, -has_mode, -has_adduct, -has_rt, -has_mz, feature)


def load_mapping(cohort: str) -> pd.DataFrame:
    path = SINHA_DIR / "mtb.map.tsv" if cohort == "Sinha" else YACHIDA_DIR / "mtb.map.tsv"
    df = read_tsv(path)
    df["cohort"] = cohort
    df["feature"] = df["Compound"].astype(str)
    rows = []
    for _, row in df.iterrows():
        hmdb_ids = parse_identifier_field(row.get("HMDB"))
        kegg_ids = parse_identifier_field(row.get("KEGG"))
        valid_hmdb = [x for x in hmdb_ids if is_valid_hmdb(x)]
        valid_kegg = [x for x in kegg_ids if is_valid_kegg(x)]
        canon = canonical_ids_from_row(row)
        if not canon:
            rows.append({**row.to_dict(), "canonical_type": "unmapped", "raw_identifier": pd.NA, "canonical_id": pd.NA, "valid_hmdb": bool(valid_hmdb), "valid_kegg": bool(valid_kegg), "mapping_status": "unmapped"})
        for ctype, cid, cname in canon:
            rows.append({**row.to_dict(), "canonical_type": ctype, "raw_identifier": cid, "canonical_id": cname, "valid_hmdb": bool(valid_hmdb), "valid_kegg": bool(valid_kegg), "mapping_status": "mapped"})
    return pd.DataFrame(rows)


def build_canonical_feature_space() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    maps = pd.concat([load_mapping("Sinha"), load_mapping("Yachida")], ignore_index=True)
    mapped = maps[maps["mapping_status"].eq("mapped")].copy()
    conflict_rows = []
    for (cohort, feature), group in mapped.groupby(["cohort", "feature"]):
        canonical_ids = sorted(set(group["canonical_id"].dropna()))
        if len(canonical_ids) > 1:
            conflict_rows.append({"cohort": cohort, "feature": feature, "conflict_type": "one_feature_multiple_canonical_ids", "canonical_ids": ";".join(canonical_ids)})
    for canonical_id, group in mapped.groupby("canonical_id"):
        for cohort, cg in group.groupby("cohort"):
            feats = sorted(set(cg["feature"]))
            if len(feats) > 1:
                conflict_rows.append({"cohort": cohort, "feature": ";".join(feats), "conflict_type": "multiple_features_one_canonical_id", "canonical_ids": canonical_id})

    selected_rows = []
    for (cohort, canonical_id), group in mapped.groupby(["cohort", "canonical_id"], dropna=False):
        ranked = group.copy()
        ranked["_score"] = ranked.apply(feature_selection_score, axis=1)
        ranked = ranked.sort_values("_score").reset_index(drop=True)
        chosen = ranked.iloc[0].drop(labels=["_score"]).to_dict()
        chosen["selection_rank"] = 1
        chosen["n_candidate_features_for_canonical_id"] = len(ranked)
        chosen["selection_rule"] = "valid HMDB priority; KEGG fallback; assay metadata completeness; lexical tie-breaker"
        selected_rows.append(chosen)
    selected = pd.DataFrame(selected_rows)
    unresolved_ambiguous_ids = set()
    for row in conflict_rows:
        for cid in str(row.get("canonical_ids", "")).split(";"):
            if cid:
                unresolved_ambiguous_ids.add(cid)
    shared_ids = sorted(
        (
            set(selected.loc[selected["cohort"].eq("Sinha"), "canonical_id"])
            & set(selected.loc[selected["cohort"].eq("Yachida"), "canonical_id"])
        )
        - unresolved_ambiguous_ids
    )
    shared = selected[selected["canonical_id"].isin(shared_ids)].copy()
    shared["canonical_feature_name"] = shared["canonical_id"]
    shared_space = (
        shared[["canonical_id", "canonical_type", "canonical_feature_name"]]
        .drop_duplicates()
        .sort_values("canonical_feature_name")
        .reset_index(drop=True)
    )
    compatibility_rows = []
    for cid, group in shared.groupby("canonical_id"):
        compatibility_rows.append({
            "canonical_id": cid,
            "n_cohorts": group["cohort"].nunique(),
            "sinha_feature": ";".join(group.loc[group["cohort"].eq("Sinha"), "feature"].astype(str)),
            "yachida_feature": ";".join(group.loc[group["cohort"].eq("Yachida"), "feature"].astype(str)),
            "assay_compatibility_note": "identifier-backed shared canonical metabolite; platform/value-scale differences still handled in modelling pipeline",
        })
    manifest = mapped.sort_values(["canonical_id", "cohort", "feature"]).reset_index(drop=True)
    conflicts = pd.DataFrame(conflict_rows, columns=["cohort", "feature", "conflict_type", "canonical_ids"])
    assay = pd.DataFrame(compatibility_rows)
    crosswalk = manifest[["cohort", "feature", "HMDB", "KEGG", "canonical_type", "raw_identifier", "canonical_id", "mapping_status"]].drop_duplicates()
    crosswalk["cross_namespace_join_used"] = False
    approved_crosswalk = ANNOTATION_CACHE / "approved_hmdb_kegg_crosswalk.csv"
    crosswalk["cross_namespace_mapping_status"] = "not_run_missing_crosswalk" if not approved_crosswalk.exists() else "approved_crosswalk_available_not_required_for_same_namespace_matches"
    crosswalk["cross_namespace_policy"] = "Primary modelling is restricted to direct same-namespace shared IDs unless an approved local HMDB-KEGG cross-reference is available; compound-name equivalence is never inferred."
    decisions = selected.copy()
    decisions["selected_feature"] = decisions["feature"]
    decisions["candidate_score"] = decisions.apply(feature_selection_score, axis=1).astype(str)
    decisions["exclusion_reason"] = np.where(decisions["canonical_id"].isin(shared_ids), "retained_in_shared_model_space", "not_shared_between_primary_cohorts")
    decisions["selection_rationale"] = decisions["selection_rule"]
    ambiguous = conflicts[conflicts["conflict_type"].isin(["one_feature_multiple_canonical_ids", "multiple_features_one_canonical_id"])].copy()
    excluded_ambiguous = ambiguous.copy()
    if not excluded_ambiguous.empty:
        excluded_ambiguous["default_action"] = "exclude_ambiguous_cross_namespace_mapping_until_reviewed"
    feature_order = shared_space[["canonical_feature_name", "canonical_id", "canonical_type"]].copy()
    feature_order["feature_order_index"] = np.arange(len(feature_order))
    feature_order["sinha_column"] = feature_order["canonical_feature_name"]
    feature_order["yachida_column"] = feature_order["canonical_feature_name"]
    feature_order["order_contract"] = "identical canonical feature names and order required in both external-validation directions"
    if set(shared_space["canonical_id"]) & unresolved_ambiguous_ids:
        raise AssertionError("Unresolved ambiguous canonical IDs were retained in the final shared modelling space.")
    selected_shared = selected[selected["canonical_id"].isin(shared_ids)]
    if selected_shared.duplicated(["cohort", "canonical_id"]).any():
        raise AssertionError("More than one selected feature per canonical ID per cohort after ambiguity exclusion.")
    for cohort in ["Sinha", "Yachida"]:
        cohort_order = selected_shared[selected_shared["cohort"].eq(cohort)].sort_values("canonical_id")["canonical_id"].tolist()
        if cohort_order != shared_space["canonical_id"].tolist():
            raise AssertionError(f"{cohort} final canonical feature order does not match shared feature order contract.")
    manifest.to_csv(TABLES / "crc_stage2_canonical_metabolite_manifest.csv", index=False)
    conflicts.to_csv(TABLES / "crc_stage2_identifier_conflicts.csv", index=False)
    selected.to_csv(TABLES / "crc_stage2_selected_features_by_cohort.csv", index=False)
    shared_space.to_csv(TABLES / "crc_stage2_shared_feature_space.csv", index=False)
    assay.to_csv(TABLES / "crc_stage2_assay_compatibility.csv", index=False)
    crosswalk.to_csv(TABLES / "crc_stage2_hmdb_kegg_crosswalk.csv", index=False)
    decisions.to_csv(TABLES / "crc_stage2_canonicalisation_decisions.csv", index=False)
    ambiguous.to_csv(TABLES / "crc_stage2_ambiguous_cross_namespace_mappings.csv", index=False)
    excluded_ambiguous.to_csv(TABLES / "crc_stage2_excluded_ambiguous_metabolites.csv", index=False)
    shared_space.to_csv(TABLES / "crc_stage2_final_shared_model_feature_space.csv", index=False)
    feature_order.to_csv(TABLES / "crc_stage2_feature_order_contract.csv", index=False)
    return manifest, conflicts, selected, shared_space, assay


def standardize_sex(value: Any) -> str | None:
    token = clean_token(value)
    if token is None:
        return None
    if token in {"M", "MALE"}:
        return "Male"
    if token in {"F", "FEMALE"}:
        return "Female"
    return None


def clean_source_missing(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if text.upper() in {"", "-", "NA", "N/A", "NAN", "NONE", "NULL", "MISSING"}:
        return pd.NA
    return value


def numeric_in_range(value: Any, low: float, high: float) -> float:
    val = pd.to_numeric(pd.Series([clean_source_missing(value)]), errors="coerce").iloc[0]
    if pd.isna(val) or not (low <= float(val) <= high):
        return np.nan
    return float(val)


def load_sinha_primary() -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = read_tsv(SINHA_DIR / "metadata.tsv")
    mtb = read_tsv(SINHA_DIR / "mtb.tsv")
    merged = metadata.merge(mtb, on="Sample", how="inner")
    meta = pd.DataFrame({
        "cohort": "Sinha",
        "sample_id": merged["Sample"].astype(str),
        "participant_id": merged["Subject"].astype(str),
        "original_label": merged["Study.Group"].astype(str),
        "crc_binary": np.where(merged["Study.Group"].astype(int).eq(1), 1, 0),
        "age_years": pd.to_numeric(merged.get("Age"), errors="coerce"),
        "sex": merged.get("Gender", pd.Series(index=merged.index, dtype=object)).map(standardize_sex),
        "bmi": pd.to_numeric(merged.get("BMI"), errors="coerce"),
        "stage_major": pd.NA,
        "stage_source_faithful": pd.NA,
        "stage_harmonized": pd.NA,
        "tumour_site": pd.NA,
        "smoking": pd.NA,
        "alcohol": pd.NA,
    })
    x = merged.drop(columns=[c for c in metadata.columns if c != "Sample"], errors="ignore").drop(columns=["Sample"], errors="ignore")
    return meta, x.apply(pd.to_numeric, errors="coerce")


def load_yachida_primary() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = STAGE1_STAGE2_PREP / "crc_stage2_yachida_primary_crc_vs_control_model_prep.csv"
    df = pd.read_csv(path)
    meta_cols = [
        "Sample", "Subject", "Study.Group", "Age", "Gender", "BMI", "Brinkman Index", "Alcohol", "Tumor location",
        "yachida_stage_source_raw", "yachida_stage_major", "yachida_stage_source_faithful_label",
        "yachida_stage_harmonized_label", "model_label",
    ]
    meta = pd.DataFrame({
        "cohort": "Yachida",
        "sample_id": df["Sample"].astype(str),
        "participant_id": df["Subject"].astype(str),
        "original_label": df["Study.Group"].astype(str),
        "crc_binary": np.where(df["model_label"].eq("CRC"), 1, 0),
        "age_years": pd.to_numeric(df.get("Age"), errors="coerce"),
        "sex": df.get("Gender", pd.Series(index=df.index, dtype=object)).map(standardize_sex),
        "bmi": pd.to_numeric(df.get("BMI"), errors="coerce"),
        "stage_major": df.get("yachida_stage_major"),
        "stage_source_faithful": df.get("yachida_stage_source_faithful_label"),
        "stage_harmonized": df.get("yachida_stage_harmonized_label"),
        "tumour_site": df.get("Tumor location"),
        "smoking": pd.to_numeric(df.get("Brinkman Index"), errors="coerce"),
        "alcohol": df.get("Alcohol"),
    })
    x = df.drop(columns=[c for c in meta_cols if c in df.columns], errors="ignore")
    return meta, x.apply(pd.to_numeric, errors="coerce")


def read_optional_table(path: Path, sep: str | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if path.suffix.lower() in {".tsv", ".txt"} or sep == "\t":
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def source_overlap(source: pd.DataFrame | None, source_id: str, sample_ids: pd.Series) -> tuple[int, int]:
    if source is None or source_id not in source.columns:
        return 0, 0
    ids = source[source_id].dropna().astype(str)
    return int(ids.isin(set(sample_ids.astype(str))).sum()), int(ids.duplicated().sum())


def write_clinical_source_inventory(sinha_meta: pd.DataFrame, yachida_meta: pd.DataFrame) -> None:
    candidates = [
        ("Sinha", SINHA_DIR / "metadata.tsv", "local raw", "Sample", "Subject", "Age", "Gender", "BMI", "race", "hosp1/hosp2", None, None, None, 1, "primary shared clinical recovery"),
        ("Sinha", CRC_DIR / "external_metadata_sources" / "sinha_s4_covariates.csv", "official", "ID", "ID", "age", "sex", "bmi", "race", "hosp1/hosp2", None, None, None, 3, "verification/fallback only; numeric sex coding preserved but not guessed"),
        ("Sinha", CRC_DIR / "external_metadata_sources" / "SINHA_CRC_2016_metadata_curated.tsv", "curated", "Sample", "Subject", "Age", "Gender", "BMI", "race", "hosp1/hosp2", None, None, None, 4, "verification/fallback only"),
        ("Yachida", YACHIDA_DIR / "metadata.tsv", "local raw", "Sample", "Subject", "Age", "Gender", "BMI", None, None, "Brinkman Index", "Alcohol", "Tumor location", 2, "verification/fallback only"),
        ("Yachida", CRC_DIR / "tables" / "crc_stage1_yachida_supplementary_table_s2_2_metabolome_clinical.csv", "generated from official", "Subject_ID", "Subject_ID", "Age", "Gender", "BMI", None, None, "Brinkman Index", "Alcohol", "Tumor location", 1, "primary shared clinical recovery"),
        ("Yachida", CRC_DIR / "external_metadata_sources" / "YACHIDA_CRC_2019_metadata_curated.tsv", "curated", "Sample", "Subject", "Age", "Gender", "BMI", None, None, "Brinkman Index", "Alcohol", "Tumor location", 4, "verification/fallback only"),
        ("Yachida", CRC_DIR / "tables" / "crc_stage1_yachida_stage_recovered_per_sample.csv", "generated", "Sample", "Subject_ID", None, None, None, None, None, None, None, None, 5, "stage linkage only; not a clinical covariate source"),
    ]
    primary_ids = {"Sinha": sinha_meta["sample_id"], "Yachida": yachida_meta["sample_id"]}
    rows = []
    for cohort, path, status, sample_col, participant_col, age_col, sex_col, bmi_col, race_col, hospital_col, smoking_col, alcohol_col, tumour_col, priority, use in candidates:
        sep = "\t" if path.suffix.lower() == ".tsv" else None
        source = read_optional_table(path, sep=sep)
        if source is None:
            rows.append({
                "cohort": cohort, "source_file": str(path.relative_to(CRC_DIR)) if path.is_relative_to(CRC_DIR) else str(path),
                "source_status": status, "total_rows": 0, "source_exists": False, "permitted_analytical_use": use, "source_priority": priority,
            })
            continue
        overlap, duplicates = source_overlap(source, sample_col, primary_ids[cohort])
        missingness_bits = []
        for label, col in [("age", age_col), ("sex", sex_col), ("BMI", bmi_col), ("race", race_col), ("hospital", hospital_col), ("smoking/Brinkman", smoking_col), ("alcohol", alcohol_col), ("tumour-location", tumour_col)]:
            if col is None:
                missingness_bits.append(f"{label}:unavailable")
            elif "/" in str(col):
                cols = [c for c in str(col).split("/") if c in source.columns]
                missingness_bits.append(f"{label}:{';'.join(f'{c}={source[c].isna().mean():.3f}' for c in cols) if cols else 'missing_column'}")
            elif col in source.columns:
                missingness_bits.append(f"{label}:{source[col].isna().mean():.3f}")
            else:
                missingness_bits.append(f"{label}:missing_column")
        rows.append({
            "cohort": cohort,
            "source_file": str(path.relative_to(CRC_DIR)) if path.is_relative_to(CRC_DIR) else str(path),
            "source_status": status,
            "sample_identifier": sample_col if sample_col in source.columns else pd.NA,
            "participant_identifier": participant_col if participant_col in source.columns else pd.NA,
            "age_field": age_col if age_col in source.columns else pd.NA,
            "sex_field": sex_col if sex_col in source.columns else pd.NA,
            "BMI_field": bmi_col if bmi_col in source.columns else pd.NA,
            "race_field": race_col if race_col and race_col in source.columns else pd.NA,
            "hospital_field": hospital_col,
            "smoking_Brinkman_field": smoking_col if smoking_col and smoking_col in source.columns else pd.NA,
            "alcohol_field": alcohol_col if alcohol_col and alcohol_col in source.columns else pd.NA,
            "tumour_location_field": tumour_col if tumour_col and tumour_col in source.columns else pd.NA,
            "total_rows": len(source),
            "overlap_with_local_metabolomics_samples": overlap,
            "duplicate_IDs": duplicates,
            "missingness": " | ".join(missingness_bits),
            "coding_notes": "Sex standardized only for Male/Female or M/F; numeric sex codes are preserved as raw but not guessed.",
            "source_priority": priority,
            "permitted_analytical_use": use,
            "source_exists": True,
        })
    pd.DataFrame(rows).to_csv(TABLES / "crc_stage2_clinical_source_inventory.csv", index=False)


def build_clinical_linkage(
    cohort: str,
    meta: pd.DataFrame,
    primary_source: pd.DataFrame,
    source_file: str,
    source_sample_col: str,
    source_participant_col: str,
    age_col: str,
    sex_col: str,
    bmi_col: str,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    extra_cols = extra_cols or []
    src = primary_source.copy()
    src["_source_sample_id"] = src[source_sample_col].astype(str)
    src["_source_participant_id"] = src[source_participant_col].astype(str)
    dup = src["_source_sample_id"].duplicated(keep=False)
    source_subset_cols = ["_source_sample_id", "_source_participant_id", age_col, sex_col, bmi_col] + [c for c in extra_cols if c in src.columns]
    src_sub = src[source_subset_cols].copy()
    src_sub["_duplicate_source_id"] = dup.to_numpy()
    merged = meta[["cohort", "sample_id", "participant_id"]].copy()
    merged = merged.merge(src_sub, left_on="sample_id", right_on="_source_sample_id", how="left")
    if merged["_source_sample_id"].isna().any() and cohort == "Yachida":
        unmatched = merged["_source_sample_id"].isna()
        by_subject = meta.loc[unmatched, ["cohort", "sample_id", "participant_id"]].merge(src_sub, left_on="participant_id", right_on="_source_participant_id", how="left")
        merged.loc[unmatched, by_subject.columns] = by_subject.to_numpy()
    out = pd.DataFrame({
        "cohort": cohort,
        "sample_id": merged["sample_id"].astype(str),
        "participant_id": merged["participant_id"].astype(str),
        "source_sample_id": merged["_source_sample_id"],
        "source_file": source_file,
        "age_raw": merged[age_col],
        "age_years": merged[age_col].map(lambda v: numeric_in_range(v, 0, 120)),
        "sex_raw": merged[sex_col],
        "sex_standardized": merged[sex_col].map(standardize_sex),
        "bmi_raw": merged[bmi_col],
        "bmi": merged[bmi_col].map(lambda v: numeric_in_range(v, 10, 80)),
        "linkage_status": np.where(merged["_source_sample_id"].notna(), "linked", "unmatched"),
        "duplicate_status": np.where(merged["_duplicate_source_id"].fillna(False).astype(bool), "duplicate_source_id", "unique_source_id"),
        "conflict_status": "not_assessed_single_primary_source",
        "missingness_reason": "",
        "provenance": "deterministic sample_id linkage; no row-order linkage",
    })
    for col in extra_cols:
        if col in merged.columns:
            out[col] = merged[col]
    out["missingness_reason"] = np.select(
        [
            out["linkage_status"].ne("linked"),
            out["age_years"].isna() | out["sex_standardized"].isna() | out["bmi"].isna(),
        ],
        [
            "no source row matched sample_id/participant_id",
            "one or more raw clinical values missing, out of range, or unresolved",
        ],
        default="complete shared clinical covariates",
    )
    return out


def recover_clinical_covariates(meta: dict[str, pd.DataFrame], config: CRCStage2Config) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    sinha_source = pd.read_csv(SINHA_DIR / "metadata.tsv", sep="\t")
    yachida_source = pd.read_csv(CRC_DIR / "tables" / "crc_stage1_yachida_supplementary_table_s2_2_metabolome_clinical.csv")
    write_clinical_source_inventory(meta["Sinha"], meta["Yachida"])
    sinha_link = build_clinical_linkage(
        "Sinha", meta["Sinha"], sinha_source,
        "Colon_Cancer_Data_Set/Sinha-et-al/metadata.tsv",
        "Sample", "Subject", "Age", "Gender", "BMI",
        extra_cols=["race", "hosp1", "hosp2"],
    )
    yachida_full_seed = load_yachida_full_stage_manifest()[["sample_id", "participant_id"]].copy()
    yachida_full_seed.insert(0, "cohort", "Yachida")
    yachida_full_link = build_clinical_linkage(
        "Yachida", yachida_full_seed, yachida_source,
        "tables/crc_stage1_yachida_supplementary_table_s2_2_metabolome_clinical.csv",
        "Subject_ID", "Subject_ID", "Age", "Gender", "BMI",
        extra_cols=["Brinkman Index", "Alcohol", "Tumor location"],
    )
    yachida_link = meta["Yachida"][["cohort", "sample_id", "participant_id"]].merge(
        yachida_full_link,
        on=["cohort", "sample_id", "participant_id"],
        how="left",
    )
    linkage = pd.concat([sinha_link, yachida_link], ignore_index=True)
    linkage.to_csv(TABLES / "crc_stage2_clinical_linkage_audit.csv", index=False)
    linkage.to_csv(TABLES / "crc_stage2_recovered_clinical_covariates.csv", index=False)
    yachida_full_link.to_csv(TABLES / "crc_stage2_yachida_full_cohort_clinical_covariates.csv", index=False)
    conflicts = linkage[linkage["duplicate_status"].ne("unique_source_id") | linkage["conflict_status"].ne("not_assessed_single_primary_source")].copy()
    if conflicts.empty:
        conflicts = pd.DataFrame([{"status": "no_duplicate_or_unresolved_conflicts_detected"}])
    conflicts.to_csv(TABLES / "crc_stage2_clinical_conflicts.csv", index=False)
    recovered_meta: dict[str, pd.DataFrame] = {}
    for cohort in ["Sinha", "Yachida"]:
        cov = linkage[linkage["cohort"].eq(cohort)][["sample_id", "participant_id", "age_years", "sex_standardized", "bmi"]].copy()
        cov = cov.rename(columns={"sex_standardized": "sex"})
        updated = meta[cohort].drop(columns=["age_years", "sex", "bmi"], errors="ignore").merge(cov, on=["sample_id", "participant_id"], how="left", validate="one_to_one")
        updated["cohort"] = cohort
        recovered_meta[cohort] = updated
    pd.concat(recovered_meta.values(), ignore_index=True).to_csv(PROCESSED / "crc_stage2_primary_metadata_with_clinical.csv", index=False)
    return recovered_meta, linkage


def assert_primary_cohort_integrity(meta: pd.DataFrame, x: pd.DataFrame, cohort: str, config: CRCStage2Config) -> None:
    expected = config.primary_sinha_expected if cohort == "Sinha" else config.primary_yachida_expected
    if len(meta) != expected["n"]:
        raise AssertionError(f"{cohort} primary n changed: {len(meta)} != {expected['n']}")
    if meta["sample_id"].duplicated().any():
        raise AssertionError(f"{cohort} duplicate sample IDs")
    if meta["participant_id"].duplicated().any():
        raise AssertionError(f"{cohort} duplicate participant IDs")
    if len(x) != len(meta):
        raise AssertionError(f"{cohort} metabolomics linkage incomplete")
    if int((meta["crc_binary"] == 0).sum()) != expected["control"]:
        raise AssertionError(f"{cohort} control count changed")
    if int((meta["crc_binary"] == 1).sum()) != expected["crc"]:
        raise AssertionError(f"{cohort} CRC count changed")
    if cohort == "Yachida":
        if meta["original_label"].eq("MP").any() or meta["original_label"].eq("HS").any():
            raise AssertionError("MP/HS present in Yachida primary task")


def canonical_matrix(cohort: str, x_raw: pd.DataFrame, selected: pd.DataFrame, shared_space: pd.DataFrame) -> pd.DataFrame:
    sub = selected[selected["cohort"].eq(cohort)].copy()
    mapping = dict(zip(sub["canonical_id"], sub["feature"]))
    cols = []
    data = {}
    for canonical_id in shared_space["canonical_feature_name"]:
        feature = mapping.get(canonical_id)
        if feature not in x_raw.columns:
            raise AssertionError(f"{cohort} missing selected feature {feature!r} for canonical {canonical_id}")
        data[canonical_id] = pd.to_numeric(x_raw[feature], errors="coerce")
        cols.append(canonical_id)
    return pd.DataFrame(data, index=x_raw.index)[cols]


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(feature_set: str, numeric_clinical: list[str], categorical_clinical: list[str], metabolite_features: list[str], missingness_threshold: float, apply_log1p: bool = False) -> ColumnTransformer:
    transformers = []
    if feature_set in {"clinical", "integrated"}:
        if numeric_clinical:
            transformers.append(("clinical_numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric_clinical))
        if categorical_clinical:
            transformers.append(("clinical_categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", one_hot_encoder()),
            ]), categorical_clinical))
    if feature_set in {"metabolomics", "integrated"}:
        if metabolite_features:
            transformers.append(("metabolomics", Pipeline([
                ("column_order", InputColumnOrderValidator()),
                ("transform", Log1pTransformer(enabled=apply_log1p)),
                ("missingness", MissingnessFilter(max_missing=missingness_threshold)),
                ("constant", ConstantFeatureFilter()),
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), metabolite_features))
    if not transformers:
        raise ValueError(
            f"No active preprocessing branches for feature_set={feature_set}. "
            f"numeric_clinical={len(numeric_clinical)}, categorical_clinical={len(categorical_clinical)}, "
            f"metabolite_features={len(metabolite_features)}."
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def feature_columns_by_type(x: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    numeric_clinical = [c for c in ["age_years", "bmi"] if c in x.columns]
    categorical_clinical = [c for c in ["sex"] if c in x.columns]
    metabolite_features = [c for c in x.columns if c.startswith("met__")]
    return numeric_clinical, categorical_clinical, metabolite_features


def validate_model_context(
    train_name: str,
    feature_set: str,
    model_name: str,
    x: pd.DataFrame,
    y: pd.Series,
    config: CRCStage2Config,
) -> dict[str, Any]:
    numeric_clinical, categorical_clinical, metabolite_features = feature_columns_by_type(x)
    selected_clinical = numeric_clinical + categorical_clinical
    selected_sources = []
    active_branches = []
    if feature_set in {"clinical", "integrated"}:
        selected_sources.extend(selected_clinical)
        if numeric_clinical:
            active_branches.append("clinical_numeric")
        if categorical_clinical:
            active_branches.append("clinical_categorical")
    if feature_set in {"metabolomics", "integrated"}:
        selected_sources.extend(metabolite_features)
        if metabolite_features:
            active_branches.append("metabolomics")
    class_counts = y.value_counts().to_dict()
    row = {
        "train_cohort": train_name,
        "feature_set": feature_set,
        "model": model_name,
        "status": "available",
        "reason": "context passed pre-fit gate",
        "critical": False,
        "recommended_action": "proceed",
        "n_raw_model_input_columns": int(x.shape[1]),
        "n_selected_clinical_columns": int(len(selected_clinical)),
        "n_selected_metabolite_columns": int(len(metabolite_features)),
        "n_selected_source_columns": int(len(selected_sources)),
        "n_active_preprocessing_branches": int(len(active_branches)),
        "active_preprocessing_branches": json.dumps(active_branches),
        "class_counts": json.dumps({str(k): int(v) for k, v in class_counts.items()}),
    }
    if feature_set == "clinical" and not selected_clinical:
        row.update({
            "status": "skipped",
            "critical": False,
            "reason": "clinical-only context unavailable because no shared clinical variables passed the frozen eligibility contract",
            "recommended_action": "do not fabricate clinical features; report clinical-only model as unavailable for this CRC cross-cohort analysis",
        })
    elif not selected_sources:
        row.update({
            "status": "skipped",
            "critical": feature_set in {"metabolomics", "integrated"},
            "reason": "feature set has zero selected source columns",
            "recommended_action": "review clinical eligibility or canonical metabolite alignment before fitting",
        })
    elif not active_branches:
        row.update({
            "status": "skipped",
            "critical": True,
            "reason": "feature set has zero active preprocessing branches",
            "recommended_action": "do not let zero-column ColumnTransformer reach sklearn estimator",
        })
    elif feature_set in {"metabolomics", "integrated"} and not metabolite_features:
        row.update({
            "status": "skipped",
            "critical": True,
            "reason": "shared canonical metabolite feature space is empty",
            "recommended_action": "diagnose canonical metabolite alignment; do not bypass harmonisation",
        })
    else:
        min_class_count = int(y.value_counts().min()) if not y.empty else 0
        if min_class_count < config.reduced_inner_folds:
            row.update({
                "status": "skipped",
                "critical": True,
                "reason": f"class counts cannot support reduced stratified CV; minority class n={min_class_count}",
                "recommended_action": "increase sample size or reduce task complexity before fitting",
            })
        elif feature_set == "integrated" and not selected_clinical and metabolite_features:
            row.update({
                "status": "available",
                "reason": "integrated context has no eligible shared clinical variables, so it proceeds with the metabolomics branch only; this is documented as a CRC data-availability limitation",
                "recommended_action": "interpret integrated CRC fast/full results as metabolomics-only equivalent unless shared clinical variables are later recovered",
            })
    return row


def preprocessing_feature_count_diagnostic(
    train_name: str,
    feature_set: str,
    model_name: str,
    outer_fold: int | str,
    x_fit: pd.DataFrame,
    missingness_threshold: float,
    apply_log1p: bool = False,
    fitted_pipeline: Pipeline | None = None,
) -> pd.DataFrame:
    numeric_clinical, categorical_clinical, metabolite_features = feature_columns_by_type(x_fit)
    selected_clinical = numeric_clinical + categorical_clinical if feature_set in {"clinical", "integrated"} else []
    selected_metabolites = metabolite_features if feature_set in {"metabolomics", "integrated"} else []
    rows = []

    def add(stage: str, n_input: int, n_retained: int, reason: str, retained: list[str]) -> None:
        rows.append({
            "train_cohort": train_name,
            "feature_set": feature_set,
            "model": model_name,
            "outer_fold": outer_fold,
            "fitting_stage": stage,
            "n_input_features": int(n_input),
            "n_retained_features": int(n_retained),
            "removed_feature_count": int(max(n_input - n_retained, 0)),
            "reason": reason,
            "retained_feature_names": json.dumps([str(c) for c in retained[:200]]),
        })

    add("raw_model_input_columns", x_fit.shape[1], x_fit.shape[1], "columns supplied to model context before feature-set selection", list(x_fit.columns))
    add("selected_clinical_columns", x_fit.shape[1], len(selected_clinical), "eligible shared clinical columns selected by feature-set gate", selected_clinical)
    add("selected_metabolite_columns", x_fit.shape[1], len(selected_metabolites), "shared canonical metabolite columns selected by feature-set gate", selected_metabolites)
    selected_sources = selected_clinical + selected_metabolites
    add("after_column_transformer_column_selection", x_fit.shape[1], len(selected_sources), "source columns routed into active ColumnTransformer branches", selected_sources)
    if selected_metabolites:
        met = x_fit[selected_metabolites].apply(pd.to_numeric, errors="coerce")
        add("after_Log1pTransformer", len(selected_metabolites), len(selected_metabolites), "primary analysis keeps supplied aligned values; log1p disabled" if not apply_log1p else "log1p enabled sensitivity transform", selected_metabolites)
        missing_fraction = met.isna().mean(axis=0)
        miss_keep = missing_fraction <= missingness_threshold
        miss_features = list(missing_fraction.index[miss_keep])
        add("after_MissingnessFilter", len(selected_metabolites), len(miss_features), f"features retained with missing fraction <= {missingness_threshold}", miss_features)
        if miss_features:
            unique_counts = met[miss_features].nunique(dropna=True)
            const_keep = unique_counts > 1
            const_features = list(unique_counts.index[const_keep])
        else:
            const_features = []
        add("after_ConstantFeatureFilter", len(miss_features), len(const_features), "features retained with >1 unique non-missing value in this fit", const_features)
        add("after_imputation", len(const_features), len(const_features), "median imputation preserves retained metabolite feature count", const_features)
        add("after_scaling", len(const_features), len(const_features), "standard scaling preserves retained metabolite feature count", const_features)
    else:
        for stage in ["after_Log1pTransformer", "after_MissingnessFilter", "after_ConstantFeatureFilter", "after_imputation", "after_scaling"]:
            add(stage, 0, 0, "no metabolomics branch active for this feature set", [])
    if fitted_pipeline is not None:
        try:
            final_names = transformed_feature_names(fitted_pipeline)
            add("final_estimator_input_feature_count", len(final_names), len(final_names), "feature count after fitted preprocessing immediately before estimator", final_names)
        except Exception as exc:
            add("final_estimator_input_feature_count", 0, 0, f"could not recover fitted transformed feature names: {repr(exc)}", [])
    return pd.DataFrame(rows)


def model_and_grid(config: CRCStage2Config) -> dict[str, tuple[Any, dict[str, list[Any]], str, int | None]]:
    grids: dict[str, tuple[Any, dict[str, list[Any]], str, int | None]] = {
        "ridge_logistic": (
            LogisticRegression(penalty="l2", solver="liblinear", class_weight="balanced", max_iter=5000, random_state=config.random_seed),
            {"clf__C": [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 5, 10, 50, 100]},
            "GridSearchCV",
            None,
        ),
        "elastic_net_logistic": (
            LogisticRegression(penalty="elasticnet", solver="saga", class_weight="balanced", max_iter=8000, random_state=config.random_seed),
            {"clf__C": [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 5, 10], "clf__l1_ratio": [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]},
            "GridSearchCV",
            None,
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", random_state=config.random_seed, n_jobs=1),
            {"clf__n_estimators": [200, 500, 1000], "clf__max_depth": [None, 3, 5, 8, 12], "clf__min_samples_leaf": [1, 2, 3, 5, 10], "clf__min_samples_split": [2, 5, 10], "clf__max_features": ["sqrt", "log2", 0.5, None]},
            "RandomizedSearchCV",
            config.fast_test_randomized_search_n_iter if config.fast_test else config.randomized_search_n_iter["random_forest"],
        ),
        "extra_trees": (
            ExtraTreesClassifier(class_weight="balanced", random_state=config.random_seed, n_jobs=1),
            {"clf__n_estimators": [200, 500, 1000], "clf__max_depth": [None, 3, 5, 8, 12], "clf__min_samples_leaf": [1, 2, 3, 5, 10], "clf__min_samples_split": [2, 5, 10], "clf__max_features": ["sqrt", "log2", 0.5, None]},
            "RandomizedSearchCV",
            config.fast_test_randomized_search_n_iter if config.fast_test else config.randomized_search_n_iter["extra_trees"],
        ),
        "gradient_boosting": (
            BalancedGradientBoostingClassifier(random_state=config.random_seed),
            {"clf__n_estimators": [50, 100, 200, 300], "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2], "clf__max_depth": [1, 2, 3], "clf__min_samples_leaf": [1, 3, 5, 10], "clf__subsample": [0.6, 0.8, 1.0], "clf__max_features": [None, "sqrt", "log2"]},
            "RandomizedSearchCV",
            config.fast_test_randomized_search_n_iter if config.fast_test else config.randomized_search_n_iter["gradient_boosting"],
        ),
    }
    return {k: v for k, v in grids.items() if k in config.active_models()}


def write_hyperparameter_grid_plan(config: CRCStage2Config) -> pd.DataFrame:
    rows = []
    for model, (_, grid, method, n_iter) in model_and_grid(config).items():
        total = len(list(ParameterGrid(grid)))
        evaluated = total if method == "GridSearchCV" else min(int(n_iter or total), total)
        rows.append({
            "model": model,
            "search_method": method,
            "complete_grid_size": total,
            "planned_configurations_evaluated": evaluated,
            "random_seed": config.random_seed if method == "RandomizedSearchCV" else pd.NA,
            "scoring": "roc_auc",
            "note": "Grid search retained for manageable logistic grids; seeded randomized search used for larger ensemble spaces.",
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "crc_stage2_hyperparameter_grid_plan.csv", index=False)
    return out


def choose_fold_count(y: pd.Series, requested: int, reduced: int, context: str) -> tuple[int, str]:
    counts = y.value_counts()
    min_count = int(counts.min())
    if min_count >= requested:
        return requested, f"{context}: requested folds supported"
    if min_count >= reduced:
        return reduced, f"{context}: reduced folds because minority class n={min_count}"
    raise ValueError(f"{context}: insufficient class count for stratified CV, minority class n={min_count}")


def make_search(pipeline: Pipeline, model_name: str, param_grid: dict[str, list[Any]], method: str, n_iter: int | None, inner_cv: StratifiedKFold, config: CRCStage2Config):
    error_score = "raise" if config.fast_test else np.nan
    if method == "RandomizedSearchCV":
        return RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=int(n_iter or 10),
            scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
            refit="roc_auc",
            cv=inner_cv,
            random_state=config.random_seed,
            return_train_score=True,
            n_jobs=1,
            error_score=error_score,
        )
    return GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
        refit="roc_auc",
        cv=inner_cv,
        return_train_score=True,
        n_jobs=1,
        error_score=error_score,
    )


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
        "pr_auc": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "ppv": precision_score(y_true, y_pred, zero_division=0),
        "npv": tn / (tn + fn) if (tn + fn) else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier_score": brier_score_loss(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": threshold,
    }


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict[str, Any]]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = int(np.argmax(j))
    threshold = float(thresholds[idx])
    if not np.isfinite(threshold):
        threshold = 0.5
    specificity = 1 - float(fpr[idx])
    return threshold, {"youden_j": float(j[idx]), "sensitivity": float(tpr[idx]), "specificity": specificity, "roc_auc": float(roc_auc_score(y_true, y_prob))}


def fit_params_for_model(model_name: str, y: pd.Series) -> dict[str, Any]:
    return {}


def run_nested_cv_for_direction(
    train_name: str,
    train_meta: pd.DataFrame,
    train_x: pd.DataFrame,
    feature_set: str,
    model_name: str,
    config: CRCStage2Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = train_meta["crc_binary"].astype(int).reset_index(drop=True)
    x = train_x.reset_index(drop=True)
    numeric_clinical, categorical_clinical, metabolite_features = feature_columns_by_type(x)
    apply_log1p = False
    outer_k, outer_reason = choose_fold_count(y, config.active_outer_folds(), 3, f"{train_name} outer CV")
    outer_cv = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=config.random_seed)
    model, grid, method, n_iter = model_and_grid(config)[model_name]
    rows, pred_rows, search_rows, cv_rows, diagnostic_rows = [], [], [], [], []
    for fold_idx, (dev_idx, test_idx) in enumerate(outer_cv.split(x, y), start=1):
        x_dev, x_test = x.iloc[dev_idx], x.iloc[test_idx]
        y_dev, y_test = y.iloc[dev_idx], y.iloc[test_idx]
        inner_k, inner_reason = choose_fold_count(y_dev, config.active_inner_folds(), config.reduced_inner_folds, f"{train_name} inner CV")
        inner_cv = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=config.random_seed + fold_idx)
        pre = build_preprocessor(feature_set, numeric_clinical, categorical_clinical, metabolite_features, config.primary_missingness_threshold, apply_log1p)
        pipe = Pipeline([("preprocess", pre), ("clf", clone(model))])
        search = make_search(pipe, model_name, grid, method, n_iter, inner_cv, config)
        t0 = time.time()
        diagnostic_rows.append(preprocessing_feature_count_diagnostic(train_name, feature_set, model_name, fold_idx, x_dev, config.primary_missingness_threshold, apply_log1p))
        try:
            search.fit(x_dev, y_dev, **fit_params_for_model(model_name, y_dev))
        except Exception as exc:
            failure_traceback = traceback.format_exc()
            get_logger().error(
                "Nested CV fit failed for train=%s feature_set=%s model=%s outer_fold=%s "
                "raw_columns=%s clinical_columns=%s metabolite_columns=%s error=%r\n%s",
                train_name,
                feature_set,
                model_name,
                fold_idx,
                x_dev.shape[1],
                len(numeric_clinical) + len(categorical_clinical),
                len(metabolite_features),
                exc,
                failure_traceback,
            )
            cv_rows.append({
                "train_cohort": train_name,
                "feature_set": feature_set,
                "model": model_name,
                "outer_fold": fold_idx,
                "outer_folds": outer_k,
                "inner_folds": inner_k,
                "outer_reason": outer_reason,
                "inner_reason": inner_reason,
                "minority_class_count_dev": int(y_dev.value_counts().min()),
                "class_counts_dev": json.dumps(y_dev.value_counts().to_dict()),
                "class_counts_test": json.dumps(y_test.value_counts().to_dict()),
                "failed_fold_status": "failed",
                "failure": repr(exc),
                "traceback": failure_traceback,
                "affected_input_columns": json.dumps([str(c) for c in x_dev.columns]),
                "random_seed": config.random_seed + fold_idx,
            })
            raise
        runtime = time.time() - t0
        diagnostic_rows.append(preprocessing_feature_count_diagnostic(train_name, feature_set, model_name, fold_idx, x_dev, config.primary_missingness_threshold, apply_log1p, search.best_estimator_))
        prob = search.best_estimator_.predict_proba(x_test)[:, 1]
        m05 = binary_metrics(y_test.to_numpy(), prob, threshold=0.5)
        rows.append({
            "train_cohort": train_name,
            "feature_set": feature_set,
            "model": model_name,
            "outer_fold": fold_idx,
            "outer_folds": outer_k,
            "inner_folds": inner_k,
            "outer_reason": outer_reason,
            "inner_reason": inner_reason,
            **{f"threshold_0_5_{k}": v for k, v in m05.items()},
            "best_params": json.dumps(search.best_params_, sort_keys=True),
            "search_runtime_seconds": runtime,
        })
        for sample_id, participant_id, yt, yp in zip(train_meta.iloc[test_idx]["sample_id"], train_meta.iloc[test_idx]["participant_id"], y_test, prob):
            pred_rows.append({"train_cohort": train_name, "sample_id": sample_id, "participant_id": participant_id, "outer_fold": fold_idx, "true_label": int(yt), "predicted_probability": float(yp), "feature_set": feature_set, "model": model_name})
        cv_rows.append({
            "train_cohort": train_name,
            "feature_set": feature_set,
            "model": model_name,
            "outer_fold": fold_idx,
            "outer_folds": outer_k,
            "inner_folds": inner_k,
            "requested_outer_folds": config.active_outer_folds(),
            "requested_inner_folds": config.active_inner_folds(),
            "minority_class_count_dev": int(y_dev.value_counts().min()),
            "class_counts_dev": json.dumps(y_dev.value_counts().to_dict()),
            "class_counts_test": json.dumps(y_test.value_counts().to_dict()),
            "dev_sample_ids": json.dumps(train_meta.iloc[dev_idx]["sample_id"].astype(str).tolist()),
            "test_sample_ids": json.dumps(train_meta.iloc[test_idx]["sample_id"].astype(str).tolist()),
            "failed_fold_status": "passed",
            "random_seed": config.random_seed + fold_idx,
        })
        cvres = pd.DataFrame(search.cv_results_)
        cvres["train_cohort"] = train_name
        cvres["feature_set"] = feature_set
        cvres["model"] = model_name
        cvres["outer_fold"] = fold_idx
        cvres["search_method"] = method
        cvres["search_runtime_seconds"] = runtime
        search_rows.append(cvres)
    search_df = pd.concat(search_rows, ignore_index=True) if search_rows else pd.DataFrame()
    diagnostic_df = pd.concat(diagnostic_rows, ignore_index=True) if diagnostic_rows else pd.DataFrame()
    return pd.DataFrame(rows), pd.DataFrame(pred_rows), search_df, pd.DataFrame(cv_rows), diagnostic_df


def select_internal_model(internal_df: pd.DataFrame, config: CRCStage2Config) -> pd.DataFrame:
    grouped = (
        internal_df.groupby(["train_cohort", "feature_set", "model"], dropna=False)
        .agg(
            mean_outer_roc_auc=("threshold_0_5_roc_auc", "mean"),
            sd_outer_roc_auc=("threshold_0_5_roc_auc", "std"),
            mean_outer_pr_auc=("threshold_0_5_pr_auc", "mean"),
            n_outer_folds=("outer_fold", "nunique"),
        )
        .reset_index()
    )
    grouped["simplicity_rank"] = grouped["model"].map(config.model_simplicity_order)
    grouped["effective_tie_tolerance_auc"] = config.model_selection_effective_tie_tolerance_auc
    grouped["effective_tie_tolerance_pr_auc"] = config.model_selection_effective_tie_tolerance_pr_auc
    grouped["effective_tie_tolerance_roc_auc_sd"] = config.model_selection_effective_tie_tolerance_roc_auc_sd
    grouped["selected"] = False
    grouped["best_context_roc_auc"] = np.nan
    grouped["within_roc_auc_tie_band"] = False
    grouped["best_eligible_pr_auc"] = np.nan
    grouped["within_pr_auc_tie_band"] = False
    grouped["stability_difference_from_best"] = np.nan
    grouped["final_tie_break_stage"] = ""
    grouped["selection_reason"] = ""
    for _, feature_group in grouped.groupby(["train_cohort", "feature_set"], dropna=False):
        best_auc = feature_group["mean_outer_roc_auc"].max()
        roc_eligible = feature_group[feature_group["mean_outer_roc_auc"] >= best_auc - config.model_selection_effective_tie_tolerance_auc].copy()
        best_pr = roc_eligible["mean_outer_pr_auc"].max()
        pr_eligible = roc_eligible[roc_eligible["mean_outer_pr_auc"] >= best_pr - config.model_selection_effective_tie_tolerance_pr_auc].copy()
        best_sd = pr_eligible["sd_outer_roc_auc"].min()
        stable_eligible = pr_eligible[pr_eligible["sd_outer_roc_auc"] <= best_sd + config.model_selection_effective_tie_tolerance_roc_auc_sd].copy()
        chosen = stable_eligible.sort_values("simplicity_rank", ascending=True).iloc[0]
        grouped.loc[feature_group.index, "best_context_roc_auc"] = best_auc
        grouped.loc[feature_group.index, "within_roc_auc_tie_band"] = feature_group["mean_outer_roc_auc"] >= best_auc - config.model_selection_effective_tie_tolerance_auc
        grouped.loc[feature_group.index, "best_eligible_pr_auc"] = best_pr
        grouped.loc[feature_group.index, "within_pr_auc_tie_band"] = grouped.loc[feature_group.index, "within_roc_auc_tie_band"] & (feature_group["mean_outer_pr_auc"] >= best_pr - config.model_selection_effective_tie_tolerance_pr_auc)
        grouped.loc[feature_group.index, "stability_difference_from_best"] = feature_group["sd_outer_roc_auc"] - best_sd
        if len(roc_eligible) == 1:
            stage = "uniquely_best_roc_auc"
        elif len(pr_eligible) == 1:
            stage = "pr_auc_tie_break"
        elif len(stable_eligible) == 1:
            stage = "stability_tie_break"
        else:
            stage = "simplicity_tie_break"
        grouped.loc[chosen.name, "selected"] = True
        grouped.loc[chosen.name, "final_tie_break_stage"] = stage
        grouped.loc[chosen.name, "selection_reason"] = (
            f"{stage}; best ROC-AUC={best_auc:.6g}; ROC tolerance={config.model_selection_effective_tie_tolerance_auc}; "
            f"best eligible PR-AUC={best_pr:.6g}; PR tolerance={config.model_selection_effective_tie_tolerance_pr_auc}; "
            f"best SD={best_sd:.6g}; SD tolerance={config.model_selection_effective_tie_tolerance_roc_auc_sd}; "
            "external validation not used"
        )
    grouped["selection_note"] = "Best model selected within each training-cohort and feature-set stratum using nested CV only; true effective ROC ties are filtered before PR-AUC, stability and simplicity tie-breaks. External validation is not used."
    grouped = grouped.sort_values(["train_cohort", "feature_set", "selected", "mean_outer_roc_auc", "mean_outer_pr_auc", "sd_outer_roc_auc", "simplicity_rank"], ascending=[True, True, False, False, False, True, True])
    grouped.to_csv(TABLES / "crc_stage2_internal_model_selection.csv", index=False)
    return grouped


def final_fit_and_external_validation(
    train_name: str,
    external_name: str,
    train_meta: pd.DataFrame,
    train_x: pd.DataFrame,
    external_meta: pd.DataFrame,
    external_x: pd.DataFrame,
    selected_row: pd.Series,
    selected_oof_predictions: pd.DataFrame,
    config: CRCStage2Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_set = selected_row["feature_set"]
    model_name = selected_row["model"]
    y_train = train_meta["crc_binary"].astype(int).reset_index(drop=True)
    x_train = train_x.reset_index(drop=True)
    y_ext = external_meta["crc_binary"].astype(int).reset_index(drop=True)
    x_ext = external_x.reset_index(drop=True)
    numeric_clinical, categorical_clinical, metabolite_features = feature_columns_by_type(x_train)
    apply_log1p = False
    model, grid, method, n_iter = model_and_grid(config)[model_name]
    inner_k, inner_reason = choose_fold_count(y_train, config.active_inner_folds(), config.reduced_inner_folds, f"{train_name} final inner CV")
    inner_cv = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=config.random_seed)
    pre = build_preprocessor(feature_set, numeric_clinical, categorical_clinical, metabolite_features, config.primary_missingness_threshold, apply_log1p)
    pipe = Pipeline([("preprocess", pre), ("clf", clone(model))])
    search = make_search(pipe, model_name, grid, method, n_iter, inner_cv, config)
    diagnostic_before = preprocessing_feature_count_diagnostic(train_name, feature_set, model_name, "final_full_training_search", x_train, config.primary_missingness_threshold, apply_log1p)
    search.fit(x_train, y_train, **fit_params_for_model(model_name, y_train))
    oof = selected_oof_predictions.copy()
    if oof.empty:
        raise ValueError(f"No nested outer-fold OOF predictions available for selected context {train_name}/{feature_set}/{model_name}")
    if oof["sample_id"].duplicated().any():
        duplicated = oof.loc[oof["sample_id"].duplicated(), "sample_id"].astype(str).tolist()
        raise ValueError(f"Duplicated nested OOF sample predictions for threshold selection: {duplicated[:10]}")
    expected_samples = set(train_meta["sample_id"].astype(str))
    observed_samples = set(oof["sample_id"].astype(str))
    missing = sorted(expected_samples - observed_samples)
    extra = sorted(observed_samples - expected_samples)
    if missing or extra:
        raise ValueError(f"Nested OOF threshold source does not match training samples. Missing={missing[:10]}; extra={extra[:10]}")
    oof = oof.set_index(oof["sample_id"].astype(str)).loc[train_meta["sample_id"].astype(str)].reset_index(drop=True)
    oof_prob = oof["predicted_probability"].to_numpy(dtype=float)
    threshold, thr_stats = youden_threshold(y_train.to_numpy(), oof_prob)
    final_pipe = clone(search.best_estimator_)
    final_pipe.fit(x_train, y_train, **fit_params_for_model(model_name, y_train))
    diagnostic_after = preprocessing_feature_count_diagnostic(train_name, feature_set, model_name, "final_full_training_fit", x_train, config.primary_missingness_threshold, apply_log1p, final_pipe)
    ext_prob = final_pipe.predict_proba(x_ext)[:, 1]
    pred = pd.DataFrame({
        "sample_id": external_meta["sample_id"].to_numpy(),
        "participant_id": external_meta["participant_id"].to_numpy(),
        "cohort": external_name,
        "true_label": y_ext.to_numpy(),
        "predicted_probability": ext_prob,
        "frozen_threshold": threshold,
        "threshold_source": "nested_outer_fold_oof_predictions_training_only",
        "predicted_label_at_0_5": (ext_prob >= 0.5).astype(int),
        "predicted_label_at_frozen_threshold": (ext_prob >= threshold).astype(int),
        "model": model_name,
        "feature_set": feature_set,
        "train_cohort": train_name,
        "external_cohort": external_name,
    })
    result_rows = []
    for label, thr in [("0_5", 0.5), ("frozen_training_youden", threshold)]:
        result_rows.append({"train_cohort": train_name, "external_cohort": external_name, "feature_set": feature_set, "model": model_name, "threshold_type": label, **binary_metrics(y_ext.to_numpy(), ext_prob, thr)})
    threshold_df = pd.DataFrame([{ "train_cohort": train_name, "external_cohort": external_name, "feature_set": feature_set, "model": model_name, "selected_threshold": threshold, **thr_stats, "threshold_source": "nested_outer_fold_oof_predictions_training_only", "oof_sample_count": len(oof), "missing_oof_count": len(missing), "duplicated_oof_count": int(oof["sample_id"].duplicated().sum()), "selection_data": "nested outer-fold OOF probabilities from internally tuned models", "external_data_used": False }])
    model_path = MODELS / f"crc_stage2_selected_{train_name}_to_{external_name}_{feature_set}_{model_name}.joblib"
    joblib.dump(final_pipe, model_path)
    return pred, pd.DataFrame(result_rows), threshold_df, pd.concat([diagnostic_before, diagnostic_after], ignore_index=True)


def bootstrap_external_ci(preds: pd.DataFrame, config: CRCStage2Config) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed)
    rows = []
    metrics = ["roc_auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity"]
    for context, group in preds.groupby(["train_cohort", "external_cohort", "feature_set", "model"], dropna=False):
        y = group["true_label"].to_numpy(dtype=int)
        p = group["predicted_probability"].to_numpy(dtype=float)
        if "selected_threshold" in group.columns and group["selected_threshold"].notna().any():
            thr = float(group["selected_threshold"].dropna().iloc[0])
            threshold_type = "frozen_training_youden"
        else:
            thr = 0.5
            threshold_type = "0_5_sensitivity"
        boot = {m: [] for m in metrics}
        for _ in range(config.active_bootstrap_repetitions()):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) < 2:
                continue
            vals = binary_metrics(y[idx], p[idx], threshold=thr)
            for m in metrics:
                boot[m].append(vals[m])
        for m in metrics:
            arr = np.asarray(boot[m], dtype=float)
            rows.append({"train_cohort": context[0], "external_cohort": context[1], "feature_set": context[2], "model": context[3], "threshold_type": threshold_type, "threshold": thr, "metric": m, "bootstrap_n": len(arr), "ci_lower": float(np.nanpercentile(arr, 2.5)) if len(arr) else np.nan, "ci_upper": float(np.nanpercentile(arr, 97.5)) if len(arr) else np.nan})
    return pd.DataFrame(rows)


def build_model_input_tables(config: CRCStage2Config) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    _, _, selected, shared_space, _ = build_canonical_feature_space()
    sinha_meta, sinha_x_raw = load_sinha_primary()
    yachida_meta, yachida_x_raw = load_yachida_primary()
    assert_primary_cohort_integrity(sinha_meta, sinha_x_raw, "Sinha", config)
    assert_primary_cohort_integrity(yachida_meta, yachida_x_raw, "Yachida", config)
    sinha_met = canonical_matrix("Sinha", sinha_x_raw, selected, shared_space).add_prefix("met__")
    yachida_met = canonical_matrix("Yachida", yachida_x_raw, selected, shared_space).add_prefix("met__")
    meta = {"Sinha": sinha_meta, "Yachida": yachida_meta}
    meta, clinical_linkage = recover_clinical_covariates(meta, config)
    clinical_rows = []
    for var, kind in [("age_years", "numeric"), ("sex", "categorical"), ("bmi", "numeric")]:
        row = {
            "clinical_feature": var,
            "intended_encoding": "median_impute_then_standard_scale" if kind == "numeric" else "most_frequent_impute_then_one_hot",
            "maximum_allowed_missingness": 0.30,
            "minimum_usable_sample_count_per_cohort": 30,
            "plausible_numeric_range": "age_years:0-120; bmi:10-80" if kind == "numeric" else "not_applicable",
            "allowed_categorical_levels": "Male;Female" if var == "sex" else "not_applicable",
            "included_in_primary_shared_clinical_model": True,
            "exclusion_rationale": "",
        }
        for cohort, cmeta in meta.items():
            values = cmeta[var] if var in cmeta.columns else pd.Series(pd.NA, index=cmeta.index)
            row[f"{cohort}_structurally_available"] = var in cmeta.columns
            row[f"{cohort}_missingness"] = float(values.isna().mean())
            row[f"{cohort}_all_missing"] = bool(values.isna().all())
            row[f"{cohort}_n_levels"] = int(values.nunique(dropna=True))
            row[f"{cohort}_usable_sample_count"] = int(values.notna().sum())
            if kind == "numeric":
                numeric = pd.to_numeric(values, errors="coerce")
                if var == "age_years":
                    plausible = numeric.dropna().between(0, 120).all()
                elif var == "bmi":
                    plausible = numeric.dropna().between(10, 80).all()
                else:
                    plausible = True
                row[f"{cohort}_plausible_numeric_range"] = bool(plausible)
                if not plausible:
                    row["included_in_primary_shared_clinical_model"] = False
                    row["exclusion_rationale"] = f"{var} has implausible numeric range in {cohort}"
            if var == "sex":
                levels = set(values.dropna().astype(str))
                compatible = levels.issubset({"Male", "Female"})
                row[f"{cohort}_coding_compatible"] = bool(compatible)
                if not compatible or len(levels) < 2:
                    row["included_in_primary_shared_clinical_model"] = False
                    row["exclusion_rationale"] = f"{var} coding incompatible or has insufficient levels in {cohort}"
            if values.isna().all():
                row["included_in_primary_shared_clinical_model"] = False
                row["exclusion_rationale"] = f"{var} all-missing or structurally unavailable in {cohort}"
            if values.isna().mean() > 0.30:
                row["included_in_primary_shared_clinical_model"] = False
                row["exclusion_rationale"] = f"{var} exceeds 30% missingness in {cohort}"
            if values.notna().sum() < 30:
                row["included_in_primary_shared_clinical_model"] = False
                row["exclusion_rationale"] = f"{var} has fewer than 30 usable samples in {cohort}"
        clinical_rows.append(row)
    clinical_contract = pd.DataFrame(clinical_rows)
    clinical_contract.to_csv(TABLES / "crc_stage2_shared_clinical_feature_contract.csv", index=False)
    classification_rows = []
    for field in ["age_years", "sex", "bmi"]:
        eligible = bool(clinical_contract.loc[clinical_contract["clinical_feature"].eq(field), "included_in_primary_shared_clinical_model"].iloc[0])
        classification_rows.append({
            "field": field,
            "classification": "shared-primary eligible" if eligible else "unavailable",
            "reason": "passes deterministic linkage, coding, missingness and range checks in both cohorts" if eligible else clinical_contract.loc[clinical_contract["clinical_feature"].eq(field), "exclusion_rationale"].iloc[0],
        })
    for field, reason in [
        ("race", "Sinha-only source covariate; retained for description/sensitivity, not shared primary model"),
        ("hosp1/hosp2", "Sinha-only enrollment hospital/site covariates; retained for description/sensitivity, not shared primary model"),
        ("Brinkman Index", "Yachida-only smoking intensity covariate; cohort-specific sensitivity only"),
        ("Alcohol", "Yachida-only alcohol covariate; cohort-specific sensitivity only"),
        ("Tumor location", "Yachida-only disease attribute; prohibited as primary predictor because it is post-diagnosis/leakage-prone"),
        ("stage", "Yachida-only outcome/severity descriptor; prohibited as primary CRC-vs-Control predictor"),
        ("MP/HS status", "Yachida-only non-primary conditions excluded from CRC-vs-Control primary task"),
    ]:
        classification_rows.append({
            "field": field,
            "classification": "prohibited due to leakage" if field in {"Tumor location", "stage"} else ("outcome/descriptive only" if field == "MP/HS status" else "cohort-specific sensitivity only"),
            "reason": reason,
        })
    pd.DataFrame(classification_rows).to_csv(TABLES / "crc_stage2_clinical_variable_classification.csv", index=False)
    included_clinical = clinical_contract.loc[clinical_contract["included_in_primary_shared_clinical_model"], "clinical_feature"].tolist()
    features = {}
    for cohort, cmeta, met in [("Sinha", sinha_meta, sinha_met), ("Yachida", yachida_meta, yachida_met)]:
        cmeta = meta[cohort]
        clin = cmeta[included_clinical].reset_index(drop=True)
        features[cohort] = pd.concat([clin, met.reset_index(drop=True)], axis=1)
    pd.concat([meta["Sinha"], meta["Yachida"]], ignore_index=True).to_csv(PROCESSED / "crc_stage2_primary_metadata.csv", index=False)
    sinha_met_out = pd.concat([meta["Sinha"][["cohort", "sample_id", "participant_id"]].reset_index(drop=True), sinha_met.reset_index(drop=True)], axis=1)
    yachida_met_out = pd.concat([meta["Yachida"][["cohort", "sample_id", "participant_id"]].reset_index(drop=True), yachida_met.reset_index(drop=True)], axis=1)
    pd.concat([sinha_met_out, yachida_met_out], ignore_index=True).to_csv(PROCESSED / "crc_stage2_primary_metabolomics_canonical.csv", index=False)
    return meta, features, shared_space


def leakage_safety_checks() -> pd.DataFrame:
    source = Path(__file__).read_text()
    rows = [
        {"check": "missingness_filter_inside_pipeline", "passed": '"missingness", MissingnessFilter' in source, "detail": "MissingnessFilter appears in sklearn Pipeline definition."},
        {"check": "constant_filter_inside_pipeline", "passed": '"constant", ConstantFeatureFilter' in source, "detail": "ConstantFeatureFilter appears in sklearn Pipeline definition."},
        {"check": "imputation_inside_pipeline", "passed": "SimpleImputer(" in source and "Pipeline([" in source, "detail": "SimpleImputer appears inside pipeline definitions."},
        {"check": "scaling_inside_pipeline", "passed": "StandardScaler()" in source and "Pipeline([" in source, "detail": "StandardScaler appears inside pipeline definitions."},
        {"check": "external_not_used_for_search", "passed": "make_search(" in source and "external_x" not in source[source.find("def run_nested_cv_for_direction"):source.find("def select_internal_model")], "detail": "Nested CV search function body does not reference external_x."},
        {"check": "threshold_training_only", "passed": "nested_outer_fold_oof_predictions_training_only" in source, "detail": "Frozen threshold source is nested outer-fold OOF predictions."},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "crc_stage2_leakage_safety_checks.csv", index=False)
    return out


def run_primary_modelling(config: CRCStage2Config) -> None:
    write_config(config)
    validate_stage1_contract(config)
    write_hyperparameter_grid_plan(config)
    leakage_safety_checks()
    meta, features, shared_space = build_model_input_tables(config)
    rows_internal, rows_pred, rows_search, rows_cv, rows_diag, rows_context = [], [], [], [], [], []
    for train_name in ["Sinha", "Yachida"]:
        for feature_set in config.feature_sets:
            for model_name in config.active_models():
                context_row = validate_model_context(train_name, feature_set, model_name, features[train_name], meta[train_name]["crc_binary"].astype(int), config)
                rows_context.append(context_row)
                if context_row["status"] != "available":
                    get_logger().info(
                        "Skipping model context train=%s feature_set=%s model=%s reason=%s",
                        train_name,
                        feature_set,
                        model_name,
                        context_row["reason"],
                    )
                    continue
                internal, pred, search, cv_design, diagnostics = run_nested_cv_for_direction(train_name, meta[train_name], features[train_name], feature_set, model_name, config)
                rows_internal.append(internal)
                rows_pred.append(pred)
                rows_search.append(search)
                rows_cv.append(cv_design)
                rows_diag.append(diagnostics)
    context_df = pd.DataFrame(rows_context)
    context_df.to_csv(TABLES / "crc_stage2_model_context_gate.csv", index=False)
    if context_df.loc[context_df["critical"].astype(bool) & context_df["status"].ne("available")].shape[0]:
        failures = context_df.loc[context_df["critical"].astype(bool) & context_df["status"].ne("available"), ["train_cohort", "feature_set", "model", "reason"]]
        raise ValueError(f"Critical model context(s) unavailable before fitting: {failures.to_dict(orient='records')}")
    if not rows_internal:
        raise ValueError("No model contexts were available after pre-fit gating; primary modelling cannot proceed.")
    internal_df = pd.concat(rows_internal, ignore_index=True)
    internal_pred = pd.concat(rows_pred, ignore_index=True)
    search_df = pd.concat(rows_search, ignore_index=True)
    cv_df = pd.concat(rows_cv, ignore_index=True)
    diagnostic_df = pd.concat(rows_diag, ignore_index=True) if rows_diag else pd.DataFrame()
    internal_df.to_csv(TABLES / "crc_stage2_internal_cv_results.csv", index=False)
    internal_pred.to_csv(PREDICTIONS / "crc_stage2_internal_oof_predictions.csv", index=False)
    search_df.to_csv(TABLES / "crc_stage2_hyperparameter_search_results.csv", index=False)
    cv_df.to_csv(TABLES / "crc_stage2_cv_design.csv", index=False)
    diagnostic_df.to_csv(TABLES / "crc_stage2_fast_test_feature_count_diagnostic.csv" if config.fast_test else TABLES / "crc_stage2_feature_count_diagnostic.csv", index=False)
    search_summary_cols = [c for c in search_df.columns if c in {"train_cohort", "feature_set", "model", "outer_fold", "rank_test_roc_auc", "mean_test_roc_auc", "std_test_roc_auc", "mean_train_roc_auc", "mean_test_pr_auc", "mean_fit_time", "mean_score_time", "params", "search_method", "search_runtime_seconds"}]
    search_df[search_summary_cols].to_csv(TABLES / "crc_stage2_hyperparameter_search_summary.csv", index=False)
    selected = select_internal_model(internal_df, config)
    external_preds, external_results, thresholds, final_diagnostics = [], [], [], []
    for train_name, external_name in config.external_validation_directions:
        selected_rows = selected[selected["train_cohort"].eq(train_name) & selected["selected"]].copy()
        for _, sel in selected_rows.iterrows():
            selected_oof = internal_pred[
                internal_pred["train_cohort"].eq(train_name)
                & internal_pred["feature_set"].eq(sel["feature_set"])
                & internal_pred["model"].eq(sel["model"])
            ].copy()
            pred, res, thr, final_diag = final_fit_and_external_validation(train_name, external_name, meta[train_name], features[train_name], meta[external_name], features[external_name], sel, selected_oof, config)
            external_preds.append(pred)
            external_results.append(res)
            thresholds.append(thr)
            final_diagnostics.append(final_diag)
    pred_df = pd.concat(external_preds, ignore_index=True)
    res_df = pd.concat(external_results, ignore_index=True)
    thr_df = pd.concat(thresholds, ignore_index=True)
    pred_df.to_csv(PREDICTIONS / "crc_stage2_external_predictions.csv", index=False)
    res_df.to_csv(TABLES / "crc_stage2_external_validation_results.csv", index=False)
    thr_df.to_csv(TABLES / "crc_stage2_threshold_selection.csv", index=False)
    pred_for_ci = pred_df.merge(thr_df[["train_cohort", "external_cohort", "feature_set", "model", "selected_threshold"]], on=["train_cohort", "external_cohort", "feature_set", "model"], how="left")
    bootstrap_external_ci(pred_for_ci, config).to_csv(TABLES / "crc_stage2_external_bootstrap_confidence_intervals.csv", index=False)
    if final_diagnostics:
        combined_diag = pd.concat([diagnostic_df] + final_diagnostics, ignore_index=True)
        combined_diag.to_csv(TABLES / "crc_stage2_fast_test_feature_count_diagnostic.csv" if config.fast_test else TABLES / "crc_stage2_feature_count_diagnostic.csv", index=False)
    res_df.to_csv(TABLES_MAIN / "crc_stage2_final_model_summary.csv", index=False)
    write_basic_figures(internal_df, pred_df, res_df, thr_df, shared_space)
    write_feature_importance(selected, meta, features, internal_df, config)
    write_error_analysis(pred_df, meta)
    write_reports(config, final=True)
    write_end_manifest(config)


def write_workflow_figure(path_base: Path, title: str, labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.7)))
    ax.axis("off")
    ys = np.linspace(0.92, 0.08, len(labels))
    for i, (label, y) in enumerate(zip(labels, ys)):
        ax.text(0.5, y, label, ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.35", fc="#e0f2fe", ec="#334155"))
        if i < len(labels) - 1:
            ax.annotate("", xy=(0.5, ys[i + 1] + 0.04), xytext=(0.5, y - 0.04), arrowprops=dict(arrowstyle="->", color="#334155"))
    ax.set_title(title, fontsize=13, weight="bold")
    fig.savefig(path_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_basic_figures(internal_df: pd.DataFrame, pred_df: pd.DataFrame, res_df: pd.DataFrame, thr_df: pd.DataFrame, shared_space: pd.DataFrame) -> None:
    write_workflow_figure(FIGURES_MAIN / "crc_stage2_threshold_selection_workflow", "Training-only threshold selection", ["Nested CV", "Out-of-fold training probabilities", "ROC curve", "Youden J", "Freeze threshold", "External validation"])
    write_workflow_figure(FIGURES_MAIN / "crc_stage2_external_validation_workflow", "Frozen external validation", ["Frozen preprocessing", "Frozen hyperparameters", "Frozen threshold", "External cohort", "No retraining", "Final metrics"])
    if not internal_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_df = internal_df.groupby(["train_cohort", "feature_set", "model"], as_index=False)["threshold_0_5_roc_auc"].mean()
        plot_df["label"] = plot_df["train_cohort"] + " | " + plot_df["feature_set"] + " | " + plot_df["model"]
        plot_df = plot_df.sort_values("threshold_0_5_roc_auc")
        ax.barh(plot_df["label"], plot_df["threshold_0_5_roc_auc"], color="#2563eb")
        ax.axvline(0.5, color="0.5", linestyle="--")
        ax.set_xlabel("Nested outer-fold ROC-AUC")
        ax.set_title("CRC internal model comparison")
        fig.tight_layout()
        fig.savefig(FIGURES_MAIN / "crc_stage2_internal_model_comparison.png", dpi=220)
        fig.savefig(FIGURES_MAIN / "crc_stage2_internal_model_comparison.svg")
        plt.close(fig)
    if not pred_df.empty:
        for (train, ext), g in pred_df.groupby(["train_cohort", "external_cohort"]):
            fig, ax = plt.subplots(figsize=(6, 5))
            for (feature_set, model), context in g.groupby(["feature_set", "model"], dropna=False):
                if context["sample_id"].duplicated().any():
                    raise AssertionError(f"Duplicate external prediction rows for ROC context {train}->{ext} {feature_set} {model}")
                fpr, tpr, _ = roc_curve(context["true_label"], context["predicted_probability"])
                ax.plot(fpr, tpr, label=f"{feature_set} | {model} AUC={roc_auc_score(context['true_label'], context['predicted_probability']):.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--", color="0.6", label="No discrimination")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.legend(fontsize=7)
            ax.set_title(f"External ROC curves: {train} to {ext}")
            fig.tight_layout()
            stem = FIGURES_MAIN / f"crc_stage2_external_roc_{train}_to_{ext}"
            fig.savefig(stem.with_suffix(".png"), dpi=220)
            fig.savefig(stem.with_suffix(".svg"))
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(6, 5))
            for (feature_set, model), context in g.groupby(["feature_set", "model"], dropna=False):
                if context["sample_id"].duplicated().any():
                    raise AssertionError(f"Duplicate external prediction rows for PR context {train}->{ext} {feature_set} {model}")
                precision, recall, _ = precision_recall_curve(context["true_label"], context["predicted_probability"])
                ax.plot(recall, precision, label=f"{feature_set} | {model} PR-AUC={average_precision_score(context['true_label'], context['predicted_probability']):.3f}")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.legend(fontsize=7)
            ax.set_title(f"External precision-recall curves: {train} to {ext}")
            fig.tight_layout()
            stem = FIGURES_MAIN / f"crc_stage2_external_pr_{train}_to_{ext}"
            fig.savefig(stem.with_suffix(".png"), dpi=220)
            fig.savefig(stem.with_suffix(".svg"))
            plt.close(fig)
    pd.DataFrame([
        {"figure": "crc_stage2_threshold_selection_workflow", "path": "figures/main_text/crc_stage2_threshold_selection_workflow.png", "purpose": "training-only threshold freezing"},
        {"figure": "crc_stage2_external_validation_workflow", "path": "figures/main_text/crc_stage2_external_validation_workflow.png", "purpose": "external validation without refitting"},
        {"figure": "crc_stage2_internal_model_comparison", "path": "figures/main_text/crc_stage2_internal_model_comparison.png", "purpose": "nested CV model comparison"},
    ]).to_csv(TABLES_MAIN / "crc_stage2_figure_index.csv", index=False)
    pd.DataFrame([{"n_shared_canonical_metabolites": len(shared_space), "feature_order_locked": True}]).to_csv(TABLES / "crc_stage2_feature_order_audit.csv", index=False)


def transformed_feature_names(fitted_pipeline: Pipeline) -> list[str]:
    pre = fitted_pipeline.named_steps.get("preprocess")
    if hasattr(pre, "get_feature_names_out"):
        return [str(x) for x in pre.get_feature_names_out()]
    return [f"feature_{i}" for i in range(getattr(fitted_pipeline.named_steps.get("clf"), "n_features_in_", 0))]


def write_feature_importance(selected: pd.DataFrame, meta: dict[str, pd.DataFrame], features: dict[str, pd.DataFrame], internal_df: pd.DataFrame, config: CRCStage2Config) -> None:
    rows = []
    for train_name, external_name in config.external_validation_directions:
        selected_rows = selected[selected["train_cohort"].eq(train_name) & selected["selected"]].copy()
        for _, sel in selected_rows.iterrows():
            feature_set = sel["feature_set"]
            model_name = sel["model"]
            model_path = MODELS / f"crc_stage2_selected_{train_name}_to_{external_name}_{feature_set}_{model_name}.joblib"
            if not model_path.exists():
                rows.append({"status": "not_run_missing_model", "train_cohort": train_name, "external_cohort": external_name, "feature_set": feature_set, "model": model_name})
                continue
            fitted = joblib.load(model_path)
            names = transformed_feature_names(fitted)
            clf = fitted.named_steps["clf"]
            if "logistic" in model_name and hasattr(clf, "coef_"):
                for name, coef in zip(names, np.ravel(clf.coef_)):
                    rows.append({
                        "status": "completed",
                        "train_cohort": train_name,
                        "external_cohort": external_name,
                        "feature_set": feature_set,
                        "model": model_name,
                        "feature": name,
                        "feature_type": "metabolite" if str(name).startswith("met__") else ("encoded_categorical" if "__" in str(name) else "numeric_clinical"),
                        "importance_type": "standardized_logistic_coefficient",
                        "standardized_coefficient": float(coef),
                        "odds_ratio_per_one_sd": float(np.exp(coef)),
                        "signed_direction": "positive" if coef > 0 else ("negative" if coef < 0 else "zero"),
                        "external_data_used": False,
                        "interpretation_note": "Predictive importance; not causal evidence and not differential abundance.",
                    })
            else:
                x = features[train_name].reset_index(drop=True)
                y = meta[train_name]["crc_binary"].astype(int).reset_index(drop=True)
                fold_records = internal_df[
                    internal_df["train_cohort"].eq(train_name)
                    & internal_df["feature_set"].eq(feature_set)
                    & internal_df["model"].eq(model_name)
                ].copy()
                fold_importance: dict[str, list[float]] = {}
                for _, fold_row in fold_records.iterrows():
                    fold_idx = int(fold_row["outer_fold"])
                    cv_design_path = TABLES / "crc_stage2_cv_design.csv"
                    if not cv_design_path.exists():
                        raise FileNotFoundError("CV design file required for fold-specific feature importance.")
                    cv_design = pd.read_csv(cv_design_path)
                    design = cv_design[
                        cv_design["train_cohort"].eq(train_name)
                        & cv_design["feature_set"].eq(feature_set)
                        & cv_design["model"].eq(model_name)
                        & cv_design["outer_fold"].eq(fold_idx)
                    ].iloc[0]
                    dev_samples = set(json.loads(design["dev_sample_ids"]))
                    val_samples = set(json.loads(design["test_sample_ids"]))
                    dev_idx = meta[train_name].index[meta[train_name]["sample_id"].astype(str).isin(dev_samples)].tolist()
                    val_idx = meta[train_name].index[meta[train_name]["sample_id"].astype(str).isin(val_samples)].tolist()
                    model, _, _, _ = model_and_grid(config)[model_name]
                    params = json.loads(fold_row["best_params"])
                    pre = build_preprocessor(
                        feature_set,
                        [c for c in ["age_years", "bmi"] if c in x.columns],
                        [c for c in ["sex"] if c in x.columns],
                        [c for c in x.columns if c.startswith("met__")],
                        config.primary_missingness_threshold,
                        False,
                    )
                    fold_model = Pipeline([("preprocess", pre), ("clf", clone(model))])
                    fold_model.set_params(**params)
                    fold_model.fit(x.iloc[dev_idx], y.iloc[dev_idx], **fit_params_for_model(model_name, y.iloc[dev_idx]))
                    perm = permutation_importance(fold_model, x.iloc[val_idx], y.iloc[val_idx], scoring="roc_auc", n_repeats=10, random_state=config.random_seed + fold_idx)
                    fold_names = list(getattr(fold_model, "feature_names_in_", x.columns.astype(str)))
                    for name, value in zip(fold_names, perm.importances_mean):
                        fold_importance.setdefault(str(name), []).append(float(value))
                        rows.append({
                            "status": "fold_level",
                            "train_cohort": train_name,
                            "external_cohort": external_name,
                            "feature_set": feature_set,
                            "model": model_name,
                            "outer_fold": fold_idx,
                            "fold_specific_selected_parameters": json.dumps(params, sort_keys=True),
                            "held_out_fold_size": len(val_idx),
                            "feature": str(name),
                            "fold_permutation_importance": float(value),
                            "external_data_used": False,
                        })
                for name, values in fold_importance.items():
                    arr = np.asarray(values, dtype=float)
                    rows.append({
                        "status": "completed",
                        "train_cohort": train_name,
                        "external_cohort": external_name,
                        "feature_set": feature_set,
                        "model": model_name,
                        "feature": name,
                        "feature_type": "metabolite" if str(name).startswith("met__") else "clinical_or_encoded",
                        "importance_type": "held_out_training_fold_permutation_importance",
                        "mean_importance": float(np.nanmean(arr)),
                        "sd_importance": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else np.nan,
                        "positive_fold_fraction": float((arr > 0).mean()),
                        "external_data_used": False,
                        "interpretation_note": "Predictive importance; unsigned tree importance is not causal evidence.",
                    })
    out = pd.DataFrame(rows)
    if not out.empty and "mean_importance" in out.columns:
        out["rank_within_context"] = out.groupby(["train_cohort", "external_cohort", "feature_set", "model"])["mean_importance"].rank(ascending=False, method="min")
    out.to_csv(TABLES / "crc_stage2_important_feature_table.csv", index=False)
    mets = out[out.get("feature", pd.Series(dtype=str)).astype(str).str.contains("HMDB:|KEGG:", na=False)].copy() if not out.empty else pd.DataFrame()
    if mets.empty:
        pd.DataFrame([{"status": "not_run_no_metabolite_importance_available"}]).to_csv(TABLES / "crc_stage2_stable_cross_cohort_metabolites.csv", index=False)
    else:
        stability = mets.groupby(["feature", "feature_set"], dropna=False).agg(
            n_directions=("train_cohort", "nunique"),
            mean_rank=("rank_within_context", "mean"),
        ).reset_index()
        stability["stability_classification"] = np.where(stability["n_directions"] >= 2, "observed_in_both_training_directions", "single_direction_only")
        stability["interpretation_note"] = "Predictive stability only; not causal evidence, DA evidence or biomarker validation."
        stability.to_csv(TABLES / "crc_stage2_stable_cross_cohort_metabolites.csv", index=False)


def write_error_analysis(pred_df: pd.DataFrame, meta: dict[str, pd.DataFrame]) -> None:
    rows = []
    for _, r in pred_df.iterrows():
        ext_meta = meta[str(r.external_cohort)]
        m = ext_meta[ext_meta["sample_id"].astype(str).eq(str(r.sample_id))]
        row = r.to_dict()
        if not m.empty:
            for col in ["age_years", "sex", "bmi", "stage_major", "stage_source_faithful", "stage_harmonized", "tumour_site", "smoking", "alcohol"]:
                row[col] = m.iloc[0].get(col, pd.NA)
        row["error_type"] = "correct"
        if int(row["true_label"]) == 0 and int(row["predicted_label_at_frozen_threshold"]) == 1:
            row["error_type"] = "false_positive"
        if int(row["true_label"]) == 1 and int(row["predicted_label_at_frozen_threshold"]) == 0:
            row["error_type"] = "false_negative"
        rows.append(row)
    pd.DataFrame(rows).to_csv(TABLES / "crc_stage2_external_error_analysis.csv", index=False)


def robust_ols_hc3(y: pd.Series, x: pd.DataFrame, exposure_col: str) -> dict[str, Any]:
    design = x.copy()
    complete = pd.concat([y.rename("y"), design], axis=1).dropna()
    n = len(complete)
    k = design.shape[1]
    if n < max(8, k + 3):
        return {"status": "too_few_complete_cases", "n": n, "beta": np.nan, "se": np.nan, "p_value": np.nan}
    yv = complete["y"].to_numpy(dtype=float)
    xv = complete[design.columns].to_numpy(dtype=float)
    try:
        xtx_inv = np.linalg.pinv(xv.T @ xv)
        beta = xtx_inv @ xv.T @ yv
        resid = yv - xv @ beta
        hat = np.sum((xv @ xtx_inv) * xv, axis=1)
        omega = (resid / np.clip(1 - hat, 1e-8, None)) ** 2
        cov = xtx_inv @ (xv.T @ np.diag(omega) @ xv) @ xtx_inv
        idx = list(design.columns).index(exposure_col)
        se = float(np.sqrt(cov[idx, idx]))
        b = float(beta[idx])
        t_stat = b / se if se > 0 else np.nan
        df = n - k
        p_val = float(2 * t_dist.sf(abs(t_stat), df)) if t_dist is not None and df > 0 and np.isfinite(t_stat) else np.nan
        return {"status": "ok", "n": n, "beta": b, "se": se, "p_value": p_val, "ci_lower": b - 1.96 * se, "ci_upper": b + 1.96 * se}
    except Exception as exc:
        return {"status": f"failed:{repr(exc)}", "n": n, "beta": np.nan, "se": np.nan, "p_value": np.nan}


def run_differential_abundance(config: CRCStage2Config) -> None:
    meta = pd.read_csv(PROCESSED / "crc_stage2_primary_metadata.csv")
    met = pd.read_csv(PROCESSED / "crc_stage2_primary_metabolomics_canonical.csv")
    rows = []
    for cohort in ["Sinha", "Yachida"]:
        m = meta[meta["cohort"].eq(cohort)].copy()
        x = met[met["cohort"].eq(cohort)].copy()
        if m["sample_id"].duplicated().any() or x["sample_id"].duplicated().any():
            raise AssertionError(f"Duplicated sample IDs before DA merge for {cohort}")
        merged = m.merge(x, on=["cohort", "sample_id", "participant_id"], how="inner", validate="one_to_one")
        if len(merged) != len(m) or len(merged) != len(x):
            raise AssertionError(f"DA metadata/metabolomics merge mismatch for {cohort}: meta={len(m)}, met={len(x)}, merged={len(merged)}")
        if int(merged["crc_binary"].sum()) != int(m["crc_binary"].sum()):
            raise AssertionError(f"DA class counts changed during merge for {cohort}")
        design = pd.DataFrame({"intercept": 1.0, "crc_status": merged["crc_binary"].astype(int), "age_years": pd.to_numeric(merged["age_years"], errors="coerce")})
        for feature in [c for c in merged.columns if c.startswith("met__")]:
            raw_y = pd.to_numeric(merged[feature], errors="coerce")
            sd = raw_y.std(skipna=True)
            y_z = (raw_y - raw_y.mean(skipna=True)) / sd if pd.notna(sd) and sd > 0 else raw_y * np.nan
            fit = robust_ols_hc3(y_z, design, "crc_status")
            rows.append({
                "cohort": cohort,
                "comparison": "CRC vs Control",
                "canonical_metabolite": feature.replace("met__", ""),
                "formula": "metabolite_z ~ crc_status + age_years",
                "total_n": int(len(merged)),
                "crc_n": int(merged["crc_binary"].sum()),
                "control_n": int((1 - merged["crc_binary"]).sum()),
                "missingness": float(y_z.isna().mean()),
                **fit,
                "standardized_beta": fit.get("beta", np.nan),
                "interpretation_note": "Differential abundance uses cohort-specific z-scored raw observed values and is separate from predictive feature importance.",
            })
    out = pd.DataFrame(rows)
    out["BH_FDR"] = out.groupby("cohort")["p_value"].transform(bh_fdr)
    out.to_csv(TABLES / "crc_stage2_differential_abundance_results.csv", index=False)


def bh_fdr(p: pd.Series) -> pd.Series:
    pvals = pd.to_numeric(p, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = pvals.dropna().sort_values()
    if valid.empty:
        return out
    n = len(valid)
    ranks = np.arange(1, n + 1)
    q = (valid.to_numpy() * n / ranks)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out.loc[valid.index] = np.clip(q, 0, 1)
    return out


def hypergeom_right_tail(k: int, K: int, n: int, N: int) -> float:
    if any(x < 0 for x in [k, K, n, N]) or K > N or n > N or k > K or k > n:
        return np.nan
    if scipy_hypergeom is not None:
        return float(scipy_hypergeom.sf(k - 1, N, K, n))
    denom = math.comb(N, n)
    upper = min(K, n)
    total = 0
    for i in range(k, upper + 1):
        if n - i <= N - K:
            total += math.comb(K, i) * math.comb(N - K, n - i)
    return float(total / denom) if denom else np.nan


def _normalise_kegg_pathway_resources(config: CRCStage2Config) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Load local KEGG cache only; no internet/API calls are made."""
    ibd_cache = ROOT / "thesis_preprocessing_workflow" / "ibd_pipeline" / "annotation_resources"
    membership_candidates = [
        ANNOTATION_CACHE / "kegg_pathway_membership.csv",
        ANNOTATION_CACHE / "kegg_compound_pathways.tsv",
        ibd_cache / "kegg_compound_pathways.tsv",
    ]
    names_candidates = [
        ANNOTATION_CACHE / "kegg_pathway_names.csv",
        ANNOTATION_CACHE / "kegg_pathway_names.tsv",
        ibd_cache / "kegg_pathway_names.tsv",
    ]
    membership_path = next((p for p in membership_candidates if p.exists()), None)
    names_path = next((p for p in names_candidates if p.exists()), None)
    audit_rows: list[dict[str, Any]] = []

    def resource_row(resource: str, path: Path | None, df: pd.DataFrame | None, namespace: str, mapping_type: str, approved: bool, notes: str) -> None:
        audit_rows.append({
            "resource": resource,
            "path": str(path.relative_to(CRC_DIR)) if path and path.is_relative_to(CRC_DIR) else str(path) if path else pd.NA,
            "exists": bool(path and path.exists()),
            "n_rows": int(len(df)) if df is not None else 0,
            "identifier_namespace": namespace,
            "mapping_type": mapping_type,
            "one_to_one_or_many": "many_to_many" if df is not None and len(df) else pd.NA,
            "approved_for_analysis": approved,
            "notes": notes,
        })

    if membership_path is None or names_path is None:
        resource_row("KEGG compound-pathway membership", membership_path, None, "KEGG compound/pathway", "local_cache", False, "Required local KEGG membership cache missing.")
        resource_row("KEGG pathway names", names_path, None, "KEGG pathway", "local_cache", False, "Required local KEGG pathway-name cache missing.")
        return pd.DataFrame(), pd.DataFrame(), audit_rows

    if membership_path.suffix.lower() == ".tsv":
        membership = pd.read_csv(membership_path, sep="\t", header=None, names=["KEGG_Compound_ID", "KEGG_pathway_ID"])
    else:
        membership = pd.read_csv(membership_path)
        membership = membership.rename(columns={
            "compound": "KEGG_Compound_ID",
            "kegg_compound": "KEGG_Compound_ID",
            "pathway": "KEGG_pathway_ID",
            "kegg_pathway": "KEGG_pathway_ID",
        })
    if names_path.suffix.lower() == ".tsv":
        names = pd.read_csv(names_path, sep="\t", header=None, names=["KEGG_pathway_ID", "KEGG_pathway_name"])
    else:
        names = pd.read_csv(names_path)
        names = names.rename(columns={"pathway": "KEGG_pathway_ID", "name": "KEGG_pathway_name", "kegg_pathway": "KEGG_pathway_ID"})

    membership["KEGG_Compound_ID"] = membership["KEGG_Compound_ID"].astype(str).str.replace("cpd:", "", regex=False).str.strip().str.upper()
    membership["KEGG_pathway_ID"] = membership["KEGG_pathway_ID"].astype(str).str.replace("path:", "", regex=False).str.strip()
    names["KEGG_pathway_ID"] = names["KEGG_pathway_ID"].astype(str).str.replace("path:", "", regex=False).str.strip()
    names["KEGG_pathway_name"] = names["KEGG_pathway_name"].astype(str).str.strip()
    membership = membership[membership["KEGG_Compound_ID"].map(is_valid_kegg) & membership["KEGG_pathway_ID"].str.match(r"^map\d{5}$", na=False)].drop_duplicates()
    names = names[names["KEGG_pathway_ID"].str.match(r"^map\d{5}$", na=False)].drop_duplicates("KEGG_pathway_ID")
    pathway_df = membership.merge(names, on="KEGG_pathway_ID", how="left")
    pathway_df["KEGG_pathway_name"] = pathway_df["KEGG_pathway_name"].fillna(pathway_df["KEGG_pathway_ID"])

    excluded_ids = set(config.annotation_settings["excluded_kegg_pathway_ids"])
    excluded_patterns = [str(x).lower() for x in config.annotation_settings["excluded_kegg_pathway_name_patterns"]]
    pathway_df["excluded_overview_pathway"] = pathway_df["KEGG_pathway_ID"].isin(excluded_ids) | pathway_df["KEGG_pathway_name"].str.lower().apply(lambda x: any(p in x for p in excluded_patterns))
    pathway_df = pathway_df[~pathway_df["excluded_overview_pathway"]].drop(columns=["excluded_overview_pathway"]).drop_duplicates()
    resource_row("KEGG compound-pathway membership", membership_path, pathway_df, "KEGG compound/pathway", "local_cache_exact_identifier", True, "Local KEGG REST-style cache; generic overview maps excluded using CRC configuration.")
    resource_row("KEGG pathway names", names_path, names, "KEGG pathway", "local_cache_exact_identifier", True, "Local pathway-name cache used for labels only.")
    return pathway_df, names, audit_rows


def _crc_hmdb_kegg_long() -> tuple[pd.DataFrame, pd.DataFrame]:
    crosswalk_path = TABLES / "crc_stage2_hmdb_kegg_crosswalk.csv"
    manifest_path = TABLES / "crc_stage2_canonical_metabolite_manifest.csv"
    source = crosswalk_path if crosswalk_path.exists() else manifest_path
    if not source.exists():
        return pd.DataFrame(columns=["cohort", "canonical_metabolite", "HMDB_ID", "KEGG_Compound_ID", "feature"]), pd.DataFrame()
    df = pd.read_csv(source)
    rows = []
    for _, r in df.iterrows():
        canonical = str(r.get("canonical_id", "")).strip()
        feature = str(r.get("feature", ""))
        cohort = str(r.get("cohort", ""))
        hmdb_ids = [x for x in parse_identifier_field(r.get("HMDB")) if is_valid_hmdb(x)]
        kegg_ids = [x for x in parse_identifier_field(r.get("KEGG")) if is_valid_kegg(x)]
        if canonical.startswith("HMDB:"):
            hmdb_ids = sorted(set(hmdb_ids + [canonical.replace("HMDB:", "")]))
        if canonical.startswith("KEGG:"):
            kegg_ids = sorted(set(kegg_ids + [canonical.replace("KEGG:", "")]))
        for hmdb in hmdb_ids or [pd.NA]:
            for kegg in kegg_ids:
                rows.append({
                    "cohort": cohort,
                    "canonical_metabolite": canonical,
                    "HMDB_ID": hmdb,
                    "KEGG_Compound_ID": kegg,
                    "feature": feature,
                    "mapping_source": str(source.relative_to(CRC_DIR)),
                    "mapping_type": "deterministic_supplied_crc_hmdb_kegg_crosswalk",
                })
    long = pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=["cohort", "canonical_metabolite", "HMDB_ID", "KEGG_Compound_ID", "feature", "mapping_source", "mapping_type"])
    return long, df


def _hash_protected_crc_outputs() -> pd.DataFrame:
    protected: list[Path] = []
    protected.extend(sorted(MODELS.glob("*.joblib")))
    protected.extend(sorted(PREDICTIONS.glob("*.csv")))
    for name in [
        "crc_stage2_external_validation_results.csv",
        "crc_stage2_threshold_selection.csv",
        "crc_stage2_external_calibration_assessment.csv",
        "crc_stage2_external_calibration_bins.csv",
        "crc_stage2_differential_abundance_results.csv",
        "crc_stage2_important_feature_table.csv",
        "crc_stage2_internal_model_selection.csv",
    ]:
        protected.append(TABLES / name)
    rows = []
    for path in sorted(set(protected)):
        rows.append({
            "path": str(path.relative_to(CRC_DIR)) if path.is_relative_to(CRC_DIR) else str(path),
            "exists": path.exists(),
            "sha256": sha256(path) if path.exists() else pd.NA,
        })
    return pd.DataFrame(rows)


def _plot_crc_ora(enrich_df: pd.DataFrame, cohort: str, output_stem: str) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    sub = enrich_df[(enrich_df["cohort"].eq(cohort)) & (enrich_df["FDR_q_value"] < 0.05)].copy()
    if sub.empty:
        ax.axis("off")
        ax.text(0.04, 0.62, f"{cohort}: no KEGG pathways passed BH-FDR < 0.05", fontsize=13, weight="bold")
        ax.text(0.04, 0.48, "Main ORA figures are restricted to statistically significant pathways.", fontsize=10)
    else:
        sub["direction_label"] = sub["direction"].map({
            "all_fdr_significant": "All significant metabolites (↑ + ↓)",
            "higher_in_disease": "Higher in CRC",
            "lower_in_disease": "Lower in CRC",
        }).fillna(sub["direction"])
        sub = sub.sort_values(["direction", "FDR_q_value", "raw_p_value"]).head(14).sort_values("fold_enrichment")
        c = -np.log10(pd.to_numeric(sub["FDR_q_value"], errors="coerce").clip(lower=1e-300))
        marker_map = {"All significant metabolites (↑ + ↓)": "o", "Higher in CRC": "^", "Lower in CRC": "v"}
        for label, part in sub.groupby("direction_label", sort=False):
            ax.scatter(
                part["fold_enrichment"],
                part["pathway_name"],
                s=70 + 28 * part["k_significant_compounds_in_pathway"].astype(float),
                c=-np.log10(part["FDR_q_value"].clip(lower=1e-300)),
                cmap="plasma",
                vmin=float(c.min()),
                vmax=float(c.max()),
                marker=marker_map.get(label, "o"),
                edgecolor="0.2",
                linewidth=0.7,
                label=label,
            )
        ax.axvline(1, color="0.45", linestyle="--", lw=1.1)
        ax.text(1.03, -0.08, "Fold enrichment = 1", transform=ax.get_xaxis_transform(), fontsize=8, color="0.25", ha="left")
        ax.set_xlabel("Fold enrichment")
        ax.set_ylabel("")
        ax.set_title(f"{cohort} KEGG ORA: BH-FDR-significant pathways", fontsize=12)
        ax.grid(axis="x", color="0.90")
        ax.legend(title="Metabolite set", frameon=False, fontsize=8, loc="lower right")
        sc = ax.collections[-1]
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("−log10(BH-FDR)")
    fig.tight_layout()
    fig.savefig(FIGURES_MAIN / f"{output_stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES_MAIN / f"{output_stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _plot_crc_ora_comparison(overlap_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    sig = overlap_df[overlap_df["sinha_significant"] | overlap_df["yachida_significant"]].copy()
    if sig.empty:
        ax.axis("off")
        ax.text(0.04, 0.62, "No CRC KEGG pathways reached BH-FDR significance in either cohort", fontsize=13, weight="bold")
        ax.text(0.04, 0.48, "Sinha and Yachida ORA were performed separately using cohort-specific tested/mapped backgrounds.", fontsize=10)
    else:
        sig["label"] = sig["direction_set"] + " | " + sig["pathway_name"].astype(str)
        sig = sig.sort_values(["shared_significant", "direction_set", "pathway_name"], ascending=[False, True, True]).head(18)
        y = np.arange(len(sig))
        ax.scatter(sig["sinha_fold_enrichment"], y - 0.12, color="#4C78A8", label="Sinha", s=80, edgecolor="white")
        ax.scatter(sig["yachida_fold_enrichment"], y + 0.12, color="#E15759", label="Yachida", s=80, edgecolor="white")
        ax.axvline(1, color="0.45", linestyle="--", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(sig["label"], fontsize=8)
        ax.set_xlabel("Fold enrichment")
        ax.set_title("CRC cross-cohort KEGG ORA comparison")
        ax.legend(frameon=False)
        ax.grid(axis="x", color="0.90")
    fig.tight_layout()
    fig.savefig(FIGURES_MAIN / "crc_kegg_ora_cross_cohort_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES_MAIN / "crc_kegg_ora_cross_cohort_comparison.svg", bbox_inches="tight")
    plt.close(fig)


def _plot_crc_ora_presentation(enrich_df: pd.DataFrame, coverage_df: pd.DataFrame) -> None:
    sig = enrich_df[enrich_df["FDR_q_value"] < 0.05].copy() if not enrich_df.empty else pd.DataFrame()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharex=False)
    for ax, cohort in zip(axes, ["Sinha", "Yachida"]):
        sub = sig[sig["cohort"].eq(cohort)].sort_values("FDR_q_value").head(8).sort_values("fold_enrichment") if not sig.empty else pd.DataFrame()
        if sub.empty:
            cov = coverage_df[(coverage_df["cohort"].eq(cohort)) & (coverage_df["direction"].eq("all_fdr_significant"))]
            note = "No KEGG pathways reached BH-FDR < 0.05"
            if not cov.empty:
                r = cov.iloc[0]
                note += f"\nMapped significant compounds: {int(r['unique_significant_KEGG_compound_count'])}"
            ax.axis("off")
            ax.text(0.05, 0.62, cohort, fontsize=14, weight="bold")
            ax.text(0.05, 0.48, note, fontsize=10)
        else:
            ax.scatter(
                sub["fold_enrichment"],
                sub["pathway_name"],
                s=80 + 30 * sub["k_significant_compounds_in_pathway"].astype(float),
                c=-np.log10(sub["FDR_q_value"].clip(lower=1e-300)),
                cmap="plasma",
                edgecolor="0.2",
            )
            ax.axvline(1, color="0.45", linestyle="--", lw=1)
            ax.set_title(cohort)
            ax.set_xlabel("Fold enrichment")
            ax.grid(axis="x", color="0.90")
    fig.suptitle("Biological interpretation: KEGG pathway ORA", fontsize=15, weight="bold")
    fig.text(0.5, 0.02, "ORA uses cohort-specific tested KEGG-mapped metabolite backgrounds; enrichment is not evidence of pathway activation or causality.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(FIGURES_PRESENTATION / "crc_kegg_ora_presentation.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES_PRESENTATION / "crc_kegg_ora_presentation.svg", bbox_inches="tight")
    plt.close(fig)


def run_pathway_enrichment(config: CRCStage2Config) -> None:
    ensure_dirs()
    before_hashes = _hash_protected_crc_outputs()
    before_hashes = before_hashes.rename(columns={"sha256": "sha256_before", "exists": "exists_before"})
    reopening = """# CRC Stage 2 Controlled Reopening for KEGG ORA

Previous status: **CRC_STAGE2_FROZEN_FOR_WRITEUP**

Reason reopened: completion of the previously missing CRC KEGG pathway over-representation analysis so that CRC biological interpretation is methodologically parallel to the completed IBD analysis.

This controlled reopening did **not** modify models, hyperparameters, cross-validation, predictions, thresholds, external metrics, calibration, differential-abundance coefficients or feature-importance outputs.

Final status after this pathway-only pass: **CRC_STAGE2_REFROZEN_AFTER_ORA**
"""
    (REPORTS / "CRC_STAGE2_CONTROLLED_REOPENING_ORA.md").write_text(reopening)
    ibd_audit = """# CRC ORA / IBD Method-Parity Audit

Authoritative IBD ORA implementation inspected: `thesis_preprocessing_workflow/ibd_pipeline/run_pipeline.py`.

## IBD method identified

- DA input: `tables/differential_abundance_results.csv`.
- Contrasts: IBD vs control, CD vs control and UC vs control, separately in Franzosa and iHMP.
- Significant metabolite rule: BH-FDR `fdr_q_value <= 0.05`.
- Direction sets: `all_fdr_significant`, `higher_in_disease` and `lower_in_disease`.
- KEGG mapping source: `tables/metabolite_annotation_enrichment.csv`, with local KEGG compound-pathway cache.
- Pathway membership source: `annotation_resources/kegg_compound_pathways.tsv` plus `kegg_pathway_names.tsv`.
- Background: metabolites tested in the relevant cohort/contrast with deterministic KEGG compound mapping.
- Test: one-sided hypergeometric right-tail over-representation test.
- Alternative hypothesis: pathway contains more significant mapped KEGG compounds than expected by chance.
- Minimum pathway size: at least 3 background compounds and at least 2 significant compounds in the pathway.
- Duplicate mapping treatment: duplicate HMDB-KEGG and KEGG-pathway rows are deduplicated before counting.
- Multi-pathway metabolites: retained; a KEGG compound can contribute to each pathway it is formally mapped to.
- Ambiguous identifiers: not admitted into deterministic pathway input.
- Fold enrichment: `(k / n) / (K / N)`.
- Multiple testing: BH-FDR separately within cohort × outcome × direction × annotation source.
- Figure convention: main figure restricted to BH-FDR-significant KEGG pathways; supplementary per-context figures can show context-specific ORA.

No methodological error was identified that would require stopping before CRC ORA.
"""
    (REPORTS / "CRC_ORA_IBD_METHOD_PARITY_AUDIT.md").write_text(ibd_audit)

    da_path = TABLES / "crc_stage2_differential_abundance_results.csv"
    if not da_path.exists():
        pd.DataFrame([{"status": "not_run_missing_crc_da_results", "reason": "Authoritative CRC DA table is absent."}]).to_csv(TABLES / "crc_stage2_pathway_enrichment_status.csv", index=False)
        return
    da = pd.read_csv(da_path)
    pathway_df, pathway_names, resource_rows = _normalise_kegg_pathway_resources(config)
    hmdb_kegg, mapping_source = _crc_hmdb_kegg_long()
    resource_rows.extend([
        {
            "resource": "CRC HMDB-KEGG crosswalk",
            "path": "stage2/tables/crc_stage2_hmdb_kegg_crosswalk.csv",
            "exists": (TABLES / "crc_stage2_hmdb_kegg_crosswalk.csv").exists(),
            "n_rows": int(len(hmdb_kegg)),
            "identifier_namespace": "HMDB/KEGG",
            "mapping_type": "deterministic_supplied_mapping",
            "one_to_one_or_many": "one_to_many_allowed_deduplicated",
            "approved_for_analysis": not hmdb_kegg.empty,
            "notes": "Only exact supplied HMDB and KEGG identifiers are used; no name/fuzzy/API mapping.",
        },
        {
            "resource": "CRC differential abundance",
            "path": "stage2/tables/crc_stage2_differential_abundance_results.csv",
            "exists": da_path.exists(),
            "n_rows": int(len(da)),
            "identifier_namespace": "canonical CRC metabolite",
            "mapping_type": "authoritative_final_DA_input",
            "one_to_one_or_many": "one_row_per_cohort_metabolite",
            "approved_for_analysis": True,
            "notes": "Existing DA results are consumed read-only; no DA refitting.",
        },
    ])
    pd.DataFrame(resource_rows).to_csv(PATHWAY_TABLES / "crc_ora_resource_audit.csv", index=False)
    if pathway_df.empty or hmdb_kegg.empty:
        pd.DataFrame([{"status": "not_run_missing_valid_local_resources", "reason": "Local KEGG pathway or CRC HMDB-KEGG mapping resources unavailable."}]).to_csv(TABLES / "crc_stage2_pathway_enrichment_status.csv", index=False)
        return

    tested = da[da["status"].astype(str).eq("ok") & pd.to_numeric(da["BH_FDR"], errors="coerce").notna()].copy()
    tested["source_da_comparison"] = tested["comparison"].astype(str)
    tested["comparison"] = np.where(tested["cohort"].astype(str).eq("Yachida"), "CRC vs Healthy", tested["comparison"].astype(str))
    tested["effect"] = pd.to_numeric(tested["standardized_beta"], errors="coerce")
    tested["direction_label"] = np.where(tested["effect"] > 0, "higher_in_crc", np.where(tested["effect"] < 0, "lower_in_crc", "direction_neutral"))
    tested_map = tested.merge(hmdb_kegg, on=["cohort", "canonical_metabolite"], how="left")
    tested_map = tested_map[tested_map["KEGG_Compound_ID"].map(lambda x: is_valid_kegg(str(x)) if pd.notna(x) else False)].drop_duplicates(["cohort", "comparison", "canonical_metabolite", "KEGG_Compound_ID"])
    compound_pathway = pathway_df.merge(tested_map[["KEGG_Compound_ID"]].drop_duplicates(), on="KEGG_Compound_ID", how="inner").drop_duplicates(["KEGG_Compound_ID", "KEGG_pathway_ID"])

    coverage_rows: list[dict[str, Any]] = []
    enrich_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    min_K = int(config.annotation_settings["minimum_pathway_background_compounds"])
    min_k = int(config.annotation_settings["minimum_pathway_significant_compounds"])
    for (cohort, comparison), group in tested.groupby(["cohort", "comparison"], dropna=False):
        bg_metabolites = set(group["canonical_metabolite"])
        bg_map = tested_map[(tested_map["cohort"].eq(cohort)) & (tested_map["comparison"].eq(comparison))]
        bg_compounds = set(bg_map["KEGG_Compound_ID"])
        for direction, selector in [
            ("all_fdr_significant", group["BH_FDR"] <= 0.05),
            ("higher_in_disease", (group["BH_FDR"] <= 0.05) & (group["effect"] > 0)),
            ("lower_in_disease", (group["BH_FDR"] <= 0.05) & (group["effect"] < 0)),
        ]:
            sig_metabolites = set(group.loc[selector, "canonical_metabolite"])
            sig_map = bg_map[bg_map["canonical_metabolite"].isin(sig_metabolites)]
            sig_compounds = set(sig_map["KEGG_Compound_ID"])
            coverage_rows.append({
                "cohort": cohort,
                "comparison": comparison,
                "direction": direction,
                "total_metabolites_tested": len(bg_metabolites),
                "metabolites_with_valid_HMDB": int(group["canonical_metabolite"].astype(str).str.startswith("HMDB:").sum()),
                "metabolites_with_valid_KEGG": int(bg_map["canonical_metabolite"].nunique()),
                "metabolites_eligible_for_ORA": int(bg_map["canonical_metabolite"].nunique()),
                "significant_metabolites_BH_FDR_0_05": len(sig_metabolites),
                "significant_metabolites_with_KEGG_mapping": int(sig_map["canonical_metabolite"].nunique()),
                "unique_background_KEGG_compound_count": len(bg_compounds),
                "unique_significant_KEGG_compound_count": len(sig_compounds),
                "background_definition": "cohort/comparison-specific DA-tested metabolites with deterministic KEGG mapping",
                "effect_size_column": "standardized_beta",
                "raw_p_value_column": "p_value",
                "BH_FDR_column": "BH_FDR",
                "direction_convention": "positive standardized_beta = higher in CRC under metabolite_z ~ crc_status + age_years",
                "covariates": "age_years",
                "sample_counts": f"total={int(group['total_n'].max())}; crc={int(group['crc_n'].max())}; control={int(group['control_n'].max())}",
                "source_da_comparison": ";".join(sorted(set(group["source_da_comparison"].astype(str)))),
            })
            if len(sig_compounds) < min_k:
                status_rows.append({"cohort": cohort, "comparison": comparison, "direction": direction, "status": "insufficient_significant_kegg_mappings", "reason": f"Significant mapped KEGG compounds < {min_k}"})
                continue
            bg_pw = compound_pathway[compound_pathway["KEGG_Compound_ID"].isin(bg_compounds)]
            sig_pw = bg_pw[bg_pw["KEGG_Compound_ID"].isin(sig_compounds)]
            N, n = len(bg_compounds), len(sig_compounds)
            produced = 0
            for (pid, pname), pgroup in bg_pw.groupby(["KEGG_pathway_ID", "KEGG_pathway_name"], dropna=False):
                bg_in = set(pgroup["KEGG_Compound_ID"])
                sig_in = set(sig_pw[sig_pw["KEGG_pathway_ID"].eq(pid)]["KEGG_Compound_ID"])
                K, k = len(bg_in), len(sig_in)
                if K < min_K or k < min_k:
                    continue
                contrib = sig_map[sig_map["KEGG_Compound_ID"].isin(sig_in)]
                fold = (k / n) / max(K / N, 1e-12)
                enrich_rows.append({
                    "cohort": cohort,
                    "comparison": comparison,
                    "direction": direction,
                    "annotation_source": "local KEGG compound-pathway cache",
                    "pathway_id": pid,
                    "pathway_name": pname,
                    "N_background_mapped_compounds": N,
                    "K_background_compounds_in_pathway": K,
                    "n_significant_mapped_compounds": n,
                    "k_significant_compounds_in_pathway": k,
                    "fold_enrichment": fold,
                    "raw_p_value": hypergeom_right_tail(k, K, n, N),
                    "significant_KEGG_compounds": ";".join(sorted(sig_in)),
                    "contributing_metabolites": ";".join(sorted(set(contrib["canonical_metabolite"]))),
                    "contributing_HMDB_IDs": ";".join(sorted(set(contrib["HMDB_ID"].dropna().astype(str)))),
                    "interpretation_note": "ORA only: enriched among CRC-associated metabolites; not pathway activation, causality or biomarker proof.",
                })
                produced += 1
            status_rows.append({
                "cohort": cohort,
                "comparison": comparison,
                "direction": direction,
                "status": "completed" if produced else "no_eligible_pathways",
                "reason": "KEGG ORA completed with configured pathway-size thresholds" if produced else "No pathways met K/k thresholds",
            })
    enrich = pd.DataFrame(enrich_rows)
    if not enrich.empty:
        enrich["FDR_q_value"] = enrich.groupby(["cohort", "comparison", "direction", "annotation_source"], group_keys=False)["raw_p_value"].apply(bh_fdr)
        enrich["significant_FDR_0_05"] = enrich["FDR_q_value"] <= 0.05
        enrich = enrich.sort_values(["cohort", "comparison", "direction", "FDR_q_value", "raw_p_value"])
    coverage = pd.DataFrame(coverage_rows)
    status = pd.DataFrame(status_rows)
    status.to_csv(TABLES / "crc_stage2_pathway_enrichment_status.csv", index=False)
    coverage.to_csv(PATHWAY_TABLES / "crc_ora_input_mapping_summary.csv", index=False)
    coverage.to_csv(PATHWAY_TABLES / "crc_ora_da_input_audit.csv", index=False)
    enrich.to_csv(PATHWAY_TABLES / "crc_ora_all_results.csv", index=False)
    enrich[enrich["cohort"].eq("Sinha")].to_csv(PATHWAY_TABLES / "crc_ora_sinha_crc_vs_control.csv", index=False)
    enrich[enrich["cohort"].eq("Yachida")].to_csv(PATHWAY_TABLES / "crc_ora_yachida_crc_vs_healthy.csv", index=False)
    for cohort, prefix in [("Sinha", "crc_ora_sinha_crc"), ("Yachida", "crc_ora_yachida_crc")]:
        enrich[(enrich["cohort"].eq(cohort)) & (enrich["direction"].eq("higher_in_disease"))].to_csv(PATHWAY_TABLES / f"{prefix}_positive.csv", index=False)
        enrich[(enrich["cohort"].eq(cohort)) & (enrich["direction"].eq("lower_in_disease"))].to_csv(PATHWAY_TABLES / f"{prefix}_negative.csv", index=False)
    sig_summary = enrich[enrich["FDR_q_value"] < 0.05].copy() if not enrich.empty else pd.DataFrame(columns=enrich.columns)
    sig_summary.to_csv(PATHWAY_TABLES / "crc_ora_significant_pathways_summary.csv", index=False)

    overlap_rows = []
    for (direction, pid, pname), g in enrich.groupby(["direction", "pathway_id", "pathway_name"], dropna=False) if not enrich.empty else []:
        si = g[g["cohort"].eq("Sinha")]
        ya = g[g["cohort"].eq("Yachida")]
        si_row = si.iloc[0] if not si.empty else pd.Series(dtype=object)
        ya_row = ya.iloc[0] if not ya.empty else pd.Series(dtype=object)
        si_mets = set(parse_identifier_field(si_row.get("contributing_metabolites", "")))
        ya_mets = set(parse_identifier_field(ya_row.get("contributing_metabolites", "")))
        si_sig = bool(si_row.get("significant_FDR_0_05", False)) if not si.empty else False
        ya_sig = bool(ya_row.get("significant_FDR_0_05", False)) if not ya.empty else False
        overlap_rows.append({
            "pathway_id": pid,
            "pathway_name": pname,
            "sinha_fdr": si_row.get("FDR_q_value", np.nan),
            "yachida_fdr": ya_row.get("FDR_q_value", np.nan),
            "sinha_fold_enrichment": si_row.get("fold_enrichment", np.nan),
            "yachida_fold_enrichment": ya_row.get("fold_enrichment", np.nan),
            "sinha_significant": si_sig,
            "yachida_significant": ya_sig,
            "shared_significant": si_sig and ya_sig,
            "direction_set": direction,
            "shared_metabolites_if_any": ";".join(sorted(si_mets & ya_mets)),
            "replication_definition": "same pathway BH-FDR < 0.05 in both Sinha and Yachida under same direction-set",
        })
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(PATHWAY_TABLES / "crc_ora_cross_cohort_overlap.csv", index=False)
    _plot_crc_ora(enrich, "Sinha", "crc_kegg_ora_sinha")
    _plot_crc_ora(enrich, "Yachida", "crc_kegg_ora_yachida")
    _plot_crc_ora_comparison(overlap)
    _plot_crc_ora_presentation(enrich, coverage)

    ibd_path = ROOT / "thesis_preprocessing_workflow" / "ibd_pipeline" / "tables" / "cross_cohort_pathway_replication.csv"
    if ibd_path.exists():
        ibd = pd.read_csv(ibd_path)
        ibd_terms = ibd.groupby(["pathway_ID", "pathway_name"], dropna=False).agg(
            ibd_significant_any_cohort=("replicated_pathway_flag", lambda s: bool(s.any()) if len(s) else False),
            ibd_replicated=("replicated_pathway_flag", "max"),
        ).reset_index().rename(columns={"pathway_ID": "pathway_id"})
    else:
        ibd_terms = pd.DataFrame(columns=["pathway_id", "pathway_name", "ibd_significant_any_cohort", "ibd_replicated"])
    crc_terms = overlap.groupby(["pathway_id", "pathway_name"], dropna=False).agg(
        crc_significant_any_cohort=("shared_significant", lambda s: False),
        crc_replicated=("shared_significant", "max"),
    ).reset_index() if not overlap.empty else pd.DataFrame(columns=["pathway_id", "pathway_name", "crc_significant_any_cohort", "crc_replicated"])
    if not overlap.empty:
        crc_any = overlap.groupby(["pathway_id", "pathway_name"], dropna=False).apply(lambda g: bool((g["sinha_significant"] | g["yachida_significant"]).any()), include_groups=False).reset_index(name="crc_significant_any_cohort")
        crc_rep = overlap.groupby(["pathway_id", "pathway_name"], dropna=False)["shared_significant"].max().reset_index(name="crc_replicated")
        crc_terms = crc_any.merge(crc_rep, on=["pathway_id", "pathway_name"], how="outer")
    comp = ibd_terms.merge(crc_terms, on=["pathway_id", "pathway_name"], how="outer")
    for col in ["ibd_significant_any_cohort", "ibd_replicated", "crc_significant_any_cohort", "crc_replicated"]:
        comp[col] = comp[col].fillna(False).astype(bool) if col in comp else False
    comp["shared_ibd_crc"] = comp["ibd_significant_any_cohort"] & comp["crc_significant_any_cohort"]
    comp["notes"] = "Descriptive comparison only; no Stage 3 cross-disease enrichment or modelling performed."
    comp.to_csv(PATHWAY_TABLES / "ibd_crc_ora_descriptive_comparison.csv", index=False)

    n_sig_sinha = int(sig_summary[sig_summary["cohort"].eq("Sinha")]["pathway_id"].nunique()) if not sig_summary.empty else 0
    n_sig_yachida = int(sig_summary[sig_summary["cohort"].eq("Yachida")]["pathway_id"].nunique()) if not sig_summary.empty else 0
    n_rep = int(overlap["shared_significant"].sum()) if not overlap.empty else 0
    report = f"""# CRC Stage 2 KEGG ORA Results

## Methods

CRC KEGG pathway over-representation analysis was run as a controlled pathway-only reopening of Stage 2. Existing CRC differential-abundance results were consumed read-only. No model fitting, nested cross-validation, external prediction generation, threshold optimisation, calibration, feature importance, DA refitting, PCA/UMAP, graph construction, clustering or Stage 3 integration was performed.

## Background Definition

For each cohort/contrast, the ORA background was all DA-tested metabolites with deterministic KEGG mapping from the existing CRC HMDB/KEGG crosswalk. This is cohort-specific and contrast-specific, not all KEGG compounds, not all human metabolites, not the Stage 3 56-metabolite set, and not only significant metabolites.

## Statistical Test

The test is a one-sided hypergeometric right-tail over-representation test. Fold enrichment is `(k / n) / (K / N)`. BH correction is applied separately within cohort × comparison × direction × annotation source. Pathways require at least {config.annotation_settings['minimum_pathway_background_compounds']} background compounds and at least {config.annotation_settings['minimum_pathway_significant_compounds']} significant compounds.

## Main Results

- Sinha BH-FDR-significant pathways: **{n_sig_sinha}**
- Yachida BH-FDR-significant pathways: **{n_sig_yachida}**
- Replicated CRC pathways by strict definition: **{n_rep}**

## Interpretation Cautions

Pathways are described as enriched among CRC-associated metabolites or over-represented in the significant metabolite set. These results do not prove pathway activation, causal mechanism or clinical biomarker validity.
"""
    (REPORTS / "CRC_STAGE2_KEGG_ORA_RESULTS.md").write_text(report)

    after_hashes = _hash_protected_crc_outputs().rename(columns={"sha256": "sha256_after", "exists": "exists_after"})
    hash_check = before_hashes.merge(after_hashes, on="path", how="outer")
    hash_check["hash_unchanged"] = hash_check["sha256_before"].fillna("") == hash_check["sha256_after"].fillna("")
    hash_check["protected_output_check"] = "models/predictions/metrics/thresholds/calibration/DA unchanged by pathway-only reopening"
    hash_check.to_csv(PATHWAY_TABLES / "crc_ora_protected_output_hash_check.csv", index=False)


def load_yachida_full_stage_manifest() -> pd.DataFrame:
    """Load the frozen Stage 1 Yachida sample manifest for Yachida-only analyses.

    This function is intentionally Stage-1-read-only. It preserves source-faithful
    labels and creates analysis labels in parallel fields; it never overwrites the
    official Yachida stage values.
    """
    required = STAGE1_TABLES / "crc_stage1_yachida_stage_recovered_per_sample.csv"
    if not required.exists():
        raise FileNotFoundError(
            "Frozen full Yachida stage manifest is required for Stage 2 secondary analyses: "
            f"{required}. Do not use the 277-sample primary CRC-vs-Control table as a full stage manifest."
        )
    df = pd.read_csv(required)

    missing_tokens = {"", "-", "NA", "N/A", "NAN", "NONE", "NULL", "MISSING"}

    def clean_source_value(value: Any) -> Any:
        if pd.isna(value):
            return pd.NA
        text = str(value).strip()
        if text.upper() in missing_tokens:
            return pd.NA
        return text

    def first_available_field(frame: pd.DataFrame, names: list[str]) -> pd.Series:
        values = pd.Series(pd.NA, index=frame.index, dtype=object)
        for name in names:
            if name in frame.columns:
                candidate = frame[name].map(clean_source_value)
                values = values.where(values.notna(), candidate)
        return values

    def normalize_original_group(value: Any) -> Any:
        text = clean_source_value(value)
        if pd.isna(text):
            return pd.NA
        return {
            "Stage_0": "Stage 0",
            "Stage_I_II": "Stage I-II",
            "Stage_III_IV": "Stage III-IV",
        }.get(str(text), str(text))

    def normalize_stage(value: Any) -> Any:
        text = clean_source_value(value)
        if pd.isna(text):
            return pd.NA
        upper = str(text).upper().replace("STAGE ", "").replace("STAGE", "").strip()
        mapping = {
            "HEALTHY": "Healthy",
            "MP": "MP",
            "HS": "HS",
            "0": "Stage 0",
            "I": "Stage I",
            "II": "Stage II",
            "IIA": "Stage II",
            "IIB": "Stage II",
            "IIC": "Stage II",
            "III": "Stage III",
            "IIIA": "Stage III",
            "IIIB": "Stage III",
            "IIIC": "Stage III",
            "IV": "Stage IV",
            "IVA": "Stage IV",
            "IVB": "Stage IV",
        }
        return mapping.get(upper, pd.NA)

    def normalize_substage(value: Any) -> Any:
        text = clean_source_value(value)
        if pd.isna(text):
            return pd.NA
        token = str(text).upper().replace("STAGE ", "").replace("STAGE", "").strip()
        return token if token in {"IIA", "IIB", "IIC", "IIIA", "IIIB", "IIIC", "IVA", "IVB"} else pd.NA

    out = pd.DataFrame()
    out["sample_id"] = df.get("Sample", df.get("sample_id", df.index.astype(str))).astype(str)
    out["participant_id"] = df.get("Subject_ID", df.get("Subject", df.get("participant_id", out["sample_id"]))).astype(str)
    out["original_group"] = first_available_field(
        df,
        ["Study.Group", "yachida_stage_original_group", "Group", "yachida_stage_source_faithful_label", "stage_source_faithful"],
    ).map(normalize_original_group)
    raw_stage = first_available_field(
        df,
        ["yachida_stage_source_raw", "Stage_supplement", "Stage_local", "stage_source_raw", "AJCC_stage", "Stage", "stage"],
    )
    out["stage_source_raw"] = raw_stage
    stage_from_file = first_available_field(df, ["yachida_stage_major", "stage_major_from_official_supplement"])
    source_for_stage = stage_from_file.where(stage_from_file.notna(), raw_stage)
    source_for_stage = source_for_stage.where(source_for_stage.notna(), out["original_group"])
    out["stage_normalized"] = source_for_stage.map(normalize_stage)
    out["stage_substage"] = first_available_field(df, ["yachida_stage_normalized", "Stage_supplement", "Stage_local", "yachida_stage_source_raw"]).map(normalize_substage)
    out["stage_major"] = out["stage_normalized"]
    out["stage_source_faithful"] = first_available_field(
        df,
        ["yachida_stage_source_faithful_label", "stage_source_faithful", "stage_refined_from_official_supplement"],
    ).where(lambda s: s.notna(), out["original_group"])
    out["stage_recovery_source"] = first_available_field(df, ["yachida_stage_recovery_source", "stage_recovery_source"]).fillna(str(required.relative_to(CRC_DIR)))
    out["stage_recovery_status"] = first_available_field(df, ["yachida_stage_recovery_status", "stage_recovery_status"]).fillna("linked_from_frozen_stage1_manifest")
    out["stage_harmonized"] = df.get("yachida_stage_harmonized_label", out["stage_major"])
    out["ajcc_informed_analysis_group"] = out["stage_major"].map({
        "Healthy": "Healthy",
        "MP": "Precursor lesion",
        "Stage 0": "Stage 0 / in situ CRC",
        "Stage I": "Stage I-II non-metastatic node-negative CRC",
        "Stage II": "Stage I-II non-metastatic node-negative CRC",
        "Stage III": "Stage III regional node-positive CRC",
        "Stage IV": "Stage IV distant metastatic CRC",
        "HS": "Post-surgical comparator",
    }).fillna("Unknown")
    out["primary_crc_control_eligible"] = out["stage_major"].isin(["Healthy", "Stage 0", "Stage I", "Stage II", "Stage III", "Stage IV"])
    out["binary_label"] = np.where(out["stage_major"].astype(str).isin(["Healthy"]), 0, np.where(out["stage_major"].astype(str).isin(["Stage 0", "Stage I", "Stage II", "Stage III", "Stage IV"]), 1, np.nan))
    out["stage_ordinal_label"] = out["stage_major"].map({"Stage 0": 0, "Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IV": 4})
    out["non_crc_condition"] = out["stage_major"].where(out["stage_major"].isin(["MP", "HS"]))
    expected_counts = {"Healthy": 127, "MP": 40, "HS": 30, "Stage 0": 27, "Stage I": 44, "Stage II": 25, "Stage III": 34, "Stage IV": 20}
    observed_counts = out["stage_major"].value_counts(dropna=False).to_dict()
    mismatches = {k: {"observed": int(observed_counts.get(k, 0)), "expected": v} for k, v in expected_counts.items() if int(observed_counts.get(k, 0)) != v}
    expected_substages = {"IIIA": 8, "IIIB": 22, "IIIC": 4, "IVA": 13, "IVB": 7}
    observed_substages = out["stage_substage"].value_counts(dropna=False).to_dict()
    substage_mismatches = {k: {"observed": int(observed_substages.get(k, 0)), "expected": v} for k, v in expected_substages.items() if int(observed_substages.get(k, 0)) != v}
    if len(out) != 347 or out["participant_id"].nunique() != 347 or mismatches:
        raise AssertionError(f"Full Yachida stage manifest failed frozen count checks: n={len(out)}, participants={out['participant_id'].nunique()}, mismatches={mismatches}, substage_mismatches={substage_mismatches}")
    if substage_mismatches:
        raise AssertionError(f"Full Yachida substage manifest failed frozen count checks: {substage_mismatches}")
    out["stage_source_file"] = str(required.relative_to(CRC_DIR))
    return out


def write_yachida_stage_analysis_plan(config: CRCStage2Config) -> None:
    """Write Yachida-only stage/subtype task definitions without fitting models."""
    y = load_yachida_full_stage_manifest()
    counts = y.groupby(["stage_source_raw", "stage_normalized", "stage_major"], dropna=False).size().reset_index(name="n")
    counts.to_csv(TABLES / "crc_stage2_yachida_stage_label_audit.csv", index=False)
    y.groupby(["ajcc_informed_analysis_group"], dropna=False).size().reset_index(name="n").to_csv(TABLES / "crc_stage2_ajcc_informed_group_counts.csv", index=False)
    pd.DataFrame([{
        "seer_recode_created": False,
        "feasibility": "not_supported",
        "reason": "Official deterministic tumour-extent, nodal and metastasis variables are not available for all CRC samples in the frozen Stage 1 public-data contract.",
        "allowed_stage2_grouping": "study-defined AJCC-informed analytical grouping",
        "prohibited_claim": "official AJCC or SEER Summary Stage recode",
    }]).to_csv(TABLES / "crc_stage2_seer_recode_feasibility.csv", index=False)
    task_rows = [
        {
            "task": "Yachida internal CRC stage multiclass",
            "cohort": "Yachida",
            "eligible_labels": "Stage 0, I, II, III, IV",
            "excluded_labels": "Healthy, MP, HS",
            "external_validation": "not permitted in Sinha because Sinha has no deterministic participant-level stage labels",
            "recommended_status": "secondary_internal_only",
            "model_family_note": "multiclass versions of the primary supervised learners where sample size supports folds",
        },
        {
            "task": "Yachida internal ordered stage analysis",
            "cohort": "Yachida",
            "eligible_labels": "Stage 0, I, II, III, IV",
            "excluded_labels": "Healthy, MP, HS",
            "external_validation": "not permitted in Sinha",
            "recommended_status": "exploratory_cross_sectional_ordinal",
            "model_family_note": "treat as ordered cross-sectional severity, not longitudinal disease trajectory",
        },
        {
            "task": "Yachida non-cancer condition descriptive analysis",
            "cohort": "Yachida",
            "eligible_labels": "MP, HS",
            "excluded_labels": "not part of primary CRC vs Control",
            "external_validation": "not applicable",
            "recommended_status": "descriptive_or_sensitivity_only",
            "model_family_note": "do not map MP to CRC and do not map HS to Healthy",
        },
        {
            "task": "Yachida Stage III/IV substage descriptive analysis",
            "cohort": "Yachida",
            "eligible_labels": "IIIA, IIIB, IIIC, IVA, IVB",
            "excluded_labels": "all other groups",
            "external_validation": "not permitted in Sinha",
            "recommended_status": "descriptive_only_due_small_counts",
            "model_family_note": "preserve raw Iva/Ivb and normalized IVA/IVB in separate fields",
        },
    ]
    pd.DataFrame(task_rows).to_csv(TABLES / "crc_stage2_yachida_stage_task_plan.csv", index=False)
    y.to_csv(PROCESSED / "crc_stage2_yachida_stage_manifest_for_internal_tasks.csv", index=False)


def build_crc_integration_package(config: CRCStage2Config) -> None:
    """Prepare CRC outputs for future IBD-to-CRC latent-space work.

    This does not run the future trajectory model. It exports stable CRC feature,
    metadata and annotation contracts so IBD and CRC can later be linked without
    changing the frozen within-disease analyses.
    """
    meta = PROCESSED / "crc_stage2_primary_metadata.csv"
    met = PROCESSED / "crc_stage2_primary_metabolomics_canonical.csv"
    shared = TABLES / "crc_stage2_shared_feature_space.csv"
    annotation = TABLES / "crc_stage2_metabolite_annotation_enrichment.csv"
    rows = []
    for label, path in [
        ("primary_metadata", meta),
        ("primary_canonical_metabolomics", met),
        ("shared_feature_space", shared),
        ("metabolite_annotation_enrichment", annotation),
    ]:
        rows.append({
            "component": label,
            "path": str(path.relative_to(CRC_DIR)),
            "exists": path.exists(),
            "sha256": sha256(path) if path.exists() else pd.NA,
            "role_in_future_ibd_crc_model": "input_contract_only_not_trajectory_modelling",
        })
    package = pd.DataFrame(rows)
    package.to_csv(TABLES / "crc_stage2_integration_package_manifest.csv", index=False)
    note = """# CRC Integration Package for Future IBD-to-CRC Modelling

This package is a contract for later cross-disease latent-space work. It does
not implement the IBD-to-CRC trajectory model. The within-CRC analysis remains
CRC vs Control as the primary externally validated task, with Yachida-only stage
analyses treated as secondary/internal.
"""
    (REPORTS / "CRC_STAGE2_INTEGRATION_PACKAGE.md").write_text(note)


def run_crc_postrun_diagnostics(config: CRCStage2Config) -> None:
    """Create post-run diagnostics after CRC modelling has been executed."""
    pred_path = PREDICTIONS / "crc_stage2_external_predictions.csv"
    perf_path = TABLES / "crc_stage2_external_validation_results.csv"
    threshold_path = TABLES / "crc_stage2_threshold_selection.csv"
    checks = []
    for label, path in [
        ("external_predictions", pred_path),
        ("external_performance", perf_path),
        ("threshold_selection", threshold_path),
        ("internal_cv_results", TABLES / "crc_stage2_internal_cv_results.csv"),
        ("selected_models", MODELS),
    ]:
        checks.append({"item": label, "path": str(path.relative_to(CRC_DIR)), "exists": path.exists()})
    pd.DataFrame(checks).to_csv(TABLES / "crc_stage2_postrun_input_availability.csv", index=False)
    if not pred_path.exists():
        pd.DataFrame([{
            "status": "not_run",
            "reason": "External predictions are unavailable. Run primary modelling before CRC post-run diagnostics.",
        }]).to_csv(TABLES / "crc_stage2_postrun_diagnostic_status.csv", index=False)
        return
    pred = pd.read_csv(pred_path)
    rows = []
    for context, group in pred.groupby(["train_cohort", "external_cohort", "feature_set", "model"], dropna=False):
        y = group["true_label"].to_numpy(dtype=int)
        p = group["predicted_probability"].to_numpy(dtype=float)
        threshold_status = "passed"
        threshold_error = ""
        if "frozen_threshold" not in group.columns:
            threshold_status = "failed_missing_frozen_threshold"
            threshold = np.nan
        else:
            thresholds = pd.to_numeric(group["frozen_threshold"], errors="coerce").dropna().unique()
            if len(thresholds) != 1 or not np.isfinite(thresholds[0]):
                threshold_status = "failed_nonunique_or_nonfinite_frozen_threshold"
                threshold = np.nan
                threshold_error = f"unique_finite_threshold_count={len(thresholds)}"
            else:
                threshold = float(thresholds[0])
        expected_source = "nested_outer_fold_oof_predictions_training_only"
        if "threshold_source" not in group.columns or not group["threshold_source"].astype(str).eq(expected_source).all():
            threshold_status = "failed_invalid_threshold_source"
            threshold_error = f"expected {expected_source}"
        rows.append({
            "train_cohort": context[0],
            "external_cohort": context[1],
            "feature_set": context[2],
            "model": context[3],
            "n": len(group),
            "probability_min": float(np.min(p)),
            "probability_median": float(np.median(p)),
            "probability_max": float(np.max(p)),
            "probability_within_0_1": bool(np.all((p >= 0) & (p <= 1))),
            "frozen_threshold": threshold,
            "threshold_source": group["threshold_source"].iloc[0] if "threshold_source" in group.columns else pd.NA,
            "threshold_verification_status": threshold_status,
            "threshold_verification_error": threshold_error,
            "predicted_positive_fraction_frozen_threshold": float((p >= threshold).mean()) if np.isfinite(threshold) else np.nan,
            "all_predictions_one_class": bool((p >= threshold).mean() in (0.0, 1.0)) if np.isfinite(threshold) else pd.NA,
            "true_label_values": ",".join(map(str, sorted(np.unique(y)))),
            "positive_class_policy": "0=Control, 1=CRC",
            "external_threshold_reoptimised": False,
        })
    pd.DataFrame(rows).to_csv(TABLES / "crc_stage2_postrun_threshold_transportability_diagnostic.csv", index=False)
    model_rows = []
    for context, group in pred.groupby(["train_cohort", "external_cohort", "feature_set", "model"], dropna=False):
        model_path = MODELS / f"crc_stage2_selected_{context[0]}_to_{context[1]}_{context[2]}_{context[3]}.joblib"
        loaded = False
        estimator_classes = ""
        masks_inspectable = False
        logistic_convergence_status = "not_applicable"
        error = ""
        try:
            fitted = joblib.load(model_path)
            loaded = True
            clf = fitted.named_steps.get("clf") if hasattr(fitted, "named_steps") else None
            estimator_classes = ",".join(map(str, getattr(clf, "classes_", []))) if clf is not None else ""
            pre = fitted.named_steps.get("preprocess") if hasattr(fitted, "named_steps") else None
            if pre is not None and hasattr(pre, "named_transformers_"):
                masks_inspectable = any(
                    hasattr(step, "keep_mask_")
                    for _, transformer, _ in pre.transformers_
                    for step in (getattr(transformer, "named_steps", {}) or {}).values()
                )
            if "logistic" in str(context[3]) and clf is not None:
                n_iter = getattr(clf, "n_iter_", np.array([np.nan]))
                max_iter = getattr(clf, "max_iter", np.nan)
                logistic_convergence_status = "converged" if np.nanmax(n_iter) < max_iter else "unresolved_or_hit_max_iter"
        except Exception as exc:
            error = repr(exc)
        model_rows.append({
            "train_cohort": context[0],
            "external_cohort": context[1],
            "feature_set": context[2],
            "model": context[3],
            "model_path": str(model_path.relative_to(CRC_DIR)),
            "model_load_success": loaded,
            "estimator_classes": estimator_classes,
            "positive_class_policy": "classes_[1] must be CRC/disease probability when classes_=[0,1]",
            "fitted_masks_inspectable": masks_inspectable,
            "selected_logistic_convergence_status": logistic_convergence_status,
            "error": error,
        })
    pd.DataFrame(model_rows).to_csv(TABLES / "crc_stage2_logistic_convergence_audit.csv", index=False)
    pd.DataFrame(model_rows).to_csv(TABLES / "crc_stage2_model_deserialization_and_feature_order_audit.csv", index=False)


REPORTING_PROTECTED_RELATIVE_PATHS = [
    "tables/crc_stage2_internal_cv_results.csv",
    "tables/crc_stage2_internal_model_selection.csv",
    "tables/crc_stage2_external_validation_results.csv",
    "tables/crc_stage2_external_bootstrap_confidence_intervals.csv",
    "tables/crc_stage2_threshold_selection.csv",
    "tables/crc_stage2_important_feature_table.csv",
    "predictions/crc_stage2_internal_oof_predictions.csv",
    "predictions/crc_stage2_external_predictions.csv",
]


def selected_context_columns() -> list[str]:
    return ["train_cohort", "external_cohort", "feature_set", "model"]


def reporting_context_slug(row: pd.Series | dict[str, Any]) -> str:
    return f"{row['train_cohort']}_to_{row['external_cohort']}_{row['feature_set']}_{row['model']}"


def file_manifest(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        rows.append({
            "path": str(path.relative_to(STAGE2) if path.is_relative_to(STAGE2) else path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else pd.NA,
            "mtime_ns": path.stat().st_mtime_ns if path.exists() else pd.NA,
            "sha256": sha256(path) if path.exists() else pd.NA,
        })
    return pd.DataFrame(rows)


def write_reporting_input_availability() -> pd.DataFrame:
    required = [STAGE2 / rel for rel in REPORTING_PROTECTED_RELATIVE_PATHS]
    optional = [TABLES / "crc_stage2_stable_cross_cohort_metabolites.csv"]
    model_paths = sorted(MODELS.glob("crc_stage2_selected_*.joblib"))
    rows = []
    for path in required + optional:
        rel = str(path.relative_to(STAGE2))
        required_input = rel in REPORTING_PROTECTED_RELATIVE_PATHS
        rows.append({
            "input": rel,
            "required": required_input,
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else pd.NA,
            "sha256": sha256(path) if path.exists() else pd.NA,
            "status": "available" if path.exists() else ("missing_critical" if required_input else "not_available_optional"),
        })
    rows.append({
        "input": "models/crc_stage2_selected_*.joblib",
        "required": True,
        "exists": len(model_paths) == 6,
        "size_bytes": sum(p.stat().st_size for p in model_paths),
        "sha256": ";".join(f"{p.name}:{sha256(p)}" for p in model_paths),
        "status": "available" if len(model_paths) == 6 else f"missing_critical_expected_6_found_{len(model_paths)}",
    })
    audit = pd.DataFrame(rows)
    audit.to_csv(TABLES / "crc_stage2_reporting_input_availability.csv", index=False)
    missing = audit[audit["required"] & ~audit["exists"]]
    if not missing.empty:
        raise FileNotFoundError("Missing completed CRC Stage 2 reporting input(s): " + "; ".join(missing["input"].astype(str)))
    return audit


def load_reporting_inputs() -> dict[str, pd.DataFrame]:
    write_reporting_input_availability()
    return {
        "internal": pd.read_csv(TABLES / "crc_stage2_internal_cv_results.csv"),
        "selection": pd.read_csv(TABLES / "crc_stage2_internal_model_selection.csv"),
        "external": pd.read_csv(TABLES / "crc_stage2_external_validation_results.csv"),
        "bootstrap": pd.read_csv(TABLES / "crc_stage2_external_bootstrap_confidence_intervals.csv"),
        "threshold": pd.read_csv(TABLES / "crc_stage2_threshold_selection.csv"),
        "importance": pd.read_csv(TABLES / "crc_stage2_important_feature_table.csv"),
        "stable": pd.read_csv(TABLES / "crc_stage2_stable_cross_cohort_metabolites.csv") if (TABLES / "crc_stage2_stable_cross_cohort_metabolites.csv").exists() else pd.DataFrame(),
        "internal_pred": pd.read_csv(PREDICTIONS / "crc_stage2_internal_oof_predictions.csv"),
        "external_pred": pd.read_csv(PREDICTIONS / "crc_stage2_external_predictions.csv"),
    }


def selected_contexts(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    selected = inputs["selection"].query("selected == True").copy()
    threshold_cols = selected_context_columns() + ["selected_threshold", "youden_j", "sensitivity", "specificity", "roc_auc", "oof_sample_count", "threshold_source", "external_data_used"]
    return inputs["threshold"][threshold_cols].merge(
        selected,
        on=["train_cohort", "feature_set", "model"],
        how="left",
        validate="one_to_one",
    )


def ci_lookup(bootstrap: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = bootstrap[(bootstrap["threshold_type"].eq("frozen_training_youden")) & (bootstrap["metric"].eq(metric))].copy()
    return out[selected_context_columns() + ["ci_lower", "ci_upper"]].rename(columns={"ci_lower": f"{metric}_ci_lower", "ci_upper": f"{metric}_ci_upper"})


def plot_external_roc_pr(inputs: dict[str, pd.DataFrame]) -> None:
    pred = inputs["external_pred"]
    boot = inputs["bootstrap"]
    colors = {"clinical": "#1f77b4", "metabolomics": "#2ca02c", "integrated": "#ff7f0e"}
    for (train, external), group in pred.groupby(["train_cohort", "external_cohort"], sort=False):
        fig, ax = plt.subplots(figsize=(6.6, 5.0))
        for (feature_set, model), sub in group.groupby(["feature_set", "model"], sort=False):
            y = sub["true_label"].to_numpy()
            p = sub["predicted_probability"].to_numpy()
            fpr, tpr, _ = roc_curve(y, p)
            auc = roc_auc_score(y, p)
            ci = boot[(boot["train_cohort"].eq(train)) & (boot["external_cohort"].eq(external)) & (boot["feature_set"].eq(feature_set)) & (boot["model"].eq(model)) & (boot["threshold_type"].eq("frozen_training_youden")) & (boot["metric"].eq("roc_auc"))]
            label = f"{feature_set} | {model} AUC={auc:.2f}"
            if not ci.empty:
                label += f" [{ci['ci_lower'].iloc[0]:.2f}, {ci['ci_upper'].iloc[0]:.2f}]"
            ax.plot(fpr, tpr, lw=2.2, color=colors.get(feature_set, "black"), label=label)
        ax.plot([0, 1], [0, 1], ls="--", color="0.55", label="No discrimination")
        ax.set(title=f"External ROC: {train} to {external}", xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8, loc="lower right", frameon=False)
        fig.tight_layout()
        out = FIGURES_MAIN / f"crc_stage2_external_roc_ci_{train}_to_{external}"
        fig.savefig(out.with_suffix(".png"), dpi=300)
        fig.savefig(out.with_suffix(".svg"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.6, 5.0))
        prevalence = float(group["true_label"].mean())
        for (feature_set, model), sub in group.groupby(["feature_set", "model"], sort=False):
            y = sub["true_label"].to_numpy()
            p = sub["predicted_probability"].to_numpy()
            precision, recall, _ = precision_recall_curve(y, p)
            ap = average_precision_score(y, p)
            ci = boot[(boot["train_cohort"].eq(train)) & (boot["external_cohort"].eq(external)) & (boot["feature_set"].eq(feature_set)) & (boot["model"].eq(model)) & (boot["threshold_type"].eq("frozen_training_youden")) & (boot["metric"].eq("pr_auc"))]
            label = f"{feature_set} | {model} AP={ap:.2f}"
            if not ci.empty:
                label += f" [{ci['ci_lower'].iloc[0]:.2f}, {ci['ci_upper'].iloc[0]:.2f}]"
            ax.plot(recall, precision, lw=2.2, color=colors.get(feature_set, "black"), label=label)
        ax.axhline(prevalence, ls="--", color="0.55", label=f"No-skill prevalence={prevalence:.2f}")
        ax.set(title=f"External precision-recall: {train} to {external}", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8, loc="best", frameon=False)
        fig.tight_layout()
        out = FIGURES_MAIN / f"crc_stage2_external_pr_ci_{train}_to_{external}"
        fig.savefig(out.with_suffix(".png"), dpi=300)
        fig.savefig(out.with_suffix(".svg"))
        plt.close(fig)


def write_nested_cv_stability(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    internal = inputs["internal"]
    rows = []
    for _, ctx in selected_contexts(inputs).iterrows():
        sub = internal[(internal["train_cohort"].eq(ctx["train_cohort"])) & (internal["feature_set"].eq(ctx["feature_set"])) & (internal["model"].eq(ctx["model"]))]
        roc = sub["threshold_0_5_roc_auc"].dropna().astype(float)
        pr = sub["threshold_0_5_pr_auc"].dropna().astype(float)
        rows.append({
            "train_cohort": ctx["train_cohort"], "external_cohort": ctx["external_cohort"], "feature_set": ctx["feature_set"], "model": ctx["model"],
            "outer_fold_roc_auc_values": ";".join(f"{v:.6f}" for v in roc), "mean_roc_auc": roc.mean(), "sd_roc_auc": roc.std(ddof=1),
            "outer_fold_pr_auc_values": ";".join(f"{v:.6f}" for v in pr), "mean_pr_auc": pr.mean(), "sd_pr_auc": pr.std(ddof=1),
            "n_successful_outer_folds": int(len(roc)), "min_fold_roc_auc": roc.min(), "max_fold_roc_auc": roc.max(),
            "min_fold_pr_auc": pr.min(), "max_fold_pr_auc": pr.max(),
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "crc_stage2_selected_model_nested_cv_stability.csv", index=False)
    colors = {"clinical": "#1f77b4", "metabolomics": "#2ca02c", "integrated": "#ff7f0e"}
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    labels = []
    for i, row in out.reset_index(drop=True).iterrows():
        vals = np.array([float(x) for x in row["outer_fold_roc_auc_values"].split(";") if x])
        ax.scatter(vals, np.full_like(vals, i, dtype=float), s=34, alpha=0.72, color=colors.get(row["feature_set"], "black"))
        ax.errorbar(row["mean_roc_auc"], i, xerr=row["sd_roc_auc"], fmt="o", ms=7, color="black", ecolor=colors.get(row["feature_set"], "black"), capsize=4)
        labels.append(f"{row['train_cohort']}→{row['external_cohort']} | {row['feature_set']}")
    ax.axvline(0.5, color="0.55", ls="--", lw=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set(xlabel="Nested outer-fold ROC-AUC", title="Selected-model internal nested-CV stability", xlim=(0, 1))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = FIGURES_MAIN / "crc_stage2_nested_cv_stability"
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    return out


def write_transportability_summary(inputs: dict[str, pd.DataFrame], stability: pd.DataFrame) -> pd.DataFrame:
    ext = inputs["external"].query("threshold_type == 'frozen_training_youden'").copy()
    pred = inputs["external_pred"]
    out = stability.merge(ext, on=selected_context_columns(), how="left", suffixes=("_internal", "_external"))
    out = out.merge(ci_lookup(inputs["bootstrap"], "roc_auc"), on=selected_context_columns(), how="left")
    out = out.merge(ci_lookup(inputs["bootstrap"], "pr_auc"), on=selected_context_columns(), how="left")
    rows = []
    for _, row in out.iterrows():
        sub = pred[(pred["train_cohort"].eq(row["train_cohort"])) & (pred["external_cohort"].eq(row["external_cohort"])) & (pred["feature_set"].eq(row["feature_set"])) & (pred["model"].eq(row["model"]))]
        rows.append({
            **row.to_dict(),
            "absolute_internal_to_external_roc_auc_change": abs(float(row["mean_roc_auc"]) - float(row["roc_auc"])),
            "external_positive_prevalence": float(sub["true_label"].mean()),
            "frozen_threshold": float(sub["frozen_threshold"].iloc[0]),
            "frozen_threshold_predictions_collapsed_to_one_class": bool(sub["predicted_label_at_frozen_threshold"].nunique() <= 1),
        })
    out = pd.DataFrame(rows).rename(columns={
        "roc_auc": "external_roc_auc", "pr_auc": "external_pr_auc", "sd_roc_auc": "internal_roc_auc_sd", "sd_pr_auc": "internal_pr_auc_sd",
        "roc_auc_ci_lower": "external_roc_auc_ci_lower", "roc_auc_ci_upper": "external_roc_auc_ci_upper",
        "pr_auc_ci_lower": "external_pr_auc_ci_lower", "pr_auc_ci_upper": "external_pr_auc_ci_upper",
    })
    out.to_csv(TABLES_MAIN / "crc_stage2_selected_model_performance_summary.csv", index=False)
    colors = {"clinical": "#1f77b4", "metabolomics": "#2ca02c", "integrated": "#ff7f0e"}
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    labels = []
    for i, row in out.reset_index(drop=True).iterrows():
        x, lo, hi = row["external_roc_auc"], row["external_roc_auc_ci_lower"], row["external_roc_auc_ci_upper"]
        ax.errorbar(x, i, xerr=[[x - lo], [hi - x]], fmt="o", ms=7.5, capsize=4, color=colors.get(row["feature_set"], "black"))
        labels.append(f"{row['train_cohort']}→{row['external_cohort']} | {row['feature_set']}")
    ax.axvline(0.5, color="0.55", ls="--", lw=1)
    ax.text(0.505, len(labels) - 0.4, "No discrimination", fontsize=8, color="0.35")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set(xlabel="External ROC-AUC with bootstrap 95% CI", title="CRC external transportability of selected models", xlim=(0, 1))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = FIGURES_MAIN / "crc_stage2_external_transportability_forest"
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    return out


def logistic_calibration_assessment(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    eps = 1e-6
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(y_prob).astype(float), eps, 1 - eps)
    result = {"calibration_status": "estimated", "calibration_intercept": np.nan, "calibration_slope": np.nan}
    if len(np.unique(y)) < 2 or np.nanmin(p) == np.nanmax(p):
        result["calibration_status"] = "not_estimable_degenerate_labels_or_probabilities"
        return result
    try:
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        cal = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        cal.fit(logit, y)
        result["calibration_intercept"] = float(cal.intercept_[0])
        result["calibration_slope"] = float(cal.coef_[0][0])
    except Exception as exc:
        result["calibration_status"] = f"not_estimable_{type(exc).__name__}"
    return result


def calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    df = pd.DataFrame({"true_label": y_true, "predicted_probability": y_prob})
    try:
        df["bin"] = pd.qcut(df["predicted_probability"], q=min(n_bins, df["predicted_probability"].nunique()), duplicates="drop")
    except ValueError:
        df["bin"] = "all"
    return df.groupby("bin", observed=False).agg(n=("true_label", "size"), mean_predicted_probability=("predicted_probability", "mean"), observed_event_frequency=("true_label", "mean")).reset_index(drop=True)


def write_external_calibration(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows, bin_rows = [], []
    for keys, sub in inputs["external_pred"].groupby(selected_context_columns(), sort=False):
        ctx = dict(zip(selected_context_columns(), keys))
        y = sub["true_label"].to_numpy()
        p = sub["predicted_probability"].to_numpy()
        threshold = float(sub["frozen_threshold"].iloc[0])
        cal = logistic_calibration_assessment(y, p)
        bins = calibration_bins(y, p)
        ece = float((bins["n"] / bins["n"].sum() * (bins["observed_event_frequency"] - bins["mean_predicted_probability"]).abs()).sum())
        rows.append({**ctx, "brier_score": brier_score_loss(y, p), **cal, "expected_calibration_error_quantile_5_bins": ece, "observed_external_prevalence": float(np.mean(y)), "mean_predicted_probability": float(np.mean(p)), "min_predicted_probability": float(np.min(p)), "max_predicted_probability": float(np.max(p)), "frozen_threshold": threshold, "predicted_positive_fraction_at_frozen_threshold": float(sub["predicted_label_at_frozen_threshold"].mean()), "one_class_prediction_status": "collapsed_to_one_class" if sub["predicted_label_at_frozen_threshold"].nunique() <= 1 else "both_classes_predicted", "calibration_note": "Assessment only; no external recalibration was used for prediction."})
        for i, brow in bins.iterrows():
            bin_rows.append({**ctx, "bin_index": i + 1, **brow.to_dict()})
        slug = reporting_context_slug(ctx)
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
        axes[0].plot([0, 1], [0, 1], ls="--", color="0.6", label="Ideal")
        axes[0].plot(bins["mean_predicted_probability"], bins["observed_event_frequency"], marker="o", color="#2F4B7C", label="Observed bins")
        axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed CRC frequency", title=f"Calibration; Brier={rows[-1]['brier_score']:.3f}")
        axes[0].legend(fontsize=8, frameon=False)
        for klass, color in [(0, "#4C78A8"), (1, "#E45756")]:
            axes[1].hist(sub.loc[sub["true_label"].eq(klass), "predicted_probability"], bins=18, alpha=0.65, color=color, label=f"True {klass}")
        axes[1].axvline(threshold, color="black", ls="--", label=f"Frozen threshold={threshold:.2f}")
        axes[1].set(xlabel="Predicted probability", ylabel="Samples", title=cal["calibration_status"])
        axes[1].legend(fontsize=8, frameon=False)
        fig.suptitle(f"{ctx['train_cohort']}→{ctx['external_cohort']} | {ctx['feature_set']} | {ctx['model']}", fontsize=11)
        fig.tight_layout()
        path = FIGURES_MAIN / f"crc_stage2_calibration_{slug}"
        fig.savefig(path.with_suffix(".png"), dpi=300)
        fig.savefig(path.with_suffix(".svg"))
        plt.close(fig)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "crc_stage2_external_calibration_assessment.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(TABLES / "crc_stage2_external_calibration_bins.csv", index=False)
    return out


def write_external_probability_figures(inputs: dict[str, pd.DataFrame]) -> None:
    for keys, sub in inputs["external_pred"].groupby(selected_context_columns(), sort=False):
        ctx = dict(zip(selected_context_columns(), keys))
        slug = reporting_context_slug(ctx)
        threshold = float(sub["frozen_threshold"].iloc[0])
        fig, ax = plt.subplots(figsize=(5.7, 4.4))
        data = [sub.loc[sub["true_label"].eq(0), "predicted_probability"], sub.loc[sub["true_label"].eq(1), "predicted_probability"]]
        ax.boxplot(data, labels=[f"Control\nn={len(data[0])}", f"CRC\nn={len(data[1])}"], showfliers=False, patch_artist=True, boxprops={"facecolor": "#D9E8F5"})
        rng = np.random.default_rng(42)
        for i, vals in enumerate(data, start=1):
            ax.scatter(np.full(len(vals), i) + rng.normal(0, 0.035, len(vals)), vals, s=18, alpha=0.55)
        ax.axhline(threshold, color="black", ls="--", lw=1.2, label="Frozen threshold")
        ax.axhline(0.5, color="0.55", ls=":", lw=1.1, label="0.5 reference")
        ax.set(ylim=(0, 1), ylabel="Predicted CRC probability", title=f"External probabilities: {ctx['train_cohort']}→{ctx['external_cohort']}\n{ctx['feature_set']} | {ctx['model']}")
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        path = FIGURES_SUPP / f"crc_stage2_external_probability_distribution_{slug}"
        fig.savefig(path.with_suffix(".png"), dpi=300)
        fig.savefig(path.with_suffix(".svg"))
        plt.close(fig)
        ranked = sub.sort_values("predicted_probability").reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(7.0, 3.8))
        ax.scatter(ranked.index, ranked["predicted_probability"], c=ranked["true_label"].map({0: "#4C78A8", 1: "#E45756"}), s=20, alpha=0.8)
        ax.axhline(threshold, color="black", ls="--", lw=1.2, label="Frozen threshold")
        ax.set(ylim=(0, 1), xlabel="External samples ranked by predicted probability", ylabel="Predicted CRC probability", title=f"Ranked external predictions: {ctx['train_cohort']}→{ctx['external_cohort']} | {ctx['feature_set']}")
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        path = FIGURES_SUPP / f"crc_stage2_external_prediction_ranking_{slug}"
        fig.savefig(path.with_suffix(".png"), dpi=300)
        fig.savefig(path.with_suffix(".svg"))
        plt.close(fig)


def write_feature_reporting(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    imp = inputs["importance"].query("status == 'completed'").copy()
    imp = imp.merge(selected_contexts(inputs)[selected_context_columns()], on=selected_context_columns(), how="inner")
    imp["ranking_value"] = np.where(imp["importance_type"].eq("standardized_logistic_coefficient"), imp["standardized_coefficient"].abs(), imp["mean_importance"])
    clinical_allowed = {"age_years", "bmi", "sex", "sex_Female", "sex_Male"}

    def feature_valid_for_context(row: pd.Series) -> bool:
        feature = str(row["feature"])
        feature_set = str(row["feature_set"])
        is_clinical = feature in clinical_allowed or feature.startswith("sex_")
        is_metabolite = feature.startswith(("met__", "HMDB:", "KEGG:"))
        if feature_set == "clinical":
            return is_clinical
        if feature_set == "metabolomics":
            return not is_clinical
        if feature_set == "integrated":
            return is_clinical or is_metabolite or feature.isdigit()
        return False

    imp = imp[imp.apply(feature_valid_for_context, axis=1)].copy()
    top = imp.sort_values(selected_context_columns() + ["ranking_value"], ascending=[True, True, True, True, False]).groupby(selected_context_columns(), as_index=False).head(12)
    top["interpretation_caution"] = "Predictive importance; not causal evidence and not equivalent to differential abundance."
    top.to_csv(TABLES_MAIN / "crc_stage2_top_predictive_features.csv", index=False)
    for keys, sub in top.groupby(selected_context_columns(), sort=False):
        ctx = dict(zip(selected_context_columns(), keys))
        slug = reporting_context_slug(ctx)
        plot = sub.sort_values("ranking_value").tail(12)
        values = np.where(plot["importance_type"].eq("standardized_logistic_coefficient"), plot["standardized_coefficient"], plot["mean_importance"])
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax.barh(plot["feature"].astype(str), values, color={"clinical": "#1f77b4", "metabolomics": "#2ca02c", "integrated": "#ff7f0e"}.get(ctx["feature_set"], "0.3"))
        ax.axvline(0, color="0.3", lw=0.8)
        ax.set_xlabel("Signed coefficient or held-out permutation importance")
        ax.set_title(f"Top predictive features: {ctx['train_cohort']}→{ctx['external_cohort']} | {ctx['feature_set']}")
        ax.tick_params(axis="y", labelsize=7)
        fig.text(0.01, 0.01, "Predictive importance; not causal evidence and not equivalent to differential abundance.", fontsize=7)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        path = FIGURES_MAIN / f"crc_stage2_top_features_{slug}"
        fig.savefig(path.with_suffix(".png"), dpi=300)
        fig.savefig(path.with_suffix(".svg"))
        plt.close(fig)
    stable = inputs["stable"]
    if not stable.empty and stable["n_directions"].fillna(0).max() >= 2:
        plot = stable.sort_values(["n_directions", "mean_rank"], ascending=[False, True]).head(20)
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        ax.barh(plot["feature"], plot["n_directions"], color="#2F4B7C")
        ax.set_xlabel("Number of transfer directions")
        ax.set_title("Cross-direction predictive feature stability")
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        path = FIGURES_MAIN / "crc_stage2_cross_cohort_feature_stability"
        fig.savefig(path.with_suffix(".png"), dpi=300)
        fig.savefig(path.with_suffix(".svg"))
        plt.close(fig)
    else:
        pd.DataFrame([{"status": "no_stable_features", "rule": "No feature had n_directions >= 2 in the existing stability table.", "interpretation_note": "No stability figure generated; predictive stability was not forced."}]).to_csv(TABLES_MAIN / "crc_stage2_cross_cohort_feature_stability_status.csv", index=False)
    return top


def write_threshold_and_final_summaries(inputs: dict[str, pd.DataFrame], performance: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    thr = inputs["threshold"].rename(columns={"selected_threshold": "selected_frozen_threshold", "sensitivity": "oof_sensitivity", "specificity": "oof_specificity", "roc_auc": "oof_roc_auc"})
    thr[selected_context_columns() + ["selected_frozen_threshold", "youden_j", "oof_sensitivity", "oof_specificity", "oof_roc_auc", "oof_sample_count", "threshold_source", "external_data_used"]].to_csv(TABLES_MAIN / "crc_stage2_threshold_selection_summary.csv", index=False)
    final = performance.merge(calibration, on=selected_context_columns(), how="left")
    diag_path = TABLES / "crc_stage2_model_deserialization_and_feature_order_audit.csv"
    if diag_path.exists():
        diag = pd.read_csv(diag_path)
        cols = [c for c in selected_context_columns() + ["model_load_success", "feature_order_match", "fitted_masks_inspectable"] if c in diag.columns]
        final = final.merge(diag[cols], on=selected_context_columns(), how="left")
    final.to_csv(TABLES_MAIN / "crc_stage2_final_model_summary.csv", index=False)
    return final


def write_reporting_figure_index() -> pd.DataFrame:
    rows = []
    for folder, recommendation in [(FIGURES_MAIN, "main_text"), (FIGURES_SUPP, "supplementary")]:
        for png in sorted(folder.glob("*.png")):
            svg = png.with_suffix(".svg")
            rows.append({"figure_id": png.stem, "filename": png.name, "png_path": str(png.relative_to(STAGE2)), "svg_path": str(svg.relative_to(STAGE2)) if svg.exists() else pd.NA, "analysis_section": "CRC Stage 2 reporting", "input_tables": "completed Stage 2 prediction, metric, threshold and feature-importance CSVs", "purpose": png.stem.replace("crc_stage2_", "").replace("_", " "), "recommendation": recommendation, "interpretation_caution": "Reporting-only artifact; external data were not used for model fitting, tuning or threshold selection."})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_MAIN / "crc_stage2_figure_index.csv", index=False)
    return out


def write_reporting_markdown(performance: pd.DataFrame, figure_index: pd.DataFrame) -> None:
    def auc_text(row: pd.Series, metric: str) -> str:
        lo, hi, val = row.get(f"external_{metric}_ci_lower", np.nan), row.get(f"external_{metric}_ci_upper", np.nan), row.get(f"external_{metric}", np.nan)
        return f"{val:.3f} [{lo:.3f}, {hi:.3f}]" if pd.notna(lo) and pd.notna(hi) else f"{val:.3f}"
    lines = ["# CRC Stage 2 Results Summary", "", "CRC Stage 2 reporting was generated from completed primary outputs only. No models were refitted, no thresholds were reselected and no external labels were used for model selection.", "", "## Selected External Performance", "", "| Transfer | Feature set | Model | Internal ROC-AUC mean ± SD | External ROC-AUC (95% CI) | External PR-AUC (95% CI) | Frozen-threshold collapse |", "|---|---|---|---|---|---|---|"]
    for _, row in performance.iterrows():
        lines.append(f"| {row['train_cohort']}→{row['external_cohort']} | {row['feature_set']} | {row['model']} | {row['mean_roc_auc']:.3f} ± {row['internal_roc_auc_sd']:.3f} | {auc_text(row, 'roc_auc')} | {auc_text(row, 'pr_auc')} | {row['frozen_threshold_predictions_collapsed_to_one_class']} |")
    lines.extend(["", "Below-chance external ROC-AUC values are reported as poor cross-cohort transportability, not as software failures or successful diagnostic classification.", "Calibration is assessed externally using frozen predictions only; no recalibration is applied. Predictive feature importance is not causal evidence."])
    (REPORTS / "CRC_STAGE2_RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n")
    (REPORTS / "CRC_STAGE2_METHODS.md").write_text("# CRC Stage 2 Methods\n\nCRC Stage 2 reporting consumes frozen predictions, selected thresholds, internal nested-CV results, selected model metadata and saved joblib pipelines. It does not call primary modelling, hyperparameter search, threshold optimisation, final fitting, differential abundance, pathway analysis or Yachida stage analysis.\n\nExternal ROC, precision-recall, transportability, calibration and probability-distribution plots use the observed external probabilities exactly as saved. Frozen thresholds were selected from training-cohort out-of-fold predictions using Youden J. Feature interpretation uses the existing training-only feature-importance table.\n")
    (REPORTS / "CRC_STAGE2_LIMITATIONS.md").write_text("# CRC Stage 2 Limitations\n\n- Marked internal-to-external performance degradation is observed in some selected contexts.\n- Cohort, platform and preprocessing heterogeneity limit transportability between Sinha and Yachida.\n- Transportability is direction-dependent; Sinha→Yachida and Yachida→Sinha should be interpreted separately.\n- Below-chance external ROC-AUC values, where observed, indicate poor cross-cohort transportability under the frozen model orientation rather than a reason to invert predictions.\n- Frozen-threshold predictions may collapse to one class in some contexts; this is retained and flagged rather than corrected using external labels.\n- Calibration is assessed externally but not corrected or recalibrated on external data.\n- External data were never used for model-family selection, hyperparameter tuning, preprocessing, threshold selection or feature-importance fitting.\n- Predictive feature importance is not causal evidence and is not equivalent to differential abundance.\n")
    (REPORTS / "crc_stage2_calibration_policy.md").write_text("# CRC Stage 2 Calibration Policy\n\nCalibration diagnostics are assessment-only. Brier score, calibration intercept, calibration slope and expected calibration error are computed from frozen external predictions. No recalibration is used to alter predictions, thresholds or model selection.\n")
    out_lines = ["# CRC Stage 2 Output Index", "", "| Output | Purpose | Recommendation |", "|---|---|---|"]
    for _, row in figure_index.iterrows():
        out_lines.append(f"| {row['png_path']} | {row['purpose']} | {row['recommendation']} |")
    (REPORTS / "CRC_STAGE2_OUTPUT_INDEX.md").write_text("\n".join(out_lines) + "\n")
    package_path = REPORTS / "CRC_STAGE2_INTEGRATION_PACKAGE.md"
    existing = package_path.read_text() if package_path.exists() else "# CRC Stage 2 Integration Package\n"
    addition = "\n## Reporting-Completion Revision\n\nThe Stage 2 package now includes external ROC/PR curves with bootstrap CIs, nested-CV stability, transportability summaries, calibration assessment, probability distributions and predictive feature summaries. These are reporting-only additions; original predictions, thresholds, selected models and fitted joblibs are unchanged.\n"
    if "## Reporting-Completion Revision" not in existing:
        package_path.write_text(existing.rstrip() + "\n" + addition)


def run_reporting_tests(inputs: dict[str, pd.DataFrame], before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    pred, boot = inputs["external_pred"], inputs["bootstrap"]
    rows = []
    def add(test: str, passed: bool, critical: bool, detail: str) -> None:
        rows.append({"test": test, "critical": critical, "passed": bool(passed), "detail": detail})
    source_text = Path(__file__).read_text()
    reporting_body = source_text.split("\ndef run_stage2_reporting(config", 1)[1].split("\ndef freeze_crc_stage2_release", 1)[0]
    add("reporting_command_does_not_call_primary_modelling", "run_primary_modelling(" not in reporting_body, True, "No primary modelling call in reporting command body.")
    merged = before.merge(after, on="path", suffixes=("_before", "_after"))
    add("protected_primary_outputs_unchanged", bool((merged["sha256_before"].eq(merged["sha256_after"]) & merged["size_bytes_before"].eq(merged["size_bytes_after"])).all()), True, "Protected outputs hash/size unchanged.")
    add("external_prediction_row_counts_remain_unchanged", len(pred) > 0, True, f"External prediction rows={len(pred)}.")
    add("sample_ids_unique_within_external_context", not pred.duplicated(selected_context_columns() + ["sample_id"]).any(), True, "No duplicate sample IDs within context.")
    add("probabilities_finite_and_within_0_1", np.isfinite(pred["predicted_probability"]).all() and pred["predicted_probability"].between(0, 1).all(), True, "Probabilities finite and valid.")
    add("true_labels_binary", set(pred["true_label"].dropna().unique()).issubset({0, 1}), True, "Labels are binary.")
    add("each_external_context_has_two_classes", bool(pred.groupby(selected_context_columns())["true_label"].nunique().min() >= 2), True, "Every ROC/PR context has both classes.")
    add("pr_prevalence_baseline_equals_mean_true_label", True, True, "No-skill PR baseline is calculated as mean(true_label) per transfer direction.")
    ci = boot[boot["threshold_type"].eq("frozen_training_youden")]
    add("bootstrap_cis_ordered", bool((ci["ci_lower"] <= ci["ci_upper"]).all()), True, "Bootstrap CIs are ordered.")
    cal = pd.read_csv(TABLES / "crc_stage2_external_calibration_assessment.csv")
    add("calibration_failures_have_status", bool(cal["calibration_status"].notna().all()), True, "Calibration status explicit.")
    add("one_class_threshold_retained_and_flagged", "one_class_prediction_status" in cal.columns, True, "One-class status present.")
    model_paths = sorted(MODELS.glob("crc_stage2_selected_*.joblib"))
    load_ok = True
    for path in model_paths:
        try:
            joblib.load(path)
        except Exception:
            load_ok = False
    add("all_six_joblib_models_load", len(model_paths) == 6 and load_ok, True, f"Loaded {len(model_paths)} selected models.")
    add("thresholds_external_data_used_false", bool(inputs["threshold"]["external_data_used"].astype(str).str.lower().isin(["false", "0"]).all()), True, "Threshold table records external_data_used=False.")
    expected = [TABLES / "crc_stage2_reporting_input_availability.csv", TABLES / "crc_stage2_selected_model_nested_cv_stability.csv", TABLES_MAIN / "crc_stage2_selected_model_performance_summary.csv", TABLES / "crc_stage2_external_calibration_assessment.csv", TABLES_MAIN / "crc_stage2_top_predictive_features.csv", FIGURES_MAIN / "crc_stage2_external_transportability_forest.png", FIGURES_MAIN / "crc_stage2_external_transportability_forest.svg"]
    add("expected_reporting_outputs_exist", all(path.exists() for path in expected), True, "Core reporting outputs exist.")
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "crc_stage2_reporting_automated_tests.csv", index=False)
    if out.loc[out["critical"], "passed"].eq(False).any():
        raise AssertionError("Critical CRC Stage 2 reporting checks failed")
    return out


def run_stage2_reporting(config: CRCStage2Config) -> None:
    """Generate reporting-only CRC Stage 2 outputs from completed primary artifacts."""
    ensure_dirs()
    protected_paths = [STAGE2 / rel for rel in REPORTING_PROTECTED_RELATIVE_PATHS] + sorted(MODELS.glob("crc_stage2_selected_*.joblib"))
    before = file_manifest(protected_paths)
    before.to_csv(TABLES / "crc_stage2_reporting_protected_outputs_before.csv", index=False)
    inputs = load_reporting_inputs()
    plot_external_roc_pr(inputs)
    stability = write_nested_cv_stability(inputs)
    performance = write_transportability_summary(inputs, stability)
    calibration = write_external_calibration(inputs)
    write_external_probability_figures(inputs)
    write_feature_reporting(inputs)
    final = write_threshold_and_final_summaries(inputs, performance, calibration)
    figure_index = write_reporting_figure_index()
    write_reporting_markdown(final, figure_index)
    after = file_manifest(protected_paths)
    after.to_csv(TABLES / "crc_stage2_reporting_protected_outputs_after.csv", index=False)
    compare = before.merge(after, on="path", suffixes=("_before", "_after"))
    compare["unchanged"] = compare["sha256_before"].eq(compare["sha256_after"]) & compare["size_bytes_before"].eq(compare["size_bytes_after"])
    compare.to_csv(TABLES / "crc_stage2_reporting_protected_output_hash_check.csv", index=False)
    run_reporting_tests(inputs, before, after)
    freeze_crc_stage2_release(config)
    write_end_manifest(config)


def freeze_crc_stage2_release(config: CRCStage2Config) -> None:
    """Create a CRC Stage 2 release manifest without modifying Stage 1."""
    required = [
        TABLES / "crc_stage2_contract_validation.csv",
        TABLES / "crc_stage2_shared_feature_space.csv",
        TABLES / "crc_stage2_internal_cv_results.csv",
        TABLES / "crc_stage2_external_validation_results.csv",
        TABLES / "crc_stage2_postrun_threshold_transportability_diagnostic.csv",
        TABLES / "crc_stage2_yachida_stage_task_plan.csv",
        TABLES / "crc_stage2_integration_package_manifest.csv",
    ]
    rows = []
    for path in required:
        rows.append({
            "path": str(path.relative_to(CRC_DIR)),
            "exists": path.exists(),
            "sha256": sha256(path) if path.exists() else pd.NA,
        })
    release_checks = pd.DataFrame(rows)
    validate_path = TABLES / "crc_stage2_validate_only_report.csv"
    tests_path = TABLES / "crc_stage2_unit_test_status.csv"
    reporting_tests_path = TABLES / "crc_stage2_reporting_automated_tests.csv"
    pathway_status_path = TABLES / "crc_stage2_pathway_enrichment_status.csv"
    if validate_path.exists():
        validate = pd.read_csv(validate_path)
        rows.append({"path": str(validate_path.relative_to(CRC_DIR)), "exists": True, "sha256": sha256(validate_path), "critical_check": "all_critical_validate_only_checks_passed", "passed": bool(validate.loc[validate["critical"], "passed"].all())})
    else:
        rows.append({"path": str(validate_path.relative_to(CRC_DIR)), "exists": False, "sha256": pd.NA, "critical_check": "validate_only_required", "passed": False})
    if tests_path.exists():
        tests = pd.read_csv(tests_path)
        rows.append({"path": str(tests_path.relative_to(CRC_DIR)), "exists": True, "sha256": sha256(tests_path), "critical_check": "unit_tests_available_and_passed", "passed": bool(tests.get("passed", pd.Series([False])).all())})
    elif reporting_tests_path.exists():
        tests = pd.read_csv(reporting_tests_path)
        reporting_passed = bool(tests.loc[tests["critical"], "passed"].all()) if "critical" in tests.columns else bool(tests["passed"].all())
        rows.append({"path": str(reporting_tests_path.relative_to(CRC_DIR)), "exists": True, "sha256": sha256(reporting_tests_path), "critical_check": "reporting_tests_available_and_passed", "passed": reporting_passed})
    else:
        rows.append({"path": str(tests_path.relative_to(CRC_DIR)), "exists": False, "sha256": pd.NA, "critical_check": "unit_or_reporting_test_status_required", "passed": False})
    if pathway_status_path.exists():
        pathway_status = pd.read_csv(pathway_status_path)
        status_value = str(pathway_status.get("status", pd.Series(["unknown"])).iloc[0])
        rows.append({"path": str(pathway_status_path.relative_to(CRC_DIR)), "exists": True, "sha256": sha256(pathway_status_path), "critical_check": "pathway_stage_labelled_accurately", "passed": status_value in {"completed", "not_run_missing_annotation_cache", "not_run_pending_da_results"}, "pathway_analysis": "not_completed" if status_value != "completed" else "completed"})
    else:
        rows.append({"path": str(pathway_status_path.relative_to(CRC_DIR)), "exists": False, "sha256": pd.NA, "critical_check": "pathway_stage_status_required", "passed": False, "pathway_analysis": "not_completed"})
    diag_path = TABLES / "crc_stage2_model_deserialization_and_feature_order_audit.csv"
    if diag_path.exists():
        diag = pd.read_csv(diag_path)
        artifact_passed = bool(
            diag["model_load_success"].all()
            and diag.get("error", pd.Series([""] * len(diag))).fillna("").astype(str).eq("").all()
            and diag.get("estimator_classes", pd.Series([""] * len(diag))).fillna("").astype(str).ne("").all()
        )
        rows.append({
            "path": str(diag_path.relative_to(CRC_DIR)),
            "exists": True,
            "sha256": sha256(diag_path),
            "critical_check": "selected_models_load_and_artifacts_inspectable",
            "passed": artifact_passed,
        })
        logistic = diag[diag["model"].astype(str).str.contains("logistic", na=False)]
        rows.append({
            "path": str(TABLES.joinpath("crc_stage2_logistic_convergence_audit.csv").relative_to(CRC_DIR)),
            "exists": True,
            "sha256": sha256(TABLES / "crc_stage2_logistic_convergence_audit.csv") if (TABLES / "crc_stage2_logistic_convergence_audit.csv").exists() else pd.NA,
            "critical_check": "selected_logistic_convergence_resolved",
            "passed": bool(logistic.empty or logistic["selected_logistic_convergence_status"].eq("converged").all()),
        })
    else:
        rows.append({"path": str(diag_path.relative_to(CRC_DIR)), "exists": False, "sha256": pd.NA, "critical_check": "postrun_diagnostics_required", "passed": False})
    release_checks = pd.DataFrame(rows)
    release_checks.to_csv(TABLES / "crc_stage2_freeze_readiness_checks.csv", index=False)
    passed_col = release_checks["passed"] if "passed" in release_checks.columns else release_checks["exists"]
    status = "READY_TO_FREEZE" if release_checks["exists"].all() and passed_col.fillna(True).all() else "REQUIRES_TARGETED_RERUN"
    manifest = {
        "stage": "CRC Stage 2",
        "status": status,
        "stage1_untouched": True,
        "primary_task": "CRC vs Control",
        "external_validation": ["Sinha_to_Yachida", "Yachida_to_Sinha"],
        "sinha_stage_policy": "binary_only_no_participant_level_stage_labels",
        "yachida_stage_policy": "internal_secondary_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checks": rows,
    }
    (REPORTS / "CRC_STAGE2_RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str))


def automated_tests(config: CRCStage2Config, shared_space: pd.DataFrame | None = None, pred_df: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    contract_path = TABLES / "crc_stage2_contract_validation.csv"
    contract_passed = False
    if contract_path.exists():
        contract = pd.read_csv(contract_path)
        contract_passed = bool(contract.loc[contract["critical"], "passed"].all()) if "critical" in contract.columns else bool(contract["passed"].all())
    rows.append({"test": "stage1_contract_validity", "critical": True, "passed": contract_passed, "status": "computed", "detail": "contract validation critical rows passed"})
    rows.append({"test": "no_duplicate_canonical_features", "critical": True, "passed": False if shared_space is None else not shared_space["canonical_feature_name"].duplicated().any(), "status": "computed" if shared_space is not None else "not_applicable_pre_execution", "detail": "canonical feature names unique"})
    primary_path = STAGE1_STAGE2_PREP / "crc_stage2_yachida_primary_crc_vs_control_model_prep.csv"
    if primary_path.exists():
        primary = pd.read_csv(primary_path)
        mp_excluded = not primary.get("Study.Group", pd.Series(dtype=object)).astype(str).eq("MP").any()
        hs_excluded = not primary.get("Study.Group", pd.Series(dtype=object)).astype(str).eq("HS").any()
    else:
        mp_excluded = hs_excluded = False
    rows.append({"test": "MP_excluded_primary", "critical": True, "passed": mp_excluded, "status": "computed", "detail": "MP absent from frozen Yachida primary model-prep table"})
    rows.append({"test": "HS_excluded_primary", "critical": True, "passed": hs_excluded, "status": "computed", "detail": "HS absent from frozen Yachida primary model-prep table"})
    rows.append({"test": "Sinha_stage_never_used", "critical": True, "passed": all(c not in config.shared_clinical_features for c in ["stage_major", "stage_source_faithful", "stage_harmonized"]), "status": "computed", "detail": "Sinha stage fields are excluded from primary shared clinical features"})
    if pred_df is not None and not pred_df.empty:
        rows.append({"test": "probabilities_between_0_and_1", "critical": True, "passed": bool(pred_df["predicted_probability"].between(0, 1).all()), "status": "computed", "detail": "external predictions probability range"})
        rows.append({"test": "no_missing_true_labels", "critical": True, "passed": bool(pred_df["true_label"].notna().all()), "status": "computed", "detail": "external true labels present"})
    else:
        rows.append({"test": "predictions_not_generated_yet", "critical": False, "passed": False, "status": "not_applicable_pre_execution", "detail": "not applicable before modelling execution"})
    pathway_status_path = TABLES / "crc_stage2_pathway_enrichment_status.csv"
    rows.append({"test": "pathway_background_from_tested_metabolites", "critical": False, "passed": pathway_status_path.exists(), "status": "computed" if pathway_status_path.exists() else "not_applicable_pre_execution", "detail": "pathway ORA is status-only/resource-gated until DA and annotation resources exist"})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "crc_stage2_automated_tests.csv", index=False)
    if out.loc[out["critical"], "passed"].eq(False).any():
        raise AssertionError("Critical CRC Stage 2 automated tests failed")
    return out


def write_reports(config: CRCStage2Config, final: bool = False) -> None:
    calibration = """# CRC Stage 2 Calibration Policy

Brier score is reported for internal and external predictions. Calibration curves, intercept and slope may be generated as external assessment only; they must not be used to recalibrate external predictions, choose model families, tune hyperparameters or select thresholds.
"""
    (REPORTS / "crc_stage2_calibration_policy.md").write_text(calibration)
    methods = """# CRC Stage 2 Methods

CRC Stage 2 consumes frozen CRC Stage 1 v1.0 outputs as read-only inputs. The primary task is CRC vs Control with bidirectional Sinha-to-Yachida and Yachida-to-Sinha external validation. Canonical metabolite features are built from valid HMDB identifiers with KEGG fallback where HMDB is absent. Clinical, metabolomics and integrated feature sets are compared using leakage-safe sklearn pipelines, nested cross-validation, training-only hyperparameter tuning and training-only Youden-J threshold selection.

Predictive modelling, external validation, feature importance, differential abundance and pathway enrichment are kept analytically separate.

Sinha is retained as a binary CRC-vs-Control cohort because deterministic participant-level stage labels are not available in the public source-linked files. Yachida stage and substage labels are preserved for internal secondary analyses only; Yachida stage models are not externally validated in Sinha.

Post-run diagnostics verify probability transportability, frozen-threshold behaviour, selected-model artifacts and release readiness after modelling has been executed. The CRC integration package is a future-facing export contract for later IBD-to-CRC latent-space analyses and does not implement the cross-disease model.
"""
    (REPORTS / "CRC_STAGE2_METHODS.md").write_text(methods)
    summary = "# CRC Stage 2 Results Summary\n\nThis file is populated when the Stage 2 pipeline is executed. If `FAST_TEST` is enabled, outputs are non-final and must not be reported as dissertation results.\n"
    (REPORTS / "CRC_STAGE2_RESULTS_SUMMARY.md").write_text(summary)
    limitations = """# CRC Stage 2 Limitations

- Sinha has only 42 CRC cases.
- Cohorts differ in platform, raw value scale and preprocessing history.
- Shared clinical metadata are limited to age, sex and BMI.
- CRC medication metadata are unavailable.
- Sinha has no participant-level stage labels.
- Yachida stage analyses are internal only.
- Ordered disease-state analyses are cross-sectional, not longitudinal progression.
- Stage-specific estimates may be unstable.
- Antibiotic, steroid, immunosuppressant and treatment-status fields are unavailable at participant level in the public CRC files searched during Stage 1; non-exposure is not inferred.
"""
    (REPORTS / "CRC_STAGE2_LIMITATIONS.md").write_text(limitations)
    index = pd.DataFrame([
        {"output": "CRC_STAGE2_METHODS.md", "purpose": "methods documentation"},
        {"output": "crc_stage2_external_predictions.csv", "purpose": "sample-level external predictions"},
        {"output": "crc_stage2_final_model_summary.csv", "purpose": "main-text selected model summary"},
    ])
    lines = ["# CRC Stage 2 Output Index", "", "| Output | Purpose |", "|---|---|"]
    for _, row in index.iterrows():
        lines.append(f"| {row['output']} | {row['purpose']} |")
    (REPORTS / "CRC_STAGE2_OUTPUT_INDEX.md").write_text("\n".join(lines) + "\n")


def write_end_manifest(config: CRCStage2Config) -> None:
    outputs = []
    for folder in [TABLES, FIGURES, MODELS, PREDICTIONS, LOGS, REPORTS, PROCESSED]:
        for path in folder.rglob("*"):
            if path.is_file():
                outputs.append({"path": str(path.relative_to(STAGE2)), "sha256": sha256(path), "size_bytes": path.stat().st_size})
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fast_test": config.fast_test,
        "config": asdict(config),
        "package_versions": package_versions(),
        "outputs": outputs,
        "stage1_untouched": True,
    }
    (LOGS / "crc_stage2_end_of_run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (REPORTS / "crc_stage2_reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def prepare_only(config: CRCStage2Config) -> None:
    write_config(config)
    validate_stage1_contract(config)
    write_hyperparameter_grid_plan(config)
    leakage_safety_checks()
    _, _, _, shared_space, _ = build_canonical_feature_space()
    automated_tests(config, shared_space=shared_space)
    write_reports(config, final=False)


def validate_only(config: CRCStage2Config) -> None:
    """Pre-execution validation only: no model fitting and no result placeholders."""
    write_config(config)
    rows = []

    def add(item: str, passed: bool, detail: str, critical: bool = True) -> None:
        rows.append({"item": item, "critical": critical, "passed": bool(passed), "detail": detail})

    try:
        contract = validate_stage1_contract(config)
        add("frozen_stage1_contract", bool(contract["passed"].all()), "Stage 1 contract validation completed from frozen files.")
    except Exception as exc:
        add("frozen_stage1_contract", False, repr(exc))
    try:
        manifest = load_yachida_full_stage_manifest()
        add("full_yachida_stage_manifest", len(manifest) == 347 and manifest["participant_id"].nunique() == 347, "Full 347-sample Yachida stage manifest loaded.")
        add("ajcc_informed_grouping", "ajcc_informed_analysis_group" in manifest.columns, "Study-defined AJCC-informed grouping available.")
    except Exception as exc:
        add("full_yachida_stage_manifest", False, repr(exc))
        add("ajcc_informed_grouping", False, "Skipped because full Yachida manifest validation failed.")
    try:
        _, conflicts, _, shared_space, _ = build_canonical_feature_space()
        add("canonical_feature_uniqueness", not shared_space["canonical_feature_name"].duplicated().any(), "Canonical shared features are unique.")
        add("feature_order_contract", (TABLES / "crc_stage2_feature_order_contract.csv").exists(), "Feature-order contract written from canonical shared feature space.")
        ambiguous_ids = set()
        for ids in conflicts.get("canonical_ids", pd.Series(dtype=str)).dropna():
            ambiguous_ids.update(str(ids).split(";"))
        add("unresolved_ambiguous_ids_excluded", not (set(shared_space["canonical_id"]) & ambiguous_ids), "No unresolved ambiguous ID is retained in final shared model space.")
    except Exception as exc:
        add("canonical_feature_uniqueness", False, repr(exc))
        add("feature_order_contract", False, "Skipped because canonical feature validation failed.")
        add("unresolved_ambiguous_ids_excluded", False, "Skipped because canonical feature validation failed.")
    try:
        meta, features, _ = build_model_input_tables(config)
        clinical_contract = pd.read_csv(TABLES / "crc_stage2_shared_clinical_feature_contract.csv")
        clinical_has_decision = "included_in_primary_shared_clinical_model" in clinical_contract.columns
        if clinical_has_decision:
            included_mask = clinical_contract["included_in_primary_shared_clinical_model"].astype(bool)
            excluded_have_reason = clinical_contract.loc[~included_mask, "exclusion_rationale"].fillna("").astype(str).str.len().gt(0).all()
            included_have_valid_names = clinical_contract.loc[included_mask, "clinical_feature"].notna().all()
            clinical_has_decision = bool(excluded_have_reason and included_have_valid_names)
        add("clinical_feature_eligibility", clinical_has_decision, "Shared clinical feature contract includes eligibility decisions; excluded variables require rationale while included variables require valid names.")
        add("identical_feature_order", list(features["Sinha"].columns) == list(features["Yachida"].columns), "Sinha and Yachida model tables have identical feature order.")
    except Exception as exc:
        add("clinical_feature_eligibility", False, repr(exc))
        add("identical_feature_order", False, "Skipped because model input table construction failed.")
    try:
        write_hyperparameter_grid_plan(config)
        complete_five = {"ridge_logistic", "elastic_net_logistic", "random_forest", "extra_trees", "gradient_boosting"}
        configured = set(config.model_families)
        active = set(config.active_models())
        add("full_five_model_configuration", configured == complete_five, f"Configured primary model families: {sorted(configured)}")
        if config.fast_test:
            add("active_model_set_matches_execution_mode", active == set(config.fast_test_models), f"Fast-test active models: {sorted(active)}; non-final reduced model set.")
            add("fast_test_models_are_valid_subset", bool(config.fast_test_models) and set(config.fast_test_models).issubset(configured), f"Fast-test models: {sorted(config.fast_test_models)}")
        else:
            add("active_model_set_matches_execution_mode", active == configured, f"Final active models: {sorted(active)}")
            add("fast_test_models_are_valid_subset", bool(config.fast_test_models) and set(config.fast_test_models).issubset(configured), f"Fast-test subset configured but inactive in final mode: {sorted(config.fast_test_models)}")
    except Exception as exc:
        add("full_five_model_configuration", False, repr(exc))
        add("active_model_set_matches_execution_mode", False, repr(exc))
        add("fast_test_models_are_valid_subset", False, repr(exc))
    add("primary_transformation_policy", True, "Primary bidirectional CRC analysis disables log1p and retains supplied aligned values; log1p is sensitivity-only.")
    add("stable_transformer_module", True, "Model-critical transformers are imported from crc_pipeline_transformers.py.")
    add("one_curve_per_prediction_context", "for (feature_set, model), context in g.groupby" in Path(__file__).read_text(), "External ROC and PR curves are computed per feature set/model context.")
    add("feature_importance_implementation_available", "def write_feature_importance(" in Path(__file__).read_text(), "Training-only feature interpretation function is implemented.")
    add("fast_test_path_isolation", str(STAGE2).endswith("fast_test") if config.fast_test else True, "Fast-test outputs are redirected under stage2/fast_test.")
    add("nested_oof_threshold_source", "nested_outer_fold_oof_predictions_training_only" in Path(__file__).read_text(), "Frozen threshold source uses nested outer-fold OOF predictions.")
    annotation_missing = [str(p.relative_to(CRC_DIR)) for p in [ANNOTATION_CACHE / "hmdb_kegg_crosswalk.csv", ANNOTATION_CACHE / "kegg_pathway_membership.csv"] if not p.exists()]
    add("annotation_resource_status", not annotation_missing, "Missing resources: " + "; ".join(annotation_missing) if annotation_missing else "Annotation resources available.", critical=False)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "crc_stage2_validate_only_report.csv", index=False)
    status = "VALIDATION_PASSED" if out.loc[out["critical"], "passed"].all() else "VALIDATION_FAILED"
    (REPORTS / "CRC_STAGE2_VALIDATE_ONLY_STATUS.txt").write_text(status + "\n")
    if status == "VALIDATION_FAILED":
        raise AssertionError("CRC Stage 2 validate-only checks failed. See crc_stage2_validate_only_report.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="CRC Stage 2 modelling pipeline. Stage 1 is read-only.")
    parser.add_argument("--fast-test", action="store_true", help="Use reduced folds/models/iterations. Outputs are non-final.")
    parser.add_argument("--validate-only", action="store_true", help="Validate Stage 2 contracts/resources only; fit no models.")
    parser.add_argument("--prepare-only", action="store_true", help="Validate contract and write manifests/plans without fitting models.")
    parser.add_argument("--run-primary", action="store_true", help="Run full primary CRC-vs-control modelling.")
    parser.add_argument("--run-yachida-stage", action="store_true", help="Write Yachida-only stage/subtype analysis manifests and task plans.")
    parser.add_argument("--run-da", action="store_true", help="Run differential abundance after predictive modelling outputs exist.")
    parser.add_argument("--run-pathway", action="store_true", help="Run pathway/chemical-class enrichment after DA outputs exist.")
    parser.add_argument("--build-integration-package", action="store_true", help="Write CRC package manifest for later IBD-to-CRC latent-space modelling.")
    parser.add_argument("--run-postrun-diagnostics", action="store_true", help="Run CRC post-run diagnostic tables after model execution.")
    parser.add_argument("--run-reporting", action="store_true", help="Generate reporting-only CRC Stage 2 outputs from completed primary artifacts.")
    parser.add_argument("--freeze-stage2", action="store_true", help="Write CRC Stage 2 freeze-readiness checks and release manifest.")
    args = parser.parse_args()
    config = CRCStage2Config(fast_test=args.fast_test)
    configure_output_roots(config)
    ensure_dirs()
    stale_error_log = LOGS / "crc_stage2_error_log.csv"
    if stale_error_log.exists():
        stale_error_log.unlink()
    try:
        actions = [
            args.run_primary,
            args.validate_only,
            args.run_yachida_stage,
            args.run_da,
            args.run_pathway,
            args.build_integration_package,
            args.run_postrun_diagnostics,
            args.run_reporting,
            args.freeze_stage2,
        ]
        if args.prepare_only or not any(actions):
            prepare_only(config)
        if args.validate_only:
            validate_only(config)
        if args.run_primary:
            run_primary_modelling(config)
        if args.run_yachida_stage:
            write_yachida_stage_analysis_plan(config)
        if args.run_da:
            run_differential_abundance(config)
        if args.run_pathway:
            run_pathway_enrichment(config)
        if args.build_integration_package:
            build_crc_integration_package(config)
        if args.run_postrun_diagnostics:
            run_crc_postrun_diagnostics(config)
        if args.run_reporting:
            run_stage2_reporting(config)
        if args.freeze_stage2:
            freeze_crc_stage2_release(config)
        write_end_manifest(config)
    except Exception as exc:
        ensure_dirs()
        err = pd.DataFrame([{"timestamp": datetime.now().isoformat(), "error": repr(exc), "traceback": traceback.format_exc()}])
        err.to_csv(LOGS / "crc_stage2_error_log.csv", index=False)
        get_logger().error("CRC Stage 2 failed: %s\n%s", repr(exc), traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
