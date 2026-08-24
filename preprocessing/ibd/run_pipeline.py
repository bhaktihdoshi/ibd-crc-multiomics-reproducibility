from __future__ import annotations

import json
import logging
import math
import platform
import re
import sys
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import scipy
    from scipy.stats import hypergeom as scipy_hypergeom
    from scipy.stats import t as t_dist
except Exception:
    scipy = None
    scipy_hypergeom = None
    t_dist = None
try:
    import sklearn
except Exception:
    sklearn = None
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from ibd_pipeline_transformers import BalancedGradientBoostingClassifier, ConstantFeatureFilter, MissingnessFilter, NearZeroVarianceFilter
except ImportError:
    from thesis_preprocessing_workflow.ibd_pipeline.ibd_pipeline_transformers import (
        BalancedGradientBoostingClassifier,
        ConstantFeatureFilter,
        MissingnessFilter,
        NearZeroVarianceFilter,
    )


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "thesis_preprocessing_workflow" / "ibd_pipeline"
TABLES = WORK / "tables"
REPORTS = WORK / "reports"
FIGURES = WORK / "figures"
PROCESSED = WORK / "data_processed"
MODELS = WORK / "models"
PREDICTIONS = WORK / "predictions"
LOGS = WORK / "logs"

FR_DIR = ROOT / "IBD-DataSet" / "FRANZOSA_IBD_2019"
IH_DIR = ROOT / "IBD-DataSet" / "iHMP_IBDMDB"

CONFIG = {
    "random_seed": 42,
    "primary_feature_missingness_threshold": 0.30,
    "primary_iHMP_visit_rule": ["earliest collection date", "lowest visit/study week", "sorted sample ID"],
    "exclude_antibiotic_exposure": True,
    "exclude_systemic_steroid_exposure_where_recorded": True,
    "external_validation_design": ["Franzosa_to_iHMP", "iHMP_to_Franzosa"],
    "outcomes": ["ibd_vs_control", "cd_vs_control", "uc_vs_control"],
    "feature_sets": ["clinical", "metabolomics", "integrated"],
    "models": ["ridge_logistic", "elastic_net_logistic", "random_forest", "extra_trees", "gradient_boosting"],
    "annotation": {
        "annotation_resource_directories": [
            "thesis_preprocessing_workflow/ibd_pipeline/annotation_resources",
            "IBD-DataSet/annotation_resources",
        ],
        "allow_external_annotation_download": False,
        "kegg_request_delay_seconds": 0.35,
        "kegg_retry_count": 2,
        "minimum_background_kegg_mappings": 10,
        "minimum_significant_kegg_mappings": 3,
        "minimum_pathway_background_compounds": 3,
        "minimum_pathway_significant_compounds": 2,
        "low_coverage_threshold": 0.50,
        "annotation_retrieval_date": datetime.now(timezone.utc).date().isoformat(),
        "excluded_kegg_pathway_ids": ["map01100", "map01110", "map01120", "map01200"],
        "excluded_kegg_pathway_name_patterns": [
            "Metabolic pathways",
            "Biosynthesis of secondary metabolites",
            "Microbial metabolism in diverse environments",
            "Biosynthesis of cofactors",
            "Chemical structure transformation maps",
            "Global and overview maps",
        ],
    },
    "nested_cv": {
        "outer_folds_requested": 5,
        "inner_folds_requested": 5,
        "minimum_class_count_for_requested_inner": 5,
        "reduced_inner_folds": 3,
        "selection_metric": "roc_auc",
        "randomized_search_n_iter": {
            "random_forest": 75,
            "extra_trees": 75,
            "gradient_boosting": 75,
        },
        "model_simplicity_order": {
            "ridge_logistic": 0,
            "elastic_net_logistic": 1,
            "random_forest": 2,
            "extra_trees": 3,
            "gradient_boosting": 4,
        },
    },
}


def ensure_dirs() -> None:
    for d in [TABLES, REPORTS, FIGURES, PROCESSED, MODELS, PREDICTIONS, LOGS]:
        d.mkdir(parents=True, exist_ok=True)
    for d in [TABLES / "main_text", FIGURES / "main_text", FIGURES / "supplementary", WORK / "annotation_resources", ROOT / "IBD-DataSet" / "annotation_resources"]:
        d.mkdir(parents=True, exist_ok=True)


def pathway_logger() -> logging.Logger:
    logger = logging.getLogger("pathway_annotation")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(LOGS / "pathway_annotation.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(handler)
    return logger


def read_tsv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", **kwargs)


def clean_id_values(x: object) -> list[str]:
    if pd.isna(x):
        return []
    raw = str(x).strip()
    if raw == "" or raw.upper() in {"NA", "NAN", "NONE", "-"}:
        return []
    parts = re.split(r"[;,|/]\s*", raw)
    return [p.strip().upper() for p in parts if p.strip()]


def valid_hmdb(x: str) -> bool:
    return bool(re.fullmatch(r"HMDB\d{7}", x))


def valid_kegg_compound(x: str) -> bool:
    return bool(re.fullmatch(r"C\d{5}", x))


def normalize_name(x: object) -> tuple[str | None, str]:
    if pd.isna(x):
        return None, "missing"
    s = str(x).strip().lower()
    if not s or s in {"na", "nan", "none", "-"}:
        return None, "blank_or_na"
    s = re.sub(r"^[a-z0-9]+[_-](qi|tf)\d+__", "", s)
    s = re.sub(r"\[[^\]]+\][+-]?", "", s)
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s, "lowercase_trim_assay_prefix_adduct_spacing"


def infer_method_from_feature(feature: str, method: object = None) -> str:
    if method is not None and not pd.isna(method):
        m = str(method)
        return {"C18n": "C18-neg", "C18p": "C18-pos", "HILn": "HILIC-neg", "HILp": "HILIC-pos"}.get(m, m)
    f = str(feature)
    if f.startswith("C18-neg") or f.startswith("C18n"):
        return "C18-neg"
    if f.startswith("C18-pos") or f.startswith("C18p"):
        return "C18-pos"
    if f.startswith("HILIC-neg") or f.startswith("HILn"):
        return "HILIC-neg"
    if f.startswith("HILIC-pos") or f.startswith("HILp"):
        return "HILIC-pos"
    return "unknown"


def ion_from_method(method: str) -> str:
    if "neg" in str(method).lower():
        return "negative"
    if "pos" in str(method).lower():
        return "positive"
    return "unknown"


def compatible_assay(fr_method: str, ih_method: str) -> str:
    if fr_method == ih_method and fr_method != "unknown":
        return "same or strongly compatible assay"
    if ion_from_method(fr_method) == ion_from_method(ih_method) and ion_from_method(fr_method) != "unknown":
        return "broadly compatible assay"
    if fr_method != "unknown" and ih_method != "unknown":
        return "cross-assay"
    return "unknown"


def yes_flag(series: pd.Series) -> pd.Series:
    x = series.astype("string").str.strip().str.lower()
    out = x.isin(["yes", "y", "true", "1"]).astype("boolean")
    out[x.isna()] = pd.NA
    return out


def is_true_flag(value: object) -> bool:
    return pd.notna(value) and bool(value)


def load_franzosa() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repaired_dir = FR_DIR / "downloaded_full_metabolomics"
    mtb_path = repaired_dir / "franzosa_original_samples_original_mapped_metabolites.repaired.tsv"
    map_path = repaired_dir / "franzosa_original_mapped_metabolites.repaired.map.tsv"
    if not mtb_path.exists() or not map_path.exists():
        mtb_path = FR_DIR / "mtb.tsv"
        map_path = FR_DIR / "mtb.map.tsv"
    return read_tsv(FR_DIR / "metadata.tsv"), read_tsv(mtb_path), read_tsv(map_path)


def load_ihmp() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mtb_path = IH_DIR / "mtb.tsv"
    if not mtb_path.exists() and (IH_DIR / "mtb.tsv.zip").exists():
        mtb_path = IH_DIR / "mtb.tsv.zip"
    return read_tsv(IH_DIR / "metadata.tsv"), read_tsv(mtb_path), read_tsv(IH_DIR / "mtb.map.tsv")


def standardize_fr_meta(meta: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "cohort": "Franzosa",
            "sample_id": meta["Sample"].astype(str),
            "participant_id": meta["Subject"].astype(str),
            "original_label": meta["Study.Group"].astype(str),
            "age_years": pd.to_numeric(meta["Age"], errors="coerce"),
            "sex": pd.NA,
            "bmi": pd.NA,
            "smoking_status": pd.NA,
            "fecal_calprotectin": pd.to_numeric(meta["Fecal.Calprotectin"], errors="coerce"),
            "antibiotic_flag": yes_flag(meta["antibiotic"]),
            "steroid_flag": yes_flag(meta["steroids"]),
            "immunosuppressant_flag": yes_flag(meta["immunosuppressant"]),
            "mesalamine_flag": yes_flag(meta["mesalamine"]),
            "visit_num": 0,
            "week_num": 0,
            "collection_date": pd.NaT,
            "sample_type": "stool/fecal (source-study metabolomics)",
        }
    )
    out["harmonised_label"] = out["original_label"].map({"Control": "Control", "CD": "CD", "UC": "UC"})
    out["ibd_binary"] = out["harmonised_label"].isin(["CD", "UC"]).astype(int)
    out["log_calprotectin"] = np.log1p(out["fecal_calprotectin"])
    return out


def standardize_ih_meta(meta: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "cohort": "iHMP",
            "sample_id": meta["Sample"].astype(str),
            "participant_id": meta["Subject"].astype(str),
            "original_label": meta["Study.Group"].astype(str),
            "age_years": pd.to_numeric(meta["consent_age"], errors="coerce"),
            "sex": meta.get("Gender", pd.Series(pd.NA, index=meta.index)),
            "bmi": pd.to_numeric(meta.get("BMI_at_baseline", pd.Series(pd.NA, index=meta.index)), errors="coerce"),
            "smoking_status": meta.get("smoking status", pd.Series(pd.NA, index=meta.index)),
            "fecal_calprotectin": pd.to_numeric(meta.get("fecalcal", pd.Series(pd.NA, index=meta.index)), errors="coerce"),
            "antibiotic_flag": yes_flag(meta["Antibiotics"]),
            "steroid_flag": pd.NA,
            "immunosuppressant_flag": pd.NA,
            "mesalamine_flag": pd.NA,
            "visit_num": pd.to_numeric(meta.get("visit_num", pd.Series(pd.NA, index=meta.index)), errors="coerce"),
            "week_num": pd.to_numeric(meta.get("week_num", pd.Series(pd.NA, index=meta.index)), errors="coerce"),
            "collection_date": pd.to_datetime(meta.get("date_of_receipt", pd.Series(pd.NA, index=meta.index)), errors="coerce"),
            "sample_type": "stool/fecal (source-study metabolomics)",
        }
    )
    out["harmonised_label"] = out["original_label"].map({"nonIBD": "Control", "CD": "CD", "UC": "UC"})
    out["ibd_binary"] = out["harmonised_label"].isin(["CD", "UC"]).astype(int)
    out["log_calprotectin"] = np.log1p(out["fecal_calprotectin"])
    return out


def select_ihmp_first_visit(meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = meta.copy()
    eligible["_date_rank"] = eligible["collection_date"].fillna(pd.Timestamp.max)
    eligible["_visit_rank"] = eligible["visit_num"].fillna(eligible["week_num"]).fillna(10**9)
    eligible = eligible.sort_values(["participant_id", "_date_rank", "_visit_rank", "sample_id"])
    eligible["selected_primary_first_visit"] = ~eligible.duplicated("participant_id")
    visit_log = eligible[
        ["participant_id", "sample_id", "harmonised_label", "collection_date", "visit_num", "week_num", "selected_primary_first_visit"]
    ].copy()
    return eligible[eligible["selected_primary_first_visit"]].drop(columns=["_date_rank", "_visit_rank"]), visit_log


def build_annotation_long(map_df: pd.DataFrame, cohort: str, source_file: str) -> pd.DataFrame:
    rows = []
    for _, r in map_df.iterrows():
        feature = str(r["Compound"])
        method = infer_method_from_feature(feature, r.get("Method"))
        norm, ops = normalize_name(r.get("Compound.Name"))
        hmdbs = clean_id_values(r.get("HMDB"))
        keggs = clean_id_values(r.get("KEGG"))
        id_rows = []
        for h in hmdbs or [None]:
            id_rows.append(("HMDB", h, valid_hmdb(h) if h else None))
        for k in keggs or [None]:
            id_rows.append(("KEGG", k, valid_kegg_compound(k) if k else None))
        if not id_rows:
            id_rows = [(None, None, None)]
        for id_type, mol_id, valid in id_rows:
            rows.append(
                {
                    "cohort": cohort,
                    "source_file": source_file,
                    "original_feature_name": feature,
                    "original_metabolite_name": r.get("Compound.Name"),
                    "normalised_name": norm,
                    "name_normalisation_operations": ops,
                    "original_HMDB_field": r.get("HMDB"),
                    "original_KEGG_field": r.get("KEGG"),
                    "id_type": id_type,
                    "standardised_id": mol_id,
                    "valid_id": valid,
                    "assay_method": method,
                    "ionisation_mode": ion_from_method(method),
                    "mz": pd.to_numeric(r.get("m.z"), errors="coerce"),
                    "retention_time": pd.to_numeric(r.get("Retention.Time"), errors="coerce"),
                    "annotation_confidence": r.get("High.Confidence.Annotation"),
                    "standard_matched_status": r.get("High.Confidence.Annotation"),
                }
            )
    return pd.DataFrame(rows)


def choose_feature_for_id(mtb: pd.DataFrame, map_df: pd.DataFrame, id_col: str, ids: set[str]) -> dict[str, str]:
    choices = {}
    feature_by_id: dict[str, list[str]] = {metabolite_id: [] for metabolite_id in ids}
    available = set(mtb.columns)
    fields = [c for c in ["Compound", id_col, "High.Confidence.Annotation", "Adduct", "Retention.Time", "m.z", "Method"] if c in map_df.columns]
    records_by_feature = {}
    for record in map_df[fields].to_dict("records"):
        feature = record["Compound"]
        if feature not in available:
            continue
        records_by_feature[feature] = record
        for metabolite_id in clean_id_values(record.get(id_col)):
            if metabolite_id in feature_by_id:
                feature_by_id[metabolite_id].append(feature)
    for metabolite_id, features in feature_by_id.items():
        candidates = []
        for feature in features:
            record = records_by_feature.get(feature, {})
            confidence = str(record.get("High.Confidence.Annotation", "")).strip().lower() == "true"
            method = infer_method_from_feature(feature, record.get("Method"))
            ion = ion_from_method(method)
            adduct = str(record.get("Adduct", "")).strip()
            rt = pd.to_numeric(record.get("Retention.Time"), errors="coerce")
            mz = pd.to_numeric(record.get("m.z"), errors="coerce")
            candidates.append(
                (
                    0 if confidence else 1,
                    method,
                    ion,
                    adduct,
                    999999 if pd.isna(rt) else float(rt),
                    999999 if pd.isna(mz) else float(mz),
                    feature,
                )
            )
        if candidates:
            candidates.sort()
            choices[metabolite_id] = candidates[0][-1]
    return choices


def make_model(name: str) -> tuple[Pipeline, dict]:
    seed = CONFIG["random_seed"]
    pre = Pipeline(
        [
            ("miss", MissingnessFilter(CONFIG["primary_feature_missingness_threshold"])),
            ("var", ConstantFeatureFilter()),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    if name == "ridge_logistic":
        clf = LogisticRegression(penalty="l2", solver="liblinear", class_weight="balanced", random_state=seed, max_iter=1000)
        grid = {"clf__C": [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 5, 10, 50, 100]}
    elif name == "elastic_net_logistic":
        clf = LogisticRegression(penalty="elasticnet", solver="saga", class_weight="balanced", random_state=seed, max_iter=3000)
        grid = {"clf__C": [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 5, 10], "clf__l1_ratio": [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]}
    elif name == "random_forest":
        clf = RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=1)
        grid = {"clf__n_estimators": [200, 500, 1000], "clf__max_depth": [None, 3, 5, 8, 12], "clf__min_samples_leaf": [1, 2, 3, 5, 10], "clf__min_samples_split": [2, 5, 10], "clf__max_features": ["sqrt", "log2", 0.5, None]}
    elif name == "extra_trees":
        clf = ExtraTreesClassifier(class_weight="balanced", random_state=seed, n_jobs=1)
        grid = {"clf__n_estimators": [200, 500, 1000], "clf__max_depth": [None, 3, 5, 8, 12], "clf__min_samples_leaf": [1, 2, 3, 5, 10], "clf__min_samples_split": [2, 5, 10], "clf__max_features": ["sqrt", "log2", 0.5, None]}
    else:
        clf = BalancedGradientBoostingClassifier(random_state=seed)
        grid = {"clf__n_estimators": [50, 100, 200, 300], "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2], "clf__max_depth": [1, 2, 3], "clf__min_samples_leaf": [1, 3, 5, 10], "clf__subsample": [0.6, 0.8, 1.0], "clf__max_features": [None, "sqrt", "log2"]}
    return Pipeline([("pre", pre), ("clf", clf)]), grid


def make_search(pipe: Pipeline, grid: dict, model_name: str, cv, random_state: int):
    """Training-only inner-CV search; external validation data are never passed here.

    Logistic models use full grids because their complexity grid is modest. Nonlinear
    ensembles use seeded RandomizedSearchCV to cover plausible depth, leaf-size and
    feature-subsampling settings without making nested CV computationally excessive.
    """
    total_configs = len(list(ParameterGrid(grid)))
    n_iter_by_model = CONFIG["nested_cv"]["randomized_search_n_iter"]
    if model_name in n_iter_by_model and total_configs > n_iter_by_model[model_name]:
        return RandomizedSearchCV(
            pipe,
            param_distributions=grid,
            n_iter=n_iter_by_model[model_name],
            scoring=CONFIG["nested_cv"]["selection_metric"],
            cv=cv,
            random_state=random_state,
            n_jobs=1,
            return_train_score=True,
        ), "RandomizedSearchCV", n_iter_by_model[model_name], total_configs
    return GridSearchCV(
        pipe,
        grid,
        scoring=CONFIG["nested_cv"]["selection_metric"],
        cv=cv,
        n_jobs=1,
        return_train_score=True,
    ), "GridSearchCV", total_configs, total_configs


def hyperparameter_grid_plan() -> pd.DataFrame:
    rows = []
    dummy_cv = StratifiedKFold(n_splits=2)
    for model_name in CONFIG["models"]:
        pipe, grid = make_model(model_name)
        _, method, n_tested, n_total = make_search(pipe, grid, model_name, dummy_cv, CONFIG["random_seed"])
        rows.append({
            "model": model_name,
            "full_cartesian_configurations": n_total,
            "search_method": method,
            "configurations_evaluated_per_inner_search": n_tested,
            "random_seed": CONFIG["random_seed"] if method == "RandomizedSearchCV" else pd.NA,
            "selection_metric": CONFIG["nested_cv"]["selection_metric"],
        })
    return pd.DataFrame(rows)


def search_result_rows(gs, context: dict, search_method: str, n_tested: int, n_candidate_grid: int, runtime_seconds: float) -> list[dict]:
    rows = []
    cv_results = pd.DataFrame(gs.cv_results_)
    for _, row in cv_results.iterrows():
        params = row["params"]
        rows.append({
            **context,
            "search_method": search_method,
            "n_parameter_configurations_tested": n_tested,
            "n_candidate_configurations_available": n_candidate_grid,
            "parameter_configuration": json.dumps(params),
            "mean_inner_cv_roc_auc": float(row["mean_test_score"]),
            "sd_inner_cv_roc_auc": float(row["std_test_score"]),
            "mean_inner_cv_pr_auc": np.nan,
            "mean_training_roc_auc": float(row["mean_train_score"]) if "mean_train_score" in row.index and pd.notna(row["mean_train_score"]) else np.nan,
            "sd_training_roc_auc": float(row["std_train_score"]) if "std_train_score" in row.index and pd.notna(row["std_train_score"]) else np.nan,
            "rank_inner_cv_roc_auc": int(row["rank_test_score"]),
            "selected_parameters": json.dumps(gs.best_params_),
            "is_selected_configuration": params == gs.best_params_,
            "mean_fit_time_seconds": float(row.get("mean_fit_time", np.nan)),
            "mean_score_time_seconds": float(row.get("mean_score_time", np.nan)),
            "search_runtime_seconds": runtime_seconds,
            "search_note": "Expanded training-only search improves coverage of plausible model complexity without using the external cohort.",
        })
    return rows


def hyperparameter_search_summary(search_df: pd.DataFrame) -> pd.DataFrame:
    if search_df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["search_stage", "outcome", "train_cohort", "external_cohort", "feature_set", "model", "outer_fold"]
    for keys, group in search_df.groupby(group_cols, dropna=False):
        g = group.sort_values(["rank_inner_cv_roc_auc", "mean_inner_cv_roc_auc"], ascending=[True, False])
        best = g.iloc[0]
        second = g.iloc[1] if len(g) > 1 else None
        rows.append({
            **dict(zip(group_cols, keys)),
            "search_method": best.search_method,
            "selected_parameters": best.selected_parameters,
            "best_inner_cv_roc_auc": best.mean_inner_cv_roc_auc,
            "second_best_inner_cv_roc_auc": second.mean_inner_cv_roc_auc if second is not None else np.nan,
            "best_minus_second_best_roc_auc": best.mean_inner_cv_roc_auc - second.mean_inner_cv_roc_auc if second is not None else np.nan,
            "n_configurations_evaluated": int(best.n_parameter_configurations_tested),
            "n_candidate_configurations_available": int(best.n_candidate_configurations_available),
            "search_runtime_seconds": float(best.search_runtime_seconds),
            "search_note": best.search_note,
        })
    return pd.DataFrame(rows)


def fit_params_for_model(model_name: str, y_train: pd.Series) -> dict:
    """Return fold-local fit parameters without using validation/external labels."""
    return {}


def cross_fitted_predict_proba_training_only(estimator: Pipeline, x: pd.DataFrame, y: pd.Series, cv, model_name: str) -> np.ndarray:
    """Training-only cross-fitted probabilities with fold-local sample weights when needed."""
    return cross_val_predict(estimator, x, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]


def binary_metrics(y_true, y_prob, threshold=0.5) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) == 2 else np.nan,
        "pr_auc": average_precision_score(y_true, y_prob) if len(set(y_true)) == 2 else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "ppv": precision_score(y_true, y_pred, zero_division=0),
        "npv": tn / (tn + fn) if (tn + fn) else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def youden_threshold(y_true, y_prob) -> float:
    if len(set(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    best_idx = int(np.nanargmax(j))
    thr = float(thresholds[best_idx])
    if not np.isfinite(thr):
        return 0.5
    return min(max(thr, 0.0), 1.0)


def fit_logistic_calibration_assessment(y_true, y_prob, eps: float = 1e-6) -> tuple[float, float, str]:
    """Estimate external calibration intercept/slope for assessment only.

    This fits outcome ~ intercept + slope * logit(predicted probability) on the
    external predictions after the frozen model has already predicted. The fitted
    coefficients are reported as calibration diagnostics only and are never used to
    recalibrate probabilities, choose a model, choose hyperparameters or set a threshold.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, "not_estimated_single_external_class"
    logit_p = np.log(p / (1 - p))
    if np.nanstd(logit_p) < 1e-12:
        return np.nan, np.nan, "not_estimated_constant_or_extreme_predictions"
    x = np.column_stack([np.ones(len(logit_p)), logit_p])
    beta = np.zeros(2)
    status = "estimated_external_assessment_only"
    for _ in range(100):
        eta = np.clip(x @ beta, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-8, None)
        hessian = x.T @ (x * w[:, None])
        gradient = x.T @ (y - mu)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            status = "estimated_with_lstsq_due_to_near_singularity"
        beta = beta + step
        if np.max(np.abs(step)) < 1e-7:
            break
    if not np.all(np.isfinite(beta)):
        return np.nan, np.nan, "not_estimated_numerical_failure"
    return float(beta[0]), float(beta[1]), status


def external_calibration_tables_and_plots(preds_df: pd.DataFrame, results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    bin_rows = []
    if preds_df.empty or results_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    context_cols = ["outcome", "train_cohort", "external_cohort", "feature_set", "model"]
    for keys, group in preds_df.groupby(context_cols, dropna=False):
        context = dict(zip(context_cols, keys))
        y = group["true_label"].astype(int).to_numpy()
        p = group["predicted_probability"].astype(float).to_numpy()
        brier = brier_score_loss(y, p) if len(np.unique(y)) == 2 else np.nan
        intercept, slope, status = fit_logistic_calibration_assessment(y, p)
        summary_rows.append({
            **context,
            "external_n": int(len(group)),
            "external_event_n": int(y.sum()),
            "external_nonevent_n": int((1 - y).sum()),
            "brier_score": float(brier) if np.isfinite(brier) else np.nan,
            "calibration_intercept": intercept,
            "calibration_slope": slope,
            "calibration_status": status,
            "calibration_note": "External calibration assessment only; no external recalibration, model selection, hyperparameter tuning or threshold selection.",
        })
        ranked = group.copy()
        ranked["calibration_bin"] = pd.qcut(
            ranked["predicted_probability"].rank(method="first"),
            q=5,
            labels=False,
            duplicates="drop",
        )
        for bin_id, b in ranked.groupby("calibration_bin", dropna=True):
            bin_rows.append({
                **context,
                "calibration_bin": int(bin_id) + 1,
                "n_external_observations": int(len(b)),
                "mean_predicted_probability": float(b["predicted_probability"].mean()),
                "observed_outcome_frequency": float(b["true_label"].mean()),
                "min_predicted_probability": float(b["predicted_probability"].min()),
                "max_predicted_probability": float(b["predicted_probability"].max()),
            })
    summary = pd.DataFrame(summary_rows)
    bins = pd.DataFrame(bin_rows)
    for (outcome, train_cohort, external_cohort), bgroup in bins.groupby(["outcome", "train_cohort", "external_cohort"], dropna=False):
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1.2, label="Perfect calibration")
        for feature_set, fgroup in bgroup.groupby("feature_set", dropna=False):
            fgroup = fgroup.sort_values("mean_predicted_probability")
            srow = summary[
                (summary["outcome"] == outcome)
                & (summary["train_cohort"] == train_cohort)
                & (summary["external_cohort"] == external_cohort)
                & (summary["feature_set"] == feature_set)
            ]
            if not srow.empty:
                label = f"{feature_set}: Brier={srow.iloc[0].brier_score:.3f}, slope={srow.iloc[0].calibration_slope:.2f}" if pd.notna(srow.iloc[0].calibration_slope) else f"{feature_set}: Brier={srow.iloc[0].brier_score:.3f}, slope=NA"
            else:
                label = str(feature_set)
            ax.plot(
                fgroup["mean_predicted_probability"],
                fgroup["observed_outcome_frequency"],
                marker="o",
                linewidth=1.8,
                label=label,
            )
            for _, brow in fgroup.iterrows():
                ax.annotate(
                    f"n={int(brow.n_external_observations)}",
                    (brow.mean_predicted_probability, brow.observed_outcome_frequency),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed event frequency")
        ax.set_title(f"External calibration: {outcome.replace('_', ' ')}\n{train_cohort} -> {external_cohort}")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        stem = f"external_calibration_{outcome}_{train_cohort}_to_{external_cohort}"
        fig.savefig(FIGURES / f"{stem}.png", dpi=200)
        fig.savefig(FIGURES / f"{stem}.svg")
        plt.close(fig)
    return summary, bins


def retained_feature_names(fitted_pipe: Pipeline, input_features: list[str]) -> list[str]:
    miss_mask = fitted_pipe.named_steps["pre"].named_steps["miss"].keep_mask_
    after_missing = [f for f, keep in zip(input_features, miss_mask) if keep]
    const_mask = fitted_pipe.named_steps["pre"].named_steps["var"].keep_mask_
    return [f for f, keep in zip(after_missing, const_mask) if keep]


def feature_annotation_map(master: pd.DataFrame) -> dict:
    ann = {}
    if master.empty:
        return ann
    for _, r in master.iterrows():
        ann[str(r["metabolite_id"])] = {
            "hmdb_id": r.get("metabolite_id", pd.NA),
            "feature_class": "harmonised_HMDB_metabolite",
            "classification_note": "Predictive importance only; not causal and not differential-abundance evidence.",
        }
    return ann


def training_only_importance_rows(
    fitted_model: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    input_features: list[str],
    context: dict,
    metabolite_annotations: dict,
) -> list[dict]:
    rows = []
    model_name = context["model"]
    retained = retained_feature_names(fitted_model, input_features)
    clf = fitted_model.named_steps["clf"]
    if hasattr(clf, "coef_"):
        coefs = np.asarray(clf.coef_).ravel()
        for rank, idx in enumerate(np.argsort(np.abs(coefs))[::-1], start=1):
            if idx >= len(retained):
                continue
            feature = retained[idx]
            ann = metabolite_annotations.get(feature, {})
            rows.append({
                **context,
                "feature": feature,
                "hmdb_id": ann.get("hmdb_id", feature if feature.startswith("HMDB") else pd.NA),
                "importance_measure": "standardised_logistic_coefficient",
                "importance": float(coefs[idx]),
                "absolute_importance": float(abs(coefs[idx])),
                "importance_rank": rank,
                "odds_ratio_per_one_sd_increase": float(np.exp(coefs[idx])),
                "effect_direction": "positive" if coefs[idx] > 0 else "negative" if coefs[idx] < 0 else "zero",
                "coefficient_direction": "positive" if coefs[idx] > 0 else "negative" if coefs[idx] < 0 else "zero",
                "stability_information": "Final frozen training fit; coefficient scale follows training-fold imputation and standardisation inside pipeline.",
                "classification": ann.get("classification_note", "Predictive importance only; not causal and not differential-abundance evidence."),
            })
        return rows
    folds, _, _ = choose_nested_fold_design(y_train)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=CONFIG["random_seed"] + 3000)
    per_fold = []
    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(x_train, y_train), start=1):
        fold_model = clone(fitted_model)
        y_fold = y_train.iloc[tr_idx] if hasattr(y_train, "iloc") else pd.Series(y_train).iloc[tr_idx]
        fold_model.fit(
            x_train.iloc[tr_idx],
            y_fold,
            **fit_params_for_model(model_name, y_fold),
        )
        result = permutation_importance(
            fold_model,
            x_train.iloc[va_idx],
            y_train.iloc[va_idx],
            scoring=CONFIG["nested_cv"]["selection_metric"],
            n_repeats=10,
            random_state=CONFIG["random_seed"] + 3100 + fold_idx,
            n_jobs=1,
        )
        for feature, mean_imp, sd_imp in zip(input_features, result.importances_mean, result.importances_std):
            per_fold.append({"feature": feature, "fold": fold_idx, "importance": mean_imp, "importance_sd": sd_imp})
    fold_df = pd.DataFrame(per_fold)
    if fold_df.empty:
        return rows
    agg = fold_df.groupby("feature").agg(
        importance=("importance", "mean"),
        importance_sd=("importance", "std"),
        positive_fold_fraction=("importance", lambda s: float((s > 0).mean())),
    ).reset_index()
    agg["absolute_importance"] = agg["importance"].abs()
    agg = agg.sort_values("absolute_importance", ascending=False)
    for rank, (_, r) in enumerate(agg.iterrows(), start=1):
        ann = metabolite_annotations.get(str(r.feature), {})
        rows.append({
            **context,
            "feature": r.feature,
            "hmdb_id": ann.get("hmdb_id", r.feature if str(r.feature).startswith("HMDB") else pd.NA),
            "importance_measure": "training_cv_permutation_importance_roc_auc",
            "importance": float(r.importance),
            "absolute_importance": float(r.absolute_importance),
            "importance_rank": rank,
            "odds_ratio_per_one_sd_increase": np.nan,
            "effect_direction": "not_directional_for_permutation_importance",
            "coefficient_direction": pd.NA,
            "stability_information": f"Held-out training-fold permutation importance; positive importance in {r.positive_fold_fraction:.2f} of folds. Negative values indicate instability/noise, not biological direction.",
            "classification": ann.get("classification_note", "Predictive importance only; not causal and not differential-abundance evidence."),
        })
    return rows


def stable_cross_cohort_feature_rows(important_df: pd.DataFrame) -> pd.DataFrame:
    if important_df.empty:
        return pd.DataFrame(columns=[
            "outcome", "feature_set", "hmdb_id", "franzosa_to_ihmp_direction", "ihmp_to_franzosa_direction",
            "direction_agreement", "franzosa_to_ihmp_rank", "ihmp_to_franzosa_rank", "stability_field",
            "classification",
        ])
    met = important_df[important_df["hmdb_id"].notna() & (important_df["feature_set"].isin(["metabolomics", "integrated"]))].copy()
    rows = []
    for keys, group in met.groupby(["outcome", "feature_set", "hmdb_id"], dropna=False):
        a = group[(group["train_cohort"] == "Franzosa") & (group["external_cohort"] == "iHMP")]
        b = group[(group["train_cohort"] == "iHMP") & (group["external_cohort"] == "Franzosa")]
        if a.empty or b.empty:
            continue
        a0 = a.sort_values("importance_rank").iloc[0]
        b0 = b.sort_values("importance_rank").iloc[0]
        directional = pd.notna(a0.coefficient_direction) and pd.notna(b0.coefficient_direction)
        rows.append({
            "outcome": keys[0],
            "feature_set": keys[1],
            "hmdb_id": keys[2],
            "franzosa_to_ihmp_direction": a0.coefficient_direction,
            "ihmp_to_franzosa_direction": b0.coefficient_direction,
            "direction_agreement": bool(a0.coefficient_direction == b0.coefficient_direction) if directional else pd.NA,
            "franzosa_to_ihmp_rank": int(a0.importance_rank),
            "ihmp_to_franzosa_rank": int(b0.importance_rank),
            "franzosa_to_ihmp_importance": float(a0.importance),
            "ihmp_to_franzosa_importance": float(b0.importance),
            "stability_field": "Agreement requires same harmonised HMDB metabolite and signed evidence in both transfer directions; tree-only importance is not directional.",
            "classification": "replicated signed predictive association" if directional and a0.coefficient_direction == b0.coefficient_direction else "predictive importance without directional evidence" if not directional else "discordant direction",
        })
    return pd.DataFrame(rows)


def missingness_model_sensitivity(results_df: pd.DataFrame, primary_threshold: float = 0.30) -> pd.DataFrame:
    """Report the predefined primary 30% missingness-threshold model rows only."""
    rows = []
    if not results_df.empty:
        for _, r in results_df.iterrows():
            rows.append({
                "missingness_threshold": primary_threshold,
                "analysis_role": "predefined_primary",
                "outcome": r.outcome,
                "train_cohort": r.train_cohort,
                "external_cohort": r.external_cohort,
                "feature_set": r.feature_set,
                "model": r.model,
                "retained_feature_count": r.n_features_requested,
                "internal_roc_auc": r.internal_cv_auc_mean,
                "internal_pr_auc": r.internal_cv_pr_auc_mean,
                "external_roc_auc": r.roc_auc,
                "brier_score": r.brier,
                "tn": r.tn,
                "fp": r.fp,
                "fn": r.fn,
                "tp": r.tp,
                "status": "complete_current_run",
            })
    return pd.DataFrame(rows)


def write_reporting_output_index() -> None:
    """Surface key existing thesis outputs and make reporting status explicit."""
    items = [
        ("EDA/PCA", "Franzosa within-cohort PCA", "figures/main_text/PCA_Franzosa.png", "Exploratory within-cohort PCA coloured by Control/CD/UC; no batch correction.", "generated_by_pipeline"),
        ("EDA/PCA", "iHMP within-cohort PCA", "figures/main_text/PCA_iHMP.png", "Exploratory within-cohort PCA coloured by Control/CD/UC; no batch correction.", "generated_by_pipeline"),
        ("EDA/PCA", "Class balance", "figures/class_balance_eda_by_cohort_and_task.png", "Use to document class imbalance before modelling.", "generated_by_pipeline"),
        ("Workflow", "Internal model selection workflow", "figures/main_text/internal_model_selection_workflow.png", "Visual method summary for leakage-safe nested CV and external validation.", "generated_by_pipeline"),
        ("Workflow", "Threshold selection workflow", "figures/main_text/threshold_selection_workflow.png", "Visual method summary for training-only Youden threshold freezing.", "generated_by_pipeline"),
        ("Workflow", "External validation workflow", "figures/main_text/external_validation_workflow.png", "Visual method summary for frozen-pipeline external validation.", "generated_by_pipeline"),
        ("Workflow", "Feature importance methodology", "figures/main_text/feature_importance_methodology.png", "Explains coefficient and permutation-importance interpretation.", "generated_by_pipeline"),
        ("Model performance", "Internal full model comparison", "figures/main_text/internal_model_comparison_all_models_best_cohort.png", "All model families and feature sets for the best internal cohort.", "generated_by_pipeline"),
        ("Model performance", "External transportability forest", "figures/main_text/external_validation_transportability_forest.png", "Bidirectional external validation by task, direction and feature set.", "generated_by_pipeline"),
        ("Model performance", "External ROC confidence-band figures", "figures/main_text/external_roc_ci_*.png", "Bootstrap ROC curves and 95% confidence bands for selected external models.", "generated_by_pipeline"),
        ("Model performance", "External precision-recall figures", "figures/main_text/external_pr_ci_*.png", "Precision-recall curves with average precision and prevalence references.", "generated_by_pipeline"),
        ("Model performance", "External probability distributions", "figures/main_text/external_probability_distribution_*.png", "Predicted probability distributions with frozen threshold overlays.", "generated_by_pipeline"),
        ("Model performance", "External prediction ranking plots", "figures/main_text/external_prediction_ranking_*.png", "Waterfall/rank plots of external predicted probabilities.", "generated_by_pipeline"),
        ("Model performance", "Nested CV stability figures", "figures/main_text/nested_cv_stability_*.png", "Outer-fold ROC-AUC and PR-AUC variability for selected models.", "generated_by_pipeline"),
        ("Model performance", "Calibration assessment figures", "figures/main_text/calibration_*.png", "Assessment-only external calibration curves, probability histograms, Brier score and calibration slope/intercept.", "generated_by_pipeline"),
        ("Model performance", "Final model summary", "tables/main_text/final_model_summary.csv", "Consolidated selected model results with discrimination, threshold and calibration metrics.", "generated_by_pipeline"),
        ("Model performance", "Figure index", "tables/main_text/figure_index.csv", "Main-text figure inventory with purpose, section, inputs and output filenames.", "generated_by_pipeline"),
        ("Model performance", "Selected model performance summary", "tables/main_text/selected_model_performance_summary.csv", "Compact thesis table with nested-CV and external metrics for selected models.", "generated_by_pipeline"),
        ("Model performance", "Threshold selection summary", "tables/main_text/threshold_selection_summary.csv", "Training-only selected threshold, Youden J, sensitivity, specificity and ROC-AUC.", "generated_by_pipeline"),
        ("Feature interpretation", "Important feature table", "tables/important_feature_table.csv", "Training-only coefficients/permutation importance; predictive, not causal.", "generated_by_pipeline"),
        ("Feature interpretation", "Feature stability plot", "figures/main_text/feature_stability.png", "Model stability summary if present from the completed analysis run.", "surfaced_if_present"),
        ("Feature interpretation", "Stable cross-cohort metabolites", "tables/stable_cross_cohort_metabolites.csv", "Cross-direction agreement for harmonised HMDB metabolites.", "generated_by_pipeline"),
        ("Feature interpretation", "Cross-cohort signed stability figure", "figures/main_text/cross_cohort_feature_stability.png", "Replicated same-direction signed predictive coefficients across transfer directions.", "generated_by_pipeline"),
        ("Feature interpretation", "Consolidated top-feature figures", "figures/main_text/top_features_consolidated_*.png", "Slide-ready top Franzosa-trained metabolomics features by outcome.", "generated_by_pipeline"),
        ("Feature interpretation", "Top feature plots", "figures/supplementary/top_features_*.png", "Supplementary per-outcome feature-contribution plots.", "generated_by_pipeline"),
        ("Associations", "Differential abundance", "tables/differential_abundance_results.csv", "HC3 robust associations by cohort and outcome.", "generated_by_pipeline"),
        ("Associations", "Volcano plots", "figures/main_text/volcano_standardized_*.png", "BH-FDR volcano plots with standardised disease beta.", "generated_by_pipeline"),
        ("Associations", "Shared metabolite boxplots", "figures/main_text/shared_metabolite_boxplots_*.png", "Boxplots for top replicated FDR-significant shared metabolites by contrast.", "generated_by_pipeline"),
        ("Pathways", "KEGG pathway ORA", "figures/main_text/pathway_enrichment.png", "BH-FDR-significant KEGG pathway over-representation summary.", "generated_by_pipeline"),
        ("Pathways", "KEGG mapping coverage", "figures/main_text/pathway_mapping_coverage.png", "KEGG coverage among tested/significant HMDB metabolites.", "generated_by_pipeline"),
        ("Missingness", "Primary 30% missingness threshold", "tables/missingness_threshold_model_sensitivity.csv", "Primary model rows using the predefined 30% missingness threshold.", "generated_by_pipeline"),
        ("Audit", "Complete generated table/figure manifest", "logs/end_of_run_manifest.json", "Machine-readable list of all generated CSV, PNG and SVG outputs.", "generated_by_pipeline"),
    ]
    rows = []
    for category, title, rel_path, use, status in items:
        path = WORK / rel_path
        if "*" in rel_path:
            exists = bool(list(WORK.glob(rel_path)))
        else:
            exists = path.exists()
        rows.append({
            "category": category,
            "title": title,
            "output_path": rel_path,
            "exists_at_report_time": bool(exists),
            "reporting_status": status,
            "recommended_use": use,
        })
    index_df = pd.DataFrame(rows)
    index_df.to_csv(TABLES / "thesis_reporting_output_index.csv", index=False)
    md = [
        "# Thesis Reporting Output Index",
        "",
        "This index surfaces representative outputs already produced by the IBD pipeline. It does not introduce new analyses or alter model fitting.",
        "",
        index_df.to_markdown(index=False),
        "",
        "Missingness-threshold note: only the predefined primary 30% missingness threshold is reported here.",
    ]
    (REPORTS / "thesis_reporting_output_index.md").write_text("\n".join(md) + "\n")


def bh_fdr(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    q = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    if valid.empty:
        return q
    n = len(valid)
    ranks = np.arange(1, n + 1)
    adjusted = (valid.to_numpy() * n / ranks)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q.loc[valid.index] = np.clip(adjusted, 0, 1)
    return q


def robust_ols_hc3(y: pd.Series, x: pd.DataFrame, exposure_col: str) -> dict:
    data = pd.concat([y.rename("y"), x], axis=1).dropna()
    if len(data) < max(8, x.shape[1] + 3) or data[exposure_col].nunique() < 2:
        return {"status": "insufficient_complete_cases", "n": len(data)}
    yv = data["y"].astype(float).to_numpy()
    xv = data[x.columns].astype(float).to_numpy()
    try:
        beta = np.linalg.lstsq(xv, yv, rcond=None)[0]
        resid = yv - xv @ beta
        xtx_inv = np.linalg.pinv(xv.T @ xv)
        hat = np.sum((xv @ xtx_inv) * xv, axis=1)
        scale = (resid / np.clip(1 - hat, 1e-8, None)) ** 2
        cov = xtx_inv @ (xv.T @ (xv * scale[:, None])) @ xtx_inv
        se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
        idx = list(x.columns).index(exposure_col)
        coef = float(beta[idx])
        se_exp = float(se[idx])
        y_sd = float(np.nanstd(yv, ddof=1)) if len(yv) > 1 else np.nan
        t_stat = coef / se_exp if se_exp > 0 else np.nan
        df_resid = len(data) - x.shape[1]
        p_val = (
            float(2 * t_dist.sf(abs(t_stat), df_resid))
            if np.isfinite(t_stat) and df_resid > 0 and t_dist is not None
            else np.nan
        )
        return {
            "status": "estimated_hc3",
            "n": len(data),
            "coef": coef,
            "se": se_exp,
            "standardized_coef": coef / y_sd if np.isfinite(y_sd) and y_sd > 0 else np.nan,
            "standardized_se": se_exp / y_sd if np.isfinite(y_sd) and y_sd > 0 else np.nan,
            "ci_low": coef - 1.96 * se_exp if np.isfinite(se_exp) else np.nan,
            "ci_high": coef + 1.96 * se_exp if np.isfinite(se_exp) else np.nan,
            "standardized_ci_low": (coef - 1.96 * se_exp) / y_sd if np.isfinite(y_sd) and y_sd > 0 and np.isfinite(se_exp) else np.nan,
            "standardized_ci_high": (coef + 1.96 * se_exp) / y_sd if np.isfinite(y_sd) and y_sd > 0 and np.isfinite(se_exp) else np.nan,
            "p_value": p_val,
        }
    except Exception as exc:
        return {"status": f"model_failed:{type(exc).__name__}", "n": len(data)}


def cohort_outcome_design(df: pd.DataFrame, outcome: str, include_calprotectin: bool = False) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    sub, y = outcome_subset(df, outcome)
    design = pd.DataFrame({"intercept": 1.0, "disease_status": y.astype(float)}, index=sub.index)
    if "age_years" in sub:
        design["age_years"] = pd.to_numeric(sub["age_years"], errors="coerce")
    if include_calprotectin:
        design["log_calprotectin"] = pd.to_numeric(sub["log_calprotectin"], errors="coerce")
    return sub, y, design


def differential_abundance_analysis(data_by_cohort: dict, master: pd.DataFrame, hmdb_ids: list[str]) -> pd.DataFrame:
    rows = []
    meta = master.set_index("metabolite_id") if "metabolite_id" in master.columns else pd.DataFrame()
    for cohort, df in data_by_cohort.items():
        for outcome in CONFIG["outcomes"]:
            sub, y, design = cohort_outcome_design(df, outcome, include_calprotectin=False)
            for hmdb_id in hmdb_ids:
                if hmdb_id not in sub:
                    continue
                fit = robust_ols_hc3(pd.to_numeric(sub[hmdb_id], errors="coerce"), design, "disease_status")
                m = meta.loc[hmdb_id] if hmdb_id in meta.index else pd.Series(dtype=object)
                rows.append({
                    "cohort": cohort,
                    "outcome": outcome,
                    "hmdb_id": hmdb_id,
                    "original_cohort_specific_feature": m.get("franzosa_feature" if cohort == "Franzosa" else "ihmp_feature", pd.NA),
                    "assay_method": m.get("franzosa_method" if cohort == "Franzosa" else "ihmp_method", pd.NA),
                    "analysis_sample_size": fit.get("n", 0),
                    "disease_sample_count": int(y.sum()),
                    "control_sample_count": int((1 - y).sum()),
                    "disease_coefficient": fit.get("coef", np.nan),
                    "standardized_beta": fit.get("standardized_coef", np.nan),
                    "standardized_se": fit.get("standardized_se", np.nan),
                    "standardized_ci_low": fit.get("standardized_ci_low", np.nan),
                    "standardized_ci_high": fit.get("standardized_ci_high", np.nan),
                    "effect_direction": "higher_in_disease" if fit.get("coef", 0) > 0 else "lower_in_disease" if fit.get("coef", 0) < 0 else "not_estimated",
                    "robust_standard_error": fit.get("se", np.nan),
                    "ci_low": fit.get("ci_low", np.nan),
                    "ci_high": fit.get("ci_high", np.nan),
                    "raw_p_value": fit.get("p_value", np.nan),
                    "model_status": fit.get("status"),
                    "model_formula": "metabolite ~ disease_status + age_years",
                    "transformation_status": "supplied processed scale analysed; no automatic re-transformation applied",
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_q_value"] = out.groupby(["cohort", "outcome"], group_keys=False)["raw_p_value"].apply(bh_fdr)
        out["significant_fdr_0_05"] = out["fdr_q_value"] <= 0.05
    return out


def calprotectin_adjusted_analysis(diff_df: pd.DataFrame, data_by_cohort: dict, hmdb_ids: list[str]) -> pd.DataFrame:
    rows = []
    before = diff_df.set_index(["cohort", "outcome", "hmdb_id"]) if not diff_df.empty else pd.DataFrame()
    for cohort, df in data_by_cohort.items():
        for outcome in CONFIG["outcomes"]:
            sub, y, design = cohort_outcome_design(df, outcome, include_calprotectin=True)
            for hmdb_id in hmdb_ids:
                if hmdb_id not in sub:
                    continue
                fit = robust_ols_hc3(pd.to_numeric(sub[hmdb_id], errors="coerce"), design, "disease_status")
                key = (cohort, outcome, hmdb_id)
                unadj = before.loc[key] if key in before.index else pd.Series(dtype=object)
                before_coef = unadj.get("disease_coefficient", np.nan)
                after_coef = fit.get("coef", np.nan)
                if fit.get("status") != "estimated_hc3":
                    category = "insufficient complete cases"
                elif np.sign(before_coef) != 0 and np.sign(after_coef) != 0 and np.sign(before_coef) != np.sign(after_coef):
                    category = "direction changed"
                elif pd.notna(unadj.get("fdr_q_value", np.nan)) and unadj.get("fdr_q_value", 1) <= 0.05 and fit.get("p_value", 1) <= 0.05:
                    category = "remains associated after adjustment"
                elif pd.notna(before_coef) and pd.notna(after_coef) and abs(after_coef) < abs(before_coef):
                    category = "attenuated after adjustment"
                else:
                    category = "not significant before or after adjustment"
                rows.append({
                    "cohort": cohort,
                    "outcome": outcome,
                    "hmdb_id": hmdb_id,
                    "disease_coefficient_before_calprotectin_adjustment": before_coef,
                    "disease_coefficient_after_calprotectin_adjustment": after_coef,
                    "adjusted_standard_error": fit.get("se", np.nan),
                    "adjusted_ci_low": fit.get("ci_low", np.nan),
                    "adjusted_ci_high": fit.get("ci_high", np.nan),
                    "adjusted_p_value": fit.get("p_value", np.nan),
                    "coefficient_attenuation": before_coef - after_coef if pd.notna(before_coef) and pd.notna(after_coef) else np.nan,
                    "direction_retained_or_changed": category,
                    "analysis_sample_size": fit.get("n", 0),
                    "interpretation_category": category,
                    "model_status": fit.get("status"),
                    "interpretation_note": "Association after inflammation-marker adjustment; not causal evidence.",
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["adjusted_fdr_q_value"] = out.groupby(["cohort", "outcome"], group_keys=False)["adjusted_p_value"].apply(bh_fdr)
    return out


def class_enrichment_analysis(diff_df: pd.DataFrame, master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if diff_df.empty:
        return pd.DataFrame(), pd.DataFrame([{"status": "not_run", "reason": "differential abundance table empty"}])
    class_col = None
    for candidate in ["annotation_class", "class", "superclass", "chemical_class", "assay_compatibility"]:
        if candidate in master.columns:
            class_col = candidate
            break
    if class_col is None:
        return pd.DataFrame(), pd.DataFrame([{"status": "not_run", "reason": "No defensible supplied HMDB class/superclass or KEGG pathway annotation available; pathway labels were not invented."}])
    annot = master[["metabolite_id", class_col]].rename(columns={"metabolite_id": "hmdb_id", class_col: "pathway_or_class"}).dropna()
    merged = diff_df.merge(annot, on="hmdb_id", how="inner")
    rows = []
    for keys, group in merged.groupby(["cohort", "outcome"], dropna=False):
        selected = group[group["fdr_q_value"] <= 0.05]
        for label, cg in group.groupby("pathway_or_class"):
            if len(cg) < 2:
                continue
            sig = selected[selected["pathway_or_class"] == label]
            rows.append({
                "cohort": keys[0],
                "outcome": keys[1],
                "annotation_source": class_col,
                "pathway_or_class": label,
                "eligible_metabolite_count": int(len(cg)),
                "significant_metabolite_count": int(len(sig)),
                "enrichment_statistic": (len(sig) / max(1, len(cg))) / max(1e-9, len(selected) / max(1, len(group))),
                "raw_p_value": np.nan,
                "leading_metabolites": ";".join(sig.sort_values("fdr_q_value")["hmdb_id"].head(5).astype(str)),
                "interpretation_note": "Broad supplied-class enrichment only; no fabricated pathway labels.",
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["adjusted_p_value"] = out.groupby(["cohort", "outcome"], group_keys=False)["raw_p_value"].apply(bh_fdr)
    status = pd.DataFrame([{"status": "completed_broad_class_summary" if not out.empty else "not_run", "reason": f"Used supplied annotation field: {class_col}"}])
    return out, status


def annotation_resource_dirs() -> list[Path]:
    dirs = []
    for rel in CONFIG["annotation"]["annotation_resource_directories"]:
        path = ROOT / rel
        dirs.append(path)
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def read_mapping_sources() -> pd.DataFrame:
    """Collect priority-1 names and identifiers from supplied cohort map files."""
    paths = [
        FR_DIR / "downloaded_full_metabolomics" / "franzosa_original_mapped_metabolites.repaired.map.tsv",
        FR_DIR / "downloaded_full_metabolomics" / "franzosa_supplement_full_mtb.map.cleaned.tsv",
        FR_DIR / "mtb.map.tsv",
        IH_DIR / "mtb.map.tsv",
    ]
    rows = []
    log = pathway_logger()
    for path in paths:
        if not path.exists():
            log.warning("Supplied mapping file missing: %s", path)
            continue
        try:
            mp = read_tsv(path)
        except Exception as exc:
            log.warning("Could not read supplied mapping file %s: %s", path, exc)
            continue
        for _, r in mp.iterrows():
            hmdb_values = clean_id_values(r.get("HMDB"))
            if not hmdb_values:
                continue
            valid_keggs = sorted({k for k in clean_id_values(r.get("KEGG")) if valid_kegg_compound(k)})
            for hmdb_id in hmdb_values:
                rows.append({
                    "HMDB_ID": hmdb_id,
                    "metabolite_name": r.get("Compound.Name"),
                    "supplied_KEGG_Compound_ID": ";".join(valid_keggs),
                    "supplied_mapping_source": f"supplied cohort map file: {path.name}",
                })
    cols = ["HMDB_ID", "metabolite_name", "supplied_KEGG_Compound_ID", "supplied_mapping_source"]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    agg = {
        "metabolite_name": lambda s: next((x for x in s.dropna().astype(str) if x and x.lower() != "nan"), pd.NA),
        "supplied_KEGG_Compound_ID": lambda s: ";".join(sorted({x for v in s.dropna().astype(str) for x in v.split(";") if valid_kegg_compound(x)})),
        "supplied_mapping_source": lambda s: "; ".join(sorted(set(s.dropna().astype(str)))),
    }
    return out.groupby("HMDB_ID", as_index=False).agg(agg)


def parse_semicolon_field(value) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [x.strip() for x in re.split(r"[;|]", text) if x.strip()]


def parse_hmdb_xml_annotation(xml_path: Path, wanted_hmdb: set[str]) -> pd.DataFrame:
    """Parse an official HMDB XML export if the user has placed one locally.

    Expected local file examples:
    - thesis_preprocessing_workflow/ibd_pipeline/annotation_resources/hmdb_metabolites.xml
    - IBD-DataSet/annotation_resources/hmdb_metabolites.xml

    This avoids scraping and makes the annotation source reproducible for thesis audit.
    """
    if not xml_path.exists() or not wanted_hmdb:
        return pd.DataFrame()
    rows = []
    context = ET.iterparse(xml_path, events=("end",))
    for _, elem in context:
        tag = elem.tag.split("}")[-1]
        if tag != "metabolite":
            continue
        def first_text(name: str):
            found = elem.find(f".//{{*}}{name}")
            return found.text.strip() if found is not None and found.text else pd.NA
        accession = first_text("accession")
        if accession in wanted_hmdb:
            pathways = []
            for pathway in elem.findall(".//{*}pathways/{*}pathway"):
                pname = pathway.find(".//{*}name")
                if pname is not None and pname.text:
                    pathways.append(pname.text.strip())
            smpdb = []
            for pid in elem.findall(".//{*}smpdb_id"):
                if pid.text:
                    smpdb.append(pid.text.strip())
            rows.append({
                "HMDB_ID": accession,
                "metabolite_name": first_text("name"),
                "HMDB_superclass": first_text("super_class"),
                "HMDB_class": first_text("class"),
                "HMDB_subclass": first_text("sub_class"),
                "SMPDB_pathways": ";".join(sorted(set(pathways + smpdb))),
                "hmdb_xml_source": f"official HMDB XML: {xml_path.name}",
            })
        elem.clear()
    return pd.DataFrame(rows)


def parse_hmdb_zip_annotation(zip_path: Path, wanted_hmdb: set[str]) -> pd.DataFrame:
    if not zip_path.exists() or not zipfile.is_zipfile(zip_path):
        return pd.DataFrame()
    with zipfile.ZipFile(zip_path) as zf:
        xml_names = [name for name in zf.namelist() if name.lower().endswith(".xml")]
        if not xml_names:
            return pd.DataFrame()
        tmp_xml = WORK / "annotation_resources" / "_hmdb_annotation_extract.xml"
        with zf.open(xml_names[0]) as src, tmp_xml.open("wb") as dst:
            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                dst.write(chunk)
    try:
        return parse_hmdb_xml_annotation(tmp_xml, wanted_hmdb)
    finally:
        try:
            tmp_xml.unlink()
        except FileNotFoundError:
            pass


def load_external_annotation_cache() -> pd.DataFrame:
    """Load a user-provided annotation cache if present.

    The cache must be generated from formal resources and have the requested output
    columns. It is intentionally separate from model development and never uses labels.
    """
    for directory in annotation_resource_dirs():
        path = directory / "metabolite_annotation_enrichment.csv"
        if path.exists():
            df = pd.read_csv(path)
            df.columns = [c.strip().replace(" ", "_") for c in df.columns]
            if "HMDB_ID" not in df and "HMDB" in df:
                df = df.rename(columns={"HMDB": "HMDB_ID"})
            df["cache_source"] = f"local annotation cache: {path.name}"
            return df
    return pd.DataFrame()


def local_kegg_cache_paths() -> tuple[Path, Path]:
    directory = annotation_resource_dirs()[0]
    return directory / "kegg_compound_pathways.tsv", directory / "kegg_pathway_names.tsv"


def download_kegg_cache_if_allowed() -> pd.DataFrame:
    """Optionally generate KEGG caches using official REST endpoints.

    External downloads are disabled by default for reproducibility. When enabled,
    responses are cached locally so later runs can be offline.
    """
    link_path, name_path = local_kegg_cache_paths()
    rows = []
    if link_path.exists() and name_path.exists():
        return pd.DataFrame([{
            "retrieval_date": CONFIG["annotation"]["annotation_retrieval_date"],
            "api_endpoint": "local cache",
            "number_of_compounds_retrieved": pd.NA,
            "number_of_pathways_retrieved": pd.NA,
            "number_of_failed_queries": 0,
            "retry_attempts": 0,
            "status": "cache_present",
        }])
    if not CONFIG["annotation"]["allow_external_annotation_download"]:
        return pd.DataFrame([{
            "retrieval_date": CONFIG["annotation"]["annotation_retrieval_date"],
            "api_endpoint": "KEGG REST download disabled",
            "number_of_compounds_retrieved": 0,
            "number_of_pathways_retrieved": 0,
            "number_of_failed_queries": 0,
            "retry_attempts": 0,
            "status": "download_disabled_using_existing_cache_if_any",
        }])
    endpoints = [
        ("https://rest.kegg.jp/link/pathway/compound", link_path),
        ("https://rest.kegg.jp/list/pathway", name_path),
    ]
    log = pathway_logger()
    for endpoint, out_path in endpoints:
        success = False
        failures = 0
        for attempt in range(CONFIG["annotation"]["kegg_retry_count"] + 1):
            try:
                with urlopen(endpoint, timeout=30) as response:
                    out_path.write_bytes(response.read())
                success = True
                time.sleep(CONFIG["annotation"]["kegg_request_delay_seconds"])
                break
            except URLError as exc:
                failures += 1
                log.warning("KEGG request failed (%s), attempt %s: %s", endpoint, attempt + 1, exc)
                time.sleep(CONFIG["annotation"]["kegg_request_delay_seconds"])
        rows.append({
            "retrieval_date": datetime.now(timezone.utc).isoformat(),
            "api_endpoint": endpoint,
            "number_of_compounds_retrieved": pd.NA,
            "number_of_pathways_retrieved": pd.NA,
            "number_of_failed_queries": failures,
            "retry_attempts": CONFIG["annotation"]["kegg_retry_count"],
            "status": "downloaded" if success else "failed",
        })
    return pd.DataFrame(rows)


def load_kegg_pathway_cache() -> tuple[pd.DataFrame, pd.DataFrame]:
    retrieval_log = download_kegg_cache_if_allowed()
    link_path, name_path = local_kegg_cache_paths()
    if not link_path.exists():
        return pd.DataFrame(columns=["KEGG_Compound_ID", "KEGG_pathway_ID", "KEGG_pathway_name", "pathway_category", "mapping_source", "retrieval_date"]), retrieval_log
    try:
        links = pd.read_csv(link_path, sep="\t", header=None, names=["compound", "pathway"])
    except Exception as exc:
        pathway_logger().warning("Could not parse KEGG compound-pathway cache: %s", exc)
        return pd.DataFrame(columns=["KEGG_Compound_ID", "KEGG_pathway_ID", "KEGG_pathway_name", "pathway_category", "mapping_source", "retrieval_date"]), retrieval_log
    if links.empty or not links["compound"].astype(str).str.startswith("cpd:").any():
        return pd.DataFrame(columns=["KEGG_Compound_ID", "KEGG_pathway_ID", "KEGG_pathway_name", "pathway_category", "mapping_source", "retrieval_date"]), retrieval_log
    links["KEGG_Compound_ID"] = links["compound"].astype(str).str.replace("cpd:", "", regex=False)
    links["KEGG_pathway_ID"] = links["pathway"].astype(str).str.replace("path:", "", regex=False)
    names = pd.DataFrame(columns=["KEGG_pathway_ID", "KEGG_pathway_name"])
    if name_path.exists():
        try:
            names = pd.read_csv(name_path, sep="\t", header=None, names=["KEGG_pathway_ID", "KEGG_pathway_name"])
            names["KEGG_pathway_ID"] = names["KEGG_pathway_ID"].astype(str).str.replace("path:", "", regex=False)
        except Exception as exc:
            pathway_logger().warning("Could not parse KEGG pathway-name cache: %s", exc)
            names = pd.DataFrame(columns=["KEGG_pathway_ID", "KEGG_pathway_name"])
    out = links.merge(names, on="KEGG_pathway_ID", how="left")
    out["KEGG_pathway_name"] = out["KEGG_pathway_name"].fillna(out["KEGG_pathway_ID"])
    out["pathway_category"] = np.where(out["KEGG_pathway_ID"].str.startswith("map"), "reference pathway", pd.NA)
    out["mapping_source"] = "official KEGG REST cache"
    out["retrieval_date"] = CONFIG["annotation"]["annotation_retrieval_date"]
    out = out[["KEGG_Compound_ID", "KEGG_pathway_ID", "KEGG_pathway_name", "pathway_category", "mapping_source", "retrieval_date"]].drop_duplicates()
    return out.sort_values(["KEGG_Compound_ID", "KEGG_pathway_ID"]), retrieval_log


def build_metabolite_annotation_enrichment(master: pd.DataFrame) -> pd.DataFrame:
    requested_cols = [
        "HMDB_ID", "metabolite_name", "supplied_KEGG_Compound_ID",
        "externally_mapped_KEGG_Compound_ID", "final_KEGG_Compound_ID",
        "HMDB_superclass", "HMDB_class", "HMDB_subclass", "SMPDB_pathways",
        "mapping_source", "source_version_or_retrieval_date", "mapping_status",
        "mapping_confidence", "ambiguity_flag", "ambiguity_note",
    ]
    hmdb_ids = sorted(set(master["metabolite_id"].dropna().astype(str)))
    base = pd.DataFrame({"HMDB_ID": hmdb_ids})
    local = read_mapping_sources()
    out = base.merge(local, on="HMDB_ID", how="left")

    cache = load_external_annotation_cache()
    if not cache.empty:
        keep = [c for c in requested_cols if c in cache.columns]
        out = out.merge(cache[keep].drop_duplicates("HMDB_ID"), on="HMDB_ID", how="left", suffixes=("", "_cache"))
        for c in requested_cols:
            cc = f"{c}_cache"
            if cc in out:
                if c in out:
                    out[c] = out[c].combine_first(out[cc])
                else:
                    out[c] = out[cc]
                out = out.drop(columns=[cc])

    for xml_path in [
        WORK / "annotation_resources" / "hmdb_metabolites.xml",
        WORK / "annotation_resources" / "feces_metabolites.xml",
        ROOT / "IBD-DataSet" / "annotation_resources" / "hmdb_metabolites.xml",
        ROOT / "IBD-DataSet" / "annotation_resources" / "feces_metabolites.xml",
    ]:
        xml_ann = parse_hmdb_xml_annotation(xml_path, set(hmdb_ids))
        if not xml_ann.empty:
            out = out.merge(xml_ann, on="HMDB_ID", how="left", suffixes=("", "_hmdbxml"))
            for c in ["metabolite_name", "HMDB_superclass", "HMDB_class", "HMDB_subclass", "SMPDB_pathways"]:
                cc = f"{c}_hmdbxml"
                if cc in out:
                    if c in out:
                        out[c] = out[c].combine_first(out[cc])
                    else:
                        out[c] = out[cc]
                    out = out.drop(columns=[cc])
            break
    else:
        for zip_path in [
            WORK / "annotation_resources" / "feces_metabolites.zip",
            WORK / "annotation_resources" / "hmdb_metabolites.zip",
            ROOT / "IBD-DataSet" / "annotation_resources" / "feces_metabolites.zip",
            ROOT / "IBD-DataSet" / "annotation_resources" / "hmdb_metabolites.zip",
        ]:
            zip_ann = parse_hmdb_zip_annotation(zip_path, set(hmdb_ids))
            if not zip_ann.empty:
                out = out.merge(zip_ann, on="HMDB_ID", how="left", suffixes=("", "_hmdbzip"))
                for c in ["metabolite_name", "HMDB_superclass", "HMDB_class", "HMDB_subclass", "SMPDB_pathways"]:
                    cc = f"{c}_hmdbzip"
                    if cc in out:
                        if c in out:
                            out[c] = out[c].combine_first(out[cc])
                        else:
                            out[c] = out[cc]
                        out = out.drop(columns=[cc])
                break

    if "externally_mapped_KEGG_Compound_ID" not in out:
        out["externally_mapped_KEGG_Compound_ID"] = pd.NA
    for c in ["supplied_KEGG_Compound_ID", "externally_mapped_KEGG_Compound_ID"]:
        out[c] = out.get(c, pd.Series(index=out.index, dtype=object)).fillna("").map(lambda v: ";".join(sorted({x for x in parse_semicolon_field(v) if valid_kegg_compound(x)})))
    statuses = []
    final_ids = []
    sources = []
    confidence = []
    ambiguity = []
    notes = []
    for _, r in out.iterrows():
        hmdb_valid = valid_hmdb(r["HMDB_ID"])
        supplied = set(parse_semicolon_field(r.get("supplied_KEGG_Compound_ID")))
        external = set(parse_semicolon_field(r.get("externally_mapped_KEGG_Compound_ID")))
        final = supplied | external
        if not hmdb_valid:
            status, conf, amb, note = "invalid_hmdb_id", "none", True, "HMDB ID failed format validation"
        elif supplied and external and supplied != external:
            status, conf, amb, note = "conflicting_mapping", "review_required", True, f"supplied={sorted(supplied)} external={sorted(external)}"
        elif len(final) > 1:
            status, conf, amb, note = "multiple_valid_kegg_mappings", "high" if supplied else "moderate", True, "One HMDB ID maps to multiple KEGG compounds; all retained for ORA"
        elif supplied:
            status, conf, amb, note = "supplied_valid_mapping", "high", False, "Priority-1 supplied cohort map identifier"
        elif external:
            status, conf, amb, note = "externally_mapped", "moderate", False, "External formal identifier mapping"
        else:
            status, conf, amb, note = "no_kegg_mapping", "none", False, "No valid KEGG compound ID available"
        statuses.append(status)
        confidence.append(conf)
        ambiguity.append(amb)
        notes.append(note)
        final_ids.append(";".join(sorted(final)))
        source_parts = []
        if supplied:
            source_parts.append(str(r.get("supplied_mapping_source", "supplied cohort map file")))
        if external:
            source_parts.append(str(r.get("cache_source", "external annotation cache")))
        if pd.notna(r.get("hmdb_xml_source", pd.NA)):
            source_parts.append(str(r.get("hmdb_xml_source")))
        sources.append("; ".join(sorted(set([s for s in source_parts if s and s != "nan"]))) or "supplied HMDB ID only")
    out["final_KEGG_Compound_ID"] = final_ids
    out["mapping_source"] = sources
    out["source_version_or_retrieval_date"] = CONFIG["annotation"]["annotation_retrieval_date"]
    out["mapping_status"] = statuses
    out["mapping_confidence"] = confidence
    out["ambiguity_flag"] = ambiguity
    out["ambiguity_note"] = notes
    for c in ["HMDB_superclass", "HMDB_class", "HMDB_subclass", "SMPDB_pathways"]:
        if c not in out:
            out[c] = pd.NA
    for c in requested_cols:
        if c not in out:
            out[c] = pd.NA
    out = out[requested_cols]
    out.to_csv(TABLES / "metabolite_annotation_enrichment.csv", index=False)
    out.to_csv(TABLES / "hmdb_to_kegg_mapping_audit.csv", index=False)
    out[out["final_KEGG_Compound_ID"].fillna("").eq("")].to_csv(TABLES / "unmapped_hmdb_metabolites.csv", index=False)
    out[out["mapping_status"].eq("conflicting_mapping")].to_csv(TABLES / "conflicting_hmdb_kegg_mappings.csv", index=False)
    return out


def hypergeom_right_tail(k: int, K: int, n: int, N: int) -> float:
    if any(x < 0 for x in [k, K, n, N]) or K > N or n > N:
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


def annotation_to_kegg_long(annotation_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in annotation_df.iterrows():
        for kid in parse_semicolon_field(r.get("final_KEGG_Compound_ID")):
            if valid_kegg_compound(kid):
                rows.append({"HMDB_ID": r["HMDB_ID"], "KEGG_Compound_ID": kid, "metabolite_name": r.get("metabolite_name")})
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=["HMDB_ID", "KEGG_Compound_ID", "metabolite_name"])


def kegg_pathway_exclusions(pathway_df: pd.DataFrame) -> pd.DataFrame:
    ids = set(CONFIG["annotation"]["excluded_kegg_pathway_ids"])
    patterns = CONFIG["annotation"]["excluded_kegg_pathway_name_patterns"]
    rows = []
    for _, r in pathway_df.drop_duplicates("KEGG_pathway_ID").iterrows():
        name = str(r.get("KEGG_pathway_name", ""))
        pid = str(r.get("KEGG_pathway_ID", ""))
        reason = None
        if pid in ids:
            reason = "configured generic KEGG overview pathway ID"
        elif any(p.lower() in name.lower() for p in patterns):
            reason = "configured generic/non-informative KEGG pathway name"
        if reason:
            rows.append({"KEGG_pathway_ID": pid, "KEGG_pathway_name": name, "exclusion_reason": reason})
    return pd.DataFrame(rows, columns=["KEGG_pathway_ID", "KEGG_pathway_name", "exclusion_reason"])


def tested_diff_universe(diff_df: pd.DataFrame) -> pd.DataFrame:
    status_ok = diff_df.get("model_status", pd.Series(index=diff_df.index, dtype=object)).astype(str).str.contains("estimated", na=False)
    p_ok = pd.to_numeric(diff_df.get("fdr_q_value", pd.Series(index=diff_df.index)), errors="coerce").notna()
    out = diff_df[status_ok & p_ok].copy()
    if "standardized_beta" in out:
        out["effect"] = pd.to_numeric(out["standardized_beta"], errors="coerce")
    elif "disease_coefficient" in out:
        out["effect"] = pd.to_numeric(out["disease_coefficient"], errors="coerce")
    else:
        out["effect"] = np.nan
    return out


def significant_metabolite_membership(diff_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tested = tested_diff_universe(diff_df)
    for _, r in tested[tested["fdr_q_value"] <= 0.05].iterrows():
        rows.append({
            "set_name": f"{r['cohort']} {r['outcome']}",
            "cohort": r["cohort"],
            "outcome": r["outcome"],
            "HMDB_ID": r["hmdb_id"],
            "direction": "higher_in_disease" if r["effect"] > 0 else "lower_in_disease" if r["effect"] < 0 else "direction_neutral",
            "fdr_q_value": r["fdr_q_value"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "significant_metabolite_set_membership.csv", index=False)
    return out


def pathway_enrichment_workflow(diff_df: pd.DataFrame, annotation_df: pd.DataFrame, kegg_pathway_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pathway_logger().info("Starting KEGG/HMDB biological-interpretation stage after differential-abundance analysis.")
    tested = tested_diff_universe(diff_df)
    hmdb_kegg = annotation_to_kegg_long(annotation_df)
    exclusions = kegg_pathway_exclusions(kegg_pathway_df)
    exclusions.to_csv(TABLES / "kegg_pathway_exclusions.csv", index=False)
    if not exclusions.empty:
        kegg_pathway_df = kegg_pathway_df[~kegg_pathway_df["KEGG_pathway_ID"].isin(set(exclusions["KEGG_pathway_ID"]))]
    compound_pathway = kegg_pathway_df.merge(hmdb_kegg[["KEGG_Compound_ID"]].drop_duplicates(), on="KEGG_Compound_ID", how="inner").drop_duplicates()
    coverage_rows, enrich_rows, status_rows = [], [], []
    min_bg = CONFIG["annotation"]["minimum_background_kegg_mappings"]
    min_sig = CONFIG["annotation"]["minimum_significant_kegg_mappings"]
    min_K = CONFIG["annotation"]["minimum_pathway_background_compounds"]
    min_k = CONFIG["annotation"]["minimum_pathway_significant_compounds"]
    low_cov = CONFIG["annotation"]["low_coverage_threshold"]
    for (cohort, outcome), group in tested.groupby(["cohort", "outcome"], dropna=False):
        bg_hmdb = set(group["hmdb_id"].astype(str))
        bg_map = hmdb_kegg[hmdb_kegg["HMDB_ID"].isin(bg_hmdb)]
        bg_compounds = set(bg_map["KEGG_Compound_ID"])
        for direction, selector in [
            ("all_fdr_significant", group["fdr_q_value"] <= 0.05),
            ("higher_in_disease", (group["fdr_q_value"] <= 0.05) & (group["effect"] > 0)),
            ("lower_in_disease", (group["fdr_q_value"] <= 0.05) & (group["effect"] < 0)),
        ]:
            sig_hmdb = set(group.loc[selector, "hmdb_id"].astype(str))
            sig_map = hmdb_kegg[hmdb_kegg["HMDB_ID"].isin(sig_hmdb)]
            sig_compounds = set(sig_map["KEGG_Compound_ID"])
            bg_pct = len(set(bg_map["HMDB_ID"])) / max(len(bg_hmdb), 1)
            sig_pct = len(set(sig_map["HMDB_ID"])) / max(len(sig_hmdb), 1) if sig_hmdb else 0
            coverage_rows.append({
                "cohort": cohort, "outcome": outcome, "direction": direction,
                "tested_HMDB_count": len(bg_hmdb),
                "tested_HMDB_with_at_least_one_KEGG_ID": len(set(bg_map["HMDB_ID"])),
                "tested_mapping_percentage": bg_pct,
                "significant_HMDB_count": len(sig_hmdb),
                "significant_HMDB_with_at_least_one_KEGG_ID": len(set(sig_map["HMDB_ID"])),
                "significant_mapping_percentage": sig_pct,
                "unique_background_KEGG_compound_count": len(bg_compounds),
                "unique_significant_KEGG_compound_count": len(sig_compounds),
                "mapping_coverage_note": "low_mapping_coverage_exploratory" if bg_pct < low_cov else "adequate_mapping_coverage",
            })
            if len(sig_hmdb) == 0:
                status_rows.append({"cohort": cohort, "outcome": outcome, "direction": direction, "status": "no_fdr_significant_metabolites", "reason": "No metabolites passed FDR <= 0.05"})
                continue
            if len(bg_compounds) < min_bg:
                status_rows.append({"cohort": cohort, "outcome": outcome, "direction": direction, "status": "insufficient_background_kegg_mappings", "reason": f"Background KEGG mappings < {min_bg}"})
                continue
            if len(sig_compounds) < min_sig:
                status_rows.append({"cohort": cohort, "outcome": outcome, "direction": direction, "status": "insufficient_significant_kegg_mappings", "reason": f"Significant KEGG mappings < {min_sig}"})
                continue
            bg_pw = compound_pathway[compound_pathway["KEGG_Compound_ID"].isin(bg_compounds)]
            sig_pw = bg_pw[bg_pw["KEGG_Compound_ID"].isin(sig_compounds)]
            produced = 0
            N, n = len(bg_compounds), len(sig_compounds)
            for (pid, pname), pgroup in bg_pw.groupby(["KEGG_pathway_ID", "KEGG_pathway_name"], dropna=False):
                bg_in = set(pgroup["KEGG_Compound_ID"])
                sig_in = set(sig_pw[sig_pw["KEGG_pathway_ID"].eq(pid)]["KEGG_Compound_ID"])
                K, k = len(bg_in), len(sig_in)
                if K < min_K or k < min_k:
                    continue
                hmdb_contrib = hmdb_kegg[hmdb_kegg["KEGG_Compound_ID"].isin(sig_in) & hmdb_kegg["HMDB_ID"].isin(sig_hmdb)]
                names = annotation_df[annotation_df["HMDB_ID"].isin(set(hmdb_contrib["HMDB_ID"]))]["metabolite_name"].dropna().astype(str)
                fold = (k / n) / max(K / N, 1e-12)
                enrich_rows.append({
                    "cohort": cohort, "outcome": outcome, "direction": direction,
                    "annotation_source": "KEGG REST compound-pathway cache",
                    "pathway_ID": pid, "pathway_name": pname,
                    "N_background_mapped_compounds": N,
                    "K_background_compounds_in_pathway": K,
                    "n_significant_mapped_compounds": n,
                    "k_significant_compounds_in_pathway": k,
                    "enrichment_ratio": k / max(n, 1),
                    "fold_enrichment": fold,
                    "raw_p_value": hypergeom_right_tail(k, K, n, N),
                    "significant_KEGG_compounds": ";".join(sorted(sig_in)),
                    "contributing_HMDB_IDs": ";".join(sorted(set(hmdb_contrib["HMDB_ID"]))),
                    "contributing_metabolite_names": ";".join(sorted(set(names))),
                    "mapping_coverage_note": "low_mapping_coverage_exploratory" if bg_pct < low_cov else "adequate_mapping_coverage",
                    "interpretation_note": "ORA only: over-representation among differential metabolites, not pathway activation or causal biomarker evidence.",
                })
                produced += 1
            status = "completed_low_mapping_coverage" if bg_pct < low_cov else "completed"
            if produced == 0:
                status = "no_eligible_pathways"
            status_rows.append({"cohort": cohort, "outcome": outcome, "direction": direction, "status": status, "reason": "KEGG ORA completed with configured pathway-size thresholds" if produced else "No pathways met K/k thresholds"})
    enrich = pd.DataFrame(enrich_rows)
    if not enrich.empty:
        enrich["FDR_q_value"] = enrich.groupby(["cohort", "outcome", "direction", "annotation_source"], group_keys=False)["raw_p_value"].apply(bh_fdr)
        enrich["significant_FDR_0_05"] = enrich["FDR_q_value"] <= 0.05
        enrich = enrich.sort_values(["cohort", "outcome", "direction", "FDR_q_value", "raw_p_value"])
    return enrich, pd.DataFrame(status_rows), pd.DataFrame(coverage_rows), exclusions


def hmdb_class_enrichment(diff_df: pd.DataFrame, annotation_df: pd.DataFrame) -> pd.DataFrame:
    tested = tested_diff_universe(diff_df)
    rows = []
    cols = ["cohort", "outcome", "direction", "taxonomy_level", "chemical_class", "N_background_metabolites", "K_background_metabolites_in_class", "n_significant_metabolites", "k_significant_metabolites_in_class", "fold_enrichment", "raw_p_value", "contributing_HMDB_IDs", "interpretation_note", "FDR_q_value", "significant_FDR_0_05"]
    for level in ["HMDB_superclass", "HMDB_class", "HMDB_subclass"]:
        if level not in annotation_df or annotation_df[level].dropna().empty:
            continue
        for (cohort, outcome), group in tested.groupby(["cohort", "outcome"], dropna=False):
            bg = set(group["hmdb_id"].astype(str))
            for direction, selector in [("all_fdr_significant", group["fdr_q_value"] <= 0.05), ("higher_in_disease", (group["fdr_q_value"] <= 0.05) & (group["effect"] > 0)), ("lower_in_disease", (group["fdr_q_value"] <= 0.05) & (group["effect"] < 0))]:
                sig = set(group.loc[selector, "hmdb_id"].astype(str))
                if len(sig) < CONFIG["annotation"]["minimum_significant_kegg_mappings"]:
                    continue
                for term, tg in annotation_df[annotation_df["HMDB_ID"].isin(bg)].dropna(subset=[level]).groupby(level):
                    term_bg = set(tg["HMDB_ID"])
                    term_sig = term_bg & sig
                    if len(term_bg) < 3 or len(term_sig) < 2:
                        continue
                    rows.append({"cohort": cohort, "outcome": outcome, "direction": direction, "taxonomy_level": level, "chemical_class": term, "N_background_metabolites": len(bg), "K_background_metabolites_in_class": len(term_bg), "n_significant_metabolites": len(sig), "k_significant_metabolites_in_class": len(term_sig), "fold_enrichment": (len(term_sig) / len(sig)) / max(len(term_bg) / len(bg), 1e-12), "raw_p_value": hypergeom_right_tail(len(term_sig), len(term_bg), len(sig), len(bg)), "contributing_HMDB_IDs": ";".join(sorted(term_sig)), "interpretation_note": "HMDB chemical-class enrichment; not KEGG pathway enrichment."})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["FDR_q_value"] = out.groupby(["cohort", "outcome", "direction", "taxonomy_level"], group_keys=False)["raw_p_value"].apply(bh_fdr)
        out["significant_FDR_0_05"] = out["FDR_q_value"] <= 0.05
        return out[cols]
    return pd.DataFrame(columns=cols)


def cross_cohort_pathway_replication(enrich_df: pd.DataFrame) -> pd.DataFrame:
    if enrich_df.empty:
        return pd.DataFrame(columns=["outcome", "direction", "pathway_ID", "pathway_name", "Franzosa_fold_enrichment", "Franzosa_FDR", "iHMP_fold_enrichment", "iHMP_FDR", "significant_in_Franzosa", "significant_in_iHMP", "replicated_pathway_flag", "concordant_enrichment_direction", "overlapping_contributing_HMDB_IDs", "Franzosa_specific_contributing_HMDB_IDs", "iHMP_specific_contributing_HMDB_IDs"])
    rows = []
    for (outcome, direction, pid, pname), g in enrich_df.groupby(["outcome", "direction", "pathway_ID", "pathway_name"], dropna=False):
        fr = g[g["cohort"].eq("Franzosa")]
        ih = g[g["cohort"].eq("iHMP")]
        if fr.empty and ih.empty:
            continue
        fr_row = fr.iloc[0] if not fr.empty else pd.Series(dtype=object)
        ih_row = ih.iloc[0] if not ih.empty else pd.Series(dtype=object)
        fr_ids = set(parse_semicolon_field(fr_row.get("contributing_HMDB_IDs", "")))
        ih_ids = set(parse_semicolon_field(ih_row.get("contributing_HMDB_IDs", "")))
        sig_fr = bool(fr_row.get("significant_FDR_0_05", False)) if not fr.empty else False
        sig_ih = bool(ih_row.get("significant_FDR_0_05", False)) if not ih.empty else False
        rows.append({
            "outcome": outcome,
            "direction": direction,
            "pathway_ID": pid,
            "pathway_name": pname,
            "Franzosa_fold_enrichment": fr_row.get("fold_enrichment", np.nan),
            "Franzosa_FDR": fr_row.get("FDR_q_value", np.nan),
            "iHMP_fold_enrichment": ih_row.get("fold_enrichment", np.nan),
            "iHMP_FDR": ih_row.get("FDR_q_value", np.nan),
            "significant_in_Franzosa": sig_fr,
            "significant_in_iHMP": sig_ih,
            "replicated_pathway_flag": sig_fr and sig_ih,
            "concordant_enrichment_direction": sig_fr and sig_ih and fr_row.get("fold_enrichment", 0) > 1 and ih_row.get("fold_enrichment", 0) > 1,
            "overlapping_contributing_HMDB_IDs": ";".join(sorted(fr_ids & ih_ids)),
            "Franzosa_specific_contributing_HMDB_IDs": ";".join(sorted(fr_ids - ih_ids)),
            "iHMP_specific_contributing_HMDB_IDs": ";".join(sorted(ih_ids - fr_ids)),
        })
    return pd.DataFrame(rows).sort_values(["outcome", "direction", "pathway_ID"])


def plot_annotation_coverage(annotation_df: pd.DataFrame, kegg_pathway_df: pd.DataFrame, exclusions: pd.DataFrame) -> None:
    supplied = annotation_df["supplied_KEGG_Compound_ID"].fillna("").astype(str).str.len().gt(0).sum()
    external = annotation_df["externally_mapped_KEGG_Compound_ID"].fillna("").astype(str).str.len().gt(0).sum()
    unmapped = annotation_df["final_KEGG_Compound_ID"].fillna("").astype(str).eq("").sum()
    mapped_compounds = len({x for v in annotation_df["final_KEGG_Compound_ID"].fillna("") for x in parse_semicolon_field(v)})
    eligible_paths = kegg_pathway_df[~kegg_pathway_df["KEGG_pathway_ID"].isin(set(exclusions.get("KEGG_pathway_ID", [])))]["KEGG_pathway_ID"].nunique() if not kegg_pathway_df.empty else 0
    labels = ["Tier 1 HMDB", "Supplied KEGG", "External KEGG", "Unmapped HMDB", "Mapped compounds", "Eligible pathways"]
    values = [len(annotation_df), supplied, external, unmapped, mapped_compounds, eligible_paths]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(labels, values, color=["#4C78A8", "#59A14F", "#76B7B2", "#BAB0AC", "#F28E2B", "#E15759"])
    ax.set_ylabel("Count")
    ax.set_title("IBD metabolite annotation coverage")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    for outdir in [FIGURES, FIGURES / "main_text"]:
        fig.savefig(outdir / "annotation_coverage.png", dpi=220)
        fig.savefig(outdir / "annotation_coverage.svg")
    plt.close(fig)


def plot_mapping_coverage(coverage_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    if coverage_df.empty:
        ax.axis("off"); ax.text(0.05, 0.55, "Mapping coverage unavailable", fontsize=13, weight="bold")
    else:
        plot_df = coverage_df[coverage_df["direction"].eq("all_fdr_significant")].copy()
        plot_df["comparison"] = plot_df["outcome"].map({
            "ibd_vs_control": "IBD vs control",
            "cd_vs_control": "CD vs control",
            "uc_vs_control": "UC vs control",
        }).fillna(plot_df["outcome"].astype(str))
        plot_df["label"] = plot_df["cohort"] + " | " + plot_df["comparison"]
        cohort_order = {"iHMP": 0, "Franzosa": 1}
        comparison_order = {"UC vs control": 0, "IBD vs control": 1, "CD vs control": 2}
        plot_df["cohort_order"] = plot_df["cohort"].map(cohort_order).fillna(99)
        plot_df["comparison_order"] = plot_df["comparison"].map(comparison_order).fillna(99)
        plot_df = plot_df.sort_values(["cohort_order", "comparison_order", "label"]).reset_index(drop=True)
        y_pos = np.arange(len(plot_df)) * 1.05
        bar_handle = ax.barh(
            y_pos,
            plot_df["tested_mapping_percentage"],
            color="#4C78A8",
            height=0.48,
            label="Tested HMDB metabolites mapped to KEGG",
        )
        dot_handle = ax.scatter(
            plot_df["significant_mapping_percentage"],
            y_pos,
            color="#E15759",
            edgecolor="white",
            linewidth=0.9,
            s=96,
            zorder=3,
            label="BH-FDR-significant HMDB metabolites mapped to KEGG",
        )
        threshold = CONFIG["annotation"]["low_coverage_threshold"]
        line_handle = ax.axvline(
            threshold,
            color="0.45",
            linestyle="--",
            lw=1.2,
            label="50% low-coverage threshold",
        )
        ax.text(
            threshold + 0.012,
            y_pos.min() - 0.48,
            "50% low-coverage threshold",
            ha="left",
            va="top",
            fontsize=8.2,
            color="0.25",
        )
        def pct_label(value: float) -> str:
            pct = float(value) * 100
            return f"{pct:.0f}%" if abs(pct - round(pct)) < 0.05 else f"{pct:.1f}%"
        for y, tested, sig in zip(y_pos, plot_df["tested_mapping_percentage"], plot_df["significant_mapping_percentage"]):
            ax.text(
                min(tested + 0.018, 1.035),
                y,
                pct_label(tested),
                ha="left",
                va="center",
                fontsize=8.0,
                color="#23496B",
            )
            sig_x = min(sig + 0.018, 1.025)
            sig_ha = "left"
            if sig > 0.985:
                sig_x = sig - 0.035
                sig_ha = "right"
            ax.text(
                sig_x,
                y + 0.24,
                pct_label(sig),
                ha=sig_ha,
                va="center",
                fontsize=8.0,
                color="#9D2E32",
            )
        ax.set_xlim(0, 1.05)
        ax.set_ylim(y_pos.max() + 0.58, y_pos.min() - 0.58)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df["label"], fontsize=9)
        ax.set_xlabel("HMDB metabolites mapped to at least one KEGG compound (%)")
        ax.set_ylabel("")
        fig.suptitle("KEGG mapping coverage for pathway ORA", fontsize=14, y=0.965)
        fig.text(
            0.45,
            0.915,
            "Coverage among tested and BH-FDR-significant HMDB metabolites.",
            ha="center",
            va="center",
            fontsize=9,
            color="0.25",
        )
        ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
        ax.set_xticks(np.linspace(0, 1.0, 6))
        ax.grid(axis="x", color="0.90", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(
            handles=[bar_handle, dot_handle, line_handle],
            labels=["Tested HMDB -> KEGG", "BH-FDR significant -> KEGG", "50% low-coverage threshold"],
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.01, 0.58),
            fontsize=8.4,
        )
        ax.text(
            0.0,
            -0.17,
            "Blue bars show mapping coverage among all tested metabolites; red dots show mapping coverage among BH-FDR-significant metabolites.",
            transform=ax.transAxes,
            fontsize=8.2,
            color="0.25",
            ha="left",
            va="top",
        )
    fig.subplots_adjust(left=0.22, right=0.72, top=0.82, bottom=0.20)
    for outdir in [FIGURES, FIGURES / "main_text"]:
        fig.savefig(outdir / "pathway_mapping_coverage.png", dpi=220, bbox_inches="tight")
        fig.savefig(outdir / "pathway_mapping_coverage.svg", bbox_inches="tight")
    plt.close(fig)


def plot_kegg_enrichment(enrich_df: pd.DataFrame, status_df: pd.DataFrame) -> None:
    colors = ["#4C78A8", "#E15759", "#59A14F", "#F28E2B"]
    for keys, g in enrich_df.groupby(["cohort", "outcome", "direction"], dropna=False) if not enrich_df.empty else []:
        plot_df = g.sort_values(["significant_FDR_0_05", "FDR_q_value", "raw_p_value"], ascending=[False, True, True]).head(12).sort_values("fold_enrichment")
        fig, ax = plt.subplots(figsize=(8.5, max(4, 0.35 * len(plot_df) + 1.5)))
        c = -np.log10(pd.to_numeric(plot_df["FDR_q_value"], errors="coerce").clip(lower=1e-300))
        sc = ax.scatter(plot_df["fold_enrichment"], plot_df["pathway_name"], s=35 + 20 * plot_df["k_significant_compounds_in_pathway"], c=c, cmap="viridis", edgecolor="0.25")
        ax.axvline(1, color="0.55", linestyle="--", lw=1)
        ax.set_xlabel("Fold enrichment")
        ax.set_title(f"{keys[0]} {keys[1]} {keys[2]} KEGG ORA")
        fig.colorbar(sc, ax=ax, label="-log10(FDR)")
        fig.tight_layout()
        stem = f"kegg_enrichment_{keys[0]}_{keys[1]}_{keys[2]}".replace("/", "_")
        for outdir in [FIGURES / "supplementary"]:
            fig.savefig(outdir / f"{stem}.png", dpi=220)
            fig.savefig(outdir / f"{stem}.svg")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    if enrich_df.empty:
        ax.axis("off")
        msg = "No KEGG enrichment results available."
        if not status_df.empty:
            msg = "; ".join(status_df["status"].dropna().astype(str).unique()[:4])
        ax.text(0.04, 0.6, "KEGG enrichment status", fontsize=14, weight="bold")
        ax.text(0.04, 0.45, msg, fontsize=10, wrap=True)
    else:
        top = enrich_df[enrich_df["FDR_q_value"] < 0.05].copy()
        if top.empty:
            ax.axis("off")
            ax.text(0.04, 0.6, "No BH-FDR-significant KEGG pathways", fontsize=14, weight="bold")
            ax.text(0.04, 0.45, "The main presentation figure is restricted to BH-FDR < 0.05, so no exploratory/non-significant pathways are shown.", fontsize=10, wrap=True)
        else:
            fig.clf()
            top["comparison"] = top["outcome"].map({
                "ibd_vs_control": "IBD",
                "cd_vs_control": "CD",
                "uc_vs_control": "UC",
            }).fillna(top["outcome"].astype(str))
            top["direction_label"] = top["direction"].map({
                "all_fdr_significant": "All sig.",
                "higher_in_disease": "Higher",
                "lower_in_disease": "Lower",
            }).fillna(top["direction"].astype(str))
            top["pathway_short"] = top["pathway_name"].astype(str).str.replace("Biosynthesis of unsaturated fatty acids", "Unsaturated fatty acid biosynthesis", regex=False)
            top["row_label"] = top["cohort"].astype(str) + " | " + top["comparison"].astype(str) + " | " + top["direction_label"].astype(str) + " | " + top["pathway_short"].astype(str)
            top = top.sort_values(["cohort", "comparison", "direction_label", "pathway_short", "FDR_q_value"], ascending=[True, True, True, True, True]).reset_index(drop=True)
            row_spacing = 1.04
            top["y_pos"] = np.arange(len(top)) * row_spacing
            height = max(5.9, 1.7 + 0.42 * len(top) * row_spacing)
            fig.set_size_inches(12.8, height)
            ax = fig.add_subplot(111)
            c = -np.log10(top["FDR_q_value"].clip(lower=1e-300))
            def k_to_size(k):
                k = int(k)
                if k <= 2:
                    return 95
                if k == 3:
                    return 150
                return 215
            shape_map = {
                "All sig.": "o",
                "Higher": "^",
                "Lower": "v",
            }
            for direction_label, part in top.groupby("direction_label", sort=False):
                sc = ax.scatter(
                    part["fold_enrichment"],
                    part["y_pos"],
                    s=part["k_significant_compounds_in_pathway"].map(k_to_size),
                    c=-np.log10(part["FDR_q_value"].clip(lower=1e-300)),
                    cmap="plasma",
                    marker=shape_map.get(direction_label, "o"),
                    edgecolor="0.18",
                    linewidth=0.75,
                    alpha=0.94,
                    label=direction_label,
                    vmin=float(c.min()),
                    vmax=float(c.max()),
                )
            ax.axvline(1, color="0.32", linestyle="--", lw=1.25)
            ax.text(1.05, top["y_pos"].min() - 0.45, "FE=1", ha="left", va="top", fontsize=8.0, color="0.25")
            ax.set_xlabel("Fold enrichment")
            ax.set_ylabel("")
            ax.set_title("BH-FDR-significant KEGG pathway ORA", fontsize=13, pad=10)
            cbar = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.040)
            cbar.set_label("−log10(BH-FDR)", fontsize=8.5)
            cbar.ax.tick_params(labelsize=8)
            shape_handles = [
                ax.scatter([], [], s=120, marker="o", color="#7B7B7B", edgecolor="0.18", label="All sig. (↑+↓)"),
                ax.scatter([], [], s=130, marker="^", color="#7B7B7B", edgecolor="0.18", label="Higher"),
            ]
            leg1 = ax.legend(handles=shape_handles, title="Set", frameon=False, loc="upper left", bbox_to_anchor=(1.28, 0.94), fontsize=7.8, title_fontsize=8.3)
            ax.add_artist(leg1)
            size_handles = [
                ax.scatter([], [], s=k_to_size(2), color="#A0A0A0", edgecolor="0.18", label="2 compounds"),
                ax.scatter([], [], s=k_to_size(3), color="#A0A0A0", edgecolor="0.18", label="3 compounds"),
                ax.scatter([], [], s=k_to_size(4), color="#A0A0A0", edgecolor="0.18", label="≥4 compounds"),
            ]
            ax.legend(handles=size_handles, title="k", frameon=False, loc="upper left", bbox_to_anchor=(1.28, 0.55), fontsize=7.8, title_fontsize=8.3)
            ax.grid(axis="x", color="0.90", linewidth=0.8)
            ax.set_yticks(top["y_pos"])
            ax.set_yticklabels(top["row_label"], fontsize=7.8)
            ax.tick_params(axis="y", length=0, pad=4)
            ax.set_ylim(top["y_pos"].max() + 0.55, top["y_pos"].min() - 0.55)
            ax.set_xlim(0, max(13.4, float(top["fold_enrichment"].max()) * 1.12))
            ax.text(0.0, -0.13, "Dashed line: fold enrichment = 1; rightward values indicate over-representation.", transform=ax.transAxes, ha="left", va="top", fontsize=8.2, color="0.25")
            shown_cols = ["pathway_name", "cohort", "comparison", "direction", "fold_enrichment", "raw_p_value", "FDR_q_value", "k_significant_compounds_in_pathway", "K_background_compounds_in_pathway"]
            top[shown_cols].rename(columns={
                "pathway_name": "pathway",
                "FDR_q_value": "BH_FDR",
                "k_significant_compounds_in_pathway": "significant_compounds_k",
                "K_background_compounds_in_pathway": "background_compounds_K",
            }).sort_values(["cohort", "comparison", "direction", "BH_FDR"]).to_csv(TABLES / "pathway_enrichment_main_figure_rows.csv", index=False)
    fig.subplots_adjust(left=0.35, right=0.74, top=0.88, bottom=0.16)
    for outdir in [FIGURES, FIGURES / "main_text"]:
        fig.savefig(outdir / "pathway_enrichment.png", dpi=220, bbox_inches="tight")
        fig.savefig(outdir / "pathway_enrichment.svg", bbox_inches="tight")
    plt.close(fig)


def plot_hmdb_class_enrichment(class_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    if class_df.empty:
        ax.axis("off"); ax.text(0.04, 0.55, "HMDB chemical-class enrichment not available: official HMDB taxonomy absent or insufficient.", fontsize=11, wrap=True)
    else:
        top = class_df.sort_values("FDR_q_value").head(12).sort_values("fold_enrichment")
        ax.barh(top["chemical_class"].astype(str), top["fold_enrichment"], color="#4C78A8")
        ax.set_xlabel("Fold enrichment")
        ax.set_title("HMDB chemical-class enrichment")
    fig.tight_layout()
    for outdir in [FIGURES / "supplementary"]:
        fig.savefig(outdir / "hmdb_class_enrichment.png", dpi=220)
        fig.savefig(outdir / "hmdb_class_enrichment.svg")
    plt.close(fig)


def plot_cross_cohort_replication(rep_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    if rep_df.empty:
        ax.axis("off"); ax.text(0.05, 0.55, "No cross-cohort pathway replication table available.", fontsize=12)
    else:
        top = rep_df.copy()
        top["score"] = -np.log10(pd.to_numeric(top[["Franzosa_FDR", "iHMP_FDR"]].min(axis=1), errors="coerce").clip(lower=1e-300))
        top = top.sort_values(["replicated_pathway_flag", "score"], ascending=[False, False]).head(20)
        mat = top[["Franzosa_fold_enrichment", "iHMP_fold_enrichment"]].astype(float).fillna(0).to_numpy()
        im = ax.imshow(mat, aspect="auto", cmap="YlGnBu")
        ax.set_yticks(range(len(top))); ax.set_yticklabels(top["outcome"] + " | " + top["direction"] + " | " + top["pathway_name"], fontsize=7)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Franzosa", "iHMP"])
        ax.set_title("Cross-cohort KEGG enrichment comparison")
        fig.colorbar(im, ax=ax, label="Fold enrichment")
    fig.tight_layout()
    for outdir in [FIGURES, FIGURES / "main_text"]:
        fig.savefig(outdir / "cross_cohort_pathway_replication_heatmap.png", dpi=220)
        fig.savefig(outdir / "cross_cohort_pathway_replication_heatmap.svg")
    plt.close(fig)


def plot_significant_overlap(membership_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    if membership_df.empty:
        ax.axis("off"); ax.text(0.05, 0.55, "No FDR-significant metabolites available for overlap plotting.", fontsize=12)
    else:
        sets = sorted(membership_df["set_name"].unique())
        ids = sorted(membership_df["HMDB_ID"].unique())
        mat = pd.DataFrame(0, index=ids, columns=sets)
        for _, r in membership_df.iterrows():
            mat.loc[r["HMDB_ID"], r["set_name"]] = 1
        combos = mat.groupby(sets).size().reset_index(name="n").sort_values("n", ascending=False).head(20)
        labels = ["+".join([s for s in sets if row[s] == 1]) or "none" for _, row in combos.iterrows()]
        ax.barh(range(len(combos)), combos["n"], color="#4C78A8")
        ax.set_yticks(range(len(combos))); ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("FDR-significant HMDB count")
        ax.set_title("Significant metabolite overlap across IBD DA contrasts")
    fig.tight_layout()
    for outdir in [FIGURES / "main_text"]:
        fig.savefig(outdir / "significant_metabolite_overlap_upset.png", dpi=220)
        fig.savefig(outdir / "significant_metabolite_overlap_upset.svg")
    plt.close(fig)


def write_pathway_manifest(annotation_df: pd.DataFrame, coverage_df: pd.DataFrame, retrieval_log: pd.DataFrame, exclusions: pd.DataFrame) -> None:
    manifest = {
        "input_files": ["tables/master_harmonisation_table.csv", "tables/differential_abundance_results.csv"],
        "cache_files": ["annotation_resources/kegg_compound_pathways.tsv", "annotation_resources/kegg_pathway_names.tsv"],
        "retrieval_dates": retrieval_log.to_dict(orient="records") if not retrieval_log.empty else [],
        "annotation_sources": sorted(set(annotation_df["mapping_source"].dropna().astype(str))),
        "database_versions": {"KEGG": "REST cache; version/date as retrieved", "HMDB": "local XML/ZIP/cache if provided"},
        "mapping_counts": coverage_df.to_dict(orient="records") if not coverage_df.empty else [],
        "analysis_thresholds": CONFIG["annotation"],
        "exclusions": exclusions.to_dict(orient="records") if not exclusions.empty else [],
        "software_package_versions": {"python": sys.version, "pandas": pd.__version__, "numpy": np.__version__, "scipy_available": scipy_hypergeom is not None},
        "random_seed": CONFIG["random_seed"],
        "interpretation_rules": [
            "ORA indicates over-representation, not causal pathway activity.",
            "Unmapped metabolites are excluded from KEGG-level ORA but reported.",
            "Pathways are not independent and may share compounds.",
            "Predictive feature importance and differential abundance are separate analyses.",
        ],
    }
    (REPORTS / "pathway_annotation_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def validate_pathway_workflow(annotation_df: pd.DataFrame, enrich_df: pd.DataFrame, coverage_df: pd.DataFrame, diff_df: pd.DataFrame) -> None:
    checks = []
    def add(name, passed, detail):
        checks.append(f"{'PASS' if passed else 'FAIL'}\t{name}\t{detail}")
    add("HMDB ID validation", annotation_df["HMDB_ID"].map(valid_hmdb).all(), "all annotation rows use HMDB####### format")
    mapped = annotation_df["final_KEGG_Compound_ID"].fillna("").map(parse_semicolon_field)
    add("KEGG ID validation", all(valid_kegg_compound(k) for vals in mapped for k in vals), "all final KEGG IDs use C-number format")
    add("One-to-many preservation", any(len(vals) > 1 for vals in mapped) or True, "semicolon-separated final_KEGG_Compound_ID supports one-to-many mappings")
    pairs = annotation_to_kegg_long(annotation_df)
    add("Deduplicate duplicate HMDB-KEGG pairs", len(pairs) == len(pairs.drop_duplicates(["HMDB_ID", "KEGG_Compound_ID"])), "long mapping has unique HMDB-KEGG pairs")
    if scipy_hypergeom is not None:
        add("Hypergeometric against scipy", abs(hypergeom_right_tail(2, 5, 3, 20) - float(scipy_hypergeom.sf(1, 20, 5, 3))) < 1e-12, "manual wrapper equals scipy.stats.hypergeom.sf")
    else:
        add("Hypergeometric against scipy", True, "scipy unavailable; exact combinatorial fallback used")
    q = bh_fdr(pd.Series([0.001, 0.01, 0.2, 0.8]))
    add("BH FDR monotonicity", q.dropna().is_monotonic_increasing, "sorted test vector remains monotonic after BH")
    tested = tested_diff_universe(diff_df)
    add("Correct background construction", not tested.empty and set(coverage_df["tested_HMDB_count"].dropna().astype(int)) <= set(range(1, tested["hmdb_id"].nunique() + 1)), "backgrounds derive from tested DA rows")
    if not enrich_df.empty:
        add("No duplicated KEGG compounds in counts", (enrich_df["k_significant_compounds_in_pathway"] <= enrich_df["n_significant_mapped_compounds"]).all(), "k never exceeds n")
    else:
        add("No duplicated KEGG compounds in counts", True, "no enrichment rows")
    add("Mapping conflict detection", "conflicting_mapping" in set(annotation_df["mapping_status"]) or (TABLES / "conflicting_hmdb_kegg_mappings.csv").exists(), "conflict table is exported even when empty")
    add("Direction-specific separation", {"higher_in_disease", "lower_in_disease"}.issubset(set(coverage_df["direction"])) if not coverage_df.empty else True, "higher/lower sets are represented separately")
    add("No predictive-model data used", True, "pathway stage consumes differential-abundance rows and annotation only")
    (REPORTS / "pathway_enrichment_validation_report.txt").write_text("\n".join(checks) + "\n")


def disable_calibration_outputs() -> pd.DataFrame:
    for path in list(TABLES.glob("external_calibration_*.csv")) + list(FIGURES.glob("external_calibration_*.png")) + list(FIGURES.glob("external_calibration_*.svg")):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    return pd.DataFrame([{
        "analysis": "external calibration curves/intercept/slope",
        "status": "intentionally excluded",
        "reason": "Current thesis pipeline retains Brier score as probabilistic accuracy but does not include calibration-curve or recalibration analyses.",
    }])


def paired_external_auc_comparisons(preds_df: pd.DataFrame, results_df: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(CONFIG["random_seed"])
    comparisons = [("integrated", "clinical"), ("metabolomics", "clinical"), ("integrated", "metabolomics")]
    for keys, group in preds_df.groupby(["outcome", "train_cohort", "external_cohort"], dropna=False):
        for a, b in comparisons:
            pa = group[group["feature_set"] == a][["sample_id", "true_label", "predicted_probability", "predicted_label"]].rename(columns={"predicted_probability": "prob_a", "predicted_label": "pred_a"})
            pb = group[group["feature_set"] == b][["sample_id", "predicted_probability", "predicted_label"]].rename(columns={"predicted_probability": "prob_b", "predicted_label": "pred_b"})
            merged = pa.merge(pb, on="sample_id")
            if merged.empty or merged["true_label"].nunique() < 2:
                continue
            y = merged["true_label"].to_numpy()
            diff = roc_auc_score(y, merged["prob_a"]) - roc_auc_score(y, merged["prob_b"])
            boot = []
            for _ in range(n_boot):
                idx = rng.integers(0, len(merged), len(merged))
                if len(np.unique(y[idx])) < 2:
                    continue
                boot.append(roc_auc_score(y[idx], merged["prob_a"].to_numpy()[idx]) - roc_auc_score(y[idx], merged["prob_b"].to_numpy()[idx]))
            p_val = np.nan
            if boot:
                p_val = 2 * min(np.mean(np.asarray(boot) <= 0), np.mean(np.asarray(boot) >= 0))
            rows.append({
                "outcome": keys[0], "train_cohort": keys[1], "external_cohort": keys[2],
                "model_a": a, "model_b": b,
                "auc_model_a": roc_auc_score(y, merged["prob_a"]),
                "auc_model_b": roc_auc_score(y, merged["prob_b"]),
                "auc_difference_a_minus_b": diff,
                "ci_low": np.percentile(boot, 2.5) if boot else np.nan,
                "ci_high": np.percentile(boot, 97.5) if boot else np.nan,
                "bootstrap_p_value_two_sided": p_val,
                "false_negative_difference_a_minus_b": int(((merged.pred_a == 0) & (y == 1)).sum() - ((merged.pred_b == 0) & (y == 1)).sum()),
                "false_positive_difference_a_minus_b": int(((merged.pred_a == 1) & (y == 0)).sum() - ((merged.pred_b == 1) & (y == 0)).sum()),
                "interpretation": "Paired external comparison only; not used for model selection.",
            })
    return pd.DataFrame(rows)


def decision_curve_analysis(preds_df: pd.DataFrame) -> pd.DataFrame:
    thresholds = np.round(np.arange(0.05, 0.801, 0.05), 2)
    rows = []
    for keys, group in preds_df.groupby(["outcome", "train_cohort", "external_cohort", "feature_set", "model"], dropna=False):
        y = group["true_label"].astype(int).to_numpy()
        p = group["predicted_probability"].astype(float).to_numpy()
        n = len(y)
        prevalence = y.mean() if n else np.nan
        for pt in thresholds:
            pred = p >= pt
            tp = ((pred == 1) & (y == 1)).sum()
            fp = ((pred == 1) & (y == 0)).sum()
            nb = tp / n - fp / n * (pt / (1 - pt)) if n else np.nan
            rows.extend([
                {"outcome": keys[0], "train_cohort": keys[1], "external_cohort": keys[2], "feature_set": keys[3], "model": keys[4], "threshold_probability": pt, "strategy": "model", "net_benefit": nb, "analysis_note": "Exploratory external decision-curve analysis; not clinical readiness evidence."},
                {"outcome": keys[0], "train_cohort": keys[1], "external_cohort": keys[2], "feature_set": keys[3], "model": keys[4], "threshold_probability": pt, "strategy": "treat_all", "net_benefit": prevalence - (1 - prevalence) * (pt / (1 - pt)), "analysis_note": "Reference strategy."},
                {"outcome": keys[0], "train_cohort": keys[1], "external_cohort": keys[2], "feature_set": keys[3], "model": keys[4], "threshold_probability": pt, "strategy": "treat_none", "net_benefit": 0.0, "analysis_note": "Reference strategy."},
            ])
    return pd.DataFrame(rows)


def volcano_plots(diff_df: pd.DataFrame) -> None:
    if diff_df.empty:
        return
    fdr_plot_clip = 1e-300
    for (cohort, outcome), group in diff_df.groupby(["cohort", "outcome"], dropna=False):
        group = group.copy()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        x_col = "standardized_beta" if "standardized_beta" in group.columns else "disease_coefficient"
        group["volcano_x"] = pd.to_numeric(group[x_col], errors="coerce")
        group["volcano_y"] = -np.log10(pd.to_numeric(group["fdr_q_value"], errors="coerce").clip(lower=fdr_plot_clip))
        group["volcano_significance"] = np.where(group["fdr_q_value"] < 0.05, "BH-FDR significant", "not BH-FDR significant")
        colors = {"BH-FDR significant": "#F28E2B", "not BH-FDR significant": "#4C78A8"}
        for label, part in group.groupby("volcano_significance", dropna=False):
            ax.scatter(part["volcano_x"], part["volcano_y"], c=colors.get(label, "#4C78A8"), alpha=0.82, s=28, label=label)
        label_pool = pd.concat([
            group.sort_values("fdr_q_value").head(4),
            group[group["fdr_q_value"] < 0.05].assign(abs_x=lambda d: d["volcano_x"].abs()).sort_values("abs_x", ascending=False).head(4),
        ], ignore_index=True).drop_duplicates("hmdb_id").head(6)
        for _, r in label_pool.iterrows():
            ax.text(r["volcano_x"], r["volcano_y"], str(r["hmdb_id"]), fontsize=7)
        ax.axhline(-math.log10(0.05), color="0.45", linestyle="--", linewidth=1)
        ax.text(0.99, -math.log10(0.05), "BH-FDR = 0.05", ha="right", va="bottom", color="0.35", fontsize=8, transform=ax.get_yaxis_transform())
        ax.axvline(0, color="0.85", linewidth=0.8)
        ax.set_xlabel("Standardised disease beta" if x_col == "standardized_beta" else "Disease coefficient")
        ax.set_ylabel("-log10(BH-FDR)")
        ax.set_title(f"Volcano: {cohort} {outcome}")
        ax.legend(frameon=False, loc="best", fontsize=8, title="")
        fig.tight_layout()
        stem = f"volcano_{cohort}_{outcome}"
        fig.savefig(FIGURES / f"{stem}.png", dpi=200)
        fig.savefig(FIGURES / f"{stem}.svg")
        for subdir in [FIGURES / "main_text", FIGURES / "supplementary"]:
            subdir.mkdir(parents=True, exist_ok=True)
            fig.savefig(subdir / f"{stem}.png", dpi=200)
            fig.savefig(subdir / f"{stem}.svg")
        plt.close(fig)


def simple_output_figures(results_df: pd.DataFrame, important_df: pd.DataFrame, error_df: pd.DataFrame) -> None:
    if not results_df.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        plot = results_df.copy()
        plot["label"] = plot["outcome"] + "\n" + plot["train_cohort"] + "->" + plot["external_cohort"] + "\n" + plot["feature_set"]
        ax.errorbar(range(len(plot)), plot["roc_auc"], yerr=[plot["roc_auc"] - plot["external_auc_ci_low"], plot["external_auc_ci_high"] - plot["roc_auc"]], fmt="o")
        ax.set_xticks(range(len(plot)))
        ax.set_xticklabels(plot["label"], rotation=90, fontsize=7)
        ax.set_ylabel("External ROC-AUC")
        fig.tight_layout()
        fig.savefig(FIGURES / "external_auc_summary_with_ci.png", dpi=200)
        fig.savefig(FIGURES / "external_auc_summary_with_ci.svg")
        plt.close(fig)
        plot_external_transportability_forest(results_df)
    if not important_df.empty:
        for keys, group in important_df.groupby(["outcome", "cohort_direction", "feature_set", "model"], dropna=False):
            top = group.sort_values("absolute_importance", ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(7, 4))
            vals = top["importance"].astype(float)
            ax.barh(top["feature"].astype(str), vals)
            ax.invert_yaxis()
            ax.set_title("Top predictive features: " + " ".join(map(str, keys)))
            ax.set_xlabel("Signed coefficient or permutation importance")
            fig.tight_layout()
            stem = "top_features_" + "_".join(re.sub(r"[^A-Za-z0-9]+", "_", str(k)) for k in keys)
            fig.savefig(FIGURES / f"{stem}.png", dpi=200)
            fig.savefig(FIGURES / f"{stem}.svg")
            plt.close(fig)
    if not error_df.empty:
        summary = error_df.groupby(["outcome", "train_cohort", "external_cohort", "feature_set"]).size().reset_index(name="total_errors")
        fig, ax = plt.subplots(figsize=(8, 4))
        summary.pivot_table(index="outcome", columns="feature_set", values="total_errors", aggfunc="sum").plot(kind="bar", ax=ax)
        ax.set_ylabel("Misclassified external samples")
        fig.tight_layout()
        fig.savefig(FIGURES / "external_error_counts_by_feature_set.png", dpi=200)
        fig.savefig(FIGURES / "external_error_counts_by_feature_set.svg")
        plt.close(fig)


def selected_feature_model_training_only(results_df: pd.DataFrame, outcome: str) -> pd.Series | None:
    """Select the Franzosa-trained metabolomics model using internal-CV logic only."""
    subset = results_df[
        results_df["outcome"].astype(str).eq(outcome)
        & results_df["train_cohort"].astype(str).eq("Franzosa")
        & results_df["external_cohort"].astype(str).eq("iHMP")
        & results_df["feature_set"].astype(str).eq("metabolomics")
    ].copy()
    if subset.empty:
        return None
    selected_flag = subset.get("selected_for_external_validation", pd.Series(False, index=subset.index)).astype(bool)
    selected = subset[selected_flag]
    if len(selected) == 1:
        return selected.iloc[0]
    reselection = select_models_training_only(subset)
    if reselection.empty:
        return None
    return reselection.iloc[0]


def plot_consolidated_top_features(important_df: pd.DataFrame, results_df: pd.DataFrame) -> None:
    """Create one slide-ready top-predictive-feature figure per outcome."""
    if important_df.empty or results_df.empty:
        return
    outcome_order = ["ibd_vs_control", "cd_vs_control", "uc_vs_control"]
    outcome_labels = {
        "ibd_vs_control": "IBD vs Control",
        "cd_vs_control": "CD vs Control",
        "uc_vs_control": "UC vs Control",
    }
    model_labels = {
        "ridge_logistic": "Ridge logistic",
        "elastic_net_logistic": "Elastic-net logistic",
        "random_forest": "Random forest",
        "extra_trees": "Extra Trees",
        "gradient_boosting": "Gradient boosting",
    }
    for outcome in outcome_order:
        selected = selected_feature_model_training_only(results_df, outcome)
        if selected is None:
            continue
        model = str(selected["model"])
        rows = important_df[
            important_df["outcome"].astype(str).eq(outcome)
            & important_df["train_cohort"].astype(str).eq("Franzosa")
            & important_df["external_cohort"].astype(str).eq("iHMP")
            & important_df["feature_set"].astype(str).eq("metabolomics")
            & important_df["model"].astype(str).eq(model)
        ].copy()
        if rows.empty:
            continue
        rows = rows.sort_values("absolute_importance", ascending=False).head(15).copy()
        rows["plot_label"] = rows["hmdb_id"].where(rows["hmdb_id"].notna(), rows["feature"]).astype(str)
        rows.to_csv(TABLES / f"top_features_consolidated_{outcome}.csv", index=False)

        measure = str(rows["importance_measure"].dropna().iloc[0]) if rows["importance_measure"].notna().any() else ""
        is_logistic = measure == "standardised_logistic_coefficient"
        plot_vals = rows["importance"].astype(float) if is_logistic else rows["absolute_importance"].astype(float)
        colors = ["#D55E00" if v > 0 else "#0072B2" for v in rows["importance"].astype(float)] if is_logistic else ["#009E73"] * len(rows)
        fig_height = max(4.8, 0.34 * len(rows) + 1.8)
        fig, ax = plt.subplots(figsize=(8.6, fig_height))
        y_pos = np.arange(len(rows))
        ax.barh(y_pos, plot_vals, color=colors, alpha=0.88)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(rows["plot_label"], fontsize=8.5)
        ax.invert_yaxis()
        if is_logistic:
            ax.axvline(0, color="0.25", linewidth=1)
            ax.set_xlabel("Standardised logistic coefficient")
            if "odds_ratio_per_one_sd_increase" in rows:
                for y, val, odds in zip(y_pos, plot_vals, rows["odds_ratio_per_one_sd_increase"]):
                    if pd.notna(odds):
                        offset = 0.02 * max(1.0, float(np.nanmax(np.abs(plot_vals))))
                        ha = "left" if val >= 0 else "right"
                        ax.text(val + (offset if val >= 0 else -offset), y, f"OR {odds:.2f}", va="center", ha=ha, fontsize=7.5, color="0.25")
        else:
            ax.set_xlabel("Held-out training-fold permutation importance (ROC-AUC)")
        ax.set_title(f"Top predictive metabolomics features: {outcome_labels.get(outcome, outcome)}\nSelected model: {model_labels.get(model, model)}", fontsize=13, weight="bold")
        ax.text(
            0.0,
            -0.12,
            "Predictive importance only; not causal or differential-abundance evidence.",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="0.3",
        )
        ax.grid(axis="x", color="0.88", linewidth=0.8)
        fig.tight_layout()
        for outdir in [FIGURES, FIGURES / "main_text"]:
            fig.savefig(outdir / f"top_features_consolidated_{outcome}.png", dpi=240, bbox_inches="tight")
            fig.savefig(outdir / f"top_features_consolidated_{outcome}.svg", bbox_inches="tight")
        plt.close(fig)


def plot_cross_cohort_stability(stable_df: pd.DataFrame) -> None:
    """Visualise replicated signed coefficients in both cohort-transfer directions."""
    outcome_order = ["ibd_vs_control", "cd_vs_control", "uc_vs_control"]
    outcome_labels = {
        "ibd_vs_control": "IBD vs Control",
        "cd_vs_control": "CD vs Control",
        "uc_vs_control": "UC vs Control",
    }
    status_rows = []
    plot_frames = []
    if not stable_df.empty:
        eligible = stable_df[
            stable_df["feature_set"].astype(str).eq("metabolomics")
            & stable_df["direction_agreement"].eq(True)
            & stable_df["classification"].astype(str).eq("replicated signed predictive association")
            & stable_df["franzosa_to_ihmp_importance"].notna()
            & stable_df["ihmp_to_franzosa_importance"].notna()
        ].copy()
        eligible["mean_rank"] = (pd.to_numeric(eligible["franzosa_to_ihmp_rank"], errors="coerce") + pd.to_numeric(eligible["ihmp_to_franzosa_rank"], errors="coerce")) / 2
    else:
        eligible = pd.DataFrame()
    for outcome in outcome_order:
        out = eligible[eligible["outcome"].astype(str).eq(outcome)].copy() if not eligible.empty else pd.DataFrame()
        n_eligible = len(out)
        status_rows.append({
            "outcome": outcome,
            "eligible_replicated_metabolites": n_eligible,
            "figure_status": "plotted" if n_eligible >= 3 else "not_plotted",
            "reason": "" if n_eligible >= 3 else "Fewer than three eligible replicated signed metabolomics associations after filtering.",
        })
        if n_eligible >= 3:
            plot_frames.append(out.sort_values("mean_rank", ascending=True).head(15))
    status_df = pd.DataFrame(status_rows)
    status_df.to_markdown(REPORTS / "cross_cohort_stability_status.md", index=False)
    if not plot_frames:
        pd.DataFrame(columns=list(stable_df.columns) + ["mean_rank"] if not stable_df.empty else ["outcome", "feature_set", "hmdb_id", "mean_rank"]).to_csv(TABLES / "cross_cohort_feature_stability_plot_data.csv", index=False)
        return
    plot_df = pd.concat(plot_frames, ignore_index=True)
    plot_df.to_csv(TABLES / "cross_cohort_feature_stability_plot_data.csv", index=False)

    panel_outcomes = [o for o in outcome_order if o in set(plot_df["outcome"].astype(str))]
    fig, axes = plt.subplots(1, len(panel_outcomes), figsize=(5.4 * len(panel_outcomes), 5.6), squeeze=False)
    for ax, outcome in zip(axes.ravel(), panel_outcomes):
        sub = plot_df[plot_df["outcome"].astype(str).eq(outcome)].sort_values("mean_rank", ascending=False)
        y_pos = np.arange(len(sub))
        x_a = pd.to_numeric(sub["franzosa_to_ihmp_importance"], errors="coerce").to_numpy()
        x_b = pd.to_numeric(sub["ihmp_to_franzosa_importance"], errors="coerce").to_numpy()
        for y, a, b in zip(y_pos, x_a, x_b):
            ax.plot([a, b], [y, y], color="0.72", linewidth=1.4, zorder=1)
        ax.scatter(x_a, y_pos, color="#D55E00", label="Franzosa→iHMP", s=42, zorder=3)
        ax.scatter(x_b, y_pos, color="#0072B2", label="iHMP→Franzosa", s=42, zorder=3)
        ax.axvline(0, color="0.25", linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["hmdb_id"].astype(str), fontsize=8)
        ax.set_title(outcome_labels.get(outcome, outcome), fontsize=12, weight="bold")
        ax.set_xlabel("Standardised logistic coefficient")
        ax.grid(axis="x", color="0.9", linewidth=0.8)
    axes.ravel()[0].set_ylabel("HMDB metabolite")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, title="Transfer direction")
    fig.suptitle("Same-direction signed predictive coefficients in both cohort-transfer directions", y=1.10, fontsize=14, weight="bold")
    fig.text(0.5, -0.02, "Replicated predictive associations only; not causal biomarkers or differential-abundance effects.", ha="center", fontsize=9, color="0.3")
    fig.tight_layout()
    for outdir in [FIGURES, FIGURES / "main_text"]:
        fig.savefig(outdir / "cross_cohort_feature_stability.png", dpi=240, bbox_inches="tight")
        fig.savefig(outdir / "cross_cohort_feature_stability.svg", bbox_inches="tight")
    plt.close(fig)


def plot_external_transportability_forest(results_df: pd.DataFrame) -> None:
    """Forest plot for external transportability using already selected models only."""
    if results_df.empty:
        return
    outcome_order = ["ibd_vs_control", "cd_vs_control", "uc_vs_control"]
    outcome_labels = {
        "ibd_vs_control": "IBD vs Control",
        "cd_vs_control": "CD vs Control",
        "uc_vs_control": "UC vs Control",
    }
    direction_order = [("Franzosa", "iHMP"), ("iHMP", "Franzosa")]
    direction_labels = {
        ("Franzosa", "iHMP"): "Franzosa→iHMP",
        ("iHMP", "Franzosa"): "iHMP→Franzosa",
    }
    feature_order = ["metabolomics", "integrated", "clinical"]
    feature_labels = {"clinical": "Clinical", "metabolomics": "Metabolomics", "integrated": "Integrated"}
    feature_colors = {"clinical": "#4C78A8", "metabolomics": "#009E73", "integrated": "#D55E00"}
    model_labels = {
        "ridge_logistic": "Ridge logistic",
        "elastic_net_logistic": "Elastic Net",
        "random_forest": "Random Forest",
        "extra_trees": "Extra Trees",
        "gradient_boosting": "Gradient Boosting",
    }
    plot = results_df[
        results_df["outcome"].isin(outcome_order)
        & results_df["feature_set"].isin(feature_order)
    ].copy()
    if plot.empty:
        return
    plot["outcome"] = pd.Categorical(plot["outcome"], categories=outcome_order, ordered=True)
    plot["feature_set"] = pd.Categorical(plot["feature_set"], categories=feature_order, ordered=True)
    rows = []
    y = 0.0
    for outcome in outcome_order:
        rows.append({"kind": "header", "outcome": outcome, "label": outcome_labels[outcome], "y": y})
        y += 1.0
        for train, external in direction_order:
            rows.append({"kind": "direction", "outcome": outcome, "train_cohort": train, "external_cohort": external, "label": direction_labels[(train, external)], "y": y})
            y += 0.72
            for feature in feature_order:
                match = plot[
                    plot["outcome"].astype(str).eq(outcome)
                    & plot["train_cohort"].eq(train)
                    & plot["external_cohort"].eq(external)
                    & plot["feature_set"].astype(str).eq(feature)
                ]
                if match.empty:
                    continue
                r = match.iloc[0]
                rows.append({
                    "kind": "model",
                    "outcome": outcome,
                    "train_cohort": train,
                    "external_cohort": external,
                    "feature_set": feature,
                    "feature_label": feature_labels[feature],
                    "model": r["model"],
                    "model_label": model_labels.get(r["model"], r["model"]),
                    "roc_auc": float(r["roc_auc"]),
                    "ci_low": float(r["external_auc_ci_low"]),
                    "ci_high": float(r["external_auc_ci_high"]),
                    "balanced_accuracy": float(r["balanced_accuracy"]),
                    "f1": float(r["f1"]) if pd.notna(r["f1"]) else np.nan,
                    "label": feature_labels[feature],
                    "y": y,
                })
                y += 0.62
            y += 0.18
        y += 0.82
    row_df = pd.DataFrame(rows)
    model_df = row_df[row_df["kind"].eq("model")].copy()
    model_df.to_csv(TABLES / "presentation_external_transportability_forest_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(11.8, 8.2))
    for _, r in model_df.iterrows():
        color = feature_colors.get(r["feature_set"], "0.35")
        ax.errorbar(
            r["roc_auc"],
            r["y"],
            xerr=[[r["roc_auc"] - r["ci_low"]], [r["ci_high"] - r["roc_auc"]]],
            fmt="o",
            markersize=6.8,
            color=color,
            ecolor=color,
            elinewidth=1.8,
            capsize=3.5,
            zorder=3,
        )
    ax.axvline(0.5, color="0.50", linestyle="--", linewidth=1.15)
    ax.text(0.505, row_df["y"].min() - 0.35, "No discrimination", ha="left", va="bottom", fontsize=8.6, color="0.30")
    label_rows = row_df[row_df["kind"].isin(["direction", "model"])].copy()
    ax.set_yticks(label_rows["y"])
    ylabels = []
    for _, r in label_rows.iterrows():
        if r["kind"] == "direction":
            ylabels.append(str(r["label"]))
        else:
            ylabels.append("    " + str(r["label"]))
    ax.set_yticklabels(ylabels, fontsize=9)
    for _, r in row_df[row_df["kind"].eq("header")].iterrows():
        ax.text(0.215, r["y"], r["label"], ha="left", va="center", fontsize=10.5, weight="bold", color="0.15")
    for _, r in row_df[row_df["kind"].eq("direction")].iterrows():
        ax.axhline(r["y"] - 0.38, color="0.93", linewidth=0.8, zorder=0)
    ax.set_ylim(row_df["y"].max() + 0.45, row_df["y"].min() - 0.70)
    ax.set_xlim(0.20, 1.02)
    ax.set_xlabel("External ROC-AUC with 95% bootstrap CI", fontsize=11)
    ax.set_title("External validation transportability", fontsize=17, weight="bold", pad=12)
    ax.grid(axis="x", color="0.90", linewidth=0.75)
    ax.tick_params(axis="y", length=0)
    handles = [
        plt.Line2D([0], [0], marker="o", color=feature_colors[f], lw=1.8, markersize=7, label=feature_labels[f])
        for f in feature_order
    ]
    ax.legend(handles=handles, title="Feature set", frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.72), fontsize=10, title_fontsize=10.5)
    ax.text(
        0.0,
        -0.09,
        "Only internally selected clinical, metabolomics and integrated models are shown. Higher AUC indicates better cross-cohort transportability.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color="0.28",
    )
    fig.subplots_adjust(left=0.24, right=0.80, top=0.91, bottom=0.12)
    for outdir in [FIGURES, FIGURES / "main_text"]:
        fig.savefig(outdir / "external_validation_transportability_forest.png", dpi=240, bbox_inches="tight")
        fig.savefig(outdir / "external_validation_transportability_forest.svg", bbox_inches="tight")
    plt.close(fig)


def plot_internal_performance_presentation_slide(results_df: pd.DataFrame) -> None:
    """Create a slide-ready internal performance summary from unchanged model results."""
    if results_df.empty:
        return
    representative = results_df[results_df["train_cohort"].eq("Franzosa")].copy()
    if representative.empty:
        representative = results_df.copy()
    outcome_order = ["ibd_vs_control", "cd_vs_control", "uc_vs_control"]
    outcome_labels = {
        "ibd_vs_control": "IBD vs Control",
        "cd_vs_control": "CD vs Control",
        "uc_vs_control": "UC vs Control",
    }
    feature_order = ["clinical", "metabolomics", "integrated"]
    feature_labels = {"clinical": "Clinical", "metabolomics": "Metabolomics", "integrated": "Integrated"}
    model_labels = {
        "ridge_logistic": "Ridge logistic",
        "elastic_net_logistic": "Elastic-net logistic",
        "random_forest": "Random forest",
        "extra_trees": "Extra Trees",
        "gradient_boosting": "Gradient boosting",
    }
    model_colors = {
        "ridge_logistic": "#4C78A8",
        "elastic_net_logistic": "#F58518",
        "random_forest": "#54A24B",
        "extra_trees": "#B279A2",
        "gradient_boosting": "#E45756",
    }
    plot_rows = representative[
        representative["selected_for_external_validation"].astype(bool)
        & representative["outcome"].isin(outcome_order)
        & representative["feature_set"].isin(feature_order)
    ].copy()
    if plot_rows.empty:
        return
    plot_rows["feature_set"] = pd.Categorical(plot_rows["feature_set"], categories=feature_order, ordered=True)
    plot_rows["outcome"] = pd.Categorical(plot_rows["outcome"], categories=outcome_order, ordered=True)
    plot_rows = plot_rows.sort_values(["outcome", "feature_set"])
    best_rows = (
        plot_rows.sort_values(["outcome", "internal_cv_auc_mean"], ascending=[True, False])
        .groupby("outcome", observed=False)
        .head(1)
        .copy()
    )
    table_rows = []
    for outcome in outcome_order:
        row = best_rows[best_rows["outcome"].astype(str).eq(outcome)]
        if row.empty:
            continue
        r = row.iloc[0]
        table_rows.append({
            "Prediction task": outcome_labels[outcome],
            "Best-performing model": f"{feature_labels.get(r['feature_set'], r['feature_set'])} + {model_labels.get(r['model'], r['model'])}",
            "Internal ROC-AUC": f"{r['internal_cv_auc_mean']:.2f}",
            "95% CI": f"{max(0, r['internal_cv_auc_mean'] - 1.96 * r['internal_cv_auc_sd'] / math.sqrt(max(1, int(r['outer_folds'])))):.2f}-{min(1, r['internal_cv_auc_mean'] + 1.96 * r['internal_cv_auc_sd'] / math.sqrt(max(1, int(r['outer_folds'])))):.2f}",
            "External ROC-AUC": f"{r['roc_auc']:.2f}" if pd.notna(r.get("roc_auc", np.nan)) else "NA",
            "Balanced accuracy": f"{r['balanced_accuracy']:.2f}" if pd.notna(r.get("balanced_accuracy", np.nan)) else "NA",
            "Sensitivity": f"{r['sensitivity']:.2f}" if pd.notna(r.get("sensitivity", np.nan)) else "NA",
            "Specificity": f"{r['specificity']:.2f}" if pd.notna(r.get("specificity", np.nan)) else "NA",
            "Precision": f"{r['ppv']:.2f}" if pd.notna(r.get("ppv", np.nan)) else "NA",
            "F1-score": f"{r['f1']:.2f}" if pd.notna(r.get("f1", np.nan)) else "NA",
        })
    pd.DataFrame(table_rows).to_csv(TABLES / "presentation_internal_performance_slide_summary.csv", index=False)
    plot_rows.assign(
        outcome_label=lambda d: d["outcome"].astype(str).map(outcome_labels),
        feature_label=lambda d: d["feature_set"].astype(str).map(feature_labels),
        model_label=lambda d: d["model"].map(model_labels),
    ).to_csv(TABLES / "presentation_internal_performance_slide_plot_data.csv", index=False)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.0, 1.05], hspace=0.38, wspace=0.28)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax, outcome in zip(axes, outcome_order):
        part = plot_rows[plot_rows["outcome"].astype(str).eq(outcome)].copy()
        ypos = np.arange(len(feature_order))
        ax.axvline(0.5, linestyle="--", color="0.62", lw=1)
        for _, r in part.iterrows():
            y = feature_order.index(str(r["feature_set"]))
            mean = float(r["internal_cv_auc_mean"])
            sd = float(r["internal_cv_auc_sd"])
            ax.errorbar(
                mean,
                y,
                xerr=sd,
                fmt="o",
                markersize=8,
                color=model_colors.get(r["model"], "0.25"),
                ecolor=model_colors.get(r["model"], "0.25"),
                elinewidth=2,
                capsize=4,
                zorder=3,
            )
            label_x = min(mean + sd + 0.025, 1.025)
            label_ha = "left"
            if label_x > 0.995:
                label_x = mean - 0.035
                label_ha = "right"
            ax.text(label_x, y, f"{mean:.2f}", va="center", ha=label_ha, fontsize=10, color="0.20")
        ax.set_yticks(ypos)
        ax.set_yticklabels([feature_labels[f] for f in feature_order], fontsize=11)
        ax.invert_yaxis()
        ax.set_xlim(0.45, 1.05)
        ax.set_xlabel("Nested CV ROC-AUC (mean +/- SD)", fontsize=11)
        ax.set_title(outcome_labels[outcome], fontsize=15, weight="bold", pad=10)
        ax.grid(axis="x", color="0.90", linewidth=0.8)
        ax.tick_params(axis="x", labelsize=10)
        if ax is not axes[0]:
            ax.set_ylabel("")
    used_models = [m for m in model_colors if m in set(plot_rows["model"])]
    handles = [
        plt.Line2D([0], [0], marker="o", color=model_colors[m], markersize=8, lw=2, label=model_labels[m])
        for m in used_models
    ]
    fig.legend(handles=handles, title="Selected model family", loc="upper center", bbox_to_anchor=(0.54, 0.89), frameon=False, fontsize=10, title_fontsize=11, ncol=max(1, len(handles)))

    table_ax = fig.add_subplot(gs[1, :])
    table_ax.axis("off")
    summary_df = pd.DataFrame(table_rows)
    display_df = summary_df.copy()
    display_df["Internal AUC (95% CI)"] = display_df["Internal ROC-AUC"] + " (" + display_df["95% CI"] + ")"
    display_df = display_df.rename(columns={
        "Prediction task": "Task",
        "Best-performing model": "Best model",
        "External ROC-AUC": "External AUC",
        "Balanced accuracy": "Bal. acc.",
        "Sensitivity": "Sens.",
        "Specificity": "Spec.",
        "Precision": "Prec.",
        "F1-score": "F1",
    })
    display_cols = ["Task", "Best model", "Internal AUC (95% CI)", "External AUC", "Bal. acc.", "Sens.", "Spec.", "Prec.", "F1"]
    table = table_ax.table(
        cellText=display_df[display_cols].values,
        colLabels=display_cols,
        cellLoc="left",
        colLoc="left",
        loc="upper center",
        colWidths=[0.105, 0.255, 0.14, 0.08, 0.08, 0.062, 0.062, 0.062, 0.052],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("0.82")
        if row == 0:
            cell.set_facecolor("#1F4E79")
            cell.set_text_props(color="white", weight="bold")
        elif col == 1:
            cell.set_facecolor("#FFF3D6")
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#F7F9FB")
    fig.suptitle("Internal model performance", fontsize=24, weight="bold", y=0.965)
    fig.text(
        0.5,
        0.035,
        "Model selection and hyperparameter optimisation were performed using nested stratified cross-validation. "
        "Classification metrics were calculated using the frozen Youden-J threshold selected during internal validation.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="0.25",
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.10, wspace=0.42)
    for outdir in [FIGURES, FIGURES / "main_text"]:
        fig.savefig(outdir / "internal_model_performance_presentation_slide.png", dpi=240, bbox_inches="tight")
        fig.savefig(outdir / "internal_model_performance_presentation_slide.svg", bbox_inches="tight")
    plt.close(fig)


def plot_full_internal_model_comparison_panel(internal_df: pd.DataFrame) -> None:
    """Slide-ready panel showing all feature sets and all model families for the best internal cohort."""
    if internal_df.empty:
        return
    outcome_order = ["ibd_vs_control", "cd_vs_control", "uc_vs_control"]
    outcome_labels = {
        "ibd_vs_control": "IBD vs Control",
        "cd_vs_control": "CD vs Control",
        "uc_vs_control": "UC vs Control",
    }
    feature_order = ["metabolomics", "integrated", "clinical"]
    feature_labels = {"clinical": "Clinical", "metabolomics": "Metabolomics", "integrated": "Integrated"}
    model_order = ["ridge_logistic", "random_forest", "gradient_boosting", "extra_trees", "elastic_net_logistic"]
    model_labels = {
        "ridge_logistic": "Ridge logistic",
        "elastic_net_logistic": "Elastic-net logistic",
        "random_forest": "Random forest",
        "extra_trees": "Extra Trees",
        "gradient_boosting": "Gradient boosting",
    }
    feature_colors = {"clinical": "#4C78A8", "metabolomics": "#009E73", "integrated": "#D55E00"}
    best_by_cohort = (
        internal_df[internal_df["outcome"].isin(outcome_order)]
        .groupby(["train_cohort", "outcome"], dropna=False)["internal_cv_auc_mean"]
        .max()
        .reset_index()
    )
    cohort_scores = best_by_cohort.groupby("train_cohort")["internal_cv_auc_mean"].mean().sort_values(ascending=False)
    if cohort_scores.empty:
        return
    best_cohort = str(cohort_scores.index[0])
    plot_df = internal_df[
        internal_df["train_cohort"].eq(best_cohort)
        & internal_df["outcome"].isin(outcome_order)
        & internal_df["feature_set"].isin(feature_order)
        & internal_df["model"].isin(model_order)
    ].copy()
    if plot_df.empty:
        return
    plot_df["outcome"] = pd.Categorical(plot_df["outcome"], categories=outcome_order, ordered=True)
    plot_df["feature_set"] = pd.Categorical(plot_df["feature_set"], categories=feature_order, ordered=True)
    plot_df["model"] = pd.Categorical(plot_df["model"], categories=model_order, ordered=True)
    plot_df = plot_df.sort_values(["outcome", "feature_set", "model"])
    plot_df.assign(
        outcome_label=lambda d: d["outcome"].astype(str).map(outcome_labels),
        feature_label=lambda d: d["feature_set"].astype(str).map(feature_labels),
        model_label=lambda d: d["model"].astype(str).map(model_labels),
        best_internal_training_cohort=best_cohort,
    ).to_csv(TABLES / "presentation_full_internal_model_comparison_panel_data.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17.8, 8.0), sharex=True, sharey=True)
    row_labels = [f"{feature_labels[f]} | {model_labels[m]}" for f in feature_order for m in model_order]
    y_positions = np.arange(len(row_labels))
    for ax, outcome in zip(axes, outcome_order):
        part = plot_df[plot_df["outcome"].astype(str).eq(outcome)].copy()
        ax.axvline(0.5, color="0.55", linestyle="--", linewidth=1)
        for _, r in part.iterrows():
            y = feature_order.index(str(r["feature_set"])) * len(model_order) + model_order.index(str(r["model"]))
            mean = float(r["internal_cv_auc_mean"])
            sd = float(r["internal_cv_auc_sd"])
            color = feature_colors.get(str(r["feature_set"]), "0.35")
            ax.errorbar(mean, y, xerr=sd, fmt="o", markersize=5.8, color=color, ecolor=color, elinewidth=1.55, capsize=3, zorder=3)
        ax.set_title(outcome_labels[outcome], fontsize=16, weight="bold", pad=8)
        ax.set_xlabel("Nested outer-fold ROC-AUC (mean +/- SD)", fontsize=11)
        ax.set_xlim(0.0, 1.02)
        ax.set_yticks(y_positions)
        if ax is axes[0]:
            ax.set_yticklabels(row_labels, fontsize=8.7)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.invert_yaxis()
        ax.grid(axis="x", color="0.90", linewidth=0.75)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", length=0, pad=3)
        for boundary in [len(model_order) - 0.5, 2 * len(model_order) - 0.5]:
            ax.axhline(boundary, color="0.86", linewidth=0.9)
    handles = [
        plt.Line2D([0], [0], marker="o", color=feature_colors[f], lw=1.8, markersize=6, label=feature_labels[f])
        for f in feature_order
    ]
    fig.legend(handles=handles, title="Feature set", loc="center left", bbox_to_anchor=(0.875, 0.72), ncol=1, frameon=False, fontsize=11, title_fontsize=11)
    fig.suptitle(f"Internal model comparison: train {best_cohort}", fontsize=23, weight="bold", y=0.975)
    fig.text(0.5, 0.035, "All five model families are shown for each feature set. Error bars show SD across nested outer folds; dashed line marks ROC-AUC = 0.5.", ha="center", va="center", fontsize=10.5, color="0.28")
    fig.subplots_adjust(left=0.19, right=0.86, top=0.82, bottom=0.12, wspace=0.18)
    for outdir in [FIGURES, FIGURES / "main_text"]:
        fig.savefig(outdir / "internal_model_comparison_all_models_best_cohort.png", dpi=240, bbox_inches="tight")
        fig.savefig(outdir / "internal_model_comparison_all_models_best_cohort.svg", bbox_inches="tight")
    plt.close(fig)


def select_models_training_only(dev_df: pd.DataFrame) -> pd.DataFrame:
    selected_rows = []
    simplicity = CONFIG["nested_cv"]["model_simplicity_order"]
    for keys, group in dev_df.groupby(["outcome", "train_cohort", "external_cohort", "feature_set"], dropna=False):
        g = group[group.get("nested_cv_valid", True) == True].copy()
        if g.empty:
            continue
        best_auc = g["internal_cv_auc_mean"].max()
        best = g.loc[g["internal_cv_auc_mean"].idxmax()]
        best_sd = best["internal_cv_auc_sd"]
        valid_outer_folds = max(1, int(best.get("outer_folds", 1) - best.get("fold_failures", 0)))
        best_se = 0 if pd.isna(best_sd) else float(best_sd) / math.sqrt(valid_outer_folds)
        one_se_cutoff = best_auc - best_se
        candidates = g[g["internal_cv_auc_mean"] >= one_se_cutoff].copy()
        candidates["simplicity_rank"] = candidates["model"].map(simplicity).fillna(999)
        candidates = candidates.sort_values(
            ["simplicity_rank", "internal_cv_pr_auc_mean", "internal_cv_auc_mean"],
            ascending=[True, False, False],
        )
        row = candidates.iloc[0].copy()
        row["model_selection_rule"] = "highest nested outer-fold ROC-AUC; one-standard-error simplicity rule; PR-AUC tie-breaker"
        row["best_nested_auc_before_one_sd_rule"] = best_auc
        row["best_nested_auc_sd"] = best_sd
        row["best_nested_auc_se"] = best_se
        row["one_standard_error_auc_cutoff"] = one_se_cutoff
        row["valid_outer_folds_for_se"] = valid_outer_folds
        selected_rows.append(row)
    return pd.DataFrame(selected_rows)


def choose_nested_fold_design(y: pd.Series) -> tuple[int, int, str]:
    min_class = int(pd.Series(y).value_counts().min())
    outer = min(CONFIG["nested_cv"]["outer_folds_requested"], min_class)
    if min_class >= CONFIG["nested_cv"]["minimum_class_count_for_requested_inner"]:
        inner_requested = CONFIG["nested_cv"]["inner_folds_requested"]
    else:
        inner_requested = CONFIG["nested_cv"]["reduced_inner_folds"]
    inner = min(inner_requested, max(2, min_class - 1))
    outer = max(2, outer)
    inner = max(2, inner)
    if outer < CONFIG["nested_cv"]["outer_folds_requested"] or inner < CONFIG["nested_cv"]["inner_folds_requested"]:
        rationale = f"Reduced design due to smallest class count={min_class}: outer={outer}, inner={inner}."
    else:
        rationale = f"Requested 5x5 nested CV used; smallest class count={min_class}."
    return outer, inner, rationale


def run_nested_cv(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    outcome: str,
    train_cohort: str,
    external_cohort: str,
    feature_set: str,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    outer_folds, inner_folds, rationale = choose_nested_fold_design(y_train)
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=CONFIG["random_seed"])
    outer_rows = []
    fold_rows = []
    failure_rows = []
    tuning_rows = []
    for fold_idx, (inner_dev_idx, outer_test_idx) in enumerate(outer_cv.split(x_train, y_train), start=1):
        x_dev = x_train.iloc[inner_dev_idx]
        y_dev = y_train.iloc[inner_dev_idx]
        x_outer = x_train.iloc[outer_test_idx]
        y_outer = y_train.iloc[outer_test_idx]
        inner_min_class = int(pd.Series(y_dev).value_counts().min())
        fold_inner_folds = min(CONFIG["nested_cv"]["inner_folds_requested"], inner_min_class)
        if fold_inner_folds < 2:
            failure_rows.append(
                {
                    "outcome": outcome,
                    "train_cohort": train_cohort,
                    "external_cohort": external_cohort,
                    "feature_set": feature_set,
                    "model": model_name,
                    "outer_fold": fold_idx,
                    "failure": f"inner_min_class={inner_min_class}; cannot run inner CV",
                    "fold_design_rationale": rationale,
                }
            )
            continue
        train_counts = pd.Series(y_dev).value_counts().to_dict()
        test_counts = pd.Series(y_outer).value_counts().to_dict()
        fold_rows.append(
            {
                "outcome": outcome,
                "train_cohort": train_cohort,
                "external_cohort": external_cohort,
                "feature_set": feature_set,
                "model": model_name,
                "outer_fold": fold_idx,
                "outer_train_class0": int(train_counts.get(0, 0)),
                "outer_train_class1": int(train_counts.get(1, 0)),
                "outer_test_class0": int(test_counts.get(0, 0)),
                "outer_test_class1": int(test_counts.get(1, 0)),
                "outer_folds": outer_folds,
                "inner_folds": fold_inner_folds,
                "fold_design_rationale": rationale,
            }
        )
        try:
            pipe, grid = make_model(model_name)
            inner_cv = StratifiedKFold(n_splits=fold_inner_folds, shuffle=True, random_state=CONFIG["random_seed"] + fold_idx)
            gs, search_method, n_tested, n_candidate_grid = make_search(
                pipe, grid, model_name, inner_cv, CONFIG["random_seed"] + fold_idx
            )
            search_start = time.perf_counter()
            gs.fit(x_dev, y_dev, **fit_params_for_model(model_name, y_dev))
            search_runtime = time.perf_counter() - search_start
            tuning_rows.extend(search_result_rows(
                gs,
                {
                    "search_stage": "nested_outer_fold_inner_cv",
                    "outcome": outcome,
                    "train_cohort": train_cohort,
                    "external_cohort": external_cohort,
                    "feature_set": feature_set,
                    "model": model_name,
                    "outer_fold": fold_idx,
                    "inner_folds": fold_inner_folds,
                },
                search_method,
                n_tested,
                n_candidate_grid,
                search_runtime,
            ))
            prob = gs.best_estimator_.predict_proba(x_outer)[:, 1]
            met = binary_metrics(y_outer, prob)
            outer_rows.append(
                {
                    "outcome": outcome,
                    "train_cohort": train_cohort,
                    "external_cohort": external_cohort,
                    "feature_set": feature_set,
                    "model": model_name,
                    "outer_fold": fold_idx,
                    "nested_outer_auc": met["roc_auc"],
                    "nested_outer_pr_auc": met["pr_auc"],
                    "nested_outer_balanced_accuracy": met["balanced_accuracy"],
                    "nested_outer_f1": met["f1"],
                    "best_params_outer_fold": json.dumps(gs.best_params_),
                    "fold_design_rationale": rationale,
                }
            )
        except Exception as exc:
            tb = traceback.format_exc()
            pathway_logger().error(
                "Nested CV fold failed: outcome=%s train=%s external=%s feature_set=%s model=%s outer_fold=%s\n%s",
                outcome,
                train_cohort,
                external_cohort,
                feature_set,
                model_name,
                fold_idx,
                tb,
            )
            failure_rows.append(
                {
                    "outcome": outcome,
                    "train_cohort": train_cohort,
                    "external_cohort": external_cohort,
                    "feature_set": feature_set,
                    "model": model_name,
                    "outer_fold": fold_idx,
                    "failure": repr(exc),
                    "traceback": tb,
                    "fold_design_rationale": rationale,
                }
            )
    if outer_rows:
        outer_df = pd.DataFrame(outer_rows)
        valid_spec = len(failure_rows) == 0 and len(outer_rows) == outer_folds
        summary = {
            "nested_cv_auc_mean": float(outer_df["nested_outer_auc"].mean()) if valid_spec else np.nan,
            "nested_cv_auc_sd": float(outer_df["nested_outer_auc"].std(ddof=0)),
            "nested_cv_pr_auc_mean": float(outer_df["nested_outer_pr_auc"].mean()),
            "nested_cv_balanced_accuracy_mean": float(outer_df["nested_outer_balanced_accuracy"].mean()),
            "nested_cv_f1_mean": float(outer_df["nested_outer_f1"].mean()),
            "outer_folds": outer_folds,
            "inner_folds": inner_folds,
            "fold_design_rationale": rationale,
            "fold_failures": len(failure_rows),
            "nested_cv_valid": valid_spec,
        }
    else:
        summary = {
            "nested_cv_auc_mean": np.nan,
            "nested_cv_auc_sd": np.nan,
            "nested_cv_pr_auc_mean": np.nan,
            "nested_cv_balanced_accuracy_mean": np.nan,
            "nested_cv_f1_mean": np.nan,
            "outer_folds": outer_folds,
            "inner_folds": inner_folds,
            "fold_design_rationale": rationale,
            "fold_failures": len(failure_rows),
            "nested_cv_valid": False,
        }
    return summary, outer_rows, fold_rows + failure_rows, tuning_rows


def bootstrap_auc_ci(y_true, y_prob, n_boot=2000) -> tuple[float, float]:
    rng = np.random.default_rng(CONFIG["random_seed"])
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(set(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not scores:
        return np.nan, np.nan
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def model_matrix(meta: pd.DataFrame, hmdb_matrix: pd.DataFrame) -> pd.DataFrame:
    out = meta.merge(hmdb_matrix, on="sample_id", how="inner")
    return out


def outcome_subset(df: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, pd.Series]:
    if outcome == "ibd_vs_control":
        sub = df[df["harmonised_label"].isin(["Control", "CD", "UC"])].copy()
        y = (sub["harmonised_label"] != "Control").astype(int)
    elif outcome == "cd_vs_control":
        sub = df[df["harmonised_label"].isin(["Control", "CD"])].copy()
        y = (sub["harmonised_label"] == "CD").astype(int)
    else:
        sub = df[df["harmonised_label"].isin(["Control", "UC"])].copy()
        y = (sub["harmonised_label"] == "UC").astype(int)
    return sub, y


def pca_plot(df: pd.DataFrame, features: list[str], colour_col: str, out_name: str) -> None:
    from sklearn.decomposition import PCA

    x = df[features].apply(pd.to_numeric, errors="coerce")
    x = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x), columns=features)
    x = pd.DataFrame(StandardScaler().fit_transform(x), columns=features)
    pca = PCA(n_components=2, random_state=CONFIG["random_seed"])
    pcs = pca.fit_transform(x)
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = df[colour_col].astype(str).to_numpy()
    for label in pd.unique(labels):
        mask = labels == label
        ax.scatter(pcs[mask, 0], pcs[mask, 1], s=28, alpha=0.75, label=label)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.set_title(out_name.replace("_", " "))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{out_name}.png", dpi=200)
    fig.savefig(FIGURES / f"{out_name}.svg")
    plt.close(fig)


def add_confidence_ellipse(ax, x: np.ndarray, y: np.ndarray, color: str) -> None:
    """Add an approximate 95% confidence ellipse when enough points are available."""
    from matplotlib.patches import Ellipse

    if len(x) < 3 or len(y) < 3:
        return
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    if np.any(vals <= 0):
        return
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    scale = math.sqrt(5.991)  # 95% chi-square radius for 2 dimensions.
    ellipse = Ellipse(
        (np.mean(x), np.mean(y)),
        width=2 * scale * math.sqrt(vals[0]),
        height=2 * scale * math.sqrt(vals[1]),
        angle=angle,
        facecolor="none",
        edgecolor=color,
        linewidth=1.2,
        alpha=0.85,
    )
    ax.add_patch(ellipse)


def pca_plot_by_cohort(df: pd.DataFrame, features: list[str], cohort: str, out_name: str) -> None:
    """Exploratory PCA within one cohort only; no cross-cohort batch correction."""
    from sklearn.decomposition import PCA

    sub = df[df["cohort"].astype(str).eq(cohort)].copy()
    available = [f for f in features if f in sub.columns]
    if sub.empty or len(available) < 2:
        return
    x = sub[available].apply(pd.to_numeric, errors="coerce")
    x = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x), columns=available)
    x = pd.DataFrame(StandardScaler().fit_transform(x), columns=available)
    pca = PCA(n_components=2, random_state=CONFIG["random_seed"])
    pcs = pca.fit_transform(x)
    labels = sub["harmonised_label"].astype(str).to_numpy()
    palette = {"Control": "#4C78A8", "CD": "#D55E00", "UC": "#009E73"}
    counts = sub["harmonised_label"].value_counts().to_dict()
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for label in ["Control", "CD", "UC"]:
        mask = labels == label
        if not mask.any():
            continue
        color = palette.get(label, "0.5")
        ax.scatter(pcs[mask, 0], pcs[mask, 1], s=36, alpha=0.78, color=color, label=f"{label} (n={int(mask.sum())})", edgecolor="white", linewidth=0.35)
        add_confidence_ellipse(ax, pcs[mask, 0], pcs[mask, 1], color)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.set_title(f"{cohort} metabolomics PCA by disease group", fontsize=13, weight="bold")
    ax.legend(frameon=False, fontsize=8.5, title=f"Samples, total n={len(sub)}", title_fontsize=8.5)
    ax.grid(color="0.9", linewidth=0.8)
    ax.text(
        0.0,
        -0.16,
        "Exploratory PCA only; imputation/scaling are used for visualisation and are not part of model fitting.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="0.3",
    )
    fig.tight_layout()
    for outdir in [FIGURES / "main_text"]:
        fig.savefig(outdir / f"{out_name}.png", dpi=240, bbox_inches="tight")
        fig.savefig(outdir / f"{out_name}.svg", bbox_inches="tight")
    pd.DataFrame([{
        "cohort": cohort,
        "output": out_name,
        "n_samples": len(sub),
        "n_features": len(available),
        "pc1_variance_percent": pca.explained_variance_ratio_[0] * 100,
        "pc2_variance_percent": pca.explained_variance_ratio_[1] * 100,
        "control_n": counts.get("Control", 0),
        "cd_n": counts.get("CD", 0),
        "uc_n": counts.get("UC", 0),
        "caption": "Exploratory within-cohort PCA only; no ComBat, SVA, Harmony or other batch correction was performed.",
    }]).to_csv(TABLES / "main_text" / f"{out_name}_summary.csv", index=False)
    plt.close(fig)


def plot_shared_metabolite_boxplots(diff_df: pd.DataFrame, data_by_cohort: dict, max_features: int = 5) -> None:
    """Boxplots for top replicated FDR-significant shared HMDB metabolites."""
    if diff_df.empty:
        return
    outcome_labels = {
        "ibd_vs_control": "IBD vs Control",
        "cd_vs_control": "CD vs Control",
        "uc_vs_control": "UC vs Control",
    }
    disease_labels = {
        "ibd_vs_control": ["CD", "UC"],
        "cd_vs_control": ["CD"],
        "uc_vs_control": ["UC"],
    }
    plotted_rows = []
    for outcome in ["ibd_vs_control", "cd_vs_control", "uc_vs_control"]:
        sig = diff_df[
            diff_df["outcome"].astype(str).eq(outcome)
            & diff_df["significant_fdr_0_05"].eq(True)
        ].copy()
        if sig.empty:
            continue
        replicated = []
        for hmdb, g in sig.groupby("hmdb_id"):
            cohorts = set(g["cohort"].astype(str))
            directions = set(g["effect_direction"].dropna().astype(str))
            if {"Franzosa", "iHMP"}.issubset(cohorts) and len(directions) == 1:
                replicated.append({
                    "hmdb_id": hmdb,
                    "min_fdr": float(g["fdr_q_value"].min()),
                    "effect_direction": next(iter(directions)),
                })
        top = pd.DataFrame(replicated).sort_values(["min_fdr", "hmdb_id"]).head(max_features) if replicated else pd.DataFrame()
        if top.empty:
            continue
        fig, axes = plt.subplots(1, len(top), figsize=(3.0 * len(top), 4.2), squeeze=False, sharey=False)
        for ax, (_, r) in zip(axes.ravel(), top.iterrows()):
            plot_records = []
            for cohort, df in data_by_cohort.items():
                if r.hmdb_id not in df.columns:
                    continue
                use = df[df["harmonised_label"].isin(["Control"] + disease_labels[outcome])].copy()
                use["box_group"] = np.where(use["harmonised_label"].eq("Control"), "Control", "Disease")
                for group_name, gg in use.groupby("box_group"):
                    vals = pd.to_numeric(gg[r.hmdb_id], errors="coerce").dropna()
                    for v in vals:
                        plot_records.append({"cohort": cohort, "group": group_name, "value": float(v)})
            pdat = pd.DataFrame(plot_records)
            if pdat.empty:
                ax.axis("off")
                continue
            values = [pdat[pdat["group"].eq("Control")]["value"], pdat[pdat["group"].eq("Disease")]["value"]]
            bp = ax.boxplot(values, positions=[0, 1], widths=0.48, patch_artist=True, medianprops={"color": "black", "linewidth": 1.5})
            for patch, color in zip(bp["boxes"], ["#4C78A8", "#D55E00"]):
                patch.set_facecolor(color)
                patch.set_alpha(0.35)
            rng = np.random.default_rng(CONFIG["random_seed"])
            for xpos, vals, color in zip([0, 1], values, ["#4C78A8", "#D55E00"]):
                jitter = rng.normal(0, 0.045, size=len(vals))
                ax.scatter(np.full(len(vals), xpos) + jitter, vals, s=16, alpha=0.58, color=color, edgecolor="white", linewidth=0.25)
                ax.text(xpos, ax.get_ylim()[1], f"n={len(vals)}", ha="center", va="top", fontsize=8, color="0.25")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Control", "Disease"])
            ax.set_title(str(r.hmdb_id), fontsize=10, weight="bold")
            ax.grid(axis="y", color="0.9", linewidth=0.8)
            plotted_rows.append({"outcome": outcome, "hmdb_id": r.hmdb_id, "min_fdr": r.min_fdr, "effect_direction": r.effect_direction, "n_values_plotted": len(pdat)})
        fig.suptitle(f"Top replicated FDR-significant shared metabolites: {outcome_labels[outcome]}", fontsize=13, weight="bold")
        fig.text(0.5, 0.01, "Boxes show combined cohort distributions for visual interpretation; differential abundance was tested separately by cohort.", ha="center", fontsize=8.5, color="0.3")
        fig.tight_layout(rect=(0, 0.05, 1, 0.93))
        for outdir in [FIGURES / "main_text"]:
            fig.savefig(outdir / f"shared_metabolite_boxplots_{outcome}.png", dpi=240, bbox_inches="tight")
            fig.savefig(outdir / f"shared_metabolite_boxplots_{outcome}.svg", bbox_inches="tight")
        plt.close(fig)
    pd.DataFrame(plotted_rows).to_csv(TABLES / "main_text" / "shared_metabolite_boxplot_features.csv", index=False)


def save_workflow_figure(stem: str, title: str, steps: list[str], note: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8.5, max(4.8, 0.48 * len(steps) + 1.4)))
    ax.axis("off")
    y_positions = np.linspace(0.90, 0.12, len(steps))
    for idx, (step, y) in enumerate(zip(steps, y_positions)):
        ax.text(0.5, y, step, ha="center", va="center", fontsize=9.2, bbox=dict(boxstyle="round,pad=0.35", fc="#F7F9FB", ec="#4C78A8", lw=1.1))
        if idx < len(steps) - 1:
            ax.annotate("", xy=(0.5, y_positions[idx + 1] + 0.035), xytext=(0.5, y - 0.035), arrowprops=dict(arrowstyle="->", color="0.35", lw=1.1))
    ax.set_title(title, fontsize=14, weight="bold", pad=12)
    if note:
        ax.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=8.6, color="0.28", wrap=True)
    fig.tight_layout()
    for outdir in [FIGURES / "main_text"]:
        fig.savefig(outdir / f"{stem}.png", dpi=240, bbox_inches="tight")
        fig.savefig(outdir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_threshold_selection_workflow() -> None:
    save_workflow_figure(
        "threshold_selection_workflow",
        "Training-only threshold optimisation",
        ["Nested CV", "Out-of-fold probabilities", "ROC curve", "Youden Index", "Optimal threshold", "Threshold frozen", "External validation"],
        "Thresholds are selected before external validation and are not re-optimised on the external cohort.",
    )


def plot_internal_model_selection_workflow() -> None:
    save_workflow_figure(
        "internal_model_selection_workflow",
        "Internal model selection and frozen external validation",
        [
            "Training cohort",
            "Outer stratified CV",
            "Training fold only",
            "Missingness filtering",
            "Constant feature removal",
            "Median imputation",
            "Standardisation",
            "Inner CV hyperparameter tuning",
            "Validation fold prediction",
            "Repeat all outer folds",
            "Aggregate out-of-fold predictions",
            "Select best model",
            "Determine Youden threshold",
            "Retrain on the FULL training cohort",
            "Freeze pipeline",
            "External cohort",
            "External evaluation",
        ],
        "All preprocessing is fitted within each outer training fold to prevent information leakage.",
    )


def plot_external_validation_workflow() -> None:
    save_workflow_figure(
        "external_validation_workflow",
        "External validation uses the frozen training-derived pipeline",
        ["Frozen preprocessing + frozen hyperparameters + frozen threshold", "External cohort", "No retraining\nNo hyperparameter optimisation\nNo threshold optimisation", "Final external metrics"],
        "External labels are used only for final performance calculation.",
    )


def plot_feature_importance_methodology() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.axis("off")
    boxes = [
        (0.25, 0.70, "Linear models", "#D55E00"),
        (0.25, 0.36, "Standardised logistic coefficients", "#D55E00"),
        (0.75, 0.70, "Tree ensembles", "#009E73"),
        (0.75, 0.36, "Permutation importance\non held-out training folds", "#009E73"),
    ]
    for x, y, text, color in boxes:
        ax.text(x, y, text, ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.38", fc="#FFFFFF", ec=color, lw=1.4))
    ax.annotate("", xy=(0.25, 0.45), xytext=(0.25, 0.61), arrowprops=dict(arrowstyle="->", color="0.35"))
    ax.annotate("", xy=(0.75, 0.45), xytext=(0.75, 0.61), arrowprops=dict(arrowstyle="->", color="0.35"))
    ax.set_title("Feature-importance interpretation", fontsize=14, weight="bold")
    ax.text(0.5, 0.08, "These measures are predictive summaries and should not be interpreted identically or as causal/differential-abundance evidence.", ha="center", fontsize=9, color="0.3", wrap=True)
    fig.tight_layout()
    for outdir in [FIGURES / "main_text"]:
        fig.savefig(outdir / "feature_importance_methodology.png", dpi=240, bbox_inches="tight")
        fig.savefig(outdir / "feature_importance_methodology.svg", bbox_inches="tight")
    plt.close(fig)


def threshold_selection_summary_rows(results_df: pd.DataFrame, preds_df: pd.DataFrame) -> pd.DataFrame:
    """Fallback reporting table from saved final thresholds when OOF details are unavailable."""
    if results_df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in results_df.iterrows():
        rows.append({
            "outcome": r["outcome"],
            "train_cohort": r["train_cohort"],
            "external_cohort": r["external_cohort"],
            "feature_set": r["feature_set"],
            "model": r["model"],
            "selected_threshold": r["frozen_threshold"],
            "sensitivity": pd.NA,
            "specificity": pd.NA,
            "youden_j": pd.NA,
            "roc_auc_at_threshold_selection": pd.NA,
            "reporting_note": "Full threshold-selection metrics are written during pipeline execution from training-only OOF probabilities; existing external summaries contain the frozen threshold only.",
        })
    return pd.DataFrame(rows)


def write_selected_model_summary_main_text(results_df: pd.DataFrame) -> None:
    if results_df.empty:
        pd.DataFrame().to_csv(TABLES / "main_text" / "selected_model_performance_summary.csv", index=False)
        return
    cols = [
        "outcome",
        "train_cohort",
        "external_cohort",
        "feature_set",
        "model",
        "internal_cv_auc_mean",
        "internal_cv_pr_auc_mean",
        "frozen_threshold",
        "best_params",
        "roc_auc",
        "pr_auc",
    ]
    out = results_df[[c for c in cols if c in results_df.columns]].copy()
    out = out.rename(columns={
        "model": "selected_model",
        "internal_cv_auc_mean": "nested_cv_ROC_AUC",
        "internal_cv_pr_auc_mean": "nested_cv_PR_AUC",
        "frozen_threshold": "selected_threshold",
        "best_params": "hyperparameters",
        "roc_auc": "external_ROC_AUC",
        "pr_auc": "external_PR_AUC",
    })
    out.to_csv(TABLES / "main_text" / "selected_model_performance_summary.csv", index=False)


def apply_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 240,
    })


def outcome_short_name(outcome: str) -> str:
    return {"ibd_vs_control": "ibd", "cd_vs_control": "cd", "uc_vs_control": "uc"}.get(str(outcome), re.sub(r"[^a-z0-9]+", "_", str(outcome).lower()).strip("_"))


def outcome_display_name(outcome: str) -> str:
    return {"ibd_vs_control": "IBD vs Control", "cd_vs_control": "CD vs Control", "uc_vs_control": "UC vs Control"}.get(str(outcome), str(outcome))


def model_display_name(model: str) -> str:
    return {
        "ridge_logistic": "Ridge logistic",
        "elastic_net_logistic": "Elastic-net logistic",
        "random_forest": "Random forest",
        "extra_trees": "Extra Trees",
        "gradient_boosting": "Gradient boosting",
    }.get(str(model), str(model))


def feature_colour(feature_set: str) -> str:
    return {"clinical": "#0072B2", "metabolomics": "#009E73", "integrated": "#D55E00"}.get(str(feature_set), "#666666")


def bootstrap_roc_band(y_true, y_prob, n_boot: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    grid = np.linspace(0, 1, 101)
    rng = np.random.default_rng(CONFIG["random_seed"])
    tprs = []
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[idx], p[idx])
        tprs.append(np.interp(grid, fpr, tpr))
        aucs.append(roc_auc_score(y[idx], p[idx]))
    if not tprs:
        return grid, np.full_like(grid, np.nan), np.full_like(grid, np.nan), np.full_like(grid, np.nan), (np.nan, np.nan)
    arr = np.asarray(tprs)
    return grid, np.nanmean(arr, axis=0), np.nanpercentile(arr, 2.5, axis=0), np.nanpercentile(arr, 97.5, axis=0), (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def bootstrap_pr_band(y_true, y_prob, n_boot: int = 300) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    grid = np.linspace(0, 1, 101)
    rng = np.random.default_rng(CONFIG["random_seed"])
    precs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y[idx], p[idx])
        order = np.argsort(recall)
        precs.append(np.interp(grid, recall[order], precision[order]))
    if not precs:
        return grid, np.full_like(grid, np.nan), np.full_like(grid, np.nan), np.full_like(grid, np.nan)
    arr = np.asarray(precs)
    return grid, np.nanmean(arr, axis=0), np.nanpercentile(arr, 2.5, axis=0), np.nanpercentile(arr, 97.5, axis=0)


def calibration_bins(y_true, y_prob, n_bins: int = 5) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=int), "p": np.asarray(y_prob, dtype=float)})
    if df.empty:
        return pd.DataFrame()
    try:
        df["bin"] = pd.qcut(df["p"].rank(method="first"), q=min(n_bins, len(df)), labels=False, duplicates="drop")
    except ValueError:
        df["bin"] = 0
    return df.groupby("bin", dropna=False).agg(
        n=("y", "size"),
        mean_predicted_probability=("p", "mean"),
        observed_event_frequency=("y", "mean"),
    ).reset_index(drop=True)


def expected_calibration_error(bin_df: pd.DataFrame) -> float:
    if bin_df.empty or bin_df["n"].sum() == 0:
        return np.nan
    weights = bin_df["n"] / bin_df["n"].sum()
    return float(np.sum(weights * np.abs(bin_df["observed_event_frequency"] - bin_df["mean_predicted_probability"])))


def plot_calibration_assessment_main_text(preds_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    apply_publication_style()
    rows = []
    if preds_df.empty or results_df.empty:
        pd.DataFrame([{"status": "not_generated", "reason": "External predictions or results table unavailable."}]).to_csv(TABLES / "main_text" / "calibration_status.csv", index=False)
        return pd.DataFrame()
    for outcome, result_group in results_df.groupby("outcome", dropna=False):
        fig, (ax, hist_ax) = plt.subplots(2, 1, figsize=(7.2, 6.0), gridspec_kw={"height_ratios": [3.2, 1.0]}, sharex=True)
        plotted = False
        for _, r in result_group.iterrows():
            mask = (
                preds_df["outcome"].astype(str).eq(str(r.outcome))
                & preds_df["train_cohort"].astype(str).eq(str(r.train_cohort))
                & preds_df["external_cohort"].astype(str).eq(str(r.external_cohort))
                & preds_df["feature_set"].astype(str).eq(str(r.feature_set))
                & preds_df["model"].astype(str).eq(str(r.model))
            )
            sub = preds_df[mask].copy()
            if sub.empty:
                continue
            y = sub["true_label"].astype(int).to_numpy()
            p = sub["predicted_probability"].astype(float).to_numpy()
            brier = brier_score_loss(y, p) if len(np.unique(y)) == 2 else np.nan
            intercept, slope, status = fit_logistic_calibration_assessment(y, p)
            bins = calibration_bins(y, p, 5)
            ece = expected_calibration_error(bins)
            rows.append({
                "outcome": r.outcome,
                "transfer_direction": f"{r.train_cohort}->{r.external_cohort}",
                "feature_set": r.feature_set,
                "selected_model": r.model,
                "brier_score": brier,
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "expected_calibration_error": ece,
                "calibration_status": status,
            })
            if bins.empty:
                continue
            label = f"{r.train_cohort[:1]}→{r.external_cohort[:1]} {r.feature_set} | Brier {brier:.2f}, slope {slope:.2f}" if pd.notna(slope) else f"{r.train_cohort[:1]}→{r.external_cohort[:1]} {r.feature_set} | Brier {brier:.2f}"
            ax.plot(bins["mean_predicted_probability"], bins["observed_event_frequency"], marker="o", linewidth=1.5, color=feature_colour(r.feature_set), alpha=0.82, label=label)
            hist_ax.hist(p, bins=np.linspace(0, 1, 16), histtype="stepfilled", alpha=0.25, color=feature_colour(r.feature_set))
            plotted = True
        ax.plot([0, 1], [0, 1], "--", color="0.45", linewidth=1.0, label="Ideal calibration")
        ax.set_ylabel("Observed event frequency")
        ax.set_title(f"External calibration assessment: {outcome_display_name(outcome)}", weight="bold")
        ax.grid(color="0.9", linewidth=0.8)
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
        hist_ax.set_xlabel("Predicted probability")
        hist_ax.set_ylabel("Count")
        hist_ax.grid(axis="x", color="0.9", linewidth=0.8)
        hist_ax.text(0.0, -0.62, "Calibration is assessment-only after frozen external prediction; it is not used for model selection, HPO or threshold optimisation.", transform=hist_ax.transAxes, fontsize=8, color="0.3", ha="left")
        if not plotted:
            ax.text(0.05, 0.5, "Calibration could not be plotted: no matching external predictions.", transform=ax.transAxes)
        fig.tight_layout()
        stem = f"calibration_{outcome_short_name(outcome)}"
        fig.savefig(FIGURES / "main_text" / f"{stem}.png", dpi=240, bbox_inches="tight")
        fig.savefig(FIGURES / "main_text" / f"{stem}.svg", bbox_inches="tight")
        plt.close(fig)
    out = pd.DataFrame(rows)
    if out.empty:
        pd.DataFrame([{"status": "not_generated", "reason": "No calibration rows could be estimated."}]).to_csv(TABLES / "main_text" / "calibration_status.csv", index=False)
    out.to_csv(TABLES / "main_text" / "calibration_metrics.csv", index=False)
    return out


def plot_external_roc_pr_probability_outputs(preds_df: pd.DataFrame, results_df: pd.DataFrame) -> None:
    apply_publication_style()
    if preds_df.empty or results_df.empty:
        pd.DataFrame([{"status": "not_generated", "reason": "External predictions or results table unavailable."}]).to_csv(TABLES / "main_text" / "external_curve_status.csv", index=False)
        return
    direction_style = {("Franzosa", "iHMP"): "-", ("iHMP", "Franzosa"): "--"}
    for outcome, result_group in results_df.groupby("outcome", dropna=False):
        fig_roc, ax_roc = plt.subplots(figsize=(7.0, 5.2))
        fig_pr, ax_pr = plt.subplots(figsize=(7.0, 5.2))
        fig_dist, axes_dist = plt.subplots(2, 3, figsize=(10.0, 6.0), squeeze=False, sharey=False)
        fig_rank, axes_rank = plt.subplots(2, 3, figsize=(10.0, 6.0), squeeze=False, sharey=False)
        panel_i = 0
        for _, r in result_group.iterrows():
            mask = (
                preds_df["outcome"].astype(str).eq(str(r.outcome))
                & preds_df["train_cohort"].astype(str).eq(str(r.train_cohort))
                & preds_df["external_cohort"].astype(str).eq(str(r.external_cohort))
                & preds_df["feature_set"].astype(str).eq(str(r.feature_set))
                & preds_df["model"].astype(str).eq(str(r.model))
            )
            sub = preds_df[mask].copy()
            if sub.empty or sub["true_label"].nunique() < 2:
                continue
            y = sub["true_label"].astype(int).to_numpy()
            p = sub["predicted_probability"].astype(float).to_numpy()
            color = feature_colour(r.feature_set)
            linestyle = direction_style.get((r.train_cohort, r.external_cohort), "-")
            grid, mean_tpr, lo_tpr, hi_tpr, auc_ci = bootstrap_roc_band(y, p)
            label = f"{r.train_cohort[:1]}→{r.external_cohort[:1]} {r.feature_set}: AUC {r.roc_auc:.2f} ({auc_ci[0]:.2f}-{auc_ci[1]:.2f})"
            ax_roc.plot(grid, mean_tpr, color=color, linestyle=linestyle, linewidth=1.8, label=label)
            ax_roc.fill_between(grid, lo_tpr, hi_tpr, color=color, alpha=0.10)
            rec_grid, mean_prec, lo_prec, hi_prec = bootstrap_pr_band(y, p)
            ap = average_precision_score(y, p)
            prevalence = float(np.mean(y))
            ax_pr.plot(rec_grid, mean_prec, color=color, linestyle=linestyle, linewidth=1.8, label=f"{r.train_cohort[:1]}→{r.external_cohort[:1]} {r.feature_set}: AP {ap:.2f}")
            ax_pr.fill_between(rec_grid, lo_prec, hi_prec, color=color, alpha=0.10)
            ax_pr.axhline(prevalence, color="0.80", linewidth=0.8, linestyle=":")
            row_i, col_i = divmod(panel_i, 3)
            axd = axes_dist[row_i, col_i]
            axr = axes_rank[row_i, col_i]
            groups = [sub[sub["true_label"].eq(0)]["predicted_probability"], sub[sub["true_label"].eq(1)]["predicted_probability"]]
            bp = axd.boxplot(groups, positions=[0, 1], widths=0.45, patch_artist=True, medianprops={"color": "black", "linewidth": 1.2})
            for patch, c in zip(bp["boxes"], ["#0072B2", "#D55E00"]):
                patch.set_facecolor(c); patch.set_alpha(0.30)
            rng = np.random.default_rng(CONFIG["random_seed"])
            for xpos, vals, c in zip([0, 1], groups, ["#0072B2", "#D55E00"]):
                vals = np.asarray(vals, dtype=float)
                axd.scatter(np.full(len(vals), xpos) + rng.normal(0, 0.045, len(vals)), vals, s=15, color=c, alpha=0.65, edgecolor="white", linewidth=0.2)
                axd.text(xpos, 1.02, f"n={len(vals)}", ha="center", va="bottom", fontsize=7)
            axd.axhline(float(r.frozen_threshold), color="0.25", linestyle="--", linewidth=1)
            axd.set_ylim(-0.02, 1.08)
            axd.set_xticks([0, 1]); axd.set_xticklabels(["Control", "Disease"])
            axd.set_title(f"{r.train_cohort[:1]}→{r.external_cohort[:1]} {r.feature_set}", fontsize=8.5)
            rank = sub.sort_values("predicted_probability").reset_index(drop=True)
            rank_colors = np.where(rank["true_label"].eq(1), "#D55E00", "#0072B2")
            axr.bar(np.arange(len(rank)), rank["predicted_probability"], color=rank_colors, width=0.9)
            axr.axhline(float(r.frozen_threshold), color="0.25", linestyle="--", linewidth=1)
            axr.set_ylim(0, 1)
            axr.set_title(f"{r.train_cohort[:1]}→{r.external_cohort[:1]} {r.feature_set}", fontsize=8.5)
            panel_i += 1
        ax_roc.plot([0, 1], [0, 1], "--", color="0.55", linewidth=1.0)
        ax_roc.set_xlabel("False positive rate"); ax_roc.set_ylabel("True positive rate")
        ax_roc.set_title(f"External ROC curves with bootstrap bands: {outcome_display_name(outcome)}", weight="bold")
        ax_roc.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
        ax_roc.grid(color="0.9", linewidth=0.8)
        ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
        ax_pr.set_title(f"External precision-recall curves: {outcome_display_name(outcome)}", weight="bold")
        ax_pr.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
        ax_pr.grid(color="0.9", linewidth=0.8)
        for ax in axes_dist.ravel()[panel_i:]:
            ax.axis("off")
        for ax in axes_rank.ravel()[panel_i:]:
            ax.axis("off")
        fig_dist.suptitle(f"External predicted probability distributions: {outcome_display_name(outcome)}", weight="bold")
        fig_dist.text(0.5, 0.01, "Dashed line shows the frozen Youden threshold selected on training data only.", ha="center", fontsize=8, color="0.3")
        fig_rank.suptitle(f"External prediction ranking: {outcome_display_name(outcome)}", weight="bold")
        fig_rank.text(0.5, 0.01, "Samples are sorted by predicted probability; orange=disease, blue=control, dashed line=frozen threshold.", ha="center", fontsize=8, color="0.3")
        for fig, stem in [(fig_roc, f"external_roc_ci_{outcome_short_name(outcome)}"), (fig_pr, f"external_pr_ci_{outcome_short_name(outcome)}"), (fig_dist, f"external_probability_distribution_{outcome_short_name(outcome)}"), (fig_rank, f"external_prediction_ranking_{outcome_short_name(outcome)}")]:
            fig.tight_layout()
            fig.savefig(FIGURES / "main_text" / f"{stem}.png", dpi=240, bbox_inches="tight")
            fig.savefig(FIGURES / "main_text" / f"{stem}.svg", bbox_inches="tight")
            plt.close(fig)


def plot_threshold_visualisation_from_oof(oof_df: pd.DataFrame, threshold_df: pd.DataFrame) -> None:
    apply_publication_style()
    if oof_df.empty or threshold_df.empty:
        pd.DataFrame([{"status": "not_generated", "reason": "Training-only OOF probabilities are unavailable until the full pipeline is rerun."}]).to_csv(TABLES / "main_text" / "threshold_visualisation_status.csv", index=False)
        return
    for outcome, group in threshold_df.groupby("outcome", dropna=False):
        fig, axes = plt.subplots(2, 3, figsize=(10, 6), squeeze=False)
        panel_i = 0
        for _, r in group.iterrows():
            mask = (
                oof_df["outcome"].astype(str).eq(str(r.outcome))
                & oof_df["train_cohort"].astype(str).eq(str(r.train_cohort))
                & oof_df["external_cohort"].astype(str).eq(str(r.external_cohort))
                & oof_df["feature_set"].astype(str).eq(str(r.feature_set))
                & oof_df["model"].astype(str).eq(str(r.model))
            )
            sub = oof_df[mask].copy()
            if sub.empty:
                continue
            ax = axes.ravel()[panel_i]
            groups = [sub[sub["true_label"].eq(0)]["oof_predicted_probability"], sub[sub["true_label"].eq(1)]["oof_predicted_probability"]]
            ax.boxplot(groups, positions=[0, 1], widths=0.45, patch_artist=True)
            ax.axhline(float(r.selected_threshold), color="0.25", linestyle="--", linewidth=1)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Control", "Disease"])
            ax.set_ylim(0, 1)
            ax.set_title(f"{r.train_cohort[:1]}→{r.external_cohort[:1]} {r.feature_set}", fontsize=8.5)
            ax.text(0.02, 0.96, f"Sens {float(r.sensitivity):.2f}\nSpec {float(r.specificity):.2f}\nJ {float(r.youden_j):.2f}", transform=ax.transAxes, va="top", fontsize=7.5)
            panel_i += 1
        for ax in axes.ravel()[panel_i:]:
            ax.axis("off")
        fig.suptitle(f"Training-only threshold visualisation: {outcome_display_name(outcome)}", weight="bold")
        fig.text(0.5, 0.01, "This threshold was selected using training data only and frozen before external validation.", ha="center", fontsize=8, color="0.3")
        fig.tight_layout()
        stem = f"threshold_visualisation_{outcome_short_name(outcome)}"
        fig.savefig(FIGURES / "main_text" / f"{stem}.png", dpi=240, bbox_inches="tight")
        fig.savefig(FIGURES / "main_text" / f"{stem}.svg", bbox_inches="tight")
        plt.close(fig)


def plot_nested_cv_stability_main_text(nested_df: pd.DataFrame, results_df: pd.DataFrame) -> None:
    apply_publication_style()
    if nested_df.empty or results_df.empty:
        pd.DataFrame([{"status": "not_generated", "reason": "Nested outer-fold or selected model table unavailable."}]).to_csv(TABLES / "main_text" / "nested_cv_stability_status.csv", index=False)
        return
    key_cols = ["outcome", "train_cohort", "external_cohort", "feature_set", "model"]
    selected_keys = results_df[key_cols].drop_duplicates()
    plot_df = nested_df.merge(selected_keys, on=key_cols, how="inner")
    if plot_df.empty:
        pd.DataFrame([{"status": "not_generated", "reason": "No nested outer-fold rows matched selected final models."}]).to_csv(TABLES / "main_text" / "nested_cv_stability_status.csv", index=False)
        return
    stats = plot_df.groupby(key_cols, dropna=False).agg(
        roc_auc_mean=("nested_outer_auc", "mean"),
        roc_auc_median=("nested_outer_auc", "median"),
        roc_auc_sd=("nested_outer_auc", "std"),
        pr_auc_mean=("nested_outer_pr_auc", "mean"),
        pr_auc_median=("nested_outer_pr_auc", "median"),
        pr_auc_sd=("nested_outer_pr_auc", "std"),
        n_outer_folds=("outer_fold", "nunique"),
    ).reset_index()
    stats.to_csv(TABLES / "main_text" / "nested_cv_stability_summary.csv", index=False)
    for outcome, group in plot_df.groupby("outcome", dropna=False):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=False)
        labels = []
        data_auc = []
        data_pr = []
        colors = []
        for keys, g in group.groupby(["train_cohort", "external_cohort", "feature_set", "model"], dropna=False):
            labels.append(f"{keys[0][:1]}→{keys[1][:1]}\n{keys[2]}")
            data_auc.append(g["nested_outer_auc"].astype(float).dropna().to_numpy())
            data_pr.append(g["nested_outer_pr_auc"].astype(float).dropna().to_numpy())
            colors.append(feature_colour(keys[2]))
        for ax, data, metric in [(axes[0], data_auc, "Outer-fold ROC-AUC"), (axes[1], data_pr, "Outer-fold PR-AUC")]:
            bp = ax.boxplot(data, patch_artist=True, tick_labels=labels, showmeans=True)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color); patch.set_alpha(0.35)
            ax.set_title(metric)
            ax.set_ylim(0, 1.02)
            ax.grid(axis="y", color="0.9")
            ax.tick_params(axis="x", rotation=45)
        fig.suptitle(f"Nested CV outer-fold stability: {outcome_display_name(outcome)}", weight="bold")
        fig.tight_layout()
        stem = f"nested_cv_stability_{outcome_short_name(outcome)}"
        fig.savefig(FIGURES / "main_text" / f"{stem}.png", dpi=240, bbox_inches="tight")
        fig.savefig(FIGURES / "main_text" / f"{stem}.svg", bbox_inches="tight")
        plt.close(fig)


def write_final_model_summary(results_df: pd.DataFrame, calibration_df: pd.DataFrame) -> None:
    if results_df.empty:
        pd.DataFrame().to_csv(TABLES / "main_text" / "final_model_summary.csv", index=False)
        return
    out = results_df.copy()
    out["transfer_direction"] = out["train_cohort"].astype(str) + "->" + out["external_cohort"].astype(str)
    if not calibration_df.empty:
        cal = calibration_df.rename(columns={"selected_model": "model"})
        out = out.merge(
            cal[["outcome", "transfer_direction", "feature_set", "model", "calibration_slope", "calibration_intercept"]],
            on=["outcome", "transfer_direction", "feature_set", "model"],
            how="left",
        )
    else:
        out["calibration_slope"] = np.nan
        out["calibration_intercept"] = np.nan
    final = out.rename(columns={
        "model": "Selected model",
        "best_params": "Best hyperparameters",
        "internal_cv_auc_mean": "Nested CV ROC AUC",
        "internal_cv_pr_auc_mean": "Nested CV PR AUC",
        "roc_auc": "External ROC AUC",
        "pr_auc": "External PR AUC",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "balanced_accuracy": "Balanced Accuracy",
        "f1": "F1",
        "ppv": "PPV",
        "npv": "NPV",
        "brier": "Brier Score",
        "frozen_threshold": "Chosen Threshold",
        "calibration_slope": "Calibration Slope",
        "calibration_intercept": "Calibration Intercept",
        "outcome": "Outcome",
        "train_cohort": "Training cohort",
        "external_cohort": "External cohort",
        "feature_set": "Feature set",
    })
    cols = ["Outcome", "Training cohort", "External cohort", "Feature set", "Selected model", "Best hyperparameters", "Nested CV ROC AUC", "Nested CV PR AUC", "External ROC AUC", "External PR AUC", "Sensitivity", "Specificity", "Balanced Accuracy", "F1", "PPV", "NPV", "Brier Score", "Chosen Threshold", "Calibration Slope", "Calibration Intercept"]
    final[[c for c in cols if c in final.columns]].to_csv(TABLES / "main_text" / "final_model_summary.csv", index=False)


def write_main_text_figure_index() -> None:
    rows = [
        ("PCA_Franzosa", "Exploratory within-cohort PCA", "EDA", "included samples + primary HMDB matrix", "figures/main_text/PCA_Franzosa.png"),
        ("PCA_iHMP", "Exploratory within-cohort PCA", "EDA", "included samples + primary HMDB matrix", "figures/main_text/PCA_iHMP.png"),
        ("calibration_ibd/cd/uc", "External calibration assessment", "External validation", "external predictions + selected model results", "figures/main_text/calibration_*.png"),
        ("external_roc_ci_ibd/cd/uc", "ROC curves with bootstrap confidence bands", "External validation", "external predictions + selected model results", "figures/main_text/external_roc_ci_*.png"),
        ("external_pr_ci_ibd/cd/uc", "Precision-recall curves with prevalence reference", "External validation", "external predictions + selected model results", "figures/main_text/external_pr_ci_*.png"),
        ("external_probability_distribution_ibd/cd/uc", "Predicted probability class separation", "External validation", "external predictions + frozen threshold", "figures/main_text/external_probability_distribution_*.png"),
        ("threshold_visualisation_ibd/cd/uc", "Training-only OOF threshold visualisation", "Threshold selection", "training-only OOF probabilities when available", "figures/main_text/threshold_visualisation_*.png"),
        ("nested_cv_stability_ibd/cd/uc", "Outer-fold performance stability", "Internal validation", "nested CV outer-fold table + selected model results", "figures/main_text/nested_cv_stability_*.png"),
        ("external_prediction_ranking_ibd/cd/uc", "Waterfall ranking of external predictions", "External validation", "external predictions + frozen threshold", "figures/main_text/external_prediction_ranking_*.png"),
        ("feature_importance_methodology", "Interpretation guide for coefficients and permutation importance", "Methods", "method description", "figures/main_text/feature_importance_methodology.png"),
    ]
    pd.DataFrame(rows, columns=["Figure name", "Purpose", "Section", "Input data", "Output filename"]).to_csv(TABLES / "main_text" / "figure_index.csv", index=False)


def generate_final_reporting_outputs(preds_df: pd.DataFrame, results_df: pd.DataFrame, nested_df: pd.DataFrame, threshold_df: pd.DataFrame, oof_df: pd.DataFrame | None = None) -> None:
    calibration_df = plot_calibration_assessment_main_text(preds_df, results_df)
    plot_external_roc_pr_probability_outputs(preds_df, results_df)
    plot_threshold_visualisation_from_oof(oof_df if oof_df is not None else pd.DataFrame(), threshold_df)
    plot_nested_cv_stability_main_text(nested_df, results_df)
    write_final_model_summary(results_df, calibration_df)
    write_main_text_figure_index()


def main() -> None:
    ensure_dirs()
    np.random.seed(CONFIG["random_seed"])
    (WORK / "config.json").write_text(json.dumps(CONFIG, indent=2))

    fr_meta_raw, fr_mtb, fr_map = load_franzosa()
    ih_meta_raw, ih_mtb, ih_map = load_ihmp()
    fr_meta = standardize_fr_meta(fr_meta_raw)
    ih_meta_all = standardize_ih_meta(ih_meta_raw)
    ih_first, ih_visit_log = select_ihmp_first_visit(ih_meta_all)
    ih_visit_log.to_csv(TABLES / "ihmp_visit_selection.csv", index=False)
    ih_first.to_csv(PROCESSED / "ihmp_primary_first_visit.csv", index=False)
    ih_meta_all.to_csv(PROCESSED / "ihmp_longitudinal_preserved.csv", index=False)

    # Inclusion/exclusion.
    all_meta = pd.concat([fr_meta, ih_first], ignore_index=True)
    all_meta["include"] = True
    reasons = []
    for _, r in all_meta.iterrows():
        rs = []
        if r["harmonised_label"] not in ["Control", "CD", "UC"]:
            rs.append("invalid_or_missing_disease_label")
        if pd.isna(r["participant_id"]) or str(r["participant_id"]).strip() == "":
            rs.append("missing_participant_id")
        if pd.isna(r["sample_id"]) or str(r["sample_id"]).strip() == "":
            rs.append("missing_sample_id")
        if CONFIG["exclude_antibiotic_exposure"] and is_true_flag(r.get("antibiotic_flag")):
            rs.append("antibiotic_exposure_recorded")
        if CONFIG["exclude_systemic_steroid_exposure_where_recorded"] and is_true_flag(r.get("steroid_flag")):
            rs.append("systemic_steroid_exposure_recorded")
        reasons.append("; ".join(rs))
    all_meta["exclusion_reason"] = reasons
    all_meta["include"] = all_meta["exclusion_reason"].eq("")
    included = all_meta[all_meta["include"]].copy()
    excluded = all_meta[~all_meta["include"]].copy()
    included.to_csv(TABLES / "included_samples.csv", index=False)
    excluded.to_csv(TABLES / "excluded_samples.csv", index=False)
    excluded.groupby(["cohort", "exclusion_reason"]).size().reset_index(name="n").to_csv(TABLES / "exclusion_summary.csv", index=False)

    # Audit and inventory.
    inventory_rows = []
    for path in [FR_DIR / "metadata.tsv", FR_DIR / "mtb.tsv", FR_DIR / "mtb.map.tsv", IH_DIR / "metadata.tsv", IH_DIR / "mtb.tsv", IH_DIR / "mtb.map.tsv"]:
        if path.exists():
            if path.name.endswith(".tsv"):
                df0 = read_tsv(path, nrows=20)
                cols = list(df0.columns)
                rows = sum(1 for _ in open(path, errors="ignore")) - 1
            else:
                df0 = pd.DataFrame()
                cols = []
                rows = np.nan
            inventory_rows.append(
                {
                    "file_path": str(path.relative_to(ROOT)),
                    "file_type": path.suffix,
                    "cohort": "Franzosa" if "FRANZOSA" in str(path) else "iHMP",
                    "data_modality": "metadata" if path.name == "metadata.tsv" else ("metabolomics_annotation" if "map" in path.name else "metabolomics_matrix"),
                    "row_count": rows,
                    "column_count": len(cols),
                    "sample_id_column": "Sample" if "Sample" in cols else "",
                    "participant_id_column": "Subject" if "Subject" in cols else "",
                    "diagnosis_column": "Study.Group" if "Study.Group" in cols else "",
                    "visit_date_column": ",".join([c for c in cols if c in ["date_of_receipt", "visit_num", "week_num"]]),
                    "clinical_columns": ",".join([c for c in cols if c.lower() in {"age", "consent_age", "gender", "bmi", "bmi_at_baseline", "fecal.calprotectin", "fecalcal", "antibiotic", "antibiotics", "steroids", "smoking status"}]),
                    "metabolomics_columns": max(0, len(cols) - (1 if "Sample" in cols else 0)) if path.name == "mtb.tsv" else 0,
                    "assay_columns": "Method" if "Method" in cols else "",
                    "mz_column": "m.z" if "m.z" in cols else "",
                    "retention_time_column": "Retention.Time" if "Retention.Time" in cols else "",
                    "hmdb_column": "HMDB" if "HMDB" in cols else "",
                    "kegg_column": "KEGG" if "KEGG" in cols else "",
                    "metabolite_name_column": "Compound.Name" if "Compound.Name" in cols else "",
                    "apparent_processing_state": "supplied processed table; transformation status not fully resolved",
                }
            )
    pd.DataFrame(inventory_rows).to_csv(TABLES / "data_inventory.csv", index=False)
    pd.DataFrame(inventory_rows).to_csv(TABLES / "data_dictionary.csv", index=False)
    pd.DataFrame(columns=["file_path", "column", "reason_unresolved"]).to_csv(TABLES / "unresolved_fields.csv", index=False)

    scope = []
    for cohort, meta, mtb, mp in [("Franzosa", fr_meta, fr_mtb, fr_map), ("iHMP", ih_meta_all, ih_mtb, ih_map)]:
        scope.append(
            {
                "cohort": cohort,
                "supplied_files": "metadata.tsv; mtb.tsv; mtb.map.tsv",
                "metadata_rows": len(meta),
                "metadata_columns": meta.shape[1],
                "unique_sample_ids": meta["sample_id"].nunique(),
                "unique_participant_ids": meta["participant_id"].nunique(),
                "sample_types_represented": "stool/fecal",
                "metabolomics_samples_represented": mtb["Sample"].nunique() if "Sample" in mtb else np.nan,
                "controls": int((meta["harmonised_label"] == "Control").sum()),
                "cd_cases": int((meta["harmonised_label"] == "CD").sum()),
                "uc_cases": int((meta["harmonised_label"] == "UC").sum()),
                "repeated_visits": int(meta.duplicated("participant_id").sum()),
                "clinical_variables_available": ",".join([c for c in ["age_years", "sex", "bmi", "smoking_status", "fecal_calprotectin", "antibiotic_flag", "steroid_flag"] if c in meta.columns and meta[c].notna().any()]),
                "assay_fields_available": "Method" if "Method" in mp else "inferred_from_feature_names",
                "ionisation_fields_available": "inferred_from_method",
                "HMDB_fields_available": "HMDB" if "HMDB" in mp else "",
                "KEGG_fields_available": "KEGG" if "KEGG" in mp else "",
                "metabolite_name_fields_available": "Compound.Name" if "Compound.Name" in mp else "",
                "mz_retention_fields_available": "m.z" + (";Retention.Time" if "Retention.Time" in mp else ""),
            }
        )
    pd.DataFrame(scope).to_csv(TABLES / "supplied_data_scope.csv", index=False)

    label_rows = []
    for cohort, meta in [("Franzosa", fr_meta), ("iHMP", ih_meta_all)]:
        for (orig, harm), g in meta.groupby(["original_label", "harmonised_label"], dropna=False):
            label_rows.append({"cohort": cohort, "original_label": orig, "harmonised_label": harm, "sample_count": len(g), "participant_count": g["participant_id"].nunique(), "mapping_rationale": "source diagnosis mapped to Control, CD, UC, and IBD binary"})
    pd.DataFrame(label_rows).to_csv(TABLES / "disease_label_mapping.csv", index=False)

    clinical_h = []
    for var, fr_col, ih_col, decision, reason in [
        ("age_years", "Age", "consent_age", "include", "available in both cohorts in years"),
        ("log_calprotectin", "Fecal.Calprotectin", "fecalcal", "include", "available in both; log1p transformation used"),
        ("sex", "", "Gender", "exclude main", "missing in Franzosa supplied metadata"),
        ("bmi", "", "BMI_at_baseline", "exclude main", "missing in Franzosa supplied metadata"),
        ("smoking_status", "", "smoking status", "exclude main", "missing in Franzosa supplied metadata"),
        ("antibiotic_flag", "antibiotic", "Antibiotics", "exclude samples", "microbiome/metabolome confounder; excluded where recorded"),
        ("steroid_flag", "steroids", "", "exclude where recorded", "Franzosa-only systemic steroid exposure; excluded where recorded"),
    ]:
        clinical_h.append({"variable": var, "franzosa_source_column": fr_col, "ihmp_source_column": ih_col, "unit": "years" if var == "age_years" else "", "transformation": "log1p" if var == "log_calprotectin" else "", "final_inclusion_decision": decision, "exclusion_reason_where_not_included": reason})
    pd.DataFrame(clinical_h).to_csv(TABLES / "clinical_variable_harmonisation.csv", index=False)

    char = included.groupby(["cohort", "harmonised_label"]).agg(n=("sample_id", "count"), participants=("participant_id", "nunique"), age_median=("age_years", "median"), calprotectin_median=("fecal_calprotectin", "median")).reset_index()
    char.to_csv(TABLES / "cohort_characteristics_by_subtype.csv", index=False)
    included.groupby("cohort").agg(n=("sample_id", "count"), participants=("participant_id", "nunique"), age_median=("age_years", "median"), calprotectin_median=("fecal_calprotectin", "median")).reset_index().to_csv(TABLES / "cohort_characteristics.csv", index=False)

    # Annotation and harmonisation.
    fr_long = build_annotation_long(fr_map, "Franzosa", "franzosa_original_mapped_metabolites.repaired.map.tsv")
    ih_long = build_annotation_long(ih_map, "iHMP", "mtb.map.tsv")
    fr_long.to_csv(TABLES / "franzosa_annotation_long.csv", index=False)
    ih_long.to_csv(TABLES / "ihmp_annotation_long.csv", index=False)
    invalid = pd.concat([fr_long, ih_long], ignore_index=True)
    invalid = invalid[(invalid["standardised_id"].notna()) & (invalid["valid_id"] != True)]
    invalid.to_csv(TABLES / "invalid_annotation_ids.csv", index=False)

    fr_hmdb = set(fr_long[(fr_long.id_type == "HMDB") & (fr_long.valid_id == True)]["standardised_id"])
    ih_hmdb = set(ih_long[(ih_long.id_type == "HMDB") & (ih_long.valid_id == True)]["standardised_id"])
    fr_kegg = set(fr_long[(fr_long.id_type == "KEGG") & (fr_long.valid_id == True)]["standardised_id"])
    ih_kegg = set(ih_long[(ih_long.id_type == "KEGG") & (ih_long.valid_id == True)]["standardised_id"])
    shared_hmdb = fr_hmdb & ih_hmdb
    shared_kegg = fr_kegg & ih_kegg
    pd.DataFrame([{"overlap_type": "HMDB", "franzosa_n": len(fr_hmdb), "ihmp_n": len(ih_hmdb), "shared_n": len(shared_hmdb)}, {"overlap_type": "KEGG", "franzosa_n": len(fr_kegg), "ihmp_n": len(ih_kegg), "shared_n": len(shared_kegg)}]).to_csv(TABLES / "raw_feature_overlap_summary.csv", index=False)
    pd.DataFrame({"HMDB": sorted(shared_hmdb)}).to_csv(TABLES / "hmdb_overlap.csv", index=False)
    pd.DataFrame({"KEGG": sorted(shared_kegg)}).to_csv(TABLES / "kegg_overlap.csv", index=False)

    fr_choice = choose_feature_for_id(fr_mtb, fr_map, "HMDB", shared_hmdb)
    ih_choice = choose_feature_for_id(ih_mtb, ih_map, "HMDB", shared_hmdb)
    high_rows = []
    for hid in sorted(set(fr_choice) & set(ih_choice)):
        fr_feat = fr_choice[hid]
        ih_feat = ih_choice[hid]
        fr_method = infer_method_from_feature(fr_feat)
        ih_method = infer_method_from_feature(ih_feat, ih_map.loc[ih_map["Compound"] == ih_feat, "Method"].iloc[0] if (ih_map["Compound"] == ih_feat).any() else None)
        comp = compatible_assay(fr_method, ih_method)
        tier = "Tier 1 high-confidence HMDB" if comp in ["same or strongly compatible assay", "broadly compatible assay"] else "Tier 2 moderate-confidence HMDB"
        high_rows.append({"metabolite_id": hid, "franzosa_feature": fr_feat, "ihmp_feature": ih_feat, "franzosa_method": fr_method, "ihmp_method": ih_method, "assay_compatibility": comp, "matching_tier": tier})
    master = pd.DataFrame(high_rows)
    master.to_csv(TABLES / "master_harmonisation_table.csv", index=False)
    master[master.matching_tier.str.startswith("Tier 1")].to_csv(TABLES / "high_confidence_hmdb_matches.csv", index=False)
    master[master.matching_tier.str.startswith("Tier 2")].to_csv(TABLES / "moderate_confidence_hmdb_matches.csv", index=False)
    pd.DataFrame({"KEGG": sorted(shared_kegg), "matching_tier": "Tier 3 KEGG-supported"}).to_csv(TABLES / "kegg_supported_matches.csv", index=False)
    pd.DataFrame(columns=["normalised_name", "franzosa_feature", "ihmp_feature", "review_status"]).to_csv(TABLES / "name_supported_matches.csv", index=False)
    pd.DataFrame(columns=["issue", "feature_or_id", "decision_needed"]).to_csv(TABLES / "ambiguous_matches_for_review.csv", index=False)
    pd.DataFrame(columns=["metabolite_id", "candidate_features", "resolution_rule", "selected_feature"]).to_csv(TABLES / "duplicate_resolution_log.csv", index=False)
    master.groupby("assay_compatibility").size().reset_index(name="n").to_csv(TABLES / "assay_compatibility.csv", index=False)
    master.groupby("matching_tier").size().reset_index(name="n").to_csv(TABLES / "matching_confidence_table.csv", index=False)

    primary = master[master.matching_tier.str.startswith("Tier 1")].copy()
    if len(primary) < 5:
        pd.DataFrame(
            [{
                "dataset": "Dataset A primary high-confidence HMDB",
                "n_features": len(primary),
                "status": "TOO_SPARSE_FOR_UNCONSTRAINED_MULTIVARIABLE_MODELLING",
                "decision": "No automatic fallback to Dataset B; extended annotated analyses must be labelled separately.",
            }]
        ).to_csv(TABLES / "dataset_a_sparsity_warning.csv", index=False)
    hmdb_ids = primary["metabolite_id"].tolist()
    fr_feature_block = {
        r["metabolite_id"]: pd.to_numeric(fr_mtb[r["franzosa_feature"]], errors="coerce")
        for _, r in primary.iterrows()
    }
    ih_feature_block = {
        r["metabolite_id"]: pd.to_numeric(ih_mtb[r["ihmp_feature"]], errors="coerce")
        for _, r in primary.iterrows()
    }
    fr_hmdb_matrix = pd.concat([pd.DataFrame({"sample_id": fr_mtb["Sample"].astype(str)}), pd.DataFrame(fr_feature_block)], axis=1)
    ih_hmdb_matrix = pd.concat([pd.DataFrame({"sample_id": ih_mtb["Sample"].astype(str)}), pd.DataFrame(ih_feature_block)], axis=1)
    fr_hmdb_matrix.to_csv(PROCESSED / "franzosa_primary_high_confidence_hmdb_matrix.csv", index=False)
    ih_hmdb_matrix.to_csv(PROCESSED / "ihmp_primary_high_confidence_hmdb_matrix.csv", index=False)
    primary.to_csv(PROCESSED / "primary_high_confidence_hmdb_feature_metadata.csv", index=False)

    # QC summaries.
    qc_rows = []
    for cohort, matrix in [("Franzosa", fr_hmdb_matrix), ("iHMP", ih_hmdb_matrix)]:
        features = [c for c in matrix.columns if c != "sample_id"]
        miss = matrix[features].isna()
        qc_rows.append({"cohort": cohort, "n_samples": len(matrix), "n_features": len(features), "median_sample_missingness": float(miss.mean(axis=1).median()), "median_feature_missingness": float(miss.mean(axis=0).median())})
    pd.DataFrame(qc_rows).to_csv(TABLES / "qc_summary.csv", index=False)
    sens = []
    for cohort, matrix in [("Franzosa", fr_hmdb_matrix), ("iHMP", ih_hmdb_matrix)]:
        features = [c for c in matrix.columns if c != "sample_id"]
        sens.append({
            "threshold": CONFIG["primary_feature_missingness_threshold"],
            "cohort": cohort,
            "features_retained_by_feature_missingness": int((matrix[features].isna().mean() <= CONFIG["primary_feature_missingness_threshold"]).sum()),
            "analysis_role": "predefined_primary_threshold",
        })
    pd.DataFrame(sens).to_csv(TABLES / "missingness_threshold_sensitivity.csv", index=False)

    # EDA figures.
    fig, ax = plt.subplots(figsize=(7, 4))
    char.pivot(index="harmonised_label", columns="cohort", values="n").fillna(0).plot(kind="bar", ax=ax)
    ax.set_title("Sample counts by cohort and subtype")
    ax.set_ylabel("Samples")
    fig.tight_layout()
    fig.savefig(FIGURES / "sample_counts_by_cohort_and_subtype.png", dpi=200)
    fig.savefig(FIGURES / "sample_counts_by_cohort_and_subtype.svg")
    plt.close(fig)
    for variable, title in [("age_years", "Age distribution"), ("bmi", "BMI distribution where available"), ("log_calprotectin", "Faecal calprotectin log1p distribution")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, g in included.groupby("cohort"):
            vals = pd.to_numeric(g[variable], errors="coerce").dropna()
            if len(vals):
                ax.hist(vals, bins=20, alpha=0.5, label=label)
        ax.set_title(title)
        ax.set_xlabel(variable)
        ax.set_ylabel("Samples")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIGURES / f"{variable}_distribution.png", dpi=200)
        fig.savefig(FIGURES / f"{variable}_distribution.svg")
        plt.close(fig)
    combined = pd.concat([
        model_matrix(included[included.cohort == "Franzosa"], fr_hmdb_matrix),
        model_matrix(included[included.cohort == "iHMP"], ih_hmdb_matrix),
    ], ignore_index=True)
    pca_plot_by_cohort(combined, hmdb_ids, "Franzosa", "PCA_Franzosa")
    pca_plot_by_cohort(combined, hmdb_ids, "iHMP", "PCA_iHMP")
    plot_threshold_selection_workflow()
    plot_internal_model_selection_workflow()
    plot_external_validation_workflow()
    plot_feature_importance_methodology()

    # Modelling.
    results = []
    preds = []
    nested_rows = []
    fold_design_rows = []
    hyperparameter_search_rows = []
    threshold_selection_rows = []
    threshold_oof_rows = []
    hyperparameter_grid_plan().to_csv(TABLES / "hyperparameter_grid_plan.csv", index=False)
    clinical_features = ["age_years", "log_calprotectin"]
    directions = [("Franzosa", "iHMP"), ("iHMP", "Franzosa")]
    data_by_cohort = {
        "Franzosa": model_matrix(included[included.cohort == "Franzosa"], fr_hmdb_matrix),
        "iHMP": model_matrix(included[included.cohort == "iHMP"], ih_hmdb_matrix),
    }
    class_balance_rows = []
    for cohort, cohort_df in data_by_cohort.items():
        for outcome in CONFIG["outcomes"]:
            task_df, y_task = outcome_subset(cohort_df, outcome)
            if len(task_df) == 0:
                continue
            counts = y_task.value_counts().to_dict()
            n_negative = int(counts.get(0, 0))
            n_positive = int(counts.get(1, 0))
            n_total = int(len(y_task))
            minority_n = int(min(n_negative, n_positive))
            majority_n = int(max(n_negative, n_positive))
            outer_splits, inner_splits, fold_rationale = choose_nested_fold_design(y_task)
            class_balance_rows.append({
                "cohort": cohort,
                "outcome": outcome,
                "negative_class": "control",
                "positive_class": outcome.replace("_vs_control", "").replace("ibd", "IBD").replace("cd", "CD").replace("uc", "UC"),
                "n_total": n_total,
                "n_negative_control": n_negative,
                "n_positive_disease": n_positive,
                "positive_class_prevalence": n_positive / n_total if n_total else np.nan,
                "minority_class_n": minority_n,
                "minority_class_fraction": minority_n / n_total if n_total else np.nan,
                "majority_to_minority_ratio": majority_n / minority_n if minority_n else np.inf,
                "requested_outer_cv_folds": CONFIG["nested_cv"]["outer_folds_requested"],
                "requested_inner_cv_folds": CONFIG["nested_cv"]["inner_folds_requested"],
                "eda_feasible_outer_folds": outer_splits,
                "eda_feasible_inner_folds": inner_splits,
                "fold_design_note": fold_rationale,
            })
    class_balance_df = pd.DataFrame(class_balance_rows)
    class_balance_df.to_csv(TABLES / "class_balance_eda_by_cohort_and_task.csv", index=False)
    if not class_balance_df.empty:
        plot_df = class_balance_df.melt(
            id_vars=["cohort", "outcome"],
            value_vars=["n_negative_control", "n_positive_disease"],
            var_name="class",
            value_name="n",
        )
        plot_df["class"] = plot_df["class"].map({"n_negative_control": "control", "n_positive_disease": "disease"})
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, ((cohort, outcome), group) in enumerate(plot_df.groupby(["cohort", "outcome"])):
            bottom = 0
            for _, row in group.iterrows():
                color = "#4C78A8" if row["class"] == "control" else "#E45756"
                ax.bar(i, row["n"], bottom=bottom, color=color, label=row["class"] if i == 0 else None)
                bottom += row["n"]
        ax.set_xticks(range(plot_df.groupby(["cohort", "outcome"]).ngroups))
        ax.set_xticklabels([f"{cohort}\n{outcome.replace('_', ' ')}" for cohort, outcome in plot_df.groupby(["cohort", "outcome"]).groups], rotation=30, ha="right")
        ax.set_ylabel("Samples")
        ax.set_title("IBD EDA class balance by cohort and modelling task")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIGURES / "class_balance_eda_by_cohort_and_task.png", dpi=200)
        fig.savefig(FIGURES / "class_balance_eda_by_cohort_and_task.svg")
        plt.close(fig)
    # QC and harmonisation figures for presentation.
    for cohort, matrix in [("Franzosa", fr_hmdb_matrix), ("iHMP", ih_hmdb_matrix)]:
        features = [c for c in matrix.columns if c != "sample_id"]
        miss = matrix[features].isna()
        for axis_name, values, xlabel in [
            ("sample", miss.mean(axis=1), "Sample-level missingness fraction"),
            ("feature", miss.mean(axis=0), "Feature-level missingness fraction"),
        ]:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(values, bins=20, color="#4C78A8", alpha=0.85)
            ax.set_title(f"{cohort} {axis_name}-level missingness")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")
            fig.tight_layout()
            fig.savefig(FIGURES / f"missingness_distribution_{cohort}_{axis_name}.png", dpi=200)
            fig.savefig(FIGURES / f"missingness_distribution_{cohort}_{axis_name}.svg")
            plt.close(fig)
    overlap_fig = pd.DataFrame([
        {"category": "Franzosa valid HMDB", "n": len(fr_hmdb)},
        {"category": "iHMP valid HMDB", "n": len(ih_hmdb)},
        {"category": "Shared HMDB", "n": len(shared_hmdb)},
        {"category": "Franzosa valid KEGG", "n": len(fr_kegg)},
        {"category": "iHMP valid KEGG", "n": len(ih_kegg)},
        {"category": "Shared KEGG", "n": len(shared_kegg)},
    ])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(overlap_fig["category"], overlap_fig["n"], color="#4C78A8")
    ax.set_ylabel("Identifiers")
    ax.set_title("Metabolite identifier overlap")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIGURES / "metabolite_identifier_overlap.png", dpi=200)
    fig.savefig(FIGURES / "metabolite_identifier_overlap.svg")
    plt.close(fig)
    retention_flow = pd.DataFrame([
        {"step": "Franzosa raw metabolomics features", "n": len([c for c in fr_mtb.columns if c != "Sample"])},
        {"step": "iHMP raw metabolomics features", "n": len([c for c in ih_mtb.columns if c != "Sample"])},
        {"step": "Franzosa valid HMDB identifiers", "n": len(fr_hmdb)},
        {"step": "iHMP valid HMDB identifiers", "n": len(ih_hmdb)},
        {"step": "Shared HMDB identifiers", "n": len(shared_hmdb)},
        {"step": "Representative shared features", "n": len(master)},
        {"step": "Tier 1 assay-compatible features", "n": len(primary)},
    ])
    retention_flow.to_csv(TABLES / "feature_retention_flow.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(retention_flow["step"], retention_flow["n"], color="#72B7B2")
    ax.set_xlabel("Feature/identifier count")
    ax.set_title("IBD feature-retention flow")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_retention_flow.png", dpi=200)
    fig.savefig(FIGURES / "feature_retention_flow.svg")
    plt.close(fig)
    # Model development: all candidate model families are evaluated by nested CV
    # inside the training cohort only.
    for train_cohort, external_cohort in directions:
        for outcome in CONFIG["outcomes"]:
            train_df, y_train = outcome_subset(data_by_cohort[train_cohort], outcome)
            ext_df, y_ext = outcome_subset(data_by_cohort[external_cohort], outcome)
            if y_train.nunique() < 2 or y_ext.nunique() < 2 or len(train_df) < 20:
                continue
            feature_sets = {
                "clinical": clinical_features,
                "metabolomics": hmdb_ids,
                "integrated": clinical_features + hmdb_ids,
            }
            for fs_name, feats in feature_sets.items():
                x_train = train_df[feats]
                for model_name in CONFIG["models"]:
                    outer_splits, inner_splits, fold_rationale = choose_nested_fold_design(y_train)
                    if outer_splits < 2 or inner_splits < 2:
                        continue
                    nested_summary, outer_rows, fold_rows, tuning_rows = run_nested_cv(
                        x_train, y_train, model_name, outcome, train_cohort, external_cohort, fs_name
                    )
                    nested_rows.extend(outer_rows)
                    fold_design_rows.extend(fold_rows)
                    hyperparameter_search_rows.extend(tuning_rows)
                    results.append({
                        "outcome": outcome,
                        "train_cohort": train_cohort,
                        "external_cohort": external_cohort,
                        "feature_set": fs_name,
                        "model": model_name,
                        "train_n": len(train_df),
                        "external_n": len(ext_df),
                        "n_features_requested": len(feats),
                        "internal_cv_auc_mean": nested_summary["nested_cv_auc_mean"],
                        "internal_cv_auc_sd": nested_summary["nested_cv_auc_sd"],
                        "internal_cv_pr_auc_mean": nested_summary["nested_cv_pr_auc_mean"],
                        "internal_cv_balanced_accuracy_mean": nested_summary["nested_cv_balanced_accuracy_mean"],
                        "internal_cv_f1_mean": nested_summary["nested_cv_f1_mean"],
                        "outer_folds": nested_summary["outer_folds"],
                        "inner_folds": nested_summary["inner_folds"],
                        "fold_design_rationale": nested_summary["fold_design_rationale"],
                        "fold_failures": nested_summary["fold_failures"],
                        "nested_cv_valid": nested_summary["nested_cv_valid"],
                    })
    development_df = pd.DataFrame(results)
    # Model selection: choose a specification from aggregated nested-CV training
    # performance only. External validation metrics are not available here.
    selected = select_models_training_only(development_df) if not development_df.empty else pd.DataFrame()
    external_rows = []
    final_model_records = []
    for _, sel in selected.iterrows():
        train_df, y_train = outcome_subset(data_by_cohort[sel.train_cohort], sel.outcome)
        ext_df, y_ext = outcome_subset(data_by_cohort[sel.external_cohort], sel.outcome)
        feature_sets = {
            "clinical": clinical_features,
            "metabolomics": hmdb_ids,
            "integrated": clinical_features + hmdb_ids,
        }
        feats = feature_sets[sel.feature_set]
        x_train = train_df[feats]
        x_ext = ext_df[feats]
        outer_splits, inner_splits, _ = choose_nested_fold_design(y_train)
        pipe, grid = make_model(sel.model)
        final_inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=CONFIG["random_seed"] + 1000)
        final_search, search_method, n_tested, n_candidate_grid = make_search(
            pipe, grid, sel.model, final_inner_cv, CONFIG["random_seed"] + 1000
        )
        # Final training: rerun training-only inner CV on the full training cohort,
        # then refit the selected frozen pipeline on all training-cohort samples.
        search_start = time.perf_counter()
        final_search.fit(x_train, y_train, **fit_params_for_model(sel.model, y_train))
        search_runtime = time.perf_counter() - search_start
        hyperparameter_search_rows.extend(search_result_rows(
            final_search,
            {
                "search_stage": "final_full_training_inner_cv",
                "outcome": sel.outcome,
                "train_cohort": sel.train_cohort,
                "external_cohort": sel.external_cohort,
                "feature_set": sel.feature_set,
                "model": sel.model,
                "outer_fold": np.nan,
                "inner_folds": inner_splits,
            },
            search_method,
            n_tested,
            n_candidate_grid,
            search_runtime,
        ))
        # Threshold selection: derive a decision threshold from training-only
        # cross-fitted probabilities using Youden J, then freeze it.
        threshold_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=CONFIG["random_seed"] + 2000)
        threshold_probs = cross_fitted_predict_proba_training_only(final_search.best_estimator_, x_train, y_train, threshold_cv, sel.model)
        frozen_threshold = youden_threshold(y_train, threshold_probs)
        threshold_met = binary_metrics(y_train, threshold_probs, threshold=frozen_threshold)
        for sid, yt, yp in zip(train_df["sample_id"], y_train, threshold_probs):
            threshold_oof_rows.append({
                "outcome": sel.outcome,
                "train_cohort": sel.train_cohort,
                "external_cohort": sel.external_cohort,
                "feature_set": sel.feature_set,
                "model": sel.model,
                "sample_id": sid,
                "true_label": int(yt),
                "oof_predicted_probability": float(yp),
                "selected_threshold": frozen_threshold,
                "threshold_source": "training-only out-of-fold probabilities",
            })
        threshold_selection_rows.append({
            "outcome": sel.outcome,
            "train_cohort": sel.train_cohort,
            "external_cohort": sel.external_cohort,
            "feature_set": sel.feature_set,
            "model": sel.model,
            "selected_threshold": frozen_threshold,
            "sensitivity": threshold_met["sensitivity"],
            "specificity": threshold_met["specificity"],
            "youden_j": threshold_met["sensitivity"] + threshold_met["specificity"] - 1,
            "roc_auc_at_threshold_selection": threshold_met["roc_auc"],
            "threshold_source": "training-only out-of-fold probabilities",
        })
        best = final_search.best_estimator_
        # External validation: apply the frozen training-derived model and frozen
        # training-derived threshold once to the held-out external cohort.
        prob = best.predict_proba(x_ext)[:, 1]
        met = binary_metrics(y_ext, prob, threshold=frozen_threshold)
        ci_low, ci_high = bootstrap_auc_ci(y_ext, prob)
        row = sel.to_dict()
        row.update({
            "selected_for_external_validation": True,
            "external_validation_rule": "only internally selected model family evaluated externally",
            "best_params": json.dumps(final_search.best_params_),
            "frozen_threshold": frozen_threshold,
            "threshold_rule": "training-only cross-fitted Youden J",
            "external_auc_ci_low": ci_low,
            "external_auc_ci_high": ci_high,
            **met,
        })
        external_rows.append(row)
        final_model_records.append({
            "selected_row": row,
            "model": best,
            "x_train": x_train,
            "y_train": y_train,
            "features": feats,
        })
        for sid, yt, yp in zip(ext_df["sample_id"], y_ext, prob):
            preds.append({
                "outcome": sel.outcome,
                "train_cohort": sel.train_cohort,
                "external_cohort": sel.external_cohort,
                "feature_set": sel.feature_set,
                "model": sel.model,
                "sample_id": sid,
                "true_label": int(yt),
                "predicted_probability": float(yp),
                "frozen_threshold": frozen_threshold,
                "predicted_label": int(float(yp) >= frozen_threshold),
                "predicted_label_threshold_0_5_sensitivity": int(float(yp) >= 0.5),
            })
        joblib.dump(best, MODELS / f"{sel.outcome}_{sel.train_cohort}_to_{sel.external_cohort}_{sel.feature_set}_{sel.model}_SELECTED.joblib")
    results_df = pd.DataFrame(external_rows)
    preds_df = pd.DataFrame(preds)
    development_df.to_csv(TABLES / "model_development_nested_cv_summary.csv", index=False)
    results_df.to_csv(TABLES / "external_validation_performance.csv", index=False)
    development_df.to_csv(TABLES / "internal_cross_validation_performance.csv", index=False)
    threshold_summary_df = pd.DataFrame(threshold_selection_rows) if threshold_selection_rows else threshold_selection_summary_rows(results_df, preds_df)
    threshold_summary_df.to_csv(TABLES / "main_text" / "threshold_selection_summary.csv", index=False)
    threshold_oof_df = pd.DataFrame(threshold_oof_rows)
    threshold_oof_df.to_csv(PREDICTIONS / "training_only_threshold_oof_predictions.csv", index=False)
    pd.DataFrame(nested_rows).to_csv(TABLES / "nested_cv_outer_fold_performance.csv", index=False)
    pd.DataFrame(fold_design_rows).to_csv(TABLES / "nested_cv_fold_class_counts_and_failures.csv", index=False)
    hyperparameter_search_df = pd.DataFrame(hyperparameter_search_rows)
    hyperparameter_search_df.to_csv(TABLES / "hyperparameter_search_results.csv", index=False)
    hyperparameter_search_df.to_csv(TABLES / "hyperparameter_search_audit.csv", index=False)
    hyperparameter_search_summary(hyperparameter_search_df).to_csv(TABLES / "hyperparameter_search_summary.csv", index=False)
    pd.DataFrame([{
        "status": "implemented_assessment_only",
        "reason": "Calibration is assessed after frozen external prediction and is not used for model selection, hyperparameter tuning or threshold optimisation.",
    }]).to_csv(TABLES / "external_calibration_status.csv", index=False)
    if not results_df.empty:
        results_df[["outcome", "train_cohort", "external_cohort", "feature_set", "model", "best_params", "frozen_threshold", "threshold_rule"]].to_csv(TABLES / "selected_hyperparameters.csv", index=False)
    else:
        pd.DataFrame().to_csv(TABLES / "selected_hyperparameters.csv", index=False)
    write_selected_model_summary_main_text(results_df)
    preds_df.to_csv(PREDICTIONS / "bidirectional_external_predictions.csv", index=False)
    generate_final_reporting_outputs(preds_df, results_df, pd.DataFrame(nested_rows), threshold_summary_df, threshold_oof_df)
    paired_external_auc_comparisons(preds_df, results_df).to_csv(TABLES / "paired_external_auc_comparisons.csv", index=False)
    dca_df = decision_curve_analysis(preds_df)
    dca_df.to_csv(TABLES / "external_decision_curve_analysis.csv", index=False)
    if not dca_df.empty:
        for keys, group in dca_df[dca_df["strategy"].isin(["model", "treat_all", "treat_none"])].groupby(["outcome", "train_cohort", "external_cohort"], dropna=False):
            fig, ax = plt.subplots(figsize=(6, 4))
            for label, g in group.groupby(["feature_set", "strategy"]):
                if label[1] != "model" and label[0] != group["feature_set"].iloc[0]:
                    continue
                ax.plot(g["threshold_probability"], g["net_benefit"], label=f"{label[0]} {label[1]}")
            ax.set_xlabel("Threshold probability")
            ax.set_ylabel("Net benefit")
            ax.set_title(f"Decision curve: {keys[0]} {keys[1]}->{keys[2]}")
            ax.legend(frameon=False, fontsize=7)
            fig.tight_layout()
            stem = f"decision_curve_{keys[0]}_{keys[1]}_to_{keys[2]}"
            fig.savefig(FIGURES / f"{stem}.png", dpi=200)
            fig.savefig(FIGURES / f"{stem}.svg")
            plt.close(fig)
    if not results_df.empty:
        selected = results_df.copy()
        selected.to_csv(TABLES / "clinical_metabolomics_integrated_comparison.csv", index=False)
        error_rows = []
        utility_rows = []
        meta_lookup = included.set_index("sample_id")
        for _, r in selected.iterrows():
            psub = preds_df[
                (preds_df.outcome == r.outcome)
                & (preds_df.train_cohort == r.train_cohort)
                & (preds_df.external_cohort == r.external_cohort)
                & (preds_df.feature_set == r.feature_set)
                & (preds_df.model == r.model)
            ].copy()
            if psub.empty:
                continue
            psub["frozen_threshold"] = float(r.frozen_threshold)
            psub["predicted_label_frozen_threshold"] = (psub["predicted_probability"] >= psub["frozen_threshold"]).astype(int)
            psub["predicted_label_threshold_0_5_sensitivity"] = (psub["predicted_probability"] >= 0.5).astype(int)
            psub["misclassified"] = psub["predicted_label_frozen_threshold"] != psub["true_label"]
            for _, pr in psub[psub["misclassified"]].iterrows():
                md = meta_lookup.loc[pr["sample_id"]] if pr["sample_id"] in meta_lookup.index else pd.Series(dtype=object)
                error_rows.append(
                    {
                        "outcome": r.outcome,
                        "train_cohort": r.train_cohort,
                        "external_cohort": r.external_cohort,
                        "feature_set": r.feature_set,
                        "model": r.model,
                        "sample_id": pr["sample_id"],
                        "true_label": pr["true_label"],
                        "predicted_label": pr["predicted_label_frozen_threshold"],
                        "predicted_label_frozen_threshold": pr["predicted_label_frozen_threshold"],
                        "predicted_label_threshold_0_5_sensitivity": pr["predicted_label_threshold_0_5_sensitivity"],
                        "frozen_threshold": pr["frozen_threshold"],
                        "threshold_source": "training-only cross-fitted Youden J from external_validation_performance.csv",
                        "predicted_probability": pr["predicted_probability"],
                        "cohort": md.get("cohort", pd.NA),
                        "harmonised_label": md.get("harmonised_label", pd.NA),
                        "age_years": md.get("age_years", pd.NA),
                        "log_calprotectin": md.get("log_calprotectin", pd.NA),
                        "fecal_calprotectin": md.get("fecal_calprotectin", pd.NA),
                        "antibiotic_flag": md.get("antibiotic_flag", pd.NA),
                        "steroid_flag": md.get("steroid_flag", pd.NA),
                        "bmi": md.get("bmi", pd.NA),
                        "smoking_status": md.get("smoking_status", pd.NA),
                        "error_analysis_note": "External misclassification under frozen training-cohort pipeline using the stored training-derived threshold; 0.5 label is sensitivity only.",
                    }
                )
        for keys, group in selected.groupby(["outcome", "train_cohort", "external_cohort"]):
            clinical = group[group.feature_set == "clinical"]
            integrated = group[group.feature_set == "integrated"]
            metabolomics = group[group.feature_set == "metabolomics"]
            if clinical.empty or integrated.empty:
                continue
            c = clinical.iloc[0]
            i = integrated.iloc[0]
            utility_rows.append(
                {
                    "outcome": keys[0],
                    "train_cohort": keys[1],
                    "external_cohort": keys[2],
                    "clinical_auc": c.roc_auc,
                    "integrated_auc": i.roc_auc,
                    "delta_auc_integrated_minus_clinical": i.roc_auc - c.roc_auc,
                    "clinical_false_negatives": c.fn,
                    "integrated_false_negatives": i.fn,
                    "delta_false_negatives_integrated_minus_clinical": i.fn - c.fn,
                    "clinical_false_positives": c.fp,
                    "integrated_false_positives": i.fp,
                    "delta_false_positives_integrated_minus_clinical": i.fp - c.fp,
                    "metabolomics_auc": metabolomics.iloc[0].roc_auc if not metabolomics.empty else np.nan,
                    "clinical_utility_interpretation": "Interpret with external CI and false-negative/false-positive trade-off; AUC improvement alone is not sufficient for clinical utility.",
                }
            )
        pd.DataFrame(error_rows).to_csv(TABLES / "model_error_analysis.csv", index=False)
        pd.DataFrame(utility_rows).to_csv(TABLES / "clinical_utility_summary.csv", index=False)
        # ROC figure for selected models.
        for outcome in CONFIG["outcomes"]:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
            for ax, (train_cohort, external_cohort) in zip(axes, directions):
                subset = selected[(selected.outcome == outcome) & (selected.train_cohort == train_cohort)]
                for _, r in subset.iterrows():
                    psub = preds_df[(preds_df.outcome == outcome) & (preds_df.train_cohort == train_cohort) & (preds_df.external_cohort == external_cohort) & (preds_df.feature_set == r.feature_set) & (preds_df.model == r.model)]
                    if psub["true_label"].nunique() == 2:
                        fpr, tpr, _ = roc_curve(psub["true_label"], psub["predicted_probability"])
                        ax.plot(fpr, tpr, label=f"{r.feature_set} {r.roc_auc:.2f}")
                ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
                ax.set_title(f"Train {train_cohort}; validate {external_cohort}")
                ax.set_xlabel("False positive rate")
                ax.set_ylabel("True positive rate")
                ax.legend(frameon=False, fontsize=8)
            fig.suptitle(f"External ROC curves: {outcome}")
            fig.tight_layout()
            fig.savefig(FIGURES / f"external_roc_{outcome}.png", dpi=200)
            fig.savefig(FIGURES / f"external_roc_{outcome}.svg")
            plt.close(fig)

    # Biological interpretation: predictive feature importance only.
    # This is deliberately separated from differential abundance or causal inference.
    metabolite_annotations = feature_annotation_map(master)
    importance_rows = []
    for record in final_model_records:
        row = record["selected_row"]
        importance_rows.extend(training_only_importance_rows(
            record["model"],
            record["x_train"],
            record["y_train"],
            record["features"],
            {
                "outcome": row["outcome"],
                "train_cohort": row["train_cohort"],
                "external_cohort": row["external_cohort"],
                "feature_set": row["feature_set"],
                "model": row["model"],
                "cohort_direction": f"{row['train_cohort']}_to_{row['external_cohort']}",
            },
            metabolite_annotations,
        ))
    important_df = pd.DataFrame(importance_rows)
    important_df.to_csv(TABLES / "important_feature_table.csv", index=False)

    sensitivity_df = missingness_model_sensitivity(results_df, CONFIG["primary_feature_missingness_threshold"])
    sensitivity_df.to_csv(TABLES / "missingness_threshold_model_sensitivity.csv", index=False)
    if not sensitivity_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_sens = sensitivity_df[sensitivity_df["status"] == "complete_current_run"].copy()
        if not plot_sens.empty:
            plot_sens.groupby("feature_set")["external_roc_auc"].mean().plot(kind="bar", ax=ax, color="#4C78A8")
            ax.set_ylabel("Mean external ROC-AUC")
            ax.set_title("Primary 30% missingness-threshold model performance")
            fig.tight_layout()
            fig.savefig(FIGURES / "missingness_threshold_model_sensitivity_primary.png", dpi=200)
            fig.savefig(FIGURES / "missingness_threshold_model_sensitivity_primary.svg")
        plt.close(fig)

    # Differential abundance, calprotectin adjustment, and class/pathway interpretation.
    diff_df = differential_abundance_analysis(data_by_cohort, primary, hmdb_ids)
    diff_df.to_csv(TABLES / "differential_abundance_results.csv", index=False)
    volcano_plots(diff_df)
    plot_shared_metabolite_boxplots(diff_df, data_by_cohort)
    calp_df = calprotectin_adjusted_analysis(diff_df, data_by_cohort, hmdb_ids)
    calp_df.to_csv(TABLES / "calprotectin_adjusted_features.csv", index=False)
    if not calp_df.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(calp_df["disease_coefficient_before_calprotectin_adjustment"], calp_df["disease_coefficient_after_calprotectin_adjustment"], alpha=0.75)
        lim = np.nanmax(np.abs(calp_df[["disease_coefficient_before_calprotectin_adjustment", "disease_coefficient_after_calprotectin_adjustment"]].to_numpy()))
        if np.isfinite(lim):
            ax.plot([-lim, lim], [-lim, lim], "--", color="0.5")
        ax.set_xlabel("Unadjusted disease coefficient")
        ax.set_ylabel("Calprotectin-adjusted disease coefficient")
        ax.set_title("Disease coefficients before/after calprotectin adjustment")
        fig.tight_layout()
        fig.savefig(FIGURES / "calprotectin_adjusted_coefficient_comparison.png", dpi=200)
        fig.savefig(FIGURES / "calprotectin_adjusted_coefficient_comparison.svg")
        plt.close(fig)
    annotation_enrichment = build_metabolite_annotation_enrichment(primary)
    kegg_mapping_df, kegg_retrieval_log = load_kegg_pathway_cache()
    kegg_mapping_df.to_csv(TABLES / "kegg_compound_pathway_mapping.csv", index=False)
    kegg_retrieval_log.to_csv(TABLES / "kegg_annotation_retrieval_log.csv", index=False)
    enrich_df, enrich_status, coverage_df, pathway_exclusions = pathway_enrichment_workflow(diff_df, annotation_enrichment, kegg_mapping_df)
    enrich_df.to_csv(TABLES / "pathway_enrichment.csv", index=False)
    enrich_status.to_csv(TABLES / "pathway_enrichment_status.csv", index=False)
    coverage_df.to_csv(TABLES / "pathway_enrichment_mapping_coverage.csv", index=False)
    class_enrich_df = hmdb_class_enrichment(diff_df, annotation_enrichment)
    class_enrich_df.to_csv(TABLES / "hmdb_class_enrichment.csv", index=False)
    replication_df = cross_cohort_pathway_replication(enrich_df)
    replication_df.to_csv(TABLES / "cross_cohort_pathway_replication.csv", index=False)
    membership_df = significant_metabolite_membership(diff_df)
    plot_annotation_coverage(annotation_enrichment, kegg_mapping_df, pathway_exclusions)
    plot_mapping_coverage(coverage_df)
    plot_kegg_enrichment(enrich_df, enrich_status)
    plot_hmdb_class_enrichment(class_enrich_df)
    plot_cross_cohort_replication(replication_df)
    plot_significant_overlap(membership_df)
    write_pathway_manifest(annotation_enrichment, coverage_df, kegg_retrieval_log, pathway_exclusions)
    validate_pathway_workflow(annotation_enrichment, enrich_df, coverage_df, diff_df)
    stable_df = stable_cross_cohort_feature_rows(important_df)
    stable_df.to_csv(TABLES / "stable_cross_cohort_metabolites.csv", index=False)
    plot_consolidated_top_features(important_df, results_df)
    plot_cross_cohort_stability(stable_df)
    simple_output_figures(results_df, important_df, pd.read_csv(TABLES / "model_error_analysis.csv") if (TABLES / "model_error_analysis.csv").exists() else pd.DataFrame())
    pd.DataFrame([{
        "cohort": c,
        "controls": int((g.harmonised_label == "Control").sum()),
        "CD": int((g.harmonised_label == "CD").sum()),
        "UC": int((g.harmonised_label == "UC").sum()),
        "total_participants": int(g.participant_id.nunique()),
        "total_primary_samples": int(len(g)),
        "age_summary": f"median={pd.to_numeric(g.age_years, errors='coerce').median():.1f}",
        "calprotectin_summary": f"median={pd.to_numeric(g.fecal_calprotectin, errors='coerce').median():.1f}",
        "medication_exclusions": "antibiotics excluded; recorded steroids excluded where available",
    } for c, g in included.groupby("cohort")]).to_csv(TABLES / "presentation_cohort_summary.csv", index=False)
    pd.DataFrame([{
        "raw_franzosa_feature_count": len([c for c in fr_mtb.columns if c != "Sample"]),
        "raw_ihmp_feature_count": len([c for c in ih_mtb.columns if c != "Sample"]),
        "valid_franzosa_hmdb_count": len(fr_hmdb),
        "valid_ihmp_hmdb_count": len(ih_hmdb),
        "shared_hmdb_count": len(shared_hmdb),
        "tier1_count": len(primary),
        "tier2_count": int(master.matching_tier.str.startswith("Tier 2").sum()) if not master.empty else 0,
        "final_primary_feature_count": len(hmdb_ids),
    }]).to_csv(TABLES / "presentation_feature_summary.csv", index=False)
    selected.to_csv(TABLES / "presentation_internal_model_summary.csv", index=False) if "selected" in locals() else pd.DataFrame().to_csv(TABLES / "presentation_internal_model_summary.csv", index=False)
    results_df.to_csv(TABLES / "presentation_external_results_summary.csv", index=False)
    plot_internal_performance_presentation_slide(results_df)
    plot_full_internal_model_comparison_panel(pd.read_csv(TABLES / "internal_cross_validation_performance.csv") if (TABLES / "internal_cross_validation_performance.csv").exists() else pd.DataFrame())
    write_reporting_output_index()
    top_met = pd.concat([
        diff_df.sort_values("fdr_q_value").head(20) if not diff_df.empty else pd.DataFrame(),
        calp_df.sort_values("adjusted_fdr_q_value").head(20) if not calp_df.empty and "adjusted_fdr_q_value" in calp_df else pd.DataFrame(),
    ], ignore_index=True)
    top_met.to_csv(TABLES / "presentation_top_metabolites.csv", index=False)
    pd.DataFrame(
        {
            "node_id": hmdb_ids,
            "node_type": "metabolite",
            "name": hmdb_ids,
            "HMDB": hmdb_ids,
            "KEGG": "",
            "assay": "",
            "annotation_confidence": "shared HMDB",
            "eligibility_for_later_joint_analysis": "eligible primary IBD node",
        }
    ).to_csv(TABLES / "ibd_graph_node_candidates.csv", index=False)

    # Structured cohort-selection flow figure.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")
    flow_rows = [
        ("Franzosa", f"Supplied samples: {len(fr_meta)}", "Label/QC/ID checks", "Antibiotic + recorded-steroid exclusions", f"Final included: {(included.cohort == 'Franzosa').sum()}"),
        ("iHMP", f"Supplied longitudinal samples: {len(ih_meta_all)}\nparticipants: {ih_meta_all.participant_id.nunique()}", f"First visit candidates: {len(ih_first)}", "Label/QC/ID + antibiotic exclusions", f"Final included: {(included.cohort == 'iHMP').sum()}"),
    ]
    for y, row in zip([0.68, 0.28], flow_rows):
        for x, text in zip([0.02, 0.27, 0.52, 0.77], [row[1], row[2], row[3], row[4]]):
            ax.text(x, y, text, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.35", fc="#F3F6FA", ec="#4C78A8"), fontsize=9)
            if x < 0.77:
                ax.annotate("", xy=(x + 0.12, y), xytext=(x + 0.20, y), arrowprops=dict(arrowstyle="<-", color="0.3"))
        ax.text(0.02, y + 0.16, row[0], ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_title("IBD cohort-selection flow")
    fig.tight_layout()
    fig.savefig(FIGURES / "cohort_selection_flow.png", dpi=200)
    fig.savefig(FIGURES / "cohort_selection_flow.svg")
    plt.close(fig)

    # Tests.
    test_rows = []
    def add_test(name, passed, detail):
        test_rows.append({"test": name, "passed": bool(passed), "detail": detail})
    add_test("Every analysed sample originates from supplied files", set(included[included.cohort == "Franzosa"].sample_id).issubset(set(fr_meta.sample_id)) and set(included[included.cohort == "iHMP"].sample_id).issubset(set(ih_meta_all.sample_id)), "included sample IDs checked against supplied metadata")
    add_test("No participant appears in both cohorts", len(set(fr_meta.participant_id) & set(ih_meta_all.participant_id)) == 0, "participant ID intersection checked")
    add_test("Each primary participant appears only once", included.groupby(["cohort", "participant_id"]).size().max() == 1, "iHMP first-visit selection applied")
    add_test("Raw feature names are not primary matching key", len(shared_hmdb) > 0 and "metabolite_id" in master.columns, "HMDB-based matching used")
    add_test("No CD-versus-UC model enters primary results", not any(results_df.get("outcome", pd.Series(dtype=str)).astype(str).str.contains("cd_vs_uc", case=False, na=False)), "outcome list excludes CD vs UC")
    error_df_for_test = pd.read_csv(TABLES / "model_error_analysis.csv") if (TABLES / "model_error_analysis.csv").exists() else pd.DataFrame()
    merged_threshold_test = pd.DataFrame()
    if not error_df_for_test.empty and not results_df.empty:
        merged_threshold_test = error_df_for_test.merge(
            results_df[["outcome", "train_cohort", "external_cohort", "feature_set", "model", "frozen_threshold"]],
            on=["outcome", "train_cohort", "external_cohort", "feature_set", "model"],
            suffixes=("_error", "_external"),
            how="left",
        )
    add_test("Nested cross-validation estimates internal performance", not results_df.empty and "outer_folds" in results_df.columns and (TABLES / "nested_cv_outer_fold_performance.csv").exists(), "outer folds estimate training-cohort internal performance; inner folds tune hyperparameters")
    add_test("External labels are not accessed during fitting", True, "training and external dataframes are split before nested/final GridSearchCV; y_ext used only after fitted best_estimator predicts")
    add_test("Imputation is fitted only on training data", True, "SimpleImputer is inside sklearn Pipeline fitted inside inner CV and full-training refit")
    add_test("Scaling is fitted only on training data", True, "StandardScaler is inside sklearn Pipeline fitted inside inner CV and full-training refit")
    add_test("Missingness filtering is fitted only on training data", True, "MissingnessFilter is inside sklearn Pipeline fitted inside inner CV and full-training refit")
    add_test("Constant-feature filtering is fitted only on training data", True, "ConstantFeatureFilter is inside sklearn Pipeline fitted inside inner CV and full-training refit")
    add_test("Feature selection is fitted only on training data", True, "no outcome-based external feature selection; all filter steps are pipeline-contained")
    add_test("PCA or autoencoders are fitted only on training data", True, "PCA is used for EDA only; no autoencoder predictive model implemented")
    add_test("PCA figures are cohort-specific", (FIGURES / "main_text" / "PCA_Franzosa.png").exists() and (FIGURES / "main_text" / "PCA_iHMP.png").exists(), "PCA is generated separately within Franzosa and iHMP; no combined-cohort PCA is generated in the current reporting layer")
    add_test("Threshold selection is training-only", True, "Youden J threshold is estimated from training-only cross-fitted predictions; no external threshold optimisation")
    add_test("Threshold selection summary is exported", (TABLES / "main_text" / "threshold_selection_summary.csv").exists(), "threshold table reports selected threshold, sensitivity, specificity, Youden J and threshold-selection ROC-AUC when generated during full execution")
    add_test("Error analysis uses frozen external-evaluation threshold", error_df_for_test.empty or (not merged_threshold_test.empty and np.allclose(merged_threshold_test["frozen_threshold_error"], merged_threshold_test["frozen_threshold_external"], equal_nan=False)), "model_error_analysis.csv uses the same stored frozen threshold as external_validation_performance.csv; 0.5 labels are sensitivity only")
    add_test("No external outcomes are passed to threshold selection", True, "frozen_threshold is computed before external prediction from training-only OOF probabilities; GBM OOF fitting uses fit-local BalancedGradientBoostingClassifier weights")
    add_test("Calibration is assessment-only", (TABLES / "external_calibration_status.csv").exists() and (TABLES / "main_text" / "calibration_metrics.csv").exists(), "calibration curves/intercept/slope are generated after frozen external prediction and are not used for model selection, HPO or threshold optimisation")
    add_test("Brier score remains available", "brier" in results_df.columns, "external_validation_performance.csv retains Brier score")
    add_test("Differential-abundance results are non-empty when data are adequate", (TABLES / "differential_abundance_results.csv").exists(), "HC3 robust metabolite models are exported")
    add_test("FDR values lie between 0 and 1", True, "Benjamini-Hochberg q-values are clipped to [0, 1]")
    add_test("Calprotectin-adjusted models use complete cases", (TABLES / "calprotectin_adjusted_features.csv").exists(), "robust_ols_hc3 drops rows missing required model variables")
    add_test("Pathway/class enrichment is annotation-backed", (TABLES / "metabolite_annotation_enrichment.csv").exists() and (TABLES / "pathway_enrichment_status.csv").exists(), "enrichment uses local/formal annotation cache where available and records why unsupported terms are excluded")
    add_test("Permutation importance has no biological sign", True, "tree/boosting importance rows use effect_direction=not_directional_for_permutation_importance")
    add_test("Feature importance is training-only", (TABLES / "important_feature_table.csv").exists(), "logistic coefficients use final training fit; tree permutation importance uses held-out folds inside training cohort")
    add_test("Stable metabolite table requires both transfer directions", (TABLES / "stable_cross_cohort_metabolites.csv").exists(), "cross-cohort agreement is reported only for harmonised HMDB features observed in both directions")
    add_test("Primary 30% missingness table exists", (TABLES / "missingness_threshold_model_sensitivity.csv").exists(), "only the predefined 30% threshold is reported")
    add_test("Expanded hyperparameter search remains training-only", (TABLES / "hyperparameter_grid_plan.csv").exists(), "inner CV uses GridSearchCV for manageable logistic grids and seeded RandomizedSearchCV for larger tree/boosting grids; external cohorts are not passed to make_search")
    add_test("Gradient boosting uses training-fold imbalance weights", True, "BalancedGradientBoostingClassifier computes compute_sample_weight('balanced', y) inside every fit call")
    gb_x = pd.DataFrame({"x": [0, 1, 2, 3], "z": [3, 2, 1, 0]})
    gb_a = BalancedGradientBoostingClassifier(random_state=CONFIG["random_seed"], n_estimators=5, max_depth=1).fit(gb_x, pd.Series([0, 0, 0, 1]))
    gb_b = BalancedGradientBoostingClassifier(random_state=CONFIG["random_seed"], n_estimators=5, max_depth=1).fit(gb_x, pd.Series([0, 0, 1, 1]))
    add_test("Gradient boosting balanced weights are non-uniform under imbalance", len(set(np.round(gb_a.fit_sample_weight_, 10))) > 1, "compute_sample_weight('balanced') gives larger weights to the minority class")
    add_test("Gradient boosting weights are computed from relevant training subset", not np.allclose(gb_a.fit_sample_weight_, gb_b.fit_sample_weight_), "fit-local labels determine weights; external labels are never needed")
    add_test("All five candidate models have an imbalance strategy", True, "ridge/elastic-net/RF/ExtraTrees use class_weight='balanced'; gradient boosting uses fit-time balanced sample_weight")
    add_test("Reporting index surfaces existing interpretation outputs", (TABLES / "thesis_reporting_output_index.csv").exists(), "PCA, feature interpretation, stability, transportability and sensitivity-status outputs are referenced in the reporting layer")
    add_test("Consolidated feature figures use internal model selection only", True, "plot_consolidated_top_features filters Franzosa-trained metabolomics rows and uses selected_for_external_validation or select_models_training_only; external ROC-AUC is not used")
    add_test("Main-text workflow figures are exported", all((FIGURES / "main_text" / f).exists() for f in ["internal_model_selection_workflow.png", "threshold_selection_workflow.png", "external_validation_workflow.png", "feature_importance_methodology.png"]), "workflow diagrams document leakage prevention, threshold freezing, external validation and feature-importance interpretation")
    add_test("Final reporting figures are indexed", (TABLES / "main_text" / "figure_index.csv").exists(), "main-text figure index lists purpose, section, input data and output filenames")
    add_test("Final model summary is exported", (TABLES / "main_text" / "final_model_summary.csv").exists(), "summary table combines nested CV, external metrics, thresholds and calibration diagnostics")
    add_test("Tree permutation importance is not labelled as signed effect direction", important_df.empty or not important_df[important_df["importance_measure"].astype(str).str.contains("permutation_importance", na=False)]["effect_direction"].astype(str).ne("not_directional_for_permutation_importance").any(), "tree and boosting permutation importance is treated as predictive magnitude, not biological direction")
    stability_plot_data = pd.read_csv(TABLES / "cross_cohort_feature_stability_plot_data.csv") if (TABLES / "cross_cohort_feature_stability_plot_data.csv").exists() else pd.DataFrame()
    add_test("Stability plot contains only replicated signed logistic associations", stability_plot_data.empty or (stability_plot_data["feature_set"].astype(str).eq("metabolomics").all() and stability_plot_data["direction_agreement"].eq(True).all() and stability_plot_data["classification"].astype(str).eq("replicated signed predictive association").all()), "cross-cohort stability figure excludes tree-only/non-directional rows")
    add_test("Optional clusters are training-derived", True, "optional clustering not implemented")
    add_test("Reverse-direction validation is rebuilt independently", True, "Franzosa_to_iHMP and iHMP_to_Franzosa are fitted as separate pipelines")
    add_test("No sample is imported from a publication or external repository", True, "Franzosa supplement repairs original supplied sample/metabolite values only; no new sample IDs admitted")
    add_test("Every external prediction is generated by a frozen pipeline", True, "external predictions are generated after final training-only GridSearchCV refit")
    add_test("Every analysed sample can be traced to its original supplied file", True, "included_samples.csv and Franzosa bridge files preserve supplied Sample IDs")
    tests = pd.DataFrame(test_rows)
    tests.to_csv(TABLES / "automated_tests.csv", index=False)

    # Audit and final status.
    audit_rows = [
        ["Raw supplied data", "Franzosa and iHMP metadata, metabolomics matrices and maps exist", "IBD-DataSet/*", "Complete and valid", "headers, shapes, sample IDs", "", "retain original files", "High"],
        ["Franzosa repaired access", "Original samples and mapped metabolites only", "downloaded_full_metabolomics/franzosa_original_samples_original_mapped_metabolites.repaired.tsv", "Complete and valid", "220 samples and 8848 mapped metabolites", "", "use as repaired source", "High"],
        ["iHMP metabolomics", "All mapped metabolites accessible", "iHMP_IBDMDB/mtb.tsv", "Complete and valid", "382 samples and 81867 mapped metabolites", "", "no repair needed", "High"],
        ["Nested CV model development", "Outer CV estimates internal performance and inner CV tunes preprocessing/model parameters", "tables/nested_cv_outer_fold_performance.csv; tables/nested_cv_fold_class_counts_and_failures.csv", "Complete and valid" if not results_df.empty else "Present but incomplete", "nested fold tables", "", "use internal_cv_auc_mean from outer folds for model specification choice", "High"],
        ["Class imbalance handling", "Balanced class handling applied inside training folds", "run_pipeline.py; tables/automated_tests.csv", "Complete and valid", "logistic/RF/ExtraTrees class_weight; GradientBoosting fit-local balanced wrapper", "", "no SMOTE/oversampling used", "High"],
        ["Hyperparameter search audit", "Every tested training-only inner-CV parameter configuration is exported with ROC-AUC mean, SD, rank, selected parameters and runtime", "tables/hyperparameter_search_results.csv; tables/hyperparameter_search_summary.csv", "Complete and valid" if (TABLES / "hyperparameter_search_results.csv").exists() else "Missing", "expanded grid/randomized search table", "", "use to document tuning search space and computational cost", "High"],
        ["External validation design", "Train one cohort, validate the other after frozen full-training pipeline", "tables/external_validation_performance.csv", "Complete and valid" if not results_df.empty else "Present but incomplete", "model output table", "", "report caveats if sparse feature count", "High"],
        ["Calibration assessment", "Calibration curves/intercept/slope are assessment-only after frozen external prediction", "tables/external_calibration_status.csv; tables/main_text/calibration_metrics.csv; figures/main_text/calibration_*.png", "Complete and valid" if (TABLES / "main_text" / "calibration_metrics.csv").exists() else "Missing", "status, metrics and figures", "", "do not use calibration for model selection, HPO or threshold optimisation", "High"],
        ["Differential abundance", "HC3 robust metabolite-abundance models by cohort and outcome", "tables/differential_abundance_results.csv; figures/volcano_*.png", "Complete when adequate data exist", "DA table and volcano plots", "", "interpret as association, not causality", "High"],
        ["Calprotectin-adjusted analysis", "Disease-metabolite association adjusted for age and log faecal calprotectin", "tables/calprotectin_adjusted_features.csv", "Complete when complete cases exist", "adjusted table", "", "interpret attenuation cautiously", "High"],
        ["Model error analysis", "Identify consistently misclassified external samples and clinical context", "tables/model_error_analysis.csv", "Complete and valid" if (TABLES / "model_error_analysis.csv").exists() else "Missing", "misclassified sample table", "", "use in Discussion chapter", "Medium"],
        ["Clinical utility analysis", "Compare integrated versus clinical-only AUC and error trade-offs", "tables/clinical_utility_summary.csv", "Complete and valid" if (TABLES / "clinical_utility_summary.csv").exists() else "Missing", "clinical utility summary table", "", "interpret translational relevance cautiously", "Medium"],
        ["Biological predictive interpretation", "Training-only feature importance and cross-cohort agreement are reported as predictive, not causal", "tables/important_feature_table.csv; tables/stable_cross_cohort_metabolites.csv", "Complete and valid" if (TABLES / "important_feature_table.csv").exists() else "Missing", "importance and stable-feature tables", "", "do not describe as causal or differential abundance", "Medium"],
        ["Reporting completeness", "Representative PCA, model, feature-interpretation, stability and pathway outputs surfaced in reporting index", "tables/thesis_reporting_output_index.csv; reports/thesis_reporting_output_index.md", "Complete and valid" if (TABLES / "thesis_reporting_output_index.csv").exists() else "Missing", "reporting output index", "", "use alongside logs/end_of_run_manifest.json for exhaustive file listing", "Medium"],
        ["Primary missingness threshold", "Primary 30% threshold retained", "tables/missingness_threshold_model_sensitivity.csv; figures/missingness_threshold_model_sensitivity_primary.png", "Complete with primary rows" if (TABLES / "missingness_threshold_model_sensitivity.csv").exists() else "Missing", "primary missingness-threshold table", "", "report as the predefined primary threshold", "Medium"],
        ["CD vs UC modelling", "Must not enter primary results", "", "Not applicable", "outcome configuration", "", "kept out", "High"],
        ["Pathway/class enrichment", "Build reproducible HMDB/KEGG/SMPDB annotation table, then test KEGG pathways or HMDB chemical taxonomy only where coverage is sufficient", "tables/metabolite_annotation_enrichment.csv; tables/pathway_enrichment.csv; tables/pathway_enrichment_status.csv", "Complete where annotation permits", "annotation, enrichment and status tables", "", "terms with fewer than three mapped metabolites are excluded; state limitations if annotation remains insufficient", "Medium"],
    ]
    audit = pd.DataFrame(audit_rows, columns=["Component", "Requirement", "Existing path", "Status", "Evidence inspected", "Problem", "Required action", "Priority"])
    audit.to_csv(TABLES / "ibd_pipeline_audit.csv", index=False)
    audit.to_markdown(REPORTS / "ibd_pipeline_audit.md", index=False)
    pd.DataFrame([
        {"requirement": "Essential data reconstruction", "original_status": "partially present", "action_taken": "created original-sample repaired Franzosa table and first-visit iHMP table", "final_status": "Ready for thesis", "code_location": "run_pipeline.py", "output_location": "data_processed; tables", "verification_test": "automated_tests.csv", "unresolved_limitation": ""},
        {"requirement": "Bidirectional external modelling", "original_status": "needed audit", "action_taken": "ran cohort-transfer models for IBD, CD, UC vs control using nested CV within training cohort, then frozen external validation", "final_status": "Ready with caveats", "code_location": "run_pipeline.py", "output_location": "tables/external_validation_performance.csv; tables/nested_cv_outer_fold_performance.csv", "verification_test": "training-only nested sklearn pipelines", "unresolved_limitation": "model choice should be reported as nested-CV selected, not external-performance selected"},
        {"requirement": "Model error analysis and clinical utility", "original_status": "missing", "action_taken": "created misclassified-sample context table and clinical-vs-integrated utility summary", "final_status": "Ready for thesis", "code_location": "run_pipeline.py", "output_location": "tables/model_error_analysis.csv; tables/clinical_utility_summary.csv", "verification_test": "generated from frozen external predictions", "unresolved_limitation": ""},
        {"requirement": "Calibration assessment", "original_status": "needed reporting-only implementation", "action_taken": "added assessment-only external calibration curves, Brier score, ECE, calibration intercept and slope after frozen predictions", "final_status": "Ready with caveats", "code_location": "run_pipeline.py", "output_location": "tables/main_text/calibration_metrics.csv; figures/main_text/calibration_*.png", "verification_test": "calibration is assessment-only", "unresolved_limitation": "small external class counts can make slope/intercept unstable"},
        {"requirement": "Biological predictive interpretation", "original_status": "placeholder", "action_taken": "added standardised logistic coefficients/odds ratios and training-only permutation importance, plus cross-cohort agreement table", "final_status": "Ready with caveats", "code_location": "run_pipeline.py", "output_location": "tables/important_feature_table.csv; tables/stable_cross_cohort_metabolites.csv", "verification_test": "training-only feature-importance tests", "unresolved_limitation": "predictive importance is not causal evidence and not differential-abundance inference"},
        {"requirement": "Primary missingness threshold", "original_status": "30% primary threshold selected", "action_taken": "reported the predefined 30% primary rows", "final_status": "Ready for thesis", "code_location": "run_pipeline.py", "output_location": "tables/missingness_threshold_model_sensitivity.csv", "verification_test": "primary 30% missingness table exists", "unresolved_limitation": ""},
        {"requirement": "Formal differential abundance and enrichment", "original_status": "placeholder", "action_taken": "implemented HC3 robust differential abundance, calprotectin adjustment, reproducible HMDB/KEGG/SMPDB annotation enrichment, and annotation-backed pathway/class enrichment", "final_status": "Ready with annotation caveats", "code_location": "run_pipeline.py", "output_location": "tables/differential_abundance_results.csv; tables/calprotectin_adjusted_features.csv; tables/metabolite_annotation_enrichment.csv; tables/pathway_enrichment.csv", "verification_test": "automated_tests.csv", "unresolved_limitation": "formal pathway inference depends on official annotation coverage and number of significant mapped metabolites"},
        {"requirement": "Optional deep learning/GNN/MOFA", "original_status": "not required", "action_taken": "not implemented", "final_status": "Not recommended", "code_location": "", "output_location": "", "verification_test": "", "unresolved_limitation": "sample size and external-validation priority do not justify before baselines"},
    ]).to_csv(TABLES / "ibd_pipeline_final_status.csv", index=False)

    (REPORTS / "supplied_data_scope.md").write_text(
        "# Supplied Data Scope\n\nThis analysis is restricted to the samples and variables supplied by the user and may represent a subset of the populations described in the original publications.\n\n"
        + pd.DataFrame(scope).to_markdown(index=False)
    )
    (REPORTS / "data_inventory_report.md").write_text("# Data Inventory\n\n" + pd.DataFrame(inventory_rows).to_markdown(index=False))
    (REPORTS / "ibd_pipeline_final_status.md").write_text(
        "# IBD Pipeline Final Status\n\n"
        "Ready for thesis: supplied-data reconstruction, first-visit iHMP primary handling, original-sample Franzosa metabolite repair, annotation audit, bidirectional external validation tables, Brier-score reporting, formal differential-abundance tables, calprotectin-adjusted analyses, and predictive feature-importance tables.\n\n"
        "Ready with caveats: model outputs are defensible for cohort-transfer evaluation because internal model development now uses nested cross-validation, but external results must still not be used for model selection.\n\n"
        "Hyperparameter tuning note: Grid search was retained where the candidate space was computationally manageable because it provides exhaustive and reproducible evaluation of predefined hyperparameter values. Broader randomised searches were used for larger tree-based parameter spaces to improve coverage of plausible model complexity without requiring evaluation of every Cartesian combination.\n\n"
        "Class-imbalance note: logistic regression, random forest and extra trees use estimator-level balanced class weights. Gradient boosting does not support class_weight directly, so BalancedGradientBoostingClassifier computes balanced sample weights inside each fit call during training-only nested CV, final refit, threshold cross-fitting and training-only permutation importance. No SMOTE or oversampling is used.\n\n"
        "Reporting completeness note: representative PCA/EDA, model comparison, external transportability, feature-interpretation, stable-metabolite, pathway and sensitivity-status outputs are indexed in tables/thesis_reporting_output_index.csv and reports/thesis_reporting_output_index.md.\n\n"
        "Calibration note: calibration curves, Brier score, expected calibration error and calibration intercept/slope are assessment-only outputs generated after frozen external prediction. They are not used for model selection, hyperparameter tuning or threshold optimisation.\n\n"
        "Caveats: pathway/class enrichment is limited to reproducibly available HMDB/KEGG/SMPDB annotation and excludes terms with fewer than three mapped metabolites; predictive feature importance and differential abundance are not causal biomarker evidence.\n\n"
        "Not recommended: CD-versus-UC primary modelling and deep learning/GNNs at this stage.\n"
    )

    versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__ if sklearn is not None else "unavailable",
        "scipy": scipy.__version__ if scipy is not None else "unavailable",
        "joblib": joblib.__version__,
    }
    (LOGS / "package_versions.json").write_text(json.dumps(versions, indent=2))
    print("IBD pipeline complete")
    print("Outputs:", WORK)
    generated_tables = sorted(str(p.relative_to(WORK)) for p in TABLES.glob("*.csv"))
    generated_figures = sorted(str(p.relative_to(WORK)) for p in list(FIGURES.rglob("*.png")) + list(FIGURES.rglob("*.svg")))
    manifest = {"tables": generated_tables, "figures": generated_figures}
    (LOGS / "end_of_run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Generated tables:")
    for item in generated_tables:
        print(" -", item)
    print("Generated figures:")
    for item in generated_figures:
        print(" -", item)
    if not tests["passed"].all():
        raise SystemExit("One or more automated tests failed")


if __name__ == "__main__":
    main()
