from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from joblib import dump, load
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_stage2_modelling as crc_stage2
from crc_pipeline_transformers import BalancedGradientBoostingClassifier
from crc_pipeline_transformers import ConstantFeatureFilter, MissingnessFilter
from run_stage2_modelling import (
    CRCStage2Config,
    build_preprocessor,
    build_model_input_tables,
    preprocessing_feature_count_diagnostic,
    select_internal_model,
    validate_model_context,
    configure_output_roots,
    ensure_dirs,
)


def _rows(records):
    rows = []
    for rec in records:
        base = {
            "train_cohort": "Sinha",
            "feature_set": "metabolomics",
            "outer_fold": 1,
            "threshold_0_5_roc_auc": rec["auc"],
            "threshold_0_5_pr_auc": rec["pr"],
        }
        for fold in range(5):
            row = base.copy()
            row["outer_fold"] = fold + 1
            row["model"] = rec["model"]
            row["threshold_0_5_roc_auc"] = rec["auc"] + rec.get("fold_offsets", [0, 0, 0, 0, 0])[fold]
            rows.append(row)
    return pd.DataFrame(rows)


def test_effective_tie_pr_auc_wins_over_slightly_higher_auc():
    config = CRCStage2Config(model_selection_effective_tie_tolerance_auc=0.01, model_selection_effective_tie_tolerance_pr_auc=0.001)
    internal = _rows([
        {"model": "ridge_logistic", "auc": 0.804, "pr": 0.70},
        {"model": "elastic_net_logistic", "auc": 0.800, "pr": 0.82},
    ])
    selected = select_internal_model(internal, config)
    winner = selected[selected["selected"]].iloc[0]
    assert winner["model"] == "elastic_net_logistic"
    assert winner["final_tie_break_stage"] == "pr_auc_tie_break"


def test_effective_tie_stability_wins_when_pr_auc_tied():
    config = CRCStage2Config(
        model_selection_effective_tie_tolerance_auc=0.01,
        model_selection_effective_tie_tolerance_pr_auc=0.01,
        model_selection_effective_tie_tolerance_roc_auc_sd=0.001,
    )
    internal = _rows([
        {"model": "ridge_logistic", "auc": 0.800, "pr": 0.80, "fold_offsets": [-0.05, 0.04, -0.02, 0.03, 0.00]},
        {"model": "random_forest", "auc": 0.799, "pr": 0.801, "fold_offsets": [-0.005, 0.004, -0.002, 0.003, 0.000]},
    ])
    selected = select_internal_model(internal, config)
    winner = selected[selected["selected"]].iloc[0]
    assert winner["model"] == "random_forest"
    assert winner["final_tie_break_stage"] == "stability_tie_break"


def test_effective_tie_simplicity_wins_when_auc_pr_and_stability_tied():
    config = CRCStage2Config(model_selection_effective_tie_tolerance_auc=0.01, model_selection_effective_tie_tolerance_pr_auc=0.01)
    internal = _rows([
        {"model": "ridge_logistic", "auc": 0.800, "pr": 0.80},
        {"model": "random_forest", "auc": 0.800, "pr": 0.80},
    ])
    selected = select_internal_model(internal, config)
    winner = selected[selected["selected"]].iloc[0]
    assert winner["model"] == "ridge_logistic"
    assert winner["final_tie_break_stage"] == "simplicity_tie_break"


def test_balanced_gradient_boosting_fit_local_weights_change_with_subset():
    x = pd.DataFrame({"a": np.arange(8), "b": np.arange(8)[::-1]})
    y_balanced = np.array([0, 0, 1, 1])
    y_imbalanced = np.array([0, 0, 0, 1])
    clf_a = BalancedGradientBoostingClassifier(random_state=42, n_estimators=5, max_depth=1).fit(x.iloc[:4], y_balanced)
    clf_b = BalancedGradientBoostingClassifier(random_state=42, n_estimators=5, max_depth=1).fit(x.iloc[:4], y_imbalanced)
    assert not np.allclose(clf_a.fit_sample_weight_, clf_b.fit_sample_weight_)


def test_balanced_gradient_boosting_clone_search_and_joblib_roundtrip(tmp_path):
    x = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1, 0]})
    y = np.array([0, 0, 0, 1, 1, 1])
    clf = BalancedGradientBoostingClassifier(random_state=42, n_estimators=5, max_depth=1)
    cloned = clone(clf)
    assert cloned.get_params()["n_estimators"] == 5
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    search = GridSearchCV(clf, {"n_estimators": [5], "max_depth": [1]}, scoring="roc_auc", cv=cv)
    search.fit(x, y)
    proba = search.best_estimator_.predict_proba(x)
    assert proba.shape == (6, 2)
    assert list(search.best_estimator_.classes_) == [0, 1]
    path = tmp_path / "balanced_gb.joblib"
    dump(search.best_estimator_, path)
    loaded = load(path)
    assert loaded.predict_proba(x).shape == (6, 2)


def test_missingness_filter_fails_when_all_features_removed():
    x = pd.DataFrame({"a": [np.nan, np.nan, 1.0], "b": [np.nan, np.nan, np.nan]})
    filt = MissingnessFilter(max_missing=0.30)
    try:
        filt.fit(x)
    except ValueError as exc:
        assert "removed all features" in str(exc)
    else:
        raise AssertionError("MissingnessFilter should fail rather than retaining an arbitrary feature")


def test_constant_filter_fails_when_all_features_removed():
    x = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [np.nan, 2.0, 2.0]})
    filt = ConstantFeatureFilter()
    try:
        filt.fit(x)
    except ValueError as exc:
        assert "removed all features" in str(exc)
    else:
        raise AssertionError("ConstantFeatureFilter should fail rather than retaining an arbitrary feature")


def test_empty_clinical_context_is_skipped_explicitly():
    cfg = CRCStage2Config(fast_test=True)
    x = pd.DataFrame({"met__HMDB:1": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0, 0, 1, 1])
    row = validate_model_context("Synthetic", "clinical", "ridge_logistic", x, y, cfg)
    assert row["status"] == "skipped"
    assert "clinical-only context unavailable" in row["reason"]
    assert row["critical"] is False


def test_zero_metabolite_space_is_critical_for_metabolomics():
    cfg = CRCStage2Config(fast_test=True)
    x = pd.DataFrame({"age_years": [40, 50, 60, 70]})
    y = pd.Series([0, 0, 1, 1])
    row = validate_model_context("Synthetic", "metabolomics", "ridge_logistic", x, y, cfg)
    assert row["status"] == "skipped"
    assert row["critical"] is True
    assert "zero selected source columns" in row["reason"] or "metabolite feature space is empty" in row["reason"]


def test_no_active_column_transformer_branches_raise():
    try:
        build_preprocessor("clinical", [], [], [], 0.30)
    except ValueError as exc:
        assert "No active preprocessing branches" in str(exc)
    else:
        raise AssertionError("build_preprocessor should fail when no branch is active")


def test_metabolomics_and_integrated_contexts_with_valid_shared_features():
    cfg = CRCStage2Config(fast_test=True)
    x = pd.DataFrame({"met__HMDB:1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "met__HMDB:2": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]})
    y = pd.Series([0, 0, 0, 1, 1, 1])
    met = validate_model_context("Synthetic", "metabolomics", "ridge_logistic", x, y, cfg)
    integ = validate_model_context("Synthetic", "integrated", "ridge_logistic", x, y, cfg)
    assert met["status"] == "available"
    assert integ["status"] == "available"
    assert integ["n_selected_metabolite_columns"] == 2


def test_feature_count_tracking_records_first_zero_stage():
    x = pd.DataFrame({"met__HMDB:1": [np.nan, np.nan, 1.0], "met__HMDB:2": [np.nan, np.nan, np.nan]})
    diag = preprocessing_feature_count_diagnostic("Synthetic", "metabolomics", "ridge_logistic", 1, x, 0.30)
    missing_row = diag[diag["fitting_stage"].eq("after_MissingnessFilter")].iloc[0]
    assert missing_row["n_retained_features"] == 0


def test_crc_clinical_recovery_restores_shared_age_sex_bmi():
    cfg = CRCStage2Config(fast_test=True)
    configure_output_roots(cfg)
    ensure_dirs()
    meta, features, _ = build_model_input_tables(cfg)
    for cohort, expected_n in {"Sinha": 131, "Yachida": 277}.items():
        assert len(meta[cohort]) == expected_n
        assert meta[cohort][["age_years", "sex", "bmi"]].notna().all().all()
        assert {"age_years", "sex", "bmi"}.issubset(features[cohort].columns)
        assert not {"stage_major", "tumour_site"}.intersection(features[cohort].columns)
    contract = pd.read_csv(crc_stage2.TABLES / "crc_stage2_shared_clinical_feature_contract.csv")
    included = set(contract.loc[contract["included_in_primary_shared_clinical_model"], "clinical_feature"])
    assert included == {"age_years", "sex", "bmi"}


def test_integrated_input_differs_from_metabolomics_after_clinical_recovery():
    cfg = CRCStage2Config(fast_test=True)
    configure_output_roots(cfg)
    ensure_dirs()
    meta, features, _ = build_model_input_tables(cfg)
    y = meta["Sinha"]["crc_binary"].astype(int)
    clinical = validate_model_context("Sinha", "clinical", "ridge_logistic", features["Sinha"], y, cfg)
    met = validate_model_context("Sinha", "metabolomics", "ridge_logistic", features["Sinha"], y, cfg)
    integrated = validate_model_context("Sinha", "integrated", "ridge_logistic", features["Sinha"], y, cfg)
    assert clinical["status"] == "available"
    assert met["n_selected_source_columns"] == 110
    assert integrated["n_selected_source_columns"] == 113


def test_clinical_linkage_outputs_have_no_duplicates_or_conflicts():
    cfg = CRCStage2Config(fast_test=True)
    configure_output_roots(cfg)
    ensure_dirs()
    build_model_input_tables(cfg)
    linkage = pd.read_csv(crc_stage2.TABLES / "crc_stage2_clinical_linkage_audit.csv")
    assert linkage.groupby(["cohort", "sample_id"]).size().max() == 1
    assert linkage["linkage_status"].eq("linked").all()
    assert linkage["duplicate_status"].eq("unique_source_id").all()
    assert linkage["age_years"].between(0, 120).all()
    assert linkage["bmi"].between(10, 80).all()
    assert set(linkage["sex_standardized"].dropna().unique()) == {"Male", "Female"}
