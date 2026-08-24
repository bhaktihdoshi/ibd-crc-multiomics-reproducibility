from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

SCRIPT_DIR = Path("/Users/bhaktidoshi/Downloads/Thesis Data/thesis_preprocessing_workflow/stage4_multiomics_integration/scripts")
sys.path.insert(0, str(SCRIPT_DIR))

from run_stage4c import (  # noqa: E402
    CohortBlock,
    SCCA_PENALTIES,
    ThresholdedDenseCCA,
    build_mofa_input_for_disease,
    input_integrity,
    loading_similarity,
    match_components,
    orient_factor_scores,
    orient_loadings,
    orient_loadings_matrix,
    prepared_views,
    scca_method_metadata,
    serialize_view_matrix,
    scca_component_associations,
    validate_scca_loading_outputs,
    synthetic_scca_postprocessing_smoke,
)


def synthetic_block() -> CohortBlock:
    return CohortBlock(
        disease="IBD",
        cohort="Franzosa",
        microbiome=pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3", "s4"],
                "mic__A": [10.0, 20.0, 30.0, 40.0],
                "mic__B": [5.0, 3.0, 2.0, 1.0],
                "mic__C": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        metabolome=pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3", "s4"],
                "met__M1": [1.0, 2.0, np.nan, 4.0],
                "met__M2": [4.0, 3.0, 2.0, 1.0],
            }
        ),
        metadata=pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3", "s4"],
                "disease_group": ["Control", "CD", "UC", "CD"],
            }
        ),
    )


def test_stage4c_prepared_views_exclude_phenotype_columns() -> None:
    views = prepared_views(synthetic_block())
    assert set(views) == {"microbiome", "metabolome"}
    for frame in views.values():
        assert "disease_group" not in frame.columns
        assert "sample_id" not in frame.columns
        assert np.isfinite(frame.to_numpy()).all()


def test_stage4c_thresholded_dense_cca_is_exploratory_not_primary_scca() -> None:
    estimator = ThresholdedDenseCCA(n_components=1, sparsity_quantile=0.50)
    cloned = clone(estimator)
    assert cloned.__class__.__name__ == "ThresholdedDenseCCA"
    assert "EXPLORATORY_ONLY_NOT_TRUE_SCCA" in (cloned.__doc__ or "")


def test_stage4c_true_scca_method_metadata_uses_pma_penalised_objective() -> None:
    meta = scca_method_metadata()
    text = " ".join(meta.astype(str).iloc[0].tolist())
    assert "PMA" in text
    assert "PMA::CCA" in text
    assert "penaltyx" in text
    assert "penaltyz" in text
    assert "optimisation objective" in text
    assert "not thesis sCCA" in text


def test_stage4c_orient_loadings_largest_absolute_loading_positive() -> None:
    loadings = pd.Series({"a": -0.8, "b": 0.2, "c": 0.1})
    oriented = orient_loadings(loadings)
    assert oriented.loc["a"] > 0


def test_stage4c_loading_similarity_is_sign_robust() -> None:
    a = pd.Series({"f1": 1.0, "f2": 0.5, "f3": -0.2})
    b = pd.Series({"f1": -1.0, "f2": -0.5, "f3": 0.2})
    sim = loading_similarity(a, b)
    assert sim["n_common"] == 3
    assert sim["cosine_abs"] > 0.99


def test_stage4c_input_integrity_reports_failures_on_wrong_dimensions() -> None:
    audit = input_integrity({("IBD", "Franzosa"): synthetic_block()})
    assert "status" in audit.columns
    assert audit["status"].eq("fail").any()


def test_stage4c_pandas3_mofa_long_serializer_preserves_full_grid() -> None:
    matrix = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]}, index=["s1", "s2"])
    long = serialize_view_matrix(matrix, ["s1", "s2"], "GroupA", "metabolome")
    assert list(long.columns) == ["sample", "feature", "view", "group", "value"]
    assert len(long) == 4
    assert set(long["sample"]) == {"s1", "s2"}
    assert set(long["feature"]) == {"f1", "f2"}
    assert set(long["group"]) == {"GroupA"}
    assert set(long["view"]) == {"metabolome"}


def test_stage4c_group_aware_mofa_construction_keeps_views_and_groups_out_of_features() -> None:
    block_a = synthetic_block()
    block_b = CohortBlock(
        disease="IBD",
        cohort="iHMP",
        microbiome=block_a.microbiome.assign(sample_id=["t1", "t2", "t3", "t4"]),
        metabolome=block_a.metabolome.assign(sample_id=["t1", "t2", "t3", "t4"]),
        metadata=block_a.metadata.assign(sample_id=["t1", "t2", "t3", "t4"]),
    )
    blocks = {("IBD", "Franzosa"): block_a, ("IBD", "iHMP"): block_b}
    valid = {
        ("IBD", "Franzosa"): {"microbiome": ["mic__A", "mic__B", "mic__C"], "metabolome": ["met__M1", "met__M2"]},
        ("IBD", "iHMP"): {"microbiome": ["mic__A", "mic__B", "mic__C"], "metabolome": ["met__M1", "met__M2"]},
    }
    data_df, meta_df, audit_rows = build_mofa_input_for_disease("IBD", ["Franzosa", "iHMP"], blocks, valid)
    assert set(data_df["group"]) == {"Franzosa", "iHMP"}
    assert set(data_df["view"]) == {"microbiome", "metabolome"}
    assert "disease_group" not in data_df.columns
    assert not set(data_df["feature"]).intersection({"Franzosa", "iHMP", "Control", "CD", "UC"})
    assert len(audit_rows) == 4


def test_stage4c_orients_z_factors_by_samples() -> None:
    z = np.arange(24).reshape(3, 8)
    out, orientation = orient_factor_scores(z, n_samples=8, n_factors=3, label="synthetic Z")
    assert out.shape == (8, 3)
    assert orientation == "factors_x_samples_transposed"


def test_stage4c_orients_z_samples_by_factors() -> None:
    z = np.arange(24).reshape(8, 3)
    out, orientation = orient_factor_scores(z, n_samples=8, n_factors=3, label="synthetic Z")
    assert out.shape == (8, 3)
    assert orientation == "samples_x_factors"


def test_stage4c_orients_w_factors_by_features() -> None:
    w = np.arange(15).reshape(3, 5)
    out, orientation = orient_loadings_matrix(w, n_features=5, n_factors=3, label="synthetic W")
    assert out.shape == (5, 3)
    assert orientation == "factors_x_features_transposed"


def test_stage4c_invalid_orientation_fails_loudly() -> None:
    try:
        orient_factor_scores(np.zeros((4, 4)), n_samples=8, n_factors=3, label="bad Z")
    except ValueError as exc:
        assert "incompatible Z shape" in str(exc)
    else:
        raise AssertionError("Invalid Z orientation did not fail")


def test_stage4c_scca_penalty_grid_valid_for_pma_objective() -> None:
    assert SCCA_PENALTIES
    assert all(0 < penalty <= 1 for penalty in SCCA_PENALTIES)


def test_stage4c_component_matching_can_swap_component_order() -> None:
    class Dummy:
        pass
    model_a = Dummy()
    model_b = Dummy()
    model_a.n_components = 2
    model_b.n_components = 2
    model_a.x_loadings_ = np.array([[1.0, 0.0], [0.8, 0.1], [0.0, 1.0]])
    model_a.y_loadings_ = np.array([[1.0, 0.0], [0.7, 0.2], [0.0, 1.0]])
    model_b.x_loadings_ = model_a.x_loadings_[:, ::-1]
    model_b.y_loadings_ = model_a.y_loadings_[:, ::-1]
    features = {"microbiome": ["mic__g1", "mic__g2", "mic__g3"], "metabolome": ["met__m1", "met__m2", "met__m3"]}
    matched = match_components(model_a, model_b, features, features)
    assert matched
    assert matched[0]["component_b"] == 2


def test_stage4c_scca_component_associations_are_posthoc_annotated() -> None:
    score = pd.Series([0.1, 0.2, 1.0, 1.1, -0.1, 0.0], index=[f"s{i}" for i in range(6)])
    meta = pd.DataFrame({"sample_id": [f"s{i}" for i in range(6)], "disease_group": ["Control", "Control", "CD", "CD", "UC", "UC"]})
    rows = scca_component_associations(score, meta, "IBD", "Synthetic", component=1)
    assert rows
    assert all(row["phenotype_used_for_fit_or_tuning"] is False for row in rows)


def test_stage4c_scca_loading_validation_requires_one_row_per_feature_per_component() -> None:
    summary = pd.DataFrame({"disease": ["IBD", "IBD"], "cohort": ["Franzosa", "Franzosa"], "component": [1, 2]})
    micro = pd.DataFrame({
        "disease": ["IBD"] * 100,
        "cohort": ["Franzosa"] * 100,
        "component": [1] * 50 + [2] * 50,
        "view": ["microbiome"] * 100,
        "feature_id": [f"mic__g{i}" for i in range(50)] * 2,
        "feature_name": [f"g{i}" for i in range(50)] * 2,
        "loading": np.linspace(-1, 1, 100),
        "selected": [True, False] * 50,
    })
    met = pd.DataFrame({
        "disease": ["IBD"] * 332,
        "cohort": ["Franzosa"] * 332,
        "component": [1] * 166 + [2] * 166,
        "view": ["metabolome"] * 332,
        "feature_id": [f"met__m{i}" for i in range(166)] * 2,
        "feature_name": [f"m{i}" for i in range(166)] * 2,
        "loading": np.linspace(-1, 1, 332),
        "selected": [True, False] * 166,
    })
    ok, audit = validate_scca_loading_outputs(summary, micro, met)
    assert ok
    assert audit["status"].eq("pass").all()
    bad_ok, bad_audit = validate_scca_loading_outputs(summary, micro.head(2), met.head(2))
    assert not bad_ok
    assert bad_audit["status"].eq("fail").any()


def test_stage4c_scca_synthetic_postprocessing_smoke_reaches_success() -> None:
    result = synthetic_scca_postprocessing_smoke()
    assert result == {
        "schemas": True,
        "matching": True,
        "figures": True,
        "report": True,
        "completion_logic": True,
    }
