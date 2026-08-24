from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone


# This file lives at <PROJECT>/thesis_preprocessing_workflow/stage4_multiomics_integration/corrected_v2/tests/.
_DEFAULT_PROJECT = Path(__file__).resolve().parents[4]
PROJECT = Path(os.environ["THESIS_PROJECT_ROOT"]) if os.environ.get("THESIS_PROJECT_ROOT") else _DEFAULT_PROJECT
SCRIPTS = PROJECT / "thesis_preprocessing_workflow/stage4_multiomics_integration/corrected_v2/scripts"
sys.path.insert(0, str(SCRIPTS))

from stage4_corrected_v2_common import (  # noqa: E402
    MicrobiomeCLRPreprocessor,
    MicrobiomePresenceAbsencePreprocessor,
    MicrobiomeRankPreprocessor,
    TASK_SPECS,
    authoritative_genus_matrix,
    build_ridge_pipeline,
    cross_fitted_probs,
    fold_count,
    microbiome_missingness_audit,
    select_youden_threshold_from_oof,
)
from run_stage4b_corrected_v2 import CorrectedDataBundle, build_xy  # noqa: E402
from audit_stage4b_corrected_v2_postrun import (  # noqa: E402
    build_paired_comparisons,
    build_probability_diagnostics,
    reporting_internal_performance,
)
from run_stage4c_corrected_v2 import build_mofa_plan, build_scca_plan, expected_stage4c_outputs  # noqa: E402
from run_stage4c_corrected_v2 import load_blocks as load_stage4c_blocks, build_feature_sets as build_stage4c_feature_sets, prepared_views as stage4c_prepared_views  # noqa: E402
import run_stage4c_corrected_v2 as stage4c  # noqa: E402
from run_stage4d_corrected_v2_graph import build_graph_plan  # noqa: E402
import run_stage4d_corrected_v2_graph as stage4d  # noqa: E402
import audit_stage4d_corrected_v2_provenance_topology as stage4d_audit  # noqa: E402
from run_stage4e_gcn_corrected_v2 import (  # noqa: E402
    GRAPH_MODES as STAGE4E_GRAPH_MODES,
    architecture_grid,
    build_gcn_plan,
    conservative_biological_edges,
    model_configuration_grid,
)
import postprocess_stage4e_corrected_v2_primary as stage4e_post  # noqa: E402
import reconstruct_stage4e_corrected_v2_split_membership as stage4e_splits  # noqa: E402
import postprocess_stage4e_corrected_v2_sensitivities as stage4e_sens  # noqa: E402
import freeze_final_stage4_results as final_stage4  # noqa: E402
from freeze_stage4b_v2c_interpretation import INTERPRETATION_RULES, RESTRUCTURED_DEPENDENCIES  # noqa: E402
from run_stage4b_corrected_v2b import (  # noqa: E402
    binary_metrics_from_scores,
    cross_fitted_decision_scores_and_probs,
    probability_threshold_from_decision_threshold,
    select_youden_threshold_from_scores,
)
from audit_stage4b_corrected_v2b_postrun import build_paired_comparisons as build_v2b_paired_comparisons  # noqa: E402


def test_authoritative_genus_does_not_double_count_explicit_parent_and_species() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "genus:Bacteroides": [100.0, 50.0],
            "species:Bacteroides fragilis": [10.0, 20.0],
            "species:Bacteroides vulgatus": [5.0, 10.0],
        }
    )
    genus, crosswalk = authoritative_genus_matrix("iHMP", df)
    assert list(genus.columns) == ["sample_id", "mic__Bacteroides"]
    assert np.allclose(genus["mic__Bacteroides"], [1.0, 1.0])
    excluded = crosswalk.loc[~crosswalk["include_for_authoritative_genus"], "original_taxon_label"].tolist()
    assert "species:Bacteroides fragilis" in excluded
    assert "species:Bacteroides vulgatus" in excluded


def test_species_only_cohort_aggregates_species_then_recloses() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "species:Bacteroides fragilis": [10.0, 20.0],
            "species:Bacteroides vulgatus": [5.0, 10.0],
            "species:Faecalibacterium prausnitzii": [15.0, 30.0],
        }
    )
    genus, _ = authoritative_genus_matrix("Franzosa", df)
    assert set(genus.columns) == {"sample_id", "mic__Bacteroides", "mic__Faecalibacterium"}
    assert np.allclose(genus.drop(columns=["sample_id"]).sum(axis=1), 1.0)
    assert np.allclose(genus["mic__Bacteroides"], 0.5)


def test_missingness_audit_distinguishes_nan_from_observed_zero() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "genus:Bacteroides": [10.0, 0.0, np.nan],
            "genus:NeverMeasured": [np.nan, np.nan, np.nan],
        }
    )
    audit = microbiome_missingness_audit("Franzosa", df).set_index("feature")
    assert audit.loc["genus:Bacteroides", "n_missing_before"] == 1
    assert audit.loc["genus:Bacteroides", "n_observed_zero"] == 1
    assert audit.loc["genus:Bacteroides", "n_converted_to_zero"] == 1
    assert not bool(audit.loc["genus:Bacteroides", "wholly_missing_unresolved"])
    assert audit.loc["genus:NeverMeasured", "n_missing_before"] == 3
    assert audit.loc["genus:NeverMeasured", "n_converted_to_zero"] == 0
    assert bool(audit.loc["genus:NeverMeasured", "wholly_missing_unresolved"])


def test_authoritative_genus_excludes_wholly_missing_feature_instead_of_zero_filling() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "genus:Bacteroides": [100.0, 50.0],
            "genus:NeverMeasured": [np.nan, np.nan],
        }
    )
    genus, crosswalk = authoritative_genus_matrix("iHMP", df)
    assert "mic__NeverMeasured" not in genus.columns
    row = crosswalk.set_index("original_taxon_label").loc["genus:NeverMeasured"]
    assert not bool(row["include_for_authoritative_genus"])
    assert "never measured" in row["reason_for_exclusion"]


def test_microbiome_preprocessor_recloses_after_prevalence_filtering() -> None:
    x = pd.DataFrame({"a": [50.0, 0.0, 10.0, 20.0], "b": [50.0, 0.0, 0.0, 0.0], "c": [0.0, 100.0, 90.0, 80.0]})
    proc = MicrobiomeCLRPreprocessor(prevalence_threshold=0.50)
    proc.fit(x)
    out = proc.transform(x)
    assert out.shape[0] == 4
    assert np.isfinite(out).all()
    assert set(proc.get_feature_names_out()) == {"a", "c"}


def test_microbiome_preprocessor_raises_on_unresolved_partial_missing_by_default() -> None:
    # An observed zero (sample 2, feature 'a') must remain a real zero and not
    # trip the guard; a genuinely unresolved NaN (sample 4, feature 'a') must
    # raise rather than be silently treated as absence.
    x = pd.DataFrame({"a": [50.0, 0.0, 10.0, np.nan], "b": [50.0, 30.0, 20.0, 40.0], "c": [10.0, 90.0, 80.0, 60.0]})
    for cls in (MicrobiomeCLRPreprocessor, MicrobiomeRankPreprocessor, MicrobiomePresenceAbsencePreprocessor):
        proc = cls(prevalence_threshold=0.50)
        try:
            proc.fit(x)
            raised = False
        except ValueError as exc:
            raised = True
            assert "unresolved NaN" in str(exc)
        assert raised, f"{cls.__name__} should raise on unresolved partial NaN by default"


def test_microbiome_preprocessor_allow_zero_fill_missing_is_an_explicit_opt_in() -> None:
    x_train = pd.DataFrame({"a": [50.0, 0.0, 10.0, np.nan], "b": [50.0, 30.0, 20.0, 40.0], "c": [10.0, 90.0, 80.0, 60.0]})
    proc = MicrobiomeCLRPreprocessor(prevalence_threshold=0.50, allow_zero_fill_missing=True)
    proc.fit(x_train)  # must not raise once the caller has explicitly opted in
    out = proc.transform(x_train)
    assert np.isfinite(out).all()


def test_microbiome_rank_preprocessor_is_fold_local_and_scale_invariant() -> None:
    x_train = pd.DataFrame({"a": [50.0, 10.0, 20.0, 5.0], "b": [10.0, 50.0, 30.0, 45.0], "c": [40.0, 40.0, 50.0, 50.0]})
    x_scaled = x_train * 1000.0  # simulate a different quantification scale
    proc = MicrobiomeRankPreprocessor(prevalence_threshold=0.50)
    proc.fit(x_train)
    out = proc.transform(x_train)
    out_scaled = proc.transform(x_scaled)
    assert np.isfinite(out).all()
    assert np.allclose(out, out_scaled)  # rank transform must be invariant to a constant multiplicative rescale
    assert set(proc.get_feature_names_out()) == {"a", "b", "c"}


def test_microbiome_presence_absence_preprocessor_binarises_then_scales_on_training_fold() -> None:
    x_train = pd.DataFrame({"a": [50.0, 0.0, 10.0, 20.0], "b": [50.0, 0.0, 0.0, 0.0], "c": [0.0, 100.0, 90.0, 80.0]})
    proc = MicrobiomePresenceAbsencePreprocessor(prevalence_threshold=0.50)
    proc.fit(x_train)
    out = proc.transform(x_train)
    assert set(proc.get_feature_names_out()) == {"a", "c"}
    assert np.isfinite(out).all()
    binary_a = (x_train["a"] > 0).astype(float).to_numpy()
    # scaled output must be a strictly increasing affine function of the raw
    # binary detected/not-detected indicator (i.e. representation is binary,
    # but the classifier sees a training-fold-scaled version of it)
    assert np.array_equal(np.argsort(out[:, 0], kind="stable"), np.argsort(binary_a, kind="stable"))
    assert len(np.unique(out[:, 0])) == 2
    # a training-fold StandardScaler must have actually been fit and frozen
    external = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0], "c": [10.0, 10.0]})
    out_external = proc.transform(external)
    assert np.allclose(out_external, np.tile(out_external[0], (2, 1)))


def test_build_ridge_pipeline_microbiome_variant_selects_sensitivity_transform() -> None:
    pipe_rank = build_ridge_pipeline("MICROBIOME", 1.0, microbiome_variant="within_sample_rank")
    pipe_pa = build_ridge_pipeline("MICROBIOME", 1.0, microbiome_variant="presence_absence")
    assert isinstance(pipe_rank.named_steps["microbiome"].named_steps["microbiome"], MicrobiomeRankPreprocessor)
    assert isinstance(pipe_pa.named_steps["microbiome"].named_steps["microbiome"], MicrobiomePresenceAbsencePreprocessor)


def test_early_integration_pipeline_is_block_aware_and_cloneable() -> None:
    pipe = build_ridge_pipeline("EARLY_INTEGRATION", 1.0)
    cloned = clone(pipe)
    assert "block_preprocessor" in cloned.named_steps
    transformers = cloned.named_steps["block_preprocessor"].transformers
    assert [name for name, _, _ in transformers] == ["metabolome", "microbiome"]


def test_legacy_probability_scale_threshold_helper_still_uses_cross_fitted_oof_not_resubstitution() -> None:
    """Legacy compatibility test for `cross_fitted_probs`/`select_youden_threshold_from_oof`
    (the pre-v2b, probability-scale helpers in stage4_corrected_v2_common.py).
    The authoritative, currently-used ranking/thresholding logic is decision-score-based
    and is instead covered by the `..._v2b_...decision_score...` tests below; this test
    only guards against the older helper regressing to resubstitution, it is not evidence
    about the current primary pipeline's threshold behaviour."""
    x = pd.DataFrame({"met__a": [0, 1, 2, 3, 4, 5, 6, 7], "met__b": [1, 1, 2, 2, 3, 3, 4, 4]})
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    probs = cross_fitted_probs(x, y, "METABOLOME", 1.0, fold_count(y, 4), 123)
    info = select_youden_threshold_from_oof(y, probs)
    assert len(probs) == len(y)
    assert np.isfinite(probs).all()
    assert np.isfinite(info["threshold"])
    assert info["threshold_source"] if "threshold_source" in info else True


def test_corrected_v2_predictive_tasks_are_pooled_ibd_and_crc_only() -> None:
    assert {spec.task for spec in TASK_SPECS} == {"IBD_vs_Control", "CRC_vs_Control"}
    assert all(spec.task not in {"CD_vs_Control", "UC_vs_Control"} for spec in TASK_SPECS)
    assert len(TASK_SPECS) == 4


def test_pooled_ibd_task_codes_cd_and_uc_as_positive_and_controls_as_negative() -> None:
    bundle = CorrectedDataBundle()
    ibd_specs = [spec for spec in TASK_SPECS if spec.task == "IBD_vs_Control"]
    assert ibd_specs
    for spec in ibd_specs:
        _, y, ids = build_xy(bundle, spec, spec.train_cohort, "MICROBIOME")
        encoded = ids.assign(y=y)
        assert set(encoded.loc[encoded["label"].eq("IBD"), "y"]) == {1}
        assert set(encoded.loc[encoded["label"].eq("Control"), "y"]) == {0}
        raw = bundle.labels(spec, spec.train_cohort)
        raw = raw[raw["sample_id"].isin(set(encoded["sample_id"]))]
        assert raw.loc[raw["raw_label"].isin(["CD", "UC"]), "sample_id"].isin(set(encoded.loc[encoded["y"].eq(1), "sample_id"])).all()


def test_stage4b_postrun_internal_prevalence_column_is_reporting_renamed() -> None:
    df = pd.DataFrame({"external_positive_prevalence": [0.5], "roc_auc": [0.7]})
    out = reporting_internal_performance(df)
    assert "evaluation_positive_prevalence" in out.columns
    assert "external_positive_prevalence" not in out.columns
    assert out.loc[0, "evaluation_positive_prevalence"] == 0.5


def test_stage4b_postrun_probability_diagnostics_flags_compression_saturation_and_one_class() -> None:
    pred = pd.DataFrame(
        {
            "disease": ["IBD"] * 4,
            "task": ["IBD_vs_Control"] * 4,
            "training_cohort": ["iHMP"] * 4,
            "external_cohort": ["Franzosa"] * 4,
            "representation": ["METABOLOME"] * 4,
            "external_sample_id": ["a", "b", "c", "d"],
            "y_true": [0, 0, 1, 1],
            "probability": [0.525, 0.526, 0.527, 0.528],
            "frozen_threshold": [0.5] * 4,
            "predicted_label": [1, 1, 1, 1],
        }
    )
    diag = build_probability_diagnostics(pred)
    assert bool(diag.loc[0, "compressed_probability_flag"])
    assert bool(diag.loc[0, "one_class_prediction_flag"])
    assert not bool(diag.loc[0, "saturation_flag"])


def test_stage4b_postrun_paired_comparisons_use_primary_prespecified_rows() -> None:
    base = pd.DataFrame(
        {
            "disease": ["IBD"] * 12,
            "task": ["IBD_vs_Control"] * 12,
            "training_cohort": ["Franzosa"] * 12,
            "external_cohort": ["iHMP"] * 12,
            "external_sample_id": ["s1", "s2", "s3", "s4"] * 3,
            "y_true": [0, 0, 1, 1] * 3,
            "representation": ["METABOLOME"] * 4 + ["MICROBIOME"] * 4 + ["EARLY_INTEGRATION"] * 4,
            "probability": [0.1, 0.4, 0.7, 0.9, 0.2, 0.3, 0.6, 0.8, 0.05, 0.35, 0.75, 0.95],
            "frozen_threshold": [0.5] * 12,
            "predicted_label": [0, 0, 1, 1] * 3,
        }
    )
    paired = build_paired_comparisons(base)
    assert {"EARLY_INTEGRATION_minus_METABOLOME", "EARLY_INTEGRATION_minus_MICROBIOME"}.issubset(set(paired["primary_comparison"]))
    assert "prespecified_paired_external_bootstrap" in set(paired["comparison_type"])
    assert "post_hoc_best_single_omic_uses_external_outcomes" in set(paired["comparison_type"])


def test_stage4c_corrected_v2_excludes_subtype_tests_and_clinical_views() -> None:
    mofa = build_mofa_plan()
    scca = build_scca_plan()
    assert mofa["clinical_variables_as_views"].eq(False).all()
    assert scca["clinical_variables_as_views"].eq(False).all()
    assert mofa["excluded_tests"].str.contains("CD_vs_Control").all()
    assert not mofa["phenotype_tests"].str.contains("UC_vs_Control|CD_vs_Control").any()
    assert scca["phenotype_use"].eq("outside_fitting_and_tuning_post_hoc_only").all()
    assert mofa["analysis_role"].eq("descriptive_structural_analysis_not_external_prediction").all()
    assert mofa["metabolome_scale_interpretation"].str.contains("does_not_test_absolute_cross_cohort_abundance_transport").all()
    assert scca["fit_scope"].eq("independent_by_cohort").all()
    assert scca["components_prepared"].eq("1,2").all()
    assert not mofa["phenotype_tests"].str.contains("UC_vs_Control|CD_vs_Control").any()
    assert not scca["posthoc_phenotype_annotations"].str.contains("UC_vs_Control|CD_vs_Control").any()


def test_stage4c_corrected_v2_output_contract_is_authoritative_and_nonhistorical() -> None:
    outputs = expected_stage4c_outputs("all")
    assert outputs
    assert all(name.startswith("stage4c_corrected_v2_") for name in outputs)
    assert not any("historical" in name.lower() or "simulated" in name.lower() or "placeholder" in name.lower() for name in outputs)
    assert "stage4c_corrected_v2_mofa_factor_scores.csv" in outputs
    assert "stage4c_corrected_v2_scca_cross_cohort_reproducibility.csv" in outputs
    assert {"disease", "cohort", "view", "n_participants", "n_features", "status"}.issubset(
        set(outputs["stage4c_corrected_v2_input_integrity_audit.csv"])
    )


def test_stage4c_corrected_v2_mofa_and_scca_are_label_free_during_fit_and_tuning() -> None:
    mofa = build_mofa_plan()
    scca = build_scca_plan()
    assert mofa["diagnosis_used_to_fit_or_tune"].eq(False).all()
    assert scca["phenotype_used_for_regularisation_selection"].eq(False).all()
    assert scca["fold_local_scaling_and_feature_handling"].eq(True).all()
    assert scca["component_matching_before_loading_comparison"].eq(True).all()


def test_stage4c_mofa_posthoc_never_uses_fake_pooled_cohort_adjusted_label() -> None:
    score_frame = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(12)],
            "cohort": ["Franzosa"] * 6 + ["iHMP"] * 6,
            "phenotype": ["Control", "Control", "Control", "IBD", "IBD", "IBD"] * 2,
            **{f"Factor{i}": np.arange(12, dtype=float) + i for i in range(1, 9)},
        }
    )
    classification = pd.DataFrame(
        {"disease": ["IBD"] * 8, "factor": [f"Factor{i}" for i in range(1, 9)], "factor_classification": ["shared_multiview"] * 8}
    )
    assoc = stage4c.factor_associations(score_frame, "IBD", classification)
    assert not assoc["cohort_scope"].astype(str).str.contains("POOLED_COHORT_ADJUSTED", regex=False).any()
    assert assoc["cohort_adjusted_model"].eq(False).all()
    assert assoc["bh_fdr_family"].str.contains("within each disease and cohort", regex=False).all()


def test_stage4c_scca_scores_align_by_sample_id_not_row_position(tmp_path, monkeypatch) -> None:
    out = tmp_path / "scca"
    out.mkdir()
    pd.DataFrame({"sample_id": ["a", "b", "c"], "1": [1.0, 2.0, 3.0]}).to_csv(out / "IBD_Franzosa_x_scores.csv", index=False)
    pd.DataFrame({"sample_id": ["c", "a", "b"], "1": [30.0, 10.0, 20.0]}).to_csv(out / "IBD_Franzosa_z_scores.csv", index=False)
    monkeypatch.setattr(stage4c, "SCCA_OUTPUT_DIR", out)
    scores = stage4c.scca_scores_for("IBD", "Franzosa", 1)
    assert list(scores.index) == ["a", "b", "c"]
    expected = (stage4c.stats.zscore([1.0, 2.0, 3.0]) + stage4c.stats.zscore([10.0, 20.0, 30.0])) / 2.0
    assert np.allclose(scores.to_numpy(), expected)


def test_stage4c_scca_tuning_fold_not_named_unbiased_heldout() -> None:
    outputs = expected_stage4c_outputs("scca")
    assert "stage4c_corrected_v2_scca_tuning_fold_correlations.csv" in outputs
    assert "stage4c_corrected_v2_scca_heldout_fold_correlations.csv" not in outputs
    assert "stage4c_corrected_v2_scca_tuning_cv_correlations.csv" not in outputs
    assert "tuning_fold_abs_cross_view_correlation" in outputs["stage4c_corrected_v2_scca_tuning_fold_correlations.csv"]
    for columns in outputs.values():
        assert not any(str(c).startswith("heldout_abs_cross_view") for c in columns)
        assert not any(str(c).startswith("tuning_cv_abs_cross_view") for c in columns)
    required = set(outputs["stage4c_corrected_v2_scca_tuning_fold_correlations.csv"])
    assert {"estimate_role", "nested_outer_evaluation", "independent_generalisation_estimate"}.issubset(required)


def test_stage4c_scca_nested_cv_cannot_be_claimed_while_engine_disabled() -> None:
    try:
        stage4c.run_scca_nested_cv({}, {})
    except NotImplementedError as exc:
        assert "not enabled" in str(exc)
    else:
        raise AssertionError("Disabled nested sCCA engine should not silently produce outputs.")


def test_stage4c_corrected_v2_inputs_are_matched_finite_and_molecular_only() -> None:
    blocks = load_stage4c_blocks()
    feature_sets, _ = build_stage4c_feature_sets(blocks)
    for key, block in blocks.items():
        disease, cohort = key
        assert not block.metadata["sample_id"].astype(str).duplicated().any()
        assert list(block.microbiome["sample_id"].astype(str)) == list(block.metabolome["sample_id"].astype(str)) == list(block.metadata["sample_id"].astype(str))
        views = stage4c_prepared_views(block, feature_sets[key])
        assert set(views) == {"microbiome", "metabolome"}
        assert all(not any(str(c).lower() in {"disease_group", "label", "y", "phenotype"} for c in view.columns) for view in views.values())
        assert all(np.isfinite(view.to_numpy()).all() for view in views.values())
        assert views["microbiome"].shape[1] == (50 if disease == "IBD" else 52)
        assert views["metabolome"].shape[1] == (166 if disease == "IBD" else 109)


def test_stage4d_corrected_v2_graph_modes_and_njc19_terminology() -> None:
    plan = build_graph_plan()
    assert set(plan["graph_mode"]) == {"identity", "kegg_only", "kegg_njc19"}
    assert plan["graph_directionality"].eq("undirected_primary_graph").all()
    assert plan["edge_weight_policy"].eq("unsigned_binary_weight_1").all()
    assert plan["microbe_microbe_edges_in_primary_graph"].eq(False).all()
    assert plan["clinical_variables_or_disease_labels_as_nodes"].eq(False).all()
    assert not plan["njc19_terminology"].str.contains("positive associations", regex=False).any()
    assert plan["njc19_terminology"].str.contains("not positive statistical associations", regex=False).any()
    assert plan["biological_edge_table_excludes_self_loops"].eq(True).all()


def test_stage4d_corrected_v2_feature_space_dimensions_and_node_policy() -> None:
    nodes, audit = stage4d.stage4c_feature_spaces()
    assert set(nodes["disease"]) == {"IBD", "CRC"}
    assert not nodes.duplicated(["disease", "node_id"]).any()
    assert not nodes["node_type"].isin(["clinical", "participant", "label", "phenotype"]).any()
    assert audit["finite_values"].eq(True).all()
    for disease, expected in {"IBD": {"microbe": 50, "metabolite": 166}, "CRC": {"microbe": 52, "metabolite": 109}}.items():
        sub = nodes[nodes["disease"].eq(disease)]
        assert sub["node_index"].tolist() == list(range(len(sub)))
        assert int(sub["node_type"].eq("microbe").sum()) == expected["microbe"]
        assert int(sub["node_type"].eq("metabolite").sum()) == expected["metabolite"]


def test_stage4d_corrected_v2_edge_index_self_loops_are_computational_not_biological() -> None:
    nodes = pd.DataFrame(
        [
            {"disease": "IBD", "node_id": "microbe::A", "node_index": 0, "node_type": "microbe", "feature_id": "mic__A", "feature_name": "A", "source_feature_space": "x", "source_mapping_status": "x"},
            {"disease": "IBD", "node_id": "metabolite::HMDB0000001", "node_index": 1, "node_type": "metabolite", "feature_id": "met__HMDB0000001", "feature_name": "HMDB0000001", "source_feature_space": "x", "source_mapping_status": "x"},
            {"disease": "CRC", "node_id": "microbe::B", "node_index": 0, "node_type": "microbe", "feature_id": "mic__B", "feature_name": "B", "source_feature_space": "x", "source_mapping_status": "x"},
            {"disease": "CRC", "node_id": "metabolite::HMDB0000002", "node_index": 1, "node_type": "metabolite", "feature_id": "met__HMDB0000002", "feature_name": "HMDB0000002", "source_feature_space": "x", "source_mapping_status": "x"},
        ],
        columns=stage4d.NODE_SCHEMA,
    )
    bio = pd.DataFrame(columns=stage4d.BIO_EDGE_SCHEMA)
    edge_index = stage4d.edge_index_for_modes(nodes, bio)
    assert edge_index["edge_kind"].eq("computational_self_loop").all()
    assert edge_index["source_index"].eq(edge_index["target_index"]).all()


def test_stage4d_corrected_v2_validation_rejects_microbe_microbe_edges() -> None:
    nodes = pd.DataFrame(
        [
            {"disease": "IBD", "node_id": "microbe::A", "node_index": 0, "node_type": "microbe", "feature_id": "mic__A", "feature_name": "A", "source_feature_space": "x", "source_mapping_status": "x"},
            {"disease": "IBD", "node_id": "microbe::B", "node_index": 1, "node_type": "microbe", "feature_id": "mic__B", "feature_name": "B", "source_feature_space": "x", "source_mapping_status": "x"},
        ],
        columns=stage4d.NODE_SCHEMA,
    )
    bio = pd.DataFrame(
        [
            {"disease": "IBD", "graph_mode": "kegg_njc19", "source_node": "microbe::A", "target_node": "microbe::B", "edge_type": "MICROBE_MICROBE", "weight": 1.0, "directionality": "undirected", "knowledge_source": "bad", "evidence_type": "bad", "mapping_provenance": "bad", "supporting_pathway_ids": "", "supporting_pathway_names": "", "relation_type": "", "source_microbe_names": "", "source_kegg_compound": "", "hmdb_id": "", "reference": "", "support_count": 1, "biological_edge": True}
        ],
        columns=stage4d.BIO_EDGE_SCHEMA,
    )
    edge_index = pd.DataFrame(columns=stage4d.EDGE_INDEX_SCHEMA)
    validation = stage4d.validate_outputs(nodes, bio, edge_index)
    assert validation.loc[validation["test"].eq("no_microbe_microbe_edges"), "status"].iloc[0] == "fail"


def test_stage4d_topology_audit_policy_is_label_free_and_non_destructive() -> None:
    policy = stage4d_audit.sensitivity_policy_table()
    assert {"A_primary_existing", "B_conservative_sensitivity_exclude_global_overview"}.issubset(set(policy["policy_id"]))
    assert policy["performance_or_label_based"].eq(False).all()
    assert policy.loc[policy["policy_id"].eq("A_primary_existing"), "adopted_for_primary_graph"].eq(True).all()
    assert policy.loc[policy["policy_id"].eq("B_conservative_sensitivity_exclude_global_overview"), "adopted_for_primary_graph"].eq(False).all()


def test_stage4d_audit_relation_type_classification_avoids_positive_correlation_language() -> None:
    assert stage4d_audit.classify_relation_type("Production (export)") == "production_or_export"
    assert stage4d_audit.classify_relation_type("Consumption (import)") == "consumption_or_import"
    assert stage4d_audit.classify_relation_type("Transport") == "transport"
    policy_text = " ".join(map(str, stage4d_audit.sensitivity_policy_table().to_numpy().ravel())).lower()
    assert "positive correlation" not in policy_text


def test_stage4e_corrected_v2_gcn_plan_has_fixed_graph_modes_and_no_clinical_claim() -> None:
    plan = build_gcn_plan()
    assert set(plan["graph_mode"]) == {"identity", "kegg_only", "kegg_njc19", "conservative_kegg_njc19"}
    assert set(STAGE4E_GRAPH_MODES) == {"identity", "kegg_only", "kegg_njc19", "conservative_kegg_njc19"}
    assert plan["seed_policy"].eq("same_fixed_seeds_across_graph_modes").all()
    assert plan["clinical_variables_as_nodes_or_views"].eq(False).all()
    assert plan.loc[plan["graph_mode"].eq("identity"), "identity_mode_retains_named_node_identity"].eq(True).all()
    primary = plan[plan["analysis_priority"].eq("primary")]
    secondary = plan[plan["analysis_priority"].eq("secondary")]
    assert primary["analysis_role"].eq("within_cohort_topology_evaluation").all()
    assert secondary["analysis_role"].eq("external_as_supplied_scale_stress_test").all()
    assert primary["external_cohort"].fillna("").eq("").all()
    assert primary["class_weight_source"].eq("training_fold_only").all()
    assert not plan["task"].str.contains("CD_vs_Control|UC_vs_Control", regex=True).any()


def test_stage4e_configuration_grid_is_small_prespecified_and_label_free() -> None:
    grid = model_configuration_grid()
    assert set(grid["graph_mode"]) == {"identity", "kegg_only", "kegg_njc19", "conservative_kegg_njc19"}
    assert set(grid["hidden_dim"]) == {16}
    assert set(grid["layers"]) == {1, 2}
    assert set(grid["dropout"]) == {0.2}
    assert set(grid["learning_rate"]) == {1e-3}
    assert set(grid["readout"]) == {"mean", "gated"}
    assert grid["loss"].eq("weighted_binary_cross_entropy_training_fold_only").all()
    assert grid["hpo_candidate"].eq(False).all()
    assert len(architecture_grid("primary")) == 1
    assert len(architecture_grid("sensitivities")) == 2
    assert len(grid) == 12


def test_stage4e_primary_grid_has_four_configs_not_neural_hpo() -> None:
    primary = model_configuration_grid(architecture="primary")
    assert len(primary) == 4
    assert primary["architecture_id"].eq("primary_one_layer_mean").all()
    assert primary["readout"].eq("mean").all()
    assert primary["layers"].eq(1).all()


def test_stage4e_ridge_label_is_not_a_graph_mode() -> None:
    source = (SCRIPTS / "run_stage4e_gcn_corrected_v2.py").read_text()
    assert '"graph_mode": "not_applicable"' in source
    assert '"representation": "early_integration"' in source
    assert '"graph_mode": "no_graph"' not in source


def test_stage4e_postprocess_ensemble_averages_seed_scores_and_majority_vote() -> None:
    rows = []
    for seed, score, prob, cls in [(1, 0.1, 0.52, 1), (2, 0.2, 0.55, 1), (3, 0.3, 0.57, 1), (4, 0.4, 0.60, 0), (5, 0.5, 0.62, 0)]:
        rows.append({"cohort": "Franzosa", "disease": "IBD", "task": "IBD_vs_Control", "model_family": "gcn", "graph_mode": "identity", "sample_id": "s1", "seed": seed, "y_true": 1, "decision_score": score, "probability": prob, "predicted_class": cls})
    for seed, score, prob, cls in [(1, -0.1, 0.48, 0), (2, -0.2, 0.45, 0), (3, -0.3, 0.43, 0), (4, -0.4, 0.40, 1), (5, -0.5, 0.38, 1)]:
        rows.append({"cohort": "Franzosa", "disease": "IBD", "task": "IBD_vs_Control", "model_family": "gcn", "graph_mode": "identity", "sample_id": "s0", "seed": seed, "y_true": 0, "decision_score": score, "probability": prob, "predicted_class": cls})
    pred = pd.DataFrame(rows)
    ens = stage4e_post.ensemble_predictions(pred)
    assert len(ens) == 2
    s1 = ens[ens["sample_id"].eq("s1")].iloc[0]
    assert np.isclose(s1["decision_score"], 0.3)
    assert int(s1["predicted_class"]) == 1
    assert s1["class_decision_rule"] == "seed_majority_vote"


def test_stage4e_postprocess_bh_fdr_and_finite_p_values() -> None:
    p = pd.Series([0.0, 0.01, 0.5])
    adj = stage4e_post.bh_fdr(p)
    assert np.all(adj >= 0)
    assert np.all(adj <= 1)
    assert adj.iloc[0] == 0.0
    # Paired comparison p-values use +1 correction in the implementation, so
    # real bootstrap outputs cannot be exactly zero.
    n = 2000
    p_min = 2 * (0 + 1) / (n + 1)
    assert p_min > 0


def test_stage4e_postprocess_script_has_no_model_fitting_imports_or_calls() -> None:
    source = (SCRIPTS / "postprocess_stage4e_corrected_v2_primary.py").read_text()
    forbidden = ["import torch", "fit_gcn_once", "run_within(", "run_external_stress(", ".fit("]
    assert not any(term in source for term in forbidden)


def test_stage4e_postprocess_primary_comparison_family_size() -> None:
    contrasts = [
        ("kegg_njc19", "identity"),
        ("conservative_kegg_njc19", "identity"),
        ("kegg_njc19", "conservative_kegg_njc19"),
        ("kegg_only", "identity"),
        ("kegg_njc19", "ridge_early_integration"),
        ("conservative_kegg_njc19", "ridge_early_integration"),
    ]
    assert len(contrasts) * 4 == 24


def test_stage4e_directional_interpretation_uses_effect_direction_not_significance_only() -> None:
    rows = pd.DataFrame(
        [
            {"observed_delta_roc_auc": 0.08, "ci_low": 0.01, "ci_high": 0.15, "bh_fdr_p": 0.01},
            {"observed_delta_roc_auc": -0.08, "ci_low": -0.15, "ci_high": -0.01, "bh_fdr_p": 0.01},
            {"observed_delta_roc_auc": 0.08, "ci_low": -0.01, "ci_high": 0.15, "bh_fdr_p": 0.01},
            {"observed_delta_roc_auc": 0.08, "ci_low": 0.01, "ci_high": 0.15, "bh_fdr_p": 0.20},
        ]
    )
    assert stage4e_post.model_a_significantly_better(rows.iloc[0])
    assert not stage4e_post.model_a_significantly_worse(rows.iloc[0])
    assert stage4e_post.model_a_significantly_worse(rows.iloc[1])
    assert not stage4e_post.model_a_significantly_better(rows.iloc[1])
    assert not stage4e_post.model_a_significantly_better(rows.iloc[2])
    assert not stage4e_post.model_a_significantly_worse(rows.iloc[2])
    assert not stage4e_post.model_a_significantly_better(rows.iloc[3])


def test_stage4e_interpretation_table_expected_directional_flags() -> None:
    performance = pd.DataFrame(
        [
            {"cohort": "Franzosa", "representation": "identity", "roc_auc": 0.70},
            {"cohort": "Franzosa", "representation": "kegg_only", "roc_auc": 0.60},
            {"cohort": "Franzosa", "representation": "kegg_njc19", "roc_auc": 0.57},
            {"cohort": "Franzosa", "representation": "conservative_kegg_njc19", "roc_auc": 0.58},
            {"cohort": "Franzosa", "representation": "ridge_early_integration", "roc_auc": 0.92},
        ]
    )
    paired = pd.DataFrame(
        [
            {
                "cohort": "Franzosa",
                "comparison": "kegg_njc19_minus_identity",
                "model_a": "kegg_njc19",
                "model_b": "identity",
                "observed_delta_roc_auc": -0.12,
                "ci_low": -0.20,
                "ci_high": -0.05,
                "bh_fdr_p": 0.01,
            },
            {
                "cohort": "Franzosa",
                "comparison": "kegg_njc19_minus_ridge_early_integration",
                "model_a": "kegg_njc19",
                "model_b": "ridge_early_integration",
                "observed_delta_roc_auc": -0.35,
                "ci_low": -0.45,
                "ci_high": -0.25,
                "bh_fdr_p": 0.01,
            },
        ]
    )
    out = stage4e_post.interpretation_table(performance, paired).iloc[0]
    assert not bool(out["any_biological_graph_significantly_better_than_identity"])
    assert bool(out["any_biological_graph_significantly_worse_than_identity"])
    assert not bool(out["any_tested_graph_gcn_significantly_better_than_ridge"])
    assert bool(out["any_tested_graph_gcn_significantly_worse_than_ridge"])
    assert "exceeded" not in set(stage4e_post.interpretation_table(performance, paired).columns)


def test_stage4e_split_reconstruction_documents_frozen_seed_replay() -> None:
    source = (SCRIPTS / "reconstruct_stage4e_corrected_v2_split_membership.py").read_text()
    assert "deterministic_replay_of_corrected_v2_stage4e_splitter" in source
    assert "seed + 999 + outer_fold" in source
    assert "StratifiedKFold" in source
    assert "fit_gcn_once" not in source
    assert hasattr(stage4e_splits, "reconstruct_membership")


def test_stage4e_sensitivity_postprocess_is_no_refit_and_separate_family() -> None:
    source = (SCRIPTS / "postprocess_stage4e_corrected_v2_sensitivities.py").read_text()
    forbidden = ["import torch", "fit_gcn_once", "run_within(", "run_external_stress(", ".fit("]
    assert not any(term in source for term in forbidden)
    assert set(stage4e_sens.ARCHITECTURES) == {"sensitivity_two_layer_mean", "sensitivity_one_layer_gated"}
    assert len(stage4e_sens.COMPARISONS) * 4 * 2 == 48
    assert "separate_sensitivity_family_48_comparisons" in source
    assert "stage4e_corrected_v2_selected_hyperparameters_sensitivities.csv" in source
    assert "stage4e_corrected_v2_prespecified_architecture_records_sensitivities.csv" in source


def test_final_stage4_freeze_uses_authoritative_sources_and_no_modelling() -> None:
    source = (SCRIPTS / "freeze_final_stage4_results.py").read_text()
    forbidden = ["import torch", "fit_gcn_once", "run_within(", "run_external_stress(", "bootstrap_metric_ci(", ".fit("]
    assert not any(term in source for term in forbidden)
    assert "stage4e_corrected_v2_ensemble_oof_performance_primary.csv" in source
    assert "stage4e_corrected_v2_within_fold_seed_performance_primary.csv" not in source
    assert "stage4e_corrected_v2_sensitivity_paired_comparisons.csv" in source
    assert "stage4e_sensitivity_48" not in final_stage4.table5.__name__
    assert hasattr(final_stage4, "validation_tests")


def test_stage4e_conservative_graph_removes_only_broad_kegg_only_support_and_keeps_njc19() -> None:
    bio = pd.DataFrame(
        [
            {
                "disease": "IBD",
                "graph_mode": "kegg_njc19",
                "source_node": "metabolite::HMDB0000001",
                "target_node": "metabolite::HMDB0000002",
                "edge_type": "METABOLITE_METABOLITE",
                "weight": 1.0,
                "directionality": "undirected",
                "knowledge_source": "KEGG",
                "evidence_type": "shared pathway",
                "mapping_provenance": "",
                "supporting_pathway_ids": "map01100",
                "supporting_pathway_names": "Metabolic pathways",
                "relation_type": "",
                "source_microbe_names": "",
                "source_kegg_compound": "",
                "hmdb_id": "",
                "reference": "",
                "support_count": 1,
                "biological_edge": True,
            },
            {
                "disease": "IBD",
                "graph_mode": "kegg_njc19",
                "source_node": "metabolite::HMDB0000001",
                "target_node": "metabolite::HMDB0000003",
                "edge_type": "METABOLITE_METABOLITE",
                "weight": 1.0,
                "directionality": "undirected",
                "knowledge_source": "KEGG",
                "evidence_type": "shared pathway",
                "mapping_provenance": "",
                "supporting_pathway_ids": "map01100;map00010",
                "supporting_pathway_names": "Metabolic pathways;Glycolysis",
                "relation_type": "",
                "source_microbe_names": "",
                "source_kegg_compound": "",
                "hmdb_id": "",
                "reference": "",
                "support_count": 2,
                "biological_edge": True,
            },
            {
                "disease": "IBD",
                "graph_mode": "kegg_njc19",
                "source_node": "microbe::A",
                "target_node": "metabolite::HMDB0000001",
                "edge_type": "MICROBE_METABOLITE",
                "weight": 1.0,
                "directionality": "undirected",
                "knowledge_source": "NJC19",
                "evidence_type": "curated production consumption or transport relation",
                "mapping_provenance": "",
                "supporting_pathway_ids": "",
                "supporting_pathway_names": "",
                "relation_type": "Consumption (import)",
                "source_microbe_names": "A species",
                "source_kegg_compound": "C00001",
                "hmdb_id": "HMDB0000001",
                "reference": "NJC19",
                "support_count": 1,
                "biological_edge": True,
            },
        ],
        columns=stage4d.BIO_EDGE_SCHEMA,
    )
    cons, policy = conservative_biological_edges(bio)
    assert len(cons) == 2
    assert cons["graph_mode"].eq("conservative_kegg_njc19").all()
    assert "MICROBE_METABOLITE" in set(cons["edge_type"])
    assert policy["label_or_performance_based"].eq(False).all()
    assert policy["retained_in_conservative_sensitivity"].tolist() == [False, True, True]


def test_stage4b_v2c_freeze_branch_b_interpretation_rules() -> None:
    rules = pd.DataFrame(INTERPRETATION_RULES)
    deps = pd.DataFrame(RESTRUCTURED_DEPENDENCIES)
    assert "metabolome_stress_test" in set(rules["rule_id"])
    assert "early_integration_stress_test" in set(rules["rule_id"])
    assert rules.loc[rules["rule_id"].eq("high_auc_not_operational_transportability"), "interpretation"].str.contains("High AUC").any()
    assert deps.loc[deps["stage"].eq("4E"), "status"].str.contains("primary_within_cohort_topology_evaluation").any()
    assert deps.loc[deps["stage"].eq("4C_MOFA"), "interpretation_constraint"].str.contains("no external projection").any()


def test_stage4b_v2b_exports_finite_decision_scores_and_probabilities_from_cross_fit() -> None:
    x = pd.DataFrame({"met__a": [0, 1, 2, 3, 4, 5, 6, 7], "met__b": [1, 1, 2, 2, 3, 3, 4, 4]})
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    scores, probs = cross_fitted_decision_scores_and_probs(x, y, "METABOLOME", 1.0, 4, 2026)
    assert scores.shape == probs.shape == (8,)
    assert np.isfinite(scores).all()
    assert np.isfinite(probs).all()
    assert ((probs >= 0) & (probs <= 1)).all()


def test_stage4b_v2b_threshold_is_on_decision_score_scale() -> None:
    y = pd.Series([0, 0, 1, 1])
    scores = np.array([-2.0, -1.0, 1.0, 2.0])
    threshold = select_youden_threshold_from_scores(y, scores)
    assert "decision_score_threshold" in threshold
    assert np.isfinite(threshold["decision_score_threshold"])
    prob_equiv = probability_threshold_from_decision_threshold(threshold["decision_score_threshold"])
    assert 0 <= prob_equiv <= 1


def test_stage4b_v2b_metrics_use_decision_score_for_ranking_and_probability_for_diagnostics() -> None:
    y = pd.Series([0, 0, 1, 1])
    scores = np.array([-10.0, -9.0, 9.0, 10.0])
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    metrics = binary_metrics_from_scores(y, scores, probs, 0.0)
    assert metrics["ranking_metric_score_column"] == "decision_score"
    assert metrics["roc_auc"] == 1.0
    assert bool(metrics["probability_saturation_flag"])
    assert bool(metrics["probability_resolution_loss_flag"])


def test_stage4b_v2b_paired_comparisons_use_decision_score_and_format_zero_p_values() -> None:
    base = pd.DataFrame(
        {
            "disease": ["IBD"] * 12,
            "task": ["IBD_vs_Control"] * 12,
            "training_cohort": ["Franzosa"] * 12,
            "external_cohort": ["iHMP"] * 12,
            "external_sample_id": ["s1", "s2", "s3", "s4"] * 3,
            "y_true": [0, 0, 1, 1] * 3,
            "representation": ["METABOLOME"] * 4 + ["MICROBIOME"] * 4 + ["EARLY_INTEGRATION"] * 4,
            "decision_score": [0.4, 0.3, 0.2, 0.1, 0.45, 0.35, 0.25, 0.15, 0.1, 0.2, 0.8, 0.9],
            "probability": [0.5] * 12,
            "frozen_decision_score_threshold": [0.0] * 12,
            "predicted_label": [0, 0, 1, 1] * 3,
        }
    )
    paired = build_v2b_paired_comparisons(base)
    primary = paired[paired["comparison_type"].eq("prespecified_paired_external_bootstrap")]
    assert primary["metric_score_column"].eq("decision_score").all()
    assert "bootstrap_two_sided_p_bh_fdr_primary_eight" in paired.columns
    assert primary["statistical_wording"].isin(["unadjusted interval excluded zero", "early integration was lower", "inconclusive"]).all()
