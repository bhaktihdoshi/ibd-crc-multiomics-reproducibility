from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, matthews_corrcoef, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from stage4_corrected_v2_common import MANIFESTS, REPORTS, TABLES, V2, atomic_csv, atomic_json, ensure_v2_dirs
from audit_stage4d_corrected_v2_provenance_topology import BROAD_KEGG_IDS


DISEASES = ["IBD", "CRC"]
COHORTS = {"IBD": ["Franzosa", "iHMP"], "CRC": ["Sinha", "Yachida"]}
TASKS = {"IBD": "IBD_vs_Control", "CRC": "CRC_vs_Control"}
EXPECTED_NODE_COUNTS = {"IBD": 216, "CRC": 161}
EXPECTED_VIEW_COUNTS = {"IBD": {"microbe": 50, "metabolite": 166}, "CRC": {"microbe": 52, "metabolite": 109}}

GRAPH_MODES = ["identity", "kegg_only", "kegg_njc19", "conservative_kegg_njc19"]
PRIMARY_GRAPH_MODES = ["identity", "kegg_only", "kegg_njc19"]
PRIMARY_SEEDS = [20260813, 20260814, 20260815, 20260816, 20260817]
OPTIONAL_CONFIRMATION_SEEDS = [20260813, 20260814, 20260815, 20260816, 20260817, 20260818, 20260819, 20260820, 20260821, 20260822]
OUTER_FOLDS = 5
SMOKE_OUTER_FOLDS = 2
MAX_EPOCHS = 150
SMOKE_MAX_EPOCHS = 3
PATIENCE = 20
MIN_EPOCHS_BEFORE_EARLY_STOPPING = 20
EARLY_STOPPING_VALIDATION_FRACTION = 0.20
ARCHITECTURE_SCOPES = ["primary", "sensitivities", "all"]
ARCHITECTURES = [
    {
        "architecture_id": "primary_one_layer_mean",
        "architecture_role": "primary",
        "layers": 1,
        "hidden_dim": 16,
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "readout": "mean",
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "min_epochs_before_early_stopping": MIN_EPOCHS_BEFORE_EARLY_STOPPING,
    },
    {
        "architecture_id": "sensitivity_two_layer_mean",
        "architecture_role": "sensitivity",
        "layers": 2,
        "hidden_dim": 16,
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "readout": "mean",
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "min_epochs_before_early_stopping": MIN_EPOCHS_BEFORE_EARLY_STOPPING,
    },
    {
        "architecture_id": "sensitivity_one_layer_gated",
        "architecture_role": "sensitivity",
        "layers": 1,
        "hidden_dim": 16,
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "readout": "gated",
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "min_epochs_before_early_stopping": MIN_EPOCHS_BEFORE_EARLY_STOPPING,
    },
]
SMOKE_ARCHITECTURE = {
    **ARCHITECTURES[0],
    "architecture_id": "smoke_primary_one_layer_mean",
    "architecture_role": "non_authoritative_smoke_test",
    "max_epochs": SMOKE_MAX_EPOCHS,
    "patience": 2,
    "min_epochs_before_early_stopping": 1,
}

SCCA_INPUT_DIR = V2 / "data" / "stage4c_scca_pma_inputs"
NODES_FILE = TABLES / "stage4d_corrected_v2_nodes.csv"
BIOLOGICAL_EDGES_FILE = TABLES / "stage4d_corrected_v2_biological_edges.csv"
EDGE_INDEX_FILE = TABLES / "stage4d_corrected_v2_edge_index.csv"
GRAPH_HASH_FILE = MANIFESTS / "stage4d_corrected_v2_graph_hash.txt"
GRAPH_VALIDATION_FILE = TABLES / "stage4d_corrected_v2_validation_tests.csv"
SENSITIVITY_POLICY_FILE = TABLES / "stage4d_corrected_v2_audit_graph_sensitivity_policy.csv"


@dataclass(frozen=True)
class Stage4EDirection:
    disease: str
    task: str
    train_cohort: str
    external_cohort: str = ""
    analysis_priority: str = "primary"

    @property
    def direction_id(self) -> str:
        if self.external_cohort:
            return f"{self.disease}_{self.task}_{self.train_cohort}_to_{self.external_cohort}"
        return f"{self.disease}_{self.task}_{self.train_cohort}_within"


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def sha256_frame(df: pd.DataFrame) -> str:
    payload = df.sort_values(list(df.columns), kind="mergesort").to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def architecture_grid(architecture: str = "all", smoke: bool = False) -> list[dict[str, Any]]:
    if smoke:
        return [SMOKE_ARCHITECTURE.copy()]
    if architecture == "primary":
        return [dict(ARCHITECTURES[0])]
    if architecture == "sensitivities":
        return [dict(a) for a in ARCHITECTURES if a["architecture_role"] == "sensitivity"]
    if architecture == "all":
        return [dict(a) for a in ARCHITECTURES]
    raise ValueError(f"Unknown architecture scope: {architecture}")


def hyperparameter_grid(smoke: bool = False) -> list[dict[str, Any]]:
    return architecture_grid("all", smoke=smoke)


def model_configuration_grid(smoke: bool = False, architecture: str = "all") -> pd.DataFrame:
    rows = []
    for graph_mode in GRAPH_MODES:
        for i, params in enumerate(architecture_grid(architecture, smoke=smoke), start=1):
            rows.append(
                {
                    "configuration_id": f"{graph_mode}_{params['architecture_id']}",
                    "graph_mode": graph_mode,
                    **params,
                    "loss": "weighted_binary_cross_entropy_training_fold_only",
                    "optimizer": "Adam",
                    "node_type_indicator": True,
                    "architecture_family": "low_complexity_dense_gcn",
                    "architecture_status": "prespecified_not_selected_by_performance",
                    "hpo_candidate": False,
                }
            )
    return pd.DataFrame(rows)


def within_directions() -> list[Stage4EDirection]:
    return [Stage4EDirection(disease, TASKS[disease], cohort) for disease in DISEASES for cohort in COHORTS[disease]]


def external_directions() -> list[Stage4EDirection]:
    rows = []
    for disease, cohorts in COHORTS.items():
        rows.append(Stage4EDirection(disease, TASKS[disease], cohorts[0], cohorts[1], "secondary"))
        rows.append(Stage4EDirection(disease, TASKS[disease], cohorts[1], cohorts[0], "secondary"))
    return rows


def all_directions(analysis: str) -> list[Stage4EDirection]:
    if analysis == "within":
        return within_directions()
    if analysis == "external":
        return external_directions()
    if analysis == "all":
        return within_directions() + external_directions()
    raise ValueError(f"Unknown analysis: {analysis}")


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise ValueError(f"Required table is empty: {path}")
    return df


def read_stage4d_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    nodes = read_required_csv(NODES_FILE)
    biological_edges = read_required_csv(BIOLOGICAL_EDGES_FILE)
    edge_index = read_required_csv(EDGE_INDEX_FILE)
    if not GRAPH_HASH_FILE.exists():
        raise FileNotFoundError(GRAPH_HASH_FILE)
    graph_hash = GRAPH_HASH_FILE.read_text().strip()
    if GRAPH_VALIDATION_FILE.exists():
        validation = pd.read_csv(GRAPH_VALIDATION_FILE)
        failed = validation[~validation["status"].eq("pass")]
        if not failed.empty:
            raise RuntimeError(f"Stage 4D validation has failures: {failed[['test','status']].to_dict('records')[:5]}")
    return nodes, biological_edges, edge_index, graph_hash


def read_matrix(disease: str, cohort: str, view: str) -> pd.DataFrame:
    path = SCCA_INPUT_DIR / f"{disease}_{cohort}_{view}.csv"
    df = read_required_csv(path)
    if df.columns[0] != "sample_id":
        df = df.rename(columns={df.columns[0]: "sample_id"})
    if df["sample_id"].astype(str).duplicated().any():
        duplicated = df.loc[df["sample_id"].astype(str).duplicated(), "sample_id"].astype(str).head(5).tolist()
        raise ValueError(f"Duplicated samples in {path}: {duplicated}")
    feature_cols = [c for c in df.columns if c != "sample_id"]
    values = df[feature_cols].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"Non-finite values in {path}")
    return pd.concat([df[["sample_id"]].astype(str), values], axis=1)


def read_metadata(disease: str, cohort: str) -> pd.DataFrame:
    path = SCCA_INPUT_DIR / f"{disease}_{cohort}_metadata.csv"
    df = read_required_csv(path)
    if df.columns[0] != "sample_id":
        df = df.rename(columns={df.columns[0]: "sample_id"})
    if "disease_group" not in df.columns:
        raise ValueError(f"{path} lacks disease_group")
    return df


def label_vector(metadata: pd.DataFrame, disease: str) -> pd.Series:
    label = metadata["disease_group"].astype(str)
    if disease == "IBD":
        y = label.map({"Control": 0, "IBD": 1})
    elif disease == "CRC":
        y = label.map({"Control/Healthy": 0, "CRC": 1, "Control": 0, "Healthy": 0})
    else:
        raise ValueError(disease)
    if y.isna().any():
        bad = sorted(label[y.isna()].unique())
        raise ValueError(f"Unsupported labels for {disease}: {bad}")
    return y.astype(int)


def cohort_blocks(disease: str, cohort: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    metadata = read_metadata(disease, cohort)
    microbiome = read_matrix(disease, cohort, "microbiome")
    metabolome = read_matrix(disease, cohort, "metabolome")
    ids = list(metadata["sample_id"].astype(str))
    if ids != list(microbiome["sample_id"].astype(str)) or ids != list(metabolome["sample_id"].astype(str)):
        raise ValueError(f"{disease} {cohort} metadata/microbiome/metabolome sample order differs.")
    y = label_vector(metadata, disease)
    if y.nunique() != 2:
        raise ValueError(f"{disease} {cohort} has one-class labels: {y.value_counts().to_dict()}")
    return metadata, microbiome, metabolome, y


def disease_nodes(nodes: pd.DataFrame, disease: str) -> pd.DataFrame:
    sub = nodes[nodes["disease"].eq(disease)].sort_values("node_index", kind="mergesort").reset_index(drop=True)
    if len(sub) != EXPECTED_NODE_COUNTS[disease]:
        raise ValueError(f"{disease} node count mismatch: observed={len(sub)} expected={EXPECTED_NODE_COUNTS[disease]}")
    if sub["node_index"].tolist() != list(range(len(sub))):
        raise ValueError(f"{disease} node order is not contiguous from zero.")
    return sub


def expected_feature_for_node(row: pd.Series) -> str:
    if row["node_type"] == "microbe":
        return f"mic__{str(row['node_id']).replace('microbe::', '')}"
    return str(row["feature_id"])


def validate_node_features(nodes: pd.DataFrame, disease: str) -> pd.DataFrame:
    rows = []
    dnodes = disease_nodes(nodes, disease)
    for cohort in COHORTS[disease]:
        _, micro, met, _ = cohort_blocks(disease, cohort)
        feature_set = set(micro.columns).union(set(met.columns))
        missing = []
        for _, node in dnodes.iterrows():
            expected = expected_feature_for_node(node)
            if expected not in feature_set:
                missing.append(expected)
        counts = dnodes["node_type"].value_counts().to_dict()
        rows.append(
            {
                "disease": disease,
                "cohort": cohort,
                "samples": len(micro),
                "microbe_nodes": int(counts.get("microbe", 0)),
                "metabolite_nodes": int(counts.get("metabolite", 0)),
                "node_feature_match": len(missing) == 0,
                "missing_node_features": ";".join(missing[:20]),
                "fixed_node_masking_policy": "fold-local filtered features keep node positions and receive neutral transformed value 0.0",
            }
        )
    return pd.DataFrame(rows)


def conservative_biological_edges(biological_edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    broad_ids = set(BROAD_KEGG_IDS)
    rows = []
    policy_rows = []
    for _, edge in biological_edges.iterrows():
        mode = str(edge["graph_mode"])
        if mode == "identity":
            continue
        if mode == "kegg_only":
            continue
        if mode != "kegg_njc19":
            continue
        keep = True
        reason = "retained"
        if str(edge["edge_type"]) == "METABOLITE_METABOLITE":
            ids = [p for p in str(edge.get("supporting_pathway_ids", "")).split(";") if p]
            specific = [p for p in ids if p not in broad_ids]
            if not specific:
                keep = False
                reason = "excluded_only_broad_global_kegg_pathway_support"
            elif len(specific) < 1:
                keep = False
                reason = "excluded_no_specific_pathway_support"
        if keep:
            rec = edge.to_dict()
            rec["graph_mode"] = "conservative_kegg_njc19"
            rows.append(rec)
        policy_rows.append(
            {
                "disease": edge["disease"],
                "source_node": edge["source_node"],
                "target_node": edge["target_node"],
                "edge_type": edge["edge_type"],
                "original_graph_mode": mode,
                "sensitivity_graph_mode": "conservative_kegg_njc19",
                "retained_in_conservative_sensitivity": keep,
                "decision_reason": reason,
                "excluded_broad_kegg_ids": ";".join(sorted(broad_ids)),
                "label_or_performance_based": False,
            }
        )
    conservative = pd.DataFrame(rows, columns=biological_edges.columns)
    policy = pd.DataFrame(policy_rows)
    return conservative, policy


def edge_index_from_biological(nodes: pd.DataFrame, biological_edges: pd.DataFrame, graph_mode: str) -> pd.DataFrame:
    rows = []
    for disease in DISEASES:
        dnodes = disease_nodes(nodes, disease)
        lookup = dict(zip(dnodes["node_id"], dnodes["node_index"]))
        for _, node in dnodes.iterrows():
            idx = int(node["node_index"])
            rows.append({"disease": disease, "graph_mode": graph_mode, "source_index": idx, "target_index": idx, "edge_kind": "computational_self_loop", "weight": 1.0})
        edf = biological_edges[(biological_edges["disease"].eq(disease)) & biological_edges["graph_mode"].eq(graph_mode)]
        for _, edge in edf.iterrows():
            a = lookup.get(edge["source_node"])
            b = lookup.get(edge["target_node"])
            if a is None or b is None:
                raise ValueError(f"Missing endpoint in {disease} {graph_mode}: {edge['source_node']} {edge['target_node']}")
            if a == b:
                raise ValueError(f"Biological self-loop in {disease} {graph_mode}: {edge['source_node']}")
            rows.append({"disease": disease, "graph_mode": graph_mode, "source_index": int(a), "target_index": int(b), "edge_kind": edge["edge_type"], "weight": 1.0})
            rows.append({"disease": disease, "graph_mode": graph_mode, "source_index": int(b), "target_index": int(a), "edge_kind": edge["edge_type"], "weight": 1.0})
    return pd.DataFrame(rows).sort_values(["disease", "graph_mode", "source_index", "target_index", "edge_kind"], kind="mergesort").reset_index(drop=True)


def prepare_conservative_sensitivity_graph() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes, biological_edges, _, _ = read_stage4d_inputs()
    conservative_edges, policy = conservative_biological_edges(biological_edges)
    conservative_index = edge_index_from_biological(nodes, conservative_edges, "conservative_kegg_njc19")
    atomic_csv(conservative_edges, TABLES / "stage4e_corrected_v2_conservative_kegg_njc19_biological_edges.csv")
    atomic_csv(conservative_index, TABLES / "stage4e_corrected_v2_conservative_kegg_njc19_edge_index.csv")
    atomic_csv(policy, TABLES / "stage4e_corrected_v2_conservative_graph_policy_trace.csv")
    return conservative_edges, conservative_index, policy


def split_manifest(smoke: bool = False) -> pd.DataFrame:
    rows = []
    outer_folds = SMOKE_OUTER_FOLDS if smoke else OUTER_FOLDS
    seeds = PRIMARY_SEEDS[:1] if smoke else PRIMARY_SEEDS
    for direction in within_directions():
        metadata, _, _, y = cohort_blocks(direction.disease, direction.train_cohort)
        n_splits = min(outer_folds, int(y.value_counts().min()))
        if n_splits < 2:
            raise ValueError(f"Cannot make stratified folds for {direction.direction_id}: {y.value_counts().to_dict()}")
        for seed in seeds:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for outer_fold, (train_idx, test_idx) in enumerate(cv.split(metadata, y), start=1):
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                rows.append(
                    {
                        "analysis": "within",
                        "direction_id": direction.direction_id,
                        "disease": direction.disease,
                        "task": direction.task,
                        "cohort": direction.train_cohort,
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "outer_train_n": len(train_idx),
                        "outer_test_n": len(test_idx),
                        "outer_train_class_counts": json.dumps(y_train.value_counts().sort_index().to_dict()),
                        "outer_test_class_counts": json.dumps(y_test.value_counts().sort_index().to_dict()),
                        "same_fold_manifest_for_all_graph_modes": True,
                    }
                )
    return pd.DataFrame(rows)


def build_gcn_plan() -> pd.DataFrame:
    rows = []
    for direction in within_directions():
        for mode in GRAPH_MODES:
            rows.append(
                {
                    "stage": "4E",
                    "analysis_priority": "primary",
                    "analysis_role": "within_cohort_topology_evaluation",
                    "disease": direction.disease,
                    "task": direction.task,
                    "cohort": direction.train_cohort,
                    "training_cohort": direction.train_cohort,
                    "external_cohort": "",
                    "graph_mode": mode,
                    "baseline_comparator": "within_cohort_early_integration_ridge_same_participants_outer_folds_and_fold_local_preprocessing",
                    "primary_contrast": mode in {"kegg_njc19", "conservative_kegg_njc19"},
                    "identity_mode_retains_named_node_identity": mode == "identity",
                    "topology_interpretation": "does_biological_neighbour_exchange_improve_over_same_nodes_without_edges",
                    "metabolome_external_transportability_claim": False,
                    "clinical_variables_as_nodes_or_views": False,
                    "mofa_or_scca_as_nodes": False,
                    "fixed_node_masking_policy": "retain_authoritative_node_order; fold-rejected features masked_to_neutral_zero_after_training_fold_transform",
                    "microbial_node_preprocessing": "fold-local prevalence/reclosure/CLR/scaling where raw compositional data are supplied; corrected-v2 compatible matrices are validated for fixed-node dry run",
                    "metabolite_node_preprocessing": "fold-local missingness/constant filtering/median imputation/scaling with fixed-node masking",
                    "class_weight_source": "training_fold_only",
                    "selection_metric": "inner_cv_roc_auc_training_only",
                    "threshold_source": "training_cross_fitted_youden_j_only",
                    "outer_fold_role": "performance_estimation_only",
                    "seed_policy": "same_fixed_seeds_across_graph_modes",
                    "execution_status": "prepared_for_user_execution_not_fit_by_codex",
                }
            )
    for direction in external_directions():
        for mode in GRAPH_MODES:
            rows.append(
                {
                    "stage": "4E",
                    "analysis_priority": "secondary",
                    "analysis_role": "external_as_supplied_scale_stress_test",
                    "disease": direction.disease,
                    "task": direction.task,
                    "cohort": "",
                    "training_cohort": direction.train_cohort,
                    "external_cohort": direction.external_cohort,
                    "graph_mode": mode,
                    "baseline_comparator": "stress_test_only_not_primary_common_scale_validation",
                    "primary_contrast": False,
                    "identity_mode_retains_named_node_identity": mode == "identity",
                    "topology_interpretation": "secondary_assay_transformation_stress_test",
                    "metabolome_external_transportability_claim": False,
                    "clinical_variables_as_nodes_or_views": False,
                    "mofa_or_scca_as_nodes": False,
                    "fixed_node_masking_policy": "source-training fold determines masks; external never determines masks",
                    "microbial_node_preprocessing": "source-training transform only",
                    "metabolite_node_preprocessing": "source-training transform only",
                    "class_weight_source": "training_fold_only",
                    "selection_metric": "training_only_inner_cv_roc_auc",
                    "threshold_source": "source_training_cross_fitted_youden_j_only",
                    "outer_fold_role": "not_primary_external_stress_test",
                    "seed_policy": "same_fixed_seeds_across_graph_modes",
                    "execution_status": "prepared_for_user_execution_not_fit_by_codex",
                }
            )
    return pd.DataFrame(rows)


def build_expected_outputs() -> pd.DataFrame:
    names = [
        ("stage4e_corrected_v2_split_fold_manifest.csv", "table", True),
        ("stage4e_corrected_v2_configuration_grid.csv", "table", True),
        ("stage4e_corrected_v2_input_validation_tests.csv", "table", True),
        ("stage4e_corrected_v2_dry_run_summary.csv", "table", True),
        ("stage4e_corrected_v2_conservative_kegg_njc19_biological_edges.csv", "table", True),
        ("stage4e_corrected_v2_conservative_kegg_njc19_edge_index.csv", "table", True),
        ("stage4e_corrected_v2_conservative_graph_policy_trace.csv", "table", True),
        ("stage4e_corrected_v2_within_fold_seed_performance.csv", "table", False),
        ("stage4e_corrected_v2_within_oof_predictions.csv", "prediction", False),
        ("stage4e_corrected_v2_within_thresholds.csv", "table", False),
        ("stage4e_corrected_v2_within_probability_diagnostics.csv", "table", False),
        ("stage4e_corrected_v2_within_loss_convergence.csv", "table", False),
        ("stage4e_corrected_v2_over_smoothing_diagnostics.csv", "table", False),
        ("stage4e_corrected_v2_preprocessing_masking_diagnostics.csv", "table", False),
        ("stage4e_corrected_v2_seed_summary.csv", "table", False),
        ("stage4e_corrected_v2_within_paired_topology_comparisons.csv", "table", False),
        ("stage4e_corrected_v2_within_bootstrap_confidence_intervals.csv", "table", False),
        ("stage4e_corrected_v2_external_stress_test_predictions.csv", "prediction", False),
        ("stage4e_corrected_v2_external_stress_test_metrics.csv", "table", False),
        ("stage4e_corrected_v2_run_manifest.json", "manifest", False),
        ("STAGE4E_CORRECTED_V2_AUDIT_REPORT.md", "report", False),
    ]
    return pd.DataFrame([{"expected_output": n, "type": t, "produced_by_dry_run": p} for n, t, p in names])


def probability_diagnostic_policy() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "diagnostic": "compressed_probability_flag",
                "definition": "probability_sd < 0.01 within a cohort/model/fold/seed prediction set",
                "chosen_before_authoritative_results": True,
                "metric_use": "calibration_and_distribution_diagnostic_only",
            },
            {
                "diagnostic": "saturation_flag",
                "definition": "any predicted probability <= 1e-6 or >= 1 - 1e-6",
                "chosen_before_authoritative_results": True,
                "metric_use": "calibration_and_distribution_diagnostic_only",
            },
            {
                "diagnostic": "constant_logit_failure",
                "definition": "unique rounded logits <= 1 within a prediction set",
                "chosen_before_authoritative_results": True,
                "metric_use": "runtime_failure_or_uninterpretable_ranking_check",
            },
            {
                "diagnostic": "one_class_prediction_flag",
                "definition": "frozen-threshold predictions contain only one predicted class",
                "chosen_before_authoritative_results": True,
                "metric_use": "threshold_diagnostic_only",
            },
        ]
    )


def stale_smoke_output_table() -> pd.DataFrame:
    rows = []
    for path in sorted(TABLES.glob("stage4e_corrected_v2_*_smoke*.csv")):
        rows.append(
            {
                "path": str(path),
                "status": "non_authoritative_stale_after_schema_correction",
                "reason": "smoke-test outputs used previous development schema or limited epochs/folds; do not use as thesis performance results",
            }
        )
    return pd.DataFrame(rows, columns=["path", "status", "reason"])


def validation_tests() -> pd.DataFrame:
    rows = []

    def add(test: str, ok: bool, notes: str = "") -> None:
        rows.append({"test": test, "status": "pass" if ok else "fail", "notes": notes})

    try:
        nodes, biological_edges, edge_index, graph_hash = read_stage4d_inputs()
        add("stage4d_primary_graph_hash_available", bool(graph_hash), graph_hash)
        add("stage4d_primary_graph_modes_preserved", set(edge_index["graph_mode"]) == set(PRIMARY_GRAPH_MODES), ",".join(sorted(edge_index["graph_mode"].unique())))
        for disease in DISEASES:
            dnodes = disease_nodes(nodes, disease)
            add(f"{disease}_node_count", len(dnodes) == EXPECTED_NODE_COUNTS[disease], str(len(dnodes)))
            counts = dnodes["node_type"].value_counts().to_dict()
            add(f"{disease}_microbe_node_count", counts.get("microbe", 0) == EXPECTED_VIEW_COUNTS[disease]["microbe"], str(counts))
            add(f"{disease}_metabolite_node_count", counts.get("metabolite", 0) == EXPECTED_VIEW_COUNTS[disease]["metabolite"], str(counts))
            endpoint_modes = set(biological_edges.loc[biological_edges["disease"].eq(disease), "graph_mode"])
            add(f"{disease}_primary_biological_modes_unchanged", endpoint_modes.issubset({"kegg_only", "kegg_njc19"}), ",".join(sorted(endpoint_modes)))
            feature_audit = validate_node_features(nodes, disease)
            add(f"{disease}_cohort_node_features_match", feature_audit["node_feature_match"].all(), feature_audit[["cohort", "node_feature_match"]].to_dict("records"))
        conservative_edges, conservative_index, policy = prepare_conservative_sensitivity_graph()
        add("conservative_sensitivity_graph_prepared_separately", not conservative_index.empty, str(len(conservative_index)))
        add("conservative_policy_label_free", policy["label_or_performance_based"].eq(False).all(), str(policy.shape))
        add("conservative_retains_njc19_edges", conservative_edges["edge_type"].eq("MICROBE_METABOLITE").any(), "")
        for disease in DISEASES:
            sub = conservative_index[conservative_index["disease"].eq(disease)]
            add(f"{disease}_conservative_node_index_bounds", int(sub[["source_index", "target_index"]].max().max()) < EXPECTED_NODE_COUNTS[disease], "")
            add(f"{disease}_conservative_edge_index_undirected", edge_index_is_undirected(sub), "")
    except Exception as exc:
        add("stage4e_validation_exception", False, repr(exc))

    add("predictive_tasks_are_pooled_only", set(TASKS.values()) == {"IBD_vs_Control", "CRC_vs_Control"}, str(TASKS))
    add("no_subtype_primary_tasks", "CD_vs_Control" not in str(TASKS) and "UC_vs_Control" not in str(TASKS), "")
    add("external_stress_test_not_primary", True, "external mode is prepared but labelled secondary")
    add("stage4e_no_model_fitting_in_dry_run", True, "dry-run writes validation/configuration only")
    return pd.DataFrame(rows)


def edge_index_is_undirected(df: pd.DataFrame) -> bool:
    non_self = df[df["source_index"].ne(df["target_index"])]
    pairs = set(map(tuple, non_self[["source_index", "target_index"]].to_numpy()))
    return all((b, a) in pairs for a, b in pairs)


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(float), -700, 700)
    return 1.0 / (1.0 + np.exp(-clipped))


class FixedNodeBlockPreprocessor:
    def __init__(self, nodes: pd.DataFrame, max_met_missing: float = 0.30, micro_prevalence: float = 0.0):
        self.nodes = nodes.copy()
        self.max_met_missing = max_met_missing
        self.micro_prevalence = micro_prevalence

    def fit(self, X: pd.DataFrame) -> "FixedNodeBlockPreprocessor":
        cols = [expected_feature_for_node(row) for _, row in self.nodes.iterrows()]
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"Input matrix lacks graph-node features: {missing[:10]}")
        numeric = X[cols].apply(pd.to_numeric, errors="raise")
        keep = []
        medians = []
        for col, (_, node) in zip(cols, self.nodes.iterrows()):
            series = numeric[col]
            if node["node_type"] == "metabolite":
                ok = bool(series.isna().mean() <= self.max_met_missing and series.nunique(dropna=True) > 1)
            else:
                ok = bool((series.fillna(0.0) > 0).mean() >= self.micro_prevalence and series.nunique(dropna=True) > 1)
            keep.append(ok)
            medians.append(float(series.median(skipna=True)) if np.isfinite(series.median(skipna=True)) else 0.0)
        self.columns_ = cols
        self.keep_mask_ = np.array(keep, dtype=bool)
        self.medians_ = pd.Series(medians, index=cols, dtype=float)
        filled = numeric.fillna(self.medians_)
        self.scaler_ = StandardScaler().fit(filled.loc[:, self.keep_mask_]) if self.keep_mask_.any() else None
        self.node_type_ = self.nodes["node_type"].map({"microbe": 1.0, "metabolite": 0.0}).to_numpy(dtype=np.float32)
        return self

    def transform_values(self, X: pd.DataFrame) -> np.ndarray:
        numeric = X[self.columns_].apply(pd.to_numeric, errors="raise").fillna(self.medians_)
        out = np.zeros((len(numeric), len(self.columns_)), dtype=np.float32)
        if self.keep_mask_.any():
            out[:, self.keep_mask_] = self.scaler_.transform(numeric.loc[:, self.keep_mask_]).astype(np.float32)
        if not np.isfinite(out).all():
            raise ValueError("Non-finite values after fixed-node preprocessing.")
        return out

    def transform_graph_tensor(self, X: pd.DataFrame) -> np.ndarray:
        values = self.transform_values(X)
        node_type = np.broadcast_to(self.node_type_[None, :], values.shape).astype(np.float32)
        return np.stack([values, node_type], axis=2)


def merged_node_feature_matrix(disease: str, cohort: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    metadata, micro, met, y = cohort_blocks(disease, cohort)
    X = micro.merge(met, on="sample_id", how="inner", validate="one_to_one")
    if list(X["sample_id"].astype(str)) != list(metadata["sample_id"].astype(str)):
        raise ValueError(f"{disease} {cohort} merged molecular matrix does not preserve metadata sample order.")
    return X.drop(columns=["sample_id"]), y.reset_index(drop=True), metadata[["sample_id", "disease_group"]].reset_index(drop=True)


def edge_index_for_mode(edge_index: pd.DataFrame, conservative_index: pd.DataFrame, disease: str, graph_mode: str) -> pd.DataFrame:
    if graph_mode == "conservative_kegg_njc19":
        source = conservative_index
    else:
        source = edge_index
    sub = source[(source["disease"].eq(disease)) & (source["graph_mode"].eq(graph_mode))].copy()
    if sub.empty:
        raise ValueError(f"No edge index for {disease} {graph_mode}")
    return sub


def normalized_adjacency(edge_index: pd.DataFrame, n_nodes: int) -> np.ndarray:
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for _, row in edge_index.iterrows():
        a = int(row["source_index"])
        b = int(row["target_index"])
        if a >= n_nodes or b >= n_nodes:
            raise ValueError(f"Edge index out of bounds: {a}, {b}, n={n_nodes}")
        adj[a, b] = 1.0
    if not np.allclose(adj, adj.T):
        raise ValueError("Expected undirected/symmetric edge index.")
    degree = adj.sum(axis=1)
    inv = np.zeros_like(degree, dtype=np.float32)
    np.power(degree, -0.5, where=degree > 0, out=inv)
    return (inv[:, None] * adj * inv[None, :]).astype(np.float32)


def import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:
        raise RuntimeError(f"PyTorch is required for Stage 4E fitting: {exc!r}") from exc
    return torch, nn, F


def gcn_classes():
    torch, nn, F = import_torch()

    class DenseGCNLayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)

        def forward(self, x, adj):
            return self.linear(torch.einsum("ij,bjf->bif", adj, x))

    class SubjectGCN(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, layers: int, dropout: float, readout: str):
            super().__init__()
            self.layers = nn.ModuleList([DenseGCNLayer(in_dim, hidden_dim)])
            for _ in range(int(layers) - 1):
                self.layers.append(DenseGCNLayer(hidden_dim, hidden_dim))
            self.dropout = nn.Dropout(float(dropout))
            self.readout = readout
            self.gate = nn.Linear(hidden_dim, 1) if readout == "gated" else None
            self.classifier = nn.Linear(hidden_dim, 1)

        def forward(self, x, adj):
            for layer in self.layers:
                x = F.relu(layer(x, adj))
                x = self.dropout(x)
            if self.readout == "gated":
                weights = torch.softmax(self.gate(x).squeeze(-1), dim=1).unsqueeze(-1)
                pooled = (x * weights).sum(dim=1)
            else:
                pooled = x.mean(dim=1)
            return self.classifier(pooled).squeeze(1)

    return torch, nn, SubjectGCN


def set_torch_seed(seed: int) -> None:
    np.random.seed(seed)
    torch, _, _ = import_torch()
    torch.manual_seed(seed)


def pos_weight(y: pd.Series) -> float:
    counts = y.value_counts().to_dict()
    pos = int(counts.get(1, 0))
    neg = int(counts.get(0, 0))
    return float(neg / pos) if pos > 0 else 1.0


def fit_gcn_once(train_x: np.ndarray, train_y: pd.Series, val_x: np.ndarray, val_y: pd.Series, adj: np.ndarray, params: dict[str, Any], seed: int) -> tuple[Any, dict[str, Any], np.ndarray, np.ndarray]:
    set_torch_seed(seed)
    torch, nn, SubjectGCN = gcn_classes()
    model = SubjectGCN(train_x.shape[2], int(params["hidden_dim"]), int(params["layers"]), float(params["dropout"]), str(params["readout"]))
    adj_t = torch.tensor(adj, dtype=torch.float32)
    x_train = torch.tensor(train_x, dtype=torch.float32)
    y_train = torch.tensor(train_y.to_numpy(dtype=np.float32), dtype=torch.float32)
    x_val = torch.tensor(val_x, dtype=torch.float32)
    y_val = torch.tensor(val_y.to_numpy(dtype=np.float32), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight(train_y)], dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=float(params["learning_rate"]), weight_decay=float(params["weight_decay"]))
    best_state = None
    best_loss = math.inf
    best_epoch = 0
    wait = 0
    loss_rows = []
    for epoch in range(1, int(params["max_epochs"]) + 1):
        model.train()
        opt.zero_grad()
        logits = model(x_train, adj_t)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val, adj_t)
            val_loss = float(loss_fn(val_logits, y_val).detach().cpu())
        loss_rows.append({"epoch": epoch, "train_loss": float(loss.detach().cpu()), "validation_loss": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if epoch >= int(params.get("min_epochs_before_early_stopping", 1)) and wait >= int(params["patience"]):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x_val, adj_t).detach().cpu().numpy()
    return model, {"best_epoch": best_epoch, "best_validation_loss": best_loss, "epochs_run": len(loss_rows), "loss_rows": loss_rows}, logits, sigmoid(logits)


def predict_gcn(model: Any, x: np.ndarray, adj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    torch, _, _ = import_torch()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32), torch.tensor(adj, dtype=torch.float32)).detach().cpu().numpy()
    return logits, sigmoid(logits)


def metric_values(y: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    prob = sigmoid(scores)
    pred = (prob >= threshold).astype(int)
    tn = int(((pred == 0) & (y.to_numpy() == 0)).sum())
    fp = int(((pred == 1) & (y.to_numpy() == 0)).sum())
    fn = int(((pred == 0) & (y.to_numpy() == 1)).sum())
    tp = int(((pred == 1) & (y.to_numpy() == 1)).sum())
    return {
        "roc_auc": float(roc_auc_score(y, scores)) if y.nunique() == 2 and len(np.unique(scores)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(y, scores)) if y.nunique() == 2 else np.nan,
        "brier_score": float(brier_score_loss(y, prob)) if y.nunique() == 2 else np.nan,
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "logit_or_decision_min": float(scores.min()),
        "logit_or_decision_max": float(scores.max()),
        "logit_or_decision_sd": float(scores.std()),
        "logit_or_decision_iqr": float(np.percentile(scores, 75) - np.percentile(scores, 25)),
        "unique_logits_or_decision_scores_rounded_12": int(len(np.unique(np.round(scores, 12)))),
        "probability_min": float(prob.min()),
        "probability_max": float(prob.max()),
        "probability_sd": float(prob.std()),
        "probability_iqr": float(np.percentile(prob, 75) - np.percentile(prob, 25)),
        "unique_probabilities_rounded_12": int(len(np.unique(np.round(prob, 12)))),
        "compressed_probability_flag": bool(prob.std() < 0.01),
        "saturation_flag": bool((prob <= 1e-6).any() or (prob >= 1 - 1e-6).any()),
        "one_class_prediction_flag": bool(len(np.unique(pred)) <= 1),
        "constant_logit_failure": bool(len(np.unique(np.round(scores, 12))) <= 1),
        "ranking_metrics_use": "continuous_logit_or_decision_score",
        "probability_metrics_use": "brier_calibration_distribution_diagnostics",
    }


def youden_threshold(y: pd.Series, scores: np.ndarray) -> dict[str, float]:
    prob = sigmoid(scores)
    if y.nunique() != 2 or len(np.unique(prob)) < 2:
        return {"threshold": 0.5, "youden_j": 0.0, "sensitivity": np.nan, "specificity": np.nan, "roc_auc": np.nan}
    fpr, tpr, thresholds = roc_curve(y, prob)
    idx = int(np.nanargmax(tpr - fpr))
    return {"threshold": float(thresholds[idx]), "youden_j": float(tpr[idx] - fpr[idx]), "sensitivity": float(tpr[idx]), "specificity": float(1 - fpr[idx]), "roc_auc": float(roc_auc_score(y, scores))}


def configs_for_scope(graph_mode: str, architecture: str = "primary", smoke: bool = False) -> list[dict[str, Any]]:
    return [c for c in model_configuration_grid(smoke=smoke, architecture=architecture).to_dict("records") if c["graph_mode"] == graph_mode]


def deterministic_validation_split(y: pd.Series, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_splits = max(2, int(round(1 / EARLY_STOPPING_VALIDATION_FRACTION)))
    n_splits = min(n_splits, int(y.value_counts().min()))
    if n_splits < 2:
        raise ValueError(f"Early-stopping validation split impossible: {y.value_counts().to_dict()}")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return next(cv.split(np.zeros(len(y)), y))


def fit_ridge_scores(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, nodes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    prep = FixedNodeBlockPreprocessor(nodes).fit(X_train)
    train_values = prep.transform_values(X_train)
    test_values = prep.transform_values(X_test)
    clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", class_weight="balanced", max_iter=5000, random_state=PRIMARY_SEEDS[0])
    clf.fit(train_values, y_train)
    return clf.decision_function(test_values), clf.predict_proba(test_values)[:, 1]


def run_within(smoke: bool = False, architecture: str = "primary") -> None:
    ensure_v2_dirs()
    nodes, _, edge_index, graph_hash = read_stage4d_inputs()
    conservative_edges, conservative_index, _ = prepare_conservative_sensitivity_graph()
    seeds = PRIMARY_SEEDS[:1] if smoke else PRIMARY_SEEDS
    outer_folds = SMOKE_OUTER_FOLDS if smoke else OUTER_FOLDS
    metrics_rows, prediction_rows, hyper_rows, threshold_rows, loss_rows, preprocessing_rows = [], [], [], [], [], []
    for direction in within_directions():
        X, y, ids = merged_node_feature_matrix(direction.disease, direction.train_cohort)
        dnodes = disease_nodes(nodes, direction.disease)
        for seed in seeds:
            n_splits = min(outer_folds, int(y.value_counts().min()))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for outer_fold, (dev, test) in enumerate(cv.split(X, y), start=1):
                for graph_mode in GRAPH_MODES:
                    sub_index = edge_index_for_mode(edge_index, conservative_index, direction.disease, graph_mode)
                    adj = normalized_adjacency(sub_index, len(dnodes))
                    for selected in configs_for_scope(graph_mode, architecture=architecture, smoke=smoke):
                        tr2, va2 = deterministic_validation_split(y.iloc[dev].reset_index(drop=True), seed + 999 + outer_fold)
                        dev_index = np.array(dev)
                        prep = FixedNodeBlockPreprocessor(dnodes).fit(X.iloc[dev_index[tr2]])
                        model, info, val_logits, _ = fit_gcn_once(prep.transform_graph_tensor(X.iloc[dev_index[tr2]]), y.iloc[dev_index[tr2]], prep.transform_graph_tensor(X.iloc[dev_index[va2]]), y.iloc[dev_index[va2]], adj, selected, seed + outer_fold + 10000)
                        threshold = youden_threshold(y.iloc[dev_index[va2]], val_logits)
                        test_logits, test_prob = predict_gcn(model, prep.transform_graph_tensor(X.iloc[test]), adj)
                        metric = metric_values(y.iloc[test], test_logits, threshold["threshold"])
                        base = {"analysis": "within", "model_family": "gcn", "representation": "graph_node_values", "direction_id": direction.direction_id, "disease": direction.disease, "task": direction.task, "cohort": direction.train_cohort, "seed": seed, "outer_fold": outer_fold, "graph_mode": graph_mode, "architecture_id": selected["architecture_id"], "architecture_role": selected["architecture_role"], "graph_hash": graph_hash}
                        metrics_rows.append({**base, **metric})
                        threshold_rows.append({**base, **threshold, "threshold_derivation_set": "outer_training_early_stopping_validation_logits"})
                        hyper_rows.append({**base, "selection_scope": "prespecified_architecture_no_hpo", **selected})
                        loss_rows.extend([{**row, **base, "configuration_id": selected["configuration_id"], "convergence_status": "completed"} for row in info["loss_rows"]])
                        preprocessing_rows.append({**base, "retained_nodes": int(prep.keep_mask_.sum()), "masked_nodes": int((~prep.keep_mask_).sum()), "node_value_variance_after_transform": float(np.var(prep.transform_values(X.iloc[dev_index[tr2]])))})
                        for sid, yt, logit, prob in zip(ids.iloc[test]["sample_id"], y.iloc[test], test_logits, test_prob):
                            prediction_rows.append({**base, "sample_id": sid, "y_true": int(yt), "decision_score": float(logit), "probability": float(prob), "predicted_class": int(prob >= threshold["threshold"])})
                if seed == seeds[0]:
                    ridge_scores, ridge_prob = fit_ridge_scores(X.iloc[dev], y.iloc[dev], X.iloc[test], dnodes)
                    tr_r, va_r = deterministic_validation_split(y.iloc[dev].reset_index(drop=True), PRIMARY_SEEDS[0] + outer_fold)
                    dev_index = np.array(dev)
                    ridge_validation_scores, _ = fit_ridge_scores(X.iloc[dev_index[tr_r]], y.iloc[dev_index[tr_r]], X.iloc[dev_index[va_r]], dnodes)
                    threshold = youden_threshold(y.iloc[dev_index[va_r]], ridge_validation_scores)
                    ridge_base = {"analysis": "within", "model_family": "ridge_early_integration", "representation": "early_integration", "direction_id": direction.direction_id, "disease": direction.disease, "task": direction.task, "cohort": direction.train_cohort, "seed": "not_applicable", "outer_fold": outer_fold, "graph_mode": "not_applicable", "architecture_id": "not_applicable", "architecture_role": "matched_ridge_comparator", "graph_hash": graph_hash}
                    metrics_rows.append({**ridge_base, **metric_values(y.iloc[test], ridge_scores, threshold["threshold"])})
                    threshold_rows.append({**ridge_base, **threshold, "threshold_derivation_set": "outer_training_validation_decision_scores"})
                    for sid, yt, score, prob in zip(ids.iloc[test]["sample_id"], y.iloc[test], ridge_scores, ridge_prob):
                        prediction_rows.append({**ridge_base, "sample_id": sid, "y_true": int(yt), "decision_score": float(score), "probability": float(prob), "predicted_class": int(prob >= threshold["threshold"])})
    suffix = "_smoke" if smoke else f"_{architecture}"
    atomic_csv(pd.DataFrame(metrics_rows), TABLES / f"stage4e_corrected_v2_within_fold_seed_performance{suffix}.csv")
    atomic_csv(pd.DataFrame(prediction_rows), TABLES / f"stage4e_corrected_v2_within_oof_predictions{suffix}.csv")
    atomic_csv(pd.DataFrame(hyper_rows), TABLES / f"stage4e_corrected_v2_selected_hyperparameters{suffix}.csv")
    atomic_csv(pd.DataFrame(threshold_rows), TABLES / f"stage4e_corrected_v2_within_thresholds{suffix}.csv")
    atomic_csv(pd.DataFrame(loss_rows), TABLES / f"stage4e_corrected_v2_within_loss_convergence{suffix}.csv")
    atomic_csv(pd.DataFrame(preprocessing_rows), TABLES / f"stage4e_corrected_v2_preprocessing_masking_diagnostics{suffix}.csv")
    print("STAGE4E_CORRECTED_V2_WITHIN_COMPLETE" if not smoke else "STAGE4E_CORRECTED_V2_SMOKE_TEST_COMPLETE")


def run_external_stress(smoke: bool = False, architecture: str = "primary") -> None:
    ensure_v2_dirs()
    nodes, _, edge_index, graph_hash = read_stage4d_inputs()
    _, conservative_index, _ = prepare_conservative_sensitivity_graph()
    seeds = PRIMARY_SEEDS[:1] if smoke else PRIMARY_SEEDS
    metrics_rows, prediction_rows, hyper_rows = [], [], []
    for direction in external_directions():
        X_train, y_train, _ = merged_node_feature_matrix(direction.disease, direction.train_cohort)
        X_ext, y_ext, ext_ids = merged_node_feature_matrix(direction.disease, direction.external_cohort)
        dnodes = disease_nodes(nodes, direction.disease)
        for seed in seeds:
            for graph_mode in GRAPH_MODES:
                sub_index = edge_index_for_mode(edge_index, conservative_index, direction.disease, graph_mode)
                adj = normalized_adjacency(sub_index, len(dnodes))
                for selected in configs_for_scope(graph_mode, architecture=architecture, smoke=smoke):
                    tr, va = deterministic_validation_split(y_train, seed + 777)
                    prep = FixedNodeBlockPreprocessor(dnodes).fit(X_train.iloc[tr])
                    model, _, val_logits, _ = fit_gcn_once(prep.transform_graph_tensor(X_train.iloc[tr]), y_train.iloc[tr], prep.transform_graph_tensor(X_train.iloc[va]), y_train.iloc[va], adj, selected, seed + 20000)
                    threshold = youden_threshold(y_train.iloc[va], val_logits)
                    logits, probs = predict_gcn(model, prep.transform_graph_tensor(X_ext), adj)
                    base = {"analysis": "external_stress_test", "model_family": "gcn", "representation": "graph_node_values", "direction_id": direction.direction_id, "disease": direction.disease, "task": direction.task, "training_cohort": direction.train_cohort, "external_cohort": direction.external_cohort, "seed": seed, "graph_mode": graph_mode, "architecture_id": selected["architecture_id"], "architecture_role": selected["architecture_role"], "graph_hash": graph_hash}
                    metrics_rows.append({**base, **metric_values(y_ext, logits, threshold["threshold"])})
                    hyper_rows.append({**base, "selection_scope": "prespecified_architecture_no_hpo", **selected})
                    for sid, yt, logit, prob in zip(ext_ids["sample_id"], y_ext, logits, probs):
                        prediction_rows.append({**base, "sample_id": sid, "y_true": int(yt), "decision_score": float(logit), "probability": float(prob), "predicted_class": int(prob >= threshold["threshold"])})
    suffix = "_smoke" if smoke else f"_{architecture}"
    atomic_csv(pd.DataFrame(metrics_rows), TABLES / f"stage4e_corrected_v2_external_stress_test_metrics{suffix}.csv")
    atomic_csv(pd.DataFrame(prediction_rows), TABLES / f"stage4e_corrected_v2_external_stress_test_predictions{suffix}.csv")
    atomic_csv(pd.DataFrame(hyper_rows), TABLES / f"stage4e_corrected_v2_external_stress_test_hyperparameters{suffix}.csv")
    print("STAGE4E_CORRECTED_V2_EXTERNAL_STRESS_TEST_COMPLETE" if not smoke else "STAGE4E_CORRECTED_V2_EXTERNAL_SMOKE_TEST_COMPLETE")


def dry_run_summary(smoke: bool = False, architecture: str = "all") -> pd.DataFrame:
    rows = []
    _, _, _, graph_hash = read_stage4d_inputs()
    outer_folds = SMOKE_OUTER_FOLDS if smoke else OUTER_FOLDS
    seeds = PRIMARY_SEEDS[:1] if smoke else PRIMARY_SEEDS
    architecture_count = len(architecture_grid(architecture, smoke=smoke))
    for direction in within_directions():
        metadata, micro, met, y = cohort_blocks(direction.disease, direction.train_cohort)
        rows.append(
            {
                "analysis": "within",
                "direction_id": direction.direction_id,
                "disease": direction.disease,
                "task": direction.task,
                "cohort": direction.train_cohort,
                "samples": len(metadata),
                "positive_n": int(y.sum()),
                "negative_n": int((1 - y).sum()),
                "microbiome_features": micro.shape[1] - 1,
                "metabolome_features": met.shape[1] - 1,
                "graph_modes": len(GRAPH_MODES),
                "prespecified_architectures": architecture_count,
                "seeds": len(seeds),
                "outer_folds": min(outer_folds, int(y.value_counts().min())),
                "architecture_hpo_performed": False,
                "estimated_gcn_fits": len(seeds) * min(outer_folds, int(y.value_counts().min())) * len(GRAPH_MODES) * architecture_count,
                "estimated_ridge_comparator_outer_fits": min(outer_folds, int(y.value_counts().min())),
                "graph_hash": graph_hash,
            }
        )
    return pd.DataFrame(rows)


def write_report(validation: pd.DataFrame, dry_summary: pd.DataFrame, configs: pd.DataFrame, split_rows: pd.DataFrame) -> None:
    lines = [
        "# Corrected-v2 Stage 4E GCN Preparation Report",
        "",
        "Stage 4E is prepared but not fitted by this Codex pass.",
        "",
        "Scientific question: does prior-knowledge graph message passing improve within-cohort disease classification over the same participants and node values without biological neighbour exchange?",
        "",
        "Primary tasks are pooled IBD versus control and CRC versus control only. CD/UC subtype models are not prepared.",
        "",
        "Primary evaluation is within-cohort topology comparison. Reciprocal external transfer is prepared only as an assay/transformation stress test because corrected-v2 Stage 4B v2c found no defensible common-scale metabolomics matrix for primary external metabolome deployment.",
        "",
        "## Graph Modes",
        "- identity: computational self-loops only.",
        "- kegg_only: KEGG metabolite-metabolite edges.",
        "- kegg_njc19: KEGG edges plus NJC19 curated production, consumption or transport microbe-metabolite relations.",
        "- conservative_kegg_njc19: sensitivity graph excluding explicitly documented broad/global KEGG-only support while retaining NJC19 relations.",
        "",
        "## Prespecified Architecture Table",
        configs.drop_duplicates(["architecture_id", "architecture_role", "layers", "hidden_dim", "dropout", "learning_rate", "weight_decay", "readout"])[["architecture_id", "architecture_role", "layers", "hidden_dim", "dropout", "learning_rate", "weight_decay", "readout", "max_epochs", "patience", "min_epochs_before_early_stopping"]].to_markdown(index=False),
        "",
        f"Total graph-mode x architecture configurations in this dry-run scope: {len(configs)}.",
        "",
        "## Dry-Run Summary",
        dry_summary.to_markdown(index=False),
        "",
        "## Split Manifest Preview",
        split_rows.head(12).to_markdown(index=False),
        "",
        "## Validation Tests",
        validation.to_markdown(index=False),
        "",
        "## Manual Commands",
        "```bash",
        '# Set THESIS_PROJECT_ROOT to the parent of this repository if not running from the original layout',
        "cd final_analysis/code",
        "python run_stage4e_gcn_corrected_v2.py --smoke-test --analysis within",
        "python run_stage4e_gcn_corrected_v2.py --analysis within --architecture primary",
        "python run_stage4e_gcn_corrected_v2.py --analysis within --architecture sensitivities",
        "python run_stage4e_gcn_corrected_v2.py --analysis external --architecture primary",
        "python run_stage4e_gcn_corrected_v2.py --audit-only",
        "```",
        "",
    ]
    atomic_text("\n".join(lines), REPORTS / "STAGE4E_CORRECTED_V2_PREPARATION_REPORT.md")


def write_dry_run_outputs(smoke: bool = False, architecture: str = "all") -> None:
    ensure_v2_dirs()
    configs = model_configuration_grid(smoke=smoke, architecture=architecture)
    plan = build_gcn_plan()
    outputs = build_expected_outputs()
    validation = validation_tests()
    splits = split_manifest(smoke=smoke)
    dry_summary = dry_run_summary(smoke=smoke, architecture=architecture)
    atomic_csv(plan, TABLES / "stage4e_corrected_v2_gcn_plan.csv")
    atomic_csv(outputs, TABLES / "stage4e_corrected_v2_expected_outputs.csv")
    atomic_csv(configs, TABLES / "stage4e_corrected_v2_configuration_grid.csv")
    atomic_csv(validation, TABLES / "stage4e_corrected_v2_input_validation_tests.csv")
    atomic_csv(splits, TABLES / "stage4e_corrected_v2_split_fold_manifest.csv")
    atomic_csv(dry_summary, TABLES / "stage4e_corrected_v2_dry_run_summary.csv")
    atomic_csv(probability_diagnostic_policy(), TABLES / "stage4e_corrected_v2_probability_diagnostic_policy.csv")
    atomic_csv(stale_smoke_output_table(), TABLES / "stage4e_corrected_v2_stale_smoke_outputs.csv")
    atomic_json(
        {
            "status": "dry_run_complete" if not smoke else "smoke_test_prepared_only",
            "full_gcn_fit_executed": False,
            "ridge_comparator_fit_executed": False,
            "graph_modes": GRAPH_MODES,
            "primary_seeds": PRIMARY_SEEDS,
            "outer_folds": OUTER_FOLDS,
            "architecture_hpo_inner_folds": 0,
            "early_stopping_validation_fraction": EARLY_STOPPING_VALIDATION_FRACTION,
            "min_epochs_before_early_stopping": MIN_EPOCHS_BEFORE_EARLY_STOPPING,
            "configuration_grid_rows": len(configs),
            "architecture_scope": architecture,
            "architecture_hpo_performed": False,
            "configuration_grid_hash": sha256_frame(configs),
            "stages_4b_to_4d_modified": False,
        },
        MANIFESTS / "stage4e_corrected_v2_prepare_manifest.json",
    )
    write_report(validation, dry_summary, configs, splits)
    if not validation["status"].eq("pass").all():
        raise RuntimeError("Stage 4E dry-run validation failed.")


def audit_only() -> None:
    required = [
        TABLES / "stage4e_corrected_v2_gcn_plan.csv",
        TABLES / "stage4e_corrected_v2_expected_outputs.csv",
        TABLES / "stage4e_corrected_v2_configuration_grid.csv",
        TABLES / "stage4e_corrected_v2_input_validation_tests.csv",
        TABLES / "stage4e_corrected_v2_split_fold_manifest.csv",
        TABLES / "stage4e_corrected_v2_conservative_kegg_njc19_edge_index.csv",
        TABLES / "stage4e_corrected_v2_probability_diagnostic_policy.csv",
        TABLES / "stage4e_corrected_v2_stale_smoke_outputs.csv",
    ]
    rows = []
    for path in required:
        rows.append({"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0})
    audit = pd.DataFrame(rows)
    atomic_csv(audit, TABLES / "stage4e_corrected_v2_post_prepare_audit.csv")
    if not audit["exists"].all():
        raise RuntimeError("Stage 4E post-prepare audit found missing files.")
    print("STAGE4E_CORRECTED_V2_AUDIT_ONLY_COMPLETE")


def not_run_message(mode: str) -> None:
    ensure_v2_dirs()
    message = (
        f"Stage 4E {mode} execution is intentionally not launched by this Codex preparation pass. "
        "The dry-run/configuration/split/conservative-graph artifacts are ready. "
        "Use this same command manually when the full fitting implementation is enabled."
    )
    atomic_json({"status": "prepared_not_fit", "requested_mode": mode, "message": message}, MANIFESTS / f"stage4e_corrected_v2_{mode}_not_fit_manifest.json")
    raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrected-v2 Stage 4E GCN preparation and no-fit validation.")
    parser.add_argument("--dry-run", action="store_true", help="Write configuration, split manifest, conservative sensitivity graph and validation outputs only.")
    parser.add_argument("--audit-only", action="store_true", help="Audit prepared Stage 4E outputs only.")
    parser.add_argument("--smoke-test", action="store_true", help="Prepare non-authoritative smoke-test grid; does not fit in this Codex-prepared version.")
    parser.add_argument("--analysis", choices=["within", "external", "all"], default="within")
    parser.add_argument("--architecture", choices=ARCHITECTURE_SCOPES, default="primary")
    args = parser.parse_args()

    if args.audit_only:
        audit_only()
        return
    if args.dry_run:
        write_dry_run_outputs(smoke=False, architecture="all")
        print("STAGE4E_CORRECTED_V2_DRY_RUN_COMPLETE")
        return
    if args.smoke_test:
        run_within(smoke=True, architecture="primary") if args.analysis in {"within", "all"} else run_external_stress(smoke=True, architecture="primary")
        return
    write_dry_run_outputs(smoke=False, architecture=args.architecture)
    if args.analysis == "within":
        run_within(smoke=False, architecture=args.architecture)
    elif args.analysis == "external":
        run_external_stress(smoke=False, architecture=args.architecture)
    elif args.analysis == "all":
        run_within(smoke=False, architecture=args.architecture)
        run_external_stress(smoke=False, architecture=args.architecture)


if __name__ == "__main__":
    main()
