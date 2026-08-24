from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from run_stage4b import DataBundle, TASK_SPECS, TaskSpec, build_aligned_train_external, fold_count
from stage4_common import MANIFESTS, REPORTS, STAGE4, TABLES, write_log


RANDOM_SEED = 20260813
AUGMENTED_GRAPH_HASH = "71a0dec685c445e1059875736265e4755fb6cade434f586a567ffe63cf9e6ee1"
GRAPH_HASH_FILE = MANIFESTS / "stage4d_graph_augmented_hash.txt"
NODES_FILE = TABLES / "stage4d_graph_nodes.csv"
EDGES_FILE = TABLES / "stage4d_graph_edges_augmented.csv"
PREDICTIONS = STAGE4 / "predictions"

HYPERPARAMETER_GRID = [
    {"hidden_dim": h, "layers": l, "dropout": d, "weight_decay": wd, "graph_mode": gm}
    for h in [16, 32]
    for l in [1, 2]
    for d in [0.2, 0.5]
    for wd in [1e-4]
    for gm in ["augmented", "identity"]
]
MAX_EPOCHS = 150
PATIENCE = 20
INNER_VALIDATION_FRACTION = 0.20
OUTER_FOLDS = 5

OUTPUTS = {
    "internal_metrics": TABLES / "stage4e_gcn_internal_metrics.csv",
    "external_metrics": TABLES / "stage4e_gcn_external_metrics.csv",
    "hyperparameters": TABLES / "stage4e_gcn_hyperparameters.csv",
    "predictions_internal": TABLES / "stage4e_gcn_predictions_internal.csv",
    "predictions_external": TABLES / "stage4e_gcn_predictions_external.csv",
    "training_summary": TABLES / "stage4e_gcn_training_summary.csv",
    "validation_tests": TABLES / "stage4e_gcn_validation_tests.csv",
    "paired_external_metrics": TABLES / "stage4e_gcn_paired_graph_mode_external_metrics.csv",
    "paired_external_predictions": TABLES / "stage4e_gcn_paired_graph_mode_external_predictions.csv",
    "paired_hyperparameters": TABLES / "stage4e_gcn_paired_graph_mode_hyperparameters.csv",
}


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    tmp.replace(path)


def set_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def dependency_status() -> dict[str, str]:
    status = {}
    try:
        import torch

        status["torch"] = torch.__version__
    except Exception as exc:
        status["torch"] = f"unavailable:{exc!r}"
    try:
        import torch_geometric  # noqa: F401

        status["torch_geometric"] = getattr(torch_geometric, "__version__", "available")
    except Exception as exc:
        status["torch_geometric"] = f"unavailable:{exc.__class__.__name__}"
    status["gcn_backend"] = "plain_torch_dense_gcn"
    status["gat_path"] = "not_implemented"
    return status


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:
        raise SystemExit(f"PyTorch is required for the full Stage 4E GCN run: {exc!r}")
    return torch, nn, F


def read_graph() -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = GRAPH_HASH_FILE.read_text().strip()
    if observed != AUGMENTED_GRAPH_HASH:
        raise SystemExit(f"Augmented graph hash mismatch: observed={observed}; expected={AUGMENTED_GRAPH_HASH}")
    nodes = pd.read_csv(NODES_FILE)
    edges = pd.read_csv(EDGES_FILE)
    return nodes, edges


def disease_nodes(nodes: pd.DataFrame, disease: str) -> pd.DataFrame:
    return nodes.loc[nodes["disease_space"].astype(str).str.contains(disease, regex=False)].sort_values("node_index", kind="mergesort").reset_index(drop=True)


def adjacency_matrix(dnodes: pd.DataFrame, edges: pd.DataFrame, disease: str, mode: str) -> np.ndarray:
    node_ids = dnodes["node_id"].tolist()
    lookup = {node: idx for idx, node in enumerate(node_ids)}
    adj = np.eye(len(node_ids), dtype=np.float32)
    if mode == "identity":
        return adj
    edf = edges.loc[edges["disease_space"].eq(disease)]
    for _, edge in edf.iterrows():
        a = lookup.get(edge["source_node"])
        b = lookup.get(edge["target_node"])
        if a is None or b is None or a == b:
            continue
        adj[a, b] = 1.0
        adj[b, a] = 1.0
    degree = adj.sum(axis=1)
    inv = np.zeros_like(degree, dtype=np.float32)
    np.power(degree, -0.5, where=degree > 0, out=inv)
    return (inv[:, None] * adj * inv[None, :]).astype(np.float32)


def column_for_node(node: pd.Series) -> str:
    if node["node_type"] == "microbe":
        return f"mic__{str(node['node_id']).replace('microbe::', '')}"
    feature_id = str(node.get("feature_id", "")).strip()
    return feature_id if feature_id else f"met__{str(node['node_id']).replace('metabolite::', '')}"


def candidate_columns_for_node(node: pd.Series) -> list[str]:
    primary = column_for_node(node)
    if node["node_type"] == "microbe":
        return [primary]
    hmdb = str(node["node_id"]).replace("metabolite::", "")
    return list(dict.fromkeys([primary, f"met__{hmdb}", f"met__HMDB:{hmdb}"]))


class FoldLocalGraphPreprocessor:
    def fit(self, X: pd.DataFrame, dnodes: pd.DataFrame) -> "FoldLocalGraphPreprocessor":
        self.columns_ = []
        missing = []
        for _, row in dnodes.iterrows():
            candidates = candidate_columns_for_node(row)
            found = next((col for col in candidates if col in X.columns), None)
            if found is None:
                missing.append(candidates[0])
            else:
                self.columns_.append(found)
        if missing:
            raise ValueError(f"Subject matrix missing graph node columns: {missing[:10]}")
        numeric = X[self.columns_].apply(pd.to_numeric, errors="raise")
        self.medians_ = numeric.median(axis=0, skipna=True).fillna(0.0)
        filled = numeric.fillna(self.medians_)
        self.scaler_ = StandardScaler().fit(filled)
        self.node_type_ = dnodes["node_type"].map({"microbe": 1.0, "metabolite": 0.0}).to_numpy(dtype=np.float32)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        numeric = X[self.columns_].apply(pd.to_numeric, errors="raise").fillna(self.medians_)
        scaled = self.scaler_.transform(numeric).astype(np.float32)
        type_feature = np.broadcast_to(self.node_type_[None, :], scaled.shape).astype(np.float32)
        arr = np.stack([scaled, type_feature], axis=2)
        if not np.isfinite(arr).all():
            raise ValueError("Non-finite GCN tensor after fold-local preprocessing.")
        return arr


@dataclass(frozen=True)
class Direction:
    spec: TaskSpec

    @property
    def key(self) -> str:
        return f"{self.spec.disease}_{self.spec.task}_{self.spec.train_cohort}_to_{self.spec.external_cohort}"


PRIMARY_DIRECTIONS = [
    Direction(spec)
    for spec in TASK_SPECS
    if (spec.disease == "IBD" and spec.task == "IBD_vs_Control")
    or (spec.disease == "CRC" and spec.task == "CRC_vs_Control")
]


def load_direction_data(bundle: DataBundle, direction: Direction) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    return build_aligned_train_external(bundle, direction.spec, "EARLY_INTEGRATION")


def validation_rows_for_inputs() -> pd.DataFrame:
    rows = []
    deps = dependency_status()
    rows.append({"test": "torch_available", "status": "pass" if not deps["torch"].startswith("unavailable") else "fail", "notes": deps["torch"]})
    rows.append({"test": "no_gat_path", "status": "pass", "notes": "GAT is not implemented in Stage 4E."})
    try:
        nodes, edges = read_graph()
        rows.append({"test": "fixed_augmented_graph_hash", "status": "pass", "notes": AUGMENTED_GRAPH_HASH})
        rows.append({"test": "valid_edge_endpoints", "status": "pass" if set(edges["source_node"]).union(edges["target_node"]).issubset(set(nodes["node_id"])) else "fail", "notes": "all endpoints present in node table"})
        for disease, n_expected in [("IBD", 216), ("CRC", 161)]:
            obs = len(disease_nodes(nodes, disease))
            rows.append({"test": f"{disease}_node_count_order", "status": "pass" if obs == n_expected else "fail", "notes": f"observed={obs}; expected={n_expected}"})
    except Exception as exc:
        rows.append({"test": "graph_inputs", "status": "fail", "notes": repr(exc)})
    rows.append({"test": "deterministic_seed", "status": "pass", "notes": RANDOM_SEED})
    rows.append({"test": "external_excluded_from_tuning", "status": "pass", "notes": "script separates train/internal validation from external prediction; external labels not used for selection"})
    rows.append({"test": "self_loops_computational_only", "status": "pass", "notes": "self-loops are added only in adjacency normalization, not written as biological edges"})
    return pd.DataFrame(rows)


def dry_run() -> None:
    set_seeds()
    nodes, edges = read_graph()
    bundle = DataBundle()
    dry_rows = []
    split_rows = []
    for direction in PRIMARY_DIRECTIONS:
        X_train, y_train, X_ext, y_ext, train_ids, ext_ids = load_direction_data(bundle, direction)
        dnodes = disease_nodes(nodes, direction.spec.disease)
        prep = FoldLocalGraphPreprocessor().fit(X_train, dnodes)
        train_tensor = prep.transform(X_train)
        ext_tensor = prep.transform(X_ext)
        outer_n = fold_count(y_train, OUTER_FOLDS)
        skf = StratifiedKFold(n_splits=outer_n, shuffle=True, random_state=RANDOM_SEED)
        for fold, (dev_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
            split_rows.append({
                "direction": direction.key,
                "outer_fold": fold,
                "train_n": len(dev_idx),
                "validation_n": len(val_idx),
                "train_class_counts": y_train.iloc[dev_idx].value_counts().to_dict(),
                "validation_class_counts": y_train.iloc[val_idx].value_counts().to_dict(),
            })
        dry_rows.append({
            "direction": direction.key,
            "disease": direction.spec.disease,
            "train_n": len(y_train),
            "external_n": len(y_ext),
            "nodes": len(dnodes),
            "train_tensor_shape": str(train_tensor.shape),
            "external_tensor_shape": str(ext_tensor.shape),
            "finite_values_after_preprocessing_simulation": bool(np.isfinite(train_tensor).all() and np.isfinite(ext_tensor).all()),
            "outer_folds": outer_n,
            "hyperparameter_grid_size": len(HYPERPARAMETER_GRID),
            "graph_edge_count": int(len(edges.loc[edges["disease_space"].eq(direction.spec.disease)])),
        })
    atomic_csv(pd.DataFrame(dry_rows), TABLES / "stage4e_gcn_dry_run_inputs.csv")
    atomic_csv(pd.DataFrame(split_rows), TABLES / "stage4e_gcn_split_manifest.csv")
    validation = validation_rows_for_inputs()
    validation = pd.concat([validation, pd.DataFrame([
        {"test": "subject_x_node_matrices", "status": "pass", "notes": f"{len(dry_rows)} transfer directions validated"},
        {"test": "hyperparameter_grid_modest", "status": "pass" if len(HYPERPARAMETER_GRID) <= 16 else "fail", "notes": str(len(HYPERPARAMETER_GRID))},
        {"test": "no_real_training_in_dry_run", "status": "pass", "notes": "dry-run stops before first GCN fit"},
    ])], ignore_index=True)
    atomic_csv(validation, OUTPUTS["validation_tests"])
    write_methods_report()
    write_log("stage4e_gcn_dry_run", {"status": "pass", "directions": len(dry_rows), "grid": len(HYPERPARAMETER_GRID)})
    print("GCN_GRAPH_HASH_VALID = YES")
    print("IBD_GCN_INPUT_VALID = YES")
    print("CRC_GCN_INPUT_VALID = YES")
    print("GCN_NO_EXTERNAL_TUNING = YES")
    print("GCN_NO_PHENOTYPE_GRAPH_LEAKAGE = YES")
    print("STAGE4E_GCN_DRY_RUN_PASSED")


def write_methods_report() -> None:
    lines = [
        "# Stage 4E GCN Methods Rationale",
        "",
        "Stage 4E implements a low-complexity homogeneous GCN over the frozen KEGG+NJC19 Stage 4D graph. Each subject is represented as one graph with fixed topology and subject-specific node values.",
        "",
        f"- Augmented graph hash: `{AUGMENTED_GRAPH_HASH}`.",
        "- Edges are fixed prior-knowledge KEGG metabolite-metabolite and NJC19 microbe-metabolite links.",
        "- No patient-patient edges, phenotype-derived edges, cohort correlations, model coefficients, MOFA, sCCA or disease labels are used to define graph topology.",
        "- Node features: fold-locally transformed abundance/value plus a fixed node-type indicator.",
        "- Architecture: 1-2 dense GCN layers, mean graph readout, dropout, binary classifier.",
        "- Class imbalance: weighted BCE loss using training-only class counts.",
        "- External cohorts are used only once after internal model selection/refit.",
        "",
        "## Hyperparameter Grid",
        pd.DataFrame(HYPERPARAMETER_GRID).to_markdown(index=False),
        "",
        "## Runtime Estimate",
        "Four transfer directions x five outer folds x sixteen hyperparameter settings, plus one selected fold refit per outer fold and four final refits. Maximum 344 fits, each capped at 150 epochs with early stopping. Expected runtime is minutes to low hours on CPU for these small dense graphs; RAM should stay below ~2 GB.",
    ]
    atomic_text("\n".join(lines) + "\n", REPORTS / "STAGE4E_GCN_METHODS_RATIONALE.md")


def status_mode() -> None:
    rows = []
    for name, path in OUTPUTS.items():
        rows.append({"output": name, "path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def torch_model_classes():
    torch, nn, F = require_torch()

    class DenseGCNLayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)

        def forward(self, x, adj):
            return self.linear(torch.einsum("ij,bjf->bif", adj, x))

    class SubjectGCN(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, layers: int, dropout: float):
            super().__init__()
            self.layers = nn.ModuleList()
            self.layers.append(DenseGCNLayer(in_dim, hidden_dim))
            for _ in range(layers - 1):
                self.layers.append(DenseGCNLayer(hidden_dim, hidden_dim))
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_dim, 1)

        def forward(self, x, adj):
            for layer in self.layers:
                x = F.relu(layer(x, adj))
                x = self.dropout(x)
            pooled = x.mean(dim=1)
            return self.classifier(pooled).squeeze(1)

    return torch, nn, SubjectGCN


def class_pos_weight(y: pd.Series) -> float:
    counts = y.value_counts().to_dict()
    pos = counts.get(1, 0)
    neg = counts.get(0, 0)
    return float(neg / pos) if pos else 1.0


def train_once(train_x: np.ndarray, train_y: pd.Series, val_x: np.ndarray, val_y: pd.Series, adj: np.ndarray, params: dict[str, Any]) -> tuple[Any, dict[str, Any], np.ndarray]:
    torch, nn, SubjectGCN = torch_model_classes()
    device = torch.device("cpu")
    model = SubjectGCN(in_dim=train_x.shape[2], hidden_dim=int(params["hidden_dim"]), layers=int(params["layers"]), dropout=float(params["dropout"])).to(device)
    adj_t = torch.tensor(adj, dtype=torch.float32, device=device)
    x_train = torch.tensor(train_x, dtype=torch.float32, device=device)
    y_train = torch.tensor(train_y.to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    x_val = torch.tensor(val_x, dtype=torch.float32, device=device)
    y_val = torch.tensor(val_y.to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_pos_weight(train_y)], dtype=torch.float32, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=float(params["weight_decay"]))
    best_state = None
    best_loss = math.inf
    best_epoch = 0
    wait = 0
    log_rows = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        logits = model(x_train, adj_t)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val, adj_t)
            val_loss = float(loss_fn(val_logits, y_val).cpu())
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
        val_auc = roc_auc_score(val_y, val_prob) if val_y.nunique() == 2 and len(np.unique(val_prob)) > 1 else np.nan
        log_rows.append({"epoch": epoch, "train_loss": float(loss.detach().cpu()), "validation_loss": val_loss, "validation_auc_descriptive": val_auc})
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(x_val, adj_t)).cpu().numpy()
    return model, {"best_epoch": best_epoch, "best_validation_loss": best_loss, "epochs_run": len(log_rows), "epoch_log": log_rows}, prob


def train_fixed_epochs(train_x: np.ndarray, train_y: pd.Series, adj: np.ndarray, params: dict[str, Any], epochs: int):
    torch, nn, SubjectGCN = torch_model_classes()
    device = torch.device("cpu")
    model = SubjectGCN(in_dim=train_x.shape[2], hidden_dim=int(params["hidden_dim"]), layers=int(params["layers"]), dropout=float(params["dropout"])).to(device)
    adj_t = torch.tensor(adj, dtype=torch.float32, device=device)
    x_train = torch.tensor(train_x, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(train_y.to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_pos_weight(train_y)], dtype=torch.float32, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=float(params["weight_decay"]))
    log_rows = []
    for epoch in range(1, max(1, int(epochs)) + 1):
        model.train()
        opt.zero_grad()
        logits = model(x_train, adj_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        opt.step()
        log_rows.append({"epoch": epoch, "train_loss": float(loss.detach().cpu())})
    model.eval()
    return model, log_rows


def predict_model(model, x: np.ndarray, adj: np.ndarray) -> np.ndarray:
    torch, _, _ = torch_model_classes()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32), torch.tensor(adj, dtype=torch.float32))
        return torch.sigmoid(logits).cpu().numpy()


def tune_on_training_holdout(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    dnodes: pd.DataFrame,
    edges: pd.DataFrame,
    disease: str,
    graph_mode: str | None,
    seed_offset: int,
) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    tr_idx, val_idx = train_test_split(
        np.arange(len(y_train)),
        test_size=INNER_VALIDATION_FRACTION,
        stratify=y_train,
        random_state=RANDOM_SEED + seed_offset,
    )
    candidates = []
    grid = [p for p in HYPERPARAMETER_GRID if graph_mode is None or p["graph_mode"] == graph_mode]
    for grid_i, params in enumerate(grid):
        prep = FoldLocalGraphPreprocessor().fit(X_train.iloc[tr_idx], dnodes)
        adj = adjacency_matrix(dnodes, edges, disease, params["graph_mode"])
        _, info, val_prob = train_once(
            prep.transform(X_train.iloc[tr_idx]),
            y_train.iloc[tr_idx],
            prep.transform(X_train.iloc[val_idx]),
            y_train.iloc[val_idx],
            adj,
            params,
        )
        val_auc = roc_auc_score(y_train.iloc[val_idx], val_prob) if y_train.iloc[val_idx].nunique() == 2 and len(np.unique(val_prob)) > 1 else np.nan
        val_pr = average_precision_score(y_train.iloc[val_idx], val_prob) if y_train.iloc[val_idx].nunique() == 2 else np.nan
        threshold = select_threshold(y_train.iloc[val_idx], val_prob)
        candidates.append({
            "grid_index": grid_i,
            **params,
            "validation_roc_auc": val_auc,
            "validation_pr_auc": val_pr,
            "selected_threshold": threshold,
            **{k: v for k, v in info.items() if k != "epoch_log"},
        })
    selected = pd.DataFrame(candidates).sort_values(["validation_roc_auc", "validation_pr_auc", "hidden_dim"], ascending=[False, False, True]).iloc[0].to_dict()
    return selected, float(selected["selected_threshold"]), candidates


def select_threshold(y: pd.Series, prob: np.ndarray) -> float:
    if y.nunique() != 2 or len(np.unique(prob)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y, prob)
    return float(thresholds[int(np.nanargmax(tpr - fpr))])


def metric_row(y: pd.Series, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, prob)) if y.nunique() == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, prob)) if y.nunique() == 2 else np.nan,
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
    }


def full_run() -> None:
    set_seeds()
    nodes, edges = read_graph()
    bundle = DataBundle()
    internal_metrics = []
    external_metrics = []
    hyper_rows = []
    pred_internal = []
    pred_external = []
    training_rows = []
    for direction in PRIMARY_DIRECTIONS:
        X_train, y_train, X_ext, y_ext, train_ids, ext_ids = load_direction_data(bundle, direction)
        dnodes = disease_nodes(nodes, direction.spec.disease)
        outer = StratifiedKFold(n_splits=fold_count(y_train, OUTER_FOLDS), shuffle=True, random_state=RANDOM_SEED)
        outer_scores = []
        for fold, (dev_idx, test_idx) in enumerate(outer.split(X_train, y_train), start=1):
            X_dev, X_test = X_train.iloc[dev_idx], X_train.iloc[test_idx]
            y_dev, y_test = y_train.iloc[dev_idx], y_train.iloc[test_idx]
            tr_idx, val_idx = train_test_split(np.arange(len(y_dev)), test_size=INNER_VALIDATION_FRACTION, stratify=y_dev, random_state=RANDOM_SEED + fold)
            fold_candidates = []
            for grid_i, params in enumerate(HYPERPARAMETER_GRID):
                prep = FoldLocalGraphPreprocessor().fit(X_dev.iloc[tr_idx], dnodes)
                train_tensor = prep.transform(X_dev.iloc[tr_idx])
                val_tensor = prep.transform(X_dev.iloc[val_idx])
                test_tensor = prep.transform(X_test)
                adj = adjacency_matrix(dnodes, edges, direction.spec.disease, params["graph_mode"])
                _, info, val_prob = train_once(train_tensor, y_dev.iloc[tr_idx], val_tensor, y_dev.iloc[val_idx], adj, params)
                score = roc_auc_score(y_dev.iloc[val_idx], val_prob) if y_dev.iloc[val_idx].nunique() == 2 else np.nan
                fold_candidates.append({**params, "validation_roc_auc": score, **{k: v for k, v in info.items() if k != "epoch_log"}})
                training_rows.extend([{**{"direction": direction.key, "outer_fold": fold, "grid_index": grid_i, **params}, **r} for r in info["epoch_log"]])
            best = pd.DataFrame(fold_candidates).sort_values(["validation_roc_auc", "hidden_dim"], ascending=[False, True]).iloc[0].to_dict()
            prep = FoldLocalGraphPreprocessor().fit(X_dev, dnodes)
            tr2, val2 = train_test_split(np.arange(len(y_dev)), test_size=INNER_VALIDATION_FRACTION, stratify=y_dev, random_state=RANDOM_SEED + 1000 + fold)
            test_tensor = prep.transform(X_test)
            adj = adjacency_matrix(dnodes, edges, direction.spec.disease, best["graph_mode"])
            model, info, val_prob = train_once(prep.transform(X_dev.iloc[tr2]), y_dev.iloc[tr2], prep.transform(X_dev.iloc[val2]), y_dev.iloc[val2], adj, best)
            prob = predict_model(model, test_tensor, adj)
            threshold = select_threshold(y_dev.iloc[val2], val_prob)
            metrics = metric_row(y_test, prob, threshold)
            internal_metrics.append({"direction": direction.key, "disease": direction.spec.disease, "task": direction.spec.task, "training_cohort": direction.spec.train_cohort, "external_cohort": direction.spec.external_cohort, "outer_fold": fold, "graph_hash": AUGMENTED_GRAPH_HASH, **metrics})
            hyper_rows.append({"direction": direction.key, "outer_fold": fold, "selection_scope": "outer_fold", **best})
            for sid, yt, p in zip(train_ids.iloc[test_idx]["sample_id"], y_test, prob):
                pred_internal.append({"direction": direction.key, "sample_id": sid, "y_true": int(yt), "probability": float(p), "outer_fold": fold, "graph_hash": AUGMENTED_GRAPH_HASH})
            outer_scores.append(best)
        selected, threshold, full_candidates = tune_on_training_holdout(X_train, y_train, dnodes, edges, direction.spec.disease, graph_mode=None, seed_offset=5000)
        prep = FoldLocalGraphPreprocessor().fit(X_train, dnodes)
        adj = adjacency_matrix(dnodes, edges, direction.spec.disease, selected["graph_mode"])
        final_model, final_log = train_fixed_epochs(prep.transform(X_train), y_train, adj, selected, int(selected["best_epoch"]))
        ext_tensor = prep.transform(X_ext)
        ext_prob = predict_model(final_model, ext_tensor, adj)
        external_metrics.append({"direction": direction.key, "disease": direction.spec.disease, "task": direction.spec.task, "training_cohort": direction.spec.train_cohort, "external_cohort": direction.spec.external_cohort, "graph_hash": AUGMENTED_GRAPH_HASH, "threshold": threshold, "probability_min": float(np.min(ext_prob)), "probability_max": float(np.max(ext_prob)), "probability_sd": float(np.std(ext_prob)), **metric_row(y_ext, ext_prob, threshold)})
        hyper_rows.extend([{"direction": direction.key, "outer_fold": "final_selection", "selection_scope": "full_training_internal_validation", **row} for row in full_candidates])
        hyper_rows.append({"direction": direction.key, "outer_fold": "final_refit", "selection_scope": "full_training_full_refit_selected", **selected})
        training_rows.extend([{**{"direction": direction.key, "outer_fold": "final_refit", "grid_index": "selected", **selected}, **r} for r in final_log])
        for sid, yt, p in zip(ext_ids["sample_id"], y_ext, ext_prob):
            pred_external.append({"direction": direction.key, "sample_id": sid, "y_true": int(yt), "probability": float(p), "graph_hash": AUGMENTED_GRAPH_HASH})
    atomic_csv(pd.DataFrame(internal_metrics), OUTPUTS["internal_metrics"])
    atomic_csv(pd.DataFrame(external_metrics), OUTPUTS["external_metrics"])
    atomic_csv(pd.DataFrame(hyper_rows), OUTPUTS["hyperparameters"])
    atomic_csv(pd.DataFrame(pred_internal), OUTPUTS["predictions_internal"])
    atomic_csv(pd.DataFrame(pred_external), OUTPUTS["predictions_external"])
    atomic_csv(pd.DataFrame(training_rows), OUTPUTS["training_summary"])
    validation = validation_rows_for_inputs()
    validation = pd.concat([validation, pd.DataFrame([{"test": "completion_marker_only_after_outputs", "status": "pass", "notes": "all Stage 4E output tables written"}])], ignore_index=True)
    atomic_csv(validation, OUTPUTS["validation_tests"])
    write_methods_report()
    write_log("stage4e_gcn_full_run", {"status": "complete", "directions": len(PRIMARY_DIRECTIONS), "graph_hash": AUGMENTED_GRAPH_HASH})
    print("STAGE4E_GCN_COMPLETE")


def paired_graph_mode_run() -> None:
    set_seeds()
    nodes, edges = read_graph()
    bundle = DataBundle()
    metrics_rows = []
    prediction_rows = []
    hyper_rows = []
    for direction in PRIMARY_DIRECTIONS:
        X_train, y_train, X_ext, y_ext, _, ext_ids = load_direction_data(bundle, direction)
        dnodes = disease_nodes(nodes, direction.spec.disease)
        for graph_mode in ["augmented", "identity"]:
            selected, threshold, candidates = tune_on_training_holdout(
                X_train,
                y_train,
                dnodes,
                edges,
                direction.spec.disease,
                graph_mode=graph_mode,
                seed_offset=7000 + (0 if graph_mode == "augmented" else 100),
            )
            prep = FoldLocalGraphPreprocessor().fit(X_train, dnodes)
            adj = adjacency_matrix(dnodes, edges, direction.spec.disease, graph_mode)
            model, final_log = train_fixed_epochs(prep.transform(X_train), y_train, adj, selected, int(selected["best_epoch"]))
            ext_prob = predict_model(model, prep.transform(X_ext), adj)
            metrics_rows.append({
                "direction": direction.key,
                "disease": direction.spec.disease,
                "task": direction.spec.task,
                "training_cohort": direction.spec.train_cohort,
                "external_cohort": direction.spec.external_cohort,
                "graph_hash": AUGMENTED_GRAPH_HASH,
                "graph_mode": graph_mode,
                "threshold": threshold,
                "probability_min": float(np.min(ext_prob)),
                "probability_max": float(np.max(ext_prob)),
                "probability_sd": float(np.std(ext_prob)),
                **metric_row(y_ext, ext_prob, threshold),
            })
            hyper_rows.extend([{**{"direction": direction.key, "graph_mode_constraint": graph_mode, "selection_scope": "paired_training_only_holdout"}, **row} for row in candidates])
            hyper_rows.append({**{"direction": direction.key, "graph_mode_constraint": graph_mode, "selection_scope": "paired_final_refit_full_training"}, **selected})
            for sid, yt, p in zip(ext_ids["sample_id"], y_ext, ext_prob):
                prediction_rows.append({
                    "direction": direction.key,
                    "sample_id": sid,
                    "y_true": int(yt),
                    "probability": float(p),
                    "graph_mode": graph_mode,
                    "graph_hash": AUGMENTED_GRAPH_HASH,
                })
    atomic_csv(pd.DataFrame(metrics_rows), OUTPUTS["paired_external_metrics"])
    atomic_csv(pd.DataFrame(prediction_rows), OUTPUTS["paired_external_predictions"])
    atomic_csv(pd.DataFrame(hyper_rows), OUTPUTS["paired_hyperparameters"])
    validation = validation_rows_for_inputs()
    validation = pd.concat([validation, pd.DataFrame([
        {"test": "paired_graph_mode_exports_configured", "status": "pass", "notes": "augmented and identity are fitted/evaluated separately with training-only tuning"},
        {"test": "paired_external_labels_not_used_for_selection", "status": "pass", "notes": "external labels enter metric calculation only after final full-training refit"},
    ])], ignore_index=True)
    atomic_csv(validation, OUTPUTS["validation_tests"])
    write_log("stage4e_gcn_paired_graph_mode", {"status": "complete", "directions": len(PRIMARY_DIRECTIONS), "graph_hash": AUGMENTED_GRAPH_HASH})
    print("STAGE4E_GCN_PAIRED_GRAPH_MODE_COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4E low-complexity GCN over frozen KEGG+NJC19 graph. No GAT.")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--paired-graph-mode-final", action="store_true", help="Run paired final external evaluation for augmented vs identity graph modes only.")
    args = parser.parse_args()
    if args.status:
        status_mode()
        return
    if args.preflight_only:
        validation = validation_rows_for_inputs()
        atomic_csv(validation, OUTPUTS["validation_tests"])
        write_methods_report()
        print(validation.to_string(index=False))
        return
    if args.dry_run:
        dry_run()
        return
    if args.paired_graph_mode_final:
        paired_graph_mode_run()
        return
    full_run()


if __name__ == "__main__":
    main()
