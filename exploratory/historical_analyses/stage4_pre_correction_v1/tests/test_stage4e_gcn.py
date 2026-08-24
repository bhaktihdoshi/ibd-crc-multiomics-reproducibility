from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_DIR = Path("/Users/bhaktidoshi/Downloads/Thesis Data/thesis_preprocessing_workflow/stage4_multiomics_integration/scripts")
sys.path.insert(0, str(SCRIPT_DIR))

from run_stage4e_gcn import (  # noqa: E402
    AUGMENTED_GRAPH_HASH,
    FoldLocalGraphPreprocessor,
    adjacency_matrix,
    column_for_node,
    dependency_status,
    read_graph,
    set_seeds,
    torch_model_classes,
)


def test_fixed_graph_hash_verified() -> None:
    nodes, edges = read_graph()
    assert not nodes.empty
    assert not edges.empty
    assert AUGMENTED_GRAPH_HASH == "71a0dec685c445e1059875736265e4755fb6cade434f586a567ffe63cf9e6ee1"


def test_node_order_aligns_to_subject_features() -> None:
    nodes = pd.DataFrame([
        {"node_id": "microbe::Bacteroides", "node_type": "microbe"},
        {"node_id": "metabolite::HMDB0000122", "node_type": "metabolite"},
    ])
    assert [column_for_node(row) for _, row in nodes.iterrows()] == ["mic__Bacteroides", "met__HMDB0000122"]
    X = pd.DataFrame({"mic__Bacteroides": [1.0, 2.0], "met__HMDB0000122": [np.nan, 3.0]})
    prep = FoldLocalGraphPreprocessor().fit(X, nodes)
    arr = prep.transform(X)
    assert arr.shape == (2, 2, 2)
    assert np.isfinite(arr).all()


def test_self_loops_are_computational_only() -> None:
    nodes = pd.DataFrame([
        {"node_id": "microbe::A", "node_type": "microbe"},
        {"node_id": "metabolite::B", "node_type": "metabolite"},
    ])
    edges = pd.DataFrame([{"source_node": "microbe::A", "target_node": "metabolite::B", "disease_space": "IBD"}])
    ident = adjacency_matrix(nodes, edges, "IBD", "identity")
    aug = adjacency_matrix(nodes, edges, "IBD", "augmented")
    assert np.allclose(ident, np.eye(2))
    assert aug[0, 1] > 0


def test_no_gat_path_declared() -> None:
    assert dependency_status()["gat_path"] == "not_implemented"


def test_tiny_synthetic_forward_backward_if_torch_available() -> None:
    if dependency_status()["torch"].startswith("unavailable"):
        pytest.skip("torch unavailable")
    set_seeds()
    torch, nn, SubjectGCN = torch_model_classes()
    model = SubjectGCN(in_dim=2, hidden_dim=4, layers=1, dropout=0.0)
    x = torch.randn(3, 5, 2)
    adj = torch.eye(5)
    y = torch.tensor([0.0, 1.0, 1.0])
    logits = model(x, adj)
    loss = nn.BCEWithLogitsLoss()(logits, y)
    loss.backward()
    assert logits.shape == (3,)
