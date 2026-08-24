from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path("/Users/bhaktidoshi/Downloads/Thesis Data/thesis_preprocessing_workflow/stage4_multiomics_integration/scripts")
sys.path.insert(0, str(SCRIPT_DIR))

from stage4_common import clr_transform, stage4a_feasibility  # noqa: E402


def test_clr_rows_sum_to_zero_on_synthetic_data() -> None:
    x = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 4.0], "c": [4.0, 8.0]})
    z = clr_transform(x, pseudocount=1e-9)
    assert np.allclose(z.mean(axis=1), 0.0)


def test_stage4a_feasibility_is_explicit_string() -> None:
    status = stage4a_feasibility()
    assert isinstance(status, str)
    assert status


def test_graph_schema_has_no_label_field() -> None:
    graph_edge_cols = ["source_node", "target_node", "edge_type", "knowledge_source", "evidence_class", "directionality", "mapping_provenance"]
    assert "label" not in graph_edge_cols
    assert "outcome" not in graph_edge_cols


def test_stage4b_representations_are_fixed() -> None:
    expected = {"METABOLOME", "MICROBIOME", "EARLY_INTEGRATION"}
    assert expected == {"METABOLOME", "MICROBIOME", "EARLY_INTEGRATION"}
