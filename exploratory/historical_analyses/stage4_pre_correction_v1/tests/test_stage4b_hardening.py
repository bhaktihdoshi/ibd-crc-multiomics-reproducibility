from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone


PROJECT = Path("/Users/bhaktidoshi/Downloads/Thesis Data")
SCRIPT = PROJECT / "thesis_preprocessing_workflow/stage4_multiomics_integration/scripts/run_stage4b.py"


def load_stage4b():
    spec = importlib.util.spec_from_file_location("run_stage4b_test_module", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stage4a_final_gate_resolves_ready():
    module = load_stage4b()
    status = module.resolve_stage4a_final_status()
    assert status["ready"] is True
    assert status["overall_feasibility"] == "FOUR_COHORT_MULTIOMICS_READY"
    assert status["stage4b_ibd"] == "GO"
    assert status["stage4b_crc"] == "GO"


def test_pipeline_objects_clone_without_fit():
    module = load_stage4b()
    for representation in module.REPRESENTATIONS:
        pipe = module.build_pipeline(representation, module.C_GRID[0])
        cloned = clone(pipe)
        assert cloned is not pipe
        assert cloned.named_steps["clf"].penalty == "l2"
        assert cloned.named_steps["clf"].class_weight == "balanced"


def test_microbiome_preprocessor_edge_cases():
    module = load_stage4b()
    x = pd.DataFrame({"a": [0.1, 0.0, 0.3, 0.0], "b": [0.0, 0.0, 0.0, 0.0], "c": [0.2, 0.4, 0.0, 0.1]})
    proc = module.MicrobiomePreprocessor(prevalence_threshold=0.25)
    proc.fit(x)
    out = proc.transform(x)
    assert out.shape[0] == 4
    assert np.isfinite(out).all()
    assert "b" not in set(proc.get_feature_names_out())


def test_metabolome_filters_edge_cases():
    module = load_stage4b()
    x = pd.DataFrame({"a": [1.0, np.nan, 2.0, 3.0], "b": [1.0, 1.0, 1.0, 1.0], "c": [np.nan, np.nan, np.nan, np.nan]})
    miss = module.MissingnessFilter(max_missing_fraction=0.50).fit(x)
    filtered = miss.transform(x)
    assert list(filtered.columns) == ["a", "b"]
    const = module.ConstantFeatureFilter().fit(filtered)
    assert list(const.get_feature_names_out()) == ["a"]


def test_real_task_schema_construction_without_fit():
    module = load_stage4b()
    bundle = module.DataBundle()
    checked = 0
    for spec in module.TASK_SPECS:
        for representation in module.REPRESENTATIONS:
            x_train, y_train, x_ext, y_ext, _, _ = module.build_aligned_train_external(bundle, spec, representation)
            assert x_train.shape[1] == x_ext.shape[1]
            assert list(x_train.columns) == list(x_ext.columns)
            assert y_train.nunique() == 2
            assert y_ext.nunique() == 2
            checked += 1
    assert checked == 24
