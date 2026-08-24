from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
from joblib import dump, load
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crc_pipeline_transformers import ConstantFeatureFilter, InputColumnOrderValidator, Log1pTransformer, MissingnessFilter


def test_crc_transformer_joblib_roundtrip(tmp_path):
    x = pd.DataFrame({
        "met__HMDB:HMDB0000001": [1.0, 2.0, np.nan, 4.0],
        "met__KEGG:C00001": [0.2, 0.2, 0.2, 0.2],
        "met__HMDB:HMDB0000002": [3.0, 4.0, 5.0, 6.0],
    })
    pipe = Pipeline([
        ("column_order", InputColumnOrderValidator()),
        ("log1p", Log1pTransformer(enabled=False)),
        ("missingness", MissingnessFilter(max_missing=0.30)),
        ("constant", ConstantFeatureFilter()),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    pipe.fit(x)
    path = tmp_path / "crc_transformer_roundtrip.joblib"
    dump(pipe, path)
    loaded = load(path)
    transformed = loaded.transform(x)
    assert transformed.shape[0] == x.shape[0]
    assert hasattr(loaded.named_steps["missingness"], "keep_mask_")
    assert hasattr(loaded.named_steps["constant"], "keep_mask_")
    with pytest.raises(ValueError):
        loaded.transform(x[list(reversed(x.columns))])


def test_log1p_transformer_rejects_negative_values():
    x = pd.DataFrame({"met__HMDB:HMDB0000001": [1.0, -0.1, 2.0]})
    transformer = Log1pTransformer(enabled=True)
    with pytest.raises(ValueError):
        transformer.fit(x)


def test_gradient_boosting_weights_are_fold_local_length():
    y_full = pd.Series([0, 0, 0, 0, 1, 1])
    dev_idx = [0, 1, 2, 4]
    y_dev = y_full.iloc[dev_idx]
    weights = compute_sample_weight("balanced", y_dev)
    assert len(weights) == len(y_dev)
    assert len(weights) != len(y_full)
