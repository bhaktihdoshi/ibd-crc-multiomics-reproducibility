from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibd_pipeline_transformers import (  # noqa: E402
    BalancedGradientBoostingClassifier,
    ConstantFeatureFilter,
    MissingnessFilter,
)


def test_balanced_gradient_boosting_classifier_sklearn_contract(tmp_path):
    x = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1, 0]})
    y = np.array([0, 0, 0, 1, 1, 1])
    clf = BalancedGradientBoostingClassifier(random_state=42, n_estimators=5, max_depth=1)
    assert is_classifier(clf)
    cloned = clone(clf)
    assert cloned.get_params()["n_estimators"] == 5
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    search = GridSearchCV(clf, {"n_estimators": [5], "max_depth": [1]}, scoring="roc_auc", cv=cv)
    search.fit(x, y)
    fitted = search.best_estimator_
    proba = fitted.predict_proba(x)
    assert proba.shape == (6, 2)
    assert list(fitted.classes_) == [0, 1]
    assert fitted.feature_importances_ is not None
    path = tmp_path / "balanced_gb.joblib"
    joblib.dump(fitted, path)
    loaded = joblib.load(path)
    assert loaded.predict_proba(x).shape == (6, 2)


def test_balanced_gradient_boosting_weights_are_fit_local():
    x = pd.DataFrame({"a": [0, 1, 2, 3], "b": [3, 2, 1, 0]})
    imbalanced = BalancedGradientBoostingClassifier(random_state=42, n_estimators=5, max_depth=1).fit(x, [0, 0, 0, 1])
    balanced = BalancedGradientBoostingClassifier(random_state=42, n_estimators=5, max_depth=1).fit(x, [0, 0, 1, 1])
    assert not np.allclose(imbalanced.fit_sample_weight_, balanced.fit_sample_weight_)


def test_filters_raise_on_real_non_numeric_columns():
    x = pd.DataFrame({"numeric": [1, 2, 3], "bad": ["a", "b", "c"]})
    for filt in [MissingnessFilter(), ConstantFeatureFilter()]:
        try:
            filt.fit(x)
        except ValueError as exc:
            assert "received non-numeric column" in str(exc)
        else:
            raise AssertionError(f"{type(filt).__name__} should reject non-numeric columns")


def test_filters_accept_numeric_strings_and_all_missing_columns():
    x = pd.DataFrame({"numeric_string": ["1.0", "2.0", "3.0"], "all_missing": [np.nan, np.nan, np.nan]})
    miss = MissingnessFilter(max_missing=1.0).fit(x)
    const = ConstantFeatureFilter().fit(x)
    assert "numeric_string" in set(miss.retained_feature_names_)
    assert "numeric_string" in set(const.retained_feature_names_)
