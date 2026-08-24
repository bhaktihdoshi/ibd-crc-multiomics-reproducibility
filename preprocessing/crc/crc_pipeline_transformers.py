from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight


class InputColumnOrderValidator(BaseEstimator, TransformerMixin):
    """Record and enforce exact input columns for fitted sklearn pipelines."""

    def fit(self, X, y=None):
        x = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(x.columns.astype(str))
        if len(set(self.feature_names_in_)) != len(self.feature_names_in_):
            duplicated = pd.Series(self.feature_names_in_).loc[pd.Series(self.feature_names_in_).duplicated()].tolist()
            raise ValueError(f"Duplicate input columns detected before fitting: {duplicated}")
        return self

    def transform(self, X):
        x = pd.DataFrame(X)
        observed = list(x.columns.astype(str))
        expected = list(self.feature_names_in_)
        missing = sorted(set(expected) - set(observed))
        additional = sorted(set(observed) - set(expected))
        if missing or additional or observed != expected:
            raise ValueError(
                "Input columns do not match fitted training order. "
                f"Missing={missing}; additional={additional}; "
                "same_set_different_order="
                f"{not missing and not additional and observed != expected}"
            )
        return x

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


class MissingnessFilter(BaseEstimator, TransformerMixin):
    """Drop columns above a training-fold missingness threshold."""

    def __init__(self, max_missing: float = 0.30):
        self.max_missing = max_missing

    def fit(self, X, y=None):
        x = pd.DataFrame(X)
        numeric = x.apply(pd.to_numeric, errors="coerce")
        bad_cols = [str(c) for c in x.columns if numeric[c].isna().all() and not x[c].isna().all()]
        if bad_cols:
            raise ValueError(
                f"MissingnessFilter received non-numeric column(s) {bad_cols}; "
                "encode categorical features before this pipeline step."
            )
        self.feature_names_in_ = np.asarray(x.columns.astype(str))
        self.missing_fraction_ = x.isna().mean(axis=0).to_numpy()
        self.keep_mask_ = self.missing_fraction_ <= self.max_missing
        if not self.keep_mask_.any():
            raise ValueError(
                "MissingnessFilter removed all features. "
                f"Input feature count={x.shape[1]}; max_missing={self.max_missing}; "
                f"minimum observed missingness={float(np.nanmin(self.missing_fraction_)) if len(self.missing_fraction_) else np.nan}."
            )
        self.retained_feature_names_ = self.feature_names_in_[self.keep_mask_]
        return self

    def transform(self, X):
        return np.asarray(pd.DataFrame(X))[:, self.keep_mask_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.retained_feature_names_, dtype=object)


class ConstantFeatureFilter(BaseEstimator, TransformerMixin):
    """Remove features that are constant within the training fold."""

    def fit(self, X, y=None):
        x = pd.DataFrame(X)
        numeric = x.apply(pd.to_numeric, errors="coerce")
        bad_cols = [str(c) for c in x.columns if numeric[c].isna().all() and not x[c].isna().all()]
        if bad_cols:
            raise ValueError(
                f"ConstantFeatureFilter received non-numeric column(s) {bad_cols}; "
                "encode categorical features before this pipeline step."
            )
        self.feature_names_in_ = np.asarray(x.columns.astype(str))
        self.n_unique_nonmissing_ = numeric.nunique(dropna=True).to_numpy()
        self.keep_mask_ = self.n_unique_nonmissing_ > 1
        if not self.keep_mask_.any():
            raise ValueError(
                "ConstantFeatureFilter removed all features. "
                f"Input feature count={x.shape[1]}; all columns have <=1 unique non-missing value."
            )
        self.retained_feature_names_ = self.feature_names_in_[self.keep_mask_]
        return self

    def transform(self, X):
        return np.asarray(pd.DataFrame(X))[:, self.keep_mask_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.retained_feature_names_, dtype=object)


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """Optional sensitivity-only log1p transform for non-negative values."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def fit(self, X, y=None):
        x = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(x.columns.astype(str))
        if self.enabled and (x.apply(pd.to_numeric, errors="coerce") < 0).any().any():
            raise ValueError("Log1pTransformer cannot be fitted to negative values. Disable log1p or use valid non-negative inputs.")
        return self

    def transform(self, X):
        x = pd.DataFrame(X)
        if not self.enabled:
            return np.asarray(x)
        numeric = x.apply(pd.to_numeric, errors="coerce")
        if (numeric < 0).any().any():
            raise ValueError("Log1pTransformer received negative values during transform; clipping is not allowed.")
        return np.log1p(numeric.to_numpy())

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


# Backward-compatible alias if an earlier draft referenced the cohort-aware name.
CohortAwareLog1pTransformer = Log1pTransformer


class BalancedGradientBoostingClassifier(ClassifierMixin, BaseEstimator):
    """GradientBoostingClassifier with balanced weights computed inside fit().

    The class intentionally computes `compute_sample_weight("balanced", y)` only
    from the exact labels passed into each fit call. This avoids passing a
    precomputed outer-development weight vector through inner CV, where the
    weight values would reflect labels from samples outside the current inner
    training split.
    """

    def __init__(
        self,
        loss: str = "log_loss",
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        criterion: str = "friedman_mse",
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_depth: int | None = 3,
        min_impurity_decrease: float = 0.0,
        init=None,
        random_state=None,
        max_features=None,
        verbose: int = 0,
        max_leaf_nodes: int | None = None,
        warm_start: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int | None = None,
        tol: float = 1e-4,
        ccp_alpha: float = 0.0,
    ):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.init = init
        self.random_state = random_state
        self.max_features = max_features
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.ccp_alpha = ccp_alpha

    def _estimator_params(self) -> dict:
        return {
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "subsample": self.subsample,
            "criterion": self.criterion,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_weight_fraction_leaf": self.min_weight_fraction_leaf,
            "max_depth": self.max_depth,
            "min_impurity_decrease": self.min_impurity_decrease,
            "init": self.init,
            "random_state": self.random_state,
            "max_features": self.max_features,
            "verbose": self.verbose,
            "max_leaf_nodes": self.max_leaf_nodes,
            "warm_start": self.warm_start,
            "validation_fraction": self.validation_fraction,
            "n_iter_no_change": self.n_iter_no_change,
            "tol": self.tol,
            "ccp_alpha": self.ccp_alpha,
        }

    def fit(self, X, y):
        self.fit_sample_weight_ = compute_sample_weight("balanced", y)
        self.estimator_ = GradientBoostingClassifier(**self._estimator_params())
        self.estimator_.fit(X, y, sample_weight=self.fit_sample_weight_)
        self.classes_ = self.estimator_.classes_
        self.n_features_in_ = getattr(self.estimator_, "n_features_in_", None)
        self.feature_names_in_ = getattr(self.estimator_, "feature_names_in_", None)
        self.n_estimators_ = getattr(self.estimator_, "n_estimators_", None)
        self.feature_importances_ = getattr(self.estimator_, "feature_importances_", None)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def decision_function(self, X):
        return self.estimator_.decision_function(X)
