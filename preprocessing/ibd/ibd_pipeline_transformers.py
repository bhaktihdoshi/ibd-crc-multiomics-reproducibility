from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight


class MissingnessFilter(BaseEstimator, TransformerMixin):
    """Drop columns above a training-fold missingness threshold."""

    def __init__(self, max_missing: float = 0.30):
        self.max_missing = max_missing

    def fit(self, X, y=None):
        x = pd.DataFrame(X)
        numeric = x.apply(pd.to_numeric, errors="coerce")
        bad_cols = [
            str(col)
            for col in x.columns
            if numeric[col].isna().all() and not x[col].isna().all()
        ]
        if bad_cols:
            raise ValueError(
                f"MissingnessFilter received non-numeric column(s) {bad_cols}; "
                "encode categorical features before this pipeline step."
            )
        self.feature_names_in_ = np.asarray(x.columns.astype(str))
        self.missing_fraction_ = x.isna().mean(axis=0).to_numpy()
        self.keep_mask_ = self.missing_fraction_ <= self.max_missing
        if not self.keep_mask_.any():
            self.keep_mask_[np.argmin(self.missing_fraction_)] = True
        self.retained_feature_names_ = self.feature_names_in_[self.keep_mask_]
        return self

    def transform(self, X):
        return np.asarray(pd.DataFrame(X))[:, self.keep_mask_]


class ConstantFeatureFilter(BaseEstimator, TransformerMixin):
    """Remove features that are constant within the training fold."""

    def fit(self, X, y=None):
        x = pd.DataFrame(X)
        numeric = x.apply(pd.to_numeric, errors="coerce")
        bad_cols = [
            str(col)
            for col in x.columns
            if numeric[col].isna().all() and not x[col].isna().all()
        ]
        if bad_cols:
            raise ValueError(
                f"ConstantFeatureFilter received non-numeric column(s) {bad_cols}; "
                "encode categorical features before this pipeline step."
            )
        self.feature_names_in_ = np.asarray(x.columns.astype(str))
        self.n_unique_nonmissing_ = numeric.nunique(dropna=True).to_numpy()
        self.keep_mask_ = self.n_unique_nonmissing_ > 1
        if not self.keep_mask_.any():
            self.keep_mask_[0] = True
        self.retained_feature_names_ = self.feature_names_in_[self.keep_mask_]
        return self

    def transform(self, X):
        return np.asarray(pd.DataFrame(X))[:, self.keep_mask_]


# Backward-compatible alias for older text/reports and any historic pickles.
NearZeroVarianceFilter = ConstantFeatureFilter


class BalancedGradientBoostingClassifier(ClassifierMixin, BaseEstimator):
    """GradientBoostingClassifier with balanced weights computed inside fit()."""

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
