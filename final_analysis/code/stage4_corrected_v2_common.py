from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _default_project_root() -> Path:
    # This file lives at <PROJECT>/thesis_preprocessing_workflow/stage4_multiomics_integration/corrected_v2/scripts/.
    return Path(__file__).resolve().parents[4]


# THESIS_PROJECT_ROOT lets this checkout run from any location (e.g. a cloned
# GitHub repo) without editing source; it defaults to the path derived from
# this file's own location, which is correct for the original laptop layout
# too, so no behaviour changes unless the env var is set.
PROJECT = Path(os.environ["THESIS_PROJECT_ROOT"]) if os.environ.get("THESIS_PROJECT_ROOT") else _default_project_root()
ROOT = PROJECT / "thesis_preprocessing_workflow"
STAGE4 = ROOT / "stage4_multiomics_integration"
V2 = STAGE4 / "corrected_v2"
SCRIPTS = V2 / "scripts"
TABLES = V2 / "tables"
PREDICTIONS = V2 / "predictions"
REPORTS = V2 / "reports"
FIGURES = V2 / "figures"
MANIFESTS = V2 / "manifests"
LOGS = V2 / "logs"
SOURCE_PROCESSED = STAGE4 / "data" / "processed"

RANDOM_SEED = 20260813
C_GRID = [0.01, 0.1, 1.0, 10.0]
OUTER_FOLDS = 5
INNER_FOLDS = 5
MICRO_PREVALENCE_THRESHOLD = 0.10
MET_MISSINGNESS_THRESHOLD = 0.30
PSEUDOCOUNT = 1e-6

COHORT_GENUS_POLICY = {
    "Franzosa": "aggregate_species_to_genus",
    "iHMP": "explicit_genus_only",
    "Sinha": "explicit_genus_only",
    "Yachida": "explicit_genus_only",
}

MICROBIOME_INPUTS = {
    "Franzosa": SOURCE_PROCESSED / "franzosa_microbiome_canonical_unfiltered.csv",
    "iHMP": SOURCE_PROCESSED / "ihmp_microbiome_canonical_unfiltered.csv",
    "Sinha": SOURCE_PROCESSED / "sinha_microbiome_canonical_unfiltered.csv",
    "Yachida": SOURCE_PROCESSED / "yachida_microbiome_canonical_unfiltered.csv",
}

METABOLOME_INPUTS = {
    "Franzosa": ROOT / "ibd_pipeline" / "data_processed" / "franzosa_primary_high_confidence_hmdb_matrix.csv",
    "iHMP": ROOT / "ibd_pipeline" / "data_processed" / "ihmp_primary_high_confidence_hmdb_matrix.csv",
    "CRC": ROOT / "crc_pipeline" / "stage2" / "data_processed" / "crc_stage2_primary_metabolomics_canonical.csv",
}

METADATA_INPUTS = {
    "IBD": ROOT / "ibd_pipeline" / "tables" / "included_samples.csv",
    "CRC": ROOT / "crc_pipeline" / "stage2" / "data_processed" / "crc_stage2_primary_metadata_with_clinical.csv",
}


@dataclass(frozen=True)
class TaskSpec:
    disease: str
    task: str
    train_cohort: str
    external_cohort: str
    positive_label: str
    negative_label: str


TASK_SPECS = [
    TaskSpec("IBD", "IBD_vs_Control", "Franzosa", "iHMP", "IBD", "Control"),
    TaskSpec("IBD", "IBD_vs_Control", "iHMP", "Franzosa", "IBD", "Control"),
    TaskSpec("CRC", "CRC_vs_Control", "Sinha", "Yachida", "CRC", "Control/Healthy"),
    TaskSpec("CRC", "CRC_vs_Control", "Yachida", "Sinha", "CRC", "Control/Healthy"),
]


def ensure_v2_dirs() -> None:
    for path in [TABLES, PREDICTIONS, REPORTS, FIGURES, MANIFESTS, LOGS, SCRIPTS, V2 / "data" / "processed"]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def taxon_rank(label: str) -> str:
    text = str(label)
    if text.startswith("genus:"):
        return "genus"
    if text.startswith("species:"):
        return "species"
    if text.startswith("family:"):
        return "family"
    rank = "unknown"
    for marker, name in [("k__", "kingdom"), ("p__", "phylum"), ("c__", "class"), ("o__", "order"), ("f__", "family"), ("g__", "genus"), ("s__", "species")]:
        if marker in text:
            rank = name
    return rank


def canonical_genus(label: str) -> str:
    text = str(label).strip()
    if text.startswith("genus:"):
        return text.split(":", 1)[1].strip()
    if text.startswith("species:"):
        species = text.split(":", 1)[1].strip().replace("_", " ")
        return species.split()[0] if species else ""
    if "|g__" in text:
        for token in text.split("|"):
            if token.startswith("g__"):
                genus = token.split("__", 1)[1].strip()
                return genus if genus and not genus.lower().startswith(("unknown", "unclassified")) else ""
    if "|s__" in text:
        species = text.split("|")[-1].split("__", 1)[1].strip().replace("_", " ")
        return species.split()[0] if species else ""
    words = text.replace("_", " ").split()
    return words[0] if words and not words[0].lower().startswith(("unknown", "unclassified", "root;k__")) else ""


def audit_microbiome_source(cohort: str, df: pd.DataFrame) -> dict[str, Any]:
    feature_cols = [c for c in df.columns if c != "sample_id"]
    ranks = pd.Series([taxon_rank(c) for c in feature_cols], dtype="object")
    numeric = df[feature_cols].apply(pd.to_numeric, errors="coerce") if feature_cols else pd.DataFrame(index=df.index)
    row_sums = numeric.fillna(0.0).sum(axis=1)
    return {
        "cohort": cohort,
        "sample_n": len(df),
        "feature_p": len(feature_cols),
        "explicit_genus_features": int(ranks.eq("genus").sum()),
        "species_features": int(ranks.eq("species").sum()),
        "family_or_other_features": int((~ranks.isin(["genus", "species"])).sum()),
        "duplicated_taxon_labels": int(pd.Index(feature_cols).duplicated().sum()),
        "duplicated_sample_ids": int(df["sample_id"].astype(str).duplicated().sum()) if "sample_id" in df else np.nan,
        "median_row_sum": float(row_sums.median()) if len(row_sums) else np.nan,
        "min_row_sum": float(row_sums.min()) if len(row_sums) else np.nan,
        "max_row_sum": float(row_sums.max()) if len(row_sums) else np.nan,
        "overall_zero_fraction": float((numeric.fillna(0.0).to_numpy() == 0).mean()) if not numeric.empty else np.nan,
        "overall_missing_fraction": float(numeric.isna().to_numpy().mean()) if not numeric.empty else np.nan,
        "authoritative_genus_policy": COHORT_GENUS_POLICY[cohort],
    }


def microbiome_missingness_audit(cohort: str, df: pd.DataFrame) -> pd.DataFrame:
    """Per-feature NaN vs. observed-zero audit, computed before any fillna(0.0).

    An explicit 0 in the source tables represents non-detection/absence and
    is always a valid observation. A NaN cell means the value was never
    resolved — e.g. it arose from an upstream join across cohorts with
    different taxon sets — and is NOT automatically equivalent to zero;
    treating unresolved missingness as absence is a modelling decision that
    must be made explicitly per cohort/source, not assumed generically. This
    function only reports the distinction; it does not decide how partial
    missingness should be handled (see `_require_no_unresolved_missing`,
    which by default raises rather than zero-fills). A feature that is NaN
    for *every* sample in a cohort was never measured in that cohort at all
    and is always excluded upstream in `authoritative_genus_matrix` rather
    than treated as a real zero observation, regardless of the partial-
    missingness policy.
    """
    feature_cols = [c for c in df.columns if c != "sample_id"]
    numeric = df[feature_cols].apply(pd.to_numeric, errors="coerce") if feature_cols else pd.DataFrame(index=df.index)
    n_samples = len(df)
    n_missing_before = numeric.isna().sum(axis=0)
    wholly_missing = n_missing_before.eq(n_samples) if n_samples else n_missing_before.eq(-1)
    n_converted_to_zero = n_missing_before.where(~wholly_missing, 0)
    records = {
        "cohort": cohort,
        "feature": feature_cols,
        "n_samples": n_samples,
        "n_missing_before": n_missing_before.to_numpy(),
        "n_observed_zero": (numeric.eq(0.0)).sum(axis=0).to_numpy(),
        "n_converted_to_zero": n_converted_to_zero.to_numpy(),
        "wholly_missing_unresolved": wholly_missing.to_numpy(),
        "fraction_missing": (n_missing_before / n_samples).to_numpy() if n_samples else np.full(len(feature_cols), np.nan),
    }
    return pd.DataFrame(records)


def authoritative_genus_matrix(cohort: str, df: pd.DataFrame, allow_zero_fill_missing: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "sample_id" not in df.columns:
        raise KeyError(f"{cohort} microbiome matrix lacks sample_id")
    policy = COHORT_GENUS_POLICY[cohort]
    feature_cols = [c for c in df.columns if c != "sample_id"]
    missingness = microbiome_missingness_audit(cohort, df).set_index("feature")
    wholly_missing_cols = set(missingness.index[missingness["wholly_missing_unresolved"]])
    records = []
    selected: dict[str, list[str]] = {}
    for col in feature_cols:
        rank = taxon_rank(col)
        genus = canonical_genus(col)
        include = False
        method = "excluded"
        reason = ""
        if col in wholly_missing_cols:
            reason = "feature never measured in this cohort (NaN for every sample); excluded rather than zero-filled"
        elif not genus:
            reason = "no canonical genus parsed"
        elif policy == "explicit_genus_only":
            include = rank == "genus"
            method = "explicit_genus_profile" if include else "excluded_descendant_or_non_genus"
            reason = "authoritative explicit genus retained" if include else "explicit genus profile exists; descendant/non-genus features excluded to prevent double counting"
        elif policy == "aggregate_species_to_genus":
            include = rank == "species"
            method = "species_to_genus_sum" if include else "excluded_non_species"
            reason = "species aggregated because no authoritative full genus profile is available" if include else "not part of species-only aggregation policy"
        else:
            raise ValueError(f"Unknown genus policy for {cohort}: {policy}")
        records.append(
            {
                "cohort": cohort,
                "original_taxon_label": col,
                "original_rank": rank,
                "canonical_genus": genus,
                "include_for_authoritative_genus": include,
                "transformation_or_aggregation_method": method,
                "source_table": str(MICROBIOME_INPUTS[cohort]),
                "mapping_confidence": "deterministic_label_parse" if genus else "unmapped",
                "reason_for_exclusion": "" if include else reason,
            }
        )
        if include:
            selected.setdefault(genus, []).append(col)
    if not selected:
        raise ValueError(f"No authoritative genus features selected for {cohort}")
    genus_series: dict[str, pd.Series] = {}
    for genus in sorted(selected):
        raw = df[selected[genus]].apply(pd.to_numeric, errors="coerce")
        # Wholly-unmeasured source columns were already excluded from `selected`
        # above. Any NaN remaining here is *partial* missingness in a column that
        # IS part of the authoritative genus construction, so it goes through the
        # same strict guard as the modelling preprocessors: raise by default,
        # only zero-fill if the caller has explicitly opted in.
        values = _require_no_unresolved_missing(raw, allow_zero_fill_missing, f"authoritative_genus_matrix({cohort}, genus={genus})")
        genus_series[f"mic__{genus}"] = values.sum(axis=1).reset_index(drop=True)
    out = pd.concat([pd.DataFrame({"sample_id": df["sample_id"].astype(str).reset_index(drop=True)}), pd.DataFrame(genus_series)], axis=1)
    values = out.drop(columns=["sample_id"]).clip(lower=0.0)
    row_sums = values.sum(axis=1)
    if (row_sums <= 0).any():
        bad = out.loc[row_sums <= 0, "sample_id"].astype(str).head(5).tolist()
        raise ValueError(f"{cohort} has zero-sum samples after authoritative genus construction: {bad}")
    closed = values.div(row_sums, axis=0)
    out = pd.concat([out[["sample_id"]], closed], axis=1)
    return out, pd.DataFrame(records)


class MissingnessFilter(BaseEstimator, TransformerMixin):
    def __init__(self, max_missing_fraction: float = MET_MISSINGNESS_THRESHOLD):
        self.max_missing_fraction = max_missing_fraction

    def fit(self, X, y=None):
        x = pd.DataFrame(X).apply(pd.to_numeric, errors="raise")
        self.feature_names_in_ = np.array(x.columns.astype(str))
        self.keep_mask_ = x.isna().mean(axis=0).to_numpy() <= self.max_missing_fraction
        if not self.keep_mask_.any():
            raise ValueError("MissingnessFilter removed all features.")
        self.feature_names_out_ = self.feature_names_in_[self.keep_mask_]
        return self

    def transform(self, X):
        x = pd.DataFrame(X).apply(pd.to_numeric, errors="raise")
        return x.loc[:, self.keep_mask_]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


class ConstantFeatureFilter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        x = pd.DataFrame(X).apply(pd.to_numeric, errors="raise")
        self.feature_names_in_ = np.array(x.columns.astype(str))
        self.keep_mask_ = x.nunique(dropna=True).to_numpy() > 1
        if not self.keep_mask_.any():
            raise ValueError("ConstantFeatureFilter removed all features.")
        self.feature_names_out_ = self.feature_names_in_[self.keep_mask_]
        return self

    def transform(self, X):
        x = pd.DataFrame(X).apply(pd.to_numeric, errors="raise")
        return x.loc[:, self.keep_mask_]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


def _require_no_unresolved_missing(x: pd.DataFrame, allow_zero_fill_missing: bool, context: str) -> pd.DataFrame:
    """Guard against silently treating a partially-missing microbiome value as absence.

    Explicit zeros in the source tables mean "detected absent" and are always
    valid. A NaN reaching this point means a value was never resolved by the
    upstream missingness audit/authoritative-genus construction. That is not
    automatically equivalent to zero: absence must be an explicit, documented
    convention. By default this raises so a future partially-missing feature
    cannot be silently zero-filled; pass allow_zero_fill_missing=True only
    when a specific cohort/source has a documented convention where NaN does
    mean non-detection.
    """
    if not x.isna().any().any():
        return x
    if not allow_zero_fill_missing:
        offending = x.isna().sum(axis=0)
        offending = offending[offending > 0]
        raise ValueError(
            f"{context}: unresolved NaN reached the microbiome preprocessor for features "
            f"{offending.to_dict()}; either the upstream missingness audit should have excluded "
            "or resolved these, or allow_zero_fill_missing=True must be set explicitly for a "
            "cohort/source with a documented absence-equals-NaN convention."
        )
    return x.fillna(0.0)


class MicrobiomeCLRPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, prevalence_threshold: float = MICRO_PREVALENCE_THRESHOLD, pseudocount: float = PSEUDOCOUNT, allow_zero_fill_missing: bool = False):
        self.prevalence_threshold = prevalence_threshold
        self.pseudocount = pseudocount
        self.allow_zero_fill_missing = allow_zero_fill_missing

    def fit(self, X, y=None):
        x = pd.DataFrame(X).apply(pd.to_numeric, errors="raise")
        x = _require_no_unresolved_missing(x, self.allow_zero_fill_missing, "MicrobiomeCLRPreprocessor.fit")
        if (x < 0).any().any():
            raise ValueError("Microbiome matrix contains negative values.")
        self.feature_names_in_ = np.array(x.columns.astype(str))
        self.keep_mask_ = ((x > 0).mean(axis=0).to_numpy() >= self.prevalence_threshold)
        if not self.keep_mask_.any():
            raise ValueError("Microbiome prevalence filtering removed all features.")
        self.feature_names_out_ = self.feature_names_in_[self.keep_mask_]
        closed = self._reclose(x.loc[:, self.keep_mask_])
        transformed = self._clr(closed)
        self.scaler_ = StandardScaler().fit(transformed)
        return self

    @staticmethod
    def _reclose(x: pd.DataFrame) -> pd.DataFrame:
        x = x.clip(lower=0.0)
        row_sums = x.sum(axis=1)
        if (row_sums <= 0).any():
            raise ValueError("A microbiome sample has zero total abundance after fold-local filtering.")
        return x.div(row_sums, axis=0)

    def _clr(self, x: pd.DataFrame) -> pd.DataFrame:
        logged = np.log(x + self.pseudocount)
        clr = logged.sub(logged.mean(axis=1), axis=0)
        if not np.isfinite(clr.to_numpy()).all():
            raise ValueError("CLR produced NaN or infinite values.")
        return clr

    def transform(self, X):
        x = pd.DataFrame(X).apply(pd.to_numeric, errors="raise")
        x = _require_no_unresolved_missing(x, self.allow_zero_fill_missing, "MicrobiomeCLRPreprocessor.transform")
        closed = self._reclose(x.loc[:, self.keep_mask_])
        return self.scaler_.transform(self._clr(closed))

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


def metabolome_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("missingness", MissingnessFilter(MET_MISSINGNESS_THRESHOLD)),
            ("constant", ConstantFeatureFilter()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


class MicrobiomeRankPreprocessor(BaseEstimator, TransformerMixin):
    """Fold-local within-sample rank transform of prevalence-filtered genera.

    A scale-robust sensitivity variant, not a claim of cross-platform assay
    equivalence. Uses the identical fold-local prevalence filter as
    MicrobiomeCLRPreprocessor, but instead of CLR replaces each retained
    genus's proportion with its within-sample rank (ties averaged), then
    z-standardises using training-fold mean/SD, frozen and applied to
    validation/external data. Rank removes dependence on absolute abundance
    magnitude, so it reduces (but does not eliminate) sensitivity to
    multiplicative abundance-scale differences between platforms (e.g. 16S
    vs. shotgun quantification). It cannot correct for taxon-specific primer
    bias, detection-limit differences, reference-database differences, or
    other platform-specific measurement error.
    """

    def __init__(self, prevalence_threshold: float = MICRO_PREVALENCE_THRESHOLD, allow_zero_fill_missing: bool = False):
        self.prevalence_threshold = prevalence_threshold
        self.allow_zero_fill_missing = allow_zero_fill_missing

    def fit(self, X, y=None):
        x = pd.DataFrame(X).apply(pd.to_numeric, errors="raise")
        x = _require_no_unresolved_missing(x, self.allow_zero_fill_missing, "MicrobiomeRankPreprocessor.fit")
        if (x < 0).any().any():
            raise ValueError("Microbiome matrix contains negative values.")
        self.feature_names_in_ = np.array(x.columns.astype(str))
        self.keep_mask_ = ((x > 0).mean(axis=0).to_numpy() >= self.prevalence_threshold)
        if not self.keep_mask_.any():
            raise ValueError("Microbiome prevalence filtering removed all features.")
        self.feature_names_out_ = self.feature_names_in_[self.keep_mask_]
        self.scaler_ = StandardScaler().fit(self._rank(x.loc[:, self.keep_mask_]))
        return self

    @staticmethod
    def _rank(x: pd.DataFrame) -> pd.DataFrame:
        ranked = x.rank(axis=1, method="average")
        if not np.isfinite(ranked.to_numpy()).all():
            raise ValueError("Within-sample rank transform produced NaN or infinite values.")
        return ranked

    def transform(self, X):
        x = pd.DataFrame(X).apply(pd.to_numeric, errors="raise")
        x = _require_no_unresolved_missing(x, self.allow_zero_fill_missing, "MicrobiomeRankPreprocessor.transform")
        return self.scaler_.transform(self._rank(x.loc[:, self.keep_mask_]))

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


class MicrobiomePresenceAbsencePreprocessor(BaseEstimator, TransformerMixin):
    """Fold-local detected/not-detected binary transform of prevalence-filtered genera.

    A scale-robust sensitivity variant, not a claim of cross-platform assay
    equivalence. Same fold-local prevalence filter as MicrobiomeCLRPreprocessor;
    abundance values are then collapsed to a binary detected (>0) indicator,
    discarding all quantitative abundance information, then z-standardised
    using the training-fold mean/SD (frozen and applied unchanged to
    validation/external data) so the L2 penalty geometry matches the other
    microbiome variants and the comparison isolates representation rather
    than feature scale. Presence/absence removes dependence on abundance
    magnitude entirely, but it cannot correct for taxon-specific primer bias,
    detection-limit differences, reference-database differences, or other
    platform-specific measurement error between assays.
    """

    def __init__(self, prevalence_threshold: float = MICRO_PREVALENCE_THRESHOLD, allow_zero_fill_missing: bool = False):
        self.prevalence_threshold = prevalence_threshold
        self.allow_zero_fill_missing = allow_zero_fill_missing

    def fit(self, X, y=None):
        x = pd.DataFrame(X).apply(pd.to_numeric, errors="raise")
        x = _require_no_unresolved_missing(x, self.allow_zero_fill_missing, "MicrobiomePresenceAbsencePreprocessor.fit")
        if (x < 0).any().any():
            raise ValueError("Microbiome matrix contains negative values.")
        self.feature_names_in_ = np.array(x.columns.astype(str))
        self.keep_mask_ = ((x > 0).mean(axis=0).to_numpy() >= self.prevalence_threshold)
        if not self.keep_mask_.any():
            raise ValueError("Microbiome prevalence filtering removed all features.")
        self.feature_names_out_ = self.feature_names_in_[self.keep_mask_]
        self.scaler_ = StandardScaler().fit(self._binary(x.loc[:, self.keep_mask_]))
        return self

    @staticmethod
    def _binary(x: pd.DataFrame) -> pd.DataFrame:
        binary = (x > 0).astype(float)
        if not np.isfinite(binary.to_numpy()).all():
            raise ValueError("Presence/absence transform produced NaN or infinite values.")
        return binary

    def transform(self, X):
        x = pd.DataFrame(X).apply(pd.to_numeric, errors="raise")
        x = _require_no_unresolved_missing(x, self.allow_zero_fill_missing, "MicrobiomePresenceAbsencePreprocessor.transform")
        return self.scaler_.transform(self._binary(x.loc[:, self.keep_mask_]))

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


MICROBIOME_SENSITIVITY_VARIANTS = ("clr_zscore", "within_sample_rank", "presence_absence")


def microbiome_pipeline(variant: str = "clr_zscore") -> Pipeline:
    if variant == "clr_zscore":
        return Pipeline([("microbiome", MicrobiomeCLRPreprocessor())])
    if variant == "within_sample_rank":
        return Pipeline([("microbiome", MicrobiomeRankPreprocessor())])
    if variant == "presence_absence":
        return Pipeline([("microbiome", MicrobiomePresenceAbsencePreprocessor())])
    raise ValueError(f"Unknown microbiome sensitivity variant: {variant}")


def build_ridge_pipeline(representation: str, c_value: float, microbiome_variant: str = "clr_zscore") -> Pipeline:
    clf = LogisticRegression(
        penalty="l2",
        C=c_value,
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_SEED,
    )
    if representation == "METABOLOME":
        return Pipeline([("metabolome", metabolome_pipeline()), ("clf", clf)])
    if representation == "MICROBIOME":
        return Pipeline([("microbiome", microbiome_pipeline(microbiome_variant)), ("clf", clf)])
    if representation == "EARLY_INTEGRATION":
        pre = ColumnTransformer(
            [
                ("metabolome", metabolome_pipeline(), lambda x: [c for c in x.columns if str(c).startswith("met__")]),
                ("microbiome", microbiome_pipeline(microbiome_variant), lambda x: [c for c in x.columns if str(c).startswith("mic__")]),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return Pipeline([("block_preprocessor", pre), ("clf", clf)])
    raise ValueError(representation)


def fold_count(y: pd.Series, desired: int = 5) -> int:
    min_class = int(y.value_counts().min())
    if min_class < 2:
        raise ValueError(f"Cannot split one class or class count <2: {y.value_counts().to_dict()}")
    return min(desired, min_class)


def cross_fitted_probs(X: pd.DataFrame, y: pd.Series, representation: str, c_value: float, folds: int, seed: int) -> np.ndarray:
    probs = np.full(len(y), np.nan)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, val_idx in cv.split(X, y):
        pipe = build_ridge_pipeline(representation, c_value)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        probs[val_idx] = pipe.predict_proba(X.iloc[val_idx])[:, 1]
    if np.isnan(probs).any():
        raise ValueError("Cross-fitted probabilities contain NaN.")
    return probs


def select_youden_threshold_from_oof(y: pd.Series, prob: np.ndarray) -> dict[str, float]:
    if y.nunique() != 2 or len(np.unique(prob)) < 2:
        return {"threshold": 0.5, "youden_j": 0.0, "sensitivity": np.nan, "specificity": np.nan, "roc_auc": np.nan}
    fpr, tpr, thresholds = roc_curve(y, prob)
    j = tpr - fpr
    idx = int(np.nanargmax(j))
    return {"threshold": float(thresholds[idx]), "youden_j": float(j[idx]), "sensitivity": float(tpr[idx]), "specificity": float(1 - fpr[idx]), "roc_auc": float(roc_auc_score(y, prob))}


def binary_metrics(y: pd.Series, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    tn = int(((pred == 0) & (y.to_numpy() == 0)).sum())
    fp = int(((pred == 1) & (y.to_numpy() == 0)).sum())
    fn = int(((pred == 0) & (y.to_numpy() == 1)).sum())
    tp = int(((pred == 1) & (y.to_numpy() == 1)).sum())
    return {
        "roc_auc": float(roc_auc_score(y, prob)) if y.nunique() == 2 and len(np.unique(prob)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(y, prob)) if y.nunique() == 2 else np.nan,
        "external_positive_prevalence": float(y.mean()),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "probability_mean": float(np.mean(prob)),
        "probability_min": float(np.min(prob)),
        "probability_max": float(np.max(prob)),
        "probability_sd": float(np.std(prob)),
        "degenerate_probability_flag": bool(len(np.unique(np.round(prob, 12))) <= 1),
        "one_class_prediction_flag": bool(len(np.unique(pred)) <= 1),
    }


def stratified_bootstrap_metric_ci(y: pd.Series, prob: np.ndarray, metric: str, n_boot: int = 2000, seed: int = RANDOM_SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y_arr = y.to_numpy(dtype=int)
    pos = np.where(y_arr == 1)[0]
    neg = np.where(y_arr == 0)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, size=len(pos), replace=True), rng.choice(neg, size=len(neg), replace=True)])
        yy = y_arr[idx]
        pp = prob[idx]
        if len(np.unique(yy)) < 2:
            continue
        if metric == "roc_auc" and len(np.unique(pp)) > 1:
            vals.append(roc_auc_score(yy, pp))
        elif metric == "pr_auc":
            vals.append(average_precision_score(yy, pp))
    if not vals:
        return {f"{metric}_ci_low": np.nan, f"{metric}_ci_high": np.nan, f"{metric}_bootstrap_n": 0}
    return {f"{metric}_ci_low": float(np.percentile(vals, 2.5)), f"{metric}_ci_high": float(np.percentile(vals, 97.5)), f"{metric}_bootstrap_n": len(vals)}
