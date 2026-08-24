from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from run_stage4b_corrected_v2b import OUTPUT_PREFIX, CorrectedDataBundle, aligned_train_external, pipeline_scores
from stage4_corrected_v2_common import PREDICTIONS, TABLES, TASK_SPECS, atomic_csv, build_ridge_pipeline, ensure_v2_dirs


REPRESENTATIONS = ["METABOLOME", "MICROBIOME", "EARLY_INTEGRATION"]
QUANTILES = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]


def feature_blocks(columns: list[str], representation: str) -> dict[str, list[str]]:
    if representation == "METABOLOME":
        return {"metabolome": columns}
    if representation == "MICROBIOME":
        return {"microbiome": columns}
    return {
        "metabolome": [c for c in columns if str(c).startswith("met__")],
        "microbiome": [c for c in columns if str(c).startswith("mic__")],
    }


def transformed_matrix(pipe, X: pd.DataFrame, representation: str, block: str) -> np.ndarray:
    if representation == "METABOLOME":
        return np.asarray(pipe.named_steps["metabolome"].transform(X), dtype=float)
    if representation == "MICROBIOME":
        return np.asarray(pipe.named_steps["microbiome"].transform(X), dtype=float)
    pre = pipe.named_steps["block_preprocessor"]
    transformer = pre.named_transformers_[block]
    cols = [c for c in X.columns if str(c).startswith("met__" if block == "metabolome" else "mic__")]
    return np.asarray(transformer.transform(X[cols]), dtype=float)


def raw_summary(values: pd.DataFrame) -> dict[str, Any]:
    arr = values.to_numpy(dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"raw_min": np.nan, "raw_max": np.nan, "raw_q50": np.nan, "raw_nonfinite_n": int(values.size)}
    return {
        "raw_min": float(np.min(arr)),
        "raw_max": float(np.max(arr)),
        "raw_q50": float(np.quantile(arr, 0.5)),
        "raw_nonfinite_n": int(values.size - len(arr)),
    }


def transformed_summary(arr: np.ndarray, prefix: str) -> dict[str, Any]:
    flat = np.asarray(arr, dtype=float).ravel()
    finite = flat[np.isfinite(flat)]
    out: dict[str, Any] = {f"{prefix}_nonfinite_n": int(flat.size - len(finite))}
    if len(finite) == 0:
        for q in QUANTILES:
            out[f"{prefix}_q{int(q * 100):02d}"] = np.nan
        out.update({f"{prefix}_min": np.nan, f"{prefix}_max": np.nan, f"{prefix}_abs_gt5_fraction": np.nan, f"{prefix}_abs_gt10_fraction": np.nan, f"{prefix}_abs_gt50_fraction": np.nan})
        return out
    for q in QUANTILES:
        out[f"{prefix}_q{int(q * 100):02d}"] = float(np.quantile(finite, q))
    out.update(
        {
            f"{prefix}_min": float(np.min(finite)),
            f"{prefix}_max": float(np.max(finite)),
            f"{prefix}_abs_gt5_fraction": float(np.mean(np.abs(finite) > 5)),
            f"{prefix}_abs_gt10_fraction": float(np.mean(np.abs(finite) > 10)),
            f"{prefix}_abs_gt50_fraction": float(np.mean(np.abs(finite) > 50)),
        }
    )
    return out


def infer_input_transform(block: str) -> str:
    if block == "metabolome":
        return "numeric HMDB matrix as supplied to corrected-v2 Stage 4B; fold-local missingness, median imputation and StandardScaler are source-fitted"
    return "relative-abundance genus matrix; prevalence filtering, closure, CLR and StandardScaler are source-fitted"


def run_diagnostics() -> None:
    ensure_v2_dirs()
    bundle = CorrectedDataBundle()
    threshold_path = TABLES / f"{OUTPUT_PREFIX}_thresholds.csv"
    if not threshold_path.exists():
        threshold_path = TABLES / "stage4b_corrected_v2_thresholds.csv"
    thresholds = pd.read_csv(threshold_path)
    rows = []
    for spec in TASK_SPECS:
        for rep in REPRESENTATIONS:
            X_train, y_train, _, X_ext, _, _ = aligned_train_external(bundle, spec, rep)
            match = thresholds[
                thresholds["disease"].eq(spec.disease)
                & thresholds["task"].eq(spec.task)
                & thresholds["training_cohort"].eq(spec.train_cohort)
                & thresholds["external_cohort"].eq(spec.external_cohort)
                & thresholds["representation"].eq(rep)
            ]
            selected_c = float(match["selected_C"].iloc[0]) if not match.empty and "selected_C" in match else 1.0
            pipe = build_ridge_pipeline(rep, selected_c)
            pipe.fit(X_train, y_train)
            source_score, source_prob = pipeline_scores(pipe, X_train)
            external_score, external_prob = pipeline_scores(pipe, X_ext)
            blocks = feature_blocks(list(X_train.columns), rep)
            for block, cols in blocks.items():
                if not cols:
                    continue
                source_raw = X_train[cols].apply(pd.to_numeric, errors="raise")
                external_raw = X_ext[cols].apply(pd.to_numeric, errors="raise")
                source_transformed = transformed_matrix(pipe, X_train, rep, block)
                external_transformed = transformed_matrix(pipe, X_ext, rep, block)
                rows.append(
                    {
                        "disease": spec.disease,
                        "task": spec.task,
                        "training_cohort": spec.train_cohort,
                        "external_cohort": spec.external_cohort,
                        "representation": rep,
                        "feature_block": block,
                        "source_n": int(len(X_train)),
                        "external_n": int(len(X_ext)),
                        "source_feature_n": int(len(cols)),
                        "external_feature_n": int(len(cols)),
                        "input_transform_description": infer_input_transform(block),
                        "block_specific_standardisation_fit_on_source_only": True,
                        "selected_C_used_for_diagnostic_refit": selected_c,
                        **{f"source_{k}": v for k, v in raw_summary(source_raw).items()},
                        **{f"external_{k}": v for k, v in raw_summary(external_raw).items()},
                        **transformed_summary(source_transformed, "source_transformed"),
                        **transformed_summary(external_transformed, "external_transformed"),
                        "source_max_abs_model_logit": float(np.max(np.abs(source_score))),
                        "external_max_abs_model_logit": float(np.max(np.abs(external_score))),
                        "source_probability_exact_zero_fraction": float(np.mean(source_prob <= 1e-15)),
                        "source_probability_exact_one_fraction": float(np.mean(source_prob >= 1 - 1e-15)),
                        "external_probability_exact_zero_fraction": float(np.mean(external_prob <= 1e-15)),
                        "external_probability_exact_one_fraction": float(np.mean(external_prob >= 1 - 1e-15)),
                        "source_unique_decision_scores": int(pd.Series(np.round(source_score, 12)).nunique(dropna=True)),
                        "external_unique_decision_scores": int(pd.Series(np.round(external_score, 12)).nunique(dropna=True)),
                        "source_unique_probabilities": int(pd.Series(np.round(source_prob, 12)).nunique(dropna=True)),
                        "external_unique_probabilities": int(pd.Series(np.round(external_prob, 12)).nunique(dropna=True)),
                        "external_labels_used": False,
                    }
                )
    atomic_csv(pd.DataFrame(rows), TABLES / f"{OUTPUT_PREFIX}_transformation_domain_shift_diagnostics.csv")
    print("STAGE4B_CORRECTED_V2B_TRANSFORM_SHIFT_DIAGNOSTICS_COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit corrected-v2b source-vs-external transformation/domain shift without altering models.")
    parser.add_argument("--run", action="store_true", help="Run diagnostic refits using source labels only and export shift diagnostics.")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("Use --run to create transformation/domain-shift diagnostics. This does not use external labels or alter Stage 4B outputs.")
    run_diagnostics()


if __name__ == "__main__":
    main()
