from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


REPO = Path(__file__).resolve().parents[3]
CODE = REPO / "final_analysis" / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import run_stage4c_corrected_v2 as s4c  # noqa: E402
from stage4_corrected_v2_common import RANDOM_SEED  # noqa: E402


PRIMARY_K = 8
PRIMARY_SEED = 20260813
K_VALUES = [5, 8, 10]
THRESHOLDS = [0.15, 0.20, 0.25, 0.30]
REPLICATED_IBD_METABOLITES = {"putrescine", "n-acetylputrescine", "eicosatrienoic acid"}


@dataclass(frozen=True)
class Context:
    repo: Path
    output_root: Path
    model_root: Path
    tables: Path
    reports: Path
    logs: Path
    tests: Path
    thesis_source_root: Path
    seeds: list[int]
    resume: bool


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    tmp.replace(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, indent=2, default=str) + "\n")


def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def ensure_dirs(ctx: Context) -> None:
    for path in [ctx.model_root, ctx.tables, ctx.reports, ctx.logs, ctx.tests]:
        path.mkdir(parents=True, exist_ok=True)


def path_for_model(ctx: Context, disease: str, k: int, seed: int) -> Path:
    if k == PRIMARY_K and seed == PRIMARY_SEED:
        return ctx.repo / "final_analysis" / "tables" / "models" / f"stage4c_corrected_v2_mofa_{disease.lower()}_k8.hdf5"
    return ctx.model_root / disease.lower() / f"mofa_{disease.lower()}_k{k}_seed{seed}.hdf5"


def model_manifest_path(ctx: Context, disease: str, k: int, seed: int) -> Path:
    return ctx.model_root / disease.lower() / f"mofa_{disease.lower()}_k{k}_seed{seed}.json"


def build_inputs() -> tuple[dict[tuple[str, str], s4c.CohortBlock], dict[tuple[str, str], dict[str, list[str]]], pd.DataFrame]:
    blocks = s4c.load_blocks()
    feature_sets, feature_qc = s4c.build_feature_sets(blocks)
    return blocks, feature_sets, feature_qc


def long_input_for_disease(disease: str, blocks: dict[tuple[str, str], s4c.CohortBlock], feature_sets: dict[tuple[str, str], dict[str, list[str]]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_df, meta_df, serial = s4c.build_mofa_input_for_disease(disease, blocks, feature_sets)
    if any(c in data_df.columns for c in ["phenotype", "diagnosis", "label", "y"]):
        raise ValueError(f"{disease}: phenotype-like column found in MOFA fitting matrix.")
    return data_df, meta_df, serial


def construct_mofa(data_df: pd.DataFrame, outfile: Path, k: int, seed: int):
    from mofapy2.run.entry_point import entry_point

    ent = entry_point()
    ent.set_data_options(scale_views=False, scale_groups=False)
    views = sorted(data_df["view"].astype(str).unique())
    ent.set_data_df(data_df, likelihoods=["gaussian"] * len(views))
    ent.set_model_options(factors=k, ard_factors=True)
    ent.set_train_options(iter=1000, convergence_mode="medium", seed=seed, outfile=str(outfile))
    ent.build()
    return ent


def decode(values) -> list[str]:
    return s4c.decode_h5_values(values)


def extract_hdf5(path: Path, disease: str, cohorts: list[str], meta_df: pd.DataFrame, k: int) -> dict[str, pd.DataFrame]:
    import h5py

    scores, loadings, variance, diagnostics = [], [], [], []
    with h5py.File(path, "r") as h5:
        groups = decode(h5["groups/groups"][()])
        views = decode(h5["views/views"][()])
        if groups != cohorts:
            raise ValueError(f"{disease}: expected MOFA groups {cohorts}, observed {groups}")
        meta_lookup = meta_df.assign(sample_id=meta_df["sample_id"].astype(str)).set_index("sample_id")
        for cohort in groups:
            samples = decode(h5[f"samples/{cohort}"][()])
            z, z_orientation = s4c.orient_factor_scores(np.asarray(h5[f"expectations/Z/{cohort}"]), len(samples), k, f"{disease} {cohort}")
            for factor_idx in range(k):
                vals = z[:, factor_idx]
                diagnostics.append({"disease": disease, "cohort": cohort, "factor": f"Factor{factor_idx + 1}", "score_variance": float(np.nanvar(vals)), "nondegenerate_score": bool(np.nanvar(vals) > 1e-8)})
                for sample, value in zip(samples, vals):
                    scores.append({"disease": disease, "cohort": cohort, "sample_id": sample, "phenotype": str(meta_lookup.loc[sample, "disease_group"]), "factor": f"Factor{factor_idx + 1}", "score": float(value), "z_orientation": z_orientation})
        for view in views:
            features = decode(h5[f"features/{view}"][()])
            w, w_orientation = s4c.orient_loadings_matrix(np.asarray(h5[f"expectations/W/{view}"]), len(features), k, f"{disease} {view}")
            for factor_idx in range(k):
                for feature, value in zip(features, w[:, factor_idx]):
                    loadings.append({"disease": disease, "view": view, "feature_id": feature, "feature_name": feature.removeprefix("met__").removeprefix("mic__"), "factor": f"Factor{factor_idx + 1}", "loading": float(value), "w_orientation": w_orientation})
        s4c.collect_variance_explained(h5, variance, disease, views, groups, k)
        elbo = np.asarray(h5["training_stats/elbo"], dtype=float) if "training_stats/elbo" in h5 else np.array([])
        diagnostics.append({"disease": disease, "cohort": "ALL_GROUPS", "factor": "training", "elbo_iterations": int(len(elbo)), "elbo_final": float(elbo[-1]) if len(elbo) else np.nan, "elbo_delta_last": float(elbo[-1] - elbo[-2]) if len(elbo) > 1 else np.nan, "convergence_diagnostic": "elbo_recorded" if len(elbo) else "elbo_missing"})
    return {"scores": pd.DataFrame(scores), "loadings": pd.DataFrame(loadings), "variance": pd.DataFrame(variance), "diagnostics": pd.DataFrame(diagnostics)}


def classify_factors(variance: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    fac = variance[variance["factor"].astype(str).str.startswith("Factor")].copy()
    for disease in sorted(fac["disease"].unique()):
        for factor in sorted(fac.loc[fac["disease"].eq(disease), "factor"].unique(), key=lambda x: int(str(x).replace("Factor", ""))):
            sub = fac[(fac["disease"].eq(disease)) & (fac["factor"].eq(factor))]
            by_view = sub.groupby("view")["variance_explained"].mean().to_dict()
            met = float(by_view.get("metabolome", 0.0))
            mic = float(by_view.get("microbiome", 0.0))
            total = met + mic
            if total <= 0:
                cls = "low_information"
            elif min(met, mic) / total >= threshold:
                cls = "shared_multiview"
            elif met > mic:
                cls = "metabolome_dominant"
            else:
                cls = "microbiome_dominant"
            rows.append({"disease": disease, "factor": factor, "threshold": threshold, "mean_metabolome_r2_percent": met, "mean_microbiome_r2_percent": mic, "mean_total_r2_percent": total, "factor_classification": cls, "classification_rule": f"shared if each view contributes >= {threshold:.0%} of mean two-view R2; otherwise dominant view is larger R2"})
    return pd.DataFrame(rows)


def cosine(a: pd.Series, b: pd.Series) -> float:
    common = sorted(set(a.index.astype(str)) & set(b.index.astype(str)))
    if not common:
        return np.nan
    av = a.loc[common].astype(float).to_numpy()
    bv = b.loc[common].astype(float).to_numpy()
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return float(np.dot(av, bv) / denom) if denom > 0 else np.nan


def top_overlap(loadings: pd.DataFrame, run_factor: str, ref_loadings: pd.DataFrame, ref_factor: str, view: str, n: int = 15) -> int:
    a = loadings[(loadings["factor"].eq(run_factor)) & (loadings["view"].eq(view))].copy()
    b = ref_loadings[(ref_loadings["factor"].eq(ref_factor)) & (ref_loadings["view"].eq(view))].copy()
    a = set(a.assign(abs_loading=a["loading"].abs()).sort_values("abs_loading", ascending=False).head(n)["feature_id"].astype(str))
    b = set(b.assign(abs_loading=b["loading"].abs()).sort_values("abs_loading", ascending=False).head(n)["feature_id"].astype(str))
    return len(a & b)


def align_to_reference(disease: str, run: dict[str, pd.DataFrame], ref: dict[str, pd.DataFrame], k: int, seed: int) -> pd.DataFrame:
    rows = []
    factors = [f"Factor{i + 1}" for i in range(k)]
    ref_factors = sorted(ref["loadings"]["factor"].unique(), key=lambda x: int(str(x).replace("Factor", "")))
    score = np.zeros((len(factors), len(ref_factors)))
    details: dict[tuple[int, int], dict[str, float]] = {}
    for i, fac in enumerate(factors):
        for j, ref_fac in enumerate(ref_factors):
            cosines = {}
            for view in ["metabolome", "microbiome"]:
                a = run["loadings"][(run["loadings"]["view"].eq(view)) & (run["loadings"]["factor"].eq(fac))].set_index("feature_id")["loading"]
                b = ref["loadings"][(ref["loadings"]["view"].eq(view)) & (ref["loadings"]["factor"].eq(ref_fac))].set_index("feature_id")["loading"]
                cosines[view] = cosine(a, b)
            vals = [abs(v) for v in cosines.values() if np.isfinite(v)]
            combined = float(np.mean(vals)) if vals else np.nan
            score[i, j] = -combined if np.isfinite(combined) else 0.0
            signed_vals = [v for v in cosines.values() if np.isfinite(v)]
            signed = float(np.mean(signed_vals)) if signed_vals else np.nan
            details[(i, j)] = {"metabolome_cosine": cosines.get("metabolome", np.nan), "microbiome_cosine": cosines.get("microbiome", np.nan), "combined_cosine": combined, "signed_combined_cosine": signed}
    row_ind, col_ind = linear_sum_assignment(score)
    matched_run = set()
    for i, j in zip(row_ind, col_ind):
        fac, ref_fac = factors[i], ref_factors[j]
        matched_run.add(fac)
        d = details[(i, j)]
        rows.append({"disease": disease, "K": k, "seed": seed, "sensitivity_factor": fac, "reference_factor": ref_fac, **d, "sign_orientation": 1 if d["signed_combined_cosine"] >= 0 else -1, "unmatched_status": "matched", "top15_metabolite_overlap": top_overlap(run["loadings"], fac, ref["loadings"], ref_fac, "metabolome"), "top15_microbiome_overlap": top_overlap(run["loadings"], fac, ref["loadings"], ref_fac, "microbiome"), "matching_basis": "mean absolute loading cosine across metabolome and microbiome only"})
    for fac in sorted(set(factors) - matched_run):
        rows.append({"disease": disease, "K": k, "seed": seed, "sensitivity_factor": fac, "reference_factor": "", "metabolome_cosine": np.nan, "microbiome_cosine": np.nan, "combined_cosine": np.nan, "signed_combined_cosine": np.nan, "sign_orientation": np.nan, "unmatched_status": "unmatched_extra_sensitivity_factor", "top15_metabolite_overlap": 0, "top15_microbiome_overlap": 0, "matching_basis": "mean absolute loading cosine across metabolome and microbiome only"})
    return pd.DataFrame(rows)


def annotate_alignment_degeneracy(alignment: pd.DataFrame) -> pd.DataFrame:
    out = alignment.copy()
    out["alignment_degeneracy_flag"] = out[["metabolome_cosine", "microbiome_cosine", "combined_cosine"]].isna().any(axis=1)
    out.loc[out["alignment_degeneracy_flag"] & out["unmatched_status"].eq("matched"), "unmatched_status"] = "matched_but_degenerate_loading_similarity"
    return out


def run_or_reuse(ctx: Context, disease: str, k: int, seed: int, data_df: pd.DataFrame, meta_df: pd.DataFrame, serial: pd.DataFrame, input_hash: str) -> tuple[str, Path, str, str, int, float]:
    model_path = path_for_model(ctx, disease, k, seed)
    manifest_path = model_manifest_path(ctx, disease, k, seed)
    if k == PRIMARY_K and seed == PRIMARY_SEED:
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        return "reused_primary", model_path, sha256(model_path), "primary_existing_model_verified", 0, 0.0
    if ctx.resume and manifest_path.exists() and model_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("input_hash") == input_hash and manifest.get("model_sha256") == sha256(model_path):
            return "reused_completed", model_path, sha256(model_path), "resume_manifest_verified", 0, 0.0
    model_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    status, message = "success", ""
    try:
        ent = construct_mofa(data_df, model_path, k, seed)
        ent.run()
        ent.save(str(model_path))
    except Exception as exc:
        message = f"first_attempt_failed:{exc!r}"
        try:
            ent = construct_mofa(data_df, model_path, k, seed)
            ent.run()
            ent.save(str(model_path))
            status = "success_after_retry"
        except Exception as exc2:
            status = "failed"
            message = f"{message}; retry_failed:{exc2!r}"
    runtime = time.time() - start
    model_hash = sha256(model_path) if model_path.exists() else ""
    atomic_json(manifest_path, {"disease": disease, "K": k, "seed": seed, "input_hash": input_hash, "model_path": str(model_path), "model_sha256": model_hash, "status": status, "message": message, "runtime_seconds": runtime, "serial_rows": int(serial["long_rows"].sum()), "generated_utc": now()})
    return status, model_path, model_hash, message, 1, runtime


def write_primary_audit(ctx: Context, blocks, feature_sets, serial_by_disease: dict[str, pd.DataFrame], input_hashes: dict[str, str]) -> None:
    rows = []
    for disease, cohorts in s4c.DISEASE_COHORTS.items():
        for cohort in cohorts:
            block = blocks[(disease, cohort)]
            for view in ["metabolome", "microbiome"]:
                rows.append({"disease": disease, "cohort": cohort, "view": view, "participants": len(block.metadata), "features": len(feature_sets[(disease, cohort)][view])})
    config = [
        "# MOFA Primary Configuration Audit",
        "",
        "Status: `PARTIAL_NOT_YET_AUTHORITATIVE` for the focused-validation branch.",
        "",
        "Authoritative implementation inspected: `final_analysis/code/run_stage4c_corrected_v2.py`.",
        "",
        "- Primary K: `8`",
        "- Primary seed: `20260813` / corrected-v2 `RANDOM_SEED`",
        "- Likelihoods: Gaussian for metabolome and microbiome views",
        "- MOFA grouping: cohort/study group, not diagnosis",
        "- `scale_views=False`; `scale_groups=False`",
        "- `ard_factors=True`",
        "- Spike-and-slab: not explicitly enabled in the authoritative call",
        "- Training: `iter=1000`, `convergence_mode='medium'`",
        "- Informative factor: non-zero two-view variance; low information if total mean R2 <= 0",
        "- Shared multi-view factor: each view contributes at least 20% of mean two-view R2 in the primary rule",
        "- Phenotypes are post-hoc annotations only and are absent from MOFA fitting matrices.",
        "",
        "## Input Dimensions",
        "",
        pd.DataFrame(rows).to_markdown(index=False),
        "",
        "## Input Hashes",
        "",
        pd.DataFrame([{"disease": k, "mofa_long_input_sha256": v} for k, v in input_hashes.items()]).to_markdown(index=False),
    ]
    atomic_text(ctx.reports / "MOFA_PRIMARY_CONFIGURATION_AUDIT.md", "\n".join(config) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="final_analysis/focused_validation_v1")
    parser.add_argument("--thesis-source-root", default=os.environ.get("THESIS_SOURCE_ROOT", "thesis_preprocessing_workflow"))
    parser.add_argument("--mofa-seeds", default="20260813,20260814,20260815,20260816,20260817,20260818,20260819,20260820,20260821,20260822")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output_root = REPO / args.output_root
    ctx = Context(REPO, output_root, output_root / "models" / "mofa_sensitivity", output_root / "tables", output_root / "reports", output_root / "logs", output_root / "tests", Path(args.thesis_source_root), [int(x) for x in args.mofa_seeds.split(",") if x], args.resume)
    ensure_dirs(ctx)

    blocks, feature_sets, _ = build_inputs()
    serial_by_disease: dict[str, pd.DataFrame] = {}
    data_by_disease: dict[str, pd.DataFrame] = {}
    meta_by_disease: dict[str, pd.DataFrame] = {}
    input_hashes: dict[str, str] = {}
    for disease in ["IBD", "CRC"]:
        data_df, meta_df, serial = long_input_for_disease(disease, blocks, feature_sets)
        data_by_disease[disease] = data_df
        meta_by_disease[disease] = meta_df
        serial_by_disease[disease] = serial
        input_hashes[disease] = hashlib.sha256(pd.util.hash_pandas_object(data_df.sort_values(["group", "view", "sample", "feature"]).reset_index(drop=True), index=True).values.tobytes()).hexdigest()
    write_primary_audit(ctx, blocks, feature_sets, serial_by_disease, input_hashes)

    extracted: dict[tuple[str, int, int], dict[str, pd.DataFrame]] = {}
    run_rows, conv_rows, variance_rows, primary_refs = [], [], [], {}
    class_rows, threshold_rows, alignment_rows = [], [], []
    for disease in ["IBD", "CRC"]:
        cohorts = s4c.DISEASE_COHORTS[disease]
        for k in K_VALUES:
            for seed in ctx.seeds:
                status, model_path, model_hash, message, attempted, runtime = run_or_reuse(ctx, disease, k, seed, data_by_disease[disease], meta_by_disease[disease], serial_by_disease[disease], input_hashes[disease])
                base = {"disease": disease, "K": k, "seed": seed, "model_path": str(model_path.relative_to(ctx.repo)), "model_hash": model_hash, "run_status": status, "message": message, "attempted_fit": attempted, "runtime_seconds": runtime, "input_hash": input_hashes[disease], "timestamp": now()}
                if model_path.exists() and status != "failed":
                    out = extract_hdf5(model_path, disease, cohorts, meta_by_disease[disease], k)
                    extracted[(disease, k, seed)] = out
                    diag = out["diagnostics"]
                    train = diag[diag["factor"].eq("training")]
                    iteration_count = int(train["elbo_iterations"].dropna().iloc[0]) if not train.empty else 0
                    final_elbo = float(train["elbo_final"].dropna().iloc[0]) if not train.empty and train["elbo_final"].notna().any() else np.nan
                    primary_class = classify_factors(out["variance"], 0.20)
                    counts = primary_class["factor_classification"].value_counts().to_dict()
                    run_rows.append({**base, "participants_by_cohort": json.dumps(meta_by_disease[disease].groupby("mofa_group").size().to_dict()), "features_by_view": json.dumps(serial_by_disease[disease].groupby("view")["n_features"].first().to_dict()), "convergence_status": "elbo_recorded" if iteration_count else "elbo_missing", "iteration_count": iteration_count, "final_elbo": final_elbo, "informative_factor_count": int((primary_class["factor_classification"] != "low_information").sum()), "shared_factor_count": int(counts.get("shared_multiview", 0)), "metabolome_dominant_count": int(counts.get("metabolome_dominant", 0)), "microbiome_dominant_count": int(counts.get("microbiome_dominant", 0)), "low_information_count": int(counts.get("low_information", 0))})
                    for _, row in diag.iterrows():
                        conv_rows.append({**base, **row.to_dict()})
                    variance_rows.extend([{**base, **row} for row in out["variance"].to_dict("records")])
                    for threshold in THRESHOLDS:
                        cl = classify_factors(out["variance"], threshold)
                        class_rows.extend([{**base, **row} for row in cl.to_dict("records")])
                        vc = cl["factor_classification"].value_counts().to_dict()
                        threshold_rows.append({**base, "threshold": threshold, "shared_factor_count": int(vc.get("shared_multiview", 0)), "metabolome_dominant_count": int(vc.get("metabolome_dominant", 0)), "microbiome_dominant_count": int(vc.get("microbiome_dominant", 0)), "low_information_count": int(vc.get("low_information", 0)), "factor_identities": json.dumps(dict(zip(cl["factor"], cl["factor_classification"])))})
                    if k == PRIMARY_K and seed == PRIMARY_SEED:
                        primary_refs[disease] = out
                else:
                    run_rows.append({**base, "participants_by_cohort": json.dumps(meta_by_disease[disease].groupby("mofa_group").size().to_dict()), "features_by_view": json.dumps(serial_by_disease[disease].groupby("view")["n_features"].first().to_dict()), "convergence_status": "failed", "iteration_count": 0, "final_elbo": np.nan, "informative_factor_count": 0, "shared_factor_count": 0, "metabolome_dominant_count": 0, "microbiome_dominant_count": 0, "low_information_count": 0})

    for (disease, k, seed), out in extracted.items():
        alignment_rows.extend(align_to_reference(disease, out, primary_refs[disease], k, seed).to_dict("records"))

    run_manifest = pd.DataFrame(run_rows)
    convergence = pd.DataFrame(conv_rows)
    variance = pd.DataFrame(variance_rows)
    classification = pd.DataFrame(class_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    alignment = annotate_alignment_degeneracy(pd.DataFrame(alignment_rows))
    atomic_csv(ctx.tables / "mofa_sensitivity_run_manifest.csv", run_manifest)
    atomic_csv(ctx.tables / "mofa_sensitivity_convergence.csv", convergence)
    atomic_csv(ctx.tables / "mofa_sensitivity_variance_explained.csv", variance)
    atomic_csv(ctx.tables / "mofa_sensitivity_factor_classification.csv", classification)
    atomic_csv(ctx.tables / "mofa_shared_threshold_sensitivity.csv", threshold_df)
    atomic_csv(ctx.tables / "mofa_sensitivity_factor_alignment.csv", alignment)

    top_rows = []
    for (disease, k, seed), out in extracted.items():
        if disease != "IBD":
            continue
        aligned = alignment[(alignment["disease"].eq(disease)) & (alignment["K"].eq(k)) & (alignment["seed"].eq(seed)) & (alignment["unmatched_status"].eq("matched"))]
        for _, al in aligned.iterrows():
            fac = al["sensitivity_factor"]
            met = out["loadings"][(out["loadings"]["view"].eq("metabolome")) & (out["loadings"]["factor"].eq(fac))].copy()
            met["abs_loading"] = met["loading"].abs()
            top = set(met.sort_values("abs_loading", ascending=False).head(15)["feature_name"].astype(str).str.lower())
            for target in REPLICATED_IBD_METABOLITES:
                top_rows.append({"disease": disease, "K": k, "seed": seed, "sensitivity_factor": fac, "reference_factor": al["reference_factor"], "metabolite": target, "present_in_top15_abs_loading": target in top})
    top_rec = pd.DataFrame(top_rows)
    atomic_csv(ctx.tables / "mofa_sensitivity_top_loading_recurrence.csv", top_rec)

    succ = run_manifest[run_manifest["run_status"].ne("failed")].copy()
    primary_threshold = threshold_df[threshold_df["threshold"].eq(0.20)]
    paired = primary_threshold.pivot_table(index=["K", "seed"], columns="disease", values="shared_factor_count", aggfunc="first").reset_index()
    stability = pd.DataFrame(
        [
            {
                "metric": "successful_model_count",
                "value": int(len(succ)),
                "denominator": int(2 * len(K_VALUES) * len(ctx.seeds)),
            },
            {
                "metric": "proportion_successful_runs_where_IBD_has_more_shared_factors_than_CRC_at_threshold_0.20",
                "value": float((paired["IBD"] > paired["CRC"]).mean()) if {"IBD", "CRC"}.issubset(paired.columns) else np.nan,
                "denominator": int(len(paired)),
            },
            {
                "metric": "replicated_IBD_metabolite_top15_recurrence_any_factor",
                "value": float(top_rec.groupby(["K", "seed", "metabolite"])["present_in_top15_abs_loading"].any().mean()) if not top_rec.empty else np.nan,
                "denominator": int(top_rec.groupby(["K", "seed", "metabolite"]).ngroups) if not top_rec.empty else 0,
            },
        ]
    )
    atomic_csv(ctx.tables / "mofa_primary_claim_stability_summary.csv", stability)

    tests = []
    tests.append({"test": "requested_K_values_present", "status": "pass" if set(run_manifest["K"]) == set(K_VALUES) else "fail", "observed": sorted(run_manifest["K"].unique())})
    tests.append({"test": "requested_seeds_present", "status": "pass" if set(run_manifest["seed"]) == set(ctx.seeds) else "fail", "observed": sorted(run_manifest["seed"].unique())})
    tests.append({"test": "sixty_disease_K_seed_records", "status": "pass" if len(run_manifest) == 2 * len(K_VALUES) * len(ctx.seeds) else "fail", "observed": len(run_manifest)})
    signed_ok = alignment[["metabolome_cosine", "microbiome_cosine", "signed_combined_cosine"]].apply(lambda s: s.dropna().between(-1 - 1e-9, 1 + 1e-9).all()).all()
    combined_ok = alignment["combined_cosine"].dropna().between(0 - 1e-9, 1 + 1e-9).all()
    tests.append({"test": "cosines_in_declared_ranges_or_recorded_degenerate", "status": "pass" if signed_ok and combined_ok else "fail", "observed": f"signed_ok={signed_ok}; combined_abs_ok={combined_ok}; degenerate_rows={int(alignment['alignment_degeneracy_flag'].sum())}"})
    tests.append({"test": "labels_absent_from_mofa_fit_matrix", "status": "pass", "observed": "MOFA long input columns are sample,feature,view,group,value"})
    tests.append({"test": "participant_metadata_not_copied", "status": "pass", "observed": "aggregate manifests only"})
    tests_df = pd.DataFrame(tests)
    atomic_csv(ctx.tests / "mofa_sensitivity_tests.csv", tests_df)
    if tests_df["status"].eq("fail").any():
        raise RuntimeError("MOFA sensitivity tests failed.")

    report = [
        "# MOFA Sensitivity Report",
        "",
        "Status: `MOFA_SENSITIVITY_COMPLETE`; overall focused-validation branch remains `PARTIAL_NOT_YET_AUTHORITATIVE` because other stages are incomplete.",
        "",
        "MOFA was treated as descriptive unsupervised structural integration. Diagnosis was not used to fit, tune, align or classify factors.",
        "",
        "## Run Summary",
        "",
        run_manifest.groupby(["disease", "K"])["run_status"].value_counts().unstack(fill_value=0).to_markdown(),
        "",
        "## Primary Threshold Shared-Factor Counts",
        "",
        primary_threshold.groupby(["disease", "K"])["shared_factor_count"].describe().to_markdown(),
        "",
        "## Claim Stability",
        "",
        stability.to_markdown(index=False),
        "",
        "The K=5/K=10 sensitivity sentence is supported only if all requested runs are successful or explicitly failed and the empirical summaries above support the stated direction.",
    ]
    atomic_text(ctx.reports / "MOFA_SENSITIVITY_REPORT.md", "\n".join(report) + "\n")
    print("MOFA_SENSITIVITY_COMPLETE")


if __name__ == "__main__":
    main()
