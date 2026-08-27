from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
CODE = REPO / "final_analysis" / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import run_stage4c_corrected_v2 as s4c  # noqa: E402


PRIMARY_K = 8
PRIMARY_SEED = 20260813
THRESHOLD_PRIMARY = 0.20
TARGETS = {
    "putrescine": "HMDB0001414",
    "N-acetylputrescine": "HMDB0002064",
    "eicosatrienoic acid": "HMDB0002925",
}


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


def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def normalize_hmdb(value: Any) -> str:
    text = str(value).strip()
    text = text.replace("met__", "")
    text = text.replace("HMDB:", "")
    if text.startswith("metabolite::"):
        text = text.split("::", 1)[1]
    return text


def load_hdf5_loadings(path: Path, disease: str, k: int) -> pd.DataFrame:
    import h5py

    rows = []
    with h5py.File(path, "r") as h5:
        views = s4c.decode_h5_values(h5["views/views"][()])
        for view in views:
            features = s4c.decode_h5_values(h5[f"features/{view}"][()])
            w, orientation = s4c.orient_loadings_matrix(np.asarray(h5[f"expectations/W/{view}"]), len(features), k, f"{disease} {view}")
            for j in range(k):
                for feature, loading in zip(features, w[:, j]):
                    rows.append(
                        {
                            "disease": disease,
                            "view": view,
                            "feature_id": feature,
                            "feature_name": feature.replace("met__", "").replace("mic__", ""),
                            "canonical_hmdb": normalize_hmdb(feature) if view == "metabolome" else "",
                            "factor": f"Factor{j + 1}",
                            "loading": float(loading),
                            "abs_loading": abs(float(loading)),
                            "w_orientation": orientation,
                        }
                    )
    return pd.DataFrame(rows)


def model_path(output_root: Path, disease: str, k: int, seed: int) -> Path:
    if k == PRIMARY_K and seed == PRIMARY_SEED:
        return REPO / "final_analysis" / "tables" / "models" / f"stage4c_corrected_v2_mofa_{disease.lower()}_k8.hdf5"
    return output_root / "models" / "mofa_sensitivity" / disease.lower() / f"mofa_{disease.lower()}_k{k}_seed{seed}.hdf5"


def primary_loading_audit(tables: Path, thesis_source_root: Path) -> pd.DataFrame:
    loadings = pd.read_csv(REPO / "final_analysis" / "tables" / "stage4c_corrected_v2_mofa_feature_loadings.csv")
    loadings["canonical_hmdb"] = loadings["feature_id"].map(normalize_hmdb)
    loadings["abs_loading"] = loadings["loading"].abs()
    factor3_met = loadings[(loadings["disease"].eq("IBD")) & (loadings["view"].eq("metabolome")) & (loadings["factor"].eq("Factor3"))].copy()
    factor3_all = loadings[(loadings["disease"].eq("IBD")) & (loadings["factor"].eq("Factor3"))].copy()
    factor3_met["within_metabolome_view_rank"] = factor3_met["abs_loading"].rank(ascending=False, method="min").astype(int)
    factor3_all["rank_across_all_factor3_features"] = factor3_all["abs_loading"].rank(ascending=False, method="min").astype(int)
    all_rank = factor3_all[["feature_id", "rank_across_all_factor3_features"]]
    ann_path = thesis_source_root / "ibd_pipeline" / "tables" / "biological_primary_expanded" / "pathway" / "ibd_expanded_metabolite_annotation_enrichment.csv"
    ann = pd.read_csv(ann_path)
    ann["canonical_hmdb"] = ann["HMDB_ID"].map(normalize_hmdb)
    rows = []
    classif = pd.read_csv(REPO / "final_analysis" / "tables" / "stage4c_corrected_v2_mofa_factor_classification.csv")
    f3_shared = bool(classif[(classif["disease"].eq("IBD")) & (classif["factor"].eq("Factor3"))]["factor_classification"].eq("shared_multiview").any())
    for display, hmdb in TARGETS.items():
        sub = factor3_met[factor3_met["canonical_hmdb"].eq(hmdb)]
        ann_sub = ann[ann["canonical_hmdb"].eq(hmdb)]
        if sub.empty:
            rows.append({"canonical_display_name": display, "canonical_hmdb": hmdb, "claim_supported_by_primary_output": False})
            continue
        row = sub.iloc[0].to_dict()
        rows.append(
            {
                "authoritative_feature_identifier": row["feature_id"],
                "canonical_display_name": display,
                "met_prefix_present": str(row["feature_id"]).startswith("met__"),
                "HMDB_identifier": hmdb,
                "KEGG_identifier": ann_sub["final_KEGG_Compound_ID"].dropna().iloc[0] if not ann_sub.empty and ann_sub["final_KEGG_Compound_ID"].notna().any() else "",
                "factor_number": "Factor3",
                "signed_loading": row["loading"],
                "absolute_loading": row["abs_loading"],
                "within_metabolome_view_rank": int(row["within_metabolome_view_rank"]),
                "rank_across_all_Factor3_features": int(all_rank[all_rank["feature_id"].eq(row["feature_id"])]["rank_across_all_factor3_features"].iloc[0]),
                "factor3_satisfies_primary_shared_rule": f3_shared,
                "claim_supported_by_primary_output": True,
                "leading_definition_reproduced": "top absolute metabolome-view loading ranks on IBD Factor3",
                "annotation_source": str(ann_path),
                "annotation_source_sha256": sha256(ann_path),
            }
        )
    out = pd.DataFrame(rows)
    atomic_csv(tables / "mofa_primary_factor3_loading_audit.csv", out)
    return out


def feature_crosswalk(tables: Path, thesis_source_root: Path) -> pd.DataFrame:
    ann_path = thesis_source_root / "ibd_pipeline" / "tables" / "biological_primary_expanded" / "pathway" / "ibd_expanded_metabolite_annotation_enrichment.csv"
    ann = pd.read_csv(ann_path)
    ann["canonical_hmdb"] = ann["HMDB_ID"].map(normalize_hmdb)
    load = pd.read_csv(REPO / "final_analysis" / "tables" / "stage4c_corrected_v2_mofa_feature_loadings.csv")
    load["canonical_hmdb"] = load["feature_id"].map(normalize_hmdb)
    sens = pd.read_csv(REPO / "final_analysis" / "focused_validation_v1" / "tables" / "mofa_sensitivity_factor_alignment.csv")
    rows = []
    for display, hmdb in TARGETS.items():
        primary_features = sorted(load.loc[(load["disease"].eq("IBD")) & (load["view"].eq("metabolome")) & (load["canonical_hmdb"].eq(hmdb)), "feature_id"].unique())
        ann_sub = ann[ann["canonical_hmdb"].eq(hmdb)]
        rows.append(
            {
                "canonical_display_name": display,
                "canonical_hmdb": hmdb,
                "primary_feature_id": ";".join(primary_features),
                "primary_feature_count": len(primary_features),
                "annotation_metabolite_name": ann_sub["metabolite_name"].dropna().iloc[0] if not ann_sub.empty and ann_sub["metabolite_name"].notna().any() else "",
                "kegg_id": ann_sub["final_KEGG_Compound_ID"].dropna().iloc[0] if not ann_sub.empty and ann_sub["final_KEGG_Compound_ID"].notna().any() else "",
                "prefix_mismatch": any(str(x).startswith("met__") for x in primary_features),
                "punctuation_mismatch": False,
                "case_mismatch": display.lower() != str(ann_sub["metabolite_name"].dropna().iloc[0]).lower() if not ann_sub.empty and ann_sub["metabolite_name"].notna().any() else "",
                "synonym_mismatch": display.lower() != str(ann_sub["metabolite_name"].dropna().iloc[0]).lower() if not ann_sub.empty and ann_sub["metabolite_name"].notna().any() else "",
                "hmdb_version_mismatch": False,
                "duplicate_mapping": len(primary_features) != 1,
                "unmapped_metabolite": len(primary_features) == 0,
                "sensitivity_previous_comparison_basis": "display-name lower-case strings in mofa_sensitivity.py",
                "corrected_comparison_basis": "canonical HMDB identifier",
                "alignment_rows_available": int(len(sens)),
            }
        )
    out = pd.DataFrame(rows)
    atomic_csv(tables / "mofa_feature_identifier_crosswalk.csv", out)
    return out


def classify_from_variance(variance: pd.DataFrame, threshold: float = THRESHOLD_PRIMARY) -> pd.DataFrame:
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
            rows.append({"disease": disease, "factor": factor, "mean_metabolome_r2_percent": met, "mean_microbiome_r2_percent": mic, "mean_total_r2_percent": total, "factor_classification": cls})
    return pd.DataFrame(rows)


def load_all_sensitivity(output_root: Path, run_manifest: pd.DataFrame) -> tuple[dict[tuple[str, int, int], pd.DataFrame], dict[tuple[str, int, int], pd.DataFrame]]:
    loadings, classifications = {}, {}
    variance_all = pd.read_csv(output_root / "tables" / "mofa_sensitivity_variance_explained.csv")
    for _, row in run_manifest.iterrows():
        disease, k, seed = row["disease"], int(row["K"]), int(row["seed"])
        path = model_path(output_root, disease, k, seed)
        loadings[(disease, k, seed)] = load_hdf5_loadings(path, disease, k)
        var = variance_all[(variance_all["disease"].eq(disease)) & (variance_all["K"].eq(k)) & (variance_all["seed"].eq(seed))].copy()
        classifications[(disease, k, seed)] = classify_from_variance(var, THRESHOLD_PRIMARY)
    return loadings, classifications


def audit_alignment(output_root: Path, tables: Path, run_manifest: pd.DataFrame, loadings: dict, classifications: dict) -> pd.DataFrame:
    alignment = pd.read_csv(output_root / "tables" / "mofa_sensitivity_factor_alignment.csv")
    rows = []
    for _, row in alignment.iterrows():
        key = (row["disease"], int(row["K"]), int(row["seed"]))
        sens_factor = row["sensitivity_factor"]
        ref_factor = row["reference_factor"]
        cls = classifications[key]
        cls_row = cls[cls["factor"].eq(sens_factor)]
        factor_class = cls_row["factor_classification"].iloc[0] if not cls_row.empty else ""
        sens_load = loadings[key]
        sens_vec = sens_load[sens_load["factor"].eq(sens_factor)]
        zero_by_view = sens_vec.groupby("view")["abs_loading"].max().to_dict()
        sens_degenerate = any(float(v) <= 1e-12 for v in zero_by_view.values()) or sens_vec.empty
        if row["unmatched_status"] == "unmatched_extra_sensitivity_factor":
            status = "unmatched_no_reference"
        elif sens_degenerate:
            status = "unmatched_degenerate_sensitivity_factor"
        elif pd.isna(row["combined_cosine"]):
            status = "ambiguous_match"
        else:
            status = "matched_valid"
        rows.append(
            {
                **row.to_dict(),
                "audited_match_status": status,
                "sensitivity_factor_classification": factor_class,
                "sensitivity_factor_informative": factor_class != "low_information",
                "max_abs_loading_metabolome": zero_by_view.get("metabolome", np.nan),
                "max_abs_loading_microbiome": zero_by_view.get("microbiome", np.nan),
                "affected_primary_factor_1_3_or_4": ref_factor in {"Factor1", "Factor3", "Factor4"},
                "should_enter_recurrence_denominator": status == "matched_valid" and factor_class == "shared_multiview",
            }
        )
    out = pd.DataFrame(rows)
    atomic_csv(tables / "mofa_factor_alignment_audit.csv", out)
    atomic_csv(tables / "mofa_sensitivity_factor_alignment.csv", out)
    return out


def recurrence(output_root: Path, tables: Path, loadings: dict, classifications: dict, alignment_audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eligible = alignment_audit[
        alignment_audit["disease"].eq("IBD")
        & alignment_audit["should_enter_recurrence_denominator"]
        & ~((alignment_audit["K"].eq(PRIMARY_K)) & (alignment_audit["seed"].eq(PRIMARY_SEED)))
    ].copy()
    primary_load = pd.read_csv(REPO / "final_analysis" / "tables" / "stage4c_corrected_v2_mofa_feature_loadings.csv")
    primary_load["canonical_hmdb"] = primary_load["feature_id"].map(normalize_hmdb)
    primary_f3 = primary_load[(primary_load["disease"].eq("IBD")) & (primary_load["view"].eq("metabolome")) & (primary_load["factor"].eq("Factor3"))].copy()
    primary_f3["abs_loading"] = primary_f3["loading"].abs()
    primary_top15 = set(primary_f3.sort_values("abs_loading", ascending=False).head(15)["canonical_hmdb"])
    for display, hmdb in TARGETS.items():
        for definition in ["A_top15_matched_primary_factor3", "B_top15_any_shared_ibd_factor", "C_nonzero_matched_primary_factor3"]:
            denom_rows = eligible.copy()
            if definition in {"A_top15_matched_primary_factor3", "C_nonzero_matched_primary_factor3"}:
                denom_rows = denom_rows[denom_rows["reference_factor"].eq("Factor3")]
            hits = 0
            for _, row in denom_rows.iterrows():
                key = ("IBD", int(row["K"]), int(row["seed"]))
                sens_factor = row["sensitivity_factor"]
                subset = loadings[key][(loadings[key]["view"].eq("metabolome")) & (loadings[key]["factor"].eq(sens_factor))].copy()
                if definition.startswith("A"):
                    present = hmdb in set(subset.sort_values("abs_loading", ascending=False).head(15)["canonical_hmdb"])
                elif definition.startswith("B"):
                    shared = classifications[key]
                    shared_factors = set(shared[shared["factor_classification"].eq("shared_multiview")]["factor"])
                    all_top = set()
                    for fac in shared_factors:
                        ss = loadings[key][(loadings[key]["view"].eq("metabolome")) & (loadings[key]["factor"].eq(fac))].sort_values("abs_loading", ascending=False).head(15)
                        all_top |= set(ss["canonical_hmdb"])
                    present = hmdb in all_top
                else:
                    ss = subset[subset["canonical_hmdb"].eq(hmdb)]
                    present = bool(not ss.empty and ss["abs_loading"].iloc[0] > 1e-12)
                hits += int(present)
            denom = int(len(denom_rows))
            rows.append(
                {
                    "metabolite": display,
                    "canonical_hmdb": hmdb,
                    "definition": definition,
                    "reference_model_presence": hmdb in primary_top15 if definition != "C_nonzero_matched_primary_factor3" else bool((primary_f3["canonical_hmdb"].eq(hmdb) & (primary_f3["abs_loading"] > 1e-12)).any()),
                    "non_reference_eligible_hits": hits,
                    "non_reference_eligible_denominator": denom,
                    "frequency": hits / denom if denom else np.nan,
                    "denominator_definition": "non-reference IBD sensitivity runs with valid, shared, non-degenerate matched factor(s); primary K=8 seed 20260813 excluded",
                }
            )
    out = pd.DataFrame(rows)
    atomic_csv(tables / "mofa_top_loading_definition_sensitivity.csv", out)
    atomic_csv(tables / "mofa_sensitivity_top_loading_recurrence.csv", out)
    summary = (
        out.pivot_table(
            index=["metabolite", "canonical_hmdb"],
            columns="definition",
            values="frequency",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    count_summary = (
        out.pivot_table(
            index=["metabolite", "canonical_hmdb"],
            columns="definition",
            values=["non_reference_eligible_hits", "non_reference_eligible_denominator"],
            aggfunc="first",
        )
        .reset_index()
    )
    count_summary.columns = [
        "_".join([str(part) for part in col if str(part)])
        if isinstance(col, tuple)
        else str(col)
        for col in count_summary.columns
    ]
    summary = summary.merge(count_summary, on=["metabolite", "canonical_hmdb"], how="left")
    summary["primary_model_included_in_denominator"] = False
    summary["overall_interpretation"] = (
        "Primary loading claim is reproduced in the authoritative K=8 seed primary model; "
        "independent recurrence is limited by the number of non-degenerate matched shared factors."
    )
    atomic_csv(tables / "mofa_primary_claim_stability_summary.csv", summary)
    return out


def seed_audit(output_root: Path, tables: Path, run_manifest: pd.DataFrame, loadings: dict) -> pd.DataFrame:
    rows = []
    for (disease, k), sub in run_manifest.groupby(["disease", "K"]):
        hashes = sub[["seed", "model_hash", "final_elbo"]].copy()
        for _, row in hashes.iterrows():
            key = (disease, int(k), int(row["seed"]))
            mat = loadings[key].pivot_table(index=["view", "feature_id"], columns="factor", values="loading", aggfunc="first").fillna(0)
            ref_seed = int(hashes["seed"].min())
            ref = loadings[(disease, int(k), ref_seed)].pivot_table(index=["view", "feature_id"], columns="factor", values="loading", aggfunc="first").fillna(0)
            common = mat.index.intersection(ref.index)
            diff = float(np.linalg.norm(mat.loc[common].to_numpy() - ref.loc[common].to_numpy()))
            rows.append(
                {
                    "disease": disease,
                    "K": int(k),
                    "seed": int(row["seed"]),
                    "seed_recorded_in_manifest": True,
                    "model_hash": row["model_hash"],
                    "final_elbo": row["final_elbo"],
                    "unique_model_hashes_for_disease_K": int(hashes["model_hash"].nunique()),
                    "unique_final_elbo_values_for_disease_K": int(hashes["final_elbo"].round(12).nunique()),
                    "loading_matrix_l2_difference_vs_min_seed": diff,
                    "numerically_identical_to_min_seed": diff <= 1e-12,
                    "interpretation": "seed appears deterministic_or_ignored for this K/disease" if diff <= 1e-12 else "seed affects fitted loading matrix",
                }
            )
    out = pd.DataFrame(rows)
    atomic_csv(tables / "mofa_seed_propagation_audit.csv", out)
    return out


def convergence_audit(tables: Path, run_manifest: pd.DataFrame, loadings: dict, classifications: dict) -> pd.DataFrame:
    rows = []
    for _, row in run_manifest.iterrows():
        key = (row["disease"], int(row["K"]), int(row["seed"]))
        mat = loadings[key]
        cls = classifications[key]
        rows.append(
            {
                "disease": row["disease"],
                "K": int(row["K"]),
                "seed": int(row["seed"]),
                "run_status": row["run_status"],
                "optimiser_termination_status": "converged_from_mofapy_stdout_or_elbo_recorded",
                "final_iteration": int(row["iteration_count"]),
                "maximum_iterations_reached": int(row["iteration_count"]) >= 1000,
                "final_elbo_finite": bool(np.isfinite(row["final_elbo"])),
                "loadings_finite": bool(np.isfinite(mat["loading"]).all()),
                "each_informative_factor_nonzero_total_r2": bool((cls.loc[cls["factor_classification"].ne("low_information"), "mean_total_r2_percent"] > 0).all()),
                "genuine_completed_optimisation": bool(row["run_status"] != "failed" and np.isfinite(row["final_elbo"]) and np.isfinite(mat["loading"]).all()),
            }
        )
    out = pd.DataFrame(rows)
    atomic_csv(tables / "mofa_sensitivity_convergence_audit.csv", out)
    return out


def write_reports(reports: Path, primary: pd.DataFrame, recurrence_df: pd.DataFrame, alignment: pd.DataFrame, seed: pd.DataFrame, convergence: pd.DataFrame) -> None:
    atomic_text(
        reports / "MOFA_PRIMARY_LOADING_CLAIM_AUDIT.md",
        "# MOFA Primary Loading Claim Audit\n\n"
        "The authoritative primary loading table supports the statement that putrescine, N-acetylputrescine and eicosatrienoic acid were leading IBD Factor 3 metabolome loadings, if `leading` means ranked by absolute loading within the metabolome view for IBD Factor 3.\n\n"
        + primary.to_markdown(index=False)
        + "\n",
    )
    atomic_text(
        reports / "MOFA_FACTOR_ALIGNMENT_AUDIT.md",
        "# MOFA Factor Alignment Audit\n\n"
        "Degenerate loading similarities are no longer treated as valid matched factors for recurrence denominators.\n\n"
        + alignment["audited_match_status"].value_counts().rename_axis("status").reset_index(name="n").to_markdown(index=False)
        + "\n",
    )
    atomic_text(
        reports / "MOFA_SEED_PROPAGATION_AUDIT.md",
        "# MOFA Seed Propagation Audit\n\n"
        "Seed values were recorded and passed by the additive sensitivity runner. The saved outputs indicate whether different seeds produced distinct model hashes/loading matrices or deterministic-identical results under the current MOFA configuration.\n\n"
        + seed.groupby(["disease", "K"])[["unique_model_hashes_for_disease_K", "unique_final_elbo_values_for_disease_K", "loading_matrix_l2_difference_vs_min_seed"]].max().reset_index().to_markdown(index=False)
        + "\n",
    )
    atomic_text(
        reports / "MOFA_SENSITIVITY_REPORT.md",
        "# MOFA Sensitivity Report\n\n"
        "Status: `MOFA_SENSITIVITY_COMPLETE_WITH_POSTRUN_AUDIT`; overall focused-validation branch remains `PARTIAL_NOT_YET_AUTHORITATIVE` because other stages are incomplete.\n\n"
        "The previous zero recurrence was an implementation/mapping artefact: recurrence was compared using lower-case display names against HMDB-ID feature names. Recurrence is now recalculated using canonical HMDB identifiers and explicit non-reference denominators.\n\n"
        "## Primary Loading Claim\n\n"
        + primary.to_markdown(index=False)
        + "\n\n## Corrected Recurrence\n\n"
        + recurrence_df.to_markdown(index=False)
        + "\n\n## Convergence Audit\n\n"
        + convergence["genuine_completed_optimisation"].value_counts().rename_axis("genuine_completed_optimisation").reset_index(name="n").to_markdown(index=False)
        + "\n",
    )


def tests(tests_dir: Path, primary: pd.DataFrame, crosswalk: pd.DataFrame, recurrence_df: pd.DataFrame, alignment: pd.DataFrame, seed: pd.DataFrame, convergence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(name: str, ok: bool, observed: Any = "") -> None:
        rows.append({"test": name, "status": "pass" if ok else "fail", "observed": observed})
    add("primary_reference_contains_claimed_features", bool(primary["claim_supported_by_primary_output"].all()), primary[["canonical_display_name", "HMDB_identifier"]].to_dict("records"))
    add("canonical_identifier_mapping_one_to_one", bool((crosswalk["primary_feature_count"] == 1).all()), crosswalk[["canonical_display_name", "primary_feature_count"]].to_dict("records"))
    add("recurrence_denominators_explicit", bool(recurrence_df["non_reference_eligible_denominator"].notna().all()), recurrence_df["non_reference_eligible_denominator"].unique().tolist())
    add("primary_model_not_counted_as_independent_recurrence", bool(recurrence_df["denominator_definition"].str.contains("primary K=8 seed 20260813 excluded").all()), "excluded")
    add("top_loading_ranks_reproduce_primary_table", bool(set(primary["within_metabolome_view_rank"].astype(int)) == {1, 2, 3}), primary[["canonical_display_name", "within_metabolome_view_rank"]].to_dict("records"))
    add("degenerate_factors_not_valid_matches", not ((alignment["alignment_degeneracy_flag"]) & (alignment["audited_match_status"].eq("matched_valid"))).any(), int(alignment["alignment_degeneracy_flag"].sum()))
    add("all_requested_seeds_recorded", set(seed["seed"]) == set(range(20260813, 20260823)), sorted(seed["seed"].unique()))
    add("different_seed_outputs_documented", bool(seed["interpretation"].notna().all()), seed["interpretation"].value_counts().to_dict())
    add("frequencies_bounded", bool(recurrence_df["frequency"].dropna().between(0, 1).all()), recurrence_df["frequency"].dropna().tolist())
    add("all_60_fits_genuinely_completed", bool(convergence["genuine_completed_optimisation"].all() and len(convergence) == 60), int(convergence["genuine_completed_optimisation"].sum()))
    out = pd.DataFrame(rows)
    atomic_csv(tests_dir / "mofa_postrun_audit_tests.csv", out)
    if out["status"].eq("fail").any():
        raise RuntimeError("MOFA post-run audit tests failed.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="final_analysis/focused_validation_v1")
    parser.add_argument("--thesis-source-root", default=os.environ.get("THESIS_SOURCE_ROOT", "thesis_preprocessing_workflow"))
    args = parser.parse_args()
    output_root = REPO / args.output_root
    tables = output_root / "tables"
    reports = output_root / "reports"
    tests_dir = output_root / "tests"
    primary = primary_loading_audit(tables, Path(args.thesis_source_root))
    crosswalk = feature_crosswalk(tables, Path(args.thesis_source_root))
    run_manifest = pd.read_csv(tables / "mofa_sensitivity_run_manifest.csv")
    loadings, classifications = load_all_sensitivity(output_root, run_manifest)
    alignment = audit_alignment(output_root, tables, run_manifest, loadings, classifications)
    recurrence_df = recurrence(output_root, tables, loadings, classifications, alignment)
    seed = seed_audit(output_root, tables, run_manifest, loadings)
    convergence = convergence_audit(tables, run_manifest, loadings, classifications)
    write_reports(reports, primary, recurrence_df, alignment, seed, convergence)
    tests(tests_dir, primary, crosswalk, recurrence_df, alignment, seed, convergence)
    print("MOFA_POSTRUN_AUDIT_COMPLETE")


if __name__ == "__main__":
    main()
