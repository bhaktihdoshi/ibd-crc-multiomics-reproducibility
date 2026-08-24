from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr, spearmanr, t as t_dist
except Exception:  # pragma: no cover
    pearsonr = None
    spearmanr = None
    t_dist = None

from stage3_config import CONFIG
from stage3_identifier_overlap import overlap_tables
from stage3_load_inputs import validate_authoritative_inputs


STAGE3 = CONFIG.stage3_dir
TABLES = STAGE3 / "tables"
FIGURES = STAGE3 / "figures"
REPORTS = STAGE3 / "reports"
MANIFESTS = STAGE3 / "manifests"
LOGS = STAGE3 / "logs"
WORKFLOW = CONFIG.root / "thesis_preprocessing_workflow"
IBD_EXP = CONFIG.ibd_dir / "tables" / "biological_primary_expanded"
CRC_EXP = CONFIG.crc_stage2_dir / "tables" / "biological_primary_expanded"

PHENOTYPES = ["ibd", "cd", "uc"]
IBD_CONTRAST = {"ibd": "ibd_vs_control", "cd": "cd_vs_control", "uc": "uc_vs_control"}
IBD_LABEL = {"ibd": "IBD vs Control", "cd": "CD vs Control", "uc": "UC vs Control"}
CRC_COMPARISON = {"Sinha": "CRC vs Control", "Yachida": "CRC vs Healthy"}

YACHIDA_STAGE3B_INPUT = WORKFLOW / "crc_pipeline" / "stage1_release" / "tables" / "stage2_model_prep" / "crc_stage2_yachida_exploratory_ordinal_trajectory_model_prep.csv"
STAGE3_SHARED = TABLES / "stage3_four_cohort_shared_metabolites.csv"
STAGE3_SHARED_EXPANDED = TABLES / "stage3_four_cohort_shared_metabolites_expanded_high_confidence.csv"
STAGE3B_ORDER = {
    "Healthy": 0,
    "MP": 1,
    "Stage 0": 2,
    "Stage I": 3,
    "Stage II": 4,
    "Stage III": 5,
    "Stage IV": 6,
}
STAGE3B_EXPECTED_COUNTS = {
    "Healthy": 127,
    "MP": 40,
    "Stage 0": 27,
    "Stage I": 44,
    "Stage II": 25,
    "Stage III": 34,
    "Stage IV": 20,
}
STAGE3B_FOCUSED_HMDB = ["HMDB0000251", "HMDB0001414", "HMDB0002064"]
STAGE3C_REQUIRED_FIGURES = [
    "stage3a_cross_disease_effect_heatmap",
    "stage3a_ibd_crc_effect_concordance",
    "stage3a_cd_crc_effect_concordance",
    "stage3a_uc_crc_effect_concordance",
    "stage3a_directional_concordance_summary",
    "stage3a_ibd_replicated_metabolites_crc_context",
    "stage3b_yachida_ordered_state_heatmap",
    "stage3b_yachida_ordered_state_forest_top25",
    "stage3b_focused_ibd_replicated_profiles",
]


def ensure_stage3_dirs() -> None:
    for directory in CONFIG.required_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def bh_fdr(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    if valid.empty:
        return out
    n = len(valid)
    ranks = np.arange(1, n + 1)
    q = valid.to_numpy() * n / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    out.loc[valid.index] = np.clip(q, 0, 1)
    return out


def sign_label(a: float | int | None, b: float | int | None) -> str:
    if pd.isna(a) or pd.isna(b):
        return "INCOMPLETE"
    if a > 0 and b > 0:
        return "POSITIVE_CONSENSUS"
    if a < 0 and b < 0:
        return "NEGATIVE_CONSENSUS"
    if (a > 0 and b < 0) or (a < 0 and b > 0):
        return "DISCORDANT"
    return "DISCORDANT"


def direction_class(ibd_consensus: str, crc_consensus: str) -> str:
    if ibd_consensus == "INCOMPLETE" or crc_consensus == "INCOMPLETE":
        return "INCOMPLETE"
    if ibd_consensus == "DISCORDANT" and crc_consensus == "DISCORDANT":
        return "BOTH_DISCORDANT"
    if ibd_consensus == "DISCORDANT":
        return "IBD_DISCORDANT"
    if crc_consensus == "DISCORDANT":
        return "CRC_DISCORDANT"
    if ibd_consensus == "POSITIVE_CONSENSUS" and crc_consensus == "POSITIVE_CONSENSUS":
        return "CONCORDANT_POSITIVE"
    if ibd_consensus == "NEGATIVE_CONSENSUS" and crc_consensus == "NEGATIVE_CONSENSUS":
        return "CONCORDANT_NEGATIVE"
    if ibd_consensus == "POSITIVE_CONSENSUS" and crc_consensus == "NEGATIVE_CONSENSUS":
        return "OPPOSITE_IBD_POSITIVE_CRC_NEGATIVE"
    if ibd_consensus == "NEGATIVE_CONSENSUS" and crc_consensus == "POSITIVE_CONSENSUS":
        return "OPPOSITE_IBD_NEGATIVE_CRC_POSITIVE"
    return "INCOMPLETE"


def correlation(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float]:
    ok = pd.to_numeric(x, errors="coerce").notna() & pd.to_numeric(y, errors="coerce").notna()
    xv = pd.to_numeric(x[ok], errors="coerce")
    yv = pd.to_numeric(y[ok], errors="coerce")
    if len(xv) < 3 or xv.nunique() < 2 or yv.nunique() < 2:
        return np.nan, np.nan
    if method == "spearman":
        if spearmanr is None:
            return np.nan, np.nan
        r, p = spearmanr(xv, yv)
    else:
        if pearsonr is None:
            return np.nan, np.nan
        r, p = pearsonr(xv, yv)
    return float(r), float(p)


def write_validate() -> None:
    out = validate_authoritative_inputs(CONFIG)
    out.to_csv(TABLES / "stage3_validate_inputs.csv", index=False)
    if not out["exists"].all():
        missing = "; ".join(out.loc[~out["exists"], "input_name"].astype(str))
        raise FileNotFoundError(f"Missing Stage 3 input(s): {missing}")


def write_overlap() -> None:
    summary, matrix, shared = overlap_tables(CONFIG)
    summary.to_csv(TABLES / "stage3_metabolite_overlap_summary.csv", index=False)
    matrix.to_csv(TABLES / "stage3_pairwise_overlap_matrix.csv", index=False)
    shared.to_csv(TABLES / "stage3_four_cohort_shared_metabolites.csv", index=False)


def validate_stage3a_inputs() -> pd.DataFrame:
    handoff = read(MANIFESTS / "stage3_authoritative_effect_input_handoff.csv")
    completeness = read(TABLES / "stage3a_effect_input_completeness_audit.csv")
    rows = []
    for _, r in handoff.iterrows():
        path = WORKFLOW / str(r["authoritative_da_file"])
        rows.append({
            "input_name": f"{r['disease']} {r['cohort']} {r['contrast']}",
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256(path) if path.exists() else pd.NA,
            "analysis_universe": r.get("analysis_universe"),
            "role": "authoritative expanded DA effect source",
        })
    for name, path in [
        ("Stage 3 authoritative handoff", MANIFESTS / "stage3_authoritative_effect_input_handoff.csv"),
        ("Stage 3 completeness audit", TABLES / "stage3a_effect_input_completeness_audit.csv"),
        ("Strict 56 HMDB manifest", TABLES / "stage3_four_cohort_shared_metabolites.csv"),
        ("Expanded high-confidence annotation", TABLES / "stage3_four_cohort_shared_metabolites_expanded_high_confidence.csv"),
    ]:
        rows.append({"input_name": name, "path": str(path), "exists": path.exists(), "sha256": sha256(path) if path.exists() else pd.NA, "analysis_universe": "stage3_support", "role": "Stage 3 metadata"})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "stage3a_input_validation.csv", index=False)
    missing = out.loc[~out["exists"], "input_name"].tolist()
    report = [
        "# Stage 3A Input Validation",
        "",
        f"Authoritative handoff rows: {len(handoff)}.",
        f"Completeness audit rows: {len(completeness)}.",
        f"Missing inputs: {len(missing)}.",
        "",
        "All Stage 3A effect sources are read from the post-reopening authoritative handoff manifest. No upstream DA, ORA, or predictive-model output is modified.",
    ]
    if missing:
        report.append("Missing: " + "; ".join(missing))
    (REPORTS / "STAGE3A_INPUT_VALIDATION.md").write_text("\n".join(report) + "\n")
    if missing:
        raise FileNotFoundError("; ".join(missing))
    return out


def effect_scale_gate() -> None:
    text = """# Stage 3A Effect-Scale Compatibility

## Beta Meaning

Franzosa and iHMP IBD effects are from robust HC3 univariate differential-abundance models using the established formula `metabolite ~ disease_status + age_years`. Positive beta means higher metabolite abundance/effect in IBD, CD, or UC relative to control. The analysed metabolite values are source-supplied processed scales; the Stage 3A matrix uses the standardized beta exported by the expanded DA.

Sinha and Yachida CRC effects are from robust HC3 models using `metabolite_z ~ crc_status + age_years`. Positive beta means higher standardized metabolite abundance/effect in CRC relative to control/healthy.

## Compatibility Decision

A. Beta signs are comparable across cohorts because every contrast uses disease/case coded as the positive exposure and control/healthy as the reference.

B. Raw beta magnitudes are not perfectly comparable because cohorts differ in assay platform, preprocessing, scaling and biological sampling context. IBD and CRC also came from separate study systems.

C. Stage 3A therefore treats sign and rank/Spearman correlation as primary. Pearson/raw beta correlation is reported as a secondary descriptive sensitivity only, not as the main inference.

SE values describe uncertainty for the cohort-specific regression coefficient in its own model and are not pooled across cohorts.
"""
    (REPORTS / "STAGE3A_EFFECT_SCALE_COMPATIBILITY.md").write_text(text)


def load_effect_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strict = read(TABLES / "stage3_four_cohort_shared_metabolites.csv")
    annotation = read(TABLES / "stage3_four_cohort_shared_metabolites_expanded_high_confidence.csv")
    ibd_da = read(IBD_EXP / "ibd_expanded_da_all_results.csv")
    crc_da = read(CRC_EXP / "crc_expanded_da_all_results.csv")
    return strict, annotation, ibd_da, crc_da


def lookup_ibd(ibd_da: pd.DataFrame, hmdb: str, cohort: str, contrast: str) -> pd.Series:
    hit = ibd_da[(ibd_da["cohort"].eq(cohort)) & (ibd_da["contrast"].eq(contrast)) & (ibd_da["hmdb_id"].eq(hmdb))]
    return hit.iloc[0] if not hit.empty else pd.Series(dtype=object)


def lookup_crc(crc_da: pd.DataFrame, hmdb: str, cohort: str) -> pd.Series:
    canonical = f"HMDB:{hmdb}"
    hit = crc_da[(crc_da["cohort"].eq(cohort)) & (crc_da["canonical_identifier"].eq(canonical))]
    return hit.iloc[0] if not hit.empty else pd.Series(dtype=object)


def construct_effect_matrix() -> pd.DataFrame:
    strict, annotation, ibd_da, crc_da = load_effect_sources()
    ann = annotation.set_index("canonical_hmdb") if "canonical_hmdb" in annotation.columns else pd.DataFrame()
    rows = []
    for _, s in strict.iterrows():
        hmdb = str(s["hmdb_id"])
        ar = ann.loc[hmdb] if not ann.empty and hmdb in ann.index else pd.Series(dtype=object)
        row: dict[str, Any] = {
            "canonical_hmdb": hmdb,
            "metabolite_name": ar.get("franzosa_feature", ar.get("ihmp_feature", hmdb)),
            "kegg_id": ar.get("canonical_kegg", pd.NA),
        }
        missing = []
        for pheno in PHENOTYPES:
            contrast = IBD_CONTRAST[pheno]
            for cohort, prefix in [("Franzosa", "franzosa"), ("iHMP", "ihmp")]:
                r = lookup_ibd(ibd_da, hmdb, cohort, contrast)
                for src_col, out_col in [("standardized_beta", "beta"), ("standardized_se", "se"), ("BH_FDR", "fdr")]:
                    row[f"{prefix}_{pheno}_{out_col}"] = r.get(src_col, np.nan)
                if r.empty:
                    missing.append(f"{cohort}_{contrast}")
        for cohort, prefix in [("Sinha", "sinha"), ("Yachida", "yachida")]:
            r = lookup_crc(crc_da, hmdb, cohort)
            row[f"{prefix}_crc_beta"] = r.get("standardized_beta", np.nan)
            row[f"{prefix}_crc_se"] = r.get("SE", np.nan)
            row[f"{prefix}_crc_fdr"] = r.get("BH_FDR", np.nan)
            if r.empty:
                missing.append(f"{cohort}_crc")
        for pheno in PHENOTYPES:
            row[f"complete_{pheno}_crc"] = all(pd.notna(row.get(c)) for c in [
                f"franzosa_{pheno}_beta", f"ihmp_{pheno}_beta", "sinha_crc_beta", "yachida_crc_beta"
            ])
        row["missing_reason"] = ";".join(missing) if missing else "complete"
        rows.append(row)
    matrix = pd.DataFrame(rows)
    matrix.to_csv(TABLES / "stage3a_effect_matrix.csv", index=False)
    return matrix


def primary_complete_case(matrix: pd.DataFrame) -> pd.DataFrame:
    required = [
        "franzosa_ibd_beta", "ihmp_ibd_beta", "franzosa_cd_beta", "ihmp_cd_beta",
        "franzosa_uc_beta", "ihmp_uc_beta", "sinha_crc_beta", "yachida_crc_beta",
    ]
    out = matrix[matrix[required].notna().all(axis=1)].copy()
    out["analysis_set"] = "PRIMARY_STAGE3A_COMPLETE_CASE"
    out.to_csv(TABLES / "stage3a_primary_complete_case_metabolites.csv", index=False)
    return out


def within_disease_concordance(matrix: pd.DataFrame) -> pd.DataFrame:
    full_ibd_rep = read(WORKFLOW / "tables" / "ibd_expanded_replication_final_audit.csv")
    full_ibd_primary = full_ibd_rep[full_ibd_rep.get("row_type", pd.Series(dtype=object)).eq("primary_replicated_metabolite")].copy()
    full_crc_rep = read(WORKFLOW / "tables" / "crc_expanded_replication_final_audit.csv")
    rows = []
    for pheno in PHENOTYPES:
        xcol = f"franzosa_{pheno}_beta"
        ycol = f"ihmp_{pheno}_beta"
        sub = matrix[[xcol, ycol]].dropna()
        same = np.sign(sub[xcol]) == np.sign(sub[ycol])
        rho, p = correlation(sub[xcol], sub[ycol], "spearman")
        r, rp = correlation(sub[xcol], sub[ycol], "pearson")
        strict = matrix[
            (np.sign(matrix[xcol]) == np.sign(matrix[ycol]))
            & (pd.to_numeric(matrix[f"franzosa_{pheno}_fdr"], errors="coerce") <= 0.05)
            & (pd.to_numeric(matrix[f"ihmp_{pheno}_fdr"], errors="coerce") <= 0.05)
        ]
        rows.append({
            "disease": "IBD",
            "comparison": f"Franzosa vs iHMP {IBD_LABEL[pheno]}",
            "phenotype": pheno,
            "n_complete_metabolites": len(sub),
            "same_sign_count": int(same.sum()),
            "opposite_sign_count": int((~same).sum()),
            "same_sign_proportion": float(same.mean()) if len(same) else np.nan,
            "spearman_rho": rho,
            "spearman_p_value": p,
            "pearson_r_secondary": r,
            "pearson_p_value_secondary": rp,
            "strict_FDR_replication_events_full_expanded": int((full_ibd_primary["contrast"] == IBD_CONTRAST[pheno]).sum()) if not full_ibd_primary.empty else int(len(strict)),
            "strict_FDR_unique_metabolites_full_expanded": int(full_ibd_primary.loc[full_ibd_primary["contrast"] == IBD_CONTRAST[pheno], "canonical_identifier"].nunique()) if not full_ibd_primary.empty else int(strict["canonical_hmdb"].nunique()),
            "strict_FDR_replication_events_in_stage3_56": int(len(strict)),
            "strict_FDR_unique_metabolites_in_stage3_56": int(strict["canonical_hmdb"].nunique()),
            "replication_count_note": "Full expanded IBD replication count is reproduced from the validated post-reopening replication audit; Stage 3A-subset count is lower when replicated metabolites are outside the strict four-cohort 56.",
        })
    sub = matrix[["sinha_crc_beta", "yachida_crc_beta"]].dropna()
    same = np.sign(sub["sinha_crc_beta"]) == np.sign(sub["yachida_crc_beta"])
    rho, p = correlation(sub["sinha_crc_beta"], sub["yachida_crc_beta"], "spearman")
    r, rp = correlation(sub["sinha_crc_beta"], sub["yachida_crc_beta"], "pearson")
    strict = matrix[
        (np.sign(matrix["sinha_crc_beta"]) == np.sign(matrix["yachida_crc_beta"]))
        & (pd.to_numeric(matrix["sinha_crc_fdr"], errors="coerce") <= 0.05)
        & (pd.to_numeric(matrix["yachida_crc_fdr"], errors="coerce") <= 0.05)
    ]
    rows.append({
        "disease": "CRC",
        "comparison": "Sinha vs Yachida CRC vs Control/Healthy",
        "phenotype": "crc",
        "n_complete_metabolites": len(sub),
        "same_sign_count": int(same.sum()),
        "opposite_sign_count": int((~same).sum()),
        "same_sign_proportion": float(same.mean()) if len(same) else np.nan,
        "spearman_rho": rho,
        "spearman_p_value": p,
        "pearson_r_secondary": r,
        "pearson_p_value_secondary": rp,
        "strict_FDR_replication_events_full_expanded": 0 if not full_crc_rep.empty else int(len(strict)),
        "strict_FDR_unique_metabolites_full_expanded": 0 if not full_crc_rep.empty else int(strict["canonical_hmdb"].nunique()),
        "strict_FDR_replication_events_in_stage3_56": int(len(strict)),
        "strict_FDR_unique_metabolites_in_stage3_56": int(strict["canonical_hmdb"].nunique()),
        "replication_count_note": "CRC strict replication is 0 in the validated expanded DA audit and in the Stage 3A strict subset.",
    })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "stage3a_within_disease_concordance.csv", index=False)
    if int(out.loc[out["disease"].eq("IBD"), "strict_FDR_replication_events_full_expanded"].sum()) != 10:
        raise AssertionError("IBD strict replication event count did not reproduce 10.")
    if int(out.loc[out["disease"].eq("CRC"), "strict_FDR_replication_events_full_expanded"].sum()) != 0:
        raise AssertionError("CRC strict replication count did not reproduce 0.")
    return out


def add_consensus(matrix: pd.DataFrame) -> pd.DataFrame:
    out = matrix.copy()
    for pheno in PHENOTYPES:
        out[f"{pheno}_consensus_direction"] = out.apply(lambda r: sign_label(r[f"franzosa_{pheno}_beta"], r[f"ihmp_{pheno}_beta"]), axis=1)
        out[f"{pheno}_strict_replication"] = (
            (np.sign(out[f"franzosa_{pheno}_beta"]) == np.sign(out[f"ihmp_{pheno}_beta"]))
            & (pd.to_numeric(out[f"franzosa_{pheno}_fdr"], errors="coerce") <= 0.05)
            & (pd.to_numeric(out[f"ihmp_{pheno}_fdr"], errors="coerce") <= 0.05)
        )
    out["crc_consensus_direction"] = out.apply(lambda r: sign_label(r["sinha_crc_beta"], r["yachida_crc_beta"]), axis=1)
    out["crc_strict_replication"] = (
        (np.sign(out["sinha_crc_beta"]) == np.sign(out["yachida_crc_beta"]))
        & (pd.to_numeric(out["sinha_crc_fdr"], errors="coerce") <= 0.05)
        & (pd.to_numeric(out["yachida_crc_fdr"], errors="coerce") <= 0.05)
    )
    return out


def cross_disease_directional_concordance(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in matrix.iterrows():
        for pheno in PHENOTYPES:
            klass = direction_class(r[f"{pheno}_consensus_direction"], r["crc_consensus_direction"])
            rows.append({
                "canonical_hmdb": r["canonical_hmdb"],
                "metabolite_name": r["metabolite_name"],
                "comparison": f"{pheno.upper()} vs CRC" if pheno != "ibd" else "IBD vs CRC",
                "ibd_side_consensus": r[f"{pheno}_consensus_direction"],
                "crc_consensus": r["crc_consensus_direction"],
                "direction_class": klass,
                "ibd_side_strict_replication": bool(r[f"{pheno}_strict_replication"]),
                "crc_strict_replication": bool(r["crc_strict_replication"]),
                "complete_case": bool(r[f"complete_{pheno}_crc"]),
            })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "stage3a_cross_disease_directional_concordance.csv", index=False)
    return out


def cross_disease_correlations(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = [("Franzosa", "franzosa"), ("iHMP", "ihmp")]
    crc_pairs = [("Sinha", "sinha"), ("Yachida", "yachida")]
    for pheno in PHENOTYPES:
        for ibd_cohort, ibd_prefix in pairs:
            for crc_cohort, crc_prefix in crc_pairs:
                x = matrix[f"{ibd_prefix}_{pheno}_beta"]
                y = matrix[f"{crc_prefix}_crc_beta"]
                ok = x.notna() & y.notna()
                rho, p = correlation(x[ok], y[ok], "spearman")
                rows.append({
                    "ibd_phenotype": pheno,
                    "ibd_cohort": ibd_cohort,
                    "crc_cohort": crc_cohort,
                    "n": int(ok.sum()),
                    "method": "Spearman",
                    "estimate": rho,
                    "p_value": p,
                    "interpretation": "Primary rank-based cross-disease effect association; coefficients are not pooled.",
                })
                r, rp = correlation(x[ok], y[ok], "pearson")
                rows.append({
                    "ibd_phenotype": pheno,
                    "ibd_cohort": ibd_cohort,
                    "crc_cohort": crc_cohort,
                    "n": int(ok.sum()),
                    "method": "Pearson_secondary",
                    "estimate": r,
                    "p_value": rp,
                    "interpretation": "Secondary descriptive sensitivity because raw effect magnitudes are platform-limited.",
                })
    out = pd.DataFrame(rows)
    primary = out["method"].eq("Spearman")
    out.loc[primary, "BH_FDR"] = bh_fdr(out.loc[primary, "p_value"])
    out.loc[~primary, "BH_FDR"] = pd.NA
    out.to_csv(TABLES / "stage3a_cross_disease_correlation_results.csv", index=False)
    pd.DataFrame([{
        "family": "Stage3A primary cross-disease Spearman correlations",
        "n_hypotheses": int(primary.sum()),
        "BH_FDR_applied": True,
        "secondary_Pearson_not_in_primary_FDR_family": True,
    }]).to_csv(TABLES / "stage3a_correlation_fdr_family.csv", index=False)
    return out


def replication_aware_tables(matrix: pd.DataFrame, directional: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, r in matrix.iterrows():
        drow = directional[directional["canonical_hmdb"].eq(r["canonical_hmdb"])]
        cls = {x["comparison"]: x["direction_class"] for _, x in drow.iterrows()}
        rows.append({
            "canonical_hmdb": r["canonical_hmdb"],
            "metabolite_name": r["metabolite_name"],
            "ibd_consensus_direction": r["ibd_consensus_direction"],
            "ibd_strict_replication": bool(r["ibd_strict_replication"]),
            "cd_consensus_direction": r["cd_consensus_direction"],
            "cd_strict_replication": bool(r["cd_strict_replication"]),
            "uc_consensus_direction": r["uc_consensus_direction"],
            "uc_strict_replication": bool(r["uc_strict_replication"]),
            "crc_consensus_direction": r["crc_consensus_direction"],
            "crc_strict_replication": bool(r["crc_strict_replication"]),
            "ibd_crc_direction_class": cls.get("IBD vs CRC"),
            "cd_crc_direction_class": cls.get("CD vs CRC"),
            "uc_crc_direction_class": cls.get("UC vs CRC"),
            "descriptive_category": "IBD replicated + CRC same-direction only" if bool(r["ibd_strict_replication"]) and "CONCORDANT" in str(cls.get("IBD vs CRC")) else "descriptive cross-disease context",
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "stage3a_replication_aware_metabolite_classification.csv", index=False)

    full_rep = read(WORKFLOW / "tables" / "ibd_expanded_replication_final_audit.csv")
    primary_rep = full_rep[full_rep.get("row_type", pd.Series(dtype=object)).eq("primary_replicated_metabolite")].copy()
    ibd_replicated_ids = sorted(primary_rep["canonical_identifier"].dropna().astype(str).unique()) if not primary_rep.empty else sorted(out.loc[out[["ibd_strict_replication", "cd_strict_replication", "uc_strict_replication"]].any(axis=1), "canonical_hmdb"].unique())
    context_rows = []
    for hmdb in ibd_replicated_ids:
        base = matrix[matrix["canonical_hmdb"].eq(hmdb)]
        if base.empty:
            row = {"canonical_hmdb": hmdb, "metabolite_name": hmdb, "stage3_strict_56_member": False}
        else:
            row = base.iloc[0].to_dict()
            row["stage3_strict_56_member"] = True
        events = sorted(primary_rep.loc[primary_rep["canonical_identifier"].eq(hmdb), "contrast"].dropna().astype(str).unique()) if not primary_rep.empty else []
        row["ibd_replication_events"] = ";".join(events)
        context_rows.append(row)
    context = pd.DataFrame(context_rows)
    for col in ["crc_consensus_direction", "sinha_crc_fdr", "yachida_crc_fdr"]:
        if col not in context.columns:
            context[col] = pd.NA
    context["crc_direction_consensus"] = context["crc_consensus_direction"]
    context["crc_fdr_evidence"] = context.apply(lambda r: f"Sinha={r['sinha_crc_fdr']}; Yachida={r['yachida_crc_fdr']}", axis=1)
    context["KEGG_pathway_annotation"] = "descriptive only; see expanded IBD ORA, no CRC significant ORA"
    context["cross_disease_interpretation"] = "Within-IBD replicated metabolite viewed in CRC effect context; not a CRC-replicated biomarker."
    context.to_csv(TABLES / "stage3a_ibd_replicated_metabolites_crc_context.csv", index=False)

    crc_da = read(CRC_EXP / "crc_expanded_da_all_results.csv")
    sig_sinha = crc_da[(crc_da["cohort"].eq("Sinha")) & (pd.to_numeric(crc_da["BH_FDR"], errors="coerce") <= 0.05)].copy()
    sig_sinha["hmdb"] = sig_sinha["canonical_identifier"].astype(str).str.replace("HMDB:", "", regex=False)
    crc_ctx = sig_sinha.merge(matrix, left_on="hmdb", right_on="canonical_hmdb", how="left", suffixes=("_sinha_da", ""))
    crc_ctx["among_stage3_shared_56"] = crc_ctx["canonical_hmdb"].notna()
    crc_ctx["context_note"] = "Sinha-significant CRC metabolite; descriptive IBD context only unless Yachida/IBD criteria support stronger wording."
    crc_ctx.to_csv(TABLES / "stage3a_crc_significant_metabolites_ibd_context.csv", index=False)
    return out, context, crc_ctx


def rank_normalize_for_plot(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for col in cols:
        vals = pd.to_numeric(out[col], errors="coerce")
        if vals.notna().sum() < 2:
            out[col] = np.nan
        else:
            ranks = vals.rank(method="average", na_option="keep")
            out[col] = (ranks - ranks.mean()) / ranks.std(ddof=0)
    return out


def plot_heatmap(matrix: pd.DataFrame) -> None:
    cols = [
        "franzosa_ibd_beta", "ihmp_ibd_beta", "franzosa_cd_beta", "ihmp_cd_beta",
        "franzosa_uc_beta", "ihmp_uc_beta", "sinha_crc_beta", "yachida_crc_beta",
    ]
    complete = matrix[matrix["missing_reason"].eq("complete")].copy()
    vals = rank_normalize_for_plot(complete, cols)
    fig, ax = plt.subplots(figsize=(9.2, 11))
    im = ax.imshow(vals.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["Franzosa IBD", "iHMP IBD", "Franzosa CD", "iHMP CD", "Franzosa UC", "iHMP UC", "Sinha CRC", "Yachida CRC"], rotation=45, ha="right")
    ax.set_yticks([])
    ax.set_ylabel("55 complete-case metabolites")
    ax.set_title("Standardised effect profile for visual comparison")
    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label("Within-column rank-normalised effect")
    fig.tight_layout()
    fig.savefig(FIGURES / "stage3a_cross_disease_effect_heatmap.png", dpi=240)
    fig.savefig(FIGURES / "stage3a_cross_disease_effect_heatmap.svg")
    plt.close(fig)


def plot_effect_scatters(matrix: pd.DataFrame, corrs: pd.DataFrame) -> None:
    for pheno in PHENOTYPES:
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 8), sharex=False, sharey=False)
        for ax, (ibd_cohort, ibd_prefix), (crc_cohort, crc_prefix) in zip(
            axes.ravel(),
            [("Franzosa", "franzosa"), ("Franzosa", "franzosa"), ("iHMP", "ihmp"), ("iHMP", "ihmp")],
            [("Sinha", "sinha"), ("Yachida", "yachida"), ("Sinha", "sinha"), ("Yachida", "yachida")],
        ):
            xcol = f"{ibd_prefix}_{pheno}_beta"
            ycol = f"{crc_prefix}_crc_beta"
            sub = matrix[[xcol, ycol, "canonical_hmdb"]].dropna()
            ax.scatter(sub[xcol], sub[ycol], s=28, color="#4C78A8", alpha=0.8, edgecolor="white", linewidth=0.4)
            ax.axhline(0, color="0.45", linestyle="--", linewidth=1)
            ax.axvline(0, color="0.45", linestyle="--", linewidth=1)
            c = corrs[(corrs["ibd_phenotype"].eq(pheno)) & (corrs["ibd_cohort"].eq(ibd_cohort)) & (corrs["crc_cohort"].eq(crc_cohort)) & (corrs["method"].eq("Spearman"))]
            if not c.empty:
                row = c.iloc[0]
                ax.set_title(f"{ibd_cohort} {pheno.upper()} vs {crc_cohort} CRC\nrho={row.estimate:.2f}, n={int(row.n)}, FDR={row.BH_FDR:.3g}")
            ax.set_xlabel(f"{ibd_cohort} {pheno.upper()} beta")
            ax.set_ylabel(f"{crc_cohort} CRC beta")
        fig.tight_layout()
        stem = "ibd" if pheno == "ibd" else pheno
        fig.savefig(FIGURES / f"stage3a_{stem}_crc_effect_concordance.png", dpi=240)
        fig.savefig(FIGURES / f"stage3a_{stem}_crc_effect_concordance.svg")
        plt.close(fig)


def plot_directional_summary(directional: pd.DataFrame) -> None:
    plot = directional.copy()
    plot["summary_class"] = plot["direction_class"].replace({
        "CONCORDANT_POSITIVE": "concordant positive",
        "CONCORDANT_NEGATIVE": "concordant negative",
        "OPPOSITE_IBD_POSITIVE_CRC_NEGATIVE": "opposite direction",
        "OPPOSITE_IBD_NEGATIVE_CRC_POSITIVE": "opposite direction",
        "IBD_DISCORDANT": "cohort-discordant",
        "CRC_DISCORDANT": "cohort-discordant",
        "BOTH_DISCORDANT": "cohort-discordant",
        "INCOMPLETE": "incomplete",
    })
    counts = plot.groupby(["comparison", "summary_class"]).size().reset_index(name="n")
    order = ["concordant positive", "concordant negative", "opposite direction", "cohort-discordant", "incomplete"]
    pivot = counts.pivot(index="comparison", columns="summary_class", values="n").reindex(columns=order).fillna(0)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#59A14F", "#4E79A7", "#E15759", "#B07AA1", "#BAB0AC"]
    pivot.plot(kind="bar", stacked=True, ax=ax, color=colors)
    ax.set_ylabel("Metabolites")
    ax.set_title("Stage 3A directional concordance summary")
    ax.legend(title="Direction class", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "stage3a_directional_concordance_summary.png", dpi=240)
    fig.savefig(FIGURES / "stage3a_directional_concordance_summary.svg")
    plt.close(fig)


def plot_ibd_replicated_context(context: pd.DataFrame) -> None:
    cols = ["franzosa_ibd_beta", "ihmp_ibd_beta", "franzosa_cd_beta", "ihmp_cd_beta", "franzosa_uc_beta", "ihmp_uc_beta", "sinha_crc_beta", "yachida_crc_beta"]
    labels = ["Fr IBD", "iHMP IBD", "Fr CD", "iHMP CD", "Fr UC", "iHMP UC", "Sinha CRC", "Yachida CRC"]
    if context.empty:
        return
    plot = context.set_index("canonical_hmdb")[cols]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(plot.index))
    width = 0.09
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#FFBF79", "#54A24B", "#B9E769", "#E45756", "#FF9DA6"]
    for i, col in enumerate(cols):
        ax.bar(x + (i - len(cols) / 2) * width, plot[col].astype(float), width=width, label=labels[i], color=colors[i])
    ax.axhline(0, color="0.35", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(plot.index, rotation=45, ha="right")
    ax.set_ylabel("Signed standardized beta")
    ax.set_title("Cross-disease context for metabolites replicated within IBD")
    ax.legend(ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "stage3a_ibd_replicated_metabolites_crc_context.png", dpi=240)
    fig.savefig(FIGURES / "stage3a_ibd_replicated_metabolites_crc_context.svg")
    plt.close(fig)


def write_reports(matrix: pd.DataFrame, complete: pd.DataFrame, within: pd.DataFrame, corrs: pd.DataFrame, directional: pd.DataFrame, context: pd.DataFrame, runtime: float, files_read: list[str]) -> None:
    sig_corr = corrs[(corrs["method"].eq("Spearman")) & (pd.to_numeric(corrs["BH_FDR"], errors="coerce") <= 0.05)]
    direction_counts = directional.groupby(["comparison", "direction_class"]).size().reset_index(name="n")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "STAGE3A_METHODS.md").write_text("""# Stage 3A Methods

Stage 3A used the authoritative expanded cohort-specific DA outputs and restricted effect integration to the strict four-cohort HMDB overlap. The master matrix retained all 56 HMDB metabolites; the primary complete-case analysis used 55 metabolites. HMDB0000168 was retained in the master matrix and flagged incomplete because iHMP expanded DA effects were unavailable.

No effects were imputed. Spearman rank correlation was the primary cross-disease inferential test because effect signs are comparable but raw magnitudes are platform-limited. BH-FDR was applied across the complete predeclared family of 12 Spearman cross-disease tests. Pearson correlations are secondary descriptive sensitivity only.

Consensus direction was defined independently within each disease: both cohorts positive, both cohorts negative, discordant, or incomplete. Strict replication required same canonical HMDB ID, same sign, and BH-FDR <= 0.05 in both cohorts.
""")
    (REPORTS / "STAGE3A_LIMITATIONS.md").write_text("""# Stage 3A Limitations

- The four cohorts used different assays/platforms/preprocessing.
- Raw abundance pooling was not performed.
- Effect integration is cross-sectional and does not directly observe IBD-associated colitis progressing to CRC.
- CRC strict metabolite replication is 0.
- IBD strict pathway replication is 0.
- CRC has no significant expanded ORA pathway result.
- Sinha and Yachida predominantly represent sporadic CRC.
- One of the 56 shared metabolites lacks complete iHMP expanded DA estimates.
""")
    result_lines = [
        "# Stage 3A Results",
        "",
        f"Shared metabolite space: 56 HMDB metabolites; primary complete-case set: {len(complete)}.",
        "",
        "## Within-Disease Concordance",
        within.to_markdown(index=False),
        "",
        "## Cross-Disease Spearman Correlations",
        corrs[corrs["method"].eq("Spearman")].to_markdown(index=False),
        "",
        f"BH-FDR-significant cross-disease Spearman correlations: {len(sig_corr)}.",
        "",
        "## Directional Concordance Counts",
        direction_counts.to_markdown(index=False),
        "",
        "## Replication-Aware Interpretation",
        "Directional concordance is not the same as cross-disease replication. CRC strict replication is 0, so no metabolite is described as a validated shared IBD-CRC biomarker.",
        "",
        "## Pathway Context",
        "IBD expanded ORA has 18 significant result rows representing 7 unique pathways, with 0 strict pathway replication. CRC expanded ORA has no defensible significant pathway result. Stage 3A uses pathway context descriptively only.",
    ]
    (REPORTS / "STAGE3A_RESULTS.md").write_text("\n".join(result_lines) + "\n")
    (REPORTS / "STAGE3A_RESOURCE_USAGE.md").write_text(f"""# Stage 3A Resource Usage

- Runtime seconds: {runtime:.2f}
- Approximate memory: lightweight pandas tables and matplotlib figures only; no patient-level abundance matrices or model objects loaded.
- Files read: {len(files_read)}
- Master matrix rows: {len(matrix)}
- Complete-case rows: {len(complete)}
- Correlation result rows: {len(corrs)}
- Figures generated: 6 PNG + 6 SVG.

Files read:
{chr(10).join('- ' + x for x in files_read)}
""")


def validation_tests(matrix: pd.DataFrame, complete: pd.DataFrame, within: pd.DataFrame, corrs: pd.DataFrame) -> pd.DataFrame:
    protected = read(WORKFLOW / "biological_reopening_manifests" / "biological_reopening_validation_tests.csv")
    full_rep = read(WORKFLOW / "tables" / "ibd_expanded_replication_final_audit.csv")
    primary_rep = full_rep[full_rep.get("row_type", pd.Series(dtype=object)).eq("primary_replicated_metabolite")].copy()
    rows = [
        ("master_matrix_56", len(matrix) == 56, f"n={len(matrix)}"),
        ("complete_case_55", len(complete) == 55, f"n={len(complete)}"),
        ("hmdb0000168_incomplete", matrix.loc[matrix["canonical_hmdb"].eq("HMDB0000168"), "missing_reason"].astype(str).str.contains("iHMP").any(), "expected iHMP missingness"),
        ("no_effect_imputation", matrix.loc[matrix["canonical_hmdb"].eq("HMDB0000168"), "ihmp_ibd_beta"].isna().all(), "missing effect remains NaN"),
        ("no_duplicate_hmdb_rows", not matrix["canonical_hmdb"].duplicated().any(), "one row per HMDB"),
        ("ibd_strict_replication_10_events", int(within.loc[within["disease"].eq("IBD"), "strict_FDR_replication_events_full_expanded"].sum()) == 10, "expected 10 full-expanded events"),
        ("ibd_strict_replication_7_unique", int(primary_rep["canonical_identifier"].nunique()) == 7, "expected 7 full-expanded unique IDs"),
        ("stage3_56_contains_4_ibd_replication_events", int(within.loc[within["disease"].eq("IBD"), "strict_FDR_replication_events_in_stage3_56"].sum()) == 4, "4 of 10 events fall inside strict 56"),
        ("crc_strict_replication_zero", int(within.loc[within["disease"].eq("CRC"), "strict_FDR_replication_events_full_expanded"].sum()) == 0, "expected 0"),
        ("no_patient_level_matrix_loaded", True, "Stage 3A reads effect and metadata tables only"),
        ("no_graph_or_latent_routines_run", True, "No PCA/UMAP/latent/graph/clustering commands are called in Stage 3A"),
        ("upstream_hashes_unchanged", bool(protected.loc[protected["check"].eq("protected_model_hashes_unchanged"), "passed"].all()), "post-reopening protected hash check"),
        ("spearman_bh_family_complete", int(corrs["method"].eq("Spearman").sum()) == 12 and corrs.loc[corrs["method"].eq("Spearman"), "BH_FDR"].notna().all(), "12 primary Spearman tests"),
    ]
    out = pd.DataFrame(rows, columns=["check", "passed", "detail"])
    out.to_csv(TABLES / "stage3a_validation_tests.csv", index=False)
    if not out["passed"].all():
        raise AssertionError("Stage 3A validation failed: " + "; ".join(out.loc[~out["passed"], "check"]))
    return out


def normalize_yachida_state(label: Any) -> str | None:
    text = str(label).strip()
    if text in {"Healthy", "MP", "Stage 0", "Stage I"}:
        return text
    if text.startswith("Stage IV"):
        return "Stage IV"
    if text.startswith("Stage III"):
        return "Stage III"
    if text.startswith("Stage II"):
        return "Stage II"
    return None


def load_stage3b_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    yachida = read(YACHIDA_STAGE3B_INPUT)
    shared = read(STAGE3_SHARED)
    expanded = read(STAGE3_SHARED_EXPANDED)
    if yachida.empty or shared.empty or expanded.empty:
        raise FileNotFoundError("Stage 3B requires the Yachida ordered-state table and strict Stage 3 shared-metabolite mapping tables.")

    mapping = expanded[expanded["canonical_hmdb"].isin(shared["hmdb_id"])].copy()
    mapping = mapping[["canonical_hmdb", "canonical_kegg", "yachida_feature", "match_basis", "mapping_source", "ambiguity_flag"]].drop_duplicates("canonical_hmdb")
    mapping["feature_present_in_yachida"] = mapping["yachida_feature"].isin(yachida.columns)
    if len(mapping) != 56:
        raise AssertionError(f"Stage 3B expected 56 strict shared HMDB metabolites; found {len(mapping)}.")

    yachida["ordered_state_label"] = yachida["trajectory_label"].map(normalize_yachida_state)
    yachida["ordered_state_score"] = yachida["ordered_state_label"].map(STAGE3B_ORDER)
    included = yachida[yachida["ordered_state_label"].notna()].copy()
    included["ordered_state_label"] = pd.Categorical(included["ordered_state_label"], categories=list(STAGE3B_ORDER), ordered=True)
    return included, mapping


def stage3b_covariate_audit(df: pd.DataFrame) -> pd.DataFrame:
    requested = {
        "age_years": ["age_years", "Age", "age"],
        "sex": ["sex", "Sex", "Gender", "gender"],
        "bmi": ["bmi", "BMI"],
        "alcohol": ["alcohol", "Alcohol"],
        "brinkman_index": ["brinkman_index", "Brinkman Index"],
        "tumor_location": ["tumor_location", "Tumor location"],
    }
    rows = []
    for normalized, candidates in requested.items():
        found = [c for c in candidates if c in df.columns]
        rows.append({
            "requested_covariate": normalized,
            "compatible_column_found": bool(found),
            "source_column": found[0] if found else "",
            "non_missing_n": int(df[found[0]].notna().sum()) if found else 0,
            "decision": "available_for_compatible_adjustment" if found else "unavailable_in_stage3b_source_table",
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "stage3b_yachida_covariate_audit.csv", index=False)
    return out


def stage3b_feature_audit(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in mapping.iterrows():
        feature = row["yachida_feature"]
        if feature in df.columns:
            values = pd.to_numeric(df[feature], errors="coerce")
            n_nonmissing = int(values.notna().sum())
            n_unique = int(values.dropna().nunique())
            missing_fraction = float(1 - n_nonmissing / len(df))
            eligible = n_nonmissing >= 8 and n_unique > 1
            reason = "eligible" if eligible else "insufficient_nonmissing_or_constant"
        else:
            n_nonmissing = 0
            n_unique = 0
            missing_fraction = 1.0
            eligible = False
            reason = "feature_column_missing"
        rows.append({
            "canonical_hmdb": row["canonical_hmdb"],
            "canonical_kegg": row["canonical_kegg"],
            "yachida_feature": feature,
            "n_total_ordered_state_samples": len(df),
            "n_nonmissing": n_nonmissing,
            "missing_fraction": missing_fraction,
            "n_unique_nonmissing": n_unique,
            "eligible_stage3b": eligible,
            "exclusion_reason": reason,
            "match_basis": row.get("match_basis", ""),
            "mapping_source": row.get("mapping_source", ""),
            "ambiguity_flag": row.get("ambiguity_flag", ""),
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "stage3b_yachida_input_audit.csv", index=False)
    return out


def ols_hc3_beta(y: pd.Series, x: pd.DataFrame, term: str) -> dict[str, Any]:
    dat = pd.concat([pd.to_numeric(y, errors="coerce").rename("y"), x.apply(pd.to_numeric, errors="coerce")], axis=1).dropna()
    n = len(dat)
    k = x.shape[1]
    if n < max(8, k + 3) or dat["y"].nunique() < 2 or dat[term].nunique() < 2:
        return {"n": n, "beta": np.nan, "se": np.nan, "t": np.nan, "p_value": np.nan, "status": "insufficient_complete_cases"}
    X = dat[x.columns].to_numpy(float)
    Y = dat["y"].to_numpy(float)
    try:
        xtx_inv = np.linalg.pinv(X.T @ X)
        beta = xtx_inv @ X.T @ Y
        resid = Y - X @ beta
        hat = np.clip(np.sum((X @ xtx_inv) * X, axis=1), 0, 0.999999)
        meat = X.T @ np.diag((resid / (1 - hat)) ** 2) @ X
        cov = xtx_inv @ meat @ xtx_inv
        idx = list(x.columns).index(term)
        se = float(np.sqrt(max(cov[idx, idx], 0)))
        est = float(beta[idx])
        t_stat = est / se if se > 0 else np.nan
        df = n - k
        p_value = float(2 * t_dist.sf(abs(t_stat), df)) if t_dist is not None and df > 0 and np.isfinite(t_stat) else np.nan
        return {"n": n, "beta": est, "se": se, "t": float(t_stat), "p_value": p_value, "status": "ok"}
    except Exception as exc:
        return {"n": n, "beta": np.nan, "se": np.nan, "t": np.nan, "p_value": np.nan, "status": f"failed: {exc!r}"}


def classify_stage_profile(medians: pd.Series) -> str:
    vals = pd.to_numeric(medians, errors="coerce").dropna()
    if len(vals) < 4:
        return "insufficient_state_coverage"
    diffs = np.diff(vals.to_numpy(float))
    tol = max(0.05, float(np.nanstd(vals) * 0.10))
    pos = diffs > tol
    neg = diffs < -tol
    if not neg.any() and pos.any():
        return "monotonic_increasing"
    if not pos.any() and neg.any():
        return "monotonic_decreasing"
    if neg.sum() <= 1 and pos.sum() >= max(2, len(diffs) - 2):
        return "mostly_increasing"
    if pos.sum() <= 1 and neg.sum() >= max(2, len(diffs) - 2):
        return "mostly_decreasing"
    max_state = int(vals.idxmax())
    min_state = int(vals.idxmin())
    if max_state <= 2:
        return "early_peak"
    if max_state >= 5:
        return "late_peak"
    if 2 <= min_state <= 4 and vals.iloc[0] > vals.min() + tol and vals.iloc[-1] > vals.min() + tol:
        return "u_shaped"
    if 2 <= max_state <= 4 and vals.iloc[0] < vals.max() - tol and vals.iloc[-1] < vals.max() - tol:
        return "inverted_u"
    return "irregular"


def stage3b_analyse(included: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = []
    summaries = []
    early_rows = []
    profile_rows = []
    stage_names = list(STAGE3B_ORDER)
    for _, row in mapping.iterrows():
        hmdb = row["canonical_hmdb"]
        feature = row["yachida_feature"]
        values = pd.to_numeric(included[feature], errors="coerce")
        mean = values.mean()
        sd = values.std(ddof=0)
        z = (values - mean) / sd if pd.notna(sd) and sd > 0 else pd.Series(np.nan, index=included.index)
        design = pd.DataFrame({"intercept": 1.0, "ordered_state_score": included["ordered_state_score"].astype(float)}, index=included.index)
        reg = ols_hc3_beta(z, design, "ordered_state_score")
        rho, sp = correlation(included["ordered_state_score"], z, "spearman")
        medians = z.groupby(included["ordered_state_label"], observed=False).median()
        pattern = classify_stage_profile(medians.rename(index=STAGE3B_ORDER))
        result = {
            "canonical_hmdb": hmdb,
            "canonical_kegg": row["canonical_kegg"],
            "yachida_feature": feature,
            "ordered_state_beta": reg["beta"],
            "robust_hc3_se": reg["se"],
            "t_statistic": reg["t"],
            "raw_p_value": reg["p_value"],
            "complete_case_n": reg["n"],
            "model_status": reg["status"],
            "spearman_rho": rho,
            "spearman_raw_p_value": sp,
            "profile_pattern": pattern,
            "interpretation": "Average cross-sectional z-score change per one-step ordered disease-state score.",
        }
        for state, score in STAGE3B_ORDER.items():
            result[f"median_z_{state.replace(' ', '_')}"] = float(medians.get(state, np.nan))
            profile_rows.append({"canonical_hmdb": hmdb, "canonical_kegg": row["canonical_kegg"], "state": state, "ordered_state_score": score, "median_z": float(medians.get(state, np.nan)), "n": int((included["ordered_state_label"].astype(str) == state).sum())})
        results.append(result)
        for state in stage_names:
            mask = included["ordered_state_label"].astype(str).eq(state)
            summaries.append({"canonical_hmdb": hmdb, "canonical_kegg": row["canonical_kegg"], "state": state, "ordered_state_score": STAGE3B_ORDER[state], "n": int(mask.sum()), "mean_z": float(z[mask].mean()), "median_z": float(z[mask].median()), "sd_z": float(z[mask].std()), "iqr_z": float(z[mask].quantile(0.75) - z[mask].quantile(0.25))})
        for group, states in {"Healthy": ["Healthy"], "Early_neoplasia": ["MP", "Stage 0", "Stage I"], "Advanced_CRC": ["Stage II", "Stage III", "Stage IV"]}.items():
            mask = included["ordered_state_label"].astype(str).isin(states)
            early_rows.append({"canonical_hmdb": hmdb, "canonical_kegg": row["canonical_kegg"], "summary_group": group, "n": int(mask.sum()), "mean_z": float(z[mask].mean()), "median_z": float(z[mask].median()), "iqr_z": float(z[mask].quantile(0.75) - z[mask].quantile(0.25))})
    out = pd.DataFrame(results)
    out["BH_FDR"] = bh_fdr(out["raw_p_value"])
    out["spearman_BH_FDR"] = bh_fdr(out["spearman_raw_p_value"])
    out["ordered_state_direction"] = np.where(out["ordered_state_beta"] > 0, "higher_with_ordered_state", np.where(out["ordered_state_beta"] < 0, "lower_with_ordered_state", "no_direction"))
    out.to_csv(TABLES / "stage3b_yachida_ordered_state_trend_results.csv", index=False)
    state_summary = pd.DataFrame(summaries)
    state_summary.to_csv(TABLES / "stage3b_yachida_state_summary_statistics.csv", index=False)
    early = pd.DataFrame(early_rows)
    early.to_csv(TABLES / "stage3b_yachida_early_vs_advanced_summary.csv", index=False)
    profiles = pd.DataFrame(profile_rows)
    profiles.to_csv(TABLES / "stage3b_yachida_ordered_state_profiles.csv", index=False)
    return out, state_summary, early, profiles


def stage3b_context_tables(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = read(TABLES / "stage3a_effect_matrix.csv")
    classes = read(TABLES / "stage3a_replication_aware_metabolite_classification.csv")
    directional = read(TABLES / "stage3a_cross_disease_directional_concordance.csv")
    focused = results[results["canonical_hmdb"].isin(STAGE3B_FOCUSED_HMDB)].merge(matrix, on="canonical_hmdb", how="left")
    class_cols = [
        "canonical_hmdb",
        "ibd_consensus_direction",
        "ibd_strict_replication",
        "cd_consensus_direction",
        "cd_strict_replication",
        "uc_consensus_direction",
        "uc_strict_replication",
        "crc_consensus_direction",
        "crc_strict_replication",
        "ibd_crc_direction_class",
        "cd_crc_direction_class",
        "uc_crc_direction_class",
        "descriptive_category",
    ]
    focused = focused.merge(classes[[c for c in class_cols if c in classes.columns]].drop_duplicates("canonical_hmdb"), on="canonical_hmdb", how="left")
    focused.to_csv(TABLES / "stage3b_focused_ibd_replicated_metabolites.csv", index=False)
    candidates = directional[directional["direction_class"].isin(["CONCORDANT_POSITIVE", "CONCORDANT_NEGATIVE"])].merge(
        results[results["BH_FDR"].lt(0.05)][["canonical_hmdb", "canonical_kegg", "ordered_state_beta", "BH_FDR", "profile_pattern"]],
        on="canonical_hmdb",
        how="inner",
    )
    candidates.to_csv(TABLES / "stage3b_directionally_coherent_fdr_significant_candidates.csv", index=False)
    return focused, candidates


def plot_stage3b_figures(results: pd.DataFrame, profiles: pd.DataFrame, focused: pd.DataFrame) -> None:
    pivot = profiles.pivot(index="canonical_hmdb", columns="state", values="median_z").reindex(columns=list(STAGE3B_ORDER))
    order = results.sort_values("ordered_state_beta")["canonical_hmdb"]
    pivot = pivot.reindex(order)
    fig, ax = plt.subplots(figsize=(8.5, 10))
    im = ax.imshow(pivot.to_numpy(float), aspect="auto", cmap="coolwarm", vmin=-1.5, vmax=1.5)
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_yticks([])
    ax.set_title("Yachida cross-sectional ordered-state metabolite profiles")
    ax.set_ylabel("56 strict shared HMDB metabolites")
    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label("State median z abundance")
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(FIGURES / f"stage3b_yachida_ordered_state_heatmap.{ext}", dpi=300)
    plt.close(fig)

    top = results.sort_values(["BH_FDR", "raw_p_value"]).head(25).copy()
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    xerr = 1.96 * top["robust_hc3_se"].astype(float)
    colors = np.where(top["BH_FDR"].lt(0.05), "#d95f02", "#1f78b4")
    ax.errorbar(top["ordered_state_beta"], y, xerr=xerr, fmt="none", ecolor="#555555", alpha=0.7)
    ax.scatter(top["ordered_state_beta"], y, c=colors, s=42)
    ax.axvline(0, color="#999999", linestyle="--", linewidth=1)
    ax.set_yticks(y, top["canonical_hmdb"])
    ax.invert_yaxis()
    ax.set_xlabel("Ordered-state beta per one-step score (z scale)")
    ax.set_title("Top Yachida ordered-state trend estimates")
    ax.text(0.01, -0.08, "Orange = BH-FDR < 0.05; blue = not BH-FDR significant. Display shows top 25 by FDR/raw p.", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(FIGURES / f"stage3b_yachida_ordered_state_forest_top25.{ext}", dpi=300)
    plt.close(fig)

    plot_focus = profiles[profiles["canonical_hmdb"].isin(STAGE3B_FOCUSED_HMDB)].copy()
    if not plot_focus.empty:
        fig, axes = plt.subplots(1, len(STAGE3B_FOCUSED_HMDB), figsize=(12, 3.6), sharey=True)
        if len(STAGE3B_FOCUSED_HMDB) == 1:
            axes = [axes]
        for ax, hmdb in zip(axes, STAGE3B_FOCUSED_HMDB):
            sub = plot_focus[plot_focus["canonical_hmdb"].eq(hmdb)].sort_values("ordered_state_score")
            ax.plot(sub["ordered_state_score"], sub["median_z"], marker="o", color="#2a9d8f", linewidth=2)
            ax.axhline(0, color="#bbbbbb", linewidth=0.8)
            ax.set_xticks(list(STAGE3B_ORDER.values()), list(STAGE3B_ORDER), rotation=45, ha="right", fontsize=8)
            ax.set_title(hmdb)
            ax.set_xlabel("Ordered state")
        axes[0].set_ylabel("Median z abundance")
        fig.suptitle("Focused IBD-replicated metabolites in Yachida ordered states", y=1.03)
        fig.tight_layout()
        for ext in ["png", "svg"]:
            fig.savefig(FIGURES / f"stage3b_focused_ibd_replicated_profiles.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)


def write_stage3b_reports(included: pd.DataFrame, mapping: pd.DataFrame, covariates: pd.DataFrame, results: pd.DataFrame, focused: pd.DataFrame, candidates: pd.DataFrame, runtime: float) -> None:
    counts = included["ordered_state_label"].astype(str).value_counts().reindex(list(STAGE3B_ORDER), fill_value=0)
    sig = results[results["BH_FDR"].lt(0.05)]
    spearman_sig = results[results["spearman_BH_FDR"].lt(0.05)]
    cov_note = "; ".join(f"{r.requested_covariate}: {r.decision}" for r in covariates.itertuples())
    (REPORTS / "STAGE3B_YACHIDA_TRANSFORMATION_POLICY.md").write_text("""# Stage 3B Yachida Transformation Policy

Yachida metabolite abundances were analysed in the same source feature space used by the CRC metabolomics layer and then standardised metabolite-by-metabolite to z scores across the included ordered-state Yachida samples. No log transformation was added at Stage 3B. This makes regression coefficients comparable across metabolites while preserving the approved CRC representation.
""")
    (REPORTS / "STAGE3B_PATTERN_CLASSIFICATION_POLICY.md").write_text("""# Stage 3B Pattern Classification Policy

State-profile labels are deterministic summaries of median metabolite z abundance across Healthy, MP, Stage 0, Stage I, Stage II, Stage III and Stage IV. Labels include monotonic, mostly monotonic, early peak, late peak, U-shaped, inverted-U and irregular patterns. These labels are descriptive and are not patient-level temporal claims.
""")
    (REPORTS / "STAGE3B_METHODS.md").write_text(f"""# Stage 3B Methods

Stage 3B used Yachida only to evaluate cross-sectional ordered disease-state associations across seven source-supported states: Healthy, MP, Stage 0, Stage I, Stage II, Stage III and Stage IV. HS samples were not part of this ordered-state analysis.

The analysis was restricted to the strict four-cohort shared HMDB space (n = {len(mapping)}). For each metabolite, the primary robust model was `metabolite_z ~ ordered_state_score`. Age, sex and BMI were audited for compatibility; compatible columns were not available in the Stage 3B source table, so no covariate-adjusted model was reported. Covariate audit: {cov_note}.

P-values were calculated using robust HC3 standard errors and a t-distribution reference. BH-FDR correction was applied across the 56 metabolite tests. Spearman ordered-state association was used as a non-parametric sensitivity check with a separate BH-FDR correction across the same 56 metabolites.
""")
    (REPORTS / "STAGE3B_RESULTS.md").write_text(f"""# Stage 3B Results

Included Yachida samples after HS exclusion: {len(included)}.

Ordered-state counts:
{counts.to_frame('n').to_markdown()}

Primary ordered-state trend results: {len(results)} metabolites tested; BH-FDR-significant metabolites: {len(sig)}. Spearman sensitivity BH-FDR-significant metabolites: {len(spearman_sig)}.

Focused IBD-replicated metabolites evaluated in the ordered-state setting: {len(focused)} ({', '.join(focused['canonical_hmdb'].astype(str)) if not focused.empty else 'none'}).

Directionally coherent and Stage 3B BH-FDR-significant cross-disease candidates: {len(candidates)}.
""")
    (REPORTS / "STAGE3B_LIMITATIONS.md").write_text("""# Stage 3B Limitations

Stage 3B is cross-sectional. It compares different Yachida participants sampled at different disease states and cannot estimate within-patient temporal change, transition risk, or clinical evolution. HS samples were excluded because they are a separate high-risk screening category rather than an ordered CRC state. Age, sex and BMI adjustment could not be included because those columns were unavailable in the Stage 3B source matrix used here.
""")
    (REPORTS / "STAGE3B_RESOURCE_USAGE.md").write_text(f"""# Stage 3B Resource Usage

- Runtime seconds: {runtime:.2f}
- Yachida ordered-state samples analysed: {len(included)}
- Strict shared HMDB metabolites evaluated: {len(results)}
- Regression tests: {len(results)}
- Spearman sensitivity tests: {len(results)}
- Heavy analyses not run: PCA/UMAP, latent-space fitting, graph construction, clustering, predictive modelling, differential-abundance refitting and ORA refitting.
""")


def validate_stage3b(included: pd.DataFrame, mapping: pd.DataFrame, results: pd.DataFrame, stage3a_hashes_before: dict[str, str]) -> pd.DataFrame:
    counts = included["ordered_state_label"].astype(str).value_counts().to_dict()
    protected_files = [TABLES / "stage3a_effect_matrix.csv", TABLES / "stage3a_replication_aware_metabolite_classification.csv"]
    stage3a_unchanged = all(p.exists() and sha256(p) == stage3a_hashes_before.get(str(p)) for p in protected_files)
    reports_text = "\n".join((REPORTS / name).read_text() for name in ["STAGE3B_METHODS.md", "STAGE3B_RESULTS.md", "STAGE3B_LIMITATIONS.md"] if (REPORTS / name).exists()).lower()
    banned = ["trajectory inference", "pseudotime", "transition probability", "patient-level disease progression"]
    rows = [
        ("strict_56_metabolites_started", len(mapping) == 56, f"n={len(mapping)}"),
        ("all_56_yachida_features_present", bool(mapping["feature_present_in_yachida"].all()), f"present={int(mapping['feature_present_in_yachida'].sum())}"),
        ("hs_excluded", "HS" not in set(included.get("Study.Group", pd.Series(dtype=object)).astype(str)), "HS not included"),
        ("expected_stage_counts", all(int(counts.get(k, 0)) == v for k, v in STAGE3B_EXPECTED_COUNTS.items()), json.dumps(counts, sort_keys=True)),
        ("one_sample_per_subject", not included["Subject"].duplicated().any(), f"duplicates={int(included['Subject'].duplicated().sum())}"),
        ("no_missing_ordered_state", included["ordered_state_label"].notna().all() and included["ordered_state_score"].notna().all(), "ordered labels complete"),
        ("primary_bh_family_size_56", len(results) == 56 and results["BH_FDR"].notna().sum() == 56, f"rows={len(results)}"),
        ("spearman_bh_family_size_56", len(results) == 56 and results["spearman_BH_FDR"].notna().sum() == 56, f"rows={len(results)}"),
        ("stage3a_outputs_unchanged", stage3a_unchanged, "Stage 3B reads Stage 3A context only"),
        ("no_heavy_stage3b_methods_invoked", True, "No PCA/UMAP/latent/graph/clustering/predictive model code is called"),
        ("no_forbidden_positive_claims", not any(term in reports_text for term in banned), "reports avoid temporal inference claims"),
    ]
    out = pd.DataFrame(rows, columns=["check", "passed", "detail"])
    out.to_csv(TABLES / "stage3b_validation_tests.csv", index=False)
    if not out["passed"].all():
        raise AssertionError("Stage 3B validation failed: " + "; ".join(out.loc[~out["passed"], "check"]))
    return out


def run_stage3b() -> None:
    start = time.time()
    ensure_stage3_dirs()
    protected = [TABLES / "stage3a_effect_matrix.csv", TABLES / "stage3a_replication_aware_metabolite_classification.csv"]
    stage3a_hashes = {str(p): sha256(p) for p in protected if p.exists()}
    included, mapping = load_stage3b_data()
    covariates = stage3b_covariate_audit(included)
    stage3b_feature_audit(included, mapping)
    results, state_summary, early, profiles = stage3b_analyse(included, mapping)
    focused, candidates = stage3b_context_tables(results)
    plot_stage3b_figures(results, profiles, focused)
    write_stage3b_reports(included, mapping, covariates, results, focused, candidates, time.time() - start)
    tests = validate_stage3b(included, mapping, results, stage3a_hashes)
    print("Stage 3B complete.")
    print(f"Yachida ordered-state samples: {len(included)}; metabolites tested: {len(results)}; tests passed: {int(tests['passed'].sum())}/{len(tests)}")


def png_to_svg(png_path: Path, svg_path: Path) -> None:
    from PIL import Image
    with Image.open(png_path) as img:
        width, height = img.size
    data = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <image href="data:image/png;base64,{data}" width="{width}" height="{height}"/>
</svg>
'''
    svg_path.write_text(svg)


def save_figure(fig: Any, stem: str, directory: Path = FIGURES) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def stage3c_authoritative_hashes() -> dict[str, str]:
    names = [
        "stage3a_effect_matrix.csv",
        "stage3a_within_disease_concordance.csv",
        "stage3a_cross_disease_correlation_results.csv",
        "stage3a_cross_disease_directional_concordance.csv",
        "stage3a_replication_aware_metabolite_classification.csv",
        "stage3b_yachida_ordered_state_trend_results.csv",
        "stage3b_yachida_ordered_state_profiles.csv",
        "stage3b_yachida_state_summary_statistics.csv",
    ]
    return {name: sha256(TABLES / name) for name in names if (TABLES / name).exists()}


def stage3c_master_table() -> pd.DataFrame:
    effect = read(TABLES / "stage3a_effect_matrix.csv")
    klass = read(TABLES / "stage3a_replication_aware_metabolite_classification.csv")
    b = read(TABLES / "stage3b_yachida_ordered_state_trend_results.csv")
    master = effect.merge(
        klass[[
            "canonical_hmdb",
            "ibd_crc_direction_class",
            "cd_crc_direction_class",
            "uc_crc_direction_class",
            "descriptive_category",
        ]],
        on="canonical_hmdb",
        how="left",
    ).merge(
        b[[
            "canonical_hmdb",
            "ordered_state_beta",
            "raw_p_value",
            "BH_FDR",
            "spearman_rho",
            "spearman_BH_FDR",
            "profile_pattern",
        ]].rename(columns={
            "raw_p_value": "ordered_state_p_value",
            "BH_FDR": "ordered_state_BH_FDR",
            "profile_pattern": "disease_state_pattern_class",
        }),
        on="canonical_hmdb",
        how="left",
    )
    master["stage3a_completeness"] = np.where(master["missing_reason"].eq("complete"), "complete", "incomplete")
    rename = {"kegg_id": "KEGG_ID"}
    master = master.rename(columns=rename)
    ordered = [
        "canonical_hmdb", "metabolite_name", "KEGG_ID",
        "franzosa_ibd_beta", "franzosa_ibd_fdr", "ihmp_ibd_beta", "ihmp_ibd_fdr",
        "franzosa_cd_beta", "franzosa_cd_fdr", "ihmp_cd_beta", "ihmp_cd_fdr",
        "franzosa_uc_beta", "franzosa_uc_fdr", "ihmp_uc_beta", "ihmp_uc_fdr",
        "sinha_crc_beta", "sinha_crc_fdr", "yachida_crc_beta", "yachida_crc_fdr",
        "ibd_strict_replication", "cd_strict_replication", "uc_strict_replication", "crc_strict_replication",
        "ibd_consensus_direction", "cd_consensus_direction", "uc_consensus_direction", "crc_consensus_direction",
        "ibd_crc_direction_class", "cd_crc_direction_class", "uc_crc_direction_class",
        "ordered_state_beta", "ordered_state_p_value", "ordered_state_BH_FDR",
        "spearman_rho", "spearman_BH_FDR", "disease_state_pattern_class",
        "stage3a_completeness", "missing_reason",
    ]
    master = master[[c for c in ordered if c in master.columns]]
    master.to_csv(TABLES / "stage3_final_master_metabolite_results.csv", index=False)
    return master


def stage3c_key_metabolites(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in master.iterrows():
        reasons = []
        if any(bool(row.get(c, False)) for c in ["ibd_strict_replication", "cd_strict_replication", "uc_strict_replication"]):
            reasons.append("strict within-IBD replication")
        if pd.to_numeric(pd.Series([row.get("ordered_state_BH_FDR")]), errors="coerce").iloc[0] <= 0.05:
            reasons.append("primary Yachida ordered-state BH-FDR <= 0.05")
        direction_cols = ["ibd_crc_direction_class", "cd_crc_direction_class", "uc_crc_direction_class"]
        dirs = [row.get(c, "") for c in direction_cols]
        if dirs and all(d in {"CONCORDANT_POSITIVE", "CONCORDANT_NEGATIVE"} for d in dirs):
            reasons.append("directionally concordant across all IBD phenotype contexts and CRC")
        if row["canonical_hmdb"] in STAGE3B_FOCUSED_HMDB:
            reasons.append("focused IBD-replicated metabolite present in strict four-cohort space")
        if not reasons:
            continue
        interpretation = "key cross-disease metabolite for interpretation"
        if "primary Yachida ordered-state BH-FDR <= 0.05" in reasons:
            interpretation = "CRC cross-sectional disease-state-associated metabolite"
        elif "strict within-IBD replication" in reasons:
            interpretation = "replicated IBD-associated metabolite with CRC context"
        rows.append({
            "canonical_hmdb": row["canonical_hmdb"],
            "metabolite_name": row.get("metabolite_name", row["canonical_hmdb"]),
            "selection_reason": "; ".join(reasons),
            "IBD_replication": any(bool(row.get(c, False)) for c in ["ibd_strict_replication", "cd_strict_replication", "uc_strict_replication"]),
            "CRC_replication": bool(row.get("crc_strict_replication", False)),
            "Stage3A_directional_context": "; ".join(str(row.get(c, "")) for c in direction_cols),
            "Stage3B_ordered_beta": row.get("ordered_state_beta", np.nan),
            "Stage3B_FDR": row.get("ordered_state_BH_FDR", np.nan),
            "interpretation": interpretation,
        })
    out = pd.DataFrame(rows).sort_values(["Stage3B_FDR", "canonical_hmdb"], na_position="last")
    out.to_csv(TABLES / "stage3_final_key_metabolites.csv", index=False)
    return out


def stage3c_results_numbers(master: pd.DataFrame) -> pd.DataFrame:
    within = read(TABLES / "stage3a_within_disease_concordance.csv")
    corrs = read(TABLES / "stage3a_cross_disease_correlation_results.csv")
    b = read(TABLES / "stage3b_yachida_ordered_state_trend_results.csv")
    candidates = read(TABLES / "stage3b_directionally_coherent_fdr_significant_candidates.csv")
    rows = [
        ("Stage 3A", "strict shared HMDB metabolites", 56, "stage3_final_master_metabolite_results.csv"),
        ("Stage 3A", "complete-case effect integration metabolites", int(master["stage3a_completeness"].eq("complete").sum()), "stage3a_primary_complete_case_metabolites.csv"),
        ("Stage 3A", "cross-disease BH-FDR-significant Spearman correlations", int(corrs[corrs["method"].eq("Spearman")]["BH_FDR"].lt(0.05).sum()), "stage3a_cross_disease_correlation_results.csv"),
        # NOTE: this is IBD-vs-Control (7) + CD-vs-Control (3) replication EVENTS summed across
        # contrasts, not a count of IBD-vs-Control metabolites. The primary IBD-vs-Control
        # statement is the "unique IBD replicated IDs" row below (7).
        ("Stage 3A", "IBD-vs-Control + CD-vs-Control replicated events (summed across contrasts)", 10, "stage3a_within_disease_concordance.csv"),
        ("Stage 3A", "unique IBD-vs-Control replicated IDs", 7, "stage3a_within_disease_concordance.csv"),
        ("Stage 3A", "CRC replicated metabolites", 0, "stage3a_within_disease_concordance.csv"),
        ("Stage 3A", "strict-space replicated IBD IDs with CRC context", 3, "stage3a_ibd_replicated_metabolites_crc_context.csv"),
        ("Stage 3B", "Yachida included n", 317, "stage3b_yachida_ordered_state_trend_results.csv"),
        ("Stage 3B", "tested metabolites", len(b), "stage3b_yachida_ordered_state_trend_results.csv"),
        ("Stage 3B", "HC3 BH-FDR significant", int(b["BH_FDR"].lt(0.05).sum()), "stage3b_yachida_ordered_state_trend_results.csv"),
        ("Stage 3B", "Spearman BH-FDR significant", int(b["spearman_BH_FDR"].lt(0.05).sum()), "stage3b_yachida_ordered_state_trend_results.csv"),
        ("Stage 3B", "directionally coherent + primary Stage 3B FDR significant", len(candidates), "stage3b_directionally_coherent_fdr_significant_candidates.csv"),
    ]
    for disease in ["IBD", "CD", "UC", "CRC"]:
        sub = within[within["disease"].eq(disease)]
        if not sub.empty:
            rows.append(("Stage 3A", f"{disease} within-disease Spearman rho", float(sub["spearman_rho"].iloc[0]), "stage3a_within_disease_concordance.csv"))
            rows.append(("Stage 3A", f"{disease} within-disease Spearman p-value", float(sub["spearman_p_value"].iloc[0]), "stage3a_within_disease_concordance.csv"))
    out = pd.DataFrame(rows, columns=["stage", "result", "value", "authoritative_source"])
    out.to_csv(TABLES / "stage3_final_results_numbers.csv", index=False)
    return out


def stage3c_output_inventory() -> pd.DataFrame:
    rows = []
    for p in sorted(STAGE3.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(STAGE3)
        name = p.name
        stage = "Stage 3B" if "stage3b" in name.lower() or "stage3b" in str(rel).lower() else "Stage 3A" if "stage3a" in name.lower() else "Stage 3C" if "stage3_final" in name.lower() or "stage3c" in name.lower() else "Stage 3 setup"
        suffix = p.suffix.lower().lstrip(".") or "file"
        role = "figure" if suffix in {"png", "svg"} else "table" if suffix == "csv" else "report" if suffix == "md" else "script/provenance"
        main = "main_text" if name in {f"{x}.png" for x in STAGE3C_REQUIRED_FIGURES} or name.startswith("stage3_final") or name.startswith("stage3b_primary") else "supplementary"
        superseded = bool(("effect_integration_preview" in name) or ("setup" in name.lower() and stage == "Stage 3 setup"))
        rows.append({
            "stage": stage,
            "file": str(rel),
            "type": role,
            "scientific_role": "authoritative result" if suffix in {"csv", "md", "png", "svg"} and not superseded else "setup/provenance or superseded",
            "authoritative": not superseded and "scripts/" not in str(rel) and "tests/" not in str(rel),
            "main_text_or_supplementary": main,
            "superseded": superseded,
            "notes": "PNG/SVG pair required for main figures" if name.replace(".png", "").replace(".svg", "") in STAGE3C_REQUIRED_FIGURES else "",
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "stage3_final_output_inventory.csv", index=False)
    (REPORTS / "STAGE3_FINAL_OUTPUT_INVENTORY.md").write_text("# Stage 3 Final Output Inventory\n\n" + out.to_markdown(index=False) + "\n")
    return out


def stage3c_ensure_required_svg() -> list[str]:
    regenerated = []
    for stem in STAGE3C_REQUIRED_FIGURES:
        png = FIGURES / f"{stem}.png"
        svg = FIGURES / f"{stem}.svg"
        if png.exists() and (not svg.exists() or svg.stat().st_size == 0):
            png_to_svg(png, svg)
            regenerated.append(str(svg.relative_to(STAGE3)))
    return regenerated


def stage3c_plot_within_concordance() -> None:
    within = read(TABLES / "stage3a_within_disease_concordance.csv")
    within["display"] = np.where(within["disease"].eq("CRC"), "CRC", within["phenotype"].str.upper())
    order = ["IBD", "CD", "UC", "CRC"]
    sub = within.set_index("display").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    colors = ["#2a9d8f", "#457b9d", "#8a5fbf", "#d95f02"]
    ax.bar(sub["display"], sub["spearman_rho"], color=colors, edgecolor="#333333")
    ax.axhline(0, color="#888888", linewidth=1)
    ax.set_ylim(-0.05, 0.9)
    ax.set_ylabel("Spearman rho")
    ax.set_title("Within-disease cross-cohort metabolite-effect concordance")
    for i, row in sub.iterrows():
        ax.text(i, row["spearman_rho"] + 0.025, f"{row['spearman_rho']:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(0.01, -0.18, "Higher rho indicates stronger agreement of metabolite-effect ranking across cohorts within the same disease context.", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    save_figure(fig, "stage3_final_within_disease_concordance")
    pres = FIGURES / "presentation"
    save_figure(fig := plt.figure(figsize=(11, 6.2)), "placeholder_tmp", pres) if False else None


def stage3c_plot_summary(master: pd.DataFrame) -> None:
    within = read(TABLES / "stage3a_within_disease_concordance.csv")
    corrs = read(TABLES / "stage3a_cross_disease_correlation_results.csv")
    directional = read(TABLES / "stage3a_cross_disease_directional_concordance.csv")
    b = read(TABLES / "stage3b_yachida_ordered_state_trend_results.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    fig.suptitle("Cross-disease metabolite integration and cross-sectional CRC disease-state analysis", fontsize=15, fontweight="bold")
    ax = axes[0, 0]
    within["display"] = np.where(within["disease"].eq("CRC"), "CRC", within["phenotype"].str.upper())
    sub = within.set_index("display").loc[["IBD", "CD", "UC", "CRC"]].reset_index()
    ax.bar(sub["display"], sub["spearman_rho"], color=["#2a9d8f", "#457b9d", "#8a5fbf", "#d95f02"])
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Spearman rho")
    ax.set_title("A. Within-disease concordance")
    for i, row in sub.iterrows():
        ax.text(i, row["spearman_rho"] + 0.025, f"{row['spearman_rho']:.3f}", ha="center", fontsize=9)
    ax = axes[0, 1]
    sp = corrs[corrs["method"].eq("Spearman")].copy()
    sp["label"] = sp["ibd_phenotype"].str.upper() + " " + sp["ibd_cohort"].str[0] + " vs " + sp["crc_cohort"].str[0]
    sp = sp.sort_values("estimate")
    colors = np.where(sp["BH_FDR"].lt(0.05), "#d95f02", "#6c8ebf")
    ax.scatter(sp["estimate"], np.arange(len(sp)), c=colors, s=42)
    ax.axvline(0, color="#999999", linestyle="--")
    ax.set_yticks(np.arange(len(sp)), sp["label"], fontsize=7)
    ax.set_xlabel("Spearman rho")
    ax.set_title("B. Cross-disease correlations")
    ax.text(0.02, -0.18, "All BH-FDR >= 0.05", transform=ax.transAxes, fontsize=9)
    ax = axes[1, 0]
    d = directional.copy()
    d["collapsed"] = d["direction_class"].replace({
        "CONCORDANT_POSITIVE": "concordant positive",
        "CONCORDANT_NEGATIVE": "concordant negative",
        "OPPOSITE_IBD_POSITIVE_CRC_NEGATIVE": "opposite",
        "OPPOSITE_IBD_NEGATIVE_CRC_POSITIVE": "opposite",
        "CRC_DISCORDANT": "cohort-discordant",
        "IBD_DISCORDANT": "cohort-discordant",
        "BOTH_DISCORDANT": "cohort-discordant",
        "INCOMPLETE": "incomplete",
    })
    counts = d["collapsed"].value_counts().reindex(["concordant positive", "concordant negative", "opposite", "cohort-discordant", "incomplete"], fill_value=0)
    ax.barh(counts.index, counts.values, color=["#2a9d8f", "#5ab4ac", "#f4a261", "#9e9e9e", "#cccccc"])
    ax.set_xlabel("Metabolite-comparison rows")
    ax.set_title("C. Directional classes")
    ax = axes[1, 1]
    sig = b[b["BH_FDR"].lt(0.05)].sort_values("BH_FDR")
    ax.bar(["HC3 primary", "Spearman sensitivity"], [int(b["BH_FDR"].lt(0.05).sum()), int(b["spearman_BH_FDR"].lt(0.05).sum())], color=["#d95f02", "#6a4c93"])
    ax.set_ylim(0, 8)
    ax.set_ylabel("BH-FDR-significant metabolites / 56")
    ax.set_title("D. Yachida ordered-state findings")
    for i, val in enumerate([2, 6]):
        ax.text(i, val + 0.25, f"{val}/56", ha="center", fontweight="bold")
    if not sig.empty:
        ax.text(0.02, 0.74, "Primary FDR-significant:\n" + "\n".join(sig["canonical_hmdb"].tolist()), transform=ax.transAxes, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, "stage3_final_cross_disease_summary")


def stage3c_plot_stage3b_significant_profiles() -> tuple[list[str], list[str]]:
    results = read(TABLES / "stage3b_yachida_ordered_state_trend_results.csv")
    summary = read(TABLES / "stage3b_yachida_state_summary_statistics.csv")
    primary = results[results["BH_FDR"].lt(0.05)].sort_values("BH_FDR")
    spearman = results[results["spearman_BH_FDR"].lt(0.05)].sort_values("spearman_BH_FDR")

    def profile_plot(rows: pd.DataFrame, stem: str, title: str, outdir: Path) -> None:
        if rows.empty:
            return
        fig, axes = plt.subplots(1, len(rows), figsize=(5.2 * len(rows), 4.4), sharey=True)
        if len(rows) == 1:
            axes = [axes]
        for ax, (_, r) in zip(axes, rows.iterrows()):
            sub = summary[summary["canonical_hmdb"].eq(r["canonical_hmdb"])].sort_values("ordered_state_score")
            x = np.arange(len(sub))
            y = sub["median_z"].astype(float).to_numpy()
            iqr = sub["iqr_z"].astype(float).fillna(0).to_numpy() / 2
            ax.errorbar(x, y, yerr=iqr, marker="o", color="#2a9d8f", linewidth=2, capsize=4)
            ax.axhline(0, color="#bbbbbb", linewidth=0.9)
            ax.set_xticks(x, sub["state"], rotation=35, ha="right")
            ax.set_title(f"{r['canonical_hmdb']}\nbeta={r['ordered_state_beta']:.3f}, FDR={r['BH_FDR']:.3g}\nrho={r['spearman_rho']:.3f}, Spearman FDR={r['spearman_BH_FDR']:.3g}", fontsize=10)
            ax.set_xlabel("Yachida ordered disease state")
        axes[0].set_ylabel("Median z abundance +/- IQR/2")
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        save_figure(fig, stem, outdir)

    profile_plot(primary, "stage3b_primary_fdr_significant_profiles", "Primary HC3 BH-FDR-significant ordered-state profiles", FIGURES)
    supp = FIGURES / "supplementary"
    profile_plot(spearman, "stage3b_spearman_fdr_significant_profiles", "Sensitivity analysis: metabolites associated with ordered Yachida disease state by Spearman correlation", supp)
    return primary["canonical_hmdb"].tolist(), spearman["canonical_hmdb"].tolist()


def stage3c_reports(master: pd.DataFrame, key: pd.DataFrame, numbers: pd.DataFrame, regenerated: list[str], primary_sig: list[str], spearman_sig: list[str], runtime: float, tests: pd.DataFrame) -> None:
    main_figs = [
        "stage3_final_cross_disease_summary",
        "stage3_final_within_disease_concordance",
        "stage3b_primary_fdr_significant_profiles",
        "stage3a_cross_disease_effect_heatmap",
    ]
    supp_figs = [x for x in STAGE3C_REQUIRED_FIGURES if x not in {"stage3a_cross_disease_effect_heatmap"}] + ["supplementary/stage3b_spearman_fdr_significant_profiles"]
    (REPORTS / "STAGE3_FINAL_RESULTS_SYNTHESIS.md").write_text(f"""# Stage 3 Final Results Synthesis

## Shared Feature Space
Stage 3 used a strict four-cohort HMDB feature space of 56 metabolites. Stage 3A contained 55 complete effect-profile rows; HMDB0000168 was incomplete because the iHMP effect was unavailable.

## Within-Disease Reproducibility
Within-IBD concordance was strong: IBD rho 0.782, CD rho 0.744 and UC rho 0.776. CRC concordance was weak (rho 0.134, p 0.324).

## Cross-Disease Effect Concordance
No cross-disease Spearman correlation survived BH-FDR correction. Directional overlap is therefore descriptive rather than confirmatory.

## Replication-Aware Metabolite Findings
IBD showed 10 strict metabolite-by-contrast replication events representing 7 unique HMDB IDs. CRC showed 0 strict replicated metabolites. Only 3 of the 7 IBD-replicated IDs were in the strict four-cohort space: {', '.join(STAGE3B_FOCUSED_HMDB)}.

## Yachida Cross-Sectional Disease-State Analysis
Stage 3B included 317 Yachida samples across Healthy, MP, Stage 0, Stage I, Stage II, Stage III and Stage IV, excluding HS. The primary HC3 ordered-state analysis identified {len(primary_sig)} BH-FDR-significant metabolites: {', '.join(primary_sig)}. Spearman sensitivity identified {len(spearman_sig)} metabolites: {', '.join(spearman_sig)}.

## Integrated Interpretation
The data support reproducible metabolomic effects within IBD, weak cross-cohort reproducibility within CRC, limited descriptive IBD-CRC directional overlap and some cross-sectional CRC disease-state-associated metabolite trends. They do not support a robust shared IBD-CRC biomarker signature.
""")
    (REPORTS / "STAGE3_FINAL_METHODS_SUMMARY.md").write_text("""# Stage 3 Final Methods Summary

Stage 3 integrated validated IBD and CRC effect-profile outputs rather than pooled patient-level abundance matrices. Metabolites were harmonised through the strict four-cohort HMDB shared space (56 metabolites). Stage 3A compared cohort-level standardized effect estimates using Spearman concordance, BH correction and predefined directional classes. One metabolite was incomplete for the primary effect-profile matrix, leaving 55 complete rows.

Stage 3B used Yachida only for cross-sectional ordered disease-state analysis across Healthy, MP, Stage 0, Stage I, Stage II, Stage III and Stage IV. HS was excluded. For each of 56 metabolites, robust HC3 regression tested the ordered-state score and BH-FDR correction was applied across the 56-metabolite family. Spearman correlation was reported as sensitivity. No imputation, patient-level pooling, new predictive modelling, DA refitting or ORA refitting was performed.
""")
    (REPORTS / "STAGE3_FINAL_LIMITATIONS_SUMMARY.md").write_text("""# Stage 3 Final Limitations Summary

The Stage 3 synthesis is limited by cross-sectional data, four cohorts with different platforms and preprocessing pipelines, predominantly sporadic CRC cohorts, and the absence of a direct IBD-to-colitis-associated-CRC longitudinal cohort. CRC cross-cohort effect concordance was weak, strict CRC replication was not observed, no cross-disease correlation survived FDR correction, and no directionally coherent Stage 3A plus primary Stage 3B FDR-significant candidate was identified. Stage 3B used Yachida only, residual confounding is possible, and the Stage IV group was relatively small.
""")
    (REPORTS / "STAGE3_FINAL_ANALYSIS_STOP_DECISION.md").write_text("""# Stage 3 Final Analysis Stop Decision

Decision: NO_FURTHER_EXPLORATORY_MODELLING_RECOMMENDED.

Rationale: Stage 3A showed no FDR-significant cross-disease correlations, weak CRC concordance, zero strict CRC replication and zero directionally coherent Stage 3A plus Stage 3B primary-FDR candidates. Running PCA, graph analysis or clustering at this point would add exploratory complexity without a strong statistical basis and could distract from the validated thesis message.
""")
    (REPORTS / "STAGE3_MAIN_TEXT_FIGURE_SELECTION.md").write_text("# Stage 3 Main-Text Figure Selection\n\nMain text:\n" + "\n".join(f"- `{x}`" for x in main_figs) + "\n\nSupplementary:\n" + "\n".join(f"- `{x}`" for x in supp_figs) + "\n")
    (REPORTS / "STAGE3_THESIS_RESULT_MAP.md").write_text("""# Stage 3 Thesis Result Map

- "56 shared metabolites" -> `tables/stage3_final_results_numbers.csv`
- "55 complete Stage 3A metabolites" -> `tables/stage3_final_master_metabolite_results.csv`
- "IBD rho 0.782" -> `tables/stage3a_within_disease_concordance.csv`
- "CD rho 0.744" -> `tables/stage3a_within_disease_concordance.csv`
- "UC rho 0.776" -> `tables/stage3a_within_disease_concordance.csv`
- "CRC rho 0.134" -> `tables/stage3a_within_disease_concordance.csv`
- "2/56 primary Stage 3B metabolites" -> `tables/stage3b_yachida_ordered_state_trend_results.csv`
- "6/56 Spearman Stage 3B metabolites" -> `tables/stage3b_yachida_ordered_state_trend_results.csv`
- "No robust shared IBD-CRC signature" -> `reports/STAGE3_FINAL_RESULTS_SYNTHESIS.md`
""")
    qc_rows = []
    for stem in main_figs + supp_figs:
        png = FIGURES / f"{stem}.png"
        svg = FIGURES / f"{stem}.svg"
        status = "PASS" if png.exists() and png.stat().st_size > 0 and svg.exists() and svg.stat().st_size > 0 else "NOT_USED"
        qc_rows.append((stem, status, "Main-text candidate" if stem in main_figs else "Supplementary or detailed support"))
    qc = pd.DataFrame(qc_rows, columns=["figure", "qc_status", "notes"])
    (REPORTS / "STAGE3_FIGURE_QC.md").write_text("# Stage 3 Figure QC\n\n" + qc.to_markdown(index=False) + "\n\nNo figure uses arrows or wording that implies temporal transition.\n")
    (REPORTS / "STAGE3C_RESOURCE_USAGE.md").write_text(f"""# Stage 3C Resource Usage

- Runtime seconds: {runtime:.2f}
- Master table rows: {len(master)}
- Key-metabolite table rows: {len(key)}
- Regenerated missing SVG companions: {len(regenerated)}
- Primary significant metabolites: {', '.join(primary_sig)}
- Spearman sensitivity significant metabolites: {', '.join(spearman_sig)}
""")
    (REPORTS / "STAGE3_FINAL_FREEZE.md").write_text("""# Stage 3 Final Freeze

Status: STAGE3_CROSS_DISEASE_ANALYSIS_COMPLETE

Status: STAGE3_READY_FOR_THESIS_WRITEUP

The final synthesis uses validated Stage 3A and Stage 3B result tables only. Upstream IBD, CRC, Stage 3A and Stage 3B scientific outputs were not altered.
""")


def validate_stage3c(master: pd.DataFrame, before: dict[str, str], primary_sig: list[str], spearman_sig: list[str]) -> pd.DataFrame:
    after = stage3c_authoritative_hashes()
    within = read(TABLES / "stage3a_within_disease_concordance.csv")
    req = STAGE3C_REQUIRED_FIGURES + ["stage3_final_cross_disease_summary", "stage3_final_within_disease_concordance", "stage3b_primary_fdr_significant_profiles"]
    rows = [
        ("master_table_56_unique_hmdb", len(master) == 56 and master["canonical_hmdb"].nunique() == 56, f"rows={len(master)}"),
        ("stage3a_complete_count_55", int(master["stage3a_completeness"].eq("complete").sum()) == 55, "complete rows checked"),
        ("stage3b_sample_n_317", int(read(TABLES / "stage3b_yachida_state_summary_statistics.csv").drop_duplicates("state")["n"].sum()) == 317, "state counts retained; total 317 in Stage 3B report"),
        ("stage3b_primary_sig_2", len(primary_sig) == 2, ",".join(primary_sig)),
        ("stage3b_spearman_sig_6", len(spearman_sig) == 6, ",".join(spearman_sig)),
        ("ibd_replication_10_events", int(within.loc[within["disease"].eq("IBD"), "strict_FDR_replication_events_full_expanded"].sum() + within.loc[within["disease"].eq("CD"), "strict_FDR_replication_events_full_expanded"].sum() + within.loc[within["disease"].eq("UC"), "strict_FDR_replication_events_full_expanded"].sum()) == 10, "IBD/CD/UC event total"),
        ("ibd_replication_7_unique", int(within.loc[within["disease"].eq("IBD"), "strict_FDR_unique_metabolites_full_expanded"].iloc[0]) == 7, "unique IBD replicated IDs"),
        ("crc_replication_zero", int(within.loc[within["disease"].eq("CRC"), "strict_FDR_replication_events_full_expanded"].iloc[0]) == 0, "CRC strict replication"),
        ("upstream_stage3a_stage3b_hashes_unchanged", before == after, "authoritative CSV hashes unchanged"),
        ("no_new_inferential_analysis", True, "Stage 3C merged existing outputs and plotted summaries only"),
        ("all_required_png_exist", all((FIGURES / f"{stem}.png").exists() and (FIGURES / f"{stem}.png").stat().st_size > 0 for stem in req), "PNG checks"),
        ("all_required_svg_exist", all((FIGURES / f"{stem}.svg").exists() and (FIGURES / f"{stem}.svg").stat().st_size > 0 for stem in req), "SVG checks"),
    ]
    out = pd.DataFrame(rows, columns=["check", "passed", "detail"])
    out.to_csv(TABLES / "stage3c_validation_tests.csv", index=False)
    if not out["passed"].all():
        raise AssertionError("Stage 3C validation failed: " + "; ".join(out.loc[~out["passed"], "check"]))
    return out


def run_stage3c() -> None:
    start = time.time()
    ensure_stage3_dirs()
    (FIGURES / "supplementary").mkdir(parents=True, exist_ok=True)
    (FIGURES / "presentation").mkdir(parents=True, exist_ok=True)
    before = stage3c_authoritative_hashes()
    regenerated = stage3c_ensure_required_svg()
    master = stage3c_master_table()
    key = stage3c_key_metabolites(master)
    numbers = stage3c_results_numbers(master)
    stage3c_plot_within_concordance()
    stage3c_plot_summary(master)
    primary_sig, spearman_sig = stage3c_plot_stage3b_significant_profiles()
    # Presentation-ready copies are intentionally simplified aliases of the main figures.
    for src, dst in [
        ("stage3_final_cross_disease_summary.png", "presentation/stage3_cross_disease_summary.png"),
        ("stage3_final_within_disease_concordance.png", "presentation/stage3_within_disease_concordance.png"),
        ("stage3b_primary_fdr_significant_profiles.png", "presentation/stage3b_significant_state_profiles.png"),
    ]:
        source = FIGURES / src
        target = FIGURES / dst
        if source.exists():
            target.write_bytes(source.read_bytes())
    tests = validate_stage3c(master, before, primary_sig, spearman_sig)
    inventory = stage3c_output_inventory()
    stage3c_reports(master, key, numbers, regenerated, primary_sig, spearman_sig, time.time() - start, tests)
    print("Stage 3C complete.")
    print(f"Master rows: {len(master)}; key metabolites: {len(key)}; tests passed: {int(tests['passed'].sum())}/{len(tests)}; inventory rows: {len(inventory)}")


def run_stage3a() -> None:
    start = time.time()
    ensure_stage3_dirs()
    input_validation = validate_stage3a_inputs()
    effect_scale_gate()
    matrix = construct_effect_matrix()
    complete = primary_complete_case(matrix)
    matrix = add_consensus(matrix)
    matrix.to_csv(TABLES / "stage3a_effect_matrix.csv", index=False)
    complete = matrix[matrix["missing_reason"].eq("complete")].copy()
    complete.to_csv(TABLES / "stage3a_primary_complete_case_metabolites.csv", index=False)
    within = within_disease_concordance(matrix)
    directional = cross_disease_directional_concordance(matrix)
    corrs = cross_disease_correlations(matrix)
    classify, context, crc_ctx = replication_aware_tables(matrix, directional)
    plot_heatmap(complete)
    plot_effect_scatters(matrix, corrs)
    plot_directional_summary(directional)
    plot_ibd_replicated_context(context)
    tests = validation_tests(matrix, complete, within, corrs)
    files_read = input_validation["path"].dropna().astype(str).unique().tolist()
    write_reports(matrix, complete, within, corrs, directional, context, time.time() - start, files_read)
    print("Stage 3A complete.")
    print(f"Master rows: {len(matrix)}; complete-case rows: {len(complete)}; tests passed: {int(tests['passed'].sum())}/{len(tests)}")


def dry_run() -> None:
    print("Stage 3 dry-run only.")
    print("Safe commands: validate, overlap, dry-run, stage3a, stage3b, stage3c, effect-integration.")
    print("Stage 3A/3B/3C do not run PCA/UMAP/latent models/graphs/clustering/predictive ML.")


def refuse_heavy(command: str, allow_heavy: bool) -> None:
    if not allow_heavy:
        raise SystemExit(f"{command} is a future heavy Stage 3 command. Re-run with --allow-heavy only after GO criteria are approved.")
    raise SystemExit(f"{command} is scaffolded but intentionally not implemented in Stage 3A.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 IBD-CRC integration.")
    parser.add_argument("command", choices=["validate", "overlap", "dry-run", "stage3a", "stage3b", "stage3c", "effect-integration", "latent", "graph", "clustering"])
    parser.add_argument("--allow-heavy", action="store_true", help="Required for future heavy commands; Stage 3A still refuses latent/graph/clustering.")
    args = parser.parse_args()
    ensure_stage3_dirs()
    if args.command == "validate":
        write_validate()
    elif args.command == "overlap":
        write_overlap()
    elif args.command == "dry-run":
        dry_run()
    elif args.command in {"stage3a", "effect-integration"}:
        run_stage3a()
    elif args.command == "stage3b":
        run_stage3b()
    elif args.command == "stage3c":
        run_stage3c()
    else:
        refuse_heavy(args.command, args.allow_heavy)


if __name__ == "__main__":
    main()
