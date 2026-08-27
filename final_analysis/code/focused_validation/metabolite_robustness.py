from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist


REPO = Path(__file__).resolve().parents[3]
EXPECTED_NAMES = {
    "taurine",
    "ketodeoxycholate",
    "arachidonic acid",
    "putrescine",
    "N-acetylputrescine",
    "adrenic acid",
    "eicosatrienoic acid",
}
COHORTS = ("Franzosa", "iHMP")


def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    tmp.replace(path)


def bh_fdr(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    q = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    if valid.empty:
        return q
    n = len(valid)
    ranks = np.arange(1, n + 1)
    adjusted = valid.to_numpy() * n / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q.loc[valid.index] = np.clip(adjusted, 0, 1)
    return q


def robust_ols_hc3(y: pd.Series, x: pd.DataFrame, exposure_col: str) -> dict[str, Any]:
    data = pd.concat([y.rename("y"), x], axis=1).dropna()
    if len(data) < max(8, x.shape[1] + 3) or data[exposure_col].nunique() < 2:
        return {"status": "insufficient_complete_cases", "n": len(data)}
    yv = data["y"].astype(float).to_numpy()
    xv = data[x.columns].astype(float).to_numpy()
    try:
        beta = np.linalg.lstsq(xv, yv, rcond=None)[0]
        resid = yv - xv @ beta
        xtx_inv = np.linalg.pinv(xv.T @ xv)
        hat = np.sum((xv @ xtx_inv) * xv, axis=1)
        scale = (resid / np.clip(1 - hat, 1e-8, None)) ** 2
        cov = xtx_inv @ (xv.T @ (xv * scale[:, None])) @ xtx_inv
        se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
        idx = list(x.columns).index(exposure_col)
        coef = float(beta[idx])
        se_exp = float(se[idx])
        y_sd = float(np.nanstd(yv, ddof=1)) if len(yv) > 1 else np.nan
        t_stat = coef / se_exp if se_exp > 0 else np.nan
        df_resid = len(data) - x.shape[1]
        p_val = float(2 * t_dist.sf(abs(t_stat), df_resid)) if np.isfinite(t_stat) and df_resid > 0 else np.nan
        return {
            "status": "estimated_hc3",
            "n": len(data),
            "coef": coef,
            "se": se_exp,
            "ci_low": coef - 1.96 * se_exp if np.isfinite(se_exp) else np.nan,
            "ci_high": coef + 1.96 * se_exp if np.isfinite(se_exp) else np.nan,
            "standardized_coef": coef / y_sd if np.isfinite(y_sd) and y_sd > 0 else np.nan,
            "standardized_se": se_exp / y_sd if np.isfinite(y_sd) and y_sd > 0 else np.nan,
            "p_value": p_val,
            "complete_case_indices": data.index.to_list(),
            "hat": hat,
        }
    except Exception as exc:
        return {"status": f"model_failed:{type(exc).__name__}", "n": len(data)}


def model_matrix(thesis: Path, cohort: str) -> pd.DataFrame:
    included = pd.read_csv(thesis / "ibd_pipeline" / "tables" / "included_samples.csv")
    meta = included[included["cohort"].eq(cohort)].copy()
    matrix_name = "franzosa_primary_high_confidence_hmdb_matrix.csv" if cohort == "Franzosa" else "ihmp_primary_high_confidence_hmdb_matrix.csv"
    matrix = pd.read_csv(thesis / "ibd_pipeline" / "data_processed" / matrix_name)
    return meta.merge(matrix, on="sample_id", how="inner")


def ibd_design(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    sub = df[df["harmonised_label"].isin(["Control", "CD", "UC"])].copy()
    y = (sub["harmonised_label"] != "Control").astype(int)
    design = pd.DataFrame({"intercept": 1.0, "disease_status": y.astype(float)}, index=sub.index)
    design["age_years"] = pd.to_numeric(sub["age_years"], errors="coerce")
    return sub, y, design


def claim_set(thesis: Path) -> pd.DataFrame:
    prov = pd.read_csv(thesis / "stage3_ibd_crc_integration" / "manifests" / "stage2_3_manuscript_claim_provenance.csv")
    row = prov[prov["claim_id"].eq("seven_replicated_ibd_metabolites")]
    if row.empty:
        raise RuntimeError("Authoritative seven-metabolite claim provenance row is missing.")
    audit = pd.read_csv(thesis / "tables" / "ibd_expanded_replication_final_audit.csv")
    reps = audit[(audit["row_type"].eq("primary_replicated_metabolite")) & (audit["contrast"].eq("ibd_vs_control"))].copy()
    reps = reps.rename(columns={"canonical_identifier": "HMDB_ID"})
    found = set(reps["metabolite_name"].astype(str))
    if found != EXPECTED_NAMES:
        raise RuntimeError(f"Unexpected replicated metabolite set: {sorted(found)}")
    return reps[["HMDB_ID", "metabolite_name"]].drop_duplicates().sort_values("metabolite_name")


def primary_audit(thesis: Path, tables: Path, targets: pd.DataFrame) -> pd.DataFrame:
    da = pd.read_csv(thesis / "ibd_pipeline" / "tables" / "biological_primary_expanded" / "ibd_expanded_da_all_results.csv")
    meta = pd.read_csv(thesis / "ibd_pipeline" / "data_processed" / "primary_high_confidence_hmdb_feature_metadata.csv")
    meta = meta.rename(columns={"metabolite_id": "HMDB_ID"})
    rows = []
    for cohort in COHORTS:
        df = model_matrix(thesis, cohort)
        sub, _, design = ibd_design(df)
        for _, target in targets.iterrows():
            hmdb = target["HMDB_ID"]
            fit = robust_ols_hc3(pd.to_numeric(sub[hmdb], errors="coerce"), design, "disease_status")
            auth = da[(da["cohort"].eq(cohort)) & (da["outcome"].eq("ibd_vs_control")) & (da["hmdb_id"].eq(hmdb))]
            if auth.empty:
                raise RuntimeError(f"Missing authoritative DA row for {cohort} {hmdb}")
            a = auth.iloc[0]
            source = meta[meta["HMDB_ID"].eq(hmdb)].iloc[0]
            coef_delta = abs(float(fit["coef"]) - float(a["disease_coefficient"]))
            p_delta = abs(float(fit["p_value"]) - float(a["raw_p_value"]))
            if coef_delta > 1e-6 or p_delta > 1e-8:
                raise RuntimeError(f"Primary coefficient/p-value reproduction failed for {cohort} {hmdb}: {coef_delta=} {p_delta=}")
            rows.append(
                {
                    "cohort": cohort,
                    "authoritative_canonical_identifier": hmdb,
                    "HMDB_identifier": hmdb,
                    "display_name": target["metabolite_name"],
                    "source_matrix_column_in_Franzosa": source["franzosa_feature"],
                    "source_matrix_column_in_iHMP": source["ihmp_feature"],
                    "full_data_disease_coefficient": fit["coef"],
                    "HC3_standard_error": fit["se"],
                    "ci_low": fit["ci_low"],
                    "ci_high": fit["ci_high"],
                    "raw_p_value": fit["p_value"],
                    "BH_adjusted_p_value": a["fdr_q_value"],
                    "direction": "higher_in_disease" if fit["coef"] > 0 else "lower_in_disease",
                    "tested_family_size": int(auth["fdr_q_value"].notna().sum() and len(da[(da["cohort"].eq(cohort)) & (da["outcome"].eq("ibd_vs_control")) & (da["raw_p_value"].notna())])),
                    "analysis_sample_size": fit["n"],
                    "model_formula": "metabolite ~ disease_status + age_years",
                    "reproduces_authoritative_primary_result": True,
                }
            )
    out = pd.DataFrame(rows)
    atomic_csv(tables / "replicated_metabolite_primary_result_audit.csv", out)
    return out


def distribution_figure(thesis: Path, figures: Path, tables: Path, targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in COHORTS:
        df = model_matrix(thesis, cohort)
        sub, _, _ = ibd_design(df)
        for _, target in targets.iterrows():
            vals = pd.to_numeric(sub[target["HMDB_ID"]], errors="coerce")
            mu, sd = vals.mean(skipna=True), vals.std(skipna=True, ddof=1)
            z = (vals - mu) / sd if np.isfinite(sd) and sd > 0 else vals * np.nan
            for i, (idx, value) in enumerate(z.items(), start=1):
                if pd.isna(value):
                    continue
                rows.append(
                    {
                        "anonymous_plot_id": f"{cohort}_{target['HMDB_ID']}_{i:03d}",
                        "cohort": cohort,
                        "binary_group": "IBD" if sub.loc[idx, "harmonised_label"] != "Control" else "Control",
                        "HMDB_identifier": target["HMDB_ID"],
                        "display_name": target["metabolite_name"],
                        "within_cohort_standardized_plot_value": float(value),
                    }
                )
    source = pd.DataFrame(rows)
    atomic_csv(tables / "supp_figure_s1_source_values.csv", source)
    order = targets["metabolite_name"].tolist()
    colours = {"Control": "#0072B2", "IBD": "#D55E00"}
    fig, axes = plt.subplots(len(order), 2, figsize=(9.5, 14), sharey=False)
    rng = np.random.default_rng(20260813)
    for r, name in enumerate(order):
        for c, cohort in enumerate(COHORTS):
            ax = axes[r, c]
            panel = source[(source["display_name"].eq(name)) & (source["cohort"].eq(cohort))]
            groups = ["Control", "IBD"]
            data = [panel.loc[panel["binary_group"].eq(g), "within_cohort_standardized_plot_value"].to_numpy() for g in groups]
            bp = ax.boxplot(data, positions=[0, 1], widths=0.55, patch_artist=True, showfliers=False, medianprops={"color": "black", "linewidth": 1.4})
            for patch, g in zip(bp["boxes"], groups):
                patch.set_facecolor(colours[g])
                patch.set_alpha(0.28)
                patch.set_edgecolor("black")
            for pos, g, arr in zip([0, 1], groups, data):
                jitter = rng.normal(pos, 0.055, size=len(arr))
                ax.scatter(jitter, arr, s=12, alpha=0.7, color=colours[g], edgecolor="white", linewidth=0.25)
                ax.text(pos, ax.get_ylim()[1], f"n={len(arr)}", ha="center", va="top", fontsize=7)
            ax.axhline(0, color="#888888", linewidth=0.7, linestyle=":")
            ax.set_xticks([0, 1], groups)
            ax.set_title(f"{cohort}: {name}", fontsize=9)
            if c == 0:
                ax.set_ylabel("Within-cohort z-score")
            ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([0], [0], marker="o", color="w", label=k, markerfacecolor=v, markersize=7) for k, v in colours.items()]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, title="Group")
    fig.suptitle("Replicated IBD metabolites by cohort and disease group", y=0.995, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.006, "Values are standardised within cohort for plotting only; this is not a common concentration scale.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 0.975])
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "supp_figure_s1_replicated_metabolite_distributions.png", dpi=300)
    fig.savefig(figures / "supp_figure_s1_replicated_metabolite_distributions.pdf")
    plt.close(fig)
    return source


def loo_influence(thesis: Path, tables: Path, targets: pd.DataFrame, primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows, summary_rows = [], []
    for cohort in COHORTS:
        df = model_matrix(thesis, cohort)
        sub, _, design = ibd_design(df)
        for _, target in targets.iterrows():
            hmdb = target["HMDB_ID"]
            full = robust_ols_hc3(pd.to_numeric(sub[hmdb], errors="coerce"), design, "disease_status")
            cc_idx = list(full.get("complete_case_indices", []))
            coefs, pvals, dfbetas = [], [], []
            for omit_n, idx in enumerate(cc_idx, start=1):
                keep = sub.index.difference([idx], sort=False)
                fit = robust_ols_hc3(pd.to_numeric(sub.loc[keep, hmdb], errors="coerce"), design.loc[keep], "disease_status")
                coef = fit.get("coef", np.nan)
                se = fit.get("se", np.nan)
                dfbeta = (coef - full["coef"]) / se if np.isfinite(se) and se > 0 else np.nan
                coefs.append(coef)
                pvals.append(fit.get("p_value", np.nan))
                dfbetas.append(dfbeta)
                detail_rows.append(
                    {
                        "cohort": cohort,
                        "HMDB_identifier": hmdb,
                        "display_name": target["metabolite_name"],
                        "anonymous_omission_index": omit_n,
                        "disease_coefficient": coef,
                        "HC3_standard_error": se,
                        "ci_low": fit.get("ci_low", np.nan),
                        "ci_high": fit.get("ci_high", np.nan),
                        "raw_p_value": fit.get("p_value", np.nan),
                        "dfbeta_like_influence": dfbeta,
                    }
                )
            coef_s = pd.Series(coefs, dtype=float)
            p_s = pd.Series(pvals, dtype=float)
            full_coef = float(full["coef"])
            abs_change = (coef_s - full_coef).abs()
            relative = abs_change / abs(full_coef) if full_coef != 0 else pd.Series(np.nan, index=coef_s.index)
            sign_retained = np.sign(coef_s) == np.sign(full_coef)
            raw_sig_full = float(full["p_value"]) < 0.05
            raw_sig_loo = p_s < 0.05
            direction_freq = float(sign_retained.mean()) if len(sign_retained) else np.nan
            max_rel = float(relative.max(skipna=True)) if relative.notna().any() else np.nan
            if not coefs or not np.isfinite(full_coef):
                category = "insufficient_or_degenerate"
            elif direction_freq < 1.0:
                category = "sign_reversal_under_omission"
            elif np.isfinite(max_rel) and max_rel <= 0.20:
                category = "direction_stable_and_magnitude_stable"
            else:
                category = "direction_stable_but_magnitude_sensitive"
            summary_rows.append(
                {
                    "cohort": cohort,
                    "HMDB_identifier": hmdb,
                    "display_name": target["metabolite_name"],
                    "full_data_coefficient": full_coef,
                    "full_data_HC3_ci_low": full["ci_low"],
                    "full_data_HC3_ci_high": full["ci_high"],
                    "minimum_leave_one_out_coefficient": coef_s.min(skipna=True),
                    "maximum_leave_one_out_coefficient": coef_s.max(skipna=True),
                    "median_leave_one_out_coefficient": coef_s.median(skipna=True),
                    "percent_omissions_retaining_full_data_sign": 100 * direction_freq,
                    "largest_absolute_coefficient_change": abs_change.max(skipna=True),
                    "largest_relative_coefficient_change": max_rel,
                    "any_omission_reverses_sign": bool((~sign_retained).any()),
                    "any_omission_changes_raw_p_below_0_05_status": bool((raw_sig_loo != raw_sig_full).any()),
                    "coefficient_cv_across_omissions": float(coef_s.std(skipna=True, ddof=1) / abs(coef_s.mean(skipna=True))) if abs(coef_s.mean(skipna=True)) > 0 else np.nan,
                    "maximum_absolute_DFBETA_like": float(pd.Series(dfbetas, dtype=float).abs().max(skipna=True)),
                    "leave_one_out_fit_count": len(cc_idx),
                    "correction_level_significance_recomputed": False,
                    "BH_loo_note": "Complete-family BH-FDR was not recomputed under each omission; report is limited to coefficient, interval and raw-p-value stability.",
                    "interpretation_category": category,
                    "prespecified_magnitude_stability_rule": "100% sign retention and max absolute coefficient change <=20% of full-data absolute coefficient",
                }
            )
    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)
    atomic_csv(tables / "replicated_metabolite_leave_one_out_influence.csv", detail)
    atomic_csv(tables / "replicated_metabolite_influence_summary.csv", summary)
    return detail, summary


def reports(reports_dir: Path, primary: pd.DataFrame, summary: pd.DataFrame) -> None:
    wide = summary.pivot_table(index="display_name", columns="cohort", values="interpretation_category", aggfunc="first").reset_index()
    report = [
        "# Replicated Metabolite Robustness Report",
        "",
        "Status: `METABOLITE_ROBUSTNESS_COMPLETE`; overall focused-validation branch remains `PARTIAL_NOT_YET_AUTHORITATIVE`.",
        "",
        "This is a post-selection influence analysis of the seven IBD-vs-control metabolites previously identified as strict same-direction, BH-FDR replicated findings in Franzosa and iHMP. It is not a new discovery screen.",
        "",
        "Primary univariate models were reproduced as `metabolite ~ disease_status + age_years` with HC3 robust standard errors using the exact cohort-specific HMDB matrices and included-sample manifest used by the authoritative IBD analysis.",
        "",
        "Within-cohort standardisation in Supplementary Figure S1 is for display only and does not harmonise assay scales across Franzosa and iHMP.",
        "",
        "Complete-family BH-FDR was not recomputed after every leave-one-participant-out deletion; therefore correction-level leave-one-out significance stability is not claimed.",
        "",
        "Participant-level influence stability does not establish biological mechanism.",
        "",
        "## Direction and Magnitude Stability",
        "",
        wide.to_markdown(index=False),
        "",
        "## Primary Result Audit",
        "",
        primary.to_markdown(index=False),
    ]
    atomic_text(reports_dir / "REPLICATED_METABOLITE_PRIMARY_AUDIT.md", "# Replicated Metabolite Primary Audit\n\n" + primary.to_markdown(index=False) + "\n")
    atomic_text(reports_dir / "REPLICATED_METABOLITE_ROBUSTNESS_REPORT.md", "\n".join(report) + "\n")


def tests(tests_dir: Path, targets: pd.DataFrame, primary: pd.DataFrame, detail: pd.DataFrame, summary: pd.DataFrame, source_values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(name: str, ok: bool, observed: Any = "", expected: Any = "") -> None:
        rows.append({"test": name, "status": "pass" if ok else "fail", "observed": observed, "expected": expected})
    add("seven_identifiers_match_authoritative_provenance", set(targets["metabolite_name"]) == EXPECTED_NAMES and len(targets) == 7, sorted(targets["metabolite_name"]), sorted(EXPECTED_NAMES))
    add("primary_full_data_results_reproduced", bool(primary["reproduces_authoritative_primary_result"].all()) and len(primary) == 14, len(primary), 14)
    combos = set(zip(detail["cohort"], detail["HMDB_identifier"]))
    add("all_expected_cohort_metabolite_combinations_present", len(combos) == 14, len(combos), 14)
    counts = detail.groupby(["cohort", "HMDB_identifier"]).size().reset_index(name="n").merge(summary[["cohort", "HMDB_identifier", "leave_one_out_fit_count"]], on=["cohort", "HMDB_identifier"])
    add("leave_one_out_fit_counts_equal_complete_case_sample_size", bool((counts["n"] == counts["leave_one_out_fit_count"]).all()), counts.to_dict("records"), "equal")
    dup = detail.duplicated(["cohort", "HMDB_identifier", "anonymous_omission_index"]).any()
    add("no_anonymous_omission_index_duplicated", not bool(dup), bool(dup), False)
    freq = summary["percent_omissions_retaining_full_data_sign"] / 100.0
    add("sign_retention_frequencies_bounded", bool(freq.between(0, 1).all()), freq.tolist(), "[0,1]")
    add("source_values_match_plotted_scope", len(source_values) > 0 and set(source_values["display_name"]) == EXPECTED_NAMES, len(source_values), "non-empty seven-metabolite figure source")
    forbidden = ["sample_id", "participant_id", "original_label", "collection_date"]
    public_paths = ["primary", "detail", "summary", "source"]
    public_cols = pd.Series(
        list(primary.columns) + list(detail.columns) + list(summary.columns) + list(source_values.columns),
        name="col",
    )
    add("public_outputs_contain_no_original_participant_identifiers", not public_cols.isin(forbidden).any(), public_cols[public_cols.isin(forbidden)].tolist(), "none")
    out = pd.DataFrame(rows)
    atomic_csv(tests_dir / "metabolite_robustness_tests.csv", out)
    if out["status"].eq("fail").any():
        raise RuntimeError("metabolite robustness tests failed")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="final_analysis/focused_validation_v1")
    parser.add_argument("--thesis-source-root", default=os.environ.get("THESIS_SOURCE_ROOT", "thesis_preprocessing_workflow"))
    args = parser.parse_args()
    output = REPO / args.output_root
    thesis = Path(args.thesis_source_root)
    tables = output / "tables"
    figures = output / "figures"
    reports_dir = output / "reports"
    tests_dir = output / "tests"
    targets = claim_set(thesis)
    primary = primary_audit(thesis, tables, targets)
    source_values = distribution_figure(thesis, figures, tables, targets)
    detail, summary = loo_influence(thesis, tables, targets, primary)
    reports(reports_dir, primary, summary)
    tests(tests_dir, targets, primary, detail, summary, source_values)
    print("METABOLITE_ROBUSTNESS_COMPLETE")


if __name__ == "__main__":
    main()
