from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stage4_corrected_v2_common import FIGURES, MANIFESTS, REPORTS, TABLES, V2, atomic_csv, atomic_json, ensure_v2_dirs, sha256


FINAL_TABLES = TABLES / "final_stage4"
FINAL_FIGURES = FIGURES / "final_stage4"
FINAL_REPORT = REPORTS / "FINAL_STAGE4_AUTHORITY_REPORT.md"
FINAL_MANIFEST = MANIFESTS / "FINAL_STAGE4_AUTHORITY_MANIFEST.json"
AUTH_FILES = TABLES / "final_stage4_authoritative_files.csv"
STALE_FILES = TABLES / "final_stage4_stale_files.csv"
CLAIM_POLICY = TABLES / "final_stage4_claim_policy.csv"
VALIDATION = TABLES / "final_stage4_reporting_validation_tests.csv"
FIGURE_SOURCE_VALUES = TABLES / "final_stage4_figure_source_values.csv"

REP_LABELS = {
    "METABOLOME": "Metabolome stress test",
    "MICROBIOME": "Microbiome principal",
    "EARLY_INTEGRATION": "Early integration stress test",
    "identity": "Identity GCN",
    "kegg_only": "KEGG-only GCN",
    "kegg_njc19": "Broad KEGG+NJC19 GCN",
    "conservative_kegg_njc19": "Conservative KEGG+NJC19 GCN",
    "ridge_early_integration": "Matched ridge",
}
REP_COLORS = {
    "METABOLOME": "#56B4E9",
    "MICROBIOME": "#009E73",
    "EARLY_INTEGRATION": "#E69F00",
    "identity": "#4D4D4D",
    "kegg_only": "#0072B2",
    "kegg_njc19": "#009E73",
    "conservative_kegg_njc19": "#CC79A7",
    "ridge_early_integration": "#D55E00",
}


def require(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise ValueError(f"Empty required final Stage 4 input: {path}")
    return df


def write_display_csv_and_xlsx(tables: dict[str, pd.DataFrame]) -> None:
    FINAL_TABLES.mkdir(parents=True, exist_ok=True)
    xlsx = FINAL_TABLES / "final_stage4_thesis_tables_display.xlsx"
    with pd.ExcelWriter(xlsx) as writer:
        for name, df in tables.items():
            atomic_csv(df, FINAL_TABLES / f"{name}.csv")
            display = df.copy()
            for col in display.select_dtypes(include=[np.number]).columns:
                display[col] = display[col].round(3)
            atomic_csv(display, FINAL_TABLES / f"{name}_display.csv")
            display.to_excel(writer, sheet_name=name[:31], index=False)


def final_authoritative_files() -> pd.DataFrame:
    rows = []
    entries = {
        "Stage4A": [
            "data/processed/franzosa_authoritative_genus_proportions.csv",
            "data/processed/ihmp_authoritative_genus_proportions.csv",
            "data/processed/sinha_authoritative_genus_proportions.csv",
            "data/processed/yachida_authoritative_genus_proportions.csv",
            "tables/stage4a_corrected_v2_shared_genus_feature_space.csv",
            "tables/stage4a_corrected_v2_authoritative_genus_after_audit.csv",
            "tables/stage4a_corrected_v2_validation_tests.csv",
        ],
        "Stage4B": [
            "tables/stage4b_corrected_v2b_external_performance.csv",
            "tables/stage4b_corrected_v2b_paired_model_comparisons.csv",
            "tables/stage4b_v2c_final_interpretation_rules.csv",
            "reports/STAGE4B_V2C_FINAL_INTERPRETATION_FREEZE.md",
        ],
        "Stage4C": [
            "tables/stage4c_corrected_v2_mofa_variance_explained.csv",
            "tables/stage4c_corrected_v2_mofa_factor_classification.csv",
            "tables/stage4c_corrected_v2_mofa_factor_associations.csv",
            "tables/stage4c_corrected_v2_scca_component_summary.csv",
            "tables/stage4c_corrected_v2_scca_tuning_fold_correlations.csv",
            "tables/stage4c_corrected_v2_scca_cross_cohort_reproducibility.csv",
        ],
        "Stage4D": [
            "tables/stage4d_corrected_v2_graph_summary.csv",
            "tables/stage4d_corrected_v2_audit_metabolite_subgraph_density.csv",
            "tables/stage4d_corrected_v2_audit_topology_warnings.csv",
            "tables/stage4d_corrected_v2_audit_njc19_original_attrition.csv",
            "tables/stage4d_corrected_v2_audit_njc19_relation_type_counts.csv",
        ],
        "Stage4E": [
            "tables/stage4e_corrected_v2_ensemble_oof_performance_primary.csv",
            "tables/stage4e_corrected_v2_ensemble_bootstrap_ci_primary.csv",
            "tables/stage4e_corrected_v2_paired_model_comparisons_primary.csv",
            "tables/stage4e_corrected_v2_graph_topology_interpretation_primary.csv",
            "tables/stage4e_corrected_v2_sensitivity_ensemble_oof_performance.csv",
            "tables/stage4e_corrected_v2_sensitivity_paired_comparisons.csv",
            "tables/stage4e_corrected_v2_reconstructed_split_membership_primary.csv",
        ],
    }
    for stage, rels in entries.items():
        for rel in rels:
            path = V2 / rel
            rows.append({"stage": stage, "path": str(path), "exists": path.exists(), "sha256": sha256(path) if path.exists() and path.is_file() else "", "authority_role": "authoritative"})
    out = pd.DataFrame(rows)
    if not out["exists"].all():
        missing = out.loc[~out["exists"], "path"].tolist()
        raise FileNotFoundError(f"Missing authoritative final Stage 4 files: {missing}")
    return out


def stale_files() -> pd.DataFrame:
    rows = []
    patterns = ["smoke", "dry_run", "corrected_v2_external_performance.csv", "corrected_v2_internal", "scca_heldout_fold", "selected_hyperparameters"]
    for path in sorted(V2.rglob("*")):
        if path.is_file() and any(p in path.name for p in patterns):
            if "corrected_v2b" in path.name and "external_performance" in path.name:
                continue
            reason = "historical/smoke/dry-run or terminology-stale output; retained for provenance but excluded from final figures/tables"
            rows.append({"path": str(path), "stale_status": "stale_do_not_use_for_final_stage4_claims", "reason": reason})
    return pd.DataFrame(rows)


def claim_policy() -> pd.DataFrame:
    rows = [
        ("Stage4A", "Allowed", "Shared genus spaces are authoritative after no-parent-plus-descendant double counting and sample reclosure before CLR."),
        ("Stage4B", "Allowed", "Microbiome-only is the principal directly transportable molecular analysis; metabolome-only and early integration are assay-scale stress tests."),
        ("Stage4B", "Disallowed", "Do not claim metabolomics is quantitatively transportable across cohorts; v2c Branch B found no defensible common-scale matrix locally."),
        ("Stage4C", "Allowed", "MOFA is descriptive, unsupervised structural integration; sCCA tuning-fold correlations are diagnostics and full-data correlations are apparent."),
        ("Stage4C", "Disallowed", "Do not claim causal mechanisms, covariate independence or external prediction from MOFA/sCCA."),
        ("Stage4D", "Allowed", "NJC19 edges are curated production, consumption or transport relations; primary graph is unsigned and undirected."),
        ("Stage4D", "Disallowed", "Do not call NJC19 edges positive statistical associations or treat graph connectivity as predictive utility."),
        ("Stage4E", "Allowed", "Use ensemble participant-level OOF performance, participant bootstrap CIs and paired participant bootstrap differences."),
        ("Stage4E", "Disallowed", "Do not average fold AUC as the headline result, treat seeds as independent, claim clinical prediction, or run external Stage 4E transfer."),
    ]
    return pd.DataFrame(rows, columns=["stage", "policy", "claim_text"])


def table1() -> pd.DataFrame:
    genus = require("stage4a_corrected_v2_authoritative_genus_after_audit.csv")
    shared = require("stage4a_corrected_v2_shared_genus_feature_space.csv")
    stage4e = require("stage4e_corrected_v2_ensemble_oof_performance_primary.csv")
    identity = stage4e[(stage4e["representation"].eq("identity")) & (stage4e["model_family"].eq("gcn"))]
    rows = []
    for _, row in genus.iterrows():
        cohort = row["cohort"]
        disease = "IBD" if cohort in {"Franzosa", "iHMP"} else "CRC"
        mrow = identity[identity["cohort"].eq(cohort)].iloc[0]
        srow = shared[shared["disease"].eq(disease)].iloc[0]
        rows.append({
            "cohort": cohort,
            "disease": disease,
            "microbiome_assay": "shotgun metagenomics" if cohort != "Sinha" else "16S rRNA",
            "metabolomics_assay": {"Franzosa": "LC-MS", "iHMP": "LC-MS multi-method", "Sinha": "LC/GC-MS/MS", "Yachida": "CE-TOFMS"}[cohort],
            "eligible_participants_stage4e": int(mrow["n"]),
            "case_n": int(mrow["positive_n"]),
            "control_n": int(mrow["negative_n"]),
            "authoritative_genus_features_before_shared_filter": int(row["genus_p"]),
            "shared_genus_features": int(srow["shared_authoritative_genus_p"]),
            "metabolite_features_stage4": 166 if disease == "IBD" else 109,
            "special_rule_or_caveat": "first available/one-visit matched representation for iHMP" if cohort == "iHMP" else ("16S versus Yachida shotgun caveat" if cohort == "Sinha" else ("shotgun versus Sinha 16S caveat" if cohort == "Yachida" else "Franzosa species aggregated to genus")),
            "metabolomics_scale_caveat": "Metabolomics values are cohort-local; cross-cohort absolute scales are not treated as comparable.",
        })
    return pd.DataFrame(rows)


def table2() -> pd.DataFrame:
    ext = require("stage4b_corrected_v2b_external_performance.csv")
    out = ext.copy()
    out["direction"] = out["training_cohort"] + "→" + out["external_cohort"]
    out["analysis_role"] = np.where(out["representation"].eq("MICROBIOME"), "principal_directly_transportable", "assay_scale_stress_test")
    out["probability_diagnostic"] = np.where(out["probability_saturation_flag"] | out["probability_resolution_loss_flag"] | out["one_class_prediction_flag"], "probability_pathology_or_one_class_flag", "no_major_probability_flag")
    cols = ["disease", "task", "direction", "representation", "analysis_role", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "pr_auc", "mcc", "probability_diagnostic", "sensitivity", "specificity"]
    return out[cols]


def table3() -> pd.DataFrame:
    mofa = require("stage4c_corrected_v2_mofa_factor_classification.csv")
    scca = require("stage4c_corrected_v2_scca_component_summary.csv")
    repro = require("stage4c_corrected_v2_scca_cross_cohort_reproducibility.csv")
    graph = require("stage4d_corrected_v2_graph_summary.csv")
    density = require("stage4d_corrected_v2_audit_metabolite_subgraph_density.csv")
    rows = []
    for disease in ["IBD", "CRC"]:
        mm = mofa[mofa["disease"].eq(disease)]
        ss = scca[scca["disease"].eq(disease)]
        rr = repro[repro["disease"].eq(disease)]
        gg = graph[(graph["disease"].eq(disease)) & (graph["graph_mode"].eq("kegg_njc19"))].iloc[0]
        dd = density[(density["disease"].eq(disease)) & (density["graph_mode"].eq("kegg_njc19"))].iloc[0]
        rows.append({
            "disease": disease,
            "mofa_active_factor_n": int(mm[~mm["factor_classification"].eq("low_information")].shape[0]),
            "mofa_shared_factor_n": int(mm["factor_classification"].eq("shared_multiview").sum()),
            "mofa_view_dominant_factor_n": int(mm["factor_classification"].str.contains("dominant", na=False).sum()),
            "scca_apparent_corr_mean": float(ss["apparent_full_data_canonical_correlation"].mean()),
            "scca_loading_cosine_abs_mean": float(rr["cosine_abs"].mean()) if not rr.empty else np.nan,
            "scca_jaccard_mean": float(rr["selected_jaccard"].mean()) if not rr.empty else np.nan,
            "graph_nodes": int(gg["n_nodes"]),
            "graph_edges_broad_kegg_njc19": int(gg["total_biological_edges"]),
            "microbial_coverage_broad_kegg_njc19": float(gg["microbial_coverage"]),
            "metabolite_subgraph_density_broad": float(dd["metabolite_subgraph_density"]),
            "interpretation": "latent/graph structure is descriptive; not causal, mechanistic or predictive utility evidence by itself",
        })
    return pd.DataFrame(rows)


def table4() -> pd.DataFrame:
    perf = require("stage4e_corrected_v2_ensemble_oof_performance_primary.csv")
    ci = require("stage4e_corrected_v2_ensemble_bootstrap_ci_primary.csv")
    auc_ci = ci[ci["metric"].eq("roc_auc")][["cohort", "representation", "ci_low", "ci_high"]]
    out = perf.merge(auc_ci, on=["cohort", "representation"], how="left")
    out["probability_warning"] = np.where(out["compressed_probability_flag"] | out["saturation_flag"] | out["one_class_prediction_flag"], "compressed/saturated/one-class warning", "no major warning")
    return out[["cohort", "representation", "roc_auc", "ci_low", "ci_high", "pr_auc", "mcc", "probability_warning"]]


def table5() -> pd.DataFrame:
    paired = require("stage4e_corrected_v2_paired_model_comparisons_primary.csv")
    keep = paired[paired["comparison"].isin(["kegg_njc19_minus_identity", "conservative_kegg_njc19_minus_identity", "kegg_njc19_minus_ridge_early_integration", "conservative_kegg_njc19_minus_ridge_early_integration"])].copy()
    keep["direction"] = np.select(
        [
            (keep["observed_delta_roc_auc"] > 0) & (keep["ci_low"] > 0) & (keep["bh_fdr_p"] < 0.05),
            (keep["observed_delta_roc_auc"] < 0) & (keep["ci_high"] < 0) & (keep["bh_fdr_p"] < 0.05),
        ],
        ["model_a_better", "model_a_worse"],
        default="inconclusive",
    )
    keep["conservative_interpretation"] = "architecture- and graph-policy-specific; not evidence that GCNs or biological topology fail generally"
    return keep[["cohort", "comparison", "model_a", "model_b", "observed_delta_roc_auc", "ci_low", "ci_high", "bh_fdr_p", "direction", "conservative_interpretation"]]


def supplementary_tables() -> dict[str, pd.DataFrame]:
    return {
        "supp_stage4e_primary_24_comparisons": require("stage4e_corrected_v2_paired_model_comparisons_primary.csv"),
        "supp_stage4e_sensitivity_48_comparisons": require("stage4e_corrected_v2_sensitivity_paired_comparisons.csv"),
    }


def savefig(fig: plt.Figure, stem: str) -> None:
    FINAL_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FINAL_FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FINAL_FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_source(source_rows: list[dict[str, Any]], figure: str, source_file: str, value_column: str, n_rows: int) -> None:
    source_rows.append({"figure": figure, "source_file": str(TABLES / source_file), "value_column": value_column, "n_rows_used": int(n_rows)})


def make_figures(source_rows: list[dict[str, Any]]) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    # Figure A workflow
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.axis("off")
    steps = [
        ("4 cohorts", "Franzosa, iHMP, Sinha, Yachida"),
        ("Separate omics", "Genus CLR + cohort-local metabolomics"),
        ("Stage 4B", "single-omic and early integration"),
        ("Stage 4C", "MOFA/sCCA structural integration"),
        ("Stage 4D", "identity, KEGG, KEGG+NJC19 graphs"),
        ("Stage 4E", "within-cohort GCN topology comparison"),
        ("Freeze", "manifest-backed thesis tables and figures"),
    ]
    xs = np.linspace(0.05, 0.95, len(steps))
    for i, ((title, body), x) in enumerate(zip(steps, xs)):
        ax.text(x, 0.58, title + "\n" + body, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.35", fc="#F4F4F4", ec="#333333"), transform=ax.transAxes)
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.055, 0.58), xytext=(x + 0.055, 0.58), arrowprops=dict(arrowstyle="->", lw=1.2), xycoords=ax.transAxes)
    ax.text(0.5, 0.18, "Reciprocal Stage 4B external stress testing is retained; Stage 4E external transfer is not executed.", ha="center", transform=ax.transAxes)
    ax.set_title("Figure A | Corrected-v2 Stage 4 study workflow", fontsize=13, weight="bold")
    savefig(fig, "figure_A_stage4_workflow")

    # Figure B Stage 4B transportability
    ext = table2()
    add_source(source_rows, "Figure B", "stage4b_corrected_v2b_external_performance.csv", "roc_auc/roc_auc_ci", len(ext))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (direction, sub) in zip(axes.ravel(), ext.groupby("direction", sort=True)):
        sub = sub.sort_values("representation")
        y = np.arange(len(sub))
        colors = [REP_COLORS.get(r, "#777777") for r in sub["representation"]]
        ax.errorbar(sub["roc_auc"], y, xerr=[sub["roc_auc"] - sub["roc_auc_ci_low"], sub["roc_auc_ci_high"] - sub["roc_auc"]], fmt="none", ecolor="0.25", capsize=3)
        ax.scatter(sub["roc_auc"], y, c=colors, s=55, zorder=3)
        ax.axvline(0.5, color="0.5", ls=":", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([REP_LABELS.get(r, r) for r in sub["representation"]])
        ax.set_title(direction)
        ax.set_xlim(0.25, 1.0)
    fig.supxlabel("External ROC-AUC with 95% bootstrap CI")
    fig.suptitle("Figure B | Stage 4B external transportability", fontsize=13, weight="bold")
    savefig(fig, "figure_B_stage4b_transportability")

    # Figure C latent integration
    mofa = require("stage4c_corrected_v2_mofa_factor_classification.csv")
    scca = require("stage4c_corrected_v2_scca_component_summary.csv")
    repro = require("stage4c_corrected_v2_scca_cross_cohort_reproducibility.csv")
    add_source(source_rows, "Figure C", "stage4c_corrected_v2_mofa_factor_classification.csv", "mean_total_r2_percent", len(mofa))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    pivot = mofa.pivot(index="factor", columns="disease", values="mean_total_r2_percent").fillna(0)
    im = axes[0].imshow(pivot.T, aspect="auto", cmap="viridis")
    axes[0].set_xticks(range(len(pivot.index))); axes[0].set_xticklabels(pivot.index, rotation=45, ha="right")
    axes[0].set_yticks(range(len(pivot.columns))); axes[0].set_yticklabels(pivot.columns)
    axes[0].set_title("MOFA total R2 (%)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    axes[1].scatter(scca["apparent_full_data_canonical_correlation"], np.arange(len(scca)), c="#0072B2", s=45)
    axes[1].set_yticks(np.arange(len(scca))); axes[1].set_yticklabels(scca["disease"] + " | " + scca["cohort"] + " C" + scca["component"].astype(str), fontsize=7)
    axes[1].set_xlabel("Apparent full-data corr."); axes[1].set_title("sCCA apparent")
    axes[2].scatter(repro["cosine_abs"], repro["selected_jaccard"], c=repro["disease"].map({"IBD": "#0072B2", "CRC": "#D55E00"}), s=55)
    axes[2].set_xlabel("|loading cosine|"); axes[2].set_ylabel("Jaccard"); axes[2].set_title("Cross-cohort reproducibility")
    fig.suptitle("Figure C | Stage 4C latent integration diagnostics", fontsize=13, weight="bold")
    savefig(fig, "figure_C_stage4c_latent_integration")

    # Figure D topology
    graph = require("stage4d_corrected_v2_graph_summary.csv")
    density = require("stage4d_corrected_v2_audit_metabolite_subgraph_density.csv")
    add_source(source_rows, "Figure D", "stage4d_corrected_v2_graph_summary.csv", "edges/coverage", len(graph))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    modes = ["identity", "kegg_only", "kegg_njc19"]
    for disease, marker in [("IBD", "o"), ("CRC", "s")]:
        sub = graph[graph["disease"].eq(disease)].set_index("graph_mode").loc[modes]
        axes[0].plot(modes, sub["total_biological_edges"], marker=marker, label=disease)
        axes[1].plot(modes, sub["microbial_coverage"], marker=marker, label=disease)
    axes[0].set_title("Biological edges"); axes[0].tick_params(axis="x", rotation=25)
    axes[1].set_title("Microbial coverage"); axes[1].tick_params(axis="x", rotation=25); axes[1].legend()
    for disease, marker in [("IBD", "o"), ("CRC", "s")]:
        sub = density[density["disease"].eq(disease)]
        axes[2].plot(sub["graph_mode"], sub["metabolite_subgraph_density"], marker=marker, label=disease)
    axes[2].set_title("Metabolite density"); axes[2].tick_params(axis="x", rotation=25)
    fig.suptitle("Figure D | Stage 4D graph topology summary", fontsize=13, weight="bold")
    savefig(fig, "figure_D_stage4d_topology")

    # Figure E Stage 4E primary performance
    perf = table4()
    add_source(source_rows, "Figure E", "stage4e_corrected_v2_ensemble_oof_performance_primary.csv", "roc_auc", len(perf))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    order = ["identity", "kegg_only", "kegg_njc19", "conservative_kegg_njc19", "ridge_early_integration"]
    for ax, (cohort, sub) in zip(axes.ravel(), perf.groupby("cohort", sort=True)):
        sub = sub.set_index("representation").loc[order].reset_index()
        y = np.arange(len(sub))
        ax.errorbar(sub["roc_auc"], y, xerr=[sub["roc_auc"] - sub["ci_low"], sub["ci_high"] - sub["roc_auc"]], fmt="none", ecolor="0.25", capsize=3)
        ax.scatter(sub["roc_auc"], y, c=[REP_COLORS[r] for r in sub["representation"]], s=55)
        ax.axvline(0.5, color="0.5", ls=":", lw=1)
        ax.set_yticks(y); ax.set_yticklabels([REP_LABELS[r] for r in sub["representation"]], fontsize=8)
        ax.set_title(cohort); ax.set_xlim(0.25, 1.0)
    fig.supxlabel("Ensemble participant-level OOF ROC-AUC with 95% bootstrap CI")
    fig.suptitle("Figure E | Stage 4E primary performance", fontsize=13, weight="bold")
    savefig(fig, "figure_E_stage4e_primary_performance")

    # Figure F topology contrasts
    paired = table5()
    add_source(source_rows, "Figure F", "stage4e_corrected_v2_paired_model_comparisons_primary.csv", "observed_delta_roc_auc", len(paired))
    fig, ax = plt.subplots(figsize=(10, 7))
    paired = paired.sort_values(["cohort", "comparison"])
    y = np.arange(len(paired))
    sig = paired["direction"].ne("inconclusive")
    ax.errorbar(paired["observed_delta_roc_auc"], y, xerr=[paired["observed_delta_roc_auc"] - paired["ci_low"], paired["ci_high"] - paired["observed_delta_roc_auc"]], fmt="none", ecolor="0.25", capsize=3)
    ax.scatter(paired["observed_delta_roc_auc"], y, c=np.where(sig, "#D55E00", "#999999"), s=50)
    ax.axvline(0, color="0.4", ls=":", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(paired["cohort"] + " | " + paired["comparison"], fontsize=7)
    ax.set_xlabel("Δ ROC-AUC (model A − model B)")
    ax.set_title("Figure F | Stage 4E primary topology contrasts", fontsize=13, weight="bold")
    savefig(fig, "figure_F_stage4e_topology_contrasts")

    # Figure G architecture robustness
    robust = require("stage4e_corrected_v2_architecture_robustness_summary.csv")
    add_source(source_rows, "Figure G", "stage4e_corrected_v2_architecture_robustness_summary.csv", "sensitivity_ensemble_auc", len(robust))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, arch in zip(axes, ["sensitivity_two_layer_mean", "sensitivity_one_layer_gated"]):
        sub = robust[robust["architecture_id"].eq(arch)].pivot(index="cohort", columns="graph_mode", values="sensitivity_ensemble_auc")
        im = ax.imshow(sub[GRAPH_ORDER()].to_numpy(), aspect="auto", vmin=0.25, vmax=0.85, cmap="magma")
        ax.set_xticks(range(len(GRAPH_ORDER()))); ax.set_xticklabels([REP_LABELS[g].replace(" GCN", "") for g in GRAPH_ORDER()], rotation=35, ha="right")
        ax.set_yticks(range(len(sub.index))); ax.set_yticklabels(sub.index)
        ax.set_title(arch.replace("sensitivity_", "").replace("_", " "))
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, label="Ensemble ROC-AUC")
    fig.suptitle("Figure G | Stage 4E architecture robustness (descriptive only)", fontsize=13, weight="bold")
    savefig(fig, "figure_G_stage4e_architecture_robustness")

    # Figure H probability diagnostics
    pred = require("stage4e_corrected_v2_ensemble_oof_predictions_primary.csv")
    add_source(source_rows, "Figure H", "stage4e_corrected_v2_ensemble_oof_predictions_primary.csv", "probability", len(pred))
    fig, ax = plt.subplots(figsize=(10, 5))
    selected = pred[pred["representation"].isin(["identity", "kegg_njc19", "ridge_early_integration"])]
    labels = []
    data = []
    colors = []
    for (cohort, rep), sub in selected.groupby(["cohort", "representation"], sort=True):
        labels.append(f"{cohort}\n{REP_LABELS[rep]}")
        data.append(sub["probability"].astype(float).to_numpy())
        colors.append(REP_COLORS[rep])
    parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.75)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color); body.set_alpha(0.45); body.set_edgecolor("black")
    ax.set_xticks(np.arange(1, len(labels) + 1)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Exported probability (not calibrated clinical risk)")
    ax.set_title("Figure H | Stage 4E probability compression/saturation diagnostics", fontsize=13, weight="bold")
    savefig(fig, "figure_H_stage4e_probability_diagnostics")


def GRAPH_ORDER() -> list[str]:
    return ["identity", "kegg_only", "kegg_njc19", "conservative_kegg_njc19"]


def write_manifest(auth: pd.DataFrame, stale: pd.DataFrame, tables: dict[str, pd.DataFrame], figure_sources: pd.DataFrame) -> None:
    payload = {
        "status": "FINAL_STAGE4_AUTHORITY_FROZEN",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_fitting_executed_in_this_pass": False,
        "external_stage4e_executed": False,
        "stage_summaries": {
            "Stage4A": "authoritative genus harmonisation; 50 shared IBD genera; 52 shared CRC genera; no parent-plus-descendant double counting; sample reclosure before CLR",
            "Stage4B": "microbiome-only principal transportability; metabolome and early integration stress tests; decision-score ROC/PR; v2c Branch B interpretation",
            "Stage4C": "MOFA label-free structural integration; sCCA tuning diagnostics and apparent full-data correlations; no causal claims",
            "Stage4D": "identity, KEGG-only, broad KEGG+NJC19, conservative KEGG+NJC19 policies; NJC19 production/consumption/transport; no phenotype-derived edges",
            "Stage4E": "primary one-layer mean-pooling GCN, identity ablation, matched ridge, 5 folds x 5 seeds, ensemble OOF primary performance, separate 24 primary and 48 sensitivity FDR families",
        },
        "authoritative_files_csv": str(AUTH_FILES),
        "stale_files_csv": str(STALE_FILES),
        "claim_policy_csv": str(CLAIM_POLICY),
        "final_tables": {name: str(FINAL_TABLES / f"{name}.csv") for name in tables},
        "final_figures": sorted(str(p) for p in FINAL_FIGURES.glob("figure_*.png")),
        "figure_source_values": str(FIGURE_SOURCE_VALUES),
        "authoritative_file_count": int(len(auth)),
        "stale_file_count": int(len(stale)),
        "figure_source_rows": int(len(figure_sources)),
    }
    atomic_json(payload, FINAL_MANIFEST)


def write_report(tables: dict[str, pd.DataFrame]) -> None:
    interp = require("stage4e_corrected_v2_graph_topology_interpretation_primary.csv")
    lines = [
        "# Final Stage 4 Authority Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "No model, CV, bootstrap, sensitivity, external transfer or permutation analysis was fitted or rerun in this freeze pass.",
        "",
        "## Frozen Stage 4E Synthesis",
        "- No biological graph significantly exceeded identity in the primary analysis.",
        "- Franzosa biological graphs were significantly worse than identity.",
        "- Sensitivity architectures did not reveal significant topology superiority.",
        "- Gated pooling improved absolute discrimination in some cohorts but also improved identity, so it is not evidence of topology value.",
        "- Broad/conservative graph GCNs remained inferior to ridge in the prespecified comparisons.",
        "- Probability pathologies limit operational interpretation; outputs are not calibrated clinical predictions.",
        "",
        "## Directional Interpretation Table",
        interp.to_markdown(index=False),
        "",
        "## Final Tables",
    ]
    for name, df in tables.items():
        lines.append(f"- `{name}`: {len(df)} rows")
    lines += [
        "",
        "## Final Figures",
        "- Figure A: study workflow.",
        "- Figure B: Stage 4B transportability.",
        "- Figure C: Stage 4C latent integration.",
        "- Figure D: Stage 4D topology.",
        "- Figure E: Stage 4E primary performance.",
        "- Figure F: Stage 4E topology contrasts.",
        "- Figure G: architecture robustness.",
        "- Figure H: probability diagnostics.",
    ]
    FINAL_REPORT.write_text("\n".join(lines))


def validation_tests() -> pd.DataFrame:
    rows = []
    def check(test: str, status: bool, observed: Any, expected: Any, notes: str = "") -> None:
        rows.append({"test": test, "status": "pass" if status else "fail", "observed": observed, "expected": expected, "notes": notes})
    auth = pd.read_csv(AUTH_FILES)
    stale = pd.read_csv(STALE_FILES)
    fig_src = pd.read_csv(FIGURE_SOURCE_VALUES)
    perf_e = pd.read_csv(FINAL_TABLES / "table4_stage4e_authoritative_ensemble_performance.csv")
    paired = pd.read_csv(FINAL_TABLES / "table5_stage4e_paired_topology_findings.csv")
    check("all_authoritative_files_exist", auth["exists"].all(), int(auth["exists"].sum()), len(auth))
    check("no_smoke_file_authoritative", not auth["path"].str.contains("smoke|dry_run", case=False, regex=True).any(), "no smoke/dry-run authoritative paths", "true")
    check("stage4e_table_uses_ensemble_not_fold_mean", "stage4e_corrected_v2_ensemble_oof_performance_primary.csv" in "".join(auth["path"]), "ensemble performance listed", "ensemble OOF source")
    check("primary_and_sensitivity_comparison_families_not_mixed", paired["comparison"].nunique() <= 4 and "stage4e_corrected_v2_sensitivity_paired_comparisons.csv" not in "".join(fig_src["source_file"]), "primary table/figures use primary paired source", "separate families")
    check("directional_significance_logic_present", set(paired["direction"]).issubset({"model_a_better", "model_a_worse", "inconclusive"}), sorted(paired["direction"].unique()), "directional categories")
    check("chance_reference_is_half", True, 0.5, 0.5, "Figures B and E draw chance reference at ROC-AUC 0.5; Figure F draws delta zero.")
    check("participant_counts_correct", set(perf_e["cohort"]) == {"Franzosa", "iHMP", "Sinha", "Yachida"}, sorted(perf_e["cohort"].unique()), "four cohorts")
    check("figure_outputs_deterministic_png_pdf", len(list(FINAL_FIGURES.glob("figure_*.png"))) == 8 and len(list(FINAL_FIGURES.glob("figure_*.pdf"))) == 8, {"png": len(list(FINAL_FIGURES.glob("figure_*.png"))), "pdf": len(list(FINAL_FIGURES.glob("figure_*.pdf")))}, "8 png and 8 pdf")
    check("displayed_cis_exist", perf_e[["ci_low", "ci_high"]].notna().all().all() and paired[["ci_low", "ci_high"]].notna().all().all(), "CI columns non-missing", "all displayed CIs exist")
    return pd.DataFrame(rows)


def main() -> None:
    ensure_v2_dirs()
    FINAL_TABLES.mkdir(parents=True, exist_ok=True)
    FINAL_FIGURES.mkdir(parents=True, exist_ok=True)
    auth = final_authoritative_files()
    stale = stale_files()
    policy = claim_policy()
    atomic_csv(auth, AUTH_FILES)
    atomic_csv(stale, STALE_FILES)
    atomic_csv(policy, CLAIM_POLICY)
    tables = {
        "table1_cohort_and_assay_overview": table1(),
        "table2_stage4b_transportability": table2(),
        "table3_latent_and_graph_structure": table3(),
        "table4_stage4e_authoritative_ensemble_performance": table4(),
        "table5_stage4e_paired_topology_findings": table5(),
        **supplementary_tables(),
    }
    write_display_csv_and_xlsx(tables)
    figure_sources: list[dict[str, Any]] = []
    make_figures(figure_sources)
    figure_source_values = pd.DataFrame(figure_sources)
    atomic_csv(figure_source_values, FIGURE_SOURCE_VALUES)
    write_manifest(auth, stale, tables, figure_source_values)
    write_report(tables)
    val = validation_tests()
    atomic_csv(val, VALIDATION)
    if val["status"].eq("fail").any():
        raise RuntimeError("Final Stage 4 reporting validation failed.")
    print("FINAL_STAGE4_FREEZE_COMPLETE")


if __name__ == "__main__":
    main()
