from __future__ import annotations

from pathlib import Path
import textwrap

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


PROJECT = Path("/Users/bhaktidoshi/Downloads/Thesis Data")
ROOT = PROJECT / "thesis_preprocessing_workflow"
STAGE4 = ROOT / "stage4_multiomics_integration"
TABLES = STAGE4 / "tables"
OUT = ROOT / "final_thesis_figures"

PALETTE = {
    "IBD": "#0072B2",
    "CRC": "#D55E00",
    "Microbiome": "#009E73",
    "Metabolome": "#56B4E9",
    "Early integration": "#E69F00",
    "Augmented GCN": "#CC79A7",
    "Identity ablation": "#7F7F7F",
    "MOFA": "#6A3D9A",
    "sCCA": "#A6761D",
}
REP_LABELS = {
    "MICROBIOME": "Microbiome",
    "METABOLOME": "Metabolome",
    "EARLY_INTEGRATION": "Early integration",
}
DIRECTION_LABELS = {
    "Franzosa_to_iHMP": "Franzosa \u2192 iHMP",
    "iHMP_to_Franzosa": "iHMP \u2192 Franzosa",
    "Sinha_to_Yachida": "Sinha \u2192 Yachida",
    "Yachida_to_Sinha": "Yachida \u2192 Sinha",
}


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def panel_label(ax, label: str) -> None:
    ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")


def direction_from_row(row: pd.Series) -> str:
    key = f"{row['training_cohort']}_to_{row['external_cohort']}"
    return DIRECTION_LABELS.get(key, key.replace("_to_", " \u2192 "))


def figure1() -> None:
    fs = pd.read_csv(TABLES / "stage4a_final_multiomics_feature_space.csv")
    display_metabolite_counts = {
        # Stage 4A retained 110 CRC metabolites, but one was excluded from the
        # technically valid downstream Stage 4C/D/E feature order. Use the frozen
        # downstream count for the thesis overview figure.
        ("CRC", "Sinha"): 109,
        ("CRC", "Yachida"): 109,
    }
    fig = plt.figure(figsize=(11, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    ax1.axis("off")
    y = 0.82
    for disease, colour in [("IBD", PALETTE["IBD"]), ("CRC", PALETTE["CRC"])]:
        part = fs.loc[fs["disease"].eq(disease)]
        ax1.text(0.02, y + 0.08, disease, color=colour, fontweight="bold", fontsize=12, transform=ax1.transAxes)
        for i, (_, row) in enumerate(part.iterrows()):
            x = 0.05 + i * 0.46
            metabolite_count = display_metabolite_counts.get(
                (row["disease"], row["cohort"]),
                int(row["metabolite_p"]),
            )
            box = FancyBboxPatch((x, y - 0.08), 0.38, 0.22, boxstyle="round,pad=0.02,rounding_size=0.02", fc="white", ec=colour, lw=1.6, transform=ax1.transAxes)
            ax1.add_patch(box)
            ax1.text(x + 0.02, y + 0.06, row["cohort"], fontweight="bold", transform=ax1.transAxes)
            ax1.text(x + 0.02, y - 0.01, f"n={int(row['matched_n'])}\n{int(row['shared_microbial_p'])} genera\n{metabolite_count} metabolites", transform=ax1.transAxes, va="top")
        y -= 0.42
    ax1.set_title("Four stool multi-omics cohorts")
    panel_label(ax1, "A")

    ax2.axis("off")
    steps = ["Matched stool\nmicrobiome + metabolome", "Single-omic ridge\nand early integration", "MOFA + sCCA\nlatent integration", "KEGG + NJC19\nprior graph", "GCN external\ntransportability"]
    xs = np.linspace(0.05, 0.86, len(steps))
    for i, (x, step) in enumerate(zip(xs, steps)):
        box = FancyBboxPatch((x, 0.45), 0.14, 0.26, boxstyle="round,pad=0.02,rounding_size=0.02", fc="#F7F7F7", ec="#555555", lw=1, transform=ax2.transAxes)
        ax2.add_patch(box)
        ax2.text(x + 0.07, 0.58, step, ha="center", va="center", transform=ax2.transAxes)
        if i < len(steps) - 1:
            ax2.add_patch(FancyArrowPatch((x + 0.145, 0.58), (xs[i + 1] - 0.01, 0.58), arrowstyle="-|>", mutation_scale=10, color="#555555", transform=ax2.transAxes))
    ax2.set_title("Analytical progression, not a causal trajectory")
    panel_label(ax2, "B")

    ax3.axis("off")
    directions = [("IBD", "Franzosa \u2192 iHMP"), ("IBD", "iHMP \u2192 Franzosa"), ("CRC", "Sinha \u2192 Yachida"), ("CRC", "Yachida \u2192 Sinha")]
    for i, (disease, direction) in enumerate(directions):
        x = 0.08 + i * 0.23
        colour = PALETTE[disease]
        ax3.text(x, 0.72, disease, color=colour, fontweight="bold", ha="center", transform=ax3.transAxes)
        ax3.text(x, 0.54, direction, ha="center", transform=ax3.transAxes)
        ax3.text(x, 0.36, "Train cohort\n\u2192 untouched external cohort", ha="center", transform=ax3.transAxes, fontsize=9)
    ax3.set_title("Bidirectional external validation design")
    panel_label(ax3, "C")
    fig.suptitle("Figure 1. Study design and cross-cohort validation framework", fontweight="bold")
    save_figure(fig, "figure1_study_design")


def figure2() -> None:
    ext = pd.read_csv(TABLES / "stage4b_external_performance.csv")
    internal = pd.read_csv(TABLES / "stage4b_internal_performance.csv")
    ext = ext[ext["task"].isin(["IBD_vs_Control", "CRC_vs_Control"])].copy()
    ext["Representation"] = ext["representation"].map(REP_LABELS)
    ext["Direction"] = ext.apply(direction_from_row, axis=1)
    internal = internal[internal["task"].isin(["IBD_vs_Control", "CRC_vs_Control"])].copy()
    internal_mean = internal.groupby(["disease", "training_cohort", "external_cohort", "representation"], as_index=False)["roc_auc"].mean()
    comp = ext.merge(internal_mean, on=["disease", "training_cohort", "external_cohort", "representation"], suffixes=("_external", "_internal"))
    comp["gap"] = comp["roc_auc_internal"] - comp["roc_auc_external"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for ax, disease in zip(axes[0], ["IBD", "CRC"]):
        part = ext.loc[ext["disease"].eq(disease)]
        dirs = part["Direction"].drop_duplicates().tolist()
        reps = ["Metabolome", "Microbiome", "Early integration"]
        width = 0.22
        x = np.arange(len(dirs))
        for j, rep in enumerate(reps):
            vals = [part.loc[part["Direction"].eq(d) & part["Representation"].eq(rep), "roc_auc"].iloc[0] for d in dirs]
            ax.bar(x + (j - 1) * width, vals, width=width, label=rep, color=PALETTE[rep])
        ax.axhline(0.5, ls="--", lw=1, color="#777777")
        ax.set_xticks(x)
        ax.set_xticklabels(dirs, rotation=15, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("External ROC-AUC")
        ax.set_title(f"{disease}: external transportability")
        panel_label(ax, "A" if disease == "IBD" else "B")
    axes[0, 1].legend(title="Representation", frameon=False, loc="lower right")

    for ax, disease in zip(axes[1], ["IBD", "CRC"]):
        part = comp.loc[comp["disease"].eq(disease)]
        labels = [f"{direction_from_row(r)}\n{REP_LABELS[r['representation']]}" for _, r in part.iterrows()]
        y = np.arange(len(part))
        colors = [PALETTE[REP_LABELS[r]] for r in part["representation"]]
        ax.barh(y, part["gap"], color=colors)
        ax.axvline(0, color="#777777", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Mean internal ROC-AUC minus external ROC-AUC")
        ax.set_title(f"{disease}: generalisation gap")
        panel_label(ax, "C" if disease == "IBD" else "D")
    fig.suptitle("Figure 2. Stage 4B single-omic and early-integration predictive reproducibility", fontweight="bold")
    save_figure(fig, "figure2_stage4b_external_prediction")


def figure3() -> None:
    mofa_var = pd.read_csv(TABLES / "stage4c_mofa_factor_variance_explained.csv")
    scca = pd.read_csv(TABLES / "stage4c_scca_component_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    for ax, disease, label in [(axes[0, 0], "IBD", "A"), (axes[0, 1], "CRC", "B")]:
        part = mofa_var[(mofa_var["disease"].eq(disease)) & (~mofa_var["factor"].eq("total"))].copy()
        pivot = part.groupby(["factor", "view"], as_index=False)["variance_explained"].mean().pivot(index="factor", columns="view", values="variance_explained").fillna(0)
        pivot = pivot.reindex([f"Factor{i}" for i in range(1, 9)])
        x = np.arange(len(pivot))
        bottom = np.zeros(len(pivot))
        for view, color in [("metabolome", PALETTE["Metabolome"]), ("microbiome", PALETTE["Microbiome"])]:
            vals = pivot.get(view, pd.Series(0, index=pivot.index)).to_numpy()
            ax.bar(x, vals, bottom=bottom, label=view.capitalize(), color=color)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(1, 9)])
        ax.set_xlabel("MOFA factor")
        ax.set_ylabel("Mean explained variance (%)")
        ax.set_title(f"{disease}: MOFA factor-view structure")
        panel_label(ax, label)
    axes[0, 1].legend(frameon=False, title="View")
    for ax, disease, label in [(axes[1, 0], "IBD", "C"), (axes[1, 1], "CRC", "D")]:
        part = scca[scca["disease"].eq(disease)].copy()
        ax.bar(part["cohort"], part["canonical_correlation"], color=PALETTE["sCCA"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Canonical correlation")
        ax.set_title(f"{disease}: cohort-specific sCCA")
        panel_label(ax, label)
    fig.suptitle("Figure 3. Stage 4C latent microbiome-metabolome integration", fontweight="bold")
    save_figure(fig, "figure3_stage4c_latent_integration")


def figure4() -> None:
    qc = pd.read_csv(TABLES / "stage4d_graph_qc_before_after_njc19.csv")
    cov = pd.read_csv(TABLES / "stage4d_graph_source_coverage_augmented.csv")
    fig = plt.figure(figsize=(11, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    boxes = [("KEGG\nmetabolite-metabolite", 0.08, 0.62, PALETTE["Metabolome"]), ("NJC19\nmicrobe-metabolite", 0.08, 0.28, PALETTE["Microbiome"]), ("Augmented fixed\nprior graph", 0.58, 0.45, "#444444")]
    for text, x, y, color in boxes:
        patch = FancyBboxPatch((x, y), 0.32, 0.18, boxstyle="round,pad=0.02,rounding_size=0.02", fc="white", ec=color, lw=1.7, transform=ax0.transAxes)
        ax0.add_patch(patch)
        ax0.text(x + 0.16, y + 0.09, text, ha="center", va="center", transform=ax0.transAxes)
    ax0.add_patch(FancyArrowPatch((0.40, 0.71), (0.58, 0.54), arrowstyle="-|>", mutation_scale=11, color="#666666", transform=ax0.transAxes))
    ax0.add_patch(FancyArrowPatch((0.40, 0.37), (0.58, 0.50), arrowstyle="-|>", mutation_scale=11, color="#666666", transform=ax0.transAxes))
    ax0.set_title("Prior sources, not phenotype-derived edges")
    panel_label(ax0, "A")

    ax1 = fig.add_subplot(gs[0, 1])
    after = qc[qc["qc_stage"].eq("after")]
    ax1.bar(after["disease_space"], after["microbe_metabolite_edges"], color=[PALETTE["IBD"], PALETTE["CRC"]])
    ax1.set_ylabel("NJC19 microbe-metabolite edges")
    ax1.set_title("Curated cross-omic edges added")
    panel_label(ax1, "B")

    ax2 = fig.add_subplot(gs[1, 0])
    for disease, color in [("IBD", PALETTE["IBD"]), ("CRC", PALETTE["CRC"])]:
        part = qc[qc["disease_space"].eq(disease)].set_index("qc_stage").loc[["before", "after"]]
        ax2.plot(["Before", "After"], part["isolated_node_count"], marker="o", lw=2, label=disease, color=color)
    ax2.set_ylabel("Isolated nodes")
    ax2.set_title("Connectivity improved after NJC19 augmentation")
    ax2.legend(frameon=False)
    panel_label(ax2, "C")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(cov["disease_space"], cov["microbial_coverage_percent"], color=[PALETTE["IBD"], PALETTE["CRC"]])
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("Genera with cross-omic edge (%)")
    ax3.set_title("Microbial graph coverage")
    panel_label(ax3, "D")
    fig.suptitle("Figure 4. Stage 4D curated prior graph augmentation", fontweight="bold")
    save_figure(fig, "figure4_stage4d_graph_augmentation")


def figure5() -> None:
    gcn = pd.read_csv(TABLES / "stage4e_gcn_paired_graph_mode_external_metrics.csv")
    stage4b = pd.read_csv(TABLES / "stage4b_external_performance.csv")
    stage4b = stage4b[stage4b["task"].isin(["IBD_vs_Control", "CRC_vs_Control"])].copy()
    best = stage4b.sort_values("roc_auc", ascending=False).groupby(["disease", "training_cohort", "external_cohort"], as_index=False).first()
    best["Direction"] = best.apply(direction_from_row, axis=1)
    best["Model"] = best["representation"].map(REP_LABELS)
    gcn["Direction"] = gcn.apply(direction_from_row, axis=1)
    gcn["Graph mode"] = gcn["graph_mode"].map({"augmented": "Augmented GCN", "identity": "Identity ablation"})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for ax, disease, label in [(axes[0, 0], "IBD", "A"), (axes[0, 1], "CRC", "B")]:
        part = gcn[gcn["disease"].eq(disease)]
        dirs = part["Direction"].drop_duplicates().tolist()
        x = np.arange(len(dirs))
        width = 0.32
        for j, mode in enumerate(["Augmented GCN", "Identity ablation"]):
            vals = [part.loc[part["Direction"].eq(d) & part["Graph mode"].eq(mode), "roc_auc"].iloc[0] for d in dirs]
            ax.bar(x + (j - 0.5) * width, vals, width=width, label=mode, color=PALETTE[mode])
        ax.axhline(0.5, ls="--", lw=1, color="#777777")
        ax.set_xticks(x)
        ax.set_xticklabels(dirs, rotation=15, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("External ROC-AUC")
        ax.set_title(f"{disease}: graph mode comparison")
        panel_label(ax, label)
    axes[0, 1].legend(frameon=False, title="GCN mode")

    ax = axes[1, 0]
    rows = []
    for _, r in best.iterrows():
        rows.append({"Disease": r["disease"], "Direction": r["Direction"], "Model": f"Stage 4B best\n({r['Model']})", "ROC-AUC": r["roc_auc"], "Color": PALETTE[r["Model"]]})
    for _, r in gcn[gcn["graph_mode"].eq("augmented")].iterrows():
        rows.append({"Disease": r["disease"], "Direction": r["Direction"], "Model": "Augmented GCN", "ROC-AUC": r["roc_auc"], "Color": PALETTE["Augmented GCN"]})
    cmp = pd.DataFrame(rows)
    labels = [f"{r['Disease']}\n{r['Direction']}\n{r['Model']}" for _, r in cmp.iterrows()]
    y = np.arange(len(cmp))
    ax.barh(y, cmp["ROC-AUC"], color=cmp["Color"])
    ax.axvline(0.5, ls="--", lw=1, color="#777777")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("External ROC-AUC")
    ax.set_title("GCN compared with frozen Stage 4B best model")
    panel_label(ax, "C")

    ax = axes[1, 1]
    probs = gcn.copy()
    probs["range"] = probs["probability_max"] - probs["probability_min"]
    y = np.arange(len(probs))
    colors = probs["Graph mode"].map(PALETTE)
    ax.barh(y, probs["range"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['Direction']}\n{r['Graph mode']}" for _, r in probs.iterrows()], fontsize=7)
    ax.set_xlabel("External predicted probability range")
    ax.set_title("Probability compression/saturation diagnostic")
    panel_label(ax, "D")
    fig.suptitle("Figure 5. Stage 4E GCN external validation and graph ablation", fontweight="bold")
    save_figure(fig, "figure5_stage4e_gcn_external_validation")


def write_manifest() -> None:
    rows = [
        {"figure": "figure1_study_design", "panel": "A-C", "source_tables": "stage4a_final_multiomics_feature_space.csv", "script": __file__},
        {"figure": "figure2_stage4b_external_prediction", "panel": "A-D", "source_tables": "stage4b_external_performance.csv; stage4b_internal_performance.csv", "script": __file__},
        {"figure": "figure3_stage4c_latent_integration", "panel": "A-D", "source_tables": "stage4c_mofa_factor_variance_explained.csv; stage4c_scca_component_summary.csv", "script": __file__},
        {"figure": "figure4_stage4d_graph_augmentation", "panel": "A-D", "source_tables": "stage4d_graph_qc_before_after_njc19.csv; stage4d_graph_source_coverage_augmented.csv", "script": __file__},
        {"figure": "figure5_stage4e_gcn_external_validation", "panel": "A-D", "source_tables": "stage4e_gcn_paired_graph_mode_external_metrics.csv; stage4b_external_performance.csv", "script": __file__},
    ]
    pd.DataFrame(rows).to_csv(OUT / "FINAL_FIGURE_PANEL_PROVENANCE.csv", index=False)
    status_rows = [
        {
            "figure": "figure1_study_design",
            "main_or_supplement": "main",
            "display_item": 1,
            "status_before_user_run": "NEEDS_USER_GENERATION_AND_VISUAL_QA",
            "reason": "Script written; final PNG/PDF/SVG not generated by Codex under USER-RUNS rule.",
        },
        {
            "figure": "figure2_stage4b_external_prediction",
            "main_or_supplement": "main",
            "display_item": 2,
            "status_before_user_run": "NEEDS_USER_GENERATION_AND_VISUAL_QA",
            "reason": "Script written; final PNG/PDF/SVG not generated by Codex under USER-RUNS rule.",
        },
        {
            "figure": "figure3_stage4c_latent_integration",
            "main_or_supplement": "main",
            "display_item": 3,
            "status_before_user_run": "NEEDS_USER_GENERATION_AND_VISUAL_QA",
            "reason": "Script written; final PNG/PDF/SVG not generated by Codex under USER-RUNS rule.",
        },
        {
            "figure": "figure4_stage4d_graph_augmentation",
            "main_or_supplement": "main",
            "display_item": 4,
            "status_before_user_run": "NEEDS_USER_GENERATION_AND_VISUAL_QA",
            "reason": "Script written; final PNG/PDF/SVG not generated by Codex under USER-RUNS rule.",
        },
        {
            "figure": "figure5_stage4e_gcn_external_validation",
            "main_or_supplement": "main",
            "display_item": 5,
            "status_before_user_run": "NEEDS_USER_GENERATION_AND_VISUAL_QA",
            "reason": "Script written; final PNG/PDF/SVG not generated by Codex under USER-RUNS rule.",
        },
    ]
    pd.DataFrame(status_rows).to_csv(OUT / "FINAL_FIGURE_SET_STATUS.csv", index=False)


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    write_manifest()
    print(f"Final thesis figures written to: {OUT}")


if __name__ == "__main__":
    main()
