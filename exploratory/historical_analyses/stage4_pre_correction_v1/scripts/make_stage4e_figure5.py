from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


PROJECT = Path("/Users/bhaktidoshi/Downloads/Thesis Data")
STAGE4 = PROJECT / "thesis_preprocessing_workflow" / "stage4_multiomics_integration"
TABLES = STAGE4 / "tables"
FIGURES = STAGE4 / "figures"

GCN_METRICS = TABLES / "stage4e_gcn_paired_graph_mode_external_metrics.csv"
GCN_PREDICTIONS = TABLES / "stage4e_gcn_paired_graph_mode_external_predictions.csv"
GCN_SELECTED_METRICS = TABLES / "stage4e_gcn_external_metrics.csv"
GCN_SELECTED_PREDICTIONS = TABLES / "stage4e_gcn_predictions_external.csv"
STAGE4B_EXTERNAL = TABLES / "stage4b_external_performance.csv"

OUT_PNG = FIGURES / "stage4e_figure5_gcn_external_validation.png"
OUT_PDF = FIGURES / "stage4e_figure5_gcn_external_validation.pdf"

GRAPH_MODE_LABELS = {
    "augmented": "Augmented GCN",
    "identity": "Identity graph",
}
GRAPH_MODE_COLORS = {
    "augmented": "#CC79A7",
    "identity": "#7F7F7F",
}
STAGE4B_COLORS = {
    "METABOLOME": "#56B4E9",
    "MICROBIOME": "#009E73",
    "EARLY_INTEGRATION": "#E69F00",
}
REP_LABELS = {
    "METABOLOME": "Stage 4B best: metabolome",
    "MICROBIOME": "Stage 4B best: microbiome",
    "EARLY_INTEGRATION": "Stage 4B best: early integration",
}
DIRECTION_LABELS = {
    ("Franzosa", "iHMP"): "Franzosa->iHMP",
    ("iHMP", "Franzosa"): "iHMP->Franzosa",
    ("Sinha", "Yachida"): "Sinha->Yachida",
    ("Yachida", "Sinha"): "Yachida->Sinha",
}
DIRECTION_ORDER = [
    ("Franzosa", "iHMP"),
    ("iHMP", "Franzosa"),
    ("Sinha", "Yachida"),
    ("Yachida", "Sinha"),
]


def direction_label(row: pd.Series) -> str:
    return DIRECTION_LABELS.get(
        (row["training_cohort"], row["external_cohort"]),
        f"{row['training_cohort']}->{row['external_cohort']}",
    )


def direction_sort_key(row: pd.Series) -> int:
    pair = (row["training_cohort"], row["external_cohort"])
    return DIRECTION_ORDER.index(pair) if pair in DIRECTION_ORDER else 99


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gcn_metrics = pd.read_csv(GCN_METRICS)
    gcn_predictions = pd.read_csv(GCN_PREDICTIONS)
    stage4b_external = pd.read_csv(STAGE4B_EXTERNAL)

    selected_metrics = pd.read_csv(GCN_SELECTED_METRICS)
    selected_predictions = pd.read_csv(GCN_SELECTED_PREDICTIONS)
    if selected_metrics.empty or selected_predictions.empty:
        raise ValueError("Selected Stage 4E external metric/prediction tables are empty.")

    required = {
        "direction",
        "disease",
        "task",
        "training_cohort",
        "external_cohort",
        "graph_mode",
        "probability_min",
        "probability_max",
        "roc_auc",
    }
    missing = required.difference(gcn_metrics.columns)
    if missing:
        raise ValueError(f"Missing required GCN metric columns: {sorted(missing)}")

    for df in [gcn_metrics, stage4b_external]:
        df["direction_label"] = df.apply(direction_label, axis=1)
        df["direction_order"] = df.apply(direction_sort_key, axis=1)

    return gcn_metrics, gcn_predictions, stage4b_external


def best_stage4b_by_direction(stage4b_external: pd.DataFrame) -> pd.DataFrame:
    stage4b = stage4b_external[
        stage4b_external["task"].isin(["IBD_vs_Control", "CRC_vs_Control"])
    ].copy()
    stage4b = stage4b.sort_values(
        ["disease", "training_cohort", "external_cohort", "roc_auc"],
        ascending=[True, True, True, False],
    )
    best = stage4b.groupby(
        ["disease", "task", "training_cohort", "external_cohort"],
        as_index=False,
    ).first()
    best["direction_label"] = best.apply(direction_label, axis=1)
    best["direction_order"] = best.apply(direction_sort_key, axis=1)
    best["model_label"] = best["representation"].map(REP_LABELS)
    return best.sort_values("direction_order")


def plot_auc_by_disease(ax: plt.Axes, metrics: pd.DataFrame, disease: str, panel: str) -> None:
    part = metrics.loc[metrics["disease"].eq(disease)].copy()
    part = part.sort_values(["direction_order", "graph_mode"])
    directions = [DIRECTION_LABELS[pair] for pair in DIRECTION_ORDER if pair[0] in set(part["training_cohort"])]
    x = np.arange(len(directions))
    width = 0.34
    for offset, mode in [(-0.5, "augmented"), (0.5, "identity")]:
        vals = []
        for direction in directions:
            val = part.loc[
                part["direction_label"].eq(direction) & part["graph_mode"].eq(mode),
                "roc_auc",
            ]
            vals.append(float(val.iloc[0]) if len(val) else np.nan)
        bars = ax.bar(
            x + offset * width,
            vals,
            width=width,
            color=GRAPH_MODE_COLORS[mode],
            label=GRAPH_MODE_LABELS[mode],
        )
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.025,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.axhline(0.5, color="#555555", linestyle=":", linewidth=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(directions, rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("External ROC-AUC")
    ax.set_title(f"{panel}. {disease}: augmented vs identity graph")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)


def plot_stage4b_comparison(ax: plt.Axes, metrics: pd.DataFrame, stage4b_best: pd.DataFrame) -> None:
    rows = []
    aug = metrics.loc[metrics["graph_mode"].eq("augmented")].copy()
    for _, row in aug.iterrows():
        rows.append(
            {
                "label": f"{row['direction_label']}\nAugmented GCN",
                "roc_auc": row["roc_auc"],
                "color": GRAPH_MODE_COLORS["augmented"],
                "order": row["direction_order"] * 2,
            }
        )
    for _, row in stage4b_best.iterrows():
        rows.append(
            {
                "label": f"{row['direction_label']}\n{row['model_label']}",
                "roc_auc": row["roc_auc"],
                "color": STAGE4B_COLORS[row["representation"]],
                "order": row["direction_order"] * 2 + 1,
            }
        )
    comp = pd.DataFrame(rows).sort_values("order", ascending=False)
    y = np.arange(len(comp))
    ax.barh(y, comp["roc_auc"], color=comp["color"])
    for yi, val in zip(y, comp["roc_auc"]):
        ax.text(min(val + 0.025, 0.98), yi, f"{val:.2f}", va="center", fontsize=8)
    ax.axvline(0.5, color="#555555", linestyle=":", linewidth=1.4)
    ax.text(0.505, len(comp) - 0.2, "Chance", fontsize=8, color="#555555", va="top")
    ax.set_xlim(0, 1.05)
    ax.set_yticks(y)
    ax.set_yticklabels(comp["label"], fontsize=8)
    ax.set_xlabel("External ROC-AUC")
    ax.set_title("C. Augmented GCN vs best frozen Stage 4B model")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)


def plot_probability_range(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    probs = metrics.copy()
    probs["probability_range"] = probs["probability_max"] - probs["probability_min"]
    probs = probs.sort_values(["direction_order", "graph_mode"], ascending=[False, False])
    labels = [
        f"{row.direction_label}\n{GRAPH_MODE_LABELS[row.graph_mode]}"
        for row in probs.itertuples(index=False)
    ]
    y = np.arange(len(probs))
    colors = probs["graph_mode"].map(GRAPH_MODE_COLORS)
    ax.barh(y, probs["probability_range"], color=colors)
    for yi, row in enumerate(probs.itertuples(index=False)):
        ax.text(
            row.probability_range + 0.015,
            yi,
            f"{row.probability_min:.3f}-{row.probability_max:.3f}",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, 1.05)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("External predicted-probability range")
    ax.set_title("D. Probability compression or saturation")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )

    metrics, _predictions, stage4b_external = load_inputs()
    stage4b_best = best_stage4b_by_direction(stage4b_external)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.6), constrained_layout=False)
    plot_auc_by_disease(axes[0, 0], metrics, "IBD", "A")
    plot_auc_by_disease(axes[0, 1], metrics, "CRC", "B")
    plot_stage4b_comparison(axes[1, 0], metrics, stage4b_best)
    plot_probability_range(axes[1, 1], metrics)

    legend_items = [
        Patch(facecolor=GRAPH_MODE_COLORS["augmented"], label="Augmented GCN"),
        Patch(facecolor=GRAPH_MODE_COLORS["identity"], label="Identity graph ablation"),
        Patch(facecolor=STAGE4B_COLORS["METABOLOME"], label="Stage 4B best: metabolome"),
        Patch(facecolor=STAGE4B_COLORS["MICROBIOME"], label="Stage 4B best: microbiome"),
        Patch(facecolor=STAGE4B_COLORS["EARLY_INTEGRATION"], label="Stage 4B best: early integration"),
    ]
    fig.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.105),
        ncol=3,
        frameon=False,
        title="Model / graph mode",
    )
    fig.suptitle("Figure 5 | Stage 4E GCN external validation and graph ablation", fontweight="bold")
    fig.text(
        0.5,
        0.025,
        "The dotted line denotes chance discrimination at ROC-AUC=0.5. "
        "Probabilities are frozen external predictions from saved Stage 4E outputs.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.20, hspace=0.42, wspace=0.22)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
