from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stage4_corrected_v2_common import FIGURES, TABLES, ensure_v2_dirs


COLORS = {
    "IBD": "#0072B2",
    "CRC": "#D55E00",
    "METABOLOME": "#56B4E9",
    "MICROBIOME": "#009E73",
    "EARLY_INTEGRATION": "#E69F00",
}


def require(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required corrected-v2 table is missing. Run corrected analyses first: {path}")
    return pd.read_csv(path)


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_external_transportability() -> None:
    ext = require(TABLES / "stage4b_corrected_v2_external_performance.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.ravel()
    contexts = ext[["disease", "task", "training_cohort", "external_cohort"]].drop_duplicates().sort_values(["disease", "task", "training_cohort"])
    for ax, (_, ctx) in zip(axes, contexts.head(4).iterrows()):
        sub = ext.merge(pd.DataFrame([ctx]), on=["disease", "task", "training_cohort", "external_cohort"])
        x = np.arange(len(sub))
        ax.bar(x, sub["roc_auc"], color=[COLORS[r] for r in sub["representation"]])
        ax.errorbar(
            x,
            sub["roc_auc"],
            yerr=[
                sub["roc_auc"] - sub["roc_auc_ci_low"],
                sub["roc_auc_ci_high"] - sub["roc_auc"],
            ],
            fmt="none",
            ecolor="black",
            capsize=3,
            lw=1,
        )
        ax.axhline(0.5, color="0.4", ls=":", lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["representation"].str.replace("_", "\n"), fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"{ctx['task']}: {ctx['training_cohort']}→{ctx['external_cohort']}")
        ax.set_ylabel("External ROC-AUC")
    fig.suptitle("Corrected-v2 external predictive transportability")
    save(fig, "display3_corrected_v2_external_transportability")


def figure_early_integration_delta() -> None:
    ext = require(TABLES / "stage4b_corrected_v2_external_performance.csv")
    rows = []
    keys = ["disease", "task", "training_cohort", "external_cohort"]
    for ctx, sub in ext.groupby(keys):
        vals = sub.set_index("representation")["roc_auc"]
        if "EARLY_INTEGRATION" not in vals:
            continue
        for comparator in ["METABOLOME", "MICROBIOME"]:
            if comparator in vals:
                rows.append({**dict(zip(keys, ctx)), "comparison": f"Early integration - {comparator.lower()}", "delta_auc": vals["EARLY_INTEGRATION"] - vals[comparator]})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No early-integration delta rows available.")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(df))))
    y = np.arange(len(df))
    ax.scatter(df["delta_auc"], y, color=df["disease"].map(COLORS), s=55)
    ax.axvline(0, color="0.35", ls=":", lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.task} | {r.training_cohort}→{r.external_cohort}\n{r.comparison}" for r in df.itertuples()], fontsize=8)
    ax.set_xlabel("Δ external ROC-AUC")
    ax.set_title("Corrected-v2 early-integration benefit over single omics")
    save(fig, "display4_corrected_v2_early_integration_delta_auc")


def main() -> None:
    ensure_v2_dirs()
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    figure_external_transportability()
    figure_early_integration_delta()
    print("CORRECTED_V2_THESIS_FIGURES_READY")


if __name__ == "__main__":
    main()
