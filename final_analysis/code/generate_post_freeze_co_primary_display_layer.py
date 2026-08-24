"""New, thesis-facing co-primary display table + figure (items 16-17).

This is deliberately a SEPARATE script from `freeze_final_stage4_results.py`,
not an edit to it. `freeze_final_stage4_results.py` is the historical
generator of the 20-August frozen authoritative files (including
`table2_stage4b_transportability.csv` and `figure_B_...png`); if its
REP_LABELS/analysis_role logic were edited in place, a future accidental
re-run of that script would silently regenerate — and thereby mutate — those
frozen, hash-tracked outputs with the new co-primary wording. That would
defeat the entire "keep the historical freeze intact, add an addendum on top"
approach used elsewhere in this patch (see
manifests/POST_FREEZE_ANALYSIS_ROLE_ADDENDUM.json).

Instead, this script reads the same already-frozen numeric source
(`stage4b_corrected_v2b_external_performance.csv`) and independently builds a
new display table and new figure at NEW paths, under the co-primary framing.
It performs no model fitting and does not import from or write to anything
inside `tables/final_stage4/` or `figures/final_stage4/`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from stage4_corrected_v2_common import FIGURES, TABLES, atomic_csv, ensure_v2_dirs


# Short per-panel axis labels (role is conveyed once, by the legend, rather than
# repeated on every one of the 12 y-ticks) plus the full legend wording.
REP_SHORT_LABELS = {"METABOLOME": "Metabolome", "MICROBIOME": "Microbiome", "EARLY_INTEGRATION": "Early integration"}
REP_LEGEND_LABELS = {
    "METABOLOME": "Metabolome (co-primary)",
    "MICROBIOME": "Microbiome (co-primary)",
    "EARLY_INTEGRATION": "Early integration (comparator)",
}
# Okabe-Ito colorblind-safe palette; same mapping used consistently everywhere
# this representation is plotted in the corrected_v2 figure set (matches
# REP_COLORS in freeze_final_stage4_results.py).
REP_COLORS = {"METABOLOME": "#56B4E9", "MICROBIOME": "#009E73", "EARLY_INTEGRATION": "#E69F00"}
# Bottom-to-top plotting order (y=0 is the bottom row of each panel), chosen so the
# reading order top-to-bottom is Microbiome, Metabolome, Early integration.
REP_PLOT_ORDER = ["EARLY_INTEGRATION", "METABOLOME", "MICROBIOME"]

OUT_TABLE = TABLES / "post_freeze_table2_stage4b_transportability_co_primary.csv"
OUT_FIGURE_STEM = "post_freeze_figure_B_stage4b_transportability_co_primary"


def build_co_primary_table2() -> pd.DataFrame:
    ext = pd.read_csv(TABLES / "stage4b_corrected_v2b_external_performance.csv", low_memory=False)
    out = ext.copy()
    out["direction"] = out["training_cohort"] + "→" + out["external_cohort"]
    out["analysis_role"] = np.where(out["representation"].eq("EARLY_INTEGRATION"), "integration_comparison", "co_primary_single_omic")
    # The microbiome platform caveat (16S vs. shotgun confounded with cohort) is CRC-specific
    # (Sinha=16S, Yachida=shotgun); it does not apply to the IBD pair, which is shotgun vs.
    # shotgun on both sides (Franzosa, iHMP) — though database/processing heterogeneity remains
    # even there. Applying the CRC wording to IBD rows would be a disease-irrelevant caveat.
    out["external_interpretation_scope"] = np.select(
        [
            out["representation"].eq("MICROBIOME") & out["disease"].eq("IBD"),
            out["representation"].eq("MICROBIOME") & out["disease"].eq("CRC"),
            out["representation"].eq("METABOLOME"),
        ],
        [
            "Cross-cohort transportability of harmonised genus composition. Caveat: both Franzosa "
            "and iHMP are shotgun metagenomics, but taxonomic database/processing heterogeneity "
            "between the two source pipelines remains, so this is not a perfectly harmonised "
            "representation either.",
            "Cross-cohort transportability of harmonised genus composition. Caveat: Sinha (16S) vs. "
            "Yachida (shotgun) is platform-confounded with cohort, so this is not a perfectly "
            "harmonised representation either.",
            "Cross-cohort performance under the supplied cohort-specific processed metabolomic "
            "matrices. Caveat: upstream quantitative-scale equivalence across assays (LC-MS, "
            "Metabolon, CE-TOFMS) was not verified; HMDB ID matching establishes feature identity, "
            "not common units, response factors, or transformations.",
        ],
        default="Whether combining the two co-primary molecular layers improves on either single-omic representation.",
    )
    out["probability_diagnostic"] = np.where(
        out["probability_saturation_flag"] | out["probability_resolution_loss_flag"] | out["one_class_prediction_flag"],
        "probability_pathology_or_one_class_flag",
        "no_major_probability_flag",
    )
    cols = [
        "disease", "task", "direction", "representation", "analysis_role", "external_interpretation_scope",
        "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "pr_auc", "mcc", "probability_diagnostic", "sensitivity", "specificity",
    ]
    return out[cols]


# Explicit panel order: top row = IBD (Franzosa<->iHMP), bottom row = CRC (Sinha<->Yachida).
# A plain alphabetical groupby("direction") sort scatters these (ASCII puts lowercase
# "iHMP->Franzosa" after all the uppercase-first directions), mixing the two diseases
# across the 2x2 grid, which is confusing to read.
PANEL_ORDER = ["Franzosa→iHMP", "iHMP→Franzosa", "Sinha→Yachida", "Yachida→Sinha"]


def make_figure(table2: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 13,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), sharex=True)
    fig.subplots_adjust(hspace=0.55, wspace=0.35, left=0.14, right=0.97, top=0.84, bottom=0.16)
    by_direction = dict(tuple(table2.groupby("direction")))
    missing = [d for d in PANEL_ORDER if d not in by_direction]
    if missing:
        raise ValueError(f"table2 is missing expected direction(s) {missing}; found {list(by_direction)}")
    for ax, direction in zip(axes.ravel(), PANEL_ORDER):
        sub = by_direction[direction].set_index("representation").loc[REP_PLOT_ORDER].reset_index()
        y = np.arange(len(sub))
        colors = [REP_COLORS.get(r, "#777777") for r in sub["representation"]]
        ax.errorbar(sub["roc_auc"], y, xerr=[sub["roc_auc"] - sub["roc_auc_ci_low"], sub["roc_auc_ci_high"] - sub["roc_auc"]], fmt="none", ecolor="0.25", capsize=4, lw=1.6, zorder=2)
        ax.scatter(sub["roc_auc"], y, c=colors, s=110, zorder=3, edgecolors="white", linewidths=0.8)
        ax.axvline(0.5, color="0.55", ls=":", lw=1.3, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels([REP_SHORT_LABELS.get(r, r) for r in sub["representation"]])
        ax.set_title(direction, pad=10, weight="bold")
        ax.set_xlim(0.25, 1.0)
        ax.tick_params(axis="both", length=4)
        ax.margins(y=0.35)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=REP_COLORS[r], markeredgecolor="white", markersize=13, label=REP_LEGEND_LABELS[r])
        for r in ["MICROBIOME", "METABOLOME", "EARLY_INTEGRATION"]
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, frameon=False, fontsize=13, handletextpad=0.6, columnspacing=1.8)

    fig.suptitle("Figure B (post-freeze) | Stage 4B external transportability — microbiome and metabolome co-primary", fontsize=17, weight="bold", y=0.985)
    fig.text(0.5, 0.095, "External ROC-AUC with 95% bootstrap CI  (dotted line = chance, AUC = 0.5)", ha="center", fontsize=13)
    fig.text(0.5, 0.035, "Metabolome retains an upstream cross-assay quantitative-scale caveat (see table footnote); this does not demote its analytical priority.", ha="center", fontsize=11, style="italic", wrap=True)

    out_path = FIGURES / "final_stage4"
    out_path.mkdir(parents=True, exist_ok=True)
    png = out_path / f"{OUT_FIGURE_STEM}.png"
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"Wrote {png}")


def main() -> None:
    ensure_v2_dirs()
    table2 = build_co_primary_table2()
    atomic_csv(table2, OUT_TABLE)
    print(f"Wrote {OUT_TABLE}")
    make_figure(table2)
    print("\nCo-primary display table (all rows):")
    print(table2[["direction", "representation", "analysis_role", "roc_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
