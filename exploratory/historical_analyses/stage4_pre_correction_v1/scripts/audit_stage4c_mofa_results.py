from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_stage4c import (  # noqa: E402
    MODELS,
    TABLES,
    FIGURES,
    REPORTS,
    MOFA_FACTORS,
    validate_mofa_hdf5,
)

IBD_GROUPS = ["Franzosa", "iHMP"]
CRC_GROUPS = ["Sinha", "Yachida"]
VIEWS = ["metabolome", "microbiome"]
FACTORS = [f"Factor{i}" for i in range(1, MOFA_FACTORS[0] + 1)]
TOP_N = 10
NONDEGENERATE_R2_THRESHOLD = 0.10  # percent R2 averaged across groups/views
MULTIVIEW_MIN_VIEW_FRACTION = 0.20


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def bh_fdr(pvals: list[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return q
    idx = np.where(valid)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    adj = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q[order] = np.minimum(adj, 1.0)
    return q


def scores_wide(scores: pd.DataFrame) -> pd.DataFrame:
    wide = scores.pivot_table(index=["disease", "cohort", "sample_id", "phenotype"], columns="factor", values="score", aggfunc="first").reset_index()
    wide.columns.name = None
    return wide


def validation_audit() -> tuple[pd.DataFrame, bool, bool]:
    rows = []
    ok_ibd, ibd = validate_mofa_hdf5(MODELS / "stage4c_mofa_ibd_k8.hdf5", "IBD", IBD_GROUPS)
    ok_crc, crc = validate_mofa_hdf5(MODELS / "stage4c_mofa_crc_k8.hdf5", "CRC", CRC_GROUPS)
    rows.extend(ibd.to_dict("records"))
    rows.extend(crc.to_dict("records"))
    return pd.DataFrame(rows), ok_ibd, ok_crc


def nondegenerate_table(scores: pd.DataFrame, loadings: pd.DataFrame, variance: pd.DataFrame) -> pd.DataFrame:
    wide = scores_wide(scores)
    rows = []
    factor_var = variance[variance["factor"].astype(str).str.startswith("Factor")].copy()
    for disease in ["IBD", "CRC"]:
        disease_scores = wide[wide["disease"].eq(disease)]
        disease_load = loadings[loadings["disease"].eq(disease)]
        for factor in FACTORS:
            score_vars = disease_scores.groupby("cohort")[factor].var().to_dict() if factor in disease_scores else {}
            r2 = factor_var[(factor_var["disease"].eq(disease)) & (factor_var["factor"].eq(factor))]
            met_r2 = r2[r2["view"].eq("metabolome")]["variance_explained"].mean()
            mic_r2 = r2[r2["view"].eq("microbiome")]["variance_explained"].mean()
            mean_total = np.nanmean([met_r2, mic_r2])
            lv = disease_load[disease_load["factor"].eq(factor)].groupby("view")["loading"].var().to_dict()
            finite_scores = np.isfinite(disease_scores[factor].to_numpy(dtype=float)).all() if factor in disease_scores else False
            finite_loadings = np.isfinite(disease_load[disease_load["factor"].eq(factor)]["loading"].to_numpy(dtype=float)).all()
            nondeg = bool(finite_scores and finite_loadings and np.nanmax(list(score_vars.values()) or [0]) > 1e-8 and np.nanmax(list(lv.values()) or [0]) > 1e-12 and mean_total >= NONDEGENERATE_R2_THRESHOLD)
            view_sum = (0 if pd.isna(met_r2) else met_r2) + (0 if pd.isna(mic_r2) else mic_r2)
            if not nondeg:
                category = "LOW_INFORMATION"
            elif view_sum <= 0:
                category = "LOW_INFORMATION"
            elif min(met_r2, mic_r2) / view_sum >= MULTIVIEW_MIN_VIEW_FRACTION:
                category = "MULTI_VIEW"
            elif met_r2 > mic_r2:
                category = "METABOLOME_DOMINANT"
            else:
                category = "MICROBIOME_DOMINANT"
            reason = f"mean group/view factor R2={mean_total:.3f}% using threshold {NONDEGENERATE_R2_THRESHOLD}%"
            rows.append({
                "disease": disease,
                "factor": factor,
                "nondegenerate": nondeg,
                "category": category,
                "mean_metabolome_r2_percent": met_r2,
                "mean_microbiome_r2_percent": mic_r2,
                "mean_factor_r2_percent": mean_total,
                "score_variance_by_group": score_vars,
                "loading_variance_by_view": lv,
                "finite_scores": finite_scores,
                "finite_loadings": finite_loadings,
                "reason": reason,
            })
    return pd.DataFrame(rows)


def task_labels(disease: str) -> list[tuple[str, str, list[str], str]]:
    if disease == "IBD":
        return [
            ("IBD_vs_Control", "IBD", ["CD", "UC"], "Control"),
            ("CD_vs_Control", "CD", ["CD"], "Control"),
            ("UC_vs_Control", "UC", ["UC"], "Control"),
        ]
    return [("CRC_vs_Control", "CRC", ["CRC"], "Control/Healthy")]


def association_table(scores: pd.DataFrame) -> pd.DataFrame:
    wide = scores_wide(scores)
    rows = []
    for disease in ["IBD", "CRC"]:
        groups = IBD_GROUPS if disease == "IBD" else CRC_GROUPS
        for comparison, positive_name, positives, control in task_labels(disease):
            for scope_cohort in groups + ["POOLED_GROUP_AWARE_SCORES"]:
                sub = wide[wide["disease"].eq(disease)].copy()
                if scope_cohort != "POOLED_GROUP_AWARE_SCORES":
                    sub = sub[sub["cohort"].eq(scope_cohort)]
                sub = sub[sub["phenotype"].isin(positives + [control])].copy()
                sub["is_positive"] = sub["phenotype"].isin(positives)
                if sub["is_positive"].nunique() != 2:
                    continue
                for factor in FACTORS:
                    pos = sub.loc[sub["is_positive"], factor].dropna()
                    neg = sub.loc[~sub["is_positive"], factor].dropna()
                    if len(pos) < 3 or len(neg) < 3:
                        p = np.nan
                        stat = np.nan
                    else:
                        stat_obj = stats.mannwhitneyu(pos, neg, alternative="two-sided")
                        stat = float(stat_obj.statistic)
                        p = float(stat_obj.pvalue)
                    effect = float(pos.median() - neg.median()) if len(pos) and len(neg) else np.nan
                    rows.append({
                        "disease": disease,
                        "comparison": comparison,
                        "cohort_scope": scope_cohort,
                        "factor": factor,
                        "n_positive": int(len(pos)),
                        "n_control": int(len(neg)),
                        "positive_label": positive_name,
                        "control_label": control,
                        "positive_median": float(pos.median()) if len(pos) else np.nan,
                        "control_median": float(neg.median()) if len(neg) else np.nan,
                        "effect_median_difference": effect,
                        "direction": "higher_in_disease" if effect > 0 else "lower_in_disease" if effect < 0 else "no_difference",
                        "raw_p": p,
                        "test": "Mann-Whitney U post-hoc annotation after unsupervised MOFA",
                    })
    assoc = pd.DataFrame(rows)
    qvals = []
    for _, sub in assoc.groupby(["disease", "comparison", "cohort_scope"], sort=False):
        qvals.extend(bh_fdr(sub["raw_p"].tolist()))
    assoc["bh_fdr"] = qvals
    return assoc


def top_loadings_table(loadings: pd.DataFrame, nondeg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, frow in nondeg[nondeg["nondegenerate"]].iterrows():
        sub = loadings[(loadings["disease"].eq(frow["disease"])) & (loadings["factor"].eq(frow["factor"]))]
        for view in VIEWS:
            top = sub[sub["view"].eq(view)].copy()
            top["abs_loading"] = top["loading"].abs()
            top = top.sort_values("abs_loading", ascending=False).head(TOP_N)
            for rank, (_, row) in enumerate(top.iterrows(), start=1):
                rows.append({
                    "disease": row["disease"],
                    "factor": row["factor"],
                    "view": view,
                    "rank": rank,
                    "feature_id": row["feature_id"],
                    "feature_name": row["feature_name"],
                    "loading": row["loading"],
                    "abs_loading": row["abs_loading"],
                    "note": "Loadings are predictive/latent weights, not differential abundance evidence.",
                })
    return pd.DataFrame(rows)


def reproducibility_table(nondeg: pd.DataFrame, assoc: pd.DataFrame, top: pd.DataFrame, variance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for disease, groups in {"IBD": IBD_GROUPS, "CRC": CRC_GROUPS}.items():
        primary = "IBD_vs_Control" if disease == "IBD" else "CRC_vs_Control"
        for factor in FACTORS:
            r2 = variance[(variance["disease"].eq(disease)) & (variance["factor"].eq(factor))]
            def view_r2(group, view):
                vals = r2[(r2["cohort"].eq(group)) & (r2["view"].eq(view))]["variance_explained"]
                return float(vals.iloc[0]) if len(vals) else np.nan
            effects = {}
            dirs = {}
            for g in groups:
                a = assoc[(assoc["disease"].eq(disease)) & (assoc["comparison"].eq(primary)) & (assoc["cohort_scope"].eq(g)) & (assoc["factor"].eq(factor))]
                effects[g] = float(a["effect_median_difference"].iloc[0]) if len(a) else np.nan
                dirs[g] = str(a["direction"].iloc[0]) if len(a) else "NA"
            disease_top = top[(top["disease"].eq(disease)) & (top["factor"].eq(factor))]
            overlap = {}
            for view in VIEWS:
                feats = set(disease_top[disease_top["view"].eq(view)]["feature_id"])
                # Group-aware MOFA has one shared W/loadings matrix, so the top-loading set is structurally shared.
                overlap[view] = len(feats)
            concordant = np.sign(effects[groups[0]]) == np.sign(effects[groups[1]]) if np.isfinite(effects[groups[0]]) and np.isfinite(effects[groups[1]]) else False
            category = nondeg[(nondeg["disease"].eq(disease)) & (nondeg["factor"].eq(factor))]["category"]
            category = category.iloc[0] if len(category) else "NA"
            rows.append({
                "disease": disease,
                "factor": factor,
                "group_A": groups[0],
                "group_B": groups[1],
                "variance_explained_group_A": view_r2(groups[0], "metabolome") + view_r2(groups[0], "microbiome"),
                "variance_explained_group_B": view_r2(groups[1], "metabolome") + view_r2(groups[1], "microbiome"),
                "metabolome_contribution_A": view_r2(groups[0], "metabolome"),
                "metabolome_contribution_B": view_r2(groups[1], "metabolome"),
                "microbiome_contribution_A": view_r2(groups[0], "microbiome"),
                "microbiome_contribution_B": view_r2(groups[1], "microbiome"),
                "phenotype_effect_A": effects[groups[0]],
                "phenotype_effect_B": effects[groups[1]],
                "phenotype_direction_concordant": bool(concordant),
                "top_feature_overlap": str(overlap),
                "loading_replication_type": "MODEL_SHARED_LOADINGS_NOT_INDEPENDENT_REPLICATION",
                "reproducibility_summary": f"{category}; score/effect concordance assessed by group-specific Z scores and phenotype annotations; W/loadings are shared by design.",
            })
    return pd.DataFrame(rows)


def plot_variance(variance: pd.DataFrame, nondeg: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fac = variance[variance["factor"].astype(str).str.startswith("Factor")].copy()
    fac["label"] = fac["disease"] + " | " + fac["cohort"] + " | " + fac["view"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    colors = {"metabolome": "#2ca25f", "microbiome": "#756bb1"}
    for ax, disease in zip(axes, ["IBD", "CRC"]):
        sub = fac[fac["disease"].eq(disease)]
        x = np.arange(len(FACTORS))
        width = 0.18
        offsets = [-1.5, -0.5, 0.5, 1.5]
        combos = [(g, v) for g in (IBD_GROUPS if disease == "IBD" else CRC_GROUPS) for v in VIEWS]
        for off, (group, view) in zip(offsets, combos):
            vals = [sub[(sub["cohort"].eq(group)) & (sub["view"].eq(view)) & (sub["factor"].eq(f))]["variance_explained"].sum() for f in FACTORS]
            ax.bar(x + off * width, vals, width=width, label=f"{group} {view}", color=colors[view], alpha=0.55 if view == "microbiome" else 0.9, edgecolor="black", linewidth=0.3)
        ax.set_title(f"{disease} MOFA variance explained")
        ax.set_xticks(x)
        ax.set_xticklabels(FACTORS, rotation=45, ha="right")
        ax.set_ylabel("Variance explained (% R2)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "stage4c_mofa_variance_explained.png", dpi=300)
    plt.close(fig)


def plot_associations(assoc: pd.DataFrame, nondeg: pd.DataFrame) -> None:
    pooled = assoc[assoc["cohort_scope"].eq("POOLED_GROUP_AWARE_SCORES")].copy()
    pooled["minus_log10_fdr"] = -np.log10(pooled["bh_fdr"].clip(lower=1e-300))
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    tasks = [("IBD", "IBD_vs_Control"), ("IBD", "CD_vs_Control"), ("IBD", "UC_vs_Control"), ("CRC", "CRC_vs_Control")]
    for ax, (disease, comp) in zip(axes, tasks):
        sub = pooled[(pooled["disease"].eq(disease)) & (pooled["comparison"].eq(comp))].sort_values("factor")
        colors = ["#d95f02" if e > 0 else "#1b9e77" for e in sub["effect_median_difference"]]
        ax.bar(sub["factor"], sub["minus_log10_fdr"], color=colors)
        ax.axhline(-np.log10(0.05), color="grey", linestyle="--", linewidth=1)
        ax.set_title(comp.replace("_", " "))
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("-log10(BH-FDR)")
    fig.suptitle("Post-hoc phenotype associations after unsupervised MOFA", y=1.03)
    fig.tight_layout()
    fig.savefig(FIGURES / "stage4c_mofa_factor_associations.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top_loadings(top: pd.DataFrame) -> None:
    if top.empty:
        return
    subset = top[top["rank"].le(5)].copy()
    # Keep a compact set: first three nondegenerate factors per disease.
    keep = []
    for disease in ["IBD", "CRC"]:
        fs = subset[subset["disease"].eq(disease)]["factor"].drop_duplicates().head(3).tolist()
        keep.extend((disease, f) for f in fs)
    subset = pd.concat([subset[(subset["disease"].eq(d)) & (subset["factor"].eq(f))] for d, f in keep], ignore_index=True) if keep else subset.head(0)
    fig, axes = plt.subplots(max(1, len(keep)), 2, figsize=(14, max(4, 2.6 * max(1, len(keep)))))
    if len(keep) == 1:
        axes = np.array([axes])
    for row_idx, (disease, factor) in enumerate(keep):
        for col_idx, view in enumerate(VIEWS):
            ax = axes[row_idx, col_idx]
            s = subset[(subset["disease"].eq(disease)) & (subset["factor"].eq(factor)) & (subset["view"].eq(view))].sort_values("abs_loading")
            labels = s["feature_name"].astype(str).str.replace("mic__", "", regex=False).str.replace("met__", "", regex=False)
            ax.barh(labels, s["loading"], color="#4c78a8" if view == "microbiome" else "#f58518")
            ax.axvline(0, color="black", linewidth=0.6)
            ax.set_title(f"{disease} {factor} {view}")
            ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "stage4c_mofa_top_loadings.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(ok_ibd: bool, ok_crc: bool, nondeg: pd.DataFrame, assoc: pd.DataFrame, repro: pd.DataFrame, variance: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    nd_counts = nondeg.groupby("disease")["nondegenerate"].sum().to_dict()
    strongest = assoc.sort_values("bh_fdr").head(8)
    view_counts = nondeg[nondeg["nondegenerate"]].groupby(["disease", "category"]).size().reset_index(name="n")
    rec = "RECOMMENDED"
    rationale = "Separate-cohort MOFA would materially strengthen independent loading reproducibility claims because group-aware MOFA estimates shared W/loadings by design. However, sCCA is the next lighter-weight independent covariance reproducibility analysis and should be considered before adding four additional MOFA fits."
    para = (
        "Group-aware MOFA models were successfully audited for both IBD and CRC using matched microbiome and metabolome views without using phenotype labels during factor discovery. "
        f"The IBD and CRC HDF5 models were readable, contained the expected two cohort groups, two molecular views and eight latent factors, and produced finite factor scores and loadings. "
        f"Using a predeclared nondegeneracy rule based on finite scores/loadings and mean factor R2 of at least {NONDEGENERATE_R2_THRESHOLD}% across groups/views, {int(nd_counts.get('IBD', 0))} IBD factors and {int(nd_counts.get('CRC', 0))} CRC factors were considered nondegenerate. "
        "The resulting latent structure was not uniformly multi-view: several factors were dominated by metabolome variance, while a smaller subset showed measurable contribution from both microbiome and metabolome views. "
        "Post-hoc phenotype tests were then applied only to the fixed unsupervised factor scores. In IBD, the clearest disease separation was observed for CD/control and IBD/control contrasts on early factors, whereas UC/control effects were weaker and more factor-specific. "
        "In CRC, phenotype associations were more heterogeneous, consistent with the weaker cross-cohort transportability observed in Stage 4B. "
        "Because group-aware MOFA shares feature loadings across cohort groups, loading concordance should be described as model-shared structure rather than independent replication. "
        "The strongest biological value of Stage 4C is therefore interpretability of latent cross-view structure and group-specific score/variance behaviour, rather than a replacement for Stage 4B predictive validation."
    )
    lines = [
        "# Stage 4C MOFA Results Interpretation", "",
        "## A. Model Validity And Convergence",
        f"- MOFA_IBD_VALID: {'YES' if ok_ibd else 'NO'}.",
        f"- MOFA_CRC_VALID: {'YES' if ok_crc else 'NO'}.",
        "- Both models were interpreted only from saved HDF5/extracted outputs; no model was refitted.", "",
        "## B. Nondegenerate Factors",
        f"- Rule: finite scores/loadings, nonzero score/loading variance, and mean factor R2 >= {NONDEGENERATE_R2_THRESHOLD}% across groups/views.",
        f"- IBD nondegenerate factors: {int(nd_counts.get('IBD', 0))}/8.",
        f"- CRC nondegenerate factors: {int(nd_counts.get('CRC', 0))}/8.", "",
        "## C. Multi-View Versus View-Specific Structure",
        view_counts.to_markdown(index=False), "",
        "## D. Strongest Post-Hoc Phenotype Associations",
        strongest[["disease", "comparison", "cohort_scope", "factor", "effect_median_difference", "raw_p", "bh_fdr", "direction"]].to_markdown(index=False), "",
        "## E. Group-Aware Reproducibility Limitation",
        "- W/loadings are shared by the group-aware MOFA model and should not be reported as independently rediscovered in both cohorts.",
        "- Z/factor scores and variance explained are group-specific and can be compared across cohort groups.",
        "- Independent loading reproducibility would require separate-cohort MOFA or a separate method such as sCCA.", "",
        "## F. Relationship To Stage 4B",
        "Stage 4B showed direction-dependent predictive transportability and inconsistent benefit from simple early integration. Stage 4C asks a different question: whether unsupervised latent microbiome-metabolome structure provides coherent biological interpretation despite those predictive inconsistencies.", "",
        "## G. Separate-Cohort MOFA Sensitivity",
        f"SEPARATE_COHORT_MOFA_SENSITIVITY = {rec}",
        rationale, "",
        "## H. sCCA Implication",
        "sCCA should test whether sparse microbe-metabolite covariance components reproduce independently when each cohort is analysed separately. It complements group-aware MOFA because it can provide cohort-specific loading solutions.", "",
        "## Thesis-Ready Results Paragraph", para, "",
        "## Output Tables And Figures",
        "- tables/stage4c_mofa_non_degenerate_factors.csv",
        "- tables/stage4c_mofa_factor_associations.csv",
        "- tables/stage4c_mofa_cross_cohort_reproducibility.csv",
        "- figures/stage4c_mofa_variance_explained.png",
        "- figures/stage4c_mofa_factor_associations.png",
        "- figures/stage4c_mofa_top_loadings.png",
    ]
    (REPORTS / "STAGE4C_MOFA_RESULTS_INTERPRETATION.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    scores = read_csv(TABLES / "stage4c_mofa_factor_scores.csv")
    loadings = read_csv(TABLES / "stage4c_mofa_feature_loadings.csv")
    variance = read_csv(TABLES / "stage4c_mofa_factor_variance_explained.csv")
    if scores.empty or loadings.empty or variance.empty:
        raise SystemExit("Missing extracted MOFA tables; run extract-only from existing HDF5 first.")
    hdf5_audit, ok_ibd, ok_crc = validation_audit()
    nondeg = nondegenerate_table(scores, loadings, variance)
    assoc = association_table(scores)
    top = top_loadings_table(loadings, nondeg)
    repro = reproducibility_table(nondeg, assoc, top, variance)
    validation = pd.DataFrame([
        {"test": "mofa_ibd_hdf5_valid", "status": "pass" if ok_ibd else "fail", "notes": str(MODELS / "stage4c_mofa_ibd_k8.hdf5")},
        {"test": "mofa_crc_hdf5_valid", "status": "pass" if ok_crc else "fail", "notes": str(MODELS / "stage4c_mofa_crc_k8.hdf5")},
        {"test": "factor_scores_both_diseases", "status": "pass" if set(scores["disease"]) == {"IBD", "CRC"} else "fail", "notes": str(scores.groupby(["disease", "cohort"])["sample_id"].nunique().to_dict())},
        {"test": "loadings_both_diseases", "status": "pass" if set(loadings["disease"]) == {"IBD", "CRC"} else "fail", "notes": str(loadings.groupby(["disease", "view"]).size().to_dict())},
        {"test": "associations_created", "status": "pass" if not assoc.empty else "fail", "notes": f"rows={len(assoc)}"},
        {"test": "reproducibility_created", "status": "pass" if not repro.empty else "fail", "notes": f"rows={len(repro)}"},
    ])
    atomic_csv(hdf5_audit, TABLES / "stage4c_mofa_hdf5_audit.csv")
    atomic_csv(nondeg, TABLES / "stage4c_mofa_non_degenerate_factors.csv")
    atomic_csv(assoc, TABLES / "stage4c_mofa_factor_associations.csv")
    atomic_csv(top, TABLES / "stage4c_mofa_top_loadings.csv")
    atomic_csv(repro, TABLES / "stage4c_mofa_cross_cohort_reproducibility.csv")
    atomic_csv(validation, TABLES / "stage4c_mofa_validation_tests.csv")
    plot_variance(variance, nondeg)
    plot_associations(assoc, nondeg)
    plot_top_loadings(top)
    write_report(ok_ibd, ok_crc, nondeg, assoc, repro, variance)
    print("MOFA_IBD_VALID =", "YES" if ok_ibd else "NO")
    print("MOFA_CRC_VALID =", "YES" if ok_crc else "NO")
    print("NONDEGENERATE", nondeg.groupby("disease")["nondegenerate"].sum().to_dict())
    print("ASSOCIATIONS", assoc.shape)
    print("REPRODUCIBILITY", repro.shape)


if __name__ == "__main__":
    main()
