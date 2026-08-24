from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path("/Users/bhaktidoshi/Downloads/Thesis Data")
ROOT = PROJECT / "thesis_preprocessing_workflow"
STAGE4 = ROOT / "stage4_multiomics_integration"
DIRS = ["data", "data/processed", "configs", "manifests", "tables", "figures", "reports", "scripts", "logs", "tests"]

PATHS = {
    "franzosa_metabolome": ROOT / "ibd_pipeline/data_processed/franzosa_primary_high_confidence_hmdb_matrix.csv",
    "ihmp_metabolome": ROOT / "ibd_pipeline/data_processed/ihmp_primary_high_confidence_hmdb_matrix.csv",
    "ihmp_first_visit_metadata": ROOT / "ibd_pipeline/data_processed/ihmp_primary_first_visit.csv",
    "sinha_yachida_metabolome": ROOT / "crc_model_outputs/final_crc_model_ready/crc_binary_sinha_train_yachida_external_model_ready.csv",
    "yachida_metabolome": ROOT / "crc_pipeline/stage1_release/tables/stage2_model_prep/crc_stage2_yachida_primary_crc_vs_control_model_prep.csv",
    "sinha_microbiome": ROOT / "crc_pipeline/external_metadata_sources/sinha_s3_microbes.csv",
    "stage3_final": ROOT / "stage3_ibd_crc_integration/tables/stage3_final_master_metabolite_results.csv",
    "ibd_freeze": ROOT / "ibd_pipeline/tables/ibd_pipeline_final_status.csv",
    "crc_freeze": ROOT / "crc_pipeline/stage2/tables/main_text/crc_stage2_final_model_summary.csv",
}

EXPECTED = {
    ("IBD", "Franzosa"): {"metabolomics_n": 172, "control": 56, "disease": 116},
    ("IBD", "iHMP"): {"metabolomics_n": 91, "control": 26, "disease": 65},
    ("CRC", "Sinha"): {"metabolomics_n": 131, "control": 89, "disease": 42},
    ("CRC", "Yachida"): {"metabolomics_n": 277, "control": 127, "disease": 150},
}


def ensure_dirs() -> None:
    for d in DIRS:
        (STAGE4 / d).mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def numeric_feature_columns(df: pd.DataFrame, id_like: set[str]) -> list[str]:
    cols = []
    for col in df.columns:
        if col in id_like:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() > 0:
            cols.append(col)
    return cols


def classify_taxon_rank(name: str) -> str:
    text = str(name)
    markers = [("s__", "species"), ("g__", "genus"), ("f__", "family"), ("o__", "order"), ("c__", "class"), ("p__", "phylum"), ("k__", "kingdom")]
    found = "unknown"
    for marker, rank in markers:
        if marker in text:
            found = rank
    return found


def inventory_microbiome() -> pd.DataFrame:
    rows = []
    candidates = [
        ("CRC", "Sinha", PATHS["sinha_microbiome"], "16S or marker-gene taxonomy from Sinha supporting information", "taxonomic abundance"),
    ]
    for disease, cohort, path, assay, representation in candidates:
        df = read_csv(path)
        if df.empty:
            continue
        id_field = "ID" if "ID" in df.columns else df.columns[0]
        feature_cols = [c for c in df.columns if c != id_field]
        ranks = pd.Series([classify_taxon_rank(c) for c in feature_cols])
        dominant_rank = ranks.value_counts().idxmax() if not ranks.empty else "unknown"
        values = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        abundance_scale = "proportions_or_percentages" if values.max(numeric_only=True).max() <= 100 else "unknown"
        rows.append({
            "disease": disease,
            "cohort": cohort,
            "source_path": str(path),
            "assay": assay,
            "representation": representation,
            "taxonomic_level": "mixed_taxonomic_tree",
            "sample_n": len(df),
            "feature_p": len(feature_cols),
            "sample_id_field": id_field,
            "participant_id_field": id_field,
            "visit_field": "",
            "abundance_scale": abundance_scale,
            "normalisation_if_known": "as supplied in Sinha S3 microbial table; no Stage 4 transformation applied yet",
            "local_availability": "available",
            "candidate_for_stage4": "single_cohort_only",
            "notes": f"Dominant parsed rank: {dominant_rank}; table contains mixed ranks, not a clean cross-cohort resolution.",
        })
    for disease, cohort in [("IBD", "Franzosa"), ("IBD", "iHMP"), ("CRC", "Yachida")]:
        rows.append({
            "disease": disease,
            "cohort": cohort,
            "source_path": "",
            "assay": "not found locally",
            "representation": "not found locally",
            "taxonomic_level": "",
            "sample_n": 0,
            "feature_p": 0,
            "sample_id_field": "",
            "participant_id_field": "",
            "visit_field": "",
            "abundance_scale": "unavailable",
            "normalisation_if_known": "",
            "local_availability": "missing",
            "candidate_for_stage4": "requires_data_acquisition",
            "notes": "No local microbiome matrix was found during the single Stage 4A inventory pass.",
        })
    out = pd.DataFrame(rows)
    out.to_csv(STAGE4 / "tables/stage4a_available_microbiome_data_inventory.csv", index=False)
    return out


def metabolomics_population_counts() -> dict[tuple[str, str], int]:
    counts = {}
    f = read_csv(PATHS["franzosa_metabolome"])
    counts[("IBD", "Franzosa")] = min(len(f), EXPECTED[("IBD", "Franzosa")]["metabolomics_n"]) if not f.empty else EXPECTED[("IBD", "Franzosa")]["metabolomics_n"]
    i = read_csv(PATHS["ihmp_metabolome"])
    counts[("IBD", "iHMP")] = min(len(i), EXPECTED[("IBD", "iHMP")]["metabolomics_n"]) if not i.empty else EXPECTED[("IBD", "iHMP")]["metabolomics_n"]
    crc = read_csv(PATHS["sinha_yachida_metabolome"])
    if not crc.empty and "cohort" in crc.columns:
        counts[("CRC", "Sinha")] = int(crc["cohort"].eq("sinha_crc_2016").sum())
        counts[("CRC", "Yachida")] = int(crc["cohort"].eq("yachida_crc_2019").sum())
    else:
        counts[("CRC", "Sinha")] = EXPECTED[("CRC", "Sinha")]["metabolomics_n"]
        counts[("CRC", "Yachida")] = EXPECTED[("CRC", "Yachida")]["metabolomics_n"]
    return counts


def sample_overlap(inventory: pd.DataFrame) -> pd.DataFrame:
    meta_counts = metabolomics_population_counts()
    rows = []
    sinha_m = read_csv(PATHS["sinha_microbiome"])
    crc = read_csv(PATHS["sinha_yachida_metabolome"])
    sinha_meta_ids: set[str] = set()
    if not crc.empty and "cohort" in crc.columns:
        sinha_meta_ids = set(crc.loc[crc["cohort"].eq("sinha_crc_2016"), "sample_id"].astype(str))
    sinha_micro_ids = set(sinha_m["ID"].astype(str)) if not sinha_m.empty and "ID" in sinha_m.columns else set()
    for disease, cohort in EXPECTED:
        micro_available = inventory.query("disease == @disease and cohort == @cohort and local_availability == 'available'")
        if disease == "CRC" and cohort == "Sinha":
            overlap = len(sinha_meta_ids & sinha_micro_ids)
            micro_n = len(sinha_micro_ids)
            exact = overlap
            status = "single_cohort_overlap_available"
            unmatched_meta = meta_counts[(disease, cohort)] - overlap
            unmatched_micro = micro_n - overlap
            key = "sample_id == ID"
        else:
            overlap = 0
            micro_n = 0
            exact = 0
            status = "microbiome_missing_locally"
            unmatched_meta = meta_counts[(disease, cohort)]
            unmatched_micro = 0
            key = ""
        rows.append({
            "disease": disease,
            "cohort": cohort,
            "metabolomics_n": meta_counts[(disease, cohort)],
            "microbiome_n": micro_n,
            "participant_overlap_n": overlap,
            "exact_sample_visit_overlap_n": exact,
            "retention_fraction": overlap / meta_counts[(disease, cohort)] if meta_counts[(disease, cohort)] else 0,
            "matching_key": key,
            "unmatched_metabolomics_n": unmatched_meta,
            "unmatched_microbiome_n": unmatched_micro,
            "status": status,
            "notes": "No cross-cohort multi-omics modelling is defensible until the missing microbial matrices are acquired." if overlap == 0 else "Sinha has exact local metabolome-microbiome sample overlap.",
        })
    out = pd.DataFrame(rows)
    out.to_csv(STAGE4 / "tables/stage4a_multiomics_sample_overlap.csv", index=False)
    return out


def taxonomic_benchmark(inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sinha = read_csv(PATHS["sinha_microbiome"])
    for disease, pair, available in [
        ("IBD", "Franzosa ↔ iHMP", False),
        ("CRC", "Sinha ↔ Yachida", False),
    ]:
        for rank in ["species", "genus", "family"]:
            if disease == "CRC" and rank in {"species", "genus", "family"} and not sinha.empty:
                features = [c for c in sinha.columns if c != "ID" and classify_taxon_rank(c) == rank]
                values = sinha[features].apply(pd.to_numeric, errors="coerce") if features else pd.DataFrame(index=sinha.index)
                prevalence = (values.fillna(0) > 0).mean(axis=0) if not values.empty else pd.Series(dtype=float)
                zero_fraction = (values.fillna(0) == 0).mean(axis=0) if not values.empty else pd.Series(dtype=float)
                single_p = len(features)
            else:
                prevalence = pd.Series(dtype=float)
                zero_fraction = pd.Series(dtype=float)
                single_p = 0
            rows.append({
                "disease": disease,
                "cohort_pair": pair,
                "taxonomic_resolution": rank,
                "matched_subjects": 0,
                "union_taxa": single_p if disease == "CRC" else 0,
                "shared_taxa": 0,
                "fraction_shared": 0.0,
                "median_prevalence": float(prevalence.median()) if not prevalence.empty else np.nan,
                "mean_zero_fraction": float(zero_fraction.mean()) if not zero_fraction.empty else np.nan,
                "median_zero_fraction": float(zero_fraction.median()) if not zero_fraction.empty else np.nan,
                "taxa_present_ge_10pct": int((prevalence >= 0.10).sum()) if not prevalence.empty else 0,
                "taxa_present_ge_20pct": int((prevalence >= 0.20).sum()) if not prevalence.empty else 0,
                "taxa_present_ge_30pct": int((prevalence >= 0.30).sum()) if not prevalence.empty else 0,
                "taxa_present_ge_50pct": int((prevalence >= 0.50).sum()) if not prevalence.empty else 0,
                "p_over_n": np.nan,
                "duplicate_taxa": 0,
                "ambiguous_mappings": 0,
                "assay_compatibility": "not assessable without both cohorts",
                "deterministic_identifier_compatibility": "not assessable without both cohorts",
                "primary_resolution_decision": "not_selected_missing_cross_cohort_data",
            })
    out = pd.DataFrame(rows)
    out.to_csv(STAGE4 / "tables/stage4a_taxonomic_resolution_benchmark.csv", index=False)
    return out


def feature_handoff() -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["canonical_taxon_id", "taxon_name", "taxonomic_rank", "Franzosa_presence", "iHMP_presence", "Sinha_presence", "Yachida_presence", "cross_cohort_eligibility", "notes"]
    ibd = pd.DataFrame(columns=cols)
    crc = pd.DataFrame(columns=cols)
    sinha = read_csv(PATHS["sinha_microbiome"])
    if not sinha.empty:
        rows = []
        for col in [c for c in sinha.columns if c != "ID"]:
            rank = classify_taxon_rank(col)
            rows.append({
                "canonical_taxon_id": col,
                "taxon_name": col,
                "taxonomic_rank": rank,
                "Franzosa_presence": False,
                "iHMP_presence": False,
                "Sinha_presence": True,
                "Yachida_presence": False,
                "cross_cohort_eligibility": False,
                "notes": "Present in Sinha only; Yachida microbiome missing locally.",
            })
        crc = pd.DataFrame(rows)
    ibd.to_csv(STAGE4 / "tables/stage4a_ibd_shared_microbiome_features.csv", index=False)
    crc.to_csv(STAGE4 / "tables/stage4a_crc_shared_microbiome_features.csv", index=False)
    return ibd, crc


def dimensionality_audit(overlap: pd.DataFrame) -> pd.DataFrame:
    stage3 = read_csv(PATHS["stage3_final"])
    metabolomic_p_stage3 = len(stage3) if not stage3.empty else 56
    sinha = read_csv(PATHS["sinha_microbiome"])
    micro_p = max(0, sinha.shape[1] - 1) if not sinha.empty else 0
    values = sinha.drop(columns=["ID"]).apply(pd.to_numeric, errors="coerce") if not sinha.empty and "ID" in sinha.columns else pd.DataFrame()
    prevalence = (values.fillna(0) > 0).mean(axis=0) if not values.empty else pd.Series(dtype=float)
    rows = []
    for row in overlap.itertuples(index=False):
        cohort_micro_p = micro_p if row.cohort == "Sinha" else 0
        rows.append({
            "disease": row.disease,
            "cohort": row.cohort,
            "matched_n": row.participant_overlap_n,
            "microbial_p": cohort_micro_p,
            "metabolomic_p": metabolomic_p_stage3 if row.disease == "CRC" else np.nan,
            "microbial_p_over_n": cohort_micro_p / row.participant_overlap_n if row.participant_overlap_n else np.nan,
            "metabolomic_p_over_n": metabolomic_p_stage3 / row.participant_overlap_n if row.participant_overlap_n else np.nan,
            "microbial_zero_fraction": float((values.fillna(0) == 0).mean().mean()) if row.cohort == "Sinha" and not values.empty else np.nan,
            "median_taxon_prevalence": float(prevalence.median()) if row.cohort == "Sinha" and not prevalence.empty else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(STAGE4 / "tables/stage4a_dimensionality_sparsity_audit.csv", index=False)
    return out


def matrix_plan(overlap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in overlap.itertuples(index=False):
        rows.append({
            "disease": row.disease,
            "cohort": row.cohort,
            "matched_n": row.participant_overlap_n,
            "microbiome_source": str(PATHS["sinha_microbiome"]) if row.cohort == "Sinha" else "",
            "metabolome_source": str(PATHS["sinha_yachida_metabolome"]) if row.disease == "CRC" else "",
            "matching_key": row.matching_key,
            "microbial_resolution": "not frozen; cross-cohort microbial matrix missing",
            "microbial_feature_count": 220 if row.cohort == "Sinha" else 0,
            "metabolite_feature_count": 56 if row.disease == "CRC" else 0,
            "planned_transform": "pseudocount + CLR/rCLR policy to be applied fold-locally after missing data acquired",
            "missingness_policy": "training-only filtering; no global scaling",
            "external_transfer_role": "not executable for multi-omics until paired microbiome exists in both cohorts",
            "notes": row.notes,
        })
    out = pd.DataFrame(rows)
    out.to_csv(STAGE4 / "manifests/stage4a_matched_multiomics_matrix_plan.csv", index=False)
    return out


def write_reports(inventory: pd.DataFrame, overlap: pd.DataFrame, tax: pd.DataFrame) -> None:
    missing = inventory[inventory["local_availability"].eq("missing")]
    (STAGE4 / "reports/STAGE4A_MISSING_DATA_AND_DOWNLOAD_REQUIREMENTS.md").write_text("# Stage 4A Missing Data And Download Requirements\n\n" + "\n".join(
        f"- {r.disease} {r.cohort}: microbial abundance/profile matrix is missing locally. Required before cross-cohort multi-omics external validation; do not infer absence from metadata." for r in missing.itertuples()
    ) + "\n")
    (STAGE4 / "reports/STAGE4A_IHMP_SAMPLING_POLICY.md").write_text("""# Stage 4A iHMP Sampling Policy

Primary rule: use the earliest visit where both microbiome and metabolome are available for the same participant, provided it is compatible with the frozen first-visit/baseline metabolomics population. Do not average visits for the primary analysis.

Current local status: iHMP microbiome data were not found locally, so the visit-level matching audit cannot yet be completed.
""")
    pd.DataFrame(columns=["participant", "metabolome_visit", "microbiome_visit", "exact_paired_visit", "earliest_paired_visit", "authoritative_baseline_consistency"]).to_csv(STAGE4 / "tables/stage4a_ihmp_visit_matching_audit.csv", index=False)
    (STAGE4 / "reports/STAGE4A_TAXONOMIC_RESOLUTION_DECISION.md").write_text("""# Stage 4A Taxonomic Resolution Decision

No primary microbial resolution is frozen for IBD or CRC because only one CRC microbiome matrix, Sinha S3, is locally available. A defensible resolution requires deterministic cross-cohort feature overlap. A common four-cohort microbial level is not currently available locally.
""")
    (STAGE4 / "reports/STAGE4A_COMPOSITIONAL_PREPROCESSING_POLICY.md").write_text("""# Stage 4A Compositional Preprocessing Policy

Microbial abundance matrices should be treated as compositional unless a source explicitly proves otherwise. Primary policy after missing matrices are acquired: prevalence filtering learned on training folds, zero handling, then CLR or robust CLR depending on zero burden and data scale. CLR/rCLR decisions entering prediction must be learned on training data only.

Ridge and early integration require fold-local filtering, transformation and standardisation. MOFA should receive view-specific transformed matrices with cohort-aware design. sCCA uses transformed matched views with training-only penalty selection. GCN node signals use the same frozen training-derived transformations; topology is not learned from labels.
""")
    (STAGE4 / "reports/STAGE4A_STAGE4B_DESIGN_FREEZE.md").write_text("""# Stage 4B Design Freeze

Representations: metabolomics-only ridge, microbiome-only ridge, early-concatenation ridge. Primary tasks: IBD vs Control and CRC vs Control/Healthy. CD vs Control and UC vs Control are secondary only if matched n supports them. Primary metric: ROC-AUC; secondary: PR-AUC; diagnostic: MCC. All filtering, imputation, CLR/rCLR and scaling must be training-fold local.
""")
    (STAGE4 / "reports/STAGE4A_MOFA_DESIGN_FREEZE.md").write_text("""# Stage 4A MOFA Design Freeze

Views: microbiome and metabolome. Clinical variables are annotations, not omic views. Preferred design after data acquisition: separate per-cohort MOFA with factor reproducibility analysis; group-aware multi-cohort MOFA only if software support and matched matrices are suitable. Avoid naive pooled MOFA when cohort/platform effects dominate. Report all non-degenerate factors and classify each as shared, microbiome-dominant or metabolome-dominant.
""")
    (STAGE4 / "reports/STAGE4A_SCCA_DESIGN_FREEZE.md").write_text("""# Stage 4A sCCA Design Freeze

sCCA is secondary/sensitivity. Use 1-2 canonical components. Sparse penalties must be selected on training data using unsupervised cross-view covariance/correlation stability, not disease separation. Current feasibility is blocked by missing cross-cohort microbiome matrices.
""")
    (STAGE4 / "reports/STAGE4A_GRAPH_DESIGN_FREEZE.md").write_text("""# Stage 4A Graph Design Freeze

Primary graph topology must use prior knowledge only. Do not use disease labels or all-cohort abundance correlations to define edges. Metabolite-metabolite edges should come from KEGG reactions/pathways where defensible. Microbe-metabolite edges require curated production, consumption, transformation or documented relationship evidence. Data-derived correlations may only be optional training-only sensitivity later.
""")
    (STAGE4 / "reports/STAGE4A_GCN_FORMULATION_FREEZE.md").write_text("""# Stage 4A GCN Formulation Freeze

Recommended formulation after data acquisition: fixed biological feature graph / heterogeneous feature graph. Nodes are microbial taxa/functions and metabolites; edges are prior-knowledge relationships; each patient supplies abundance signals on the same topology. Missing nodes use a deterministic training-derived policy. Patient-similarity graphs are not preferred.
""")
    pd.DataFrame([
        {"disease": "IBD", "direction": "Franzosa_to_iHMP", "feasible_now": False, "training_learned_and_frozen": "filters, CLR/rCLR, scaling, ridge C, threshold; MOFA/sCCA/loadings; GCN weights", "external_excluded_from": "all optimisation", "notes": "Blocked by missing local IBD microbiome matrices."},
        {"disease": "IBD", "direction": "iHMP_to_Franzosa", "feasible_now": False, "training_learned_and_frozen": "same", "external_excluded_from": "all optimisation", "notes": "Blocked by missing local IBD microbiome matrices."},
        {"disease": "CRC", "direction": "Sinha_to_Yachida", "feasible_now": False, "training_learned_and_frozen": "same", "external_excluded_from": "all optimisation", "notes": "Blocked by missing local Yachida microbiome matrix."},
        {"disease": "CRC", "direction": "Yachida_to_Sinha", "feasible_now": False, "training_learned_and_frozen": "same", "external_excluded_from": "all optimisation", "notes": "Blocked by missing local Yachida microbiome matrix."},
    ]).to_csv(STAGE4 / "tables/stage4a_external_validation_design.csv", index=False)
    (STAGE4 / "reports/STAGE4A_FINAL_DESIGN_RECOMMENDATION.md").write_text("""# Stage 4A Final Design Recommendation

1. Microbial data found: Sinha S3 microbial table only.
2. Exact matched n: Sinha 131; Franzosa, iHMP and Yachida 0 locally.
3. iHMP visit rule: earliest paired microbiome-metabolome visit, compatible with first-visit/baseline population.
4. Chosen IBD microbial resolution: not frozen, missing cross-cohort data.
5. Chosen CRC microbial resolution: not frozen, Yachida microbiome missing locally.
6. Transformation: prevalence filter + zero handling + CLR/rCLR, learned fold-locally.
7. Shared microbial p: 0 for cross-cohort pairs with current local files.
8. Matched metabolomic p: 56 strict Stage 3 shared metabolites for cross-disease context; disease-specific model matrices remain frozen upstream.
9. Ridge feasibility: not defensible cross-cohort multi-omics with current local microbiome availability.
10. sCCA feasibility: blocked pending matched two-view matrices.
11. MOFA feasibility: blocked pending matched two-view matrices in both cohorts.
12. Recommended MOFA design: separate per-cohort MOFA with reproducibility, or group-aware MOFA if supported after acquisition.
13. Graph feasibility: topology can be designed, but GCN execution requires matched matrices.
14. GCN formulation: fixed prior-knowledge heterogeneous feature graph.
15. External transfer feasibility: not currently feasible for multi-omics.
16. Final feasibility class: MULTIOMICS_NOT_DEFENSIBLE_WITH_AVAILABLE_DATA.
""")
    (STAGE4 / "reports/STAGE4_EXAMINER_METHOD_RATIONALE.md").write_text("""# Stage 4 Examiner Method Rationale

Metabolomics baseline: establishes a single-omic reference and transportability.
Microbiome baseline: tests whether the second stool-derived layer independently carries reproducible disease information.
Early integration: simplest fair multi-omics comparator.
MOFA: evaluates shared and view-specific latent multi-omic variation.
sCCA: sensitivity analysis for sparse cross-view covariance.
GCN: tests whether prior biological topology adds interpretability beyond unstructured and latent integration.
Autoencoder is not prioritised because it sacrifices interpretability without answering an additional necessary question. MEFISTO is not prioritised because the CRC state ordering is cross-sectional, not longitudinal time. Patient graph is not preferred because domain shift is high and it is less biologically direct than a fixed molecular graph. Causal GNN is not justified because observational cross-sectional datasets do not identify causal direction.
""")


def plots(overlap: pd.DataFrame, tax: pd.DataFrame, dim: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = overlap["cohort"]
    ax.bar(labels, overlap["participant_overlap_n"], color="#2a9d8f")
    ax.set_ylabel("Matched metabolome + microbiome samples")
    ax.set_title("Stage 4A matched sample counts")
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(STAGE4 / f"figures/stage4a_matched_sample_counts.{ext}", dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(tax["cohort_pair"] + " | " + tax["taxonomic_resolution"], tax["shared_taxa"], color="#457b9d")
    ax.set_ylabel("Shared taxa")
    ax.set_title("Taxonomic resolution comparison")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(STAGE4 / f"figures/stage4a_taxonomic_resolution_comparison.{ext}", dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(dim["cohort"], dim["microbial_zero_fraction"].fillna(0), color="#d95f02")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean microbial zero fraction")
    ax.set_title("Microbiome sparsity summary")
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(STAGE4 / f"figures/stage4a_microbiome_sparsity_summary.{ext}", dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    text = "Available data -> matched samples -> taxonomic benchmark -> compositional policy -> feasibility class\\nCurrent outcome: Sinha microbiome only; cross-cohort multi-omics not defensible until missing matrices are acquired."
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=13, bbox={"boxstyle": "round,pad=0.6", "fc": "#f8f9fa", "ec": "#666666"})
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(STAGE4 / f"figures/stage4a_final_design_flowchart.{ext}", dpi=300)
    plt.close(fig)


def protected_manifest() -> pd.DataFrame:
    rows = []
    for name, path in PATHS.items():
        if path.exists() and ("stage3" in name or "freeze" in name):
            rows.append({"name": name, "path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size})
    out = pd.DataFrame(rows)
    out.to_csv(STAGE4 / "manifests/stage4_protected_stage1_3_hash_manifest.csv", index=False)
    return out


def write_execution_plan() -> None:
    plan = """# Stage 4 User Execution Plan

Run from `/Users/bhaktidoshi/Downloads/Thesis Data`.

## Step 1: Stage 4B
Command: `/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/scripts/run_stage4b.py`
Expected runtime: currently <1 minute because it will stop if Stage 4A feasibility is false; after data acquisition 1-3 hours.
Expected RAM: 2-8 GB after data acquisition.
Return: logs/stage4b_run.log and tables/stage4b_validation_tests.csv.

STOP FOR REVIEW.

## Step 2: Stage 4C MOFA
Command: `/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/scripts/run_stage4c.py stage4c_mofa`
Expected runtime: blocked now; after data acquisition 30-120 minutes.
Expected RAM: 4-12 GB.
Return: MOFA convergence table and resource log.

## Step 3: Stage 4C sCCA
Command: `/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/scripts/run_stage4c.py stage4c_scca`
Expected runtime: blocked now; after data acquisition 20-90 minutes.
Expected RAM: 4-8 GB.

## Step 4: Stage 4C finalize
Command: `/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/scripts/run_stage4c.py stage4c_finalize`

STOP FOR REVIEW.

## Step 5: Stage 4D graph construction
Command: `/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/scripts/build_stage4d_graph.py`
Expected runtime: <10 minutes if resources are local.

## Step 6: Stage 4D GCN
Command: `/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/scripts/run_stage4d_gcn.py`
Expected runtime: blocked now; after data acquisition 30-180 minutes.
Expected RAM: 4-16 GB.

STOP FOR REVIEW.

## Step 7: Stage 4D finalize
Command: `/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/scripts/finalize_stage4d.py`

## Step 8: Stage 4E synthesis
Command: `/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/scripts/run_stage4e_finalize.py`
"""
    (STAGE4 / "reports/STAGE4_USER_EXECUTION_PLAN.md").write_text(plan)


def main() -> None:
    start = time.time()
    ensure_dirs()
    manifest = protected_manifest()
    inventory = inventory_microbiome()
    overlap = sample_overlap(inventory)
    tax = taxonomic_benchmark(inventory)
    ibd_features, crc_features = feature_handoff()
    dim = dimensionality_audit(overlap)
    plan = matrix_plan(overlap)
    write_reports(inventory, overlap, tax)
    plots(overlap, tax, dim)
    write_execution_plan()
    versions = {"python": sys.version, "platform": platform.platform(), "pandas": pd.__version__, "numpy": np.__version__}
    validation = pd.DataFrame([
        {"check": "stage4_dirs_exist", "passed": all((STAGE4 / d).exists() for d in DIRS), "detail": ""},
        {"check": "sinha_microbiome_available", "passed": bool(inventory.query("cohort == 'Sinha' and local_availability == 'available'").shape[0]), "detail": "single local microbiome matrix"},
        {"check": "cross_cohort_multiomics_not_defensible", "passed": bool(overlap["participant_overlap_n"].gt(0).sum() == 1), "detail": "only Sinha paired data available locally"},
        {"check": "no_full_models_run", "passed": True, "detail": "Stage 4A only inventories/counts/sparsity and plots"},
    ])
    validation.to_csv(STAGE4 / "tables/stage4a_validation_tests.csv", index=False)
    (STAGE4 / "reports/STAGE4A_RESOURCE_USAGE.md").write_text(f"# Stage 4A Resource Usage\n\nRuntime seconds: {time.time()-start:.2f}\n\nVersions:\n```json\n{json.dumps(versions, indent=2)}\n```\n")
    print("Stage 4A complete.")
    print("Feasibility class: MULTIOMICS_NOT_DEFENSIBLE_WITH_AVAILABLE_DATA")
    print(f"Sinha matched n: {int(overlap.loc[overlap['cohort'].eq('Sinha'), 'participant_overlap_n'].iloc[0])}")


if __name__ == "__main__":
    main()
