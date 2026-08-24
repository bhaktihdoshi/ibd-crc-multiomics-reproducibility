from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage4_corrected_v2_common import PROJECT, REPORTS, ROOT, TABLES, atomic_csv, ensure_v2_dirs


IBD_RUNNER = ROOT / "ibd_pipeline" / "run_pipeline.py"
CRC_RUNNER = ROOT / "crc_pipeline" / "run_stage2_modelling.py"
STAGE4_COMMON = ROOT / "stage4_multiomics_integration" / "corrected_v2" / "scripts" / "stage4_corrected_v2_common.py"


COHORTS = {
    "Franzosa": {
        "disease": "IBD",
        "final_matrix": ROOT / "ibd_pipeline" / "data_processed" / "franzosa_primary_high_confidence_hmdb_matrix.csv",
        "source_file": PROJECT / "IBD-DataSet" / "FRANZOSA_IBD_2019" / "downloaded_full_metabolomics" / "franzosa_original_samples_original_mapped_metabolites.repaired.tsv",
        "source_map": PROJECT / "IBD-DataSet" / "FRANZOSA_IBD_2019" / "downloaded_full_metabolomics" / "franzosa_original_mapped_metabolites.repaired.map.tsv",
        "assay_platform": "LC-MS feature names/mapping indicate HILIC and C18, positive/negative ionisation where present",
        "original_measurement_type": "as-supplied metabolomics feature intensities/abundances; exact upstream unit not documented in local pipeline",
        "raw_or_processed_status": "processed/repaired matrix from supplied/downloaded Franzosa files; not raw instrument files",
        "verified_log_transformation": "none applied by thesis pipeline before Stage 4 matrix",
        "verified_normalisation": "none applied by thesis pipeline before Stage 4 matrix",
        "verified_centering": "none applied by thesis pipeline before Stage 4 matrix",
        "verified_scaling": "none applied by thesis pipeline before Stage 4 matrix",
        "verified_batch_correction": "none applied by thesis pipeline before Stage 4 matrix",
        "missing_value_encoding": "pandas numeric coercion; source missing values retained as NA/NaN",
        "zero_meaning": "0 not globally recoded as missing in the Stage 4 matrix; biological/technical zero meaning not verified",
        "feature_aggregation": "one selected source feature per shared high-confidence HMDB identifier; no averaging",
        "hmdb_mapping": "supplied repaired map HMDB IDs; Tier 1 high-confidence HMDB matches selected",
        "units_where_reported": "not reported in local pipeline artifacts",
        "writer_function_or_block": "ibd_pipeline/run_pipeline.py: master high-confidence HMDB selection then pd.concat source feature series; to_csv(PROCESSED/franzosa_primary_high_confidence_hmdb_matrix.csv)",
        "evidence_files": "ibd_pipeline/run_pipeline.py; ibd_pipeline/reports/supplied_data_scope.md; franzosa_original_mapped_metabolites.repaired.map.tsv",
    },
    "iHMP": {
        "disease": "IBD",
        "final_matrix": ROOT / "ibd_pipeline" / "data_processed" / "ihmp_primary_high_confidence_hmdb_matrix.csv",
        "source_file": PROJECT / "IBD-DataSet" / "iHMP_IBDMDB" / "mtb.tsv",
        "source_map": PROJECT / "IBD-DataSet" / "iHMP_IBDMDB" / "mtb.map.tsv",
        "assay_platform": "LC-MS map Method includes HILIC-pos/neg and C18-pos/neg style methods",
        "original_measurement_type": "as-supplied metabolomics feature intensities/abundances; exact upstream unit not documented in local pipeline",
        "raw_or_processed_status": "processed matrix from supplied iHMP/IBDMDB mtb.tsv; not raw instrument files",
        "verified_log_transformation": "none applied by thesis pipeline before Stage 4 matrix",
        "verified_normalisation": "none applied by thesis pipeline before Stage 4 matrix",
        "verified_centering": "none applied by thesis pipeline before Stage 4 matrix",
        "verified_scaling": "none applied by thesis pipeline before Stage 4 matrix",
        "verified_batch_correction": "none applied by thesis pipeline before Stage 4 matrix",
        "missing_value_encoding": "pandas numeric coercion; source missing values retained as NA/NaN",
        "zero_meaning": "0 not globally recoded as missing in the Stage 4 matrix; biological/technical zero meaning not verified",
        "feature_aggregation": "one selected source feature per shared high-confidence HMDB identifier; no averaging",
        "hmdb_mapping": "supplied mtb.map.tsv HMDB IDs; Tier 1 high-confidence HMDB matches selected",
        "units_where_reported": "not reported in local pipeline artifacts",
        "writer_function_or_block": "ibd_pipeline/run_pipeline.py: master high-confidence HMDB selection then pd.concat source feature series; to_csv(PROCESSED/ihmp_primary_high_confidence_hmdb_matrix.csv)",
        "evidence_files": "ibd_pipeline/run_pipeline.py; ibd_pipeline/reports/supplied_data_scope.md; iHMP_IBDMDB/mtb.map.tsv",
    },
    "Sinha": {
        "disease": "CRC",
        "final_matrix": ROOT / "crc_pipeline" / "stage2" / "data_processed" / "crc_stage2_primary_metabolomics_canonical.csv",
        "source_file": PROJECT / "Colon_Cancer_Data_Set" / "Sinha-et-al" / "mtb.tsv",
        "source_map": ROOT / "crc_pipeline" / "stage2" / "tables" / "crc_stage2_final_shared_model_feature_space.csv",
        "assay_platform": "fecal metabolomics; exact platform not re-derived by Stage 2 code; source paper/local metadata should be cited for assay details",
        "original_measurement_type": "as-supplied aligned metabolomics values; numerical range suggests processed/log-like values but this is inference, not a verified transformation",
        "raw_or_processed_status": "processed curated table from local Sinha mtb.tsv; not raw instrument files",
        "verified_log_transformation": "none applied by thesis Stage 2 primary pipeline; config says values appear transformed/standardized",
        "verified_normalisation": "none applied by thesis Stage 2 primary matrix writer",
        "verified_centering": "none applied by thesis Stage 2 primary matrix writer",
        "verified_scaling": "none applied by thesis Stage 2 primary matrix writer before Stage 4",
        "verified_batch_correction": "none applied by thesis Stage 2 primary matrix writer",
        "missing_value_encoding": "clean_source_missing handles blanks/-/NA for clinical values; metabolomics are numeric-coerced and missing retained as NaN",
        "zero_meaning": "0 not globally recoded as missing in the canonical matrix",
        "feature_aggregation": "one deterministic selected assay feature per canonical metabolite; no averaging",
        "hmdb_mapping": "CRC Stage 2 canonical priority: valid HMDB first; KEGG fallback only when HMDB absent; Stage 4 uses HMDB canonical columns",
        "units_where_reported": "not reported in local Stage 2 matrix artifacts",
        "writer_function_or_block": "crc_pipeline/run_stage2_modelling.py: load_sinha_primary -> canonical_matrix -> add met__ prefix -> concat into crc_stage2_primary_metabolomics_canonical.csv",
        "evidence_files": "crc_pipeline/run_stage2_modelling.py; crc_pipeline/CRC_STAGE2_TRANSFORMATION_POLICY.md; crc_pipeline/reports/crc_stage1_data_audit.md",
    },
    "Yachida": {
        "disease": "CRC",
        "final_matrix": ROOT / "crc_pipeline" / "stage2" / "data_processed" / "crc_stage2_primary_metabolomics_canonical.csv",
        "source_file": ROOT / "crc_pipeline" / "tables" / "stage2_model_prep" / "crc_stage2_yachida_primary_crc_vs_control_model_prep.csv",
        "source_map": ROOT / "crc_pipeline" / "stage2" / "tables" / "crc_stage2_final_shared_model_feature_space.csv",
        "assay_platform": "fecal metabolomics; source feature names are KEGG-style metabolite labels in Stage 2 prep table",
        "original_measurement_type": "as-supplied metabolomics abundance/intensity values; exact upstream unit not documented in local Stage 2 writer",
        "raw_or_processed_status": "processed Stage 1/Stage 2 model-prep table; not raw instrument files",
        "verified_log_transformation": "none applied by thesis Stage 2 primary pipeline; Stage 3B policy also states no added log transform",
        "verified_normalisation": "none applied by thesis Stage 2 primary matrix writer",
        "verified_centering": "none applied by thesis Stage 2 primary matrix writer",
        "verified_scaling": "none applied by thesis Stage 2 primary matrix writer before Stage 4",
        "verified_batch_correction": "none applied by thesis Stage 2 primary matrix writer",
        "missing_value_encoding": "numeric coercion; source missing values retained as NaN",
        "zero_meaning": "0 retained as numeric value in primary matrix; no global zero-as-missing rule",
        "feature_aggregation": "one deterministic selected assay feature per canonical metabolite; no averaging",
        "hmdb_mapping": "CRC Stage 2 canonical priority: valid HMDB first; KEGG fallback only when HMDB absent; Stage 4 uses HMDB canonical columns",
        "units_where_reported": "not reported in local Stage 2 matrix artifacts",
        "writer_function_or_block": "crc_pipeline/run_stage2_modelling.py: load_yachida_primary -> canonical_matrix -> add met__ prefix -> concat into crc_stage2_primary_metabolomics_canonical.csv",
        "evidence_files": "crc_pipeline/run_stage2_modelling.py; crc_pipeline/CRC_STAGE2_TRANSFORMATION_POLICY.md; stage3_ibd_crc_integration/reports/STAGE3B_YACHIDA_TRANSFORMATION_POLICY.md",
    },
}


def numeric_summary(cohort: str, info: dict[str, Any]) -> dict[str, Any]:
    path = Path(info["final_matrix"])
    df = pd.read_csv(path, low_memory=False)
    if cohort in {"Sinha", "Yachida"}:
        df = df[df["cohort"].astype(str).eq(cohort)].copy()
        features = [c for c in df.columns if str(c).startswith("met__")]
    else:
        features = [c for c in df.columns if c != "sample_id"]
    x = df[features].apply(pd.to_numeric, errors="coerce")
    values = x.to_numpy(dtype=float).ravel()
    finite = values[np.isfinite(values)]
    return {
        "final_sample_n": int(len(df)),
        "final_feature_n": int(len(features)),
        "final_value_min": float(np.min(finite)) if len(finite) else np.nan,
        "final_value_q25": float(np.quantile(finite, 0.25)) if len(finite) else np.nan,
        "final_value_median": float(np.quantile(finite, 0.50)) if len(finite) else np.nan,
        "final_value_q75": float(np.quantile(finite, 0.75)) if len(finite) else np.nan,
        "final_value_max": float(np.max(finite)) if len(finite) else np.nan,
        "final_missing_fraction": float(np.mean(~np.isfinite(values))) if len(values) else np.nan,
        "final_zero_fraction_observed": float(np.mean(finite == 0)) if len(finite) else np.nan,
        "negative_value_present": bool(np.any(finite < 0)) if len(finite) else False,
        "nonnegative_value_present": bool(np.any(finite >= 0)) if len(finite) else False,
    }


def source_shape(path: Path) -> str:
    if not path.exists():
        return "missing"
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    try:
        df = pd.read_csv(path, sep=sep, nrows=5, low_memory=False)
        return f"exists; columns preview={list(df.columns[:8])}"
    except Exception as exc:
        return f"exists; preview failed: {exc!r}"


def build_audit_table() -> pd.DataFrame:
    rows = []
    for cohort, info in COHORTS.items():
        row = {
            "cohort": cohort,
            "disease": info["disease"],
            "source_file": str(info["source_file"]),
            "source_file_status": source_shape(Path(info["source_file"])),
            "mapping_or_feature_space_file": str(info["source_map"]),
            "final_stage4_matrix": str(info["final_matrix"]),
            "assay_platform": info["assay_platform"],
            "original_measurement_type": info["original_measurement_type"],
            "raw_or_processed_status": info["raw_or_processed_status"],
            "verified_log_transformation": info["verified_log_transformation"],
            "verified_normalisation": info["verified_normalisation"],
            "verified_centering": info["verified_centering"],
            "verified_scaling": info["verified_scaling"],
            "verified_batch_correction": info["verified_batch_correction"],
            "missing_value_encoding": info["missing_value_encoding"],
            "zero_meaning": info["zero_meaning"],
            "feature_aggregation": info["feature_aggregation"],
            "HMDB_mapping": info["hmdb_mapping"],
            "units_where_reported": info["units_where_reported"],
            "pipeline_functions_applied_before_stage4_matrix": info["writer_function_or_block"],
            "stage4_preprocessing_after_matrix": "Stage 4 model pipeline applies missingness filtering, median imputation and StandardScaler fitted on training/source folds only; not a common upstream harmonisation",
            "evidence_files": info["evidence_files"],
            "verified_vs_inferred_note": "Transformation statements are verified from local code/reports; assay units and exact upstream source-study normalisation are unresolved unless explicitly stated.",
        }
        row.update(numeric_summary(cohort, info))
        rows.append(row)
    return pd.DataFrame(rows)


def build_options_table() -> pd.DataFrame:
    rows = [
        {
            "rank": 1,
            "option": "Declare as-supplied metabolomics external transfer infeasible as primary; retain as stress-test/diagnostic only",
            "scientific_estimand": "Transportability of already processed, cohort-specific metabolomics matrices under a frozen source-trained model",
            "external_data_information_used": "External feature values only for prediction/evaluation; no external labels for preprocessing/tuning",
            "prospective_or_transductive": "prospective for prediction, but not chemically common-scale",
            "leakage_risk": "low for modelling, high risk of misleading biological interpretation if treated as common-scale",
            "interpretation": "Failure/saturation indicates assay/upstream-transformation incompatibility rather than definitive absence of metabolomic signal",
            "limitations": "Cannot fairly compare intensity-based external metabolomics models without raw-data harmonisation or verified common transformation",
            "effect_on_early_integration": "Metabolite block can dominate logits; early integration external claims should be paused or labelled stress-test",
            "effect_on_MOFA": "Do not run cross-cohort MOFA on as-supplied metabolite intensities as if common-scale",
            "effect_on_sCCA": "Do not interpret cross-cohort metabolite loading replication from incompatible intensity scales",
            "effect_on_GCN": "GCN using metabolite node features would inherit scale incompatibility; pause until representation decision",
            "recommendation": "recommended_primary_decision",
        },
        {
            "rank": 2,
            "option": "Find/generate a genuinely common upstream transformation from less-processed matrices for all four cohorts",
            "scientific_estimand": "External prediction using metabolite features processed by a shared deterministic pipeline",
            "external_data_information_used": "Only each sample's raw/less-processed values plus predefined transformation; no labels",
            "prospective_or_transductive": "prospective if transformation is sample-local or source-frozen and raw units are comparable",
            "leakage_risk": "low if fully predefined and does not estimate target cohort distribution",
            "interpretation": "Most defensible metabolomics primary analysis if source files support it",
            "limitations": "Current audit did not verify less-processed common raw matrices for all four; source-study platforms may still differ",
            "effect_on_early_integration": "Could restore fair integrated model if metabolomics becomes common-scale",
            "effect_on_MOFA": "Could permit MOFA if views are transformed comparably and batch/group structure is explicit",
            "effect_on_sCCA": "Could permit cross-cohort loading comparison with careful held-out validation",
            "effect_on_GCN": "Could permit metabolite-node features in GCN under explicit graph modes",
            "recommendation": "preferred_if_feasible_but_not_currently_verified",
        },
        {
            "rank": 3,
            "option": "Scale-invariant sample-local representation, e.g. within-sample percentile/rank features",
            "scientific_estimand": "Relative within-sample metabolite ordering rather than absolute abundance/intensity",
            "external_data_information_used": "Only each individual sample's feature vector; no cohort distribution or labels",
            "prospective_or_transductive": "prospective single-sample compatible",
            "leakage_risk": "low if ties/missing rules are predefined",
            "interpretation": "Tests whether disease signal is in within-sample metabolite profile shape",
            "limitations": "Removes absolute concentration and between-sample intensity information; ranks across chemically diverse metabolites may be biologically blunt",
            "effect_on_early_integration": "May reduce metabolite domination but changes estimand",
            "effect_on_MOFA": "Possible sensitivity only; latent factors become rank-profile factors",
            "effect_on_sCCA": "Possible sensitivity only; loadings reflect rank contrasts",
            "effect_on_GCN": "Possible if node features are ranks, but biological interpretation changes",
            "recommendation": "sensitivity_or_v2c_candidate_after_supervisor_approval",
        },
        {
            "rank": 4,
            "option": "Source-defined robust transformations: source log where valid, clipping/winsorisation, robust scaling",
            "scientific_estimand": "Frozen source-domain decision rule with bounded influence of extreme target values",
            "external_data_information_used": "External sample values transformed using source-defined parameters only",
            "prospective_or_transductive": "prospective if parameters are source-only",
            "leakage_risk": "low, but may not solve unit mismatch",
            "interpretation": "Stress-tests whether saturation is driven by extreme numeric leverage",
            "limitations": "Cannot repair multi-order unit incompatibility if target values are on a different upstream scale",
            "effect_on_early_integration": "May prevent numerical domination but not guarantee scientific comparability",
            "effect_on_MOFA": "Not sufficient for cross-cohort latent interpretation unless common scale is defensible",
            "effect_on_sCCA": "May stabilise numerics but loadings remain hard to compare",
            "effect_on_GCN": "May stabilise training/inference; biological claims remain limited",
            "recommendation": "sensitivity_only",
        },
        {
            "rank": 5,
            "option": "Target-cohort feature normalisation, e.g. independently z-score external cohort",
            "scientific_estimand": "Cohort-level transductive normalisation performance, not single-patient external validation",
            "external_data_information_used": "Uses target/external cohort distribution",
            "prospective_or_transductive": "transductive cohort-level",
            "leakage_risk": "no label leakage if labels unused, but external feature-distribution information is used",
            "interpretation": "May be acceptable only as explicitly transductive sensitivity",
            "limitations": "Not a fully frozen prospective validation; can mask deployment-relevant cohort shift",
            "effect_on_early_integration": "Could improve numerics but changes validation claim",
            "effect_on_MOFA": "Could support exploratory cohort-normalised latent structure, not deployment",
            "effect_on_sCCA": "Could support exploratory within-cohort correlation, not external validation",
            "effect_on_GCN": "Not appropriate for primary frozen external GCN validation",
            "recommendation": "not_primary",
        },
    ]
    return pd.DataFrame(rows)


def write_decision_report(audit: pd.DataFrame, options: pd.DataFrame) -> None:
    recommended = options.loc[options["rank"].eq(1)].iloc[0]
    stale = [
        "corrected_v2 probability-metric Stage 4B metabolomics and early-integration external ROC/PR values",
        "corrected_v2b decision-score Stage 4B metabolomics and early-integration external values, if interpreted as solved rather than diagnostic",
        "Stage 4B paired early-integration comparisons involving metabolomics, until representation comparability is resolved",
        "Stage 4C MOFA/sCCA plans using as-supplied metabolomics as a common quantitative view",
        "Stage 4D/E GCN plans using metabolite node features from as-supplied intensities for external transfer",
        "Any thesis figure claiming metabolomics/early-integration cross-cohort transportability from these as-supplied intensity matrices",
    ]
    files_to_modify = [
        "corrected_v2/scripts/run_stage4b_corrected_v2b.py or a future run_stage4b_corrected_v2c.py if a new representation is approved",
        "corrected_v2/scripts/audit_stage4b_corrected_v2b_transform_shift.py if additional representation diagnostics are needed",
        "corrected_v2/scripts/run_stage4c_corrected_v2.py before MOFA/sCCA execution",
        "corrected_v2/scripts/run_stage4d_corrected_v2_graph.py before graph feature export",
        "corrected_v2/scripts/run_stage4e_gcn_corrected_v2.py before GCN execution",
    ]
    text = "\n".join(
        [
            "# Stage 4B v2c Metabolomics Comparability Decision",
            "",
            "Stage 4C-E remain stopped. No Stage 4B model was rerun for this decision file.",
            "",
            "## Evidence-Supported Decision",
            "",
            "The current as-supplied metabolomics matrices are not defensible as a common primary cross-cohort quantitative representation for external transfer. The local code verifies that selected HMDB columns were copied from cohort-specific processed matrices without a common upstream log/normalisation/centering/scaling/batch-correction step before Stage 4. Stage 4 scaling is source-fitted and leakage-safe, but it cannot make incompatible upstream units comparable.",
            "",
            f"Recommended primary approach: **{recommended['option']}**.",
            "",
            "This means metabolomics external prediction and early integration should be treated as a transportability stress test unless a v2c common-scale representation is explicitly approved and generated. Microbiome-only external transfer remains structurally more defensible because the corrected-v2 pipeline uses relative abundances with CLR preprocessing.",
            "",
            "## Cohort-by-Cohort Upstream Audit",
            "",
            audit.to_markdown(index=False),
            "",
            "## Ranked Analysis Options",
            "",
            options.to_markdown(index=False),
            "",
            "## Sensitivity Plan",
            "",
            "1. First attempt to locate less-processed/raw-compatible metabolomics matrices for all four cohorts and define one deterministic common transformation. Do not proceed unless the transformation provenance is verified.",
            "2. If common upstream processing is not possible, prepare a separate v2c sample-local rank/percentile sensitivity. Missing values and ties must be predefined; no external labels or target-cohort statistics may be used.",
            "3. Retain as-supplied intensity transfer only as a diagnostic stress test of transportability, not a primary biomarker result.",
            "4. Use target-cohort z-scoring only if explicitly labelled transductive cohort-level normalisation; never present it as frozen single-patient external validation.",
            "5. Source-defined robust scaling/clipping may be tested only as a numerical stress sensitivity; it does not by itself solve incompatible upstream units.",
            "",
            "## Files And Scripts That Would Require Modification For A Future v2c Model",
            "",
            *[f"- `{item}`" for item in files_to_modify],
            "",
            "## Current Outputs That Are Stale Or Provisional",
            "",
            *[f"- {item}" for item in stale],
            "",
            "## Statistical Reporting Check",
            "",
            "- v2b paired results remain provisional until representation comparability is resolved.",
            "- P-values should display `p < 0.001`, not `p = 0`.",
            "- BH-FDR must be applied across the eight prespecified paired comparisons.",
            "- Do not call a result supported solely from an unadjusted interval if FDR does not support it.",
            "- Post-hoc best-single-omic comparisons must remain labelled post-hoc.",
            "- Probability saturation and one-class threshold behaviour must be reported separately from decision-score AUC.",
            "",
        ]
    )
    (REPORTS / "STAGE4B_V2C_METABOLOMICS_COMPARABILITY_DECISION.md").write_text(text)
    atomic_csv(pd.DataFrame({"stale_output_or_claim": stale}), TABLES / "stage4b_v2c_stale_outputs.csv")
    atomic_csv(pd.DataFrame({"script_or_file": files_to_modify}), TABLES / "stage4b_v2c_files_requiring_future_modification.csv")


def main() -> None:
    ensure_v2_dirs()
    audit = build_audit_table()
    options = build_options_table()
    atomic_csv(audit, TABLES / "stage4b_v2c_metabolomics_upstream_transformation_audit.csv")
    atomic_csv(options, TABLES / "stage4b_v2c_ranked_analysis_options.csv")
    write_decision_report(audit, options)
    print("STAGE4B_V2C_METABOLOMICS_COMPARABILITY_AUDIT_COMPLETE")


if __name__ == "__main__":
    main()
