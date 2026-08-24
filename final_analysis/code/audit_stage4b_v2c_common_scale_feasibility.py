from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from run_stage4b_corrected_v2b import CorrectedDataBundle
from stage4_corrected_v2_common import PROJECT, REPORTS, TABLES, atomic_csv, ensure_v2_dirs


CANDIDATES = [
    {
        "cohort": "Franzosa",
        "path": PROJECT / "IBD-DataSet/FRANZOSA_IBD_2019/downloaded_full_metabolomics/franzosa_original_samples_original_mapped_metabolites.repaired.tsv",
        "candidate_role": "downloaded/repaired source metabolomics",
        "sample_id_column": "Sample",
        "exclude_columns": {"Sample", "label"},
        "metabolite_identifier_source": "feature name plus repaired map file HMDB/KEGG",
        "assay_platform": "LC-MS; feature names/map indicate HILIC/C18 and ionisation",
        "raw_vs_processed_status": "processed/repaired as supplied; not raw instrument data",
        "units": "not locally documented",
        "normalisation": "not locally verified",
        "log_transformation": "not locally verified",
        "centring_scaling": "not locally verified",
        "missing_value_encoding": "NA/blank parsed by pandas; source-specific semantics unresolved",
    },
    {
        "cohort": "Franzosa",
        "path": PROJECT / "IBD-DataSet/FRANZOSA_IBD_2019/downloaded_full_metabolomics/franzosa_supplement_full_mtb.cleaned.tsv",
        "candidate_role": "downloaded supplementary cleaned metabolomics",
        "sample_id_column": "Sample",
        "exclude_columns": {"Sample"},
        "metabolite_identifier_source": "feature name plus supplement cleaned map HMDB/KEGG",
        "assay_platform": "LC-MS; HILIC/C18 inferred from feature names/map",
        "raw_vs_processed_status": "processed supplement table; not raw instrument data",
        "units": "not locally documented",
        "normalisation": "not locally verified",
        "log_transformation": "not locally verified",
        "centring_scaling": "not locally verified",
        "missing_value_encoding": "NA/blank parsed by pandas; source-specific semantics unresolved",
    },
    {
        "cohort": "Franzosa",
        "path": PROJECT / "thesis_preprocessing_workflow/ibd_pipeline/data_processed/franzosa_primary_high_confidence_hmdb_matrix.csv",
        "candidate_role": "Stage 4 current high-confidence HMDB matrix",
        "sample_id_column": "sample_id",
        "exclude_columns": {"sample_id"},
        "metabolite_identifier_source": "HMDB column names",
        "assay_platform": "LC-MS; inherited from selected source features",
        "raw_vs_processed_status": "processed thesis matrix selecting one source feature per HMDB",
        "units": "not locally documented",
        "normalisation": "none added by thesis writer",
        "log_transformation": "none added by thesis writer",
        "centring_scaling": "none before Stage 4 model pipeline",
        "missing_value_encoding": "NaN retained",
    },
    {
        "cohort": "iHMP",
        "path": PROJECT / "IBD-DataSet/iHMP_IBDMDB/mtb.tsv",
        "candidate_role": "supplied iHMP metabolomics",
        "sample_id_column": "Sample",
        "exclude_columns": {"Sample"},
        "metabolite_identifier_source": "feature name plus mtb.map.tsv HMDB/KEGG",
        "assay_platform": "LC-MS; map Method includes HILIC/C18 style methods",
        "raw_vs_processed_status": "processed as supplied; not raw instrument data",
        "units": "not locally documented",
        "normalisation": "not locally verified",
        "log_transformation": "not locally verified",
        "centring_scaling": "not locally verified",
        "missing_value_encoding": "NA/blank parsed by pandas; source-specific semantics unresolved",
    },
    {
        "cohort": "iHMP",
        "path": PROJECT / "thesis_preprocessing_workflow/ibd_pipeline/data_processed/ihmp_primary_high_confidence_hmdb_matrix.csv",
        "candidate_role": "Stage 4 current high-confidence HMDB matrix",
        "sample_id_column": "sample_id",
        "exclude_columns": {"sample_id"},
        "metabolite_identifier_source": "HMDB column names",
        "assay_platform": "LC-MS; inherited from selected source features",
        "raw_vs_processed_status": "processed thesis matrix selecting one source feature per HMDB",
        "units": "not locally documented",
        "normalisation": "none added by thesis writer",
        "log_transformation": "none added by thesis writer",
        "centring_scaling": "none before Stage 4 model pipeline",
        "missing_value_encoding": "NaN retained",
    },
    {
        "cohort": "Sinha",
        "path": PROJECT / "Colon_Cancer_Data_Set/Sinha-et-al/mtb.tsv",
        "candidate_role": "local source metabolomics",
        "sample_id_column": "Sample",
        "exclude_columns": {"Sample"},
        "metabolite_identifier_source": "feature name plus mtb.map.tsv HMDB/KEGG",
        "assay_platform": "fecal metabolomics; exact local platform metadata unresolved",
        "raw_vs_processed_status": "processed curated table; not raw instrument data",
        "units": "not locally documented",
        "normalisation": "not locally verified; values appear processed but range alone is not proof",
        "log_transformation": "not locally verified",
        "centring_scaling": "not locally verified",
        "missing_value_encoding": "NA/blank parsed by pandas; source-specific semantics unresolved",
    },
    {
        "cohort": "Yachida",
        "path": PROJECT / "Colon_Cancer_Data_Set/Yachinda_et-al/mtb.tsv",
        "candidate_role": "local source metabolomics",
        "sample_id_column": "Sample",
        "exclude_columns": {"Sample"},
        "metabolite_identifier_source": "feature name plus mtb.map.tsv HMDB/KEGG",
        "assay_platform": "fecal metabolomics; exact local platform metadata unresolved",
        "raw_vs_processed_status": "processed curated table; not raw instrument data",
        "units": "not locally documented",
        "normalisation": "not locally verified",
        "log_transformation": "not locally verified",
        "centring_scaling": "not locally verified",
        "missing_value_encoding": "NA/blank parsed by pandas; source-specific semantics unresolved",
    },
    {
        "cohort": "Yachida",
        "path": PROJECT / "thesis_preprocessing_workflow/crc_pipeline/tables/stage2_model_prep/crc_stage2_yachida_primary_crc_vs_control_model_prep.csv",
        "candidate_role": "Stage 2 Yachida primary CRC-vs-control prep",
        "sample_id_column": "Sample",
        "exclude_columns": {
            "Sample", "Subject", "Study.Group", "yachida_stage_source_raw", "yachida_stage_normalized",
            "yachida_stage_major", "yachida_stage_major_with_healthy", "yachida_stage_source_faithful_label",
            "yachida_stage_harmonized_label", "trajectory_label", "trajectory_order", "trajectory_eligible",
            "model_label",
        },
        "metabolite_identifier_source": "KEGG-style feature names, later mapped to HMDB where possible",
        "assay_platform": "fecal metabolomics; source feature names are KEGG-style metabolite labels",
        "raw_vs_processed_status": "processed model-prep table; not raw instrument data",
        "units": "not locally documented",
        "normalisation": "not locally verified",
        "log_transformation": "none added by Stage 2 writer",
        "centring_scaling": "none before Stage 2/Stage 4 model pipeline",
        "missing_value_encoding": "NaN retained after numeric coercion",
    },
    {
        "cohort": "Sinha/Yachida",
        "path": PROJECT / "thesis_preprocessing_workflow/crc_pipeline/stage2/data_processed/crc_stage2_primary_metabolomics_canonical.csv",
        "candidate_role": "Stage 4 current CRC canonical HMDB matrix",
        "sample_id_column": "sample_id",
        "exclude_columns": {"cohort", "sample_id", "participant_id"},
        "metabolite_identifier_source": "met__HMDB canonical columns",
        "assay_platform": "inherited from Sinha/Yachida source tables",
        "raw_vs_processed_status": "processed thesis canonical matrix selecting one feature per canonical metabolite",
        "units": "not locally documented",
        "normalisation": "none added by Stage 2 writer",
        "log_transformation": "none added by Stage 2 writer",
        "centring_scaling": "none before Stage 4 model pipeline",
        "missing_value_encoding": "NaN retained",
    },
    {
        "cohort": "derived_all",
        "path": PROJECT / "thesis_preprocessing_workflow/stage4_multiomics_integration/data/stage4c_scca_pma_inputs/IBD_Franzosa_metabolome.csv",
        "candidate_role": "derived Stage 4C sCCA input",
        "sample_id_column": "sample_id",
        "exclude_columns": {"sample_id"},
        "metabolite_identifier_source": "derived from Stage 4 matrices",
        "assay_platform": "derived; not independent source",
        "raw_vs_processed_status": "derived analysis input; unsuitable as primary source provenance",
        "units": "not primary",
        "normalisation": "derived by prior Stage 4C scripts; not accepted for primary feasibility",
        "log_transformation": "not primary",
        "centring_scaling": "not primary",
        "missing_value_encoding": "not primary",
    },
]


def corrected_v2_sample_ids() -> dict[str, set[str]]:
    bundle = CorrectedDataBundle()
    ids: dict[str, set[str]] = {}
    for cohort in ["Franzosa", "iHMP", "Sinha", "Yachida"]:
        disease = "IBD" if cohort in {"Franzosa", "iHMP"} else "CRC"
        if disease == "IBD":
            sub = bundle.ibd_samples[bundle.ibd_samples["cohort"].eq(cohort)].copy()
            ids[cohort] = set(sub["sample_id"].astype(str)) & bundle.matched_ids[cohort]
        else:
            sub = bundle.crc_meta[bundle.crc_meta["cohort"].eq(cohort)].copy()
            ids[cohort] = set(sub["sample_id"].astype(str)) & bundle.matched_ids[cohort]
    return ids


def read_candidate(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", low_memory=False)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def detect_feature_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    cols = [c for c in df.columns if c not in exclude]
    numeric_cols = []
    for col in cols:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            numeric_cols.append(col)
    return numeric_cols


def summarize_values(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    if not feature_cols:
        return {"value_min": np.nan, "value_median": np.nan, "value_max": np.nan, "negative_values_present": False, "zero_fraction_observed": np.nan}
    vals = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).ravel()
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return {"value_min": np.nan, "value_median": np.nan, "value_max": np.nan, "negative_values_present": False, "zero_fraction_observed": np.nan}
    return {
        "value_min": float(np.min(finite)),
        "value_median": float(np.median(finite)),
        "value_max": float(np.max(finite)),
        "negative_values_present": bool(np.any(finite < 0)),
        "zero_fraction_observed": float(np.mean(finite == 0)),
    }


def hmdb_possible(feature_cols: list[str], candidate: dict[str, Any]) -> bool:
    text = " ".join(feature_cols[:50]) + " " + str(candidate["metabolite_identifier_source"])
    return "HMDB" in text.upper()


def inventory() -> pd.DataFrame:
    ids = corrected_v2_sample_ids()
    rows = []
    for candidate in CANDIDATES:
        path = Path(candidate["path"])
        df = read_candidate(path)
        row = {k: v for k, v in candidate.items() if k != "exclude_columns"}
        row["exact_path"] = str(path)
        row["exists"] = df is not None
        if df is None:
            row.update({"sample_count": 0, "feature_count": 0, "corrected_v2_population_sample_match_n": 0})
            rows.append(row)
            continue
        sample_col = candidate["sample_id_column"]
        feature_cols = detect_feature_columns(df, candidate["exclude_columns"])
        sample_ids = set(df[sample_col].dropna().astype(str)) if sample_col in df.columns else set()
        cohort = candidate["cohort"]
        if cohort == "Sinha/Yachida":
            match_n = int(len((set(df.loc[df["cohort"].eq("Sinha"), sample_col].astype(str)) & ids["Sinha"]) | (set(df.loc[df["cohort"].eq("Yachida"), sample_col].astype(str)) & ids["Yachida"])))
            corrected_n = len(ids["Sinha"]) + len(ids["Yachida"])
        elif cohort == "derived_all":
            match_n = np.nan
            corrected_n = np.nan
        else:
            match_n = int(len(sample_ids & ids.get(cohort, set())))
            corrected_n = len(ids.get(cohort, set()))
        row.update(
            {
                "sample_count": int(len(df)),
                "feature_count": int(len(feature_cols)),
                "sample_id_column_present": sample_col in df.columns,
                "corrected_v2_population_sample_match_n": match_n,
                "corrected_v2_population_expected_n": corrected_n,
                "sample_ids_match_corrected_v2_population": bool(match_n == corrected_n) if pd.notna(match_n) else False,
                "metabolite_identifiers": candidate["metabolite_identifier_source"],
                "HMDB_level_harmonisation_possible": hmdb_possible(feature_cols, candidate),
                "transformation_verified_or_inferred": "verified where thesis code/reports state no added transformation; upstream source-study processing mostly unresolved",
                "scale_meaningfully_comparable_with_paired_cohort": False,
                "scale_comparability_reason": "No local evidence of verified compatible measurement semantics/units and prior v2b diagnostics showed multi-order external shift for metabolomics transfers.",
                "same_log_function_caveat": "Applying the same log function would not by itself make cross-platform response scales comparable.",
                **summarize_values(df, feature_cols),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def acceptance_tests(inv: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("IBD", "Franzosa", "iHMP"),
        ("CRC", "Sinha", "Yachida"),
    ]
    rows = []
    criteria = [
        "deterministic_sample_matching",
        "deterministic_metabolite_identity_matching",
        "verified_compatible_measurement_semantics",
        "identical_mathematical_preprocessing_can_be_applied",
        "no_target_outcomes_used",
        "no_target_cohort_distribution_used_for_source_model_deployment",
        "no_multi_order_transformed_shift_like_v2b",
        "scientifically_interpretable_meaning",
    ]
    for disease, a, b in pairs:
        a_rows = inv[inv["cohort"].eq(a)]
        b_rows = inv[inv["cohort"].eq(b)]
        sample_ok = bool(a_rows["sample_ids_match_corrected_v2_population"].any() and b_rows["sample_ids_match_corrected_v2_population"].any())
        hmdb_ok = bool(a_rows["HMDB_level_harmonisation_possible"].any() and b_rows["HMDB_level_harmonisation_possible"].any())
        verdicts = {
            "deterministic_sample_matching": (sample_ok, "At least one local matrix per cohort matches the corrected-v2 sample population."),
            "deterministic_metabolite_identity_matching": (hmdb_ok, "HMDB-level matching exists for current canonical matrices and/or map-backed source matrices."),
            "verified_compatible_measurement_semantics": (False, "No local evidence verifies compatible units/measurement semantics across the paired cohorts."),
            "identical_mathematical_preprocessing_can_be_applied": (False, "A shared mathematical function can be coded, but compatibility of upstream response scales is not verified; same log does not solve arbitrary LC-MS response scales."),
            "no_target_outcomes_used": (True, "Feasibility audit does not require outcome labels."),
            "no_target_cohort_distribution_used_for_source_model_deployment": (True, "Frozen source-model deployment can avoid target distribution, but only if representation is otherwise comparable."),
            "no_multi_order_transformed_shift_like_v2b": (False, "Existing v2b diagnostics showed multi-order transformed shift/saturation for metabolomics transfer."),
            "scientifically_interpretable_meaning": (False, "As-supplied intensity values do not have a verified shared quantitative meaning across cohorts."),
        }
        for crit in criteria:
            passed, evidence = verdicts[crit]
            rows.append(
                {
                    "disease": disease,
                    "cohort_pair": f"{a}<->{b}",
                    "criterion": crit,
                    "passed": passed,
                    "evidence": evidence,
                    "blocking_if_failed": crit
                    in {
                        "deterministic_sample_matching",
                        "deterministic_metabolite_identity_matching",
                        "verified_compatible_measurement_semantics",
                        "identical_mathematical_preprocessing_can_be_applied",
                        "no_multi_order_transformed_shift_like_v2b",
                        "scientifically_interpretable_meaning",
                    },
                }
            )
    out = pd.DataFrame(rows)
    pair_status = (
        out.groupby(["disease", "cohort_pair"])
        .apply(lambda g: bool(g["passed"].all()), include_groups=False)
        .reset_index(name="all_acceptance_criteria_passed")
    )
    return out.merge(pair_status, on=["disease", "cohort_pair"], how="left")


def write_report(inv: pd.DataFrame, tests: pd.DataFrame) -> None:
    feasible = bool(tests.groupby(["disease", "cohort_pair"])["passed"].all().any())
    final_statement = "COMMON-SCALE PRIMARY REPRESENTATION FEASIBLE" if feasible else "NO DEFENSIBLE COMMON-SCALE METABOLOMICS MATRIX AVAILABLE LOCALLY."
    branch = "A" if feasible else "B"
    recommendation = (
        "Common-scale primary representation is feasible."
        if feasible
        else "No common-scale representation is feasible; use v2b as stress test and restructure later stages accordingly."
    )
    text = "\n".join(
        [
            "# Stage 4B v2c Final Metabolomics Feasibility Decision",
            "",
            "Bounded local audit only. No models were run, no downloads were performed, v2b was not modified, and Stage 4C-E remain stopped.",
            "",
            f"## Decision: {final_statement}",
            "",
            f"Recommended branch: **{branch}. {recommendation}**",
            "",
            "A transductive or sample-local sensitivity remains possible, but it cannot replace primary frozen external validation unless explicitly reframed as a different estimand.",
            "",
            "## Why This Is The Final Local Decision",
            "",
            "- Local files provide deterministic sample matching and metabolite/HMDB matching for some matrices.",
            "- They do not provide verified compatible measurement semantics, units, or a common upstream transformation across Franzosa/iHMP or Sinha/Yachida.",
            "- Applying the same log function would not make arbitrary cross-platform LC-MS response scales comparable.",
            "- Existing v2b diagnostics already showed multi-order external transformed shifts for metabolomics transfers.",
            "- Therefore the strict acceptance criteria for a common primary representation are not met.",
            "",
            "## Candidate Matrix Inventory",
            "",
            inv.to_markdown(index=False),
            "",
            "## Strict Acceptance Tests",
            "",
            tests.to_markdown(index=False),
            "",
            "## Branch Consequence",
            "",
            "Stage 4B metabolomics and early-integration external transfer should be retained as a stress test/diagnostic, not a primary common-scale metabolomics validation. Later Stage 4C-E scripts should be restructured so primary cross-cohort multi-omics claims do not depend on as-supplied metabolomics intensities as though they share a quantitative scale.",
            "",
        ]
    )
    (REPORTS / "STAGE4B_V2C_FINAL_METABOLOMICS_FEASIBILITY_DECISION.md").write_text(text)


def main() -> None:
    ensure_v2_dirs()
    inv = inventory()
    tests = acceptance_tests(inv)
    atomic_csv(inv, TABLES / "stage4b_v2c_candidate_metabolomics_matrix_inventory.csv")
    atomic_csv(tests, TABLES / "stage4b_v2c_common_scale_acceptance_tests.csv")
    write_report(inv, tests)
    print("STAGE4B_V2C_COMMON_SCALE_FEASIBILITY_AUDIT_COMPLETE")


if __name__ == "__main__":
    main()
