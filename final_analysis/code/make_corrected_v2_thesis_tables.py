from __future__ import annotations

from pathlib import Path

import pandas as pd

from stage4_corrected_v2_common import REPORTS, TABLES, V2, atomic_csv, ensure_v2_dirs


def require(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required corrected-v2 result table is missing. Run the corrected pipeline first: {path}")
    return pd.read_csv(path)


def cohort_table() -> pd.DataFrame:
    feature_space = require(TABLES / "stage4a_corrected_v2_shared_genus_feature_space.csv")
    after = require(TABLES / "stage4a_corrected_v2_authoritative_genus_after_audit.csv")
    counts = require(TABLES / "stage4a_corrected_v2_descriptive_class_counts.csv")
    rows = []
    for _, row in after.iterrows():
        disease = "IBD" if row["cohort"] in {"Franzosa", "iHMP"} else "CRC"
        shared = feature_space.loc[feature_space["disease"].eq(disease), "shared_authoritative_genus_p"].iloc[0]
        cohort_counts = counts.loc[counts["cohort"].astype(str).eq(str(row["cohort"]))]
        count_row = cohort_counts.iloc[0].to_dict() if not cohort_counts.empty else {}
        rows.append(
            {
                "disease": disease,
                "cohort": row["cohort"],
                "microbial_assay": "shotgun metagenomics" if row["cohort"] in {"Franzosa", "iHMP", "Yachida"} else "16S rRNA V3-V4",
                "metabolomics_assay": {
                    "Franzosa": "untargeted LC-MS Orbitrap",
                    "iHMP": "multi-method LC-MS Orbitrap",
                    "Sinha": "non-targeted Metabolon LC/GC-MS/MS",
                    "Yachida": "CE-TOFMS",
                }[row["cohort"]],
                "matched_participant_count": int(row["sample_n"]),
                "control_n": count_row.get("control_n", pd.NA),
                "pooled_ibd_n": count_row.get("pooled_ibd_n", pd.NA),
                "cd_n_descriptive_only": count_row.get("cd_n_descriptive_only", pd.NA),
                "uc_n_descriptive_only": count_row.get("uc_n_descriptive_only", pd.NA),
                "crc_n": count_row.get("crc_n", pd.NA),
                "authoritative_genus_policy": row["authoritative_genus_policy"],
                "authoritative_genus_features": int(row["genus_p"]),
                "shared_genus_features_for_transfer": int(shared),
                "median_prevalence": row["median_prevalence"],
                "overall_zero_fraction": row["overall_zero_fraction"],
                "clinical_variable_policy": "Clinical variables are descriptive only in corrected-v2; they were not sufficiently complete and harmonised for a fair common cross-cohort predictive baseline.",
            }
        )
    return pd.DataFrame(rows)


def predictive_results_table() -> pd.DataFrame:
    ext = require(TABLES / "stage4b_corrected_v2_external_performance.csv")
    thr = require(TABLES / "stage4b_corrected_v2_thresholds.csv")
    merged = ext.merge(
        thr[["disease", "task", "training_cohort", "external_cohort", "representation", "selected_C", "threshold"]],
        on=["disease", "task", "training_cohort", "external_cohort", "representation"],
        how="left",
    )
    cols = [
        "disease",
        "task",
        "training_cohort",
        "external_cohort",
        "representation",
        "selected_C",
        "threshold",
        "roc_auc",
        "roc_auc_ci_low",
        "roc_auc_ci_high",
        "pr_auc",
        "pr_auc_ci_low",
        "pr_auc_ci_high",
        "external_positive_prevalence",
        "mcc",
        "sensitivity",
        "specificity",
        "precision",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "probability_sd",
        "degenerate_probability_flag",
        "one_class_prediction_flag",
    ]
    return merged[[c for c in cols if c in merged.columns]]


def main() -> None:
    ensure_v2_dirs()
    atomic_csv(cohort_table(), TABLES / "thesis_table1_corrected_v2_cohort_overview.csv")
    atomic_csv(predictive_results_table(), TABLES / "thesis_table2_corrected_v2_predictive_results.csv")
    (REPORTS / "CORRECTED_V2_TABLES_README.md").write_text(
        "# Corrected-v2 Thesis Tables\n\n"
        "These tables are generated only from corrected-v2 outputs. If a required corrected result is missing, the script fails rather than using stale historical values.\n"
    )
    print("CORRECTED_V2_THESIS_TABLES_READY")


if __name__ == "__main__":
    main()
