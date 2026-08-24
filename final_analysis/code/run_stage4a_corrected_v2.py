from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from stage4_corrected_v2_common import (
    COHORT_GENUS_POLICY,
    METADATA_INPUTS,
    MICROBIOME_INPUTS,
    TABLES,
    V2,
    atomic_csv,
    audit_microbiome_source,
    authoritative_genus_matrix,
    ensure_v2_dirs,
    read_csv,
)


def run_audit_and_harmonisation() -> None:
    ensure_v2_dirs()
    before_rows = []
    after_rows = []
    crosswalk_rows = []
    matrices = {}
    for cohort, path in MICROBIOME_INPUTS.items():
        source = read_csv(path)
        source["sample_id"] = source["sample_id"].astype(str)
        before_rows.append(audit_microbiome_source(cohort, source))
        genus, crosswalk = authoritative_genus_matrix(cohort, source)
        matrices[cohort] = genus
        crosswalk_rows.append(crosswalk)
        atomic_csv(genus, V2 / "data" / "processed" / f"{cohort.lower()}_authoritative_genus_proportions.csv")
        values = genus.drop(columns=["sample_id"])
        row_sums = values.sum(axis=1)
        prevalence = (values > 0).mean(axis=0)
        after_rows.append(
            {
                "cohort": cohort,
                "authoritative_genus_policy": COHORT_GENUS_POLICY[cohort],
                "sample_n": len(genus),
                "genus_p": values.shape[1],
                "median_row_sum_after_reclosure": float(row_sums.median()),
                "min_row_sum_after_reclosure": float(row_sums.min()),
                "max_row_sum_after_reclosure": float(row_sums.max()),
                "row_sums_approximately_one": bool(np.allclose(row_sums.to_numpy(), 1.0, atol=1e-8)),
                "negative_values": int((values < 0).to_numpy().sum()),
                "infinite_or_nan_values": int((~np.isfinite(values.to_numpy())).sum()),
                "overall_zero_fraction": float((values.to_numpy() == 0).mean()),
                "median_prevalence": float(prevalence.median()),
                "features_present_ge_10pct": int((prevalence >= 0.10).sum()),
                "features_present_ge_20pct": int((prevalence >= 0.20).sum()),
                "features_present_ge_50pct": int((prevalence >= 0.50).sum()),
            }
        )

    atomic_csv(pd.DataFrame(before_rows), TABLES / "stage4a_corrected_v2_microbiome_before_audit.csv")
    atomic_csv(pd.DataFrame(after_rows), TABLES / "stage4a_corrected_v2_authoritative_genus_after_audit.csv")
    atomic_csv(pd.concat(crosswalk_rows, ignore_index=True), TABLES / "stage4a_corrected_v2_taxon_crosswalk.csv")

    shared_rows = []
    for disease, a, b in [("IBD", "Franzosa", "iHMP"), ("CRC", "Sinha", "Yachida")]:
        fa = set(matrices[a].columns) - {"sample_id"}
        fb = set(matrices[b].columns) - {"sample_id"}
        shared = sorted(fa & fb)
        atomic_csv(pd.DataFrame({"canonical_name": [s.replace("mic__", "") for s in shared], "feature_name": shared}), TABLES / f"stage4a_corrected_v2_{disease.lower()}_shared_genera.csv")
        shared_rows.append(
            {
                "disease": disease,
                "cohort_a": a,
                "cohort_b": b,
                "cohort_a_authoritative_genus_p": len(fa),
                "cohort_b_authoritative_genus_p": len(fb),
                "shared_authoritative_genus_p": len(shared),
                "shared_feature_order_deterministic": shared == sorted(shared),
                "notes": "Shared schema is defined by feature availability only; target outcomes and target abundance distributions are not used for fitting decisions.",
            }
        )
    atomic_csv(pd.DataFrame(shared_rows), TABLES / "stage4a_corrected_v2_shared_genus_feature_space.csv")
    atomic_csv(descriptive_class_counts(), TABLES / "stage4a_corrected_v2_descriptive_class_counts.csv")

    tests = []
    for row in after_rows:
        tests.extend(
            [
                {"test": f"{row['cohort']}_one_row_per_sample", "status": "pass" if row["sample_n"] > 0 else "fail", "observed": row["sample_n"]},
                {"test": f"{row['cohort']}_one_column_per_genus", "status": "pass" if row["genus_p"] > 0 else "fail", "observed": row["genus_p"]},
                {"test": f"{row['cohort']}_row_sums_one", "status": "pass" if row["row_sums_approximately_one"] else "fail", "observed": row["median_row_sum_after_reclosure"]},
                {"test": f"{row['cohort']}_no_negative_values", "status": "pass" if row["negative_values"] == 0 else "fail", "observed": row["negative_values"]},
                {"test": f"{row['cohort']}_no_nan_or_inf", "status": "pass" if row["infinite_or_nan_values"] == 0 else "fail", "observed": row["infinite_or_nan_values"]},
            ]
        )
    atomic_csv(pd.DataFrame(tests), TABLES / "stage4a_corrected_v2_validation_tests.csv")


def descriptive_class_counts() -> pd.DataFrame:
    rows = []
    ibd = read_csv(METADATA_INPUTS["IBD"])
    for cohort, sub in ibd.groupby("cohort", dropna=False):
        if str(cohort) not in {"Franzosa", "iHMP"}:
            continue
        labels = sub["harmonised_label"].replace({"IBD_control": "Control"}).astype(str)
        counts = labels.value_counts().to_dict()
        rows.append(
            {
                "disease": "IBD",
                "cohort": cohort,
                "control_n": int(counts.get("Control", 0)),
                "pooled_ibd_n": int(counts.get("CD", 0) + counts.get("UC", 0)),
                "cd_n_descriptive_only": int(counts.get("CD", 0)),
                "uc_n_descriptive_only": int(counts.get("UC", 0)),
                "subtype_modelling_status": "not_modelled_in_corrected_v2",
            }
        )
    crc = read_csv(METADATA_INPUTS["CRC"])
    for cohort, sub in crc.groupby("cohort", dropna=False):
        if str(cohort) not in {"Sinha", "Yachida"}:
            continue
        counts = sub["crc_binary"].map({0: "Control/Healthy", 1: "CRC"}).astype(str).value_counts().to_dict()
        rows.append(
            {
                "disease": "CRC",
                "cohort": cohort,
                "control_n": int(counts.get("Control/Healthy", 0)),
                "pooled_ibd_n": 0,
                "cd_n_descriptive_only": 0,
                "uc_n_descriptive_only": 0,
                "crc_n": int(counts.get("CRC", 0)),
                "subtype_modelling_status": "not_applicable",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrected-v2 Stage 4A authoritative genus harmonisation.")
    parser.add_argument("--audit-only", action="store_true", help="The script is lightweight; audit-only still writes corrected-v2 audit tables.")
    parser.parse_args()
    run_audit_and_harmonisation()
    print("STAGE4A_CORRECTED_V2_AUTHORITATIVE_GENUS_READY")


if __name__ == "__main__":
    main()
