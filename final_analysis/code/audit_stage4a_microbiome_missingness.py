from __future__ import annotations

import pandas as pd

from stage4_corrected_v2_common import (
    MICROBIOME_INPUTS,
    TABLES,
    atomic_csv,
    microbiome_missingness_audit,
    read_csv,
)


AUDIT_TABLE = TABLES / "stage4a_microbiome_missingness_audit.csv"
AUDIT_SUMMARY = TABLES / "stage4a_microbiome_missingness_summary.csv"


def main() -> None:
    per_feature = []
    summaries = []
    for cohort, path in MICROBIOME_INPUTS.items():
        df = read_csv(path)
        audit = microbiome_missingness_audit(cohort, df)
        per_feature.append(audit)
        summaries.append(
            {
                "cohort": cohort,
                "n_samples": len(df),
                "n_features": len(audit),
                "n_features_with_any_missing": int((audit["n_missing_before"] > 0).sum()),
                "n_cells_missing_before": int(audit["n_missing_before"].sum()),
                "n_cells_converted_to_zero": int(audit["n_converted_to_zero"].sum()),
                "n_features_wholly_missing_unresolved": int(audit["wholly_missing_unresolved"].sum()),
            }
        )
    per_feature_df = pd.concat(per_feature, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    atomic_csv(per_feature_df, AUDIT_TABLE)
    atomic_csv(summary_df, AUDIT_SUMMARY)

    unresolved = per_feature_df[per_feature_df["wholly_missing_unresolved"]]
    print(summary_df.to_string(index=False))
    if len(unresolved):
        print("\nWholly-missing (unresolved) features excluded from authoritative genus construction:")
        print(unresolved[["cohort", "feature"]].to_string(index=False))
    else:
        print("\nNo wholly-missing features found; all NaN cells are partial and were zero-filled per the absence convention.")


if __name__ == "__main__":
    main()
