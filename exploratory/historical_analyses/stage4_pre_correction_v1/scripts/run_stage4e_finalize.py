from __future__ import annotations

import pandas as pd

from stage4_common import REPORTS, TABLES


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["disease", "representation", "direction", "roc_auc", "pr_auc", "mcc", "source_table"]).to_csv(TABLES / "stage4e_final_representation_performance.csv", index=False)
    pd.DataFrame(columns=["feature_type", "feature_id", "feature_name", "evidence_source", "cross_cohort_support", "notes"]).to_csv(TABLES / "stage4e_final_convergent_features.csv", index=False)
    pd.DataFrame(columns=["subnetwork_id", "nodes", "edges", "supporting_methods", "interpretation"]).to_csv(TABLES / "stage4e_final_subnetworks.csv", index=False)
    pd.DataFrame(columns=["result", "value", "source"]).to_csv(TABLES / "FINAL_THESIS_RESULTS_NUMBERS.csv", index=False)
    (REPORTS / "STAGE4E_MAIN_FIGURE_SELECTION.md").write_text("""# Stage 4E Main Figure Selection

Preferred final figure strategy after user execution: study framework, external performance by representation, MOFA latent factor result, GCN/network result, and cross-method biological convergence.
""")
    (REPORTS / "FINAL_THESIS_METHODS_SUMMARY.md").write_text("Stage 4E is a no-fitting synthesis script. It reads completed Stage 4 outputs only.\n")
    (REPORTS / "FINAL_THESIS_RESULTS_SYNTHESIS.md").write_text("Stage 4E results synthesis is pending user-executed Stage 4B-4D analyses.\n")
    (REPORTS / "FINAL_THESIS_DISCUSSION_POINTS.md").write_text("Discussion points will be finalized after Stage 4B-4D outputs exist.\n")
    (REPORTS / "FINAL_EXAMINER_QUESTIONS_AND_ANSWERS.md").write_text("Examiner Q&A will be populated after user-executed analyses.\n")
    (REPORTS / "STAGE4E_FIGURE_QC.md").write_text("No Stage 4E figures generated yet because upstream user-executed model outputs are absent.\n")
    print("Stage 4E synthesis scaffold complete; no model fitting performed.")


if __name__ == "__main__":
    main()
