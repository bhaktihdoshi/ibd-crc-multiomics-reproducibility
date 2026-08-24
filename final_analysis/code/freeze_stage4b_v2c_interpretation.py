from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from stage4_corrected_v2_common import MANIFESTS, REPORTS, TABLES, atomic_json, ensure_v2_dirs


PRESERVED_OUTPUTS = [
    "tables/stage4b_corrected_v2b_external_performance.csv",
    "tables/stage4b_corrected_v2b_paired_model_comparisons.csv",
    "tables/stage4b_corrected_v2b_probability_diagnostics.csv",
    "tables/stage4b_corrected_v2b_transformation_domain_shift_diagnostics.csv",
    "reports/STAGE4B_V2C_FINAL_METABOLOMICS_FEASIBILITY_DECISION.md",
]


INTERPRETATION_RULES = [
    {
        "rule_id": "stage4b_principal_transportable_analysis",
        "interpretation": "Microbiome-only reciprocal validation is the principal directly transportable molecular analysis.",
        "qualification": "Retain the Sinha 16S-versus-Yachida shotgun limitation.",
    },
    {
        "rule_id": "metabolome_stress_test",
        "interpretation": "Metabolome-only transfer is an as-supplied-scale stress test.",
        "qualification": "Paired cohorts lack verified common metabolomics measurement semantics and units.",
    },
    {
        "rule_id": "early_integration_stress_test",
        "interpretation": "Early integration is also an as-supplied-scale stress test.",
        "qualification": "It inherits the metabolomics incompatibility.",
    },
    {
        "rule_id": "decision_score_auc",
        "interpretation": "Decision-score ROC-AUC measures ranking under domain shift.",
        "qualification": "It is not evidence of deployable calibrated probability transport.",
    },
    {
        "rule_id": "classification_and_calibration_not_claimed",
        "interpretation": "Saturated/compressed probabilities and one-class predictions prevent clinical classification or calibration claims.",
        "qualification": "Thresholded outputs are diagnostic only when transfer probabilities fail.",
    },
    {
        "rule_id": "high_auc_not_operational_transportability",
        "interpretation": "High AUC must not be described as operational transportability where thresholds or probabilities fail.",
        "qualification": "AUC and deployable clinical classification answer different questions.",
    },
    {
        "rule_id": "crc_biology_not_intrinsically_less_reproducible",
        "interpretation": "No claim should be made that CRC biology is intrinsically less reproducible.",
        "qualification": "Observed performance reflects these cohorts, platforms, feature spaces and stress-test assumptions.",
    },
    {
        "rule_id": "paired_comparisons_conditional",
        "interpretation": "Paired comparisons remain conditional on the stress-test representation.",
        "qualification": "They should not be used as proof of common-scale metabolomics transport.",
    },
]

RESTRUCTURED_DEPENDENCIES = [
    {
        "stage": "4B",
        "artifact_scope": "microbiome_only_external_transfer",
        "status": "principal_transportable_molecular_analysis",
        "interpretation_constraint": "retain Sinha 16S versus Yachida shotgun caveat",
    },
    {
        "stage": "4B",
        "artifact_scope": "metabolome_only_external_transfer",
        "status": "stress_test_only",
        "interpretation_constraint": "do not present as common-scale external metabolomics validation",
    },
    {
        "stage": "4B",
        "artifact_scope": "early_integration_external_transfer",
        "status": "stress_test_only",
        "interpretation_constraint": "inherits metabolomics measurement-scale incompatibility",
    },
    {
        "stage": "4C_MOFA",
        "artifact_scope": "latent_factor_analysis",
        "status": "descriptive_structural_analysis_to_be_rerun",
        "interpretation_constraint": "cohort-local preprocessing; no external projection; pooled phenotype annotation only",
    },
    {
        "stage": "4C_sCCA",
        "artifact_scope": "cross_view_association",
        "status": "independent_cohort_fits_to_be_rerun",
        "interpretation_constraint": "held-out correlations separate from final refit; no phenotype in fitting/tuning",
    },
    {
        "stage": "4D",
        "artifact_scope": "prior_graph",
        "status": "three_modes_to_be_constructed",
        "interpretation_constraint": "identity, KEGG-only, KEGG+NJC19; unsigned undirected binary graph",
    },
    {
        "stage": "4E",
        "artifact_scope": "GCN",
        "status": "primary_within_cohort_topology_evaluation_to_be_run",
        "interpretation_constraint": "external mode secondary stress test only",
    },
]

FIGURE_POLICY = [
    {"figure_intent": "External Stage 4B ROC curves/forest estimates", "required_label": "bootstrap CIs; microbiome principal, metabolome/integration stress-test where shown", "generate_now": False},
    {"figure_intent": "Paired early-integration delta AUC forest", "required_label": "stress-test comparisons", "generate_now": False},
    {"figure_intent": "MOFA variance heatmap and factor-score distributions", "required_label": "cohort-by-view structural analysis", "generate_now": False},
    {"figure_intent": "sCCA held-out distributions and loading reproducibility", "required_label": "independently fitted cohort-local sCCA", "generate_now": False},
    {"figure_intent": "Identity to KEGG to KEGG+NJC19 connectivity", "required_label": "connectivity not biological activity", "generate_now": False},
    {"figure_intent": "GCN paired fold/seed plots", "required_label": "within-cohort topology ablation", "generate_now": False},
    {"figure_intent": "External probability distributions", "required_label": "saturation/compression diagnostics", "generate_now": False},
    {"figure_intent": "Cross-method feature-evidence matrix", "required_label": "evidence synthesis, not causal proof", "generate_now": False},
]


def write_freeze_report(rules: pd.DataFrame, deps: pd.DataFrame, figures: pd.DataFrame) -> None:
    text = "\n".join(
        [
            "# Stage 4B v2c Final Interpretation Freeze",
            "",
            "This file freezes the interpretation of corrected-v2b Stage 4B after the bounded v2c common-scale metabolomics feasibility audit.",
            "",
            "Final feasibility decision: **NO DEFENSIBLE COMMON-SCALE METABOLOMICS MATRIX AVAILABLE LOCALLY.**",
            "",
            "Adopted branch: **B. Freeze v2b as the as-supplied cross-platform stress-test run and restructure Stage 4C-E accordingly.**",
            "",
            "No historical corrected-v2 or v2b outputs are overwritten by this freeze.",
            "",
            "## Preserved Outputs",
            "",
            "\n".join(f"- `{item}`" for item in PRESERVED_OUTPUTS),
            "",
            "## Interpretation Rules",
            "",
            rules.to_markdown(index=False),
            "",
            "## Restructured Stage Dependency Map",
            "",
            deps.to_markdown(index=False),
            "",
            "## Figure Policy",
            "",
            figures.to_markdown(index=False),
            "",
            "## Thesis Wording",
            "",
            "- Metabolite identifiers could be harmonised, but assay measurement scales could not be verified as quantitatively harmonised.",
            "- Identifier harmonisation is not equivalent to quantitative harmonisation.",
            "- Source-fitted scaling was leakage-safe, but it cannot repair incompatible upstream LC-MS response scales.",
            "- v2b metabolome-only and early-integration external results should be described as stress tests under domain shift.",
            "- Microbiome-only reciprocal validation remains the principal directly transportable molecular Stage 4B analysis, with the Sinha/Yachida platform caveat.",
            "- Corrected numerical Results should not be rewritten until restructured Stage 4C-E analyses have been run by the user.",
            "",
        ]
    )
    (REPORTS / "STAGE4B_V2C_FINAL_INTERPRETATION_FREEZE.md").write_text(text)


def main() -> None:
    ensure_v2_dirs()
    rules = pd.DataFrame(INTERPRETATION_RULES)
    deps = pd.DataFrame(RESTRUCTURED_DEPENDENCIES)
    figures = pd.DataFrame(FIGURE_POLICY)
    rules.to_csv(TABLES / "stage4b_v2c_final_interpretation_rules.csv", index=False)
    deps.to_csv(TABLES / "stage4b_v2c_restructured_stage_dependency_map.csv", index=False)
    figures.to_csv(TABLES / "stage4b_v2c_figure_policy.csv", index=False)
    write_freeze_report(rules, deps, figures)
    atomic_json(
        {
            "status": "frozen",
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "final_decision": "NO_DEFENSIBLE_COMMON_SCALE_METABOLOMICS_MATRIX_AVAILABLE_LOCALLY",
            "adopted_branch": "B",
            "historical_outputs_overwritten": False,
            "preserved_outputs": PRESERVED_OUTPUTS,
            "interpretation_rules": INTERPRETATION_RULES,
            "restructured_dependencies": RESTRUCTURED_DEPENDENCIES,
            "figure_policy": FIGURE_POLICY,
        },
        MANIFESTS / "stage4b_v2c_final_interpretation_manifest.json",
    )
    print("STAGE4B_V2C_FINAL_INTERPRETATION_FROZEN")


if __name__ == "__main__":
    main()
