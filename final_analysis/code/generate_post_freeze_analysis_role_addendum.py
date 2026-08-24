"""Post-freeze re-framing addendum: microbiome-only and metabolome-only as
co-primary single-omic analyses (early integration remains the separate
integration comparator).

The 20-August frozen `FINAL_STAGE4_AUTHORITY_MANIFEST.json` and
`final_stage4_claim_policy.csv` describe Stage 4B as "microbiome principal,
metabolome/early-integration stress tests". That framing is being revised:
both microbiome-only and metabolome-only were, in fact, run through the
identical prespecified nested-CV/decision-score/bootstrap protocol on
matched participants with reciprocal external validation — the modelling
code never treated them asymmetrically. The asymmetry was only in how the
*results* were narratively hierarchised, because metabolomic cross-cohort
quantitative scale equivalence was never established. This script does NOT
change any numerical result and does NOT edit the frozen manifest or claim
policy CSV in place — it writes a superseding addendum, per the instruction
to keep the original freeze transparently intact rather than silently
rewritten.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from stage4_corrected_v2_common import MANIFESTS, atomic_csv, atomic_json


ANALYSIS_ROLES = {
    "MICROBIOME": {
        "analysis_role": "co_primary_single_omic",
        "external_interpretation_scope": "cross-cohort transportability of harmonised genus composition",
    },
    "METABOLOME": {
        "analysis_role": "co_primary_single_omic",
        "external_interpretation_scope": (
            "cross-cohort reproducibility/transportability of the matched metabolomic representation "
            "under the supplied cohort-specific processed matrices; cross-assay quantitative equivalence "
            "was not established (HMDB matching gives feature identity, not common units, response "
            "factors, or upstream transformations)"
        ),
    },
    "EARLY_INTEGRATION": {
        "analysis_role": "integration_comparison",
        "external_interpretation_scope": "whether combining the two co-primary molecular layers improves on either single-omic representation",
    },
}

CLAIM_POLICY_ADDENDUM_ROWS = [
    {"stage": "Stage4B", "policy": "Allowed", "claim_text": "Microbiome-only and metabolome-only were prespecified single-omic representations evaluated under the same modelling and external-validation protocol. In the final thesis reporting framework, they are treated as co-primary single-omic analyses, while the metabolomic analysis retains an additional cross-assay quantitative-scale limitation."},
    {"stage": "Stage4B", "policy": "Allowed", "claim_text": "Metabolome showed the reported external ROC-AUC/PR-AUC under the supplied cohort-specific processed matrices."},
    {"stage": "Stage4B", "policy": "Allowed", "claim_text": "Early integration is a separate comparative-integration analysis testing whether combining both co-primary molecular layers improves on either single-omic representation; it is not itself a third co-primary analysis."},
    {"stage": "Stage4B", "policy": "Disallowed", "claim_text": "Metabolomic values were quantitatively harmonised across assays."},
    {"stage": "Stage4B", "policy": "Disallowed", "claim_text": "Z-scoring made the metabolomic assays equivalent across cohorts."},
    {"stage": "Stage4B", "policy": "Disallowed", "claim_text": "Any single external ROC-AUC value (e.g. 0.846) demonstrates a clinically transportable metabolomic biomarker."},
    {"stage": "Stage4B", "policy": "Superseded", "claim_text": "SUPERSEDES the frozen Stage4B claim-policy row: 'Microbiome-only is the principal directly transportable molecular analysis; metabolome-only and early integration are assay-scale stress tests.' The original frozen row is retained unmodified in final_stage4_claim_policy.csv for historical transparency; this addendum is the current framing."},
]


def main() -> None:
    payload = {
        "status": "POST_FREEZE_ANALYSIS_ROLE_ADDENDUM",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "authority_freeze_timestamp_utc": "2026-08-20T08:47:40.049787+00:00",
        "does_not_modify_frozen_files": True,
        "reframing_rationale": (
            "Both MICROBIOME and METABOLOME representations already run through the identical "
            "prespecified nested-CV, decision-score-ranking, source-cross-fitted-threshold, and "
            "2000-resample bootstrap protocol on matched participants with reciprocal external "
            "validation (see run_stage4b_corrected_v2b.py: REPRESENTATIONS loop treats both "
            "symmetrically). 'Co-primary' does not mean identically interpretable: metabolomics "
            "retains an explicit cross-assay quantitative-scale caveat that microbiome composition "
            "does not, because HMDB feature-identity matching does not establish common units, "
            "response factors, or upstream transformations across cohorts."
        ),
        "analysis_roles": ANALYSIS_ROLES,
        "conceptual_hierarchy": {
            "co_primary_molecular_analyses": ["MICROBIOME", "METABOLOME"],
            "comparative_integration_analysis": ["EARLY_INTEGRATION"],
            "structural_descriptive_multiomics": ["MOFA", "sCCA"],
            "prior_knowledge_topology_experiment": ["GCN_vs_identity_vs_ridge"],
        },
    }
    atomic_json(payload, MANIFESTS / "POST_FREEZE_ANALYSIS_ROLE_ADDENDUM.json")
    atomic_csv(pd.DataFrame(CLAIM_POLICY_ADDENDUM_ROWS), MANIFESTS / "post_freeze_claim_policy_addendum.csv")
    print(f"Wrote {MANIFESTS / 'POST_FREEZE_ANALYSIS_ROLE_ADDENDUM.json'}")
    print(f"Wrote {MANIFESTS / 'post_freeze_claim_policy_addendum.csv'}")


if __name__ == "__main__":
    main()
