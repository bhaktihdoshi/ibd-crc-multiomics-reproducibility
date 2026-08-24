from __future__ import annotations

from pathlib import Path

import pandas as pd

STAGE3 = Path(
    "/Users/bhaktidoshi/Downloads/Thesis Data/thesis_preprocessing_workflow/stage3_ibd_crc_integration"
)


def test_strict_four_way_overlap_is_preserved_at_56_hmdb_ids():
    strict = pd.read_csv(STAGE3 / "tables" / "stage3_four_cohort_shared_metabolites.csv")
    assert len(strict) == 56
    assert strict["hmdb_id"].is_unique
    assert strict["hmdb_id"].str.match(r"^HMDB\d+$").all()


def test_expanded_high_confidence_contains_strict_set_and_no_ambiguous_rows():
    strict = pd.read_csv(STAGE3 / "tables" / "stage3_four_cohort_shared_metabolites.csv")
    expanded = pd.read_csv(
        STAGE3 / "tables" / "stage3_four_cohort_shared_metabolites_expanded_high_confidence.csv"
    )

    assert set(strict["hmdb_id"]) <= set(expanded["canonical_hmdb"])
    assert expanded["canonical_entity_id"].is_unique
    assert set(expanded["match_basis"]).issubset(
        {"STRICT_HMDB_MATCH", "DETERMINISTIC_RESCUED_MATCH"}
    )
    assert not expanded["match_basis"].str.contains("NAME", case=False, na=False).any()
    assert not expanded["ambiguity_flag"].fillna(False).astype(bool).any()


def test_expanded_rows_are_linked_to_all_four_cohorts():
    expanded = pd.read_csv(
        STAGE3 / "tables" / "stage3_four_cohort_shared_metabolites_expanded_high_confidence.csv"
    )
    cohort_feature_columns = [
        "franzosa_feature",
        "ihmp_feature",
        "sinha_feature",
        "yachida_feature",
    ]
    for column in cohort_feature_columns:
        assert column in expanded.columns
        assert expanded[column].notna().all()


def test_name_matches_are_candidates_only_and_not_primary_expanded_features():
    name_candidates = pd.read_csv(STAGE3 / "tables" / "stage3_name_match_candidates.csv")
    expanded = pd.read_csv(
        STAGE3 / "tables" / "stage3_four_cohort_shared_metabolites_expanded_high_confidence.csv"
    )

    assert "match_basis" in expanded.columns
    assert not expanded["match_basis"].str.contains("name", case=False, na=False).any()
    if not name_candidates.empty:
        assert "status" in name_candidates.columns
        assert not name_candidates["status"].eq("DETERMINISTIC_NAME_ONLY_PRIMARY").any()


def test_ambiguous_identifier_mappings_are_excluded_from_expanded_set():
    ambiguity = pd.read_csv(STAGE3 / "tables" / "stage3_mapping_ambiguity_audit.csv")
    expanded = pd.read_csv(
        STAGE3 / "tables" / "stage3_four_cohort_shared_metabolites_expanded_high_confidence.csv"
    )

    ambiguous_hmdb = set(ambiguity.get("hmdb_id", pd.Series(dtype=str)).dropna())
    ambiguous_kegg = set(ambiguity.get("kegg_id", pd.Series(dtype=str)).dropna())
    assert not (set(expanded["canonical_hmdb"].dropna()) & ambiguous_hmdb)
    assert not (set(expanded["canonical_kegg"].dropna()) & ambiguous_kegg)


def test_structural_rescue_reports_unavailable_structural_identifiers_when_absent():
    structural = pd.read_csv(STAGE3 / "tables" / "stage3_structural_identifier_rescue_audit.csv")
    assert "status" in structural.columns
    assert set(structural["status"]).issubset(
        {
            "UNRESOLVED_NO_LOCAL_STRUCTURAL_IDENTIFIER",
            "DETERMINISTIC_RESCUED_MATCH",
            "AMBIGUOUS_CANDIDATE",
            "STRICT_HMDB_MATCH",
        }
    )


def test_mapping_summary_matches_primary_and_expanded_tables():
    strict = pd.read_csv(STAGE3 / "tables" / "stage3_four_cohort_shared_metabolites.csv")
    expanded = pd.read_csv(
        STAGE3 / "tables" / "stage3_four_cohort_shared_metabolites_expanded_high_confidence.csv"
    )
    summary = pd.read_csv(STAGE3 / "tables" / "stage3_mapping_expansion_summary.csv").iloc[0]

    assert int(summary["strict_four_way_hmdb_overlap"]) == len(strict)
    assert int(summary["total_expanded_high_confidence_overlap"]) == len(expanded)
    assert int(summary["absolute_increase"]) == len(expanded) - len(strict)
