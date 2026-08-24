from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_stage2_modelling as crc_stage2

CRC_DIR = Path("/Users/bhaktidoshi/Downloads/Thesis Data/thesis_preprocessing_workflow/crc_pipeline")
STAGE2 = CRC_DIR / "stage2"
PATHWAY = STAGE2 / "tables" / "pathway"


def test_hypergeometric_wrapper_matches_scipy_when_available():
    observed = crc_stage2.hypergeom_right_tail(k=2, K=5, n=3, N=20)
    if crc_stage2.scipy_hypergeom is not None:
        expected = float(crc_stage2.scipy_hypergeom.sf(1, 20, 5, 3))
        assert abs(observed - expected) < 1e-12
    else:
        assert 0 <= observed <= 1


def test_crc_ora_outputs_are_cohort_separate_and_background_counts_valid():
    sinha = pd.read_csv(PATHWAY / "crc_ora_sinha_crc_vs_control.csv")
    yachida = pd.read_csv(PATHWAY / "crc_ora_yachida_crc_vs_healthy.csv")
    for df, cohort in [(sinha, "Sinha"), (yachida, "Yachida")]:
        if df.empty:
            continue
        assert set(df["cohort"]) == {cohort}
        assert (df["k_significant_compounds_in_pathway"] <= df["K_background_compounds_in_pathway"]).all()
        assert (df["k_significant_compounds_in_pathway"] <= df["n_significant_mapped_compounds"]).all()
        assert (df["K_background_compounds_in_pathway"] <= df["N_background_mapped_compounds"]).all()
        assert df["FDR_q_value"].between(0, 1).all()


def test_crc_ora_background_is_not_stage3_four_way_set():
    coverage = pd.read_csv(PATHWAY / "crc_ora_input_mapping_summary.csv")
    assert not coverage.empty
    assert not (coverage["unique_background_KEGG_compound_count"] == 56).all()
    assert coverage["background_definition"].str.contains("DA-tested metabolites", case=False).all()


def test_crc_ora_significant_inputs_are_from_crc_da_table():
    da = pd.read_csv(STAGE2 / "tables" / "crc_stage2_differential_abundance_results.csv")
    significant = da.loc[pd.to_numeric(da["BH_FDR"], errors="coerce") <= 0.05, ["cohort", "comparison", "canonical_metabolite"]].drop_duplicates()
    coverage = pd.read_csv(PATHWAY / "crc_ora_input_mapping_summary.csv")
    assert {"Sinha", "Yachida"}.issubset(set(coverage["cohort"]))
    assert int(coverage["significant_metabolites_BH_FDR_0_05"].sum()) >= len(significant)


def test_crc_ora_uses_local_resources_and_no_api_enrichment():
    resource = pd.read_csv(PATHWAY / "crc_ora_resource_audit.csv")
    assert resource["approved_for_analysis"].fillna(False).any()
    joined_notes = " ".join(resource["notes"].fillna("").astype(str))
    assert "no name/fuzzy/api" in joined_notes.lower() or "local" in joined_notes.lower()


def test_crc_ora_protected_outputs_unchanged():
    hashes = pd.read_csv(PATHWAY / "crc_ora_protected_output_hash_check.csv")
    existing = hashes[hashes["exists_before"].fillna(False) & hashes["exists_after"].fillna(False)]
    assert not existing.empty
    assert existing["hash_unchanged"].all()


def test_no_duplicate_compound_pathway_rows_distort_counts():
    all_results = pd.read_csv(PATHWAY / "crc_ora_all_results.csv")
    if all_results.empty:
        return
    for _, row in all_results.iterrows():
        compounds = [x for x in str(row["significant_KEGG_compounds"]).split(";") if x]
        assert len(compounds) == len(set(compounds))
        assert len(compounds) == int(row["k_significant_compounds_in_pathway"])
