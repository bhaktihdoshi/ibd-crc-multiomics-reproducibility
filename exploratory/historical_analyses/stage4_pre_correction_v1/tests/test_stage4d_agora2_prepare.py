from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path("/Users/bhaktidoshi/Downloads/Thesis Data/thesis_preprocessing_workflow/stage4_multiomics_integration/scripts")
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_stage4d_agora2 import (  # noqa: E402
    coverage_table,
    genus_panmodel_decision,
    genus_panmodel_coverage,
    genus_from_model_filename,
    model_name_from_filename,
    normalize,
    parse_agora2_index,
    parse_micromap_listing_text,
    panmodel_genus_from_filename,
    parse_size_to_mb,
)


def test_agora2_species_to_genus_mapping_from_filename() -> None:
    assert genus_from_model_filename("Bacteroides_fragilis_NCTC_9343.xml") == "Bacteroides"
    assert model_name_from_filename("Bacteroides_fragilis_NCTC_9343.xml") == "Bacteroides_fragilis_NCTC_9343"


def test_agora2_size_parser() -> None:
    assert parse_size_to_mb("8.0M") == 8.0
    assert round(parse_size_to_mb("1024K"), 3) == 1.0
    assert parse_size_to_mb("1G") == 1024.0


def test_agora2_index_parser_extracts_sbml_rows() -> None:
    html = '<a href="Acidaminococcus_fermentans_DSM_20731.xml">Acidaminococcus_fermentans_DSM_20731.xml</a></td><td>2023-03-23 15:01</td><td>9.3M</td>'
    df = parse_agora2_index(html)
    assert len(df) == 1
    assert df.iloc[0]["genus"] == "Acidaminococcus"
    assert df.iloc[0]["size_mb"] == 9.3


def test_agora2_coverage_uses_unique_genera() -> None:
    manifest = pd.DataFrame([
        {"canonical_genus": "A", "used_in_IBD": True, "used_in_CRC": False, "model_filename": "A_a.xml", "match_status": "MATCHED_AGORA2_MODEL"},
        {"canonical_genus": "A", "used_in_IBD": True, "used_in_CRC": False, "model_filename": "A_b.xml", "match_status": "MATCHED_AGORA2_MODEL"},
        {"canonical_genus": "B", "used_in_IBD": False, "used_in_CRC": True, "model_filename": "", "match_status": "NO_AGORA2_MODEL_FOR_GENUS"},
    ])
    cov = coverage_table(manifest)
    ibd = cov.loc[cov["space"].eq("IBD")].iloc[0]
    assert ibd["genera_represented_in_AGORA2"] == 1
    assert ibd["matched_model_files"] == 2


def test_agora2_normalization_is_safe() -> None:
    assert normalize("Bacteroides_fragilis") == "bacteroides fragilis"


def test_micromap_panmodel_filename_to_genus() -> None:
    assert panmodel_genus_from_filename("pan_Bacteroides_genus.xml") == "Bacteroides"
    assert panmodel_genus_from_filename("Bacteroides_genus_panmodel.xml") == "Bacteroides"


def test_micromap_listing_parser_keeps_only_candidate_pan_xml() -> None:
    text = "pan_Bacteroides_genus.xml strain_level_model.xml AGORA2_pan_Faecalibacterium.xml notes.pdf"
    df = parse_micromap_listing_text(text)
    assert set(df["canonical_genus_candidate"]) == {"Bacteroides", "Faecalibacterium"}
    assert df["schema_sufficiency_status"].eq("UNINSPECTED_FILE_SCHEMA").all()


def test_genus_panmodel_coverage_counts_unique_genera() -> None:
    manifest = pd.DataFrame([
        {"canonical_genus": "A", "used_in_IBD": True, "used_in_CRC": False, "panmodel_file": "pan_A.xml", "match_status": "MATCHED_MICROMAP_GENUS_PANMODEL"},
        {"canonical_genus": "A", "used_in_IBD": True, "used_in_CRC": False, "panmodel_file": "pan_A_duplicate.xml", "match_status": "MATCHED_MICROMAP_GENUS_PANMODEL"},
        {"canonical_genus": "B", "used_in_IBD": False, "used_in_CRC": True, "panmodel_file": "", "match_status": "NO_MICROMAP_GENUS_PANMODEL_FOR_GENUS"},
    ])
    cov = genus_panmodel_coverage(manifest)
    ibd = cov.loc[cov["space"].eq("IBD")].iloc[0]
    assert ibd["genera_represented_in_MicroMap_panmodels"] == 1
    assert ibd["required_genus_panmodel_files"] == 2


def test_panmodel_decision_requires_accessible_size_metadata() -> None:
    manifest = pd.DataFrame([
        {"canonical_genus": "A", "used_in_IBD": True, "used_in_CRC": True, "panmodel_file": "pan_A.xml", "match_status": "MATCHED_MICROMAP_GENUS_PANMODEL"},
    ])
    cov = pd.DataFrame([{
        "space": "Union",
        "stage4_genera": 1,
        "genera_represented_in_MicroMap_panmodels": 1,
        "expected_taxonomic_coverage_percent": 100.0,
        "required_genus_panmodel_files": 1,
        "estimated_download_size": "unresolved from accessible metadata",
    }])
    assert genus_panmodel_decision(manifest, cov) == "AGORA2_GENUS_PANMODEL_NOT_ACCESSIBLE"
