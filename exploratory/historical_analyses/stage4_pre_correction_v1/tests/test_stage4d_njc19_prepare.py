from __future__ import annotations

import sys
import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path("/Users/bhaktidoshi/Downloads/Thesis Data/thesis_preprocessing_workflow/stage4_multiomics_integration/scripts")
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_stage4d_njc19 import (  # noqa: E402
    build_outputs,
    canonical_hmdb,
    canonical_kegg,
    discover_njc19_source,
    extract_candidate_relations,
    extract_json_relations,
    species_to_genus,
)


def test_species_to_genus_mapping() -> None:
    assert species_to_genus("Bacteroides fragilis") == "Bacteroides"
    assert species_to_genus("Faecalibacterium_prausnitzii") == "Faecalibacterium"


def test_kegg_hmdb_canonical_mapping_helpers() -> None:
    assert canonical_kegg("cpd:C00031") == "C00031"
    assert canonical_hmdb("met__HMDB:HMDB0000122") == "HMDB0000122"


def test_json_source_discovery(tmp_path) -> None:
    (tmp_path / "NJC19_network.json").write_text(json.dumps({}))
    (tmp_path / "NJC19_organism.json").write_text(json.dumps({"Bacteroides fragilis": {"name": "Bacteroides fragilis"}}))
    (tmp_path / "NJC19_compound.json").write_text(json.dumps({}))
    src = discover_njc19_source(tmp_path)
    assert "Bacteroides fragilis" in src["organism"]


def test_explicit_json_network_relation_parsing(tmp_path) -> None:
    (tmp_path / "NJC19_network.json").write_text(json.dumps({
        "NJC19_00001": {
            "Species": "Bacteroides fragilis",
            "Small-molecule metabolite or macromolecule": "D-Glucose (Glucose)",
            "Metabolic activity": "Consumption (import)",
            "Ref. #": ["1"],
        }
    }))
    (tmp_path / "NJC19_organism.json").write_text(json.dumps({"Bacteroides fragilis": {"name": "Bacteroides fragilis"}}))
    (tmp_path / "NJC19_compound.json").write_text(json.dumps({
        "D-Glucose (Glucose)": {"type": "small molecule", "name": ["D-Glucose", "Glucose"], "KEGG compound identifier": ["C00031"]}
    }))
    rel = extract_json_relations(tmp_path)
    assert len(rel) == 1
    assert rel.iloc[0]["canonical_genus"] == "Bacteroides"
    assert rel.iloc[0]["source_metabolite_id"] == "C00031"


def test_duplicate_relation_collapse_with_monkeypatched_features(monkeypatch) -> None:
    import prepare_stage4d_njc19 as njc19

    monkeypatch.setattr(
        njc19,
        "stage4_features",
        lambda: (
            pd.DataFrame([{"disease": "IBD", "canonical_genus": "Bacteroides"}, {"disease": "CRC", "canonical_genus": "Bacteroides"}]),
            pd.DataFrame([{"disease": "IBD", "HMDB_ID": "HMDB0000122"}, {"disease": "CRC", "HMDB_ID": "HMDB0000122"}]),
        ),
    )
    monkeypatch.setattr(njc19, "stage4_kegg_hmdb_crosswalk", lambda: pd.DataFrame([{"HMDB_ID": "HMDB0000122", "KEGG": "C00031"}]))
    rel = pd.DataFrame([
        {"source_file": "x.csv", "source_microbe_name": "Bacteroides fragilis", "canonical_genus": "Bacteroides", "source_metabolite_id": "C00031", "relation_type": "transport", "reference": "NJC19"},
        {"source_file": "x.csv", "source_microbe_name": "Bacteroides ovatus", "canonical_genus": "Bacteroides", "source_metabolite_id": "C00031", "relation_type": "transport", "reference": "NJC19"},
    ])
    _, _, edges, _, _ = build_outputs(rel)
    ibd = edges.loc[edges["disease"].eq("IBD")].iloc[0]
    assert ibd["support_count"] == 2
    assert len(edges.loc[edges["disease"].eq("IBD")]) == 1


def test_no_phenotype_leakage_columns_used() -> None:
    df = pd.DataFrame([
        {"species": "Bacteroides fragilis", "kegg": "C00031", "relation": "transport", "disease_label": "case", "cohort": "IBD"},
    ])
    rel = extract_candidate_relations([df])
    assert "disease_label" not in rel.columns
    assert "cohort" not in rel.columns
