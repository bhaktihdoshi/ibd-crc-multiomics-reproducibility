from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path("/Users/bhaktidoshi/Downloads/Thesis Data/thesis_preprocessing_workflow/stage4_multiomics_integration/scripts")
sys.path.insert(0, str(SCRIPT_DIR))

from augment_stage4d_graph_njc19 import final_validation, njc19_to_augmented_edges  # noqa: E402


def test_njc19_edges_convert_to_microbe_metabolite() -> None:
    df = pd.DataFrame([{"disease": "IBD", "canonical_genus": "Bacteroides", "HMDB_ID": "HMDB0000122", "source_microbe_names": "Bacteroides fragilis", "source_kegg_compound": "C00031", "relation_type": "Consumption (import)", "reference": "NJC19", "support_count": 1}])
    edges = njc19_to_augmented_edges(df)
    assert edges.iloc[0]["source_node"] == "microbe::Bacteroides"
    assert edges.iloc[0]["target_node"] == "metabolite::HMDB0000122"
    assert edges.iloc[0]["edge_type"] == "MICROBE_METABOLITE"
    assert edges.iloc[0]["weight"] == 1.0


def test_njc19_duplicate_edges_collapse() -> None:
    df = pd.DataFrame([
        {"disease": "IBD", "canonical_genus": "Bacteroides", "HMDB_ID": "HMDB0000122", "source_microbe_names": "a", "source_kegg_compound": "C00031", "relation_type": "Consumption", "reference": "NJC19", "support_count": 1},
        {"disease": "IBD", "canonical_genus": "Bacteroides", "HMDB_ID": "HMDB0000122", "source_microbe_names": "a", "source_kegg_compound": "C00031", "relation_type": "Consumption", "reference": "NJC19", "support_count": 1},
    ])
    assert len(njc19_to_augmented_edges(df)) == 1


def test_validation_flags_no_forbidden_source_terms() -> None:
    nodes = pd.DataFrame([
        {"node_id": "microbe::Bacteroides", "node_index": 0, "node_type": "microbe", "feature_id": "mic__Bacteroides", "feature_name": "", "disease_space": "IBD", "source_mapping_status": "x"},
        {"node_id": "metabolite::HMDB0000122", "node_index": 1, "node_type": "metabolite", "feature_id": "met__HMDB0000122", "feature_name": "", "disease_space": "IBD", "source_mapping_status": "x"},
    ])
    base_edges = pd.DataFrame(columns=["source_node", "target_node", "edge_type", "weight", "directionality", "knowledge_source", "evidence_type", "mapping_provenance", "disease_space", "supporting_pathway_ids", "supporting_pathway_names"])
    cross_edges = pd.DataFrame([{
        "source_node": "microbe::Bacteroides", "target_node": "metabolite::HMDB0000122", "edge_type": "MICROBE_METABOLITE", "weight": 1.0, "directionality": "undirected", "knowledge_source": "NJC19", "evidence_type": "explicit", "mapping_provenance": "mapping", "disease_space": "IBD", "supporting_pathway_ids": "", "supporting_pathway_names": "", "edge_source": "NJC19", "source_microbe_id": "Bacteroides fragilis", "source_metabolite_id": "C00031", "canonical_genus": "Bacteroides", "HMDB_ID": "HMDB0000122", "relation_type": "Consumption", "reference": "NJC19", "support_count": 1, "mapping_method": "explicit"
    }])
    augmented = cross_edges.copy()
    validation = final_validation(nodes, base_edges, cross_edges, augmented)
    assert validation.loc[validation["test"].eq("no_model_or_phenotype_edge_source"), "status"].iloc[0] == "pass"
