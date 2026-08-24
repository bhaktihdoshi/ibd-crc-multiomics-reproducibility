from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path("/Users/bhaktidoshi/Downloads/Thesis Data/thesis_preprocessing_workflow/stage4_multiomics_integration/scripts")
sys.path.insert(0, str(SCRIPT_DIR))

from build_stage4d_graph import (  # noqa: E402
    BASELINE_GRAPH_HASH,
    EDGE_SCHEMA,
    AUGMENTED_EDGE_SCHEMA,
    NODE_SCHEMA,
    augment_edges,
    build_crossomic_edges,
    build_metabolite_edges,
    canonical_hmdb,
    infer_genus,
    is_relation_count_column,
    split_multi_value,
    graph_hash,
    graph_stats,
    source_audit,
    synthetic_graph_fixture,
)


def test_stage4d_synthetic_node_ids_are_unique_and_ordered() -> None:
    nodes, _, _ = synthetic_graph_fixture()
    assert list(nodes.columns) == NODE_SCHEMA
    assert not nodes.duplicated(["disease_space", "node_id"]).any()
    assert nodes["node_index"].tolist() == [0, 1, 2]
    assert set(nodes["node_type"]) == {"microbe", "metabolite"}


def test_stage4d_builds_only_valid_prior_metabolite_edges() -> None:
    nodes, mapping, pathways = synthetic_graph_fixture()
    edges = build_metabolite_edges(nodes, mapping, pathways, {"map00010": "Glycolysis / Gluconeogenesis"})
    assert list(edges.columns) == EDGE_SCHEMA
    assert len(edges) == 1
    assert edges.iloc[0]["edge_type"] == "METABOLITE_METABOLITE"
    assert edges.iloc[0]["directionality"] == "undirected"
    assert float(edges.iloc[0]["weight"]) == 1.0
    assert edges.iloc[0]["source_node"] != edges.iloc[0]["target_node"]


def test_stage4d_no_duplicate_edges_after_pathway_union() -> None:
    nodes, mapping, pathways = synthetic_graph_fixture()
    pathways = pd.concat([pathways, pathways.assign(pathway_id="map99999")], ignore_index=True)
    edges = build_metabolite_edges(nodes, mapping, pathways, {"map00010": "A", "map99999": "B"})
    assert len(edges) == 1
    assert "map00010" in edges.iloc[0]["supporting_pathway_ids"]
    assert "map99999" in edges.iloc[0]["supporting_pathway_ids"]


def test_stage4d_invalid_endpoints_not_created() -> None:
    nodes, mapping, pathways = synthetic_graph_fixture()
    mapping = mapping.loc[mapping["hmdb_id"].ne("HMDB0000002")]
    edges = build_metabolite_edges(nodes, mapping, pathways, {})
    assert edges.empty


def test_stage4d_graph_hash_is_deterministic() -> None:
    nodes, mapping, pathways = synthetic_graph_fixture()
    edges = build_metabolite_edges(nodes, mapping, pathways, {"map00010": "A"})
    sources = source_audit(nodes, mapping, pathways)
    config = {"graph_design": "synthetic", "edge_weight_rule": "binary"}
    assert graph_hash(nodes, edges, sources, config) == graph_hash(nodes, edges, sources, config)


def test_stage4d_isolated_and_component_accounting() -> None:
    nodes, mapping, pathways = synthetic_graph_fixture()
    edges = build_metabolite_edges(nodes, mapping, pathways, {"map00010": "A"})
    qc, isolated, components = graph_stats(nodes, edges)
    assert int(qc.iloc[0]["isolated_node_count"]) == 1
    assert isolated.iloc[0]["node_id"] == "microbe::A"
    assert sorted(components["component_size"].tolist()) == [1, 2]


def test_stage4d_graph_schema_has_no_phenotype_leakage_fields() -> None:
    forbidden = {"label", "outcome", "phenotype", "prediction", "coefficient", "fold", "auc", "disease_group"}
    graph_columns = {c.lower() for c in NODE_SCHEMA + EDGE_SCHEMA}
    assert forbidden.isdisjoint(graph_columns)


def test_stage4d_hmdb_mapping_requires_identifier() -> None:
    assert canonical_hmdb("HMDB:HMDB0000001") == "HMDB0000001"
    assert canonical_hmdb("met__HMDB:HMDB0000042") == "HMDB0000042"
    assert canonical_hmdb("butyrate") is None


def test_stage4d_embedded_list_field_parsing() -> None:
    assert split_multi_value("HMDB0000001; HMDB0000002|not available") == ["HMDB0000001", "HMDB0000002"]


def test_stage4d_mimedb_relation_count_column_is_not_treated_as_taxon() -> None:
    assert is_relation_count_column("microbe_relations")
    assert not is_relation_count_column("microbe_name")


def test_stage4d_species_to_genus_collapse_is_deterministic() -> None:
    genus, ambiguous, method = infer_genus("Bacteroides fragilis strain X")
    assert genus == "Bacteroides"
    assert not ambiguous
    assert method == "species_or_name_to_genus"


def test_stage4d_ambiguous_taxa_are_flagged() -> None:
    genus, ambiguous, _ = infer_genus("uncultured bacterium")
    assert genus is None
    assert ambiguous


def test_stage4d_crossomic_edges_keep_provenance_and_binary_weights() -> None:
    nodes, mapping, pathways = synthetic_graph_fixture()
    base = build_metabolite_edges(nodes, mapping, pathways, {"map00010": "A"})
    mimedb = pd.DataFrame([
        {
            "source": "MIMEDB",
            "source_microbe_id": "m1",
            "source_metabolite_id": "c1",
            "source_microbe_name": "A species",
            "source_metabolite_name": "m1",
            "canonical_genus": "A",
            "HMDB_ID": "HMDB0000001",
            "relation_type": "reaction",
            "evidence_type": "MiMeDB curated microbial metabolome record",
            "reference": "ref",
            "support_count": 1,
            "mapping_method": "species_or_name_to_genus;direct_hmdb_identifier",
            "ambiguous_taxon": False,
            "ambiguous_metabolite": False,
            "status": "mapped",
        }
    ])
    cross, annotated = build_crossomic_edges(nodes, mimedb)
    assert list(cross.columns) == AUGMENTED_EDGE_SCHEMA
    assert len(cross) == 1
    assert cross.iloc[0]["edge_type"] == "MICROBE_METABOLITE"
    assert cross.iloc[0]["edge_source"] == "MIMEDB"
    assert float(cross.iloc[0]["weight"]) == 1.0
    combined = augment_edges(base, cross)
    assert set(combined["edge_type"]) == {"METABOLITE_METABOLITE", "MICROBE_METABOLITE"}
    assert annotated.iloc[0]["included_disease_spaces"] == "IBD"


def test_stage4d_crossomic_duplicate_edges_collapse_support() -> None:
    nodes, _, _ = synthetic_graph_fixture()
    mimedb = pd.DataFrame([
        {
            "source": "MIMEDB",
            "source_microbe_id": source_id,
            "source_metabolite_id": "c1",
            "source_microbe_name": "A species",
            "source_metabolite_name": "m1",
            "canonical_genus": "A",
            "HMDB_ID": "HMDB0000001",
            "relation_type": "reaction",
            "evidence_type": "MiMeDB curated microbial metabolome record",
            "reference": "",
            "support_count": 1,
            "mapping_method": "species_or_name_to_genus;direct_hmdb_identifier",
            "ambiguous_taxon": False,
            "ambiguous_metabolite": False,
            "status": "mapped",
        }
        for source_id in ["m1", "m2"]
    ])
    cross, _ = build_crossomic_edges(nodes, mimedb)
    assert len(cross) == 1
    assert int(cross.iloc[0]["support_count"]) == 2


def test_stage4d_augmented_hash_deterministic_and_baseline_constant_available() -> None:
    nodes, mapping, pathways = synthetic_graph_fixture()
    base = build_metabolite_edges(nodes, mapping, pathways, {"map00010": "A"})
    sources = source_audit(nodes, mapping, pathways)
    config = {"baseline_graph_hash": BASELINE_GRAPH_HASH, "augmentation_source": "MIMEDB"}
    assert BASELINE_GRAPH_HASH == "d423255659a9f43069cc14885e1a2e2608d66d75bcaeabafd7a103c6532fb27b"
    assert graph_hash(nodes, base, sources, config) == graph_hash(nodes, base, sources, config)
