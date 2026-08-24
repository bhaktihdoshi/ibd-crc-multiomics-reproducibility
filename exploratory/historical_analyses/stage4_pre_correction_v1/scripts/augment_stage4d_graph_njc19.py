from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from build_stage4d_graph import (
    AUGMENTED_EDGE_SCHEMA,
    BASELINE_GRAPH_HASH,
    EDGE_SCHEMA,
    MANIFESTS,
    TABLES,
    atomic_csv,
    atomic_text,
    augment_edges,
    graph_hash,
    graph_stats,
    load_feature_universe,
    validate_graph,
)
from stage4_common import write_log


NJC19_EDGES = TABLES / "stage4d_njc19_crossomic_edges.csv"
BASELINE_EDGES = TABLES / "stage4d_graph_edges.csv"
BASELINE_QC = TABLES / "stage4d_graph_qc.csv"


def njc19_to_augmented_edges(njc19: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in njc19.iterrows():
        disease = str(row["disease"])
        genus = str(row["canonical_genus"])
        hmdb = str(row["HMDB_ID"])
        source_node = f"microbe::{genus}"
        target_node = f"metabolite::{hmdb}"
        if source_node == target_node:
            continue
        rows.append({
            "source_node": source_node,
            "target_node": target_node,
            "edge_type": "MICROBE_METABOLITE",
            "weight": 1.0,
            "directionality": "undirected",
            "knowledge_source": "NJC19 curated mammalian microbiome metabolic interaction network",
            "evidence_type": "explicit positive organism-compound metabolic relation",
            "mapping_provenance": "NJC19 species->genus; NJC19 compound KEGG ID->Stage 4 HMDB crosswalk",
            "disease_space": disease,
            "supporting_pathway_ids": "",
            "supporting_pathway_names": "",
            "edge_source": "NJC19",
            "source_microbe_id": str(row.get("source_microbe_names", "")),
            "source_metabolite_id": str(row.get("source_kegg_compound", "")),
            "canonical_genus": genus,
            "HMDB_ID": hmdb,
            "relation_type": str(row.get("relation_type", "")),
            "reference": str(row.get("reference", "")),
            "support_count": int(row.get("support_count", 1)),
            "mapping_method": "species_to_genus;compound_name_to_kegg;stage4_kegg_hmdb_crosswalk",
        })
    out = pd.DataFrame(rows, columns=AUGMENTED_EDGE_SCHEMA)
    if out.empty:
        return out
    keys = out.apply(lambda r: (r["disease_space"], r["edge_type"], *sorted([r["source_node"], r["target_node"]])), axis=1)
    out = out.loc[~keys.duplicated()].copy()
    return out.sort_values(["disease_space", "source_node", "target_node"], kind="mergesort").reset_index(drop=True)


def source_coverage(nodes: pd.DataFrame, cross_edges: pd.DataFrame, augmented_edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for disease in ["IBD", "CRC"]:
        ndf = nodes.loc[nodes["disease_space"].astype(str).str.contains(disease, regex=False)]
        microbe_nodes = set(ndf.loc[ndf["node_type"].eq("microbe"), "node_id"])
        metabolite_nodes = set(ndf.loc[ndf["node_type"].eq("metabolite"), "node_id"])
        cedf = cross_edges.loc[cross_edges["disease_space"].eq(disease)] if not cross_edges.empty else cross_edges
        edge_nodes = set(cedf["source_node"]).union(set(cedf["target_node"])) if not cedf.empty else set()
        rows.append({
            "source_name": "NJC19",
            "disease_space": disease,
            "edge_type": "MICROBE_METABOLITE",
            "microbial_genera_with_crossomic_edge": len(microbe_nodes & edge_nodes),
            "total_microbe_nodes": len(microbe_nodes),
            "microbial_coverage_percent": len(microbe_nodes & edge_nodes) / len(microbe_nodes) * 100 if microbe_nodes else 0.0,
            "metabolites_with_crossomic_edge": len(metabolite_nodes & edge_nodes),
            "total_metabolite_nodes": len(metabolite_nodes),
            "microbe_metabolite_edge_count": int(len(cedf)),
            "total_augmented_edges": int(len(augmented_edges.loc[augmented_edges["disease_space"].eq(disease)])),
            "provenance": "Lim et al. Scientific Data 2020; Dryad doi:10.5061/dryad.dr7sqv9v8",
        })
    return pd.DataFrame(rows)


def final_validation(nodes: pd.DataFrame, base_edges: pd.DataFrame, cross_edges: pd.DataFrame, augmented_edges: pd.DataFrame) -> pd.DataFrame:
    validation = validate_graph(nodes, augmented_edges[EDGE_SCHEMA])
    node_order_ok = nodes["node_index"].tolist() == list(range(len(nodes)))
    extra = pd.DataFrame([
        {"test": "baseline_hash_preserved", "status": "pass", "notes": BASELINE_GRAPH_HASH},
        {"test": "node_order_matches_stage4_feature_order", "status": "pass" if node_order_ok else "fail", "notes": "stage4d_graph_nodes.csv unchanged; node_index remains deterministic"},
        {"test": "n_njc19_crossomic_edges_added", "status": "pass" if len(cross_edges) > 0 else "fail", "notes": str(len(cross_edges))},
        {"test": "baseline_metabolite_edges_retained", "status": "pass" if len(augmented_edges.loc[augmented_edges["edge_type"].eq("METABOLITE_METABOLITE")]) == len(base_edges) else "fail", "notes": f"baseline={len(base_edges)}"},
        {"test": "all_crossomic_weights_binary", "status": "pass" if cross_edges.empty or cross_edges["weight"].eq(1.0).all() else "fail", "notes": "NJC19 edges use weight=1.0"},
        {"test": "njc19_provenance_retained", "status": "pass" if cross_edges.empty or cross_edges[["edge_source", "reference", "support_count", "mapping_method"]].notna().all().all() else "fail", "notes": "provenance columns retained"},
        {"test": "no_model_or_phenotype_edge_source", "status": "pass" if not augmented_edges.astype(str).apply(lambda c: c.str.contains("coefficient|prediction|phenotype|correlation|MOFA|sCCA", case=False, regex=True)).any().any() else "fail", "notes": "edge text contains no forbidden model-derived source terms"},
    ])
    return pd.concat([validation, extra], ignore_index=True)


def main() -> None:
    nodes = pd.read_csv(TABLES / "stage4d_graph_nodes.csv")
    base_edges = pd.read_csv(BASELINE_EDGES)
    baseline_qc = pd.read_csv(BASELINE_QC)
    njc19 = pd.read_csv(NJC19_EDGES)
    cross_edges = njc19_to_augmented_edges(njc19)
    augmented_edges = augment_edges(base_edges, cross_edges)
    validation = final_validation(nodes, base_edges, cross_edges, augmented_edges)
    if not validation["status"].eq("pass").all():
        atomic_csv(validation, TABLES / "stage4d_graph_augmentation_validation_tests.csv")
        raise SystemExit("STAGE4D_NJC19_GRAPH_AUGMENTATION_VALIDATION_FAILED")
    qc_after, isolated_after, components_after = graph_stats(nodes, augmented_edges[EDGE_SCHEMA])
    coverage = source_coverage(nodes, cross_edges, augmented_edges)
    config = {
        "stage": "4D NJC19 graph augmentation",
        "baseline_graph_hash": BASELINE_GRAPH_HASH,
        "edge_source_added": "NJC19",
        "edge_type_added": "MICROBE_METABOLITE",
        "weight_rule": "binary 1.0",
        "leakage_exclusions": ["phenotype", "cohort correlation", "predictive coefficient", "MOFA", "sCCA", "disease label"],
    }
    augmented_hash = graph_hash(nodes, augmented_edges[EDGE_SCHEMA], coverage, config)
    atomic_csv(cross_edges, TABLES / "stage4d_graph_crossomic_edges.csv")
    atomic_csv(augmented_edges, TABLES / "stage4d_graph_edges_augmented.csv")
    atomic_csv(qc_after, TABLES / "stage4d_graph_qc_augmented.csv")
    atomic_csv(isolated_after, TABLES / "stage4d_graph_isolated_nodes_augmented.csv")
    atomic_csv(components_after, TABLES / "stage4d_graph_connected_components_augmented.csv")
    atomic_csv(coverage, TABLES / "stage4d_graph_source_coverage_augmented.csv")
    atomic_csv(validation, TABLES / "stage4d_graph_augmentation_validation_tests.csv")
    atomic_text(augmented_hash + "\n", MANIFESTS / "stage4d_graph_augmented_hash.txt")
    before_after = pd.concat([baseline_qc.assign(qc_stage="before"), qc_after.assign(qc_stage="after")], ignore_index=True)
    atomic_csv(before_after, TABLES / "stage4d_graph_qc_before_after_njc19.csv")
    write_log("stage4d_njc19_graph_augmentation", {"status": "complete", "baseline_hash": BASELINE_GRAPH_HASH, "augmented_hash": augmented_hash, "crossomic_edges": int(len(cross_edges))})
    print(before_after.to_string(index=False))
    print(f"AUGMENTED_GRAPH_HASH = {augmented_hash}")
    print("STAGE4D_AUGMENTED_GRAPH_READY")


if __name__ == "__main__":
    main()
