from __future__ import annotations

import argparse
import json
import re
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage4_corrected_v2_common import ROOT, STAGE4, TABLES, REPORTS, atomic_csv
from run_stage4d_corrected_v2_graph import (
    BIO_EDGE_SCHEMA,
    DISEASES,
    GRAPH_MODES,
    NJC19_EDGES,
    canonical_hmdb,
    canonical_kegg,
)


NJC19_EXTRACTED = STAGE4 / "data" / "stage4d_prior_sources" / "njc19" / "extracted"
NJC19_NETWORK = NJC19_EXTRACTED / "NJC19_network.json"
NJC19_COMPOUND = NJC19_EXTRACTED / "NJC19_compound.json"

NODES = TABLES / "stage4d_corrected_v2_nodes.csv"
BIO_EDGES = TABLES / "stage4d_corrected_v2_biological_edges.csv"
GRAPH_SUMMARY = TABLES / "stage4d_corrected_v2_graph_summary.csv"
VALIDATION = TABLES / "stage4d_corrected_v2_validation_tests.csv"

IBD_HMDB_KEGG_AUDIT = ROOT / "ibd_pipeline" / "tables" / "biological_primary_expanded" / "pathway" / "hmdb_to_kegg_mapping_audit.csv"
CRC_HMDB_KEGG_CROSSWALK = ROOT / "crc_pipeline" / "stage2" / "tables" / "crc_stage2_hmdb_kegg_crosswalk.csv"

BROAD_KEGG_IDS = {
    "map01100": "Metabolic pathways",
    "map01110": "Biosynthesis of secondary metabolites",
    "map01120": "Microbial metabolism in diverse environments",
    "map01200": "Carbon metabolism",
    "map01210": "2-Oxocarboxylic acid metabolism",
    "map01212": "Fatty acid metabolism",
    "map01230": "Biosynthesis of amino acids",
    "map01232": "Nucleotide metabolism",
}


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def species_to_genus(value: object) -> str:
    text = re.sub(r"[_|;]+", " ", str(value)).strip()
    text = re.sub(r"\s+", " ", text)
    return text.split(" ", 1)[0] if text else ""


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def compound_name_to_kegg(compounds: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for display_name, payload in compounds.items():
        ids = payload.get("KEGG compound identifier", []) if isinstance(payload, dict) else []
        kegg = next((canonical_kegg(x) for x in ids if canonical_kegg(x)), None)
        if not kegg:
            continue
        names = [display_name]
        if isinstance(payload, dict):
            raw = payload.get("name", [])
            names.extend(raw if isinstance(raw, list) else [raw])
        for name in names:
            mapping[normalize(name)] = kegg
    return mapping


def load_kegg_hmdb_crosswalk() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    if IBD_HMDB_KEGG_AUDIT.exists():
        ibd = pd.read_csv(IBD_HMDB_KEGG_AUDIT, dtype=str)
        for _, row in ibd.iterrows():
            hmdb = canonical_hmdb(row.get("HMDB_ID"))
            kegg = canonical_kegg(row.get("final_KEGG_Compound_ID")) or canonical_kegg(row.get("supplied_KEGG_Compound_ID"))
            if hmdb and kegg:
                rows.append({"HMDB_ID": hmdb, "KEGG": kegg, "source": str(IBD_HMDB_KEGG_AUDIT)})
    if CRC_HMDB_KEGG_CROSSWALK.exists():
        crc = pd.read_csv(CRC_HMDB_KEGG_CROSSWALK, dtype=str)
        for _, row in crc.iterrows():
            hmdb = canonical_hmdb(row.get("HMDB") or row.get("canonical_id"))
            kegg = canonical_kegg(row.get("KEGG"))
            if hmdb and kegg:
                rows.append({"HMDB_ID": hmdb, "KEGG": kegg, "source": str(CRC_HMDB_KEGG_CROSSWALK)})
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=["HMDB_ID", "KEGG", "source"])


def extract_original_njc19_relations() -> pd.DataFrame:
    network = load_json(NJC19_NETWORK)
    compounds = load_json(NJC19_COMPOUND)
    compound_map = compound_name_to_kegg(compounds)
    rows: list[dict[str, Any]] = []
    for relation_id, payload in network.items():
        if not isinstance(payload, dict):
            continue
        species = str(payload.get("Species", "")).strip()
        compound = str(payload.get("Small-molecule metabolite or macromolecule", "")).strip()
        relation = str(payload.get("Metabolic activity", "")).strip()
        relation_norm = normalize(relation)
        valid_relation = any(term in relation_norm for term in ["production", "consumption", "transport", "import", "export"])
        genus = species_to_genus(species)
        kegg = compound_map.get(normalize(compound))
        rows.append(
            {
                "source_relation_id": relation_id,
                "source_species": species,
                "canonical_genus": genus,
                "source_metabolite_name": compound,
                "source_kegg_compound": kegg or "",
                "relation_type": relation,
                "valid_relation_type": bool(valid_relation),
                "parseable_microbe": bool(species),
                "parseable_genus": bool(genus),
                "parseable_metabolite_name": bool(compound),
                "mapped_to_kegg": bool(kegg),
            }
        )
    return pd.DataFrame(rows)


def classify_relation_type(value: object) -> str:
    text = normalize(value)
    if "production" in text or "export" in text:
        return "production_or_export"
    if "consumption" in text or "import" in text:
        return "consumption_or_import"
    if "transport" in text:
        return "transport"
    return "other"


def njc19_provenance(nodes: pd.DataFrame, bio: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original = extract_original_njc19_relations()
    crosswalk = load_kegg_hmdb_crosswalk()
    prepared = pd.read_csv(NJC19_EDGES) if NJC19_EDGES.exists() else pd.DataFrame()
    original_mapped = original.merge(crosswalk, left_on="source_kegg_compound", right_on="KEGG", how="left")
    rows = []
    for disease in DISEASES:
        disease_nodes = nodes[nodes["disease"].eq(disease)]
        genera = set(disease_nodes.loc[disease_nodes["node_type"].eq("microbe"), "node_id"].str.replace("microbe::", "", regex=False))
        hmdb = set(disease_nodes.loc[disease_nodes["node_type"].eq("metabolite"), "node_id"].str.replace("metabolite::", "", regex=False))
        valid = original_mapped[original_mapped["valid_relation_type"]].copy()
        mapped_genus = valid[valid["canonical_genus"].isin(genera)].copy()
        mapped_met = valid[valid["HMDB_ID"].isin(hmdb)].copy()
        both = valid[valid["canonical_genus"].isin(genera) & valid["HMDB_ID"].isin(hmdb)].copy()
        final = bio[(bio["disease"].eq(disease)) & bio["graph_mode"].eq("kegg_njc19") & bio["edge_type"].eq("MICROBE_METABOLITE")]
        prepared_disease = prepared[prepared["disease"].eq(disease)] if "disease" in prepared.columns else prepared.iloc[0:0]
        rows.append(
            {
                "disease": disease,
                "earliest_auditable_input": str(NJC19_NETWORK) if NJC19_NETWORK.exists() else str(NJC19_EDGES),
                "original_njc19_rows": int(len(original)),
                "unique_original_relations": int(original["source_relation_id"].nunique()) if not original.empty else 0,
                "valid_production_consumption_transport_records": int(len(valid)),
                "records_with_parseable_microbe_name": int(original["parseable_microbe"].sum()) if not original.empty else 0,
                "records_mapped_species_to_canonical_genus": int(valid["parseable_genus"].sum()) if not valid.empty else 0,
                "records_with_parseable_metabolite_name": int(original["parseable_metabolite_name"].sum()) if not original.empty else 0,
                "records_mapped_to_kegg": int(valid["mapped_to_kegg"].sum()) if not valid.empty else 0,
                "records_mapped_to_hmdb": int(valid["HMDB_ID"].notna().sum()) if not valid.empty else 0,
                "records_retained_in_ibd_feature_space": int(len(both)) if disease == "IBD" else np.nan,
                "records_retained_in_crc_feature_space": int(len(both)) if disease == "CRC" else np.nan,
                "prepared_candidate_rows_for_disease": int(len(prepared_disease)),
                "duplicate_relations_collapsed": int(max(len(both) - len(final), 0)),
                "final_unique_graph_edges": int(len(final)),
                "interpretation": "current Stage 4D source_records in graph attrition was disease-filtered prepared candidate count, not total original NJC19 network size",
            }
        )
    attr = pd.DataFrame(rows)
    relation_counts = (
        prepared.assign(relation_class=prepared["relation_type"].map(classify_relation_type))
        .groupby(["disease", "relation_class"], dropna=False)
        .size()
        .reset_index(name="n_prepared_candidate_records")
        if not prepared.empty
        else pd.DataFrame(columns=["disease", "relation_class", "n_prepared_candidate_records"])
    )
    retained = bio[(bio["graph_mode"].eq("kegg_njc19")) & bio["edge_type"].eq("MICROBE_METABOLITE")].copy()
    support = retained[["disease", "source_node", "target_node", "relation_type", "source_microbe_names", "support_count"]].copy()
    support["source_species_count"] = support["source_microbe_names"].fillna("").map(lambda x: len([v for v in str(x).split(";") if v]))
    species_by_genus = (
        prepared.groupby(["disease", "canonical_genus"])["source_microbe_names"]
        .agg(lambda s: len(set(";".join(s.astype(str)).split(";"))))
        .reset_index(name="source_species_collapsed_to_genus")
        if not prepared.empty
        else pd.DataFrame(columns=["disease", "canonical_genus", "source_species_collapsed_to_genus"])
    )
    ambiguous = pd.DataFrame(
        [
            {
                "source": "NJC19",
                "ambiguous_genus_mappings": int((original["canonical_genus"].astype(str).eq("")).sum()) if not original.empty else 0,
                "ambiguous_metabolite_mappings": int((original["source_kegg_compound"].astype(str).eq("")).sum()) if not original.empty else 0,
                "note": "empty canonical genus or KEGG mapping in original NJC19 parse; no fuzzy mapping added in corrected-v2 graph",
            }
        ]
    )
    return attr, relation_counts, support, species_by_genus, ambiguous


def degree_tables(nodes: pd.DataFrame, bio: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    degree_rows: list[dict[str, Any]] = []
    for disease in DISEASES:
        ndf = nodes[nodes["disease"].eq(disease)]
        for mode in GRAPH_MODES:
            edf = bio[(bio["disease"].eq(disease)) & bio["graph_mode"].eq(mode)]
            degree = {node: 0 for node in ndf["node_id"]}
            for _, edge in edf.iterrows():
                degree[edge["source_node"]] += 1
                degree[edge["target_node"]] += 1
            for _, node in ndf.iterrows():
                degree_rows.append(
                    {
                        "disease": disease,
                        "graph_mode": mode,
                        "node_id": node["node_id"],
                        "node_type": node["node_type"],
                        "degree": degree[node["node_id"]],
                    }
                )
    degree_df = pd.DataFrame(degree_rows)
    summary = (
        degree_df.groupby(["disease", "graph_mode", "node_type"])["degree"]
        .agg(
            n_nodes="count",
            median_degree="median",
            q1_degree=lambda x: float(np.percentile(x, 25)),
            q3_degree=lambda x: float(np.percentile(x, 75)),
            p95_degree=lambda x: float(np.percentile(x, 95)),
            max_degree="max",
        )
        .reset_index()
    )
    summary["iqr_degree"] = summary["q3_degree"] - summary["q1_degree"]
    top20 = (
        degree_df.sort_values(["disease", "graph_mode", "degree", "node_id"], ascending=[True, True, False, True], kind="mergesort")
        .groupby(["disease", "graph_mode"], as_index=False)
        .head(20)
        .reset_index(drop=True)
    )
    return degree_df, summary, top20


def kegg_pathway_diagnostics(bio: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    kegg = bio[bio["edge_type"].eq("METABOLITE_METABOLITE")].copy()
    pathway_rows: list[dict[str, Any]] = []
    for _, edge in kegg.iterrows():
        ids = [x for x in str(edge["supporting_pathway_ids"]).split(";") if x]
        names = [x for x in str(edge["supporting_pathway_names"]).split(";") if x]
        for i, pid in enumerate(ids):
            pathway_rows.append(
                {
                    "disease": edge["disease"],
                    "graph_mode": edge["graph_mode"],
                    "pathway_id": pid,
                    "pathway_name": names[i] if i < len(names) else pid,
                    "source_node": edge["source_node"],
                    "target_node": edge["target_node"],
                    "edge_key": "|".join(sorted([edge["source_node"], edge["target_node"]])),
                    "is_global_or_overview_pathway": pid in BROAD_KEGG_IDS or "global" in normalize(names[i] if i < len(names) else ""),
                }
            )
    pe = pd.DataFrame(pathway_rows)
    if pe.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    contrib = (
        pe.groupby(["disease", "graph_mode", "pathway_id", "pathway_name", "is_global_or_overview_pathway"])
        .agg(
            edge_contributions=("edge_key", "nunique"),
            retained_metabolites=("source_node", lambda s: len(set(s).union(set(pe.loc[s.index, "target_node"])))),
        )
        .reset_index()
    )
    support = (
        kegg.assign(pathway_support_count=kegg["supporting_pathway_ids"].fillna("").map(lambda x: len([v for v in str(x).split(";") if v])))
        .groupby(["disease", "graph_mode", "pathway_support_count"])
        .size()
        .reset_index(name="n_edges")
    )
    support_total = support.groupby(["disease", "graph_mode"])["n_edges"].transform("sum")
    support["proportion_edges"] = support["n_edges"] / support_total
    one_path = (
        kegg.assign(pathway_support_count=kegg["supporting_pathway_ids"].fillna("").map(lambda x: len([v for v in str(x).split(";") if v])))
        .groupby(["disease", "graph_mode"])
        .agg(
            edges_supported_by_one_pathway=("pathway_support_count", lambda x: int((x == 1).sum())),
            total_metabolite_edges=("pathway_support_count", "count"),
        )
        .reset_index()
    )
    one_path["proportion_edges_supported_by_one_pathway"] = one_path["edges_supported_by_one_pathway"] / one_path["total_metabolite_edges"]
    broad = (
        pe[pe["is_global_or_overview_pathway"]]
        .groupby(["disease", "graph_mode", "pathway_id", "pathway_name"])
        .agg(broad_pathway_edge_contributions=("edge_key", "nunique"))
        .reset_index()
    )
    return contrib, support.merge(one_path, on=["disease", "graph_mode"], how="left"), broad


def metabolite_density(nodes: pd.DataFrame, bio: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for disease in DISEASES:
        n_met = int(nodes[(nodes["disease"].eq(disease)) & nodes["node_type"].eq("metabolite")].shape[0])
        possible = n_met * (n_met - 1) / 2
        for mode in GRAPH_MODES:
            n_edges = int(bio[(bio["disease"].eq(disease)) & bio["graph_mode"].eq(mode) & bio["edge_type"].eq("METABOLITE_METABOLITE")].shape[0])
            rows.append({"disease": disease, "graph_mode": mode, "metabolite_nodes": n_met, "metabolite_metabolite_edges": n_edges, "metabolite_subgraph_density": n_edges / possible if possible else 0.0})
    return pd.DataFrame(rows)


def component_distribution(nodes: pd.DataFrame, bio: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for disease in DISEASES:
        node_ids = nodes[nodes["disease"].eq(disease)]["node_id"].tolist()
        for mode in GRAPH_MODES:
            edf = bio[(bio["disease"].eq(disease)) & bio["graph_mode"].eq(mode)]
            adj = {node: set() for node in node_ids}
            for _, edge in edf.iterrows():
                adj[edge["source_node"]].add(edge["target_node"])
                adj[edge["target_node"]].add(edge["source_node"])
            seen = set()
            components = []
            for node in node_ids:
                if node in seen:
                    continue
                queue = deque([node])
                seen.add(node)
                comp = []
                while queue:
                    cur = queue.popleft()
                    comp.append(cur)
                    for nxt in adj[cur]:
                        if nxt not in seen:
                            seen.add(nxt)
                            queue.append(nxt)
                components.append(comp)
            for rank, comp in enumerate(sorted(components, key=lambda c: (-len(c), c[0])), start=1):
                rows.append({"disease": disease, "graph_mode": mode, "component_rank": rank, "component_size": len(comp), "component_proportion": len(comp) / len(node_ids)})
    return pd.DataFrame(rows)


def sensitivity_policy_table() -> pd.DataFrame:
    rows = [
        {
            "policy_id": "A_primary_existing",
            "policy_name": "share_at_least_one_retained_kegg_pathway",
            "description": "Completed primary graph rule: connect retained metabolites if they share at least one retained KEGG pathway.",
            "adopted_for_primary_graph": True,
            "performance_or_label_based": False,
        }
    ]
    for pid, name in sorted(BROAD_KEGG_IDS.items()):
        rows.append(
            {
                "policy_id": "B_conservative_sensitivity_exclude_global_overview",
                "policy_name": "exclude_documented_global_or_overview_kegg_pathways",
                "description": "Prepared sensitivity only: remove explicitly listed broad KEGG overview/global maps and optionally require remaining specific-pathway support.",
                "excluded_kegg_pathway_id": pid,
                "excluded_kegg_pathway_name": name,
                "adopted_for_primary_graph": False,
                "performance_or_label_based": False,
            }
        )
    return pd.DataFrame(rows)


def warnings_table(graph_summary: pd.DataFrame, density: pd.DataFrame, pathway_contrib: pd.DataFrame, degree_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in graph_summary.iterrows():
        if float(row.get("largest_component_proportion", 0)) >= 0.75:
            rows.append({"disease": row["disease"], "graph_mode": row["graph_mode"], "warning": "largest_component_contains_most_nodes", "value": row["largest_component_proportion"]})
    for _, row in density.iterrows():
        if float(row["metabolite_subgraph_density"]) >= 0.25:
            rows.append({"disease": row["disease"], "graph_mode": row["graph_mode"], "warning": "mapped_metabolite_density_high", "value": row["metabolite_subgraph_density"]})
    if not pathway_contrib.empty:
        for (disease, mode), part in pathway_contrib.groupby(["disease", "graph_mode"]):
            total = part["edge_contributions"].sum()
            broad = part.loc[part["is_global_or_overview_pathway"], "edge_contributions"].sum()
            if total and broad / total >= 0.25:
                rows.append({"disease": disease, "graph_mode": mode, "warning": "broad_pathways_contribute_large_edge_share", "value": broad / total})
    for _, row in degree_summary.iterrows():
        if row["node_type"] == "metabolite" and float(row["max_degree"]) >= 0.5 * int(row["n_nodes"]):
            rows.append({"disease": row["disease"], "graph_mode": row["graph_mode"], "warning": "degree_hubs_could_dominate_gcn_message_passing", "value": row["max_degree"]})
    return pd.DataFrame(rows)


def write_report(tables: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# Corrected-v2 Stage 4D Provenance And Topology Audit",
        "",
        "This audit reads completed Stage 4D graph outputs only. It does not modify the primary graph and does not run Stage 4E.",
        "",
        "## NJC19 Attrition Interpretation",
        "",
        "The completed graph's NJC19 `source_records` should be interpreted as disease-filtered prepared candidate records, not the total original NJC19 network size. The audit table below traces back to the original local NJC19 JSON where available.",
        "",
        tables["njc19_attrition"].to_markdown(index=False),
        "",
        "## KEGG Topology Warnings",
        "",
        tables["warnings"].to_markdown(index=False) if not tables["warnings"].empty else "No topology warnings were triggered by the predefined thresholds.",
        "",
        "## Sensitivity Policy Prepared",
        "",
        "Policy A remains the completed primary graph. Policy B is prepared as a future sensitivity and must not be chosen using disease labels or predictive performance.",
        "",
        tables["sensitivity_policy"].head(12).to_markdown(index=False),
        "",
    ]
    atomic_text("\n".join(lines), REPORTS / "STAGE4D_CORRECTED_V2_PROVENANCE_TOPOLOGY_AUDIT.md")


def run_audit() -> None:
    for path in [NODES, BIO_EDGES, GRAPH_SUMMARY, VALIDATION]:
        if not path.exists():
            raise FileNotFoundError(f"Missing completed Stage 4D output required for audit: {path}")
    nodes = pd.read_csv(NODES)
    bio = pd.read_csv(BIO_EDGES)
    graph_summary = pd.read_csv(GRAPH_SUMMARY)
    validation = pd.read_csv(VALIDATION)
    if not validation["status"].eq("pass").all():
        raise RuntimeError("Completed Stage 4D validation table contains failures; audit refused.")

    nj_attr, rel_counts, support, species, ambiguous = njc19_provenance(nodes, bio)
    degree, degree_summary, top20 = degree_tables(nodes, bio)
    pathway_contrib, support_dist, broad = kegg_pathway_diagnostics(bio)
    density = metabolite_density(nodes, bio)
    components = component_distribution(nodes, bio)
    policy = sensitivity_policy_table()
    warnings = warnings_table(graph_summary, density, pathway_contrib, degree_summary)

    outputs = {
        "stage4d_corrected_v2_audit_njc19_original_attrition.csv": nj_attr,
        "stage4d_corrected_v2_audit_njc19_relation_type_counts.csv": rel_counts,
        "stage4d_corrected_v2_audit_njc19_retained_edge_support.csv": support,
        "stage4d_corrected_v2_audit_njc19_species_to_genus_support.csv": species,
        "stage4d_corrected_v2_audit_njc19_ambiguous_mappings.csv": ambiguous,
        "stage4d_corrected_v2_audit_node_degree_distribution.csv": degree,
        "stage4d_corrected_v2_audit_degree_summary.csv": degree_summary,
        "stage4d_corrected_v2_audit_top20_high_degree_nodes.csv": top20,
        "stage4d_corrected_v2_audit_connected_component_distribution.csv": components,
        "stage4d_corrected_v2_audit_kegg_pathway_edge_contributions.csv": pathway_contrib,
        "stage4d_corrected_v2_audit_kegg_edge_support_distribution.csv": support_dist,
        "stage4d_corrected_v2_audit_broad_kegg_pathway_edges.csv": broad,
        "stage4d_corrected_v2_audit_metabolite_subgraph_density.csv": density,
        "stage4d_corrected_v2_audit_graph_sensitivity_policy.csv": policy,
        "stage4d_corrected_v2_audit_topology_warnings.csv": warnings,
    }
    for name, df in outputs.items():
        atomic_csv(df, TABLES / name)
    write_report({"njc19_attrition": nj_attr, "warnings": warnings, "sensitivity_policy": policy})
    print("STAGE4D_CORRECTED_V2_PROVENANCE_TOPOLOGY_AUDIT_COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight corrected-v2 Stage 4D provenance/topology audit only.")
    parser.add_argument("--run", action="store_true", help="Read completed Stage 4D outputs and write audit-only diagnostics.")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("Use --run for the lightweight audit. This script never builds graphs or runs Stage 4E.")
    run_audit()


if __name__ == "__main__":
    main()
