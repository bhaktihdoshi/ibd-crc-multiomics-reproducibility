from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage4_corrected_v2_common import PROJECT, ROOT, STAGE4, TABLES, REPORTS, MANIFESTS, V2, atomic_csv, atomic_json, ensure_v2_dirs

DISEASES = ["IBD", "CRC"]
GRAPH_MODES = ["identity", "kegg_only", "kegg_njc19"]
EXPECTED_COUNTS = {"IBD": {"microbe": 50, "metabolite": 166}, "CRC": {"microbe": 52, "metabolite": 109}}
SCCA_INPUT_DIR = V2 / "data" / "stage4c_scca_pma_inputs"
PRIOR_TABLES = STAGE4 / "tables"
NJC19_EDGES = PRIOR_TABLES / "stage4d_njc19_crossomic_edges.csv"
NJC19_TAXON = PRIOR_TABLES / "stage4d_njc19_taxon_mapping.csv"
NJC19_MET = PRIOR_TABLES / "stage4d_njc19_metabolite_mapping.csv"
IBD_HMDB_KEGG_AUDIT = ROOT / "ibd_pipeline" / "tables" / "biological_primary_expanded" / "pathway" / "hmdb_to_kegg_mapping_audit.csv"
CRC_HMDB_KEGG_CROSSWALK = ROOT / "crc_pipeline" / "stage2" / "tables" / "crc_stage2_hmdb_kegg_crosswalk.csv"
KEGG_COMPOUND_PATHWAYS = ROOT / "ibd_pipeline" / "annotation_resources" / "kegg_compound_pathways.tsv"
KEGG_PATHWAY_NAMES = ROOT / "ibd_pipeline" / "annotation_resources" / "kegg_pathway_names.tsv"

NODE_SCHEMA = [
    "disease", "node_id", "node_index", "node_type", "feature_id", "feature_name",
    "source_feature_space", "source_mapping_status",
]
BIO_EDGE_SCHEMA = [
    "disease", "graph_mode", "source_node", "target_node", "edge_type", "weight",
    "directionality", "knowledge_source", "evidence_type", "mapping_provenance",
    "supporting_pathway_ids", "supporting_pathway_names", "relation_type", "source_microbe_names",
    "source_kegg_compound", "hmdb_id", "reference", "support_count", "biological_edge",
]
EDGE_INDEX_SCHEMA = ["disease", "graph_mode", "source_index", "target_index", "edge_kind", "weight"]


def canonical_hmdb(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.removeprefix("met__")
    if text.startswith("HMDB:"):
        text = text.split(":", 1)[1]
    if text.upper().startswith("HMDB"):
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            return f"HMDB{int(digits):07d}"
    return None


def canonical_kegg(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na"}:
        return None
    if text.startswith("cpd:"):
        text = text.split(":", 1)[1]
    if text.startswith("C") and len(text) >= 6 and text[1:6].isdigit():
        return text[:6]
    return None


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_stage4c_matrix(disease: str, cohort: str, view: str) -> pd.DataFrame:
    path = SCCA_INPUT_DIR / f"{disease}_{cohort}_{view}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 4C-compatible {view} matrix: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty Stage 4C-compatible {view} matrix: {path}")
    first = df.columns[0]
    if first != "sample_id":
        df = df.rename(columns={first: "sample_id"})
    if "sample_id" not in df.columns:
        raise ValueError(f"{path} lacks sample_id column")
    features = [c for c in df.columns if c != "sample_id"]
    values = df[features].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"Non-finite values in Stage 4C-compatible {view} matrix: {path}")
    return df[["sample_id"] + features]


def stage4c_feature_spaces() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    cohorts = {"IBD": ["Franzosa", "iHMP"], "CRC": ["Sinha", "Yachida"]}
    for disease, disease_cohorts in cohorts.items():
        micro_sets, met_sets = [], []
        for cohort in disease_cohorts:
            micro = read_stage4c_matrix(disease, cohort, "microbiome")
            met = read_stage4c_matrix(disease, cohort, "metabolome")
            micro_features = [c for c in micro.columns if c != "sample_id"]
            met_features = [c for c in met.columns if c != "sample_id"]
            micro_sets.append(set(micro_features))
            met_sets.append(set(met_features))
            audit.extend([
                {"disease": disease, "cohort": cohort, "view": "microbiome", "sample_count": len(micro), "feature_count": len(micro_features), "finite_values": True},
                {"disease": disease, "cohort": cohort, "view": "metabolome", "sample_count": len(met), "feature_count": len(met_features), "finite_values": True},
            ])
        common_micro = sorted(set.intersection(*micro_sets))
        common_met = sorted(set.intersection(*met_sets), key=lambda x: canonical_hmdb(x) or x)
        observed = {"microbe": len(common_micro), "metabolite": len(common_met)}
        expected = EXPECTED_COUNTS[disease]
        if observed != expected:
            raise ValueError(f"{disease} Stage 4C feature dimensions differ from expected: observed={observed}; expected={expected}")
        for idx, feature in enumerate(common_micro):
            genus = str(feature).removeprefix("mic__")
            rows.append({
                "disease": disease, "node_id": f"microbe::{genus}", "node_index": idx,
                "node_type": "microbe", "feature_id": feature, "feature_name": genus,
                "source_feature_space": "corrected_v2_stage4c_common_microbiome", "source_mapping_status": "authoritative shared canonical genus",
            })
        offset = len(common_micro)
        for j, feature in enumerate(common_met):
            hmdb = canonical_hmdb(feature)
            if hmdb is None:
                raise ValueError(f"{disease} metabolite feature lacks canonical HMDB identifier: {feature}")
            rows.append({
                "disease": disease, "node_id": f"metabolite::{hmdb}", "node_index": offset + j,
                "node_type": "metabolite", "feature_id": feature, "feature_name": hmdb,
                "source_feature_space": "corrected_v2_stage4c_common_metabolome", "source_mapping_status": "retained Stage 4C-compatible HMDB metabolite",
            })
    nodes = pd.DataFrame(rows, columns=NODE_SCHEMA)
    if nodes.duplicated(["disease", "node_id"]).any():
        dup = nodes.loc[nodes.duplicated(["disease", "node_id"], keep=False), ["disease", "node_id"]].head().to_dict("records")
        raise ValueError(f"Duplicated disease-specific node IDs: {dup}")
    return nodes, pd.DataFrame(audit)


def load_hmdb_kegg_mapping() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    if IBD_HMDB_KEGG_AUDIT.exists():
        df = pd.read_csv(IBD_HMDB_KEGG_AUDIT)
        for _, r in df.iterrows():
            hmdb = canonical_hmdb(r.get("HMDB_ID"))
            kegg = canonical_kegg(r.get("final_KEGG_Compound_ID")) or canonical_kegg(r.get("supplied_KEGG_Compound_ID")) or canonical_kegg(r.get("externally_mapped_KEGG_Compound_ID"))
            if hmdb and kegg:
                rows.append({"hmdb_id": hmdb, "kegg_compound_id": kegg, "source": str(IBD_HMDB_KEGG_AUDIT), "mapping_status": str(r.get("mapping_status", "mapped"))})
    if CRC_HMDB_KEGG_CROSSWALK.exists():
        df = pd.read_csv(CRC_HMDB_KEGG_CROSSWALK)
        for _, r in df.iterrows():
            hmdb = canonical_hmdb(r.get("HMDB") or r.get("canonical_id") or r.get("feature"))
            kegg = canonical_kegg(r.get("KEGG"))
            if hmdb and kegg:
                rows.append({"hmdb_id": hmdb, "kegg_compound_id": kegg, "source": str(CRC_HMDB_KEGG_CROSSWALK), "mapping_status": str(r.get("mapping_status", "mapped"))})
    if NJC19_MET.exists():
        df = pd.read_csv(NJC19_MET)
        for _, r in df.iterrows():
            hmdb = canonical_hmdb(r.get("HMDB_ID"))
            kegg = canonical_kegg(r.get("KEGG"))
            if hmdb and kegg:
                rows.append({"hmdb_id": hmdb, "kegg_compound_id": kegg, "source": str(NJC19_MET), "mapping_status": "NJC19 Stage 4 crosswalk"})
    out = pd.DataFrame(rows, columns=["hmdb_id", "kegg_compound_id", "source", "mapping_status"]).drop_duplicates()
    return out.sort_values(["hmdb_id", "kegg_compound_id", "source"], kind="mergesort").reset_index(drop=True)


def load_kegg_pathways() -> tuple[pd.DataFrame, dict[str, str]]:
    if not KEGG_COMPOUND_PATHWAYS.exists():
        raise FileNotFoundError(f"Missing KEGG compound-pathway table: {KEGG_COMPOUND_PATHWAYS}")
    cp = pd.read_csv(KEGG_COMPOUND_PATHWAYS, sep="\t", header=None, names=["compound", "pathway"])
    cp["kegg_compound_id"] = cp["compound"].map(canonical_kegg)
    cp["pathway_id"] = cp["pathway"].astype(str).str.replace("path:", "", regex=False)
    cp = cp.loc[cp["kegg_compound_id"].notna() & cp["pathway_id"].notna(), ["kegg_compound_id", "pathway_id"]].drop_duplicates()
    names: dict[str, str] = {}
    if KEGG_PATHWAY_NAMES.exists():
        pn = pd.read_csv(KEGG_PATHWAY_NAMES, sep="\t", header=None, names=["pathway_id", "pathway_name"])
        names = dict(zip(pn["pathway_id"].astype(str), pn["pathway_name"].astype(str)))
    return cp, names


def build_kegg_edges(nodes: pd.DataFrame, hmdb_kegg: pd.DataFrame, kegg_pathways: pd.DataFrame, pathway_names: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    attr_rows = []
    edge_rows: list[dict[str, Any]] = []
    for disease in DISEASES:
        mets = nodes[(nodes["disease"].eq(disease)) & nodes["node_type"].eq("metabolite")].copy()
        mets["hmdb_id"] = mets["node_id"].str.replace("metabolite::", "", regex=False)
        mapped = mets.merge(hmdb_kegg, on="hmdb_id", how="left")
        with_kegg = mapped[mapped["kegg_compound_id"].notna()].copy()
        with_path = with_kegg.merge(kegg_pathways, on="kegg_compound_id", how="inner")
        usable = with_path.groupby("pathway_id")["node_id"].nunique().reset_index(name="retained_metabolites_in_pathway")
        usable = usable[usable["retained_metabolites_in_pathway"] >= 2]
        attr_rows.append({
            "disease": disease,
            "source": "KEGG",
            "stage4_retained_metabolites": int(len(mets)),
            "hmdb_with_kegg_compound": int(with_kegg["hmdb_id"].nunique()),
            "hmdb_with_kegg_pathway": int(with_path["hmdb_id"].nunique()),
            "pathways_with_at_least_two_retained_metabolites": int(len(usable)),
            "unique_metabolite_metabolite_edges": 0,
        })
        valid_pathways = set(usable["pathway_id"])
        for pathway_id, part in with_path[with_path["pathway_id"].isin(valid_pathways)].groupby("pathway_id", sort=True):
            unique_nodes = sorted(part["node_id"].unique())
            for i, a in enumerate(unique_nodes):
                for b in unique_nodes[i + 1:]:
                    edge_rows.append({
                        "disease": disease, "graph_mode": "kegg_only", "source_node": a, "target_node": b,
                        "edge_type": "METABOLITE_METABOLITE", "weight": 1.0, "directionality": "undirected",
                        "knowledge_source": "KEGG compound-pathway co-membership",
                        "evidence_type": "shared retained KEGG pathway",
                        "mapping_provenance": "HMDB-to-KEGG crosswalk plus KEGG pathway membership",
                        "supporting_pathway_ids": pathway_id, "supporting_pathway_names": pathway_names.get(str(pathway_id), str(pathway_id)),
                        "relation_type": "", "source_microbe_names": "", "source_kegg_compound": "", "hmdb_id": "", "reference": "KEGG local cache", "support_count": 1, "biological_edge": True,
                    })
    edges = pd.DataFrame(edge_rows, columns=BIO_EDGE_SCHEMA)
    if not edges.empty:
        collapsed = []
        for _, group in edges.groupby(["disease", "source_node", "target_node", "edge_type"], sort=True):
            first = group.iloc[0].to_dict()
            first["supporting_pathway_ids"] = ";".join(sorted(group["supporting_pathway_ids"].astype(str).unique()))
            first["supporting_pathway_names"] = ";".join(sorted(group["supporting_pathway_names"].astype(str).unique()))
            first["support_count"] = int(group["supporting_pathway_ids"].nunique())
            collapsed.append(first)
        edges = pd.DataFrame(collapsed, columns=BIO_EDGE_SCHEMA).sort_values(["disease", "source_node", "target_node"], kind="mergesort").reset_index(drop=True)
    attr = pd.DataFrame(attr_rows)
    if not edges.empty:
        counts = edges.groupby("disease").size().to_dict()
        attr["unique_metabolite_metabolite_edges"] = attr["disease"].map(counts).fillna(0).astype(int)
    return edges, attr


def build_njc19_edges(nodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not NJC19_EDGES.exists():
        raise FileNotFoundError(f"Missing NJC19 prepared edge table: {NJC19_EDGES}")
    src = pd.read_csv(NJC19_EDGES)
    required = {"disease", "canonical_genus", "HMDB_ID", "source", "source_microbe_names", "source_kegg_compound", "relation_type", "reference", "support_count"}
    missing = sorted(required - set(src.columns))
    if missing:
        raise ValueError(f"NJC19 edge table missing columns: {missing}")
    allowed = src["relation_type"].astype(str).str.lower().str.contains("production|consumption|transport|import|export", regex=True, na=False)
    src = src.assign(valid_relation_type=allowed)
    rows = []
    attr = []
    for disease in DISEASES:
        disease_nodes = nodes[nodes["disease"].eq(disease)]
        microbe_nodes = set(disease_nodes.loc[disease_nodes["node_type"].eq("microbe"), "node_id"])
        metabolite_nodes = set(disease_nodes.loc[disease_nodes["node_type"].eq("metabolite"), "node_id"])
        dsrc = src[src["disease"].eq(disease)].copy()
        valid = dsrc[dsrc["valid_relation_type"]].copy()
        valid["source_node"] = "microbe::" + valid["canonical_genus"].astype(str)
        valid["target_node"] = "metabolite::" + valid["HMDB_ID"].map(canonical_hmdb).astype(str)
        mapped_genus = valid[valid["source_node"].isin(microbe_nodes)].copy()
        mapped_met = valid[valid["target_node"].isin(metabolite_nodes)].copy()
        both = valid[valid["source_node"].isin(microbe_nodes) & valid["target_node"].isin(metabolite_nodes)].copy()
        for _, r in both.iterrows():
            a, b = str(r["source_node"]), str(r["target_node"])
            if a == b:
                continue
            rows.append({
                "disease": disease, "graph_mode": "kegg_njc19", "source_node": a, "target_node": b,
                "edge_type": "MICROBE_METABOLITE", "weight": 1.0, "directionality": "undirected",
                "knowledge_source": "NJC19 curated mammalian microbiome metabolic network",
                "evidence_type": "curated production consumption or transport relation",
                "mapping_provenance": "NJC19 species-to-genus collapse plus KEGG/HMDB Stage 4 crosswalk",
                "supporting_pathway_ids": "", "supporting_pathway_names": "",
                "relation_type": str(r["relation_type"]), "source_microbe_names": str(r["source_microbe_names"]),
                "source_kegg_compound": str(r["source_kegg_compound"]), "hmdb_id": canonical_hmdb(r["HMDB_ID"]),
                "reference": str(r["reference"]), "support_count": int(r.get("support_count", 1) or 1), "biological_edge": True,
            })
        unique_keys = set((x["source_node"], x["target_node"]) for x in rows if x["disease"] == disease)
        attr.append({
            "disease": disease, "source": "NJC19", "source_records": int(len(dsrc)),
            "valid_relation_type": int(len(valid)), "mapped_genus": int(len(mapped_genus)),
            "mapped_metabolite": int(len(mapped_met)), "both_endpoints_retained": int(len(both)),
            "unique_graph_edges": int(len(unique_keys)),
            "terminology": "curated production, consumption or transport relations; not positive correlations",
        })
    edges = pd.DataFrame(rows, columns=BIO_EDGE_SCHEMA)
    if not edges.empty:
        collapsed = []
        for _, group in edges.groupby(["disease", "source_node", "target_node", "edge_type"], sort=True):
            first = group.iloc[0].to_dict()
            first["relation_type"] = ";".join(sorted(group["relation_type"].astype(str).unique()))
            first["source_microbe_names"] = ";".join(sorted(group["source_microbe_names"].astype(str).unique()))
            first["source_kegg_compound"] = ";".join(sorted(group["source_kegg_compound"].astype(str).unique()))
            first["reference"] = ";".join(sorted(group["reference"].astype(str).unique()))
            first["support_count"] = int(group["support_count"].sum())
            collapsed.append(first)
        edges = pd.DataFrame(collapsed, columns=BIO_EDGE_SCHEMA).sort_values(["disease", "source_node", "target_node"], kind="mergesort").reset_index(drop=True)
    return edges, pd.DataFrame(attr)


def biological_edges_by_mode(kegg_edges: pd.DataFrame, njc19_edges: pd.DataFrame) -> pd.DataFrame:
    frames = []
    empty = pd.DataFrame(columns=BIO_EDGE_SCHEMA)
    for disease in DISEASES:
        ident = empty.copy()
        ident["disease"] = []
        frames.append(ident)
        kegg = kegg_edges[kegg_edges["disease"].eq(disease)].copy()
        kegg["graph_mode"] = "kegg_only"
        frames.append(kegg)
        aug = pd.concat([kegg.assign(graph_mode="kegg_njc19"), njc19_edges[njc19_edges["disease"].eq(disease)].copy()], ignore_index=True)
        if not aug.empty:
            aug["graph_mode"] = aug["graph_mode"].fillna("kegg_njc19")
        frames.append(aug)
    out = pd.concat(frames, ignore_index=True) if frames else empty
    if out.empty:
        return empty
    return out[BIO_EDGE_SCHEMA].sort_values(["disease", "graph_mode", "edge_type", "source_node", "target_node"], kind="mergesort").reset_index(drop=True)


def edge_index_for_modes(nodes: pd.DataFrame, biological_edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for disease in DISEASES:
        ndf = nodes[nodes["disease"].eq(disease)]
        idx = dict(zip(ndf["node_id"], ndf["node_index"]))
        for mode in GRAPH_MODES:
            for node_id, node_index in idx.items():
                rows.append({"disease": disease, "graph_mode": mode, "source_index": node_index, "target_index": node_index, "edge_kind": "computational_self_loop", "weight": 1.0})
            edf = biological_edges[(biological_edges["disease"].eq(disease)) & (biological_edges["graph_mode"].eq(mode))]
            for _, e in edf.iterrows():
                a, b = idx[e["source_node"]], idx[e["target_node"]]
                rows.append({"disease": disease, "graph_mode": mode, "source_index": a, "target_index": b, "edge_kind": e["edge_type"], "weight": 1.0})
                rows.append({"disease": disease, "graph_mode": mode, "source_index": b, "target_index": a, "edge_kind": e["edge_type"], "weight": 1.0})
    return pd.DataFrame(rows, columns=EDGE_INDEX_SCHEMA).sort_values(["disease", "graph_mode", "source_index", "target_index", "edge_kind"], kind="mergesort").reset_index(drop=True)


def graph_stats(nodes: pd.DataFrame, biological_edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    qc_rows, iso_rows, comp_rows = [], [], []
    for disease in DISEASES:
        ndf = nodes[nodes["disease"].eq(disease)].copy()
        node_ids = ndf["node_id"].tolist()
        for mode in GRAPH_MODES:
            edf = biological_edges[(biological_edges["disease"].eq(disease)) & (biological_edges["graph_mode"].eq(mode))]
            adj = {n: set() for n in node_ids}
            for _, e in edf.iterrows():
                adj[e["source_node"]].add(e["target_node"])
                adj[e["target_node"]].add(e["source_node"])
            degrees = pd.Series({n: len(v) for n, v in adj.items()}, dtype=float)
            isolated = degrees[degrees.eq(0)].index.tolist()
            for n in isolated:
                rec = ndf[ndf["node_id"].eq(n)].iloc[0].to_dict()
                rec.update({"graph_mode": mode, "isolation_meaning": "isolated from biological prior edges; computational self-loop retained in edge-index"})
                iso_rows.append(rec)
            seen, comps = set(), []
            for n in node_ids:
                if n in seen:
                    continue
                q, comp = deque([n]), []
                seen.add(n)
                while q:
                    cur = q.popleft(); comp.append(cur)
                    for nxt in adj[cur]:
                        if nxt not in seen:
                            seen.add(nxt); q.append(nxt)
                comps.append(sorted(comp))
            for cid, comp in enumerate(sorted(comps, key=lambda c: (-len(c), c[0])), start=1):
                comp_rows.append({"disease": disease, "graph_mode": mode, "component_id": cid, "component_size": len(comp), "component_proportion": len(comp) / len(node_ids), "node_ids": ";".join(comp)})
            possible = len(node_ids) * (len(node_ids) - 1) / 2
            edge_count = len(edf)
            edge_nodes = set(edf["source_node"]).union(set(edf["target_node"])) if not edf.empty else set()
            micro = set(ndf.loc[ndf["node_type"].eq("microbe"), "node_id"])
            met = set(ndf.loc[ndf["node_type"].eq("metabolite"), "node_id"])
            qc_rows.append({
                "disease": disease, "graph_mode": mode, "n_nodes": len(node_ids),
                "microbe_nodes": len(micro), "metabolite_nodes": len(met), "total_biological_edges": edge_count,
                "metabolite_metabolite_edges": int(edf["edge_type"].eq("METABOLITE_METABOLITE").sum()) if not edf.empty else 0,
                "microbe_metabolite_edges": int(edf["edge_type"].eq("MICROBE_METABOLITE").sum()) if not edf.empty else 0,
                "microbe_microbe_edges": int(edf["edge_type"].eq("MICROBE_MICROBE").sum()) if not edf.empty else 0,
                "isolated_nodes": len(isolated), "isolated_microbe_nodes": len(micro - edge_nodes), "isolated_metabolite_nodes": len(met - edge_nodes),
                "connected_components": len(comps), "largest_component_size": max(map(len, comps)) if comps else 0,
                "largest_component_proportion": (max(map(len, comps)) / len(node_ids)) if comps else 0.0,
                "graph_density": float(edge_count / possible) if possible else 0.0,
                "microbial_coverage": len(micro & edge_nodes) / len(micro) if micro else 0.0,
                "biological_edges_exclude_self_loops": True,
            })
    return pd.DataFrame(qc_rows), pd.DataFrame(iso_rows), pd.DataFrame(comp_rows)


def validate_outputs(nodes: pd.DataFrame, bio: pd.DataFrame, edge_index: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(test: str, ok: bool, notes: str = "") -> None:
        rows.append({"test": test, "status": "pass" if ok else "fail", "notes": notes})
    add("node_schema", list(nodes.columns) == NODE_SCHEMA, str(list(nodes.columns)))
    add("biological_edge_schema", list(bio.columns) == BIO_EDGE_SCHEMA, str(list(bio.columns)))
    add("edge_index_schema", list(edge_index.columns) == EDGE_INDEX_SCHEMA, str(list(edge_index.columns)))
    add("unique_disease_node_ids", not nodes.duplicated(["disease", "node_id"]).any())
    add("no_clinical_label_or_participant_nodes", not nodes["node_type"].isin(["clinical", "participant", "label", "phenotype"]).any())
    add("no_biological_self_edges", bio.empty or not bio["source_node"].eq(bio["target_node"]).any())
    add("all_biological_weights_binary", bio.empty or bio["weight"].astype(float).eq(1.0).all())
    add("all_biological_edges_undirected", bio.empty or bio["directionality"].eq("undirected").all())
    add("no_microbe_microbe_edges", bio.empty or not bio["edge_type"].eq("MICROBE_MICROBE").any())
    forbidden = "label|phenotype|coefficient|correlation|prediction|auc|mofa|scca"
    add("no_model_or_phenotype_edge_provenance", bio.empty or not bio.astype(str).apply(lambda c: c.str.contains(forbidden, case=False, regex=True)).any().any())
    for disease in DISEASES:
        ndf = nodes[nodes["disease"].eq(disease)]
        add(f"{disease}_node_index_order", ndf["node_index"].tolist() == list(range(len(ndf))))
        for nt, expected in EXPECTED_COUNTS[disease].items():
            add(f"{disease}_{nt}_count", int(ndf["node_type"].eq(nt).sum()) == expected, f"expected={expected}")
        node_set = set(ndf["node_id"])
        for mode in GRAPH_MODES:
            edf = bio[(bio["disease"].eq(disease)) & (bio["graph_mode"].eq(mode))]
            endpoints_ok = edf.empty or (set(edf["source_node"]).issubset(node_set) and set(edf["target_node"]).issubset(node_set))
            add(f"{disease}_{mode}_edge_endpoints_retained", endpoints_ok)
            if mode == "identity":
                add(f"{disease}_identity_biological_edges_zero", len(edf) == 0, f"observed={len(edf)}")
            if mode == "kegg_only":
                add(f"{disease}_kegg_only_metabolite_edges_only", edf.empty or set(edf["edge_type"]) == {"METABOLITE_METABOLITE"})
            if mode == "kegg_njc19":
                add(f"{disease}_kegg_njc19_valid_edge_types", edf.empty or set(edf["edge_type"]).issubset({"METABOLITE_METABOLITE", "MICROBE_METABOLITE"}))
        id_edges = set()
        ko = set(map(tuple, bio[(bio["disease"].eq(disease)) & bio["graph_mode"].eq("kegg_only")][["source_node", "target_node", "edge_type"]].to_numpy()))
        aug = set(map(tuple, bio[(bio["disease"].eq(disease)) & bio["graph_mode"].eq("kegg_njc19")][["source_node", "target_node", "edge_type"]].to_numpy()))
        add(f"{disease}_edge_monotonicity", id_edges.issubset(ko) and ko.issubset(aug))
        for a, b in [("identity", "kegg_only"), ("identity", "kegg_njc19")]:
            na = nodes[nodes["disease"].eq(disease)]["node_id"].tolist()
            add(f"{disease}_{a}_{b}_same_node_order", na == na)
    return pd.DataFrame(rows)


def graph_hash(nodes: pd.DataFrame, bio: pd.DataFrame, edge_index: pd.DataFrame) -> str:
    payload = {
        "nodes": nodes.sort_values(["disease", "node_index"]).to_dict("records"),
        "biological_edges": bio.sort_values(BIO_EDGE_SCHEMA).to_dict("records") if not bio.empty else [],
        "edge_index": edge_index.sort_values(EDGE_INDEX_SCHEMA).to_dict("records"),
        "graph_modes": GRAPH_MODES,
        "policy": "corrected_v2_branch_b_unsigned_undirected_prior_graph",
    }
    return sha256_text(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")))


def build_graph_plan() -> pd.DataFrame:
    rows = []
    for mode in GRAPH_MODES:
        rows.append({
            "graph_mode": mode,
            "display_name": {"identity": "identity/self-information", "kegg_only": "KEGG-only", "kegg_njc19": "KEGG + NJC19"}[mode],
            "metabolite_metabolite_edges": "KEGG pathway co-membership" if mode in {"kegg_only", "kegg_njc19"} else "none",
            "microbe_metabolite_edges": "NJC19 curated production, consumption, or transport relations" if mode == "kegg_njc19" else "none",
            "self_loops": True,
            "edge_weight_policy": "unsigned_binary_weight_1",
            "graph_directionality": "undirected_primary_graph",
            "node_scope": "corrected_v2_stage4c_microbial_genera_and_hmdb_metabolites_only",
            "biological_edge_table_excludes_self_loops": True,
            "microbe_microbe_edges_in_primary_graph": False,
            "clinical_variables_or_disease_labels_as_nodes": False,
            "preserve_njc19_relation_type": True,
            "njc19_terminology": "curated production, consumption, or transport relations; not positive statistical associations or positive correlations",
            "execution_status": "prepared_or_complete_depending_on_command",
        })
    return pd.DataFrame(rows)


def build_dependency_map() -> pd.DataFrame:
    return pd.DataFrame([
        {"stage": "4D", "dependency": "corrected-v2 Stage 4C-compatible microbiome/metabolome matrices", "required": True, "blocking_if_missing": True},
        {"stage": "4D", "dependency": "local HMDB-to-KEGG mappings and KEGG compound-pathway cache", "required": True, "blocking_if_missing": True},
        {"stage": "4D", "dependency": "prepared NJC19 curated relation table", "required": True, "blocking_if_missing": True},
        {"stage": "4D", "dependency": "no disease labels, clinical variables, correlations, coefficients or model outputs", "required": True, "blocking_if_missing": True},
    ])


def dry_run() -> None:
    ensure_v2_dirs()
    plan = build_graph_plan(); deps = build_dependency_map()
    nodes, input_audit = stage4c_feature_spaces()
    hmdb_kegg = load_hmdb_kegg_mapping(); kegg_pathways, _ = load_kegg_pathways()
    source_audit = source_availability_audit(nodes, hmdb_kegg, kegg_pathways)
    atomic_csv(plan, TABLES / "stage4d_corrected_v2_graph_mode_plan.csv")
    atomic_csv(deps, TABLES / "stage4d_corrected_v2_dependency_map.csv")
    atomic_csv(input_audit, TABLES / "stage4d_corrected_v2_input_feature_audit.csv")
    atomic_csv(source_audit, TABLES / "stage4d_corrected_v2_source_availability_audit.csv")
    atomic_csv(stale_stage4d_outputs(), TABLES / "stage4d_corrected_v2_stale_output_map.csv")
    write_report(plan, deps, input_audit, source_audit, None)
    atomic_json({"status": "prepared_only", "full_graph_constructed": False, "graph_modes": GRAPH_MODES, "stage4e_modified_or_run": False}, MANIFESTS / "stage4d_corrected_v2_prepare_manifest.json")
    print("STAGE4D_CORRECTED_V2_GRAPH_PREPARED_ONLY")


def source_availability_audit(nodes: pd.DataFrame, hmdb_kegg: pd.DataFrame, kegg_pathways: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for disease in DISEASES:
        mets = nodes[(nodes["disease"].eq(disease)) & nodes["node_type"].eq("metabolite")].copy()
        mets["hmdb_id"] = mets["node_id"].str.replace("metabolite::", "", regex=False)
        mapped = mets.merge(hmdb_kegg, on="hmdb_id", how="left")
        with_kegg = mapped[mapped["kegg_compound_id"].notna()]
        with_path = with_kegg.merge(kegg_pathways, on="kegg_compound_id", how="inner")
        rows.append({"disease": disease, "source": "KEGG", "retained_metabolites": len(mets), "mapped_kegg_compounds": int(with_kegg["hmdb_id"].nunique()), "mapped_kegg_pathways": int(with_path["hmdb_id"].nunique()), "source_file": f"{IBD_HMDB_KEGG_AUDIT}; {CRC_HMDB_KEGG_CROSSWALK}; {KEGG_COMPOUND_PATHWAYS}"})
    if NJC19_EDGES.exists():
        nj = pd.read_csv(NJC19_EDGES)
        for disease in DISEASES:
            rows.append({"disease": disease, "source": "NJC19", "source_records": int(nj[nj["disease"].eq(disease)].shape[0]), "source_file": str(NJC19_EDGES), "terminology": "curated production/consumption/transport relations, not positive correlations"})
    return pd.DataFrame(rows)


def stale_stage4d_outputs() -> pd.DataFrame:
    stale = sorted(str(p) for p in (STAGE4 / "tables").glob("stage4d_graph*.csv")) + sorted(str(p) for p in (STAGE4 / "tables").glob("stage4d_njc19*.csv"))
    return pd.DataFrame([{"path": p, "status": "stale_for_corrected_v2_until_full_stage4d_build", "reason": "historical/non-corrected-v2 Stage 4D output"} for p in stale])


def write_report(plan: pd.DataFrame, deps: pd.DataFrame, input_audit: pd.DataFrame, source_audit: pd.DataFrame, validation: pd.DataFrame | None) -> None:
    lines = [
        "# Corrected-v2 Stage 4D Graph Construction Audit",
        "",
        "Stage 4D builds prior-knowledge feature graphs only. It does not train GCN/GAT models and does not use labels, clinical variables, coefficients, correlations, MOFA, sCCA or participant nodes as graph edges/nodes.",
        "",
        "## Graph Modes", plan.to_markdown(index=False), "", "## Dependencies", deps.to_markdown(index=False),
        "", "## Stage 4C Feature-Space Audit", input_audit.to_markdown(index=False), "", "## Prior-Source Audit", source_audit.to_markdown(index=False),
    ]
    if validation is not None:
        lines += ["", "## Validation", validation.to_markdown(index=False)]
    lines += ["", "Manual full build command:", "", "```bash", "python thesis_preprocessing_workflow/stage4_multiomics_integration/corrected_v2/scripts/run_stage4d_corrected_v2_graph.py --build", "```", ""]
    atomic_text("\n".join(lines), REPORTS / "STAGE4D_CORRECTED_V2_GRAPH_SCOPE.md")


def build_full() -> None:
    ensure_v2_dirs()
    nodes, input_audit = stage4c_feature_spaces()
    hmdb_kegg = load_hmdb_kegg_mapping(); kegg_pathways, pathway_names = load_kegg_pathways()
    kegg_edges, kegg_attr = build_kegg_edges(nodes, hmdb_kegg, kegg_pathways, pathway_names)
    njc19_edges, njc19_attr = build_njc19_edges(nodes)
    bio = biological_edges_by_mode(kegg_edges, njc19_edges)
    edge_index = edge_index_for_modes(nodes, bio)
    validation = validate_outputs(nodes, bio, edge_index)
    if not validation["status"].eq("pass").all():
        atomic_csv(validation, TABLES / "stage4d_corrected_v2_validation_tests.csv")
        raise RuntimeError("Corrected-v2 Stage 4D graph validation failed.")
    qc, isolated, components = graph_stats(nodes, bio)
    source_audit = source_availability_audit(nodes, hmdb_kegg, kegg_pathways)
    hash_value = graph_hash(nodes, bio, edge_index)
    atomic_csv(nodes, TABLES / "stage4d_corrected_v2_nodes.csv")
    atomic_csv(bio, TABLES / "stage4d_corrected_v2_biological_edges.csv")
    atomic_csv(edge_index, TABLES / "stage4d_corrected_v2_edge_index.csv")
    for disease in DISEASES:
        atomic_csv(nodes[nodes["disease"].eq(disease)], TABLES / f"stage4d_corrected_v2_{disease.lower()}_nodes.csv")
        for mode in GRAPH_MODES:
            atomic_csv(edge_index[(edge_index["disease"].eq(disease)) & edge_index["graph_mode"].eq(mode)], TABLES / f"stage4d_corrected_v2_{disease.lower()}_{mode}_edge_index.csv")
    atomic_csv(qc, TABLES / "stage4d_corrected_v2_graph_summary.csv")
    atomic_csv(isolated, TABLES / "stage4d_corrected_v2_isolated_nodes.csv")
    atomic_csv(components, TABLES / "stage4d_corrected_v2_connected_components.csv")
    atomic_csv(kegg_attr, TABLES / "stage4d_corrected_v2_kegg_mapping_attrition.csv")
    atomic_csv(njc19_attr, TABLES / "stage4d_corrected_v2_njc19_mapping_attrition.csv")
    atomic_csv(validation, TABLES / "stage4d_corrected_v2_validation_tests.csv")
    atomic_csv(input_audit, TABLES / "stage4d_corrected_v2_input_feature_audit.csv")
    atomic_csv(source_audit, TABLES / "stage4d_corrected_v2_source_availability_audit.csv")
    atomic_csv(stale_stage4d_outputs(), TABLES / "stage4d_corrected_v2_stale_output_map.csv")
    atomic_json({"status": "complete", "graph_hash": hash_value, "graph_modes": GRAPH_MODES, "stage4e_modified_or_run": False}, MANIFESTS / "stage4d_corrected_v2_graph_manifest.json")
    atomic_text(hash_value + "\n", MANIFESTS / "stage4d_corrected_v2_graph_hash.txt")
    write_report(build_graph_plan(), build_dependency_map(), input_audit, source_audit, validation)
    print("STAGE4D_CORRECTED_V2_GRAPH_COMPLETE")


def audit_only() -> None:
    required = {
        "stage4d_corrected_v2_nodes.csv": NODE_SCHEMA,
        "stage4d_corrected_v2_biological_edges.csv": BIO_EDGE_SCHEMA,
        "stage4d_corrected_v2_edge_index.csv": EDGE_INDEX_SCHEMA,
        "stage4d_corrected_v2_graph_summary.csv": ["disease", "graph_mode", "n_nodes", "total_biological_edges"],
        "stage4d_corrected_v2_validation_tests.csv": ["test", "status", "notes"],
    }
    rows = []
    for name, cols in required.items():
        path = TABLES / name
        rec = {"output_file": str(path), "exists": path.exists(), "schema_status": "not_checked", "status": "fail", "notes": ""}
        if path.exists():
            df = pd.read_csv(path, nrows=5)
            missing = [c for c in cols if c not in df.columns]
            rec.update({"schema_status": "pass" if not missing else "fail", "status": "pass" if not missing else "fail", "notes": "" if not missing else f"missing={missing}"})
        rows.append(rec)
    audit = pd.DataFrame(rows)
    atomic_csv(audit, TABLES / "stage4d_corrected_v2_postrun_audit.csv")
    if not audit["status"].eq("pass").all():
        raise RuntimeError("Corrected-v2 Stage 4D postrun audit failed.")
    print("STAGE4D_CORRECTED_V2_GRAPH_AUDIT_COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrected-v2 Stage 4D graph construction only; no Stage 4E fitting.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Inspect inputs and write plans/audits only; do not construct graph files.")
    group.add_argument("--build", action="store_true", help="Build corrected-v2 Stage 4D graph templates and edge-index files.")
    group.add_argument("--audit-only", action="store_true", help="Audit existing corrected-v2 Stage 4D outputs only.")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.build:
        build_full()
    elif args.audit_only:
        audit_only()


if __name__ == "__main__":
    main()
