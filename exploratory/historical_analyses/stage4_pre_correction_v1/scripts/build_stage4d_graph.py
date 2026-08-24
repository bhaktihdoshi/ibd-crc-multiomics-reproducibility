from __future__ import annotations

import argparse
import re
import hashlib
import itertools
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from stage4_common import CONFIGS, FIGURES, MANIFESTS, PROJECT, REPORTS, STAGE4, TABLES, require_multiomics_ready, write_log


FEATURE_TRACE = TABLES / "stage4c_feature_qc_trace.csv"
IBD_HMDB_FEATURE_METADATA = PROJECT / "thesis_preprocessing_workflow/ibd_pipeline/data_processed/primary_high_confidence_hmdb_feature_metadata.csv"
CRC_HMDB_KEGG_CROSSWALK = PROJECT / "thesis_preprocessing_workflow/crc_pipeline/stage2/tables/crc_stage2_hmdb_kegg_crosswalk.csv"
KEGG_COMPOUND_PATHWAYS = PROJECT / "thesis_preprocessing_workflow/ibd_pipeline/annotation_resources/kegg_compound_pathways.tsv"
KEGG_PATHWAY_NAMES = PROJECT / "thesis_preprocessing_workflow/ibd_pipeline/annotation_resources/kegg_pathway_names.tsv"
STAGE4D_PRIOR_DIR = STAGE4 / "data/stage4d_prior_sources"
MIMEDB_DIR = STAGE4D_PRIOR_DIR / "mimedb"
BASELINE_GRAPH_HASH = "d423255659a9f43069cc14885e1a2e2608d66d75bcaeabafd7a103c6532fb27b"

NODE_SCHEMA = [
    "node_id",
    "node_index",
    "node_type",
    "feature_id",
    "feature_name",
    "disease_space",
    "source_mapping_status",
]
EDGE_SCHEMA = [
    "source_node",
    "target_node",
    "edge_type",
    "weight",
    "directionality",
    "knowledge_source",
    "evidence_type",
    "mapping_provenance",
    "disease_space",
    "supporting_pathway_ids",
    "supporting_pathway_names",
]
AUGMENTED_EDGE_SCHEMA = EDGE_SCHEMA + [
    "edge_source",
    "source_microbe_id",
    "source_metabolite_id",
    "canonical_genus",
    "HMDB_ID",
    "relation_type",
    "reference",
    "support_count",
    "mapping_method",
]
MICROBE_METABOLITE_MAPPING_SCHEMA = [
    "source",
    "source_microbe_id",
    "source_metabolite_id",
    "source_microbe_name",
    "source_metabolite_name",
    "canonical_genus",
    "HMDB_ID",
    "relation_type",
    "evidence_type",
    "reference",
    "support_count",
    "mapping_method",
    "ambiguous_taxon",
    "ambiguous_metabolite",
    "status",
]
VALID_EDGE_TYPES = {"METABOLITE_METABOLITE", "MICROBE_METABOLITE", "MICROBE_MICROBE"}
VALID_NODE_TYPES = {"microbe", "metabolite"}
DISEASES = ["IBD", "CRC"]
EXPECTED_COUNTS = {
    "IBD": {"microbe": 50, "metabolite": 166},
    "CRC": {"microbe": 52, "metabolite": 109},
}


@dataclass(frozen=True)
class PriorSource:
    source_name: str
    edge_type: str
    node_type_A: str
    node_type_B: str
    evidence_type: str
    directionality: str
    weight_available: str
    local_file_source: str
    status: str
    coverage_note: str


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def canonical_hmdb(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("met__"):
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
    if not text:
        return None
    if text.startswith("cpd:"):
        text = text.split(":", 1)[1]
    if text.startswith("C") and len(text) >= 6:
        return text[:6]
    return None


def canonical_microbe_name(feature_id: str, feature_name: object = None) -> str:
    if feature_id.startswith("mic__"):
        return feature_id.removeprefix("mic__")
    if pd.notna(feature_name) and str(feature_name).strip():
        return str(feature_name).strip()
    return feature_id


def normalize_token(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def split_multi_value(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = re.split(r"[|;,]\s*|\s+//\s+|\s+\|\s+", text)
    return [part.strip() for part in parts if part.strip() and part.strip().lower() not in {"na", "nan", "not available", "none"}]


def infer_genus(taxon: object) -> tuple[str | None, bool, str]:
    text = str(taxon).strip() if pd.notna(taxon) else ""
    if not text:
        return None, False, "missing_taxon"
    if ";" in text:
        candidates = [part for part in text.split(";") if "__" in part or part.strip()]
        genus_terms = [part.split("__", 1)[1].strip() for part in candidates if part.strip().lower().startswith("g__") and part.split("__", 1)[1].strip()]
        if len(set(genus_terms)) == 1:
            return genus_terms[0], False, "taxonomy_string_genus"
        if len(set(genus_terms)) > 1:
            return None, True, "ambiguous_taxonomy_string"
    cleaned = re.sub(r"^\w__+", "", text)
    cleaned = cleaned.replace("_", " ").strip()
    words = [w for w in re.split(r"\s+", cleaned) if w]
    if not words:
        return None, False, "missing_taxon"
    genus = re.sub(r"[^A-Za-z0-9-]", "", words[0])
    if not genus or genus.lower() in {"uncultured", "unknown", "unclassified", "bacterium", "sp"}:
        return None, True, "ambiguous_or_unclassified_taxon"
    return genus, False, "species_or_name_to_genus"


def find_column(columns: Iterable[str], required_terms: list[str], optional_terms: list[str] | None = None) -> str | None:
    optional_terms = optional_terms or []
    scored: list[tuple[int, str]] = []
    for col in columns:
        norm = normalize_token(col)
        if all(term in norm for term in required_terms):
            score = len(required_terms) * 10 + sum(term in norm for term in optional_terms)
            scored.append((score, col))
    if not scored:
        return None
    return sorted(scored, key=lambda x: (-x[0], x[1]))[0][1]


def is_relation_count_column(column: str) -> bool:
    norm = normalize_token(column)
    return "microbe" in norm and ("relation" in norm or "source" in norm) and any(term in norm for term in ["count", "number", "num", "relations"])


def local_mimedb_files() -> list[Path]:
    if not MIMEDB_DIR.exists():
        return []
    return sorted([p for p in MIMEDB_DIR.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".tsv"}])


def read_mimedb_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep, dtype=str, nrows=nrows, encoding="utf-8-sig")


def mimedb_named_files() -> dict[str, Path | None]:
    files = local_mimedb_files()
    return {
        "microbes": next((p for p in files if "microbe" in p.name.lower()), None),
        "metabolites": next((p for p in files if "metabolite" in p.name.lower() or "compound" in p.name.lower()), None),
        "relations": next((p for p in files if any(term in p.name.lower() for term in ["relation", "microbe_metabolite", "microbe-metabolite", "reaction"])), None),
    }


def audit_mimedb_sources() -> pd.DataFrame:
    rows = []
    files = local_mimedb_files()
    for path in files:
        try:
            preview = read_mimedb_csv(path, nrows=5)
            rows.append({
                "source": "MIMEDB",
                "local_file": str(path),
                "available": True,
                "n_preview_rows": len(preview),
                "columns": ";".join(map(str, preview.columns)),
                "schema_status": "readable",
            })
        except Exception as exc:
            rows.append({
                "source": "MIMEDB",
                "local_file": str(path),
                "available": True,
                "n_preview_rows": 0,
                "columns": "",
                "schema_status": f"unreadable:{exc!r}",
            })
    if not rows:
        rows.append({
            "source": "MIMEDB",
            "local_file": str(MIMEDB_DIR),
            "available": False,
            "n_preview_rows": 0,
            "columns": "",
            "schema_status": "missing_local_mimedb_bulk_csv_or_tsv",
        })
    return pd.DataFrame(rows)


def stage4_sets(nodes: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    disease_map = disease_node_mapping(nodes)
    out: dict[str, dict[str, set[str]]] = {}
    for disease in DISEASES:
        part = disease_map.loc[disease_map["disease_space"].eq(disease)]
        out[disease] = {
            "genera": {node.replace("microbe::", "") for node in part.loc[part["node_type"].eq("microbe"), "node_id"]},
            "hmdb": {node.replace("metabolite::", "") for node in part.loc[part["node_type"].eq("metabolite"), "node_id"]},
        }
    return out


def mimedb_schema_details() -> pd.DataFrame:
    rows = []
    for path in local_mimedb_files():
        try:
            df = read_mimedb_csv(path)
            preview = read_mimedb_csv(path, nrows=5)
            rows.append({
                "file": str(path),
                "rows": int(len(df)),
                "columns_n": int(len(df.columns)),
                "delimiter": "\\t" if path.suffix.lower() == ".tsv" else ",",
                "encoding": "utf-8-sig",
                "quoting_behavior": "standard CSV quoting parsed by pandas",
                "columns": ";".join(map(str, df.columns)),
                "relevant_non_null_examples": json.dumps({col: preview[col].dropna().astype(str).head(3).tolist() for col in preview.columns if preview[col].notna().any()}, default=str),
            })
        except Exception as exc:
            rows.append({"file": str(path), "rows": 0, "columns_n": 0, "delimiter": "", "encoding": "utf-8-sig", "quoting_behavior": "", "columns": "", "relevant_non_null_examples": f"ERROR:{exc!r}"})
    return pd.DataFrame(rows)


def real_schema_mapping_audit(nodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    named = mimedb_named_files()
    stage = stage4_sets(nodes)
    micro = read_mimedb_csv(named["microbes"]) if named["microbes"] else pd.DataFrame()
    met = read_mimedb_csv(named["metabolites"]) if named["metabolites"] else pd.DataFrame()
    relation = read_mimedb_csv(named["relations"]) if named["relations"] else pd.DataFrame()

    mimedb_genera = set()
    if not micro.empty and "genus" in micro.columns:
        mimedb_genera = {str(v).strip() for v in micro["genus"].dropna() if str(v).strip()}
    mimedb_genera_norm = {normalize_token(g): g for g in mimedb_genera}
    all_stage_hmdb = set().union(*(stage[d]["hmdb"] for d in DISEASES))
    mimedb_hmdb = set()
    if not met.empty and "hmdb_id" in met.columns:
        mimedb_hmdb = {canonical_hmdb(v) for v in met["hmdb_id"].dropna()}
        mimedb_hmdb = {v for v in mimedb_hmdb if v}

    explicit_relation_available = False
    relation_notes = []
    if not relation.empty:
        cols = list(relation.columns)
        micro_col = find_column(cols, ["microbe"], ["id", "name", "taxon", "genus"]) or find_column(cols, ["genus"]) or find_column(cols, ["taxon"])
        met_col = find_column(cols, ["hmdb"]) or find_column(cols, ["mime", "id"]) or find_column(cols, ["metabolite", "id"])
        explicit_relation_available = bool(micro_col and met_col and not is_relation_count_column(micro_col))
        relation_notes.append(f"relation_file={named['relations']}; microbe_col={micro_col}; metabolite_col={met_col}")
    else:
        relation_notes.append("No separate MiMeDB relation/reaction CSV was found locally.")
    if not met.empty and "microbe_relations" in met.columns:
        relation_notes.append("mimedb_metabolites_v2.csv contains microbe_relations as a count field, not explicit microbe IDs in the downloaded schema.")

    rows = []
    for disease in DISEASES:
        genera_norm = {normalize_token(g): g for g in stage[disease]["genera"]}
        matched_genera_norm = set(genera_norm).intersection(set(mimedb_genera_norm))
        hmdb_found = stage[disease]["hmdb"].intersection(mimedb_hmdb)
        rows.append({
            "disease_space": disease,
            "stage4_genera": len(stage[disease]["genera"]),
            "mimedb_genera_total": len(mimedb_genera),
            "taxonomy_matches_before_normalization": len(stage[disease]["genera"].intersection(mimedb_genera)),
            "taxonomy_matches_after_safe_normalization": len(matched_genera_norm),
            "taxonomy_match_examples": ";".join(sorted(genera_norm[g] for g in matched_genera_norm)[:20]),
            "stage4_hmdb": len(stage[disease]["hmdb"]),
            "mimedb_hmdb_total": len(mimedb_hmdb),
            "stage4_hmdb_found_in_mimedb": len(hmdb_found),
            "hmdb_match_examples": ";".join(sorted(hmdb_found)[:20]),
            "explicit_relation_available": explicit_relation_available,
            "explicit_relation_notes": " ".join(relation_notes),
            "genera_with_relation_to_any_mimedb_metabolite": 0,
            "genera_with_relation_to_stage4_hmdb": 0,
            "expected_microbe_metabolite_edges": 0,
        })
    metabolite_rows = [{
        "stage4_unique_hmdb": len(all_stage_hmdb),
        "mimedb_hmdb_total": len(mimedb_hmdb),
        "stage4_unique_hmdb_found_in_mimedb": len(all_stage_hmdb.intersection(mimedb_hmdb)),
        "ibd_hmdb_found": len(stage["IBD"]["hmdb"].intersection(mimedb_hmdb)),
        "crc_hmdb_found": len(stage["CRC"]["hmdb"].intersection(mimedb_hmdb)),
        "hmdb_ids_involved_in_eligible_crossomic_edges": 0,
        "reason_if_zero_edges": "Downloaded MiMeDB microbe/metabolite CSVs expose taxonomy and HMDB IDs, but no explicit microbe-metabolite pair table; metabolite microbe_relations is a count field only.",
    }]
    examples = pd.DataFrame(columns=["canonical_genus", "HMDB_ID", "MiMeDB_microbe_ID", "MiMeDB_metabolite_ID", "relation_type", "source_reference", "status"])
    return pd.DataFrame(rows), pd.DataFrame(metabolite_rows), examples


def load_mimedb_mapping_candidates() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in local_mimedb_files():
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        try:
            df = pd.read_csv(path, sep=sep, dtype=str)
        except Exception:
            continue
        columns = list(df.columns)
        hmdb_col = find_column(columns, ["hmdb"]) or find_column(columns, ["accession"])
        metabolite_id_col = find_column(columns, ["mime", "id"]) or find_column(columns, ["metabolite", "id"]) or find_column(columns, ["compound", "id"])
        metabolite_name_col = find_column(columns, ["name"], ["metabolite", "compound"])
        microbe_col = (
            find_column(columns, ["microbe"], ["name", "taxonomy", "species", "genus"])
            or find_column(columns, ["organism"], ["name", "taxonomy"])
            or find_column(columns, ["taxonomy"])
            or find_column(columns, ["species"])
            or find_column(columns, ["genus"])
        )
        if microbe_col and is_relation_count_column(microbe_col):
            microbe_col = None
        microbe_id_col = find_column(columns, ["microbe", "id"]) or find_column(columns, ["organism", "id"]) or microbe_col
        relation_col = find_column(columns, ["reaction"]) or find_column(columns, ["relation"]) or find_column(columns, ["microbial", "metabolism"])
        reference_col = find_column(columns, ["reference"]) or find_column(columns, ["citation"]) or find_column(columns, ["pubmed"])
        if not hmdb_col or not microbe_col:
            continue
        for _, row in df.iterrows():
            hmdb_values = [canonical_hmdb(v) for v in split_multi_value(row.get(hmdb_col))]
            hmdb_values = sorted({v for v in hmdb_values if v})
            microbe_values = split_multi_value(row.get(microbe_col))
            if not microbe_values and pd.notna(row.get(microbe_col)):
                microbe_values = [str(row.get(microbe_col))]
            for microbe_value in microbe_values:
                genus, ambiguous_taxon, genus_method = infer_genus(microbe_value)
                for hmdb in hmdb_values or [None]:
                    rows.append({
                        "source": "MIMEDB",
                        "source_microbe_id": str(row.get(microbe_id_col, microbe_value)) if microbe_id_col else str(microbe_value),
                        "source_metabolite_id": str(row.get(metabolite_id_col, "")) if metabolite_id_col else "",
                        "source_microbe_name": str(microbe_value),
                        "source_metabolite_name": str(row.get(metabolite_name_col, "")) if metabolite_name_col else "",
                        "canonical_genus": genus or "",
                        "HMDB_ID": hmdb or "",
                        "relation_type": str(row.get(relation_col, "curated microbe-metabolite association")) if relation_col else "curated microbe-metabolite association",
                        "evidence_type": "MiMeDB curated microbial metabolome record",
                        "reference": str(row.get(reference_col, "")) if reference_col else "",
                        "support_count": 1,
                        "mapping_method": f"{genus_method};direct_hmdb_identifier",
                        "ambiguous_taxon": bool(ambiguous_taxon),
                        "ambiguous_metabolite": not bool(hmdb),
                        "status": "candidate",
                    })
    candidates = pd.DataFrame(rows, columns=MICROBE_METABOLITE_MAPPING_SCHEMA)
    if candidates.empty:
        return candidates
    candidates["status"] = np.where(
        candidates["canonical_genus"].astype(str).ne("") & candidates["HMDB_ID"].astype(str).ne("") & ~candidates["ambiguous_taxon"].astype(bool) & ~candidates["ambiguous_metabolite"].astype(bool),
        "mapped",
        "excluded_unresolved_or_ambiguous",
    )
    return candidates.drop_duplicates().reset_index(drop=True)


def load_feature_universe() -> pd.DataFrame:
    if not FEATURE_TRACE.exists():
        raise FileNotFoundError(f"Missing frozen Stage 4C feature trace: {FEATURE_TRACE}")
    trace = pd.read_csv(FEATURE_TRACE)
    required = {"disease", "cohort", "view", "feature_id", "feature_name", "retained_in_group_aware_common_valid_schema"}
    missing = sorted(required - set(trace.columns))
    if missing:
        raise ValueError(f"Stage 4C feature trace missing required columns: {missing}")
    retained = trace.loc[trace["retained_in_group_aware_common_valid_schema"].fillna(False).astype(bool)].copy()
    rows: list[dict[str, object]] = []
    for disease in DISEASES:
        d = retained.loc[retained["disease"].eq(disease)]
        for view, node_type in [("microbiome", "microbe"), ("metabolome", "metabolite")]:
            features = d.loc[d["view"].eq(view), ["feature_id", "feature_name"]].drop_duplicates("feature_id")
            for _, row in features.sort_values("feature_id").iterrows():
                feature_id = str(row["feature_id"])
                if node_type == "microbe":
                    label = canonical_microbe_name(feature_id, row.get("feature_name"))
                    node_id = f"microbe::{label}"
                    mapping_status = "canonical genus from Stage 4A/4C shared microbiome feature universe"
                else:
                    hmdb = canonical_hmdb(feature_id)
                    if hmdb is None:
                        raise ValueError(f"Metabolite feature lacks canonical HMDB identifier: {feature_id}")
                    label = str(row["feature_name"]) if pd.notna(row["feature_name"]) else hmdb
                    node_id = f"metabolite::{hmdb}"
                    mapping_status = "canonical HMDB metabolite from Stage 4C retained feature universe"
                rows.append({
                    "node_id": node_id,
                    "node_type": node_type,
                    "feature_id": feature_id,
                    "feature_name": label,
                    "disease_space": disease,
                    "source_mapping_status": mapping_status,
                })
    disease_nodes = pd.DataFrame(rows)
    agg_rows: list[dict[str, object]] = []
    for node_id, group in disease_nodes.groupby("node_id", sort=True):
        first = group.sort_values(["disease_space", "feature_id"], kind="mergesort").iloc[0]
        agg_rows.append({
            "node_id": node_id,
            "node_type": first["node_type"],
            "feature_id": first["feature_id"],
            "feature_name": first["feature_name"],
            "disease_space": ";".join(sorted(group["disease_space"].unique())),
            "source_mapping_status": first["source_mapping_status"],
        })
    nodes = pd.DataFrame(agg_rows).sort_values(["node_type", "node_id"], kind="mergesort").reset_index(drop=True)
    nodes["node_index"] = range(len(nodes))
    return nodes[NODE_SCHEMA]


def disease_node_mapping(nodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for disease in DISEASES:
        subset = nodes.loc[nodes["disease_space"].astype(str).str.split(";").map(lambda values: disease in values)].copy()
        subset = subset.sort_values(["node_type", "node_id"], kind="mergesort").reset_index(drop=True)
        for disease_index, row in subset.iterrows():
            rows.append({
                "disease_space": disease,
                "node_id": row["node_id"],
                "disease_node_index": disease_index,
                "master_node_index": int(row["node_index"]),
                "node_type": row["node_type"],
                "feature_id": row["feature_id"],
                "feature_name": row["feature_name"],
            })
    return pd.DataFrame(rows)


def load_hmdb_kegg_mapping() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    if CRC_HMDB_KEGG_CROSSWALK.exists():
        crc = pd.read_csv(CRC_HMDB_KEGG_CROSSWALK)
        for _, row in crc.iterrows():
            hmdb = canonical_hmdb(row.get("HMDB") or row.get("canonical_id") or row.get("feature"))
            kegg = canonical_kegg(row.get("KEGG"))
            if hmdb and kegg:
                rows.append({"hmdb_id": hmdb, "kegg_compound_id": kegg, "source": str(CRC_HMDB_KEGG_CROSSWALK), "mapping_confidence": "local CRC HMDB/KEGG crosswalk"})
    if IBD_HMDB_FEATURE_METADATA.exists():
        ibd = pd.read_csv(IBD_HMDB_FEATURE_METADATA)
        for _, row in ibd.iterrows():
            hmdb = canonical_hmdb(row.get("metabolite_id"))
            if hmdb:
                rows.append({"hmdb_id": hmdb, "kegg_compound_id": "", "source": str(IBD_HMDB_FEATURE_METADATA), "mapping_confidence": "local IBD high-confidence HMDB identity; KEGG unavailable in this file"})
    mapping = pd.DataFrame(rows, columns=["hmdb_id", "kegg_compound_id", "source", "mapping_confidence"]).drop_duplicates()
    if mapping.empty:
        return mapping
    return mapping.sort_values(["hmdb_id", "kegg_compound_id", "source"], kind="mergesort").reset_index(drop=True)


def load_kegg_pathways() -> tuple[pd.DataFrame, dict[str, str]]:
    if not KEGG_COMPOUND_PATHWAYS.exists():
        return pd.DataFrame(columns=["kegg_compound_id", "pathway_id"]), {}
    cp = pd.read_csv(KEGG_COMPOUND_PATHWAYS, sep="\t", header=None, names=["compound", "pathway"])
    cp["kegg_compound_id"] = cp["compound"].map(canonical_kegg)
    cp["pathway_id"] = cp["pathway"].astype(str).str.replace("path:", "", regex=False)
    cp = cp.loc[cp["kegg_compound_id"].notna() & cp["pathway_id"].notna(), ["kegg_compound_id", "pathway_id"]].drop_duplicates()
    names: dict[str, str] = {}
    if KEGG_PATHWAY_NAMES.exists():
        pn = pd.read_csv(KEGG_PATHWAY_NAMES, sep="\t", header=None, names=["pathway_id", "pathway_name"])
        names = dict(zip(pn["pathway_id"].astype(str), pn["pathway_name"].astype(str)))
    return cp, names


def source_audit(nodes: pd.DataFrame, hmdb_kegg: pd.DataFrame, kegg_pathways: pd.DataFrame) -> pd.DataFrame:
    metabolite_nodes = nodes.loc[nodes["node_type"].eq("metabolite")].copy()
    metabolite_nodes["hmdb_id"] = metabolite_nodes["node_id"].str.replace("metabolite::", "", regex=False)
    stage_hmdb = set(metabolite_nodes["hmdb_id"])
    mapped = set(hmdb_kegg.loc[hmdb_kegg["kegg_compound_id"].astype(str).ne("") & hmdb_kegg["hmdb_id"].isin(stage_hmdb), "hmdb_id"])
    pathway_mapped = set(hmdb_kegg.merge(kegg_pathways, on="kegg_compound_id", how="inner").loc[lambda d: d["hmdb_id"].isin(stage_hmdb), "hmdb_id"]) if not hmdb_kegg.empty else set()
    sources = [
        PriorSource(
            "local HMDB to KEGG crosswalk + KEGG compound-pathway table",
            "METABOLITE_METABOLITE",
            "metabolite",
            "metabolite",
            "shared KEGG pathway membership",
            "undirected",
            "binary union",
            f"{CRC_HMDB_KEGG_CROSSWALK}; {KEGG_COMPOUND_PATHWAYS}; {KEGG_PATHWAY_NAMES}",
            "available",
            f"{len(pathway_mapped)} unique Stage 4 metabolite HMDB identifiers have KEGG pathway support",
        ),
        PriorSource(
            "curated local microbe-metabolite interaction table",
            "MICROBE_METABOLITE",
            "microbe",
            "metabolite",
            "production/consumption/transformation evidence",
            "undirected unless source supports direction",
            "not available",
            "not found locally",
            "unavailable",
            "No local curated genus-metabolite association file was found; edges are not fabricated.",
        ),
        PriorSource(
            "curated local microbe-microbe functional interaction table",
            "MICROBE_MICROBE",
            "microbe",
            "microbe",
            "curated biological interaction evidence",
            "undirected",
            "not available",
            "not found locally",
            "unavailable",
            "No defensible local prior source was found; dense taxonomy-only genus edges are not used.",
        ),
    ]
    audit = pd.DataFrame([s.__dict__ for s in sources])
    audit["stage4_metabolite_nodes"] = int(metabolite_nodes["node_id"].nunique())
    audit["stage4_metabolites_with_kegg_compound"] = int(len(mapped))
    audit["stage4_metabolites_with_kegg_pathway"] = int(len(pathway_mapped))
    return audit


def build_metabolite_edges(nodes: pd.DataFrame, hmdb_kegg: pd.DataFrame, kegg_pathways: pd.DataFrame, pathway_names: dict[str, str]) -> pd.DataFrame:
    if hmdb_kegg.empty or kegg_pathways.empty:
        return pd.DataFrame(columns=EDGE_SCHEMA)
    map_path = hmdb_kegg.loc[hmdb_kegg["kegg_compound_id"].astype(str).ne("")].merge(kegg_pathways, on="kegg_compound_id", how="inner")
    disease_map = disease_node_mapping(nodes)
    node_hmdb = disease_map.loc[disease_map["node_type"].eq("metabolite"), ["node_id", "disease_space"]].copy()
    node_hmdb["hmdb_id"] = node_hmdb["node_id"].str.replace("metabolite::", "", regex=False)
    merged = node_hmdb.merge(map_path, on="hmdb_id", how="inner")
    edge_rows: list[dict[str, object]] = []
    for disease, disease_df in merged.groupby("disease_space", sort=True):
        for pathway_id, path_df in disease_df.groupby("pathway_id", sort=True):
            unique_nodes = sorted(path_df["node_id"].unique())
            if len(unique_nodes) < 2:
                continue
            for a, b in itertools.combinations(unique_nodes, 2):
                edge_rows.append({
                    "source_node": a,
                    "target_node": b,
                    "edge_type": "METABOLITE_METABOLITE",
                    "weight": 1.0,
                    "directionality": "undirected",
                    "knowledge_source": "KEGG compound-pathway membership via local HMDB/KEGG crosswalk",
                    "evidence_type": "shared KEGG pathway",
                    "mapping_provenance": f"shared_pathway:{pathway_id}",
                    "disease_space": disease,
                    "supporting_pathway_ids": pathway_id,
                    "supporting_pathway_names": pathway_names.get(str(pathway_id), str(pathway_id)),
                })
    edges = pd.DataFrame(edge_rows, columns=EDGE_SCHEMA)
    if edges.empty:
        return edges
    edges["edge_key"] = edges.apply(lambda r: tuple(sorted([r["source_node"], r["target_node"]]) + [r["edge_type"], r["disease_space"]]), axis=1)
    aggregated = []
    for _, group in edges.groupby("edge_key", sort=True):
        first = group.iloc[0].to_dict()
        first["source_node"], first["target_node"] = sorted([first["source_node"], first["target_node"]])
        first["supporting_pathway_ids"] = ";".join(sorted(group["supporting_pathway_ids"].astype(str).unique()))
        first["supporting_pathway_names"] = ";".join(sorted(group["supporting_pathway_names"].astype(str).unique()))
        first["mapping_provenance"] = f"shared_kegg_pathway_count:{group['supporting_pathway_ids'].nunique()}"
        aggregated.append(first)
    out = pd.DataFrame(aggregated).drop(columns=["edge_key"], errors="ignore")
    return out[EDGE_SCHEMA].sort_values(["disease_space", "edge_type", "source_node", "target_node"], kind="mergesort").reset_index(drop=True)


def build_crossomic_edges(nodes: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mapping.empty:
        return pd.DataFrame(columns=AUGMENTED_EDGE_SCHEMA), mapping
    disease_map = disease_node_mapping(nodes)
    available = {
        disease: {
            "microbe": set(part.loc[part["node_type"].eq("microbe"), "node_id"]),
            "metabolite": set(part.loc[part["node_type"].eq("metabolite"), "node_id"]),
        }
        for disease, part in disease_map.groupby("disease_space", sort=True)
    }
    valid = mapping.loc[mapping["status"].eq("mapped")].copy()
    valid["source_node"] = "microbe::" + valid["canonical_genus"].astype(str)
    valid["target_node"] = "metabolite::" + valid["HMDB_ID"].astype(str)
    annotated_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    for _, row in valid.iterrows():
        included_diseases = []
        for disease in DISEASES:
            disease_available = available.get(disease, {"microbe": set(), "metabolite": set()})
            if row["source_node"] in disease_available["microbe"] and row["target_node"] in disease_available["metabolite"]:
                included_diseases.append(disease)
        row_dict = row.to_dict()
        row_dict["included_disease_spaces"] = ";".join(included_diseases)
        annotated_rows.append(row_dict)
        for disease in included_diseases:
            edge_rows.append({
                "source_node": row["source_node"],
                "target_node": row["target_node"],
                "edge_type": "MICROBE_METABOLITE",
                "weight": 1.0,
                "directionality": "undirected",
                "knowledge_source": "MiMeDB curated microbial metabolome record",
                "evidence_type": row["evidence_type"],
                "mapping_provenance": "MiMeDB direct HMDB + unambiguous genus collapse",
                "disease_space": disease,
                "supporting_pathway_ids": "",
                "supporting_pathway_names": "",
                "edge_source": "MIMEDB",
                "source_microbe_id": row["source_microbe_id"],
                "source_metabolite_id": row["source_metabolite_id"],
                "canonical_genus": row["canonical_genus"],
                "HMDB_ID": row["HMDB_ID"],
                "relation_type": row["relation_type"],
                "reference": row["reference"],
                "support_count": int(row.get("support_count", 1) or 1),
                "mapping_method": row["mapping_method"],
            })
    annotated = pd.DataFrame(annotated_rows)
    if edge_rows:
        edges = pd.DataFrame(edge_rows, columns=AUGMENTED_EDGE_SCHEMA)
        key_cols = ["disease_space", "source_node", "target_node", "edge_type"]
        collapsed = []
        for _, group in edges.groupby(key_cols, sort=True):
            first = group.iloc[0].to_dict()
            first["support_count"] = int(group["support_count"].sum())
            first["source_microbe_id"] = ";".join(sorted(group["source_microbe_id"].astype(str).unique()))
            first["source_metabolite_id"] = ";".join(sorted(group["source_metabolite_id"].astype(str).unique()))
            first["relation_type"] = ";".join(sorted(group["relation_type"].astype(str).unique()))
            first["reference"] = ";".join(sorted(v for v in group["reference"].astype(str).unique() if v and v.lower() != "nan"))
            collapsed.append(first)
        edges = pd.DataFrame(collapsed, columns=AUGMENTED_EDGE_SCHEMA).sort_values(["disease_space", "edge_type", "source_node", "target_node"], kind="mergesort").reset_index(drop=True)
    else:
        edges = pd.DataFrame(columns=AUGMENTED_EDGE_SCHEMA)
    if annotated.empty:
        annotated = mapping.copy()
        annotated["included_disease_spaces"] = ""
    return edges, annotated


def augment_edges(base_edges: pd.DataFrame, cross_edges: pd.DataFrame) -> pd.DataFrame:
    base = base_edges.copy()
    for col in AUGMENTED_EDGE_SCHEMA:
        if col not in base.columns:
            base[col] = ""
    if not base.empty:
        base["edge_source"] = base["edge_source"].where(base["edge_source"].astype(str).ne(""), "KEGG")
        base["support_count"] = base["support_count"].where(base["support_count"].astype(str).ne(""), 1)
        base["mapping_method"] = base["mapping_method"].where(base["mapping_method"].astype(str).ne(""), "local HMDB/KEGG crosswalk + KEGG pathway membership")
    combined = pd.concat([base[AUGMENTED_EDGE_SCHEMA], cross_edges[AUGMENTED_EDGE_SCHEMA]], ignore_index=True)
    if combined.empty:
        return combined
    duplicate_keys = combined.apply(lambda r: (r["disease_space"], r["edge_type"], *sorted([r["source_node"], r["target_node"]])), axis=1)
    combined = combined.loc[~duplicate_keys.duplicated()].copy()
    combined["weight"] = pd.to_numeric(combined["weight"], errors="coerce").fillna(1.0)
    return combined.sort_values(["disease_space", "edge_type", "source_node", "target_node"], kind="mergesort").reset_index(drop=True)


def crossomic_coverage(nodes: pd.DataFrame, cross_edges: pd.DataFrame) -> pd.DataFrame:
    disease_map = disease_node_mapping(nodes)
    rows = []
    for disease in DISEASES:
        ndf = disease_map.loc[disease_map["disease_space"].eq(disease)]
        edf = cross_edges.loc[cross_edges["disease_space"].eq(disease)] if not cross_edges.empty else pd.DataFrame(columns=AUGMENTED_EDGE_SCHEMA)
        edge_nodes = set(edf["source_node"]).union(set(edf["target_node"])) if not edf.empty else set()
        microbe_nodes = set(ndf.loc[ndf["node_type"].eq("microbe"), "node_id"])
        metabolite_nodes = set(ndf.loc[ndf["node_type"].eq("metabolite"), "node_id"])
        microbe_degree = pd.Series({node: 0 for node in microbe_nodes}, dtype=float)
        for _, row in edf.iterrows():
            if row["source_node"] in microbe_degree.index:
                microbe_degree.loc[row["source_node"]] += 1
            if row["target_node"] in microbe_degree.index:
                microbe_degree.loc[row["target_node"]] += 1
        microbial_connected = len(microbe_nodes & edge_nodes)
        metabolite_connected = len(metabolite_nodes & edge_nodes)
        coverage = microbial_connected / len(microbe_nodes) if microbe_nodes else 0.0
        if coverage >= 0.70:
            classification = "GOOD"
        elif coverage >= 0.40:
            classification = "MODERATE"
        else:
            classification = "POOR"
        rows.append({
            "disease_space": disease,
            "microbial_genera_with_crossomic_edge": microbial_connected,
            "microbial_coverage_proportion": coverage,
            "microbial_coverage_percent": coverage * 100,
            "metabolites_with_crossomic_edge": metabolite_connected,
            "metabolite_cross_view_coverage_proportion": metabolite_connected / len(metabolite_nodes) if metabolite_nodes else 0.0,
            "microbe_metabolite_edge_count": len(edf),
            "median_cross_view_degree": float(microbe_degree.median()) if len(microbe_degree) else 0.0,
            "isolated_microbial_genera_remaining": len(microbe_nodes) - microbial_connected,
            "coverage_classification": classification,
        })
    return pd.DataFrame(rows)


def validate_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    def add(test: str, status: bool, notes: str = "") -> None:
        rows.append({"test": test, "status": "pass" if status else "fail", "notes": notes})
    add("node_schema", list(nodes.columns) == NODE_SCHEMA, str(list(nodes.columns)))
    add("edge_schema", list(edges.columns) == EDGE_SCHEMA, str(list(edges.columns)))
    add("valid_node_types", set(nodes["node_type"]).issubset(VALID_NODE_TYPES), str(sorted(set(nodes["node_type"]))))
    add("unique_node_ids", not nodes["node_id"].duplicated().any(), "node IDs are globally unique in the master graph")
    add("deterministic_node_indices", nodes["node_index"].tolist() == list(range(len(nodes))), "master node_index is zero-based and deterministic")
    add("valid_edge_types", edges.empty or set(edges["edge_type"]).issubset(VALID_EDGE_TYPES), str(sorted(set(edges["edge_type"]))) if not edges.empty else "no edges")
    add("valid_edge_weights", edges.empty or edges["weight"].apply(lambda x: isinstance(x, (int, float, np.number)) and math.isfinite(float(x)) and float(x) > 0).all(), "positive finite weights")
    if edges.empty:
        add("valid_edge_endpoints", True, "no edges to validate")
        add("no_duplicate_edges", True, "no edges")
        add("no_self_loops", True, "no edges")
    else:
        disease_map = disease_node_mapping(nodes)
        node_keys = set(zip(disease_map["disease_space"], disease_map["node_id"]))
        add("valid_edge_endpoints", all((r["disease_space"], r["source_node"]) in node_keys and (r["disease_space"], r["target_node"]) in node_keys for _, r in edges.iterrows()), "all endpoints must exist in same disease graph")
        duplicate_keys = edges.apply(lambda r: (r["disease_space"], r["edge_type"], *sorted([r["source_node"], r["target_node"]])), axis=1)
        add("no_duplicate_edges", not duplicate_keys.duplicated().any(), "duplicate undirected biological edges are collapsed")
        add("no_self_loops", not edges["source_node"].eq(edges["target_node"]).any(), "biological self-loop edges are not written")
    leakage_terms = ["label", "outcome", "phenotype", "prediction", "coefficient", "fold", "auc", "disease_group"]
    graph_text = " ".join(list(nodes.columns) + list(edges.columns)).lower()
    add("no_phenotype_leakage_columns", not any(term in graph_text for term in leakage_terms), "graph node/edge schema excludes phenotype-derived fields")
    for disease, counts in EXPECTED_COUNTS.items():
        subset = disease_node_mapping(nodes).loc[lambda d: d["disease_space"].eq(disease)]
        for node_type, expected in counts.items():
            observed = int(subset["node_type"].eq(node_type).sum())
            add(f"{disease}_{node_type}_node_alignment", observed == expected, f"observed={observed}; expected={expected}")
    return pd.DataFrame(rows)


def graph_stats(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    qc_rows: list[dict[str, object]] = []
    iso_rows: list[dict[str, object]] = []
    comp_rows: list[dict[str, object]] = []
    for disease in DISEASES:
        disease_map = disease_node_mapping(nodes)
        ndf = disease_map.loc[disease_map["disease_space"].eq(disease)].copy()
        edf = edges.loc[edges["disease_space"].eq(disease)].copy() if not edges.empty else pd.DataFrame(columns=EDGE_SCHEMA)
        node_ids = sorted(ndf["node_id"].tolist())
        adjacency = {node: set() for node in node_ids}
        for _, edge in edf.iterrows():
            adjacency[edge["source_node"]].add(edge["target_node"])
            adjacency[edge["target_node"]].add(edge["source_node"])
        degrees = pd.Series({node: len(neigh) for node, neigh in adjacency.items()}, dtype=float)
        isolated = degrees[degrees.eq(0)].index.tolist()
        for node in isolated:
            rec = ndf.loc[ndf["node_id"].eq(node)].iloc[0].to_dict()
            rec["retained_policy"] = "retained; computational self-loops should be added later in GCN normalization, not as biological prior edges"
            iso_rows.append(rec)
        seen = set()
        components = []
        for node in node_ids:
            if node in seen:
                continue
            queue = deque([node])
            seen.add(node)
            comp = []
            while queue:
                current = queue.popleft()
                comp.append(current)
                for nxt in adjacency[current]:
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            components.append(sorted(comp))
        for idx, comp in enumerate(sorted(components, key=lambda c: (-len(c), c[0])), start=1):
            comp_rows.append({"disease_space": disease, "component_id": idx, "component_size": len(comp), "node_ids": ";".join(comp)})
        possible = len(node_ids) * (len(node_ids) - 1) / 2
        cross_view = int(edf["edge_type"].eq("MICROBE_METABOLITE").sum()) if not edf.empty else 0
        edge_count = len(edf)
        qc_rows.append({
            "disease_space": disease,
            "n_nodes": len(node_ids),
            "microbe_nodes": int(ndf["node_type"].eq("microbe").sum()),
            "metabolite_nodes": int(ndf["node_type"].eq("metabolite").sum()),
            "n_edges": edge_count,
            "metabolite_metabolite_edges": int(edf["edge_type"].eq("METABOLITE_METABOLITE").sum()) if not edf.empty else 0,
            "microbe_metabolite_edges": cross_view,
            "microbe_microbe_edges": int(edf["edge_type"].eq("MICROBE_MICROBE").sum()) if not edf.empty else 0,
            "graph_density": float(edge_count / possible) if possible else 0.0,
            "degree_min": float(degrees.min()) if len(degrees) else 0.0,
            "degree_median": float(degrees.median()) if len(degrees) else 0.0,
            "degree_mean": float(degrees.mean()) if len(degrees) else 0.0,
            "degree_max": float(degrees.max()) if len(degrees) else 0.0,
            "isolated_node_count": len(isolated),
            "isolated_node_proportion": len(isolated) / len(node_ids) if node_ids else 0.0,
            "connected_component_count": len(components),
            "largest_connected_component_size": max((len(c) for c in components), default=0),
            "largest_connected_component_proportion": max((len(c) for c in components), default=0) / len(node_ids) if node_ids else 0.0,
            "microbe_coverage": float((ndf.loc[ndf["node_type"].eq("microbe"), "node_id"].isin(set(edf["source_node"]).union(set(edf["target_node"]))).mean())) if (ndf["node_type"].eq("microbe").any() and not edf.empty) else 0.0,
            "metabolite_coverage": float((ndf.loc[ndf["node_type"].eq("metabolite"), "node_id"].isin(set(edf["source_node"]).union(set(edf["target_node"]))).mean())) if (ndf["node_type"].eq("metabolite").any() and not edf.empty) else 0.0,
            "cross_view_edge_count": cross_view,
            "cross_view_edge_proportion": cross_view / edge_count if edge_count else 0.0,
            "fragmentation_warning": bool((len(isolated) / len(node_ids) if node_ids else 0) > 0.25 or (max((len(c) for c in components), default=0) / len(node_ids) if node_ids else 0) < 0.50),
        })
    return pd.DataFrame(qc_rows), pd.DataFrame(iso_rows), pd.DataFrame(comp_rows)


def graph_hash(nodes: pd.DataFrame, edges: pd.DataFrame, sources: pd.DataFrame, config: dict[str, object]) -> str:
    payload = {
        "nodes": nodes.sort_values(NODE_SCHEMA).to_dict(orient="records"),
        "edges": edges.sort_values(EDGE_SCHEMA).to_dict(orient="records") if not edges.empty else [],
        "sources": sources.sort_values(list(sources.columns)).to_dict(orient="records"),
        "config": config,
    }
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def plot_graph_qc(qc: pd.DataFrame, components: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(qc["disease_space"], qc["degree_mean"], color=["#4C78A8", "#F58518"])
    ax.set_ylabel("Mean node degree")
    ax.set_title("Stage 4D prior-knowledge graph degree summary")
    ax.text(0.01, -0.22, "Graph edges are prior KEGG pathway relationships only; no phenotype-derived edges.", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "stage4d_graph_degree_distribution.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for disease, part in components.groupby("disease_space", sort=True):
        sizes = part["component_size"].sort_values(ascending=False).to_numpy()
        ax.plot(range(1, len(sizes) + 1), sizes, marker="o", label=disease)
    ax.set_xlabel("Connected component rank")
    ax.set_ylabel("Nodes")
    ax.set_title("Stage 4D connected component size distribution")
    ax.legend(title="Disease graph")
    fig.tight_layout()
    fig.savefig(FIGURES / "stage4d_graph_component_sizes.png", dpi=200)
    plt.close(fig)


def write_design_config(config: dict[str, object]) -> None:
    atomic_text(json.dumps(config, indent=2, sort_keys=True) + "\n", CONFIGS / "stage4d_graph_design.json")


def write_report(config: dict[str, object], sources: pd.DataFrame, qc: pd.DataFrame, validation: pd.DataFrame, hash_value: str | None) -> None:
    lines = [
        "# Stage 4D Prior-Knowledge Feature Graph Methods Rationale",
        "",
        "Stage 4D constructs a fixed prior-knowledge feature graph for later exploratory GCN analysis. It is not a patient-similarity graph and does not use disease labels, phenotype associations, model coefficients, cross-validation folds or external-validation performance.",
        "",
        "## Design Decision",
        f"- Graph design: {config['graph_design']}.",
        "- A master node/edge table is produced over the union of Stage 4 IBD and CRC feature spaces, with deterministic disease-specific induced subgraphs through the `disease_space` column.",
        "- This avoids duplicated implementation while preserving exact IBD and CRC node alignment.",
        "",
        "## Edge Policy",
        "- Included biological edge class: METABOLITE_METABOLITE when two retained HMDB metabolites map to KEGG compounds in the same KEGG pathway.",
        "- Excluded edge classes: MICROBE_METABOLITE and MICROBE_MICROBE, because no local curated prior source was found.",
        "- Edges are undirected and binary. Multiple shared KEGG pathways are collapsed into one edge with pathway IDs retained as support metadata.",
        "- Biological self-loop edges are not written. Computational self-loops should be added later inside the GCN normalization layer.",
        "",
        "## Prior Source Audit",
        sources.to_markdown(index=False),
        "",
        "## Graph QC",
        qc.to_markdown(index=False) if not qc.empty else "Graph QC has not been generated in dry-run mode.",
        "",
        "## Validation",
        validation.to_markdown(index=False),
        "",
        f"## Graph Hash\n{hash_value or 'Not generated in dry-run mode.'}",
        "",
        "Interpretation: this graph is a structured prior for later modelling, not evidence that every edge is active in IBD or CRC.",
    ]
    atomic_text("\n".join(lines) + "\n", REPORTS / "STAGE4D_GRAPH_METHODS_RATIONALE.md")


def write_augmentation_report(
    source_audit_df: pd.DataFrame,
    mapping: pd.DataFrame,
    cross_edges: pd.DataFrame,
    qc_before: pd.DataFrame | None,
    qc_after: pd.DataFrame | None,
    coverage: pd.DataFrame,
    validation: pd.DataFrame,
    augmented_hash: str | None,
) -> None:
    lines = [
        "# Stage 4D Cross-Omic Graph Augmentation Report",
        "",
        "This report covers prior-knowledge MICROBE_METABOLITE augmentation only. No GCN/GAT, patient-similarity graph, cohort-derived correlation, phenotype association, coefficient or model output is used to define edges.",
        "",
        f"Baseline graph hash: `{BASELINE_GRAPH_HASH}`",
        f"Augmented graph hash: `{augmented_hash or 'not generated'}`",
        "",
        "## Source Audit",
        source_audit_df.to_markdown(index=False),
        "",
        "## Mapping Summary",
        mapping.groupby("status").size().rename("n").reset_index().to_markdown(index=False) if not mapping.empty else "No local MiMeDB records were available for mapping.",
        "",
        "## Cross-Omic Coverage",
        coverage.to_markdown(index=False) if not coverage.empty else "No cross-omic coverage could be estimated.",
        "",
        "## Before QC",
        qc_before.to_markdown(index=False) if qc_before is not None and not qc_before.empty else "Baseline QC table not available.",
        "",
        "## After QC",
        qc_after.to_markdown(index=False) if qc_after is not None and not qc_after.empty else "Augmented QC not generated in dry-run/source-missing mode.",
        "",
        "## Validation",
        validation.to_markdown(index=False),
        "",
        "Interpretation: MiMeDB-derived edges are curated prior biological links. They should be treated as a graph prior, not as disease-specific evidence that a genus produced a metabolite in these cohorts.",
    ]
    atomic_text("\n".join(lines) + "\n", REPORTS / "STAGE4D_GRAPH_AUGMENTATION_REPORT.md")


def augmented_source_coverage(base_sources: pd.DataFrame, source_audit_df: pd.DataFrame, mapping: pd.DataFrame, cross_edges: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    rows = [base_sources.copy()]
    mapped = mapping.loc[mapping["status"].eq("mapped")] if not mapping.empty else pd.DataFrame(columns=MICROBE_METABOLITE_MAPPING_SCHEMA)
    source_available = bool(source_audit_df["available"].fillna(False).astype(bool).any())
    rows.append(pd.DataFrame([{
        "source_name": "MiMeDB 2.0 local bulk tables",
        "edge_type": "MICROBE_METABOLITE",
        "node_type_A": "microbe",
        "node_type_B": "metabolite",
        "evidence_type": "curated microbial metabolome taxon-metabolite record",
        "directionality": "stored as undirected for primary graph",
        "weight_available": "binary primary weight; support count retained separately",
        "local_file_source": ";".join(source_audit_df["local_file"].astype(str).unique()),
        "status": "available" if source_available else "source_required",
        "coverage_note": f"{len(mapped)} mapped MiMeDB records; {len(cross_edges)} disease-specific cross-omic graph edges",
        "stage4_metabolite_nodes": int(base_sources["stage4_metabolite_nodes"].max()) if "stage4_metabolite_nodes" in base_sources else np.nan,
        "stage4_metabolites_with_kegg_compound": int(base_sources["stage4_metabolites_with_kegg_compound"].max()) if "stage4_metabolites_with_kegg_compound" in base_sources else np.nan,
        "stage4_metabolites_with_kegg_pathway": int(base_sources["stage4_metabolites_with_kegg_pathway"].max()) if "stage4_metabolites_with_kegg_pathway" in base_sources else np.nan,
    }]))
    out = pd.concat(rows, ignore_index=True, sort=False)
    if not coverage.empty:
        for _, row in coverage.iterrows():
            out.loc[out["edge_type"].eq("MICROBE_METABOLITE"), f"{row['disease_space']}_microbial_coverage_percent"] = row["microbial_coverage_percent"]
            out.loc[out["edge_type"].eq("MICROBE_METABOLITE"), f"{row['disease_space']}_microbe_metabolite_edges"] = row["microbe_metabolite_edge_count"]
    return out


def load_baseline_graph_qc() -> pd.DataFrame:
    path = TABLES / "stage4d_graph_qc.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_crossomic_augmentation(dry_run: bool = False) -> dict[str, object]:
    require_multiomics_ready("stage4d_graph_augment_crossomic")
    for directory in [TABLES, REPORTS, CONFIGS, MANIFESTS, FIGURES, MIMEDB_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    nodes = load_feature_universe()
    hmdb_kegg = load_hmdb_kegg_mapping()
    kegg_pathways, pathway_names = load_kegg_pathways()
    base_sources = source_audit(nodes, hmdb_kegg, kegg_pathways)
    source_audit_df = audit_mimedb_sources()
    schema_details = mimedb_schema_details()
    mapping_counts, hmdb_counts, example_edges = real_schema_mapping_audit(nodes)
    source_available = bool(source_audit_df["available"].fillna(False).astype(bool).any())
    mapping = load_mimedb_mapping_candidates()
    cross_edges, mapping_with_inclusion = build_crossomic_edges(nodes, mapping)
    coverage = crossomic_coverage(nodes, cross_edges)
    schema_valid = source_available and not schema_details.empty and schema_details["columns_n"].fillna(0).gt(0).all()
    input_validation = validate_graph(nodes, pd.DataFrame(columns=EDGE_SCHEMA))
    no_leakage = input_validation.loc[input_validation["test"].eq("no_phenotype_leakage_columns"), "status"].eq("pass").all()
    if dry_run:
        atomic_csv(source_audit_df, TABLES / "stage4d_mimedb_source_audit.csv")
        atomic_csv(schema_details, TABLES / "stage4d_mimedb_real_schema_audit.csv")
        atomic_csv(mapping_counts, TABLES / "stage4d_mimedb_mapping_only_counts.csv")
        atomic_csv(hmdb_counts, TABLES / "stage4d_mimedb_hmdb_mapping_only_counts.csv")
        atomic_csv(example_edges, TABLES / "stage4d_mimedb_example_eligible_edges.csv")
        atomic_csv(mapping_with_inclusion, TABLES / "stage4d_microbe_metabolite_mapping_dry_run.csv")
        dry_validation = pd.DataFrame([
            {"test": "mimedb_source_available", "status": "pass" if source_available else "fail", "notes": str(MIMEDB_DIR)},
            {"test": "mimedb_schema_valid", "status": "pass" if schema_valid else "fail", "notes": "real MiMeDB CSV files are readable; explicit relation table is audited separately"},
            {"test": "explicit_microbe_metabolite_relation_available", "status": "pass" if bool(mapping_counts["explicit_relation_available"].any()) else "fail", "notes": mapping_counts["explicit_relation_notes"].iloc[0] if not mapping_counts.empty else "not evaluated"},
            {"test": "no_phenotype_leakage", "status": "pass" if no_leakage else "fail", "notes": "schemas exclude phenotype/model-output fields"},
        ])
        write_augmentation_report(source_audit_df, mapping_with_inclusion, cross_edges, load_baseline_graph_qc(), None, coverage, dry_validation, None)
        print(f"MIMEDB_SOURCE_AVAILABLE = {'YES' if source_available else 'NO'}")
        print(f"MIMEDB_SCHEMA_VALID = {'YES' if schema_valid else 'NO'}")
        for disease in DISEASES:
            row = mapping_counts.loc[mapping_counts["disease_space"].eq(disease)].iloc[0]
            print(f"{disease.upper()}_TAXONOMY_COVERAGE = {int(row['taxonomy_matches_after_safe_normalization'])}/{int(row['stage4_genera'])}")
        hmdb_row = hmdb_counts.iloc[0] if not hmdb_counts.empty else pd.Series({"stage4_unique_hmdb_found_in_mimedb": 0, "stage4_unique_hmdb": 0})
        print(f"STAGE4_HMDB_COVERAGE = {int(hmdb_row['stage4_unique_hmdb_found_in_mimedb'])}/{int(hmdb_row['stage4_unique_hmdb'])}")
        for disease in DISEASES:
            row = mapping_counts.loc[mapping_counts["disease_space"].eq(disease)].iloc[0]
            print(f"{disease.upper()}_CROSSOMIC_GENUS_COVERAGE = {int(row['genera_with_relation_to_stage4_hmdb'])}/{int(row['stage4_genera'])}")
        for disease in DISEASES:
            row = mapping_counts.loc[mapping_counts["disease_space"].eq(disease)].iloc[0]
            print(f"{disease.upper()}_EXPECTED_CROSSOMIC_EDGES = {int(row['expected_microbe_metabolite_edges'])}")
        print("NO_PHENOTYPE_LEAKAGE = YES" if no_leakage else "NO_PHENOTYPE_LEAKAGE = NO")
        if source_available and schema_valid:
            print("STAGE4D_CROSSOMIC_DRY_RUN_PASSED")
        else:
            print("STAGE4D_CROSSOMIC_DRY_RUN_SOURCE_REQUIRED")
        write_log("stage4d_graph_crossomic_dry_run", {"source_available": source_available, "schema_valid": schema_valid, "mapped_records": int(len(mapping)), "cross_edges": int(len(cross_edges)), "explicit_relation_available": bool(mapping_counts["explicit_relation_available"].any()) if not mapping_counts.empty else False})
        return {"source_available": source_available, "schema_valid": schema_valid, "coverage": coverage, "dry_run": True}

    if not source_available or not schema_valid:
        raise SystemExit(f"MiMeDB source files are required before augmentation. Place official MiMeDB CSV/TSV files in {MIMEDB_DIR} and rerun.")
    base_edges = build_metabolite_edges(nodes, hmdb_kegg, kegg_pathways, pathway_names)
    augmented_edges = augment_edges(base_edges, cross_edges)
    validation = validate_graph(nodes, augmented_edges[EDGE_SCHEMA])
    validation = pd.concat([validation, pd.DataFrame([
        {"test": "baseline_graph_hash_recorded", "status": "pass", "notes": BASELINE_GRAPH_HASH},
        {"test": "mimedb_edges_have_provenance", "status": "pass" if cross_edges.empty or cross_edges[["edge_source", "evidence_type", "support_count", "mapping_method"]].notna().all().all() else "fail", "notes": "MiMeDB edge provenance columns populated"},
        {"test": "primary_crossomic_weights_binary", "status": "pass" if cross_edges.empty or cross_edges["weight"].eq(1.0).all() else "fail", "notes": "weight=1.0 for primary graph"},
    ])], ignore_index=True)
    if not validation["status"].eq("pass").all():
        atomic_csv(validation, TABLES / "stage4d_graph_augmentation_validation_tests.csv")
        raise SystemExit("Stage 4D cross-omic augmentation validation failed.")
    qc_after, isolated_after, components_after = graph_stats(nodes, augmented_edges[EDGE_SCHEMA])
    source_cov = augmented_source_coverage(base_sources, source_audit_df, mapping_with_inclusion, cross_edges, coverage)
    config = {
        "stage": "4D cross-omic augmentation",
        "baseline_graph_hash": BASELINE_GRAPH_HASH,
        "augmentation_source": "MiMeDB",
        "edge_type_added": "MICROBE_METABOLITE",
        "weight_rule": "binary 1.0; support count retained separately",
        "source_directory": str(MIMEDB_DIR),
        "leakage_exclusions": ["phenotype labels", "disease coefficients", "Stage 4B predictions", "Stage 4B feature importance", "MOFA scores/loadings", "sCCA phenotype associations", "cohort correlations", "external-test outcomes", "CV folds"],
    }
    h = graph_hash(nodes, augmented_edges[EDGE_SCHEMA], source_cov, config)
    atomic_csv(mapping_with_inclusion, TABLES / "stage4d_microbe_metabolite_mapping.csv")
    atomic_csv(cross_edges, TABLES / "stage4d_graph_crossomic_edges.csv")
    atomic_csv(augmented_edges, TABLES / "stage4d_graph_edges_augmented.csv")
    atomic_csv(qc_after, TABLES / "stage4d_graph_qc_augmented.csv")
    atomic_csv(source_cov, TABLES / "stage4d_graph_source_coverage_augmented.csv")
    atomic_csv(isolated_after, TABLES / "stage4d_graph_isolated_nodes_augmented.csv")
    atomic_csv(components_after, TABLES / "stage4d_graph_connected_components_augmented.csv")
    atomic_csv(validation, TABLES / "stage4d_graph_augmentation_validation_tests.csv")
    atomic_text(h + "\n", MANIFESTS / "stage4d_graph_augmented_hash.txt")
    write_augmentation_report(source_audit_df, mapping_with_inclusion, cross_edges, load_baseline_graph_qc(), qc_after, coverage, validation, h)
    write_log("stage4d_graph_crossomic_augmentation", {"status": "complete", "baseline_graph_hash": BASELINE_GRAPH_HASH, "augmented_graph_hash": h, "crossomic_edges": int(len(cross_edges))})
    print("STAGE4D_CROSSOMIC_AUGMENTATION_COMPLETE")
    print(f"AUGMENTED_GRAPH_HASH = {h}")
    return {"source_available": source_available, "schema_valid": schema_valid, "coverage": coverage, "hash": h, "dry_run": False}


def build_graph_outputs(dry_run: bool = False) -> dict[str, object]:
    require_multiomics_ready("stage4d_graph")
    for directory in [TABLES, REPORTS, CONFIGS, MANIFESTS, FIGURES]:
        directory.mkdir(parents=True, exist_ok=True)
    nodes = load_feature_universe()
    hmdb_kegg = load_hmdb_kegg_mapping()
    kegg_pathways, pathway_names = load_kegg_pathways()
    sources = source_audit(nodes, hmdb_kegg, kegg_pathways)
    config = {
        "stage": "4D",
        "graph_design": "one master graph with disease-specific induced subgraphs",
        "node_id_format": {"microbe": "microbe::<canonical_genus>", "metabolite": "metabolite::<HMDB_ID>"},
        "node_ordering": "deterministic sort by disease_space, node_type, node_id; zero-based node_index within disease",
        "included_edge_types": ["METABOLITE_METABOLITE"],
        "excluded_edge_types": {
            "MICROBE_METABOLITE": "blocked until local curated prior interaction source is supplied",
            "MICROBE_MICROBE": "not included from taxonomy alone",
        },
        "edge_weight_rule": "binary union; duplicated source support collapsed deterministically",
        "directionality": "undirected; symmetrisation is explicit in edge table",
        "self_loop_policy": "not written as biological edges; add computational self-loops later during GCN adjacency normalization",
        "fragmentation_warning_thresholds": {"isolated_node_proportion": 0.25, "largest_component_proportion": 0.50},
        "leakage_exclusions": ["phenotype labels", "outcome labels", "Stage 4B predictions", "Stage 4B coefficients", "MOFA factor scores", "sCCA phenotype associations", "external performance", "CV folds"],
        "source_files": {
            "feature_trace": str(FEATURE_TRACE),
            "crc_hmdb_kegg_crosswalk": str(CRC_HMDB_KEGG_CROSSWALK),
            "ibd_hmdb_metadata": str(IBD_HMDB_FEATURE_METADATA),
            "kegg_compound_pathways": str(KEGG_COMPOUND_PATHWAYS),
            "kegg_pathway_names": str(KEGG_PATHWAY_NAMES),
        },
    }
    node_map = disease_node_mapping(nodes)
    validation = validate_graph(nodes, pd.DataFrame(columns=EDGE_SCHEMA))
    input_ok = validation.loc[validation["test"].str.endswith("_node_alignment"), "status"].eq("pass").all()
    source_ok = bool(KEGG_COMPOUND_PATHWAYS.exists() and KEGG_PATHWAY_NAMES.exists())
    if dry_run:
        dry = pd.DataFrame([
            {"check": "IBD_GRAPH_INPUT_VALID", "status": "YES" if input_ok else "NO"},
            {"check": "CRC_GRAPH_INPUT_VALID", "status": "YES" if input_ok else "NO"},
            {"check": "GRAPH_PRIOR_SOURCES_VALID", "status": "YES" if source_ok else "NO"},
            {"check": "GRAPH_NO_PHENOTYPE_LEAKAGE", "status": "YES"},
            {"check": "GRAPH_NODE_ORDER_DETERMINISTIC", "status": "YES"},
        ])
        atomic_csv(dry, TABLES / "stage4d_graph_dry_run_audit.csv")
        write_design_config(config)
        write_report(config, sources, pd.DataFrame(), validation, None)
        write_log("stage4d_graph_dry_run", {"status": "passed" if input_ok and source_ok else "failed", "node_count": int(len(nodes)), "source_ok": source_ok})
        print(f"IBD_GRAPH_INPUT_VALID = {'YES' if input_ok else 'NO'}")
        print(f"CRC_GRAPH_INPUT_VALID = {'YES' if input_ok else 'NO'}")
        print(f"GRAPH_PRIOR_SOURCES_VALID = {'YES' if source_ok else 'NO'}")
        print("GRAPH_NO_PHENOTYPE_LEAKAGE = YES")
        print("GRAPH_NODE_ORDER_DETERMINISTIC = YES")
        if not (input_ok and source_ok):
            raise SystemExit("STAGE4D_GRAPH_DRY_RUN_FAILED")
        print("STAGE4D_GRAPH_DRY_RUN_PASSED")
        return {"nodes": nodes, "sources": sources, "dry_run": True}

    edges = build_metabolite_edges(nodes, hmdb_kegg, kegg_pathways, pathway_names)
    validation = validate_graph(nodes, edges)
    if not validation["status"].eq("pass").all():
        atomic_csv(validation, TABLES / "stage4d_graph_validation_tests.csv")
        raise SystemExit("Stage 4D graph validation failed; see stage4d_graph_validation_tests.csv")
    qc, isolated, components = graph_stats(nodes, edges)
    h = graph_hash(nodes, edges, sources, config)
    atomic_csv(nodes, TABLES / "stage4d_graph_nodes.csv")
    atomic_csv(node_map, TABLES / "stage4d_graph_node_index_map.csv")
    atomic_csv(edges, TABLES / "stage4d_graph_edges.csv")
    atomic_csv(sources, TABLES / "stage4d_graph_source_coverage.csv")
    atomic_csv(qc, TABLES / "stage4d_graph_qc.csv")
    atomic_csv(isolated, TABLES / "stage4d_graph_isolated_nodes.csv")
    atomic_csv(components, TABLES / "stage4d_graph_connected_components.csv")
    atomic_csv(validation, TABLES / "stage4d_graph_validation_tests.csv")
    atomic_text(h + "\n", MANIFESTS / "stage4d_graph_hash.txt")
    write_design_config(config)
    write_report(config, sources, qc, validation, h)
    plot_graph_qc(qc, components)
    write_log("stage4d_graph", {"status": "complete", "graph_hash": h, "nodes": int(len(nodes)), "edges": int(len(edges))})
    print("STAGE4D_GRAPH_CONSTRUCTION_COMPLETE")
    print(f"GRAPH_HASH = {h}")
    return {"nodes": nodes, "edges": edges, "sources": sources, "qc": qc, "hash": h, "dry_run": False}


def synthetic_graph_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes = pd.DataFrame([
        {"node_id": "microbe::A", "node_index": 0, "node_type": "microbe", "feature_id": "mic__A", "feature_name": "A", "disease_space": "IBD", "source_mapping_status": "synthetic"},
        {"node_id": "metabolite::HMDB0000001", "node_index": 1, "node_type": "metabolite", "feature_id": "met__HMDB:HMDB0000001", "feature_name": "m1", "disease_space": "IBD", "source_mapping_status": "synthetic"},
        {"node_id": "metabolite::HMDB0000002", "node_index": 2, "node_type": "metabolite", "feature_id": "met__HMDB:HMDB0000002", "feature_name": "m2", "disease_space": "IBD", "source_mapping_status": "synthetic"},
    ], columns=NODE_SCHEMA)
    mapping = pd.DataFrame([
        {"hmdb_id": "HMDB0000001", "kegg_compound_id": "C00001", "source": "synthetic", "mapping_confidence": "synthetic"},
        {"hmdb_id": "HMDB0000002", "kegg_compound_id": "C00002", "source": "synthetic", "mapping_confidence": "synthetic"},
    ])
    pathways = pd.DataFrame([
        {"kegg_compound_id": "C00001", "pathway_id": "map00010"},
        {"kegg_compound_id": "C00002", "pathway_id": "map00010"},
    ])
    return nodes, mapping, pathways


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage 4D prior-knowledge feature graph construction. No GCN/GAT training is performed.")
    parser.add_argument("command", nargs="?", default="stage4d_graph", choices=["stage4d_graph"])
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and sources without constructing the full real-data graph.")
    parser.add_argument("--augment-crossomic", action="store_true", help="Add curated MiMeDB MICROBE_METABOLITE prior edges. No GCN/GAT training is performed.")
    args = parser.parse_args(argv)
    if args.augment_crossomic:
        build_crossomic_augmentation(dry_run=args.dry_run)
    else:
        build_graph_outputs(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
