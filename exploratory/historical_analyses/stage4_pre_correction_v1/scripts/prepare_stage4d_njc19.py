from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import pandas as pd

from stage4_common import PROJECT, STAGE4, TABLES, write_log


NJC19_DIR = STAGE4 / "data/stage4d_prior_sources/njc19"
NJC19_ZIP = NJC19_DIR / "NJC19_network_data_and_code.zip"
NJC19_EXTRACTED = NJC19_DIR / "extracted"
IBD_HMDB_KEGG = PROJECT / "thesis_preprocessing_workflow/ibd_pipeline/tables/hmdb_to_kegg_mapping_audit.csv"
CRC_HMDB_KEGG = PROJECT / "thesis_preprocessing_workflow/crc_pipeline/stage2/tables/crc_stage2_hmdb_kegg_crosswalk.csv"
FEATURE_TRACE = TABLES / "stage4c_feature_qc_trace.csv"


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def canonical_hmdb(value: object) -> str | None:
    match = re.search(r"HMDB\d{7}", str(value))
    return match.group(0) if match else None


def canonical_kegg(value: object) -> str | None:
    match = re.search(r"\bC\d{5}\b", str(value))
    return match.group(0) if match else None


def species_to_genus(value: object) -> str:
    text = re.sub(r"[_|;]+", " ", str(value)).strip()
    text = re.sub(r"\s+", " ", text)
    return text.split(" ", 1)[0] if text else ""


def read_json_source(source: Path, filename: str) -> dict:
    if source.is_dir():
        path = source / filename
        return json.loads(path.read_text()) if path.exists() else {}
    if source.exists() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as zf:
            match = next((name for name in zf.namelist() if Path(name).name == filename), None)
            if match is None:
                return {}
            with zf.open(match) as handle:
                return json.loads(handle.read().decode("utf-8"))
    return {}


def discover_njc19_source(source: Path) -> dict[str, dict]:
    return {
        "network": read_json_source(source, "NJC19_network.json"),
        "organism": read_json_source(source, "NJC19_organism.json"),
        "compound": read_json_source(source, "NJC19_compound.json"),
    }


def stage4_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    trace = pd.read_csv(FEATURE_TRACE)
    retained = trace.loc[trace["retained_in_group_aware_common_valid_schema"].fillna(False).astype(bool)].copy()
    microbes = retained.loc[retained["view"].eq("microbiome"), ["disease", "feature_id"]].drop_duplicates()
    microbes["canonical_genus"] = microbes["feature_id"].astype(str).str.replace("mic__", "", regex=False)
    metab = retained.loc[retained["view"].eq("metabolome"), ["disease", "feature_id"]].drop_duplicates()
    metab["HMDB_ID"] = metab["feature_id"].map(canonical_hmdb)
    return microbes, metab.dropna(subset=["HMDB_ID"])


def stage4_kegg_hmdb_crosswalk() -> pd.DataFrame:
    frames = []
    if IBD_HMDB_KEGG.exists():
        ibd = pd.read_csv(IBD_HMDB_KEGG, dtype=str)
        if {"HMDB_ID", "final_KEGG_Compound_ID"}.issubset(ibd.columns):
            frames.append(ibd.rename(columns={"final_KEGG_Compound_ID": "KEGG"})[["HMDB_ID", "KEGG"]])
    if CRC_HMDB_KEGG.exists():
        crc = pd.read_csv(CRC_HMDB_KEGG, dtype=str)
        if {"HMDB", "KEGG"}.issubset(crc.columns):
            frames.append(crc.rename(columns={"HMDB": "HMDB_ID"})[["HMDB_ID", "KEGG"]])
    if not frames:
        return pd.DataFrame(columns=["HMDB_ID", "KEGG"])
    out = pd.concat(frames, ignore_index=True)
    out["HMDB_ID"] = out["HMDB_ID"].map(canonical_hmdb)
    out["KEGG"] = out["KEGG"].map(canonical_kegg)
    return out.dropna().drop_duplicates()


def find_njc19_tables(zip_path: Path = NJC19_ZIP) -> list[pd.DataFrame]:
    if not zip_path.exists() or not zipfile.is_zipfile(zip_path):
        return []
    tables = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith((".csv", ".tsv", ".txt")):
                continue
            with zf.open(name) as handle:
                try:
                    sep = "\t" if name.lower().endswith((".tsv", ".txt")) else ","
                    df = pd.read_csv(handle, sep=sep, dtype=str)
                except Exception:
                    continue
            df["__source_file"] = name
            tables.append(df)
    return tables


def compound_name_to_kegg(compounds: dict) -> dict[str, str]:
    mapping = {}
    for display_name, payload in compounds.items():
        ids = payload.get("KEGG compound identifier", []) if isinstance(payload, dict) else []
        kegg = next((canonical_kegg(x) for x in ids if canonical_kegg(x)), None)
        if not kegg:
            continue
        names = [display_name]
        if isinstance(payload, dict):
            raw_names = payload.get("name", [])
            names.extend(raw_names if isinstance(raw_names, list) else [raw_names])
        for name in names:
            mapping[normalize(name)] = kegg
    return mapping


def extract_json_relations(source: Path = NJC19_EXTRACTED) -> pd.DataFrame:
    src = discover_njc19_source(source)
    network = src["network"]
    organism = src["organism"]
    compounds = src["compound"]
    if not network or not organism or not compounds:
        return pd.DataFrame(columns=["source_file", "source_microbe_name", "canonical_genus", "source_metabolite_id", "relation_type", "reference"])
    compound_map = compound_name_to_kegg(compounds)
    positive_terms = ("production", "consumption", "export", "import", "transport")
    rows = []
    for relation_id, payload in network.items():
        if not isinstance(payload, dict):
            continue
        species = str(payload.get("Species", "")).strip()
        compound_name = str(payload.get("Small-molecule metabolite or macromolecule", "")).strip()
        activity = str(payload.get("Metabolic activity", "")).strip()
        if not species or species not in organism:
            continue
        if not any(term in normalize(activity) for term in positive_terms):
            continue
        kegg = compound_map.get(normalize(compound_name))
        if not kegg:
            continue
        refs = payload.get("Ref. #", [])
        if not isinstance(refs, list):
            refs = [refs]
        rows.append({
            "source_file": "NJC19_network.json",
            "source_relation_id": relation_id,
            "source_microbe_name": species,
            "canonical_genus": species_to_genus(species),
            "source_metabolite_name": compound_name,
            "source_metabolite_id": kegg,
            "relation_type": activity,
            "reference": "Lim et al. Scientific Data 2020; doi:10.1038/s41597-020-0516-5; Dryad doi:10.5061/dryad.dr7sqv9v8",
            "source_reference_ids": ";".join(map(str, refs)),
        })
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=["source_file", "source_relation_id", "source_microbe_name", "canonical_genus", "source_metabolite_name", "source_metabolite_id", "relation_type", "reference", "source_reference_ids"])


def extract_candidate_relations(tables: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    microbe_terms = ("species", "organism", "microbe", "bacteria", "taxon")
    metabolite_terms = ("kegg", "compound", "metabolite")
    relation_terms = ("relation", "interaction", "edge", "transport", "metabolic", "type")
    positive_terms = ("transport", "metabolic", "produce", "consume", "uptake", "secrete", "exchange", "positive")
    for df in tables:
        cols = {c: normalize(c) for c in df.columns}
        microbe_cols = [c for c, n in cols.items() if any(t in n for t in microbe_terms)]
        metabolite_cols = [c for c, n in cols.items() if any(t in n for t in metabolite_terms)]
        relation_cols = [c for c, n in cols.items() if any(t in n for t in relation_terms)]
        if not microbe_cols or not metabolite_cols:
            continue
        for _, row in df.iterrows():
            microbe = next((str(row[c]) for c in microbe_cols if str(row.get(c, "")).strip() and str(row.get(c, "")).lower() != "nan"), "")
            kegg = next((canonical_kegg(row[c]) for c in metabolite_cols if canonical_kegg(row.get(c, ""))), None)
            relation_text = "; ".join(str(row.get(c, "")) for c in relation_cols)
            if not microbe or not kegg:
                continue
            if relation_cols and not any(term in normalize(relation_text) for term in positive_terms):
                continue
            rows.append({
                "source_file": row.get("__source_file", ""),
                "source_microbe_name": microbe,
                "canonical_genus": species_to_genus(microbe),
                "source_metabolite_id": kegg,
                "relation_type": relation_text if relation_text else "explicit_njc19_relation",
                "reference": "Lim et al. Scientific Data 2020; doi:10.1038/s41597-020-0516-5; Dryad doi:10.5061/dryad.dr7sqv9v8",
            })
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=["source_file", "source_microbe_name", "canonical_genus", "source_metabolite_id", "relation_type", "reference"])


def build_outputs(relations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    microbes, metabolites = stage4_features()
    crosswalk = stage4_kegg_hmdb_crosswalk()
    stage4_genera = microbes[["disease", "canonical_genus"]].drop_duplicates()
    stage4_hmdb = metabolites[["disease", "HMDB_ID"]].drop_duplicates()

    if relations.empty:
        taxon = stage4_genera.copy()
        taxon["matched_in_njc19"] = False
        metmap = stage4_hmdb.copy()
        metmap["KEGG"] = ""
        metmap["matched_in_njc19"] = False
        edges = pd.DataFrame(columns=["disease", "canonical_genus", "HMDB_ID", "source", "source_microbe_names", "source_kegg_compound", "relation_type", "reference", "support_count"])
    else:
        rel = relations.copy()
        rel["_genus_norm"] = rel["canonical_genus"].map(normalize)
        stage4_genera["_genus_norm"] = stage4_genera["canonical_genus"].map(normalize)
        rel = rel.merge(stage4_genera, on="_genus_norm", how="inner", suffixes=("", "_stage4"))
        rel = rel.merge(crosswalk, left_on="source_metabolite_id", right_on="KEGG", how="inner")
        rel = rel.merge(stage4_hmdb, on=["disease", "HMDB_ID"], how="inner")
        taxon = stage4_genera.drop(columns=["_genus_norm"]).drop_duplicates().copy()
        matched = set(rel["canonical_genus_stage4"].fillna(rel["canonical_genus"]).astype(str))
        taxon["matched_in_njc19"] = taxon["canonical_genus"].isin(matched)
        metmap = stage4_hmdb.merge(crosswalk, on="HMDB_ID", how="left")
        matched_hmdb = set(rel["HMDB_ID"])
        metmap["matched_in_njc19"] = metmap["HMDB_ID"].isin(matched_hmdb)
        group_cols = ["disease", "canonical_genus_stage4", "HMDB_ID", "source_metabolite_id"]
        edges = rel.groupby(group_cols, dropna=False).agg(
            source_microbe_names=("source_microbe_name", lambda x: ";".join(sorted(set(map(str, x))))),
            relation_type=("relation_type", lambda x: ";".join(sorted(set(map(str, x))))),
            reference=("reference", "first"),
            support_count=("source_microbe_name", "nunique"),
        ).reset_index()
        edges = edges.rename(columns={"canonical_genus_stage4": "canonical_genus", "source_metabolite_id": "source_kegg_compound"})
        edges["source"] = "NJC19"
        edges = edges[["disease", "canonical_genus", "HMDB_ID", "source", "source_microbe_names", "source_kegg_compound", "relation_type", "reference", "support_count"]]

    coverage_rows = []
    for disease, total_genera, total_met in [("IBD", 50, 166), ("CRC", 52, 109)]:
        e = edges.loc[edges["disease"].eq(disease)] if not edges.empty else edges
        matched_genera = taxon.loc[taxon["disease"].eq(disease) & taxon["matched_in_njc19"], "canonical_genus"].nunique() if "matched_in_njc19" in taxon.columns else 0
        edge_genera = e["canonical_genus"].nunique() if not e.empty else 0
        matched_met = e["HMDB_ID"].nunique() if not e.empty else 0
        edge_count = len(e)
        coverage = edge_genera / total_genera * 100
        coverage_rows.append({
            "disease": disease,
            "matched_genera": matched_genera,
            "total_stage4_genera": total_genera,
            "genera_with_eligible_stage4_metabolite_edge": edge_genera,
            "matched_metabolites": matched_met,
            "total_stage4_metabolites": total_met,
            "expected_crossomic_edge_count": edge_count,
            "coverage_percent": coverage,
            "coverage_grade": "GOOD" if coverage >= 70 else "MODERATE" if coverage >= 40 else "POOR",
        })
    status = "STAGE4D_NJC19_READY" if any(r["coverage_percent"] >= 40 for r in coverage_rows) else "STAGE4D_NJC19_POOR"
    if relations.empty:
        status = "STAGE4D_NJC19_BLOCKED"
    return taxon, metmap, edges, pd.DataFrame(coverage_rows), status


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NJC19 explicit microbe-metabolite prior edges only. No graph finalization, no GCN/GAT.")
    parser.add_argument("--source-zip", type=Path, default=NJC19_ZIP)
    parser.add_argument("--source-dir", type=Path, default=NJC19_EXTRACTED)
    args = parser.parse_args()
    relations = extract_json_relations(args.source_dir)
    if relations.empty:
        tables = find_njc19_tables(args.source_zip)
        relations = extract_candidate_relations(tables)
    taxon, metmap, edges, coverage, status = build_outputs(relations)
    atomic_csv(taxon, TABLES / "stage4d_njc19_taxon_mapping.csv")
    atomic_csv(metmap, TABLES / "stage4d_njc19_metabolite_mapping.csv")
    atomic_csv(edges, TABLES / "stage4d_njc19_crossomic_edges.csv")
    atomic_csv(coverage, TABLES / "stage4d_njc19_coverage.csv")
    write_log("stage4d_njc19_prepare", {"status": status, "source_dir": str(args.source_dir), "source_zip": str(args.source_zip), "source_obtained": bool(len(relations)), "candidate_relations": int(len(relations)), "edges": int(len(edges))})
    print(coverage.to_string(index=False))
    print(status)


if __name__ == "__main__":
    main()
