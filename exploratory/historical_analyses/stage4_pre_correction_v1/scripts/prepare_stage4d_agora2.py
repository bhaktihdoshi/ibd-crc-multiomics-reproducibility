from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from stage4_common import PROJECT, REPORTS, STAGE4, TABLES, write_log


AGORA2_BASE = "https://www.vmh.life/files/reconstructions/AGORA2/version2.01/"
AGORA2_SBML_INDEX = AGORA2_BASE + "sbml_files/individual_reconstructions/"
AGORA2_PRIOR = STAGE4 / "data/stage4d_prior_sources/agora2"
AGORA2_INDEX_HTML = AGORA2_PRIOR / "agora2_sbml_individual_reconstructions_index.html"
MICROMAP_DATAVERSE_URL = "https://dataverse.harvard.edu/dataverse/micromap"
MICROMAP_DATAVERSE_API_URL = "https://dataverse.harvard.edu/api/dataverses/micromap/contents"
FEATURE_TRACE = TABLES / "stage4c_feature_qc_trace.csv"
MIMEDB_MICROBES = STAGE4 / "data/stage4d_prior_sources/mimedb/mimedb_microbes_v2.csv"


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def model_name_from_filename(filename: str) -> str:
    return filename.removesuffix(".xml")


def genus_from_model_filename(filename: str) -> str:
    return model_name_from_filename(filename).split("_", 1)[0]


def parse_size_to_mb(size_text: str) -> float:
    text = str(size_text).strip()
    match = re.match(r"([0-9.]+)\s*([KMG])", text, flags=re.I)
    if not match:
        return float("nan")
    value = float(match.group(1))
    unit = match.group(2).upper()
    if unit == "K":
        return value / 1024
    if unit == "G":
        return value * 1024
    return value


def fetch_index() -> None:
    AGORA2_PRIOR.mkdir(parents=True, exist_ok=True)
    req = Request(AGORA2_SBML_INDEX, headers={"User-Agent": "Stage4D-AGORA2-metadata-only"})
    with urlopen(req, timeout=60) as response:
        text = response.read().decode("utf-8", errors="replace")
    atomic_text(text, AGORA2_INDEX_HTML)


def parse_agora2_index(index_html: str) -> pd.DataFrame:
    rows = []
    pattern = re.compile(r'href="([^"]+\.xml)".*?</a>\s*</td><td[^>]*>([^<]*)</td><td[^>]*>([^<]*)', re.I)
    for filename, modified, size in pattern.findall(index_html):
        filename = html.unescape(filename)
        rows.append({
            "AGORA2_species": model_name_from_filename(filename).replace("_", " "),
            "AGORA2_model_id": model_name_from_filename(filename),
            "taxonomy_id_if_available": "",
            "model_filename": filename,
            "model_url": AGORA2_SBML_INDEX + filename,
            "size_text": size.strip(),
            "size_mb": parse_size_to_mb(size),
            "genus": genus_from_model_filename(filename),
            "last_modified": modified.strip(),
        })
    return pd.DataFrame(rows)


def panmodel_genus_from_filename(filename: str) -> str:
    stem = Path(str(filename)).name.removesuffix(".xml")
    stem = re.sub(r"(?i)^AGORA2[_\-\s]*", "", stem)
    stem = re.sub(r"(?i)^pan[_\-\s]*", "", stem)
    stem = re.sub(r"(?i)[_\-\s]*(genus|panmodel|reconstruction|visuali[sz]ation).*$", "", stem)
    return stem.split("_", 1)[0].split("-", 1)[0].strip()


def parse_micromap_listing_text(text: str) -> pd.DataFrame:
    """Parse an exported Dataverse/HTML/file listing without downloading model archives.

    MicroMap pan-reconstructions are expected to be CellDesigner XML visualisations.
    The parser is intentionally conservative: it records candidate genus-level XML
    files but does not treat them as biologically sufficient until an actual file
    schema confirms reaction/metabolite identifiers and extracellular semantics.
    """
    rows = []
    seen = set()
    for match in re.finditer(r'(?P<name>[A-Za-z0-9][A-Za-z0-9_.\-\s]*?\.xml)', text):
        name = html.unescape(match.group("name")).strip()
        lower = name.lower()
        if "agora2" not in lower and "pan" not in lower and "genus" not in lower:
            continue
        if not ("pan" in lower or "genus" in lower):
            continue
        if name in seen:
            continue
        seen.add(name)
        genus = panmodel_genus_from_filename(name)
        rows.append({
            "panmodel_name": Path(name).stem,
            "panmodel_file": name,
            "canonical_genus_candidate": genus,
            "download_url_or_identifier": "",
            "estimated_size": "",
            "file_format": "CellDesigner XML candidate",
            "schema_sufficiency_status": "UNINSPECTED_FILE_SCHEMA",
        })
    return pd.DataFrame(rows)


def read_micromap_listing(listing_path: Path | None) -> pd.DataFrame:
    if listing_path is None:
        return pd.DataFrame(columns=[
            "panmodel_name",
            "panmodel_file",
            "canonical_genus_candidate",
            "download_url_or_identifier",
            "estimated_size",
            "file_format",
            "schema_sufficiency_status",
        ])
    return parse_micromap_listing_text(listing_path.read_text(errors="replace"))


def stage4_genera() -> pd.DataFrame:
    trace = pd.read_csv(FEATURE_TRACE)
    retained = trace.loc[trace["retained_in_group_aware_common_valid_schema"].fillna(False).astype(bool)].copy()
    mic = retained.loc[retained["view"].eq("microbiome"), ["disease", "feature_id"]].drop_duplicates()
    mic["canonical_genus"] = mic["feature_id"].astype(str).str.replace("mic__", "", regex=False)
    rows = []
    for genus, part in mic.groupby("canonical_genus", sort=True):
        diseases = set(part["disease"])
        rows.append({"canonical_genus": genus, "used_in_IBD": "IBD" in diseases, "used_in_CRC": "CRC" in diseases})
    return pd.DataFrame(rows)


def build_genus_panmodel_manifest(listing_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    genera = stage4_genera()
    listing = read_micromap_listing(listing_path)
    genera["_genus_norm"] = genera["canonical_genus"].map(normalize)
    if listing.empty:
        manifest = genera.copy()
        manifest["panmodel_name"] = ""
        manifest["panmodel_file"] = ""
        manifest["match_status"] = "MICROMAP_LISTING_NOT_ACCESSIBLE"
        manifest["download_url_or_identifier"] = ""
        manifest["estimated_size"] = ""
        coverage = genus_panmodel_coverage(manifest)
        return manifest.drop(columns=["_genus_norm"]), coverage, "AGORA2_GENUS_PANMODEL_NOT_ACCESSIBLE"

    listing["_genus_norm"] = listing["canonical_genus_candidate"].map(normalize)
    merged = genera.merge(listing, on="_genus_norm", how="left")
    merged["match_status"] = merged["panmodel_file"].notna().map(lambda ok: "MATCHED_MICROMAP_GENUS_PANMODEL" if ok else "NO_MICROMAP_GENUS_PANMODEL_FOR_GENUS")
    manifest = merged[[
        "canonical_genus",
        "used_in_IBD",
        "used_in_CRC",
        "panmodel_name",
        "panmodel_file",
        "match_status",
        "download_url_or_identifier",
        "estimated_size",
    ]].fillna("").sort_values(["canonical_genus", "panmodel_file"], kind="mergesort")
    coverage = genus_panmodel_coverage(manifest)
    decision = genus_panmodel_decision(manifest, coverage)
    return manifest, coverage, decision


def genus_panmodel_coverage(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, mask, total in [
        ("IBD", manifest["used_in_IBD"].astype(bool), 50),
        ("CRC", manifest["used_in_CRC"].astype(bool), 52),
        ("Union", pd.Series([True] * len(manifest)), 64),
    ]:
        part = manifest.loc[mask]
        matched = part.loc[part["match_status"].eq("MATCHED_MICROMAP_GENUS_PANMODEL"), "canonical_genus"].nunique()
        rows.append({
            "space": label,
            "stage4_genera": total,
            "genera_represented_in_MicroMap_panmodels": matched,
            "expected_taxonomic_coverage_percent": matched / total * 100,
            "required_genus_panmodel_files": int(part.loc[part["match_status"].eq("MATCHED_MICROMAP_GENUS_PANMODEL"), "panmodel_file"].astype(str).nunique()),
            "estimated_download_size": "unresolved from accessible metadata",
            "estimated_extracted_size": "unresolved from accessible metadata",
            "parser_runtime_estimate": "minutes if genus-level CellDesigner XML files are accessible; no FBA",
            "ram_estimate": "single-file streaming XML parser should stay under ~1 GB",
        })
    return pd.DataFrame(rows)


def genus_panmodel_decision(manifest: pd.DataFrame, coverage: pd.DataFrame) -> str:
    if manifest["match_status"].eq("MICROMAP_LISTING_NOT_ACCESSIBLE").all():
        return "AGORA2_GENUS_PANMODEL_NOT_ACCESSIBLE"
    union = coverage.loc[coverage["space"].eq("Union")].iloc[0]
    if float(union["expected_taxonomic_coverage_percent"]) < 85:
        return "AGORA2_GENUS_PANMODEL_INSUFFICIENT"
    if "unresolved" in str(union["estimated_download_size"]).lower():
        return "AGORA2_GENUS_PANMODEL_NOT_ACCESSIBLE"
    return "AGORA2_GENUS_PANMODEL_PREFERRED"


def write_genus_panmodel_outputs(manifest: pd.DataFrame, coverage: pd.DataFrame, decision: str) -> None:
    atomic_csv(manifest, TABLES / "stage4d_agora2_genus_panmodel_manifest.csv")
    atomic_csv(coverage, TABLES / "stage4d_agora2_genus_panmodel_coverage.csv")
    lines = [
        "# Stage 4D AGORA2 Genus Pan-Model Route",
        "",
        "No strain-scale AGORA2 acquisition, graph construction, metabolic simulation, GCN or GAT was performed.",
        "",
        "## Official Source Status",
        f"- MicroMap Dataverse: {MICROMAP_DATAVERSE_URL}",
        f"- Dataverse metadata API checked from code path: {MICROMAP_DATAVERSE_API_URL}",
        "- MicroMap paper source: Thinnes et al., npj Biofilms and Microbiomes 2025, doi:10.1038/s41522-025-00853-0.",
        "- Paper-reported resource: 422 AGORA2 genus-level pan-reconstruction visualisations generated by merging AGORA2 reconstructions belonging to the same genus.",
        "- Paper-reported format: CellDesigner .xml visualisation files.",
        "",
        "## Biological Sufficiency Rule",
        "A future genus-metabolite edge is eligible only if the genus pan-reconstruction contains an exchange reaction or explicit extracellular transport capability involving the Stage 4 metabolite. Intracellular metabolite presence alone is not eligible. If the pan-model is a union of strain reactions, an edge means at least one AGORA2 reconstruction in that genus encodes the capability.",
        "",
        "## Metabolite Mapping Rule",
        "Use AGORA2/VMH metabolite identifiers from the pan-model file, then map through audited VMH/KEGG/HMDB crosswalks to Stage 4 HMDB identifiers. Do not use fuzzy metabolite-name matching in the primary route.",
        "",
        "## Coverage",
        coverage.to_markdown(index=False),
        "",
        "## Pan-Model vs Strain-Level Recommendation",
        "The genus route is taxonomically better aligned to the Stage 4 genus nodes and would avoid the previously estimated 3,877 strain-level files (~61.9 GB). It loses strain-specific resolution, but that loss is acceptable for a genus-level prior graph if exchange/extracellular capabilities are recoverable from the genus files. Because Dataverse file metadata was not accessible in this environment and no genus XML schema was inspected here, the route is not yet frozen as preferred.",
        "",
        "## Exact User Command",
        "After manually exporting or saving official MicroMap Dataverse file metadata/listing, run:",
        "",
        "```bash",
        "python thesis_preprocessing_workflow/stage4_multiomics_integration/scripts/prepare_stage4d_agora2.py --genus-panmodel --micromap-listing /path/to/micromap_dataverse_file_listing.html",
        "```",
        "",
        f"## Final Status: {decision}",
    ]
    atomic_text("\n".join(lines) + "\n", REPORTS / "STAGE4D_AGORA2_GENUS_PANMODEL_ROUTE.md")


def mimedb_taxonomy_ids() -> pd.DataFrame:
    if not MIMEDB_MICROBES.exists():
        return pd.DataFrame(columns=["canonical_genus", "AGORA2_species", "taxonomy_id_if_available"])
    df = pd.read_csv(MIMEDB_MICROBES, dtype=str, encoding="utf-8-sig")
    if "genus" not in df.columns:
        return pd.DataFrame(columns=["canonical_genus", "AGORA2_species", "taxonomy_id_if_available"])
    out = pd.DataFrame({
        "canonical_genus": df["genus"].astype(str),
        "AGORA2_species": df.get("species", pd.Series([""] * len(df))).astype(str),
        "taxonomy_id_if_available": df.get("ncbi_tax_id", pd.Series([""] * len(df))).astype(str),
    }).drop_duplicates()
    return out


def build_taxon_manifest(fetch: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fetch:
        fetch_index()
    genera = stage4_genera()
    tax = mimedb_taxonomy_ids()
    if not AGORA2_INDEX_HTML.exists():
        manifest = genera.copy()
        for col in ["AGORA2_species", "AGORA2_model_id", "taxonomy_id_if_available", "model_filename"]:
            manifest[col] = ""
        manifest["match_status"] = "AGORA2_INDEX_NOT_FETCHED"
        coverage = coverage_table(manifest)
        return manifest[["canonical_genus", "used_in_IBD", "used_in_CRC", "AGORA2_species", "AGORA2_model_id", "taxonomy_id_if_available", "model_filename", "match_status"]], coverage
    index = parse_agora2_index(AGORA2_INDEX_HTML.read_text())
    index["_genus_norm"] = index["genus"].map(normalize)
    genera["_genus_norm"] = genera["canonical_genus"].map(normalize)
    merged = genera.merge(index, on="_genus_norm", how="left")
    merged["match_status"] = merged["model_filename"].notna().map(lambda ok: "MATCHED_AGORA2_MODEL" if ok else "NO_AGORA2_MODEL_FOR_GENUS")
    if not tax.empty:
        merged = merged.merge(tax, on=["canonical_genus", "AGORA2_species"], how="left", suffixes=("", "_mimedb"))
        merged["taxonomy_id_if_available"] = merged["taxonomy_id_if_available"].where(merged["taxonomy_id_if_available"].astype(str).ne(""), merged.get("taxonomy_id_if_available_mimedb", ""))
    manifest = merged[["canonical_genus", "used_in_IBD", "used_in_CRC", "AGORA2_species", "AGORA2_model_id", "taxonomy_id_if_available", "model_filename", "match_status"]].sort_values(["canonical_genus", "AGORA2_model_id"], kind="mergesort")
    return manifest, coverage_table(manifest, index)


def coverage_table(manifest: pd.DataFrame, index: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for label, mask, total in [
        ("IBD", manifest["used_in_IBD"].astype(bool), 50),
        ("CRC", manifest["used_in_CRC"].astype(bool), 52),
        ("Union", pd.Series([True] * len(manifest)), 64),
    ]:
        part = manifest.loc[mask]
        represented = part.loc[part["match_status"].eq("MATCHED_AGORA2_MODEL"), "canonical_genus"].nunique()
        models = int(part["model_filename"].fillna("").astype(str).ne("").sum())
        size = float("nan")
        if index is not None and "size_mb" in index.columns:
            files = set(part["model_filename"].dropna().astype(str))
            size = float(index.loc[index["model_filename"].isin(files), "size_mb"].sum())
        rows.append({
            "space": label,
            "stage4_genera": total,
            "genera_represented_in_AGORA2": represented,
            "expected_taxonomic_coverage_percent": represented / total * 100,
            "matched_model_files": models,
            "estimated_download_size_mb": size,
            "estimated_extracted_size_mb": size,
            "parser_runtime_estimate": "minutes for targeted SBML files; no FBA",
            "ram_estimate": "single-file streaming XML parser should stay under ~1 GB",
        })
    return pd.DataFrame(rows)


def write_placeholder_outputs(manifest: pd.DataFrame, coverage: pd.DataFrame) -> None:
    atomic_csv(manifest, TABLES / "stage4d_agora2_taxon_manifest.csv")
    download = manifest.loc[manifest["model_filename"].fillna("").astype(str).ne(""), ["AGORA2_model_id", "model_filename"]].drop_duplicates().copy()
    if not download.empty:
        download["model_url"] = AGORA2_SBML_INDEX + download["model_filename"].astype(str)
        download["local_model_file"] = download["model_filename"].map(lambda x: str(AGORA2_PRIOR / "sbml_targeted" / str(x)))
        download["download_command"] = download.apply(lambda r: f'curl -L -o "{r["local_model_file"]}" "{r["model_url"]}"', axis=1)
    else:
        download = pd.DataFrame(columns=["AGORA2_model_id", "model_filename", "model_url", "local_model_file", "download_command"])
    atomic_csv(download, TABLES / "stage4d_agora2_targeted_download_manifest.csv")
    script_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'mkdir -p "{AGORA2_PRIOR / "sbml_targeted"}"',
        "# Metadata-derived targeted AGORA2 SBML acquisition. No model fitting, no FBA, no GCN/GAT.",
    ]
    script_lines.extend(download["download_command"].astype(str).tolist())
    atomic_text("\n".join(script_lines) + "\n", AGORA2_PRIOR / "download_stage4d_targeted_agora2_sbml.sh")
    atomic_csv(pd.DataFrame(columns=["AGORA2_metabolite_id", "VMH_id", "KEGG_id", "HMDB_ID", "mapping_route", "mapping_status"]), TABLES / "stage4d_agora2_metabolite_mapping.csv")
    atomic_csv(pd.DataFrame(columns=["canonical_genus", "HMDB_ID", "edge_type", "weight", "supporting_species_count", "supporting_model_count", "AGORA2_model_ids", "metabolite_identifier", "mapping_route"]), TABLES / "stage4d_agora2_crossomic_edges.csv")
    atomic_csv(coverage, TABLES / "stage4d_agora2_coverage.csv")


def write_report(manifest: pd.DataFrame, coverage: pd.DataFrame) -> None:
    matched = int(manifest["model_filename"].fillna("").astype(str).ne("").sum())
    lines = [
        "# Stage 4D AGORA2 Methods Rationale",
        "",
        "AGORA2/VMH is prepared only as an external prior source for future MICROBE_METABOLITE edges. No graph construction, flux optimisation, FBA, GCN/GAT or predictive modelling is performed here.",
        "",
        "## Official Sources",
        f"- AGORA2 version 2.01: {AGORA2_BASE}",
        f"- Individual SBML reconstructions: {AGORA2_SBML_INDEX}",
        "- Reference: Heinken et al., Nature Biotechnology 2023, doi:10.1038/s41587-022-01628-0.",
        "",
        "## Edge Biology Rule For Future Execution",
        "An eligible edge means an AGORA2 reconstruction contains an exchange or extracellular transport capability for a Stage 4 metabolite. Intracellular metabolites alone are not eligible. No FBA is required for presence/absence extraction.",
        "",
        "## Parser Plan",
        "For each targeted SBML file, parse reaction IDs and species/metabolite identifiers. Retain exchange reactions and extracellular transport reactions only, then map VMH/KEGG/HMDB identifiers to Stage 4 HMDB IDs using audited identifier crosswalks. Collapse species/strain support to genus-metabolite candidate edges with binary primary weight 1.0.",
        "",
        "## Current Coverage",
        coverage.to_markdown(index=False),
        "",
        f"Matched model files currently listed: {matched}. If zero, fetch the AGORA2 directory index metadata before interpreting coverage.",
    ]
    atomic_text("\n".join(lines) + "\n", REPORTS / "STAGE4D_AGORA2_METHODS_RATIONALE.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stage 4D AGORA2/VMH targeted acquisition metadata only. No model download or graph construction.")
    parser.add_argument("--fetch-index", action="store_true", help="Fetch only the official AGORA2 individual SBML directory HTML index. Does not download model files.")
    parser.add_argument("--genus-panmodel", action="store_true", help="Audit MicroMap AGORA2 genus pan-model route. Does not download pan-model files.")
    parser.add_argument("--micromap-listing", type=Path, default=None, help="Optional official MicroMap/Dataverse file-listing export for genus pan-model matching.")
    args = parser.parse_args()
    if args.genus_panmodel:
        manifest, coverage, decision = build_genus_panmodel_manifest(args.micromap_listing)
        write_genus_panmodel_outputs(manifest, coverage, decision)
        write_log("stage4d_agora2_genus_panmodel", {"status": decision, "manifest_rows": int(len(manifest)), "listing_provided": args.micromap_listing is not None})
        print(coverage.to_string(index=False))
        print(decision)
        return
    manifest, coverage = build_taxon_manifest(fetch=args.fetch_index)
    write_placeholder_outputs(manifest, coverage)
    write_report(manifest, coverage)
    write_log("stage4d_agora2_prepare", {"status": "prepared", "fetch_index": args.fetch_index, "manifest_rows": int(len(manifest))})
    print(coverage.to_string(index=False))
    print("STAGE4D_AGORA2_PREPARATION_COMPLETE")


if __name__ == "__main__":
    main()
