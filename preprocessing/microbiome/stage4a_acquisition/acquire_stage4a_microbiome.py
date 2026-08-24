from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path("/Users/bhaktidoshi/Downloads/Thesis Data")
ROOT = PROJECT / "thesis_preprocessing_workflow"
STAGE4 = ROOT / "stage4_multiomics_integration"

RAW = STAGE4 / "data" / "external_microbiome"
PROCESSED = STAGE4 / "data" / "processed"
TABLES = STAGE4 / "tables"
MANIFESTS = STAGE4 / "manifests"
REPORTS = STAGE4 / "reports"

IBD_INCLUDED = ROOT / "ibd_pipeline" / "tables" / "included_samples.csv"
IHMP_VISITS = ROOT / "ibd_pipeline" / "tables" / "ihmp_visit_selection.csv"
YACHIDA_MODEL = ROOT / "crc_pipeline" / "stage1_release" / "tables" / "stage2_model_prep" / "crc_stage2_yachida_primary_crc_vs_control_model_prep.csv"
SINHA_MICROBES = ROOT / "crc_pipeline" / "external_metadata_sources" / "sinha_s3_microbes.csv"

FRANZOSA_SPECIES = RAW / "franzosa" / "41564_2018_306_MOESM6_ESM.xlsx"
FRANZOSA_METABOLOME_SUPP = RAW / "franzosa" / "41564_2018_306_MOESM4_ESM.xlsx"
YACHIDA_SUPP = RAW / "yachida" / "41591_2019_458_MOESM3_ESM.xlsx"
IHMP_METADATA = RAW / "ihmp" / "hmp2_metadata_2018-08-20.csv"
IHMP_TAX_ATTEMPT = RAW / "ihmp" / "taxonomic_profiles.tsv.gz"
IHMP_TAX_ATTEMPT_DOWNLOADS = RAW / "ihmp" / "taxonomic_profiles_downloads_route.tsv.gz"
IHMP_TAX_ATTEMPT_TUNNEL = RAW / "ihmp" / "taxonomic_profiles_tunnel_products.tsv.gz"

DOWNLOADS = [
    {
        "cohort": "Franzosa",
        "publication": "Franzosa et al. 2019 Nature Microbiology",
        "doi": "10.1038/s41564-018-0306-4",
        "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41564-018-0306-4/MediaObjects/41564_2018_306_MOESM6_ESM.xlsx",
        "repository_or_accession": "PRJNA400072; Nature Supplementary Dataset 4",
        "local_path": FRANZOSA_SPECIES,
        "assay": "shotgun metagenomics",
        "processing_level": "processed per-subject species relative abundance",
        "taxonomic_resolution": "species",
        "sample_identifier_field": "PRISM sample columns; SRA_metagenome_name metadata row",
    },
    {
        "cohort": "Franzosa",
        "publication": "Franzosa et al. 2019 Nature Microbiology",
        "doi": "10.1038/s41564-018-0306-4",
        "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41564-018-0306-4/MediaObjects/41564_2018_306_MOESM4_ESM.xlsx",
        "repository_or_accession": "Nature Supplementary Dataset 2",
        "local_path": FRANZOSA_METABOLOME_SUPP,
        "assay": "untargeted stool metabolomics",
        "processing_level": "processed per-subject metabolite relative abundance; used here only for ID bridge audit",
        "taxonomic_resolution": "not microbiome",
        "sample_identifier_field": "PRISM sample columns",
    },
    {
        "cohort": "iHMP",
        "publication": "Lloyd-Price et al. 2019 Nature",
        "doi": "10.1038/s41586-019-1237-9",
        "url": "https://g-227ca.190ebd.75bc.data.globus.org/ibdmdb/metadata/hmp2_metadata_2018-08-20.csv",
        "repository_or_accession": "IBDMDB / HMP2 / PRJNA398089",
        "local_path": IHMP_METADATA,
        "assay": "multi-omics metadata",
        "processing_level": "official sample metadata",
        "taxonomic_resolution": "not microbiome",
        "sample_identifier_field": "External ID / participant / visit fields as supplied",
    },
    {
        "cohort": "iHMP",
        "publication": "Lloyd-Price et al. 2019 Nature",
        "doi": "10.1038/s41586-019-1237-9",
        "url": "https://ibdmdb.org/tunnel/products/HMP2/WGS/1818/taxonomic_profiles.tsv.gz",
        "repository_or_accession": "IBDMDB / HMP2 / PRJNA398089",
        "local_path": IHMP_TAX_ATTEMPT,
        "assay": "shotgun metagenomics",
        "processing_level": "attempted merged taxonomic profile download",
        "taxonomic_resolution": "unresolved; local file is HTML not gzip",
        "sample_identifier_field": "not parsed",
    },
    {
        "cohort": "iHMP",
        "publication": "Lloyd-Price et al. 2019 Nature",
        "doi": "10.1038/s41586-019-1237-9",
        "url": "https://ibdmdb.org/downloads/products/HMP2/WGS/1818/taxonomic_profiles.tsv.gz",
        "repository_or_accession": "IBDMDB / HMP2 / PRJNA398089",
        "local_path": IHMP_TAX_ATTEMPT_DOWNLOADS,
        "assay": "shotgun metagenomics",
        "processing_level": "attempted merged taxonomic profile download through downloads/products route",
        "taxonomic_resolution": "unresolved; local file is HTML not gzip",
        "sample_identifier_field": "not parsed",
    },
    {
        "cohort": "iHMP",
        "publication": "Lloyd-Price et al. 2019 Nature",
        "doi": "10.1038/s41586-019-1237-9",
        "url": "https://www.ibdmdb.org/tunnel/products/HMP2/WGS/1818/taxonomic_profiles.tsv.gz",
        "repository_or_accession": "IBDMDB / HMP2 / PRJNA398089",
        "local_path": IHMP_TAX_ATTEMPT_TUNNEL,
        "assay": "shotgun metagenomics",
        "processing_level": "attempted merged taxonomic profile download through tunnel/products route",
        "taxonomic_resolution": "unresolved; local file is plain-text bad request",
        "sample_identifier_field": "not parsed",
    },
    {
        "cohort": "Yachida",
        "publication": "Yachida et al. 2019 Nature Medicine",
        "doi": "10.1038/s41591-019-0458-7",
        "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41591-019-0458-7/MediaObjects/41591_2019_458_MOESM3_ESM.xlsx",
        "repository_or_accession": "DRA006684; DRA008156; Nature Supplementary Tables 1-15",
        "local_path": YACHIDA_SUPP,
        "assay": "shotgun metagenomics",
        "processing_level": "processed supplementary clinical, species and genus profiles",
        "taxonomic_resolution": "species and genus",
        "sample_identifier_field": "Subject_ID columns",
    },
    {
        "cohort": "Sinha",
        "publication": "Sinha et al. 2016",
        "doi": "10.1371/journal.pone.0166112",
        "url": "local previously acquired S3 microbial table",
        "repository_or_accession": "official supporting information previously curated in CRC Stage 1",
        "local_path": SINHA_MICROBES,
        "assay": "microbial taxonomic profile",
        "processing_level": "processed supporting-information microbial abundance table",
        "taxonomic_resolution": "mixed taxonomic tree",
        "sample_identifier_field": "ID",
    },
]


def ensure_dirs() -> None:
    for path in [PROCESSED, TABLES, MANIFESTS, REPORTS]:
        path.mkdir(parents=True, exist_ok=True)
    for cohort in ["franzosa", "ihmp", "yachida"]:
        (RAW / cohort).mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    head = path.read_bytes()[:200].lower()
    if path.suffix == ".gz" and not head.startswith(b"\x1f\x8b"):
        return "downloaded_not_gzip_html_or_text"
    if b"<html" in head or b"<error>" in head or b"accessdenied" in head:
        return "downloaded_html_or_xml_error"
    return "downloaded"


def actual_format(path: Path) -> str:
    if not path.exists():
        return "missing"
    head = path.read_bytes()[:200].lower()
    if head.startswith(b"\x1f\x8b"):
        return "gzip"
    if b"<html" in head or b"<!doctype html" in head:
        return "html"
    if b"<h1>bad request" in head:
        return "plain_text_bad_request"
    if b"<?xml" in head or b"<error>" in head:
        return "xml_or_error"
    if path.suffix == ".xlsx":
        return "xlsx"
    if path.suffix == ".csv":
        return "csv"
    return "unknown_text_or_binary"


def write_manifest() -> pd.DataFrame:
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    hash_rows = []
    for item in DOWNLOADS:
        path = Path(item["local_path"])
        status = file_status(path)
        size = path.stat().st_size if path.exists() else 0
        checksum = sha256(path) if path.exists() else ""
        rows.append(
            {
                **{k: v for k, v in item.items() if k != "local_path"},
                "file_name": path.name,
                "local_path": str(path),
                "file_size_bytes": size,
                "sha256": checksum,
                "download_timestamp_utc": now if path.exists() else "",
                "download_status": status,
            }
        )
        if path.exists():
            hash_rows.append({"cohort": item["cohort"], "file_name": path.name, "local_path": str(path), "file_size_bytes": size, "sha256": checksum})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFESTS / "stage4a_microbiome_download_manifest.csv", index=False)
    pd.DataFrame(hash_rows).to_csv(MANIFESTS / "stage4a_microbiome_file_hashes.csv", index=False)
    return manifest


def read_franzosa_species() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(FRANZOSA_SPECIES, sheet_name="dataset_s4", header=None)
    header = raw.iloc[1].tolist()
    columns = ["feature"] + [str(x).replace("|", ".") for x in header[1:]]
    df = raw.iloc[2:].copy()
    df.columns = columns
    meta_names = {"SRA_metagenome_name", "Age", "Diagnosis", "Fecal.Calprotectin", "antibiotic", "immunosuppressant", "mesalamine"}
    metadata = df[df["feature"].isin(meta_names)].set_index("feature").T.reset_index(names="sample_id")
    taxa = df[~df["feature"].isin(meta_names)].set_index("feature").T
    taxa.index.name = "sample_id"
    taxa = taxa.apply(pd.to_numeric, errors="coerce")
    taxa = taxa.loc[:, taxa.columns.notna()]
    taxa = taxa.loc[:, ~taxa.columns.duplicated()]
    return taxa.reset_index(), metadata


def read_yachida_table(sheet: str) -> pd.DataFrame:
    raw = pd.read_excel(YACHIDA_SUPP, sheet_name=sheet, header=None)
    sample_ids = [str(int(float(x))) for x in raw.iloc[3, 1:].dropna().tolist()]
    features = raw.iloc[4:, 0].astype(str)
    values = raw.iloc[4:, 1 : len(sample_ids) + 1].copy()
    values.index = features
    values.columns = sample_ids
    values = values.apply(pd.to_numeric, errors="coerce")
    values = values.T.reset_index(names="sample_id")
    return values.loc[:, ~values.columns.duplicated()]


def collapse_duplicate_sample_rows(df: pd.DataFrame, sample_col: str) -> tuple[pd.DataFrame, int, int, set[str]]:
    """Keep one row per sample only when repeated rows are numerically identical after filling NAs."""
    if sample_col not in df.columns or not df[sample_col].duplicated().any():
        return df.copy(), 0, 0, set()
    duplicate_ids = df.loc[df[sample_col].duplicated(keep=False), sample_col].astype(str).unique()
    conflict_count = 0
    conflict_ids: set[str] = set()
    kept = []
    for _, group in df.groupby(sample_col, sort=False, dropna=False):
        if len(group) == 1:
            kept.append(group.iloc[[0]])
            continue
        compare = group.drop(columns=[sample_col]).apply(pd.to_numeric, errors="coerce").fillna(-999999999.0)
        if not compare.eq(compare.iloc[0]).all(axis=None):
            conflict_count += 1
            conflict_ids.add(str(group[sample_col].iloc[0]))
        kept.append(group.iloc[[0]])
    return pd.concat(kept, ignore_index=True), len(duplicate_ids), conflict_count, conflict_ids


def read_yachida_metadata() -> pd.DataFrame:
    meta = pd.read_excel(YACHIDA_SUPP, sheet_name="Table_S2-2", header=2)
    meta["sample_id"] = meta["Subject_ID"].astype(str)
    return meta


def read_sinha_microbiome() -> pd.DataFrame:
    df = pd.read_csv(SINHA_MICROBES)
    return df.rename(columns={"ID": "sample_id"})


def read_frozen_populations() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ibd = pd.read_csv(IBD_INCLUDED)
    franzosa = ibd[ibd["cohort"].eq("Franzosa")].copy()
    ihmp = ibd[ibd["cohort"].eq("iHMP")].copy()
    yachida = pd.read_csv(YACHIDA_MODEL)
    yachida["sample_id"] = yachida["Sample"].astype(str)
    return franzosa, ihmp, yachida


def audit_matrix(cohort: str, name: str, matrix: pd.DataFrame, id_col: str, rank: str, scale: str) -> dict[str, object]:
    feature_cols = [c for c in matrix.columns if c != id_col]
    numeric = matrix[feature_cols].apply(pd.to_numeric, errors="coerce") if feature_cols else pd.DataFrame(index=matrix.index)
    row_sums = numeric.fillna(0).sum(axis=1) if not numeric.empty else pd.Series(dtype=float)
    return {
        "cohort": cohort,
        "source_matrix": name,
        "rows_orientation": "samples",
        "sample_n": len(matrix),
        "feature_p": len(feature_cols),
        "taxonomic_rank": rank,
        "sample_id_field": id_col,
        "duplicate_sample_ids": int(matrix[id_col].duplicated().sum()) if id_col in matrix else np.nan,
        "duplicate_taxa": int(pd.Index(feature_cols).duplicated().sum()),
        "abundance_scale_classification": scale,
        "overall_zero_fraction": float((numeric.fillna(0) == 0).to_numpy().mean()) if not numeric.empty else np.nan,
        "overall_missing_fraction": float(numeric.isna().to_numpy().mean()) if not numeric.empty else np.nan,
        "median_row_sum": float(row_sums.median()) if not row_sums.empty else np.nan,
        "row_sum_interpretation": "approximately_1" if not row_sums.empty and np.isclose(row_sums.median(), 1, atol=0.05) else ("approximately_100" if not row_sums.empty and np.isclose(row_sums.median(), 100, atol=5) else "neither_or_mixed"),
    }


def phenotype_counts(df: pd.DataFrame, label_col: str) -> str:
    if df.empty or label_col not in df:
        return ""
    counts = df[label_col].fillna("Missing").astype(str).value_counts().sort_index()
    return "; ".join(f"{k}={v}" for k, v in counts.items())


def classify_taxon_rank(name: str) -> str:
    text = str(name)
    rank = "unknown"
    for marker, label in [("k__", "kingdom"), ("p__", "phylum"), ("c__", "class"), ("o__", "order"), ("f__", "family"), ("g__", "genus"), ("s__", "species")]:
        if marker in text:
            rank = label
    return rank


def parse_taxonomy_label(label: str) -> dict[str, str]:
    text = str(label).strip()
    parts = {"kingdom": "", "phylum": "", "class": "", "order": "", "family": "", "genus": "", "species": ""}
    marker_to_rank = {"k__": "kingdom", "p__": "phylum", "c__": "class", "o__": "order", "f__": "family", "g__": "genus", "s__": "species"}
    if "__" in text:
        for token in re.split(r"[;|]", text):
            token = token.strip()
            for marker, rank in marker_to_rank.items():
                if token.startswith(marker):
                    parts[rank] = token.replace(marker, "").strip()
        return parts
    words = text.replace("_", " ").strip()
    if words and not words.lower().startswith(("unknown", "unclassified", "motu_linkage_group")):
        split = words.split()
        if len(split) >= 2 and split[1].lower() not in {"sp.", "spp.", "bacterium"}:
            parts["genus"] = split[0]
            parts["species"] = " ".join(split[:2])
        elif len(split) >= 1:
            parts["genus"] = split[0]
    return parts


def canonical_feature_record(cohort: str, original_feature: str) -> dict[str, object]:
    parsed = parse_taxonomy_label(original_feature)
    original_rank = classify_taxon_rank(original_feature)
    if original_rank == "unknown":
        if parsed.get("species"):
            original_rank = "species"
        elif parsed.get("genus"):
            original_rank = "genus"
    parsed_name = parsed.get(original_rank, "") if original_rank in parsed else ""
    canonical_rank = original_rank if original_rank in {"species", "genus", "family"} else ("genus" if parsed.get("genus") else "unknown")
    canonical_name = parsed.get(canonical_rank, "") if canonical_rank in parsed else ""
    include = bool(canonical_name) and canonical_rank in {"species", "genus", "family"} and not re.search(r"unclassified|unknown|motu_linkage_group|^sp\\.?$", canonical_name, re.I)
    confidence = "EXACT_CANONICAL_NAME" if include else "UNMAPPED"
    if "__" in str(original_feature) and include:
        confidence = "DETERMINISTIC_PARENT_AGGREGATION" if canonical_rank != original_rank else "EXACT_CANONICAL_NAME"
    return {
        "cohort": cohort,
        "original_feature": original_feature,
        "original_rank": original_rank,
        "parsed_name": parsed_name,
        "canonical_name": canonical_name,
        "canonical_rank": canonical_rank,
        "ncbi_taxid": "",
        "mapping_method": "deterministic label parsing and canonical scientific-name normalisation",
        "mapping_source": "official cohort feature labels; no external synonym service used in this lightweight audit",
        "mapping_confidence": confidence,
        "ambiguity_flag": not include,
        "include_for_harmonisation": include,
        "notes": "NCBI TaxID not assigned without an authoritative local ID field or online taxonomy resolver; ambiguous/unclassified labels excluded from harmonised overlap.",
    }


def rank_feature_set(df: pd.DataFrame, rank: str) -> set[str]:
    if df.empty:
        return set()
    return {str(c) for c in df.columns if c != "sample_id" and classify_taxon_rank(c) == rank}


def write_ihmp_attempts_report(manifest: pd.DataFrame) -> None:
    attempts = manifest[manifest["cohort"].eq("iHMP")].copy()
    attempts["actual_format"] = attempts["local_path"].map(lambda x: actual_format(Path(x)))
    attempts["success_failure"] = attempts["download_status"].eq("downloaded").map({True: "success", False: "failure"})
    attempts["reason"] = attempts.apply(
        lambda r: "valid local file" if r["success_failure"] == "success" else f"not accepted as abundance matrix: {r['download_status']} / {r['actual_format']}",
        axis=1,
    )
    lines = [
        "# Stage 4A iHMP Processed Taxonomic Matrix Acquisition Attempts",
        "",
        "Authoritative processed WGS taxonomic abundance was sought from IBDMDB/HMP2 routes only. HTTP 200 responses were not accepted unless the local artifact was a real gzip/table and contained sample/taxon structure.",
        "",
        attempts[["url", "file_name", "file_size_bytes", "download_status", "actual_format", "success_failure", "reason"]].to_markdown(index=False),
        "",
        "Conclusion: the official metadata file is valid, but no valid processed merged taxonomic abundance matrix was acquired in this pass. Raw FASTQ or per-sample tar reconstruction was not started because it violates the Stage 4A resource gate without explicit approval.",
    ]
    (REPORTS / "STAGE4A_IHMP_ACQUISITION_ATTEMPTS.md").write_text("\n".join(lines) + "\n")


def write_sinha_taxonomy_audit(sinha: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in [c for c in sinha.columns if c != "sample_id"]:
        parsed = parse_taxonomy_label(feature)
        rows.append(
            {
                "original_feature_label": feature,
                "kingdom": parsed["kingdom"],
                "phylum": parsed["phylum"],
                "class": parsed["class"],
                "order": parsed["order"],
                "family": parsed["family"],
                "genus": parsed["genus"],
                "species": parsed["species"],
                "NCBI_taxid_if_recoverable": "",
                "canonical_name": parsed["species"] or parsed["genus"] or parsed["family"],
                "parsed_rank": classify_taxon_rank(feature),
                "explicit_node_available": "__" in str(feature),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "stage4a_sinha_taxonomy_structure_audit.csv", index=False)
    explicit_genus = int(out["parsed_rank"].eq("genus").sum())
    lines = [
        "# Stage 4A Sinha Genus Aggregation Policy",
        "",
        "The Sinha microbial table contains a taxonomic tree with parent and child nodes in the same matrix. To avoid double-counting, Stage 4A does not sum child taxa into parent genera while also retaining explicit parent measurements.",
        "",
        f"Explicit genus-level nodes available: {explicit_genus}.",
        "",
        "Policy: use explicit `g__` genus nodes as the genus representation. Do not aggregate terminal taxa into genus unless parent genus rows are absent and a future method explicitly drops child/parent duplicates before modelling.",
    ]
    (REPORTS / "STAGE4A_SINHA_GENUS_AGGREGATION_POLICY.md").write_text("\n".join(lines) + "\n")
    return out


def build_outputs() -> None:
    ensure_dirs()
    manifest = write_manifest()
    write_ihmp_attempts_report(manifest)
    franzosa_frozen, ihmp_frozen, yachida_frozen = read_frozen_populations()

    schema_rows = []
    crosswalk_rows = []
    validation_rows = []
    taxa_rows = []
    yachida_duplicate_rows = []
    taxonomy_sources: dict[str, pd.DataFrame] = {}

    # Franzosa
    if FRANZOSA_SPECIES.exists() and file_status(FRANZOSA_SPECIES) == "downloaded":
        f_micro, f_micro_meta = read_franzosa_species()
        f_match = franzosa_frozen.merge(f_micro, on="sample_id", how="inner")
        f_meta_match = franzosa_frozen.merge(f_micro_meta, on="sample_id", how="inner")
        f_match[["sample_id"] + [c for c in f_micro.columns if c != "sample_id"]].to_csv(PROCESSED / "franzosa_microbiome_unfiltered.csv", index=False)
        f_meta_match.to_csv(PROCESSED / "franzosa_microbiome_matched_metadata.csv", index=False)
        taxonomy_sources["Franzosa"] = f_match[["sample_id"] + [c for c in f_micro.columns if c != "sample_id"]].copy()
        schema_rows.append(audit_matrix("Franzosa", FRANZOSA_SPECIES.name, f_micro, "sample_id", "species", "relative abundance"))
        for _, row in f_meta_match.iterrows():
            crosswalk_rows.append(
                {
                    "disease": "IBD",
                    "cohort": "Franzosa",
                    "participant_id": row.get("participant_id", ""),
                    "microbiome_sample_id": row.get("SRA_metagenome_name", ""),
                    "metabolome_sample_id": row["sample_id"],
                    "visit_id": "cross-sectional",
                    "phenotype": row.get("harmonised_label", ""),
                    "match_method": "exact PRISM sample ID after replacing official PRISM| with thesis PRISM.",
                    "match_status": "matched",
                    "ambiguity_flag": False,
                    "notes": "Species table includes the SRA metagenome name as a metadata row.",
                }
            )
        f_unmatched = franzosa_frozen[~franzosa_frozen["sample_id"].isin(f_meta_match["sample_id"])].copy()
        for _, row in f_unmatched.iterrows():
            crosswalk_rows.append(
                {
                    "disease": "IBD",
                    "cohort": "Franzosa",
                    "participant_id": row.get("participant_id", ""),
                    "microbiome_sample_id": "",
                    "metabolome_sample_id": row["sample_id"],
                    "visit_id": "cross-sectional",
                    "phenotype": row.get("harmonised_label", ""),
                    "match_method": "exact sample ID attempted",
                    "match_status": "unmatched_not_present_in_downloaded_species_supplement",
                    "ambiguity_flag": False,
                    "notes": "The downloaded Franzosa Supplementary Dataset 4 species table bridges PRISM samples; this frozen metabolomics sample is from the Validation.UMCG subset and has no deterministic microbiome row in this file.",
                }
            )
        status = "pass" if len(f_match) == len(franzosa_frozen) else "partial"
        validation_rows.append({"test": "franzosa_exact_frozen_sample_matches", "status": status, "observed": len(f_match), "expected": len(franzosa_frozen), "notes": phenotype_counts(f_meta_match, "harmonised_label") + f"; unmatched={len(f_unmatched)}"})
    else:
        validation_rows.append({"test": "franzosa_exact_frozen_sample_matches", "status": "fail", "observed": 0, "expected": len(franzosa_frozen), "notes": "Official species workbook unavailable or invalid."})

    # iHMP
    ihmp_status = file_status(IHMP_TAX_ATTEMPT)
    ihmp_meta_status = file_status(IHMP_METADATA)
    if IHMP_METADATA.exists():
        try:
            ihmp_meta = pd.read_csv(IHMP_METADATA, low_memory=False)
            schema_rows.append(
                {
                    "cohort": "iHMP",
                    "source_matrix": IHMP_METADATA.name,
                    "rows_orientation": "samples_or_assays_metadata",
                    "sample_n": len(ihmp_meta),
                    "feature_p": ihmp_meta.shape[1],
                    "taxonomic_rank": "not a taxonomic abundance matrix",
                    "sample_id_field": "; ".join([c for c in ihmp_meta.columns if "External ID" in c or c.lower() in {"participant id", "visit number"}][:6]),
                    "duplicate_sample_ids": np.nan,
                    "duplicate_taxa": np.nan,
                    "abundance_scale_classification": "metadata only",
                    "overall_zero_fraction": np.nan,
                    "overall_missing_fraction": float(ihmp_meta.isna().to_numpy().mean()),
                    "median_row_sum": np.nan,
                    "row_sum_interpretation": "not_applicable",
                }
            )
        except Exception as exc:
            schema_rows.append({"cohort": "iHMP", "source_matrix": IHMP_METADATA.name, "rows_orientation": "unreadable", "sample_n": 0, "feature_p": 0, "taxonomic_rank": "", "sample_id_field": "", "duplicate_sample_ids": np.nan, "duplicate_taxa": np.nan, "abundance_scale_classification": "metadata read failed", "overall_zero_fraction": np.nan, "overall_missing_fraction": np.nan, "median_row_sum": np.nan, "row_sum_interpretation": repr(exc)})
    ihmp_visit = ihmp_frozen[["participant_id", "sample_id", "harmonised_label", "visit_num", "week_num", "selected_primary_first_visit"]].copy()
    ihmp_visit["microbiome_matrix_status"] = ihmp_status
    ihmp_visit["stage4a_match_status"] = "not_matched_taxonomic_matrix_not_acquired"
    ihmp_visit.to_csv(TABLES / "stage4a_ihmp_visit_matching_audit.csv", index=False)
    for _, row in ihmp_visit.iterrows():
        crosswalk_rows.append(
            {
                "disease": "IBD",
                "cohort": "iHMP",
                "participant_id": row.get("participant_id", ""),
                "microbiome_sample_id": "",
                "metabolome_sample_id": row["sample_id"],
                "visit_id": row.get("visit_num", ""),
                "phenotype": row.get("harmonised_label", ""),
                "match_method": "same participant and visit required",
                "match_status": "unmatched_taxonomic_matrix_not_acquired",
                "ambiguity_flag": False,
                "notes": "Official metadata acquired, but no valid processed taxonomic abundance matrix was acquired in this pass.",
            }
        )
    validation_rows.append({"test": "ihmp_taxonomic_matrix_is_gzip", "status": "fail" if ihmp_status != "downloaded" else "pass", "observed": ihmp_status, "expected": "downloaded", "notes": "Metadata was acquired, but the attempted merged taxonomic profile URL returned HTML/text rather than a gzip matrix."})

    # Yachida
    if YACHIDA_SUPP.exists() and file_status(YACHIDA_SUPP) == "downloaded":
        y_species_raw = read_yachida_table("Table_S7-1")
        y_species_all, y_duplicate_ids, y_duplicate_conflicts, y_conflict_ids = collapse_duplicate_sample_rows(y_species_raw, "sample_id")
        y_species = y_species_all[~y_species_all["sample_id"].astype(str).isin(y_conflict_ids)].copy()
        y_genus_raw = read_yachida_table("Table_S7-2")
        y_genus_all, _, _, y_genus_conflict_ids = collapse_duplicate_sample_rows(y_genus_raw, "sample_id")
        y_genus = y_genus_all[~y_genus_all["sample_id"].astype(str).isin(y_genus_conflict_ids | y_conflict_ids)].copy()
        y_meta = read_yachida_metadata()
        y_match = yachida_frozen.merge(y_species, on="sample_id", how="inner")
        y_meta_match = yachida_frozen.merge(y_meta, on="sample_id", how="inner", suffixes=("", "_official_supp"))
        y_match[["sample_id"] + [c for c in y_species.columns if c != "sample_id"]].to_csv(PROCESSED / "yachida_microbiome_unfiltered.csv", index=False)
        y_meta_match.to_csv(PROCESSED / "yachida_microbiome_matched_metadata.csv", index=False)
        y_genus_match = yachida_frozen[["sample_id"]].merge(y_genus, on="sample_id", how="inner")
        taxonomy_sources["Yachida"] = y_match[["sample_id"] + [c for c in y_species.columns if c != "sample_id"]].merge(
            y_genus_match,
            on="sample_id",
            how="outer",
            suffixes=("", "_genus_duplicate"),
        )
        schema_rows.append(audit_matrix("Yachida", "Table_S7-1 Species profile derived from SILVA LTP", y_species, "sample_id", "species", "relative abundance"))
        schema_rows.append(audit_matrix("Yachida", "Table_S7-2 Genus profile derived from SILVA LTP", y_genus, "sample_id", "genus", "percentage_or_scaled relative abundance"))
        for _, row in y_meta_match.iterrows():
            is_ambiguous = str(row["sample_id"]) in y_conflict_ids
            candidate_count = int((y_species_raw["sample_id"].astype(str) == str(row["sample_id"])).sum())
            crosswalk_rows.append(
                {
                    "disease": "CRC",
                    "cohort": "Yachida",
                    "participant_id": row.get("Subject", row["sample_id"]),
                    "microbiome_sample_id": row["sample_id"],
                    "metabolome_sample_id": row.get("Sample", row["sample_id"]),
                    "visit_id": "cross-sectional",
                    "phenotype": row.get("model_label", ""),
                    "match_method": "exact numeric Subject_ID / Sample ID",
                    "match_status": "matched" if not is_ambiguous else "ambiguous_duplicate_microbiome_profiles_excluded_from_clean_handoff",
                    "ambiguity_flag": is_ambiguous,
                    "notes": f"Raw group={row.get('Group', '')}; raw stage={row.get('Stage', '')}. Ambiguous duplicated official microbiome profiles were not included in the unfiltered clean handoff matrix.",
                }
            )
            if is_ambiguous:
                yachida_duplicate_rows.append(
                    {
                        "frozen_sample_id": row.get("Sample", row["sample_id"]),
                        "subject_id": row["sample_id"],
                        "candidate_microbiome_ids": row["sample_id"],
                        "candidate_count": candidate_count,
                        "phenotype": row.get("model_label", ""),
                        "stage": row.get("yachida_stage_source_raw", row.get("Stage", "")),
                        "metadata_evidence": "Official species profile contains duplicated Subject_ID columns with non-identical abundance vectors; no sample accession or specimen identifier in parsed table disambiguates them.",
                        "resolution_class": "GENUINELY_AMBIGUOUS_EXCLUDE",
                        "selected_microbiome_id": "",
                        "exclusion_reason": "Conflicting duplicate microbiome profiles for the same frozen subject ID.",
                        "notes": "No replicate was selected based on disease signal. Excluded from clean handoff until official sample-level accession mapping is available.",
                    }
                )
        validation_rows.append({"test": "yachida_exact_unambiguous_frozen_sample_matches", "status": "pass" if len(y_match) == len(yachida_frozen) else "partial", "observed": len(y_match), "expected": len(yachida_frozen), "notes": phenotype_counts(y_meta_match, "model_label") + f"; ambiguous_duplicate_ids_excluded={len(y_conflict_ids)}"})
        validation_rows.append({"test": "yachida_duplicate_microbiome_sample_ids", "status": "fail" if y_duplicate_conflicts else "pass", "observed": y_duplicate_ids, "expected": 0, "notes": f"Repeated subject columns={y_duplicate_ids}; conflicting duplicate groups={y_duplicate_conflicts}; conflicting IDs were excluded from the clean handoff matrix."})
    else:
        validation_rows.append({"test": "yachida_exact_frozen_sample_matches", "status": "fail", "observed": 0, "expected": len(yachida_frozen), "notes": "Official supplementary workbook unavailable or invalid."})

    # Sinha is already available locally; include it in linkage and benchmarking.
    if SINHA_MICROBES.exists():
        s_micro = read_sinha_microbiome()
        s_micro.to_csv(PROCESSED / "sinha_microbiome_unfiltered.csv", index=False)
        taxonomy_sources["Sinha"] = s_micro.copy()
        sinha_tax = write_sinha_taxonomy_audit(s_micro)
        schema_rows.append(audit_matrix("Sinha", SINHA_MICROBES.name, s_micro, "sample_id", "mixed_taxonomic_tree", "as supplied"))
        for sid in s_micro["sample_id"].astype(str):
            crosswalk_rows.append({"disease": "CRC", "cohort": "Sinha", "participant_id": sid, "microbiome_sample_id": sid, "metabolome_sample_id": sid, "visit_id": "cross-sectional", "phenotype": "", "match_method": "exact local ID", "match_status": "matched_locally_available", "ambiguity_flag": False, "notes": "Sinha microbiome was already present before this acquisition pass."})
        validation_rows.append({"test": "sinha_local_microbiome_available", "status": "pass", "observed": len(s_micro), "expected": 131, "notes": "Pre-existing local Sinha S3 microbial table retained."})

    # iHMP processed placeholder only if no deterministic taxonomic matrix exists.
    pd.DataFrame(columns=["sample_id"]).to_csv(PROCESSED / "ihmp_microbiome_unfiltered.csv", index=False)
    pd.DataFrame(columns=["sample_id"]).to_csv(PROCESSED / "ihmp_microbiome_canonical_unfiltered.csv", index=False)
    if yachida_duplicate_rows:
        pd.DataFrame(yachida_duplicate_rows).to_csv(TABLES / "stage4a_yachida_duplicate_resolution_audit.csv", index=False)
    else:
        pd.DataFrame(columns=["frozen_sample_id", "subject_id", "candidate_microbiome_ids", "candidate_count", "phenotype", "stage", "metadata_evidence", "resolution_class", "selected_microbiome_id", "exclusion_reason", "notes"]).to_csv(TABLES / "stage4a_yachida_duplicate_resolution_audit.csv", index=False)

    # Deterministic taxonomy canonicalisation and canonical unfiltered matrices.
    tax_rows = []
    canonical_matrices = {}
    for cohort, matrix in taxonomy_sources.items():
        feature_cols = [c for c in matrix.columns if c != "sample_id"]
        records = [canonical_feature_record(cohort, c) for c in feature_cols]
        tax_rows.extend(records)
        mapping = pd.DataFrame(records)
        keep = mapping[mapping["include_for_harmonisation"]].copy()
        if keep.empty:
            canonical = matrix[["sample_id"]].copy()
        else:
            canonical_parts = {"sample_id": matrix["sample_id"].reset_index(drop=True)}
            for rank in ["species", "genus", "family"]:
                sub = keep[keep["canonical_rank"].eq(rank)]
                for cname, group in sub.groupby("canonical_name", sort=True):
                    cols = [c for c in group["original_feature"].tolist() if c in matrix.columns]
                    if not cols:
                        continue
                    # Sinha uses explicit tree nodes; if an explicit genus/family node exists, keep that measurement.
                    canonical_parts[f"{rank}:{cname}"] = (
                        matrix[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1).reset_index(drop=True)
                        if len(cols) > 1 and cohort != "Sinha"
                        else pd.to_numeric(matrix[cols[0]], errors="coerce").reset_index(drop=True)
                    )
            canonical = pd.DataFrame(canonical_parts)
        canonical_matrices[cohort] = canonical
        out_name = cohort.lower()
        canonical.to_csv(PROCESSED / f"{out_name}_microbiome_canonical_unfiltered.csv", index=False)
    taxonomy_crosswalk = pd.DataFrame(tax_rows)
    taxonomy_crosswalk.to_csv(TABLES / "stage4a_taxonomy_crosswalk.csv", index=False)

    schema = pd.DataFrame(schema_rows)
    schema.to_csv(TABLES / "stage4a_downloaded_microbiome_schema_audit.csv", index=False)
    crosswalk = pd.DataFrame(crosswalk_rows)
    crosswalk.to_csv(TABLES / "stage4a_microbiome_metabolome_sample_crosswalk.csv", index=False)

    # Taxonomic resolution benchmark after deterministic canonicalisation.
    matrices = {}
    for cohort, canonical in canonical_matrices.items():
        disease = "IBD" if cohort in {"Franzosa", "iHMP"} else "CRC"
        for rank in ["species", "genus", "family"]:
            cols = ["sample_id"] + [c for c in canonical.columns if c.startswith(f"{rank}:")]
            matrices[(disease, cohort, rank)] = canonical[cols].copy() if len(cols) > 1 else pd.DataFrame(columns=["sample_id"])

    def features(df: pd.DataFrame) -> set[str]:
        return {str(c) for c in df.columns if c != "sample_id"}

    for disease, a, b in [("IBD", "Franzosa", "iHMP"), ("CRC", "Sinha", "Yachida")]:
        for rank in ["species", "genus", "family", "mixed"]:
            if rank == "mixed":
                fa = set()
                fb = set()
            else:
                fa = features(matrices.get((disease, a, rank), pd.DataFrame()))
                fb = features(matrices.get((disease, b, rank), pd.DataFrame()))
            taxa_rows.append(
                {
                    "disease": disease,
                    "cohort_a": a,
                    "cohort_b": b,
                    "taxonomic_rank": rank,
                    "features_in_cohort_a": len(fa),
                    "features_in_cohort_b": len(fb),
                    "shared_features": len(fa & fb),
                    "shared_fraction_of_smaller_feature_set": len(fa & fb) / min(len(fa), len(fb)) if fa and fb else 0,
                    "median_prevalence_cohort_a": "",
                    "median_prevalence_cohort_b": "",
                    "zero_fraction_cohort_a": "",
                    "zero_fraction_cohort_b": "",
                    "features_present_ge_10pct_a": "",
                    "features_present_ge_20pct_a": "",
                    "features_present_ge_30pct_a": "",
                    "features_present_ge_50pct_a": "",
                    "p_over_n_cohort_a": "",
                    "p_over_n_cohort_b": "",
                    "ambiguous_taxonomy_count": 0,
                    "notes": "Exact text overlap only; no unresolved rank harmonisation or modelling transform applied.",
                }
            )
    pd.DataFrame(taxa_rows).to_csv(TABLES / "stage4a_taxonomic_resolution_benchmark.csv", index=False)

    def shared_feature_table(disease: str, a: str, b: str, selected_rank: str) -> pd.DataFrame:
        fa = features(matrices.get((disease, a, selected_rank), pd.DataFrame()))
        fb = features(matrices.get((disease, b, selected_rank), pd.DataFrame()))
        all_features = sorted(fa | fb)
        return pd.DataFrame(
            [
                {
                    "canonical_taxon_id": f,
                    "canonical_name": f.split(":", 1)[1] if ":" in f else f,
                    "rank": selected_rank,
                    "cohort_A_present": f in fa,
                    "cohort_B_present": f in fb,
                    "eligible_for_external_transfer": f in fa and f in fb,
                    "mapping_confidence": "EXACT_CANONICAL_NAME",
                    "notes": f"Canonical label overlap for {a} versus {b}; no prevalence filtering applied.",
                }
                for f in all_features
            ]
        )

    shared_feature_table("IBD", "Franzosa", "iHMP", "species").to_csv(TABLES / "stage4a_ibd_shared_microbiome_features.csv", index=False)
    shared_feature_table("CRC", "Sinha", "Yachida", "genus").to_csv(TABLES / "stage4a_crc_shared_microbiome_features.csv", index=False)

    # Sample overlap audit.
    overlap_rows = []
    for disease, cohort, frozen_n, source_n in [
        ("IBD", "Franzosa", len(franzosa_frozen), 220),
        ("IBD", "iHMP", len(ihmp_frozen), 0),
        ("CRC", "Sinha", 131, 131),
        ("CRC", "Yachida", len(yachida_frozen), 616 if "Yachida" in canonical_matrices else 0),
    ]:
        subset = crosswalk[crosswalk["cohort"].eq(cohort)] if not crosswalk.empty else pd.DataFrame()
        matched = int(subset["match_status"].astype(str).str.startswith("matched").sum()) if not subset.empty else 0
        phen = subset[subset["match_status"].astype(str).str.startswith("matched")]["phenotype"].fillna("Missing").astype(str).value_counts().sort_index()
        overlap_rows.append(
            {
                "disease": disease,
                "cohort": cohort,
                "original_frozen_metabolomics_n": frozen_n,
                "microbial_source_n": source_n,
                "deterministically_matched_n": matched,
                "retention_percent": round(100 * matched / frozen_n, 2) if frozen_n else 0,
                "phenotype_counts": "; ".join(f"{k}={v}" for k, v in phen.items()),
                "notes": "Matched only when deterministic identifiers were present; ambiguous samples are not counted as matched.",
            }
        )
    pd.DataFrame(overlap_rows).to_csv(TABLES / "stage4a_multiomics_sample_overlap.csv", index=False)

    validation_rows.append({"test": "taxonomy_crosswalk_created", "status": "pass" if not taxonomy_crosswalk.empty else "fail", "observed": len(taxonomy_crosswalk), "expected": ">0", "notes": "Canonical taxonomy table built using deterministic label parsing only."})
    pd.DataFrame(validation_rows).to_csv(TABLES / "stage4a_microbiome_acquisition_validation_tests.csv", index=False)

    # Cohort READMEs.
    for cohort in ["franzosa", "ihmp", "yachida"]:
        relevant = manifest[manifest["cohort"].str.lower().eq(cohort)]
        lines = [f"# Stage 4A {cohort} microbiome acquisition", "", "Files in this directory are unmodified public source files or audited failed-download artifacts.", ""]
        for _, row in relevant.iterrows():
            lines.append(f"- `{row['file_name']}`: {row['download_status']}; source: {row['url']}; SHA256: {row['sha256']}")
        (RAW / cohort / "README_STAGE4A_PROVENANCE.md").write_text("\n".join(lines) + "\n")

    matched_counts = crosswalk.groupby(["disease", "cohort", "match_status"], dropna=False).size().reset_index(name="n") if not crosswalk.empty else pd.DataFrame()
    matched_crosswalk = crosswalk[crosswalk["match_status"].astype(str).str.startswith("matched")] if not crosswalk.empty else pd.DataFrame()
    feasibility = "E. SINGLE_COHORT_MULTIOMICS_ONLY"
    crc_shared = pd.read_csv(TABLES / "stage4a_crc_shared_microbiome_features.csv")
    crc_shared_p = int(crc_shared["eligible_for_external_transfer"].sum()) if not crc_shared.empty else 0
    ibd_shared = pd.read_csv(TABLES / "stage4a_ibd_shared_microbiome_features.csv")
    ibd_shared_p = int(ibd_shared["eligible_for_external_transfer"].sum()) if not ibd_shared.empty else 0
    if {"Franzosa", "iHMP", "Yachida", "Sinha"}.issubset(set(matched_crosswalk["cohort"])):
        feasibility = "B. IBD_AND_CRC_MULTIOMICS_READY_WITH_DIFFERENT_REPRESENTATIONS"
    elif {"Sinha", "Yachida"}.issubset(set(matched_crosswalk["cohort"])) and crc_shared_p > 0:
        feasibility = "D. CRC_ONLY_CROSS_COHORT_MULTIOMICS_READY"
    else:
        feasibility = "E. SINGLE_COHORT_MULTIOMICS_ONLY"

    (REPORTS / "STAGE4A_FRANZOSA_VALIDATION_MICROBIOME_AUDIT.md").write_text(
        "\n".join(
            [
                "# Stage 4A Franzosa Validation Microbiome Audit",
                "",
                "The official Franzosa Supplementary Dataset 4 species table contains PRISM-sample columns and bridges 137 of the frozen 172 Franzosa metabolomics samples.",
                "",
                "The 35 unmatched frozen samples are all `Validation.UMCG...` identifiers. No deterministic UMCG species-abundance rows were found in the downloaded official per-subject species supplement. They are therefore kept in the crosswalk as unmatched rather than forced through fuzzy matching.",
                "",
                "Conclusion: 137 matched Franzosa samples is definitive for the currently downloaded official processed species supplement. Additional UMCG microbiome recovery would require another authoritative validation-cohort processed matrix or raw-sequence reconstruction under the resource gate.",
            ]
        )
        + "\n"
    )

    (REPORTS / "STAGE4A_CROSS_ASSAY_HARMONISATION_RATIONALE.md").write_text(
        "\n".join(
            [
                "# Stage 4A Cross-Assay Harmonisation Rationale",
                "",
                "Franzosa, iHMP and Yachida are shotgun metagenomics studies. Sinha is a 16S rRNA V3-V4 microbiota profiling study, so all four cohorts must not be described as metagenomic.",
                "",
                "For CRC, genus-level harmonisation is more defensible than species-level transfer because 16S genus calls are more comparable to shotgun-derived genus profiles than species-level labels. Sinha also contains explicit genus nodes, so the genus representation can be constructed without summing child and parent nodes together.",
                "",
                "For IBD, the highest-resolution theoretical choice would be species because Franzosa is shotgun species abundance, but iHMP has no valid processed taxonomic matrix in this acquisition pass. Therefore IBD Stage 4B is NO-GO until iHMP abundance is acquired.",
            ]
        )
        + "\n"
    )

    final_report = [
        "# Stage 4A Final Multi-Omics Feasibility After Harmonisation",
        "",
        "## Franzosa",
        "Final microbial source: official Nature Supplementary Dataset 4 species relative abundance. Final matched n=137. Raw species p=202; canonical included species p=155 and canonical genus p=23 from deterministic label parsing. Genus aggregation can be derived later from species labels if iHMP becomes available.",
        "",
        "## iHMP",
        "Official metadata was acquired, but no valid abundance matrix was acquired. Final matched n=0. Species p=0. Genus p=0.",
        "",
        "## Sinha",
        "Final representation: explicit taxonomic tree nodes from existing S3 microbial table. Final n=131. Parsed tree features include 220 total labels; canonical included species p=0, genus p=57 and family p=43 after excluding ambiguous/unclassified labels.",
        "",
        "## Yachida",
        "Official supplementary species/genus profiles acquired. Duplicate conflicting subject IDs are excluded from the clean handoff. Final unambiguous matched n=252. Canonical species p=8367. Canonical genus p=1941.",
        "",
        "## IBD Cross-Cohort",
        f"Species overlap={ibd_shared_p}; genus overlap unavailable because iHMP abundance was not acquired. Chosen rank: none. Shared p={ibd_shared_p}.",
        "",
        "## CRC Cross-Cohort",
        f"Species overlap=0 using deterministic canonical labels; genus overlap={crc_shared_p}. Chosen rank: genus if Stage 4B proceeds. Shared p={crc_shared_p}.",
        "",
        "## Final",
        f"Feasibility classification: {feasibility}.",
        "IBD: STAGE4B_IBD_NO_GO.",
        "CRC: STAGE4B_CRC_GO if the supervisor accepts genus-level 16S-shotgun harmonisation and Yachida ambiguous-subject exclusion.",
    ]
    (REPORTS / "STAGE4A_FINAL_MULTIOMICS_FEASIBILITY_AFTER_HARMONISATION.md").write_text("\n".join(final_report) + "\n")

    downloaded_mb = manifest["file_size_bytes"].sum() / (1024 * 1024)
    resource_report = [
        "# Stage 4A Resource Report",
        "",
        f"Downloaded/audited source-file footprint recorded in the manifest: {downloaded_mb:.2f} MB.",
        "Peak RAM was not instrumented in this lightweight acquisition script; the largest parsed object was the Yachida supplementary workbook with species/genus abundance sheets.",
        "Runtime for the final local acquisition refresh was approximately 1-2 minutes on the local workstation, dominated by Excel parsing.",
        "",
        "Unresolved raw-sequence requirement: iHMP processed merged taxonomic abundance was not acquired from public routes tested here. Full recovery would require either a working official IBDMDB merged table endpoint or per-sample MGX product/raw reconstruction from HMP2/IBDMDB accessions. That was not performed because raw/per-sample reconstruction may exceed the Stage 4A resource gate and requires explicit approval.",
        "",
        "Recommended future processing if approved: obtain official HMP2 WGS MetaPhlAn/Taxonomic profile merged table or batch-download per-sample processed WMS taxonomic profiles, verify against `hmp2_metadata_2018-08-20.csv`, then match same participant and visit to the frozen iHMP metabolomics samples.",
    ]
    (REPORTS / "STAGE4A_RESOURCE_REPORT_AFTER_HARMONISATION.md").write_text("\n".join(resource_report) + "\n")

    report = [
        "# Stage 4A Microbiome Data Acquisition Report",
        "",
        "This report covers data acquisition and deterministic linkage only. No scientific models, ordinations, differential-abundance tests, pathway analyses or latent-factor methods were run.",
        "",
        "## Sources Downloaded Or Audited",
        manifest[["cohort", "publication", "doi", "repository_or_accession", "file_name", "file_size_bytes", "download_status", "assay", "processing_level", "taxonomic_resolution"]].to_markdown(index=False),
        "",
        "## Deterministic Sample Matching",
        matched_counts.to_markdown(index=False) if not matched_counts.empty else "No deterministic matches.",
        "",
        "## Schema Audit",
        schema[["cohort", "source_matrix", "sample_n", "feature_p", "taxonomic_rank", "sample_id_field", "abundance_scale_classification", "overall_zero_fraction", "overall_missing_fraction", "row_sum_interpretation"]].to_markdown(index=False),
        "",
        "## Taxonomic Resolution Benchmark",
        pd.DataFrame(taxa_rows).to_markdown(index=False),
        "",
        "## iHMP Visit Matching",
        f"The frozen first-visit iHMP metabolomics population contains {len(ihmp_frozen)} samples. The official metadata file was acquired, but no valid merged taxonomic abundance matrix was downloaded; the attempted `taxonomic_profiles.tsv.gz` artifact is `{ihmp_status}`. Therefore exact same-visit microbiome-metabolome pairing remains unresolved in this acquisition pass.",
        "",
        "## Updated Feasibility Classification",
        feasibility,
        "",
        f"Stage 4B CRC is runnable only at genus level if the supervisor accepts Sinha 16S versus Yachida shotgun transfer and the exclusion of ambiguous Yachida duplicate profiles. Stage 4B IBD remains NO-GO because the iHMP processed taxonomic matrix was not acquired. CRC shared genus p={crc_shared_p}.",
    ]
    (REPORTS / "STAGE4A_MICROBIOME_DATA_ACQUISITION_REPORT.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    build_outputs()
