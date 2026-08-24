from __future__ import annotations

import json
import re
import textwrap
import hashlib
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "Colon_Cancer_Data_Set"
OUT_DIR = ROOT / "thesis_preprocessing_workflow" / "crc_pipeline"
TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "reports"
EXTERNAL_DIR = OUT_DIR / "external_metadata_sources"
STAGE2_DIR = TABLE_DIR / "stage2_model_prep"
RANDOM_SEED = 42
PRIMARY_MISSINGNESS_THRESHOLD = 0.30


COHORTS = {
    "Sinha": RAW_DIR / "Sinha-et-al",
    "Yachida": RAW_DIR / "Yachinda_et-al",
}


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def clean_identifier(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "-", "none"}:
        return None
    return text.upper()


def split_ids(value: object) -> list[str]:
    if pd.isna(value):
        return []
    parts = re.split(r"[;,|]\s*", str(value))
    return [x for x in (clean_identifier(p) for p in parts) if x]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalise_yachida_stage(value: object) -> str | None:
    text = clean_identifier(value)
    if text is None:
        return None
    text = text.replace("STAGE_", "").replace("STAGE ", "")
    aliases = {"IVA": "IVA", "IVB": "IVB", "IVA.": "IVA", "IVB.": "IVB"}
    if text in aliases:
        return aliases[text]
    return text


def yachida_stage_major(stage: object) -> str | None:
    st = normalise_yachida_stage(stage)
    if st is None:
        return None
    if st == "0":
        return "Stage 0"
    if st == "I":
        return "Stage I"
    if st.startswith("II") and not st.startswith("III"):
        return "Stage II"
    if st.startswith("III"):
        return "Stage III"
    if st.startswith("IV"):
        return "Stage IV"
    return None


def yachida_source_faithful_label(group: object, stage: object) -> str:
    g = str(group)
    st = normalise_yachida_stage(stage)
    if g == "Healthy":
        return "Healthy"
    if g == "MP":
        return "MP"
    if g == "HS":
        return "HS"
    if st == "0":
        return "Stage 0"
    if st in {"I", "IIA", "IIB", "IIC", "II"}:
        return f"Stage {st}"
    if st in {"IIIA", "IIIB", "IIIC", "III"}:
        return f"Stage {st}"
    if st in {"IVA", "IVB", "IV"}:
        return f"Stage {st}"
    return "Unknown"


def yachida_harmonized_stage_label(group: object, stage: object) -> str:
    g = str(group)
    major = yachida_stage_major(stage)
    if g == "Healthy":
        return "Healthy"
    if g == "MP":
        return "Precursor lesion"
    if g == "HS":
        return "Post-surgical control"
    if major == "Stage 0":
        return "Non-invasive CRC"
    if major in {"Stage I", "Stage II"}:
        return "Localized CRC"
    if major == "Stage III":
        return "Regional CRC"
    if major == "Stage IV":
        return "Metastatic CRC"
    return "Unknown"


def yachida_trajectory_fields(group: object, stage: object) -> tuple[str | None, float, bool, str]:
    source_label = yachida_source_faithful_label(group, stage)
    order = {
        "Healthy": 0,
        "MP": 1,
        "Stage 0": 2,
        "Stage I": 3,
        "Stage IIA": 4,
        "Stage IIB": 4,
        "Stage IIC": 4,
        "Stage II": 4,
        "Stage IIIA": 5,
        "Stage IIIB": 5,
        "Stage IIIC": 5,
        "Stage III": 5,
        "Stage IVA": 6,
        "Stage IVB": 6,
        "Stage IV": 6,
    }
    if source_label == "HS":
        return source_label, np.nan, False, "HS is a post-surgical control group, not part of the exploratory cross-sectional ordered disease-state variable"
    if source_label == "Unknown":
        return source_label, np.nan, False, "stage unresolved"
    return source_label, float(order[source_label]), True, ""


def infer_file_role(path: Path, df: pd.DataFrame) -> str:
    name = path.name.lower()
    if "metadata" in name:
        return "sample-level metadata / phenotype / clinical covariates"
    if "map" in name:
        return "feature-level metabolite annotation and identifier mapping"
    if "mtb" in name:
        return "sample-by-metabolite abundance matrix"
    if path.suffix.lower() == ".pdf":
        return "source research paper"
    return "unclassified CRC source file"


def infer_level(df: pd.DataFrame, role: str) -> str:
    cols = set(df.columns)
    if "Sample" in cols and "Subject" in cols:
        return "sample-level with participant identifier"
    if "Sample" in cols:
        return "sample-level abundance table"
    if {"Compound", "HMDB", "KEGG"} & cols:
        return "feature-level annotation"
    return "unknown"


def disease_mapping(cohort: str, metadata: pd.DataFrame) -> pd.DataFrame:
    out = metadata.copy()
    if cohort == "Sinha":
        out["crc_status"] = out["Study.Group"].map({0: "Control", 1: "CRC"})
        out["stage_model_label"] = np.nan
        out["adenoma_status"] = np.nan
        out["tumour_site"] = np.nan
    else:
        group = out["Study.Group"].astype(str)
        out["crc_status"] = np.select(
            [group.eq("Healthy"), group.isin(["Stage_0", "Stage_I_II", "Stage_III_IV"])],
            ["Control", "CRC"],
            default="Non-primary/adenoma/HS",
        )
        out["stage_model_label"] = group.map(
            {
                "Healthy": "Control",
                "MP": "Multiple polypoid adenoma",
                "Stage_0": "Stage 0 / intramucosal carcinoma",
                "Stage_I_II": "Stage I/II CRC",
                "Stage_III_IV": "Stage III/IV CRC",
            }
        )
        stage_clean = out.get("Stage", pd.Series(index=out.index, dtype=object)).astype(str).str.upper()
        out["stage_individual"] = stage_clean.replace({"-": np.nan, "NAN": np.nan})
        out["stage_major"] = np.select(
            [
                stage_clean.eq("0"),
                stage_clean.eq("I"),
                stage_clean.str.startswith("II") & ~stage_clean.str.startswith("III"),
                stage_clean.str.startswith("III"),
                stage_clean.str.startswith("IV"),
            ],
            ["Stage 0", "Stage I", "Stage II", "Stage III", "Stage IV"],
            default=None,
        )
        out["stage_refined_model_label"] = np.select(
            [
                group.eq("Healthy"),
                group.eq("MP"),
                group.eq("Stage_0"),
                group.eq("Stage_I_II"),
                stage_clean.str.startswith("III"),
                stage_clean.str.startswith("IV"),
            ],
            [
                "Control",
                "Multiple polypoid adenoma",
                "Stage 0 / intramucosal carcinoma",
                "Stage I/II CRC",
                "Stage III CRC",
                "Stage IV CRC",
            ],
            default=None,
        )
        out["adenoma_status"] = group.map({"MP": "Multiple polypoid adenoma"})
        out["tumour_site"] = out.get("Tumor location", pd.Series(index=out.index, dtype=object))
    return out


def numeric_matrix(mtb: pd.DataFrame) -> pd.DataFrame:
    return mtb.drop(columns=[mtb.columns[0]]).apply(pd.to_numeric, errors="coerce")


def make_inventory() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW_DIR.rglob("*")):
        if not path.is_file() or path.name == ".DS_Store":
            continue
        if path.suffix.lower() in {".tsv", ".csv", ".txt"}:
            df = read_tsv(path) if path.suffix.lower() == ".tsv" else pd.read_csv(path)
            role = infer_file_role(path, df)
            identifier_cols = [c for c in df.columns if c.lower() in {"sample", "subject", "compound", "hmdb", "kegg"}]
            rows.append(
                {
                    "path": str(path),
                    "file_type": path.suffix.lower(),
                    "n_rows": len(df),
                    "n_columns": len(df.columns),
                    "likely_role": role,
                    "identifier_columns": "; ".join(identifier_cols),
                    "data_level": infer_level(df, role),
                    "duplicate_key_count": int(df[identifier_cols[0]].duplicated().sum()) if identifier_cols else np.nan,
                }
            )
        elif path.suffix.lower() == ".pdf":
            rows.append(
                {
                    "path": str(path),
                    "file_type": ".pdf",
                    "n_rows": np.nan,
                    "n_columns": np.nan,
                    "likely_role": "source research paper",
                    "identifier_columns": "",
                    "data_level": "study-level methods/source",
                    "duplicate_key_count": np.nan,
                }
            )
    return pd.DataFrame(rows)


def metadata_audit(cohort: str, metadata: pd.DataFrame, mtb: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labelled = disease_mapping(cohort, metadata)
    repeated = labelled.groupby("Subject")["Sample"].nunique().reset_index(name="samples_per_subject")
    count_rows = []
    for col in [
        "Study.Group",
        "crc_status",
        "stage_model_label",
        "stage_refined_model_label",
        "stage_major",
        "Stage",
        "Tumor location",
        "Gender",
    ]:
        if col in labelled.columns:
            for value, sub in labelled.groupby(col, dropna=False):
                count_rows.append(
                    {
                        "cohort": cohort,
                        "label_column": col,
                        "class": value,
                        "n_samples": len(sub),
                        "n_participants": sub["Subject"].nunique() if "Subject" in sub else np.nan,
                    }
                )
    missing_rows = []
    important = [
        "Sample",
        "Subject",
        "Study.Group",
        "Age",
        "Gender",
        "BMI",
        "Stage",
        "Brinkman Index",
        "Alcohol",
        "Tumor location",
        "race",
        "hosp1",
        "hosp2",
        "Shared.w.ERAWIJANTARI_2020",
    ]
    for col in important:
        missing_rows.append(
            {
                "cohort": cohort,
                "field": col,
                "available": col in labelled.columns,
                "n_missing": int(labelled[col].isna().sum()) if col in labelled.columns else len(labelled),
                "missing_fraction": float(labelled[col].isna().mean()) if col in labelled.columns else 1.0,
                "n_unique_nonmissing": int(labelled[col].nunique(dropna=True)) if col in labelled.columns else 0,
            }
        )
    mtb_samples = set(mtb["Sample"].astype(str))
    meta_samples = set(labelled["Sample"].astype(str))
    flow = pd.DataFrame(
        [
            {"cohort": cohort, "item": "metadata_samples", "n": len(meta_samples)},
            {"cohort": cohort, "item": "metabolomics_samples", "n": len(mtb_samples)},
            {"cohort": cohort, "item": "samples_in_both_metadata_and_metabolomics", "n": len(meta_samples & mtb_samples)},
            {"cohort": cohort, "item": "metadata_only_samples", "n": len(meta_samples - mtb_samples)},
            {"cohort": cohort, "item": "metabolomics_only_samples", "n": len(mtb_samples - meta_samples)},
            {"cohort": cohort, "item": "unique_participants", "n": labelled["Subject"].nunique()},
            {"cohort": cohort, "item": "participants_with_repeated_samples", "n": int((repeated["samples_per_subject"] > 1).sum())},
        ]
    )
    return pd.DataFrame(count_rows), pd.DataFrame(missing_rows), flow


def feature_audit(cohort: str, mtb: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    vals = numeric_matrix(mtb)
    rows = []
    for col in vals.columns:
        s = vals[col]
        rows.append(
            {
                "cohort": cohort,
                "feature": col,
                "missing_fraction": float(s.isna().mean()),
                "zero_fraction": float((s == 0).mean()),
                "min": float(s.min(skipna=True)),
                "median": float(s.median(skipna=True)),
                "max": float(s.max(skipna=True)),
                "variance": float(s.var(skipna=True)),
            }
        )
    feature_qc = pd.DataFrame(rows)

    map_df = mapping.copy()
    map_df["cohort"] = cohort
    map_df["HMDB_clean"] = map_df["HMDB"].map(clean_identifier) if "HMDB" in map_df else None
    map_df["KEGG_clean"] = map_df["KEGG"].map(clean_identifier) if "KEGG" in map_df else None
    map_df["feature_in_abundance"] = map_df["Compound"].isin(vals.columns) if "Compound" in map_df else False
    return feature_qc, map_df


def identifier_overlap(maps: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    sinha = maps["Sinha"]
    yachida = maps["Yachida"]
    for raw_col in ["HMDB", "KEGG"]:
        a = set(str(x).strip() for x in sinha[raw_col].dropna() if str(x).strip())
        b = set(str(x).strip() for x in yachida[raw_col].dropna() if str(x).strip())
        rows.append(
            {
                "identifier_type": f"{raw_col}_raw",
                "Sinha_nonmissing_unique": len(a),
                "Yachida_nonmissing_unique": len(b),
                "shared_unique_identifiers": len(a & b),
            }
        )
    for id_col in ["HMDB_clean", "KEGG_clean"]:
        a = set(sinha[id_col].dropna())
        b = set(yachida[id_col].dropna())
        rows.append(
            {
                "identifier_type": id_col.replace("_clean", ""),
                "Sinha_nonmissing_unique": len(a),
                "Yachida_nonmissing_unique": len(b),
                "shared_unique_identifiers": len(a & b),
            }
        )

    long_rows = []
    for cohort, df in maps.items():
        for _, row in df.iterrows():
            ids = split_ids(row.get("HMDB_clean")) + split_ids(row.get("KEGG_clean"))
            for idv in sorted(set(ids)):
                long_rows.append({"cohort": cohort, "feature": row["Compound"], "identifier": idv})
    long = pd.DataFrame(long_rows)
    shared = set(long.loc[long["cohort"].eq("Sinha"), "identifier"]) & set(long.loc[long["cohort"].eq("Yachida"), "identifier"])
    rows.append(
        {
            "identifier_type": "HMDB_or_KEGG_unified",
            "Sinha_nonmissing_unique": long.loc[long["cohort"].eq("Sinha"), "identifier"].nunique(),
            "Yachida_nonmissing_unique": long.loc[long["cohort"].eq("Yachida"), "identifier"].nunique(),
            "shared_unique_identifiers": len(shared),
        }
    )
    shared_table = long[long["identifier"].isin(shared)].sort_values(["identifier", "cohort", "feature"])
    return pd.DataFrame(rows), shared_table


def transformation_summary(cohort: str, mtb: pd.DataFrame) -> pd.DataFrame:
    vals = numeric_matrix(mtb)
    flat = vals.to_numpy(dtype=float).ravel()
    flat = flat[np.isfinite(flat)]
    return pd.DataFrame(
        [
            {
                "cohort": cohort,
                "n_values": len(flat),
                "min": float(np.min(flat)),
                "p01": float(np.quantile(flat, 0.01)),
                "median": float(np.median(flat)),
                "mean": float(np.mean(flat)),
                "p99": float(np.quantile(flat, 0.99)),
                "max": float(np.max(flat)),
                "zero_fraction": float(np.mean(flat == 0)),
                "negative_fraction": float(np.mean(flat < 0)),
                "recommended_exploratory_transform": "none; values appear already transformed/standardized" if cohort == "Sinha" else "log1p for exploratory plots only; raw non-negative abundances are zero-inflated",
                "modeling_note": "any transformation choice must be inside the training-fold sklearn pipeline",
            }
        ]
    )


def plot_pca(cohort: str, metadata: pd.DataFrame, mtb: pd.DataFrame) -> None:
    labelled = disease_mapping(cohort, metadata)
    merged = labelled.merge(mtb, on="Sample", how="inner")
    features = mtb.columns[1:]
    x = merged[features].apply(pd.to_numeric, errors="coerce")
    x = x.loc[:, x.notna().mean() >= (1 - PRIMARY_MISSINGNESS_THRESHOLD)]
    x = x.loc[:, x.var(skipna=True) > 0]
    if cohort == "Yachida":
        x = np.log1p(x.clip(lower=0))
    x = x.fillna(x.median())
    z = StandardScaler().fit_transform(x)
    coords = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(z)
    pca = PCA(n_components=2, random_state=RANDOM_SEED).fit(z)
    plot_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "group": merged["crc_status"].astype(str)})
    plot_df = plot_df[plot_df["group"].isin(["Control", "CRC", "Non-primary/adenoma/HS"])]
    plt.figure(figsize=(7.5, 5.4))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="group", s=58, alpha=0.85)
    plt.axhline(0, color="0.9", lw=0.8)
    plt.axvline(0, color="0.9", lw=0.8)
    plt.title(f"Exploratory PCA within {cohort} CRC cohort")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    n_text = ", ".join(f"{k}: n={v}" for k, v in plot_df["group"].value_counts().items())
    plt.figtext(0.01, 0.01, f"Exploratory only; preprocessing here is descriptive, not model fitting. {n_text}", fontsize=8)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ["png", "svg"]:
        plt.savefig(FIGURE_DIR / f"crc_stage1_pca_{cohort}.{ext}", dpi=300)
    plt.close()


def plot_missingness(feature_qc: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=feature_qc, x="cohort", y="zero_fraction", color="#9ecae1")
    sns.stripplot(data=feature_qc, x="cohort", y="zero_fraction", color="#08519c", alpha=0.25, size=2)
    plt.ylabel("Feature zero fraction")
    plt.xlabel("")
    plt.title("CRC metabolomics zero fraction by cohort")
    plt.tight_layout()
    for ext in ["png", "svg"]:
        plt.savefig(FIGURE_DIR / f"crc_stage1_zero_fraction_by_cohort.{ext}", dpi=300)
    plt.close()


def extract_yachida_supplementary_metabolome_clinical() -> pd.DataFrame:
    path = EXTERNAL_DIR / "yachida_supplementary_tables_1_15.xlsx"
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_excel(path, sheet_name="Table_S2-2", header=None)
    header_idx = raw.index[raw.iloc[:, 0].astype(str).eq("Subject_ID")]
    if len(header_idx) == 0:
        return pd.DataFrame()
    header_idx = int(header_idx[0])
    clinical = raw.iloc[header_idx + 1 :].copy()
    clinical.columns = raw.iloc[header_idx].astype(str).str.strip()
    clinical = clinical.dropna(how="all")
    clinical = clinical[clinical["Subject_ID"].notna()].copy()
    clinical["Subject_ID"] = clinical["Subject_ID"].astype(str)
    clinical.to_csv(TABLE_DIR / "crc_stage1_yachida_supplementary_table_s2_2_metabolome_clinical.csv", index=False)
    return clinical


def external_metadata_recovery_audit(local_metadata: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    source_specs = [
        {
            "cohort": "Sinha",
            "source_name": "PLOS supplementary S4 covariates",
            "path": EXTERNAL_DIR / "sinha_s4_covariates.csv",
            "sample_col": "ID",
            "url": "https://journals.plos.org/plosone/article/file?type=supplementary&id=info:doi/10.1371/journal.pone.0152126.s004",
            "reader": lambda p: pd.read_csv(p),
        },
        {
            "cohort": "Sinha",
            "source_name": "Borenstein curated metadata",
            "path": EXTERNAL_DIR / "SINHA_CRC_2016_metadata_curated.tsv",
            "sample_col": "Sample",
            "url": "https://raw.githubusercontent.com/borenstein-lab/microbiome-metabolome-curated-data/main/data/processed_data/SINHA_CRC_2016/metadata.tsv",
            "reader": lambda p: pd.read_csv(p, sep="\t"),
        },
        {
            "cohort": "Yachida",
            "source_name": "Nature supplementary Table S2-2 metabolome clinical data",
            "path": EXTERNAL_DIR / "yachida_supplementary_tables_1_15.xlsx",
            "sample_col": "Subject_ID",
            "url": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41591-019-0458-7/MediaObjects/41591_2019_458_MOESM3_ESM.xlsx",
            "reader": lambda p: extract_yachida_supplementary_metabolome_clinical(),
        },
        {
            "cohort": "Yachida",
            "source_name": "Borenstein curated metadata",
            "path": EXTERNAL_DIR / "YACHIDA_CRC_2019_metadata_curated.tsv",
            "sample_col": "Sample",
            "url": "https://raw.githubusercontent.com/borenstein-lab/microbiome-metabolome-curated-data/main/data/processed_data/YACHIDA_CRC_2019/metadata.tsv",
            "reader": lambda p: pd.read_csv(p, sep="\t"),
        },
    ]
    field_terms = {
        "antibiotic": ["antibiotic", "antibiotics", "antimicrobial"],
        "steroid": ["steroid", "corticosteroid", "prednisone", "prednisolone"],
        "medication_or_treatment": ["medication", "medicine", "drug", "treatment", "therapy"],
        "stage_or_subtype": ["stage", "subtype", "histology", "group", "tumor location", "tumour location"],
        "smoking": ["smoking", "smok", "brinkman"],
        "alcohol": ["alcohol"],
    }
    for spec in source_specs:
        path = spec["path"]
        if not path.exists():
            rows.append(
                {
                    "cohort": spec["cohort"],
                    "source_name": spec["source_name"],
                    "source_file": str(path),
                    "source_url": spec["url"],
                    "downloaded": False,
                    "n_rows": 0,
                    "n_columns": 0,
                    "sample_overlap_with_local": 0,
                    "columns_recovered": "",
                    "sample_level_antibiotic_available": False,
                    "sample_level_steroid_available": False,
                    "sample_level_stage_or_subtype_available": False,
                    "action_taken": "not available locally; download source first",
                }
            )
            continue
        df = spec["reader"](path)
        cols = [str(c) for c in df.columns]
        lower_cols = {c: c.lower() for c in cols}
        found = {
            name: [c for c in cols if any(term in lower_cols[c] for term in terms)]
            for name, terms in field_terms.items()
        }
        if spec["cohort"] == "Sinha":
            # In Sinha, `case` and `Study.Group` are the binary CRC/control outcome,
            # not stage, tumour site, histology, or molecular subtype labels.
            found["stage_or_subtype"] = [
                c for c in found["stage_or_subtype"] if lower_cols[c] not in {"case", "study.group"}
            ]
        local = local_metadata[spec["cohort"]]
        local_samples = set(local["Sample"].astype(str))
        source_samples = set(df[spec["sample_col"]].astype(str)) if spec["sample_col"] in df.columns else set()
        action = "no new sample-level antibiotic/steroid fields recovered"
        if spec["cohort"] == "Yachida" and found["stage_or_subtype"]:
            action = "confirms local stage/tumour-location/smoking/alcohol metadata; no medication fields recovered"
        if spec["cohort"] == "Sinha" and not found["stage_or_subtype"]:
            action = "confirms binary case-control only; no sample-level CRC stage/subtype or medication fields recovered"
        rows.append(
            {
                "cohort": spec["cohort"],
                "source_name": spec["source_name"],
                "source_file": str(path),
                "source_url": spec["url"],
                "downloaded": True,
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "sample_overlap_with_local": len(local_samples & source_samples),
                "columns_recovered": "; ".join(cols),
                "sample_level_antibiotic_available": bool(found["antibiotic"]),
                "sample_level_steroid_available": bool(found["steroid"]),
                "sample_level_stage_or_subtype_available": bool(found["stage_or_subtype"]),
                "sample_level_smoking_available": bool(found["smoking"]),
                "sample_level_alcohol_available": bool(found["alcohol"]),
                "relevant_columns_found": json.dumps(found),
                "action_taken": action,
            }
        )
    return pd.DataFrame(rows)


def sinha_official_supporting_info_audit(local_metadata: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit official Sinha supplement files for participant-level stage/medication fields."""
    source_specs = [
        ("S1 Table", EXTERNAL_DIR / "sinha_s1_correlated_metabolites.xlsx", "xlsx"),
        ("S2 Table", EXTERNAL_DIR / "sinha_s2_metabolites.csv", "csv"),
        ("S3 Table", EXTERNAL_DIR / "sinha_s3_microbes.csv", "csv"),
        ("S4 Table", EXTERNAL_DIR / "sinha_s4_covariates.csv", "csv"),
        ("S5 Table", EXTERNAL_DIR / "sinha_s5_crc_72_pairs.xlsx", "xlsx"),
        ("Borenstein curated metadata", EXTERNAL_DIR / "SINHA_CRC_2016_metadata_curated.tsv", "tsv"),
    ]
    term_groups = {
        "stage_terms": ["stage", "invasive", "metasta", "non-invasive", "noninvasive"],
        "antibiotic_terms": ["antibiotic", "antimicrobial"],
        "steroid_terms": ["steroid", "corticosteroid", "prednisone", "prednisolone"],
        "immunosuppressive_terms": ["immunosuppress", "azathioprine", "mercaptopurine", "biologic"],
        "treatment_terms": ["treatment", "treated", "therapy", "medication", "medicine", "drug"],
    }
    local_samples = set(local_metadata["Sinha"]["Sample"].astype(str))
    rows = []
    for source_name, path, kind in source_specs:
        if not path.exists():
            rows.append(
                {
                    "source": source_name,
                    "path": str(path),
                    "downloaded": False,
                    "n_rows": 0,
                    "n_columns": 0,
                    "id_column": "",
                    "sample_overlap_with_local": 0,
                    "stage_field_exists": False,
                    "stage_column_candidates": "",
                    "medication_column_candidates": "",
                    "notes": "source file not downloaded",
                }
            )
            continue
        if kind == "xlsx":
            xl = pd.ExcelFile(path)
            frames = []
            for sheet in xl.sheet_names:
                frames.append(pd.read_excel(path, sheet_name=sheet))
            df = pd.concat(frames, axis=0, ignore_index=True, sort=False)
        elif kind == "tsv":
            df = pd.read_csv(path, sep="\t")
        else:
            df = pd.read_csv(path)
        cols = [str(c) for c in df.columns]
        lower_cols = {c: c.lower() for c in cols}
        stage_cols = [c for c in cols if any(t in lower_cols[c] for t in term_groups["stage_terms"])]
        medication_cols = [
            c
            for c in cols
            if any(
                t in lower_cols[c]
                for group in ["antibiotic_terms", "steroid_terms", "immunosuppressive_terms", "treatment_terms"]
                for t in term_groups[group]
            )
        ]
        id_col = next((c for c in ["ID", "Sample", "Subject"] if c in df.columns), "")
        overlap = len(local_samples & set(df[id_col].dropna().astype(str))) if id_col else 0
        rows.append(
            {
                "source": source_name,
                "path": str(path),
                "downloaded": True,
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "id_column": id_col,
                "sample_overlap_with_local": overlap,
                "stage_field_exists": bool(stage_cols),
                "stage_column_candidates": "; ".join(stage_cols),
                "medication_column_candidates": "; ".join(medication_cols),
                "notes": "official sample-level source; no participant-level CRC stage or medication field found"
                if not stage_cols and not medication_cols
                else "candidate terms found; inspect manually before use",
            }
        )
    audit = pd.DataFrame(rows)

    # Publication Table 1 contains aggregate stage percentages, not participant-level IDs.
    stage_counts = pd.DataFrame(
        [
            {
                "source": "Sinha et al. 2016 publication Table 1",
                "stage_category": "Non-invasive",
                "reported_percent_among_crc_cases": 21.4,
                "n_crc_cases": 42,
                "implied_count": 9,
                "participant_level_mapping_available": False,
            },
            {
                "source": "Sinha et al. 2016 publication Table 1",
                "stage_category": "Invasive, no known metastases",
                "reported_percent_among_crc_cases": 42.9,
                "n_crc_cases": 42,
                "implied_count": 18,
                "participant_level_mapping_available": False,
            },
            {
                "source": "Sinha et al. 2016 publication Table 1",
                "stage_category": "Metastatic",
                "reported_percent_among_crc_cases": 33.3,
                "n_crc_cases": 42,
                "implied_count": 14,
                "participant_level_mapping_available": False,
            },
            {
                "source": "Sinha et al. 2016 publication Table 1",
                "stage_category": "Missing",
                "reported_percent_among_crc_cases": 2.4,
                "n_crc_cases": 42,
                "implied_count": 1,
                "participant_level_mapping_available": False,
            },
        ]
    )
    return audit, stage_counts


def sinha_stage_linkage_decision() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "question": "Does participant-level Sinha cancer stage exist in official supporting data?",
                "answer": "No",
                "exact_stage_column_name": "",
                "available_outcome_column": "case",
                "available_outcome_coding": "0=control; 1=CRC case",
                "local_id_match": "S4 ID maps one-to-one to local Sample/Subject for 131/131 records",
                "unmatched_records": 0,
                "duplicated_records": 0,
                "conflicting_records": 0,
                "stage_counts_among_crc_cases": "Publication Table 1 aggregate only: Non-invasive n=9; Invasive/no known metastases n=18; Metastatic n=14; Missing n=1",
                "permitted_use": "aggregate descriptive context only",
                "not_permitted_use": "no sample-level stage sensitivity analysis; no secondary stage prediction task; no TNM/AJCC recoding",
            }
        ]
    )


def sinha_linked_source_stage_search_audit(local_metadata: dict[str, pd.DataFrame]) -> pd.DataFrame:
    local = local_metadata["Sinha"].copy()
    local_ids = set(local["Sample"].astype(str))
    rows = [
        {
            "candidate_source": "Sinha et al. 2016 Figshare record",
            "url": "https://api.figshare.com/v2/articles/3129739",
            "exact_file_or_table": "article file manifest",
            "stage_related_column_name": "",
            "original_coding": "",
            "linkage_identifier": "",
            "matches_local_sample_or_subject": False,
            "local_samples_linked": 0,
            "unmatched_local_samples": len(local_ids),
            "duplicated_linkage_records": 0,
            "conflicting_stage_records": 0,
            "missing_stage_records": 0,
            "counts_by_recovered_stage": "",
            "audit_result": "official repository exposes S1-S5 only; those files are audited separately",
        },
        {
            "candidate_source": "NYU Data Catalog dataset 10121",
            "url": "https://datacatalog.med.nyu.edu/dataset/10121",
            "exact_file_or_table": "catalog record linking to Figshare",
            "stage_related_column_name": "",
            "original_coding": "",
            "linkage_identifier": "",
            "matches_local_sample_or_subject": False,
            "local_samples_linked": 0,
            "unmatched_local_samples": len(local_ids),
            "duplicated_linkage_records": 0,
            "conflicting_stage_records": 0,
            "missing_stage_records": 0,
            "counts_by_recovered_stage": "",
            "audit_result": "catalog confirms 42 CRC cases and 89 controls and points to Figshare; no additional participant-level stage file is exposed in the catalog page",
        },
        {
            "candidate_source": "Ahn et al. 2013 JNCI source microbiome study",
            "url": "https://doi.org/10.1093/jnci/djt300",
            "exact_file_or_table": "linked source publication / supplementary trail",
            "stage_related_column_name": "",
            "original_coding": "",
            "linkage_identifier": "",
            "matches_local_sample_or_subject": False,
            "local_samples_linked": 0,
            "unmatched_local_samples": len(local_ids),
            "duplicated_linkage_records": 0,
            "conflicting_stage_records": 0,
            "missing_stage_records": 0,
            "counts_by_recovered_stage": "",
            "audit_result": "source study confirms same CRC/control context but no deterministic participant-level stage table matching local CRC### identifiers was recovered",
        },
        {
            "candidate_source": "Goedert et al. 2014 Carcinogenesis source metabolomics study",
            "url": "https://doi.org/10.1093/carcin/bgu131",
            "exact_file_or_table": "linked source publication / supplementary trail",
            "stage_related_column_name": "",
            "original_coding": "",
            "linkage_identifier": "",
            "matches_local_sample_or_subject": False,
            "local_samples_linked": 0,
            "unmatched_local_samples": len(local_ids),
            "duplicated_linkage_records": 0,
            "conflicting_stage_records": 0,
            "missing_stage_records": 0,
            "counts_by_recovered_stage": "",
            "audit_result": "stage is discussed at aggregate/disease-extent level, but no sample-level stage column keyed to local identifiers was recovered",
        },
        {
            "candidate_source": "Schiffman et al. 1989 Cancer Research fecal mutagenicity source study",
            "url": "https://pubmed.ncbi.nlm.nih.gov/2655896/",
            "exact_file_or_table": "source epidemiology publication",
            "stage_related_column_name": "",
            "original_coding": "",
            "linkage_identifier": "",
            "matches_local_sample_or_subject": False,
            "local_samples_linked": 0,
            "unmatched_local_samples": len(local_ids),
            "duplicated_linkage_records": 0,
            "conflicting_stage_records": 0,
            "missing_stage_records": 0,
            "counts_by_recovered_stage": "",
            "audit_result": "no modern participant-level machine-readable stage file with local identifiers was recovered",
        },
        {
            "candidate_source": "Schiffman et al. 1989 Cancer Research fecapentaene source study",
            "url": "https://pubmed.ncbi.nlm.nih.gov/2917361/",
            "exact_file_or_table": "source epidemiology publication",
            "stage_related_column_name": "",
            "original_coding": "",
            "linkage_identifier": "",
            "matches_local_sample_or_subject": False,
            "local_samples_linked": 0,
            "unmatched_local_samples": len(local_ids),
            "duplicated_linkage_records": 0,
            "conflicting_stage_records": 0,
            "missing_stage_records": 0,
            "counts_by_recovered_stage": "",
            "audit_result": "no modern participant-level machine-readable stage file with local identifiers was recovered",
        },
    ]
    return pd.DataFrame(rows)


def yachida_stage_recovery_audit(local_metadata: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    supp = extract_yachida_supplementary_metabolome_clinical()
    local = local_metadata["Yachida"].copy()
    if supp.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    local["_merge_sample_id"] = local["Sample"].astype(str)
    supp["_merge_subject_id"] = supp["Subject_ID"].astype(str)
    merged = local.merge(
        supp,
        left_on="_merge_sample_id",
        right_on="_merge_subject_id",
        how="left",
        suffixes=("_local", "_supplement"),
    )
    merged["yachida_stage_source_raw"] = merged["Stage_supplement"]
    merged["yachida_stage_normalized"] = merged["Stage_supplement"].map(normalise_yachida_stage)
    stage_supp = merged["yachida_stage_normalized"].astype(str)
    merged["yachida_stage_major"] = merged["Stage_supplement"].map(yachida_stage_major)
    merged["yachida_stage_substage"] = np.where(
        merged["yachida_stage_major"].isin(["Stage III", "Stage IV"]),
        merged["yachida_stage_normalized"],
        "",
    )
    merged["yachida_stage_original_group"] = merged["Study.Group"]
    merged["yachida_stage_source_faithful_label"] = [
        yachida_source_faithful_label(g, s) for g, s in zip(merged["Study.Group"], merged["Stage_supplement"])
    ]
    merged["yachida_stage_harmonized_label"] = [
        yachida_harmonized_stage_label(g, s) for g, s in zip(merged["Study.Group"], merged["Stage_supplement"])
    ]
    trajectory = [yachida_trajectory_fields(g, s) for g, s in zip(merged["Study.Group"], merged["Stage_supplement"])]
    merged["trajectory_label"] = [x[0] for x in trajectory]
    merged["trajectory_order"] = [x[1] for x in trajectory]
    merged["trajectory_eligible"] = [x[2] for x in trajectory]
    merged["trajectory_exclusion_reason"] = [x[3] for x in trajectory]
    merged["yachida_stage_recovery_source"] = "Yachida et al. 2019 Nature Medicine Supplementary Table S2-2"
    merged["yachida_stage_recovery_status"] = np.where(merged["Subject_ID"].notna(), "matched", "unmatched")
    refined = np.select(
        [
            merged["Study.Group"].eq("Healthy"),
            merged["Study.Group"].eq("MP"),
            merged["Study.Group"].eq("Stage_0"),
            merged["Study.Group"].eq("Stage_I_II"),
            stage_supp.str.startswith("III"),
            stage_supp.str.startswith("IV"),
        ],
        [
            "Control",
            "Multiple polypoid adenoma",
            "Stage 0 / intramucosal carcinoma",
            "Stage I/II CRC",
            "Stage III CRC",
            "Stage IV CRC",
        ],
        default=None,
    )
    merged["stage_refined_from_official_supplement"] = refined
    merged["stage_major_from_official_supplement"] = np.select(
        [
            stage_supp.eq("0"),
            stage_supp.eq("I"),
            stage_supp.str.startswith("II") & ~stage_supp.str.startswith("III"),
            stage_supp.str.startswith("III"),
            stage_supp.str.startswith("IV"),
        ],
        ["Stage 0", "Stage I", "Stage II", "Stage III", "Stage IV"],
        default=None,
    )
    expected_major_by_group = {
        "Stage_0": {"Stage 0"},
        "Stage_I_II": {"Stage I", "Stage II"},
        "Stage_III_IV": {"Stage III", "Stage IV"},
        "Healthy": {None},
        "MP": {None},
        "HS": {None},
    }
    mismatch_mask = [
        row["yachida_stage_major"] not in expected_major_by_group.get(row["Study.Group"], set())
        for _, row in merged.iterrows()
    ]
    mismatches = merged.loc[
        mismatch_mask,
        ["Sample", "Subject", "Study.Group", "Stage_local", "Group", "Stage_supplement"],
    ]
    detailed = merged[
        [
            "Sample",
            "Subject",
            "Study.Group",
            "Stage_local",
            "Subject_ID",
            "Group",
            "Stage_supplement",
            "yachida_stage_source_raw",
            "yachida_stage_normalized",
            "yachida_stage_major",
            "yachida_stage_substage",
            "yachida_stage_original_group",
            "yachida_stage_source_faithful_label",
            "yachida_stage_harmonized_label",
            "trajectory_label",
            "trajectory_order",
            "trajectory_eligible",
            "trajectory_exclusion_reason",
            "yachida_stage_recovery_source",
            "yachida_stage_recovery_status",
            "stage_major_from_official_supplement",
            "stage_refined_from_official_supplement",
        ]
    ].copy()
    detailed.to_csv(TABLE_DIR / "crc_stage1_yachida_stage_recovered_per_sample.csv", index=False)
    summary_rows = [
        {
            "source": "Yachida Nature supplementary Table S2-2",
            "question": "Can individual Stage III versus Stage IV be recovered?",
            "answer": "Yes",
            "official_columns": "Subject_ID; Group; Stage",
            "local_samples": len(local),
            "supplement_rows": len(supp),
            "local_sample_overlap": int(merged["Subject_ID"].notna().sum()),
            "unmatched_local_samples": int(merged["Subject_ID"].isna().sum()),
            "duplicated_official_subject_ids": int(supp["Subject_ID"].duplicated().sum()),
            "stage_mismatch_records": len(mismatches),
            "stage_iii_local_paired_subset_n": int(stage_supp.str.startswith("III").sum()),
            "stage_iv_local_paired_subset_n": int(stage_supp.str.startswith("IV").sum()),
            "recommended_use": "Use for Yachida-only secondary stage analyses. These analyses are not externally validated in Sinha.",
        }
    ]
    summary = pd.DataFrame(summary_rows)
    counts = (
        detailed.groupby(["Study.Group", "yachida_stage_source_raw", "yachida_stage_source_faithful_label", "yachida_stage_harmonized_label", "trajectory_eligible"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["Study.Group", "yachida_stage_source_raw"])
    )
    counts.to_csv(TABLE_DIR / "crc_stage1_yachida_recovered_stage_counts.csv", index=False)
    linkage_conflicts = mismatches.copy()
    linkage_conflicts.to_csv(TABLE_DIR / "crc_stage1_yachida_stage_linkage_conflicts.csv", index=False)
    return summary, counts, detailed, linkage_conflicts


def make_task_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def crc_stage2_task_tables(local_metadata: dict[str, pd.DataFrame], yachida_manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mtb = read_tsv(COHORTS["Yachida"] / "mtb.tsv")
    manifest = yachida_manifest.copy()
    manifest["yachida_stage_major_with_healthy"] = np.where(
        manifest["yachida_stage_source_faithful_label"].eq("Healthy"),
        "Healthy",
        manifest["yachida_stage_major"],
    )
    manifest["crc_primary_binary_label"] = np.where(
        manifest["yachida_stage_source_faithful_label"].eq("Healthy"),
        "Control",
        np.where(
            manifest["yachida_stage_harmonized_label"].isin(["Non-invasive CRC", "Localized CRC", "Regional CRC", "Metastatic CRC"]),
            "CRC",
            None,
        ),
    )
    merged = manifest.merge(mtb, on="Sample", how="inner")
    feature_cols = [c for c in mtb.columns if c != "Sample"]
    base_cols = [
        "Sample",
        "Subject",
        "Study.Group",
        "Age",
        "Gender",
        "BMI",
        "Brinkman Index",
        "Alcohol",
        "Tumor location",
        "yachida_stage_source_raw",
        "yachida_stage_normalized",
        "yachida_stage_major",
        "yachida_stage_major_with_healthy",
        "yachida_stage_source_faithful_label",
        "yachida_stage_harmonized_label",
        "trajectory_label",
        "trajectory_order",
        "trajectory_eligible",
    ]
    base_cols = [c for c in base_cols if c in merged.columns]
    tasks = []

    guidance = {
        "Primary Yachida CRC vs Control": ("Primary", "5-fold nested CV", "Primary external-validation task component; technically possible and scientifically reliable enough for primary CRC-vs-control modelling."),
        "Healthy vs MP": ("Exploratory", "Exploratory only", "MP is a precursor-lesion group; useful descriptively but not externally validated in Sinha."),
        "Healthy vs Stage 0": ("Secondary", "Exploratory only", "Recovered official stage supports this Yachida-only contrast, but stage-specific sample size is modest."),
        "Healthy vs Stage I": ("Secondary", "Exploratory only", "Recovered official stage supports this Yachida-only contrast, but it remains internally evaluated only."),
        "Healthy vs Stage II": ("Secondary", "Exploratory only", "Recovered official stage supports this Yachida-only contrast; n=25 disease samples makes formal claims fragile."),
        "Healthy vs Stage III": ("Secondary", "3-fold nested CV", "Recovered official stage supports this Yachida-only contrast; 3-fold nested CV is preferred over 5-fold for reliability."),
        "Healthy vs Stage IV": ("Secondary", "Exploratory only", "Metastatic CRC has only 20 disease samples; use cautiously and do not externally validate in Sinha."),
        "Healthy vs localized CRC": ("Secondary", "Exploratory only", "Coarse recovered-stage grouping; internally evaluated only."),
        "Healthy vs regional CRC": ("Secondary", "3-fold nested CV", "Coarse recovered-stage grouping corresponding to Stage III; internally evaluated only."),
        "Healthy vs metastatic CRC": ("Secondary", "Exploratory only", "Coarse recovered-stage grouping with small metastatic sample size; internally evaluated only."),
        "Fine multiclass source-faithful stage": ("Exploratory", "Exploratory only", "Fine multiclass analysis is underpowered for several substages and internally evaluated only."),
        "Coarse harmonized stage": ("Exploratory", "Exploratory only", "Coarse multiclass analysis is useful for descriptive structure but not a primary model."),
        "Exploratory cross-sectional ordered disease-state variable": ("Exploratory", "Exploratory only", "Ordering represents disease states across independently sampled individuals, not observed longitudinal progression within patients."),
    }

    def technical_cv(minority_n: int) -> str:
        if minority_n >= 25:
            return "5-fold technically possible"
        if minority_n >= 10:
            return "3-fold technically possible"
        return "descriptive only"

    def add_task_row(task_name: str, label_column: str, definition: str, control_definition: str, eligible: pd.DataFrame, save_name: str, counts: pd.Series, note_tail: str) -> None:
        analysis_type, cv_strategy, note = guidance[task_name]
        minority_n = int(counts.min()) if not counts.empty else 0
        tasks.append(
            {
                "task_name": task_name,
                "cohort_scope": "Yachida only" if task_name != "Primary Yachida CRC vs Control" else "Yachida only component of shared primary task",
                "label_column": label_column,
                "positive_or_multiclass_definition": definition,
                "control_definition": control_definition,
                "n_samples": len(eligible),
                "n_controls": int(counts.get("Control", counts.get("Healthy", 0))),
                "n_disease_or_noncontrol": int(len(eligible) - counts.get("Control", counts.get("Healthy", 0))),
                "minority_class_n": minority_n,
                "technical_cv_feasibility": technical_cv(minority_n),
                "recommended_analysis_type": analysis_type,
                "recommended_cv_strategy": cv_strategy,
                "stage2_prep_file": save_name,
                "notes": f"{note} {note_tail}",
            }
        )

    def add_binary(task_name: str, disease_labels: list[str], control_labels: list[str] = ["Healthy"], label_source: str = "yachida_stage_source_faithful_label") -> None:
        eligible = merged[merged[label_source].isin(control_labels + disease_labels)].copy()
        eligible["model_label"] = np.where(eligible[label_source].isin(control_labels), "Control", "Disease")
        save_name = f"crc_stage2_yachida_{make_task_slug(task_name)}_model_prep.csv"
        eligible[base_cols + ["model_label"] + feature_cols].to_csv(STAGE2_DIR / save_name, index=False)
        add_task_row(
            task_name,
            "model_label",
            "; ".join(disease_labels),
            "; ".join(control_labels),
            eligible,
            save_name,
            eligible["model_label"].value_counts(),
            "Raw, non-imputed, non-scaled; all trainable preprocessing must occur inside future CV pipelines.",
        )

    add_binary("Healthy vs MP", ["MP"])
    for label in ["Stage 0", "Stage I", "Stage II", "Stage III", "Stage IV"]:
        add_binary(f"Healthy vs {label}", [label], label_source="yachida_stage_major_with_healthy")
    add_binary("Healthy vs localized CRC", ["Localized CRC"], label_source="yachida_stage_harmonized_label")
    add_binary("Healthy vs regional CRC", ["Regional CRC"], label_source="yachida_stage_harmonized_label")
    add_binary("Healthy vs metastatic CRC", ["Metastatic CRC"], label_source="yachida_stage_harmonized_label")

    for task_name, label_col, levels in [
        ("Fine multiclass source-faithful stage", "yachida_stage_source_faithful_label", ["Healthy", "MP", "Stage 0", "Stage I", "Stage II", "Stage III", "Stage IV"]),
        ("Coarse harmonized stage", "yachida_stage_harmonized_label", ["Healthy", "Precursor lesion", "Non-invasive CRC", "Localized CRC", "Regional CRC", "Metastatic CRC"]),
    ]:
        eligible = merged[merged[label_col].isin(levels)].copy()
        eligible["model_label"] = eligible[label_col]
        save_name = f"crc_stage2_yachida_{make_task_slug(task_name)}_model_prep.csv"
        eligible[base_cols + ["model_label"] + feature_cols].to_csv(STAGE2_DIR / save_name, index=False)
        add_task_row(
            task_name,
            "model_label",
            "; ".join(levels),
            "included as its own class",
            eligible,
            save_name,
            eligible["model_label"].value_counts(),
            f"Class counts: {eligible['model_label'].value_counts().to_dict()}; raw, non-imputed, non-scaled.",
        )

    ordinal = merged[merged["trajectory_eligible"].eq(True)].copy()
    ordinal["model_label"] = ordinal["trajectory_order"]
    save_name = "crc_stage2_yachida_exploratory_ordinal_trajectory_model_prep.csv"
    ordinal[base_cols + ["model_label"] + feature_cols].to_csv(STAGE2_DIR / save_name, index=False)
    vc = ordinal["trajectory_label"].value_counts().sort_index()
    add_task_row(
        "Exploratory cross-sectional ordered disease-state variable",
        "trajectory_order",
        "Healthy=0; MP=1; Stage 0=2; Stage I=3; Stage II=4; Stage III=5; Stage IV=6",
        "Healthy at trajectory order 0; HS excluded",
        ordinal,
        save_name,
        ordinal["trajectory_label"].value_counts(),
        f"Class counts: {vc.to_dict()}; raw, non-imputed, non-scaled.",
    )

    primary = merged[merged["crc_primary_binary_label"].isin(["Control", "CRC"])].copy()
    primary["model_label"] = primary["crc_primary_binary_label"]
    save_name = "crc_stage2_yachida_primary_crc_vs_control_model_prep.csv"
    primary[base_cols + ["model_label"] + feature_cols].to_csv(STAGE2_DIR / save_name, index=False)
    add_task_row(
        "Primary Yachida CRC vs Control",
        "model_label",
        "Stage 0; Stage I; Stage II; Stage III; Stage IV as CRC",
        "Healthy only; MP and HS excluded",
        primary,
        save_name,
        primary["model_label"].value_counts(),
        "Raw, non-imputed, non-scaled; use with Sinha binary only after shared-feature alignment.",
    )

    task_df = pd.DataFrame(tasks)
    task_df.to_csv(TABLE_DIR / "crc_stage2_task_eligibility.csv", index=False)
    count_rows = []
    for csv_path in sorted(STAGE2_DIR.glob("crc_stage2_yachida_*_model_prep.csv")):
        df = pd.read_csv(csv_path, usecols=["model_label"])
        for label, n in df["model_label"].value_counts(dropna=False).items():
            count_rows.append({"stage2_prep_file": csv_path.name, "model_label": label, "n": int(n)})
    counts = pd.DataFrame(count_rows)
    counts.to_csv(TABLE_DIR / "crc_stage2_task_sample_counts.csv", index=False)
    return task_df, counts

def plot_yachida_stage_recovery(yachida_manifest: pd.DataFrame, task_df: pd.DataFrame) -> None:
    if yachida_manifest.empty:
        return
    for col, title, filename in [
        ("yachida_stage_source_faithful_label", "Yachida source-faithful stage labels", "crc_stage1_yachida_detailed_stage_distribution"),
        ("yachida_stage_harmonized_label", "Yachida harmonized CRC progression labels", "crc_stage1_yachida_harmonized_progression_distribution"),
        ("trajectory_label", "Yachida exploratory cross-sectional ordered disease-state counts", "crc_stage1_yachida_trajectory_counts"),
    ]:
        counts = yachida_manifest[col].value_counts().reset_index()
        counts.columns = [col, "n"]
        plt.figure(figsize=(9, 4.8))
        sns.barplot(data=counts, x=col, y="n", color="#4C78A8")
        plt.title(title)
        plt.xlabel("")
        plt.ylabel("Samples")
        plt.xticks(rotation=30, ha="right")
        for i, row in counts.iterrows():
            plt.text(i, row["n"] + 1, str(int(row["n"])), ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        for ext in ["png", "svg"]:
            plt.savefig(FIGURE_DIR / f"{filename}.{ext}", dpi=300)
        plt.close()

    if not task_df.empty:
        plt.figure(figsize=(10, max(5, 0.32 * len(task_df))))
        plot_df = task_df.sort_values("n_samples", ascending=True)
        sns.barplot(data=plot_df, y="task_name", x="n_samples", hue="recommended_analysis_type", dodge=False)
        plt.title("CRC Stage 2 task eligibility from recovered Yachida labels")
        plt.xlabel("Eligible samples")
        plt.ylabel("")
        plt.legend(title="Recommended analysis type", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        for ext in ["png", "svg"]:
            plt.savefig(FIGURE_DIR / f"crc_stage2_task_eligibility.{ext}", dpi=300)
        plt.close()


def run_stage_recovery_tests(yachida_manifest: pd.DataFrame, yachida_conflicts: pd.DataFrame, sinha_decision: pd.DataFrame) -> pd.DataFrame:
    tests = []

    def record(name: str, passed: bool, observed: object, expected: object, note: str = "") -> None:
        tests.append({"test": name, "passed": bool(passed), "observed": observed, "expected": expected, "note": note})

    record("Yachida local samples linked to official S2-2", len(yachida_manifest), 347, 347)
    unmatched = int(yachida_manifest["Subject_ID"].isna().sum())
    record("Yachida unmatched official stage rows", unmatched == 0, unmatched, 0)
    record("Yachida stage conflicts", len(yachida_conflicts), 0, 0)
    record("Yachida Stage III count", int(yachida_manifest["yachida_stage_major"].eq("Stage III").sum()), 34, 34)
    record("Yachida Stage IV count", int(yachida_manifest["yachida_stage_major"].eq("Stage IV").sum()), 20, 20)
    record("Yachida MP not mapped to CRC", "CRC" not in set(yachida_manifest.loc[yachida_manifest["Study.Group"].eq("MP"), "yachida_stage_harmonized_label"]), True, True)
    record("Yachida HS not mapped to Healthy", "Healthy" not in set(yachida_manifest.loc[yachida_manifest["Study.Group"].eq("HS"), "yachida_stage_harmonized_label"]), True, True)
    record("Yachida HS excluded from trajectory", bool((~yachida_manifest.loc[yachida_manifest["Study.Group"].eq("HS"), "trajectory_eligible"]).all()), True, True)
    record("Sinha supports binary CRC vs Control only after source audit", sinha_decision["answer"].iloc[0] == "No", True, True)
    out = pd.DataFrame(tests)
    out.to_csv(TABLE_DIR / "crc_stage1_stage_recovery_tests.csv", index=False)
    if not out["passed"].all():
        failed = out.loc[~out["passed"], "test"].tolist()
        raise AssertionError(f"Stage recovery tests failed: {failed}")
    return out


def write_report(tables: dict[str, pd.DataFrame]) -> None:
    inv = tables["inventory"]
    counts = tables["class_counts"]
    overlap = tables["identifier_overlap"]
    flows = tables["sample_flow"]
    recovery = tables.get("external_metadata_recovery_audit", pd.DataFrame())
    sinha_support = tables.get("sinha_official_supporting_info_audit", pd.DataFrame())
    sinha_source_search = tables.get("sinha_linked_source_stage_search_audit", pd.DataFrame())
    sinha_stage_counts = tables.get("sinha_publication_table1_aggregate_stage_counts", pd.DataFrame())
    sinha_stage_decision = tables.get("sinha_stage_linkage_decision", pd.DataFrame())
    yachida_stage_recovery = tables.get("yachida_stage_recovery_audit", pd.DataFrame())
    yachida_stage_counts = tables.get("yachida_recovered_stage_counts", pd.DataFrame())
    stage2_tasks = tables.get("stage2_task_eligibility", pd.DataFrame())
    stage_tests = tables.get("stage_recovery_tests", pd.DataFrame())
    y_stage = counts[(counts["cohort"].eq("Yachida")) & (counts["label_column"].eq("Study.Group"))]
    s_group = counts[(counts["cohort"].eq("Sinha")) & (counts["label_column"].eq("Study.Group"))]
    shared_unified = int(overlap.loc[overlap["identifier_type"].eq("HMDB_or_KEGG_unified"), "shared_unique_identifiers"].iloc[0])
    shared_hmdb = int(overlap.loc[overlap["identifier_type"].eq("HMDB"), "shared_unique_identifiers"].iloc[0])
    shared_kegg = int(overlap.loc[overlap["identifier_type"].eq("KEGG"), "shared_unique_identifiers"].iloc[0])

    md = f"""# CRC Stage 1 Data Audit and Preprocessing Plan

Generated: {datetime.now().isoformat(timespec='seconds')}

## Scope

This is a preliminary CRC audit only. No nested cross-validation, model fitting, external validation, differential abundance, pathway enrichment, feature importance, imputation, or scaling has been run.

## CRC Cohorts Found

- **Sinha et al., 2016**: fecal microbiota/metabolome CRC case-control cohort. Supplied data contain 131 samples and 131 participants.
- **Yachida et al., 2019**: fecal metagenomic/metabolomic colorectal neoplasia cohort with stage-related labels. Supplied data contain 347 samples and 347 participants.

Both supplied CRC metabolomics datasets are stool/fecal, so tissue source is not a within-CRC mismatch at this stage.

## Source File Inventory

{inv.to_markdown(index=False)}

## Cohort Audit

### Sinha

The metadata use `Study.Group` as a binary disease label. The supplied values are 0 and 1. Based on the paper and metadata, this is interpreted as control and CRC, respectively.

{s_group.to_markdown(index=False)}

Key clinical metadata available in Sinha are age, sex/gender, BMI, race, and hospital indicators. Stage, tumour site, adenoma labels, antibiotic exposure and steroid exposure are not supplied in the local metadata.

External/source-data check: PLOS Supplementary S4 was downloaded and audited. It confirms sample-level binary case status plus age, BMI, sex, race and hospital indicators, but it does **not** contain sample-level antibiotics, steroids, CRC stage, tumour site or CRC molecular/pathology subtypes. The Sinha paper reports aggregate smoking and cancer-stage information in Table 1, but the released sample-level files audited here do not provide those fields per participant.

#### Sinha Cancer-Stage Linkage Check

The Sinha publication Table 1 reports aggregate CRC case stage categories but the official supporting data do not include a participant-level cancer-stage column. Therefore the categories cannot be mapped one-to-one to local `Sample`/`Subject` IDs. These categories must not be translated to TNM/AJCC stage labels without an explicit source.

{sinha_stage_decision.to_markdown(index=False) if not sinha_stage_decision.empty else 'No Sinha stage-linkage decision table available.'}

{sinha_stage_counts.to_markdown(index=False) if not sinha_stage_counts.empty else 'No aggregate Sinha stage information captured.'}

Official Sinha supporting files searched:

{sinha_support.to_markdown(index=False) if not sinha_support.empty else 'No Sinha supporting-file audit available.'}

Linked Sinha source-study and repository search:

{sinha_source_search.to_markdown(index=False) if not sinha_source_search.empty else 'No linked-source stage search audit available.'}

Conclusion: Sinha stage categories support only aggregate descriptive context, not a formal secondary prediction task and not deterministic sample-level sensitivity analysis.

### Yachida

The metadata use `Study.Group` for healthy controls, colorectal neoplasia stage groups, multiple polypoid adenoma, and history-of-surgery controls.

{y_stage.to_markdown(index=False)}

Yachida is the only supplied CRC cohort with stage/subtype-like disease information. Based on the official sample-level data currently available and audited, Sinha supports binary CRC vs Control analyses only. No participant-level stage labels could be deterministically linked to the local samples.

External/source-data check: Nature Supplementary Tables were downloaded and audited. Supplementary Table S2-2 confirms sample-level metabolome clinical fields for 406 metabolome samples: group, stage, age, gender, BMI, Brinkman Index, alcohol and tumour location. The 347 locally supplied Yachida samples are a curated paired subset. No antibiotic or steroid field was found.

#### Yachida Stage III Versus Stage IV Recovery

Official Supplementary Table S2-2 contains individual `Stage` values. All 347 local paired Yachida samples map to this official table by `Sample`/`Subject_ID`, with no stage conflicts. Therefore Stage III and Stage IV can be separated for Yachida-only stage analysis, while keeping Sinha as binary only.

{yachida_stage_recovery.to_markdown(index=False) if not yachida_stage_recovery.empty else 'No Yachida stage-recovery audit available.'}

{yachida_stage_counts.to_markdown(index=False) if not yachida_stage_counts.empty else 'No recovered Yachida stage counts available.'}

Yachida source-faithful labels preserve the official `Stage` field, including Stage III and Stage IV substages. Harmonized labels are used only to create clinically interpretable groups: Healthy, Precursor lesion, Non-invasive CRC, Localized CRC, Regional CRC, Metastatic CRC, Post-surgical control, or Unknown. MP is treated as a precursor-lesion group, not CRC. HS is treated as a post-surgical control group, not a healthy/no-colonoscopy control.

## External Metadata Recovery Audit

{recovery.to_markdown(index=False) if not recovery.empty else 'No external metadata sources were available locally during this run.'}

## Sample and Participant Structure

{flows.to_markdown(index=False)}

Both local CRC tables have one sample per subject, no duplicated sample IDs, and complete sample overlap between metadata and metabolomics.

## Identifier Overlap

{overlap.to_markdown(index=False)}

Exact shared identifiers after cleaning:

- HMDB shared identifiers: **{shared_hmdb}**
- KEGG shared identifiers: **{shared_kegg}**
- Unified HMDB or KEGG shared identifiers: **{shared_unified}**

This is adequate for a direct shared-metabolite binary CRC external-validation design, but feature-space compatibility must still be handled carefully because Sinha values appear already transformed/standardized while Yachida values are raw non-negative abundance-like values with many zeros.

## Valid Prediction Tasks

Recommended primary shared task:

- **CRC vs Control**, using Sinha `Study.Group` 1 vs 0. For Yachida, use Healthy versus CRC, where CRC consists of recovered official Stage 0, Stage I, Stage II, Stage III and Stage IV. MP and HS are excluded.

Recommended secondary Yachida-only tasks:

- **Stage-specific colorectal neoplasia classification**, limited to labels actually present in Yachida.
- **CRC stage group vs control** comparisons such as Stage 0 vs Healthy, Stage I vs Healthy, Stage II vs Healthy, Stage III vs Healthy and Stage IV vs Healthy, if sample sizes remain adequate. Stage III and IV can be recovered for Yachida only from official Supplementary Table S2-2.
- **Adenoma/MP vs Healthy** as an exploratory precursor-lesion task, not directly externally validated in Sinha.
- **Exploratory cross-sectional ordered disease-state variable**, ordered Healthy → MP → Stage 0 → Stage I → Stage II → Stage III → Stage IV. This ordering represents disease states across independently sampled individuals and must not be interpreted as observed longitudinal progression within patients.

Stage 2 task eligibility and raw model-prep files:

{stage2_tasks.to_markdown(index=False) if not stage2_tasks.empty else 'No Stage 2 task eligibility table available.'}

Tasks not recommended for external validation at Stage 1:

- Stage-specific model transfer to Sinha, because Sinha has no stage/subtype labels.
- Tumour-site models, because tumour location is only available in Yachida and only applies to neoplasia cases.
- Medication-exclusion analyses matching IBD antibiotic/steroid exclusions, because those fields are not supplied for either CRC cohort.
- Sinha stage/subtype prediction, because only aggregate publication counts are available and no sample-level stage field exists in official supporting data.
- HS as a standard healthy control, because Yachida defines HS as normal with history of colorectal surgery, which is biologically and clinically distinct from healthy/no remarkable colonoscopy controls.

## Proposed Cohort-Selection Rules

1. Retain samples with non-missing sample ID, participant ID, disease label and metabolomics row.
2. Use one sample per participant. The current local CRC data already satisfy this; if future raw downloads contain repeated samples, select the earliest/pre-treatment/preoperative sample where available.
3. For the Yachida component of the primary binary CRC vs Control task, use Healthy as Control and recovered official Stage 0, Stage I, Stage II, Stage III and Stage IV as CRC. Exclude MP and HS.
4. For Sinha primary binary CRC vs control, use Study.Group 0 as control and 1 as CRC.
5. Do not exclude antibiotic/steroid samples unless the variable is recovered from an official metadata source. Missing medication fields should be reported as unavailable, not treated as unexposed.
6. Do not remove PCA outliers unless an independent study QC flag exists.

## Proposed Harmonized CRC Metadata Schema

| harmonized field | Sinha support | Yachida support | stage-1 handling |
|---|---:|---:|---|
| sample_id | yes: `Sample` | yes: `Sample` | required |
| participant_id | yes: `Subject` | yes: `Subject` | required/grouping ID |
| cohort | derived | derived | required |
| crc_status | yes: `Study.Group` | yes: derived from `Study.Group` | primary label |
| stage | no | yes: `Stage` and `Study.Group` | Yachida-only |
| yachida_stage_source_raw | no | yes: official `Stage` from Supplementary Table S2-2 | preserve raw source field |
| yachida_stage_normalized | no | yes | capitalization normalization only, e.g. Iva/Ivb to IVA/IVB |
| yachida_stage_source_faithful_label | no | yes | Yachida-only secondary; preserves source-level stage/substage |
| yachida_stage_harmonized_label | no | yes | Yachida-only clinical grouping |
| trajectory_order | no | yes | exploratory cross-sectional ordinal analysis; HS excluded |
| adenoma_status | no | yes: MP | Yachida-only exploratory |
| tumour_site | no | yes: `Tumor location` | Yachida-only exploratory/case-only |
| age | yes | yes | shared clinical feature |
| sex | yes: `Gender` | yes: `Gender` | shared clinical feature |
| bmi | yes: `BMI` | yes: `BMI` | shared clinical feature |
| smoking | no | yes: `Brinkman Index` | Yachida-only; not primary cross-cohort |
| alcohol | no | yes: `Alcohol` | Yachida-only; not primary cross-cohort |
| batch/centre | hospital indicators | none obvious | cohort-specific/report as possible batch proxy |
| antibiotic_exposure | no | no | unavailable |
| steroid_exposure | no | no | unavailable |

## Proposed Leakage-Safe Preprocessing

For modelling later, all trainable preprocessing must live inside sklearn pipelines and be fitted using training folds only:

1. missingness filtering, predefined primary threshold >30% feature missingness removed;
2. constant-feature removal;
3. median imputation;
4. transformation if justified for that cohort/feature set;
5. z-score scaling;
6. model fitting.

No imputation, scaling, PCA, feature selection or threshold optimisation should be fitted globally before nested cross-validation.

## Transformation Recommendation

- **Sinha**: values include negatives and are centred around zero, consistent with already transformed/standardized metabolomics values. Do not apply log/log1p to Sinha.
- **Yachida**: values are non-negative, highly zero-inflated and right-skewed. For exploratory plots, log1p is appropriate. For future modelling, evaluate a documented log1p-transform step inside the training-fold pipeline rather than applying it globally.

Zeros should not be recoded as missing without source-study evidence. In Yachida, zeros are abundant and probably represent absent/below-detection abundance-like measurements; they should be treated cautiously, not automatically as NA.

## External Validation Feasibility

Bidirectional CRC vs Control transfer is feasible in principle:

- Sinha -> Yachida: train on Sinha binary CRC/control, externally evaluate on Yachida Healthy vs CRC stage groups.
- Yachida -> Sinha: train on Yachida Healthy vs CRC stage groups, externally evaluate on Sinha binary CRC/control.

Main risks:

- different metabolomics platforms and preprocessing scales;
- Sinha lacks stage/subtype information;
- Yachida includes non-primary MP and HS groups that need explicit exclusion or separate handling;
- only shared identifier-backed metabolite features should be used for direct cross-cohort modelling;
- clinical covariates are partly shared, but smoking/alcohol/tumour location are Yachida-only.

## Potential Leakage Risks To Avoid

- globally filtering features before nested CV;
- globally imputing, transforming or scaling before nested CV;
- selecting the external-validation direction based on external AUC;
- using Yachida stage labels to tune a Sinha binary model;
- treating HS or MP as controls without a clinical rationale;
- treating missing medication metadata as no medication exposure.

## Files Appearing Missing

- Sinha stage/subtype/tumour-site metadata.
- Participant-level antibiotic exposure, recent antibiotic use, corticosteroid/steroid exposure, immunosuppressive medication and treatment-status fields for both CRC cohorts. Absence of these fields is recorded as unavailable, not as non-exposure.
- Sample-level Sinha stage, smoking, tumour-site or molecular/pathology subtype data. The paper provides some aggregate information, but not enough to safely add participant-level labels.
- Technical batch/run variables beyond Sinha hospital indicators and Yachida shared-with-ERAWIJANTARI flag.
- A definitive statement in the local mapping files about whether Yachida zeros are non-detects, true zeros or imputed zeros.

## Exact Next Implementation Step

After approval, create a CRC modelling-preparation notebook/script that constructs two model-ready views without training models:

1. shared binary CRC vs Control external-validation table using only Sinha/Yachida shared metabolites and shared clinical covariates;
2. Yachida-only stage/precursor table preserving recovered official Stage 0, Stage I, Stage II, Stage III, Stage IV and MP where sample size supports it.

## Automated Stage-Recovery Checks

{stage_tests.to_markdown(index=False) if not stage_tests.empty else 'No automated stage-recovery tests were run.'}
"""
    (REPORT_DIR / "crc_stage1_data_audit.md").write_text(md)


def run_stage1_audit() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)

    inventory = make_inventory()
    inventory.to_csv(TABLE_DIR / "crc_stage1_file_inventory.csv", index=False)

    class_counts = []
    metadata_missingness = []
    sample_flows = []
    feature_qcs = []
    maps = {}
    transform_rows = []
    local_metadata = {}

    for cohort, folder in COHORTS.items():
        metadata = read_tsv(folder / "metadata.tsv")
        local_metadata[cohort] = metadata
        mtb = read_tsv(folder / "mtb.tsv")
        mapping = read_tsv(folder / "mtb.map.tsv")
        counts, miss, flow = metadata_audit(cohort, metadata, mtb)
        fq, mp = feature_audit(cohort, mtb, mapping)
        class_counts.append(counts)
        metadata_missingness.append(miss)
        sample_flows.append(flow)
        feature_qcs.append(fq)
        maps[cohort] = mp
        transform_rows.append(transformation_summary(cohort, mtb))
        plot_pca(cohort, metadata, mtb)

    tables = {
        "inventory": inventory,
        "class_counts": pd.concat(class_counts, ignore_index=True),
        "metadata_missingness": pd.concat(metadata_missingness, ignore_index=True),
        "sample_flow": pd.concat(sample_flows, ignore_index=True),
        "feature_qc": pd.concat(feature_qcs, ignore_index=True),
        "feature_mapping_manifest": pd.concat(maps.values(), ignore_index=True),
        "transformation_summary": pd.concat(transform_rows, ignore_index=True),
    }
    overlap, shared = identifier_overlap(maps)
    tables["identifier_overlap"] = overlap
    tables["shared_identifier_manifest"] = shared
    tables["external_metadata_recovery_audit"] = external_metadata_recovery_audit(local_metadata)
    sinha_support, sinha_stage_counts = sinha_official_supporting_info_audit(local_metadata)
    tables["sinha_official_supporting_info_audit"] = sinha_support
    tables["sinha_linked_source_stage_search_audit"] = sinha_linked_source_stage_search_audit(local_metadata)
    tables["sinha_publication_table1_aggregate_stage_counts"] = sinha_stage_counts
    tables["sinha_stage_linkage_decision"] = sinha_stage_linkage_decision()
    yachida_stage_summary, yachida_stage_counts, yachida_stage_manifest, yachida_stage_conflicts = yachida_stage_recovery_audit(local_metadata)
    tables["yachida_stage_recovery_audit"] = yachida_stage_summary
    tables["yachida_recovered_stage_counts"] = yachida_stage_counts
    tables["yachida_stage_recovered_per_sample"] = yachida_stage_manifest
    tables["yachida_stage_linkage_conflicts"] = yachida_stage_conflicts
    if not yachida_stage_manifest.empty:
        stage2_tasks, stage2_counts = crc_stage2_task_tables(local_metadata, yachida_stage_manifest)
        tables["stage2_task_eligibility"] = stage2_tasks
        tables["stage2_task_sample_counts"] = stage2_counts
        plot_yachida_stage_recovery(yachida_stage_manifest, stage2_tasks)
        tables["stage_recovery_tests"] = run_stage_recovery_tests(
            yachida_stage_manifest,
            yachida_stage_conflicts,
            tables["sinha_stage_linkage_decision"],
        )

    for name, df in tables.items():
        if name == "inventory":
            continue
        df.to_csv(TABLE_DIR / f"crc_stage1_{name}.csv", index=False)

    plot_missingness(tables["feature_qc"])
    write_report(tables)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "CRC stage 1 data audit only",
        "no_model_training": True,
        "primary_missingness_threshold_for_future_modeling": PRIMARY_MISSINGNESS_THRESHOLD,
        "source_hashes": {
            str(p.relative_to(ROOT)): file_sha256(p)
            for p in sorted(
                list(RAW_DIR.glob("*/*.tsv"))
                + list(RAW_DIR.glob("*/*.pdf"))
                + list(EXTERNAL_DIR.glob("*"))
            )
            if p.is_file()
        },
        "outputs": sorted(
            str(p.relative_to(OUT_DIR))
            for p in list(TABLE_DIR.glob("crc_stage1_*"))
            + list(TABLE_DIR.glob("crc_stage2_*"))
            + list(STAGE2_DIR.glob("crc_stage2_*"))
            + list(FIGURE_DIR.glob("crc_stage1_*"))
            + list(FIGURE_DIR.glob("crc_stage2_*"))
            + [REPORT_DIR / "crc_stage1_data_audit.md"]
        ),
    }
    (OUT_DIR / "crc_stage1_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))




# -----------------------------------------------------------------------------
# Stage 1 freeze/release utilities. These consume existing Stage 1 outputs.
# -----------------------------------------------------------------------------

PIPELINE_DIR = OUT_DIR
STAGE2_PREP_DIR = STAGE2_DIR
RELEASE_DIR = PIPELINE_DIR / "stage1_release"
VERSION = "1.0"
STATUS = "Frozen"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unavailable"


def package_versions() -> dict[str, str]:
    versions = {}
    for name in ["numpy", "pandas", "matplotlib", "seaborn", "sklearn"]:
        try:
            module_name = "sklearn" if name == "sklearn" else name
            module = __import__(module_name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = "unavailable"
    return versions


def release_read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / name)


def assert_equal(label: str, observed: object, expected: object, rows: list[dict[str, object]]) -> None:
    passed = observed == expected
    rows.append({"check": label, "observed": observed, "expected": expected, "passed": passed})
    if not passed:
        raise AssertionError(f"{label}: observed {observed!r}, expected {expected!r}")


def validate_integrity() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    flow = release_read_csv("crc_stage1_sample_flow.csv")
    counts = release_read_csv("crc_stage1_class_counts.csv")
    overlap = release_read_csv("crc_stage1_identifier_overlap.csv")
    y_stage = release_read_csv("crc_stage1_yachida_recovered_stage_counts.csv")
    stage_tests = release_read_csv("crc_stage1_stage_recovery_tests.csv")
    task = release_read_csv("crc_stage2_task_eligibility.csv")

    def flow_n(cohort: str, item: str) -> int:
        return int(flow.loc[(flow["cohort"].eq(cohort)) & (flow["item"].eq(item)), "n"].iloc[0])

    assert_equal("Sinha metadata samples", flow_n("Sinha", "metadata_samples"), 131, rows)
    assert_equal("Sinha metabolomics samples", flow_n("Sinha", "metabolomics_samples"), 131, rows)
    assert_equal("Sinha unique participants", flow_n("Sinha", "unique_participants"), 131, rows)
    assert_equal("Sinha repeated participants", flow_n("Sinha", "participants_with_repeated_samples"), 0, rows)
    assert_equal("Yachida metadata samples", flow_n("Yachida", "metadata_samples"), 347, rows)
    assert_equal("Yachida metabolomics samples", flow_n("Yachida", "metabolomics_samples"), 347, rows)
    assert_equal("Yachida unique participants", flow_n("Yachida", "unique_participants"), 347, rows)
    assert_equal("Yachida repeated participants", flow_n("Yachida", "participants_with_repeated_samples"), 0, rows)

    def class_n(cohort: str, label_column: str, klass: object) -> int:
        mask = counts["cohort"].eq(cohort) & counts["label_column"].eq(label_column) & counts["class"].astype(str).eq(str(klass))
        return int(counts.loc[mask, "n_samples"].iloc[0])

    assert_equal("Sinha controls", class_n("Sinha", "Study.Group", 0), 89, rows)
    assert_equal("Sinha CRC cases", class_n("Sinha", "Study.Group", 1), 42, rows)
    for klass, expected in {
        "Healthy": 127,
        "MP": 40,
        "HS": 30,
        "Stage_0": 27,
        "Stage_I_II": 69,
        "Stage_III_IV": 54,
    }.items():
        assert_equal(f"Yachida original {klass}", class_n("Yachida", "Study.Group", klass), expected, rows)

    def y_count(label: str) -> int:
        return int(y_stage.loc[y_stage["yachida_stage_source_faithful_label"].eq(label), "n"].sum())

    for label, expected in {
        "Healthy": 127,
        "MP": 40,
        "HS": 30,
        "Stage 0": 27,
        "Stage I": 44,
        "Stage IIA": 24,
        "Stage IIC": 1,
        "Stage IIIA": 8,
        "Stage IIIB": 22,
        "Stage IIIC": 4,
        "Stage IVA": 13,
        "Stage IVB": 7,
    }.items():
        assert_equal(f"Yachida recovered {label}", y_count(label), expected, rows)

    def overlap_n(kind: str) -> int:
        return int(overlap.loc[overlap["identifier_type"].eq(kind), "shared_unique_identifiers"].iloc[0])

    assert_equal("Shared HMDB identifiers", overlap_n("HMDB"), 110, rows)
    assert_equal("Shared KEGG identifiers", overlap_n("KEGG"), 107, rows)
    assert_equal("Shared unified HMDB/KEGG identifiers", overlap_n("HMDB_or_KEGG_unified"), 217, rows)
    assert_equal("All automated stage recovery tests passed", bool(stage_tests["passed"].all()), True, rows)

    def task_row(name: str) -> pd.Series:
        return task.loc[task["task_name"].eq(name)].iloc[0]

    primary = task_row("Primary Yachida CRC vs Control")
    assert_equal("Primary Yachida binary n", int(primary["n_samples"]), 277, rows)
    assert_equal("Primary Yachida control n", int(primary["n_controls"]), 127, rows)
    assert_equal("Primary Yachida CRC n", int(primary["n_disease_or_noncontrol"]), 150, rows)
    assert_equal("Primary analysis strategy frozen", primary["recommended_cv_strategy"], "5-fold nested CV", rows)

    out = pd.DataFrame(rows)
    out.to_csv(PIPELINE_DIR / "crc_stage1_freeze_integrity_checks.csv", index=False)
    return out


def provenance_table() -> pd.DataFrame:
    rows = [
        ("Sample", "Local metadata", "metadata.tsv", "Sample", False, "none", "identifier", True, True, True, "sample identifier", "string", "none expected"),
        ("Subject", "Local metadata", "metadata.tsv", "Subject", False, "none", "participant identifier", True, True, True, "participant/grouping identifier", "string", "none expected"),
        ("Study.Group", "Local metadata", "metadata.tsv", "Study.Group", False, "none", "original disease/group label", True, True, True, "primary binary label source and original Yachida group audit", "categorical", "none expected"),
        ("Age", "Local metadata", "metadata.tsv / S4 / Table S2-2", "Age/age", False, "none", "clinical covariate", True, True, True, "shared clinical metadata", "numeric", "see metadata missingness table"),
        ("Gender", "Local metadata", "metadata.tsv / S4 / Table S2-2", "Gender/sex", False, "none", "clinical covariate", True, True, True, "shared clinical metadata", "categorical", "see metadata missingness table"),
        ("BMI", "Local metadata", "metadata.tsv / S4 / Table S2-2", "BMI/bmi", False, "none", "clinical covariate", True, True, True, "shared clinical metadata", "numeric", "see metadata missingness table"),
        ("race", "Sinha S4", "sinha_s4_covariates.csv", "race", False, "none", "Sinha-only clinical/batch context", False, False, True, "Sinha descriptive context only", "categorical", "see metadata missingness table"),
        ("hosp1/hosp2", "Sinha S4", "sinha_s4_covariates.csv", "hosp1/hosp2", False, "none", "Sinha hospital/batch proxy", False, False, True, "Sinha cohort context; not shared with Yachida", "binary", "see metadata missingness table"),
        ("Brinkman Index", "Yachida Supplementary Table S2-2", "yachida_supplementary_tables_1_15.xlsx", "Brinkman Index", False, "none", "Yachida-only smoking exposure proxy", False, True, True, "Yachida-only exploratory covariate", "numeric", "see metadata missingness table"),
        ("Alcohol", "Yachida Supplementary Table S2-2", "yachida_supplementary_tables_1_15.xlsx", "Alcohol", False, "none", "Yachida-only alcohol covariate", False, True, True, "Yachida-only exploratory covariate", "categorical/numeric as supplied", "see metadata missingness table"),
        ("Tumor location", "Yachida Supplementary Table S2-2", "yachida_supplementary_tables_1_15.xlsx", "Tumor location", False, "none", "tumour site", False, True, True, "Yachida case/stage-context only", "categorical", "missing for controls/non-cases"),
        ("crc_status", "Derived", "metadata.tsv", "Study.Group", True, "Sinha 0/1 to Control/CRC; Yachida Healthy vs recovered CRC states", "primary binary label support", True, False, True, "binary CRC vs Control task", "categorical", "MP/HS excluded where appropriate"),
        ("yachida_stage_source_raw", "Yachida Supplementary Table S2-2", "yachida_supplementary_tables_1_15.xlsx", "Stage", False, "preserved raw", "stage provenance", False, True, True, "source-faithful stage recovery", "categorical", "missing/not applicable for Healthy/MP/HS"),
        ("yachida_stage_normalized", "Derived", "crc_stage1_yachida_stage_recovered_per_sample.csv", "yachida_stage_source_raw", True, "capitalization only", "stage grouping", False, True, True, "consistent recovered-stage labels", "categorical", "missing/not applicable for Healthy/MP/HS"),
        ("yachida_stage_major", "Derived", "crc_stage1_yachida_stage_recovered_per_sample.csv", "yachida_stage_normalized", True, "collapse substages to Stage 0/I/II/III/IV", "secondary stage analyses", False, True, True, "Yachida-only Healthy vs stage contrasts", "categorical", "missing/not applicable for Healthy/MP/HS"),
        ("yachida_stage_substage", "Derived", "crc_stage1_yachida_stage_recovered_per_sample.csv", "yachida_stage_normalized", True, "retain III/IV substages", "descriptive substage audit", False, False, True, "source-faithful detail; not primary modelling", "categorical", "blank outside Stage III/IV"),
        ("yachida_stage_harmonized_label", "Derived", "crc_stage1_yachida_stage_recovered_per_sample.csv", "Study.Group + yachida_stage_major", True, "clinical grouping", "secondary/exploratory stage grouping", False, True, True, "coarse biological interpretation", "categorical", "Unknown if unresolved"),
        ("trajectory_label", "Derived", "crc_stage1_yachida_stage_recovered_per_sample.csv", "source-faithful labels", True, "ordered disease-state label", "exploratory ordered disease-state variable", False, False, True, "cross-sectional ordering only", "categorical", "HS excluded"),
        ("trajectory_order", "Derived", "crc_stage1_yachida_stage_recovered_per_sample.csv", "trajectory_label", True, "Healthy=0 to Stage IV=6", "exploratory ordered disease-state variable", False, False, True, "not longitudinal progression", "numeric ordinal", "missing for HS"),
        ("crc_primary_binary_label", "Derived", "crc_stage2_yachida_primary_crc_vs_control_model_prep.csv", "Study.Group + recovered stages", True, "Healthy=Control; Stage 0/I/II/III/IV=CRC", "primary Yachida binary label", True, False, False, "shared primary CRC vs Control modelling", "categorical", "MP/HS excluded"),
        ("model_label", "Derived", "stage2_model_prep/*.csv", "task-specific label source", True, "task-specific relabelling", "Stage 2 model label", True, True, True, "formal Stage 2 input label", "categorical/numeric", "none expected in task files"),
    ]
    cols = [
        "variable",
        "original_source",
        "original_file",
        "original_column",
        "derived",
        "transformation_applied",
        "used_for",
        "primary_model",
        "secondary_model",
        "exploratory_analysis",
        "reason_for_inclusion",
        "expected_data_type",
        "missing_values",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(PIPELINE_DIR / "CRC_STAGE1_DATA_PROVENANCE.csv", index=False)
    return df


def complete_data_dictionary(provenance: pd.DataFrame) -> pd.DataFrame:
    stage_dict = release_read_csv("crc_stage1_stage_label_data_dictionary.csv")
    metadata_missing = release_read_csv("crc_stage1_metadata_missingness.csv")
    feature_rows = [
        {
            "field": "metabolomics feature columns",
            "definition": "Raw cohort-specific metabolomics abundance/intensity features from mtb.tsv.",
            "allowed_values": "numeric; zeros retained as observed",
            "units": "as supplied by source dataset",
            "source": "Sinha/Yachida mtb.tsv and mtb.map.tsv",
            "intended_use": "future Stage 2 modelling after leakage-safe train-fold preprocessing",
            "missingness_reference": "crc_stage1_feature_qc.csv",
        },
        {
            "field": "HMDB_clean / KEGG_clean",
            "definition": "Cleaned metabolite identifiers used for cross-cohort identifier overlap.",
            "allowed_values": "identifier strings or missing",
            "units": "not applicable",
            "source": "mtb.map.tsv",
            "intended_use": "shared metabolite external-validation feature space",
            "missingness_reference": "crc_stage1_identifier_overlap.csv",
        },
    ]
    prov_rows = []
    for _, row in provenance.iterrows():
        miss = metadata_missing.loc[metadata_missing["field"].eq(row["variable"])]
        missing_desc = "see source table"
        if not miss.empty:
            missing_desc = "; ".join(f"{r.cohort}: {r.n_missing} missing ({r.missing_fraction:.3f})" for r in miss.itertuples())
        prov_rows.append(
            {
                "field": row["variable"],
                "definition": row["used_for"],
                "allowed_values": "see provenance/source file",
                "units": "not applicable unless numeric clinical field",
                "source": f"{row['original_source']} / {row['original_file']} / {row['original_column']}",
                "intended_use": row["reason_for_inclusion"],
                "missingness_reference": missing_desc,
            }
        )
    out = pd.concat([pd.DataFrame(prov_rows), stage_dict.rename(columns={"intended_use": "intended_use"}), pd.DataFrame(feature_rows)], ignore_index=True, sort=False)
    if "units" not in out:
        out["units"] = "not applicable"
    out["units"] = out["units"].fillna("not applicable")
    out["missingness_reference"] = out["missingness_reference"].fillna("see source table")
    out.to_csv(PIPELINE_DIR / "CRC_STAGE1_COMPLETE_DATA_DICTIONARY.csv", index=False)
    return out


def write_stage2_spec() -> None:
    task = release_read_csv("crc_stage2_task_eligibility.csv")
    overlap = release_read_csv("crc_stage1_identifier_overlap.csv")
    y_counts = release_read_csv("crc_stage1_yachida_recovered_stage_counts.csv")
    text = f"""# CRC Stage 2 Input Specification

Version: {VERSION}
Status: {STATUS}

Stage 2 modelling scripts may read only frozen Stage 1 release files or their source-equivalent originals. They must not modify any Stage 1 table, figure, manifest, report or raw prep file.

## Allowed Stage 2 Input Files

- `tables/crc_stage2_task_eligibility.csv`
- `tables/crc_stage2_task_sample_counts.csv`
- `tables/stage2_model_prep/*.csv`
- `tables/crc_stage1_shared_identifier_manifest.csv`
- `tables/crc_stage1_identifier_overlap.csv`
- `tables/crc_stage1_stage_label_data_dictionary.csv`
- `CRC_STAGE1_DATA_PROVENANCE.csv`
- `CRC_STAGE1_COMPLETE_DATA_DICTIONARY.csv`

## Files Stage 2 Must Never Modify

- Raw data under `Colon_Cancer_Data_Set/`
- All files under `stage1_release/`
- All `crc_stage1_*` audit tables
- All `crc_stage2_*` raw prep files generated by Stage 1
- Stage 1 report, release notes, manifest and integrity checks

## Required Identifiers

- `Sample`
- `Subject`
- `Study.Group`
- `model_label` in every Stage 2 task file

## Expected Counts

- Sinha: 131 samples, 131 subjects, 89 controls, 42 CRC cases.
- Yachida: 347 samples, 347 subjects.
- Primary Yachida CRC vs Control task: 277 samples, 127 controls, 150 CRC.

## Expected Recovered Yachida Stage Counts

{y_counts.to_markdown(index=False)}

## Expected Identifier Overlap

{overlap.to_markdown(index=False)}

## Stage 2 Task Contract

{task.to_markdown(index=False)}

## Modelling Boundary

Stage 2 must fit missingness filtering, imputation, transformation, scaling, feature selection, dimensionality reduction, threshold selection and model fitting inside training folds only. External-validation cohorts must remain untouched until final evaluation.
"""
    (PIPELINE_DIR / "CRC_STAGE2_INPUT_SPECIFICATION.md").write_text(text)


def write_release_notes() -> None:
    text = f"""# CRC Stage 1 Release Notes

Version: {VERSION}
Status: {STATUS}

## Summary

CRC Stage 1 is frozen as the reproducible baseline for future modelling. It contains the Sinha and Yachida CRC audits, identifier harmonization, official Yachida stage recovery, Stage 2 raw task tables and integrity checks.

## Major Accomplishments

- Audited Sinha and Yachida local metadata, metabolomics matrices and mapping files.
- Confirmed both CRC cohorts use stool/fecal samples.
- Audited Sinha official supporting information and linked source-study trail.
- Determined that Sinha supports binary CRC vs Control analyses only with currently available sample-level data.
- Recovered individual Yachida Stage III and Stage IV labels from official Supplementary Table S2-2.
- Created Yachida source-faithful, harmonized and exploratory ordered disease-state labels.
- Created raw non-imputed, non-scaled Stage 2 model-preparation tables.

## Known Limitations

- Sinha participant-level cancer stage could not be deterministically linked.
- Antibiotic, steroid/corticosteroid, immunosuppressive medication and treatment-status metadata are unavailable.
- Tumour site is unavailable for Sinha.
- Yachida stage-specific tasks are internally evaluated only.
- The ordered disease-state variable is cross-sectional and not longitudinal progression.
- Stage IV and substage-level analyses have small sample sizes.

## Primary Modelling Task

Binary CRC vs Control. For Yachida, CRC consists of recovered official Stage 0, Stage I, Stage II, Stage III and Stage IV. MP and HS are excluded.

## Secondary Tasks

Yachida-only Healthy vs Stage 0/I/II/III/IV and selected harmonized stage contrasts, interpreted as secondary or exploratory depending on sample size and clinical reliability.

## Future Work

Proceed to Stage 2 modelling using the frozen Stage 1 contract with leakage-safe nested cross-validation.
"""
    (PIPELINE_DIR / "RELEASE_NOTES.md").write_text(text)


def write_methods_and_checklist() -> None:
    methods = """# CRC Stage 1 Methods Summary

Two stool/fecal CRC metabolomics cohorts were audited before model development: Sinha et al. 2016 and Yachida et al. 2019. Local metadata, metabolomics matrices and metabolite mapping files were inventoried, checked for sample/subject duplication, and summarized for clinical labels, missingness and feature identifier overlap. No modelling, imputation, scaling, feature selection, differential abundance or pathway analysis was performed in Stage 1.

Sinha contained 131 one-sample-per-subject observations and supports binary CRC vs Control analysis using the audited `Study.Group` field. Official Sinha supporting files S1-S5, the Figshare record, curated metadata and linked source-study trail were audited for participant-level disease extent, metastatic status, tumour site and medication variables. No deterministic participant-level cancer-stage field could be linked to local sample identifiers, so publication-level stage proportions were retained only as aggregate context.

Yachida contained 347 one-sample-per-subject observations. Original local metadata used coarse `Study.Group` labels including `Stage_I_II` and `Stage_III_IV`. Official Yachida Supplementary Table S2-2 was used to recover individual stage and substage labels. All local Yachida samples linked to the official table and no true stage conflicts were found. Recovered official labels support Yachida-only secondary analyses of Healthy vs Stage 0, Stage I, Stage II, Stage III and Stage IV.

Identifier harmonization used cleaned HMDB and KEGG identifiers from the supplied mapping files. The frozen audit records 110 shared HMDB identifiers, 107 shared KEGG identifiers and 217 unified HMDB/KEGG shared identifiers. These shared identifiers define the future cross-cohort metabolomics feature space for external-validation modelling.

The primary Stage 2 task is binary CRC vs Control. In Yachida, CRC is defined as recovered official Stage 0, Stage I, Stage II, Stage III or Stage IV; MP and HS are excluded. Secondary Yachida-only stage analyses are permitted but are not externally validated in Sinha. The exploratory ordered disease-state variable represents independently sampled disease states and must not be interpreted as longitudinal progression.
"""
    (PIPELINE_DIR / "CRC_STAGE1_METHODS_SUMMARY.md").write_text(methods)

    checklist = """# CRC Stage 2 Modelling Checklist

- Read frozen Stage 1 inputs only.
- Never modify Stage 1 release files.
- Do not preprocess the full dataset before CV.
- Do not scale before CV.
- Do not impute before CV.
- Do not filter missingness before CV except inside training folds.
- Do not remove constant or low-variance features outside CV.
- Do not select features outside CV.
- Do not fit PCA, latent models or other dimensionality reduction outside CV.
- Use nested CV for model development.
- Tune hyperparameters using training-only inner CV.
- Select thresholds using training-only out-of-fold predictions.
- Freeze preprocessing, selected hyperparameters and threshold before external validation.
- Use shared HMDB/KEGG identifier-backed metabolites for cross-cohort external validation.
- Keep Sinha binary CRC vs Control only unless new deterministic participant-level stage metadata are recovered.
- Keep Yachida stage-specific tasks separate from shared external-validation modelling.
- Do not treat MP as CRC in the primary task.
- Do not treat HS as Healthy in the primary task.
- Treat antibiotic/steroid/treatment metadata as unavailable, not as non-exposure.
- Report ROC-AUC, PR-AUC, balanced accuracy, sensitivity, specificity, calibration and confidence intervals where feasible.
- Preserve random seeds, package versions and all model artifacts for reproducibility.
"""
    (PIPELINE_DIR / "CRC_STAGE2_MODELLING_CHECKLIST.md").write_text(checklist)

    appendix = """# CRC Stage 1 Methods Appendix

## Data Sources

Sinha et al. 2016 and Yachida et al. 2019 stool/fecal CRC metabolomics datasets were used. Local metadata, metabolomics and metabolite mapping files were audited together with official Sinha supporting files and Yachida Supplementary Table S2-2.

## Inclusion Criteria

Samples were retained in Stage 1 if metadata and metabolomics identifiers could be linked. Both cohorts had one sample per subject in the local data.

## Exclusion Criteria

No model-time exclusions were applied in Stage 1. For the future primary binary task, Yachida MP and HS are excluded by definition: MP is a precursor-lesion group, while HS is a post-surgical control group.

## Metadata Recovery

Sinha supporting files and linked source resources did not provide deterministic participant-level stage, antibiotic, steroid, treatment or tumour-site fields. Yachida Supplementary Table S2-2 provided stage, age, gender, BMI, Brinkman Index, alcohol and tumour location.

## Stage Recovery

Yachida official stages were recovered from Supplementary Table S2-2. Raw labels were preserved, capitalization was normalized where needed, major stage and harmonized labels were derived, and automated tests confirmed expected counts.

## Harmonization

Metabolite identifier harmonization used cleaned HMDB and KEGG identifiers. Clinical harmonization retained only source-supported fields and documented cohort-specific fields separately.

## Justification For Binary Primary Task

Binary CRC vs Control is the only shared outcome supported by both cohorts. Yachida contributes Healthy vs recovered Stage 0/I/II/III/IV; Sinha contributes binary Control vs CRC.

## Justification For Secondary Tasks

Yachida-only Healthy vs stage analyses are supported by recovered official stage labels but cannot be externally validated in Sinha.

## Justification For Excluding MP

MP is a multiple polypoid adenoma/precursor-lesion group and is not mapped to CRC in the primary task.

## Justification For Excluding HS

HS is a post-surgical control group and is not equivalent to Healthy/no-neoplasia controls.

## Justification For Not Using Antibiotics/Steroids

Participant-level antibiotic and steroid variables were unavailable in the audited official CRC metadata. Missing metadata are not interpreted as non-exposure.

## Justification For Not Assigning Sinha Stages

Sinha reports aggregate disease-extent counts but does not provide deterministic participant-level stage labels linked to local sample identifiers. Individual labels were therefore not reconstructed.
"""
    (PIPELINE_DIR / "CRC_STAGE1_METHODS_APPENDIX.md").write_text(appendix)


def make_provenance_diagram() -> None:
    import matplotlib.pyplot as plt

    labels = [
        "Raw datasets",
        "Metadata audit",
        "Feature audit",
        "Identifier harmonization",
        "Yachida stage recovery",
        "Task eligibility",
        "Frozen Stage 1 outputs",
        "Stage 2 modelling",
    ]
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.axis("off")
    y_positions = np.linspace(0.92, 0.08, len(labels))
    for i, (label, y) in enumerate(zip(labels, y_positions)):
        fc = "#dbeafe" if i < len(labels) - 1 else "#fef3c7"
        ax.text(
            0.5,
            y,
            label,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", fc=fc, ec="#334155", lw=1.2),
        )
        if i < len(labels) - 1:
            ax.annotate("", xy=(0.5, y_positions[i + 1] + 0.045), xytext=(0.5, y - 0.045), arrowprops=dict(arrowstyle="->", lw=1.5, color="#334155"))
    ax.text(0.5, 0.015, "CRC Stage 1 Freeze v1.0: Stage 2 consumes frozen outputs without modifying them.", ha="center", fontsize=9)
    for ext in ["png", "svg"]:
        fig.savefig(PIPELINE_DIR / f"CRC_STAGE1_DATA_PROVENANCE_DIAGRAM.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def release_files() -> list[Path]:
    files = [
        PIPELINE_DIR / "crc_pipeline.py",
        PIPELINE_DIR / "crc_stage1_manifest.json",
        PIPELINE_DIR / "crc_stage1_freeze_integrity_checks.csv",
        PIPELINE_DIR / "CRC_STAGE1_DATA_PROVENANCE.csv",
        PIPELINE_DIR / "CRC_STAGE1_COMPLETE_DATA_DICTIONARY.csv",
        PIPELINE_DIR / "CRC_STAGE2_INPUT_SPECIFICATION.md",
        PIPELINE_DIR / "RELEASE_NOTES.md",
        PIPELINE_DIR / "CRC_STAGE1_METHODS_SUMMARY.md",
        PIPELINE_DIR / "CRC_STAGE2_MODELLING_CHECKLIST.md",
        PIPELINE_DIR / "CRC_STAGE1_METHODS_APPENDIX.md",
        PIPELINE_DIR / "CRC_STAGE1_DATA_PROVENANCE_DIAGRAM.png",
        PIPELINE_DIR / "CRC_STAGE1_DATA_PROVENANCE_DIAGRAM.svg",
        REPORT_DIR / "crc_stage1_data_audit.md",
    ]
    files.extend(sorted(TABLE_DIR.glob("crc_stage1_*.csv")))
    files.extend(sorted(TABLE_DIR.glob("crc_stage2_*.csv")))
    files.extend(sorted(STAGE2_PREP_DIR.glob("*.csv")))
    files.extend(sorted(FIGURE_DIR.glob("crc_stage1_*.*")))
    files.extend(sorted(FIGURE_DIR.glob("crc_stage2_*.*")))
    return [p for p in files if p.exists()]


def write_manifest(files_to_release: list[Path]) -> dict[str, object]:
    input_files = sorted(
        [p for p in RAW_DIR.glob("*/*") if p.is_file()]
        + [p for p in EXTERNAL_DIR.glob("*") if p.is_file()]
    )
    manifest = {
        "project": "CRC Stage 1",
        "version": VERSION,
        "status": STATUS,
        "release_date": datetime.now().isoformat(timespec="seconds"),
        "script_version": f"crc_pipeline.py:{VERSION}",
        "git_commit": git_commit(),
        "random_seed": RANDOM_SEED,
        "primary_missingness_threshold": PRIMARY_MISSINGNESS_THRESHOLD,
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "input_files": [
            {"path": str(p.relative_to(ROOT)), "size_bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in input_files
        ],
        "generated_outputs": [
            {"path": str(p.relative_to(PIPELINE_DIR)), "size_bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in files_to_release
        ],
        "frozen_contract": "Stage 2 may read these outputs but must not modify them.",
    }
    (PIPELINE_DIR / "CRC_STAGE1_RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def copy_release(files_to_release: list[Path]) -> None:
    if RELEASE_DIR.exists():
        shutil.rmtree(RELEASE_DIR)
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    for p in files_to_release + [PIPELINE_DIR / "CRC_STAGE1_RELEASE_MANIFEST.json"]:
        rel = p.relative_to(PIPELINE_DIR)
        dest = RELEASE_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)


def freeze_stage1_release() -> None:
    integrity = validate_integrity()
    provenance = provenance_table()
    complete_data_dictionary(provenance)
    write_stage2_spec()
    write_release_notes()
    write_methods_and_checklist()
    make_provenance_diagram()
    files = release_files()
    manifest = write_manifest(files)
    copy_release(files)
    print(json.dumps({
        "release": f"CRC Stage 1 v{VERSION}",
        "status": STATUS,
        "integrity_checks": int(len(integrity)),
        "files_released": len(files) + 1,
        "release_dir": str(RELEASE_DIR),
        "manifest": str(PIPELINE_DIR / "CRC_STAGE1_RELEASE_MANIFEST.json"),
    }, indent=2))


def _load_stage2_module():
    """Load the CRC Stage 2 implementation without executing it as a script."""
    import importlib.util
    import sys

    path = OUT_DIR / "run_stage2_modelling.py"
    if not path.exists():
        raise FileNotFoundError(
            "CRC Stage 2 implementation is missing. Expected: "
            f"{path}. Restore it before running Stage 2 commands."
        )

    module_name = "crc_stage2_modelling"
    spec = importlib.util.spec_from_file_location(module_name, path)

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import CRC Stage 2 implementation from {path}"
        )

    module = importlib.util.module_from_spec(spec)

    # Required before exec_module so dataclasses and other runtime
    # introspection can resolve the module correctly under Python 3.13.
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module


def run_stage2_prepare_only() -> None:
    stage2 = _load_stage2_module()
    config = stage2.CRCStage2Config()
    stage2.ensure_dirs()
    stage2.prepare_only(config)
    stage2.write_yachida_stage_analysis_plan(config)
    stage2.build_crc_integration_package(config)
    stage2.write_end_manifest(config)


def run_stage2_validate_only() -> None:
    stage2 = _load_stage2_module()
    config = stage2.CRCStage2Config()
    stage2.ensure_dirs()
    stage2.validate_only(config)
    stage2.write_end_manifest(config)


def run_stage2_primary_models() -> None:
    stage2 = _load_stage2_module()
    config = stage2.CRCStage2Config()
    stage2.ensure_dirs()
    stage2.run_primary_modelling(config)
    stage2.write_end_manifest(config)


def run_stage2_secondary_yachida_stage() -> None:
    stage2 = _load_stage2_module()
    config = stage2.CRCStage2Config()
    stage2.ensure_dirs()
    stage2.write_yachida_stage_analysis_plan(config)
    stage2.write_end_manifest(config)


def run_stage2_differential_abundance() -> None:
    stage2 = _load_stage2_module()
    config = stage2.CRCStage2Config()
    stage2.ensure_dirs()
    stage2.run_differential_abundance(config)
    stage2.write_end_manifest(config)


def run_stage2_pathway_enrichment() -> None:
    stage2 = _load_stage2_module()
    config = stage2.CRCStage2Config()
    stage2.ensure_dirs()
    stage2.run_pathway_enrichment(config)
    stage2.write_end_manifest(config)


def run_stage2_postrun_diagnostics() -> None:
    stage2 = _load_stage2_module()
    config = stage2.CRCStage2Config()
    stage2.ensure_dirs()
    stage2.run_crc_postrun_diagnostics(config)
    stage2.write_end_manifest(config)


def run_stage2_reporting() -> None:
    stage2 = _load_stage2_module()
    config = stage2.CRCStage2Config()
    stage2.ensure_dirs()
    stage2.run_stage2_reporting(config)


def run_stage2_build_integration_package() -> None:
    stage2 = _load_stage2_module()
    config = stage2.CRCStage2Config()
    stage2.ensure_dirs()
    stage2.build_crc_integration_package(config)
    stage2.write_end_manifest(config)


def run_stage2_freeze_release() -> None:
    stage2 = _load_stage2_module()
    config = stage2.CRCStage2Config()
    stage2.ensure_dirs()
    stage2.freeze_crc_stage2_release(config)
    stage2.write_end_manifest(config)


def write_crc_command_index() -> None:
    commands = [
        ("audit", "Regenerate CRC Stage 1 audit outputs."),
        ("freeze", "Validate and freeze CRC Stage 1 v1.0 release."),
        ("stage2-prepare", "Validate frozen Stage 1 contract and build Stage 2 manifests/plans only."),
        ("stage2-validate", "Validate Stage 2 inputs/contracts/resources only; fit no models."),
        ("stage2-primary", "Run primary CRC-vs-Control nested CV and bidirectional external validation."),
        ("stage2-yachida-stage", "Write Yachida stage/subtype analysis plan only; no secondary model fitting."),
        ("stage2-da", "Run CRC differential abundance after model-ready tables exist."),
        ("stage2-pathway", "Run pathway-resource/readiness check; ORA remains incomplete unless full annotation cache and DA sets are available."),
        ("stage2-postrun", "Run CRC post-run diagnostics after primary modelling."),
        ("stage2-reporting", "Generate reporting-only CRC Stage 2 tables, figures, reports and release manifest from completed outputs."),
        ("stage2-integration", "Build CRC integration package manifest for later IBD-to-CRC modelling."),
        ("stage2-freeze", "Write CRC Stage 2 freeze-readiness checks and release manifest."),
    ]
    pd.DataFrame(commands, columns=["command", "purpose"]).to_csv(TABLE_DIR / "crc_pipeline_command_index.csv", index=False)





def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="CRC thesis pipeline: Stage 1 audit/freeze plus Stage 2 modelling entry points.")
    parser.add_argument(
        "command",
        nargs="?",
        default="freeze",
        choices=[
            "audit",
            "freeze",
            "stage2-prepare",
            "stage2-validate",
            "stage2-primary",
            "stage2-yachida-stage",
            "stage2-da",
            "stage2-pathway",
            "stage2-postrun",
            "stage2-reporting",
            "stage2-integration",
            "stage2-freeze",
            "commands",
        ],
        help="Run a CRC pipeline stage. Default: freeze.",
    )
    args = parser.parse_args()
    if args.command == "audit":
        run_stage1_audit()
    elif args.command == "freeze":
        freeze_stage1_release()
    elif args.command == "stage2-prepare":
        run_stage2_prepare_only()
    elif args.command == "stage2-validate":
        run_stage2_validate_only()
    elif args.command == "stage2-primary":
        run_stage2_primary_models()
    elif args.command == "stage2-yachida-stage":
        run_stage2_secondary_yachida_stage()
    elif args.command == "stage2-da":
        run_stage2_differential_abundance()
    elif args.command == "stage2-pathway":
        run_stage2_pathway_enrichment()
    elif args.command == "stage2-postrun":
        run_stage2_postrun_diagnostics()
    elif args.command == "stage2-reporting":
        run_stage2_reporting()
    elif args.command == "stage2-integration":
        run_stage2_build_integration_package()
    elif args.command == "stage2-freeze":
        run_stage2_freeze_release()
    elif args.command == "commands":
        write_crc_command_index()


if __name__ == "__main__":
    main()
