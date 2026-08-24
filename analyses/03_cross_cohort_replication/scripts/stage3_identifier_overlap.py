from __future__ import annotations

import re
from itertools import combinations

import pandas as pd

from stage3_config import CONFIG, Stage3Config
from stage3_load_inputs import read_csv


def normalize_hmdb(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.startswith("HMDB:"):
        text = text.split(":", 1)[1]
    match = re.search(r"HMDB\d+", text)
    return match.group(0) if match else None


def cohort_hmdb_sets(config: Stage3Config = CONFIG) -> dict[str, set[str]]:
    ibd = read_csv(config.ibd_dir / "tables" / "differential_abundance_standardized_results.csv")
    crc = read_csv(config.crc_stage2_dir / "tables" / "crc_stage2_canonical_metabolite_manifest.csv")
    sets: dict[str, set[str]] = {}
    for cohort, sub in ibd.groupby("cohort"):
        sets[str(cohort)] = set(filter(None, sub["hmdb_id"].map(normalize_hmdb)))
    for cohort, sub in crc.groupby("cohort"):
        valid = sub[sub["valid_hmdb"].astype(bool)]
        sets[str(cohort)] = set(filter(None, valid["HMDB"].map(normalize_hmdb)))
    return sets


def overlap_tables(config: Stage3Config = CONFIG) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sets = cohort_hmdb_sets(config)
    cohorts = ["Franzosa", "iHMP", "Sinha", "Yachida"]
    summary = [{"comparison": c, "n_metabolites": len(sets.get(c, set()))} for c in cohorts]
    for a, b in combinations(cohorts, 2):
        summary.append({"comparison": f"{a} ∩ {b}", "n_metabolites": len(sets.get(a, set()) & sets.get(b, set()))})
    four = set.intersection(*(sets.get(c, set()) for c in cohorts))
    summary.append({"comparison": "Franzosa ∩ iHMP ∩ Sinha ∩ Yachida", "n_metabolites": len(four)})
    matrix_rows = []
    for a in cohorts:
        row = {"cohort": a, "n_metabolites": len(sets.get(a, set()))}
        for b in cohorts:
            row[b] = len(sets.get(a, set()) & sets.get(b, set()))
        matrix_rows.append(row)
    shared = pd.DataFrame({"hmdb_id": sorted(four), "canonical_id": [f"HMDB:{x}" for x in sorted(four)]})
    return pd.DataFrame(summary), pd.DataFrame(matrix_rows), shared

