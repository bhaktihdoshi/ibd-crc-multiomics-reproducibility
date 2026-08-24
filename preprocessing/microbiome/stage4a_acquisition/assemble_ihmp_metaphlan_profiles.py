from __future__ import annotations

from pathlib import Path
import pandas as pd

PROJECT = Path("/Users/bhaktidoshi/Downloads/Thesis Data")
STAGE4 = PROJECT / "thesis_preprocessing_workflow/stage4_multiomics_integration"
profile_dir = STAGE4 / "data/external_microbiome/ihmp/metaphlan_profiles"
out = STAGE4 / "data/processed/ihmp_microbiome_canonical_unfiltered.csv"

profiles = sorted(profile_dir.glob("*_metaphlan_profile.tsv"))
if not profiles:
    raise SystemExit("No MetaPhlAn profiles found. Run the user-executed raw reconstruction first.")

series = []
for path in profiles:
    sample = path.name.replace("_metaphlan_profile.tsv", "")
    df = pd.read_csv(path, sep="\t", comment="#", header=None)
    if df.shape[1] < 3:
        continue
    taxa = df.iloc[:, 0].astype(str)
    abundance = pd.to_numeric(df.iloc[:, -1], errors="coerce")
    s = pd.Series(abundance.values, index=taxa.values, name=sample)
    series.append(s)
matrix = pd.concat(series, axis=1).T.fillna(0.0).reset_index(names="sample_id")
matrix.to_csv(out, index=False)
print(f"Wrote {matrix.shape[0]} samples x {matrix.shape[1]-1} taxa to {out}")
