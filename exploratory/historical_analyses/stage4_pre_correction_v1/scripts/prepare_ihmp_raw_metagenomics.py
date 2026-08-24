from __future__ import annotations

from pathlib import Path
import pandas as pd

PROJECT = Path("/Users/bhaktidoshi/Downloads/Thesis Data")
STAGE4 = PROJECT / "thesis_preprocessing_workflow/stage4_multiomics_integration"
manifest = pd.read_csv(STAGE4 / "manifests/stage4a_ihmp_raw_reconstruction_manifest.csv")
eligible = manifest[manifest["raw_reconstruction_status"].eq("READY_FOR_USER_RECONSTRUCTION")].copy()
out = STAGE4 / "manifests/ihmp_metaphlan_samples_to_run.txt"
out.write_text("\n".join(eligible["microbiome_sample_id"].astype(str)) + "\n")
print(f"Wrote {len(eligible)} sample IDs to {out}")
