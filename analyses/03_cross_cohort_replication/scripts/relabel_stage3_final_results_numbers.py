"""Write a relabelled copy of stage3_final_results_numbers.csv.

The existing `tables/stage3_final_results_numbers.csv` has a row labelled
"IBD replicated events" with value 10. That is easy to misread as "10
IBD-vs-Control replicated metabolites" — it is actually IBD-vs-Control (7)
+ CD-vs-Control (3) replication events summed across two different
contrasts. The primary IBD-vs-Control statement remains 7 (the "unique IBD
replicated IDs" row).

`run_stage3.py`'s row-generating source has already been corrected for
future full pipeline reruns. This script does NOT run the Stage 3 pipeline
(stage3_final_results_numbers.csv is not hash-tracked/frozen the way Stage
4's outputs are, so a contained relabel is low-risk, but there is no need to
invoke the whole pipeline just to fix one label) — it reads the existing
CSV, renames only that one ambiguous row's label, and writes a
`_relabelled` copy for citation in drafting. The original file is left
untouched.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CFG_SCRIPTS = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(CFG_SCRIPTS))
from stage3_config import Stage3Config  # noqa: E402


CFG = Stage3Config()
SRC = CFG.stage3_dir / "tables" / "stage3_final_results_numbers.csv"
OUT = CFG.stage3_dir / "tables" / "stage3_final_results_numbers_relabelled.csv"

RELABEL = {
    "IBD replicated events": "IBD-vs-Control + CD-vs-Control replicated events (summed across contrasts; NOT a metabolite count)",
    "unique IBD replicated IDs": "unique IBD-vs-Control replicated IDs (primary IBD-vs-Control statement)",
}


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    df = pd.read_csv(SRC)
    df["result"] = df["result"].replace(RELABEL)
    tmp = OUT.with_suffix(OUT.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(OUT)
    print(f"Wrote {OUT}")
    print(df[df["result"].isin(RELABEL.values())].to_string(index=False))


if __name__ == "__main__":
    main()
