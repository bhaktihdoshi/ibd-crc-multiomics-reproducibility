# Tests

## What each group covers

| Location | Covers | How to run | Expected current result |
|---|---|---|---|
| `final_analysis/tests/test_corrected_v2_integrity.py` | The authoritative Stage 4 pipeline: no genus double-counting, fold-local preprocessing correctness, pooled-task-only scope, decision-score threshold/metrics, sCCA label-freeness and tuning-vs-held-out honesty, graph mode/edge-type validation, ensemble averaging, BH-FDR family sizes, seed-replay reproducibility, authoritative-source freeze-script checks, microbiome missingness semantics, sensitivity-variant preprocessing. | `python3 -m pytest final_analysis/tests/test_corrected_v2_integrity.py -q` | 50/50 passing at time of repository freeze |
| `tests/stage3/` | Stage 3 setup and taxon-mapping expansion | `python3 -m pytest tests/stage3/ -q` | Passing at time of freeze |
| `tests/ibd/` | IBD pipeline diagnostics | `python3 -m pytest tests/ibd/ -q` | Passing at time of freeze |
| `tests/crc/` | CRC KEGG over-representation-analysis outputs | `python3 -m pytest tests/crc/ -q` | Passing at time of freeze |
| `exploratory/historical_analyses/stage4_pre_correction_v1/tests/` | Pre-correction Stage 4 pipeline (historical only — not run as part of verifying the final analysis) | `python3 -m pytest exploratory/historical_analyses/stage4_pre_correction_v1/tests/ -q` | Historical; not maintained against the current pipeline |

## Running everything relevant to the final analysis

```bash
python3 -m pytest final_analysis/tests/ tests/stage3/ tests/ibd/ tests/crc/ -q
```

## Do not rerun heavy modelling to "verify" this repository

The current final state reflects the verified test/QA status already recorded in `final_analysis/reports/final_stage4_qa_report.json` at the time the analysis was frozen. Running the test suite (fast, no model training beyond small logistic-regression refits) is sufficient to confirm the code is intact; it does not require rerunning the full GCN/MOFA/sCCA fitting pipeline.
