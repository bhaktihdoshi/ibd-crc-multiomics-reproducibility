# MOFA Primary Configuration Audit

Status: `PARTIAL_NOT_YET_AUTHORITATIVE` for the focused-validation branch.

Authoritative implementation inspected: `final_analysis/code/run_stage4c_corrected_v2.py`.

- Primary K: `8`
- Primary seed: `20260813` / corrected-v2 `RANDOM_SEED`
- Likelihoods: Gaussian for metabolome and microbiome views
- MOFA grouping: cohort/study group, not diagnosis
- `scale_views=False`; `scale_groups=False`
- `ard_factors=True`
- Spike-and-slab: not explicitly enabled in the authoritative call
- Training: `iter=1000`, `convergence_mode='medium'`
- Informative factor: non-zero two-view variance; low information if total mean R2 <= 0
- Shared multi-view factor: each view contributes at least 20% of mean two-view R2 in the primary rule
- Phenotypes are post-hoc annotations only and are absent from MOFA fitting matrices.

## Input Dimensions

| disease   | cohort   | view       |   participants |   features |
|:----------|:---------|:-----------|---------------:|-----------:|
| IBD       | Franzosa | metabolome |            137 |        166 |
| IBD       | Franzosa | microbiome |            137 |         50 |
| IBD       | iHMP     | metabolome |             91 |        166 |
| IBD       | iHMP     | microbiome |             91 |         50 |
| CRC       | Sinha    | metabolome |            131 |        109 |
| CRC       | Sinha    | microbiome |            131 |         52 |
| CRC       | Yachida  | metabolome |            252 |        109 |
| CRC       | Yachida  | microbiome |            252 |         52 |

## Input Hashes

| disease   | mofa_long_input_sha256                                           |
|:----------|:-----------------------------------------------------------------|
| IBD       | c602f4cbc928202dc4a24f7780d6210b98b97cd4176a2662149c10afff580792 |
| CRC       | 5769af27bd91e8d38279c30b6582c2a0856ff68aaa1aa808e3e8076502acbfdc |
