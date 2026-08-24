# Stage 4 environment (recorded 2026-08-24)

Captured from the machine that produced the 20 August frozen Stage 4A-4E
outputs and the post-freeze patch. No `conda`/`pip` lock file existed for
Stage 4 specifically before this; the IBD pipeline has its own
`requirements.txt` elsewhere in the repo, but Stage 4 additionally depends on
MOFA (Python + its R backend is not required — mofapy2 is pure Python/numpy)
and the R `PMA` package for sCCA.

## Python

| Package | Version |
|---|---|
| Python | 3.13.5 |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| scikit-learn | 1.6.1 |
| torch | 2.11.0 |
| h5py | 3.12.1 |
| mofapy2 | 0.7.4 |
| matplotlib | 3.10.0 |
| seaborn | 0.13.2 |

Reproduce with:
```bash
python3 -m venv stage4_env
source stage4_env/bin/activate
pip install "numpy==2.4.4" "pandas==3.0.2" "scikit-learn==1.6.1" "torch==2.11.0" \
  "h5py==3.12.1" "mofapy2==0.7.4" "matplotlib==3.10.0" "seaborn==0.13.2" pytest
```

## R (Stage 4C sCCA only)

| Package | Version |
|---|---|
| R | 4.5.1 |
| PMA | 1.2.4 |

## Reproducibility notes

- `THESIS_PROJECT_ROOT` environment variable overrides the project root used
  by `stage4_corrected_v2_common.py` and the test suite; unset, it defaults
  to the path derived from each file's own location, which matches the
  original layout.
- This project directory is not a git repository. Initialising one
  (`git init` at the `Thesis Data` root) is recommended so future patches
  carry real commit-level provenance instead of only file hashes — see
  `corrected_v2/manifests/POST_FREEZE_CODE_PATCH_MANIFEST.json` for why this
  matters.
