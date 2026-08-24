# Software Environment

## Stage 4 (multi-omics integration)

See `docs/software_environment_stage4.md` for the exact package versions used (Python, numpy, pandas, scikit-learn, torch, h5py, mofapy2, matplotlib, seaborn, R, PMA).

## IBD and CRC pipelines

See `preprocessing/ibd/` and `preprocessing/crc/` for pipeline-specific `requirements.txt` files where present (copied from the original pipeline directories).

## General requirements

A top-level `requirements.txt` at the repository root captures the common Python dependencies needed to run the majority of the analysis code. Where a specific stage needs an additional package (e.g. `torch` for the GCN, `mofapy2` for MOFA, R + the `PMA` package for sCCA), that is noted in the relevant stage's own documentation.

## R environment (Stage 4C sCCA only)

R 4.5.1, package `PMA` version 1.2.4 (or compatible). No other R dependencies are required elsewhere in the pipeline.
