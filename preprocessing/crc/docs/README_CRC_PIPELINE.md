# CRC Pipeline

This directory contains the CRC analysis stream for the dissertation project.

Stage 1 is frozen and must remain read-only. Stage 2 will consume the frozen Stage 1 contract to build CRC-vs-Control predictive models, perform bidirectional Sinha/Yachida external validation, run post-run diagnostics, prepare Yachida-only secondary stage analyses, and export CRC outputs for later IBD-to-CRC integration.

The single user-facing CRC entry point is:

`crc_pipeline.py`

The Stage 2 implementation is imported by this entry point and should not modify Stage 1 outputs.
