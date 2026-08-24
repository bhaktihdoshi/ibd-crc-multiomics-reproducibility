# Stage 4A iHMP Recovery Escalation Report

## Search Result
- Processed merged matrix found: no single valid merged matrix was recovered from the previously attempted endpoints.
- Processed per-sample profiles found and downloaded: 91/91 official IBDMDB/Globus HMP2 MGX Taxonomic Profiles 3.0 files.
- Raw exact files found: raw reconstruction was retained only as a contingency route; direct raw-tar reconstruction is not needed for Stage 4A because processed profiles were recovered.

## Identifier Resolution
- Frozen iHMP n: 91.
- Exact same-visit and same external-ID metagenomics metadata matches: 91.
- Exact same-visit distinct aliquot/specimen matches: 0.
- Ambiguous same-visit metagenomics rows requiring review: 0.
- Unmatched same-visit metagenomics metadata: 0.

## Processed Profile Recovery
- Valid per-sample MetaPhlAn 3.0 profiles: 91/91.
- Unfiltered matrix: `/Users/bhaktidoshi/Downloads/Thesis Data/thesis_preprocessing_workflow/stage4_multiomics_integration/data/processed/ihmp_microbiome_unfiltered.csv` with 91 samples x 585 taxa.
- Canonical genus/species matrix: `/Users/bhaktidoshi/Downloads/Thesis Data/thesis_preprocessing_workflow/stage4_multiomics_integration/data/processed/ihmp_microbiome_canonical_unfiltered.csv` with 91 samples x 466 taxa.
- Source: official IBDMDB HMP2 2018.18 Metagenomics products page and Globus-hosted Taxonomic Profiles 3.0 TSV files.

## Raw Reconstruction Result
- Raw reconstruction-ready sample IDs retained for contingency only: 91.
- Total download GB: not computable from local metadata without resolving raw file sizes; expected to be substantial and therefore not downloaded.
- Decompressed FASTQ/disk/runtime/RAM: not instrumented because heavy reconstruction was not started. Expect tens to hundreds of GB depending on exact FASTQ availability and paired-end layout.
- Scripts written:
  - `scripts/prepare_ihmp_raw_metagenomics.py`
  - `scripts/run_ihmp_metaphlan.sh`
  - `scripts/assemble_ihmp_metaphlan_profiles.py`

## Publication-Level Limitation
Franzosa profiles were already processed in the publication-specific workflow. If iHMP is reconstructed now with a modern MetaPhlAn/database version, the output will be reproducible but not historically identical to the original HMP2 processed table. That version mismatch must be treated as an assay-processing limitation.

## Final Outcome
IHMP_PROCESSED_MICROBIOME_RECOVERED

## Stage 4 Status
IBD readiness: STAGE4B_IBD_READY_FOR_SUPERVISOR_REVIEW.
CRC readiness: STAGE4B_CRC_GO after supervisor review.
Overall feasibility class: D remains current for completed data; IBD can be revisited after user-executed iHMP reconstruction.

## User Command If Raw Reconstruction Is Approved
From `/Users/bhaktidoshi/Downloads/Thesis Data`:

```bash
/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/scripts/prepare_ihmp_raw_metagenomics.py
bash thesis_preprocessing_workflow/stage4_multiomics_integration/scripts/run_ihmp_metaphlan.sh
/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/scripts/assemble_ihmp_metaphlan_profiles.py
```
