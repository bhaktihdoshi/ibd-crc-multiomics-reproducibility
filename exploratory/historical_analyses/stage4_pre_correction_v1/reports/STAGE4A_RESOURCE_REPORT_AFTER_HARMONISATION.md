# Stage 4A Resource Report

Downloaded/audited source-file footprint recorded in the manifest: 107.58 MB.
Peak RAM was not instrumented in this lightweight acquisition script; the largest parsed object was the Yachida supplementary workbook with species/genus abundance sheets.
Runtime for the final local acquisition refresh was approximately 1-2 minutes on the local workstation, dominated by Excel parsing.

Unresolved raw-sequence requirement: iHMP processed merged taxonomic abundance was not acquired from public routes tested here. Full recovery would require either a working official IBDMDB merged table endpoint or per-sample MGX product/raw reconstruction from HMP2/IBDMDB accessions. That was not performed because raw/per-sample reconstruction may exceed the Stage 4A resource gate and requires explicit approval.

Recommended future processing if approved: obtain official HMP2 WGS MetaPhlAn/Taxonomic profile merged table or batch-download per-sample processed WMS taxonomic profiles, verify against `hmp2_metadata_2018-08-20.csv`, then match same participant and visit to the frozen iHMP metabolomics samples.
