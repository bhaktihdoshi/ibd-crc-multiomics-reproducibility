# Stage 4D AGORA2 Methods Rationale

AGORA2/VMH is prepared only as an external prior source for future MICROBE_METABOLITE edges. No graph construction, flux optimisation, FBA, GCN/GAT or predictive modelling is performed here.

## Official Sources
- AGORA2 version 2.01: https://www.vmh.life/files/reconstructions/AGORA2/version2.01/
- Individual SBML reconstructions: https://www.vmh.life/files/reconstructions/AGORA2/version2.01/sbml_files/individual_reconstructions/
- Reference: Heinken et al., Nature Biotechnology 2023, doi:10.1038/s41587-022-01628-0.

## Edge Biology Rule For Future Execution
An eligible edge means an AGORA2 reconstruction contains an exchange or extracellular transport capability for a Stage 4 metabolite. Intracellular metabolites alone are not eligible. No FBA is required for presence/absence extraction.

## Parser Plan
For each targeted SBML file, parse reaction IDs and species/metabolite identifiers. Retain exchange reactions and extracellular transport reactions only, then map VMH/KEGG/HMDB identifiers to Stage 4 HMDB IDs using audited identifier crosswalks. Collapse species/strain support to genus-metabolite candidate edges with binary primary weight 1.0.

## Current Coverage
| space   |   stage4_genera |   genera_represented_in_AGORA2 |   expected_taxonomic_coverage_percent |   matched_model_files |   estimated_download_size_mb |   estimated_extracted_size_mb | parser_runtime_estimate                 | ram_estimate                                             |
|:--------|----------------:|-------------------------------:|--------------------------------------:|----------------------:|-----------------------------:|------------------------------:|:----------------------------------------|:---------------------------------------------------------|
| IBD     |              50 |                             50 |                              100      |                  3734 |                      60480.8 |                       60480.8 | minutes for targeted SBML files; no FBA | single-file streaming XML parser should stay under ~1 GB |
| CRC     |              52 |                             50 |                               96.1538 |                  3778 |                      61047.6 |                       61047.6 | minutes for targeted SBML files; no FBA | single-file streaming XML parser should stay under ~1 GB |
| Union   |              64 |                             62 |                               96.875  |                  3877 |                      61945.1 |                       61945.1 | minutes for targeted SBML files; no FBA | single-file streaming XML parser should stay under ~1 GB |

Matched model files currently listed: 3877. If zero, fetch the AGORA2 directory index metadata before interpreting coverage.
