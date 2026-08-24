# References and Data Sources

This repository does not include literature-review or source-paper PDFs (see repository exclusion policy). This file lists the publications/data sources the analysis draws on by title, so a reader can locate them independently.

## Cohort source publications

- Franzosa, E. A., et al. "Gut microbiome structure and metabolic activity in inflammatory bowel disease." *Nature Microbiology* (2019).
- Integrative Human Microbiome Project (iHMP) / IBDMDB. Multi-omics data on inflammatory bowel disease. Data portal: https://ibdmdb.org/
- Sinha, R., et al. Colorectal cancer 16S rRNA microbiome cohort (see original publication for exact citation).
- Yachida, S., et al. "Metagenomic and metabolomic analyses reveal distinct stage-specific phenotypes of the gut microbiota in colorectal cancer." *Nature Medicine* (2019).

## Reference databases

- MiMeDB (Metabolomics Innovation Centre). Metabolite reference database used in Stage 4D prior-knowledge graph construction: https://mimedb.org/
- KEGG (Kyoto Encyclopedia of Genes and Genomes) pathway database, used for Stage 4D graph edges.
- NJC19 curated microbe–metabolite production/consumption/transport relations, used for Stage 4D graph edges.
- HMDB (Human Metabolome Database), used for metabolite identifier harmonisation across cohorts.

## Methodological references

The analysis uses standard methods (CLR transformation for compositional microbiome data, MOFA for multi-omics factor analysis, sparse CCA via the PMA R package, graph convolutional networks) — see the thesis document itself for full methodological citations once available (`thesis/`).

## Note on completeness

This list is not a full bibliography — it identifies the data sources and reference databases the code in this repository directly consumes. For the complete literature review, see the thesis document.
