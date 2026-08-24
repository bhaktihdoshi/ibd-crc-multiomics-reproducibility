# CRC ORA / IBD Method-Parity Audit

Authoritative IBD ORA implementation inspected: `thesis_preprocessing_workflow/ibd_pipeline/run_pipeline.py`.

## IBD method identified

- DA input: `tables/differential_abundance_results.csv`.
- Contrasts: IBD vs control, CD vs control and UC vs control, separately in Franzosa and iHMP.
- Significant metabolite rule: BH-FDR `fdr_q_value <= 0.05`.
- Direction sets: `all_fdr_significant`, `higher_in_disease` and `lower_in_disease`.
- KEGG mapping source: `tables/metabolite_annotation_enrichment.csv`, with local KEGG compound-pathway cache.
- Pathway membership source: `annotation_resources/kegg_compound_pathways.tsv` plus `kegg_pathway_names.tsv`.
- Background: metabolites tested in the relevant cohort/contrast with deterministic KEGG compound mapping.
- Test: one-sided hypergeometric right-tail over-representation test.
- Alternative hypothesis: pathway contains more significant mapped KEGG compounds than expected by chance.
- Minimum pathway size: at least 3 background compounds and at least 2 significant compounds in the pathway.
- Duplicate mapping treatment: duplicate HMDB-KEGG and KEGG-pathway rows are deduplicated before counting.
- Multi-pathway metabolites: retained; a KEGG compound can contribute to each pathway it is formally mapped to.
- Ambiguous identifiers: not admitted into deterministic pathway input.
- Fold enrichment: `(k / n) / (K / N)`.
- Multiple testing: BH-FDR separately within cohort × outcome × direction × annotation source.
- Figure convention: main figure restricted to BH-FDR-significant KEGG pathways; supplementary per-context figures can show context-specific ORA.

No methodological error was identified that would require stopping before CRC ORA.
