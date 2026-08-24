# Stage 4A Final Design Recommendation

1. Microbial data found: Sinha S3 microbial table only.
2. Exact matched n: Sinha 131; Franzosa, iHMP and Yachida 0 locally.
3. iHMP visit rule: earliest paired microbiome-metabolome visit, compatible with first-visit/baseline population.
4. Chosen IBD microbial resolution: not frozen, missing cross-cohort data.
5. Chosen CRC microbial resolution: not frozen, Yachida microbiome missing locally.
6. Transformation: prevalence filter + zero handling + CLR/rCLR, learned fold-locally.
7. Shared microbial p: 0 for cross-cohort pairs with current local files.
8. Matched metabolomic p: 56 strict Stage 3 shared metabolites for cross-disease context; disease-specific model matrices remain frozen upstream.
9. Ridge feasibility: not defensible cross-cohort multi-omics with current local microbiome availability.
10. sCCA feasibility: blocked pending matched two-view matrices.
11. MOFA feasibility: blocked pending matched two-view matrices in both cohorts.
12. Recommended MOFA design: separate per-cohort MOFA with reproducibility, or group-aware MOFA if supported after acquisition.
13. Graph feasibility: topology can be designed, but GCN execution requires matched matrices.
14. GCN formulation: fixed prior-knowledge heterogeneous feature graph.
15. External transfer feasibility: not currently feasible for multi-omics.
16. Final feasibility class: MULTIOMICS_NOT_DEFENSIBLE_WITH_AVAILABLE_DATA.
