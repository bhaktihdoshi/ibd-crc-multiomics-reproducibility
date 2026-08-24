# Final Thesis Figure Audit

Project root: `/Users/bhaktidoshi/Downloads/Thesis Data`

Audit scope: final thesis figure set for the completed IBD, CRC, cross-disease and Stage 4 multi-omics analyses. No modelling, graph construction, thresholding, tuning or frozen numerical results were changed during this pass.

## Current Figure Inventory

The recursive figure inventory was written to:

`thesis_preprocessing_workflow/final_thesis_figures/CURRENT_FIGURE_INVENTORY.csv`

Inventory summary:

| Category | Count |
|---|---:|
| Total image/vector files found | 1135 |
| PNG files | 776 |
| SVG files | 351 |
| PDF figure files | 8 |
| IBD figures and backups | 1013 |
| CRC figures | 102 |
| Stage 4 figures | 19 |
| Stage 3 figures | 1 |
| Duplicate/backup files | 621 |
| Stage-specific historical outputs | 268 |
| Supplement candidates | 201 |
| Obsolete/duplicate outputs | 26 |
| Current Stage 4 main candidates | 19 |

Interpretation: the project contains many valid intermediate and backup figures. These should not be used directly in the final thesis without curation. The proposed final thesis figure set below avoids duplicate panels and keeps the display-item count under 10.

## Duplicate And Obsolete Figure Handling

Do not delete raw or intermediate analysis outputs at this stage. Instead:

- Use `final_thesis_figures/` as the clean final figure export directory.
- Treat postrun backup figures and older IBD/CRC figure variants as archive/supplement only.
- Use the final figure provenance table to trace every displayed panel back to frozen Stage 4 tables.

## Missing Thesis Figures Identified

The completed analysis already contains many single-stage figures, but the final thesis still needed:

1. A single whole-project study-design figure connecting IBD, CRC, cross-cohort validation and Stage 4 multi-omics.
2. A compact Stage 4B external-validation figure focused on transportability rather than every candidate model.
3. A latent-integration figure combining MOFA and sCCA in one interpretable display.
4. A graph-construction and augmentation figure showing why NJC19 was needed.
5. A final GCN graph-ablation figure comparing augmented and identity graph modes against Stage 4B.
6. A figure-provenance manifest tying each panel to source tables and scripts.

## Final Main Figure Set

Target display-item budget: 5 main figures, safely below the requested maximum of 10.

| Display item | File stem | Thesis question answered | Status before user run |
|---:|---|---|---|
| 1 | `figure1_study_design` | What data were used and how were cohorts validated? | NEEDS USER GENERATION + VISUAL QA |
| 2 | `figure2_stage4b_external_prediction` | Do single-omic or early-integrated signatures transport across cohorts? | NEEDS USER GENERATION + VISUAL QA |
| 3 | `figure3_stage4c_latent_integration` | Are shared latent microbiome-metabolome structures detectable? | NEEDS USER GENERATION + VISUAL QA |
| 4 | `figure4_stage4d_graph_augmentation` | Does curated prior knowledge improve graph connectivity? | NEEDS USER GENERATION + VISUAL QA |
| 5 | `figure5_stage4e_gcn_external_validation` | Does graph propagation improve external prediction? | NEEDS USER GENERATION + VISUAL QA |

The generation script will save each figure as PNG, PDF and SVG under:

`thesis_preprocessing_workflow/final_thesis_figures/`

## Supplement Candidates

Recommended supplement groups:

- Stage 4A feature-space QC and sparsity plots.
- Full Stage 4B internal model comparison and all representation-specific metrics.
- Full Stage 4C MOFA factor association/loadings and sCCA feature-overlap plots.
- Stage 4D component-size and degree-distribution diagnostics.
- Stage 4E probability diagnostics and per-direction prediction tables.
- Earlier IBD and CRC preprocessing/EDA figures that are not central to the final Stage 4 thesis story.

## Visual Theme

Use one consistent colour-blind accessible palette:

| Concept | Colour |
|---|---|
| IBD | Blue `#0072B2` |
| CRC | Vermillion `#D55E00` |
| Microbiome | Green `#009E73` |
| Metabolome | Sky blue `#56B4E9` |
| Early integration | Orange `#E69F00` |
| Augmented GCN | Purple-pink `#CC79A7` |
| Identity ablation | Grey `#7F7F7F` |
| MOFA | Violet `#6A3D9A` |
| sCCA | Brown `#A6761D` |

Typography and layout:

- Arial or local sans-serif fallback.
- 300 dpi PNG export.
- Editable PDF/SVG export.
- Panel labels A-D in bold.
- Legends outside crowded axes where possible.
- Dashed ROC-AUC line always means chance discrimination at 0.5.
- Arrows in schematic panels indicate analysis order, not disease causality.

## Axis And Legend Audit

Planned final figures:

- ROC-AUC axes use 0-1 bounds and include a 0.5 chance line.
- Graph augmentation panels use raw counts or percentages with explicit labels.
- Latent integration panels separate MOFA explained variance from sCCA canonical correlation.
- GCN probability-range panel is labelled as a diagnostic, not a performance metric.
- Legends distinguish representation, graph mode and omic view consistently.

Pending visual QA after user generation:

- Check that y-axis labels do not overlap in Figure 2C-D and Figure 5C-D.
- Check that long cohort-direction labels remain readable on PowerPoint slides.
- Check that all legends are visible and not compressed.
- Check that the CRC metabolite count in Figure 1 displays the frozen downstream count of 109.

## Source Provenance Per Panel

The script writes:

`thesis_preprocessing_workflow/final_thesis_figures/FINAL_FIGURE_PANEL_PROVENANCE.csv`

Planned source mapping:

| Figure | Panels | Source tables |
|---|---|---|
| Figure 1 | A-C | `stage4a_final_multiomics_feature_space.csv` |
| Figure 2 | A-D | `stage4b_external_performance.csv`; `stage4b_internal_performance.csv` |
| Figure 3 | A-D | `stage4c_mofa_factor_variance_explained.csv`; `stage4c_scca_component_summary.csv` |
| Figure 4 | A-D | `stage4d_graph_qc_before_after_njc19.csv`; `stage4d_graph_source_coverage_augmented.csv` |
| Figure 5 | A-D | `stage4e_gcn_paired_graph_mode_external_metrics.csv`; `stage4b_external_performance.csv` |

## Figure Legends

Figure legends were written to:

`thesis_preprocessing_workflow/final_thesis_figures/FIGURE_LEGENDS.md`

Each legend is under 200 words and avoids overstating causality, clinical utility or independent replication.

## Resolution And Export

Planned outputs for each final figure:

- `.png` at 300 dpi for slides.
- `.pdf` for thesis submission and vector-safe export.
- `.svg` for editable vector work.

No final PNG/PDF/SVG files were generated during this pass because the active instruction is CODEX WRITES; USER RUNS.

## Unresolved Issues

1. Final visual PASS cannot be claimed until the user runs the generation script and the rendered images are visually inspected.
2. The final main set intentionally excludes detailed IBD-only and CRC-only earlier-stage figures; these should remain supplement candidates.
3. Stage 4E GCN external probabilities are known to show saturation/compression. This is preserved as a diagnostic finding, not corrected away.
4. CRC is not treated as colitis-associated cancer; figure wording should remain “CRC” or “sporadic CRC cohorts” unless a cohort-specific source supports CAC.

## Exact User Commands

Generate final figures:

```bash
cd "/Users/bhaktidoshi/Downloads/Thesis Data"
/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/scripts/generate_final_thesis_figures.py
```

List generated outputs:

```bash
find thesis_preprocessing_workflow/final_thesis_figures -maxdepth 1 -type f | sort
```

Optional syntax check:

```bash
cd "/Users/bhaktidoshi/Downloads/Thesis Data"
/opt/anaconda3/bin/python -m py_compile thesis_preprocessing_workflow/stage4_multiomics_integration/scripts/generate_final_thesis_figures.py
```

## Expected Outputs After User Run

- `figure1_study_design.png/.pdf/.svg`
- `figure2_stage4b_external_prediction.png/.pdf/.svg`
- `figure3_stage4c_latent_integration.png/.pdf/.svg`
- `figure4_stage4d_graph_augmentation.png/.pdf/.svg`
- `figure5_stage4e_gcn_external_validation.png/.pdf/.svg`
- `FINAL_FIGURE_PANEL_PROVENANCE.csv`
- `FINAL_FIGURE_SET_STATUS.csv`

## Final Figure Status

Overall status: `NEEDS FIX`

Meaning: scripts, legends, inventory and audit report are prepared, and syntax compilation passed. Final rendered figures still require user execution and visual inspection before the thesis figure set can be marked `PASS`.
