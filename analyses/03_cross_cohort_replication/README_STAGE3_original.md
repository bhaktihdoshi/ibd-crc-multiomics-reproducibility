# Stage 3 IBD-CRC Integration Workspace

This workspace scaffolds Stage 3 / Phase 3 of the thesis: cross-disease integration of frozen IBD and CRC metabolomics results.

Purpose: evaluate whether stool metabolite signals are shared, disease-specific, directionally concordant/discordant, reproducible across independent cohorts, and consistent with a cross-sectional IBD-CRC disease-state framework.

Frozen upstream dependencies:
- IBD final outputs under `thesis_preprocessing_workflow/ibd_pipeline/`
- CRC Stage 2 frozen outputs under `thesis_preprocessing_workflow/crc_pipeline/stage2/`

Executed in this setup pass:
- directory scaffold
- authoritative upstream manifest
- lightweight HMDB identifier overlap audit
- annotation availability audit from existing mappings only
- schema-level raw integration risk assessment
- design/specification documents

Not executed:
- PCA/UMAP/t-SNE
- latent-space fitting
- graph construction
- clustering
- trajectory inference
- cross-disease machine learning
- nested CV/model fitting
- DA/pathway refitting
- bootstrap analysis
