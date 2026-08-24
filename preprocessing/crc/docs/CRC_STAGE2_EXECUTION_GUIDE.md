# CRC Stage 2 Execution Guide

Execution should proceed through checkpoints, not as one unreviewed run.

1. Run validate-only.
2. Inspect validation output.
3. Run unit tests.
4. Run fast-test validate-only.
5. Run fast-test primary smoke modelling.
6. Inspect fast-test outputs and logs.
   - Confirm `tables/crc_stage2_clinical_source_inventory.csv`.
   - Confirm `tables/crc_stage2_clinical_linkage_audit.csv`.
   - Confirm `tables/crc_stage2_shared_clinical_feature_contract.csv` includes the audited shared clinical variables.
   - Confirm clinical, metabolomics and integrated contexts run distinctly when shared clinical variables are eligible.
7. Run primary CRC-vs-Control modelling.
8. Run CRC post-run diagnostics.
9. Inspect diagnostic and freeze recommendation.
10. Correct only genuine implementation problems if necessary.
11. Write the Yachida stage/subtype analysis plan.
12. Run differential abundance.
13. Inspect DA outputs.
14. Run the pathway-resource/readiness check; full ORA should proceed only after annotation resources and DA significant sets are valid.
15. Build the CRC integration package.
16. Run final CRC freeze tooling.
17. Proceed later to IBD-to-CRC latent-space integration.

Do not run a one-command all-stage execution before reviewing diagnostics.
