# Rank Metabolomics Reference Audit

Authoritative corrected-v2b reference inspected before rank sensitivity implementation.

- TaskSpecs: [TaskSpec(disease='IBD', task='IBD_vs_Control', train_cohort='Franzosa', external_cohort='iHMP', positive_label='IBD', negative_label='Control'), TaskSpec(disease='IBD', task='IBD_vs_Control', train_cohort='iHMP', external_cohort='Franzosa', positive_label='IBD', negative_label='Control'), TaskSpec(disease='CRC', task='CRC_vs_Control', train_cohort='Sinha', external_cohort='Yachida', positive_label='CRC', negative_label='Control/Healthy'), TaskSpec(disease='CRC', task='CRC_vs_Control', train_cohort='Yachida', external_cohort='Sinha', positive_label='CRC', negative_label='Control/Healthy')]
- Source-training missingness rule: fixed `>30%` feature missingness exclusion fitted inside source-training folds.
- Zero-variance rule: constant features removed inside the training partition after the relevant transformation.
- Original imputation rule: feature-wise median fitted on source-training data only.
- Original scaling rule: StandardScaler fitted on source-training data only.
- Ridge C grid: `[0.01, 0.1, 1.0, 10.0]`.
- Outer/inner folds: 5/5, adapted downward only if class counts require it.
- Random seed: 20260813.
- Threshold selection: Youden J on source cross-fitted decision scores; external cohort not used.
- Ranking score column: decision_score for ROC-AUC/PR-AUC; probability retained for Brier/diagnostics.
- Bootstrap: 2000 participant-level stratified resamples.

| disease   | task           | training_cohort   | external_cohort   |   train_n |   external_n |   metabolite_feature_count |   train_positive_n |   train_control_n |   external_positive_n |   external_control_n | participant_order_sha256                                         | feature_order_sha256                                             |
|:----------|:---------------|:------------------|:------------------|----------:|-------------:|---------------------------:|-------------------:|------------------:|----------------------:|---------------------:|:-----------------------------------------------------------------|:-----------------------------------------------------------------|
| IBD       | IBD_vs_Control | Franzosa          | iHMP              |       137 |           91 |                        166 |                 81 |                56 |                    65 |                   26 | 900882e253aa6ab595d07ab966594ef2c245bc06c9a0796b42177e5a301debc3 | 95289dca0c02fae750eca0ecb08fb12f850f07d1bee70d1978d6a5552eaff093 |
| IBD       | IBD_vs_Control | iHMP              | Franzosa          |        91 |          137 |                        166 |                 65 |                26 |                    81 |                   56 | 349df18b9eb1a2ca93bca01b0e98ed52c24789760356f5dfa3f39af9d29fa841 | 95289dca0c02fae750eca0ecb08fb12f850f07d1bee70d1978d6a5552eaff093 |
| CRC       | CRC_vs_Control | Sinha             | Yachida           |       131 |          252 |                        110 |                 42 |                89 |                   125 |                  127 | ec405d5a2ceb8d6f3f71d695733d9ad00591fb3314c18cb12bba270d1e95fa5b | 1b836088ec6f273ceb3caef3d7e412596f56c0ba473b4bfc00198b008678c4b6 |
| CRC       | CRC_vs_Control | Yachida           | Sinha             |       252 |          131 |                        110 |                125 |               127 |                    42 |                   89 | 65b1fc3e679da07341a6b787d3afaacf09609e12d4df96721c743a858a025728 | 1b836088ec6f273ceb3caef3d7e412596f56c0ba473b4bfc00198b008678c4b6 |
