# Corrected-v2 Stage 4E GCN Preparation Report

Stage 4E is prepared but not fitted by this Codex pass.

Scientific question: does prior-knowledge graph message passing improve within-cohort disease classification over the same participants and node values without biological neighbour exchange?

Primary tasks are pooled IBD versus control and CRC versus control only. CD/UC subtype models are not prepared.

Primary evaluation is within-cohort topology comparison. Reciprocal external transfer is prepared only as an assay/transformation stress test because corrected-v2 Stage 4B v2c found no defensible common-scale metabolomics matrix for primary external metabolome deployment.

## Graph Modes
- identity: computational self-loops only.
- kegg_only: KEGG metabolite-metabolite edges.
- kegg_njc19: KEGG edges plus NJC19 curated production, consumption or transport microbe-metabolite relations.
- conservative_kegg_njc19: sensitivity graph excluding explicitly documented broad/global KEGG-only support while retaining NJC19 relations.

## Prespecified Architecture Table
| architecture_id             | architecture_role   |   layers |   hidden_dim |   dropout |   learning_rate |   weight_decay | readout   |   max_epochs |   patience |   min_epochs_before_early_stopping |
|:----------------------------|:--------------------|---------:|-------------:|----------:|----------------:|---------------:|:----------|-------------:|-----------:|-----------------------------------:|
| sensitivity_two_layer_mean  | sensitivity         |        2 |           16 |       0.2 |           0.001 |         0.0001 | mean      |          150 |         20 |                                 20 |
| sensitivity_one_layer_gated | sensitivity         |        1 |           16 |       0.2 |           0.001 |         0.0001 | gated     |          150 |         20 |                                 20 |

Total graph-mode x architecture configurations in this dry-run scope: 8.

## Dry-Run Summary
| analysis   | direction_id                       | disease   | task           | cohort   |   samples |   positive_n |   negative_n |   microbiome_features |   metabolome_features |   graph_modes |   prespecified_architectures |   seeds |   outer_folds | architecture_hpo_performed   |   estimated_gcn_fits |   estimated_ridge_comparator_outer_fits | graph_hash                                                       |
|:-----------|:-----------------------------------|:----------|:---------------|:---------|----------:|-------------:|-------------:|----------------------:|----------------------:|--------------:|-----------------------------:|--------:|--------------:|:-----------------------------|---------------------:|----------------------------------------:|:-----------------------------------------------------------------|
| within     | IBD_IBD_vs_Control_Franzosa_within | IBD       | IBD_vs_Control | Franzosa |       137 |           81 |           56 |                    50 |                   166 |             4 |                            2 |       5 |             5 | False                        |                  200 |                                       5 | 51d0f96ba698785c7ad43bf41414338c84c90a3aa21d8bcb0a8b7cb566892afd |
| within     | IBD_IBD_vs_Control_iHMP_within     | IBD       | IBD_vs_Control | iHMP     |        91 |           65 |           26 |                    50 |                   166 |             4 |                            2 |       5 |             5 | False                        |                  200 |                                       5 | 51d0f96ba698785c7ad43bf41414338c84c90a3aa21d8bcb0a8b7cb566892afd |
| within     | CRC_CRC_vs_Control_Sinha_within    | CRC       | CRC_vs_Control | Sinha    |       131 |           42 |           89 |                    52 |                   109 |             4 |                            2 |       5 |             5 | False                        |                  200 |                                       5 | 51d0f96ba698785c7ad43bf41414338c84c90a3aa21d8bcb0a8b7cb566892afd |
| within     | CRC_CRC_vs_Control_Yachida_within  | CRC       | CRC_vs_Control | Yachida  |       252 |          125 |          127 |                    52 |                   109 |             4 |                            2 |       5 |             5 | False                        |                  200 |                                       5 | 51d0f96ba698785c7ad43bf41414338c84c90a3aa21d8bcb0a8b7cb566892afd |

## Split Manifest Preview
| analysis   | direction_id                       | disease   | task           | cohort   |     seed |   outer_fold |   outer_train_n |   outer_test_n | outer_train_class_counts   | outer_test_class_counts   | same_fold_manifest_for_all_graph_modes   |
|:-----------|:-----------------------------------|:----------|:---------------|:---------|---------:|-------------:|----------------:|---------------:|:---------------------------|:--------------------------|:-----------------------------------------|
| within     | IBD_IBD_vs_Control_Franzosa_within | IBD       | IBD_vs_Control | Franzosa | 20260813 |            1 |             109 |             28 | {"0": 45, "1": 64}         | {"0": 11, "1": 17}        | True                                     |
| within     | IBD_IBD_vs_Control_Franzosa_within | IBD       | IBD_vs_Control | Franzosa | 20260813 |            2 |             109 |             28 | {"0": 44, "1": 65}         | {"0": 12, "1": 16}        | True                                     |
| within     | IBD_IBD_vs_Control_Franzosa_within | IBD       | IBD_vs_Control | Franzosa | 20260813 |            3 |             110 |             27 | {"0": 45, "1": 65}         | {"0": 11, "1": 16}        | True                                     |
| within     | IBD_IBD_vs_Control_Franzosa_within | IBD       | IBD_vs_Control | Franzosa | 20260813 |            4 |             110 |             27 | {"0": 45, "1": 65}         | {"0": 11, "1": 16}        | True                                     |
| within     | IBD_IBD_vs_Control_Franzosa_within | IBD       | IBD_vs_Control | Franzosa | 20260813 |            5 |             110 |             27 | {"0": 45, "1": 65}         | {"0": 11, "1": 16}        | True                                     |
| within     | IBD_IBD_vs_Control_Franzosa_within | IBD       | IBD_vs_Control | Franzosa | 20260814 |            1 |             109 |             28 | {"0": 45, "1": 64}         | {"0": 11, "1": 17}        | True                                     |
| within     | IBD_IBD_vs_Control_Franzosa_within | IBD       | IBD_vs_Control | Franzosa | 20260814 |            2 |             109 |             28 | {"0": 44, "1": 65}         | {"0": 12, "1": 16}        | True                                     |
| within     | IBD_IBD_vs_Control_Franzosa_within | IBD       | IBD_vs_Control | Franzosa | 20260814 |            3 |             110 |             27 | {"0": 45, "1": 65}         | {"0": 11, "1": 16}        | True                                     |
| within     | IBD_IBD_vs_Control_Franzosa_within | IBD       | IBD_vs_Control | Franzosa | 20260814 |            4 |             110 |             27 | {"0": 45, "1": 65}         | {"0": 11, "1": 16}        | True                                     |
| within     | IBD_IBD_vs_Control_Franzosa_within | IBD       | IBD_vs_Control | Franzosa | 20260814 |            5 |             110 |             27 | {"0": 45, "1": 65}         | {"0": 11, "1": 16}        | True                                     |
| within     | IBD_IBD_vs_Control_Franzosa_within | IBD       | IBD_vs_Control | Franzosa | 20260815 |            1 |             109 |             28 | {"0": 45, "1": 64}         | {"0": 11, "1": 17}        | True                                     |
| within     | IBD_IBD_vs_Control_Franzosa_within | IBD       | IBD_vs_Control | Franzosa | 20260815 |            2 |             109 |             28 | {"0": 44, "1": 65}         | {"0": 12, "1": 16}        | True                                     |

## Validation Tests
| test                                               | status   | notes                                                                                                |
|:---------------------------------------------------|:---------|:-----------------------------------------------------------------------------------------------------|
| stage4d_primary_graph_hash_available               | pass     | 51d0f96ba698785c7ad43bf41414338c84c90a3aa21d8bcb0a8b7cb566892afd                                     |
| stage4d_primary_graph_modes_preserved              | pass     | identity,kegg_njc19,kegg_only                                                                        |
| IBD_node_count                                     | pass     | 216                                                                                                  |
| IBD_microbe_node_count                             | pass     | {'metabolite': 166, 'microbe': 50}                                                                   |
| IBD_metabolite_node_count                          | pass     | {'metabolite': 166, 'microbe': 50}                                                                   |
| IBD_primary_biological_modes_unchanged             | pass     | kegg_njc19,kegg_only                                                                                 |
| IBD_cohort_node_features_match                     | pass     | [{'cohort': 'Franzosa', 'node_feature_match': True}, {'cohort': 'iHMP', 'node_feature_match': True}] |
| CRC_node_count                                     | pass     | 161                                                                                                  |
| CRC_microbe_node_count                             | pass     | {'metabolite': 109, 'microbe': 52}                                                                   |
| CRC_metabolite_node_count                          | pass     | {'metabolite': 109, 'microbe': 52}                                                                   |
| CRC_primary_biological_modes_unchanged             | pass     | kegg_njc19,kegg_only                                                                                 |
| CRC_cohort_node_features_match                     | pass     | [{'cohort': 'Sinha', 'node_feature_match': True}, {'cohort': 'Yachida', 'node_feature_match': True}] |
| conservative_sensitivity_graph_prepared_separately | pass     | 6879                                                                                                 |
| conservative_policy_label_free                     | pass     | (9831, 10)                                                                                           |
| conservative_retains_njc19_edges                   | pass     |                                                                                                      |
| IBD_conservative_node_index_bounds                 | pass     |                                                                                                      |
| IBD_conservative_edge_index_undirected             | pass     |                                                                                                      |
| CRC_conservative_node_index_bounds                 | pass     |                                                                                                      |
| CRC_conservative_edge_index_undirected             | pass     |                                                                                                      |
| predictive_tasks_are_pooled_only                   | pass     | {'IBD': 'IBD_vs_Control', 'CRC': 'CRC_vs_Control'}                                                   |
| no_subtype_primary_tasks                           | pass     |                                                                                                      |
| external_stress_test_not_primary                   | pass     | external mode is prepared but labelled secondary                                                     |
| stage4e_no_model_fitting_in_dry_run                | pass     | dry-run writes validation/configuration only                                                         |

## Manual Commands
```bash
cd "/Users/bhaktidoshi/Downloads/Thesis Data"
python thesis_preprocessing_workflow/stage4_multiomics_integration/corrected_v2/scripts/run_stage4e_gcn_corrected_v2.py --smoke-test --analysis within
python thesis_preprocessing_workflow/stage4_multiomics_integration/corrected_v2/scripts/run_stage4e_gcn_corrected_v2.py --analysis within --architecture primary
python thesis_preprocessing_workflow/stage4_multiomics_integration/corrected_v2/scripts/run_stage4e_gcn_corrected_v2.py --analysis within --architecture sensitivities
python thesis_preprocessing_workflow/stage4_multiomics_integration/corrected_v2/scripts/run_stage4e_gcn_corrected_v2.py --analysis external --architecture primary
python thesis_preprocessing_workflow/stage4_multiomics_integration/corrected_v2/scripts/run_stage4e_gcn_corrected_v2.py --audit-only
```
