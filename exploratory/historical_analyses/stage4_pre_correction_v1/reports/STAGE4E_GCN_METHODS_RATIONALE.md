# Stage 4E GCN Methods Rationale

Stage 4E implements a low-complexity homogeneous GCN over the frozen KEGG+NJC19 Stage 4D graph. Each subject is represented as one graph with fixed topology and subject-specific node values.

- Augmented graph hash: `71a0dec685c445e1059875736265e4755fb6cade434f586a567ffe63cf9e6ee1`.
- Edges are fixed prior-knowledge KEGG metabolite-metabolite and NJC19 microbe-metabolite links.
- No patient-patient edges, phenotype-derived edges, cohort correlations, model coefficients, MOFA, sCCA or disease labels are used to define graph topology.
- Node features: fold-locally transformed abundance/value plus a fixed node-type indicator.
- Architecture: 1-2 dense GCN layers, mean graph readout, dropout, binary classifier.
- Class imbalance: weighted BCE loss using training-only class counts.
- External cohorts are used only once after internal model selection/refit.

## Hyperparameter Grid
|   hidden_dim |   layers |   dropout |   weight_decay | graph_mode   |
|-------------:|---------:|----------:|---------------:|:-------------|
|           16 |        1 |       0.2 |         0.0001 | augmented    |
|           16 |        1 |       0.2 |         0.0001 | identity     |
|           16 |        1 |       0.5 |         0.0001 | augmented    |
|           16 |        1 |       0.5 |         0.0001 | identity     |
|           16 |        2 |       0.2 |         0.0001 | augmented    |
|           16 |        2 |       0.2 |         0.0001 | identity     |
|           16 |        2 |       0.5 |         0.0001 | augmented    |
|           16 |        2 |       0.5 |         0.0001 | identity     |
|           32 |        1 |       0.2 |         0.0001 | augmented    |
|           32 |        1 |       0.2 |         0.0001 | identity     |
|           32 |        1 |       0.5 |         0.0001 | augmented    |
|           32 |        1 |       0.5 |         0.0001 | identity     |
|           32 |        2 |       0.2 |         0.0001 | augmented    |
|           32 |        2 |       0.2 |         0.0001 | identity     |
|           32 |        2 |       0.5 |         0.0001 | augmented    |
|           32 |        2 |       0.5 |         0.0001 | identity     |

## Runtime Estimate
Four transfer directions x five outer folds x sixteen hyperparameter settings, plus one selected fold refit per outer fold and four final refits. Maximum 344 fits, each capped at 150 epochs with early stopping. Expected runtime is minutes to low hours on CPU for these small dense graphs; RAM should stay below ~2 GB.
