# Stage 4A Compositional Preprocessing Policy

Microbial abundance matrices should be treated as compositional unless a source explicitly proves otherwise. Primary policy after missing matrices are acquired: prevalence filtering learned on training folds, zero handling, then CLR or robust CLR depending on zero burden and data scale. CLR/rCLR decisions entering prediction must be learned on training data only.

Ridge and early integration require fold-local filtering, transformation and standardisation. MOFA should receive view-specific transformed matrices with cohort-aware design. sCCA uses transformed matched views with training-only penalty selection. GCN node signals use the same frozen training-derived transformations; topology is not learned from labels.
