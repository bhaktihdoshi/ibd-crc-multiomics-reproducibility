# Stage 4A Sinha Genus Aggregation Policy

The Sinha microbial table contains a taxonomic tree with parent and child nodes in the same matrix. To avoid double-counting, Stage 4A does not sum child taxa into parent genera while also retaining explicit parent measurements.

Explicit genus-level nodes available: 92.

Policy: use explicit `g__` genus nodes as the genus representation. Do not aggregate terminal taxa into genus unless parent genus rows are absent and a future method explicitly drops child/parent duplicates before modelling.
