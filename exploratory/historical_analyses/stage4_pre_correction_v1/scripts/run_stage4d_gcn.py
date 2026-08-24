from __future__ import annotations

import pandas as pd

from stage4_common import REPORTS, TABLES, require_multiomics_ready


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "architecture": "fixed_prior_knowledge_feature_graph_gcn",
        "layers": 2,
        "hidden_dim": 32,
        "dropout": 0.2,
        "optimizer": "Adam",
        "external_validation": "frozen_topology_and_training_derived_weights",
        "status": "blocked_until_stage4a_ready",
    }]).to_csv(TABLES / "stage4d_gcn_architecture_plan.csv", index=False)
    (REPORTS / "STAGE4D_METHODS.md").write_text("""# Stage 4D Methods

The primary GCN uses a fixed prior-knowledge biological feature graph. Each patient provides abundance signals on identical nodes. Node scaling and missing-node handling are learned from training data only. No GAT is included in the primary pipeline.
""")
    require_multiomics_ready("stage4d_gcn")
    raise NotImplementedError("GCN execution is blocked until Stage 4A feasibility is READY.")


if __name__ == "__main__":
    main()
