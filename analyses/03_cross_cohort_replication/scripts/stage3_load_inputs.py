from __future__ import annotations

from pathlib import Path

import pandas as pd

from stage3_config import CONFIG, Stage3Config


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def authoritative_input_paths(config: Stage3Config = CONFIG) -> dict[str, Path]:
    return {
        "ibd_da_standardized": config.ibd_dir / "tables" / "differential_abundance_standardized_results.csv",
        "ibd_mapping": config.ibd_dir / "tables" / "hmdb_to_kegg_mapping_audit.csv",
        "ibd_stable_metabolites": config.ibd_dir / "tables" / "stable_cross_cohort_metabolites.csv",
        "ibd_final_models": config.ibd_dir / "tables" / "main_text" / "final_model_summary.csv",
        "crc_authoritative_manifest": config.crc_stage2_dir / "reports" / "CRC_STAGE2_AUTHORITATIVE_OUTPUT_MANIFEST.csv",
        "crc_metabolite_manifest": config.crc_stage2_dir / "tables" / "crc_stage2_canonical_metabolite_manifest.csv",
        "crc_da": config.crc_stage2_dir / "tables" / "crc_stage2_differential_abundance_results.csv",
        "crc_final_models": config.crc_stage2_dir / "tables" / "main_text" / "crc_stage2_selected_model_performance_summary.csv",
        "crc_yachida_stage_manifest": config.crc_stage2_dir / "data_processed" / "crc_stage2_yachida_stage_manifest_for_internal_tasks.csv",
    }


def validate_authoritative_inputs(config: Stage3Config = CONFIG) -> pd.DataFrame:
    rows = []
    for name, path in authoritative_input_paths(config).items():
        rows.append({"input_name": name, "path": str(path), "exists": path.exists()})
    return pd.DataFrame(rows)

