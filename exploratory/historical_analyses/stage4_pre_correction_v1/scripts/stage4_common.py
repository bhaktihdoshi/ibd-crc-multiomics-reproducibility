from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _default_project_root() -> Path:
    # This file lives at <PROJECT>/thesis_preprocessing_workflow/stage4_multiomics_integration/scripts/.
    return Path(__file__).resolve().parents[3]


PROJECT = Path(os.environ["THESIS_PROJECT_ROOT"]) if os.environ.get("THESIS_PROJECT_ROOT") else _default_project_root()
ROOT = PROJECT / "thesis_preprocessing_workflow"
STAGE4 = ROOT / "stage4_multiomics_integration"
TABLES = STAGE4 / "tables"
FIGURES = STAGE4 / "figures"
REPORTS = STAGE4 / "reports"
LOGS = STAGE4 / "logs"
CONFIGS = STAGE4 / "configs"
MANIFESTS = STAGE4 / "manifests"

RANDOM_SEED = 20260813


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_log(stage: str, payload: dict) -> Path:
    LOGS.mkdir(parents=True, exist_ok=True)
    path = LOGS / f"{stage}_run.log"
    payload = {
        "stage": stage,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "platform": platform.platform(),
        "random_seed": RANDOM_SEED,
        **payload,
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path


def resolve_stage4a_final_status() -> dict:
    """Resolve Stage 4A status only from the final frozen design outputs."""
    final_report = REPORTS / "STAGE4A_FINAL_FROZEN_DESIGN.md"
    feature_space = TABLES / "stage4a_final_multiomics_feature_space.csv"
    external_design = TABLES / "stage4a_external_validation_design.csv"
    validation_tests = TABLES / "stage4a_final_freeze_validation_tests.csv"
    required = [final_report, feature_space, external_design, validation_tests]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return {"ready": False, "reason": "missing_final_stage4a_outputs", "missing": missing}

    report_text = final_report.read_text()
    tests = pd.read_csv(validation_tests)
    failed = tests.loc[~tests["status"].eq("pass"), "test"].astype(str).tolist()
    if failed:
        return {"ready": False, "reason": "final_stage4a_validation_failed", "failed_tests": failed, "source": str(validation_tests)}

    fs = pd.read_csv(feature_space)
    ev = pd.read_csv(external_design)
    by_key = {(row["disease"], row["cohort"]): row for _, row in fs.iterrows()}
    eligible = bool(ev["eligible_for_stage4b"].fillna(False).astype(bool).all())
    ibd_go = "STAGE4B_IBD_GO" in report_text
    crc_go = "STAGE4B_CRC_GO" in report_text
    overall = "FOUR_COHORT_MULTIOMICS_READY" if "FOUR_COHORT_MULTIOMICS_READY" in report_text else "UNKNOWN"
    status = {
        "ready": bool(ibd_go and crc_go and eligible),
        "source": str(final_report),
        "validation_source": str(validation_tests),
        "overall_feasibility": overall,
        "stage4b_ibd": "GO" if ibd_go else "NO_GO",
        "stage4b_crc": "GO" if crc_go else "NO_GO",
        "ibd_microbiome_rank": str(by_key[("IBD", "Franzosa")]["microbial_rank"]),
        "crc_microbiome_rank": str(by_key[("CRC", "Sinha")]["microbial_rank"]),
        "ibd_shared_microbial_p": int(by_key[("IBD", "Franzosa")]["shared_microbial_p"]),
        "crc_shared_microbial_p": int(by_key[("CRC", "Sinha")]["shared_microbial_p"]),
        "ibd_shared_metabolite_p": int(by_key[("IBD", "Franzosa")]["shared_metabolite_p"]),
        "crc_shared_metabolite_p": int(by_key[("CRC", "Sinha")]["shared_metabolite_p"]),
        "eligible_external_directions": int(ev["eligible_for_stage4b"].fillna(False).astype(bool).sum()),
    }
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    (MANIFESTS / "stage4a_final_feasibility_status.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def stage4a_feasibility() -> str:
    status = resolve_stage4a_final_status()
    if status.get("ready"):
        return status.get("overall_feasibility", "FOUR_COHORT_MULTIOMICS_READY")
    return str(status.get("reason", "STAGE4A_FINAL_STATUS_NOT_READY"))


def require_multiomics_ready(stage: str) -> None:
    status = resolve_stage4a_final_status()
    if not status.get("ready"):
        write_log(stage, {"status": "blocked", "reason": status})
        raise SystemExit(
            f"{stage} is blocked by final Stage 4A feasibility status: {status}. "
            "Resolve the final frozen Stage 4A validation outputs, then rerun this script."
        )
    print("Stage 4A final feasibility:")
    print(status["overall_feasibility"])
    print("")
    print("IBD:")
    print(f"STAGE4B_IBD_{status['stage4b_ibd']}")
    print("")
    print("CRC:")
    print(f"STAGE4B_CRC_{status['stage4b_crc']}")
    print("")
    print(f"Source: {status['source']}")
    print("Proceeding with Stage 4B user-executed analysis.")


def clr_transform(matrix: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    x = matrix.apply(pd.to_numeric, errors="raise")
    if (x < 0).any().any():
        raise ValueError("CLR input contains negative values.")
    logged = np.log(x + pseudocount)
    return logged.sub(logged.mean(axis=1), axis=0)


def no_global_preprocessing_note() -> str:
    return "All filtering, CLR/rCLR, imputation and scaling must be fitted on training folds only."
