"""Final reproducibility bundle: one JSON pointing at every provenance
artifact produced across this patch, plus a fresh timestamp/hash snapshot.

This does NOT recompute anything — it is packaging, not analysis. It reads
and re-references (with existence checks and current sha256 where relevant):
  - the 20-August Stage 4 authority freeze and its authoritative-files hash registry
  - the post-freeze code patch manifest
  - the co-primary analysis-role addendum
  - the latest final_stage4_qa_report.json (QA result)
  - the CRC microbiome sensitivity run manifest + external performance (sensitivity result)
  - the post-freeze equivalence check result
  - the Stage 2/3 manuscript-claim provenance map
  - the Stage 4 environment/version record
  - Python/package versions actually running right now

No git repository exists in this project directory (confirmed at bundle-
generation time below), so there are no commit hashes to record; this is
flagged explicitly rather than silently omitted.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from stage4_corrected_v2_common import MANIFESTS, PROJECT, REPORTS, TABLES, atomic_json, sha256


STAGE3_MANIFESTS = PROJECT / "thesis_preprocessing_workflow" / "stage3_ibd_crc_integration" / "manifests"
STAGE4_ENV_FILE = PROJECT / "thesis_preprocessing_workflow" / "stage4_multiomics_integration" / "corrected_v2" / "STAGE4_ENVIRONMENT.md"

REFERENCED_ARTIFACTS = {
    "stage4_authority_freeze_manifest": MANIFESTS / "FINAL_STAGE4_AUTHORITY_MANIFEST.json",
    "stage4_authoritative_files_hash_registry": TABLES / "final_stage4_authoritative_files.csv",
    "post_freeze_code_patch_manifest": MANIFESTS / "POST_FREEZE_CODE_PATCH_MANIFEST.json",
    "post_freeze_analysis_role_addendum": MANIFESTS / "POST_FREEZE_ANALYSIS_ROLE_ADDENDUM.json",
    "post_freeze_claim_policy_addendum": MANIFESTS / "post_freeze_claim_policy_addendum.csv",
    "final_qa_report": REPORTS / "final_stage4_qa_report.json",
    "crc_microbiome_sensitivity_run_manifest": MANIFESTS / "sensitivity_crc_microbiome_run_manifest.json",
    "crc_microbiome_sensitivity_external_performance": TABLES / "sensitivity_crc_microbiome_external_performance.csv",
    "crc_microbiome_sensitivity_replication_check": TABLES / "sensitivity_crc_microbiome_primary_replication_check.csv",
    "crc_microbiome_sensitivity_paired_comparisons": TABLES / "sensitivity_crc_microbiome_paired_comparisons_bh_fdr.csv",
    "post_freeze_equivalence_check": MANIFESTS / "stage4_post_freeze_equivalence_check.json",
    "stage2_3_manuscript_claim_provenance": STAGE3_MANIFESTS / "stage2_3_manuscript_claim_provenance.json",
    "stage4_environment_record": STAGE4_ENV_FILE,
}


def is_git_repo(path: Path) -> bool:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True)
    return result.returncode == 0 and result.stdout.strip() == "true"


def load_json_if_exists(path: Path):
    if path.exists() and path.suffix == ".json":
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
    return None


def main() -> None:
    artifacts = {}
    missing = []
    for name, path in REFERENCED_ARTIFACTS.items():
        entry = {"path": str(path), "exists": path.exists()}
        if path.exists():
            entry["sha256"] = sha256(path)
            entry["mtime_utc"] = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        else:
            missing.append(name)
        artifacts[name] = entry

    qa = load_json_if_exists(REFERENCED_ARTIFACTS["final_qa_report"])
    equivalence = load_json_if_exists(REFERENCED_ARTIFACTS["post_freeze_equivalence_check"])
    sensitivity_manifest = load_json_if_exists(REFERENCED_ARTIFACTS["crc_microbiome_sensitivity_run_manifest"])
    stage2_3 = load_json_if_exists(REFERENCED_ARTIFACTS["stage2_3_manuscript_claim_provenance"])

    bundle = {
        "status": "FINAL_REPRODUCIBILITY_BUNDLE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "version_control": {
            "is_git_repository": is_git_repo(PROJECT),
            "note": "No git repository detected at PROJECT root; commit-level provenance is unavailable. Only file sha256 hashes and timestamps are recorded below. Recommend `git init` before final submission (see POST_FREEZE_CODE_PATCH_MANIFEST.json).",
        },
        "summary_verdicts": {
            "final_qa_overall_status": qa.get("overall_status") if qa else "NOT_FOUND — run run_final_stage4_qa.py",
            "post_freeze_equivalence_overall_status": equivalence.get("overall_status") if equivalence else "NOT_FOUND — run run_stage4_post_freeze_equivalence_check.py",
            "crc_microbiome_sensitivity_status": sensitivity_manifest.get("status") if sensitivity_manifest else "NOT_FOUND — run run_stage4b_crc_microbiome_sensitivity.py",
            "stage2_3_provenance_n_claims": stage2_3.get("n_claims") if stage2_3 else "NOT_FOUND — run generate_stage2_3_manuscript_claim_provenance.py",
        },
        "artifacts": artifacts,
        "missing_artifacts": missing,
    }
    out_path = MANIFESTS / "FINAL_REPRODUCIBILITY_BUNDLE.json"
    atomic_json(bundle, out_path)
    print(f"Wrote {out_path}")
    print(json.dumps(bundle["summary_verdicts"], indent=2))
    if missing:
        print(f"\nWARNING: {len(missing)} referenced artifact(s) not found: {missing}")


if __name__ == "__main__":
    main()
