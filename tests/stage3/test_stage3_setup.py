from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

STAGE3 = Path("/Users/bhaktidoshi/Downloads/Thesis Data/thesis_preprocessing_workflow/stage3_ibd_crc_integration")


def test_authoritative_input_manifest_exists_and_paths_are_recorded():
    manifest = pd.read_csv(STAGE3 / "manifests" / "stage3_upstream_input_manifest.csv")
    assert {"disease", "analysis_type", "source_file", "authoritative_status"}.issubset(manifest.columns)
    assert len(manifest) > 0


def test_four_way_overlap_contains_valid_hmdb_only():
    shared = pd.read_csv(STAGE3 / "tables" / "stage3_four_cohort_shared_metabolites.csv")
    assert shared["hmdb_id"].is_unique
    assert shared["hmdb_id"].str.match(r"^HMDB\d+$").all()


def test_no_name_only_matching_policy_is_documented():
    upstream = pd.read_csv(STAGE3 / "manifests" / "stage3_upstream_input_manifest.csv")
    joined = " ".join(upstream["notes"].fillna("").astype(str).tolist())
    assert "name-only" not in joined.lower()


def test_yachida_hs_not_in_ordered_policy():
    policy = (STAGE3 / "reports" / "STAGE3_YACHIDA_STAGE_USAGE_POLICY.md").read_text()
    assert "HS is a post-surgical control group and must be excluded" in policy


def test_dry_run_does_not_execute_heavy_routines():
    result = subprocess.run(
        [sys.executable, str(STAGE3 / "scripts" / "run_stage3.py"), "dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "dry-run only" in result.stdout
