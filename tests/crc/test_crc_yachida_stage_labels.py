from __future__ import annotations

import pandas as pd


def _derive_stage_for_test(study_group, raw_stage):
    source_for_stage = raw_stage if pd.notna(raw_stage) else study_group
    token = str(source_for_stage).strip()
    upper = token.upper().replace("STAGE ", "").replace("STAGE", "").strip()
    mapping = {
        "HEALTHY": "Healthy",
        "MP": "MP",
        "HS": "HS",
        "0": "Stage 0",
        "I": "Stage I",
        "II": "Stage II",
        "IIA": "Stage II",
        "IIB": "Stage II",
        "IIC": "Stage II",
        "III": "Stage III",
        "IIIA": "Stage III",
        "IIIB": "Stage III",
        "IIIC": "Stage III",
        "IV": "Stage IV",
        "IVA": "Stage IV",
        "IVB": "Stage IV",
    }
    return mapping.get(upper, pd.NA)


def test_non_crc_groups_do_not_require_raw_ajcc_stage():
    assert _derive_stage_for_test("Healthy", pd.NA) == "Healthy"
    assert _derive_stage_for_test("MP", pd.NA) == "MP"
    assert _derive_stage_for_test("HS", pd.NA) == "HS"


def test_stage_ii_substages_collapse_to_stage_ii():
    assert _derive_stage_for_test("CRC", "II") == "Stage II"
    assert _derive_stage_for_test("CRC", "IIA") == "Stage II"
    assert _derive_stage_for_test("CRC", "IIB") == "Stage II"
    assert _derive_stage_for_test("CRC", "IIC") == "Stage II"
