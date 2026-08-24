# CRC Stage 2 Stage and Subtype Policy

Sinha will remain binary only: Control versus CRC. Participant-level stage labels were not deterministically recoverable from public source-linked files. Aggregate publication counts will not be converted into individual sample labels.

Yachida secondary analyses will use the frozen full-stage manifest:

`tables/crc_stage1_yachida_stage_recovered_per_sample.csv`

Stage 2 will preserve parallel fields:

- `stage_source_raw`
- `stage_normalized`
- `stage_major`
- `stage_substage`
- `stage_source_faithful`
- `stage_recovery_source`
- `stage_recovery_status`
- `ajcc_informed_analysis_group`

The `ajcc_informed_analysis_group` field will be described as a study-defined AJCC-informed analytical grouping, not an official AJCC recode, SEER Summary Stage recode, or registry-stage conversion.

Yachida stage models will be internal-only and will not be externally validated in Sinha.
