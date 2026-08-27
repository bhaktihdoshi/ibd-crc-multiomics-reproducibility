# Existing Pipeline Test Recovery

Status: `PARTIAL_NOT_YET_AUTHORITATIVE`

Frozen scripts were not edited for path recovery. Commands were run with `THESIS_SOURCE_ROOT`, `THESIS_PROJECT_ROOT` and `PYTHONPATH` pointing at the available local trees.

| check_family                   | command                                                              |   returncode | log                                                             | scientific_result   |
|:-------------------------------|:---------------------------------------------------------------------|-------------:|:----------------------------------------------------------------|:--------------------|
| existing_frozen_pipeline_tests | /opt/anaconda3/bin/python -m pytest final_analysis/tests             |            0 | final_analysis/focused_validation_v1/logs/existing_tests.log    | True                |
| existing_final_stage4_qa       | /opt/anaconda3/bin/python final_analysis/code/run_final_stage4_qa.py |            0 | final_analysis/focused_validation_v1/logs/existing_final_qa.log | True                |

A non-zero return code here is logged as a recovery/runtime issue, not as a scientific model failure.
