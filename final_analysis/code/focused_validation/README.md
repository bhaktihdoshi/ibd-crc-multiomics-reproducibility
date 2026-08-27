# Focused Validation Addendum

This folder contains reproducible code for the additive focused-validation bundle used after the frozen primary analysis was completed.

## Purpose

The focused-validation bundle supports thesis integration only. It does not replace the original frozen primary analyses and does not retrospectively become a prespecified primary analysis.

Included stages:

- complete predictive reporting for frozen external validation results
- paired H3 contrasts
- rank-metabolomics sensitivity
- MOFA sensitivity
- replicated-metabolite influence analysis
- sCCA bootstrap stability
- nested outer sCCA
- final thesis display and release-audit packaging

Graph and GCN analyses remain exploratory extended analyses outside the focused dissertation narrative.

## Entrypoint

Use `run_all.py` as the main entrypoint. Stage names are passed with `--stage`, for example:

```bash
python final_analysis/code/focused_validation/run_all.py --stage qa,manifest --resume
```

Long-running stages include `mofa_sensitivity`, `rank_metabolomics`, `metabolite_robustness`, `scca_stability`, and `scca_nested`. The `latent_figures`, `qa`, `manifest`, and release-packaging steps are reporting/packaging stages and read completed outputs.

## Inputs

Private/raw cohort data are not distributed in this public repository. To rerun the focused-validation code, obtain the original datasets separately and point the scripts to the completed local thesis preprocessing workflow with:

```bash
export THESIS_SOURCE_ROOT=/path/to/local/thesis_preprocessing_workflow
export THESIS_PROJECT_ROOT=/path/to/local/project-root
```

The scripts otherwise use repository-relative paths where possible.

## Resume Behaviour

Most stages support `--resume` and reuse completed outputs when their expected manifests and files are present. This prevents accidentally rerunning expensive stages when preparing reports.

## Outputs

Focused-validation outputs are written under:

```text
final_analysis/focused_validation_v1/
```

Public thesis-facing outputs include selected tables, reports, and figures. Local provenance, logs, model files, checkpoints, and private-path audits are intentionally not part of the public release set.

## Privacy Boundary

Original participant identifiers, participant-level clinical metadata, raw assay matrices, private absolute paths, credentials, and local runtime caches must not be included in public focused-validation outputs. Use the release audit generated under `final_analysis/focused_validation_v1/reports/` before staging files for GitHub.
