# IBD-only externally validated pipeline

Run from the Thesis Data folder:

```bash
python3 thesis_preprocessing_workflow/ibd_pipeline/run_pipeline.py
```

The pipeline uses only supplied Franzosa and iHMP sample IDs, repairs Franzosa metabolite access from the published supplement without adding samples, selects first eligible iHMP visits for the primary analysis, and runs bidirectional external validation.
