# Stage 4 User Execution Plan

Run from `/Users/bhaktidoshi/Downloads/Thesis Data`.

## Step 1: Stage 4B
Command: `/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/scripts/run_stage4b.py`
Expected runtime: currently <1 minute because it will stop if Stage 4A feasibility is false; after data acquisition 1-3 hours.
Expected RAM: 2-8 GB after data acquisition.
Return: logs/stage4b_run.log and tables/stage4b_validation_tests.csv.

STOP FOR REVIEW.

## Step 2: Stage 4C MOFA
Command: `/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/scripts/run_stage4c.py stage4c_mofa`
Expected runtime: blocked now; after data acquisition 30-120 minutes.
Expected RAM: 4-12 GB.
Return: MOFA convergence table and resource log.

## Step 3: Stage 4C sCCA
Command: `/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/scripts/run_stage4c.py stage4c_scca`
Expected runtime: blocked now; after data acquisition 20-90 minutes.
Expected RAM: 4-8 GB.

## Step 4: Stage 4C finalize
Command: `/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/scripts/run_stage4c.py stage4c_finalize`

STOP FOR REVIEW.

## Step 5: Stage 4D graph construction
Command: `/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/scripts/build_stage4d_graph.py`
Expected runtime: <10 minutes if resources are local.

## Step 6: Stage 4D GCN
Command: `/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/scripts/run_stage4d_gcn.py`
Expected runtime: blocked now; after data acquisition 30-180 minutes.
Expected RAM: 4-16 GB.

STOP FOR REVIEW.

## Step 7: Stage 4D finalize
Command: `/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/scripts/finalize_stage4d.py`

## Step 8: Stage 4E synthesis
Command: `/opt/anaconda3/bin/python thesis_preprocessing_workflow/stage4_multiomics_integration/scripts/run_stage4e_finalize.py`
