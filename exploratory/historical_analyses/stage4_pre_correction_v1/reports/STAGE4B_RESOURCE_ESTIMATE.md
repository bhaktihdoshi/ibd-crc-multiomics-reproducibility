# Stage 4B Resource Estimate

Task/directions: 8; representations: 3; C values: 4.
inner candidates 8x3x5x5x4 + outer refits 8x3x5 + final inner candidates 8x3x5x4 + final refits 8x3 = 3024.
Stage 4A simple estimate retained for planning: 2424.
Expected runtime on a MacBook Air: low hours is plausible; final runtime depends on disk speed and sklearn BLAS threads.
Peak RAM: expected <8 GB for dense ridge/logistic matrices at the frozen n/p dimensions.
Disk: expected <1 GB for CSV outputs/predictions/logs unless optional model serialization is later enabled.
Dominant runtime component: repeated inner-CV ridge/logistic fits for early-integration representations.
