# Stage 3 Representation Decision

## Option 1: Raw patient-level abundance

Not recommended as primary. It is highly sensitive to platform/cohort shifts and risks clustering by cohort rather than disease state.

## Option 2: Within-cohort standardised patient-level abundance

Possible secondary/sensitivity representation. More robust than raw values, but still vulnerable to cohort effects and small four-way feature space.

## Option 3: Disease-level metabolite effect profiles

Recommended primary representation. It is more robust to platform differences, interpretable as disease-associated direction/effect, and aligned with thesis-safe claims about cross-sectional disease-state signals rather than patient-level progression.

## Recommendation

Primary: **disease-level metabolite effect profiles**.

Secondary/sensitivity: **within-cohort standardised patient-level abundance**, only if GO criteria are satisfied and cohort confounding diagnostics do not dominate.
