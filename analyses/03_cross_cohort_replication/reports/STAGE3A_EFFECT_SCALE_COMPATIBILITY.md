# Stage 3A Effect-Scale Compatibility

## Beta Meaning

Franzosa and iHMP IBD effects are from robust HC3 univariate differential-abundance models using the established formula `metabolite ~ disease_status + age_years`. Positive beta means higher metabolite abundance/effect in IBD, CD, or UC relative to control. The analysed metabolite values are source-supplied processed scales; the Stage 3A matrix uses the standardized beta exported by the expanded DA.

Sinha and Yachida CRC effects are from robust HC3 models using `metabolite_z ~ crc_status + age_years`. Positive beta means higher standardized metabolite abundance/effect in CRC relative to control/healthy.

## Compatibility Decision

A. Beta signs are comparable across cohorts because every contrast uses disease/case coded as the positive exposure and control/healthy as the reference.

B. Raw beta magnitudes are not perfectly comparable because cohorts differ in assay platform, preprocessing, scaling and biological sampling context. IBD and CRC also came from separate study systems.

C. Stage 3A therefore treats sign and rank/Spearman correlation as primary. Pearson/raw beta correlation is reported as a secondary descriptive sensitivity only, not as the main inference.

SE values describe uncertainty for the cohort-specific regression coefficient in its own model and are not pooled across cohorts.
