# CRC Stage 1 Release Notes

Version: 1.0
Status: Frozen

## Summary

CRC Stage 1 is frozen as the reproducible baseline for future modelling. It contains the Sinha and Yachida CRC audits, identifier harmonization, official Yachida stage recovery, Stage 2 raw task tables and integrity checks.

## Major Accomplishments

- Audited Sinha and Yachida local metadata, metabolomics matrices and mapping files.
- Confirmed both CRC cohorts use stool/fecal samples.
- Audited Sinha official supporting information and linked source-study trail.
- Determined that Sinha supports binary CRC vs Control analyses only with currently available sample-level data.
- Recovered individual Yachida Stage III and Stage IV labels from official Supplementary Table S2-2.
- Created Yachida source-faithful, harmonized and exploratory ordered disease-state labels.
- Created raw non-imputed, non-scaled Stage 2 model-preparation tables.

## Known Limitations

- Sinha participant-level cancer stage could not be deterministically linked.
- Antibiotic, steroid/corticosteroid, immunosuppressive medication and treatment-status metadata are unavailable.
- Tumour site is unavailable for Sinha.
- Yachida stage-specific tasks are internally evaluated only.
- The ordered disease-state variable is cross-sectional and not longitudinal progression.
- Stage IV and substage-level analyses have small sample sizes.

## Primary Modelling Task

Binary CRC vs Control. For Yachida, CRC consists of recovered official Stage 0, Stage I, Stage II, Stage III and Stage IV. MP and HS are excluded.

## Secondary Tasks

Yachida-only Healthy vs Stage 0/I/II/III/IV and selected harmonized stage contrasts, interpreted as secondary or exploratory depending on sample size and clinical reliability.

## Future Work

Proceed to Stage 2 modelling using the frozen Stage 1 contract with leakage-safe nested cross-validation.
