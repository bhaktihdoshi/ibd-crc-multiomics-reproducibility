# Stage 3 Raw Integration Compatibility

Classification: **C. HIGH_DOMAIN_SHIFT_RISK**

Direct raw patient-level pooling is not recommended at this stage. CRC Stage 2 documented substantial Sinha/Yachida external transportability problems and transformed-feature domain shift. IBD and CRC analyses also use different cohort-specific harmonisation/preprocessing histories.

A cautious future option is **within-cohort standardised patient-level abundance** as a secondary/sensitivity representation, but the primary Stage 3 representation should be disease-level metabolite effect profiles.

Schema audit table: `tables/stage3_raw_integration_schema_audit.csv`.

Key reasons:
- multiple cohorts/platforms with heterogeneous preprocessing histories
- CRC Sinha-to-Yachida integrated model showed extreme external transformed values under frozen training scaler
- iHMP has longitudinal origin but primary IBD modelling used first visit; CRC cohorts are cross-sectional
- direct pooling risks learning cohort/platform rather than disease biology
