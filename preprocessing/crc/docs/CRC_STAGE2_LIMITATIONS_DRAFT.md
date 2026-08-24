# CRC Stage 2 Limitations Draft

Expected limitations include:

- Sinha has no deterministic participant-level stage labels.
- CRC medication variables such as antibiotic, steroid, immunosuppressant and treatment status are unavailable in the public participant-level CRC files.
- Cohorts differ in measurement platform, scaling and preprocessing history.
- Shared clinical metadata are now recoverable for age, sex and BMI in both primary cohorts, but these variables still require fold-local imputation/encoding/scaling and should not be interpreted as harmonising all clinical context between studies.
- Race/enrollment hospital are Sinha-specific and Brinkman index, alcohol, tumour location and stage are Yachida-specific; these are not used in the primary shared CRC model.
- Yachida stage analyses are internal-only and cross-sectional.
- Ordered stage analyses will not demonstrate temporal within-patient progression.
- Pathway enrichment will depend on annotation-resource coverage and will be interpreted as over-representation, not pathway activation or causality.
