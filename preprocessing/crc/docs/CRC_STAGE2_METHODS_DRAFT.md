# CRC Stage 2 Methods Draft

The primary CRC task will be CRC versus Control with bidirectional external validation.

Sinha will contribute binary Control/CRC labels only. Yachida Healthy samples will define controls and Stage 0/I/II/III/IV samples will define CRC cases for the primary task; MP and HS will be excluded from the primary externally validated model.

Canonical metabolite features will be derived using namespace-qualified HMDB-first identifiers with KEGG fallback only when HMDB is absent. Ambiguous cross-namespace mappings will be excluded by default unless an approved cross-reference resolves them.

Model development will use leakage-safe sklearn pipelines and nested cross-validation within the development cohort only. External cohorts will not influence preprocessing, tuning, model selection, threshold selection or feature selection.

## Clinical Covariate Recovery

Before Stage 2 model fitting, shared clinical covariates are recovered from source-level metadata rather than relying only on intermediate model-prep tables. Sinha age, sex and BMI are recovered from `Colon_Cancer_Data_Set/Sinha-et-al/metadata.tsv`, with Sinha S4 covariates retained as official verification/fallback metadata. Yachida age, sex and BMI are recovered from the Stage 1 generated copy of official Supplementary Table S2-2, `tables/crc_stage1_yachida_supplementary_table_s2_2_metabolome_clinical.csv`, using deterministic `Sample`/`Subject_ID` linkage.

The final shared clinical contract currently includes `age_years`, `sex` and `bmi` for the primary cross-cohort CRC model. Age and BMI are converted to numeric values only if they fall in plausible ranges, and sex is standardized only when raw coding is unambiguous (`Male/Female` or `M/F`). Imputation, one-hot encoding and scaling remain inside training-fold sklearn pipelines.

Cohort-specific variables are preserved separately but are not used in the primary shared model: Sinha race and enrollment hospital/site; Yachida Brinkman index, alcohol, tumour location and stage. Yachida stage and tumour location are prohibited as primary predictors because they are disease descriptors and would risk leakage.
