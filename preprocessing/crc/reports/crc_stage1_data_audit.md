# CRC Stage 1 Data Audit and Preprocessing Plan

Generated: documentation refinement on existing Stage 1 outputs. No audit rerun, modelling, imputation, scaling, feature selection, differential abundance, pathway analysis or validation was performed during this pass.

## Scope

This document summarizes the completed CRC Stage 1 audit and harmonized Stage 2 planning. The audit preserves the existing Sinha audit, Yachida audit, identifier harmonization, official Yachida Stage III/IV recovery, validation checks and raw Stage 2 model-preparation tables.

## Terminology Rule

Two stage systems are kept separate throughout the CRC workflow:

- **Original local Yachida metadata** uses `Study.Group` labels: `Healthy`, `MP`, `HS`, `Stage_0`, `Stage_I_II`, and `Stage_III_IV`.
- **Recovered official Yachida metadata** uses individual stage labels from Supplementary Table S2-2: Stage 0, Stage I, Stage II, Stage III and Stage IV, with substages where available.

Original local labels are used only when discussing the supplied local metadata. Recovered official labels are used for Stage 2 Yachida-only stage analyses.

## Cohorts

- **Sinha et al., 2016**: fecal microbiota/metabolome CRC case-control cohort with 131 local samples and 131 participants.
- **Yachida et al., 2019**: fecal metagenomic/metabolomic colorectal neoplasia cohort with 347 local samples and 347 participants.

Both local CRC metabolomics datasets are stool/fecal, so tissue source is not a within-CRC mismatch at this stage.

## Sinha Audit

Sinha `Study.Group` is interpreted as binary Control versus CRC.

| cohort   | label_column   |   class |   n_samples |   n_participants |
|:---------|:---------------|--------:|------------:|-----------------:|
| Sinha    | Study.Group    |       0 |          89 |               89 |
| Sinha    | Study.Group    |       1 |          42 |               42 |

Based on the official sample-level data currently available and audited, Sinha supports binary CRC vs Control analyses only. No participant-level stage labels could be deterministically linked to the local samples.

| question                                                                     | answer   |   exact_stage_column_name | available_outcome_column   | available_outcome_coding   | local_id_match                                                    |   unmatched_records |   duplicated_records |   conflicting_records | stage_counts_among_crc_cases                                                                                          | permitted_use                      | not_permitted_use                                                                                    |
|:-----------------------------------------------------------------------------|:---------|--------------------------:|:---------------------------|:---------------------------|:------------------------------------------------------------------|--------------------:|---------------------:|----------------------:|:----------------------------------------------------------------------------------------------------------------------|:-----------------------------------|:-----------------------------------------------------------------------------------------------------|
| Does participant-level Sinha cancer stage exist in official supporting data? | No       |                       nan | case                       | 0=control; 1=CRC case      | S4 ID maps one-to-one to local Sample/Subject for 131/131 records |                   0 |                    0 |                     0 | Publication Table 1 aggregate only: Non-invasive n=9; Invasive/no known metastases n=18; Metastatic n=14; Missing n=1 | aggregate descriptive context only | no sample-level stage sensitivity analysis; no secondary stage prediction task; no TNM/AJCC recoding |

Publication Table 1 provides aggregate CRC disease-extent counts only. These are useful for context but must not be assigned to individual local samples.

| source                                | stage_category                |   reported_percent_among_crc_cases |   n_crc_cases |   implied_count | participant_level_mapping_available   |
|:--------------------------------------|:------------------------------|-----------------------------------:|--------------:|----------------:|:--------------------------------------|
| Sinha et al. 2016 publication Table 1 | Non-invasive                  |                               21.4 |            42 |               9 | False                                 |
| Sinha et al. 2016 publication Table 1 | Invasive, no known metastases |                               42.9 |            42 |              18 | False                                 |
| Sinha et al. 2016 publication Table 1 | Metastatic                    |                               33.3 |            42 |              14 | False                                 |
| Sinha et al. 2016 publication Table 1 | Missing                       |                                2.4 |            42 |               1 | False                                 |

Official supporting files searched:

| source                      | path                                                                                                                                               | downloaded   |   n_rows |   n_columns | id_column   |   sample_overlap_with_local | stage_field_exists   |   stage_column_candidates |   medication_column_candidates | notes                                                                                  |
|:----------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------|:-------------|---------:|------------:|:------------|----------------------------:|:---------------------|--------------------------:|-------------------------------:|:---------------------------------------------------------------------------------------|
| S1 Table                    | /Users/bhaktidoshi/Downloads/Thesis Data/thesis_preprocessing_workflow/crc_pipeline/external_metadata_sources/sinha_s1_correlated_metabolites.xlsx | True         |      125 |           7 | nan         |                           0 | False                |                       nan |                            nan | official sample-level source; no participant-level CRC stage or medication field found |
| S2 Table                    | /Users/bhaktidoshi/Downloads/Thesis Data/thesis_preprocessing_workflow/crc_pipeline/external_metadata_sources/sinha_s2_metabolites.csv             | True         |      131 |         531 | ID          |                         131 | False                |                       nan |                            nan | official sample-level source; no participant-level CRC stage or medication field found |
| S3 Table                    | /Users/bhaktidoshi/Downloads/Thesis Data/thesis_preprocessing_workflow/crc_pipeline/external_metadata_sources/sinha_s3_microbes.csv                | True         |      131 |         221 | ID          |                         131 | False                |                       nan |                            nan | official sample-level source; no participant-level CRC stage or medication field found |
| S4 Table                    | /Users/bhaktidoshi/Downloads/Thesis Data/thesis_preprocessing_workflow/crc_pipeline/external_metadata_sources/sinha_s4_covariates.csv              | True         |      131 |          25 | ID          |                         131 | False                |                       nan |                            nan | official sample-level source; no participant-level CRC stage or medication field found |
| S5 Table                    | /Users/bhaktidoshi/Downloads/Thesis Data/thesis_preprocessing_workflow/crc_pipeline/external_metadata_sources/sinha_s5_crc_72_pairs.xlsx           | True         |       78 |          17 | nan         |                           0 | False                |                       nan |                            nan | official sample-level source; no participant-level CRC stage or medication field found |
| Borenstein curated metadata | /Users/bhaktidoshi/Downloads/Thesis Data/thesis_preprocessing_workflow/crc_pipeline/external_metadata_sources/SINHA_CRC_2016_metadata_curated.tsv  | True         |      131 |          13 | Sample      |                         131 | False                |                       nan |                            nan | official sample-level source; no participant-level CRC stage or medication field found |

Linked source-study and repository search:

| candidate_source                                                      | url                                           | exact_file_or_table                             |   stage_related_column_name |   original_coding |   linkage_identifier | matches_local_sample_or_subject   |   local_samples_linked |   unmatched_local_samples |   duplicated_linkage_records |   conflicting_stage_records |   missing_stage_records |   counts_by_recovered_stage | audit_result                                                                                                                                      |
|:----------------------------------------------------------------------|:----------------------------------------------|:------------------------------------------------|----------------------------:|------------------:|---------------------:|:----------------------------------|-----------------------:|--------------------------:|-----------------------------:|----------------------------:|------------------------:|----------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------|
| Sinha et al. 2016 Figshare record                                     | https://api.figshare.com/v2/articles/3129739  | article file manifest                           |                         nan |               nan |                  nan | False                             |                      0 |                       131 |                            0 |                           0 |                       0 |                         nan | official repository exposes S1-S5 only; those files are audited separately                                                                        |
| NYU Data Catalog dataset 10121                                        | https://datacatalog.med.nyu.edu/dataset/10121 | catalog record linking to Figshare              |                         nan |               nan |                  nan | False                             |                      0 |                       131 |                            0 |                           0 |                       0 |                         nan | catalog confirms 42 CRC cases and 89 controls and points to Figshare; no additional participant-level stage file is exposed in the catalog page   |
| Ahn et al. 2013 JNCI source microbiome study                          | https://doi.org/10.1093/jnci/djt300           | linked source publication / supplementary trail |                         nan |               nan |                  nan | False                             |                      0 |                       131 |                            0 |                           0 |                       0 |                         nan | source study confirms same CRC/control context but no deterministic participant-level stage table matching local CRC### identifiers was recovered |
| Goedert et al. 2014 Carcinogenesis source metabolomics study          | https://doi.org/10.1093/carcin/bgu131         | linked source publication / supplementary trail |                         nan |               nan |                  nan | False                             |                      0 |                       131 |                            0 |                           0 |                       0 |                         nan | stage is discussed at aggregate/disease-extent level, but no sample-level stage column keyed to local identifiers was recovered                   |
| Schiffman et al. 1989 Cancer Research fecal mutagenicity source study | https://pubmed.ncbi.nlm.nih.gov/2655896/      | source epidemiology publication                 |                         nan |               nan |                  nan | False                             |                      0 |                       131 |                            0 |                           0 |                       0 |                         nan | no modern participant-level machine-readable stage file with local identifiers was recovered                                                      |
| Schiffman et al. 1989 Cancer Research fecapentaene source study       | https://pubmed.ncbi.nlm.nih.gov/2917361/      | source epidemiology publication                 |                         nan |               nan |                  nan | False                             |                      0 |                       131 |                            0 |                           0 |                       0 |                         nan | no modern participant-level machine-readable stage file with local identifiers was recovered                                                      |

## Yachida Audit

Original local metadata group counts are shown below. These are original `Study.Group` labels and therefore intentionally retain `Stage_I_II` and `Stage_III_IV`.

| cohort   | label_column   | class        |   n_samples |   n_participants |
|:---------|:---------------|:-------------|------------:|-----------------:|
| Yachida  | Study.Group    | HS           |          30 |               30 |
| Yachida  | Study.Group    | Healthy      |         127 |              127 |
| Yachida  | Study.Group    | MP           |          40 |               40 |
| Yachida  | Study.Group    | Stage_0      |          27 |               27 |
| Yachida  | Study.Group    | Stage_III_IV |          54 |               54 |
| Yachida  | Study.Group    | Stage_I_II   |          69 |               69 |

Official Supplementary Table S2-2 contains individual `Stage` values. All 347 local Yachida samples link by `Sample`/`Subject_ID`; no true stage conflicts were found. Recovered official stages allow Yachida-only secondary analyses of Healthy vs Stage 0, Healthy vs Stage I, Healthy vs Stage II, Healthy vs Stage III and Healthy vs Stage IV. These analyses are not externally validated in Sinha.

| source                                  | question                                               | answer   | official_columns         |   local_samples |   supplement_rows |   local_sample_overlap |   unmatched_local_samples |   duplicated_official_subject_ids |   stage_mismatch_records |   stage_iii_local_paired_subset_n |   stage_iv_local_paired_subset_n | recommended_use                                                                                      |
|:----------------------------------------|:-------------------------------------------------------|:---------|:-------------------------|----------------:|------------------:|-----------------------:|--------------------------:|----------------------------------:|-------------------------:|----------------------------------:|---------------------------------:|:-----------------------------------------------------------------------------------------------------|
| Yachida Nature supplementary Table S2-2 | Can individual Stage III versus Stage IV be recovered? | Yes      | Subject_ID; Group; Stage |             347 |               406 |                    347 |                         0 |                                 0 |                        0 |                                34 |                               20 | Use for Yachida-only secondary stage analyses. These analyses are not externally validated in Sinha. |

Recovered official Yachida stage counts:

| Study.Group   | yachida_stage_source_raw   | yachida_stage_source_faithful_label   | yachida_stage_harmonized_label   | trajectory_eligible   |   n |
|:--------------|:---------------------------|:--------------------------------------|:---------------------------------|:----------------------|----:|
| HS            | -                          | HS                                    | Post-surgical control            | False                 |  30 |
| Healthy       | -                          | Healthy                               | Healthy                          | True                  | 127 |
| MP            | -                          | MP                                    | Precursor lesion                 | True                  |  40 |
| Stage_0       | 0                          | Stage 0                               | Non-invasive CRC                 | True                  |  27 |
| Stage_III_IV  | IIIA                       | Stage IIIA                            | Regional CRC                     | True                  |   8 |
| Stage_III_IV  | IIIB                       | Stage IIIB                            | Regional CRC                     | True                  |  22 |
| Stage_III_IV  | IIIC                       | Stage IIIC                            | Regional CRC                     | True                  |   4 |
| Stage_III_IV  | IVA                        | Stage IVA                             | Metastatic CRC                   | True                  |  12 |
| Stage_III_IV  | IVB                        | Stage IVB                             | Metastatic CRC                   | True                  |   6 |
| Stage_III_IV  | Iva                        | Stage IVA                             | Metastatic CRC                   | True                  |   1 |
| Stage_III_IV  | Ivb                        | Stage IVB                             | Metastatic CRC                   | True                  |   1 |
| Stage_I_II    | I                          | Stage I                               | Localized CRC                    | True                  |  44 |
| Stage_I_II    | IIA                        | Stage IIA                             | Localized CRC                    | True                  |  24 |
| Stage_I_II    | IIC                        | Stage IIC                             | Localized CRC                    | True                  |   1 |

## Harmonized Stage Fields Data Dictionary

| field                               | definition                                                                                           | source                                                                     | allowed_values                                                                                                             | intended_use                                                                                                  |
|:------------------------------------|:-----------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------|
| Study.Group                         | Original local Yachida metadata group before official stage recovery. Uses coarse local labels only. | Local metadata.tsv / Borenstein curated Yachida metadata                   | Healthy; MP; HS; Stage_0; Stage_I_II; Stage_III_IV                                                                         | Audit original local metadata and define original cohort groups; do not mix with recovered Stage I-IV labels. |
| yachida_stage_source_raw            | Raw official individual stage value exactly as recovered from Yachida Supplementary Table S2-2.      | Yachida et al. 2019 Nature Medicine Supplementary Table S2-2, column Stage | -; 0; I; IIA; IIC; IIIA; IIIB; IIIC; IVA; IVB; Iva; Ivb in the local subset                                                | Source provenance and reproducibility; preserve raw capitalization.                                           |
| yachida_stage_normalized            | Capitalization-normalized version of the official Stage field only.                                  | Derived from yachida_stage_source_raw                                      | None; 0; I; IIA; IIC; IIIA; IIIB; IIIC; IVA; IVB                                                                           | Internal consistency for grouping; not a new biological mapping.                                              |
| yachida_stage_major                 | Major recovered official stage collapsed from normalized individual substages.                       | Derived from yachida_stage_normalized                                      | Stage 0; Stage I; Stage II; Stage III; Stage IV; missing for Healthy/MP/HS                                                 | Yachida-only secondary Healthy vs Stage 0/I/II/III/IV analyses.                                               |
| yachida_stage_substage              | Recovered official substage retained where present for Stage III and Stage IV.                       | Derived from yachida_stage_normalized                                      | IIIA; IIIB; IIIC; IVA; IVB; blank otherwise                                                                                | Descriptive audit only unless future sample size supports substage modelling.                                 |
| yachida_stage_source_faithful_label | Human-readable label preserving official recovered stage or original non-CRC group.                  | Derived from Study.Group and yachida_stage_source_raw                      | Healthy; MP; HS; Stage 0; Stage I; Stage IIA; Stage IIC; Stage IIIA; Stage IIIB; Stage IIIC; Stage IVA; Stage IVB; Unknown | Source-faithful reporting and exploratory fine labels.                                                        |
| yachida_stage_harmonized_label      | Clinically interpretable grouping of original groups and recovered official stages.                  | Derived from Study.Group and yachida_stage_major                           | Healthy; Precursor lesion; Non-invasive CRC; Localized CRC; Regional CRC; Metastatic CRC; Post-surgical control; Unknown   | Coarse Yachida-only secondary/exploratory analyses; MP is not CRC and HS is not Healthy.                      |
| trajectory_label                    | Label used for the exploratory cross-sectional ordered disease-state variable.                       | Derived from source-faithful Yachida labels                                | Healthy; MP; Stage 0; Stage I; Stage IIA/IIC; Stage IIIA/IIIB/IIIC; Stage IVA/IVB; HS excluded                             | Exploratory ordered disease-state description only; not observed longitudinal progression.                             |
| trajectory_order                    | Numeric order for independently sampled disease states.                                              | Derived from trajectory_label                                              | Healthy=0; MP=1; Stage 0=2; Stage I=3; Stage II=4; Stage III=5; Stage IV=6; missing for HS                                 | Exploratory ordinal methods only; must not be interpreted as within-patient temporal follow-up.               |
| crc_primary_binary_label            | Primary binary CRC-vs-control label for Yachida component of the shared external-validation task.    | Derived from recovered official stages and original Healthy group          | Control for Healthy; CRC for Stage 0, Stage I, Stage II, Stage III, and Stage IV; missing/excluded for MP and HS                                           | Primary CRC vs Healthy task; externally validated only as binary CRC vs Control with Sinha.                   |

## External Metadata Recovery Audit

| cohort   | source_name                                              | source_file                                                                                                                                          | source_url                                                                                                                                  | downloaded   |   n_rows |   n_columns |   sample_overlap_with_local | columns_recovered                                                                                                                                                                                                                                                                                                                                              | sample_level_antibiotic_available   | sample_level_steroid_available   | sample_level_stage_or_subtype_available   | sample_level_smoking_available   | sample_level_alcohol_available   | relevant_columns_found                                                                                                                                                                  | action_taken                                                                                        |
|:---------|:---------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------|:-------------|---------:|------------:|----------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------|:---------------------------------|:------------------------------------------|:---------------------------------|:---------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------|
| Sinha    | PLOS supplementary S4 covariates                         | /Users/bhaktidoshi/Downloads/Thesis Data/thesis_preprocessing_workflow/crc_pipeline/external_metadata_sources/sinha_s4_covariates.csv                | https://journals.plos.org/plosone/article/file?type=supplementary&id=info:doi/10.1371/journal.pone.0152126.s004                             | True         |      131 |          25 |                         131 | ID; case; age; bmi; sex; race; hosp1; hosp2; PALMITOYL_SPHINGOMYELIN; MANDELATE; P_HYDROXYBENZALDEHYDE; ALPHA_TOCOPHEROL; GAMMA_TOCOPHEROL; SITOSTANOL; _3_DEHYDROCARNITINE; PTERIN; CONJUGATED_LINOLEATE__18_2N7; N__2_FUROYL_GLYCINE; P_AMINOBENZOATE__PABA_; c_Clostridia; f_Lachnospiraceae; p_Fusobacteria; g_Fusobacterium; g_Atopobium; g_Porphyromonas | False                               | False                            | False                                     | False                            | False                            | {"antibiotic": [], "steroid": [], "medication_or_treatment": [], "stage_or_subtype": [], "smoking": [], "alcohol": []}                                                                  | confirms binary case-control only; no sample-level CRC stage/subtype or medication fields recovered |
| Sinha    | Borenstein curated metadata                              | /Users/bhaktidoshi/Downloads/Thesis Data/thesis_preprocessing_workflow/crc_pipeline/external_metadata_sources/SINHA_CRC_2016_metadata_curated.tsv    | https://raw.githubusercontent.com/borenstein-lab/microbiome-metabolome-curated-data/main/data/processed_data/SINHA_CRC_2016/metadata.tsv    | True         |      131 |          13 |                         131 | Dataset; Sample; Subject; Study.Group; Age; Age.Units; Gender; BMI; DOI; Publication.Name; race; hosp1; hosp2                                                                                                                                                                                                                                                  | False                               | False                            | False                                     | False                            | False                            | {"antibiotic": [], "steroid": [], "medication_or_treatment": [], "stage_or_subtype": [], "smoking": [], "alcohol": []}                                                                  | confirms binary case-control only; no sample-level CRC stage/subtype or medication fields recovered |
| Yachida  | Nature supplementary Table S2-2 metabolome clinical data | /Users/bhaktidoshi/Downloads/Thesis Data/thesis_preprocessing_workflow/crc_pipeline/external_metadata_sources/yachida_supplementary_tables_1_15.xlsx | https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41591-019-0458-7/MediaObjects/41591_2019_458_MOESM3_ESM.xlsx | True         |      406 |           9 |                         347 | Subject_ID; Group; Stage; Age; Gender; BMI; Brinkman Index; Alcohol; Tumor location                                                                                                                                                                                                                                                                            | False                               | False                            | True                                      | True                             | True                             | {"antibiotic": [], "steroid": [], "medication_or_treatment": [], "stage_or_subtype": ["Group", "Stage", "Tumor location"], "smoking": ["Brinkman Index"], "alcohol": ["Alcohol"]}       | confirms local stage/tumour-location/smoking/alcohol metadata; no medication fields recovered       |
| Yachida  | Borenstein curated metadata                              | /Users/bhaktidoshi/Downloads/Thesis Data/thesis_preprocessing_workflow/crc_pipeline/external_metadata_sources/YACHIDA_CRC_2019_metadata_curated.tsv  | https://raw.githubusercontent.com/borenstein-lab/microbiome-metabolome-curated-data/main/data/processed_data/YACHIDA_CRC_2019/metadata.tsv  | True         |      347 |          15 |                         347 | Dataset; Sample; Subject; Study.Group; Age; Age.Units; Gender; BMI; DOI; Publication.Name; Stage; Brinkman Index; Alcohol; Tumor location; Shared.w.ERAWIJANTARI_2020                                                                                                                                                                                          | False                               | False                            | True                                      | True                             | True                             | {"antibiotic": [], "steroid": [], "medication_or_treatment": [], "stage_or_subtype": ["Study.Group", "Stage", "Tumor location"], "smoking": ["Brinkman Index"], "alcohol": ["Alcohol"]} | confirms local stage/tumour-location/smoking/alcohol metadata; no medication fields recovered       |

Antibiotic exposure, steroid/corticosteroid exposure, immunosuppressive medication and treatment-status variables were not recovered at participant level for either CRC cohort. Absence of these fields is recorded as unavailable, not as non-exposure.

## Sample and Participant Structure

| cohort   | item                                      |   n |
|:---------|:------------------------------------------|----:|
| Sinha    | metadata_samples                          | 131 |
| Sinha    | metabolomics_samples                      | 131 |
| Sinha    | samples_in_both_metadata_and_metabolomics | 131 |
| Sinha    | metadata_only_samples                     |   0 |
| Sinha    | metabolomics_only_samples                 |   0 |
| Sinha    | unique_participants                       | 131 |
| Sinha    | participants_with_repeated_samples        |   0 |
| Yachida  | metadata_samples                          | 347 |
| Yachida  | metabolomics_samples                      | 347 |
| Yachida  | samples_in_both_metadata_and_metabolomics | 347 |
| Yachida  | metadata_only_samples                     |   0 |
| Yachida  | metabolomics_only_samples                 |   0 |
| Yachida  | unique_participants                       | 347 |
| Yachida  | participants_with_repeated_samples        |   0 |

Both local CRC tables have one sample per subject, no duplicated sample IDs, and complete sample overlap between metadata and metabolomics.

## Identifier Overlap

| identifier_type      |   Sinha_nonmissing_unique |   Yachida_nonmissing_unique |   shared_unique_identifiers |
|:---------------------|--------------------------:|----------------------------:|----------------------------:|
| HMDB_raw             |                       354 |                         420 |                         110 |
| KEGG_raw             |                       189 |                         431 |                         107 |
| HMDB                 |                       354 |                         420 |                         110 |
| KEGG                 |                       189 |                         431 |                         107 |
| HMDB_or_KEGG_unified |                       543 |                         851 |                         217 |

The shared identifier space supports a future binary CRC external-validation design, but modelling must still account for platform/scale differences: Sinha values appear already transformed/standardized, whereas Yachida values are raw non-negative, zero-inflated abundance-like values.

## Valid Prediction Tasks

### Primary External-Validation Task

Primary external-validation task:

Healthy

vs

CRC

where CRC consists of:

- Stage 0
- Stage I
- Stage II
- Stage III
- Stage IV

MP excluded.

HS excluded.

For Sinha, this maps to binary CRC vs Control only. For Yachida, Healthy is Control and recovered official Stage 0/I/II/III/IV are CRC.

### Secondary Yachida-Only Stage Analyses

Recovered official stage labels allow Yachida-only analyses of:

- Healthy vs Stage 0
- Healthy vs Stage I
- Healthy vs Stage II
- Healthy vs Stage III
- Healthy vs Stage IV

These are secondary or exploratory internal analyses only. They should not be described as externally validated because Sinha has no deterministically linked participant-level stage labels.

### Exploratory Cross-Sectional Ordered Disease-State Variable

The exploratory cross-sectional ordered disease-state variable is ordered Healthy → MP → Stage 0 → Stage I → Stage II → Stage III → Stage IV.

This ordering represents disease states across independently sampled individuals. It must not be interpreted as observed longitudinal progression within patients.

## Stage 2 Task Eligibility and Planning

| task_name                                                  | cohort_scope                                  | label_column     | positive_or_multiclass_definition                                                        | control_definition                         |   n_samples |   n_controls |   n_disease_or_noncontrol |   minority_class_n | technical_cv_feasibility    | recommended_analysis_type   | recommended_cv_strategy   | stage2_prep_file                                                        | notes                                                                                                                                                                                                                                                                                                                                                                                         |
|:-----------------------------------------------------------|:----------------------------------------------|:-----------------|:-----------------------------------------------------------------------------------------|:-------------------------------------------|------------:|-------------:|--------------------------:|-------------------:|:----------------------------|:----------------------------|:--------------------------|:------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Healthy vs MP                                              | Yachida only                                  | model_label      | MP                                                                                       | Healthy                                    |         167 |          127 |                        40 |                 40 | 5-fold technically possible | Exploratory                 | Exploratory only          | crc_stage2_yachida_healthy_vs_mp_model_prep.csv                         | MP is a precursor-lesion group; useful descriptively but not externally validated in Sinha. Raw, non-imputed, non-scaled; all trainable preprocessing must occur inside future CV pipelines.                                                                                                                                                                                                  |
| Healthy vs Stage 0                                         | Yachida only                                  | model_label      | Stage 0                                                                                  | Healthy                                    |         154 |          127 |                        27 |                 27 | 5-fold technically possible | Secondary                   | Exploratory only          | crc_stage2_yachida_healthy_vs_stage_0_model_prep.csv                    | Recovered official stage supports this Yachida-only contrast, but stage-specific sample size is modest. Raw, non-imputed, non-scaled; all trainable preprocessing must occur inside future CV pipelines.                                                                                                                                                                                      |
| Healthy vs Stage I                                         | Yachida only                                  | model_label      | Stage I                                                                                  | Healthy                                    |         171 |          127 |                        44 |                 44 | 5-fold technically possible | Secondary                   | Exploratory only          | crc_stage2_yachida_healthy_vs_stage_i_model_prep.csv                    | Recovered official stage supports this Yachida-only contrast, but it remains internally evaluated only. Raw, non-imputed, non-scaled; all trainable preprocessing must occur inside future CV pipelines.                                                                                                                                                                                      |
| Healthy vs Stage II                                        | Yachida only                                  | model_label      | Stage II                                                                                 | Healthy                                    |         152 |          127 |                        25 |                 25 | 5-fold technically possible | Secondary                   | Exploratory only          | crc_stage2_yachida_healthy_vs_stage_ii_model_prep.csv                   | Recovered official stage supports this Yachida-only contrast; n=25 disease samples makes formal claims fragile. Raw, non-imputed, non-scaled; all trainable preprocessing must occur inside future CV pipelines.                                                                                                                                                                              |
| Healthy vs Stage III                                       | Yachida only                                  | model_label      | Stage III                                                                                | Healthy                                    |         161 |          127 |                        34 |                 34 | 5-fold technically possible | Secondary                   | 3-fold nested CV          | crc_stage2_yachida_healthy_vs_stage_iii_model_prep.csv                  | Recovered official stage supports this Yachida-only contrast; 3-fold nested CV is preferred over 5-fold for reliability. Raw, non-imputed, non-scaled; all trainable preprocessing must occur inside future CV pipelines.                                                                                                                                                                     |
| Healthy vs Stage IV                                        | Yachida only                                  | model_label      | Stage IV                                                                                 | Healthy                                    |         147 |          127 |                        20 |                 20 | 3-fold technically possible | Secondary                   | Exploratory only          | crc_stage2_yachida_healthy_vs_stage_iv_model_prep.csv                   | Metastatic CRC has only 20 disease samples; use cautiously and do not externally validate in Sinha. Raw, non-imputed, non-scaled; all trainable preprocessing must occur inside future CV pipelines.                                                                                                                                                                                          |
| Healthy vs localized CRC                                   | Yachida only                                  | model_label      | Localized CRC                                                                            | Healthy                                    |         196 |          127 |                        69 |                 69 | 5-fold technically possible | Secondary                   | Exploratory only          | crc_stage2_yachida_healthy_vs_localized_crc_model_prep.csv              | Coarse recovered-stage grouping; internally evaluated only. Raw, non-imputed, non-scaled; all trainable preprocessing must occur inside future CV pipelines.                                                                                                                                                                                                                                  |
| Healthy vs regional CRC                                    | Yachida only                                  | model_label      | Regional CRC                                                                             | Healthy                                    |         161 |          127 |                        34 |                 34 | 5-fold technically possible | Secondary                   | 3-fold nested CV          | crc_stage2_yachida_healthy_vs_regional_crc_model_prep.csv               | Coarse recovered-stage grouping corresponding to Stage III; internally evaluated only. Raw, non-imputed, non-scaled; all trainable preprocessing must occur inside future CV pipelines.                                                                                                                                                                                                       |
| Healthy vs metastatic CRC                                  | Yachida only                                  | model_label      | Metastatic CRC                                                                           | Healthy                                    |         147 |          127 |                        20 |                 20 | 3-fold technically possible | Secondary                   | Exploratory only          | crc_stage2_yachida_healthy_vs_metastatic_crc_model_prep.csv             | Coarse recovered-stage grouping with small metastatic sample size; internally evaluated only. Raw, non-imputed, non-scaled; all trainable preprocessing must occur inside future CV pipelines.                                                                                                                                                                                                |
| Fine multiclass source-faithful stage                      | Yachida only                                  | model_label      | Healthy; MP; Stage 0; Stage I; Stage II; Stage III; Stage IV                             | included as its own class                  |         238 |          127 |                       111 |                 27 | 5-fold technically possible | Exploratory                 | Exploratory only          | crc_stage2_yachida_fine_multiclass_source_faithful_stage_model_prep.csv | Fine multiclass analysis is underpowered for several substages and internally evaluated only. Class counts: {'Healthy': 127, 'Stage I': 44, 'MP': 40, 'Stage 0': 27}; raw, non-imputed, non-scaled.                                                                                                                                                             |
| Coarse harmonized stage                                    | Yachida only                                  | model_label      | Healthy; Precursor lesion; Non-invasive CRC; Localized CRC; Regional CRC; Metastatic CRC | included as its own class                  |         317 |          127 |                       190 |                 20 | 3-fold technically possible | Exploratory                 | Exploratory only          | crc_stage2_yachida_coarse_harmonized_stage_model_prep.csv               | Coarse multiclass analysis is useful for descriptive structure but not a primary model. Class counts: {'Healthy': 127, 'Localized CRC': 69, 'Precursor lesion': 40, 'Regional CRC': 34, 'Non-invasive CRC': 27, 'Metastatic CRC': 20}; raw, non-imputed, non-scaled.                                                                                            |
| Exploratory cross-sectional ordered disease-state variable | Yachida only                                  | trajectory_order | Healthy=0; MP=1; Stage 0=2; Stage I=3; Stage II=4; Stage III=5; Stage IV=6               | Healthy at trajectory order 0; HS excluded |         317 |          127 |                       190 |                  1 | descriptive only            | Exploratory                 | Exploratory only          | crc_stage2_yachida_exploratory_ordinal_trajectory_model_prep.csv        | Ordering represents disease states across independently sampled individuals, not observed longitudinal progression within patients. Class counts: {'Healthy': 127, 'MP': 40, 'Stage 0': 27, 'Stage I': 44, 'Stage IIA': 24, 'Stage IIC': 1, 'Stage IIIA': 8, 'Stage IIIB': 22, 'Stage IIIC': 4, 'Stage IVA': 13, 'Stage IVB': 7}; raw, non-imputed, non-scaled. |
| Primary Yachida CRC vs Control                             | Yachida only component of shared primary task | model_label      | Stage 0; Stage I; Stage II; Stage III; Stage IV as CRC                                                               | Healthy only; MP and HS excluded           |         277 |          127 |                       150 |                127 | 5-fold technically possible | Primary                     | 5-fold nested CV          | crc_stage2_yachida_primary_crc_vs_control_model_prep.csv                | Primary external-validation task component; technically possible and scientifically reliable enough for primary CRC-vs-control modelling. Raw, non-imputed, non-scaled; use with Sinha binary only after shared-feature alignment.                                                                                                                                                            |

The table distinguishes `technical_cv_feasibility` from `recommended_cv_strategy`. Fold counts are therefore not recommended solely from class counts; they are interpreted with disease stage, scientific reliability and external-validation support.

## Figure Captions / Interpretation

- `crc_stage1_pca_Sinha`: Sinha cohort-only exploratory PCA, coloured by binary CRC/control status. Exploratory only.
- `crc_stage1_pca_Yachida`: Yachida cohort-only exploratory PCA, coloured by original CRC/control/non-primary grouping. Exploratory only.
- `crc_stage1_yachida_detailed_stage_distribution`: recovered official Yachida stage/substage labels from Supplementary Table S2-2.
- `crc_stage1_yachida_harmonized_progression_distribution`: harmonized Yachida labels for clinical interpretation. MP is Precursor lesion, not CRC. HS is Post-surgical control, not Healthy.
- `crc_stage1_yachida_trajectory_counts`: exploratory cross-sectional ordered disease-state variable; this is not longitudinal follow-up.
- `crc_stage2_task_eligibility`: Stage 2 planning summary using recovered official Yachida labels and manual scientific CV recommendations.

## Proposed Cohort-Selection Rules

1. Retain samples with non-missing sample ID, participant ID, disease label and metabolomics row.
2. Use one sample per participant. The current local CRC data already satisfy this.
3. For the Yachida component of the primary binary CRC vs Control task, use Healthy as Control and recovered official Stage 0; Stage I; Stage II; Stage III; Stage IV as CRC. Exclude MP and HS.
4. For Sinha, use binary `Study.Group`: 0 as Control and 1 as CRC.
5. Do not exclude antibiotic/steroid/treatment-exposed samples unless official participant-level variables are recovered. Missing medication fields should be reported as unavailable, not treated as unexposed.
6. Do not remove PCA outliers unless an independent study QC flag exists.

## Leakage-Safe Future Modelling

All trainable preprocessing must be fitted inside future training folds only:

1. missingness filtering using the predefined primary 30% threshold;
2. constant-feature removal;
3. imputation;
4. transformation if justified;
5. scaling;
6. feature selection or dimensionality reduction, if used;
7. model fitting and threshold selection.

No imputation, scaling, PCA, feature selection or threshold optimisation should be fitted globally before nested cross-validation.

## Transformation Recommendation

| cohort   |   n_values |      min |      p01 |      median |        mean |         p99 |          max |   zero_fraction |   negative_fraction | recommended_exploratory_transform                                               | modeling_note                                                               |
|:---------|-----------:|---------:|---------:|------------:|------------:|------------:|-------------:|----------------:|--------------------:|:--------------------------------------------------------------------------------|:----------------------------------------------------------------------------|
| Sinha    |      69430 | -6.72543 | -2.86646 | -0.00954541 |  -0.0391872 |     2.91341 |      7.61339 |      0.00593403 |            0.508829 | none; values appear already transformed/standardized                            | any transformation choice must be inside the training-fold sklearn pipeline |
| Yachida  |     156150 |  0       |  0       |  0          | 690.598     | 14929.6     | 665181       |      0.603324   |            0        | log1p for exploratory plots only; raw non-negative abundances are zero-inflated | any transformation choice must be inside the training-fold sklearn pipeline |

Zeros should not be recoded as missing without source-study evidence. In Yachida, zeros are abundant and likely reflect absence/below-detection abundance-like measurements; they should be treated cautiously, not automatically as NA.

## Limitations

- Sinha participant-level stage information could not be recovered from the official sample-level data currently available and audited.
- Sinha supports binary CRC vs Control analyses only under the current audited data release.
- Antibiotic metadata are unavailable for both CRC cohorts.
- Steroid/corticosteroid metadata are unavailable for both CRC cohorts.
- Treatment-status and immunosuppressive-medication metadata are unavailable for both CRC cohorts.
- Tumour site is unavailable for Sinha; Yachida tumour location is case/stage-context metadata only.
- Yachida stage analyses are internally valid only and cannot be externally validated in Sinha at stage level.
- The exploratory ordered disease-state variable is cross-sectional and does not demonstrate biological progression.
- Stage-specific tasks have relatively small sample sizes, especially Stage IV and substage-level analyses.
- MP and HS require careful handling: MP is a precursor-lesion group, while HS is a post-surgical control group.

## Automated Stage-Recovery Checks

| test                                                         | passed   |   observed |   expected |   note |
|:-------------------------------------------------------------|:---------|-----------:|-----------:|-------:|
| Yachida local samples linked to official S2-2                | True     |        347 |        347 |    nan |
| Yachida unmatched official stage rows                        | True     |          0 |          0 |    nan |
| Yachida stage conflicts                                      | True     |          0 |          0 |    nan |
| Yachida Stage III count                                      | True     |         34 |         34 |    nan |
| Yachida Stage IV count                                       | True     |         20 |         20 |    nan |
| Yachida MP not mapped to CRC                                 | True     |       True |       True |    nan |
| Yachida HS not mapped to Healthy                             | True     |       True |       True |    nan |
| Yachida HS excluded from trajectory                          | True     |       True |       True |    nan |
| Sinha supports binary CRC vs Control only after source audit | True     |       True |       True |    nan |

## Remaining Issues Before Stage 2 Modelling

1. Decide whether Stage II, Stage IV and multiclass tasks should remain descriptive/exploratory only in the dissertation results.
2. Decide whether Yachida tumour location, Brinkman Index and alcohol should be used only in Yachida-only analyses or excluded from shared external-validation modelling.
3. Implement the future modelling script with all preprocessing inside nested CV pipelines.
4. Keep Sinha stage information out of modelling unless a deterministic participant-level official source is later recovered.
