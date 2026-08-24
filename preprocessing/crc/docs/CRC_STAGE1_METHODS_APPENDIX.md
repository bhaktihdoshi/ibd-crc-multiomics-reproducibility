# CRC Stage 1 Methods Appendix

## Data Sources

Sinha et al. 2016 and Yachida et al. 2019 stool/fecal CRC metabolomics datasets were used. Local metadata, metabolomics and metabolite mapping files were audited together with official Sinha supporting files and Yachida Supplementary Table S2-2.

## Inclusion Criteria

Samples were retained in Stage 1 if metadata and metabolomics identifiers could be linked. Both cohorts had one sample per subject in the local data.

## Exclusion Criteria

No model-time exclusions were applied in Stage 1. For the future primary binary task, Yachida MP and HS are excluded by definition: MP is a precursor-lesion group, while HS is a post-surgical control group.

## Metadata Recovery

Sinha supporting files and linked source resources did not provide deterministic participant-level stage, antibiotic, steroid, treatment or tumour-site fields. Yachida Supplementary Table S2-2 provided stage, age, gender, BMI, Brinkman Index, alcohol and tumour location.

## Stage Recovery

Yachida official stages were recovered from Supplementary Table S2-2. Raw labels were preserved, capitalization was normalized where needed, major stage and harmonized labels were derived, and automated tests confirmed expected counts.

## Harmonization

Metabolite identifier harmonization used cleaned HMDB and KEGG identifiers. Clinical harmonization retained only source-supported fields and documented cohort-specific fields separately.

## Justification For Binary Primary Task

Binary CRC vs Control is the only shared outcome supported by both cohorts. Yachida contributes Healthy vs recovered Stage 0/I/II/III/IV; Sinha contributes binary Control vs CRC.

## Justification For Secondary Tasks

Yachida-only Healthy vs stage analyses are supported by recovered official stage labels but cannot be externally validated in Sinha.

## Justification For Excluding MP

MP is a multiple polypoid adenoma/precursor-lesion group and is not mapped to CRC in the primary task.

## Justification For Excluding HS

HS is a post-surgical control group and is not equivalent to Healthy/no-neoplasia controls.

## Justification For Not Using Antibiotics/Steroids

Participant-level antibiotic and steroid variables were unavailable in the audited official CRC metadata. Missing metadata are not interpreted as non-exposure.

## Justification For Not Assigning Sinha Stages

Sinha reports aggregate disease-extent counts but does not provide deterministic participant-level stage labels linked to local sample identifiers. Individual labels were therefore not reconstructed.
