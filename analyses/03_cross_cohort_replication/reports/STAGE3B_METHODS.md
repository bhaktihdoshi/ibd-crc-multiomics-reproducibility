# Stage 3B Methods

Stage 3B used Yachida only to evaluate cross-sectional ordered disease-state associations across seven source-supported states: Healthy, MP, Stage 0, Stage I, Stage II, Stage III and Stage IV. HS samples were not part of this ordered-state analysis.

The analysis was restricted to the strict four-cohort shared HMDB space (n = 56). For each metabolite, the primary robust model was `metabolite_z ~ ordered_state_score`. Age, sex and BMI were audited for compatibility; compatible columns were not available in the Stage 3B source table, so no covariate-adjusted model was reported. Covariate audit: age_years: unavailable_in_stage3b_source_table; sex: unavailable_in_stage3b_source_table; bmi: unavailable_in_stage3b_source_table; alcohol: unavailable_in_stage3b_source_table; brinkman_index: unavailable_in_stage3b_source_table; tumor_location: unavailable_in_stage3b_source_table.

P-values were calculated using robust HC3 standard errors and a t-distribution reference. BH-FDR correction was applied across the 56 metabolite tests. Spearman ordered-state association was used as a non-parametric sensitivity check with a separate BH-FDR correction across the same 56 metabolites.
