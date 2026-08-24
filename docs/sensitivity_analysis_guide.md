# Sensitivity Analysis Guide

## CRC microbiome representation sensitivity (post-freeze)

**What it tests**: whether the primary CRC microbiome CLR-based transportability result (Sinha→Yachida ROC-AUC ≈0.563, Yachida→Sinha ≈0.582) depends on the abundance-scale representation, given that Sinha (16S) and Yachida (shotgun) differ in quantification platform.

**Variants**: `clr_zscore` (primary, replicated exactly as an acceptance check), `within_sample_rank` (scale-robust — invariant to a constant multiplicative rescale), `presence_absence` (scale-robust — binary detected/not-detected, training-fold-scaled to match L2 penalty geometry across variants).

**Design**: identical frozen Stage 4B nested-CV/threshold/bootstrap protocol, restricted to the CRC task and MICROBIOME representation. An automated acceptance check aborts the run if the `clr_zscore` arm does not reproduce the frozen primary AUCs within 1e-6 before the rank/presence-absence results are interpreted.

**Result**: for the Yachida-external direction, both rank and presence/absence performed significantly *worse* than CLR (BH-FDR q=0.036 both) — the result is representation-sensitive in that direction. For the Sinha-external direction, both comparisons were inconclusive.

**Interpretation**: neither variant "solves" the 16S-vs-shotgun confound (platform is still perfectly confounded with cohort for CRC) — they test whether the *specific numeric result* is scale-dependent, not whether the underlying biology transports.

**Location**: `final_analysis/code/run_stage4b_crc_microbiome_sensitivity.py`; outputs at `final_analysis/tables/sensitivity_crc_microbiome_*`.

## Domain-shift / transformation audit (Stage 4B)

Saved for every transfer and representation: source/external raw ranges and quantiles, extreme-z fractions, non-finite counts, decision-score ranges/uniqueness, probability range/SD, exact-0/1-probability flags, one-class-prediction flags. Location: `final_analysis/code/audit_stage4b_corrected_v2b_transform_shift.py`.

## Metabolomics common-scale feasibility

Investigated whether a defensible common-scale metabolomics matrix could be constructed locally; found no defensible option (see `docs/limitations.md`). Location: `final_analysis/code/audit_stage4b_v2c_common_scale_feasibility.py`, `audit_stage4b_v2c_metabolomics_comparability.py`.

## Stage 4E architecture/topology sensitivity

Sensitivity architecture variants exist alongside the primary GCN design, with a separate 48-comparison BH-FDR family (distinct from the 24-comparison primary family). See `final_analysis/tables/supp_stage4e_sensitivity_48_comparisons.csv`.

## Not run (documented as future work, not scope creep)

Genuinely nested outer-CV sCCA evaluation, sCCA bootstrap loading-selection stability, covariate-residualised MOFA/sCCA, and additional metabolomics rank/robust-scaling sensitivities were considered and explicitly not run — see `docs/limitations.md` for why each is left as a limitation rather than resolved.
