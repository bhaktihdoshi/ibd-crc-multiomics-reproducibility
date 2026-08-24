# Data Provenance

## Code licence vs. data licences vs. third-party resources

These are three separate things and must not be conflated:

1. **CODE LICENCE** — applies to the analysis code, scripts, notebooks, and documentation authored by the thesis project team in this repository. **MIT Licence** (see `LICENSE`), chosen after confirming no third-party licensed code was copied into the repository (checked via a full-repository scan for copyright headers/attribution comments — none found).
2. **DATA LICENCES** — the four source cohorts (Franzosa, iHMP/IBDMDB, Sinha, Yachida) and the MiMeDB reference database each have their own data-use terms, set by their original publishers/consortia, entirely independent of whatever code licence is chosen for this repository. None of that raw/source data is redistributed here — see `data/README.md`.
3. **THIRD-PARTY RESOURCES** — KEGG, HMDB, and NJC19 are third-party reference resources used as inputs to Stage 4D graph construction and metabolite-ID harmonisation. They are not redistributed in this repository; only the thesis team's code that *uses* them (and, where small enough, thesis-derived crosswalk tables referencing their identifiers) is included.

## What this repository actually contains, data-wise

- Thesis-team-authored processing/analysis code (falls under the code licence).
- Thesis-team-derived small crosswalk/harmonisation tables and processed feature matrices below the 10MB size threshold (see `docs/large_file_audit.csv` for what was excluded purely for size).
- No raw third-party data files (see `data/README.md` for the full exclusion list and how to obtain each source independently).

## Derived data redistribution

Even where a derived table is small enough to include and does not directly redistribute copyrighted third-party content, it may still indirectly reveal proprietary/restricted source content (e.g. a near-complete re-derivation of a restricted dataset). No such case was identified during curation, but see `docs/privacy_security_audit.md` for the one flagged review item (participant-level clinical metadata redistribution terms).
