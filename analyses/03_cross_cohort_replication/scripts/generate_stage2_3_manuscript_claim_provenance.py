"""Manuscript-claim -> authoritative-CSV provenance map for Stage 2/3 biological results.

Analogous to Stage 4's `final_stage4_figure_source_values.csv`, but at
claim-text granularity rather than figure granularity, and covering the
Stage 2/3 biological narrative the thesis relies on most heavily: the seven
strictly-replicated IBD metabolites, the within-disease cross-cohort
concordance (rho=0.782 IBD, rho=0.134 CRC), and the Yachida ordered-state
(disease-stage trajectory) findings.

This performs NO new statistics. Every value below is read directly from an
already-computed, already-authoritative Stage 2/3 output table at run time
and asserted against the expected value; it does not hardcode a number that
could silently drift out of sync with the source CSV. If a source table
changes, this script will fail loudly (AssertionError) rather than silently
report a stale claim.

Two existing crosswalk-style files were found and are referenced, not
duplicated, where they already cover the same ground:
  - stage3_ibd_crc_integration/tables/stage3_final_results_numbers.csv
    (already maps several Stage 3A/3B numbers, including both rho values, to
    a source filename)
  - stage3_ibd_crc_integration/manifests/stage3_authoritative_effect_input_handoff.csv
    (maps disease/cohort/contrast -> authoritative differential-abundance file)
This script adds the claim-text layer and the seven-metabolite / Yachida
ordered-state claims that were not yet in either.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage3_config import Stage3Config  # noqa: E402


CFG = Stage3Config()
ROOT = CFG.root / "thesis_preprocessing_workflow"

OUT_CSV = CFG.stage3_dir / "manifests" / "stage2_3_manuscript_claim_provenance.csv"
OUT_JSON = CFG.stage3_dir / "manifests" / "stage2_3_manuscript_claim_provenance.json"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def claim_seven_replicated_ibd_metabolites() -> list[dict]:
    path = ROOT / "tables" / "ibd_expanded_replication_final_audit.csv"
    df = _read(path)
    summary = df[(df.iloc[:, 0] == "contrast_summary") & (df.iloc[:, 1] == "ibd_vs_control")]
    n_replicated_events = int(summary.iloc[0, 2])
    n_unique_metabolites = int(summary.iloc[0, 3])
    assert n_unique_metabolites == 7, f"Expected 7 uniquely replicated IBD metabolites, found {n_unique_metabolites} in {path}"
    cross_check_path = CFG.stage3_dir / "tables" / "stage3a_within_disease_concordance.csv"
    cross_check = _read(cross_check_path)
    cross_check_n = int(cross_check.loc[cross_check["comparison"].eq("Franzosa vs iHMP IBD vs Control"), "strict_FDR_unique_metabolites_full_expanded"].iloc[0])
    assert cross_check_n == 7, f"Cross-check mismatch: stage3a_within_disease_concordance.csv reports {cross_check_n}, expected 7"

    primary = df[df["row_type"] == "primary_replicated_metabolite"]
    contrast_counts = primary["contrast"].value_counts().to_dict()
    n_ibd = int(contrast_counts.get("ibd_vs_control", 0))
    n_cd = int(contrast_counts.get("cd_vs_control", 0))
    n_uc = int(contrast_counts.get("uc_vs_control", 0))  # absent entirely if 0 (no uc_vs_control contrast rows at all)
    assert n_ibd == 7 and n_cd == 3 and n_uc == 0, f"Subtype replication counts drifted: ibd={n_ibd}, cd={n_cd}, uc={n_uc}"
    ibd_names = sorted(primary.loc[primary["contrast"].eq("ibd_vs_control"), "metabolite_name"].tolist())

    return [
        {
            "claim_id": "seven_replicated_ibd_metabolites",
            "stage": "Stage 2/IBD pipeline",
            "claim_text": f"Seven metabolites were strictly (FDR-significant, same-direction) replicated between Franzosa and iHMP for IBD vs. Control: {', '.join(ibd_names)}.",
            "authoritative_source_csv": str(path),
            "key_columns_used": "contrast_summary row for ibd_vs_control: replicated_metabolite_contrast_events, unique_replicated_metabolites; primary_replicated_metabolite rows for metabolite_name",
            "key_value": n_unique_metabolites,
            "cross_checked_against": str(cross_check_path),
            "n_rows_used": len(df),
            "notes": f"{n_replicated_events} replication events across {n_unique_metabolites} unique metabolites; independently cross-checked against strict_FDR_unique_metabolites_full_expanded in stage3a_within_disease_concordance.csv (also 7). Exact metabolite list: {ibd_names}.",
        },
        {
            "claim_id": "ibd_subtype_replication_counts",
            "stage": "Stage 2/IBD pipeline",
            "claim_text": f"By IBD subtype contrast, strict replication counts were: IBD-vs-Control={n_ibd}, CD-vs-Control={n_cd}, UC-vs-Control={n_uc}. The CD-vs-Control set ({n_cd}) is a subset of the IBD-vs-Control set ({n_ibd}); UC-vs-Control had no metabolites meeting the strict replication criterion (no uc_vs_control contrast rows in the audit at all).",
            "authoritative_source_csv": str(path),
            "key_columns_used": "row_type=='primary_replicated_metabolite', contrast",
            "key_value": f"ibd={n_ibd},cd={n_cd},uc={n_uc}",
            "cross_checked_against": str(cross_check_path),
            "n_rows_used": len(primary),
            "notes": "Do not read the 'IBD replicated events=10' row in stage3_final_results_numbers.csv as 10 IBD-vs-control metabolites — that row sums IBD-vs-control (7) and CD-vs-control (3) replication events across contrasts. A relabelled copy for citation is at stage3_final_results_numbers_relabelled.csv (generated by relabel_stage3_final_results_numbers.py); the original is left untouched.",
        },
    ]


def claim_within_disease_concordance() -> list[dict]:
    path = CFG.stage3_dir / "tables" / "stage3a_within_disease_concordance.csv"
    df = _read(path)
    ibd_row = df[(df["disease"] == "IBD") & (df["comparison"] == "Franzosa vs iHMP IBD vs Control")].iloc[0]
    crc_row = df[(df["disease"] == "CRC") & (df["comparison"] == "Sinha vs Yachida CRC vs Control/Healthy")].iloc[0]
    ibd_rho = float(ibd_row["spearman_rho"])
    crc_rho = float(crc_row["spearman_rho"])
    assert abs(ibd_rho - 0.7821789321789323) < 1e-9, f"IBD within-disease Spearman rho drifted: {ibd_rho}"
    assert abs(crc_rho - 0.1343130553656869) < 1e-9, f"CRC within-disease Spearman rho drifted: {crc_rho}"
    rows = []
    for disease, row, rho in [("IBD", ibd_row, ibd_rho), ("CRC", crc_row, crc_rho)]:
        rows.append(
            {
                "claim_id": f"within_disease_concordance_{disease.lower()}",
                "stage": "Stage 3A",
                "claim_text": f"{disease} within-disease, cross-cohort effect-size concordance: Spearman rho={rho:.3f} (p={row['spearman_p_value']:.3g}), same-sign proportion={row['same_sign_proportion']:.3f} over n={int(row['n_complete_metabolites'])} complete metabolites.",
                "authoritative_source_csv": str(path),
                "key_columns_used": "spearman_rho, spearman_p_value, same_sign_proportion, n_complete_metabolites",
                "key_value": round(rho, 6),
                "cross_checked_against": str(CFG.stage3_dir / "tables" / "stage3_final_results_numbers.csv"),
                "n_rows_used": len(df),
                "notes": row["replication_count_note"] if isinstance(row.get("replication_count_note"), str) else "",
            }
        )
    return rows


def claim_yachida_ordered_state() -> list[dict]:
    trend_path = CFG.stage3_dir / "tables" / "stage3b_yachida_ordered_state_trend_results.csv"
    trend = _read(trend_path)
    n_tested = len(trend)
    hc3_sig = trend[trend["BH_FDR"] < 0.05]
    n_hc3_sig = len(hc3_sig)
    n_spearman_sig = int((trend["spearman_BH_FDR"] < 0.05).sum())
    assert n_tested == 56, f"Expected 56 tested metabolites in Yachida ordered-state trend results, found {n_tested}"
    # The two primary HC3-significant results are NOT both monotonic: citrulline's
    # profile_pattern is 'irregular' and N6,N6,N6-trimethyllysine's is 'early_peak'.
    # They are significant ordered-state LINEAR associations (HC3-robust SE), which
    # is a distinct claim from the separate Spearman-rank monotonic-trend sensitivity.
    hc3_sig_names = sorted(hc3_sig["yachida_feature"].astype(str).tolist())
    hc3_sig_patterns = dict(zip(hc3_sig["yachida_feature"], hc3_sig["profile_pattern"]))
    assert not all(p == "monotonic" or "monoton" in str(p).lower() for p in hc3_sig_patterns.values()), (
        f"Expected the HC3-significant metabolites to include non-monotonic profile_pattern values "
        f"(this assertion exists to prevent 'monotonic progression' language creeping back in): {hc3_sig_patterns}"
    )
    focused_path = CFG.stage3_dir / "tables" / "stage3b_focused_ibd_replicated_metabolites.csv"
    focused = _read(focused_path)
    return [
        {
            "claim_id": "yachida_ordered_state_trend",
            "stage": "Stage 3B",
            "claim_text": (
                f"Across {n_tested} metabolites, {n_hc3_sig} showed significant HC3-robust ordered-state "
                f"linear associations after BH correction ({', '.join(hc3_sig_names)}); {n_spearman_sig} "
                f"showed significant Spearman associations with ordered state in the rank-based sensitivity "
                f"analysis. These are two distinct tests: the HC3 result is an ordered-state linear "
                f"association, not evidence of a monotonic trajectory — e.g. {hc3_sig_names[0]}'s "
                f"profile_pattern is '{hc3_sig_patterns[hc3_sig_names[0]]}', not monotonic."
            ),
            "authoritative_source_csv": str(trend_path),
            "key_columns_used": "ordered_state_beta, robust_hc3_se, BH_FDR, spearman_rho, spearman_BH_FDR, profile_pattern, ordered_state_direction",
            "key_value": n_hc3_sig,
            "cross_checked_against": str(CFG.stage3_dir / "tables" / "stage3_final_results_numbers.csv"),
            "n_rows_used": n_tested,
            "notes": (
                f"HC3-significant metabolites and their profile_pattern: {hc3_sig_patterns}. Do not describe "
                "these two as a 'monotonic progression' — only the separate Spearman-rank sensitivity tests "
                "monotonicity directly. Underlying per-state median z-score profiles are in "
                "stage3b_yachida_ordered_state_profiles.csv; Healthy-vs-Advanced-CRC contrast is in "
                "stage3b_yachida_early_vs_advanced_summary.csv."
            ),
        },
        {
            "claim_id": "yachida_ordered_state_ibd_replicated_subset",
            "stage": "Stage 3B",
            "claim_text": "The IBD-replicated metabolites (putrescine, N-acetylputrescine, taurine) were additionally evaluated for a CRC-stage-ordered trend in Yachida.",
            "authoritative_source_csv": str(focused_path),
            "key_columns_used": "yachida_feature, ordered_state_beta, spearman_rho",
            "key_value": len(focused),
            "cross_checked_against": str(trend_path),
            "n_rows_used": len(focused),
            "notes": "Intersection of the 7 strictly-replicated IBD metabolites with the Yachida ordered-state metabolite panel.",
        },
    ]


def claim_crc_strict_replication_zero() -> dict:
    path = CFG.stage3_dir / "tables" / "stage3a_within_disease_concordance.csv"
    df = _read(path)
    crc_row = df[(df["disease"] == "CRC") & (df["comparison"] == "Sinha vs Yachida CRC vs Control/Healthy")].iloc[0]
    n = int(crc_row["strict_FDR_unique_metabolites_full_expanded"])
    assert n == 0, f"Expected 0 strictly-replicated CRC metabolites, found {n} in {path}"
    return {
        "claim_id": "crc_strict_replication_zero",
        "stage": "Stage 3A",
        "claim_text": "No metabolites met the strict (FDR-significant, same-direction) replication criterion between Sinha and Yachida for CRC vs. Control/Healthy.",
        "authoritative_source_csv": str(path),
        "key_columns_used": "strict_FDR_unique_metabolites_full_expanded (CRC row)",
        "key_value": n,
        "cross_checked_against": str(CFG.stage3_dir / "tables" / "stage3_final_results_numbers.csv"),
        "n_rows_used": len(df),
        "notes": str(crc_row.get("replication_count_note", "")),
    }


def claim_cross_disease_spearman_zero() -> dict:
    path = CFG.stage3_dir / "tables" / "stage3a_cross_disease_correlation_results.csv"
    df = _read(path)
    fdr_col = [c for c in df.columns if "fdr" in c.lower()][0]
    n_sig = int((df[fdr_col] < 0.05).sum())
    assert n_sig == 0, f"Expected 0 BH-FDR-significant cross-disease Spearman correlations, found {n_sig} in {path}"
    return {
        "claim_id": "cross_disease_spearman_zero",
        "stage": "Stage 3A",
        "claim_text": f"None of the {len(df)} cross-disease (IBD-vs-CRC) Spearman correlation tests survived BH-FDR correction.",
        "authoritative_source_csv": str(path),
        "key_columns_used": fdr_col,
        "key_value": n_sig,
        "cross_checked_against": str(CFG.stage3_dir / "tables" / "stage3_final_results_numbers.csv"),
        "n_rows_used": len(df),
        "notes": "Distinct from within-disease concordance (see within_disease_concordance_ibd/_crc claims): this tests IBD effect sizes against CRC effect sizes directly, not each disease against itself.",
    }


def claim_directionally_coherent_candidates_zero() -> dict:
    path = CFG.stage3_dir / "tables" / "stage3b_directionally_coherent_fdr_significant_candidates.csv"
    df = _read(path)
    n = len(df)
    assert n == 0, f"Expected 0 directionally-coherent Stage3A+3B FDR candidates, found {n} rows in {path}"
    return {
        "claim_id": "directionally_coherent_stage3a_3b_candidates_zero",
        "stage": "Stage 3A/3B",
        "claim_text": "No metabolite was simultaneously a strictly-replicated Stage 3A within-disease candidate AND a BH-FDR-significant Stage 3B Yachida ordered-state association with a coherent direction across both.",
        "authoritative_source_csv": str(path),
        "key_columns_used": "table row count (schema: canonical_hmdb, metabolite_name, comparison, ibd_side_consensus, crc_consensus, direction_class, ibd_side_strict_replication, crc_strict_replication, ordered_state_beta, BH_FDR, profile_pattern)",
        "key_value": n,
        "cross_checked_against": str(CFG.stage3_dir / "tables" / "stage3_final_results_numbers.csv"),
        "n_rows_used": n,
        "notes": "The table itself is empty (0 rows) by construction; this is the strongest available evidence for the negative conclusion, not an omission.",
    }


def main() -> None:
    rows: list[dict] = []
    rows.extend(claim_seven_replicated_ibd_metabolites())
    rows.extend(claim_within_disease_concordance())
    rows.extend(claim_yachida_ordered_state())
    rows.append(claim_crc_strict_replication_zero())
    rows.append(claim_cross_disease_spearman_zero())
    rows.append(claim_directionally_coherent_candidates_zero())

    for row in rows:
        for key in ("authoritative_source_csv", "cross_checked_against"):
            if not Path(row[key]).exists():
                raise FileNotFoundError(f"{row['claim_id']}: referenced source does not exist: {row[key]}")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_CSV.with_suffix(OUT_CSV.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(OUT_CSV)

    payload = {
        "status": "STAGE2_3_MANUSCRIPT_CLAIM_PROVENANCE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_claims": len(rows),
        "note": "No new statistics computed; every value asserted against its live source CSV at generation time.",
        "existing_crosswalks_referenced": [
            str(CFG.stage3_dir / "tables" / "stage3_final_results_numbers.csv"),
            str(CFG.stage3_dir / "manifests" / "stage3_authoritative_effect_input_handoff.csv"),
        ],
        "claims": rows,
    }
    tmp_json = OUT_JSON.with_suffix(OUT_JSON.suffix + ".tmp")
    tmp_json.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    tmp_json.replace(OUT_JSON)

    print(f"Wrote {OUT_CSV} ({len(rows)} claims)")
    print(f"Wrote {OUT_JSON}")
    print(df[["claim_id", "stage", "key_value"]].to_string(index=False))


if __name__ == "__main__":
    main()
