# =============================================================================
# command to run (requires results_r1b_r2_r3_composition_runs.json,
# results_c3_r2_jackknife.json, and results_c4_high_res_shuffled_label_
# control.json to already exist on the volume):
#   modal run run_d1_transcription_dump.py::main
# run_d1_transcription_dump.py
#
# D1 -- TRANSCRIPTION CLEANUP. Pure read-only dump, no retraining, no new
# statistical test. Pre-registered in DECISIONS.md's "D1 / D2" section.
#
# Prints, in one pass, the five values RESULTS_LEDGER.md's L046 (H1
# consolidation audit) flagged as missing from the ledger:
#   1. R1(b) encode-vs-test: pre_cal + post_cal, raw AND balanced.
#   2. R2 search-vs-memorize (encode-only): post_calibration_balanced_
#      accuracy_mean (pre_cal is already on record, L039).
#   3. R3 (lure-removed): full pre_cal/post_cal, raw AND balanced.
#   4. C3's full 29-row leave-one-out jackknife table.
#   5. C4's exact null SD, both 95% CI bounds (percentile + normal-approx),
#      and BOTH p-value conventions (raw and add-one-corrected) side by
#      side, plus the two consistency-check booleans.
#
# Usage: modal run run_d1_transcription_dump.py::main
# =============================================================================

import modal

app    = modal.App("bci-d1-transcription-dump")
volume = modal.Volume.from_name("eeg-data-vol")

R1B_R2_R3_JSON = "/data/results_r1b_r2_r3_composition_runs.json"
C3_JSON         = "/data/results_c3_r2_jackknife.json"
C4_JSON         = "/data/results_c4_high_res_shuffled_label_control.json"
VOLUME_PATH     = "/data"

image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=300, memory=1024)
def dump_all():
    import json

    with open(R1B_R2_R3_JSON) as f:
        r1b_r2_r3 = json.load(f)
    with open(C3_JSON) as f:
        c3 = json.load(f)
    with open(C4_JSON) as f:
        c4 = json.load(f)

    r1b = r1b_r2_r3["R1b_encode_vs_test"]
    r2 = r1b_r2_r3["R2_search_vs_memorize_encode_only"]
    r3 = r1b_r2_r3["R3_lure_removed"]

    out = {
        "R1b": {
            "pre_calibration_accuracy_mean": r1b["pre_calibration_accuracy_mean"],
            "post_calibration_accuracy_mean": r1b["post_calibration_accuracy_mean"],
            "pre_calibration_balanced_accuracy_mean": r1b["pre_calibration_balanced_accuracy_mean"],
            "post_calibration_balanced_accuracy_mean": r1b["post_calibration_balanced_accuracy_mean"],
            "pre_calibration_auc_mean": r1b.get("pre_calibration_auc_mean"),
            "post_calibration_auc_mean": r1b.get("post_calibration_auc_mean"),
            "n_folds": r1b["n_folds"],
        },
        "R2": {
            "pre_calibration_accuracy_mean": r2["pre_calibration_accuracy_mean"],
            "post_calibration_accuracy_mean": r2["post_calibration_accuracy_mean"],
            "pre_calibration_balanced_accuracy_mean": r2["pre_calibration_balanced_accuracy_mean"],
            "post_calibration_balanced_accuracy_mean": r2["post_calibration_balanced_accuracy_mean"],
            "n_folds": r2["n_folds"],
        },
        "R3": {
            "pre_calibration_accuracy_mean": r3["pre_calibration_accuracy_mean"],
            "post_calibration_accuracy_mean": r3["post_calibration_accuracy_mean"],
            "pre_calibration_balanced_accuracy_mean": r3["pre_calibration_balanced_accuracy_mean"],
            "post_calibration_balanced_accuracy_mean": r3["post_calibration_balanced_accuracy_mean"],
            "n_folds": r3["n_folds"],
        },
        "R1b_verdict": r1b_r2_r3.get("R1b_verdict"),
        "C3_jackknife_full_table": c3["loo_results"],
        "C3_summary": {
            "any_exclusion_drops_below_0.5223": c3["any_exclusion_drops_below_0.5223"],
            "individually_influential_folds": c3["individually_influential_folds"],
            "loo_mean_range": c3["loo_mean_range"],
        },
        "C4_exact": {
            "null_distribution_mean": c4["null_distribution_mean"],
            "null_distribution_sd": c4["null_distribution_sd"],
            "null_distribution_percentile_95ci": c4["null_distribution_percentile_95ci"],
            "null_distribution_normal_approx_95ci": c4["null_distribution_normal_approx_95ci"],
            "p_value_raw": c4["p_value_raw"],
            "p_value_add_one_corrected": c4["p_value_add_one_corrected"],
            "n_shuffles_at_least_as_extreme": c4["n_shuffles_at_least_as_extreme"],
            "reproduces_c1_first_30_exactly": c4["reproduces_c1_first_30_exactly"],
            "agrees_with_c1_on_ci_side": c4["agrees_with_c1_on_ci_side"],
            "consistency_ok": c4["consistency_ok"],
        },
    }

    print("=" * 78)
    print("D1 TRANSCRIPTION DUMP")
    print("=" * 78)
    print("\n--- R1(b) encode-vs-test ---")
    for k, v in out["R1b"].items():
        print(f"  {k}: {v}")
    print(f"  verdict: {out['R1b_verdict']}")

    print("\n--- R2 search-vs-memorize (encode-only) ---")
    for k, v in out["R2"].items():
        print(f"  {k}: {v}")

    print("\n--- R3 (lure-removed) ---")
    for k, v in out["R3"].items():
        print(f"  {k}: {v}")

    print("\n--- C3 jackknife summary ---")
    for k, v in out["C3_summary"].items():
        print(f"  {k}: {v}")
    print("  full 29-row table:")
    for row in out["C3_jackknife_full_table"]:
        print(f"    excl sub-{row['excluded_subject']}: fold_val={row['excluded_fold_pre_cal_balanced']:.4f} "
              f"loo_mean={row['loo_mean']:.4f} shift={row['shift_from_full_mean']:+.4f} "
              f"influential={row['individually_influential_flag']} below_ci={row['drops_below_null_ci_upper_bound']}")

    print("\n--- C4 exact numbers ---")
    for k, v in out["C4_exact"].items():
        print(f"  {k}: {v}")

    return out


@app.local_entrypoint()
def main():
    print("D1 -- transcription dump: R1(b)/R2/R3 full numbers, C3's 29-row table, C4's exact SD/CI/p-split.")
    print("Pure read-only, no retraining. Pre-registered in DECISIONS.md's 'D1 / D2' section.\n")
    result = dump_all.remote()
    print("\n(structured result also returned above for copy-paste into RESULTS_LEDGER.md)")
