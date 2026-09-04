# =============================================================================
# command to run (requires results_r1b_r2_r3_composition_runs.json to
# already exist on the volume -- produced by
# run_r1b_r2_r3_composition_runs.py::main):
#   modal run run_c3_r2_jackknife.py::main
# run_c3_r2_jackknife.py
#
# C3 -- JACKKNIFE SENSITIVITY OF THE R2 PRE-CAL MEAN.
# Pre-registered in DECISIONS.md's "C3 / C4 -- robustness checks on the
# accepted R2 signal" section, BEFORE this script runs.
#
# NOT a new classifier run. C1/C2 (DECISIONS.md, ledger L040/L041) accepted
# R2's pre_cal_balanced as a genuine, subject-generalizable signal (real
# value outside C1's shuffled-label 95% CI on both methods -- see
# REAL_R2_MEAN/NULL_95CI_UPPER_BOUND below for the live values this run
# actually uses). This checks whether that mean is being driven by one or
# two influential subjects -- a leave-one-subject-out (LOO) jackknife over
# the 29 per-fold pre_cal_balanced accuracy values ALREADY produced by
# run_r1b_r2_r3_composition_runs.py's R2_search_vs_memorize_encode_only run
# (read directly from results_r1b_r2_r3_composition_runs.json on the
# volume, no retraining).
#
# PRE-REGISTERED REFERENCE VALUES (DECISIONS.md, fixed before running):
#   real R2 mean, null 95% CI upper boundary (from C1's actual run), and
#   half-gap threshold = (real R2 mean - null 95% CI upper boundary) / 2.
#   GATE C4: both inputs are now read at runtime from their authoritative
#   source files (R2 mean from results_r1b_r2_r3_composition_runs.json,
#   already loaded below for the jackknife itself; null CI upper bound from
#   results_c1_r2_shuffled_label_control.json, exactly as C4 does) -- never
#   hardcoded here. Hardcoding either one here is exactly how the GATE
#   C2/C3 staleness bugs happened (N11's untraceable 0.5223, and the
#   0.5737 left in a C4 f-string); a runtime read cannot go stale.
#
# PRE-REGISTERED VERDICT RULE: if no single-fold exclusion drops the LOO
#   mean below the null 95% CI upper boundary -> not an artifact of one/two
#   influential subjects, reported as an added robustness line. If any
#   exclusion does -> name the fold(s), report the mean with/without it,
#   plainly, not a footnote. Separately (a reporting flag, not a second
#   gate): any exclusion whose LOO mean shifts from the full-sample mean by
#   more than the half-gap threshold is flagged as individually influential
#   regardless of which side of the CI boundary it lands on.
#
# Usage: modal run run_c3_r2_jackknife.py::main
# =============================================================================

import modal

app    = modal.App("bci-c3-r2-jackknife")
volume = modal.Volume.from_name("eeg-data-vol")

RESULTS_JSON_PATH  = "/data/results_r1b_r2_r3_composition_runs.json"
C1_RESULTS_JSON    = "/data/results_c1_r2_shuffled_label_control.json"
OUTPUT_JSON        = "/data/results_c3_r2_jackknife.json"
VOLUME_PATH        = "/data"

# GATE C4: NULL_95CI_UPPER_BOUND and REAL_R2_MEAN were hardcoded literals
# copied from C1's/R2's output at the time each gate ran -- the same
# failure mode that produced N11 (an untraceable 0.5223) and then the GATE
# C2/C3 staleness bugs (a stale 0.5737 surviving a mean update). Both are
# now read at runtime inside run_c3_jackknife() from their source JSONs on
# the volume, with a hard assertion that the file/key exists -- no fallback
# literal, no default value. HALF_GAP_THRESHOLD is computed from those two
# runtime values, per DECISIONS.md's formula (not an independent literal).

# GATE C5: thread-count pinning applied for consistency across all six
# pipeline scripts (see run_c4_high_res_shuffled_label_control.py's image
# comment for why this must be baked in via .env(), not set in the function
# body). C3 itself does no BLAS-heavy computation (pure np.mean/np.sum over
# already-persisted scalars), so it was not implicated in the confirmed
# non-determinism -- pinned anyway so nothing here can drift either.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4")
    .env({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    })
)


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=600, memory=2048)
def run_c3_jackknife(git_short_hash: str = "nogit"):
    import json
    import os
    import numpy as np

    # GATE C4 de-literalisation: NULL_95CI_UPPER_BOUND read at runtime from
    # C1's actual output, exactly as run_c4_high_res_shuffled_label_control.py
    # already does -- no fallback literal, no default value, fail loud if
    # either the file or the key is missing.
    assert os.path.exists(C1_RESULTS_JSON), (
        f"[C3 FATAL] C1's results file not found at {C1_RESULTS_JSON} -- C3 depends on C1's actual "
        "null 95% CI upper bound and has no fallback literal to fall back on. Run C1 first."
    )
    with open(C1_RESULTS_JSON) as f:
        c1_payload = json.load(f)
    assert "null_distribution_percentile_95ci" in c1_payload, (
        f"[C3 FATAL] {C1_RESULTS_JSON} is missing key 'null_distribution_percentile_95ci' -- "
        "cannot derive NULL_95CI_UPPER_BOUND."
    )
    NULL_95CI_UPPER_BOUND = c1_payload["null_distribution_percentile_95ci"][1]

    with open(RESULTS_JSON_PATH) as f:
        results_payload = json.load(f)

    r2 = results_payload["R2_search_vs_memorize_encode_only"]
    fold_results = r2["fold_results"]
    n_folds = len(fold_results)

    pre_cal_bal = np.array([fr["pre_cal_plausibility"]["balanced_accuracy"] for fr in fold_results])
    test_subjects = [fr["test_subject"] for fr in fold_results]
    # N16 instrumentation: surface per-fold N and per-class correct/total
    # (persisted upstream by run_r1b_r2_r3_composition_runs.py) alongside the
    # jackknife statistics, so Table S1-style claims trace to this artifact
    # directly rather than requiring a second lookup into the R2 JSON.
    n_test_per_fold = [fr["n_test"] for fr in fold_results]
    per_class_per_fold = [fr.get("pre_cal_per_class_counts") for fr in fold_results]

    full_mean_recomputed = float(np.mean(pre_cal_bal))
    reported_mean = r2["pre_calibration_balanced_accuracy_mean"]
    sanity_diff = abs(full_mean_recomputed - reported_mean)
    print(f"n_folds={n_folds} | full_mean_recomputed_from_fold_results={full_mean_recomputed:.6f} | "
          f"reported_summary_mean={reported_mean:.6f} | diff={sanity_diff:.8f}")
    assert sanity_diff < 1e-6, (
        f"[C3 SANITY FAIL] the mean recomputed from fold_results ({full_mean_recomputed:.6f}) does not match "
        f"the summary's own pre_calibration_balanced_accuracy_mean ({reported_mean:.6f}) -- the fold-level "
        "data and the summary statistic are inconsistent. Halting before jackknife interpretation."
    )

    # GATE C4 de-literalisation: REAL_R2_MEAN is the value just verified above
    # (independently recomputed from fold_results, not a copied literal) --
    # this is what HALF_GAP_THRESHOLD's formula uses, per DECISIONS.md.
    REAL_R2_MEAN = full_mean_recomputed
    HALF_GAP_THRESHOLD = (REAL_R2_MEAN - NULL_95CI_UPPER_BOUND) / 2.0
    print(f"Runtime reference values: REAL_R2_MEAN={REAL_R2_MEAN}, NULL_95CI_UPPER_BOUND={NULL_95CI_UPPER_BOUND}, "
          f"HALF_GAP_THRESHOLD={HALF_GAP_THRESHOLD}")

    total = pre_cal_bal.sum()
    loo_means = []
    for i in range(n_folds):
        loo_mean = float((total - pre_cal_bal[i]) / (n_folds - 1))
        shift = loo_mean - full_mean_recomputed
        influential_flag = abs(shift) > HALF_GAP_THRESHOLD
        below_ci_flag = loo_mean < NULL_95CI_UPPER_BOUND
        loo_means.append({
            "excluded_subject": str(test_subjects[i]),
            "excluded_fold_pre_cal_balanced": float(pre_cal_bal[i]),
            "excluded_fold_n_test": n_test_per_fold[i],
            "excluded_fold_per_class_counts": per_class_per_fold[i],
            "loo_mean": loo_mean,
            "shift_from_full_mean": shift,
            "individually_influential_flag": bool(influential_flag),
            "drops_below_null_ci_upper_bound": bool(below_ci_flag),
        })
        flag_str = " <-- INFLUENTIAL" if influential_flag else ""
        below_str = f" <-- DROPS BELOW {NULL_95CI_UPPER_BOUND:.4f}" if below_ci_flag else ""
        print(f"  excl sub-{test_subjects[i]}: fold_val={pre_cal_bal[i]:.4f} loo_mean={loo_mean:.4f} "
              f"shift={shift:+.4f}{flag_str}{below_str}")

    loo_mean_values = np.array([r["loo_mean"] for r in loo_means])
    loo_min, loo_max = float(loo_mean_values.min()), float(loo_mean_values.max())
    any_below_threshold = any(r["drops_below_null_ci_upper_bound"] for r in loo_means)
    influential_folds = [r["excluded_subject"] for r in loo_means if r["individually_influential_flag"]]

    if not any_below_threshold:
        verdict = (f"NOT AN ARTIFACT OF ONE OR TWO INFLUENTIAL SUBJECTS -- no single-fold exclusion drops the "
                    f"LOO mean below the pre-registered {NULL_95CI_UPPER_BOUND} threshold. LOO mean range: "
                    f"[{loo_min:.4f}, {loo_max:.4f}]. Reported as an added robustness line.")
    else:
        below_folds = [r["excluded_subject"] for r in loo_means if r["drops_below_null_ci_upper_bound"]]
        verdict = (f"NOT ROBUST TO ONE SUBJECT -- excluding sub-{below_folds} drops the LOO mean below the "
                    f"pre-registered {NULL_95CI_UPPER_BOUND} threshold. LOO mean range: [{loo_min:.4f}, "
                    f"{loo_max:.4f}]. Must be reported plainly alongside the C1 verdict, not as a footnote.")

    if influential_folds:
        verdict += (f" Separately flagged as individually influential (shift > {HALF_GAP_THRESHOLD:.4f} "
                     f"regardless of the {NULL_95CI_UPPER_BOUND:.4f} threshold): sub-{influential_folds}.")
    else:
        verdict += f" No fold individually shifts the mean by more than the {HALF_GAP_THRESHOLD:.4f} half-gap threshold."

    print(f"\n{'='*70}\n  C3 PRE-REGISTERED VERDICT\n{'='*70}\n  {verdict}")

    results_payload = {
        "n_folds": n_folds,
        "full_mean_recomputed": full_mean_recomputed,
        "reported_summary_mean": reported_mean,
        "loo_mean_range": [loo_min, loo_max],
        "null_95ci_upper_bound_used": NULL_95CI_UPPER_BOUND,
        "any_exclusion_drops_below_null_ci_upper_bound": bool(any_below_threshold),
        "individually_influential_folds": influential_folds,
        "half_gap_threshold": HALF_GAP_THRESHOLD,
        "loo_results": loo_means,
        "verdict": verdict,
    }
    # GATE C5 STEP 3: non-overwriting artifact -- see run_c1's identical
    # comment for why (Modal Volumes have no version history).
    import datetime
    utc_ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    stamped_path = OUTPUT_JSON.replace(".json", f"_{utc_ts}_{git_short_hash}.json")
    with open(stamped_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_payload, f, indent=2)
    volume.commit()
    print(f"Saved (immutable): {stamped_path}")
    print(f"Saved (convenience pointer, overwritten): {OUTPUT_JSON}")

    results_payload["stamped_output_path"] = stamped_path
    return results_payload


@app.local_entrypoint()
def main():
    import subprocess, os
    try:
        git_short_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
        ).decode().strip()
    except Exception:
        git_short_hash = "nogit"
    print("C3 -- jackknife sensitivity of the R2 pre_cal_balanced mean (leave-one-subject-out).")
    print("Pre-registered in DECISIONS.md's 'C3 / C4' section BEFORE this run.")
    print("Reference values (real R2 mean, null 95% CI upper bound, half-gap threshold) are now read "
          "at runtime from C1's/R2's own output JSONs -- printed by the remote function once loaded.")
    print(f"git_short_hash for this run's stamped output filename: {git_short_hash}\n")
    result = run_c3_jackknife.remote(git_short_hash=git_short_hash)
    print("\nRESULT:")
    for k, v in result.items():
        if k != "loo_results":
            print(f"  {k}: {v}")
