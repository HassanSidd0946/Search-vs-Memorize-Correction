# =============================================================================
# command to run:
#   modal run scripts/drift_c_posthoc_analysis.py::main
# scripts/drift_c_posthoc_analysis.py
#
# F-DRIFT-C POST-HOC TREND ANALYSIS (added 2026-08-20, DECISIONS.md /
# RESULTS_LEDGER.md L016)
#
# WHY THIS SCRIPT EXISTS:
#   F-DRIFT-C's PRE-REGISTERED interpretation rule ("monotone rise from
#   chance toward ~0.64") was MIS-SPECIFIED: strict pointwise monotonicity
#   across 7 noisy single-seed accuracy values is not robust to small
#   (0.007-0.011) wiggles, and the real run produced exactly such wiggles
#   at k=2 and k=25 despite the underlying pattern being a clear
#   threshold/step function (flat at chance for separations <=116s, a
#   step to ~0.6358 at 232s, plateauing at ~0.6413 by 465s).
#
#   Per explicit instruction, this mis-specification is disclosed, NOT
#   silently corrected: the original strict-monotonicity verdict (NOT MET)
#   stands on the record exactly as computed. This script adds a SEPARATE,
#   explicitly POST-HOC analysis on top of the same already-completed
#   F-DRIFT-C data (no new LOSO training, pure post-processing of
#   results_condition4_drift_control_c_dose_response.json):
#     (b) Spearman rank correlation between seconds-of-separation and
#         combined pre_cal accuracy across the 7 k-values, with a
#         subject-level bootstrap CI on rho.
#     (c) A two-group paired contrast: low-separation k's {1,2,5,10,25}
#         (<=116s) vs. high-separation k's {50,100} (>=232s), per-subject
#         paired Wilcoxon signed-rank test.
#
#   Required disclosure language (verbatim, per instruction): "our
#   pre-registered criterion was mis-specified; we report both the
#   original verdict and a revised trend analysis." This script's output
#   JSON carries that string verbatim in its "disclosure" field, and both
#   the original verdict and this revised analysis must always be quoted
#   TOGETHER wherever F-DRIFT-C is cited -- never one without the other.
#
# INPUT: reads the EXISTING results_condition4_drift_control_c_dose_response.json
#        from the volume (written by run_step4_drift_control_c_dose_response.py).
#        Does not touch or re-run any LOSO fold.
#
# Usage: modal run scripts/drift_c_posthoc_analysis.py::main
# =============================================================================

import modal

app    = modal.App("bci-condition4-drift-control-c-posthoc")
volume = modal.Volume.from_name("eeg-data-vol")

VOLUME_PATH = "/data"
INPUT_JSON  = "/data/results_condition4_drift_control_c_dose_response.json"
OUTPUT_JSON = "/data/results_condition4_drift_control_c_posthoc.json"

K_SWEEP        = [1, 2, 5, 10, 25, 50, 100]
LOW_SEP_KS     = [1, 2, 5, 10, 25]     # <=116s
HIGH_SEP_KS    = [50, 100]              # >=232s
N_BOOTSTRAP    = 2000
BOOTSTRAP_SEED = 20260820

DISCLOSURE_TEXT = (
    "our pre-registered criterion was mis-specified; we report both the "
    "original verdict and a revised trend analysis."
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy<2", "scipy==1.14.1")
)


@app.function(image=image, cpu=1.0, volumes={VOLUME_PATH: volume}, timeout=600)
def run_posthoc():
    import json
    import numpy as np
    from scipy.stats import spearmanr, wilcoxon

    with open(INPUT_JSON) as f:
        payload = json.load(f)
    sweep = payload["sweep_results"]

    # -------------------------------------------------------------------
    # Build per-subject, per-k combined pre_cal accuracy: average of that
    # subject's search_only and memorize_only pre_calibration_acc at k
    # (each subject appears exactly once per class per k -- LOSO).
    # -------------------------------------------------------------------
    per_subject_k = {}   # subject -> {k: combined_pre_cal}
    seconds_at_k = {}
    for k in K_SWEEP:
        entry = sweep[str(k)]
        seconds_at_k[k] = float(entry["seconds_at_k"])
        by_subject = {}
        for cls_key in ("search_only", "memorize_only"):
            for rec in entry[cls_key]["fold_results"]:
                sub = rec["test_subject"]
                by_subject.setdefault(sub, []).append(rec["pre_calibration_acc"])
        for sub, vals in by_subject.items():
            assert len(vals) == 2, (
                f"subject {sub} at k={k} has {len(vals)} pre_cal values, expected 2 "
                f"(one per real class) -- input JSON structure changed unexpectedly"
            )
            per_subject_k.setdefault(sub, {})[k] = float(np.mean(vals))

    subjects = sorted(per_subject_k.keys())
    n_subjects = len(subjects)
    assert n_subjects == 29, f"expected 29 subjects, found {n_subjects}"
    for sub in subjects:
        assert set(per_subject_k[sub].keys()) == set(K_SWEEP), (
            f"subject {sub} missing k-values: {set(K_SWEEP) - set(per_subject_k[sub].keys())}"
        )

    # Matrix: n_subjects x 7, columns ordered by K_SWEEP.
    acc_matrix = np.array([[per_subject_k[sub][k] for k in K_SWEEP] for sub in subjects])
    seconds_vec = np.array([seconds_at_k[k] for k in K_SWEEP])

    # -------------------------------------------------------------------
    # (b) Spearman rank correlation, point estimate on the 7-point
    #     cross-subject-mean curve, plus a subject-level bootstrap CI.
    # -------------------------------------------------------------------
    mean_curve = acc_matrix.mean(axis=0)
    rho_point, p_two_sided = spearmanr(seconds_vec, mean_curve)
    # One-sided p for H1: rho > 0 (accuracy rises with separation).
    p_one_sided = p_two_sided / 2.0 if rho_point > 0 else 1.0 - p_two_sided / 2.0

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_rhos = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        resample_idx = rng.integers(0, n_subjects, size=n_subjects)
        resample_curve = acc_matrix[resample_idx].mean(axis=0)
        r, _ = spearmanr(seconds_vec, resample_curve)
        boot_rhos[b] = r
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])

    spearman_result = {
        "seconds_at_k": {str(k): seconds_at_k[k] for k in K_SWEEP},
        "mean_pre_cal_at_k": {str(k): float(mean_curve[i]) for i, k in enumerate(K_SWEEP)},
        "rho": float(rho_point),
        "p_two_sided": float(p_two_sided),
        "p_one_sided_rho_gt_0": float(p_one_sided),
        "bootstrap_n": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_ci_95": [float(ci_lo), float(ci_hi)],
        "note": "bootstrap resamples SUBJECTS (with replacement, n=29), recomputes the "
                "7-point cross-subject-mean curve and its Spearman rho per resample; "
                "CI is the 2.5/97.5 percentile of the resulting rho distribution.",
    }

    # -------------------------------------------------------------------
    # (c) Two-group paired contrast: low-separation vs. high-separation
    #     k's, per subject, Wilcoxon signed-rank.
    # -------------------------------------------------------------------
    low_idx = [K_SWEEP.index(k) for k in LOW_SEP_KS]
    high_idx = [K_SWEEP.index(k) for k in HIGH_SEP_KS]
    low_group = acc_matrix[:, low_idx].mean(axis=1)    # per-subject mean over low-sep k's
    high_group = acc_matrix[:, high_idx].mean(axis=1)  # per-subject mean over high-sep k's

    stat_greater, p_greater = wilcoxon(high_group, low_group, alternative="greater")
    stat_two_sided, p_two_sided_wc = wilcoxon(high_group, low_group, alternative="two-sided")

    wilcoxon_result = {
        "low_sep_ks": LOW_SEP_KS, "high_sep_ks": HIGH_SEP_KS,
        "low_sep_seconds_range": [seconds_at_k[LOW_SEP_KS[0]], seconds_at_k[LOW_SEP_KS[-1]]],
        "high_sep_seconds_range": [seconds_at_k[HIGH_SEP_KS[0]], seconds_at_k[HIGH_SEP_KS[-1]]],
        "n_subjects_paired": n_subjects,
        "low_group_mean": float(low_group.mean()), "low_group_std": float(low_group.std()),
        "high_group_mean": float(high_group.mean()), "high_group_std": float(high_group.std()),
        "wilcoxon_statistic_greater": float(stat_greater),
        "p_one_sided_high_gt_low": float(p_greater),
        "wilcoxon_statistic_two_sided": float(stat_two_sided),
        "p_two_sided": float(p_two_sided_wc),
        "note": "paired per-subject: each subject contributes one low-sep mean "
                "(average pre_cal over k in {1,2,5,10,25}) and one high-sep mean "
                "(average pre_cal over k in {50,100}); alternative='greater' tests "
                "whether high-separation accuracy is stochastically greater than "
                "low-separation accuracy.",
    }

    # C3 plausibility assertions, printed next to the numbers.
    assert -1.0 <= rho_point <= 1.0, f"rho out of range: {rho_point}"
    assert 0.0 <= p_one_sided <= 1.0, f"p_one_sided out of range: {p_one_sided}"
    assert -1.0 <= ci_lo <= ci_hi <= 1.0, f"bootstrap CI out of range: [{ci_lo}, {ci_hi}]"
    assert 0.0 <= p_greater <= 1.0, f"wilcoxon p out of range: {p_greater}"
    assert wilcoxon_result["n_subjects_paired"] == 29

    results_payload = {
        "disclosure": DISCLOSURE_TEXT,
        "original_pre_registered_verdict": (
            "NOT MET -- strict pointwise monotonicity across the 7 single-seed "
            "pre_cal values failed at k=2 and k=25 (wiggles of 0.007-0.011), despite "
            "the underlying pattern being a clear threshold/step function (flat at "
            "chance for separations <=116s, step to ~0.6358 at 232s, plateau at "
            "~0.6413 by 465s). This original verdict is NOT superseded or deleted by "
            "the analysis below -- both must always be reported together."
        ),
        "post_hoc_status": "Both analyses below (spearman_correlation, two_group_contrast) "
                            "are POST-HOC -- they were NOT pre-registered before "
                            "run_step4_drift_control_c_dose_response.py ran, and were added "
                            "only after the strict-monotonicity criterion was found to be "
                            "mis-specified. They analyze the SAME already-completed dose-"
                            "response data (no new LOSO training).",
        "spearman_correlation": spearman_result,
        "two_group_contrast": wilcoxon_result,
        "input_source": INPUT_JSON,
    }

    print(f"\n{'='*70}\nF-DRIFT-C POST-HOC TREND ANALYSIS\n{'='*70}")
    print(f"DISCLOSURE: {DISCLOSURE_TEXT}\n")
    print(f"(a) Original pre-registered verdict: NOT MET (see disclosure above)")
    print(f"(b) Spearman rho={rho_point:.4f}, one-sided p={p_one_sided:.4f}, "
          f"95% bootstrap CI over subjects=[{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"(c) Low-sep (<=116s) mean={low_group.mean():.4f}, "
          f"High-sep (>=232s) mean={high_group.mean():.4f}, "
          f"Wilcoxon one-sided p(high>low)={p_greater:.4f}, two-sided p={p_two_sided_wc:.4f}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_payload, f, indent=2)
    volume.commit()
    print(f"\nSaved: {OUTPUT_JSON}")

    return results_payload


@app.local_entrypoint()
def main():
    result = run_posthoc.remote()
    import json
    print(json.dumps(result, indent=2))
