# =============================================================================
# scripts/variance_gap_all_subjects.py
#
# command to run:
#   modal run scripts/variance_gap_all_subjects.py::main
#
# F-OCULAR(d) — FRONTOPOLAR-VS-CENTRAL-PARIETAL VARIANCE GAP, ALL 29 SUBJECTS
# (AUDIT.md Phase 0.5 Priority 3d replication)
#
# WHY THIS SCRIPT EXISTS:
#   AUDIT.md Phase 0.5 Priority 3(d) found, on sub-01 alone (410 epochs,
#   205/205 class-balanced): a ~22.5% relative class-conditional variance
#   gap at 7 frontopolar channels (Search 159.4 uV^2 vs. Memorize 123.5
#   uV^2) but only a ~2.0% relative gap at 9 central-parietal channels
#   (Search 69.8 uV^2 vs. Memorize 68.4 uV^2) -- an order-of-magnitude
#   larger class-correlated difference concentrated almost entirely at the
#   ocular-proximal electrodes, directionally consistent with an ocular
#   (saccade/blink) confound (Search = visual search = more eye movement).
#   AUDIT.md explicitly flagged this as a single-subject computation that
#   "should be replicated across all 29 (or 30) subjects before being
#   treated as confirmed." This script is that replication.
#
# METHODOLOGY (re-derived to match Priority 3d's description; the original
#   Priority 3d numbers came from an inline, unsaved exploratory computation
#   during the audit, not from a checked-in script, so this is a faithful
#   re-implementation rather than a literal byte-for-byte reuse):
#   For each subject, for each channel group (frontopolar: Fp1, Fp2, AF7,
#   AF3, AFz, AF4, AF8; central-parietal: Cz, CPz, Pz, CP1, CP2, P1, P2, C3,
#   C4), for each class (0=Search, 1=Memorize):
#     1. Select that class's trials x that channel group's rows from the
#        subject's checkpointed epoched signal (uV, see unit note below).
#     2. Per channel: pool all (trial, time) samples for that class, compute
#        mean(|x|) and var(x) over the pooled array.
#     3. Average the per-channel statistic across the channels in the group
#        to get one number per (subject, group, class, statistic) cell --
#        matching Priority 3d's single-number-per-group-per-class table.
#   Relative gap = 100 * (Search - Memorize) / Memorize, matching Priority
#   3d's reporting convention (Search consistently higher in the sub-01
#   finding).
#
# CHANNEL-ORDER SAFETY: uses the same live sub-01 raw .vhdr verification as
#   F-OCULAR(a)/(c) (NOT the "column-order ASSUMED" layout the visualization
#   scripts use) -- see run_step4_matched_spatial_control_frontal_ablated.py
#   header for the full justification.
#
# UNITS: run_data_engine_on_modal.py's checkpoints store
#   `epochs.get_data(copy=True)`, which MNE returns in VOLTS (SI units), not
#   microvolts. Priority 3d's reported numbers (~6-9 for mean|amp|, ~60-160
#   for variance) are in the microvolt range typical of real scalp EEG, so
#   this script multiplies the checkpointed signal by 1e6 before computing
#   any statistic, to reproduce comparable units.
#
# DATA SOURCE: reads the per-subject epoched checkpoints already cached by
#   run_data_engine_on_modal.py under /data/checkpoints/{sub_tag}.npz on the
#   Modal volume -- NO re-download, NO re-preprocessing (this is a read-only
#   aggregation, hence AUDIT.md's "Low" cost estimate).
#
# STATUS: written and syntax-verified only, per AUDIT.md -- NOT YET EXECUTED.
#   Requires a real Modal account/volume; cannot be run from this sandbox.
# =============================================================================

import modal

app    = modal.App("bci-ocular-variance-gap-all-subjects")
volume = modal.Volume.from_name("eeg-data-vol")

VOLUME_MOUNT_PATH = "/data"
CHECKPOINT_DIR    = "/data/checkpoints"     # populated by run_data_engine_on_modal.py
PROBE_DIR         = "/data/_probe_openneuro_channel_order_variance_gap"
OUTPUT_JSON       = "/data/results_ocular_variance_gap_all_subjects.json"

NUM_SUBJECTS = 30   # sub-09 expected missing (truncated .eeg export, AUDIT.md D2)
N_CHANNELS   = 62

FRONTOPOLAR_CHANNELS      = ["Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8"]
CENTRAL_PARIETAL_CHANNELS = ["Cz", "CPz", "Pz", "CP1", "CP2", "P1", "P2", "C3", "C4"]

VOLTS_TO_MICROVOLTS = 1.0e6

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # scipy MUST be pinned alongside mne: unpinned scipy resolves to a
        # version that removed scipy.special.sph_harm (deprecated 1.15,
        # gone by 1.17), which mne==1.7.1 still imports at `import mne.io`
        # time -- crashes before any of our code runs. 1.14.1 matches the
        # already-proven-working pin in run_data_engine_on_modal.py and
        # run_step4_ica_cleaned_control.py's ::build stage.
        "numpy<2",
        "scipy==1.14.1",
        "mne==1.7.1",
        "openneuro-py==2024.2.0",
    )
)


@app.function(image=image, cpu=2.0, volumes={VOLUME_MOUNT_PATH: volume}, timeout=3600, memory=8192)
def compute_variance_gap_all_subjects():
    import os, glob, shutil, logging, json
    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("ocular-variance-gap")

    # =========================================================================
    # STEP 0: VERIFY THE TRUE CHANNEL ORDER (not assumed) — same approach as
    # F-OCULAR(a)/(c).
    # =========================================================================
    log.info("Verifying authoritative channel order from sub-01 raw .vhdr ...")
    import mne
    import openneuro

    mne.set_log_level("WARNING")
    os.makedirs(PROBE_DIR, exist_ok=True)
    openneuro.download(dataset="ds005189", target_dir=PROBE_DIR, include=["sub-01"])
    vhdr_files = glob.glob(os.path.join(PROBE_DIR, "**", "*.vhdr"), recursive=True)
    assert vhdr_files, f"Could not find sub-01 .vhdr under {PROBE_DIR} for channel-order verification."
    raw_probe = mne.io.read_raw_brainvision(vhdr_files[0], preload=False, verbose=False)
    verified_ch_names = list(raw_probe.info["ch_names"])
    shutil.rmtree(PROBE_DIR, ignore_errors=True)

    assert len(verified_ch_names) == N_CHANNELS, (
        f"Verified sub-01 raw channel count ({len(verified_ch_names)}) does not match "
        f"the expected {N_CHANNELS}. Channel order cannot be trusted — aborting."
    )
    needed = FRONTOPOLAR_CHANNELS + CENTRAL_PARIETAL_CHANNELS
    missing = [c for c in needed if c not in verified_ch_names]
    assert not missing, f"Channel group names not found in verified montage: {missing}"

    frontopolar_idx = [verified_ch_names.index(c) for c in FRONTOPOLAR_CHANNELS]
    central_parietal_idx = [verified_ch_names.index(c) for c in CENTRAL_PARIETAL_CHANNELS]
    log.info(f"Frontopolar indices ({FRONTOPOLAR_CHANNELS}): {frontopolar_idx}")
    log.info(f"Central-parietal indices ({CENTRAL_PARIETAL_CHANNELS}): {central_parietal_idx}")

    # =========================================================================
    # STEP 1: for each cached per-subject checkpoint, compute the four cells
    # (frontopolar/central-parietal x Search/Memorize) x 2 statistics.
    # =========================================================================
    def group_stats(X_uv, y, ch_idx, cls):
        """X_uv: (n_epochs, n_channels, n_times) in microvolts. Returns
        (mean_abs_amplitude, variance) averaged across ch_idx, for trials
        where y == cls, matching Priority 3d's one-number-per-group-per-class
        convention (see module docstring, step 2-3)."""
        mask = y == cls
        sub_X = X_uv[mask][:, ch_idx, :]   # (n_trials_cls, n_group_channels, n_times)
        per_channel_mean_abs = np.mean(np.abs(sub_X), axis=(0, 2))   # (n_group_channels,)
        per_channel_var = np.var(sub_X, axis=(0, 2))                  # (n_group_channels,)
        return float(per_channel_mean_abs.mean()), float(per_channel_var.mean())

    def rel_gap(search_val, memorize_val):
        return 100.0 * (search_val - memorize_val) / memorize_val if memorize_val != 0 else float("nan")

    per_subject_results = []
    missing_subjects = []

    for sub_idx in range(1, NUM_SUBJECTS + 1):
        sub_id = f"{sub_idx:02d}"
        sub_tag = f"sub-{sub_id}"
        cp = os.path.join(CHECKPOINT_DIR, f"{sub_tag}.npz")
        if not os.path.exists(cp):
            log.warning(f"[{sub_tag}] No checkpoint found — skipping (expected for sub-09).")
            missing_subjects.append(sub_tag)
            continue

        data = np.load(cp, allow_pickle=True)
        X_volts = data["X"].astype(np.float64)   # (n_epochs, 62, n_times), VOLTS per MNE convention
        y = data["y"].astype(np.int64)
        assert X_volts.shape[1] == N_CHANNELS, (
            f"[{sub_tag}] Expected {N_CHANNELS} channels, got {X_volts.shape[1]}"
        )
        n0, n1 = int(np.sum(y == 0)), int(np.sum(y == 1))
        assert n0 > 0 and n1 > 0, f"[{sub_tag}] Missing one class entirely (n0={n0}, n1={n1})"

        X_uv = X_volts * VOLTS_TO_MICROVOLTS

        fp_search_amp, fp_search_var     = group_stats(X_uv, y, frontopolar_idx, 0)
        fp_memorize_amp, fp_memorize_var = group_stats(X_uv, y, frontopolar_idx, 1)
        cp_search_amp, cp_search_var     = group_stats(X_uv, y, central_parietal_idx, 0)
        cp_memorize_amp, cp_memorize_var = group_stats(X_uv, y, central_parietal_idx, 1)

        record = {
            "subject": sub_tag, "n_search_trials": n0, "n_memorize_trials": n1,
            "frontopolar": {
                "search_mean_abs_amplitude_uv": fp_search_amp, "memorize_mean_abs_amplitude_uv": fp_memorize_amp,
                "search_variance_uv2": fp_search_var, "memorize_variance_uv2": fp_memorize_var,
                "relative_variance_gap_pct": rel_gap(fp_search_var, fp_memorize_var),
            },
            "central_parietal": {
                "search_mean_abs_amplitude_uv": cp_search_amp, "memorize_mean_abs_amplitude_uv": cp_memorize_amp,
                "search_variance_uv2": cp_search_var, "memorize_variance_uv2": cp_memorize_var,
                "relative_variance_gap_pct": rel_gap(cp_search_var, cp_memorize_var),
            },
        }
        record["frontopolar_gap_exceeds_central_parietal_gap"] = (
            record["frontopolar"]["relative_variance_gap_pct"] > record["central_parietal"]["relative_variance_gap_pct"]
        )
        record["search_greater_than_memorize_frontopolar_variance"] = fp_search_var > fp_memorize_var
        per_subject_results.append(record)

        log.info(
            f"[{sub_tag}] Frontopolar gap: {record['frontopolar']['relative_variance_gap_pct']:+.2f}% | "
            f"Central-parietal gap: {record['central_parietal']['relative_variance_gap_pct']:+.2f}% | "
            f"n_trials={n0+n1}"
        )

    assert len(per_subject_results) == NUM_SUBJECTS - 1, (
        f"Expected exactly {NUM_SUBJECTS - 1} subjects with checkpoints (30 - sub-09), "
        f"got {len(per_subject_results)}. Missing: {missing_subjects}. Investigate before trusting this aggregation."
    )

    # =========================================================================
    # STEP 2: pooled/aggregate summary across all subjects
    # =========================================================================
    fp_gaps = np.array([r["frontopolar"]["relative_variance_gap_pct"] for r in per_subject_results])
    cp_gaps = np.array([r["central_parietal"]["relative_variance_gap_pct"] for r in per_subject_results])
    n_subjects = len(per_subject_results)
    n_replicating_direction = int(np.sum([r["search_greater_than_memorize_frontopolar_variance"] for r in per_subject_results]))
    n_frontopolar_gap_larger = int(np.sum([r["frontopolar_gap_exceeds_central_parietal_gap"] for r in per_subject_results]))

    from scipy import stats as scipy_stats
    try:
        wilcoxon_stat, wilcoxon_p = scipy_stats.wilcoxon(fp_gaps, cp_gaps)
        wilcoxon_result = {"statistic": float(wilcoxon_stat), "p_value": float(wilcoxon_p)}
    except Exception as e:
        wilcoxon_result = {"error": str(e)}

    summary = {
        "n_subjects": n_subjects,
        "sub01_reference_frontopolar_gap_pct": 22.5,     # AUDIT.md Priority 3d, single-subject finding
        "sub01_reference_central_parietal_gap_pct": 2.0,  # AUDIT.md Priority 3d, single-subject finding
        "mean_frontopolar_relative_gap_pct": float(fp_gaps.mean()),
        "std_frontopolar_relative_gap_pct": float(fp_gaps.std()),
        "mean_central_parietal_relative_gap_pct": float(cp_gaps.mean()),
        "std_central_parietal_relative_gap_pct": float(cp_gaps.std()),
        "n_subjects_replicating_search_gt_memorize_frontopolar_variance": n_replicating_direction,
        "n_subjects_out_of": n_subjects,
        "n_subjects_where_frontopolar_gap_exceeds_central_parietal_gap": n_frontopolar_gap_larger,
        "wilcoxon_signed_rank_frontopolar_vs_central_parietal_gap": wilcoxon_result,
    }

    log.info(f"\n{'='*70}\n  VARIANCE-GAP REPLICATION — {n_subjects} subjects (all except sub-09)\n{'='*70}")
    log.info(f"  Mean frontopolar relative gap     : {summary['mean_frontopolar_relative_gap_pct']:+.2f}% "
             f"(+/- {summary['std_frontopolar_relative_gap_pct']:.2f}%) | sub-01 alone was +22.5%")
    log.info(f"  Mean central-parietal relative gap: {summary['mean_central_parietal_relative_gap_pct']:+.2f}% "
             f"(+/- {summary['std_central_parietal_relative_gap_pct']:.2f}%) | sub-01 alone was +2.0%")
    log.info(f"  Subjects replicating Search>Memorize frontopolar variance direction: "
             f"{n_replicating_direction}/{n_subjects}")
    log.info(f"  Subjects where frontopolar gap > central-parietal gap: {n_frontopolar_gap_larger}/{n_subjects}")
    log.info(f"  Wilcoxon signed-rank (frontopolar gap vs central-parietal gap, paired by subject): "
             f"{wilcoxon_result}")

    results_payload = {
        "condition": "F-OCULAR(d) — frontopolar-vs-central-parietal variance-gap replication, all subjects",
        "channel_groups": {
            "frontopolar_channels": FRONTOPOLAR_CHANNELS, "frontopolar_indices": frontopolar_idx,
            "central_parietal_channels": CENTRAL_PARIETAL_CHANNELS, "central_parietal_indices": central_parietal_idx,
            "verified_channel_order_source": "sub-01 raw .vhdr, read via mne.io.read_raw_brainvision",
        },
        "missing_subjects": missing_subjects,
        "per_subject_results": per_subject_results,
        "summary": summary,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_payload, f, indent=2)
    volume.commit()
    log.info(f"  Saved: {OUTPUT_JSON}")

    # =========================================================================
    # C3 plausibility assertions -- printed next to the numbers, not silent.
    # Run AFTER the write above so a failing assertion never suppresses the
    # diagnostic artifact.
    # =========================================================================
    assert n_subjects == 29, f"[C3 PLAUSIBILITY FAIL] expected 29 unique subjects, got {n_subjects}"
    for r in per_subject_results:
        for group in ("frontopolar", "central_parietal"):
            for stat_key in ("search_variance_uv2", "memorize_variance_uv2"):
                v = r[group][stat_key]
                assert v > 0.0 and np.isfinite(v), (
                    f"[C3 PLAUSIBILITY FAIL] {r['subject']} {group}.{stat_key} = {v} "
                    f"(variance must be strictly positive and finite -- zero/NaN signals a channel-index "
                    f"or unit-scaling bug, not a real finding)"
                )
            gap = r[group]["relative_variance_gap_pct"]
            assert np.isfinite(gap), f"[C3 PLAUSIBILITY FAIL] {r['subject']} {group} relative gap is non-finite ({gap})"
    log.info(f"  [C3] plausibility: 29/29 subjects, all per-group variances strictly positive and finite -- OK")

    return {
        "n_subjects": n_subjects,
        "mean_frontopolar_relative_gap_pct": summary["mean_frontopolar_relative_gap_pct"],
        "mean_central_parietal_relative_gap_pct": summary["mean_central_parietal_relative_gap_pct"],
        "n_subjects_replicating_direction": n_replicating_direction,
        "output_path": OUTPUT_JSON,
    }


@app.local_entrypoint()
def main():
    print("F-OCULAR(d) — Frontopolar-vs-central-parietal variance-gap replication, all subjects")
    print(f"Frontopolar: {FRONTOPOLAR_CHANNELS}")
    print(f"Central-parietal: {CENTRAL_PARIETAL_CHANNELS}")
    print("Reads cached per-subject checkpoints from run_data_engine_on_modal.py — no re-download.\n")
    results = compute_variance_gap_all_subjects.remote()
    print("\nVARIANCE-GAP REPLICATION RESULTS:")
    for k, v in results.items():
        print(f"  {k:<45}: {v}")
