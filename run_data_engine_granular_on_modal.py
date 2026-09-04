# =============================================================================
# command to run:
#   modal run run_data_engine_granular_on_modal.py::main
# run_data_engine_granular_on_modal.py
#
# GRANULAR DATA ENGINE — supports R1(b)/R2/R3 (RESULTS_LEDGER.md L037/L038,
# DECISIONS.md "R1 / R2 / R3 pre-registration").
#
# WHY THIS SCRIPT EXISTS:
#   `run_data_engine_on_modal.py`'s EVENT_ID collapses each block's four
#   distinct marker codes (10/11/12/13 or 20/21/22/23 -- Encode-phase onset,
#   Test-phase Target/Distractor/New-Lure onset respectively) into one
#   binary class label, and its per-subject raw BIDS folder is deleted
#   (`shutil.rmtree(raw_dir)`) immediately after that subject's epochs are
#   extracted -- so the code-level distinction cannot be recovered from
#   anything already on the volume. R1(b)/R2/R3 all need the ORIGINAL
#   per-epoch marker code (which phase, and for the test phase, which
#   item-type), not just the collapsed binary label.
#
#   This script re-downloads and re-epochs from raw BIDS ONCE (identical
#   filter/resample/epoching parameters to the original data engine, so the
#   output is directly comparable), but keeps all 8 marker codes distinct
#   (0-7) rather than collapsing them. This is the single expensive step;
#   `run_r1b_r2_r3_composition_runs.py` derives all three (R1b/R2/R3) label
#   sets from this ONE granular array via masking, so raw EEG is only
#   downloaded and epoched once, not three times.
#
# GRANULAR_EVENT_ID -> code (0-7):
#   0 = Stimulus/ 10 (Search  Encode)      4 = Stimulus/ 20 (Memorize Encode)
#   1 = Stimulus/ 11 (Search  Test-Target) 5 = Stimulus/ 21 (Memorize Test-Target)
#   2 = Stimulus/ 12 (Search  Test-Distr.) 6 = Stimulus/ 22 (Memorize Test-Distr.)
#   3 = Stimulus/ 13 (Search  Test-Lure)   7 = Stimulus/ 23 (Memorize Test-Lure)
#
# Usage: modal run run_data_engine_granular_on_modal.py::main
# =============================================================================

import modal

app = modal.App("bci-data-engine-granular")

# GATE C6: Image.env() alone was shown NOT to hold for OMP/MKL/OPENBLAS on
# the classifier scripts (Modal overwrites them to match cpu allocation
# after the image's ENV layer) -- threadpoolctl added here too for
# consistency, cpu left at 4.0 (not implicated in the confirmed
# non-determinism, and MNE's filter/resample step legitimately benefits
# from parallelism across 29 subjects; not reduced without evidence this
# script needs it). numpy tightened from an open floor (">=1.26.0") to an
# exact pin so a future image rebuild can't silently change the resolved
# BLAS-backed release.
eeg_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "mne==1.7.1",
        "openneuro-py==2024.2.0",
        "numpy==1.26.4",
        "scipy==1.14.1",
        "threadpoolctl==3.6.0",
        "tqdm",
    )
    .env({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    })
)

volume = modal.Volume.from_name("eeg-data-vol")
VOLUME_MOUNT_PATH = "/data"
CHECKPOINT_DIR   = "/data/checkpoints_granular"

NUM_SUBJECTS = 30

# Identical preprocessing constants to run_data_engine_on_modal.py -- the
# only change is that every one of the 8 raw codes gets its own integer,
# instead of being pre-collapsed into 2 classes.
GRANULAR_EVENT_ID = {
    "Stimulus/ 10": 0, "Stimulus/ 11": 1, "Stimulus/ 12": 2, "Stimulus/ 13": 3,
    "Stimulus/ 20": 4, "Stimulus/ 21": 5, "Stimulus/ 22": 6, "Stimulus/ 23": 7,
}
CODE_NAMES = {
    0: "search_encode", 1: "search_test_target", 2: "search_test_distractor", 3: "search_test_lure",
    4: "memorize_encode", 5: "memorize_test_target", 6: "memorize_test_distractor", 7: "memorize_test_lure",
}

# =============================================================================
# SUB-01 PRACTICE-TRIAL EXCLUSION (fourth-pass audit, GATE A/A2/B).
#
# sub-01's raw .vmrk carries 55 code-10 (search_encode) and 55 code-20
# (memorize_encode) markers, but sub-01's Encode behavioural TSV logs only
# 50 trials per task -- the only subject/code where marker counts and
# behavioural rows disagree (results/composition_check_all29.csv, all 29
# subjects, all 8 codes).
#
# Response-time alignment (interval[i] = rt[i] + fixed task ITI, validated
# on sub-02/sub-30 at ratio<0.02 and corr>0.999; see
# results/rt_alignment_check_all29.csv for the full 29-subject cohort,
# where sub-01 is the only flagged subject on RAW COUNTS and NO subject is
# flagged on the alignment residual) localizes the ten unmatched epochs to
# the first 5 chronological code-10 markers and the first 5 chronological
# code-20 markers: markers 6-55 fit the model at the cohort noise floor
# (residual SD ~4ms, corr>0.999, margin >100x over every other window);
# markers 1-5 do not correspond to any logged behavioural trial.
#
# B0 confirmed these are unlogged PRACTICE trials, not a truncated log:
# sub-01's 50 logged Encode rows are in perfect 50<->50 bijection with the
# 50 Test-phase Target probes, per task (0 missing either direction) --
# identical in shape to the sub-02 control.
#
# Native (pre-filter, pre-resample, 1000 Hz) sample positions of the ten
# markers to be excluded, asserted below before ANY processing is applied.
#
# NOTE: values below are already adjusted -1 from the raw .vmrk text. The
# BrainVision .vmrk "Position in data points" field is 1-indexed per the
# format spec (this is what GATE A/A2's direct-.vmrk-text analysis used:
# 229081, 232952, ... / 1642730, 1646632, ...); mne.events_from_annotations
# returns 0-indexed sample numbers. Confirmed empirically on the first
# real Modal run: all 5 code-0 positions came back exactly 1 lower than
# the raw .vmrk value (229081->229080, 232952->232951, 236330->236329,
# 239721->239720, 242976->242975) -- a systematic, fully-explained offset,
# not a sign the identification no longer holds.
SUB01_EXCLUDE_NATIVE_SAMPLE_POS = {
    0: [229080, 232951, 236329, 239720, 242975],       # search_encode markers 1-5 (MNE 0-indexed)
    4: [1642729, 1646631, 1651031, 1655295, 1658986],  # memorize_encode markers 1-5 (MNE 0-indexed)
}
# =============================================================================

TMIN        = -0.2
TMAX        =  0.8
BASELINE    = (None, 0)
L_FREQ      =  1.0
H_FREQ      = 40.0
RESAMPLE_HZ = 250


@app.function(
    image=eeg_image,
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=10800,
    memory=16384,
    cpu=4.0,
)
def run_granular_pipeline(git_short_hash: str = "nogit"):
    import os, glob, shutil, logging, traceback
    import numpy as np
    import scipy
    import mne
    import openneuro
    from threadpoolctl import threadpool_limits, threadpool_info

    mne.set_log_level("WARNING")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("bci-engine-granular")

    log.info(f"numpy={np.__version__} scipy={scipy.__version__} mne={mne.__version__}")
    log.info(f"threadpool_info BEFORE threadpoolctl pin: {threadpool_info()}")
    threadpool_limits(limits=1)  # not restored -- single-use container, see run_c4's identical comment
    log.info(f"threadpool_info AFTER threadpoolctl pin (every entry must show num_threads=1): "
             f"{threadpool_info()}")
    _npcfg = np.show_config(mode="dicts")
    _simd = _npcfg.get("SIMD Extensions", {})
    _blas = _npcfg.get("Build Dependencies", {}).get("blas", {})
    hardware_info = {
        "numpy_version": np.__version__, "scipy_version": scipy.__version__, "mne_version": mne.__version__,
        "simd_found": _simd.get("found", []), "simd_not_found": _simd.get("not found", []),
        "blas_name": _blas.get("name"), "blas_version": _blas.get("version"),
        "openblas_configuration": _blas.get("openblas configuration"),
        "threadpool_info_after_pin": threadpool_info(),
    }
    log.info(f"hardware_info: {hardware_info}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    failed_subjects = []

    for sub_idx in range(1, NUM_SUBJECTS + 1):
        sub_id  = f"{sub_idx:02d}"
        sub_tag = f"sub-{sub_id}"
        raw_dir = os.path.join(VOLUME_MOUNT_PATH, "openneuro", sub_tag)

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{sub_tag}.npz")
        if os.path.exists(checkpoint_path):
            if sub_id == "01":
                # sub-01's checkpoint may predate the practice-trial exclusion
                # added above (fourth-pass GATE C). A stale 55/55 checkpoint
                # must be reprocessed, not silently skipped -- validate before
                # trusting it.
                existing = np.load(checkpoint_path, allow_pickle=True)
                existing_code = existing["code"]
                n_search_enc = int(np.sum(existing_code == 0))
                n_mem_enc = int(np.sum(existing_code == 4))
                if n_search_enc == 50 and n_mem_enc == 50:
                    log.info(f"[{sub_tag}] ALREADY DONE (post-exclusion checkpoint verified 50/50) — skipping.")
                    continue
                else:
                    log.warning(f"[{sub_tag}] STALE CHECKPOINT (search_encode={n_search_enc}, "
                                f"memorize_encode={n_mem_enc}, expected 50/50 post-exclusion) — "
                                "deleting and reprocessing from raw BIDS.")
                    os.remove(checkpoint_path)
            else:
                log.info(f"[{sub_tag}] ALREADY DONE — checkpoint found, skipping.")
                continue

        log.info(f"{'='*60}\nProcessing {sub_tag} ({sub_idx}/{NUM_SUBJECTS})\n{'='*60}")

        try:
            openneuro.download(
                dataset="ds005189",
                target_dir=os.path.join(VOLUME_MOUNT_PATH, "openneuro"),
                include=[sub_tag],
            )

            vhdr_files = glob.glob(os.path.join(raw_dir, "**", "*.vhdr"), recursive=True)
            if not vhdr_files:
                raise FileNotFoundError(f"No .vhdr found under {raw_dir}")
            vhdr_path = vhdr_files[0]

            raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)

            # --- sub-01 exclusion assertion, PASS 1: native sample positions ---
            # Checked BEFORE filter/resample so the sample positions are directly
            # comparable to the .vmrk-derived values documented above. Fails loud
            # if the raw marker structure has changed since GATE A/A2/B -- this
            # must never silently mis-fire on a re-download or a dataset update.
            if sub_id == "01":
                events_native, _ = mne.events_from_annotations(raw, event_id=GRANULAR_EVENT_ID, verbose=False)
                for code_val, expected_first5 in SUB01_EXCLUDE_NATIVE_SAMPLE_POS.items():
                    pos = events_native[events_native[:, 2] == code_val][:, 0]
                    assert len(pos) == 55, (
                        f"[SUB-01 EXCLUSION ASSERTION FAILED] code {code_val}: expected 55 native markers, "
                        f"found {len(pos)}. Halting -- do not proceed with a stale exclusion."
                    )
                    actual_first5 = pos[:5].tolist()
                    assert actual_first5 == expected_first5, (
                        f"[SUB-01 EXCLUSION ASSERTION FAILED] code {code_val}: expected first-5 native sample "
                        f"positions {expected_first5}, found {actual_first5}. Halting -- the practice-trial "
                        "identification from GATE A2 no longer holds for this data."
                    )
                log.info("[sub-01] Exclusion assertion PASS 1 (native sample positions) PASSED for "
                         "both encode codes.")

            raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, method="iir", verbose=False)
            raw.resample(sfreq=RESAMPLE_HZ, verbose=False)

            events, event_id_found = mne.events_from_annotations(raw, event_id=GRANULAR_EVENT_ID, verbose=False)
            if len(events) == 0:
                raise ValueError(f"Zero events found for {sub_tag}")

            epochs = mne.Epochs(
                raw, events, event_id=event_id_found,
                tmin=TMIN, tmax=TMAX, baseline=BASELINE,
                preload=True, reject=None, verbose=False,
            )
            if len(epochs) == 0:
                raise ValueError(f"Zero epochs survived for {sub_tag}")

            X_sub = epochs.get_data(copy=True).astype(np.float32)

            inv_event_map = {event_id_found[k]: v for k, v in GRANULAR_EVENT_ID.items() if k in event_id_found}
            code_sub = np.array([inv_event_map[c] for c in epochs.events[:, 2]], dtype=np.int64)
            subject_ids_sub = np.array([sub_id] * len(code_sub), dtype=object)

            # --- sub-01 exclusion, PASS 2: apply and assert (Amendment 3) ---
            # Applied ONCE, here, at the epoching layer -- no per-script filter
            # downstream. code_sub preserves chronological marker order (built
            # directly from epochs.events, which mne.Epochs keeps time-sorted),
            # so "first 5 occurrences of a code" here is the same set already
            # verified against native sample positions in PASS 1 above.
            if sub_id == "01":
                drop_mask = np.zeros(len(code_sub), dtype=bool)
                for code_val in SUB01_EXCLUDE_NATIVE_SAMPLE_POS:
                    idx_this_code = np.where(code_sub == code_val)[0]
                    assert len(idx_this_code) == 55, (
                        f"[SUB-01 EXCLUSION ASSERTION FAILED] code {code_val}: expected 55 epochs pre-exclusion "
                        f"after epoching, found {len(idx_this_code)}."
                    )
                    drop_mask[idx_this_code[:5]] = True
                n_dropped = int(drop_mask.sum())
                assert n_dropped == 10, (
                    f"[SUB-01 EXCLUSION ASSERTION FAILED] expected to drop exactly 10 epochs "
                    f"(5 search_encode + 5 memorize_encode), actually dropped {n_dropped}."
                )
                X_sub = X_sub[~drop_mask]
                code_sub = code_sub[~drop_mask]
                subject_ids_sub = subject_ids_sub[~drop_mask]
                post_counts = {c: int(np.sum(code_sub == c)) for c in SUB01_EXCLUDE_NATIVE_SAMPLE_POS}
                assert post_counts[0] == 50 and post_counts[4] == 50, (
                    f"[SUB-01 EXCLUSION ASSERTION FAILED] post-exclusion encode counts are not 50/50: "
                    f"{post_counts}."
                )
                log.info(f"[sub-01] Exclusion assertion PASS 2 PASSED: dropped exactly 10 epochs "
                         f"(5 search_encode + 5 memorize_encode, unlogged practice trials per GATE A2/B). "
                         f"Post-exclusion encode counts: search_encode=50, memorize_encode=50. "
                         f"X_sub now: {X_sub.shape}")

            # Build-time consistency check: this MUST reproduce the exact
            # per-code counts documented in RESULTS_LEDGER.md L037/S1-S4
            # (50 encode / 50 target / 25 distractor / 75 lure, PER TASK).
            counts = {c: int(np.sum(code_sub == c)) for c in range(8)}
            log.info(f"[{sub_tag}] Extracted | X: {X_sub.shape} | per-code counts: "
                     f"{ {CODE_NAMES[c]: n for c, n in counts.items()} }")
            for enc_code, task_name in ((0, "search"), (4, "memorize")):
                tgt, dis, lur = counts[enc_code + 1], counts[enc_code + 2], counts[enc_code + 3]
                if not (40 <= counts[enc_code] <= 60 and 40 <= tgt <= 60 and 15 <= dis <= 35 and 60 <= lur <= 90):
                    log.warning(f"[{sub_tag}] {task_name}: counts outside the expected ~50/50/25/75 "
                                f"range (encode={counts[enc_code]}, target={tgt}, distractor={dis}, lure={lur}) "
                                "-- not fatal, but flagged. sub-01's practice-trial exclusion above is asserted "
                                "separately and should already bring it to flat 50/50; any OTHER subject "
                                "landing here needs manual investigation before this dataset is trusted.")

            np.savez_compressed(checkpoint_path, X=X_sub, code=code_sub, subjects=subject_ids_sub)
            volume.commit()
            log.info(f"[{sub_tag}] Checkpoint saved & committed.")

        except Exception as e:
            log.error(f"[{sub_tag}] FAILED — {type(e).__name__}: {e}")
            log.error(traceback.format_exc())
            failed_subjects.append(sub_tag)

        finally:
            if os.path.exists(raw_dir):
                shutil.rmtree(raw_dir)
                log.info(f"[{sub_tag}] Raw folder deleted.")

    log.info("All subjects done. Merging checkpoints ...")
    all_X, all_code, all_subjects = [], [], []
    for sub_idx in range(1, NUM_SUBJECTS + 1):
        sub_id = f"{sub_idx:02d}"
        cp = os.path.join(CHECKPOINT_DIR, f"sub-{sub_id}.npz")
        if os.path.exists(cp):
            data = np.load(cp, allow_pickle=True)
            all_X.append(data["X"]); all_code.append(data["code"]); all_subjects.append(data["subjects"])
            log.info(f"[sub-{sub_id}] Loaded from checkpoint: {data['X'].shape}")
        else:
            log.warning(f"[sub-{sub_id}] No checkpoint found — was failed/skipped.")

    if not all_X:
        raise RuntimeError("No checkpoints found to merge!")

    n_merged = len(all_X)
    n_expected = NUM_SUBJECTS - len(failed_subjects)
    assert n_merged == n_expected, (
        f"Subject-count mismatch at merge: expected {n_expected} ({NUM_SUBJECTS} - {len(failed_subjects)} "
        f"failed = {failed_subjects}), found {n_merged}. Investigate before trusting this dataset."
    )

    X_all = np.concatenate(all_X, axis=0)
    code_all = np.concatenate(all_code, axis=0)
    subjects_all = np.concatenate(all_subjects, axis=0)

    # --- Global post-exclusion assertion (Amendment 3, item 2) ---
    # Every subject in the merged set must now be flat 50/50 on the two
    # encode codes -- sub-01 via the exclusion above, everyone else because
    # results/composition_check_all29.csv already found no other subject
    # with any code mismatch. Fail loud, not a warning, if this does not hold.
    merged_subjects = sorted(np.unique(subjects_all).tolist())
    per_subject_encode_report = {}
    bad_subjects = {}
    for s in merged_subjects:
        smask = subjects_all == s
        n_search_encode = int(np.sum((code_all == 0) & smask))
        n_memorize_encode = int(np.sum((code_all == 4) & smask))
        per_subject_encode_report[s] = {"search_encode": n_search_encode, "memorize_encode": n_memorize_encode}
        if n_search_encode != 50 or n_memorize_encode != 50:
            bad_subjects[s] = per_subject_encode_report[s]
    assert not bad_subjects, (
        f"[GLOBAL EXCLUSION ASSERTION FAILED] the following subjects are NOT flat 50/50 encode after "
        f"the sub-01 exclusion: {bad_subjects}. Halting -- do not trust this dataset for R2/downstream analyses."
    )
    total_search_encode = int(np.sum(code_all == 0))
    total_memorize_encode = int(np.sum(code_all == 4))
    total_encode = total_search_encode + total_memorize_encode
    assert len(merged_subjects) == 29, (
        f"[GLOBAL EXCLUSION ASSERTION FAILED] expected 29 subjects in the merged set, found "
        f"{len(merged_subjects)}: {merged_subjects}."
    )
    assert total_search_encode == 1450 and total_memorize_encode == 1450 and total_encode == 2900, (
        f"[GLOBAL EXCLUSION ASSERTION FAILED] expected 1,450/1,450 (2,900 total) encode epochs across "
        f"29 subjects, found search_encode={total_search_encode}, memorize_encode={total_memorize_encode}, "
        f"total={total_encode}."
    )
    log.info(f"[GLOBAL ASSERTION PASSED] all {len(merged_subjects)} subjects flat 50/50 encode; "
             f"total R2 dataset = {total_encode} (1,450 search_encode + 1,450 memorize_encode).")
    log.info(f"Per-subject encode report: {per_subject_encode_report}")

    output_path = os.path.join(VOLUME_MOUNT_PATH, "processed_eeg_all_subjects_granular.npz")
    np.savez_compressed(output_path, X=X_all, code=code_all, subjects=subjects_all, hardware_info=hardware_info)

    # GATE C5 STEP 3: non-overwriting artifact -- see the C1/C3/C4/D2/R1b
    # scripts' identical comment. NOTE the cost tradeoff here specifically:
    # this file is ~640 MB, so every re-run doubles storage rather than a
    # few KB like the other five scripts' JSONs. Stamped anyway per the
    # explicit "all six scripts" instruction, since this is the single
    # dataset every downstream number in the project derives from -- losing
    # a prior version of it silently would be strictly worse than the
    # storage cost. Delete old stamped copies manually if volume space
    # becomes a concern; the convenience pointer is what all other scripts
    # actually read, so deleting old stamped copies never breaks anything.
    stamped_output_path = output_path.replace(
        ".npz", f"_{__import__('datetime').datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{git_short_hash}.npz"
    )
    np.savez_compressed(stamped_output_path, X=X_all, code=code_all, subjects=subjects_all, hardware_info=hardware_info)

    log.info(f"Final granular dataset saved (convenience pointer, overwritten): {output_path}")
    log.info(f"Final granular dataset saved (immutable): {stamped_output_path}")
    log.info(f"Shape X: {X_all.shape} | Total epochs: {len(code_all)}")
    counts_all = {CODE_NAMES[c]: int(np.sum(code_all == c)) for c in range(8)}
    log.info(f"Per-code totals: {counts_all}")
    log.info(f"Failed: {failed_subjects}")

    volume.commit()
    log.info("Volume committed. DONE!")

    return {
        "output_path": output_path, "stamped_output_path": stamped_output_path,
        "shape_X": X_all.shape, "total_epochs": int(len(code_all)),
        "per_code_totals": counts_all, "failed_subjects": failed_subjects,
        "hardware_info": hardware_info,
        "global_encode_assertion": "PASSED (29/29 subjects flat 50/50, total 2,900, 1,450 per class)",
        "per_subject_encode_report": per_subject_encode_report,
    }


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
    print("Launching GRANULAR BCI Data Engine (8-way marker code, supports R1b/R2/R3) ...")
    print(f"git_short_hash for this run's stamped output filename: {git_short_hash}")
    result = run_granular_pipeline.remote(git_short_hash=git_short_hash)
    print("\n" + "="*60)
    for k, v in result.items():
        print(f"  {k:<20}: {v}")
    print("="*60)
    print("\nNext step: modal run run_r1b_r2_r3_composition_runs.py::main")
