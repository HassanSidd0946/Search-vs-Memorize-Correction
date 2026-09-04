# =============================================================================
# run_step4_ica_cleaned_control.py
#
# CONDITION 4 — ICA-CLEANED PREPROCESSING CONTROL (AUDIT.md F-OCULAR(b))
#
# WHY THIS SCRIPT EXISTS:
#   AUDIT.md's Phase 0.5 ocular-confound findings (frontopolar-vs-central-
#   parietal variance gap; F-OCULAR family) raise the possibility that some
#   of the spatial classifier's discriminative signal is riding on eye-
#   movement/blink artifacts leaking into frontal channels, rather than on
#   genuine cortical activity related to Search-vs-Memorize encoding.
#   F-OCULAR(a) (run_step4_matched_spatial_control_frontal_ablated.py)
#   answers this by REMOVING the suspect channels entirely. This script
#   answers the complementary question with a different method: instead of
#   dropping the frontal channels, CLEAN them — fit ICA on each subject's
#   continuous (filtered, resampled) signal, identify components correlated
#   with blink/saccade activity using Fp1 as an EOG surrogate (there is no
#   dedicated EOG channel in this montage — see AUDIT.md Phase 0.5 Priority 3
#   / D3), remove those components, and re-epoch. If the matched-spatial-
#   control accuracy (70.78% +/- 11.86%, all 62 channels, uncleaned) survives
#   ICA cleaning largely intact, that is evidence the discriminative signal
#   is not primarily an artifact of a few blink-related components. If it
#   collapses, that is independent corroborating evidence for the ocular-
#   confound concern F-OCULAR(a) is designed to test directly.
#
#   NOTE: ICA cleaning and channel ablation are not redundant controls. ICA
#   removes specific temporally-defined artifact COMPONENTS from all 62
#   channels (a component can contaminate any channel, not just frontal
#   ones, and genuine frontal cortical signal that survives ICA is kept).
#   Channel ablation removes 9 whole CHANNELS regardless of what is or isn't
#   contaminated in them. Agreement between the two controls is much
#   stronger evidence than either alone.
#
# WHY find_bads_eog(ch_name="Fp1") INSTEAD OF mne-icalabel:
#   AUDIT.md's Fix-ID table lists "find_bads_eog(ch_name='Fp1') (and/or
#   mne-icalabel)" as acceptable. This script uses find_bads_eog only, to
#   keep the Modal image dependency-light and avoid mne-icalabel's bundled
#   pretrained-model download (an extra point of non-determinism/failure in
#   a remote sandbox with no interactive debugging). find_bads_eog's
#   correlation-based approach against a frontal-pole channel is standard
#   practice for montages without a dedicated EOG channel. This choice is
#   logged here, not silently made, per AUDIT.md's disclosure convention.
#
# WHY A FRESH RAW->EPOCH PASS (not reusing existing checkpoints):
#   run_data_engine_on_modal.py's per-subject checkpoints already contain
#   EPOCHED data (post mne.Epochs), but ICA must be fit on CONTINUOUS data
#   before epoching (MNE's documented recommendation, and the only way blink
#   components spanning multiple trials are identifiable at all). This
#   script therefore re-downloads and re-preprocesses raw BrainVision files
#   through filter+resample (IDENTICAL constants to run_data_engine_on_modal.py)
#   before inserting the new ICA step and epoching. This is the source of
#   this control's "Medium" cost estimate in AUDIT.md's Fix-ID table.
#
# TWO-STAGE DESIGN (build then compare), mirroring the data engine's own
# preemption-safe checkpoint strategy so a Modal timeout/preemption during
# the ~29-subject ICA pass does not lose completed subjects:
#   1. `modal run run_step4_ica_cleaned_control.py::build`
#      -> per-subject checkpoints under /data/checkpoints_ica_cleaned/,
#         merged into /data/processed_eeg_all_subjects_ica_cleaned.npz
#   2. `modal run run_step4_ica_cleaned_control.py::main`
#      -> loads BOTH the existing uncleaned dataset
#         (/data/processed_eeg_all_subjects.npz) and the new cleaned one,
#         asserts they contain the identical trials (same shape/y/subjects,
#         since ICA changes signal content but not which epochs survive —
#         reject=None in both), runs the byte-for-byte-identical EA +
#         identity-reference tangent-space + 15%-shrinkage-calibration LOSO
#         pipeline from run_step4_matched_spatial_control.py on each
#         independently (single seed=42, matching that script's convention;
#         this is a Medium-risk supporting control, not the CRITICAL D3-
#         gated decisive control F-OCULAR(a) is, so it does not require the
#         5-seed treatment), and reports both side by side.
#
# STATUS: written and syntax-verified only, per AUDIT.md — NOT YET EXECUTED.
#   Requires a real Modal account/volume; cannot be run from this sandbox.
# =============================================================================

import modal

app    = modal.App("bci-condition4-ica-cleaned-control")
volume = modal.Volume.from_name("eeg-data-vol")

VOLUME_MOUNT_PATH        = "/data"
CHECKPOINT_DIR_CLEANED   = "/data/checkpoints_ica_cleaned"
CLEANED_DATA_PATH        = "/data/processed_eeg_all_subjects_ica_cleaned.npz"
UNCLEANED_DATA_PATH      = "/data/processed_eeg_all_subjects.npz"
OUTPUT_JSON              = "/data/results_condition4_ica_cleaned_control.json"

NUM_SUBJECTS = 30   # sub-09 is expected to fail (truncated .eeg export, AUDIT.md D2)

# Preprocessing constants — IDENTICAL to run_data_engine_on_modal.py, so the
# only difference between the cleaned and uncleaned datasets is the ICA step.
EVENT_ID = {
    "Stimulus/ 10": 0, "Stimulus/ 11": 0,
    "Stimulus/ 12": 0, "Stimulus/ 13": 0,
    "Stimulus/ 20": 1, "Stimulus/ 21": 1,
    "Stimulus/ 22": 1, "Stimulus/ 23": 1,
}
TMIN, TMAX   = -0.2, 0.8
BASELINE     = (None, 0)
L_FREQ       = 1.0
H_FREQ       = 40.0
RESAMPLE_HZ  = 250
SFREQ, N_CHANNELS = 250, 62

# ICA constants
ICA_EOG_SURROGATE_CH = "Fp1"    # frontal-pole channel used as EOG surrogate — no dedicated EOG channel in this montage
ICA_N_COMPONENTS     = 0.99     # explained-variance fraction (MNE-recommended default for artifact-removal ICA)
ICA_METHOD           = "fastica"
ICA_RANDOM_STATE      = 42
ICA_EOG_THRESHOLD     = "auto"

# Downstream classification constants — IDENTICAL to run_step4_matched_spatial_control.py
CAL_FRACTION       = 0.15
RANDOM_SEED        = 42
COV_SHRINKAGE       = 0.1
PCA_MAX_COMPONENTS = 35
LOGREG_C           = 1.0
SHRINK_GRID        = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS    = 3

# Reference numbers (for logging/comparison only)
MATCHED_SPATIAL_CONTROL_MEAN_ACC = 0.7078   # run_step4_matched_spatial_control.py, full 29-fold, uncleaned, all 62 channels

build_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "mne==1.7.1",
        "openneuro-py==2024.2.0",
        "numpy>=1.26.0",
        "scipy==1.14.1",
        "scikit-learn==1.4.2",   # ICA's fastica backend depends on sklearn
        "tqdm",
    )
)

compare_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy<2", "scikit-learn==1.4.2", "scipy")
)


# =============================================================================
# STAGE 1 — BUILD: raw -> filter -> resample -> ICA-clean -> epoch, checkpointed
# =============================================================================
@app.function(
    image=build_image,
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=10800,
    memory=16384,
    cpu=4.0,
)
def build_ica_cleaned_dataset():
    import os, glob, shutil, logging, traceback
    import numpy as np
    import mne
    import openneuro

    mne.set_log_level("WARNING")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("ica-cleaned-build")

    os.makedirs(CHECKPOINT_DIR_CLEANED, exist_ok=True)
    failed_subjects = []

    for sub_idx in range(1, NUM_SUBJECTS + 1):
        sub_id  = f"{sub_idx:02d}"
        sub_tag = f"sub-{sub_id}"
        raw_dir = os.path.join(VOLUME_MOUNT_PATH, "openneuro", sub_tag)

        checkpoint_path = os.path.join(CHECKPOINT_DIR_CLEANED, f"{sub_tag}.npz")
        if os.path.exists(checkpoint_path):
            log.info(f"[{sub_tag}] ALREADY DONE — checkpoint found, skipping.")
            continue

        log.info(f"{'='*60}\nProcessing {sub_tag} ({sub_idx}/{NUM_SUBJECTS})\n{'='*60}")

        try:
            log.info(f"[{sub_tag}] Downloading ...")
            openneuro.download(
                dataset="ds005189",
                target_dir=os.path.join(VOLUME_MOUNT_PATH, "openneuro"),
                include=[sub_tag],
            )

            vhdr_files = glob.glob(os.path.join(raw_dir, "**", "*.vhdr"), recursive=True)
            if not vhdr_files:
                raise FileNotFoundError(f"No .vhdr found under {raw_dir}")
            vhdr_path = vhdr_files[0]
            log.info(f"[{sub_tag}] EEG file: {vhdr_path}")

            raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
            log.info(
                f"[{sub_tag}] Loaded | Ch: {raw.info['nchan']} | "
                f"Sfreq: {raw.info['sfreq']} Hz | Dur: {raw.times[-1]:.1f}s"
            )

            assert ICA_EOG_SURROGATE_CH in raw.ch_names, (
                f"[{sub_tag}] Expected EOG-surrogate channel "
                f"'{ICA_EOG_SURROGATE_CH}' not found in {raw.ch_names}"
            )

            # STEP: filter (identical to main data engine; also satisfies
            # MNE's recommendation that ICA be fit on data high-pass filtered >=1Hz)
            raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, method="iir", verbose=False)

            # STEP: resample (identical to main data engine)
            raw.resample(sfreq=RESAMPLE_HZ, verbose=False)

            # STEP: ICA fit + blink/saccade component removal (NEW vs. main data engine)
            ica = mne.preprocessing.ICA(
                n_components=ICA_N_COMPONENTS,
                method=ICA_METHOD,
                random_state=ICA_RANDOM_STATE,
                max_iter="auto",
            )
            ica.fit(raw, picks="eeg", verbose=False)

            eog_indices, eog_scores = ica.find_bads_eog(
                raw, ch_name=ICA_EOG_SURROGATE_CH, threshold=ICA_EOG_THRESHOLD, verbose=False
            )
            ica.exclude = eog_indices
            n_components_fit = ica.n_components_
            n_components_excluded = len(eog_indices)
            top_scores = sorted(
                [float(abs(s)) for s in np.asarray(eog_scores).ravel()], reverse=True
            )[:5]

            log.info(
                f"[{sub_tag}] ICA: fit {n_components_fit} components, "
                f"excluded {n_components_excluded} EOG-correlated component(s) "
                f"{eog_indices} | top |scores|: {top_scores}"
            )
            if n_components_excluded == 0:
                log.warning(
                    f"[{sub_tag}] find_bads_eog identified ZERO components above "
                    f"threshold — cleaned data for this subject will be numerically "
                    f"identical to uncleaned. Logged, not silently dropped, per "
                    f"AUDIT.md F-SILENT discipline."
                )

            raw_clean = raw.copy()
            ica.apply(raw_clean, verbose=False)

            # STEP: events + epochs (identical to main data engine, on cleaned signal)
            events, event_id_found = mne.events_from_annotations(raw_clean, event_id=EVENT_ID, verbose=False)
            if len(events) == 0:
                raise ValueError(f"Zero events found for {sub_tag}")

            epochs = mne.Epochs(
                raw_clean, events, event_id=event_id_found,
                tmin=TMIN, tmax=TMAX, baseline=BASELINE,
                preload=True, reject=None, verbose=False,
            )
            if len(epochs) == 0:
                raise ValueError(f"Zero epochs survived for {sub_tag}")

            X_sub = epochs.get_data(copy=True).astype(np.float32)
            inv_event_map = {event_id_found[k]: v for k, v in EVENT_ID.items() if k in event_id_found}
            y_sub = np.array([inv_event_map[c] for c in epochs.events[:, 2]], dtype=np.int32)
            subject_ids_sub = np.array([sub_id] * len(y_sub), dtype=object)

            log.info(
                f"[{sub_tag}] Extracted (cleaned) | X: {X_sub.shape} | "
                f"class0={np.sum(y_sub==0)}, class1={np.sum(y_sub==1)}"
            )

            np.savez_compressed(
                checkpoint_path,
                X=X_sub, y=y_sub, subjects=subject_ids_sub,
                n_ica_components_fit=n_components_fit,
                n_ica_components_excluded=n_components_excluded,
                eog_component_indices=np.array(eog_indices, dtype=np.int32),
                eog_top_scores=np.array(top_scores, dtype=np.float32),
            )
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

    # ---- Merge ----
    log.info("All subjects done. Merging ICA-cleaned checkpoints ...")
    all_X, all_y, all_subjects = [], [], []
    ica_metadata = {}

    for sub_idx in range(1, NUM_SUBJECTS + 1):
        sub_id = f"{sub_idx:02d}"
        cp = os.path.join(CHECKPOINT_DIR_CLEANED, f"sub-{sub_id}.npz")
        if os.path.exists(cp):
            data = np.load(cp, allow_pickle=True)
            all_X.append(data["X"])
            all_y.append(data["y"])
            all_subjects.append(data["subjects"])
            ica_metadata[f"sub-{sub_id}"] = {
                "n_ica_components_fit": int(data["n_ica_components_fit"]),
                "n_ica_components_excluded": int(data["n_ica_components_excluded"]),
                "eog_component_indices": data["eog_component_indices"].tolist(),
            }
            log.info(f"[sub-{sub_id}] Loaded from checkpoint: {data['X'].shape}")
        else:
            log.warning(f"[sub-{sub_id}] No checkpoint found — was failed/skipped.")

    if not all_X:
        raise RuntimeError("No checkpoints found to merge!")

    n_merged = len(all_X)
    n_expected = NUM_SUBJECTS - len(failed_subjects)
    assert n_merged == n_expected, (
        f"Subject-count mismatch at merge: expected {n_expected} checkpoints "
        f"({NUM_SUBJECTS} total - {len(failed_subjects)} failed = {failed_subjects}), "
        f"but found {n_merged} on disk. Investigate before trusting this dataset."
    )

    X_all        = np.concatenate(all_X, axis=0)
    y_all        = np.concatenate(all_y, axis=0)
    subjects_all = np.concatenate(all_subjects, axis=0)

    np.savez_compressed(CLEANED_DATA_PATH, X=X_all, y=y_all, subjects=subjects_all)
    import json
    with open("/data/ica_cleaning_metadata.json", "w") as f:
        json.dump(ica_metadata, f, indent=2)

    log.info(f"Final ICA-cleaned dataset saved: {CLEANED_DATA_PATH}")
    log.info(f"Shape X: {X_all.shape} | Total epochs: {len(y_all)}")
    log.info(f"Failed: {failed_subjects}")

    volume.commit()
    log.info("Volume committed. BUILD DONE!")

    return {
        "output_path": CLEANED_DATA_PATH,
        "shape_X": X_all.shape,
        "total_epochs": int(len(y_all)),
        "failed_subjects": failed_subjects,
        "subjects_with_zero_components_excluded": [
            k for k, v in ica_metadata.items() if v["n_ica_components_excluded"] == 0
        ],
    }


# =============================================================================
# STAGE 2 — COMPARE: run the byte-for-byte-identical matched-spatial-control
# pipeline on both the cleaned and uncleaned datasets, side by side.
# =============================================================================
@app.function(image=compare_image, cpu=4.0, volumes={VOLUME_MOUNT_PATH: volume}, timeout=86400, memory=16384)
def run_ica_vs_uncleaned_comparison():
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
    from sklearn.metrics import confusion_matrix, f1_score
    import logging, time, math, json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("ica-vs-uncleaned")

    np.random.seed(RANDOM_SEED)

    uncleaned = np.load(UNCLEANED_DATA_PATH, allow_pickle=True)
    cleaned   = np.load(CLEANED_DATA_PATH, allow_pickle=True)

    Xu, yu, su = uncleaned["X"].astype(np.float32), uncleaned["y"].astype(np.int64), uncleaned["subjects"]
    Xc, yc, sc = cleaned["X"].astype(np.float32), cleaned["y"].astype(np.int64), cleaned["subjects"]

    # F-SILENT-style hardening: this comparison is only apples-to-apples if
    # both datasets contain the identical trials in the identical order.
    # ICA changes signal CONTENT, not epoch survival (reject=None in both
    # pipelines) — if this assertion fails, something upstream diverged
    # (e.g. a subject failed in one build but not the other) and the
    # side-by-side comparison below would be silently invalid.
    assert Xu.shape == Xc.shape, f"Shape mismatch: uncleaned {Xu.shape} vs cleaned {Xc.shape}"
    assert np.array_equal(yu, yc), "Label arrays differ between uncleaned and cleaned datasets"
    assert np.array_equal(su, sc), "Subject arrays differ between uncleaned and cleaned datasets"
    assert len(np.unique(su)) == 29, (
        f"Expected exactly 29 subjects (30 - sub-09 exclusion per AUDIT.md D2), "
        f"got {len(np.unique(su))}: {sorted(np.unique(su).tolist())}"
    )
    log.info(f"Trial-identity check passed: {Xu.shape}, {len(np.unique(su))} subjects, both datasets aligned.")

    unique_subjects = sorted(np.unique(su).tolist())

    # =========================================================================
    # RIEMANNIAN / EA / CALIBRATION UTILITIES — IDENTICAL to
    # run_step4_matched_spatial_control.py (byte-for-byte, for comparability)
    # =========================================================================
    def trial_covariances(X, shrinkage=COV_SHRINKAGE):
        Xc_ = X - X.mean(axis=2, keepdims=True)
        cov = np.einsum("nct,ndt->ncd", Xc_, Xc_) / (X.shape[2] - 1)
        eye = np.eye(X.shape[1], dtype=cov.dtype)[None, :, :]
        tr = np.trace(cov, axis1=1, axis2=2) / X.shape[1]
        return (1 - shrinkage) * cov + shrinkage * tr[:, None, None] * eye

    def matrix_sqrt_inv_sqrt(mat, eps=1e-8):
        eigvals, eigvecs = np.linalg.eigh(mat)
        eigvals = np.clip(eigvals, eps, None)
        sv, isv = np.sqrt(eigvals), 1.0 / np.sqrt(eigvals)
        return (eigvecs * sv) @ eigvecs.T, (eigvecs * isv) @ eigvecs.T

    def fit_ea_whitening(X_train28, shrinkage=COV_SHRINKAGE):
        covs = trial_covariances(X_train28, shrinkage)
        _, W = matrix_sqrt_inv_sqrt(covs.mean(axis=0))
        return W

    def apply_ea_whitening_signal(X, W):
        return np.einsum("cd,ndt->nct", W, X)

    def tangent_vectorize(covs, eps=1e-8):
        N, Cc, _ = covs.shape
        out = np.empty((N, Cc * (Cc + 1) // 2), dtype=np.float32)
        iu = np.triu_indices(Cc)
        for n in range(N):
            eigvals, eigvecs = np.linalg.eigh(covs[n])
            eigvals = np.clip(eigvals, eps, None)
            log_mat = (eigvecs * np.log(eigvals)) @ eigvecs.T
            vec = log_mat[iu].copy()
            vec[iu[0] != iu[1]] *= math.sqrt(2.0)
            out[n] = vec
        return out

    def compute_binary_metrics(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        return {"sensitivity": float(sens), "specificity": float(spec),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)), "confusion_matrix": cm.tolist()}

    def linear_predict(coef, intercept, X):
        return ((X @ coef.T + intercept).ravel() > 0).astype(int)

    def fit_shrinkage_classifier(X_train_pca, y_train, X_cal_pca, y_cal):
        global_clf = LogisticRegression(C=LOGREG_C, max_iter=5000, random_state=RANDOM_SEED).fit(X_train_pca, y_train)
        local_clf_full = LogisticRegression(C=LOGREG_C, max_iter=5000, random_state=RANDOM_SEED).fit(X_cal_pca, y_cal)
        n_splits = max(min(SHRINK_CV_FOLDS, np.bincount(y_cal).min()), 2)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        shrink_scores = {s: [] for s in SHRINK_GRID}
        for tr_idx, val_idx in skf.split(X_cal_pca, y_cal):
            X_tr, X_val, y_tr, y_val = X_cal_pca[tr_idx], X_cal_pca[val_idx], y_cal[tr_idx], y_cal[val_idx]
            if len(np.unique(y_tr)) < 2:
                continue
            local_fold = LogisticRegression(C=LOGREG_C, max_iter=5000, random_state=RANDOM_SEED).fit(X_tr, y_tr)
            for shrink in SHRINK_GRID:
                coef_b = shrink * local_fold.coef_ + (1 - shrink) * global_clf.coef_
                icpt_b = shrink * local_fold.intercept_ + (1 - shrink) * global_clf.intercept_
                shrink_scores[shrink].append((linear_predict(coef_b, icpt_b, X_val) == y_val).mean())
        mean_scores = {s: (np.mean(v) if v else -1.0) for s, v in shrink_scores.items()}
        best_shrink = max(mean_scores, key=mean_scores.get)
        coef_final = best_shrink * local_clf_full.coef_ + (1 - best_shrink) * global_clf.coef_
        icpt_final = best_shrink * local_clf_full.intercept_ + (1 - best_shrink) * global_clf.intercept_
        return coef_final, icpt_final, best_shrink, global_clf

    def run_loso(X_np, y_np, subjects_np, arm_name):
        fold_records, all_test_acc = [], []
        for fold_idx, test_sub in enumerate(unique_subjects):
            fold_start = time.time()
            is_holdout = subjects_np == test_sub
            X_train28, y_train28 = X_np[~is_holdout], y_np[~is_holdout]
            X_k, y_k = X_np[is_holdout], y_np[is_holdout]

            mu = X_train28.mean(axis=(0, 2), keepdims=True)
            sd = X_train28.std(axis=(0, 2), keepdims=True) + 1e-6
            X_train28_z = ((X_train28 - mu) / sd).astype(np.float32)
            X_k_z = ((X_k - mu) / sd).astype(np.float32)

            W = fit_ea_whitening(X_train28_z)
            X_train28_aligned = apply_ea_whitening_signal(X_train28_z, W).astype(np.float32)
            X_k_aligned = apply_ea_whitening_signal(X_k_z, W).astype(np.float32)

            tan_train28 = tangent_vectorize(trial_covariances(X_train28_aligned))
            tan_k = tangent_vectorize(trial_covariances(X_k_aligned))
            tangent_dim = tan_train28.shape[1]

            feat_train28, feat_k = tan_train28, tan_k

            sss = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - CAL_FRACTION), random_state=RANDOM_SEED)
            cal_idx, test_idx = next(sss.split(feat_k, y_k))
            feat_cal, y_cal = feat_k[cal_idx], y_k[cal_idx]
            feat_test, y_test = feat_k[test_idx], y_k[test_idx]

            scaler = StandardScaler()
            feat_train28_z = scaler.fit_transform(feat_train28)
            feat_cal_z = scaler.transform(feat_cal)
            feat_test_z = scaler.transform(feat_test)

            n_components = min(PCA_MAX_COMPONENTS, feat_train28_z.shape[1] - 1, feat_train28_z.shape[0] - 1)
            pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
            X_train28_pca = pca.fit_transform(feat_train28_z)
            X_cal_pca = pca.transform(feat_cal_z)
            X_test_pca = pca.transform(feat_test_z)

            coef_final, icpt_final, best_shrink, global_clf = fit_shrinkage_classifier(
                X_train28_pca, y_train28, X_cal_pca, y_cal)
            pre_cal_acc = float((global_clf.predict(X_test_pca) == y_test).mean())
            final_preds = linear_predict(coef_final, icpt_final, X_test_pca)
            best_test_acc = float((final_preds == y_test).mean())
            metrics = compute_binary_metrics(y_test, final_preds)

            log.info(
                f"  [{arm_name}] FOLD {fold_idx+1}/{len(unique_subjects)} sub-{test_sub} -> "
                f"pre_cal={pre_cal_acc:.4f} post_cal={best_test_acc:.4f} (shrink={best_shrink:.2f}) "
                f"[{time.time()-fold_start:.0f}s]"
            )

            fold_records.append({
                "fold_index": fold_idx, "test_subject": str(test_sub),
                "tangent_dim": int(tangent_dim), "best_shrink_weight": float(best_shrink),
                "pre_calibration_acc": pre_cal_acc, "post_calibration_acc": best_test_acc,
                **metrics,
            })
            all_test_acc.append(best_test_acc)

        mean_acc, std_acc = float(np.mean(all_test_acc)), float(np.std(all_test_acc))
        log.info(f"[{arm_name}] Mean +/- Std Acc: {mean_acc:.4f} +/- {std_acc:.4f}")
        return {"arm": arm_name, "fold_results": fold_records, "mean_accuracy": mean_acc, "std_accuracy": std_acc,
                "n_folds": len(unique_subjects)}

    log.info("Running UNCLEANED arm ...")
    uncleaned_results = run_loso(Xu, yu, su, "uncleaned")

    log.info("Running ICA-CLEANED arm ...")
    cleaned_results = run_loso(Xc, yc, sc, "ica_cleaned")

    delta = cleaned_results["mean_accuracy"] - uncleaned_results["mean_accuracy"]
    log.info(
        f"\n{'='*70}\n  ICA-CLEANED vs UNCLEANED — {len(unique_subjects)} folds each\n{'='*70}\n"
        f"  Uncleaned (this run)     : {uncleaned_results['mean_accuracy']*100:.2f}% +/- {uncleaned_results['std_accuracy']*100:.2f}%\n"
        f"  ICA-cleaned (this run)   : {cleaned_results['mean_accuracy']*100:.2f}% +/- {cleaned_results['std_accuracy']*100:.2f}%\n"
        f"  Delta (cleaned-uncleaned): {delta*100:+.2f}pp\n"
        f"  Reference — matched-spatial-control.py (uncleaned, prior run): {MATCHED_SPATIAL_CONTROL_MEAN_ACC*100:.2f}%\n"
    )

    results_payload = {
        "condition": "Condition 4 — ICA-CLEANED vs UNCLEANED PREPROCESSING CONTROL (F-OCULAR(b))",
        "hyperparameters": {
            "cal_fraction": CAL_FRACTION, "pca_max_components": PCA_MAX_COMPONENTS,
            "logreg_C": LOGREG_C, "shrink_grid": SHRINK_GRID, "shrink_cv_folds": SHRINK_CV_FOLDS,
            "random_seed": RANDOM_SEED, "cov_shrinkage": COV_SHRINKAGE,
            "ica_n_components": ICA_N_COMPONENTS, "ica_method": ICA_METHOD,
            "ica_eog_surrogate_channel": ICA_EOG_SURROGATE_CH, "ica_eog_threshold": ICA_EOG_THRESHOLD,
        },
        "uncleaned": uncleaned_results,
        "ica_cleaned": cleaned_results,
        "delta_cleaned_minus_uncleaned": delta,
        "reference_matched_spatial_control_prior_run_mean_acc": MATCHED_SPATIAL_CONTROL_MEAN_ACC,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_payload, f, indent=2)
    volume.commit()
    log.info(f"Saved: {OUTPUT_JSON}")

    # =========================================================================
    # C3 plausibility assertions -- printed next to the numbers, not silent.
    # Run AFTER the write above so a failing assertion never suppresses the
    # diagnostic artifact.
    # =========================================================================
    for arm_res in (uncleaned_results, cleaned_results):
        acc = arm_res["mean_accuracy"]
        assert 0.0 <= acc <= 1.0, f"[C3 PLAUSIBILITY FAIL] {arm_res['arm']} mean accuracy {acc} outside [0,1]"
        assert arm_res["n_folds"] == 29, (
            f"[C3 PLAUSIBILITY FAIL] {arm_res['arm']} expected 29 folds, got {arm_res['n_folds']}"
        )
    log.info(f"  [C3] plausibility: both arms 29/29 folds, accuracies in [0,1] -- OK")

    return {
        "uncleaned_mean_accuracy": uncleaned_results["mean_accuracy"],
        "ica_cleaned_mean_accuracy": cleaned_results["mean_accuracy"],
        "delta_cleaned_minus_uncleaned": delta,
        "output_path": OUTPUT_JSON,
    }


@app.local_entrypoint()
def build():
    print("Stage 1/2 — Building ICA-cleaned dataset (raw -> filter -> resample -> ICA -> epoch) ...")
    result = build_ica_cleaned_dataset.remote()
    print("\n" + "="*60)
    for k, v in result.items():
        print(f"  {k:<40}: {v}")
    print("="*60)
    print("\nNext: modal run run_step4_ica_cleaned_control.py::main")


@app.local_entrypoint()
def main():
    print("Stage 2/2 — Comparing ICA-cleaned vs uncleaned matched-spatial-control LOSO ...")
    print("(Requires Stage 1 already run — see run_step4_ica_cleaned_control.py::build)\n")
    results = run_ica_vs_uncleaned_comparison.remote()
    print("\nICA-CLEANED vs UNCLEANED RESULTS:")
    for k, v in results.items():
        print(f"  {k:<35}: {v}")
