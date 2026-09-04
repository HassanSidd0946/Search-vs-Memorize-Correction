# =============================================================================
# scripts/ocular_margin_correlation.py
#
# command to run (both stages happen in one Modal call):
#   modal run scripts/ocular_margin_correlation.py::main
#
# F-OCULAR(c) — SURROGATE HEOG/VEOG vs. CLASSIFIER DECISION-MARGIN CORRELATION
# (AUDIT.md Phase 1, analysis-only, depends on Phase-2-scale LOSO training)
#
# WHY THIS SCRIPT EXISTS:
#   F-OCULAR(a) (channel ablation) and F-OCULAR(b) (ICA cleaning) both ask
#   "does accuracy survive removing/cleaning the suspect frontal signal?" —
#   a coarse, aggregate test. This script asks a finer-grained, per-trial
#   question instead: on the FULL 62-channel matched-spatial-control model
#   (identical to run_step4_matched_spatial_control.py), is the classifier's
#   continuous decision margin on each held-out test trial correlated with
#   how much ocular-artifact-like activity that specific trial contains? If
#   the model were substantially keying off blinks/saccades, trials with
#   large surrogate-EOG amplitude should systematically produce more
#   extreme (or more particular) decision margins. A near-zero correlation
#   is evidence AGAINST the ocular-confound hypothesis at the level of
#   individual trials, independent of and complementary to F-OCULAR(a)/(b)'s
#   aggregate accuracy comparisons.
#
# SURROGATE HEOG/VEOG DEFINITION (no dedicated EOG channel in this montage,
#   see AUDIT.md Phase 0.5 Priority 3 / F-OCULAR(a) header):
#     surrogate HEOG(t) = X[F7](t) - X[F8](t)   (horizontal eye-movement proxy)
#     surrogate VEOG(t) = X[Fp1](t) - X[Fp2](t) (blink/vertical-movement proxy)
#   Per trial, two summary statistics are computed for each surrogate signal:
#   variance across the epoch, and peak-to-peak amplitude (max - min).
#   Channel indices are VERIFIED (not assumed) via the same live sub-01
#   raw .vhdr lookup used by F-OCULAR(a) — see that script's header for the
#   full justification of why this matters for correctness.
#
# WHY THIS IS SELF-CONTAINED RATHER THAN READING A SEPARATE PHASE-2 ARTIFACT:
#   AUDIT.md's Fix-ID table describes this as depending on "Phase 2 trained
#   models for decision margins". No existing script currently persists
#   per-trial decision margins (the LOSO drivers only log aggregate fold
#   metrics), and bolting margin-logging onto an already-executed/verified
#   script like run_step4_matched_spatial_control.py would require a re-run
#   to regenerate that already-reported 70.78% number's artifacts. To avoid
#   that coupling, this script runs its OWN byte-for-byte-identical copy of
#   the matched-spatial-control LOSO pipeline (single seed=42, all 62
#   channels — the full/uncleaned/unablated model is the correct one to
#   probe here, since F-OCULAR(a)/(b) already cover the ablated/cleaned
#   cases) and captures the per-test-trial decision margin as it goes. The
#   expensive part (one full 29-fold LOSO training pass) is exactly the
#   "Phase 2 trained models" cost AUDIT.md's table refers to; the
#   correlation/CI computation on top of it is the "negligible once per-fold
#   margins exist" part.
#
# STATUS: written and syntax-verified only, per AUDIT.md — NOT YET EXECUTED.
#   Requires a real Modal account/volume; cannot be run from this sandbox.
# =============================================================================

import modal

app    = modal.App("bci-condition4-ocular-margin-correlation")
volume = modal.Volume.from_name("eeg-data-vol")

RAW_DATA_PATH = "/data/processed_eeg_all_subjects.npz"
OUTPUT_JSON   = "/data/results_ocular_margin_correlation.json"
VOLUME_PATH   = "/data"
PROBE_DIR     = "/data/_probe_openneuro_channel_order_margin_corr"

SFREQ, N_CHANNELS = 250, 62

# Surrogate HEOG/VEOG channel pairs (AUDIT.md Phase 0.5 Priority 3).
HEOG_PAIR = ("F7", "F8")
VEOG_PAIR = ("Fp1", "Fp2")

CAL_FRACTION       = 0.15
RANDOM_SEED        = 42
COV_SHRINKAGE       = 0.1
PCA_MAX_COMPONENTS = 35
LOGREG_C           = 1.0
SHRINK_GRID        = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS    = 3

N_BOOTSTRAP = 2000   # percentile-bootstrap resamples for Spearman rho CI

# Reference numbers (for logging/comparison only)
MATCHED_SPATIAL_CONTROL_MEAN_ACC = 0.7078   # run_step4_matched_spatial_control.py, full 29-fold, seed 42

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # scipy MUST be pinned alongside mne: unpinned scipy resolves to a
        # version that removed scipy.special.sph_harm (deprecated 1.15,
        # gone by 1.17), which mne==1.7.1 still imports at `import mne.io`
        # time -- crashes before any of our code runs. 1.14.1 matches the
        # already-proven-working pin in run_data_engine_on_modal.py and
        # run_step4_ica_cleaned_control.py's ::build stage.
        "numpy<2", "scikit-learn==1.4.2", "scipy==1.14.1",
        "mne==1.7.1", "openneuro-py==2024.2.0",
    )
)


@app.function(image=image, cpu=4.0, volumes={VOLUME_PATH: volume}, timeout=86400, memory=16384)
def run_ocular_margin_correlation():

    import os, glob, shutil, logging, time, math, json
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
    from sklearn.metrics import confusion_matrix, f1_score
    from scipy.stats import spearmanr

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("ocular-margin-correlation")

    np.random.seed(RANDOM_SEED)

    # =========================================================================
    # STEP 0: VERIFY THE TRUE CHANNEL ORDER (not assumed) — same approach as
    # F-OCULAR(a); see run_step4_matched_spatial_control_frontal_ablated.py header.
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
    needed = [*HEOG_PAIR, *VEOG_PAIR]
    missing = [c for c in needed if c not in verified_ch_names]
    assert not missing, f"Surrogate-EOG channel names not found in verified montage: {missing}"

    idx_f7, idx_f8   = verified_ch_names.index(HEOG_PAIR[0]), verified_ch_names.index(HEOG_PAIR[1])
    idx_fp1, idx_fp2 = verified_ch_names.index(VEOG_PAIR[0]), verified_ch_names.index(VEOG_PAIR[1])
    log.info(f"Verified indices — F7={idx_f7}, F8={idx_f8}, Fp1={idx_fp1}, Fp2={idx_fp2}")

    # =========================================================================
    # LOAD DATA + COMPUTE PER-TRIAL SURROGATE HEOG/VEOG FEATURES (once, up front)
    # =========================================================================
    raw = np.load(RAW_DATA_PATH, allow_pickle=True)
    X_np = raw["X"].astype(np.float32)
    y_np = raw["y"].astype(np.int64)
    subjects_np = raw["subjects"]
    N, C, T = X_np.shape
    N_CLASSES = int(y_np.max()) + 1
    assert C == N_CHANNELS and N_CLASSES == 2
    assert len(np.unique(subjects_np)) == 29, (
        f"Expected exactly 29 subjects (30 - sub-09 exclusion per AUDIT.md D2), "
        f"got {len(np.unique(subjects_np))}: {sorted(np.unique(subjects_np).tolist())}"
    )
    log.info(f"X: {X_np.shape} | Subjects total: {len(np.unique(subjects_np))}")

    surrogate_heog = X_np[:, idx_f7, :] - X_np[:, idx_f8, :]     # (N, T)
    surrogate_veog = X_np[:, idx_fp1, :] - X_np[:, idx_fp2, :]   # (N, T)

    heog_var = surrogate_heog.var(axis=1)
    heog_p2p = surrogate_heog.max(axis=1) - surrogate_heog.min(axis=1)
    veog_var = surrogate_veog.var(axis=1)
    veog_p2p = surrogate_veog.max(axis=1) - surrogate_veog.min(axis=1)
    # original_flat_index ties each of these per-trial values back to a
    # specific row of X_np/y_np/subjects_np, so they can be joined against
    # the test-set margins produced by the LOSO loop below regardless of
    # how each fold's train/cal/test split reorders things.
    original_flat_index = np.arange(N, dtype=np.int64)

    unique_subjects = sorted(np.unique(subjects_np).tolist())
    log.info(f"Running folds: {unique_subjects}")

    # =========================================================================
    # RIEMANNIAN / EA UTILITIES — IDENTICAL to run_step4_matched_spatial_control.py
    # =========================================================================
    def trial_covariances(X, shrinkage=COV_SHRINKAGE):
        Xc = X - X.mean(axis=2, keepdims=True)
        cov = np.einsum("nct,ndt->ncd", Xc, Xc) / (X.shape[2] - 1)
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
        N_, Cc, _ = covs.shape
        out = np.empty((N_, Cc * (Cc + 1) // 2), dtype=np.float32)
        iu = np.triu_indices(Cc)
        for n in range(N_):
            eigvals, eigvecs = np.linalg.eigh(covs[n])
            eigvals = np.clip(eigvals, eps, None)
            log_mat = (eigvecs * np.log(eigvals)) @ eigvecs.T
            vec = log_mat[iu].copy()
            vec[iu[0] != iu[1]] *= math.sqrt(2.0)
            out[n] = vec
        return out

    # =========================================================================
    # CALIBRATION UTILITIES — IDENTICAL to run_step4_matched_spatial_control.py
    # =========================================================================
    def compute_binary_metrics(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        return {"sensitivity": float(sens), "specificity": float(spec),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)), "confusion_matrix": cm.tolist()}

    def linear_margin(coef, intercept, X):
        # Continuous decision-function score BEFORE thresholding — this is
        # what F-OCULAR(c) correlates against surrogate ocular amplitude,
        # not the thresholded 0/1 prediction.
        return (X @ coef.T + intercept).ravel()

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
                preds = (linear_margin(coef_b, icpt_b, X_val) > 0).astype(int)
                shrink_scores[shrink].append((preds == y_val).mean())
        mean_scores = {s: (np.mean(v) if v else -1.0) for s, v in shrink_scores.items()}
        best_shrink = max(mean_scores, key=mean_scores.get)
        coef_final = best_shrink * local_clf_full.coef_ + (1 - best_shrink) * global_clf.coef_
        icpt_final = best_shrink * local_clf_full.intercept_ + (1 - best_shrink) * global_clf.intercept_
        return coef_final, icpt_final, best_shrink, global_clf

    # =========================================================================
    # LOSO — MATCHED SPATIAL-ONLY CONTROL, capturing per-test-trial decision
    # margins (single seed=42; this is an analysis-only supporting control,
    # not the CRITICAL D3-gated F-OCULAR(a), so it does not need 5 seeds)
    # =========================================================================
    fold_records, all_test_acc = [], []
    trial_margins, trial_true_labels, trial_pred_labels = [], [], []
    trial_original_index, trial_test_subject = [], []

    for fold_idx, test_sub in enumerate(unique_subjects):
        fold_start = time.time()
        log.info(f"\n{'='*70}\n  FOLD {fold_idx+1}/{len(unique_subjects)} — sub-{test_sub}\n{'='*70}")

        is_holdout = subjects_np == test_sub
        holdout_orig_idx = original_flat_index[is_holdout]

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

        # NOTE: track a LOCAL index into X_k/feat_k (0..len(X_k)-1) through the
        # split so we can map cal/test rows back to holdout_orig_idx afterward.
        local_idx = np.arange(len(y_k))
        sss = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - CAL_FRACTION), random_state=RANDOM_SEED)
        cal_idx, test_idx = next(sss.split(feat_k, y_k))
        feat_cal, y_cal = feat_k[cal_idx], y_k[cal_idx]
        feat_test, y_test = feat_k[test_idx], y_k[test_idx]
        test_orig_idx = holdout_orig_idx[test_idx]

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
        margins = linear_margin(coef_final, icpt_final, X_test_pca)
        final_preds = (margins > 0).astype(int)
        best_test_acc = float((final_preds == y_test).mean())
        metrics = compute_binary_metrics(y_test, final_preds)

        log.info(f"  RESULT -> pre_cal={pre_cal_acc:.4f}  post_cal={best_test_acc:.4f} (shrink={best_shrink:.2f}) "
                 f"| tangent_dim={tangent_dim} | n_test_trials={len(y_test)}")

        fold_records.append({
            "fold_index": fold_idx, "test_subject": str(test_sub),
            "tangent_dim": int(tangent_dim), "best_shrink_weight": float(best_shrink),
            "pre_calibration_acc": pre_cal_acc, "post_calibration_acc": best_test_acc,
            "n_test_trials": int(len(y_test)),
            **metrics,
        })
        all_test_acc.append(best_test_acc)

        trial_margins.append(margins)
        trial_true_labels.append(y_test)
        trial_pred_labels.append(final_preds)
        trial_original_index.append(test_orig_idx)
        trial_test_subject.extend([str(test_sub)] * len(y_test))

        log.info(f"  Fold elapsed: {time.time()-fold_start:.0f}s")

    mean_acc, std_acc = float(np.mean(all_test_acc)), float(np.std(all_test_acc))
    log.info(f"\nMatched-spatial-control (this run, margin-logging) mean acc: {mean_acc:.4f} +/- {std_acc:.4f} "
             f"(reference {MATCHED_SPATIAL_CONTROL_MEAN_ACC*100:.2f}%)")

    # =========================================================================
    # JOIN test-trial margins back to their surrogate HEOG/VEOG features via
    # original_flat_index, then compute pooled Spearman correlations + CIs.
    # =========================================================================
    all_margins = np.concatenate(trial_margins)
    all_orig_idx = np.concatenate(trial_original_index)
    all_true = np.concatenate(trial_true_labels)
    all_pred = np.concatenate(trial_pred_labels)

    n_test_trials_total = len(all_margins)
    log.info(f"Pooled held-out test trials with margins: {n_test_trials_total} "
             f"(out of {N} total trials; {CAL_FRACTION:.0%} of each fold's holdout goes to calibration, not here)")

    matched_heog_var = heog_var[all_orig_idx]
    matched_heog_p2p = heog_p2p[all_orig_idx]
    matched_veog_var = veog_var[all_orig_idx]
    matched_veog_p2p = veog_p2p[all_orig_idx]

    def spearman_with_bootstrap_ci(x, y, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
        rho, pval = spearmanr(x, y)
        rng = np.random.default_rng(seed)
        n = len(x)
        boot_rhos = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            sel = rng.integers(0, n, size=n)
            r, _ = spearmanr(x[sel], y[sel])
            boot_rhos[b] = r if not np.isnan(r) else 0.0
        ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])
        return {"spearman_rho": float(rho), "p_value": float(pval),
                "bootstrap_ci_95": [float(ci_lo), float(ci_hi)], "n_bootstrap": n_boot}

    correlations = {
        "margin_vs_surrogate_heog_variance": spearman_with_bootstrap_ci(all_margins, matched_heog_var),
        "margin_vs_surrogate_heog_peak_to_peak": spearman_with_bootstrap_ci(all_margins, matched_heog_p2p),
        "margin_vs_surrogate_veog_variance": spearman_with_bootstrap_ci(all_margins, matched_veog_var),
        "margin_vs_surrogate_veog_peak_to_peak": spearman_with_bootstrap_ci(all_margins, matched_veog_p2p),
        # |margin| (confidence magnitude, sign-agnostic) vs ocular amplitude,
        # in case the relationship is with classifier CONFIDENCE rather than
        # which class it favors.
        "abs_margin_vs_surrogate_heog_variance": spearman_with_bootstrap_ci(np.abs(all_margins), matched_heog_var),
        "abs_margin_vs_surrogate_veog_variance": spearman_with_bootstrap_ci(np.abs(all_margins), matched_veog_var),
    }

    for name, res in correlations.items():
        log.info(f"  {name}: rho={res['spearman_rho']:+.4f} (95% CI [{res['bootstrap_ci_95'][0]:+.4f}, "
                 f"{res['bootstrap_ci_95'][1]:+.4f}]), p={res['p_value']:.4g}")

    results_payload = {
        "condition": "F-OCULAR(c) — surrogate HEOG/VEOG vs. classifier decision-margin correlation "
                      "(matched-spatial-control, all 62 channels, seed=42)",
        "channel_verification": {
            "verified_channel_order_source": "sub-01 raw .vhdr, read via mne.io.read_raw_brainvision",
            "heog_pair": HEOG_PAIR, "heog_indices": [int(idx_f7), int(idx_f8)],
            "veog_pair": VEOG_PAIR, "veog_indices": [int(idx_fp1), int(idx_fp2)],
        },
        "n_folds": len(unique_subjects),
        "matched_spatial_control_mean_accuracy_this_run": mean_acc,
        "matched_spatial_control_std_accuracy_this_run": std_acc,
        "reference_matched_spatial_control_mean_acc": MATCHED_SPATIAL_CONTROL_MEAN_ACC,
        "hyperparameters": {
            "cal_fraction": CAL_FRACTION, "pca_max_components": PCA_MAX_COMPONENTS,
            "logreg_C": LOGREG_C, "shrink_grid": SHRINK_GRID, "shrink_cv_folds": SHRINK_CV_FOLDS,
            "random_seed": RANDOM_SEED, "cov_shrinkage": COV_SHRINKAGE,
        },
        "fold_results": fold_records,
        "n_test_trials_total": int(n_test_trials_total),
        "correlations": correlations,
        # Per-trial data retained for post-hoc plotting/re-analysis, not just
        # the aggregate correlation numbers.
        "per_trial": {
            "original_flat_index": all_orig_idx.tolist(),
            "test_subject": trial_test_subject,
            "true_label": all_true.tolist(),
            "predicted_label": all_pred.tolist(),
            "decision_margin": all_margins.tolist(),
            "surrogate_heog_variance": matched_heog_var.tolist(),
            "surrogate_heog_peak_to_peak": matched_heog_p2p.tolist(),
            "surrogate_veog_variance": matched_veog_var.tolist(),
            "surrogate_veog_peak_to_peak": matched_veog_p2p.tolist(),
        },
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
    assert 0.0 <= mean_acc <= 1.0, f"[C3 PLAUSIBILITY FAIL] mean accuracy {mean_acc} outside [0,1]"
    assert len(unique_subjects) == 29, f"[C3 PLAUSIBILITY FAIL] expected 29 folds, got {len(unique_subjects)}"
    assert n_test_trials_total > 0, f"[C3 PLAUSIBILITY FAIL] n_test_trials_total is {n_test_trials_total} (must be > 0)"
    for name, res in correlations.items():
        rho = res["spearman_rho"]
        assert np.isfinite(rho) and -1.0 <= rho <= 1.0, (
            f"[C3 PLAUSIBILITY FAIL] {name} spearman_rho={rho} outside [-1,1] or non-finite"
        )
    log.info(f"  [C3] plausibility: 29/29 folds, mean acc in [0,1], all rho in [-1,1] and finite, "
             f"{n_test_trials_total} test trials pooled -- OK")

    return {
        "matched_spatial_control_mean_accuracy_this_run": mean_acc,
        "n_test_trials_total": int(n_test_trials_total),
        "output_path": OUTPUT_JSON,
        **{k: v["spearman_rho"] for k, v in correlations.items()},
    }


@app.local_entrypoint()
def main():
    print("F-OCULAR(c) — Surrogate HEOG/VEOG vs. classifier decision-margin correlation")
    print(f"HEOG = X[{HEOG_PAIR[0]}] - X[{HEOG_PAIR[1]}], VEOG = X[{VEOG_PAIR[0]}] - X[{VEOG_PAIR[1]}] "
          "(channel indices verified live from sub-01's raw .vhdr, not assumed)")
    print("Runs one full 29-fold matched-spatial-control LOSO pass (all 62 channels, seed=42) "
          "to obtain per-test-trial decision margins, then correlates against per-trial "
          "surrogate-EOG variance/peak-to-peak via Spearman rho + bootstrap CI.\n")
    results = run_ocular_margin_correlation.remote()
    print("\nOCULAR MARGIN CORRELATION RESULTS:")
    for k, v in results.items():
        print(f"  {k:<45}: {v}")
