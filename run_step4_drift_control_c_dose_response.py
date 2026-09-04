# =============================================================================
# command to run:
#   modal run run_step4_drift_control_c_dose_response.py::main
# run_step4_drift_control_c_dose_response.py
#
# F-DRIFT-C — PARAMETRIC DOSE-RESPONSE SWEEP (new fix-ID, added 2026-08-19,
# AUDIT.md fix register; DECISIONS.md pre-registered rule)
#
# WHY THIS SCRIPT EXISTS:
#   F-DRIFT (early/late split, ~half-block separation -> pre_cal 0.6418) and
#   F-DRIFT-B (odd/even interleaving, ~zero separation -> pre_cal 0.5085,
#   chance) together establish that decodability depends on temporal
#   separation, but only at two points. This script parametrizes the
#   separation continuously to test whether decodability scales SMOOTHLY
#   with separation (the central claim for the methodological paper) or
#   jumps/plateaus in a way inconsistent with a simple drift account.
#
# TEST DESIGN (pre-registered in DECISIONS.md BEFORE this script runs):
#   Pseudo-label defined by ALTERNATING RUNS OF LENGTH k within each real
#   class block: pseudo_label(i) = (i // k) % 2, where i is the 0-indexed
#   within-class chronological trial position (same convention as
#   run_step4_drift_control.py / run_step4_drift_control_b_interleaved.py).
#     k=1   is EXACTLY F-DRIFT-B's interleaving.
#     k=100 (~n/2 for n~200 trials/class/subject) is EXACTLY F-DRIFT's
#           early/late split.
#   Sweep k in {1, 2, 5, 10, 25, 50, 100}. Both real classes (Search-only,
#   Memorize-only) tested separately, single seed=42, full 29-fold LOSO,
#   identical calibrated pipeline (byte-identical EA/tangent/PCA/shrinkage-
#   calibration to the rest of the pre-F3 Batch 1/F-DRIFT family), for
#   every k.
#
# MANDATORY ENDPOINT EQUIVALENCE CHECK (HALT if it fails):
#   k=1 and k=100 must reproduce F-DRIFT-B (pre=0.5085, post=0.5023) and
#   F-DRIFT (pre=0.6418, post=0.7112) respectively, within +/-0.03 on
#   post-calibration accuracy. If either endpoint falls outside tolerance,
#   this script HALTS (hard assertion failure) before any interpretation
#   is attempted -- the parameterization would not be equivalent to the
#   already-accepted controls, and the curve would not mean anything
#   until debugged.
#
# SECONDS CONVERSION (real trial onset times, NOT approximated):
#   k (a trial count) is converted to mean temporal separation in seconds
#   using REAL trial onset times, extracted via mne.events_from_annotations
#   on each subject's raw .vhdr/.vmrk, using the EXACT SAME EVENT_ID
#   mapping already used in production (run_data_engine_on_modal.py).
#   Per-subject, per-class mean inter-trial interval (ITI) is computed
#   from these real onsets; seconds_at_k = k * mean_ITI_seconds (cross-
#   subject mean, with its own std reported). The number of stimulus
#   events found via this independent extraction is cross-checked against
#   the known per-subject/class trial counts already in
#   processed_eeg_all_subjects.npz -- a mismatch means the extraction is
#   misaligned with the processed dataset and is NOT trusted (hard fail).
#
# PRE-REGISTERED INTERPRETATION RULE (DECISIONS.md, fixed BEFORE running,
# applied to PRE-calibration accuracy -- the primary axis, uncontaminated
# by within-subject calibration):
#   monotone rise from chance toward ~0.64 as seconds of separation increase
#       -> decodability scales with temporal separation; central figure of
#          the methodological paper.
#   flat or non-monotone
#       -> the drift account is incomplete; report plainly and stop for
#          discussion before interpreting further.
#
# Usage: modal run run_step4_drift_control_c_dose_response.py::main
# =============================================================================

import modal

app    = modal.App("bci-condition4-drift-control-c-dose-response")
volume = modal.Volume.from_name("eeg-data-vol")

RAW_DATA_PATH = "/data/processed_eeg_all_subjects.npz"
OUTPUT_JSON   = "/data/results_condition4_drift_control_c_dose_response.json"
OUTPUT_PLOT   = "/data/results_condition4_drift_control_c_dose_response.png"
VOLUME_PATH   = "/data"
PROBE_DIR     = "/data/_probe_openneuro_channel_order_drift_c"

SFREQ, N_CHANNELS = 250, 62

RANDOM_SEED         = 42
COV_SHRINKAGE       = 0.1
PCA_MAX_COMPONENTS  = 35
LOGREG_C            = 1.0
CAL_FRACTION        = 0.15
SHRINK_GRID         = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS     = 3

K_SWEEP = [1, 2, 5, 10, 25, 50, 100]

# Exact production event-marker mapping (run_data_engine_on_modal.py's
# EVENT_ID) -- reused verbatim so the onset-time extraction matches the
# processed dataset's trial identity exactly.
EVENT_ID = {
    "Stimulus/ 10": 0, "Stimulus/ 11": 0,
    "Stimulus/ 12": 0, "Stimulus/ 13": 0,
    "Stimulus/ 20": 1, "Stimulus/ 21": 1,
    "Stimulus/ 22": 1, "Stimulus/ 23": 1,
}

# DECISIONS.md pre-registered reference numbers + tolerance (fixed BEFORE
# this script runs) for the mandatory endpoint equivalence check.
FDRIFT_B_PRE_CAL_REFERENCE  = 0.5085   # k=1 endpoint
FDRIFT_B_POST_CAL_REFERENCE = 0.5023
FDRIFT_PRE_CAL_REFERENCE    = 0.6418   # k=100 endpoint
FDRIFT_POST_CAL_REFERENCE   = 0.7112
REAL_LABEL_PRE_CAL_REFERENCE = 0.5201  # for the plot's horizontal reference line
ENDPOINT_TOLERANCE = 0.03

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # scipy MUST be pinned alongside mne -- see AUDIT.md's scipy/mne
        # correction (2026-08-19): unpinned scipy resolves to a version
        # that removed scipy.special.sph_harm, which mne==1.7.1 still
        # imports. 1.14.1 matches the already-proven-working pin.
        "numpy<2", "scikit-learn==1.4.2", "scipy==1.14.1",
        "mne==1.7.1", "openneuro-py==2024.2.0", "matplotlib==3.9.2",
    )
)


@app.function(image=image, cpu=4.0, volumes={VOLUME_PATH: volume}, timeout=86400, memory=16384)
def run_drift_control_c():

    import os, glob, shutil, logging, time, math, json
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
    from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("condition4-drift-control-c")

    np.random.seed(RANDOM_SEED)

    raw_npz = np.load(RAW_DATA_PATH, allow_pickle=True)
    X_np = raw_npz["X"].astype(np.float32)
    y_np = raw_npz["y"].astype(np.int64)
    subjects_np = raw_npz["subjects"]
    N, C, T = X_np.shape
    N_CLASSES = int(y_np.max()) + 1
    assert C == N_CHANNELS and N_CLASSES == 2
    assert len(np.unique(subjects_np)) == 29, (
        f"Expected exactly 29 subjects (30 - sub-09 exclusion per AUDIT.md D2), "
        f"got {len(np.unique(subjects_np))}"
    )
    unique_subjects = sorted(np.unique(subjects_np).tolist())
    log.info(f"X: {X_np.shape} | Subjects: {len(unique_subjects)}")

    # =========================================================================
    # STEP 0: REAL TRIAL ONSET TIMES -- extracted independently per subject
    # via mne.events_from_annotations on the raw .vhdr/.vmrk, using the exact
    # production EVENT_ID mapping. NOT approximated. Cross-checked against
    # the processed dataset's own per-subject/class trial counts.
    # =========================================================================
    log.info(f"\n{'='*70}\n  STEP 0: extracting REAL trial onset times for all 29 subjects\n{'='*70}")
    import mne
    import openneuro
    mne.set_log_level("WARNING")

    per_subject_class_iti = {}   # sub -> {0: seconds, 1: seconds}
    for sub in unique_subjects:
        sub_tag = f"sub-{sub}"
        sub_probe_dir = os.path.join(PROBE_DIR, sub_tag)
        os.makedirs(sub_probe_dir, exist_ok=True)
        openneuro.download(dataset="ds005189", target_dir=sub_probe_dir, include=[sub_tag])
        vhdr_files = glob.glob(os.path.join(sub_probe_dir, "**", "*.vhdr"), recursive=True)
        assert vhdr_files, f"Could not find {sub_tag}'s .vhdr under {sub_probe_dir}"
        raw = mne.io.read_raw_brainvision(vhdr_files[0], preload=False, verbose=False)
        events, event_id_found = mne.events_from_annotations(raw, event_id=EVENT_ID, verbose=False)
        assert len(events) > 0, f"[{sub_tag}] zero stimulus events found -- onset extraction broken"
        sfreq = raw.info["sfreq"]
        onset_seconds = events[:, 0] / sfreq
        labels = events[:, 2]

        # Cross-check against the ALREADY-PROCESSED dataset's own counts.
        sub_mask_npz = subjects_np == sub
        class_iti = {}
        for cls in (0, 1):
            n_processed = int((y_np[sub_mask_npz] == cls).sum())
            cls_onsets = np.sort(onset_seconds[labels == cls])
            n_extracted = len(cls_onsets)
            assert n_extracted == n_processed, (
                f"[{sub_tag}] class {cls}: onset-extraction found {n_extracted} events but the "
                f"processed dataset has {n_processed} trials -- MISMATCH, onset-time extraction is "
                f"NOT trusted, halting before any k-sweep computation."
            )
            mean_iti = float((cls_onsets[-1] - cls_onsets[0]) / (n_extracted - 1)) if n_extracted > 1 else float("nan")
            class_iti[cls] = {"n_events": n_extracted, "mean_iti_seconds": mean_iti}
        per_subject_class_iti[sub] = class_iti
        log.info(f"  [{sub_tag}] class0: n={class_iti[0]['n_events']} ITI={class_iti[0]['mean_iti_seconds']:.3f}s | "
                 f"class1: n={class_iti[1]['n_events']} ITI={class_iti[1]['mean_iti_seconds']:.3f}s")
        shutil.rmtree(sub_probe_dir, ignore_errors=True)

    all_itis = [
        per_subject_class_iti[sub][cls]["mean_iti_seconds"]
        for sub in unique_subjects for cls in (0, 1)
        if not math.isnan(per_subject_class_iti[sub][cls]["mean_iti_seconds"])
    ]
    mean_iti_seconds = float(np.mean(all_itis))
    std_iti_seconds = float(np.std(all_itis))
    log.info(f"\n  Cross-subject mean ITI: {mean_iti_seconds:.4f}s +/- {std_iti_seconds:.4f}s "
             f"(n={len(all_itis)} subject-class cells, all onset counts verified against the "
             f"processed dataset)")

    # =========================================================================
    # RIEMANNIAN / EA UTILITIES -- IDENTICAL to the rest of the F-DRIFT family.
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

    def fit_ea_whitening(X_train, shrinkage=COV_SHRINKAGE):
        covs = trial_covariances(X_train, shrinkage)
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

    def compute_binary_metrics(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        return {"sensitivity": float(sens), "specificity": float(spec),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)), "confusion_matrix": cm.tolist()}

    # C3 hardening (2026-08-20, per F-DRIFT-E's INVALID-DESIGN failure -- "all accuracies in
    # [0,1]" is vacuous against a base-rate artifact). Every classification result must report
    # class balance, majority-class rate, accuracy-minus-majority-rate, and balanced accuracy,
    # and fails loudly if balance falls outside 45/55 unless the caller explicitly declares an
    # imbalanced design.
    def c3_balance_check(y_true, y_pred, acc, label, declared_imbalanced_design=False):
        counts = np.bincount(y_true, minlength=2)
        n = len(y_true)
        balance = (counts / n).tolist()
        majority_rate = float(counts.max() / n)
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        lift = float(acc - majority_rate)
        print(f"    [C3] {label}: class_balance={[round(b, 4) for b in balance]} "
              f"majority_rate={majority_rate:.4f} acc={acc:.4f} "
              f"acc_minus_majority={lift:+.4f} balanced_acc={bal_acc:.4f}")
        if not declared_imbalanced_design:
            assert 0.45 <= min(balance) and max(balance) <= 0.55, (
                f"[C3 FAIL] {label}: class balance {balance} is outside the 45/55 band and this "
                "script does not declare an imbalanced design. Fix the design (e.g. balance the "
                "pseudo-classes) or set declared_imbalanced_design=True and treat balanced_accuracy "
                "as the primary metric."
            )
        return {"class_balance": balance, "majority_class_rate": majority_rate,
                "accuracy_minus_majority_rate": lift, "balanced_accuracy": bal_acc}

    def linear_predict(coef, intercept, X):
        return ((X @ coef.T + intercept).ravel() > 0).astype(int)

    def fit_shrinkage_classifier(X_train_pca, y_train, X_cal_pca, y_cal, seed):
        global_clf = LogisticRegression(C=LOGREG_C, max_iter=5000, random_state=seed).fit(X_train_pca, y_train)
        local_clf_full = LogisticRegression(C=LOGREG_C, max_iter=5000, random_state=seed).fit(X_cal_pca, y_cal)
        n_splits = max(min(SHRINK_CV_FOLDS, np.bincount(y_cal).min()), 2)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        shrink_scores = {s: [] for s in SHRINK_GRID}
        for tr_idx, val_idx in skf.split(X_cal_pca, y_cal):
            X_tr, X_val, y_tr, y_val = X_cal_pca[tr_idx], X_cal_pca[val_idx], y_cal[tr_idx], y_cal[val_idx]
            if len(np.unique(y_tr)) < 2:
                continue
            local_fold = LogisticRegression(C=LOGREG_C, max_iter=5000, random_state=seed).fit(X_tr, y_tr)
            for shrink in SHRINK_GRID:
                coef_b = shrink * local_fold.coef_ + (1 - shrink) * global_clf.coef_
                icpt_b = shrink * local_fold.intercept_ + (1 - shrink) * global_clf.intercept_
                shrink_scores[shrink].append((linear_predict(coef_b, icpt_b, X_val) == y_val).mean())
        mean_scores = {s: (np.mean(v) if v else -1.0) for s, v in shrink_scores.items()}
        best_shrink = max(mean_scores, key=mean_scores.get)
        coef_final = best_shrink * local_clf_full.coef_ + (1 - best_shrink) * global_clf.coef_
        icpt_final = best_shrink * local_clf_full.intercept_ + (1 - best_shrink) * global_clf.intercept_
        return coef_final, icpt_final, best_shrink, global_clf

    # =========================================================================
    # ONE (k, real_class) PSEUDO-LABEL TEST -- pseudo_label(i) = (i // k) % 2.
    # =========================================================================
    def run_drift_test_c(real_class, real_class_name, k):
        idx_list, y_pseudo_list, subj_list = [], [], []
        for sub in unique_subjects:
            sub_idx = np.where(subjects_np == sub)[0]
            cls_idx = sub_idx[y_np[sub_idx] == real_class]
            n = len(cls_idx)
            assert n >= 2, f"sub-{sub} has only {n} trials of class {real_class}"
            pseudo = ((np.arange(n) // k) % 2).astype(np.int64)
            idx_list.append(cls_idx)
            y_pseudo_list.append(pseudo)
            subj_list.append(np.full(n, sub))
        pseudo_idx = np.concatenate(idx_list)
        y_pseudo = np.concatenate(y_pseudo_list)
        subjects_pseudo = np.concatenate(subj_list)
        X_pseudo = X_np[pseudo_idx]

        fold_records = []
        pre_cal_accs, post_cal_accs = [], []

        for fold_idx, test_sub in enumerate(unique_subjects):
            is_holdout = subjects_pseudo == test_sub
            X_train28, y_train28 = X_pseudo[~is_holdout], y_pseudo[~is_holdout]
            X_k, y_k = X_pseudo[is_holdout], y_pseudo[is_holdout]

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

            sss = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - CAL_FRACTION), random_state=RANDOM_SEED)
            cal_idx, test_idx = next(sss.split(tan_k, y_k))
            feat_cal, y_cal = tan_k[cal_idx], y_k[cal_idx]
            feat_test, y_test = tan_k[test_idx], y_k[test_idx]

            scaler = StandardScaler()
            feat_train28_z = scaler.fit_transform(tan_train28)
            feat_cal_z = scaler.transform(feat_cal)
            feat_test_z = scaler.transform(feat_test)

            n_components = min(PCA_MAX_COMPONENTS, feat_train28_z.shape[1] - 1, feat_train28_z.shape[0] - 1)
            pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
            X_train28_pca = pca.fit_transform(feat_train28_z)
            X_cal_pca = pca.transform(feat_cal_z)
            X_test_pca = pca.transform(feat_test_z)

            coef_final, icpt_final, best_shrink, global_clf = fit_shrinkage_classifier(
                X_train28_pca, y_train28, X_cal_pca, y_cal, RANDOM_SEED)
            pre_cal_acc = float((global_clf.predict(X_test_pca) == y_test).mean())
            final_preds = linear_predict(coef_final, icpt_final, X_test_pca)
            post_cal_acc = float((final_preds == y_test).mean())
            metrics = compute_binary_metrics(y_test, final_preds)
            pre_cal_plaus = c3_balance_check(y_test, global_clf.predict(X_test_pca), pre_cal_acc,
                                              f"k={k} {real_class_name} fold sub-{test_sub} pre_cal")
            post_cal_plaus = c3_balance_check(y_test, final_preds, post_cal_acc,
                                               f"k={k} {real_class_name} fold sub-{test_sub} post_cal")

            fold_records.append({
                "fold_index": fold_idx, "test_subject": str(test_sub),
                "tangent_dim": int(tangent_dim), "best_shrink_weight": float(best_shrink),
                "pre_calibration_acc": pre_cal_acc, "post_calibration_acc": post_cal_acc,
                "pre_cal_plausibility": pre_cal_plaus, "post_cal_plausibility": post_cal_plaus,
                **metrics,
            })
            pre_cal_accs.append(pre_cal_acc)
            post_cal_accs.append(post_cal_acc)

        pre_mean, pre_std = float(np.mean(pre_cal_accs)), float(np.std(pre_cal_accs))
        post_mean, post_std = float(np.mean(post_cal_accs)), float(np.std(post_cal_accs))
        return {
            "k": k, "real_class": real_class, "real_class_name": real_class_name,
            "fold_results": fold_records,
            "pre_calibration_accuracy_mean": pre_mean, "pre_calibration_accuracy_std": pre_std,
            "post_calibration_accuracy_mean": post_mean, "post_calibration_accuracy_std": post_std,
        }

    # =========================================================================
    # THE SWEEP -- 7 k-values x 2 real classes = 14 full 29-fold LOSO passes.
    # =========================================================================
    sweep_results = {}
    for k in K_SWEEP:
        sweep_start = time.time()
        log.info(f"\n{'#'*70}\n  k={k}\n{'#'*70}")
        res_search = run_drift_test_c(0, "search_only", k)
        res_memorize = run_drift_test_c(1, "memorize_only", k)
        combined_pre = float(np.mean([res_search["pre_calibration_accuracy_mean"],
                                       res_memorize["pre_calibration_accuracy_mean"]]))
        combined_post = float(np.mean([res_search["post_calibration_accuracy_mean"],
                                        res_memorize["post_calibration_accuracy_mean"]]))
        # std across the pooled 58 (29x2) per-fold accuracies, for plot error bars.
        pooled_pre = [r["pre_calibration_acc"] for r in res_search["fold_results"]] + \
                     [r["pre_calibration_acc"] for r in res_memorize["fold_results"]]
        pooled_post = [r["post_calibration_acc"] for r in res_search["fold_results"]] + \
                      [r["post_calibration_acc"] for r in res_memorize["fold_results"]]
        seconds_at_k = k * mean_iti_seconds
        sweep_results[k] = {
            "search_only": res_search, "memorize_only": res_memorize,
            "combined_pre_calibration_accuracy": combined_pre,
            "combined_post_calibration_accuracy": combined_post,
            "combined_pre_calibration_std_across_folds": float(np.std(pooled_pre)),
            "combined_post_calibration_std_across_folds": float(np.std(pooled_post)),
            "seconds_at_k": seconds_at_k,
        }
        log.info(f"  k={k} ({seconds_at_k:.1f}s) -> combined pre_cal={combined_pre:.4f} "
                 f"post_cal={combined_post:.4f} [{time.time()-sweep_start:.0f}s elapsed]")

    # =========================================================================
    # MANDATORY ENDPOINT EQUIVALENCE CHECK -- HALT if k=1/k=100 don't match
    # F-DRIFT-B/F-DRIFT within tolerance.
    # =========================================================================
    k1 = sweep_results[1]
    k100 = sweep_results[100]
    endpoint_checks = {
        "k1_vs_fdrift_b_pre": abs(k1["combined_pre_calibration_accuracy"] - FDRIFT_B_PRE_CAL_REFERENCE),
        "k1_vs_fdrift_b_post": abs(k1["combined_post_calibration_accuracy"] - FDRIFT_B_POST_CAL_REFERENCE),
        "k100_vs_fdrift_pre": abs(k100["combined_pre_calibration_accuracy"] - FDRIFT_PRE_CAL_REFERENCE),
        "k100_vs_fdrift_post": abs(k100["combined_post_calibration_accuracy"] - FDRIFT_POST_CAL_REFERENCE),
    }
    log.info(f"\n{'='*70}\n  ENDPOINT EQUIVALENCE CHECK (tolerance={ENDPOINT_TOLERANCE})\n{'='*70}")
    for name, diff in endpoint_checks.items():
        log.info(f"  {name}: diff={diff:.4f} {'OK' if diff <= ENDPOINT_TOLERANCE else 'FAIL'}")
    failing_checks = [name for name, diff in endpoint_checks.items() if diff > ENDPOINT_TOLERANCE]
    if failing_checks:
        # Still write what we have before halting, so a failure leaves a diagnostic artifact.
        partial_payload = {
            "HALTED": True,
            "reason": f"Endpoint equivalence check FAILED: {failing_checks}. The k=1/k=100 "
                      f"parameterization does not reproduce F-DRIFT-B/F-DRIFT within "
                      f"+/-{ENDPOINT_TOLERANCE} -- the dose-response curve does not mean anything "
                      f"until this is debugged.",
            "endpoint_checks": endpoint_checks,
            "sweep_results_so_far": sweep_results,
            "mean_iti_seconds": mean_iti_seconds, "std_iti_seconds": std_iti_seconds,
        }
        with open(OUTPUT_JSON, "w") as f:
            json.dump(partial_payload, f, indent=2)
        volume.commit()
        raise AssertionError(
            f"[HALT] Endpoint equivalence check failed: {failing_checks}. Diagnostic written to "
            f"{OUTPUT_JSON}. Per DECISIONS.md, the dose-response curve must not be interpreted "
            f"until this is resolved."
        )
    log.info("  All endpoint checks passed -- parameterization is equivalent, proceeding to interpret the curve.")

    # =========================================================================
    # PRE-REGISTERED INTERPRETATION (applied to PRE-calibration accuracy).
    # =========================================================================
    k_sorted = sorted(K_SWEEP)
    pre_cal_curve = [sweep_results[k]["combined_pre_calibration_accuracy"] for k in k_sorted]
    is_monotone_rising = all(pre_cal_curve[i] <= pre_cal_curve[i + 1] + 1e-9 for i in range(len(pre_cal_curve) - 1))
    rises_toward_target = pre_cal_curve[-1] >= 0.60   # "toward ~0.64" -- treat >=0.60 as qualifying
    if is_monotone_rising and rises_toward_target:
        curve_verdict = ("MONOTONE RISE from chance toward ~0.64 -- decodability scales with temporal "
                          "separation. This is the central figure of the methodological paper.")
    else:
        curve_verdict = ("FLAT OR NON-MONOTONE -- the drift account is incomplete. Reported plainly; "
                          "do not interpret further without discussion.")
    log.info(f"\n{'='*70}\n  F-DRIFT-C SUMMARY\n{'='*70}")
    for k in k_sorted:
        r = sweep_results[k]
        log.info(f"  k={k:>4} ({r['seconds_at_k']:>7.1f}s): pre_cal={r['combined_pre_calibration_accuracy']:.4f} "
                 f"post_cal={r['combined_post_calibration_accuracy']:.4f}")
    log.info(f"  Monotone rising? {is_monotone_rising} | reaches >=0.60? {rises_toward_target}")
    log.info(f"  DECISIONS.md VERDICT: {curve_verdict}")

    # =========================================================================
    # PLOT: pre_cal vs seconds of separation, real-label pre_cal as horizontal
    # reference line.
    # =========================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [sweep_results[k]["seconds_at_k"] for k in k_sorted]
    ys = [sweep_results[k]["combined_pre_calibration_accuracy"] for k in k_sorted]
    yerr = [sweep_results[k]["combined_pre_calibration_std_across_folds"] for k in k_sorted]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=4, label="Pseudo-label pre_cal (F-DRIFT-C)")
    ax.axhline(REAL_LABEL_PRE_CAL_REFERENCE, linestyle="--", color="gray",
               label=f"Real-label pre_cal ({REAL_LABEL_PRE_CAL_REFERENCE})")
    ax.axhline(0.5, linestyle=":", color="lightgray", label="Chance (0.5)")
    ax.set_xlabel("Mean temporal separation (seconds)")
    ax.set_ylabel("Pre-calibration accuracy")
    ax.set_title("F-DRIFT-C: decodability vs. temporal separation")
    ax.legend(loc="lower right")
    ax.set_ylim(0.4, 0.75)
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=150)
    plt.close(fig)
    log.info(f"  Saved plot: {OUTPUT_PLOT}")

    results_payload = {
        "condition": "F-DRIFT-C — parametric dose-response sweep: pseudo_label(i) = (i // k) % 2, "
                      "k in {1,2,5,10,25,50,100}, both real classes, identical calibrated pipeline, "
                      "single seed=42, full 29-fold LOSO per (k, class)",
        "hyperparameters": {
            "pca_max_components": PCA_MAX_COMPONENTS, "logreg_C": LOGREG_C,
            "random_seed": RANDOM_SEED, "cov_shrinkage": COV_SHRINKAGE, "cal_fraction": CAL_FRACTION,
            "k_sweep": K_SWEEP,
        },
        "onset_time_extraction": {
            "mean_iti_seconds": mean_iti_seconds, "std_iti_seconds": std_iti_seconds,
            "per_subject_class_iti": per_subject_class_iti,
        },
        "endpoint_equivalence_check": {"checks": endpoint_checks, "tolerance": ENDPOINT_TOLERANCE, "passed": True},
        "sweep_results": {str(k): sweep_results[k] for k in k_sorted},
        "curve": {"k_values": k_sorted, "seconds": xs, "pre_cal": ys, "pre_cal_std": yerr},
        "is_monotone_rising": is_monotone_rising,
        "reaches_target": rises_toward_target,
        "decisions_md_verdict": curve_verdict,
        "reference_numbers": {
            "real_label_pre_cal": REAL_LABEL_PRE_CAL_REFERENCE,
            "fdrift_b_pre_cal": FDRIFT_B_PRE_CAL_REFERENCE, "fdrift_b_post_cal": FDRIFT_B_POST_CAL_REFERENCE,
            "fdrift_pre_cal": FDRIFT_PRE_CAL_REFERENCE, "fdrift_post_cal": FDRIFT_POST_CAL_REFERENCE,
        },
        "output_plot_path": OUTPUT_PLOT,
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
    for k in k_sorted:
        r = sweep_results[k]
        assert 0.0 <= r["combined_pre_calibration_accuracy"] <= 1.0, (
            f"[C3 PLAUSIBILITY FAIL] k={k} combined pre_cal outside [0,1]"
        )
        assert 0.0 <= r["combined_post_calibration_accuracy"] <= 1.0, (
            f"[C3 PLAUSIBILITY FAIL] k={k} combined post_cal outside [0,1]"
        )
        assert len(r["search_only"]["fold_results"]) == 29 and len(r["memorize_only"]["fold_results"]) == 29, (
            f"[C3 PLAUSIBILITY FAIL] k={k} expected 29 folds per class"
        )
    assert mean_iti_seconds > 0, f"[C3 PLAUSIBILITY FAIL] mean_iti_seconds={mean_iti_seconds} must be > 0"
    log.info(f"  [C3] plausibility: {len(K_SWEEP)} k-values x 2 classes x 29 folds, all accuracies in "
             f"[0,1], mean ITI > 0 -- OK")

    return {
        "mean_iti_seconds": mean_iti_seconds,
        "endpoint_checks_passed": True,
        "k1_pre_cal": sweep_results[1]["combined_pre_calibration_accuracy"],
        "k100_pre_cal": sweep_results[100]["combined_pre_calibration_accuracy"],
        "is_monotone_rising": is_monotone_rising,
        "decisions_md_verdict": curve_verdict,
        "output_path": OUTPUT_JSON,
        "output_plot_path": OUTPUT_PLOT,
    }


@app.local_entrypoint()
def main():
    print("F-DRIFT-C — Parametric dose-response sweep")
    print(f"k in {K_SWEEP} -- k=1 reproduces F-DRIFT-B, k=100 reproduces F-DRIFT (endpoint check is mandatory).")
    print("Step 0: extract REAL trial onset times for all 29 subjects (one-time, ~15-20 min).")
    print("Then: 7 k-values x 2 real classes x 29-fold LOSO, single seed=42.")
    print("Pre-registered rule (DECISIONS.md): monotone rise toward ~0.64 -> central figure of the paper | "
          "flat/non-monotone -> drift account incomplete, stop for discussion.")
    print("Est. cost: ~1.5-2 hours total.\n")
    results = run_drift_control_c.remote()
    print("\nF-DRIFT-C RESULTS:")
    for k, v in results.items():
        print(f"  {k:<28}: {v}")
