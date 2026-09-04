# =============================================================================
# command to run:
#   modal run run_step4_drift_control_g_real_labels_onset_excluded.py::main
# run_step4_drift_control_g_real_labels_onset_excluded.py
#
# F-DRIFT-G — REAL LABELS UNDER ONSET EXCLUSION, THE MISSING CELL (new
# fix-ID, added 2026-08-20, AUDIT.md fix register; DECISIONS.md
# pre-registered rule)
#
# WHY THIS SCRIPT EXISTS:
#   F-DRIFT-F(a) (ledger L028) showed that dropping the first 50 trials of
#   each block collapses the PSEUDO-label k-sweep to chance (0.4974-0.5037
#   across all k) -- the F-DRIFT-C "temporal separation" interpretation is
#   WITHDRAWN, replaced by an onset-driven account. But the REAL
#   Search-vs-Memorize contrast under the SAME onset exclusion has never
#   been tested. This is the one remaining cell that could change the
#   project's conclusion: if the real contrast SURVIVES onset exclusion at
#   a meaningfully elevated accuracy, part of the effect may be genuine
#   task signal the onset artifact was masking, not merely confounding
#   with it.
#
# TEST DESIGN (pre-registered in DECISIONS.md BEFORE this script runs):
#   Drop the first 50 trials of each block for every subject (IDENTICAL
#   truncation to F-DRIFT-F(a)), REAL Search-vs-Memorize labels, identical
#   calibrated pipeline (EA/tangent/PCA/shrinkage-calibration, matching
#   the rest of the pre-F3 F-DRIFT family), single seed=42, full 29-fold
#   LOSO. Reports BALANCED pre_cal and post_cal (per FIX 1's hardening).
#
# PRE-REGISTERED VERDICT RULE (DECISIONS.md, fixed BEFORE this script
# runs, balanced-accuracy terms, applied to post_cal):
#   post_cal < 0.55
#       -> the real contrast is ENTIRELY onset artifact -- the
#          methodological null result is complete and final.
#   post_cal >= 0.65
#       -> genuine task signal exists and was masked by the onset
#          artifact -- this becomes a POSITIVE result and the paper
#          changes completely. HALT and report before any interpretation.
#   0.55-0.65
#       -> partial -- report both components, with the share
#          attributable to onset quantified.
#   pre_cal is ALSO reported regardless of branch -- a rise there
#   specifically would indicate genuine cross-subject task transfer,
#   which nothing in this audit has shown so far.
#
# Reference values (unrestricted, untruncated real contrast):
#   pre_cal=0.5201, post_cal=0.7078 (RESULTS_LEDGER.md L009/L011).
#
# Usage: modal run run_step4_drift_control_g_real_labels_onset_excluded.py::main
# =============================================================================

import modal

app    = modal.App("bci-condition4-drift-control-g-real-onset-excluded")
volume = modal.Volume.from_name("eeg-data-vol")

RAW_DATA_PATH = "/data/processed_eeg_all_subjects.npz"
OUTPUT_JSON   = "/data/results_condition4_drift_control_g_real_onset_excluded.json"
VOLUME_PATH   = "/data"

SFREQ, N_CHANNELS = 250, 62

RANDOM_SEED         = 42
COV_SHRINKAGE       = 0.1
PCA_MAX_COMPONENTS  = 35
LOGREG_C            = 1.0
CAL_FRACTION        = 0.15
SHRINK_GRID         = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS     = 3

ONSET_DROP_N = 50   # identical truncation to F-DRIFT-F(a)

# DECISIONS.md pre-registered reference numbers + thresholds (fixed before
# this script runs).
REAL_LABEL_PRE_CAL_REFERENCE  = 0.5201   # unrestricted real contrast (L009/L011)
REAL_LABEL_POST_CAL_REFERENCE = 0.7078
ENTIRELY_ARTIFACT_THRESHOLD   = 0.55     # post_cal < this -> entirely onset artifact
GENUINE_SIGNAL_THRESHOLD      = 0.65     # post_cal >= this -> genuine signal, positive result

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy<2", "scikit-learn==1.4.2", "scipy")
)


@app.function(image=image, cpu=4.0, volumes={VOLUME_PATH: volume}, timeout=86400, memory=16384)
def run_drift_control_g():

    import logging, time, math, json
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
    from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("condition4-drift-control-g")

    np.random.seed(RANDOM_SEED)

    raw = np.load(RAW_DATA_PATH, allow_pickle=True)
    X_np = raw["X"].astype(np.float32)
    y_np = raw["y"].astype(np.int64)
    subjects_np = raw["subjects"]
    N, C, T = X_np.shape
    N_CLASSES = int(y_np.max()) + 1
    assert C == N_CHANNELS and N_CLASSES == 2
    log.info(f"X: {X_np.shape} | Subjects total: {len(np.unique(subjects_np))}")
    assert len(np.unique(subjects_np)) == 29, (
        f"Expected exactly 29 subjects (30 - sub-09 exclusion per AUDIT.md D2), "
        f"got {len(np.unique(subjects_np))}: {sorted(np.unique(subjects_np).tolist())}"
    )
    unique_subjects = sorted(np.unique(subjects_np).tolist())

    # =========================================================================
    # ONSET-EXCLUDED, REAL-LABEL dataset: drop the first ONSET_DROP_N trials
    # of EACH class-block for every subject -- identical truncation logic to
    # F-DRIFT-F(a), but keeping the REAL label (no pseudo-label construction).
    # =========================================================================
    idx_list, y_list, subj_list = [], [], []
    for sub in unique_subjects:
        sub_idx = np.where(subjects_np == sub)[0]
        for cls in (0, 1):
            cls_idx = sub_idx[y_np[sub_idx] == cls]
            n_full = len(cls_idx)
            assert n_full > ONSET_DROP_N + 10, (
                f"sub-{sub} class {cls} has only {n_full} trials, cannot drop "
                f"{ONSET_DROP_N} onset trials and still have a usable sequence"
            )
            truncated_idx = cls_idx[ONSET_DROP_N:]
            idx_list.append(truncated_idx)
            y_list.append(np.full(len(truncated_idx), cls, dtype=np.int64))
            subj_list.append(np.full(len(truncated_idx), sub))
    onset_excl_idx = np.concatenate(idx_list)
    y_onset_excl = np.concatenate(y_list)
    subjects_onset_excl = np.concatenate(subj_list)
    X_onset_excl = X_np[onset_excl_idx]
    log.info(f"Onset-excluded dataset: N={len(y_onset_excl)}, "
             f"class balance={np.bincount(y_onset_excl).tolist()}")

    # =========================================================================
    # RIEMANNIAN / EA / CLASSIFIER UTILITIES -- IDENTICAL to the rest of the
    # pre-F3 F-DRIFT family.
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
        _, W_ = matrix_sqrt_inv_sqrt(covs.mean(axis=0))
        return W_

    def apply_ea_whitening_signal(X, W_):
        return np.einsum("cd,ndt->nct", W_, X)

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

    # C3 hardening (2026-08-20, per F-DRIFT-E's INVALID-DESIGN failure).
    # Onset-truncated real-label class balance should stay near natural
    # (both classes lose the same 50-trial onset region), so this is NOT
    # declared imbalanced -- the gate should pass on its own.
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
                "script does not declare an imbalanced design."
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
    # SINGLE LOSO PASS -- REAL labels, onset-excluded, seed=42, 29 folds.
    # =========================================================================
    fold_records = []
    pre_cal_accs, post_cal_accs, pre_cal_bal_accs, post_cal_bal_accs = [], [], [], []

    for fold_idx, test_sub in enumerate(unique_subjects):
        fold_start = time.time()
        is_holdout = subjects_onset_excl == test_sub
        X_train, y_train = X_onset_excl[~is_holdout], y_onset_excl[~is_holdout]
        X_k, y_k = X_onset_excl[is_holdout], y_onset_excl[is_holdout]
        assert len(np.unique(y_train)) == 2 and len(np.unique(y_k)) == 2, (
            f"fold sub-{test_sub}: training pool or held-out subject missing a class"
        )

        mu = X_train.mean(axis=(0, 2), keepdims=True)
        sd = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
        X_train_z = ((X_train - mu) / sd).astype(np.float32)
        X_k_z = ((X_k - mu) / sd).astype(np.float32)

        W_ea = fit_ea_whitening(X_train_z)
        X_train_aligned = apply_ea_whitening_signal(X_train_z, W_ea).astype(np.float32)
        X_k_aligned = apply_ea_whitening_signal(X_k_z, W_ea).astype(np.float32)

        tan_train = tangent_vectorize(trial_covariances(X_train_aligned))
        tan_k = tangent_vectorize(trial_covariances(X_k_aligned))
        tangent_dim = tan_train.shape[1]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - CAL_FRACTION), random_state=RANDOM_SEED)
        cal_idx, test_idx = next(sss.split(tan_k, y_k))
        feat_cal, y_cal = tan_k[cal_idx], y_k[cal_idx]
        feat_test, y_test = tan_k[test_idx], y_k[test_idx]

        scaler = StandardScaler()
        feat_train_z = scaler.fit_transform(tan_train)
        feat_cal_z = scaler.transform(feat_cal)
        feat_test_z = scaler.transform(feat_test)

        n_components = min(PCA_MAX_COMPONENTS, feat_train_z.shape[1] - 1, feat_train_z.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
        X_train_pca = pca.fit_transform(feat_train_z)
        X_cal_pca = pca.transform(feat_cal_z)
        X_test_pca = pca.transform(feat_test_z)

        coef_final, icpt_final, best_shrink, global_clf = fit_shrinkage_classifier(
            X_train_pca, y_train, X_cal_pca, y_cal, RANDOM_SEED)
        pre_cal_preds = global_clf.predict(X_test_pca)
        pre_cal_acc = float((pre_cal_preds == y_test).mean())
        final_preds = linear_predict(coef_final, icpt_final, X_test_pca)
        post_cal_acc = float((final_preds == y_test).mean())
        metrics = compute_binary_metrics(y_test, final_preds)
        pre_cal_plaus = c3_balance_check(y_test, pre_cal_preds, pre_cal_acc,
                                          f"fold sub-{test_sub} pre_cal")
        post_cal_plaus = c3_balance_check(y_test, final_preds, post_cal_acc,
                                           f"fold sub-{test_sub} post_cal")

        log.info(f"  fold {fold_idx+1}/{len(unique_subjects)} sub-{test_sub} -> "
                 f"pre_cal={pre_cal_acc:.4f} (bal={pre_cal_plaus['balanced_accuracy']:.4f}) "
                 f"post_cal={post_cal_acc:.4f} (bal={post_cal_plaus['balanced_accuracy']:.4f}) "
                 f"shrink={best_shrink:.2f} [{time.time()-fold_start:.0f}s]")

        fold_records.append({
            "fold_index": fold_idx, "test_subject": str(test_sub),
            "tangent_dim": int(tangent_dim), "best_shrink_weight": float(best_shrink),
            "pre_calibration_acc": pre_cal_acc, "post_calibration_acc": post_cal_acc,
            "pre_cal_plausibility": pre_cal_plaus, "post_cal_plausibility": post_cal_plaus,
            **metrics,
        })
        pre_cal_accs.append(pre_cal_acc)
        post_cal_accs.append(post_cal_acc)
        pre_cal_bal_accs.append(pre_cal_plaus["balanced_accuracy"])
        post_cal_bal_accs.append(post_cal_plaus["balanced_accuracy"])

    pre_mean = float(np.mean(pre_cal_accs))
    post_mean = float(np.mean(post_cal_accs))
    pre_bal_mean = float(np.mean(pre_cal_bal_accs))
    post_bal_mean = float(np.mean(post_cal_bal_accs))

    # =========================================================================
    # DECISIONS.md's pre-registered F-DRIFT-G verdict, applied to post_cal
    # balanced accuracy.
    # =========================================================================
    if post_bal_mean < ENTIRELY_ARTIFACT_THRESHOLD:
        verdict = (f"ENTIRELY ONSET ARTIFACT -- post_cal_balanced ({post_bal_mean:.4f}) < "
                   f"{ENTIRELY_ARTIFACT_THRESHOLD}. The real contrast is entirely onset artifact; "
                   "the methodological null result is complete and final.")
    elif post_bal_mean >= GENUINE_SIGNAL_THRESHOLD:
        verdict = (f"GENUINE SIGNAL, POSITIVE RESULT -- post_cal_balanced ({post_bal_mean:.4f}) >= "
                   f"{GENUINE_SIGNAL_THRESHOLD}. Genuine task signal exists and was masked by the "
                   "onset artifact. HALT AND REPORT before any interpretation -- the paper changes "
                   "completely.")
    else:
        verdict = (f"PARTIAL -- post_cal_balanced ({post_bal_mean:.4f}) between "
                   f"{ENTIRELY_ARTIFACT_THRESHOLD} and {GENUINE_SIGNAL_THRESHOLD}. Report both "
                   "components; quantify the share attributable to onset.")

    log.info(f"\n{'='*70}\n  F-DRIFT-G SUMMARY\n{'='*70}")
    log.info(f"  pre_cal={pre_mean:.4f} (balanced={pre_bal_mean:.4f}) "
             f"post_cal={post_mean:.4f} (balanced={post_bal_mean:.4f})")
    log.info(f"  Reference (unrestricted real contrast): pre_cal={REAL_LABEL_PRE_CAL_REFERENCE} "
             f"post_cal={REAL_LABEL_POST_CAL_REFERENCE}")
    log.info(f"  DECISIONS.md VERDICT: {verdict}")

    results_payload = {
        "condition": "F-DRIFT-G — real Search-vs-Memorize labels under onset exclusion (first 50 "
                      "trials of each block dropped per subject), identical calibrated pipeline, "
                      "single seed=42, full 29-fold LOSO",
        "hyperparameters": {
            "pca_max_components": PCA_MAX_COMPONENTS, "logreg_C": LOGREG_C,
            "random_seed": RANDOM_SEED, "cov_shrinkage": COV_SHRINKAGE, "cal_fraction": CAL_FRACTION,
            "onset_drop_n": ONSET_DROP_N,
        },
        "fold_results": fold_records,
        "pre_calibration_accuracy_mean": pre_mean,
        "post_calibration_accuracy_mean": post_mean,
        "pre_calibration_balanced_accuracy_mean": pre_bal_mean,
        "post_calibration_balanced_accuracy_mean": post_bal_mean,
        "decisions_md_verdict": verdict,
        "reference_numbers": {
            "real_label_pre_cal_unrestricted": REAL_LABEL_PRE_CAL_REFERENCE,
            "real_label_post_cal_unrestricted": REAL_LABEL_POST_CAL_REFERENCE,
            "entirely_artifact_threshold": ENTIRELY_ARTIFACT_THRESHOLD,
            "genuine_signal_threshold": GENUINE_SIGNAL_THRESHOLD,
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
    assert 0.0 <= pre_mean <= 1.0 and 0.0 <= post_mean <= 1.0
    assert len(fold_records) == 29, f"[C3 PLAUSIBILITY FAIL] expected 29 folds, got {len(fold_records)}"
    log.info(f"  [C3] plausibility: 29 folds, all accuracies in [0,1] -- OK")

    return {
        "pre_cal": pre_mean, "pre_cal_balanced": pre_bal_mean,
        "post_cal": post_mean, "post_cal_balanced": post_bal_mean,
        "decisions_md_verdict": verdict,
        "output_path": OUTPUT_JSON,
    }


@app.local_entrypoint()
def main():
    print("F-DRIFT-G — real labels under onset exclusion (the missing cell)")
    print(f"Drop first {ONSET_DROP_N} trials of each block per subject, REAL Search-vs-Memorize "
          "labels, identical calibrated pipeline, single seed=42, full 29-fold LOSO.")
    print(f"Reference (unrestricted): pre_cal={REAL_LABEL_PRE_CAL_REFERENCE} "
          f"post_cal={REAL_LABEL_POST_CAL_REFERENCE}")
    print(f"Pre-registered rule (DECISIONS.md, balanced post_cal): "
          f"<{ENTIRELY_ARTIFACT_THRESHOLD} -> ENTIRELY ARTIFACT (null result final) | "
          f">={GENUINE_SIGNAL_THRESHOLD} -> GENUINE SIGNAL (positive result, halt and report) | "
          f"in between -> partial.\n")
    results = run_drift_control_g.remote()
    print("\nF-DRIFT-G RESULTS:")
    for k, v in results.items():
        print(f"  {k}: {v}")
