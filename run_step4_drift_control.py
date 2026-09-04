# =============================================================================
# command to run:
#   modal run run_step4_drift_control.py::main
# run_step4_drift_control.py
#
# F-DRIFT — WITHIN-SESSION TIME-DRIFT CONTROL (new fix-ID, added 2026-08-19,
# AUDIT.md fix register; DECISIONS.md pre-registered rule; GATES Batch 2)
#
# WHY THIS SCRIPT EXISTS:
#   Batch 1's Tier-1 finding (STATUS.md, 2026-08-19) established that every
#   reported Condition-4 accuracy number is POST-CALIBRATION: 15% of the
#   held-out subject's OWN labeled trials personalize the classifier before
#   the remaining 85% is scored (pre-calibration accuracy is at or near
#   chance, RESULTS_LEDGER.md's L009). F-PARITY (ledger L003) independently
#   found position-third decodability of 0.4753 against a 0.3333 chance
#   floor -- real, exploitable within-session temporal-position signal
#   exists in this feature space. Because class label is PERFECTLY
#   CONFOUNDED with block (every Search trial precedes every Memorize trial,
#   per D2/F-PARITY), the calibrated pipeline COULD be decoding time-in-
#   session rather than task content, and the calibration step -- which
#   personalizes on a 15% sample of the held-out subject's OWN trials --
#   is exactly the mechanism that could let a time-drift signal leak into a
#   number that looks like a task-decoding accuracy.
#
# TEST (pre-registered in DECISIONS.md BEFORE this script runs):
#   For each REAL class block (Search-only, Memorize-only) separately,
#   split that subject's own chronologically-ordered trials for that block
#   at its midpoint into a PSEUDO-label: pseudo-class 0 = first (earlier)
#   half, pseudo-class 1 = second (later) half. Both pseudo-classes are
#   drawn from the SAME real task -- task content is held constant, so any
#   decodability of the pseudo-label can only come from within-session
#   time-position signal (drift), not from task/class content. Then run
#   the IDENTICAL calibrated LOSO pipeline (byte-identical EA/tangent/PCA/
#   shrinkage-calibration code to run_step4_matched_spatial_control.py's
#   pre-F3 family -- same family already used by F-OCULAR(a)/(c)/F-PARITY,
#   for internal consistency within Batch 1) on this pseudo-label, single
#   seed=42, full 29-fold LOSO with the SAME 15%-calibration mechanism.
#
#   Run twice (Search-only pseudo-test, Memorize-only pseudo-test) rather
#   than pooling across both real classes, so "both pseudo-classes drawn
#   from the SAME task" holds literally and unambiguously for each test.
#
# ALSO (per explicit instruction): instrument the REAL calibration split
#   (the SAME StratifiedShuffleSplit(test_size=0.85, random_state=seed)
#   mechanism used throughout this codebase's calibrated drivers) to report
#   how the 15% calibration sample distributes across blocks (trivially:
#   proportional to each class's own count, since the split stratifies by
#   class) and across WITHIN-BLOCK TIME (early/mid/late thirds, using
#   F-PARITY's own compute_within_block_thirds convention) -- this
#   quantifies how much drift signal the calibration step has access to,
#   directly informing results/METHODS_FACTS.md §8's flagged open question.
#
# SCOPE CAVEAT (DECISIONS.md, pre-registered 2026-08-19, BEFORE this run
#   reports): the pseudo-test splits early/late WITHIN one real class
#   block -- both pseudo-classes sit inside the same block, no gap. The
#   REAL Search-vs-Memorize contrast spans the block boundary itself PLUS
#   a ~400s break -- strictly more temporal separation and a physical
#   discontinuity. A near-chance pseudo-test result is therefore a LOWER
#   BOUND on the drift available to the real contrast -- it only rules out
#   FINE-GRAINED within-block drift, not coarser between-block drift. This
#   is stated explicitly in every interpretation string this script emits.
#
# ALSO (cheap addition, same run, DECISIONS.md pre-registered 2026-08-19):
#   one additional single-seed=42, full-29-fold LOSO pass on the REAL
#   Search-vs-Memorize labels (identical calibrated pipeline). For every
#   scored (85%) test trial, compute its temporal distance (in same-class
#   trial-count units) to the nearest calibration trial OF THE SAME REAL
#   CLASS, bin into quartiles of that distance, and report accuracy per
#   quartile. Decaying accuracy with distance is evidence the classifier
#   is riding drift (exploiting proximity to its own calibration sample);
#   flat accuracy is evidence against that mechanism. No extra folds
#   beyond this one real-label pass -- purely a grouping of its own
#   already-produced predictions.
#
# Usage: modal run run_step4_drift_control.py::main
# =============================================================================

import modal

app    = modal.App("bci-condition4-drift-control")
volume = modal.Volume.from_name("eeg-data-vol")

RAW_DATA_PATH = "/data/processed_eeg_all_subjects.npz"
OUTPUT_JSON   = "/data/results_condition4_drift_control.json"
VOLUME_PATH   = "/data"

SFREQ, N_CHANNELS = 250, 62

RANDOM_SEED         = 42
COV_SHRINKAGE       = 0.1
PCA_MAX_COMPONENTS  = 35
LOGREG_C            = 1.0
CAL_FRACTION        = 0.15
SHRINK_GRID         = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS     = 3

# DECISIONS.md pre-registered F-DRIFT rule (recorded 2026-08-19, BEFORE this
# script runs): applied to POST-calibration pseudo-label accuracy.
DRIFT_NOT_DRIVING_THRESHOLD = 0.55   # < this -> drift not driving the result
DRIFT_DETECTOR_THRESHOLD    = 0.65   # > this -> pipeline is a drift detector

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy<2", "scikit-learn==1.4.2", "scipy")
)


@app.function(image=image, cpu=4.0, volumes={VOLUME_PATH: volume}, timeout=86400, memory=16384)
def run_drift_control():

    import logging, time, math, json
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
    from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("condition4-drift-control")

    np.random.seed(RANDOM_SEED)

    raw = np.load(RAW_DATA_PATH, allow_pickle=True)
    X_np = raw["X"].astype(np.float32)
    y_np = raw["y"].astype(np.int64)
    subjects_np = raw["subjects"]
    N, C, T = X_np.shape
    N_CLASSES = int(y_np.max()) + 1
    assert C == N_CHANNELS and N_CLASSES == 2
    log.info(f"X: {X_np.shape} | Subjects total: {len(np.unique(subjects_np))}")
    # F-SILENT hardening.
    assert len(np.unique(subjects_np)) == 29, (
        f"Expected exactly 29 subjects (30 - sub-09 exclusion per AUDIT.md D2), "
        f"got {len(np.unique(subjects_np))}: {sorted(np.unique(subjects_np).tolist())}"
    )
    unique_subjects = sorted(np.unique(subjects_np).tolist())

    # =========================================================================
    # RIEMANNIAN / EA UTILITIES -- IDENTICAL to run_step4_matched_spatial_
    # control.py / run_step4_parity_split_control.py's pre-F3 pooled-only EA,
    # deliberately unchanged for family consistency with the rest of Batch 1.
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
    # DIAGNOSTIC: real 15% calibration split's distribution across blocks and
    # within-block time (early/mid/late thirds). Uses the REAL labels/split
    # mechanism, seed=42, computed once per subject (no LOSO needed -- this
    # characterizes the split itself, not a trained classifier).
    # =========================================================================
    def within_block_thirds_for_subject(y_sub):
        thirds = np.full(len(y_sub), -1, dtype=np.int64)
        for cls in (0, 1):
            cls_idx = np.where(y_sub == cls)[0]
            n = len(cls_idx)
            if n == 0:
                continue
            thirds[cls_idx] = np.minimum((np.arange(n) * 3) // n, 2)
        return thirds

    cal_split_third_counts = {"early": 0, "mid": 0, "late": 0}
    cal_split_class_counts = {0: 0, 1: 0}
    cal_split_total = 0
    for test_sub in unique_subjects:
        sub_mask = subjects_np == test_sub
        y_sub = y_np[sub_mask]
        thirds_sub = within_block_thirds_for_subject(y_sub)
        sss_diag = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - CAL_FRACTION), random_state=RANDOM_SEED)
        cal_idx_sub, _ = next(sss_diag.split(np.zeros((len(y_sub), 1)), y_sub))
        for i in cal_idx_sub:
            cal_split_third_counts[["early", "mid", "late"][thirds_sub[i]]] += 1
            cal_split_class_counts[int(y_sub[i])] += 1
            cal_split_total += 1
    calibration_distribution_diagnostic = {
        "description": "Where the REAL 15% calibration sample falls, aggregated across all 29 subjects "
                        "(seed=42, the same StratifiedShuffleSplit mechanism used throughout this codebase).",
        "n_calibration_trials_total": cal_split_total,
        "by_class": {str(k): v for k, v in cal_split_class_counts.items()},
        "by_within_block_third": cal_split_third_counts,
        "by_within_block_third_fraction": {
            k: (v / cal_split_total if cal_split_total else float("nan")) for k, v in cal_split_third_counts.items()
        },
    }
    log.info(f"Calibration-split time distribution: {calibration_distribution_diagnostic}")

    # =========================================================================
    # ONE PSEUDO-LABEL DRIFT TEST, restricted to ONE real class's trials so
    # both pseudo-classes are drawn from the SAME task -- full 29-fold LOSO
    # with the IDENTICAL calibrated pipeline (single seed=42).
    # =========================================================================
    def run_drift_test(real_class, real_class_name):
        log.info(f"\n{'='*70}\n  F-DRIFT TEST: pseudo-label within real class {real_class} "
                 f"({real_class_name})\n{'='*70}")

        # Build the pseudo-dataset: only this class's trials, per subject,
        # early-half -> pseudo 0, late-half -> pseudo 1 (chronological order
        # preserved by construction -- epochs.get_data() never reorders trials).
        idx_list, y_pseudo_list, subj_list = [], [], []
        for sub in unique_subjects:
            sub_idx = np.where(subjects_np == sub)[0]
            cls_idx = sub_idx[y_np[sub_idx] == real_class]
            n = len(cls_idx)
            assert n >= 2, f"sub-{sub} has only {n} trials of class {real_class} -- cannot midpoint-split."
            midpoint = n // 2
            pseudo = np.zeros(n, dtype=np.int64)
            pseudo[midpoint:] = 1
            idx_list.append(cls_idx)
            y_pseudo_list.append(pseudo)
            subj_list.append(np.full(n, sub))
        pseudo_idx = np.concatenate(idx_list)
        y_pseudo = np.concatenate(y_pseudo_list)
        subjects_pseudo = np.concatenate(subj_list)
        X_pseudo = X_np[pseudo_idx]
        log.info(f"  Pseudo-dataset: {X_pseudo.shape[0]} trials (class-{real_class}-only), "
                 f"pseudo-class balance: {np.bincount(y_pseudo).tolist()}")

        fold_records = []
        pre_cal_accs, post_cal_accs = [], []

        for fold_idx, test_sub in enumerate(unique_subjects):
            fold_start = time.time()
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
                                              f"{real_class_name} fold sub-{test_sub} pre_cal")
            post_cal_plaus = c3_balance_check(y_test, final_preds, post_cal_acc,
                                               f"{real_class_name} fold sub-{test_sub} post_cal")

            log.info(f"  [{real_class_name}] fold {fold_idx+1}/{len(unique_subjects)} sub-{test_sub} -> "
                     f"pre_cal={pre_cal_acc:.4f} post_cal={post_cal_acc:.4f} (shrink={best_shrink:.2f}) "
                     f"[{time.time()-fold_start:.0f}s]")

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
        log.info(f"  [{real_class_name}] DONE -- pre_cal={pre_mean:.4f}+/-{pre_std:.4f} "
                 f"post_cal={post_mean:.4f}+/-{post_std:.4f}")

        return {
            "real_class": real_class, "real_class_name": real_class_name,
            "n_pseudo_trials": int(X_pseudo.shape[0]),
            "pseudo_class_balance": np.bincount(y_pseudo).tolist(),
            "fold_results": fold_records,
            "pre_calibration_accuracy_mean": pre_mean, "pre_calibration_accuracy_std": pre_std,
            "post_calibration_accuracy_mean": post_mean, "post_calibration_accuracy_std": post_std,
        }

    result_search = run_drift_test(0, "search_only")
    result_memorize = run_drift_test(1, "memorize_only")

    # =========================================================================
    # REAL-LABEL DISTANCE-TO-CALIBRATION-TRIAL ANALYSIS (DECISIONS.md
    # pre-registered, added 2026-08-19). ONE additional single-seed=42,
    # full-29-fold LOSO pass on the REAL Search-vs-Memorize labels, same
    # calibrated pipeline. For every scored test trial, record its distance
    # (in same-class trial-count units) to the nearest calibration trial of
    # the SAME real class, then bin into quartiles and report accuracy per
    # quartile -- a grouping of this pass's own predictions, no extra folds.
    # =========================================================================
    def within_class_position(y_sub):
        """0-indexed chronological position of each trial WITHIN its own
        real class's block (class 0's own sequence, class 1's own sequence)."""
        pos = np.full(len(y_sub), -1, dtype=np.int64)
        for cls in (0, 1):
            cls_idx = np.where(y_sub == cls)[0]
            pos[cls_idx] = np.arange(len(cls_idx))
        return pos

    log.info(f"\n{'='*70}\n  REAL-LABEL DISTANCE-TO-CALIBRATION-TRIAL ANALYSIS "
             f"(seed={RANDOM_SEED}, full 29-fold LOSO, real Search-vs-Memorize labels)\n{'='*70}")

    real_fold_records = []
    real_pre_cal_accs, real_post_cal_accs = [], []
    distance_records = []   # list of {"distance": int, "correct": bool}

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
        real_pre_cal_acc = float((global_clf.predict(X_test_pca) == y_test).mean())
        real_final_preds = linear_predict(coef_final, icpt_final, X_test_pca)
        real_post_cal_acc = float((real_final_preds == y_test).mean())
        real_metrics = compute_binary_metrics(y_test, real_final_preds)
        real_pre_cal_plaus = c3_balance_check(y_test, global_clf.predict(X_test_pca), real_pre_cal_acc,
                                               f"real-label fold sub-{test_sub} pre_cal")
        real_post_cal_plaus = c3_balance_check(y_test, real_final_preds, real_post_cal_acc,
                                                f"real-label fold sub-{test_sub} post_cal")

        # --- distance-to-nearest-same-class-calibration-trial, per scored trial ---
        pos_k = within_class_position(y_k)
        for local_i, pred, true_label in zip(test_idx, real_final_preds, y_test):
            cls = int(y_k[local_i])
            same_class_cal_local_idx = cal_idx[y_k[cal_idx] == cls]
            if len(same_class_cal_local_idx) == 0:
                continue   # stratified split guarantees this shouldn't happen, but skip defensively
            dist = int(np.min(np.abs(pos_k[local_i] - pos_k[same_class_cal_local_idx])))
            distance_records.append({"distance": dist, "correct": bool(pred == true_label)})

        log.info(f"  [real-label] fold {fold_idx+1}/{len(unique_subjects)} sub-{test_sub} -> "
                 f"pre_cal={real_pre_cal_acc:.4f} post_cal={real_post_cal_acc:.4f} (shrink={best_shrink:.2f}) "
                 f"[{time.time()-fold_start:.0f}s]")

        real_fold_records.append({
            "fold_index": fold_idx, "test_subject": str(test_sub),
            "tangent_dim": int(tangent_dim), "best_shrink_weight": float(best_shrink),
            "pre_calibration_acc": real_pre_cal_acc, "post_calibration_acc": real_post_cal_acc,
            "pre_cal_plausibility": real_pre_cal_plaus, "post_cal_plausibility": real_post_cal_plaus,
            **real_metrics,
        })
        real_pre_cal_accs.append(real_pre_cal_acc)
        real_post_cal_accs.append(real_post_cal_acc)

    real_pre_mean, real_pre_std = float(np.mean(real_pre_cal_accs)), float(np.std(real_pre_cal_accs))
    real_post_mean, real_post_std = float(np.mean(real_post_cal_accs)), float(np.std(real_post_cal_accs))
    log.info(f"  [real-label] DONE -- pre_cal={real_pre_mean:.4f}+/-{real_pre_std:.4f} "
             f"post_cal={real_post_mean:.4f}+/-{real_post_std:.4f}")

    # --- bin pooled scored trials into quartiles of distance, report accuracy per quartile.
    # RANK-based split (equal COUNT per group via sorted index, np.array_split), not a
    # percentile-VALUE threshold + np.digitize: distance is a small-range integer with heavy
    # ties (many test trials share the same nearest-calibration-trial distance), so a
    # value-based percentile edge can coincide with the bulk of the data and leave a quartile
    # empty. Rank-based splitting guarantees four non-degenerate, near-equal-size groups
    # regardless of tie structure (verified against a tied-heavy synthetic case beforehand).
    all_distances = np.array([r["distance"] for r in distance_records])
    all_correct = np.array([r["correct"] for r in distance_records])
    sort_order = np.argsort(all_distances, kind="stable")
    quartile_groups = np.array_split(sort_order, 4)   # 4 groups of near-equal size, nearest->farthest
    accuracy_by_distance_quartile = {}
    for q, group_idx in enumerate(quartile_groups):
        n_q = int(len(group_idx))
        accuracy_by_distance_quartile[f"q{q+1}"] = {
            "n_trials": n_q,
            "distance_range": [
                float(all_distances[group_idx].min()) if n_q else None,
                float(all_distances[group_idx].max()) if n_q else None,
            ],
            "accuracy": float(all_correct[group_idx].mean()) if n_q else None,
        }
    q_accs = [accuracy_by_distance_quartile[f"q{q+1}"]["accuracy"] for q in range(4)]
    is_monotonically_decaying = all(
        q_accs[i] is not None and q_accs[i + 1] is not None and q_accs[i] >= q_accs[i + 1]
        for i in range(3)
    )
    log.info(f"  Accuracy by distance-to-nearest-calibration-trial quartile: {accuracy_by_distance_quartile}")
    log.info(f"  Monotonically decaying (q1->q4, nearest->farthest)? {is_monotonically_decaying}")

    real_label_distance_analysis = {
        "description": "Real Search-vs-Memorize labels, single seed=42, full 29-fold LOSO, identical "
                        "calibrated pipeline. Distance = trial-count to nearest calibration trial of the "
                        "SAME real class, along that class's own chronological sequence.",
        "pre_calibration_accuracy_mean": real_pre_mean, "pre_calibration_accuracy_std": real_pre_std,
        "post_calibration_accuracy_mean": real_post_mean, "post_calibration_accuracy_std": real_post_std,
        "fold_results": real_fold_records,
        "n_scored_trials_pooled": len(distance_records),
        "accuracy_by_distance_quartile": accuracy_by_distance_quartile,
        "monotonically_decaying_q1_to_q4": is_monotonically_decaying,
        "interpretation": (
            "Decaying accuracy from nearest (q1) to farthest (q4) quartile is evidence the classifier is "
            "riding drift (exploiting temporal proximity to its own calibration sample). Flat accuracy is "
            "evidence against that mechanism. Reported alongside, not instead of, the pseudo-label tests "
            "below -- see DECISIONS.md for why the pseudo-label result is a lower bound, not a full "
            "clearance, of the drift hypothesis."
        ),
    }

    mean_post_cal_drift_acc = float(np.mean([
        result_search["post_calibration_accuracy_mean"], result_memorize["post_calibration_accuracy_mean"],
    ]))
    mean_pre_cal_drift_acc = float(np.mean([
        result_search["pre_calibration_accuracy_mean"], result_memorize["pre_calibration_accuracy_mean"],
    ]))

    # =========================================================================
    # DECISIONS.md's pre-registered F-DRIFT verdict, applied mechanically to
    # POST-calibration pseudo-accuracy (the same stage the real headline
    # numbers are reported at).
    # =========================================================================
    LOWER_BOUND_CAVEAT = (
        "SCOPE CAVEAT (DECISIONS.md, pre-registered before this run reported): the pseudo-test splits "
        "early/late WITHIN one real class block -- both pseudo-classes sit inside the same block, no "
        "gap. The REAL Search-vs-Memorize contrast spans the block boundary itself PLUS a ~400s break "
        "-- strictly more separation and a physical discontinuity a within-block split cannot "
        "reproduce. This pseudo-test result is therefore a LOWER BOUND on the drift available to the "
        "real contrast, not a full accounting."
    )
    if mean_post_cal_drift_acc < DRIFT_NOT_DRIVING_THRESHOLD:
        drift_verdict = ("fine-grained within-block drift is NOT driving the pseudo-contrast "
                          f"(pseudo-accuracy {mean_post_cal_drift_acc:.4f} < {DRIFT_NOT_DRIVING_THRESHOLD}). "
                          f"{LOWER_BOUND_CAVEAT} This does NOT by itself clear coarser BETWEEN-block drift "
                          "-- see the real-label distance-to-calibration-trial analysis below for that.")
    elif mean_post_cal_drift_acc > DRIFT_DETECTOR_THRESHOLD:
        drift_verdict = ("the pipeline is a within-session drift detector even at the fine-grained "
                          f"within-block scale, and the primary contrast does NOT survive "
                          f"(pseudo-accuracy {mean_post_cal_drift_acc:.4f} > {DRIFT_DETECTOR_THRESHOLD})")
    else:
        drift_verdict = ("partial contamination -- report both, quantify the share "
                          f"({DRIFT_NOT_DRIVING_THRESHOLD} <= pseudo-accuracy {mean_post_cal_drift_acc:.4f} <= {DRIFT_DETECTOR_THRESHOLD}). "
                          f"{LOWER_BOUND_CAVEAT}")

    log.info(f"\n{'='*70}\n  F-DRIFT SUMMARY\n{'='*70}")
    log.info(f"  search_only : pre_cal={result_search['pre_calibration_accuracy_mean']:.4f} "
             f"post_cal={result_search['post_calibration_accuracy_mean']:.4f}")
    log.info(f"  memorize_only: pre_cal={result_memorize['pre_calibration_accuracy_mean']:.4f} "
             f"post_cal={result_memorize['post_calibration_accuracy_mean']:.4f}")
    log.info(f"  MEAN pre_cal={mean_pre_cal_drift_acc:.4f}  MEAN post_cal={mean_post_cal_drift_acc:.4f}")
    log.info(f"  DECISIONS.md VERDICT: {drift_verdict}")

    results_payload = {
        "condition": "F-DRIFT — within-session time-drift control: pseudo-label (early-half vs. "
                      "late-half of the SAME real class block) run through the identical calibrated "
                      "pipeline, single seed=42, full 29-fold LOSO",
        "hyperparameters": {
            "pca_max_components": PCA_MAX_COMPONENTS, "logreg_C": LOGREG_C,
            "random_seed": RANDOM_SEED, "cov_shrinkage": COV_SHRINKAGE, "cal_fraction": CAL_FRACTION,
        },
        "calibration_distribution_diagnostic": calibration_distribution_diagnostic,
        "drift_tests": {"search_only": result_search, "memorize_only": result_memorize},
        "real_label_distance_analysis": real_label_distance_analysis,
        "mean_pre_calibration_accuracy": mean_pre_cal_drift_acc,
        "mean_post_calibration_accuracy": mean_post_cal_drift_acc,
        "decisions_md_verdict": drift_verdict,
        "decisions_md_thresholds": {
            "drift_not_driving_below": DRIFT_NOT_DRIVING_THRESHOLD,
            "drift_detector_above": DRIFT_DETECTOR_THRESHOLD,
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
    for res in (result_search, result_memorize):
        assert 0.0 <= res["pre_calibration_accuracy_mean"] <= 1.0, (
            f"[C3 PLAUSIBILITY FAIL] {res['real_class_name']} pre_cal mean outside [0,1]"
        )
        assert 0.0 <= res["post_calibration_accuracy_mean"] <= 1.0, (
            f"[C3 PLAUSIBILITY FAIL] {res['real_class_name']} post_cal mean outside [0,1]"
        )
        assert len(res["fold_results"]) == 29, (
            f"[C3 PLAUSIBILITY FAIL] {res['real_class_name']} expected 29 folds, got {len(res['fold_results'])}"
        )
        assert res["pseudo_class_balance"][0] > 0 and res["pseudo_class_balance"][1] > 0, (
            f"[C3 PLAUSIBILITY FAIL] {res['real_class_name']} pseudo-class balance has an empty class: "
            f"{res['pseudo_class_balance']}"
        )
    assert cal_split_total > 0, "[C3 PLAUSIBILITY FAIL] calibration-distribution diagnostic saw zero trials"
    assert abs(sum(cal_split_third_counts.values()) - cal_split_total) == 0, (
        "[C3 PLAUSIBILITY FAIL] calibration-distribution third counts do not sum to the total"
    )
    assert 0.0 <= real_pre_mean <= 1.0, f"[C3 PLAUSIBILITY FAIL] real-label pre_cal mean {real_pre_mean} outside [0,1]"
    assert 0.0 <= real_post_mean <= 1.0, f"[C3 PLAUSIBILITY FAIL] real-label post_cal mean {real_post_mean} outside [0,1]"
    assert len(real_fold_records) == 29, (
        f"[C3 PLAUSIBILITY FAIL] real-label pass expected 29 folds, got {len(real_fold_records)}"
    )
    assert len(distance_records) > 0, "[C3 PLAUSIBILITY FAIL] distance-to-calibration-trial analysis saw zero scored trials"
    n_quartiles_with_trials = sum(1 for q in range(4) if accuracy_by_distance_quartile[f"q{q+1}"]["n_trials"] > 0)
    assert n_quartiles_with_trials >= 3, (
        f"[C3 PLAUSIBILITY FAIL] only {n_quartiles_with_trials}/4 distance quartiles have any trials -- "
        f"quartile binning degenerate, likely too few distinct distance values"
    )
    for q in range(4):
        acc = accuracy_by_distance_quartile[f"q{q+1}"]["accuracy"]
        if acc is not None:
            assert 0.0 <= acc <= 1.0, f"[C3 PLAUSIBILITY FAIL] distance quartile q{q+1} accuracy {acc} outside [0,1]"
    log.info(f"  [C3] plausibility: 2 pseudo drift tests + 1 real-label pass, all 29 folds each, "
             f"accuracies in [0,1], pseudo-classes both non-empty, calibration-distribution counts "
             f"consistent, {n_quartiles_with_trials}/4 distance quartiles populated -- OK")

    return {
        "mean_pre_calibration_accuracy": mean_pre_cal_drift_acc,
        "mean_post_calibration_accuracy": mean_post_cal_drift_acc,
        "search_only_post_cal": result_search["post_calibration_accuracy_mean"],
        "memorize_only_post_cal": result_memorize["post_calibration_accuracy_mean"],
        "decisions_md_verdict": drift_verdict,
        "real_label_post_cal": real_post_mean,
        "accuracy_by_distance_quartile": accuracy_by_distance_quartile,
        "monotonically_decaying_q1_to_q4": is_monotonically_decaying,
        "output_path": OUTPUT_JSON,
    }


@app.local_entrypoint()
def main():
    print("F-DRIFT — Within-session time-drift control")
    print("Pseudo-label = early-half vs. late-half of the SAME real class block (task held constant).")
    print("Run through the identical calibrated LOSO pipeline, single seed=42, full 29 folds, x2 (Search-only, Memorize-only).")
    print("PLUS one real-label pass (same pipeline) for the accuracy-vs-distance-to-calibration-trial analysis.")
    print(f"Pre-registered verdict thresholds (DECISIONS.md): <{DRIFT_NOT_DRIVING_THRESHOLD} not driving | "
          f">{DRIFT_DETECTOR_THRESHOLD} drift detector | in between = partial contamination.")
    print("SCOPE CAVEAT: the pseudo-test result is a LOWER BOUND (within-block only) -- it does not by "
          "itself clear between-block drift; see the real-label distance-quartile analysis for that.\n")
    results = run_drift_control.remote()
    print("\nF-DRIFT RESULTS:")
    for k, v in results.items():
        print(f"  {k:<32}: {v}")
