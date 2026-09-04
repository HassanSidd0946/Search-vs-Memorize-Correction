# =============================================================================
# command to run:
#   modal run run_step4_drift_control_e_boundary_shift.py::main
# run_step4_drift_control_e_boundary_shift.py
#
# F-DRIFT-E — BOUNDARY-PRIVILEGE CHECK, FIXED SYMMETRIC WINDOW (REDESIGNED
# 2026-08-20, AUDIT.md fix register; DECISIONS.md "F-DRIFT-E REDESIGN"
# pre-registered rule)
#
# WHY THIS REDESIGN EXISTS:
#   The original version of this script (letting pseudo-class size vary
#   freely with shift fraction and block length) was diagnosed
#   INVALID-DESIGN by direct review of its results (RESULTS_LEDGER.md
#   L023): severe, uncontrolled class imbalance (as extreme as 1451/10159
#   at the 25%/75% shifts) meant every reported accuracy tracked the
#   majority-class base rate, not any genuine boundary-privilege signal.
#   That run is logged, not deleted, but NONE of its verdicts may be
#   cited. This rewrite fixes the design at the root: N, balance, and
#   temporal span are now IDENTICAL across every position, so split
#   LOCATION is the only variable.
#
# TEST DESIGN (pre-registered in DECISIONS.md BEFORE this script runs):
#   For each split position, take W trials immediately BEFORE and W
#   trials immediately AFTER the split point. W is a SINGLE constant
#   across ALL positions and ALL subjects, computed directly from the
#   data as the largest value feasible at the most extreme shift (the
#   minimum before/after headroom across every position and every
#   subject) -- not hardcoded, so it self-adjusts correctly to whatever
#   the real per-subject block lengths turn out to be.
#
#   7 positions: 25%/50%/75% through block 1, 25%/50%/75% through
#   block 2, AND the TRUE BOUNDARY itself under the SAME windowing (W
#   before, W after -- NOT the full-data 0.7078 reference, which uses
#   ~4x more trials at full 29-fold scale and is not comparable to a
#   matched-N measurement). Each position run through the SAME
#   calibrated LOSO pipeline, single seed=42, full 29-fold LOSO.
#
#   Reported per position: pre_cal, post_cal, balanced accuracy, N,
#   class balance, majority rate, and the temporal span the window
#   covers in seconds (constant across positions since 2W trials is
#   constant -- computed via a mean-ITI reference REUSED from
#   F-DRIFT-C's independently-extracted real onset times, ledger L016,
#   ~4.65s/trial -- NOT re-extracted here, to avoid a second expensive
#   29-subject raw-file download pass; disclosed as an approximation
#   assuming uniform ITI, not an independent measurement).
#
# FIX 1 (C3 HARDENING, applied throughout): every classification result
#   asserts and prints class balance, majority-class rate, accuracy-
#   minus-majority-rate, and balanced accuracy, failing loudly if
#   balance falls outside 45/55 -- by construction every window here is
#   exactly W/W = 50/50, so this check should never fire; if it does,
#   that itself signals a bug in the windowing logic, not a design
#   property to route around.
#
# PRE-REGISTERED VERDICT RULE (DECISIONS.md, fixed BEFORE this script
# runs, BALANCED-ACCURACY terms, NOT strict ordering):
#   all shifted positions within 0.05 of the true-boundary-at-W value
#       -> the true boundary is NOT privileged -- any temporal partition
#          at matched N/balance/span yields the same calibrated accuracy.
#   true-boundary-at-W exceeds every shifted position by > 0.05
#       -> the true boundary IS privileged -- report and stop for
#          discussion; would materially change the conclusion.
#   mixed -> report the full 7-point profile, no further interpretation.
#
# Usage: modal run run_step4_drift_control_e_boundary_shift.py::main
# =============================================================================

import modal

app    = modal.App("bci-condition4-drift-control-e-boundary-shift")
volume = modal.Volume.from_name("eeg-data-vol")

RAW_DATA_PATH = "/data/processed_eeg_all_subjects.npz"
OUTPUT_JSON   = "/data/results_condition4_drift_control_e_boundary_shift.json"
VOLUME_PATH   = "/data"

SFREQ, N_CHANNELS = 250, 62

RANDOM_SEED         = 42
COV_SHRINKAGE       = 0.1
PCA_MAX_COMPONENTS  = 35
LOGREG_C            = 1.0
CAL_FRACTION        = 0.15
SHRINK_GRID         = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS     = 3

# (label, block_num, frac). block_num=0 is the special-cased true boundary
# (split = n1 exactly); block_num in {1,2} uses frac * that block's length.
POSITIONS = [
    ("block1_25pct", 1, 0.25), ("block1_50pct", 1, 0.50), ("block1_75pct", 1, 0.75),
    ("block2_25pct", 2, 0.25), ("block2_50pct", 2, 0.50), ("block2_75pct", 2, 0.75),
    ("true_boundary", 0, None),
]

# Reused from F-DRIFT-C's independently-extracted real onset times
# (RESULTS_LEDGER.md L016: 465s at k=100 trials => ~4.65s/trial). NOT
# re-extracted here -- disclosed approximation, not a new measurement.
MEAN_ITI_SECONDS_REFERENCE = 4.65

NOT_PRIVILEGED_TOLERANCE = 0.05   # "within 0.05" -> not privileged
PRIVILEGED_MARGIN        = 0.05   # true boundary exceeds EVERY shifted position by this much -> privileged

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy<2", "scikit-learn==1.4.2", "scipy")
)


@app.function(image=image, cpu=4.0, volumes={VOLUME_PATH: volume}, timeout=86400, memory=16384)
def run_drift_control_e():

    import logging, time, math, json
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
    from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("condition4-drift-control-e")

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
    # Per-subject block boundary detection -- IDENTICAL to the original
    # F-DRIFT-E's contiguity-verified approach (not implicated in the
    # invalid-design failure; that failure was in the windowing, not here).
    # =========================================================================
    def find_block_boundary(sub):
        sub_idx = np.where(subjects_np == sub)[0]
        y_sub = y_np[sub_idx]
        change_points = np.where(y_sub != y_sub[0])[0]
        assert len(change_points) > 0, f"sub-{sub}: only one class present, cannot find a block boundary"
        n1 = int(change_points[0])
        assert np.all(y_sub[:n1] == y_sub[0]), f"sub-{sub}: block 1 is not a contiguous run"
        assert np.all(y_sub[n1:] == y_sub[n1]), f"sub-{sub}: block 2 is not a contiguous run"
        assert y_sub[0] != y_sub[n1], f"sub-{sub}: block 1/2 classes are identical, boundary detection failed"
        n2 = len(y_sub) - n1
        return sub_idx, n1, n2

    block_info = {sub: find_block_boundary(sub) for sub in unique_subjects}
    log.info("Block boundaries detected for all 29 subjects (contiguity verified).")

    def split_offset_for(sub, block_num, frac):
        _, n1, n2 = block_info[sub]
        if block_num == 0:
            return n1
        elif block_num == 1:
            return int(round(frac * n1))
        else:
            return n1 + int(round(frac * n2))

    # =========================================================================
    # FIXED WINDOW SIZE W -- computed directly from data as the largest value
    # feasible at the most extreme position: the global minimum before/after
    # headroom across every (position, subject) pair. Not hardcoded.
    # =========================================================================
    min_headroom = float("inf")
    binding_case = None
    for name, block_num, frac in POSITIONS:
        for sub in unique_subjects:
            _, n1, n2 = block_info[sub]
            n_total = n1 + n2
            split_offset = split_offset_for(sub, block_num, frac)
            before, after = split_offset, n_total - split_offset
            if before < min_headroom:
                min_headroom, binding_case = before, (name, sub, "before")
            if after < min_headroom:
                min_headroom, binding_case = after, (name, sub, "after")
    W = int(math.floor(min_headroom))
    assert W >= 20, f"computed fixed window W={W} is too small to be a usable design (binding case: {binding_case})"
    log.info(f"Fixed window W={W} trials (binding case: {binding_case}, pooled 2W*29={2*W*29} trials/position)")
    total_span_seconds = float(2 * W * MEAN_ITI_SECONDS_REFERENCE)
    log.info(f"Temporal span per window (2W trials, MEAN_ITI_SECONDS_REFERENCE="
             f"{MEAN_ITI_SECONDS_REFERENCE}s/trial, reused from F-DRIFT-C L016): "
             f"{total_span_seconds:.1f}s")

    # =========================================================================
    # RIEMANNIAN / EA UTILITIES -- IDENTICAL to run_step4_drift_control.py's
    # pre-F3 pooled-only EA (family consistency).
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

    # FIX 1 -- C3 hardening (2026-08-20). Every windowed pseudo-class pair
    # here is exactly W/W = 50/50 by construction, so this check should
    # NEVER fire; if it does, that signals a windowing bug, not a design
    # property to route around.
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
                f"[C3 FAIL] {label}: class balance {balance} is outside the 45/55 band -- this "
                "windowed design is supposed to be exactly 50/50 by construction, so this "
                "indicates a bug in the windowing logic, not an acceptable imbalanced design."
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
    # Build the FIXED-WINDOW pseudo-label for ONE position across all 29
    # subjects: W trials immediately before the split (pseudo-class 0), W
    # trials immediately after (pseudo-class 1) -- identical N/balance for
    # every subject and every position.
    # =========================================================================
    def build_windowed_pseudo_labels(block_num, frac):
        idx_list, y_pseudo_list, subj_list = [], [], []
        for sub in unique_subjects:
            sub_idx, n1, n2 = block_info[sub]
            split_offset = split_offset_for(sub, block_num, frac)
            before_idx = sub_idx[split_offset - W: split_offset]
            after_idx = sub_idx[split_offset: split_offset + W]
            assert len(before_idx) == W and len(after_idx) == W, (
                f"sub-{sub} block={block_num} frac={frac}: window size mismatch "
                f"(before={len(before_idx)}, after={len(after_idx)}, expected W={W})"
            )
            idx_list.append(np.concatenate([before_idx, after_idx]))
            y_pseudo_list.append(np.concatenate([np.zeros(W, dtype=np.int64), np.ones(W, dtype=np.int64)]))
            subj_list.append(np.full(2 * W, sub))
        pseudo_idx = np.concatenate(idx_list)
        y_pseudo = np.concatenate(y_pseudo_list)
        subjects_pseudo = np.concatenate(subj_list)
        return pseudo_idx, y_pseudo, subjects_pseudo

    # =========================================================================
    # LOSO for ONE position -- pseudo labels, single seed=42, all 29 subjects.
    # =========================================================================
    def run_position_test(name, block_num, frac):
        log.info(f"\n{'='*70}\n  F-DRIFT-E (redesigned): {name}\n{'='*70}")
        pseudo_idx, y_pseudo, subjects_pseudo = build_windowed_pseudo_labels(block_num, frac)
        X_pseudo = X_np[pseudo_idx]
        n_total_trials = len(y_pseudo)
        log.info(f"  N={n_total_trials} (2W={2*W} per subject x 29), "
                 f"pseudo-class balance={np.bincount(y_pseudo).tolist()}")

        fold_records = []
        pre_cal_accs, post_cal_accs, pre_cal_bal_accs, post_cal_bal_accs = [], [], [], []

        for fold_idx, test_sub in enumerate(unique_subjects):
            fold_start = time.time()
            is_holdout = subjects_pseudo == test_sub
            X_train, y_train = X_pseudo[~is_holdout], y_pseudo[~is_holdout]
            X_k, y_k = X_pseudo[is_holdout], y_pseudo[is_holdout]
            assert len(np.unique(y_train)) == 2 and len(np.unique(y_k)) == 2, (
                f"{name} fold sub-{test_sub}: missing a pseudo-class"
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
                                              f"{name} fold sub-{test_sub} pre_cal")
            post_cal_plaus = c3_balance_check(y_test, final_preds, post_cal_acc,
                                               f"{name} fold sub-{test_sub} post_cal")

            log.info(f"  [{name}] fold {fold_idx+1}/{len(unique_subjects)} sub-{test_sub} -> "
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

        pre_mean, pre_std = float(np.mean(pre_cal_accs)), float(np.std(pre_cal_accs))
        post_mean, post_std = float(np.mean(post_cal_accs)), float(np.std(post_cal_accs))
        pre_bal_mean = float(np.mean(pre_cal_bal_accs))
        post_bal_mean = float(np.mean(post_cal_bal_accs))
        log.info(f"  [{name}] DONE -- pre_cal={pre_mean:.4f}+/-{pre_std:.4f} (bal={pre_bal_mean:.4f}) "
                 f"post_cal={post_mean:.4f}+/-{post_std:.4f} (bal={post_bal_mean:.4f})")

        return {
            "position": name, "block": block_num, "fraction": frac,
            "N_total_trials": n_total_trials, "N_per_subject_per_class": W,
            "temporal_span_seconds": total_span_seconds,
            "fold_results": fold_records,
            "pre_calibration_accuracy_mean": pre_mean, "pre_calibration_accuracy_std": pre_std,
            "post_calibration_accuracy_mean": post_mean, "post_calibration_accuracy_std": post_std,
            "pre_calibration_balanced_accuracy_mean": pre_bal_mean,
            "post_calibration_balanced_accuracy_mean": post_bal_mean,
        }

    position_results = {name: run_position_test(name, block_num, frac) for name, block_num, frac in POSITIONS}

    # =========================================================================
    # DECISIONS.md's pre-registered F-DRIFT-E REDESIGN verdict -- applied to
    # BALANCED accuracy, using true_boundary-AT-W as the reference (NOT the
    # full-data 0.7078 figure, which uses far more trials and is not
    # comparable at matched N).
    # =========================================================================
    true_boundary_bal_acc = position_results["true_boundary"]["post_calibration_balanced_accuracy_mean"]
    shifted_names = [n for n, _, _ in POSITIONS if n != "true_boundary"]
    diffs = {n: true_boundary_bal_acc - position_results[n]["post_calibration_balanced_accuracy_mean"]
             for n in shifted_names}

    all_within_tolerance = all(abs(d) <= NOT_PRIVILEGED_TOLERANCE for d in diffs.values())
    all_exceed_margin = all(d > PRIVILEGED_MARGIN for d in diffs.values())

    if all_within_tolerance:
        verdict = (f"NOT PRIVILEGED -- every shifted position's post_cal balanced accuracy is "
                   f"within {NOT_PRIVILEGED_TOLERANCE} of the true-boundary-at-W value "
                   f"({true_boundary_bal_acc:.4f}). Any temporal partition at matched N/balance/"
                   "span yields the same calibrated accuracy.")
    elif all_exceed_margin:
        verdict = (f"PRIVILEGED -- the true-boundary-at-W value ({true_boundary_bal_acc:.4f}) "
                   f"exceeds every shifted position by more than {PRIVILEGED_MARGIN}. The true "
                   "boundary carries information a generic temporal split does not. Report and "
                   "stop for discussion; this would materially change the conclusion.")
    else:
        verdict = "MIXED -- report the full 7-point profile. No further interpretation without discussion."

    log.info(f"\n{'='*70}\n  F-DRIFT-E (redesigned) SUMMARY\n{'='*70}")
    log.info(f"  Fixed window W={W} trials/class/subject, N={2*W*29}/position, "
             f"temporal span={total_span_seconds:.1f}s")
    for name, _, _ in POSITIONS:
        r = position_results[name]
        log.info(f"  {name:<16}: pre_cal={r['pre_calibration_accuracy_mean']:.4f} "
                 f"post_cal={r['post_calibration_accuracy_mean']:.4f} "
                 f"post_cal_balanced={r['post_calibration_balanced_accuracy_mean']:.4f}")
    log.info(f"  True-boundary-at-W (reference): {true_boundary_bal_acc:.4f}")
    log.info(f"  Diffs (true_boundary - shifted): { {k: round(v,4) for k,v in diffs.items()} }")
    log.info(f"  DECISIONS.md VERDICT: {verdict}")

    results_payload = {
        "condition": "F-DRIFT-E REDESIGN — fixed symmetric window (W trials before/after each "
                      "split point), 7 positions (6 shifted + true boundary at matched N), "
                      "identical calibrated pipeline, single seed=42, full 29-fold LOSO per position",
        "hyperparameters": {
            "pca_max_components": PCA_MAX_COMPONENTS, "logreg_C": LOGREG_C,
            "random_seed": RANDOM_SEED, "cov_shrinkage": COV_SHRINKAGE, "cal_fraction": CAL_FRACTION,
        },
        "fixed_window_W": W, "window_binding_case": str(binding_case),
        "temporal_span_seconds": total_span_seconds,
        "mean_iti_seconds_reference": MEAN_ITI_SECONDS_REFERENCE,
        "mean_iti_reference_source": "reused from F-DRIFT-C's independently-extracted real onset "
                                      "times (RESULTS_LEDGER.md L016), not re-extracted here",
        "position_results": position_results,
        "true_boundary_at_w_reference": true_boundary_bal_acc,
        "diffs_from_true_boundary": diffs,
        "decisions_md_verdict": verdict,
        "invalidated_prior_attempt": "the original (non-windowed) F-DRIFT-E run is logged as "
                                      "INVALID-DESIGN, RESULTS_LEDGER.md L023 -- superseded by this run",
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
    for name, r in position_results.items():
        assert 0.0 <= r["pre_calibration_accuracy_mean"] <= 1.0
        assert 0.0 <= r["post_calibration_accuracy_mean"] <= 1.0
        assert len(r["fold_results"]) == 29, (
            f"[C3 PLAUSIBILITY FAIL] {name} expected 29 folds, got {len(r['fold_results'])}"
        )
        assert r["N_total_trials"] == 2 * W * 29, (
            f"[C3 PLAUSIBILITY FAIL] {name} N_total_trials={r['N_total_trials']} != 2*W*29={2*W*29}"
        )
    assert len(position_results) == 7, f"[C3 PLAUSIBILITY FAIL] expected 7 positions, got {len(position_results)}"
    log.info(f"  [C3] plausibility: 7 positions x 29 folds = 203 folds, identical N={2*W*29} per "
             f"position, all balance checks passed (50/50 by construction) -- OK")

    return {
        name: {
            "pre_cal": position_results[name]["pre_calibration_accuracy_mean"],
            "post_cal": position_results[name]["post_calibration_accuracy_mean"],
            "post_cal_balanced": position_results[name]["post_calibration_balanced_accuracy_mean"],
        }
        for name, _, _ in POSITIONS
    } | {"verdict": verdict, "fixed_window_W": W, "output_path": OUTPUT_JSON}


@app.local_entrypoint()
def main():
    print("F-DRIFT-E (REDESIGNED) — boundary-privilege check, fixed symmetric window")
    print("7 positions: 25/50/75% through block 1, 25/50/75% through block 2, and the true "
          "boundary itself -- all under the SAME fixed W-trial window (computed from data, not "
          "hardcoded), so N/balance/span are identical across positions.")
    print("Identical calibrated pipeline, single seed=42, full 29-fold LOSO per position (203 folds).")
    print("Includes C3 balance hardening throughout (class balance/majority rate/balanced accuracy, "
          "hard-fails outside 45/55 -- should never fire here since windows are exactly 50/50).")
    print(f"Pre-registered rule (DECISIONS.md, BALANCED-ACCURACY terms): all shifted positions "
          f"within {NOT_PRIVILEGED_TOLERANCE} of true-boundary-at-W -> NOT PRIVILEGED | "
          f"true boundary exceeds every shifted position by >{PRIVILEGED_MARGIN} -> PRIVILEGED | "
          f"mixed -> report only.\n")
    results = run_drift_control_e.remote()
    print("\nF-DRIFT-E (REDESIGNED) RESULTS:")
    for k, v in results.items():
        print(f"  {k}: {v}")
