# =============================================================================
# command to run:
#   modal run run_step4_drift_control_f_onset_exclusion.py::main
# run_step4_drift_control_f_onset_exclusion.py
#
# F-DRIFT-F — ONSET-EXCLUSION TEST (new fix-ID, added 2026-08-20, AUDIT.md
# fix register; DECISIONS.md pre-registered rule) -- FINAL CONTROL,
# regardless of outcome.
#
# WHY THIS SCRIPT EXISTS:
#   F-DRIFT-E REDESIGN's post-hoc result (RESULTS_LEDGER.md L025) found
#   that calibrated balanced accuracy tracks proximity to a BLOCK ONSET,
#   not proximity to the real task boundary: a pseudo-label with ZERO
#   task content, entirely inside one block (`block1_25pct`), reached
#   0.8585 balanced accuracy against the true boundary's 0.8968. This
#   reopened F-DRIFT-C's "temporal separation" interpretation as
#   PROVISIONAL (ledger L026): under F-DRIFT-C's
#   `pseudo_label(i) = (i // k) % 2` construction, label 0 always
#   contains a block's onset trials, with the onset region's
#   concentration within label 0 varying systematically with k. The
#   F-DRIFT-C curve may therefore be a dose-response in ONSET
#   CONCENTRATION rather than in temporal SEPARATION. This script tests
#   that directly.
#
# TEST DESIGN (pre-registered in DECISIONS.md BEFORE this script runs):
#
#   (a) ONSET-EXCLUDED K-SWEEP. Re-run F-DRIFT-C's k-sweep
#       (pseudo_label(i) = (i // k) % 2, k in {1,2,5,10,25,50,100}, both
#       real classes separately) AFTER dropping the first 50 trials of
#       each block for every subject, with k recomputed relative to the
#       TRUNCATED (onset-excluded) sequence (i re-indexed from 0 within
#       the remaining ~140-170 trials). Same calibrated pipeline,
#       seed=42, full 29-fold LOSO per (k, class). Reports BALANCED
#       pre_cal/post_cal per k (per FIX 1's hardening -- this test is
#       explicitly declared an imbalanced design at high k, since a
#       truncated ~150-trial sequence cannot split evenly at k=50/100;
#       balanced accuracy is the primary metric here, matching the
#       instruction to report "balanced pre_cal and post_cal per k").
#
#   (b) ONSET-DISTANCE PARAMETRIC SWEEP. Split positions at trial
#       distances {25,50,75,100,125,150,175} from each block's own onset
#       (block 1's onset = session trial 0; block 2's onset = the true
#       boundary, trial n1) -- 7 distances x 2 blocks = 14 positions.
#       Same FIXED SYMMETRIC WINDOW design as the F-DRIFT-E redesign (W
#       trials before/after the split point, W re-derived from data as
#       the largest value feasible at the tightest position -- NOT
#       hardcoded, and will come out smaller than F-DRIFT-E's W=48 given
#       distance=25 is the tightest constraint here). Same calibrated
#       pipeline, seed=42, full 29-fold LOSO per position (14x29=406
#       folds). Reports balanced post_cal per (distance, block), plus a
#       POOLED-by-distance value (mean of block1/block2 at the same
#       distance) as the primary curve. Balance is exactly 50/50 by
#       construction here (same as F-DRIFT-E), so the C3 gate is NOT
#       declared imbalanced for this sub-test.
#
# PRE-REGISTERED VERDICT RULE (DECISIONS.md, fixed BEFORE this script
# runs, BALANCED-ACCURACY terms):
#   (a) collapses to chance (<0.55 balanced) at every k
#       -> the onset transient explains the entire drift effect; the
#          F-DRIFT-C separation interpretation is WITHDRAWN and replaced
#          by an onset-transient account.
#   (a) retains a rise (highest-k balanced accuracy >= 0.62)
#       -> temporal separation contributes independently of onset
#          concentration; both mechanisms are reported.
#   (b) operationalized via SPEARMAN RANK CORRELATION between distance-
#       from-onset and the pooled-by-distance balanced post_cal curve --
#       NOT strict pointwise ordering, per the F-DRIFT-C mis-
#       specification lesson (a strict-ordering check on a handful of
#       noisy single-seed points is fragile), applied proactively here.
#       rho <= -0.6 and one-sided p < 0.05 (H1: rho < 0)
#           -> accuracy decays with distance from onset; confirms the
#              onset account.
#       otherwise (near zero / not significant)
#           -> accuracy is flat across distances; the onset account is
#              wrong, L025's pattern needs another explanation. Report
#              and stop for discussion.
#
# EXTENDS PAST THE PREVIOUSLY DECLARED FINAL CONTROL (F-DRIFT-E) because
# the E redesign CHANGED THE MECHANISM under discussion rather than
# confirming or cleanly rejecting the original hypothesis -- recorded
# explicitly in DECISIONS.md. F-DRIFT-F is FINAL regardless of its own
# outcome; results/SYNTHESIS.md is rewritten once it reports, then the
# paper-scope decision is made.
#
# Usage: modal run run_step4_drift_control_f_onset_exclusion.py::main
# =============================================================================

import modal

app    = modal.App("bci-condition4-drift-control-f-onset-exclusion")
volume = modal.Volume.from_name("eeg-data-vol")

RAW_DATA_PATH = "/data/processed_eeg_all_subjects.npz"
OUTPUT_JSON   = "/data/results_condition4_drift_control_f_onset_exclusion.json"
VOLUME_PATH   = "/data"

SFREQ, N_CHANNELS = 250, 62

RANDOM_SEED         = 42
COV_SHRINKAGE       = 0.1
PCA_MAX_COMPONENTS  = 35
LOGREG_C            = 1.0
CAL_FRACTION        = 0.15
SHRINK_GRID         = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS     = 3

# --- (a) onset-excluded k-sweep ---
K_SWEEP        = [1, 2, 5, 10, 25, 50, 100]
ONSET_DROP_N   = 50   # trials dropped from the start of each block, per subject/class

# --- (b) onset-distance parametric sweep ---
DISTANCES = [25, 50, 75, 100, 125, 150, 175]
BLOCKS    = [1, 2]

# DECISIONS.md pre-registered thresholds (fixed before this script runs).
COLLAPSE_TO_CHANCE_THRESHOLD = 0.55   # (a): every k below this -> onset explains everything
INDEPENDENT_RISE_THRESHOLD   = 0.62   # (a): highest k at/above this -> separation also contributes
DECAY_RHO_THRESHOLD          = -0.6   # (b): rho at/below this (and p<0.05) -> confirms onset account
DECAY_P_THRESHOLD             = 0.05

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy<2", "scikit-learn==1.4.2", "scipy==1.14.1")
)


@app.function(image=image, cpu=4.0, volumes={VOLUME_PATH: volume}, timeout=86400, memory=16384)
def run_drift_control_f():

    import logging, time, math, json
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
    from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("condition4-drift-control-f")

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

    # FIX 1 -- C3 hardening (2026-08-20, per F-DRIFT-E's INVALID-DESIGN
    # failure). Sub-test (a) explicitly declares an imbalanced design at
    # high k (a truncated ~150-trial sequence cannot split 50/50 at
    # k=50/100) -- balanced_accuracy is reported and used as the primary
    # metric there. Sub-test (b) is exactly 50/50 by construction and is
    # NOT declared imbalanced -- the gate should never fire there.
    def c3_balance_check(y_true, y_pred, acc, label, declared_imbalanced_design=False):
        counts = np.bincount(y_true, minlength=2)
        n = len(y_true)
        balance = (counts / n).tolist()
        majority_rate = float(counts.max() / n)
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        lift = float(acc - majority_rate)
        print(f"    [C3] {label}: class_balance={[round(b, 4) for b in balance]} "
              f"majority_rate={majority_rate:.4f} acc={acc:.4f} "
              f"acc_minus_majority={lift:+.4f} balanced_acc={bal_acc:.4f} "
              f"{'[DECLARED IMBALANCED]' if declared_imbalanced_design else ''}")
        if not declared_imbalanced_design:
            assert 0.45 <= min(balance) and max(balance) <= 0.55, (
                f"[C3 FAIL] {label}: class balance {balance} is outside the 45/55 band and this "
                "test does not declare an imbalanced design."
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
    # Generic per-(idx, y_pseudo, subjects_pseudo) LOSO runner shared by both
    # sub-tests. `declared_imbalanced` is forwarded to the C3 gate.
    # =========================================================================
    def run_loso(X_pseudo, y_pseudo, subjects_pseudo, label, declared_imbalanced=False):
        fold_records = []
        pre_cal_accs, post_cal_accs, pre_cal_bal_accs, post_cal_bal_accs = [], [], [], []

        for fold_idx, test_sub in enumerate(unique_subjects):
            fold_start = time.time()
            is_holdout = subjects_pseudo == test_sub
            X_train, y_train = X_pseudo[~is_holdout], y_pseudo[~is_holdout]
            X_k, y_k = X_pseudo[is_holdout], y_pseudo[is_holdout]
            assert len(np.unique(y_train)) == 2 and len(np.unique(y_k)) == 2, (
                f"{label} fold sub-{test_sub}: missing a pseudo-class"
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
                                              f"{label} fold sub-{test_sub} pre_cal", declared_imbalanced)
            post_cal_plaus = c3_balance_check(y_test, final_preds, post_cal_acc,
                                               f"{label} fold sub-{test_sub} post_cal", declared_imbalanced)

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

        return {
            "label": label,
            "fold_results": fold_records,
            "pre_calibration_accuracy_mean": float(np.mean(pre_cal_accs)),
            "post_calibration_accuracy_mean": float(np.mean(post_cal_accs)),
            "pre_calibration_balanced_accuracy_mean": float(np.mean(pre_cal_bal_accs)),
            "post_calibration_balanced_accuracy_mean": float(np.mean(post_cal_bal_accs)),
        }

    # =========================================================================
    # (a) ONSET-EXCLUDED K-SWEEP
    # =========================================================================
    def run_onset_excluded_k_test(real_class, real_class_name, k):
        idx_list, y_pseudo_list, subj_list = [], [], []
        for sub in unique_subjects:
            sub_idx = np.where(subjects_np == sub)[0]
            cls_idx = sub_idx[y_np[sub_idx] == real_class]
            n_full = len(cls_idx)
            assert n_full > ONSET_DROP_N + 10, (
                f"sub-{sub} class {real_class} has only {n_full} trials, cannot drop "
                f"{ONSET_DROP_N} onset trials and still have a usable sequence"
            )
            truncated_idx = cls_idx[ONSET_DROP_N:]
            n = len(truncated_idx)
            pseudo = ((np.arange(n) // k) % 2).astype(np.int64)
            idx_list.append(truncated_idx)
            y_pseudo_list.append(pseudo)
            subj_list.append(np.full(n, sub))
        pseudo_idx = np.concatenate(idx_list)
        y_pseudo = np.concatenate(y_pseudo_list)
        subjects_pseudo = np.concatenate(subj_list)
        X_pseudo = X_np[pseudo_idx]

        # k>=50 relative to a ~140-170-trial truncated sequence cannot split
        # 50/50 -- declared imbalanced for those k, matching this test's
        # documented, expected property (not a bug).
        declared_imbalanced = k >= 50
        return run_loso(X_pseudo, y_pseudo, subjects_pseudo,
                         f"onset_excl_k={k}_{real_class_name}", declared_imbalanced)

    log.info(f"\n{'#'*70}\n  (a) ONSET-EXCLUDED K-SWEEP\n{'#'*70}")
    k_sweep_results = {}
    for k in K_SWEEP:
        sweep_start = time.time()
        res_search = run_onset_excluded_k_test(0, "search_only", k)
        res_memorize = run_onset_excluded_k_test(1, "memorize_only", k)
        combined_pre_bal = float(np.mean([res_search["pre_calibration_balanced_accuracy_mean"],
                                           res_memorize["pre_calibration_balanced_accuracy_mean"]]))
        combined_post_bal = float(np.mean([res_search["post_calibration_balanced_accuracy_mean"],
                                            res_memorize["post_calibration_balanced_accuracy_mean"]]))
        k_sweep_results[k] = {
            "search_only": res_search, "memorize_only": res_memorize,
            "combined_pre_calibration_balanced_accuracy": combined_pre_bal,
            "combined_post_calibration_balanced_accuracy": combined_post_bal,
        }
        log.info(f"  k={k}: combined pre_cal_balanced={combined_pre_bal:.4f} "
                 f"post_cal_balanced={combined_post_bal:.4f} [{time.time()-sweep_start:.0f}s]")

    k_sorted = sorted(K_SWEEP)
    pre_bal_curve = [k_sweep_results[k]["combined_pre_calibration_balanced_accuracy"] for k in k_sorted]
    # bool(...) here and below: numpy scalar comparisons (e.g. from spearmanr's float64
    # outputs) yield numpy.bool_, which json.dump cannot serialize -- cast to native bool
    # at the point of computation, not just at the point of use, so this can't recur.
    collapses_to_chance = bool(all(v < COLLAPSE_TO_CHANCE_THRESHOLD for v in pre_bal_curve))
    highest_k_bal = pre_bal_curve[-1]
    retains_independent_rise = bool(highest_k_bal >= INDEPENDENT_RISE_THRESHOLD)

    if collapses_to_chance:
        verdict_a = (f"COLLAPSES TO CHANCE -- every k's pre_cal_balanced < {COLLAPSE_TO_CHANCE_THRESHOLD} "
                     f"once the first {ONSET_DROP_N} trials of each block are excluded. The onset "
                     "transient explains the entire drift effect. The F-DRIFT-C separation "
                     "interpretation is WITHDRAWN and replaced by an onset-transient account.")
    elif retains_independent_rise:
        verdict_a = (f"RETAINS A RISE -- highest-k pre_cal_balanced ({highest_k_bal:.4f}) >= "
                     f"{INDEPENDENT_RISE_THRESHOLD} even after onset exclusion. Temporal separation "
                     "contributes independently of onset concentration. Both mechanisms are reported.")
    else:
        verdict_a = (f"IN BETWEEN -- neither collapses to chance nor clearly retains an independent "
                     f"rise (highest-k pre_cal_balanced={highest_k_bal:.4f}). Report the full curve; "
                     "do not assert either mechanism without discussion.")
    log.info(f"  (a) VERDICT: {verdict_a}")

    # =========================================================================
    # (b) ONSET-DISTANCE PARAMETRIC SWEEP -- fixed symmetric window, exactly
    # as the F-DRIFT-E redesign, but split points defined by ABSOLUTE trial
    # distance from each block's own onset rather than by fraction.
    # =========================================================================
    def find_block_boundary(sub):
        sub_idx = np.where(subjects_np == sub)[0]
        y_sub = y_np[sub_idx]
        change_points = np.where(y_sub != y_sub[0])[0]
        assert len(change_points) > 0, f"sub-{sub}: only one class present, cannot find a block boundary"
        n1 = int(change_points[0])
        assert np.all(y_sub[:n1] == y_sub[0]), f"sub-{sub}: block 1 is not a contiguous run"
        assert np.all(y_sub[n1:] == y_sub[n1]), f"sub-{sub}: block 2 is not a contiguous run"
        n2 = len(y_sub) - n1
        return sub_idx, n1, n2

    block_info = {sub: find_block_boundary(sub) for sub in unique_subjects}

    def onset_split_offset(sub, block_num, distance):
        _, n1, n2 = block_info[sub]
        return distance if block_num == 1 else n1 + distance

    positions_b = [(f"block{b}_d{d}", b, d) for b in BLOCKS for d in DISTANCES]

    min_headroom = float("inf")
    binding_case = None
    for name, block_num, d in positions_b:
        for sub in unique_subjects:
            _, n1, n2 = block_info[sub]
            n_total = n1 + n2
            split_offset = onset_split_offset(sub, block_num, d)
            before, after = split_offset, n_total - split_offset
            if before < min_headroom:
                min_headroom, binding_case = before, (name, sub, "before")
            if after < min_headroom:
                min_headroom, binding_case = after, (name, sub, "after")
    W = int(math.floor(min_headroom))
    assert W >= 10, f"computed fixed window W={W} too small (binding case: {binding_case})"
    log.info(f"\n{'#'*70}\n  (b) ONSET-DISTANCE PARAMETRIC SWEEP -- W={W} "
             f"(binding case: {binding_case})\n{'#'*70}")

    def build_windowed_pseudo_labels(block_num, distance):
        idx_list, y_pseudo_list, subj_list = [], [], []
        for sub in unique_subjects:
            sub_idx, n1, n2 = block_info[sub]
            split_offset = onset_split_offset(sub, block_num, distance)
            before_idx = sub_idx[split_offset - W: split_offset]
            after_idx = sub_idx[split_offset: split_offset + W]
            assert len(before_idx) == W and len(after_idx) == W, (
                f"sub-{sub} block={block_num} d={distance}: window size mismatch"
            )
            idx_list.append(np.concatenate([before_idx, after_idx]))
            y_pseudo_list.append(np.concatenate([np.zeros(W, dtype=np.int64), np.ones(W, dtype=np.int64)]))
            subj_list.append(np.full(2 * W, sub))
        return (np.concatenate(idx_list), np.concatenate(y_pseudo_list), np.concatenate(subj_list))

    distance_results = {}
    for name, block_num, d in positions_b:
        pseudo_idx, y_pseudo, subjects_pseudo = build_windowed_pseudo_labels(block_num, d)
        X_pseudo = X_np[pseudo_idx]
        r = run_loso(X_pseudo, y_pseudo, subjects_pseudo, name, declared_imbalanced=False)
        distance_results[name] = r
        log.info(f"  {name}: post_cal_balanced="
                 f"{r['post_calibration_balanced_accuracy_mean']:.4f}")

    pooled_by_distance = {}
    for d in DISTANCES:
        vals = [distance_results[f"block{b}_d{d}"]["post_calibration_balanced_accuracy_mean"] for b in BLOCKS]
        pooled_by_distance[d] = float(np.mean(vals))

    distances_sorted = sorted(DISTANCES)
    pooled_curve = [pooled_by_distance[d] for d in distances_sorted]
    # scipy's spearmanr returns numpy.float64 -- cast to native float immediately so
    # every downstream comparison (and the eventual json.dump) sees plain Python types.
    rho, p_two_sided = (float(v) for v in spearmanr(distances_sorted, pooled_curve))
    p_one_sided = p_two_sided / 2.0 if rho < 0 else 1.0 - p_two_sided / 2.0

    decays_with_distance = bool((rho <= DECAY_RHO_THRESHOLD) and (p_one_sided < DECAY_P_THRESHOLD))
    if decays_with_distance:
        verdict_b = (f"DECAYS WITH DISTANCE -- rho={rho:.4f} <= {DECAY_RHO_THRESHOLD}, "
                     f"one-sided p={p_one_sided:.4f} < {DECAY_P_THRESHOLD}. Confirms the onset account.")
    else:
        verdict_b = (f"FLAT -- rho={rho:.4f}, one-sided p={p_one_sided:.4f} does not meet the decay "
                     f"bar (rho<={DECAY_RHO_THRESHOLD} and p<{DECAY_P_THRESHOLD}). The onset account is "
                     "wrong as stated; L025's pattern needs another explanation. Report and stop for discussion.")
    log.info(f"  (b) VERDICT: {verdict_b}")

    # =========================================================================
    # SAVE + C3 PLAUSIBILITY (write-before-raise, per established convention)
    # =========================================================================
    results_payload = {
        "condition": "F-DRIFT-F — onset-exclusion test: (a) onset-excluded k-sweep, "
                      "(b) onset-distance parametric sweep. FINAL CONTROL.",
        "hyperparameters": {
            "pca_max_components": PCA_MAX_COMPONENTS, "logreg_C": LOGREG_C,
            "random_seed": RANDOM_SEED, "cov_shrinkage": COV_SHRINKAGE, "cal_fraction": CAL_FRACTION,
        },
        "a_onset_excluded_k_sweep": {
            "onset_drop_n": ONSET_DROP_N,
            "sweep_results": {str(k): k_sweep_results[k] for k in k_sorted},
            "pre_cal_balanced_curve": {str(k): pre_bal_curve[i] for i, k in enumerate(k_sorted)},
            "collapses_to_chance": collapses_to_chance,
            "retains_independent_rise": retains_independent_rise,
            "verdict": verdict_a,
        },
        "b_onset_distance_sweep": {
            "fixed_window_W": W, "window_binding_case": str(binding_case),
            "distance_results": distance_results,
            "pooled_by_distance": pooled_by_distance,
            "spearman_rho": float(rho), "spearman_p_two_sided": float(p_two_sided),
            "spearman_p_one_sided_rho_lt_0": float(p_one_sided),
            "decays_with_distance": decays_with_distance,
            "verdict": verdict_b,
        },
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_payload, f, indent=2)
    volume.commit()
    log.info(f"  Saved: {OUTPUT_JSON}")

    for k in K_SWEEP:
        assert len(k_sweep_results[k]["search_only"]["fold_results"]) == 29
        assert len(k_sweep_results[k]["memorize_only"]["fold_results"]) == 29
    for name in distance_results:
        assert len(distance_results[name]["fold_results"]) == 29, (
            f"[C3 PLAUSIBILITY FAIL] {name} expected 29 folds"
        )
    assert len(distance_results) == 14, f"[C3 PLAUSIBILITY FAIL] expected 14 positions, got {len(distance_results)}"
    log.info(f"  [C3] plausibility: (a) 7k x 2 classes x 29 folds=406, (b) 14 positions x 29 folds=406 -- OK")

    return {
        "a_verdict": verdict_a, "a_highest_k_balanced": highest_k_bal,
        "a_pre_cal_balanced_curve": {str(k): pre_bal_curve[i] for i, k in enumerate(k_sorted)},
        "b_verdict": verdict_b, "b_spearman_rho": float(rho), "b_spearman_p_one_sided": float(p_one_sided),
        "b_pooled_by_distance": pooled_by_distance, "b_fixed_window_W": W,
        "output_path": OUTPUT_JSON,
    }


@app.local_entrypoint()
def main():
    print("F-DRIFT-F — onset-exclusion test (FINAL CONTROL, regardless of outcome)")
    print("(a) onset-excluded k-sweep: F-DRIFT-C's k-sweep re-run after dropping each block's "
          "first 50 trials per subject, k recomputed on the truncated sequence.")
    print("(b) onset-distance parametric sweep: 7 distances x 2 blocks = 14 fixed-window positions, "
          "balanced accuracy vs. distance from the nearest block onset.")
    print(f"Pre-registered rule (DECISIONS.md, balanced-accuracy terms): "
          f"(a) all k < {COLLAPSE_TO_CHANCE_THRESHOLD} -> onset explains everything, separation "
          f"WITHDRAWN | highest k >= {INDEPENDENT_RISE_THRESHOLD} -> both mechanisms reported. "
          f"(b) Spearman rho<={DECAY_RHO_THRESHOLD} and p<{DECAY_P_THRESHOLD} -> confirms onset "
          f"account | otherwise -> flat, report and stop for discussion.\n")
    results = run_drift_control_f.remote()
    print("\nF-DRIFT-F RESULTS:")
    for k, v in results.items():
        print(f"  {k}: {v}")
