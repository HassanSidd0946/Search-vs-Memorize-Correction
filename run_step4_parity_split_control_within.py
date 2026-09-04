# =============================================================================
# command to run:
#   modal run run_step4_parity_split_control_within.py::main
# run_step4_parity_split_control_within.py
#
# F-PARITY-WITHIN — THE MISSING CELL OF THE PARITY DESIGN (new fix-ID, added
# 2026-08-19, AUDIT.md fix register; DECISIONS.md pre-registered rule)
#
# WHY THIS SCRIPT EXISTS:
#   The real Search-vs-Memorize contrast has the LARGEST temporal separation
#   of anything tested in this project (block boundary + ~400s break) yet
#   the WORST cross-subject transfer measured so far: real-label pre_cal =
#   0.5201 (RESULTS_LEDGER.md L009), vs. F-DRIFT's within-block early/late
#   split pre_cal = 0.6418 (a SMALLER temporal separation giving BETTER
#   transfer -- the opposite of what a pure "more separation = more
#   decodable" story predicts on its own).
#
#   HYPOTHESIS: F-PARITY's odd/even block-order counterbalance (D2 -- odd
#   subjects run Search-first, even subjects run Memorize-first) makes
#   "early trial in the session" mean Class 0 for half the training pool
#   and Class 1 for the other half. Every OTHER driver in this codebase
#   trains on a MIXED-parity 28-subject pool, so this time-position cue
#   CANCELS OUT across the training pool -- suppressing exactly the kind
#   of temporal signal F-DRIFT/F-DRIFT-B show the pipeline is otherwise
#   very good at exploiting. Within a SINGLE parity group, the time-
#   position cue is consistent across every training subject (no
#   cancellation), so if the hypothesis is right, within-parity LOSO
#   should recover much higher pre_cal accuracy than the mixed-parity
#   real-label number.
#
# TEST DESIGN (pre-registered in DECISIONS.md BEFORE this script runs):
#   Two separate LOSO runs, REAL labels, identical calibrated pipeline
#   (byte-identical EA/tangent/PCA/shrinkage-calibration to the rest of
#   the pre-F3 Batch 1/F-DRIFT family), single seed=42:
#     - odd_only:  14-fold LOSO among the 14 odd-numbered subjects only
#                  (all Search-first).
#     - even_only: 15-fold LOSO among the 15 even-numbered subjects only
#                  (all Memorize-first).
#
# PRE-REGISTERED VERDICT RULE (DECISIONS.md, fixed BEFORE this script runs):
#   within-parity pre_cal rises toward ~0.64 (vs. cross-parity's already-
#   measured ~0.49, F-PARITY ledger L003)
#       -> hypothesis CONFIRMED: the apparent task signal is block-order/
#          time, visible when block order is consistent, cancelling when
#          it is not.
#   within-parity pre_cal stays near 0.52
#       -> hypothesis REJECTED: the asymmetry needs another explanation,
#          must NOT be asserted in the write-up.
#
# Usage: modal run run_step4_parity_split_control_within.py::main
# =============================================================================

import modal

app    = modal.App("bci-condition4-parity-split-control-within")
volume = modal.Volume.from_name("eeg-data-vol")

RAW_DATA_PATH = "/data/processed_eeg_all_subjects.npz"
OUTPUT_JSON   = "/data/results_condition4_parity_split_control_within.json"
VOLUME_PATH   = "/data"

SFREQ, N_CHANNELS = 250, 62

RANDOM_SEED         = 42
COV_SHRINKAGE       = 0.1
PCA_MAX_COMPONENTS  = 35
LOGREG_C            = 1.0
CAL_FRACTION        = 0.15
SHRINK_GRID         = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS     = 3

# DECISIONS.md pre-registered reference numbers (2026-08-19, fixed before
# this script runs) -- for the interpretation rule and logging only, not
# recomputed here.
FDRIFT_PRE_CAL_REFERENCE = 0.6418        # F-DRIFT early/late pseudo pre_cal
REAL_LABEL_PRE_CAL_REFERENCE = 0.5201    # mixed-parity real-label pre_cal (L009)
FPARITY_CROSS_ODD_TO_EVEN = 0.4927       # F-PARITY train_odd_test_even (L003)
FPARITY_CROSS_EVEN_TO_ODD = 0.4750       # F-PARITY train_even_test_odd (L003)
CONFIRM_THRESHOLD = 0.60                 # "rises toward ~0.64" -- treat >=0.60 as confirming
REJECT_THRESHOLD_BAND = 0.03             # "stays near 0.52" -- within +/-0.03 of 0.52 (D2's real-label figure)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy<2", "scikit-learn==1.4.2", "scipy")
)


@app.function(image=image, cpu=4.0, volumes={VOLUME_PATH: volume}, timeout=86400, memory=16384)
def run_parity_split_control_within():

    import logging, time, math, json
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
    from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("condition4-parity-split-within")

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
    odd_subjects  = [s for s in unique_subjects if int(s) % 2 == 1]
    even_subjects = [s for s in unique_subjects if int(s) % 2 == 0]
    assert (len(odd_subjects), len(even_subjects)) == (14, 15), (
        f"Expected 14 odd / 15 even subjects (D2), got {len(odd_subjects)} odd / {len(even_subjects)} even"
    )
    log.info(f"Odd subjects  (n={len(odd_subjects)}): {odd_subjects}")
    log.info(f"Even subjects (n={len(even_subjects)}): {even_subjects}")

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
    # LOSO WITHIN one parity subgroup -- REAL labels, single seed=42.
    # =========================================================================
    def run_within_parity(group_name, group_subjects):
        log.info(f"\n{'='*70}\n  F-PARITY-WITHIN: {group_name} ({len(group_subjects)} subjects)\n{'='*70}")
        group_mask = np.isin(subjects_np, group_subjects)
        X_group, y_group, subjects_group = X_np[group_mask], y_np[group_mask], subjects_np[group_mask]

        fold_records = []
        pre_cal_accs, post_cal_accs = [], []

        for fold_idx, test_sub in enumerate(group_subjects):
            fold_start = time.time()
            is_holdout = subjects_group == test_sub
            X_train, y_train = X_group[~is_holdout], y_group[~is_holdout]
            X_k, y_k = X_group[is_holdout], y_group[is_holdout]

            mu = X_train.mean(axis=(0, 2), keepdims=True)
            sd = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
            X_train_z = ((X_train - mu) / sd).astype(np.float32)
            X_k_z = ((X_k - mu) / sd).astype(np.float32)

            W = fit_ea_whitening(X_train_z)
            X_train_aligned = apply_ea_whitening_signal(X_train_z, W).astype(np.float32)
            X_k_aligned = apply_ea_whitening_signal(X_k_z, W).astype(np.float32)

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
            pre_cal_acc = float((global_clf.predict(X_test_pca) == y_test).mean())
            final_preds = linear_predict(coef_final, icpt_final, X_test_pca)
            post_cal_acc = float((final_preds == y_test).mean())
            metrics = compute_binary_metrics(y_test, final_preds)
            pre_cal_plaus = c3_balance_check(y_test, global_clf.predict(X_test_pca), pre_cal_acc,
                                              f"{group_name} fold sub-{test_sub} pre_cal")
            post_cal_plaus = c3_balance_check(y_test, final_preds, post_cal_acc,
                                               f"{group_name} fold sub-{test_sub} post_cal")

            log.info(f"  [{group_name}] fold {fold_idx+1}/{len(group_subjects)} sub-{test_sub} -> "
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
        log.info(f"  [{group_name}] DONE -- pre_cal={pre_mean:.4f}+/-{pre_std:.4f} "
                 f"post_cal={post_mean:.4f}+/-{post_std:.4f}")

        return {
            "group": group_name, "n_subjects": len(group_subjects), "subjects": group_subjects,
            "fold_results": fold_records,
            "pre_calibration_accuracy_mean": pre_mean, "pre_calibration_accuracy_std": pre_std,
            "post_calibration_accuracy_mean": post_mean, "post_calibration_accuracy_std": post_std,
        }

    result_odd = run_within_parity("odd_only", odd_subjects)
    result_even = run_within_parity("even_only", even_subjects)

    # Pooled (n-weighted) mean across both groups, matching each group's actual fold count.
    all_pre = [r["pre_calibration_acc"] for r in result_odd["fold_results"]] + \
              [r["pre_calibration_acc"] for r in result_even["fold_results"]]
    all_post = [r["post_calibration_acc"] for r in result_odd["fold_results"]] + \
               [r["post_calibration_acc"] for r in result_even["fold_results"]]
    pooled_pre_cal = float(np.mean(all_pre))
    pooled_post_cal = float(np.mean(all_post))

    # =========================================================================
    # DECISIONS.md's pre-registered F-PARITY-WITHIN verdict.
    # =========================================================================
    if pooled_pre_cal >= CONFIRM_THRESHOLD:
        verdict = ("HYPOTHESIS CONFIRMED -- within-parity pre_cal "
                   f"({pooled_pre_cal:.4f}) rises toward F-DRIFT's ~0.64 reference, well above the "
                   f"cross-parity values (0.4927/0.4750, F-PARITY ledger L003). The apparent task "
                   "signal is block-order/time, visible when block order is consistent across the "
                   "training pool and cancelling when it is not.")
    elif abs(pooled_pre_cal - REAL_LABEL_PRE_CAL_REFERENCE) <= REJECT_THRESHOLD_BAND:
        verdict = (f"HYPOTHESIS REJECTED -- within-parity pre_cal ({pooled_pre_cal:.4f}) stays near "
                   f"the mixed-parity real-label figure ({REAL_LABEL_PRE_CAL_REFERENCE}). The "
                   "asymmetry between real-label and pseudo-label transfer needs another explanation "
                   "and must NOT be asserted as block-order/time in the write-up.")
    else:
        verdict = (f"INTERMEDIATE -- within-parity pre_cal ({pooled_pre_cal:.4f}) is neither close to "
                   f"the ~0.64 confirming reference nor to the ~0.52 rejecting reference. Report "
                   "plainly; do not assert either conclusion without discussion.")

    log.info(f"\n{'='*70}\n  F-PARITY-WITHIN SUMMARY\n{'='*70}")
    log.info(f"  odd_only (14-fold) : pre_cal={result_odd['pre_calibration_accuracy_mean']:.4f} "
             f"post_cal={result_odd['post_calibration_accuracy_mean']:.4f}")
    log.info(f"  even_only (15-fold): pre_cal={result_even['pre_calibration_accuracy_mean']:.4f} "
             f"post_cal={result_even['post_calibration_accuracy_mean']:.4f}")
    log.info(f"  POOLED (29 folds)  : pre_cal={pooled_pre_cal:.4f} post_cal={pooled_post_cal:.4f}")
    log.info(f"  Reference -- F-DRIFT early/late pre_cal: {FDRIFT_PRE_CAL_REFERENCE} | "
             f"mixed-parity real-label pre_cal: {REAL_LABEL_PRE_CAL_REFERENCE} | "
             f"F-PARITY cross-parity: {FPARITY_CROSS_ODD_TO_EVEN}/{FPARITY_CROSS_EVEN_TO_ODD}")
    log.info(f"  DECISIONS.md VERDICT: {verdict}")

    results_payload = {
        "condition": "F-PARITY-WITHIN — LOSO within each parity subgroup (odd-only 14-fold, even-only "
                      "15-fold), REAL Search-vs-Memorize labels, identical calibrated pipeline, "
                      "single seed=42",
        "hyperparameters": {
            "pca_max_components": PCA_MAX_COMPONENTS, "logreg_C": LOGREG_C,
            "random_seed": RANDOM_SEED, "cov_shrinkage": COV_SHRINKAGE, "cal_fraction": CAL_FRACTION,
        },
        "odd_subjects": odd_subjects, "even_subjects": even_subjects,
        "results": {"odd_only": result_odd, "even_only": result_even},
        "pooled_pre_calibration_accuracy": pooled_pre_cal,
        "pooled_post_calibration_accuracy": pooled_post_cal,
        "decisions_md_verdict": verdict,
        "reference_numbers": {
            "fdrift_early_late_pre_cal": FDRIFT_PRE_CAL_REFERENCE,
            "real_label_mixed_parity_pre_cal": REAL_LABEL_PRE_CAL_REFERENCE,
            "fparity_cross_odd_to_even": FPARITY_CROSS_ODD_TO_EVEN,
            "fparity_cross_even_to_odd": FPARITY_CROSS_EVEN_TO_ODD,
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
    for res in (result_odd, result_even):
        assert 0.0 <= res["pre_calibration_accuracy_mean"] <= 1.0, (
            f"[C3 PLAUSIBILITY FAIL] {res['group']} pre_cal mean outside [0,1]"
        )
        assert 0.0 <= res["post_calibration_accuracy_mean"] <= 1.0, (
            f"[C3 PLAUSIBILITY FAIL] {res['group']} post_cal mean outside [0,1]"
        )
    assert len(result_odd["fold_results"]) == 14, (
        f"[C3 PLAUSIBILITY FAIL] odd_only expected 14 folds, got {len(result_odd['fold_results'])}"
    )
    assert len(result_even["fold_results"]) == 15, (
        f"[C3 PLAUSIBILITY FAIL] even_only expected 15 folds, got {len(result_even['fold_results'])}"
    )
    log.info(f"  [C3] plausibility: 14+15=29 folds total, all accuracies in [0,1] -- OK")

    return {
        "odd_only_pre_cal": result_odd["pre_calibration_accuracy_mean"],
        "odd_only_post_cal": result_odd["post_calibration_accuracy_mean"],
        "even_only_pre_cal": result_even["pre_calibration_accuracy_mean"],
        "even_only_post_cal": result_even["post_calibration_accuracy_mean"],
        "pooled_pre_cal": pooled_pre_cal,
        "pooled_post_cal": pooled_post_cal,
        "decisions_md_verdict": verdict,
        "output_path": OUTPUT_JSON,
    }


@app.local_entrypoint()
def main():
    print("F-PARITY-WITHIN — LOSO within each parity subgroup (the missing cell of the parity design)")
    print("odd_only: 14-fold LOSO among odd-numbered (Search-first) subjects, REAL labels")
    print("even_only: 15-fold LOSO among even-numbered (Memorize-first) subjects, REAL labels")
    print("Identical calibrated pipeline, single seed=42.")
    print(f"Pre-registered rule (DECISIONS.md): pre_cal ~0.64 -> hypothesis CONFIRMED (block-order/time) | "
          f"pre_cal ~0.52 -> hypothesis REJECTED.\n")
    results = run_parity_split_control_within.remote()
    print("\nF-PARITY-WITHIN RESULTS:")
    for k, v in results.items():
        print(f"  {k:<24}: {v}")
