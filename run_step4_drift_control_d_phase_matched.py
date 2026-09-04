# =============================================================================
# command to run:
#   modal run run_step4_drift_control_d_phase_matched.py::main
# run_step4_drift_control_d_phase_matched.py
#
# F-DRIFT-D — PHASE-MATCHED BLOCK CONTRAST, DRIFT-RESET HYPOTHESIS (new
# fix-ID, added 2026-08-20, AUDIT.md fix register; DECISIONS.md
# pre-registered rule)
#
# WHY THIS SCRIPT EXISTS:
#   F-PARITY-WITHIN (RUN 2026-08-20, REJECTED, DECISIONS.md/RESULTS_LEDGER.md
#   L017) ruled out block-order/time-cancellation as the explanation for why
#   the real-label contrast (largest temporal separation: block boundary +
#   ~400s break) transfers WORSE (pre_cal=0.5201) than F-DRIFT's within-block
#   early/late split (smaller separation, pre_cal=0.6418).
#
#   NEW HYPOTHESIS: drift is phase-locked to BLOCK ONSET and RESETS at the
#   ~400s inter-block break, rather than accumulating monotonically across
#   the whole session. Under this hypothesis, within-block position (early
#   vs. late) is highly decodable (F-DRIFT's 0.6418) because it is measured
#   relative to each block's own onset, while raw "which block" identity is
#   much less decodable once within-block position is controlled for --
#   because both blocks independently ramp through the same drift
#   trajectory rather than one continuing where the other left off.
#
# TEST DESIGN (pre-registered in DECISIONS.md BEFORE this script runs):
#   Label = REAL block/class identity (0=Search, 1=Memorize -- same as the
#   real Search-vs-Memorize label). Every trial is additionally tagged with
#   its within-block position third (early/mid/late), computed with the
#   SAME chronological-thirds logic already used in
#   run_step4_parity_split_control.py's compute_within_block_thirds
#   (thirds = np.minimum((np.arange(n) * 3) // n, 2), per subject per class).
#   Three separate binary classification problems, one per phase:
#     early: early-third-of-Search-block vs. early-third-of-Memorize-block
#     mid:   mid-third vs. mid-third
#     late:  late-third vs. late-third
#   Each phase run through the SAME calibrated LOSO pipeline (identical
#   EA/tangent/PCA/shrinkage-calibration to the rest of the pre-F3
#   Batch 1/F-DRIFT family), single seed=42, full 29-fold LOSO. Report
#   pre_calibration_acc/post_calibration_acc per phase AND pooled (mean
#   across the three phases).
#
# PRE-REGISTERED VERDICT RULE (DECISIONS.md, fixed BEFORE this script runs,
# stated in accuracy terms, NOT as a strict ordering -- per the F-DRIFT-C
# lesson that strict ordering across noisy single-seed points is fragile):
#   pooled phase-matched pre_cal < 0.55 (near chance), while within-block
#   early/late (F-DRIFT) sits at 0.6418
#       -> drift is phase-locked to block onset and RESETS at the break --
#          hypothesis SUPPORTED.
#   pooled phase-matched pre_cal >= 0.60
#       -> block identity carries cross-subject signal beyond position --
#          hypothesis REJECTED; the real-vs-pseudo transfer asymmetry still
#          needs another explanation.
#   0.55-0.60
#       -> inconclusive. Report plainly and stop for discussion.
#
# Usage: modal run run_step4_drift_control_d_phase_matched.py::main
# =============================================================================

import modal

app    = modal.App("bci-condition4-drift-control-d-phase-matched")
volume = modal.Volume.from_name("eeg-data-vol")

RAW_DATA_PATH = "/data/processed_eeg_all_subjects.npz"
OUTPUT_JSON   = "/data/results_condition4_drift_control_d_phase_matched.json"
VOLUME_PATH   = "/data"

SFREQ, N_CHANNELS = 250, 62

RANDOM_SEED         = 42
COV_SHRINKAGE       = 0.1
PCA_MAX_COMPONENTS  = 35
LOGREG_C            = 1.0
CAL_FRACTION        = 0.15
SHRINK_GRID         = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS     = 3

PHASES = ["early", "mid", "late"]

# DECISIONS.md pre-registered reference numbers (2026-08-20, fixed before
# this script runs) -- for the interpretation rule and logging only, not
# recomputed here.
FDRIFT_PRE_CAL_REFERENCE     = 0.6418   # F-DRIFT early/late pseudo pre_cal
REAL_LABEL_PRE_CAL_REFERENCE = 0.5201   # mixed-parity real-label pre_cal (L009)
SUPPORTED_THRESHOLD          = 0.55     # pooled pre_cal < this -> SUPPORTED
REJECTED_THRESHOLD           = 0.60     # pooled pre_cal >= this -> REJECTED

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy<2", "scikit-learn==1.4.2", "scipy")
)


@app.function(image=image, cpu=4.0, volumes={VOLUME_PATH: volume}, timeout=86400, memory=16384)
def run_drift_control_d():

    import logging, time, math, json
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
    from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("condition4-drift-control-d")

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
    # Within-block position thirds -- IDENTICAL logic to
    # run_step4_parity_split_control.py's compute_within_block_thirds:
    # chronological thirds within each subject's own class block.
    # =========================================================================
    def compute_within_block_thirds(y_np, subjects_np, unique_subjects):
        position_third = np.full(len(y_np), -1, dtype=np.int64)
        for sub in unique_subjects:
            sub_idx = np.where(subjects_np == sub)[0]
            for cls in (0, 1):
                cls_idx = sub_idx[y_np[sub_idx] == cls]
                n = len(cls_idx)
                assert n >= 3, f"sub-{sub} class {cls} has only {n} trials, cannot split into thirds"
                thirds = np.minimum((np.arange(n) * 3) // n, 2)
                position_third[cls_idx] = thirds
        assert (position_third >= 0).all(), "Every trial must be assigned a within-block position third."
        return position_third

    position_third_np = compute_within_block_thirds(y_np, subjects_np, unique_subjects)
    log.info(f"Within-block position thirds computed: "
             f"counts={np.bincount(position_third_np).tolist()} (early/mid/late)")

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
    # LOSO restricted to ONE phase (both real classes, position-third-matched)
    # -- REAL labels, single seed=42, all 29 subjects.
    # =========================================================================
    def run_phase(phase_name, phase_idx):
        log.info(f"\n{'='*70}\n  F-DRIFT-D: phase={phase_name}\n{'='*70}")
        phase_mask = position_third_np == phase_idx
        X_phase, y_phase, subjects_phase = X_np[phase_mask], y_np[phase_mask], subjects_np[phase_mask]
        log.info(f"  n_trials in phase '{phase_name}': {len(y_phase)} "
                 f"(class balance: {np.bincount(y_phase).tolist()})")

        fold_records = []
        pre_cal_accs, post_cal_accs = [], []

        for fold_idx, test_sub in enumerate(unique_subjects):
            fold_start = time.time()
            is_holdout = subjects_phase == test_sub
            X_train, y_train = X_phase[~is_holdout], y_phase[~is_holdout]
            X_k, y_k = X_phase[is_holdout], y_phase[is_holdout]
            assert len(np.unique(y_train)) == 2, (
                f"phase={phase_name} fold sub-{test_sub}: training pool missing a class"
            )
            assert len(np.unique(y_k)) == 2, (
                f"phase={phase_name} fold sub-{test_sub}: held-out subject missing a class in this phase"
            )

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
                                              f"{phase_name} fold sub-{test_sub} pre_cal")
            post_cal_plaus = c3_balance_check(y_test, final_preds, post_cal_acc,
                                               f"{phase_name} fold sub-{test_sub} post_cal")

            log.info(f"  [{phase_name}] fold {fold_idx+1}/{len(unique_subjects)} sub-{test_sub} -> "
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
        log.info(f"  [{phase_name}] DONE -- pre_cal={pre_mean:.4f}+/-{pre_std:.4f} "
                 f"post_cal={post_mean:.4f}+/-{post_std:.4f}")

        return {
            "phase": phase_name, "n_trials": int(len(y_phase)),
            "fold_results": fold_records,
            "pre_calibration_accuracy_mean": pre_mean, "pre_calibration_accuracy_std": pre_std,
            "post_calibration_accuracy_mean": post_mean, "post_calibration_accuracy_std": post_std,
        }

    phase_results = {phase: run_phase(phase, idx) for idx, phase in enumerate(PHASES)}

    # Pooled = mean of the three phase-level means (matches F-DRIFT-C's
    # "combined_*" convention: mean of components, not fold-count-weighted
    # pooling, so each phase contributes equally regardless of trial count).
    pooled_pre_cal = float(np.mean([phase_results[p]["pre_calibration_accuracy_mean"] for p in PHASES]))
    pooled_post_cal = float(np.mean([phase_results[p]["post_calibration_accuracy_mean"] for p in PHASES]))

    # =========================================================================
    # DECISIONS.md's pre-registered F-DRIFT-D verdict.
    # =========================================================================
    if pooled_pre_cal < SUPPORTED_THRESHOLD:
        verdict = (f"HYPOTHESIS SUPPORTED -- pooled phase-matched pre_cal ({pooled_pre_cal:.4f}) is near "
                   f"chance while F-DRIFT's within-block early/late split sits at "
                   f"{FDRIFT_PRE_CAL_REFERENCE}. Drift appears phase-locked to block onset and RESETS "
                   "at the ~400s inter-block break, rather than accumulating across the whole session.")
    elif pooled_pre_cal >= REJECTED_THRESHOLD:
        verdict = (f"HYPOTHESIS REJECTED -- pooled phase-matched pre_cal ({pooled_pre_cal:.4f}) shows "
                   "block identity carries cross-subject signal beyond within-block position. The "
                   "real-vs-pseudo transfer asymmetry still needs another explanation.")
    else:
        verdict = (f"INCONCLUSIVE -- pooled phase-matched pre_cal ({pooled_pre_cal:.4f}) falls between "
                   f"{SUPPORTED_THRESHOLD} and {REJECTED_THRESHOLD}. Report plainly; do not assert "
                   "either conclusion without discussion.")

    log.info(f"\n{'='*70}\n  F-DRIFT-D SUMMARY\n{'='*70}")
    for p in PHASES:
        r = phase_results[p]
        log.info(f"  {p:<5} (n={r['n_trials']:>4}, 29 folds): "
                 f"pre_cal={r['pre_calibration_accuracy_mean']:.4f} "
                 f"post_cal={r['post_calibration_accuracy_mean']:.4f}")
    log.info(f"  POOLED (mean of 3 phases): pre_cal={pooled_pre_cal:.4f} post_cal={pooled_post_cal:.4f}")
    log.info(f"  Reference -- F-DRIFT early/late pre_cal: {FDRIFT_PRE_CAL_REFERENCE} | "
             f"mixed-parity real-label pre_cal: {REAL_LABEL_PRE_CAL_REFERENCE}")
    log.info(f"  DECISIONS.md VERDICT: {verdict}")

    results_payload = {
        "condition": "F-DRIFT-D — phase-matched block contrast (early/mid/late within-block-position "
                      "thirds), REAL Search-vs-Memorize labels, identical calibrated pipeline, "
                      "single seed=42, full 29-fold LOSO per phase",
        "hyperparameters": {
            "pca_max_components": PCA_MAX_COMPONENTS, "logreg_C": LOGREG_C,
            "random_seed": RANDOM_SEED, "cov_shrinkage": COV_SHRINKAGE, "cal_fraction": CAL_FRACTION,
        },
        "results": phase_results,
        "pooled_pre_calibration_accuracy": pooled_pre_cal,
        "pooled_post_calibration_accuracy": pooled_post_cal,
        "decisions_md_verdict": verdict,
        "reference_numbers": {
            "fdrift_early_late_pre_cal": FDRIFT_PRE_CAL_REFERENCE,
            "real_label_mixed_parity_pre_cal": REAL_LABEL_PRE_CAL_REFERENCE,
            "supported_threshold": SUPPORTED_THRESHOLD,
            "rejected_threshold": REJECTED_THRESHOLD,
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
    for p in PHASES:
        r = phase_results[p]
        assert 0.0 <= r["pre_calibration_accuracy_mean"] <= 1.0, (
            f"[C3 PLAUSIBILITY FAIL] phase={p} pre_cal mean outside [0,1]"
        )
        assert 0.0 <= r["post_calibration_accuracy_mean"] <= 1.0, (
            f"[C3 PLAUSIBILITY FAIL] phase={p} post_cal mean outside [0,1]"
        )
        assert len(r["fold_results"]) == 29, (
            f"[C3 PLAUSIBILITY FAIL] phase={p} expected 29 folds, got {len(r['fold_results'])}"
        )
    assert 0.0 <= pooled_pre_cal <= 1.0 and 0.0 <= pooled_post_cal <= 1.0
    log.info(f"  [C3] plausibility: 3 phases x 29 folds = 87 folds total, all accuracies in [0,1] -- OK")

    return {
        "early_pre_cal": phase_results["early"]["pre_calibration_accuracy_mean"],
        "early_post_cal": phase_results["early"]["post_calibration_accuracy_mean"],
        "mid_pre_cal": phase_results["mid"]["pre_calibration_accuracy_mean"],
        "mid_post_cal": phase_results["mid"]["post_calibration_accuracy_mean"],
        "late_pre_cal": phase_results["late"]["pre_calibration_accuracy_mean"],
        "late_post_cal": phase_results["late"]["post_calibration_accuracy_mean"],
        "pooled_pre_cal": pooled_pre_cal,
        "pooled_post_cal": pooled_post_cal,
        "decisions_md_verdict": verdict,
        "output_path": OUTPUT_JSON,
    }


@app.local_entrypoint()
def main():
    print("F-DRIFT-D — phase-matched block contrast (drift-reset hypothesis)")
    print("Label = real block/class identity, restricted per phase (early/mid/late within-block "
          "position third) so both blocks are compared at MATCHED within-block position.")
    print("Identical calibrated pipeline, single seed=42, full 29-fold LOSO per phase (87 folds total).")
    print(f"Pre-registered rule (DECISIONS.md): pooled pre_cal < {SUPPORTED_THRESHOLD} -> SUPPORTED "
          f"(drift resets at the break) | >= {REJECTED_THRESHOLD} -> REJECTED | in between -> inconclusive.\n")
    results = run_drift_control_d.remote()
    print("\nF-DRIFT-D RESULTS:")
    for k, v in results.items():
        print(f"  {k:<24}: {v}")
