# =============================================================================
# command to run:
#   modal run run_step4_drift_control_b_interleaved.py::main
# run_step4_drift_control_b_interleaved.py
#
# F-DRIFT-B — INTERLEAVED (ODD/EVEN) PSEUDO-LABEL CONTROL (new fix-ID, added
# 2026-08-19, AUDIT.md fix register; DECISIONS.md pre-registered rule)
#
# WHY THIS SCRIPT EXISTS:
#   F-DRIFT's ACCEPTED result (DECISIONS.md, RESULTS_LEDGER.md): a pseudo-
#   label carrying ZERO task information (early-half vs. late-half of the
#   SAME real class block) reaches post-calibration accuracy of 0.7112,
#   essentially matching the real Search-vs-Memorize classifier's 0.7078.
#   The pre-registered >0.65 "drift detector" branch fired -- the primary
#   contrast does not survive as a task-decoding result.
#
#   F-DRIFT's early/late split still has SOME temporal separation (up to
#   half a block's worth of trials between the two pseudo-classes' centers).
#   This script (F-DRIFT-B) removes that separation almost entirely: the
#   pseudo-label is ODD-numbered vs. EVEN-numbered trials (by within-class
#   chronological position), so adjacent trials always carry different
#   pseudo-labels and the two pseudo-classes are uniformly interleaved
#   throughout the block, not concentrated in an early half vs. a late half.
#
#   This isolates whether F-DRIFT's result is SPECIFICALLY driven by
#   temporal separation (if so, F-DRIFT-B should collapse to chance, since
#   there is essentially no temporal separation left to exploit) or whether
#   it reflects some other artifact of the pseudo-labeling/calibration
#   mechanism unrelated to time (if so, F-DRIFT-B should ALSO be high, and
#   DECISIONS.md's pre-registered rule requires a full re-examination).
#
# TEST DESIGN: structurally IDENTICAL to run_step4_drift_control.py's
#   pseudo-label tests -- same EA/tangent/PCA/shrinkage-calibration code,
#   same seed=42, same full 29-fold LOSO, same 15% calibration mechanism,
#   same per-real-class (Search-only / Memorize-only) separation so both
#   pseudo-classes are drawn from the SAME task in each test. The ONLY
#   difference is the pseudo-label assignment: odd/even interleaving
#   instead of early/late midpoint split.
#
# PRE-REGISTERED VERDICT RULE (DECISIONS.md, fixed BEFORE this script runs):
#   interleaved acc < 0.55 (chance) while F-DRIFT sits at ~0.71
#       -> confirms the effect is SPECIFICALLY temporal separation; this
#          pair becomes the central evidence for the drift-detector finding.
#   interleaved acc > 0.65
#       -> the mechanism is NOT drift; HALT and report, every prior
#          interpretation needs re-examination.
#   in between (0.55-0.65)
#       -> report both, do not interpret further without discussion.
#
# Usage: modal run run_step4_drift_control_b_interleaved.py::main
# =============================================================================

import modal

app    = modal.App("bci-condition4-drift-control-b-interleaved")
volume = modal.Volume.from_name("eeg-data-vol")

RAW_DATA_PATH = "/data/processed_eeg_all_subjects.npz"
OUTPUT_JSON   = "/data/results_condition4_drift_control_b_interleaved.json"
VOLUME_PATH   = "/data"

SFREQ, N_CHANNELS = 250, 62

RANDOM_SEED         = 42
COV_SHRINKAGE       = 0.1
PCA_MAX_COMPONENTS  = 35
LOGREG_C            = 1.0
CAL_FRACTION        = 0.15
SHRINK_GRID         = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS     = 3

# DECISIONS.md pre-registered F-DRIFT-B rule (recorded 2026-08-19, BEFORE
# this script runs): applied to POST-calibration mean interleaved-pseudo
# accuracy, and compared against F-DRIFT's already-accepted 0.7112.
INTERLEAVED_CHANCE_THRESHOLD  = 0.55   # < this -> confirms temporal-separation-specific effect
INTERLEAVED_NOT_DRIFT_THRESHOLD = 0.65 # > this -> mechanism is NOT drift, HALT and re-examine everything
FDRIFT_REFERENCE_POST_CAL = 0.7112     # F-DRIFT's accepted mean pseudo post-cal accuracy, for the table
FDRIFT_REFERENCE_PRE_CAL  = 0.6418
REAL_LABEL_REFERENCE_POST_CAL = 0.7078
REAL_LABEL_REFERENCE_PRE_CAL  = 0.5201

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy<2", "scikit-learn==1.4.2", "scipy")
)


@app.function(image=image, cpu=4.0, volumes={VOLUME_PATH: volume}, timeout=86400, memory=16384)
def run_drift_control_b():

    import logging, time, math, json
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
    from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("condition4-drift-control-b")

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
    # RIEMANNIAN / EA UTILITIES -- IDENTICAL to run_step4_drift_control.py's
    # pre-F3 pooled-only EA (family consistency with the rest of Batch 1).
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
    # ONE INTERLEAVED PSEUDO-LABEL TEST, restricted to ONE real class's
    # trials so both pseudo-classes are drawn from the SAME task -- full
    # 29-fold LOSO with the IDENTICAL calibrated pipeline (single seed=42).
    # =========================================================================
    def run_drift_test_b(real_class, real_class_name):
        log.info(f"\n{'='*70}\n  F-DRIFT-B TEST: interleaved (odd/even) pseudo-label within real "
                 f"class {real_class} ({real_class_name})\n{'='*70}")

        # Pseudo-label = odd/even by WITHIN-CLASS chronological position --
        # adjacent trials always differ in pseudo-label; zero net temporal
        # separation between the two pseudo-classes (uniformly interleaved).
        idx_list, y_pseudo_list, subj_list = [], [], []
        for sub in unique_subjects:
            sub_idx = np.where(subjects_np == sub)[0]
            cls_idx = sub_idx[y_np[sub_idx] == real_class]
            n = len(cls_idx)
            assert n >= 2, f"sub-{sub} has only {n} trials of class {real_class} -- cannot interleave-split."
            pseudo = (np.arange(n) % 2).astype(np.int64)   # 0,1,0,1,... by chronological position
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

    result_search = run_drift_test_b(0, "search_only")
    result_memorize = run_drift_test_b(1, "memorize_only")

    mean_post_cal = float(np.mean([
        result_search["post_calibration_accuracy_mean"], result_memorize["post_calibration_accuracy_mean"],
    ]))
    mean_pre_cal = float(np.mean([
        result_search["pre_calibration_accuracy_mean"], result_memorize["pre_calibration_accuracy_mean"],
    ]))

    # =========================================================================
    # DECISIONS.md's pre-registered F-DRIFT-B verdict.
    # =========================================================================
    if mean_post_cal < INTERLEAVED_CHANCE_THRESHOLD:
        verdict = ("CONFIRMS the effect is specifically driven by TEMPORAL SEPARATION, not some other "
                   f"pseudo-labeling artifact (interleaved accuracy {mean_post_cal:.4f} < "
                   f"{INTERLEAVED_CHANCE_THRESHOLD}, vs. F-DRIFT's early/late accuracy "
                   f"{FDRIFT_REFERENCE_POST_CAL:.4f}). This pair becomes the central evidence for the "
                   "drift-detector conclusion.")
    elif mean_post_cal > INTERLEAVED_NOT_DRIFT_THRESHOLD:
        verdict = ("The mechanism is NOT drift -- an interleaved split has no meaningful temporal-"
                   f"separation signal to exploit, yet interleaved accuracy ({mean_post_cal:.4f}) is "
                   f"still > {INTERLEAVED_NOT_DRIFT_THRESHOLD}. HALT AND REPORT: every prior "
                   "interpretation in this project needs re-examination.")
    else:
        verdict = (f"In between ({INTERLEAVED_CHANCE_THRESHOLD} <= {mean_post_cal:.4f} <= "
                   f"{INTERLEAVED_NOT_DRIFT_THRESHOLD}) -- report both, do not interpret further "
                   "without discussion.")

    log.info(f"\n{'='*70}\n  F-DRIFT-B SUMMARY\n{'='*70}")
    log.info(f"  search_only  : pre_cal={result_search['pre_calibration_accuracy_mean']:.4f} "
             f"post_cal={result_search['post_calibration_accuracy_mean']:.4f}")
    log.info(f"  memorize_only: pre_cal={result_memorize['pre_calibration_accuracy_mean']:.4f} "
             f"post_cal={result_memorize['post_calibration_accuracy_mean']:.4f}")
    log.info(f"  MEAN pre_cal={mean_pre_cal:.4f}  MEAN post_cal={mean_post_cal:.4f}")
    log.info(f"  COMBINED TABLE (F-DRIFT-B vs. F-DRIFT vs. real-label, all accepted 2026-08-19 numbers):")
    log.info(f"    interleaved (F-DRIFT-B) : pre_cal={mean_pre_cal:.4f}  post_cal={mean_post_cal:.4f}")
    log.info(f"    early/late  (F-DRIFT)   : pre_cal={FDRIFT_REFERENCE_PRE_CAL:.4f}  post_cal={FDRIFT_REFERENCE_POST_CAL:.4f}")
    log.info(f"    real label  (F-DRIFT)   : pre_cal={REAL_LABEL_REFERENCE_PRE_CAL:.4f}  post_cal={REAL_LABEL_REFERENCE_POST_CAL:.4f}")
    log.info(f"  DECISIONS.md VERDICT: {verdict}")

    results_payload = {
        "condition": "F-DRIFT-B — interleaved (odd/even) pseudo-label control: pseudo-label alternates "
                      "every trial WITHIN the SAME real class block (near-zero temporal separation "
                      "between pseudo-classes), identical calibrated pipeline, single seed=42, "
                      "full 29-fold LOSO",
        "hyperparameters": {
            "pca_max_components": PCA_MAX_COMPONENTS, "logreg_C": LOGREG_C,
            "random_seed": RANDOM_SEED, "cov_shrinkage": COV_SHRINKAGE, "cal_fraction": CAL_FRACTION,
        },
        "drift_tests": {"search_only": result_search, "memorize_only": result_memorize},
        "mean_pre_calibration_accuracy": mean_pre_cal,
        "mean_post_calibration_accuracy": mean_post_cal,
        "decisions_md_verdict": verdict,
        "decisions_md_thresholds": {
            "confirms_temporal_below": INTERLEAVED_CHANCE_THRESHOLD,
            "not_drift_above": INTERLEAVED_NOT_DRIFT_THRESHOLD,
        },
        "combined_table_vs_fdrift": {
            "interleaved_fdrift_b":   {"pre_cal": mean_pre_cal, "post_cal": mean_post_cal},
            "early_late_fdrift":      {"pre_cal": FDRIFT_REFERENCE_PRE_CAL, "post_cal": FDRIFT_REFERENCE_POST_CAL},
            "real_label_fdrift":      {"pre_cal": REAL_LABEL_REFERENCE_PRE_CAL, "post_cal": REAL_LABEL_REFERENCE_POST_CAL},
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
        # Interleaving should give a near-exact 50/50 split (off by at most 1 per subject,
        # summed over 29 subjects) -- a large imbalance would indicate the modulo-2 logic broke.
        n0, n1 = res["pseudo_class_balance"]
        assert abs(n0 - n1) <= 29, (
            f"[C3 PLAUSIBILITY FAIL] {res['real_class_name']} interleaved pseudo-class balance "
            f"({n0} vs {n1}) is too skewed for a per-subject odd/even split across 29 subjects"
        )
    log.info(f"  [C3] plausibility: 2 interleaved drift tests x 29 folds, accuracies in [0,1], "
             f"pseudo-classes both non-empty and near-balanced -- OK")

    return {
        "mean_pre_calibration_accuracy": mean_pre_cal,
        "mean_post_calibration_accuracy": mean_post_cal,
        "search_only_post_cal": result_search["post_calibration_accuracy_mean"],
        "memorize_only_post_cal": result_memorize["post_calibration_accuracy_mean"],
        "decisions_md_verdict": verdict,
        "output_path": OUTPUT_JSON,
    }


@app.local_entrypoint()
def main():
    print("F-DRIFT-B — Interleaved (odd/even) pseudo-label control")
    print("Pseudo-label alternates every trial WITHIN the SAME real class block (near-zero temporal separation).")
    print("Run through the identical calibrated LOSO pipeline, single seed=42, full 29 folds, x2 (Search-only, Memorize-only).")
    print(f"Pre-registered verdict thresholds (DECISIONS.md): <{INTERLEAVED_CHANCE_THRESHOLD} confirms "
          f"temporal-separation-specific | >{INTERLEAVED_NOT_DRIFT_THRESHOLD} NOT drift, halt and re-examine "
          f"| in between = report both, no further interpretation.")
    print(f"Reference (F-DRIFT, accepted): early/late post_cal={FDRIFT_REFERENCE_POST_CAL}, "
          f"real-label post_cal={REAL_LABEL_REFERENCE_POST_CAL}\n")
    results = run_drift_control_b.remote()
    print("\nF-DRIFT-B RESULTS:")
    for k, v in results.items():
        print(f"  {k:<32}: {v}")
