# =============================================================================
# command to run (AFTER run_data_engine_granular_on_modal.py::main has
# completed and committed processed_eeg_all_subjects_granular.npz):
#   modal run run_r1b_r2_r3_composition_runs.py::main
# run_r1b_r2_r3_composition_runs.py
#
# R1(b) / R2 / R3 — RESULTS_LEDGER.md L037/L038 GLOBAL INVALIDATION NOTICE
# follow-up. Pre-registered in DECISIONS.md's "R1 / R2 / R3 pre-registration"
# section, BEFORE this script runs.
#
# THREE RUNS, THEN STOP (explicit instruction — do not run V2 after this;
# do not touch PAPER/draft.md or PAPER/outline.md).
#
# R1(b) — DIRECT ENCODE-VS-TEST DECODE
#   y = 0 if code in {search_encode, memorize_encode} else 1 (test-phase,
#   any item type). ALL epochs used (~400/subject). This class split is
#   NATURALLY IMBALANCED (~100 encode vs ~300 test per subject, ~25/75) —
#   declared_imbalanced_design=True, balanced accuracy / AUC is primary.
#   PRE-REGISTERED JOINT CRITERION (fixed before running): IF R1(a)'s
#   composition-gap arithmetic (RESULTS_LEDGER.md L038; already run
#   locally, 13/13 mappings fit) AND this decode's post_cal balanced
#   accuracy > 0.85, THEN the unified composition explanation is CONFIRMED
#   IN FULL and the drift/rest-break-discontinuity mechanistic account
#   (F-DRIFT through F-DRIFT-G) is WITHDRAWN.
#
# R2 — THE CORRECTLY-DEFINED SEARCH-VS-MEMORIZE CONTRAST
#   Only codes {search_encode, memorize_encode} used (50/task/subject,
#   2,900 total) — the one epoch class with an unambiguous, uncontaminated,
#   one-per-behavioral-trial Search/Memorize label. y = 0 (search) / 1
#   (memorize), exactly balanced by construction. NO PASS/FAIL THRESHOLD IS
#   PRE-REGISTERED — report it whatever it shows, per explicit instruction.
#   This is the only version of the Search-vs-Memorize contrast in this
#   codebase whose name means what it says.
#
# R3 — LURE-REMOVED CONTRAST
#   The nominal 200-epoch/class contrast with the 75 New-Lure epochs/class
#   dropped (125/class/subject: 50 encode + 50 target + 25 distractor).
#   y = 0 (search: search_encode/target/distractor) / 1 (memorize:
#   memorize_encode/target/distractor), exactly balanced by construction.
#   Descriptive — quantifies how much of the original 70.78% was
#   contributed by epochs (New-Lure) carrying no encoding-condition
#   information at all.
#
# Reused verbatim from the rest of the pre-F3 F-DRIFT family: EA whitening,
# tangent-space vectorization, shrinkage calibration, C3 balance hardening.
# Single seed=42, full LOSO (fold count = subjects present in the granular
# dataset, expected 29 per AUDIT.md D2 / sub-09's known upstream failure).
#
# Usage: modal run run_r1b_r2_r3_composition_runs.py::main
# =============================================================================

import modal

app    = modal.App("bci-r1b-r2-r3-composition-runs")
volume = modal.Volume.from_name("eeg-data-vol")

GRANULAR_DATA_PATH = "/data/processed_eeg_all_subjects_granular.npz"
OUTPUT_JSON         = "/data/results_r1b_r2_r3_composition_runs.json"
VOLUME_PATH         = "/data"

SFREQ, N_CHANNELS = 250, 62

RANDOM_SEED         = 42
COV_SHRINKAGE       = 0.1
PCA_MAX_COMPONENTS  = 35
LOGREG_C            = 1.0
CAL_FRACTION        = 0.15
SHRINK_GRID         = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS     = 3

# code -> (task, phase) per run_data_engine_granular_on_modal.py's mapping
SEARCH_ENCODE, SEARCH_TARGET, SEARCH_DISTRACTOR, SEARCH_LURE = 0, 1, 2, 3
MEMORIZE_ENCODE, MEMORIZE_TARGET, MEMORIZE_DISTRACTOR, MEMORIZE_LURE = 4, 5, 6, 7
ENCODE_CODES = {SEARCH_ENCODE, MEMORIZE_ENCODE}
LURE_CODES   = {SEARCH_LURE, MEMORIZE_LURE}

# DECISIONS.md pre-registered R1(b) joint-criterion threshold (fixed before running).
R1B_POST_CAL_BALANCED_CONFIRM_THRESHOLD = 0.85

# GATE C6 STEP 1: Image.env() shown NOT to hold for OMP/MKL/OPENBLAS -- see
# run_c4_high_res_shuffled_label_control.py's image comment for the full
# account. Fixed via threadpoolctl + cpu 4.0 -> 1.0. NOTE (GATE C6 STEP 3):
# despite this bug being live during the actual re-run of this script, R2's
# pre_cal_balanced, R3's pre_cal_balanced, and R1b's post_cal_balanced all
# reproduced bit-for-bit exactly across a genuine cross-architecture change
# (Haswell-class -> Skylake-X/AVX512 host) -- see GATE C6 report for the
# per-fold diff. Only post-calibration values (which run a shrinkage-grid
# model-selection step) moved, by exactly one trial each. Pinned here
# anyway so that reproducibility is guaranteed rather than merely observed.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scikit-learn==1.4.2", "scipy==1.13.1", "threadpoolctl==3.6.0")
    .env({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    })
)


@app.function(image=image, cpu=1.0, volumes={VOLUME_PATH: volume}, timeout=86400, memory=16384)
def run_r1b_r2_r3(git_short_hash: str = "nogit"):

    import logging, time, math, json, os
    import numpy as np
    import scipy
    import sklearn
    from threadpoolctl import threadpool_limits, threadpool_info
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
    from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, roc_auc_score

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log_versions = logging.getLogger("r1b-r2-r3-versions")
    log_versions.info(f"numpy={np.__version__} scipy={scipy.__version__} sklearn={sklearn.__version__}")
    log_versions.info("thread env vars (informational only, not proof of pin -- see GATE C6): " + ", ".join(
        f"{k}={os.environ.get(k)}" for k in
        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    ))
    # GATE C7 STEP 1b: smoke test only -- real protection is the `with` block
    # around _core() below. See run_c4's identical comment.
    log_versions.info(f"threadpool_info BEFORE threadpoolctl pin (smoke test): {threadpool_info()}")
    with threadpool_limits(limits=1):
        log_versions.info(f"threadpool_info INSIDE threadpoolctl pin (smoke test, every entry must "
                           f"show num_threads=1): {threadpool_info()}")

    _npcfg = np.show_config(mode="dicts")
    _simd = _npcfg.get("SIMD Extensions", {})
    _blas = _npcfg.get("Build Dependencies", {}).get("blas", {})
    hardware_info = {
        "numpy_version": np.__version__, "scipy_version": scipy.__version__, "sklearn_version": sklearn.__version__,
        "simd_found": _simd.get("found", []), "simd_not_found": _simd.get("not found", []),
        "blas_name": _blas.get("name"), "blas_version": _blas.get("version"),
        "openblas_configuration": _blas.get("openblas configuration"),
        "threadpool_info_after_pin": threadpool_info(),
    }
    log_versions.info(f"hardware_info: {hardware_info}")
    log = logging.getLogger("r1b-r2-r3")

    def _core():
        np.random.seed(RANDOM_SEED)

        raw = np.load(GRANULAR_DATA_PATH, allow_pickle=True)
        X_np = raw["X"].astype(np.float32)
        code_np = raw["code"].astype(np.int64)
        subjects_np = raw["subjects"]
        N, C, T = X_np.shape
        assert C == N_CHANNELS
        unique_subjects = sorted(np.unique(subjects_np).tolist())
        n_subjects = len(unique_subjects)
        log.info(f"X: {X_np.shape} | Subjects: {n_subjects} {unique_subjects}")
        if n_subjects != 29:
            log.warning(f"Expected 29 subjects (30 - sub-09 exclusion per AUDIT.md D2), got {n_subjects}. "
                         "Proceeding with what is present — not fatal, but flagged.")

        # =========================================================================
        # RIEMANNIAN / EA / CLASSIFIER UTILITIES -- IDENTICAL to the rest of the
        # pre-F3 F-DRIFT family (run_step4_drift_control_g_real_labels_onset_excluded.py).
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
                    "run does not declare an imbalanced design."
                )
            return {"class_balance": balance, "majority_class_rate": majority_rate,
                    "accuracy_minus_majority_rate": lift, "balanced_accuracy": bal_acc}

        def linear_predict(coef, intercept, X):
            return ((X @ coef.T + intercept).ravel() > 0).astype(int)

        def linear_decision_score(coef, intercept, X):
            return (X @ coef.T + intercept).ravel()

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
        # Generic LOSO runner, reused for all three (R1b/R2/R3) label sets.
        # =========================================================================
        def run_loso(X_all, y_all, subjects_all, run_name, declared_imbalanced_design):
            folds, pre_accs, post_accs, pre_bal, post_bal, pre_aucs, post_aucs = [], [], [], [], [], [], []
            for fold_idx, test_sub in enumerate(unique_subjects):
                fold_start = time.time()
                is_holdout = subjects_all == test_sub
                X_train, y_train = X_all[~is_holdout], y_all[~is_holdout]
                X_k, y_k = X_all[is_holdout], y_all[is_holdout]
                if len(np.unique(y_train)) < 2 or len(np.unique(y_k)) < 2:
                    log.warning(f"  [{run_name}] fold sub-{test_sub}: missing a class, skipping fold")
                    continue

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
                if len(np.unique(y_cal)) < 2 or len(np.unique(y_test)) < 2:
                    log.warning(f"  [{run_name}] fold sub-{test_sub}: cal/test split missing a class, skipping fold")
                    continue

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
                pre_cal_scores = global_clf.decision_function(X_test_pca)
                pre_cal_acc = float((pre_cal_preds == y_test).mean())
                final_preds = linear_predict(coef_final, icpt_final, X_test_pca)
                final_scores = linear_decision_score(coef_final, icpt_final, X_test_pca)
                post_cal_acc = float((final_preds == y_test).mean())
                metrics = compute_binary_metrics(y_test, final_preds)

                # N16 instrumentation: explicit per-class test-set N and correct/total,
                # for both pre- and post-calibration predictions, derived from the
                # same confusion matrices already computed -- so C3/Table-S1-style
                # claims are traceable to a persisted per-fold artifact, not just
                # a single accuracy scalar.
                def per_class_counts(y_true, y_pred):
                    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                    tn, fp, fn, tp = cm.ravel()
                    n0, n1 = int(tn + fp), int(fn + tp)
                    return {
                        "class0_correct": int(tn), "class0_total": n0,
                        "class1_correct": int(tp), "class1_total": n1,
                    }
                pre_cal_per_class = per_class_counts(y_test, pre_cal_preds)
                post_cal_per_class = per_class_counts(y_test, final_preds)

                pre_cal_plaus = c3_balance_check(y_test, pre_cal_preds, pre_cal_acc,
                                                  f"[{run_name}] fold sub-{test_sub} pre_cal",
                                                  declared_imbalanced_design)
                post_cal_plaus = c3_balance_check(y_test, final_preds, post_cal_acc,
                                                   f"[{run_name}] fold sub-{test_sub} post_cal",
                                                   declared_imbalanced_design)

                pre_auc = float(roc_auc_score(y_test, pre_cal_scores)) if len(np.unique(y_test)) == 2 else float("nan")
                post_auc = float(roc_auc_score(y_test, final_scores)) if len(np.unique(y_test)) == 2 else float("nan")

                log.info(f"  [{run_name}] fold {fold_idx+1}/{n_subjects} sub-{test_sub} n_test={len(y_test)} -> "
                         f"pre_cal={pre_cal_acc:.4f}(bal={pre_cal_plaus['balanced_accuracy']:.4f},auc={pre_auc:.4f}) "
                         f"post_cal={post_cal_acc:.4f}(bal={post_cal_plaus['balanced_accuracy']:.4f},auc={post_auc:.4f}) "
                         f"shrink={best_shrink:.2f} [{time.time()-fold_start:.0f}s]")

                folds.append({
                    "fold_index": fold_idx, "test_subject": str(test_sub), "n_test": int(len(y_test)),
                    "n_train": int(len(y_train)), "tangent_dim": int(tangent_dim),
                    "best_shrink_weight": float(best_shrink),
                    "pre_calibration_acc": pre_cal_acc, "post_calibration_acc": post_cal_acc,
                    "pre_calibration_auc": pre_auc, "post_calibration_auc": post_auc,
                    "pre_cal_plausibility": pre_cal_plaus, "post_cal_plausibility": post_cal_plaus,
                    "pre_cal_per_class_counts": pre_cal_per_class,
                    "post_cal_per_class_counts": post_cal_per_class,
                    **metrics,
                })
                pre_accs.append(pre_cal_acc); post_accs.append(post_cal_acc)
                pre_bal.append(pre_cal_plaus["balanced_accuracy"]); post_bal.append(post_cal_plaus["balanced_accuracy"])
                pre_aucs.append(pre_auc); post_aucs.append(post_auc)

            summary = {
                "run_name": run_name, "n_folds": len(folds),
                "pre_calibration_accuracy_mean": float(np.mean(pre_accs)) if pre_accs else float("nan"),
                "post_calibration_accuracy_mean": float(np.mean(post_accs)) if post_accs else float("nan"),
                "pre_calibration_balanced_accuracy_mean": float(np.mean(pre_bal)) if pre_bal else float("nan"),
                "post_calibration_balanced_accuracy_mean": float(np.mean(post_bal)) if post_bal else float("nan"),
                "pre_calibration_auc_mean": float(np.mean(pre_aucs)) if pre_aucs else float("nan"),
                "post_calibration_auc_mean": float(np.mean(post_aucs)) if post_aucs else float("nan"),
                "fold_results": folds,
            }
            return summary

        # =========================================================================
        # R1(b) — encode vs. test, ALL epochs, naturally imbalanced (~25/75).
        # =========================================================================
        y_r1b = np.where(np.isin(code_np, list(ENCODE_CODES)), 0, 1).astype(np.int64)
        r1b_class_balance = np.bincount(y_r1b).tolist()
        log.info(f"R1(b) encode-vs-test dataset: N={len(y_r1b)}, class balance={r1b_class_balance} "
                 f"(expected N=11,600 across 29 subjects at uniform 400/subject post sub-01 exclusion: "
                 f"encode 2,900 + test-phase 8,700)")
        assert len(y_r1b) == 11600, (
            f"[R1b GATE-C ASSERTION FAILED] expected N=11600 (29 subjects x 400/subject, uniform post-exclusion), "
            f"got N={len(y_r1b)}. Halt and investigate before trusting this run."
        )
        summary_r1b = run_loso(X_np, y_r1b, subjects_np, "R1b_encode_vs_test", declared_imbalanced_design=True)

        # =========================================================================
        # R2 — the correctly-defined Search-vs-Memorize contrast: encode-only.
        # =========================================================================
        mask_r2 = np.isin(code_np, [SEARCH_ENCODE, MEMORIZE_ENCODE])
        X_r2, code_r2, subjects_r2 = X_np[mask_r2], code_np[mask_r2], subjects_np[mask_r2]
        y_r2 = np.where(code_r2 == SEARCH_ENCODE, 0, 1).astype(np.int64)
        r2_class_balance = np.bincount(y_r2).tolist()
        log.info(f"R2 encode-only dataset: N={len(y_r2)}, class balance={r2_class_balance} "
                 f"(expected 2,900 total, 1,450/class, 50/task/subject -- sub-01's 10 unlogged practice-trial "
                 f"epochs already excluded at the epoching layer, see run_data_engine_granular_on_modal.py)")
        assert len(y_r2) == 2900 and r2_class_balance == [1450, 1450], (
            f"[R2 GATE-C ASSERTION FAILED] expected N=2900, class balance [1450, 1450]; got N={len(y_r2)}, "
            f"class balance={r2_class_balance}. The granular npz does not reflect the sub-01 exclusion -- "
            "halt and investigate before trusting any R2/R1b/R3/C1/C3/C4/D2 number."
        )
        summary_r2 = run_loso(X_r2, y_r2, subjects_r2, "R2_search_vs_memorize_encode_only",
                               declared_imbalanced_design=False)

        # =========================================================================
        # R3 — lure-removed contrast: 125 epochs/class/subject.
        # =========================================================================
        mask_r3 = ~np.isin(code_np, list(LURE_CODES))
        X_r3, code_r3, subjects_r3 = X_np[mask_r3], code_np[mask_r3], subjects_np[mask_r3]
        # mask_r3 already dropped codes 3 and 7 (New-Lure); remaining codes 0-2 are
        # Search-side (encode/target/distractor), 4-6 are Memorize-side.
        y_r3 = np.where(code_r3 < 4, 0, 1).astype(np.int64)
        r3_class_balance = np.bincount(y_r3).tolist()
        log.info(f"R3 lure-removed dataset: N={len(y_r3)}, class balance={r3_class_balance} "
                 f"(expected 125/class/subject, uniform post sub-01 exclusion: 7,250 total, 3,625/class)")
        assert len(y_r3) == 7250 and r3_class_balance == [3625, 3625], (
            f"[R3 GATE-C ASSERTION FAILED] expected N=7250, class balance [3625, 3625] (125/class/subject x 29 "
            f"subjects); got N={len(y_r3)}, class balance={r3_class_balance}. Halt and investigate."
        )
        summary_r3 = run_loso(X_r3, y_r3, subjects_r3, "R3_lure_removed", declared_imbalanced_design=False)

        # =========================================================================
        # R1(b) pre-registered joint criterion, applied mechanically.
        # =========================================================================
        r1b_post_bal = summary_r1b["post_calibration_balanced_accuracy_mean"]
        if r1b_post_bal > R1B_POST_CAL_BALANCED_CONFIRM_THRESHOLD:
            r1b_verdict = (f"CONFIRMED IN FULL — R1(b) post_cal_balanced ({r1b_post_bal:.4f}) > "
                            f"{R1B_POST_CAL_BALANCED_CONFIRM_THRESHOLD}, and R1(a)'s composition-gap arithmetic "
                            "already fits 13/13 checked mappings (RESULTS_LEDGER.md L038). The unified "
                            "composition explanation is CONFIRMED IN FULL; the drift/rest-break-discontinuity "
                            "mechanistic account (F-DRIFT through F-DRIFT-G) is WITHDRAWN.")
        else:
            r1b_verdict = (f"NOT CONFIRMED BY THIS THRESHOLD — R1(b) post_cal_balanced ({r1b_post_bal:.4f}) <= "
                            f"{R1B_POST_CAL_BALANCED_CONFIRM_THRESHOLD}. Report as-is; do not force the joint "
                            "criterion's conclusion. R1(a)'s composition-gap fit (L038) stands on its own "
                            "regardless of this threshold's outcome.")
        log.info(f"\n{'='*70}\n  R1(b) PRE-REGISTERED VERDICT\n{'='*70}\n  {r1b_verdict}")

        results_payload = {
            "hyperparameters": {
                "pca_max_components": PCA_MAX_COMPONENTS, "logreg_C": LOGREG_C, "random_seed": RANDOM_SEED,
                "cov_shrinkage": COV_SHRINKAGE, "cal_fraction": CAL_FRACTION,
            },
            "n_subjects": n_subjects, "subjects": unique_subjects,
            "R1b_encode_vs_test": summary_r1b,
            "R1b_verdict": r1b_verdict,
            "R2_search_vs_memorize_encode_only": summary_r2,
            "R3_lure_removed": summary_r3,
            "hardware_info": hardware_info,
        }
        # GATE C5 STEP 3: non-overwriting artifact. This is the headline
        # pipeline -- every downstream artifact (C1, C3, C4, D2, Table S1, every
        # figure) inherits from this single execution. See run_c1's identical
        # comment for why Modal Volumes' lack of version history makes this
        # mandatory, not optional, here above all.
        import datetime
        utc_ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        stamped_path = OUTPUT_JSON.replace(".json", f"_{utc_ts}_{git_short_hash}.json")
        with open(stamped_path, "w") as f:
            json.dump(results_payload, f, indent=2)
        with open(OUTPUT_JSON, "w") as f:
            json.dump(results_payload, f, indent=2)
        volume.commit()
        log.info(f"Saved (immutable): {stamped_path}")
        log.info(f"Saved (convenience pointer, overwritten): {OUTPUT_JSON}")

        # C3 plausibility assertions -- after the write, so a failing assertion
        # never suppresses the diagnostic artifact.
        for s in (summary_r1b, summary_r2, summary_r3):
            assert s["n_folds"] > 0, f"[C3 PLAUSIBILITY FAIL] {s['run_name']}: zero usable folds"
            assert 0.0 <= s["post_calibration_accuracy_mean"] <= 1.0
        log.info("[C3] plausibility: all three runs produced >=1 usable fold, accuracies in [0,1] -- OK")

        return {
            "R1b_post_cal_balanced": r1b_post_bal, "R1b_verdict": r1b_verdict,
            "R2_pre_cal_balanced": summary_r2["pre_calibration_balanced_accuracy_mean"],
            "R2_post_cal_balanced": summary_r2["post_calibration_balanced_accuracy_mean"],
            "R3_pre_cal_balanced": summary_r3["pre_calibration_balanced_accuracy_mean"],
            "R3_post_cal_balanced": summary_r3["post_calibration_balanced_accuracy_mean"],
            "output_path": OUTPUT_JSON,
            "stamped_output_path": stamped_path,
        }

    # GATE C7 STEP 1b: real `with` block around the numerical work --
    # see run_c4_high_res_shuffled_label_control.py's identical comment.
    with threadpool_limits(limits=1):
        log_versions.info(f"threadpool_info INSIDE with-block (right before the "
                          f"numerical pipeline runs): {threadpool_info()}")
        return _core()


@app.local_entrypoint()
def main():
    import subprocess, os
    try:
        git_short_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
        ).decode().strip()
    except Exception:
        git_short_hash = "nogit"
    print("R1(b) / R2 / R3 — encode-vs-test decode, encode-only Search-vs-Memorize, lure-removed contrast")
    print("Pre-registered in DECISIONS.md BEFORE this run. Requires processed_eeg_all_subjects_granular.npz "
          "(run run_data_engine_granular_on_modal.py::main first if it does not exist yet).")
    print(f"R1(b) joint-criterion threshold (post_cal_balanced): > {R1B_POST_CAL_BALANCED_CONFIRM_THRESHOLD} "
          "-> unified explanation CONFIRMED IN FULL, drift account WITHDRAWN.")
    print("R2/R3: no pass/fail threshold pre-registered — report whatever they show.")
    print(f"git_short_hash for this run's stamped output filename: {git_short_hash}\n")
    results = run_r1b_r2_r3.remote(git_short_hash=git_short_hash)
    print("\nRESULTS:")
    for k, v in results.items():
        print(f"  {k}: {v}")
