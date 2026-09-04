# =============================================================================
# command to run:
#   modal run run_nested_cv_selection.py::pilot
#   modal run run_nested_cv_selection.py::full
#   modal run run_nested_cv_selection.py::full --inner-cv loso --ea-mode per-subject
# run_nested_cv_selection.py
#
# F8 — NESTED-CV HYPERPARAMETER SELECTION FOR PCA n_components / LOGREG C / SVM C
#
# WHY THIS SCRIPT EXISTS:
#   AUDIT.md's Q5 hyperparameter-leakage review (Fix-ID F8) found that three
#   hyperparameters used throughout the LOSO drivers were never validated by
#   ANY method -- they are asserted defaults, hand-picked once and reused
#   everywhere: PCA_MAX_COMPONENTS=35 and LOGREG_C=1.0 in
#   run_step4_matched_spatial_control.py / run_step4_condition4_asymmetric_mamba.py,
#   and SVC_C=1.0 in run_baselines_mdm_tssvm_tslda_csp.py's TS-SVM baseline.
#   (The ONE hyperparameter that WAS already being selected -- the
#   calibration shrinkage-blend weight, SHRINK_GRID -- was reviewed in the
#   same Q5 pass and judged safe, because it is chosen via an internal
#   3-fold stratified CV on the calibration split D_cal only, never touching
#   D_test; that mechanism is untouched here and reused verbatim.)
#
#   This script closes that gap: for each LOSO outer fold, it runs a
#   SUBJECT-DISJOINT inner cross-validation over the 28-subject training
#   pool ONLY (never touching the held-out test subject) to select
#   (pca_n_components, logreg_C) jointly for the spatial tangent-space
#   pipeline, and svc_C independently for the TS-SVM baseline. The held-out
#   subject is then scored exactly once, with whatever hyperparameters the
#   inner CV selected -- so no test-subject information can leak into
#   hyperparameter selection (the exact failure mode Q5 was checking for).
#
#   Inner folds are GROUPED BY SUBJECT (GroupKFold / LeaveOneGroupOut over
#   the 28 training-pool subjects' IDs), never by trial -- a naive
#   trial-level inner K-fold would let a subject's trials appear in both the
#   inner-train and inner-validation splits, which is exactly the kind of
#   subject-level leakage F-LEAK's entire monkeypatch-based verifier exists
#   to catch elsewhere in this codebase. Using trial-level splits here would
#   silently reintroduce that same leakage one level down.
#
#   Per AUDIT.md's own Fix-ID table note ("Inner-LOSO or inner-5-fold
#   hyperparameter search... Largest single cost item -- schedule last"),
#   both inner-CV granularities are offered via `--inner-cv`:
#     - "5fold" (default): GroupKFold(n_splits=5) over the 28 training-pool
#       subjects -- 5 inner fits per grid point per outer fold, tractable.
#     - "loso":  LeaveOneGroupOut over the 28 training-pool subjects -- the
#       gold-standard inner-LOSO AUDIT.md names as the alternative, 28 inner
#       fits per grid point per outer fold (~5.6x the cost of "5fold").
#
#   This is a validation/diagnostic script, not a new reported accuracy
#   condition -- it does NOT append to the shared results/loso_runs.csv
#   (which is reserved for the actual Search-vs-Memorize conditions being
#   compared in F13's statistics). Instead it answers a narrower question:
#   "were the asserted-default hyperparameters actually close to what an
#   honest, leakage-safe search would have picked, and does using the
#   properly-selected values change the headline accuracy?" Output is its
#   own JSON artifact with, per outer fold, the selected hyperparameters and
#   resulting held-out accuracy, plus grid-wide selection-frequency counts
#   (to show whether the defaults were near-consensus or noisy/unstable
#   across folds) and a grand-mean-accuracy comparison against the
#   fixed-default reference numbers.
#
#   Single seed only (RANDOM_SEED=42, not F4's 5-seed loop): this script's
#   purpose is hyperparameter-selection VALIDATION, not a reportable
#   headline number competing in F13's statistics -- running it 5x under
#   different seeds would multiply an already "largest single cost item" by
#   5x for no analytical benefit (the question being answered, "are the
#   defaults close to nested-CV-selected values", does not need repeated
#   seeds to answer, unlike a between-condition accuracy comparison).
#
#   Built with F3 (`--ea-mode`), F9 (`--cov-estimator`), and F14 (extended
#   metrics + permutation null) from the start, reusing
#   run_step4_matched_spatial_control.py's spatial pipeline (EA whitening,
#   trial_covariances, tangent_vectorize, StandardScaler->PCA->shrinkage
#   calibration) and run_baselines_mdm_tssvm_tslda_csp.py's TS-SVM arm
#   (StandardScaler + linear SVC on tangent vectors, zero-shot, no
#   calibration split) verbatim except for the newly-searched hyperparameters.
#
# Usage: modal run run_nested_cv_selection.py::pilot
#        modal run run_nested_cv_selection.py::full --inner-cv loso
# =============================================================================

import modal

app    = modal.App("bci-f8-nested-cv-selection")
volume = modal.Volume.from_name("eeg-data-vol")

RAW_DATA_PATH = "/data/processed_eeg_all_subjects.npz"
OUTPUT_JSON   = "/data/results_nested_cv_selection.json"
VOLUME_PATH   = "/data"

# Reference numbers (fixed-default pipelines this script is validating against)
MATCHED_SPATIAL_CONTROL_FULL_MEAN_ACC = 0.6779   # PCA=35, logreg_C=1.0 fixed
TSSVM_ZEROSHOT_FULL_MEAN_ACC          = None      # filled in by user once F5 has run; left None if unknown

SFREQ, N_CHANNELS = 250, 62

RANDOM_SEED         = 42                        # single seed -- see header for rationale
CAL_FRACTION        = 0.15
COV_SHRINKAGE_FIXED = 0.1                        # F9: "fixed" cov-estimator mode
N_PERMUTATIONS      = 1000                       # F14: permutation-derived empirical chance level

# --- F8 nested-search grids (the three "asserted defaults" Q5 flagged) ---
PCA_GRID       = [10, 20, 35, 50, 65]
LOGREG_C_GRID  = [0.01, 0.1, 1.0, 10.0, 100.0]
SVC_C_GRID     = [0.01, 0.1, 1.0, 10.0, 100.0]

# --- Calibration-shrinkage blend (unchanged from matched_spatial_control.py;
# already validated safe by AUDIT.md Q5 -- reused verbatim, not re-searched) ---
SHRINK_GRID     = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS = 3

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy<2", "scikit-learn==1.4.2", "scipy")
    .add_local_python_source("eeg_alignment")
)


@app.function(image=image, cpu=8.0, volumes={VOLUME_PATH: volume}, timeout=86400, memory=16384)
def run_nested_cv_selection(pilot=True, pilot_n_folds=5, inner_cv="5fold",
                             ea_mode="pooled", cov_estimator="fixed"):

    import os
    import time
    import json
    import logging
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GroupKFold, LeaveOneGroupOut
    from sklearn.covariance import LedoitWolf
    from sklearn.metrics import (
        confusion_matrix, f1_score, roc_auc_score, cohen_kappa_score, balanced_accuracy_score,
    )
    import eeg_alignment as ea

    assert ea_mode in ("none", "pooled", "per-subject", "riemannian"), f"Unknown --ea-mode: {ea_mode!r}"
    assert cov_estimator in ("fixed", "lwf"), f"Unknown --cov-estimator: {cov_estimator!r}"
    assert inner_cv in ("5fold", "loso"), f"Unknown --inner-cv: {inner_cv!r}"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("f8-nested-cv")

    raw = np.load(RAW_DATA_PATH, allow_pickle=True)
    X_np = raw["X"].astype(np.float32)
    y_np = raw["y"].astype(np.int64)
    subjects_np = raw["subjects"]
    N, C, T = X_np.shape
    N_CLASSES = int(y_np.max()) + 1
    assert C == N_CHANNELS and N_CLASSES == 2
    log.info(f"X: {X_np.shape} | pilot={pilot} | inner_cv={inner_cv} | ea_mode={ea_mode} | cov_estimator={cov_estimator}")
    if not pilot:
        # F-SILENT hardening -- see other drivers for rationale.
        assert len(np.unique(subjects_np)) == 29, (
            f"Expected exactly 29 subjects (30 - sub-09 exclusion per AUDIT.md D2), "
            f"got {len(np.unique(subjects_np))}: {sorted(np.unique(subjects_np).tolist())}"
        )

    all_subjects = sorted(np.unique(subjects_np).tolist())
    outer_subjects = all_subjects[:pilot_n_folds] if pilot else all_subjects
    log.info(f"Outer (LOSO) folds: {outer_subjects}")

    # =========================================================================
    # F9: covariance estimator dispatch (identical to other drivers)
    # =========================================================================
    def compute_trial_covariances(X):
        if cov_estimator == "fixed":
            covs = ea.trial_covariances(X, shrinkage=COV_SHRINKAGE_FIXED)
            return covs, float(COV_SHRINKAGE_FIXED)
        Nn, Cc, Tt = X.shape
        covs = np.empty((Nn, Cc, Cc), dtype=np.float64)
        lambdas = np.empty(Nn, dtype=np.float64)
        for i in range(Nn):
            lw = LedoitWolf().fit(X[i].T)
            covs[i] = lw.covariance_
            lambdas[i] = lw.shrinkage_
        return covs.astype(np.float32), float(lambdas.mean())

    # =========================================================================
    # F14: extended metrics + permutation-derived empirical chance level
    # =========================================================================
    def compute_extended_metrics(y_true, y_pred, y_score):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        try:
            auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else float("nan")
        except ValueError:
            auc = float("nan")
        return {
            "sensitivity": float(sens), "specificity": float(spec),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
            "roc_auc": float(auc),
        }

    def permutation_null_stats(y_true, y_pred, seed, n_perm=N_PERMUTATIONS):
        rng = np.random.RandomState(seed)
        y_true = np.asarray(y_true)
        observed_acc = float((y_pred == y_true).mean())
        null_accs = np.empty(n_perm)
        for i in range(n_perm):
            null_accs[i] = (y_pred == rng.permutation(y_true)).mean()
        p_value = (np.sum(null_accs >= observed_acc) + 1) / (n_perm + 1)
        return {
            "permutation_empirical_chance_level": float(null_accs.mean()),
            "permutation_null_std_acc": float(null_accs.std()),
            "permutation_p_value": float(p_value),
        }

    def linear_predict_scores(coef, intercept, X):
        return (X @ coef.T + intercept).ravel()

    # =========================================================================
    # Calibration-shrinkage blend -- UNCHANGED from matched_spatial_control.py
    # (already validated safe by AUDIT.md Q5; not re-searched here). Takes
    # logreg_C as a parameter now (it used to be the fixed module constant).
    # =========================================================================
    def fit_shrinkage_classifier(X_train_pca, y_train, X_cal_pca, y_cal, logreg_C, seed):
        global_clf = LogisticRegression(C=logreg_C, max_iter=5000, random_state=seed).fit(X_train_pca, y_train)
        local_clf_full = LogisticRegression(C=logreg_C, max_iter=5000, random_state=seed).fit(X_cal_pca, y_cal)
        n_splits = max(min(SHRINK_CV_FOLDS, np.bincount(y_cal).min()), 2)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        shrink_scores = {s: [] for s in SHRINK_GRID}
        for tr_idx, val_idx in skf.split(X_cal_pca, y_cal):
            X_tr, X_val, y_tr, y_val = X_cal_pca[tr_idx], X_cal_pca[val_idx], y_cal[tr_idx], y_cal[val_idx]
            if len(np.unique(y_tr)) < 2:
                continue
            local_fold = LogisticRegression(C=logreg_C, max_iter=5000, random_state=seed).fit(X_tr, y_tr)
            for shrink in SHRINK_GRID:
                coef_b = shrink * local_fold.coef_ + (1 - shrink) * global_clf.coef_
                icpt_b = shrink * local_fold.intercept_ + (1 - shrink) * global_clf.intercept_
                preds = (linear_predict_scores(coef_b, icpt_b, X_val) > 0).astype(int)
                shrink_scores[shrink].append((preds == y_val).mean())
        mean_scores = {s: (np.mean(v) if v else -1.0) for s, v in shrink_scores.items()}
        best_shrink = max(mean_scores, key=mean_scores.get)
        coef_final = best_shrink * local_clf_full.coef_ + (1 - best_shrink) * global_clf.coef_
        icpt_final = best_shrink * local_clf_full.intercept_ + (1 - best_shrink) * global_clf.intercept_
        return coef_final, icpt_final, best_shrink, global_clf

    # =========================================================================
    # F8 core: subject-disjoint inner CV, run ONCE per outer fold on the
    # 28-subject training pool ONLY. Returns the grid point(s) with the
    # highest mean inner-validation accuracy.
    # =========================================================================
    def inner_splitter(subs_train28):
        if inner_cv == "loso":
            return list(LeaveOneGroupOut().split(np.zeros(len(subs_train28)), groups=subs_train28))
        n_splits = min(5, len(np.unique(subs_train28)))
        return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(subs_train28)), groups=subs_train28))

    def select_pca_logreg_C(tan_train28, y_train28, subs_train28, seed):
        splits = inner_splitter(subs_train28)
        scores = {(p, c): [] for p in PCA_GRID for c in LOGREG_C_GRID}
        for tr_idx, val_idx in splits:
            X_tr, X_val = tan_train28[tr_idx], tan_train28[val_idx]
            y_tr, y_val = y_train28[tr_idx], y_train28[val_idx]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                continue
            scaler = StandardScaler().fit(X_tr)
            X_tr_z, X_val_z = scaler.transform(X_tr), scaler.transform(X_val)
            for p in PCA_GRID:
                n_comp = min(p, X_tr_z.shape[1] - 1, X_tr_z.shape[0] - 1)
                pca = PCA(n_components=n_comp, random_state=seed).fit(X_tr_z)
                X_tr_pca, X_val_pca = pca.transform(X_tr_z), pca.transform(X_val_z)
                for c in LOGREG_C_GRID:
                    clf = LogisticRegression(C=c, max_iter=5000, random_state=seed).fit(X_tr_pca, y_tr)
                    scores[(p, c)].append((clf.predict(X_val_pca) == y_val).mean())
        mean_scores = {k: (np.mean(v) if v else -1.0) for k, v in scores.items()}
        best_p, best_c = max(mean_scores, key=mean_scores.get)
        return best_p, best_c, mean_scores

    def select_svc_C(tan_train28, y_train28, subs_train28, seed):
        splits = inner_splitter(subs_train28)
        scores = {c: [] for c in SVC_C_GRID}
        for tr_idx, val_idx in splits:
            X_tr, X_val = tan_train28[tr_idx], tan_train28[val_idx]
            y_tr, y_val = y_train28[tr_idx], y_train28[val_idx]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                continue
            scaler = StandardScaler().fit(X_tr)
            X_tr_z, X_val_z = scaler.transform(X_tr), scaler.transform(X_val)
            for c in SVC_C_GRID:
                svm = SVC(kernel="linear", C=c).fit(X_tr_z, y_tr)
                scores[c].append((svm.predict(X_val_z) == y_val).mean())
        mean_scores = {k: (np.mean(v) if v else -1.0) for k, v in scores.items()}
        best_c = max(mean_scores, key=mean_scores.get)
        return best_c, mean_scores

    # =========================================================================
    # F3 diagnostics: same label-free assertion + subject-decodability check
    # as other LOSO drivers, computed once up front.
    # =========================================================================
    mu_all = X_np.mean(axis=(0, 2), keepdims=True)
    sd_all = X_np.std(axis=(0, 2), keepdims=True) + 1e-6
    X_all_z = ((X_np - mu_all) / sd_all).astype(np.float32)
    covs_all_pre, _ = compute_trial_covariances(X_all_z)
    y_decoy_a = np.random.RandomState(0).randint(0, 2, size=len(subjects_np))
    y_decoy_b = 1 - y_decoy_a
    ea.verify_label_free(covs_all_pre, subjects_np, mode=ea_mode, y_decoy_a=y_decoy_a, y_decoy_b=y_decoy_b)
    log.info("  [F3] verify_label_free PASSED (alignment provably ignores labels)")

    # =========================================================================
    # F8 outer loop: single seed, LOSO over `outer_subjects`. Each outer fold
    # runs its OWN inner CV (subject-disjoint, training-pool-only) to select
    # hyperparameters, then scores the held-out subject exactly once.
    # =========================================================================
    seed = RANDOM_SEED
    np.random.seed(seed)

    spatial_fold_records, svm_fold_records = [], []
    selected_pca_counts = {p: 0 for p in PCA_GRID}
    selected_logreg_c_counts = {c: 0 for c in LOGREG_C_GRID}
    selected_svc_c_counts = {c: 0 for c in SVC_C_GRID}

    for fold_idx, test_sub in enumerate(outer_subjects):
        fold_start = time.time()
        log.info(f"\n{'='*70}\n  OUTER FOLD {fold_idx+1}/{len(outer_subjects)} — sub-{test_sub}\n{'='*70}")

        is_holdout = subjects_np == test_sub
        X_train28, y_train28, subs_train28 = X_np[~is_holdout], y_np[~is_holdout], subjects_np[~is_holdout]
        X_k, y_k = X_np[is_holdout], y_np[is_holdout]

        mu = X_train28.mean(axis=(0, 2), keepdims=True)
        sd = X_train28.std(axis=(0, 2), keepdims=True) + 1e-6
        X_train28_z = ((X_train28 - mu) / sd).astype(np.float32)
        X_k_z = ((X_k - mu) / sd).astype(np.float32)

        covs_train28, _ = compute_trial_covariances(X_train28_z)
        covs_k, _ = compute_trial_covariances(X_k_z)

        W_train = ea.euclidean_align(covs_train28, subs_train28, mode=ea_mode)
        if ea_mode in ("none", "pooled"):
            X_train28_aligned = ea.apply_ea_whitening_signal(X_train28_z, W_train)
            X_k_aligned = ea.apply_ea_whitening_signal(X_k_z, W_train)
        else:
            X_train28_aligned = ea.apply_ea_whitening_signal_per_subject(X_train28_z, subs_train28, W_train)
            subs_k = np.full(len(X_k_z), test_sub)
            W_k = ea.euclidean_align(covs_k, subs_k, mode=ea_mode)
            X_k_aligned = ea.apply_ea_whitening_signal_per_subject(X_k_z, subs_k, W_k)

        covs_train28_aligned, _ = compute_trial_covariances(X_train28_aligned)
        covs_k_aligned, _ = compute_trial_covariances(X_k_aligned)

        tan_train28 = ea.tangent_vectorize(covs_train28_aligned)
        tan_k = ea.tangent_vectorize(covs_k_aligned)
        tangent_dim = tan_train28.shape[1]

        # --- Inner CV selection (subject-disjoint, training-pool only) ---
        inner_start = time.time()
        best_p, best_c_logreg, pca_logreg_scores = select_pca_logreg_C(tan_train28, y_train28, subs_train28, seed)
        best_c_svc, svc_scores = select_svc_C(tan_train28, y_train28, subs_train28, seed)
        log.info(
            f"  [Inner CV, {inner_cv}] selected pca_n_components={best_p} logreg_C={best_c_logreg} "
            f"svc_C={best_c_svc}  (inner search took {time.time()-inner_start:.0f}s)"
        )
        selected_pca_counts[best_p] += 1
        selected_logreg_c_counts[best_c_logreg] += 1
        selected_svc_c_counts[best_c_svc] += 1

        # --- Spatial (tangent -> PCA -> shrinkage-calibrated logreg) arm,
        # using the inner-CV-selected (pca_n_components, logreg_C) ---
        sss = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - CAL_FRACTION), random_state=seed)
        cal_idx, test_idx = next(sss.split(tan_k, y_k))
        feat_cal, y_cal = tan_k[cal_idx], y_k[cal_idx]
        feat_test, y_test = tan_k[test_idx], y_k[test_idx]

        scaler = StandardScaler()
        feat_train28_z = scaler.fit_transform(tan_train28)
        feat_cal_z = scaler.transform(feat_cal)
        feat_test_z = scaler.transform(feat_test)

        n_components = min(best_p, feat_train28_z.shape[1] - 1, feat_train28_z.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=seed)
        X_train28_pca = pca.fit_transform(feat_train28_z)
        X_cal_pca = pca.transform(feat_cal_z)
        X_test_pca = pca.transform(feat_test_z)

        coef_final, icpt_final, best_shrink, global_clf = fit_shrinkage_classifier(
            X_train28_pca, y_train28, X_cal_pca, y_cal, best_c_logreg, seed)
        pre_cal_acc = float((global_clf.predict(X_test_pca) == y_test).mean())
        final_scores = linear_predict_scores(coef_final, icpt_final, X_test_pca)
        final_preds = (final_scores > 0).astype(int)
        spatial_acc = float((final_preds == y_test).mean())
        spatial_metrics = compute_extended_metrics(y_test, final_preds, final_scores)
        spatial_perm = permutation_null_stats(y_test, final_preds, seed=seed)

        spatial_fold_records.append({
            "fold_index": fold_idx, "test_subject": str(test_sub),
            "selected_pca_n_components": int(best_p), "selected_logreg_C": float(best_c_logreg),
            "inner_cv_val_accuracy": float(pca_logreg_scores[(best_p, best_c_logreg)]),
            "tangent_dim": int(tangent_dim),
            "pre_calibration_acc": pre_cal_acc, "post_calibration_acc": spatial_acc,
            "best_shrink_weight": float(best_shrink),
            **spatial_metrics, **spatial_perm,
        })

        # --- TS-SVM arm (zero-shot, no calibration split -- matches F5's
        # convention), using the inner-CV-selected svc_C ---
        scaler_svm = StandardScaler().fit(tan_train28)
        svm = SVC(kernel="linear", C=best_c_svc).fit(scaler_svm.transform(tan_train28), y_train28)
        svm_preds = svm.predict(scaler_svm.transform(tan_k))
        svm_scores = svm.decision_function(scaler_svm.transform(tan_k))
        svm_acc = float((svm_preds == y_k).mean())
        svm_metrics = compute_extended_metrics(y_k, svm_preds, svm_scores)
        svm_perm = permutation_null_stats(y_k, svm_preds, seed=seed)

        svm_fold_records.append({
            "fold_index": fold_idx, "test_subject": str(test_sub),
            "selected_svc_C": float(best_c_svc),
            "inner_cv_val_accuracy": float(svc_scores[best_c_svc]),
            "zeroshot_acc": svm_acc,
            **svm_metrics, **svm_perm,
        })

        log.info(
            f"  RESULT -> spatial post_cal={spatial_acc:.4f} (pca={best_p}, C={best_c_logreg}) "
            f"| TS-SVM zero-shot={svm_acc:.4f} (C={best_c_svc}) "
            f"| fold elapsed: {time.time()-fold_start:.0f}s"
        )

    spatial_grand_mean = float(np.mean([r["post_calibration_acc"] for r in spatial_fold_records]))
    svm_grand_mean = float(np.mean([r["zeroshot_acc"] for r in svm_fold_records]))

    log.info(f"\n{'='*70}\n  F8 NESTED-CV SELECTION — {len(outer_subjects)} outer folds ({'PILOT' if pilot else 'FULL'})\n{'='*70}")
    log.info(f"  Spatial (nested-CV-selected PCA/C)   grand mean acc: {spatial_grand_mean:.4f}")
    log.info(f"  TS-SVM  (nested-CV-selected C)        grand mean acc: {svm_grand_mean:.4f}")
    log.info(f"  Reference — matched_spatial_control (fixed PCA=35, C=1.0): {MATCHED_SPATIAL_CONTROL_FULL_MEAN_ACC:.4f}")
    log.info(f"  PCA n_components selection frequency: {selected_pca_counts}")
    log.info(f"  logreg_C selection frequency:          {selected_logreg_c_counts}")
    log.info(f"  svc_C selection frequency:             {selected_svc_c_counts}")

    results_payload = {
        "description": "F8 -- leakage-safe (subject-disjoint) nested-CV selection of PCA n_components, "
                        "logreg_C, and svc_C, previously asserted defaults never validated by any method.",
        "pilot": pilot, "inner_cv": inner_cv, "ea_mode": ea_mode, "cov_estimator": cov_estimator,
        "seed": seed, "n_outer_folds": len(outer_subjects),
        "grids": {"pca_n_components": PCA_GRID, "logreg_C": LOGREG_C_GRID, "svc_C": SVC_C_GRID},
        "spatial_arm": {
            "grand_mean_accuracy": spatial_grand_mean,
            "fold_results": spatial_fold_records,
            "selected_pca_n_components_frequency": {str(k): v for k, v in selected_pca_counts.items()},
            "selected_logreg_C_frequency": {str(k): v for k, v in selected_logreg_c_counts.items()},
        },
        "tssvm_arm": {
            "grand_mean_accuracy": svm_grand_mean,
            "fold_results": svm_fold_records,
            "selected_svc_C_frequency": {str(k): v for k, v in selected_svc_c_counts.items()},
        },
        "reference_matched_spatial_control_fixed_hparams_full_mean_acc": MATCHED_SPATIAL_CONTROL_FULL_MEAN_ACC,
        "reference_tssvm_zeroshot_full_mean_acc": TSSVM_ZEROSHOT_FULL_MEAN_ACC,
    }
    if not pilot:
        with open(OUTPUT_JSON, "w") as f:
            json.dump(results_payload, f, indent=2)
        volume.commit()
        log.info(f"  Saved: {OUTPUT_JSON}")

    # =========================================================================
    # C3 plausibility assertions -- printed next to the numbers, not silent.
    # Run AFTER the write above so a failing assertion never suppresses the
    # diagnostic artifact.
    # =========================================================================
    assert 0.0 <= spatial_grand_mean <= 1.0, f"[C3 PLAUSIBILITY FAIL] spatial grand mean {spatial_grand_mean} outside [0,1]"
    assert 0.0 <= svm_grand_mean <= 1.0, f"[C3 PLAUSIBILITY FAIL] TS-SVM grand mean {svm_grand_mean} outside [0,1]"
    if not pilot:
        assert len(outer_subjects) == 29, f"[C3 PLAUSIBILITY FAIL] expected 29 outer folds, got {len(outer_subjects)}"
    log.info(f"  [C3] plausibility: {len(outer_subjects)} outer folds, spatial/TS-SVM grand means in [0,1] -- OK")

    return {
        "spatial_grand_mean_accuracy": spatial_grand_mean,
        "tssvm_grand_mean_accuracy": svm_grand_mean,
        "n_outer_folds": len(outer_subjects),
        "selected_pca_n_components_frequency": selected_pca_counts,
        "selected_logreg_C_frequency": selected_logreg_c_counts,
        "selected_svc_C_frequency": selected_svc_c_counts,
    }


@app.local_entrypoint(name="pilot")
def pilot_entrypoint(pilot_n_folds: int = 5, inner_cv: str = "5fold",
                      ea_mode: str = "pooled", cov_estimator: str = "fixed"):
    print("F8 — NESTED-CV HYPERPARAMETER SELECTION (PILOT)")
    print(f"pilot_n_folds={pilot_n_folds}  inner_cv={inner_cv}  ea_mode={ea_mode}  cov_estimator={cov_estimator}\n")
    results = run_nested_cv_selection.remote(
        pilot=True, pilot_n_folds=pilot_n_folds, inner_cv=inner_cv, ea_mode=ea_mode, cov_estimator=cov_estimator)
    print("\nPILOT RESULTS:")
    for k, v in results.items():
        print(f"  {k:<36}: {v}")


@app.local_entrypoint(name="full")
def full_entrypoint(inner_cv: str = "5fold", ea_mode: str = "pooled", cov_estimator: str = "fixed"):
    print("F8 — NESTED-CV HYPERPARAMETER SELECTION (FULL, all 29 LOSO folds)")
    print(f"inner_cv={inner_cv}  ea_mode={ea_mode}  cov_estimator={cov_estimator}")
    print("WARNING: this is AUDIT.md's own flagged 'largest single cost item' -- "
          "expect a long run, especially with --inner-cv loso.\n")
    results = run_nested_cv_selection.remote(
        pilot=False, inner_cv=inner_cv, ea_mode=ea_mode, cov_estimator=cov_estimator)
    print("\nFULL RESULTS:")
    for k, v in results.items():
        print(f"  {k:<36}: {v}")
