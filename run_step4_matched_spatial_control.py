# =============================================================================
# command to run:
#   modal run run_step4_matched_spatial_control.py::main
#   modal run run_step4_matched_spatial_control.py::main --ea-mode per-subject --cov-estimator lwf
# run_step4_matched_spatial_control.py
#
# CONDITION 4 — MATCHED SPATIAL-ONLY CONTROL (Audit Fix, Missing Piece #1)
#
# WHY THIS SCRIPT EXISTS:
#   The brutal-audit review of run_step4_condition4_asymmetric_mamba.py found
#   that the official Condition 4v2 baseline (67.79%, tangent_features_ea.npz
#   -> run_step4_condition4_subspace_calibration.py) and the new Asymmetric
#   Fusion result (71.28%) do NOT share the same spatial feature pipeline:
#     - Condition 4v2: per-subject EA + a GLOBAL Riemannian Frechet-mean
#       tangent-space reference fit across all 29 subjects at once (a
#       self-disclosed "soft population-level leak" in
#       run_step2_alignment_on_modal.py, lines 158-163).
#     - Asymmetric Fusion: EA fit on the 28-subject training pool only
#       (applied uniformly to train28 + held-out), tangent vectors computed
#       with an IDENTITY reference (no Frechet mean, no leak).
#   The Step 3 -> Step 4 "+3.49pp" delta was therefore confounded: part of
#   it could be attributable to the pipeline change, not to adding Mamba.
#
#   This script is the fix: it is BYTE-FOR-BYTE IDENTICAL to
#   run_step4_condition4_asymmetric_mamba.py's spatial branch (same
#   covariance estimation, EA whitening, tangent_vectorize, calibration
#   utilities, hyperparameters), but with the ENTIRE Mamba temporal branch
#   deleted. The raw 1953-dim tangent vector is fed directly into
#   StandardScaler -> PCA(35) -> 15% shrinkage calibration, with nothing
#   concatenated to it. This gives a true apples-to-apples "Step 3" anchor
#   to compare the full Asymmetric Fusion result against, isolating the
#   Mamba branch's marginal value.
#
# F3/F4/F9/F14 UPGRADE (Gate-1-rejection response, 2026-08-18):
#   - F3:  EA whitening now goes through the shared eeg_alignment.py module,
#          with an explicit `--ea-mode {none,pooled,per-subject,riemannian}`
#          flag (default "pooled", preserving the original reported numbers).
#          A one-time label-free assertion and pre/post subject-decodability
#          diagnostic are logged before the LOSO loop starts.
#   - F7:  `--ea-mode none` added (identity no-op whitening) so this script
#          can serve as the "tangent-space, no-EA" cell of F7's 2x2
#          alignment/representation grid, directly comparable to
#          run_step4_eegnet_ea.py's "EEGNet, no-EA"/"EEGNet, EA" cells.
#   - F4:  the LOSO sweep now runs across 5 seeds (SEEDS below), every
#          seed x fold result is written to results/loso_runs.csv in tidy
#          long format. There is no best-run/best-epoch selection anywhere
#          in this script (this pipeline has no training epochs; the only
#          per-fold "selection" is the calibration-shrinkage weight, chosen
#          by cross-validation on the calibration split ONLY, which is
#          legitimate hyperparameter selection, not results cherry-picking).
#   - F9:  `--cov-estimator {fixed,lwf}` -- "fixed" keeps the original 0.1
#          fixed-shrinkage covariance estimator; "lwf" uses per-trial
#          Ledoit-Wolf data-driven shrinkage, with the realised lambda
#          logged per fold.
#   - F14: extended metrics (AUC, Cohen's kappa, balanced accuracy, macro-F1,
#          sensitivity, specificity) plus a permutation-derived empirical
#          chance level (1000 label permutations per fold) recorded
#          alongside every fold's accuracy.
#
# Usage: modal run run_step4_matched_spatial_control.py::main
#        modal run run_step4_matched_spatial_control.py::main --ea-mode riemannian --cov-estimator lwf
# =============================================================================

import modal

app    = modal.App("bci-condition4-matched-spatial-control")
volume = modal.Volume.from_name("eeg-data-vol")

RAW_DATA_PATH = "/data/processed_eeg_all_subjects.npz"
OUTPUT_JSON   = "/data/results_condition4_matched_spatial_control.json"
LOSO_CSV_PATH = "/data/results/loso_runs.csv"
VOLUME_PATH   = "/data"

# Reference numbers (for logging/comparison only)
CONDITION1B_FULL_MEAN_ACC    = 0.5564   # EEGNet + 15% calib, full 29-fold
CONDITION4V2_FULL_MEAN_ACC   = 0.6779   # original (leaky-tangent-space) spatial-only, full 29-fold
ASYMMETRIC_FUSION_MEAN_ACC   = 0.7128   # full 29-fold True-Mamba asymmetric fusion

SFREQ, N_CHANNELS = 250, 62

CAL_FRACTION        = 0.15
SEEDS                = [42, 43, 44, 45, 46]   # F4: 5-seed loop, no best-of-N selection
COV_SHRINKAGE_FIXED  = 0.1                     # F9: "fixed" cov-estimator mode
PCA_MAX_COMPONENTS  = 35
LOGREG_C            = 1.0
SHRINK_GRID         = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS     = 3
N_PERMUTATIONS       = 1000                    # F14: permutation-derived empirical chance level
SUBJECT_DECODABILITY_SPLITS = 5                # F3: subject-decodability diagnostic

# Canonical results/loso_runs.csv schema, SHARED verbatim with
# run_step4_condition4_asymmetric_mamba.py so both LOSO drivers append to one
# unified tidy table (order-independent: whichever driver runs first writes
# this exact header; the other driver's rows use restval="" for columns that
# don't apply to it, e.g. this script has no fusion_mode/temporal_dim).
LOSO_CSV_FIELDNAMES = [
    "script", "condition", "seed", "ea_mode", "cov_estimator", "fusion_mode",
    "fold_index", "test_subject", "tangent_dim", "temporal_dim", "fused_dim", "n_temporal_params",
    "temporal_variance_share", "realized_cov_lambda_mean", "best_shrink_weight",
    "pre_calibration_acc", "post_calibration_acc",
    "sensitivity", "specificity", "f1", "macro_f1", "balanced_accuracy", "cohen_kappa", "roc_auc",
    "permutation_empirical_chance_level", "permutation_null_std_acc", "permutation_p_value",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy<2", "scikit-learn==1.4.2", "scipy")
    .add_local_python_source("eeg_alignment")
)


@app.function(image=image, cpu=4.0, volumes={VOLUME_PATH: volume}, timeout=86400, memory=16384)
def run_matched_spatial_control(ea_mode="pooled", cov_estimator="fixed"):

    import os
    import csv
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
    from sklearn.covariance import LedoitWolf
    from sklearn.metrics import (
        confusion_matrix, f1_score, roc_auc_score, cohen_kappa_score, balanced_accuracy_score,
    )
    import logging, time, json
    import eeg_alignment as ea

    assert ea_mode in ("none", "pooled", "per-subject", "riemannian"), f"Unknown --ea-mode: {ea_mode!r}"
    assert cov_estimator in ("fixed", "lwf"), f"Unknown --cov-estimator: {cov_estimator!r}"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("condition4-matched-spatial")

    raw = np.load(RAW_DATA_PATH, allow_pickle=True)
    X_np = raw["X"].astype(np.float32)
    y_np = raw["y"].astype(np.int64)
    subjects_np = raw["subjects"]
    N, C, T = X_np.shape
    N_CLASSES = int(y_np.max()) + 1
    assert C == N_CHANNELS and N_CLASSES == 2
    log.info(f"X: {X_np.shape} | Subjects total: {len(np.unique(subjects_np))} | ea_mode={ea_mode} | cov_estimator={cov_estimator}")
    # F-SILENT hardening: a silently-failed subject upstream (e.g. sub-09's
    # truncated .eeg export, see AUDIT.md D2) must never produce a
    # clean-looking-but-incomplete LOSO run.
    assert len(np.unique(subjects_np)) == 29, (
        f"Expected exactly 29 subjects (30 - sub-09 exclusion per AUDIT.md D2), "
        f"got {len(np.unique(subjects_np))}: {sorted(np.unique(subjects_np).tolist())}"
    )

    unique_subjects = sorted(np.unique(subjects_np).tolist())
    log.info(f"Running folds: {unique_subjects}")

    # =========================================================================
    # F9: covariance estimator dispatch (fixed-shrinkage vs. Ledoit-Wolf)
    # =========================================================================
    def compute_trial_covariances(X, seed):
        if cov_estimator == "fixed":
            covs = ea.trial_covariances(X, shrinkage=COV_SHRINKAGE_FIXED)
            return covs, float(COV_SHRINKAGE_FIXED)
        # "lwf": per-trial Ledoit-Wolf data-driven shrinkage; realised lambda
        # varies per trial, we report the mean over the trials passed in.
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
            "confusion_matrix": cm.tolist(),
        }

    def permutation_null_stats(y_true, y_pred, seed, n_perm=N_PERMUTATIONS):
        """Label-permutation test: reuses the ALREADY-FIT classifier's
        predictions, shuffles the true labels n_perm times, and compares.
        This gives an empirical chance-level distribution for THIS fold's
        test set (no refitting required, so it's cheap)."""
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
                preds = (linear_predict_scores(coef_b, icpt_b, X_val) > 0).astype(int)
                shrink_scores[shrink].append((preds == y_val).mean())
        mean_scores = {s: (np.mean(v) if v else -1.0) for s, v in shrink_scores.items()}
        best_shrink = max(mean_scores, key=mean_scores.get)
        coef_final = best_shrink * local_clf_full.coef_ + (1 - best_shrink) * global_clf.coef_
        icpt_final = best_shrink * local_clf_full.intercept_ + (1 - best_shrink) * global_clf.intercept_
        return coef_final, icpt_final, best_shrink, global_clf

    # =========================================================================
    # F3 diagnostics: label-free assertion + subject-decodability, computed
    # ONCE up front on the full 29-subject pool (before any per-fold LOSO
    # split). These are sanity checks on the alignment procedure itself, not
    # part of the LOSO accuracy claim, so they run independent of `seed`.
    # =========================================================================
    mu_all = X_np.mean(axis=(0, 2), keepdims=True)
    sd_all = X_np.std(axis=(0, 2), keepdims=True) + 1e-6
    X_all_z = ((X_np - mu_all) / sd_all).astype(np.float32)
    covs_all_pre, _ = compute_trial_covariances(X_all_z, SEEDS[0])
    tan_all_pre = ea.tangent_vectorize(covs_all_pre)

    y_decoy_a = np.random.RandomState(0).randint(0, 2, size=len(subjects_np))
    y_decoy_b = 1 - y_decoy_a
    ea.verify_label_free(covs_all_pre, subjects_np, mode=ea_mode, y_decoy_a=y_decoy_a, y_decoy_b=y_decoy_b)
    log.info("  [F3] verify_label_free PASSED (alignment provably ignores labels)")

    W_all = ea.euclidean_align(covs_all_pre, subjects_np, mode=ea_mode)
    if ea_mode in ("none", "pooled"):
        X_all_aligned = ea.apply_ea_whitening_signal(X_all_z, W_all)
    else:
        X_all_aligned = ea.apply_ea_whitening_signal_per_subject(X_all_z, subjects_np, W_all)
    covs_all_post, _ = compute_trial_covariances(X_all_aligned, SEEDS[0])
    tan_all_post = ea.tangent_vectorize(covs_all_post)

    subj_decode_acc_pre, subj_decode_chance, subj_decode_n = ea.subject_decodability_accuracy(
        tan_all_pre, subjects_np, n_splits=SUBJECT_DECODABILITY_SPLITS)
    subj_decode_acc_post, _, _ = ea.subject_decodability_accuracy(
        tan_all_post, subjects_np, n_splits=SUBJECT_DECODABILITY_SPLITS)
    log.info(
        f"  [F3] Subject decodability -- pre-alignment: {subj_decode_acc_pre:.4f} | "
        f"post-alignment: {subj_decode_acc_post:.4f} | chance: {subj_decode_chance:.4f} "
        f"(n_subjects={subj_decode_n}, ea_mode={ea_mode})"
    )
    # C2 (DECISIONS.md): post-alignment subject-decodability is EXPECTED to
    # land near chance (~1/29). A value far BELOW chance is a bug signal, not
    # a success -- halt and investigate rather than silently proceeding.
    if subj_decode_acc_post < subj_decode_chance * 0.5:
        halt_diagnostic = {
            "halted_at": "C2 subject-decodability check, before the main LOSO loop",
            "ea_mode": ea_mode, "cov_estimator": cov_estimator,
            "subject_decodability_pre_alignment": subj_decode_acc_pre,
            "subject_decodability_post_alignment": subj_decode_acc_post,
            "subject_decodability_chance_level": subj_decode_chance,
            "subject_decodability_n_subjects": subj_decode_n,
        }
        halt_json_path = OUTPUT_JSON + ".c2_halt_diagnostic.json"
        with open(halt_json_path, "w") as f:
            json.dump(halt_diagnostic, f, indent=2)
        volume.commit()
        log.warning(f"  [C2 HALT] diagnostic written before raising: {halt_json_path}")
        raise AssertionError(
            f"[C2 HALT] post-alignment subject-decodability ({subj_decode_acc_post:.4f}) is "
            f"far BELOW chance ({subj_decode_chance:.4f}) -- this is a bug signal (see "
            f"DECISIONS.md C2), not a success. Investigate before proceeding; do not adjust "
            f"this threshold to match the observed value. Diagnostic written to {halt_json_path}."
        )

    # =========================================================================
    # F4: 5-seed LOSO loop — MATCHED SPATIAL-ONLY CONTROL (no Mamba branch,
    # no fusion). Every seed x fold result is recorded; results/loso_runs.csv
    # is the tidy long-format artifact, no best-run selection happens anywhere.
    # =========================================================================
    csv_rows = []
    per_seed_mean_acc = []

    for seed in SEEDS:
        np.random.seed(seed)
        fold_records, all_test_acc = [], []

        for fold_idx, test_sub in enumerate(unique_subjects):
            fold_start = time.time()
            log.info(f"\n{'='*70}\n  SEED {seed} | FOLD {fold_idx+1}/{len(unique_subjects)} — sub-{test_sub}\n{'='*70}")

            is_holdout = subjects_np == test_sub
            X_train28, y_train28, subs_train28 = X_np[~is_holdout], y_np[~is_holdout], subjects_np[~is_holdout]
            X_k, y_k = X_np[is_holdout], y_np[is_holdout]

            mu = X_train28.mean(axis=(0, 2), keepdims=True)
            sd = X_train28.std(axis=(0, 2), keepdims=True) + 1e-6
            X_train28_z = ((X_train28 - mu) / sd).astype(np.float32)
            X_k_z = ((X_k - mu) / sd).astype(np.float32)

            covs_train28, lam_train = compute_trial_covariances(X_train28_z, seed)
            covs_k, lam_k = compute_trial_covariances(X_k_z, seed)
            realized_lambda_mean = float(np.mean([lam_train, lam_k]))

            W_train = ea.euclidean_align(covs_train28, subs_train28, mode=ea_mode)
            if ea_mode in ("none", "pooled"):
                # Single pool-wide W (identity for "none"), applied uniformly.
                X_train28_aligned = ea.apply_ea_whitening_signal(X_train28_z, W_train)
                X_k_aligned = ea.apply_ea_whitening_signal(X_k_z, W_train)
            else:
                # per-subject / riemannian: each of the 28 training subjects
                # gets its own W from its own trials, AND the held-out
                # subject gets its OWN unsupervised W from its OWN (unlabeled)
                # trials -- legitimate because EA never touches y.
                X_train28_aligned = ea.apply_ea_whitening_signal_per_subject(X_train28_z, subs_train28, W_train)
                subs_k = np.full(len(X_k_z), test_sub)
                W_k = ea.euclidean_align(covs_k, subs_k, mode=ea_mode)
                X_k_aligned = ea.apply_ea_whitening_signal_per_subject(X_k_z, subs_k, W_k)

            covs_train28_aligned, _ = compute_trial_covariances(X_train28_aligned, seed)
            covs_k_aligned, _ = compute_trial_covariances(X_k_aligned, seed)

            # --- SPATIAL ONLY: raw tangent vector, untouched by any nn.Module, NO Mamba ---
            tan_train28 = ea.tangent_vectorize(covs_train28_aligned)
            tan_k = ea.tangent_vectorize(covs_k_aligned)
            tangent_dim = tan_train28.shape[1]

            # --- No fusion: feature vector IS the raw tangent vector ---
            feat_train28 = tan_train28
            feat_k = tan_k

            sss = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - CAL_FRACTION), random_state=seed)
            cal_idx, test_idx = next(sss.split(feat_k, y_k))
            feat_cal, y_cal = feat_k[cal_idx], y_k[cal_idx]
            feat_test, y_test = feat_k[test_idx], y_k[test_idx]

            scaler = StandardScaler()
            feat_train28_z = scaler.fit_transform(feat_train28)
            feat_cal_z = scaler.transform(feat_cal)
            feat_test_z = scaler.transform(feat_test)

            n_components = min(PCA_MAX_COMPONENTS, feat_train28_z.shape[1] - 1, feat_train28_z.shape[0] - 1)
            pca = PCA(n_components=n_components, random_state=seed)
            X_train28_pca = pca.fit_transform(feat_train28_z)
            X_cal_pca = pca.transform(feat_cal_z)
            X_test_pca = pca.transform(feat_test_z)

            coef_final, icpt_final, best_shrink, global_clf = fit_shrinkage_classifier(
                X_train28_pca, y_train28, X_cal_pca, y_cal, seed)
            pre_cal_acc = float((global_clf.predict(X_test_pca) == y_test).mean())
            final_scores = linear_predict_scores(coef_final, icpt_final, X_test_pca)
            final_preds = (final_scores > 0).astype(int)
            best_test_acc = float((final_preds == y_test).mean())
            metrics = compute_extended_metrics(y_test, final_preds, final_scores)
            perm_stats = permutation_null_stats(y_test, final_preds, seed=seed)

            log.info(
                f"  RESULT -> pre_cal={pre_cal_acc:.4f}  post_cal={best_test_acc:.4f} (shrink={best_shrink:.2f}) "
                f"| auc={metrics['roc_auc']:.4f} kappa={metrics['cohen_kappa']:.4f} "
                f"| perm_chance={perm_stats['permutation_empirical_chance_level']:.4f} p={perm_stats['permutation_p_value']:.4f} "
                f"| tangent_dim={tangent_dim} | realized_lambda={realized_lambda_mean:.4f}"
            )

            row = {
                "script": "run_step4_matched_spatial_control",
                "condition": "matched_spatial_control",
                "seed": seed, "ea_mode": ea_mode, "cov_estimator": cov_estimator,
                "fold_index": fold_idx, "test_subject": str(test_sub),
                "tangent_dim": int(tangent_dim),
                "realized_cov_lambda_mean": realized_lambda_mean,
                "best_shrink_weight": float(best_shrink),
                "pre_calibration_acc": pre_cal_acc,
                "post_calibration_acc": best_test_acc,
                "sensitivity": metrics["sensitivity"], "specificity": metrics["specificity"],
                "f1": metrics["f1"], "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"], "cohen_kappa": metrics["cohen_kappa"],
                "roc_auc": metrics["roc_auc"],
                "permutation_empirical_chance_level": perm_stats["permutation_empirical_chance_level"],
                "permutation_null_std_acc": perm_stats["permutation_null_std_acc"],
                "permutation_p_value": perm_stats["permutation_p_value"],
            }
            csv_rows.append(row)
            fold_records.append({**row, "confusion_matrix": metrics["confusion_matrix"]})
            all_test_acc.append(best_test_acc)
            log.info(f"  Fold elapsed: {time.time()-fold_start:.0f}s")

        seed_mean_acc = float(np.mean(all_test_acc))
        per_seed_mean_acc.append(seed_mean_acc)
        log.info(f"\nSEED {seed} DONE — mean acc {seed_mean_acc:.4f} over {len(unique_subjects)} folds")

    grand_mean_acc = float(np.mean(per_seed_mean_acc))
    grand_std_acc = float(np.std(per_seed_mean_acc))

    log.info(f"\n{'='*70}\n  MATCHED SPATIAL-ONLY CONTROL — {len(SEEDS)} seeds x {len(unique_subjects)} folds (FULL)\n{'='*70}")
    log.info(f"  Grand mean-of-seed-means Acc: {grand_mean_acc:.4f} ± {grand_std_acc:.4f} (across seeds, no cherry-picking)")
    log.info(f"  Per-seed means: {per_seed_mean_acc}")
    log.info(
        "\n  COMPARISON (reference numbers, original pooled-EA pipeline):\n"
        f"    EEGNet + 15% Calib (full 29-fold)               : {CONDITION1B_FULL_MEAN_ACC*100:.2f}%\n"
        f"    Condition 4v2 spatial-only, ORIGINAL/leaky pipe  : {CONDITION4V2_FULL_MEAN_ACC*100:.2f}%\n"
        f"    THIS RUN — matched spatial-only control          : {grand_mean_acc*100:.2f}% (ea_mode={ea_mode}, cov_estimator={cov_estimator})\n"
        f"    Asymmetric Fusion (raw tangent + True-Mamba)     : {ASYMMETRIC_FUSION_MEAN_ACC*100:.2f}%\n"
    )

    # F4: write the tidy long-format CSV (one row per seed x fold), appending
    # to the SAME file/schema used by run_step4_condition4_asymmetric_mamba.py.
    os.makedirs(os.path.dirname(LOSO_CSV_PATH), exist_ok=True)
    write_header = not os.path.exists(LOSO_CSV_PATH)
    with open(LOSO_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOSO_CSV_FIELDNAMES, restval="")
        if write_header:
            writer.writeheader()
        writer.writerows(csv_rows)

    results_payload = {
        "condition": "Condition 4 — MATCHED SPATIAL-ONLY CONTROL (identity-reference tangent, no Mamba)",
        "ea_mode": ea_mode, "cov_estimator": cov_estimator,
        "seeds": SEEDS, "n_folds": len(unique_subjects),
        "hyperparameters": {
            "cal_fraction": CAL_FRACTION, "pca_max_components": PCA_MAX_COMPONENTS,
            "logreg_C": LOGREG_C, "shrink_grid": SHRINK_GRID, "shrink_cv_folds": SHRINK_CV_FOLDS,
            "cov_shrinkage_fixed": COV_SHRINKAGE_FIXED, "n_permutations": N_PERMUTATIONS,
        },
        "f3_diagnostics": {
            "verify_label_free_passed": True,
            "subject_decodability_pre_alignment": subj_decode_acc_pre,
            "subject_decodability_post_alignment": subj_decode_acc_post,
            "subject_decodability_chance_level": subj_decode_chance,
            "subject_decodability_n_subjects": subj_decode_n,
        },
        "per_seed_mean_accuracy": per_seed_mean_acc,
        "grand_mean_accuracy": grand_mean_acc, "grand_std_accuracy": grand_std_acc,
        "fold_results_last_seed": fold_records,
        "loso_csv_path": LOSO_CSV_PATH,
        "reference_condition1b_full_mean_acc": CONDITION1B_FULL_MEAN_ACC,
        "reference_condition4v2_full_mean_acc_original_leaky_pipeline": CONDITION4V2_FULL_MEAN_ACC,
        "reference_asymmetric_fusion_mean_acc": ASYMMETRIC_FUSION_MEAN_ACC,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_payload, f, indent=2)
    volume.commit()
    log.info(f"  Saved: {OUTPUT_JSON}")
    log.info(f"  Saved: {LOSO_CSV_PATH}")

    # =========================================================================
    # C3 plausibility assertions -- printed next to the numbers, not silent.
    # Run AFTER both writes above so a failing assertion never suppresses the
    # diagnostic artifacts.
    # =========================================================================
    assert 0.0 <= grand_mean_acc <= 1.0, f"[C3 PLAUSIBILITY FAIL] grand mean accuracy {grand_mean_acc} outside [0,1]"
    assert len(unique_subjects) == 29, f"[C3 PLAUSIBILITY FAIL] expected 29 subjects, got {len(unique_subjects)}"
    assert len(SEEDS) == 5, f"[C3 PLAUSIBILITY FAIL] expected 5 seeds, got {len(SEEDS)}"
    assert len(per_seed_mean_acc) == len(SEEDS), (
        f"[C3 PLAUSIBILITY FAIL] expected {len(SEEDS)} per-seed means, got {len(per_seed_mean_acc)}"
    )
    for s_mean in per_seed_mean_acc:
        assert 0.0 <= s_mean <= 1.0, f"[C3 PLAUSIBILITY FAIL] per-seed mean accuracy {s_mean} outside [0,1]"
    CHANCE_LEVEL = 0.5
    if grand_mean_acc <= CHANCE_LEVEL:
        log.warning(
            f"  [C3] plausibility WARNING: grand mean accuracy ({grand_mean_acc:.4f}) is at or below "
            f"chance ({CHANCE_LEVEL}) -- unexpected for this pipeline (prior full-channel run: "
            f"{CONDITION4V2_FULL_MEAN_ACC*100:.2f}%). Worth a second look before trusting this number."
        )
    log.info(f"  [C3] plausibility: 29/29 subjects, {len(SEEDS)}/5 seeds, all accuracies in [0,1] -- OK")

    return {
        "grand_mean_accuracy": grand_mean_acc, "grand_std_accuracy": grand_std_acc,
        "per_seed_mean_accuracy": per_seed_mean_acc,
        "subject_decodability_pre_alignment": subj_decode_acc_pre,
        "subject_decodability_post_alignment": subj_decode_acc_post,
        "output_path": OUTPUT_JSON, "loso_csv_path": LOSO_CSV_PATH,
        "n_folds": len(unique_subjects), "n_seeds": len(SEEDS),
    }


@app.local_entrypoint()
def main(ea_mode: str = "pooled", cov_estimator: str = "fixed"):
    print("Condition 4 — MATCHED SPATIAL-ONLY CONTROL (Audit Fix)")
    print(f"ea_mode={ea_mode}  cov_estimator={cov_estimator}  seeds={SEEDS}")
    print("Same EA + identity-reference tangent-vectorize code as")
    print("run_step4_condition4_asymmetric_mamba.py, Mamba branch entirely removed.\n")
    results = run_matched_spatial_control.remote(ea_mode=ea_mode, cov_estimator=cov_estimator)
    print("\nMATCHED SPATIAL-ONLY CONTROL RESULTS:")
    for k, v in results.items():
        print(f"  {k:<32}: {v}")
