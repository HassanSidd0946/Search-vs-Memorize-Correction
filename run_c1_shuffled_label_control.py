# =============================================================================
# command to run (AFTER run_r1b_r2_r3_composition_runs.py::main has
# completed and committed results_r1b_r2_r3_composition_runs.json / the
# granular npz still needs to be present on the volume):
#   modal run run_c1_shuffled_label_control.py::main
# run_c1_shuffled_label_control.py
#
# C1 — SHUFFLED-LABEL CONTROL ON THE R2 DATASET SPECIFICALLY.
# Pre-registered in DECISIONS.md's "C1 / C2 — R2 shuffled-label control +
# sub-03 outlier check" section, BEFORE this script runs.
#
# WHY THIS SCRIPT EXISTS:
#   R2 (run_r1b_r2_r3_composition_runs.py) reported pre_cal_balanced=0.5773
#   on the encode-only Search-vs-Memorize contrast -- the first real-label,
#   cross-subject, pre-calibration number in this audit that is not at
#   chance. Before this is treated as anything more than "interesting,
#   unverified," it needs a shuffled-label null computed on the EXACT R2
#   dataset (N=2,900, uniformly 50 Search-encode + 50 Memorize-encode
#   epochs/subject across all 29 subjects -- sub-01's 10 unlogged
#   practice-trial epochs excluded at the epoching layer, see
#   run_data_engine_granular_on_modal.py) -- NOT F-LEAK's old shuffled-label
#   numbers, which were computed on the contaminated 200-epoch/class
#   dataset and do not bound this one.
#
# PROCEDURE: 30 independent within-subject label shuffles (each subject's
#   own R2 labels -- however many that subject has -- permuted among
#   themselves, exactly preserving that subject's own class count by
#   construction, not a fixed ==50 assumption), each run through the
#   IDENTICAL LOSO/EA/tangent/shrinkage-calibration pipeline as R2 itself
#   (byte-identical to run_r1b_r2_r3_composition_runs.py's run_loso),
#   seed=42 for every pipeline-internal RNG, full 29-fold LOSO per shuffle.
#
# IMPLEMENTATION NOTE (does not change what is computed, only the order):
#   EA whitening, covariance/tangent-space vectorization, and PCA are all
#   fit on X_train/X_k alone -- none of them consume y -- so these steps
#   are label-independent and IDENTICAL across all 30 shuffles for a given
#   fold. This script computes them ONCE per fold and reuses them across
#   shuffles, redoing only the label-dependent steps (stratified
#   calibration/test split, global/local classifier fits, shrinkage CV)
#   per shuffle. This is a ~30x reduction in redundant eigendecomposition
#   cost, not a change to the pipeline -- every shuffle still gets its own
#   independently-stratified calibration split and its own independently
#   fit classifier, exactly as if the full pipeline had been re-run from
#   scratch for that shuffle.
#
# PRE-REGISTERED VERDICT RULE (DECISIONS.md, fixed before running): if the
#   real R2 pre_cal_balanced (REAL_R2_PRE_CAL_BALANCED, set below) falls
#   OUTSIDE the shuffled-label 95% CI (primary = empirical percentile of
#   the 30 shuffle-level values) -> genuine subject-generalizable signal,
#   stated plainly. If INSIDE -> it does not currently distinguish itself
#   from within-subject shuffling alone at this N, stated with equal
#   plainness.
#
# Usage: modal run run_c1_shuffled_label_control.py::main
# =============================================================================

import modal

app    = modal.App("bci-c1-r2-shuffled-label-control")
volume = modal.Volume.from_name("eeg-data-vol")

GRANULAR_DATA_PATH = "/data/processed_eeg_all_subjects_granular.npz"
OUTPUT_JSON         = "/data/results_c1_r2_shuffled_label_control.json"
VOLUME_PATH         = "/data"

SFREQ, N_CHANNELS = 250, 62

RANDOM_SEED         = 42   # every pipeline-internal RNG (classifier fits, CV, cal split) -- unchanged across shuffles
COV_SHRINKAGE       = 0.1
PCA_MAX_COMPONENTS  = 35
LOGREG_C            = 1.0
CAL_FRACTION        = 0.15
SHRINK_GRID         = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS     = 3

SEARCH_ENCODE, MEMORIZE_ENCODE = 0, 4

N_SHUFFLES        = 30
SHUFFLE_BASE_SEED = 90210  # fixed, date-independent, distinct from RANDOM_SEED=42 -- one RNG per shuffle: SHUFFLE_BASE_SEED + shuffle_idx

# DECISIONS.md pre-registered reference value -- the real R2 pre_cal_balanced
# this null distribution is being compared against. NOT re-derived here from
# the granular npz (that would silently couple this script's correctness to
# a second, independent LOSO run) -- it is the number already reported and
# recorded from run_r1b_r2_r3_composition_runs.py's actual completed run.
#
# GATE C2: updated to the post-exclusion R2 value (N=2,900), transcribed
# from the actual completed run_r1b_r2_r3_composition_runs.py output.
REAL_R2_PRE_CAL_BALANCED = 0.5773379921335015

# GATE C6 STEP 1: Image.env() was shown NOT to hold for OMP/MKL/OPENBLAS --
# Modal overwrites those to match `cpu` at container start, AFTER the
# image's ENV layer. Fixed via threadpoolctl (BLAS runtime API, not an env
# var) plus cpu dropped 4.0 -> 1.0 as a second, independent mechanism. See
# run_c4_high_res_shuffled_label_control.py's image comment for the full
# account of how this was caught.
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
def run_c1(git_short_hash: str = "nogit"):

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
    log = logging.getLogger("c1-r2-shuffle")

    log.info(f"numpy={np.__version__} scipy={scipy.__version__} sklearn={sklearn.__version__}")
    # GATE C7 STEP 1b: this is a smoke test only -- the real protection is the
    # `with threadpool_limits(limits=1):` block around _core() below, not this
    # bare call. See run_c4_high_res_shuffled_label_control.py's identical comment.
    log.info(f"threadpool_info BEFORE threadpoolctl pin (smoke test): {threadpool_info()}")
    with threadpool_limits(limits=1):
        log.info(f"threadpool_info INSIDE threadpoolctl pin (smoke test, every entry must show "
                 f"num_threads=1): {threadpool_info()}")

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
    log.info(f"hardware_info: {hardware_info}")

    def _core():
        raw = np.load(GRANULAR_DATA_PATH, allow_pickle=True)
        X_all_np = raw["X"].astype(np.float32)
        code_all_np = raw["code"].astype(np.int64)
        subjects_all_np = raw["subjects"]

        mask_r2 = np.isin(code_all_np, [SEARCH_ENCODE, MEMORIZE_ENCODE])
        X_r2 = X_all_np[mask_r2]
        code_r2 = code_all_np[mask_r2]
        subjects_r2 = subjects_all_np[mask_r2]
        y_r2_real = np.where(code_r2 == SEARCH_ENCODE, 0, 1).astype(np.int64)

        N, C, T = X_r2.shape
        assert C == N_CHANNELS
        unique_subjects = sorted(np.unique(subjects_r2).tolist())
        n_subjects = len(unique_subjects)
        log.info(f"R2 encode-only dataset: X={X_r2.shape}, N={N}, subjects={n_subjects}, "
                 f"class_balance={np.bincount(y_r2_real).tolist()} (expect uniformly 50/subject/class post "
                 f"sub-01 exclusion, N=2,900 total)")

        # C3-style pre-check: every subject's own search/memorize encode counts must be
        # EQUAL to each other for a within-subject permutation shuffle to be a true
        # balance-preserving label shuffle. Written as an explicit equality check
        # rather than a hardcoded ==50 assumption on principle -- pre-exclusion,
        # sub-01 was the counterexample this check was designed to still pass
        # correctly on (55/55, not 50/50); post-exclusion all 29 subjects are
        # uniformly 50/50, but the check stays general rather than assuming that.
        per_subject_counts = {}
        for s in unique_subjects:
            sub_mask = subjects_r2 == s
            counts = np.bincount(y_r2_real[sub_mask], minlength=2)
            per_subject_counts[s] = counts.tolist()
            assert counts[0] == counts[1], (
                f"[C1 PRE-CHECK FAIL] sub-{s}: R2 class counts {counts.tolist()} are NOT balanced within this "
                "subject -- within-subject shuffling would not preserve balance as designed. Halting before any shuffle runs."
            )
        log.info(f"[C1 pre-check] every subject is internally balanced (search_count == memorize_count) in R2 -- "
                 f"within-subject shuffling is balance-preserving by construction. Per-subject counts: {per_subject_counts}")

        # =========================================================================
        # RIEMANNIAN / EA / CLASSIFIER UTILITIES -- IDENTICAL to
        # run_r1b_r2_r3_composition_runs.py (itself identical to the rest of the
        # pre-F3 F-DRIFT family).
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

        def c3_balance_check(y_true, y_pred, acc, label, declared_imbalanced_design=False, verbose=False):
            counts = np.bincount(y_true, minlength=2)
            n = len(y_true)
            balance = (counts / n).tolist()
            majority_rate = float(counts.max() / n)
            bal_acc = float(balanced_accuracy_score(y_true, y_pred))
            lift = float(acc - majority_rate)
            if verbose:
                print(f"    [C3] {label}: class_balance={[round(b, 4) for b in balance]} "
                      f"majority_rate={majority_rate:.4f} acc={acc:.4f} "
                      f"acc_minus_majority={lift:+.4f} balanced_acc={bal_acc:.4f}")
            if not declared_imbalanced_design:
                assert 0.45 <= min(balance) and max(balance) <= 0.55, (
                    f"[C3 FAIL] {label}: class balance {balance} is outside the 45/55 band."
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
        # PASS 1 -- precompute the label-independent per-fold features ONCE.
        # EA whitening / covariance / tangent-vectorization / scaler / PCA never
        # see y, so they are identical for every shuffle. Computed here once per
        # fold and reused in Pass 2 below.
        # =========================================================================
        log.info(f"Pass 1/2: precomputing label-independent EA/tangent/PCA features for {n_subjects} folds ...")
        fold_cache = []
        for fold_idx, test_sub in enumerate(unique_subjects):
            t0 = time.time()
            is_holdout = subjects_r2 == test_sub
            X_train = X_r2[~is_holdout]
            X_k = X_r2[is_holdout]

            mu = X_train.mean(axis=(0, 2), keepdims=True)
            sd = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
            X_train_z = ((X_train - mu) / sd).astype(np.float32)
            X_k_z = ((X_k - mu) / sd).astype(np.float32)

            W_ea = fit_ea_whitening(X_train_z)
            X_train_aligned = apply_ea_whitening_signal(X_train_z, W_ea).astype(np.float32)
            X_k_aligned = apply_ea_whitening_signal(X_k_z, W_ea).astype(np.float32)

            tan_train = tangent_vectorize(trial_covariances(X_train_aligned))
            tan_k = tangent_vectorize(trial_covariances(X_k_aligned))

            scaler = StandardScaler()
            feat_train_z = scaler.fit_transform(tan_train)
            feat_k_z = scaler.transform(tan_k)

            n_components = min(PCA_MAX_COMPONENTS, feat_train_z.shape[1] - 1, feat_train_z.shape[0] - 1)
            pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
            X_train_pca = pca.fit_transform(feat_train_z)
            X_k_pca = pca.transform(feat_k_z)

            fold_cache.append({
                "test_sub": test_sub,
                "is_holdout": is_holdout,          # boolean mask into the R2 arrays, order-preserving
                "X_train_pca": X_train_pca,        # rows aligned with y_r2_real[~is_holdout]
                "X_k_pca": X_k_pca,                # rows aligned with y_r2_real[is_holdout]
                "tangent_dim": int(tan_train.shape[1]),
            })
            log.info(f"  [precompute] fold {fold_idx+1}/{n_subjects} sub-{test_sub} "
                     f"n_train={len(X_train)} n_k={len(X_k)} tangent_dim={tan_train.shape[1]} "
                     f"[{time.time()-t0:.0f}s]")

        # =========================================================================
        # PASS 2 -- for each of 30 within-subject label shuffles, redo only the
        # label-dependent steps (stratified cal/test split, classifier fits,
        # shrinkage CV) against the cached label-independent features.
        # =========================================================================
        log.info(f"Pass 2/2: running {N_SHUFFLES} within-subject label shuffles through the calibration pipeline ...")
        shuffle_summaries = []
        for shuffle_idx in range(N_SHUFFLES):
            t_shuf = time.time()
            rng = np.random.RandomState(SHUFFLE_BASE_SEED + shuffle_idx)
            y_shuffled = y_r2_real.copy()
            for s in unique_subjects:
                sub_idx = np.where(subjects_r2 == s)[0]
                y_shuffled[sub_idx] = rng.permutation(y_r2_real[sub_idx])

            pre_bal_folds, post_bal_folds = [], []
            for fc in fold_cache:
                is_holdout = fc["is_holdout"]
                y_train = y_shuffled[~is_holdout]
                y_k = y_shuffled[is_holdout]
                X_train_pca, X_k_pca = fc["X_train_pca"], fc["X_k_pca"]

                if len(np.unique(y_train)) < 2 or len(np.unique(y_k)) < 2:
                    log.warning(f"  [shuffle {shuffle_idx}] sub-{fc['test_sub']}: missing a class after "
                                "shuffling, skipping fold (should not happen given the balance pre-check)")
                    continue

                sss = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - CAL_FRACTION), random_state=RANDOM_SEED)
                cal_idx, test_idx = next(sss.split(X_k_pca, y_k))
                X_cal_pca, y_cal = X_k_pca[cal_idx], y_k[cal_idx]
                X_test_pca, y_test = X_k_pca[test_idx], y_k[test_idx]
                if len(np.unique(y_cal)) < 2 or len(np.unique(y_test)) < 2:
                    log.warning(f"  [shuffle {shuffle_idx}] sub-{fc['test_sub']}: cal/test split missing a "
                                "class, skipping fold")
                    continue

                coef_final, icpt_final, best_shrink, global_clf = fit_shrinkage_classifier(
                    X_train_pca, y_train, X_cal_pca, y_cal, RANDOM_SEED)

                pre_cal_preds = global_clf.predict(X_test_pca)
                pre_cal_acc = float((pre_cal_preds == y_test).mean())
                final_preds = linear_predict(coef_final, icpt_final, X_test_pca)
                post_cal_acc = float((final_preds == y_test).mean())

                pre_plaus = c3_balance_check(y_test, pre_cal_preds, pre_cal_acc,
                                              f"shuffle{shuffle_idx} sub-{fc['test_sub']} pre_cal")
                post_plaus = c3_balance_check(y_test, final_preds, post_cal_acc,
                                               f"shuffle{shuffle_idx} sub-{fc['test_sub']} post_cal")
                pre_bal_folds.append(pre_plaus["balanced_accuracy"])
                post_bal_folds.append(post_plaus["balanced_accuracy"])

            shuffle_pre_bal_mean = float(np.mean(pre_bal_folds)) if pre_bal_folds else float("nan")
            shuffle_post_bal_mean = float(np.mean(post_bal_folds)) if post_bal_folds else float("nan")
            shuffle_summaries.append({
                "shuffle_idx": shuffle_idx, "n_folds": len(pre_bal_folds),
                "pre_cal_balanced_accuracy_mean": shuffle_pre_bal_mean,
                "post_cal_balanced_accuracy_mean": shuffle_post_bal_mean,
            })
            log.info(f"  [shuffle {shuffle_idx+1}/{N_SHUFFLES}] n_folds={len(pre_bal_folds)} "
                     f"pre_cal_balanced={shuffle_pre_bal_mean:.4f} post_cal_balanced={shuffle_post_bal_mean:.4f} "
                     f"[{time.time()-t_shuf:.0f}s]")

        # =========================================================================
        # Aggregate the null distribution and apply the pre-registered rule.
        # =========================================================================
        pre_bal_values = np.array([s["pre_cal_balanced_accuracy_mean"] for s in shuffle_summaries
                                    if not math.isnan(s["pre_cal_balanced_accuracy_mean"])])
        assert len(pre_bal_values) == N_SHUFFLES, (
            f"[C3 PLAUSIBILITY FAIL] expected {N_SHUFFLES} usable shuffle-level values, got {len(pre_bal_values)}"
        )

        null_mean = float(np.mean(pre_bal_values))
        null_sd = float(np.std(pre_bal_values, ddof=1))
        pctl_lo, pctl_hi = float(np.percentile(pre_bal_values, 2.5)), float(np.percentile(pre_bal_values, 97.5))
        z = 1.959963984540054
        normal_lo, normal_hi = null_mean - z * null_sd, null_mean + z * null_sd

        real_val = REAL_R2_PRE_CAL_BALANCED
        inside_percentile = pctl_lo <= real_val <= pctl_hi
        inside_normal = normal_lo <= real_val <= normal_hi

        if not inside_percentile:
            verdict = (f"GENUINE, SUBJECT-GENERALIZABLE SIGNAL — real R2 pre_cal_balanced ({real_val:.4f}) falls "
                        f"OUTSIDE the shuffled-label 95% percentile CI [{pctl_lo:.4f}, {pctl_hi:.4f}] "
                        f"(null mean={null_mean:.4f}, SD={null_sd:.4f}, N={N_SHUFFLES} shuffles).")
        else:
            verdict = (f"NOT DISTINGUISHED FROM SHUFFLE NOISE AT THIS N — real R2 pre_cal_balanced ({real_val:.4f}) "
                        f"falls INSIDE the shuffled-label 95% percentile CI [{pctl_lo:.4f}, {pctl_hi:.4f}] "
                        f"(null mean={null_mean:.4f}, SD={null_sd:.4f}, N={N_SHUFFLES} shuffles). This does not "
                        "currently distinguish itself from what within-subject shuffling alone produces at this N.")
        if inside_percentile != inside_normal:
            verdict += (f" NOTE: the primary percentile CI and the secondary normal-approximation CI "
                         f"[{normal_lo:.4f}, {normal_hi:.4f}] DISAGREE on which side of the boundary "
                         f"{real_val:.4f} falls — both are reported, do not silently pick one.")

        log.info(f"\n{'='*70}\n  C1 PRE-REGISTERED VERDICT\n{'='*70}\n  {verdict}")

        results_payload = {
            "real_R2_pre_cal_balanced": real_val,
            "n_shuffles": N_SHUFFLES,
            "shuffle_base_seed": SHUFFLE_BASE_SEED,
            "null_distribution_mean": null_mean,
            "null_distribution_sd": null_sd,
            "null_distribution_percentile_95ci": [pctl_lo, pctl_hi],
            "null_distribution_normal_approx_95ci": [normal_lo, normal_hi],
            "inside_percentile_ci": bool(inside_percentile),
            "inside_normal_approx_ci": bool(inside_normal),
            "verdict": verdict,
            "shuffle_level_results": shuffle_summaries,
            "hyperparameters": {
                "pca_max_components": PCA_MAX_COMPONENTS, "logreg_C": LOGREG_C, "random_seed": RANDOM_SEED,
                "cov_shrinkage": COV_SHRINKAGE, "cal_fraction": CAL_FRACTION,
            },
            "n_subjects": n_subjects, "subjects": unique_subjects,
            "hardware_info": hardware_info,
        }
        # GATE C5 STEP 3: Modal Volumes overwrite in place with no version
        # history -- a stable filename destroys the evidence for the run before
        # it (this is exactly how the 02:36 UTC C4 run was permanently lost
        # except for a lucky local pull before the 03:24 UTC run overwrote it).
        # Every run now also writes an immutable, UTC-timestamped + git-hash-
        # stamped copy; OUTPUT_JSON is kept as a convenience pointer to latest.
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

        return {
            "null_mean": null_mean, "null_sd": null_sd,
            "percentile_95ci": [pctl_lo, pctl_hi], "normal_approx_95ci": [normal_lo, normal_hi],
            "real_R2_pre_cal_balanced": real_val,
            "verdict": verdict,
            "output_path": OUTPUT_JSON,
            "stamped_output_path": stamped_path,
        }

    # GATE C7 STEP 1b: real `with` block around the numerical work --
    # see run_c4_high_res_shuffled_label_control.py's identical comment.
    with threadpool_limits(limits=1):
        log.info(f"threadpool_info INSIDE with-block (right before the numerical "
                 f"pipeline runs): {threadpool_info()}")
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
    print("C1 — shuffled-label control on the exact R2 encode-only dataset (N=2,900).")
    print("Pre-registered in DECISIONS.md's 'C1 / C2' section BEFORE this run.")
    print(f"Real R2 pre_cal_balanced being tested against the null: {REAL_R2_PRE_CAL_BALANCED}")
    print(f"{N_SHUFFLES} within-subject label shuffles, identical LOSO/EA/calibration pipeline per shuffle.")
    print(f"git_short_hash for this run's stamped output filename: {git_short_hash}\n")
    results = run_c1.remote(git_short_hash=git_short_hash)
    print("\nRESULTS:")
    for k, v in results.items():
        print(f"  {k}: {v}")
