# =============================================================================
# command to run (requires run_c1_shuffled_label_control.py::main to have
# already completed and committed both processed_eeg_all_subjects_granular
# .npz AND results_c1_r2_shuffled_label_control.json to the volume):
#   modal run run_c4_high_res_shuffled_label_control.py::main
# run_c4_high_res_shuffled_label_control.py
#
# C4 -- HIGHER-RESOLUTION PERMUTATION P-VALUE (500 SHUFFLES).
# Pre-registered in DECISIONS.md's "C3 / C4 -- robustness checks on the
# accepted R2 signal" section, BEFORE this script runs.
#
# WHY: C1 (30 shuffles, ledger L040, DECISIONS.md) accepted R2's
#   pre_cal_balanced as a genuine signal (real value outside the 95% CI --
#   see REAL_R2_PRE_CAL_BALANCED below for the live value this run
#   actually uses). 30 shuffles gives a minimum resolvable empirical p of
#   1/31 -- adequate for that CI-membership test, not for a headline
#   statistic. This reruns the IDENTICAL C1 procedure at 500 within-subject
#   label shuffles.
#
# IDENTICAL PIPELINE, SAME OPTIMIZATION: EA whitening / tangent-space
#   vectorization / PCA are fit on X_train/X_k alone (never consume y), so
#   these are precomputed ONCE per fold and reused across all 500 shuffles
#   -- exactly C1's design, just at higher N. This was already verified
#   bit-identical to a naive full-recompute-per-shuffle baseline before C1
#   was trusted; the same code path is reused here unmodified.
#
# SAME SHUFFLE_BASE_SEED AS C1: the first 30 of these 500 shuffles are the
#   IDENTICAL shuffles C1 already ran (deterministic RNG per shuffle
#   index -- np.random.RandomState(SHUFFLE_BASE_SEED + shuffle_idx)). This
#   script loads C1's actual output JSON and checks this reproduces
#   EXACTLY, not just approximately -- a free internal-consistency
#   guarantee, separate from the statistical consistency check below.
#
# PRE-REGISTERED CONSISTENCY RULE (DECISIONS.md, fixed before running): the
#   500-shuffle percentile 95% CI must exclude the real R2 value on the
#   SAME side (above the upper bound) as the 30-shuffle CI did. If the two
#   disagree on which side of the boundary it falls, or if the first 30 of
#   the 500 shuffle values do not exactly reproduce C1's original 30
#   values, STOP and report the discrepancy explicitly before either
#   result is cited -- never silently prefer one over the other.
#
# Usage: modal run run_c4_high_res_shuffled_label_control.py::main
# =============================================================================

import modal

app    = modal.App("bci-c4-r2-shuffled-label-control-highres")
volume = modal.Volume.from_name("eeg-data-vol")

GRANULAR_DATA_PATH = "/data/processed_eeg_all_subjects_granular.npz"
C1_RESULTS_JSON     = "/data/results_c1_r2_shuffled_label_control.json"
OUTPUT_JSON         = "/data/results_c4_high_res_shuffled_label_control.json"
VOLUME_PATH         = "/data"

SFREQ, N_CHANNELS = 250, 62

RANDOM_SEED         = 42
COV_SHRINKAGE       = 0.1
PCA_MAX_COMPONENTS  = 35
LOGREG_C            = 1.0
CAL_FRACTION        = 0.15
SHRINK_GRID         = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS     = 3

SEARCH_ENCODE, MEMORIZE_ENCODE = 0, 4

N_SHUFFLES        = 500   # raised from C1's 30, per C4's pre-registration
SHUFFLE_BASE_SEED = 90210 # IDENTICAL to C1 -- shuffles 0..29 of this run are C1's original 30

# GATE C2: updated to the post-exclusion R2 value (N=2,900), same value as
# run_c1_shuffled_label_control.py, transcribed from the actual completed
# run_r1b_r2_r3_composition_runs.py output.
REAL_R2_PRE_CAL_BALANCED = 0.5773379921335015

# GATE C5: see run_c1_shuffled_label_control.py's image comment for why
# these are set here (before numpy import, survives warm-container reuse)
# rather than as os.environ writes inside the function body. This is the
# script the non-determinism was actually caught on: two identical-code,
# identical-data runs produced null_mean 0.49950918394623284 (02:36 UTC) vs
# 0.4995108259823577 (03:24 UTC) -- a direct bit-level diff of the two
# persisted shuffle_level_pre_cal_balanced arrays found exactly 2 of 500
# shuffle values differing (indices 266, 388), both by the identical delta,
# neither below index 30. Traced to unpinned BLAS/LAPACK thread count on
# cpu=4.0, not a seeding bug -- the intentional RNG (SHUFFLE_BASE_SEED +
# shuffle_idx) and every sklearn random_state were already fixed.
# GATE C6 STEP 1: Image.env() was shown NOT to hold -- Modal overwrites
# OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS at container start to
# match the function's `cpu` allocation, applied AFTER the image's ENV
# layer. NUMEXPR_NUM_THREADS/VECLIB_MAXIMUM_THREADS are not Modal-managed
# and did hold. Fix: (1) threadpoolctl, which sets thread counts through
# the BLAS runtime's own API (dlopen'd library handle) rather than an env
# var read at import time, so a post-hoc write from Modal's container
# startup cannot undo it; (2) cpu dropped from 4.0 to 1.0 so Modal's own
# override now WRITES "1" instead of fighting the pin -- belt and braces,
# per gate instruction, not a replacement for (1). Image.env() is left in
# place (harmless, still correct for the two non-Modal-managed vars) but
# is no longer trusted alone for OMP/MKL/OPENBLAS.
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
def run_c4(git_short_hash: str = "nogit"):

    import logging, time, math, json, os
    import numpy as np
    import scipy
    import sklearn
    from threadpoolctl import threadpool_limits, threadpool_info
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
    from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("c4-r2-shuffle-highres")

    log.info(f"numpy={np.__version__} scipy={scipy.__version__} sklearn={sklearn.__version__}")
    log.info("thread env vars (informational only -- GATE C6: Modal overwrites OMP/MKL/OPENBLAS to "
              "match cpu allocation after this image's ENV layer runs, so these are NOT proof of the "
              "pin): " + ", ".join(
        f"{k}={os.environ.get(k)}" for k in
        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    ))

    # GATE C6 STEP 1b/1c: the accepted proof standard is threadpool_info()
    # itself reporting num_threads=1 for every BLAS/OpenMP library actually
    # loaded -- not the env vars above, and not a successful edit.
    # GATE C7 STEP 1b: this early call+print pair is a smoke test only (does
    # the pin take effect at all, this early in the container's life) -- it
    # is NOT what protects the actual numerical work below. That protection
    # comes from the real `with threadpool_limits(limits=1):` block wrapping
    # _core()'s call further down, which re-asserts the limit at the moment
    # the pipeline runs regardless of what this container did in any prior
    # invocation. Reasoning about whether a bare, unrestored call "persists"
    # across warm-container reuse was exactly the mistake GATE C6 flagged;
    # the `with` block removes the need to reason about it at all.
    log.info(f"threadpool_info BEFORE threadpoolctl pin (smoke test): {threadpool_info()}")
    with threadpool_limits(limits=1):
        log.info(f"threadpool_info INSIDE threadpoolctl pin (smoke test, every entry must show "
                 f"num_threads=1): {threadpool_info()}")

    # GATE C6 STEP 2b: hardware (SIMD dispatch, BLAS build) recorded into
    # the JSON artifact itself, not only the log -- a reproducibility claim
    # needs the hardware in the artifact, since OpenBLAS is DYNAMIC_ARCH=1
    # and dispatches different kernels (different reduction order) on
    # different host CPU generations even with threads correctly pinned.
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
        # Load C1's actual completed output for the reproducibility + consistency checks.
        with open(C1_RESULTS_JSON) as f:
            c1_payload = json.load(f)
        c1_shuffle_values = [s["pre_cal_balanced_accuracy_mean"] for s in
                              sorted(c1_payload["shuffle_level_results"], key=lambda r: r["shuffle_idx"])]
        assert len(c1_shuffle_values) == 30, f"expected 30 shuffle values in C1's output, got {len(c1_shuffle_values)}"
        c1_null_mean = c1_payload["null_distribution_mean"]
        c1_null_sd = c1_payload["null_distribution_sd"]
        c1_pctl_ci = c1_payload["null_distribution_percentile_95ci"]
        log.info(f"Loaded C1's original 30-shuffle result: mean={c1_null_mean:.4f}, SD={c1_null_sd:.4f}, "
                 f"percentile_95ci={c1_pctl_ci}")

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
        log.info(f"R2 encode-only dataset: X={X_r2.shape}, N={N}, subjects={n_subjects}")

        for s in unique_subjects:
            sub_mask = subjects_r2 == s
            counts = np.bincount(y_r2_real[sub_mask], minlength=2)
            assert counts[0] == counts[1], (
                f"[C4 PRE-CHECK FAIL] sub-{s}: R2 class counts {counts.tolist()} are NOT balanced within this "
                "subject. Halting before any shuffle runs."
            )
        log.info("[C4 pre-check] every subject internally balanced in R2 -- matches C1's own pre-check.")

        # =========================================================================
        # IDENTICAL utilities to run_c1_shuffled_label_control.py.
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

        def c3_balance_check(y_true, y_pred, declared_imbalanced_design=False):
            counts = np.bincount(y_true, minlength=2)
            balance = (counts / len(y_true)).tolist()
            bal_acc = float(balanced_accuracy_score(y_true, y_pred))
            if not declared_imbalanced_design:
                assert 0.45 <= min(balance) and max(balance) <= 0.55, f"[C3 FAIL] class balance {balance} outside 45/55."
            return bal_acc

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
        # PASS 1 -- precompute label-independent per-fold features ONCE (identical
        # to C1's design and verified equivalence).
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
                "test_sub": test_sub, "is_holdout": is_holdout,
                "X_train_pca": X_train_pca, "X_k_pca": X_k_pca,
            })
            log.info(f"  [precompute] fold {fold_idx+1}/{n_subjects} sub-{test_sub} [{time.time()-t0:.0f}s]")

        # =========================================================================
        # PASS 2 -- 500 within-subject label shuffles against the cached features.
        # =========================================================================
        log.info(f"Pass 2/2: running {N_SHUFFLES} within-subject label shuffles ...")
        shuffle_pre_bal_means = []
        for shuffle_idx in range(N_SHUFFLES):
            t_shuf = time.time()
            rng = np.random.RandomState(SHUFFLE_BASE_SEED + shuffle_idx)
            y_shuffled = y_r2_real.copy()
            for s in unique_subjects:
                sub_idx = np.where(subjects_r2 == s)[0]
                y_shuffled[sub_idx] = rng.permutation(y_r2_real[sub_idx])

            pre_bal_folds = []
            for fc in fold_cache:
                is_holdout = fc["is_holdout"]
                y_train = y_shuffled[~is_holdout]
                y_k = y_shuffled[is_holdout]
                X_train_pca, X_k_pca = fc["X_train_pca"], fc["X_k_pca"]

                if len(np.unique(y_train)) < 2 or len(np.unique(y_k)) < 2:
                    continue

                sss = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - CAL_FRACTION), random_state=RANDOM_SEED)
                cal_idx, test_idx = next(sss.split(X_k_pca, y_k))
                X_cal_pca, y_cal = X_k_pca[cal_idx], y_k[cal_idx]
                X_test_pca, y_test = X_k_pca[test_idx], y_k[test_idx]
                if len(np.unique(y_cal)) < 2 or len(np.unique(y_test)) < 2:
                    continue

                _, _, _, global_clf = fit_shrinkage_classifier(X_train_pca, y_train, X_cal_pca, y_cal, RANDOM_SEED)
                pre_cal_preds = global_clf.predict(X_test_pca)
                pre_bal_folds.append(c3_balance_check(y_test, pre_cal_preds))

            shuffle_pre_bal_mean = float(np.mean(pre_bal_folds)) if pre_bal_folds else float("nan")
            shuffle_pre_bal_means.append(shuffle_pre_bal_mean)
            if (shuffle_idx + 1) % 25 == 0 or shuffle_idx < 3:
                log.info(f"  [shuffle {shuffle_idx+1}/{N_SHUFFLES}] pre_cal_balanced={shuffle_pre_bal_mean:.4f} "
                         f"[{time.time()-t_shuf:.0f}s]")

        # =========================================================================
        # Reproducibility check: shuffles 0..29 of this run must EXACTLY match C1's
        # original 30 values (same SHUFFLE_BASE_SEED, same deterministic RNG).
        # =========================================================================
        first_30 = shuffle_pre_bal_means[:30]
        max_abs_diff = max(abs(a - b) for a, b in zip(first_30, c1_shuffle_values))
        reproduces_c1_exactly = max_abs_diff < 1e-9
        log.info(f"[C4 reproducibility check] max|first_30_of_500 - C1_original_30| = {max_abs_diff:.2e} -> "
                 f"{'EXACT MATCH' if reproduces_c1_exactly else 'MISMATCH -- INVESTIGATE'}")

        # =========================================================================
        # Aggregate the 500-shuffle null distribution and apply the pre-registered rule.
        # =========================================================================
        values = np.array(shuffle_pre_bal_means)
        null_mean = float(np.mean(values))
        null_sd = float(np.std(values, ddof=1))
        pctl_lo, pctl_hi = float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
        z = 1.959963984540054
        normal_lo, normal_hi = null_mean - z * null_sd, null_mean + z * null_sd

        n_at_least_as_extreme = int(np.sum(values >= REAL_R2_PRE_CAL_BALANCED))
        p_raw = n_at_least_as_extreme / N_SHUFFLES
        p_corrected = (n_at_least_as_extreme + 1) / (N_SHUFFLES + 1)

        inside_percentile_500 = pctl_lo <= REAL_R2_PRE_CAL_BALANCED <= pctl_hi
        inside_percentile_30 = c1_pctl_ci[0] <= REAL_R2_PRE_CAL_BALANCED <= c1_pctl_ci[1]
        same_side = (not inside_percentile_500) and (not inside_percentile_30)

        consistency_ok = reproduces_c1_exactly and same_side
        if not reproduces_c1_exactly:
            consistency_note = (f"MISMATCH: first 30 of the 500 shuffles do NOT exactly reproduce C1's original 30 "
                                 f"values (max abs diff={max_abs_diff:.2e}). STOP -- do not cite either result until "
                                 "this is investigated.")
        elif not same_side:
            consistency_note = (f"DISAGREEMENT: the 30-shuffle CI ({'excludes' if not inside_percentile_30 else 'includes'} "
                                 f"{REAL_R2_PRE_CAL_BALANCED}) and the 500-shuffle CI "
                                 f"({'excludes' if not inside_percentile_500 else 'includes'} "
                                 f"{REAL_R2_PRE_CAL_BALANCED}) do not agree on which side of the boundary "
                                 f"{REAL_R2_PRE_CAL_BALANCED} falls. STOP -- do not cite either result until this "
                                 "is investigated.")
        else:
            consistency_note = (f"CONSISTENT: first 30 of 500 exactly reproduce C1's original 30 values, and both the "
                                 f"30-shuffle and 500-shuffle percentile CIs exclude {REAL_R2_PRE_CAL_BALANCED} on the "
                                 "same (above) side.")

        verdict = (f"C4 empirical p-value (fraction of {N_SHUFFLES} shuffles >= {REAL_R2_PRE_CAL_BALANCED}): "
                    f"raw={p_raw:.4f} ({n_at_least_as_extreme}/{N_SHUFFLES}), add-one-corrected={p_corrected:.4f}. "
                    f"500-shuffle null: mean={null_mean:.4f}, SD={null_sd:.4f}, percentile_95ci=[{pctl_lo:.4f},{pctl_hi:.4f}], "
                    f"normal_approx_95ci=[{normal_lo:.4f},{normal_hi:.4f}]. {consistency_note}")

        log.info(f"\n{'='*70}\n  C4 RESULT\n{'='*70}\n  {verdict}")

        results_payload = {
            "real_R2_pre_cal_balanced": REAL_R2_PRE_CAL_BALANCED,
            "n_shuffles": N_SHUFFLES,
            "shuffle_base_seed": SHUFFLE_BASE_SEED,
            "p_value_raw": p_raw,
            "p_value_add_one_corrected": p_corrected,
            "n_shuffles_at_least_as_extreme": n_at_least_as_extreme,
            "null_distribution_mean": null_mean,
            "null_distribution_sd": null_sd,
            "null_distribution_percentile_95ci": [pctl_lo, pctl_hi],
            "null_distribution_normal_approx_95ci": [normal_lo, normal_hi],
            "reproduces_c1_first_30_exactly": bool(reproduces_c1_exactly),
            "max_abs_diff_vs_c1_first_30": max_abs_diff,
            "agrees_with_c1_on_ci_side": bool(same_side),
            "consistency_ok": bool(consistency_ok),
            "consistency_note": consistency_note,
            "c1_reference": {"null_mean": c1_null_mean, "null_sd": c1_null_sd, "percentile_95ci": c1_pctl_ci},
            "hardware_info": hardware_info,
            "verdict": verdict,
            "shuffle_level_pre_cal_balanced": shuffle_pre_bal_means,
        }
        # GATE C5 STEP 3: non-overwriting artifact. This is the exact script
        # whose overwrite-in-place destroyed the 02:36 UTC run's evidence the
        # moment the 03:24 UTC run committed over it -- only a local pull taken
        # beforehand survived. Never again: every run keeps its own copy.
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
            "p_value_raw": p_raw, "p_value_add_one_corrected": p_corrected,
            "null_mean": null_mean, "null_sd": null_sd,
            "percentile_95ci": [pctl_lo, pctl_hi], "normal_approx_95ci": [normal_lo, normal_hi],
            "reproduces_c1_first_30_exactly": reproduces_c1_exactly,
            "agrees_with_c1_on_ci_side": same_side,
            "consistency_ok": consistency_ok,
            "verdict": verdict,
            "output_path": OUTPUT_JSON,
            "stamped_output_path": stamped_path,
        }

    # GATE C7 STEP 1b: wrap the actual numerical work in a real `with` block
    # rather than relying on an unrestored bare call + reasoning about warm-
    # container persistence -- this guarantees the pin is (re-)active for
    # this invocation's numerical work regardless of container history.
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
    print("C4 -- higher-resolution (500-shuffle) permutation p-value on the exact R2 dataset.")
    print("Pre-registered in DECISIONS.md's 'C3 / C4' section BEFORE this run.")
    print(f"Real R2 pre_cal_balanced being tested against the null: {REAL_R2_PRE_CAL_BALANCED}")
    print(f"{N_SHUFFLES} shuffles, same SHUFFLE_BASE_SEED as C1 -- first 30 must exactly reproduce C1's result.")
    print(f"git_short_hash for this run's stamped output filename: {git_short_hash}\n")
    results = run_c4.remote(git_short_hash=git_short_hash)
    print("\nRESULTS:")
    for k, v in results.items():
        print(f"  {k}: {v}")
