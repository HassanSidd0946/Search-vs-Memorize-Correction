# =============================================================================
# command to run:
#   modal run run_d2_parity_split_check.py::main
# run_d2_parity_split_check.py
#
# D2 -- PARITY-SPLIT CHECK ON R2 (task-instruction vs. block-order).
# Pre-registered in DECISIONS.md's "D1 / D2" section, BEFORE this script
# runs.
#
# WHY: H3 (RESULTS_LEDGER.md L048) named "does R2's signal reflect
#   task instruction specifically, vs. block-order/session-position" as an
#   OPEN QUESTION -- neither F-STIM nor F-PARITY-WITHIN was computed on
#   R2's own clean encode-only dataset. This is the direct test.
#
# DESIGN: split the 29 R2 subjects by parity per D2's counterbalancing
#   scheme (14 odd = Search-first, 15 even = Memorize-first). Run the
#   IDENTICAL LOSO/EA/tangent/shrinkage-calibration pipeline SEPARATELY
#   within each parity group -- a held-out subject's training pool is
#   restricted to the OTHER subjects of the SAME parity group only
#   (13-fold LOSO within odd, 14-fold within even). Report pre_cal_balanced
#   per group.
#
# NULL REFERENCE: the existing C1/C4 pooled null (29-subject mixed-parity
#   pool) is NOT valid at this smaller per-group N -- a mean over 13-14
#   folds has a wider natural spread than a mean over 29. A FRESH
#   within-group shuffled-label null is run instead: 30 within-subject
#   label shuffles restricted to that group's own subjects, identical
#   precompute-once-per-fold design already verified equivalent for C1.
#
# PRE-REGISTERED VERDICT RULE (DECISIONS.md, fixed before running):
#   BOTH groups outside their own null's 95% CI (high side) -> signal not
#     explained by block-order/position alone (position/label are
#     inversely mapped between the two groups) -- strengthens the
#     task-instruction reading.
#   ONLY ONE group outside its own null -> name plainly as evidence the
#     signal may be position-driven, even though it complicates the
#     finding.
#   NEITHER group outside its own null -> inconsistent with C1/C4's
#     full-pool result -- halt and report for discussion, no post-hoc
#     reconciliation.
#
# Usage: modal run run_d2_parity_split_check.py::main
# =============================================================================

import modal

app    = modal.App("bci-d2-parity-split-check")
volume = modal.Volume.from_name("eeg-data-vol")

GRANULAR_DATA_PATH = "/data/processed_eeg_all_subjects_granular.npz"
OUTPUT_JSON         = "/data/results_d2_parity_split_check.json"
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

N_SHUFFLES        = 30
SHUFFLE_BASE_SEED = 71717  # distinct from C1/C4's 90210 -- an independent within-group null, not a subset of theirs

# GATE C6 STEP 1: Image.env() shown NOT to hold for OMP/MKL/OPENBLAS -- see
# run_c4_high_res_shuffled_label_control.py's image comment. Fixed via
# threadpoolctl + cpu 4.0 -> 1.0.
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
def run_d2(git_short_hash: str = "nogit"):

    import logging, time, math, json, os
    import numpy as np
    import scipy
    import sklearn
    from threadpoolctl import threadpool_limits, threadpool_info
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("d2-parity-split")

    log.info(f"numpy={np.__version__} scipy={scipy.__version__} sklearn={sklearn.__version__}")
    # GATE C7 STEP 1b: smoke test only -- real protection is the `with` block
    # around _core() below. See run_c4's identical comment.
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

        # D2 counterbalancing: odd subject number -> Search-first; even -> Memorize-first.
        odd_subjects = [s for s in unique_subjects if int(s) % 2 == 1]
        even_subjects = [s for s in unique_subjects if int(s) % 2 == 0]
        log.info(f"Parity split: odd (Search-first) n={len(odd_subjects)} {odd_subjects}")
        log.info(f"Parity split: even (Memorize-first) n={len(even_subjects)} {even_subjects}")

        # =========================================================================
        # IDENTICAL utilities to run_c1_shuffled_label_control.py / run_r1b_r2_r3_composition_runs.py.
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

        def c3_balance_check(y_true, y_pred):
            counts = np.bincount(y_true, minlength=2)
            balance = (counts / len(y_true)).tolist()
            bal_acc = float(balanced_accuracy_score(y_true, y_pred))
            assert 0.45 <= min(balance) and max(balance) <= 0.55, f"[D2 C3 FAIL] class balance {balance} outside 45/55."
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
        # Per-group precompute: label-independent EA/tangent/PCA features, ONCE
        # per fold, restricted to a given group's own subject list.
        # =========================================================================
        def precompute_group_folds(group_subjects, group_name):
            group_mask = np.isin(subjects_r2, group_subjects)
            X_group = X_r2[group_mask]
            y_group = y_r2_real[group_mask]
            subj_group = subjects_r2[group_mask]

            fold_cache = []
            for fold_idx, test_sub in enumerate(group_subjects):
                t0 = time.time()
                is_holdout = subj_group == test_sub
                X_train = X_group[~is_holdout]
                X_k = X_group[is_holdout]

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
                log.info(f"  [{group_name} precompute] fold {fold_idx+1}/{len(group_subjects)} sub-{test_sub} "
                         f"[{time.time()-t0:.0f}s]")
            return fold_cache, y_group, subj_group

        def run_real_label_loso(fold_cache, y_group, subj_group, group_name):
            pre_bal_folds = []
            for fc in fold_cache:
                is_holdout = fc["is_holdout"]
                y_train, y_k = y_group[~is_holdout], y_group[is_holdout]
                X_train_pca, X_k_pca = fc["X_train_pca"], fc["X_k_pca"]

                sss = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - CAL_FRACTION), random_state=RANDOM_SEED)
                cal_idx, test_idx = next(sss.split(X_k_pca, y_k))
                X_cal_pca, y_cal = X_k_pca[cal_idx], y_k[cal_idx]
                X_test_pca, y_test = X_k_pca[test_idx], y_k[test_idx]
                if len(np.unique(y_cal)) < 2 or len(np.unique(y_test)) < 2:
                    log.warning(f"  [{group_name}] sub-{fc['test_sub']}: cal/test split missing a class, skipping")
                    continue

                _, _, best_shrink, global_clf = fit_shrinkage_classifier(X_train_pca, y_train, X_cal_pca, y_cal, RANDOM_SEED)
                pre_cal_preds = global_clf.predict(X_test_pca)
                pre_bal_folds.append(c3_balance_check(y_test, pre_cal_preds))
            return float(np.mean(pre_bal_folds)), len(pre_bal_folds)

        def run_within_group_null(fold_cache, y_group, subj_group, group_subjects, group_name):
            shuffle_means = []
            for shuffle_idx in range(N_SHUFFLES):
                rng = np.random.RandomState(SHUFFLE_BASE_SEED + shuffle_idx)
                y_shuffled = y_group.copy()
                for s in group_subjects:
                    sub_idx = np.where(subj_group == s)[0]
                    y_shuffled[sub_idx] = rng.permutation(y_group[sub_idx])

                pre_bal_folds = []
                for fc in fold_cache:
                    is_holdout = fc["is_holdout"]
                    y_train, y_k = y_shuffled[~is_holdout], y_shuffled[is_holdout]
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
                shuffle_means.append(float(np.mean(pre_bal_folds)) if pre_bal_folds else float("nan"))
                if (shuffle_idx + 1) % 10 == 0:
                    log.info(f"  [{group_name} null] shuffle {shuffle_idx+1}/{N_SHUFFLES} = {shuffle_means[-1]:.4f}")
            return np.array(shuffle_means)

        # =========================================================================
        # Run both groups: real-label within-group LOSO + within-group null.
        # =========================================================================
        results = {}
        for group_subjects, group_name in [(odd_subjects, "odd_search_first"), (even_subjects, "even_memorize_first")]:
            log.info(f"\n{'='*60}\n{group_name}: precomputing {len(group_subjects)}-fold features\n{'='*60}")
            fold_cache, y_group, subj_group = precompute_group_folds(group_subjects, group_name)

            real_pre_cal_bal, n_folds_used = run_real_label_loso(fold_cache, y_group, subj_group, group_name)
            log.info(f"[{group_name}] REAL pre_cal_balanced = {real_pre_cal_bal:.4f} (n_folds={n_folds_used})")

            null_values = run_within_group_null(fold_cache, y_group, subj_group, group_subjects, group_name)
            null_mean = float(np.mean(null_values))
            null_sd = float(np.std(null_values, ddof=1))
            pctl_lo, pctl_hi = float(np.percentile(null_values, 2.5)), float(np.percentile(null_values, 97.5))
            z = 1.959963984540054
            normal_lo, normal_hi = null_mean - z * null_sd, null_mean + z * null_sd
            outside_pctl = not (pctl_lo <= real_pre_cal_bal <= pctl_hi)
            outside_normal = not (normal_lo <= real_pre_cal_bal <= normal_hi)

            results[group_name] = {
                "n_subjects": len(group_subjects), "subjects": group_subjects,
                "real_pre_cal_balanced": real_pre_cal_bal, "n_folds_used": n_folds_used,
                "null_mean": null_mean, "null_sd": null_sd,
                "null_percentile_95ci": [pctl_lo, pctl_hi], "null_normal_approx_95ci": [normal_lo, normal_hi],
                "outside_percentile_ci": bool(outside_pctl), "outside_normal_approx_ci": bool(outside_pctl),
                "null_shuffle_values": null_values.tolist(),
            }
            log.info(f"[{group_name}] null mean={null_mean:.4f} SD={null_sd:.4f} "
                     f"percentile_95ci=[{pctl_lo:.4f},{pctl_hi:.4f}] -> outside_ci={outside_pctl}")

        # =========================================================================
        # Apply the pre-registered verdict rule.
        # =========================================================================
        odd_outside = results["odd_search_first"]["outside_percentile_ci"]
        even_outside = results["even_memorize_first"]["outside_percentile_ci"]

        if odd_outside and even_outside:
            verdict = ("BOTH groups individually fall outside their own within-group null's 95% CI -- the signal is "
                        "NOT explained by block-order/session-position alone (position and label are inversely mapped "
                        "between the two parity groups). This strengthens the task-instruction reading.")
        elif odd_outside or even_outside:
            which = "odd_search_first" if odd_outside else "even_memorize_first"
            verdict = (f"ONLY ONE group ({which}) falls outside its own within-group null's 95% CI -- named plainly "
                        "per pre-registration: this is evidence the signal MAY BE POSITION-DRIVEN rather than "
                        "task-driven. Reported even though it complicates the finding.")
        else:
            verdict = ("NEITHER group falls outside its own within-group null's 95% CI -- inconsistent with C1/C4's "
                        "full-pool result. Per pre-registration: halt and report for discussion, no post-hoc "
                        "reconciliation attempted here.")

        log.info(f"\n{'='*70}\n  D2 PRE-REGISTERED VERDICT\n{'='*70}\n  {verdict}")

        results_payload = {"groups": results, "verdict": verdict, "hardware_info": hardware_info}
        # GATE C5 STEP 3: non-overwriting artifact -- see run_c1's identical comment.
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
            "odd_search_first_pre_cal_balanced": results["odd_search_first"]["real_pre_cal_balanced"],
            "odd_search_first_null_mean": results["odd_search_first"]["null_mean"],
            "odd_search_first_outside_ci": odd_outside,
            "even_memorize_first_pre_cal_balanced": results["even_memorize_first"]["real_pre_cal_balanced"],
            "even_memorize_first_null_mean": results["even_memorize_first"]["null_mean"],
            "even_memorize_first_outside_ci": even_outside,
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
    print("D2 -- parity-split check on R2 (task-instruction vs. block-order).")
    print("Pre-registered in DECISIONS.md's 'D1 / D2' section BEFORE this run.")
    print(f"git_short_hash for this run's stamped output filename: {git_short_hash}\n")
    results = run_d2.remote(git_short_hash=git_short_hash)
    print("\nRESULTS:")
    for k, v in results.items():
        print(f"  {k}: {v}")
