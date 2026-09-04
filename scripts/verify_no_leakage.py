# =============================================================================
# command to run:
#   modal run scripts/verify_no_leakage.py::main
# scripts/verify_no_leakage.py
#
# F-LEAK — Data-leakage sanity checks (Fix-ID table, BLOCKING)
#
# WHY THIS SCRIPT EXISTS AND MUST RUN FIRST:
#   No result produced by any LOSO driver in this codebase (matched-spatial-
#   control, asymmetric-fusion, joint-dualbranch, or any of the new F3/F11/F7
#   variants) can be trusted until it is demonstrated that the pipeline
#   cannot achieve above-chance accuracy from pure noise or from labels that
#   have been decoupled from the data. This script runs three checks against
#   a self-contained copy of the matched-spatial-control LOSO pipeline
#   (byte-for-byte identical math to run_step4_matched_spatial_control.py,
#   restricted to a handful of folds for speed):
#
#   1. REAL      — real X, real y. Sanity baseline: must be well above chance
#                  (proves the harness itself is capable of learning signal;
#                  a leak-free pipeline that also can't learn anything real
#                  would trivially "pass" checks 2/3 for the wrong reason).
#   2. SHUFFLED_LABELS — real X, y globally shuffled (label-data relationship
#                  destroyed) BEFORE any fold splitting. Expected: chance
#                  level (~50%). Above-chance accuracy here means some stage
#                  of the pipeline is leaking label information into features
#                  independent of the true X-y relationship.
#   3. NOISE_FEATURES  — real y, X replaced with Gaussian noise matched to
#                  the real per-channel mean/std (so scale artifacts can't
#                  trivially explain a pass/fail). Expected: chance level.
#                  Above-chance accuracy here means some stage of the
#                  pipeline is leaking test-set-derived structure into
#                  training regardless of whether the signal is real.
#
#   4. FIT-CALL MONKEYPATCHING — on the REAL (unshuffled, unmodified) run,
#      StandardScaler.fit/fit_transform, PCA.fit/fit_transform, and
#      LogisticRegression.fit are wrapped to record the trial indices passed
#      to every single fit call. After each fold, asserts BY CONSTRUCTION
#      that no fit call ever received a trial index belonging to the held-out
#      test subject. This is a static proof for the exact code path used to
#      produce every reported number, not just a statistical inference from
#      shuffled/noise conditions.
#
# Runs a 5-fold mini-LOSO (not the full 29) for speed -- this is a
# sanity gate, not a result. Hard-fails (raises AssertionError, nonzero
# process exit under `modal run`) if any check indicates leakage.
#
# Usage: modal run scripts/verify_no_leakage.py::main
# =============================================================================

import modal

app    = modal.App("bci-verify-no-leakage")
volume = modal.Volume.from_name("eeg-data-vol")

RAW_DATA_PATH = "/data/processed_eeg_all_subjects.npz"
OUTPUT_JSON   = "/data/results_verify_no_leakage.json"
VOLUME_PATH   = "/data"

SFREQ, N_CHANNELS = 250, 62

N_LEAK_CHECK_FOLDS = 5     # mini-LOSO: first 5 subjects (sorted), not all 29 -- speed
CAL_FRACTION       = 0.15
RANDOM_SEED        = 42
COV_SHRINKAGE      = 0.1
PCA_MAX_COMPONENTS = 35
LOGREG_C           = 1.0
SHRINK_GRID        = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS    = 3

# Pass/fail thresholds
CHANCE_LEVEL              = 0.5
# DECISIONS.md pass criterion (recorded 2026-08-18, pre-registered before
# Batch 0): pooled shuffled/noise accuracy's 95% Wilson CI must contain
# CHANCE_LEVEL, AND pooled accuracy must fall within CHANCE_LEVEL +/- this band.
MEAN_BAND_HALFWIDTH       = 0.03
REAL_MUST_EXCEED_ACC      = 0.55   # sanity: real condition must clear this or the harness itself is broken

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy<2", "scikit-learn==1.4.2", "scipy")
)


@app.function(image=image, cpu=4.0, volumes={VOLUME_PATH: volume}, timeout=3600, memory=16384)
def run_leakage_checks():

    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
    from scipy.stats import binomtest
    import logging, math, json, functools

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("verify-no-leakage")

    np.random.seed(RANDOM_SEED)

    raw = np.load(RAW_DATA_PATH, allow_pickle=True)
    X_np = raw["X"].astype(np.float32)
    y_np = raw["y"].astype(np.int64)
    subjects_np = raw["subjects"]
    N, C, T = X_np.shape
    assert C == N_CHANNELS
    assert len(np.unique(subjects_np)) == 29, (
        f"Expected exactly 29 subjects (F-SILENT hardening), got {len(np.unique(subjects_np))}"
    )

    # Global trial index, stable across all conditions -- this is what the
    # fit-call monkeypatch cross-checks against.
    global_trial_idx = np.arange(N)

    unique_subjects = sorted(np.unique(subjects_np).tolist())[:N_LEAK_CHECK_FOLDS]
    log.info(f"F-LEAK mini-LOSO folds ({N_LEAK_CHECK_FOLDS} of 29): {unique_subjects}")

    # =========================================================================
    # RIEMANNIAN / EA UTILITIES -- byte-for-byte identical math to
    # run_step4_matched_spatial_control.py (pre-F3, pooled EA; F-LEAK checks
    # the harness mechanics, not the EA-mode choice, and must not itself
    # depend on F3 landing first).
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

    def fit_ea_whitening(X_train28, shrinkage=COV_SHRINKAGE):
        covs = trial_covariances(X_train28, shrinkage)
        _, W = matrix_sqrt_inv_sqrt(covs.mean(axis=0))
        return W

    def apply_ea_whitening_signal(X, W):
        return np.einsum("cd,ndt->nct", W, X)

    def tangent_vectorize(covs, eps=1e-8):
        N, Cc, _ = covs.shape
        out = np.empty((N, Cc * (Cc + 1) // 2), dtype=np.float32)
        iu = np.triu_indices(Cc)
        for n in range(N):
            eigvals, eigvecs = np.linalg.eigh(covs[n])
            eigvals = np.clip(eigvals, eps, None)
            log_mat = (eigvecs * np.log(eigvals)) @ eigvecs.T
            vec = log_mat[iu].copy()
            vec[iu[0] != iu[1]] *= math.sqrt(2.0)
            out[n] = vec
        return out

    def linear_predict(coef, intercept, X):
        return ((X @ coef.T + intercept).ravel() > 0).astype(int)

    def fit_shrinkage_classifier(X_train_pca, y_train, X_cal_pca, y_cal):
        global_clf = LogisticRegression(C=LOGREG_C, max_iter=5000, random_state=RANDOM_SEED).fit(X_train_pca, y_train)
        local_clf_full = LogisticRegression(C=LOGREG_C, max_iter=5000, random_state=RANDOM_SEED).fit(X_cal_pca, y_cal)
        n_splits = max(min(SHRINK_CV_FOLDS, np.bincount(y_cal).min()), 2)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        shrink_scores = {s: [] for s in SHRINK_GRID}
        for tr_idx, val_idx in skf.split(X_cal_pca, y_cal):
            X_tr, X_val, y_tr, y_val = X_cal_pca[tr_idx], X_cal_pca[val_idx], y_cal[tr_idx], y_cal[val_idx]
            if len(np.unique(y_tr)) < 2:
                continue
            local_fold = LogisticRegression(C=LOGREG_C, max_iter=5000, random_state=RANDOM_SEED).fit(X_tr, y_tr)
            for shrink in SHRINK_GRID:
                coef_b = shrink * local_fold.coef_ + (1 - shrink) * global_clf.coef_
                icpt_b = shrink * local_fold.intercept_ + (1 - shrink) * global_clf.intercept_
                shrink_scores[shrink].append((linear_predict(coef_b, icpt_b, X_val) == y_val).mean())
        mean_scores = {s: (np.mean(v) if v else -1.0) for s, v in shrink_scores.items()}
        best_shrink = max(mean_scores, key=mean_scores.get)
        coef_final = best_shrink * local_clf_full.coef_ + (1 - best_shrink) * global_clf.coef_
        icpt_final = best_shrink * local_clf_full.intercept_ + (1 - best_shrink) * global_clf.intercept_
        return coef_final, icpt_final, best_shrink

    # =========================================================================
    # Fit-call monkeypatch — records (n_rows, source_indices) for every fit
    # call made during the REAL condition, so we can assert by construction
    # that no fit() ever touches held-out test-subject trial rows.
    # =========================================================================
    fit_call_log = []          # list of dicts: {"fold": int, "call": str, "n_rows": int}
    fit_call_violations = []   # populated if a fit call's tagged indices include test-subject rows
    _current_fold_state = {"test_subject_row_ids": None, "fold_idx": None}

    def _tag_rows(X, row_ids):
        """Attach the originating global trial indices to an array via a
        side-table keyed by array id() -- avoids mutating the numeric arrays
        sklearn operates on."""
        _row_id_registry[id(X)] = row_ids
        return X

    _row_id_registry = {}

    def _check_call(name, X):
        row_ids = _row_id_registry.get(id(X))
        n_rows = X.shape[0]
        entry = {"fold": _current_fold_state["fold_idx"], "call": name, "n_rows": int(n_rows)}
        fit_call_log.append(entry)
        if row_ids is not None and _current_fold_state["test_subject_row_ids"] is not None:
            leaked = set(row_ids) & set(_current_fold_state["test_subject_row_ids"])
            if leaked:
                # Cast to native int -- row_ids/leaked may hold numpy int64
                # scalars (from untagged numpy-array set() construction),
                # which json.dump cannot serialize.
                fit_call_violations.append({**entry, "leaked_row_ids": sorted(int(x) for x in leaked)})

    def make_patched(orig_fn, name):
        @functools.wraps(orig_fn)
        def wrapper(self, X, *args, **kwargs):
            _check_call(name, X)
            return orig_fn(self, X, *args, **kwargs)
        return wrapper

    orig_scaler_fit = StandardScaler.fit
    orig_scaler_fit_transform = StandardScaler.fit_transform
    orig_pca_fit = PCA.fit
    orig_pca_fit_transform = PCA.fit_transform
    orig_logreg_fit = LogisticRegression.fit

    StandardScaler.fit = make_patched(orig_scaler_fit, "StandardScaler.fit")
    StandardScaler.fit_transform = make_patched(orig_scaler_fit_transform, "StandardScaler.fit_transform")
    PCA.fit = make_patched(orig_pca_fit, "PCA.fit")
    PCA.fit_transform = make_patched(orig_pca_fit_transform, "PCA.fit_transform")
    LogisticRegression.fit = make_patched(orig_logreg_fit, "LogisticRegression.fit")

    def unpatch():
        StandardScaler.fit = orig_scaler_fit
        StandardScaler.fit_transform = orig_scaler_fit_transform
        PCA.fit = orig_pca_fit
        PCA.fit_transform = orig_pca_fit_transform
        LogisticRegression.fit = orig_logreg_fit

    # =========================================================================
    # One LOSO fold of the matched-spatial-control pipeline, parametrized by
    # which X/y arrays to use (real / shuffled-label / noise-feature), and
    # whether to tag rows for the fit-call monkeypatch (only on "real").
    # =========================================================================
    def run_fold(X_all, y_all, test_sub, fold_idx, tag_rows_for_leak_check=False):
        is_holdout = subjects_np == test_sub
        train_row_ids = global_trial_idx[~is_holdout]
        holdout_row_ids = global_trial_idx[is_holdout]

        X_train28, y_train28 = X_all[~is_holdout], y_all[~is_holdout]
        X_k, y_k = X_all[is_holdout], y_all[is_holdout]

        mu = X_train28.mean(axis=(0, 2), keepdims=True)
        sd = X_train28.std(axis=(0, 2), keepdims=True) + 1e-6
        X_train28_z = ((X_train28 - mu) / sd).astype(np.float32)
        X_k_z = ((X_k - mu) / sd).astype(np.float32)

        W = fit_ea_whitening(X_train28_z)
        X_train28_aligned = apply_ea_whitening_signal(X_train28_z, W).astype(np.float32)
        X_k_aligned = apply_ea_whitening_signal(X_k_z, W).astype(np.float32)

        tan_train28 = tangent_vectorize(trial_covariances(X_train28_aligned))
        tan_k = tangent_vectorize(trial_covariances(X_k_aligned))

        sss = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - CAL_FRACTION), random_state=RANDOM_SEED)
        cal_idx, test_idx = next(sss.split(tan_k, y_k))
        feat_cal, y_cal = tan_k[cal_idx], y_k[cal_idx]
        feat_test, y_test = tan_k[test_idx], y_k[test_idx]
        cal_row_ids = holdout_row_ids[cal_idx]
        test_row_ids = holdout_row_ids[test_idx]

        if tag_rows_for_leak_check:
            _current_fold_state["fold_idx"] = fold_idx
            # FORBIDDEN rows = ONLY the held-out subject's final TEST split
            # (D_test), not the calibration split (D_cal). D_cal is
            # LEGITIMATELY used to fit local_clf_full/local_fold inside
            # fit_shrinkage_classifier -- that is the calibration-shrinkage
            # mechanism itself, already reviewed as leak-safe in AUDIT.md's
            # Q5 finding ("internal CV on D_cal only, never touching
            # D_test"). Treating D_cal as forbidden here would flag that
            # legitimate fit as a false-positive violation (confirmed via a
            # real Modal run 2026-08-18 -- see AUDIT.md correction).
            _current_fold_state["test_subject_row_ids"] = set(test_row_ids.tolist())
            _tag_rows(tan_train28, train_row_ids)
            _tag_rows(feat_cal, cal_row_ids)

        scaler = StandardScaler()
        feat_train28_z = scaler.fit_transform(tan_train28)
        feat_cal_z = scaler.transform(feat_cal)
        feat_test_z = scaler.transform(feat_test)

        n_components = min(PCA_MAX_COMPONENTS, feat_train28_z.shape[1] - 1, feat_train28_z.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
        if tag_rows_for_leak_check:
            _tag_rows(feat_train28_z, train_row_ids)
        X_train28_pca = pca.fit_transform(feat_train28_z)
        X_cal_pca = pca.transform(feat_cal_z)
        X_test_pca = pca.transform(feat_test_z)

        if tag_rows_for_leak_check:
            _tag_rows(X_train28_pca, train_row_ids)
            _tag_rows(X_cal_pca, cal_row_ids)

        coef_final, icpt_final, best_shrink = fit_shrinkage_classifier(
            X_train28_pca, y_train28, X_cal_pca, y_cal)
        final_preds = linear_predict(coef_final, icpt_final, X_test_pca)
        acc = float((final_preds == y_test).mean())

        if tag_rows_for_leak_check:
            _current_fold_state["test_subject_row_ids"] = None
            _current_fold_state["fold_idx"] = None

        return acc, int(len(y_test))

    # =========================================================================
    # Condition 1: REAL (with fit-call monkeypatch active)
    # =========================================================================
    log.info("=" * 70)
    log.info("CONDITION 1/3: REAL (X real, y real) -- fit-call monkeypatch active")
    log.info("=" * 70)
    real_accs, real_ns = [], []
    for fold_idx, test_sub in enumerate(unique_subjects):
        acc, n_test = run_fold(X_np, y_np, test_sub, fold_idx, tag_rows_for_leak_check=True)
        log.info(f"  [REAL] fold {fold_idx} (sub-{test_sub}): acc={acc:.4f} n_test={n_test}")
        real_accs.append(acc)
        real_ns.append(n_test)

    unpatch()  # done recording; conditions 2/3 don't need the overhead/side-table growth

    # =========================================================================
    # Condition 2: SHUFFLED LABELS (X real, y globally shuffled BEFORE split)
    # =========================================================================
    log.info("=" * 70)
    log.info("CONDITION 2/3: SHUFFLED_LABELS (X real, y globally shuffled)")
    log.info("=" * 70)
    rng = np.random.RandomState(RANDOM_SEED)
    y_shuffled = y_np.copy()
    rng.shuffle(y_shuffled)
    shuffled_accs, shuffled_ns = [], []
    for fold_idx, test_sub in enumerate(unique_subjects):
        acc, n_test = run_fold(X_np, y_shuffled, test_sub, fold_idx, tag_rows_for_leak_check=False)
        log.info(f"  [SHUFFLED_LABELS] fold {fold_idx} (sub-{test_sub}): acc={acc:.4f} n_test={n_test}")
        shuffled_accs.append(acc)
        shuffled_ns.append(n_test)

    # =========================================================================
    # Condition 3: NOISE FEATURES (y real, X replaced by matched-moment Gaussian noise)
    # =========================================================================
    log.info("=" * 70)
    log.info("CONDITION 3/3: NOISE_FEATURES (y real, X replaced by Gaussian noise)")
    log.info("=" * 70)
    ch_mu = X_np.mean(axis=(0, 2), keepdims=True)
    ch_sd = X_np.std(axis=(0, 2), keepdims=True) + 1e-6
    X_noise = rng.normal(loc=0.0, scale=1.0, size=X_np.shape).astype(np.float32) * ch_sd + ch_mu
    noise_accs, noise_ns = [], []
    for fold_idx, test_sub in enumerate(unique_subjects):
        acc, n_test = run_fold(X_noise, y_np, test_sub, fold_idx, tag_rows_for_leak_check=False)
        log.info(f"  [NOISE_FEATURES] fold {fold_idx} (sub-{test_sub}): acc={acc:.4f} n_test={n_test}")
        noise_accs.append(acc)
        noise_ns.append(n_test)

    # =========================================================================
    # Verdict
    #
    # DECISIONS.md's pre-registered F-LEAK pass criterion (recorded 2026-08-18,
    # BEFORE this script is run on real data): for SHUFFLED_LABELS and
    # NOISE_FEATURES, using the POOLED fold-level accuracy (total_correct /
    # total_n across the 5 mini-LOSO folds), (1) the 95% Wilson score CI on
    # that pooled accuracy must contain 0.50, AND (2) the pooled mean must
    # fall within 0.50 +/- 0.03. This replaces the earlier ad hoc
    # (mean_acc < 0.60 AND binomial p >= 0.05) criterion -- the binomial
    # p-value is retained below as an auxiliary diagnostic, not a gate.
    # =========================================================================
    def pooled_stats(accs, ns):
        total_n = sum(ns)
        total_correct = int(round(sum(a * n for a, n in zip(accs, ns))))
        pooled_acc = total_correct / total_n
        return pooled_acc, total_correct, total_n

    def wilson_ci(k, n, z=1.959963984540054):   # z for 95% two-sided
        if n == 0:
            return 0.0, 1.0
        phat = k / n
        denom = 1.0 + (z ** 2) / n
        center = (phat + (z ** 2) / (2 * n)) / denom
        margin = (z * math.sqrt((phat * (1 - phat) / n) + (z ** 2) / (4 * n ** 2))) / denom
        return max(0.0, center - margin), min(1.0, center + margin)

    def binomial_pvalue_vs_chance(k, n):
        return float(binomtest(k, n, CHANCE_LEVEL, alternative="greater").pvalue)

    real_mean = float(np.mean(real_accs))
    shuffled_mean = float(np.mean(shuffled_accs))
    noise_mean = float(np.mean(noise_accs))

    shuffled_pooled_acc, shuffled_correct, shuffled_total = pooled_stats(shuffled_accs, shuffled_ns)
    noise_pooled_acc, noise_correct, noise_total = pooled_stats(noise_accs, noise_ns)
    shuffled_ci_lo, shuffled_ci_hi = wilson_ci(shuffled_correct, shuffled_total)
    noise_ci_lo, noise_ci_hi = wilson_ci(noise_correct, noise_total)
    shuffled_pval = binomial_pvalue_vs_chance(shuffled_correct, shuffled_total)
    noise_pval = binomial_pvalue_vs_chance(noise_correct, noise_total)

    def chance_band_and_ci_pass(pooled_acc, ci_lo, ci_hi):
        ci_contains_chance = ci_lo <= CHANCE_LEVEL <= ci_hi
        mean_in_band = abs(pooled_acc - CHANCE_LEVEL) <= MEAN_BAND_HALFWIDTH
        return ci_contains_chance and mean_in_band

    checks = {
        "real_condition_sanity": {
            "pass": real_mean >= REAL_MUST_EXCEED_ACC,
            "mean_acc": real_mean,
            "threshold": REAL_MUST_EXCEED_ACC,
            "meaning": "Harness must be able to learn real signal at all, else checks 2/3 are vacuous.",
        },
        "shuffled_labels_no_leak": {
            "pass": chance_band_and_ci_pass(shuffled_pooled_acc, shuffled_ci_lo, shuffled_ci_hi),
            "mean_acc_per_fold": shuffled_mean,
            "pooled_acc": shuffled_pooled_acc,
            "wilson_95ci": [shuffled_ci_lo, shuffled_ci_hi],
            "mean_band": [CHANCE_LEVEL - MEAN_BAND_HALFWIDTH, CHANCE_LEVEL + MEAN_BAND_HALFWIDTH],
            "binomial_pvalue_vs_chance_auxiliary_only": shuffled_pval,
            "pooled_correct_of_total": f"{shuffled_correct}/{shuffled_total}",
        },
        "noise_features_no_leak": {
            "pass": chance_band_and_ci_pass(noise_pooled_acc, noise_ci_lo, noise_ci_hi),
            "mean_acc_per_fold": noise_mean,
            "pooled_acc": noise_pooled_acc,
            "wilson_95ci": [noise_ci_lo, noise_ci_hi],
            "mean_band": [CHANCE_LEVEL - MEAN_BAND_HALFWIDTH, CHANCE_LEVEL + MEAN_BAND_HALFWIDTH],
            "binomial_pvalue_vs_chance_auxiliary_only": noise_pval,
            "pooled_correct_of_total": f"{noise_correct}/{noise_total}",
        },
        "fit_call_monkeypatch_no_leak": {
            "pass": len(fit_call_violations) == 0,
            "n_fit_calls_recorded": len(fit_call_log),
            "n_violations": len(fit_call_violations),
            "violations": fit_call_violations,
        },
    }

    overall_pass = all(c["pass"] for c in checks.values())

    log.info("=" * 70)
    log.info(f"  F-LEAK VERDICT: {'PASS' if overall_pass else 'FAIL -- LEAKAGE DETECTED'}")
    log.info("=" * 70)
    for name, c in checks.items():
        log.info(f"  {name}: {'PASS' if c['pass'] else 'FAIL'}")
    log.info(f"  shuffled: pooled_acc={shuffled_pooled_acc:.4f} 95%CI=[{shuffled_ci_lo:.4f},{shuffled_ci_hi:.4f}] "
             f"(criterion: CI contains 0.50 AND pooled_acc in [0.47,0.53])")
    log.info(f"  noise   : pooled_acc={noise_pooled_acc:.4f} 95%CI=[{noise_ci_lo:.4f},{noise_ci_hi:.4f}] "
             f"(criterion: CI contains 0.50 AND pooled_acc in [0.47,0.53])")

    results_payload = {
        "description": "F-LEAK: shuffled-label / noise-feature / fit-call-monkeypatch data-leakage sanity checks",
        "n_folds": N_LEAK_CHECK_FOLDS,
        "folds_used": [str(s) for s in unique_subjects],
        "note": "Mini-LOSO (5 of 29 folds) for speed -- this is a gate, not a headline result.",
        "pass_criterion": "DECISIONS.md (recorded 2026-08-18, pre-registered before Batch 0): pooled accuracy's "
                           "95% Wilson CI must contain 0.50 AND pooled accuracy must be within 0.50 +/- 0.03, "
                           "for both shuffled-labels and noise-features conditions.",
        "real_condition": {"per_fold_acc": real_accs, "mean_acc": real_mean},
        "shuffled_labels_condition": {"per_fold_acc": shuffled_accs, "mean_acc_per_fold": shuffled_mean,
                                       "pooled_acc": shuffled_pooled_acc, "wilson_95ci": [shuffled_ci_lo, shuffled_ci_hi]},
        "noise_features_condition": {"per_fold_acc": noise_accs, "mean_acc_per_fold": noise_mean,
                                      "pooled_acc": noise_pooled_acc, "wilson_95ci": [noise_ci_lo, noise_ci_hi]},
        "checks": checks,
        "overall_pass": overall_pass,
    }

    # Write results to disk BEFORE any assertion below can raise, so a failed
    # run still leaves a diagnostic artifact on the volume instead of nothing.
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_payload, f, indent=2)
    volume.commit()
    log.info(f"  Saved: {OUTPUT_JSON}")

    # C3 plausibility assertions -- printed next to the numbers, not silent.
    # Run AFTER the write above (see comment there) so these never suppress
    # the diagnostic artifact if one of them fails.
    for cname, m in (("real", real_mean), ("shuffled_labels", shuffled_mean), ("noise_features", noise_mean)):
        assert 0.0 <= m <= 1.0, f"[C3 PLAUSIBILITY FAIL] {cname} mean accuracy {m} outside [0,1]"
    assert len(np.unique(subjects_np)) == 29, (
        f"[C3 PLAUSIBILITY FAIL] expected 29 unique subjects in the loaded dataset, "
        f"got {len(np.unique(subjects_np))}"
    )
    assert len(unique_subjects) == N_LEAK_CHECK_FOLDS, (
        f"[C3 PLAUSIBILITY FAIL] expected exactly {N_LEAK_CHECK_FOLDS} mini-LOSO folds, "
        f"got {len(unique_subjects)}"
    )
    log.info(
        f"  [C3] plausibility: real/shuffled/noise means in [0,1] -- OK "
        f"(real={real_mean:.4f} shuffled={shuffled_mean:.4f} noise={noise_mean:.4f}); "
        f"29/29 subjects loaded; {len(unique_subjects)}/{N_LEAK_CHECK_FOLDS} mini-LOSO folds run"
    )

    if not overall_pass:
        failing = [name for name, c in checks.items() if not c["pass"]]
        raise AssertionError(
            f"F-LEAK FAILED -- leakage indicators found in: {failing}. "
            f"No downstream Phase 2 result may be trusted until this is resolved. "
            f"See {OUTPUT_JSON} for full detail (already written to disk above)."
        )

    return {"overall_pass": overall_pass, "output_path": OUTPUT_JSON,
            "real_mean_acc": real_mean, "shuffled_pooled_acc": shuffled_pooled_acc, "noise_pooled_acc": noise_pooled_acc}


@app.local_entrypoint()
def main():
    print("F-LEAK -- data-leakage sanity checks (shuffled-label / noise-feature / fit-call monkeypatch)")
    print("BLOCKING: this must pass before any other Phase 2 result is trusted.\n")
    results = run_leakage_checks.remote()
    print("\nF-LEAK RESULTS:")
    for k, v in results.items():
        print(f"  {k:<20}: {v}")
