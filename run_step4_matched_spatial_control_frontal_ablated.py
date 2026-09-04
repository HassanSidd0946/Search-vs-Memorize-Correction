# =============================================================================
# command to run:
#   modal run run_step4_matched_spatial_control_frontal_ablated.py::main
# run_step4_matched_spatial_control_frontal_ablated.py
#
# F-OCULAR(a) — FRONTAL-ABLATION MATCHED SPATIAL-ONLY CONTROL, NULL-
# DISTRIBUTION DESIGN (AUDIT.md D3, CRITICAL — decisive control;
# DECISIONS.md A1/A2, revised 2026-08-19)
#
# WHY THIS SCRIPT EXISTS:
#   AUDIT.md Priority 3 (ocular-artifact exposure assessment) found that
#   ds005189's raw .vhdr files declare NO dedicated EOG channel at all (the
#   BIDS sidecar's "EOGChannelCount: 1" does not match the actual 62-channel,
#   all-scalp-named montage). That means every model trained on this dataset
#   has unrestricted access to Fp1/Fp2/AF7/AF8 and the rest of the frontopolar
#   strip -- the electrodes most saturated by blink and saccade artifact.
#   AUDIT.md D3 makes this a HARD GATE: no LOSO accuracy number is to be
#   reported in Phase 2 until this frontal-ablation re-run has been completed
#   and reviewed.
#
# DESIGN (DECISIONS.md A1/A2 -- see that entry for full rationale, including
# why the FIRST version of A2 was rejected):
#   A1 (matched unablated reference): the existing 70.78% reference
#   (run_step4_matched_spatial_control.py) is NOT a valid unablated
#   comparator -- different code path, different seed list. So this script
#   runs an "unablated" arm (0 channels dropped) through the SAME shared
#   per-fold function (`run_one_arm`) as the frontal-ablated arm -- "same
#   code path" is true by construction, not by flag-matching.
#
#   A2 (random-ablation control, REVISED to a null distribution): a single
#   fixed 9-channel "random" control was tried first and rejected -- it
#   happened to land on F-OCULAR(d)'s central-parietal group, which is
#   exactly where the hypothesized COGNITIVE signal is localized (Fig. 7),
#   so ablating it would remove real signal by construction and bias the
#   verdict toward "dimensionality effect" regardless of whether ocular
#   contamination is real. The fix: draw 20 independent 9-channel sets at
#   random from a pool that excludes BOTH the frontal/ocular channels AND
#   the central-parietal signal cluster (62 - 9 - 13 = 40-channel pool),
#   using a fixed, reproducible RNG seed (20260819, recorded in
#   DECISIONS.md BEFORE this script runs). Each draw is evaluated at ONE
#   seed only (42) to keep cost bounded; `unablated`/`frontal_ablated`
#   stay at the full 5 seeds. (Originally 10 draws; raised to 20 -- at 10,
#   the smallest achievable one-sided p is ~0.09, so a frontal drop landing
#   8th/9th of 10 would be uninterpretable; 20 draws takes the floor to
#   ~0.048.) The frontal-ablation drop is then compared against the
#   DISTRIBUTION of the 20 random-draw drops (90th-percentile rule +
#   one-sided p-value, not a mean-vs-mean ratio) -- see DECISIONS.md.
#
# STANDING DEPENDENCY (DECISIONS.md, not a footnote): this script hardcodes
#   its own pre-F3 pooled-only EA and SEEDS=[42,101,202,303,404], NOT the
#   canonical F4 seed list or F3's parametrized eeg_alignment.py module.
#   The arms here are internally matched (valid ablation comparison), but
#   these accuracies are NOT comparable to any Batch 2+ number, and the
#   ocular verdict from this control is CONDITIONAL on the pre-F3
#   alignment. If Batch 2 shows F3 materially changes the spatial
#   pipeline's behavior, F-OCULAR(a) must be RE-RUN under the new
#   alignment before its verdict is treated as final.
#
# CHANNEL-ORDER SAFETY (why this isn't a hardcoded guess):
#   processed_eeg_all_subjects.npz stores ONLY X/y/subjects -- no channel
#   names are saved anywhere in the data-engine pipeline. This script
#   re-derives the authoritative column order by re-downloading ONE
#   subject's raw .vhdr (sub-01) and reading raw.info['ch_names'] directly
#   via MNE -- epochs.get_data() never reorders or renames channels, so
#   this is the true column order of X_np's channel axis, not an assumption.
#
# Usage: modal run run_step4_matched_spatial_control_frontal_ablated.py::main
# =============================================================================

import modal

app    = modal.App("bci-condition4-matched-spatial-control-frontal-ablated")
volume = modal.Volume.from_name("eeg-data-vol")

RAW_DATA_PATH = "/data/processed_eeg_all_subjects.npz"
OUTPUT_JSON   = "/data/results_condition4_matched_spatial_control_frontal_ablated.json"
VOLUME_PATH   = "/data"
PROBE_DIR     = "/data/_probe_openneuro_channel_order"

# Reference numbers from OTHER scripts/pipelines -- for orientation/logging
# ONLY. Per DECISIONS.md A1, these are NOT valid comparators for the
# ablation delta (different code path, different seeds) -- the valid
# unablated comparator is THIS script's own "unablated" arm, below.
CONDITION1B_FULL_MEAN_ACC          = 0.5564   # EEGNet + 15% calib, full 29-fold
CONDITION4V2_FULL_MEAN_ACC         = 0.6779   # original (leaky-tangent-space) spatial-only, full 29-fold
MATCHED_SPATIAL_CONTROL_MEAN_ACC   = 0.7078   # run_step4_matched_spatial_control.py (DIFFERENT code path/seeds -- not a matched comparator)
ASYMMETRIC_FUSION_MEAN_ACC         = 0.7128   # full 29-fold True-Mamba asymmetric fusion

SFREQ, N_CHANNELS = 250, 62

# AUDIT.md D3 / F-OCULAR(a) spec: frontopolar + frontal strip.
FRONTAL_ABLATION_CHANNELS = ["Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F8"]

# DECISIONS.md A2 (revised 2026-08-19): excluded from the random-draw pool
# because this is where the hypothesized cognitive signal is localized
# (Fig. 7) -- ablating any of THESE channels would remove real signal, not
# probe a "random" dimensionality effect.
CENTRAL_PARIETAL_SIGNAL_CLUSTER = ["C1", "C2", "C3", "C4", "Cz", "CP1", "CP2", "CPz", "P1", "P2", "P3", "P4", "Pz"]

# DECISIONS.md A2 (revised again 2026-08-19: 10 -> 20 draws -- at 10 draws
# the smallest achievable one-sided p is ~0.09, so a frontal drop landing
# 8th/9th of 10 is uninterpretable; 20 draws takes the floor to ~0.048 and
# stabilises the tail). Fixed, reproducible RNG -- chosen and recorded
# BEFORE this script runs, not after seeing results.
RANDOM_DRAW_RNG_SEED     = 20260819
N_RANDOM_DRAWS           = 20
N_ABLATE_PER_DRAW        = 9
RANDOM_DRAW_SINGLE_SEED  = 42   # each draw evaluated at ONE seed only (cost control)

CAL_FRACTION       = 0.15
SEEDS              = [42, 101, 202, 303, 404]   # unablated / frontal_ablated arms only
COV_SHRINKAGE       = 0.1
PCA_MAX_COMPONENTS = 35
LOGREG_C           = 1.0
SHRINK_GRID        = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS    = 3

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # scipy MUST be pinned alongside mne: unpinned scipy resolves to a
        # version that removed scipy.special.sph_harm (deprecated 1.15,
        # gone by 1.17), which mne==1.7.1 still imports at `import mne.io`
        # time -- crashes before any of our code runs. 1.14.1 matches the
        # already-proven-working pin in run_data_engine_on_modal.py and
        # run_step4_ica_cleaned_control.py's ::build stage.
        "numpy<2", "scikit-learn==1.4.2", "scipy==1.14.1",
        "mne==1.7.1", "openneuro-py==2024.2.0",
    )
)


@app.function(image=image, cpu=4.0, volumes={VOLUME_PATH: volume}, timeout=86400, memory=16384)
def run_matched_spatial_control_frontal_ablated():

    import os, glob, shutil, logging, time, math, json
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
    from sklearn.metrics import confusion_matrix, f1_score
    from scipy.stats import wilcoxon

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("condition4-matched-spatial-frontal-ablated")

    # =========================================================================
    # STEP 0: VERIFY THE TRUE CHANNEL ORDER (not assumed) — see header.
    # =========================================================================
    log.info("Verifying authoritative channel order from sub-01 raw .vhdr ...")
    import mne
    import openneuro

    mne.set_log_level("WARNING")
    os.makedirs(PROBE_DIR, exist_ok=True)
    openneuro.download(dataset="ds005189", target_dir=PROBE_DIR, include=["sub-01"])
    vhdr_files = glob.glob(os.path.join(PROBE_DIR, "**", "*.vhdr"), recursive=True)
    assert vhdr_files, f"Could not find sub-01 .vhdr under {PROBE_DIR} for channel-order verification."
    raw_probe = mne.io.read_raw_brainvision(vhdr_files[0], preload=False, verbose=False)
    verified_ch_names = list(raw_probe.info["ch_names"])
    shutil.rmtree(PROBE_DIR, ignore_errors=True)

    assert len(verified_ch_names) == N_CHANNELS, (
        f"Verified sub-01 raw channel count ({len(verified_ch_names)}) does not match "
        f"the expected {N_CHANNELS}. Channel order cannot be trusted for ablation — aborting."
    )
    missing_frontal = [c for c in FRONTAL_ABLATION_CHANNELS if c not in verified_ch_names]
    assert not missing_frontal, f"Frontal-ablation channels not found in verified montage: {missing_frontal}"
    missing_cp = [c for c in CENTRAL_PARIETAL_SIGNAL_CLUSTER if c not in verified_ch_names]
    assert not missing_cp, f"Central-parietal signal-cluster channels not found in verified montage: {missing_cp}"
    log.info(f"Verified channel order (n={len(verified_ch_names)}): {verified_ch_names}")

    # =========================================================================
    # DECISIONS.md A2: build the 40-channel random-draw pool and generate the
    # N_RANDOM_DRAWS reproducible draws from the fixed RNG seed.
    # =========================================================================
    excluded = set(FRONTAL_ABLATION_CHANNELS) | set(CENTRAL_PARIETAL_SIGNAL_CLUSTER)
    draw_pool = [c for c in verified_ch_names if c not in excluded]
    assert len(draw_pool) == N_CHANNELS - len(FRONTAL_ABLATION_CHANNELS) - len(CENTRAL_PARIETAL_SIGNAL_CLUSTER), (
        f"[A2 SANITY FAIL] expected a {N_CHANNELS - len(FRONTAL_ABLATION_CHANNELS) - len(CENTRAL_PARIETAL_SIGNAL_CLUSTER)}"
        f"-channel draw pool, got {len(draw_pool)}: {draw_pool}"
    )
    draw_rng = np.random.RandomState(RANDOM_DRAW_RNG_SEED)
    random_draws = [
        sorted(draw_rng.choice(draw_pool, size=N_ABLATE_PER_DRAW, replace=False).tolist())
        for _ in range(N_RANDOM_DRAWS)
    ]
    log.info(f"Random-draw pool (n={len(draw_pool)}, excludes frontal+central-parietal): {draw_pool}")
    for i, d in enumerate(random_draws):
        log.info(f"  draw {i}: {d}")

    # =========================================================================
    # LOAD DATA (once, shared across all arms)
    # =========================================================================
    raw = np.load(RAW_DATA_PATH, allow_pickle=True)
    X_np_full = raw["X"].astype(np.float32)
    y_np = raw["y"].astype(np.int64)
    subjects_np = raw["subjects"]
    N, C, T = X_np_full.shape
    N_CLASSES = int(y_np.max()) + 1
    assert C == N_CHANNELS and N_CLASSES == 2
    log.info(f"X: {X_np_full.shape} | Subjects total: {len(np.unique(subjects_np))}")
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
    # RIEMANNIAN / EA UTILITIES — IDENTICAL to run_step4_matched_spatial_control.py
    # (pre-F3 pooled-only EA, deliberately unchanged -- see header's standing
    # dependency note)
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
        """RAW tangent vectors, IDENTITY reference — no learned compression,
        no group Frechet mean, ever. Identical to the matched-spatial-control script."""
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

    # =========================================================================
    # CALIBRATION UTILITIES — IDENTICAL to run_step4_matched_spatial_control.py
    # =========================================================================
    def compute_binary_metrics(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        return {"sensitivity": float(sens), "specificity": float(spec),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)), "confusion_matrix": cm.tolist()}

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
    # SHARED PER-ARM RUNNER — ONE function, called once per arm with a
    # different channel-keep mask and its own seed list, so "same code path"
    # holds by construction (DECISIONS.md A1).
    # =========================================================================
    def run_one_arm(X_np_arm, arm_name, seeds_for_arm):
        seed_summaries = []
        all_fold_records = []
        all_test_acc_pooled = []

        for seed in seeds_for_arm:
            np.random.seed(seed)
            seed_start = time.time()
            fold_records, seed_test_acc = [], []

            for fold_idx, test_sub in enumerate(unique_subjects):
                fold_start = time.time()
                log.info(f"\n{'='*70}\n  [{arm_name}] SEED {seed} — FOLD {fold_idx+1}/{len(unique_subjects)} — sub-{test_sub}\n{'='*70}")

                is_holdout = subjects_np == test_sub
                X_train28, y_train28 = X_np_arm[~is_holdout], y_np[~is_holdout]
                X_k, y_k = X_np_arm[is_holdout], y_np[is_holdout]

                mu = X_train28.mean(axis=(0, 2), keepdims=True)
                sd = X_train28.std(axis=(0, 2), keepdims=True) + 1e-6
                X_train28_z = ((X_train28 - mu) / sd).astype(np.float32)
                X_k_z = ((X_k - mu) / sd).astype(np.float32)

                W = fit_ea_whitening(X_train28_z)
                X_train28_aligned = apply_ea_whitening_signal(X_train28_z, W).astype(np.float32)
                X_k_aligned = apply_ea_whitening_signal(X_k_z, W).astype(np.float32)

                # --- SPATIAL ONLY: raw tangent vector, untouched by any nn.Module, NO Mamba ---
                tan_train28 = tangent_vectorize(trial_covariances(X_train28_aligned))
                tan_k = tangent_vectorize(trial_covariances(X_k_aligned))
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
                final_preds = linear_predict(coef_final, icpt_final, X_test_pca)
                best_test_acc = float((final_preds == y_test).mean())
                metrics = compute_binary_metrics(y_test, final_preds)

                log.info(f"  RESULT -> pre_cal={pre_cal_acc:.4f}  post_cal={best_test_acc:.4f} (shrink={best_shrink:.2f}) "
                         f"| tangent_dim={tangent_dim}")

                record = {
                    "arm": arm_name, "seed": seed, "fold_index": fold_idx, "test_subject": str(test_sub),
                    "tangent_dim": int(tangent_dim), "best_shrink_weight": float(best_shrink),
                    "pre_calibration_acc": pre_cal_acc, "post_calibration_acc": best_test_acc,
                    **metrics,
                }
                fold_records.append(record)
                all_fold_records.append(record)
                seed_test_acc.append(best_test_acc)
                all_test_acc_pooled.append(best_test_acc)
                log.info(f"  Fold elapsed: {time.time()-fold_start:.0f}s")

            seed_mean, seed_std = float(np.mean(seed_test_acc)), float(np.std(seed_test_acc))
            seed_summaries.append({"seed": seed, "mean_accuracy": seed_mean, "std_accuracy": seed_std,
                                    "n_folds": len(unique_subjects)})
            log.info(f"\n  [{arm_name}] SEED {seed} DONE — mean_acc={seed_mean:.4f} ± {seed_std:.4f} "
                     f"(elapsed {time.time()-seed_start:.0f}s)")

        grand_mean_of_seed_means = float(np.mean([s["mean_accuracy"] for s in seed_summaries]))
        grand_std_of_seed_means  = float(np.std([s["mean_accuracy"] for s in seed_summaries]))
        pooled_mean = float(np.mean(all_test_acc_pooled))
        pooled_std  = float(np.std(all_test_acc_pooled))

        return {
            "arm": arm_name,
            "seeds": seeds_for_arm,
            "seed_summaries": seed_summaries,
            "fold_results": all_fold_records,
            "mean_of_seed_means_accuracy": grand_mean_of_seed_means,
            "std_of_seed_means_accuracy": grand_std_of_seed_means,
            "pooled_mean_accuracy": pooled_mean,
            "pooled_std_accuracy": pooled_std,
        }

    # =========================================================================
    # RUN ALL ARMS — same code path, only channel-keep mask + seed list vary.
    # unablated/frontal_ablated: full 5 seeds. Each random draw: 1 seed (42).
    # =========================================================================
    def keep_idx_for(drop_channels):
        drop_idx = sorted(verified_ch_names.index(c) for c in drop_channels)
        return [i for i in range(N_CHANNELS) if i not in drop_idx], drop_idx

    arm_results = {}
    arm_channel_info = {}

    for arm_name, drop_channels, seeds_for_arm in [
        ("unablated", [], SEEDS),
        ("frontal_ablated", FRONTAL_ABLATION_CHANNELS, SEEDS),
    ] + [
        (f"random_draw_{i}", random_draws[i], [RANDOM_DRAW_SINGLE_SEED]) for i in range(N_RANDOM_DRAWS)
    ]:
        keep_idx, drop_idx = keep_idx_for(drop_channels)
        X_np_arm = X_np_full[:, keep_idx, :]
        arm_channel_info[arm_name] = {
            "dropped_channels": drop_channels,
            "dropped_indices_in_verified_order": drop_idx,
            "n_channels_kept": len(keep_idx),
        }
        log.info(f"\n{'#'*70}\n  ARM: {arm_name} — dropping {len(drop_channels)} channels {drop_channels} "
                 f"(keeping {len(keep_idx)}/{N_CHANNELS}), seeds={seeds_for_arm}\n{'#'*70}")
        arm_results[arm_name] = run_one_arm(X_np_arm, arm_name, seeds_for_arm)

    # =========================================================================
    # A1: per-subject paired differences + Wilcoxon, frontal_ablated vs
    # unablated (both 5-seed arms -- seed-average per subject first, matching
    # F13's run_statistics.py convention, then pair by subject).
    # =========================================================================
    def per_subject_mean_acc(fold_records):
        by_subject = {str(s): [] for s in unique_subjects}
        for r in fold_records:
            by_subject[r["test_subject"]].append(r["post_calibration_acc"])
        return {s: float(np.mean(v)) for s, v in by_subject.items()}

    subj_order = [str(s) for s in unique_subjects]
    unablated_subj_acc = per_subject_mean_acc(arm_results["unablated"]["fold_results"])
    frontal_subj_acc = per_subject_mean_acc(arm_results["frontal_ablated"]["fold_results"])
    unablated_vec = np.array([unablated_subj_acc[s] for s in subj_order])
    frontal_vec = np.array([frontal_subj_acc[s] for s in subj_order])
    frontal_diff = frontal_vec - unablated_vec

    def safe_wilcoxon(a, b):
        try:
            stat, p = wilcoxon(a, b)
            return {"statistic": float(stat), "p_value": float(p)}
        except Exception as e:
            return {"error": str(e)}

    frontal_wilcoxon = safe_wilcoxon(frontal_vec, unablated_vec)
    paired_differences = {
        "frontal_ablated_minus_unablated": {
            "per_subject": {s: float(d) for s, d in zip(subj_order, frontal_diff)},
            "mean": float(frontal_diff.mean()), "std": float(frontal_diff.std()),
            "wilcoxon_vs_unablated": frontal_wilcoxon,
        },
    }

    # =========================================================================
    # A2 (revised, DECISIONS.md): frontal-ablation drop vs. the DISTRIBUTION
    # of N_RANDOM_DRAWS random-draw drops, all on the SAME single-seed (seed=42) basis
    # so the comparison is apples-to-apples with the single-seed draws.
    # =========================================================================
    def seed_acc(arm_name, seed):
        for s in arm_results[arm_name]["seed_summaries"]:
            if s["seed"] == seed:
                return s["mean_accuracy"]
        raise KeyError(f"seed {seed} not found in arm {arm_name}'s seed_summaries")

    unablated_seed42_acc = seed_acc("unablated", RANDOM_DRAW_SINGLE_SEED)
    frontal_seed42_acc = seed_acc("frontal_ablated", RANDOM_DRAW_SINGLE_SEED)
    frontal_drop_seed42 = unablated_seed42_acc - frontal_seed42_acc

    random_draw_accs = [seed_acc(f"random_draw_{i}", RANDOM_DRAW_SINGLE_SEED) for i in range(N_RANDOM_DRAWS)]
    random_draw_drops = [unablated_seed42_acc - a for a in random_draw_accs]

    random_drop_90th_pct = float(np.percentile(random_draw_drops, 90))
    n_random_drops_below_frontal = int(sum(1 for d in random_draw_drops if d < frontal_drop_seed42))
    frontal_percentile_within_random_distribution = 100.0 * n_random_drops_below_frontal / N_RANDOM_DRAWS

    # One-sided permutation-style p-value (add-one correction, same convention
    # as scripts/verify_no_leakage.py's permutation_null_stats): fraction of
    # random draws whose drop is AT LEAST as extreme as frontal's, +1 in
    # numerator/denominator so p can never be reported as exactly 0. At
    # N_RANDOM_DRAWS=20 the floor is 1/21 ~= 0.0476.
    n_random_drops_at_least_as_extreme = int(sum(1 for d in random_draw_drops if d >= frontal_drop_seed42))
    a2_one_sided_p_value = (n_random_drops_at_least_as_extreme + 1) / (N_RANDOM_DRAWS + 1)

    # BUG FIX (2026-08-19, caught on the real run): the interpretation used to
    # gate on `frontal_drop_seed42 > random_drop_90th_pct`, comparing frontal's
    # raw drop against `np.percentile(..., 90)`'s INTERPOLATED value. At small
    # N that interpolated value can sit strictly between two draws' actual
    # values, so a frontal drop that is exactly AT the 90th percentile BY RANK
    # (e.g. beaten by only 2 of 20 draws) can still numerically exceed the
    # interpolated value and get misclassified as "beyond" it. The rank-based
    # one-sided p-value computed above does not have this ambiguity, so it is
    # now the actual gating statistic; "beyond the 90th percentile" (one-sided
    # alpha=0.10) is implemented as p < 0.10, and the interpretation string
    # always reports BOTH the percentile rank and the p-value together, never
    # a bare verdict.
    a2_significant = a2_one_sided_p_value < 0.10
    a2_interpretation = (
        (f"frontal-specific effect, consistent with ocular contamination "
         f"(frontal is at the {frontal_percentile_within_random_distribution:.0f}th percentile of the "
         f"{N_RANDOM_DRAWS} random-draw drops, one-sided p={a2_one_sided_p_value:.3f} < 0.10)")
        if a2_significant else
        (f"no significant evidence of frontal-specific contamination "
         f"(frontal is at the {frontal_percentile_within_random_distribution:.0f}th percentile of the "
         f"{N_RANDOM_DRAWS} random-draw drops, one-sided p={a2_one_sided_p_value:.3f}, NOT < 0.10 -- "
         f"a random 9-channel drop of this size or larger is unremarkable under the null)")
    )

    # Across-seed std (unablated/frontal_ablated, 5 seeds) vs. across-draw std
    # (random null, 20 single-seed draws) -- reported side by side because the
    # null's spread mixes channel-selection variance with single-seed noise
    # that the 5-seed arms' std does not contain; conflating the two would
    # misread ordinary seed jitter as a channel-driven effect (per instruction).
    noise_decomposition = {
        "unablated_std_across_5_seeds": arm_results["unablated"]["std_of_seed_means_accuracy"],
        "frontal_ablated_std_across_5_seeds": arm_results["frontal_ablated"]["std_of_seed_means_accuracy"],
        "random_draw_accs_std_across_20_draws_single_seed": float(np.std(random_draw_accs)),
        "random_draw_drops_std_across_20_draws_single_seed": float(np.std(random_draw_drops)),
        "note": "unablated/frontal_ablated std is across 5 SEEDS (same channels, same fold split shape, "
                "different RNG draws for calibration split/PCA/shrinkage-CV/LogReg init) -- pure seed jitter. "
                "random_draw std is across 20 different CHANNEL SELECTIONS at a SINGLE seed -- channel-"
                "selection variance ONLY, not seed jitter. The two are not directly comparable magnitudes; "
                "they are reported together so seed jitter is not mistaken for a channel-driven effect.",
    }

    a2_null_distribution = {
        "basis": f"single-seed (seed={RANDOM_DRAW_SINGLE_SEED}), pooled across 29 folds -- matches the "
                 f"random draws' single-seed evaluation",
        "unablated_acc_seed42": unablated_seed42_acc,
        "frontal_ablated_acc_seed42": frontal_seed42_acc,
        "frontal_drop_seed42": frontal_drop_seed42,
        "n_random_draws": N_RANDOM_DRAWS,
        "random_draw_channels": random_draws,
        "random_draw_accs_seed42": random_draw_accs,
        "random_draw_drops_seed42": random_draw_drops,
        "random_draw_drop_90th_percentile": random_drop_90th_pct,
        "frontal_drop_percentile_within_random_distribution": frontal_percentile_within_random_distribution,
        "one_sided_p_value": a2_one_sided_p_value,
        "rng_seed_for_draws": RANDOM_DRAW_RNG_SEED,
        "noise_decomposition": noise_decomposition,
        "interpretation": a2_interpretation,
    }

    log.info(f"\n{'='*70}\n  SUMMARY — unablated/frontal_ablated: {len(SEEDS)} seeds each; "
             f"{N_RANDOM_DRAWS} random draws @ seed={RANDOM_DRAW_SINGLE_SEED} each\n{'='*70}")
    for arm_name in ["unablated", "frontal_ablated"]:
        r = arm_results[arm_name]
        log.info(f"  [{arm_name:<16}] mean-of-seed-means={r['mean_of_seed_means_accuracy']:.4f} ± "
                 f"{r['std_of_seed_means_accuracy']:.4f} (across-5-seed std) | "
                 f"n_channels_kept={arm_channel_info[arm_name]['n_channels_kept']}")
    log.info(f"  frontal_ablated - unablated (5-seed paired): mean diff = "
             f"{paired_differences['frontal_ablated_minus_unablated']['mean']:+.4f} | Wilcoxon: {frontal_wilcoxon}")
    log.info(f"  [A2, seed=42 basis] frontal_drop={frontal_drop_seed42:+.4f} | "
             f"random-draw drops (n={N_RANDOM_DRAWS})={[f'{d:+.4f}' for d in random_draw_drops]} | "
             f"90th pct={random_drop_90th_pct:+.4f} | one-sided p={a2_one_sided_p_value:.4f} | "
             f"frontal is at the {frontal_percentile_within_random_distribution:.0f}th percentile of the random distribution")
    log.info(f"  Noise decomposition: unablated std(5 seeds)={noise_decomposition['unablated_std_across_5_seeds']:.4f} | "
             f"frontal std(5 seeds)={noise_decomposition['frontal_ablated_std_across_5_seeds']:.4f} | "
             f"random-draw-accs std({N_RANDOM_DRAWS} draws, 1 seed)="
             f"{noise_decomposition['random_draw_accs_std_across_20_draws_single_seed']:.4f}")
    log.info(f"  A2 INTERPRETATION: {a2_interpretation}")
    log.info(
        "\n  ORIENTATION ONLY -- NOT valid matched comparators (different code path/seeds, DECISIONS.md A1):\n"
        f"    EEGNet + 15% Calib (full 29-fold)                 : {CONDITION1B_FULL_MEAN_ACC*100:.2f}%\n"
        f"    Condition 4v2 spatial-only, ORIGINAL/leaky pipe    : {CONDITION4V2_FULL_MEAN_ACC*100:.2f}%\n"
        f"    Matched spatial-only control (different script)   : {MATCHED_SPATIAL_CONTROL_MEAN_ACC*100:.2f}%\n"
        f"    Asymmetric Fusion (raw tangent + True-Mamba)       : {ASYMMETRIC_FUSION_MEAN_ACC*100:.2f}%\n"
    )

    results_payload = {
        "condition": "Condition 4 — MATCHED SPATIAL-ONLY CONTROL, NULL-DISTRIBUTION ABLATION DESIGN "
                      "(unablated + frontal_ablated at 5 seeds; N_RANDOM_DRAWS random_draw arms at 1 seed each; "
                      "identity-reference tangent, pool-only EA, no Mamba)",
        "n_folds": len(unique_subjects),
        "standing_dependency": "This control uses a pre-F3 hardcoded pooled-only EA and SEEDS="
                                f"{SEEDS}, NOT the canonical F4 seed list or F3's parametrized "
                                "eeg_alignment.py module. Arms here are internally matched (valid "
                                "ablation comparison), but these accuracies are NOT comparable to "
                                "Batch 2+ numbers, and the ocular verdict is conditional on the "
                                "pre-F3 alignment -- see DECISIONS.md.",
        "pipeline_settings_shared_across_all_arms": {
            "ea_implementation": "pooled-only hardcoded fit_ea_whitening/apply_ea_whitening_signal "
                                  "(pre-F3 code, unchanged across all arms)",
            "cov_estimator": "fixed", "cov_shrinkage": COV_SHRINKAGE,
            "fusion_mode": "none (spatial-only, no temporal branch, no fusion in this script)",
            "unablated_frontal_seeds": SEEDS, "random_draw_seed": RANDOM_DRAW_SINGLE_SEED,
            "cal_fraction": CAL_FRACTION, "pca_max_components": PCA_MAX_COMPONENTS,
            "logreg_C": LOGREG_C, "shrink_grid": SHRINK_GRID, "shrink_cv_folds": SHRINK_CV_FOLDS,
            "code_path": "single shared run_one_arm() function called once per arm -- only the "
                          "channel-keep mask and seed list differ between arms (DECISIONS.md A1)",
        },
        "verified_channel_order_source": "sub-01 raw .vhdr, read via mne.io.read_raw_brainvision "
                                          "(NOT the assumed row layout used by the visualization scripts)",
        "verified_channel_order": verified_ch_names,
        "random_draw_pool": {
            "pool_channels": draw_pool, "pool_size": len(draw_pool),
            "excluded_frontal": FRONTAL_ABLATION_CHANNELS,
            "excluded_central_parietal_signal_cluster": CENTRAL_PARIETAL_SIGNAL_CLUSTER,
            "rng_seed": RANDOM_DRAW_RNG_SEED, "n_draws": N_RANDOM_DRAWS, "n_channels_per_draw": N_ABLATE_PER_DRAW,
        },
        "arms": {
            arm_name: {**arm_channel_info[arm_name], **arm_results[arm_name]}
            for arm_name in arm_results
        },
        "paired_differences": paired_differences,
        "a2_null_distribution": a2_null_distribution,
        "orientation_only_not_matched_comparators": {
            "reference_condition1b_full_mean_acc": CONDITION1B_FULL_MEAN_ACC,
            "reference_condition4v2_full_mean_acc_original_leaky_pipeline": CONDITION4V2_FULL_MEAN_ACC,
            "reference_matched_spatial_control_different_script_mean_acc": MATCHED_SPATIAL_CONTROL_MEAN_ACC,
            "reference_asymmetric_fusion_mean_acc": ASYMMETRIC_FUSION_MEAN_ACC,
        },
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_payload, f, indent=2)
    volume.commit()
    log.info(f"  Saved: {OUTPUT_JSON}")

    # =========================================================================
    # C3 plausibility assertions -- printed next to the numbers, not silent.
    # Run AFTER the write above so a failing assertion never suppresses the
    # diagnostic artifact. Structural checks (subject/seed count, [0,1]
    # range) hard-fail; a below-chance mean is only WARNED on for ABLATED
    # arms (a legitimate possible outcome per DECISIONS.md), not for the
    # unablated reference (which should not collapse).
    # =========================================================================
    CHANCE_LEVEL = 0.5
    for arm_name, r in arm_results.items():
        expected_seeds = len(SEEDS) if arm_name in ("unablated", "frontal_ablated") else 1
        assert 0.0 <= r["mean_of_seed_means_accuracy"] <= 1.0, (
            f"[C3 PLAUSIBILITY FAIL] [{arm_name}] mean-of-seed-means accuracy "
            f"{r['mean_of_seed_means_accuracy']} outside [0,1]"
        )
        assert 0.0 <= r["pooled_mean_accuracy"] <= 1.0, (
            f"[C3 PLAUSIBILITY FAIL] [{arm_name}] pooled accuracy {r['pooled_mean_accuracy']} outside [0,1]"
        )
        assert len(r["seed_summaries"]) == expected_seeds, (
            f"[C3 PLAUSIBILITY FAIL] [{arm_name}] expected {expected_seeds} seed summaries, "
            f"got {len(r['seed_summaries'])}"
        )
        if arm_name == "unablated" and r["mean_of_seed_means_accuracy"] <= CHANCE_LEVEL:
            log.warning(
                f"  [C3] plausibility WARNING: the UNABLATED reference arm's accuracy "
                f"({r['mean_of_seed_means_accuracy']:.4f}) is at or below chance -- this arm is not "
                f"expected to collapse; unlike the ablated arms this is worth investigating as a bug."
            )
        elif r["mean_of_seed_means_accuracy"] <= CHANCE_LEVEL:
            log.warning(
                f"  [C3] plausibility WARNING: [{arm_name}] mean accuracy "
                f"({r['mean_of_seed_means_accuracy']:.4f}) is at or below chance. For an ABLATED arm "
                f"that is a legitimate possible outcome -- read against DECISIONS.md's interpretation "
                f"rules, do not treat this print alone as a bug."
            )
    assert len(unique_subjects) == 29, f"[C3 PLAUSIBILITY FAIL] expected 29 unique subjects, got {len(unique_subjects)}"
    assert len(SEEDS) == 5, f"[C3 PLAUSIBILITY FAIL] expected 5 seeds for unablated/frontal_ablated, got {len(SEEDS)}"
    assert len(random_draws) == N_RANDOM_DRAWS, (
        f"[C3 PLAUSIBILITY FAIL] expected {N_RANDOM_DRAWS} random draws, got {len(random_draws)}"
    )
    for i, d in enumerate(random_draws):
        assert len(d) == N_ABLATE_PER_DRAW, (
            f"[C3 PLAUSIBILITY FAIL] random draw {i} has {len(d)} channels, expected {N_ABLATE_PER_DRAW}"
        )
    log.info(f"  [C3] plausibility: 2 five-seed arms + {N_RANDOM_DRAWS} single-seed random draws, "
             f"29/29 subjects, all accuracies in [0,1], all draws have {N_ABLATE_PER_DRAW} channels -- OK")

    return {
        "unablated_mean_of_seed_means": arm_results["unablated"]["mean_of_seed_means_accuracy"],
        "frontal_ablated_mean_of_seed_means": arm_results["frontal_ablated"]["mean_of_seed_means_accuracy"],
        "frontal_minus_unablated_mean_diff_5seed": paired_differences["frontal_ablated_minus_unablated"]["mean"],
        "frontal_drop_seed42": frontal_drop_seed42,
        "random_draw_drop_90th_percentile": random_drop_90th_pct,
        "frontal_percentile_within_random_distribution": frontal_percentile_within_random_distribution,
        "a2_one_sided_p_value": a2_one_sided_p_value,
        "a2_interpretation": a2_interpretation,
        "output_path": OUTPUT_JSON,
        "n_folds": len(unique_subjects),
    }


@app.local_entrypoint()
def main():
    print("F-OCULAR(a) — Condition 4 MATCHED SPATIAL-ONLY CONTROL, NULL-DISTRIBUTION ABLATION DESIGN")
    print(f"Arms: unablated (5 seeds) / frontal_ablated (5 seeds) / {N_RANDOM_DRAWS}x random_draw (1 seed each)")
    print(f"  frontal_ablated drops: {FRONTAL_ABLATION_CHANNELS}")
    print(f"  random-draw pool excludes frontal + central-parietal signal cluster "
          f"({CENTRAL_PARIETAL_SIGNAL_CLUSTER}); {N_RANDOM_DRAWS} draws from RNG seed {RANDOM_DRAW_RNG_SEED}")
    print("Channel order verified live from sub-01's raw .vhdr (not assumed).")
    n_fold_execs = (len(SEEDS) * 2 + N_RANDOM_DRAWS * 1) * 29
    print(f"Cost: (5+5)x29 + {N_RANDOM_DRAWS}x1x29 = {n_fold_execs} fold-executions -- est. ~4-4.5 hours.\n")
    results = run_matched_spatial_control_frontal_ablated.remote()
    print("\nNULL-DISTRIBUTION FRONTAL-ABLATION RESULTS:")
    for k, v in results.items():
        print(f"  {k:<42}: {v}")
