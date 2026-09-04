# =============================================================================
# command to run:
#   modal run run_baselines_mdm_tssvm_tslda_csp.py::pilot
#   modal run run_baselines_mdm_tssvm_tslda_csp.py::full
#   modal run run_baselines_mdm_tssvm_tslda_csp.py::full --ea-mode per-subject --cov-estimator lwf
#   modal run run_baselines_mdm_tssvm_tslda_csp.py::full --run-spdnet False
# run_baselines_mdm_tssvm_tslda_csp.py
#
# F5 — CLASSICAL BCI BASELINES (Audit Fix, AUDIT.md Fix-ID table row F5)
#
# WHY THIS SCRIPT EXISTS:
#   Every reported number in this codebase so far (EEGNet, tangent-space +
#   shrinkage-calibrated LogReg, the Mamba fusion architectures) is a modern
#   deep/hybrid method. None of them has been benchmarked against the
#   decades-standard classical BCI-transfer pipelines a reviewer will expect
#   to see: Minimum Distance to Mean (MDM), tangent-space + SVM, tangent-
#   space + shrinkage-LDA, and classic supervised CSP+LDA. Without these,
#   the paper's implicit claim ("Mamba adds value") is unfalsifiable -- there
#   is no classical floor to show the Mamba/tangent-space results actually
#   clear. This script adds that floor, plus a time-boxed, honestly-allowed-
#   to-fail attempt at an SPDNet-style geometric-deep baseline (Huang & Van
#   Gool, 2017), per AUDIT.md's F5 row.
#
# NAMING CORRECTION (DECISIONS.md C1, 2026-08-18): this baseline was
# originally mislabeled "TSMNet" in this script, STATUS.md, and AUDIT.md.
# It implements BiMap/ReEig/LogEig, which is SPDNet (Huang & Van Gool 2017),
# NOT TSMNet (Kobler et al. 2022) -- TSMNet's defining component, SPDDSMBN
# (SPD domain-specific momentum batch normalization), is absent from this
# implementation. Renamed everywhere to "SPDNet-style geometric baseline
# (our implementation)"; TSMNet itself was not re-implemented -- doing so
# properly means integrating the official github.com/rkobler/TSMNet code
# (which requires the `geoopt` package), and this codebase's convention is
# to avoid new, unverified pip dependencies (see the pyriemann/mne deviation
# note directly below) rather than ship an approximation under the original
# authors' name.
#
# DEVIATION FROM AUDIT.md's LITERAL SUGGESTION ("pyriemann-based classifiers"):
#   AUDIT.md's F5 row suggests building these on the `pyriemann` package.
#   This script instead HAND-IMPLEMENTS MDM (AIRM-distance nearest-centroid)
#   and CSP (generalized-eigenvalue spatial filters) directly on top of the
#   ALREADY-UNIT-TESTED SPD math in eeg_alignment.py (matrix_log,
#   matrix_sqrt_inv_sqrt, riemannian_mean_covariance, tangent_vectorize),
#   rather than adding pyriemann or mne as new, unverified pip dependencies.
#   This mirrors the same reasoning already applied elsewhere in this
#   codebase (EEGNet's max-norm weight clamp is hand-rolled instead of
#   depending on a Keras-parity shim). TS-SVM and TS-LDA(shrinkage) use
#   eeg_alignment.tangent_vectorize() (identity reference, leak-safe, the
#   same tangent features every other tangent-space driver in this repo
#   uses) feeding standard scikit-learn SVC / LinearDiscriminantAnalysis.
#
# EVALUATION PROTOCOL: ZERO-SHOT LOSO (no 15% calibration split).
#   The four classical baselines and the SPDNet-style baseline are evaluated
#   zero-shot -- the classifier is fit on the 28-subject training pool ONLY
#   and tested on ALL of held-out subject k's trials, with no per-subject
#   calibration stage. This matches classical BCI-transfer literature
#   convention (MDM/CSP+LDA papers report zero-shot cross-subject transfer,
#   not calibrated transfer) and gives a fair floor comparison against this
#   codebase's own zero-shot reference points: CONDITION1_ZEROSHOT_ACC=0.5214,
#   CONDITION2_BASELINE_ACC=0.5553, CONDITION3_BASELINE_ACC=0.5552. Because
#   there is no calibration stage, `pre_calibration_acc` and
#   `post_calibration_acc` in the shared CSV both hold the SAME zero-shot
#   test accuracy (documented reuse, not a bug) -- this keeps the row shape
#   identical to every calibrated driver's rows without expanding F13's
#   `run_statistics.py` schema.
#
# WHY THE 4 CLASSICAL BASELINES ARE FIT ONCE PER FOLD, NOT ONCE PER SEED:
#   MDM, TS-SVM, TS-LDA(shrinkage), and CSP+LDA are ALL DETERMINISTIC given
#   the (seed-independent) aligned covariances/signal for a fold -- none of
#   them has a stochastic training step, so refitting under a different
#   "seed" label would produce BIT-IDENTICAL predictions. Logging 5 separately
#   -fit-looking identical rows would misrepresent what happened. Instead,
#   each classical baseline is fit and evaluated EXACTLY ONCE per fold; the
#   F4 5-seed loop is then used ONLY to vary the seed of the (inherently
#   stochastic) permutation-null estimate, and the SPDNet-style baseline's
#   actual retraining (see below). This still yields one CSV row per
#   (baseline, seed, fold) -- matching every other driver's tidy long-format
#   schema for F13's statistics script -- but with honestly-identical
#   accuracy/metric columns across the 5 seed rows for the 4 classical
#   baselines, and genuinely seed-varying rows for the SPDNet-style baseline.
#
# SPDNet-style geometric baseline (Huang & Van Gool, 2017) — TIME-BOXED,
# ALLOWED TO FAIL (per AUDIT.md), OUR IMPLEMENTATION (not TSMNet, see the
# naming correction above):
#   A SIMPLIFIED SPDNet-style deep manifold network: one BiMap layer
#   (Stiefel-manifold-constrained linear map on the covariance, W^T C W,
#   re-orthonormalized via QR retraction after every optimizer step -- the
#   same "constrain, then re-project" pattern already used for EEGNet's
#   max-norm weight clamp in this codebase), one ReEig layer (eigenvalue
#   floor/rectification), one LogEig layer (matrix log, reusing
#   eeg_alignment.matrix_log's eigh-based implementation logic), then a
#   linear classifier on the half-vectorized log output. This is NOT the
#   full published SPDNet architecture (no multi-block hierarchy) and it is
#   NOT TSMNet (no SPDDSMBN -- the per-subject/riemannian EA modes already
#   implemented for F3 serve a similar recentering role here, but are not a
#   substitute for TSMNet's learned domain-specific batch norm) -- it is a
#   best-effort, CPU-only, small-budget reimplementation of SPDNet's core
#   BiMap/ReEig/LogEig block only. Actual gradient training makes this
#   baseline's seed loop genuinely meaningful (unlike the 4 deterministic
#   baselines above). Each (fold, seed) attempt is wrapped in its own
#   try/except: a failure (NaN loss, eigh non-convergence, etc.) is caught,
#   logged, and recorded honestly in the final JSON summary
#   (spdnet_n_attempted/succeeded/failed + first-N error strings) WITHOUT
#   aborting the run or silently dropping the failure -- per the user's
#   explicit "honest failure documentation" directive. Can be disabled
#   entirely via `--run-spdnet False` if it proves too unstable on the real
#   dataset, without touching the 4 solid classical baselines.
#
# F3/F4/F9/F14 (built in from the start, matching every other post-Gate-1
# driver in this codebase):
#   - F3:  EA via the shared eeg_alignment.py module, `--ea-mode
#          {none,pooled,per-subject,riemannian}` (default "pooled"). Applied
#          to the z-scored signal before covariance re-estimation, same
#          point in the pipeline as every other LOSO driver.
#   - F4:  `::pilot` (5 folds, 1 seed) / `::full` (29 folds, 5 seeds) split,
#          shared `results/loso_runs.csv`.
#   - F9:  `--cov-estimator {fixed,lwf}` for the covariance feeding EA/MDM/
#          TS-SVM/TS-LDA/CSP (the SPDNet-style baseline operates on the same
#          aligned covariances as its input, so F9 applies to it too).
#   - F14: extended metrics (AUC, kappa, balanced accuracy, macro-F1,
#          sensitivity, specificity) + permutation-derived empirical chance
#          level for every baseline's predictions on every fold.
#
# CSV SCHEMA NOTE: reuses the existing generic `tangent_dim` column for
# TS-SVM/TS-LDA's tangent-vector width AND (documented reuse, same pattern
# as F7's EEGNet fused_dim reuse) for CSP+LDA's log-variance feature width;
# reuses `n_temporal_params` for the SPDNet-style baseline's trainable
# parameter count. MDM has no fixed-width feature vector (it classifies
# directly in covariance/AIRM-distance space) so leaves `tangent_dim` blank
# via restval="".
# `fusion_mode`/`temporal_dim`/`fused_dim`/`best_shrink_weight` are all left
# blank (not applicable -- no fusion, no calibration-shrinkage stage).
#
# LEAKAGE CONTROLS:
#   1. All four classical baselines + the SPDNet-style baseline are fit on
#      the 28-subject training pool ONLY; held-out subject k's trials are
#      touched exactly once, at final zero-shot evaluation.
#   2. StandardScaler (TS-SVM/TS-LDA/CSP+LDA) fit on the 28-subject pool
#      only, applied (not refit) to subject k.
#   3. CSP spatial filters and MDM/SPDNet-style-baseline class-conditional
#      means are computed from 28-subject TRAINING LABELS only; subject k's
#      labels are never used for anything except final scoring.
#   4. EA whitening itself is label-free by construction (F3's
#      euclidean_align takes no `y` parameter) and the held-out subject's
#      own W (per-subject/riemannian modes) is computed from its own
#      UNLABELED trials only.
#
# Usage: modal run run_baselines_mdm_tssvm_tslda_csp.py::pilot
#        modal run run_baselines_mdm_tssvm_tslda_csp.py::full --ea-mode riemannian --cov-estimator lwf
# =============================================================================

import modal

app    = modal.App("bci-f5-classical-baselines")
volume = modal.Volume.from_name("eeg-data-vol")

RAW_DATA_PATH = "/data/processed_eeg_all_subjects.npz"
OUTPUT_JSON   = "/data/results_f5_classical_baselines.json"
LOSO_CSV_PATH = "/data/results/loso_runs.csv"
VOLUME_PATH   = "/data"

# Reference numbers (for logging/comparison only) -- all zero-shot, matching
# this script's own evaluation protocol.
CONDITION1_ZEROSHOT_ACC = 0.5214
CONDITION2_BASELINE_ACC = 0.5553
CONDITION3_BASELINE_ACC = 0.5552
CONDITION4_BASELINE_ACC = 0.6779   # NOTE: calibrated, not zero-shot -- listed for orientation only

SFREQ, N_CHANNELS = 250, 62

RANDOM_SEED          = 42
SEEDS                = [42, 43, 44, 45, 46]   # F4: 5-seed loop (meaningful for the SPDNet-style baseline + permutation RNG only -- see header)
COV_SHRINKAGE_FIXED  = 0.1                     # F9: "fixed" cov-estimator mode
N_PERMUTATIONS        = 1000                   # F14: permutation-derived empirical chance level
SUBJECT_DECODABILITY_SPLITS = 5                # F3: subject-decodability diagnostic

# --- Classical baseline hyperparameters ---
SVC_C              = 1.0
N_CSP_COMPONENTS   = 6   # 3 filter pairs (largest + smallest generalized-eigenvalue ends), classic CSP default

# --- SPDNet-style geometric baseline (our implementation) hyperparameters ---
BIMAP_HIDDEN_DIM    = 30
REEIG_EPS           = 1e-4
LOGEIG_EPS          = 1e-6
SPDNET_EPOCHS        = 50
SPDNET_LR            = 1e-2
SPDNET_MAX_ERRORS_LOGGED = 20   # cap on how many full error strings go into the JSON (avoid bloat on repeated failure)

# Canonical results/loso_runs.csv schema, SHARED verbatim with every other
# LOSO driver in this codebase so all of them append to one unified tidy
# table (order-independent: whichever driver runs first writes this exact
# header; every other driver's rows use restval="" for columns that don't
# apply to it).
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
    .pip_install("numpy<2", "scikit-learn==1.4.2", "scipy", "torch==2.2.0")
    .add_local_python_source("eeg_alignment")
)


@app.function(image=image, cpu=4.0, volumes={VOLUME_PATH: volume}, timeout=86400, memory=16384)
def run_classical_baselines(pilot=True, pilot_n_folds=5, ea_mode="pooled", cov_estimator="fixed", run_spdnet=True):

    import os
    import csv
    import math
    import time
    import json
    import logging
    import numpy as np
    import scipy.linalg
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.covariance import LedoitWolf
    from sklearn.metrics import (
        confusion_matrix, f1_score, roc_auc_score, cohen_kappa_score, balanced_accuracy_score,
    )
    import eeg_alignment as ea

    assert ea_mode in ("none", "pooled", "per-subject", "riemannian"), f"Unknown --ea-mode: {ea_mode!r}"
    assert cov_estimator in ("fixed", "lwf"), f"Unknown --cov-estimator: {cov_estimator!r}"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("f5-classical-baselines")

    raw = np.load(RAW_DATA_PATH, allow_pickle=True)
    X_np = raw["X"].astype(np.float32)
    y_np = raw["y"].astype(np.int64)
    subjects_np = raw["subjects"]
    N, C, T = X_np.shape
    N_CLASSES = int(y_np.max()) + 1
    assert C == N_CHANNELS and N_CLASSES == 2
    log.info(f"X: {X_np.shape} | Subjects total: {len(np.unique(subjects_np))} | ea_mode={ea_mode} | "
             f"cov_estimator={cov_estimator} | run_spdnet={run_spdnet}")
    # F-SILENT hardening: a silently-failed subject upstream (e.g. sub-09's
    # truncated .eeg export, see AUDIT.md D2) must never produce a
    # clean-looking-but-incomplete LOSO run.
    assert len(np.unique(subjects_np)) == 29, (
        f"Expected exactly 29 subjects (30 - sub-09 exclusion per AUDIT.md D2), "
        f"got {len(np.unique(subjects_np))}: {sorted(np.unique(subjects_np).tolist())}"
    )

    unique_subjects = sorted(np.unique(subjects_np).tolist())
    fold_subjects = unique_subjects[:pilot_n_folds] if pilot else unique_subjects
    seeds = [RANDOM_SEED] if pilot else SEEDS
    log.info(f"Running folds: {fold_subjects} | seeds: {seeds}")

    # =========================================================================
    # F9: covariance estimator dispatch (fixed-shrinkage vs. Ledoit-Wolf)
    # =========================================================================
    def compute_trial_covariances(X, seed):
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
            "confusion_matrix": cm.tolist(),
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

    # =========================================================================
    # MDM (Minimum Distance to Mean) -- hand-rolled AIRM-distance
    # nearest-centroid classifier, built directly on eeg_alignment.py's
    # existing (already unit-tested) SPD utilities.
    # =========================================================================
    def airm_distance(A, B):
        """Affine-invariant Riemannian distance between two SPD matrices."""
        _, A_inv_sqrt = ea.matrix_sqrt_inv_sqrt(A)
        M = A_inv_sqrt @ B @ A_inv_sqrt
        log_M = ea.matrix_log(M)
        return float(np.linalg.norm(log_M, ord="fro"))

    def mdm_fit_predict(covs_train, y_train, covs_test):
        class_means = {
            c: ea.riemannian_mean_covariance(covs_train[y_train == c]) for c in (0, 1)
        }
        n = len(covs_test)
        preds = np.empty(n, dtype=np.int64)
        scores = np.empty(n, dtype=np.float64)
        for i in range(n):
            d0 = airm_distance(covs_test[i], class_means[0])
            d1 = airm_distance(covs_test[i], class_means[1])
            scores[i] = d0 - d1          # >0 => closer to class 1 => predict 1
            preds[i] = 1 if scores[i] > 0 else 0
        return preds, scores

    # =========================================================================
    # CSP (classic supervised Common Spatial Patterns) -- hand-rolled via
    # scipy's generalized eigenvalue solver, no `mne` dependency.
    # =========================================================================
    def fit_csp_filters(covs_train, y_train, n_components=N_CSP_COMPONENTS):
        C0 = ea.euclidean_mean_covariance(covs_train[y_train == 0])
        C1 = ea.euclidean_mean_covariance(covs_train[y_train == 1])
        composite = C0 + C1
        eigvals, eigvecs = scipy.linalg.eigh(C0, composite)   # ascending order
        n_pairs = n_components // 2
        filters = np.concatenate([eigvecs[:, :n_pairs], eigvecs[:, -n_pairs:]], axis=1)  # (C, n_components)
        return filters

    def apply_csp_logvar(X, filters):
        """X: (N, C, T) aligned signal, filters: (C, m) -> (N, m) log-variance features."""
        projected = np.einsum("cm,nct->nmt", filters, X)
        var = projected.var(axis=2)
        var = var / (var.sum(axis=1, keepdims=True) + 1e-12)
        return np.log(var + 1e-10).astype(np.float32)

    # =========================================================================
    # SPDNet-style geometric baseline (our implementation; Huang & Van Gool,
    # 2017) -- see header for the explicit list of simplifications vs. the
    # published architecture, and the naming-correction note (this is NOT
    # TSMNet -- no SPDDSMBN).
    # =========================================================================
    def build_and_train_spdnet(covs_train, y_train, covs_test, seed, n_channels):
        import torch
        import torch.nn as nn
        import torch.optim as optim

        torch.manual_seed(seed)

        class BiMapLayer(nn.Module):
            def __init__(self, dim_in, dim_out):
                super().__init__()
                W0, _ = torch.linalg.qr(torch.randn(dim_in, dim_out))
                self.W = nn.Parameter(W0)

            def forward(self, Cb):   # Cb: (N, dim_in, dim_in) -> (N, dim_out, dim_out)
                return torch.einsum("ij,njk,kl->nil", self.W.T, Cb, self.W)

            def retract(self):
                # Stiefel-manifold retraction via QR, applied after every optimizer
                # step -- same "constrain, then re-project" pattern as EEGNet's
                # max-norm weight clamp elsewhere in this codebase.
                with torch.no_grad():
                    Q, _ = torch.linalg.qr(self.W)
                    self.W.copy_(Q)

        class ReEigLayer(nn.Module):
            def forward(self, Cb):
                eigvals, eigvecs = torch.linalg.eigh(Cb)
                eigvals = torch.clamp(eigvals, min=REEIG_EPS)
                return eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)

        class LogEigLayer(nn.Module):
            def forward(self, Cb):
                eigvals, eigvecs = torch.linalg.eigh(Cb)
                eigvals = torch.clamp(eigvals, min=LOGEIG_EPS)
                return eigvecs @ torch.diag_embed(torch.log(eigvals)) @ eigvecs.transpose(-1, -2)

        def vectorize_spd_batch(Cb):
            d = Cb.shape[-1]
            idx0, idx1 = torch.triu_indices(d, d)
            vec = Cb[:, idx0, idx1].clone()
            off = idx0 != idx1
            vec[:, off] = vec[:, off] * math.sqrt(2.0)
            return vec

        class SimplifiedSPDNet(nn.Module):
            def __init__(self, n_ch, hidden_dim, n_classes=2):
                super().__init__()
                self.bimap1 = BiMapLayer(n_ch, hidden_dim)
                self.reeig = ReEigLayer()
                self.logeig = LogEigLayer()
                feat_dim = hidden_dim * (hidden_dim + 1) // 2
                self.classifier = nn.Linear(feat_dim, n_classes)

            def forward(self, Cb):
                Cb = self.bimap1(Cb)
                Cb = self.reeig(Cb)
                Cb = self.logeig(Cb)
                feat = vectorize_spd_batch(Cb)
                return self.classifier(feat)

        device = torch.device("cpu")
        model = SimplifiedSPDNet(n_channels, BIMAP_HIDDEN_DIM, n_classes=2).to(device)
        opt = optim.Adam(model.parameters(), lr=SPDNET_LR)

        C_train_t = torch.tensor(covs_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        C_test_t = torch.tensor(covs_test, dtype=torch.float32)

        for epoch in range(SPDNET_EPOCHS):
            model.train()
            opt.zero_grad()
            logits = model(C_train_t)
            loss = nn.functional.cross_entropy(logits, y_train_t)
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"SPDNet-style baseline loss became non-finite ({float(loss)}) at epoch {epoch}")
            loss.backward()
            opt.step()
            model.bimap1.retract()

        model.eval()
        with torch.no_grad():
            logits_test = model(C_test_t)
            probs = torch.softmax(logits_test, dim=1)
            preds = logits_test.argmax(dim=1).numpy().astype(np.int64)
            scores = (probs[:, 1] - probs[:, 0]).numpy()

        n_params = sum(p.numel() for p in model.parameters())
        return preds, scores, int(n_params)

    # =========================================================================
    # F3 diagnostics: label-free assertion + subject-decodability, computed
    # ONCE up front on the full 29-subject pool.
    # =========================================================================
    mu_all = X_np.mean(axis=(0, 2), keepdims=True)
    sd_all = X_np.std(axis=(0, 2), keepdims=True) + 1e-6
    X_all_z = ((X_np - mu_all) / sd_all).astype(np.float32)
    covs_all_pre, _ = compute_trial_covariances(X_all_z, RANDOM_SEED)
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
    covs_all_post, _ = compute_trial_covariances(X_all_aligned, RANDOM_SEED)
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
    # Main LOSO loop -- classical baselines fit ONCE per fold (deterministic),
    # SPDNet-style baseline retrained per (fold, seed).
    # =========================================================================
    csv_rows = []
    baseline_names = ["mdm", "ts_svm", "ts_lda_shrinkage", "csp_lda"]
    baseline_acc_by_fold = {name: [] for name in baseline_names}
    spdnet_acc_by_seed = {s: [] for s in seeds}
    spdnet_n_attempted, spdnet_n_succeeded = 0, 0
    spdnet_errors = []

    for fold_idx, test_sub in enumerate(fold_subjects):
        fold_start = time.time()
        log.info(f"\n{'='*70}\n  FOLD {fold_idx+1}/{len(fold_subjects)} — sub-{test_sub}\n{'='*70}")

        is_holdout = subjects_np == test_sub
        X_train28, y_train28, subs_train28 = X_np[~is_holdout], y_np[~is_holdout], subjects_np[~is_holdout]
        X_k, y_k = X_np[is_holdout], y_np[is_holdout]

        mu = X_train28.mean(axis=(0, 2), keepdims=True)
        sd = X_train28.std(axis=(0, 2), keepdims=True) + 1e-6
        X_train28_z = ((X_train28 - mu) / sd).astype(np.float32)
        X_k_z = ((X_k - mu) / sd).astype(np.float32)

        covs_train28, lam_train = compute_trial_covariances(X_train28_z, RANDOM_SEED)
        covs_k, lam_k = compute_trial_covariances(X_k_z, RANDOM_SEED)
        realized_lambda_mean = float(np.mean([lam_train, lam_k]))

        W_train = ea.euclidean_align(covs_train28, subs_train28, mode=ea_mode)
        if ea_mode in ("none", "pooled"):
            X_train28_aligned = ea.apply_ea_whitening_signal(X_train28_z, W_train)
            X_k_aligned = ea.apply_ea_whitening_signal(X_k_z, W_train)
        else:
            X_train28_aligned = ea.apply_ea_whitening_signal_per_subject(X_train28_z, subs_train28, W_train)
            subs_k = np.full(len(X_k_z), test_sub)
            W_k = ea.euclidean_align(covs_k, subs_k, mode=ea_mode)
            X_k_aligned = ea.apply_ea_whitening_signal_per_subject(X_k_z, subs_k, W_k)

        covs_train28_aligned, _ = compute_trial_covariances(X_train28_aligned, RANDOM_SEED)
        covs_k_aligned, _ = compute_trial_covariances(X_k_aligned, RANDOM_SEED)

        tan_train28 = ea.tangent_vectorize(covs_train28_aligned)
        tan_k = ea.tangent_vectorize(covs_k_aligned)
        tangent_dim = tan_train28.shape[1]

        # --- fit + predict all 4 classical baselines ONCE for this fold ---
        preds_mdm, scores_mdm = mdm_fit_predict(covs_train28_aligned, y_train28, covs_k_aligned)

        scaler_svm = StandardScaler().fit(tan_train28)
        svm = SVC(kernel="linear", C=SVC_C).fit(scaler_svm.transform(tan_train28), y_train28)
        preds_svm = svm.predict(scaler_svm.transform(tan_k))
        scores_svm = svm.decision_function(scaler_svm.transform(tan_k))

        scaler_lda = StandardScaler().fit(tan_train28)
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
            scaler_lda.transform(tan_train28), y_train28)
        preds_lda = lda.predict(scaler_lda.transform(tan_k))
        scores_lda = lda.decision_function(scaler_lda.transform(tan_k))

        csp_filters = fit_csp_filters(covs_train28_aligned, y_train28)
        feat_train28_csp = apply_csp_logvar(X_train28_aligned, csp_filters)
        feat_k_csp = apply_csp_logvar(X_k_aligned, csp_filters)
        scaler_csp = StandardScaler().fit(feat_train28_csp)
        lda_csp = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
            scaler_csp.transform(feat_train28_csp), y_train28)
        preds_csp = lda_csp.predict(scaler_csp.transform(feat_k_csp))
        scores_csp = lda_csp.decision_function(scaler_csp.transform(feat_k_csp))

        baseline_predictions = {
            "mdm":               (preds_mdm, scores_mdm, None),
            "ts_svm":            (preds_svm, scores_svm, tangent_dim),
            "ts_lda_shrinkage":  (preds_lda, scores_lda, tangent_dim),
            "csp_lda":           (preds_csp, scores_csp, feat_train28_csp.shape[1]),
        }

        for name, (preds, scores, feat_dim) in baseline_predictions.items():
            acc = float((preds == y_k).mean())
            baseline_acc_by_fold[name].append(acc)
            log.info(f"  [{name}] zero-shot acc = {acc:.4f}")

        for seed in seeds:
            for name, (preds, scores, feat_dim) in baseline_predictions.items():
                metrics = compute_extended_metrics(y_k, preds, scores)
                perm_stats = permutation_null_stats(y_k, preds, seed=seed)
                acc = float((preds == y_k).mean())
                row = {
                    "script": "run_baselines_mdm_tssvm_tslda_csp",
                    "condition": name,
                    "seed": seed, "ea_mode": ea_mode, "cov_estimator": cov_estimator,
                    "fold_index": fold_idx, "test_subject": str(test_sub),
                    "realized_cov_lambda_mean": realized_lambda_mean,
                    "pre_calibration_acc": acc, "post_calibration_acc": acc,
                    "sensitivity": metrics["sensitivity"], "specificity": metrics["specificity"],
                    "f1": metrics["f1"], "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"], "cohen_kappa": metrics["cohen_kappa"],
                    "roc_auc": metrics["roc_auc"],
                    "permutation_empirical_chance_level": perm_stats["permutation_empirical_chance_level"],
                    "permutation_null_std_acc": perm_stats["permutation_null_std_acc"],
                    "permutation_p_value": perm_stats["permutation_p_value"],
                }
                if feat_dim is not None:
                    row["tangent_dim"] = int(feat_dim)
                csv_rows.append(row)

            if run_spdnet:
                spdnet_n_attempted += 1
                try:
                    preds_t, scores_t, n_params_t = build_and_train_spdnet(
                        covs_train28_aligned, y_train28, covs_k_aligned, seed, N_CHANNELS)
                    acc_t = float((preds_t == y_k).mean())
                    spdnet_acc_by_seed[seed].append(acc_t)
                    metrics_t = compute_extended_metrics(y_k, preds_t, scores_t)
                    perm_stats_t = permutation_null_stats(y_k, preds_t, seed=seed)
                    csv_rows.append({
                        "script": "run_baselines_mdm_tssvm_tslda_csp",
                        "condition": "spdnet_geometric_baseline",
                        "seed": seed, "ea_mode": ea_mode, "cov_estimator": cov_estimator,
                        "fold_index": fold_idx, "test_subject": str(test_sub),
                        "n_temporal_params": n_params_t,
                        "realized_cov_lambda_mean": realized_lambda_mean,
                        "pre_calibration_acc": acc_t, "post_calibration_acc": acc_t,
                        "sensitivity": metrics_t["sensitivity"], "specificity": metrics_t["specificity"],
                        "f1": metrics_t["f1"], "macro_f1": metrics_t["macro_f1"],
                        "balanced_accuracy": metrics_t["balanced_accuracy"], "cohen_kappa": metrics_t["cohen_kappa"],
                        "roc_auc": metrics_t["roc_auc"],
                        "permutation_empirical_chance_level": perm_stats_t["permutation_empirical_chance_level"],
                        "permutation_null_std_acc": perm_stats_t["permutation_null_std_acc"],
                        "permutation_p_value": perm_stats_t["permutation_p_value"],
                    })
                    spdnet_n_succeeded += 1
                    log.info(f"  [spdnet_geometric_baseline] seed={seed} zero-shot acc = {acc_t:.4f} (n_params={n_params_t})")
                except Exception as e:
                    err_str = f"fold={fold_idx} sub-{test_sub} seed={seed}: {type(e).__name__}: {e}"
                    log.warning(f"  [spdnet_geometric_baseline] FAILED -- {err_str}")
                    if len(spdnet_errors) < SPDNET_MAX_ERRORS_LOGGED:
                        spdnet_errors.append(err_str)

        log.info(f"  Fold elapsed: {time.time()-fold_start:.0f}s")

    # =========================================================================
    # Aggregate + report
    # =========================================================================
    baseline_grand_mean = {name: float(np.mean(accs)) for name, accs in baseline_acc_by_fold.items()}
    baseline_grand_std = {name: float(np.std(accs)) for name, accs in baseline_acc_by_fold.items()}

    spdnet_per_seed_mean = [float(np.mean(v)) for v in spdnet_acc_by_seed.values() if len(v) > 0]
    spdnet_grand_mean = float(np.mean(spdnet_per_seed_mean)) if spdnet_per_seed_mean else float("nan")
    spdnet_grand_std = float(np.std(spdnet_per_seed_mean)) if spdnet_per_seed_mean else float("nan")

    log.info(f"\n{'='*70}\n  F5 CLASSICAL BASELINES — {len(fold_subjects)} folds ({'PILOT' if pilot else 'FULL'})\n{'='*70}")
    for name in baseline_names:
        log.info(f"    {name:<20}: {baseline_grand_mean[name]*100:.2f}% (+/- {baseline_grand_std[name]*100:.2f}pp across folds)")
    if run_spdnet:
        log.info(f"    {'spdnet_geometric_baseline':<26}: {spdnet_grand_mean*100:.2f}% (+/- {spdnet_grand_std*100:.2f}pp across seeds) "
                  f"| attempted={spdnet_n_attempted} succeeded={spdnet_n_succeeded} failed={spdnet_n_attempted - spdnet_n_succeeded}")
    log.info(
        "\n  COMPARISON (zero-shot reference numbers from other conditions):\n"
        f"    Condition 1 zero-shot EEGNet     : {CONDITION1_ZEROSHOT_ACC*100:.2f}%\n"
        f"    Condition 2 baseline             : {CONDITION2_BASELINE_ACC*100:.2f}%\n"
        f"    Condition 3 baseline             : {CONDITION3_BASELINE_ACC*100:.2f}%\n"
        f"    Condition 4 baseline (CALIBRATED, not zero-shot, for orientation): {CONDITION4_BASELINE_ACC*100:.2f}%\n"
    )

    # F4: write the tidy long-format CSV, appending to the SAME shared file
    # used by every other LOSO driver.
    if not pilot:
        os.makedirs(os.path.dirname(LOSO_CSV_PATH), exist_ok=True)
        write_header = not os.path.exists(LOSO_CSV_PATH)
        with open(LOSO_CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOSO_CSV_FIELDNAMES, restval="")
            if write_header:
                writer.writeheader()
            writer.writerows(csv_rows)

    results_payload = {
        "condition": "F5 — Classical BCI baselines (MDM, TS-SVM, TS-LDA(shrinkage), CSP+LDA) + time-boxed SPDNet-style geometric baseline",
        "ea_mode": ea_mode, "cov_estimator": cov_estimator, "pilot": pilot,
        "n_folds": len(fold_subjects), "seeds": seeds,
        "hyperparameters": {
            "svc_C": SVC_C, "n_csp_components": N_CSP_COMPONENTS,
            "cov_shrinkage_fixed": COV_SHRINKAGE_FIXED, "n_permutations": N_PERMUTATIONS,
            "spdnet_bimap_hidden_dim": BIMAP_HIDDEN_DIM, "spdnet_epochs": SPDNET_EPOCHS,
            "spdnet_lr": SPDNET_LR, "spdnet_reeig_eps": REEIG_EPS, "spdnet_logeig_eps": LOGEIG_EPS,
        },
        "f3_diagnostics": {
            "verify_label_free_passed": True,
            "subject_decodability_pre_alignment": subj_decode_acc_pre,
            "subject_decodability_post_alignment": subj_decode_acc_post,
            "subject_decodability_chance_level": subj_decode_chance,
            "subject_decodability_n_subjects": subj_decode_n,
        },
        "baseline_grand_mean_accuracy": baseline_grand_mean,
        "baseline_grand_std_accuracy_across_folds": baseline_grand_std,
        "spdnet_geometric_baseline": {
            "attempted": run_spdnet,
            "n_attempted": spdnet_n_attempted, "n_succeeded": spdnet_n_succeeded,
            "n_failed": spdnet_n_attempted - spdnet_n_succeeded,
            "grand_mean_accuracy": spdnet_grand_mean, "grand_std_accuracy_across_seeds": spdnet_grand_std,
            "per_seed_mean_accuracy": spdnet_per_seed_mean,
            "error_examples": spdnet_errors,
            "note": ("Simplified single-BiMap-block SPDNet (Huang & Van Gool, 2017) reimplementation -- "
                     "our implementation, NOT TSMNet (Kobler et al. 2022): TSMNet's defining SPDDSMBN "
                     "component (SPD domain-specific batch normalization) was NOT implemented here. "
                     "TSMNet itself was not re-implemented because doing so properly requires the "
                     "official github.com/rkobler/TSMNet code and its `geoopt` dependency, which this "
                     "codebase avoids per its no-new-fragile-dependency convention (see DECISIONS.md C1). "
                     "See script header for the explicit simplification list vs. published SPDNet. "
                     "Failures (if any) are honestly reported here, not hidden."),
        },
        "loso_csv_path": LOSO_CSV_PATH if not pilot else None,
        "reference_condition1_zeroshot_acc": CONDITION1_ZEROSHOT_ACC,
        "reference_condition2_baseline_acc": CONDITION2_BASELINE_ACC,
        "reference_condition3_baseline_acc": CONDITION3_BASELINE_ACC,
        "reference_condition4_baseline_acc_calibrated": CONDITION4_BASELINE_ACC,
    }
    if not pilot:
        with open(OUTPUT_JSON, "w") as f:
            json.dump(results_payload, f, indent=2)
        volume.commit()
        log.info(f"  Saved: {OUTPUT_JSON}")
        log.info(f"  Saved: {LOSO_CSV_PATH}")

    # =========================================================================
    # C3 plausibility assertions -- printed next to the numbers, not silent.
    # Run AFTER the write above so a failing assertion never suppresses the
    # diagnostic artifact.
    # =========================================================================
    all_accs_for_check = dict(baseline_grand_mean)
    if run_spdnet and spdnet_n_succeeded > 0:
        all_accs_for_check["spdnet_geometric_baseline"] = spdnet_grand_mean
    for name, acc in all_accs_for_check.items():
        assert 0.0 <= acc <= 1.0, f"[C3 PLAUSIBILITY FAIL] {name} grand-mean accuracy {acc} outside [0,1]"
    log.info(f"  [C3] plausibility: all grand-mean accuracies in [0,1] -- OK ({list(all_accs_for_check.keys())})")
    n_subjects_seen = len(fold_subjects) if pilot else len(unique_subjects)
    if not pilot:
        assert n_subjects_seen == 29, f"[C3 PLAUSIBILITY FAIL] expected 29 unique subjects, got {n_subjects_seen}"
        assert len(seeds) == 5, f"[C3 PLAUSIBILITY FAIL] expected 5 seeds in ::full, got {len(seeds)}"
        log.info(f"  [C3] plausibility: 29/29 subjects, {len(seeds)}/5 seeds -- OK")

    return {
        "baseline_grand_mean_accuracy": baseline_grand_mean,
        "spdnet_geometric_baseline_grand_mean_accuracy": spdnet_grand_mean,
        "spdnet_geometric_baseline_n_attempted": spdnet_n_attempted, "spdnet_geometric_baseline_n_succeeded": spdnet_n_succeeded,
        "subject_decodability_pre_alignment": subj_decode_acc_pre,
        "subject_decodability_post_alignment": subj_decode_acc_post,
        "n_folds": len(fold_subjects), "n_seeds": len(seeds),
        "output_path": OUTPUT_JSON if not pilot else None, "loso_csv_path": LOSO_CSV_PATH if not pilot else None,
    }


@app.local_entrypoint(name="pilot")
def pilot_entrypoint(ea_mode: str = "pooled", cov_estimator: str = "fixed", run_spdnet: bool = True):
    print("F5 — Classical BCI Baselines [PILOT: 5 folds, 1 seed, no CSV write]")
    print(f"ea_mode={ea_mode}  cov_estimator={cov_estimator}  run_spdnet={run_spdnet}\n")
    results = run_classical_baselines.remote(
        pilot=True, pilot_n_folds=5, ea_mode=ea_mode, cov_estimator=cov_estimator, run_spdnet=run_spdnet)
    print("\nPILOT RESULTS:")
    for k, v in results.items():
        print(f"  {k:<32}: {v}")


@app.local_entrypoint(name="full")
def full_entrypoint(ea_mode: str = "pooled", cov_estimator: str = "fixed", run_spdnet: bool = True):
    print("F5 — Classical BCI Baselines [FULL: 29 folds x 5 seeds]")
    print(f"ea_mode={ea_mode}  cov_estimator={cov_estimator}  run_spdnet={run_spdnet}\n")
    results = run_classical_baselines.remote(
        pilot=False, ea_mode=ea_mode, cov_estimator=cov_estimator, run_spdnet=run_spdnet)
    print("\nFULL RESULTS:")
    for k, v in results.items():
        print(f"  {k:<32}: {v}")
