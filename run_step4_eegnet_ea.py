# =============================================================================
# command to run:
#   modal run run_step4_eegnet_ea.py::pilot
#   modal run run_step4_eegnet_ea.py::pilot --ea-mode per-subject --cov-estimator lwf
#   modal run run_step4_eegnet_ea.py::full  --ea-mode none
#   modal run run_step4_eegnet_ea.py::full  --ea-mode riemannian --cov-estimator lwf
#
# run_step4_eegnet_ea.py
#
# F7 -- EEGNet + Euclidean/Riemannian Alignment: the "EEGNet" column of the
# F7 2x2 alignment/representation grid.
#
# WHY THIS SCRIPT EXISTS:
#   run_step4_condition1_eegnet_calibrated.py established the EEGNet+15%-
#   calibration baseline (55.64% full 29-fold), but EA is NOT applied to its
#   input anywhere (documented as a known gap in AUDIT.md's EEGNet-baseline
#   row: "EA is not applied to its input (F7)"). Meanwhile every tangent-
#   space driver (run_step4_matched_spatial_control.py,
#   run_step4_condition4_asymmetric_mamba.py) DOES apply EA. That makes any
#   EEGNet-vs-tangent-space comparison confounded: is EEGNet weaker because
#   of its architecture, or because it never got the same input-alignment
#   treatment? This script closes that gap and completes the 2x2 grid:
#
#                    no-EA                          EA (--ea-mode != none)
#     tangent-space  run_step4_matched_spatial_    run_step4_matched_spatial_
#                    control.py --ea-mode none     control.py --ea-mode {pooled,
#                                                   per-subject,riemannian}
#     EEGNet         run_step4_eegnet_ea.py         run_step4_eegnet_ea.py
#                    --ea-mode none                 --ea-mode {pooled,
#                    (~= condition1_eegnet_          per-subject,riemannian}
#                    calibrated.py's original
#                    no-EA numbers, up to RNG)
#
#   Both grid rows are now driven by the exact SAME `--ea-mode` flag and the
#   exact same eeg_alignment.py module (F3) -- no separate "skip EA" code
#   path anywhere, so the no-EA and EA cells are byte-for-byte identical
#   except for the alignment step itself.
#
# WHERE EA IS APPLIED (matches run_step4_condition4_asymmetric_mamba.py's
# temporal branch exactly, since EEGNet -- like Mamba -- consumes the raw
# (channels, time) signal directly, not precomputed tangent vectors):
#   1. Per-channel/time z-score, fit on the 28-subject training pool only.
#   2. EA whitening (per `--ea-mode`) applied to the z-scored RAW signal --
#      "pooled": one W from the 28-subject pool, applied to pool + held-out.
#      "per-subject"/"riemannian": each of the 28 pool subjects gets its own
#        W from its own trials; the held-out subject ALSO gets its own
#        unsupervised W from its own (unlabeled) trials only (legitimate --
#        EA never touches y). "none": identity W (F7's no-EA cell).
#   3. The ALIGNED signal is what EEGNet is pretrained on and what
#      penultimate features are extracted from -- EA happens once, upstream
#      of the architecture, exactly like the tangent-space drivers.
#
# CALIBRATION METHOD -- unchanged from run_step4_condition1_eegnet_
# calibrated.py (PCA + global/local shrinkage-blended LogisticRegression on
# frozen-backbone penultimate features), so any accuracy delta vs. that
# script's original numbers is attributable to EA alone, not a calibration
# methodology change.
#
# AUTHENTIC EEGNet ARCHITECTURE (Lawhern et al., 2018) -- copied verbatim
# from run_step4_condition1_eegnet_calibrated.py (same F1/D/F2/dropout,
# same manual max-norm constraint implementation; PyTorch has no native
# Keras-style kernel_constraint=max_norm).
#
# F4/F9/F14 (same conventions as the other two LOSO drivers, built in from
# the start rather than retrofitted):
#   - F4:  5-seed loop (SEEDS below) for `::full`; `::pilot` stays a single
#          seed over a small fold subset for a fast diagnostic. Every
#          seed x fold result appends to the SAME shared results/
#          loso_runs.csv (F13 depends on this table covering every arm).
#          No best-run/best-epoch selection: pretraining early-stopping uses
#          an internal train-pool validation split only (legitimate
#          train-time model selection, not results cherry-picking).
#   - F9:  `--cov-estimator {fixed,lwf}` controls the covariance estimator
#          used INSIDE the EA-fitting step (trial_covariances / per-trial
#          Ledoit-Wolf) -- EEGNet's own architecture never sees a covariance
#          matrix, but EA's whitening reference does need one.
#   - F14: AUC, Cohen's kappa, balanced accuracy, macro-F1, sensitivity,
#          specificity, plus a permutation-derived empirical chance level
#          (1000 label permutations per fold, reusing the already-fit
#          classifier's predictions).
#
# CSV SCHEMA NOTE: this script writes into the SAME results/loso_runs.csv /
# LOSO_CSV_FIELDNAMES used by the tangent-space drivers. EEGNet has no
# spatial-tangent / temporal-Mamba split, so "tangent_dim" and "temporal_dim"
# are left blank (restval="") for its rows; the EEGNet penultimate feature
# width (the dimensionality entering the shared PCA->calibration stage) is
# recorded in "fused_dim" (same column tangent+Mamba fusion rows use for
# their pre-joint-PCA feature width) and EEGNet's own trainable parameter
# count is recorded in "n_temporal_params" (same column the Mamba driver
# uses for its temporal branch's parameter count) -- reusing these generic
# columns rather than adding EEGNet-only columns keeps F13's cross-arm
# statistics script schema-agnostic.
#
# LEAKAGE CONTROLS (same as run_step4_condition1_eegnet_calibrated.py):
#   1. Backbone pretraining (28 subjects) never sees held-out subject k's
#      cal or test data. Early-stopping validation is carved from the 28
#      training subjects only.
#   2. StandardScaler/PCA for calibration fit ONLY on 28-subject penultimate
#      features.
#   3. D_cal/D_test disjointness asserted explicitly.
#   4. Global LogReg fit only on the 28-subject pool; local LogReg and the
#      shrinkage weight are selected using D_cal ONLY (internal CV). D_test
#      touched exactly once, for final evaluation.
#   5. EEGNet weights frozen before subject k's data is ever seen.
#   6. EA whitening itself is label-free by construction (F3's
#      euclidean_align takes no `y` parameter) and the held-out subject's
#      own W (per-subject/riemannian modes) is computed from its own
#      UNLABELED trials only.
#
# Usage: modal run run_step4_eegnet_ea.py::pilot
#        modal run run_step4_eegnet_ea.py::full --ea-mode per-subject --cov-estimator lwf
# =============================================================================

import modal

app    = modal.App("bci-f7-eegnet-ea")
volume = modal.Volume.from_name("eeg-data-vol")

RAW_DATA_PATH = "/data/processed_eeg_all_subjects.npz"
OUTPUT_JSON   = "/data/results_f7_eegnet_ea.json"
LOSO_CSV_PATH = "/data/results/loso_runs.csv"
VOLUME_PATH   = "/data"

# Reference numbers (for logging/comparison only)
CONDITION1_ZEROSHOT_ACC       = 0.5214   # EEGNet, zero-shot LOSO, no EA, no calib (given)
CONDITION1B_NO_EA_CALIB_ACC   = 0.5564   # EEGNet + 15% calib, no EA (run_step4_condition1_eegnet_calibrated.py)

# ── CONFIRMED (step1_2_filter_epoch.py: SFREQ_NEW=250, epoch -0.2s:0.8s) ────
SFREQ            = 250
N_CHANNELS       = 62
KERN_LENGTH      = 125     # SFREQ // 2, per Lawhern et al. convention
# ────────────────────────────────────────────────────────────────────────────

CAL_FRACTION        = 0.15
RANDOM_SEED          = 42
SEEDS                = [42, 43, 44, 45, 46]   # F4: 5-seed loop for ::full
COV_SHRINKAGE_FIXED  = 0.1                     # F9: "fixed" cov-estimator mode (used only for EA fitting)
PCA_MAX_COMPONENTS  = 35
LOGREG_C            = 1.0
SHRINK_GRID         = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS     = 3
N_PERMUTATIONS       = 1000                    # F14
SUBJECT_DECODABILITY_SPLITS = 5                # F3

# EEGNet hyperparameters (Lawhern et al. 2018 defaults) -- identical to
# run_step4_condition1_eegnet_calibrated.py.
EEGNET_F1            = 8
EEGNET_D             = 2
EEGNET_F2            = EEGNET_F1 * EEGNET_D   # 16
EEGNET_DROPOUT       = 0.5
EEGNET_DEPTHWISE_MAXNORM = 1.0
EEGNET_DENSE_MAXNORM     = 0.25

PRETRAIN_EPOCHS   = 100
PRETRAIN_BATCH    = 64
PRETRAIN_LR       = 1e-3
PRETRAIN_WD       = 1e-4
PRETRAIN_ES_PAT   = 10
INTERNAL_VAL_FRAC = 0.10

# Canonical results/loso_runs.csv schema, SHARED verbatim with
# run_step4_matched_spatial_control.py and
# run_step4_condition4_asymmetric_mamba.py -- see the CSV SCHEMA NOTE above
# for how EEGNet-specific quantities map onto these generic column names.
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
    .pip_install(
        "torch==2.2.0",
        "numpy<2",
        "scikit-learn==1.4.2",
        "scipy",
    )
    .add_local_python_source("eeg_alignment")
)


# =============================================================================
# SECTION 1.5: STANDALONE INSPECTOR
#   modal run run_step4_eegnet_ea.py::inspect
# =============================================================================
@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=300)
def inspect_npz():
    import numpy as np
    raw = np.load(RAW_DATA_PATH, allow_pickle=True)
    print(f"\nArchive: {RAW_DATA_PATH}")
    print(f"Keys found: {raw.files}\n")
    for k in raw.files:
        arr = raw[k]
        print(f"  '{k}': shape={arr.shape} dtype={arr.dtype}")
    return raw.files


@app.local_entrypoint(name="inspect")
def inspect_entrypoint():
    keys = inspect_npz.remote()
    print(f"\nKeys in {RAW_DATA_PATH}: {keys}")


# =============================================================================
# SECTION 2: MODAL FUNCTION
# =============================================================================
@app.function(
    image=image,
    gpu="A10G",
    volumes={VOLUME_PATH: volume},
    timeout=86400,
    memory=16384,
)
def run_eegnet_ea(pilot=True, pilot_n_folds=5, ea_mode="pooled", cov_estimator="fixed"):

    import os
    import csv
    import copy
    import math
    import time
    import json
    import logging
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
    from sklearn.covariance import LedoitWolf
    from sklearn.metrics import (
        confusion_matrix, f1_score, roc_auc_score, cohen_kappa_score, balanced_accuracy_score,
    )
    import eeg_alignment as ea

    assert ea_mode in ("none", "pooled", "per-subject", "riemannian"), f"Unknown --ea-mode: {ea_mode!r}"
    assert cov_estimator in ("fixed", "lwf"), f"Unknown --cov-estimator: {cov_estimator!r}"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("f7-eegnet-ea")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device} | pilot={pilot} | ea_mode={ea_mode} | cov_estimator={cov_estimator}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # =========================================================================
    # SECTION 3: DATA LOADING (RAW epoched EEG)
    # =========================================================================
    raw = np.load(RAW_DATA_PATH, allow_pickle=True)
    X_np = raw["X"].astype(np.float32)
    y_np = raw["y"].astype(np.int64)
    subjects_np = raw["subjects"]

    N, C, T = X_np.shape
    N_CLASSES = int(y_np.max()) + 1
    assert C == N_CHANNELS and N_CLASSES == 2
    log.info(f"X: {X_np.shape} | y: {y_np.shape} | Classes: {N_CLASSES}")
    # F-SILENT hardening: a silently-failed subject upstream (e.g. sub-09's
    # truncated .eeg export, see AUDIT.md D2) must never produce a
    # clean-looking-but-incomplete LOSO run.
    assert len(np.unique(subjects_np)) == 29, (
        f"Expected exactly 29 subjects (30 - sub-09 exclusion per AUDIT.md D2), "
        f"got {len(np.unique(subjects_np))}: {sorted(np.unique(subjects_np).tolist())}"
    )

    # =========================================================================
    # SECTION 4: AUTHENTIC EEGNet ARCHITECTURE (Lawhern et al., 2018) --
    # copied verbatim from run_step4_condition1_eegnet_calibrated.py.
    # =========================================================================
    class DepthwiseConv2dMaxNorm(nn.Module):
        def __init__(self, in_ch, depth_mult, kernel_size):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, in_ch * depth_mult, kernel_size, groups=in_ch, bias=False)

        def forward(self, x):
            return self.conv(x)

    class SeparableConv2d(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size, padding):
            super().__init__()
            self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False)
            self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)

        def forward(self, x):
            return self.pointwise(self.depthwise(x))

    class EEGNet(nn.Module):
        def __init__(self, n_channels, n_times, n_classes=2,
                     F1=EEGNET_F1, D=EEGNET_D, F2=EEGNET_F2,
                     kern_length=KERN_LENGTH, dropout=EEGNET_DROPOUT):
            super().__init__()
            self.conv1 = nn.Conv2d(1, F1, (1, kern_length), padding=(0, kern_length // 2), bias=False)
            self.bn1 = nn.BatchNorm2d(F1)
            self.depthwise = DepthwiseConv2dMaxNorm(F1, D, (n_channels, 1))
            self.bn2 = nn.BatchNorm2d(F1 * D)
            self.elu1 = nn.ELU()
            self.pool1 = nn.AvgPool2d((1, 4))
            self.drop1 = nn.Dropout(dropout)

            self.separable = SeparableConv2d(F1 * D, F2, (1, 16), padding=(0, 8))
            self.bn3 = nn.BatchNorm2d(F2)
            self.elu2 = nn.ELU()
            self.pool2 = nn.AvgPool2d((1, 8))
            self.drop2 = nn.Dropout(dropout)

            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_channels, n_times)
                feat = self._features(dummy)
                self.penultimate_dim = feat.shape[1]

            self.classifier = nn.Linear(self.penultimate_dim, n_classes)

        def _features(self, x):
            x = self.conv1(x); x = self.bn1(x)
            x = self.depthwise(x); x = self.bn2(x); x = self.elu1(x)
            x = self.pool1(x); x = self.drop1(x)
            x = self.separable(x); x = self.bn3(x); x = self.elu2(x)
            x = self.pool2(x); x = self.drop2(x)
            return torch.flatten(x, start_dim=1)

        def forward_features(self, x):
            return self._features(x)

        def forward(self, x):
            return self.classifier(self._features(x))

    def apply_max_norm_constraints(model):
        with torch.no_grad():
            w = model.depthwise.conv.weight
            norms = w.view(w.size(0), -1).norm(dim=1, keepdim=True)
            desired = torch.clamp(norms, max=EEGNET_DEPTHWISE_MAXNORM)
            scale = (desired / (norms + 1e-8)).view(-1, 1, 1, 1)
            w.mul_(scale)

            w2 = model.classifier.weight
            norms2 = w2.norm(dim=1, keepdim=True)
            desired2 = torch.clamp(norms2, max=EEGNET_DENSE_MAXNORM)
            scale2 = desired2 / (norms2 + 1e-8)
            w2.mul_(scale2)

    def train_one_epoch(model, loader, optimizer, criterion):
        model.train()
        total_loss, n_correct, n_total = 0.0, 0, 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            apply_max_norm_constraints(model)
            total_loss += loss.item()
            n_correct += (logits.argmax(1) == yb).sum().item()
            n_total += yb.size(0)
        return total_loss / len(loader), n_correct / n_total

    @torch.no_grad()
    def evaluate(model, loader, criterion):
        model.eval()
        total_loss, n_correct, n_total = 0.0, 0, 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            total_loss += criterion(logits, yb).item()
            n_correct += (logits.argmax(1) == yb).sum().item()
            n_total += yb.size(0)
        return total_loss / len(loader), n_correct / n_total

    @torch.no_grad()
    def extract_penultimate_features(model, X_np_subset):
        model.eval()
        X_t = torch.from_numpy(X_np_subset).unsqueeze(1).to(device)
        return model.forward_features(X_t).cpu().numpy()

    def make_loader(X, y, batch_size, shuffle):
        ds = TensorDataset(torch.from_numpy(X).unsqueeze(1), torch.from_numpy(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)

    # =========================================================================
    # F9: covariance estimator dispatch -- used ONLY for fitting the EA
    # whitening reference (trial_covariances()), not by EEGNet itself.
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
    # ONCE up front on the full 29-subject pool (same standard EA-quality
    # check the tangent-space drivers run -- independent of EEGNet).
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
    # SECTION 6: LOSO LOOP (5 seeds x 29 folds for ::full, 1 seed x
    # pilot_n_folds for ::pilot)
    # =========================================================================
    unique_subjects = sorted(np.unique(subjects_np).tolist())
    fold_subjects = unique_subjects[:pilot_n_folds] if pilot else unique_subjects
    seeds = [RANDOM_SEED] if pilot else SEEDS

    log.info(f"  Seeds: {seeds} | Folds: {len(fold_subjects)}/{len(unique_subjects)} "
             f"({'PILOT' if pilot else 'FULL'})")

    csv_rows = []
    per_seed_mean_acc = []
    fold_records_last_seed = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        fold_records, all_test_acc = [], []

        for fold_idx, test_sub in enumerate(fold_subjects):
            fold_start = time.time()
            log.info(f"\n{'='*70}\n  SEED {seed} | FOLD {fold_idx+1}/{len(fold_subjects)} — sub-{test_sub}\n{'='*70}")

            is_holdout = subjects_np == test_sub
            X_train28, y_train28, subs_train28 = X_np[~is_holdout], y_np[~is_holdout], subjects_np[~is_holdout]
            X_k, y_k = X_np[is_holdout], y_np[is_holdout]

            # ---- Per-channel/time z-score, fit on 28-subj pool only ----
            mu = X_train28.mean(axis=(0, 2), keepdims=True)
            sd = X_train28.std(axis=(0, 2), keepdims=True) + 1e-6
            X_train28_z = ((X_train28 - mu) / sd).astype(np.float32)
            X_k_z = ((X_k - mu) / sd).astype(np.float32)

            # ---- F3/F9: EA whitening on the z-scored RAW signal ----
            covs_train28, lam_train = compute_trial_covariances(X_train28_z, seed)
            covs_k, lam_k = compute_trial_covariances(X_k_z, seed)
            realized_lambda_mean = float(np.mean([lam_train, lam_k]))

            W_train = ea.euclidean_align(covs_train28, subs_train28, mode=ea_mode)
            if ea_mode in ("none", "pooled"):
                X_train28_aligned = ea.apply_ea_whitening_signal(X_train28_z, W_train).astype(np.float32)
                X_k_aligned = ea.apply_ea_whitening_signal(X_k_z, W_train).astype(np.float32)
            else:
                X_train28_aligned = ea.apply_ea_whitening_signal_per_subject(
                    X_train28_z, subs_train28, W_train).astype(np.float32)
                subs_k = np.full(len(X_k_z), test_sub)
                W_k = ea.euclidean_align(covs_k, subs_k, mode=ea_mode)
                X_k_aligned = ea.apply_ea_whitening_signal_per_subject(X_k_z, subs_k, W_k).astype(np.float32)

            X_pt_tr, X_pt_val, y_pt_tr, y_pt_val = train_test_split(
                X_train28_aligned, y_train28,
                test_size=INTERNAL_VAL_FRAC, stratify=y_train28, random_state=seed,
            )
            pretrain_loader = make_loader(X_pt_tr, y_pt_tr, PRETRAIN_BATCH, shuffle=True)
            ptval_loader = make_loader(X_pt_val, y_pt_val, PRETRAIN_BATCH, shuffle=False)
            criterion = nn.CrossEntropyLoss()

            # =====================================================================
            # STAGE A — Pretrain EEGNet on the ALIGNED 28-subject pool
            # =====================================================================
            model = EEGNet(n_channels=C, n_times=T, n_classes=N_CLASSES).to(device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer = torch.optim.Adam(model.parameters(), lr=PRETRAIN_LR, weight_decay=PRETRAIN_WD)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=PRETRAIN_EPOCHS, eta_min=PRETRAIN_LR * 0.01)

            best_val_loss, best_state_dict, es_counter = math.inf, None, 0
            for epoch in range(1, PRETRAIN_EPOCHS + 1):
                tr_loss, tr_acc = train_one_epoch(model, pretrain_loader, optimizer, criterion)
                scheduler.step()
                val_loss, val_acc = evaluate(model, ptval_loader, criterion)
                if epoch == 1 or epoch % 20 == 0 or epoch == PRETRAIN_EPOCHS:
                    log.info(f"    Ep {epoch:03d}/{PRETRAIN_EPOCHS} | Train acc={tr_acc:.3f} "
                             f"| IntVal loss={val_loss:.4f} acc={val_acc:.3f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = copy.deepcopy(model.state_dict())
                    es_counter = 0
                else:
                    es_counter += 1
                    if es_counter >= PRETRAIN_ES_PAT:
                        log.info(f"    Early stop at epoch {epoch}")
                        break

            model.load_state_dict(best_state_dict)
            for p in model.parameters():
                p.requires_grad = False
            model.eval()

            # =====================================================================
            # STAGE B — Extract penultimate features (aligned pool + aligned k)
            # =====================================================================
            feat_train28 = extract_penultimate_features(model, X_train28_aligned)
            feat_k = extract_penultimate_features(model, X_k_aligned)

            sss = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - CAL_FRACTION), random_state=seed)
            cal_idx, test_idx = next(sss.split(feat_k, y_k))
            assert set(cal_idx.tolist()).isdisjoint(set(test_idx.tolist())), \
                "Leakage detected: D_cal and D_test overlap!"
            feat_cal, y_cal = feat_k[cal_idx], y_k[cal_idx]
            feat_test, y_test = feat_k[test_idx], y_k[test_idx]

            scaler = StandardScaler()
            feat_train28_z = scaler.fit_transform(feat_train28)
            feat_cal_z = scaler.transform(feat_cal)
            feat_test_z = scaler.transform(feat_test)

            n_components = min(PCA_MAX_COMPONENTS, model.penultimate_dim - 1, feat_train28_z.shape[0] - 1)
            pca = PCA(n_components=n_components, random_state=seed)
            X_train28_pca = pca.fit_transform(feat_train28_z)
            X_cal_pca = pca.transform(feat_cal_z)
            X_test_pca = pca.transform(feat_test_z)

            # =====================================================================
            # STAGE C — Shrinkage-blended calibration
            # =====================================================================
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
                f"| perm_chance={perm_stats['permutation_empirical_chance_level']:.4f} "
                f"p={perm_stats['permutation_p_value']:.4f} "
                f"| penultimate_dim={model.penultimate_dim} realized_lambda={realized_lambda_mean:.4f}"
            )

            row = {
                "script": "run_step4_eegnet_ea",
                "condition": "eegnet_ea",
                "seed": seed, "ea_mode": ea_mode, "cov_estimator": cov_estimator,
                "fold_index": fold_idx, "test_subject": str(test_sub),
                "fused_dim": int(model.penultimate_dim),
                "n_temporal_params": int(n_params),
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

            del model, optimizer, scheduler, best_state_dict
            if device.type == "cuda":
                torch.cuda.empty_cache()

        seed_mean_acc = float(np.mean(all_test_acc))
        per_seed_mean_acc.append(seed_mean_acc)
        fold_records_last_seed = fold_records
        log.info(f"\nSEED {seed} DONE — mean acc {seed_mean_acc:.4f} over {len(fold_subjects)} folds")

    grand_mean_acc = float(np.mean(per_seed_mean_acc))
    grand_std_acc = float(np.std(per_seed_mean_acc))

    log.info(f"\n{'='*70}\n  F7 EEGNet+EA — {len(seeds)} seed(s) x {len(fold_subjects)} fold(s) "
             f"({'PILOT' if pilot else 'FULL'})\n{'='*70}")
    log.info(f"  Grand mean-of-seed-means Acc: {grand_mean_acc:.4f} ± {grand_std_acc:.4f}")
    log.info(
        "\n  COMPARISON (reference numbers):\n"
        f"    Condition 1: EEGNet, zero-shot, no EA, no calib   : {CONDITION1_ZEROSHOT_ACC*100:.2f}%\n"
        f"    Condition 1b: EEGNet + 15% calib, NO EA (given)   : {CONDITION1B_NO_EA_CALIB_ACC*100:.2f}%\n"
        f"    THIS RUN — EEGNet + 15% calib, ea_mode={ea_mode:<11}: {grand_mean_acc*100:.2f}% "
        f"(cov_estimator={cov_estimator})\n"
    )

    if not pilot:
        os.makedirs(os.path.dirname(LOSO_CSV_PATH), exist_ok=True)
        write_header = not os.path.exists(LOSO_CSV_PATH)
        with open(LOSO_CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOSO_CSV_FIELDNAMES, restval="")
            if write_header:
                writer.writeheader()
            writer.writerows(csv_rows)
        log.info(f"  Saved: {LOSO_CSV_PATH}")

    results_payload = {
        "condition": "F7 — EEGNet + Euclidean/Riemannian Alignment",
        "pilot": pilot, "ea_mode": ea_mode, "cov_estimator": cov_estimator,
        "seeds": seeds, "n_folds": len(fold_subjects),
        "hyperparameters": {
            "cal_fraction": CAL_FRACTION, "pca_max_components": PCA_MAX_COMPONENTS,
            "logreg_C": LOGREG_C, "shrink_grid": SHRINK_GRID, "shrink_cv_folds": SHRINK_CV_FOLDS,
            "cov_shrinkage_fixed": COV_SHRINKAGE_FIXED, "n_permutations": N_PERMUTATIONS,
            "eegnet_F1": EEGNET_F1, "eegnet_D": EEGNET_D, "eegnet_F2": EEGNET_F2,
            "eegnet_dropout": EEGNET_DROPOUT,
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
        "fold_results_last_seed": fold_records_last_seed,
        "loso_csv_path": LOSO_CSV_PATH if not pilot else None,
        "reference_condition1_zeroshot_acc": CONDITION1_ZEROSHOT_ACC,
        "reference_condition1b_no_ea_calib_acc": CONDITION1B_NO_EA_CALIB_ACC,
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
    assert 0.0 <= grand_mean_acc <= 1.0, f"[C3 PLAUSIBILITY FAIL] grand mean accuracy {grand_mean_acc} outside [0,1]"
    for s_mean in per_seed_mean_acc:
        assert 0.0 <= s_mean <= 1.0, f"[C3 PLAUSIBILITY FAIL] per-seed mean accuracy {s_mean} outside [0,1]"
    if not pilot:
        assert len(fold_subjects) == 29, f"[C3 PLAUSIBILITY FAIL] expected 29 subjects on ::full, got {len(fold_subjects)}"
        assert len(seeds) == 5, f"[C3 PLAUSIBILITY FAIL] expected 5 seeds on ::full, got {len(seeds)}"
    log.info(f"  [C3] plausibility: {len(fold_subjects)} folds, {len(seeds)} seed(s), "
             f"all accuracies in [0,1] -- OK")

    return {
        "grand_mean_accuracy": grand_mean_acc, "grand_std_accuracy": grand_std_acc,
        "per_seed_mean_accuracy": per_seed_mean_acc,
        "subject_decodability_pre_alignment": subj_decode_acc_pre,
        "subject_decodability_post_alignment": subj_decode_acc_post,
        "n_folds": len(fold_subjects), "n_seeds": len(seeds),
        "output_path": OUTPUT_JSON if not pilot else None,
    }


# =============================================================================
# SECTION 3: LOCAL ENTRYPOINTS
# =============================================================================
@app.local_entrypoint(name="pilot")
def pilot_entrypoint(ea_mode: str = "pooled", cov_estimator: str = "fixed"):
    print("\n" + "="*70)
    print("  F7 — EEGNet + EA — PILOT (1 seed, 5 folds, fast diagnostic)")
    print("="*70)
    print(f"  ea_mode={ea_mode}  cov_estimator={cov_estimator}\n")
    results = run_eegnet_ea.remote(pilot=True, pilot_n_folds=5, ea_mode=ea_mode, cov_estimator=cov_estimator)
    print("\nPILOT RESULTS:")
    for k, v in results.items():
        print(f"  {k:<32}: {v}")


@app.local_entrypoint(name="full")
def full_entrypoint(ea_mode: str = "pooled", cov_estimator: str = "fixed"):
    print("\n" + "="*70)
    print("  F7 — EEGNet + EA — FULL (5 seeds x 29 folds)")
    print("="*70)
    print(f"  ea_mode={ea_mode}  cov_estimator={cov_estimator}\n")
    results = run_eegnet_ea.remote(pilot=False, ea_mode=ea_mode, cov_estimator=cov_estimator)
    print("\nFULL RESULTS:")
    for k, v in results.items():
        print(f"  {k:<32}: {v}")
