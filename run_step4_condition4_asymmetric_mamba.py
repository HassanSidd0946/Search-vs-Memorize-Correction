# =============================================================================
# command to run:
#   modal run run_step4_condition4_asymmetric_mamba.py::pilot   (Phase 1, 5-fold)
#   modal run run_step4_condition4_asymmetric_mamba.py::full    (Phase 2, 29-fold)
#   modal run run_step4_condition4_asymmetric_mamba.py::full --ea-mode per-subject --fusion-mode scaled-concat
# run_step4_condition4_asymmetric_mamba.py
#
# CONDITION 4 — ASYMMETRIC FUSION: RAW 1953-D SPATIAL TANGENT VECTOR
# + LIGHTWEIGHT TRUE-MAMBA TEMPORAL EMBEDDING (SELECTIVE SSM, PARALLEL SCAN)
#
# WHY THIS SCRIPT EXISTS (diagnostics from this session):
#   - Condition 4v2 (spatial-only: raw 1953-dim tangent -> PCA(35) -> 15%
#     shrinkage calibration, NO neural net at all) scored 67.79% full
#     29-fold. That's the number any dual-branch model has to add value on
#     top of, not regress from.
#   - The first dual-branch (Mamba + spatial) pilot bottlenecked the
#     spatial branch to a learned 32-dim embedding before fusion and
#     collapsed to 55.70% (5-fold). Ablations isolated two independent,
#     additive causes:
#       (a) ~5.7pp lost purely from compressing the 1953-dim tangent
#           vector to 32 dims with a learned projection before PCA ever
#           saw it (raw-tangent diagnostic: 65.99% vs. the bottlenecked
#           spatial-only ablation's 60.41%, same 5 subjects).
#       (b) ~4.7pp further lost from symmetric mean-pooling fusion
#           diluting the (already bottlenecked) spatial signal with the
#           temporal one.
#   - Conclusion: the spatial branch must stay RAW and uncompressed all
#     the way to the joint PCA/calibration stage. The temporal branch can
#     stay tiny — it's meant to add context, not replace the primary
#     signal — but it must be a genuinely trained Selective SSM (Mamba),
#     not a GRU/LSTM stand-in, and it must not be forced through a
#     symmetric pooling fusion with the (much larger) spatial vector.
#
# ARCHITECTURE — ASYMMETRIC FUSION:
#   Spatial branch:  EA-whitened trial covariance -> Log-Euclidean tangent
#                     space -> RAW 1953-dim vector. Never passed through
#                     any nn.Module — no learned projection, no bottleneck.
#   Temporal branch: EA-whitened raw signal (62ch x 251t) -> per-timestep
#                     Linear(62->16) -> ONE lightweight Mamba block
#                     (d_model=16, d_state=8, expand=2, causal depthwise
#                     conv, TRUE selective-scan SSM via a pure-PyTorch
#                     parallel/associative (Hillis-Steele) scan — NOT a
#                     sequential Python for-loop over 251 timesteps, and
#                     NOT torch.compile'd) -> mean-pool over time -> the
#                     raw 16-dim embedding (no extra projection head, so
#                     the legacy-fused vector is exactly 1953 + 16 = 1969
#                     dims). This branch has its own small classifier head
#                     and is pretrained end-to-end (cross-entropy, AdamW,
#                     early stopping) on the 28-subject pool, exactly like
#                     the spatial-only ablation's branch pretraining —
#                     THEN frozen.
#   Fusion:          torch.cat([raw_tangent_1953, temporal_emb_16]) done
#                     AFTER the temporal branch is frozen, at the NumPy
#                     feature level — no joint backprop between a huge raw
#                     tangent vector and tiny Mamba activations, no
#                     learned fusion layer. See F11 fusion-mode note below.
#   Calibration:     StandardScaler -> PCA(35) -> 15% few-shot shrinkage-
#                     blended LogisticRegression, fit/tuned exactly as
#                     every prior script this session (identical leakage
#                     controls: scaler + PCA + global clf fit ONLY on the
#                     28-subject training pool; local clf and shrink
#                     weight chosen via internal CV on D_cal only; D_test
#                     touched exactly once).
#
# GUARDRAILS HONORED:
#   - No torch.compile anywhere (caused memory deadlocks previously).
#   - No GRU/LSTM fallback — the temporal branch is a genuine selective SSM.
#   - Dropout=0.3, AdamW weight_decay=1e-2, early-stop patience=8 on the
#     temporal branch's pretraining (an internal-validation-split early
#     stop, NOT cross-seed/cross-run best-result cherry-picking — see F4
#     note below), to keep the tiny Mamba block from overfitting the
#     28-subject pool.
#   - Batch-level logging every 50 batches during temporal pretraining.
#
# F3/F4/F9/F11/F14 UPGRADE (Gate-1-rejection response, 2026-08-18):
#   - F3:  EA whitening now goes through the shared eeg_alignment.py module
#          (`--ea-mode {pooled,per-subject,riemannian}`, default "pooled").
#          The SAME aligned signal feeds both the spatial tangent branch and
#          the temporal Mamba branch. A one-time label-free assertion and
#          pre/post subject-decodability diagnostic are logged before the
#          LOSO loop starts.
#   - F4:  `full` runs a 5-seed loop (SEEDS below); every seed x fold result
#          is appended to the same tidy results/loso_runs.csv written by
#          run_step4_matched_spatial_control.py. `pilot` intentionally stays
#          single-seed (SEEDS[:1]) — it is a fast go/no-go diagnostic gate,
#          not a reported manuscript number, so 5x'ing its GPU cost buys
#          nothing; the "no best-run cherry-picking" guarantee applies to
#          REPORTED results, which only ever come from `full`. Within a
#          fold, temporal-branch early stopping (on the internal 10%
#          validation split carved out of the 28-subject training pool) is
#          kept — that is legitimate train-time model selection, not
#          cherry-picking across independent attempts, and never touches
#          the held-out subject.
#   - F9:  `--cov-estimator {fixed,lwf}` for the spatial branch's covariance
#          estimation ("lwf" = per-trial Ledoit-Wolf shrinkage, realised
#          lambda logged per fold). Does not affect the temporal branch,
#          which consumes the raw aligned time-domain signal, not covariances.
#   - F11: `--fusion-mode {legacy,scaled-concat,split-pca}`:
#            - "legacy":       original behaviour, raw concat of the 1953-d
#                               tangent vector and the 16-d temporal embedding.
#            - "scaled-concat": each block is independently z-scored (two
#                               PER-BLOCK StandardScalers, each fit on the
#                               28-subject training pool ONLY) before concat,
#                               so the joint PCA doesn't implicitly favour
#                               whichever block has larger raw magnitude.
#            - "split-pca":    each block is independently z-scored AND the
#                               spatial block gets its own PCA (also fit on
#                               the training pool only) BEFORE concatenation
#                               with the (untouched-by-PCA) temporal block;
#                               the existing joint StandardScaler->PCA(35)->
#                               calibration stage then runs on top, unchanged,
#                               for a controlled comparison across fusion arms.
#          Every fold logs `temporal_variance_share`: the fraction of the
#          FINAL joint-PCA's retained variance attributable to the temporal
#          block's dimensions (sum over components of
#          explained_variance_ratio_i * ||loading_i restricted to temporal
#          dims||^2, valid because each loading vector has unit norm).
#   - F14: extended metrics (AUC, Cohen's kappa, balanced accuracy, macro-F1,
#          sensitivity, specificity) plus a permutation-derived empirical
#          chance level (1000 label permutations per fold, reusing the
#          already-fit classifier's predictions) recorded per fold.
#
# TWO ENTRYPOINTS:
#   pilot -> subjects 01-05 only (Phase 1), single seed. Diagnostic bar: mean
#            acc must exceed 66.11%; ~66.5%+ is the bar to auto-proceed to Phase 2.
#   full  -> all 29 subjects, strict LOSO, 5 seeds (Phase 2). Writes the
#            manuscript-candidate results_condition4_asymmetric_dualbranch.json
#            and appends every seed x fold row to results/loso_runs.csv.
#
# Usage:
#   modal run run_step4_condition4_asymmetric_mamba.py::pilot
#   modal run run_step4_condition4_asymmetric_mamba.py::full
#   modal run run_step4_condition4_asymmetric_mamba.py::full --ea-mode riemannian --cov-estimator lwf --fusion-mode split-pca
# =============================================================================

import modal

app    = modal.App("bci-condition4-asymmetric-mamba")
volume = modal.Volume.from_name("eeg-data-vol")

RAW_DATA_PATH     = "/data/processed_eeg_all_subjects.npz"
CONDITION1B_JSON  = "/data/results_condition1b_eegnet_calibrated.json"
CONDITION3_JSON   = "/data/results_condition3_ea_zeroshot.json"
OUTPUT_JSON_PILOT = "/data/results_condition4_asymmetric_dualbranch_pilot5.json"
OUTPUT_JSON_FULL  = "/data/results_condition4_asymmetric_dualbranch.json"
LOSO_CSV_PATH     = "/data/results/loso_runs.csv"
VOLUME_PATH       = "/data"

# Reference numbers established earlier this session (for logging/comparison only)
CONDITION1B_FULL_MEAN_ACC        = 0.5564   # EEGNet + 15% calib, full 29-fold
CONDITION4V2_FULL_MEAN_ACC       = 0.6779   # spatial-only raw tangent, full 29-fold
CONDITION4V2_SUBSET_MEAN_ACC     = 0.6599   # spatial-only raw tangent, subjects 01-05 only
DUALBRANCH_PILOT_MEAN_ACC        = 0.5570   # bottlenecked dual-branch, 5-fold pilot
SPATIAL_ONLY_ABLATION_5FOLD_ACC  = 0.6041   # 32-dim bottlenecked spatial-only, 5-fold
PILOT_MUST_BEAT_ACC              = 0.6611   # diagnostic bar this pilot must clear
PILOT_GOOD_ACC                   = 0.6650   # bar for treating Phase 1 as a green light

SFREQ, N_CHANNELS = 250, 62

CAL_FRACTION         = 0.15
RANDOM_SEED          = 42
SEEDS                 = [42, 43, 44, 45, 46]   # F4: 5-seed loop for `full`, no best-of-N reporting
COV_SHRINKAGE_FIXED  = 0.1                      # F9: "fixed" cov-estimator mode
PCA_MAX_COMPONENTS   = 35
LOGREG_C             = 1.0
SHRINK_GRID          = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS      = 3
N_PERMUTATIONS        = 1000                    # F14: permutation-derived empirical chance level
SUBJECT_DECODABILITY_SPLITS = 5                 # F3: subject-decodability diagnostic

# Canonical results/loso_runs.csv schema, SHARED verbatim with
# run_step4_matched_spatial_control.py so both LOSO drivers append to one
# unified tidy table regardless of run order.
LOSO_CSV_FIELDNAMES = [
    "script", "condition", "seed", "ea_mode", "cov_estimator", "fusion_mode",
    "fold_index", "test_subject", "tangent_dim", "temporal_dim", "fused_dim", "n_temporal_params",
    "temporal_variance_share", "realized_cov_lambda_mean", "best_shrink_weight",
    "pre_calibration_acc", "post_calibration_acc",
    "sensitivity", "specificity", "f1", "macro_f1", "balanced_accuracy", "cohen_kappa", "roc_auc",
    "permutation_empirical_chance_level", "permutation_null_std_acc", "permutation_p_value",
]

# --- Temporal (Mamba) branch hyperparameters — per instructions, UNCHANGED ---
D_MODEL     = 16
D_STATE     = 8
N_LAYERS    = 1
EXPAND      = 2
DT_RANK     = 4
CONV_KERNEL = 4
DROPOUT     = 0.3

PRETRAIN_EPOCHS, PRETRAIN_BATCH, PRETRAIN_LR = 30, 32, 1e-3
PRETRAIN_WD, PRETRAIN_ES_PAT = 1e-2, 8
INTERNAL_VAL_FRAC   = 0.10
LOG_EVERY_N_BATCHES = 50

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.2.0", "numpy<2", "scikit-learn==1.4.2", "scipy")
    .add_local_python_source("eeg_alignment")
)


@app.function(image=image, gpu="L4", volumes={VOLUME_PATH: volume}, timeout=86400, memory=16384)
def run_condition4_asymmetric(pilot: bool = True, pilot_n_folds: int = 5,
                               ea_mode: str = "pooled", cov_estimator: str = "fixed",
                               fusion_mode: str = "legacy"):

    import os
    import csv
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
    from sklearn.covariance import LedoitWolf
    from sklearn.metrics import (
        confusion_matrix, f1_score, roc_auc_score, cohen_kappa_score, balanced_accuracy_score,
    )
    import logging, time, math, json, copy
    import eeg_alignment as ea

    assert ea_mode in ("pooled", "per-subject", "riemannian"), f"Unknown --ea-mode: {ea_mode!r}"
    assert cov_estimator in ("fixed", "lwf"), f"Unknown --cov-estimator: {cov_estimator!r}"
    assert fusion_mode in ("legacy", "scaled-concat", "split-pca"), f"Unknown --fusion-mode: {fusion_mode!r}"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("condition4-asymmetric")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device} | mode: {'PILOT (5-fold)' if pilot else 'FULL (29-fold LOSO)'} "
             f"| ea_mode={ea_mode} | cov_estimator={cov_estimator} | fusion_mode={fusion_mode}")

    raw = np.load(RAW_DATA_PATH, allow_pickle=True)
    X_np = raw["X"].astype(np.float32)
    y_np = raw["y"].astype(np.int64)
    subjects_np = raw["subjects"]
    N, C, T = X_np.shape
    N_CLASSES = int(y_np.max()) + 1
    assert C == N_CHANNELS and N_CLASSES == 2
    log.info(f"X: {X_np.shape} | Subjects total: {len(np.unique(subjects_np))}")
    # F-SILENT hardening: a silently-failed subject upstream (e.g. sub-09's
    # truncated .eeg export, see AUDIT.md D2) must never produce a
    # clean-looking-but-incomplete full LOSO run.
    if not pilot:
        assert len(np.unique(subjects_np)) == 29, (
            f"Expected exactly 29 subjects (30 - sub-09 exclusion per AUDIT.md D2), "
            f"got {len(np.unique(subjects_np))}: {sorted(np.unique(subjects_np).tolist())}"
        )

    all_subjects = sorted(np.unique(subjects_np).tolist())
    unique_subjects = all_subjects[:pilot_n_folds] if pilot else all_subjects
    output_json = OUTPUT_JSON_PILOT if pilot else OUTPUT_JSON_FULL
    seeds = [RANDOM_SEED] if pilot else SEEDS  # F4: pilot stays single-seed (see module docstring)
    log.info(f"Running folds: {unique_subjects} | seeds: {seeds}")

    condition1b_per_subject_acc = {}
    if os.path.exists(CONDITION1B_JSON):
        with open(CONDITION1B_JSON) as f:
            c1b = json.load(f)
        for rec in c1b.get("fold_results", []):
            condition1b_per_subject_acc[str(rec["test_subject"])] = float(rec["post_calibration_acc"])

    condition3_per_subject_acc = {}
    if os.path.exists(CONDITION3_JSON):
        with open(CONDITION3_JSON) as f:
            c3 = json.load(f)
        for rec in c3.get("fold_results", []):
            condition3_per_subject_acc[str(rec["test_subject"])] = float(rec["test_accuracy"])

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
    # TRUE MAMBA — lightweight, pure PyTorch, parallel (Hillis-Steele) scan.
    # No mamba_ssm / causal_conv1d CUDA kernels, no torch.compile, no
    # sequential Python loop over timesteps. UNCHANGED from prior session.
    # =========================================================================
    def parallel_scan(A, Bt):
        """
        Associative (Hillis-Steele) inclusive scan solving the linear
        recurrence h_t = A_t * h_{t-1} + Bt_t  (h_{-1} = 0) along dim=1 (T).
        A, Bt: (batch, T, d_inner, d_state). log2(T) sequential rounds,
        each a fully vectorized elementwise tensor op over the WHOLE
        sequence at once — this IS the parallel scan, not a per-timestep
        Python loop.
        """
        A = A.clone()
        Bt = Bt.clone()
        Tlen = A.shape[1]
        offset = 1
        while offset < Tlen:
            A_shift = torch.ones_like(A)
            B_shift = torch.zeros_like(Bt)
            A_shift[:, offset:] = A[:, :-offset]
            B_shift[:, offset:] = Bt[:, :-offset]
            Bt = A * B_shift + Bt
            A = A * A_shift
            offset *= 2
        return Bt

    class SelectiveSSMBlock(nn.Module):
        """One lightweight true-Mamba block: causal depthwise conv -> SiLU
        -> input-dependent (selective) delta/B/C -> selective scan (parallel,
        associative) -> output gate -> output projection. d_model=16,
        d_state=8 keeps this tiny and fast."""

        def __init__(self, d_model=D_MODEL, d_state=D_STATE, expand=EXPAND,
                     dt_rank=DT_RANK, conv_kernel=CONV_KERNEL, dropout=DROPOUT):
            super().__init__()
            d_inner = expand * d_model
            self.d_inner, self.d_state, self.dt_rank = d_inner, d_state, dt_rank

            self.norm = nn.LayerNorm(d_model)
            self.in_proj = nn.Linear(d_model, 2 * d_inner)
            self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=conv_kernel,
                                     groups=d_inner, padding=conv_kernel - 1, bias=True)
            self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state)
            self.dt_proj = nn.Linear(dt_rank, d_inner)
            A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1))
            self.A_log = nn.Parameter(A_log)           # (d_inner, d_state) — S4D-real init
            self.D = nn.Parameter(torch.ones(d_inner))  # skip connection
            self.out_proj = nn.Linear(d_inner, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # x: (B, T, d_model)
            B_, T_, _ = x.shape
            residual = x
            x = self.norm(x)
            xz = self.in_proj(x)                          # (B,T,2*d_inner)
            x_in, z = xz.chunk(2, dim=-1)                  # each (B,T,d_inner)

            x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :T_]   # causal (symmetric pad + truncate)
            x_conv = F.silu(x_conv.transpose(1, 2))        # (B,T,d_inner)

            x_dbc = self.x_proj(x_conv)                    # (B,T,dt_rank+2*d_state)
            delta_raw, Bmat, Cmat = torch.split(
                x_dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            delta = F.softplus(self.dt_proj(delta_raw))    # (B,T,d_inner)

            A = -torch.exp(self.A_log)                     # (d_inner,d_state), strictly negative
            deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))     # (B,T,d_inner,d_state)
            deltaBx = delta.unsqueeze(-1) * Bmat.unsqueeze(2) * x_conv.unsqueeze(-1)  # (B,T,d_inner,d_state)

            h = parallel_scan(deltaA, deltaBx)              # (B,T,d_inner,d_state)
            y = (h * Cmat.unsqueeze(2)).sum(-1) + self.D.unsqueeze(0).unsqueeze(0) * x_conv  # (B,T,d_inner)

            y = y * F.silu(z)                               # gate
            out = self.dropout(self.out_proj(y))            # (B,T,d_model)
            return residual + out

    class MambaTemporalBranch(nn.Module):
        """EA-whitened raw signal (B,62,251) -> per-timestep Linear(62->16)
        -> N_LAYERS selective-SSM blocks -> mean-pool over time -> the raw
        16-dim embedding. This IS the full temporal embedding fed to
        fusion — no extra projection head, so the legacy-fused vector is
        exactly 1953 + D_MODEL = 1969 dims."""

        def __init__(self, n_channels=N_CHANNELS, d_model=D_MODEL, n_layers=N_LAYERS, dropout=DROPOUT):
            super().__init__()
            self.in_proj = nn.Linear(n_channels, d_model)
            self.blocks = nn.ModuleList([SelectiveSSMBlock(d_model=d_model, dropout=dropout) for _ in range(n_layers)])
            self.final_norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x_signal):
            # x_signal: (B, C=62, T=251)
            x = x_signal.transpose(1, 2)         # (B,T,C)
            x = self.in_proj(x)                  # (B,T,d_model)
            for blk in self.blocks:
                x = blk(x)
            x = self.final_norm(x)
            x = x.mean(dim=1)                    # mean-pool over time -> (B,d_model)
            return self.dropout(x)

    class TemporalOnlyModel(nn.Module):
        """Temporal branch + its own small classifier head, used ONLY to
        pretrain the Mamba block via cross-entropy on the 28-subject pool.
        Discarded after pretraining — only forward_features() (the raw
        16-dim embedding) survives into fusion."""

        def __init__(self, n_channels=N_CHANNELS, d_model=D_MODEL, n_classes=2):
            super().__init__()
            self.temporal = MambaTemporalBranch(n_channels, d_model)
            self.classifier = nn.Linear(d_model, n_classes)

        def forward_features(self, x_signal):
            return self.temporal(x_signal)

        def forward(self, x_signal):
            return self.classifier(self.forward_features(x_signal))

    # =========================================================================
    # TRAIN / EVAL / EXTRACT / CALIBRATION UTILITIES
    # =========================================================================
    def make_loader(X_arr, y, batch_size, shuffle, drop_last=False):
        ds = TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                           pin_memory=True, drop_last=drop_last)

    def train_one_epoch(model, loader, optimizer, criterion, epoch):
        model.train()
        total_loss, n_correct, n_total = 0.0, 0, 0
        for batch_idx, (Xb, yb) in enumerate(loader, start=1):
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_correct += (logits.argmax(1) == yb).sum().item()
            n_total += yb.size(0)
            if batch_idx % LOG_EVERY_N_BATCHES == 0:
                log.info(f"      ep {epoch} batch {batch_idx}/{len(loader)} | "
                         f"running loss={total_loss/batch_idx:.4f} acc={n_correct/n_total:.3f}")
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
    def extract_temporal_features(model, X_signal, batch_size=64):
        model.eval()
        feats = []
        n = X_signal.shape[0]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            Xb = torch.from_numpy(X_signal[start:end]).to(device)
            feats.append(model.forward_features(Xb).cpu().numpy())
        return np.concatenate(feats, axis=0)

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
        """Identical algorithm to every prior script this session."""
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
    # F11: fusion-mode dispatch. All three arms return (feat_train28, feat_k)
    # with the temporal block ALWAYS as the last D_MODEL columns (needed for
    # the temporal_variance_share computation below). The existing joint
    # StandardScaler -> PCA(<=35) -> shrinkage-calibration stage runs
    # UNCHANGED after this, for a controlled comparison across fusion arms.
    # =========================================================================
    def fuse_features(tan_train28, temporal_emb_train28, tan_k, temporal_emb_k, seed):
        if fusion_mode == "legacy":
            feat_train28 = np.concatenate([tan_train28, temporal_emb_train28], axis=1)
            feat_k = np.concatenate([tan_k, temporal_emb_k], axis=1)
            return feat_train28, feat_k

        # "scaled-concat" and "split-pca" both start with per-block scalers
        # fit on the 28-subject TRAINING POOL ONLY.
        spatial_scaler = StandardScaler().fit(tan_train28)
        temporal_scaler = StandardScaler().fit(temporal_emb_train28)
        tan_train28_z = spatial_scaler.transform(tan_train28)
        tan_k_z = spatial_scaler.transform(tan_k)
        temp_train28_z = temporal_scaler.transform(temporal_emb_train28)
        temp_k_z = temporal_scaler.transform(temporal_emb_k)

        if fusion_mode == "scaled-concat":
            feat_train28 = np.concatenate([tan_train28_z, temp_train28_z], axis=1)
            feat_k = np.concatenate([tan_k_z, temp_k_z], axis=1)
            return feat_train28, feat_k

        # "split-pca": spatial block ALSO gets its own PCA, fit on the
        # training pool only. Temporal block (already only 16-d) is left
        # standardized, not further PCA'd.
        n_spatial_components = min(PCA_MAX_COMPONENTS, tan_train28_z.shape[1] - 1, tan_train28_z.shape[0] - 1)
        spatial_pca = PCA(n_components=n_spatial_components, random_state=seed).fit(tan_train28_z)
        tan_train28_pca = spatial_pca.transform(tan_train28_z)
        tan_k_pca = spatial_pca.transform(tan_k_z)
        feat_train28 = np.concatenate([tan_train28_pca, temp_train28_z], axis=1)
        feat_k = np.concatenate([tan_k_pca, temp_k_z], axis=1)
        return feat_train28, feat_k

    def compute_temporal_variance_share(pca, temporal_block_width):
        """Fraction of the joint PCA's retained variance attributable to the
        temporal block's dims. Each pca.components_[i] is a unit-norm
        loading vector over the fused feature space, so the squared norm of
        its restriction to the temporal-dim slice IS that component's
        temporal share; weighting by explained_variance_ratio_ and summing
        gives the overall share."""
        components = pca.components_                      # (n_components, fused_dim)
        evr = pca.explained_variance_ratio_
        temporal_slice = components[:, -temporal_block_width:]
        temporal_sq_norm_per_component = np.sum(temporal_slice ** 2, axis=1)
        return float(np.sum(evr * temporal_sq_norm_per_component))

    # =========================================================================
    # F3 diagnostics: label-free assertion + subject-decodability, computed
    # ONCE up front on the full subject pool being used this run (before any
    # per-fold LOSO split, and independent of `seed`).
    # =========================================================================
    mu_all = X_np.mean(axis=(0, 2), keepdims=True)
    sd_all = X_np.std(axis=(0, 2), keepdims=True) + 1e-6
    X_all_z = ((X_np - mu_all) / sd_all).astype(np.float32)
    covs_all_pre, _ = compute_trial_covariances(X_all_z, seeds[0])
    tan_all_pre = ea.tangent_vectorize(covs_all_pre)

    y_decoy_a = np.random.RandomState(0).randint(0, 2, size=len(subjects_np))
    y_decoy_b = 1 - y_decoy_a
    ea.verify_label_free(covs_all_pre, subjects_np, mode=ea_mode, y_decoy_a=y_decoy_a, y_decoy_b=y_decoy_b)
    log.info("  [F3] verify_label_free PASSED (alignment provably ignores labels)")

    W_all = ea.euclidean_align(covs_all_pre, subjects_np, mode=ea_mode)
    if ea_mode == "pooled":
        X_all_aligned = ea.apply_ea_whitening_signal(X_all_z, W_all)
    else:
        X_all_aligned = ea.apply_ea_whitening_signal_per_subject(X_all_z, subjects_np, W_all)
    covs_all_post, _ = compute_trial_covariances(X_all_aligned, seeds[0])
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
            "is_pilot": pilot, "ea_mode": ea_mode, "cov_estimator": cov_estimator, "fusion_mode": fusion_mode,
            "subject_decodability_pre_alignment": subj_decode_acc_pre,
            "subject_decodability_post_alignment": subj_decode_acc_post,
            "subject_decodability_chance_level": subj_decode_chance,
            "subject_decodability_n_subjects": subj_decode_n,
        }
        halt_json_path = output_json + ".c2_halt_diagnostic.json"
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
    # LOSO — ASYMMETRIC FUSION, F4 seed loop
    # =========================================================================
    csv_rows = []
    per_seed_mean_acc = []
    fold_records_last_seed = []

    for seed in seeds:
        torch.manual_seed(seed)
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
            if ea_mode == "pooled":
                X_train28_aligned = ea.apply_ea_whitening_signal(X_train28_z, W_train).astype(np.float32)
                X_k_aligned = ea.apply_ea_whitening_signal(X_k_z, W_train).astype(np.float32)
            else:
                X_train28_aligned = ea.apply_ea_whitening_signal_per_subject(
                    X_train28_z, subs_train28, W_train).astype(np.float32)
                subs_k = np.full(len(X_k_z), test_sub)
                W_k = ea.euclidean_align(covs_k, subs_k, mode=ea_mode)
                X_k_aligned = ea.apply_ea_whitening_signal_per_subject(X_k_z, subs_k, W_k).astype(np.float32)

            covs_train28_aligned, _ = compute_trial_covariances(X_train28_aligned, seed)
            covs_k_aligned, _ = compute_trial_covariances(X_k_aligned, seed)

            # --- SPATIAL: raw tangent vector, untouched by any nn.Module ---
            tan_train28 = ea.tangent_vectorize(covs_train28_aligned)
            tan_k = ea.tangent_vectorize(covs_k_aligned)
            tangent_dim = tan_train28.shape[1]

            # --- TEMPORAL: pretrain the Mamba-only model on the 28-subject pool ---
            Xtr_sig, Xval_sig, ytr, yval = train_test_split(
                X_train28_aligned, y_train28, test_size=INTERNAL_VAL_FRAC,
                stratify=y_train28, random_state=seed)
            train_loader = make_loader(Xtr_sig, ytr, PRETRAIN_BATCH, True, drop_last=True)
            val_loader = make_loader(Xval_sig, yval, PRETRAIN_BATCH, False)
            criterion = nn.CrossEntropyLoss()

            temporal_model = TemporalOnlyModel().to(device)
            n_params = sum(p.numel() for p in temporal_model.parameters() if p.requires_grad)
            optimizer = torch.optim.AdamW(temporal_model.parameters(), lr=PRETRAIN_LR, weight_decay=PRETRAIN_WD)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS, eta_min=PRETRAIN_LR * 0.01)

            log.info(f"  Temporal Mamba model: {n_params:,} params "
                     f"(d_model={D_MODEL} d_state={D_STATE} n_layers={N_LAYERS} expand={EXPAND})")
            t0 = time.time()
            best_val_loss, best_state, es_ctr = math.inf, None, 0
            for epoch in range(1, PRETRAIN_EPOCHS + 1):
                tr_loss, tr_acc = train_one_epoch(temporal_model, train_loader, optimizer, criterion, epoch)
                scheduler.step()
                val_loss, val_acc = evaluate(temporal_model, val_loader, criterion)
                if epoch == 1 or epoch % 5 == 0:
                    log.info(f"    Ep {epoch:02d}/{PRETRAIN_EPOCHS} | train acc={tr_acc:.3f} | "
                             f"val loss={val_loss:.4f} acc={val_acc:.3f} | {time.time()-t0:.0f}s elapsed")
                if val_loss < best_val_loss:
                    best_val_loss, best_state, es_ctr = val_loss, copy.deepcopy(temporal_model.state_dict()), 0
                else:
                    es_ctr += 1
                    if es_ctr >= PRETRAIN_ES_PAT:
                        log.info(f"    Early stop at epoch {epoch}")
                        break
            temporal_model.load_state_dict(best_state)
            for p in temporal_model.parameters():
                p.requires_grad = False
            temporal_model.eval()
            log.info(f"  Temporal pretraining wall time: {time.time()-t0:.0f}s")

            temporal_emb_train28 = extract_temporal_features(temporal_model, X_train28_aligned)
            temporal_emb_k = extract_temporal_features(temporal_model, X_k_aligned)

            # --- FUSION (F11) ---
            feat_train28, feat_k = fuse_features(tan_train28, temporal_emb_train28, tan_k, temporal_emb_k, seed)
            fused_dim_pre_joint = feat_train28.shape[1]

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
            temporal_variance_share = compute_temporal_variance_share(pca, D_MODEL)

            coef_final, icpt_final, best_shrink, global_clf = fit_shrinkage_classifier(
                X_train28_pca, y_train28, X_cal_pca, y_cal, seed)
            pre_cal_acc = float((global_clf.predict(X_test_pca) == y_test).mean())
            final_scores = linear_predict_scores(coef_final, icpt_final, X_test_pca)
            final_preds = (final_scores > 0).astype(int)
            best_test_acc = float((final_preds == y_test).mean())
            metrics = compute_extended_metrics(y_test, final_preds, final_scores)
            perm_stats = permutation_null_stats(y_test, final_preds, seed=seed)

            c1b_acc = condition1b_per_subject_acc.get(str(test_sub))
            log.info(
                f"  RESULT -> pre_cal={pre_cal_acc:.4f}  post_cal={best_test_acc:.4f} (shrink={best_shrink:.2f}) "
                f"| auc={metrics['roc_auc']:.4f} kappa={metrics['cohen_kappa']:.4f} "
                f"| temporal_var_share={temporal_variance_share:.4f} "
                f"| tangent_dim={tangent_dim} temporal_dim={D_MODEL} fused_dim_pre_joint={fused_dim_pre_joint}"
            )
            if c1b_acc is not None:
                log.info(f"      Δ vs EEGNet+Calib (sub-{test_sub}: {c1b_acc*100:.2f}%) -> {(best_test_acc-c1b_acc)*100:+.2f} pp")

            row = {
                "script": "run_step4_condition4_asymmetric_mamba",
                "condition": "asymmetric_fusion",
                "seed": seed, "ea_mode": ea_mode, "cov_estimator": cov_estimator, "fusion_mode": fusion_mode,
                "fold_index": fold_idx, "test_subject": str(test_sub),
                "tangent_dim": int(tangent_dim), "temporal_dim": int(D_MODEL),
                "fused_dim": int(fused_dim_pre_joint), "n_temporal_params": int(n_params),
                "temporal_variance_share": temporal_variance_share,
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
            fold_records.append({
                **row, "confusion_matrix": metrics["confusion_matrix"],
                "condition1b_eegnet_calib_acc": c1b_acc,
                "condition3_zero_shot_acc": condition3_per_subject_acc.get(str(test_sub)),
            })
            all_test_acc.append(best_test_acc)
            log.info(f"  Fold elapsed: {time.time()-fold_start:.0f}s")

            del temporal_model, optimizer, scheduler, best_state
            if device.type == "cuda":
                torch.cuda.empty_cache()

        seed_mean_acc = float(np.mean(all_test_acc))
        per_seed_mean_acc.append(seed_mean_acc)
        fold_records_last_seed = fold_records
        log.info(f"\nSEED {seed} DONE — mean acc {seed_mean_acc:.4f} over {len(unique_subjects)} folds")

    grand_mean_acc = float(np.mean(per_seed_mean_acc))
    grand_std_acc = float(np.std(per_seed_mean_acc))

    log.info(f"\n{'='*70}\n  ASYMMETRIC FUSION — {len(seeds)} seed(s) x {len(unique_subjects)} folds ({'PILOT' if pilot else 'FULL'})\n{'='*70}")
    log.info(f"  Grand mean-of-seed-means Acc: {grand_mean_acc:.4f} ± {grand_std_acc:.4f}")
    log.info(f"  Per-seed means: {per_seed_mean_acc}")
    log.info(
        "\n  COMPARISON:\n"
        f"    EEGNet + 15% Calib (full 29-fold)          : {CONDITION1B_FULL_MEAN_ACC*100:.2f}%\n"
        f"    Condition 4v2 spatial-only (full 29-fold)  : {CONDITION4V2_FULL_MEAN_ACC*100:.2f}%\n"
        f"    Condition 4v2 spatial-only (subjects 01-05): {CONDITION4V2_SUBSET_MEAN_ACC*100:.2f}%\n"
        f"    Bottlenecked dual-branch pilot (5-fold)    : {DUALBRANCH_PILOT_MEAN_ACC*100:.2f}%\n"
        f"    THIS RUN ({'pilot' if pilot else 'full'}, ea_mode={ea_mode}, fusion_mode={fusion_mode}) : {grand_mean_acc*100:.2f}%\n"
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
        "condition": "Condition 4 — ASYMMETRIC FUSION (raw 1953-d tangent + lightweight True-Mamba temporal)",
        "is_pilot": pilot, "n_folds": len(unique_subjects),
        "ea_mode": ea_mode, "cov_estimator": cov_estimator, "fusion_mode": fusion_mode, "seeds": seeds,
        "hyperparameters": {
            "d_model": D_MODEL, "d_state": D_STATE, "n_layers": N_LAYERS, "expand": EXPAND,
            "dt_rank": DT_RANK, "conv_kernel": CONV_KERNEL, "dropout": DROPOUT,
            "pretrain_wd": PRETRAIN_WD, "pretrain_es_patience": PRETRAIN_ES_PAT,
            "cal_fraction": CAL_FRACTION, "pca_max_components": PCA_MAX_COMPONENTS,
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
        "fold_results_last_seed": fold_records_last_seed,
        "loso_csv_path": LOSO_CSV_PATH if not pilot else None,
        "reference_condition1b_full_mean_acc": CONDITION1B_FULL_MEAN_ACC,
        "reference_condition4v2_full_mean_acc": CONDITION4V2_FULL_MEAN_ACC,
        "reference_condition4v2_subset_mean_acc": CONDITION4V2_SUBSET_MEAN_ACC,
        "reference_dualbranch_pilot_mean_acc": DUALBRANCH_PILOT_MEAN_ACC,
    }
    with open(output_json, "w") as f:
        json.dump(results_payload, f, indent=2)
    volume.commit()
    log.info(f"  Saved: {output_json}")

    # =========================================================================
    # C3 plausibility assertions -- printed next to the numbers, not silent.
    # Run AFTER both writes above so a failing assertion never suppresses the
    # diagnostic artifacts.
    # =========================================================================
    assert 0.0 <= grand_mean_acc <= 1.0, f"[C3 PLAUSIBILITY FAIL] grand mean accuracy {grand_mean_acc} outside [0,1]"
    for s_mean in per_seed_mean_acc:
        assert 0.0 <= s_mean <= 1.0, f"[C3 PLAUSIBILITY FAIL] per-seed mean accuracy {s_mean} outside [0,1]"
    assert len(per_seed_mean_acc) == len(seeds), (
        f"[C3 PLAUSIBILITY FAIL] expected {len(seeds)} per-seed means, got {len(per_seed_mean_acc)}"
    )
    if not pilot:
        assert len(unique_subjects) == 29, f"[C3 PLAUSIBILITY FAIL] expected 29 subjects on ::full, got {len(unique_subjects)}"
        assert len(seeds) == 5, f"[C3 PLAUSIBILITY FAIL] expected 5 seeds on ::full, got {len(seeds)}"
    CHANCE_LEVEL = 0.5
    if grand_mean_acc <= CHANCE_LEVEL:
        log.warning(
            f"  [C3] plausibility WARNING: grand mean accuracy ({grand_mean_acc:.4f}) is at or below "
            f"chance ({CHANCE_LEVEL}) -- unexpected for this pipeline. Worth a second look before trusting this number."
        )
    log.info(f"  [C3] plausibility: {len(unique_subjects)} subjects, {len(seeds)} seed(s), "
             f"all accuracies in [0,1] -- OK")

    return {
        "grand_mean_accuracy": grand_mean_acc, "grand_std_accuracy": grand_std_acc,
        "per_seed_mean_accuracy": per_seed_mean_acc,
        "subject_decodability_pre_alignment": subj_decode_acc_pre,
        "subject_decodability_post_alignment": subj_decode_acc_post,
        "output_path": output_json, "is_pilot": pilot, "n_folds": len(unique_subjects),
    }


@app.local_entrypoint(name="pilot")
def pilot_entrypoint(ea_mode: str = "pooled", cov_estimator: str = "fixed", fusion_mode: str = "legacy"):
    print("Condition 4 — ASYMMETRIC FUSION — PHASE 1: 5-fold pilot (subjects 01-05)")
    print("Raw 1953-d tangent + lightweight True-Mamba (d_model=16,d_state=8,n_layers=1),")
    print(f"parallel-scan selective SSM, no bottleneck, no GRU/LSTM, no torch.compile.")
    print(f"ea_mode={ea_mode}  cov_estimator={cov_estimator}  fusion_mode={fusion_mode}  (single seed, diagnostic gate)")
    print(f"Must beat {PILOT_MUST_BEAT_ACC*100:.2f}%; {PILOT_GOOD_ACC*100:.2f}%+ is the green light for Phase 2.\n")
    results = run_condition4_asymmetric.remote(
        pilot=True, pilot_n_folds=5, ea_mode=ea_mode, cov_estimator=cov_estimator, fusion_mode=fusion_mode)
    print("\nPHASE 1 PILOT RESULTS:")
    for k, v in results.items():
        print(f"  {k:<32}: {v}")
    mean_acc = results["grand_mean_accuracy"]
    if mean_acc >= PILOT_GOOD_ACC:
        print(f"\n  >= {PILOT_GOOD_ACC*100:.2f}% — GREEN LIGHT for Phase 2 (full 29-fold run).")
    elif mean_acc > PILOT_MUST_BEAT_ACC:
        print(f"\n  Beat {PILOT_MUST_BEAT_ACC*100:.2f}% but below the {PILOT_GOOD_ACC*100:.2f}% green-light bar — review before Phase 2.")
    else:
        print(f"\n  Did NOT beat {PILOT_MUST_BEAT_ACC*100:.2f}% — do not proceed to Phase 2 without investigating.")


@app.local_entrypoint(name="full")
def full_entrypoint(ea_mode: str = "pooled", cov_estimator: str = "fixed", fusion_mode: str = "legacy"):
    print("Condition 4 — ASYMMETRIC FUSION — PHASE 2: full 29-fold strict LOSO, 5 seeds")
    print(f"ea_mode={ea_mode}  cov_estimator={cov_estimator}  fusion_mode={fusion_mode}  seeds={SEEDS}\n")
    results = run_condition4_asymmetric.remote(
        pilot=False, ea_mode=ea_mode, cov_estimator=cov_estimator, fusion_mode=fusion_mode)
    print("\nPHASE 2 FULL RESULTS:")
    for k, v in results.items():
        print(f"  {k:<32}: {v}")
