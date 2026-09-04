# =============================================================================
# command to run:
#   modal run run_step4_condition4_joint_dualbranch_mamba.py::pilot   (Phase 1, 5-fold)
#   modal run run_step4_condition4_joint_dualbranch_mamba.py::full    (Phase 2, 29-fold)
# run_step4_condition4_joint_dualbranch_mamba.py
#
# F-JOINT FIX 1 — GENUINE JOINT END-TO-END DUAL-BRANCH TRAINING (AUDIT.md
# Phase 0.5 Priority 5 / Fix-ID table, CRITICAL correctness defect)
#
# WHY THIS SCRIPT EXISTS:
#   AUDIT.md's F-JOINT finding: run_step4_condition4_asymmetric_mamba.py (the
#   script this one supersedes as the "primary" variant, PRESERVED UNCHANGED
#   as the legacy pretrain-freeze arm for comparison) trains its temporal
#   Mamba branch STANDALONE against its own private classifier head
#   (TemporalOnlyModel, its own nn.Linear(16,2), independent early stopping),
#   then FREEZES it and concatenates its 16-D output with the raw 1953-D
#   spatial tangent vector at the NumPy level — outside any autograd graph.
#   No gradient ever flows jointly between the two branches in the legacy
#   script, and the manuscript's Fig. 2 "single end-to-end dual-branch
#   network trained jointly" description is factually wrong about it.
#   Mechanistically, this also means the temporal branch is trained to be
#   REDUNDANT with the spatial branch (it never sees the spatial branch's
#   errors/residual signal during its own training) rather than
#   COMPLEMENTARY to it — the direct explanation for why the matched-control
#   ablation found the Mamba branch's marginal contribution statistically
#   indistinguishable from zero (70.78% vs. 71.28%, Wilcoxon p=0.92).
#
#   This script implements AUDIT.md's recommended Fix 1 ("methodologically
#   cleanest... makes Fig. 2's 'joint dual-branch' description literally
#   true"), using the cost-reducing compromise the fix description itself
#   explicitly permits: "wrapping [the spatial branch] as a fixed
#   (non-learned) differentiable feature extractor feeding a learned
#   fusion+classifier on top of both branches" — rather than the
#   Medium-High-risk full PyTorch reimplementation of the EA/Log-Euclidean
#   tangent-space math (which has no learned parameters in EITHER script,
#   legacy or this one, so there is nothing to gain from making it
#   differentiable; it was never the thing F-JOINT's defect is about).
#
# WHAT IS ACTUALLY JOINT HERE (the fix):
#   A single nn.Module (`JointDualBranchModel`) holds BOTH the temporal
#   Mamba branch AND the fusion+classifier head. One forward pass concats
#   the FIXED (precomputed, no-grad) 1953-D spatial tangent vector with the
#   LEARNED 16-D temporal embedding, and ONE CrossEntropyLoss is backpropped
#   through the whole graph in ONE optimizer loop (single AdamW, single
#   CosineAnnealingLR, single early-stopping criterion on the FUSED
#   validation loss — not the temporal branch's own private loss). The
#   temporal branch's weight updates are therefore driven by how much it
#   improves the FUSED prediction beyond what the spatial features alone
#   already give — the exact mechanism the legacy script lacks, and the
#   direct fix for F-JOINT defect (b) (redundant-by-construction), not just
#   defect (a) (the Fig. 2 mislabeling).
#
# WHAT IS STILL NOT LEARNED (explicitly disclosed, not hidden):
#   - The EA whitening + Log-Euclidean tangent-space map stays exactly as
#     in every prior script this session: NumPy/eigendecomposition, fit
#     once per fold on the 28-subject training pool, no learned parameters.
#     It was never learned in the legacy script either — F-JOINT's defect
#     is entirely about the temporal-branch/fusion coupling, not this step.
#   - AFTER the joint-training stage (this script's actual fix), the
#     now-jointly-trained temporal branch is frozen and its embeddings are
#     re-extracted, then run through the SAME StandardScaler -> PCA(35) ->
#     15% few-shot shrinkage-calibrated LogisticRegression pipeline as every
#     other condition in this paper, for cross-condition comparability. This
#     final calibration classifier is deliberately classical (not part of
#     the joint autograd graph) — it is a separate, well-validated few-shot
#     subject-calibration mechanism used identically across the whole paper,
#     not the thing Fig. 2 / F-JOINT claims is jointly trained.
#
# GUARDRAILS HONORED (unchanged from the legacy script):
#   - No torch.compile anywhere.
#   - No GRU/LSTM fallback — genuine selective SSM (Mamba), same
#     hyperparameters (d_model=16, d_state=8, n_layers=1, expand=2),
#     parallel/associative (Hillis-Steele) scan, not a sequential loop.
#   - Dropout=0.3, AdamW weight_decay=1e-2, early-stop patience=8.
#   - Batch-level logging every 50 batches during joint training.
#
# TWO ENTRYPOINTS (mirrors the legacy script):
#   pilot -> subjects 01-05 only (Phase 1 diagnostic).
#   full  -> all 29 subjects, strict LOSO (Phase 2). Writes
#            results_condition4_joint_dualbranch.json.
#
# Usage:
#   modal run run_step4_condition4_joint_dualbranch_mamba.py::pilot
#   modal run run_step4_condition4_joint_dualbranch_mamba.py::full
# =============================================================================

import modal

app    = modal.App("bci-condition4-joint-dualbranch-mamba")
volume = modal.Volume.from_name("eeg-data-vol")

RAW_DATA_PATH     = "/data/processed_eeg_all_subjects.npz"
CONDITION1B_JSON  = "/data/results_condition1b_eegnet_calibrated.json"
CONDITION3_JSON   = "/data/results_condition3_ea_zeroshot.json"
LEGACY_FULL_JSON  = "/data/results_condition4_asymmetric_dualbranch.json"
OUTPUT_JSON_PILOT = "/data/results_condition4_joint_dualbranch_pilot5.json"
OUTPUT_JSON_FULL  = "/data/results_condition4_joint_dualbranch.json"
VOLUME_PATH       = "/data"

# Reference numbers established earlier this session (for logging/comparison only)
CONDITION1B_FULL_MEAN_ACC        = 0.5564   # EEGNet + 15% calib, full 29-fold
CONDITION4V2_FULL_MEAN_ACC       = 0.6779   # spatial-only raw tangent, full 29-fold (original leaky-EA pipe)
MATCHED_SPATIAL_CONTROL_MEAN_ACC = 0.7078   # matched spatial-only control, full 29-fold, pool-only EA
LEGACY_ASYMMETRIC_FUSION_ACC     = 0.7128   # legacy pretrain-freeze asymmetric fusion, full 29-fold

SFREQ, N_CHANNELS = 250, 62

CAL_FRACTION       = 0.15
RANDOM_SEED        = 42
COV_SHRINKAGE      = 0.1
PCA_MAX_COMPONENTS = 35
LOGREG_C           = 1.0
SHRINK_GRID        = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS    = 3

# --- Temporal (Mamba) branch hyperparameters — IDENTICAL to the legacy script ---
D_MODEL     = 16
D_STATE     = 8
N_LAYERS    = 1
EXPAND      = 2
DT_RANK     = 4
CONV_KERNEL = 4
DROPOUT     = 0.3

# --- Joint-training stage (supersedes the legacy script's "pretraining" stage) ---
JOINT_EPOCHS, JOINT_BATCH, JOINT_LR = 30, 32, 1e-3
JOINT_WD, JOINT_ES_PAT = 1e-2, 8
INTERNAL_VAL_FRAC   = 0.10
LOG_EVERY_N_BATCHES = 50

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.2.0", "numpy<2", "scikit-learn==1.4.2", "scipy")
)


@app.function(image=image, gpu="L4", volumes={VOLUME_PATH: volume}, timeout=86400, memory=16384)
def run_condition4_joint_dualbranch(pilot: bool = True, pilot_n_folds: int = 5):

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
    from sklearn.metrics import confusion_matrix, f1_score
    import logging, time, math, json, copy, os

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("condition4-joint-dualbranch")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device} | mode: {'PILOT (5-fold)' if pilot else 'FULL (29-fold LOSO)'}")

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
    log.info(f"Running folds: {unique_subjects}")

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

    legacy_per_subject_acc = {}
    if os.path.exists(LEGACY_FULL_JSON):
        with open(LEGACY_FULL_JSON) as f:
            legacy = json.load(f)
        for rec in legacy.get("fold_results", []):
            legacy_per_subject_acc[str(rec["test_subject"])] = float(rec["post_calibration_acc"])

    # =========================================================================
    # RIEMANNIAN / EA UTILITIES — identical to every prior script this session.
    # No learned parameters here in either the legacy or this script; F-JOINT
    # is entirely about the temporal-branch/fusion coupling, not this step.
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
        """RAW tangent vectors — no learned compression, ever."""
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
    # TRUE MAMBA — IDENTICAL architecture/hyperparameters to the legacy script.
    # Pure PyTorch, parallel (Hillis-Steele) scan. No mamba_ssm /
    # causal_conv1d CUDA kernels, no torch.compile, no sequential loop.
    # =========================================================================
    def parallel_scan(A, Bt):
        """
        Associative (Hillis-Steele) inclusive scan solving the linear
        recurrence h_t = A_t * h_{t-1} + Bt_t  (h_{-1} = 0) along dim=1 (T).
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
        """One lightweight true-Mamba block — identical to the legacy script."""

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
        16-dim embedding. Identical to the legacy script's temporal branch —
        what changes in this script is HOW this branch's weights get their
        gradient, not its architecture."""

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

    class JointDualBranchModel(nn.Module):
        """THE FIX: a single nn.Module holding both branches + the fusion
        classifier head. forward() takes the FIXED (precomputed, no learned
        parameters) spatial tangent vector alongside the raw EA-aligned
        signal, runs the signal through the temporal branch, concatenates,
        and produces logits from ONE fusion_classifier. One CrossEntropyLoss
        on these logits backprops through self.temporal AND
        self.fusion_classifier jointly in the same optimizer step — this is
        the literal "single nn.Module containing both branches + fusion +
        classifier head, backprop the final cross-entropy loss through the
        whole graph in one optimizer loop" that AUDIT.md's F-JOINT fix 1
        specifies. The spatial tangent vector has no parameters of its own,
        so no gradient needs to (or does) flow into it — it is the
        "fixed (non-learned) differentiable feature extractor" the fix
        description explicitly allows substituting for a full EA/tangent-
        space PyTorch reimplementation."""

        def __init__(self, spatial_dim, n_channels=N_CHANNELS, d_model=D_MODEL, n_classes=2):
            super().__init__()
            self.temporal = MambaTemporalBranch(n_channels, d_model)
            self.fusion_classifier = nn.Linear(spatial_dim + d_model, n_classes)

        def forward_temporal_features(self, x_signal):
            return self.temporal(x_signal)

        def forward(self, spatial_tan, x_signal):
            temporal_emb = self.forward_temporal_features(x_signal)   # LEARNED, part of the graph
            fused = torch.cat([spatial_tan, temporal_emb], dim=-1)    # spatial_tan is fixed/no-grad
            return self.fusion_classifier(fused)

    # =========================================================================
    # TRAIN / EVAL / EXTRACT / CALIBRATION UTILITIES
    # =========================================================================
    def make_joint_loader(spatial_arr, signal_arr, y, batch_size, shuffle, drop_last=False):
        ds = TensorDataset(torch.from_numpy(spatial_arr), torch.from_numpy(signal_arr), torch.from_numpy(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                           pin_memory=True, drop_last=drop_last)

    def train_one_epoch_joint(model, loader, optimizer, criterion, epoch):
        model.train()
        total_loss, n_correct, n_total = 0.0, 0, 0
        for batch_idx, (Sb, Xb, yb) in enumerate(loader, start=1):
            Sb, Xb, yb = Sb.to(device), Xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(Sb, Xb)
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
    def evaluate_joint(model, loader, criterion):
        model.eval()
        total_loss, n_correct, n_total = 0.0, 0, 0
        for Sb, Xb, yb in loader:
            Sb, Xb, yb = Sb.to(device), Xb.to(device), yb.to(device)
            logits = model(Sb, Xb)
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
            feats.append(model.forward_temporal_features(Xb).cpu().numpy())
        return np.concatenate(feats, axis=0)

    def compute_binary_metrics(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        return {"sensitivity": float(sens), "specificity": float(spec),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)), "confusion_matrix": cm.tolist()}

    def linear_predict(coef, intercept, X):
        return ((X @ coef.T + intercept).ravel() > 0).astype(int)

    def fit_shrinkage_classifier(X_train_pca, y_train, X_cal_pca, y_cal):
        """Identical algorithm to every prior script this session — the
        few-shot calibration methodology is unchanged by F-JOINT's fix,
        only the temporal embeddings feeding into it are now jointly-trained."""
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
        return coef_final, icpt_final, best_shrink, global_clf

    # =========================================================================
    # LOSO — JOINT DUAL-BRANCH FUSION
    # =========================================================================
    fold_records, all_test_acc = [], []

    for fold_idx, test_sub in enumerate(unique_subjects):
        fold_start = time.time()
        log.info(f"\n{'='*70}\n  FOLD {fold_idx+1}/{len(unique_subjects)} — sub-{test_sub}\n{'='*70}")

        is_holdout = subjects_np == test_sub
        X_train28, y_train28 = X_np[~is_holdout], y_np[~is_holdout]
        X_k, y_k = X_np[is_holdout], y_np[is_holdout]

        mu = X_train28.mean(axis=(0, 2), keepdims=True)
        sd = X_train28.std(axis=(0, 2), keepdims=True) + 1e-6
        X_train28_z = ((X_train28 - mu) / sd).astype(np.float32)
        X_k_z = ((X_k - mu) / sd).astype(np.float32)

        W = fit_ea_whitening(X_train28_z)
        X_train28_aligned = apply_ea_whitening_signal(X_train28_z, W).astype(np.float32)
        X_k_aligned = apply_ea_whitening_signal(X_k_z, W).astype(np.float32)

        # --- SPATIAL: raw tangent vector, FIXED, untouched by any nn.Module ---
        tan_train28 = tangent_vectorize(trial_covariances(X_train28_aligned))
        tan_k = tangent_vectorize(trial_covariances(X_k_aligned))
        tangent_dim = tan_train28.shape[1]

        # --- JOINT TRAINING STAGE: temporal branch + fusion classifier, ONE loss ---
        Str_sig, Sval_sig, Xtr_sig, Xval_sig, ytr, yval = train_test_split(
            tan_train28, X_train28_aligned, y_train28, test_size=INTERNAL_VAL_FRAC,
            stratify=y_train28, random_state=RANDOM_SEED)
        train_loader = make_joint_loader(Str_sig, Xtr_sig, ytr, JOINT_BATCH, True, drop_last=True)
        val_loader = make_joint_loader(Sval_sig, Xval_sig, yval, JOINT_BATCH, False)
        criterion = nn.CrossEntropyLoss()

        joint_model = JointDualBranchModel(spatial_dim=tangent_dim).to(device)
        n_params_temporal = sum(p.numel() for p in joint_model.temporal.parameters() if p.requires_grad)
        n_params_total = sum(p.numel() for p in joint_model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(joint_model.parameters(), lr=JOINT_LR, weight_decay=JOINT_WD)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=JOINT_EPOCHS, eta_min=JOINT_LR * 0.01)

        log.info(f"  Joint model: {n_params_total:,} total params "
                 f"({n_params_temporal:,} temporal, {n_params_total - n_params_temporal:,} fusion head) "
                 f"(d_model={D_MODEL} d_state={D_STATE} n_layers={N_LAYERS} expand={EXPAND})")
        t0 = time.time()
        best_val_loss, best_state, es_ctr = math.inf, None, 0
        for epoch in range(1, JOINT_EPOCHS + 1):
            tr_loss, tr_acc = train_one_epoch_joint(joint_model, train_loader, optimizer, criterion, epoch)
            scheduler.step()
            val_loss, val_acc = evaluate_joint(joint_model, val_loader, criterion)
            if epoch == 1 or epoch % 5 == 0:
                log.info(f"    Ep {epoch:02d}/{JOINT_EPOCHS} | train acc={tr_acc:.3f} | "
                         f"val loss={val_loss:.4f} acc={val_acc:.3f} | {time.time()-t0:.0f}s elapsed")
            if val_loss < best_val_loss:
                best_val_loss, best_state, es_ctr = val_loss, copy.deepcopy(joint_model.state_dict()), 0
            else:
                es_ctr += 1
                if es_ctr >= JOINT_ES_PAT:
                    log.info(f"    Early stop at epoch {epoch}")
                    break
        joint_model.load_state_dict(best_state)
        for p in joint_model.parameters():
            p.requires_grad = False
        joint_model.eval()
        log.info(f"  Joint training wall time: {time.time()-t0:.0f}s")

        # --- Freeze the now-jointly-trained temporal branch; re-extract embeddings.
        #     The joint fusion_classifier's weights are discarded here — they were
        #     only the training-time head that supplied the joint gradient signal;
        #     the actual fix is that self.temporal's weights were shaped by it. ---
        temporal_emb_train28 = extract_temporal_features(joint_model, X_train28_aligned)
        temporal_emb_k = extract_temporal_features(joint_model, X_k_aligned)

        # --- FUSION for calibration: straight concat, same as legacy script ---
        feat_train28 = np.concatenate([tan_train28, temporal_emb_train28], axis=1)
        feat_k = np.concatenate([tan_k, temporal_emb_k], axis=1)
        fused_dim = feat_train28.shape[1]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - CAL_FRACTION), random_state=RANDOM_SEED)
        cal_idx, test_idx = next(sss.split(feat_k, y_k))
        feat_cal, y_cal = feat_k[cal_idx], y_k[cal_idx]
        feat_test, y_test = feat_k[test_idx], y_k[test_idx]

        scaler = StandardScaler()
        feat_train28_z = scaler.fit_transform(feat_train28)
        feat_cal_z = scaler.transform(feat_cal)
        feat_test_z = scaler.transform(feat_test)

        n_components = min(PCA_MAX_COMPONENTS, feat_train28_z.shape[1] - 1, feat_train28_z.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
        X_train28_pca = pca.fit_transform(feat_train28_z)
        X_cal_pca = pca.transform(feat_cal_z)
        X_test_pca = pca.transform(feat_test_z)

        coef_final, icpt_final, best_shrink, global_clf = fit_shrinkage_classifier(
            X_train28_pca, y_train28, X_cal_pca, y_cal)
        pre_cal_acc = float((global_clf.predict(X_test_pca) == y_test).mean())
        final_preds = linear_predict(coef_final, icpt_final, X_test_pca)
        best_test_acc = float((final_preds == y_test).mean())
        metrics = compute_binary_metrics(y_test, final_preds)

        c1b_acc = condition1b_per_subject_acc.get(str(test_sub))
        legacy_acc = legacy_per_subject_acc.get(str(test_sub))
        log.info(f"  RESULT -> pre_cal={pre_cal_acc:.4f}  post_cal={best_test_acc:.4f} (shrink={best_shrink:.2f}) "
                 f"| tangent_dim={tangent_dim} temporal_dim={D_MODEL} fused_dim={fused_dim}")
        if legacy_acc is not None:
            log.info(f"      Δ vs legacy pretrain-freeze arm (sub-{test_sub}: {legacy_acc*100:.2f}%) -> "
                     f"{(best_test_acc-legacy_acc)*100:+.2f} pp")

        fold_records.append({
            "fold_index": fold_idx, "test_subject": str(test_sub),
            "tangent_dim": int(tangent_dim), "temporal_dim": int(D_MODEL), "fused_dim": int(fused_dim),
            "n_temporal_params": int(n_params_temporal), "n_joint_model_params": int(n_params_total),
            "best_shrink_weight": float(best_shrink),
            "pre_calibration_acc": pre_cal_acc, "post_calibration_acc": best_test_acc,
            **metrics,
            "condition1b_eegnet_calib_acc": c1b_acc,
            "condition3_zero_shot_acc": condition3_per_subject_acc.get(str(test_sub)),
            "legacy_pretrain_freeze_acc": legacy_acc,
        })
        all_test_acc.append(best_test_acc)
        log.info(f"  Fold elapsed: {time.time()-fold_start:.0f}s")

        del joint_model, optimizer, scheduler, best_state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_acc, std_acc = float(np.mean(all_test_acc)), float(np.std(all_test_acc))

    log.info(f"\n{'='*70}\n  JOINT DUAL-BRANCH FUSION — {len(unique_subjects)} folds ({'PILOT' if pilot else 'FULL'})\n{'='*70}")
    log.info(f"  Mean ± Std Acc: {mean_acc:.4f} ± {std_acc:.4f}")
    log.info(
        "\n  COMPARISON:\n"
        f"    EEGNet + 15% Calib (full 29-fold)                : {CONDITION1B_FULL_MEAN_ACC*100:.2f}%\n"
        f"    Condition 4v2 spatial-only, ORIGINAL/leaky pipe  : {CONDITION4V2_FULL_MEAN_ACC*100:.2f}%\n"
        f"    Matched spatial-only control, all 62 channels   : {MATCHED_SPATIAL_CONTROL_MEAN_ACC*100:.2f}%\n"
        f"    Legacy pretrain-freeze asymmetric fusion (full)  : {LEGACY_ASYMMETRIC_FUSION_ACC*100:.2f}%\n"
        f"    THIS RUN — joint dual-branch ({'pilot' if pilot else 'full'})            : {mean_acc*100:.2f}%\n"
    )

    results_payload = {
        "condition": "Condition 4 — JOINT END-TO-END DUAL-BRANCH FUSION (F-JOINT fix 1: single nn.Module, "
                      "shared loss, joint backprop through temporal branch + fusion head; spatial tangent "
                      "vector used as a fixed, non-learned differentiable feature extractor)",
        "is_pilot": pilot, "n_folds": len(unique_subjects),
        "hyperparameters": {
            "d_model": D_MODEL, "d_state": D_STATE, "n_layers": N_LAYERS, "expand": EXPAND,
            "dt_rank": DT_RANK, "conv_kernel": CONV_KERNEL, "dropout": DROPOUT,
            "joint_wd": JOINT_WD, "joint_es_patience": JOINT_ES_PAT,
            "cal_fraction": CAL_FRACTION, "pca_max_components": PCA_MAX_COMPONENTS,
        },
        "fold_results": fold_records, "mean_accuracy": mean_acc, "std_accuracy": std_acc,
        "reference_condition1b_full_mean_acc": CONDITION1B_FULL_MEAN_ACC,
        "reference_condition4v2_full_mean_acc": CONDITION4V2_FULL_MEAN_ACC,
        "reference_matched_spatial_control_mean_acc": MATCHED_SPATIAL_CONTROL_MEAN_ACC,
        "reference_legacy_pretrain_freeze_asymmetric_fusion_acc": LEGACY_ASYMMETRIC_FUSION_ACC,
    }
    with open(output_json, "w") as f:
        json.dump(results_payload, f, indent=2)
    volume.commit()
    log.info(f"  Saved: {output_json}")

    # =========================================================================
    # C3 plausibility assertions -- printed next to the numbers, not silent.
    # Run AFTER the write above so a failing assertion never suppresses the
    # diagnostic artifact.
    # =========================================================================
    assert 0.0 <= mean_acc <= 1.0, f"[C3 PLAUSIBILITY FAIL] mean accuracy {mean_acc} outside [0,1]"
    if not pilot:
        assert len(unique_subjects) == 29, f"[C3 PLAUSIBILITY FAIL] expected 29 subjects on full run, got {len(unique_subjects)}"
    for rec in fold_records:
        assert rec["n_temporal_params"] > 0, (
            f"[C3 PLAUSIBILITY FAIL] sub-{rec['test_subject']} n_temporal_params={rec['n_temporal_params']} "
            f"(must be > 0 -- a zero here previously indicated a MAC/param-counting wiring bug elsewhere in "
            f"this codebase, see F15's known follow-up)"
        )
    log.info(f"  [C3] plausibility: {len(unique_subjects)} folds, mean acc in [0,1], "
             f"all fold n_temporal_params > 0 -- OK")

    return {"mean_accuracy": mean_acc, "std_accuracy": std_acc, "output_path": output_json,
            "is_pilot": pilot, "n_folds": len(unique_subjects)}


@app.local_entrypoint(name="pilot")
def pilot_entrypoint():
    print("Condition 4 — JOINT DUAL-BRANCH FUSION — PHASE 1: 5-fold pilot (subjects 01-05)")
    print("F-JOINT fix 1: temporal Mamba branch + fusion classifier head trained")
    print("jointly (one nn.Module, one loss, one optimizer loop) instead of the")
    print("legacy script's standalone-pretrain-then-freeze arm (preserved unchanged")
    print(f"in run_step4_condition4_asymmetric_mamba.py for comparison).\n")
    results = run_condition4_joint_dualbranch.remote(pilot=True, pilot_n_folds=5)
    print("\nPHASE 1 PILOT RESULTS:")
    for k, v in results.items():
        print(f"  {k:<20}: {v}")
    print(f"\n  Compare against legacy pretrain-freeze arm ({LEGACY_ASYMMETRIC_FUSION_ACC*100:.2f}% full-29) "
          f"and matched spatial-only control ({MATCHED_SPATIAL_CONTROL_MEAN_ACC*100:.2f}% full-29) before "
          f"deciding whether to run Phase 2 (full 29-fold).")


@app.local_entrypoint(name="full")
def full_entrypoint():
    print("Condition 4 — JOINT DUAL-BRANCH FUSION — PHASE 2: full 29-fold strict LOSO\n")
    results = run_condition4_joint_dualbranch.remote(pilot=False)
    print("\nPHASE 2 FULL RESULTS:")
    for k, v in results.items():
        print(f"  {k:<20}: {v}")
