# =============================================================================
# command to run (TWO PHASES, run in order):
#   modal run run_capacity_sweep.py::screen
#   modal run run_capacity_sweep.py::full
#   modal run run_capacity_sweep.py::screen --ea-mode per-subject --cov-estimator lwf
#   modal run run_capacity_sweep.py::full --top-k 3
# run_capacity_sweep.py
#
# F-CAPACITY — TEMPORAL-BRANCH CAPACITY SWEEP + MATCHED-PARAMETER LSTM CONTROL
# (Audit Fix, AUDIT.md Fix-ID table row F-CAPACITY, motivated by Q5: "D_MODEL/
# D_STATE/N_LAYERS were never validated by ANY method — they are asserted
# defaults... F-CAPACITY's sweep is still mandatory -- it is the only way to
# know whether the temporal branch's null result reflects the data or
# reflects under-capacity, and right now the codebase has literally never
# tried a second value.")
#
# WHY THIS SCRIPT EXISTS:
#   Every Mamba temporal branch in this codebase (run_step4_condition4_
#   asymmetric_mamba.py, run_step4_condition4_joint_dualbranch_mamba.py) uses
#   exactly one hardcoded hyperparameter triple: d_model=16, d_state=8,
#   n_layers=1. If the temporal branch's marginal contribution to fusion
#   turns out to be small, a reviewer's first question is "did you even try
#   a bigger model?" -- and right now the honest answer is no. This script
#   answers that question directly, in ISOLATION from the spatial tangent
#   branch and the fusion/calibration machinery (which would otherwise
#   confound "temporal capacity" with "fusion strategy" effects) by
#   evaluating the temporal branch ALONE, ZERO-SHOT, exactly like
#   Condition 3's own baseline protocol (CONDITION3_BASELINE_ACC=0.5552,
#   the closest existing reference point for "temporal signal alone, no
#   calibration").
#
# GRID (per AUDIT.md's F-CAPACITY row, verbatim):
#   d_model  in {16, 32, 64, 128}
#   d_state  in {8, 16, 32}
#   n_layers in {1, 2}
#   => 4 x 3 x 2 = 24 configurations. expand=2, dt_rank=4, conv_kernel=4,
#   dropout=0.3 and ALL pretraining hyperparameters (epochs/batch/lr/wd/
#   early-stop patience) are held FIXED at the original driver's values
#   across every config, so that any accuracy difference between configs is
#   attributable to ARCHITECTURE CAPACITY, not to a differing training
#   recipe. dt_rank is intentionally NOT scaled with d_model (kept at 4
#   everywhere) -- this matches the fixed value already used at d_model=16
#   in the original driver; AUDIT.md's grid explicitly names only
#   d_model/d_state/n_layers as swept dimensions.
#
# TWO-PHASE PROTOCOL (per AUDIT.md: "screening then 3-config x5-seed"):
#   PHASE 1 (`::screen`, this is the "Largest architecture-search cost"
#   item AUDIT.md flags -- kept to a SINGLE seed to bound it): for every
#   one of the 24 configs, pretrain + zero-shot-evaluate across the full
#   29-fold LOSO (or `pilot_n_folds` folds in pilot mode) at ONE seed
#   (RANDOM_SEED=42). EA/z-score/covariance work is computed ONCE PER FOLD
#   (outer loop), not once per config (inner loop), since it does not
#   depend on the temporal architecture -- this avoids 24x redundant spatial
#   preprocessing. Results ranked by grand mean zero-shot accuracy across
#   folds; top-`TOP_K_CONFIGS` (default 3) configs + full ranked table
#   saved to `/data/results_capacity_screening.json`.
#   PHASE 2 (`::full`): loads the Phase-1 JSON from the volume, takes its
#   top-K configs, builds ONE matched-parameter LSTM control sized (via
#   binary search over hidden_size, num_layers pinned to the BEST config's
#   n_layers) to match the #1-ranked Mamba config's total trainable
#   parameter count, then re-runs all (K Mamba configs + 1 LSTM control)
#   arms through the FULL F4 5-seed x 29-fold LOSO protocol for
#   statistically defensible final numbers. Requires `::screen` to have
#   been run first (asserts the screening JSON exists on the volume, with
#   an explicit instruction in the error message if not).
#
# MATCHED-PARAMETER LSTM CONTROL:
#   A plain `nn.LSTM(input_size=N_CHANNELS, hidden_size=H, num_layers=L)`
#   (batch_first, single direction) + LayerNorm + mean-pool over time +
#   linear classifier head -- structurally the simplest possible recurrent
#   analogue of MambaTemporalBranch (same in/out shape contract, same
#   final_norm + mean-pool + dropout + Linear head), with L pinned to the
#   best Mamba config's n_layers and H found via binary search (LSTM
#   parameter count is monotone increasing in hidden_size for fixed
#   num_layers, so binary search over H converges exactly) to match that
#   config's total trainable parameter count as closely as possible. This
#   answers "is Mamba's selective-SSM mechanism doing something an
#   equally-sized plain recurrent network couldn't," not just "is bigger
#   better."
#
# F3/F4/F9 (F14 metrics also computed; F11 fusion-mode N/A -- no fusion in
# this script, temporal branch classifies alone):
#   - F3: EA via the shared eeg_alignment.py module, `--ea-mode
#         {none,pooled,per-subject,riemannian}` (default "pooled").
#   - F4: Phase 1 is intentionally single-seed (screening only, never a
#         reported number); Phase 2 runs the full 5-seed loop for the
#         configs that survive screening -- the only "selection" happening
#         is choosing WHICH ARCHITECTURE to report on Phase-1 accuracy
#         (disclosed, not hidden), never a best-of-N seed pick within a
#         fixed architecture.
#   - F9: `--cov-estimator {fixed,lwf}` for the EA-fitting covariance only
#         (the temporal branch itself consumes the aligned raw signal, not
#         covariances).
#
# EVALUATION PROTOCOL: ZERO-SHOT (no 15% calibration split), exactly like
#   F5's classical baselines and Condition 3's own baseline -- the temporal-
#   only classifier head is used directly on 100% of the held-out subject's
#   trials. `pre_calibration_acc` == `post_calibration_acc` in the shared
#   CSV for the same reason as F5 (no calibration stage exists here).
#
# CSV SCHEMA NOTE: reuses `temporal_dim` (Mamba's d_model / LSTM's hidden
# size) and `n_temporal_params` (actual trainable parameter count, verified
# by direct instantiation+count, not just the search target). `tangent_dim`/
# `fused_dim`/`fusion_mode`/`best_shrink_weight` all blank (n/a -- no
# spatial branch, no fusion, no calibration stage in this script).
#
# LEAKAGE CONTROLS:
#   1. Every config's pretraining uses the 28-subject training pool only;
#      held-out subject k's trials are touched exactly once, at final
#      zero-shot evaluation. Early-stopping validation is carved from the
#      28 training subjects only (INTERNAL_VAL_FRAC).
#   2. EA whitening is label-free by construction (F3); the held-out
#      subject's own W (per-subject/riemannian modes) is computed from its
#      own UNLABELED trials only.
#   3. Model weights are frozen (eval mode, no_grad) before extracting
#      predictions on subject k.
#
# Usage: modal run run_capacity_sweep.py::screen
#        modal run run_capacity_sweep.py::full --top-k 3
# =============================================================================

import modal

app    = modal.App("bci-f-capacity-sweep")
volume = modal.Volume.from_name("eeg-data-vol")

RAW_DATA_PATH        = "/data/processed_eeg_all_subjects.npz"
SCREENING_JSON        = "/data/results_capacity_screening.json"
FULL_JSON             = "/data/results_capacity_full.json"
LOSO_CSV_PATH         = "/data/results/loso_runs.csv"
VOLUME_PATH           = "/data"

CONDITION3_BASELINE_ACC = 0.5552   # zero-shot reference point (temporal signal, no calibration)

SFREQ, N_CHANNELS = 250, 62

RANDOM_SEED          = 42
SEEDS                = [42, 43, 44, 45, 46]   # F4: Phase-2 5-seed loop
COV_SHRINKAGE_FIXED  = 0.1                     # F9: "fixed" cov-estimator mode

# --- F-CAPACITY grid (exactly AUDIT.md's stated dimensions) ---
D_MODEL_GRID   = [16, 32, 64, 128]
D_STATE_GRID   = [8, 16, 32]
N_LAYERS_GRID  = [1, 2]
TOP_K_CONFIGS  = 3

# --- Fixed (not swept) Mamba architecture hyperparameters ---
EXPAND      = 2
DT_RANK     = 4
CONV_KERNEL = 4
DROPOUT     = 0.3

# --- Fixed (not swept) temporal-branch pretraining hyperparameters, IDENTICAL
# across every config so accuracy differences are attributable to capacity,
# not training recipe. Verbatim values from run_step4_condition4_asymmetric_mamba.py. ---
PRETRAIN_EPOCHS, PRETRAIN_BATCH, PRETRAIN_LR = 30, 32, 1e-3
PRETRAIN_WD, PRETRAIN_ES_PAT = 1e-2, 8
INTERNAL_VAL_FRAC   = 0.10
LOG_EVERY_N_BATCHES = 50

# Canonical results/loso_runs.csv schema, SHARED verbatim with every other
# LOSO driver in this codebase.
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
    .pip_install("torch==2.2.0", "numpy<2", "scikit-learn==1.4.2", "scipy")
    .add_local_python_source("eeg_alignment")
)


# =============================================================================
# Note: the parametrized Mamba temporal-only classes and LSTM matched-control
# classes are duplicated verbatim inside both Modal functions below (Modal
# serializes each @app.function's closure independently -- this mirrors how
# F9/F14 helpers are already duplicated across every other driver in this
# codebase rather than factored into a shared importable module).
# =============================================================================


@app.function(image=image, gpu="L4", volumes={VOLUME_PATH: volume}, timeout=86400, memory=16384)
def run_capacity_screening(pilot: bool = True, pilot_n_folds: int = 5,
                            ea_mode: str = "pooled", cov_estimator: str = "fixed"):

    import os
    import csv
    import copy
    import math
    import time
    import json
    import logging
    import itertools
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from sklearn.covariance import LedoitWolf
    import eeg_alignment as ea

    assert ea_mode in ("none", "pooled", "per-subject", "riemannian"), f"Unknown --ea-mode: {ea_mode!r}"
    assert cov_estimator in ("fixed", "lwf"), f"Unknown --cov-estimator: {cov_estimator!r}"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("f-capacity-screen")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device} | mode: {'PILOT' if pilot else 'FULL'} | ea_mode={ea_mode} | cov_estimator={cov_estimator}")

    raw = np.load(RAW_DATA_PATH, allow_pickle=True)
    X_np = raw["X"].astype(np.float32)
    y_np = raw["y"].astype(np.int64)
    subjects_np = raw["subjects"]
    N, C, T = X_np.shape
    N_CLASSES = int(y_np.max()) + 1
    assert C == N_CHANNELS and N_CLASSES == 2
    if not pilot:
        assert len(np.unique(subjects_np)) == 29, (
            f"Expected exactly 29 subjects (30 - sub-09 exclusion per AUDIT.md D2), "
            f"got {len(np.unique(subjects_np))}: {sorted(np.unique(subjects_np).tolist())}"
        )

    all_subjects = sorted(np.unique(subjects_np).tolist())
    fold_subjects = all_subjects[:pilot_n_folds] if pilot else all_subjects
    configs = [
        {"d_model": dm, "d_state": ds, "n_layers": nl}
        for dm, ds, nl in itertools.product(D_MODEL_GRID, D_STATE_GRID, N_LAYERS_GRID)
    ]
    log.info(f"Screening {len(configs)} configs x {len(fold_subjects)} folds @ seed={RANDOM_SEED}")

    # =========================================================================
    # F9: covariance estimator dispatch
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
    # Parametrized Mamba temporal-only stack (see header for design notes)
    # =========================================================================
    def parallel_scan(A, Bt):
        A = A.clone(); Bt = Bt.clone()
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
        def __init__(self, d_model, d_state, expand=EXPAND, dt_rank=DT_RANK,
                     conv_kernel=CONV_KERNEL, dropout=DROPOUT):
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
            self.A_log = nn.Parameter(A_log)
            self.D = nn.Parameter(torch.ones(d_inner))
            self.out_proj = nn.Linear(d_inner, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            B_, T_, _ = x.shape
            residual = x
            x = self.norm(x)
            xz = self.in_proj(x)
            x_in, z = xz.chunk(2, dim=-1)
            x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :T_]
            x_conv = F.silu(x_conv.transpose(1, 2))
            x_dbc = self.x_proj(x_conv)
            delta_raw, Bmat, Cmat = torch.split(x_dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            delta = F.softplus(self.dt_proj(delta_raw))
            A = -torch.exp(self.A_log)
            deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
            deltaBx = delta.unsqueeze(-1) * Bmat.unsqueeze(2) * x_conv.unsqueeze(-1)
            h = parallel_scan(deltaA, deltaBx)
            y = (h * Cmat.unsqueeze(2)).sum(-1) + self.D.unsqueeze(0).unsqueeze(0) * x_conv
            y = y * F.silu(z)
            out = self.dropout(self.out_proj(y))
            return residual + out

    class MambaTemporalBranch(nn.Module):
        def __init__(self, n_channels, d_model, d_state, n_layers, dropout=DROPOUT):
            super().__init__()
            self.in_proj = nn.Linear(n_channels, d_model)
            self.blocks = nn.ModuleList(
                [SelectiveSSMBlock(d_model=d_model, d_state=d_state, dropout=dropout) for _ in range(n_layers)])
            self.final_norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x_signal):
            x = x_signal.transpose(1, 2)
            x = self.in_proj(x)
            for blk in self.blocks:
                x = blk(x)
            x = self.final_norm(x)
            x = x.mean(dim=1)
            return self.dropout(x)

    class TemporalOnlyModel(nn.Module):
        def __init__(self, n_channels, d_model, d_state, n_layers, n_classes=2):
            super().__init__()
            self.temporal = MambaTemporalBranch(n_channels, d_model, d_state, n_layers)
            self.classifier = nn.Linear(d_model, n_classes)

        def forward_features(self, x_signal):
            return self.temporal(x_signal)

        def forward(self, x_signal):
            return self.classifier(self.forward_features(x_signal))

    def make_loader(X_arr, y, batch_size, shuffle, drop_last=False):
        ds = TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                           pin_memory=True, drop_last=drop_last)

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

    def pretrain_and_zeroshot_eval(model, X_train28_aligned, y_train28, X_k_aligned, y_k, seed):
        Xtr_sig, Xval_sig, ytr, yval = train_test_split(
            X_train28_aligned, y_train28, test_size=INTERNAL_VAL_FRAC, stratify=y_train28, random_state=seed)
        train_loader = make_loader(Xtr_sig, ytr, PRETRAIN_BATCH, True, drop_last=True)
        val_loader = make_loader(Xval_sig, yval, PRETRAIN_BATCH, False)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=PRETRAIN_WD)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS, eta_min=PRETRAIN_LR * 0.01)
        best_val_loss, best_state, es_ctr = math.inf, None, 0
        for epoch in range(1, PRETRAIN_EPOCHS + 1):
            train_one_epoch(model, train_loader, optimizer, criterion)
            scheduler.step()
            val_loss, val_acc = evaluate(model, val_loader, criterion)
            if val_loss < best_val_loss:
                best_val_loss, best_state, es_ctr = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                es_ctr += 1
                if es_ctr >= PRETRAIN_ES_PAT:
                    break
        model.load_state_dict(best_state)
        model.eval()
        k_loader = make_loader(X_k_aligned, y_k, PRETRAIN_BATCH, False)
        all_logits = []
        with torch.no_grad():
            for Xb, yb in k_loader:
                Xb = Xb.to(device)
                all_logits.append(model(Xb).cpu())
        logits = torch.cat(all_logits, dim=0)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = logits.argmax(dim=1).numpy().astype(np.int64)
        scores = probs[:, 1] - probs[:, 0]
        return preds, scores

    # =========================================================================
    # Screening loop: OUTER = folds (EA/covariance computed once per fold),
    # INNER = 24 configs (avoids redundant spatial preprocessing).
    # =========================================================================
    csv_rows = []
    acc_by_config = {(c["d_model"], c["d_state"], c["n_layers"]): [] for c in configs}
    n_params_by_config = {}

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

        covs_train28, _ = compute_trial_covariances(X_train28_z, RANDOM_SEED)
        covs_k, _ = compute_trial_covariances(X_k_z, RANDOM_SEED)

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

        for cfg in configs:
            d_model, d_state, n_layers = cfg["d_model"], cfg["d_state"], cfg["n_layers"]
            torch.manual_seed(RANDOM_SEED)
            model = TemporalOnlyModel(N_CHANNELS, d_model, d_state, n_layers).to(device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            n_params_by_config[(d_model, d_state, n_layers)] = n_params

            preds, scores = pretrain_and_zeroshot_eval(
                model, X_train28_aligned, y_train28, X_k_aligned, y_k, RANDOM_SEED)
            acc = float((preds == y_k).mean())
            acc_by_config[(d_model, d_state, n_layers)].append(acc)

            csv_rows.append({
                "script": "run_capacity_sweep",
                "condition": f"capacity_screen_mamba_d{d_model}_s{d_state}_l{n_layers}",
                "seed": RANDOM_SEED, "ea_mode": ea_mode, "cov_estimator": cov_estimator,
                "fold_index": fold_idx, "test_subject": str(test_sub),
                "temporal_dim": int(d_model), "n_temporal_params": int(n_params),
                "pre_calibration_acc": acc, "post_calibration_acc": acc,
            })
        log.info(f"  Fold elapsed: {time.time()-fold_start:.0f}s")

    # =========================================================================
    # Aggregate + rank
    # =========================================================================
    ranked = []
    for (d_model, d_state, n_layers), accs in acc_by_config.items():
        ranked.append({
            "d_model": d_model, "d_state": d_state, "n_layers": n_layers,
            "n_params": n_params_by_config[(d_model, d_state, n_layers)],
            "mean_accuracy": float(np.mean(accs)), "std_accuracy": float(np.std(accs)),
            "n_folds": len(accs),
        })
    ranked.sort(key=lambda r: r["mean_accuracy"], reverse=True)
    top_k = ranked[:TOP_K_CONFIGS]

    log.info(f"\n{'='*70}\n  SCREENING DONE — {len(configs)} configs x {len(fold_subjects)} folds (seed={RANDOM_SEED})\n{'='*70}")
    for r in ranked:
        log.info(f"    d_model={r['d_model']:>3} d_state={r['d_state']:>2} n_layers={r['n_layers']} "
                  f"| n_params={r['n_params']:>6,} | mean_acc={r['mean_accuracy']*100:.2f}% (+/- {r['std_accuracy']*100:.2f}pp)")
    log.info(f"\n  TOP-{TOP_K_CONFIGS}: {top_k}")
    log.info(f"  Reference (Condition 3 zero-shot baseline): {CONDITION3_BASELINE_ACC*100:.2f}%")

    if not pilot:
        os.makedirs(os.path.dirname(LOSO_CSV_PATH), exist_ok=True)
        write_header = not os.path.exists(LOSO_CSV_PATH)
        with open(LOSO_CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOSO_CSV_FIELDNAMES, restval="")
            if write_header:
                writer.writeheader()
            writer.writerows(csv_rows)

    payload = {
        "phase": "screening", "pilot": pilot, "ea_mode": ea_mode, "cov_estimator": cov_estimator,
        "seed": RANDOM_SEED, "n_folds": len(fold_subjects),
        "grid": {"d_model": D_MODEL_GRID, "d_state": D_STATE_GRID, "n_layers": N_LAYERS_GRID},
        "n_configs": len(configs), "top_k": TOP_K_CONFIGS,
        "ranked_configs": ranked, "top_k_configs": top_k,
        "reference_condition3_baseline_acc": CONDITION3_BASELINE_ACC,
    }
    if not pilot:
        with open(SCREENING_JSON, "w") as f:
            json.dump(payload, f, indent=2)
        volume.commit()
        log.info(f"  Saved: {SCREENING_JSON}  (read by ::full)")

    # =========================================================================
    # C3 plausibility assertions -- printed next to the numbers, not silent.
    # Run AFTER the write above so a failing assertion never suppresses the
    # diagnostic artifact.
    # =========================================================================
    for r in ranked:
        assert 0.0 <= r["mean_accuracy"] <= 1.0, (
            f"[C3 PLAUSIBILITY FAIL] config d_model={r['d_model']} d_state={r['d_state']} "
            f"n_layers={r['n_layers']} mean_accuracy={r['mean_accuracy']} outside [0,1]"
        )
        assert r["n_params"] > 0, (
            f"[C3 PLAUSIBILITY FAIL] config d_model={r['d_model']} d_state={r['d_state']} "
            f"n_layers={r['n_layers']} n_params={r['n_params']} (must be > 0)"
        )
    if not pilot:
        assert len(fold_subjects) == 29, f"[C3 PLAUSIBILITY FAIL] expected 29 folds, got {len(fold_subjects)}"
    log.info(f"  [C3] plausibility: all {len(ranked)} configs have accuracy in [0,1] and n_params > 0 -- OK")

    return {"top_k_configs": top_k, "n_configs_screened": len(configs), "n_folds": len(fold_subjects)}


@app.function(image=image, gpu="L4", volumes={VOLUME_PATH: volume}, timeout=86400, memory=16384)
def run_capacity_full(pilot: bool = True, pilot_n_folds: int = 5,
                       ea_mode: str = "pooled", cov_estimator: str = "fixed", top_k: int = TOP_K_CONFIGS):

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
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from sklearn.covariance import LedoitWolf
    from sklearn.metrics import (
        confusion_matrix, f1_score, roc_auc_score, cohen_kappa_score, balanced_accuracy_score,
    )
    import eeg_alignment as ea

    assert ea_mode in ("none", "pooled", "per-subject", "riemannian"), f"Unknown --ea-mode: {ea_mode!r}"
    assert cov_estimator in ("fixed", "lwf"), f"Unknown --cov-estimator: {cov_estimator!r}"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("f-capacity-full")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert os.path.exists(SCREENING_JSON), (
        f"{SCREENING_JSON} not found. Run `modal run run_capacity_sweep.py::screen` "
        f"BEFORE `::full` -- screening only writes this file when it runs non-pilot "
        f"(pilot=False, which is `::screen`'s own default, so a bare "
        f"`modal run run_capacity_sweep.py::screen` already satisfies this; only "
        f"`--pilot True` skips the write)."
    )
    with open(SCREENING_JSON) as f:
        screening = json.load(f)
    top_configs = screening["ranked_configs"][:top_k]
    assert len(top_configs) > 0, "Screening JSON has no ranked configs."
    best_cfg = top_configs[0]
    log.info(f"Loaded screening results: top-{top_k} configs = {top_configs}")

    raw = np.load(RAW_DATA_PATH, allow_pickle=True)
    X_np = raw["X"].astype(np.float32)
    y_np = raw["y"].astype(np.int64)
    subjects_np = raw["subjects"]
    N, C, T = X_np.shape
    N_CLASSES = int(y_np.max()) + 1
    assert C == N_CHANNELS and N_CLASSES == 2
    if not pilot:
        assert len(np.unique(subjects_np)) == 29, (
            f"Expected exactly 29 subjects (30 - sub-09 exclusion per AUDIT.md D2), "
            f"got {len(np.unique(subjects_np))}: {sorted(np.unique(subjects_np).tolist())}"
        )

    all_subjects = sorted(np.unique(subjects_np).tolist())
    fold_subjects = all_subjects[:pilot_n_folds] if pilot else all_subjects
    seeds = [RANDOM_SEED] if pilot else SEEDS
    log.info(f"Running folds: {fold_subjects} | seeds: {seeds}")

    # =========================================================================
    # F9: covariance estimator dispatch
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
        }

    def permutation_null_stats(y_true, y_pred, seed, n_perm=1000):
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
    # Parametrized Mamba temporal-only stack (identical to run_capacity_screening)
    # =========================================================================
    def parallel_scan(A, Bt):
        A = A.clone(); Bt = Bt.clone()
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
        def __init__(self, d_model, d_state, expand=EXPAND, dt_rank=DT_RANK,
                     conv_kernel=CONV_KERNEL, dropout=DROPOUT):
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
            self.A_log = nn.Parameter(A_log)
            self.D = nn.Parameter(torch.ones(d_inner))
            self.out_proj = nn.Linear(d_inner, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            B_, T_, _ = x.shape
            residual = x
            x = self.norm(x)
            xz = self.in_proj(x)
            x_in, z = xz.chunk(2, dim=-1)
            x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :T_]
            x_conv = F.silu(x_conv.transpose(1, 2))
            x_dbc = self.x_proj(x_conv)
            delta_raw, Bmat, Cmat = torch.split(x_dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            delta = F.softplus(self.dt_proj(delta_raw))
            A = -torch.exp(self.A_log)
            deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
            deltaBx = delta.unsqueeze(-1) * Bmat.unsqueeze(2) * x_conv.unsqueeze(-1)
            h = parallel_scan(deltaA, deltaBx)
            y = (h * Cmat.unsqueeze(2)).sum(-1) + self.D.unsqueeze(0).unsqueeze(0) * x_conv
            y = y * F.silu(z)
            out = self.dropout(self.out_proj(y))
            return residual + out

    class MambaTemporalBranch(nn.Module):
        def __init__(self, n_channels, d_model, d_state, n_layers, dropout=DROPOUT):
            super().__init__()
            self.in_proj = nn.Linear(n_channels, d_model)
            self.blocks = nn.ModuleList(
                [SelectiveSSMBlock(d_model=d_model, d_state=d_state, dropout=dropout) for _ in range(n_layers)])
            self.final_norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x_signal):
            x = x_signal.transpose(1, 2)
            x = self.in_proj(x)
            for blk in self.blocks:
                x = blk(x)
            x = self.final_norm(x)
            x = x.mean(dim=1)
            return self.dropout(x)

    class TemporalOnlyModel(nn.Module):
        def __init__(self, n_channels, d_model, d_state, n_layers, n_classes=2):
            super().__init__()
            self.temporal = MambaTemporalBranch(n_channels, d_model, d_state, n_layers)
            self.classifier = nn.Linear(d_model, n_classes)

        def forward_features(self, x_signal):
            return self.temporal(x_signal)

        def forward(self, x_signal):
            return self.classifier(self.forward_features(x_signal))

    # =========================================================================
    # Matched-parameter LSTM control (see header for design notes)
    # =========================================================================
    class LSTMTemporalBranch(nn.Module):
        def __init__(self, n_channels, hidden_size, num_layers, dropout=DROPOUT):
            super().__init__()
            self.lstm = nn.LSTM(input_size=n_channels, hidden_size=hidden_size,
                                 num_layers=num_layers, batch_first=True)
            self.final_norm = nn.LayerNorm(hidden_size)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x_signal):
            x = x_signal.transpose(1, 2)
            out, _ = self.lstm(x)
            out = self.final_norm(out)
            out = out.mean(dim=1)
            return self.dropout(out)

    class TemporalOnlyModelLSTM(nn.Module):
        def __init__(self, n_channels, hidden_size, num_layers, n_classes=2):
            super().__init__()
            self.temporal = LSTMTemporalBranch(n_channels, hidden_size, num_layers)
            self.classifier = nn.Linear(hidden_size, n_classes)

        def forward_features(self, x_signal):
            return self.temporal(x_signal)

        def forward(self, x_signal):
            return self.classifier(self.forward_features(x_signal))

    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def find_matched_lstm_hidden_size(target_params, num_layers, n_channels, n_classes=2, lo=2, hi=1024):
        """Binary search: LSTM param count is monotone increasing in hidden_size
        for fixed num_layers, so this converges to the closest achievable match."""
        def params_for(hidden):
            m = TemporalOnlyModelLSTM(n_channels, hidden, num_layers, n_classes)
            return count_params(m)
        best_hidden, best_diff = lo, abs(params_for(lo) - target_params)
        l, h = lo, hi
        while l <= h:
            mid = (l + h) // 2
            p_mid = params_for(mid)
            diff = abs(p_mid - target_params)
            if diff < best_diff:
                best_diff, best_hidden = diff, mid
            if p_mid < target_params:
                l = mid + 1
            else:
                h = mid - 1
        return best_hidden, params_for(best_hidden)

    target_params = int(best_cfg["n_params"])
    matched_hidden, matched_params = find_matched_lstm_hidden_size(
        target_params, num_layers=best_cfg["n_layers"], n_channels=N_CHANNELS)
    log.info(f"  Matched-parameter LSTM control: hidden_size={matched_hidden} num_layers={best_cfg['n_layers']} "
             f"-> {matched_params:,} params (target from best Mamba config: {target_params:,} params, "
             f"{abs(matched_params - target_params) / target_params * 100:.2f}% off)")

    def make_loader(X_arr, y, batch_size, shuffle, drop_last=False):
        ds = TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                           pin_memory=True, drop_last=drop_last)

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

    def pretrain_and_zeroshot_eval(model, X_train28_aligned, y_train28, X_k_aligned, y_k, seed):
        Xtr_sig, Xval_sig, ytr, yval = train_test_split(
            X_train28_aligned, y_train28, test_size=INTERNAL_VAL_FRAC, stratify=y_train28, random_state=seed)
        train_loader = make_loader(Xtr_sig, ytr, PRETRAIN_BATCH, True, drop_last=True)
        val_loader = make_loader(Xval_sig, yval, PRETRAIN_BATCH, False)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=PRETRAIN_WD)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS, eta_min=PRETRAIN_LR * 0.01)
        best_val_loss, best_state, es_ctr = math.inf, None, 0
        for epoch in range(1, PRETRAIN_EPOCHS + 1):
            train_one_epoch(model, train_loader, optimizer, criterion)
            scheduler.step()
            val_loss, val_acc = evaluate(model, val_loader, criterion)
            if val_loss < best_val_loss:
                best_val_loss, best_state, es_ctr = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                es_ctr += 1
                if es_ctr >= PRETRAIN_ES_PAT:
                    break
        model.load_state_dict(best_state)
        model.eval()
        k_loader = make_loader(X_k_aligned, y_k, PRETRAIN_BATCH, False)
        all_logits = []
        with torch.no_grad():
            for Xb, yb in k_loader:
                Xb = Xb.to(device)
                all_logits.append(model(Xb).cpu())
        logits = torch.cat(all_logits, dim=0)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = logits.argmax(dim=1).numpy().astype(np.int64)
        scores = probs[:, 1] - probs[:, 0]
        return preds, scores

    # =========================================================================
    # Full F4 5-seed x 29-fold loop over {top-K Mamba configs, LSTM control}
    # =========================================================================
    arms = [{"kind": "mamba", "d_model": c["d_model"], "d_state": c["d_state"], "n_layers": c["n_layers"]}
            for c in top_configs]
    arms.append({"kind": "lstm", "hidden_size": matched_hidden, "num_layers": best_cfg["n_layers"]})

    csv_rows = []
    acc_by_arm_seed = {i: {s: [] for s in seeds} for i in range(len(arms))}

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        for fold_idx, test_sub in enumerate(fold_subjects):
            fold_start = time.time()
            log.info(f"\n{'='*70}\n  SEED {seed} | FOLD {fold_idx+1}/{len(fold_subjects)} — sub-{test_sub}\n{'='*70}")

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
                X_train28_aligned = ea.apply_ea_whitening_signal(X_train28_z, W_train).astype(np.float32)
                X_k_aligned = ea.apply_ea_whitening_signal(X_k_z, W_train).astype(np.float32)
            else:
                X_train28_aligned = ea.apply_ea_whitening_signal_per_subject(
                    X_train28_z, subs_train28, W_train).astype(np.float32)
                subs_k = np.full(len(X_k_z), test_sub)
                W_k = ea.euclidean_align(covs_k, subs_k, mode=ea_mode)
                X_k_aligned = ea.apply_ea_whitening_signal_per_subject(X_k_z, subs_k, W_k).astype(np.float32)

            for arm_idx, arm in enumerate(arms):
                if arm["kind"] == "mamba":
                    model = TemporalOnlyModel(N_CHANNELS, arm["d_model"], arm["d_state"], arm["n_layers"]).to(device)
                    temporal_dim = arm["d_model"]
                    condition = f"capacity_full_mamba_d{arm['d_model']}_s{arm['d_state']}_l{arm['n_layers']}"
                else:
                    model = TemporalOnlyModelLSTM(N_CHANNELS, arm["hidden_size"], arm["num_layers"]).to(device)
                    temporal_dim = arm["hidden_size"]
                    condition = "capacity_full_lstm_matched_control"
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                preds, scores = pretrain_and_zeroshot_eval(
                    model, X_train28_aligned, y_train28, X_k_aligned, y_k, seed)
                acc = float((preds == y_k).mean())
                acc_by_arm_seed[arm_idx][seed].append(acc)
                metrics = compute_extended_metrics(y_k, preds, scores)
                perm_stats = permutation_null_stats(y_k, preds, seed=seed)

                csv_rows.append({
                    "script": "run_capacity_sweep", "condition": condition,
                    "seed": seed, "ea_mode": ea_mode, "cov_estimator": cov_estimator,
                    "fold_index": fold_idx, "test_subject": str(test_sub),
                    "temporal_dim": int(temporal_dim), "n_temporal_params": int(n_params),
                    "realized_cov_lambda_mean": realized_lambda_mean,
                    "pre_calibration_acc": acc, "post_calibration_acc": acc,
                    "sensitivity": metrics["sensitivity"], "specificity": metrics["specificity"],
                    "f1": metrics["f1"], "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"], "cohen_kappa": metrics["cohen_kappa"],
                    "roc_auc": metrics["roc_auc"],
                    "permutation_empirical_chance_level": perm_stats["permutation_empirical_chance_level"],
                    "permutation_null_std_acc": perm_stats["permutation_null_std_acc"],
                    "permutation_p_value": perm_stats["permutation_p_value"],
                })
                log.info(f"    [{condition}] acc={acc:.4f}")

            log.info(f"  Fold elapsed: {time.time()-fold_start:.0f}s")

    arm_summaries = []
    for arm_idx, arm in enumerate(arms):
        per_seed_mean = [float(np.mean(acc_by_arm_seed[arm_idx][s])) for s in seeds]
        arm_summaries.append({
            "arm": arm, "per_seed_mean_accuracy": per_seed_mean,
            "grand_mean_accuracy": float(np.mean(per_seed_mean)), "grand_std_accuracy": float(np.std(per_seed_mean)),
        })

    log.info(f"\n{'='*70}\n  F-CAPACITY FULL RESULTS — {len(arms)} arms x {len(seeds)} seeds x {len(fold_subjects)} folds\n{'='*70}")
    for s in arm_summaries:
        log.info(f"    {s['arm']} -> {s['grand_mean_accuracy']*100:.2f}% (+/- {s['grand_std_accuracy']*100:.2f}pp)")

    if not pilot:
        os.makedirs(os.path.dirname(LOSO_CSV_PATH), exist_ok=True)
        write_header = not os.path.exists(LOSO_CSV_PATH)
        with open(LOSO_CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOSO_CSV_FIELDNAMES, restval="")
            if write_header:
                writer.writeheader()
            writer.writerows(csv_rows)

    payload = {
        "phase": "full", "pilot": pilot, "ea_mode": ea_mode, "cov_estimator": cov_estimator,
        "seeds": seeds, "n_folds": len(fold_subjects), "top_k": top_k,
        "screening_source": SCREENING_JSON, "top_configs_used": top_configs,
        "matched_lstm_control": {
            "hidden_size": matched_hidden, "num_layers": best_cfg["n_layers"],
            "n_params": matched_params, "target_params_from_best_mamba": target_params,
        },
        "arm_summaries": arm_summaries,
        "reference_condition3_baseline_acc": CONDITION3_BASELINE_ACC,
    }
    if not pilot:
        with open(FULL_JSON, "w") as f:
            json.dump(payload, f, indent=2)
        volume.commit()
        log.info(f"  Saved: {FULL_JSON}")
        log.info(f"  Saved: {LOSO_CSV_PATH}")

    # =========================================================================
    # C3 plausibility assertions -- printed next to the numbers, not silent.
    # Run AFTER the writes above so a failing assertion never suppresses the
    # diagnostic artifacts.
    # =========================================================================
    for s in arm_summaries:
        assert 0.0 <= s["grand_mean_accuracy"] <= 1.0, (
            f"[C3 PLAUSIBILITY FAIL] arm {s['arm']} grand_mean_accuracy={s['grand_mean_accuracy']} outside [0,1]"
        )
        assert len(s["per_seed_mean_accuracy"]) == len(seeds), (
            f"[C3 PLAUSIBILITY FAIL] arm {s['arm']} has {len(s['per_seed_mean_accuracy'])} per-seed means, "
            f"expected {len(seeds)}"
        )
    assert matched_params > 0, f"[C3 PLAUSIBILITY FAIL] matched LSTM control n_params={matched_params} (must be > 0)"
    if not pilot:
        assert len(fold_subjects) == 29, f"[C3 PLAUSIBILITY FAIL] expected 29 folds, got {len(fold_subjects)}"
        assert len(seeds) == 5, f"[C3 PLAUSIBILITY FAIL] expected 5 seeds, got {len(seeds)}"
    log.info(f"  [C3] plausibility: {len(arm_summaries)} arms all in [0,1] with {len(seeds)} seed(s) each, "
             f"matched LSTM n_params={matched_params} > 0 -- OK")

    return {"arm_summaries": arm_summaries, "matched_lstm_hidden_size": matched_hidden,
            "matched_lstm_n_params": matched_params, "n_folds": len(fold_subjects), "n_seeds": len(seeds)}


@app.local_entrypoint(name="screen")
def screen_entrypoint(pilot: bool = False, pilot_n_folds: int = 5,
                       ea_mode: str = "pooled", cov_estimator: str = "fixed"):
    print("F-CAPACITY Phase 1 — SCREENING (24 configs, single seed)")
    print(f"pilot={pilot}  ea_mode={ea_mode}  cov_estimator={cov_estimator}\n")
    print("NOTE: defaults to pilot=False (full 29-fold screening) because a single-seed")
    print("      screening run is already the cheap phase -- pass --pilot True for a")
    print("      fast 5-fold smoke test before committing to the full screen.\n")
    results = run_capacity_screening.remote(
        pilot=pilot, pilot_n_folds=pilot_n_folds, ea_mode=ea_mode, cov_estimator=cov_estimator)
    print("\nSCREENING RESULTS:")
    for k, v in results.items():
        print(f"  {k:<24}: {v}")


@app.local_entrypoint(name="full")
def full_entrypoint(pilot: bool = False, pilot_n_folds: int = 5,
                     ea_mode: str = "pooled", cov_estimator: str = "fixed", top_k: int = TOP_K_CONFIGS):
    print("F-CAPACITY Phase 2 — FULL (top-K configs + matched LSTM control, 5-seed)")
    print(f"pilot={pilot}  ea_mode={ea_mode}  cov_estimator={cov_estimator}  top_k={top_k}")
    print("Requires `::screen` (non-pilot) to have already run and saved results_capacity_screening.json.\n")
    results = run_capacity_full.remote(
        pilot=pilot, pilot_n_folds=pilot_n_folds, ea_mode=ea_mode, cov_estimator=cov_estimator, top_k=top_k)
    print("\nFULL RESULTS:")
    for k, v in results.items():
        print(f"  {k:<24}: {v}")
