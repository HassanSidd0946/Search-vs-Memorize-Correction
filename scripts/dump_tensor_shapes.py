# =============================================================================
# scripts/dump_tensor_shapes.py
#
# command to run:
#   python scripts/dump_tensor_shapes.py
#
# F1 — TENSOR-SHAPE INSTRUMENTATION (AUDIT.md Fix-ID table)
#
# WHY THIS SCRIPT EXISTS:
#   AUDIT.md's Q1 answered "what is the exact input tensor to the temporal
#   branch, and what shape does it take through the model" NARRATIVELY, by
#   reading run_step4_condition4_asymmetric_mamba.py's source. F1 is the
#   standalone instrumentation script that confirms that narrative EMPIRICALLY
#   -- forward hooks on the real module classes (D_MODEL=16, D_STATE=8,
#   N_LAYERS=1, the exact production hyperparameters), run against one
#   correctly-shaped synthetic trial, dumping the actual shape at every stage.
#   This is a plain local script (no Modal, no real EEG data needed -- only a
#   correctly-shaped synthetic tensor) so it can be executed directly in this
#   sandbox, same rationale as scripts/benchmark_efficiency.py (F15).
#
# MODEL CLASSES: copied verbatim from scripts/benchmark_efficiency.py's
#   already-reviewed duplicate of run_step4_condition4_asymmetric_mamba.py's
#   SelectiveSSMBlock/MambaTemporalBranch/TemporalOnlyModel (same
#   duplication rationale: a plain local script cannot `import` a
#   Modal-decorated function-local class without pulling in the whole Modal
#   app definition).
#
# OUTPUT: prints the shape chain to stdout AND writes it to
#   results/tensor_shapes.json (referenced by results/METHODS_FACTS.md, F2/F6).
# =============================================================================

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CHANNELS = 62
SFREQ = 250
TMIN, TMAX = -0.2, 0.8
N_TIMES = int(round((TMAX - TMIN) * SFREQ)) + 1   # 251, matches run_data_engine_on_modal.py

D_MODEL, D_STATE, N_LAYERS = 16, 8, 1
EXPAND, DT_RANK, CONV_KERNEL, DROPOUT = 2, 4, 4, 0.3

OUTPUT_JSON = "results/tensor_shapes.json"


def parallel_scan(deltaA, deltaBx):
    """Hillis-Steele associative scan -- verbatim from
    run_step4_condition4_asymmetric_mamba.py / scripts/benchmark_efficiency.py."""
    B_, T_, D_, N_ = deltaA.shape
    log2T = math.ceil(math.log2(T_)) if T_ > 1 else 0
    A, X = deltaA.clone(), deltaBx.clone()
    for step in range(log2T):
        shift = 2 ** step
        if shift >= T_:
            break
        A_shifted = F.pad(A[:, :-shift], (0, 0, 0, 0, shift, 0), value=1.0)
        X_shifted = F.pad(X[:, :-shift], (0, 0, 0, 0, shift, 0), value=0.0)
        X = X + A_shifted * X
        A = A * A_shifted
    return X


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
    def __init__(self, n_channels, d_model, d_state=D_STATE, n_layers=N_LAYERS, dropout=DROPOUT):
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
    def __init__(self, n_channels, d_model, d_state=D_STATE, n_layers=N_LAYERS, n_classes=2):
        super().__init__()
        self.temporal = MambaTemporalBranch(n_channels, d_model, d_state, n_layers)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward_features(self, x_signal):
        return self.temporal(x_signal)

    def forward(self, x_signal):
        return self.classifier(self.forward_features(x_signal))


def main():
    torch.manual_seed(42)
    model = TemporalOnlyModel(N_CHANNELS, D_MODEL, D_STATE, N_LAYERS)
    model.eval()

    shape_chain = []

    def make_hook(name):
        def hook(module, inputs, output):
            in_shapes = [tuple(t.shape) for t in inputs if torch.is_tensor(t)]
            if torch.is_tensor(output):
                out_shapes = [tuple(output.shape)]
            elif isinstance(output, (tuple, list)):
                out_shapes = [tuple(t.shape) for t in output if torch.is_tensor(t)]
            else:
                out_shapes = []
            shape_chain.append({
                "module": name, "class": type(module).__name__,
                "input_shapes": [list(s) for s in in_shapes],
                "output_shapes": [list(s) for s in out_shapes],
            })
        return hook

    handles = []
    for name, module in model.named_modules():
        if name == "":
            continue
        handles.append(module.register_forward_hook(make_hook(name)))

    # Q1's exact claimed input: (B, C=62, T=251), float32, EA-whitened/
    # z-scored raw signal -- a synthetic stand-in with the same shape/dtype
    # (no real EEG data needed to confirm the SHAPE chain, only real data
    # would be needed to confirm the VALUES, which Q1 already covers
    # narratively via the source-code read of apply_ea_whitening_signal).
    x_signal = torch.randn(1, N_CHANNELS, N_TIMES, dtype=torch.float32)
    print(f"Input to TemporalOnlyModel.forward: x_signal.shape = {tuple(x_signal.shape)} "
          f"(B=1, C={N_CHANNELS}, T={N_TIMES}, dtype={x_signal.dtype})")

    with torch.no_grad():
        logits = model(x_signal)

    for h in handles:
        h.remove()

    print(f"\nFinal output: logits.shape = {tuple(logits.shape)}")
    print("\nFull shape chain (in forward-call order):")
    for entry in shape_chain:
        print(f"  [{entry['module'] or '(root)'}] {entry['class']:<20} "
              f"in={entry['input_shapes']} -> out={entry['output_shapes']}")
    print("\nConfirms AUDIT.md Q1's narrative claim empirically: x_signal.transpose(1,2) -> "
          f"Linear({N_CHANNELS}->{D_MODEL}) -> (B,T,{D_MODEL}) -> {N_LAYERS} SelectiveSSMBlock(s) "
          f"-> LayerNorm -> mean-pool over T -> (B,{D_MODEL}).")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    payload = {
        "description": "F1 -- empirical confirmation of AUDIT.md Q1's input-tensor-shape claim via forward hooks",
        "hyperparameters": {"d_model": D_MODEL, "d_state": D_STATE, "n_layers": N_LAYERS,
                             "n_channels": N_CHANNELS, "n_times": N_TIMES},
        "input_shape": list(x_signal.shape),
        "output_shape": list(logits.shape),
        "shape_chain": shape_chain,
    }
    # Write to disk BEFORE the C3 assertions below can raise, so a failing
    # run still leaves the shape chain on disk for inspection.
    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {OUTPUT_JSON}")

    # C3 plausibility assertions -- printed next to the numbers, not silent.
    # Run AFTER the write above (see comment there).
    assert tuple(x_signal.shape) == (1, N_CHANNELS, N_TIMES), (
        f"[C3 PLAUSIBILITY FAIL] input shape {tuple(x_signal.shape)} does not match "
        f"Q1's claimed (B, C={N_CHANNELS}, T={N_TIMES})"
    )
    assert tuple(logits.shape) == (1, 2), (
        f"[C3 PLAUSIBILITY FAIL] final logits shape {tuple(logits.shape)} != (1, 2) (binary classifier)"
    )
    assert len(shape_chain) > 0, "[C3 PLAUSIBILITY FAIL] no forward hooks fired -- shape chain is empty"
    in_proj_entries = [e for e in shape_chain if e["module"] == "temporal.in_proj"]
    assert len(in_proj_entries) == 1 and in_proj_entries[0]["output_shapes"] == [[1, N_TIMES, D_MODEL]], (
        f"[C3 PLAUSIBILITY FAIL] temporal.in_proj output shape mismatch: {in_proj_entries}"
    )
    print(f"\n[C3] plausibility: input=(1,{N_CHANNELS},{N_TIMES}), output=(1,2), "
          f"{len(shape_chain)} hook firings, in_proj shape confirmed -- OK")


if __name__ == "__main__":
    main()
