"""
scripts/benchmark_efficiency.py

F15 -- efficiency benchmark: trainable-parameter counts, MACs (multiply-
accumulate operations), batch-1 latency (CPU, and GPU if available), and
peak memory for every model variant actually used in this codebase's
production drivers.

WHY THIS SCRIPT EXISTS:
    AUDIT.md's Fix-ID table (F15) asks for params/MACs/latency/memory per
    model variant so the manuscript can report a fair efficiency comparison
    alongside accuracy. No such numbers existed anywhere in the codebase
    before this script.

WHY THIS IS A PLAIN LOCAL SCRIPT, NOT A MODAL SCRIPT:
    Unlike every LOSO driver, this benchmark needs no real EEG data and no
    GPU cluster -- only correctly-SHAPED synthetic input tensors (the model
    architectures are fixed by their hyperparameters, independent of what
    data flows through them). Per AUDIT.md's own cost note ("Low --
    Minutes"), it is designed to actually run in a plain local Python
    environment (CPU always; GPU too, if `torch.cuda.is_available()`).

MODELS BENCHMARKED (verbatim class definitions duplicated from their
source drivers, at each driver's ACTUAL production hyperparameters --
not the F-CAPACITY sweep grid, since that sweep has not selected a winner
yet; see the "KNOWN FOLLOW-UP" note at the bottom):
  - "mamba_temporal_branch": TemporalOnlyModel, D_MODEL=16/D_STATE=8/N_LAYERS=1,
    from run_step4_condition4_asymmetric_mamba.py (the actual deployed
    temporal-only pretraining architecture).
  - "eegnet": EEGNet, F1=8/D=2/F2=16/kern_length=125, from
    run_step4_eegnet_ea.py (identical architecture to
    run_step4_condition1_eegnet_calibrated.py).
  - "joint_dualbranch_fusion": JointDualBranchModel (spatial_dim=1953,
    d_model=16), from run_step4_condition4_joint_dualbranch_mamba.py --
    F-JOINT's single end-to-end nn.Module holding both branches + the
    fusion classifier head; this is the flagship "our model" architecture.
  - Classical baselines (MDM, CSP+LDA, TS-SVM, TS-LDA -- from
    run_baselines_mdm_tssvm_tslda_csp.py): these are not nn.Modules, so
    "params"/"MACs" don't apply in the same sense; only prediction latency
    at batch=1 is measured for these, using fitted sklearn estimators on
    synthetic separable data.

MAC COUNTING METHODOLOGY:
    Forward hooks register on every nn.Linear / nn.Conv1d / nn.Conv2d
    submodule and compute MACs analytically from the layer's weight shape
    and the actual output tensor shape seen at runtime (standard
    weight-reuse accounting, no new profiling dependency such as
    `thop`/`fvcore`/`ptflops` -- consistent with this codebase's established
    no-new-fragile-dependency convention, see F5's docstring). The Mamba
    branch's custom parallel-scan recurrence (SelectiveSSMBlock.forward in
    the source drivers) is NOT a standard hookable layer, so its cost is
    added analytically per block: the elementwise combine work of the
    log-depth Blelloch-style doubling scan is
    n_layers * ceil(log2(T)) * T * d_inner * d_state * 2 MACs (two
    elementwise multiply-accumulate passes -- for A and for Bt -- per
    doubling step), plus the final h*Cmat reduction (T*d_inner*d_state) and
    D*x_conv skip term (T*d_inner). This is flagged in the output as
    "scan_macs_analytical" (estimated) separately from
    "layer_macs_hooked" (exact, from hooks), so a reader can see exactly
    which portion of the total is exact vs. estimated.

MEMORY METHODOLOGY:
    CPU memory uses `torch.profiler` with `profile_memory=True`, summing
    each op's self CPU memory usage -- an upper-bound PROXY for peak
    activation memory (reported as "cpu_memory_proxy_mb"), not a true
    allocator-level peak. Python's stdlib `tracemalloc` was tried first but
    only tracks CPython-heap allocations, not PyTorch's C++-level tensor
    allocator, so it silently reported near-zero for every real forward
    pass; the POSIX-only `resource.getrusage` used elsewhere in this
    codebase is also not usable here since this script must run on the
    Windows sandbox this codebase is developed in. GPU peak memory (exact,
    reported as "peak_memory_mb") uses `torch.cuda.max_memory_allocated()`
    (reset before each model's
    benchmark), only measured if CUDA is available.

Usage: python scripts/benchmark_efficiency.py
       python scripts/benchmark_efficiency.py --n-times 251 --n-warmup 10 --n-iters 100 --out results_efficiency_benchmark.json
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CHANNELS = 62
SPATIAL_DIM = 1953   # 62*63/2 upper-triangle tangent-vector dim, matches run_step4_condition4_asymmetric_mamba.py

# Production hyperparameters, verbatim from their respective source drivers.
MAMBA_D_MODEL, MAMBA_D_STATE, MAMBA_N_LAYERS = 16, 8, 1
MAMBA_EXPAND, MAMBA_DT_RANK, MAMBA_CONV_KERNEL, MAMBA_DROPOUT = 2, 4, 4, 0.3

EEGNET_F1, EEGNET_D, EEGNET_F2 = 8, 2, 16
EEGNET_KERN_LENGTH = 125
EEGNET_DROPOUT = 0.5


# =============================================================================
# Model classes -- duplicated verbatim from their source drivers (see
# docstring above for exact provenance of each).
# =============================================================================
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
    def __init__(self, d_model, d_state, expand=MAMBA_EXPAND, dt_rank=MAMBA_DT_RANK,
                 conv_kernel=MAMBA_CONV_KERNEL, dropout=MAMBA_DROPOUT):
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
    def __init__(self, n_channels, d_model, d_state=MAMBA_D_STATE, n_layers=MAMBA_N_LAYERS, dropout=MAMBA_DROPOUT):
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
    """From run_step4_condition4_asymmetric_mamba.py -- the temporal-only
    pretraining architecture (zero-shot-evaluated before being frozen and
    fused with the spatial branch)."""
    def __init__(self, n_channels, d_model, d_state=MAMBA_D_STATE, n_layers=MAMBA_N_LAYERS, n_classes=2):
        super().__init__()
        self.temporal = MambaTemporalBranch(n_channels, d_model, d_state, n_layers)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward_features(self, x_signal):
        return self.temporal(x_signal)

    def forward(self, x_signal):
        return self.classifier(self.forward_features(x_signal))


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
    """From run_step4_eegnet_ea.py, identical to
    run_step4_condition1_eegnet_calibrated.py."""
    def __init__(self, n_channels, n_times, n_classes=2,
                 F1=EEGNET_F1, D=EEGNET_D, F2=EEGNET_F2,
                 kern_length=EEGNET_KERN_LENGTH, dropout=EEGNET_DROPOUT):
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


class JointDualBranchModel(nn.Module):
    """From run_step4_condition4_joint_dualbranch_mamba.py (F-JOINT) -- the
    single end-to-end nn.Module holding both branches + fusion head."""
    def __init__(self, spatial_dim, n_channels=N_CHANNELS, d_model=MAMBA_D_MODEL, n_classes=2):
        super().__init__()
        self.temporal = MambaTemporalBranch(n_channels, d_model)
        self.fusion_classifier = nn.Linear(spatial_dim + d_model, n_classes)

    def forward_temporal_features(self, x_signal):
        return self.temporal(x_signal)

    def forward(self, spatial_tan, x_signal):
        temporal_emb = self.forward_temporal_features(x_signal)
        fused = torch.cat([spatial_tan, temporal_emb], dim=-1)
        return self.fusion_classifier(fused)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# MAC counting: forward hooks on standard layers (exact) + analytical
# add-on for the Mamba selective-scan recurrence (estimated, see docstring).
# =============================================================================
def count_hooked_macs(model, forward_fn):
    macs = {"total": 0}
    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            if isinstance(module, nn.Linear):
                out_features = output.shape[-1]
                n_positions = output.numel() // out_features
                macs["total"] += n_positions * module.in_features * out_features
            elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                out_elems_per_channel = output[0, 0].numel()   # spatial positions per output channel
                kernel_numel = int(np.prod(module.kernel_size))
                in_ch_per_group = module.in_channels // module.groups
                macs["total"] += (output.shape[0] * module.out_channels * out_elems_per_channel
                                   * kernel_numel * in_ch_per_group)
        return hook

    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            handles.append(m.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        forward_fn()

    for h in handles:
        h.remove()
    return macs["total"]


def mamba_scan_macs_analytical(n_layers, d_inner, d_state, T):
    """See module docstring's 'MAC COUNTING METHODOLOGY' for derivation."""
    n_doubling_steps = int(np.ceil(np.log2(max(T, 2))))
    combine_macs = n_doubling_steps * T * d_inner * d_state * 2
    reduction_macs = T * d_inner * d_state
    skip_macs = T * d_inner
    return n_layers * (combine_macs + reduction_macs + skip_macs)


# =============================================================================
# Latency + memory benchmarking
# =============================================================================
def benchmark_latency(forward_fn, device, n_warmup=10, n_iters=50):
    for _ in range(n_warmup):
        with torch.no_grad():
            forward_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        with torch.no_grad():
            forward_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    return {
        "mean_ms": float(times.mean() * 1000), "std_ms": float(times.std() * 1000),
        "median_ms": float(np.median(times) * 1000),
        "min_ms": float(times.min() * 1000), "max_ms": float(times.max() * 1000),
        "n_iters": n_iters,
    }


def benchmark_cpu_peak_memory_mb(forward_fn):
    """`tracemalloc` only tracks CPython-heap allocations, not PyTorch's
    C++-level tensor allocator -- it reports near-zero for any real forward
    pass and would be actively misleading here. Instead this uses
    `torch.profiler` (already a torch dependency, no new package) with
    `profile_memory=True` and sums each op's self CPU memory usage. This is
    an upper-bound PROXY for peak activation memory (summing rather than
    true allocator-level peak tracking, since torch.profiler's op-level
    summary doesn't expose a single peak-RSS number) -- documented as such
    in the output field name."""
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        with torch.no_grad():
            forward_fn()
    total_bytes = sum(max(evt.self_cpu_memory_usage, 0) for evt in prof.key_averages())
    return total_bytes / (1024 ** 2)


def benchmark_gpu_peak_memory_mb(forward_fn):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    with torch.no_grad():
        forward_fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def benchmark_nn_model(name, model_forward_builder, device, n_warmup, n_iters, extra_macs=0, results_so_far=None):
    """`model_forward_builder(device)` must build ONE model instance, move it
    to `device`, and return `(model, forward_fn)` where `forward_fn` calls
    THAT SAME model instance -- hooks are registered on `model` and only
    fire if `forward_fn` actually invokes it. `results_so_far` (optional):
    the caller's accumulated results list, used only to checkpoint already-
    completed models to disk if THIS model's plausibility check fails."""
    model, forward_fn = model_forward_builder(device)

    params = count_params(model)
    layer_macs = count_hooked_macs(model, forward_fn)
    total_macs = layer_macs + extra_macs

    latency = benchmark_latency(forward_fn, device, n_warmup=n_warmup, n_iters=n_iters)
    if device.type == "cuda":
        mem_mb, mem_field = benchmark_gpu_peak_memory_mb(forward_fn), "peak_memory_mb"
    else:
        mem_mb, mem_field = benchmark_cpu_peak_memory_mb(forward_fn), "cpu_memory_proxy_mb"

    result = {
        "model": name, "device": device.type,
        "n_params": int(params),
        "layer_macs_hooked": int(layer_macs),
        "scan_macs_analytical": int(extra_macs),
        "total_macs": int(total_macs),
        "latency_batch1": latency,
        mem_field: float(mem_mb),
    }

    # C3 plausibility assertions -- printed next to the numbers, not silent.
    # This is the systematic version of the check that originally caught the
    # MAC-hook wiring bug (EEGNet reporting exactly 0 MACs) by manual
    # inspection -- see STATUS.md's F15 row for that history. `results` (the
    # full run's accumulator, passed in by main()) is written to disk BEFORE
    # any assertion below can raise, so a failing run still leaves every
    # already-completed model's numbers on disk, not just this one's.
    try:
        assert params > 0, f"[C3 PLAUSIBILITY FAIL] {name}: n_params={params} (must be > 0)"
        assert total_macs > 0, (
            f"[C3 PLAUSIBILITY FAIL] {name}: total_macs={total_macs} (layer_macs_hooked={layer_macs}, "
            f"scan_macs_analytical={extra_macs}) -- zero MACs on a model with parameters signals a hook-"
            f"wiring bug (forward_fn calling a different instance than the hooked model), not a real result."
        )
        assert latency["mean_ms"] > 0.0, (
            f"[C3 PLAUSIBILITY FAIL] {name}: latency mean_ms={latency['mean_ms']} (must be > 0)"
        )
    except AssertionError:
        if results_so_far is not None:
            diagnostic_path = "results_efficiency_benchmark.c3_failure_diagnostic.json"
            with open(diagnostic_path, "w") as f:
                json.dump({"completed_results_before_failure": results_so_far, "failing_result": result},
                          f, indent=2)
            print(f"  [C3 HALT] diagnostic written before raising: {diagnostic_path}")
        raise
    print(f"  [C3] {name}: params={params} > 0, total_macs={total_macs} > 0, "
          f"latency={latency['mean_ms']:.3f}ms > 0 -- plausibility OK")

    return result


def benchmark_classical_baseline(name, fit_fn, predict_fn, n_warmup=10, n_iters=50):
    """Latency-only, no MACs/params -- see module docstring."""
    estimator = fit_fn()
    for _ in range(n_warmup):
        predict_fn(estimator)
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        predict_fn(estimator)
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    return {
        "model": name, "device": "cpu", "n_params": None, "total_macs": None,
        "latency_batch1": {
            "mean_ms": float(times.mean() * 1000), "std_ms": float(times.std() * 1000),
            "median_ms": float(np.median(times) * 1000),
            "min_ms": float(times.min() * 1000), "max_ms": float(times.max() * 1000),
            "n_iters": n_iters,
        },
        "peak_memory_mb": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-times", type=int, default=251,
                         help="Samples per trial -- (TMAX-TMIN)*RESAMPLE_HZ+1 = (0.8-(-0.2))*250+1 = 251 "
                              "per run_data_engine_on_modal.py's TMIN/TMAX/RESAMPLE_HZ (MNE inclusive endpoint "
                              "convention). Override with the real checkpoint's exact T if it differs.")
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-iters", type=int, default=50)
    parser.add_argument("--out", default="results_efficiency_benchmark.json")
    args = parser.parse_args()

    T = args.n_times
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    else:
        print("NOTE: CUDA not available in this environment -- GPU latency/memory numbers will be "
              "absent from the report. Re-run this script on a CUDA machine (e.g. via `modal run`, "
              "or any GPU box) to fill them in; only CPU numbers are produced here.")

    results = []

    scan_macs = mamba_scan_macs_analytical(
        MAMBA_N_LAYERS, MAMBA_EXPAND * MAMBA_D_MODEL, MAMBA_D_STATE, T)

    for device in devices:
        # --- mamba_temporal_branch ---
        def build_mamba(dev):
            m = TemporalOnlyModel(N_CHANNELS, MAMBA_D_MODEL, MAMBA_D_STATE, MAMBA_N_LAYERS).to(dev).eval()
            x = torch.randn(1, N_CHANNELS, T, device=dev)
            return m, (lambda: m(x))
        results.append(benchmark_nn_model(
            "mamba_temporal_branch", build_mamba, device,
            args.n_warmup, args.n_iters, extra_macs=scan_macs, results_so_far=results))

        # --- eegnet ---
        def build_eegnet(dev):
            m = EEGNet(N_CHANNELS, T).to(dev).eval()
            x = torch.randn(1, 1, N_CHANNELS, T, device=dev)
            return m, (lambda: m(x))
        results.append(benchmark_nn_model("eegnet", build_eegnet, device, args.n_warmup, args.n_iters,
                                           results_so_far=results))

        # --- joint_dualbranch_fusion ---
        def build_joint(dev):
            m = JointDualBranchModel(SPATIAL_DIM, N_CHANNELS, MAMBA_D_MODEL).to(dev).eval()
            spatial = torch.randn(1, SPATIAL_DIM, device=dev)
            signal = torch.randn(1, N_CHANNELS, T, device=dev)
            return m, (lambda: m(spatial, signal))
        results.append(benchmark_nn_model(
            "joint_dualbranch_fusion", build_joint, device,
            args.n_warmup, args.n_iters, extra_macs=scan_macs, results_so_far=results))

    # --- classical baselines: latency-only, CPU, fit once on synthetic
    # separable covariance/tangent data (mirrors run_baselines_mdm_tssvm_tslda_csp.py) ---
    from sklearn.svm import SVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    rng = np.random.RandomState(0)
    n_train, tangent_dim = 200, SPATIAL_DIM
    Xtr = rng.normal(size=(n_train, tangent_dim)).astype(np.float32)
    ytr = rng.randint(0, 2, size=n_train)
    x1 = rng.normal(size=(1, tangent_dim)).astype(np.float32)

    def fit_svm():
        return SVC(kernel="linear", C=1.0).fit(Xtr, ytr)
    results.append(benchmark_classical_baseline(
        "ts_svm", fit_svm, lambda est: est.predict(x1), args.n_warmup, args.n_iters))

    def fit_lda():
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, ytr)
    results.append(benchmark_classical_baseline(
        "ts_lda_shrinkage", fit_lda, lambda est: est.predict(x1), args.n_warmup, args.n_iters))

    # MDM (AIRM-distance nearest-centroid) and CSP+LDA are hand-rolled (not
    # sklearn estimators) in run_baselines_mdm_tssvm_tslda_csp.py; a
    # from-scratch prediction-path reimplementation here would duplicate
    # nontrivial SPD-manifold code purely for a latency number. Flagged as a
    # documented gap rather than silently omitted -- see KNOWN FOLLOW-UP.
    results.append({
        "model": "mdm", "device": "cpu", "n_params": None, "total_macs": None,
        "latency_batch1": None, "peak_memory_mb": None,
        "note": "NOT MEASURED -- hand-rolled AIRM-distance estimator in "
                "run_baselines_mdm_tssvm_tslda_csp.py, not reimplemented here. See KNOWN FOLLOW-UP.",
    })
    results.append({
        "model": "csp_lda", "device": "cpu", "n_params": None, "total_macs": None,
        "latency_batch1": None, "peak_memory_mb": None,
        "note": "NOT MEASURED -- hand-rolled CSP filters in "
                "run_baselines_mdm_tssvm_tslda_csp.py, not reimplemented here. See KNOWN FOLLOW-UP.",
    })

    payload = {
        "description": "F15 -- params/MACs/latency/peak-memory efficiency benchmark for every "
                        "model variant used in this codebase's production drivers.",
        "n_times": T, "n_channels": N_CHANNELS, "spatial_dim": SPATIAL_DIM,
        "n_warmup": args.n_warmup, "n_iters": args.n_iters,
        "cuda_available": torch.cuda.is_available(),
        "results": results,
        "known_follow_up": [
            "LSTM matched-capacity control (run_capacity_sweep.py's TemporalOnlyModelLSTM) is NOT "
            "included -- its hidden_size is only determined by find_matched_lstm_hidden_size after "
            "F-CAPACITY's ::full entrypoint actually runs on Modal (not yet run). Re-run this script "
            "with that hidden_size once available, rather than benchmarking a placeholder size now.",
            "MDM and CSP+LDA (hand-rolled SPD-manifold estimators in "
            "run_baselines_mdm_tssvm_tslda_csp.py) have no latency numbers here -- would require "
            "duplicating their prediction-path code; flagged, not fabricated.",
            "GPU numbers are absent unless this script is re-run on a CUDA-capable machine.",
        ],
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {args.out}")
    for r in results:
        lat = r.get("latency_batch1")
        lat_str = f"{lat['mean_ms']:.3f}ms" if lat else "N/A"
        mem = r.get("peak_memory_mb", r.get("cpu_memory_proxy_mb"))
        mem_str = f"{mem:.3f}MB" if mem is not None else "N/A"
        print(f"  {r['model']:<28} device={r['device']:<4} params={str(r.get('n_params')):>10} "
              f"MACs={str(r.get('total_macs')):>12} latency={lat_str:>12} mem={mem_str:>10}")


if __name__ == "__main__":
    main()
