"""
Figure 4: D2 parity-split replication.
Data source: results_d2_parity_split_check.json (odd/even subgroup
real values, within-group nulls, 95% CIs, and the raw 30-shuffle
null values for each subgroup).

GATE D correction: the committed JSON stores the REAL 30 within-group
null shuffle values for each subgroup (`null_shuffle_values`) --
contrary to this script's earlier docstring, both panels below plot
the actual empirical null distributions, not a modeled N(mean, SD)
sample. The "gap above null mean" tolerance (+/-0.001, not the
blanket +/-0.0005) and the null-SD tolerance (+/-0.0001) were derived
and containment-verified in GATE D STEP 1a.
"""
import json
import matplotlib.pyplot as plt
import os

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"

SCRIPT_DIR = os.path.dirname(__file__)
OUTDIR = os.path.join(SCRIPT_DIR, "..")
REPO_ROOT = os.path.join(SCRIPT_DIR, "..", "..", "..")

with open(os.path.join(REPO_ROOT, "results_d2_parity_split_check.json"), encoding="utf-8") as f:
    _d2 = json.load(f)

_odd = _d2["groups"]["odd_search_first"]
_even = _d2["groups"]["even_memorize_first"]

GAP_TOL = 0.001    # GATE D STEP 1a: real-value's own quantum (8.5e-4/7.9e-4) exceeds +/-0.0005
SD_TOL = 0.0001    # GATE D STEP 1a: one-flip SD sensitivity (~5.6e-5/6.3e-5) exceeds bare-4dp safety

def _group_spec(g, label, color):
    real = g["real_pre_cal_balanced"]
    mean = g["null_mean"]
    return dict(
        real=real, mean=mean, sd=g["null_sd"], ci=tuple(g["null_percentile_95ci"]),
        gap=real - mean, n=len(g["null_shuffle_values"]),
        samples=g["null_shuffle_values"], label=label, color=color,
    )

ODD = _group_spec(_odd, f"Odd subjects (Search-first, n={_odd['n_subjects']})", "#1c4d8f")
EVEN = _group_spec(_even, f"Even subjects (Memorize-first, n={_even['n_subjects']})", "#1a7a34")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=False)

for ax, spec, hatch in zip(axes, (ODD, EVEN), ("//", "\\\\")):
    samples = spec["samples"]
    ax.hist(samples, bins=10, density=True, color=spec["color"], alpha=0.30,
            edgecolor=spec["color"], hatch=hatch, linewidth=1.0,
            label=f"Within-group null\n(empirical, n={spec['n']})")

    # shade the 95% CI region
    ax.axvspan(spec["ci"][0], spec["ci"][1], color=spec["color"], alpha=0.12, zorder=1)
    ax.axvline(spec["ci"][0], color=spec["color"], linestyle=":", linewidth=1.2)
    ax.axvline(spec["ci"][1], color=spec["color"], linestyle=":", linewidth=1.2)

    ax.axvline(spec["real"], color="#b02418", linestyle="--", linewidth=2.0, zorder=5)

    ymax = ax.get_ylim()[1]
    ax.text(spec["real"] + 0.004, ymax * 0.55, f"real\n({spec['real']:.4f})",
            color="#b02418", fontsize=9, fontweight="bold", ha="left", va="center")
    ax.text(0.02, 0.97,
            f"null mean={spec['mean']:.4f}\nnull SD={spec['sd']:.4f} $\\pm$ {SD_TOL}\n"
            f"95% CI=[{spec['ci'][0]:.4f}, {spec['ci'][1]:.4f}]",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.6, color=spec["color"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=spec["color"], alpha=0.85))

    ax.annotate(f"gap above null: +{spec['gap']:.4f} $\\pm$ {GAP_TOL}",
                xy=(spec["real"], ymax * 0.15), xytext=(0.5, 0.20), textcoords="axes fraction",
                fontsize=9.2, ha="center", color="#333333",
                arrowprops=dict(arrowstyle="-", color="#888888", linewidth=0.8))

    ax.set_title(spec["label"], fontsize=10.5, pad=8)
    ax.set_xlabel("Balanced accuracy (pre-calibration)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8.2, frameon=True)

axes[0].set_ylabel("Density", fontsize=10)
axes[0].set_xlim(0.44, 0.66)  # shared scale so the gap-size difference is visible
axes[1].set_xlim(0.44, 0.66)

fig.text(0.5, 0.985,
          "Empirical null densities: real 30-shuffle within-group values from the committed run JSON",
          ha="center", va="top", fontsize=8.6, style="italic", color="#555555")

fig.text(0.5, 0.015,
          f"Gap asymmetry: odd = {ODD['gap']:.4f} vs. even = {EVEN['gap']:.4f} "
          f"(~{ODD['gap']/EVEN['gap']:.1f}x); unresolved",
          ha="center", va="bottom", fontsize=9.5, color="#333333")

plt.tight_layout(rect=(0, 0.06, 1, 0.94))
for ext in ("pdf", "png"):
    plt.savefig(os.path.join(OUTDIR, f"fig4_d2_parity_split.{ext}"),
                dpi=300 if ext == "png" else None, bbox_inches="tight")
plt.close(fig)
print("Saved fig4_d2_parity_split.pdf / .png")
print(f"ODD real={ODD['real']}, mean={ODD['mean']}, sd={ODD['sd']}, gap={ODD['gap']}, ci={ODD['ci']}")
print(f"EVEN real={EVEN['real']}, mean={EVEN['mean']}, sd={EVEN['sd']}, gap={EVEN['gap']}, ci={EVEN['ci']}")
