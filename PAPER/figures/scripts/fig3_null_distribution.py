"""
Figure 3: R2 vs. null distributions (C1 30-shuffle, C4 500-shuffle).
Data sources: results_c1_r2_shuffled_label_control.json and
results_c4_high_res_shuffled_label_control.json.

GATE D correction: both committed JSONs store the REAL per-shuffle
values (`shuffle_level_results[].pre_cal_balanced_accuracy_mean` for
C1, `shuffle_level_pre_cal_balanced` for C4) -- contrary to this
script's earlier docstring, per-shuffle values are NOT missing from
the record. Both panels below plot the actual empirical null
distributions, not a modeled N(mean, SD) sample.
"""
import json
import matplotlib.pyplot as plt
import os

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"

SCRIPT_DIR = os.path.dirname(__file__)
OUTDIR = os.path.join(SCRIPT_DIR, "..")
REPO_ROOT = os.path.join(SCRIPT_DIR, "..", "..", "..")

with open(os.path.join(REPO_ROOT, "results_c1_r2_shuffled_label_control.json"), encoding="utf-8") as f:
    _c1 = json.load(f)
with open(os.path.join(REPO_ROOT, "results_c4_high_res_shuffled_label_control.json"), encoding="utf-8") as f:
    _c4 = json.load(f)

R2_REAL = _c1["real_R2_pre_cal_balanced"]
assert R2_REAL == _c4["real_R2_pre_cal_balanced"]  # both files must agree on the value being tested

c1_samples = [r["pre_cal_balanced_accuracy_mean"] for r in _c1["shuffle_level_results"]]
c4_samples = _c4["shuffle_level_pre_cal_balanced"]

C1_MEAN, C1_SD, C1_N = _c1["null_distribution_mean"], _c1["null_distribution_sd"], _c1["n_shuffles"]
C4_MEAN, C4_SD, C4_N = _c4["null_distribution_mean"], _c4["null_distribution_sd"], _c4["n_shuffles"]
P_ADD_ONE_CORRECTED = _c4["p_value_add_one_corrected"]

assert len(c1_samples) == C1_N
assert len(c4_samples) == C4_N

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=False)

panel_specs = [
    (axes[0], c1_samples, C1_MEAN, C1_SD, C1_N, "C1: 30-shuffle null", "#555555", "//"),
    (axes[1], c4_samples, C4_MEAN, C4_SD, C4_N, "C4: 500-shuffle null", "#1c4d8f", "\\\\"),
]

for ax, samples, mean, sd, n, label, color, hatch in panel_specs:
    bins = min(18, max(8, n // 4))
    ax.hist(samples, bins=bins, density=True, color=color, alpha=0.35,
            edgecolor=color, hatch=hatch, linewidth=1.0, label=f"{label}\n(empirical, n={n})")
    ax.axvline(R2_REAL, color="#b02418", linestyle="--", linewidth=2.0, zorder=5)
    ax.axvline(mean, color=color, linestyle=":", linewidth=1.4, zorder=4)
    ax.set_xlabel("Balanced accuracy (pre-calibration)", fontsize=10)
    ax.text(0.02, 0.97, f"mean={mean:.4f}\nSD={sd:.4f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.85))
    ax.legend(loc="upper right", fontsize=8.6, frameon=True)

axes[0].set_ylabel("Density", fontsize=10)

# Real-value label (shared)
axes[0].text(R2_REAL + 0.003, axes[0].get_ylim()[1] * 0.55, f"R2 real\n({R2_REAL:.4f})",
             color="#b02418", fontsize=9, ha="left", va="center", fontweight="bold")
axes[1].text(R2_REAL + 0.003, axes[1].get_ylim()[1] * 0.55, f"R2 real\n({R2_REAL:.4f})",
             color="#b02418", fontsize=9, ha="left", va="center", fontweight="bold")

# Annotations required by spec
fig.text(0.5, 0.965,
          "Empirical null densities: real per-shuffle values from the committed run JSONs",
          ha="center", va="top", fontsize=8.6, style="italic", color="#555555")

axes[1].annotate(f"p = {P_ADD_ONE_CORRECTED:.3f}\n(add-one corrected,\n{C4_N} shuffles)",
                  xy=(R2_REAL, axes[1].get_ylim()[1] * 0.30),
                  xytext=(0.62, 0.55), textcoords="axes fraction",
                  fontsize=9, color="#333333", ha="left",
                  arrowprops=dict(arrowstyle="-", color="#888888", linewidth=0.8, shrinkA=0, shrinkB=0))

# Gap annotation, placed centrally between the two panels.
# GATE D STEP 3c: draft.md's current Figure 2 caption states the gap vs.
# C4's null MEAN (7.78 +/- 0.05pp, NUMBERS.md's "vs. C4 null mean" row),
# not vs. the 0.50 chance floor (a different, also-valid, but not-cited-here
# quantity, 7.73 +/- 0.05pp) -- matched to the caption actually in force.
gap_vs_c4_pp = (R2_REAL - C4_MEAN) * 100
GAP_PP_TOL = 0.05
fig.text(0.5, 0.02,
          f"{gap_vs_c4_pp:.2f} $\\pm$ {GAP_PP_TOL}pp above C4's null mean "
          f"({R2_REAL:.4f} vs. {C4_MEAN:.4f})",
          ha="center", va="bottom", fontsize=9.5, color="#333333")

plt.tight_layout(rect=(0, 0.05, 1, 0.94))
for ext in ("pdf", "png"):
    plt.savefig(os.path.join(OUTDIR, f"fig3_null_distribution.{ext}"),
                dpi=300 if ext == "png" else None, bbox_inches="tight")
plt.close(fig)
print("Saved fig3_null_distribution.pdf / .png")
print(f"R2_REAL={R2_REAL}, C1_MEAN={C1_MEAN}, C1_SD={C1_SD}, C4_MEAN={C4_MEAN}, C4_SD={C4_SD}, P={P_ADD_ONE_CORRECTED}, gap_vs_c4_pp={gap_vs_c4_pp}")
