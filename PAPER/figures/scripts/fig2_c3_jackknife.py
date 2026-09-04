"""
Figure 2: C3 leave-one-subject-out jackknife plot.
Data source: results_c3_r2_jackknife.json (full 29-row LOO-mean table,
full-sample mean, and the pre-registered null-CI-upper threshold used
as C3's reference line) and results_c4_high_res_shuffled_label_control.json
(C4 null mean, second reference line). De-hardcoded GATE D STEP 2 --
every value below is read from these committed JSONs, not transcribed
by hand.
"""
import json
import matplotlib.pyplot as plt
import os

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"

SCRIPT_DIR = os.path.dirname(__file__)
OUTDIR = os.path.join(SCRIPT_DIR, "..")
REPO_ROOT = os.path.join(SCRIPT_DIR, "..", "..", "..")

with open(os.path.join(REPO_ROOT, "results_c3_r2_jackknife.json"), encoding="utf-8") as f:
    _c3 = json.load(f)
with open(os.path.join(REPO_ROOT, "results_c4_high_res_shuffled_label_control.json"), encoding="utf-8") as f:
    _c4 = json.load(f)

loo_rows = _c3["loo_results"]
assert len(loo_rows) == 29  # sub-09 already excluded from the study cohort

FULL_MEAN = _c3["full_mean_recomputed"]
NULL_CI_UPPER = _c3["null_95ci_upper_bound_used"]
C4_NULL_MEAN = _c4["null_distribution_mean"]

subjects_sorted = sorted(loo_rows, key=lambda r: r["excluded_subject"])
labels = [f"sub-{r['excluded_subject']}" for r in subjects_sorted]
values = [r["loo_mean"] for r in subjects_sorted]
x = list(range(len(subjects_sorted)))

row_max = max(loo_rows, key=lambda r: r["loo_mean"])
row_min = min(loo_rows, key=lambda r: r["loo_mean"])
idx_max = [r["excluded_subject"] for r in subjects_sorted].index(row_max["excluded_subject"])
idx_min = [r["excluded_subject"] for r in subjects_sorted].index(row_min["excluded_subject"])

fig, ax = plt.subplots(figsize=(11, 4.6))

ax.plot(x, values, marker="o", markersize=5, linewidth=1.2, color="#1c4d8f",
        markerfacecolor="#e2ecfa", markeredgecolor="#1c4d8f", zorder=4,
        label="Leave-one-out mean")

ax.axhline(FULL_MEAN, color="#333333", linestyle="-", linewidth=1.6, zorder=2)
ax.axhline(NULL_CI_UPPER, color="#b02418", linestyle="--", linewidth=1.6, zorder=2)
ax.axhline(C4_NULL_MEAN, color="#555555", linestyle=":", linewidth=1.4, zorder=2)

ax.text(len(x) - 0.5, FULL_MEAN + 0.0015, f"Full-sample mean ({FULL_MEAN:.4f})",
        ha="right", va="bottom", fontsize=9, color="#333333")
ax.text(len(x) - 0.5, NULL_CI_UPPER - 0.0025, f"Null CI upper bound ({NULL_CI_UPPER:.4f})",
        ha="right", va="top", fontsize=9, color="#b02418")
ax.text(0.5, C4_NULL_MEAN - 0.0025, f"C4 null mean ({C4_NULL_MEAN:.4f})",
        ha="left", va="top", fontsize=8.6, color="#555555")

# Highlight and label the two most extreme points (identity + value both from the data)
for idx, row, note in [
    (idx_max, row_max, f"sub-{row_max['excluded_subject']} excluded\n(highest, {row_max['loo_mean']:.4f})"),
    (idx_min, row_min, f"sub-{row_min['excluded_subject']} excluded\n(lowest, {row_min['loo_mean']:.4f})"),
]:
    val = row["loo_mean"]
    ax.scatter([idx], [val], s=90, facecolor="#f2d24b", edgecolor="#333333",
               zorder=6, linewidth=1.2)
    is_max = row is row_max
    y_off = 0.006 if is_max else -0.006
    va = "bottom" if is_max else "top"
    ax.annotate(note, xy=(idx, val), xytext=(idx, val + y_off),
                ha="center", va=va, fontsize=8.8, fontweight="bold", color="#333333",
                arrowprops=dict(arrowstyle="-", color="#888888", linewidth=0.8))

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=90, fontsize=8)
ax.set_xlabel("Subject excluded", fontsize=10)
ax.set_ylabel("Leave-one-out mean balanced accuracy", fontsize=10)
ax.set_ylim(0.49, 0.60)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(os.path.join(OUTDIR, f"fig2_c3_jackknife.{ext}"),
                dpi=300 if ext == "png" else None, bbox_inches="tight")
plt.close(fig)
print("Saved fig2_c3_jackknife.pdf / .png")
print(f"FULL_MEAN={FULL_MEAN}, NULL_CI_UPPER={NULL_CI_UPPER}, C4_NULL_MEAN={C4_NULL_MEAN}")
print(f"max: sub-{row_max['excluded_subject']} = {row_max['loo_mean']}, min: sub-{row_min['excluded_subject']} = {row_min['loo_mean']}")
