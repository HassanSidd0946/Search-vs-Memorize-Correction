"""
Figure 1: Discovery-sequence timeline.
Data source: results_r1b_r2_r3_composition_runs_20260825T042232Z_a3293792.json
(the rerun -- NUMBERS.md STEP 2e / GATE D STEP 1c attribute every R1(b)/R2/R3
quantity to this file, not the original run). Internal ledger IDs are
deliberately NOT rendered into the image itself (they are not a citable
source for a reader outside this project); provenance lives in the
manuscript's own captions and reference list instead.
70.78%, "13/13", and "4 checks" are genuine historical/count constants,
not read from a JSON field.
"""
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
from matplotlib.lines import Line2D
import os

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"  # Arial lacks the U+2713 checkmark glyph
plt.rcParams["hatch.linewidth"] = 0.5  # lighter hatching so node text stays legible

SCRIPT_DIR = os.path.dirname(__file__)
OUTDIR = os.path.join(SCRIPT_DIR, "..")
REPO_ROOT = os.path.join(SCRIPT_DIR, "..", "..", "..")

with open(os.path.join(REPO_ROOT, "results_r1b_r2_r3_composition_runs_20260825T042232Z_a3293792.json"),
          encoding="utf-8") as f:
    _runs = json.load(f)

R1B_POST_CAL = _runs["R1b_encode_vs_test"]["post_calibration_balanced_accuracy_mean"]
R2_PRE_CAL = _runs["R2_search_vs_memorize_encode_only"]["pre_calibration_balanced_accuracy_mean"]

fig, ax = plt.subplots(figsize=(11.5, 5.0))
ax.set_xlim(0, 11.5)
ax.set_ylim(-0.3, 5.0)
ax.axis("off")

box_w, box_h = 1.9, 1.1
y_center = 3.5
node_x = [0.6, 2.9, 5.2, 7.5]  # nodes 1-4 centers (left edge coords below)

node_style = dict(boxstyle="round,pad=0.15,rounding_size=0.08", linewidth=1.8)

# Node 1 — original headline (invalid). W6-round-2: solid ~12% tint +
# coloured border instead of crosshatch -- the busiest hatch pattern
# was the least legible under the text, worst of all in grayscale.
b1 = FancyBboxPatch((node_x[0], y_center - box_h / 2), box_w, box_h,
                     facecolor=mcolors.to_rgba("#b02418", alpha=0.12),
                     edgecolor="#b02418", linewidth=2.0,
                     **{k: v for k, v in node_style.items() if k != "linewidth"})
ax.add_patch(b1)
ax.text(node_x[0] + box_w / 2, y_center + 0.18, "70.78%", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#7a1810")
ax.text(node_x[0] + box_w / 2, y_center - 0.20, "(original)", ha="center", va="center",
        fontsize=9.5, color="#7a1810")
# strikethrough / X marker over node 1
ax.plot([node_x[0] + 0.12, node_x[0] + box_w - 0.12],
        [y_center - box_h / 2 + 0.12, y_center + box_h / 2 - 0.12],
        color="#7a1810", linewidth=2.2, solid_capstyle="round", zorder=5)
ax.plot([node_x[0] + 0.12, node_x[0] + box_w - 0.12],
        [y_center + box_h / 2 - 0.12, y_center - box_h / 2 + 0.12],
        color="#7a1810", linewidth=2.2, solid_capstyle="round", zorder=5)
ax.text(node_x[0] + box_w / 2, y_center - box_h / 2 - 0.42,
        "Contaminated contrast", ha="center", va="top", fontsize=9, color="#333333")

# Node 2 — R1(a) composition diagnosis (neutral/diagnostic)
b2 = FancyBboxPatch((node_x[1], y_center - box_h / 2), box_w, box_h,
                     facecolor="#eaeaea", edgecolor="#555555", hatch="..",
                     **node_style)
ax.add_patch(b2)
ax.text(node_x[1] + box_w / 2, y_center + 0.18, "R1(a): 13/13", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#333333")
ax.text(node_x[1] + box_w / 2, y_center - 0.20, "mappings fit", ha="center", va="center",
        fontsize=9.5, color="#333333")
ax.text(node_x[1] + box_w / 2, y_center - box_h / 2 - 0.42,
        "Composition arithmetic\nfits all prior effects",
        ha="center", va="top", fontsize=9, color="#333333")

# Node 3 — R1(b) joint criterion (confirmed / green)
b3 = FancyBboxPatch((node_x[2], y_center - box_h / 2), box_w, box_h,
                     facecolor="#e3f3e6", edgecolor="#1a7a34", hatch="//",
                     **node_style)
ax.add_patch(b3)
ax.text(node_x[2] + box_w / 2, y_center + 0.18, f"R1(b): {R1B_POST_CAL:.4f} ✓", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#0f5c22")
ax.text(node_x[2] + box_w / 2, y_center - 0.20, "confirmed", ha="center", va="center",
        fontsize=9.5, color="#0f5c22")
ax.text(node_x[2] + box_w / 2, y_center - box_h / 2 - 0.42,
        "Joint-criterion met\n(threshold: 0.85)",
        ha="center", va="top", fontsize=9, color="#333333")

# Node 4 — R2 corrected contrast (blue/result)
b4 = FancyBboxPatch((node_x[3], y_center - box_h / 2), box_w, box_h,
                     facecolor="#e2ecfa", edgecolor="#1c4d8f", hatch="|",
                     **node_style)
ax.add_patch(b4)
ax.text(node_x[3] + box_w / 2, y_center + 0.18, f"R2: {R2_PRE_CAL:.4f}", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#12305c")
ax.text(node_x[3] + box_w / 2, y_center - 0.20, "pre-cal, encode-only", ha="center", va="center",
        fontsize=9, color="#12305c")
ax.text(node_x[3] + box_w / 2, y_center - box_h / 2 - 0.42,
        "Corrected contrast\n(encode-only, pre-cal)",
        ha="center", va="top", fontsize=9, color="#333333")

# Arrows 1->2, 2->3, 3->4
arrow_style = dict(arrowstyle="-|>", mutation_scale=18, linewidth=1.8, color="#444444")
for i in range(3):
    x0 = node_x[i] + box_w
    x1 = node_x[i + 1]
    ax.add_patch(FancyArrowPatch((x0 + 0.05, y_center), (x1 - 0.05, y_center), **arrow_style))

ax.text((node_x[0] + box_w + node_x[1]) / 2, y_center + box_h / 2 + 0.20, "invalidated",
        ha="center", va="bottom", fontsize=9.5, style="italic", color="#7a1810")
ax.text((node_x[1] + box_w + node_x[2]) / 2, y_center + box_h / 2 + 0.20, "confirmed",
        ha="center", va="bottom", fontsize=9.5, style="italic", color="#0f5c22")

# Node 5 — the four validation controls, wrapping around / annotating Node 4
# Drawn as a bracket-style box beneath Node 4 with a short connector into it,
# rather than a fifth sequential node in the main chain. Placed with enough
# clearance below Node 4's caption text to avoid overlap.
wrap_x = node_x[3] - 0.35
wrap_y = y_center - box_h / 2 - 2.15
wrap_w = box_w + 0.7
wrap_h = 0.85
# W6-round-2: same solid-tint treatment as Node 1 -- the diagonal
# hatch under "4 checks / C1/C3/C4/D2" was still hard to read.
b5 = FancyBboxPatch((wrap_x, wrap_y), wrap_w, wrap_h,
                     boxstyle="round,pad=0.12,rounding_size=0.35",
                     facecolor=mcolors.to_rgba("#1a7a34", alpha=0.12),
                     edgecolor="#1a7a34", linewidth=1.8)
ax.add_patch(b5)
ax.text(wrap_x + wrap_w / 2, wrap_y + wrap_h / 2 + 0.14, "4 checks ✓",
        ha="center", va="center", fontsize=10.5, fontweight="bold", color="#0f5c22")
ax.text(wrap_x + wrap_w / 2, wrap_y + wrap_h / 2 - 0.18, "C1 / C3 / C4 / D2",
        ha="center", va="center", fontsize=8.8, color="#0f5c22")

# connector from node 5 up into node 4, routed around the outside (right edge)
# so it does not cross Node 4's caption text column below the box.
con = ConnectionPatch(
    xyA=(wrap_x + wrap_w, wrap_y + wrap_h * 0.7), coordsA=ax.transData,
    xyB=(node_x[3] + box_w, y_center - box_h * 0.15), coordsB=ax.transData,
    connectionstyle="arc3,rad=-0.35", arrowstyle="-|>", mutation_scale=16,
    linewidth=1.6, linestyle=(0, (4, 2)), color="#1a7a34",
)
ax.add_patch(con)
ax.text(node_x[3] + box_w + 0.55, (wrap_y + wrap_h + y_center) / 2 - 0.1,
        "validates", ha="left", va="center", fontsize=8.8, style="italic", color="#1a7a34")

# Legend (grayscale-safe: hatch + color both distinguish categories)
legend_elems = [
    Line2D([0], [0], marker="s", linestyle="none", markersize=12,
           markerfacecolor="#f2f2f2", markeredgecolor="#b02418", label="Invalid"),
    Line2D([0], [0], marker="s", linestyle="none", markersize=12,
           markerfacecolor="#eaeaea", markeredgecolor="#555555", label="Diagnostic"),
    Line2D([0], [0], marker="s", linestyle="none", markersize=12,
           markerfacecolor="#e3f3e6", markeredgecolor="#1a7a34", label="Confirmed / validated"),
    Line2D([0], [0], marker="s", linestyle="none", markersize=12,
           markerfacecolor="#e2ecfa", markeredgecolor="#1c4d8f", label="Result"),
]
ax.legend(handles=legend_elems, loc="upper right", bbox_to_anchor=(1.0, 1.08),
          fontsize=8.6, frameon=True, ncol=1)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(os.path.join(OUTDIR, f"fig1_discovery_timeline.{ext}"),
                dpi=300 if ext == "png" else None, bbox_inches="tight")
plt.close(fig)
print("Saved fig1_discovery_timeline.pdf / .png")
