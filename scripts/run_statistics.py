"""
scripts/run_statistics.py

F13 -- full statistical testing suite over results/loso_runs.csv (F4's tidy
long-format LOSO output, shared by run_step4_matched_spatial_control.py and
run_step4_condition4_asymmetric_mamba.py). Runs entirely locally (pure
pandas/scipy/numpy over an already-produced CSV) -- no Modal, no dataset
dependency.

WHAT THIS SCRIPT DOES:
  An "arm" is one experimental condition + flag combination (e.g.
  "asymmetric_fusion|ea=per-subject|cov=fixed|fusion=scaled-concat"). For
  every arm, per-subject scores are first averaged ACROSS SEEDS (F4 ran 5
  seeds per arm specifically so this averaging removes seed noise before
  any paired significance test -- standard practice for paired
  nonparametric tests over repeated-measures LOSO folds). The paired
  comparison unit is then "subject S's mean score under arm A" vs "subject
  S's mean score under arm B", for every subject present in both arms.

  For every PAIR of arms sharing >= 3 common subjects, and every metric in
  METRICS_TO_TEST, this script runs:
    - A two-sided Wilcoxon signed-rank test (paired, nonparametric -- the
      right test for LOSO fold-level comparisons, which are neither
      independent across subjects nor guaranteed normal).
    - Two paired effect sizes: matched-pairs rank-biserial correlation (the
      standard nonparametric companion to Wilcoxon) and Cohen's d_z
      (paired-differences standardized mean, for readers who want the
      parametric-equivalent number too).
  Holm-Bonferroni correction is then applied across the ENTIRE family of
  p-values produced in one invocation (every arm-pair x metric combination
  together, not corrected separately per metric or per pair), per the
  user's explicit "full Wilcoxon family + Holm-Bonferroni" instruction.

Usage:
  python scripts/run_statistics.py
  python scripts/run_statistics.py --csv results/loso_runs.csv --out results/statistics_report.json
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

METRICS_TO_TEST = [
    "post_calibration_acc", "roc_auc", "cohen_kappa", "balanced_accuracy",
    "macro_f1", "sensitivity", "specificity",
]

MIN_COMMON_SUBJECTS = 3  # below this, a signed-rank test is not meaningfully powered


def load_loso_runs(csv_path):
    df = pd.read_csv(csv_path)
    required = {"script", "condition", "seed", "ea_mode", "cov_estimator", "fusion_mode",
                "test_subject"} | set(METRICS_TO_TEST)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")
    return df


def make_arm_id(row):
    """Unique string identifying one experimental arm (condition + every
    flag that could change its numbers), independent of seed/fold."""
    fusion = row["fusion_mode"]
    fusion = fusion if isinstance(fusion, str) and fusion != "" else "n/a"
    return f"{row['condition']}|ea={row['ea_mode']}|cov={row['cov_estimator']}|fusion={fusion}"


def per_subject_means(df):
    """Average across seeds (F4: 5 seeds/arm), per (arm, test_subject).
    Returns dict: arm_id -> DataFrame indexed by test_subject, one column
    per metric in METRICS_TO_TEST."""
    df = df.copy()
    df["arm_id"] = df.apply(make_arm_id, axis=1)
    out = {}
    for arm_id, g in df.groupby("arm_id"):
        out[arm_id] = g.groupby("test_subject")[METRICS_TO_TEST].mean()
    return out


def rank_biserial_from_diffs(diffs):
    """Matched-pairs rank-biserial correlation from the raw paired
    differences: rank |diff| (ignoring exact zeros), split rank-sums by
    sign, normalize. Standard nonparametric effect size to report
    alongside a Wilcoxon signed-rank test."""
    diffs = np.asarray(diffs)
    nz = diffs[diffs != 0]
    if len(nz) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nz))
    r_plus = ranks[nz > 0].sum()
    r_minus = ranks[nz < 0].sum()
    total = r_plus + r_minus
    return float((r_plus - r_minus) / total) if total > 0 else 0.0


def cohens_dz(diffs):
    """Paired-differences standardized mean (parametric-equivalent effect
    size, reported alongside the nonparametric rank-biserial r)."""
    diffs = np.asarray(diffs)
    sd = diffs.std(ddof=1) if len(diffs) > 1 else 0.0
    return float(diffs.mean() / sd) if sd > 0 else 0.0


def holm_bonferroni(pvalues):
    """Standard Holm step-down procedure. Returns adjusted p-values in the
    ORIGINAL order of `pvalues` (monotone-enforced, capped at 1.0)."""
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * pvalues[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def run_family_of_tests(arm_means):
    """arm_means: dict arm_id -> DataFrame(index=test_subject, columns=METRICS_TO_TEST).
    For every pair of arms sharing >= MIN_COMMON_SUBJECTS subjects, runs a
    paired Wilcoxon signed-rank test per metric. Returns the raw (pre-Holm-
    Bonferroni) test records."""
    records = []
    arm_ids = sorted(arm_means.keys())
    for arm_a, arm_b in itertools.combinations(arm_ids, 2):
        df_a, df_b = arm_means[arm_a], arm_means[arm_b]
        common_subjects = sorted(set(df_a.index) & set(df_b.index))
        if len(common_subjects) < MIN_COMMON_SUBJECTS:
            continue
        for metric in METRICS_TO_TEST:
            a_vals = df_a.loc[common_subjects, metric].to_numpy()
            b_vals = df_b.loc[common_subjects, metric].to_numpy()
            diffs = a_vals - b_vals
            if np.allclose(diffs, 0):
                stat, p = float("nan"), 1.0
            else:
                try:
                    stat, p = stats.wilcoxon(a_vals, b_vals, zero_method="wilcox", correction=False)
                except ValueError:
                    # every difference was zero after dropping ties, or n too small
                    stat, p = float("nan"), 1.0
            records.append({
                "arm_a": arm_a, "arm_b": arm_b, "metric": metric,
                "n_subjects": len(common_subjects),
                "mean_a": float(np.mean(a_vals)), "mean_b": float(np.mean(b_vals)),
                "mean_diff_a_minus_b": float(np.mean(diffs)),
                "wilcoxon_statistic": (float(stat) if stat == stat else None),
                "p_value_raw": float(p),
                "rank_biserial_r": rank_biserial_from_diffs(diffs),
                "cohens_dz": cohens_dz(diffs),
            })
    return records


def main():
    parser = argparse.ArgumentParser(description="F13: full statistical testing suite over results/loso_runs.csv")
    parser.add_argument("--csv", default=os.path.join("results", "loso_runs.csv"))
    parser.add_argument("--out", default=os.path.join("results", "statistics_report.json"))
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found. Run the LOSO drivers (F4) on Modal first "
              f"to produce it.", file=sys.stderr)
        sys.exit(1)

    df = load_loso_runs(args.csv)
    arm_means = per_subject_means(df)
    print(f"Loaded {len(df)} rows -> {len(arm_means)} distinct arm(s): {sorted(arm_means.keys())}")

    records = run_family_of_tests(arm_means)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if not records:
        print(f"No arm pairs with >= {MIN_COMMON_SUBJECTS} common subjects found -- nothing "
              f"to test yet (need >= 2 arms run on overlapping subjects).")
        report = {"csv_path": args.csv, "n_rows_loaded": len(df),
                   "n_arms": len(arm_means), "arms": sorted(arm_means.keys()), "tests": []}
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved: {args.out}")
        return

    # Holm-Bonferroni across the WHOLE family produced in this invocation.
    raw_pvalues = [r["p_value_raw"] for r in records]
    adjusted = holm_bonferroni(raw_pvalues)
    for rec, p_adj in zip(records, adjusted):
        rec["p_value_holm_bonferroni"] = p_adj
        rec["significant_at_alpha"] = bool(p_adj < args.alpha)

    n_sig = sum(r["significant_at_alpha"] for r in records)
    print(f"Ran {len(records)} paired Wilcoxon tests across {len(arm_means)} arms "
          f"({n_sig}/{len(records)} significant after Holm-Bonferroni at alpha={args.alpha}).")

    report = {
        "csv_path": args.csv, "n_rows_loaded": len(df),
        "n_arms": len(arm_means), "arms": sorted(arm_means.keys()),
        "alpha": args.alpha, "correction": "Holm-Bonferroni (family-wise, whole invocation)",
        "metrics_tested": METRICS_TO_TEST, "min_common_subjects_required": MIN_COMMON_SUBJECTS,
        "n_tests": len(records), "n_significant_after_correction": n_sig,
        "tests": records,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
