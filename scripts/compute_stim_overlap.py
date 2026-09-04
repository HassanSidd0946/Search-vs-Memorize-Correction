"""
scripts/compute_stim_overlap.py

F-STIM (Phase 1, Gate 0 approved 2026-08-17)
=============================================================================
Computes the set overlap of (scene, obj) stimuli between Search and
Memorize blocks, per subject and pooled across all subjects, using the
Encode-phase behavioral files (*_task-SearchSupRecFamEncode_beh.tsv).

WHY THIS MATTERS:
  D1 reframes the primary contrast (Search vs. Memorize) as an
  incidental-vs-intentional ENCODING-TASK decoding problem. If the visual
  stimuli themselves differ systematically between the two task blocks
  (e.g. disjoint scene/object sets), a classifier could in principle be
  partly discriminating "which pictures were shown" rather than
  "which cognitive task the subject was performing" -- a visual-stimulus
  confound distinct from the block-order/session-position confound that
  F-PARITY targets. This script quantifies whether that confound exists.

Usage:
    python scripts/compute_stim_overlap.py
    python scripts/compute_stim_overlap.py --data-dir data/ds005189 --out results_stim_overlap.json

Includes sub-09 in the per-subject table (its behavioral file is intact
even though its EEG recording is truncated -- see AUDIT.md D2) but sub-09
is EXCLUDED from the pooled "n=29" statistic to stay consistent with the
EEG analysis population. A separate n=30 pooled statistic is also reported
for completeness, since the stimulus-design question is a property of the
experiment protocol, not of any one subject's EEG usability.
=============================================================================
"""

import argparse
import csv
import glob
import json
import os


def load_encode_trials(tsv_path):
    """Return list of (scene, obj, task) tuples from one subject's Encode_beh.tsv."""
    rows = []
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append((row["scene"].strip(), row["obj"].strip(), row["task"].strip()))
    return rows


def compute_overlap(rows):
    """Given a subject's trial rows, compute Search/Memorize (scene,obj) set overlap."""
    search_set = {(s, o) for (s, o, t) in rows if t.lower().startswith("search")}
    memo_set = {(s, o) for (s, o, t) in rows if t.lower().startswith("memor")}
    other_tasks = sorted({t for (_, _, t) in rows
                           if not t.lower().startswith("search") and not t.lower().startswith("memor")})
    inter = search_set & memo_set
    union = search_set | memo_set
    jaccard = len(inter) / len(union) if union else float("nan")
    return {
        "n_trials_total": len(rows),
        "n_search_trials": sum(1 for (_, _, t) in rows if t.lower().startswith("search")),
        "n_memorize_trials": sum(1 for (_, _, t) in rows if t.lower().startswith("memor")),
        "n_search_unique_stimuli": len(search_set),
        "n_memorize_unique_stimuli": len(memo_set),
        "n_overlapping_stimuli": len(inter),
        "jaccard_index": jaccard,
        "pct_search_stimuli_also_in_memorize": (
            100.0 * len(inter) / len(search_set) if search_set else float("nan")
        ),
        "pct_memorize_stimuli_also_in_search": (
            100.0 * len(inter) / len(memo_set) if memo_set else float("nan")
        ),
        "unrecognized_task_labels": other_tasks,
    }


def main():
    parser = argparse.ArgumentParser(description="F-STIM: compute Search/Memorize (scene,obj) stimulus overlap")
    parser.add_argument("--data-dir", default="data/ds005189", help="BIDS root directory")
    parser.add_argument("--out", default="results_stim_overlap.json", help="Output JSON path")
    parser.add_argument("--excluded-subjects", default="09",
                         help="Comma-separated subject IDs excluded from the pooled n=29 EEG-analysis statistic "
                              "(default: sub-09, per AUDIT.md D2)")
    args = parser.parse_args()

    excluded = {s.strip().zfill(2) for s in args.excluded_subjects.split(",") if s.strip()}

    tsv_paths = sorted(glob.glob(os.path.join(args.data_dir, "sub-*", "beh", "*SearchSupRecFamEncode_beh.tsv")))
    if not tsv_paths:
        raise FileNotFoundError(f"No Encode_beh.tsv files found under {args.data_dir}/sub-*/beh/")

    per_subject = {}
    all_rows_29 = []   # pooled, EEG-analysis population (n=29, excludes sub-09)
    all_rows_30 = []   # pooled, full protocol population (n=30)

    for tsv_path in tsv_paths:
        fname = os.path.basename(tsv_path)
        sub_id = fname.split("_")[0].replace("sub-", "")
        rows = load_encode_trials(tsv_path)
        per_subject[f"sub-{sub_id}"] = compute_overlap(rows)

        all_rows_30.extend(rows)
        if sub_id not in excluded:
            all_rows_29.extend(rows)

    pooled_29 = compute_overlap(all_rows_29)
    pooled_30 = compute_overlap(all_rows_30)

    # Summary across per-subject Jaccard indices (sanity check: is pooled
    # result driven by one outlier subject, or consistent across subjects?)
    per_subject_jaccards = [v["jaccard_index"] for k, v in per_subject.items()
                             if k.replace("sub-", "") not in excluded]
    jaccard_min = min(per_subject_jaccards) if per_subject_jaccards else float("nan")
    jaccard_max = max(per_subject_jaccards) if per_subject_jaccards else float("nan")
    jaccard_mean = sum(per_subject_jaccards) / len(per_subject_jaccards) if per_subject_jaccards else float("nan")

    result = {
        "description": (
            "F-STIM: (scene,obj) stimulus set overlap between Search and Memorize "
            "encoding-task blocks, per subject and pooled. Quantifies whether the "
            "primary Search-vs-Memorize contrast could be confounded by a purely "
            "visual (which-pictures-were-shown) signal rather than a task/cognitive-state signal."
        ),
        "n_subjects_found": len(per_subject),
        "excluded_from_pooled_29": sorted(excluded),
        "pooled_n29_eeg_analysis_population": pooled_29,
        "pooled_n30_full_protocol": pooled_30,
        "per_subject_jaccard_summary_n29": {
            "min": jaccard_min, "max": jaccard_max, "mean": jaccard_mean,
        },
        "per_subject": per_subject,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"n_subjects found: {len(per_subject)}")
    print(f"\n--- Pooled (n=29, EEG-analysis population, excludes {sorted(excluded)}) ---")
    for k, v in pooled_29.items():
        print(f"  {k}: {v}")
    print(f"\n--- Pooled (n=30, full protocol) ---")
    for k, v in pooled_30.items():
        print(f"  {k}: {v}")
    print(f"\n--- Per-subject Jaccard index (n=29 population) ---")
    print(f"  min={jaccard_min:.4f}  max={jaccard_max:.4f}  mean={jaccard_mean:.4f}")
    print(f"\nFull per-subject breakdown written to {args.out}")


if __name__ == "__main__":
    main()
