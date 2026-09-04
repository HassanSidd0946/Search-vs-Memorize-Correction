# =============================================================================
# scripts/identify_trial50_event.py
#
# F-DRIFT-F(b)/L029 FOLLOW-UP: IDENTIFY THE REST-BREAK DISCONTINUITY (no
# Modal, pure local data inspection against the already-cached
# data/ds005189/ raw .vmrk marker files and behavioral .tsv files).
#
# NAMING (2026-08-20, per explicit instruction): the structural event
# found here is a SELF-PACED REST BREAK, not a "session-onset transient"
# -- renamed "rest-break discontinuity" everywhere (this script,
# RESULTS_LEDGER.md, DECISIONS.md, results/SYNTHESIS.md). The term
# "onset transient" is retired for this specific event; "onset" still
# correctly refers to the separate, already-established concept of block
# onset (trial index 0 of a block, from F-DRIFT-E) -- do not conflate the
# two: this event sits at trial ~50 WITHIN a block, not at the block's
# start.
#
# WHY THIS SCRIPT EXISTS:
#   F-DRIFT-F(b)'s onset-distance sweep found a step-shaped balanced-accuracy
#   profile with a spike at distance~50 trials from each block's onset,
#   replicating independently in both blocks (block1 d50=0.8343 pooled /
#   block2 d50=0.8405 raw). This is NOT a smooth decaying transient (which
#   would peak at d25) -- something structural happens near trial 50 of
#   each block. This script inspects the raw marker timing and behavioral
#   records directly to find out what, across the FULL 29-subject cohort
#   (initially run on 11 locally-cached subjects; extended to all 30 via
#   a small openneuro-py download restricted to .vmrk/.vhdr/beh files
#   only -- no multi-hundred-MB .eeg binaries needed for this analysis).
#
# METHOD: parse each subject's raw BrainVision .vmrk marker file (which is
#   what run_data_engine_on_modal.py's mne.events_from_annotations reads,
#   at the ORIGINAL 1000 Hz recording rate -- SamplingInterval=1000us per
#   the .vhdr header, confirmed for sub-01) for Stimulus markers matching
#   the exact production EVENT_ID codes (10/11/12/13 -> class 0,
#   20/21/22/23 -> class 1). For each subject, split into block1 (first
#   contiguous class) / block2 (second), compute the inter-marker time gap
#   (seconds) between every consecutive pair of matching markers within
#   each block, and specifically inspect the gap at index ~50 against the
#   block's own median gap. Cross-reference against the behavioral
#   Encode.tsv file's task_num/trial_num columns to see whether epoch-index
#   50 corresponds to a natural behavioral sub-block boundary.
#
# Usage: python scripts/identify_trial50_event.py
# =============================================================================

import csv
import re
import statistics
from pathlib import Path

DATA_ROOT = Path("data/ds005189")
SFREQ = 1000.0  # original BrainVision recording rate (SamplingInterval=1000us)

CLASS0_CODES = {"10", "11", "12", "13"}
CLASS1_CODES = {"20", "21", "22", "23"}

MK_RE = re.compile(r"^Mk\d+=([^,]*),\s*([^,]*),(\d+),")


def parse_vmrk_markers(vmrk_path):
    """Return list of (code:str, position:int) for every Stimulus marker
    whose description matches one of the production class codes."""
    markers = []
    with open(vmrk_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = MK_RE.match(line.strip())
            if not m:
                continue
            mk_type, desc, pos = m.group(1), m.group(2).strip(), int(m.group(3))
            if mk_type == "Stimulus" and (desc in CLASS0_CODES or desc in CLASS1_CODES):
                markers.append((desc, pos))
    return markers


def split_blocks(markers):
    """markers: list of (code, position) in recording order. Returns
    (block1_positions, block2_positions, block1_first_class)."""
    first_class = "0" if markers[0][0] in CLASS0_CODES else "1"
    block1, block2 = [], []
    for code, pos in markers:
        cls = "0" if code in CLASS0_CODES else "1"
        (block1 if cls == first_class else block2).append(pos)
    return block1, block2, first_class


def gaps_seconds(positions):
    return [(positions[i + 1] - positions[i]) / SFREQ for i in range(len(positions) - 1)]


def analyze_subject(sub_tag):
    vmrk_candidates = list((DATA_ROOT / sub_tag / "eeg").glob("*.vmrk"))
    if not vmrk_candidates:
        return None
    markers = parse_vmrk_markers(vmrk_candidates[0])
    if len(markers) < 60:
        return {"sub": sub_tag, "error": f"only {len(markers)} matching markers found"}

    b1, b2, first_class = split_blocks(markers)
    result = {"sub": sub_tag, "n_block1": len(b1), "n_block2": len(b2), "first_class": first_class}

    for name, positions in (("block1", b1), ("block2", b2)):
        if len(positions) < 55:
            result[f"{name}_note"] = f"only {len(positions)} events, cannot inspect gap at index 50"
            continue
        gaps = gaps_seconds(positions)
        median_gap = statistics.median(gaps)
        # gap BEFORE epoch-index 50 (0-indexed) = gaps[49] = time between
        # marker 49 and marker 50 (the 50th and 51st events).
        gap_at_50 = gaps[49]
        # also scan the whole block for the single largest gap and its index,
        # in case the real structural break isn't exactly at 50.
        max_gap = max(gaps)
        max_gap_idx = gaps.index(max_gap)
        result[f"{name}_median_gap_s"] = round(median_gap, 3)
        result[f"{name}_gap_at_idx50_s"] = round(gap_at_50, 3)
        result[f"{name}_gap_at_idx50_ratio"] = round(gap_at_50 / median_gap, 2) if median_gap > 0 else None
        result[f"{name}_max_gap_s"] = round(max_gap, 3)
        result[f"{name}_max_gap_idx"] = max_gap_idx
        result[f"{name}_max_gap_ratio"] = round(max_gap / median_gap, 2) if median_gap > 0 else None

    # Cross-reference against the behavioral Encode file's task_num switch.
    beh_path = DATA_ROOT / sub_tag / "beh" / f"{sub_tag}_task-SearchSupRecFamEncode_beh.tsv"
    if beh_path.exists():
        with open(beh_path) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        prev_task = None
        switches = []
        for i, r in enumerate(rows):
            if r["task"] != prev_task:
                switches.append((i, r["task"], r["task_num"]))
                prev_task = r["task"]
        result["beh_n_trials"] = len(rows)
        result["beh_task_switches"] = switches
        # ratio of EEG events (block1+block2) to behavioral trial rows,
        # to translate epoch-index-50 into an approximate behavioral trial
        # index within that block.
        n_beh_first_task = switches[1][0] if len(switches) > 1 else len(rows)
        events_per_class = result.get("n_block1", 0)
        if n_beh_first_task > 0:
            result["events_per_behavioral_trial_block1"] = round(events_per_class / n_beh_first_task, 2)
            result["approx_behavioral_trial_at_epoch50_block1"] = round(
                50 / (events_per_class / n_beh_first_task), 1)

    return result


def main():
    subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("sub-"))
    print(f"Found {len(subjects)} subject directories.\n")

    all_results = []
    for sub_tag in subjects:
        r = analyze_subject(sub_tag)
        if r is None:
            print(f"{sub_tag}: no .vmrk found, skipping")
            continue
        all_results.append(r)
        if "error" in r:
            print(f"{sub_tag}: {r['error']}")
            continue
        print(f"{sub_tag}: n_block1={r['n_block1']} n_block2={r['n_block2']} first_class={r['first_class']}")
        for name in ("block1", "block2"):
            if f"{name}_median_gap_s" in r:
                print(f"  {name}: median_gap={r[f'{name}_median_gap_s']}s | "
                      f"gap_at_idx50={r[f'{name}_gap_at_idx50_s']}s "
                      f"(ratio={r[f'{name}_gap_at_idx50_ratio']}x median) | "
                      f"largest_gap={r[f'{name}_max_gap_s']}s at idx={r[f'{name}_max_gap_idx']} "
                      f"(ratio={r[f'{name}_max_gap_ratio']}x median)")
            elif f"{name}_note" in r:
                print(f"  {name}: {r[f'{name}_note']}")
        if "beh_task_switches" in r:
            print(f"  behavioral Encode file: {r['beh_n_trials']} trials, "
                  f"task switches at rows {r['beh_task_switches']}")
            if "approx_behavioral_trial_at_epoch50_block1" in r:
                print(f"  ~{r['events_per_behavioral_trial_block1']} EEG events per behavioral trial "
                      f"(block1) -> epoch-index 50 approx. behavioral trial "
                      f"#{r['approx_behavioral_trial_at_epoch50_block1']}")
        print()

    # ---- SUMMARY across all subjects ----
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    ratios_50_b1 = [r["block1_gap_at_idx50_ratio"] for r in all_results
                    if r.get("block1_gap_at_idx50_ratio") is not None]
    ratios_50_b2 = [r["block2_gap_at_idx50_ratio"] for r in all_results
                    if r.get("block2_gap_at_idx50_ratio") is not None]
    max_idx_b1 = [r["block1_max_gap_idx"] for r in all_results if "block1_max_gap_idx" in r]
    max_idx_b2 = [r["block2_max_gap_idx"] for r in all_results if "block2_max_gap_idx" in r]

    print(f"Gap-at-index-50 ratio to median (block1): n={len(ratios_50_b1)}, "
          f"mean={statistics.mean(ratios_50_b1):.2f}, median={statistics.median(ratios_50_b1):.2f}, "
          f"max={max(ratios_50_b1):.2f}, min={min(ratios_50_b1):.2f}")
    print(f"Gap-at-index-50 ratio to median (block2): n={len(ratios_50_b2)}, "
          f"mean={statistics.mean(ratios_50_b2):.2f}, median={statistics.median(ratios_50_b2):.2f}, "
          f"max={max(ratios_50_b2):.2f}, min={min(ratios_50_b2):.2f}")
    print(f"\nLargest-gap index distribution (block1): {sorted(max_idx_b1)}")
    print(f"Largest-gap index distribution (block2): {sorted(max_idx_b2)}")
    n_near_50_b1 = sum(1 for i in max_idx_b1 if 40 <= i <= 60)
    n_near_50_b2 = sum(1 for i in max_idx_b2 if 40 <= i <= 60)
    print(f"\nSubjects where the SINGLE LARGEST gap in block1 falls within [40,60]: "
          f"{n_near_50_b1}/{len(max_idx_b1)}")
    print(f"Subjects where the SINGLE LARGEST gap in block2 falls within [40,60]: "
          f"{n_near_50_b2}/{len(max_idx_b2)}")

    events_per_trial = [r["events_per_behavioral_trial_block1"] for r in all_results
                         if "events_per_behavioral_trial_block1" in r]
    approx_trial_at_50 = [r["approx_behavioral_trial_at_epoch50_block1"] for r in all_results
                           if "approx_behavioral_trial_at_epoch50_block1" in r]
    if events_per_trial:
        print(f"\nEEG events per behavioral trial (block1): mean={statistics.mean(events_per_trial):.2f}, "
              f"range=[{min(events_per_trial)},{max(events_per_trial)}]")
        print(f"Approx. behavioral trial index at epoch-50 (block1): "
              f"mean={statistics.mean(approx_trial_at_50):.1f}, range=[{min(approx_trial_at_50)},{max(approx_trial_at_50)}]")


if __name__ == "__main__":
    main()
