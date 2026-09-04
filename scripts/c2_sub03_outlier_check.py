# =============================================================================
# scripts/c2_sub03_outlier_check.py
#
# C2 -- SUB-03 OUTLIER CHECK (local, no Modal, no classifier).
# Pre-registered in DECISIONS.md's "C1 / C2 -- R2 shuffled-label control +
# sub-03 outlier check" section, BEFORE this script runs.
#
# WHY THIS SCRIPT EXISTS:
#   sub-03's R2 fold is pre_cal=0.3529 (balanced 0.3571) -- the only fold
#   below chance, and well below it. Before that sits inside the reported
#   R2 mean unflagged, this checks (a) whether sub-03's encode-only epoch
#   count/class balance is the expected 50/50 (ruling out a trivial
#   labeling/count defect) and (b) whether sub-03's block order and raw
#   marker parsing match the pattern seen in the other 28 subjects (29-
#   subject cohort minus sub-09's known truncated-recording exclusion,
#   AUDIT.md D2) -- nothing exotic, just ruling out a parsing edge case
#   unique to this subject.
#
# METHOD: parses each subject's raw BrainVision .vmrk marker file directly
#   (the same ground truth run_data_engine_granular_on_modal.py's
#   mne.events_from_annotations reads) -- all 30 subjects' .vmrk/beh files
#   are already cached locally under data/ds005189/ (used by
#   scripts/phase_composition_diagnostic.py and
#   scripts/identify_trial50_event.py), so this needs no Modal/download.
#   This gives an INDEPENDENT confirmation of (a) alongside the granular
#   npz's own per-subject counts (checked directly against the npz by
#   run_c2_sub03_npz_check.py, a separate tiny Modal script) -- an
#   agreement between the two is a stronger check than either alone.
#
# D2 counterbalancing reference (AUDIT.md/DECISIONS.md): odd-numbered
#   subjects are Search-first (block 1 starts with code-10-family markers),
#   even-numbered subjects are Memorize-first (code-20-family first). sub-03
#   is odd -> expected Search-first.
#
# Usage: python scripts/c2_sub03_outlier_check.py
# =============================================================================

import re
import statistics
from pathlib import Path

DATA_ROOT = Path("data/ds005189")
SFREQ = 1000.0  # original BrainVision recording rate

CODE_NAMES = {
    "10": "search_encode", "11": "search_test_target", "12": "search_test_distractor", "13": "search_test_lure",
    "20": "memorize_encode", "21": "memorize_test_target", "22": "memorize_test_distractor", "23": "memorize_test_lure",
}
ENCODE_CODES = {"10", "20"}
ALL_CODES = set(CODE_NAMES.keys())
CLASS0_CODES = {"10", "11", "12", "13"}  # Search
CLASS1_CODES = {"20", "21", "22", "23"}  # Memorize

# Any Mk<n>=... line at all, matched or not -- used to detect lines this
# script's stricter marker regex silently fails to parse (a parsing-coverage
# check, not just a content check).
MK_LINE_RE = re.compile(r"^Mk\d+=")
MK_RE = re.compile(r"^Mk\d+=([^,]*),\s*([^,]*),(\d+),")

EXCLUDED_SUBJECT = "sub-09"  # AUDIT.md D2: truncated recording, zero usable epochs
TARGET_SUBJECT = "sub-03"


def parse_vmrk_full(vmrk_path):
    """Returns (matched_stimulus_markers, total_mk_lines, unmatched_mk_lines,
    all_stimulus_positions_including_nonclass_codes).
    matched_stimulus_markers: list of (code:str, position:int) for the 8
    production class codes only, in file order."""
    matched, total_mk_lines, unmatched_mk_lines = [], 0, 0
    all_positions_seen = []
    with open(vmrk_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not MK_LINE_RE.match(line):
                continue
            total_mk_lines += 1
            m = MK_RE.match(line)
            if not m:
                unmatched_mk_lines += 1
                continue
            mk_type, desc, pos = m.group(1), m.group(2).strip(), int(m.group(3))
            if mk_type == "Stimulus":
                all_positions_seen.append(pos)
                if desc in ALL_CODES:
                    matched.append((desc, pos))
    return matched, total_mk_lines, unmatched_mk_lines, all_positions_seen


def analyze_subject(sub_tag):
    vmrk_candidates = list((DATA_ROOT / sub_tag / "eeg").glob("*.vmrk"))
    if not vmrk_candidates:
        return {"sub": sub_tag, "error": "no .vmrk file found"}
    matched, total_mk_lines, unmatched_mk_lines, all_stim_positions = parse_vmrk_full(vmrk_candidates[0])

    if len(matched) < 60:
        return {"sub": sub_tag, "error": f"only {len(matched)} matching production-code markers found"}

    # (b1) monotonicity / duplicate-position check across ALL parsed class-code markers
    positions = [p for _, p in matched]
    is_monotonic = all(positions[i] < positions[i + 1] for i in range(len(positions) - 1))
    n_duplicate_positions = len(positions) - len(set(positions))

    # (b2) per-code counts
    per_code_counts = {code: sum(1 for c, _ in matched if c == code) for code in ALL_CODES}

    # (b3) block split + first-class / counterbalancing check
    first_class = "search" if matched[0][0] in CLASS0_CODES else "memorize"
    subj_num = int(sub_tag.split("-")[1])
    expected_first_class = "search" if subj_num % 2 == 1 else "memorize"
    counterbalance_matches_expected = (first_class == expected_first_class)

    block1, block2 = [], []
    b1_class = first_class
    for c, p in matched:
        cls = "search" if c in CLASS0_CODES else "memorize"
        (block1 if cls == b1_class else block2).append((c, p))
    n_block1, n_block2 = len(block1), len(block2)

    # (b4) per-block per-code composition -- used to independently confirm
    # the encode-only (code 10/20) count and class balance sub-03 needs
    # checked (C2 item a), plus flag any block whose composition looks off
    # relative to the expected ~50/50/25/75 pattern.
    def block_code_counts(block):
        return {code: sum(1 for c, _ in block if c == code) for code in ALL_CODES}
    b1_counts, b2_counts = block_code_counts(block1), block_code_counts(block2)

    return {
        "sub": sub_tag,
        "total_mk_lines": total_mk_lines,
        "unmatched_mk_lines": unmatched_mk_lines,
        "n_stimulus_markers_seen": len(all_stim_positions),
        "n_class_code_markers_matched": len(matched),
        "positions_monotonic": is_monotonic,
        "n_duplicate_positions": n_duplicate_positions,
        "per_code_counts_total": per_code_counts,
        "first_class": first_class,
        "expected_first_class_per_D2": expected_first_class,
        "counterbalance_matches_expected": counterbalance_matches_expected,
        "n_block1": n_block1, "n_block2": n_block2,
        "block1_class": b1_class, "block2_class": ("memorize" if b1_class == "search" else "search"),
        "block1_code_counts": b1_counts, "block2_code_counts": b2_counts,
        "encode_only_count": per_code_counts["10"] + per_code_counts["20"],
        "encode_search_count": per_code_counts["10"],
        "encode_memorize_count": per_code_counts["20"],
    }


def main():
    subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("sub-"))
    print(f"Found {len(subjects)} subject directories.\n")

    results = {}
    for s in subjects:
        results[s] = analyze_subject(s)

    cohort = [s for s in subjects if s != EXCLUDED_SUBJECT and "error" not in results[s]]
    print(f"Analyzable cohort (excluding {EXCLUDED_SUBJECT} per D2): {len(cohort)} subjects\n")

    # ---- (a) sub-03 encode-only count / class balance, stated first and plainly ----
    print("=" * 78)
    print(f"(a) {TARGET_SUBJECT} ENCODE-ONLY EPOCH COUNT / CLASS BALANCE")
    print("=" * 78)
    r03 = results[TARGET_SUBJECT]
    if "error" in r03:
        print(f"  ERROR parsing {TARGET_SUBJECT}: {r03['error']}")
    else:
        print(f"  encode_search (code 10)   = {r03['encode_search_count']}  (expected 50)")
        print(f"  encode_memorize (code 20) = {r03['encode_memorize_count']}  (expected 50)")
        print(f"  encode_only_total         = {r03['encode_only_count']}  (expected 100)")
        balance_ok = (r03['encode_search_count'] == 50 and r03['encode_memorize_count'] == 50)
        print(f"  -> {'CONFIRMED: exactly 50/50, as expected.' if balance_ok else 'ANOMALY: NOT exactly 50/50 -- see full per-code breakdown below.'}")
        print(f"\n  Full per-code counts (all 8 production codes), {TARGET_SUBJECT}:")
        for code, name in sorted(CODE_NAMES.items()):
            print(f"    {code} ({name:>24}): {r03['per_code_counts_total'][code]}")

    # ---- (b) block order / raw marker parsing, sub-03 vs. the rest of the cohort ----
    print("\n" + "=" * 78)
    print(f"(b) {TARGET_SUBJECT} BLOCK ORDER / MARKER PARSING vs. COHORT (n={len(cohort)})")
    print("=" * 78)

    print(f"\n  Counterbalancing (D2: odd subject -> Search-first):")
    n_match = sum(1 for s in cohort if results[s]["counterbalance_matches_expected"])
    print(f"    Cohort: {n_match}/{len(cohort)} subjects match their expected first-class per D2's odd/even rule.")
    mismatches = [s for s in cohort if not results[s]["counterbalance_matches_expected"]]
    if mismatches:
        print(f"    Cohort mismatches (not sub-03-specific, reported for completeness): {mismatches}")
    if "error" not in r03:
        print(f"    {TARGET_SUBJECT}: first_class={r03['first_class']}, expected={r03['expected_first_class_per_D2']}, "
              f"match={r03['counterbalance_matches_expected']}")

    print(f"\n  Marker-file parsing integrity:")
    for s in [TARGET_SUBJECT] + [c for c in cohort if c != TARGET_SUBJECT]:
        r = results[s]
        if "error" in r:
            continue
        flag = "" if (r["positions_monotonic"] and r["n_duplicate_positions"] == 0 and r["unmatched_mk_lines"] == 0) else "  <-- FLAGGED"
        if s == TARGET_SUBJECT or flag:
            print(f"    {s}: monotonic={r['positions_monotonic']} dup_positions={r['n_duplicate_positions']} "
                  f"unmatched_mk_lines={r['unmatched_mk_lines']}{flag}")

    print(f"\n  Per-code count ranges across cohort (min-max), vs. {TARGET_SUBJECT}:")
    for code, name in sorted(CODE_NAMES.items()):
        cohort_counts = [results[s]["per_code_counts_total"][code] for s in cohort]
        lo, hi = min(cohort_counts), max(cohort_counts)
        med = statistics.median(cohort_counts)
        sub03_val = r03["per_code_counts_total"][code] if "error" not in r03 else None
        in_range = (lo <= sub03_val <= hi) if sub03_val is not None else None
        print(f"    {code} ({name:>24}): cohort range=[{lo},{hi}] median={med} | {TARGET_SUBJECT}={sub03_val} "
              f"{'(within cohort range)' if in_range else '(OUTSIDE cohort range)' if in_range is False else ''}")

    print(f"\n  Block sizes (n_block1/n_block2), {TARGET_SUBJECT} vs. cohort median:")
    n1s = [results[s]["n_block1"] for s in cohort]
    n2s = [results[s]["n_block2"] for s in cohort]
    print(f"    cohort median n_block1={statistics.median(n1s)}, n_block2={statistics.median(n2s)}")
    if "error" not in r03:
        print(f"    {TARGET_SUBJECT}: n_block1={r03['n_block1']}, n_block2={r03['n_block2']}")

    # ---- (c) explicit plain-language finding statement ----
    print("\n" + "=" * 78)
    print("(c) FINDING")
    print("=" * 78)
    if "error" in r03:
        print(f"  {TARGET_SUBJECT} could not be parsed: {r03['error']}. This IS an anomaly requiring investigation.")
    else:
        anomalies = []
        if not (r03['encode_search_count'] == 50 and r03['encode_memorize_count'] == 50):
            anomalies.append("encode-only class balance is not exactly 50/50")
        if not r03["counterbalance_matches_expected"]:
            anomalies.append("first-class does not match D2's odd/even counterbalancing expectation")
        if not r03["positions_monotonic"]:
            anomalies.append("marker positions are not strictly monotonic (possible out-of-order/duplicate markers)")
        if r03["n_duplicate_positions"] > 0:
            anomalies.append(f"{r03['n_duplicate_positions']} duplicate marker positions found")
        if r03["unmatched_mk_lines"] > 0:
            anomalies.append(f"{r03['unmatched_mk_lines']} Mk lines failed to parse against the marker regex")
        for code in ALL_CODES:
            cohort_counts = [results[s]["per_code_counts_total"][code] for s in cohort]
            lo, hi = min(cohort_counts), max(cohort_counts)
            if not (lo <= r03["per_code_counts_total"][code] <= hi):
                anomalies.append(f"code {code} ({CODE_NAMES[code]}) count {r03['per_code_counts_total'][code]} "
                                  f"falls outside the cohort's [{lo},{hi}] range")
        if anomalies:
            print(f"  {TARGET_SUBJECT} shows {len(anomalies)} anomaly(ies) relative to the cohort:")
            for a in anomalies:
                print(f"    - {a}")
            print("  These are reported as found -- do not silently drop this fold without this being weighed.")
        else:
            print(f"  NO ANOMALY FOUND. {TARGET_SUBJECT}'s encode-only epoch count is exactly 50/50 (100 total), "
                  "its block order matches D2's counterbalancing expectation (Search-first, odd subject), its "
                  "marker positions are strictly monotonic with no duplicates, every Mk line parsed cleanly, "
                  "and every one of its 8 per-code counts falls within the cohort's observed range.")
            print(f"  Per this control's pre-registered reporting rule: leave {TARGET_SUBJECT}'s R2 fold in the "
                  "reported mean as-is -- nothing here supports dropping or discounting it.")


if __name__ == "__main__":
    main()
