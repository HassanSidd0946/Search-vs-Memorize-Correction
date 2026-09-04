# =============================================================================
# command to run (requires processed_eeg_all_subjects_granular.npz to
# already exist on the volume -- run run_data_engine_granular_on_modal.py
# ::main first if it does not):
#   modal run run_c2_sub03_npz_check.py::main
# run_c2_sub03_npz_check.py
#
# C2(a) -- confirm sub-03's encode-only epoch count / class balance
# DIRECTLY AGAINST THE GRANULAR NPZ (not re-derived from raw markers).
# Pre-registered in DECISIONS.md's "C1 / C2" section. Cheap, no LOSO --
# just loads the npz and counts, seconds of compute.
#
# This is the direct npz-side counterpart to
# scripts/c2_sub03_outlier_check.py's independent local .vmrk parse
# (already run, real data, real output reviewed -- found NO anomaly: 50/50
# encode-only balance, Search-first block order matching D2, clean marker
# parsing, every per-code count inside the cohort range). Agreement between
# this npz-side count and that local parse is a stronger check than either
# alone, since they derive from the same raw markers via two independent
# code paths (this script's mne.events_from_annotations-produced npz vs.
# the other script's from-scratch regex parse of the raw .vmrk text).
#
# Usage: modal run run_c2_sub03_npz_check.py::main
# =============================================================================

import modal

app    = modal.App("bci-c2-sub03-npz-check")
volume = modal.Volume.from_name("eeg-data-vol")

GRANULAR_DATA_PATH = "/data/processed_eeg_all_subjects_granular.npz"
VOLUME_PATH         = "/data"

CODE_NAMES = {
    0: "search_encode", 1: "search_test_target", 2: "search_test_distractor", 3: "search_test_lure",
    4: "memorize_encode", 5: "memorize_test_target", 6: "memorize_test_distractor", 7: "memorize_test_lure",
}
TARGET_SUBJECT_ID = "03"

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy<2")


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=600, memory=4096)
def check_sub03():
    import numpy as np

    raw = np.load(GRANULAR_DATA_PATH, allow_pickle=True)
    code_np = raw["code"].astype(np.int64)
    subjects_np = raw["subjects"]

    unique_subjects = sorted(np.unique(subjects_np).tolist())
    is_sub03 = subjects_np == TARGET_SUBJECT_ID
    n_sub03 = int(is_sub03.sum())
    if n_sub03 == 0:
        return {"error": f"subject id '{TARGET_SUBJECT_ID}' not found in granular npz. "
                          f"Available subject ids: {unique_subjects}"}

    sub03_codes = code_np[is_sub03]
    per_code_counts_sub03 = {CODE_NAMES[c]: int(np.sum(sub03_codes == c)) for c in range(8)}
    encode_search = per_code_counts_sub03["search_encode"]
    encode_memorize = per_code_counts_sub03["memorize_encode"]

    # Cohort per-code count ranges (excluding sub-09, AUDIT.md D2) for the
    # same "does sub-03 fall outside the cohort range" check the local
    # marker-parse script already ran, computed here independently from the
    # npz side.
    cohort_ranges = {}
    for c in range(8):
        counts = []
        for s in unique_subjects:
            if s == "09":
                continue
            counts.append(int(np.sum(code_np[subjects_np == s] == c)))
        cohort_ranges[CODE_NAMES[c]] = {"min": min(counts), "max": max(counts), "median": sorted(counts)[len(counts)//2]}

    balance_ok = (encode_search == 50 and encode_memorize == 50)
    in_range = all(cohort_ranges[name]["min"] <= per_code_counts_sub03[name] <= cohort_ranges[name]["max"]
                   for name in per_code_counts_sub03)

    result = {
        "subject": "sub-03",
        "n_total_epochs_sub03": n_sub03,
        "per_code_counts_sub03": per_code_counts_sub03,
        "encode_search_count": encode_search,
        "encode_memorize_count": encode_memorize,
        "encode_only_total": encode_search + encode_memorize,
        "balance_exactly_50_50": bool(balance_ok),
        "cohort_per_code_ranges_excl_sub09": cohort_ranges,
        "all_codes_within_cohort_range": bool(in_range),
    }
    print("=" * 70)
    print("C2(a) -- sub-03 per-code counts, direct from granular npz")
    print("=" * 70)
    for name, n in per_code_counts_sub03.items():
        rng = cohort_ranges[name]
        print(f"  {name:>24}: sub-03={n}  cohort=[{rng['min']},{rng['max']}] median={rng['median']}")
    print(f"\n  encode-only balance exactly 50/50: {balance_ok}")
    print(f"  all 8 codes within cohort range:   {in_range}")
    return result


@app.local_entrypoint()
def main():
    print("C2(a) -- confirming sub-03's encode-only epoch count/class balance directly against the granular npz.")
    result = check_sub03.remote()
    print("\nRESULT:")
    for k, v in result.items():
        print(f"  {k}: {v}")
