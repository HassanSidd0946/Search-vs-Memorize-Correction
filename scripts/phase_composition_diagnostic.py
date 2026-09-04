# =============================================================================
# scripts/phase_composition_diagnostic.py
#
# R1(a) PHASE-COMPOSITION DIAGNOSTIC -- local, no Modal, no classifier.
#
# For every pseudo-contrast construction used in evidence lines 4-8 of this
# audit (F-DRIFT, F-DRIFT-B, F-DRIFT-C, F-DRIFT-E), compute the REAL
# encode-phase fraction of each pseudo-class, using each subject's actual
# per-block encode-run-length (code 10/20) and block length, read directly
# from the raw .vmrk marker files -- not an assumed flat 50/200 split.
#
# This tests the unified explanation: that every prior "temporal" or
# "onset-proximity" effect in this audit is actually an encode-vs-test-phase
# COMPOSITION effect, since each ~200-epoch block is actually 50 encoding
# epochs followed by ~150 recognition-test epochs (targets/distractors/new
# lures), not 200 homogeneous "task" epochs.
#
# Usage: python scripts/phase_composition_diagnostic.py
# =============================================================================

import re
import statistics
from pathlib import Path

DATA_ROOT = Path("data/ds005189")
CLASS0_CODES = {"10", "11", "12", "13"}
CLASS1_CODES = {"20", "21", "22", "23"}
ENCODE_CODES = {"10", "20"}

MK_RE = re.compile(r"^Mk\d+=([^,]*),\s*([^,]*),(\d+),")


def parse_vmrk_codes(vmrk_path):
    """Return list of code strings (chronological order) for every Stimulus
    marker matching the production class codes."""
    codes = []
    with open(vmrk_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = MK_RE.match(line.strip())
            if not m:
                continue
            mk_type, desc, pos = m.group(1), m.group(2).strip(), int(m.group(3))
            if mk_type == "Stimulus" and (desc in CLASS0_CODES or desc in CLASS1_CODES):
                codes.append(desc)
    return codes


def split_blocks(codes):
    first_class = "0" if codes[0] in CLASS0_CODES else "1"
    block1, block2 = [], []
    for c in codes:
        cls = "0" if c in CLASS0_CODES else "1"
        (block1 if cls == first_class else block2).append(c)
    return block1, block2


def encode_run_length(block_codes):
    """Length of the leading contiguous run of encode-phase codes (10 or 20)."""
    n = 0
    for c in block_codes:
        if c in ENCODE_CODES:
            n += 1
        else:
            break
    return n


def encode_fraction(block_codes, lo, hi):
    """Fraction of block_codes[lo:hi] (0-indexed, exclusive hi) that are
    encode-phase codes."""
    window = block_codes[lo:hi]
    if not window:
        return None
    return sum(1 for c in window if c in ENCODE_CODES) / len(window)


def analyze_subject(sub_tag):
    vmrk_candidates = list((DATA_ROOT / sub_tag / "eeg").glob("*.vmrk"))
    if not vmrk_candidates:
        return None
    codes = parse_vmrk_codes(vmrk_candidates[0])
    if len(codes) < 60:
        return None
    b1, b2 = split_blocks(codes)
    n1, n2 = len(b1), len(b2)
    enc1, enc2 = encode_run_length(b1), encode_run_length(b2)

    result = {"sub": sub_tag, "n1": n1, "n2": n2, "enc1": enc1, "enc2": enc2}

    # ---- F-DRIFT-E / F-DRIFT-F(b)-style fixed positions, per block ----
    for frac in (0.25, 0.50, 0.75):
        for name, block_codes, n in (("block1", b1, n1), ("block2", b2, n2)):
            split = int(round(frac * n))
            frac_before = encode_fraction(block_codes, 0, split)
            frac_after = encode_fraction(block_codes, split, n)
            result[f"{name}_{int(frac*100)}pct_before_encode_frac"] = frac_before
            result[f"{name}_{int(frac*100)}pct_after_encode_frac"] = frac_after

    # ---- true boundary: last W of block1 vs first W of block2 (W = min(n1,n2)//4 as a representative window) ----
    W_true = min(n1, n2) // 4
    result["true_boundary_before_encode_frac"] = encode_fraction(b1, n1 - W_true, n1)
    result["true_boundary_after_encode_frac"] = encode_fraction(b2, 0, W_true)

    # ---- F-DRIFT early/late split (per-subject midpoint n//2), both real classes ----
    for name, block_codes, n in (("block1", b1, n1), ("block2", b2, n2)):
        mid = n // 2
        result[f"{name}_early_encode_frac"] = encode_fraction(block_codes, 0, mid)
        result[f"{name}_late_encode_frac"] = encode_fraction(block_codes, mid, n)

    # ---- F-DRIFT-B interleaved (odd/even), both real classes ----
    for name, block_codes, n in (("block1", b1, n1), ("block2", b2, n2)):
        even_idx_codes = block_codes[0::2]
        odd_idx_codes = block_codes[1::2]
        result[f"{name}_interleaved_even_encode_frac"] = (
            sum(1 for c in even_idx_codes if c in ENCODE_CODES) / len(even_idx_codes) if even_idx_codes else None)
        result[f"{name}_interleaved_odd_encode_frac"] = (
            sum(1 for c in odd_idx_codes if c in ENCODE_CODES) / len(odd_idx_codes) if odd_idx_codes else None)

    # ---- F-DRIFT-C k-sweep, both real classes ----
    for k in (1, 2, 5, 10, 25, 50, 100):
        for name, block_codes, n in (("block1", b1, n1), ("block2", b2, n2)):
            pseudo0_codes = [block_codes[i] for i in range(n) if (i // k) % 2 == 0]
            pseudo1_codes = [block_codes[i] for i in range(n) if (i // k) % 2 == 1]
            f0 = sum(1 for c in pseudo0_codes if c in ENCODE_CODES) / len(pseudo0_codes) if pseudo0_codes else None
            f1 = sum(1 for c in pseudo1_codes if c in ENCODE_CODES) / len(pseudo1_codes) if pseudo1_codes else None
            result[f"{name}_k{k}_pseudo0_encode_frac"] = f0
            result[f"{name}_k{k}_pseudo1_encode_frac"] = f1

    return result


def mean_of(results, key):
    vals = [r[key] for r in results if r.get(key) is not None]
    return statistics.mean(vals) if vals else None


def main():
    subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("sub-"))
    results = []
    for s in subjects:
        r = analyze_subject(s)
        if r is not None:
            results.append(r)
    print(f"Analyzed {len(results)} subjects with usable .vmrk files.\n")

    print("=" * 78)
    print("PER-SUBJECT BLOCK/ENCODE-RUN LENGTHS (first 8 shown)")
    print("=" * 78)
    for r in results[:8]:
        print(f"  {r['sub']}: n1={r['n1']} enc1={r['enc1']} | n2={r['n2']} enc2={r['enc2']}")

    print("\n" + "=" * 78)
    print("FIXED-POSITION SWEEP (F-DRIFT-E style) -- mean encode fraction, before/after split")
    print("=" * 78)
    for frac in (25, 50, 75):
        for name in ("block1", "block2"):
            b = mean_of(results, f"{name}_{frac}pct_before_encode_frac")
            a = mean_of(results, f"{name}_{frac}pct_after_encode_frac")
            print(f"  {name}_{frac}pct: before_encode_frac={b:.4f} after_encode_frac={a:.4f} "
                  f"| composition_gap={abs(b-a):.4f}")
    tb_b = mean_of(results, "true_boundary_before_encode_frac")
    tb_a = mean_of(results, "true_boundary_after_encode_frac")
    print(f"  true_boundary: before_encode_frac={tb_b:.4f} after_encode_frac={tb_a:.4f} "
          f"| composition_gap={abs(tb_b-tb_a):.4f}")

    print("\n" + "=" * 78)
    print("EARLY/LATE SPLIT (F-DRIFT style) -- mean encode fraction")
    print("=" * 78)
    for name in ("block1", "block2"):
        e = mean_of(results, f"{name}_early_encode_frac")
        l = mean_of(results, f"{name}_late_encode_frac")
        print(f"  {name}: early_encode_frac={e:.4f} late_encode_frac={l:.4f} | composition_gap={abs(e-l):.4f}")

    print("\n" + "=" * 78)
    print("INTERLEAVED ODD/EVEN (F-DRIFT-B style) -- mean encode fraction")
    print("=" * 78)
    for name in ("block1", "block2"):
        ev = mean_of(results, f"{name}_interleaved_even_encode_frac")
        od = mean_of(results, f"{name}_interleaved_odd_encode_frac")
        print(f"  {name}: even_encode_frac={ev:.4f} odd_encode_frac={od:.4f} | composition_gap={abs(ev-od):.4f}")

    print("\n" + "=" * 78)
    print("K-SWEEP (F-DRIFT-C style) -- mean encode fraction per pseudo-class, both real blocks pooled")
    print("=" * 78)
    for k in (1, 2, 5, 10, 25, 50, 100):
        f0s, f1s = [], []
        for name in ("block1", "block2"):
            f0 = mean_of(results, f"{name}_k{k}_pseudo0_encode_frac")
            f1 = mean_of(results, f"{name}_k{k}_pseudo1_encode_frac")
            if f0 is not None: f0s.append(f0)
            if f1 is not None: f1s.append(f1)
        f0_pooled = statistics.mean(f0s) if f0s else None
        f1_pooled = statistics.mean(f1s) if f1s else None
        gap = abs(f0_pooled - f1_pooled) if (f0_pooled is not None and f1_pooled is not None) else None
        print(f"  k={k:>3}: pseudo0_encode_frac={f0_pooled:.4f} pseudo1_encode_frac={f1_pooled:.4f} "
              f"| composition_gap={gap:.4f}")


if __name__ == "__main__":
    main()
