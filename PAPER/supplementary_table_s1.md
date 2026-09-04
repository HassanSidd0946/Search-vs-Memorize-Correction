Supplementary Table S1. Leave-one-subject-out (LOO) jackknife analysis of R2 pre-calibration balanced accuracy (C3).

| Subject excluded | Per-fold accuracy | LOO mean | Shift from full-sample mean |
|---|---|---|---|
| sub-01 | 0.6600 ± 0.012 | 0.5744 ± 0.0005 | -0.0030 ± 0.0005 |
| sub-02 | 0.5188 ± 0.012 | 0.5794 ± 0.0005 | +0.0021 ± 0.0005 |
| sub-03 | 0.3571 ± 0.012 | 0.5852 ± 0.0005 | +0.0079 ± 0.0005 |
| sub-04 | 0.5720 ± 0.012 | 0.5775 ± 0.0005 | +0.0002 ± 0.0005 |
| sub-05 | 0.5296 ± 0.012 | 0.5790 ± 0.0005 | +0.0017 ± 0.0005 |
| sub-06 | 0.6921 ± 0.012 | 0.5732 ± 0.0005 | -0.0041 ± 0.0005 |
| sub-07 | 0.7173 ± 0.012 | 0.5723 ± 0.0005 | -0.0050 ± 0.0005 |
| sub-08 | 0.5570 ± 0.012 | 0.5781 ± 0.0005 | +0.0007 ± 0.0005 |
| sub-10 | 0.5739 ± 0.012 | 0.5775 ± 0.0005 | +0.0001 ± 0.0005 |
| sub-11 | 0.6346 ± 0.012 | 0.5753 ± 0.0005 | -0.0020 ± 0.0005 |
| sub-12 | 0.5651 ± 0.012 | 0.5778 ± 0.0005 | +0.0004 ± 0.0005 |
| sub-13 | 0.7040 ± 0.012 | 0.5728 ± 0.0005 | -0.0045 ± 0.0005 |
| sub-14 | 0.5000 ± 0.012 | 0.5801 ± 0.0005 | +0.0028 ± 0.0005 |
| sub-15 | 0.5778 ± 0.012 | 0.5773 ± 0.0005 | 0.0000 ± 0.0005 |
| sub-16 | 0.5880 ± 0.012 | 0.5770 ± 0.0005 | -0.0004 ± 0.0005 |
| sub-17 | 0.5404 ± 0.012 | 0.5787 ± 0.0005 | +0.0013 ± 0.0005 |
| sub-18 | 0.7746 ± 0.012 | 0.5703 ± 0.0005 | -0.0070 ± 0.0005 |
| sub-19 | 0.6476 ± 0.012 | 0.5748 ± 0.0005 | -0.0025 ± 0.0005 |
| sub-20 | 0.4654 ± 0.012 | 0.5813 ± 0.0005 | +0.0040 ± 0.0005 |
| sub-21 | 0.5797 ± 0.012 | 0.5773 ± 0.0005 | -0.0001 ± 0.0005 |
| sub-22 | 0.5905 ± 0.012 | 0.5769 ± 0.0005 | -0.0005 ± 0.0005 |
| sub-23 | 0.5568 ± 0.012 | 0.5781 ± 0.0005 | +0.0007 ± 0.0005 |
| sub-24 | 0.5377 ± 0.012 | 0.5788 ± 0.0005 | +0.0014 ± 0.0005 |
| sub-25 | 0.6351 ± 0.012 | 0.5753 ± 0.0005 | -0.0021 ± 0.0005 |
| sub-26 | 0.5000 ± 0.012 | 0.5801 ± 0.0005 | +0.0028 ± 0.0005 |
| sub-27 | 0.5916 ± 0.012 | 0.5768 ± 0.0005 | -0.0005 ± 0.0005 |
| sub-28 | 0.4665 ± 0.012 | 0.5813 ± 0.0005 | +0.0040 ± 0.0005 |
| sub-29 | 0.6016 ± 0.012 | 0.5765 ± 0.0005 | -0.0009 ± 0.0005 |
| sub-30 | 0.5078 ± 0.012 | 0.5798 ± 0.0005 | +0.0025 ± 0.0005 |

Full-sample mean = 0.5773 ± 0.0005. No single exclusion drops the LOO mean below the pre-registered null CI upper bound (0.5172 ± 0.0005).

**Provenance (regenerated GATE T6 STEP 5a; per-column tolerance corrected GATE T7 STEP 1d):** every value above is read directly from the committed `results_c3_r2_jackknife.json` (`loo_results` array; `full_mean_recomputed` and `null_95ci_upper_bound_used` for the two summary figures), the same source `PAPER/figures/scripts/build_latex.py` now reads for the embedded copy of this table in `draft_elsevier.tex`/`draft_ieee.tex`, so the two cannot independently drift again. The three numeric columns carry three different tolerances, not one blanket value: per-fold accuracy (a single 85-trial fold) needs ±0.012 to contain a one-trial-flip excursion; LOO mean and shift (both effectively damped by averaging over 28 folds) are covered by ±0.0005: confirmed by three-point containment on all 87 cells, GATE T7 STEP 1c. A shift value that rounds to zero at 4dp but is actually a small negative number is rendered as 0.0000, not -0.0000.
