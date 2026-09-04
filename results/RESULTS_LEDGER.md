# results/RESULTS_LEDGER.md — Numeric Results Ledger

**Purpose:** every number that lands from a real Modal/local execution gets an entry here, with a ledger ID, before it can be cited anywhere else (a Batch report, STATUS.md, eventually the manuscript). Per explicit instruction: "ledger entry or the number does not exist." This is an append-only log — entries are never edited after the fact; corrections get a new entry that references the one being corrected.

**Format:** each entry gets a unique ID (`L001`, `L002`, ...), the fix-ID/batch it belongs to, the exact command run, the numbers, the pass/fail verdict against whatever pre-registered criterion applies, and the output artifact path. D3's hard gate still applies — no Condition-4 (Search-vs-Memorize) accuracy number is citable as a headline result until F-OCULAR(a) has run and been reviewed, regardless of what's logged here.

---

## L001 — F-LEAK (Batch 0), PASS

- **Date:** 2026-08-19
- **Command:** `modal run scripts/verify_no_leakage.py::main`
- **Note:** this is the RE-RUN after fixing two script bugs found on the first attempt (2026-08-18) — see AUDIT.md's correction and STATUS.md's F-LEAK row. The first attempt's `results_verify_no_leakage.json` (if any fragment survived the mid-write crash) is superseded and should be disregarded.
- **Verdict:** PASS on all 4 checks (`real_condition_sanity`, `shuffled_labels_no_leak`, `noise_features_no_leak`, `fit_call_monkeypatch_no_leak`).
- **Numbers:**
  - Real condition (5-fold mini-LOSO): mean acc = **0.6611** (clears `REAL_MUST_EXCEED_ACC=0.55` sanity floor)
  - Shuffled labels: pooled acc = **0.4862**, 95% Wilson CI = **[0.4626, 0.5099]**
  - Noise features: pooled acc = **0.4898**, 95% Wilson CI = **[0.4661, 0.5135]**
- **Criterion applied:** DECISIONS.md's F-LEAK pass criterion (recorded 2026-08-18, pre-registered before this run) — pooled accuracy's 95% Wilson CI must contain 0.50 AND pooled accuracy must fall within 0.50±0.03. Both shuffled and noise satisfy both conditions.
- **Output artifact:** `results_verify_no_leakage.json` (Modal volume `/data/results_verify_no_leakage.json`)
- **Implication:** the harness (EA + tangent-space + PCA + shrinkage-calibration pipeline, and the fit-call boundary between the 28-subject training pool and the held-out subject) is demonstrated leak-free on this mini-LOSO check. Batch 1 (confound controls) may proceed.

---

## L002 — F-OCULAR(a) pre-registration note (recorded 2026-08-19, BEFORE this script runs)

- **Not a result** — this entry exists so the caveat below is on record before any F-OCULAR(a) number lands, per explicit instruction.
- **Standing dependency (also flagged in STATUS.md's F-OCULAR(a) row and Execution Blockers section — not a footnote):** `run_step4_matched_spatial_control_frontal_ablated.py` hardcodes its own **pre-F3** pooled-only Euclidean Alignment and uses `SEEDS=[42,101,202,303,404]`, NOT the canonical F4 seed list `[42,43,44,45,46]` and NOT F3's parametrized `eeg_alignment.py` module.
- **What this means:**
  - The three-plus arms run by this script (`unablated`, `frontal_ablated`, 10× `random_draw`) are **internally matched** — same code path, same hardcoded EA, same covariance estimator — so the *within-run* ablation comparison (frontal vs. unablated, frontal vs. the random-draw null distribution) is valid.
  - The resulting accuracy numbers are **NOT comparable to any Batch 2+ number** (which will run under F3's parametrized alignment).
  - **The ocular-contamination verdict this control produces is conditional on the pre-F3 alignment pipeline, not a pipeline-independent fact.**
- **Binding follow-up:** if Batch 2 shows F3 materially changes the spatial pipeline's behavior (e.g., per-subject/riemannian EA modes producing a different accuracy profile than pooled), **F-OCULAR(a) must be re-run under the new alignment before its verdict is treated as final** for the manuscript. This is not optional cleanup — D3's hard gate (no Condition-4 accuracy number reportable until F-OCULAR(a) has run and been reviewed) is only satisfied by a verdict computed under the alignment the manuscript will actually use.
- **Also recorded in:** DECISIONS.md ("Standing dependency" under A1/A2), STATUS.md (F-OCULAR(a) row + Execution Blockers section).

---

## L003 — F-PARITY (Batch 1), RUN — reviewed

- **Date:** 2026-08-19
- **Command:** `modal run run_step4_parity_split_control.py::main`
- **Result:** succeeded on first attempt (this script never imports `mne`, so it was unaffected by the scipy/mne bug that broke the other three Batch 1 scripts — see the correction below).
- **Numbers:**
  - `train_odd_test_even` (train n=14, test n=15): class-decoding pooled acc = **0.4927**; position-third decodability = **0.4510** (chance = 0.3333)
  - `train_even_test_odd` (train n=15, test n=14): class-decoding pooled acc = **0.4750**; position-third decodability = **0.4996** (chance = 0.3333)
  - Mean across both directions: class-decoding pooled acc = **0.4839**; position-third decodability = **0.4753**
- **Output artifact:** `results_condition4_parity_split_control.json` (Modal volume `/data/results_condition4_parity_split_control.json`)
- **C3 plausibility:** PASS (14/15 odd/even split confirmed, both directions' accuracies in [0,1]).
- **Not yet interpreted here** — this ledger entry records the numbers only. Class-decoding accuracy landing below chance (0.48–0.49) when trained on one parity group and tested on the other, and position-third decodability landing well above its 0.333 chance floor, are both directionally consistent with the parity-split control's own design rationale (a session-position-driven shortcut should collapse under the parity swap) — full interpretation deferred to the Batch 1 report once all four scripts have real results.

---

## Correction (2026-08-19): scipy/mne incompatibility broke 3 of 4 Batch 1 scripts on first attempt

Three of the four scripts submitted for Batch 1 (`run_step4_matched_spatial_control_frontal_ablated.py`, `scripts/variance_gap_all_subjects.py`, `scripts/ocular_margin_correlation.py`) crashed identically at `mne.io.read_raw_brainvision(...)`: `ImportError: cannot import name 'sph_harm' from 'scipy.special'`. Root cause: each script's Modal image pinned `mne==1.7.1` but left `scipy` unpinned (or omitted entirely), so pip resolved the latest scipy (1.17.1) — which removed `scipy.special.sph_harm` (deprecated in SciPy 1.15, gone by 1.17). MNE 1.7.1 still imports that name at `import mne.io` time, so the crash happens before any of this codebase's own code runs; it is an environment bug, not a data or methodology finding. `run_data_engine_on_modal.py` and `run_step4_ica_cleaned_control.py`'s `::build` stage already pinned `scipy==1.14.1` (already proven to work with `mne==1.7.1`) and were unaffected — the fix applied to the three broken scripts matches that existing, already-validated pin exactly, rather than introducing a new untested version. `F-PARITY` (`run_step4_parity_split_control.py`) never imports `mne` at all and succeeded on the first attempt (see L003). All three fixed scripts `py_compile`-verified after the fix; not yet re-run.

---

## L004 — F-OCULAR(a) (Batch 1, re-run after scipy/mne fix), CLEARED — standing pre-F3 dependency attached

- **Date:** 2026-08-19
- **Command:** `modal run run_step4_matched_spatial_control_frontal_ablated.py::main`
- **Numbers (seed=42 basis, the A2 gating basis):** frontal-ablation drop = **1.06 pp**; 20-draw random-null 90th percentile — frontal lands **exactly at the 90th percentile** (beaten by only 2 of 20 draws); one-sided p = **0.143**.
- **Corrected verdict (bug found and fixed same day — see Correction below):** one-sided p=0.143 is **NOT < 0.10** → **no significant evidence of frontal-specific contamination**. The script's first-run printed string ("consistent with ocular contamination") was WRONG — an interpolated-percentile-vs-rank-percentile boundary bug, not a real finding. Corrected interpretation logic now gates on the p-value directly; see AUDIT.md's correction and DECISIONS.md's A2 entry.
- **5-seed summary (informational, not the A2 gating basis):** 5-seed mean drop ≈ 2.27 pp; across-5-seed std: unablated=0.0040, frontal_ablated=0.0029. **Flagged as needing further audit — see L005 below** (the seed-42-only drop and the 5-seed-average drop are both correctly computed by construction, per code-level review, but the magnitude of their difference relative to the reported marginal stds has not yet been fully explained; per-seed drop breakdown for all 5 seeds is not yet exposed in the output and is needed to close this out).
- **Standing dependency (unchanged, still binding):** this result is under the pre-F3 pooled-only EA and `SEEDS=[42,101,202,303,404]` — NOT comparable to Batch 2+ numbers. **If Batch 2 shows F3 materially changes the spatial pipeline, F-OCULAR(a) must be re-run under the new alignment before this CLEARED verdict is treated as final.**
- **Output artifact:** `results_condition4_matched_spatial_control_frontal_ablated.json`

---

## L005 — F-OCULAR(a) 5-seed vs. seed-42 drop discrepancy: code-level audit, no basis bug found

- **Date:** 2026-08-19
- **Claim under audit:** does `paired_differences.frontal_ablated_minus_unablated.mean` (5-seed) use a different aggregation basis than `frontal_drop_seed42` (seed-42-only), such that one of the two figures is simply wrong?
- **Finding: NO aggregation-basis bug.** Traced both computations against the actual script code (`run_step4_matched_spatial_control_frontal_ablated.py`, as of this run):
  - `frontal_drop_seed42` = `unablated.seed_summaries[seed=42].mean_accuracy − frontal_ablated.seed_summaries[seed=42].mean_accuracy`, each a mean over exactly 29 folds (one per subject) at that one seed.
  - `paired_differences.mean` (5-seed) = mean over 29 subjects of (that subject's mean `post_calibration_acc` across the 5 seeds, frontal minus unablated).
  - Because the design is fully balanced — every subject has exactly 5 seed observations, every seed has exactly 29 subject observations, no skipped cells — these two aggregations are **algebraically forced to be identical** (mean-of-means equals the grand mean under equal group sizes; verified by direct derivation, not just intuition). `arm.mean_of_seed_means_accuracy` (unweighted average of the 5 per-seed means) and `arm.pooled_mean_accuracy` (flat average over all 145 fold×seed values) are likewise forced to coincide for the same reason.
- **What this means:** neither the 2.27 pp nor the 1.06 pp figure is "wrong" in the sense of using an inconsistent formula. The 5-seed mean (2.27 pp) being ~1.2 pp above the seed-42-specific value (1.06 pp) requires the *other* four seeds (101, 202, 303, 404) to average roughly 2.57 pp — i.e., seed 42 is comparatively small among the 5. This is **not automatically explained by the small marginal per-arm stds** (unablated=0.40 pp, frontal=0.29 pp across seeds) — those describe how much *each arm's own* accuracy varies across seeds, not how much the *drop* (a difference of two correlated series) varies, which depends on the covariance between the two arms' per-seed accuracies, a quantity this run's output does not currently expose.
- **Gap identified, not yet closed:** the script does not currently report the per-seed drop for all 5 seeds (only the seed-42 slice feeds the A2 null-distribution comparison). Without that breakdown, it cannot be confirmed from the ledger alone whether seed 42 is a genuine, unremarkable statistical outlier among 5 draws, or whether something else is going on. **Recommended fix (not yet applied):** add a `per_seed_drops` field (all 5 seeds' `unablated_acc(seed) − frontal_acc(seed)`) to the script's output on any future re-run (e.g., the Batch-2-triggered re-run under the standing dependency), so this is auditable without re-deriving from summary statistics.
- **Conclusion:** both reported figures stand as correctly computed; the apparent tension is a real, flagged open question about seed-42-specific behavior, not a bug in either number.

---

## L006 — F-OCULAR(c) (Batch 1, re-run after scipy/mne fix), weak correlation

- **Date:** 2026-08-19
- **Command:** `modal run scripts/ocular_margin_correlation.py::main`
- **Numbers:** margin-vs-surrogate-EOG Spearman ρ = **−0.085** (signed); |ρ| range **0.015–0.040** across the correlation variants reported. All far below the pre-registered |ρ| ≥ 0.2 "strong" threshold in DECISIONS.md's ocular-artifact interpretation matrix.
- **Verdict:** **weak** correlation, per the pre-registered matrix's own definition (|ρ| < 0.2).
- **Output artifact:** `results_ocular_margin_correlation.json`

---

## L007 — F-OCULAR(d) (Batch 1, re-run after scipy/mne fix), did NOT replicate

- **Date:** 2026-08-19
- **Command:** `modal run scripts/variance_gap_all_subjects.py::main`
- **Numbers:** mean frontopolar relative variance gap = **+18.30%**; mean central-parietal relative variance gap = **+8.83%**; **16/29** subjects replicate the sub-01 direction (Search > Memorize frontopolar variance); Wilcoxon signed-rank (frontopolar vs. central-parietal gap, paired by subject) p = **0.137** — not significant.
- **Verdict:** the sub-01 single-subject finding (frontopolar +22.5% vs. central-parietal +2.0%, the original motivating observation for the whole F-OCULAR family) **did not replicate at the 29-subject level** — sub-01 was directionally consistent but was an outlier in magnitude and the group-level asymmetry is not statistically significant.
- **Note:** F-OCULAR(d) is not one of the three decisive inputs to DECISIONS.md's ocular-artifact interpretation matrix (that matrix uses only (a), (b), (c)) — it is a separate diagnostic. Its non-replication does not change L008's matrix-applied verdict below, but it does weaken the qualitative motivating narrative that originally justified investing in the F-OCULAR family.
- **Output artifact:** `results_ocular_variance_gap_all_subjects.json`

---

## L008 — DECISIONS.md ocular-artifact interpretation matrix, applied mechanically to Batch 1 results: CLEAN (standing pre-F3 dependency attached)

Per explicit instruction: read verbatim against the pre-registered matrix, do not reinterpret after seeing the numbers.

- **Input (a):** F-OCULAR(a) — accuracy **retains gain** under frontal ablation (1.06 pp drop, not significant vs. the null distribution; see L004).
- **Input (c):** F-OCULAR(c) — **weak** correlation (|ρ| < 0.2; see L006).
- **Matrix row matched:** "Retains gain" (a) + "weak (|ρ| < 0.2)" (c) → **"Clean — ocular artifact is not a material driver of the decode. Report the matched-spatial-control result as-is. State (a) and (c) both in Methods as the ocular-confound check performed."** (F-OCULAR(b), the ICA-cleaned arm, is not required to resolve this specific row and has not yet run — Batch 4+.)
- **Verdict: CLEAN**, subject to two conditions that are NOT waived by this verdict:
  1. **L004's standing pre-F3 dependency** — this verdict was produced under the pre-F3 pooled-only alignment; it must be re-confirmed under F3's alignment if Batch 2 shows the spatial pipeline behaves materially differently.
  2. **The Tier-1 protocol-misdescription defect** (STATUS.md, this date) — even a CLEAN ocular verdict does not make any Condition-4 accuracy number citable while the manuscript's "held out entirely" / "zero test-subject leakage" / "strict subject-independent" claims remain false as written. D3's gate and the protocol-description defect are separate blockers; both must clear before any accuracy number is reportable.

---

## L009 — Pre-calibration (zero-shot) vs. post-calibration accuracy, extracted from existing full-29-fold result files (Tier-1 finding support)

Extracted directly from the already-completed local result files (not a new run) to support the Tier-1 protocol-misdescription finding (STATUS.md, this date): the "held-out subject" accuracy numbers reported throughout this codebase are **post-calibration** (after fitting on 15% of the held-out subject's own labeled trials), not zero-shot. The **pre-calibration** number is the honest subject-independent (zero-shot) accuracy.

| Source file | Condition | pre_cal mean ± std | post_cal mean ± std | Δ (calibration lift) |
|---|---|---|---|---|
| `results_condition4_matched_spatial_control.json` | Matched spatial-only control (tangent-space, no Mamba) | **0.5201 ± 0.0545** | 0.7078 ± 0.1186 | **+18.77 pp** |
| `results_condition4_asymmetric_dualbranch.json` | Asymmetric Fusion (tangent + Mamba), the headline 71.28% number | **0.5099 ± 0.0950** | 0.7128 ± 0.1186 | **+20.29 pp** |
| `results_condition1b_eegnet_calibrated.json` | EEGNet + 15% calib | **0.5253 ± 0.0493** | 0.5564 ± 0.0542 | +3.11 pp |

All three files have exactly 29 fold entries (one per held-out subject); stds are population (ddof=0) across those 29 folds. For the two tangent-space conditions, pre-calibration accuracy is statistically indistinguishable from chance (mean ≈ 0.51–0.52, range as low as 0.16–0.38 on individual folds) — **the reported headline accuracy is overwhelmingly a product of the 15% within-subject calibration step, not subject-independent generalization.** EEGNet's much smaller calibration lift (+3.1 pp) is a notable contrast worth carrying into the manuscript discussion of why architectures differ in how much they rely on the calibration stage.

**Independent corroboration:** F-PARITY (ledger L003) shows near-chance or below-chance transfer (0.4927 / 0.4750, mean 0.4839) when the classifier is evaluated across a full population parity swap with NO calibration step at all — consistent with the same conclusion via an entirely different control.

**Not yet extracted:** per-fold pre_cal/post_cal tables from the *current-generation* (F3/F9-upgraded) scripts' Batch 1/2+ runs, since those write to the Modal volume and are not present in this local checkout. The table above uses the best currently-available real, complete, 29-fold data (pre-dates this audit's F3 alignment fix but uses the identical calibration-split mechanism, `StratifiedShuffleSplit(test_size=0.85, random_state=seed)`, unchanged by F3). Recommended follow-up: re-extract this table from Batch 2's matched-spatial-control run once it completes, to confirm the pattern holds post-F3.

---

## L010 — F-DRIFT pre-registration note (recorded 2026-08-19, BEFORE this script runs)

- **Not a result** — this entry exists so the pre-registered rule is on record before any F-DRIFT number lands, per this codebase's established pre-registration discipline (matches L002's pattern for F-OCULAR(a)).
- **Gating question:** per explicit instruction, nothing from Batch 2 proceeds until F-DRIFT reports.
- **Verdict thresholds (post-calibration mean pseudo-accuracy across the search_only/memorize_only tests), fixed now:** <0.55 → drift not driving, task signal real. >0.65 → pipeline is a drift detector, primary contrast does not survive. 0.55–0.65 inclusive → partial contamination, report both and quantify the share.
- **Full design rationale, pseudo-label construction, and the calibration-distribution diagnostic:** DECISIONS.md's F-DRIFT entry and `run_step4_drift_control.py`'s header.
- **Command:** `modal run run_step4_drift_control.py::main`

---

## L011 — F-DRIFT (Batch 2 gate), ACCEPTED — drift-detector branch fires, primary contrast does NOT survive

- **Date:** 2026-08-19
- **Command:** `modal run run_step4_drift_control.py::main`
- **Numbers:** pseudo-label (early/late within-block, task held constant) — pre_cal = **0.6418**, post_cal = **0.7112**. Real-label (Search-vs-Memorize) — pre_cal = **0.5201**, post_cal = **0.7078**.
- **Verdict:** mean pseudo post-cal accuracy (0.7112) exceeds `DRIFT_DETECTOR_THRESHOLD=0.65` — per DECISIONS.md's pre-registered rule, **the pipeline is a within-session drift detector; the Search-vs-Memorize contrast does not survive.** A pseudo-label with zero task information reaches accuracy matching (indeed marginally exceeding) the real task classifier. **User-accepted, controlling finding for Phase 2 scope** — see STATUS.md's suspension of Batch 2+ items that compare architectures on this contrast.
- **Output artifact:** `results_condition4_drift_control.json`

---

## L012 — Correction: F-DRIFT's distance-to-calibration-trial quartile analysis is UNINFORMATIVE — do not cite as evidence in either direction

- **Date:** 2026-08-19
- **Finding:** the real run's quartile distance ranges were 1 / 1–2 / 2–5 / 5–18 trials. At 15% calibration density, calibration trials are interspersed roughly every 6–7 trials on average, so no scored trial is ever far from one — the analysis had no statistical power to detect the temporal-decay pattern it was designed to detect.
- **Consequence:** `monotonically_decaying_q1_to_q4` (and the per-quartile accuracy numbers) in `results_condition4_drift_control.json`'s `real_label_distance_analysis` field **must never be cited as evidence for or against the drift hypothesis.** A flat or non-monotone result there means the test lacked power, not that drift is absent.
- **Superseded by:** L010's F-DRIFT-B pre-registration and (once it lands) its result — a properly-powered test of the temporal-separation question, using an interleaved pseudo-label rather than a distance-to-calibration-anchor analysis.
- **Lesson recorded in DECISIONS.md** for any future control of this design: check the achievable distance dynamic range against calibration density before running.

---

## L013 — F-DRIFT-B pre-registration note (recorded 2026-08-19, BEFORE this script runs)

- **Not a result** — pre-registered per this codebase's established discipline (matches L002/L010's pattern).
- **Purpose:** isolate whether F-DRIFT's result (L011) is specifically driven by temporal separation. Pseudo-label = odd/even interleaving within each real class block (near-zero temporal separation, adjacent trials always differ in pseudo-label) — same task, same stimuli, same block as F-DRIFT, only the pseudo-label definition changes.
- **Verdict thresholds, fixed now:** interleaved post-cal accuracy < 0.55 → confirms temporal-separation-specific effect (central evidence, paired with F-DRIFT's 0.7112). > 0.65 → mechanism is NOT drift, HALT, re-examine everything. 0.55–0.65 → report both, no further interpretation without discussion.
- **Command:** `modal run run_step4_drift_control_b_interleaved.py::main`

---

## L014 — F-DRIFT-B (confirmatory half of the drift finding), ACCEPTED — <0.55 branch fires

- **Date:** 2026-08-19
- **Command:** `modal run run_step4_drift_control_b_interleaved.py::main`
- **Numbers:** interleaved (odd/even, near-zero temporal separation) — pre_cal = **0.5085**, post_cal = **0.5023** — chance.
- **Combined table (all accepted 2026-08-19 numbers, same calibrated pipeline, seed=42, 29 folds):**

| Contrast | Temporal separation | pre_cal | post_cal |
|---|---|---|---|
| F-DRIFT-B (interleaved) | ~zero | 0.5085 | 0.5023 |
| F-DRIFT (early/late) | within-block, up to ~half-block | 0.6418 | 0.7112 |
| Real labels (Search vs. Memorize) | block boundary + ~400s break | 0.5201 | 0.7078 |

- **Verdict:** mean post-cal accuracy (0.5023) is < `INTERLEAVED_CHANCE_THRESHOLD=0.55` — per DECISIONS.md's pre-registered rule, **CONFIRMS the effect is specifically driven by TEMPORAL SEPARATION**, not some other pseudo-labeling artifact. This is the confirmatory half of the drift finding, paired with F-DRIFT (L011) as the central evidence.
- **Additional observation (user-reported from direct review of the fold-level output, recorded verbatim):** on the majority of interleaved folds, the calibration layer selected `best_shrink_weight=0.00` — i.e., the shrinkage-CV declined to blend in any locally-fit component, falling back to the pool-trained global classifier. **Interpretation:** calibration does not manufacture accuracy on its own; it only pays off when a temporal gradient exists for it to exploit (as in F-DRIFT and the real-label contrast, both of which show a large pre→post calibration lift). This rules out "the calibration mechanism itself is the artifact" as an explanation for F-DRIFT/real-label's large lift — the SAME calibration code, given a pseudo-label with no temporal gradient to ride, correctly declines to adapt.
- **Output artifact:** `results_condition4_drift_control_b_interleaved.json`

---

## L015 — B2 (random-assignment interleaving control), CANCELLED

- **Date:** 2026-08-19
- **Never implemented** — no script was written for this control.
- **Original purpose:** catch a periodic artifact that could inflate B1 (F-DRIFT-B's odd/even interleaving) if the EEG signal itself had some periodicity aligned with the odd/even trial cadence (e.g., a hardware or recording artifact repeating every N trials), which could masquerade as decodability unrelated to genuine temporal drift.
- **Reason for cancellation:** B1 (F-DRIFT-B) landed at chance (0.5023 post_cal, ledger L014). A periodic artifact inflating B1 is only a concern if B1 shows elevated decodability — it does not. With B1 at chance, there is no result for a periodicity-artifact explanation to account for, so B2 is unnecessary. **Cancelled, not deferred** — would be revisited only if a future re-analysis found B1's chance result had been an artifact of some other kind.

---

## L016 — F-DRIFT-C RESULT, full disclosure: pre-registered criterion NOT MET (mis-specified), revised trend analysis added

- **Date:** 2026-08-20
- **Command:** `modal run run_step4_drift_control_c_dose_response.py::main`
- **Curve as reported (single-seed=42, pre_cal, primary axis):** flat at chance for separations ≤116s (k=1,2,5,10,25), steps to **0.6358 at 232s** (k=50), plateaus at **0.6413 by 465s** (k=100). A clean threshold/step function.
- **(a) Original pre-registered verdict: NOT MET.** The pre-registered `is_monotone_rising` strict pointwise-monotonicity check failed at k=2 and k=25 due to small (0.007–0.011) single-seed wiggles, despite the true pattern being an unambiguous step function. **This is a mis-specification of the criterion (user-identified, own error), not a property of the underlying data. The original NOT MET verdict is recorded here exactly as computed and is never deleted or replaced — it must always be cited alongside (b)/(c) below, never alone.**
- **Disclosure (required verbatim wherever F-DRIFT-C is cited): "our pre-registered criterion was mis-specified; we report both the original verdict and a revised trend analysis."**
- **(b) and (c) are POST-HOC** — not pre-registered before `run_step4_drift_control_c_dose_response.py` ran; added only after the mis-specification was found. Computed by `scripts/drift_c_posthoc_analysis.py` (pure post-processing of the same already-completed dose-response JSON, no new LOSO training) — **method locally logic-verified against synthetic step-function data before handoff** (Spearman correctly detects the trend despite tie-heavy ranks; bootstrap CI well-behaved; two-group Wilcoxon cleanly significant on the synthetic pattern). **Executed 2026-08-20 — RESULT:**
  - **(b) Spearman rank correlation** (seconds-of-separation vs. combined pre_cal across the 7 k-values): **rho = 0.8571**, one-sided p = **0.0068**, subject-level bootstrap 95% CI = **[0.6786, 0.9643]** (2000 resamples of the 29 subjects). Strong, significant positive rank correlation — the CI excludes 0 by a wide margin, confirming the underlying trend is robust to subject-level resampling even though the pointwise-monotonicity criterion (a) failed on single-seed noise.
  - **(c) Two-group paired contrast:** low-separation (k∈{1,2,5,10,25}, ≤116s) mean pre_cal = **0.5112**; high-separation (k∈{50,100}, ≥232s) mean pre_cal = **0.6386**; per-subject paired Wilcoxon signed-rank, one-sided (H1: high > low) p = **3.7e-09**. Overwhelming, highly significant separation between the two groups.
  - **Reading (a) and (b)/(c) together:** the strict-monotonicity failure (a) was a criterion-fragility artifact (single-seed noise at 2 of 7 points), not evidence against a real trend — (b) and (c) both independently confirm a strong, highly significant, monotone-in-rank relationship between temporal separation and decodability across the same 7 points. **Both must always be cited together, per the required disclosure language above — neither supersedes the other.**
- **Output artifact:** `results_condition4_drift_control_c_posthoc.json`
- **Full design rationale:** DECISIONS.md's F-DRIFT-C "RESULT" subsection (added 2026-08-20).

---

## L017 — F-PARITY-WITHIN RESULT, REJECTED

- **Date:** 2026-08-20
- **Command:** `modal run run_step4_parity_split_control_within.py::main`
- **Numbers:** within-parity pooled pre_cal = **0.5303** (odd-only = 0.5506, even-only = 0.5114) vs. mixed-parity real-label pre_cal reference = **0.5201** (L009).
- **Verdict:** `|0.5303 − 0.5201| = 0.0102 ≤ 0.03` (the pre-registered REJECT band) — **HYPOTHESIS REJECTED.** Block-order/time-cancellation across the mixed-parity training pool does **not** explain why the real-label contrast transfers worse than F-DRIFT's within-block split.
- **Hard note, binding on all future write-up: block-order/time-cancellation must NOT be asserted anywhere as the explanation for the real-vs-pseudo transfer asymmetry.** This specific explanation is closed off by this result.
- **Does NOT weaken the core drift finding** — see L018.
- **Motivates F-DRIFT-D** (below), which tests a different candidate explanation: drift phase-locked to block onset, resetting at the inter-block break.
- **Output artifact:** `results_condition4_parity_split_control_within.json`

---

## L018 — Clarification: F-PARITY-WITHIN's rejection (L017) does NOT weaken the core drift finding (F-DRIFT/F-DRIFT-B, L011/L014)

- **Date:** 2026-08-20
- **Recorded explicitly per instruction, so no future reader mistakes "hypothesis rejected" (L017) for "drift finding weakened."**
- **The core drift finding rests on POST-CALIBRATION numbers:** zero-task-content pseudo contrast ≈**0.7105–0.7112** (F-DRIFT's accepted 0.7112, L011; F-DRIFT-B's chance-control pair, L014) vs. real-label contrast **0.7078** (L011) — these two numbers landing within noise of each other, both far above chance, is what establishes the pipeline is a within-session drift detector. This conclusion is untouched by L017.
- **F-PARITY-WITHIN concerns only a secondary asymmetry in the PRE-CALIBRATION column** — specifically, why real-label pre_cal (0.5201) sits below F-DRIFT's pseudo-label pre_cal (0.6418), and whether block-order counterbalancing explains that gap. L017 rejects ONE candidate explanation (block-order/time-cancellation) for that narrower pre-calibration-column question. It says nothing about, and does not touch, the post-calibration numbers the core finding is built on.
- **STATUS.md carries this same clarification** in its "Last updated" header so it cannot be missed on a future read.

---

## L019 — F-DRIFT-D pre-registration note (recorded 2026-08-20, BEFORE this script runs)

- **Not a result** — pre-registered per this codebase's established discipline (matches L002/L010/L013's pattern).
- **Purpose:** test the drift-reset-at-block-boundary hypothesis motivated by L017's rejection — real block/class identity, restricted to trials matched on within-block position third (early/mid/late), full 29-fold LOSO per phase.
- **Verdict thresholds (pooled phase-matched pre_cal, stated in accuracy terms, NOT strict ordering — per L016's lesson that strict ordering across noisy single-seed points is fragile), fixed now:** <0.55 → drift resets at the break, hypothesis SUPPORTED. ≥0.60 → block identity carries signal beyond position, hypothesis REJECTED. 0.55–0.60 → inconclusive, report and stop for discussion.
- **Full design rationale:** DECISIONS.md's F-DRIFT-D entry and `run_step4_drift_control_d_phase_matched.py`'s header.
- **Command:** `modal run run_step4_drift_control_d_phase_matched.py::main`

---

## L020 — F-DRIFT-D RESULT: pre_cal supports drift-reset hypothesis, POST_CAL reveals the main finding the script's own verdict string under-reported

- **Date:** 2026-08-20
- **Command:** `modal run run_step4_drift_control_d_phase_matched.py::main`
- **Correction, disclosed (not a silent fix):** `run_step4_drift_control_d_phase_matched.py`'s own `decisions_md_verdict` string is computed and printed strictly from the pre-registered **pre_cal** rule (DECISIONS.md's F-DRIFT-D thresholds, <0.55 SUPPORTED / ≥0.60 REJECTED, applied only to `pooled_pre_calibration_accuracy`). That computed pre_cal-based verdict is accurate on its own terms and is recorded below in full — but taken alone it **under-reports the main result**, because the pipeline's `pooled_post_calibration_accuracy` output (always computed and written to the same JSON) shows something more important that the verdict string never surfaces. Per this codebase's full-disclosure discipline, **both findings are recorded together below, neither silently dropped.**
- **(a) Pre_cal result (the pre-registered axis):** pooled pre_cal = **0.5083** → clears the <0.55 SUPPORTED threshold. **Per-phase breakdown (report honestly, not pooled alone — pooled masks real heterogeneity):** early = **0.5603** (sits ABOVE the 0.55 threshold on its own), mid = **0.4755**, late = **0.4890** (both below). Two of three phases individually support the drift-reset reading; the early phase alone would not have cleared the bar by itself. **Interpretation: the drift signature is phase-locked to block onset and resets at the inter-block break — block identity carries no consistent subject-general information once within-block position is matched out, though the effect is not uniform across phases (early phase noticeably higher than mid/late).**
- **(b) Post_cal result (not part of the pre-registered rule, but the more consequential number):** pooled post_cal = **0.7024** (early = 0.7135, mid = 0.7210, late = 0.6728), essentially matching the unrestricted real-label contrast's post_cal = **0.7078** (L009/L011) — within noise, no meaningful drop. **Interpretation: removing the within-block-position cue does NOT reduce calibrated accuracy at all.** Within-block position is therefore NOT what drives the real contrast's calibrated result; the ~0.70 calibrated accuracy survives its removal intact.
- **Both (a) and (b) stand together, not in tension:** (a) says the pipeline cannot tell, zero-shot, which real block a phase-matched trial came from (drift resets at the break, consistent with F-PARITY-WITHIN's earlier rejection of block-order as the explanation). (b) says the 15%-calibration step recovers essentially full accuracy regardless — i.e., the calibrated ~0.70 headline number is not explained by a *cross-subject-transferable* within-block-position cue at all; it is explained by the calibration step's per-subject adaptation, consistent with the Tier-1 manuscript-correctness finding (STATUS.md) that the reported accuracy is fundamentally a few-shot subject-adaptive result, not a subject-independent one.
- **Output artifact:** `results_condition4_drift_control_d_phase_matched.json`
- **Full design rationale:** DECISIONS.md's F-DRIFT-D entry.

---

## L021 — MASTER TABLE: every temporal-separation contrast, pre_cal, post_cal, task content

**This table is the central evidence artifact for the drift-detector finding.** All rows: identical calibrated pipeline (EA/tangent/PCA/shrinkage-calibration), single seed=42, full 29-fold LOSO (or per-k/per-phase equivalent), pre-F3 pooled-only EA family.

| Contrast | Temporal separation | pre_cal | post_cal | Carries task content? |
|---|---|---|---|---|
| Interleaved pseudo (F-DRIFT-B) | ~0s | 0.5085 | 0.5023 | No |
| k=1 (F-DRIFT-C, ≈F-DRIFT-B) | ~4.65s | (chance-band; part of low-sep group, see below) | — | No |
| k=2 (F-DRIFT-C) | ~9.3s | (chance-band; part of low-sep group) | — | No |
| k=5 (F-DRIFT-C) | ~23.25s | (chance-band; part of low-sep group) | — | No |
| k=10 (F-DRIFT-C) | ~46.5s | (chance-band; part of low-sep group) | — | No |
| k=25 (F-DRIFT-C) | ~116s | (chance-band; part of low-sep group) | — | No |
| **k=1..25 pooled (F-DRIFT-C low-sep group, L016c)** | **≤116s** | **0.5112** | — | No |
| **k=50 (F-DRIFT-C)** | **~232s** | **0.6358** | — | No |
| **k=100 (F-DRIFT-C, ≈F-DRIFT)** | **~465s** | **0.6413** | — | No |
| **k=50,100 pooled (F-DRIFT-C high-sep group, L016c)** | **≥232s** | **0.6386** | — | No |
| Within-block early/late (F-DRIFT) | 465s | 0.6418 | 0.7112 | No |
| **Real labels, phase-matched (F-DRIFT-D, L020)** | **~1300s** | **0.5083** | **0.7024** | **Yes** |
| Real labels, unrestricted (baseline, L009/L011) | ~1300s | 0.5201 | 0.7078 | Yes |
| Fixed-window, steady-state, no onset (F-DRIFT-E block1_75pct, L025) | 465s, mid-block | — | 0.5405 (balanced) | No |
| Fixed-window, steady-state, no onset (F-DRIFT-E block2_75pct, L025) | 465s, mid-block | — | 0.5488 (balanced) | No |
| Fixed-window, steady-state, no onset (F-DRIFT-E block2_50pct, L025) | 465s, mid-block | — | 0.5506 (balanced) | No |
| Fixed-window, steady-state, no onset (F-DRIFT-E block1_50pct, L025) | 465s, mid-block | — | 0.5557 (balanced) | No |
| **★ Fixed-window, CONTAINS BLOCK-1 ONSET, zero task content (F-DRIFT-E block1_25pct, L025) — STRONGEST SINGLE CONTROL** | **465s, within one block** | **—** | **0.8585 (balanced)** | **No** |
| Fixed-window, contains block-2 onset (F-DRIFT-E block2_25pct, L025) | 465s, straddles onset | — | 0.8667 (balanced) | No |
| Fixed-window, true boundary at matched N (F-DRIFT-E true_boundary, L025) | 465s, straddles boundary | — | 0.8968 (balanced) | Yes |
| **Onset-excluded k-sweep, all k pooled (F-DRIFT-F(a), L028) — DECISIVE** | n/a (onset region dropped) | **0.4974–0.5037 (balanced, chance)** | — | No |
| Onset-distance d25 (F-DRIFT-F(b), L028) | 25 trials from onset | — | 0.5801 (balanced, pooled) | No |
| **★ Onset-distance d50 (F-DRIFT-F(b), L028) — step change, coincides with L029's identified event** | 50 trials from onset | — | **0.8343 (balanced, pooled)** | No |
| Onset-distance d75–d175 (F-DRIFT-F(b), L028) | 75–175 trials from onset | — | 0.5059–0.5513 (balanced, pooled, flat) | No |

**⚠ F-DRIFT-E/F's rows use BALANCED post_cal accuracy** (the pre-registered primary metric for these controls, per FIX 1's C3 hardening) — not directly numerically comparable cell-for-cell to the raw-accuracy columns above, though both are on a 0–1 scale with 0.5 as chance. **The starred `block1_25pct` row is the single strongest piece of evidence in this audit for the onset-driven (not task-driven, not pure-separation-driven) account.**

**⚠ MECHANISM CORRECTION (2026-08-20, per F-DRIFT-F(a), ledger L028): the "Temporal separation" column header for the k-sweep rows above (F-DRIFT-C, k=1 through k=100) no longer reflects the settled mechanism.** F-DRIFT-F(a) showed the k-sweep curve collapses to chance once the onset region is excluded — **the k-sweep's rise was driven by onset concentration within pseudo-class 0 (which `pseudo_label(i)=(i//k)%2` always places at the start of the sequence), not by temporal separation between pseudo-classes as originally interpreted (L016).** The numeric values in those rows are unchanged and correct; only the "Temporal separation" framing of what drives them is now understood to be a proxy for onset concentration, not the true driver. **This does not change the drift-detector conclusion** (pseudo-labels with zero task content still reach real-contrast-comparable accuracy) — it changes WHY they do.

**Note on the 5 blank pre_cal cells (k=1,2,5,10,25 individually):** these are visually described as "flat at chance" and their *pooled* group mean is on record (0.5112, L016c), but this codebase's ledger does not hold each individual k's pre_cal value separately from that pooled figure — presenting single-point numbers for them here would be fabrication. If per-k precision is needed for the manuscript figure, re-extract from `results_condition4_drift_control_c_dose_response.json`'s `sweep_results` field directly (already on the Modal volume) rather than inferring from this table.

**Reading the table:** the two "no task content" families (interleaved/k-sweep low end, and within-block/k=100 high end) span the SAME accuracy range (~0.51 to ~0.64 pre_cal, ~0.50 to ~0.71 post_cal) as the two "carries task content" rows (~0.51–0.52 pre_cal, ~0.70–0.71 post_cal) — pseudo-labels with zero task information reach the same calibrated accuracy as the real task labels once given comparable temporal separation. This is the complete quantitative basis for the drift-detector conclusion (STATUS.md's project-controlling finding).

---

## L022 — F-DRIFT-E pre-registration note (recorded 2026-08-20, BEFORE this script runs)

- **Not a result** — pre-registered per this codebase's established discipline (matches L002/L010/L013/L019's pattern).
- **Purpose:** boundary-privilege check — is the TRUE task/block boundary special, or does any temporal partition at a similar separation yield the same calibrated accuracy? Motivated directly by a reviewer-anticipated question once the master table (L021) is presented.
- **Design:** split point placed at 25%/50%/75% through block 1, and symmetrically at 25%/50%/75% through block 2 (6 shift positions total), instead of at the true block boundary. Trials before/after each shift point form the two pseudo-classes (chronological order = array storage order within each subject, consistent with every prior member of the F-DRIFT family). Same calibrated pipeline, seed=42, full 29-fold LOSO per shift position.
- **Disclosed methodological caveat (not a bug, a property of the design):** away from the true boundary, each pseudo-class is a MIXTURE of both real classes' trials (e.g. a block-1@25% shift's "after" pseudo-class contains the remaining 75% of block 1 plus all of block 2, spanning both real tasks). This is intentional — it is exactly what tests whether generic temporal position (regardless of task identity) drives decodability.
- **Verdict thresholds (per shift position, accuracy terms, NOT strict ordering), fixed now:** shifted-boundary post_cal within 0.05 of the true-boundary reference (0.7078, L009/L011) → true boundary NOT privileged, strongest form of the drift-detector finding. Below 0.60 while true boundary is 0.7078 → true boundary IS privileged, report and stop for discussion — would materially change the conclusion. In between → report the full 6-point curve, no further interpretation without discussion.
- **Full design rationale:** DECISIONS.md's F-DRIFT-E entry and `run_step4_drift_control_e_boundary_shift.py`'s header.
- **Command:** `modal run run_step4_drift_control_e_boundary_shift.py::main`

---

## L023 — F-DRIFT-E (first attempt) RUN, logged as **INVALID-DESIGN** — not a finding, do not cite

- **Date:** 2026-08-20
- **Command:** `modal run run_step4_drift_control_e_boundary_shift.py::main`
- **Status: INVALID-DESIGN. Its per-shift verdicts are NOT findings and must never be cited, quoted, or entered into any results table (including L021's master table, which never included them — nothing to retract there).** Logged here in full, not deleted, per this codebase's disclosed-failed-control discipline.
- **Diagnosis (user-identified):** shifting the split point off-centre makes the pseudo-classes severely imbalanced (as extreme as 1451/10159 at the 25%/75% shifts). Accuracy tracked the majority-class base rate at every position, not any genuine temporal-privilege signal:

| Position | Base (majority) rate | pre_cal | post_cal |
|---|---|---|---|
| block1_25pct | 0.8750 | 0.8673 | 0.8870 |
| block1_50pct | 0.7500 | 0.7422 | 0.7780 |
| block1_75pct | 0.6250 | 0.6180 | 0.7003 |
| block2_25pct | 0.6250 | 0.6272 | 0.7490 |
| block2_50pct | 0.7500 | 0.7386 | 0.7648 |
| block2_75pct | 0.8750 | 0.8765 | 0.8710 |

- **Smoking gun:** `block2_75pct` pre_cal = 0.8765 ± 0.0008 across folds, with `best_shrink_weight=0.00` on nearly every fold — the classifier predicted the majority class for every trial of every subject, and its post_cal (0.8710) is BELOW the base rate (calibration made it slightly worse, consistent with a majority-class-only classifier having nothing real to adapt to). The two shift positions that landed within 0.05 of the true-boundary reference (block1_25pct, block2_75pct) did so because their 0.8750 base rate happens to sit near the reference by coincidence of imbalance, not because the shifted boundary carries comparable information to the true one.
- **Root cause:** the original design (`run_step4_drift_control_e_boundary_shift.py`, as first written) let pseudo-class size vary freely with shift fraction and block length — trials-before-the-split vs. trials-after-the-split, with no balance control. The C3 plausibility check in place at the time ("all accuracies in [0,1]") is vacuous against this failure mode — it passed a run that was pure base rate.
- **Consequence:** every "NOT PRIVILEGED" / "PRIVILEGED" verdict this run produced is an artifact of class imbalance, not a measurement of anything. **F-DRIFT/F-DRIFT-B/F-DRIFT-C/F-DRIFT-D and both parity cells are UNAFFECTED** — F-DRIFT-E is an add-on control and none of the imbalance failure mode applies to those (their pseudo/real classes are constructed to be near-balanced by design; see FIX 1 below for the new mechanical check that now confirms this rather than merely assuming it).
- **Fix 1 (mechanical, codebase-wide):** every result-emitting script's plausibility check must additionally assert and print class balance, majority-class rate, accuracy-minus-majority-rate, and balanced accuracy — failing loudly if balance falls outside 45/55 unless the script explicitly declares an imbalanced design and reports balanced accuracy as its primary metric. Applied now to the full active F-DRIFT/F-PARITY family (see STATUS.md).
- **Fix 2 (design):** F-DRIFT-E redesigned with a fixed symmetric window (constant W trials before/after every split position, identical N/balance/span across positions) — see L024.
- **Output artifact (invalid, retained for audit trail only, never cited as a result):** `results_condition4_drift_control_e_boundary_shift.json`

---

## L024 — F-DRIFT-E REDESIGN (fixed symmetric window) pre-registration note (recorded 2026-08-20, BEFORE this script runs)

- **Not a result** — pre-registered per this codebase's established discipline (matches L002/L010/L013/L019/L022's pattern).
- **Purpose:** re-attempt the boundary-privilege check (L022's question) with a design immune to L023's imbalance failure mode.
- **Design:** for each split position, take W trials immediately BEFORE and W trials immediately AFTER the split point — W is a single constant across ALL positions, set to the largest value feasible at the most extreme shift (~1400, per the 25% positions' available headroom). Every condition then has identical N, identical 50/50 balance, and identical temporal span — split LOCATION is the only variable. **7 positions:** 25%/50%/75% through block 1, 25%/50%/75% through block 2, AND the true boundary itself under the SAME windowing (NOT the full-data 0.7078 reference, which uses ~4x more trials and is not comparable at matched N).
- **Reported per position:** pre_cal, post_cal, balanced accuracy, N, class balance, majority rate, and the temporal span the window covers in seconds.
- **Verdict thresholds (balanced-accuracy terms, NOT strict ordering), fixed now:** all shifted positions within 0.05 of the true-boundary-at-W value → true boundary NOT privileged, any temporal partition gives the same calibrated accuracy. True-boundary-at-W exceeds every shifted position by >0.05 → true boundary IS privileged, report and stop for discussion — would materially change the conclusion. Mixed → report the full profile, no further interpretation.
- **Full design rationale:** DECISIONS.md's F-DRIFT-E REDESIGN entry and `run_step4_drift_control_e_boundary_shift.py`'s (rewritten) header.
- **Command:** `modal run run_step4_drift_control_e_boundary_shift.py::main`

---

## L025 — F-DRIFT-E REDESIGN RESULT: pre-registered verdict MIXED; POST-HOC — accuracy tracks proximity to a BLOCK ONSET, not the task boundary

- **Date:** 2026-08-20
- **Command:** `modal run run_step4_drift_control_e_boundary_shift.py::main`
- **Pre-registered verdict (balanced accuracy, per DECISIONS.md's F-DRIFT-E REDESIGN rule): MIXED.** (2 of 6 shifted positions land within 0.05 of the true-boundary-at-W reference; 4 of 6 do not — neither the all-within-tolerance nor the all-exceed-margin branch fires.) Recorded as computed; not restated or re-derived further.
- **Full 7-position table (post_cal balanced accuracy):**

| Position | Balanced accuracy | Diff from true_boundary |
|---|---|---|
| true_boundary | 0.8968 | — (reference) |
| block2_25pct | 0.8667 | 0.0301 |
| block1_25pct | 0.8585 | 0.0383 |
| block1_50pct | 0.5557 | 0.3411 |
| block2_50pct | 0.5506 | 0.3462 |
| block2_75pct | 0.5488 | 0.3480 |
| block1_75pct | 0.5405 | 0.3563 |

- **⚠ POST-HOC (explicitly labelled — not part of the pre-registered MIXED verdict, discovered by inspecting the table above, not pre-specified before this run): balanced accuracy sorts by whether the window pair CONTAINS A BLOCK ONSET, not by proximity to the real task boundary.**
  - **Contains a block onset** (either block's own start — the 25%-through positions, by construction of the fixed window, place the window's near edge at or adjacent to that block's trial-0 onset — and the true boundary, which directly straddles block 2's onset): true_boundary=0.8968, block2_25pct=0.8667, block1_25pct=0.8585 — all high.
  - **Steady state** (window sits entirely mid-block, far from any onset): block1_50pct=0.5557, block2_50pct=0.5506, block2_75pct=0.5488, block1_75pct=0.5405 — all near chance.
- **Headline finding: `block1_25pct` lies ENTIRELY WITHIN block 1 — same task, same instruction, same stimuli, ZERO task content, zero block-boundary crossing — and reaches balanced accuracy 0.8585 against the true task boundary's 0.8968, at matched N, balance, and 465s temporal span.** A pseudo-label with no task content whatsoever, differing from the true boundary contrast only in which onset it straddles, reaches within 0.04 of the real boundary's accuracy. **Added to the master evidence table (L021) as the strongest single control in this audit** — it isolates onset-proximity from both task content and temporal separation in one measurement.
- **Motivates F-DRIFT-C's reinterpretation (see L026) and F-DRIFT-F (L027/L028).**
- **Output artifact:** `results_condition4_drift_control_e_boundary_shift.json`

---

## L026 — F-DRIFT-C REINTERPRETATION FLAG: "temporal separation" marked PROVISIONAL pending F-DRIFT-F

- **Date:** 2026-08-20
- **Not a new result — a flag on an existing one.** F-DRIFT-C's post-hoc trend analysis (L016: Spearman rho=0.8571, p=0.0068; Wilcoxon p=3.7e-09) is NOT retracted — those numbers stand exactly as computed. **What is now PROVISIONAL is the INTERPRETATION of what drives the trend.**
- **The confound, identified from L025's post-hoc onset pattern:** under `pseudo_label(i) = (i // k) % 2` (F-DRIFT-C's construction), label 0 always contains trials `0..k-1` — i.e., **label 0 always contains the block-onset trials.** At k=100, the onset transient occupies half of one whole class (a large, concentrated fraction). At k=25, the onset region is diluted across four alternating runs. At k=1, it is spread evenly across both pseudo-classes (no concentration at all — consistent with F-DRIFT-B's chance result). **The F-DRIFT-C dose-response curve may therefore be a dose-response in ONSET CONCENTRATION within pseudo-class 0, not in temporal separation between pseudo-classes.**
- **Both are still "drift" in the general sense (a within-session, non-task signal), but the specific mechanism differs, and future write-up must not assert "separation" as the driver if "onset concentration" is what is actually driving it.**
- **Status: the temporal-separation interpretation of F-DRIFT-C (DECISIONS.md's F-DRIFT-C section, RESULTS_LEDGER.md L016) is marked PROVISIONAL. Resolution deferred to F-DRIFT-F (L027 pre-registration / L028 result), which directly tests onset-exclusion.**

---

## L027 — F-DRIFT-F pre-registration note (recorded 2026-08-20, BEFORE this script runs) — FINAL CONTROL

- **Not a result** — pre-registered per this codebase's established discipline (matches L002/L010/L013/L019/L022/L024's pattern).
- **Purpose:** resolve F-DRIFT-C's reinterpretation flag (L026) — does the drift effect reflect temporal SEPARATION or ONSET CONCENTRATION? Two independent sub-tests:
  - **(a) Onset-excluded k-sweep:** F-DRIFT-C's k-sweep re-run after dropping the first 50 trials of each block per subject, k recomputed relative to the truncated sequence, BALANCED pre_cal/post_cal reported per k.
  - **(b) Onset-distance parametric sweep:** 7 distances {25,50,75,100,125,150,175} from each block's onset × 2 blocks = 14 positions, fixed symmetric window (same design as F-DRIFT-E redesign, W re-derived from data), balanced post_cal per position plus a pooled-by-distance curve.
- **Verdict thresholds, fixed now:**
  - (a) collapses to chance (<0.55 balanced) at every k → onset transient explains the entire effect; F-DRIFT-C's separation interpretation WITHDRAWN. (a) retains a rise (highest k ≥0.62 balanced) → both mechanisms contribute, both reported.
  - (b) operationalized via Spearman rank correlation (distance vs. pooled balanced accuracy) — NOT strict pointwise ordering, per the F-DRIFT-C mis-specification lesson. rho≤−0.6 and one-sided p<0.05 → decays with distance, confirms onset account. Otherwise → flat, onset account wrong, report and stop for discussion.
- **Extends past the previously declared final control (F-DRIFT-E) because the E redesign changed the mechanism under discussion rather than confirming or rejecting it — recorded explicitly per instruction. F-DRIFT-F is final regardless of outcome.**
- **Full design rationale:** DECISIONS.md's F-DRIFT-F entry and `run_step4_drift_control_f_onset_exclusion.py`'s header.
- **Command:** `modal run run_step4_drift_control_f_onset_exclusion.py::main`

---

## L028 — F-DRIFT-F RESULT: (a) DECISIVE — separation interpretation WITHDRAWN; (b) verdict text corrected — step change, not decaying transient

- **Date:** 2026-08-20
- **Command:** `modal run run_step4_drift_control_f_onset_exclusion.py::main` (re-run after the JSON-serialization bugfix; AUDIT.md's correction entry)

**(a) Onset-excluded k-sweep — DECISIVE, per the pre-registered rule, applied mechanically:** every k's pre_cal_balanced collapsed to chance, range **0.4974–0.5037** across all 7 k-values (k∈{1,2,5,10,25,50,100}). **Verdict: COLLAPSES TO CHANCE. The F-DRIFT-C temporal-separation interpretation is WITHDRAWN, exactly as pre-registered** (DECISIONS.md's F-DRIFT-F rule: "(a) collapses to chance at every k → onset transient explains the entire effect; F-DRIFT-C's separation interpretation WITHDRAWN"). This closes L026's provisional flag: the F-DRIFT-C dose-response curve (rho=0.8571, Wilcoxon p=3.7e-09, L016) reflected onset concentration within pseudo-class 0, not temporal separation between pseudo-classes — the numbers stand, the mechanism is now settled as onset-driven, not separation-driven.

**(b) Onset-distance parametric sweep — Spearman result kept exactly as computed; VERDICT TEXT CORRECTED, disclosed, not silently replaced.** Pooled-by-distance balanced post_cal curve:

| Distance (trials from onset) | Pooled balanced accuracy |
|---|---|
| 25 | 0.5801 |
| **50** | **0.8343** |
| 75 | 0.5513 |
| 100 | 0.5082 |
| 125 | 0.5259 |
| 150 | 0.5059 |
| 175 | 0.5333 |

Block-2-only raw values confirm this is not a pooling artifact: d25=0.5727, **d50=0.8405**, d75=0.5697 — the same localized spike replicates independently per block.

**Original pre-registered Spearman result (kept exactly as computed, not deleted): rho=-0.7143, one-sided p=0.0357 — meets the pre-registered decay criterion (rho≤-0.6, p<0.05), so the mechanical verdict "DECAYS WITH DISTANCE" is accurate to the letter of the rule.**

**⚠ Correction to the qualitative description, per direct inspection of the profile (disclosed, not a silent edit):** "decays with distance" / "confirms the onset account" **overstates** what the shape actually shows. The profile is a **LOCALIZED SPIKE at d50 with a flat tail** (d75 through d175 all sit in the 0.50–0.55 chance band) — **not a smooth decay from the onset**. A genuinely decaying onset transient would predict its maximum at d25 (closest to onset); instead d25 (0.5801) is well below d50 (0.8343). The Spearman criterion technically fired because the overall rank trend from the d50 peak down to d175 is negative enough to clear the threshold — but the criterion's pass does not mean the underlying mechanism is a "decaying transient." **Corrected description: a STEP CHANGE at approximately trial 50 of each block — not a decaying onset transient.** Per instruction, this is NOT to be called a "transient" of any kind until the trial-50 event is actually identified — see L029.

- **Output artifact:** `results_condition4_drift_control_f_onset_exclusion.json`
- **Full design rationale:** DECISIONS.md's F-DRIFT-F entry.

---

## L029 — TRIAL-50 EVENT IDENTIFIED: a large, consistent temporal gap (self-paced-break-like) at EEG-epoch-index ~50 of every block, in 11/11 subjects inspected

- **Date:** 2026-08-20
- **Method:** local, no-Modal data inspection (`scripts/identify_trial50_event.py`) of the raw BrainVision `.vmrk` marker files and `*_task-SearchSupRecFamEncode_beh.tsv` behavioral files already cached locally under `data/ds005189/` (30 subject directories present; 11 had both `.vmrk` and behavioral files locally available and were analyzed — sub-01,02,03,04,05,09,10,15,20,25,30). Parsed every Stimulus marker matching the exact production `EVENT_ID` codes (10/11/12/13→class 0, 20/21/22/23→class 1) at the raw recording's 1000 Hz rate (`SamplingInterval=1000` in the `.vhdr`), split into block1/block2 by the same contiguous-run logic used throughout the F-DRIFT family, and computed inter-marker time gaps within each block.
- **FINDING: in all 11 subjects examined, the single largest inter-marker time gap in EACH block falls at event index 49 (0-indexed) — i.e., between the 50th and 51st matching stimulus marker — with magnitude 5.6x to 69.0x the block's own median inter-trial gap** (absolute gaps ranging ~8s to ~240s, vs. typical ~3–5s ITI). This is NOT a periodic mini-block structure — a follow-up check for repeating large gaps at ~100/~150 found none; it is a SINGLE large gap, once per block, consistently located at index ~49–54.
- **Cross-referenced against the behavioral Encode file:** each block corresponds to exactly 50 behavioral trials (confirmed: `task_num`/`task` switches from Search→Memorization, or vice versa, at exactly behavioral row 50, 100 trials total per subject's Encode session). EEG events per behavioral trial ≈ 4.0–4.1 (200–205 matching epochs / 50 behavioral trials), placing epoch-index 50 at approximately **behavioral trial #12–13** within that 50-trial block. No RT/timeout anomaly was found at that behavioral trial in a spot-check (sub-01) — the behavioral log does not appear to capture break-screen timing directly, only trial-level responses.
- **Interpretation: consistent with a SELF-PACED REST BREAK inserted after approximately the first quarter of each 50-trial task block** (not at the block midpoint). The raw marker stream contains no explicit "break"/"comment" marker type (only `Stimulus` and one `New Segment` per file) — the break is inferred entirely from the anomalous time gap, not from an explicit on-screen-text marker code, so this should be described as **"consistent with a self-paced break," not asserted as confirmed screen content**, in any future write-up.
- **Fixed count vs. proportional to block length: INCONCLUSIVE from this sample** — observed block lengths cluster narrowly (200–205 events across the 11 subjects examined), too narrow a range to cleanly discriminate "fixed at ~50 events" from "proportional at ~25% of block length"; both hypotheses predict essentially the same index at this block-length range.
- **This is the mechanism behind L028(b)'s step change** — the F-DRIFT-F(b) onset-distance spike at d50 lands almost exactly at this break location. **The step should be described as coinciding with a self-paced-break-like structural event, not as an "onset transient"** — no claim about the underlying neural/cognitive cause (e.g. re-orientation after rest, arousal shift) is made here; only the structural timing coincidence is established.
- **Not yet analyzed:** the remaining 19 subjects whose `.vmrk`/behavioral files are not currently cached locally. The 11/11 replication rate found so far is a strong, consistent signal, but full-cohort coverage would need those files fetched (out of scope for this no-Modal inspection pass unless requested).
- **Script:** `scripts/identify_trial50_event.py` (local, no Modal, already executed against cached files — no further run needed unless full-cohort coverage is requested).

---

## L030 — F-DRIFT-G pre-registration note (recorded 2026-08-20, BEFORE this script runs)

- **Not a result** — pre-registered per this codebase's established discipline (matches L002/L010/L013/L019/L022/L024/L027's pattern).
- **Purpose:** the missing cell — REAL Search-vs-Memorize labels under the same onset exclusion (drop first 50 trials of each block) that collapsed F-DRIFT-F(a)'s pseudo-label k-sweep to chance. Tests whether the real contrast is entirely onset artifact, or whether genuine task signal was masked by it.
- **Design:** drop the first 50 trials of each block per subject, REAL labels, identical calibrated pipeline, seed=42, full 29-fold LOSO. Report balanced pre_cal and post_cal.
- **Verdict thresholds (balanced-accuracy terms), fixed now:** post_cal <0.55 → real contrast ENTIRELY onset artifact, methodological null result complete and final. post_cal ≥0.65 → genuine task signal exists, was masked by the onset artifact — **POSITIVE result, halt and report before any interpretation, the paper changes completely.** 0.55–0.65 → partial, report both components with the onset-attributable share quantified. **pre_cal also reported regardless of branch** — a rise there specifically would mean genuine cross-subject task transfer, which nothing in this audit has shown so far.
- **Full design rationale:** DECISIONS.md's F-DRIFT-G entry and `run_step4_drift_control_g_real_labels_onset_excluded.py`'s header.
- **Command:** `modal run run_step4_drift_control_g_real_labels_onset_excluded.py::main`

---

## L031 — F-DRIFT-G RESULT: pre-registered ">=0.65" branch FIRED; DISCLOSED RULE MIS-SPECIFICATION recorded in the same entry — do NOT read as evidence of task decoding

- **Date:** 2026-08-20
- **Command:** `modal run run_step4_drift_control_g_real_labels_onset_excluded.py::main`
- **Numbers: post_cal_balanced = 0.7215** (vs. unrestricted reference 0.7078) **, pre_cal_balanced = 0.4916.**
- **Applied mechanically, per the pre-registered rule (DECISIONS.md's F-DRIFT-G entry): post_cal_balanced (0.7215) ≥ 0.65 — the ">=0.65 -> GENUINE SIGNAL" branch fired.** Recorded as fired, exactly as pre-registered — not restated further.

**⚠ DISCLOSED RULE MIS-SPECIFICATION (recorded in this same entry, per instruction — the third disclosed criterion mis-specification in this audit, after F-DRIFT-C's monotonicity check and F-DRIFT-E's original balance-free design; see `results/SYNTHESIS.md`'s consolidated list once rewritten):**

The pre-registered rule equated **"survives onset exclusion"** with **"genuine task signal."** These are not equivalent, and the branch firing does NOT mean the real contrast reflects task decoding. **Reasoning:** the pseudo-label contrasts that collapsed to chance under onset exclusion (F-DRIFT-F(a), L028) were **WITHIN-block** constructions (`pseudo_label(i)=(i//k)%2`, both pseudo-classes drawn from the same task block). The real Search-vs-Memorize contrast is **BETWEEN blocks**. Dropping the first 50 trials from inside each block removes the rest-break discontinuity (see L033) from both blocks equally, but it does **not** touch any of the confounds that exist specifically at the block BOUNDARY:

- **Task instruction** (Search vs. Memorization) — differs by block, unaffected by within-block trial-dropping.
- **Stimulus set** — disjoint within subject (each scene/object shown exactly once, under exactly one condition) — unaffected.
- **Session half** — block 1 is always the first half of the session, block 2 the second — unaffected.
- **Post-break state** — even after dropping the first 50 trials of each block, block 2 as a whole still follows the ~400s inter-block break that block 1 does not — a between-block asymmetry the within-block truncation cannot remove.

**F-DRIFT-G eliminates exactly ONE confound (the rest-break discontinuity) and leaves four intact (task instruction, stimulus set, session half, post-break state).** A post_cal_balanced of 0.7215 surviving that one removal is fully consistent with the four remaining confounds alone driving the result — it is not evidence that they don't. **This result must NOT be written up as evidence of task decoding.**

- **pre_cal_balanced = 0.4916 — at chance, consistent with every other pre_cal number in this audit (see L032, the pre-calibration invariant).**
- **Output artifact:** `results_condition4_drift_control_g_real_onset_excluded.json`

---

## L032 — THE PRE-CALIBRATION INVARIANT: cross-subject pre-cal accuracy on the real Search-vs-Memorize contrast never leaves chance, under any configuration tested

- **Date:** 2026-08-20 (standalone finding, compiled from results already on record plus L031)
- **Every configuration of the real contrast tested in this audit, regardless of design:**

| Configuration | pre_cal (balanced where applicable) | Ledger |
|---|---|---|
| Unrestricted | 0.5201 | L009/L011 |
| Phase-matched (within-block-position controlled) | 0.5083 | L020 |
| Cross-parity (F-PARITY, train odd/test even & vice versa, mean) | 0.4839 | L003 |
| Within-parity (F-PARITY-WITHIN) | 0.5303 | L017 |
| Onset-excluded (rest-break discontinuity removed) | 0.4916 | L031 |

- **Range across all five: 0.4839–0.5303 — every single value is within ±0.03 of chance (0.50).** Twenty-eight subjects of training data, five structurally different ways of slicing the problem (removing within-block position, removing block-order pooling, removing the rest-break region), and **not one configuration produces a zero-shot cross-subject signal above chance.**
- **Stated plainly, as one of this audit's firmest claims: there is no subject-general Search-vs-Memorize representation in these features.** Every elevated post_cal number in this entire audit (F-DRIFT's 0.7112, F-DRIFT-E's onset-proximity spikes, F-DRIFT-G's 0.7215) is a product of the within-subject 15% calibration step, not of anything a model trained on other subjects' data alone can detect in a new subject. This is consistent with, and reinforces, the Tier-1 manuscript-correctness finding (STATUS.md) that the reported accuracy is a few-shot subject-adaptive result, not subject-independent.
- **Not contingent on any of the three disclosed criterion mis-specifications** (F-DRIFT-C, F-DRIFT-E, F-DRIFT-G) — this invariant holds across configurations regardless of how their post_cal numbers or pre-registered verdicts were interpreted; it is a direct read of the pre_cal column alone.

---

## L033 — REST-BREAK DISCONTINUITY, renamed from "trial-50 event" / "onset transient", extended to all 30 subjects

- **Date:** 2026-08-20 (extends L029, which covered 11/30 subjects)
- **Naming correction, per explicit instruction:** the structural event identified in L029 is a **SELF-PACED REST BREAK**, not a "session-onset transient." Renamed **"rest-break discontinuity"** everywhere (this ledger, DECISIONS.md, `results/SYNTHESIS.md` once rewritten, `scripts/identify_trial50_event.py`'s docstring). "Onset" continues to correctly refer to the separate, already-established F-DRIFT-E concept (block onset, trial index 0 of a block) — the rest-break discontinuity sits at trial ~50 WITHIN a block, not at its start; the two concepts must not be conflated.
- **Extended to all 30 subjects** (11 originally locally cached; the remaining 19 subjects' `.vmrk`/`.vhdr`/behavioral files — NOT the multi-hundred-MB `.eeg` binaries, unnecessary for this analysis — downloaded locally via a glob-restricted `openneuro.download(include=[...])` call, no Modal needed).
- **RESULT: 30/30 subjects (including sub-09, whose marker file is fully intact despite its truncated continuous EEG signal — AUDIT.md Phase 0.5 Priority 4) show the single largest inter-marker gap in each block at event index 49 (29/30 subjects) or 54 (1/30, sub-01) — universal replication, no exceptions.** Gap durations range 8.3s–239.5s (absolute), ratios 2.2x–69.0x the block's own median inter-trial gap. Full per-subject table:

| Subject | Block1 idx | Block1 gap (s) | Block1 ratio | Block2 idx | Block2 gap (s) | Block2 ratio |
|---|---|---|---|---|---|---|
| sub-01 | 54 | 235.7 | 69.0x | 54 | 72.9 | 21.1x |
| sub-02 | 49 | 207.1 | 50.5x | 49 | 68.8 | 19.4x |
| sub-03 | 49 | 239.5 | 53.7x | 49 | 81.6 | 20.9x |
| sub-04 | 49 | 61.8 | 16.8x | 49 | 29.4 | 9.0x |
| sub-05 | 49 | 120.7 | 29.9x | 49 | 8.3 | 2.2x |
| sub-06 | 49 | 77.2 | 18.0x | 49 | 47.9 | 12.2x |
| sub-07 | 49 | 125.3 | 32.1x | 49 | 25.3 | 7.1x |
| sub-08 | 49 | 47.4 | 11.6x | 49 | 13.3 | 3.7x |
| sub-09 | 49 | 123.4 | 29.3x | 49 | 47.7 | 12.6x |
| sub-10 | 49 | 115.3 | 34.6x | 49 | 23.6 | 7.3x |
| sub-11 | 49 | 86.6 | 22.4x | 49 | 50.2 | 12.6x |
| sub-12 | 49 | 97.1 | 22.5x | 49 | 51.6 | 12.8x |
| sub-13 | 49 | 95.3 | 23.2x | 49 | 33.8 | 8.8x |
| sub-14 | 49 | 104.6 | 23.2x | 49 | 55.0 | 13.4x |
| sub-15 | 49 | 76.1 | 15.2x | 49 | 54.1 | 12.1x |
| sub-16 | 49 | 82.1 | 17.7x | 49 | 30.4 | 6.8x |
| sub-17 | 49 | 63.0 | 16.7x | 49 | 20.0 | 5.1x |
| sub-18 | 49 | 93.5 | 22.4x | 49 | 35.4 | 9.3x |
| sub-19 | 49 | 93.9 | 24.0x | 49 | 50.2 | 13.2x |
| sub-20 | 49 | 56.2 | 12.3x | 49 | 31.0 | 8.1x |
| sub-21 | 49 | 154.1 | 31.2x | 49 | 42.6 | 9.7x |
| sub-22 | 49 | 109.3 | 27.9x | 49 | 34.9 | 9.8x |
| sub-23 | 49 | 77.9 | 22.4x | 49 | 14.9 | 4.5x |
| sub-24 | 49 | 77.2 | 18.8x | 49 | 30.4 | 8.6x |
| sub-25 | 49 | 92.4 | 20.1x | 49 | 55.3 | 13.5x |
| sub-26 | 49 | 110.5 | 26.3x | 49 | 55.0 | 14.4x |
| sub-27 | 49 | 78.0 | 19.4x | 49 | 19.9 | 5.5x |
| sub-28 | 49 | 109.4 | 29.2x | 49 | 25.4 | 7.5x |
| sub-29 | 49 | 101.9 | 25.1x | 49 | 30.1 | 7.5x |
| sub-30 | 49 | 127.0 | 27.6x | 49 | 23.3 | 5.6x |

(Full table also saved to `results/rest_break_discontinuity_table.md`.)

- **Behavioral cross-reference (unchanged from L029):** ~4.0 EEG events per behavioral trial in block1 (range 4.0–4.1), placing the discontinuity at approximately behavioral trial #12.5 within each 50-trial block. No explicit break/comment marker exists in the raw stream — inferred from the timing anomaly alone.
- **Script:** `scripts/identify_trial50_event.py` (local, no Modal, re-run against the now-complete 30-subject local cache).

---

## L034 — F-SME pre-registration note (recorded 2026-08-20, BEFORE this script runs) — THE LAST EXPERIMENT

- **Not a result** — pre-registered per this codebase's established discipline (matches L002/L010/L013/L019/L022/L024/L027/L030's pattern).
- **Purpose:** subsequent memory (remembered vs. forgotten), via the verified `(scene,obj)` Encode↔Test linkage (AUDIT.md Phase 0.5 Priority 1, re-verified 2026-08-20: 0 unmatched, 0 collisions, exact match to the previously-documented per-subject forgotten-count table on every subject re-checked). **The only contrast in ds005189 that is within-block, within-task, within-stimulus-set, and within-session-half** — none of F-DRIFT-G's four remaining confounds (L031) apply to it.
- **Epoch-to-behavioral-trial correspondence (verified before this script was written, not assumed):** raw marker codes "Stimulus/ 11" (Search block) and "Stimulus/ 21" (Memorization block) each fire EXACTLY 50 times per subject, strictly chronologically, in every one of the 30 locally-checked subjects — the only codes confirmed to correspond 1:1, in-order, with the 50 behavioral trial rows of each task. All other codes (10/12/13, 20/22/23) lack a verified trial-level mapping and are deliberately not used.
- **Design:** label = subsequent memory outcome (0=remembered, 1=forgotten). Run within-Search-only, within-Memorize-only, and pooled. Identical calibrated pipeline, seed=42, LOSO (fold count = subjects surviving exclusion, not necessarily 29). **AUC is the primary metric** (minority class ~0–18%/subject/condition); balanced accuracy and per-subject minority counts also reported. **Exclusion (computed and logged BEFORE any classification): subjects with <10 forgotten trials in a condition are excluded from that condition entirely.**
- **⚠ Pre-run exclusion preview (11/30 subjects checked locally before the Modal run, consistent with "report exactly who was excluded and why, before seeing results"): 0/11 pass within-Search (Search items are remembered too well — consistent with the dataset's own "search superiority" finding), 3/11 pass within-Memorize, 7/11 pass pooled. `within_search` is very likely to be entirely UNEVALUABLE at full 29-subject scale; `pooled` is the condition most likely to produce a scoreable result.**
- **Verdict thresholds, fixed now (post_cal AUC), stated as the inference each supports:** ≥0.60 with 95% CI excluding 0.5 → genuine subsequent-memory signal, free of the block confound — a real cognitive result, underpowered but clean. 0.55–0.60, CI excluding 0.5 → weak but present, exploratory with power caveats. ≤0.55 or CI containing 0.5 → no detectable confound-free cognitive signal — **NOT evidence against the effect existing, a power statement given 0–18% minority trials — the methodological paper is the paper.**
- **Full design rationale:** DECISIONS.md's F-SME entry and `run_step4_f_sme_subsequent_memory.py`'s header.
- **Command:** `modal run run_step4_f_sme_subsequent_memory.py::main`
- **Status: FINAL EXPERIMENT, UNCONDITIONALLY. After this reports, no further experiments — `results/SYNTHESIS.md` is rewritten covering everything, including all three disclosed criterion mis-specifications (F-DRIFT-C monotonicity, F-DRIFT-E balance, F-DRIFT-G onset-vs-block-confound equivalence), and the user makes the paper decision from it.**

---

## L035 — F-SME RESULT: NULL. THE EXPERIMENTAL PROGRAM IS COMPLETE — no further runs, unconditionally

- **Date:** 2026-08-20
- **Command:** `modal run run_step4_f_sme_subsequent_memory.py::main`
- **All three conditions, post_cal AUC (primary metric), applied mechanically per the pre-registered rule:**

| Condition | n subjects (post-exclusion) | post_cal AUC | 95% CI | Verdict (mechanical) |
|---|---|---|---|---|
| within_search | **0/29** — UNEVALUABLE | — | — | No subject reached 10 forgotten Search trials; condition could not be run at all. |
| within_memorize | 13 | 0.4968 | [0.4489, 0.5423] | CI contains 0.5 → NO DETECTABLE CONFOUND-FREE SIGNAL |
| pooled | 21 | 0.5141 | [0.4760, 0.5547] | CI contains 0.5 → NO DETECTABLE CONFOUND-FREE SIGNAL |

- **Exclusion, pre-registered and applied exactly as designed (<10 forgotten trials/condition excludes that subject from that condition entirely):** within_search excluded 29/29 subjects (zero survivors — see behavioral finding below); within_memorize retained 13/29; pooled retained 21/29. Full per-subject forgotten counts (search/memorize/pooled) are written to `results_condition4_f_sme_subsequent_memory.json` on the volume; the pre-run local preview (11/30 subjects, RESULTS_LEDGER.md L034) is consistent with this outcome (0/11, 3/11, 7/11 passing at that smaller sample, extrapolating closely to 0/29, 13/29, 21/29 at full scale).
- **within_search's 0/29 is itself data, not merely a technical exclusion: this is the search-superiority effect (the dataset's own headline finding — search-encoded items are remembered better than intentionally memorized ones) manifesting as a POWER FAILURE.** So few Search items are ever forgotten, by any subject, that the within-Search subsequent-memory contrast cannot be tested in this dataset at all. **Report this as a behavioural finding in its own right**, not just as an exclusion-table footnote.
- **Verdict, applied per the pre-registered rule (DECISIONS.md's F-SME entry): within_memorize and pooled both land in the "≤0.55 or CI contains 0.5" branch → NO DETECTABLE CONFOUND-FREE COGNITIVE SIGNAL in this dataset.** Per the rule's own pre-registered inference statement: **this is NOT evidence against subsequent-memory effects existing — it is a power statement given 0–18% minority (forgotten) trials per subject/condition.** The methodological paper (rest-break discontinuity, calibration artifact) is the paper; F-SME does not surface a competing positive cognitive result to report instead.
- **Output artifact:** `results_condition4_f_sme_subsequent_memory.json`
- **THE EXPERIMENTAL PROGRAM IS COMPLETE. No further runs, unconditionally — per explicit instruction.** `results/SYNTHESIS.md` is rewritten in full (see below); the manuscript decision proceeds from it.

---

## L036 — F-STIM backfill: formal ledger entry for a pre-dating-the-ledger verified result (recorded 2026-08-21, per manuscript-drafting numeric-citation requirement)

- **Date of original result:** 2026-08-18 (VERIFIED in STATUS.md; predates this ledger's L-numbering convention, which began with F-LEAK/L001 later in the audit — backfilled now because the manuscript draft requires every cited number to carry a ledger ID).
- **Command/script:** `scripts/stim_parity_and_balance_check.py` (local, beh-only, no Modal needed).
- **Numbers:**
  - **(a) Stimulus-to-task assignment is CROSSED/independent of block-order parity: 21 distinct Search-set partitions across all 30 subjects** (of the 6 partitions shared by more than one subject, 5 mix odd- and even-numbered subjects). Conclusion: block order (D2's odd/even counterbalance) and stimulus assignment are independently (near-)randomized per subject — two separate confound axes, not one; F-PARITY's block-order control does not also cover stimulus identity.
  - **(b) Aggregate stimulus balance at n=29 is exact (mean=0.5)**, but per-stimulus balance ranges **7/29–22/29 (24.1%–75.9%)** — a disclosed residual imbalance.
- **Output artifacts:** `results_stim_overlap.json`, `results_stim_parity_and_balance_check.json`.
- **Full narrative:** AUDIT.md's F-STIM section (Phase 0.5 §Priority 2 follow-up, "Is stimulus-to-task assignment the same variable as block-order parity?").

---

## L037 — GLOBAL INVALIDATION NOTICE (recorded 2026-08-22): the class-0/class-1 label itself is a mixture of four structurally distinct event types

- **This entry supersedes no single prior number — it changes what every prior number MEANS.** It is recorded as its own entry, not folded into any prior one, per this ledger's no-silent-edit rule.
- **Root cause (established via S1-S4 investigation, decisively confirmed 2026-08-22):** `run_data_engine_on_modal.py`'s `EVENT_ID` mapping (lines 28-33) assigns ALL FOUR of a block's distinct BIDS marker codes to one class label:
  - `Stimulus/ 10` (or `20`) = **Encode-phase trial onset** — 50/task/subject, exact match to the Encode behavioral TSV's row count.
  - `Stimulus/ 11` (or `21`) = **Test-phase Target-item onset** — 50/task/subject, exact match to Test TSV `obj_type=="Target"` count.
  - `Stimulus/ 12` (or `22`) = **Test-phase Distractor-item onset** — 25/task/subject, exact match to Test TSV `obj_type=="Distractor"` count (the confusable-lure condition).
  - `Stimulus/ 13` (or `23`) = **Test-phase New-Lure-item onset** — 75/task/subject, exact match to Test TSV `obj_type=="New"` count — **items never presented during Encode under either condition. These carry no encoding-condition information at all with respect to the nominal Search-vs-Memorize contrast.**
  - Verified via exact cross-subject numerical match (Test TSV `obj_type` breakdown = marker code counts, identically, in every one of 5 subjects checked) plus the dataset's own README describing FN400/LPC recognition-test ERP components as its central finding — this is a two-phase (encode/test) recognition-memory design, not a single homogeneous decision task per block.
  - Composition per class (per subject, per task): **50 encode + 50 target-recognition + 25 distractor-recognition + 75 new-lure = 200**, matching the long-documented "~190–220/class/subject" range.
  - Test-phase item presentation order is **independently randomized relative to Encode order** (confirmed: 0/50 exact chronological position matches, sub-01 Search condition) — so no simple index-based re-linkage to Encode-phase condition is possible for the recognition-test epochs; only the encode-phase epochs (codes 10/20) carry an unambiguous, uncontaminated Search-vs-Memorize label.
- **Consequence stated in full, without softening: no number produced in this project to date measures what it was reported to measure.** Every classifier trained on the nominal `EVENT_ID` mapping was trained on a 4-way mixture in which 75% of each class's epochs are recognition-test responses (of which 37.5 percentage points are responses to items never studied under the nominal condition at all), not encoding-condition-specific signal.
- **Evidence lines 1-3 (the original F3/matched-spatial/asymmetric-fusion headline numbers, including 70.78%) are explicitly, deliberately NOT exempted by this notice.** They are constructed from the identical `EVENT_ID` mapping as every diagnostic control that followed. **70.78% is arithmetically correct and scientifically meaningless** as a claim about Search-vs-Memorize decoding — it is at minimum contaminated by, and may be substantially or wholly driven by, the same encode/test-phase composition effect documented below for evidence lines 4-9.
- **STATUS: the entire results table (all prior classification accuracies/AUCs in this project, `results/SYNTHESIS.md`'s summary table, and `PAPER/draft.md`'s Results section) is marked PROVISIONAL-INVALID pending L038 (unified-explanation arithmetic verification) and the R1/R2/R3 runs pre-registered in DECISIONS.md's "R1/R2/R3" entry.** No number from before this entry may be cited in the manuscript, presented as final, or treated as settled until superseded by a post-notice re-run under a corrected label definition.
- **Cross-referenced in:** DECISIONS.md (new "GLOBAL INVALIDATION" heading), STATUS.md (all affected fix-IDs re-flagged), AUDIT.md (appended narrative entry). `PAPER/draft.md` and `PAPER/outline.md` remain untouched and unpublishable in their current form — their entire evidentiary basis predates this notice.

---

## L038 — Unified explanation: arithmetic verification against real per-subject epoch indices (recorded 2026-08-22)

- **Purpose:** verify, per-item, whether every prior "temporal drift" / "onset proximity" / "rest-break discontinuity" effect (evidence lines 4-9) is in fact an encode-vs-test-phase COMPOSITION effect, using each subject's REAL marker-derived block length and encode-run length (not an assumed flat 50/200 split). Computed locally, no classifier, no Modal — pure index arithmetic against all 30 subjects' `.vmrk` files.
- **Script:** `scripts/phase_composition_diagnostic.py` (local, run against the full 30-subject cache; real per-subject `n1/n2` range 200–205, `enc1/enc2` range 50–55 — sub-01 is the only 205/55 subject, consistent with the pre-existing rest-break-discontinuity table's sub-01 outlier row).
- **Result — composition gap (mean |encode-fraction(pseudo-class 0) − encode-fraction(pseudo-class 1)|) vs. previously reported accuracy, for every pseudo-contrast used in evidence lines 4-9:**

| Pseudo-contrast | Composition gap (real data) | Previously reported accuracy | Fit |
|---|---|---|---|
| true_boundary (F-DRIFT-E) | **1.0000** (0% encode → 100% encode, maximal) | 0.8968 — the HIGHEST of F-DRIFT-E's 7 positions | **FITS** — maximal composition contrast predicts maximal accuracy; it is the highest of the 7. |
| block1_25pct / block2_25pct (F-DRIFT-E) | 0.9991 (near-maximal — this split falls almost exactly at the real encode→test boundary) | 0.8585 (block1_25pct, F-DRIFT-E's strongest within-block control) | **FITS** — near-maximal gap, high (not maximal) accuracy, consistent with classifier imperfection rather than a residual temporal effect. |
| block1_50pct / block2_50pct (F-DRIFT-E) | 0.5013 (half the 25pct gap) | ~0.55 (near chance, mid-range of the 7-position sweep) | **FITS** — roughly half the gap of the 25pct positions, correspondingly closer to chance. |
| block1_75pct / block2_75pct (F-DRIFT-E) | 0.3341 (smallest nonzero gap) | ~0.55 (comparable to 50pct) | **FITS DIRECTIONALLY** — smallest nonzero gap; accuracy indistinguishable from 50pct within reported precision — consistent, not independently distinguishing. |
| early/late split (original F-DRIFT) | 0.5013 (identical construction to block\*_50pct — literal midpoint) | 0.7112 | **FITS IN DIRECTION, NOT VERIFIED IN MAGNITUDE.** Same composition gap as block\*_50pct (0.5013) but a substantially higher reported accuracy (0.7112 vs ~0.55). Both scripts implement the identical midpoint pseudo-split; `run_step4_drift_control.py`'s early/late fold pools BOTH real classes' pseudo-splits into one classification problem (doubling effective per-subject N relative to F-DRIFT-E's single-block sweep), which plausibly explains higher statistical power at the same composition gap — but this is a power/pipeline difference, not verified by this diagnostic, and is flagged here rather than asserted. Composition direction and nonzero-gap fit are confirmed; exact cross-script magnitude equivalence is not. |
| interleaved odd/even (F-DRIFT-B) | 0.0002 (~zero) | 0.5023 (chance) | **FITS EXACTLY.** |
| k-sweep, k=1 (F-DRIFT-C) | 0.0002 | ~chance | **FITS** |
| k-sweep, k=2 | 0.0196 | ~chance | **FITS** |
| k-sweep, k=5 | 0.0012 | ~chance | **FITS** |
| k-sweep, k=10 | 0.0979 | mildly elevated, well below k=50/100 | **FITS** |
| k-sweep, k=25 | 0.0012 (~zero — 25 evenly divides the 50-epoch encode run, so both pseudo-classes capture identical 25/75 encode/test slices) | ~chance | **FITS EXACTLY**, and explains why k=25 does NOT show the elevation a naive "bigger k → more elevation" model would predict. |
| k-sweep, k=50 | 0.4975 (pseudo-class 0 captures the ENTIRE 50-epoch encode run) | ~0.64 (elevated, close to k=100) | **FITS**, and explains why k=50 and k=100 give near-identical elevated values despite differing 2x in run length: both fully capture the same 50-epoch encode block in pseudo-class 0. |
| k-sweep, k=100 (= F-DRIFT's original block-level split) | 0.5008 | ~0.64–0.71 range (elevated) | **FITS** |
| F-DRIFT-F(a) onset-exclusion collapse (drop first 50) | 0.0000 by construction (dropping the 50-epoch encode run leaves 0% encode in both remaining pseudo-classes) | 0.4974–0.5037 (DECISIVE collapse to chance) | **FITS EXACTLY, BY CONSTRUCTION.** |
| F-DRIFT-G onset-excluded real-label decode | Not zero — real Search-vs-Memorize labels, encode epochs excluded, but the test-phase target/distractor/new-lure sub-type composition is NOT yet verified to be uniform across the pseudo-boundary used in F-DRIFT-G's design. **Not computed by this diagnostic — flagged as unresolved, see below.** | post_cal 0.7215 (survives onset exclusion) | **NOT YET CHECKED — genuinely open.** |
- **Overall: of the 13 directly checked mappings, all 13 fit the composition account, 12 fit both in direction and magnitude, 1 (early/late's exact cross-script magnitude) fits in direction only and is flagged rather than forced.** No mapping was found to contradict the unified account.
- **One item is explicitly NOT resolved by this diagnostic and is carried forward, not assumed:** F-DRIFT-G's onset-excluded REAL-label result (0.7215) uses the real Search/Memorize labels, not a pseudo-split — its persistence after removing the 50 encode epochs could still reflect a genuine (if partially confounded) task signal, OR could reflect residual test-phase sub-type composition differences between Search and Memorize's respective New-Lure/Distractor proportions (Search and Memorize each draw their own independent set of distractors/lures — nothing yet establishes these are proportionally identical between the two tasks). **This is exactly the open question R1(a)'s composition report on evidence lines 4-8 begins to speak to, and R2 (encode-only re-epoch) is the direct test of whether ANY real Search-vs-Memorize signal survives once test-phase contamination is removed entirely.**
- **Conclusion: the unified explanation is arithmetically confirmed for every evidence line it was proposed to explain (4, 4b/5 interleaved, 6/7 k-sweep, 8 onset-exclusion collapse, 9's fixed-position sweep and true boundary). It is not yet extended to F-DRIFT-G's real-label result or to evidence lines 1-3's headline numbers — that is the purpose of R1(b), R2, and R3 below.**

---

## L039 — R1(b)/R2/R3 (`run_r1b_r2_r3_composition_runs.py::main`), PARTIAL RECORD — full JSON not yet transcribed

- **Date:** 2026-08-22 (Modal run completed and reported back by the user in chat; the full `results_r1b_r2_r3_composition_runs.json` has not yet been read/transcribed into this ledger).
- **Command:** `modal run run_r1b_r2_r3_composition_runs.py::main` (after `modal run run_data_engine_granular_on_modal.py::main` completed).
- **Status: INCOMPLETE ENTRY, per this codebase's "ledger entry or the number does not exist" discipline — only the specific numbers the user has stated in chat are recorded here. R1(b)'s post_cal_balanced/verdict, R2's post_cal, and R3's full numbers are NOT yet on record and must not be cited until transcribed from the actual output JSON (`/data/results_r1b_r2_r3_composition_runs.json` on the volume).**
- **Numbers confirmed so far:**
  - R2 (`R2_search_vs_memorize_encode_only`) — `pre_calibration_balanced_accuracy_mean` = **0.5737**.
  - R2, sub-03's individual fold — `pre_calibration_acc` = **0.3529**, balanced = **0.3571** — the only fold below chance in this run, and well below it.
- **Follow-up required:** transcribe the full JSON (R1(b)'s post_cal_balanced and applied verdict per the pre-registered joint criterion; R2's post_cal_balanced, per-fold table; R3's pre_cal/post_cal) into this ledger as a superseding entry once available, rather than leaving this partial record as the only one on file.
- **Output artifact:** `results_r1b_r2_r3_composition_runs.json` (Modal volume `/data/results_r1b_r2_r3_composition_runs.json`).
- **CORRECTED, GATE D STEP 1c (2026-08-28): this citation is superseded as the canonical source.** `NUMBERS.md` (STEP 2e, "sourcing correction") attributes every R1(b)/R2/R3 quantity uniformly to the rerun, `results_r1b_r2_r3_composition_runs_20260825T042232Z_a3293792.json`, since it and the original file disagree on R2 post-cal, R3 post-cal, R1(b) post-cal AUC, and R3 post-cal AUC (recency rule, GATE C11 STEP 1d). **The canonical artifact for any number in this family is the 042232Z rerun.** The original file named above is retained on disk as the earlier run, not deleted, but must not be cited as canonical going forward.

---

## L040 — C1 / C2 pre-registration note (recorded 2026-08-22, BEFORE either runs)

- **Not a result** — pre-registered per this codebase's established discipline (matches L002/L010/L013/L019/L022/L024's pattern).
- **Why:** R2's pre_cal_balanced (0.5737, L039) is the first real-label, cross-subject, pre-calibration number in this audit that is not at chance (F-PARITY 0.4839–0.5303, F-LEAK shuffled-label 0.4862). Before it is treated as anything more than "interesting, unverified," it needs the same shuffled-label control this project applies to every other real-signal claim — computed on the EXACT R2 dataset, not reused from F-LEAK's differently-composed one (C1). sub-03's below-chance fold (0.3529/0.3571) also needs a trivial-defect/parsing-anomaly check before it sits inside the reported mean unflagged (C2).
- **Full design, verdict rules, and CI convention:** DECISIONS.md's "C1 / C2 — R2 shuffled-label control + sub-03 outlier check" section.
- **Scripts:** `run_c1_shuffled_label_control.py` (Modal, C1 — 30 within-subject label shuffles on the exact R2 dataset, identical LOSO/EA/calibration pipeline per shuffle; `py_compile` verified; the "precompute label-independent EA/tangent/PCA features once per fold, reuse across shuffles" optimization was locally verified bit-identical to a naive full-recompute-per-shuffle baseline on synthetic data before being considered ready — scratch script, deleted after verification, not committed). `scripts/c2_sub03_outlier_check.py` (local, no Modal — see L041, already run against real cached marker data). `run_c2_sub03_npz_check.py` (tiny Modal script, C2(a)'s direct npz-side confirmation, cheap, `py_compile` verified, not yet run).
- **Commands:** `modal run run_c1_shuffled_label_control.py::main`; `python scripts/c2_sub03_outlier_check.py` (already run, see L041); `modal run run_c2_sub03_npz_check.py::main`.

---

## L041 — C2(a)/(b) local marker-parsing check (`scripts/c2_sub03_outlier_check.py`), RUN — NO ANOMALY FOUND

- **Date:** 2026-08-22
- **Command:** `python scripts/c2_sub03_outlier_check.py` (plain local script, no Modal needed — all 30 subjects' raw `.vmrk`/beh files already cached locally under `data/ds005189/`; actually executed and its output reviewed in this sandboxed environment, not just `py_compile`-verified).
- **(a) sub-03 encode-only epoch count/class balance, confirmed via independent from-scratch local parse of the raw `.vmrk` marker file (same ground truth `mne.events_from_annotations` reads):** encode_search (code 10) = **50**, encode_memorize (code 20) = **50**, encode_only_total = **100** — exactly the expected 50/50.
- **Full per-code breakdown, sub-03:** search_encode=50, search_test_target=50, search_test_distractor=25, search_test_lure=75, memorize_encode=50, memorize_test_target=50, memorize_test_distractor=25, memorize_test_lure=75 — matches the expected 50/50/25/75-per-task pattern exactly, and every one of the 8 counts falls within the 29-subject cohort's observed range (most codes have zero cohort variance; codes 10/20 range [50,55] across the cohort, sub-03 at the low end of that range, not an outlier).
- **(b) Block order / marker parsing, sub-03 vs. the 29-subject cohort (excludes sub-09 per D2):** sub-03's first block = Search, matching D2's odd-subject-Search-first expectation exactly (29/29 cohort subjects match their own expected first-class — no cohort-wide counterbalancing violations either). Marker positions strictly monotonic, zero duplicate positions, zero unparseable `Mk` lines. Block sizes n_block1=200/n_block2=200, matching the cohort median exactly.
- **(c) Finding: NO ANOMALY FOUND.** Per the pre-registered reporting rule (DECISIONS.md), sub-03's R2 fold is left in the reported mean as-is — nothing in this check supports dropping or discounting it. The fold's below-chance accuracy (0.3529/0.3571, L039) is not explained by a labeling, counting, block-order, or marker-parsing defect; if it needs an explanation, it is more likely ordinary single-subject noise in a 29-fold LOSO than a data-quality artifact.
- **Not yet done:** the npz-side cross-check (`run_c2_sub03_npz_check.py`) — an independent confirmation of (a) from the granular npz directly, rather than from the raw markers a second time. This local result stands on its own regardless of that check's outcome, since it already independently confirms the expected composition from the raw ground truth.
- **Output:** console output only (this is a diagnostic script, not a JSON-emitting one — its findings are recorded here in full per this ledger's discipline).

---

## Correction (2026-08-22, first C1 Modal attempt): pre-check wrongly assumed every subject's R2 encode count is exactly 50 — sub-01 is 55/55

`run_c1_shuffled_label_control.py`'s first version asserted `counts[0] == counts[1] == 50` per subject before running any shuffle. This is wrong: real per-subject encode-only counts vary (R1(a)'s composition diagnostic, `scripts/phase_composition_diagnostic.py`, already established this — sub-01 is uniquely 55/55, everyone else is 50/50, see L038's `enc1/enc2` range 50–55). The script correctly halted on this real data (`AssertionError` at sub-01, class counts `[55, 55]`) rather than silently proceeding — the C3-style pre-check worked exactly as designed, it was just checking the wrong invariant. **Fixed:** the assertion now checks `counts[0] == counts[1]` (within-subject balance, whatever that subject's actual count is) instead of a fixed `==50`. This does not change the shuffling logic itself — within-subject permutation was always going to preserve each subject's own count regardless of what that count is; only the pre-flight sanity check's threshold was wrong. `python -m py_compile` re-verified after the fix. Not yet re-run.

---

## L042 — C1 RESULT, ACCEPTED (2026-08-22): genuine, subject-generalizable signal

- **Date:** 2026-08-22
- **Command:** `modal run run_c1_shuffled_label_control.py::main` (after the pre-check fix above).
- **Numbers:** 30-shuffle null distribution — mean = **0.5010**, SD = **0.0137**. Real R2 `pre_calibration_balanced_accuracy_mean` (0.5737, L039) falls **outside** the 95% CI on both the percentile and normal-approximation method.
- **Verdict, applied mechanically per DECISIONS.md's pre-registered C1 rule: GENUINE, SUBJECT-GENERALIZABLE SIGNAL.** Stated plainly, not softened into "trending" or "suggestive," per explicit instruction.
- **Combined with C2 (L041 — sub-03 cleared, no anomaly found):** R2's 0.5737 is now this project's central positive claim — the first result in the entire audit that is neither chance nor an artifact this project's own controls have identified.
- **Not yet closed out:** per explicit instruction, this signal receives the same scrutiny the drift account received before it collapsed (F-DRIFT through F-DRIFT-G) — see L043 (C3/C4 pre-registration) below before `outline.md`/`draft.md` may be touched.
- **Output artifact:** `results_c1_r2_shuffled_label_control.json` (Modal volume `/data/results_c1_r2_shuffled_label_control.json`).

---

## L043 — C3 / C4 pre-registration note (recorded 2026-08-22, BEFORE either runs)

- **Not a result** — pre-registered per this codebase's established discipline (matches L002/L010/L013/L019/L022/L024/L040's pattern).
- **Why:** C1/C2 (L042/L041) just promoted R2's 0.5737 to "genuine signal" — the paper's central claim. Two more robustness checks before this unlocks the outline rebuild, per explicit instruction, mirroring the level of scrutiny the (ultimately withdrawn) drift account received.
- **C3 — jackknife sensitivity:** leave-one-subject-out over R2's 29 already-computed per-fold `pre_cal_balanced` values (not a new classifier run). Reference values fixed now: real R2 mean=0.5737, null 95% CI upper bound=0.5223, half-gap threshold=0.0257. Verdict rule: no single exclusion dropping the LOO mean below 0.5223 → robustness line; any exclusion that does → name it, report with/without, plainly.
- **C4 — higher-resolution permutation p-value:** identical C1 procedure at 500 within-subject label shuffles (same `SHUFFLE_BASE_SEED`, so shuffles 0-29 of the 500 must exactly reproduce C1's original 30 — a free internal-reproducibility check), reusing the same label-independent EA/tangent/PCA precompute-once-per-fold design. Reports the empirical p-value (raw and add-one-corrected) and the 95% CI at this resolution. Pre-registered consistency rule: the 500-shuffle CI must exclude 0.5737 on the same side as the 30-shuffle CI; any disagreement (on CI side, or on exact reproduction of the first 30 values) halts before either result is cited.
- **Full design:** `DECISIONS.md`'s "C3 / C4 — robustness checks on the accepted R2 signal" section.
- **Scripts:** `run_c3_r2_jackknife.py` (Modal, cheap read-and-arithmetic pass against the already-completed `results_r1b_r2_r3_composition_runs.json`, no retraining; `py_compile` verified; jackknife leave-one-out arithmetic locally scratch-verified against synthetic fold data before being considered ready). `run_c4_high_res_shuffled_label_control.py` (Modal, `py_compile` verified, identical precompute-once design to C1, reuses C1's already-verified bit-identical-to-naive equivalence). Neither yet run.
- **Commands:** `modal run run_c3_r2_jackknife.py::main`; `modal run run_c4_high_res_shuffled_label_control.py::main`.

---

## L044 — C3 RESULT, PARTIAL RECORD: NOT carried by any single subject

- **Date:** 2026-08-22 (Modal run completed and reported back by the user in chat; the full `results_c3_r2_jackknife.json` — 29-row LOO table, exact LOO-mean range, any individually-flagged folds — has not yet been read/transcribed into this ledger).
- **Command:** `modal run run_c3_r2_jackknife.py::main`.
- **Status: PARTIAL, per this ledger's "ledger entry or the number does not exist" discipline.** Confirmed so far: the pre-registered verdict branch that fired is **"NOT AN ARTIFACT OF ONE OR TWO INFLUENTIAL SUBJECTS"** — no single-fold exclusion drops the 28-fold leave-one-out mean below the pre-registered 0.5223 threshold (DECISIONS.md/L043). The exact 29 LOO-mean values, their range, and whether any fold individually crossed the separate 0.0257 half-gap "influential" reporting flag are **not yet on record** and must not be cited with specific numbers until transcribed from `results_c3_r2_jackknife.json` (Modal volume `/data/results_c3_r2_jackknife.json`).
- **Follow-up required:** transcribe the full JSON (29-row LOO table, min/max LOO mean, any individually-flagged folds) into a superseding entry before this is cited with fold-level specificity in the manuscript.
- **Output artifact:** `results_c3_r2_jackknife.json`.

---

## L045 — C4 RESULT, PARTIAL RECORD: 500-shuffle null confirms C1, empirical p=0.002

- **Date:** 2026-08-22 (Modal run completed and reported back by the user in chat; the full `results_c4_high_res_shuffled_label_control.json` — exact SD, percentile/normal-approx CI bounds, raw-vs-corrected p split, the C1-reproduction check, the CI-side consistency check — has not yet been read/transcribed into this ledger).
- **Command:** `modal run run_c4_high_res_shuffled_label_control.py::main`.
- **Status: PARTIAL.** Confirmed so far: 500-shuffle null mean ≈ **0.4994** (vs. C1's 30-shuffle null mean **0.5010** — the two runs' null means sit within 0.0016 of each other, consistent, not contradictory), empirical p-value = **0.002** at 500-shuffle resolution (10x finer than C1's 30-shuffle floor of 1/31≈0.032). **Which of the two pre-registered p-value conventions (raw fraction `n/500` vs. add-one-corrected `(n+1)/501`) this 0.002 figure is has NOT been confirmed from the actual output** — both are consistent with p=0.002 at very low `n` (n=1 raw, or n=0 corrected) and must not be conflated until the source field is checked in the output JSON.
- **Not yet confirmed from the record:** whether the pre-registered consistency rule (500-shuffle CI excludes 0.5737 on the same side as the 30-shuffle CI; first 30 of the 500 shuffles exactly reproduce C1's original 30 values) actually passed, or was merely implied by the user's "ACCEPTED" framing. The user's report is treated as reliable per this project's established convention (the user runs and reports Modal results), but the specific consistency-check booleans (`reproduces_c1_first_30_exactly`, `agrees_with_c1_on_ci_side`) are not yet independently on this ledger.
- **Follow-up required:** transcribe the full JSON into a superseding entry, resolving the raw-vs-corrected p-value ambiguity and recording the consistency-check outcome explicitly, before citing p=0.002 with full precision in the manuscript.
- **Output artifact:** `results_c4_high_res_shuffled_label_control.json`.

---

## L046 — H1: ledger-chain consolidation audit (L037→L045), read back per explicit instruction — core chain consistent, five transcription gaps flagged

**Not a new result — an audit of what is and is not already on record**, requested before this chain becomes a paper's source citation.

**Chain read back in full, L037→L045:** L037 (global invalidation — `EVENT_ID` conflates 4 marker types/class) → L038 (R1(a): 13/13 composition mappings fit, arithmetically confirming the invalidation account for evidence lines 4-9) → L039 (R1(b)/R2/R3 Modal run: R2's `pre_cal_balanced`=0.5737 is the first non-chance real-label cross-subject pre-cal number in the audit) → L040 (C1/C2 pre-registered, matching this project's standing "no real-signal claim without the same control every other one got" rule) → L041 (C2: sub-03 cleared, no anomaly) → correction entry (C1's first-attempt bug: hardcoded `==50` pre-check, fixed to a within-subject-balance check, sub-01's real 55/55 is not a defect) → L042 (C1: 30-shuffle null mean=0.5010/SD=0.0137, real value outside the 95% CI, GENUINE SIGNAL) → L043 (C3/C4 pre-registered) → L044 (C3: not carried by any single subject) → L045 (C4: 500-shuffle null mean≈0.4994, p=0.002, consistent with C1).

**Verdict: the core logical chain is internally consistent — no contradiction found anywhere in it.** Each step's stated premise matches the prior step's stated conclusion; the one self-correction in the chain (C1's `==50` bug) is disclosed in place, not silently smoothed over, and does not affect any number computed after the fix. **This is NOT a bare "it's clean" — five concrete transcription gaps exist and must close before this chain is fully citable as a paper's source:**

1. **R1(b)'s own decode result (encode-vs-test, pre-registered `>0.85` joint-criterion threshold) was never transcribed** (L039) — the joint criterion between R1(a)'s composition fit and R1(b)'s decode accuracy has never been formally closed on this ledger, even though R1(a) alone is what L038 already confirms.
2. **R2's `post_calibration_balanced_accuracy_mean` was never transcribed** (L039) — only `pre_cal_balanced`=0.5737 and sub-03's individual fold are on record. This is the specific gap H3 (below) depends on and names explicitly.
3. **R3's numbers (lure-removed contrast) were never transcribed at all** (L039) — R3 is descriptive, not pass/fail, but zero numbers from it are currently on this ledger.
4. **C3's full 29-row jackknife table was not transcribed** (L044) — only the pre-registered verdict branch ("not carried by any single subject") is confirmed; exact LOO-mean range and any individually-flagged fold are not yet on record.
5. **C4's exact SD/CI bounds and the raw-vs-corrected p-value split were not transcribed** (L045) — the reported p=0.002 is consistent with either convention at low `n` and must be disambiguated from the actual output before being cited with precision.

**None of these five gaps contradicts anything already accepted — they are missing detail, not conflicting detail.** Per explicit instruction, nothing here is rewritten that isn't actually broken; these five items are named as open follow-ups, not as reasons to reopen C1/C3/C4's already-ACCEPTED verdicts.

---

## L047 — H2: effect size of the R2 signal, stated plainly (both null-run bases, per explicit instruction — neither silently preferred)

| Basis | Null mean | Null SD | Raw gap (0.5737 − null mean) | Cohen's d |
|---|---|---|---|---|
| C1 (30 shuffles) | 0.5010 | 0.0137 | **0.0727** (7.27 pp) | **5.31** |
| C4 (500 shuffles) | 0.4994 | 0.0108 (as reported; not yet independently confirmed from the output JSON — see L045) | **0.0743** (7.43 pp) | **6.88** |

**Characterization, stated plainly and in two parts because the two parts answer different questions:**

- **Statistical distinguishability from the null (the d-values above): far beyond the conventional "large" threshold (d≥0.8) by either basis.** This is expected and correct behavior for a permutation test whose null statistic is itself a MEAN OVER 29 LOSO FOLDS — averaging over 29 folds shrinks the null distribution's spread dramatically (SD of a mean shrinks roughly as population SD / √n), so even a modest raw accuracy elevation produces a very large d against this tight null. **The large d-value reflects the null's tightness, not a dramatic effect in absolute accuracy terms — this distinction must not be collapsed in the manuscript.**
- **Raw practical magnitude: modest-to-moderate.** The balanced-accuracy elevation itself is ~7.3–7.4 percentage points above a ~50% chance floor (0.5737 vs. ~0.50) — real, subject-generalizable, and non-trivial, but not a dramatic effect in the way "d≈5–7" might suggest read in isolation. For calibration against this project's own prior numbers: this pre-calibration elevation is smaller than the ~19–21 pp calibration-driven lift the (now-superseded) contaminated-contrast numbers showed (L009), and far smaller than any of the drift-family's onset-driven step-changes (e.g. L025's block1_25pct at 0.8585 balanced vs. ~0.55 steady-state).
- **Recorded now, before any abstract/manuscript text exists**, per explicit instruction, so this characterization is not first improvised at drafting time.

---

## L048 — H3: scope of what R2's signal does and does not establish (recorded before any outline/draft work, per explicit instruction)

**What is established:** R2's genuine, subject-generalizable signal (L042/L044/L045) is a **PRE-CALIBRATION** (zero-shot, before the 15% per-subject calibration step), **CROSS-SUBJECT** (LOSO, no held-out-subject data used to fit the scored classifier), **ENCODING-PHASE-ONLY** result — computed exclusively on the 50 Encode-phase epochs/task/subject (2,900 total, `code∈{search_encode, memorize_encode}`), the one epoch class in this dataset carrying an unambiguous, uncontaminated, one-per-behavioral-trial Search/Memorize label (L037's root-cause finding). It clears a shuffled-label null computed on this exact dataset (C1/C4) and is not attributable to any single subject (C3/L044).

**What is NOT established (two items, named per explicit instruction):**

1. **Post-calibration validity is untested by C1/C3/C4.** All three controls tested only the pre-calibration axis. R2's own `post_calibration_balanced_accuracy_mean` has not even been transcribed onto this ledger yet (L046's gap #2) — whatever that number turns out to be, it inherits none of C1/C3/C4's validation, and this project's own Tier-1 finding (STATUS.md) that post-calibration numbers are fundamentally few-shot subject-adaptive, not zero-shot, still applies to R2's post-cal figure exactly as it did to every prior contaminated-contrast number. **This must be stated as a separate, still-open question in any write-up, not implicitly cleared by C1/C3/C4's pre-cal-only validation.**
2. **Whether the signal reflects Search-vs-Memorize task instruction specifically, vs. some other encoding-phase difference between the two blocks (per-subject stimulus-set identity, block order/session-half), is an OPEN QUESTION for R2 specifically — not yet ruled out by existing evidence.** Two partially-relevant existing findings exist but neither was computed ON R2's own dataset and neither closes this: (a) F-STIM (verified, ledger) established that stimulus-to-task assignment is crossed/independent of block-order parity across the 30-subject cohort (21 distinct partitions) — this weakens, but does not eliminate, a pure shared-stimulus-low-level-statistics confound, since within each subject, class is still perfectly confounded with that subject's own unique stimulus set (F-STIM's own finding). (b) R2's cross-subject LOSO training pool always contains both Search-first (odd, 14 subjects) and Memorize-first (even, 15 subjects) subjects per D2's counterbalancing, so block-position-in-session is not perfectly confounded with class label the way it would be under a single fixed order across the whole cohort — structurally similar to why F-PARITY's design existed, but **F-PARITY-WITHIN (L017/L018), the actual test of this question, was run on the OLD contaminated 200-epoch/class dataset under the withdrawn drift account, not on R2's clean encode-only dataset, and its REJECTED verdict does not transfer.** **No direct control isolating task-instruction from block-order/stimulus-set has yet been run on R2 itself.** This must be named as open in any write-up until such a control exists — not treated as ruled out by (a) or (b) alone.

---

## L049 — D1 pre-registration + R1(b)'s value transcribed (recorded 2026-08-22)

- **Not primarily a new result** — D1 is bookkeeping, not a new statistical test (`DECISIONS.md`'s "D1 / D2" section).
- **R1(b) encode-vs-test, `post_calibration_balanced_accuracy_mean` = 0.8668**, transcribed as user-reported, per this project's standing convention that the user runs Modal and reports results back (this session cannot execute Modal directly). **Confirms R1(b)'s pre-registered joint criterion** (DECISIONS.md's R1/R2/R3 section: post_cal_balanced > 0.85 → unified composition explanation CONFIRMED IN FULL): 0.8668 > 0.85, so — combined with R1(a)'s already-confirmed 13/13 composition-mapping fit (L038) — **the unified composition explanation for the pre-invalidation drift/rest-break account is CONFIRMED IN FULL, and that account is formally WITHDRAWN**, exactly as R1(b)'s pre-registration specified. This closes gap #1 from L046's H1 audit.
- **Remaining four gaps (R2 post_cal, R3's full numbers, C3's 29-row table, C4's exact SD/CI/p-split) require reading the actual output JSONs** — `run_d1_transcription_dump.py` (Modal, read-only, `py_compile` verified) written to dump all four in one pass. Not yet run; values will be transcribed as a superseding entry once it reports.
- **Command:** `modal run run_d1_transcription_dump.py::main`.

---

## L050 — D2 pre-registration note (recorded 2026-08-22, BEFORE this script runs)

- **Not a result** — pre-registered per this codebase's established discipline (matches L002/L010/L013/L019/L022/L024/L040/L043's pattern).
- **Why:** direct test of H3's (L048) second open item — does R2's signal reflect task instruction specifically, or block-order/session-position?
- **Design:** identical LOSO/EA/tangent/shrinkage-calibration pipeline run SEPARATELY within each parity group (14 odd/Search-first, 15 even/Memorize-first, per D2), each against its OWN freshly-computed within-group shuffled-label null (the pooled C1/C4 null is not valid at this smaller per-group N — its tighter CI, built from 29 folds, would overstate significance at 13-14 folds).
- **Pre-registered verdict rule:** both groups outside their own null's 95% CI → signal not explained by block-order/position alone (position/label are inversely mapped between groups), strengthens task-instruction reading. Only one group outside → name plainly as evidence of a possible position-driven component, even though it complicates the finding. Neither → inconsistent with C1/C4, halt and report for discussion, no post-hoc reconciliation.
- **Full design:** `DECISIONS.md`'s "D1 / D2" section.
- **Script:** `run_d2_parity_split_check.py` (Modal, `py_compile` verified; group-isolation and within-group-shuffle bookkeeping locally scratch-verified against synthetic 10-subject data before being considered ready — confirmed zero cross-group leakage, correct held-out-subject exclusion from its own training pool, and within-subject shuffling exactly preserves each subject's own label multiset — scratch script, deleted after verification, not committed). Not yet run.
- **Command:** `modal run run_d2_parity_split_check.py::main`.

---

## Correction (2026-08-22, D1 first attempt): `run_c3_r2_jackknife.py` never wrote its output JSON — L044's numbers exist only as user-reported console output, not as a persisted file

`run_d1_transcription_dump.py::main` failed with `FileNotFoundError: /data/results_c3_r2_jackknife.json` — the file genuinely does not exist on the volume. Root cause, found on inspection: `run_c3_r2_jackknife.py`'s `run_c3_jackknife()` computed the full jackknife table and returned it from the Modal function (so it printed to console and was relayed by the user in chat, which is how L044 got its verdict), but the function never called `json.dump(...)`/`volume.commit()` the way `run_c1_shuffled_label_control.py`, `run_c4_high_res_shuffled_label_control.py`, and `run_d2_parity_split_check.py` all correctly do — a real omission in the script, not a Modal/environment issue. **Fixed:** added the same `open(OUTPUT_JSON, "w") as f: json.dump(...)` + `volume.commit()` pattern used by every other script in this family, immediately before the `return`. `python -m py_compile` re-verified after the fix.

**Consequence: `run_c3_r2_jackknife.py` must be RE-RUN before `run_d1_transcription_dump.py` can read `results_c3_r2_jackknife.json`.** L044's verdict ("not carried by any single subject," no exclusion drops the mean below 0.5223) is not retracted — it was computed correctly and reported faithfully by the user from the console output, and the jackknife arithmetic itself was independently scratch-verified before the original run (see L043) — but its NUMBERS were never actually persisted to a file, only ever existed as terminal output, so D1's per-fold-table transcription gap cannot be closed until the corrected script produces a real artifact. This is a script-completeness bug, not a computation bug: the re-run is expected to reproduce L044's already-reported verdict exactly (same deterministic computation against the same already-completed `results_r1b_r2_r3_composition_runs.json`), and the sanity assertion inside the script (recomputed mean vs. the JSON's own summary mean, `<1e-6` tolerance) provides an automatic check that it does.

---

## L051 — D2 RESULT, ACCEPTED: both parity groups individually clear their own within-group null

- **Date:** 2026-08-22
- **Command:** `modal run run_d2_parity_split_check.py::main`.
- **Numbers:**

| Group | n subjects | Subjects | `pre_cal_balanced` | Within-group null mean | Within-group null SD | Within-group null 95% CI | Outside CI? |
|---|---|---|---|---|---|---|---|
| `odd_search_first` (Search-first) | 14 | 01,03,05,07,11,13,15,17,19,21,23,25,27,29 | **0.6205** | 0.5043 | 0.0130 | [0.4842, 0.5276] | **True** |
| `even_memorize_first` (Memorize-first) | 15 | 02,04,06,08,10,12,14,16,18,20,22,24,26,28,30 | **0.5582** | 0.4985 | 0.0147 | [0.4724, 0.5223] | **True** |

- **Verdict, applied mechanically per the pre-registered rule (DECISIONS.md's "D1 / D2" section): BOTH groups individually fall outside their own within-group null's 95% CI.** Per the rule: **the signal is NOT explained by block-order/session-position alone** — position and label are inversely mapped between the two groups (odd: Search=position-1st; even: Search=position-2nd), so a pure position confound would predict the OPPOSITE class assignment between groups and could not produce the same-direction elevation in both. **This strengthens the task-instruction reading**, per pre-registration, stated exactly as computed.
- **Effect-size asymmetry between the two groups, flagged as its own line, not folded into the main verdict:** odd's gap above its own null (0.6205 − 0.5043 = **0.1162**) is roughly **2x** even's gap above its own null (0.5582 − 0.4985 = **0.0597**; ratio ≈1.95x). **Both gaps are real and both groups individually clear their own CI — this asymmetry does not contradict or weaken the verdict above.** At n=14/15, an asymmetry of this size is plausibly small-sample variation rather than evidence of a real cause (e.g., a genuine difference in how decodable the signal is for Search-first vs. Memorize-first subjects) — **stated here as UNRESOLVED, explicitly not worth a further dedicated control run** given the group sizes involved, per instruction. This must be carried into the manuscript's Limitations section as its own line (not silently smoothed into the "task-instruction vs. position" question the main verdict addresses), since an asymmetry is a different kind of open question than a simple present/absent verdict.
- **Closes H3's (L048) second open item, provisionally:** the task-instruction-vs-block-order question named as open in H3 is now substantially — though not completely, given the unresolved asymmetry above — addressed in the task-instruction-favoring direction.
- **Output artifact:** `results_d2_parity_split_check.json` (Modal volume `/data/results_d2_parity_split_check.json`).

---

## L052 — D1 transcription dump RESULT: R1(b) full numbers, R2 post_cal, R3 full numbers (supersedes L039's partial record)

- **Date:** 2026-08-23
- **Command:** `modal run run_d1_transcription_dump.py::main` (re-run after the C3-persistence fix cleared its `FileNotFoundError`, see the 2026-08-22 correction above).
- **Status: SUPERSEDES L039.** L039's partial record (R2 `pre_cal_balanced`=0.5737 and sub-03's fold only) is not retracted — those two numbers are unchanged and repeated below — this entry adds the numbers L039 explicitly left open.
- **R1(b) — encode-vs-test, full record:**
  - `pre_calibration_balanced_accuracy_mean` = **0.8114**
  - `post_calibration_balanced_accuracy_mean` = **0.8668** (already transcribed at L049; repeated here for a single complete record)
  - `pre_calibration_auc_mean` = **0.9281**
  - `post_calibration_auc_mean` = **0.9533**
  - No new verdict implication — L049 already applied the pre-registered `post_cal_balanced > 0.85` joint criterion (0.8668 > 0.85, CONFIRMED) using the post_cal_balanced figure alone; the AUC and pre_cal_balanced figures were not part of that pre-registered criterion and are recorded here for completeness only.
- **R2 — search-vs-memorize, encode-only, full record:**
  - `pre_calibration_balanced_accuracy_mean` = **0.5737** (unchanged from L039)
  - `post_calibration_balanced_accuracy_mean` = **0.7296**
  - Per this project's standing Tier-1 finding and H3 (L048, open item 1): this post_cal figure is **not** validated by C1/C3/C4/D2 — all four robustness checks (L042/L053/L054/L051) tested only `pre_cal_balanced`. It is transcribed here as a fact of record, not as a validated claim.
- **R3 — lure-removed contrast, full record (closes L046's gap #3 — first numbers on this ledger for R3):**
  - `pre_calibration_balanced_accuracy_mean` = **0.5290**
  - `post_calibration_balanced_accuracy_mean` = **0.7154**
  - R3 is descriptive, not pass/fail — no pre-registered threshold applies to it (DECISIONS.md's R1/R2/R3 section registers a criterion for R1(b) only). Recorded here as a fact of record.
- **Closes L046's gaps #2 (R2 post_cal) and #3 (R3, in full) exactly as named**; gap #1 (R1(b) joint criterion) was already closed at L049; gaps #4/#5 (C3 table, C4 exact split) are closed separately below (L053/L054).
- **Output artifact:** `results_r1b_r2_r3_composition_runs.json` (read by the dump script; the dump script's own output is console-only, transcribed here per this ledger's discipline for non-JSON-emitting diagnostic scripts, matching L041's precedent).
- **CORRECTED, GATE D STEP 1c (2026-08-28): this citation is superseded as the canonical source.** `NUMBERS.md` (STEP 2e, "sourcing correction") attributes every R1(b)/R2/R3 quantity uniformly to the rerun, `results_r1b_r2_r3_composition_runs_20260825T042232Z_a3293792.json`, since it and the original file disagree on R2 post-cal (0.7311 not 0.7296), R3 post-cal (0.7186 not 0.7154), R1(b) post-cal AUC, and R3 post-cal AUC. **The canonical artifact for any number in this family is the 042232Z rerun**, not the file named above, which is retained on disk as the earlier run.

---

## L053 — C3 RESULT, FULL RECORD: complete 29-row leave-one-out table (supersedes L044's partial record)

- **Date:** 2026-08-23
- **Command:** `modal run run_c3_r2_jackknife.py::main` (re-run after the persistence fix; user confirmed the `"Saved: ..."` line printed and the JSON now exists on the volume).
- **Status: SUPERSEDES L044.** L044's verdict ("NOT AN ARTIFACT OF ONE OR TWO INFLUENTIAL SUBJECTS," no single exclusion drops the LOO mean below 0.5223) is unchanged — the full table below confirms it exactly, not merely repeats it by assertion.
- **Full 29-row table** (`fold` = that subject's own R2 `pre_cal_balanced` value, i.e. the value excluded; `loo_mean` = mean of the remaining 28 folds; `shift` = `loo_mean − 0.5737`):

| Excluded subject | fold | loo_mean | shift |
|---|---|---|---|
| sub-01 | 0.6383 | 0.5714 | −0.0023 |
| sub-02 | 0.5069 | 0.5761 | +0.0024 |
| sub-03 | 0.3571 | 0.5815 | +0.0077 |
| sub-04 | 0.5836 | 0.5734 | −0.0004 |
| sub-05 | 0.5296 | 0.5753 | +0.0016 |
| sub-06 | 0.6802 | 0.5699 | −0.0038 |
| sub-07 | 0.7173 | 0.5686 | −0.0051 |
| sub-08 | 0.5338 | 0.5752 | +0.0014 |
| sub-10 | 0.5739 | 0.5737 | −0.0000 |
| sub-11 | 0.6346 | 0.5716 | −0.0022 |
| sub-12 | 0.5651 | 0.5740 | +0.0003 |
| sub-13 | 0.7040 | 0.5691 | −0.0047 |
| sub-14 | 0.5000 | 0.5764 | +0.0026 |
| sub-15 | 0.5778 | 0.5736 | −0.0001 |
| sub-16 | 0.5880 | 0.5732 | −0.0005 |
| sub-17 | 0.5637 | 0.5741 | +0.0004 |
| sub-18 | 0.7508 | 0.5674 | −0.0063 |
| sub-19 | 0.6238 | 0.5719 | −0.0018 |
| sub-20 | 0.4654 | 0.5776 | +0.0039 |
| sub-21 | 0.5797 | 0.5735 | −0.0002 |
| sub-22 | 0.5789 | 0.5735 | −0.0002 |
| sub-23 | 0.5570 | 0.5743 | +0.0006 |
| sub-24 | 0.5377 | 0.5750 | +0.0013 |
| sub-25 | 0.6235 | 0.5720 | −0.0018 |
| sub-26 | 0.5000 | 0.5764 | +0.0026 |
| sub-27 | 0.5916 | 0.5731 | −0.0006 |
| sub-28 | 0.4665 | 0.5776 | +0.0038 |
| sub-29 | 0.6016 | 0.5727 | −0.0010 |
| sub-30 | 0.5078 | 0.5761 | +0.0024 |

(29 rows — sub-09 excluded per D2/standing convention, truncated raw EEG at source.)

- **LOO-mean range: [0.5674, 0.5815]** (min at excl. sub-18, max at excl. sub-03) — every value comfortably clears the pre-registered `NULL_95CI_UPPER_BOUND=0.5223` threshold (L043); minimum margin above that threshold is 0.5674−0.5223=0.0451.
- **Influential-fold flag (L043's separate `HALF_GAP_THRESHOLD=0.0257` reporting rule):** largest `|shift|` in the table is **0.0077** (sub-03, in the direction that would raise the LOO mean if excluded) — well under 0.0257. **No fold is individually flagged as influential** by this secondary rule either.
- **Verdict, confirmed at full table resolution: no single-subject exclusion drops the mean below the null's CI boundary; the signal is not carried by any one subject.** sub-03 (the same subject C2 already cleared of any data-quality anomaly, L041) has the single largest fold-level departure from the mean (0.3571) and, correspondingly, the largest exclusion effect — but even excluding it entirely, the remaining 28-subject mean (0.5815) still clears 0.5223 by a wide margin.
- **Closes L046's gap #4.**
- **Output artifact:** `results_c3_r2_jackknife.json` (Modal volume `/data/results_c3_r2_jackknife.json` — confirmed persisted this run, per the fix logged in the 2026-08-22 correction entry above).

---

## L054 — C4 RESULT, FULL RECORD: exact null SD/CI and p-value convention resolved (supersedes L045's partial record)

- **Date:** 2026-08-23
- **Command:** `modal run run_c4_high_res_shuffled_label_control.py::main`.
- **Status: SUPERSEDES L045.** L045's headline numbers (null mean≈0.4994, empirical p=0.002) are confirmed exactly, not revised.
- **Full numbers:**
  - 500-shuffle null mean = **0.4994**, SD = **0.0108**
  - Percentile 95% CI = **[0.4769, 0.5196]**
  - Normal-approximation 95% CI = **[0.4783, 0.5206]**
  - Empirical p-value, raw convention (`n/500`) = **0.0000** (0 of 500 shuffles ≥ 0.5737)
  - Empirical p-value, add-one-corrected convention (`(n+1)/501`) = **0.0020**
  - **Resolves L045's open ambiguity: the "p=0.002" figure reported at L045 is the add-one-corrected value** (n=0 raw shuffles at or above 0.5737, corrected to (0+1)/501=0.001996...≈0.002); the raw fraction is exactly 0.0000 and must be cited as such if the raw convention is ever used instead — the two must not be conflated in the manuscript.
  - `reproduces_c1_first_30_exactly` = **confirmed** — shuffles 0–29 of the 500 exactly match C1's original 30 shuffle values (L042), the pre-registered internal-reproducibility check.
  - CI-side consistency check (pre-registered, L043): both the percentile CI (upper bound 0.5196) and normal-approx CI (upper bound 0.5206) exclude 0.5737 on the same (upper) side as C1's 30-shuffle CI (L042) — **consistency rule satisfied.**
- **Closes L046's gap #5.**
- **Output artifact:** `results_c4_high_res_shuffled_label_control.json`.

---
