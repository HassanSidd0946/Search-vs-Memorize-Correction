# SYNTHESIS.md — Full audit synthesis, F-LEAK through F-SME

## ⚠ SUPERSEDED BY GLOBAL INVALIDATION NOTICE (2026-08-22) — read `RESULTS_LEDGER.md` L037/L038 before trusting anything below

**Every number in this document, including "the headline conclusion" below, was computed under an `EVENT_ID` mapping now confirmed to conflate four structurally distinct event types per class (50 Encode-phase onsets + 50 Test-phase Target + 25 Test-phase Distractor + 75 Test-phase New-Lure — the last carrying no encoding-condition information at all).** This does not just affect the diagnostic-control family below; evidence lines 1-3's headline numbers (70.78% etc.) are explicitly, deliberately included in the invalidation. The "rest-break discontinuity" mechanism this document treats as established is now understood to be the Encode→Test phase transition within each block, not a self-paced rest break — see `RESULTS_LEDGER.md` L038 for the arithmetic confirmation across every affected evidence line. Full record: `RESULTS_LEDGER.md` L037/L038, `DECISIONS.md`'s GLOBAL INVALIDATION NOTICE and R1/R2/R3 pre-registration, `AUDIT.md`'s matching narrative entry. **This document is not being rewritten until R1/R2/R3 report — treat it as a historical record of the pre-discovery state, not a current claim.**

---

**Written 2026-08-20, after F-SME (the final, unconditional experiment) reported. THE EXPERIMENTAL PROGRAM IS COMPLETE — no further runs. This is the source document for the manuscript decision.**

**Governing documents:** `STATUS.md` (fix-ID state table, authoritative for current state), `DECISIONS.md` (pre-registrations, binding), `results/RESULTS_LEDGER.md` (numeric results, ledger IDs `L001`–`L035`), `AUDIT.md` (chronological narrative log). This synthesis is a full rewrite superseding the 2026-08-20 draft written mid-audit (before F-DRIFT-E's post-hoc onset finding reopened the mechanism question) — if a number here ever conflicts with the ledger, the ledger wins.

**`PAPER/main.tex` has been archived, unrevised, to `PAPER/archive/main_superseded_2026-08.tex`. Phase 3 as originally scoped is CANCELLED. `PAPER/outline.md` (outline only, no prose) is the active manuscript effort, pending user review.**

---

## 1. The headline conclusion

**There is no subject-general Search-vs-Memorize representation in this codebase's EEG features.** Every elevated "accuracy" number reported anywhere in this project's history (the manuscript's 71.28%, the matched-spatial-control's 70.78%, F-DRIFT's 71.12% pseudo-label match, F-DRIFT-G's 72.15%) is produced by a **15%-per-subject calibration step**, not by anything a classifier trained on other subjects can detect zero-shot in a new one. The calibrated numbers themselves are further explained, almost completely, by a **self-paced rest-break discontinuity** occurring at a highly consistent point (~trial 50) within every task block, in all 30 subjects examined — not by task content, and not (once the criterion mis-specifications below are corrected) by raw temporal separation either. A confound-free control (subsequent memory) found no detectable cognitive signal, but is too underpowered to argue the reverse.

This document is a **self-audit of our own analysis pipeline applied to a public dataset** (ds005189: Helbing, Draschkow & Võ, 2025, *JoCN*). Nothing here is a claim about, or a critique of, that dataset's original authors or their own published findings, which this audit does not evaluate.

---

## 2. Foundational verification

### 2.1 F-LEAK — leakage verification (VERIFIED, ledger L001)

Four checks (real-data sanity floor, shuffled labels, noise features, a fit-call monkeypatch asserting no held-out test-subject row ever reaches a `fit()` call) confirmed the harness is leak-free: real=0.6611 mean; shuffled pooled_acc=0.4862 [0.4626,0.5099]; noise pooled_acc=0.4898 [0.4661,0.5135] — both inside the pre-registered chance band (95% Wilson CI must contain 0.50, mean within 0.50±0.03). Two script bugs (over-broad forbidden-row scoping; a numpy-int64 JSON crash) were caught and fixed before this clean pass — not evidence of leakage in production.

### 2.2 Pre_cal / post_cal decomposition — the Tier-1 finding

Every LOSO driver in this codebase holds out a subject, then fits a **shrinkage-blended per-subject calibration classifier on 15% of that subject's own labeled trials** before scoring the remaining 85%. `pre_calibration_acc` = zero-shot (global-pool classifier only). `post_calibration_acc` = after this step.

| Source | pre_cal | post_cal | Lift |
|---|---|---|---|
| Matched spatial-only control | 0.5201±0.0545 | 0.7078±0.1186 | +18.77 pp |
| Asymmetric Fusion (headline 71.28%) | 0.5099±0.0950 | 0.7128±0.1186 | +20.29 pp |
| EEGNet + 15% calib | 0.5253±0.0493 | 0.5564±0.0542 | +3.11 pp |

**The manuscript's protocol description ("held out entirely," "zero test-subject leakage into any stage," "strict subject-independent") is false as written.** The pipeline is legitimately **few-shot subject-adaptive**, not subject-independent — a real, defensible methodology, but not the one described. Not a leakage bug (F-LEAK already cleared this exact boundary); corroborated independently by F-PARITY (§2.4) with no calibration step at all.

### 2.3 F-STIM — stimulus-identity confound (VERIFIED)

Within-subject class is perfectly confounded with stimulus identity (unavoidable; only cross-subject counterbalancing defuses it). Stimulus assignment is crossed/independent of block-order parity (21 distinct partitions across 30 subjects) — a separate axis from F-PARITY. Per-stimulus balance at n=29 ranges 24.1%–75.9% despite exact aggregate balance — a disclosed residual imbalance.

### 2.4 F-OCULAR(a)–(d) — ocular-artifact controls (D3-gated, combined verdict CLEAN, ledger L008)

- **F-OCULAR(a):** 20-draw null-distribution ablation (frontal channels dropped vs. random 9-channel draws). Frontal-ablation drop = 1.06 pp, at the null distribution's 90th percentile, one-sided p=0.143 — no significant frontal-specific contamination. **Standing caveat:** hardcodes pre-F3 pooled-only EA and its own seed list, conditional on that alignment; a seed-42-vs-5-seed basis-mismatch reopening remains blocked on missing data, de-prioritized (the contrast it gates is no longer reported on regardless).
- **F-OCULAR(b):** ICA-cleaned arm, written, never run (decisive control, paired with (c), not reached before the project's scope changed).
- **F-OCULAR(c):** surrogate HEOG/VEOG vs. decision-margin correlation, |ρ| 0.015–0.040, far below the 0.2 "strong" threshold. Weak.
- **F-OCULAR(d):** frontopolar-vs-central-parietal variance gap, extended sub-01→29 subjects. Sub-01's finding (+22.5%/+2.0%) did NOT replicate at scale (+18.30%/+8.83% mean, 16/29 subjects directionally consistent, Wilcoxon p=0.137, not significant).

### 2.5 F-PARITY and F-PARITY-WITHIN — block-order counterbalancing

**F-PARITY (L003):** cross-parity LOSO (train odd/test even, vice versa), no calibration. Pooled acc 0.4927/0.4750 (mean 0.4839) — near-/below-chance. Position-third decodability 0.4510/0.4996 (chance=0.3333) — the classifier reads temporal position better than task identity even here.

**F-PARITY-WITHIN (L017, RUN 2026-08-20):** tested whether mixed-parity training pools cancel a real time-position cue. **Result: within-parity pooled pre_cal=0.5303 — within the pre-registered REJECT band. HYPOTHESIS REJECTED.** Block-order/time-cancellation does not explain why the real contrast transfers worse pre-calibration than F-DRIFT's within-block split. **Does not weaken the core drift finding** (§3), which rests on the post-calibration column; this concerns only the pre-calibration column.

---

## 3. The F-DRIFT family — the project-controlling evidence chain

### 3.1 F-DRIFT (L011) and F-DRIFT-B (L014) — the original gate

**F-DRIFT:** pseudo-label = early/late split within one real class block (zero task-content change), same calibrated pipeline. Pseudo pre_cal=0.6418, post_cal=**0.7112** vs. real pre_cal=0.5201, post_cal=**0.7078**. The pre-registered >0.65 branch fired: **the Search-vs-Memorize contrast does not survive as a task-decoding result.** User-accepted, controlling finding.

**F-DRIFT-B:** interleaved (near-zero separation) pseudo-label. pre_cal=0.5085, post_cal=**0.5023** — chance. The <0.55 branch fired: confirms the effect is specifically temporal, not an artifact of pseudo-labeling per se. Majority of folds selected `shrink=0.00` — calibration doesn't manufacture accuracy from nothing.

**F-DRIFT and F-DRIFT-B together remain the central, unchallenged evidence pair that the reported contrast is not task decoding.** Nothing below changes this — it changes *why*.

### 3.2 F-DRIFT-C — dose-response sweep (L016) — RESULT WITHDRAWN, see §5.1

`pseudo_label(i)=(i//k)%2`, k∈{1,2,5,10,25,50,100}. Curve: flat at chance ≤116s, step to 0.6358 at 232s, plateau at 0.6413 by 465s. Post-hoc Spearman rho=0.8571 (p=0.0068), two-group Wilcoxon p=3.7e-09 — both real, unchanged numbers. **The "temporal separation" interpretation of this curve is WITHDRAWN** by F-DRIFT-F(a) (§3.6) — see the mis-specification section (§5.1) for the full account.

### 3.3 F-DRIFT-D — phase-matched block contrast (L020)

Real block label, restricted to matched within-block position (early/mid/late thirds). **(a) pre_cal pooled=0.5083 → SUPPORTS the drift-resets-at-the-break hypothesis** (heterogeneous: early=0.5603 above threshold, mid=0.4755/late=0.4890 below). **(b) post_cal pooled=0.7024 — essentially unchanged from the unrestricted 0.7078.** Removing the within-block-position cue does not reduce calibrated accuracy at all — the calibrated result does not depend on that specific cue.

### 3.4 F-DRIFT-E — boundary-privilege check, two attempts (L023, L025)

**First attempt: INVALID-DESIGN (L023).** Uncontrolled class imbalance (up to 1451/10159) made every reported accuracy track the majority-class base rate, not a boundary-privilege signal (`block2_75pct` pre_cal=0.8765±0.0008 with `shrink≈0`, pure majority-class prediction). Logged, not deleted; no verdict from it citable. Motivated **FIX 1** (codebase-wide C3 balance hardening: class balance/majority-rate/lift/balanced-accuracy now mandatory, hard-fails outside 45/55 unless declared imbalanced) and **FIX 2** (fixed-symmetric-window redesign).

**Redesign (L025):** 7 positions (25/50/75% through each block + true boundary), fixed window W=48 (derived from data), balanced accuracy. **Pre-registered verdict: MIXED** (2/6 shifted positions within tolerance of the true boundary, 4/6 not). **⚠ POST-HOC finding that reframed the entire investigation:** balanced accuracy sorts by whether the window contains a **block onset**, not by proximity to the task boundary — onset-containing positions (true_boundary=0.8968, block2_25pct=0.8667, block1_25pct=0.8585) all high; steady-state positions (four remaining, 0.5405–0.5557) all near chance. **`block1_25pct` — zero task content, entirely within one block — reached 0.8585, within 0.04 of the true boundary's 0.8968, at matched N/balance/span.** Flagged as the strongest single control in this audit.

### 3.5 The rest-break discontinuity — mechanism, universal replication (L029, L033)

Local, no-Modal inspection of raw `.vmrk` marker files and behavioral `.tsv` files (initially 11/30, extended to **all 30/30 subjects** via a small glob-restricted download, no `.eeg` binaries, no Modal needed): **in every subject, the single largest inter-marker time gap in each task block falls at event index 49 (29/30 subjects) or 54 (1/30, sub-01) — universal, zero exceptions.** Gaps: 5.6x–69.0x the block's own median inter-trial gap, 8.3s–239.5s absolute (vs. ~3–5s typical ITI). Cross-referenced against the Encode behavioral file: ~4.0 EEG epochs per behavioral trial, placing the discontinuity at ~behavioral trial #12–13 of each 50-trial block. No explicit break marker exists in the raw stream — inferred from the timing anomaly alone, described as "consistent with a self-paced rest break," not confirmed screen content. **Renamed from "trial-50 event"/"onset transient" to "rest-break discontinuity" — "onset" is reserved for the separate block-onset concept (trial 0 of a block) established in §3.4; the two must not be conflated.** Full 30-subject table: `results/rest_break_discontinuity_table.md`.

### 3.6 F-DRIFT-F — onset-exclusion test (L028)

**(a) Onset-excluded k-sweep:** F-DRIFT-C's sweep re-run after dropping each block's first 50 trials. **pre_cal_balanced collapsed to chance at every k, range 0.4974–0.5037. DECISIVE — the F-DRIFT-C "temporal separation" interpretation is WITHDRAWN**, exactly as pre-registered. (F-DRIFT-C's own numbers are unchanged; only the interpretation of what drove them is settled: onset concentration within pseudo-class 0 under `pseudo_label(i)=(i//k)%2`, not separation.)

**(b) Onset-distance parametric sweep:** 14 positions (7 distances × 2 blocks), fixed window. Pooled curve: d25=0.5801, **d50=0.8343**, d75=0.5513, d100=0.5082, d125=0.5259, d150=0.5059, d175=0.5333 (block2 replicates independently: d25=0.5727, d50=0.8405, d75=0.5697). Spearman rho=-0.7143, p=0.0357 — meets the pre-registered decay criterion exactly as computed. **⚠ Verdict text corrected, disclosed:** the shape is a **localized spike at d50 with a flat tail**, not a smooth decay (which would peak at d25) — corrected description: **"a step change at approximately trial 50 of each block,"** matching the rest-break discontinuity (§3.5) almost exactly in location.

### 3.7 F-DRIFT-G — real labels under onset exclusion (L031) — RESULT DISCLOSED AS MIS-SPECIFIED, see §5.3

Real Search-vs-Memorize labels, same onset exclusion as F-DRIFT-F(a). **post_cal_balanced=0.7215, pre_cal_balanced=0.4916. The pre-registered "≥0.65 → GENUINE SIGNAL" branch fired.** **This is the third disclosed criterion mis-specification in this audit — see §5.3. The result must NOT be read as evidence of task decoding.**

---

## 4. The pre-calibration invariant (L032) — one of this audit's firmest claims

| Configuration | pre_cal (balanced where applicable) | Ledger |
|---|---|---|
| Unrestricted | 0.5201 | L009/L011 |
| Phase-matched | 0.5083 | L020 |
| Cross-parity | 0.4839 | L003 |
| Within-parity | 0.5303 | L017 |
| Onset-excluded (rest-break region removed) | 0.4916 | L031 |

**Every one of five structurally different configurations lands within ±0.03 of chance.** Twenty-eight subjects of training data, five different ways of removing structure from the problem, and not one produces a zero-shot cross-subject signal. **There is no subject-general Search-vs-Memorize representation in these features.** This claim does not depend on how any individual configuration's post_cal number or pre-registered verdict was interpreted — it is a direct, un-editorialized read of the pre_cal column.

---

## 5. THE THREE DISCLOSED CRITERION MIS-SPECIFICATIONS

This audit's discipline required every interpretation threshold be pre-registered before a run, then applied mechanically afterward. Three times, the pre-registered criterion itself turned out to be wrong — caught, disclosed in full alongside (never in place of) the original mechanical verdict, and corrected. Listed together, deliberately, so the pattern is visible to a reviewer.

### 5.1 F-DRIFT-C — strict pointwise monotonicity (caught 2026-08-20)

**What was pre-registered:** `is_monotone_rising` — strict pointwise ordering across the 7 k-sweep points (`curve[i] <= curve[i+1]` for every consecutive pair).

**What went wrong:** fragile against single-seed noise. Failed at k=2 and k=25 (0.007–0.011 wiggles) despite the true curve being an unambiguous threshold/step function. Original verdict: **NOT MET** — retained on the record exactly as computed, never deleted.

**How it was corrected:** a post-hoc (explicitly labelled as such) Spearman rank correlation with a subject-level bootstrap CI, plus a two-group paired Wilcoxon contrast — both robust to the exact noise pattern that broke the pointwise check. Both confirmed a strong, significant trend (rho=0.8571 p=0.0068; Wilcoxon p=3.7e-09). **Later superseded in kind, not in number, by F-DRIFT-F(a): the trend these tests correctly detected turned out to reflect onset concentration, not temporal separation — see §3.6.**

### 5.2 F-DRIFT-E — the original balance-free design (caught 2026-08-20)

**What was pre-registered:** shift the split point to 6 positions (25/50/75% through each block), let pseudo-class size vary freely with position and block length.

**What went wrong:** produced severe, uncontrolled class imbalance (up to 1451/10159) at the extreme shift positions. Every reported accuracy tracked the majority-class base rate, not any boundary-privilege signal — the design conflated "accuracy" with "base rate" and had no mechanism to catch the difference.

**How it was corrected:** logged as INVALID-DESIGN, not deleted (L023). Motivated a codebase-wide fix: every classification result must now report class balance, majority-class rate, accuracy-minus-majority-rate, and balanced accuracy, hard-failing outside a 45/55 band unless the design explicitly declares itself imbalanced (FIX 1). F-DRIFT-E itself was redesigned with a fixed symmetric window guaranteeing exact 50/50 balance by construction (FIX 2) — this redesign is what actually produced the onset-proximity finding (§3.4).

### 5.3 F-DRIFT-G — equating "survives onset exclusion" with "genuine task signal" (caught 2026-08-20)

**What was pre-registered:** if the real Search-vs-Memorize contrast's post_cal accuracy survives dropping each block's first 50 trials (removing the rest-break discontinuity) at ≥0.65, treat this as evidence of genuine task signal masked by the artifact.

**What went wrong:** this equates two things that are not equivalent. The pseudo-label contrasts that collapsed under the same onset exclusion (F-DRIFT-F(a)) were **within-block** constructions — both pseudo-classes drawn from the same task block. The real contrast is **between blocks**. Removing trials from inside each block removes the rest-break discontinuity from both blocks equally, but leaves every between-block confound fully intact: **task instruction** (differs by block), **stimulus set** (disjoint within subject — each scene/object shown exactly once, under exactly one condition), **session half** (block 1 always first, block 2 always second), and **post-break state** (block 2 still follows the ~400s inter-block break even with its own internal rest-break region removed). A result surviving the removal of one confound is not evidence against the other four.

**How it was corrected:** the fired branch (post_cal_balanced=0.7215) is recorded exactly as computed (L031) — but flagged, in the same entry, as not usable evidence of task decoding. This directly motivated F-SME (§6): the one contrast in the dataset immune to all four remaining confounds.

---

## 6. F-SME — subsequent memory, the last experiment (L034 pre-registration, L035 result)

**The only contrast in ds005189 that is within-block, within-task, within-stimulus-set, and within-session-half** — none of §5.3's four remaining confounds apply. Linking methodology: `(scene,obj)` uniquely joins each of a subject's 100 Encode trials to a Test-phase recognition judgment (re-verified 2026-08-20, 0 unmatched/collisions, exactly reproducing the previously-documented forgotten-count table). Epoch-to-behavioral-trial correspondence verified via marker codes 11/21 (exactly 50/subject/task, strictly chronological, confirmed across all 30 subjects) — the only codes with a verified trial-level mapping.

**RESULT: NULL.**

| Condition | n (post-exclusion) | post_cal AUC | 95% CI |
|---|---|---|---|
| within_search | **0/29 — UNEVALUABLE** | — | — |
| within_memorize | 13/29 | 0.4968 | [0.4489, 0.5423] |
| pooled | 21/29 | 0.5141 | [0.4760, 0.5547] |

**within_search's 0/29 is itself a behavioural finding, not merely an exclusion-table footnote: it IS the dataset's own search-superiority effect appearing as a power failure** — so few Search-encoded items are ever forgotten, by any subject, that the within-Search subsequent-memory contrast cannot be tested in this dataset at all. Both scoreable conditions' CIs contain 0.5 → no detectable confound-free cognitive signal, per the pre-registered rule. **This is NOT evidence against subsequent-memory effects existing — it is a power statement given 0–18% minority (forgotten) trials per subject/condition. The methodological paper is the paper.**

---

## 7. What this audit rules in, rules out, and cannot resolve in this dataset

**RULED IN:**
- The pipeline is leak-free (F-LEAK).
- The reported ~0.71–0.72 accuracy numbers are real, reproducible, correctly-computed — on a leak-safe calibration split.
- The pipeline is legitimately few-shot subject-adaptive: the calibration step reliably recovers ~0.70+ accuracy given almost any structured cue in the calibration trials (real task, within-block position, a synthetic pseudo-label) — not specifically task content.
- A rest-break discontinuity occurs at a highly consistent location (~trial 50) in every task block, in all 30 subjects — the mechanism behind the pseudo-label dose-response curves and the onset-proximity spikes.
- Ocular contamination is not a significant confound (F-OCULAR combined CLEAN, conditional on the pre-F3 alignment).
- Block-order counterbalancing does not explain the real-vs-pseudo transfer asymmetry (F-PARITY-WITHIN).
- Removing the within-block-position cue does not reduce calibrated accuracy (F-DRIFT-D post_cal) — nor does removing the rest-break region (F-DRIFT-G post_cal) — the calibrated result is robust to both individually.

**RULED OUT:**
- The manuscript's "strict subject-independent, zero test-subject leakage into any stage" protocol description — false as written.
- The Search-vs-Memorize contrast as reported being a genuine subject-independent task-decoding result — indistinguishable from a zero-task-content pseudo-label at matched temporal separation (F-DRIFT, F-DRIFT-B).
- Temporal separation, per se, as the mechanism behind the F-DRIFT-C dose-response curve — it was onset concentration (F-DRIFT-F(a), decisive).
- Block-order/time-cancellation as the explanation for the real-vs-pseudo pre-calibration transfer asymmetry (F-PARITY-WITHIN).
- "Survives onset exclusion" as sufficient evidence of genuine task signal (F-DRIFT-G's disclosed mis-specification) — the real contrast's remaining confounds (task instruction, stimulus set, session half, post-break state) are all between-block and untouched by within-block manipulation.
- Any subject-general Search-vs-Memorize representation in these features at all (the pre-calibration invariant, §4) — this is the strongest ruled-out claim in the audit.

**CANNOT BE RESOLVED IN THIS DATASET:**
- Whether genuine task-related neural signal exists at all, confounded or not — this audit shows the block-level confounds are inseparable from any such signal using within-block manipulations alone; a genuinely deconfounded test (F-SME, using subsequent memory) was attempted and is underpowered (0–18% minority trials/subject), not conclusive in either direction.
- The exact cognitive/physiological cause of the rest-break discontinuity (re-orientation after rest, arousal shift, impedance drift, etc.) — only the structural timing coincidence is established, not the mechanism.
- Whether the rest-break discontinuity is fixed-at-count or proportional-to-block-length — block lengths in this dataset are too narrowly distributed (200–205 events) to discriminate the two hypotheses.
- F-OCULAR(a)'s seed-42-vs-5-seed basis-mismatch reopening — blocked on missing per-seed data, de-prioritized since the contrast it gates is no longer reported on.

---

## 8. Standing caveats

- **Pre-F3 alignment dependency:** the entire F-DRIFT family, F-PARITY-WITHIN, and F-SME hardcode pre-F3 pooled-only Euclidean Alignment, not the parametrized `eeg_alignment.py` module F3 introduces. All verdicts above are conditional on this alignment.
- **Single-seed=42 basis:** every control in this audit beyond the original F4 5-seed headline runs uses a single seed — a deliberate cost/discipline tradeoff for diagnostic controls, not headline accuracy numbers.
- **Full-disclosure discipline applied throughout:** every mis-specified criterion in this audit (§5) is recorded with its original verdict intact, alongside — never in place of — its correction. This document inherits that discipline: it is a full-disclosure record of what was tested, pre-registered, found, and corrected — not a retroactively cleaned narrative.
- **The two model branches (tangent-space spatial + Mamba temporal) referenced in the original manuscript were never jointly trained** in any of the audited scripts — noted here for `PAPER/outline.md`'s Methods section, which must describe the pipeline as actually implemented.
