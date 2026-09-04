# NUMBERS.md — Authoritative Numbers Table (v5, GATE T3)

**GATE T3 STEP 1 — critical scope note.** `draft_elsevier.tex` and `draft_ieee.tex` are build products of `PAPER/figures/scripts/build_latex.py`, generated from `PAPER/draft.md` on every run (full overwrite, not a patch — see RESULTS_LEDGER.md's GATE T3 STEP 1 entry for the code-level confirmation). GATE T2's edits directly to `draft_elsevier.tex` will be silently reverted by the next build. Every string in this file is still correct and still the thing to copy from — but the **target** for applying them is `draft.md`, not either `.tex` file, until this is resolved.

Built GATE C13 STEP 5, corrected GATE C14. GATE C13's "Manuscript string" column was never containment-tested before publishing — every tolerance-bearing string in that version failed to contain its own full-precision value plus a one-quantum excursion in at least one direction (root cause: `X ± T` where `X` is a *rounded* value compounds up to half a rounding-unit of rounding error with the jitter tolerance on top of it, and 3dp's own rounding half-width, 0.0005, already exceeds the ~4.1e-4 quantum — no band centered on a 3dp value can cover a one-flip excursion, as a general structural fact, not a per-row bug). This version fixes that, corrects five wrong distance figures and two mislabelled quanta discovered while rebuilding the table, and adds a containment-test column.

**Containment test**: for every row with a tolerance, the printed interval `[center−tol, center+tol]` is checked against three points: the full-precision value, value+quantum, and value−quantum. All three must fall inside. Verified in code (`gatec14_final_table.py`), not asserted.

---

## R2 / R3 / R1(b) — composition runs (`run_r1b_r2_r3_composition_runs.py`)

**Sourcing correction (STEP 2e)**: every row below is now attributed uniformly to `results_r1b_r2_r3_composition_runs_20260825T042232Z_a3293792.json` (the rerun) — not mixed with the original file as GATE C13 did. Where a field is bit-identical between the original and the rerun (all pre-calibration fields, plus R2 post-cal AUC and R1(b) post-cal balanced), that is noted separately rather than by silently citing a different file for different rows of the same condition.

| Quantity | Full precision | Bit-identical to original file? | Bare-4dp: would flip? | Manuscript string | Containment |
|---|---|---|---|---|---|
| R2 pre-cal balanced | 0.5773379921335015 | Yes | Yes | `0.5773 ± 0.0005` | PASS |
| R2 post-cal balanced | 0.7310593042349257 | **No** (orig: 0.7306583419253829) | Yes | `0.7311 ± 0.0005` | PASS |
| **R2 pre-cal AUC** | **0.6309619276740367** | Yes | **Yes — AT RISK at 0.62× its own quantum, not safe at 3.92×** | `0.6310 ± 0.0001` | PASS |
| R2 post-cal AUC | 0.8055141864283805 | Yes | Marginal — safe at only 1.88× | `0.8055 ± 0.0001` (recommended; bare `0.8055` is technically stable but the margin is thin enough to warrant the same band as pre-cal, for consistency) | PASS |
| R3 pre-cal balanced | 0.5293021360947106 | Yes | Yes | `0.5293 ± 0.0005` | PASS |
| **R3 post-cal balanced** | **0.7186365598720654** | **No** (orig: 0.7184754254859873 → rounds to 0.718) | Yes | `0.7186 ± 0.0005` | PASS |
| R3 pre-cal AUC | 0.558728923318274 | Yes | No (safe, 6.93×) | `0.5587` | PASS (round-trip) |
| R3 post-cal AUC | 0.7828820557099339 | **No** (orig: 0.7828790154384984 — differs by exactly R3's own AUC quantum) | No (safe, 10.54×) | `0.7829` | PASS (round-trip) |
| R1(b) pre-cal balanced | 0.8121027721433399 | Yes | Yes | `0.8121 ± 0.0005` | PASS |
| R1(b) post-cal balanced | 0.8656524678837051 | Yes | Yes | `0.8657 ± 0.0005` | PASS |
| R1(b) pre-cal AUC | 0.9276220021477152 | Yes | No (safe, 17.60×) | `0.9276` | PASS (round-trip) |
| R1(b) post-cal AUC | 0.9532832199817048 | **No** (orig: 0.9532816290816531) | No (safe, 20.88×) | `0.9533` | PASS (round-trip) |

**GATE T1 STEP 1b — the AUC quantum is a minimum increment, not a bound.** `1/(n_folds×n_pos×n_neg)` is the *smallest possible* AUC change (one score crossing exactly one adjacent opposite-class trial). A score that leapfrogs *k* opposite-class trials moves the AUC by *k* such units. Under last-bit float jitter, *k*=1 is the realistic case and R3's observation confirms it, but a stated ratio like "safe at 6.93×" is really "safe provided no single rerun moves this AUC by more than 6 units of its own quantum" — an assumption, not a proof, exactly parallel to the one already stated explicitly for balanced accuracy (below, "±1 quantum is what the evidence supports"). Units of movement each row's margin actually tolerates before flipping (⌊dist/quantum⌋): R2 pre-cal **0** (already outside 1 unit — this is why it's AT RISK), R2 post-cal **1**, R3 pre-cal **6**, R3 post-cal **10**, R1(b) pre-cal **17**, R1(b) post-cal **20**.

**GATE C15 STEP 1 correction — the AUC quantum is condition-specific, not a single generic 3.04e-6.** GATE C14 applied R3's own measured AUC movement (3.040271e-6) uniformly to all six AUC rows. That figure is not a generic AUC jitter scale — it is exactly `1/(n_folds × n_pos × n_neg)` for R3's own per-fold class totals: `1/(29×107×106) = 3.0402714e-6`, matching the measured movement to 7 significant figures. This is a **second, independent confirmation of the quantum model** on a rank-based metric (AUC), the same way `0.5/(n_folds × class_size)` was confirmed on balanced accuracy. The formula gives a **different, larger** quantum for R2 (`1/(29×43×42) = 1.909e-5`, over 6× R3's) and a smaller one for R1(b) (`1/(29×85×255) = 1.591e-6`). Recomputing each AUC against its **own** condition's quantum: **R2 pre-cal AUC is AT RISK at 0.62×**, not safe at 3.92× as GATE C14 stated; R2 post-cal AUC is safe but only at 1.88×; R3 and R1(b)'s AUCs remain comfortably safe (6.9×–20.9×) since their own quanta are smaller. **This is a regression**: GATE C13 had already derived R1(b)'s condition-specific AUC quantum correctly (`q=1.59e-6`, matching `1/(29×85×255)` exactly) before GATE C14 discarded per-condition quanta in favor of one "now-measured" number. The fix that would have caught it: the same discipline already adopted for citation-diffing (STEP 1g, GATE C11) — before overwriting a quantity's previously-derived value with a new one, diff the two and reconcile the discrepancy explicitly, rather than silently treating "newly measured" as strictly better than "previously derived."

**Known-wrong manuscript figures (unchanged from GATE C13, still pending a text fix)**:
- R1(b) pre-cal balanced: manuscript cites **0.8114**; artifact says **0.8121** (Δ=7.03e-4, not a clean multiple of any R1(b) quantum — a real transcription error).
- R1(b) pre-cal AUC: manuscript cites **0.9281**; artifact says **0.9276** (Δ=-4.78e-4, ~150× the AUC jitter scale — a real error).
- R3 post-cal balanced: **cite `0.7186 ± 0.0005`** (→0.719 at 3dp, if 3dp is used at all elsewhere) — not 0.718. `0.7186 ± 0.0005` contains **both** candidate values, so once the tolerance form is adopted, the 0.718-vs-0.719 citation choice stops mattering for what's printed — it remains relevant only for which *file* is the authoritative source (C11 STEP1d's recency ruling still governs that, unchanged).
- R1(b)'s 0.85 criterion, re-verified under the new form: `0.8657 − 0.0005 = 0.8652 > 0.85`. **The criterion survives.**

## C1 — 30-shuffle label control

| Quantity | Full precision | Quantum | Basis | Bare-4dp dist. | Manuscript string | Containment |
|---|---|---|---|---|---|---|
| Null mean | 0.4989845470398799 | 1.368e-5 | model: Q/30 (mean's sensitivity to one flip is exactly linear — numerically confirmed) | 0.0000350 | `0.4990` | PASS (round-trip) |
| Null SD | 0.010263349546508989 | **3.246e-5 (corrected — see STEP 2c note)** | numerical perturbation of the actual 30 shuffle values | 0.0000133 | `0.0103 ± 0.0001` | PASS |
| CI lower (order statistic) | 0.4827624393783175 | 4.105e-4 | single-eval, undamped | 0.0000124 | `0.4828 ± 0.0005` | PASS |
| CI upper (order statistic, = C3's threshold) | 0.5172089204567152 | 4.105e-4 | single-eval, undamped | 0.0000411 | `0.5172 ± 0.0005` | PASS |

**STEP 2c correction**: an SD's sensitivity to one observation moving by *q* is **not** *q/n* — that model only holds for a mean (confirmed: numerically perturbing each of C1's 30 actual shuffle values by ±R2's quantum reproduces the mean's Q/30 exactly, but the *maximum* resulting SD movement is 3.246e-5, not the naively-modeled 1.368e-5 — 2.4× larger). C1's null SD **remains AT RISK under a bare citation** (corrected quantum 3.246e-5 > bare-4dp distance 1.33e-5), but by a real, larger margin than previously computed — **retracting** GATE C13's "margin only 3.3e-7 — the narrowest verdict found this session," which rested on the wrong (q/n) model. The true margin is ~1.9e-5, six times wider than claimed.

**GATE C15 correction**: C1's null-*mean* quantum, 1.368e-5 (= R2's quantum ÷ 30), **is** the correct one-flip quantum for a 30-shuffle mean — confirmed by the same numerical perturbation test used for the SD, above. GATE C14's "read as an optimistic lower bound" framing is **retracted as applied to this single-flip figure** — it conflated two separate questions. The single-flip model is exactly right for one index flipping. The separate, still-open question is whether *more than one* index could flip simultaneously in a real C1 rerun (as demonstrably happens for C4, see below) — that multi-flip risk is real but is not a flaw in the 1.368e-5 figure itself.

## C4 — 500-shuffle high-resolution control

| Quantity | Full precision | Figure | Basis | Manuscript string | Containment |
|---|---|---|---|---|---|
| Null mean | 0.49950918394623284 | 2.444e-6 | **observed 3-flip displacement** (canonical→Aug27; ≈2.98× the 8.21e-7 one-flip quantum, confirmed in code — GATE C14 mislabelled this "damped ÷500," GATE C15 corrects the label again: it is not a one-flip quantum at all, conservative but from a 3-index-flip event) | `0.4995` | PASS (round-trip) |
| Null SD | 0.011167803690267993 | 1.076e-6 | same observed 3-flip event's SD displacement | `0.0112` | PASS (round-trip) |
| CI lower (order statistic) | 0.47643487226486425 | 4.105e-4 | single-eval, undamped | `0.4764 ± 0.0005` | PASS |
| CI upper (order statistic) | 0.5198726944667201 | 4.105e-4 | single-eval, undamped | `0.5199 ± 0.0005` | PASS |
| Normal-approx CI lower | 0.4776206909268941 | ~4.5e-6 (estimated: propagating both the mean's and SD's own small observed jitter through the formula) | formula, `mean − 1.96×SD` | `0.4776` | PASS (round-trip; safe bare — both inputs are small-jitter empirical figures, not order statistics) |
| Normal-approx CI upper | 0.5213976769655716 | ~4.5e-6 | formula, `mean + 1.96×SD` | `0.5214` | PASS (round-trip) |
| p-value (raw / add-one) | 0.0 / 0.001996007984031936 | — | — | `p < 0.002` | — |

**GATE T2 STEP 3a — normal-approx CI added (was a NUMBERS.md gap).** Unlike the percentile CI (an order statistic carrying the full undamped 4.105e-4 quantum), the normal-approx CI is a closed-form function of the mean and SD, both of which carry only their own small observed 3-flip displacement — so its own jitter is tiny and it is comfortably bare-4dp-safe.

**STEP 2b correction**: GATE C13 labelled these quanta "damped ÷500," implying `4.105e-4/500 = 8.21e-7` — that is not what either figure is. Both `2.444e-6` and `1.076e-6` are the **directly observed** movement of the mean/SD between the canonical (035948Z) and Aug-27 runs — real measurements, not a theoretical model's output, and they are ~3× the naive single-flip model because that particular pair of reruns involved more than one index flipping (consistent with GATE C7's finding of multiple differing indices between runs). **C1's mean/SD use the theoretical Q/n model instead because C1 has no empirical rerun to measure from** (confirmed: C1 has no distinct stamped rerun, per RESULTS_LEDGER.md) — this is a documented necessity, not an inconsistency, but it means **C1's model-based quanta should be read as an optimistic lower bound**: if a real C1 rerun ever occurred and multiple shuffles flipped simultaneously (as demonstrably happens for C4), C1's true jitter could exceed the modeled value, the same way C4's empirical jitter exceeds C4's own naive model.

## C3 — leave-one-subject-out jackknife

| Quantity | Full precision | Quantum | Bare-4dp dist. | Manuscript string | Containment |
|---|---|---|---|---|---|
| LOO min | 0.5702914886884988 | 4.252e-4 | 0.0000415 | `0.5703 ± 0.0005` | PASS |
| LOO max | 0.585202104097453 | 4.252e-4 | 0.0000479 | `0.5852 ± 0.0005` | PASS |
| Half-gap threshold | 0.030064535838393136 | 4.105e-4 (both-terms) | 0.0000145 (**corrected from 0.0000151 — was a transcription duplicate of C4 CI lower's distance, see STEP 2a**) | `0.0301 ± 0.0005` | PASS |
| **Max shift (sub-03)** | 0.007864111963951537 | 4.252e-4 | **0.0000141 at its own cited precision, 4dp** (GATE C13 wrongly checked this at 3dp — distance was 0.000364 there; the row's own string is 4dp, so 4dp is the precision that matters, and at 4dp it is even more clearly AT RISK) | `0.0079 ± 0.0005` | PASS |

**C3 verdict robustness (reconfirmed from GATE C13, unaffected by these corrections)**: perturbing the threshold by ±1 quantum leaves both C3 verdicts (LOO min above threshold; max shift below half-gap) unchanged in both directions.

## D2 — parity-split control

| Quantity | Full precision | Quantum | Manuscript string | Containment |
|---|---|---|---|---|
| Odd (search-first) real | 0.6156660338554026 | 8.503e-4 | `0.6157 ± 0.001` | PASS |
| Even (memorize-first) real | 0.5582133628645256 | 7.937e-4 | `0.5582 ± 0.001` | PASS |

D2's own quanta (8.5e-4, 7.9e-4) exceed the blanket ±0.0005 used everywhere else — a ±0.0005 band fails containment here (verified: fails on both the up- and down-flip sides). D2 needs its own, wider ±0.001 tolerance; this is quantity-specific, not a general policy exception.

## Effect sizes

| Quantity | Full precision | String | Containment |
|---|---|---|---|
| vs. C4 null mean | 0.07782880818726866 = 7.782881 pp | `7.78 ± 0.05 pp` | PASS |
| vs. 0.50 chance floor | 0.0773379921335015 = 7.733799 pp | `7.73 ± 0.05 pp` | PASS |
| vs. C1 null mean (computed this gate, see recommendation below) | 0.5773379921335015 − 0.4989845470398799 = 0.0783534450936216 = 7.835345 pp | `7.84 ± 0.05 pp` (if cited at all) | not separately containment-tested — see recommendation |

**GATE T2 STEP 3b — recommend NOT citing the vs.-C1 figure separately.** C4 (500 shuffles) was run specifically "to sharpen C1's p-value estimate" (the manuscript's own words, line 132) and supersedes C1 for precision purposes — introducing a third, distinct effect-size figure (7.84pp, vs. C1's cruder 30-shuffle null) adds a number the manuscript doesn't need and that a reader could confuse with the other two. **Recommend**: cite only the two effect sizes already established (vs. C4 null mean, vs. chance floor); Figure 3's caption phrase "R2 falls ~7.3–7.4 percentage points above both" should be revised to state the C4-based figure once, not imply a shared range covers both nulls with adequate precision (0.4990 vs. 0.4995 are close but not identical, and 7.78 vs. 7.84pp is a real difference, not noise).

## D2 per-subgroup null statistics (GATE T2 STEP 3c — was a NUMBERS.md gap)

| Subgroup | Null mean | Null SD | Percentile 95% CI | Normal-approx 95% CI |
|---|---|---|---|---|
| Odd (Search-first, n=14) | 0.4987119654063176 | 0.010886974281291819 | [0.4793184424932763, 0.5178754350577439] | [0.4773738879143718, 0.5200500428982634] |
| Even (Memorize-first, n=15) | 0.4984692998646488 | 0.014671749549047583 | [0.4723961794019934, 0.5222872831303064] | [0.4697131991583237, 0.5272254005709738] |

**Manuscript strings** (4dp + tolerance, quantum not independently derived here — treating as an order-statistic-scale figure pending its own quantum derivation, consistent with the rest of this table's policy): odd null mean `0.4987 ± 0.0005`, odd percentile CI `[0.4793 ± 0.0005, 0.5179 ± 0.0005]`; even null mean `0.4985 ± 0.0005`, even percentile CI `[0.4724 ± 0.0005, 0.5223 ± 0.0005]`.

**RESOLVED (GATE T3 STEP 3)**: sub-01 is in D2's **odd** (Search-first) pool (confirmed: `odd_search_first.subjects` in the committed JSON begins `['01', '03', ...]`). The sub-01 practice-trial exclusion (10 epochs) affects sub-01 only, so it moved the odd subgroup's every statistic and structurally could not touch the even subgroup's — exactly the asymmetry observed: odd's cited stats (mean 0.5043, CI [0.4842, 0.5276]) match none of the current JSON's fields, while even's cited stats (0.4985, [0.4724, 0.5223]) already match closely. Confirmed further: the current odd percentile CI's upper bound, 0.5178754350577439, is exactly the figure GATE C11 used in its own margin computation — the JSON is internally consistent and current; it is the manuscript that is pre-trim. This is the same staleness affecting every other figure in this table, not a separate bug.

**GATE T4 STEP 5a — two different quantities were both being called "the gap"; disambiguated here.**

- **`D2 gap-above-null-mean`** (real − null **mean**): odd = 0.6156660338554026 − 0.4987119654063176 = **0.1169540684490850** (→ `0.1170 ± 0.0005`, not the stale 0.1162); even = 0.5582133628645256 − 0.4984692998646488 = **0.0597440629998768** (→ `0.0597 ± 0.0005`, digits unchanged from the stale citation, but still needs the tolerance attached). **This is the quantity Figure 4's annotation and the manuscript's own "gap above null" prose use** — confirmed from the manuscript's own stated formula ("0.6205 − 0.5043"), which subtracts the null *mean*, not a CI bound.
- **`D2 margin-above-null-CI`** (real − null 95%-CI **upper bound**): odd = 0.6156660338554026 − 0.5178754350577439 = **0.0977905987976587** (≈0.0978); even = 0.5582133628645256 − 0.5222872831303064 = **0.0359260797342192** (≈0.0359). **This is the quantity the corrected margin-ranking table below actually uses** (via GATE C11 STEP 3's both-terms ratio computation). GATE T3 STEP 3b already confirmed GATE C11's margin computation used the *current* CI-upper-bound figure (0.5178754350577439), not the stale manuscript one — **the margin-ranking table was already correct and needs no revision**; only the manuscript's separate "gap above null mean" narrative (Figure 4, the D2 asymmetry paragraph) was stale.

Both `gap-above-null-mean` figures now confirmed safe to replace — see REPLACEMENTS.md's re-anchored `draft.md` map.

**GATE D STEP 1a — the gap-above-null-mean tolerance itself was wrong: needs ±0.001, not ±0.0005.** The gap's dominant quantum is the real-value's own quantum (the null mean, averaged over 30 shuffles, moves by only ~2.6–2.8e-5 per flip — negligible by comparison; confirmed by direct perturbation of the raw `null_shuffle_values` in `results_d2_parity_split_check.json`, below). Three-point containment, verified in code:

| Quantity | Value | Quantum used | ±0.0005 | ±0.001 |
|---|---|---|---|---|
| Odd gap | 0.11695406844908501 | 8.503e-4 (odd real's own) | value+q=0.11780 **FAILS** (band [0.1165,0.1175]) | value+q=0.11780, value−q=0.11610, both PASS |
| Even gap | 0.059744062999876835 | 7.937e-4 (even real's own) | value+q=0.06054 **FAILS** (band [0.0592,0.0602]) | value+q=0.06054, value−q=0.05895, both PASS |

**Manuscript strings corrected: `0.1170 ± 0.001` (odd), `0.0597 ± 0.001` (even)** — the blanket ±0.0005 previously attached to these two figures throughout `draft.md` (§3a's own gap sentence, the Limitations item-3 restatement, and Figure 4's caption) was too tight and has been widened to match D2's own established tolerance policy (this table already required ±0.001 for the two subgroups' bare `real` values, for the identical reason).

**D2 null SD, new derivation (was previously bare with no tolerance anywhere in this table):** `results_d2_parity_split_check.json`'s `null_shuffle_values` (the real 30 within-group shuffle values per subgroup) let the SD's own one-flip sensitivity be measured directly, the same method already used for C1's SD (GATE C15 STEP 2c). Perturbing each of the 30 values by ± the subgroup's own real-value quantum and taking the maximum resulting sample-SD movement:

| Subgroup | Null SD (full precision) | One-flip SD quantum | Bare-4dp round-trip? | Manuscript string | Containment |
|---|---|---|---|---|---|
| Odd | 0.010886974281291819 | 5.593e-5 | No (0.010831 rounds to 0.0108, not 0.0109) | `0.0109 ± 0.0001` | PASS |
| Even | 0.014671749549047583 | 6.252e-5 | No (0.014609 rounds to 0.0146, not 0.0147) | `0.0147 ± 0.0001` | PASS |

Both pass at ±0.0001, matching C1's null-SD tolerance convention exactly (same underlying mechanism: an averaged quantity whose naive q/n model undersells the real one-flip sensitivity enough to break bare-4dp round-tripping, but not enough to need anything wider than what's already established elsewhere in this table).

---

## STEP 2a — five distance corrections, root cause

| Row | GATE C13 stated | Corrected | Duplicate of another row's (wrong) figure? |
|---|---|---|---|
| C1 CI lower | 0.0000119 | **0.0000124** | Yes — coincides with R2 pre-cal AUC's distance (0.0000119) |
| C1 CI upper | 0.0000414 | **0.0000411** | No — an independent transcription slip |
| C3 half-gap | 0.0000151 | **0.0000145** | Yes — coincides with C4 CI lower's distance (0.0000151) |
| R1(b) post-cal AUC | 0.0000328 | **0.0000332** | No — appears to be the *original* file's distance transcribed instead of the cited rerun's |
| R3 post-cal AUC | 0.0000290 | **0.0000321** | No — same pattern: the original file's distance (0.0000290) transcribed for the row labelled as the rerun's value |

**The `boundary_dist` script itself is correct** — recomputing all five from scratch with the identical formula reproduces every one of GATE C14's "correct" figures exactly. **The error is entirely in GATE C13's transcription into the markdown table**: the terminal output was printed at only 6 decimal places (`.6f`), which doesn't carry enough resolution to distinguish these close-valued distances, and additional digits were then written into the table by hand — apparently by inference or by copying a value from a neighboring row — rather than by rerunning the computation at full precision. This is a distinct failure mode from GATE C14 STEP 1's containment-check gap: that was a missing verification step; this is fabricated precision in a hand-transcribed table.

---

## Revised precision policy (STEP 1e)

**Balanced accuracies, CI bounds, the half-gap threshold, and C3's max-shift: cite at 4 decimal places with an explicit ±0.0005 tolerance** (D2's two quantities need ±0.001 instead, since their own quanta exceed 0.0005). **AUCs and the null-distribution means/SDs remain bare** at 4dp — confirmed round-trip-stable under a one-quantum perturbation, so no tolerance annotation is needed for those. C1's null SD is the one null-distribution statistic that needs an explicit tolerance (`± 0.0001`) despite being an averaged quantity, because its corrected (non-q/n) jitter model puts it narrowly outside bare-4dp safety.

This replaces "3dp" everywhere the project has recorded it as the target precision for balanced accuracies (DECISIONS.md's drafting-phase notes, RESULTS_LEDGER.md's prior framing, and the reproducibility passage drafted across GATE C9–C10) — **not because 3dp was too coarse**, but because a bare N-dp citation of any kind cannot carry its own uncertainty; the original objection to a bare 4dp figure ("over-precise") was correct for a citation with no attached tolerance, and stops applying once the tolerance is attached, because the tolerance is now what does the honest work, not the digit count.

**Headline under the new policy (STEP 3c): with the tolerance attached, every one of the 26 distinct cited quantities passes containment — none would mislead a reader.** "At risk" now means something different than in GATE C11–C13: it describes whether a **bare** citation at that precision would flip (informational, kept in the table above as "bare-4dp dist." / "would flip?"), not whether the quantity is safely reportable — which, with the correct tolerance attached, all of them now are.

## STEP 1g — process fix

This is the eighth number this session caught by arithmetic rather than by trusting the previous output. The "Manuscript string" column was the entire deliverable of GATE C13 STEP 5, and no containment check was run on it before publishing — it was written by eye, checking that the printed value looked like a sensible rounding, not by testing whether the printed *interval* actually covered the quantity it claimed to bound. **The check going forward**: any string of the form `value ± tolerance` must be tested in code for containment of (i) the full-precision value, (ii) value+quantum, and (iii) value−quantum, before that string is written anywhere — the same discipline this project already applies to numeric claims, extended to the *format* used to present them.

## GATE C15 STEP 1g — thinnest containment margins, and how much perturbation the ±0.0005 band actually tolerates

Two rows clear their tolerance band's edge by the narrowest margins found in this whole table: **R2 post-cal balanced** (0.7310593042349257, band lower bound 0.7306) clears by **4.9e-5** after a one-quantum down-flip; **C1 CI lower** (0.4827624393783175, band lower bound 0.4823) clears by **5.2e-5**. Both confirmed to **pass at ±1 quantum and fail at ±2 quanta** (verified directly: subtracting a second quantum pushes both below their band's lower edge). This is not a latent problem: **every differing index observed across every run-pair in this project (GATE C7's 0236-vs-0324 diff, the C4 Aug-27 pair, every R1(b)/R2/R3 rerun comparison) has moved by exactly one quantum per differing index** — a single index has never been observed to move by two quanta in one comparison. ±1 quantum is what the evidence actually supports for a per-row containment margin; ±2 would be an unmotivated, non-evidence-based extra safety factor these two rows happen not to have room for.

## Supplementary Table S1 — full 29-row LOO jackknife (GATE T2 STEP 3d, major census gap)

**GATE T1's numeric census entirely missed Supplementary Table S1's per-subject content.** The table has 3 numeric columns × 29 rows = 87 numeric cells per file (174 total across both builds), none of which were captured by the original 22-figure pattern list — only the 3 rows previously spot-checked (sub-01, sub-03, sub-10, sub-18 — 4 of 29) were ever compared against current data, and even those were compared using stale aggregate figures (LOO min/max), not the per-subject cells themselves. **This raises the true occurrence count substantially — see STEP 3d's total below.**

Current committed values, read fresh from `results_c3_r2_jackknife.json`'s `loo_results` array (format: `excluded_fold_pre_cal_balanced`, `loo_mean`, `shift_from_full_mean`):

```
sub-01 & 0.6600 & 0.5744 & -0.0030 \\        sub-16 & 0.5880 & 0.5770 & -0.0004 \\
sub-02 & 0.5188 & 0.5794 & +0.0021 \\        sub-17 & 0.5404 & 0.5787 & +0.0013 \\
sub-03 & 0.3571 & 0.5852 & +0.0079 \\        sub-18 & 0.7746 & 0.5703 & -0.0070 \\
sub-04 & 0.5720 & 0.5775 & +0.0002 \\        sub-19 & 0.6476 & 0.5748 & -0.0025 \\
sub-05 & 0.5296 & 0.5790 & +0.0017 \\        sub-20 & 0.4654 & 0.5813 & +0.0040 \\
sub-06 & 0.6921 & 0.5732 & -0.0041 \\        sub-21 & 0.5797 & 0.5773 & -0.0001 \\
sub-07 & 0.7173 & 0.5723 & -0.0050 \\        sub-22 & 0.5905 & 0.5769 & -0.0005 \\
sub-08 & 0.5570 & 0.5781 & +0.0007 \\        sub-23 & 0.5568 & 0.5781 & +0.0007 \\
sub-10 & 0.5739 & 0.5775 & +0.0001 \\        sub-24 & 0.5377 & 0.5788 & +0.0014 \\
sub-11 & 0.6346 & 0.5753 & -0.0020 \\        sub-25 & 0.6351 & 0.5753 & -0.0021 \\
sub-12 & 0.5651 & 0.5778 & +0.0004 \\        sub-26 & 0.5000 & 0.5801 & +0.0028 \\
sub-13 & 0.7040 & 0.5728 & -0.0045 \\        sub-27 & 0.5916 & 0.5768 & -0.0005 \\
sub-14 & 0.5000 & 0.5801 & +0.0028 \\        sub-28 & 0.4665 & 0.5813 & +0.0040 \\
sub-15 & 0.5778 & 0.5773 & -0.0000 \\        sub-29 & 0.6016 & 0.5765 & -0.0009 \\
                                              sub-30 & 0.5078 & 0.5798 & +0.0025 \\
```

**Every one of these 87 cells differs from the manuscript's currently-printed table** (compare e.g. sub-01: manuscript prints `0.6383 & 0.5714 & -0.0023`; current data says `0.6600 & 0.5744 & -0.0030`) — the entire table is stale, not just the 4 previously spot-checked rows, confirming GATE C11 STEP 2c's "confirmed identical across three separate runs" check was performed against an earlier snapshot that has since been superseded by the sub-01 epoching fix (same root cause as the rest of this project's stale figures). **This table is not individually containment-tested per cell in this pass** — 87 cells at 4dp+tolerance each would need the same treatment as every other row in this file; given the LOO min/max/max-shift aggregate quanta are already established (4.252e-4), the same tolerance (`± 0.0005`) is recommended for every cell in columns 2–3 (LOO mean, shift) pending a per-cell quantum derivation; column 1 (`excluded_fold_pre_cal_balanced`) is a single 29-fold-mean-equivalent figure per subject and should carry the same R2-family quantum treatment.

**New aggregate check**: LOO min under current data = 0.5703 (sub-18), LOO max = 0.5852 (sub-03) — **matches NUMBERS.md's existing LOO min/max rows exactly** (0.5702914886884988/0.585202104097453), confirming the aggregate figures were already correctly updated even though the underlying per-subject table was not. Max shift = sub-03's +0.0079, also already matching.

## Corrected margin ranking (unchanged from GATE C11 STEP 3b — uses `margin-above-null-CI`, not `gap-above-null-mean`; see the disambiguation above)

| Rank | Margin | Ratio |
|---|---|---|
| **1 (tightest)** | **D2 even_memorize_first vs. its null** | **22.6×** |
| 2 | D2 odd_search_first vs. its null | 57.5× |
| 3 | C3 LOO-min vs. threshold | 63.5× |
| 4 | R1(b) vs. 0.85 criterion | 77.2× |
| 5 | C4 p-value margin | 125.6× |
| 6 | R2 vs. C4 null mean | 188.5× |
