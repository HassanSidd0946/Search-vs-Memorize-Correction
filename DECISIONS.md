# DECISIONS.md — Binding Scoping Decisions

**Purpose:** durable record of binding decisions (D1–D4) and the ocular-artifact interpretation matrix, extracted verbatim from AUDIT.md so they survive context compaction independent of the narrative log. AUDIT.md remains the canonical chronological record; this file is the fast-reference ledger. If the two ever disagree, AUDIT.md's dated entry is authoritative and this file should be corrected to match.

**Re-read this file (and STATUS.md) at the start of every session and immediately after any context compaction, before taking any action.**

---

## Resolved Scoping Decisions (Gate 0 approved, 2026-08-17)

Gate 0 is **APPROVED** with the following amendments. These are binding decisions, not proposals.

### D1 — Framing: methodological paper, two contrasts, drop memory-decoding framing

The paper is reframed as a **methodological paper**: matched-control attribution of Riemannian alignment vs. temporal sequence modeling under strict LOSO, evaluated on two contrasts:
- **PRIMARY:** Search vs. Memorize (encoding-task decoding), honestly labeled as incidental-vs-intentional encoding. Blocked-design limitation stated in the abstract.
- **SECONDARY:** subsequent memory (remembered vs. forgotten) via the `(scene,obj)` linkage established in Phase 0.5 Priority 1. AUC-primary, accuracy secondary, per-subject minority-class counts disclosed. Framed as an exploratory, confound-free replication of the component attribution — NOT as a memory-decoding result.

**Source publication verified (per D4) — this directly refines the SECONDARY analysis:** the `rk_judg` field in `Test_beh.tsv` (Remember/Know/New) is Tulving's **remember-know recognition-memory paradigm**, not a DRM/false-memory paradigm. The source publication's own central finding is that search-target recollection (Remember judgments, associated with the parietal LPC ERP component) exceeds intentional-memorization recollection, while familiarity (Know judgments, associated with the FN400 component) does not differ by task. This means the SECONDARY analysis, if pursued, has a well-grounded three-way alternative to plain remembered-vs-forgotten: Remember vs. Know vs. New, matching the source paper's own process-level distinction.

**Scope of what changes now (Phase 1, Python/JSON only) vs. later (Phase 3, LaTeX only) — per Absolute Rule 2 (no LaTeX until Phase 3 approval):**
- **Now (Phase 1):** `topoplot_channel_power_features.json` keys, all matplotlib figure-generation scripts' label strings renamed from "True Memory"/"False Memory" to the honest Search/Memorize (incidental/intentional) labels. **Status: DONE (2026-08-17).** Verified via repo-wide `grep` restricted to `*.py`: zero remaining `True Memory|False Memory|true_memory|false_memory` matches. Out of scope for this pass: `AUDIT.md`, `readme.md`, `PAPER/main.tex`, and 3 generated `.html` figures.
- **Deferred to Phase 3 (LaTeX only, needs its own Gate 3 approval):** manuscript title change, abstract rewrite, all in-text true/false-memory language in `PAPER/main.tex`, and the SECONDARY-analysis write-up itself (doesn't exist as a result yet — Phase 2 must run it first).

**Clarification (2026-08-18, per Gate 1 rejection FIFTH item):** the SECONDARY subsequent-memory analysis is **not** a future-work footnote. It is a scoped, AUC-primary, confound-free replication of the component-attribution question (spatial vs. temporal branch contribution) on an independent contrast. It should be implemented in Phase 2 alongside the PRIMARY contrast, subject to the same F3/F11/F-LEAK/F13/F14 rigor — not deferred.

### D2 — sub-09 exclusion criterion (final wording, n=29)

Confirmed per Phase 0.5 Priority 4: sub-09 excluded because its exported continuous EEG recording is truncated to 109.8 s of an intended ~41.5-minute session, yielding zero usable epochs (all 500 markers fall outside the truncated data range). n=29. **Counterbalancing note:** the odd/even block-order counterbalance (Phase 0.5 Priority 2) was designed for 15 odd- and 15 even-numbered subjects out of 30; excluding sub-09 (odd) leaves **14 odd / 15 even** — a 14/15 split, not perfectly even. This asymmetry should be stated alongside the exclusion criterion in Methods.

### D3 — Ocular controls: full scope required before any accuracy number is reported

All four F-OCULAR sub-analyses (a–d) are in scope for **all 29 subjects**, and per explicit instruction, **no LOSO accuracy number is to be reported in Phase 2 until F-OCULAR(a) (the frontal-ablation matched-control re-run) has completed** and been reviewed. This is a hard gate inside Phase 2, not just a limitation footnote.

### D4 — Citations rescoped; source publication located and verified

**Removed from scope:** DRM/false-memory references — these described the wrong paradigm entirely per the Phase 0 ★ CRITICAL FINDING and Phase 0.5 Priority 1.

**Added to scope:** intentional-vs-incidental encoding, levels-of-processing, task-set/attentional-set decoding, and search-superiority literature. Paller & Wagner retained, but scoped specifically to the SECONDARY subsequent-memory analysis, not the primary contrast.

**Source publication — verified, not reconstructed from the dataset title:**

> **Helbing, J., Draschkow, D., & Võ, M. L.-H. (2025). "Incidental Encoding of Objects during Search Is Stronger Than Intentional Memorization due to Increased Recollection Rather Than Familiarity." *Journal of Cognitive Neuroscience*, 37(12), 2538–2557.** https://doi.org/10.1162/jocn.a.80

Dataset DOI: `doi:10.18112/openneuro.ds005189.v1.0.1`. This is the anchor citation for Introduction/Related Work, replacing any prior false-memory-paradigm citations.

**Clarification (2026-08-18, per Gate 1 rejection FIFTH item):** the citation rescoping above is not a proposal — it is in effect now. Any new Related Work drafting (Phase 3) should start from the D4 reference list, not the retired DRM/false-memory list.

---

## Ocular-artifact interpretation matrix (recorded 2026-08-18, BEFORE F-OCULAR scripts are run)

This matrix must be applied when F-OCULAR(a)–(d) results come back, to prevent post-hoc rationalization of whichever outcome appears. Recorded now, prior to any execution, per explicit instruction.

**Inputs:** F-OCULAR(a) = frontal/frontopolar channel ablation (accuracy retained vs. collapsed, relative to the 70.78% matched-spatial-control baseline). F-OCULAR(c) = |Spearman ρ| between surrogate HEOG/VEOG statistics and decision margin (strong ≈ |ρ| ≥ 0.2, weak ≈ |ρ| < 0.2 — using the CI-bounded estimate, not the point estimate alone). F-OCULAR(b) = whether the ICA-cleaned arm's accuracy tracks the uncleaned or the ablated arm.

| Ablation (a) | Correlation (c) | Interpretation | Required action |
|---|---|---|---|
| **Retains gain** | **weak (\|ρ\| < 0.2)** | Clean — ocular artifact is not a material driver of the decode. | Report the matched-spatial-control result as-is. State (a) and (c) both in Methods as the ocular-confound check performed. |
| **Collapses** | **strong (\|ρ\| ≥ 0.2)** | Ocular-driven — a material fraction of decoding accuracy is attributable to eye-movement/blink signal at frontal sites, not task-related cortical activity. | Reframe the headline result: report the ablated-model accuracy as the primary (conservative) number, or explicitly caveat the full-channel number as ocular-contaminated in Limitations/Methods. Do not report the full-channel number without this caveat. |
| **Collapses** | **weak** and **ICA-cleaned arm (b) survives** (tracks uncleaned, not ablated) | Genuine frontal signal — the frontal channels carry real task-related (non-ocular) signal that ablation removes as collateral damage; ICA can separate the ocular component from the genuine frontal signal. | The **ICA-cleaned arm becomes the primary reported result**, not the ablated arm and not the raw uncleaned arm. State explicitly that frontal ablation was rejected as overly conservative because it discards genuine frontal theta/signal along with the ocular artifact. |
| **Retains gain** | **strong** | Ambiguous — decision margin correlates with the ocular surrogate, but removing the associated channels doesn't hurt accuracy (suggesting the same information is redundantly available elsewhere, e.g. via volume-conducted signal at adjacent channels). | Report all three arms (uncleaned, ablated, ICA-cleaned) side by side. Add an explicit Limitations entry stating the ocular-confound question could not be fully resolved by these controls and flagging it for future work with a dedicated EOG channel. |

**Explicit caveat (must be stated wherever this matrix's conclusion is reported):** F-OCULAR(c) and F-OCULAR(b) are the decisive controls. F-OCULAR(a) alone is **necessary but not sufficient**, because frontal ablation removes genuine frontal theta activity along with any ocular artifact — a collapse under ablation is consistent with either explanation, and only the correlation/ICA evidence in (b)/(c) can distinguish them. Do not conclude "ocular-driven" or "clean" from (a) in isolation.

**Baseline reference numbers this matrix will be evaluated against:**
- Matched-spatial-control (uncleaned, full 62-channel) accuracy: **70.78%** (`run_step4_matched_spatial_control.py`, `results_condition4_matched_spatial_control.json`).
- Phase 0.5 Priority 3d single-subject (sub-01) diagnostic: frontopolar variance gap (Search vs. Memorize) ≈ **22.5% relative**; central-parietal variance gap ≈ **2.0% relative** — the motivating observation for this entire ocular-artifact investigation, not yet replicated across all 29 subjects (that replication is F-OCULAR(d), also pending execution).

---

## Gate 1 approval corrections (2026-08-18)

### C1 — F5's geometric-deep baseline is SPDNet, not TSMNet; renamed everywhere

F5's time-boxed geometric-deep baseline in `run_baselines_mdm_tssvm_tslda_csp.py` implements one BiMap layer (Stiefel-constrained, QR-retraction), one ReEig layer, and one LogEig layer feeding a linear classifier — this is SPDNet's core block (**Huang & Van Gool, 2017**), not TSMNet (Kobler et al., 2022). TSMNet's defining component, **SPDDSMBN** (SPD domain-specific batch normalization), is absent from this implementation.

**Decision (option b, per explicit instruction):** renamed everywhere — script identifiers, log labels, JSON result keys (`spdnet_geometric_baseline`), STATUS.md's F5 row, AUDIT.md — to **"SPDNet-style geometric baseline (our implementation)"**, citing Huang & Van Gool (2017). The eventual Table III row must use this name and citation, not TSMNet.

**TSMNet was not re-implemented.** Reason: a faithful TSMNet implementation requires integrating the official `github.com/rkobler/TSMNet` code, which depends on the `geoopt` package (Riemannian-manifold optimization) — a new, unverified pip dependency. This codebase's established convention (see F5's pyriemann/mne deviation note, and the dependency-minimization pattern used throughout: hand-rolled SPD math, hand-rolled CSP, hand-rolled EEGNet max-norm clamp) is to avoid new fragile dependencies in favor of hand-rolled implementations built on already-unit-tested primitives. Shipping the SPDNet-block approximation under TSMNet's name would misattribute results to an architecture that was not actually run.

### C2 — Post-alignment subject-decodability: expected near chance, not below it

Before F3 (Euclidean Alignment) runs on real data: the post-alignment subject-decodability diagnostic (`ea.subject_decodability_accuracy`, used in F5/F8 and the matched-spatial-control drivers) is **expected to land NEAR chance** (chance ≈ 1/29 ≈ 0.034 for 29-way subject classification).

A value **far BELOW chance** is a **bug signal, not a success**, and must **halt and be investigated** — it typically indicates a construction error in the diagnostic itself (e.g., a leaked or degenerate feature that makes subjects anti-correlated with their own identity by construction), not evidence that alignment "worked extra well." Landing *at or near* chance is the expected, healthy outcome; landing *above* chance (residual subject signal survived alignment) is the failure mode the diagnostic exists to catch — both are informative, but a *sub-chance* result is neither.

**Binding constraint:** do not adjust any further test threshold to match an observed value. If a threshold looks wrong once real data is run, raise it at a gate (i.e., report it and ask) rather than silently tightening or loosening it in code. `run_baselines_mdm_tssvm_tslda_csp.py` now hard-raises (`AssertionError`) if post-alignment subject-decodability falls below half of chance level, per this decision.

---

## F-OCULAR(a) additions: matched unablated reference + random-ablation control (recorded 2026-08-19, BEFORE Batch 1's F-OCULAR(a) run)

### A1 — Matched unablated reference

F-OCULAR(a)'s existing 70.78% reference number (`run_step4_matched_spatial_control.py`) is **not a valid comparator** for the frontal-ablation delta: it comes from a different, pre-revision code path (F3/F9-upgraded, parametrized `--ea-mode`/`--cov-estimator`, canonical `SEEDS=[42,43,44,45,46]`), while `run_step4_matched_spatial_control_frontal_ablated.py` predates that upgrade and hardcodes its own pooled-only EA and a different seed list (`[42,101,202,303,404]`). Comparing across these two scripts would confound "ablation effect" with "pipeline-version effect."

**Resolution:** rather than adding `--ea-mode`/`--fusion-mode`/`--cov-estimator` flags to the ablation script (which doesn't have a fusion branch to begin with — it's spatial-only), the script is refactored to run three arms — **unablated (0 channels dropped), frontal-ablated (9 channels), random-ablated (9 channels)** — through one shared per-fold function, in one Modal execution, using the exact same hardcoded EA (`fit_ea_whitening`/`apply_ea_whitening_signal`, pooled-only), the exact same fixed 0.1-shrinkage covariance estimator, and the exact same `SEEDS=[42,101,202,303,404]` for all three arms. "Same code path" is enforced by construction (one function, three channel-keep-lists), not by flag-matching across separate scripts. The output JSON states this explicitly (one `pipeline_settings_shared_across_all_arms` block) so the match is auditable.

### A2 — Random-ablation control, REVISED 2026-08-19 to a null distribution (single fixed channel set rejected)

Frontal ablation drops 9 channels (1953-D → 1431-D tangent vector). An accuracy drop is therefore ambiguous between "ocular signal removed" and "522 fewer features." A matched-count control that drops 9 *non-frontal* channels is required to disambiguate.

**Correction to the original A2 (this same entry, superseded before it was ever run):** the first version of this control used a single fixed 9-channel set — `["Cz", "CPz", "Pz", "CP1", "CP2", "P1", "P2", "C3", "C4"]`, F-OCULAR(d)'s existing "central-parietal" group. This was rejected: that exact group is where Fig. 7 localizes the hypothesized discriminative *cognitive* signal, so ablating it would remove real task signal by construction, guaranteeing a steep drop and forcing the ratio rule to read "frontal ≈ random" — clearing the ocular hypothesis on the basis of a control that could only ever produce that verdict, regardless of whether ocular contamination is actually present. A single fixed non-frontal channel set will always carry this risk (it might land on other signal-bearing regions too), so the fix is not "pick a different fixed 9 channels" but to sample a **null distribution**.

**Revised design — null distribution over 20 random draws (raised from 10, 2026-08-19):**

- **Draw pool** = all 62 channels **MINUS** the frontal/ocular set (`Fp1, Fp2, AF7, AF3, AFz, AF4, AF8, F7, F8` — 9 channels) **MINUS** the central-parietal signal cluster (`C1, C2, C3, C4, Cz, CP1, CP2, CPz, P1, P2, P3, P4, Pz` — 13 channels, deliberately broader than F-OCULAR(d)'s original 9-channel group to exclude the whole hypothesized-signal neighborhood, not just its exact center). Pool size: 62 − 9 − 13 = **40 channels**.
- **20 independent draws** (raised from an initial 10) of 9 channels each from that 40-channel pool, **without replacement within a draw** (draws may overlap with each other). **Reason for the increase:** at 10 draws the smallest achievable one-sided p-value is ~0.09 (1/11), so a frontal drop landing 8th or 9th most extreme of 10 would be uninterpretable and would have forced a full ~3-hour re-run to resolve. 20 draws takes the floor to ~0.048 (1/21) and stabilizes the tail of the distribution the 90th-percentile rule depends on.
- **Fixed, reproducible RNG:** `numpy.random.RandomState(seed=20260819)` (date-coded, chosen now, before any draw is generated or any arm is run), drawing all 20 sets in one deterministic sequence so the exact channel lists are reproducible from this seed alone.
- **Cost control:** each of the 20 draws is run at **ONE seed only** (`seed=42`, the same seed present in the unablated/frontal-ablated arms' 5-seed set, so a same-seed comparison is possible — see below), not the full 5-seed sweep. `unablated` and `frontal_ablated` remain at the full 5 seeds (`[42, 101, 202, 303, 404]`).

**Revised interpretation rule (replaces the ratio rule; percentile- and p-value-based, applied verbatim when results land):**

The frontal-ablation drop and each of the 20 random-draw drops are computed on the **same single-seed (seed=42) basis** — `unablated_acc_at_seed42 − arm_acc_at_seed42`, pooled across all 29 folds — so the comparison is apples-to-apples (frontal's 5-seed robust mean/std is *also* reported, but the percentile test itself uses the seed-42-matched value for consistency with the single-seed random draws).

| Result | Interpretation |
|---|---|
| **one-sided p < 0.10** (equivalently: frontal drop beyond the 90th percentile of the 20 random-draw drops, by RANK) | Frontal-specific effect — consistent with ocular contamination. |
| **one-sided p ≥ 0.10** (frontal drop inside the bulk of the random-draw distribution) | No significant evidence of frontal-specific contamination; a random 9-channel drop of this size or larger is unremarkable under the null. |

**Correction (2026-08-19, caught on the real Batch 1 run):** the rule is gated on the **one-sided p-value**, not a direct comparison against `numpy.percentile(random_draw_drops, 90)`'s interpolated value. The first implementation compared frontal's drop against that interpolated value directly, which is unreliable at N=20: the interpolated 90th-percentile value can sit strictly between two draws' actual values, so a frontal drop that is exactly AT the 90th percentile BY RANK (beaten by only 2 of 20 draws) numerically exceeded the interpolated value and was misclassified as "beyond" it, producing a "frontal-specific" verdict string alongside a non-significant p=0.143 in the same output — an internal contradiction. The rank-based p-value has no such ambiguity, so it is the actual gating statistic; the percentile-language above is retained only because "beyond the 90th percentile" (one-sided α=0.10) is the intuitive framing this control was designed around, and p<0.10 is its exact, unambiguous implementation.

**Both the exact percentile rank AND the one-sided permutation-style p-value are reported explicitly, together, in every interpretation string** — never collapsed to a bare pass/fail without both numbers attached, and never reporting a verdict that contradicts the attached p-value. The p-value uses the same add-one-correction convention already established in this codebase (`scripts/verify_no_leakage.py`'s `permutation_null_stats`): `p = (n_random_draws_at_least_as_extreme_as_frontal + 1) / (N_RANDOM_DRAWS + 1)`, floor ≈ 0.048 at 20 draws. With 20 draws the percentile resolution is still coarse (twentieths, not the fine resolution a 1000-permutation test would give); this is an accepted tradeoff for keeping cost bounded (single-seed draws), not treated as more precise than it is.

**Noise decomposition, reported alongside the interpretation (added 2026-08-19):** the across-seed standard deviation of `unablated`/`frontal_ablated` (5 seeds, same channels — pure seed jitter) is reported side by side with the across-draw standard deviation of the 20 random-draw accuracies/drops (single seed, 20 different channel selections — channel-selection variance only, no seed jitter). These two stds are not directly comparable magnitudes (different sources of variance), but reporting them together prevents mistaking ordinary seed jitter in the 5-seed arms for a channel-driven effect in the null distribution, or vice versa.

### REOPENED (2026-08-19): the single-seed A2 test is not basis-matched — seed 42 is an outlier, not a bug

The p-value/percentile fix above corrected a real string bug, but it did not resolve a deeper problem: the A2 test compares frontal's drop **at seed 42 only** against a null distribution that is **also only computed at seed 42**. This is internally consistent (both sides of the comparison use the same seed) but is **not representative of the 5-seed drop the rest of this control reports**. Direct numbers (unablated/frontal accuracy at seed 42 vs. their 5-seed means): unablated_seed42 = 0.7076 vs. 5-seed mean 0.7146 (seed 42 is a **low** draw for the unablated arm); frontal_seed42 = 0.6970 vs. 5-seed mean 0.6919 (seed 42 is a **high** draw for the frontal-ablated arm). Both deviations point the same way — toward a smaller apparent drop — so seed 42 is, by chance, close to the single seed (of the 5 available) that minimizes the apparent ablation cost. On the 5-seed basis the frontal drop is 2.27 pp, which **exceeds all 20 random-draw drops** (max 1.11 pp, all evaluated at seed 42) by roughly 2×; on the seed-42-only basis the drop is 1.06 pp, landing at the null's 90th percentile with p=0.143.

**This is not the earlier percentile-vs-p-value string bug re-appearing, and it is not an aggregation-formula error** (the algebraic identity proven in RESULTS_LEDGER.md's L005 — that the 5-seed paired-difference mean and the seed-summary mean-of-means must coincide — still holds and is not in question). The issue is a **basis mismatch**: comparing a 5-seed-averaged quantity's single-seed slice against a single-seed null, when that single seed happens to be an outlier on both arms in the direction that minimizes the apparent effect, understates the drop relative to what the more robust 5-seed estimate shows. **Comparing a 5-seed mean drop to a single-seed null is not basis-matched — this is exactly why the per-seed comparison (below) is the correct test, not the single-seed comparison the first two script versions used.**

**Corrected test, replacing the single-seed A2 comparison:** compute the frontal-ablation drop **individually at each of the 5 seeds** (`unablated_acc(seed) − frontal_acc(seed)`, each from that seed's own `seed_summaries` entry — no new Modal computation needed, this data already exists in the completed run's output JSON), and compare **each of the 5 per-seed drops individually** against the existing 20-draw single-seed (seed=42) null distribution (percentile rank + one-sided p-value for each). A single-seed A2 result must never be reported alone again.

**Revised A2 verdict rule (replaces the single-seed rule above):**

| Result | Interpretation |
|---|---|
| **all 5 per-seed drops land beyond the null's 90th percentile** | Ocular effect is REAL; the earlier single-seed CLEAN verdict was a seed artifact. |
| **the 5 per-seed drops straddle the 90th percentile** (some beyond, some not) | Noise-dominated; CLEAN stands as the working verdict, but the full 5-seed spread must be reported — never seed 42 alone — and this should be flagged as inconclusive-at-current-N rather than a confident CLEAN. |

This does not change the null distribution itself (still the 20 draws at seed=42 — re-drawing the null at each of the other 4 seeds would require new Modal computation and is not being requested here); it changes what is compared against that null, from one seed's drop to all five.

This is reported alongside, not instead of, the existing F-OCULAR(a)/(b)/(c) interpretation matrix above — the random-ablation control disambiguates frontal-ablation's OWN accuracy drop; it does not replace the correlation/ICA evidence in (b)/(c) as the decisive test for whether the surviving signal is ocular in nature.

### Standing dependency: this control is conditional on the pre-F3 alignment pipeline

`run_step4_matched_spatial_control_frontal_ablated.py` hardcodes its own pooled-only Euclidean Alignment (predates F3's parametrized `eeg_alignment.py` module) and uses `SEEDS=[42,101,202,303,404]`, not the canonical F4 seed list `[42,43,44,45,46]`. The arms (unablated / frontal-ablated / 20-draw-null) are **internally matched**, so the ablation comparison within this run is valid — but these accuracies are **NOT comparable to any Batch 2+ number** (which will use F3's parametrized alignment), and **the ocular verdict from this control is conditional on the pre-F3 alignment**, not a pipeline-independent fact. **If Batch 2 shows F3 materially changes the spatial pipeline's behavior (e.g., per-subject/riemannian EA modes producing a different accuracy profile than pooled), F-OCULAR(a) must be re-run under the new alignment before its verdict is treated as final.** This is recorded here as a standing dependency (also flagged in STATUS.md's F-OCULAR(a) row and Execution Blockers section, not buried as a footnote), not a one-time caveat that expires once Batch 1 completes.

---

## F-DRIFT — within-session time-drift control (new fix-ID, pre-registered 2026-08-19, BEFORE this script runs)

### Why this control exists

Batch 1 established two facts that together motivate this control:
1. **Every reported Condition-4 accuracy is post-calibration** (STATUS.md's Tier-1 flag): pre-calibration (zero-shot) accuracy is at or near chance (RESULTS_LEDGER.md L009), and the ~19–20 pp gain comes entirely from fitting a personalization step on 15% of the held-out subject's own labeled trials.
2. **F-PARITY found real, exploitable within-session position signal** (ledger L003): position-third decodability of 0.4753 against a 0.3333 chance floor.

Because class label is perfectly confounded with block (D2/F-PARITY: every Search trial precedes every Memorize trial, per subject), and the calibration step personalizes on a small sample of the held-out subject's own trials, it is not yet established whether the calibrated pipeline is decoding **task content** or **time-in-session**. This is the gating question before Batch 2.

### Test design

For each REAL class block (Search-only, Memorize-only) **separately**: split that subject's own chronologically-ordered trials for that block at its midpoint into a pseudo-label — pseudo-class 0 = first (earlier) half, pseudo-class 1 = second (later) half. **Both pseudo-classes are drawn from the SAME real task** (run separately per class, not pooled across classes, so this holds literally for each test) — task content is held constant, so any decodability of the pseudo-label can only come from within-session time-position signal, not task/class content. The identical calibrated LOSO pipeline (byte-identical EA/tangent/PCA/shrinkage-calibration code to the rest of the Batch 1 family) is then run on this pseudo-label, single seed=42, full 29-fold LOSO, same 15% calibration mechanism.

### Pre-registered verdict rule (fixed BEFORE running, applied to POST-calibration mean pseudo-accuracy — the same pipeline stage the real headline numbers are reported at)

| Result | Interpretation |
|---|---|
| **pseudo-class accuracy < 0.55** | Drift is NOT driving the result; task signal is real. |
| **pseudo-class accuracy > 0.65** | The pipeline is a within-session drift detector, and the primary contrast does NOT survive. |
| **0.55 ≤ pseudo-class accuracy ≤ 0.65** | Partial contamination — report both, quantify the share. |

Mean taken across the two pseudo-tests (search_only, memorize_only); both are also reported individually, never collapsed without the per-test numbers attached.

### Caveat, pre-registered 2026-08-19 BEFORE the run reports: the pseudo-test result is a LOWER BOUND, not a full clearance

The pseudo-contrast splits early/late **within one real class block** — both pseudo-classes sit inside the same block, with no gap between them beyond the ordinary inter-trial interval. The REAL Search-vs-Memorize contrast is different in kind, not just degree: it spans the **block boundary itself**, plus (per the dataset's session structure) a **~400 s break** between the two task blocks — strictly more temporal separation, and a physical/procedural discontinuity (task-switch, likely a pause or instruction re-read) that a within-block midpoint split cannot reproduce.

**Consequence, stated explicitly wherever this control's result is reported:** if pseudo-class accuracy is near chance (<0.55, the "task signal is real" branch of the rule above), that result **only rules out fine-grained, within-block drift** as the driver of the real effect. It does **not** clear coarser, between-block drift (e.g., a slow signal or state drift that only meaningfully differs across the ~400 s block-boundary gap, not within a single ~3-minute half-block). Pseudo-class accuracy is therefore a **lower bound** on how much within-session drift is available to the real contrast, not an upper bound or a full accounting. A "task signal is real" verdict from this test alone must be reported as conditional on this scope limitation, not as a general clearance of the drift hypothesis.

### Also pre-registered: accuracy-vs-distance-to-calibration-trial analysis (cheap addition, same run)

Alongside the two pseudo-label tests, the script also runs ONE additional single-seed=42, full-29-fold LOSO pass on the **REAL** Search-vs-Memorize labels (same calibrated pipeline, same 15% split), and for every scored (85%) test trial computes its **temporal distance to the nearest calibration trial of the same real class** — distance measured in trial-count units along that class's own chronological sequence (a class-0 test trial's distance is measured only against class-0 calibration trials, and likewise for class 1; cross-class distance is not meaningful given the block structure). Scored trials are pooled across all 29 folds and binned into **quartiles of this distance**, and accuracy is reported per quartile.

**Interpretation, pre-registered now:** a **monotonically decaying** accuracy from the nearest to the farthest quartile is evidence the calibrated classifier is exploiting local temporal/drift proximity to its own calibration sample rather than genuine class-content generalization ("riding drift"). A **flat** accuracy-vs-distance profile is evidence against that mechanism. This analysis requires no additional folds beyond the one real-label LOSO pass already needed to produce it — it is a grouping of that pass's existing per-trial predictions, not a new experiment.

### Also pre-registered: calibration-split time-distribution diagnostic

Alongside the pseudo-label test, the script instruments the REAL 15% calibration split (same `StratifiedShuffleSplit(test_size=0.85, random_state=42)` mechanism used throughout this codebase) to report how the calibration sample distributes across within-block thirds (early/mid/late) and class. This quantifies how much drift signal the calibration step has structural access to, independent of whether the pseudo-label test finds that signal exploitable. Verified in advance (scratch script, not committed) that `StratifiedShuffleSplit`'s per-class random permutation produces a roughly uniform spread across thirds on synthetic data matching the real block structure (not systematically concentrated in any one third) — this is a sanity check on the mechanism, not a substitute for the real run's actual measured distribution.

### Scope note

This control uses the same pre-F3 pooled-only EA family as F-OCULAR(a)/(c)/F-PARITY, for internal Batch 1 consistency — it therefore inherits the same standing dependency as F-OCULAR(a) (see above): its verdict is conditional on the pre-F3 alignment and must be re-examined if Batch 2's F3 pipeline behaves materially differently.

### RESULT, ACCEPTED (2026-08-19): the >0.65 branch fires — primary contrast does NOT survive

Pseudo-label (early/late within-block, task held constant) post-calibration = 0.7112, pre-calibration = 0.6418. Real-label post-calibration = 0.7078, pre-calibration = 0.5201. Per the pre-registered rule, mean pseudo-accuracy exceeds `DRIFT_DETECTOR_THRESHOLD=0.65` — **the pipeline is a within-session drift detector, and the Search-vs-Memorize contrast does not survive.** A pseudo-label carrying zero task information (both pseudo-classes drawn from the identical real task, differing only in which half of the block a trial falls in) reaches accuracy statistically indistinguishable from — indeed slightly above — the real task-labeled classifier. This is now the controlling finding for this project's Phase 2 scope; see STATUS.md's suspension of Batch 2+ items that compare architectures on this contrast.

### Correction (2026-08-19): the distance-to-calibration-trial quartile analysis is UNINFORMATIVE — do not cite it

Post-hoc review of the real run's quartile ranges (1 / 1–2 / 2–5 / 5–18 trials) shows the analysis had no statistical power to detect what it was designed to detect. At 15% calibration density, calibration trials are interspersed roughly every 6–7 trials on average — no scored trial is ever far from one. The farthest quartile tops out at 18 trials' distance, far too small a dynamic range to distinguish "riding local drift" from "riding block-wide drift" from "genuine task signal." A flat or non-monotone result from this analysis means the test lacked power, **not** that drift is absent — it must never be cited as reassurance against the drift hypothesis, in either direction. **Lesson for any future control of this design:** check the achievable distance dynamic range against the calibration density BEFORE running, not after — a sparser calibration fraction or a coarser distance unit (e.g., block-relative position rather than raw trial count) would be needed to give this kind of test real power.

---

## F-DRIFT-B — interleaved (odd/even) pseudo-label control (pre-registered 2026-08-19, BEFORE this script runs)

### Why this control exists

F-DRIFT's accepted result shows the calibrated pipeline achieves ~0.71 accuracy on a pseudo-label carrying zero task information, using an early/late within-block split. That split still has SOME temporal separation (up to half a block's worth of trials between the two pseudo-classes' centers). F-DRIFT-B removes temporal separation almost entirely, to test whether the effect specifically requires temporal separation, or whether it is some other artifact of the pseudo-labeling/calibration mechanism unrelated to time.

### Test design

Pseudo-class = **odd-numbered vs. even-numbered trials, by within-class chronological position**, within each real class block (Search-only and Memorize-only, run separately — same task, same stimuli statistics, same block, matching F-DRIFT's per-class design). Adjacent trials in the original chronological sequence always have different pseudo-labels — this is the minimum possible temporal separation between the two pseudo-classes (zero net separation once averaged: pseudo-class 0 and pseudo-class 1 trials are uniformly interleaved throughout the block, not concentrated in an early half vs. a late half). Structurally identical to F-DRIFT otherwise: identical calibrated LOSO pipeline (byte-identical EA/tangent/PCA/shrinkage-calibration), single seed=42, full 29-fold LOSO, same 15% calibration mechanism, both real classes tested separately.

### Pre-registered verdict rule (fixed BEFORE running, applied to POST-calibration mean pseudo-accuracy across the two per-class tests)

| Result | Interpretation |
|---|---|
| **interleaved accuracy < 0.55** (chance) while F-DRIFT's early/late accuracy sits at ~0.71 | Confirms the effect is specifically driven by TEMPORAL SEPARATION, not some other pseudo-labeling artifact. **This pair (F-DRIFT vs. F-DRIFT-B) becomes the central evidence** for the drift-detector conclusion. |
| **interleaved accuracy also high (> 0.65)** | The mechanism is NOT drift (an interleaved split has no meaningful temporal-separation signal to exploit) — **HALT and report; every prior interpretation in this document needs re-examination**, since the pipeline would be exploiting something other than time-position that this audit has not yet identified. |
| **in between (0.55–0.65)** | Report both, do not interpret further without discussion. |

Report `pre_calibration_acc` and `post_calibration_acc` for both F-DRIFT-B tests (search_only, memorize_only) alongside F-DRIFT's existing numbers in one combined table — never F-DRIFT-B numbers alone, since the whole point is the F-DRIFT vs. F-DRIFT-B comparison.

### RESULT, ACCEPTED (2026-08-19): <0.55 branch fires — confirmatory half of the drift finding

Interleaved pre_cal=0.5085, post_cal=0.5023 (chance) — confirms the effect is specifically driven by temporal separation, not some other pseudo-labeling artifact. See RESULTS_LEDGER.md L014 for the full combined table (F-DRIFT-B / F-DRIFT / real labels) and the shrink=0.00 observation (calibration only pays off when a temporal gradient exists — rules out "calibration itself is the artifact"). **B2 (random-assignment interleaving control) is CANCELLED** (ledger L015) — its purpose was to catch a periodicity artifact inflating B1, and B1 landed at chance, so there is nothing for that explanation to account for.

---

## F-DRIFT-C — parametric dose-response sweep (pre-registered 2026-08-19, BEFORE this script runs)

### Why this control exists

F-DRIFT and F-DRIFT-B together establish that decodability depends on temporal separation (F-DRIFT: ~half-block separation → ~0.71; F-DRIFT-B: ~zero separation → chance), but only at two points. F-DRIFT-C parametrizes the separation continuously to test whether decodability scales smoothly with temporal separation (the central claim needed for the methodological paper) or jumps/plateaus in a way inconsistent with a simple drift account.

### Test design

Pseudo-label defined by **alternating runs of length k** within each real class block: `pseudo_label(i) = (i // k) % 2`, where `i` is the 0-indexed within-class chronological trial position. `k=1` is exactly F-DRIFT-B's interleaving; `k≈n/2` (n≈200 trials/class/subject, so `k=100`) is exactly F-DRIFT's early/late split. Sweep `k ∈ {1, 2, 5, 10, 25, 50, 100}`. Both real classes (Search-only, Memorize-only) tested separately, single seed=42, full 29-fold LOSO, identical calibrated pipeline, for every k.

**Endpoint equivalence check (mandatory, HALT if it fails):** k=1 and k=100 must reproduce F-DRIFT-B (pre=0.5085, post=0.5023) and F-DRIFT (pre=0.6418, post=0.7112) respectively, within noise (tolerance: ±0.03 on post-calibration accuracy, matching this codebase's established single-seed noise band — see F-LEAK's ±0.03 mean-band precedent). If either endpoint falls outside tolerance, the script HALTS (hard assertion failure) before any interpretation is attempted — the parameterization would not be equivalent to the already-accepted controls, and the curve would not mean anything until debugged.

**Known, expected source of approximation at k=100 (verified locally before running, not a bug):** `pseudo_label(i) = (i // k) % 2` is a literal "alternating runs of length k" formula, applied with a FIXED k across all 29 subjects. Real per-class trial counts vary (~190–220, not exactly 200), so for any subject with n > 2k, the formula produces a small THIRD partial run at the tail (wrapping the pseudo-label back to 0) that F-DRIFT's original per-subject `n // 2` midpoint split does not have. Confirmed via scratch verification (not committed): k=1 is an EXACT match to F-DRIFT-B for every n; k=100 is only an approximate match to F-DRIFT's per-subject split, with per-subject trial-level disagreement growing as n departs from 200 (0% at n=200, up to ~14% at n=220). This is exactly why the endpoint check validates equivalence at the AGGREGATE cross-subject accuracy level (±0.03), not per-trial — a small aggregate difference from this tail effect is expected and within tolerance; only a large one should be read as a real parameterization problem.

**Reporting basis:** `pre_calibration_acc` is the PRIMARY axis (the clean, uncontaminated-by-within-subject-calibration measure of what transfers across subjects) — `post_calibration_acc` is also reported for every k, but pre_cal is what the interpretation rule below is applied to.

**Seconds conversion:** k (a trial count) is converted to mean temporal separation in seconds using REAL trial onset times, extracted via `mne.events_from_annotations` on each subject's raw `.vhdr`/`.vmrk` (the exact same `EVENT_ID` mapping already used in production — `run_data_engine_on_modal.py`'s `EVENT_ID` dict, `"Stimulus/ 10"`..`"Stimulus/ 23"` → 0/1), NOT approximated or assumed. Per-subject, per-class mean inter-trial interval (ITI) is computed from these real onsets; `seconds_at_k = k × mean_ITI_seconds` (cross-subject mean ITI, with its own std reported for transparency). **Cross-check, mandatory:** the number of stimulus events found via this independent onset-time extraction must match the known per-subject/class trial counts already in `processed_eeg_all_subjects.npz` — a mismatch means the onset-time extraction is misaligned with the processed dataset and must not be trusted.

**Plot:** pre_cal vs. seconds of separation, with the real-label pre_cal (0.5201) marked as a horizontal reference line, saved as a PNG artifact alongside the JSON.

### Pre-registered interpretation rule (fixed BEFORE running)

| Result | Interpretation |
|---|---|
| **Monotone rise from chance toward ~0.64** (F-DRIFT's pre_cal endpoint) as seconds of separation increase | Decodability scales with temporal separation — **this is the central figure of the methodological paper.** |
| **Flat or non-monotone** | The drift account is incomplete. Report plainly and stop for discussion before interpreting further — do not construct a post-hoc explanation for a non-monotone curve. |

### Cost

~9 min per k per real class (7 k-values × 2 classes = 14 LOSO passes × 29 folds, single seed) — budgeted ~1.5–2 hours total, plus the one-time onset-time extraction pass (29 subjects' `.vhdr`/`.vmrk`, downloaded via the same `openneuro.download` mechanism already used by F-OCULAR(a)/(c)/(d) — full per-subject download, since this codebase's `include` parameter has only been observed to operate at subject-folder granularity, not file-type granularity; correctness over download-size optimization, since a subtly-wrong custom `.vmrk` text parser would silently corrupt the whole dose-response curve without raising an error).

### RESULT (2026-08-20): pre-registered strict-monotonicity criterion NOT MET — criterion mis-specification disclosed, revised trend analysis added

**Curve as actually reported (single-seed=42, pre_cal, the primary axis):** flat at chance for separations ≤116s (k=1,2,5,10,25), steps to **0.6358 at 232s** (k=50), plateaus at **0.6413 by 465s** (k=100). This is a clean threshold/step function.

**Disclosure (user-identified, own error, recorded per this codebase's full-disclosure discipline — never a silent substitution):** the pre-registered `is_monotone_rising` check (`run_step4_drift_control_c_dose_response.py`, strict pointwise `curve[i] <= curve[i+1] + 1e-9` for all consecutive pairs) is **mis-specified for 7 noisy single-seed points** — it fails on small (0.007–0.011) wiggles at k=2 and k=25 even though the true underlying pattern is an unambiguous step function. **The original strict-monotonicity verdict is `NOT MET` and stands on the record exactly as computed — it is not deleted, replaced, or silently corrected.**

**Required disclosure language (verbatim, use wherever F-DRIFT-C is cited): "our pre-registered criterion was mis-specified; we report both the original verdict and a revised trend analysis."**

**Revised trend analysis (POST-HOC — NOT pre-registered before the original script ran; added only after the mis-specification was found; pure post-processing of the same already-completed dose-response data, no new LOSO training):**
- **(b) Spearman rank correlation** between seconds-of-separation and combined pre_cal accuracy across the 7 k-values, with a **subject-level bootstrap CI** (resample the 29 subjects with replacement, N=2000, recompute the 7-point cross-subject-mean curve and its Spearman rho per resample, CI = 2.5/97.5 percentile of the resulting rho distribution). One-sided p (H1: rho > 0) reported alongside the two-sided p. **RESULT (2026-08-20): rho = 0.8571, one-sided p = 0.0068, subject-level bootstrap 95% CI = [0.6786, 0.9643].**
- **(c) Two-group paired contrast:** k∈{1,2,5,10,25} (≤116s) vs. k∈{50,100} (≥232s), per-subject paired (each subject contributes one low-separation mean and one high-separation mean pre_cal), Wilcoxon signed-rank test, one-sided (H1: high-separation > low-separation) and two-sided both reported. **RESULT (2026-08-20): low-sep mean = 0.5112, high-sep mean = 0.6386, one-sided Wilcoxon p = 3.7e-09.**
- **Implementation:** `scripts/drift_c_posthoc_analysis.py` (reads `results_condition4_drift_control_c_dose_response.json` from the volume, writes `results_condition4_drift_control_c_posthoc.json`; no new model training, cheap). Locally logic-verified against a synthetic step-function dataset matching the real curve's shape (flat-then-step, per-subject noise) before being handed off — confirmed Spearman correctly detects the trend despite the same tie-heavy-ranks structure that made strict pointwise monotonicity fragile (this is exactly why rank correlation is the right revised tool: it is robust to the small within-flat-region wiggles that broke the original criterion), bootstrap CI well-behaved within [-1,1], two-group Wilcoxon cleanly significant on the synthetic step pattern. **RUN 2026-08-20 — matches the design intent: both tests strongly confirm the trend the pointwise criterion missed.**
- **Both the original NOT MET verdict and this post-hoc trend analysis must always be reported together** — see RESULTS_LEDGER.md L016.

---

## F-PARITY-WITHIN — the missing cell of the parity design (pre-registered 2026-08-19, BEFORE this script runs)

### Why this control exists

The real Search-vs-Memorize contrast has the LARGEST temporal separation of anything tested (block boundary + ~400s break) yet the WORST cross-subject transfer measured so far (real-label pre_cal=0.5201, vs. F-DRIFT's within-block-split pre_cal=0.6418 — a smaller separation giving BETTER transfer is the opposite of what a pure "more separation = more decodable" story predicts, unless the cross-subject aggregation itself is cancelling the signal). **Hypothesis:** F-PARITY's odd/even block-order counterbalancing (D2: odd subjects Search-first, even subjects Memorize-first) makes "early trial" mean Class 0 for half the training pool and Class 1 for the other half — when real labels are trained on a MIXED-parity pool (as every other driver in this codebase does), the time-position cue cancels out across subjects, suppressing exactly the kind of temporal signal that F-DRIFT/F-DRIFT-B show the pipeline is otherwise very good at exploiting. Within a SINGLE parity group, the time-position cue is consistent across all training subjects (no cancellation), so if the hypothesis is right, within-parity LOSO should recover much higher pre_cal accuracy.

### Test design

Two separate LOSO runs, REAL labels, identical calibrated pipeline, single seed=42:
- **Odd-only:** 14-fold LOSO among the 14 odd-numbered subjects only (all Search-first).
- **Even-only:** 15-fold LOSO among the 15 even-numbered subjects only (all Memorize-first).

Report `pre_calibration_acc` and `post_calibration_acc` for each.

### Pre-registered interpretation rule (fixed BEFORE running)

| Result | Interpretation |
|---|---|
| **Within-parity real-label pre_cal rises toward ~0.64** while the already-measured cross-parity value stays near 0.49 (F-PARITY, ledger L003: 0.4927 train-odd-test-even / 0.4750 train-even-test-odd) | **Hypothesis CONFIRMED** — the apparent task signal is block-order/time, visible when block order is consistent across the training pool and cancelling when it is not. |
| **Within-parity pre_cal stays near 0.52** | **Hypothesis REJECTED** — the asymmetry between real-label and pseudo-label transfer needs another explanation, and this hypothesis must NOT be asserted in the write-up. |

Report alongside F-DRIFT-C, not separately — both address the same underlying question (does temporal/block structure explain the transfer pattern) from complementary angles.

### RESULT, REJECTED (2026-08-20): within-parity pre_cal stays near mixed-parity value

**Numbers:** within-parity pooled pre_cal = **0.5303** (odd-only = 0.5506, even-only = 0.5114) vs. mixed-parity real-label pre_cal = **0.5201** (F-PARITY, ledger L003 basis). 0.5303 falls within the pre-registered REJECT band (`|0.5303 − 0.5201| = 0.0102 ≤ 0.03`), nowhere near the ~0.64 CONFIRM threshold.

**Verdict: HYPOTHESIS REJECTED.** Block-order/time-cancellation across the mixed-parity training pool does **not** explain why the real-label contrast transfers worse than F-DRIFT's smaller within-block separation. **Hard note: block-order/time-cancellation must NOT be asserted anywhere in the write-up as the explanation for the real-vs-pseudo transfer asymmetry** — this specific explanation is now closed off, per the pre-registered rule above.

**Does NOT weaken the core drift finding.** F-DRIFT (ledger L011) and F-DRIFT-B (ledger L014) — the controlling result that the Search-vs-Memorize contrast is a within-session drift detector — rest on **POST-calibration** numbers (pseudo post_cal ≈0.71 vs. real post_cal 0.7078, both ledger L011/L014). F-PARITY-WITHIN's rejection concerns only a secondary asymmetry in the **PRE-calibration** column (why real pre_cal=0.52 sits below F-DRIFT's pseudo pre_cal=0.64, rather than block-order explaining it away). These are separate claims about separate columns of the same table — see STATUS.md's explicit clarification note and RESULTS_LEDGER.md L018.

**Motivates F-DRIFT-D** (below): if block-order counterbalancing isn't the explanation, the next candidate is that drift is phase-locked to block onset and resets at the ~400s break, rather than accumulating across the whole session.

---

## F-DRIFT-D — phase-matched block contrast, drift-reset hypothesis (pre-registered 2026-08-20, BEFORE this script runs)

### Why this control exists

F-PARITY-WITHIN's rejection (above) rules out block-order/time-cancellation as the explanation for why the real-label contrast (largest temporal separation: block boundary + ~400s break) transfers worse (pre_cal=0.5201) than F-DRIFT's within-block early/late split (smaller separation, pre_cal=0.6418). **New hypothesis:** the drift signal is phase-locked to BLOCK ONSET and RESETS at the ~400s inter-block break, rather than accumulating monotonically across the whole session. Under this hypothesis, within-block position (early vs. late) is highly decodable (matching F-DRIFT's 0.6418) precisely because it's measured relative to each block's own onset, while raw "which block" identity is much less decodable once within-block position is controlled for, because both blocks independently ramp through the same drift trajectory rather than one continuing where the other left off.

### Test design

**Phase-matched block contrast:** label = real block/class identity (0=Search, 1=Memorize, same as the real Search-vs-Memorize label) — but every trial is additionally tagged with its **within-block position third** (early/mid/late), computed with the SAME chronological-thirds logic already used in `run_step4_parity_split_control.py`'s `compute_within_block_thirds` (`thirds = np.minimum((np.arange(n) * 3) // n, 2)`, per subject per class). Three separate binary classification problems, one per phase:
- **Early:** early-third-of-Search-block vs. early-third-of-Memorize-block trials only.
- **Mid:** mid-third vs. mid-third.
- **Late:** late-third vs. late-third.

Each phase run through the SAME calibrated LOSO pipeline (identical EA/tangent/PCA/shrinkage-calibration to the rest of the pre-F3 F-DRIFT family), single seed=42, full 29-fold LOSO. Report `pre_calibration_acc`/`post_calibration_acc` per phase (early/mid/late) AND pooled (mean across the three phases).

This isolates "does block identity carry cross-subject signal once within-block position is held constant" — the confound F-PARITY-WITHIN's real-label test could not separate, since it trained on the full block (all positions mixed) with only parity grouping as the manipulated variable.

### Pre-registered interpretation rule (fixed BEFORE running, stated in accuracy terms, NOT as a strict ordering — per the F-DRIFT-C lesson above that strict ordering across noisy single-seed points is not a robust criterion)

| Result (pooled phase-matched pre_cal) | Interpretation |
|---|---|
| **< 0.55** (near chance) while within-block early/late (F-DRIFT) sits at 0.6418 | Drift is phase-locked to block onset and **RESETS at the break** — **hypothesis SUPPORTED.** |
| **≥ 0.60** | Block identity carries cross-subject signal beyond position — **hypothesis REJECTED**; the real-vs-pseudo transfer asymmetry still needs another explanation. |
| **0.55–0.60** | Inconclusive. Report plainly and stop for discussion before interpreting further. |

Report per-phase (early/mid/late) numbers alongside the pooled verdict — a phase-dependent pattern (e.g., early phase near chance, late phase elevated) would itself be informative and must not be collapsed away by the pooled number alone.

### Cost

3 phases × 29 folds, single seed = 87 LOSO folds total (no per-class split needed, since class label is the direct target here) — comparable in cost to F-PARITY-WITHIN (~10–20 min).

### RESULT (2026-08-20): pre_cal supports the pre-registered rule — but the script's own verdict string under-reported the main finding; corrected here with full disclosure

**The pre-registered rule above was applied mechanically and correctly** to `pooled_pre_calibration_accuracy` by `run_step4_drift_control_d_phase_matched.py`'s own verdict logic: pooled pre_cal = 0.5083 < 0.55 → SUPPORTED. That computed verdict stands and is not wrong on its own narrow terms. **But it is incomplete**, because the pre-registered rule was scoped only to pre_cal and never asked the script to surface `pooled_post_calibration_accuracy` — which turns out to be the more consequential number. Per instruction, both are now recorded together, and neither is allowed to stand without the other going forward:

- **(a) pre_cal = 0.5083 pooled → SUPPORTED per the rule above.** Per-phase breakdown, reported honestly rather than collapsing to the pooled figure alone: **early = 0.5603** (individually ABOVE the 0.55 threshold), **mid = 0.4755**, **late = 0.4890** (both below). The pooled figure clearing the threshold masks real phase-to-phase heterogeneity — the early phase alone would not have supported the hypothesis on its own.
- **(b) post_cal = 0.7024 pooled** (early=0.7135, mid=0.7210, late=0.6728) — **essentially unchanged from the unrestricted real-label contrast's post_cal (0.7078, L009/L011).** This was never part of the pre-registered rule (which only thresholds pre_cal) but is the more important result: **removing the within-block-position cue does not reduce calibrated accuracy at all.**
- **Combined reading:** (a) says block identity is not zero-shot decodable once within-block position is matched out (consistent with a drift-resets-at-the-break account). (b) says the ~0.70 calibrated headline number does not depend on within-block position being available as a cue — it survives the cue's removal intact, meaning the 15%-calibration step's per-subject adaptation is doing the real work, not a cross-subject-transferable temporal-position signal. This directly reinforces the Tier-1 finding (STATUS.md) that the reported accuracy is a few-shot subject-adaptive result.
- **See RESULTS_LEDGER.md L020** for the full numeric record and disclosure.

---

## F-DRIFT-E — boundary-privilege check, shifted temporal partition (pre-registered 2026-08-20, BEFORE this script runs)

### Why this control exists

The master evidence table (RESULTS_LEDGER.md L021) shows pseudo-labels with zero task content reach the same calibrated accuracy as the real Search-vs-Memorize labels at comparable temporal separation. A reviewer's next question: is the TRUE task/block boundary itself special — carrying real information a generic temporal split would not — or does ANY partition of the session at a similar separation from the boundary yield the same result? F-DRIFT-E answers this directly rather than leaving it as an inference from the table.

### Test design

Split point placed at **25%, 50%, and 75% through block 1**, and **symmetrically at 25%, 50%, and 75% through block 2** — **6 shift positions total**, instead of the true block boundary. "Through block N" = that fraction of the way through block N's own trial count, using each subject's natural array storage order (already treated as chronological throughout the F-DRIFT family — F-DRIFT/F-DRIFT-B/F-DRIFT-C/F-DRIFT-D all rely on the same assumption for their within-block early/late and thirds constructions). Trials strictly before the shift point form pseudo-class 0; trials at/after it form pseudo-class 1. Each shift run through the SAME calibrated LOSO pipeline, single seed=42, full 29-fold LOSO.

**Disclosed methodological property, not a bug:** away from the true boundary, each pseudo-class is necessarily a MIXTURE of both real classes (e.g. a block-1@25% shift's pseudo-class-1 contains the remaining 75% of block 1 plus all of block 2, spanning both real tasks). This is intentional and is exactly what the test needs — it isolates temporal position from task identity.

### Pre-registered interpretation rule (fixed BEFORE running, accuracy terms, NOT strict ordering — per F-DRIFT-C's mis-specification lesson)

| Result (per shift position, post_cal) | Interpretation |
|---|---|
| **Within 0.05 of the true-boundary reference (0.7078, RESULTS_LEDGER.md L009/L011)** | The true boundary is NOT privileged — any temporal partition at similar separation yields the same calibrated accuracy. **Strongest form of the drift-detector finding.** |
| **Below 0.60 while the true boundary is 0.7078** | The true boundary IS privileged — it carries information a generic temporal split does not. **Report and stop for discussion; this would materially change the conclusion.** |
| **In between** | Report the full 6-point curve. No further interpretation without discussion. |

Apply per shift position (6 verdicts), plus note whether the pattern is uniform (all 6 land in the same band) or heterogeneous (e.g. shifts near the true boundary behave differently from shifts far from it) — do not collapse to a single pooled number given F-DRIFT-D's lesson that pooling can mask phase-dependent heterogeneity.

### Cost

6 shifts × 29 folds = 174 LOSO folds, single seed — comparable to less than half of F-DRIFT-C's cost (14×29=406 folds).

### RESULT (2026-08-20): **INVALID-DESIGN — do not cite any verdict from this run**

**The design above has a real flaw, caught by the user on review, not by this script's own checks:** letting pseudo-class size vary freely with shift fraction and block length produces severe class imbalance at every non-boundary position (as extreme as 1451/10159 trials at the 25%/75% shifts). Accuracy at every position tracked the majority-class base rate, not any genuine boundary-privilege signal:

| Position | Base (majority) rate | pre_cal | post_cal |
|---|---|---|---|
| block1_25pct | 0.8750 | 0.8673 | 0.8870 |
| block1_50pct | 0.7500 | 0.7422 | 0.7780 |
| block1_75pct | 0.6250 | 0.6180 | 0.7003 |
| block2_25pct | 0.6250 | 0.6272 | 0.7490 |
| block2_50pct | 0.7500 | 0.7386 | 0.7648 |
| block2_75pct | 0.8750 | 0.8765 | 0.8710 |

`block2_75pct` pre_cal = 0.8765 ± 0.0008 with `best_shrink_weight=0.00` on nearly every fold — the classifier predicted the majority class for every trial of every subject; its post_cal (0.8710) is BELOW the base rate. **The two positions that happened to land within 0.05 of the true-boundary reference (block1_25pct, block2_75pct) did so by coincidence of their 0.8750 base rate, not because the shifted boundary carries comparable information to the true one.** Every per-shift verdict this run produced is void.

**This does not touch F-DRIFT/F-DRIFT-B/F-DRIFT-C/F-DRIFT-D or either parity cell** — F-DRIFT-E is an add-on control, and none of those constructions have this failure mode (their classes are near-balanced by design; FIX 1 below now confirms this mechanically instead of leaving it assumed).

**Logged, not deleted, per this codebase's disclosed-failed-control discipline — see RESULTS_LEDGER.md L023.**

### FIX 1 — C3 hardening, codebase-wide (applies to every result-emitting script)

"All accuracies in [0,1]" is a vacuous plausibility check — it passed a run that was pure base rate. **Going forward, every classification result must additionally assert and print:** the class balance of the labels being classified, the majority-class rate, accuracy minus majority-class rate, and balanced accuracy. **Fail loudly (hard assertion) if class balance falls outside 45/55, unless the script explicitly declares itself an imbalanced design (a `DECLARED_IMBALANCED_DESIGN=True` constant) and reports balanced accuracy as its primary metric instead of raw accuracy.** Applied now to the full active F-DRIFT/F-PARITY family: `run_step4_drift_control.py`, `run_step4_drift_control_b_interleaved.py`, `run_step4_drift_control_c_dose_response.py`, `run_step4_drift_control_d_phase_matched.py`, `run_step4_parity_split_control.py`, `run_step4_parity_split_control_within.py`, and the F-DRIFT-E redesign below. **Standing requirement, recorded here as binding: any OTHER result-emitting script in this codebase (the Batch 2+ `BLOCKED-PENDING-REFRAME` scripts included) must receive this same hardening before it is ever run for a number that will be cited** — see STATUS.md's Execution Blockers section.

---

## F-DRIFT-E REDESIGN — fixed symmetric window (pre-registered 2026-08-20, BEFORE this script runs)

### Why this redesign exists

The original F-DRIFT-E design (above) confounded shift position with class balance, making its accuracy numbers uninterpretable (RESULT above, ledger L023). This redesign controls for N and balance exactly, so split LOCATION is the only variable across conditions.

### Test design

For each split position, take **W trials immediately BEFORE and W trials immediately AFTER** the split point. **W is a single constant across ALL positions**, set to the largest value feasible at the most extreme shift (~1400, per the 25% positions' available headroom before running out of trials on the short side). Every condition then has identical N (2W), identical 50/50 balance, and identical temporal span — only split location varies.

**7 positions:** 25%/50%/75% through block 1; 25%/50%/75% through block 2; **and the true boundary itself, under the SAME windowing** (W trials before, W after — NOT the full-data 0.7078 reference, which uses ~4x more trials at ~29-fold scale and is not comparable to a matched-N measurement). Same calibrated pipeline, seed=42, full 29-fold LOSO per position (7 × 29 = 203 folds).

**Reported per position:** pre_cal, post_cal, balanced accuracy, N, class balance, majority rate, and the temporal span the window covers in seconds.

### Pre-registered interpretation rule (fixed BEFORE running, BALANCED-ACCURACY terms, NOT strict ordering)

| Result | Interpretation |
|---|---|
| **All shifted positions within 0.05 of the true-boundary-at-W value** | The true boundary is NOT privileged — any temporal partition at matched N/balance/span yields the same calibrated accuracy. |
| **True-boundary-at-W exceeds every shifted position by > 0.05** | The true boundary IS privileged — it carries information a generic temporal split does not. **Report and stop for discussion; would materially change the conclusion.** |
| **Mixed** | Report the full 7-point profile across positions. No further interpretation without discussion. |

### Cost

7 positions × 29 folds = 203 LOSO folds, single seed — comparable to half of F-DRIFT-C's cost.

### RESULT (2026-08-20): pre-registered verdict MIXED; POST-HOC — accuracy tracks proximity to a BLOCK ONSET, not the task boundary

**Pre-registered verdict (balanced accuracy, applied mechanically per the rule above): MIXED.** 2 of 6 shifted positions (`block1_25pct`, `block2_25pct`) land within 0.05 of the true-boundary-at-W reference (0.8968); the other 4 (`block1_50pct`, `block2_50pct`, `block2_75pct`, `block1_75pct`) do not — neither the all-within-tolerance nor the all-exceed-margin branch fires. Recorded as computed, not restated further.

**⚠ POST-HOC (explicitly labelled — discovered by inspecting the 7-point table after the run, not pre-specified beforehand): balanced accuracy sorts by whether the window pair CONTAINS A BLOCK ONSET, not by proximity to the real task boundary.**

| Group | Position | Balanced accuracy |
|---|---|---|
| Contains a block onset | true_boundary | 0.8968 |
| Contains a block onset | block2_25pct | 0.8667 |
| Contains a block onset | block1_25pct | 0.8585 |
| Steady state | block1_50pct | 0.5557 |
| Steady state | block2_50pct | 0.5506 |
| Steady state | block2_75pct | 0.5488 |
| Steady state | block1_75pct | 0.5405 |

**Headline: `block1_25pct` lies ENTIRELY WITHIN block 1** — same task, same instruction, same stimuli, zero task content, zero block-boundary crossing — **yet reaches 0.8585 against the real task boundary's 0.8968, at matched N, balance, and 465s span.** By construction of the fixed window at this position, the window's near edge sits at or adjacent to block 1's own trial-0 onset — this is the ONLY structural feature it shares with the true boundary (which straddles block 2's onset) and with `block2_25pct` (which straddles the same true-boundary onset from the other side). **Added to RESULTS_LEDGER.md's master table (L021) as the strongest single control in this audit — see L025.**

**This changes the mechanism, not just confirms or rejects the original question.** The original F-DRIFT-E question ("is the true boundary privileged?") is answered MIXED, but the post-hoc pattern reframes it: what appears privileged is not the task boundary per se, but ANY block onset, including one with zero task content. **This directly motivates two follow-ups, both executed now:** (1) F-DRIFT-C's temporal-separation interpretation is reopened — its `pseudo_label(i)=(i//k)%2` construction always puts onset trials in label 0, so its dose-response curve may reflect onset CONCENTRATION rather than separation (see F-DRIFT-C reinterpretation flag, below). (2) F-DRIFT-F, below, directly tests onset-exclusion to resolve which mechanism is operative.

**Extending past the previously declared final control (recorded explicitly, per instruction): F-DRIFT-E was declared the final control before this run. The redesign CHANGED THE MECHANISM under discussion rather than confirming or cleanly rejecting the original hypothesis, so one more control (F-DRIFT-F, below) is authorized to close the question either way. F-DRIFT-F is final regardless of its own outcome — no further experiments follow it.**

---

## F-DRIFT-C — reinterpretation flag: "temporal separation" marked PROVISIONAL pending F-DRIFT-F (recorded 2026-08-20)

**Not a retraction.** F-DRIFT-C's post-hoc trend numbers (Spearman rho=0.8571, one-sided p=0.0068, bootstrap CI [0.6786,0.9643]; Wilcoxon p=3.7e-09 — RESULTS_LEDGER.md L016) stand exactly as computed. **What is now PROVISIONAL is the interpretation of what mechanism drives the trend.**

**The confound (identified from F-DRIFT-E REDESIGN's post-hoc onset pattern, above):** under `pseudo_label(i) = (i // k) % 2`, label 0 always contains trials `0..k-1` of each within-class block — i.e., **label 0 always contains that block's onset trials.** At k=100, the onset transient occupies half of one entire pseudo-class (maximally concentrated). At k=25, the onset region is diluted across four alternating runs. At k=1, it is spread evenly across both pseudo-classes with no concentration at all — consistent with F-DRIFT-B's chance result at that same k. **The F-DRIFT-C curve may therefore be a dose-response in ONSET CONCENTRATION within pseudo-class 0, not a dose-response in temporal SEPARATION between pseudo-classes.**

Both accounts are still "drift" in the general sense (a within-session, non-task-content signal) — but the specific mechanism differs, and it materially changes what the eventual write-up may claim. **The write-up must NOT assert "separation" as the driver if "onset concentration" is what actually drives the effect.**

**Status: the temporal-separation interpretation of F-DRIFT-C is PROVISIONAL, effective immediately.** Resolution deferred to F-DRIFT-F, below, which directly and independently tests onset-exclusion. RESULTS_LEDGER.md L026 records this flag against L016.

---

## F-DRIFT-F — onset-exclusion test (pre-registered 2026-08-20, BEFORE this script runs) — FINAL CONTROL

### Why this control exists

F-DRIFT-E REDESIGN's post-hoc pattern (above) and F-DRIFT-C's resulting reinterpretation flag both point at the same open question: is the drift effect driven by temporal SEPARATION between pseudo-classes, or by ONSET CONCENTRATION (how much of pseudo-class 0 consists of block-onset trials)? F-DRIFT-F tests this directly by removing the onset region from the data before re-running both constructions.

### Test design

**(a) Onset-excluded k-sweep.** Re-run the F-DRIFT-C k-sweep (`pseudo_label(i) = (i // k) % 2`, k ∈ {1,2,5,10,25,50,100}, both real classes separately) after **dropping the first 50 trials of each block for every subject**, then **recomputing k relative to the truncated (onset-excluded) sequence** (i.e., `i` is re-indexed from 0 within the remaining ~140–170 trials). Same calibrated pipeline, seed=42, full 29-fold LOSO per (k, class). Report **BALANCED** pre_cal and post_cal per k (per FIX 1's hardening — this is now the primary metric for every new control in this audit).

**(b) Onset-distance parametric sweep.** Split positions at trial distances **{25, 50, 75, 100, 125, 150, 175} from each block's own onset** (block 1's onset = session trial 0; block 2's onset = the true boundary, trial N1) — **7 distances × 2 blocks = 14 positions.** Same **fixed symmetric window** design as the F-DRIFT-E redesign (W trials before/after the split point, W computed from data as the largest value feasible at the tightest position — NOT hardcoded; will likely come out smaller than F-DRIFT-E's W=48 given distance=25 is the tightest constraint here). Same calibrated pipeline, seed=42, full 29-fold LOSO per position (14 × 29 = 406 folds). Report balanced post_cal accuracy per (distance, block), plus a POOLED-by-distance value (mean of block1/block2 at the same distance) as the primary curve.

### Pre-registered interpretation rule (fixed BEFORE running, BALANCED-ACCURACY terms)

**(a):**

| Result | Interpretation |
|---|---|
| **Onset-excluded k-sweep collapses to chance (<0.55 balanced) at every k** | The onset transient explains the entire drift effect. **The F-DRIFT-C separation interpretation is WITHDRAWN and replaced by an onset-transient account.** |
| **Onset-excluded k-sweep retains a rise (highest-k balanced accuracy ≥ 0.62)** | Temporal separation contributes independently of onset concentration. **Both mechanisms are reported.** |

**(b):** operationalized via **Spearman rank correlation** between distance-from-onset and the pooled-by-distance balanced post_cal curve (7 points) — chosen deliberately over a strict pointwise-ordering check, **per the F-DRIFT-C mis-specification lesson (DECISIONS.md's F-DRIFT-C RESULT section)** that strict ordering across a handful of noisy single-seed points is fragile and not a robust operationalization of "decays/rises with X."

| Result | Interpretation |
|---|---|
| **rho ≤ −0.6 and one-sided p < 0.05 (H1: rho < 0)** | Accuracy decays with distance from onset — **confirms the onset account.** |
| **rho not meeting that bar (near zero / not significant)** | Accuracy is flat across distances — **the onset account is wrong; item (a)/L025's pattern needs another explanation. Report and stop for discussion.** |

### Cost

(a): 7 k-values × 2 classes × 29 folds = 406 LOSO folds. (b): 14 positions × 29 folds = 406 LOSO folds. **Total: 812 folds, single seed** — the largest single control in this audit, but pre-authorized as the final one regardless of outcome.

### Status

**FINAL CONTROL, but extended once more (F-DRIFT-G, below) per instruction — see that section for why.** `results/SYNTHESIS.md`'s rewrite now waits for BOTH F-DRIFT-G and the trial-50 event identification (below) to report, per explicit instruction.

### RESULT (2026-08-20): (a) DECISIVE, separation interpretation WITHDRAWN; (b) verdict text corrected — a step change, not a decaying transient

**(a) Applied mechanically, per the pre-registered rule: every k's pre_cal_balanced collapsed to chance, range 0.4974–0.5037 across all 7 k-values. Verdict: COLLAPSES TO CHANCE.** Per the pre-registered consequence, **the F-DRIFT-C temporal-separation interpretation is WITHDRAWN** — this closes the PROVISIONAL flag above: F-DRIFT-C's dose-response curve (L016) reflected onset concentration within pseudo-class 0, not temporal separation between pseudo-classes. The Spearman/Wilcoxon numbers stand unchanged; the mechanism question is now settled.

**(b) The pre-registered Spearman result stands exactly as computed: rho=-0.7143, one-sided p=0.0357 — meets the decay criterion (rho≤-0.6, p<0.05), so the mechanical "DECAYS WITH DISTANCE" verdict is accurate to the letter of the pre-registered rule.** Pooled-by-distance curve: d25=0.5801, **d50=0.8343**, d75=0.5513, d100=0.5082, d125=0.5259, d150=0.5059, d175=0.5333. Block-2-only values confirm independent replication: d25=0.5727, **d50=0.8405**, d75=0.5697.

**⚠ Correction to the qualitative description (disclosed, per instruction — not a silent edit, the Spearman number and its pass/fail against the pre-registered rule are unchanged):** "decays with distance" / "confirms the onset account" overstates the shape. This is a **LOCALIZED SPIKE at d50 with a flat tail** (d75–d175 all sit in the 0.50–0.55 chance band), not a smooth decay from the onset — a true decaying transient would peak at d25, but d25 (0.5801) is well below d50 (0.8343). **Corrected description: "a step change at approximately trial 50 of each block" — NOT "a decaying onset transient."** This must not be called a "transient" of any kind until the underlying event is actually identified — see the trial-50 identification entry below, which found exactly such an event.

**See RESULTS_LEDGER.md L028** for the full numeric record.

---

## REST-BREAK DISCONTINUITY (renamed 2026-08-20 from "trial-50 event" / "onset transient"; no Modal, local data inspection)

**Naming, per explicit instruction:** the structural event identified below is a **SELF-PACED REST BREAK**, not a "session-onset transient" — renamed **"rest-break discontinuity"** everywhere in this project's documentation (this file, RESULTS_LEDGER.md, `results/SYNTHESIS.md` once rewritten, `scripts/identify_trial50_event.py`'s docstring). "Onset" remains correct for the separate, already-established F-DRIFT-E concept (block onset, trial index 0 of a block) — the rest-break discontinuity sits at trial ~50 WITHIN a block, not at its start; the two must not be conflated.

**Method and finding recorded in full in RESULTS_LEDGER.md L029 (11/30 subjects) and L033 (extended to all 30/30 subjects, 2026-08-20).** Summary: local inspection of raw `.vmrk` marker files and behavioral `.tsv` files (`scripts/identify_trial50_event.py`) found that **in ALL 30 subjects (including sub-09, whose marker file is intact despite its truncated continuous EEG signal), the single largest inter-marker time gap in each block falls at event index 49 (29/30 subjects) or 54 (1/30, sub-01) — universal replication, no exceptions.** Gap magnitudes: 5.6x–69.0x the block's own median inter-trial gap, absolute durations 8.3s–239.5s (vs. ~3–5s typical ITI). Full per-subject table in RESULTS_LEDGER.md L033 / `results/rest_break_discontinuity_table.md`. This is a single, non-periodic gap per block, consistent with a self-paced rest break inserted after approximately the first quarter of each 50-behavioral-trial block (~behavioral trial #12–13, given ~4.0 EEG epochs per behavioral trial, verified across the full cohort). No explicit break/comment marker exists in the raw stream — inferred from the timing anomaly alone, described as "consistent with a self-paced break," not asserted as confirmed screen content. **This is very likely the mechanism behind F-DRIFT-F(b)'s step change.**

---

## F-DRIFT-G — real labels under onset (rest-break-region) exclusion, the missing cell (pre-registered 2026-08-20, BEFORE this script runs)

### Why this control exists

F-DRIFT-F(a) showed that onset exclusion collapses the PSEUDO-label k-sweep to chance — but the REAL Search-vs-Memorize contrast under the same onset exclusion has never been tested. This is the one remaining cell that could change the project's conclusion: if the real contrast SURVIVES onset exclusion at a meaningfully elevated accuracy, part of the effect may be genuine task signal that the onset artifact was masking, not confounding with it.

### Test design

Drop the first 50 trials of each block for every subject (identical truncation to F-DRIFT-F(a)), REAL Search-vs-Memorize labels, identical calibrated pipeline, seed=42, full 29-fold LOSO. Report balanced pre_cal and post_cal. Reference values (unrestricted, untruncated real contrast): pre_cal=0.5201, post_cal=0.7078 (RESULTS_LEDGER.md L009/L011).

### Pre-registered interpretation rule (fixed BEFORE running, balanced-accuracy terms)

| Result (post_cal) | Interpretation |
|---|---|
| **< 0.55** | The real contrast is ENTIRELY onset artifact — **the methodological null result is complete and final.** |
| **≥ 0.65** | Genuine task signal exists and was masked by the onset artifact — **this becomes a POSITIVE result and the paper changes completely. Halt and report before any interpretation.** |
| **0.55–0.65** | Partial — report both components, with the share attributable to onset quantified. |

**Report pre_cal regardless of branch** — a rise there specifically would indicate genuine cross-subject task transfer, which nothing in this audit has shown so far (every pre_cal number to date, real or pseudo, has landed at or near chance once temporal/onset structure is controlled for).

### Extension past the previously-declared final control, recorded explicitly per instruction

F-DRIFT-F was declared final. It is extended once more because (a) settled the mechanism question (onset, not separation) but left the REAL-label question open — F-DRIFT-G is the direct test of whether that onset artifact is the WHOLE story for the real contrast or only part of it, which could materially change the project's conclusion. Run together with the rest-break discontinuity identification (above); `results/SYNTHESIS.md`'s rewrite waited for both to report.

### Cost

1 condition × 29 folds — cheap, comparable to F-PARITY-WITHIN.

### RESULT (2026-08-20): the ">=0.65 -> GENUINE SIGNAL" branch FIRED; DISCLOSED RULE MIS-SPECIFICATION recorded in the same section

**Numbers: post_cal_balanced = 0.7215** (vs. unrestricted reference 0.7078), **pre_cal_balanced = 0.4916.** Applied mechanically: post_cal_balanced ≥ 0.65, so the pre-registered ">=0.65 -> GENUINE SIGNAL" branch fired, exactly as written above. Recorded as fired — not restated further.

**⚠ DISCLOSED RULE MIS-SPECIFICATION — the third in this audit (after F-DRIFT-C's monotonicity check and F-DRIFT-E's original balance-free design):** the rule above equated "survives onset exclusion" with "genuine task signal." **These are not equivalent.** The pseudo-label contrasts that collapsed under onset exclusion (F-DRIFT-F(a)) were WITHIN-block; the real Search-vs-Memorize contrast is BETWEEN blocks. Removing the first 50 trials from inside each block removes the rest-break discontinuity from both blocks equally, but does not touch: **task instruction** (differs by block), **stimulus set** (disjoint within subject, unaffected by within-block truncation), **session half** (block 1 is always first, block 2 always second), or **post-break state** (block 2 still follows the ~400s inter-block break block 1 does not, even with its own internal rest-break region removed). **F-DRIFT-G eliminates ONE confound and leaves four intact. This result must NOT be written up as evidence of task decoding.** pre_cal_balanced=0.4916 is at chance, consistent with the pre-calibration invariant (RESULTS_LEDGER.md L032). **See RESULTS_LEDGER.md L031 for the full record.**

---

## THE PRE-CALIBRATION INVARIANT (compiled 2026-08-20, standalone finding — RESULTS_LEDGER.md L032)

Across every configuration of the real Search-vs-Memorize contrast tested in this audit — unrestricted (0.5201), phase-matched (0.5083), cross-parity (0.4839), within-parity (0.5303), onset-excluded (0.4916) — **pre-calibration cross-subject accuracy never leaves the 0.4839–0.5303 band, all within ±0.03 of chance.** Stated plainly: **there is no subject-general Search-vs-Memorize representation in these features.** Every elevated post_cal number in this audit is a product of the within-subject 15% calibration step. This is one of the audit's firmest claims and holds independent of how any individual configuration's post_cal number or pre-registered verdict was interpreted.

---

## F-SME — subsequent memory, the last experiment (pre-registered 2026-08-20, BEFORE this script runs) — FINAL EXPERIMENT, UNCONDITIONALLY

### Why this control exists

F-DRIFT-G's mis-specification (above) established that no test removing trials WITHIN a block can isolate genuine task signal from the block-level confound (task instruction, stimulus set, session half, post-break state) — the confound and any real signal are both between-block and inseparable by within-block manipulation. **Subsequent memory (remembered vs. forgotten, the classic Dm/subsequent-memory-effect paradigm) is the ONLY contrast available in ds005189 that is simultaneously within-block, within-task, within-stimulus-set, and within-session-half** — both classes are drawn from the same task block, interleaved throughout it, matched on encoding condition. None of F-DRIFT-G's four remaining confounds apply.

### Linking methodology (verified BEFORE this script was written)

`(scene, obj)` uniquely links each of a subject's 100 Encode trials to exactly one `Target` row in that subject's Test/retrieval file (AUDIT.md Phase 0.5 Priority 1). **Re-verified 2026-08-20 against locally cached data for 6 subjects: 0 unmatched, 0 collisions, forgotten-count table reproduces AUDIT.md's already-documented numbers exactly** (e.g. sub-01 Search 12%/Memorization 18%, sub-02 0%/6%, matching to the percentage point). Outcome: `rk_judg` in {Remember, Know} = remembered (0); `rk_judg` == New (a miss) = forgotten (1, minority).

**Epoch-to-behavioral-trial correspondence:** raw marker codes "Stimulus/ 11" (Search block) and "Stimulus/ 21" (Memorization block) each fire EXACTLY 50 times per subject, strictly chronologically (monotonically increasing sample position) — verified across all 30 locally-available subjects before this script was written. This is the ONLY marker-code pair confirmed to correspond 1:1, in order, with the 50 behavioral trial rows of each task; every other code (10/12/13, 20/22/23, 101/201) lacks a verified trial-level mapping and is deliberately NOT used — using them would risk silently mislabeling subsequent-memory outcome.

### Test design

Label = subsequent memory outcome. Run within-Search-only, within-Memorize-only, and pooled. Identical calibrated pipeline, seed=42, LOSO (fold count = subjects surviving exclusion for that condition, not necessarily 29). **AUC is the primary metric** (`roc_auc_score` on the continuous decision score) — minority class ~0–18%/subject/condition, so raw accuracy is meaningless. Balanced accuracy and per-subject minority-class counts also reported. pre_cal and post_cal reported separately, as always.

**Exclusion, computed and logged BEFORE any classification:** any subject with fewer than `MINORITY_MIN_TRIALS=10` forgotten trials in a condition is excluded from that condition entirely (not just from being a test fold).

**⚠ Pre-run exclusion preview (11/30 subjects checked locally before the Modal run):** **0/11 pass within-Search** (Search items are remembered too well — consistent with the dataset's own documented "search superiority" effect), **3/11 pass within-Memorize**, **7/11 pass pooled**. `within_search` is very likely entirely UNEVALUABLE at full scale; `pooled` is the condition most likely to produce a scoreable result. Disclosed now, before the full run, per instruction.

### Pre-registered interpretation rule (fixed BEFORE running, stated as the inference each threshold supports)

| Result (post_cal AUC) | Interpretation |
|---|---|
| **≥ 0.60, 95% CI excludes 0.5** | Genuine subsequent-memory signal, free of the block confound. A real cognitive result, underpowered but clean. |
| **0.55–0.60, CI excludes 0.5** | Weak but present; report as exploratory with power caveats. |
| **≤ 0.55, or CI contains 0.5** | No detectable confound-free cognitive signal in this dataset; the methodological paper is the paper. **A null here is NOT evidence against the effect existing — it is a power statement given 0–18% minority trials.** |

### Status

**FINAL EXPERIMENT, UNCONDITIONALLY.** After F-SME reports, no further experiments — `results/SYNTHESIS.md` is rewritten covering everything, including all three disclosed criterion mis-specifications in this audit (F-DRIFT-C's monotonicity check, F-DRIFT-E's original balance-free design, F-DRIFT-G's onset-vs-block-confound equivalence) listed together so a reviewer can see the pattern — and the user makes the paper decision from it.

### RESULT (2026-08-20): NULL. THE EXPERIMENTAL PROGRAM IS COMPLETE

| Condition | n (post-exclusion) | post_cal AUC | 95% CI | Verdict |
|---|---|---|---|---|
| within_search | 0/29 — UNEVALUABLE | — | — | Zero subjects reached 10 forgotten Search trials. |
| within_memorize | 13/29 | 0.4968 | [0.4489, 0.5423] | CI contains 0.5 → no detectable signal |
| pooled | 21/29 | 0.5141 | [0.4760, 0.5547] | CI contains 0.5 → no detectable signal |

Applied mechanically per the rule above: both scoreable conditions land in the "≤0.55 or CI contains 0.5" branch. **within_search's 0/29 is itself a behavioural finding, not just a technical exclusion: it IS the search-superiority effect appearing as a power failure** — so few Search-encoded items are ever forgotten (by any subject) that the contrast cannot be tested at all in this dataset. **Per the pre-registered inference: this null is NOT evidence against subsequent-memory effects existing — it is a power statement given 0–18% minority trials per subject/condition. The methodological paper is the paper.** Full record: RESULTS_LEDGER.md L035.

**THE EXPERIMENTAL PROGRAM IS COMPLETE. No further runs, unconditionally.**

---

## F-LEAK pass criterion (recorded 2026-08-18, BEFORE Batch 0 is executed)

Per the same pre-registration discipline as the ocular-artifact interpretation matrix above: this is the binding pass/fail criterion for `scripts/verify_no_leakage.py`'s shuffled-label and noise-feature checks, fixed before any real-data run, so it cannot be adjusted after seeing the result.

**Criterion:** for both the SHUFFLED_LABELS and NOISE_FEATURES conditions, using the pooled fold-level accuracy (`total_correct / total_n` across the 5 mini-LOSO folds):
1. The **95% Wilson score confidence interval** on the pooled accuracy must **contain 0.50** (chance level for this balanced binary task).
2. The **pooled mean accuracy must fall within 0.50 ± 0.03** (i.e., in `[0.47, 0.53]`).

Both conditions must hold for a given check to PASS. This replaces the script's earlier ad hoc criterion (`mean_acc < 0.60` AND `binomial p-value >= 0.05` vs. chance), which was directionally correct but not principled: a fixed 0.60 cutoff doesn't scale with sample size the way a CI does, and passing a one-sided binomial test alone doesn't rule out an accuracy that is implausibly far *below* chance (which would itself indicate a bug — an inverted label or index, e.g.) or an accuracy that is statistically significant but trivially small in practice. The old binomial p-value is retained in the script's output as an auxiliary diagnostic, not as a gating criterion.

**Rationale for the ±0.03 mean band:** at `N ≈ 2,000` pooled test trials per condition (5 mini-LOSO folds' held-out subjects), a chance-level classifier's accuracy standard error is roughly `sqrt(0.25/2000) ≈ 0.011`, so a ±0.03 band is close to ±3 standard errors under the null — tight enough to catch a real leak, loose enough not to false-positive on ordinary sampling noise.

**Real condition (sanity baseline) is unaffected by this criterion** — it still uses its own `REAL_MUST_EXCEED_ACC = 0.55` threshold (the harness must demonstrate it can learn real signal at all; this is a floor, not a chance-band check).

---

## GLOBAL INVALIDATION NOTICE (recorded 2026-08-22): the class label is a 4-way encode/test-phase mixture

**Full technical record: RESULTS_LEDGER.md L037 (root cause, exact numbers) and L038 (arithmetic verification of the unified explanation against real per-subject epoch indices).** Summarized here because it supersedes the interpretive basis of every decision recorded above it in this file.

`EVENT_ID` (`run_data_engine_on_modal.py` lines 28-33) maps four structurally distinct marker types per class into one label: 50 Encode-phase trial onsets (code 10/20), 50 Test-phase Target-recognition onsets (11/21), 25 Test-phase Distractor-recognition onsets (12/22), and 75 Test-phase New-Lure onsets (13/23, items never studied under either condition). Verified via exact cross-subject numerical match against the Test behavioral TSV's `obj_type` breakdown, 5/5 subjects checked. Test-phase item order is independently randomized relative to Encode order (0/50 position matches, sub-01).

**Binding consequence: every classification number produced by this project to date — including evidence lines 1-3's headline figures (70.78% etc.) — is PROVISIONAL-INVALID as a claim about Search-vs-Memorize decoding, until re-run under a corrected label definition.** This is not confined to the diagnostic-control family; it applies to the full results table. RESULTS_LEDGER.md L038 arithmetically confirms the composition account explains every pseudo-contrast effect in evidence lines 4-9 (13/13 mappings fit; 1 — early/late's exact cross-script magnitude — fits in direction only and is flagged, not forced).

**Binding scope going forward:** no number produced before this notice may be cited in `PAPER/draft.md` or `PAPER/outline.md` as final. Drafting stays halted. Manuscript decision is deferred until R1-R3 below report.

---

## R1 / R2 / R3 pre-registration (recorded 2026-08-22, BEFORE any of the three run)

Per this codebase's pre-registration discipline (matches the pattern of every prior fix-ID). Three runs, then stop — **V2 (grouped calibration split) is explicitly NOT run yet; it tests a smaller problem inside this larger invalid one and becomes relevant again only for whichever contrast (if any) survives R2.**

### R1(a) — Composition diagnostic, evidence lines 4-8 (no classifier; already run locally)

Reports the real encode-fraction composition of every pseudo-class construction used in F-DRIFT/F-DRIFT-B/F-DRIFT-C/F-DRIFT-E, computed from real per-subject marker sequences (not an assumed flat split). **Not a threshold check — a descriptive report.** Result: RESULTS_LEDGER.md L038.

### R1(b) — Direct encode-vs-test decode

Re-epoch and classify codes {10,20} (encode) vs. {11,12,13,21,22,23} (test), identical calibrated LOSO pipeline (EA whitening, tangent-space, shrinkage calibration, seed=42, 29 folds), report pre_cal and post_cal balanced accuracy.

**Pre-registered joint criterion (fixed now, before running):** IF (i) R1(a)'s composition gaps predict each pseudo-contrast's previously reported accuracy (already substantially confirmed by L038) AND (ii) this direct encode-vs-test decode achieves **post_cal balanced accuracy > 0.85**, THEN the unified composition explanation is CONFIRMED IN FULL and the drift/rest-break-discontinuity mechanistic account (F-DRIFT through F-DRIFT-G) is WITHDRAWN — the true separable signal was encode-vs-test phase all along, not temporal drift or a rest-break-related artifact.

### R2 — The correctly-defined Search-vs-Memorize contrast

Re-epoch using ONLY codes 10 and 20 (encoding-phase trial onset — the one epoch class with an unambiguous, uncontaminated, one-per-behavioral-trial Search/Memorize label). 50 epochs/task/subject, 2,900 epochs total. Run through the identical calibrated LOSO pipeline. Report pre_cal and post_cal balanced accuracy, class balance, and n per fold.

**This is the only version of the Search-vs-Memorize contrast in this codebase whose name means what it says. No accuracy threshold is pre-registered as pass/fail — report it whatever it shows, per explicit instruction.** If it is at or near chance, that is the headline finding, not a null result to explain away.

### R3 — Lure-removed contrast

The nominal 200-epoch/class contrast with the 75 New-Lure epochs/class dropped (125 epochs/class/subject: 50 encode + 50 target + 25 distractor). Identical pipeline. Purpose: quantify how much of the original 70.78% was contributed by epochs (New-Lure) that carry no encoding-condition information at all. Descriptive, not pass/fail.

**After R1-R3 report: STOP. Do not run V2 until it is clear which contrast, if any, survives R2. Do not touch `PAPER/draft.md` or `PAPER/outline.md`. The user decides the paper's final form from the R1-R3 results.**

---

## C1 / C2 — R2 shuffled-label control + sub-03 outlier check (pre-registered 2026-08-22, BEFORE either runs)

### Why these controls exist

R1/R2/R3 reported back (Modal run completed after the granular data engine finished). R2's `pre_calibration_balanced_accuracy_mean` = **0.5737** — the first real-label, cross-subject, pre-calibration number in this entire audit that is not at chance, sitting above every chance floor this project has established (F-PARITY 0.4839–0.5303, F-LEAK shuffled-label 0.4862). Per this codebase's standing discipline (F-LEAK, F-DRIFT's own null controls), no real-signal claim is reported without the same control applied to every other real-signal claim in this audit: a shuffled-label null computed on the EXACT dataset in question, not reused from a differently-composed dataset. **Two checks, then stop — do not touch `outline.md`/`draft.md`, both already superseded by the R1(b) composition confirmation and being rebuilt separately.**

### C1 — Shuffled-label control on the R2 dataset specifically

**Dataset:** the exact R2 encode-only dataset from `run_r1b_r2_r3_composition_runs.py` (N=2,910: ~50 Search-encode + ~50 Memorize-encode epochs/subject, 29 subjects — real per-subject counts vary, e.g. sub-01 is 55/55 per ledger L038/R1(a), always balanced WITHIN each subject even where the count departs from the modal 50). **Not F-LEAK's old shuffled-label numbers** — those were computed on the contaminated 200-epoch/class dataset and do not bound this one.

**Procedure:** within-subject label shuffling — for each of 30 independent shuffles, permute each subject's own R2 labels (however many that subject has — 100 for most, 110 for sub-01) among themselves (a true permutation of the realized values, so each subject's own within-subject class count is preserved exactly by construction, not merely approximately — this is a per-subject balance check, not a fixed ==50 check). Run the identical LOSO/EA/tangent/shrinkage-calibration pipeline (byte-identical to `run_r1b_r2_r3_composition_runs.py`'s `run_loso`) on each shuffle, seed=42 for every pipeline-internal RNG (classifier fits, CV shrink search, calibration split), full 29-fold LOSO per shuffle. Report the pooled distribution of `pre_calibration_balanced_accuracy_mean` (one value per shuffle, mean across that shuffle's 29 folds — matching R2's own reporting convention) across the 30 shuffles: mean, SD, and 95% CI.

**Implementation note (does not change what is computed, only the order):** EA whitening, covariance/tangent-space vectorization, and PCA are all fit on `X_train`/`X_k` alone — none of them consume `y` — so these steps are label-independent and identical across all 30 shuffles for a given fold. The script computes them ONCE per fold and reuses them across shuffles, redoing only the label-dependent steps (stratified calibration/test split, global/local classifier fits, shrinkage CV) per shuffle. This is a ~30x reduction in redundant eigendecomposition cost, not a change to the pipeline itself — every shuffle still gets its own independently-stratified calibration split and its own independently-fit classifier.

**95% CI convention:** primary = empirical percentile CI (2.5th/97.5th percentile of the 30 shuffle-level values) — matches this project's established preference for empirical/percentile CIs over parametric ones (F-DRIFT-C's post-hoc bootstrap CI, F-OCULAR(a)'s A2 percentile-rank null). Secondary, reported alongside for transparency: normal-approximation CI (mean ± 1.96×SD). **Disclosed resolution caveat, matching the F-OCULAR(a) A2 20-draw precedent:** at 30 shuffles, the 2.5th/97.5th empirical percentiles are coarse (interpolated between the 1st/30th sorted values) — this is an accepted tradeoff for keeping the ~30x LOSO-family cost bounded, not treated as more precise than it is.

**Pre-registered verdict rule (fixed now, before running):**

| Result | Interpretation |
|---|---|
| **Real R2 pre_cal_balanced (0.5737) falls OUTSIDE the shuffled-label 95% CI (primary = percentile)** | Genuine, subject-generalizable signal — state this plainly, not softened into "trending" or "suggestive." |
| **Real R2 pre_cal_balanced (0.5737) falls INSIDE the shuffled-label 95% CI** | 0.5737 does not currently distinguish itself from what within-subject shuffling alone produces at this N — state this with equal plainness, not softened. |

If the primary (percentile) and secondary (normal-approximation) CIs disagree on which side of the boundary 0.5737 falls, report both explicitly rather than picking one silently.

### C2 — sub-03 outlier check

sub-03's R2 pre-cal fold = 0.3529 (balanced 0.3571) — the only fold below chance, and well below it. Before this sits inside a headline mean unflagged:

**(a) Epoch count / class balance.** Confirm sub-03's encode-only epoch count and class balance directly against the granular npz (expect 50/50) — rule out a trivial labeling or count defect first. Cross-checked independently against a from-scratch local parse of sub-03's raw `.vmrk` marker file (same ground truth the granular data engine's `mne.events_from_annotations` reads) — an agreement between the two is a stronger check than either alone.

**(b) Block order / marker parsing.** Confirm sub-03's block order (odd subject → Search-first per D2's counterbalancing scheme) and raw marker parsing (code composition, position monotonicity, no duplicate/malformed entries) match the pattern seen in the other 28 subjects (29-subject cohort minus sub-09) — looking for nothing exotic, just ruling out a parsing edge case unique to this subject.

**(c) Reporting rule, fixed now:** report findings whatever they are. If nothing anomalous turns up, say so explicitly and leave sub-03's fold in the reported R2 mean as-is — do not drop it without cause.

### Scope

Both checks report plainly; neither characterizes R2 as confirmed or disconfirmed on its own — that characterization waits on C1's verdict, per explicit instruction. Do not draft any manuscript text, do not revise `outline.md`, regardless of either result.

### C1/C2 RESULT, ACCEPTED (2026-08-22): genuine, subject-generalizable signal; sub-03 cleared

**C1:** 30-shuffle null distribution — mean=**0.5010**, SD=**0.0137**. Real R2 pre_cal_balanced (0.5737) falls OUTSIDE the 95% CI on both the percentile and normal-approximation method. **Verdict: genuine, subject-generalizable signal**, per the pre-registered rule above — stated plainly, not softened.

**C2:** sub-03 shows no epoch-count, class-balance, block-order, or marker-parsing anomaly (ledger L041). Per the pre-registered reporting rule, its fold stays in the R2 mean as-is.

**Consequence:** R2's 0.5737 is now this project's central positive claim — the first result in the entire audit that is neither chance nor an artifact this project's own controls have identified. Per explicit instruction, it now receives the same level of scrutiny the drift account received before it collapsed (F-DRIFT through F-DRIFT-G) — two more checks, below, before `outline.md`/`draft.md` may be touched.

---

## C3 / C4 — robustness checks on the accepted R2 signal (pre-registered 2026-08-22, BEFORE either runs)

### Why these controls exist

C1/C2 just promoted R2's 0.5737 to "genuine signal" — the paper's central claim. This project's own history (the drift/rest-break account, accepted at F-DRIFT/F-DRIFT-B and only fully understood after F-DRIFT-E through F-DRIFT-G) is the standing reason not to stop at one confirmatory control. Two robustness checks, then this reopens `outline.md`/`draft.md`.

### C3 — Jackknife sensitivity of the R2 pre_cal_balanced mean

**Not a new classifier run** — a jackknife over the 29 per-fold `pre_cal_plausibility.balanced_accuracy` values already produced by `run_r1b_r2_r3_composition_runs.py`'s `R2_search_vs_memorize_encode_only.fold_results`, read directly from `results_r1b_r2_r3_composition_runs.json` on the volume. For each of the 29 folds, recompute the mean with that fold excluded (28-fold mean); report all 29 leave-one-out (LOO) means, their range, and the full-sample mean recomputed from the same 29 values as a sanity cross-check against the already-reported 0.5737.

**Reference values (fixed now, from the accepted C1 result above):** real R2 mean = 0.5737; null 95% CI upper boundary = 0.5223; half-gap threshold = `(0.5737 − 0.5223) / 2` = **0.0257**.

**Pre-registered verdict rule (fixed now, before running):**

| Result | Interpretation |
|---|---|
| **No single-fold exclusion drops the LOO mean below 0.5223** | The signal is not an artifact of one or two influential subjects — reported as an added robustness line, not re-litigated further. |
| **At least one exclusion drops the LOO mean below 0.5223** | Name the fold(s), report the mean with and without it/them, plainly, in the same breath as the C1 verdict — not as a footnote. |

Separately, any single-fold exclusion whose LOO mean shifts from the full-sample mean by more than the 0.0257 half-gap threshold is flagged as individually influential in the reported table, regardless of which side of the 0.5223 pre-registered threshold it falls on — this is a reporting flag, not a second pass/fail gate.

**Script:** cheap Modal read-and-arithmetic pass (no retraining) against the already-completed `results_r1b_r2_r3_composition_runs.json`.

### C4 — Higher-resolution permutation p-value (500 shuffles)

**Why:** 30 shuffles gives a minimum resolvable empirical p of 1/31 — adequate for C1's pre-registered CI-membership test, not for a headline statistic. Rerun the identical C1 procedure at 500 within-subject label shuffles, reusing the same label-independent EA/tangent/PCA precompute-once-per-fold design (no reason to recompute features that do not depend on `y`). Same `SHUFFLE_BASE_SEED` as C1, so the first 30 of the 500 shuffles are the IDENTICAL shuffles C1 already ran (deterministic RNG per shuffle index) — this gives an exact internal reproducibility check for free, not just a statistical consistency comparison.

**Report:** the empirical p-value (fraction of the 500 shuffle-level `pre_cal_balanced` values ≥ 0.5737, both the raw fraction and the add-one-corrected version this codebase already uses elsewhere — `scripts/verify_no_leakage.py`'s `permutation_null_stats`, DECISIONS.md's A2 entry), the 95% CI at this resolution (percentile primary, normal-approximation secondary, matching C1's convention), and an explicit consistency check against C1's 30-shuffle result.

**Pre-registered consistency rule (fixed now, before running):** the 500-shuffle percentile 95% CI must exclude 0.5737 on the SAME side (above the upper bound) as the 30-shuffle CI did. If the two disagree on which side of the boundary 0.5737 falls, or if the first 30 of the 500 shuffle values do not exactly reproduce C1's original 30 values, **stop and report the discrepancy explicitly before either result is cited** — do not silently prefer one over the other.

**Script:** `run_c1_shuffled_label_control.py`'s identical pipeline, `N_SHUFFLES=500`, same `SHUFFLE_BASE_SEED`.

### Scope

After both report: update `RESULTS_LEDGER.md` (new entries, not edits to L037-L041), `STATUS.md`, and `DECISIONS.md` with the finalized C1 signal claim and its two robustness checks. **Do not begin `outline.md`/`draft.md` in the same turn these report — confirm both are recorded first, then stop and report back before any drafting starts.**

---

## D1 / D2 — transcription cleanup + parity-split check on R2 (pre-registered 2026-08-22, D2 BEFORE it runs)

### D1 — transcription cleanup

Pure bookkeeping, not a new statistical test: pull the five values H1 (ledger L046) flagged as missing from their already-completed run outputs and append them to `RESULTS_LEDGER.md` as new entries (L037-L048 untouched). R1(b)'s `post_cal_balanced` is given directly (0.8668) and transcribed as user-reported per this project's standing convention that the user runs Modal and reports results back; the other four (R2 post_cal, R3's full numbers, C3's 29-row table, C4's exact SD/CI/p-convention split) require reading the actual output JSONs on the volume — `run_d1_transcription_dump.py` (Modal, read-only, no retraining) prints them in one pass so a single command produces all four. Flagged, not silently absorbed, if any value surprises against what's already accepted.

### D2 — parity-split check on R2 (task-instruction vs. block-order)

**Why:** H3 (L048) named "does the signal reflect task instruction specifically, vs. block-order/session-position" as an open question, since neither F-STIM nor F-PARITY-WITHIN was computed on R2's own clean dataset. This is the direct test.

**Design:** using the exact R2 encode-only dataset, split the 29 subjects by parity per D2's own counterbalancing scheme (14 odd = Search-first, 15 even = Memorize-first). Run the identical LOSO/EA/tangent/shrinkage-calibration pipeline **separately within each parity group** (a held-out subject's training pool is restricted to the OTHER subjects of the SAME parity group only — 13-fold LOSO within odd, 14-fold within even). Report `pre_cal_balanced` per group.

**Null reference:** the existing C1/C4 pooled null (computed on the full 29-subject mixed-parity pool) is NOT a valid reference at this smaller per-group N — a mean over 13-14 folds has a wider natural spread than a mean over 29, so the pooled null's tighter CI would make a real within-group effect look artificially more significant than it is. **A fresh within-group shuffled-label null is run instead**, at each group's own N: 30 within-subject label shuffles restricted to that group's own subjects, identical precompute-once-per-fold design (verified equivalent by C1), reporting that group's own null mean/SD/95% CI (percentile primary, normal-approx secondary, matching C1's convention).

**Pre-registered verdict rule (fixed now, before running), applied to each group's `pre_cal_balanced` vs. its OWN within-group null:**

| Result | Interpretation |
|---|---|
| **BOTH groups individually fall outside their own within-group null's 95% CI (high side)** | The signal is not explained by block-order/session-position alone — position and label are inversely mapped between the two groups (odd: Search=position-1st; even: Search=position-2nd), so a pure position confound would predict the OPPOSITE class assignment between groups and could not produce the same-direction elevation in both. Stated plainly as strengthening the task-instruction reading. |
| **Only ONE group falls outside its own null** | Name this plainly as evidence the signal may be position-driven rather than task-driven — report even though it complicates the finding; do not suppress or soften it. |
| **NEITHER group falls outside its own null** | Would be inconsistent with C1/C4's full-pool result (which used all 29 mixed-parity subjects) — halt and report for discussion before further interpretation; do not construct a post-hoc reconciliation. |

**Script:** `run_d2_parity_split_check.py` (Modal, reuses the identical EA/tangent/PCA/shrinkage-calibration + precompute-once-per-fold design already verified for C1/C4, restricted per-call to one parity subgroup's subject list).

### Scope

Record D1 and D2 as new ledger entries. **Report back before touching `outline.md`/`draft.md` — this is the last gate before drafting starts, per explicit instruction.**

---

## Drafting phase — standing rules (recorded 2026-08-23, BEFORE any `draft.md` prose is written)

Title decided: Candidate A, *"A Mislabeled Contrast, Recovered: Diagnosing and Correcting an Encode/Test-Phase Confound in Blocked EEG Decoding."* All outline decisions resolved (`PAPER/outline.md`'s "Flagged for review" section — zero items open). `draft.md` begins under these binding rules, fixed now per this project's pre-registration discipline:

1. **Section drafting order:** Methods → Results → §3a (validation controls) → §4 Protocol recipe → §5 Limitations → §6 Discussion → §7 Conclusion → §8 Data/Code Availability. **Introduction and Abstract drafted LAST**, in that order.
2. **Numeric provenance:** every number in the draft must trace to its `L0xx` entry as already cited in `outline.md`. If a number is needed that has no outline citation, flag it rather than invent one — do not silently source a number from anywhere else (a script's docstring, a ledger table not yet cited, memory of an earlier turn).
3. **Effect-size phrasing, fixed verbatim:** *"~7.3–7.4 percentage points above chance (Cohen's d ≈ 5–7 against the shuffle null's tight distribution, reflecting the null's construction from 29-fold-averaged statistics rather than a dramatic raw effect)"* — never lead with d alone, never drop the qualifier about what the tight null reflects.
4. **Framing rule applies to every sentence:** self-audit of our own pipeline, not a critique of the source dataset (ds005189) or its authors.
5. **Limitations section may not be compressed or summarized** relative to what the outline specifies, on the grounds that the finding is now positive rather than null.
6. **Draft one section at a time; stop after each for review before continuing.** Start with §2 Methods.
7. **§3a's D2 subsection specifically:** do not draft its prose before confirming the Argument A (R2's own pooled-design inversely-mapped-position argument) / Argument B (D2's own within-group-null robustness result) split, established in `outline.md`, is rendered correctly and kept separate — this is the section most likely to drift back into conflating the two if written quickly.

---

## Cross-run numerical determinism and effect-size precision (recorded 2026-08-27, GATE C6–C10)

**Finding:** the classifier pipeline (C1/C3/C4/D2/R1b/R2/R3) is not bit-exactly reproducible across Modal container invocations. Root cause, confirmed via direct index-level diffs of persisted `shuffle_level_pre_cal_balanced` arrays across three independent run-pairs: (a) unpinned BLAS/LAPACK thread count racing on multi-threaded containers, and (b) OpenBLAS's `DYNAMIC_ARCH=1` build dispatching different microarchitecture-tuned kernels (different floating-point reduction order) depending on which host a container lands on — Modal offers no stable mechanism to pin CPU microarchitecture. Two runs under matched dispatch architecture ("Zen") and threads=1 reproduced bit-for-bit (0/500 differences); a comparison against a run whose own hardware was never recorded showed 5 isolated single-trial reclassifications, each an exact integer multiple of a per-contrast quantum (1/2436 or 1/2494 for R2, from its constant 42/43 per-fold class totals).

**Mitigation applied (not yet re-verified by a from-scratch run against committed code):** `threadpoolctl.threadpool_limits(limits=1)` wrapped in a real `with` block around each script's numerical core (Image.env()-only thread pinning was tried first and shown ineffective — Modal overwrites `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` to match `cpu` allocation after the image's ENV layer runs), plus `cpu` dropped from 4.0 to 1.0 on the four classifier-heavy scripts. **Caveat:** in every run this pin has executed in so far, Modal's own `cpu=1.0`-matched env-var override had already set thread count to 1 before the pin ran, so the pin itself has never been exercised against a genuinely contested (>1) thread count. `cpu=1.0` alone is what has been empirically shown sufficient.

**Effect-size precision, corrected:** the paper's headline effect (R2's real value above C4's null mean) must be evaluated against R2's real value's own jitter, not the null mean's. The null mean is a 500-shuffle average and its own observed movement is heavily damped (~2.4×10⁻⁶); R2's real value is a **single** 29-fold evaluation on the same dataset and is subject to the *same* one-quantum jitter as any individual C4 shuffle (~4.1×10⁻⁴, ≈0.041 percentage points). The defensible effect-to-jitter ratio is **~190×** (0.0778288 / 4.105×10⁻⁴), not a null-mean-damped ~32,000×. Consequently the effect size is not defensibly stable beyond an explicit stated tolerance — at the observed 7.7826 pp, one quantum (0.041 pp) exceeds the 0.033 pp margin to the nearest 1-decimal-place rounding boundary, so even "≈7.8 pp" should carry the tolerance explicitly (e.g. "7.78 ± 0.05 pp") rather than being read as bare-precise. This is compatible with, and does not require changing, item 3's existing "~7.3–7.4 pp" range phrasing (already a range, not a false-precision point estimate) — no manuscript text has been found citing "7.783" or similar over-precise figures; this entry is preventive, not a correction to existing prose.

**Provenance note:** the mitigation above is committed at `525522dc579e7fd5b371c5ef2048a669b0a25af5`. Artifacts generated before this commit (all Aug 25 canonical artifacts, e.g. `results_c4_high_res_shuffled_label_control_20260825T035948Z_a3293792.json`) are confirmed consistent with the code actually committed at their cited hash (`a3293792` — verified via `git show`). Two Aug 27 exploratory C4 runs were generated against this fix while it was still uncommitted (a dirty working tree) and are stamped with the stale hash `a3293792`, which does not reflect the code that produced them — those two files should be treated as exploratory evidence of the fix's within-architecture behavior, not as citable canonical artifacts, pending either a re-run against committed code or explicit documentation of their actual provenance.

## Reconstructed manuscript-defect list, N8–N19 (recorded 2026-08-27, GATE T1 STEP 2)

**Provenance note: this entry is reconstructed from the project record (the GATE T1 prompt), not recovered from a file.** GATE C15/T1 searched for `fourth_pass_prompt.md` and `gate_c_approval_amendments.md` — the stated original sources for items N8–N19 — across the full git-tracked tree, every untracked `.md` file in `D:\EEG` (via filesystem glob, not just `git ls-files`), the `.claude/` directory, and this session's own scratch directories. **Neither file exists anywhere searched.** The six items below are written here so they stop depending on a chat transcript that this project's own compaction has already partially lost once.

**Standing constraint, restated so it isn't lost again**: rt-based (reaction-time) alignment verifies **epoch-to-behavioral-trial correspondence** — that a given EEG epoch matches the trial the behavioral log says it should — it does **not** verify **label correctness** (that the trial was correctly assigned to Search vs. Memorize, or that a marker code means what the event dictionary claims). These are separate claims and must not be conflated when citing this project's verification methods in the manuscript.

- **N8** — the Methods description of the "distractor" marker type is mis-defined. `draft_elsevier.tex:48` / `draft_ieee.tex:50` currently describes it as "a test probe not matching any studied item from either condition" — indistinguishable, as written, from the same sentence's own definition of the New-Lure type ("a novel-item lure never studied under either condition"). Per `DECISIONS.md`'s own EVENT_ID documentation (line 634, above), Distractor-recognition and New-Lure onsets are structurally distinct marker codes (12/22 vs. 13/23); the manuscript's prose does not currently distinguish them. Needs a definition that actually separates the two categories — not verified against the original dataset documentation in this gate; flagging the contradiction, not supplying the corrected definition.
- **N9** — remove the "sub-01 exception" framing everywhere. `draft_elsevier.tex:48,56,72` / `draft_ieee.tex:50,58,74` (plus the Table S1 sub-01 row, `draft_elsevier.tex:225` / `draft_ieee.tex:224`) describe sub-01 as legitimately contributing 55 encode epochs per task as a persisting property of the analyzed dataset. Per this session's GATE C15 STEP 2a finding, this is the **pre-exclusion** raw count — sub-01's 5 extra unlogged practice-trial epochs per task are excluded at the epoching layer (`run_data_engine_granular_on_modal.py`, hard-asserted), so the analyzed dataset has sub-01 at a uniform 50/50, the same as every other subject. The "sub-01 contributes 55" framing should not appear as a description of the final analyzed data.
- **N13** — a generator-level section-reference token leaked Elsevier's section numbering into the IEEE build. `draft_ieee.tex:50,58` contain the literal text "§2.3" — Elsevier's own section number for this material — rather than a `\S\ref{...}` cross-reference that would resolve to whatever section number Methods actually has in the IEEE template. Both builds are apparently generated from a shared source with per-target overrides, and this specific literal escaped the override.
- **N14** — five "this paper"/"this project" over-corrections. Located 39 occurrences of "this paper" or "this project" in each file; only one is "this project" per file (`draft_elsevier.tex:204` / `draft_ieee.tex:206`, the Code/Data-Availability sentence). **Could not identify which five specific instances are the flagged over-corrections** without the source file — reporting the located candidate set, not guessing which five.
- **N17** — §2.3's "Disclosed implementation corrections" passage (`draft_elsevier.tex:72` / `draft_ieee.tex:74`) currently frames the shuffled-label control script's original pre-flight check (expecting exactly 50 encode epochs/task/subject) as a naive assumption that needed loosening ("corrected to verify within-subject class balance rather than a fixed count"). Per N9 above, the original fixed-count check was **correct** — the anomaly was in the data (sub-01's unlogged practice trials), not the check. This passage needs to describe trimming the data to match the check's correct assumption, not relaxing the check to accommodate the data.
- **N19** — replace "five subjects checked directly" with the all-29, two-method verification. `draft_elsevier.tex:48` / `draft_ieee.tex:50` currently states the composition claim "was verified by an exact cross-subject numerical match against each subject's own Test-phase behavioral TSV in five subjects checked directly" — matching `DECISIONS.md:634`'s "5/5 subjects checked." If a fuller, all-29-subject, two-method verification has since superseded this 5-subject check, the manuscript should cite that instead — not verified against a specific later ledger entry in this gate.

---

## Style-sweep item (recorded 2026-08-27, GATE T1 STEP 2, moved out of the N-series GATE T2 STEP 1c, attribution corrected GATE T4 STEP 5b)

**Source, corrected**: a **user-supplied style requirement**, received together with the GATE T1 prompt but not part of the six items (N8, N9, N13, N14, N17, N19) as the user originally authored that prompt's STEP 2b — the user has confirmed N8–N19 is the complete list as written, and that this item was appended before pasting. This is a legitimate requirement either way (received in the same message, not fabricated by the assistant), but it is credited here correctly as **user-supplied**, not as a GATE T1 prompt item of the same kind as N8–N19. Verbatim, as received: *"N20 eradicate 'AI dashes'. AI models tend to overuse em-dashes (--- or —) for punctuation. Locate every em-dash in both .tex files and specify a standard punctuation replacement (comma, colon, parentheses, or rewrite). The ONLY permitted exception is the single dash immediately following the Abstract heading."* It remains a typographic style requirement, not a manuscript-defect finding of the same kind as N8–N19, and per GATE T2 STEP 1c stays moved into its own item, **scheduled after the numeric pass (not part of T2's, T3's, or T4's scope)**, so that 84 mechanical typographic edits do not share a batch with substantive corrections like N8 and N17.

**Eradicate LaTeX `---` (em-dash) overuse.** 42 occurrences in each of `draft_elsevier.tex` and `draft_ieee.tex` (84 total); the standard-punctuation replacement (comma, colon, parentheses, or a rewrite) is a per-instance judgment call requiring the surrounding sentence. The single permitted exception is the `---` immediately following `\begin{abstract}` (`draft_elsevier.tex:22`, `draft_ieee.tex:27`) — both files' `\begin{abstract}` tags are at line 21/26 respectively, one line before.

**CORRECTED, GATE R3 STEP 1c (2026-08-29): the exemption above no longer applies. No exemption currently exists.** The Abstract's em-dash pair ("a label-dictionary error in our own analysis pipeline — not in the public dataset it was applied to —") was rewritten as a sentence split (GATE R2's proposed option ii, approved and applied in GATE R3 STEP 1) rather than kept under the N20 exemption. `draft.md`'s Abstract, `draft_elsevier.tex`, and `draft_ieee.tex` now contain zero em-dashes. This note documents the change so a future sweep does not treat the exemption above as still in force; the verbatim N20 instruction quoted above is left unedited as the historical record of what was originally requested.

**GATE R3 STEP 3 (2026-08-29): dashes deliberately retained, not missed.** Recorded here so no future sweep re-flags these as an oversight.

- **`1–40 Hz` (draft.md §2.2 and §8, two occurrences)**: standard scientific range notation for the band-pass filter's frequency range, used identically both times. Kept as an en-dash rather than converted to "1 to 40 Hz" because this is the conventional way a frequency range is written in EEG/signal-processing prose; spelling it out would be non-standard for the field, not a precision gain. Source: `draft.md` itself, this project's own prose.
- **Citation ranges `[3]–[5]`-style, wherever they appear in the compiled PDFs**: generated by IEEEtran's automatic compression of consecutive citation numbers inside a single `\cite{...}` call. Not present in any source file this project controls (`draft.md`, `build_latex.py`, `references.bib`) — it is `IEEEtran.bst`'s own bibliography-formatting behavior. Changing it would require patching a standard, unvendored LaTeX class/style file, not this project's own content.
- **Bibliography page ranges (e.g. `93–102`, `2538–2557`)**: standard reference-list formatting, generated by `bibtex`/`natbib` from the `pages` field already present in `references.bib`/`references_ieee.bib` (transcribed from each cited paper's own publication record) and expected by both target journals' citation styles. Changing the punctuation here would mean altering how a published paper's own page range is displayed, not fixing this project's prose.
- **Reference `blankertz2011`'s em-dash** (`references.bib`/`references_ieee.bib`, title field: "Single-trial analysis and classification of {ERP} components---a tutorial"): part of that paper's own published title (Blankertz et al., *NeuroImage* 56(2), 2011), transcribed verbatim. Altering it would misquote the cited work's actual title — a correctness error, not a style fix. First flagged in GATE R2 STEP 4c; recorded here as a standing decision, not left as an open question.

---

**Unsourced claim flagged for removal:** a claim that the original C4 container and the R1b/R2/R3 re-run container reported different SIMD/AVX support (AVX2-only vs. AVX512F/CD/SKX/CLX) as evidence of "different host hardware, bit-for-bit identical" reproduction does not appear in any persisted JSON artifact checked this session (neither `results_r1b_r2_r3_composition_runs.json` nor its stamped copies contain a `hardware_info`/`simd_found` field at all — that instrumentation was added only in the C4/C1/D2/R1b scripts, in this same commit). If this claim appears in a project summary document, it traces at most to console log text pasted into a chat prompt, not a checkable artifact, and no persisted evidence currently demonstrates two different dispatch architectures producing identical results — the only recorded matched-architecture pair (Zen/Zen) was identical; the only recorded divergence was against a run with unrecorded hardware. Flagged for removal or re-sourcing before being cited as an established finding.
