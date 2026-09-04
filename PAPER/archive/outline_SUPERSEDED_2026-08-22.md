> **SUPERSEDED 2026-08-22 — DO NOT USE OR REVISE.**
>
> This outline was built under the pre-invalidation "rest-break discontinuity / block-level confound stack" framing. The Search-vs-Memorize contrast it outlines a paper around conflates encode-phase and three distinct test-phase event types (`RESULTS_LEDGER.md` L037) — every ledger entry this outline cites (L001-L036) predates that discovery and is PROVISIONAL-INVALID as a claim about Search-vs-Memorize decoding as originally framed. See L037/L038 and `DECISIONS.md`'s "R1 / R2 / R3" through "D1 / D2" sections for the full re-investigation and the corrected contrast (R2) that replaces this outline's central finding.
>
> **Retained here, not deleted, per this project's audit-trail discipline.** Do not cite, revise, or resume from this file — see the new `outline.md`.

---

# Manuscript outline — OUTLINE ONLY, NO PROSE DRAFTED

**Status: awaiting user review. Do not draft manuscript text from this outline until approved.**

**Working title (APPROVED):** *Calibrated Accuracy Is Not Diagnostic of Task Content in Blocked EEG Decoding: A Self-Audit and Diagnostic Protocol*

**Target venue:** *Journal of Neural Engineering*

**Source document:** `results/SYNTHESIS.md` (full audit synthesis, F-LEAK through F-SME) — every number below is traceable to a ledger entry (`results/RESULTS_LEDGER.md`, L001–L036) cited inline.

**FRAMING RULE — applies to every section below, no exceptions:** this is a **self-audit of our own analysis pipeline** applied to a public dataset (ds005189: Helbing, Draschkow & Võ, 2025, *JoCN*). The dataset's original authors made no claim this paper contradicts — their published findings are not evaluated or contested here. Nothing in this outline, and nothing in the eventual manuscript, may read as a critique of the dataset, its collection, or its authors. The subject throughout is: *our pipeline produced a misleading headline number, here is how we found out, and here is what we can and cannot conclude as a result.*

---

## Title decision (resolved)

Candidate A selected, amended: *"Calibrated Accuracy Is Not Diagnostic of Task Content in Blocked EEG Decoding: A Self-Audit and Diagnostic Protocol."* Rationale (user-supplied): candidate B's "Without Task Information" reads as a claim about the real contrast specifically, reintroducing the original over-claim in softer form — A claims only what evidence lines 1–3 and 9 jointly support (§3), and naming the protocol contribution in the title is what makes the paper citable as a methods reference, not just a single-dataset correction.

---

## Abstract sketch (revised to match C1 — no longer over-claims a single mechanism)

- A blocked binary EEG contrast (Search vs. Memorize) with within-session per-subject calibration produced an apparently strong (~71%) decoding result.
- A structured audit found: **(1)** the result is entirely a product of within-subject calibration — cross-subject zero-shot accuracy never exceeds chance across five independent test configurations (0.48–0.53); **(2)** pseudo-labels carrying zero task content reach the same calibrated accuracy as the real contrast, and this pseudo-contrast effect is explained by a previously-undocumented self-paced rest-break discontinuity at a consistent location within every task block (30/30 subjects); **(3)** however, removing the rest-break region does **not** reduce the real contrast's accuracy (0.7215 vs. 0.7078 unrestricted; pre-calibration still at chance) — the rest-break mechanism accounts for the pseudo-contrast findings but not the real one, leaving a mutually-confounded, unresolvable block-level stack (task instruction, stimulus set, session half, post-break state) as the remaining explanation; **(4)** a confound-free control (subsequent memory) found no detectable signal, though underpowered.
- Conclusion: calibrated accuracy in this design is not diagnostic of task content. Contribution: a reusable diagnostic protocol for detecting this class of artifact in blocked-design EEG/BCI decoding more generally.

---

## Structure

### 1. Introduction
- Motivate the general problem: blocked-design EEG/BCI decoding + per-subject calibration is common; when does a strong reported accuracy reflect the calibration step exploiting incidental temporal/structural regularities rather than the intended cognitive contrast?
- State the paper's contribution as a **diagnostic protocol**, demonstrated on one dataset, generalizable to any blocked-design + within-session-calibration decoding pipeline.
- Preview the ten evidence lines (§3) as the paper's spine.
- Explicit framing-rule statement here, early, in the paper's own voice (see Framing Rule above).

### 2. Methods — the pipeline as actually implemented
- Dataset: ds005189, 30 subjects, sub-09 excluded (truncated raw EEG file at source — reproduced directly, not merely reported; markers intact, continuous signal stops at ~110s of an intended ~48min session).
- Preprocessing: filter (1–40 Hz), resample (250 Hz), epoch (−0.2 to 0.8 s), pre-F3 pooled-only Euclidean Alignment — **describe as actually run, including that this alignment is NOT the parametrized module developed later in the audit and never reached production use.**
- **No artifact rejection, no ICA, and no EOG regression are applied anywhere in the pipeline — and none of the source EEG files contain a dedicated EOG channel to regress against even if this were attempted.** State as a material methods fact, not an aside — it is what motivates and necessitates §3b's ocular controls.
- Feature pipeline: trial covariance → tangent-space projection → StandardScaler → PCA(35) → shrinkage-blended per-subject calibration classifier.
- Calibration mechanism described precisely and honestly: 15% of the held-out subject's own labeled trials fit a local classifier; CV-selected shrinkage blend with the global (other-28-subjects) classifier; scored on the remaining 85%. State plainly that this makes the design **few-shot subject-adaptive**, not subject-independent, and that this is the single most consequential methodological fact in the paper.
- **Explicitly disclose: the two originally-planned model branches (tangent-space spatial classifier and a Mamba-based temporal branch) were never jointly trained in any script this audit examined** — only the tangent-space branch's numbers are load-bearing anywhere in this paper.
- Evaluation: leave-one-subject-out, single seed=42 for all diagnostic controls (5-seed protocol reserved for, and only used in, the original headline numbers being audited — state which numbers carry which seed protocol).
- C3 plausibility hardening: describe the class-balance/majority-rate/balanced-accuracy/AUC reporting standard adopted mid-audit (motivated by the F-DRIFT-E balance failure, §6) and applied to every control from that point forward.
- **State the pre-registration discipline explicitly, once, plainly (D2):** every verdict threshold reported in this paper was recorded in the supplementary pre-registration record (see Data & Code Availability) BEFORE the corresponding run, and the mechanical verdict is reported as computed even in the three cases (§6) where later analysis revised its interpretation. This is the reason the three disclosed mis-specifications read as evidence of rigor rather than inconsistency — say so directly, don't leave the reader to infer it. Cross-reference forward to §6 and to the Data & Code Availability section's pre-registration record.

### 3. Results — the ten evidence lines, in order

Each line below: state the number(s), the design that produced them, and the single-sentence inference. Full statistical detail (CIs, p-values, per-subject tables) goes in the corresponding subsection, not the headline sentence.

1. **The 70.78% headline under nominal LOSO.** Matched spatial-only control, standard 29-fold LOSO, calibrated. This is the number our initial analysis reported as evidence of subject-independent decoding. (Ledger L009/L011.)
2. **Pre-calibration accuracy at chance; the protocol is few-shot subject-adaptive, not subject-independent.** pre_cal=0.5201 vs. post_cal=0.7078 on the identical fold structure — the entire lift is attributable to the 15% calibration step. State this as the paper's first and most direct correction of the initial protocol description. (Ledger L009/L011, §2.2 of SYNTHESIS.md.)
3. **Cross-subject transfer at chance across five structurally different configurations: 0.4839 (cross-parity) / 0.4916 (onset-excluded) / 0.5083 (phase-matched) / 0.5201 (unrestricted) / 0.5303 (within-parity).** No configuration exceeds chance by more than the sampling-noise band. This is the pre-calibration invariant (§4 of SYNTHESIS.md) — the firmest single claim in the paper: there is no subject-general representation of task identity to find, under any manipulation tried. (Ledger L003/L009/L011/L017/L020/L031/L032.)
4. **Zero-task-content pseudo-contrast at 0.7112 vs. real contrast at 0.7078.** A pseudo-label constructed from an early/late split within ONE real task block (task held constant) reaches calibrated accuracy statistically indistinguishable from the real Search-vs-Memorize contrast. This is F-DRIFT, the project-controlling finding. (Ledger L011.)
5. **Interleaved pseudo-labels at 0.5023 — the effect is temporal, not an artifact of pseudo-labeling per se.** Near-zero-separation pseudo-label (odd/even interleaving within the same block) collapses to chance, ruling out "any pseudo-label produces this" as an alternative explanation and confirming the effect scales with something about trial position. (Ledger L014, F-DRIFT-B.)
6. **Break-exclusion collapses every pseudo-contrast to chance at every scale tested.** The F-DRIFT-C dose-response curve (originally read as evidence of a temporal-separation gradient, rho=0.8571 p=0.0068) collapses entirely (pre_cal_balanced 0.4974–0.5037 across all 7 k-values) once the first 50 trials of each block are excluded. This is the paper's pivot point for the *pseudo*-contrast findings: "temporal separation" is withdrawn as an explanation and replaced with the rest-break account. **Present alongside the original (withdrawn) interpretation explicitly — do not silently drop it; §6 mis-specifications section covers the full reasoning.** (Ledger L016/L028, F-DRIFT-C + F-DRIFT-F(a).)
7. **The rest-break discontinuity: 30/30 subjects, event index 49 (or 54 in 1 subject).** Direct evidence for the mechanism behind line 6: a large, consistent temporal gap in the raw marker stream at a highly reproducible location within every task block, independent of and prior to any classifier. This is what makes the paper a structural/behavioral finding, not merely a statistical one. (Ledger L029/L033; full 30-subject table as a figure, §Figures below.)
8. **Within-task split across the break: 0.8585 vs. true task boundary: 0.8968, matched N/balance/span.** A pseudo-label with ZERO task content, entirely inside one block, reaches within 0.04 of the real task boundary's accuracy under a design that exactly controls for sample size, class balance, and temporal span. This is the single strongest quantitative piece of evidence that the true task boundary is not privileged relative to a generic block-onset-proximity effect. (Ledger L025, F-DRIFT-E redesign.)
9. **⚠ LOAD-BEARING, NOT OPTIONAL — F-DRIFT-G: removing the rest-break region does NOT reduce the real contrast.** post_cal_balanced=0.7215 vs. 0.7078 unrestricted (statistically unchanged); pre_cal_balanced=0.4916, still at chance. **Inference: the rest-break mechanism (lines 6–8) is not the explanation for the real result.** Once it is controlled for, what remains is a block-level confound stack (§3a, immediately below) that this design cannot separate from any genuine task signal. **This line is what stops the paper over-claiming a single mechanism — it must appear in the main results narrative, not be relegated to a footnote or the limitations section.** (Ledger L031.)

#### 3a. The block-level confound stack (placed here, between lines 9 and 10 — not after line 10: line 10's confound-free status only makes sense once the reader has this list in hand)

- Motivated directly by line 9: once the rest-break mechanism is ruled out as the explanation for the real contrast, name exactly what remains. Four members, stated explicitly, as a stack that is **mutually confounded and unresolvable within this design** — no manipulation available in ds005189 can vary one while holding the other three fixed:
  1. **Task instruction** (Search vs. Memorization) — differs by block, by definition of the contrast itself.
  2. **Stimulus set** — see F-STIM below for the precise, corrected statement of how this operates.
  3. **Session half** — block 1 is always the first half of the session, block 2 always the second, for every subject.
  4. **Post-break state** — block 2 always follows the ~400s inter-block break that block 1 does not, even after each block's own internal rest-break region is excluded.
- **F-STIM folded in here, with the corrected framing (do not repeat the earlier, inverted phrasing):** within a single subject, the Search and Memorization stimulus sets are **fully disjoint** (Jaccard=0 — no scene/object is shown to a given subject under both conditions). Taken alone, within one subject, this disjointness is a real, unremovable confound: stimulus identity and task class are perfectly collinear for that subject. **What prevents this from being learnable across the cross-subject training pool used in every LOSO fold is that the SAME ~100 stimuli recur across the cohort, counterbalanced per subject** (F-STIM: 21 distinct stimulus-to-condition partitions across 30 subjects, verified — ledger L036) — so no single stimulus reliably predicts class *across* subjects, even though it does perfectly predict class *within* any one subject. State this precisely: cross-subject counterbalancing is what keeps stimulus identity from being **an** exploitable shortcut in a pooled cross-subject classifier — it does not "rule out" the confound, it only prevents this specific pipeline from exploiting it at the population level. (Ledger L036; `results/SYNTHESIS.md` §2.3.)
- State plainly: this stack, not the rest-break mechanism, is the best remaining candidate explanation for why the real contrast survives onset/break exclusion — and it is not resolvable with any control available in this dataset (see `results/SYNTHESIS.md` §7, "cannot be resolved in this dataset").

10. **Subsequent memory: null and underpowered; search superiority as the reason `within_search` was unevaluable.** The one confound-free control available in this dataset (remembered vs. forgotten, within-block/within-task/within-stimulus-set — see §3a immediately above for why this control is confound-free and the others are not) found AUC indistinguishable from 0.5 in both scoreable conditions (within_memorize 0.4968 [0.4489,0.5423] n=13; pooled 0.5141 [0.4760,0.5547] n=21), and could not even be run for Search trials at all (0/29 subjects reached the minority-trial threshold) — itself a behavioral replication of the dataset's own documented search-superiority effect, reported as a finding, not just a limitation. State explicitly: this null is a power statement (0–18% minority trials/subject), not evidence against subsequent-memory effects existing. (Ledger L034/L035.)

#### 3b. Alternative explanations considered and ruled out (renumbered per correction E — moved inside Results as a subsection, following line 10, so the reading order is 1–9, 3a, 10, 3b rather than 10-then-a-separate-§4; every section below is renumbered one lower as a result)

- Present as substantive completed work, not a caveat. Three independent tests, each with its own pre-registered threshold, verdict as computed:
  1. **Frontal-channel ablation.** Drop = 1.06 pp (seed-42 basis), landing at the 90th percentile of a 20-draw random-channel-ablation null distribution; one-sided p=0.143 — **not significant** (pre-registered threshold p<0.10). (Ledger L004/L005.)
  2. **Decision-margin vs. surrogate-EOG correlation — two distinct quantities, reported separately, neither dropped:** **signed** decision margin vs. surrogate EOG: Spearman ρ≈−0.085. **|decision margin|** vs. surrogate EOG: ρ = −0.015 and +0.040 (across the two surrogate variants). **Both far below** the pre-registered |ρ|≥0.2 "strong" threshold. (Ledger L006.)
  3. **Frontopolar-vs-central-parietal variance gap, cohort-wide.** Did **not replicate** at 29 subjects (+18.30% frontopolar vs. +8.83% central-parietal mean gap; only 16/29 subjects directionally consistent; Wilcoxon p=0.137, not significant) — the original single-subject (sub-01) finding that motivated this line of investigation was an outlier in magnitude, not a cohort-wide effect. (Ledger L007.)
  - **Combined verdict: CLEAN** — no significant evidence of ocular-artifact contamination across three independent, pre-registered tests.
- **Caveat kept attached, not dropped:** a seed-42-vs-5-seed basis-mismatch in test 1 remains formally unresolved (blocked on missing per-seed data) — disclose it, and note explicitly that the contrast it was designed to gate (Search-vs-Memorize decoding) is no longer load-bearing for any claim in this paper regardless, so the unresolved caveat does not weaken the paper's actual conclusions.

### 4. The diagnostic protocol as a reusable recipe
- Extract as a standalone, generalizable checklist independent of this dataset:
  1. Always report pre-calibration (zero-shot) accuracy alongside any calibrated number.
  2. Test a temporally-matched, zero-task-content pseudo-label before trusting a real-contrast result in any blocked design.
  3. Vary pseudo-label temporal separation parametrically; do not trust a single two-point comparison.
  4. Before excluding a data region as a "transient," identify what it structurally is (raw marker/timing inspection) rather than assuming a smooth decay.
  5. When a claimed effect is between-block, verify that within-block manipulations cannot silently stand in as evidence about between-block confounds — and if a within-block manipulation leaves the real effect unchanged (as here), say so explicitly rather than treating the earlier pseudo-contrast finding as if it already explained the real one (evidence line 9; §6, mis-specification 3).
  6. Harden every classification report with class balance / majority-rate / balanced accuracy / AUC as appropriate — a plausibility check phrased only as "accuracy in [0,1]" will not catch a base-rate artifact.
  7. Where a genuinely confound-free contrast exists in the data (here: subsequent memory), test it explicitly rather than resting the paper's conclusion on process-of-elimination alone.

### 5. Limitations
- Pre-F3 alignment dependency: every control uses pooled-only EA, not the parametrized module developed mid-audit; conclusions are conditional on this specific alignment.
- Single-seed basis for all diagnostic controls (seed=42 only) — a deliberate cost tradeoff, disclosed, not a 5-seed protocol.
- F-SME is underpowered by construction (0–18% minority trials/subject); its null result cannot be extended to a claim that subsequent-memory effects don't exist in this or similar paradigms.
- Rest-break discontinuity's exact cognitive/physiological cause (arousal, re-orientation, impedance drift) is not established — only the structural timing coincidence.
- Fixed-count vs. proportional-to-block-length nature of the rest-break location is inconclusive given the narrow observed block-length range (200–205 events).
- The block-level confound stack (§3a) is, by construction, unresolvable within this dataset — this is stated here as a limitation of what the audit *can* conclude, in addition to its treatment as a results finding in §3a.

### 6. The three disclosed criterion mis-specifications (dedicated section, main text)
- Present as a methodological contribution in its own right, not an appendix apology: pre-registration discipline caught its own failures three times, and each is reported with the original (wrong) criterion's mechanical result alongside its correction — modeling exactly the practice recommended in §4's protocol.
  1. **F-DRIFT-C:** strict pointwise monotonicity, too fragile for 7 noisy single-seed points → corrected via Spearman + bootstrap CI + paired Wilcoxon (both robust, both confirmed the trend) → later superseded in kind (not number) by evidence line 6.
  2. **F-DRIFT-E:** original design let pseudo-class size vary freely with shift position, conflating accuracy with base rate → corrected via a fixed-symmetric-window redesign guaranteeing exact 50/50 balance, plus a codebase-wide balance-hardening standard adopted from that point forward.
  3. **F-DRIFT-G:** "survives onset/rest-break exclusion" was equated with "genuine task signal" — a within-block manipulation was treated as informative about a between-block confound it cannot touch (task instruction, stimulus set, session half, post-break state all remain intact). Corrected by disclosing the fired branch alongside the reasoning for why it doesn't support the inference the rule implied — this is the mis-specification that directly produced evidence line 9 and the §3a confound-stack framing.

### 7. Discussion
- Position relative to the broader BCI/EEG-decoding calibration literature: when is a "subject-independent, calibrated" decoding claim actually a subject-adaptive one, and how would a reader detect the difference from a paper's methods section alone (usually: they can't, without the pre_cal number).
- Position relative to blocked-design confound literature (fatigue/vigilance/drift-as-confound is a known concern; this paper's contribution is showing a SPECIFIC, structurally-locatable event — not a vague drift — drives the pseudo-contrast findings, demonstrating a protocol to find it, AND showing that ruling out that one mechanism is not sufficient to explain the real contrast, since a separate confound stack remains).
- Explicitly state what this paper does NOT claim: it does not claim search-superiority or subsequent-memory effects are absent from this dataset or paradigm generally (§5's power caveat); it does not identify a single definitive cause of the real contrast's residual accuracy (§3a is a stack, not a resolved mechanism); it does not critique the original dataset's collection or its authors' published analyses, which used different methods and made different claims than the pipeline audited here.

### 8. Conclusion
- Restate the headline conclusion (§1 of SYNTHESIS.md) in one paragraph — calibrated accuracy in this design is not diagnostic of task content — and restate the diagnostic-protocol contribution as the paper's generalizable takeaway beyond this one dataset.

### 9. Data & Code Availability (D1 — new section; the largest gap in the prior outline, and more consequential here than in an ordinary paper, since the paper's entire claim rests on an audit trail being verifiable)

- **Source dataset:** ds005189 on OpenNeuro. **Accession:** ds005189. **DOI:** `doi:10.18112/openneuro.ds005189.v1.0.1`. **License:** CC0. (Verified directly against the dataset's own `dataset_description.json` — not assumed.)
- **This project's code repository:** public repository containing every control script referenced in this paper (F-LEAK through F-SME, ~30 scripts), `results/RESULTS_LEDGER.md` (the numeric ledger, L001–L036), and `results/SYNTHESIS.md` (the full synthesis this paper is drawn from). **Commit hash: TBD** — repository state as of the outline date carries substantial uncommitted work from this audit; a specific commit will be tagged and cited at submission, not before. **License: TBD** — no repository license has been chosen yet; needs a decision before submission (not before drafting).
- **`DECISIONS.md` as supplementary material, presented explicitly as the pre-registration record** — every verdict threshold cited in §3/§6 traces to a dated entry in this file, recorded before the corresponding run. Cross-referenced from Methods (D2).
- **Ethics:** this paper conducts no new data collection; it re-analyzes the existing ds005189 recordings under their original approval. **Source dataset's IRB approval, per its own documentation:** Ethics committee of the Faculty of Psychology and Sports Sciences, Goethe University Frankfurt, approval ID **2014-106R1**. State plainly that this paper's re-analysis is covered by, and does not require separate approval beyond, this existing approval — consistent with standard secondary-analysis practice for openly-licensed (CC0) data; confirm against journal policy at submission.

---

## Figures to plan (NOT built yet — placeholders only)

1. **Four-contrast comparison table/figure** — interleaved (0.5023) / zero-content within-block (0.7112) / real unrestricted (0.7078) / real onset-excluded (0.7215), pre_cal and post_cal side by side, task-content flag. (Source: RESULTS_LEDGER.md L021 master table.)
2. **K-sweep with and without break exclusion** — the F-DRIFT-C curve (rho=0.8571 trend) overlaid or paneled against the F-DRIFT-F(a) onset-excluded collapse (0.4974–0.5037 flat) — the paper's clearest before/after visual. (Ledger L016/L028.)
3. **Boundary-position profile** — the F-DRIFT-E 7-position balanced-accuracy profile (block1/block2 × 25/50/75% + true boundary), showing the onset-proximity pattern (high near onsets, flat mid-block). (Ledger L025.)
4. **Rest-break dot-plot across 30 subjects — form decided (C1 open question, now resolved): subject on one axis, event index on the other, gap duration encoded as point size.** The 30/30 convergence at index 49 (with sub-01 at 54 as the sole visible outlier) should be readable at a glance without reading the underlying table. (Source data already compiled: `results/rest_break_discontinuity_table.md`. Ledger L033.)
5. **Pre-cal/post-cal decomposition** — the five-configuration pre-calibration-invariant table (§4 of SYNTHESIS.md) paired with each configuration's post_cal number, visually separating "always chance" (pre_cal) from "varies with structure" (post_cal). Consider adding F-DRIFT-G's post_cal (0.7215) as a sixth point here to visually reinforce evidence line 9 — the pre_cal bar stays flat at chance while post_cal stays elevated, exactly like every other configuration. (Ledger L032 + scattered post_cal values throughout.)

---

## Decisions carried forward from user review (no longer open questions)

- **§6 mis-specifications:** main text, as outlined.
- **Evidence line 10 (subsequent memory):** Results proper, as outlined.
- **Figure 4:** dot-plot, subject × event index, gap duration as point size (see Figure 4 above).
- **Author list / contribution statement:** to be supplied by user. **[PLACEHOLDER — author list and contribution statement pending.]**

## Remaining open question

- Figure 4's exact rendering details (color, axis ordering, whether to mark sub-01's index-54 outlier distinctly) — a decision for when figure-building actually begins, not before.
