# results/METHODS_FACTS.md — Extracted Methods Facts

**Purpose:** docs-only extraction of methodological facts established during Phase 0 (AUDIT.md's Q1–Q7) and the Gate 0/Gate 1 scoping decisions (DECISIONS.md's D1–D4), collected in one place for Phase 3 manuscript drafting. This file states facts and cites their evidence; it does not report any Condition-4 accuracy number (D3's hard gate still applies — see DECISIONS.md).

Fix-IDs closed by this file: **F1** (empirical tensor-shape confirmation), **F2** (this extraction), **F6** (docs), **F12** (docs extraction of an already-verified finding).

**Status:** WRITTEN 2026-08-18. Not yet reviewed for Phase 3 manuscript language — this is a facts ledger, not manuscript prose.

---

## 1. Temporal-branch input tensor (F1, confirms AUDIT.md Q1 empirically)

`scripts/dump_tensor_shapes.py` was written and **executed locally** (no GPU/Modal needed — synthetic input tensor only) to confirm Q1's narrative claim with real forward-hook output, not just a source-code read.

**Confirmed input:** `x_signal : (B, C=62, T=251), float32` — the EA-whitened, z-scored raw signal (not a spectrogram or other transform).

**Confirmed shape chain** (`D_MODEL=16, D_STATE=8, N_LAYERS=1`, the production hyperparameters):

```
(B, 62, 251) --transpose(1,2)--> (B, 251, 62)
            --Linear(62→16)--> (B, 251, 16)
            --1× SelectiveSSMBlock--> (B, 251, 16)
            --LayerNorm--> (B, 251, 16)
            --mean-pool over T--> (B, 16)
            --Linear(16→2) [classifier head]--> (B, 2)
```

Inside the one `SelectiveSSMBlock` (`expand=2, dt_rank=4, conv_kernel=4`): `in_proj` expands 16→64 (2×32 split for x/z gates), `conv1d` operates on the 32-dim inner channel, `x_proj` produces the 20-dim (`dt_rank=4 + 2×d_state=8+8`) delta/B/C decomposition, `dt_proj` maps back to the 32-dim inner width, `out_proj` contracts 32→16.

Full per-layer shapes and raw hook output: `results/tensor_shapes.json`.

**Source:** `run_step4_condition4_asymmetric_mamba.py:302–324` (`MambaTemporalBranch`), confirmed against the deliberately-duplicated model classes in `scripts/benchmark_efficiency.py` and `scripts/dump_tensor_shapes.py` (same rationale as F15: a plain local script can't `import` a Modal-decorated function's local classes, so the architecture is copied verbatim and kept in sync by construction — same forward-pass code, hand-checked against the production file at each duplication site).

---

## 2. Temporal-branch training regime (Q2) — LEGACY pretrain-freeze script only

**Applies to `run_step4_condition4_asymmetric_mamba.py` (the legacy arm).** `run_step4_condition4_joint_dualbranch_mamba.py` (F-JOINT) uses a genuinely different, single end-to-end `nn.Module` with joint backprop — the facts below describe the *legacy* arm specifically, and the two must not be conflated in Methods text.

The legacy temporal branch is trained in a **separate, earlier stage** from the classifier that ultimately makes the LOSO prediction — it is **not** end-to-end with the final calibrated classifier:

1. `TemporalOnlyModel` = `MambaTemporalBranch` + a throwaway `nn.Linear(16,2)` classifier head.
2. Trained per-fold on the 28-subject training pool only (90/10 internal stratified split for early-stopping validation), `nn.CrossEntropyLoss()`, `AdamW(lr=1e-3, weight_decay=1e-2)`, `CosineAnnealingLR(T_max=30)`, gradient clip norm 1.0, early-stop patience 8 on validation loss.
3. Best-validation-loss checkpoint reloaded, then **every parameter is frozen** (`requires_grad=False`) and the model is set to `eval()`.
4. 16-dim embeddings extracted with `@torch.no_grad()` for both the 28-subject pool and the held-out subject.
5. These frozen embeddings are concatenated **at the NumPy level** with the raw 1953-dim spatial tangent vector — no gradient ever flows between the two branches, and no gradient flows from the final classifier back into the Mamba weights at all.
6. The fused 1969-dim vector then goes through `StandardScaler → PCA(35) → shrinkage-blended LogisticRegression` (classical, non-differentiable) — this is the classifier whose accuracy is reported.

**In one sentence:** in the legacy arm, the Mamba branch is pretrained end-to-end against its own private classification head and frozen; the reported LOSO accuracy comes from a completely separate downstream linear-calibration stage that only ever sees the frozen embedding, never a gradient. F-JOINT exists specifically to test whether this pretrain-freeze design (vs. genuine joint training) affects the result.

**Source:** `run_step4_condition4_asymmetric_mamba.py` (training loop + `extract_temporal_features`, `:453–495` in the Phase 0 read).

---

## 3. Preprocessing pipeline (Q7)

| Parameter | Value |
|---|---|
| Original sampling rate | 1000 Hz |
| Resampled to | 250 Hz |
| Band-pass | 1.0–40.0 Hz, FIR/Hamming |
| Notch | None applied |
| Reference | Contralateral mastoids (TP9, TP10) |
| Ground | FPz |
| Montage | Extended 10-10, 62 channels (all scalp-named; see note below) |
| Epoch window | −0.2 s to +0.8 s (1.0 s, 251 samples @ 250 Hz) |
| Baseline correction | (−0.2 s, 0 s) |
| Artifact rejection | None (`reject=None`) — no ICA, no channel-level quality control |
| ICA | Not used in the main pipeline (F-OCULAR(b) adds an ICA-cleaned *control* arm, not a pipeline change) |

**Channel-count note (supersedes an earlier Phase-0 read):** the BIDS sidecar declares `EOGChannelCount: 1`, but F-OCULAR(a)'s live channel-order verification (re-downloading sub-01's raw `.vhdr` and reading `raw.info['ch_names']` directly via MNE) found all 62 channels are scalp-named — no channel in the actual montage is labeled as a dedicated EOG channel. This is why F-OCULAR(a)/(c)/(d) all re-verify channel order live rather than trusting the sidecar's channel-count metadata; see DECISIONS.md's ocular-artifact interpretation matrix for how this gap in the montage motivates the F-OCULAR family of controls.

**Source:** `step1_1_fetch_inspect.py`, `step1_2_filter_epoch.py`, `run_data_engine_on_modal.py`, cross-checked against `data/ds005189/sub-01/eeg/sub-01_task-SearchSupRecFam_eeg.json`.

---

## 4. Dataset identity and framing (F12, D1, D4 — VERIFIED, not just extracted)

**Dataset:** OpenNeuro **ds005189**, "Search Superiority Recollection Familiarity" (Helbing, Draschkow & Võ, Scene Grammar Lab, Goethe University Frankfurt), DOI `doi:10.18112/openneuro.ds005189.v1.0.1`, CC0.

**Source publication (D4, verified via WebSearch + dataset README/dataset_description.json cross-check, not reconstructed from the dataset title alone):**

> Helbing, J., Draschkow, D., & Võ, M. L.-H. (2025). "Incidental Encoding of Objects during Search Is Stronger Than Intentional Memorization due to Increased Recollection Rather Than Familiarity." *Journal of Cognitive Neuroscience*, 37(12), 2538–2557. https://doi.org/10.1162/jocn.a.80

**This is NOT a true/false-memory (DRM-style) paradigm.** `EVENT_ID` groups four "Search" stimulus codes (10–13) into Class 0 (incidental encoding) and four "Memorize" stimulus codes (20–23) into Class 1 (intentional encoding). The task is "Incidental vs. Intentional Memory Encoding" during the **encoding phase** (searching for a target vs. intentionally memorizing an object) — there is no recognition-phase hit/false-alarm/lure labeling anywhere in this codebase. This directly contradicted the manuscript's original framing (title, abstract, "True Memory"/"False Memory" class names) — see D1 for the resulting reframing (methodological paper, Search-vs-Memorize primary contrast, honestly labeled incidental-vs-intentional encoding) and D4 for the resulting citation rescoping (source publication above replaces any DRM/false-memory citations; retained Paller & Wagner scoped specifically to the SECONDARY subsequent-memory analysis).

**Subject exclusion (D2):** dataset has 30 raw participants; pipeline runs on **n=29**. Root cause: sub-09's `.eeg` binary file is truncated to 109.8 s of an intended ~41.5-minute session, leaving zero events inside the available data range — a source-data export truncation, not a pipeline bug. The odd/even block-order counterbalance (designed for 15/15 out of 30) becomes **14 odd / 15 even** after excluding sub-09 (odd-numbered) — this asymmetry should be stated alongside the exclusion criterion in Methods.

**Trial counts and balance (Q6):** 9,869 total trials (Class 0: 4,935, Class 1: 4,934 — near-perfect balance), a property of the experimental design (`EVENT_ID` mapping), not a pipeline balancing step. No explicit undersampling/class-balancing code exists in the data pipeline. **Flagged as needing independent confirmation** against `participants.tsv`/`events.tsv` counts before stating "balanced by design" as settled fact in the paper (not yet done as of this writing).

---

## 5. Euclidean Alignment reference (Q4) — pre-F3 baseline, now superseded

**Historical fact, not current behavior:** prior to F3's fix, both `run_step4_condition4_asymmetric_mamba.py` and `run_step4_matched_spatial_control.py` fit EA whitening using a single **pooled** covariance mean across all 28 training subjects (`fit_ea_whitening`), then applied that same pooled `W` to the held-out test subject — i.e., the test subject was whitened using a reference derived entirely from other people's data, not their own. This contradicted the per-subject-reference mechanism in He & Wu (2020)/Kobler et al. (2022) and was the single highest-confidence Phase-0 finding (motivating F3).

**Current behavior (post-F3):** `eeg_alignment.py`'s `--ea-mode {none,pooled,per-subject,riemannian}` flag makes this an explicit, tested choice; `per-subject`/`riemannian` modes give the held-out subject its own unsupervised whitening computed from its own unlabeled trials (still label-free, still leakage-safe — EA never touches `y`). "Pooled" is retained as a mode (not removed) so the original reported numbers remain reproducible for comparison, but the fix's whole point is that "pooled" is no longer the *only* option.

---

## 6. Normalization between tangent vector and PCA (Q3)

A single joint `StandardScaler` is fit on the concatenated fused vector (spatial tangent + temporal embedding, 1969-dim in the legacy asymmetric-fusion arm), standardizing each of the 1969 individual feature columns to unit variance — this is **not** a per-block scaler that equalizes the spatial block's and temporal block's total contribution to the variance budget PCA(35) subsequently maximizes over. After this scaling, the post-scaling covariance trace is exactly 1969 (each column contributes 1), so the 16 temporal columns can contribute at most 16/1969 ≈ 0.81% of the total variance regardless of how informative they are per-dimension — a real, verifiable mechanism for PCA to systematically under-select temporal directions. This is the empirical basis for F11's `--fusion-mode {legacy,scaled-concat,split-pca}` flag, which adds per-block scaling (and optionally per-block PCA) as alternative arms.

**Source:** `run_step4_condition4_asymmetric_mamba.py` (fusion + calibration section).

---

## 7. Hyperparameter-selection provenance (Q5) — feeds F8/F-CAPACITY scoping

| Hyperparameter | Verdict |
|---|---|
| `PCA_MAX_COMPONENTS=35` | Fixed a priori (sample-size-vs-dimension argument: N_cal≈60 > P=35 keeps calibration LR well-posed), not LOSO-tuned. |
| `COV_SHRINKAGE=0.10` | Fixed a priori, but undocumented derivation — not evidence of LOSO-accuracy tuning, but not a principled analytic choice either. Motivates F9's `--cov-estimator {fixed,lwf}`. |
| `D_MODEL=16, D_STATE=8, N_LAYERS=1` (temporal Mamba) | No evidence of LOSO-accuracy tuning, but also **no evidence of any systematic capacity search ever having been run** — asserted a priori "keep it tiny" defaults. Motivates F-CAPACITY's sweep. |
| Calibration shrinkage blend (`SHRINK_GRID`) | Safe — legitimately selected via internal 3-fold CV on the calibration split only, never touching the test split. Reused verbatim (not re-searched) by every driver built since. |

**Conclusion:** no confirmed leakage of hyperparameters against LOSO test accuracy was found, but the Mamba capacity hyperparameters were never validated by any method — purely asserted defaults. This is why F-CAPACITY (capacity sweep) and F8 (nested-CV validation of PCA/logreg-C/SVC-C) are both scoped as mandatory Phase 2 items rather than optional cleanup.

---

## 8. Calibration protocol — exact mechanism (added 2026-08-19, Tier-1 manuscript correctness defect support)

**Why this section exists:** the manuscript describes the held-out subject as "held out entirely," with "zero test-subject leakage into any stage" and "strict subject-independent" evaluation. Those statements are **false as written**. Every reported Condition-4 (and Condition-1b/EEGNet) accuracy number in this codebase is **post-calibration**: after the held-out subject is selected, 15% of *that subject's own labeled trials* are used to fit a personalization step before the final 85% is scored. This is a legitimate, trial-disjoint, no-label-reuse methodology (F-LEAK verified no leakage across the train/calibration/test boundary — see AUDIT.md/STATUS.md) — but it is **few-shot subject-adaptive calibration, not subject-independent generalization**, and the manuscript's current language misdescribes it. See STATUS.md's Tier-1 flag (this date) and RESULTS_LEDGER.md's L009 for the supporting pre_cal-vs-post_cal numbers.

**Exact mechanism** (identical across every LOSO driver built on the shared calibration pattern — `run_step4_matched_spatial_control.py`, `run_step4_condition4_asymmetric_mamba.py`, `run_step4_eegnet_ea.py`, and all F-OCULAR variants):

```python
sss = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - CAL_FRACTION), random_state=seed)
cal_idx, test_idx = next(sss.split(feat_k, y_k))
```

with `CAL_FRACTION = 0.15`.

- **How the 15% is sampled:** `sklearn.model_selection.StratifiedShuffleSplit` stratifies **by class label only** (`y_k`, the held-out subject's labels). Within each class, it selects indices via a uniform random permutation seeded by `random_state=seed` — **not** by block, not by time-position, not by any other structure. Because class is (per F-STIM/D2) perfectly confounded with block within a subject's session, a class-stratified split is *implicitly* a per-block split too (each class's 15% necessarily comes from that class's own block) — but *within* that block, the specific trials chosen for calibration are a uniform-random subset, not concentrated at any particular point in the block's timeline.
- **Class balance:** yes, exactly — stratification guarantees the calibration sample's class ratio is not just approximately but proportionally derived from each class's own count.
- **Position-in-block / time distribution:** **not explicitly controlled, and not yet empirically measured against real trial data in this codebase.** Because the underlying sampling mechanism (`numpy`-permutation-based, order-agnostic) has no dependency on chronological trial order, the calibration sample is *expected in principle* to be roughly uniformly spread across a block's early/mid/late thirds — but this has not been directly verified against the real per-subject trial timestamps, and F-DRIFT (new fix-ID, see AUDIT.md/DECISIONS.md, 2026-08-19) is specifically designed to test whether within-session time-position signal is being exploited by this same calibrated pipeline, which requires knowing (or bounding) how much time-position signal the calibration sample itself has access to. Recommended follow-up: instrument one real run to log the position-third distribution of `cal_idx` per fold.
- **Are calibration trials excluded from scoring?** Yes, by construction — `cal_idx` and `test_idx` are a disjoint partition of the held-out subject's local trial indices (`sss.split` guarantees no overlap); only `test_idx` (the 85%) ever appears in a `predict`/scoring call. A calibration trial is never also a test trial.
- **What calibration trials are used for:** fitting `local_clf_full`/`local_fold` inside `fit_shrinkage_classifier` (the per-subject-personalized LogisticRegression, blended with the pool-trained `global_clf` via a CV-selected shrinkage weight — see AUDIT.md's Q5, judged leak-safe). Calibration trials are never used to fit the EA whitening, the PCA, or any pool-level (28-subject) component — only the final per-subject classifier blend.

**Numeric consequence (RESULTS_LEDGER.md L009):** pre-calibration (zero-shot) accuracy for the tangent-space conditions is ≈0.51–0.52 (statistically indistinguishable from chance), while post-calibration accuracy is ≈0.71. The ≈19–20 percentage-point gap is the calibration step's contribution, not the model's subject-independent discriminative power.

---

## Sources

This file extracts and lightly reorganizes AUDIT.md's Q1 (empirically re-confirmed by F1), Q2, Q3, Q4, Q5, Q6, Q7, and the Phase 0 ★ CRITICAL FINDING (F12), plus DECISIONS.md's D1, D2, D4. AUDIT.md remains the canonical narrative record if this summary and AUDIT.md ever disagree.
