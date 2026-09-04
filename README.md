# A Mislabeled Contrast, Recovered: Diagnosing and Correcting an Encode/Test-Phase Confound in Blocked EEG Decoding

This repository (`https://github.com/HassanSidd0946/Search-vs-Memorize-Correction`) holds the code and committed result artifacts behind a reanalysis of the public `ds005189` EEG dataset (Helbing, Draschkow, and Vo, 2025), written up as a manuscript by Muhammad Hassan Siddiqui and Muhammad Adil Usmani (both Independent Researchers, Lahore, Pakistan) targeting *NeuroImage: Reports*. The dataset is available on OpenNeuro (accession ds005189, DOI `10.18112/openneuro.ds005189.v1.0.1`) under a CC0 licence. The manuscript itself lives in `PAPER/draft.md`; this file describes what is in the repository and how to check a reported number against the code and data that produced it.

## What the analysis does

An earlier pass through this dataset used an event-code dictionary (`EVENT_ID`) that defined each of two decoding classes, Search and Memorize, as the union of four structurally distinct marker types: one encode-phase trial onset plus three separate recognition-test response types. Under that definition, 75% of each class's epochs were recognition-test responses, not the encoding event the label was meant to represent. A matched spatial-only decoding pipeline evaluated under this definition reported 70.78% balanced accuracy, a number that is arithmetically correct but not interpretable as a claim about Search versus Memorize decoding. A composition-arithmetic diagnostic (R1(a)) that predicts a construction's decoding accuracy from its real per-subject marker composition, computed with no classifier training, fit the historically reported effect in 13 of 13 checked cases. A direct classifier decode of encode-phase against test-phase identity on the re-epoched data (R1(b)) then reached 0.8657 plus or minus 0.0005 balanced accuracy against a pre-registered 0.85 threshold, confirming the composition account and withdrawing an earlier drift and rest-break-discontinuity account this project had proposed for the same contrast.

Restricting the contrast to encode-phase trial onsets only (R2, the corrected contrast) yields a pre-calibration balanced accuracy of 0.5773 plus or minus 0.0005, which is 7.73 plus or minus 0.05 percentage points above a chance floor near 50%. The corresponding post-calibration balanced accuracy is 0.7311 plus or minus 0.0005. Four independent controls establish the pre-calibration figure as a genuine, subject-generalizable signal: a shuffled-label null on the exact R2 dataset at 30 shuffles (C1, null mean 0.4990, SD 0.0103 plus or minus 0.0001) and again at 500 shuffles (C4, null mean 0.4995, SD 0.0112, p less than 0.002 by the add-one-corrected convention), a leave-one-subject-out jackknife (C3, LOO means ranging from 0.5703 plus or minus 0.0005 to 0.5852 plus or minus 0.0005), and a parity-split replication (D2, odd subgroup 0.6157 plus or minus 0.001, even subgroup 0.5582 plus or minus 0.001, each against its own within-group null). None of these four controls tests the post-calibration figure; the manuscript states this explicitly rather than implying validation it does not have. A third contrast (R3) drops only the New-Lure epochs and is reported as a descriptive composition-gradient benchmark, not a pass or fail result.

## Repository layout

```
PAPER/
  draft.md                      manuscript source (Markdown); the single authoritative text
  draft_elsevier.tex            build product of build_latex.py; do not edit directly
  draft_ieee.tex                build product of build_latex.py; do not edit directly
  draft_elsevier.pdf            compiled Elsevier-format manuscript
  draft_ieee.pdf                compiled IEEE-format manuscript
  references.bib                bibliography (Elsevier build)
  references_ieee.bib           bibliography, auto-generated from references.bib for the IEEE build
  supplementary_table_s1.md     Supplementary Table S1 source, regenerated from results_c3_r2_jackknife.json
  supplementary_table_s1.docx   Word conversion of the table above; not currently regenerated, see Open items
  cover_letter.md               draft cover letter, author details and the competing-interest statement filled in
  SUBMISSION_CHECKLIST.md       current submission state, item by item
  PROVENANCE.md                 index of every reported number, its source JSON, and its field path
  figures/
    fig1_discovery_timeline.{pdf,png}    through fig4_d2_parity_split.{pdf,png}: the four manuscript figures
    scripts/
      build_latex.py            reads PAPER/draft.md, writes both .tex files and references_ieee.bib
      fig1_discovery_timeline.py, fig2_c3_jackknife.py, fig3_null_distribution.py, fig4_d2_parity_split.py
                                 one script per figure; each reads its values from a committed result JSON
  archive/                       superseded drafts and outlines, kept for the audit trail

run_data_engine_granular_on_modal.py     Modal entrypoint: filters, epochs, and caches the corrected dataset
run_r1b_r2_r3_composition_runs.py        Modal entrypoint: R1(b), R2, and R3
run_c1_shuffled_label_control.py         Modal entrypoint: C1, the 30-shuffle null
run_c3_r2_jackknife.py                   Modal entrypoint: C3, the leave-one-subject-out jackknife
run_c4_high_res_shuffled_label_control.py Modal entrypoint: C4, the 500-shuffle null
run_d2_parity_split_check.py             Modal entrypoint: D2, the parity-split replication

results_r1b_r2_r3_composition_runs*.json           R1(b)/R2/R3 output; two files exist and disagree on four
                                                     fields, see PROVENANCE.md
results_c1_r2_shuffled_label_control.json           C1 output
results_c3_r2_jackknife.json                        C3 output, also the source for Table S1
results_c4_high_res_shuffled_label_control*.json    C4 output; five files exist (one canonical, one convenience
                                                     pointer, one earliest pull, two later reruns), see RESULTS_LEDGER.md
results_d2_parity_split_check.json                  D2 output

results/RESULTS_LEDGER.md      chronological numeric-results log for this project, entry per computed result
RESULTS_LEDGER.md              provenance table for the timestamp-and-hash-stamped result JSONs listed above
NUMBERS.md                     authoritative numbers table: every reported figure, its quantum, tolerance, and
                                containment check
GENERATOR_FIXES.md             record of defects found and fixed in build_latex.py
REPLACEMENTS.md                line-by-line record of what changed in draft.md and why
DECISIONS.md                   binding scoping decisions and every pre-registered threshold, dated before use
STATUS.md                      fix-ID state table for this audit
AUDIT.md                       chronological narrative log of the audit this repository underwent

results/                       supporting CSVs and markdown notes referenced by the ledger above
scripts/                       diagnostic and control scripts from earlier phases of this project
tests/                         unit tests (currently: alignment module tests)
```

Files not listed above (`step1_1_fetch_inspect.py` through `step4_2_mamba_benchmark.py`, `run_baselines_mdm_tssvm_tslda_csp.py`, the `run_step4_drift_control_*.py` family, `Diagrams/`, `Figures/`, and similar) belong to earlier phases of this project: an initially planned dual-branch architecture and a subsequently withdrawn drift and rest-break-discontinuity investigation, both superseded by the composition-error finding described above. They are retained for the audit trail recorded in `AUDIT.md`, `STATUS.md`, and `DECISIONS.md`, and are not part of the evidence chain behind the current manuscript.

## Requirements and how to run

The six Modal scripts listed above pin their dependencies in their own Modal image definitions, not in `requirements.txt` (that file describes an earlier, separate local environment and should not be used to reproduce the Modal pipeline). The pins actually used are Python 3.11, `numpy==1.26.4`, `scikit-learn==1.4.2`, `scipy==1.13.1`, and `threadpoolctl==3.6.0` (the data engine script pins `scipy==1.14.1` instead; `run_c3_r2_jackknife.py` needs only `numpy==1.26.4`, since it performs arithmetic over an already-computed result rather than training a classifier). All six read from and write to a Modal volume named `eeg-data-vol`.

Running any of these scripts requires a Modal account (`pip install modal`, then `modal setup` to authenticate) and is not a one-command local reproduction: `modal run run_c3_r2_jackknife.py::main`, for instance, provisions its own container, mounts the volume, and executes remotely. Re-running the pipeline is also not guaranteed to reproduce bit-identical numbers; see Reproducibility below.

## How to verify a number without re-running anything

Every figure reported in the manuscript is stored in a committed JSON file at the repository root or under `results/`, and `PAPER/PROVENANCE.md` indexes all of them: manuscript figure, source file, and exact field path. Three examples:

- The Abstract's `0.8657 ± 0.0005` (R1(b)'s post-calibration balanced accuracy) is `results_r1b_r2_r3_composition_runs_20260825T042232Z_a3293792.json`, field `R1b_encode_vs_test.post_calibration_balanced_accuracy_mean`.
- Figure 3's C4 null mean, `0.4995`, is `results_c4_high_res_shuffled_label_control.json`, field `null_distribution_mean`.
- Table S1's per-subject rows (for example sub-18's `0.5703 ± 0.0005` LOO mean) are `results_c3_r2_jackknife.json`, field `loo_results[].loo_mean`, matched to `loo_results[].excluded_subject`.

`PAPER/PROVENANCE.md` also records where two source JSONs disagree (the original R1(b)/R2/R3 run versus its rerun) and which one the manuscript cites, rather than picking silently.

## Reproducibility

Section 8 of the manuscript discloses this in full; the summary here is a small fraction of that length and should not be read as a substitute for it. The classification pipeline is not bit-exactly reproducible across runs on different compute hardware. The mechanism is unpinned BLAS and LAPACK thread-count scheduling combined with OpenBLAS's runtime `DYNAMIC_ARCH` dispatch, which can each shift a small number of individual fold-level classifications between otherwise-identical runs. The measured movement is about 4.1e-4 per affected fold for the quantities checked in this project. Every balanced accuracy, AUC, and derived threshold in the manuscript is therefore cited with an explicit tolerance rather than as a bare point estimate. This claim is scoped to the modelling layer only: the epoching layer that produces the pipeline's input has not been independently re-derived from raw data since its original run, so its own run-to-run numerical stability is unmeasured. The pipeline is not deterministic, and no claim in this repository or the manuscript states otherwise.

## How to rebuild the manuscript

`PAPER/draft.md` is the only manuscript source. `PAPER/figures/scripts/build_latex.py` reads it and writes `PAPER/draft_elsevier.tex`, `PAPER/draft_ieee.tex`, and `PAPER/references_ieee.bib` in full on every run. **Do not edit either `.tex` file directly: any direct edit is silently discarded the next time this script runs.** Manuscript-content changes belong in `draft.md`; changes to structure, formatting, or the Table S1 renderer belong in `build_latex.py`.

To rebuild:

```
python PAPER/figures/scripts/build_latex.py
```

To compile either build, from inside `PAPER/`:

```
pdflatex -interaction=nonstopmode draft_elsevier.tex
bibtex draft_elsevier
pdflatex -interaction=nonstopmode draft_elsevier.tex
pdflatex -interaction=nonstopmode draft_elsevier.tex
```

and the same four commands with `draft_ieee` in place of `draft_elsevier`. Both require `elsarticle`/`IEEEtran` class and style files from a standard TeX distribution; neither is vendored in this repository.

## Figures

Each of the four figure scripts under `PAPER/figures/scripts/` reads its values from a committed result JSON rather than containing a hardcoded number:

- `fig1_discovery_timeline.py` reads `results_r1b_r2_r3_composition_runs_20260825T042232Z_a3293792.json` for R1(b)'s and R2's headline figures.
- `fig2_c3_jackknife.py` reads `results_c3_r2_jackknife.json` and `results_c4_high_res_shuffled_label_control.json`.
- `fig3_null_distribution.py` and `fig4_d2_parity_split.py` read the real per-shuffle values stored in `results_c1_r2_shuffled_label_control.json`, `results_c4_high_res_shuffled_label_control.json`, and `results_d2_parity_split_check.json`, and plot the actual empirical null distributions rather than a modeled approximation.

Run any of the four with `python <script_name>.py` from `PAPER/figures/scripts/`; each writes its own `.pdf` and `.png` into `PAPER/figures/`.

## Licence and citation

No licence file for the code in this repository currently exists. This is an open item, not a decision to leave it unlicensed; a licence should be chosen and added before or at submission. The dataset itself (ds005189) is CC0.

A citation entry for the manuscript itself is not included here because the manuscript has no publication DOI, volume, or page numbers yet (it is still in preparation, not published). Cite the dataset it reanalyzes as: Helbing, J., Draschkow, D., and Vo, M. L.-H. (2025). Incidental encoding of objects during search is stronger than intentional memorization due to increased recollection rather than familiarity. *Journal of Cognitive Neuroscience*, 37(12), 2538 to 2557. `https://doi.org/10.1162/jocn.a.80`. The manuscript's authors are Muhammad Hassan Siddiqui (Independent Researcher, Lahore, Pakistan; ORCID 0009-0006-7271-9788; hassansiddiqui0946@gmail.com, corresponding author) and Muhammad Adil Usmani (Independent Researcher, Lahore, Pakistan; ORCID 0009-0004-2856-3419; muhammadaadilusmani@gmail.com).

## Open items

The following are not done. They are listed here, not marked as done, and not worked around:

- **IRB approval ID.** `PAPER/draft.md` Section 8 states approval ID 2014-106R1, sourced from the authors directly rather than independently verified against OpenNeuro's own published documentation for ds005189.
- **Word-limit check.** No explicit word limit for *NeuroImage: Reports* standard articles was confirmed against the live guide-for-authors page; automated fetches of that page returned HTTP 403 throughout this project's work.
- **`.docx` reconversion.** `PAPER/supplementary_table_s1.docx` has not been regenerated from the current `PAPER/supplementary_table_s1.md`; pandoc was not available in the environment this project's audit work ran in.
- **Commit hash for Section 8.** `PAPER/draft.md` states "Commit hash: TBD"; a specific commit is to be tagged at submission (the repository URL itself is filled in already).
- **CRediT authorship statement.** Added (`PAPER/draft.md` Section 8): Muhammad Hassan Siddiqui, Conceptualization/Methodology/Software/Formal analysis/Writing (original draft); Muhammad Adil Usmani, Validation/Writing (review and editing). Whether *NeuroImage: Reports* requires a specific format for this statement could not be verified against the live guide-for-authors (same access issue as the word-limit item above).
- **N14.** One item from this project's own manuscript-defect tracking (`DECISIONS.md`'s N8 to N19 entry) could not be resolved: "five over-corrections" of "this paper" or "this project" phrasing were flagged in an original review pass whose source files no longer exist anywhere in this repository or its history, so which five instances were meant could not be identified.
