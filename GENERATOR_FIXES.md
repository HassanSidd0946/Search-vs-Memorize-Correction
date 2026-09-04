# GENERATOR_FIXES.md — Required Changes to `PAPER/figures/scripts/build_latex.py`

Built GATE T4 STEP 1. `build_latex.py` reads only `PAPER/draft.md` and writes `draft_elsevier.tex`/`draft_ieee.tex` in full on every run (RESULTS_LEDGER.md, GATE T3 STEP 1). Every substitution table and hardcoded literal in the script is enumerated below. **Not applied in this gate** — file reads and analysis only.

---

## STEP 1a/1b — every substitution block, classified

| Block | Lines | Matches / substitutes | Applies to | Category |
|---|---|---|---|---|
| `build_ieee_bib()` | 29–51 | Moves a `doi` bib field into a `note` field | IEEE only (`references_ieee.bib`) | (i) formatting (bibliography, not manuscript prose) |
| `convert_section_refs()` / `SEC_LABELS` | 56–97 | `§N.M` tokens → `\S\ref{sec:xxx}` | Both (shared `P[]`) | (iv) section-reference handling |
| `convert_citations()` | 112–121 | Two exact dataset-citation phrasings → `\cite{helbing2025}` form | Both | (i)/(iv) — citation apparatus, not a number or claim |
| `mark_inline()`/`restore_inline()` | 129–146 | `**bold**`/`*italic*`/`` `code` `` → `\textbf{}`/`\emph{}`/`\texttt{}` | Both | (i) formatting only |
| `escape_special()` | 155–168 | `&,%,#,_,—,–,−,×,~` → LaTeX-safe equivalents | Both | (i) formatting only |
| `CITATION_INSERTS` | 232–288 | 15 anchored insertions of `\cite{...}` after specific substrings | Both (shared `P[]`) | (i)/(iv) — adds citation markers, no numbers/claims altered |
| `TEXT_FIXES` (W4, W5, W7) | 300–348 | Removes "this project's prior work" self-reference phrasing; renames one Limitations item title; trims repetitive meta-commentary | Both | (i) formatting/prose only — no number or claim altered |
| `TEXT_FIXES_2` (N1, N2, N3-partial, N6) | 361–406 | See below — **N1 is the one number/claim injection**; N2, N3-partial, N6 are prose de-referencing only | Both | **N1: (ii)+(iii)**; everything else in this block: (i) |
| `TEXT_FIXES_3` (N3 re-grep) | 413–438 | 9 more "this project"→"this paper" de-referencing replacements | Both | (i) formatting/prose only |
| `\textbf{}` strip on P[115] | 447–450 | Removes bold wrapping from one paragraph | Both | (i) formatting only |
| `LIST_ITEM_RE`/`as_item()` | 457–461 | Strips Markdown `"1. "` ordinals from list items | Both | (i) formatting only |
| **`S1_ROWS`** | 487–517 | **29 hardcoded tuples**, entirely independent of `draft.md` | Both (shared table renderer) | **(ii) — see below, major finding** |
| **`S1_CAPTION`** | 524–529 | Hardcoded `"Full-sample mean $= 0.5737$"` / `"($0.5223$)"` | Both | **(ii) — same finding** |
| Elsevier `sloppypar` wrap | 889–891 | Wraps one paragraph in `\begin{sloppypar}` for line-breaking | Elsevier only | (i) formatting only |

## STEP 1b — the type (ii)/(iii) items, exact before/after

### N1 (TEXT_FIXES_2, lines 368–371) — the confirmed regression

```python
(39, "This yields 50 epochs per task per subject, 2,900 epochs total, exactly balanced by the construction of the contrast itself rather than by post hoc subsampling.",
     "This yields 50 epochs per task for 28 of the 29 subjects and 55 for the remaining subject (sub-01, disclosed in §2.3), 2,910 epochs total --- exactly balanced by the construction of the contrast itself (1,455 per class) rather than by post hoc subsampling."),
(31, "this composition is fixed by the paradigm's own design at 50 encode onsets, 50 test-target responses, 25 test-distractor responses, and 75 test-new-lure responses",
     "this composition is fixed by the paradigm's own design at 50 encode onsets for 28 of the 29 subjects (sub-01 contributes 55, disclosed in §2.3), 50 test-target responses, 25 test-distractor responses, and 75 test-new-lure responses"),
```
**draft.md's current text (the "before" string above, matched verbatim by the anchor) is correct** — uniform 50/2,900, no sub-01 exception. **The generator's "after" string is what's wrong** — it injects "2,910," "1,455 per class," and the false "sub-01 contributes 55" framing into an otherwise-correct sentence. This is case **1c-(a)**: draft.md is correct, the generator corrupts it. This is the single highest-priority fix in this file.

### S1_ROWS / S1_CAPTION (lines 487–529) — an independent hardcoded data block, not a "fix"

Not a substitution applied to `draft.md` text at all — the entire 29-row, 4-column table (116 total cells: column 1 is the subject-ID string, e.g. `"sub-01"`; columns 2–4 are the three numeric fields — per-fold excluded-subject accuracy, LOO mean, and shift-from-full-sample-mean — giving 87 numeric cells, matching GATE T2's independent count) and its caption's two figures (`0.5737`, `0.5223`) are Python literals with no runtime connection to any source file. The code comment claims *"from `supplementary_table_s1.md`, values verified against L053"* — **this is misleading documentation**: `build_latex.py` never opens or reads `supplementary_table_s1.md` at runtime (confirmed: the only `open()` call in the entire script targets `draft.md` and `references.bib`). `S1_ROWS` was, at some point, manually copied from `supplementary_table_s1.md` and has not been kept in sync since — both are independently stale, coincidentally matching each other. This is case **1c-(b)**: the surrounding source (`supplementary_table_s1.md`, tracked separately) needs its own fix (queued in REPLACEMENTS.md), **and** the generator needs to actually read it rather than hardcode a copy.

## STEP 1c — correct vs. corrupted, item by item

| Item | draft.md's own text | Generator's transformation | Verdict |
|---|---|---|---|
| N1 (both halves) | Correct (uniform 50, 2,900, no sub-01 exception) | Corrupts it (injects 2,910/1,455/sub-01) | **Generator fix required** — remove or invert this substitution |
| S1_ROWS / S1_CAPTION | N/A — not in draft.md at all; lives in `supplementary_table_s1.md`, which is stale | Hardcodes a second, independently-stale copy | **Both** need fixing: update `supplementary_table_s1.md`'s data (a source-file fix, in REPLACEMENTS.md) **and** make the generator read it live (a code fix, here) |
| Every other TEXT_FIXES/TEXT_FIXES_2/TEXT_FIXES_3 entry | N/A — these are prose-only de-referencing edits with no numeric content | Passes draft.md's own (already-being-corrected) prose through unchanged in substance | No generator fix needed |

## STEP 1d — the N13 section-reference divergence, precisely located

`convert_section_refs()` itself is not the defect — it correctly converts every `§N.M` token appearing in raw `draft.md` text into `\S\ref{sec:xxx}`, and it runs identically for both targets (shared `P[]`). **The defect is that N1's replacement string (quoted above) contains a literal `"§2.3"` substring embedded directly inside a Python string in `TEXT_FIXES_2`, applied via `.replace()` *after* `convert_paragraph()` (which includes `convert_section_refs()`) has already run on `P[39]`/`P[31]`.** This literal `"§2.3"` is never passed back through `convert_section_refs()` — it survives into both outputs as a raw, non-resolving string, not a `\S\ref{}` cross-reference. It happens to display as the numerically correct section (Methods §2.3 really is "Pre-registration discipline" in both the Elsevier and IEEE builds, since both share the identical `BODY_METHODS` block and LaTeX auto-numbers subsections identically for both classes here) — but this has not been verified by compilation, and the string's fragility (not its current display) is the actual defect: any future reordering of Methods' subsections would silently desynchronize it in both builds, and the "N13" framing (IEEE-specific) undersells that the same fragile literal exists in the Elsevier build's generated output too, from the exact same source line.

## STEP 1e — hardcoded literals independent of the TEXT_FIXES blocks

| Literal | Line(s) | Value | Notes |
|---|---|---|---|
| `S1_ROWS` | 488–516 | 29 rows × 4 columns (116 cells total: 1 subject-ID column + 3 numeric columns = 87 numeric cells) | Independent of `draft.md`; see above |
| `S1_CAPTION` | 527–528 | `0.5737`, `0.5223` | Independent of `draft.md`; see above |
| `TITLE`, `KEYWORDS_IEEE`, `KEYWORDS_ELSEVIER` | 687–696 | Title and keyword strings | Not numeric/result content; not a defect |
| DOI string (`elsevier_tex_final.replace(...)`) | 890–891 | `10.18112/openneuro.ds005189.v1.0.1` | Matches `draft.md`'s own citation text exactly; not independently hardcoded data, just a formatting patch |

No other independent numeric literals found.

## GATE T5 STEP 4b — what `supplementary_table_s1.md` actually is

**Not merely an intermediate `build_latex.py` was meant to read** — it is a **real, independent submission deliverable**. Confirmed via `AUDIT.md:716-718` and `PAPER/SUBMISSION_CHECKLIST.md:24`: it was created 2026-08-23, converted to `supplementary_table_s1.docx` via pandoc, and is part of the actual "submission package" (`draft.md`, `draft.docx`, `supplementary_table_s1.{md,docx}`, `cover_letter.md`, `SUBMISSION_CHECKLIST.md`) submitted to the journal **separately from the LaTeX PDF** — `draft.md` references it by name at 3 locations (the §3a/C3 paragraph, Figure 1's caption, Figure 3's caption) rather than embedding the table itself. The checklist currently marks it `[x]` prepared/complete — **that checkbox is now false relative to current data**, since the table's content is exactly as stale as everything else in this audit. **Recommendation**: neither "keep in sync by hand" nor "retire" — regenerate both `supplementary_table_s1.md` and its `.docx` conversion directly from `results_c3_r2_jackknife.json` (the same source G2/G3 now target), so `build_latex.py`'s embedded copy and the standalone submission file are both derived from one canonical source and cannot independently drift again. Not regenerated in this gate (explicitly excluded from this gate's permitted edits).

## STEP 1f — required changes, GENERATOR_FIXES.md deliverable table

| # | Line(s) | Current behaviour | Required behaviour | Blocks the draft.md pass? |
|---|---|---|---|---|
| G1 | 368–371 | `TEXT_FIXES_2`'s N1 entries corrupt a correct draft.md sentence into "2,910/1,455 per class/sub-01 contributes 55" | Remove both N1 entries entirely (draft.md's own text is already correct and needs no substitution here) | **Yes — blocking.** Any draft.md fix to this sentence is pointless while N1 silently reverts it on next build. |
| G2 (revised, GATE T5 STEP 4) | 487–517 | `S1_ROWS` hardcodes 29 stale rows, never reads any file | **Read `results_c3_r2_jackknife.json`'s `loo_results` array directly** (not `supplementary_table_s1.md` — parsing Markdown would add a parser and leave a second copy of the same data free to drift; reading the JSON directly makes this script and `fig2_c3_jackknife.py` consume the identical source, which is what GATE D is doing anyway). Field path: `loo_results[i]` → `{"excluded_subject": "01", "excluded_fold_pre_cal_balanced": ..., "loo_mean": ..., "shift_from_full_mean": ...}`. Column mapping: `excluded_subject` (prefixed `"sub-"`) → col 1; `excluded_fold_pre_cal_balanced` → col 2 (Per-fold accuracy); `loo_mean` → col 3; `shift_from_full_mean` → col 4. | **Yes — blocking** for Table S1 specifically; does not block the rest of the draft.md numeric pass. |
| G3 (revised) | 524–529 | `S1_CAPTION` hardcodes `0.5737`/`0.5223` | Same JSON supplies both: `full_mean_recomputed` (→ `0.5773 ± 0.0005`) and `null_95ci_upper_bound_used` (→ `0.5172 ± 0.0005`) are both top-level fields of the identical `results_c3_r2_jackknife.json` — no second source needed. | Same as G2. |
| G4 | 155–168 (`escape_special`) | No rule for `±` (the tolerance symbol every corrected figure now needs) | Add `text = text.replace("±", "$\\pm$")`, matching the existing pattern for `−`/`×` | **Yes — blocking for every single numeric replacement**, since draft.md's corrected prose will use `±` throughout and needs it converted to LaTeX math mode, not left as a raw Unicode character of unverified rendering behavior. |
| G5 | (new, proposed) | No protection against overwriting a hand-edited `.tex` file | See STEP 4c below — an mtime guard | No — a safety net, not a blocker. |
| G6 | comment at 484–486 | Claims `S1_ROWS` is "from `supplementary_table_s1.md`" when it is not read from it | Correct the comment once G2 is implemented | No — documentation only, but should accompany G2. |

**Commit**: this file, alone, as a new tracked markdown file (no `.tex`/`.py`/`.md` source edits).
