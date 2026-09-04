"""
Build draft_ieee.tex and draft_elsevier.tex from PAPER/draft.md.
Programmatic conversion (not hand-transcription) so both LaTeX
variants share byte-identical body content, differing only in
preamble/frontmatter/figure-environment per journal class.
"""
import re
import os
import json

PAPER_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
DRAFT_MD = os.path.join(PAPER_DIR, "draft.md")

with open(DRAFT_MD, encoding="utf-8") as f:
    RAW = f.read()


# ----------------------------------------------------------------
# 0. F4 fix: IEEEtran.bst has zero support for a "doi" field
# (confirmed by grepping the installed .bst -- no "doi" string
# appears anywhere in it), unlike elsarticle-num.bst (11 matches),
# which is why Elsevier already renders DOIs natively and IEEE
# does not, regardless of \usepackage{doi}. Per the approved
# fallback, generate an IEEE-only bib variant with the DOI moved
# into a "note" field (which IEEEtran.bst DOES print, confirmed at
# .bst line ~1952's "format.note output" inside the article
# FUNCTION) -- references.bib itself stays untouched so Elsevier's
# native rendering is unaffected.
# ----------------------------------------------------------------
def build_ieee_bib():
    src = os.path.join(PAPER_DIR, "references.bib")
    dst = os.path.join(PAPER_DIR, "references_ieee.bib")
    with open(src, encoding="utf-8") as f:
        bib = f.read()

    def add_note(m):
        entry = m.group(0)
        doi_m = re.search(r"doi\s*=\s*\{([^}]*)\}", entry)
        if not doi_m:
            return entry
        doi = doi_m.group(1)
        note_field = f"  note    = {{\\\\href{{https://doi.org/{doi}}}{{doi:{doi}}}}},\n"
        # insert the note field right before the closing brace's
        # preceding field line (i.e., right after the opening line)
        return re.sub(r"(\n)", r"\1" + note_field, entry, count=1)

    entry_pattern = re.compile(r"@\w+\{[^,]+,.*?\n\}\n?", re.DOTALL)
    new_bib = entry_pattern.sub(add_note, bib)

    with open(dst, "w", encoding="utf-8") as f:
        f.write(new_bib)
    return dst

# ----------------------------------------------------------------
# 1. Section-reference label map (§N.M -> \ref{sec:xxx})
# ----------------------------------------------------------------
SEC_LABELS = [
    ("§3.1", "sec:headline"),
    ("§3.2", "sec:invalidation"),
    ("§3.3", "sec:r1a"),
    ("§3.4", "sec:r1b"),
    ("§3.5", "sec:r2"),
    ("§3a", "sec:validation"),
    ("§2.1", "sec:dataset"),
    ("§2.2", "sec:calibration"),
    ("§2.3", "sec:preregistration"),
    ("§1", "sec:intro"),
    ("§2", "sec:methods"),
    ("§3", "sec:results"),
    ("§4", "sec:protocol"),
    ("§5", "sec:limitations"),
    ("§6", "sec:discussion"),
    ("§7", "sec:conclusion"),
    ("§8", "sec:data"),
]
# sort longest-match-first so §3.1 is matched before bare §3, etc.
SEC_LABELS.sort(key=lambda kv: -len(kv[0]))


def convert_section_refs(text: str) -> str:
    # Handle en-dash ranges first: "§3.1–§3.2" -> "\S\ref{a}--\ref{b}"
    # (kept as a compact "§"-prefixed reference, matching the source
    # document's own compact §-notation rather than a bare number).
    def range_sub(m):
        a, b = m.group(1), m.group(2)
        la = dict(SEC_LABELS).get(a)
        lb = dict(SEC_LABELS).get(b)
        if la and lb:
            return f"\\S\\ref{{{la}}}--\\ref{{{lb}}}"
        return m.group(0)

    range_pattern = "|".join(re.escape(k) for k, _ in SEC_LABELS)
    text = re.sub(rf"({range_pattern})–({range_pattern})", range_sub, text)

    # Single references
    for tok, label in SEC_LABELS:
        text = text.replace(tok, f"\\S\\ref{{{label}}}")
    return text


# ----------------------------------------------------------------
# 2. Citation replacement (paragraph-specific, done on raw text
#    before generic escaping, since the target strings contain
#    characters -- &, parentheses, em dash -- that generic
#    escaping would otherwise mangle before we can match them)
# ----------------------------------------------------------------
NBSP = "\x01N0\x01"  # sentinel for a literal non-breaking-space "~" we insert
                     # ourselves, protected from escape_special()'s blanket
                     # "~" -> "$\sim$" substitution (which is for the source
                     # document's own approx-sign usage, not our markup)


def convert_citations(text: str) -> str:
    text = text.replace(
        "(ds005189: Helbing, Draschkow & Võ, 2025)",
        f"(ds005189{NBSP}\\cite{{helbing2025}})",
    )
    text = text.replace(
        "ds005189 (Helbing, Draschkow & Võ, 2025, *Journal of Cognitive Neuroscience*), a public",
        f"ds005189{NBSP}\\cite{{helbing2025}}, a public",
    )
    return text


# ----------------------------------------------------------------
# 3. Inline markdown -> LaTeX (bold, italic, code) via sentinel
#    tokens, so generic character escaping (step 4) does not touch
#    the LaTeX commands we insert here.
# ----------------------------------------------------------------
BOLD_OPEN, BOLD_CLOSE = "\x01B1\x01", "\x01B0\x01"
ITAL_OPEN, ITAL_CLOSE = "\x01I1\x01", "\x01I0\x01"
CODE_OPEN, CODE_CLOSE = "\x01C1\x01", "\x01C0\x01"


def mark_inline(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", lambda m: BOLD_OPEN + m.group(1) + BOLD_CLOSE, text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", lambda m: ITAL_OPEN + m.group(1) + ITAL_CLOSE, text)
    text = re.sub(r"`([^`]+?)`", lambda m: CODE_OPEN + m.group(1) + CODE_CLOSE, text)
    return text


def restore_inline(text: str) -> str:
    text = text.replace(BOLD_OPEN, "\\textbf{").replace(BOLD_CLOSE, "}")
    text = text.replace(ITAL_OPEN, "\\emph{").replace(ITAL_CLOSE, "}")
    text = text.replace(CODE_OPEN, "\\texttt{").replace(CODE_CLOSE, "}")
    text = text.replace(NBSP, "~")
    return text


# ----------------------------------------------------------------
# 4. Generic LaTeX special-character escaping (applied only to
#    plain text -- run AFTER inline markers/citations/section refs
#    have been pulled out to sentinels/raw LaTeX, so we never
#    escape characters inside \ref{}, \cite{}, \textbf{ etc.)
# ----------------------------------------------------------------
def escape_special(text: str) -> str:
    # order matters: backslash first (none expected, but safe),
    # then the rest.
    text = text.replace("&", r"\&")
    text = text.replace("%", r"\%")
    text = text.replace("#", r"\#")
    text = text.replace("_", r"\_")
    # typographic dashes / minus / times / approx
    text = text.replace("—", "---")
    text = text.replace("–", "--")
    text = text.replace("−", "$-$")
    text = text.replace("×", "$\\times$")
    text = text.replace("~", "$\\sim$")
    # GATE E STEP 1b: "<"/">" render as inverted exclamation/question
    # marks under IEEEtran's OT1 font encoding (elsarticle loads T1,
    # where they render correctly, which is why this defect was
    # invisible in the Elsevier build). Wrapping in math mode fixes
    # both encodings identically, matching the -/x/~/+- pattern above.
    text = text.replace("<", "$<$")
    text = text.replace(">", "$>$")
    # GATE T6 STEP 3b (G4): every reported figure now carries an
    # explicit tolerance (e.g. "0.5773 ± 0.0005") per NUMBERS.md's
    # precision policy -- this rule was missing entirely, so every
    # such figure would have passed through as a raw, unconverted
    # "±" character. $ and \ are never targets of any other rule in
    # this function, so this is safe regardless of where it sits in
    # the chain (confirmed GATE T6 STEP 2a/2b).
    text = text.replace("±", "$\\pm$")
    return text


def convert_paragraph(text: str) -> str:
    text = convert_citations(text)
    text = convert_section_refs(text)
    text = mark_inline(text)
    text = escape_special(text)
    text = restore_inline(text)
    return text.strip()


# arithmetic lines with unicode minus already handled by escape_special
# turning "0.6205 − 0.5043 = 0.1162" into "0.6205 $-$ 0.5043 = 0.1162",
# which is valid LaTeX (adjacent math snippets), left as-is.

# ----------------------------------------------------------------
# 5. Extract paragraphs by content anchor, not raw line number.
#
# GATE T6 STEP 1/3e fix: every index below used to be a literal
# 1-indexed line number into `draft.md`, resolved via `LINES[n-1]`.
# That desyncs the instant a paragraph is inserted or removed
# anywhere earlier in the file -- confirmed in GATE T6: GATE T5
# inserted two new paragraphs (the fold-level-bias disclosure and
# the reproducibility/cached-epoching disclosure), silently shifting
# every single index below line ~50 by +2 or +4 with no error at
# import time, and a hard crash (or, worse, a wrong-paragraph
# substitution) the moment the substitution loops ran.
#
# Fix: each entry below is a short, stable text anchor -- chosen
# from a part of its paragraph that a numeric correction would not
# touch -- searched for fresh in the CURRENT `draft.md` every time
# this script runs. The dict keys are the ORIGINAL line numbers from
# when this script was first written; they are kept only as stable
# logical IDs. Nothing else in this file changes: TEXT_FIXES,
# TEXT_FIXES_2, TEXT_FIXES_3, CITATION_INSERTS, and every `P[n]`
# reference in the BODY assembly below all continue to use these
# same integers as opaque keys, and now resolve correctly regardless
# of how many paragraphs get added or removed elsewhere in the
# document in the future.
# ----------------------------------------------------------------
LINES = RAW.split("\n")

PARAGRAPH_ANCHORS = {
    11: "A label-dictionary error in our own analysis pipeline",
    13: "Correcting the contrast to encode-phase trials only yields a pre-calibration balanced accuracy of",
    19: "Multi-phase experimental paradigms",
    21: "This paper reports two linked things, and neither subsumes the other.",
    23: "The Results section that follows is ordered as this investigation actually unfolded",
    29: "This study reanalyzes ds005189",
    31: "The analyses reported in this paper depend on a correction to how trials were assigned class labels",
    33: "This is a property of how our own analysis pipeline defined its class labels",
    35: "To determine how much of this project's own prior decoding results this composition error explained",
    37: "To move from a diagnostic fit to a confirmed account",
    39: "The corrected contrast used for all novel findings in this paper (R2) restricts to encode-phase trial onsets only",
    43: "Preprocessing follows the fixed pipeline established in this project's prior work",
    45: "Classification throughout this paper uses the same pipeline as our project's prior work",
    47: "We state plainly, as in our project's prior work, that this evaluation is",
    49: "All evaluations in this paper",
    53: "Every verdict threshold applied in §3 and §3a below was recorded, in writing",
    55: "**Disclosed implementation corrections.**",
    61: "This section is ordered as the investigation actually unfolded",
    65: "Under this project's original class-label definition",
    69: "The class label each of the two decoding targets was built from is a union of four structurally distinct marker types",
    73: "Before proposing any alternative account of what this project's prior results had actually been measuring",
    77: "We then re-epoched the dataset preserving all eight original marker codes as distinct classes and directly decoded",
    81: "Restricting to encode-phase trial onsets only",
    83: "For reference, R1(b)'s own encode-vs-test decode",
    85: "A third contrast (R3) retains the original 200-epoch-per-class construction",
    87: "is the corrected contrast's central claim carried forward through the rest of this paper",
    91: "We report the four robustness checks applied to R2's pre-calibration result",
    95: "We computed 30 within-subject label shuffles on the exact R2 dataset",
    99: "To test whether R2's result is carried by one or two influential subjects",
    103: "To sharpen C1's p-value estimate",
    107: "The final check addresses whether R2's signal could instead reflect block-order",
    109: "The first argument follows from R2's own pooled design",
    111: "The second argument is D2 itself",
    113: "We note directly here, as an observed fact rather than a resolved one",
    115: "distinguishable from a shuffled-label null at two resolutions",
    121: "The corrections described in",
    123: "**Compute real per-subject class composition from raw markers.**",
    124: "**Require a joint criterion before declaring a confound confirmed.**",
    125: "**Validate any corrected contrast against a null computed on the exact corrected dataset.**",
    126: "**Test the corrected contrast against its own structural counterbalancing.**",
    128: "Pair every step above with this project's standing reporting discipline",
    134: "Six limitations qualify the result reported in",
    136: "**Post-calibration validity is untested.**",
    138: "**Task-instruction vs. other encoding-phase differences is substantially",
    140: "**D2's parity-group effect-size asymmetry is unexplained.**",
    142: "**The effect size is modest in absolute terms.**",
    144: "**Single-seed basis.**",
    146: "**Pre-F3 alignment dependency.**",
    152: "This paper began as an investigation into a null result",
    154: "The diagnostic protocol described in",
    156: "The withdrawn drift/rest-break-discontinuity account",
    158: "Three things this paper does not claim, stated explicitly",
    164: "This paper reports three things in sequence",
    170: "**Dataset.**",
    172: "**Code.**",
    174: "**Pre-registration record.**",
    176: "**Ethics.**",
    182: "Figure 1.",
    184: "Figure 2.",
    186: "Figure 3.",
    188: "Figure 4.",
    # GATE T5 added two wholly new paragraphs to draft.md that have no
    # "original" line number (they didn't exist when this script was
    # first written). Given new, clearly-out-of-range logical IDs
    # (1000+) rather than a line number, so they're unmistakably not
    # confused with a real original-line anchor.
    1000: "At the pre-calibration, per-fold level, R2's classifier does not favour one class uniformly",
    1001: "The classification pipeline underlying every result in this paper is not bit-exactly reproducible",
    # GATE A1 STEP 5: three new backmatter declarations, same
    # out-of-range logical-ID convention as 1000/1001 above.
    1002: "This research received no specific grant from any funding agency",
    # GATE A1 (two-author revision): anchor 1003 updated from singular
    # "The author..." to plural, matching draft.md's rewording -- an
    # anchor string is a literal substring match against L(n), so it
    # must track the current wording exactly.
    1003: "The authors declare no competing financial interests or personal relationships",
    # 1004 (Generative AI disclosure) removed, GATE A2 STEP 2e -- the
    # paragraph itself was deleted from draft.md; this logical ID is
    # retired, not reused, so a stray reference elsewhere would fail
    # loudly (KeyError) rather than silently resolve to something else.
    1005: "Muhammad Hassan Siddiqui: Conceptualization, Methodology, Software",
}


def _resolve_line(orig_n: int) -> int:
    anchor = PARAGRAPH_ANCHORS[orig_n]
    matches = [i for i, line in enumerate(LINES, start=1) if anchor in line]
    if not matches:
        raise ValueError(
            f"GATE T6 anchor fix: anchor for original line {orig_n} not found "
            f"in current draft.md: {anchor!r}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"GATE T6 anchor fix: anchor for original line {orig_n} is ambiguous, "
            f"matched lines {matches}: {anchor!r}"
        )
    return matches[0]


LINE_MAP = {n: _resolve_line(n) for n in PARAGRAPH_ANCHORS}


def L(n: int) -> str:
    return LINES[LINE_MAP[n] - 1]


P = {n: convert_paragraph(L(n)) for n in [
    11, 13,                      # abstract
    19, 21, 23,                  # intro
    29, 31, 33, 35, 37, 39,       # 2.1
    43, 45, 47, 49,               # 2.2
    1000,                         # 2.2 fold-level-bias disclosure (GATE T5 addition)
    53, 55,                       # 2.3
    61,                           # results intro
    65,                           # 3.1
    69,                           # 3.2
    73,                           # 3.3
    77,                           # 3.4
    81, 83, 85, 87,               # 3.5
    91,                           # 3a intro
    95,                           # C1
    99,                           # C3
    103,                          # C4
    107, 109, 111, 113,           # D2
    115,                          # 3a closing
    121,                          # protocol intro
    123, 124, 125, 126,           # protocol items
    128,                          # protocol closing
    134,                          # limitations intro
    136, 138, 140, 142, 144, 146,  # limitations items
    152, 154, 156, 158,           # discussion
    164,                          # conclusion
    170, 172,                     # data/code
    1001,                         # data/code reproducibility disclosure (GATE T5 addition)
    174, 176,                     # data/code (cont.)
    1002, 1003, 1005,             # funding / COI / CRediT disclosures (GATE A1 addition; 1004 removed GATE A2)
    182, 184, 186, 188,           # figure captions
]}

# ----------------------------------------------------------------
# 5b. Citation injection -- targeted, anchored string replacements
# applied to specific paragraphs after conversion. Each entry is
# (line, anchor substring already present in the converted text,
# replacement with \cite{} appended). Applied once here so both
# draft_ieee.tex and draft_elsevier.tex share identical citations
# (built from the same P dict).
# ----------------------------------------------------------------
CITATION_INSERTS = [
    # Introduction
    (19, "are common in EEG and BCI research",
         "are common in EEG and BCI research \\cite{wolpaw2002,lotte2007,lotte2018}"),
    (19, "such as an encoding period followed by a recognition test",
         "such as an encoding period followed by a recognition test \\cite{pallerwagner2002,ruggcurran2007}"),
    (21, "label-dictionary confound in our own analysis pipeline",
         "label-dictionary confound \\cite{frenay2014} in our own analysis pipeline"),
    # Methods 2.1
    (29, "each block followed by its own recognition-test phase",
         "each block followed by its own recognition-test phase \\cite{polich2007}"),
    (29, "over a common set of scene images",
         "over a common set of scene images \\cite{draschkow2017,draschkowwolfe2014,vohenderson2009,woodmanluck2003}"),
    # Methods 2.2
    (43, "raw EEG is band-pass filtered from",
         "raw EEG is band-pass filtered \\cite{gramfort2013,delormemakeig2004,jas2017} from"),
    (43, "Euclidean Alignment (EA) is applied per LOSO fold",
         "Euclidean Alignment (EA) \\cite{he2020} is applied per LOSO fold"),
    (43, "not a parametrized per-subject or Riemannian-mean module",
         "not a parametrized per-subject or Riemannian-mean module \\cite{barachant2012,congedo2017,yger2017}"),
    (45, "projected to the tangent space at the Euclidean-aligned reference, standardized,",
         "projected to the tangent space \\cite{barachant2012,barachant2013} at the Euclidean-aligned reference"
         " (an alternative to spatial-filter-based approaches such as Common Spatial Patterns"
         " \\cite{koles1991,ramoser2000,blankertz2008,lotteguan2011}, which this pipeline does not use),"
         " standardized,"),
    (45, "reduced to 35 components by PCA",
         "reduced to 35 components by PCA \\cite{jolliffe2016}"),
    (45, "classified by a shrinkage-blended logistic regression",
         "classified by a shrinkage-blended logistic regression \\cite{ledoitwolf2004,friedman1989,blankertz2011}"),
    (49, "use LOSO cross-validation at a single fixed seed",
         "use leave-one-subject-out (LOSO) cross-subject evaluation \\cite{shenoy2006,vidaurre2011,samek2013,jayaram2016} at a single fixed seed"),
    # Methods 2.3
    (53, "was recorded, in writing, before the corresponding analysis was run",
         "was recorded, in writing, before the corresponding analysis was run \\cite{nosek2015,chambers2013}"),
    # Results 3.1 -- balanced accuracy first substantive use
    (65, "reported 70.78\\% balanced accuracy for Search-vs-Memorize decoding",
         "reported 70.78\\% balanced accuracy \\cite{brodersen2010} for Search-vs-Memorize decoding"),
    # Results 3.4 -- encode vs test-phase decode
    (77, "directly decoded encode-phase identity against test-phase identity",
         "directly decoded encode-phase identity against test-phase identity \\cite{pallerkutas1992,curran2000}"),
    # 3a / C1 -- shuffled-label null / permutation testing
    (95, "using the identical calibrated LOSO pipeline",
         "using the identical calibrated LOSO pipeline, per standard permutation-testing practice \\cite{nicholsholmes2002}"),
    # 3a / D2 -- parity-split / counterbalancing
    (107, "reflect block-order or session-position rather than task instruction specifically",
          "reflect block-order or session-position rather than task instruction specifically"
          ", the question a counterbalanced design is meant to guard against \\cite{greenwald1976}"),
    # Limitations item 1 -- few-shot subject-adaptive
    (136, "post-calibration accuracy is a few-shot subject-adaptive result",
          "post-calibration accuracy is a few-shot subject-adaptive result \\cite{he2020,jayaram2016}"),
    # Data availability -- OpenNeuro. Wrapped in \url{} (not \texttt{}) so
    # the DOI cannot be silently truncated by a mid-string line break --
    # this exact failure was observed in the Elsevier build ("...v1" with
    # ".0.1" dropped) and is fixed here for both targets at once.
    (170, "DOI \\texttt{10.18112/openneuro.ds005189.v1.0.1}",
          "DOI \\url{https://doi.org/10.18112/openneuro.ds005189.v1.0.1} \\cite{markiewicz2021,gorgolewski2016}"),
    # GATE A1 STEP 3a: repository URL, wrapped in \url{} (not \texttt{})
    # so hyperref makes it an actual clickable link, matching the DOI
    # fix above -- \texttt{} alone does not get auto-linked.
    (172, "are available at \\texttt{https://github.com/HassanSidd0946/Search-vs-Memorize-Correction}.",
          "are available at \\url{https://github.com/HassanSidd0946/Search-vs-Memorize-Correction}."),
]

for _line, _anchor, _replacement in CITATION_INSERTS:
    if _anchor not in P[_line]:
        raise ValueError(f"Citation anchor not found in P[{_line}]: {_anchor!r}")
    P[_line] = P[_line].replace(_anchor, _replacement, 1)

# ----------------------------------------------------------------
# 5c. Red-team-audit text fixes (W4, W5, W7), applied the same way
# as citation injection: targeted, anchored replacements on the
# already-converted P[] text, so both .tex outputs stay in sync.
# ----------------------------------------------------------------
TEXT_FIXES = [
    # --- W7: remove implied-external-citation phrasing ("this
    # project's prior work" / "superseded draft" / "prior
    # investigation") -- rephrase to describe the prior analysis
    # inline instead of pointing at an uncitable artifact.
    (29, "reproducing the same 29-subject cohort used throughout this project's prior work.",
         "reproducing the same 29-subject cohort used consistently across the analyses that preceded this correction."),
    (35, "how much of this project's own prior decoding results this composition error explained",
         "how much of the decoding results obtained before this correction this composition error explained"),
    (35, "for any pseudo- or real-class construction used in this project's history",
         "for any pseudo- or real-class construction used in the analyses preceding this correction"),
    (35, "Applied retrospectively to every contrast in this project's own prior \"temporal drift\"/rest-break-discontinuity investigation",
         "Applied retrospectively to every contrast in the \"temporal drift\"/rest-break-discontinuity investigation this paper withdraws"),
    (43, "Preprocessing follows the fixed pipeline established in this project's prior work, unchanged by",
         "Preprocessing follows the fixed pipeline established before this correction, unchanged by"),
    (43, "a property carried over unchanged from the superseded draft's own Methods description and unaffected by the R1 through D2 corrected-contrast investigation.",
         "a property unchanged from before this correction and unaffected by the R1 through D2 corrected-contrast investigation."),
    (45, "Classification throughout this paper uses the same pipeline as our project's prior work:",
         "Classification throughout this paper uses the same pipeline used before this correction:"),
    (47, "We state plainly, as in our project's prior work, that this evaluation is",
         "We state plainly, as before this correction, that this evaluation is"),
    (49, "matching the convention established in this project's prior work.",
         "matching the convention established before this correction."),
    (73, "Before proposing any alternative account of what this project's prior results had actually been measuring",
         "Before proposing any alternative account of what the pipeline's earlier results had actually been measuring"),
    (73, "of each pseudo- or real-class construction used in this project's history",
         "of each pseudo- or real-class construction used in the analyses preceding this correction"),
    (73, "Applied to every checked contrast in this project's own prior \"temporal drift\"/rest-break-discontinuity investigation",
         "Applied to every checked contrast in the \"temporal drift\"/rest-break-discontinuity investigation this paper withdraws"),
    (144, "matches the convention this project has used throughout its prior work;",
         "matches the convention used throughout the analyses preceding this correction;"),
    (146, "This pipeline choice is unchanged from this project's prior work and has not been re-tested",
          "This pipeline choice is unchanged from before this correction and has not been re-tested"),
    (154, "in this project's own prior drift investigation.",
          "in the withdrawn drift investigation described in this paper."),
    # --- W5: rename opaque internal-shorthand limitation title.
    (146, "Pre-F3 alignment dependency", "Alignment-scheme dependency"),
    # --- W4: remove self-conscious "restating this a third time"
    # meta-commentary from Limitations item 1; keep the direct
    # statement of the limitation itself (already present right
    # after the clause being cut).
    (136, "We stated this on first mention of the post-calibration number in \\S\\ref{sec:r2} "
          "and again in \\S\\ref{sec:validation}'s closing sentence, and we restate it a third "
          "time here deliberately, rather than letting those two mentions substitute for "
          "treating it as a limitation in its own right: the post-calibration figure inherits "
          "none of the validation established for the pre-calibration figure.",
          "The post-calibration figure inherits none of the validation established for the "
          "pre-calibration figure."),
]

for _line, _anchor, _replacement in TEXT_FIXES:
    if _anchor not in P[_line]:
        raise ValueError(f"Text-fix anchor not found in P[{_line}]: {_anchor!r}")
    P[_line] = P[_line].replace(_anchor, _replacement, 1)

# ----------------------------------------------------------------
# 5d. Second-pass red-team-audit fixes (N3, N6 -- N1 and N2 removed,
# GATE T6 STEP 3a: N1 corrupted a correct draft.md sentence, N2 is
# now redundant with a GATE T5 draft.md edit). Applied after
# TEXT_FIXES above, on the already-modified P[] text, so anchors for
# lines touched by the first pass (e.g. 136, 164's §3a ref) must
# match that already-modified state.
# ----------------------------------------------------------------
TEXT_FIXES_2 = [
    # --- N1: REMOVED, GATE T6 STEP 3a (G1). This substitution
    # corrupted a CORRECT draft.md sentence into a wrong one --
    # draft.md's own text at these two anchors (the 2,900/uniform-50
    # epoch-count description) is already accurate; the "verified
    # true N=2,910" comment that used to justify this substitution
    # relied on STATUS.md, a source GATE C15 established is itself
    # stale (predates the sub-01 practice-trial-exclusion fix). See
    # RESULTS_LEDGER.md's GATE T3 STEP 1 entry for the full history.
    #
    # --- N2: REMOVED, GATE T6 STEP 1c/3a (dry-run finding). This
    # substitution is now redundant: GATE T5 already applied this
    # exact rewrite directly to draft.md as part of a larger N9+N17
    # correction to the same sentence, so this anchor no longer
    # exists anywhere in draft.md and would raise ValueError on every
    # run if left in place.
    #
    # --- N3 item 1
    (49, "the plausibility-hardening checks established earlier in this project:",
         "a standard set of plausibility-hardening checks:"),
    # --- N3 item 3
    (81, "the first real-label, cross-subject, pre-calibration result in this project's entire history that is not at chance.",
         "the first Search-vs-Memorize, cross-subject, pre-calibration result reported in this paper that is not at chance."),
    # --- N3 item 4
    (128, "Pair every step above with this project's standing reporting discipline:",
          "Pair every step above with the reporting discipline followed throughout this paper:"),
    # --- N3 items 5+6 (same sentence in Limitations item 1, already
    # modified by the previous pass's TEXT_FIXES -- anchor reflects
    # that already-modified state, citation included).
    (136, "This project's own standing finding that post-calibration accuracy is a few-shot subject-adaptive result \\cite{he2020,jayaram2016}, not a subject-independent one (\\S\\ref{sec:calibration}), applies to R2's post-calibration number exactly as it applied to every post-calibration number this project has reported before it.",
          "The finding established in \\S\\ref{sec:calibration} that post-calibration accuracy is a few-shot subject-adaptive result \\cite{he2020,jayaram2016}, not a subject-independent one, applies to R2's post-calibration number exactly as it applied to every post-calibration number reported earlier in this paper."),
    # --- N6: three items listed for "four independent checks" --
    # add the missing C1/C4/C3/D2 labels so the count matches.
    (164, "a shuffled-label null at two resolutions, a leave-one-subject-out jackknife, and a parity-split replication (\\S\\ref{sec:validation})",
          "a shuffled-label null at two resolutions (C1, C4), a leave-one-subject-out jackknife (C3), and a parity-split replication (D2) (\\S\\ref{sec:validation})"),
    # --- extra instance of the same W7/N3 pattern, found during the
    # re-grep this pass (not one of N3's six, reported separately):
    # same "this project ... proposed" framing in the Conclusion.
    (164, "formally withdrew the drift/rest-break-discontinuity account this project had previously proposed for the same contrast.",
          "formally withdrew the drift/rest-break-discontinuity account previously proposed for the same contrast."),
]

for _line, _anchor, _replacement in TEXT_FIXES_2:
    if _anchor not in P[_line]:
        raise ValueError(f"Text-fix-2 anchor not found in P[{_line}]: {_anchor!r}")
    P[_line] = P[_line].replace(_anchor, _replacement, 1)

# ----------------------------------------------------------------
# 5e. N3's mandated re-grep, applied: every remaining "this project"
# instance found after TEXT_FIXES_2, each either fixed here or left
# with a reason (see the one exception below, P[172]).
# ----------------------------------------------------------------
TEXT_FIXES_3 = [
    (11, "an earlier drift/rest-break-discontinuity account this project had proposed for the same contrast.",
         "an earlier drift/rest-break-discontinuity account previously proposed for the same contrast."),
    (21, "absent from this project's original error-laden analysis",
         "absent from this paper's original error-laden analysis"),
    (23, "the reader first sees the number this project originally reported",
         "the reader first sees the number this paper originally reported"),
    (37, "the alternative drift/rest-break-discontinuity account this project had previously proposed (see \\S\\ref{sec:discussion})",
         "the alternative drift/rest-break-discontinuity account previously proposed (see \\S\\ref{sec:discussion})"),
    (61, "the reader first sees the number this project originally reported",
         "the reader first sees the number this paper originally reported"),
    (65, "Under this project's original class-label definition",
         "Under the original class-label definition"),
    (69, "Every number produced by this project before this point",
         "Every number produced by this paper before this point"),
    (77, "the drift/rest-break-discontinuity mechanistic account this project had previously proposed is formally withdrawn.",
         "the drift/rest-break-discontinuity mechanistic account previously proposed is formally withdrawn."),
    (152, "at a magnitude far more modest than this project's own original headline number (\\S\\ref{sec:headline}) suggested.",
          "at a magnitude far more modest than the original headline number (\\S\\ref{sec:headline}) suggested."),
    # NOTE: P[172] ("...available in this project's public code
    # repository") deliberately left unchanged -- it is §8's actual
    # description of a real, accessible artifact being pointed to
    # (the whole purpose of a Data/Code Availability statement), not
    # a vague reference to an uncitable prior analysis. Different
    # pattern from the rest of this sweep; not a citation gap.
]

for _line, _anchor, _replacement in TEXT_FIXES_3:
    if _anchor not in P[_line]:
        raise ValueError(f"Text-fix-3 anchor not found in P[{_line}]: {_anchor!r}")
    P[_line] = P[_line].replace(_anchor, _replacement, 1)

# --- W4: remove bold wrapping from the §3a closing paragraph (keep
# the substance; only §3.5's caveat stays bold, per instruction).
if P[115].startswith("\\textbf{") and P[115].endswith("}"):
    P[115] = P[115][len("\\textbf{"):-1]
else:
    raise ValueError(f"P[115] no longer starts/ends with \\textbf{{...}} as expected: {P[115][:60]!r}")

# strip the leading "N. **Bullet.**" markdown ordinal for list items
# (protocol + limitations) -- convert to \item form separately.
# IMPORTANT: strips the ordinal from the already-converted P[n] entry
# (not a fresh raw-text conversion) so that citation injections applied
# to P[n] above (see CITATION_INSERTS, e.g. line 136) are preserved.
LIST_ITEM_RE = re.compile(r"^\d+\.\s*")


def as_item(line_no: int) -> str:
    return LIST_ITEM_RE.sub("", P[line_no])


PROTOCOL_ITEMS = [as_item(n) for n in [123, 124, 125, 126]]
LIMITATIONS_ITEMS = [as_item(n) for n in [136, 138, 140, 142, 144, 146]]

print("Paragraphs converted:", len(P))
print("Protocol items:", len(PROTOCOL_ITEMS))
print("Limitations items:", len(LIMITATIONS_ITEMS))


def strip_fig_prefix(caption_text: str) -> str:
    return re.sub(r"^Figure \d+\.\s*", "", caption_text)


FIG_CAPTIONS = {
    1: strip_fig_prefix(P[182]),
    2: strip_fig_prefix(P[186]),  # jackknife content actually renders as compiled Figure 2
    3: strip_fig_prefix(P[184]),  # null-distribution content actually renders as compiled Figure 3
    4: strip_fig_prefix(P[188]),
}

# ----------------------------------------------------------------
# 6. Supplementary Table S1 rows.
#
# GATE T6 STEP 3c/3d (G2/G3, revised STEP 4 of the same gate): this
# used to be a 29-row literal hardcoded independently of any source
# file, with a comment claiming it came from `supplementary_table_s1.md`
# -- false; nothing in this script ever read that file. Both this
# table and its caption now read `results_c3_r2_jackknife.json`
# directly (the same canonical source `supplementary_table_s1.md` is
# separately regenerated from -- see that file's own header), so the
# embedded table and the standalone submission deliverable cannot
# independently drift again.
#
# Precision policy, GATE T7 STEP 1 (corrects GATE T6's uniform 0.0005,
# which failed containment on every one of the 29 per-fold-accuracy
# cells): the three numeric columns have three DIFFERENT quanta,
# derived from the committed per-fold class totals (43/42, constant
# across all 29 folds) and confirmed numerically by direct
# perturbation of the raw per-fold data (not just algebra):
#   Column 2 (per-fold accuracy) -- ONE fold, 85 trials (43+42):
#       a single-trial flip changes this fold's OWN balanced accuracy
#       by 0.5/42 (worst case) = ~0.0119.
#   Column 3 (LOO mean) -- a mean over the OTHER 28 folds: a flip in
#       any one of them changes the mean by (0.5/42)/28 = ~4.25e-4.
#   Column 4 (shift = (full_mean - per_fold)/28) -- NOT independent
#       of columns 2/3. Its dominant term comes from a flip in the
#       EXCLUDED subject's own fold (numerically confirmed): this
#       moves full_mean by (0.5/42)/29 = 1/(2x29x42) = 1/2436 --
#       exactly R2's own established quantum -- while leaving loo_mean
#       unchanged (the excluded fold isn't in that mean), so shift
#       moves by the same ~4.11e-4. A flip in a DIFFERENT fold moves
#       both loo_mean and full_mean by similar amounts in the same
#       direction, mostly cancelling (residual ~1.47e-5) -- the
#       excluded-subject's-own-fold term dominates and sets the
#       column's quantum.
# Required tolerance per column (quantum + half the last-place unit
# at 4dp, 0.00005): column 2 needs >=0.011955, rounded up to 0.012;
# columns 3 and 4 need >=0.00048/0.00046 respectively, both already
# covered by the existing 0.0005. All 87 cells plus both caption
# figures verified to pass three-point containment at these values
# (GATE T7 STEP 1c).
# ----------------------------------------------------------------
_C3_JSON_PATH = os.path.join(PAPER_DIR, "..", "results_c3_r2_jackknife.json")
with open(_C3_JSON_PATH, encoding="utf-8") as _f:
    _c3 = json.load(_f)

_TOL_PERFOLD = 0.012
_TOL_LOOMEAN = 0.0005
_TOL_SHIFT = 0.0005


def _fmt4(x: float, tol: float) -> str:
    return f"{x:.4f} $\\pm$ {tol}"


def _fmt4_signed(x: float, tol: float) -> str:
    # GATE T7 STEP 1e: a value that rounds to 0.0000 at 4dp but is
    # actually a small negative number (e.g. sub-15's shift,
    # -1.6e-5) must not render as "-0.0000" -- a published table
    # showing negative zero reads as a defect, not a rounding
    # artifact. Suppress the sign when the rounded magnitude is zero.
    rounded = round(x, 4)
    if rounded == 0:
        return f"0.0000 $\\pm$ {tol}"
    sign = "+" if x >= 0 else "-"
    return f"{sign}{abs(x):.4f} $\\pm$ {tol}"


S1_ROWS = [
    (
        f"sub-{r['excluded_subject']}",
        _fmt4(r["excluded_fold_pre_cal_balanced"], _TOL_PERFOLD),
        _fmt4(r["loo_mean"], _TOL_LOOMEAN),
        _fmt4_signed(r["shift_from_full_mean"], _TOL_SHIFT),
    )
    for r in _c3["loo_results"]
]
assert len(S1_ROWS) == 29

S1_TABLE_ROWS_TEX = "\n".join(
    f"{s} & {fold} & {loo} & {shift} \\\\" for s, fold, loo, shift in S1_ROWS
)

S1_CAPTION = (
    "Supplementary Table~S1. Leave-one-subject-out (LOO) jackknife analysis "
    "of R2 pre-calibration balanced accuracy (C3). "
    f"Full-sample mean $= {_c3['full_mean_recomputed']:.4f} \\pm 0.0005$. "
    "No single exclusion drops the LOO mean "
    f"below the pre-registered null CI upper bound "
    f"($ {_c3['null_95ci_upper_bound_used']:.4f} \\pm 0.0005$)."
)

# ----------------------------------------------------------------
# 7. Assemble the shared BODY (identical for both journal targets)
# ----------------------------------------------------------------
def enum(items, kind="enumerate"):
    body = "\n".join(f"\\item {it}" for it in items)
    return f"\\begin{{{kind}}}\n{body}\n\\end{{{kind}}}"


def sec1(title, label):
    return f"\\section{{{title}}}\\label{{{label}}}"


def sub1(title, label):
    return f"\\subsection{{{title}}}\\label{{{label}}}"


def subsub1(title):
    return f"\\subsubsection{{{title}}}"


BODY_INTRO = f"""{sec1('Introduction', 'sec:intro')}

{P[19]}

{P[21]}

{P[23]}"""

BODY_METHODS = f"""{sec1('Methods', 'sec:methods')}

{sub1('Dataset and the corrected epoch definition', 'sec:dataset')}

{P[29]}

{P[31]}

{P[33]}

{P[35]}

{P[37]}

{P[39]}

{sub1('Calibration mechanism', 'sec:calibration')}

{P[43]}

{P[45]}

{P[47]}

{P[49]}

{P[1000]}

{sub1('Pre-registration discipline', 'sec:preregistration')}

{P[53]}

{P[55]}"""

# ----------------------------------------------------------------
# Results body is built as an ordered list of blocks, where figure
# placeholders ("FIG:n") are substituted per-target at render time
# (IEEE uses figure* for figs 1/4, figure for figs 2/3; Elsevier
# uses figure for all four) -- see render_body() below.
# ----------------------------------------------------------------
RESULTS_BLOCKS = [
    sec1('Results', 'sec:results'),
    P[61],
    sub1('The original headline', 'sec:headline'),
    P[65],
    sub1('Invalidation', 'sec:invalidation'),
    P[69],
    "FIG:1",
    sub1('Composition diagnosis (R1(a))', 'sec:r1a'),
    P[73],
    sub1('Joint-criterion confirmation (R1(b))', 'sec:r1b'),
    P[77],
    sub1('The corrected contrast (R2)', 'sec:r2'),
    P[81],
    P[83],
    P[85],
    P[87],
    sub1('Validation controls', 'sec:validation'),
    P[91],
    subsub1('C1: Shuffled-label null on the exact R2 dataset'),
    P[95],
    subsub1('C3: Leave-one-subject-out jackknife'),
    P[99],
    "FIG:2",
    subsub1('C4: Higher-resolution shuffled-label null'),
    P[103],
    "FIG:3",
    subsub1('D2: Parity-split replication'),
    P[107],
    P[109],
    P[111],
    P[113],
    "FIG:4",
    P[115],
]


def render_results(fig_renderer) -> str:
    out = []
    for block in RESULTS_BLOCKS:
        if block.startswith("FIG:"):
            n = int(block.split(":")[1])
            out.append(fig_renderer(n))
        else:
            out.append(block)
    return "\n\n".join(out)

BODY_PROTOCOL = f"""{sec1('A Reusable Diagnostic Protocol', 'sec:protocol')}

{P[121]}

{enum(PROTOCOL_ITEMS)}

{P[128]}"""

BODY_LIMITATIONS = f"""{sec1('Limitations', 'sec:limitations')}

{P[134]}

{enum(LIMITATIONS_ITEMS)}"""

BODY_DISCUSSION = f"""{sec1('Discussion', 'sec:discussion')}

{P[152]}

{P[154]}

{P[156]}

{P[158]}"""

BODY_CONCLUSION = f"""{sec1('Conclusion', 'sec:conclusion')}

{P[164]}"""

BODY_DATA = f"""{sec1('Data and Code Availability', 'sec:data')}

{P[170]}

{P[172]}

{P[1001]}

{P[174]}

{P[176]}

{P[1002]}

{P[1003]}

{P[1005]}"""

ABSTRACT_TEXT = f"{P[11]}\n\n{P[13]}"

print("Body assembly complete.")
print(f"Abstract length (words, rough): {len(ABSTRACT_TEXT.split())}")

TITLE = ("A Mislabeled Contrast, Recovered: Diagnosing and Correcting an "
         "Encode/Test-Phase Confound in Blocked EEG Decoding")

KEYWORDS_IEEE = ("EEG decoding, label dictionary, composition artifact, "
                  "blocked paradigm, Riemannian geometry, pre-registration")

KEYWORDS_ELSEVIER = ("EEG decoding \\sep label dictionary \\sep composition "
                      "artifact \\sep blocked paradigm \\sep Riemannian "
                      "geometry \\sep pre-registration")


def full_body(fig_renderer) -> str:
    return "\n\n".join([
        BODY_INTRO,
        BODY_METHODS,
        render_results(fig_renderer),
        BODY_PROTOCOL,
        BODY_LIMITATIONS,
        BODY_DISCUSSION,
        BODY_CONCLUSION,
        BODY_DATA,
    ])


# ----------------------------------------------------------------
# IEEE figure/table renderers
# ----------------------------------------------------------------
FIG_FILES = {
    1: "fig1_discovery_timeline",
    2: "fig2_c3_jackknife",
    3: "fig3_null_distribution",
    4: "fig4_d2_parity_split",
}
FIG_WIDE = {1, 4}  # figure* in IEEE two-column


def ieee_fig(n: int) -> str:
    env = "figure*" if n in FIG_WIDE else "figure"
    width = r"\textwidth" if n in FIG_WIDE else r"\linewidth"
    return (f"\\begin{{{env}}}[!t]\n"
            f"  \\centering\n"
            f"  \\includegraphics[width={width}]{{figures/{FIG_FILES[n]}.pdf}}\n"
            f"  \\caption{{{FIG_CAPTIONS[n]}}}\n"
            f"  \\label{{fig:{n}}}\n"
            f"\\end{{{env}}}")


def elsevier_fig(n: int) -> str:
    return (f"\\begin{{figure}}[htbp]\n"
            f"  \\centering\n"
            f"  \\includegraphics[width=\\linewidth]{{figures/{FIG_FILES[n]}.pdf}}\n"
            f"  \\caption{{{FIG_CAPTIONS[n]}}}\n"
            f"  \\label{{fig:{n}}}\n"
            f"\\end{{figure}}")


# W1: \clearpage + [!p] ("own page" float placement) forces the table
# onto a clean standalone final page rather than drifting into the
# reference list. Combined with W1's other half -- moving the table's
# *source position* to after \bibliography{} in both documents below --
# so the float has nowhere earlier to drift into.
IEEE_TABLE = f"""\\clearpage
\\begin{{table*}}[!p]
\\caption{{{S1_CAPTION}}}
\\label{{tab:s1}}
\\centering
\\begin{{tabular}}{{lccc}}
\\toprule
Subject excluded & Per-fold accuracy & LOO mean & Shift from full-sample mean \\\\
\\midrule
{S1_TABLE_ROWS_TEX}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}"""

# W2(b): one descriptive sentence under the heading, since a reader who
# reaches "Supplementary Material" before the table renders (a float can
# still drift a little even with [!p]) should know what to expect.
ELSEVIER_TABLE = f"""\\clearpage
\\section*{{Supplementary Material}}
Supplementary Table~S1, referenced throughout \\S\\ref{{sec:validation}}, appears at the end of this document.

\\begin{{table}}[!p]
\\caption{{{S1_CAPTION}}}
\\label{{tab:s1}}
\\centering
\\begin{{tabular}}{{lccc}}
\\toprule
Subject excluded & Per-fold accuracy & LOO mean & Shift from full-sample mean \\\\
\\midrule
{S1_TABLE_ROWS_TEX}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

# ----------------------------------------------------------------
# IEEEtran document
# ----------------------------------------------------------------
IEEE_TEX = f"""\\documentclass[journal]{{IEEEtran}}
\\usepackage{{cite}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{hyperref}}
\\usepackage{{microtype}}
% NOTE (F4): \\usepackage{{doi}} was tried first per instruction but does
% nothing here -- IEEEtran.bst has no support for a doi field at all (0
% matches grepping the installed .bst), so the package has nothing to hook
% into. Fixed instead via references_ieee.bib (auto-generated from
% references.bib, DOI moved into a "note" field IEEEtran.bst does print),
% used only by \\bibliography{{}} below -- references.bib itself, and
% Elsevier's native doi-field rendering, are untouched.

\\title{{{TITLE}}}

\\author{{Muhammad Hassan Siddiqui and Muhammad Adil Usmani%
  \\thanks{{M. H. Siddiqui and M. A. Usmani are Independent Researchers,
  Lahore, Pakistan (e-mail: hassansiddiqui0946@gmail.com;
  muhammadaadilusmani@gmail.com; ORCID: 0009-0006-7271-9788 and
  0009-0004-2856-3419, respectively). Corresponding author:
  M. H. Siddiqui.}}}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
{ABSTRACT_TEXT}
\\end{{abstract}}

\\begin{{IEEEkeywords}}
{KEYWORDS_IEEE}
\\end{{IEEEkeywords}}

{full_body(ieee_fig)}

\\bibliographystyle{{IEEEtran}}
\\bibliography{{references_ieee}}

{IEEE_TABLE}

\\end{{document}}
"""

# ----------------------------------------------------------------
# elsarticle document
# ----------------------------------------------------------------
ELSEVIER_TEX = f"""\\documentclass[preprint,12pt]{{elsarticle}}
\\usepackage{{lineno}}
\\usepackage{{graphicx}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{hyperref}}
\\usepackage{{booktabs}}
\\usepackage[expansion=false]{{microtype}}
\\modulolinenumbers[5]

\\begin{{document}}

\\begin{{frontmatter}}

\\title{{{TITLE}}}

\\author[inst1]{{Muhammad Hassan Siddiqui\\corref{{cor1}}\\fnref{{orcid1}}}}
\\ead{{hassansiddiqui0946@gmail.com}}
\\author[inst1]{{Muhammad Adil Usmani\\fnref{{orcid2}}}}
\\cortext[cor1]{{Corresponding author.}}
\\fntext[orcid1]{{ORCID (M.H.S.): 0009-0006-7271-9788}}
\\fntext[orcid2]{{ORCID (M.A.U.): 0009-0004-2856-3419}}
\\affiliation[inst1]{{organization={{Independent Researcher}},
                    city={{Lahore}},
                    country={{Pakistan}}}}

\\begin{{abstract}}
{ABSTRACT_TEXT}
\\end{{abstract}}

\\begin{{keyword}}
{KEYWORDS_ELSEVIER}
\\end{{keyword}}

\\end{{frontmatter}}
\\linenumbers

{full_body(elsevier_fig)}

\\bibliographystyle{{elsarticle-num}}
\\bibliography{{references}}

{ELSEVIER_TABLE}

\\end{{document}}
"""

# NOTE: references.bib is maintained directly (not generated by this
# script) -- it holds 41 individually web/Crossref-verified entries.
# This script must never overwrite it.

if __name__ == "__main__":
    ieee_path = os.path.join(PAPER_DIR, "draft_ieee.tex")
    elsevier_path = os.path.join(PAPER_DIR, "draft_elsevier.tex")

    # F2-NIT: the dataset DOI \url{} breaks awkwardly right after
    # "https://doi." in Elsevier's narrower single-column body -- an
    # \mbox{} forcing it onto one unbreakable line was tried first but
    # overflows the column by 236pt (the string is wider than the
    # column itself), so a hard no-break is not viable. sloppypar
    # loosens interword spacing for just this paragraph, which is
    # enough to let TeX find a better break point without overfull
    # boxes. Scoped to Elsevier only (IEEE's two-column measure is
    # narrower still and was not flagged as having this issue).
    elsevier_tex_final = ELSEVIER_TEX.replace(
        "\\textbf{Dataset.} ds005189 (OpenNeuro accession ds005189), DOI \\url{https://doi.org/10.18112/openneuro.ds005189.v1.0.1} \\cite{markiewicz2021,gorgolewski2016}, license CC0.",
        "\\begin{sloppypar}\\textbf{Dataset.} ds005189 (OpenNeuro accession ds005189), DOI \\url{https://doi.org/10.18112/openneuro.ds005189.v1.0.1} \\cite{markiewicz2021,gorgolewski2016}, license CC0.\\end{sloppypar}",
    )

    with open(ieee_path, "w", encoding="utf-8") as f:
        f.write(IEEE_TEX)
    with open(elsevier_path, "w", encoding="utf-8") as f:
        f.write(elsevier_tex_final)
    ieee_bib_path = build_ieee_bib()

    print(f"Wrote {ieee_path}")
    print(f"Wrote {elsevier_path}")
    print(f"Wrote {ieee_bib_path} (auto-generated from references.bib, F4 fix)")
    print("references.bib left untouched (maintained separately)")

