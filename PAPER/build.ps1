<#
.SYNOPSIS
    Rebuilds draft_elsevier.pdf and draft_ieee.pdf from PAPER/draft.md, end to end.

.DESCRIPTION
    1. Runs build_latex.py to regenerate draft_elsevier.tex / draft_ieee.tex
       from draft.md (the only manuscript source).
    2. Compiles both under -jobname=build_elsevier / build_ieee (pdflatex,
       bibtex, pdflatex, pdflatex), which avoids ever writing directly to
       draft_elsevier.pdf / draft_ieee.pdf while a viewer might have them
       open.
    3. Swaps the results onto draft_elsevier.pdf and draft_ieee.pdf. Output
       filenames are always exactly these two names -- no date, hash, or
       counter -- so re-running this script overwrites rather than
       accumulating copies.
    4. Deletes every intermediate/auxiliary file it created.
    5. Prints the page count and overfull-hbox count for each build.

    If a target PDF is open elsewhere (WPS Office, a PDF viewer, etc.), the
    swap step fails cleanly: this script prints which file is locked and
    exits without deleting the last good PDF or leaving a half-copied file
    in its place. The previous draft_elsevier.pdf / draft_ieee.pdf are never
    touched until the newly compiled replacement exists and is verified.

.NOTES
    Run from anywhere; the script locates PAPER/ from its own path.
    Requires: python (on PATH), pdflatex, bibtex (a standard TeX
    distribution providing elsarticle and IEEEtran).
#>

$ErrorActionPreference = 'Stop'

$PaperDir = $PSScriptRoot
Set-Location $PaperDir

$overallFailed = $false

function Get-RequiredCommand {
    param([string]$Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        Write-Host "ERROR: '$Name' is not on PATH. Cannot continue." -ForegroundColor Red
        exit 1
    }
}

Get-RequiredCommand 'python'
Get-RequiredCommand 'pdflatex'
Get-RequiredCommand 'bibtex'

# ----------------------------------------------------------------
# STEP 1: regenerate both .tex files from draft.md
# ----------------------------------------------------------------
Write-Host "=== Regenerating draft_elsevier.tex / draft_ieee.tex from draft.md ===" -ForegroundColor Cyan
$genScript = Join-Path $PaperDir 'figures\scripts\build_latex.py'
python $genScript
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: build_latex.py failed (exit $LASTEXITCODE). Aborting; draft_elsevier.pdf / draft_ieee.pdf left untouched." -ForegroundColor Red
    exit 1
}

# ----------------------------------------------------------------
# Per-build compile + swap
# ----------------------------------------------------------------
function Build-Target {
    param(
        [string]$Target   # 'elsevier' or 'ieee'
    )

    $jobName   = "build_$Target"
    $texFile   = "draft_$Target.tex"
    $finalPdf  = "draft_$Target.pdf"
    $builtPdf  = "$jobName.pdf"
    $builtLog  = "$jobName.log"
    $consoleCapture = "$jobName.console.txt"

    # Pre-build every flag+variable argument into its own string variable
    # before passing it to a native executable via '&'. Windows PowerShell
    # 5.1 has been observed to pass an inline "-flag=$var" token to a
    # native exe LITERALLY (unexpanded, dollar sign and all) in some call
    # contexts, even though the same interpolation works fine in
    # Write-Host or a plain string assignment. Assigning the fully-
    # interpolated string to a variable first and passing that variable
    # sidesteps it reliably; passing "-jobname=$jobName" directly does not.
    $jobArg = "-jobname=$jobName"

    Write-Host ""
    Write-Host "=== Compiling $Target ($texFile -> $jobName) ===" -ForegroundColor Cyan

    # Clean any stale intermediates from a previous failed run before starting,
    # so a leftover build_elsevier.pdf from an earlier crash can't be mistaken
    # for this run's output.
    Remove-Item -Force -ErrorAction SilentlyContinue "$jobName.*"

    # pdflatex always writes its own transcript to "<jobname>.log" the
    # moment it can open that file, independent of shell redirection.
    # Redirecting *> to that SAME filename races pdflatex for the file
    # handle and makes it fail with "I can't write on file ...log" (a
    # real failure mode hit while developing this script, not a
    # hypothetical one) -- so console output from every pass here goes
    # to a distinctly-named capture file instead, and the actual stats
    # below are parsed from pdflatex's own "$builtLog", never touched by
    # this script's own redirection.
    $passOk = $true
    try {
        & pdflatex -interaction=nonstopmode -halt-on-error $jobArg $texFile *> $consoleCapture
        & bibtex $jobName *>> $consoleCapture
        & pdflatex -interaction=nonstopmode -halt-on-error $jobArg $texFile *>> $consoleCapture
        & pdflatex -interaction=nonstopmode -halt-on-error $jobArg $texFile *>> $consoleCapture
    } catch {
        $passOk = $false
    }

    if (-not (Test-Path $builtPdf)) {
        Write-Host "ERROR: $Target compile did not produce $builtPdf. $finalPdf left untouched (last good version still in place)." -ForegroundColor Red
        Write-Host "See $builtLog (if present) or $consoleCapture for details." -ForegroundColor Red
        Remove-Item -Force -ErrorAction SilentlyContinue "$jobName.aux","$jobName.bbl","$jobName.blg","$jobName.out","$jobName.spl","$consoleCapture"
        return $false
    }

    # Page count and overfull-hbox count, parsed from pdflatex's own log
    # (never redirected into by this script -- see note above).
    $logText = Get-Content $builtLog -Raw
    $pageMatch = [regex]::Match($logText, 'Output written on .*\((\d+) page')
    $pages = if ($pageMatch.Success) { $pageMatch.Groups[1].Value } else { 'unknown' }
    $overfullCount = ([regex]::Matches($logText, 'Overfull \\hbox')).Count
    $underfullCount = ([regex]::Matches($logText, 'Underfull \\hbox')).Count
    $undefinedCount = ([regex]::Matches($logText, '(?i)undefined')).Count

    Write-Host "$Target : $pages pages, $overfullCount overfull hbox(es), $underfullCount underfull hbox(es), $undefinedCount undefined-reference warning(s)" -ForegroundColor Green

    # Swap into place. The previous $finalPdf is never removed first --
    # Copy-Item -Force either replaces it atomically or throws, in which
    # case the previous file is left exactly as it was.
    $swapped = $true
    try {
        Copy-Item -Path $builtPdf -Destination $finalPdf -Force -ErrorAction Stop
    } catch {
        $swapped = $false
        Write-Host ""
        Write-Host "COULD NOT UPDATE $finalPdf : the file appears to be open in another program (a PDF viewer, WPS Office, etc.)." -ForegroundColor Yellow
        Write-Host "Close $finalPdf and re-run this script. The previous $finalPdf has NOT been changed." -ForegroundColor Yellow
        Write-Host "The newly compiled PDF is available at $builtPdf until the next run overwrites it." -ForegroundColor Yellow
    }

    # Clean up every intermediate/auxiliary file this build produced. Only
    # remove $builtPdf itself if the swap succeeded -- if it didn't, the
    # user still needs it to recover this build without recompiling.
    $toRemove = @("$jobName.aux","$jobName.bbl","$jobName.blg","$jobName.out","$jobName.spl",
                  $consoleCapture,$builtLog)
    if ($swapped) { $toRemove += $builtPdf }
    Remove-Item -Force -ErrorAction SilentlyContinue $toRemove

    return $swapped
}

$elsevierOk = Build-Target -Target 'elsevier'
$ieeeOk     = Build-Target -Target 'ieee'

Write-Host ""
if ($elsevierOk -and $ieeeOk) {
    Write-Host "=== Done. draft_elsevier.pdf and draft_ieee.pdf are up to date. ===" -ForegroundColor Cyan
    exit 0
} else {
    Write-Host "=== Finished with problems -- see messages above. ===" -ForegroundColor Red
    exit 1
}
