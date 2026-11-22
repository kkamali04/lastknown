# Stale-memory demo: recall tells you when a result is a copy of a file that has since changed,
# and tells a changed FILE whose shown PASSAGE is untouched ("this passage unchanged") from a
# changed PASSAGE ("this passage changed: read the file").
#
#   powershell -ExecutionPolicy Bypass -File examples\stale-demo\run.ps1 [-Bin path\to\recall.exe]
#
# Needs only the recall binary and ripgrep (rg) on PATH. Keyword channel only (-c fts5), so no model download.
# Everything happens in a temp HOME and a temp vault; your real %USERPROFILE%\.recall is never read or written.
# Exit 0 = every check passed, 1 = a check failed, 2 = could not run (missing binary / ripgrep).
param([string]$Bin = $(if ($env:RECALL_BIN) { $env:RECALL_BIN } else { 'recall' }))
$ErrorActionPreference = 'Continue'   # recall writes banners to stderr; do not turn them into errors
if (-not (Get-Command $Bin -ErrorAction SilentlyContinue)) { Write-Host "recall binary not found: $Bin"; exit 2 }
if (-not (Get-Command rg -ErrorAction SilentlyContinue)) { Write-Host 'ripgrep (rg) not found on PATH; recall index needs it'; exit 2 }
$Bin = (Get-Command $Bin).Source
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("recall-stale-demo-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path "$tmp\home", "$tmp\vault" | Out-Null
Get-ChildItem Env: | Where-Object { $_.Name -like 'RECALL_*' } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
$env:HOME = "$tmp\home"; $env:USERPROFILE = "$tmp\home"
$note = "$tmp\vault\api-notes.md"
$script:bad = 0
$script:out = ''

function Check([string]$desc, [bool]$ok) {
    if ($ok) { Write-Host "  PASS  $desc" } else { Write-Host "  FAIL  $desc"; $script:bad = 1 }
}
function Has([string]$s) { return $script:out.Contains($s) }
function WriteNote([string]$intro, [string]$port) {
    $text = "# API notes`n`n$intro`n`nThe API port is $port`n`nQuestions go to the platform channel.`n"
    [IO.File]::WriteAllText($note, $text, (New-Object Text.UTF8Encoding($false)))
}
function Ask {
    $script:out = (& $Bin query -c fts5 "API port" 2>&1 | ForEach-Object { "$_" }) -join "`n"
    $rc = $LASTEXITCODE
    Write-Host $script:out
    Check 'query exits 0' ($rc -eq 0)
}
function Index {
    $lines = & $Bin index 2>&1 | ForEach-Object { "$_" }
    $rc = $LASTEXITCODE
    $lines | Where-Object { $_ -match 'Indexed|docs in index' } | ForEach-Object { Write-Host $_ }
    Check 'index exits 0' ($rc -eq 0)
}

try {
    Write-Host '== 1. write a note and index it'
    WriteNote 'Kept by the platform team.' '8080'
    & $Bin settings set vault "$tmp\vault" *> $null
    Check 'settings set vault exits 0' ($LASTEXITCODE -eq 0)
    Index
    Write-Host ''
    Write-Host '== 2. query: fresh, the file matches the index'
    Ask
    Check 'finds the note with 8080' (Has '8080')
    Check 'no stale warning' (-not (Has 'STALE'))

    Write-Host ''
    Write-Host '== 3. edit ANOTHER part of the file (the intro line) WITHOUT re-indexing, then query'
    Start-Sleep -Seconds 1
    WriteNote 'Kept by the infrastructure team, since March.' '8080'
    Ask
    Check 'not flagged STALE: the passage shown is still in the file' (-not (Has 'STALE'))
    Check 'says the file changed but this passage is unchanged' (Has 'this passage unchanged')
    Check 'still shows 8080' (Has '8080')

    Write-Host ''
    Write-Host '== 4. edit the PASSAGE itself (8080 -> 9090) WITHOUT re-indexing, then query again'
    Start-Sleep -Seconds 1
    WriteNote 'Kept by the infrastructure team, since March.' '9090'
    Ask
    Check 'hit is flagged STALE' (Has 'STALE')
    Check 'says this passage changed' (Has 'this passage changed')
    Check 'names the file' (Has 'api-notes.md')
    Check 'tells you to read the real file' (Has 'read the actual file')
    Check 'the flagged hit still shows the old indexed text (8080), not a silent rewrite' (Has '8080')

    Write-Host ''
    Write-Host '== 5. re-index, then query: the fresh answer'
    Index
    Ask
    Check 'fresh answer is 9090' (Has '9090')
    Check 'old 8080 text is gone' (-not (Has '8080'))
    Check 'no stale warning' (-not (Has 'STALE'))
}
finally {
    Remove-Item -Recurse -Force -LiteralPath $tmp -ErrorAction SilentlyContinue
}
Write-Host ''
if ($script:bad -eq 0) { Write-Host 'stale-demo: PASS' } else { Write-Host 'stale-demo: FAIL' }
exit $script:bad
