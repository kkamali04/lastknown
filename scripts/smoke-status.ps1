# smoke-status.ps1 - CI smoke check: fails if embedding coverage < 99.0 %
# Usage: .\scripts\smoke-status.ps1
# Expects the release binary at <repo-root>\target\release\recall.exe
# Compatible with Windows PowerShell 5.1 (no ternary, no ??, no &&).

$ErrorActionPreference = 'Stop'

$RepoRoot  = Split-Path -Parent $PSScriptRoot
$BinaryPath = Join-Path $RepoRoot 'target\release\recall.exe'

if (-not (Test-Path $BinaryPath)) {
    Write-Error "Binary not found: $BinaryPath - build the release target first."
    exit 1
}

$Output = & $BinaryPath status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "'recall status' exited with code $LASTEXITCODE.`nOutput:`n$Output"
    exit 1
}

# Match:  embeddings    : 100998 / 100999  (100.00%)
$Pattern = 'embeddings\s*:\s*(\d+)\s*/\s*(\d+)\s*\((\d+(?:\.\d+)?)%\)'
$Line = $Output | Where-Object { $_ -match $Pattern } | Select-Object -First 1

if (-not $Line) {
    Write-Error "Could not find an 'embeddings : X / Y (Z%)' line in 'recall status' output.`nOutput:`n$Output"
    exit 1
}

$null = $Line -match $Pattern
$Embedded  = [long]   $Matches[1]
$Total     = [long]   $Matches[2]
$PctString =          $Matches[3]
$Pct       = [double] $PctString

if ($Total -eq 0) {
    Write-Error "Total document count is 0 - store may be empty or unindexed."
    exit 1
}

$Threshold = 99.0

if ($Pct -lt $Threshold) {
    Write-Error "FAIL: embedding coverage is ${Pct}% ($Embedded / $Total) - below threshold ${Threshold}%."
    exit 1
}

Write-Host "OK: embedding coverage ${Pct}% ($Embedded / $Total) - above threshold ${Threshold}%."
exit 0
