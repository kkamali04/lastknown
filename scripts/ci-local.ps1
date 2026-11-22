# Routine CI on this Windows machine instead of GitHub Actions (the free minutes are nearly gone).
#
#   powershell -ExecutionPolicy Bypass -File scripts\ci-local.ps1 [-Only test,ops,pytest,smoke,scrub,demo]
#
# Steps: build (release) | test (cargo test) | ops (cargo test --features ops-health) | pytest (ingest + mcp + installer) | scrub (scrub_e2e)
#        | smoke (README install + first use) | demo (examples\stale-demo). release_check is NOT included.
# The whole run uses a TEMP home (USERPROFILE/HOME, no inherited RECALL_* variables), so the real
# ~/.recall is never read or written; cargo/rustup homes are pinned first.
# Builds into <repo>\target-ci; one run per checkout at a time (a lock refuses a second).
# Prints PASS/FAIL per step; exits 1 if any failed or any requested step did not run.
# Linux/VPS: scripts/ci-local.sh.
[CmdletBinding()]
param([string[]]$Only = @('build', 'test', 'ops', 'pytest', 'scrub', 'smoke', 'demo'))
$ErrorActionPreference = 'Continue'
$Known = @('build', 'test', 'ops', 'pytest', 'scrub', 'smoke', 'demo')
foreach ($n in $Only) { if ($Known -notcontains $n) { Write-Host "unknown step '$n' (known: $($Known -join ', '))"; exit 2 } }
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'   # children inherit it
# Pin the toolchain homes first: a temp USERPROFILE would otherwise hide rustup/cargo.
if (-not $env:CARGO_HOME)  { $env:CARGO_HOME  = Join-Path $env:USERPROFILE '.cargo' }
if (-not $env:RUSTUP_HOME) { $env:RUSTUP_HOME = Join-Path $env:USERPROFILE '.rustup' }
$env:CARGO_TARGET_DIR = Join-Path $Root 'target-ci'
$Exe = Join-Path $env:CARGO_TARGET_DIR 'release\recall.exe'
$Lock = Join-Path $Root '.ci-local.lock'
if (-not (Get-Command rg -CommandType Application -ErrorAction SilentlyContinue)) { Write-Host 'ripgrep (rg) missing: tests need it (winget install BurntSushi.ripgrep.MSVC)'; exit 2 }
try { New-Item -ItemType Directory -Path $Lock -ErrorAction Stop | Out-Null } catch { Write-Host "another ci-local run holds $Lock (delete it if that run is dead)"; exit 2 }

$Tmp = Join-Path ([IO.Path]::GetTempPath()) ("recall-ci-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
$Home2 = Join-Path $Tmp 'home'
New-Item -ItemType Directory -Force -Path $Home2 | Out-Null
$saved = @{ USERPROFILE = $env:USERPROFILE; HOME = $env:HOME; PSModulePath = $env:PSModulePath }
Get-ChildItem Env: | Where-Object { $_.Name -like 'RECALL_*' } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
$env:USERPROFILE = $Home2; $env:HOME = $Home2
Remove-Item Env:PYTEST_ADDOPTS, Env:PYTEST_PLUGINS -ErrorAction SilentlyContinue
$env:RECALL_ERROR_LOG = Join-Path $Tmp 'errors.log'
# git's global safe.directory lived in the real home; keep repos under the real home (a vault, a tmp dir) usable.
$env:GIT_CONFIG_COUNT = '1'; $env:GIT_CONFIG_KEY_0 = 'safe.directory'; $env:GIT_CONFIG_VALUE_0 = '*'
# A child Windows PowerShell (install.ps1) needs its own default module path, not a PowerShell 7 one.
$env:PSModulePath = [Environment]::GetEnvironmentVariable('PSModulePath', 'Machine')
$results = [ordered]@{}

function Step($name, [scriptblock]$body) {
    if ($Only -notcontains $name) { return }
    Write-Host "==> $name" -ForegroundColor Cyan
    $global:LASTEXITCODE = 0
    try { & $body; $ok = ($LASTEXITCODE -eq 0) } catch { Write-Host $_; $ok = $false }
    $results[$name] = if ($ok) { 'PASS' } else { 'FAIL' }
}

try {
    $venv = Join-Path $Tmp 'venv'
    $py = Join-Path $venv 'Scripts\python.exe'
    python -m venv $venv
    if ($LASTEXITCODE -ne 0) { $results['setup'] = 'FAIL' } else {
        & $py -m pip install -q -r mcp\requirements.txt pytest
        if ($LASTEXITCODE -ne 0) { $results['setup'] = 'FAIL' }
    }
    if (-not $results.Contains('setup')) {
        Step build  { cargo build --release -j 4 }
        Step test   { cargo test --release -j 4 -- --test-threads=2 }
        Step ops    { cargo test --release -j 4 --features ops-health -- --test-threads=2 }
        Step pytest { & $py -m pytest ingest mcp/test_recall_mcp.py scripts/test_install_deps.py scripts/test_clean_install_smoke.py -q }
        Step scrub  { & $py -I scripts/scrub_e2e.py $Exe }
        Step smoke  {
            powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install.ps1 -Vault (Join-Path $Home2 'notes')
            if ($LASTEXITCODE -ne 0) { return }
            python scripts/clean_install_smoke.py
        }
        Step demo   { powershell -NoProfile -ExecutionPolicy Bypass -File examples\stale-demo\run.ps1 -Bin $Exe }
    }
}
finally {
    $env:USERPROFILE = $saved.USERPROFILE; $env:HOME = $saved.HOME; $env:PSModulePath = $saved.PSModulePath
    Remove-Item -Recurse -Force -LiteralPath $Tmp -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force -LiteralPath $Lock -ErrorAction SilentlyContinue
}
Write-Host "`n==== ci-local summary ===="
$bad = $false
foreach ($n in (@('setup') + $Only)) {
    if ($n -eq 'setup' -and -not $results.Contains('setup')) { continue }
    $r = if ($results.Contains($n)) { $results[$n] } else { 'NOT-RUN' }
    '{0,-8} {1}' -f $r, $n
    if ($r -ne 'PASS') { $bad = $true }
}
if ($bad) { exit 1 }
