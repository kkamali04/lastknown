# recall installer for Windows (x64). Linux/macOS: scripts/install.sh.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install.ps1 [-Vault DIR] [-Autostart] [-RegisterMcp] [-SkipBuild] [-CheckDeps]
#
# Needs ripgrep (rg): recall's rg channel and `index --prune-missing` shell out to it. A missing rg
# is installed with winget or choco when one is present, else the install stops. -CheckDeps runs
# only that check.
#
# Builds the binary from this checkout, fetches ONNX Runtime and the BGE-small model from their
# official sources (every file SHA-256 verified), writes the config dir, and finishes with a
# check that FAILS LOUDLY if the semantic (bge) channel cannot load. Free and offline after the
# downloads: no API keys, no paid channels.
[CmdletBinding()]
param(
    [string]$Vault = "",
    [switch]$Autostart,
    [switch]$SkipBuild,
    [switch]$RegisterMcp,
    [switch]$CheckDeps
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is ~10x faster without the progress bar
# Windows PowerShell launched from PowerShell 7 can inherit a module path that
# hides its own Utility/Archive modules. Use the legacy host's module roots.
if ($PSVersionTable.PSVersion.Major -lt 6) {
    $env:PSModulePath = (Join-Path $PSHOME 'Modules') + ';' + [Environment]::GetEnvironmentVariable('PSModulePath', 'Machine')
}

# ---- pinned downloads (digests copied from the upstream release/LFS metadata, re-verified) ----
$OrtVer = '1.24.4'
# Microsoft's NuGet distribution contains the same Windows native DLL as the
# release archive. Pin the whole package, including its LICENSE and notices.
$OrtUrl = "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/$OrtVer/microsoft.ml.onnxruntime.$OrtVer.nupkg"
$OrtSha = '4b978d5065b85e7004b6c6f60ca494bd978fbe6836cbf0a0b52d82b61ab99638'
$ModelRev = 'ea104dacec62c0de699686887e3f920caeb4f3e3'
$ModelUrl = "https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/$ModelRev"
$ModelFiles = [ordered]@{
    'onnx/model.onnx'         = '828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35'
    'tokenizer.json'          = 'd241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66'
    'config.json'             = 'fa73f90bf92c8cace1fbcb709626306f2bdbc9ea3e5b5f94b440df9b6aa56350'
    'special_tokens_map.json' = 'b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3'
    'tokenizer_config.json'   = '9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3'
}

function Die($msg) { Write-Error "INSTALL FAILED: $msg"; exit 1 }

# ---- 0. ripgrep: preflight, install if a package manager is here, else refuse ----
# An executable that runs and exits 0, not just a name (a function or alias named rg does not count).
function Test-Ripgrep {
    $exe = Get-Command rg -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $exe) { return $false }
    try { & $exe.Source --version *> $null; return ($LASTEXITCODE -eq 0) } catch { return $false }
}
function Confirm-Ripgrep {
    if (Test-Ripgrep) { return }
    Write-Host "==> ripgrep (rg) not found; recall's rg channel and prune need it. Installing."
    $tried = $true
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id BurntSushi.ripgrep.MSVC -e --silent --accept-source-agreements --accept-package-agreements | Out-Host
    } elseif (Get-Command choco -ErrorAction SilentlyContinue) {
        choco install ripgrep -y --no-progress | Out-Host
    } else { $tried = $false }
    if ($tried) {
        # A fresh install lands on the registry PATH, not this process's copy of it.
        $env:PATH = [Environment]::GetEnvironmentVariable('PATH', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('PATH', 'User') + ';' + $env:PATH
    }
    if (-not (Test-Ripgrep)) {
        Die ("ripgrep (rg) is required and could not be installed automatically.`n" +
             "  Install it, then re-run this script (open a new terminal first):`n" +
             "    winget install BurntSushi.ripgrep.MSVC`n" +
             "    choco install ripgrep`n" +
             "    scoop install ripgrep`n" +
             "    or https://github.com/BurntSushi/ripgrep/releases")
    }
    Write-Host '==> ripgrep installed'
}

function Get-Verified($url, $dest, $sha) {
    Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
    $got = (Get-FileHash -Algorithm SHA256 -LiteralPath $dest).Hash.ToLower()
    if ($got -ne $sha) { Remove-Item -LiteralPath $dest -Force; Die "checksum mismatch for $url (expected $sha, got $got)" }
}

$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root 'Cargo.toml'))) { Die "run from a recall checkout (no Cargo.toml at $Root)" }
Confirm-Ripgrep
if ($CheckDeps) { Write-Host 'dependencies OK'; exit 0 }
$RecallHome = Join-Path $env:USERPROFILE '.recall'
function PsQ($v) { "'" + ($v -replace "'", "''") + "'" }   # PowerShell single-quoted literal
$Bin = Join-Path $RecallHome 'bin'
New-Item -ItemType Directory -Force -Path $Bin, (Join-Path $RecallHome 'licenses\onnxruntime') | Out-Null
$TempBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\', '/')
$Tmp = Join-Path $TempBase ("recall-install-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null

try {
    # ---- 1. build ----
    if (-not $SkipBuild) {
        if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) { Die 'cargo not found. Install Rust first: https://rustup.rs' }
        Write-Host '==> building recall (release, default features: free/offline channels only)'
        if (-not $env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR = Join-Path $Root 'target' }
        Push-Location $Root
        try { cargo build --release; if ($LASTEXITCODE -ne 0) { Die 'cargo build failed' } } finally { Pop-Location }
    }
    $TargetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $Root 'target' }
    $Src = Join-Path $TargetDir 'release\recall.exe'
    if (-not (Test-Path $Src)) { Die "binary not found at $Src" }
    Copy-Item $Src (Join-Path $Bin 'recall.exe') -Force

    # ---- 2. ONNX Runtime (official Microsoft release) ----
    Write-Host "==> ONNX Runtime $OrtVer"
    $zip = Join-Path $Tmp 'ort.zip'
    Get-Verified $OrtUrl $zip $OrtSha
    Expand-Archive -LiteralPath $zip -DestinationPath $Tmp -Force
    $OrtDir = Join-Path $Tmp 'runtimes\win-x64\native'
    Copy-Item (Join-Path $OrtDir 'onnxruntime.dll') $Bin -Force
    Copy-Item (Join-Path $OrtDir 'onnxruntime_providers_shared.dll') $Bin -Force
    Copy-Item (Join-Path $Tmp 'LICENSE'), (Join-Path $Tmp 'ThirdPartyNotices.txt') (Join-Path $RecallHome 'licenses\onnxruntime') -Force

    # ---- 3. BGE-small model (Xenova ONNX export of BAAI/bge-small-en-v1.5, pinned commit) ----
    Write-Host '==> BGE-small-en-v1.5 model'
    $Snap = Join-Path $RecallHome 'models\models--Xenova--bge-small-en-v1.5'
    $SnapDir = Join-Path $Snap "snapshots\$ModelRev"
    New-Item -ItemType Directory -Force -Path (Join-Path $Snap 'refs'), (Join-Path $SnapDir 'onnx') | Out-Null
    foreach ($f in $ModelFiles.Keys) {
        $dest = Join-Path $SnapDir ($f -replace '/', '\')
        if ((Test-Path $dest) -and ((Get-FileHash -Algorithm SHA256 -LiteralPath $dest).Hash.ToLower() -eq $ModelFiles[$f])) { continue }
        Get-Verified "$ModelUrl/$f" $dest $ModelFiles[$f]
    }
    [IO.File]::WriteAllText((Join-Path $Snap 'refs\main'), $ModelRev)   # no trailing newline, no BOM

    # ---- 4. config + vault ----
    if (-not $Vault) { $Vault = Join-Path $env:USERPROFILE 'recall-notes' }
    New-Item -ItemType Directory -Force -Path $Vault | Out-Null
    $Vault = (Resolve-Path -LiteralPath $Vault).Path   # absolute: a relative path would follow the caller's cwd
    # The ONE place the notes folder is recorded: env.ps1, autostart and MCP registration do not pin
    # RECALL_VAULT (it would override this setting), so `recall settings set vault` changes it everywhere.
    & (Join-Path $Bin 'recall.exe') settings set vault $Vault | Out-Null; if ($LASTEXITCODE -ne 0) { Die 'could not record the vault setting' }
    @(
        '# dot-source this, or copy the lines into your PowerShell profile',
        "`$env:RECALL_CHANNELS = 'fts5,bge'",
        "`$env:PATH = $(PsQ ($Bin + ';')) + `$env:PATH"
    ) | Set-Content -Encoding UTF8 (Join-Path $RecallHome 'env.ps1')

    # ---- 5. loud semantic-channel check ----
    Write-Host '==> verifying the semantic (bge) channel loads'
    $env:RECALL_VAULT = $Vault
    & (Join-Path $Bin 'recall.exe') doctor embed
    if ($LASTEXITCODE -ne 0) { Die 'the semantic channel did NOT load (see output above). Do not use this install: it would silently fall back to keyword-only.' }

    # ---- 6. optional autostart of the warm daemon (at logon, hidden) ----
    if ($Autostart) {
        $launcher = Join-Path $RecallHome 'start-daemon.ps1'
        @(
            "`$env:RECALL_CHANNELS = 'fts5,bge'",
            "& $(PsQ (Join-Path $Bin 'recall.exe')) daemon up"
        ) | Set-Content -Encoding UTF8 $launcher
        $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$launcher`""
        $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
        $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero)
        Register-ScheduledTask -TaskName 'recall-daemon' -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
        Write-Host 'daemon registered: scheduled task "recall-daemon" (runs at logon). Remove: Unregister-ScheduledTask recall-daemon'
    }

    # ---- 7. MCP server: private venv + a copy of the server next to the binary ----
    Write-Host "==> MCP server (fastmcp in $RecallHome\venv)"
    $py = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $py) { $py = (Get-Command py -ErrorAction SilentlyContinue).Source }
    if (-not $py) { Die 'python is required for the MCP server (https://www.python.org/downloads/)' }
    & $py -m venv (Join-Path $RecallHome 'venv'); if ($LASTEXITCODE -ne 0) { Die 'could not create the python venv' }
    $McpPy = Join-Path $RecallHome 'venv\Scripts\python.exe'
    & $McpPy -m pip install -q -r (Join-Path $Root 'mcp\requirements.txt'); if ($LASTEXITCODE -ne 0) { Die 'pip install of the MCP requirements failed' }
    New-Item -ItemType Directory -Force -Path (Join-Path $RecallHome 'mcp') | Out-Null
    Copy-Item (Join-Path $Root 'mcp\recall_mcp.py') (Join-Path $RecallHome 'mcp\recall_mcp.py') -Force
    $McpSrv = Join-Path $RecallHome 'mcp\recall_mcp.py'
    $env:RECALL_EXE = Join-Path $Bin 'recall.exe'
    $ErrorActionPreference = 'Continue'   # native stderr (fastmcp banner) must not abort the install
    & $McpPy (Join-Path $Root 'mcp\smoke_handshake.py') --quiet --server $McpSrv 2>&1 | Out-Null
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE -eq 0) { Write-Host 'MCP handshake OK' } else { Die "the MCP server did not complete a handshake; run: $McpPy $Root\mcp\smoke_handshake.py --server $McpSrv" }
    if ($RegisterMcp) {
        if (Get-Command claude -ErrorAction SilentlyContinue) { claude mcp add recall -e "RECALL_EXE=$($env:RECALL_EXE)" -- $McpPy $McpSrv; if ($LASTEXITCODE -ne 0) { Die "claude mcp add failed" } }
        if (Get-Command codex -ErrorAction SilentlyContinue) { codex mcp add recall --env "RECALL_EXE=$($env:RECALL_EXE)" -- $McpPy $McpSrv; if ($LASTEXITCODE -ne 0) { Die "codex mcp add failed" } }
    }
}
finally {
    $ResolvedTmp = [IO.Path]::GetFullPath($Tmp)
    if (([IO.Path]::GetDirectoryName($ResolvedTmp) -eq $TempBase) -and
        ([IO.Path]::GetFileName($ResolvedTmp) -match '^recall-install-[0-9a-f]{32}$')) {
        Remove-Item -Recurse -Force -LiteralPath $ResolvedTmp -ErrorAction SilentlyContinue
    } else { Write-Warning 'Leaving installer temporary files: cleanup path did not pass containment check.' }
}

$McpPy = Join-Path $RecallHome 'venv\Scripts\python.exe'; $McpSrv = Join-Path $RecallHome 'mcp\recall_mcp.py'
Write-Host @"

recall installed.  home: $RecallHome   binary: $Bin\recall.exe   notes: $Vault

Next:
  . "$RecallHome\env.ps1"           # or add its two lines to your PowerShell profile
  recall index; recall embed          # index + vectorize your notes (.md/.qmd)
  recall query --top 5 -- "your question"

MCP server installed (python venv + fastmcp). Register it (or re-run with -RegisterMcp):
  Claude Code:  claude mcp add recall -e RECALL_EXE="$Bin\recall.exe" -- "$McpPy" "$McpSrv"
  Codex:        codex mcp add recall --env RECALL_EXE="$Bin\recall.exe" -- "$McpPy" "$McpSrv"
  Cursor/others (mcpServers JSON): docs\MCP.md
"@
