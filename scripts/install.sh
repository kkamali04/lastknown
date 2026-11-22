#!/usr/bin/env sh
# recall installer for Linux and macOS (arm64). Windows: scripts/install.ps1.
#
#   sh scripts/install.sh [--vault DIR] [--autostart] [--register-mcp] [--skip-build] [--check-deps]
#
# Needs ripgrep (rg): recall's rg channel and `index --prune-missing` shell out to it. A missing rg
# is installed with brew / apt-get / dnf / pacman when one is present, else the install stops.
# --check-deps runs only that check.
#
# Builds the binary from this checkout, fetches ONNX Runtime and the BGE-small model from their
# official sources (every file SHA-256 verified), writes the config dir, and finishes with a
# check that FAILS LOUDLY if the semantic (bge) channel cannot load. Free and offline after the
# downloads: no API keys, no paid channels.
set -eu

# ---- pinned downloads (digests copied from the upstream release/LFS metadata, re-verified) ----
ORT_VER="1.24.4"
ORT_URL="https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/${ORT_VER}/microsoft.ml.onnxruntime.${ORT_VER}.nupkg"
ORT_SHA="4b978d5065b85e7004b6c6f60ca494bd978fbe6836cbf0a0b52d82b61ab99638"
MODEL_REV="ea104dacec62c0de699686887e3f920caeb4f3e3"
MODEL_URL="https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/${MODEL_REV}"
MODEL_FILES="onnx/model.onnx:828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35
tokenizer.json:d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66
config.json:fa73f90bf92c8cace1fbcb709626306f2bdbc9ea3e5b5f94b440df9b6aa56350
special_tokens_map.json:b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3
tokenizer_config.json:9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3"

VAULT=""; AUTOSTART=0; SKIP_BUILD=0; REGISTER=0; CHECK_DEPS=0; RHOME="$HOME/.recall"
while [ $# -gt 0 ]; do
  case "$1" in
    --vault) VAULT="$2"; shift 2 ;;
    --autostart) AUTOSTART=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --register-mcp) REGISTER=1; shift ;;
    --check-deps) CHECK_DEPS=1; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

die() { echo "INSTALL FAILED: $*" >&2; exit 1; }
shq() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\"'\"'/g")"; }   # POSIX single-quote a value (' -> '"'"')
xmle() { printf '%s' "$1" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'; }
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ -f "$ROOT/Cargo.toml" ] || die "run from a recall checkout (no Cargo.toml at $ROOT)"

# ---- 0. ripgrep: preflight, install if a package manager is here, else refuse ----
rg_works() { command rg --version >/dev/null 2>&1; }   # an executable that runs (`command` skips shell functions)
ensure_rg() {
  rg_works && return 0
  SUDO=""; [ "$(id -u)" -eq 0 ] || SUDO="sudo"
  echo "==> ripgrep (rg) not found; recall's rg channel and prune need it. Installing."
  if command -v brew >/dev/null 2>&1; then brew install ripgrep || true
  elif command -v apt-get >/dev/null 2>&1; then { $SUDO apt-get update -qq && $SUDO apt-get install -y ripgrep; } || true
  elif command -v dnf >/dev/null 2>&1; then $SUDO dnf install -y ripgrep || true
  elif command -v pacman >/dev/null 2>&1; then $SUDO pacman -S --noconfirm ripgrep || true
  fi
  rg_works || die "ripgrep (rg) is required and could not be installed automatically.
  Install it, then re-run this script:
    macOS:          brew install ripgrep
    Debian/Ubuntu:  sudo apt-get install ripgrep
    Fedora:         sudo dnf install ripgrep
    Arch:           sudo pacman -S ripgrep
    other:          https://github.com/BurntSushi/ripgrep#installation"
  echo "==> ripgrep installed"
}
ensure_rg
[ "$CHECK_DEPS" -eq 1 ] && { echo "dependencies OK"; exit 0; }

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)  ORT_RUNTIME="linux-x64" ;;
  Linux-aarch64|Linux-arm64) ORT_RUNTIME="linux-arm64" ;;
  Darwin-arm64) ORT_RUNTIME="osx-arm64" ;;
  *) die "unsupported platform $(uname -s)-$(uname -m): the pinned ONNX Runtime ${ORT_VER} package supports Linux x64/arm64 and macOS arm64. Intel macOS is not supported by this installer." ;;
esac

command -v curl >/dev/null || die "curl is required"
PY=$(command -v python3 || command -v python) || die "Python 3.10+ is required (including venv support)"
"$PY" -c 'import sys, venv; assert sys.version_info >= (3, 10)' || die "Python 3.10+ with venv is required"
if command -v sha256sum >/dev/null; then sha256() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null; then sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
else die "sha256sum or shasum is required"; fi

fetch() { # url dest expected_sha256
  curl -fL --retry 3 -sS -o "$2" "$1" || die "download failed: $1"
  got=$(sha256 "$2")
  [ "$got" = "$3" ] || { rm -f "$2"; die "checksum mismatch for $1 (expected $3, got $got)"; }
}

BIN="$RHOME/bin"; mkdir -p "$BIN" "$RHOME/licenses"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

# ---- 1. build ----
if [ "$SKIP_BUILD" -eq 0 ]; then
  command -v cargo >/dev/null || die "cargo not found. Install Rust first: https://rustup.rs"
  echo "==> building recall (release, default features: free/offline channels only)"
  export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
  (cd "$ROOT" && cargo build --release) || die "cargo build failed"
fi
SRC="${CARGO_TARGET_DIR:-$ROOT/target}/release/recall"
[ -x "$SRC" ] || die "binary not found at $SRC"
cp "$SRC" "$BIN/recall"; chmod +x "$BIN/recall"

# ---- 2. ONNX Runtime (official Microsoft NuGet distribution) ----
echo "==> ONNX Runtime ${ORT_VER}"
fetch "$ORT_URL" "$TMP/ort.nupkg" "$ORT_SHA"
"$PY" "$ROOT/scripts/ort_package.py" "$TMP/ort.nupkg" "$ORT_RUNTIME" "$BIN" "$RHOME/licenses/onnxruntime" || die "ONNX Runtime extraction failed"

# ---- 3. BGE-small model (Xenova ONNX export of BAAI/bge-small-en-v1.5, pinned commit) ----
echo "==> BGE-small-en-v1.5 model"
SNAP="$RHOME/models/models--Xenova--bge-small-en-v1.5"
mkdir -p "$SNAP/refs" "$SNAP/snapshots/$MODEL_REV/onnx"
# here-doc, not a pipe: `die` inside a piped loop would only exit the subshell and the install
# would carry on half-finished.
while IFS=: read -r f sum; do
  dest="$SNAP/snapshots/$MODEL_REV/$f"
  if [ -f "$dest" ] && [ "$(sha256 "$dest")" = "$sum" ]; then continue; fi
  fetch "$MODEL_URL/$f" "$dest" "$sum"
done <<EOF
$MODEL_FILES
EOF
printf '%s' "$MODEL_REV" > "$SNAP/refs/main"

# ---- 4. config + vault ----
[ -n "$VAULT" ] || VAULT="$HOME/recall-notes"
mkdir -p "$VAULT"
VAULT=$(cd "$VAULT" && pwd)   # absolute: a relative path would follow the caller's cwd
# The ONE place the notes folder is recorded: env.sh, autostart and MCP registration do not pin
# RECALL_VAULT (it would override this setting), so `recall settings set vault` changes it everywhere.
"$BIN/recall" settings set vault "$VAULT" >/dev/null || die "could not record the vault setting"
cat > "$RHOME/env.sh" <<EOF
# source this, or copy the lines into your shell profile
export RECALL_CHANNELS="fts5,bge"
export PATH=$(shq "$BIN"):"\$PATH"
EOF

# ---- 5. loud semantic-channel check ----
echo "==> verifying the semantic (bge) channel loads"
if ! RECALL_VAULT="$VAULT" "$BIN/recall" doctor embed; then
  die "the semantic channel did NOT load (see output above). Do not use this install: it would silently fall back to keyword-only."
fi

# ---- 6. optional autostart of the warm daemon ----
if [ "$AUTOSTART" -eq 1 ]; then
  case "$VAULT$BIN" in *\"*|*\*) die "--autostart: paths containing a double quote or backslash are not supported" ;; esac
  if [ "$(uname -s)" = "Linux" ] && command -v systemctl >/dev/null; then
    U="$HOME/.config/systemd/user"; mkdir -p "$U"
    cat > "$U/recall-daemon.service" <<EOF
[Unit]
Description=recall warm daemon
[Service]
Environment=RECALL_CHANNELS=fts5,bge
ExecStart="$(printf '%s' "$BIN" | sed 's/%/%%/g')/recall" daemon up
Restart=on-failure
[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload && systemctl --user enable --now recall-daemon.service \
      || die "autostart requested but systemctl --user failed (no user session?). Run '$BIN/recall daemon up' yourself or drop --autostart."
    echo "daemon enabled (systemd --user). For start-at-boot without login: loginctl enable-linger \$USER"
  elif [ "$(uname -s)" = "Darwin" ]; then
    P="$HOME/Library/LaunchAgents/org.recall.daemon.plist"; mkdir -p "$(dirname "$P")"
    cat > "$P" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>org.recall.daemon</string>
<key>ProgramArguments</key><array><string>$(xmle "$BIN")/recall</string><string>daemon</string><string>up</string></array>
<key>EnvironmentVariables</key><dict><key>RECALL_CHANNELS</key><string>fts5,bge</string></dict>
<key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
</dict></plist>
EOF
    plutil -lint "$P" >/dev/null || die "generated launchd plist is invalid"
    launchctl load "$P" || die "autostart requested but launchctl load failed"
    echo "daemon registered (launchd)"
  else
    echo "autostart: no systemd/launchd found; run '$BIN/recall daemon up' yourself" >&2
  fi
fi

# ---- 7. MCP server: private venv + a copy of the server next to the binary ----
echo "==> MCP server (fastmcp in $RHOME/venv)"
"$PY" -m venv "$RHOME/venv" || die "could not create a python venv (Debian/Ubuntu: apt install python3-venv)"
"$RHOME/venv/bin/python" -m pip install -q -r "$ROOT/mcp/requirements.txt" || die "pip install of the MCP requirements failed"
mkdir -p "$RHOME/mcp"; cp "$ROOT/mcp/recall_mcp.py" "$RHOME/mcp/recall_mcp.py"
MCPPY="$RHOME/venv/bin/python"; MCPSRV="$RHOME/mcp/recall_mcp.py"
RECALL_EXE="$BIN/recall" "$MCPPY" "$ROOT/mcp/smoke_handshake.py" --quiet --server "$MCPSRV" >/dev/null 2>&1 \
  && echo "MCP handshake OK" || die "the MCP server did not complete a handshake; run: RECALL_EXE=$BIN/recall $MCPPY $ROOT/mcp/smoke_handshake.py --server $MCPSRV"
if [ "$REGISTER" -eq 1 ]; then
  if command -v claude >/dev/null; then
    claude mcp add recall -e RECALL_EXE="$BIN/recall" -- "$MCPPY" "$MCPSRV" || die "claude mcp add failed"
    echo "registered with Claude Code"
  fi
  if command -v codex >/dev/null; then
    codex mcp add recall --env RECALL_EXE="$BIN/recall" -- "$MCPPY" "$MCPSRV" || die "codex mcp add failed"
    echo "registered with Codex"
  fi
fi

cat <<EOF

recall installed.  home: $RHOME   binary: $BIN/recall   notes: $VAULT

Next:
  . "$RHOME/env.sh"                 # or add its two export lines to your shell profile
  recall index && recall embed       # index + vectorize your notes folder (.md/.qmd)
  recall query --top 5 -- "your question"

MCP server installed (python venv + fastmcp). Register it (or re-run with --register-mcp):
  Claude Code:  claude mcp add recall -e RECALL_EXE="$BIN/recall" -- "$MCPPY" "$MCPSRV"
  Codex:        codex mcp add recall --env RECALL_EXE="$BIN/recall" -- "$MCPPY" "$MCPSRV"
  Cursor/others (mcpServers JSON): docs/MCP.md
EOF
