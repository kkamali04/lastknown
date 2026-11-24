#!/bin/sh
# Routine CI on Linux (the VPS) instead of GitHub Actions. Run FROM a checkout on the box:
#
#   git -c core.autocrlf=false archive HEAD | ssh HOST 'rm -rf ~/ci/recall; mkdir -p ~/ci/recall && tar -x -C ~/ci/recall'
#   ssh HOST 'cd ~/ci/recall && sh scripts/ci-local.sh'
#
# Steps: build | test | ops (cargo test --features ops-health) | pytest | scrub | smoke (README install + first use) | demo (examples/stale-demo). Subset: CI_ONLY="test pytest".
# Installs ripgrep into ~/.local/bin (SHA-256 pinned) when missing. The whole run uses a TEMP home
# (HOME and USERPROFILE, no inherited RECALL_* variables), so the real ~/.recall is never read or
# written; cargo/rustup homes are pinned first. Builds into ./target-ci, niced, -j ${CI_JOBS:-2};
# one run per checkout at a time (a lock refuses a second).
# Prints PASS/FAIL per step; exits 1 if any failed or any requested step did not run.
# Windows: scripts/ci-local.ps1.
cd "$(dirname "$0")/.." || exit 2
ROOT=$(pwd)
JOBS=${CI_JOBS:-2}
ALL="build test ops pytest scrub smoke demo"
ONLY=${CI_ONLY:-$ALL}
for n in $ONLY; do case " $ALL " in *" $n "*) ;; *) echo "unknown step '$n' (known: $ALL)"; exit 2 ;; esac; done
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export CARGO_HOME=${CARGO_HOME:-$HOME/.cargo} RUSTUP_HOME=${RUSTUP_HOME:-$HOME/.rustup}
export CARGO_TARGET_DIR="$ROOT/target-ci"
LOCK="$ROOT/.ci-local.lock"
mkdir "$LOCK" 2>/dev/null || { echo "another ci-local run holds $LOCK (delete it if that run is dead)"; exit 2; }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP" "$LOCK"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
RESULTS=""
BAD=0

if ! command -v rg >/dev/null 2>&1; then
  v=14.1.1; a=ripgrep-$v-x86_64-unknown-linux-musl
  sha=4cf9f2741e6c465ffdb7c26f38056a59e2a2544b51f7cc128ef28337eeae4d8e
  mkdir -p "$HOME/.local/bin"
  curl -fsSL -o "$TMP/rg.tgz" "https://github.com/BurntSushi/ripgrep/releases/download/$v/$a.tar.gz" || exit 2
  echo "$sha  $TMP/rg.tgz" | sha256sum -c - >/dev/null || { echo "ripgrep checksum mismatch"; exit 2; }
  tar -xzf "$TMP/rg.tgz" -C "$TMP" "$a/rg" && mv "$TMP/$a/rg" "$HOME/.local/bin/rg" || exit 2
fi
rg --version | head -1

# From here on everything runs against a temp home.
for v in $(env | sed -n 's/^\(RECALL_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
unset PYTEST_ADDOPTS PYTEST_PLUGINS
export HOME="$TMP/home" USERPROFILE="$TMP/home" RECALL_ERROR_LOG="$TMP/errors.log"
# git's global safe.directory lived in the real home; keep repos under the real home usable.
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'
mkdir -p "$HOME"
if python3 -m venv "$TMP/venv" && "$TMP/venv/bin/python" -m pip install -q -r mcp/requirements.txt pytest; then
  PY="$TMP/venv/bin/python"
else
  printf "\n==== ci-local summary ====\nFAIL     setup\n"; exit 1
fi

step() { # step <name> <command...>
  name=$1; shift
  case " $ONLY " in *" $name "*) ;; *) return ;; esac
  echo "==> $name"
  if "$@"; then RESULTS="$RESULTS\nPASS     $name"; else RESULTS="$RESULTS\nFAIL     $name"; BAD=1; fi
}
smoke() { sh scripts/install.sh --vault "$HOME/notes" && python3 scripts/clean_install_smoke.py; }

step build  nice -n 19 cargo build --release -j "$JOBS"
step test   nice -n 19 cargo test --release -j "$JOBS" -- --test-threads=2
step ops    nice -n 19 cargo test --release -j "$JOBS" --features ops-health -- --test-threads=2
step pytest "$PY" -m pytest ingest integrations/claude-code/test_hooks.py mcp/test_recall_mcp.py scripts/test_install_deps.py scripts/test_clean_install_smoke.py -q
step scrub  "$PY" -I scripts/scrub_e2e.py "$CARGO_TARGET_DIR/release/recall"
step smoke  smoke
step demo   sh examples/stale-demo/run.sh "$CARGO_TARGET_DIR/release/recall"

for n in $ONLY; do case "$RESULTS" in *" $n"*) ;; *) RESULTS="$RESULTS\nNOT-RUN  $n"; BAD=1 ;; esac; done
printf "\n==== ci-local summary ====$RESULTS\n"
exit $BAD
