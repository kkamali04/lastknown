#!/bin/sh
# Stale-memory demo: recall tells you when a result is a copy of a file that has since changed,
# and tells a changed FILE whose shown PASSAGE is untouched ("this passage unchanged") from a
# changed PASSAGE ("this passage changed: read the file").
#
#   sh examples/stale-demo/run.sh [path/to/recall]      (default: $RECALL_BIN, else `recall` on PATH)
#
# Needs only the recall binary and ripgrep (`rg`) on PATH. Keyword channel only (-c fts5), so no model download.
# Everything happens in a temp HOME and a temp vault; your real ~/.recall is never read or written.
# Exit 0 = every check passed, 1 = a check failed, 2 = could not run (missing binary / ripgrep).
BIN=${1:-${RECALL_BIN:-recall}}
command -v "$BIN" >/dev/null 2>&1 || [ -x "$BIN" ] || { echo "recall binary not found: $BIN"; exit 2; }
command -v rg >/dev/null 2>&1 || { echo "ripgrep (rg) not found on PATH; recall index needs it"; exit 2; }
TMP=$(mktemp -d) || exit 2
trap 'rm -rf "$TMP"' EXIT
for v in $(env | sed -n 's/^\(RECALL_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
export HOME="$TMP/home" USERPROFILE="$TMP/home"
mkdir -p "$HOME" "$TMP/vault"
NOTE="$TMP/vault/api-notes.md"
BAD=0

check() { # check <description> <command...>
  desc=$1; shift
  if "$@"; then echo "  PASS  $desc"; else echo "  FAIL  $desc"; BAD=1; fi
}
has() { printf '%s\n' "$OUT" | grep -q -- "$1"; }
lacks() { ! printf '%s\n' "$OUT" | grep -q -- "$1"; }
ask() { OUT=$("$BIN" query -c fts5 "API port" 2>&1); RC=$?; printf '%s\n' "$OUT"; check "query exits 0" test "$RC" -eq 0; }
reindex() { "$BIN" index > "$TMP/index.out" 2>&1; RC=$?; grep -E "Indexed|docs in index" "$TMP/index.out"; check "index exits 0" test "$RC" -eq 0; }

echo "== 1. write a note and index it"
write_note() { # write_note <intro line> <port>
  printf '# API notes\n\n%s\n\nThe API port is %s\n\nQuestions go to the platform channel.\n' "$1" "$2" > "$NOTE"
}
write_note "Kept by the platform team." 8080
"$BIN" settings set vault "$TMP/vault" >/dev/null 2>&1
check "settings set vault exits 0" test $? -eq 0
reindex
echo
echo "== 2. query: fresh, the file matches the index"
ask
check "finds the note with 8080" has "8080"
check "no stale warning" lacks "STALE"

echo
echo "== 3. edit ANOTHER part of the file (the intro line) WITHOUT re-indexing, then query"
sleep 1
write_note "Kept by the infrastructure team, since March." 8080
ask
check "not flagged STALE: the passage shown is still in the file" lacks "STALE"
check "says the file changed but this passage is unchanged" has "this passage unchanged"
check "still shows 8080" has "8080"

echo
echo "== 4. edit the PASSAGE itself (8080 -> 9090) WITHOUT re-indexing, then query again"
sleep 1
write_note "Kept by the infrastructure team, since March." 9090
ask
check "hit is flagged STALE" has "STALE"
check "says this passage changed" has "this passage changed"
check "names the file" has "api-notes.md"
check "tells you to read the real file" has "read the actual file"
check "the flagged hit still shows the old indexed text (8080), not a silent rewrite" has "8080"

echo
echo "== 5. re-index, then query: the fresh answer"
reindex
ask
check "fresh answer is 9090" has "9090"
check "old 8080 text is gone" lacks "8080"
check "no stale warning" lacks "STALE"

echo
if [ "$BAD" -eq 0 ]; then echo "stale-demo: PASS"; else echo "stale-demo: FAIL"; fi
exit "$BAD"
