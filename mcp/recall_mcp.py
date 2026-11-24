"""Recall MCP Server — minimal, portable, read-only.

Five tools over a local `recall` index: search, semantic search, recent, identity and health.
This is the supported portable stdio server.

Design rules, each earned the hard way on this stack:

  * SAFE ARGV — flags first, then `--`, then exactly one positional. A user
    string equal to a flag name (`--no-daemon`) would otherwise be parsed as
    that flag and force a ~59 s cold load from a plain search box.
  * NO HARDCODED PATHS — everything resolves from env with sane fallbacks.
  * FAIL LOUD — a dead backend must raise, never return an empty list. The
    fts5 rescue tier means "no results" and "the semantic channel is down"
    look identical from outside; that ambiguity has cost days here.
  * READ-ONLY — no index, no embed, no gc, no save-to-store. Indexing is a
    vault-owner operation, not something a reading agent does.
  * TRUST `rank`, NEVER `score` — fused scores are not calibrated confidence
    and are not comparable across channels or queries.
  * TENANCY — no tool takes an agent, project or scope argument. The boxes this
    server sees come only from the RECALL_AGENT / RECALL_PROJECT / RECALL_TEAM
    env it was launched with, inherited by every `recall` subprocess.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
from typing import Any

from fastmcp import FastMCP

mcp = FastMCP("recall")

# --------------------------------------------------------------------------
# Configuration — env first, then a discoverable default. Nothing machine-specific.
# --------------------------------------------------------------------------

DEFAULT_TIMEOUT_S = float(os.environ.get("RECALL_MCP_TIMEOUT_S", "90"))
MAX_TOP = 50


class RecallUnavailable(RuntimeError):
    """Raised when recall cannot be found or run. Never swallowed."""


def _recall_exe() -> str:
    """Locate the recall binary. Env override wins; then the installer's dir; then the repo build; then PATH."""
    explicit = os.environ.get("RECALL_EXE")
    if explicit:
        if not pathlib.Path(explicit).is_file():
            raise RecallUnavailable(f"RECALL_EXE points at a missing file: {explicit}")
        return explicit

    # Prefer the real binary over a shell wrapper: on Windows a `recall.cmd` shim on PATH
    # costs ~13 s per call via cmd.exe versus ~80 ms for the .exe directly.
    exe_name = "recall.exe" if os.name == "nt" else "recall"
    home = pathlib.Path(os.environ.get("USERPROFILE") or os.environ.get("HOME") or ".")
    candidates = [
        home / ".recall" / "bin" / exe_name,  # where scripts/install.* puts it
        pathlib.Path(__file__).resolve().parent.parent / "target" / "release" / exe_name,
    ]
    for c in candidates:
        if c.is_file():
            return str(c)

    on_path = shutil.which(exe_name) or shutil.which("recall")
    if on_path:
        return on_path

    raise RecallUnavailable(
        "recall binary not found. Run scripts/install.sh (or install.ps1), set RECALL_EXE, "
        "or build it with: cargo build --release"
    )


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    exe = _recall_exe()
    try:
        return subprocess.run(
            [exe, *args],
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUT_S,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        raise RecallUnavailable(
            f"recall timed out after {DEFAULT_TIMEOUT_S:.0f}s. A cold daemon can take ~59s "
            f"on first query; raise RECALL_MCP_TIMEOUT_S or start it with `recall daemon up`."
        ) from exc


def _query(text: str, top: int, channels: str) -> list[dict[str, Any]]:
    """Run one query. Flags FIRST, then `--`, then exactly one positional."""
    if not text or not text.strip():
        raise ValueError("query text must not be empty")
    top = max(1, min(int(top), MAX_TOP))

    proc = _run(["query", "--top", str(top), "--channels", channels, "--json", "--", text])

    if proc.returncode != 0:
        raise RecallUnavailable(
            f"recall exited {proc.returncode} for channels={channels!r}.\n"
            f"stderr: {(proc.stderr or '').strip()[:600]}"
        )

    raw = (proc.stdout or "").strip()
    if not raw:
        raise RecallUnavailable(
            f"recall returned an EMPTY body with exit 0 for channels={channels!r}. "
            f"That is a wiring fault, not a 'no results' answer.\n"
            f"stderr: {(proc.stderr or '').strip()[:600]}"
        )

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RecallUnavailable(f"recall emitted non-JSON output: {raw[:300]!r}") from exc

    hits = parsed if isinstance(parsed, list) else (parsed.get("results") or [])

    # A dead cloud channel is masked by the fts5 rescue tier: results still
    # arrive and exit stays 0. Surface it rather than let it pass as semantic.
    stderr = (proc.stderr or "").lower()
    degraded = None
    for marker in ("403", "401", "429", "vertex", "insufficient balance", "onnx runtime",
                   "rescue", "no vectors stored", "not compiled", "unknown channel"):
        if marker in stderr:
            degraded = (proc.stderr or "").strip()[:300]
            break

    # Backend health must not depend on the result count: a degraded channel that returned
    # nothing is a failure, not "the search ran correctly and found nothing".
    if degraded and not hits:
        raise RecallUnavailable(
            f"a channel reported an error and returned no results for channels={channels!r}: "
            f"{degraded}"
        )
    for h in hits:
        if degraded:
            h["_channel_warning"] = f"a channel reported an error; results may be keyword-only: {degraded}"
    return hits


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


@mcp.tool()
def recall_search(query: str, top: int = 8, channels: str = "fts5,bge") -> str:
    """Search the local recall memory (your indexed notes, fully offline).

    Use this BEFORE answering anything about the user's projects, decisions,
    infrastructure, or past work — it is the record, and it is faster and
    more accurate than guessing or grepping the notes folder.

    Args:
        query: natural-language question or keywords.
        top: how many results (1-50). 8 is a good default.
        channels: "fts5,bge" (default, hybrid), "bge" (semantic only, ~13x
            faster), or "fts5" (keyword only).

    Returns a ranked list. TRUST `rank`; never threshold on `score`.
    """
    hits = _query(query, top, channels)
    if not hits:
        return f"No matches for {query!r} on channels={channels!r}. The search ran correctly and found nothing."

    lines = [f"{len(hits)} results for {query!r} (channels={channels}):", ""]
    if any(h.get("labels_truncated") for h in hits):
        lines[-1:] = ["NOTE: the query has more than 64 subject words; only the first 64 were checked.", ""]
    # A hit missing a query subject term is a closest match, never a plain answer (K8).
    if all(h.get("lacks") for h in hits):
        lines[-1:] = ["WARNING: no result mentions every subject term of the query; "
                      "these are closest matches, not answers.", ""]
    stale_n = sum(1 for h in hits if h.get("stale") is True)
    if stale_n:
        lines[-1:] = [f"WARNING: {stale_n} of {len(hits)} results point at files that changed on disk after recall "
                      "indexed them; read the actual files, not these excerpts.", ""]
    for h in hits:
        warn = h.get("_channel_warning")
        lines.append(f"[{h.get('rank')}] {h.get('title') or '(untitled)'}")
        if h.get("lacks"):
            lines.append(f"    CLOSEST MATCH - does not mention: {', '.join(h['lacks'])}")
        lines.append(f"    {h.get('vault_path')}")
        # stale: true = changed/deleted since indexing; stale: null = could not check (never "fresh").
        if h.get("stale") is True:
            lines.append(f"    !! STALE: {h.get('stale_note') or 'this file changed after recall indexed it; read the actual file.'}")
        elif h.get("stale") is False and h.get("stale_note"):
            lines.append(f"    ~ {h['stale_note']}")
        elif h.get("stale") is None and h.get("stale_note"):
            lines.append(f"    ? {h['stale_note']}")
        snip = (h.get("snippet") or "").replace("\n", " ").strip()
        if snip:
            lines.append(f"    {snip[:300]}")
        if warn:
            lines.append(f"    !! {warn}")
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def recall_search_semantic(query: str, top: int = 8) -> str:
    """Semantic-only search — finds notes by MEANING, not shared words.

    Use when the answer is phrased differently from the question ("how do I
    keep agents from stepping on each other" -> a note titled "worktree
    isolation"). Roughly 13x faster than the hybrid search.
    """
    return recall_search(query=query, top=top, channels="bge")


@mcp.tool()
def recall_recent(count: int = 10) -> str:
    """List the most recently saved memories (newest first). Read-only."""
    count = max(1, min(int(count), MAX_TOP))
    proc = _run(["recent", "--count", str(count)])
    if proc.returncode != 0:
        raise RecallUnavailable(
            f"recall exited {proc.returncode}: {(proc.stderr or '').strip()[:600]}"
        )
    return (proc.stdout or "").strip() or "No saved memories yet."


@mcp.tool()
def recall_whoami() -> str:
    """Which memory boxes this server reads and writes as (tenancy).

    The identity is NOT a tool argument: it is the RECALL_AGENT / RECALL_PROJECT / RECALL_TEAM
    environment this MCP server was launched with, passed straight through to recall. No identity
    means the operator (every box).
    """
    proc = _run(["whoami"])
    if proc.returncode != 0:
        raise RecallUnavailable(
            f"recall exited {proc.returncode}: {(proc.stderr or '').strip()[:600]}"
        )
    return (proc.stdout or "").strip()


@mcp.tool()
def recall_health() -> str:
    """Report whether recall is actually working, and which channels answer.

    Run this when results look wrong or empty. It distinguishes "nothing
    matched" from "a channel is down and the keyword tier is covering for it".
    """
    out = [f"binary: {_recall_exe()}"]

    status = _run(["status"])
    out.append(f"status exit={status.returncode}")
    if status.stdout:
        out.append(status.stdout.strip()[:1200])
    if status.stderr.strip():
        out.append(f"stderr: {status.stderr.strip()[:400]}")

    out.append("")
    out.append("per-channel probe:")
    for ch in ("fts5", "bge", "fts5,bge"):
        try:
            hits = _query("notes", 3, ch)
        except Exception as exc:  # noqa: BLE001 - report, never hide
            out.append(f"  {ch:<10} FAIL {exc}")
            continue
        # Results existing proves nothing: the fts5 rescue tier answers for a dead channel.
        # `source` names the channel that actually ranked each hit.
        warned = [h["_channel_warning"] for h in hits if h.get("_channel_warning")]
        wrong = [h.get("source") for h in hits if ch == "bge" and "bge" not in str(h.get("source"))]
        if warned:
            out.append(f"  {ch:<10} FAIL {warned[0]}")
        elif wrong:
            out.append(f"  {ch:<10} FAIL bge probe was answered by source={wrong[0]!r}, not bge")
        else:
            out.append(f"  {ch:<10} OK   {len(hits)} hits")
    return "\n".join(out)


if __name__ == "__main__":
    mcp.run()
