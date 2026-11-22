"""memcap.py — capture Claude Code sessions into dated vault notes that recall indexes.

Sources (both read-only):
  * Claude Code transcripts  <config>/projects/*/<uuid>.jsonl, */<uuid>/subagents/agent-*.jsonl,
                             */<uuid>/subagents/workflows/<wf>/agent-*.jsonl
  * claude-mem's SQLite DB   ~/.claude-mem/claude-mem.db (observations + session_summaries)

Output: <capture dir>/YYYY-MM-DD/<identity>[.pN].md in the vault (a dot-folder:
Obsidian neither shows nor syncs it, recall's tree walk skips it, memcap indexes it explicitly),
one group per (session identity, UTC day). Every string is scrubbed by ingest/scrub_turns.scrub_text
per field, again per part, and the result must be a fixed point of the scrubber. Then
`recall.exe index --path-list` and a read-only readback that recall.db holds exactly the new body.
Design and the review history: PLAN.md.

Exit codes: 0 ok/skip · 2 preflight refused (nothing written) · 3 partial (see memcap-health.json).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath, PureWindowsPath

MEMCAP_VERSION = 6
PART_BUDGET = 300_000          # bytes per part (frontmatter + heading reserved separately)
PART_HARD_MAX = 400_000        # asserted on the final bytes; recall skips files > 512 KB
# vault-relative capture folder; RECALL_CAPTURE_REL overrides (a dot-folder keeps it out of note listings)
def capture_rel(raw: str | None) -> str:
    """The capture folder, vault-relative and canonical. A rooted or `..` value is refused: it would
    escape the vault (and a dry run's --out) instead of landing inside it."""
    parts = [p for p in (raw or ".memory-capture").replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or ":" in parts[0] or (raw or "").replace("\\", "/").startswith("/") or ".." in parts:
        raise ValueError(f"RECALL_CAPTURE_REL must be a folder inside the vault, got {raw!r}")
    return "/".join(parts)


CAPTURE_REL = capture_rel(os.environ.get("RECALL_CAPTURE_REL"))
NIGHTLY_TASK = "Recall-Nightly-Index"   # writes recall.db; memcap must not overlap it
CHECKPOINT_EVERY = 25          # groups between durable state/journal checkpoints
LABEL_RE = re.compile(r"<REDACTED:([a-z0-9_-]+):")
REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
KNOWN_BLOCKS = {"text", "tool_use", "tool_result", "thinking", "redacted_thinking", "image",
                "document", "server_tool_use", "web_search_tool_result"}
HOME = Path.home()

OBS_COLS = ("id", "memory_session_id", "type", "title", "subtitle", "facts", "narrative",
            "files_modified", "created_at_epoch")
SUM_COLS = ("id", "memory_session_id", "request", "learned", "completed", "next_steps",
            "created_at_epoch")
SES_COLS = ("content_session_id", "memory_session_id", "project")


SEG_RE = re.compile(r"[a-z]+\d*|[A-Z]+\d*|[A-Z][a-z]+\d*|\d{1,4}")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
# Claude Code agent ids are "a" + optional chosen name + "-" + 16-17 hex; only the hex is used
# in filenames/state/reports, so a credential-shaped name can never reach an artifact.
AGENT_RE = re.compile(r"^a(?:[A-Za-z0-9_-]{1,64}-)?([0-9a-f]{16,17})$")
WF_RE = re.compile(r"^wf_[0-9a-f]{8}-[0-9a-f]{3}$")



def cwd_name(cwd: str) -> str:
    """Last component of a transcript's cwd on either OS: a transcript written on Windows
    (C:\\work\\demo-proj) is read on Linux too, where Path() would not split on backslashes."""
    windows = re.match(r"^([A-Za-z]:[\\/]|\\\\)", cwd)
    return (PureWindowsPath(cwd) if windows else PurePosixPath(cwd)).name

def keep_token(tok: str) -> bool:
    """Entropy-stage exemption for vault paths and dated doc names (scrub_text(keep=...)).

    Measured on a real session: 10 of 12 entropy redactions were paths such as
    `research/infra/JANITOR-LOG-2026-11-01` - the thing agents ask recall for. Exempt only if the
    token has '/' or a YYYY-MM-DD date AND every segment (split on / - _ .) is a plain word:
    lower, UPPER or Capitalised letters (+ trailing digits) or <= 4 digits. Applies to the entropy
    stage ONLY: every named credential/PII rule has already run on the token.
    ponytail: a secret made of plain words joined by '/' passes the entropy stage; no named
    rule would have caught it either.
    """
    if "/" not in tok and not re.search(r"\d{4}-\d{2}-\d{2}", tok):
        return False
    segs = [x for x in re.split(r"[/\-_.]", tok) if x]
    return len(segs) >= 2 and all(len(x) <= 24 and SEG_RE.fullmatch(x) for x in segs)


class Refuse(Exception):
    """Preflight failure: exit 2, write nothing."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".memcap-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        for attempt in range(8):          # Windows: AV/indexer can hold the target for a moment
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.05 * (attempt + 1))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_json(path: Path, obj) -> None:
    atomic_write(path, json.dumps(obj, indent=1, sort_keys=True).encode("utf-8"))


def load_json(p: Path) -> dict:
    """Absent = first run ({}). Unreadable or corrupt = Refuse: never treat it as empty."""
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        raise Refuse(f"cannot read {p.name}: {type(e).__name__}")
    if not isinstance(obj, dict):
        raise Refuse(f"{p.name} is not a JSON object")
    return obj


def read_lines(p: Path) -> list[str]:
    """Absent = no entries. Unreadable = Refuse (an unreadable exclude list must not re-enable)."""
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as e:
        raise Refuse(f"cannot read {p.name}: {type(e).__name__}")
    return [l.strip() for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]


def utc(ts_ms: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(ts_ms / 1000, tz=dt.timezone.utc)


def parse_ts(s) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
    except (ValueError, AttributeError):
        return None


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git_exposure(path: Path) -> list[str]:
    """Every git work tree on path's ancestry where `path` is not ignored or has tracked files."""
    problems = []
    for anc in [path, *path.parents]:
        if not (anc / ".git").exists():
            continue
        rel = path.relative_to(anc).as_posix()
        probe = f"{rel}/2000-01-01/probe.md" if rel != "." else "probe.md"
        if subprocess.run(["git", "-C", str(anc), "check-ignore", "-q", probe]).returncode != 0:
            problems.append(f"{anc}: {rel}/ is not git-ignored")
        tracked = subprocess.run(["git", "-C", str(anc), "ls-files", "--", rel],
                                 capture_output=True, text=True).stdout.strip()
        if tracked:
            problems.append(f"{anc}: {len(tracked.splitlines())} tracked file(s) under {rel}/")
    return problems


def under(path: Path, root_real: str) -> bool:
    return os.path.realpath(path).startswith(root_real + os.sep)


# ---------------------------------------------------------------------------
# transcripts
# ---------------------------------------------------------------------------
def identity_for(path: Path, projects: Path) -> str | None:
    """Session = a real UUID; agent = its opaque hex suffix (AGENT_RE); workflow = WF_RE."""
    parts = path.relative_to(projects).parts
    if len(parts) == 2:                                                   # <proj>/<uuid>.jsonl
        session, rest = path.stem, []
    elif len(parts) == 4 and parts[2] == "subagents":                     # .../subagents/agent-x
        session, rest = parts[1], [path.stem.removeprefix("agent-")]
        m = AGENT_RE.match(rest[0])
        rest = [m.group(1)] if m else ["?"]
    elif len(parts) == 6 and parts[2:4] == ("subagents", "workflows"):    # .../workflows/<wf>/agent-x
        m = AGENT_RE.match(path.stem.removeprefix("agent-"))
        session, rest = parts[1], [parts[4] if WF_RE.match(parts[4]) else "?", m.group(1) if m else "?"]
    else:
        return None
    if not UUID_RE.match(session) or "?" in rest:
        return None
    return "__".join([session, *rest])


def inventory(projects: Path) -> tuple[dict[str, Path], dict]:
    """identity -> transcript. Same identity in two project dirs = a conflict, never a silent pick."""
    out, conflicts, skipped = {}, set(), 0
    for pat in ("*/*.jsonl", "*/*/subagents/*.jsonl", "*/*/subagents/workflows/*/*.jsonl"):
        for p in projects.glob(pat):
            ident = identity_for(p, projects)
            if not ident:
                skipped += 1
            elif ident in out:
                conflicts.add(ident)
            else:
                out[ident] = p
    for ident in conflicts:
        out.pop(ident, None)
    return out, {"conflicts": sorted(conflicts), "unrecognised_files": skipped}


def read_transcript(path: Path, text_from_content) -> dict:
    """Stream one transcript up to its PRE-READ size. Only user/assistant TEXT survives."""
    st = path.stat()
    entries, seen = [], set()
    meta = {"title": "", "custom_title": "", "agent_name": "", "cwd": "", "unparsed": 0,
            "unknown_blocks": 0, "trailing_partial": False, "stat": [st.st_size, st.st_mtime_ns]}
    read = 0
    pending_bad = False
    with open(path, "rb") as fh:
        for raw in fh:
            if read + len(raw) > st.st_size:          # grew while reading: stop at the boundary
                meta["trailing_partial"] = True
                break
            read += len(raw)
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if pending_bad:                            # an earlier bad line was interior after all
                meta["unparsed"] += 1
                pending_bad = False
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                pending_bad = True
                continue
            if not isinstance(obj, dict):
                meta["unparsed"] += 1
                continue
            rtype = obj.get("type")
            if not meta["cwd"] and isinstance(obj.get("cwd"), str):
                meta["cwd"] = obj["cwd"]
            if rtype == "ai-title" and isinstance(obj.get("aiTitle"), str):
                meta["title"] = obj["aiTitle"]
            elif rtype == "custom-title":
                meta["custom_title"] = obj.get("customTitle") or obj.get("title") or meta["custom_title"]
            elif rtype == "agent-name" and isinstance(obj.get("agentName"), str):
                meta["agent_name"] = obj["agentName"]
            if rtype not in ("user", "assistant") or obj.get("isMeta"):
                continue
            msg = obj.get("message")
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                texts = [content]
            elif isinstance(content, list):
                texts = []
                for b in content:
                    btype = b.get("type") if isinstance(b, dict) else None
                    if btype == "text" and not isinstance(b.get("text"), str):
                        meta["unparsed"] += 1
                    elif btype == "text":
                        # the recall extractor's helper: top-level text blocks only, never nested
                        texts.append(text_from_content([b]))
                    elif btype not in KNOWN_BLOCKS:
                        meta["unknown_blocks"] += 1
            else:
                meta["unparsed"] += 1
                continue
            for bi, text in enumerate(texts):
                if rtype == "user":
                    if text.lstrip().startswith(("<command-", "<local-command")):
                        continue
                    text = REMINDER_RE.sub("", text)
                text = text.strip()
                key = (obj.get("uuid"), bi, text)
                if not text or key in seen:
                    continue
                seen.add(key)
                ts = parse_ts(obj.get("timestamp"))
                if ts is None:
                    meta["unparsed"] += 1
                    continue
                entries.append((ts, rtype, text.encode("utf-8", "replace").decode("utf-8")))
    if pending_bad:
        meta["trailing_partial"] = True
    meta["entries"] = entries
    return meta


# ---------------------------------------------------------------------------
# claude-mem: one snapshot, content-hash fingerprints
# ---------------------------------------------------------------------------
def read_mem(db: Path) -> dict:
    res = {"state": "ok", "fingerprints": {}, "rows": {}, "projects": {}, "unmatched": 0,
           "error": "", "db_id": ""}
    if not db.exists():
        res["state"] = "missing"
        return res
    try:
        con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=5)
        con.execute("PRAGMA busy_timeout=5000")
        for table, need in (("observations", OBS_COLS), ("session_summaries", SUM_COLS),
                            ("sdk_sessions", SES_COLS)):
            have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
            if not set(need) <= have:
                res["state"], res["error"] = "schema-unsupported", f"{table} lacks {sorted(set(need) - have)}"
                return res
        con.execute("BEGIN")
        sessions = {ms: (cs, pj) for cs, ms, pj in con.execute(
            "SELECT content_session_id, memory_session_id, project FROM sdk_sessions "
            "WHERE memory_session_id IS NOT NULL")}
        hashes: dict[str, "hashlib._Hash"] = {}
        for kind, table, cols in (("obs", "observations", OBS_COLS),
                                  ("sum", "session_summaries", SUM_COLS)):
            for row in con.execute(f"SELECT {','.join(cols)} FROM {table} "
                                   f"ORDER BY memory_session_id, created_at_epoch, id"):
                hit = sessions.get(row[1])
                if not hit or not UUID_RE.match(hit[0] or ""):
                    res["unmatched"] += 1
                    continue
                cs = hit[0]
                rec = (kind, row[0]) + row[2:]
                res["rows"].setdefault(cs, []).append(rec)
                res["projects"][cs] = hit[1]
                hashes.setdefault(cs, hashlib.sha256()).update(
                    json.dumps([hit[1], rec], default=str).encode())
        res["db_id"] = str(con.execute("PRAGMA schema_version").fetchone()[0])
        con.execute("COMMIT")
        con.close()
        res["fingerprints"] = {k: h.hexdigest()[:24] for k, h in hashes.items()}
    except sqlite3.Error as e:
        res["state"], res["error"] = "error", type(e).__name__
        res["rows"], res["fingerprints"] = {}, {}
    return res


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def split_text(text: str, budget: int) -> list[str]:
    """Cut an over-budget string at line boundaries, char-safely, never dropping anything."""
    if len(text.encode("utf-8")) <= budget:
        return [text]
    out, cur, size = [], [], 0
    for line in text.splitlines(keepends=True):
        while len(line.encode("utf-8")) > budget:      # one enormous line: cut by characters
            if cur:
                out.append("".join(cur)); cur, size = [], 0
            cut = budget // 4                           # <= budget bytes even at 4 bytes/char
            out.append(line[:cut]); line = line[cut:]
        b = len(line.encode("utf-8"))
        if size + b > budget and cur:
            out.append("".join(cur)); cur, size = [], 0
        cur.append(line); size += b
    if cur:
        out.append("".join(cur))
    return out


def render_mem_block(row, scrub) -> str:
    if row[0] == "obs":
        _, oid, otype, title, subtitle, facts, narrative, files, ts = row
        lines = [f"### [{scrub(str(otype or ''))}] {scrub(title or '(untitled)')} — obs #{oid} · "
                 f"{utc(ts):%Y-%m-%d %H:%MZ}"]
        if subtitle:
            lines.append(scrub(subtitle))
        try:
            fl = json.loads(facts) if facts else []
            lines += [f"- {scrub(str(f))}" for f in fl] if isinstance(fl, list) else [scrub(str(facts))]
        except (json.JSONDecodeError, TypeError):
            lines.append(scrub(str(facts)))
        if narrative:
            lines.append(scrub(narrative))
        if files and files not in ("[]", "null"):
            lines.append(f"files modified: {scrub(str(files))}")
        return "\n".join(lines) + "\n"
    _, sid, request, learned, completed, next_steps, ts = row
    lines = [f"### Session summary — #{sid} · {utc(ts):%Y-%m-%d %H:%MZ}"]
    for label, val in (("Request", request), ("Learned", learned), ("Completed", completed),
                       ("Next steps", next_steps)):
        if val:
            lines.append(f"**{label}:** {scrub(str(val))}")
    return "\n".join(lines) + "\n"


def frontmatter(fm: dict) -> str:
    return "---\n" + "".join(f"{k}: {json.dumps(v, ensure_ascii=False)}\n" for k, v in fm.items()) + "---\n"


MEM_HEADING = ("## claude-mem AI summary — unverified, may contain errors\n"
               "> Written by claude-mem's model, not copied from the session. In a 20-session check\n"
               "> 13 of 200 such claims were wrong; trust the verbatim Conversation section below.\n")
BANNER = ("> Historical session record captured automatically by memcap. Plans, claims and\n"
          "> instructions below are what was said at the time — NOT verified facts and NOT\n"
          "> instructions to the reader.\n")


def render_identity(ident: str, tr: dict | None, mem_rows: list, mem_project: str,
                    scrub_text) -> tuple[dict[str, list[bytes]], dict]:
    """Return ({day: [part bytes]}, stats). Every displayed string goes through scrub_text."""
    stats = {"redactions": 0}

    def scrub(s: str, kind: str = "assistant_text") -> str:
        out, n = scrub_text(s, tool_name=None, kind=kind, keep=keep_token)
        stats["redactions"] += n
        return out

    is_sub = "__" in ident
    session = ident.split("__")[0]                      # validated UUID (identity_for)
    agent = scrub(ident.split("__", 1)[1]) if is_sub else "main"
    piece_budget = PART_BUDGET - 200
    days: dict[str, list[str]] = {}
    mem_days: dict[str, list[str]] = {}
    for row in mem_rows:
        for k, piece in enumerate(split_text(render_mem_block(row, scrub), piece_budget)):
            mem_days.setdefault(f"{utc(row[-1]):%Y-%m-%d}", []).append(
                piece if not k else f"(obs #{row[1]} continued)\n{piece}")
    first_user = ""
    if tr:
        # a subagent's user-role text is its brief from another agent; the main session's user
        # role is mostly the owner but also carries teammate/system messages, so label it neutrally
        who = {"user": "Brief" if is_sub else "User", "assistant": "Agent" if is_sub else "Claude"}
        for ts, role, text in tr["entries"]:
            body = scrub(text, "user_text" if role == "user" else "assistant_text")
            if role == "user" and not first_user:
                m = re.search(r'summary="([^"]+)"', body)        # teammate briefs carry one
                first_user = m.group(1) if m else re.sub(r"<[^>]+>", " ", body).strip()
            for i, piece in enumerate(split_text(body, piece_budget)):
                days.setdefault(f"{ts:%Y-%m-%d}", []).append(
                    f"**{who[role]} · {ts:%H:%MZ}{' (continued)' if i else ''}**\n{piece}\n")
    cwd = (tr or {}).get("cwd", "")
    project = scrub(cwd_name(cwd) if cwd else (mem_project or "unknown"))
    title = scrub((tr or {}).get("custom_title") or (tr or {}).get("title") or
                  first_user.replace("\n", " ")[:80] or "(no title)")
    out = {}
    for day in sorted(set(days) | set(mem_days)):
        blocks = []
        if mem_days.get(day):
            blocks += [MEM_HEADING] + mem_days[day]
        if days.get(day):
            blocks += ["## Conversation\n"] + days[day]
        parts, cur, size = [], [], 0
        for b in blocks:
            bl = len(b.encode("utf-8")) + 1
            if cur and size + bl > PART_BUDGET:
                parts.append(cur); cur, size = [], 0
            cur.append(b); size += bl
        if cur:
            parts.append(cur)
        rendered = []
        for i, p in enumerate(parts, 1):
            fm = {"type": "session-capture", "date": day, "session": session, "agent": agent,
                  "project": project, "title": title, "part": i, "parts": len(parts),
                  "sources": [s for s, on in (("claude-code-transcript", bool(tr)),
                                              ("claude-mem", bool(mem_rows))) if on],
                  "unparsed": (tr or {}).get("unparsed", 0),
                  "unknown_blocks": (tr or {}).get("unknown_blocks", 0),
                  "memcap_version": MEMCAP_VERSION}
            head = f"# {day} · {project} · {title}" + (f" (part {i}/{len(parts)})" if len(parts) > 1 else "")
            rendered.append((frontmatter(fm) + head + "\n\n" + BANNER + "\n" + "\n".join(p)).encode("utf-8", "replace"))
        out[day] = rendered
    return out, stats


def part_path(ident: str, day: str, i: int) -> str:
    return f"{CAPTURE_REL}/{day}/{ident}{'' if i == 1 else f'.p{i}'}.md"


def split_front(data: bytes) -> tuple[str, str]:
    text = data.decode("utf-8")
    end = text.index("\n---\n") + 5
    return text[:end], text[end:]


# ---------------------------------------------------------------------------
# indexing
# ---------------------------------------------------------------------------
def stored_form(recall_exe: str, raw: bytes) -> str | None:
    """`recall stored-form`: the text the index stores for these file bytes. None if recall cannot
    answer (old binary, crash): the caller then treats the file as unconfirmed, never as ok."""
    try:
        r = subprocess.run([recall_exe, "stored-form"], input=raw, capture_output=True, timeout=120)
        return r.stdout.decode("utf-8") if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        return None


def _stored_matches(recall_exe: str, raw: bytes, body) -> bool:
    expected = stored_form(recall_exe, raw)
    return expected is not None and body == expected     # a failed helper (None) never confirms, even a NULL body


def index_and_verify(a, vault: Path, journal_path: Path, paths: list[str], owned_paths: set[str]) -> dict:
    """recall index --path-list, then confirm recall.db holds each file's CURRENT text as live.

    A journaled path whose file is missing is NOT confirmed: if memcap still owns it, it stays
    pending (the caller forces a re-render); if nothing owns it, it is dropped and reported.
    """
    # The journal holds vault-relative paths: recall must resolve them against THIS vault, not its
    # own default (which made every path look absent and "Indexed 0 docs" exit 0, 2026-11-06).
    r = subprocess.run([a.recall_exe, "index", "--path-list", str(journal_path)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "RECALL_VAULT": str(vault)})
    res = {"recall_exit": r.returncode, "dropped_missing": 0}
    if r.returncode != 0:
        res.update(state="failed: recall index exit", unconfirmed=paths, missing_owned=[])
        return res
    con = sqlite3.connect(f"file:{Path(a.recall_db).as_posix()}?mode=ro", uri=True, timeout=10)
    con.execute("PRAGMA busy_timeout=10000")
    bad, missing_owned = [], []
    for p in paths:
        f = vault / p
        if not f.exists():
            if p in owned_paths:
                bad.append(p); missing_owned.append(p)
            else:
                res["dropped_missing"] += 1
            continue
        row = con.execute("SELECT body FROM documents WHERE path = ? AND valid_to IS NULL "
                          "ORDER BY id DESC LIMIT 1", (p,)).fetchone()
        raw = f.read_bytes()        # raw bytes: no newline translation (recall stores CRLF as-is)
        # With scrub.store_redacted on, recall stores the REDACTED text, so a file holding anything
        # redactable never equals its stored body. Ask recall what it would store for these bytes
        # (the function the indexer ran) instead of re-deriving the redaction.
        if not row or (row[0] != raw.decode("utf-8") and not _stored_matches(a.recall_exe, raw, row[0])):
            bad.append(p)
    con.close()
    ok = not bad and not res["dropped_missing"]
    res.update(state="ok" if ok else f"failed: {len(bad)} unconfirmed, {res['dropped_missing']} missing",
               confirmed=len(paths) - len(bad) - res["dropped_missing"], unconfirmed=bad,
               missing_owned=missing_owned)
    return res


# ---------------------------------------------------------------------------
# main run
# ---------------------------------------------------------------------------
def alert(a, health: dict, prev: dict) -> None:
    """Post a failure (exit 2/3) to the alert command, once per distinct failure signature.

    Counts and error classes only, never note content. A clean run clears the signature so the
    next failure alerts again. A failed post is recorded in health, never raised.
    """
    code = health.get("exit_code")
    if code not in (2, 3):
        health["alert_sig"] = None if not health.get("skipped") else prev.get("alert_sig")
        return
    bits = [f"exit {code}"]
    if health.get("refused"):
        bits.append("refused: " + health["refused"][:120])
    for k in ("quarantined", "errors", "incomplete", "groups_frozen"):
        if health.get(k):
            bits.append(f"{len(health[k])} {k}")
    if health.get("pending_index"):
        bits.append(f"{health['pending_index']} not yet confirmed in recall.db")
    cm = (health.get("sources") or {}).get("claude_mem", "")
    if cm and not cm.startswith(("ok", "missing")):
        bits.append("claude-mem " + cm[:60])
    conflicts = (health.get("inventory") or {}).get("conflicts")
    if conflicts:
        bits.append(f"{len(conflicts)} duplicate sessions")
    sig = "; ".join(bits)
    health["alert_sig"] = sig
    if sig == prev.get("alert_sig") or not a.alert_cmd:
        return
    try:
        r = subprocess.run([sys.executable, a.alert_cmd, "--area", "recall", "--state", "needs",
                            "--name", "memcap", "--text",
                            f"memcap: {sig} (details: ~/.recall/memcap-health.json)"],
                           capture_output=True, timeout=60)
        health["alert_posted"] = r.returncode == 0
    except (OSError, subprocess.SubprocessError) as e:
        health["alert_posted"] = f"failed: {type(e).__name__}"


def overlaps(x: Path, y: Path) -> bool:
    x, y = x.resolve(), y.resolve()
    return x == y or x in y.parents or y in x.parents


def run(a) -> int:
    t0 = time.time()
    sys.path.insert(0, str(a.ingest_dir))
    import scrub_turns            # noqa: E402  (recall repo ingest/)
    import extract_transcript     # noqa: E402

    def scrub_text(text, tool_name=None, kind=None, keep=keep_token):
        return scrub_turns.scrub_text(text, tool_name=tool_name, kind=kind, keep=keep)

    vault, projects, state_dir = Path(a.vault), Path(a.projects), Path(a.state_dir)
    out_root = Path(a.out) if a.dry_run else vault
    capture_root = out_root / CAPTURE_REL
    work_dir = Path(a.out) if a.dry_run else state_dir      # dry runs never touch production state
    health = {"last_attempt": dt.datetime.now(dt.timezone.utc).isoformat(), "dry_run": a.dry_run}

    # --- preflight: no side effects at all before it passes ---
    try:
        if a.dry_run:
            o = Path(a.out)
            for other, name in ((vault, "the vault"), (state_dir, "the production state dir"),
                                (projects, "the transcript root")):
                if overlaps(o, other):
                    raise Refuse(f"--out overlaps {name}")
            if git_exposure(o.resolve()):
                raise Refuse("--out is inside a git work tree and not ignored there")
        else:
            probs = git_exposure(capture_root)
            if probs:
                raise Refuse("; ".join(probs))
            # a bare name ("recall") is looked up on PATH; the resolved path is what runs
            exe = a.recall_exe if Path(a.recall_exe).exists() else shutil.which(a.recall_exe)
            if not exe:
                raise Refuse(f"recall exe missing: {a.recall_exe}")
            a.recall_exe = exe
        if not projects.is_dir():
            raise Refuse(f"transcript root missing: {projects}")
    except Refuse as e:
        health.update(exit_code=2, refused=str(e))
        if not a.dry_run and state_dir.is_dir():            # production: make the refusal visible
            try:
                prev = load_json(state_dir / "memcap-health.json")
            except Refuse:
                prev = {}
            alert(a, health, prev)
            write_json(state_dir / "memcap-health.json", health)
        print(json.dumps(health, indent=1))
        return 2

    work_dir.mkdir(parents=True, exist_ok=True)
    lock = open(work_dir / "memcap.lock", "a+")
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:                       # the running instance owns log + health: touch neither
        lock.close()
        print(json.dumps({"skipped": "already running"}))
        return 0
    try:
        logp = work_dir / "memcap.log"
        if logp.exists() and logp.stat().st_size > 5_000_000:
            os.replace(logp, work_dir / "memcap.log.1")
        try:
            code, extra = run_locked(a, scrub_turns, extract_transcript, scrub_text, vault,
                                     projects, state_dir, out_root, capture_root, work_dir)
        except Refuse as e:
            code, extra = 2, {"refused": str(e)}
        health.update(extra, exit_code=code, duration_s=round(time.time() - t0, 1))
        try:
            prev = load_json(work_dir / "memcap-health.json")
        except Refuse:
            prev = {}
        health["last_success"] = (health["last_attempt"] if code == 0 and not extra.get("skipped")
                                  else prev.get("last_success"))
        if not a.dry_run:
            alert(a, health, prev)
        write_json(work_dir / "memcap-health.json", health)
        with open(logp, "a", encoding="utf-8") as log:
            log.write(json.dumps(health) + "\n")
        print(json.dumps(health, indent=1))
        return code
    finally:
        lock.close()                      # closing the handle releases the lock


def run_locked(a, scrub_turns, extract_transcript, scrub_text, vault, projects, state_dir,
               out_root, capture_root, work_dir) -> tuple[int, dict]:
    now = dt.datetime.now()
    if not a.dry_run and not a.ignore_window:
        # Measured 2026-11-02: a memcap backfill writing recall.db at 02:45 made the nightly
        # `recall embed` die with "database is locked". Stay out of its window (the task fires
        # at 02:45 local) and out of any nightly run that is still going.
        if (2, 15) <= (now.hour, now.minute) < (3, 45):
            return 0, {"skipped": "nightly index window (02:15-03:45)"}
        q = subprocess.run(["schtasks", "/query", "/tn", NIGHTLY_TASK, "/fo", "csv", "/nh"],
                           capture_output=True, text=True, errors="replace")
        if q.returncode == 0 and '"Running"' in q.stdout:
            return 0, {"skipped": f"{NIGHTLY_TASK} is running"}
    load_note = None
    if not a.dry_run and a.load_status:
        try:      # written by the machine's load monitor
            free = json.loads(Path(a.load_status).read_text(encoding="utf-8-sig"))["free_virtual_gb"]
            if float(free) < a.min_free_virtual_gb:
                return 0, {"skipped": f"low memory: free_virtual_gb {free} < {a.min_free_virtual_gb}"}
        except (OSError, ValueError, KeyError, TypeError) as e:
            load_note = f"load status unreadable ({type(e).__name__}); ran anyway"

    st_path = work_dir / "memcap-state.json"
    state = load_json(st_path)
    for key, typ in (("files", dict), ("sessions_db", dict), ("owned", dict), ("meta", dict),
                     ("had_transcript", list), ("had_mem", list)):
        if not isinstance(state.setdefault(key, typ()), typ):
            raise Refuse(f"memcap-state.json: '{key}' is not a {typ.__name__}")
    renderer = f"{MEMCAP_VERSION}:" + sha(Path(scrub_turns.__file__).read_bytes()
                                          + Path(extract_transcript.__file__).read_bytes())[:16]
    if state.get("renderer") != renderer:
        # renderer/scrubber changed: drop change-detection caches, KEEP provenance
        state.update(renderer=renderer, files={}, sessions_db={})
    if time.time() - state.get("last_reconcile", 0) > 7 * 86400:
        state.update(files={}, last_reconcile=time.time())  # weekly: re-read everything, bounded
    files_st, fps_st, owned, meta = state["files"], state["sessions_db"], state["owned"], state["meta"]
    had_tr, had_mem = set(state["had_transcript"]), set(state["had_mem"])
    excluded = set(read_lines(state_dir / "memcap-exclude.txt"))   # session ids / project names
    journal_path = work_dir / "memcap-pending.txt"
    journal = set(read_lines(journal_path))
    since = dt.datetime.fromisoformat(a.since).timestamp() if a.since else None
    root_real = os.path.realpath(capture_root)
    stats = {"groups_written": 0, "groups_frozen": [], "groups_withdrawn": 0, "groups_deferred": 0,
             "parts_written": 0, "parts_unchanged": 0, "parts_deleted": 0, "quarantined": [],
             "errors": [], "incomplete": [], "unparsed": 0, "unknown_blocks": 0, "redactions": 0}

    def checkpoint():
        atomic_write(journal_path, ("\n".join(sorted(journal)) + "\n").encode("utf-8"))
        state["had_transcript"], state["had_mem"] = sorted(had_tr), sorted(had_mem)
        write_json(st_path, state)

    def withdraw(ident: str, keep: set[str]) -> None:
        """Un-journal FIRST (a withdrawn path must never reach the index step), then delete.
        A failed delete keeps ownership so the next run retries it, and is reported."""
        failed = set()
        for rel in set(owned.get(ident, [])) - keep:
            journal.discard(rel)
            dest = out_root / rel
            if rel.startswith(CAPTURE_REL + "/") and under(dest, root_real) and dest.exists():
                try:
                    dest.unlink()
                    stats["parts_deleted"] += 1
                except OSError as e:
                    failed.add(rel)
                    stats["errors"].append(f"{ident}: withdraw {type(e).__name__}")
        owned[ident] = sorted(keep | failed)

    def is_excluded(ident: str, project: str) -> bool:
        return ident.split("__")[0] in excluded or (project and project in excluded)

    # --- exclusions apply to what is ALREADY published, whatever the sources are doing ---
    for ident in list(owned):
        if owned[ident] and is_excluded(ident, meta.get(ident, "")):
            withdraw(ident, set())
            stats["groups_withdrawn"] += 1

    # --- what changed ---
    inv, inv_report = inventory(projects)
    mem = read_mem(Path(a.mem_db))
    if mem["db_id"] and state.get("mem_db_id") not in (None, mem["db_id"]):
        fps_st.clear()                                    # DB migrated: re-check every session
    changed = [i for i, p in inv.items() if files_st.get(i) != [p.stat().st_size, p.stat().st_mtime_ns]
               and not (since and p.stat().st_mtime < since)]
    mem_changed = []
    if mem["state"] == "ok":
        keys = set(mem["fingerprints"]) | set(fps_st)    # a session whose rows vanished counts too
        mem_changed = [cs for cs in keys if fps_st.get(cs) != mem["fingerprints"].get(cs)]
        if since:
            cut = int(since * 1000)
            mem_changed = [cs for cs in mem_changed if cs not in mem["rows"]
                           or any(r[-1] >= cut for r in mem["rows"][cs])]
    # admission: one budget over the FINAL queue (transcript bytes actually read), oldest first;
    # the first item always runs so a file larger than the budget is never starved
    cand = sorted(set(changed) | set(mem_changed),
                  key=lambda i: (inv[i].stat().st_mtime_ns if i in inv else 0, i))
    todo, budget, pending_bytes = [], a.max_bytes, 0
    for ident in cand:
        cost = inv[ident].stat().st_size if ident in inv else 0
        if todo and cost > budget:
            stats["groups_deferred"] += 1
            pending_bytes += cost
            continue
        budget -= cost
        todo.append(ident)

    for n_done, ident in enumerate(todo, 1):
        main = "__" not in ident
        frozen = None
        if main and mem["state"] != "ok" and ident in had_mem:
            frozen = "claude-mem unavailable"
        elif ident not in inv and ident in had_tr:
            frozen = "transcript gone"
        if frozen:                       # never rebuild a group from fewer sources than before
            stats["groups_frozen"].append(f"{ident}: {frozen}")
            continue
        try:
            tr = read_transcript(inv[ident], extract_transcript._text_from_content) if ident in inv else None
            rows = mem["rows"].get(ident, []) if main else []
            project = (cwd_name(tr["cwd"]) if tr and tr["cwd"]
                       else mem["projects"].get(ident.split("__")[0], ""))
            if is_excluded(ident, project):
                withdraw(ident, set())
                stats["groups_withdrawn"] += 1
                continue
            if tr:
                stats["unparsed"] += tr["unparsed"]
                stats["unknown_blocks"] += tr["unknown_blocks"]
                if tr["unparsed"]:
                    stats["incomplete"].append(ident)
                    if owned.get(ident):         # never replace complete notes with a partial parse
                        stats["groups_frozen"].append(f"{ident}: incomplete parse")
                        continue
            by_day, rstats = render_identity(ident, tr, rows, mem["projects"].get(ident, ""), scrub_text)
            stats["redactions"] += rstats["redactions"]
            bad = []
            for day, parts in by_day.items():
                for j, d in enumerate(parts):
                    head, body = split_front(d)
                    # pass 2: whole-part scrub (patterns spanning joined fields)
                    body, n2 = scrub_text(body, tool_name=None, kind="assistant_text")
                    stats["redactions"] += n2
                    parts[j] = d = (head + body).encode("utf-8", "replace")
                    if len(d) >= PART_HARD_MAX:
                        bad.append(f"{day}:oversize")
                    # pass 3 (gate): the WHOLE part, frontmatter included except the validated
                    # session UUID line, must be a fixed point of the scrubber
                    whole = re.sub(r'^session: "[0-9a-f-]{36}"\n', "", d.decode("utf-8"), flags=re.M)
                    again, n3 = scrub_text(whole, tool_name=None, kind="assistant_text")
                    if n3:                       # label KINDS only, never content
                        kinds = sorted(set(LABEL_RE.findall(again)) - set(LABEL_RE.findall(whole)))
                        bad.append(f"{day}:{','.join(kinds) or 'same-kind'}")
            if bad:
                stats["quarantined"].append(f"{ident}: gate {bad}")
                withdraw(ident, set())           # a group failing the gate must not stay searchable
                continue
            produced = [part_path(ident, day, i) for day, parts in by_day.items()
                        for i in range(1, len(parts) + 1)]
            for rel in produced:
                if not under(out_root / rel, root_real):
                    raise RuntimeError("path escapes capture root")
            # record intent BEFORE mutating, so a crash can never orphan a part
            owned[ident] = sorted(set(owned.get(ident, [])) | set(produced))
            meta[ident] = project
            journal.update(produced)
            if not a.dry_run:
                checkpoint()
            for day, parts in by_day.items():
                for i, data in enumerate(parts, 1):
                    dest = out_root / part_path(ident, day, i)
                    if dest.exists() and dest.read_bytes() == data:
                        stats["parts_unchanged"] += 1
                    else:
                        atomic_write(dest, data)
                        stats["parts_written"] += 1
            withdraw(ident, set(produced))
        except Exception as e:                   # one bad source must not stop the others
            stats["errors"].append(f"{ident}: {type(e).__name__}")
            continue
        stats["groups_written"] += 1
        if tr:
            had_tr.add(ident)
            if not tr["trailing_partial"] and not tr["unparsed"]:
                files_st[ident] = tr["stat"]       # keyed by identity: no raw file names in state
        (had_mem.add if rows else had_mem.discard)(ident)
        if mem["state"] == "ok" and main:
            if ident in mem["fingerprints"]:
                fps_st[ident] = mem["fingerprints"][ident]
            else:
                fps_st.pop(ident, None)
        if not a.dry_run and n_done % CHECKPOINT_EVERY == 0:
            checkpoint()

    code = 0
    index = {"state": "skipped (dry run)" if a.dry_run else "nothing to index"}
    if not a.dry_run and journal:                # drains old pending work even if nothing changed
        checkpoint()
        owned_paths = {p for ps in owned.values() for p in ps}
        index = index_and_verify(a, vault, journal_path, sorted(journal), owned_paths)
        journal = set(index.pop("unconfirmed"))
        for rel in index.pop("missing_owned"):   # an owned file vanished: force its re-render
            ident = Path(rel).name.split(".")[0]
            files_st.pop(ident, None)
            fps_st.pop(ident, None)
        if not index["state"].startswith("ok"):
            code = 3
    if a.dry_run:
        journal = set()
    if (stats["quarantined"] or stats["errors"] or stats["incomplete"] or stats["groups_frozen"]
            or mem["state"] not in ("ok", "missing") or inv_report["conflicts"]):
        code = 3
    state["mem_db_id"] = mem["db_id"] or state.get("mem_db_id")
    checkpoint()
    if load_note:
        stats["load_note"] = load_note
    return code, dict(files_seen=len(inv), identities_processed=len(todo), pending_bytes=pending_bytes,
                      pending_index=len(journal), inventory=inv_report,
                      sources={"transcripts": str(projects),
                               "claude_mem": mem["state"] + (f" ({mem['error']})" if mem["error"] else "")},
                      unmatched_db_rows=mem["unmatched"], index=index, **stats)


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    vault = Path(os.environ.get("RECALL_VAULT") or HOME / "recall-notes")   # same default as recall
    config = Path(os.environ.get("CLAUDE_CONFIG_DIR") or HOME / ".claude")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vault", default=str(vault))
    ap.add_argument("--projects", default=str(config / "projects"))
    ap.add_argument("--mem-db", default=str(HOME / ".claude-mem" / "claude-mem.db"))
    ap.add_argument("--state-dir", default=str(HOME / ".recall"))
    ap.add_argument("--recall-exe", default=os.environ.get("RECALL_EXE") or "recall")
    ap.add_argument("--recall-db", default=str(HOME / ".recall" / "recall.db"))
    ap.add_argument("--ingest-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--since", help="YYYY-MM-DD: only sources touched on/after this date")
    ap.add_argument("--max-bytes", type=int, default=600_000_000,
                    help="transcript bytes per run (the oldest changed file always runs)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", help="dry-run root: outside the vault, not in an un-ignored git tree")
    ap.add_argument("--ignore-window", action="store_true", help="run even during the nightly recall index (02:15-03:45 or while it runs)")
    ap.add_argument("--load-status", default=os.environ.get("RECALL_LOAD_STATUS", ""),
                    help="load monitor JSON; skip the run when free_virtual_gb is below the floor ('' = off)")
    ap.add_argument("--min-free-virtual-gb", type=float, default=6.0)
    ap.add_argument("--alert-cmd", default=os.environ.get("RECALL_ALERT_CMD", ""),
                    help="dashboard poster for failures ('' = off)")
    a = ap.parse_args(argv)
    if a.dry_run and not a.out:
        ap.error("--dry-run needs --out")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
