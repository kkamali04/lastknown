"""Writes synthetic-session.jsonl: a SYNTHETIC Claude Code-shaped transcript for test_extract_transcript.

Every value is invented. It covers what the tests assert: prose turns, tool_use / tool_result
pairs (one tool named "crypt"), a tool_result longer than the 500-char cap, one bad JSON line,
skipped record types, and one user line over 1 MB.
usage: python ingest/fixtures/make_synthetic_session.py
"""
import json
from pathlib import Path

SID = "00000000-0000-4000-8000-000000000001"


def rec(i, typ, message, **extra):
    return json.dumps({"parentUuid": None if i == 0 else f"u{i - 1}", "isSidechain": False, "type": typ,
                       "message": message, "uuid": f"u{i}", "timestamp": f"2026-01-01T00:00:{i:02d}Z",
                       "sessionId": SID, **extra})


lines = [
    json.dumps({"type": "permission-mode", "mode": "default", "sessionId": SID}),
    rec(0, "user", {"role": "user", "content": "Where did I put the notes about the garden irrigation plan?"}),
    rec(1, "assistant", {"role": "assistant", "content": [
        {"type": "text", "text": "Let me search your notes for the irrigation plan."},
        {"type": "tool_use", "id": "t1", "name": "Grep", "input": {"pattern": "irrigation", "path": "notes/"}}]}),
    rec(2, "tool", {"role": "tool", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                                  "content": "notes/garden/irrigation.md:3: drip lines on the east beds"}]},
        toolName="Grep"),
    rec(3, "assistant", {"role": "assistant", "content": [
        {"type": "text", "text": "Found it in notes/garden/irrigation.md. Checking the secret store entry next."},
        {"type": "tool_use", "id": "t2", "name": "crypt", "input": {"cmd": "status"}}]}),
    rec(4, "tool", {"role": "tool", "content": [{"type": "tool_result", "tool_use_id": "t2",
                                                  "content": "vault: unlocked; entries: 3 " + "status-line " * 120}]},
        toolName="crypt"),
    "THIS LINE IS DELIBERATELY NOT JSON",
    rec(5, "user", {"role": "user", "content": "Paste of a very long log follows:\n" + "lorem ipsum dolor sit amet " * 40000}),
    rec(6, "assistant", {"role": "assistant", "content": [{"type": "text", "text": "That log is mostly repeated filler; nothing failed."}]}),
    json.dumps({"type": "last-prompt", "sessionId": SID}),
]
out = Path(__file__).with_name("synthetic-session.jsonl")
out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print(out, out.stat().st_size, "bytes")
