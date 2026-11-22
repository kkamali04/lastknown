"""test_memcap.py — synthetic-data checks for memcap.py (run: py -m pytest test_memcap.py -q)."""
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import memcap  # noqa: E402

INGEST = Path(__file__).resolve().parent
SID = "11111111-2222-3333-4444-555555555555"
SECRETS = [
    "sk-ant-api03-" + "Zq9" * 12,
    "ghp_" + "aB3dE5gH7jK9mN1pQ3sT5vW7yZ9bC1dE3fG5",
    "AWS_SECRET_ACCESS_KEY=" + "wJalrXUtnFEMI7K7MDENGbPxRfiCYzEXAMPLEKEYq",
    "postgres://admin:Sup3rS3cretPw@db.example.com:5432/prod",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
    "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n-----END PRIVATE KEY-----",
]
TOOL_INPUT = "TOOLINPUT_MARKER_never_emit"
TOOL_OUTPUT = "TOOLOUTPUT_MARKER_never_emit"


def rec(rtype, content, ts="2026-11-01T14:02:00.000Z", **kw):
    d = {"type": rtype, "uuid": kw.pop("uuid", os.urandom(6).hex()), "timestamp": ts,
         "cwd": "C:\\work\\demo-proj", "message": {"role": rtype, "content": content}}
    d.update(kw)
    return json.dumps(d)


def make_env(tmp_path, extra_lines=(), big=False):
    proj = tmp_path / "projects" / "C--work"
    proj.mkdir(parents=True)
    lines = [
        json.dumps({"type": "ai-title", "aiTitle": "Demo title " + SECRETS[1]}),
        rec("user", "please decide the cache policy " + SECRETS[0]
            + "<system-reminder>REMINDER_MARKER</system-reminder>"),
        rec("user", "SYNTHETIC_META_MARKER", isMeta=True),
        rec("assistant", [{"type": "text", "text": "DECISION: use LRU cache. " + SECRETS[2]},
                          {"type": "tool_use", "name": "Bash", "input": {"command": TOOL_INPUT}}]),
        rec("user", [{"type": "tool_result", "tool_use_id": "x", "content": TOOL_OUTPUT}]),
        rec("assistant", [{"type": "text", "text": "windows line" + chr(13) + chr(10) + "endings kept"}]),
        rec("assistant", [{"type": "text", "text": "conn " + SECRETS[3] + " jwt " + SECRETS[4]
                           + "\n" + SECRETS[5]}]),
    ]
    if big:
        lines.append(rec("assistant", [{"type": "text", "text": ("lorem ipsum dolor sit amet\n") * 40_000}]))
        lines.append(rec("assistant", [{"type": "text", "text": "NONL" + "x" * 1_000_000 + "END_NONL"}]))
    (proj / f"{SID}.jsonl").write_text("\n".join(lines + list(extra_lines)) + "\n", encoding="utf-8")
    sub = proj / SID / "subagents"
    sub.mkdir(parents=True)
    (sub / "agent-aabc123-0123456789abcdef.jsonl").write_text(rec("assistant", [{"type": "text", "text": "SUBAGENT_REPORT"}]) + "\n",
                                            encoding="utf-8")
    db = tmp_path / "mem.db"
    con = sqlite3.connect(db)
    con.executescript("""
      CREATE TABLE sdk_sessions(id INTEGER PRIMARY KEY, content_session_id TEXT, memory_session_id TEXT, project TEXT);
      CREATE TABLE observations(id INTEGER PRIMARY KEY, memory_session_id TEXT, type TEXT, title TEXT, subtitle TEXT,
        facts TEXT, narrative TEXT, files_modified TEXT, created_at_epoch INTEGER);
      CREATE TABLE session_summaries(id INTEGER PRIMARY KEY, memory_session_id TEXT, request TEXT, learned TEXT,
        completed TEXT, next_steps TEXT, created_at_epoch INTEGER);""")
    con.execute("INSERT INTO sdk_sessions VALUES (1, ?, 'm1', 'demo')", (SID,))
    con.execute("INSERT INTO observations VALUES (7, 'm1', 'decision', ?, 'sub', ?, ?, '[\"a.py\"]', 1793541720000)",
                ("OBS_TITLE " + SECRETS[1], json.dumps(["fact " + SECRETS[0]]), "narr " + SECRETS[3]))
    con.commit(); con.close()
    vault = tmp_path / "vault"
    vault.mkdir()
    subprocess.run(["git", "init", "-q", str(vault)], check=True)
    (vault / ".gitignore").write_text(memcap.CAPTURE_REL + "/\n")
    recall_db = tmp_path / "recall.db"
    c = sqlite3.connect(recall_db)
    c.execute("CREATE TABLE documents(id INTEGER PRIMARY KEY, path TEXT, body TEXT, valid_to TEXT)")
    c.commit(); c.close()
    fake = tmp_path / "fake_recall.py"
    fake.write_text(
        "import sys,sqlite3,os\n"
        f"mode=open(r'{tmp_path / 'mode'}').read().strip()\n"
        # the journal is vault-relative: memcap must tell recall which vault (regression 2026-11-06)
        # `recall stored-form`: stdin bytes -> the text the index would store
        "if sys.argv[1]=='stored-form':\n"
        "  if mode=='redact-noform': sys.exit(2)\n"
        "  d=sys.stdin.buffer.read().decode('utf-8')\n"
        "  sys.stdout.buffer.write((d.replace('REDACTME','[API_KEY:openai]') if mode=='redact' else d).encode('utf-8')); sys.exit(0)\n"
        f"if os.environ.get('RECALL_VAULT') != r'{vault}': sys.exit(7)\n"
        "if mode=='fail': sys.exit(1)\n"
        "paths=[l.strip() for l in open(sys.argv[3],encoding='utf-8') if l.strip()]\n"
        f"c=sqlite3.connect(r'{recall_db}')\n"
        "for p in paths:\n"
        f"  f=os.path.join(r'{vault}',p)\n"
        "  if not os.path.exists(f): continue\n"
        "  body=open(f,'rb').read().decode('utf-8') if mode!='stale' else 'OLD'\n"
        # store_redacted ON: recall stores the REDACTED text (a stand-in for the real scrub layers)
        "  if mode in ('redact','redact-noform'): body=body.replace('REDACTME', '[API_KEY:openai]')\n"
        "  c.execute('delete from documents where path=?',(p,))\n"
        "  c.execute('insert into documents(path,body) values(?,?)',(p,body))\n"
        "c.commit()\n")
    (tmp_path / "mode").write_text("ok")
    if os.name == "nt":
        exe = tmp_path / "fake_recall.cmd"
        exe.write_text(f'@"{sys.executable}" "{fake}" %*\n')
    else:
        exe = tmp_path / "fake_recall"
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n')
        exe.chmod(0o755)
    return {"vault": vault, "projects": tmp_path / "projects", "db": db, "recall_db": recall_db,
            "exe": exe, "state": tmp_path / "state", "tmp": tmp_path}


def run(env, *extra):
    return memcap.main(["--vault", str(env["vault"]), "--projects", str(env["projects"]),
                        "--mem-db", str(env["db"]), "--state-dir", str(env["state"]),
                        "--recall-exe", str(env["exe"]), "--recall-db", str(env["recall_db"]),
                        "--ingest-dir", str(INGEST), "--ignore-window",
                        "--load-status", "", "--alert-cmd", "", *extra])


def notes(env):
    return sorted((env["vault"] / memcap.CAPTURE_REL).rglob("*.md"))


def test_secrets_tools_reminders_never_emitted(tmp_path):
    env = make_env(tmp_path)
    assert run(env) == 0
    blob = "".join(p.read_text(encoding="utf-8") for p in notes(env))
    for s in SECRETS:
        for frag in ([s] if "\n" not in s else s.splitlines()[1:2]):
            assert frag not in blob, frag[:20]
    for marker in (TOOL_INPUT, TOOL_OUTPUT, "REMINDER_MARKER", "SYNTHETIC_META_MARKER"):
        assert marker not in blob
    assert "DECISION: use LRU cache" in blob and "OBS_TITLE" in blob and "obs #7" in blob
    main = [p for p in notes(env) if "__" not in p.name][0]
    assert '"session-capture"' in main.read_text(encoding="utf-8")
    sub = [p for p in notes(env) if "__0123456789abcdef" in p.name][0]
    assert "OBS_TITLE" not in sub.read_text(encoding="utf-8")   # claude-mem rows: main note only


def test_idempotent_second_run(tmp_path):
    env = make_env(tmp_path)
    assert run(env) == 0
    before = {p: p.stat().st_mtime_ns for p in notes(env)}
    os.utime(env["projects"] / "C--work" / f"{SID}.jsonl")      # force re-render, same content
    assert run(env) == 0
    assert {p: p.stat().st_mtime_ns for p in notes(env)} == before
    h = json.loads((env["state"] / "memcap-health.json").read_text())
    assert h["parts_written"] == 0 and h["parts_unchanged"] >= 1


def test_oversized_day_splits_without_loss(tmp_path):
    env = make_env(tmp_path, big=True)
    assert run(env) == 0
    main_parts = [p for p in notes(env) if "__" not in p.name]
    assert len(main_parts) >= 2
    for p in main_parts:
        data = p.read_bytes()
        assert len(data) < memcap.PART_HARD_MAX and data.startswith(b"---\n")
    texts = [p.read_text(encoding="utf-8") for p in main_parts]
    assert sum(t.count("lorem ipsum") for t in texts) == 40_000
    assert any("END_NONL" in t for t in texts)
    assert sum(t.count("x" * 1000) for t in texts) >= 990     # newline-free 1 MB string kept


def test_preflight_refuses_unignored_capture_dir(tmp_path):
    env = make_env(tmp_path)
    (env["vault"] / ".gitignore").write_text("")
    assert run(env) == 2
    assert notes(env) == []


def test_dry_run_refuses_out_in_git_tree(tmp_path):
    env = make_env(tmp_path)
    assert run(env, "--dry-run", "--out", str(env["vault"] / "x")) == 2
    gitdir = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(gitdir)], check=True)
    assert run(env, "--dry-run", "--out", str(gitdir / "out")) == 2
    assert run(env, "--dry-run", "--out", str(tmp_path / "dry")) == 0
    assert list((tmp_path / "dry").rglob("*.md")) and notes(env) == []


def test_index_failure_keeps_journal_and_retries(tmp_path):
    env = make_env(tmp_path)
    (tmp_path / "mode").write_text("fail")
    assert run(env) == 3
    pending = (env["state"] / "memcap-pending.txt").read_text().split()
    assert pending
    (tmp_path / "mode").write_text("ok")
    assert run(env) == 0
    assert (env["state"] / "memcap-pending.txt").read_text().strip() == ""
    c = sqlite3.connect(env["recall_db"])
    assert {r[0] for r in c.execute("select path from documents")} >= set(pending)


def test_obsolete_parts_deleted_but_foreign_files_kept(tmp_path):
    env = make_env(tmp_path, big=True)
    assert run(env) == 0
    n_big = len(notes(env))
    foreign = env["vault"] / memcap.CAPTURE_REL / "2026-11-01" / "handwritten.md"
    foreign.write_text("mine")
    f = env["projects"] / "C--work" / f"{SID}.jsonl"
    f.write_text("\n".join(l for l in f.read_text(encoding="utf-8").splitlines() if "lorem" not in l) + "\n",
                 encoding="utf-8")
    assert run(env) == 0
    assert len(notes(env)) < n_big + 1 and foreign.exists()


def test_trailing_partial_line_not_advanced(tmp_path):
    env = make_env(tmp_path, extra_lines=['{"type": "assistant", "trunc'])
    assert run(env) == 0
    st = json.loads((env["state"] / "memcap-state.json").read_text())
    assert str(env["projects"] / "C--work" / f"{SID}.jsonl") not in st["files"]


def test_part_pass_redacts_leak_from_renderer(tmp_path, monkeypatch):
    env = make_env(tmp_path)
    real = memcap.render_identity

    def leaky(*a, **k):
        out, st = real(*a, **k)
        return {d: [p + ("\nleak " + SECRETS[0]).encode() for p in parts] for d, parts in out.items()}, st
    monkeypatch.setattr(memcap, "render_identity", leaky)
    assert run(env) == 0
    assert SECRETS[0] not in "".join(p.read_text(encoding="utf-8") for p in notes(env))


def test_gate_quarantines_and_withdraws_published_group(tmp_path, monkeypatch):
    env = make_env(tmp_path)
    assert run(env) == 0
    assert any("__" not in p.name for p in notes(env))
    sys.path.insert(0, str(INGEST))
    import scrub_turns
    real = scrub_turns.scrub_text
    monkeypatch.setattr(scrub_turns, "scrub_text",
                        lambda t, tool_name=None, kind=None, keep=None: (t, 1) if "GATE_MARKER" in t else real(t, tool_name, kind, keep))
    f = env["projects"] / "C--work" / f"{SID}.jsonl"
    f.write_text(f.read_text(encoding="utf-8") + rec("assistant", [{"type": "text", "text": "GATE_MARKER"}]) + "\n",
                 encoding="utf-8")
    assert run(env) == 3
    assert not any("__" not in p.name for p in notes(env))        # main group withdrawn
    h = json.loads((env["state"] / "memcap-health.json").read_text())
    assert h["quarantined"] and "GATE_MARKER" not in json.dumps(h)


def test_exclude_list_withdraws(tmp_path):
    env = make_env(tmp_path)
    assert run(env) == 0
    (env["state"] / "memcap-exclude.txt").write_text("demo-proj\n")      # no source touched
    assert run(env) == 0
    assert notes(env) == []


def test_broken_db_freezes_combined_group(tmp_path):
    env = make_env(tmp_path)
    assert run(env) == 0
    before = {p: p.read_bytes() for p in notes(env)}
    con = sqlite3.connect(env["db"]); con.execute("ALTER TABLE observations RENAME TO obs_old"); con.commit(); con.close()
    os.utime(env["projects"] / "C--work" / f"{SID}.jsonl")
    assert run(env) == 3
    assert {p: p.read_bytes() for p in notes(env)} == before      # not rebuilt transcript-only


def test_redacting_index_confirms_but_stale_and_silent_recall_do_not(tmp_path):
    """store_redacted ON: recall stores the REDACTED text of a note holding a secret-shaped value, so
    its stored body never equals the file. memcap must confirm against what recall WOULD store
    (2026-11-06: 13 files 'unconfirmed' every hour, exit 3, once redaction went on), and must still
    reject a stale index and a recall that cannot answer."""
    env = make_env(tmp_path)
    assert run(env) == 0
    rels = [str(p.relative_to(env["vault"])).replace(os.sep, "/") for p in notes(env)]
    for p in notes(env):
        p.write_text(p.read_text(encoding="utf-8") + "\nkey REDACTME end\n", encoding="utf-8")
    journal = env["state"] / "j.txt"
    journal.write_text("\n".join(rels) + "\n", encoding="utf-8")
    a = type("A", (), {"recall_exe": str(env["exe"]), "recall_db": str(env["recall_db"])})()

    (tmp_path / "mode").write_text("ok")               # control: raw index of a changed file confirms
    assert memcap.index_and_verify(a, env["vault"], journal, rels, set())["state"] == "ok"

    (tmp_path / "mode").write_text("redact")           # index stores the redacted text -> confirmed
    res = memcap.index_and_verify(a, env["vault"], journal, rels, set())
    assert res["state"] == "ok" and res["confirmed"] == len(rels), res
    stored = sqlite3.connect(env["recall_db"]).execute("SELECT body FROM documents").fetchall()
    assert all("REDACTME" not in b[0] and "[API_KEY:openai]" in b[0] for b in stored)

    (tmp_path / "mode").write_text("stale")            # index holds OLD text -> must still fail
    assert memcap.index_and_verify(a, env["vault"], journal, rels, set())["unconfirmed"] == rels

    (tmp_path / "mode").write_text("redact-noform")    # recall cannot answer stored-form -> unconfirmed
    assert memcap.index_and_verify(a, env["vault"], journal, rels, set())["unconfirmed"] == rels
    con = sqlite3.connect(env["recall_db"]); con.execute("UPDATE documents SET body = NULL"); con.commit(); con.close()
    assert memcap._stored_matches(str(env["exe"]), b"x", None) is False      # NULL body never matches a failed helper


def test_readback_rejects_stale_index(tmp_path):
    env = make_env(tmp_path)
    (tmp_path / "mode").write_text("stale")
    assert run(env) == 3
    assert (env["state"] / "memcap-pending.txt").read_text().split()


def test_equal_length_db_edit_rerenders(tmp_path):
    env = make_env(tmp_path)
    assert run(env) == 0
    con = sqlite3.connect(env["db"]); con.execute("UPDATE observations SET subtitle='SUB' WHERE id=7"); con.commit(); con.close()
    assert run(env) == 0
    assert "SUB" in "".join(p.read_text(encoding="utf-8") for p in notes(env) if "__" not in p.name)


def test_broken_db_freeze_survives_renderer_change(tmp_path, monkeypatch):
    env = make_env(tmp_path)
    assert run(env) == 0
    before = {p: p.read_bytes() for p in notes(env)}
    con = sqlite3.connect(env["db"]); con.execute("ALTER TABLE observations RENAME TO obs_old"); con.commit(); con.close()
    monkeypatch.setattr(memcap, "MEMCAP_VERSION", 999)
    assert run(env) == 3
    main = [p for p in before if "__" not in p.name]      # the DB-backed group stays frozen
    assert main and all(p.read_bytes() == before[p] for p in main)


def test_keep_token_never_exempts_named_credentials():
    sys.path.insert(0, str(INGEST))
    import scrub_turns
    path = "research/infra/JANITOR-LOG-2026-11-01"
    b64 = "Zm9vYmFy/QWxhZGRpbjpvcGVuIHNlc2FtZQ9xK2pLm7Tq/Vw8RtYb"
    key = "sk" + "-ant-api03-" + "abcdefghijklmnopqrstuv" + "-2026-11-01"   # split: leak-gate hook
    out, n = scrub_turns.scrub_text(f"see {path} and {b64} and {key}", kind="assistant_text",
                                    keep=memcap.keep_token)
    assert path in out and b64 not in out and key not in out and n >= 2
    raw, _ = scrub_turns.scrub_text(f"see {path}", kind="assistant_text")
    assert path not in raw          # control: without keep, the entropy rule eats the path


def test_corrupt_state_or_unreadable_exclude_refuses(tmp_path):
    env = make_env(tmp_path)
    assert run(env) == 0
    (env["state"] / "memcap-state.json").write_text("{not json")
    assert run(env) == 2
    (env["state"] / "memcap-state.json").write_text("{}")
    (env["state"] / "memcap-exclude.txt").mkdir()      # exists but cannot be read as a file
    assert run(env) == 2


def test_dry_run_cannot_touch_production_state(tmp_path):
    env = make_env(tmp_path)
    (tmp_path / "mode").write_text("fail")
    assert run(env) == 3                                   # leaves a non-empty production journal
    before = {p.name: p.read_bytes() for p in env["state"].iterdir() if p.is_file()}
    assert run(env, "--dry-run", "--out", str(env["state"])) == 2
    assert run(env, "--dry-run", "--out", str(env["state"] / "sub")) == 2
    assert {p.name: p.read_bytes() for p in env["state"].iterdir() if p.is_file()} == before


def test_missing_owned_file_is_not_confirmed(tmp_path):
    env = make_env(tmp_path)
    assert run(env) == 0
    victim = [p for p in notes(env) if "__" not in p.name][0]
    victim.unlink()
    rel = victim.relative_to(env["vault"]).as_posix()
    (env["state"] / "memcap-pending.txt").write_text(rel + "\n")
    assert run(env) == 3                                   # not silently confirmed
    assert run(env) == 0 and victim.exists()               # next run re-renders it


def test_deleted_db_rows_are_removed_from_note(tmp_path):
    env = make_env(tmp_path)
    assert run(env) == 0
    con = sqlite3.connect(env["db"]); con.execute("DELETE FROM observations"); con.commit(); con.close()
    assert run(env) == 0
    assert "OBS_TITLE" not in "".join(p.read_text(encoding="utf-8") for p in notes(env))


def test_duplicate_session_in_two_dirs_is_a_visible_conflict(tmp_path):
    env = make_env(tmp_path)
    other = env["projects"] / "C--other"
    other.mkdir()
    (other / f"{SID}.jsonl").write_text(rec("user", "copy") + "\n", encoding="utf-8")
    assert run(env) == 3
    h = json.loads((env["state"] / "memcap-health.json").read_text())
    assert h["inventory"]["conflicts"] == [SID]


def test_bad_identity_does_not_block_healthy_one(tmp_path):
    env = make_env(tmp_path)
    sid2 = "22222222-3333-4444-5555-666666666666"
    (env["projects"] / "C--work" / f"{sid2}.jsonl").write_text(
        rec("assistant", [{"type": "text", "text": {"not": "a string"}}]) + "\n"
        + rec("assistant", [{"type": "text", "text": "HEALTHY2"}]) + "\n", encoding="utf-8")
    (env["projects"] / "C--work" / "not-a-uuid.jsonl").write_text("x\n")
    assert run(env) == 3                                   # sid2 incomplete -> partial, visible
    blob = "".join(p.read_text(encoding="utf-8") for p in notes(env))
    assert "DECISION: use LRU cache" in blob and "HEALTHY2" in blob
    h = json.loads((env["state"] / "memcap-health.json").read_text())
    assert h["incomplete"] == [sid2] and h["inventory"]["unrecognised_files"] == 1


def test_budget_covers_db_triggered_reads(tmp_path):
    env = make_env(tmp_path)
    sid2 = "22222222-3333-4444-5555-666666666666"
    (env["projects"] / "C--work" / f"{sid2}.jsonl").write_text(
        rec("assistant", [{"type": "text", "text": "S2"}]) + "\n", encoding="utf-8")
    con = sqlite3.connect(env["db"])
    con.execute("INSERT INTO sdk_sessions VALUES (2, ?, 'm2', 'demo')", (sid2,))
    con.execute("INSERT INTO observations VALUES (8, 'm2', 'decision', 't', 's', '[]', 'n', '[]', 1793541720000)")
    con.commit(); con.close()
    assert run(env, "--max-bytes", "1") == 0
    h = json.loads((env["state"] / "memcap-health.json").read_text())
    assert h["identities_processed"] == 1 and h["groups_deferred"] >= 2


def test_credential_shaped_agent_name_reaches_no_artifact(tmp_path):
    env = make_env(tmp_path)
    tok = "ghp" + "_" + "aB3dE5gH7jK9mN1pQ3sT5vW7yZ9bC1dE3fG5"      # split: leak-gate hook
    sub = env["projects"] / "C--work" / SID / "subagents"
    (sub / f"agent-a{tok}-fedcba9876543210.jsonl").write_text(
        rec("assistant", [{"type": "text", "text": "REPORT2"}]) + "\n", encoding="utf-8")
    run(env)
    arts = [p for p in tmp_path.rglob("*") if p.is_file() and "projects" not in p.parts]
    assert not any(tok in p.name for p in arts)
    assert not any(tok in p.read_text(encoding="utf-8", errors="replace") for p in arts
                   if p.suffix in (".md", ".json", ".txt", ".log"))
    assert "REPORT2" in "".join(p.read_text(encoding="utf-8") for p in notes(env))


def test_failed_withdraw_never_reaches_index(tmp_path, monkeypatch):
    env = make_env(tmp_path)
    assert run(env) == 0
    main = [p for p in notes(env) if "__" not in p.name][0]
    rel = main.relative_to(env["vault"]).as_posix()
    (env["state"] / "memcap-pending.txt").write_text(rel + "\n")
    (env["state"] / "memcap-exclude.txt").write_text("demo-proj\n")
    real_unlink = Path.unlink

    def boom(self, *a, **k):
        if self.name == main.name:
            raise PermissionError("sharing violation")
        return real_unlink(self, *a, **k)
    monkeypatch.setattr(Path, "unlink", boom)
    seen = []
    real_iv = memcap.index_and_verify
    monkeypatch.setattr(memcap, "index_and_verify",
                        lambda a, v, j, paths, o: seen.extend(paths) or real_iv(a, v, j, paths, o))
    assert run(env) == 3
    assert rel not in seen
    monkeypatch.setattr(Path, "unlink", real_unlink)
    assert run(env) == 0 and not main.exists()         # retried and withdrawn next run


def test_load_gate_skips_run(tmp_path):
    env = make_env(tmp_path)
    ls = tmp_path / "load.json"
    ls.write_text('{"free_virtual_gb": 2.5}')
    assert run(env, "--load-status", str(ls)) == 0
    assert notes(env) == []
    assert "low memory" in json.loads((env["state"] / "memcap-health.json").read_text())["skipped"]
    ls.write_text('{"free_virtual_gb": 9.0}')
    assert run(env, "--load-status", str(ls)) == 0 and notes(env)


def test_failure_alert_posts_once_per_signature(tmp_path):
    env = make_env(tmp_path)
    log = tmp_path / "alerts.txt"
    poster = tmp_path / "post.py"
    poster.write_text("import sys\nopen(r'%s','a').write(' '.join(sys.argv[1:])+chr(10))\n" % log)
    (tmp_path / "mode").write_text("fail")
    assert run(env, "--alert-cmd", str(poster)) == 3
    assert run(env, "--alert-cmd", str(poster)) == 3          # same failure: no second post
    lines = log.read_text().splitlines()
    assert len(lines) == 1 and "--area recall --state needs --name memcap" in lines[0]
    assert "not yet confirmed in recall.db" in lines[0]
    (tmp_path / "mode").write_text("ok")
    assert run(env, "--alert-cmd", str(poster)) == 0
    (tmp_path / "mode").write_text("fail")
    os.utime(env["projects"] / "C--work" / f"{SID}.jsonl")
    assert run(env, "--alert-cmd", str(poster)) == 3          # new failure after recovery: posts
    assert len(log.read_text().splitlines()) == 2


def test_claude_mem_section_is_labelled_unverified(tmp_path):
    env = make_env(tmp_path)
    assert run(env) == 0
    main = [p for p in notes(env) if "__" not in p.name][0].read_text(encoding="utf-8")
    assert "claude-mem AI summary — unverified, may contain errors" in main


def test_skips_while_nightly_index_runs(tmp_path, monkeypatch):
    env = make_env(tmp_path)
    real = subprocess.run

    def fake(cmd, *a, **k):
        if cmd and cmd[0] == "schtasks":
            return subprocess.CompletedProcess(cmd, 0, '"\Recall-Nightly-Index","x","Running"\n', "")
        return real(cmd, *a, **k)
    monkeypatch.setattr(memcap.subprocess, "run", fake)
    monkeypatch.setattr(memcap.dt, "datetime", type("D", (memcap.dt.datetime,), {
        "now": classmethod(lambda c, tz=None: memcap.dt.datetime(2026, 11, 2, 12, 0, tzinfo=tz))}))
    args = ["--vault", str(env["vault"]), "--projects", str(env["projects"]), "--mem-db", str(env["db"]),
            "--state-dir", str(env["state"]), "--recall-exe", str(env["exe"]), "--recall-db",
            str(env["recall_db"]), "--ingest-dir", str(INGEST), "--load-status", "", "--alert-cmd", ""]
    assert memcap.main(args) == 0
    assert notes(env) == []
    assert "Recall-Nightly-Index is running" in json.loads((env["state"] / "memcap-health.json").read_text())["skipped"]


def test_capture_rel_stays_inside_the_vault():
    assert memcap.capture_rel(None) == ".memory-capture"
    assert memcap.capture_rel("notes//./.memory-capture/") == "notes/.memory-capture"
    for bad in ("C:/elsewhere", "/abs/path", "../up", "a/../../b", "\\\\server\\share"):
        with pytest.raises(ValueError):
            memcap.capture_rel(bad)
