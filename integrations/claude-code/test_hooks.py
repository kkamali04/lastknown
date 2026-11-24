"""Exercise hook boundaries with synthetic inputs and an isolated profile; no host config."""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

spec = importlib.util.spec_from_file_location("lastknown_hooks", Path(__file__).with_name("hooks.py"))
hooks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hooks)
sys.path.insert(0, str(hooks.ROOT / "ingest"))
import memcap

SID = "11111111-2222-3333-4444-555555555555"


def fixture(tmp_path):
    vault, profile, projects = (tmp_path / p for p in ("notes", "chosen-home/.recall", "projects"))
    for p in (vault, profile, projects / "demo"):
        p.mkdir(parents=True)
    (profile / "recall.db").touch()
    (projects / "demo" / (SID + ".jsonl")).write_text("{}\n")
    exe = tmp_path / ("recall.cmd" if os.name == "nt" else "recall")
    exe.write_text('@echo recall 0.1.0\n' if os.name == "nt" else '#!/bin/sh\necho recall 0.1.0\n')
    if os.name != "nt": exe.chmod(0o755)
    args = ["--vault", str(vault), "--profile", str(profile), "--projects", str(projects),
            "--recall-exe", str(exe)]
    return args, vault, profile, projects


def test_start_supplies_availability_without_capture(tmp_path, monkeypatch, capsys):
    args, vault, profile, _ = fixture(tmp_path)
    monkeypatch.setattr(memcap, "main", lambda *a: pytest.fail("start must not capture"))
    def version(cmd, **kwargs):
        assert cmd[-1] == "--version"
        assert kwargs["env"]["USERPROFILE"] == kwargs["env"]["HOME"] == str(profile.parent)
        return subprocess.CompletedProcess(cmd, 0, "recall 0.1.0\n", "")
    monkeypatch.setattr(hooks.subprocess, "run", version)
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    assert hooks.main(["session-start", *args], {"hook_event_name": "SessionStart"}) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "recall 0.1.0" in response["hookSpecificOutput"]["additionalContext"]
    assert before == sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))


def test_end_scopes_capture_and_never_uses_input_path(tmp_path, monkeypatch):
    args, vault, profile, _ = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(memcap, "main", lambda argv, **kw: calls.append(argv) or 0)
    payload = {"hook_event_name": "SessionEnd", "session_id": SID,
               "transcript_path": "../../unrelated-secret.jsonl", "cwd": "--no-daemon"}
    assert hooks.main(["session-end", *args], payload) == 0
    argv = calls[0]
    assert argv[argv.index("--session-id") + 1] == SID and "--no-claude-mem" in argv
    assert argv[argv.index("--state-dir") + 1] == str(profile / "capture-hooks" / SID)
    assert argv[argv.index("--capture-rel") + 1] == ".memory-capture/claude-code"
    assert not any("unrelated-secret" in x or x == "--no-daemon" for x in argv)


@pytest.mark.parametrize("payload", [
    {"hook_event_name": "SessionEnd", "session_id": "../escape"},
    {"hook_event_name": "SessionEnd", "session_id": [SID]},
    {"hook_event_name": "Stop", "session_id": SID}, [],
])
def test_invalid_event_or_session_does_not_capture(tmp_path, monkeypatch, payload):
    args, *_ = fixture(tmp_path)
    monkeypatch.setattr(memcap, "main", lambda *a: pytest.fail("must refuse before capture"))
    assert hooks.main(["session-end", *args], payload) == 1


def test_partial_capture_is_not_reported_as_success(tmp_path, monkeypatch):
    args, *_ = fixture(tmp_path)
    monkeypatch.setattr(memcap, "main", lambda *a, **kw: 3)
    assert hooks.main(["session-end", *args], {"hook_event_name": "SessionEnd", "session_id": SID}) == 3


def test_payload_is_bounded_before_parsing(tmp_path, monkeypatch):
    args, *_ = fixture(tmp_path)
    monkeypatch.setattr(sys, "stdin", io.StringIO("x" * 65537))
    assert hooks.main(["session-start", *args]) == 1


def test_capture_profile_routing_with_isolated_git_preflight(tmp_path, monkeypatch):
    from test_memcap import make_env, notes
    env = make_env(tmp_path)
    # This test changes parent HOME deliberately. Keep its focus on child-profile routing;
    # the real Git-exposure preflight is exercised separately throughout test_memcap.py.
    monkeypatch.setattr(memcap, "git_exposure", lambda path: [])
    profile = tmp_path / "chosen-home" / ".recall"
    profile.mkdir(parents=True)
    old_db = env["recall_db"]
    new_db = profile / "recall.db"
    old_db.rename(new_db)
    fake = tmp_path / "fake_recall.py"
    program = fake.read_text()
    program = program.replace(str(old_db), str(new_db))
    # This executable consumes the same HOME mechanism as Rust, so a hardcoded DB stub
    # cannot hide accidental use of the inherited contributor profile.
    expected_home = str(profile.parent)
    program = program.replace('mode=open',
        f'assert os.environ.get("USERPROFILE") == {expected_home!r}\n'
        f'assert os.environ.get("HOME") == {expected_home!r}\nmode=open', 1)
    fake.write_text(program)
    wrong_home = tmp_path / "wrong-home"
    (wrong_home / ".recall").mkdir(parents=True)
    sentinel = wrong_home / ".recall" / "recall.db"
    sentinel.write_bytes(b"MUST_NOT_BE_OPENED_OR_WRITTEN")
    monkeypatch.setenv("USERPROFILE", str(wrong_home))
    monkeypatch.setenv("HOME", str(wrong_home))
    other = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    (env["projects"] / "C--work" / (other + ".jsonl")).write_text("NOT_SELECTED\n")
    monkeypatch.setattr(memcap.dt, "datetime", type("D", (memcap.dt.datetime,), {
        "now": classmethod(lambda c, tz=None: memcap.dt.datetime(2026, 11, 2, 12, 0, tzinfo=tz))}))
    args = ["session-end", "--vault", str(env["vault"]), "--profile", str(profile),
            "--projects", str(env["projects"]), "--recall-exe", str(env["exe"])]
    assert hooks.main(args, {"hook_event_name": "SessionEnd", "session_id": SID}) == 0
    text = "\n".join(p.read_text(encoding="utf-8") for p in notes(env))
    assert "DECISION: use LRU cache" in text and "SUBAGENT_REPORT" in text
    assert "NOT_SELECTED" not in text and "OBS_TITLE" not in text
    health = json.loads((profile / "capture-hooks" / SID / "memcap-health.json").read_text())
    assert health["exit_code"] == 0 and health["index"]["state"].startswith("ok")
    assert health["pending_index"] == 0
    assert sentinel.read_bytes() == b"MUST_NOT_BE_OPENED_OR_WRITTEN"
    assert os.environ["USERPROFILE"] == os.environ["HOME"] == str(wrong_home)


def test_custom_profile_binds_stored_form_as_well_as_index(tmp_path, monkeypatch):
    # Reuse the real capture test's explicit child-home assertions, but make stored text
    # differ so confirmation MUST invoke stored-form instead of the raw-body fast path.
    from test_memcap import make_env
    env = make_env(tmp_path)
    monkeypatch.setattr(memcap, "git_exposure", lambda path: [])
    profile = tmp_path / ".recall"
    profile.mkdir()
    old_db, new_db = env["recall_db"], profile / "recall.db"
    old_db.rename(new_db)
    fake = tmp_path / "fake_recall.py"
    text = fake.read_text().replace(str(old_db), str(new_db))
    text = text.replace('mode=open', f'assert os.environ.get("USERPROFILE") == {str(tmp_path)!r}\nmode=open', 1)
    fake.write_text(text)
    transcript = env["projects"] / "C--work" / (SID + ".jsonl")
    transcript.write_text(transcript.read_text() + json.dumps({"type":"user", "uuid":"extra",
        "timestamp":"2026-11-01T14:03:00Z", "message":{"role":"user","content":"REDACTME"}}) + "\n")
    (tmp_path / "mode").write_text("redact")
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "other-home"))
    monkeypatch.setenv("HOME", str(tmp_path / "other-home"))
    monkeypatch.setattr(memcap.dt, "datetime", type("D", (memcap.dt.datetime,), {
        "now": classmethod(lambda c, tz=None: memcap.dt.datetime(2026, 11, 2, 12, 0, tzinfo=tz))}))
    assert hooks.main(["session-end", "--session-id", SID, "--vault", str(env["vault"]),
                       "--profile", str(profile), "--projects", str(env["projects"]),
                       "--recall-exe", str(env["exe"])]) == 0


def test_symlink_outside_transcript_root_is_refused(tmp_path, monkeypatch):
    args, _, _, projects = fixture(tmp_path)
    source = projects / "demo" / (SID + ".jsonl")
    outside = tmp_path / "private.jsonl"
    outside.write_text("MUST_NOT_BE_READ")
    source.unlink()
    try:
        source.symlink_to(outside)
    except OSError:
        pytest.skip("symlink privilege unavailable on this host")
    monkeypatch.setattr(memcap, "main", lambda *a: pytest.fail("external source must not capture"))
    assert hooks.main(["session-end", *args], {"hook_event_name": "SessionEnd", "session_id": SID}) == 1
