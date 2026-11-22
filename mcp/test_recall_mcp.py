"""recall_health must report bge FAIL when the semantic channel is dead (G3, 2026-10-29).

Run: py -m pytest mcp/test_recall_mcp.py -q
"""
import json
import subprocess

import recall_mcp


def _fake_run(bge_rc, bge_stdout, bge_stderr):
    def run(args):
        if args[0] == "status":
            return subprocess.CompletedProcess(args, 0, "status ok", "")
        ch = args[args.index("--channels") + 1]
        if ch == "bge":
            return subprocess.CompletedProcess(args, bge_rc, bge_stdout, bge_stderr)
        hit = [{"rank": 0, "title": "t", "vault_path": "a.md", "source": "fts5"}]
        return subprocess.CompletedProcess(args, 0, json.dumps(hit), "")
    return run


def test_health_fails_bge_when_onnx_dll_missing(monkeypatch):
    monkeypatch.setattr(recall_mcp, "_recall_exe", lambda: "recall.exe")
    monkeypatch.setattr(recall_mcp, "_run", _fake_run(1, "", "bge: ONNX Runtime DLL not found"))
    report = recall_mcp.recall_health()
    assert "bge        FAIL" in report, report
    assert "fts5       OK" in report, report


def test_health_fails_bge_when_rescue_tier_answered(monkeypatch):
    rescued = json.dumps([{"rank": 0, "title": "t", "vault_path": "a.md", "source": "fts5-rescue"}])
    monkeypatch.setattr(recall_mcp, "_recall_exe", lambda: "recall.exe")
    monkeypatch.setattr(recall_mcp, "_run", _fake_run(0, rescued, ""))
    report = recall_mcp.recall_health()
    assert "bge        FAIL" in report, report


def test_health_ok_when_bge_answers(monkeypatch):
    real = json.dumps([{"rank": 0, "title": "t", "vault_path": "a.md", "source": "bge"}])
    monkeypatch.setattr(recall_mcp, "_recall_exe", lambda: "recall.exe")
    monkeypatch.setattr(recall_mcp, "_run", _fake_run(0, real, ""))
    report = recall_mcp.recall_health()
    assert "bge        OK" in report, report


def test_health_fails_bge_when_degraded_with_zero_hits(monkeypatch):
    # exit 0, stdout [], stderr names a backend fault: that is a failure, not "0 hits OK".
    monkeypatch.setattr(recall_mcp, "_recall_exe", lambda: "recall.exe")
    monkeypatch.setattr(recall_mcp, "_run", _fake_run(0, "[]", "bge: no vectors stored -- run recall embed"))
    report = recall_mcp.recall_health()
    assert "bge        FAIL" in report, report


def test_search_raises_when_degraded_with_zero_hits(monkeypatch):
    monkeypatch.setattr(recall_mcp, "_run", _fake_run(0, "[]", "bge: no vectors stored -- run recall embed"))
    try:
        out = recall_mcp.recall_search("x", channels="bge")
    except recall_mcp.RecallUnavailable:
        return
    raise AssertionError(f"must not report a clean empty search: {out}")


def test_abstain_is_a_clean_empty_answer(monkeypatch):
    # The abstain line is not a fault: [] + "abstain:" stays "found nothing".
    monkeypatch.setattr(recall_mcp, "_run", _fake_run(0, "[]", "abstain: no confident match (bge ...)"))
    assert "found nothing" in recall_mcp.recall_search("qxzvbnwplkjh", channels="bge")


def test_a_hit_lacking_a_subject_term_is_labelled_closest_match(monkeypatch):
    # K8: a hit missing a query subject term is never shown as a plain answer.
    hit = [{"rank": 0, "title": "Kestra runbook", "vault_path": "k.md", "source": "fts5", "lacks": ["gmail"]}]
    monkeypatch.setattr(recall_mcp, "_run", _fake_run(0, json.dumps(hit), ""))
    out = recall_mcp.recall_search("which gmail password rotation runbook steps?", channels="bge")
    assert "CLOSEST MATCH - does not mention: gmail" in out
    assert "WARNING: no result mentions every subject term" in out


def test_search_text_flags_stale_and_deleted_hits_and_unchecked_ones(monkeypatch):
    hits = [
        {"rank": 0, "title": "edited", "vault_path": "a.md", "stale": True,
         "stale_note": "this file changed on 2026-11-05 10:00 UTC after recall indexed it on 2026-11-01 02:45 UTC; read the actual file."},
        {"rank": 1, "title": "gone", "vault_path": "b.md", "stale": True, "deleted": True,
         "stale_note": "this file was deleted from disk after recall indexed it on 2026-11-01 02:45 UTC; the indexed text is a copy of a file that no longer exists."},
        {"rank": 2, "title": "fine", "vault_path": "c.md", "stale": False},
        {"rank": 3, "title": "unknown", "vault_path": "d.md", "stale": None,
         "stale_note": "freshness not checked: freshness_check is off"},
    ]
    monkeypatch.setattr(recall_mcp, "_run", lambda args: subprocess.CompletedProcess(args, 0, json.dumps(hits), ""))
    out = recall_mcp.recall_search("x", channels="fts5")
    assert "2 of 4 results point at files that changed on disk" in out, out
    assert "!! STALE: this file changed on 2026-11-05 10:00 UTC after recall indexed it" in out, out
    assert "deleted from disk" in out, out
    assert "? freshness not checked" in out, out
    # the fresh hit carries no marker at all
    fine = out.split("[2] fine")[1].split("[3]")[0]
    assert "STALE" not in fine and "?" not in fine, fine


def test_search_text_is_unchanged_when_nothing_is_stale(monkeypatch):
    hits = [{"rank": 0, "title": "t", "vault_path": "a.md", "stale": False}]
    monkeypatch.setattr(recall_mcp, "_run", lambda args: subprocess.CompletedProcess(args, 0, json.dumps(hits), ""))
    out = recall_mcp.recall_search("x", channels="fts5")
    assert "STALE" not in out and "WARNING" not in out, out
