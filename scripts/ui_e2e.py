"""End-to-end check of `recall ui` against a throwaway HOME and vault. Two runs:

  clean  every RECALL_* stripped from the inherited environment; fts5 only.
  real   the caller's own RECALL_* kept (e.g. RECALL_CHANNELS=fts5,bge, RECALL_ABSTAIN), the BGE
         model COPIED from ~/.recall/models, onnxruntime.dll beside the exe; search in-process, then
         through a throwaway daemon started in the throwaway HOME.

USERPROFILE/HOME, RECALL_VAULT and RECALL_ERROR_LOG always point at the temp dir, so ~/.recall,
the live DB and the live daemon are never touched. Env values are never printed.
Usage: python -I scripts/ui_e2e.py <path-to-recall.exe> [--clean-only]
"""
import http.client, json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

EXE = sys.argv[1]
EMAIL = "ui.canary@example-corp.com"
CUSTOM = "ProjectFalconX"
LIVE_MODELS = Path(os.environ.get("USERPROFILE") or os.environ["HOME"], ".recall", "models")


def env(home, vault, keep_recall):
    e = {k: v for k, v in os.environ.items() if keep_recall or not k.upper().startswith("RECALL_")}
    e.update(USERPROFILE=home, HOME=home, RECALL_VAULT=vault, RECALL_ERROR_LOG=str(Path(home, "error-log.jsonl")))
    return e


def call(port, method, path, token=None, body=None, host=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=150)
    c.putrequest(method, path, skip_host=True)
    c.putheader("Host", host or f"127.0.0.1:{port}")
    if token:
        c.putheader("X-Recall-Token", token)
    data = json.dumps(body).encode() if body is not None else b""
    c.putheader("Content-Length", str(len(data)))
    if data:
        c.putheader("Content-Type", "application/json")
    c.endheaders(data)
    r = c.getresponse()
    raw = r.read().decode()
    hdrs = {k.lower(): v for k, v in r.getheaders()}
    c.close()
    return r.status, hdrs, raw


def run(e, *args, timeout=600):
    r = subprocess.run([EXE, *args], env=e, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    assert r.returncode == 0, f"{args}: {r.stderr[-800:]}"
    return r


def start_ui(e):
    p = subprocess.Popen([EXE, "ui", "--no-open"], env=e, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8")
    m = re.search(r"http://127\.0\.0\.1:(\d+)/\?t=([0-9a-f]{64})", p.stdout.readline())
    assert m, "no URL line"
    return p, int(m.group(1)), m.group(2)


def search(port, tok, body):
    st, _, raw = call(port, "POST", "/api/query", tok, body)
    q = json.loads(raw)
    assert st == 200 and q["ok"] and q["hits"], f"{st} {raw[:600]}"
    assert EMAIL not in raw and CUSTOM not in raw, "scrub must cover every field, paths included"
    return q


def scenario(name, keep_recall):
    print(f"--- {name} ---")
    tmp = Path(tempfile.mkdtemp(prefix=f"ui_e2e_{name}_"))
    # The custom term is in the HOME path too: errors and paths that name it must come back scrubbed.
    home, vault = str(tmp / f"{CUSTOM}-home"), str(tmp / "vault")
    Path(home, ".recall").mkdir(parents=True)
    Path(vault).mkdir(parents=True)
    Path(vault, "note.md").write_text(f"# zebrafoo note\n\nzebrafoo contact {EMAIL} about the plan\n", encoding="utf-8")
    Path(vault, f"{CUSTOM}-plan.md").write_text("# zebrafoo plan\n\nzebrafoo launch checklist\n", encoding="utf-8")
    e = env(home, vault, keep_recall)
    Path(home, ".recall", "settings.json").write_text(json.dumps({"scrub": {"custom": [CUSTOM]}}))
    if keep_recall:
        shutil.copytree(LIVE_MODELS / "models--Xenova--bge-small-en-v1.5", Path(home, ".recall", "models", "models--Xenova--bge-small-en-v1.5"))
        assert Path(EXE).with_name("onnxruntime.dll").exists() or os.name != "nt", "onnxruntime.dll missing beside the exe"

    p, port, tok = start_ui(e)
    daemon = None
    try:
        # GET status on a HOME with no DB: reports it, creates nothing.
        st, _, raw = call(port, "GET", "/api/status", token=tok)
        s = json.loads(raw)
        assert st == 200 and s["store"] is None and "no database" in s["store_error"], s
        assert CUSTOM not in raw and "[CUSTOM]" in raw, "db path in the error is scrubbed"
        a_raw = call(port, "GET", "/api/activity", token=tok)[2]
        assert CUSTOM not in a_raw and CUSTOM not in call(port, "GET", "/api/settings", token=tok)[2]
        assert not Path(home, ".recall", "recall.db").exists(), "GET /api/status created a database"
        print("status before index: reports no database, created nothing")

        run(e, "index")
        if keep_recall:
            # A throwaway vault has no release benchmark artifact for the embed gate to compare to.
            run(e, "embed", "--allow-benchmark-regression")

        assert call(port, "GET", "/api/status")[0] == 403, "no token"
        assert call(port, "GET", "/api/status", token="0" * 64)[0] == 403, "wrong token"
        assert call(port, "GET", "/")[0] == 403, "page without ?t="
        assert call(port, "GET", "/api/status", token=tok, host=f"evil.example:{port}")[0] == 403, "rebinding Host"
        assert call(port, "GET", "/api/query", token=tok)[0] == 405
        st, h, page = call(port, "GET", f"/?t={tok}")
        assert st == 200 and "<title>recall</title>" in page
        assert not re.search(r"(src|href)\s*=\s*[\"']?https?:", page, re.I), "no CDN / external assets"
        assert not any(k.startswith("access-control-") for k in h), h
        print("403 without token / wrong token / bad Host; 405 GET on POST route; no CORS: OK")

        s = json.loads(call(port, "GET", "/api/status", token=tok)[2])
        assert s["store"]["docs_valid"] >= 2 and s["daemon"] == "down", s
        print("status:", s["version"], "proto", s["wire_proto"], "daemon", s["daemon"],
              "docs", s["store"]["docs_valid"], "coverage", s["store"]["coverage_pct"])

        st, _, raw = call(port, "POST", "/api/settings", tok, {"key": "refresh", "value": "every 2 hours"})
        assert st == 200 and json.loads(raw)["value"] == "every 2 hours", raw
        assert call(port, "POST", "/api/settings", tok, {"key": "refresh", "value": "whenever"})[0] == 400
        got = {r["key"]: r["value"] for r in json.loads(call(port, "GET", "/api/settings", token=tok)[2])["settings"]}
        assert got["refresh"] == "every 2 hours" and got["scrub.custom"] == "1 term set (hidden)", got
        assert run(e, "settings", "get", "refresh").stdout.strip() == "every 2 hours"
        print("settings round trip (UI and CLI agree):", got["refresh"], "|", got["scrub.custom"])

        if not keep_recall:
            # vault: set from the UI, then used with RECALL_VAULT unset; the env var still wins.
            assert call(port, "POST", "/api/settings", tok, {"key": "vault", "value": str(tmp / "nope")})[0] == 400
            st, _, raw = call(port, "POST", "/api/settings", tok, {"key": "vault", "value": vault})
            assert st == 200, raw
            nv = {k: v for k, v in e.items() if k != "RECALL_VAULT"}
            sj = json.loads(run(nv, "status", "-j").stdout)
            assert sj["vault_source"] == "settings" and Path(sj["vault"]) == Path(vault), sj
            assert json.loads(run(e, "status", "-j").stdout)["vault_source"] == "env"
            Path(vault, "added.md").write_text("# zebrafoo added\n\nzebrafoo third note\n", encoding="utf-8")
            r = run(nv, "refresh", "--force", "-j")
            assert '"new"' in r.stdout or "scan_new" in r.stdout, r.stdout[:300]
            run(nv, "index")
            assert json.loads(run(nv, "status", "-j").stdout)["docs_valid"] >= 3
            print("vault setting: status says 'settings' without RECALL_VAULT, 'env' with it; refresh + index used it")

        st, _, raw = call(port, "GET", "/api/activity", token=tok)
        a = json.loads(raw)
        assert st == 200 and a["index_runs"] and a["stale"] is not None, raw[:600]
        run0 = a["index_runs"][0]
        assert run0["files_seen"] >= 2 and run0["enum_ms"] is not None, run0
        assert a["refresh"]["setting"] == "every 2 hours" and a["refresh"]["next_due"], a["refresh"]
        assert a["daemon"]["up"] is False and len(a["not_recorded"]) == 4 and CUSTOM not in raw
        assert call(port, "GET", "/api/activity")[0] == 403
        print(f"activity: {len(a['index_runs'])} index run(s), last {run0['ran_at']} files={run0['files_seen']} enum_ms={run0['enum_ms']}; "
              f"stale={a['stale']}; next refresh due {a['refresh']['next_due']}; not recorded: {len(a['not_recorded'])} fields")

        body = {"q": "zebrafoo", "top": 5} if keep_recall else {"q": "zebrafoo", "channels": "fts5", "top": 5}
        q = search(port, tok, body)
        paths = [h["vault_path"] for h in q["hits"]]
        print(f"query ({q['path']}, {q['ms']} ms): {len(q['hits'])} hit(s), sources={[h['source'] for h in q['hits']]}, paths={paths}")
        assert q["path"] == "in-process"
        assert any("[CUSTOM]" in x for x in paths), "the custom term in a FILENAME is scrubbed"

        if keep_recall:
            daemon = subprocess.Popen([EXE, "daemon", "up"], env=dict(e, RECALL_NO_WATCHER="1"),
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(600):
                if json.loads(call(port, "GET", "/api/status", token=tok)[2])["daemon"] == "up":
                    break
                time.sleep(0.5)
            else:
                raise AssertionError("throwaway daemon never came up")
            q = search(port, tok, {"q": "zebrafoo", "top": 5})
            assert q["path"] == "daemon", q["path"]
            print(f"query via throwaway daemon ({q['ms']} ms): {len(q['hits'])} hit(s), sources={[h['source'] for h in q['hits']]}")
            # A query that fails shows its real error text, not a bare "failed".
            st, _, raw = call(port, "POST", "/api/query", tok, {"q": "zebrafoo", "channels": "nosuchchannel"})
            r = json.loads(raw)
            assert st == 200 and not r["ok"] and r["error"] and "nosuchchannel" in r["error"], raw
            print("bad channel error shown to the page:", ascii(r["error"].splitlines()[-1][:120]))
            a = json.loads(call(port, "GET", "/api/activity", token=tok)[2])
            assert a["daemon"]["up"] and a["daemon"]["uptime_s"] is not None, a["daemon"]
            assert a["errors"] and "nosuchchannel" in a["errors"][0]["error"], a["errors"][:1]
            print(f"activity with daemon: uptime {a['daemon']['uptime_s']} s; newest logged error: {ascii(a['errors'][0]['error'][:80])}")
        print(f"{name}: PASS")
    finally:
        if daemon:
            subprocess.run([EXE, "daemon", "down"], env=e, capture_output=True, timeout=30)
            try:
                daemon.wait(timeout=15)
            except subprocess.TimeoutExpired:
                daemon.kill()  # this Popen's own handle, i.e. by PID
        p.kill()
        p.wait(timeout=10)


def setup_scenario():
    """First-run setup through the page's API, with RECALL_VAULT unset."""
    print("--- setup ---")
    tmp = Path(tempfile.mkdtemp(prefix="ui_e2e_setup_"))
    home, vault = tmp / f"{CUSTOM}-home", tmp / "notes"
    Path(home, ".recall").mkdir(parents=True)
    for i in range(4):
        Path(vault, "sub").mkdir(parents=True, exist_ok=True)
        Path(vault, "sub" if i % 2 else "", f"n{i}.md").write_text(f"# kestrel note {i}\n\nkestrel field log {i}\n", encoding="utf-8")
    for d, name in [("proj/bower_components/pkg", "README.md"), ("disabled-old", "x.md")]:
        Path(vault, d).mkdir(parents=True)
        Path(vault, d, name).write_text("# kestrel vendored\n\nkestrel\n", encoding="utf-8")
    Path(vault, "pic.png").write_bytes(b"\x89PNG")
    Path(vault, "paper.pdf").write_bytes(b"%PDF")
    control = tmp / "control"  # same files, no own .git: a parent repo's ignore rules apply
    shutil.copytree(vault, control)
    # Its own repo, like a real notes folder: rg stops applying parent ignore rules here.
    subprocess.run(["git", "init", "-q", str(vault)], check=True)
    shutil.copytree(LIVE_MODELS / "models--Xenova--bge-small-en-v1.5", Path(home, ".recall", "models", "models--Xenova--bge-small-en-v1.5"))
    e = env(str(home), str(vault), keep_recall=True)
    e.pop("RECALL_VAULT")
    Path(home, ".recall", "settings.json").write_text(json.dumps({"scrub": {"custom": [CUSTOM]}}))
    p, port, tok = start_ui(e)
    try:
        st, _, raw = call(port, "POST", "/api/setup/scan", tok, {"folder": str(vault)})
        pv = json.loads(raw)
        assert st == 200 and CUSTOM not in raw, raw[:400]
        assert (pv["will_index"], pv["other_files"], pv["excluded_total"], pv["hidden_by_ignore"]) == (5, 2, 1, 0), pv
        assert [v["pattern"] for v in pv["vendor"]] == ["bower_components"] and pv["excluded"][0]["pattern"] == "disabled-", pv
        assert not Path(home, ".recall", "recall.db").exists(), "scan wrote a database"
        print(f"scan preview: will_index={pv['will_index']} vendor={pv['vendor']} excluded={pv['excluded']} other={pv['other_files']}")
        assert call(port, "POST", "/api/setup/scan", tok, {"folder": str(tmp / "nope")})[0] == 400
        cv = json.loads(call(port, "POST", "/api/setup/scan", tok, {"folder": str(control)})[2])
        print(f"control (inside a parent repo that ignores '*'): on disk {cv['notes_on_disk']}, rg sees "
              f"{cv['will_index'] + cv['excluded_total']}, hidden_by_ignore={cv['hidden_by_ignore']}")
        if Path(os.environ.get("USERPROFILE", ""), ".gitignore").exists():
            assert cv["hidden_by_ignore"] > 0, cv

        st, _, raw = call(port, "POST", "/api/setup/start", tok, {"folder": str(vault), "skip": ["bower_components"]})
        assert st == 200, raw
        assert call(port, "POST", "/api/setup/start", tok, {"folder": str(vault)})[0] == 409, "second start while running"
        t0 = time.time()
        while True:
            j = json.loads(call(port, "GET", "/api/setup/job", token=tok)[2])
            if j["state"] != "running" or time.time() - t0 > 300:
                break
            time.sleep(1)
        assert j["state"] == "done" and all(s["state"] == "done" for s in j["steps"]), j
        assert CUSTOM not in json.dumps(j)
        print(f"setup job: {j['state']} in {time.time() - t0:.1f} s; steps={[(s['name'], s['state'], s['ms']) for s in j['steps']]}")

        s = json.loads(call(port, "GET", "/api/status", token=tok)[2])
        assert s["store"]["docs_valid"] == 4 and s["store"]["coverage_pct"] == 100.0, s
        assert s["vault_source"] == "settings", s
        sj = json.loads(run(e, "status", "-j").stdout)
        assert Path(sj["vault"]) == vault and sj["vault_source"] == "settings", sj
        assert "bower_components" in Path(home, ".recall", "exclude.txt").read_text()
        q = search(port, tok, {"q": "kestrel field log", "top": 5})
        assert all("bower_components" not in h["vault_path"] for h in q["hits"])
        print(f"after setup: docs={s['store']['docs_valid']} coverage={s['store']['coverage_pct']} vault_source=settings; "
              f"search {len(q['hits'])} hit(s) via {q['path']}")
    finally:
        p.kill()
        p.wait(timeout=10)

    # Control: RECALL_VAULT pointing elsewhere outranks the setting, so setup must refuse.
    e2 = dict(e, RECALL_VAULT=str(tmp))
    p, port, tok = start_ui(e2)
    try:
        st, _, raw = call(port, "POST", "/api/setup/start", tok, {"folder": str(vault)})
        assert st == 409 and "RECALL_VAULT" in raw, raw
        print("setup refused while RECALL_VAULT points elsewhere: 409")
    finally:
        p.kill()
        p.wait(timeout=10)
    print("setup: PASS")


if __name__ == "__main__":
    if "--setup-only" not in sys.argv:
        scenario("clean", keep_recall=False)
        if "--clean-only" not in sys.argv:
            scenario("real", keep_recall=True)
    if "--clean-only" not in sys.argv:
        setup_scenario()
    print("E2E PASS")
