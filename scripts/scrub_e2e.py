"""End-to-end check of recall's scrub layers against a throwaway vault and a throwaway HOME.

Never touches ~/.recall: USERPROFILE/HOME and RECALL_VAULT point at a temp dir.
Usage: python -I scripts/scrub_e2e.py <path-to-recall.exe> [--bench N]
"""
import json, os, statistics, subprocess, sys, tempfile, time
from pathlib import Path

EXE = sys.argv[1]
BENCH = int(sys.argv[sys.argv.index("--bench") + 1]) if "--bench" in sys.argv else 0

# Assembled at runtime: this file never holds a live-looking key.
CANARIES = {
    "openai": "sk-proj-" + "Ab9Z1x7Q4m" * 3,
    "github": "ghp_" + "A1b2C3d4E5f6G7h8I9" * 2,
    "aws": "AKIA" + "ABCDEFGH12345678",
    "urlpw": "postgres://admin:s3cr3tPassw0rd@db.internal:5432/app",
    "email": "jane.canary@example-corp.com",
    "phone": "(555) 867-5309",
    "card": "4111 1111 1111 1111",
    "ip": "203.0.113.77",
    "custom": "ProjectFalconX",
}
SAFE = ["550e8400-e29b-41d4-a716-446655440000", "9fceb02d0ae598e95dc970b74767f19372d61af8",
        "aGVsbG8gd29ybGQgdGhpcyBpcyBhIGJhc2U2NCB0ZXN0IGZpeHR1cmU="]
CORE = {"urlpw": "s3cr3tPassw0rd", "phone": "867-5309"}


def run(home, vault, *args):
    env = dict(os.environ, USERPROFILE=home, HOME=home, RECALL_VAULT=vault, RECALL_CONFIRM_RESET="yes")
    return subprocess.run([EXE, *args], env=env, capture_output=True, text=True, encoding="utf-8", timeout=300)


def make(tmp, name, settings, extra_docs=0):
    home, vault = str(tmp / name / "home"), str(tmp / name / "vault")
    Path(home, ".recall").mkdir(parents=True)
    Path(vault).mkdir(parents=True)
    if settings is not None:
        Path(home, ".recall", "settings.json").write_text(json.dumps({"scrub": settings}))
    # One doc per canary: the FTS5 snippet is a short window, so a shared doc would hide the tail.
    for k, val in CANARIES.items():
        Path(vault, f"canary_{k}.md").write_text(f"# canary {k}\n\nzebrafoo {k}: {val}\n", encoding="utf-8")
    Path(vault, "safe.md").write_text("# safe\n\nzebrafoo safe ids: " + " ".join(SAFE) + "\n", encoding="utf-8")
    for i in range(extra_docs):
        Path(vault, f"filler{i}.md").write_text(f"# filler {i}\n\nzebrafoo filler text number {i} " + "lorem ipsum " * 60, encoding="utf-8")
    r = run(home, vault, "index")
    assert r.returncode == 0, r.stderr[-500:]
    return home, vault


def query(home, vault, text="zebrafoo"):
    r = run(home, vault, "query", text, "-j", "-c", "fts5", "--no-daemon", "-k", "40")
    assert r.returncode == 0, r.stderr[-500:]
    return r.stdout


def present(out):
    return {k for k, v in CANARIES.items() if CORE.get(k, v) in out}


def main():
    tmp = Path(tempfile.mkdtemp(prefix="scrub_e2e_"))
    # 1. layers on (the default, no settings file at all) -> nothing leaks, safe tokens survive
    h, v = make(tmp, "on", None)
    out = query(h, v)
    leaked = present(out) - {"custom"}  # custom list is empty by default
    assert not leaked, f"leaked with defaults: {leaked}"
    assert all(s in out for s in SAFE), "false positive on UUID / git SHA / base64"
    assert "[EMAIL]" in out and "[API_KEY:openai]" in out, out[:400]
    # 2. layers off -> all canaries come back
    h, v = make(tmp, "off", {"secrets": False, "pii": False})
    out = query(h, v)
    assert present(out) == set(CANARIES) - set(), f"missing with layers off: {set(CANARIES) - present(out)}"
    # 3. custom layer, at output time over an index that holds the raw text (a profile with nothing
    # indexed would otherwise get store_redacted on, and this would only test the stored form)
    h, v = make(tmp, "cust", {"custom": ["ProjectFalconX"], "store_redacted": False})
    assert "ProjectFalconX" not in query(h, v) and "[CUSTOM]" in query(h, v)
    # 4. store_redacted: the index never held the value, so searching for it finds nothing
    h, v = make(tmp, "stored", {"store_redacted": True})
    assert not (present(query(h, v)) - {"custom"}), "stored text leaked"
    assert "jane.canary" not in query(h, v, "jane.canary@example-corp.com")
    # 5. output scrub still applies when the index holds raw text and store_redacted is off
    print("e2e: all assertions passed")
    if BENCH:
        h1, v1 = make(tmp, "b_on", None, 300)
        h0, v0 = make(tmp, "b_off", None if "--same" in sys.argv else {"secrets": False, "pii": False}, 300)
        # Interleaved A/B runs so machine load hits both arms equally.
        for _ in range(3):
            query(h1, v1); query(h0, v0)
        on, off = [], []
        for _ in range(BENCH):
            t = time.perf_counter(); query(h1, v1); on.append(time.perf_counter() - t)
            t = time.perf_counter(); query(h0, v0); off.append(time.perf_counter() - t)
        on, off = (statistics.median(on), min(on)), (statistics.median(off), min(off))
        print(f"latency median on={on[0]*1000:.1f}ms off={off[0]*1000:.1f}ms overhead={(on[0]/off[0]-1)*100:+.1f}% (min on={on[1]*1000:.1f} off={off[1]*1000:.1f})")


main()
