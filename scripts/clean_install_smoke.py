"""After a clean `scripts/install.sh --vault ~/notes` / `install.ps1 -Vault $HOME\\notes`: drive recall the
way a new user would and fail loudly on any step. Standard library only. Used by the `clean-install`
CI job.

    python scripts/clean_install_smoke.py

Steps: the installer recorded ~/notes as the vault -> notes added there -> index -> embed -> query (the
right note first, from both channels) -> nonsense query (labelled or empty, never a crash) -> status
--json counts -> MCP handshake and tool calls through the installed venv and server copy, checking
their content -> `settings set vault` moves the vault. RECALL_VAULT is unset throughout: the installer
no longer pins it, so the setting is what a user's shell sees. Prints step times and disk use.
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

HOME = pathlib.Path(os.environ.get("USERPROFILE") or os.environ["HOME"])
RHOME = HOME / ".recall"
WIN = os.name == "nt"
RECALL = RHOME / "bin" / ("recall.exe" if WIN else "recall")
MCP_PY = RHOME / "venv" / ("Scripts/python.exe" if WIN else "bin/python")
MCP_SRV = RHOME / "mcp" / "recall_mcp.py"
HANDSHAKE = pathlib.Path(__file__).resolve().parent.parent / "mcp" / "smoke_handshake.py"
VAULT = HOME / "notes"  # what the README tells the user to pass
NOTES = {
    "garden.md": "# Tomato blight\nEarly blight shows as brown rings on the lower tomato leaves. Remove the "
                 "infected leaves, water at the base and rotate the bed next spring.\n",
    "backups.md": "# Backups\nThe database is copied every night with rclone to a B2 bucket and the copy is "
                  "restored once a month to prove it works.\n",
    "soup.md": "# Lentil soup\nOnion, carrot and red lentils, simmered for twenty minutes with cumin.\n",
}
ENV = {k: v for k, v in os.environ.items() if k != "RECALL_VAULT"}
ENV["RECALL_EXE"] = str(RECALL)
times = []


def step(name, fn):
    t = time.time()
    print(f"==> {name}", flush=True)
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - any failure fails the job, with its reason
        print(f"FAIL {name}: {e}", flush=True)
        sys.exit(1)
    times.append((name, time.time() - t))


def run(*args):
    p = subprocess.run([str(RECALL), *args], env=ENV, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=1800)
    if "panicked" in p.stderr:
        raise RuntimeError(f"recall {' '.join(args)} panicked:\n{p.stderr[-2000:]}")
    if p.returncode != 0:
        raise RuntimeError(f"recall {' '.join(args)} exited {p.returncode}:\n{p.stdout[-1500:]}\n{p.stderr[-1500:]}")
    return p


def same(a, b):
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def query(text):
    p = run("query", "--top", "3", "--json", "--", text)
    return json.loads(p.stdout or "[]"), p


def du(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def installed_vault():
    got = run("settings", "get", "vault").stdout.strip()
    if not got or not same(got, VAULT):
        raise RuntimeError(f"the installer recorded vault {got!r}, expected {VAULT}")
    for name, text in NOTES.items():
        (VAULT / name).write_text(text, encoding="utf-8")


def check_hit():
    hits, _ = query("how do I treat brown rings on tomato leaves")
    if not hits:
        raise RuntimeError("no hits for a phrase that is in garden.md")
    top = hits[0]
    if not top.get("vault_path", "").endswith("garden.md"):
        raise RuntimeError(f"garden.md is not the top hit: {[h.get('vault_path') for h in hits]}")
    if "bge" not in top.get("source", ""):
        raise RuntimeError(f"top hit did not come from the meaning channel (source={top.get('source')!r})")


def check_nonsense():
    hits, p = query("zxqvbn plorbt wuggleframe")
    label = (p.stderr + p.stdout).lower()
    if hits and not any(w in label for w in ("closest", "partial", "abstain", "no confident")):
        raise RuntimeError(f"a nonsense query returned unlabelled hits: {hits}")


def check_status():
    s = json.loads(run("status", "--json").stdout)
    if s.get("docs_valid") != len(NOTES) or s.get("chunk_embedded_docs") != len(NOTES):
        raise RuntimeError(f"status: docs_valid={s.get('docs_valid')} chunk_embedded_docs="
                           f"{s.get('chunk_embedded_docs')}, expected {len(NOTES)} each")


def health_ok(text):
    """recall_health's whole reply: status ran, and every channel probe says OK (none FAIL)."""
    lines = text.splitlines()
    probes = {ch: any(l.split()[:2] == [ch, "OK"] for l in lines) for ch in ("fts5", "bge", "fts5,bge")}
    return "status exit=0" in text and "FAIL" not in text and all(probes.values())


def check_mcp():
    results = pathlib.Path(tempfile.mkdtemp(prefix="recall-smoke-mcp-")) / "results.json"
    p = subprocess.run([str(MCP_PY), str(HANDSHAKE), "--server", str(MCP_SRV), "--query",
                        "tomato leaves brown rings", "--results", str(results)], env=ENV, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=900)
    print(p.stdout[-3000:])
    if p.returncode != 0:
        raise RuntimeError(f"MCP handshake exited {p.returncode}:\n{p.stderr[-2000:]}")
    full = json.loads(results.read_text(encoding="utf-8"))  # whole replies: the console copy is cut
    health, search = full["recall_health"]["text"], full["recall_search"]["text"]
    if not health_ok(health):
        raise RuntimeError(f"recall_health reports a problem:\n{health}")
    if "garden.md" not in search:
        raise RuntimeError(f"recall_search through MCP did not return garden.md:\n{search}")


def change_vault():
    other = pathlib.Path(tempfile.mkdtemp(prefix="recall-smoke-other-"))
    run("settings", "set", "vault", str(other))
    if not same(run("settings", "get", "vault").stdout.strip(), other):
        raise RuntimeError("settings set vault did not take")
    run("settings", "set", "vault", str(VAULT))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # echoes recall's output (note text, the scoring line's middle dot) on any console
    step("installer recorded the README vault; notes added", installed_vault)
    step("index", lambda: run("index"))
    step("embed", lambda: run("embed", "--allow-benchmark-regression"))
    step("query (expect garden.md, from bge)", check_hit)
    step("nonsense query (expect a label or nothing, no crash)", check_nonsense)
    step("status --json (3 notes, 3 embedded)", check_status)
    step("MCP handshake + tool calls (health ok, search finds garden.md)", check_mcp)
    step("settings set vault moves the vault", change_vault)
    print("\nstep times:")
    for name, s in times:
        print(f"  {s:7.1f}s  {name}")
    print(f"disk: ~/.recall {du(RHOME) / 2**20:.0f} MiB")
    print("CLEAN INSTALL SMOKE: ALL STEPS PASSED")


if __name__ == "__main__":
    main()
