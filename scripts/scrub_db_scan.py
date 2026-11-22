"""Proof that `scrub.store_redacted` keeps secrets and PII out of recall.db, and that a hit still
points at the original.

Throwaway HOME and vault only; never touches ~/.recall or a real vault. The BGE model snapshot is
COPIED into the throwaway HOME (read-only use of the weights, no user data).

  1. store_redacted ON : index + embed, then scan every file under HOME/.recall (recall.db, -wal,
     -shm, anything else) as raw bytes, ASCII and UTF-16LE. Zero canary bytes may remain.
  2. store_redacted OFF: the same scan MUST find them (proves the scan can see what it looks for).
  3. Pointers: `query --json` hits carry `redactions: [{kind, line}]`; the line in the real file
     holds the canary; the JSON never does.

Usage: python -I scripts/scrub_db_scan.py <recall.exe> <onnxruntime.dll> <bge-model-dir>
"""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

EXE, ORT, MODEL = sys.argv[1], sys.argv[2], sys.argv[3]

# Built at runtime so this file never holds a live-looking key.
C = {
    "openai": "sk-proj-" + "Ab9Z1x7Q4m" * 3,
    "github": "ghp_" + "A1b2C3d4E5f6G7h8I9" * 2,
    "aws": "AKIA" + "QWERTYUIOP123456",
    "urlpw": "s3cr3tPassw0rdZ9",
    "email": "jordan.canary@example-corp.com",
    "phone": "(555) 867-5309",
    "card": "4111 1111 1111 1111",
    "ssn": "078-05-1120",
    "ip": "203.0.113.77",
    "pem": "MIIEowIBAAKCAQEA7bq98canaryPEMbody",
    "bearer": "9f8e7d6c5b4a39f8e7d6c5b4a3zz",
}
NEEDLES = {  # the distinctive bytes that must not survive (a fragment is enough to prove a leak)
    "openai": C["openai"], "github": C["github"], "aws": C["aws"], "urlpw": C["urlpw"],
    "email": C["email"], "email_local": "jordan.canary", "phone": C["phone"], "phone_frag": "867-5309",
    "card": C["card"], "card_flat": C["card"].replace(" ", ""), "ssn": C["ssn"], "ip": C["ip"],
    "pem": C["pem"], "bearer": C["bearer"],
    "late": "ghp_" + "Z9y8X7w6V5u4T3s2R1" * 2,  # planted by second_index(), after the first index
}
NOTES = {
    "alpha.md": "# Alpha ops\n\nkeyword quokkaterm one\nOPENAI note: {openai}\nbackup mail {email}\n",
    "beta.md": "# Beta billing\n\nquokkaterm two\nCard on file {card}\nSSN {ssn}\ncall {phone}\n",
    "gamma.md": "# Gamma infra\n\nquokkaterm three\nDB postgres://admin:{urlpw}@db.internal:5432/app\nhost {ip}\nAuthorization: Bearer {bearer}\n",
    "delta.md": "# Delta keys\n\nquokkaterm four\ngithub {github}\naws {aws}\n-----BEGIN RSA PRIVATE KEY-----\n{pem}\n-----END RSA PRIVATE KEY-----\n",
    "epsilon.md": "# Epsilon plain\n\nquokkaterm five has nothing sensitive, only words about retrieval and chunking.\n",
}


def env_for(home, vault, extra=None):
    base = {k: v for k, v in os.environ.items() if k != "RECALL_SCRUB_STORE_REDACTED"}  # an inherited pin would void the ON/OFF controls
    return dict(base, USERPROFILE=home, HOME=home, RECALL_VAULT=vault, ORT_DYLIB_PATH=ORT,
                RECALL_CONFIRM_RESET="yes", RECALL_MIN_EMBED_FREE_GB="0", **(extra or {}))


def run(home, vault, *args, check=True, extra=None):
    r = subprocess.run([EXE, *args], env=env_for(home, vault, extra), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900)
    if check and r.returncode != 0:
        raise SystemExit(f"recall {' '.join(args)} failed ({r.returncode}):\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
    return r


def lock(path):
    """Make settings.json impossible to replace (read-only on Windows, immutable on Linux). False if
    this machine cannot do it, so the caller skips instead of passing on a file that was writable."""
    os.chmod(path, 0o444)
    if os.name != "nt":
        subprocess.run(["chattr", "+i", str(path)], capture_output=True)
    probe = Path(str(path) + ".probe")
    probe.write_bytes(Path(path).read_bytes())
    try:
        os.replace(probe, path)
    except OSError:
        probe.unlink()
        return True
    return False


def unlock(path):
    if os.name != "nt":
        subprocess.run(["chattr", "-i", str(path)], capture_output=True)
    os.chmod(path, 0o644)


def build(tmp, name, store_redacted, extra=None, first=(), raw_settings=None, locked=False):
    home, vault = str(tmp / name / "home"), str(tmp / name / "vault")
    rec = Path(home, ".recall")
    rec.mkdir(parents=True)
    Path(vault).mkdir(parents=True)
    shutil.copytree(MODEL, rec / "models" / Path(MODEL).name)
    if store_redacted is not None:  # None = a brand-new install: no settings file at all
        (rec / "settings.json").write_text(json.dumps({"scrub": {"store_redacted": store_redacted}}))
    if raw_settings is not None:  # a settings file recall did not write
        (rec / "settings.json").write_text(raw_settings)
    if locked and not lock(rec / "settings.json"):
        return None
    for fn, body in NOTES.items():
        Path(vault, fn).write_text(body.format(**C), encoding="utf-8")
    for cmd in first:  # commands a user runs BEFORE the first index (they create settings.json / recall.db)
        run(home, vault, *[vault if a == "<vault>" else a for a in cmd], extra=extra)
    run(home, vault, "index", extra=extra)
    e = run(home, vault, "embed", "--allow-benchmark-regression", check=False, extra=extra)
    return home, vault, e


def second_index(home, vault, name):
    """A later run: one more note with a new planted value, then `index`. The first index being
    clean proves little if the next one stores everything raw again."""
    Path(vault, "late.md").write_text("# Late\n\nquokkaterm six\ngithub " + NEEDLES["late"] + "\n", encoding="utf-8")
    run(home, vault, "index")
    n = json.loads(run(home, vault, "status", "--json").stdout)["docs_valid"]
    assert n == len(NOTES) + 1, f"{name}: the second index did not store the new note ({n} docs)"
    found, _ = scan(home)
    assert not found, f"{name}: the SECOND index stored raw text: {found}"


def scan(home):
    """Raw-byte scan of every file under HOME/.recall except the model weights."""
    found, files = {}, []
    for f in Path(home, ".recall").rglob("*"):
        if not f.is_file() or "models" in f.parts:
            continue
        data = f.read_bytes()
        files.append((f.name, len(data)))
        for k, v in NEEDLES.items():
            if v.encode() in data or v.encode("utf-16-le") in data:
                found.setdefault(k, set()).add(f.name)
    return found, files


def main():
    tmp = Path(tempfile.mkdtemp(prefix="scrub_db_scan_"))
    # ---- 1. ON
    home, vault, emb = build(tmp, "on", True)
    print("embed(on) exit", emb.returncode, (emb.stdout + emb.stderr).strip().splitlines()[-1:] )
    assert emb.returncode == 0, "embed failed; cannot claim the embed path was scanned"
    found, files = scan(home)
    print("scanned (store_redacted ON):", files)
    assert not found, f"CANARY BYTES REMAIN in the index: { {k: sorted(v) for k, v in found.items()} }"
    print("PASS 1: zero canary bytes in", len(files), "files (ascii + utf-16le)")
    # embeddings really exist (otherwise the 'embed' half of the claim is vacuous)
    st = run(home, vault, "status", "--json").stdout
    stj = json.loads(st)
    assert stj["docs_valid"] == len(NOTES) and stj["chunk_embedded_docs"] == len(NOTES), f"embed incomplete: {stj}"
    print("status: all", stj["chunk_embedded_docs"], "docs have chunk vectors")

    # ---- 3. pointers (same ON home)
    q = run(home, vault, "query", "quokkaterm", "-j", "-c", "fts5,bge", "--no-daemon", "-k", "10").stdout
    for k, v in NEEDLES.items():
        assert v not in q, f"query JSON leaked {k}"
    hits = json.loads(q)
    by = {h["vault_path"]: h for h in hits}
    assert set(NOTES) <= set(by), f"missing hits: {set(NOTES) - set(by)}"
    assert any("bge" in h["source"] for h in hits), "no hit used the vector channel; embed proof would be vacuous"
    # (kind -> text that must sit on the pointed line) per note: pointers must land on the RIGHT line
    expect = {
        "alpha.md": {"API_KEY:openai": C["openai"], "EMAIL": C["email"]},
        "beta.md": {"CARD": C["card"], "SSN": C["ssn"], "PHONE": C["phone"]},
        "gamma.md": {"URL_CREDENTIALS": C["urlpw"], "IP": C["ip"], "TOKEN": C["bearer"]},
        "delta.md": {"API_KEY:github": C["github"], "API_KEY:aws": C["aws"], "PRIVATE_KEY": "BEGIN RSA PRIVATE KEY"},
    }
    for fn, kinds in expect.items():
        reds = by[fn].get("redactions", [])
        lines = Path(vault, fn).read_text(encoding="utf-8").splitlines()
        assert all(set(r) == {"kind", "line"} for r in reds), "pointer carries more than kind+line"
        got = {(r["kind"], r["line"]) for r in reds}
        for kind, text in kinds.items():
            want = next(i + 1 for i, l in enumerate(lines) if text in l)
            assert (kind, want) in got, f"{fn}: wanted {(kind, want)}, pointers say {sorted(got)}"
    assert "redactions" not in by["epsilon.md"], "clean note must carry no pointer"
    print("PASS 3: every sensitive note has {kind,line} pointers that land on the real canary line; JSON holds no value")
    # searching for the value finds nothing (the stated trade-off)
    miss = json.loads(run(home, vault, "query", C["email"], "-j", "-c", "fts5", "--no-daemon").stdout or "[]")
    assert all(h["vault_path"] != "alpha.md" for h in miss), "alpha.md still retrievable by the redacted value"
    print("trade-off confirmed: searching the redacted email does not surface alpha.md by its value")

    # ---- 2. OFF control
    home2, vault2, emb2 = build(tmp, "off", False)
    found2, _ = scan(home2)
    must = {"openai", "github", "aws", "email", "card", "ssn", "ip", "pem", "urlpw", "bearer", "phone_frag"}
    assert must <= set(found2), f"control scan is blind to: {must - set(found2)}"
    print("PASS 2 (control): with store_redacted OFF the same scan finds", sorted(found2))
    # ---- 4. NEW install (no settings.json, no db): redaction is ON by default
    home3, vault3, emb3 = build(tmp, "fresh", None)
    assert emb3.returncode == 0
    cfg = json.loads(Path(home3, ".recall", "settings.json").read_text())
    assert cfg["scrub"]["store_redacted"] is True, cfg
    found3, _ = scan(home3)
    assert not found3, f"fresh install leaked: {found3}"
    print("PASS 4: a new install defaults store_redacted ON and its index holds zero canary bytes")
    # ---- 5. benchmark pin: RECALL_SCRUB_STORE_REDACTED=0 beats the new-install default, and status says so
    pin = {"RECALL_SCRUB_STORE_REDACTED": "0"}
    home5, vault5, _ = build(tmp, "pinned", None, extra=pin)
    found5, _ = scan(home5)
    assert {"openai", "email", "ssn"} <= set(found5), "pinned-off index should hold raw text"
    sc = json.loads(run(home5, vault5, "status", "--json", extra=pin).stdout)["scrub"]
    assert sc["store_redacted"] is False and "env" in sc["store_redacted_source"], sc
    assert not Path(home5, ".recall", "settings.json").exists(), "env pin must not write a competing default"
    print("PASS 5: benchmark pin works and is reported:", sc)
    # ---- 6. the documented install order: something else creates the profile BEFORE the first index
    for name, first in [("vaultfirst", [("settings", "set", "vault", "<vault>")]), ("statusfirst", [("status", "--json")])]:
        home6, _, emb6 = build(tmp, name, None, first=first)
        assert emb6.returncode == 0
        cfg6 = json.loads(Path(home6, ".recall", "settings.json").read_text())
        assert cfg6["scrub"]["store_redacted"] is True, cfg6
        assert name != "vaultfirst" or cfg6.get("vault"), cfg6
        found6, _ = scan(home6)
        assert not found6, f"{name}: profile created before the first index leaked: {found6}"
    print("PASS 6: `settings set vault` or `status` before the first index still stores redacted text")
    # ---- 7. an install that indexed unredacted and never set the key keeps its value on re-index
    home7, vault7, _ = build(tmp, "legacy", False)
    Path(home7, ".recall", "settings.json").write_text("{}")  # as an older version left it: no scrub key
    Path(vault7, "alpha.md").write_text(NOTES["alpha.md"].format(**C) + "edited wombatmarker\n", encoding="utf-8")
    run(home7, vault7, "index")
    q7 = json.loads(run(home7, vault7, "query", "wombatmarker", "-j", "-c", "fts5", "--no-daemon").stdout or "[]")
    assert any(h["vault_path"] == "alpha.md" for h in q7), "the re-index did not store the edited note; case 7 would prove nothing"
    assert "scrub" not in json.loads(Path(home7, ".recall", "settings.json").read_text())
    found7, _ = scan(home7)
    assert must <= set(found7), f"an existing unredacted index was changed without being asked: {must - set(found7)}"
    print("PASS 7: an existing indexed profile with no scrub setting is left as it was")
    # ---- 8. nothing indexed and no readable choice: redact even when settings.json cannot hold the key
    broken = "{ not json"
    home8, vault8, emb8 = build(tmp, "brokenjson", None, raw_settings=broken)
    assert emb8.returncode == 0
    assert Path(home8, ".recall", "settings.json").read_text() == broken, "a file recall cannot read is not recall's to rewrite"
    found8, _ = scan(home8)
    assert not found8, f"brokenjson: a profile with nothing indexed stored raw text: {found8}"
    second_index(home8, vault8, "brokenjson")
    print("PASS 8: an unreadable settings.json on a profile with nothing indexed stores redacted text, first index and second")
    valid = json.dumps({"refresh": "nightly"})
    try:
        b9 = build(tmp, "locked", None, raw_settings=valid, locked=True)
        if b9 is None:
            print("SKIPPED 9: this machine cannot make settings.json unwritable (needs chattr +i on Linux)")
        else:
            home9, vault9, emb9 = b9
            assert emb9.returncode == 0
            assert Path(home9, ".recall", "settings.json").read_text() == valid
            found9, _ = scan(home9)
            assert not found9, f"locked: the settings write failed and the index stored raw text: {found9}"
            second_index(home9, vault9, "locked")
            assert Path(home9, ".recall", "settings.json").read_text() == valid
            print("PASS 9: a settings.json that cannot be written gets a redacted first index and second")
    finally:
        unlock(tmp / "locked" / "home" / ".recall" / "settings.json")
    # ---- 10. settings.json lost after the first index: the database remembers how the index started
    home10, vault10, _ = build(tmp, "settingslost", None)
    Path(home10, ".recall", "settings.json").unlink()
    second_index(home10, vault10, "settingslost")
    print("PASS 10: deleting settings.json after the first index does not turn redaction off")
    print("ALL PASS")


main()
