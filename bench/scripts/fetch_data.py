"""Download the public datasets listed in datasets.lock into $BENCH_DATA and verify every sha256.

usage: python fetch_data.py [name ...]      (no names = everything of kind raw)
BEIR zips are also extracted to $BENCH_DATA/beir/<dataset>/<dataset>/ (zip-slip checked).
A sha mismatch is fatal: the file is deleted and nothing is extracted.
"""
import os
import shutil
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb  # noqa: E402

LOCK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets.lock")


def rows():
    for l in open(LOCK, encoding="utf-8"):
        if l.strip() and not l.startswith("#"):
            n, kind, url, sha, size, lic, local = l.rstrip("\n").split("\t")
            yield {"name": n, "kind": kind, "url": url, "sha": sha, "local": local}


def fetch(r):
    dst = os.path.join(pb.DATA, r["local"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not (os.path.exists(dst) and pb.sha256(dst) == r["sha"]):
        print("download", r["name"], flush=True)
        req = urllib.request.Request(r["url"], headers={"User-Agent": "recall-bench/1"})
        with urllib.request.urlopen(req) as resp, open(dst, "wb") as f:
            while True:
                b = resp.read(1 << 20)
                if not b:
                    break
                f.write(b)
        got = pb.sha256(dst)
        if got != r["sha"]:
            os.remove(dst)
            raise SystemExit(f"{r['name']}: sha256 {got} != locked {r['sha']}")
    print("ok", r["name"], r["sha"][:12])
    if r["name"].startswith("beir_"):
        ds = r["name"][5:]
        out = os.path.join(pb.DATA, "beir", ds)
        marker = os.path.join(out, ".verified")
        want = r["sha"]
        if not (os.path.exists(marker) and open(marker).read().strip() == want):
            tmp = out + ".tmp"
            if os.path.isdir(tmp):
                shutil.rmtree(tmp)
            if os.path.isdir(out):
                shutil.rmtree(out)  # unverified or stale extraction: redo from the verified zip
            with zipfile.ZipFile(dst) as z:
                for n in z.namelist():
                    assert ".." not in n and not n.startswith(("/", "\\")), n
                z.extractall(tmp)
            open(os.path.join(tmp, ".verified"), "w").write(want)
            os.rename(tmp, out)  # published only when complete


if __name__ == "__main__":
    want = set(sys.argv[1:])
    for r in rows():
        if (r["kind"] == "raw" and not want) or r["name"] in want:  # "raw-large" entries are downloaded only when named
            fetch(r)
