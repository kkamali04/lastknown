"""Create bench/results/<name>/ from a finished run: raw outputs, score file, exact command, binary sha256, commit.

usage: python package_result.py NAME --cmd "<exact command line(s)>" --binary PATH_OR_SHA --commit SHA
           --host "<hardware / OS>" --file SRC[:DEST] [--file ...] [--note TEXT]
Writes MANIFEST.json (with the sha256 of every copied file) next to the copied files.
"""
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv):
    name = argv[0]
    g = lambda k: [argv[i + 1] for i, x in enumerate(argv) if x == k]  # noqa: E731
    out = os.path.join(HERE, "..", "results", name)
    os.makedirs(out, exist_ok=True)
    b = g("--binary")[0]
    files = {}
    for spec in g("--file"):
        src, _, dest = spec.partition("::")
        dest = dest or os.path.basename(src)
        os.makedirs(os.path.dirname(os.path.join(out, dest)), exist_ok=True)
        shutil.copyfile(src, os.path.join(out, dest))
        files[dest] = sha(os.path.join(out, dest))
    m = {"name": name, "command": g("--cmd")[0], "binary_sha256": sha(b) if os.path.isfile(b) else b,
         "recall_commit": g("--commit")[0], "host": g("--host")[0], "note": (g("--note") or [""])[0], "files": files}
    json.dump(m, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
