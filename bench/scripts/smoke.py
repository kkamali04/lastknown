"""CI smoke test: 20 LongMemEval_S questions, keyword channel (fts5) only, no model download, abstain off.
Fails when any build/query errors, when fewer than 15 questions were scored, or when R_any@5 falls under FLOOR.
usage: RECALL_BIN=path/to/recall python bench/scripts/smoke.py
Needs: python3 (stdlib + numpy), network for the 277 MB dataset (sha256-verified).
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pb  # noqa: E402

N = 20
FLOOR = float(os.environ.get("SMOKE_FLOOR", "0.80"))  # measured value is recorded in bench/RESULTS.md
OUT = os.path.join(pb.WORK, "lme", "smoke")


def sh(*args):
    subprocess.run([sys.executable, "-I", *args], check=True, env=dict(os.environ))


sh(os.path.join(HERE, "fetch_data.py"), "longmemeval_s_cleaned")
sh(os.path.join(HERE, "lme.py"), "--mode", "official", "--max", str(N), "--channels", "fts5", "--no-embed",
   "--workers", "2", "--out", OUT)
sh(os.path.join(HERE, "lme_score.py"), "--out", OUT)
m = json.load(open(os.path.join(OUT, "metrics.json")))
t = m["recall"]["noabs:fts5"]
print(json.dumps({k: t[k] for k in ("n_scored", "recall_any@5", "recall_all@5", "recall_any@10", "errors")}))
assert not m["build_failed"], m["build_failed"]
assert t["errors"] == 0, "query errors"
assert t["n_scored"] >= 15, f"only {t['n_scored']} scored"
assert t["recall_any@5"] >= FLOOR, f"R_any@5 {t['recall_any@5']:.3f} < floor {FLOOR}"
print("SMOKE OK")
