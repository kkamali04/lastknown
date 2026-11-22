"""Merge full-mode result files: order of the first file, and a question present in a later file takes that record.
(lme_full pass 1 embedded 150 questions, pass 2 embedded the other 320; the merge has all 470 dense.)
usage: python lme_merge.py OUT.jsonl FIRST.jsonl [LATER.jsonl ...]
"""
import json
import sys

out, files = sys.argv[1], sys.argv[2:]
recs = [{json.loads(l)["question_id"]: json.loads(l) for l in open(f, encoding="utf-8") if l.strip()} for f in files]
with open(out, "w", encoding="utf-8", newline="\n") as o:
    for qid in recs[0]:
        o.write(json.dumps(next(r[qid] for r in reversed(recs) if qid in r)) + "\n")
