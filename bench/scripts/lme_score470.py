"""Score recall runs on the 470 non-_abs questions with gold = answer_session_ids (the protocol behind the
published MemPalace / gbrain / Grove LongMemEval_S numbers), next to the official 419-question filter.

usage: python lme_score470.py [--official-gold] DATA.json RESULTS.jsonl [RESULTS2.jsonl ...]
--official-gold: gold = sessions holding a user turn with has_answer, only the 419 questions that have one
(the official run_retrieval.py filter); default gold = answer_session_ids on all 470.
Questions that have a full:bge run (the dense subset) are also reported as their own slice for every run.
Ranked ids in results are mapped back to haystack session ids ('noans' -> 'answer' undo), so a run from
lme.py (user-turn docs) and one from lme_full.py (user+assistant docs) score identically.
"""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from lme_eval_utils import evaluate_retrieval  # noqa: E402

KS = [5, 10]


def main():
    args = [x for x in sys.argv[1:] if x != "--official-gold"]
    official = "--official-gold" in sys.argv
    data = {x["question_id"]: x for x in json.load(open(args[0], encoding="utf-8"))}
    out = {}
    failed = set()
    nonok = [0]
    seen = {}
    dense = set()
    for rp in args[1:]:
        for l in open(rp, encoding="utf-8"):
            if l.strip() and '"full:bge"' in l:
                dense.add(json.loads(l)["question_id"])
    for rp in args[1:]:
        res = list({r["question_id"]: r for r in map(json.loads, (l for l in open(rp, encoding="utf-8") if l.strip()))}.values())
        for r in res:
            if r.get("build_err"):
                failed.add(r["question_id"])
            e = data[r["question_id"]]
            if r["question_id"].endswith("_abs") or r.get("build_err"):
                continue
            sids = e["haystack_session_ids"]
            back = {s.replace("answer", "noans"): s for s in sids}
            gold = list(set(e["answer_session_ids"]) & set(sids))
            user_ev = any(t.get("has_answer") for s in e["haystack_sessions"] for t in s if t["role"] == "user")
            if official:
                if not user_ev:
                    continue
                gold = [sid for sid, s in zip(sids, e["haystack_sessions"])
                        if any(t.get("has_answer") for t in s if t["role"] == "user")]
            for run, v in r["runs"].items():
                if not run.startswith(("noabs:", "full:")):
                    continue
                idx = {c: i for i, c in enumerate(sids)}
                bad = len(sids)  # placeholder corpus slot: an unknown id holds its rank but is never relevant
                ranked = v["ranked"] if v.get("status", "ok") == "ok" else []  # failed query = nothing found
                if v.get("status", "ok") != "ok":
                    nonok[0] += 1
                rk = [idx.get(back.get(c, c), bad) for c in ranked]
                row = {"type": e["question_type"], "user_ev": user_ev, "dense": r["question_id"] in dense}
                for k in KS:
                    ra, rl, _ = evaluate_retrieval(rk, gold, sids + ['__unmapped__'], k=k)
                    row[f"any@{k}"], row[f"all@{k}"] = ra, rl
                assert (run, r["question_id"]) not in seen, f"{run}: {r['question_id']} appears in more than one input file"
                seen[(run, r["question_id"])] = 1
                out.setdefault(run, []).append(row)
    expected = sum(1 for q, e in data.items() if not q.endswith("_abs") and (not official or any(
        t.get("has_answer") for s in e["haystack_sessions"] for t in s if t["role"] == "user")))
    if nonok[0]:
        print(f"WARNING: {nonok[0]} non-ok query rows were scored as nothing found")
    if failed:
        print(f"WARNING: {len(failed)} questions had build errors and are NOT in the rows below: {sorted(failed)[:5]}...")
    for run, rows in sorted(out.items()):
        if len(rows) != expected:
            print(f"NOTE {run}: {len(rows)} of {expected} expected questions present (a subset run, not the full set)")
        for label, sel in ((f"all {expected} ({'official gold' if official else 'gold=answer_session_ids'})", rows),
                           ("419 subset (user-side evidence)", [r for r in rows if r["user_ev"]]),
                           ("51 assistant-evidence-only", [r for r in rows if not r["user_ev"]]),
                           (f"dense subset ({len(dense)} q)", [r for r in rows if r["dense"]])):
            if not sel:
                continue
            m = {k: float(np.mean([r[k] for r in sel])) for k in ("any@5", "all@5", "any@10", "all@10")}
            print(f"{run:16} {label:34} n={len(sel):3} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))
        by = defaultdict(list)
        for r in rows:
            by[r["type"]].append(r)
        for t, rr in sorted(by.items()):
            print(f"{run:16}   {t:28} n={len(rr):3} " + " ".join(
                f"{k}={np.mean([r[k] for r in rr]):.3f}" for k in ("any@5", "all@5", "any@10", "all@10")))


if __name__ == "__main__":
    main()
