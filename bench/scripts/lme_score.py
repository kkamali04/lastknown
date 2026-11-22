"""Score LongMemEval_S results with the official eval_utils, plus a same-machine BM25 reference.

BM25 reference = official run_retrieval.py 'flat-bm25' exactly: rank_bm25.BM25Okapi (source read from
dorianbrown/rank_bm25, Apache-2.0) over doc.split(" "), query.split(" "), np.argsort(scores)[::-1].
usage: python lme_score.py --out RESULTS_DIR [--bm25]   (RESULTS_DIR holds results.jsonl from lme.py --mode official)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lme_eval_utils import evaluate_retrieval  # noqa: E402

try:
    from rank_bm25 import BM25Okapi  # pip install rank_bm25==0.2.2 (Apache-2.0); only needed for the BM25 reference rows
except ImportError:
    BM25Okapi = None
KS = [1, 3, 5, 10]


def scored(e):
    """Official filter: not _abs, and some user turn has has_answer."""
    if "_abs" in e["question_id"]:
        return False
    return any(t.get("has_answer") for s in e["haystack_sessions"] for t in s if t["role"] == "user")


def metrics(ranked_ids, corpus_ids, correct):
    idx = {c: i for i, c in enumerate(corpus_ids)}
    rankings = [idx[c] for c in ranked_ids if c in idx]
    out = {}
    for k in KS:
        ra, rl, nd = evaluate_retrieval(rankings, correct, corpus_ids, k=k)
        out[f"recall_any@{k}"], out[f"recall_all@{k}"], out[f"ndcg_any@{k}"] = ra, rl, nd
    return out


def mean(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def main(lme=None, want_bm25=False):
    if lme is None:  # standalone: python lme_score.py [--exe PATH]
        import lme
        a = sys.argv[1:]
        lme.set_out(a[a.index("--out") + 1])
        want_bm25 = "--bm25" in a
    data = {x["question_id"]: x for x in lme.load()}
    rp = os.path.join(lme.W, "results.jsonl")
    res = [json.loads(l) for l in open(rp, encoding="utf-8") if l.strip()] if os.path.exists(rp) else []
    res = {r["question_id"]: r for r in res}  # a retried question appears twice; the later record wins
    report = {"n_results": len(res), "build_failed": [q for q, r in res.items() if r.get("build_err")]}
    per = {}
    bm25_all = []
    bm25_sub = []
    for qid, e in data.items():
        docs = lme.session_docs(e)
        cids = [c for c, _ in docs]
        correct = list({c for c in cids if "answer" in c})
        if want_bm25 and BM25Okapi is not None and scored(e):
            bm = BM25Okapi([t.split(" ") for _, t in docs])
            rank = np.argsort(bm.get_scores(e["question"].split(" ")))[::-1]
            m_bm = metrics([cids[i] for i in rank], cids, correct)
            bm25_all.append(m_bm)
            if qid in res:
                bm25_sub.append(m_bm)
        r = res.get(qid)
        if not r or r.get("build_err"):
            continue
        for run, v in r["runs"].items():
            row = {"qid": qid, "type": e["question_type"], "abs_q": "_abs" in qid, "scored": scored(e),
                   "empty": len(v["ranked"]) == 0, "status": v["status"], "ms": v["ms"]}
            row.update(metrics(v["ranked"], cids, correct))
            per.setdefault(run, []).append(row)
    keys = [f"recall_any@{k}" for k in KS] + [f"recall_all@{k}" for k in KS] + [f"ndcg_any@{k}" for k in KS]
    tab = {}
    for run, rows in sorted(per.items()):
        sc = [r for r in rows if r["scored"]]
        ab = [r for r in rows if r["abs_q"]]
        ans = [r for r in rows if not r["abs_q"]]
        tab[run] = {"n_scored": len(sc), **{k: mean(sc, k) for k in keys},
                    "errors": sum(r["status"] != "ok" for r in rows),
                    "empty_rate_answerable": mean([{"e": float(r["empty"])} for r in ans], "e"),
                    "empty_rate_abs_questions": mean([{"e": float(r["empty"])} for r in ab], "e"),
                    "n_abs_questions": len(ab), "p50_ms": float(np.median([r["ms"] for r in rows])),
                    "by_type": {t: {"n": len([r for r in sc if r["type"] == t]),
                                    **{k: mean([r for r in sc if r["type"] == t], k)
                                       for k in ("recall_any@5", "recall_all@5", "recall_any@10", "recall_all@10")}}
                                for t in sorted({r["type"] for r in sc})},
                    "abs_by_type_empty_rate": {t: mean([{"e": float(r["empty"])} for r in ab if r["type"] == t], "e")
                                               for t in sorted({r["type"] for r in ab})}}
    report["recall"] = tab
    report["bm25_ref_all500_scored"] = {"n": len(bm25_all), **{k: mean(bm25_all, k) for k in keys}}
    report["bm25_ref_same_subset"] = {"n": len(bm25_sub), **{k: mean(bm25_sub, k) for k in keys}}
    report["n_scored_total"] = sum(scored(e) for e in data.values())
    report["n_abs_total"] = sum("_abs" in q for q in data)
    # Abstention buckets on the as-shipped (abstain on) runs. Per query, from the top hit:
    # abstained = nothing returned; labelled = top hit carries `lacks` (shown as "closest match");
    # confident = top hit returned with no label. For deleted/swapped sets every confident answer is wrong.
    def buckets(rows):
        n = len(rows)
        if not n:
            return {"n": 0}
        a = sum(1 for r in rows if not r["ranked"])
        lab = sum(1 for r in rows if r["ranked"] and r["lacks"] and r["lacks"][0])
        return {"n": n, "abstained": a / n, "labelled": lab / n, "confident": (n - a - lab) / n}
    der = {}
    sets = (("official answerable (should answer)", "abs", lambda q: not q.endswith("_abs") and scored(data[q])),
            ("official _abs questions (should abstain)", "abs", lambda q: q.endswith("_abs")),
            ("derived A: answer deleted (should abstain)", "del", lambda q: True),
            ("derived B: subject swapped (should abstain)", "swap", lambda q: True))
    for name, tag, pick in sets:
        der[name] = {}
        for ch in ("fts5", "bge", "fts5+bge"):
            rows = [r["runs"][f"{tag}:{ch}"] for q, r in res.items()
                    if not r.get("build_err") and f"{tag}:{ch}" in r["runs"] and pick(q)]
            der[name][ch] = buckets(rows)
    report["abstention"] = der
    report["exe_sha256"] = lme.pb.exe_sha() if os.environ.get("RECALL_BIN") else "unknown"
    report["data_sha256"] = lme.DATA_SHA
    json.dump(report, open(os.path.join(lme.W, "metrics.json"), "w"), indent=1)
    write_report(lme, report)
    print(json.dumps(report, indent=1))


def write_report(lme, rep):
    """Plain table; published comparison numbers live in the research note, not here."""
    L = [f"# LongMemEval_S retrieval - recall build {rep['exe_sha256'][:8]}", "",
         f"Questions run: {rep['n_results']} of 500 (seed-20261101 order); scored (official filter): "
         f"see n per row; build failures: {len(rep['build_failed'])}", "",
         "| run | n | R_any@5 | R_all@5 | R_any@10 | R_all@10 | NDCG@10 | empty on answerable | empty on _abs (n) | p50 ms |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x: "-" if x != x else f"{x:.3f}"  # noqa: E731
    for run, t in rep["recall"].items():
        L.append(f"| recall {run} | {t['n_scored']} | {f(t['recall_any@5'])} | {f(t['recall_all@5'])} | "
                 f"{f(t['recall_any@10'])} | {f(t['recall_all@10'])} | {f(t['ndcg_any@10'])} | "
                 f"{f(t['empty_rate_answerable'])} | {f(t['empty_rate_abs_questions'])} ({t['n_abs_questions']}) | {t['p50_ms']:.0f} |")
    for name in ("bm25_ref_same_subset", "bm25_ref_all500_scored"):
        t = rep[name]
        L.append(f"| {name} | {t['n']} | {f(t['recall_any@5'])} | {f(t['recall_all@5'])} | {f(t['recall_any@10'])} | "
                 f"{f(t['recall_all@10'])} | {f(t['ndcg_any@10'])} | n/a | n/a | - |")
    L += ["", "## Per question type (scored questions)", "",
          "| run | type | n | R_any@5 | R_all@5 | R_any@10 | R_all@10 |", "|---|---|---|---|---|---|---|"]
    for run, t in rep["recall"].items():
        for ty, m in t["by_type"].items():
            L.append(f"| {run} | {ty} | {m['n']} | {f(m['recall_any@5'])} | {f(m['recall_all@5'])} | "
                     f"{f(m['recall_any@10'])} | {f(m['recall_all@10'])} |")
    L += ["", "## Abstention (abstain on). Official _abs set first; derived A/B are OUR checks, not official", "",
          "| set | channel | n | abstained | labelled closest-match | confident |", "|---|---|---|---|---|---|"]
    for name, chs in rep["abstention"].items():
        for ch, b in chs.items():
            if b.get("n"):
                L.append(f"| {name} | {ch} | {b['n']} | {f(b['abstained'])} | {f(b['labelled'])} | {f(b['confident'])} |")
    open(os.path.join(lme.W, "REPORT.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
