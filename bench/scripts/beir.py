r"""BEIR retrieval (nDCG@10) of a recall build. No LLM anywhere.

Protocol (BEIR test split, standard):
  doc  = "# <title>\n\n<text>" (title "untitled" when empty), one .md file per corpus doc, opaque stems
  query = every query id that has a row in qrels/test.tsv; score = nDCG@10 with linear gain (trec_eval
  ndcg_cut_10), recall@10, MRR@10; every test query stays in the denominator (a missing/failed run scores 0)
  per query: drop the doc whose id equals the query id (BEIR ignore_identical_ids), drop repeats, cut to 10
  settings: abstain OFF (this measures ranking), top 11 requested, channels fts5 / bge / fts5+bge
usage:
  python beir.py run <dataset>...      build the store (index + embed) and query it   -> $BENCH_WORK/beir/<ds>/
  python beir.py score <dataset>...    nDCG@10 etc. from the raw run files            -> metrics.json + score.txt
  python beir.py all <dataset>...
  --channels fts5,bge,fts5+bge   --no-embed (fts5 only)   --max-queries N (smoke)
"""
import json
import math
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb  # noqa: E402

SEED = 20261030
TOP = 11
K = 10
ALL_CHANNELS = "fts5,bge,fts5+bge"


def W():
    return os.path.join(pb.WORK, "beir")


def log(m):
    os.makedirs(W(), exist_ok=True)
    line = time.strftime("%H:%M:%S ") + m
    with open(os.path.join(W(), "beir.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def ddir(ds):
    if not os.path.exists(os.path.join(pb.DATA, "beir", ds, ".verified")):
        raise SystemExit(f"{ds}: extraction not verified; run fetch_data.py beir_{ds}")
    return os.path.join(pb.DATA, "beir", ds, ds)


def corpus(ds):
    docs = {}
    with open(os.path.join(ddir(ds), "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs[str(d["_id"])] = (d.get("title") or "", d.get("text") or "")
    return docs


def queries(ds):
    q = {}
    with open(os.path.join(ddir(ds), "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            q[str(d["_id"])] = d["text"]
    return q


def qrels(ds):
    out = {}
    with open(os.path.join(ddir(ds), "qrels", "test.tsv"), encoding="utf-8") as f:
        assert f.readline().lower().startswith("query-id")
        for line in f:
            if line.strip():
                a, b, c = line.rstrip("\n").split("\t")
                out.setdefault(a, {})[b] = int(c)
    return out


def serialize(title, text):
    t = " ".join(title.replace("\0", "").split())
    if t.startswith("#"):
        t = "\\" + t
    return "# " + (t or "untitled") + "\n\n" + text.replace("\r\n", "\n").replace("\r", "\n").replace("\0", "") + "\n"


def build_vault(ds, vault):
    docs = corpus(ds)
    ids = sorted(docs)
    random.Random(SEED).shuffle(ids)
    os.makedirs(vault)
    man = {}
    for i, bid in enumerate(ids):
        stem = "d%06d" % (i + 1)
        with open(os.path.join(vault, stem + ".md"), "w", encoding="utf-8", newline="\n") as f:
            f.write(serialize(*docs[bid]))
        man[stem] = bid
    return man


def run_ds(ds, channels, embed, maxq):
    base = os.path.join(W(), ds)
    home, vault = os.path.join(base, "home"), os.path.join(base, "vault")
    mpath = os.path.join(base, "manifest.json")
    cpus = pb.cpu_all()
    if not os.path.exists(os.path.join(base, "build.json")):  # build.json = build complete; anything else is a partial build
        if os.path.isdir(base):
            shutil.rmtree(base)
        os.makedirs(base, exist_ok=True)
        pb.pin_identity(os.path.join(base, "identity.json"), {"dataset": ds, "embed": embed})  # before any build work
        man = build_vault(ds, vault)
        json.dump(man, open(mpath, "w"))
        pb.wait_heavy(log)
        t = time.time()
        res = pb.build_store(home, vault, cpus=cpus, embed=embed, log=log)
        res["seconds"] = time.time() - t
        res["docs"] = len(man)
        json.dump(res, open(os.path.join(base, "build.json"), "w"), indent=1)
        log(f"built {ds}: {len(man)} docs in {res['seconds']:.0f}s")
    pb.pin_identity(os.path.join(base, "identity.json"), {"dataset": ds, "embed": embed})
    man = json.load(open(mpath))
    qs, qr = queries(ds), qrels(ds)
    qids = sorted(qr)[:maxq] if maxq else sorted(qr)
    path = os.path.join(base, "run.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            if r["status"] == "ok":
                done.add((r["qid"], r["channel"]))
    d = pb.Daemon(home, vault, False, os.path.join(base, "daemon"), cpus=cpus)
    try:
        with open(path, "a", encoding="utf-8") as f:
            for n, q in enumerate(qids):
                for ch in channels:
                    if (q, ch) in done:
                        continue
                    r = d.query(qs[q], ch.replace("+", ","), TOP)
                    st, hits = pb.parse_hits(r)
                    ids = []
                    if st == "ok":
                        for h in hits:
                            s = pb.stem_of(h)
                            if s not in man:
                                st = "invalid_mapping"
                                break
                            ids.append(man[s])
                    f.write(json.dumps({"qid": q, "channel": ch, "status": st, "ms": r["ms"], "ids": ids,
                                        "sources": [h.get("source") for h in hits]}) + "\n")
                f.flush()
                if n % 200 == 0:
                    log(f"  {ds} {n}/{len(qids)}")
    finally:
        d.stop()
    log(f"queried {ds}")


def prepare(ranked, qid):
    seen, out = set(), []
    for d in ranked:
        if d == qid or d in seen:
            continue
        seen.add(d)
        out.append(d)
    return out[:K]


def per_query(rel, lst):
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(lst) if rel.get(d, 0) > 0)
    ideal = sorted((v for v in rel.values() if v > 0), reverse=True)[:K]
    idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
    npos = sum(1 for v in rel.values() if v > 0)
    rec = sum(1 for d in lst if rel.get(d, 0) > 0) / npos if npos else 0.0
    mrr = next((1.0 / (i + 1) for i, d in enumerate(lst) if rel.get(d, 0) > 0), 0.0)
    return (dcg / idcg if idcg else 0.0), rec, mrr


def score_ds(ds, maxq=0):
    qr = qrels(ds)
    qids = sorted(qr)[:maxq] if maxq else sorted(qr)
    runs = {}
    for l in open(os.path.join(W(), ds, "run.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        runs.setdefault(r["channel"], {})[r["qid"]] = r  # last record per key wins, whatever its status
    out = {}
    for ch, allr in runs.items():
        rr = {q: v for q, v in allr.items() if v["status"] == "ok"}  # failed/missing queries stay in the denominator as 0
        rows = [per_query(qr[q], prepare(rr[q]["ids"], q) if q in rr else []) for q in qids]
        n = len(rows)
        ms = sorted(rr[q]["ms"] for q in qids if q in rr)
        out[ch] = {"ndcg@10": sum(r[0] for r in rows) / n, "recall@10": sum(r[1] for r in rows) / n,
                   "mrr@10": sum(r[2] for r in rows) / n, "n_queries": n,
                   "n_failed": sum(1 for q in qids if q not in rr),
                   "empty_results": sum(1 for q in qids if q in rr and not rr[q]["ids"]),
                   "p50_ms": ms[len(ms) // 2] if ms else None}
    return out


def main():
    a = sys.argv[1:]
    cmd = a[0]
    channels, mq, embed, rest = ALL_CHANNELS, 0, True, []
    i = 1
    while i < len(a):
        if a[i] == "--no-embed":
            embed = False
        elif a[i] == "--channels":
            channels = a[i + 1]
            i += 1
        elif a[i] == "--max-queries":
            mq = int(a[i + 1])
            i += 1
        else:
            rest.append(a[i])
        i += 1
    channels = channels.split(",")
    failed = []
    for ds in rest:
        try:
            if cmd in ("run", "all"):
                run_ds(ds, channels, embed, mq)
            if cmd in ("score", "all"):
                score_and_write(ds, mq)
        except Exception as ex:  # one dataset failing must not stop the others; the exit code reports it
            log(f"FAILED {ds}: {type(ex).__name__}: {str(ex)[-300:]}")
            failed.append(ds)
    if failed:
        raise SystemExit(f"failed datasets: {failed}")


def score_and_write(ds, mq):
    m = score_ds(ds, mq)
    base = os.path.join(W(), ds)
    json.dump(m, open(os.path.join(base, "metrics.json"), "w"), indent=1)
    with open(os.path.join(base, "score.txt"), "w") as f:
        for ch, v in m.items():
            f.write(f"{ds:10} {ch:9} nDCG@10={v['ndcg@10']:.4f} recall@10={v['recall@10']:.4f} "
                    f"MRR@10={v['mrr@10']:.4f} n={v['n_queries']} failed={v['n_failed']} "
                    f"empty={v['empty_results']} p50_ms={v['p50_ms'] if v['p50_ms'] is None else round(v['p50_ms'])}\n")
    print(open(os.path.join(base, "score.txt")).read())


if __name__ == "__main__":
    main()
