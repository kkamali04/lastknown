r"""LongMemEval_S retrieval-only run of a recall build (no LLM anywhere). Data: MIT (xiaowu0162/LongMemEval).

Two index modes, never to be mixed in one comparison (see bench/README.md):
  official  docs = user turns of one haystack session joined with " " (assistant turns dropped), as the official
            src/retrieval/run_retrieval.py does. Gold = sessions that hold a user turn with has_answer. Scored
            questions = not "_abs" and at least one such turn (419 of 500).
  full      docs = ALL turns (user + assistant) joined with "\n", as MemPalace's longmemeval_bench.py
            build_palace_and_retrieve_full. Gold = answer_session_ids. Scored = every non-_abs question (470).
Each question gets its own vault and a fresh recall.db in a per-worker scratch HOME, so nothing is shared.
Order = all 500 question ids shuffled with seed 20261101; any completed prefix is a random subset.
Record per question: runs["<tag>:<channel>"] = {status, ranked (session ids), ms, ...}. Score with lme_score.py.

LME_SPLIT=m selects LongMemEval_M (2.7 GB file, ~500 sessions / ~1.5M tokens per question; loads ~10+ GB of RAM).
usage (resumable; re-run to continue):
  python lme.py --mode official --max 500 --channels fts5,bge,fts5+bge [--abstain] [--derived] [--workers 8]
  python lme.py --mode full --channels fts5,bge,fts5+bge [--dense N]
  --no-embed   skip `recall embed` (only valid with --channels fts5)
  --out DIR    results dir (default $BENCH_WORK/lme/<binary sha8>-<mode>)
Launch under `nice -n 19 ionice -c3`; BENCH_CPUS pins one cpu per worker.
"""
import json
import os
import random
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb  # noqa: E402

SPLIT = os.environ.get("LME_SPLIT", "s")  # "s" = LongMemEval_S (~50 sessions/question), "m" = LongMemEval_M (~500)
assert SPLIT in ("s", "m"), "LME_SPLIT must be s or m"
DATA_SHA = {"s": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
            "m": "9d79e5524794a2e6900a3aa9cb7d9152c5a3e8319c9a87c25494ba1eacee495f"}[SPLIT]
SEED = 20261101
TOP = 10
W = None
HOME = None
_lock = threading.Lock()


def data_path():
    return os.path.join(pb.DATA, f"longmemeval_{SPLIT}_cleaned.json")


def derived_dir():
    return os.path.join(pb.WORK, "lme", "derived")


def set_out(out):
    global W, HOME
    W = out
    HOME = os.path.join(W, "homes")
    os.makedirs(W, exist_ok=True)


def log(m):
    line = time.strftime("%H:%M:%S ") + m
    with _lock:
        with open(os.path.join(W, "lme.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)


def load():
    assert pb.sha256(data_path()) == DATA_SHA, "longmemeval_s_cleaned.json sha mismatch (run fetch_data.py; LongMemEval_M is named: fetch_data.py longmemeval_m_cleaned)"
    return json.load(open(data_path(), encoding="utf-8"))


def order(data):
    ids = sorted(x["question_id"] for x in data)
    random.Random(SEED).shuffle(ids)
    return ids


def session_docs(e):
    """Official process_item_flat_index, session granularity ('answer' sessions with no has_answer user turn -> 'noans')."""
    out = []
    for sid, sess in zip(e["haystack_session_ids"], e["haystack_sessions"]):
        user = [t for t in sess if t["role"] == "user"]
        cid = sid
        if "answer" in sid and all(not t.get("has_answer") for t in user):
            cid = sid.replace("answer", "noans")
        out.append((cid, " ".join(t["content"] for t in user)))
    return out


def full_docs(e):
    return [(sid, "\n".join(t["content"] for t in sess)) for sid, sess in zip(e["haystack_session_ids"], e["haystack_sessions"])]


def write_vault(qid, docs):
    v = os.path.join(W, "vaults", qid)
    if os.path.isdir(v):
        shutil.rmtree(v)
    os.makedirs(v)
    stems = {}
    for i, (cid, text) in enumerate(docs):
        stem = "s%03d" % (i + 1)  # opaque: session ids contain the word 'answer'
        with open(os.path.join(v, stem + ".md"), "w", encoding="utf-8", newline="\n") as f:
            f.write(text.replace("\r\n", "\n").replace("\0", "") + "\n")
        stems[stem] = cid
    return v, stems


def worker_home(i):
    h = os.path.join(HOME, "w%d" % i)
    pb.make_home(h)
    return h


def query_all(rec, tag, text, abstain, home, vault, stems, channels, cpus):
    envq = pb.env_for(home, vault, abstain)
    for ch in channels:
        r = pb.run(["query", "--no-daemon", "--json", "--top", str(TOP), "--channels", ch.replace("+", ","), "--", text],
                   envq, timeout=600, cpus=cpus)
        st, hits = pb.parse_hits(r)
        rec["runs"][tag + ":" + ch] = {
            "status": st, "ranked": [stems.get(pb.stem_of(h), "UNMAPPED:" + pb.stem_of(h)) for h in hits],
            "ms": r["ms"], "abstain": [l for l in r["err"].splitlines() if l.startswith("abstain")][:1],
            "lacks": [h.get("lacks") for h in hits]}


def one(qid, e, home, cpus, a):
    full = a["mode"] == "full"
    docs = full_docs(e) if full else session_docs(e)
    vault, stems = write_vault(qid, docs)
    pb.fresh_db(home)
    env = pb.env_for(home, vault, False)
    t = time.time()
    dense = a["embed"] and (not full or qid in a["dense_ids"])
    ri = pb.run(["index"], env, cpus=cpus)
    re_ = pb.run(["embed", "--batch", "64", "--allow-benchmark-regression"], env, cpus=cpus) if dense else {"rc": 0, "err": ""}
    rec = {"question_id": qid, "question_type": e["question_type"], "n_sessions": len(docs), "build_s": time.time() - t,
           "index_rc": ri["rc"], "embed_rc": re_["rc"], "dense": dense, "runs": {}}
    if ri["rc"] or re_["rc"]:
        rec["build_err"] = (ri["err"] + re_["err"])[-800:]
        log(f"{qid} BUILD FAILED {rec['build_err'][-200:]}")
    elif full:
        query_all(rec, "full", e["question"], False, home, vault, stems, a["channels"] if dense else ["fts5"], cpus)
    else:
        query_all(rec, "noabs", e["question"], False, home, vault, stems, a["channels"], cpus)
        if a["abstain"]:
            query_all(rec, "abs", e["question"], True, home, vault, stems, a["channels"], cpus)
        if a["derived"] and qid in a["swap"]:
            query_all(rec, "swap", a["swap"][qid]["new_question"], True, home, vault, stems, a["channels"], cpus)
        if a["derived"] and qid in a["deleted"]:
            gone = set(a["deleted"][qid]["removed_sessions"])
            orig = {"s%03d" % (i + 1): sid for i, sid in enumerate(e["haystack_session_ids"])}
            for stem, sid in orig.items():
                if sid in gone:
                    os.remove(os.path.join(vault, stem + ".md"))
            # plain `index` does not tombstone missing files: prune explicitly (the shrink gate needs the override)
            rd = pb.run(["index", "--prune-missing=apply"], dict(env, RECALL_CONFIRM_PRUNE_SHRINK="yes"), cpus=cpus)
            rs = pb.run(["status"], env, cpus=cpus)
            live = int(rs["out"].split("docs (valid)")[1].split(":")[1].split()[0]) if "docs (valid)" in rs["out"] else -1
            rec["del_index_rc"], rec["del_valid_docs"] = rd["rc"], live
            rec["del_expected_docs"] = len(docs) - len([s for s in orig.values() if s in gone])
            if rd["rc"] == 0 and live == rec["del_expected_docs"]:
                query_all(rec, "del", e["question"], True, home, vault, stems, a["channels"], cpus)
    shutil.rmtree(vault, ignore_errors=True)
    return rec


def verify_derived():
    """Every derived file must hash to the value locked in datasets.lock before it is used."""
    lock = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets.lock")
    for l in open(lock, encoding="utf-8"):
        f = l.rstrip("\n").split("\t")
        if len(f) == 7 and f[1] == "derived":
            p = os.path.join(derived_dir(), os.path.basename(f[6]))
            assert pb.sha256(p) == f[3], f"{p} does not match datasets.lock ({f[0]})"


def required_runs(qid, a):
    """Exact run keys a finished question must hold (derived passes only when the question is in that set)."""
    if a["mode"] == "full":
        return {"full:" + c for c in (a["channels"] if qid in a["dense_ids"] else ["fts5"])}
    tags = ["noabs"] + (["abs"] if a["abstain"] else [])
    if a["derived"]:
        tags += (["swap"] if qid in a["swap"] else []) + (["del"] if qid in a["deleted"] else [])
    return {t + ":" + c for t in tags for c in a["channels"]}


def run(a):
    data = {x["question_id"]: x for x in load()}
    pb.pin_identity(os.path.join(W, "identity.json"), {
        "mode": a["mode"], "channels": a["channels"], "abstain": a["abstain"], "derived": a["derived"],
        "embed": a["embed"], "dense": a["dense"], "data_sha256": DATA_SHA, "split": SPLIT})
    ids = order(list(data.values()))
    if a["mode"] == "full":
        ids = [q for q in ids if not q.endswith("_abs")]
    a["dense_ids"] = set(ids[:a["dense"]]) if a["dense"] else set(ids)  # fixed by the canonical order, independent of --max
    ids = ids[:a["max"]]
    a["deleted"], a["swap"] = {}, {}
    if a["derived"]:
        import lme_derived
        if not os.path.exists(os.path.join(derived_dir(), "answer_deleted.jsonl")):
            lme_derived.main()
        verify_derived()
        for key, name in (("deleted", "answer_deleted.jsonl"), ("swap", "subject_swap.jsonl")):
            a[key] = {r["question_id"]: r for r in map(json.loads, open(os.path.join(derived_dir(), name), encoding="utf-8"))}
    out_path = os.path.join(W, "results.jsonl")
    done = set()
    if os.path.exists(out_path):
        # a question with a build error or a failed query is NOT done: it is retried and the later record wins in scoring
        done = set()
        for l in open(out_path, encoding="utf-8"):
            r = json.loads(l) if l.strip() else None
            if r and not r.get("build_err") and required_runs(r["question_id"], a) <= {k for k, v in r["runs"].items() if v["status"] == "ok"}:
                done.add(r["question_id"])
    todo = [q for q in ids if q not in done]
    log(f"run mode={a['mode']}: {len(done)} done, {len(todo)} to do, {a['workers']} workers, channels={a['channels']}")
    it = iter(todo)
    count = [len(done)]
    stop = threading.Event()

    def loop(i):
        home, cpus = worker_home(i), pb.cpu_for(i)
        while not stop.is_set():
            try:
                pb.wait_heavy(log)
            except SystemExit as ex:
                log(f"w{i} abort: {ex}")
                stop.set()
                return
            with _lock:
                qid = next(it, None)
            if qid is None:
                return
            rec = one(qid, data[qid], home, cpus, a)
            with _lock:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                count[0] += 1
                c = count[0]
            if c % 10 == 0:
                log(f"{c}: {qid} {rec['question_type']} build {rec['build_s']:.0f}s load1={pb.load1():.1f}")

    with ThreadPoolExecutor(a["workers"]) as ex:
        list(ex.map(loop, range(a["workers"])))
    log(f"run end; aborted={stop.is_set()}")


def parse(argv):
    g = lambda k, d: argv[argv.index(k) + 1] if k in argv else d  # noqa: E731
    a = {"mode": g("--mode", "official"), "max": int(g("--max", "500")), "workers": int(g("--workers", "8")),
         "channels": g("--channels", "fts5,bge,fts5+bge").split(","), "abstain": "--abstain" in argv,
         "derived": "--derived" in argv, "embed": "--no-embed" not in argv, "dense": int(g("--dense", "0"))}
    assert a["mode"] in ("official", "full")
    if not a["embed"]:
        assert a["channels"] == ["fts5"], "--no-embed only works with --channels fts5"
    assert SPLIT == "s" or not a["derived"], "derived abstain sets are built from LongMemEval_S only"
    set_out(g("--out", os.path.join(pb.WORK, "lme", pb.exe_sha()[:8] + "-" + a["mode"] + ("-M" if SPLIT == "m" else ""))))
    return a


if __name__ == "__main__":
    run(parse(sys.argv[1:]))
