r"""BEAM (Tavakoli et al., arXiv 2510.27246) retrieval of a recall build. No LLM anywhere.

Data: mohammadtavakoli78/BEAM at a pinned commit (code MIT; the chats are the CC BY-SA 4.0 dataset
Mohammadta/BEAM, measured here and never redistributed). Per conversation: chats/<tier>/<n>/chat.json and
chats/<tier>/<n>/probing_questions/probing_questions.json. Tiers: 100K (20 chats), 500K (35), 1M (35).

THIS IS A DERIVED RETRIEVAL METRIC, NOT THE OFFICIAL BEAM SCORE. Official BEAM scoring is an LLM judge over rubrics
(src/evaluation/compute_metrics.py), which needs a generator and a judge model; we run neither. What we measure:
  unit  = one chat message (user or assistant), "<role>: <content>", file per message, opaque stems
  query = the probing question text
  gold  = the message ids in the question's `source_chat_ids` (flattened; dict values such as
          {'first_statement': [58], 'second_statement': [24]} are merged)
  metric = recall@k (fraction of gold messages in the top k) and any@k, k in 5, 10, 20, per ability and overall
  excluded: `abstention` (no gold by design) and questions with empty source_chat_ids (counted, not scored)
Because the generator wrote `source_chat_ids` for each question, a message that merely repeats the fact elsewhere is
not gold; recall here is a lower bound on "the answer was reachable".

usage:
  python beam.py fetch [--tiers 100K,500K,1M] [--write-lock]
  python beam.py run   [--tiers ...] [--dense 100K=20,500K=6,1M=4] [--channels fts5,bge,fts5+bge]
  python beam.py score
Every conversation gets an fts5 run; the first N (seeded shuffle) per tier also get embed + bge + fts5+bge.
"""
import ast
import hashlib
import json
import os
import random
import shutil
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb  # noqa: E402

COMMIT = "b2da22eac88bb0874c64665f13457eb99835774a"
RAW = f"https://raw.githubusercontent.com/mohammadtavakoli78/BEAM/{COMMIT}/"
LOCK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "beam.lock")
N_CHATS = {"100K": 20, "500K": 35, "1M": 35}
# Conversations whose chat.json repeats message ids (a later batch restarts numbering at 0), so `source_chat_ids` cannot be
# mapped to one message. Excluded from every number and listed in RESULTS.md. Found by checking all 90 conversations;
# run_conv asserts the duplication is still there, so the list cannot go stale silently.
EXCLUDED_DUP_IDS = {("1M", 5), ("1M", 26), ("1M", 33), ("1M", 34)}
KS = [5, 10, 20]
SEED = 20261103
TOP = 20


def W():
    return os.path.join(pb.WORK, "beam")


def log(m):
    os.makedirs(W(), exist_ok=True)
    line = time.strftime("%H:%M:%S ") + m
    with open(os.path.join(W(), "beam.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def files(tier):
    for n in range(1, N_CHATS[tier] + 1):
        yield f"chats/{tier}/{n}/chat.json"
        yield f"chats/{tier}/{n}/probing_questions/probing_questions.json"


def read_lock():
    out = {}
    if os.path.exists(LOCK):
        for l in open(LOCK, encoding="utf-8"):
            if l.strip() and not l.startswith("#"):
                p, sha, size = l.rstrip("\n").split("\t")
                out[p] = (sha, int(size))
    return out


def fetch(tiers, write_lock):
    lock = read_lock()
    new = dict(lock)
    for tier in tiers:
        for p in files(tier):
            dst = os.path.join(pb.DATA, "beam", p)
            if not (os.path.exists(dst) and p in lock and pb.sha256(dst) == lock[p][0]):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with urllib.request.urlopen(urllib.request.Request(RAW + p, headers={"User-Agent": "recall-bench/1"})) as r, \
                        open(dst, "wb") as f:
                    shutil.copyfileobj(r, f)
                sha = pb.sha256(dst)
                if p in lock and sha != lock[p][0]:
                    os.remove(dst)
                    raise SystemExit(f"{p}: sha256 {sha} != locked {lock[p][0]}")
                new[p] = (sha, os.path.getsize(dst))
        log(f"fetched {tier}")
    if write_lock:
        with open(LOCK, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"# BEAM files at commit {COMMIT}: path<TAB>sha256<TAB>bytes (data is CC BY-SA 4.0; never committed)\n")
            for p in sorted(new):
                f.write(f"{p}\t{new[p][0]}\t{new[p][1]}\n")
    elif any(p not in lock for tier in tiers for p in files(tier)):
        raise SystemExit("files missing from beam.lock; run fetch --write-lock once and commit the lock")


def messages(chat):
    """[(id, role, content)] in chat order. chat.json = list of batches -> turns -> messages carrying a global `id`."""
    out = []
    for batch in chat:
        for turn in batch["turns"]:
            for m in turn:
                out.append((int(m["id"]), m["role"], m["content"]))
    return out


def flat_ids(x):
    if isinstance(x, str):
        x = ast.literal_eval(x)
    if isinstance(x, dict):
        return sorted({i for v in x.values() for i in flat_ids(v)})
    if isinstance(x, (list, tuple)):
        return sorted({i for v in x for i in flat_ids(v)})
    return [int(x)]


def questions(pq):
    """[(ability, question, gold ids)] - abstention excluded."""
    out = []
    for ability, qs in pq.items():
        if ability == "abstention":
            continue
        for q in qs:
            out.append((ability, q["question"], flat_ids(q["source_chat_ids"]) if q.get("source_chat_ids") else []))
    return out


def run_conv(tier, n, dense, channels, embed):
    base = os.path.join(W(), tier, str(n))
    d = os.path.join(pb.DATA, "beam", "chats", tier, str(n))
    lock = read_lock()
    for rel in (f"chats/{tier}/{n}/chat.json", f"chats/{tier}/{n}/probing_questions/probing_questions.json"):
        assert rel in lock and pb.sha256(os.path.join(pb.DATA, "beam", rel)) == lock[rel][0], f"{rel} does not match beam.lock"
    chat = json.load(open(os.path.join(d, "chat.json"), encoding="utf-8"))
    qs = questions(json.load(open(os.path.join(d, "probing_questions", "probing_questions.json"), encoding="utf-8")))
    msgs = messages(chat)
    ids = [m[0] for m in msgs]
    if (tier, n) in EXCLUDED_DUP_IDS:
        assert len(set(ids)) != len(ids), f"{tier}/{n} is listed as having duplicate ids but does not"
        log(f"excluded {tier}/{n}: chat.json repeats message ids ({len(ids)} messages, {len(set(ids))} distinct ids)")
        return
    assert len(set(ids)) == len(ids), f"{tier}/{n}: message ids are not unique"
    home, vault = os.path.join(base, "home"), os.path.join(base, "vault")
    cpus = pb.cpu_all()
    out_path = os.path.join(base, "results.jsonl")
    pb.pin_identity(os.path.join(base, "identity.json"), {"tier": tier, "conv": n, "embed": embed, "dense": dense, "commit": COMMIT})
    if not os.path.exists(os.path.join(base, "build.json")):
        if os.path.isdir(base):
            shutil.rmtree(base)
        os.makedirs(vault)
        stems = {}
        for i, (mid, role, content) in enumerate(msgs):
            stem = "m%05d" % (i + 1)
            stems[stem] = mid
            with open(os.path.join(vault, stem + ".md"), "w", encoding="utf-8", newline="\n") as f:
                f.write(f"{role}: {content}".replace("\0", "") + "\n")
        json.dump(stems, open(os.path.join(base, "stems.json"), "w"))
        pb.pin_identity(os.path.join(base, "identity.json"), {"tier": tier, "conv": n, "embed": embed, "dense": dense, "commit": COMMIT})
        pb.wait_heavy(log)
        t = time.time()
        res = pb.build_store(home, vault, cpus=cpus, embed=dense and embed, log=log)
        res.update(seconds=time.time() - t, messages=len(msgs), questions=len(qs))
        json.dump(res, open(os.path.join(base, "build.json"), "w"), indent=1)
        log(f"built {tier}/{n}: {len(msgs)} messages, dense={dense}, {res['seconds']:.0f}s")
    stems = json.load(open(os.path.join(base, "stems.json")))
    chans = channels if dense else ["fts5"]
    daemon = pb.Daemon(home, vault, False, os.path.join(base, "daemon"), cpus=cpus)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for qi, (ability, text, gold) in enumerate(qs):
                for ch in chans:
                    r = daemon.query(text, ch.replace("+", ","), TOP)
                    st, hits = pb.parse_hits(r)
                    f.write(json.dumps({"tier": tier, "conv": n, "qi": qi, "ability": ability, "run": ch, "status": st,
                                        "gold": gold, "dense": dense, "ms": r["ms"],
                                        "ranked": [stems.get(pb.stem_of(h), -1) for h in hits]}) + "\n")
    finally:
        daemon.stop()
    log(f"queried {tier}/{n}")


def run(tiers, dense_n, channels, embed):
    for tier in tiers:
        order = list(range(1, N_CHATS[tier] + 1))
        random.Random(SEED).shuffle(order)
        dense_set = set(order[:dense_n.get(tier, 0)]) - {n for t, n in EXCLUDED_DUP_IDS if t == tier}  # order fixed before exclusions
        for n in range(1, N_CHATS[tier] + 1):
            run_conv(tier, n, n in dense_set, channels, embed)


def cell(rows):
    sc = [dict(r, ranked=r["ranked"] if r["status"] == "ok" else []) for r in rows if r["gold"]]  # failed query = nothing found, stays in the denominator
    c = {"n_scored": len(sc), "n_no_gold": sum(not r["gold"] for r in rows), "errors": sum(r["status"] != "ok" for r in rows)}
    for k in KS:
        def rec(r):
            top = set(r["ranked"][:k])
            return sum(g in top for g in r["gold"]) / len(r["gold"])
        c[f"recall@{k}"] = sum(map(rec, sc)) / len(sc) if sc else None
        c[f"any@{k}"] = sum(bool(set(r["gold"]) & set(r["ranked"][:k])) for r in sc) / len(sc) if sc else None
    return c


def score():
    rows = []
    for tier in N_CHATS:
        for n in range(1, N_CHATS[tier] + 1):
            p = os.path.join(W(), tier, str(n), "results.jsonl")
            if os.path.exists(p):
                rows += [json.loads(l) for l in open(p, encoding="utf-8")]
    rep = {}
    for tier in N_CHATS:
        for ch in ("fts5", "bge", "fts5+bge"):
            rr = [r for r in rows if r["tier"] == tier and r["run"] == ch]
            if not rr:
                continue
            t = {"all": cell(rr), "convs": len({r["conv"] for r in rr})}
            t["complete"] = t["convs"] == N_CHATS[tier] - sum(1 for tt, _ in EXCLUDED_DUP_IDS if tt == tier) if ch == "fts5" else True
            for ab in sorted({r["ability"] for r in rr}):
                t[ab] = cell([r for r in rr if r["ability"] == ab])
            rep[f"{tier}/{ch}"] = t
        # keyword channel restricted to the conversations that also have the dense channels = like-for-like row
        dense_convs = {r["conv"] for r in rows if r["tier"] == tier and r["run"] == "bge"}
        rr = [r for r in rows if r["tier"] == tier and r["run"] == "fts5" and r["conv"] in dense_convs]
        if rr:
            rep[f"{tier}/fts5 (dense convs only)"] = {"all": cell(rr), "convs": len(dense_convs)}
    json.dump(rep, open(os.path.join(W(), "metrics.json"), "w"), indent=1)
    with open(os.path.join(W(), "score.txt"), "w") as f:
        for key, t in rep.items():
            c = t["all"]
            if c["n_scored"]:
                if not t.get("complete", True):
                    f.write(f"{key:28} WARNING INCOMPLETE: {t['convs']} conversations present" + chr(10))
                f.write(f"{key:28} convs={t['convs']:2} n={c['n_scored']:4} " + " ".join(
                    f"R@{k}={c[f'recall@{k}']:.3f}" for k in KS) + " | any " + " ".join(
                    f"{c[f'any@{k}']:.3f}" for k in KS) + f" | errors={c['errors']} no_gold={c['n_no_gold']}\n")
    print(open(os.path.join(W(), "score.txt")).read())


if __name__ == "__main__":
    a = sys.argv[1:]
    g = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
    tiers = g("--tiers", "100K,500K,1M").split(",")
    if a[0] == "fetch":
        fetch(tiers, "--write-lock" in a)
    elif a[0] == "run":
        dn = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in g("--dense", "100K=20,500K=6,1M=4").split(",")}
        run(tiers, dn, g("--channels", "fts5,bge,fts5+bge").split(","), "--no-embed" not in a)
    else:
        score()
