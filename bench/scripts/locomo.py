r"""LoCoMo (locomo10.json) retrieval of a recall build. No LLM anywhere. Data is CC BY-NC 4.0: used here for
non-commercial measurement only and never redistributed.

Protocol (upstream snap-research/locomo task_eval/evaluation.py `eval_question_answering`):
  recall_acc of a question = (number of its evidence dia_ids found in the retrieved context) / (number of evidence ids)
  at session granularity an evidence id "D3:7" counts as found when session 3 was retrieved.
  We report the mean of recall_acc at several k, plus "any" (at least one evidence id found).
  Two numbers per cell: over questions that have evidence, and `official-style`, where a question with no
  evidence scores 1.0 exactly as the upstream script does (4 of 1986 questions).
  Evidence strings such as "D8:6; D9:17" are split with the regex D\d+:\d+ (the upstream code would not split them, so
  such a question can never score on those entries there; this is a deliberate, disclosed deviation). recall@k counts each
  DISTINCT evidence unit once; recall_upstream@k and recall_official_style@k keep upstream's multiplicity (at session
  granularity two dialog ids of one session count that session twice).
Units:
  dialog  - one document per turn, "<speaker>: <text>" (+ " [shares image: <blip caption>]"); k in 5,10,25,50
  session - one document per session, "Session <n>, <date>" header + one line per turn; k in 1,2,3,5,10
One store per conversation per granularity (10 conversations). Categories are numbered as in the data file:
1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial (unanswerable; upstream names).
Two passes: tag "noabs" = abstain OFF (ranking), tag "abs" = abstain ON (as shipped), both over ALL questions. Scored cells show
the abstain-ON effect as R@k with abstained questions counting as nothing found; `empty_rate` is the share that returned nothing.
usage: python locomo.py run | score  [--channels fts5,bge,fts5+bge] [--no-embed] [--convs N]
"""
import json
import os
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb  # noqa: E402

DATA_SHA = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
DIA = re.compile(r"D\d+:\d+")
KS = {"dialog": [5, 10, 25, 50], "session": [1, 2, 3, 5, 10]}
TOP = {"dialog": 50, "session": 10}


def W():
    return os.path.join(pb.WORK, "locomo")


def log(m):
    os.makedirs(W(), exist_ok=True)
    line = time.strftime("%H:%M:%S ") + m
    with open(os.path.join(W(), "locomo.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def load():
    p = os.path.join(pb.DATA, "locomo10.json")
    assert pb.sha256(p) == DATA_SHA, "locomo10.json sha mismatch"
    return json.load(open(p, encoding="utf-8"))


def units(conv, gran):
    """[(unit id, text)] - dialog: dia_id per turn; session: 'S<n>' per session."""
    c = conv["conversation"]
    out = []
    n = 1
    while f"session_{n}" in c:
        lines = []
        for t in c[f"session_{n}"]:
            text = f"{t['speaker']}: {t['text']}"
            if t.get("blip_caption"):
                text += f" [shares image: {t['blip_caption']}]"
            if gran == "dialog":
                out.append((t["dia_id"], text))
            lines.append(text)
        if gran == "session":
            out.append((f"S{n}", f"Session {n}, {c.get(f'session_{n}_date_time', '')}\n" + "\n".join(lines)))
        n += 1
    return out


def ev_units(evidence, gran):
    """Evidence entries in order, with multiplicity, as the upstream script counts them (session: each entry -> its session)."""
    return list(evidence) if gran == "dialog" else ["S" + e[1:].split(":")[0] for e in evidence]


def run(channels, embed, nconv):
    cpus = pb.cpu_all()
    out_path = os.path.join(W(), "results.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {(r["sample_id"], r["gran"], r["i"], r["run"]) for r in map(json.loads, open(out_path, encoding="utf-8"))
                if r["status"] == "ok"}  # failed rows are retried; score() keeps the last row per key
    for conv in load()[:nconv]:
        sid = conv["sample_id"]
        for gran in ("dialog", "session"):
            base = os.path.join(W(), sid, gran)
            home, vault = os.path.join(base, "home"), os.path.join(base, "vault")
            stems = {}
            if not os.path.exists(os.path.join(base, "build.json")):  # build.json = build complete; else partial, rebuild
                if os.path.isdir(base):
                    shutil.rmtree(base)
                os.makedirs(vault)
                pb.pin_identity(os.path.join(base, "identity.json"), {"sample": sid, "gran": gran, "embed": embed})
                for i, (uid, text) in enumerate(units(conv, gran)):
                    stem = "u%04d" % (i + 1)
                    stems[stem] = uid
                    with open(os.path.join(vault, stem + ".md"), "w", encoding="utf-8", newline="\n") as f:
                        f.write(text.replace("\0", "") + "\n")
                json.dump(stems, open(os.path.join(base, "stems.json"), "w"))
                pb.wait_heavy(log)
                res = pb.build_store(home, vault, cpus=cpus, embed=embed, log=log)
                json.dump(res, open(os.path.join(base, "build.json"), "w"), indent=1)
                log(f"built {sid} {gran}: {len(stems)} units")
            pb.pin_identity(os.path.join(base, "identity.json"), {"sample": sid, "gran": gran, "embed": embed})
            stems = json.load(open(os.path.join(base, "stems.json")))
            for abst in (False, True):
                tag = "abs" if abst else "noabs"
                d = pb.Daemon(home, vault, abst, os.path.join(base, "daemon"), cpus=cpus)
                try:
                    with open(out_path, "a", encoding="utf-8") as f:
                        for i, qa in enumerate(conv["qa"]):
                            ev = ev_units(DIA.findall(" ".join(qa.get("evidence", []))), gran)
                            for ch in channels:
                                rid = f"{tag}:{ch}"
                                if (sid, gran, i, rid) in done:
                                    continue
                                r = d.query(qa["question"], ch.replace("+", ","), TOP[gran])
                                st, hits = pb.parse_hits(r)
                                ranked = [stems.get(pb.stem_of(h), "UNMAPPED") for h in hits]
                                f.write(json.dumps({"sample_id": sid, "gran": gran, "i": i, "run": rid,
                                                    "category": qa["category"], "evidence": ev, "status": st,
                                                    "ranked": ranked, "ms": r["ms"]}) + "\n")
                        f.flush()
                finally:
                    d.stop()
                log(f"{sid} {gran} {tag} done")


def cell(rows, gran):
    n_ev = [r for r in rows if r["evidence"]]
    c = {"n": len(rows), "n_with_evidence": len(n_ev), "errors": sum(r["status"] != "ok" for r in rows),
         "empty_rate": (sum(not r["ranked"] for r in rows if r["status"] == "ok") / max(1, sum(r["status"] == "ok" for r in rows))
                        if rows else None)}  # errors are counted in `errors`, never as "returned nothing"
    for k in KS[gran]:
        def rec(r, dedup=True):
            # dedup (the reported recall@k): each distinct evidence unit counts once. 109 session questions list
            # the same session twice ("D3:1; D3:7" -> S3, S3) and upstream counts it twice; that number is kept as
            # recall_upstream@k (independent Codex recompute, 2026-11-06).
            ev = list(dict.fromkeys(r["evidence"])) if dedup else r["evidence"]
            top = set(r["ranked"][:k])
            return sum(e in top for e in ev) / len(ev)
        c[f"recall@{k}"] = sum(map(rec, n_ev)) / len(n_ev) if n_ev else None
        c[f"recall_upstream@{k}"] = sum(rec(r, False) for r in n_ev) / len(n_ev) if n_ev else None
        c[f"any@{k}"] = sum(bool(set(r["evidence"]) & set(r["ranked"][:k])) for r in n_ev) / len(n_ev) if n_ev else None
        c[f"recall_official_style@{k}"] = (sum(rec(r, False) for r in n_ev) + (len(rows) - len(n_ev))) / len(rows) if rows else None
    return c


def score():
    last = {}
    for l in open(os.path.join(W(), "results.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        last[(r["sample_id"], r["gran"], r["i"], r["run"])] = r  # a retry replaces the failed row
    rows = list(last.values())
    expected = sum(len(c["qa"]) for c in load())
    rep = {}
    for gran in ("dialog", "session"):
        for rid in sorted({r["run"] for r in rows}):
            rr = [r for r in rows if r["gran"] == gran and r["run"] == rid]
            if not rr:
                continue
            t = {"all": cell(rr, gran), "1-4": cell([r for r in rr if r["category"] != 5], gran), "expected_rows": expected,
                 "complete": len(rr) == expected}
            for cat in (1, 2, 3, 4, 5):
                sub = [r for r in rr if r["category"] == cat]
                if sub:
                    t[str(cat)] = cell(sub, gran)
            ms = sorted(r["ms"] for r in rr)
            t["p50_ms"] = ms[len(ms) // 2]
            rep[f"{gran}/{rid}"] = t
    json.dump(rep, open(os.path.join(W(), "metrics.json"), "w"), indent=1)
    with open(os.path.join(W(), "score.txt"), "w") as f:
        for key, t in rep.items():
            gran = key.split("/")[0]
            key = key.replace("/noabs:", "/abstain-OFF:").replace("/abs:", "/abstain-ON:")
            if not t["complete"]:
                f.write(f"{key:30} WARNING INCOMPLETE: {t['all']['n']} of {t['expected_rows']} questions present (subset or unfinished run)" + chr(10))
            for scope in ("1-4", "all"):
                c = t[scope]
                if c["n_with_evidence"] == 0:
                    continue
                f.write(f"{key:30} cats={scope:3} n={c['n']:4} " + " ".join(
                    f"R@{k}={c[f'recall@{k}']:.3f}" for k in KS[gran]) + " | any " + " ".join(
                    f"{c[f'any@{k}']:.3f}" for k in KS[gran]) + f" | errors={c['errors']}\n")
            if "5" in t:
                f.write(f"{key:30} cat5 (adversarial) returned nothing: {t['5']['empty_rate']:.3f} (n={t['5']['n']})\n")
            if t["1-4"]["n"]:
                f.write(f"{key:30} cats1-4 returned nothing: {t['1-4']['empty_rate']:.3f}\n")
    print(open(os.path.join(W(), "score.txt")).read())


if __name__ == "__main__":
    a = sys.argv[1:]
    ch = (a[a.index("--channels") + 1] if "--channels" in a else "fts5,bge,fts5+bge").split(",")
    nc = int(a[a.index("--convs") + 1]) if "--convs" in a else 99
    if a[0] == "run":
        run(ch, "--no-embed" not in a, nc)
    else:
        score()
