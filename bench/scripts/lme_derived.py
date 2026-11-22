"""Derived "should say I don't know" sets built from LongMemEval_S. No LLM; deterministic (seed 20261101).

These are OUR checks, not part of the official benchmark. Output: scratch\\public-bench\\lme\\derived\\
  answer_deleted.jsonl  - every answerable question (official filter) with its evidence sessions removed
                          (all haystack sessions listed in answer_session_ids).
  subject_swap.jsonl    - the question's key term replaced by a term that occurs in NO session of its haystack.
Key term rule: alphabetic question token, >= 4 letters, not a stopword, that occurs in the user text of an
evidence session; pick the one with the highest IDF over that question's haystack (ties: first in question).
Questions with no such term are skipped (listed with reason). Replacement: first term of a seeded shuffle of
REPLACEMENTS that occurs (case-insensitive, whole word) in none of the question's haystack sessions.
usage: python lme_derived.py   (rewrites both files; same input -> byte-identical output)
"""
import json
import math
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lme  # noqa: E402

OUT = lme.derived_dir()
SEED = 20261101
VERSION = 1
WORD = re.compile(r"[A-Za-z]+")
STOP = set("""about above after again against also because been before being below between both could
did does doing down during each from further have having here hers herself himself into itself just
like more most myself once only other ours ourselves over same should some such than that their theirs
them themselves then there these they this those through under until very what when where which while
whom with would your yours yourself yourselves will much many make made want need know think tell
remember recall mentioned mention previous previously last first time times days weeks months years
year month week ever back still recently currently current total long ago chat talked talking said
told asked going doing get got take took kind sort""".split())
REPLACEMENTS = ["Zanzibar", "Okonkwo", "theremin", "Valparaiso", "Krakatoa", "Lindqvist", "Ouagadougou",
                "kumquat", "Tbilisi", "Szczecin", "hurdy-gurdy", "Nakagawa", "axolotl", "Reykjanes",
                "Wojciechowski", "quokka", "Paramaribo", "sousaphone", "Thibodeaux", "Ulaanbaatar"]


def words(text):
    return {w.lower() for w in WORD.findall(text)}


def main():
    data = lme.load()
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)
    repl = REPLACEMENTS[:]
    rng.shuffle(repl)
    ad, ss, skipped = [], [], []
    for e in sorted(data, key=lambda x: x["question_id"]):
        qid = e["question_id"]
        if "_abs" in qid:
            continue
        docs = lme.session_docs(e)
        if not any("answer" in c for c, _ in docs):  # official filter: needs a user-side evidence turn
            continue
        ev = set(e["answer_session_ids"])
        ad.append({"question_id": qid, "question_type": e["question_type"], "question": e["question"],
                   "removed_sessions": sorted(s for s in e["haystack_session_ids"] if s in ev)})
        sess_words = [words(t) for _, t in docs]
        ev_words = set().union(*[w for (c, _), w in zip(docs, sess_words) if "answer" in c])
        n = len(docs)
        best = None
        for tok in WORD.findall(e["question"]):
            lw = tok.lower()
            if len(lw) < 4 or lw in STOP or lw not in ev_words:
                continue
            df = sum(lw in w for w in sess_words)
            idf = math.log((n + 1) / (df + 0.5))
            if best is None or idf > best[1]:
                best = (tok, idf)
        if best is None:
            skipped.append({"question_id": qid, "reason": "no evidence-held content term in question"})
            continue
        all_words = set().union(*sess_words)
        s = rng.randrange(len(repl))
        rep = next((r for r in repl[s:] + repl[:s] if not (words(r) & all_words)), None)
        if rep is None:
            skipped.append({"question_id": qid, "reason": "every replacement occurs in haystack"})
            continue
        newq = re.sub(r"\b%s\b" % re.escape(best[0]), rep, e["question"], count=1)
        ss.append({"question_id": qid, "question_type": e["question_type"], "question": e["question"],
                   "swapped_term": best[0], "replacement": rep, "new_question": newq})
    meta = {"version": VERSION, "seed": SEED, "data_sha256": lme.DATA_SHA, "n_answer_deleted": len(ad),
            "n_subject_swap": len(ss), "skipped_subject_swap": skipped}
    for name, rows in (("answer_deleted.jsonl", ad), ("subject_swap.jsonl", ss)):
        with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "skipped_subject_swap"}), len(skipped), "skipped")


if __name__ == "__main__":
    main()
