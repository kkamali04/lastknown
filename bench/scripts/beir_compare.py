"""Compare two BEIR run.jsonl files with identical qrels and the historical TEST split.

Uses beir.py's unchanged ranking/scoring protocol. Missing/failed queries score zero.
No dataset text is written. Default split seed 20261112 reproduces the SciFact release row.

BENCH_DATA=/path/to/verified/data python3 bench/scripts/beir_compare.py scifact
    --run /path/to/new/run.jsonl --reference /path/to/old/run.jsonl --out comparison.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

import beir


def read_run(path):
    rows = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            rows.setdefault(row["channel"], {})[row["qid"]] = row
    return rows


def scores(rows, qrels, qids, channel):
    channel_rows = rows.get(channel, {})
    values, failed = [], 0
    for qid in qids:
        row = channel_rows.get(qid)
        ok = row is not None and row["status"] == "ok"
        failed += not ok
        ranked = beir.prepare(row["ids"], qid) if ok else []
        values.append(beir.per_query(qrels[qid], ranked)[0])
    return values, failed


def mean(values):
    return sum(values) / len(values)


def bootstrap(deltas, seed):
    rng = random.Random(seed)
    n = len(deltas)
    samples = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(5000))
    return [samples[125], samples[4875]]


def compare(current, reference, qrels, seed):
    result = {}
    for split in ("ALL", "TEST"):
        qids = [qid for qid in sorted(qrels) if split == "ALL" or
                int(hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest()[:8], 16) % 2 == 1]
        if not qids:
            raise ValueError(f"empty {split} split")
        channels = {}
        vectors = {}
        for channel in sorted(set(current) | set(reference)):
            new, failures = scores(current, qrels, qids, channel)
            old, old_failures = scores(reference, qrels, qids, channel)
            deltas = [a - b for a, b in zip(new, old)]
            channels[channel] = {"n": len(qids), "current_ndcg_at_10": mean(new),
                "reference_ndcg_at_10": mean(old), "delta": mean(deltas),
                "current_failed": failures, "reference_failed": old_failures,
                "per_query_scores_identical": all(a == b for a, b in zip(new, old))}
            vectors[channel] = new
        row = {"channels": channels}
        if "fts5+bge" in vectors and "fts5" in vectors:
            delta = [a - b for a, b in zip(vectors["fts5+bge"], vectors["fts5"])]
            row["current_fused_minus_keyword"] = {"delta": mean(delta), "ci95": bootstrap(delta, seed)}
        result[split] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20261112)
    args = parser.parse_args()
    result = {"dataset": args.dataset, "metric": "nDCG@10", "split_seed": args.split_seed,
        "run_sha256": hashlib.sha256(args.run.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "comparison": compare(read_run(args.run), read_run(args.reference), beir.qrels(args.dataset), args.split_seed)}
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
