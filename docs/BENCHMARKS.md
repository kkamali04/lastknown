# Public benchmark results

Retrieval only: no LLM is called anywhere, and no private question set is used. These are the numbers quoted in the
README, with the protocol and the command that reproduces each. The harness is in [`bench/`](../bench/README.md); read its
protocol section before comparing any number with another system's.

| benchmark | metric | score | notes |
|---|---|---|---|
| LongMemEval_S, official 419 questions | R_all@5, fts5+bge fused | **0.921** (386/419) (RRF fusion: 0.905, 379/419) | session-level retrieval, abstain off, whole set, release build; not held-out, see below |
| LoCoMo, session level | R@5, held-out TEST half | **0.878** (RRF fusion: 0.855) | gain +2.28 points, 95% CI [+1.13, +3.47] |
| LoCoMo, dialog turn, categories 1-4 | R@5, held-out TEST half | **0.577** (RRF fusion: 0.558) | gain +1.85 points, 95% CI [+0.69, +2.98] |
| BEIR: SciFact / NFCorpus / FiQA / SciDocs / ArguAna | nDCG@10, held-out TEST half, fts5+bge fused (keyword-only in brackets) | **0.727** (0.679) / **0.342** (0.304) / **0.416** (0.254) / **0.202** (0.164) / **0.566** (0.502) | fused beats keyword-only on all five; gains and caveats below |

## LongMemEval_S (official 419)

Protocol: each haystack session is one document made of its user turns; the query is the question; gold = sessions holding a
user turn flagged `has_answer`; the 419 of 500 questions that have one are scored (the official `run_retrieval.py` filter).
R_all@5 is the share of questions where every gold session is in the top 5.

Build and fusion: measured on the November 13, 2026 release candidate, abstain off, whole set, with the default
fusion of the keyword (fts5) and local meaning (BGE-small) channels (convex score fusion, see LoCoMo below).

| run on the same stores | R_any@5 | R_all@5 | R_all@10 | nDCG@10 |
|---|---|---|---|---|
| fts5+bge, default fusion | 0.971 | **0.921** (386/419) | 0.979 | 0.938 |
| fts5+bge, reciprocal-rank fusion (k=60) | | 0.905 (379/419) | | |
| keyword (fts5) only | | 0.866 | | |
| BGE only | | 0.883 | | |
| BM25 reference (official `run_retrieval.py`, same 419 questions) | | 0.742 | | |

Default against reciprocal-rank fusion on R_all@5: 11 questions gained, 4 lost. The reciprocal-rank figure is the same as the
one measured on an earlier build that used that fusion by default.

This is **not a held-out number**: the fusion weight was chosen on halves of public sets that included LongMemEval_S
questions, so the questions scored here were partly seen when the one weight was picked. The LoCoMo and BEIR rows are the
held-out ones.

Reproduce (whole set):

```
python3 bench/scripts/lme.py --mode official --abstain --derived
python3 bench/scripts/lme_score.py --out $BENCH_WORK/lme/<sha8>-official --bm25
```

Setup (`RECALL_BIN`, `RECALL_MODEL_DIR`, `ORT_DYLIB_PATH`, `BENCH_WORK`, `BENCH_DATA`, dataset download with checksums) is described
in [`bench/README.md`](../bench/README.md).

The reciprocal-rank figures here and in the LoCoMo rows need `RECALL_FUSION_MODE=rrf` in the environment of the queries. `bench/scripts` builds a fixed environment for the binary and does not pass that variable through, so the commands reproduce the default-fusion figures only.

## LoCoMo

Protocol: recall@k = evidence ids found in the retrieved context divided by the number of evidence ids, averaged over questions
(the upstream `task_eval/evaluation.py` definition). Session level counts an id such as `D3:7` as found when session 3 was
retrieved, and each distinct evidence session is counted once. Dialog level is per turn; the row above excludes category 5
(adversarial, unanswerable questions).

The two LoCoMo rows compare the current default fusion (convex score fusion, 0.6 weight on the BGE channel and 0.4 on keyword,
each channel min-max normalised) with reciprocal-rank fusion. The fusion weight was chosen on one half of the questions (DEV);
the rows are the other half (TEST), which was never tuned on. The intervals are paired-bootstrap intervals clustered by
conversation. Both rows were re-measured on the release build with the original split (1,036 session-level and 784
dialog-level TEST questions). LoCoMo data is licensed CC BY-NC 4.0: only the harness is distributed, never the data.

Reproduce (whole set):

```
python3 bench/scripts/locomo.py run
python3 bench/scripts/locomo.py score
```

`locomo.py` scores the whole dataset, so its totals differ from the TEST-half rows above; the DEV/TEST split and the
paired-bootstrap script are **not** part of `bench/scripts`.

## BEIR (five sets): fused against keyword-only

Protocol: BEIR test split. Each corpus document becomes one markdown file (`# <title>`, blank line, text). The query is
the BEIR query; score is nDCG@10 with linear gain; the document whose id equals the query id is dropped. Abstain off.

The row compares the default search (keyword and local BGE-small channels, convex score fusion with 0.6 weight on BGE)
with the keyword channel alone, on the same questions. Each set's questions were split in half by a hash of the query
id; the numbers are the TEST half. Intervals are paired-bootstrap 95% intervals over questions.

| set | questions | keyword-only | fused | gain in points [95% CI] |
|---|---|---|---|---|
| SciFact | 146 | 0.679 | 0.727 | +4.85 [+1.29, +8.62] |
| NFCorpus | 169 | 0.304 | 0.342 | +3.80 [+1.67, +6.12] |
| FiQA | 326 | 0.254 | 0.416 | +16.24 [+13.34, +19.21] |
| SciDocs | 498 | 0.164 | 0.202 | +3.82 [+2.81, +4.85] |
| ArguAna | 721 | 0.502 | 0.566 | +6.43 [+4.36, +8.48] |

Caveats:

- Build `fc7e598`. The FiQA, SciDocs and ArguAna stores were indexed and embedded by an older build (`07a728a`) and
  queried by `fc7e598`. SciFact and NFCorpus were built both ways and scored the same.
- SciFact was re-measured on the release build with the same split and gave the same result (fused 0.727, keyword-only
  0.679, gain +4.85 [+1.29, +8.62], 146 questions). The other four sets were not re-run on the release build.
- The fusion weight was chosen earlier on other halves of four of these sets (not ArguAna), drawn with a different
  seed. The TEST halves here are therefore not fully unseen for that one number. Weights from 0.5 to 0.8 score within
  about one point of each other.
- Measured on Linux only.

Reproduce (whole set):

```
python3 bench/scripts/beir.py all scifact nfcorpus arguana fiqa scidocs
```

`beir.py` scores the whole test split, so its totals differ from the TEST-half rows above.
`bench/scripts/beir_compare.py` now supplies the hash-based TEST split and paired bootstrap
for BEIR run files. For SciFact the historical split seed is 20261112:

```sh
python3 bench/scripts/beir_compare.py scifact --run new/run.jsonl --reference previous/run.jsonl --out comparison.json
```

It uses the same `BENCH_DATA` and verified qrels as `beir.py`, and reports both the full
300-question set and the 146-question TEST half, including failed-query counts.

## What these numbers do not show

They do not compare recall with other memory systems: published figures for other systems mix protocols and often use an LLM
reader. They also say nothing about answerability (knowing when no answer exists), which recall does not claim to solve; see the
README's "What it doesn't do".

The BEIR sets are public corpora of clean, single-topic documents. No public, repeatable benchmark of a personal notes
folder exists here, so the fused-over-keyword gain has not been shown on one, and a messy real vault may see less.
