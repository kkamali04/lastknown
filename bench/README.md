# recall public benchmark suite

Everything here measures **retrieval only** (is the right session / document / turn in the top k?). No LLM is called
anywhere, no private question set is used, and no dataset is stored in this repo. A stranger with the `recall`
binary and the commands below should reproduce our numbers (within the noise noted per benchmark).

`RESULTS.md` lists every run, bad ones included. For each run the raw per-question outputs, the score file, the exact
command, the binary sha256 and the recall commit are kept in a `results/<date>-<benchmark>/` folder under `bench/` in
the maintainers' private working repository. A public copy of this repository ships without `bench/results/`.

## Benchmarks

| benchmark | measures | unit | metric | datasets (see `datasets.lock`) |
|---|---|---|---|---|
| LongMemEval_S, **official 419** | find the chat session that holds the evidence (~50 sessions per question) | session, user turns only | R_any@k, R_all@k | `longmemeval_s_cleaned.json` (MIT) |
| LongMemEval_S, **MemPalace-style 470** | same, as published by MemPalace / gbrain / Grove | session, user + assistant turns | R_any@k, R_all@k | same file |
| LongMemEval abstain sets (**ours, derived**) | does recall refuse when the answer is not there | question | abstained / labelled / confident-wrong rate | built from the same file |
| LoCoMo | find the dialog turn / session that holds the evidence (10 long conversations) | turn, session | recall@k (fraction of evidence found) | `locomo10.json` (CC BY-NC 4.0) |
| BEIR SciFact, NFCorpus, ArguAna, FiQA, SciDocs | classic document search | document | nDCG@10 | BEIR zips |

### Protocols (read before comparing any number)

**LongMemEval, official 419.** Follows the upstream `src/retrieval/run_retrieval.py`, session granularity: each
haystack session becomes one document made of its *user* turns joined with a space; the query is the question;
gold = sessions with a user turn flagged `has_answer`. Scored questions = not `_abs` and at least one such turn =
**419 of 500** (the 51 questions whose evidence is only in assistant turns are skipped, as upstream does).
Metrics come from the upstream `eval_utils.evaluate_retrieval` (copied as `lme_eval_utils.py`).

**LongMemEval, MemPalace-style 470.** Same questions file, but the document is every turn (user *and* assistant)
joined with `\n`, gold = `answer_session_ids`, scored on **all 470 non-abstention questions**. This is the
protocol behind the published MemPalace 96.6, gbrain 98.51 and Grove 99.36 self-reported figures. It is easier than
the official 419 for assistant-evidence questions, so **never put a 470 number and a 419 number in one comparison.**
`lme_score470.py --official-gold` re-scores a full-turn run against the official gold on the 419 for a like-for-like row.

**any@k / all@k.** `R_any@k` = the share of questions where *at least one* gold session is in the top k;
`R_all@k` = the share where *every* gold session is. Multi-session questions have several gold sessions, so all@k is
the harder number. Upstream's printed "recall" is `recall_all`.

**recall specifics.** recall indexes one document per session (split into chunks, the bge model embeds each chunk) and
ranks each session by its best chunk. Retrieval is per session, never per turn. Channels: `fts5` (keyword),
`bge` (local BGE-small meaning search), `fts5+bge` (default convex fusion: 0.6 BGE, 0.4 keyword,
with per-channel min-max normalization). Earlier recorded runs used RRF; each result names its fusion.
Abstain is OFF for ranking runs.

**Derived "should say I don't know" sets (ours, not official).** Built by `lme_derived.py`, deterministic (seed
20261101, no LLM), checked by the sha256 values in `datasets.lock`:
- *answer-deleted* (419 questions): the question's evidence sessions are removed from the store (explicit
  `index --prune-missing=apply`, store size verified), so the correct answer is "not found".
- *subject-swap* (398 questions): the question's rarest evidence-held content word is replaced by a word that occurs in no
  session of that haystack (e.g. "Zanzibar"). 21 questions have no such word and are skipped (listed in `meta.json`).
Reported as the share of questions where recall abstained, returned a clearly labelled closest match, or answered
confidently (confident = wrong on these sets). The official `_abs` questions are reported the same way.

**LoCoMo.** Upstream `task_eval/evaluation.py` scores retrieval as `recall_acc` = evidence ids found in the retrieved
context / number of evidence ids, averaged over questions; at session granularity an id `D3:7` counts when session 3 was
retrieved. We follow that, at dialog-turn (k = 5, 10, 25, 50) and session (k = 1, 2, 3, 5, 10) units. The upstream
script gives a question *without* evidence a recall of 1.0 (4 of 1986); we report both "questions with evidence" and
"official-style". Categories as numbered in the data: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial
(unanswerable; the abstain-on pass reports how often recall returns nothing for them). Cat-5 questions do carry evidence
ids, so they are included in "all"; "1-4" excludes them. Evidence strings like `D8:6; D9:17` are split with a regex.

**BEIR.** Standard test split. Each corpus doc becomes `# <title>\n\n<text>`; every query with rows in
`qrels/test.tsv` is scored; nDCG@10 uses linear gain (identical to `pytrec_eval`'s `ndcg_cut_10`); a query that returns
nothing scores 0 and stays in the denominator; the doc whose id equals the query id is dropped (BEIR's
`ignore_identical_ids`); top 11 are requested so ArguAna still has 10 after that. The scorer is our own ~15 lines
(`beir.py`) and was checked against the earlier `pytrec_eval` run on SciFact (see RESULTS.md).

## One command per benchmark

Setup (once): `pip install -r bench/requirements.txt`, then

```
export RECALL_BIN=/path/to/recall                       # the build under test
export RECALL_MODEL_DIR=/path/to/models                 # holds models--Xenova--bge-small-en-v1.5 (needed for bge channels)
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so        # needed for bge channels
export BENCH_WORK=$PWD/bench-work BENCH_DATA=$PWD/bench-data
export BENCH_CPUS=0-7                                   # optional: one pinned cpu per worker
python3 bench/scripts/fetch_data.py                     # downloads + sha256-verifies everything in datasets.lock
```

Launch long runs under `nice -n 19 ionice -c3`. All runners are resumable (re-run the same command); a resume refuses to
mix builds or settings (`identity.json`), and a half-built store is rebuilt.

**Thread budget.** recall's `embed` runs an ORT pool sized to the cpus it may use *plus* a rayon pool of the same size,
so a job pinned to 8 cpus holds 16 threads (measured on the VPS: 16 threads, which pushed a shared box over its load cap).
`pb.py` sets `RAYON_NUM_THREADS=1` by default (9 threads on 8 cpus, same vectors); override with the env var.

| benchmark | command | then score |
|---|---|---|
| smoke (20 q, fts5, no model) | `python3 bench/scripts/smoke.py` | prints `SMOKE OK`. CI: copy `bench/ci/bench-smoke.yml` to `.github/workflows/` (manual dispatch only; kept here because the push token has no `workflow` scope) |
| LME official 419 (+ abstain) | `python3 bench/scripts/lme.py --mode official --abstain --derived` | `python3 bench/scripts/lme_score.py --out $BENCH_WORK/lme/<sha8>-official --bm25` |
| LME MemPalace-style 470 | `python3 bench/scripts/lme.py --mode full` | `python3 bench/scripts/lme_score470.py $BENCH_DATA/longmemeval_s_cleaned.json $BENCH_WORK/lme/<sha8>-full/results.jsonl` (add `--official-gold` for the 419 row) |
| LoCoMo | `python3 bench/scripts/locomo.py run` | `python3 bench/scripts/locomo.py score` |
| BEIR | `python3 bench/scripts/beir.py all scifact nfcorpus arguana fiqa scidocs` | scores are written by `all`; `beir.py score <ds>` re-scores |

`lme.py --mode full --dense N` embeds only the first N questions (the 2026-11-02 run did 150 then the other 320;
`lme_merge.py` joins the two files, and the merged file's sha256 is in its MANIFEST). The default embeds all 470 in one pass.

## Hardware and time (what the recorded runs used)

Linux VPS, Intel Xeon Platinum 8168 (24 vCPU shared with other jobs), 125 GB RAM, ONNX Runtime 1.24.4 CPU, BGE-small
(384-d). Runs were pinned to 8 cores under `nice 19 / ionice idle`. Rough wall-clock: LME full 470 about 2.6 h with 6
workers on a loaded box; LoCoMo and BEIR times are in `RESULTS.md` per run. Anything embedding a corpus is CPU-bound:
FiQA (57,638 docs) is the largest.

## Caveats that matter

- The Linux binary used for the VPS runs was built from recall commit `a97c216` (v3 r8, wire proto 16; the
  pre-rewrite SHA is `77fed8f`). The Windows installed build is later (`a1342ca`, proto 17, abstain labels). Ranking
  runs use abstain off, so they should not differ; the abstention numbers belong to the build named in each result.
- Earlier Windows-PC runs (BEIR 3 sets, LME official, derived sets) used an older private harness; their numbers are listed in
  `RESULTS.md` as "legacy" with their own provenance. Re-running them with these scripts is a TODO unless a row says otherwise.
- Datasets keep their own licences. LoCoMo is **non-commercial**; we only measure with it. Nothing is redistributed.
- Vendor LongMemEval numbers (94-99%) mix protocols and often use an LLM reader; only the retrieval rows here are comparable.
