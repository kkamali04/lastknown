# Evaluation: what was tested, and what it establishes

Lastknown uses established retrieval components: SQLite FTS5, BGE-small embeddings and
score fusion. Its product focus is local notes and visible source-file drift. The evidence
below measures retrieval, freshness behavior and a small coding-agent pilot separately.

## Reading the scores

**nDCG@10** measures the quality of a ranked list against relevance judgments, with more
credit for relevant documents near the top. It is not answer accuracy. SciFact's 0.7271
fused score versus 0.6786 keyword score is a gain of about 4.85 points on a 100-point nDCG
scale. It does not mean a 72.7% improvement.

**R_all@5** is the fraction of questions whose complete set of evidence sessions appears
in the top five. **LoCoMo R@5** averages the fraction of evidence recovered per question.
Compare methods within a row; these different metrics are not one common leaderboard.

## Evaluation inventory

| Study | Scope | What the evidence supports |
|---|---|---|
| BEIR retrieval audit, Nov 12 | SciFact, NFCorpus, FiQA, SciDocs, ArguAna; TEST halves | Fused-over-keyword gains across five public document sets; single-channel and fusion controls |
| Conversation retrieval, Nov 13 | LongMemEval_S official 419; LoCoMo session/turn TEST halves | Evidence retrieval from conversation histories; no LLM reader |
| Release regression, Nov 22 | New 5,183-document SciFact store, 300 questions × 3 channels; exact Linux release-binary replay | No ranked-list regression against Nov 13 in either run |
| Duplicate and multi-topic stress tests, Nov 12 | Exact-copy variants and 4/8 documents packed per note, SciFact/NFCorpus | Fused-over-keyword gains survived these synthetic corpus transformations |
| Answerability and label tests, Nov 8–12 | LoCoMo, LongMemEval and BEAM-related checks; derived negatives | The tested heuristics/judge did not justify reliable default “no answer” labels |
| Coding-agent pilot, Nov 9 | VibeMemBench, 19 scored targets; multiple memory adapters and selected repeats | No established overall coding-task gain or winning memory system |
| Direct retrieval comparisons, Nov 5–6 | MemPalace/gbrain: LME 470/60, LoCoMo 1,982, SciFact 300, NFCorpus 323 | Positive results against the tested MemPalace setup; gbrain leads LME matched-60 by one question |
| Private capture comparison, Nov 1 | Recall/memcap versus Claude-mem, 20 headless reviews / 60 questions | Top-5 correct session 43/60 versus 41/60, with unequal retrieval backends |
| Eight-set RRF → convex ablation, Nov 6 | Four BEIR sets, two LoCoMo units, BEAM 100K/500K | Historical real-binary fusion comparison; means and 16 raw-file hashes rechecked Nov 23 |
| BEAM summaries, recency, rerankers | Separate summary A/B, offline date replay, four reported local rerankers | No clear evidence for enabling these enhancements by default |
| Alternate LongMemEval protocols | Full-turn 470; LME_M exploratory 46-scored-question fused run | Protocol-specific research, kept separate from official 419-question results |
| Private-note and embedding experiments | Evolving gold, backend/chunking and corpus controls | Several old gains invalidated by gold/ID/wiring problems; no stable personal-vault superiority claim |

The dates identify historical measurements. Only SciFact was rerun for the Nov 22 alpha.
The [aggregate snapshot](../bench/evidence/public-results.json) contains the figures used
in the README/plots and SHA256 identities of the source reports. It is an aggregate
record, not a release of all historical per-query data. Original raw archives remain
outside this repository. LoCoMo data is CC BY-NC 4.0 and is not redistributed.
The separate [historical-comparison aggregate](../bench/evidence/historical-comparisons.json)
records the rival and BEAM studies, exact means, denominators, configurations and input
hashes. The Nov 23 pass rescored saved results; it did not call models or rerun products.
Aggregate files alone cannot reproduce experiments whose raw inputs are not distributed.

## Five-set BEIR audit

Metric: nDCG@10, abstain off. Columns below are the Nov 12 audit's keyword and BGE captures
and offline reconstruction of the default 0.6 BGE / 0.4 keyword fusion, after per-channel
min-max normalization.

| Dataset | TEST questions | Keyword | BGE | Fused | Gain over keyword [95% CI], points |
|---|---:|---:|---:|---:|---|
| SciFact | 146 | 0.6786 | 0.6884 | 0.7271 | +4.85 [1.29, 8.62] |
| NFCorpus | 169 | 0.3037 | 0.3197 | 0.3417 | +3.80 [1.67, 6.12] |
| FiQA | 326 | 0.2535 | 0.4132 | 0.4159 | +16.24 [13.34, 19.21] |
| SciDocs | 498 | 0.1638 | 0.1981 | 0.2020 | +3.82 [2.81, 4.85] |
| ArguAna | 721 | 0.5015 | 0.5317 | 0.5658 | +6.43 [4.36, 8.48] |

There is no large fusion gain over the best single channel on every set: FiQA's BGE-only
score is already 0.4132. The evidence supports retaining semantic search alongside keyword
search, not claiming an unusual new ranking algorithm.

Important qualifications:

- TEST here means the audit's hash-based half (seed 20261112). Four sets were not fully
  unseen when the fusion weight was selected using earlier splits. ArguAna was the clean
  exception recorded in that audit.
- The audit used build fc7e598; FiQA/SciDocs/ArguAna stores had been built by 07a728a.
  SciFact/NFCorpus controls compared old and new store builds.
- Offline fusion differs slightly from the captured binary lists: NFCorpus binary 0.3412,
  ArguAna binary 0.5649. All three other base-set values match to four decimals.
  The snapshot retains both columns. The chart is explicitly the offline fusion audit.
- Intervals are paired-bootstrap intervals over questions, not uncertainty for each
  individual bar and not comparisons with BGE-only or another memory product.
- Weights 0.5–0.8 were close in this audit. No tested small change met the stop rule for
  replacing the incumbent. Exact-copy collapse was tested and not merged.
- Synthetic duplicate/packing tests do not establish performance on real personal vaults,
  edited near-duplicates or arbitrary long notes. Some absolute scores fell under packing.

## Conversation retrieval

Nov 13 binary SHA256:
`87fcfb03a657ee609c1b53c37908192a95537bc0764f63a1f9e8bda2118d4446`.

| Dataset/protocol | n | Keyword | BGE | Default fusion | RRF |
|---|---:|---:|---:|---:|---:|
| LongMemEval_S official, R_all@5 | 419 | 0.866 | 0.883 | 0.921 | 0.905 |
| LoCoMo sessions, TEST R@5 | 1,036 | 0.858 | 0.726 | 0.878 | 0.855 |
| LoCoMo turns, categories 1–4, TEST R@5 | 784 | 0.466 | 0.496 | 0.577 | 0.558 |

LongMemEval default fusion found all evidence for 386/419 questions; the same-subset
upstream BM25 reference scored 0.742. Its whole-set number is **not held out**. Against
RRF, default fusion gained 11 questions and lost four; no confidence interval was computed
for that comparison. It is not evidence of superior answer generation.

LoCoMo kept the historical split seed 20261105. Default-minus-RRF gains were +2.28 points
[1.13, 3.47] for sessions and +1.85 [0.69, 2.98] for turns; intervals were clustered by
conversation. Those intervals do not apply to the keyword/BGE comparisons.

The older full-user-plus-assistant-turn 470-question LongMemEval protocol produced fused
R_all@5 0.887. Its gold, document construction, fusion and denominator differ from the
official 419 result. See [the dated results log](../bench/RESULTS.md); do not mix the two.

## Historical retrieval comparisons

Measured Nov 5–6, 2026; independently rescored from saved ranked lists and public gold
Nov 23. These rows compare the tested configurations, not current releases or official
end-to-end answer pipelines. There were no missing queries in the scored captures.

| Protocol / metric | n | Recall | gbrain local | MemPalace |
|---|---:|---:|---:|---:|
| LongMemEval_S full user+assistant sessions, R_any@5 | 470 | 0.9809 | Not run | 0.9213 |
| Same protocol, R_all@5 | 470 | 0.8872 | Not run | 0.7745 |
| Same protocol, R_all@10 | 470 | 0.9574 | Not run | 0.8830 |
| Matched LME subset, R_all@5 | 60 | 0.8833 | 0.9000 | 0.7500 |
| LoCoMo historical evidence-bearing questions, session R@5 | 1,982 | 0.8504 | 0.7664 | 0.4544 |
| LoCoMo historical evidence-bearing questions, turn R@5 | 1,982 | 0.5523 | 0.5304 | 0.3249 |
| SciFact full-set nDCG@10 | 300 | 0.7340 | 0.6977 | 0.6501 |
| NFCorpus full-set nDCG@10 | 323 | 0.3624 | 0.3502 | 0.3136 |

The 470-question R_all@5 counts are 417 versus 364; 70 questions succeeded only for
Recall and 17 only for MemPalace. On the 60-question subset, gbrain scored 54/60 versus
Recall 53/60 and MemPalace 45/60. That one-question difference is not an established winner.

- **MemPalace 3.12.0:** ChromaDB + all-MiniLM-L6-v2. The LongMemEval command used
  `--mode full --granularity session`. An old report called it “raw” in places; the
  command/audit establish full mode. It is not the product's headline raw-mode setup.
- **gbrain 0.60.574.0:** PGLite + local Ollama/nomic-embed-text (768 dimensions), balanced
  retrieval, reranker off, autocut off. Paid OpenAI/Voyage configurations were not run.
  The full 470-question LME run was not completed. Only top five were captured for its
  60-question LME run; no top-ten score is inferred.
- **Recall:** preserved historical rankings, not a new simultaneous run. Build labels in
  the original rival report were inconsistent; the matched-60 run used an older build.
  These are not measurements of the Nov 22 alpha against current competitors.
- LoCoMo/BEIR used adapters with common document units and scorers. The 1,982-question
  LoCoMo rows cover all categories with annotated evidence (four no-evidence questions
  excluded) and retain historical evidence multiplicity. Evidence-deduplicated session
  scores are 0.8488 / 0.7651 / 0.4544; both variants are retained in the aggregate.
  Neither variant should be mixed with the later TEST-half/category-filtered table.
- Shared VPS load and differing indexing/query work prevent a clean speed ranking.
  Vendor-published Zep/Mem0/Grove/Attemory scores are literature, not additional runs here.
  QMD was researched but was not installed or benchmarked in the retained comparison.

## Long-context fusion: BEAM and the eight-set study

This Nov 6 real-binary experiment compared RRF build `07a728a` with convex build
`da9c540`, abstain off. The BGE weight of 0.6 was frozen from DEV, with split seed
20261105. It differs from the Nov 12 fused-versus-keyword audit above.

| Dataset / unit | Questions | Metric | RRF | Convex | Gain, points |
|---|---:|---|---:|---:|---:|
| SciFact | 146 | nDCG@10 | 0.7336 | 0.7409 | +0.73 |
| NFCorpus | 159 | nDCG@10 | 0.3580 | 0.3647 | +0.67 |
| SciDocs | 487 | nDCG@10 | 0.1830 | 0.1944 | +1.14 |
| FiQA | 327 | nDCG@10 | 0.3646 | 0.4173 | +5.27 |
| LoCoMo sessions | 1,036 | R@5 | 0.8551 | 0.8780 | +2.28 |
| LoCoMo turns | 784 | R@5 | 0.5579 | 0.5768 | +1.89 |
| BEAM 100K | 193 | Derived R@5 | 0.3681 | 0.3953 | +2.72 |
| BEAM 500K | 179 | Derived R@5 | 0.3120 | 0.3436 | +3.16 |

BEAM 500K was never used to choose the weight in this study; all 179 questions were
scored. Convex improved 16 questions and worsened six. Saved conversation-cluster
bootstrap intervals were [+0.67, +4.96] points for 100K and [+1.30, +5.06] for 500K.
Those intervals were not rerun Nov 23; all means, better/worse counts and the 16 source
data hashes were rechecked. The original SciFact/NFCorpus gain intervals crossed zero.

BEAM here is a **derived evidence-retrieval test**, with no answer reader. It is not
official BEAM QA accuracy. The LoCoMo RRF turn score also differs slightly from the later
Nov 13 build. Keep these experiment identities separate.

## Private capture comparison: Claude-mem

Twenty sampled headless code-review sessions, three questions each, measured in the
Nov 1 capture study. Mechanical counts were recomputed from the saved 60-question file.

| Correct session retrieved | Recall/memcap after embedding | Claude-mem tested FTS5 path |
|---|---:|---:|
| In the top five | 43/60 | 41/60 |
| Ranked first | 22/60 | 13/60 |

Both found 34 questions; Recall alone found nine, Claude-mem alone seven, neither ten.
Recall used FTS5+BGE. Claude-mem's Chroma path was unavailable, so it used its own FTS5.
The earlier Recall 31/60 result was keyword-only before embedding and is superseded
within this experiment. This small private review sample does not rank current products.

The 20 preserved model-judge files counted 100/100 key items present for memcap versus
22 present / 28 partial / 50 absent for Claude-mem. Its 200 judged claims were 187
supported / eight unsupported / five contradicted. These are recounted **saved model
judgments**, not a new blind factual audit. Memcap included labeled Claude-mem summaries;
the old “zero errors” claim covered verbatim copied text only. Source transcripts and
inherited summaries may contain errors. No universal correctness claim follows.

Only aggregate counts and source hashes are published; private capture/query text and
session identifiers are not distributed.

## Other completed experiments and negative results

| Study | Result | Interpretation |
|---|---|---|
| BEAM 100K raw versus summaries, 20 chats / 247 sessions / 355 questions | FTS R@5: raw 0.7462, raw+summary 0.7503, summary-only 0.6358. BGE: 0.7748 / 0.7713 / 0.6044 | Added-summary intervals crossed zero; summary-only was worse. Different session metric/store from the 193-question fusion study; saved aggregates extracted, not newly rescored per query. Experimental binary provenance was incomplete. |
| Recency/stale ranking replay | Temporal TEST: LME 96 questions and LoCoMo 95 both had zero score change; other LoCoMo 459 questions changed by −0.0005 | Offline formula replay, not a binary change. Reordering did not improve top-five gold coverage; static corpora could not establish a stale penalty. Both knobs stayed zero. |
| Four local cross-encoder rerankers | Reported DEV gains approximately −0.9 to +1.4 points, none significant, with extra latency | No basis for a default reranking change; no new model run here. |
| LongMemEval_M exploration | Fused 46 scored questions: R_all@5 0.739, versus keyword 0.761 and BGE 0.652. Separate keyword run: 419 scored, 0.742 | Manifest says M despite an erroneous S report heading. A small exploratory run existed; no full matched large-context fusion rebuild was completed. |
| Nov 12 H1–H6 controls | Keyword/fusion, weight grid, RRF, exact-copy collapse, multi-topic packing, query-instruction controls | No small tested replacement met the stop rule. Exact-copy collapse was not merged; synthetic stress tests do not prove personal-vault performance. |
| Private-note gold and embedding/chunking iterations | Changing gold versions, ID/path/corpus/backend errors invalidated several early gains; a later roughly 0.593 R@5 did not beat keyword | No stable personal-vault improvement claim. Timeout-only runs are infrastructure failures, not zero accuracy. The Qwen CPU trial stopped before retrieval queries. |

## Coding-agent memory comparison

Historical VibeMemBench pilot: solver gpt-6-luna via Codex CLI 0.166.0, 20 planned targets,
19 scored after an infrastructure failure. Each first-try cell below is tasks resolved,
not retrieval accuracy. These adapter runs predate the public alpha.

For the added LLM-based arms, the harness substituted the same local Codex/Luna shim for
their default model provider and a BGE-small-class embedder for their default embeddings.
Histories were fed as session documents, with retrieved context capped at 6,000 characters
per arm. These are qualified adapter trials, not measurements of each product's current
default configuration.

| Arm | First try, same 19 targets | Meaning |
|---|---:|---|
| Memory off | 7/19 | Common baseline |
| Lastknown/recall, abstain off | 7/19 | No measured overall gain |
| Echo | 9/19 | Harness word-overlap baseline, not a commercial product |
| mem0-noinfer 2.2.1 | 7/19 | Storage/dense retrieval only; **no LLM extraction**, not the paper/managed Mem0 system |
| SimpleMem | 11/19 | Initial signal; selected repeats did not establish a reliable overall gain |
| MemoryOS | 5/19 | Adapter returned no memory for short-history targets; not a universal product verdict |

The Lastknown-minus-memory-off interval was [-15.8, +15.8] points. Seven targets were
selected for repeats because at least one arm flipped on the first try, so that subset
does not support an unbiased full-benchmark claim. On the additional two tries for those
targets, SimpleMem and mem0-noinfer each scored 8/14 versus baseline 6/14; both intervals
spanned zero. MemoryOS was not repeated. Its adapter returned nothing for eight of the
20 setup targets with at most seven sessions; one was the unscored infrastructure failure.
Thus seven of the 19 scored targets returned no memory, and 12 returned memory. Even that
12-target subset did not show a gain.

One repeated Lastknown case improved with relevant memory, but that cannot establish an
overall gain. A larger, matched, preregistered repeat is required before claiming improved
coding-agent performance. This table does not justify saying Lastknown beats Mem0,
SimpleMem or MemoryOS, and must not be compared with their current published scores.

## Answerability, freshness and limitations

The optional Haiku judge's TEST balanced accuracy was 0.734 [0.642, 0.792], below the
0.75 target; it labeled about 18% of answerable-and-retrieved questions. It remains off.
Lexical and channel-agreement labels also did not justify dependable default abstention.
Retrieval relevance, factual correctness and “the answer exists” are separate properties.
The early deleted-answer experiment was invalid because indexing was additive; later
pruned-store tests corrected that. Subject-swap negatives depend on a lexical construction,
and high retained recall is not itself good no-answer classification accuracy.

Freshness is checked through a different contract: the [portable demo](TRY-IT.md) edits
a source note, checks the stale warning, reindexes and checks the updated result. It also
checks a file edit outside the shown passage. This demonstrates file/passage drift behavior,
not that the source is factually correct. Redacted-value changes and hashing budgets remain
[documented limits](FRESHNESS.md).

## Reproduce and inspect

- [Public dataset locks, runners and environment](../bench/README.md)
- [Detailed retrieval protocols and baseline qualifications](BENCHMARKS.md)
- [Dated historical results, including invalid/alternate rows](../bench/RESULTS.md)
- [Curated release-era figures and input-report hashes](../bench/evidence/public-results.json)
- [Historical rival, BEAM and capture aggregates with source hashes](../bench/evidence/historical-comparisons.json)
- [Nov 22 SciFact raw-ID runs, comparisons and separate embedding/query provenance](https://github.com/kkamali04/lastknown/releases/download/v0.1.0-alpha.1/scifact-baseline-20261122.zip)
- [Exact four-platform native acceptance receipts](https://github.com/kkamali04/lastknown/releases/download/v0.1.0-alpha.1/NATIVE-VALIDATION.json)

Native installation success is separate from rerunning retrieval benchmarks. The alpha's
Windows/Linux/Mac acceptance does not turn historical Linux benchmark rows into measured
scores on all four platforms.
