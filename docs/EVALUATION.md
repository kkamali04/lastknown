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
| Alternate LongMemEval protocols | Full-turn 470-question run; larger-context exploratory runs | Protocol-specific research, kept separate from official 419-question results |

The dates identify historical measurements. Only SciFact was rerun for the Nov 22 alpha.
The [aggregate snapshot](../bench/evidence/public-results.json) contains the figures used
in the README/plots and SHA256 identities of the source reports. It is an aggregate
record, not a release of all historical per-query data. Original raw archives remain
outside this repository. LoCoMo data is CC BY-NC 4.0 and is not redistributed.

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

Freshness is checked through a different contract: the [portable demo](TRY-IT.md) edits
a source note, checks the stale warning, reindexes and checks the updated result. It also
checks a file edit outside the shown passage. This demonstrates file/passage drift behavior,
not that the source is factually correct. Redacted-value changes and hashing budgets remain
[documented limits](FRESHNESS.md).

## Reproduce and inspect

- [Public dataset locks, runners and environment](../bench/README.md)
- [Detailed retrieval protocols and baseline qualifications](BENCHMARKS.md)
- [Dated historical results, including invalid/alternate rows](../bench/RESULTS.md)
- [Curated aggregate figures and input-report hashes](../bench/evidence/public-results.json)
- [Nov 22 SciFact raw-ID runs, comparisons and separate embedding/query provenance](https://github.com/kkamali04/lastknown/releases/download/v0.1.0-alpha.1/scifact-baseline-20261122.zip)
- [Exact four-platform native acceptance receipts](https://github.com/kkamali04/lastknown/releases/download/v0.1.0-alpha.1/NATIVE-VALIDATION.json)

Native installation success is separate from rerunning retrieval benchmarks. The alpha's
Windows/Linux/Mac acceptance does not turn historical Linux benchmark rows into measured
scores on all four platforms.
