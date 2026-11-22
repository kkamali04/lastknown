# Results log

Every run, newest first, including the bad ones. A number only counts together with its protocol, build and commit.
Binary: `dedf74a5…` = Linux build of recall `a97c216` (v3 r8, wire proto 16, pre-rewrite SHA `77fed8f`). Abstain is
OFF for ranking rows. Raw outputs, score files and exact commands are in `results/<name>/MANIFEST.json` in the
maintainers' working repository (the `results/` folders are not part of a public copy of this repository).

## 2026-11-13 — release build: LongMemEval_S official 419, LoCoMo TEST half, BEIR SciFact TEST half (`results/2026-11-13-final-build`)

Binary: Linux build of the release source, sha256 `87fcfb03a657ee609c1b53c37908192a95537bc0764f63a1f9e8bda2118d4446`. Default
fusion is convex score fusion; the RRF rows are the same stores queried again with `RECALL_FUSION_MODE=rrf`, which
`bench/scripts` does not pass through (a small driver around the stock scripts did; it is in the results folder).

LongMemEval_S, official 419, user-turn docs, abstain off, whole set:

| channel | n | R_any@5 | R_all@5 | R_any@10 | R_all@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| fts5 | 419 | 0.957 | 0.866 | 0.974 | 0.926 | 0.900 |
| bge | 419 | 0.959 | 0.883 | 0.981 | 0.952 | 0.902 |
| **fts5+bge (default fusion)** | 419 | **0.971** | **0.921** | 0.993 | 0.979 | 0.938 |
| fts5+bge (RRF) | 419 | 0.962 | 0.905 | 0.988 | 0.962 | 0.926 |
| BM25 flat (official `run_retrieval.py`, same subset) | 419 | 0.888 | 0.742 | 0.926 | 0.823 | 0.794 |

Not held-out: the fusion weight was chosen on halves of public sets that included LongMemEval_S questions. The RRF row equals the
2026-11-02 row below, measured on build `a97c216`.

LoCoMo, TEST half of the original split (seed 20261105), R@5, default fusion against RRF, paired bootstrap clustered by conversation:

| unit | n | default | RRF | gain in points [95% CI] |
|---|---|---|---|---|
| session | 1036 | 0.878 | 0.855 | +2.28 [+1.13, +3.47] |
| dialog turn, categories 1-4 | 784 | 0.577 | 0.558 | +1.85 [+0.69, +2.98] |

BEIR SciFact, TEST half (seed 20261112, n = 146), nDCG@10: fused 0.727, keyword-only 0.679, gain +4.85 [+1.29, +8.62]. The other four
BEIR sets were not re-run on this build.

## 2026-11-02 — LongMemEval_S, MemPalace-style 470, full turns (`results/2026-11-02-lme-s-fullturn-470`)

Protocol: user+assistant turns joined, gold = `answer_session_ids`, all 470 non-abstention questions. VPS, 6 workers.

| channel | n | R_any@5 | R_all@5 | R_any@10 | R_all@10 |
|---|---|---|---|---|---|
| fts5 | 470 | 0.968 | 0.851 | 0.989 | 0.915 |
| bge | 470 | 0.962 | 0.862 | 0.985 | 0.951 |
| **fts5+bge** | 470 | **0.981** | **0.887** | 0.994 | 0.957 |

Same fused run scored with the official gold (sessions with a user `has_answer` turn, the 419 questions):
any@5 0.971, all@5 0.893, any@10 0.988, all@10 0.964 (`score419_official_fullturn.txt`). Not the official protocol (the
index holds assistant turns too); it is the like-for-like row against the official 419 numbers below.
Reproduced exactly from the raw records by `lme_score470.py`: the score file is byte-identical to the one recorded on 2026-11-02,
and the merged file's sha256 `754c10ce…` is reproduced by `lme_merge.py` from the two pass files.
Published self-reported numbers (different readers/protocol mixes, not ours): MemPalace 96.6, gbrain 98.51, Grove 99.36.

## 2026-11-02 — LongMemEval_S, official 419, user-turn docs (`results/2026-11-02-lme-s-official419-vps`)

| channel | n | R_any@5 | R_all@5 | R_any@10 | R_all@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| fts5 | 419 | 0.957 | 0.866 | 0.974 | 0.926 | 0.900 |
| bge | 419 | 0.959 | 0.883 | 0.981 | 0.952 | 0.902 |
| **fts5+bge** | 419 | **0.962** | **0.905** | 0.988 | 0.962 | 0.926 |
| BM25 flat (official `run_retrieval.py`, same machine) | 419 | 0.888 | 0.742 | 0.926 | 0.823 | 0.794 |

**Bad rows, kept on purpose.** (1) The `del:*` rows inside `results.jsonl` / `metrics.json` of this run are INVALID:
plain `recall index` does not tombstone deleted files, so the "deleted" evidence stayed searchable; use the delprune file.
(2) With abstain ON, recall refuses 65% of the *answerable* official questions (abstained 0.647 fts5 / 0.654 bge /
0.654 fused; a further 21-25% come back labelled "closest match"), so as shipped on this build the LongMemEval phrasing
is answered confidently only 9-14% of the time. The ranking rows above use abstain off.

| abstention set (abstain on, build `a97c216`) | n | abstained | labelled closest match | confident (wrong on derived sets) |
|---|---|---|---|---|
| official `_abs` questions (fused) | 30 | 0.967 | 0.033 | 0.000 |
| derived A, answer deleted, valid re-run (fused) | 419 | 0.976 | 0.021 | 0.002 |
| derived A, answer deleted (fts5) | 419 | 0.964 | 0.029 | 0.007 |
| derived B, subject swapped (all channels) | 398 | 1.000 | 0.000 | 0.000 |

Derived sets are OURS, not official. See README.

## 2026-11-01 — BEIR SciFact, NFCorpus, ArguAna, legacy Windows harness (`results/2026-11-01-beir-3sets-pc-legacy`)

Not produced by `bench/scripts`; private harness, pytrec_eval scorer, Windows build (exe sha `7343dd7b…`), abstain off.

| dataset | fts5 | bge | fts5+bge |
|---|---|---|---|
| SciFact (300 q) | 0.6869 | 0.7121 | **0.7311** |
| NFCorpus (323 q) | 0.3182 | 0.3431 | **0.3616** |
| ArguAna (1406 q) | 0.5045 | 0.5306 | **0.5589** |

## Smoke (CI) reference value

20 LongMemEval_S questions (18 scored), fts5, no model, abstain off, `python3 bench/scripts/smoke.py` on a Linux x64 build of
this tree, 2026-11-13: R_any@5 0.944, R_all@5 0.833, R_any@10 0.944, 0 query errors. The script fails under R_any@5 0.80.
