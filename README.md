# lastknown

**Memory that knows when it's stale.**

The command-line tool, crate and binary are still called `recall` in this release.

**Public alpha, Apache-2.0.** Download a prebuilt package from
[GitHub Releases](https://github.com/kkamali04/lastknown/releases), or build the source below.
[Try it in a temporary profile](docs/TRY-IT.md) to check search and stale-memory warnings
without replacing an existing Recall installation.

Windows x64, Linux x64/ARM64, and Apple Silicon macOS are the native release targets.
See [platform support and validation](docs/PLATFORMS.md) for which have actually passed,
prebuilt installation, and the manual native test workflow. Intel macOS is not supported by
the pinned semantic-search runtime.

Local-first search over your notes for you and your coding agents (MCP). Every result says whether the indexed text of its file has changed since it was indexed, so an agent is told to read the real file instead of trusting an old copy.
Free and offline after install: no API keys, and by default nothing leaves your machine (the one exception is the optional Haiku judge, off by default, which sends a question and its top passages to Anthropic through your own Claude CLI).

## See it work

`examples/stale-demo/run.sh` (or `run.ps1`) indexes a note that says "The API port is 8080", then edits the file twice without re-indexing and queries after each edit: first another line of the note (the passage shown is still in the file), then the port line itself (8080 to 9090). It uses a temp HOME and a temp vault, checks its own output, and exits non-zero if anything is wrong, so it runs in `scripts/ci-local` too (the `demo` step). It needs ripgrep (`rg`) on PATH.

```
sh examples/stale-demo/run.sh path/to/recall
```

Illustration adapted from the recorded Linux run (2026-11-13), with the score lines now hidden by default. The test below is also run against the release candidate on Windows; paths and times vary:

```text
== 1. write a note and index it
  PASS  settings set vault exits 0
Indexed 1 docs from /tmp/tmp.jO9CV3GEWt/vault
  docs in index  1
  PASS  index exits 0

== 2. query: fresh, the file matches the index

recall query: top 1 for: API port

  1. API notes
     api-notes.md
     …The [API] [port] is 8080  Questions go to the platform channel. 
  PASS  query exits 0
  PASS  finds the note with 8080
  PASS  no stale warning

== 3. edit ANOTHER part of the file (the intro line) WITHOUT re-indexing, then query

recall query: top 1 for: API port

  1. API notes
     api-notes.md
     ~ file changed, this passage unchanged: the file changed on 2026-11-13 22:33 UTC after recall indexed it on 2026-11-13 22:33 UTC, but the excerpt shown is still in it.
     …The [API] [port] is 8080  Questions go to the platform channel. 
  PASS  query exits 0
  PASS  not flagged STALE: the passage shown is still in the file
  PASS  says the file changed but this passage is unchanged
  PASS  still shows 8080

== 4. edit the PASSAGE itself (8080 -> 9090) WITHOUT re-indexing, then query again

recall query: top 1 for: API port

  Warning: 1 of 1 results point at files that changed on disk after recall indexed them; read the actual files, not these excerpts.

  1. API notes
     api-notes.md
     !! STALE: this passage changed: the file changed on 2026-11-13 22:33 UTC after recall indexed it on 2026-11-13 22:33 UTC, and a line the excerpt shown is on has changed; read the actual file.
     …The [API] [port] is 8080  Questions go to the platform channel. 
  PASS  query exits 0
  PASS  hit is flagged STALE
  PASS  says this passage changed
  PASS  names the file
  PASS  tells you to read the real file
  PASS  the flagged hit still shows the old indexed text (8080), not a silent rewrite

== 5. re-index, then query: the fresh answer
Indexed 1 docs from /tmp/tmp.jO9CV3GEWt/vault
  docs in index  1
  PASS  index exits 0

recall query: top 1 for: API port

  1. API notes
     api-notes.md
     …The [API] [port] is 9090  Questions go to the platform channel. 
  PASS  query exits 0
  PASS  fresh answer is 9090
  PASS  old 8080 text is gone
  PASS  no stale warning

stale-demo: PASS
```

In step 3 the file has changed but the excerpt shown is still in it, so the hit is annotated, not flagged. In step 4 the line the excerpt is on has changed: the query still returns the old text, but it is flagged, names the file, and says to read it. After re-indexing, the answer is current. The check compares a SHA-256 of each returned file, in the form recall stores it, with the indexed text, so re-saving identical bytes is not flagged (and neither is a change recall's stored form does not show: see "What it doesn't do"). A query hashes at most 2 MiB of file content, best hits first; past that budget it falls back to size and mtime. Details: [docs/FRESHNESS.md](docs/FRESHNESS.md).

## Measured results (public benchmarks only)

Retrieval only: no LLM is called anywhere, no private question set is used. Every row is documented in [docs/BENCHMARKS.md](docs/BENCHMARKS.md); protocols and commands are in [bench/README.md](bench/README.md) (`export RECALL_BIN=...` first, setup in that file).

The table records November 13, 2026 and earlier measurements, with dates and binary hashes in
`bench/RESULTS.md`. Passing a platform build or installation check is separate from rerunning
these retrieval benchmarks.

| benchmark | metric | score | notes | whole-set command |
|---|---|---|---|---|
| LongMemEval_S, official 419 questions | R_all@5, fts5+bge | **0.921** (386/419) (RRF fusion: 0.905, 379/419) | session-level retrieval, abstain off, whole set, measured on the release build with the default (convex) fusion. Not a held-out number: the fusion weight was chosen on halves of public sets that included LongMemEval_S questions. Keyword only 0.866, BGE only 0.883 | `python3 bench/scripts/lme.py --mode official --abstain --derived`, then `python3 bench/scripts/lme_score.py --out $BENCH_WORK/lme/<sha8>-official --bm25` |
| LoCoMo, session level | R@5, held-out TEST half | **0.878** (RRF fusion: 0.855) | gain +2.28 points, 95% CI [+1.13, +3.47], current default (convex fusion), re-measured on the release build. Scored on the TEST half only; `locomo.py` scores the whole set, so its totals differ | `python3 bench/scripts/locomo.py run`, then `python3 bench/scripts/locomo.py score` |
| LoCoMo, dialog turn, categories 1-4 | R@5, held-out TEST half | **0.577** (RRF fusion: 0.558) | gain +1.85 points, 95% CI [+0.69, +2.98]; same note | same |
| BEIR: SciFact / NFCorpus / FiQA / SciDocs / ArguAna | nDCG@10, held-out TEST half, fused (keyword-only in brackets) | **0.727** (0.679) / **0.342** (0.304) / **0.416** (0.254) / **0.202** (0.164) / **0.566** (0.502) | fused beats keyword-only by +4.85 [+1.29, +8.62] / +3.80 [+1.67, +6.12] / +16.24 [+13.34, +19.21] / +3.82 [+2.81, +4.85] / +6.43 [+4.36, +8.48] points (95% CI). Build `fc7e598`; SciFact was re-measured on the release build with the same result, the other four sets were not. The FiQA, SciDocs and ArguAna stores were built by an older build (`07a728a`) and queried by `fc7e598`. The fusion weight had been chosen earlier on other halves of four of these sets (not ArguAna), so the TEST halves are not fully unseen for that one number | `python3 bench/scripts/beir.py all scifact nfcorpus arguana fiqa scidocs` |

The commands in the last column run the whole benchmark. The two LoCoMo rows are the held-out TEST half with paired-bootstrap intervals, produced by a paired-bootstrap procedure described in [docs/BENCHMARKS.md](docs/BENCHMARKS.md) that is not part of `bench/scripts`; the whole-set run gives different totals. The LongMemEval row is a whole-set number. The RRF figures shown beside the default-fusion scores need `RECALL_FUSION_MODE=rrf` in the environment of the queries. `bench/scripts` builds a fixed environment for the binary and does not pass that variable through, so the commands reproduce the default-fusion figures only. The BEIR row is also a TEST half with paired-bootstrap intervals, documented in [docs/BENCHMARKS.md](docs/BENCHMARKS.md); `beir.py` scores the whole set, while `bench/scripts/beir_compare.py` reproduces the TEST split and paired-bootstrap comparison from saved run files.

These are not comparable with other systems' published numbers: those mix protocols and often use an LLM reader.

## What it doesn't do

- **It does not tell you when it has no answer.** An optional Haiku "can these passages answer the question?" judge exists (`RECALL_JUDGE=haiku`) but is **off**: on the held-out TEST set its balanced accuracy was 0.734 (95% CI 0.642 to 0.792), below the 0.75 bar we set, and it labelled 18% of answerable questions. Without it, the simple lexical "closest match" decision measured close to chance on conversational data (LoCoMo, BEAM: balanced accuracy 0.49 to 0.51; with the pre-registered `agree` rule, 0.46 to 0.58 on LoCoMo and LongMemEval_S TEST), so the CLI and UI do **not** print a "closest match" label or "No confident match" banner by default (`recall settings set closest_labels on` brings them back; the JSON keeps `closest_match`). When a main word of the query is in none of the indexed notes, one plain line names it. By default it never refuses or reorders results to hide a weak match (`RECALL_ABSTAIN_POLICY=refuse` restores refusing).
- **The gain from the embedding half is measured on public corpora only.** The BEIR row above uses clean, single-topic documents. No public, repeatable benchmark of a personal notes folder exists here, so the gain has not been shown on one, and a messy real vault may see less. On the public data, adding exact copies or packing several documents into one long note did not remove it.
- **It has not been shown to make a coding agent better.** On a 19-task VibeMemBench pilot, a coding agent with recall solved 7 and without memory solved 7 (difference 0.0 points, 95% CI -15.8 to +15.8). One task consistently improved with a relevant memory, one did not; the pilot cannot detect small effects.
- **Freshness has limits.** recall compares the text as it stores it. With redacted storage (the default on a new install) a change confined to a redacted value, such as one email address replaced by another, is not seen: no stale line, and `recall refresh` counts the file as unchanged. The `[EMAIL]`-style placeholder in the result already tells you to open the file for the value. A uniform line-ending flip is not a change either. The bulk scan is stat-based, so an edit that keeps both size and mtime is invisible to it (a query catches it while the file is inside its hashing budget). Only BGE vectors are re-embedded after a change. macOS is untested. See [docs/FRESHNESS.md](docs/FRESHNESS.md).
- **One notes folder** (`.md` / `.qmd`), not several roots.

## Known limits

Seen in testing and left as they are in this release.

- **Exit codes.** Some refusals print an `error:` or `Refused:` line but exit 0: `recall supersede` with an unknown or non-numeric id, `recall index --reset true` without the confirmation variable, `recall bench` without fixtures. Read the output, not only the exit code.
- **Odd values are accepted.** `--top abc` and `--top -3` return 5 results (not the default 20) and `recall recent -n abc` lists everything, all without an error. An empty query is not refused (with the vector channel on it returns notes in no meaningful order), and `recall save ""` saves an empty observation.
- **"Not found in the indexed text: ..."** names query words that are not in the index. It can print above a correct first hit when you used different words than the note does. A value recall redacted is not in the index, so searching for an email address or a key that is in a note prints this line and does not find the note. This is a lexical diagnostic, not proof that your files contain no answer. Missing-term notes and `(weak)` tags are hidden unless `--verbose` or `closest_labels` is on.
- **Diagnostic scores (`--verbose`) do not explain the order.** The default ranking uses weighted raw scores, but the number shown is `rank-sum` (JSON `rrf`), so a lower hit can show a higher number. JSON `score` is the keyword score on some rows and the vector score on others. Trust `rank`.
- **Common words count as keyword matches.** "the", "is" or "of" can make a hit read `via fts5+bge` with `fts5 0.0000`.
- **Highlight brackets run into other brackets.** A query word that occurs in a redaction placeholder is bracketed inside it (`[[API]_KEY:github]`, `[[EMAIL]]`), and the same happens with markdown: `[[[sourdough]]]` for a wikilink, `- [x] [run]` for a checkbox line.
- **Redaction wording.** `redacted in original: EMAIL@L4` means line 4 of your file holds an email address that recall replaced in its own copy. Your file is not changed. A key label such as `API_KEY:openai` names the pattern that matched, not who issued the key.
- **The stale check works on whole lines.** A hit reads "this passage changed" when any line the excerpt touches has changed, including the part of a line the excerpt cut off, so every character shown can still be in the file. When recall cannot place the excerpt at one spot in the indexed text (the same lines occur twice, or the highlighted word forms cannot be mapped back, for example), it says "this file changed" and does not judge the passage.
- **With `-c bge` alone the excerpt is the start of the note** (about 200 characters), not the part that matched. "This passage unchanged" then speaks about that opening text, not about the part your question was about, and an edit to a line the opening touches flags every such hit. The default channels show the matching passage.
- **Until `recall embed` runs, a new note or a saved observation has no vector.** It can rank below unrelated notes or be missing from the default results, and JSON `source` reads `fts5`. `-c fts5` finds it.
- **A renamed note reads as deleted** until the next `recall index` or `recall refresh`, which follow the rename.
- **Legacy counters.** `"embeddings": 0`, `legacy pooled`, `Live embeddings: 0`, `embeddings2 rows (Voyage)` and the `bge_vectors` zeros in `status`, `doctor health` and `gc` refer to an older vector table that only a store built by an old version still holds. On a new install the vectors in use are the ones counted as `chunk vectors` / `chunk_embedded_docs`. `recall refresh` never counts a `touch` as `touched-only`.
- **Rough edges in the output.** `cargo build` prints compiler warnings. `-c rg` hits have no excerpt line. `recall forget` says "reversible via DB", but no command undoes it.
- **Saved observations have no title or path**, so they are listed with an empty first line.
- **"top N" in the header** is the number of results returned, not the number asked for.
- **Three dates in the JSON.** `modified_at` is the file's mtime when it was indexed, `indexed_at` is when recall read it, `file_modified_at` is the mtime now and appears only when recall counts the file as changed.
- **Files in `~/.recall` you did not create:** `settings.json`, `refresh-state.json`, `vector-patch-*.ids` (left for a daemon to pick up; they accumulate when none runs), `prune-missing-*.txt` (one per `--prune-missing` run: document ids and paths, no note text), `error-log.jsonl` (commands that failed), `index.lock`.
- **`recall export` holds what the database holds.** With redacted storage on, that is the redacted text, so it is not a copy of your notes.
- **`recall index` does not drop deleted notes.** A note whose file was deleted stays searchable, flagged as deleted, until `recall refresh` or `recall index --prune-missing apply`.
- **Maintainer commands** (`bench`, `backfill-hashes`, `stamp-vectors-current`, `qwen3-*`) are listed in `recall --help`, marked as such. `whoami` ([docs/TENANCY.md](docs/TENANCY.md)), `settings` and `refresh` ([docs/FRESHNESS.md](docs/FRESHNESS.md)) have no entry in the Subcommands section below.

If this were continued, in this order:

Record a hash of the raw file next to the hash of the stored text, so a change inside a redacted value is flagged as a change.
Human query headings use the output scrubber, including after terminal-control normalization. Redaction is pattern based and is not a guarantee that every secret or personal detail is recognized.

---

Local-first memory search for your notes: SQLite FTS5 keyword search + in-process BGE-small
vectors (CPU, no GPU), fused into one ranking, optionally served by a warm daemon, with an MCP server so
Claude Code, Codex, Cursor and friends can search your notes. **Free and offline after install:
no API keys, no paid channels, nothing leaves your machine.** Indexes `.md` / `.qmd` files.

Windows x64 and Linux x64 have passed fresh prebuilt installation and functional checks.
Linux ARM64 and Apple Silicon macOS require the native workflow to pass before their
packages are offered. The [platform matrix](docs/PLATFORMS.md) records those boundaries.

## Quickstart

Prerequisites: [Rust](https://rustup.rs) (stable), `curl` (Linux/macOS), Python 3.10+ (only for the MCP server).

```bash
git clone https://github.com/kkamali04/lastknown.git && cd lastknown
# Linux / macOS
sh scripts/install.sh --vault ~/notes            # add --autostart to start the warm daemon at login
. ~/.recall/env.sh
```
```powershell
# Windows
powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Vault $HOME\notes   # -Autostart for the daemon
. $HOME\.recall\env.ps1
```

**What a fresh install takes** (measured by the `clean-install` CI job on 4-core GitHub runners,
starting from nothing but Rust and Python): about **4 minutes on Linux and 8 on Windows**, mostly
compiling; about **280 MB in `~/.recall`** (binary, ONNX Runtime, the BGE model, the MCP venv; 282 MB on Linux x64) plus
**0.4-0.5 GB of build output** in `target/`, which you can delete afterwards. The notes folder is
remembered as a setting; change it any time with `recall settings set vault <folder>` (then
restart the daemon if you run one: `recall daemon down`, `recall daemon up`).

The installer builds the binary, downloads ONNX Runtime 1.24.4 (Microsoft's pinned 125 MB NuGet package) and the BGE-small
model (pinned commit) with SHA-256 verification, puts everything under `~/.recall`, and ends
with `recall doctor embed`. **If the semantic channel cannot load, the installer fails loudly**
instead of silently giving you keyword-only search. It also sets up the MCP server (a private
python venv with `fastmcp`); add `--register-mcp` / `-RegisterMcp` to register it with Claude Code
and Codex if they are installed.

```bash
recall index                                   # scan your notes folder
recall embed                                   # vectorize
recall query --top 5 -- "how did we decide to back up the database"
```

> `recall embed --allow-benchmark-regression` appears in older instructions. The flag only matters on
> a maintainer's machine (one with `recall_dev_dir` set in `~/.recall/private-rules.json`), where
> `embed` is gated on that machine's benchmark results. Everywhere else there is no gate and the flag
> does nothing.

Check that results really came from both channels: the JSON `source` field must read `fts5+bge`
(or `bge`), never just `fts5-rescue`. See [docs/MCP.md](docs/MCP.md) to register the MCP server
in Claude Code, Codex, Cursor, and others (one command or config snippet each).

Configuration (all optional): the notes folder, set once with `recall settings set vault <folder>`
(or the `RECALL_VAULT` env var, which wins when set; default `~/recall-notes`; `recall status`
shows which one is in force), `RECALL_CHANNELS` (default `fts5,bge`), `RECALL_EXE` (MCP server:
path to the binary), `~/.recall/exclude.txt` (path segments to skip).
Known limit: recall indexes ONE notes folder; several roots are not supported yet. A running
daemon reads the folder when it starts, so restart it after changing `vault`.

**Colour.** Terminal output is plain text with no emoji. On a terminal, titles are bold, paths are dim (diagnostic scores are dim with `--verbose`), matched words are cyan (the `[brackets]` stay), and a stale or deleted
warning is bold yellow or red; every coloured span ends with a reset. Colour is off when the output
is piped or redirected, off when `NO_COLOR` is set, on when `CLICOLOR_FORCE` is set, and
`recall --color auto|always|never <command>` overrides both. JSON payloads are uncoloured;
an interrupted command attached to a real terminal can emit a terminal reset. Piped JSON does not receive that reset.

**Secrets and personal data.** recall replaces API keys, tokens, passwords in URLs, emails, phone
numbers and similar values with placeholders such as `[EMAIL]` in everything it returns. On a
profile with nothing indexed yet it also does so before storing (`scrub.store_redacted`, switched on
the first time recall opens the empty database, whichever command that is, and recorded in the
database so it does not depend on `settings.json`; recall also copies the choice into
`settings.json`, which is why `recall status --json` then names `settings.json` as the source),
so `recall.db` does not hold those values. An
index that already exists is not changed: if `recall status --json` shows `"store_redacted": false`
under `scrub`, your database holds the note text as written. That includes an
index first built by an earlier build with `recall settings set vault` (or the installer) run
before `recall index`. How to switch, and what the scrubber does not catch:
[docs/SCRUB.md](docs/SCRUB.md). It matches known patterns; it is not a security boundary.

`recall ui` opens a local page (127.0.0.1 only, per-launch token) for status, activity,
settings and a test search; `--no-open` prints the URL without opening a browser.
Licences of everything it downloads: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Build (manual)

```bash
cargo build --release     # default features = free/offline channels fts5 + bge
```

The `voyage`, `gemini` and `ann` Cargo features are opt-in extras (paid cloud channels / an
experimental ANN index) and are **not** needed.

Run the checks locally with `scripts/ci-local.ps1` (Windows) or `scripts/ci-local.sh` (Linux); see [CONTRIBUTING.md](CONTRIBUTING.md#ci).

## Cheat sheet (copy-paste)

```bash
# --- everyday ---
recall query "rust borrow checker lifetimes"   # fused recall, top 20
recall query "vector embeddings" -k 10         # top 10
recall save "we chose BGE-small for embeddings"  # remember an observation (keyword-searchable at once; `recall embed` adds its vector)
recall recent                                   # last 10 saved
recall recent -n 25                             # last 25 saved
recall status                                   # doc counts + db size + by-source

# --- index the vault ---
recall index                                    # build/refresh FTS5 from the vault
recall refresh                                  # pick up changed, new and deleted notes, re-embed only those
recall refresh --status                         # what the last scan found
RECALL_CONFIRM_RESET=yes recall index --reset true   # DESTRUCTIVE full rebuild (gated; the flag needs the value)

# --- consolidation / insight ("dreaming") ---
recall dream                                    # supersede near-duplicates (reversible)
recall insights                                 # surface recurring themes
recall insights -m 3                            # only themes with >=3 memories

# --- bi-temporal edits (reversible, nothing is deleted) ---
recall forget "old wrong note"                  # retire memories matching text
recall supersede 4213                           # retire one memory by id

# --- pipe / backup ---
recall export > brain-backup.json               # dump all valid memories as JSON

# --- gc (orphan embeddings) ---
recall gc                                       # dry-run: report orphaned embedding count
recall gc --apply                               # delete orphaned embeddings

# --- machine-readable output ---
recall query "rust lifetimes" --json            # emit JSON array of hits

# --- self-check ---
recall eval                                     # fast vector path vs an exact scan of the same vectors (1.0 unless built with `ann`)

# --- daemon (warm model + DB handles) ---
recall daemon status
```

| Channel (`-c`) | What it is |
|----------------|------------|
| `fts5`  | SQLite FTS5 keyword (bm25) |
| `bge`   | in-process BGE-small dense vectors (loads ONNX locally) |
| `graph` | keyword hits plus neighbouring notes from the same folders (a first version; there is no entity graph) |
| `rg`    | ripgrep parallel fallback over the vault |

Default is `fts5,bge`: keyword + in-process vectors (exact scan of every stored vector, fine for
personal-sized corpora; run `recall daemon up` to keep vectors warm). Use `-c fts5` for keyword only.

## Zero-cloud

The `recall` crate is fully offline:

- BGE-small embeddings run in-process from the ONNX model the installer downloaded (no network
  call at query or embed time).
- The `voyage` and `gemini` Cargo features (off by default) add query paths to cloud embedding
  APIs. They are maintainer-only: this repository has no command that fills the document-vector
  tables they read. See [docs/OPERATING.md](docs/OPERATING.md).
- The optional Haiku judge (`RECALL_JUDGE=haiku`, off by default) is the one thing in the default
  build that can send text off the machine, through your own Claude CLI.
- `recall query --json` reports in `source` which channel ranked each hit (`fts5`, `bge`,
  `fts5+bge`, or `fts5-rescue` when a requested channel failed). In the default build no external
  search channel is ever contacted by the Rust binary.

## Requirements

- A recent stable Rust (`Cargo.lock` is lockfile version 4, so Cargo 1.78 or newer; native release candidates use 1.96.0), a C toolchain for the bundled SQLite (MSVC Build Tools on Windows)
- ripgrep (`rg`) on PATH: `recall index` uses it to list the notes folder (the installers install it when a package manager is present)
- ~300 MB disk for the model and ONNX Runtime

## Subcommands

### `gc`

Prune embeddings that are orphaned from tombstoned or superseded documents.
Dry-run by default — shows counts without deleting anything.

**Flags:**
- `--apply` (optional): Execute the deletions (overrides dry-run mode).

```bash
recall gc           # dry-run: report orphaned embedding count
recall gc --apply   # delete orphaned embeddings
```

### `query`

Fused recall across the requested channels (default `fts5,bge`). The default pair is fused by weighted raw scores; other channel sets, and `RECALL_FUSION_MODE=rrf`, use reciprocal-rank fusion.
**Arguments:**
- `text` (required): The natural-language query to recall against.

**Flags:**
- `--top, -k` (optional): Number of results to return (default: `20`).
- `--channels, -c` (optional): Comma-separated channels to fuse.
- `--json, -j` (optional): Emit results as a JSON array. Fields: `doc_id`, `rank` (0-based), `score`, `vault_path`, `title`, `snippet`, `source`, `ranks`, `scores`, `rrf`, the freshness fields in [docs/FRESHNESS.md](docs/FRESHNESS.md), and `lacks` / `redactions` when they apply. Zero-cloud — only changes output format.
- `--via-daemon` (optional): Force routing through the warm daemon; errors if daemon is not running. Without this flag the daemon is used automatically when available.

### `save`

Capture an observation into the memory store. It is keyword-searchable at once; its vector is added by the next `recall embed`.
**Arguments:**
- `text` (required): The observation text to remember.

**Flags:**
- `--source, -s` (optional): Origin surface of the observation (cli, pi, opencode, codex, hermes, claude-code, vision).

### `status`

Store health: doc counts, per-source breakdown, superseded count, vector coverage, DB size, notes folder.

**Flags:**
- `--json, -j` (optional): Emit machine-readable JSON for CI smoke tests.

JSON fields: `docs_valid`, `docs_total`, `superseded`, `chunk_embedded_docs`, `coverage_pct`, `db_size_mb`, `vault`, `scrub`, and the legacy `embeddings`.

What two of the lines count:

- `chunk vectors` (`chunk_embedded_docs`, `coverage_pct`) is the number of documents that have at least one vector. It does not say the vector was made from the current text: after `recall index` has picked up an edited note, `recall embed` still has that note to redo, and after `recall save` the new observation has no vector until `recall embed`.
- `freshness` shows what the last scan recorded (`recall refresh`, or the watcher of a running daemon). It is not a live look at the folder: a note edited since then is not counted until the next scan. A query that returns that note does check it on the spot.

```bash
recall status
recall status --json
```

### `doctor`

Read-only health probes for agents and automation. These do not read `.env`, do
not call an LLM, and do not make paid calls.

```bash
recall doctor embed --json
recall doctor health --json
```

`doctor memory`, `doctor path-probe` and `doctor embedding-backlog` are developer checks over
recall's own eval scripts and artifacts, which are not part of this repository: all three need
`recall_dev_dir` in `~/.recall/private-rules.json` (`path-probe` also needs `--path-list`) and
stop with a message when it is missing.

`doctor embedding-backlog` runs the safe planner only. It reports counts for the
current unembedded backlog and writes a redacted `embedding_backlog_summary.json`;
it does not write a raw path list unless a separate planner command is explicitly
run with its confirmation flag.

### `daemon`

Manage the warm resident daemon (holds embedding model + DB handles).
**Arguments:**
- `action` (required): up | down | status

### `bench`

Run the maintainers' eval harness against a fixture set. The fixtures are not part of this repository: without `bench_fixtures` in `~/.recall/private-rules.json` the command prints "bench fixtures not found" and does nothing. For public benchmarks use [bench/README.md](bench/README.md); for a built-in self-check use `recall eval`.
**Flags:**
- `--overnight, -o` (optional): Run the continuous self-improving autoresearch loop (100s-1000s iterations). Requires value `true` or `1` (not a bare flag).
- `--iterations, -n` (optional): Number of benchmark iterations per run. Defaults to `50`.

### `index`

Build or refresh the keyword index from the notes folder (vectors are a separate step, `recall embed`).
**Flags:**
- `--reset, -r` (optional): DESTRUCTIVE rebuild from scratch — requires explicit confirmation. Requires value `true` or `1` (not a bare flag): `RECALL_CONFIRM_RESET=yes recall index --reset true`. A bare `--reset` exits 2. The stored vectors are dropped too, so run `recall embed` afterwards.
- `--prune-missing off|dry|apply` (optional): tombstone documents whose file was deleted (`dry` only lists them).


### `dream`

Offline consolidation pass: reconcile near-duplicate memories, supersede stale or redundant entries. Reversible (bi-temporal — nothing is deleted, only tombstoned).

```bash
recall dream
```

### `insights`

Surface recurring themes across saved memories via offline clustering.

**Flags:**
- `--min, -m` (optional): Minimum cluster size to surface (default: `2`).

```bash
recall insights            # themes with >=2 memories
recall insights -m 3       # only themes with >=3 memories
```

### `recent`

Show the most recently saved observations.

**Flags:**
- `--count, -n` (optional): Number of entries to show (default: `10`).

```bash
recall recent              # last 10 saved
recall recent -n 25        # last 25 saved
```

### `forget <text>`

Retire memories whose text matches the given string (bi-temporal — reversible, not deleted).

**Arguments:**
- `text` (required): Text to match against stored memories.

```bash
recall forget "old wrong note"
```

### `supersede <id>`

Retire a single memory by its integer id (bi-temporal — reversible).

**Arguments:**
- `id` (required): Memory id to supersede.

```bash
recall supersede 4213
```

### `export`

Dump all valid (non-tombstoned) memories as JSON to stdout (backup / piping).

```bash
recall export > brain-backup.json
```

### `index-ann`

Build (or rebuild) the HNSW ANN index from existing BGE vectors already stored in `recall.db`. Does **not** re-embed anything — reads existing vectors only. Only available when compiled with `--features ann`.

```bash
cargo build --release --features ann
recall index-ann
```

### `embed`

Vectorize indexed docs with BGE-small (resumable; only embeds docs with no vector, or with a vector made from older text).
**Flags:**
- `--batch, -b` (optional): Number of docs to embed per batch (default: 256).

```bash
recall embed          # embed all un-vectorized docs in batches of 256
recall embed -b 64    # smaller batches (lower peak memory)
```

### `eval`

Measure recall@k of the active vector search path against exact brute-force ground truth, over a short list of built-in probe phrases (not taken from your notes). What the number means: the fast vector path checked against an exact scan of the same vectors, so it is 1.0 by construction unless the build has the `ann` feature. It says nothing about how well recall finds things in your notes.
**Flags:**
- `--top, -k` (optional): Number of results per query to evaluate (default: 5).
- `--min-recall` (optional): Exit non-zero if mean recall@k falls below this threshold (CI gate). Off by default.

```bash
recall eval                        # report recall@5 (always exits 0)
recall eval -k 10                  # report recall@10
recall eval --min-recall 0.99      # gate for an `ann` build: fail if mean recall@5 < 0.99
```
