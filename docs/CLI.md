# CLI reference

Run examples from the repository or extracted package root after installation.
The executable is named `recall`. Source builds use `cargo build --release --locked`; prebuilt installation needs no Rust.

See [guided setup and settings](SETUP.md), [installation](PLATFORMS.md#prebuilt-packages),
[known limits](LIMITS.md), [MCP setup](MCP.md), and [optional capture hooks](HOOKS.md).

## Build and install from source

Requires Rust (native releases use 1.96.0), a C toolchain for bundled SQLite, Python 3.10+
with venv, ripgrep and internet access for the pinned runtime/model/MCP dependencies.
On Windows, use MSVC Build Tools. Allow roughly 300 MB for model/runtime plus build output.

```sh
git clone https://github.com/kkamali04/lastknown.git
cd lastknown
sh scripts/install.sh --vault "$HOME/notes"
. "$HOME/.recall/env.sh"
```

```powershell
# Windows, from the source checkout
powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Vault "$HOME\notes"
. "$HOME\.recall\env.ps1"
```

The installer checks the pinned runtime/model hashes and semantic backend. Add
`--register-mcp` / `-RegisterMcp` to register supported clients, or `--autostart` /
`-Autostart` to request the daemon at login. These are optional installation mutations.
For a binary build alone, run `cargo build --release --locked`; model/runtime setup remains
necessary for BGE. After installation run `recall index`, `recall embed`, then query.

Configure a different folder with `recall settings set vault <folder>`; restart a running
daemon after changing it. `RECALL_VAULT` overrides the setting. Scope is one notes folder.

## Everyday commands
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

Default embedding and search run locally after installation:

- BGE-small embeddings run in-process from the ONNX model the installer downloaded (no network
  call at query or embed time).
- The `voyage` and `gemini` Cargo features (off by default) add query paths to cloud embedding
  APIs. They are maintainer-only: this repository has no command that fills the document-vector
  tables they read. See [docs/OPERATING.md](../docs/OPERATING.md).
- The optional Haiku judge (`RECALL_JUDGE=haiku`, off by default) is the one thing in the default
  build that can send text off the machine, through your own Claude CLI.
- `recall query --json` reports in `source` which channel ranked each hit (`fts5`, `bge`,
  `fts5+bge`, or `fts5-rescue` when a requested channel failed). In the default build no external
  search channel is ever contacted by the Rust binary.

## Requirements

- Rust (native release candidates use 1.96.0), a C toolchain for the bundled SQLite (MSVC Build Tools on Windows)
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
- `--json, -j` (optional): Emit results as a JSON array. Fields: `doc_id`, `rank` (0-based), `score`, `vault_path`, `title`, `snippet`, `source`, `ranks`, `scores`, `rrf`, the freshness fields in [docs/FRESHNESS.md](../docs/FRESHNESS.md), and `lacks` / `redactions` when they apply. Zero-cloud — only changes output format.
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

Run the maintainers' eval harness against a fixture set. The fixtures are not part of this repository: without `bench_fixtures` in `~/.recall/private-rules.json` the command prints "bench fixtures not found" and does nothing. For public benchmarks use [bench/README.md](../bench/README.md); for a built-in self-check use `recall eval`.
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
