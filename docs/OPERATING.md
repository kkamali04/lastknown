# recall — Operating / Deploy Reference

`recall` is a local-first Rust memory search: SQLite FTS5 keyword search plus in-process
BGE-small (384-dim) vectors, fused into one ranking, served directly or through an optional warm
daemon. The default build is offline after the installer's one-time downloads.

> Source of truth: `recall --help` and `recall <command> --help`. The defaults below were read from
> the help output of a build of this tree. `recall --version` prints `0.1.0`.

## Commands

| Subcommand | What it does | Flags (defaults) |
|---|---|---|
| `query <text>` | Search across the requested channels and print one fused ranking | `-k/--top` (`20`), `-c/--channels` (`fts5,bge`; also `rg`, `graph`), `-j/--json`, `--via-daemon`, `--no-daemon`, `--allow-fallback`, `--verbose` |
| `save <text>` | Capture one observation into the store | `-s/--source` (`cli`), `--share agent\|project\|team\|global` (see [TENANCY.md](TENANCY.md)) |
| `index` | Build or refresh the keyword index from the notes folder | `-r/--reset true` (destructive; also needs `RECALL_CONFIRM_RESET=yes`), `--path-list`, `--dry-run`, `--prune-missing off\|dry\|apply` (`off`) |
| `embed` | Vectorize indexed docs with BGE-small (resumable; only docs with no vector, or with a vector made from older text) | `-b/--batch` (`256`), `--path-list`, `--dry-run`, `--force`, `--allow-benchmark-regression` (maintainer gate only) |
| `refresh` | Re-sync the index with files that changed on disk, then re-embed only those (see [FRESHNESS.md](FRESHNESS.md)) | `--scan-only`, `--no-embed`, `--force`, `--if-due`, `--status`, `-j/--json` |
| `settings` | View or change `~/.recall/settings.json` | `settings`, `settings get KEY`, `settings set KEY VALUE` |
| `status` | Doc counts, vector coverage, DB size and path, notes folder, scrub state | `-j/--json`, `--min-coverage <pct>` (gate; off by default) |
| `daemon <action>` | Warm resident daemon (holds the model and DB handles) | `up` (stays in the foreground), `down`, `status` |
| `ui` | Local web page on 127.0.0.1 with a per-launch token: status, settings, test search | `--no-open` |
| `doctor [check]` | Health probes | `embed` (default), `health`; `-j/--json`, `--timeout-seconds` (`20`). `memory`, `coverage`, `path-probe`, `embedding-backlog` and `safe-embed-batch` are maintainer checks |
| `eval` | recall@k of the active vector path against an exact scan, over built-in fixture queries | `-k/--top` (`5`), `--min-recall <f32>` (gate; off by default) |
| `whoami` | Print the tenancy reader this process runs as | none |
| `recent` | Most recent saved observations | `-n/--count` (`10`) |
| `forget <text>` | Retire memories matching text (reversible, not deleted) | none |
| `supersede <id>` | Retire one memory by integer id (reversible) | none |
| `dream` | Offline consolidation: supersede near-duplicate saved observations | none |
| `insights` | Recurring themes across saved observations (offline clustering) | `-m/--min` (`2`) |
| `export` | Dump all valid memories as JSON | none |
| `gc` | Prune vectors orphaned by tombstoned or superseded documents (dry-run by default) | `--apply`, `--exclude-aware` |
| `index-ann` | Build the HNSW index from stored BGE vectors | only in a build with `--features ann` |
| `bench` | The maintainers' fixture harness | needs `bench_fixtures` in `~/.recall/private-rules.json`; the fixtures are not in this repository |

`recall --help` also lists `backfill-hashes`, `stamp-vectors-current` and the `qwen3-*` commands.
They are maintainer migrations and experiments and are not covered here.

### CPU use of `embed`

`recall embed` is the one CPU-heavy command: it uses every core it is allowed to. recall does not
set its own affinity or priority, so on a shared machine limit it from outside, for example on
Linux:

```bash
taskset -c 0-3 nice -n 19 recall embed
```

A query embeds one short string and scans the stored vectors; `recall daemon up` keeps the model
and the vectors in memory between queries.

## Measuring quality

`eval` compares the active vector search path against an exact brute-force scan over the built-in
fixture queries and reports mean recall@k.

```bash
recall eval -k 5 --min-recall 0.99
```

`--min-recall` turns `eval` into a gate: it exits non-zero when mean recall@k falls below the
threshold, and also fails if there are no vectors to measure. Without `--min-recall`, `eval` only
reports and always exits 0.

What the number means: the fast vector path checked against an exact scan of the same vectors, so it is 1.0 by construction unless the build has the `ann` feature.
The probes are a short built-in list of phrases, not your notes, and the number says nothing about
retrieval quality on your notes. For that, see the public benchmarks in
[../bench/README.md](../bench/README.md).

## Deploy notes

- **No network in the default build.** The Rust binary opens no outbound connection. Its only
  sockets are loopback TCP (`127.0.0.1`, OS-assigned port) for the daemon and for `recall ui`.
  The one exception is opt-in: `RECALL_JUDGE=haiku` runs your own `claude` CLI as a child process,
  which sends the question and its top passages to Anthropic.
- **Secrets.** The default build reads no API key and needs none.
- **Cloud channels are maintainer-only.** The `voyage` and `gemini` Cargo features add query
  paths that embed the query text with a cloud API and search document vectors stored in extra
  tables. Nothing in this repository fills those tables, so the features are of no use to anyone
  but the maintainers, like `ops-health`. For the record, this is how they find a key: `voyage`
  first runs `cogent secrets get VOYAGE_API_KEY` (`cogent` is the maintainers' private secret
  tool; when it is not installed that step is skipped) and then falls back to the
  `VOYAGE_API_KEY` environment variable; `gemini` uses Google Application Default Credentials or
  the `GEMINI_API_KEY` / `GOOGLE_API_KEY` environment variable.
- **Model and runtime.** BGE-small (`Xenova/bge-small-en-v1.5`, the ONNX export of
  `BAAI/bge-small-en-v1.5`, 384-dim, CPU) is read from `~/.recall/models`; recall never downloads
  it itself, the installer does. ONNX Runtime is loaded dynamically from the library beside the
  binary (`libonnxruntime.so`, `libonnxruntime.dylib` or `onnxruntime.dll`); `ORT_DYLIB_PATH`
  overrides the location.
- **ANN feature:** off by default (`default = []`). Build with `--features ann` (`hnsw_rs`) to get
  the `index-ann` subcommand; the index persists under `~/.recall/`.
