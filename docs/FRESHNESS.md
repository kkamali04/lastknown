# File-change freshness

A recall hit is a copy of a file as it was when last indexed. This feature makes every result say
when the file has since changed, and keeps the index in step with the disk.

## What a result tells you

`recall query --json` (and so both MCP tools) add, per hit:

| field | meaning |
|---|---|
| `stale` | `true` the file was deleted, or it changed and a line the passage shown is on has changed, or it changed and recall could not check the passage (past the hashing budget, no usable excerpt, or an excerpt that cannot be placed at exactly one spot in the indexed text); `false` the file matches the index, or it changed but the passage shown is still in it (then `stale_note` and `file_modified_at` are present); `null` could not check (see `stale_note`) |
| `deleted` | `true` only when the file is gone from disk |
| `modified_at` | the file's mtime when recall indexed it (the recency signal) |
| `indexed_at` | when recall last read the file |
| `file_modified_at` | the file's mtime now; present only when recall counts the file as changed (see Known limits: a change it cannot see leaves this out) |
| `stale_note` | the reason, for example: "file changed, this passage unchanged: the file changed on <date> after recall indexed it on <date>, but the excerpt shown is still in it."; "this passage changed: the file changed on <date> after recall indexed it on <date>, and a line the excerpt shown is on has changed; read the actual file."; "this file changed on <date> after recall indexed it on <date>; read the actual file." (changed, passage not checked); "this file was deleted from disk after recall indexed it on <date>; the indexed text is a copy of a file that no longer exists."; "freshness not checked: <why>" (for example `freshness_check is off`) |

The text output prints a one-line banner above the list and a `!! STALE:` line under each stale hit; a hit whose file changed elsewhere gets a `~ file changed, this passage unchanged` line and no banner.
The check reads each returned hit's file and compares its SHA-256 with the indexed text (at most `--top`
files and at most 2 MiB of file content per query, best hits first; past that budget a hit is judged by size and mtime alone), so a `touch` or a re-save of identical bytes is not flagged and an edit that restores the
file's mtime is, as long as that hit is inside the budget (past it, a same-size edit with a restored mtime is missed). `freshness_check = off` turns it off, and the output then says `stale: null` rather than
staying silent.

## Settings (`~/.recall/settings.json`, `recall settings [list|get KEY|set KEY VALUE]`)

| key | default | values |
|---|---|---|
| `refresh` | `nightly` | `on-change`, `nightly`, `every N minutes`, `every N hours` (5 min to 7 days) |
| `nightly_at_utc` | `07:00` | `HH:MM` UTC (07:00 UTC is about 02:00-03:00 US Eastern) |
| `freshness_check` | `lazy` | `lazy`, `off` |
| `stale_penalty` | `0` (OFF) | rank positions a stale hit is demoted by, 0-10 |
| `recency_boost` | `0` (OFF) | rank positions the newest file can gain, 0-10, 30-day half-life |

The two ranking knobs are unmeasured. They stay 0 until the retrieval-quality run says otherwise. They
re-order only inside the returned top-k.

## How it works

1. **Record** (daemon thread, `notify`): file edits/deletes mark the document `stale = 1/2`; created or renamed-in
   files go into `fs_events`. Folder renames arrive as one directory event and are expanded. Hidden and build
   directories are ignored. It never embeds and never rewrites a body.
2. **Scan** (`recall refresh --scan-only`, the daemon's startup catch-up, hourly while the watcher is up, every 10 min
   when it is down): compares every live vault document to the disk by mtime (nanoseconds) + size, hashing only when
   they differ. Rows written before this feature have no recorded stat; they are hashed in batches of 20,000 per run
   until stamped.
3. **Refresh** (`recall refresh`): rewrites changed bodies in place (doc ids kept; `embedded_sha256` untouched so the
   existing stale-hash queue sees them), follows renames with `find_moved_document` (the same rule `ingest_dir` uses;
   a rename plus a copy of the same content claims nothing), tombstones deletions, then re-embeds only the changed
   documents with `recall embed --path-list <list> --force` at below-normal priority.
4. **Daemon**: any refresh (the daemon's own or a `recall refresh` typed by hand) drops the changed document ids,
   including tombstoned ones, into `~/.recall/vector-patch-*.ids`; the daemon worker collects them and the accept
   loop swaps those rows in its in-RAM vector matrix (no 17 s reload). A failed read is retried on the next query.
5. **Retry**: if the embed step fails, the paths are kept in `refresh-state.json` (`pending_embed`), retried by the next
   refresh with a back-off (60 s doubling to 30 min), and the nightly clock is not advanced.

Safety rails: one writer at a time (`~/.recall/index.lock`, an OS advisory lock released on crash; `recall index` waits
up to 20 min for it, a refresh gives up immediately and says SKIPPED); skips while CPU is above 75 % (override with
`recall refresh --force`, tune with `RECALL_REFRESH_MAX_LOAD`); a refresh tombstones at most max(50, 2 % of live docs)
and leaves a larger batch flagged for `recall index --prune-missing`; a missing vault root refuses outright.
`RECALL_NO_WATCHER=1` starts the daemon without the worker.

A scheduled task can call `recall refresh --if-due`, which exits 0 with "not due" when the setting says nothing is due.

## Platform status

| | Windows | Linux | macOS |
|---|---|---|---|
| unit + integration tests | pass (the suite is developed and run on Windows x64) | pass on a Linux x64 VPS with ripgrep installed (`scripts/ci-local.sh`, steps `test` and `ops`) | **not run** |
| watcher backend | ReadDirectoryChangesW, exercised in a sandbox daemon | inotify, compiled and unit-tested only; the live watcher was not exercised on Linux | FSEvents, **untested** |

### macOS: code paths nobody has run

1. The FSEvents watcher itself (`notify` feature `macos_fsevent`): event coalescing, whether a folder rename arrives as
   one directory event or as per-file events (both are handled), and rename events arriving as `Modify(Name)`.
2. FSEvents reports resolved paths (`/private/var/...`) where the vault may be configured as `/var/...`. `event_rel_any`
   tries the configured root and its canonical form (unit-tested with strings only). The hourly scan is the backstop if
   an event is still dropped.
3. Case-insensitive APFS: `case_insensitive_fs()` returns true on macOS and the row lookup falls back to `lower(path)`.
   Only the Windows spellings were tested.
4. CPU back-off reads `sysctl -n vm.loadavg` because macOS has no `/proc/loadavg`. The parser strips `{ }` and takes the
   first number; never run against real output.
5. The embed child runs under `nice -n 19` (as on Linux). `File::try_lock` is `flock` on both; macOS not exercised.
6. `rg --files` enumeration for new-file discovery uses the same code as `recall index`; macOS not exercised.

## Known limits

- recall compares the text as it stores it. With redacted storage on (the default for a new install) the stored text
  holds placeholders such as `[EMAIL]`, so a change confined to a redacted value (one email address replaced by another,
  a different key) leaves the stored text and its hash the same: no stale line on a query, and `recall refresh` and
  `recall index` count the file as unchanged. The placeholder shown already tells the reader to open the file for the
  value. A uniform line-ending flip (all LF to all CRLF or back) is not a change either, by design.
- The passage check works on whole lines: a hit is "this passage changed" when any line its excerpt touches has
  changed, including the part of a line the excerpt cut off. With `-c bge` alone the excerpt is the start of the note
  (about 200 characters), not the part that matched, so its verdict is about that opening text.
- The bulk scan is stat-based. An edit that keeps both the size and the mtime (to the nanosecond) is invisible to it;
  a query catches it (content hash) while the file is inside the per-query hashing budget (2 MiB, best hits first), and so does the next full `recall index`.
- A rename is only followed when the created-file queue came from a full enumeration in the same refresh. A queue built
  from watcher events alone inserts instead, because "exactly one new file has this body" is not knowable from events.
- Windows Unicode case equivalence beyond simple lower-casing is not modelled; SQLite `lower()` is ASCII only, so two
  paths differing only in the case of a non-ASCII letter can be treated as different files.
- A rename that is also an edit looks like a delete plus a new file: the old row is tombstoned and the new file gets a new
  id (its vectors are rebuilt). Only an unchanged-content rename keeps the id.
- Only BGE is re-embedded. Gemini, Voyage and Qwen3 vectors for a changed document stay stale until their own queues run.
- A changed document's old vectors keep serving until the re-embed finishes; the `stale` flag covers that window.
- On a maintainer machine (one with `recall_dev_dir` set in `~/.recall/private-rules.json`) the refresh's embed step
  is behind the embedding benchmark gate: with no benchmark artifact it reports `embed: NOT DONE` and leaves the
  documents queued. Other installs have no such gate.
- inotify has a per-user watch limit (`fs.inotify.max_user_watches`); if the watch fails the worker logs it and
  falls back to periodic scans.
