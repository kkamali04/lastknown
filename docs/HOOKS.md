# Optional Claude Code hooks and session capture

MCP provides read-only search. Hooks integrate the session lifecycle. Both are optional;
installing Lastknown or copying its skill does not enable hooks.

The supplied [hook adapter](../integrations/claude-code/hooks.py) has two events:

| Event | Action |
|---|---|
| `SessionStart` | Checks the configured binary version and supplies a short availability reminder. It does not search or copy transcripts. |
| `SessionEnd` | Captures that session's user/assistant text and subagents from the configured Claude transcript root; scrubs it, writes private dated notes, indexes them and verifies stored-body readback. |

Capture uses [memcap.py](../ingest/memcap.py), with no LLM calls, tool I/O copying or
Claude-mem database import. Notes go under `<vault>/.memory-capture/claude-code/`;
per-session state, pending-work journal and health live under
`~/.recall/capture-hooks/<session-uuid>/`. A separate folder/state prevents mixing this
hook's records with the older all-history importer.

## Enable for a project

1. Finish [runtime and MCP setup](SETUP.md). Keep a checkout of the current repository:
   the new adapter and scoped-capture options are not in the old alpha archives.
2. Decide whether the user wants session capture for this project. Prefer that scope
   over global capture. A global hook would capture sessions from every project using it.
3. Back up the project's `.claude/settings.local.json`. Merge the chosen entries from
   [hooks.example.json](../integrations/claude-code/hooks.example.json), preserving
   other settings and hooks. Use just `SessionStart` for availability without capture.
4. Replace `/ABSOLUTE/PYTHON` with the installed venv's real Python executable,
   `/ABSOLUTE/lastknown` with this checkout, and `/ABSOLUTE/notes` with the chosen notes.
   Exec-form `command` + `args` avoids shell quoting; use a current Claude Code version
   that supports it. Windows JSON paths can use forward slashes. Do not use `py.cmd` shims.
5. If Claude uses a non-default config root, add `--projects` and its absolute `projects`
   directory to the SessionEnd arguments. `CLAUDE_CONFIG_DIR` is respected by default.
   If the Lastknown profile/binary are non-default, add `--profile` / `--recall-exe`.
   A custom profile must be a `.recall` directory under its intended home. The adapter
   binds that parent home for every recall subprocess without changing the caller's environment.
6. If the notes folder is in Git, ignore `.memory-capture/` in that vault and verify it
   has no tracked capture files. The importer refuses a capture folder exposed to Git.
   A dot-folder alone does not make data private from backup/sync tools; choose those separately.

The template sets a 30-second SessionEnd budget. Current Claude Code otherwise defaults
to 1.5 seconds. Older clients may need `CLAUDE_CODE_SESSIONEND_HOOKS_TIMEOUT_MS=30000` as
well. Follow the [official hook contract](https://code.claude.com/docs/en/hooks#sessionend)
for your client version. Forced termination or a timeout can interrupt capture; the
original transcript remains the recovery source. A hook is not a guaranteed backup.

## Verify once before relying on it

Start a short Claude Code session in that project, mention a harmless distinctive
decision, and exit normally. Then inspect
`~/.recall/capture-hooks/<session-uuid>/memcap-health.json`:

- `exit_code: 0`, a fresh `last_success`, and verified indexing mean that run completed.
  A skip is not new capture success; check `skipped`, `pending_index`, `groups_deferred`,
  `pending_bytes` and error fields. A scoped run with deferred subagents returns partial.
- `exit_code: 2` means preflight refused; `3` means partial/unconfirmed work. Check the
  health record rather than assuming the host's session ended successfully means capture did.
- Query the distinctive decision with `recall query --channels fts5 -- "your phrase"`.
  Then run `recall embed` and test semantic search if you want semantic coverage.

The adapter never trusts an input `transcript_path` as an arbitrary file to read. It
validates a canonical session UUID and selects only that identity and its subagents
inside the configured transcript root. New/unsupported host transcript formats can fail
or produce partial captures. Synthetic adapter tests do not replace this host check.

The importer skips its historical 02:15–03:45 local nightly-maintenance window. On
Windows it also checks for a running `Recall-Nightly-Index` task; Unix does not call
Windows task tools. A skipped/interrupted session needs a later explicit retry:

```sh
python /path/to/lastknown/integrations/claude-code/hooks.py session-end --session-id SESSION_UUID --vault /path/to/notes --projects /path/to/.claude/projects --profile /path/to/.recall --recall-exe /path/to/.recall/bin/recall
```

Replace the UUID placeholder with the real session UUID, use the same source and
state paths as the hook, and run outside the maintenance window. `--max-bytes` is a
batch-admission budget; the first large source can exceed it and is never silently truncated.

## Privacy, exclusions and removal

Only user/assistant text is retained; this is not the full original session with tool I/O.
The transcript scrubber runs before writing and the runtime may redact again before
storage. Patterns can miss secrets or useful context; inspect notes before sharing them.
Historical plans and assistant claims stay historical, not verified facts or current instructions.

Remove the Lastknown hook entries to stop future automatic capture. Existing notes,
vectors and state are retained. To exclude an already captured session, add its UUID to
that session's state-directory `memcap-exclude.txt`, then explicitly retry that session
to withdraw its owned capture files. **Indexed copies can remain searchable until an
explicit index cleanup.** Review `recall index --prune-missing dry` and the affected paths
before deciding to apply pruning; the hook never runs broad pruning automatically.
Do not hand-edit its ownership journal. This does not erase original Claude transcripts
or independent backups.

Optional AI summaries are a separate experiment, off by default. They are not part of
this hook and may invoke a provider if enabled; see [the summary-layer contract](ai-summary-layer.md).
