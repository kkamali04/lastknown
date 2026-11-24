# Set up Lastknown

You can give an agent the [repository link](https://github.com/kkamali04/lastknown) and ask
it to follow [INSTALL_FOR_AGENTS.md](../INSTALL_FOR_AGENTS.md), or use this walkthrough.
There is no mandatory account, provider key or LLM for default search.

## Decide what to connect

| Choice | Start here |
|---|---|
| Notes | One existing `.md`/`.qmd` folder, or a new empty folder. Keep private notes outside the public source checkout. |
| Client | CLI only, Claude Code, Codex, Cursor/Desktop or another stdio MCP client. |
| Session capture | Off initially. Enable the project-local Claude Code hooks only if you want session text copied into notes. |
| Daemon | Start manually if repeated searches are slow; autostart is optional. |

## Install the runtime

Get the matching [v0.1.0-alpha.1 archive](https://github.com/kkamali04/lastknown/releases/tag/v0.1.0-alpha.1)
for Windows x64, Linux x64/ARM64 (glibc), or macOS Apple Silicon. Check its published
checksum and extract the entire archive. Intel Mac, musl and Windows ARM64 are not
validated packages. See [platform details](PLATFORMS.md).

You need Python 3.10+ with venv, ripgrep (`rg`), and internet during installation.
The installer can try a system package manager if ripgrep is missing; install `rg` yourself
first if you want to manage that separately. It downloads and verifies the pinned ONNX
runtime and BGE model, creates a private MCP venv, and writes the normal `~/.recall` profile.
It does not install hooks or enable session capture.

From the extracted package directory:

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File .\Install-Lastknown.ps1 -Vault "$HOME\notes"
. "$HOME\.recall\env.ps1"
```

```sh
# Linux / macOS
sh ./Install-Lastknown.sh --vault "$HOME/notes"
. "$HOME/.recall/env.sh"
```

For a no-profile trial, use the [temporary demo/acceptance check](TRY-IT.md) instead.
If you already use Recall/Lastknown, preserve the existing profile and discuss an upgrade
before installing over it. No notes migration is needed for this documentation/integration update.

## Choose your settings

Run `recall settings` to inspect the values, or `recall ui` for the local settings/search
page. An agent can ask you these questions instead of making all the choices silently:

| Question | Setting / behavior |
|---|---|
| Which folder? | `recall settings set vault "/absolute/path/to/notes"`. A `RECALL_VAULT` environment value overrides it. Restart the daemon after changing folders; an existing index is not automatically replaced or migrated. |
| When should notes refresh? | Default `nightly` at `07:00` UTC. Alternatives: `recall settings set refresh on-change` or `recall settings set refresh "every 30 minutes"`. Automatic work needs the running daemon or an external scheduled invocation. |
| Should changed passages be flagged? | Keep `freshness_check` at `lazy` (default). `off` makes freshness unknown rather than proving it current. |
| Redact secrets/PII before storage? | New profiles default to redaction. Inspect `scrub.secrets`, `scrub.pii`, `scrub.store_redacted`; existing profiles keep their prior settings. Custom terms use `scrub.custom`. Redaction is pattern-based, not a guarantee. |
| Reorder for age/staleness? | Leave `recency_boost` and `stale_penalty` at zero unless you want to experiment. Tests did not establish a better default. |
| Enable answerability/LLM calls? | Leave the optional Haiku judge off. It is not needed for semantic search. |

See [freshness](FRESHNESS.md) and [redaction](SCRUB.md) before changing those policies.
Ordinary indexing does not retroactively sanitize every old stored observation.

## Build the first index and search

```sh
recall index
recall embed
recall doctor embed
recall status --json
recall query --top 5 -- "a question answered by your notes"
```

Indexing/embedding writes the local store, not your existing Markdown files. Embedding is
the CPU-heavy step. A saved observation or captured note is keyword-searchable first;
run `recall embed` to give it semantic coverage. Search returns evidence, not a generated answer.

## Add the client, skill and hooks

1. Register only the requested client using [MCP.md](MCP.md). The runtime installer already
   installed the server in `~/.recall/mcp` and its Python in `~/.recall/venv`.
2. Read [the skill](../skills/lastknown/SKILL.md) in place, or copy its whole folder into
   your client's supported skills directory. Common locations: `~/.claude/skills/lastknown`
   for Claude Code, `~/.agents/skills/lastknown` for Codex. Retain an existing skill until
   its replacement is reviewed. Restart/reload the client if needed.
3. If desired, add [Claude Code hooks](HOOKS.md). Keep the current repo checkout: those
   new integrations are not bundled in the frozen alpha archives. Other clients get
   MCP/CLI search; this repo does not supply their automatic transcript capture.
4. In the client, call health, search, semantic search, recent and identity. Ask a question
   whose source you can inspect. Registration alone is not a successful connection test.

## Keep it running and undo optional integration

`recall daemon up` runs in the foreground and keeps the model warm; use another terminal
for queries. `recall daemon down` stops it. Installer autostart flags and host-specific
removal commands are in [CLI.md](CLI.md#build-and-install-from-source) / the installer output.
Starting the daemon is different from enabling login autostart.

To undo integration, remove only the `recall` client entry, copied Lastknown skill and
Lastknown hook entries you added; restore the config backup if appropriate. Existing notes
and `~/.recall` remain. For capture removal/retention details, see [HOOKS.md](HOOKS.md).
