# Connect Lastknown over MCP

Start with [the setup walkthrough](SETUP.md). Installing the runtime also installs a
private Python venv and a copy of the MCP server. Register that installed copy, so the
client does not depend on a checkout's Python environment. MCP does not enable capture;
see [hooks](HOOKS.md) for the separate opt-in integration.

`mcp/recall_mcp.py` is the one supported MCP server (stdio, read-only). It shells out to the
`recall` binary and follows the safe-invocation contract: **flags first, then `--`, then exactly
one positional** (`recall query --top 8 --channels fts5,bge --json -- "<text>"`), so a search
string such as `--no-daemon` can never be parsed as a flag.

## Tools

| Tool | What it does |
|---|---|
| `recall_search(query, top=8, channels="fts5,bge")` | hybrid keyword + vector search; trust `rank`, never `score` |
| `recall_search_semantic(query, top=8)` | vector-only (finds notes by meaning) |
| `recall_recent(count=10)` | most recently saved memories |
| `recall_whoami()` | the tenancy identity the server runs as and the boxes it may read (see [TENANCY.md](TENANCY.md)) |
| `recall_health()` | binary path, index status, and a per-channel probe that reports a dead channel instead of hiding it behind the keyword tier |

The server fails loudly: a dead backend raises an error, it never returns an empty list that
looks like "no results". It never writes to the index (`index`, `embed`, `save`, `gc` are
deliberately not exposed).

## Register an installed runtime

Inspect an existing `recall` entry before changing it. Preserve other servers, and choose
the requested project/user scope. These Claude Code examples use user scope; omit
`--scope user` for its default project-local registration. Codex registration follows
the installed CLI's configuration rules; check `codex mcp add --help` first.

Windows PowerShell:

```powershell
claude mcp add --scope user recall -e "RECALL_EXE=$HOME\.recall\bin\recall.exe" -- "$HOME\.recall\venv\Scripts\python.exe" "$HOME\.recall\mcp\recall_mcp.py"
# Or Codex:
codex mcp add recall --env "RECALL_EXE=$HOME\.recall\bin\recall.exe" -- "$HOME\.recall\venv\Scripts\python.exe" "$HOME\.recall\mcp\recall_mcp.py"
```

Linux/macOS:

```sh
claude mcp add --scope user recall -e RECALL_EXE="$HOME/.recall/bin/recall" -- "$HOME/.recall/venv/bin/python" "$HOME/.recall/mcp/recall_mcp.py"
# Or Codex:
codex mcp add recall --env RECALL_EXE="$HOME/.recall/bin/recall" -- "$HOME/.recall/venv/bin/python" "$HOME/.recall/mcp/recall_mcp.py"
```

The installed server reads the durable `vault` setting; leaving `RECALL_VAULT` out lets
`recall settings set vault` take effect after any necessary daemon restart. If you set
`RECALL_VAULT` explicitly, it overrides that setting. One profile supports one notes root.

For Cursor/Desktop and other `mcpServers` clients, merge an entry like this, replacing
each path with its real absolute path (no literal `~`/`$HOME` in JSON):

```json
{
  "mcpServers": {
    "recall": {
      "command": "/home/you/.recall/venv/bin/python",
      "args": ["/home/you/.recall/mcp/recall_mcp.py"],
      "env": { "RECALL_EXE": "/home/you/.recall/bin/recall" }
    }
  }
}
```

On Windows substitute the venv's `Scripts/python.exe`, server and `recall.exe` paths.
Restart/reload the client, discover all five tools, then call health, both search tools,
recent and identity. An entry in a settings file alone does not verify a connection.
The [Lastknown skill](../skills/lastknown/SKILL.md) supplies retrieval/use guidance.

## Development or custom environment

```bash
python -m pip install -r mcp/requirements.txt      # inside your chosen venv
python mcp/smoke_handshake.py                       # real MCP client handshake, then calls health, search, semantic search and recent
```

The server finds the binary via `RECALL_EXE`, then `~/.recall/bin/recall` (where the installer puts
it), then `target/release/`, then `PATH`. Pass `RECALL_VAULT` so it reads the right notes folder.
Use absolute paths below; replace `/path/to/recall` with your checkout, and `python` with
the actual interpreter where you installed fastmcp. These are development alternatives
to the installed-runtime configuration above.

### Claude Code

```bash
claude mcp add recall -e RECALL_VAULT="$HOME/notes" -- python /path/to/recall/mcp/recall_mcp.py
```

### Codex

```bash
codex mcp add recall --env RECALL_VAULT="$HOME/notes" -- python /path/to/recall/mcp/recall_mcp.py
```

### Cursor (`~/.cursor/mcp.json` or `<project>/.cursor/mcp.json`), Claude Desktop (`claude_desktop_config.json`), Windsurf, and other `mcpServers` clients

```json
{
  "mcpServers": {
    "recall": {
      "command": "python",
      "args": ["/path/to/recall/mcp/recall_mcp.py"],
      "env": { "RECALL_VAULT": "/home/you/notes" }
    }
  }
}
```

On Windows use `"command": "python"` (or the full path to `python.exe`) and forward-slash or
doubled-backslash paths in JSON.

### Environment

| Variable | Meaning |
|---|---|
| `RECALL_VAULT` | notes folder the index was built from |
| `RECALL_EXE` | explicit path to the `recall` binary |
| `RECALL_MCP_TIMEOUT_S` | per-call timeout, default 90 (a cold daemon-less query on a big index is slow) |
