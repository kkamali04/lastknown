# recall MCP server

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

## Setup

```bash
python -m pip install -r mcp/requirements.txt      # fastmcp
python mcp/smoke_handshake.py                       # real MCP client handshake, then calls health, search, semantic search and recent
```

The server finds the binary via `RECALL_EXE`, then `~/.recall/bin/recall` (where the installer puts
it), then `target/release/`, then `PATH`. Pass `RECALL_VAULT` so it reads the right notes folder.
Use absolute paths below; replace `/path/to/recall` with your checkout.

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
