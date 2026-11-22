"""Real MCP client handshake against recall_mcp.py over stdio.

    python mcp/smoke_handshake.py [--query "text"]

Spawns the server as a subprocess exactly as an MCP host would, performs initialize,
lists the tools, and calls each one. Exits non-zero on any failure. Uses whatever
RECALL_EXE / RECALL_VAULT / HOME the caller has set.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

SERVER = pathlib.Path(__file__).resolve().parent / "recall_mcp.py"
EXPECTED = {"recall_search", "recall_search_semantic", "recall_recent", "recall_health"}


async def main(query: str, quiet: bool = False, server: pathlib.Path = SERVER, results: pathlib.Path | None = None) -> int:
    # recall's replies carry emoji; a redirected Windows stdout is cp1252 and would crash on them
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    transport = StdioTransport(command=sys.executable, args=[str(server)], env=dict(os.environ))
    async with Client(transport) as client:  # runs the MCP initialize handshake
        tools = {t.name for t in await client.list_tools()}
        print("tools:", sorted(tools))
        missing = EXPECTED - tools
        if missing:
            print("FAIL missing tools:", sorted(missing))
            return 1
        if quiet:  # handshake + tool list only (an empty index makes search fail loudly, by design)
            return 0
        failed = 0
        full = {}
        for name, args in [
            ("recall_health", {}),
            ("recall_search", {"query": query, "top": 3}),
            ("recall_search_semantic", {"query": query, "top": 3}),
            ("recall_recent", {"count": 3}),
        ]:
            res = await client.call_tool(name, args, raise_on_error=False)
            text = "".join(getattr(c, "text", "") for c in res.content)
            status = "ERROR" if res.is_error else "ok"
            if res.is_error:
                failed += 1
            full[name] = {"error": bool(res.is_error), "text": text}
            print(f"--- {name}: {status}\n{text[:600]}")
        if results:  # the whole replies, for a caller that checks content (the console copy is cut)
            results.write_text(json.dumps(full), encoding="utf-8")
        return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="handshake and tool list only; no tool calls")
    ap.add_argument("--query", default="how do I look after plants that get leaf disease")
    ap.add_argument("--server", type=pathlib.Path, default=SERVER, help="server script (default: the repo copy)")
    ap.add_argument("--results", type=pathlib.Path, help="write every tool's full reply here as JSON")
    a = ap.parse_args()
    sys.exit(asyncio.run(main(a.query, a.quiet, a.server, a.results)))
