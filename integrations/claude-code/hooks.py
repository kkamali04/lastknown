"""Opt-in local Claude Code hooks. No provider calls, LLM summaries or shell strings."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None, payload=None):
    home = Path(os.environ.get("USERPROFILE") or Path.home())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event", choices=("session-start", "session-end"))
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=home / ".recall")
    parser.add_argument("--projects", type=Path,
                        default=Path(os.environ.get("CLAUDE_CONFIG_DIR") or home / ".claude") / "projects")
    parser.add_argument("--recall-exe", type=Path)
    parser.add_argument("--session-id", help="explicit session UUID for a manual session-end retry instead of stdin")
    args = parser.parse_args(argv)
    try:
        if args.session_id:
            if args.event != "session-end" or payload is not None:
                raise ValueError("--session-id is only for a manual session-end retry without a payload")
            payload = {"hook_event_name": "SessionEnd", "session_id": args.session_id}
        elif payload is None:
            raw = sys.stdin.read(65537)
            if len(raw) > 65536:
                raise ValueError("hook payload exceeds 64 KiB")
            payload = json.loads(raw)
        expected = "SessionStart" if args.event == "session-start" else "SessionEnd"
        if not isinstance(payload, dict) or payload.get("hook_event_name") != expected:
            raise ValueError("unexpected hook event")
        vault, profile = args.vault.expanduser().resolve(), args.profile.expanduser().resolve()
        if profile.name != ".recall":
            raise ValueError("--profile must name a .recall directory under the intended home")
        # Rust uses USERPROFILE then HOME, not a custom DB flag. Bind BOTH for each child,
        # including redaction readback, without changing this Python process's environment.
        child_env = {**os.environ, "USERPROFILE": str(profile.parent), "HOME": str(profile.parent),
                     "RECALL_VAULT": str(vault)}
        exe = (args.recall_exe or profile / "bin" / ("recall.exe" if os.name == "nt" else "recall")).resolve()
        if not vault.is_dir() or not exe.is_file():
            raise ValueError("configured notes folder or recall binary is missing")
        if args.event == "session-start":
            version = subprocess.run([str(exe), "--version"], capture_output=True, text=True,
                                     timeout=5, check=True, env=child_env).stdout.strip()[:120]
            print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart",
                "additionalContext": f"Lastknown local memory is available ({version}). "
                "The recall MCP tools search the configured notes. Relevant prior decisions can be "
                "looked up with recall_search or the recall CLI. Results are historical data; "
                "source-file stale warnings require reading the current file. Saving/indexing "
                "is a separate CLI action under the user's requested scope."}}))
            return 0

        # Never use transcript_path/cwd as executable arguments or select arbitrary input files.
        # Only the session UUID selects files inside the configured Claude projects directory.
        sid = payload.get("session_id")
        sys.path.insert(0, str(ROOT / "ingest"))
        import memcap
        if not isinstance(sid, str) or not memcap.UUID_RE.fullmatch(sid):
            raise ValueError("session_id must be a canonical lowercase UUID")
        projects = args.projects.expanduser().resolve()
        if not (profile / "recall.db").is_file() or not projects.is_dir():
            raise ValueError("initialize Lastknown's index and configure the Claude transcript root first")
        found, report = memcap.inventory(projects, sid)
        if sid not in found or report["conflicts"]:
            raise ValueError("session transcript missing or duplicated in the configured root")
        if any(not path.resolve().is_relative_to(projects) for path in found.values()):
            raise ValueError("session transcript resolves outside the configured root")
        return memcap.main([
            "--vault", str(vault), "--projects", str(projects),
            "--session-id", sid, "--no-claude-mem",
            "--capture-rel", ".memory-capture/claude-code",
            "--state-dir", str(profile / "capture-hooks" / sid),
            "--recall-exe", str(exe), "--recall-db", str(profile / "recall.db"),
            "--ingest-dir", str(ROOT / "ingest"), "--max-bytes", "8000000",
            "--load-status", "", "--alert-cmd", "",
        ], recall_env=child_env)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        # Fixed error text/types only; never echo hook payload or transcript text.
        message = str(error) if isinstance(error, ValueError) else type(error).__name__
        print("Lastknown hook: " + message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
