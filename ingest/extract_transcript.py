"""extract_transcript.py — streaming Claude Code session JSONL extractor.

Input : one Claude Code session .jsonl path (iterate line-by-line, never read()).
Output: ordered turn-record JSONL with schema
        {turn_idx, ts, role, kind, tool_name, text}

kind values:
  user_text      — user prose
  assistant_text — assistant prose
  tool_use       — assistant invoked a tool (text = <=200-char input summary)
  tool_result    — tool output (text = first 500 chars + byte note if truncated)
  skipped        — progress/system noise kept for counting but not emitted
"""

import sys

# Windows consoles default to cp1252 — force UTF-8 to avoid UnicodeEncodeError
# on arrows, ellipses, or any non-ASCII in progress output.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import argparse  # noqa: E402
import json  # noqa: E402

MAX_TOOL_RESULT_DEFAULT = 500


def _text_from_content(content) -> str:
    """Extract plain text from a message content field (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", ""))
            # tool_use and tool_result blocks handled by caller
        return " ".join(p for p in parts if p)
    return ""


def _summarise_input(inp, max_chars: int = 200) -> str:
    """Produce a short summary of a tool_use input dict."""
    if not isinstance(inp, dict):
        raw = str(inp)
    else:
        raw = json.dumps(inp, ensure_ascii=False)
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars] + f"... [{len(raw.encode())} bytes total]"


def _cap_tool_result(text: str, max_chars: int) -> str:
    """Cap tool result text, appending a byte note when truncated."""
    if len(text) <= max_chars:
        return text
    total_bytes = len(text.encode("utf-8", errors="replace"))
    return text[:max_chars] + f" ... [{total_bytes} bytes total]"


def _extract_tool_result_text(content, max_chars: int) -> tuple[str, str | None]:
    """Return (text, tool_use_id) from a tool_result content block list."""
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                inner = block.get("content", "")
                if isinstance(inner, list):
                    inner = _text_from_content(inner)
                text = _cap_tool_result(str(inner), max_chars)
                return text, block.get("tool_use_id")
    # Fallback: whole content as string
    raw = _text_from_content(content) if isinstance(content, list) else str(content)
    return _cap_tool_result(raw, max_chars), None


def iter_turns(file_path: str, max_tool_result: int = MAX_TOOL_RESULT_DEFAULT):
    """Yield turn dicts by iterating the file handle line-by-line.

    Never calls file.read() — memory-safe for multi-MB lines.
    Counts bad JSON lines and yields a summary record at the end.
    """
    bad_lines = 0
    turn_idx = 0

    with open(file_path, encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue

            if not isinstance(obj, dict):
                bad_lines += 1
                continue

            record_type = obj.get("type", "")
            ts = obj.get("timestamp", "")

            # Skip non-conversation record types
            if record_type in ("last-prompt", "permission-mode", "file-history-snapshot"):
                continue

            # Attachment lines (hooks, system noise) — skip
            if record_type == "attachment":
                continue

            message = obj.get("message")
            if not isinstance(message, dict):
                continue

            role = message.get("role", "")
            content = message.get("content", "")

            # --- user messages ---
            if role == "user":
                # Skip local-command meta lines
                if isinstance(content, str) and content.startswith("<local-command"):
                    continue
                if isinstance(content, str) and content.startswith("<command-name>"):
                    continue
                text = _text_from_content(content) if isinstance(content, list) else str(content)
                if not text.strip():
                    continue
                yield {
                    "turn_idx": turn_idx,
                    "ts": ts,
                    "role": "user",
                    "kind": "user_text",
                    "tool_name": None,
                    "text": text,
                }
                turn_idx += 1

            # --- assistant messages ---
            elif role == "assistant":
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type", "")
                        if btype == "text":
                            text = block.get("text", "")
                            if text.strip():
                                yield {
                                    "turn_idx": turn_idx,
                                    "ts": ts,
                                    "role": "assistant",
                                    "kind": "assistant_text",
                                    "tool_name": None,
                                    "text": text,
                                }
                                turn_idx += 1
                        elif btype == "tool_use":
                            tool_name = block.get("name") or block.get("tool", "unknown")
                            summary = _summarise_input(block.get("input", {}))
                            yield {
                                "turn_idx": turn_idx,
                                "ts": ts,
                                "role": "assistant",
                                "kind": "tool_use",
                                "tool_name": tool_name,
                                "text": summary,
                            }
                            turn_idx += 1
                else:
                    text = str(content)
                    if text.strip():
                        yield {
                            "turn_idx": turn_idx,
                            "ts": ts,
                            "role": "assistant",
                            "kind": "assistant_text",
                            "tool_name": None,
                            "text": text,
                        }
                        turn_idx += 1

            # --- tool result messages ---
            elif role == "tool":
                # tool_name may be on the outer obj (Claude Code puts it there)
                tool_name = obj.get("toolName") or obj.get("tool_name")
                text, _ = _extract_tool_result_text(content, max_tool_result)
                # If tool_name not on outer obj, try to find it in content block
                if not tool_name and isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            tool_name = block.get("tool_name") or block.get("name")
                            break
                yield {
                    "turn_idx": turn_idx,
                    "ts": ts,
                    "role": "tool",
                    "kind": "tool_result",
                    "tool_name": tool_name,
                    "text": text,
                }
                turn_idx += 1

    if bad_lines:
        print(f"[extract_transcript] {bad_lines} bad JSON line(s) skipped", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Stream-extract Claude Code session JSONL into turn records."
    )
    parser.add_argument("in_path", help="Path to input .jsonl session file")
    parser.add_argument("--out", default="-", help="Output path (default: stdout)")
    parser.add_argument(
        "--max-tool-result",
        type=int,
        default=MAX_TOOL_RESULT_DEFAULT,
        help="Max chars for tool_result text before capping (default: 500)",
    )
    args = parser.parse_args()

    if args.out == "-":
        out_fh = sys.stdout
        close_out = False
    else:
        out_fh = open(args.out, "w", encoding="utf-8")
        close_out = True

    try:
        count = 0
        for turn in iter_turns(args.in_path, max_tool_result=args.max_tool_result):
            out_fh.write(json.dumps(turn, ensure_ascii=False) + "\n")
            count += 1
        print(f"[extract_transcript] wrote {count} turn(s)", file=sys.stderr)
    finally:
        if close_out:
            out_fh.close()


if __name__ == "__main__":
    main()
