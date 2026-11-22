"""scrub_turns.py — Transcript-grade secret/PII scrubber for turn-record JSONL.

Input schema (from P1 extractor):
    {turn_idx, ts, role, kind, tool_name, text}

Applies:
  - 13 PII_PATTERNS ported from conversation-memory-ingest.mjs
  - 7 ADVERSARIAL_CONTENT_PATTERNS ported from vault-lance-indexer.py
  - NEW entropy-based patterns: KEY=value, connection strings, JWT eyJ,
    base64url/hex ≥32 chars with Shannon entropy >4.0
  - Tool-aware hard-drop: tool_name OR tool input text matching secret
    surface keywords → text replaced with <DROPPED:tool-secret-surface>

Replacement convention: <REDACTED:kind:sha8> (mirrors .mjs, not [REDACTED]).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Sequence

# ---------------------------------------------------------------------------
# Stdout safe-encode (python-cli-windows-cp1252-stdout-crash skill)
# ---------------------------------------------------------------------------
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# PII_PATTERNS — ported from conversation-memory-ingest.mjs lines 82-94
# ---------------------------------------------------------------------------
PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email",   re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("phone",   re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("ssn",     re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("ipv4",    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("sk-key",  re.compile(r"\bsk-(?:or-|ant-)?[A-Za-z0-9_\-]{20,}\b")),
    ("ghp",     re.compile(r"\bghp_[A-Za-z0-9]{30,}\b")),
    ("sbp",     re.compile(r"\bsbp_[A-Za-z0-9]{30,}\b")),
    ("fcr",     re.compile(r"\bfcr_[A-Za-z0-9]{20,}\b")),
    ("tvly",    re.compile(r"\btvly-[A-Za-z0-9_\-]{20,}\b")),
    ("apify",   re.compile(r"\bapify_api_[A-Za-z0-9]{30,}\b")),
    ("discord", re.compile(r"\b[MN][A-Za-z0-9_\-]{23,}\.[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]{20,}\b")),
    ("aws",     re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
]

# ---------------------------------------------------------------------------
# ADVERSARIAL_CONTENT_PATTERNS — ported from vault-lance-indexer.py lines 111-137
# ---------------------------------------------------------------------------
ADVERSARIAL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("api_key_assignment", re.compile(
        r'(?:api_key|secret_key|access_token|auth_token)\s*[=:]\s*["\']([A-Za-z0-9_\-]{20,})["\']',
        re.IGNORECASE,
    )),
    ("bearer_token_value", re.compile(
        r'Bearer\s+[A-Za-z0-9+/=]{20,}',
        re.IGNORECASE,
    )),
    ("sk_key",         re.compile(r'sk-[A-Za-z0-9]{20,}')),
    ("aws_access_key", re.compile(r'AKIA[A-Z0-9]{16}')),
    ("pem_private_key",re.compile(r'-----BEGIN\s+(?:RSA\s+)?PRIVATE KEY-----')),
    ("password_value", re.compile(
        r'password\s*[=:]\s*["\'][^"\']{8,}["\']',
        re.IGNORECASE,
    )),
    ("jwt_token", re.compile(
        r'eyJ[A-Za-z0-9+/=]{10,}\.[A-Za-z0-9+/=]{10,}\.[A-Za-z0-9+/=_\-]{10,}',
    )),
]

# ---------------------------------------------------------------------------
# NEW patterns
# ---------------------------------------------------------------------------
# KEY=value where value ≥ 16 chars (env-pair style)
_ENV_KEY_VALUE = re.compile(
    r'\b[A-Z][A-Z0-9_]{2,}\s*=\s*([A-Za-z0-9+/=_\-]{16,})\b'
)
# Connection strings: scheme://user:pass@host
_CONN_STRING = re.compile(
    r'[a-zA-Z][a-zA-Z0-9+\-.]{1,20}://[^@\s]{1,80}:[^@\s]{1,80}@[^\s"\']{1,120}'
)
# PEM block (full header — belt-and-suspenders over ADVERSARIAL set)
_PEM_BLOCK = re.compile(r'-----BEGIN [A-Z ]+-----[\s\S]*?-----END [A-Z ]+-----')
# eyJ JWT header (standalone, not already caught by adversarial full-JWT pattern)
_JWT_HEADER = re.compile(r'\beyJ[A-Za-z0-9_\-]{10,}')

NEW_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("env-pair",    _ENV_KEY_VALUE),
    ("conn-string", _CONN_STRING),
    ("pem-block",   _PEM_BLOCK),
    ("jwt-header",  _JWT_HEADER),
]

# ---------------------------------------------------------------------------
# Entropy-based scanner for base64url / hex runs ≥ 32 chars, entropy > 4.0
# ---------------------------------------------------------------------------
_HIGH_ENTROPY_CANDIDATE = re.compile(r'[A-Za-z0-9+/=_\-]{32,}')
_HEX_CANDIDATE = re.compile(r'\b[0-9a-fA-F]{32,}\b')


def _shannon(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


def _redact_high_entropy(text: str, hits: list[str], keep=None) -> str:
    """Replace high-entropy opaque tokens with REDACTED labels.

    - base64url runs ≥32 chars: Shannon entropy >4.0
    - hex runs ≥32 chars: Shannon entropy >3.5 (hex alphabet caps at ~4.0;
      a 40-char all-unique hex token scores ~3.97 which is indistinguishable
      from a real secret token at that length)
    """
    def _replace_b64(m: re.Match) -> str:
        tok = m.group(0)
        if keep is not None and keep(tok):
            return tok
        if _shannon(tok) > 4.0:
            label = _make_label("entropy-b64url", tok)
            hits.append(label)
            return label
        return tok

    def _replace_hex(m: re.Match) -> str:
        tok = m.group(0)
        if keep is not None and keep(tok):
            return tok
        if _shannon(tok) > 3.5:
            label = _make_label("entropy-hex", tok)
            hits.append(label)
            return label
        return tok

    text = _HIGH_ENTROPY_CANDIDATE.sub(_replace_b64, text)
    text = _HEX_CANDIDATE.sub(_replace_hex, text)
    return text


# ---------------------------------------------------------------------------
# Tool-aware hard-drop
# ---------------------------------------------------------------------------
_TOOL_DROP_RE = re.compile(
    r'crypt|secrets?|credential|keyring|vault|api[-_]?key|password|token|login',
    re.IGNORECASE,
)
_DROPPED_PLACEHOLDER = "<DROPPED:tool-secret-surface>"


def _should_drop(tool_name: str | None, text: str) -> bool:
    if tool_name and _TOOL_DROP_RE.search(tool_name):
        return True
    # Also check first 300 chars of tool-use input (tool_use kind only)
    if text and _TOOL_DROP_RE.search(text[:300]):
        return True
    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sha8(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:8]


def _make_label(kind: str, value: str) -> str:
    return f"<REDACTED:{kind}:{_sha8(value)}>"


def _apply_pattern_list(
    text: str,
    patterns: list[tuple[str, re.Pattern]],
    hits: list[str],
) -> str:
    for kind, pat in patterns:
        def _repl(m: re.Match, _kind: str = kind) -> str:
            label = _make_label(_kind, m.group(0))
            hits.append(label)
            return label
        text = pat.sub(_repl, text)
    return text


def scrub_text(text: str, tool_name: str | None = None, kind: str | None = None,
               keep=None) -> tuple[str, int]:
    """Return (scrubbed_text, redacted_count).

    If tool_name or tool input matches the secret surface, returns the hard-drop placeholder.
    `keep(token) -> bool` exempts a token from the ENTROPY stage only (it runs last, after
    every credential and PII pattern has already been applied), e.g. a caller that knows a
    long slash-separated run is a file path. It cannot exempt anything from the named rules.
    """
    if not isinstance(text, str):
        return text, 0

    # Hard-drop check (tool-aware)
    if kind == "tool_use" and _should_drop(tool_name, text):
        return _DROPPED_PLACEHOLDER, 1
    if tool_name and _TOOL_DROP_RE.search(tool_name):
        return _DROPPED_PLACEHOLDER, 1

    hits: list[str] = []
    # NEW patterns first: conn-string must run before PII so the email
    # heuristic inside a postgres:// URL does not pre-mangle the match.
    text = _apply_pattern_list(text, NEW_PATTERNS, hits)
    text = _apply_pattern_list(text, ADVERSARIAL_PATTERNS, hits)
    text = _apply_pattern_list(text, PII_PATTERNS, hits)
    text = _redact_high_entropy(text, hits, keep)
    return text, len(hits)


# ---------------------------------------------------------------------------
# Main streaming processor
# ---------------------------------------------------------------------------
def process(in_path: Path, out_path: Path) -> None:
    total_redacted = 0
    total_dropped = 0
    total_turns = 0

    with (
        in_path.open("r", encoding="utf-8", errors="replace") as fin,
        out_path.open("w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            tool_name = record.get("tool_name")
            kind = record.get("kind")
            text = record.get("text", "")

            scrubbed, n = scrub_text(text, tool_name=tool_name, kind=kind)
            record["text"] = scrubbed

            if scrubbed == _DROPPED_PLACEHOLDER:
                total_dropped += 1
            else:
                total_redacted += n

            total_turns += 1
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"scrub_turns: {total_turns} turns, {total_redacted} redacted, "
        f"{total_dropped} dropped",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scrub PII/secrets from turn-record JSONL")
    parser.add_argument("in_path", type=Path, help="Input turn-record JSONL")
    parser.add_argument("--out", required=True, type=Path, help="Output scrubbed JSONL")
    args = parser.parse_args(argv)

    if not args.in_path.exists():
        print(f"ERROR: input not found: {args.in_path}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    process(args.in_path, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
