#!/usr/bin/env python3
"""
chonkie_chunk — OPTIONAL, opt-in semantic/token-aware chunking for recall ingest.

WHY THIS EXISTS
  recall's dense channel embeds one vector per WHOLE .md/.qmd file with a
  384/768-dim BGE model whose window is 512 tokens (model_max_length=512).
  Measured on the live corpus: 68.8% of documents exceed ~2k chars, so most of
  their body is silently truncated away from the dense channel. Chunking each
  document into <=budget-token pieces makes the whole body embeddable.

ISOLATION CONTRACT
  * recall's Rust core NEVER imports this — it is a Python ingest/ sidecar that
    runs BEFORE the indexer, exactly like ingest/wispr and ingest/mail. Zero new
    Rust deps; the core keeps booting unchanged.
  * Importing THIS module is stdlib-only. The optional `chonkie` dependency is
    imported LAZILY inside its provider function; a missing dep surfaces as a
    provider error and the seam falls through to the always-available stdlib
    recursive splitter (the "floor").
  * DEFAULT OFF. With the flag off, chunk_text() returns exactly one chunk
    holding the whole body — byte-for-byte the input recall embeds today. Opt-in
    is via RECALL_FLAG_CHONKIE=1 (or enabled=True).

PROVIDERS (first working one wins, config-ordered)
  chonkie   — semantic/token-aware chunking via the `chonkie` package (lazy).
  stdlib    — dependency-free recursive-character splitter (the floor). Always
              available; token count approximated as len/4 (BGE ~4 chars/token).

CLI (opt-in mirror; never touches the source tree)
  py chonkie_chunk.py --in <dir-or-file> --out <mirror-dir> [--enable]
      Flag OFF  -> writes each source .md/.qmd through UNCHANGED (1:1, current
                   behavior) so the mirror is provably identical to today.
      Flag ON   -> writes N chunk files per source, each <=budget tokens, with
                   frontmatter linking back to source path + chunk_idx/total.

Env:
  RECALL_FLAG_CHONKIE        "1"/"true"/"yes"/"on" enables chunking (default off)
  RECALL_CHUNK_ORDER         provider order csv (default "chonkie,stdlib")
  RECALL_CHUNK_BUDGET_TOKENS target max tokens/chunk (default 460, < BGE 512)
  RECALL_CHUNK_OVERLAP_TOKENS overlap between chunks in tokens (default 32)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass, field

# ~chars per token for the BGE tokenizer on English prose. Only used to keep the
# dependency-free path honestly under the 512-token window; the chonkie provider
# uses a real tokenizer when installed.
CHARS_PER_TOKEN = 4
DEFAULT_BUDGET_TOKENS = int(os.environ.get("RECALL_CHUNK_BUDGET_TOKENS", "460"))
DEFAULT_OVERLAP_TOKENS = int(os.environ.get("RECALL_CHUNK_OVERLAP_TOKENS", "32"))
DEFAULT_ORDER = os.environ.get("RECALL_CHUNK_ORDER", "chonkie,stdlib")

# Recursive separators, coarsest first (LangChain-style). The final "" forces a
# hard character split so a single unbroken run can never exceed the budget.
_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def flag_on(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def approx_tokens(text: str) -> int:
    """Dependency-free token estimate (chars/4, min 1 for non-empty)."""
    n = len(text)
    if n == 0:
        return 0
    return max(1, (n + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


@dataclass
class Chunk:
    text: str
    idx: int
    total: int
    provider: str
    tokens: int = field(default=0)

    def __post_init__(self):
        if not self.tokens:
            self.tokens = approx_tokens(self.text)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "idx": self.idx,
            "total": self.total,
            "provider": self.provider,
            "tokens": self.tokens,
        }


# --------------------------------------------------------------------------- #
# provider: stdlib recursive splitter (the always-available floor)
# --------------------------------------------------------------------------- #
def _split_recursive(text: str, budget_chars: int, seps: list[str]) -> list[str]:
    """Split text into pieces <= budget_chars, preferring coarse separators.
    Preserves all non-separator content; only whitespace at boundaries is lost."""
    if len(text) <= budget_chars:
        s = text.strip()
        return [s] if s else []
    sep = seps[0] if seps else ""
    rest = seps[1:] if len(seps) > 1 else [""]

    if sep == "":
        # Hard character split — last resort, guarantees termination.
        return [text[i:i + budget_chars] for i in range(0, len(text), budget_chars)]

    parts = text.split(sep)
    out: list[str] = []
    buf = ""
    for i, part in enumerate(parts):
        piece = part + (sep if i < len(parts) - 1 else "")
        if len(piece) > budget_chars:
            # This fragment alone is oversize — flush buffer, recurse finer.
            if buf.strip():
                out.append(buf.strip())
                buf = ""
            out.extend(_split_recursive(piece, budget_chars, rest))
            continue
        if len(buf) + len(piece) > budget_chars:
            if buf.strip():
                out.append(buf.strip())
            buf = piece
        else:
            buf += piece
    if buf.strip():
        out.append(buf.strip())
    return [c for c in out if c]


def stdlib_chunk(text: str, budget_tokens: int, overlap_tokens: int) -> list[str]:
    """Dependency-free token-aware chunker (the floor). Never raises on real text."""
    budget_chars = max(1, budget_tokens * CHARS_PER_TOKEN)
    pieces = _split_recursive(text, budget_chars, _SEPARATORS)
    if overlap_tokens <= 0 or len(pieces) <= 1:
        return pieces
    # Prepend a tail of the previous chunk to each chunk for context continuity.
    overlap_chars = max(1, overlap_tokens * CHARS_PER_TOKEN)
    out = [pieces[0]]
    for prev, cur in zip(pieces, pieces[1:]):
        tail = prev[-overlap_chars:]
        out.append((tail + " " + cur).strip())
    return out


# --------------------------------------------------------------------------- #
# provider: chonkie (optional, lazy import — semantic/token-aware)
# --------------------------------------------------------------------------- #
def chonkie_chunk(text: str, budget_tokens: int, overlap_tokens: int) -> list[str]:
    """Real semantic/token-aware chunking via the `chonkie` package.

    Lazily imported so a missing dependency falls through to the stdlib floor.
    Prefers RecursiveChunker (structure-aware, no model download); this is a
    thin adapter that returns plain-text chunks. Raises on any failure so the
    seam degrades gracefully."""
    from chonkie import RecursiveChunker  # lazy: optional dep

    # 🔴 tokenizer= is MANDATORY. Without it chonkie 1.7.0 counts CHARACTERS while
    # still reporting them as `token_count`, producing ~4x too many chunks. That is
    # what poisoned the embeddings3 table (7.87 chunks/doc vs a correct 3.23).
    # Fixed in vertex_backfill.py on 2026-07-31 but MISSED here, which left the live
    # ingest path still generating bad chunks.
    #
    # 🔴 2026-08-06: that fix was pasted between `chunker =` and its value, leaving
    # `chunker = # comment` — a SyntaxError that made this WHOLE MODULE unimportable.
    # `stdlib_chunk` lives here too, and it is the floor that vertex_backfill falls back
    # to when chonkie errors, so the safety net raised SyntaxError instead of catching.
    # Caught by run_embedding.py's pre-flight chunker gate before any paid call.
    chunker = RecursiveChunker(chunk_size=budget_tokens, tokenizer="gpt2")
    chunks = chunker(text)
    out = []
    for c in chunks:
        t = getattr(c, "text", None)
        if t is None:
            t = str(c)
        t = t.strip()
        if t:
            out.append(t)
    if not out:
        raise RuntimeError("chonkie produced no chunks")
    return out


DEFAULT_REGISTRY = {
    "chonkie": chonkie_chunk,
    "stdlib": stdlib_chunk,
}


def resolve_chunk_chain(order: str | None = None) -> list[str]:
    """Config-ordered provider chain, always terminating at the stdlib floor."""
    raw = (order if order is not None else DEFAULT_ORDER).split(",")
    chain = [p.strip() for p in raw if p.strip() in DEFAULT_REGISTRY]
    if "stdlib" not in chain:
        chain.append("stdlib")  # floor is mandatory
    return chain


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def chunk_text(
    text: str,
    *,
    enabled: bool | None = None,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    order: str | None = None,
    registry: dict | None = None,
    emit=None,
) -> list[Chunk]:
    """Chunk `text` into <=budget_tokens pieces.

    enabled=None  -> read RECALL_FLAG_CHONKIE (default OFF).
    enabled=False -> passthrough: exactly ONE chunk holding the whole body,
                     identical to what recall embeds today (current behavior).
    enabled=True  -> first working provider in the chain wins; a provider that
                     raises or returns nothing falls through to the next, with
                     the stdlib floor guaranteeing a result.
    """
    if enabled is None:
        enabled = flag_on("RECALL_FLAG_CHONKIE", default=False)

    if not enabled:
        body = text if text is not None else ""
        return [Chunk(body, idx=0, total=1, provider="passthrough")]

    reg = registry if registry is not None else DEFAULT_REGISTRY
    chain = resolve_chunk_chain(order)
    for name in chain:
        fn = reg.get(name)
        if fn is None:
            continue
        try:
            pieces = fn(text, budget_tokens, overlap_tokens)
        except Exception as e:  # missing dep / provider fault -> fall through
            if emit:
                emit("warn", f"chunk:{name} failed: {type(e).__name__}: {e}")
            continue
        if not pieces:
            continue
        total = len(pieces)
        return [Chunk(p, idx=i, total=total, provider=name) for i, p in enumerate(pieces)]
    # Unreachable in practice (stdlib floor never returns empty for non-empty text)
    body = text if text is not None else ""
    return [Chunk(body, idx=0, total=1, provider="passthrough")]


# --------------------------------------------------------------------------- #
# CLI — opt-in mirror writer (never mutates the source tree)
# --------------------------------------------------------------------------- #
def _iter_docs(root: str):
    if os.path.isfile(root):
        yield root
        return
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".md", ".qmd")):
                yield os.path.join(dirpath, f)


def _write_chunks(src: str, in_root: str, out_root: str, chunks: list[Chunk]) -> int:
    rel = os.path.relpath(src, in_root) if os.path.isdir(in_root) else os.path.basename(src)
    stem, ext = os.path.splitext(rel)
    written = 0
    for ch in chunks:
        if ch.total == 1:
            out_rel = rel  # passthrough / single chunk keeps the original name
        else:
            out_rel = f"{stem}.chunk{ch.idx:04d}{ext}"
        out_path = os.path.join(out_root, out_rel)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        digest = hashlib.sha256(ch.text.encode("utf-8", "replace")).hexdigest()[:16]
        if ch.total == 1 and ch.provider == "passthrough":
            payload = ch.text  # byte-identical to source
        else:
            payload = (
                "---\n"
                f"source: {rel.replace(os.sep, '/')}\n"
                f"chunk_idx: {ch.idx}\n"
                f"chunk_total: {ch.total}\n"
                f"chunk_provider: {ch.provider}\n"
                f"chunk_tokens: {ch.tokens}\n"
                f"chunk_sha: {digest}\n"
                "tags: [recall-ingest, chunked]\n"
                "---\n\n"
                f"{ch.text}\n"
            )
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(payload)
        written += 1
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="recall opt-in Chonkie chunker (mirror writer)")
    ap.add_argument("--in", dest="in_root", required=True, help="source .md/.qmd file or dir")
    ap.add_argument("--out", dest="out_root", required=True, help="mirror output dir")
    ap.add_argument("--enable", action="store_true", help="force chunking on (else RECALL_FLAG_CHONKIE)")
    ap.add_argument("--budget-tokens", type=int, default=DEFAULT_BUDGET_TOKENS)
    ap.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS)
    ap.add_argument("--order", default=None)
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    enabled = True if args.enable else None
    n_docs = n_chunks = 0
    for src in _iter_docs(args.in_root):
        try:
            with open(src, encoding="utf-8", errors="replace") as fh:
                body = fh.read()
        except OSError:
            continue
        chunks = chunk_text(
            body, enabled=enabled,
            budget_tokens=args.budget_tokens, overlap_tokens=args.overlap_tokens,
            order=args.order, emit=lambda k, m: print(f"[{k}] {m}"),
        )
        n_chunks += _write_chunks(src, args.in_root, args.out_root, chunks)
        n_docs += 1
    state = "ON" if (args.enable or flag_on("RECALL_FLAG_CHONKIE")) else "OFF (passthrough)"
    print(f"chunking: {state}")
    print(f"docs: {n_docs}  chunk-files: {n_chunks}  out: {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
