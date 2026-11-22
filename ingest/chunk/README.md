# ingest/chunk — opt-in semantic/token-aware chunking (Chonkie)

Optional recall ingest sidecar. **Default OFF** — recall behaves exactly as
today unless you opt in. Zero new Rust deps; the core never imports this.

## Why
recall embeds one dense vector per WHOLE `.md`/`.qmd` file with a BGE model whose
window is **512 tokens**. On the live corpus **68.8%** of docs exceed ~2k chars,
so most of their body is truncated out of the dense channel. Chunking each doc
into ≤budget-token pieces makes the whole body embeddable.

## Providers (first working one wins)
- `chonkie` — real semantic/token-aware chunking (lazy import; `pip install chonkie`).
- `stdlib` — dependency-free recursive-character splitter (the always-available floor).

A missing/failing `chonkie` falls through to the stdlib floor — never crashes.

## Use
```bash
# opt-in mirror: writes chunk files into a SEPARATE tree (never touches source)
py chonkie_chunk.py --in <dir-or-file> --out <mirror-dir> --enable
# flag off (default): mirror is byte-identical to source (current behavior)
py chonkie_chunk.py --in <dir-or-file> --out <mirror-dir>
```
Env: `RECALL_FLAG_CHONKIE=1` (enable), `RECALL_CHUNK_ORDER`, `RECALL_CHUNK_BUDGET_TOKENS`
(default 460), `RECALL_CHUNK_OVERLAP_TOKENS` (default 32).

## Test
```bash
py test_chunk_seam.py      # stdlib-only, 14 tests, no chonkie needed
```

## Status
- stdlib floor: **E2E-working** (proven by the self-test).
- chonkie provider: **stubbed until `pip install chonkie`** (lazy adapter, untested live).
