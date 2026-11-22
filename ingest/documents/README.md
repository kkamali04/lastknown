# ingest/documents — opt-in PDF/docx → markdown front door (Marker)

Optional recall ingest sidecar. **Default OFF** — recall behaves exactly as
today unless you opt in. Zero new Rust deps; the core never imports this. Slots
into the existing `exporter → .md → vault → recall index` pattern (same as
`ingest/wispr`, `ingest/mail`, `ingest/screenshots`) — **zero recall-core changes**.

## Why
recall only ingests `.md`/`.qmd` (extension gate `src/store.rs:421`, enumerate
glob `src/store.rs:1035`). PDFs and Word docs are invisible to it. This exporter
turns them into `.md` the indexer already sees.

## Providers (first working one wins)
- `marker` — high-quality PDF + docx → markdown (lazy import; `pip install marker-pdf`).
  Required for PDF (there is no dependency-free PDF text path).
- `stdlib` — dependency-free `.docx` → markdown (unzip `word/document.xml`, read
  `<w:p>`/`<w:t>`). The floor for Word docs. DTD/entity XML is **refused** (XXE /
  billion-laughs block, stdlib-only).

A missing/failing `marker` falls through; a PDF with marker absent returns an
honest `None` (no output), never a crash.

## Parser choice (gate with the owner)
A standing auto-memory decision says "recall parser = Docling, chunkr A/B only" —
never implemented in this repo. Parser here is a swappable **provider**
(`RECALL_MARKER_ORDER`), so Marker vs Docling stays a config flip, not a rewrite.

## Use
```bash
py marker_export.py --in <file-or-dir> --out <vault-dir> --enable
```
Env: `RECALL_FLAG_MARKER=1` (enable), `RECALL_MARKER_ORDER` (default `marker,stdlib`).
Output is idempotent (skips files whose `marker-export sha=` tag is unchanged).

## Test
```bash
py test_marker_seam.py     # stdlib-only, 10 tests, builds a real .docx, no marker needed
```

## Status
- stdlib `.docx` floor: **E2E-working** (proven by the self-test on a built .docx).
- marker PDF/docx provider: **stubbed until `pip install marker-pdf`** (lazy adapter, untested live).
