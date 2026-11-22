#!/usr/bin/env python3
"""
marker_export — OPTIONAL, opt-in PDF/docx -> clean-markdown front door for recall.

WHY THIS EXISTS
  recall only ingests .md/.qmd (extension gate in src/store.rs:421, enumerate
  glob in src/store.rs:1035). PDFs and Word docs are INVISIBLE to it. This is a
  new exporter that turns those documents into .md the indexer can already see —
  the exact "exporter -> .md -> vault -> recall index" pattern used by
  ingest/wispr, ingest/mail, and ingest/screenshots. ZERO recall-core changes.

ISOLATION CONTRACT
  * recall's Rust core NEVER imports this — it is an ingest/ sidecar run BEFORE
    indexing. Zero new Rust deps; the core keeps booting unchanged.
  * Importing THIS module is stdlib-only. The optional `marker`/`marker-pdf`
    dependency is imported LAZILY inside its provider function. A missing dep
    surfaces as a provider error -> the seam falls through to the next provider.
  * DEFAULT OFF. With the flag off, export_document() returns None (the document
    stays invisible to recall — today's behavior). Opt-in via RECALL_FLAG_MARKER=1
    (or enabled=True).

PROVIDERS (first working one wins, config-ordered)
  marker  — high-quality PDF + docx -> markdown via the `marker` package (lazy).
            Needed for PDF; there is no dependency-free PDF text path.
  stdlib  — dependency-free .docx -> text/markdown (a .docx is a zip of
            word/document.xml; paragraphs are <w:p>, runs are <w:t>). The floor
            for Word docs. Does NOT handle .pdf (returns nothing -> honest gap).

NOTE ON PARSER CHOICE
  A standing decision in auto-memory ("recall parser = Docling, chunkr A/B only")
  competes with Marker. That decision was never implemented in this repo. This
  module makes the parser a swappable PROVIDER (order via RECALL_MARKER_ORDER),
  so Marker vs Docling stays a config flip, not a code rewrite. Gate the final
  choice with the owner; nothing here is on by default.

CLI (opt-in; writes .md into the vault, idempotent)
  py marker_export.py --in <file-or-dir> --out <vault-dir> [--enable]

Env:
  RECALL_FLAG_MARKER   "1"/"true"/"yes"/"on" enables export (default off)
  RECALL_MARKER_ORDER  provider order csv (default "marker,stdlib")
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

DEFAULT_ORDER = os.environ.get("RECALL_MARKER_ORDER", "marker,stdlib")
SUPPORTED_EXT = (".pdf", ".docx")
_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def flag_on(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- #
# provider: marker (optional, lazy import — best-quality PDF + docx)
# --------------------------------------------------------------------------- #
def marker_convert(src_path: str) -> str:
    """Convert a PDF/docx to markdown via the `marker` package.

    Lazily imported so a missing dependency falls through. Marker's API has
    shifted across versions; this adapter tries the modern converter first and
    the legacy function second, then raises so the seam degrades gracefully."""
    # Modern API (marker-pdf >= 1.x): PdfConverter + model dict.
    try:
        from marker.converters.pdf import PdfConverter          # lazy: optional dep
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(src_path)
        md, _meta, _images = text_from_rendered(rendered)
        md = (md or "").strip()
        if md:
            return md
        raise RuntimeError("marker produced empty markdown")
    except ImportError:
        pass  # fall through to legacy API
    # Legacy API (marker-pdf < 1.x): convert_single_pdf.
    from marker.convert import convert_single_pdf              # lazy: optional dep
    from marker.models import load_all_models
    full_text, _images, _meta = convert_single_pdf(src_path, load_all_models())
    md = (full_text or "").strip()
    if not md:
        raise RuntimeError("marker produced empty markdown")
    return md


# --------------------------------------------------------------------------- #
# provider: stdlib docx (dependency-free floor for Word docs)
# --------------------------------------------------------------------------- #
def _safe_fromstring(xml: bytes) -> "ET.Element":
    """Parse XML with XXE / billion-laughs blocked using stdlib only.

    We cannot pull defusedxml (zero-new-deps mandate), so we reject any DTD or
    entity declaration BEFORE parsing. Both attack classes (external-entity XXE
    and internal-entity billion-laughs) require a <!DOCTYPE ...> with <!ENTITY>
    defs; legitimate OOXML word/document.xml never contains either. A small
    leading window scan is enough — expat processes the DOCTYPE prologue first."""
    head = xml[:4096].lstrip()
    lowered = head.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise RuntimeError("docx XML contains a DTD/entity declaration; refusing to parse")
    return ET.fromstring(xml)


def stdlib_convert(src_path: str) -> str:
    """Dependency-free .docx -> markdown. Raises for anything else so the seam
    reports an honest gap (no stdlib PDF text extraction exists)."""
    if not src_path.lower().endswith(".docx"):
        raise RuntimeError("stdlib provider handles .docx only")
    with zipfile.ZipFile(src_path) as z:
        with z.open("word/document.xml") as fh:
            xml = fh.read()
    root = _safe_fromstring(xml)
    lines: list[str] = []
    body = root.find(f"{_W_NS}body")
    if body is None:
        body = root
    for para in body.iter(f"{_W_NS}p"):
        texts = [t.text for t in para.iter(f"{_W_NS}t") if t.text]
        line = "".join(texts).strip()
        # crude heading detection: a pStyle whose name starts with "Heading"
        style = para.find(f"{_W_NS}pPr/{_W_NS}pStyle")
        if style is not None:
            val = style.get(f"{_W_NS}val", "")
            m = re.match(r"[Hh]eading(\d)", val)
            if m and line:
                level = min(6, int(m.group(1)))
                line = ("#" * level) + " " + line
        lines.append(line)
    md = "\n\n".join(l for l in lines).strip()
    md = re.sub(r"\n{3,}", "\n\n", md)
    if not md:
        raise RuntimeError("stdlib docx produced empty text")
    return md


DEFAULT_REGISTRY = {
    "marker": marker_convert,
    "stdlib": stdlib_convert,
}


def resolve_marker_chain(order: str | None = None) -> list[str]:
    raw = (order if order is not None else DEFAULT_ORDER).split(",")
    return [p.strip() for p in raw if p.strip() in DEFAULT_REGISTRY]


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def export_document(
    src_path: str,
    *,
    enabled: bool | None = None,
    order: str | None = None,
    registry: dict | None = None,
    emit=None,
) -> str | None:
    """Convert a PDF/docx to markdown.

    enabled=None  -> read RECALL_FLAG_MARKER (default OFF).
    enabled=False -> returns None: the document stays invisible to recall, i.e.
                     today's behavior (recall never ingested PDF/docx).
    enabled=True  -> first working provider wins; a provider that raises falls
                     through to the next. Returns None if every provider fails
                     (e.g. a PDF with marker not installed) — an honest gap, not
                     a crash.
    """
    if enabled is None:
        enabled = flag_on("RECALL_FLAG_MARKER", default=False)
    if not enabled:
        return None

    ext = os.path.splitext(src_path)[1].lower()
    if ext not in SUPPORTED_EXT:
        if emit:
            emit("warn", f"marker: unsupported extension {ext}")
        return None

    reg = registry if registry is not None else DEFAULT_REGISTRY
    for name in resolve_marker_chain(order):
        fn = reg.get(name)
        if fn is None:
            continue
        try:
            md = fn(src_path)
        except Exception as e:
            if emit:
                emit("warn", f"marker:{name} failed: {type(e).__name__}: {e}")
            continue
        if md and md.strip():
            return md.strip()
    return None


def _front_matter(src_path: str, md: str) -> str:
    digest = hashlib.sha256(md.encode("utf-8", "replace")).hexdigest()[:16]
    base = os.path.basename(src_path)
    return (
        "---\n"
        f"source_document: {base}\n"
        f"source_ext: {os.path.splitext(src_path)[1].lower().lstrip('.')}\n"
        f"content_sha: {digest}\n"
        "tags: [recall-ingest, document-import]\n"
        "---\n\n"
        f"<!-- marker-export sha={digest} -->\n"
        f"# {os.path.splitext(base)[0]}\n\n"
        f"{md}\n"
    )


def _iter_docs(root: str):
    if os.path.isfile(root):
        yield root
        return
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.lower().endswith(SUPPORTED_EXT):
                yield os.path.join(dirpath, f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="recall opt-in Marker document front door")
    ap.add_argument("--in", dest="in_root", required=True, help="PDF/docx file or dir")
    ap.add_argument("--out", dest="out_root", required=True, help="vault dir for .md output")
    ap.add_argument("--enable", action="store_true", help="force export on (else RECALL_FLAG_MARKER)")
    ap.add_argument("--order", default=None)
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    enabled = True if args.enable else None
    if enabled is None and not flag_on("RECALL_FLAG_MARKER"):
        print("marker: OFF (RECALL_FLAG_MARKER unset) — no documents exported")
        return 0

    os.makedirs(args.out_root, exist_ok=True)
    exported = skipped = failed = 0
    for src in _iter_docs(args.in_root):
        md = export_document(src, enabled=enabled,
                             order=args.order, emit=lambda k, m: print(f"[{k}] {m}"))
        if not md:
            failed += 1
            print(f"[skip] no provider handled: {os.path.basename(src)}")
            continue
        stem = os.path.splitext(os.path.basename(src))[0]
        out_path = os.path.join(args.out_root, f"{stem}.md")
        payload = _front_matter(src, md)
        marker_tag = f"sha={hashlib.sha256(md.encode('utf-8','replace')).hexdigest()[:16]}"
        if os.path.exists(out_path):
            with open(out_path, encoding="utf-8", errors="replace") as fh:
                if marker_tag in fh.read(4096):
                    skipped += 1
                    continue
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(payload)
        exported += 1
    print(f"marker: ON  exported: {exported}  unchanged: {skipped}  no-provider: {failed}")
    print(f"output: {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
