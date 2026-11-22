#!/usr/bin/env python3
"""
test_marker_seam — stdlib-only self-test for recall's opt-in Marker front door.

No marker package needed: PDF-quality parsing (marker) is faked/absent via the
injectable registry, and the dependency-free .docx floor does real extraction on
a .docx this test builds in-memory. Proves:
  1. flag OFF  -> export returns None (PDF/docx stay invisible — behavior today)
  2. docx floor-> flag ON extracts real text + headings from a built .docx
  3. fallback  -> marker missing falls through to the stdlib docx floor
  4. honest gap-> a PDF with no marker returns None (not a crash)
  5. security  -> a docx whose XML declares a DTD/entity is refused (XXE block)
  6. zero-dep  -> importing the module pulls no third-party module
  7. CLI       -> flag ON writes .md with frontmatter; re-run is idempotent

Run:  py test_marker_seam.py
"""
import os
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import marker_export as me

BOOT_PULLED_MARKER = "marker" in sys.modules

_DOCX_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Recall Marker Title</w:t></w:r></w:p>
<w:p><w:r><w:t>Chonkie and Marker </w:t></w:r><w:r><w:t>are opt-in front doors.</w:t></w:r></w:p>
<w:p><w:r><w:t>DOCX_BODY_SENTINEL_5551</w:t></w:r></w:p>
</w:body>
</w:document>"""

_DOCX_XML_XXE = """<?xml version="1.0"?>
<!DOCTYPE w:document [ <!ENTITY xxe "PWNED"> ]>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body><w:p><w:r><w:t>&xxe;</w:t></w:r></w:p></w:body>
</w:document>"""


def _make_docx(path: str, document_xml: str) -> str:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0"?><Types xmlns='
                   '"http://schemas.openxmlformats.org/package/2006/content-types"/>')
        z.writestr("word/document.xml", document_xml)
    return path


class FlagOffIsTodaysBehavior(unittest.TestCase):
    def setUp(self):
        os.environ.pop("RECALL_FLAG_MARKER", None)
        self.tmp = tempfile.mkdtemp()
        self.docx = _make_docx(os.path.join(self.tmp, "a.docx"), _DOCX_XML)

    def test_disabled_returns_none(self):
        self.assertIsNone(me.export_document(self.docx, enabled=False))

    def test_env_unset_defaults_off(self):
        self.assertIsNone(me.export_document(self.docx))


class DocxFloorExtracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.docx = _make_docx(os.path.join(self.tmp, "a.docx"), _DOCX_XML)

    def test_stdlib_extracts_text_and_heading(self):
        md = me.export_document(self.docx, enabled=True, order="stdlib")
        self.assertIsNotNone(md)
        self.assertIn("DOCX_BODY_SENTINEL_5551", md)
        self.assertIn("are opt-in front doors.", md)
        self.assertIn("# Recall Marker Title", md)     # Heading1 -> "# "

    def test_marker_missing_falls_through_to_stdlib(self):
        def boom(src):
            raise ImportError("simulated absent marker")
        reg = {"marker": boom, "stdlib": me.stdlib_convert}
        warns = []
        md = me.export_document(self.docx, enabled=True, order="marker,stdlib",
                                registry=reg, emit=lambda k, m: warns.append(m))
        self.assertIn("DOCX_BODY_SENTINEL_5551", md)
        self.assertTrue(any("marker:marker failed" in m for m in warns))


class HonestGaps(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_pdf_without_marker_returns_none_not_crash(self):
        pdf = os.path.join(self.tmp, "x.pdf")
        with open(pdf, "wb") as f:
            f.write(b"%PDF-1.7\n% not real content\n")
        def boom(src):
            raise ImportError("no marker")
        reg = {"marker": boom, "stdlib": me.stdlib_convert}
        self.assertIsNone(me.export_document(pdf, enabled=True,
                                             order="marker,stdlib", registry=reg))

    def test_unsupported_extension_returns_none(self):
        txt = os.path.join(self.tmp, "note.txt")
        with open(txt, "w") as f:
            f.write("hi")
        self.assertIsNone(me.export_document(txt, enabled=True))


class SecurityXxe(unittest.TestCase):
    def test_dtd_entity_docx_is_refused(self):
        tmp = tempfile.mkdtemp()
        bad = _make_docx(os.path.join(tmp, "evil.docx"), _DOCX_XML_XXE)
        with self.assertRaises(RuntimeError):
            me.stdlib_convert(bad)                       # provider raises
        # via the seam, a refused parse is an honest None, never "PWNED"
        md = me.export_document(bad, enabled=True, order="stdlib")
        self.assertIsNone(md)


class ZeroDepBoot(unittest.TestCase):
    def test_import_pulled_no_marker(self):
        self.assertFalse(BOOT_PULLED_MARKER,
                         "importing marker_export must stay stdlib-only")


class Cli(unittest.TestCase):
    def setUp(self):
        os.environ.pop("RECALL_FLAG_MARKER", None)
        self.tmp = tempfile.mkdtemp()
        self.in_dir = os.path.join(self.tmp, "docs")
        self.out_dir = os.path.join(self.tmp, "vault")
        os.makedirs(self.in_dir)
        _make_docx(os.path.join(self.in_dir, "report.docx"), _DOCX_XML)

    def test_flag_on_writes_markdown_and_is_idempotent(self):
        rc = me.main(["--in", self.in_dir, "--out", self.out_dir, "--enable", "--order", "stdlib"])
        self.assertEqual(rc, 0)
        out = os.path.join(self.out_dir, "report.md")
        self.assertTrue(os.path.exists(out))
        with open(out, encoding="utf-8") as f:
            body = f.read()
        self.assertIn("DOCX_BODY_SENTINEL_5551", body)
        self.assertIn("marker-export sha=", body)
        # second run detects the marker tag and skips (idempotent)
        rc2 = me.main(["--in", self.in_dir, "--out", self.out_dir, "--enable", "--order", "stdlib"])
        self.assertEqual(rc2, 0)

    def test_flag_off_writes_nothing(self):
        rc = me.main(["--in", self.in_dir, "--out", self.out_dir, "--order", "stdlib"])
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.exists(os.path.join(self.out_dir, "report.md")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
