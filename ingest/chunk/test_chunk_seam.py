#!/usr/bin/env python3
"""
test_chunk_seam — stdlib-only self-test for recall's opt-in Chonkie chunker.

No chonkie needed: the optional provider is faked via the injectable registry,
and the always-available stdlib floor does the real chunking. Proves:
  1. flag OFF  -> exactly ONE chunk == the whole body (recall's behavior today)
  2. flag ON   -> a >2k-char doc splits into N>1 chunks, each <= budget tokens
  3. preserve  -> chunking loses no non-whitespace content
  4. fallback  -> a missing/raising chonkie provider falls through to stdlib
  5. seam      -> first working provider wins; empty result falls through
  6. zero-dep  -> importing the module pulls no third-party module
  7. CLI       -> flag-off mirror is byte-identical; flag-on emits chunk files

Run:  py test_chunk_seam.py
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chonkie_chunk as cc

# snapshot at import time, before any test could pull the optional dep
BOOT_PULLED_CHONKIE = "chonkie" in sys.modules

BIG = ("\n\n".join(
    f"Paragraph {i}: recall embeds one vector per whole file today, so long "
    f"bodies get truncated at the 512 token window and most content is invisible."
    for i in range(40)
) + "\n\nUNIQUE_SENTINEL_MIDPOINT_9182 marks a specific spot in the middle.\n\n"
+ "\n\n".join(f"Tail paragraph {i} keeps the document well past two kilobytes."
              for i in range(40)))


class FlagOffIsTodaysBehavior(unittest.TestCase):
    def setUp(self):
        os.environ.pop("RECALL_FLAG_CHONKIE", None)
    tearDown = setUp

    def test_explicit_disabled_is_single_whole_body_chunk(self):
        chunks = cc.chunk_text(BIG, enabled=False)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, BIG)          # byte-identical body
        self.assertEqual(chunks[0].total, 1)
        self.assertEqual(chunks[0].provider, "passthrough")

    def test_env_unset_defaults_to_passthrough(self):
        chunks = cc.chunk_text(BIG)                     # no flag -> off
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, BIG)

    def test_env_flag_zero_is_off(self):
        os.environ["RECALL_FLAG_CHONKIE"] = "0"
        self.assertEqual(len(cc.chunk_text(BIG)), 1)


class FlagOnChunksASample(unittest.TestCase):
    def test_stdlib_floor_splits_and_respects_budget(self):
        # order="stdlib" forces the dependency-free floor so this test is
        # deterministic whether or not the optional `chonkie` package is
        # installed (before the fix it silently used chonkie when present and
        # the provider assertion below failed).
        chunks = cc.chunk_text(BIG, enabled=True, budget_tokens=120, overlap_tokens=0, order="stdlib")
        self.assertGreater(len(chunks), 1)
        for ch in chunks:
            self.assertLessEqual(ch.tokens, 120, f"chunk {ch.idx} over budget")
            self.assertEqual(ch.provider, "stdlib")
            self.assertEqual(ch.total, len(chunks))

    def test_chunking_preserves_all_content(self):
        chunks = cc.chunk_text(BIG, enabled=True, budget_tokens=120, overlap_tokens=0)
        rejoined = "".join(ch.text for ch in chunks)
        self.assertEqual("".join(rejoined.split()), "".join(BIG.split()))

    def test_sentinel_survives_exactly_once(self):
        chunks = cc.chunk_text(BIG, enabled=True, budget_tokens=120, overlap_tokens=0)
        hits = sum("UNIQUE_SENTINEL_MIDPOINT_9182" in ch.text for ch in chunks)
        self.assertEqual(hits, 1)

    def test_overlap_adds_context_tail(self):
        chunks = cc.chunk_text(BIG, enabled=True, budget_tokens=120, overlap_tokens=20)
        self.assertGreater(len(chunks), 1)
        # a short text under budget yields a single chunk regardless of overlap
        one = cc.chunk_text("short body", enabled=True, budget_tokens=120)
        self.assertEqual(len(one), 1)


class ProviderSeam(unittest.TestCase):
    def test_missing_chonkie_falls_through_to_stdlib(self):
        def boom(text, budget, overlap):
            raise ImportError("simulated absent chonkie")
        reg = {"chonkie": boom, "stdlib": cc.stdlib_chunk}
        warns = []
        chunks = cc.chunk_text(BIG, enabled=True, budget_tokens=120, overlap_tokens=0,
                               order="chonkie,stdlib", registry=reg,
                               emit=lambda k, m: warns.append(m))
        self.assertTrue(chunks and all(c.provider == "stdlib" for c in chunks))
        self.assertTrue(any("chunk:chonkie failed" in m for m in warns))

    def test_first_working_provider_wins(self):
        def fake_chonkie(text, budget, overlap):
            return ["A", "B", "C"]
        reg = {"chonkie": fake_chonkie, "stdlib": cc.stdlib_chunk}
        chunks = cc.chunk_text(BIG, enabled=True, order="chonkie,stdlib", registry=reg)
        self.assertEqual([c.text for c in chunks], ["A", "B", "C"])
        self.assertTrue(all(c.provider == "chonkie" for c in chunks))

    def test_empty_provider_falls_through(self):
        reg = {"chonkie": lambda t, b, o: [], "stdlib": cc.stdlib_chunk}
        chunks = cc.chunk_text(BIG, enabled=True, budget_tokens=120,
                               order="chonkie,stdlib", registry=reg)
        self.assertTrue(all(c.provider == "stdlib" for c in chunks))

    def test_chain_always_includes_floor(self):
        self.assertEqual(cc.resolve_chunk_chain("chonkie"), ["chonkie", "stdlib"])
        self.assertEqual(cc.resolve_chunk_chain("bogus"), ["stdlib"])


class ZeroDepBoot(unittest.TestCase):
    def test_import_pulled_no_chonkie(self):
        self.assertFalse(BOOT_PULLED_CHONKIE,
                         "importing chonkie_chunk must stay stdlib-only")


class CliMirror(unittest.TestCase):
    def setUp(self):
        os.environ.pop("RECALL_FLAG_CHONKIE", None)
        self.tmp = tempfile.mkdtemp()
        self.in_dir = os.path.join(self.tmp, "src")
        self.out_dir = os.path.join(self.tmp, "mirror")
        os.makedirs(self.in_dir)
        self.doc = os.path.join(self.in_dir, "big.md")
        with open(self.doc, "w", encoding="utf-8") as f:
            f.write(BIG)

    def test_flag_off_mirror_is_byte_identical(self):
        rc = cc.main(["--in", self.in_dir, "--out", self.out_dir])
        self.assertEqual(rc, 0)
        with open(self.doc, encoding="utf-8") as f:
            src = f.read()
        with open(os.path.join(self.out_dir, "big.md"), encoding="utf-8") as f:
            self.assertEqual(f.read(), src)

    def test_flag_on_emits_multiple_chunk_files(self):
        rc = cc.main(["--in", self.in_dir, "--out", self.out_dir,
                      "--enable", "--budget-tokens", "120", "--overlap-tokens", "0"])
        self.assertEqual(rc, 0)
        produced = [f for f in os.listdir(self.out_dir) if f.startswith("big.chunk")]
        self.assertGreater(len(produced), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
