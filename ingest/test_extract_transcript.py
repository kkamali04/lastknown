"""Tests for extract_transcript.py."""

import inspect
import json
import pathlib
import tempfile

from ingest.extract_transcript import (
    MAX_TOOL_RESULT_DEFAULT,
    _cap_tool_result,
    _summarise_input,
    iter_turns,
)

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "synthetic-session.jsonl"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def collect(path, **kwargs):
    return list(iter_turns(str(path), **kwargs))


# ---------------------------------------------------------------------------
# Fixture parses without error
# ---------------------------------------------------------------------------

def test_fixture_parses():
    turns = collect(FIXTURE)
    assert len(turns) >= 5, f"Expected >=5 turns, got {len(turns)}"


def test_fixture_schema():
    turns = collect(FIXTURE)
    required = {"turn_idx", "ts", "role", "kind", "tool_name", "text"}
    for t in turns:
        assert required == set(t.keys()), f"Missing keys in {t}"


def test_turn_idx_monotonic():
    turns = collect(FIXTURE)
    idxs = [t["turn_idx"] for t in turns]
    assert idxs == list(range(len(idxs)))


# ---------------------------------------------------------------------------
# >1MB line handled without whole-file read
# ---------------------------------------------------------------------------

def test_large_line_handled():
    """The >1MB line must not crash and must be parsed."""
    turns = collect(FIXTURE)
    # At least one user_text turn must have text length > 500_000 chars
    large = [t for t in turns if t["kind"] == "user_text" and len(t["text"]) > 500_000]
    assert large, "Expected at least one turn from the >1MB line"


def test_no_whole_file_read():
    """iter_turns must never call file.read() — verify via source inspection."""
    src = inspect.getsource(iter_turns)
    assert "fh.read()" not in src, "iter_turns must not call file.read()"
    assert "fh.readlines()" not in src, "iter_turns must not call file.readlines()"


def test_iterator_based_implementation():
    """iter_turns must use 'for ... in fh' iteration, not read()."""
    src = inspect.getsource(iter_turns)
    assert "for raw_line in fh" in src, "iter_turns must iterate file handle line-by-line"


# ---------------------------------------------------------------------------
# tool_result capped at 500 chars
# ---------------------------------------------------------------------------

def test_tool_result_cap():
    long_body = "X" * 2000
    capped = _cap_tool_result(long_body, 500)
    assert len(capped) > 500  # includes note
    assert capped.startswith("X" * 500)
    assert "bytes total" in capped


def test_tool_result_not_truncated_when_short():
    short = "hello world"
    assert _cap_tool_result(short, 500) == short


def test_fixture_tool_result_capped():
    """All tool_result turns from the fixture respect the default cap."""
    turns = collect(FIXTURE)
    for t in turns:
        if t["kind"] == "tool_result":
            # text may exceed cap only if note is appended
            body_before_note = t["text"].split(" ... [")[0]
            assert len(body_before_note) <= MAX_TOOL_RESULT_DEFAULT


# ---------------------------------------------------------------------------
# Bad JSON line counted, not fatal
# ---------------------------------------------------------------------------

def test_bad_json_not_fatal():
    """A file with one bad JSON line must not raise; good lines still emitted."""
    lines = [
        json.dumps({"type": "user", "message": {"role": "user", "content": "hi"},
                    "timestamp": "2026-01-01T00:00:00Z", "sessionId": "s1"}),
        "THIS IS NOT VALID JSON",
        json.dumps({"type": "assistant", "message": {"role": "assistant",
                    "content": [{"type": "text", "text": "hello"}]},
                    "timestamp": "2026-01-01T00:00:01Z", "sessionId": "s1"}),
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                     encoding="utf-8", delete=False) as f:
        f.write("\n".join(lines) + "\n")
        tmp = f.name

    turns = collect(tmp)
    assert len(turns) == 2, f"Expected 2 good turns, got {len(turns)}"


def test_bad_json_count_in_fixture():
    """The fixture has exactly 1 bad JSON line; iter_turns must not raise."""
    # Just verifying no exception is the assertion
    turns = collect(FIXTURE)
    assert len(turns) > 0


# ---------------------------------------------------------------------------
# tool_name populated on tool records
# ---------------------------------------------------------------------------

def test_tool_name_on_tool_use():
    turns = collect(FIXTURE)
    tool_use_turns = [t for t in turns if t["kind"] == "tool_use"]
    assert tool_use_turns, "Expected at least one tool_use turn"
    for t in tool_use_turns:
        assert t["tool_name"] is not None, f"tool_name missing on tool_use: {t}"


def test_tool_name_on_tool_result():
    turns = collect(FIXTURE)
    tool_result_turns = [t for t in turns if t["kind"] == "tool_result"]
    assert tool_result_turns, "Expected at least one tool_result turn"
    for t in tool_result_turns:
        assert t["tool_name"] is not None, f"tool_name missing on tool_result: {t}"


def test_tool_name_crypt():
    """The fixture has a tool_result from a tool named 'crypt' — verify it."""
    turns = collect(FIXTURE)
    crypt_results = [t for t in turns if t["kind"] == "tool_result"
                     and t["tool_name"] == "crypt"]
    assert crypt_results, "Expected a tool_result with tool_name='crypt'"


def test_tool_name_none_on_prose():
    turns = collect(FIXTURE)
    prose = [t for t in turns if t["kind"] in ("user_text", "assistant_text")]
    assert prose, "Expected prose turns"
    for t in prose:
        assert t["tool_name"] is None, f"tool_name should be None on prose: {t}"


# ---------------------------------------------------------------------------
# input summary capped at 200 chars
# ---------------------------------------------------------------------------

def test_summarise_input_truncates():
    big = {"key": "V" * 500}
    result = _summarise_input(big)
    # 200 chars of content + " ... [NNN bytes total]" note (up to ~30 chars)
    assert len(result) <= 230
    assert "bytes total" in result
    # The raw serialised text must exceed 200 chars (confirms truncation happened)
    import json as _json
    raw = _json.dumps(big, ensure_ascii=False)
    assert len(raw) > 200


def test_summarise_input_small():
    small = {"cmd": "status"}
    result = _summarise_input(small)
    assert "status" in result
    assert len(result) < 200
