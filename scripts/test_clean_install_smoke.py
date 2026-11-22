"""The clean-install smoke's MCP health check must see the WHOLE reply (Codex clean-install r2: a
failed channel probe past character 600 was cut off and passed)."""
from clean_install_smoke import health_ok

STATUS = "binary: recall\nstatus exit=0\n" + "x" * 1100 + "\n\nper-channel probe:\n"


def test_all_channels_ok_passes():
    assert health_ok(STATUS + "  fts5       OK   3 hits\n  bge        OK   3 hits\n  fts5,bge   OK   3 hits")


def test_a_failed_probe_far_into_the_reply_fails():
    text = STATUS + "  fts5       OK   3 hits\n  bge        FAIL no vectors\n  fts5,bge   OK   3 hits"
    assert text.index("FAIL") > 600
    assert not health_ok(text)


def test_a_missing_probe_fails():
    assert not health_ok(STATUS + "  fts5       OK   3 hits")
