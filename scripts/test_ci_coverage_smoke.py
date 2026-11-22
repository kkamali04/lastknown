"""Hermetic regression tests for ci_coverage_smoke.main().

No live recall binary: subprocess.run and os.path.isfile are monkeypatched so the
threshold logic is exercised in isolation. Locks the contract fixed in 554a7d0
(default floor 99.0 must PASS the structural 100998/100999 baseline, yet still
RED on catastrophic loss or a strict floor).

    py -m pytest scripts/test_ci_coverage_smoke.py -q
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ci_coverage_smoke", Path(__file__).with_name("ci_coverage_smoke.py")
)
smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke)


class _FakeProc:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def _patch(monkeypatch, *, stdout, returncode=0, min_cov=None):
    monkeypatch.setattr(os.path, "isfile", lambda _p: True)
    monkeypatch.setattr(
        smoke.subprocess,
        "run",
        lambda *a, **k: _FakeProc(stdout=stdout, returncode=returncode),
    )
    monkeypatch.setenv("RECALL_EXE", "recall.exe")
    if min_cov is None:
        monkeypatch.delenv("RECALL_MIN_COVERAGE", raising=False)
    else:
        monkeypatch.setenv("RECALL_MIN_COVERAGE", min_cov)


_BASELINE = '{"coverage_pct": 99.99901, "docs_valid": 100999, '\
            '"docs_total": 101004, "embeddings": 100998}'


def test_default_floor_passes_structural_baseline(monkeypatch):
    # The exact case that was permanently RED before 554a7d0.
    _patch(monkeypatch, stdout=_BASELINE)
    assert smoke.main() == 0


def test_full_coverage_passes(monkeypatch):
    _patch(monkeypatch, stdout='{"coverage_pct": 100.0, "docs_valid": 1, '
                               '"docs_total": 1, "embeddings": 1}')
    assert smoke.main() == 0


def test_strict_floor_still_bites(monkeypatch):
    _patch(monkeypatch, stdout=_BASELINE, min_cov="100.0")
    assert smoke.main() == 1


def test_catastrophic_loss_caught(monkeypatch):
    _patch(monkeypatch, stdout='{"coverage_pct": 50.0, "docs_valid": 100, '
                               '"docs_total": 100, "embeddings": 50}')
    assert smoke.main() == 1


def test_missing_coverage_key_fails(monkeypatch):
    _patch(monkeypatch, stdout='{"docs_valid": 1, "docs_total": 1}')
    assert smoke.main() == 1


def test_recall_nonzero_exit_fails(monkeypatch):
    _patch(monkeypatch, stdout="", returncode=2)
    assert smoke.main() == 1


def test_invalid_json_fails(monkeypatch):
    _patch(monkeypatch, stdout="not json at all")
    assert smoke.main() == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
