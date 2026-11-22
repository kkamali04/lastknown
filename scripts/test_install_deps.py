"""The installers refuse without ripgrep instead of installing a recall that silently degrades.

Each case runs the installer's `--check-deps` / `-CheckDeps` with PATH reduced to a temp dir, so no
rg and no package manager is visible; then again with a stub `rg` on that PATH.
"""
import os, shutil, stat, subprocess, sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent


def _stub_rg(bindir: Path, code: int = 0) -> None:
    """A stub `rg` that prints a version and exits `code` (non-zero = a broken install)."""
    if os.name == "nt":
        (bindir / "rg.cmd").write_text(f"@echo ripgrep 0.0-stub\r\n@exit /b {code}\r\n")
    else:
        p = bindir / "rg"
        p.write_text(f"#!/bin/sh\necho ripgrep 0.0-stub\nexit {code}\n")
        p.chmod(p.stat().st_mode | stat.S_IEXEC)


@pytest.mark.skipif(os.name == "nt", reason="install.sh is the Linux/macOS installer; install.ps1 is tested below")
def test_install_sh_refuses_without_rg_and_passes_with_it(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for tool in ("dirname", "id"):            # what the script runs before and inside the rg check
        os.symlink(shutil.which(tool), bindir / tool)
    run = lambda: subprocess.run(["/bin/sh", str(SCRIPTS / "install.sh"), "--check-deps"],
                                 env={"PATH": str(bindir), "HOME": str(tmp_path)},
                                 capture_output=True, text=True, timeout=60)
    r = run()
    assert r.returncode == 1, (r.stdout, r.stderr)
    assert "ripgrep (rg) is required" in r.stderr and "apt-get install ripgrep" in r.stderr, r.stderr
    _stub_rg(bindir, code=2)                 # on PATH but broken: still refused
    r = run()
    assert r.returncode == 1 and "ripgrep (rg) is required" in r.stderr, (r.stdout, r.stderr)
    _stub_rg(bindir)
    r = run()
    assert r.returncode == 0 and "dependencies OK" in r.stdout, (r.stdout, r.stderr)


def _powershell():
    return shutil.which("pwsh") or (shutil.which("powershell") if os.name == "nt" else None)


@pytest.mark.skipif(_powershell() is None, reason="no PowerShell on this machine")
def test_install_ps1_refuses_without_rg_and_passes_with_it(tmp_path):
    ps = _powershell()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    env = {"PATH": str(bindir), "TEMP": str(tmp_path), "TMP": str(tmp_path), "HOME": str(tmp_path)}
    if os.name == "nt":
        env["SystemRoot"] = os.environ.get("SystemRoot", r"C:\Windows")
        env["PATHEXT"] = ".COM;.EXE;.BAT;.CMD"
    run = lambda: subprocess.run([ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                                  str(SCRIPTS / "install.ps1"), "-CheckDeps"],
                                 env=env, capture_output=True, text=True, timeout=120)
    r = run()
    assert r.returncode != 0, (r.stdout, r.stderr)
    assert "ripgrep (rg) is required" in r.stdout + r.stderr, (r.stdout, r.stderr)
    _stub_rg(bindir, code=2)                 # on PATH but broken: still refused
    r = run()
    assert r.returncode != 0 and "ripgrep (rg) is required" in r.stdout + r.stderr, (r.stdout, r.stderr)
    _stub_rg(bindir)
    r = run()
    assert r.returncode == 0 and "dependencies OK" in r.stdout, (r.stdout, r.stderr)
