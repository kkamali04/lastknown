"""Exercise a prebuilt package in a NEW, isolated profile. No autostart or MCP registration.

The native release workflow runs this before uploading an artifact. Only synthetic notes are used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    package = args.package_dir.resolve()
    manifest_bytes = (package / "MANIFEST-SHA256.txt").read_bytes()
    expected = {"MANIFEST-SHA256.txt": hashlib.sha256(manifest_bytes).hexdigest()}
    for line in manifest_bytes.decode("utf-8").splitlines():
        digest, path = line.split("  ", 1)
        expected[path] = digest
    if args.home.exists():
        parser.error("test home must not already exist")
    args.home.mkdir(parents=True)
    home = str(args.home.resolve())
    # Neither provider settings nor a developer's live profile can leak into the test.
    overrides = {"ORT_DYLIB_PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "DYLD_LIBRARY_PATH",
                 "DYLD_FALLBACK_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "PYTHONPATH", "PYTHONHOME"}
    env = {k: v for k, v in os.environ.items() if k.upper() not in overrides and
           not k.upper().startswith(("RECALL_", "ANTHROPIC_", "OPENAI_", "VOYAGE_", "GEMINI_", "PIP_"))}
    env.update(HOME=home, USERPROFILE=home, RECALL_CHANNELS="fts5,bge", TERM="dumb",
               XDG_CONFIG_HOME=str(args.home / ".config"), XDG_CACHE_HOME=str(args.home / ".cache"))
    temp = args.home / "tmp"
    temp.mkdir()
    env.update(TMP=str(temp), TEMP=str(temp), TMPDIR=str(temp),
               PIP_CONFIG_FILE=os.devnull, PIP_CACHE_DIR=str(args.home / ".cache/pip"),
               PIP_DISABLE_PIP_VERSION_CHECK="1")
    windows = os.name == "nt"
    if windows:
        local = args.home / "AppData/Local"
        roaming = args.home / "AppData/Roaming"
        local.mkdir(parents=True)
        roaming.mkdir(parents=True)
        env.update(LOCALAPPDATA=str(local), APPDATA=str(roaming),
                   PSModuleAnalysisCachePath=str(local / "ModuleAnalysisCache"),
                   HOMEDRIVE=args.home.drive, HOMEPATH=str(args.home)[len(args.home.drive):])
    exe = str(args.home / ".recall" / "bin" / ("recall.exe" if windows else "recall"))
    install = (["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(package / "Install-Lastknown.ps1"), "-Vault", str(args.home / "notes")]
               if windows else ["sh", str(package / "Install-Lastknown.sh"), "--vault", str(args.home / "notes")])
    commands = [("install", install),
                ("first-use", [sys.executable, str(package / "scripts/clean_install_smoke.py")]),
                ("scrub", [sys.executable, "-I", str(package / "scripts/scrub_e2e.py"), exe]),
                ("ui-and-daemon", [sys.executable, "-I", str(package / "scripts/ui_e2e.py"), exe])]
    if not windows:
        commands.append(("freshness-demo", ["sh", str(package / "examples/stale-demo/run.sh"), exe]))
    results = {"platform": sys.platform, "steps": [], "passed": False}
    args.results.parent.mkdir(parents=True, exist_ok=True)
    try:
        for name, command in commands:
            print(f"==> {name}", flush=True)
            start = time.monotonic()
            log = args.results.parent / (name + ".log")
            with log.open("w", encoding="utf-8") as output:
                process = subprocess.run(command, env=env, cwd=package, stdout=output,
                                         stderr=subprocess.STDOUT, timeout=1800)
            results["steps"].append({"name": name, "exit_code": process.returncode,
                                     "seconds": round(time.monotonic() - start, 2)})
            if process.returncode:
                raise RuntimeError(f"{name} failed; inspect {log}")
            if name == "install":
                installed = args.home / ".recall" / "bin"
                runtime = "onnxruntime.dll" if windows else ("libonnxruntime.dylib" if sys.platform == "darwin" else "libonnxruntime.so")
                paths = [installed / runtime, installed / Path(exe).name]
                paths += sorted((args.home / ".recall" / "licenses" / "onnxruntime").glob("*"))
                results["installed_sha256"] = {path.relative_to(args.home).as_posix():
                    hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        actual = {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in package.rglob("*") if p.is_file()}
        if actual != expected:
            raise RuntimeError("acceptance changed package files or created unexpected files; inspect the isolated package directory")
        results["package_unchanged"] = True
        results["passed"] = True
    finally:
        args.results.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("NATIVE PACKAGE ACCEPTANCE PASSED")


if __name__ == "__main__":
    main()
