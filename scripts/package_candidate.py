"""Package a clean native build, tracked source, and target-specific Rust notices.

Does not publish. The output directory must be new. Run bundle_licenses.py first.
"""
import argparse
import datetime
import hashlib
import io
import json
import platform
from pathlib import Path
import subprocess
import tarfile
import zipfile


TARGETS = {
    "x86_64-pc-windows-msvc": "windows-x64",
    "x86_64-unknown-linux-gnu": "linux-x64",
    "aarch64-unknown-linux-gnu": "linux-arm64",
    "aarch64-apple-darwin": "macos-arm64",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--licenses", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    git = lambda *cmd: subprocess.check_output(["git", *cmd], cwd=root)
    if git("status", "--porcelain", "--untracked-files=normal").strip():
        parser.error("commit source changes and keep generated output outside the checkout")
    if args.out.exists():
        parser.error("output already exists; choose a new directory")
    target = subprocess.check_output(["rustc", "-vV"], text=True).split("host: ")[1].splitlines()[0]
    if target != args.target:
        parser.error(f"native packaging requires {args.target}, but rustc host is {target}")
    binary = args.binary.read_bytes()
    manifest = json.loads((args.licenses / "manifest.json").read_text(encoding="utf-8"))
    if manifest["target"] != args.target or manifest["cargo_lock_sha256"] != sha(git("show", "HEAD:Cargo.lock")):
        parser.error("license inventory does not match the target and committed Cargo.lock")
    # Canonical committed bytes, without .git, untracked files, CRLF conversion, or caches.
    files, modes = {}, {}
    with tarfile.open(fileobj=io.BytesIO(git("archive", "--format=tar", "HEAD"))) as archive:
        for member in archive:
            if member.isdir():
                continue
            if not member.isfile():
                parser.error(f"unsupported source member: {member.name}")
            if member.name.startswith("licenses/rust/"):
                continue  # Replace the earlier Windows inventory with this target's inventory.
            files[member.name] = archive.extractfile(member).read()
            modes[member.name] = member.mode
    for package in manifest["packages"]:
        for entry in package["files"]:
            if sha((args.licenses / entry["path"]).read_bytes()) != entry["sha256"]:
                parser.error(f"license hash mismatch: {entry['path']}")
    for path in args.licenses.rglob("*"):
        if path.is_file():
            files["licenses/rust/" + path.relative_to(args.licenses).as_posix()] = path.read_bytes()
    name = "lastknown-" + TARGETS[args.target]
    windows = args.target.endswith("windows-msvc")
    files["prebuilt/release/recall" + (".exe" if windows else "")] = binary
    if windows:
        files["Install-Lastknown.ps1"] = b'''[CmdletBinding()]
param([string]$Vault = (Join-Path $env:USERPROFILE 'recall-notes'))
$ErrorActionPreference = 'Stop'
$PreviousTarget = $env:CARGO_TARGET_DIR
try {
    $env:CARGO_TARGET_DIR = Join-Path $PSScriptRoot 'prebuilt'
    & (Join-Path $PSScriptRoot 'scripts/install.ps1') -SkipBuild -Vault $Vault
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    if ($null -eq $PreviousTarget) { Remove-Item Env:CARGO_TARGET_DIR -ErrorAction SilentlyContinue }
    else { $env:CARGO_TARGET_DIR = $PreviousTarget }
}
'''
        command = "powershell -ExecutionPolicy Bypass -File .\\Install-Lastknown.ps1"
    else:
        files["Install-Lastknown.sh"] = b'''#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export CARGO_TARGET_DIR="$ROOT/prebuilt"
exec sh "$ROOT/scripts/install.sh" --skip-build "$@"
'''
        command = "sh ./Install-Lastknown.sh"
    files["README-PACKAGE.md"] = (f"# Lastknown ({TARGETS[args.target]})\n\n"
        "Extract the entire archive, then run from its directory:\n\n"
        f"```\n{command}\n```\n\n"
        "No Rust compiler is needed for this prebuilt package. Python 3.10+ (with venv), "
        "ripgrep, and internet access are required at installation. The installer fetches "
        "the pinned ONNX Runtime, BGE model, and MCP Python dependencies; search then works "
        "offline. See README.md and docs/PLATFORMS.md for limits and source-build instructions.\n\n"
        "Apache-2.0 applies to Lastknown's source. Bundled third-party notices retain their "
        "own terms; see LICENSE, NOTICE, and licenses/.\n").encode()
    info = {"source_commit": git("rev-parse", "HEAD").decode().strip(), "target": args.target,
            "features": "default", "binary_sha256": sha(binary), "onnxruntime": "1.24.4",
            "rustc": subprocess.check_output(["rustc", "-Vv"], text=True).strip(),
            "build_system": platform.platform(),
            "packaged_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    files["BUILD-INFO.json"] = (json.dumps(info, indent=2) + "\n").encode()
    files["MANIFEST-SHA256.txt"] = "".join(f"{sha(data)}  {path}\n" for path, data in sorted(files.items())).encode()
    args.out.mkdir(parents=True)
    suffix = ".zip" if windows else ".tar.gz"
    artifact = args.out / (name + suffix)
    if windows:
        with zipfile.ZipFile(artifact, "w", zipfile.ZIP_DEFLATED) as archive:
            for path, data in sorted(files.items()):
                archive.writestr(f"{name}/{path}", data)
        with zipfile.ZipFile(artifact) as archive:
            assert archive.testzip() is None
            assert {p[len(name) + 1:]: archive.read(p) for p in archive.namelist()} == files
    else:
        with tarfile.open(artifact, "w:gz") as archive:
            for path, data in sorted(files.items()):
                member = tarfile.TarInfo(f"{name}/{path}")
                member.size = len(data)
                member.mode = 0o755 if path.endswith(".sh") or path == "prebuilt/release/recall" else modes.get(path, 0o644)
                archive.addfile(member, io.BytesIO(data))
        with tarfile.open(artifact) as archive:
            assert {m.name[len(name) + 1:]: archive.extractfile(m).read() for m in archive} == files
    # Verify/install the bytes read back from the actual archive, rather than an unrelated tree.
    stage = args.out / name
    for path, data in files.items():
        destination = stage / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        if not windows and (path.endswith(".sh") or path == "prebuilt/release/recall"):
            destination.chmod(0o755)
    (args.out / "SHA256SUMS.txt").write_text(f"{sha(artifact.read_bytes())}  {artifact.name}\n", encoding="utf-8")
    print(artifact)


if __name__ == "__main__":
    main()
