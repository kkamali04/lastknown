"""Collect verbatim license/notice files from the locked default Rust dependency graph.

Run from any directory: python scripts/bundle_licenses.py --target x86_64-pc-windows-msvc
Requires the dependency sources already in Cargo's cache; never downloads them. The output
contains portable package provenance and hashes, not local registry paths. Build dependencies
are included conservatively. Optional features need their own inventory before distribution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out or root / "licenses" / "rust"
    if out.exists():
        parser.error("output already exists; use a new directory to preserve prior evidence")
    metadata = json.loads(subprocess.check_output([
        "cargo", "metadata", "--locked", "--offline", "--format-version", "1",
        "--filter-platform", args.target], cwd=root))
    graph = {node["id"] for node in metadata["resolve"]["nodes"]}
    packages = sorted((p for p in metadata["packages"]
        if p["id"] in graph and p["id"] not in metadata["workspace_members"]),
        key=lambda p: (p["name"], p["version"]))
    checksums = {(p["name"], p["version"]): p.get("checksum") for p in
        tomllib.loads((root / "Cargo.lock").read_text(encoding="utf-8"))["package"]}
    # Check everything before writing so incomplete coverage cannot look successful.
    inventory = []
    for p in packages:
        source = Path(p["manifest_path"]).parent
        files = sorted(f for f in source.rglob("*") if f.is_file() and
            f.name.upper().startswith(("LICENSE", "LICENCE", "COPYING", "COPYRIGHT", "NOTICE")))
        if p.get("license_file"):
            files = sorted(set(files) | {source / p["license_file"]})
        if not files:
            raise SystemExit(f"No license files found for {p['name']} {p['version']}")
        inventory.append((p, source, files))
    out.mkdir(parents=True)
    manifest = {"target": args.target, "features": "default", "includes_build_dependencies": True,
        "cargo_lock_sha256": hashlib.sha256((root / "Cargo.lock").read_bytes()).hexdigest(), "packages": []}
    rows = ["# Locked Rust dependency licenses", "",
        f"Default features, `{args.target}`. Includes build dependencies; not a claim that every package is linked.",
        "Verbatim license/notice files from Cargo's verified source cache are listed in `manifest.json`.",
        "Optional-feature and other-platform distributions must regenerate their inventories.", "",
        "| Package | Declared SPDX expression | License files |", "|---|---|---|"]
    for p, source, files in inventory:
        key = f"{p['name']}-{p['version']}"
        entry = {"name": p["name"], "version": p["version"], "license": p.get("license"),
            "source": p.get("source"), "crate_checksum": checksums[(p["name"], p["version"])], "files": []}
        links = []
        for file in files:
            relative = Path(key) / file.relative_to(source)
            data = file.read_bytes()
            dest = out / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            entry["files"].append({"path": relative.as_posix(), "sha256": hashlib.sha256(data).hexdigest()})
            links.append(f"[{file.relative_to(source).as_posix()}]({relative.as_posix()})")
        manifest["packages"].append(entry)
        rows.append(f"| {key} | {p.get('license') or 'UNSPECIFIED'} | {', '.join(links)} |")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Collected {len(packages)} packages, {sum(len(e['files']) for e in manifest['packages'])} license/notice files")


if __name__ == "__main__":
    main()
