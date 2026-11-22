"""Collect verbatim license/notice files from the locked default Rust dependency graph.

Run from any directory: python scripts/bundle_licenses.py --target x86_64-pc-windows-msvc
Requires the dependency sources already in Cargo's cache; never downloads them. The output
contains portable package provenance and hashes, not local registry paths. Build dependencies
are included conservatively. Optional features need their own inventory before distribution.
Two exact objc2 releases omit license files; their pinned upstream declaration and selected
Apache-2.0 text are vendored under licenses/overrides and verified before use.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib


def missing_objc2_notices(package, source, checksum, overrides):
    """Only the two audited registry releases may use these pinned upstream notices."""
    name = package["name"]
    if name not in {"objc2-core-foundation", "objc2-core-services"} or package["version"] != "0.3.2":
        raise SystemExit(f"No license files found for {name} {package['version']}")
    provenance = json.loads((overrides / "provenance.json").read_text(encoding="utf-8"))
    expected = provenance["crates"][name]
    vcs = json.loads((source / ".cargo_vcs_info.json").read_text(encoding="utf-8"))
    if (package.get("source") != "registry+https://github.com/rust-lang/crates.io-index"
        or package.get("repository") != provenance["repository"]
        or package.get("license") != provenance["declared_license"]
        or expected["version"] != package["version"] or expected["checksum"] != checksum
        or vcs != {"git": {"sha1": provenance["upstream_commit"]}, "path_in_vcs": f"framework-crates/{name}"}
        or provenance["selected_license"] != "Apache-2.0"):
        raise SystemExit(f"Pinned upstream license identity mismatch for {name}")
    if {f["path"] for f in provenance["files"]} != {"LICENSE-UPSTREAM.md", "LICENSE-APACHE-2.0.txt"} or len(provenance["files"]) != 2:
        raise SystemExit("Unexpected objc2 upstream notice set")
    files = []
    for entry in provenance["files"]:
        file = overrides / entry["path"]
        if hashlib.sha256(file.read_bytes()).hexdigest() != entry["sha256"]:
            raise SystemExit(f"Pinned upstream notice hash mismatch: {entry['path']}")
        files.append(file)
    origin = {k: provenance[k] for k in ("repository", "upstream_commit", "files")}
    return files, origin


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
        origin = None
        files = sorted(f for f in source.rglob("*") if f.is_file() and
            f.name.upper().startswith(("LICENSE", "LICENCE", "COPYING", "COPYRIGHT", "NOTICE")))
        if p.get("license_file"):
            files = sorted(set(files) | {source / p["license_file"]})
        if not files:
            files, origin = missing_objc2_notices(p, source, checksums[(p["name"], p["version"])],
                                                 root / "licenses/overrides/objc2-0.3.2")
            source = root / "licenses/overrides/objc2-0.3.2"
        inventory.append((p, source, files, origin))
    out.mkdir(parents=True)
    manifest = {"target": args.target, "features": "default", "includes_build_dependencies": True,
        "cargo_lock_sha256": hashlib.sha256((root / "Cargo.lock").read_bytes()).hexdigest(), "packages": []}
    rows = ["# Locked Rust dependency licenses", "",
        f"Default features, `{args.target}`. Includes build dependencies; not a claim that every package is linked.",
        "Verbatim license/notice files from Cargo's verified source cache are listed in `manifest.json`.",
        "The two objc2 0.3.2 releases that omit notices use pinned upstream files; their origin and selected license are explicit in the manifest.",
        "Optional-feature and other-platform distributions must regenerate their inventories.", "",
        "| Package | Declared SPDX expression | License files |", "|---|---|---|"]
    for p, source, files, origin in inventory:
        key = f"{p['name']}-{p['version']}"
        entry = {"name": p["name"], "version": p["version"], "license": p.get("license"),
            "source": p.get("source"), "crate_checksum": checksums[(p["name"], p["version"])], "files": []}
        if origin is not None:
            entry.update(selected_license="Apache-2.0", license_files_origin=origin)
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
