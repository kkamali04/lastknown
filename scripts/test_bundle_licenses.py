"""Missing notices stay fatal except for verified, unmodified pinned upstream files."""
import json
from pathlib import Path
import shutil

import pytest

from bundle_licenses import missing_objc2_notices


def test_unknown_missing_license_is_not_replaced_by_a_generic_text(tmp_path):
    with pytest.raises(SystemExit, match="No license files found"):
        missing_objc2_notices({"name": "unrecognized-dependency", "version": "0.3.2"}, tmp_path, "unknown", tmp_path)


def test_modified_pinned_notice_is_rejected(tmp_path):
    overrides = tmp_path / "overrides"
    shutil.copytree(Path(__file__).resolve().parents[1] / "licenses/overrides/objc2-0.3.2", overrides)
    provenance = json.loads((overrides / "provenance.json").read_text())
    name = "objc2-core-foundation"
    source = tmp_path / "crate"
    source.mkdir()
    (source / ".cargo_vcs_info.json").write_text(json.dumps({
        "git": {"sha1": provenance["upstream_commit"]}, "path_in_vcs": f"framework-crates/{name}"}))
    package = {"name": name, "version": "0.3.2", "source": "registry+https://github.com/rust-lang/crates.io-index",
               "repository": provenance["repository"], "license": provenance["declared_license"]}
    (overrides / "LICENSE-APACHE-2.0.txt").write_bytes(b"modified text")
    with pytest.raises(SystemExit, match="notice hash mismatch"):
        missing_objc2_notices(package, source, provenance["crates"][name]["checksum"], overrides)
