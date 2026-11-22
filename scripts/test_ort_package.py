"""Native-package selection must neither install the wrong ABI nor trust ZIP paths."""
import zipfile

import pytest

import ort_package


@pytest.mark.parametrize("runtime", ort_package.LIBRARIES)
def test_extract_only_selected_runtime_and_notices(tmp_path, runtime):
    archive = tmp_path / "package.zip"
    with zipfile.ZipFile(archive, "w") as package:
        for platform, libraries in ort_package.LIBRARIES.items():
            for name in libraries:
                package.writestr(f"runtimes/{platform}/native/{name}", platform.encode())
        for name in ort_package.NOTICES:
            package.writestr(name, b"verbatim notice")
        package.writestr("../escaped.txt", b"must not extract")
    destination, notices = tmp_path / "bin", tmp_path / "notices"
    ort_package.extract(archive, runtime, destination, notices)
    assert sorted(f.name for f in destination.iterdir()) == sorted(ort_package.LIBRARIES[runtime])
    assert all(f.read_bytes() == runtime.encode() for f in destination.iterdir())
    assert all((notices / name).read_bytes() == b"verbatim notice" for name in ort_package.NOTICES)
    assert not (tmp_path / "escaped.txt").exists()


def test_missing_library_does_not_create_partial_install(tmp_path):
    archive = tmp_path / "package.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("runtimes/linux-x64/native/libonnxruntime.so", b"runtime")
    with pytest.raises(KeyError):
        ort_package.extract(archive, "linux-x64", tmp_path / "bin", tmp_path / "notices")
    assert not (tmp_path / "bin").exists()


def test_intel_mac_rejected_before_writes(tmp_path):
    with pytest.raises(KeyError):
        ort_package.extract(tmp_path / "absent.zip", "osx-x64", tmp_path / "bin", tmp_path / "notices")
    assert not (tmp_path / "bin").exists()
