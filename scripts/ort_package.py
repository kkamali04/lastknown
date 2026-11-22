"""Extract only the selected native libraries and notices from a verified Microsoft NuGet ZIP.

The caller must verify the pinned archive SHA256 before invoking this helper.
No .NET installation is needed. Never extract arbitrary archive paths.
"""
import argparse
from pathlib import Path
import zipfile


LIBRARIES = {
    "linux-x64": ("libonnxruntime.so", "libonnxruntime_providers_shared.so"),
    "linux-arm64": ("libonnxruntime.so", "libonnxruntime_providers_shared.so"),
    "osx-arm64": ("libonnxruntime.dylib",),
}
NOTICES = ("LICENSE", "ThirdPartyNotices.txt")


def extract(archive, runtime, destination, notices):
    libraries = LIBRARIES[runtime]  # Reject an unsupported runtime before writing anything.
    entries = [(f"runtimes/{runtime}/native/{name}", Path(destination) / name)
               for name in libraries]
    entries += [(name, Path(notices) / name) for name in NOTICES]
    with zipfile.ZipFile(archive) as package:
        # Check every member before writing; a missing library must fail the installation.
        contents = [(path, package.read(member)) for member, path in entries]
    for path, data in contents:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("runtime", choices=LIBRARIES)
    parser.add_argument("destination", type=Path)
    parser.add_argument("notices", type=Path)
    args = parser.parse_args()
    extract(args.archive, args.runtime, args.destination, args.notices)


if __name__ == "__main__":
    main()
