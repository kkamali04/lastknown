# Native platforms and release validation

Lastknown's Rust source is portable, but a working semantic search installation also needs a
compatible ONNX Runtime, ripgrep, Python with venv, and the BGE model. Compiling a target alone
does not verify installation, search, file watching, or the local UI on that platform.

| Platform | Native candidate target | Required validation for a prebuilt release |
|---|---|---|
| Windows x64 | `x86_64-pc-windows-msvc` | Native Windows workflow and fresh package acceptance |
| Linux x64 (glibc) | `x86_64-unknown-linux-gnu` | Native Ubuntu 24.04 workflow and fresh package acceptance |
| Linux ARM64 (glibc) | `aarch64-unknown-linux-gnu` | Native Ubuntu ARM64 workflow and fresh package acceptance |
| macOS Apple Silicon | `aarch64-apple-darwin` | Native macOS 15 workflow and fresh package acceptance |
| macOS Intel | — | Not supported by the pinned ONNX Runtime 1.24.4 NuGet package |
| Windows ARM64, Linux musl, mobile, BSD | — | No validated prebuilt package in this release |

This table defines the release requirements. Completed runs and exact artifact hashes are
recorded with each [GitHub release](https://github.com/kkamali04/lastknown/releases), including
its `NATIVE-VALIDATION.json` receipt. A source build or a prepared workflow alone is not a pass.

The Linux validation baseline is Ubuntu 24.04. Older glibc distributions and other Linux
distributions need separate testing. WSL evidence is Linux user-space execution, with a
Windows-managed kernel and filesystem integration; it is not an independent Linux desktop test.
macOS builds are command-line binaries, not signed/notarized application bundles. Launching
the UI with `--no-open` exercises the HTTP interface; automatic browser launch and autostart
registration still need interactive checks on the intended desktop.

The installer downloads the pinned Microsoft NuGet package and extracts only its selected
native libraries and verbatim notices, after checking the entire archive SHA256. No .NET
runtime is required. Intel macOS is refused explicitly rather than silently losing BGE search.

## Prebuilt packages

Extract the entire matching archive. Run `sh ./Install-Lastknown.sh` on Linux/macOS, or
`powershell -ExecutionPolicy Bypass -File .\Install-Lastknown.ps1` on Windows.
Use `--vault /path/to/notes` (Unix) or `-Vault C:\path\to\notes` (Windows) to choose your notes.
Python 3.10+ with venv, ripgrep, and internet access are needed at installation; Rust is not.
The model and native runtime are downloaded during installation. Each package contains
source, Apache-2.0 LICENSE, third-party notices, BUILD-INFO.json, and a SHA256 file manifest.

## Native build and release tests

The manual **Native release candidates** workflow runs on the actual OS/architecture for
each supported target. It tests the default Rust build and Python components, builds a native
binary, collects that target's dependency licenses, packages it, verifies archive readback,
and tests the packaged installer in a fresh profile. Acceptance covers actual BGE embeddings
and search, all MCP tools, redaction, local UI/daemon requests, and Unix freshness behavior.
Only a passing job uploads its candidate. The source artifact waits for all native jobs.

It has no publishing permissions, release/tag creation, paid API tests, or push triggers.
GitHub runner billing depends on repository visibility and the account's included minutes;
review that before dispatching it on a private repository.

For a local native run, after committing a clean source tree:

```sh
cargo test --release --locked -- --test-threads=2
cargo build --release --locked
cargo fetch --locked --target x86_64-unknown-linux-gnu
python3 scripts/bundle_licenses.py --target x86_64-unknown-linux-gnu --out /tmp/lastknown-licenses
python3 scripts/package_candidate.py --target x86_64-unknown-linux-gnu --binary target/release/recall --licenses /tmp/lastknown-licenses --out /tmp/lastknown-package
python3 scripts/verify_platform_candidate.py --package-dir /tmp/lastknown-package/lastknown-linux-x64 --home /tmp/lastknown-test-home --results /tmp/lastknown-evidence/result.json
```

Use new output/profile paths each time. Packaging requires Python 3.11+ and the license
collection requires cached Cargo sources. Installing the resulting package requires Python 3.10+.
Keep the per-platform result files with the exact artifact hashes; a prepared workflow is
not evidence that its jobs have run.

An existing repository whose default branch already has `ci.yml` can run these candidates
from a staging branch without merging it: manually dispatch `ci.yml` on that branch with
the boolean `native_candidates=true`. That calls the branch's native workflow and skips
routine CI. A push alone starts neither workflow. This is useful for private review before
creating a public release; private-runner minutes still apply.
