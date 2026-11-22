# Third-party notices

Lastknown's own code is Apache-2.0 (`LICENSE`, `NOTICE`). Third-party components retain their own licenses; Apache-2.0 does not replace those terms.

## Adapted source

Hidden-character handling in `src/scrub.rs` (`is_default_ignorable` and `ESCAPE_SEQUENCES`) was adapted from `crates/ai-memory-core/src/sanitize.rs` in [akitaonrails/ai-memory](https://github.com/akitaonrails/ai-memory). Copyright (c) 2026 Fabio Akita. The full MIT permission and warranty notice is retained in [`licenses/ai-memory/LICENSE`](licenses/ai-memory/LICENSE).

## Rust dependencies

[`licenses/rust/README.md`](licenses/rust/README.md) lists the locked default-feature graph for the target named in `licenses/rust/manifest.json`, including build dependencies. The source checkout retains the Windows x64 inventory; native binary packages replace it with their own target's inventory. The manifest records each crate's declared SPDX expression, registry package checksum, and the SHA-256 of each copied license/notice file. Files come verbatim from Cargo's source cache, except for the two pinned upstream notices described below. Regenerate for a new lockfile, target or feature selection; an inventory does not cover optional-feature or other-platform binary distributions. See `scripts/bundle_licenses.py` and `scripts/package_candidate.py`.

The macOS graph includes `objc2-core-foundation` and `objc2-core-services` 0.3.2, whose crate archives omit license files. Their declared license is `Zlib OR Apache-2.0 OR MIT`; this distribution selects Apache-2.0 for them. [`licenses/overrides/objc2-0.3.2`](licenses/overrides/objc2-0.3.2) retains the complete, verbatim upstream license declaration at their recorded Git revision, including its Apple SDK discussion, and the official full Apache-2.0 text. The collector checks exact crate checksums, repository, version, SPDX expression, VCS revision and notice hashes, then records that origin and selection in the target inventory. Unknown missing-license cases still fail.

## ONNX Runtime

Version 1.24.4, MIT, Copyright (c) Microsoft Corporation. Full `LICENSE` and `ThirdPartyNotices.txt` are retained under [`licenses/onnxruntime`](licenses/onnxruntime). They were checked against the pinned Microsoft NuGet package. Both installers extract only the native libraries for their supported platform, not the managed .NET assembly. No .NET runtime is required by recall.

- Official package: https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/1.24.4
- Package SHA-256: `4b978d5065b85e7004b6c6f60ca494bd978fbe6836cbf0a0b52d82b61ab99638`
- Windows `onnxruntime.dll` SHA-256: `b95efb2113b603bbbf3f191061c5516a871ed546893c820e4f3b7b6c358dbf2a`

The installers also copy upstream notices into `~/.recall/licenses/onnxruntime`. Both use this same pinned NuGet archive. Linux/macOS require their own native validation before advertising a tested binary; see [platform support](docs/PLATFORMS.md).

## Downloaded model (not bundled in the source or binary archive)

BGE-small-en-v1.5, ONNX export at Xenova commit `ea104dacec62c0de699686887e3f920caeb4f3e3`; five file digests are pinned in the installers. The [BAAI model card](https://huggingface.co/BAAI/bge-small-en-v1.5) declares MIT; the [Xenova export](https://huggingface.co/Xenova/bge-small-en-v1.5) identifies the original model. Neither repository supplies a separate LICENSE file in the inspected snapshot. Keep this provenance distinction when redistributing model weights; this release archive does not redistribute them.

## Optional Python tools

The MCP server and ingest helpers install their own Python dependencies from their requirement files. Those packages are not bundled in the release archive; their package licenses continue to apply. Optional paid-provider features and benchmark datasets are not enabled or bundled by the default build.
