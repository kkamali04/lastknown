# Try Lastknown before using your own notes

Download and extract the entire archive for your platform from
[GitHub Releases](https://github.com/kkamali04/lastknown/releases). The binary is still named
`recall`. This is an alpha: read the known limits in README.md and the platform matrix.

## Two-minute freshness demo

This needs ripgrep (`rg`) on PATH, but no Rust, Python, model download, or existing Recall
profile. It makes a temporary notes folder, searches a note saying port 8080, changes that
note to 9090 without indexing, checks the stale warning, then reindexes and checks 9090.
It removes only its own temporary fixture. Run from the extracted package directory.

Windows, in a **new PowerShell process**:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\examples\stale-demo\run.ps1 -Bin .\prebuilt\release\recall.exe
```

Linux/macOS:

```sh
sh examples/stale-demo/run.sh "$PWD/prebuilt/release/recall"
```

Success ends with `stale-demo: PASS`. This tests keyword retrieval and freshness, not BGE.
Typical query output is below; paths and timestamps vary:

```text
recall query: top 1 for: API port

  1. API notes
     api-notes.md
     …The [API] [port] is 8080  Questions go to the platform channel.
```

After changing the passage without reindexing, the stored excerpt still contains 8080 and
the result says `!! STALE: this passage changed` and `read the actual file`. After indexing,
the result shows 9090 without that warning. Terminal color is automatic; piped output is plain.

## Full prebuilt acceptance check

This downloads the pinned runtime/model and MCP dependencies, then tests actual BGE search,
indexing, status, all MCP tools, redaction, and local UI/daemon requests. It uses synthetic
notes and a new test profile. Your normal installation and agent registrations are not changed.
Internet, Python 3.10+ with venv, and working ripgrep are required; Rust is unnecessary.

Windows (from the extracted package):

```powershell
py scripts/verify_platform_candidate.py --package-dir "$PWD" --home "$env:TEMP\lastknown-alpha-test" --results "$env:TEMP\lastknown-alpha-evidence\result.json"
```

Linux/macOS:

```sh
python3 scripts/verify_platform_candidate.py --package-dir "$PWD" --home "$HOME/lastknown-alpha-test" --results "$HOME/lastknown-alpha-evidence/result.json"
```

The test HOME must not already exist. Choose a different name for another run; never point
it at your normal HOME. Success ends with `NATIVE PACKAGE ACCEPTANCE PASSED`, and result.json
records passing steps, installed executable/runtime hashes, and `package_unchanged: true`.
Step logs sit beside the result file. This check uses `ui --no-open`; open the UI interactively
after installation to check your normal browser too. Autostart and agent registration need
separate manual checks if you choose to enable those options.

## Use your notes

After the temporary tests pass, run the package's Install-Lastknown script with your chosen
notes folder as described in docs/PLATFORMS.md. Installation writes to your normal `.recall`
profile. Then run `recall index`, `recall embed`, and `recall query -- "your question"`.
First load the new installation's environment in that terminal so an older PATH shim cannot
take precedence: Windows `. "$env:USERPROFILE\.recall\env.ps1"`; Unix `. "$HOME/.recall/env.sh"`.
Run `recall ui` for the local browser interface. A query returns passages and their source
files; it does not generate an LLM answer. Always read the current file when it reports stale.

## Repeat a benchmark

Start with the public SciFact protocol in bench/README.md. Use a fresh BENCH_WORK, the exact
release binary, and the locked dataset. Run all 300 questions; keep failed questions in the
denominator and compare the same split. The November 13 published TEST-half row uses 146
questions; it must not be compared with a whole-set score. Corpus downloads and benchmark
stores stay outside this repository. Results measure retrieval, not general agent productivity.
