# Contributing (lite)

- Build: `cargo build --release` (default features only: free, offline channels `fts5` + `bge`).
- Test: `cargo test --release`; Python: `python -m pip install -r mcp/requirements.txt pytest && python -m pytest ingest integrations/claude-code/test_hooks.py mcp/test_recall_mcp.py`.
- Keep the **safe-invocation contract** for every caller that spawns `recall`: flags first, then `--`,
  then exactly one positional (`recall query --top 8 --json -- "<text>"`). A user string that equals a
  flag name must never be parsed as that flag.
- Never add a default that needs a paid API or a personal path. Paid channels (`voyage`, `gemini`)
  stay opt-in Cargo features.
- A search that "works" must be proven by `source` in the JSON (`fts5+bge`), not by exit code 0: the
  keyword tier can answer for a dead semantic channel.
- No secrets in the repo. CI runs gitleaks; test fixtures must be obviously synthetic.
- Small commits, plain messages that say what and why.

## CI

GitHub Actions minutes are scarce, so routine CI runs on our own machines. The workflow
(`.github/workflows/ci.yml`) only runs on a manual `workflow_dispatch` or a PR labelled `ci`; it has no push trigger.

- Windows: `powershell -ExecutionPolicy Bypass -File scripts\ci-local.ps1` (cargo test, pytest for ingest + mcp + installer, `scrub_e2e`, and the README install + first-use smoke on a temp home).
- Linux: `git -c core.autocrlf=false archive HEAD | ssh HOST 'rm -rf ~/ci/recall; mkdir -p ~/ci/recall && tar -x -C ~/ci/recall'`, then `ssh HOST 'cd ~/ci/recall && sh scripts/ci-local.sh'`. It installs ripgrep into `~/.local/bin` if missing.

Both print PASS/FAIL per step and exit 1 on any failure; `-Only test,pytest` / `CI_ONLY="test pytest"` run a subset.
They never touch `~/.recall` (the smoke and `scrub_e2e` use a temp home). `release_check` is not part of them.
