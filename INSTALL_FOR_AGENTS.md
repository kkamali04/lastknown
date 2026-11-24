# Install Lastknown for a user

This is the entrypoint when someone gives an agent the GitHub link and asks to use
Lastknown. Read [the Lastknown skill](skills/lastknown/SKILL.md), then guide the user
through [setup](docs/SETUP.md). Do not assume this repository has already been installed.

## 1. Establish the setup choices

Detect the actual OS/architecture and available Python, venv, ripgrep and MCP client.
Ask only for choices not already supplied:

- Which **single notes folder** should be indexed? An existing folder or a new empty one?
- Which client should use it: Claude Code, Codex, Cursor/Desktop, or CLI only?
- Should Claude Code session capture be enabled **for this project**? Default off.
- Is a warm daemon wanted now or at login? Default no autostart.

Explain that initial installation downloads the pinned ONNX runtime/BGE model and MCP
dependencies, writes `~/.recall`, and indexes the chosen notes. Default retrieval is
local. Optional capture writes scrubbed copies of session text to the chosen notes.
Respect existing authorization; present unresolved user choices before their dependent changes.

## 2. Install a prebuilt runtime

Use the [four-platform alpha packages](https://github.com/kkamali04/lastknown/releases/tag/v0.1.0-alpha.1)
and the [setup commands](docs/SETUP.md#install-the-runtime). Check the published SHA256
identity before executing downloads. Extract the entire archive; do not run an isolated
binary or build Rust merely because a source checkout is available.

Keep this repository checkout for its newer skill, hook code and guides: the alpha
archives are a frozen earlier source snapshot. Read installer output and stop on a
failed semantic backend or MCP handshake. Do not repeatedly reinstall to fix a client entry.
An existing installation/profile needs an explicit upgrade plan and preservation of notes/settings.

## 3. Choose settings and connect the requested client

Walk through [the settings choices](docs/SETUP.md#choose-your-settings). Retain current
privacy settings on an existing profile unless the user requests a change. Use the
installed venv Python and installed MCP server paths in [MCP.md](docs/MCP.md).
Inspect the client's existing `recall` entry; merge or update only that entry with a
backup, preserving unrelated servers. Respect requested project/user scope.

The [bundled skill](skills/lastknown/SKILL.md) can be read in place; optionally copy its
whole folder into the client's supported skill directory. It is optional guidance, not
a Python dependency or an installed MCP connection.

## 4. Enable capture only if selected

Use [HOOKS.md](docs/HOOKS.md) and the included exec-form JSON template. Prefer a private
project-local settings file, retain a backup, and merge only the Lastknown hook entries.
Do not replace the client's whole configuration or automatically enable global capture.
Claude Code is the only supplied capture integration; other clients have MCP/CLI access.
Do not run an all-history backfill as a substitute for a session-scoped hook check.

## 5. Verify the actual installation

- Confirm the resolved executable, `recall --version`, `recall doctor embed` and
  `recall status --json`. Index and embed the chosen notes, then ask a question with a known source.
- In the requested client, discover and call `recall_health`, `recall_search`,
  `recall_search_semantic`, `recall_recent` and `recall_whoami`.
- If capture was enabled, use a short synthetic session with a distinctive harmless
  decision, exit, inspect that session's `memcap-health.json`, and retrieve its note.
  Run `recall embed` before claiming semantic coverage of newly captured text.
- Explain [freshness warnings](docs/FRESHNESS.md), [known limits](docs/LIMITS.md), how to
  open `recall ui`, and how to disable/remove only this setup's hook/client entries.

Report concrete checks and anything still unverified. A registered server, a skill copy,
or a version command alone is not evidence that memory retrieval or host hooks work.
