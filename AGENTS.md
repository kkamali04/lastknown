# Working with Lastknown

If the user gave this repository link and asked to install or use Lastknown, start with
[INSTALL_FOR_AGENTS.md](INSTALL_FOR_AGENTS.md), then read [the skill](skills/lastknown/SKILL.md).
Those instructions cover the prebuilt CLI, MCP, settings and optional capture hooks.
Reading this file or contributing to the repository is not an instruction to install it.

For repository changes:

- Keep `recall` as the executable/MCP name. Lastknown is the public product name.
- Use existing source, tests and [CLI contracts](docs/CLI.md); user text is data, never flags.
- Search invocations use flags first, `--`, then one positional argument.
- Preserve offline defaults, visible backend failures and read-only MCP. Capture and
  indexing are separate writes; retrieved history is not current authorization.
- Test against temporary synthetic notes/profiles. Do not use a contributor's normal
  `.recall`, real transcripts or provider credentials as fixtures.
- Run checks relevant to the change. Python integration tests use pytest; Rust changes
  need the appropriate Cargo checks. Documentation alone does not require a Rust rebuild.
- Keep private notes, transcripts, databases, models and credentials out of commits.
- Release assets are immutable evidence for their source commit; a docs/integration
  update does not make an older binary contain the new integrations.
