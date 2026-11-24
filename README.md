![lastknown — Local memory that checks its sources.](docs/images/lastknown-hero.png)

<p align="center">
  <a href="INSTALL_FOR_AGENTS.md">Set up with an agent</a> ·
  <a href="docs/SETUP.md">Install yourself</a> ·
  <a href="docs/EVALUATION.md">Benchmarks</a> ·
  <a href="LICENSE">Apache 2.0</a>
</p>

Lastknown gives you and your agents a searchable memory of your Markdown notes and saved
decisions. Search by words or meaning, get the source passage, and see whether its lines
changed since indexing. Use the same local index through the **CLI, MCP or browser UI**.

## Why Lastknown?

- **Check the source before relying on memory.** Distinguish a changed passage from a
  file edited elsewhere. Deleted sources and unchecked freshness are visible too.
- **Keep your notes as the source of truth.** Index existing `.md`/`.qmd` files, save
  observations from the CLI, or opt into Claude Code session capture. No mandatory LLM
  summarization step.
- **Run retrieval locally.** SQLite FTS5 + BGE-small through ONNX, with incremental
  indexing/embedding, pattern-based redaction and an optional warm daemon. No provider key
  or graph service for default search; offline after initial setup downloads.
- **Connect the whole workflow.** Five read-only MCP tools, a reusable agent skill,
  optional Claude Code hooks and a settings UI. Writes remain explicit CLI/capture actions.

**The distinguishing focus is source-file and passage drift**, combined with a notes-first
local workflow. MemPalace also preserves verbatim history; Claude-mem emphasizes generated
observations; gbrain connects a broader personal/team brain; QMD offers multi-collection
search with local LLM reranking. These overlap with Lastknown: local operation and hybrid
search are established approaches. [Compare the workflows and tested results →](docs/CHOOSING.md)

## Get started

Give your coding agent this prompt:

> Set up https://github.com/kkamali04/lastknown for me. Read INSTALL_FOR_AGENTS.md and the
> bundled Lastknown skill. Walk me through the notes folder, client connection, settings
> and optional capture hooks, then verify search and MCP. Use a prebuilt release.

The repo includes [the skill](skills/lastknown/SKILL.md), [setup walkthrough](docs/SETUP.md),
[MCP instructions](docs/MCP.md) and [hook code/configuration](docs/HOOKS.md). Reading the
skill does not install or enable anything by itself.

### Manual installation

Download and extract **the entire archive**. Requires Python 3.10+ with venv, ripgrep and
internet for pinned runtime/model and MCP dependencies. Rust is unnecessary.

| Windows x64 | Linux x64 | Linux ARM64 | macOS Apple Silicon |
|---|---|---|---|
| [ZIP](https://github.com/kkamali04/lastknown/releases/download/v0.1.0-alpha.1/lastknown-windows-x64.zip) | [tar.gz](https://github.com/kkamali04/lastknown/releases/download/v0.1.0-alpha.1/lastknown-linux-x64.tar.gz) | [tar.gz](https://github.com/kkamali04/lastknown/releases/download/v0.1.0-alpha.1/lastknown-linux-arm64.tar.gz) | [tar.gz](https://github.com/kkamali04/lastknown/releases/download/v0.1.0-alpha.1/lastknown-macos-arm64.tar.gz) |

```powershell
# Windows — from the extracted package
powershell -ExecutionPolicy Bypass -File .\Install-Lastknown.ps1 -Vault "$HOME\notes"
. "$HOME\.recall\env.ps1"
```

```sh
# Linux / macOS — from the extracted package
sh ./Install-Lastknown.sh --vault "$HOME/notes"
. "$HOME/.recall/env.sh"
```

Then:

```sh
recall index
recall embed
recall query --top 5 -- "why did we choose this cache?"
recall save -- "We chose an LRU cache for bounded memory use."
recall embed
recall refresh
recall ui
```

The executable remains named `recall`. [Try the isolated freshness demo first →](docs/TRY-IT.md)
All four packages passed [native acceptance](https://github.com/kkamali04/lastknown/actions/runs/45003961345).
[Platform requirements and checksums →](docs/PLATFORMS.md)

### Connect an agent

| Component | What it provides | Setup |
|---|---|---|
| MCP: Claude Code, Codex, Cursor, Claude Desktop and other stdio clients | Search, semantic search, recent memories, identity and health | [MCP](docs/MCP.md) |
| Lastknown skill | Search-before-answer, source checks, explicit save/refresh workflow | [Bundled skill](skills/lastknown/SKILL.md) |
| Claude Code hooks, opt-in | Startup availability context; session-end capture into private notes | [Hooks](docs/HOOKS.md) |
| CLI / local browser UI | Save, index, embed, refresh, settings and diagnostics | [CLI](docs/CLI.md) / [settings walkthrough](docs/SETUP.md#choose-your-settings) |

The **new skill/hooks live in this repo**; the existing alpha archives contain the earlier
runtime. Keep a checkout for these integrations. Capture is off until configured, supports
Claude Code transcripts only, and needs a separate embedding pass for semantic retrieval.

## What the tests show

Historical Nov 5–6 **retrieval comparisons**, rescored from saved outputs Nov 23.
These are the tested versions/configurations, not current-product rankings or answer accuracy.

| Same-protocol test | n | Recall | gbrain local | MemPalace full / adapter |
|---|---:|---:|---:|---:|
| LongMemEval_S full-turn, all evidence sessions in top 5 | 470 | **0.887** | Not run | 0.774 |
| Matched LongMemEval subset, same metric | 60 | 0.883 | **0.900** | 0.750 |
| LoCoMo evidence-bearing questions, session recall@5 | 1,982 | **0.850** | 0.766 | 0.454 |
| SciFact full-set nDCG@10 | 300 | **0.734** | 0.698 | 0.650 |
| NFCorpus full-set nDCG@10 | 323 | **0.362** | 0.350 | 0.314 |

That is **417/470 versus 364/470** on strict LongMemEval evidence coverage; gbrain led the
60-question subset by one question. MemPalace used 3.12.0/MiniLM/Chroma; gbrain used
0.60.574.0/local nomic/PGLite with reranking off. Recall build labels had provenance limits.
**MemPalace's headline raw/hybrid configurations were not tested here.**

Other studies include the official 419-question LongMemEval protocol (**386/419**), five
BEIR sets, LoCoMo TEST splits, BEAM 100K/500K, a private Claude-mem capture comparison and
summary/recency/reranker controls. The coding-agent pilot found **7/19 with Recall and
7/19 with memory off**. Only SciFact was rerun for the alpha: all **900 query/channel
ranked lists** matched the Nov 13 baseline, including a replay with the released Linux binary.
[Full results, negative findings, configurations and reproducibility limits →](docs/EVALUATION.md)

## Status

**Public alpha.** One notes folder; no hosted service. Freshness checks source text, not
factual truth, and changes confined to redacted values can be missed. The optional cloud
answerability judge stays off. [Known limits](docs/LIMITS.md) · [Privacy](docs/SCRUB.md) ·
[Source build](docs/CLI.md#build-and-install-from-source) · [Contributing](CONTRIBUTING.md)

[Apache 2.0](LICENSE) for Lastknown's source; dependencies retain their
[third-party licenses](THIRD_PARTY_NOTICES.md).
