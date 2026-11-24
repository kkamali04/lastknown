# Choosing Lastknown

Lastknown is a local search and source-drift tool for an existing Markdown notes folder.
It is useful when the problem is finding a prior decision or instruction and noticing
that the underlying file has changed before an agent relies on the old excerpt.

Its distinguishing choice is to check the retrieved excerpt against its current source
file: an edit elsewhere can leave the passage current; changed excerpt lines get a stale
warning. This sits alongside a local CLI, five read-only MCP tools, settings UI, incremental
index/embedding maintenance, a warm daemon and optional raw Claude Code capture.

That combination is the product's focus. FTS5, BGE, fusion, local operation, skills,
hooks and verbatim capture are established ideas, and source-drift tools exist elsewhere.
The choice is about the workflow and the evidence you need, not a universal algorithmic moat.

## Match the tool to the job

| Your main need | A relevant option | Why consider it |
|---|---|---|
| Search one existing Markdown notes folder and see when source text changed | **Lastknown** | Local SQLite + BGE, source paths, file/passage drift, CLI/MCP/browser access |
| Organize verbatim conversation history into scoped wings/rooms/drawers | [MemPalace](https://github.com/MemPalace/mempalace) | Verbatim mining, pluggable retrieval backends and agent-guided setup; Lastknown centers the existing notes folder and current source passage |
| Automatically generate observations and inject compact context across sessions | [Claude-mem](https://github.com/thedotmack/claude-mem) | Observer-driven capture/compression and broad host integrations; Lastknown's default search/capture needs no LLM summarizer |
| Maintain a broader personal/company brain across agents | [gbrain](https://github.com/garrytan/gbrain) | Shared knowledge workflows and agent setup/hooks; Lastknown currently supports one local notes root rather than a hosted/team service |
| Search multiple document collections with local query expansion and LLM reranking | [QMD](https://github.com/tobi/qmd) | Its README documents collections, contextual metadata, hybrid search and a local LLM reranking pipeline |
| Extract and manage user/application memories, with SDKs or managed infrastructure | [Mem0](https://github.com/mem0ai/mem0) | Application-memory APIs and self-hosted/managed options; exact features and published results depend on deployment |
| Build time-aware entity and relationship graphs into an application | [Graphiti](https://github.com/getzep/graphiti) | Temporal graph framework with provenance; bring the database, model and surrounding application |
| Anchor coding memories/decisions to code and detect structural drift | [OpenLore](https://github.com/clay-good/OpenLore) | Code-oriented memory and guardrails, with source-linked freshness and structural analysis |
| Check memory evidence against changes to source files | [memwarden](https://github.com/saiyam1814/memwarden) | Source-linked memory checks and trust states for coding agents |

These are intended-workflow comparisons, based on the projects' own documentation checked
Nov 24, 2026. They are not scored head-to-head evaluations, exhaustive feature matrices or
claims that another project lacks a feature its README does not mention. In particular,
local operation, hybrid search and stale-memory detection are not unique to Lastknown.

## Where Lastknown fits

- Your notes already contain the information you want agents to retrieve.
- You want the source path and a visible warning when the indexed excerpt's lines changed.
- You want default search and embeddings to run locally without a provider API key or LLM.
- You prefer a packaged CLI with MCP/browser access over operating a graph service.

## Where another approach may fit better

- You need several notes roots or rich collection management: Lastknown currently indexes one.
- You need automatic fact extraction from every conversation: Lastknown's opt-in Claude
  capture preserves text; it does not turn every session into verified extracted facts.
- You need temporal entity reasoning, a shared managed service or a production memory API.
- You need demonstrated coding-task improvement or a reliable “no answer” classifier:
  Lastknown's evidence does not establish either.

## What was compared experimentally

[Historical retrieval trials](EVALUATION.md#historical-retrieval-comparisons) tested
MemPalace 3.12.0 and gbrain 0.60.574.0 on common LME/LoCoMo/BEIR protocols. Recall led the
tested MemPalace setup; gbrain led the 60-question LME subset by one question while Recall
had higher point scores on LoCoMo and two BEIR corpora. These qualified, historical
configurations are separate from the workflow comparisons above.

A [private Claude-mem study](EVALUATION.md#private-capture-comparison-claude-mem) used only
20 headless reviews and unequal retrieval backends. The separate
[coding-agent pilot](EVALUATION.md#coding-agent-memory-comparison) did not establish an
overall winner against memory-off, mem0-noinfer, SimpleMem or MemoryOS.
MemPalace's newer raw/hybrid vendor figures use other modes/denominators; see the
[protocol comparison](EVALUATION.md#relating-these-trials-to-mempalaces-published-results).
QMD had a source/feature comparison, not a completed benchmark. There are no measured
Graphiti/OpenLore/memwarden accuracy comparisons here.

Choose on the workflow you can verify with your data and comparable evidence. None of
these trials establishes a universal winner or current-product leaderboard.

## Architecture in brief

Your Markdown/QMD files → local keyword index + BGE chunk vectors → fused ranked passages
→ source-file/passage freshness check → CLI, read-only MCP or loopback browser UI.

Opt-in Claude Code hooks → scrubbed dated notes → explicit indexing/readback → later
embedding for semantic coverage. The [skill and setup guide](../INSTALL_FOR_AGENTS.md)
teach the agent this workflow; MCP itself cannot save, index or embed.

Indexing/search use your files as inputs. Saved observations live in the local store.
New profiles redact known patterns before storage by default; existing profiles retain their
prior setting. The scrubber is not a security boundary. The optional Haiku judge is off by
default and can send text through your Claude CLI if enabled. See [redaction](SCRUB.md),
[scope](TENANCY.md) and [known limits](LIMITS.md).
