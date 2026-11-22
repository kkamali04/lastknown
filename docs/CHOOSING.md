# Choosing Lastknown

Lastknown is a local search and source-drift tool for an existing Markdown notes folder.
It is useful when the problem is finding a prior decision or instruction and noticing
that the underlying file has changed before an agent relies on the old excerpt.

Its implementation combines standard hybrid retrieval with file/passage freshness checks,
incremental index/embedding maintenance, pattern-based redaction, a warm local daemon,
read-only MCP and a local browser interface. Those features make a usable workflow;
they are not evidence of a new memory algorithm or universal retrieval superiority.

## Match the tool to the job

| Your main need | A relevant option | Why consider it |
|---|---|---|
| Search one existing Markdown notes folder and see when source text changed | **Lastknown** | Local SQLite + BGE, source paths, file/passage drift, CLI/MCP/browser access |
| Search multiple document collections with local query expansion and LLM reranking | [QMD](https://github.com/tobi/qmd) | Its README documents collections, contextual metadata, hybrid search and a local LLM reranking pipeline |
| Extract and manage user/application memories, with SDKs or managed infrastructure | [Mem0](https://github.com/mem0ai/mem0) | Application-memory APIs and self-hosted/managed options; exact features and published results depend on deployment |
| Build time-aware entity and relationship graphs into an application | [Graphiti](https://github.com/getzep/graphiti) | Temporal graph framework with provenance; bring the database, model and surrounding application |
| Anchor coding memories/decisions to code and detect structural drift | [OpenLore](https://github.com/clay-good/OpenLore) | Code-oriented memory and guardrails, with source-linked freshness and structural analysis |
| Check memory evidence against changes to source files | [memwarden](https://github.com/saiyam1814/memwarden) | Source-linked memory checks and trust states for coding agents |

These are intended-workflow comparisons, based on the projects' own documentation checked
Nov 22, 2026. They are not scored head-to-head evaluations, exhaustive feature matrices or
claims that another project lacks a feature its README does not mention. In particular,
local operation, hybrid search and stale-memory detection are not unique to Lastknown.

## Where Lastknown fits

- Your notes already contain the information you want agents to retrieve.
- You want the source path and a visible warning when the indexed excerpt's lines changed.
- You want default search and embeddings to run locally without a provider API key or LLM.
- You prefer a packaged CLI with MCP/browser access over operating a graph service.

## Where another approach may fit better

- You need several notes roots or rich collection management: Lastknown currently indexes one.
- You need automatic fact extraction from every conversation: that is not its default model.
- You need temporal entity reasoning, a shared managed service or a production memory API.
- You need demonstrated coding-task improvement or a reliable “no answer” classifier:
  Lastknown's evidence does not establish either.

The [evaluation inventory](EVALUATION.md) includes the broader historical tests and their
limits. The small agent-memory pilot did not establish an overall winner. Choose on the
workflow you can verify with your data, not comparisons between incompatible public scores.

## Architecture in brief

Your Markdown/QMD files → local keyword index + BGE chunk vectors → fused ranked passages
→ source-file/passage freshness check → CLI, read-only MCP or loopback browser UI.

Indexing/search use your files as inputs. Saved observations live in the local store.
New profiles redact known patterns before storage by default; existing profiles retain their
prior setting. The scrubber is not a security boundary. The optional Haiku judge is off by
default and can send text through your Claude CLI if enabled. See [redaction](SCRUB.md),
[scope](TENANCY.md) and [known limits](LIMITS.md).
