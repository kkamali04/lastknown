# AI summary layer (optional, default OFF)

`ingest/summarize.py` writes an extractive summary beside each memcap raw note:
`<note>.summary.v1.<gen>.md`. Every bullet is a verbatim quote of the note plus its offset in the
sanitised text the model saw. Quotes not found verbatim are dropped and counted
(`dropped_unsupported` in the frontmatter). recall.db is never written; the indexer picks files up.

## What leaves the machine
Turning it on sends scrubbed session text to the model provider (Anthropic for `claude`, OpenAI for
`codex`) under the user's own subscription login. Layers, in order: memcap's existing scrub
(`scrub_turns.scrub_text`), the configured redaction list, a prompt-injection pre-scan (own patterns,
plus `skillspector scan --no-llm` when it completes with no degraded analyzer; HIGH/CRITICAL lines are
redacted and logged, more than 20 flagged lines skips the note), then an output scan (scrub + canaries).
All of this is best-effort, not a guarantee. Provider retention follows the provider's terms for the
subscription; local artifacts are the queue DB and the summary files (delete them to clean up).

## Accepted residual risk
No OS sandbox on Windows. Isolation is: fresh session, empty temp cwd, all tools denied for claude
(`--tools ""`), `codex exec -s read-only --ephemeral`, no MCP, allowlisted child env. Codex has no
flag that removes its shell tool, so for codex the read-only sandbox is the boundary.

## Turn on
```
setx RECALL_SUMMARIES claude          # off|claude|codex, default off
py ingest/summarize.py list-models
py ingest/summarize.py reconcile      # enqueue raw notes with no summary
py ingest/summarize.py run --limit 20
```
Settings: `~/.recall/summaries.json` and env `RECALL_SUMMARIES`, `RECALL_SUMMARY_MODEL`,
`RECALL_SUMMARY_DAILY_CAP` (200), `RECALL_SUMMARY_MAX_INPUT_BYTES` (60000), `RECALL_SUMMARY_REDACT`
(`;`-separated regexes), `RECALL_SUMMARY_PAUSE_FILE` (default `~/.recall/summaries.pause`; create the
file to stop new attempts at once). `canaries` (strings that must never appear in output) is a config
key. It refuses to run if `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or similar is in the environment, or
if `claude auth status` / `codex login status` does not show subscription login.
