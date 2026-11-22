# Memory boxes: per agent, team and project

recall can keep several agents' memories apart in one `recall.db`. Every memory lives in one
**box**, and every agent sees only some boxes.

| Box | Written by | Who can read it |
|---|---|---|
| `global` | you (no agent set), and every indexed vault file | everyone |
| `team:<name>` | an agent with `--share team` | agents in that team |
| `project:<name>` | an agent's default save | agents in that project |
| `agent:<name>` | an agent with `--share agent` | that agent only |

## Who you are

recall reads three environment variables, normally set in an agent's harness or MCP config:

```
RECALL_AGENT=reviewer      # required for any scoped use
RECALL_PROJECT=billing     # optional
RECALL_TEAM=platform       # optional
```

- **None set:** you are the operator and see every box. This is exactly how recall worked before
  boxes existed, so existing setups need no change.
- An agent reads `global`, its team, its project and its own box. It never reads another agent's
  box, or another team's or project's.
- A bad value fails instead of falling back. Values must be lowercase `a-z 0-9 . _ -` and 1-64
  characters. Setting a project or team without an agent also fails. Either way the command exits
  1 with no results.
- `recall whoami` prints who you are and which boxes you can read, e.g.
  `agent reviewer: global, agent:reviewer, project:billing, team:platform`, or `operator (all scopes)`.

## Saving

```
recall save "text"                    # agent: into its project (its own box if no project)
recall save --share agent "text"      # agent: private to this agent
recall save --share project "text"    # agent: its project (needs RECALL_PROJECT)
recall save --share team "text"       # agent: its team (needs RECALL_TEAM)
recall save --share global "text"     # agent: everyone
```

- With no agent set, `recall save` goes to `global`, as before. `--share` other than `global`
  needs an agent.
- A memory's box cannot be changed later. To move it, save it again in the new box and retire the
  old copy.

## What an agent may run

`query`, `save`, `recent`, `export`, `forget`, `supersede`, `status`, `whoami`.

Every other command reads or changes the whole store (`index`, `refresh`, `embed`, `gc`,
`doctor`, `dream`, `insights`, `bench`, `eval`, `daemon`, `settings`, ...). Those exit 1 with "operator-only" when an
agent is set.

- `forget` and `supersede` run as an agent retire only memories that agent wrote, or that sit in
  its own box. Anything else reads as "0 retired" / "not found" (exit code 0, the same as an id
  that does not exist, so nothing is revealed).
- `status` run as an agent shows only that agent's view and how many memories it can see.

## MCP

The recall MCP server (`mcp/recall_mcp.py`) inherits `RECALL_AGENT` / `RECALL_PROJECT` /
`RECALL_TEAM` from the environment it is launched with. Set them in the MCP client's config for
that agent. No tool takes an agent or box argument. `recall_whoami` reports the identity.

## Limits

- **Not a security wall.** Every agent runs as the same OS user and could open `recall.db`
  directly. Boxes keep cooperating agents' context apart; they do not stop a hostile program.
- **Old copies of the `recall` binary ignore these variables** and answer with the full view. Replace
  every copy before you rely on boxes.
- The warm daemon must be started with no agent set. It applies each caller's identity per
  request.
