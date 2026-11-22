//! Per-agent / per-team / per-project memory scoping (tenancy v1).
//!
//! Every `documents` row carries one `scope` tag — `global`, `team:<slug>`, `project:<slug>` or
//! `agent:<slug>` — and an `agent_id` (who wrote it; NULL = operator / indexer / pre-v1). A reader
//! is resolved ONCE from the process environment the harness set (`RECALL_AGENT`,
//! `RECALL_PROJECT`, `RECALL_TEAM`) and never from a query, a tool argument or a request body.
//!
//! 🔴 FAIL-CLOSED, the opposite of `TierFilter` (which deliberately lets NULL tiers through).
//! A malformed value, an empty value, or a team/project without an agent is an error: the command
//! exits non-zero with zero hits. There is no fallback to the operator view once ANY of the three
//! variables is present.
//!
//! ⚠️ Not a security boundary: every agent runs as the same OS user and can open `recall.db`
//! directly. For the same reason one statistical side channel is ACCEPTED, not closed: FTS5's
//! `bm25()` computes IDF over the one shared index, so a scoped reader's scores and order shift
//! slightly with documents it cannot see. Closing that needs per-scope FTS indexes; it is not
//! worth it while every reader can open the file and read those documents outright. This keeps
//! cooperating agents' context apart (design:
//! the tenancy design notes).

use anyhow::{bail, Result};
use std::cell::RefCell;

/// The tag every pre-tenancy row, every vault file and every operator save carries.
pub const GLOBAL: &str = "global";

/// Decided 2026-11-06: with no identity variable set the reader is the
/// operator — today's full view — so existing installs behave exactly as before. Flip to `true`
/// to refuse unidentified callers instead.
pub const ENFORCE_IDENTITY: bool = false;

const ENV_AGENT: &str = "RECALL_AGENT";
const ENV_PROJECT: &str = "RECALL_PROJECT";
const ENV_TEAM: &str = "RECALL_TEAM";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reader {
    /// Sees every scope. Emits no SQL clause at all, so the operator's queries are byte-identical
    /// to the pre-tenancy build.
    Operator,
    /// Sees exactly `allowed` (design §11 Q2: global + own team + own project + own agent box).
    Agent { agent: String, allowed: Vec<String> },
}

thread_local! {
    static CURRENT: RefCell<Reader> = const { RefCell::new(Reader::Operator) };
}

/// The reader for this thread. The CLI sets it once at startup ([`init_from_env`]); the daemon
/// will set it per request.
pub fn current() -> Reader {
    CURRENT.with(|c| c.borrow().clone())
}

/// Replace this thread's reader, returning the previous one.
pub fn set(r: Reader) -> Reader {
    CURRENT.with(|c| std::mem::replace(&mut *c.borrow_mut(), r))
}

/// Resolve the reader from the environment and install it for this thread.
pub fn init_from_env() -> Result<Reader> {
    // A PRESENT variable that is not valid Unicode is an error, never "unset" (Codex tenancy b1:
    // `env::var(..).ok()` folded NotUnicode into None and so into the operator view).
    let get = |k: &str| -> Result<Option<String>> {
        match std::env::var_os(k) {
            None => Ok(None),
            Some(v) => v
                .into_string()
                .map(Some)
                .map_err(|_| anyhow::anyhow!("recall: {k} is not valid Unicode: refusing (no operator fallback)")),
        }
    };
    let r = resolve(get(ENV_AGENT)?, get(ENV_PROJECT)?, get(ENV_TEAM)?)?;
    set(r.clone());
    Ok(r)
}

/// Pure resolution, split out so tests never mutate process environment (not thread-safe under the
/// parallel test harness — the same reason `fts_search_with_tiers` exists).
pub fn resolve(agent: Option<String>, project: Option<String>, team: Option<String>) -> Result<Reader> {
    if agent.is_none() && project.is_none() && team.is_none() {
        if ENFORCE_IDENTITY {
            bail!("recall: no agent identity ({ENV_AGENT} unset) and tenancy enforcement is on");
        }
        return Ok(Reader::Operator);
    }
    let Some(agent) = agent else {
        bail!("recall: {ENV_PROJECT}/{ENV_TEAM} set without {ENV_AGENT}: refusing (no operator fallback)");
    };
    validate_slug(ENV_AGENT, &agent)?;
    let mut allowed = vec![GLOBAL.to_string(), format!("agent:{agent}")];
    if let Some(p) = project {
        validate_slug(ENV_PROJECT, &p)?;
        allowed.push(format!("project:{p}"));
    }
    if let Some(t) = team {
        validate_slug(ENV_TEAM, &t)?;
        allowed.push(format!("team:{t}"));
    }
    Ok(Reader::Agent { agent, allowed })
}

/// `^[a-z0-9][a-z0-9._-]{0,63}$` — the shared tenant-id slug rule. Empty is an error,
/// not "unset": a harness that templates an empty value must not silently become the operator.
pub fn validate_slug(what: &str, s: &str) -> Result<()> {
    let ok = !s.is_empty()
        && s.len() <= 64
        && s.as_bytes()[0].is_ascii_alphanumeric()
        && s.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"._-".contains(&b));
    if !ok {
        bail!("recall: {what}={s:?} is not a valid id (lowercase a-z 0-9 . _ -, 1-64 chars, starts alphanumeric)");
    }
    Ok(())
}

/// Is `tag` a well-formed scope tag? Mirrors the `documents_scope_ok_i` trigger.
pub fn valid_tag(tag: &str) -> bool {
    if tag == GLOBAL {
        return true;
    }
    match tag.split_once(':') {
        Some((kind, slug)) => {
            matches!(kind, "team" | "project" | "agent") && validate_slug("scope", slug).is_ok()
        }
        None => false,
    }
}

impl Reader {
    pub fn is_operator(&self) -> bool {
        matches!(self, Reader::Operator)
    }

    pub fn allows(&self, tag: &str) -> bool {
        match self {
            Reader::Operator => true,
            Reader::Agent { allowed, .. } => allowed.iter().any(|a| a == tag),
        }
    }

    /// SQL fragment appended to an existing WHERE, e.g. `" AND d.scope IN ('global','agent:w1')"`,
    /// or `""` for the operator. Only tags that passed [`validate_slug`] are ever interpolated
    /// (the same constant-only argument as `TierFilter::sql_clause`); `valid_tag` is re-checked
    /// here so a hand-built `Reader` can never inject.
    pub fn sql_clause(&self, alias: &str) -> String {
        match self {
            Reader::Operator => String::new(),
            Reader::Agent { allowed, .. } => {
                let quoted: Vec<String> = allowed
                    .iter()
                    .filter(|t| valid_tag(t))
                    .map(|t| format!("'{t}'"))
                    .collect();
                format!(" AND {alias}.scope IN ({})", quoted.join(","))
            }
        }
    }

    /// SQL fragment restricting a RETIRE (forget / supersede) to rows this reader owns: rows it
    /// wrote, or rows in its own agent box. `""` for the operator. Shared rows written by others
    /// are out of reach even though they are readable, and a miss reads as "not found" / "0
    /// retired", never "forbidden", so it reveals nothing.
    pub fn write_clause(&self, alias: &str) -> String {
        match self {
            Reader::Operator => String::new(),
            // `agent` passed validate_slug, so it is safe to interpolate.
            Reader::Agent { agent, .. } => {
                format!(" AND ({alias}.agent_id = '{agent}' OR {alias}.scope = 'agent:{agent}')")
            }
        }
    }

    /// `(agent, project, team)` as the environment named them — what a client sends the daemon,
    /// which re-validates it with [`resolve`]. All `None` for the operator.
    pub fn identity(&self) -> (Option<String>, Option<String>, Option<String>) {
        match self {
            Reader::Operator => (None, None, None),
            Reader::Agent { agent, allowed } => {
                let kind = |k: &str| allowed.iter().find_map(|t| t.strip_prefix(k).map(str::to_string));
                (Some(agent.clone()), kind("project:"), kind("team:"))
            }
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Reader::Operator => "operator (all scopes)".to_string(),
            Reader::Agent { agent, allowed } => format!("agent {agent}: {}", allowed.join(", ")),
        }
    }
}

thread_local! {
    /// Ids the current query's reader may NOT see, for the resident vector channels (set by
    /// [`DeniedGuard`] for the duration of one fused query; `None` = operator or no query running).
    static DENIED: RefCell<Option<std::rc::Rc<std::collections::HashSet<i64>>>> = const { RefCell::new(None) };
}

/// Installs the current reader's denied-id set for one query and clears it on drop.
///
/// Vector channels score one unscoped resident matrix (it serves every reader). Without this they
/// would spend their top-k on rows the per-channel SQL gate then removes, so a scoped reader would
/// get FEWER than k candidates. The SQL gate (`Store::allowed_ids`) stays the authority; this only
/// keeps k full. The set is small: only non-`global` rows can ever be denied.
pub struct DeniedGuard;

impl DeniedGuard {
    pub fn install(denied: Option<std::collections::HashSet<i64>>) -> Self {
        DENIED.with(|d| *d.borrow_mut() = denied.map(std::rc::Rc::new));
        DeniedGuard
    }
}

impl Drop for DeniedGuard {
    fn drop(&mut self) {
        DENIED.with(|d| *d.borrow_mut() = None);
    }
}

/// Is `id` denied to the current query's reader? Always false for the operator.
pub fn is_denied(id: i64) -> bool {
    DENIED.with(|d| d.borrow().as_ref().is_some_and(|s| s.contains(&id)))
}

/// Remove denied docs from a per-doc score map before top-k. A no-op for the operator.
pub fn drop_denied<V>(per_doc: &mut std::collections::HashMap<i64, V>) {
    let denied = DENIED.with(|d| d.borrow().clone());
    if let Some(denied) = denied {
        per_doc.retain(|id, _| !denied.contains(id));
    }
}

/// Decided 2026-11-06: where an agent's `recall save` goes when it names
/// no `--share`. `"project"` = its project box when `RECALL_PROJECT` is set, else its own box.
pub const DEFAULT_SHARE: &str = "project";

/// The scope tag a save goes to, and the `agent_id` it records. `share` is `agent` / `project` /
/// `team` / `global` (None = [`DEFAULT_SHARE`]). Only boxes in the caller's own read set are
/// accepted, so an agent can never write into a box it cannot see. The operator writes `global`
/// with no author, exactly as before.
pub fn save_target(reader: &Reader, share: Option<&str>) -> Result<(String, Option<String>)> {
    match reader {
        Reader::Operator => match share {
            None | Some("global") => Ok((GLOBAL.to_string(), None)),
            Some(s) => bail!("recall: --share {s} needs an agent identity (RECALL_AGENT); the operator saves to global"),
        },
        Reader::Agent { agent, allowed } => {
            let pick = |kind: &str| allowed.iter().find(|t| t.starts_with(&format!("{kind}:"))).cloned();
            let want = share.unwrap_or(DEFAULT_SHARE);
            let tag = match want {
                "global" => Some(GLOBAL.to_string()),
                "agent" => Some(format!("agent:{agent}")),
                // The DEFAULT falls back to the agent's own box when no project is set; an explicit
                // `--share project` without one is an error, never a guess.
                "project" if share.is_none() => pick("project").or_else(|| Some(format!("agent:{agent}"))),
                "project" => pick("project"),
                "team" => pick("team"),
                other => bail!("recall: --share must be agent, project, team or global (got {other:?})"),
            };
            let Some(tag) = tag else {
                bail!("recall: --share {want} needs RECALL_{} to be set", want.to_uppercase());
            };
            Ok((tag, Some(agent.clone())))
        }
    }
}

/// The scope clause for the current thread's reader. Read surfaces append this next to the tier
/// clause; resident vector loads deliberately do NOT (they load every scope once and filter per
/// query, so one daemon can serve many readers).
pub fn clause(alias: &str) -> String {
    current().sql_clause(alias)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &str) -> Option<String> {
        Some(v.to_string())
    }

    #[test]
    fn no_identity_is_operator_and_emits_no_sql() {
        let r = resolve(None, None, None).unwrap();
        assert_eq!(r, Reader::Operator);
        assert_eq!(r.sql_clause("d"), "");
    }

    #[test]
    fn agent_reads_global_own_project_own_team() {
        let r = resolve(s("w1"), s("p"), s("a")).unwrap();
        assert_eq!(r.sql_clause("d"), " AND d.scope IN ('global','agent:w1','project:p','team:a')");
        assert!(r.allows("global") && r.allows("agent:w1"));
        assert!(!r.allows("agent:w2") && !r.allows("project:q") && !r.allows("team:b"));
    }

    #[test]
    fn identity_fails_closed() {
        for (a, p, t) in [
            (s(""), None, None),
            (s("Bad!"), None, None),
            (s("w1"), s("p q"), None),
            (s("w1'"), None, None),
            (None, s("p"), None),
            (None, None, s("a")),
            (s(&"x".repeat(65)), None, None),
            (s("-w"), None, None),
        ] {
            assert!(resolve(a.clone(), p.clone(), t.clone()).is_err(), "{a:?} {p:?} {t:?} must refuse");
        }
    }

    #[test]
    fn tag_grammar() {
        for ok in ["global", "agent:w1", "team:a.b", "project:x-1_2"] {
            assert!(valid_tag(ok), "{ok}");
        }
        for bad in ["", "Global", "agent:", "agent:a:b", "org:a", "agent:A", &format!("team:{}", "x".repeat(65))] {
            assert!(!valid_tag(bad), "{bad}");
        }
    }

    #[test]
    fn save_target_defaults_and_refusals() {
        let op = Reader::Operator;
        assert_eq!(save_target(&op, None).unwrap(), ("global".into(), None));
        assert!(save_target(&op, Some("agent")).is_err());
        let w = resolve(s("w1"), s("p"), None).unwrap();
        assert_eq!(save_target(&w, None).unwrap().0, "project:p", "default = project when set");
        assert_eq!(save_target(&w, Some("agent")).unwrap(), ("agent:w1".into(), Some("w1".into())));
        assert_eq!(save_target(&w, Some("global")).unwrap().0, "global");
        assert!(save_target(&w, Some("team")).is_err(), "no RECALL_TEAM: refuse, never guess");
        assert!(save_target(&w, Some("project:q")).is_err());
        let lone = resolve(s("w1"), None, None).unwrap();
        assert_eq!(save_target(&lone, None).unwrap().0, "agent:w1", "no project: own box");
        assert!(save_target(&lone, Some("project")).is_err());
    }

    #[test]
    fn write_clause_is_ownership_not_visibility() {
        assert_eq!(Reader::Operator.write_clause("d"), "");
        let w = resolve(s("w1"), s("p"), None).unwrap();
        assert_eq!(w.write_clause("d"), " AND (d.agent_id = 'w1' OR d.scope = 'agent:w1')");
    }

    #[test]
    fn set_is_thread_local_and_returns_previous() {
        let prev = set(resolve(s("w9"), None, None).unwrap());
        assert_eq!(prev, Reader::Operator);
        assert!(!current().is_operator());
        std::thread::spawn(|| assert!(current().is_operator())).join().unwrap();
        set(Reader::Operator);
    }
}
