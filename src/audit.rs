// audit.rs — mistake-graveyard append helper
// Every error path in recall calls audit::log_error() before exiting.

use std::fs::OpenOptions;
use std::io::Write;

/// Error log path: `RECALL_ERROR_LOG`, else `~/.recall/error-log.jsonl`.
fn graveyard_path() -> String {
    if let Ok(p) = std::env::var("RECALL_ERROR_LOG") {
        return p;
    }
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default();
    std::path::Path::new(&home)
        .join(".recall")
        .join("error-log.jsonl")
        .to_string_lossy()
        .into_owned()
}

#[derive(serde::Serialize)]
struct GraveyardEntry<'a> {
    ts: String,
    cli: &'a str,
    context: &'a str,
    error: &'a str,
}

/// Append an error entry to the error log.
/// Non-fatal — if the graveyard file cannot be written, prints a warning only.
pub fn log_error(context: &str, error: &str) {
    let entry = GraveyardEntry {
        ts: chrono_now(),
        cli: "recall",
        context,
        error,
    };

    let path = graveyard_path();
    let line = match serde_json::to_string(&entry) {
        Ok(l) => l,
        Err(_) => return,
    };

    // Ensure directory exists
    if let Some(parent) = std::path::Path::new(&path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    if let Ok(mut f) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = writeln!(f, "{}", line);
    } else {
        eprintln!("recall: warning: audit: could not append to {}", path);
    }
}

fn chrono_now() -> String {
    // Simple ISO-8601 without pulling in chrono dep by default.
    // If chrono is in Cargo.toml, replace this with chrono::Utc::now().to_rfc3339()
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("unix:{}", secs)
}
