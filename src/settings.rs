// settings.rs — durable user settings in `~/.recall/settings.json`.
//
// Read on every query and every refresh tick, so it follows the `exclude.txt` rule: loading never
// creates `~/.recall`, a missing file means defaults, and a malformed file degrades to defaults
// with a warning rather than failing a query. `recall settings set` is the only writer, and it
// validates before it writes, so a typo cannot park a bad value in the file.
//
// Keys (all optional):
//   refresh          "nightly" (default) | "on-change" | "every N minutes" | "every N hours"
//   freshness_check  "lazy" (default) | "off"   — flag hits whose file differs from the index
//   stale_penalty    rank positions a stale hit is demoted by. 0 = OFF (default, unmeasured)
//   recency_boost    rank positions the newest file can gain. 0 = OFF (default, unmeasured)
//   nightly_at_utc   "HH:MM" UTC time `nightly` fires (default "07:00" = 02:00-03:00 US Eastern)
//   scrub.secrets / scrub.pii / scrub.store_redacted   on | off   (nested `scrub` object; see scrub.rs)
//   scrub.custom     JSON list of terms / `re:<regex>`, or one term; "" clears
//   vault            the notes folder recall indexes (one folder; RECALL_VAULT wins if set); "" clears
//   closest_labels   on | off (default off): the per-hit "closest match" label and "No confident match" banner

use anyhow::{bail, Context, Result};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RefreshMode {
    OnChange,
    /// Run every `n` minutes.
    Every(u64),
    Nightly,
}

impl RefreshMode {
    /// `on-change` | `nightly` | `every N minute(s)|hour(s)` | `every Nm` | `every Nh`.
    pub fn parse(s: &str) -> Result<Self> {
        let t = s.trim().to_lowercase();
        match t.as_str() {
            "on-change" | "onchange" | "on_change" => return Ok(Self::OnChange),
            "nightly" => return Ok(Self::Nightly),
            _ => {}
        }
        let rest = t.strip_prefix("every").map(str::trim).unwrap_or(&t);
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        let unit = rest[digits.len()..].trim();
        let n: u64 = digits
            .parse()
            .with_context(|| format!("refresh '{s}': expected on-change, nightly or 'every N minutes|hours'"))?;
        let minutes = match unit {
            "m" | "min" | "mins" | "minute" | "minutes" => n,
            "h" | "hr" | "hrs" | "hour" | "hours" => n.saturating_mul(60),
            _ => bail!("refresh '{s}': unit must be minutes or hours"),
        };
        // Floor: a refresh scans the whole vault, so a shorter period would be a busy loop.
        if !(5..=7 * 24 * 60).contains(&minutes) {
            bail!("refresh '{s}': interval must be between 5 minutes and 7 days");
        }
        Ok(Self::Every(minutes))
    }

    pub fn label(&self) -> String {
        match self {
            Self::OnChange => "on-change".into(),
            Self::Nightly => "nightly".into(),
            Self::Every(m) if m % 60 == 0 => format!("every {} hours", m / 60),
            Self::Every(m) => format!("every {m} minutes"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    pub refresh: RefreshMode,
    pub freshness_check: bool,
    pub stale_penalty: f64,
    pub recency_boost: f64,
    pub nightly_at_utc: (u32, u32),
    /// The `scrub` object, parsed by the scrubber's own reader so there is one parser for it.
    pub scrub: crate::scrub::Config,
    /// The notes folder (`store::vault_root`); RECALL_VAULT overrides it. One root in v1.
    pub vault: Option<PathBuf>,
    /// `query::closest_labels_on`: print the closest-match label and banner. Default OFF (cli-quality addendum A).
    pub closest_labels: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            refresh: RefreshMode::Nightly,
            freshness_check: true,
            stale_penalty: 0.0,
            recency_boost: 0.0,
            nightly_at_utc: (7, 0),
            scrub: crate::scrub::Config::default(),
            vault: None,
            closest_labels: false,
        }
    }
}

pub const KEYS: [&str; 11] = [
    "refresh",
    "freshness_check",
    "stale_penalty",
    "recency_boost",
    "nightly_at_utc",
    "scrub.secrets",
    "scrub.pii",
    "scrub.custom",
    "scrub.store_redacted",
    "vault",
    "closest_labels",
];

pub fn settings_path() -> PathBuf {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default();
    PathBuf::from(home).join(".recall").join("settings.json")
}

fn parse_hhmm(s: &str) -> Result<(u32, u32)> {
    let (h, m) = s.trim().split_once(':').context("expected HH:MM")?;
    let (h, m): (u32, u32) = (h.parse()?, m.parse()?);
    if h > 23 || m > 59 {
        bail!("'{s}' is not a valid HH:MM time");
    }
    Ok((h, m))
}

fn parse_on_off(key: &str, v: &str) -> Result<bool> {
    match v.trim().to_lowercase().as_str() {
        "on" | "true" => Ok(true),
        "off" | "false" => Ok(false),
        _ => bail!("{key}: expected on or off"),
    }
}

/// `scrub.custom`: a JSON list of strings, one bare term, or "" / "[]" to clear. A `re:` entry
/// must compile, so `set` refuses it instead of the scrubber skipping it at query time.
fn parse_custom(v: &str) -> Result<Vec<String>> {
    let t = v.trim();
    let list: Vec<String> = if t.starts_with('[') {
        serde_json::from_str(t).context("scrub.custom: expected a JSON list of strings")?
    } else if t.is_empty() {
        Vec::new()
    } else {
        vec![t.to_string()]
    };
    let list: Vec<String> = list.into_iter().filter(|x| !x.is_empty()).collect();
    for (i, e) in list.iter().enumerate() {
        if let Some(re) = e.strip_prefix("re:") {
            // the entry is not echoed: it may be the sensitive term
            regex::Regex::new(re).map_err(|_| anyhow::anyhow!("scrub.custom entry #{} is not a valid regex", i + 1))?;
        }
    }
    // The scrubber joins every entry into ONE alternation; if that fails, no custom term applies.
    if !list.is_empty() && crate::scrub::compile_custom(&list).is_none() {
        bail!("scrub.custom: the entries do not combine into one pattern (e.g. a repeated (?P<name>) group)");
    }
    Ok(list)
}

/// `vault`: an existing folder, stored absolute; "" clears (back to RECALL_VAULT / the default).
fn parse_vault(v: &str) -> Result<Option<PathBuf>> {
    let t = v.trim().trim_matches('"');
    if t.is_empty() {
        return Ok(None);
    }
    let p = PathBuf::from(t);
    if !p.is_dir() {
        bail!("vault: '{t}' is not an existing folder");
    }
    Ok(Some(std::path::absolute(&p).unwrap_or(p)))
}

fn parse_rank_shift(key: &str, v: &str) -> Result<f64> {
    let x: f64 = v.trim().parse().with_context(|| format!("{key}: expected a number"))?;
    if !x.is_finite() || !(0.0..=10.0).contains(&x) {
        bail!("{key}: must be between 0 and 10 rank positions (0 = off)");
    }
    Ok(x)
}

fn on_off(b: bool) -> String {
    (if b { "on" } else { "off" }).into()
}

impl Settings {
    /// Apply one `key=value`, validating it. The single parser for both the file and the CLI.
    pub fn apply(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "refresh" => self.refresh = RefreshMode::parse(value)?,
            "freshness_check" => {
                self.freshness_check = match value.trim().to_lowercase().as_str() {
                    "lazy" | "on" | "true" => true,
                    "off" | "false" => false,
                    _ => bail!("freshness_check: expected lazy or off"),
                }
            }
            "stale_penalty" => self.stale_penalty = parse_rank_shift(key, value)?,
            "recency_boost" => self.recency_boost = parse_rank_shift(key, value)?,
            "nightly_at_utc" => self.nightly_at_utc = parse_hhmm(value)?,
            "scrub.secrets" => self.scrub.secrets = parse_on_off(key, value)?,
            "scrub.pii" => self.scrub.pii = parse_on_off(key, value)?,
            "scrub.store_redacted" => self.scrub.store_redacted = parse_on_off(key, value)?,
            "scrub.custom" => self.scrub.custom = parse_custom(value)?,
            "vault" => self.vault = parse_vault(value)?,
            "closest_labels" => self.closest_labels = parse_on_off(key, value)?,
            other => bail!("unknown setting '{other}' (known: {})", KEYS.join(", ")),
        }
        Ok(())
    }

    pub fn get(&self, key: &str) -> Option<String> {
        Some(match key {
            "refresh" => self.refresh.label(),
            "freshness_check" => (if self.freshness_check { "lazy" } else { "off" }).into(),
            "stale_penalty" => self.stale_penalty.to_string(),
            "recency_boost" => self.recency_boost.to_string(),
            "nightly_at_utc" => format!("{:02}:{:02}", self.nightly_at_utc.0, self.nightly_at_utc.1),
            "scrub.secrets" => on_off(self.scrub.secrets),
            "scrub.pii" => on_off(self.scrub.pii),
            "scrub.store_redacted" => on_off(self.scrub.store_redacted),
            "scrub.custom" => serde_json::to_string(&self.scrub.custom).unwrap_or_default(),
            "vault" => self.vault.as_ref().map(|p| p.display().to_string()).unwrap_or_default(),
            "closest_labels" => on_off(self.closest_labels),
            _ => return None,
        })
    }

    fn from_json(text: &str) -> (Self, Vec<String>) {
        let mut s = Self { scrub: crate::scrub::Config::from_json(text), ..Self::default() };
        let mut warnings = Vec::new();
        let Ok(serde_json::Value::Object(map)) = serde_json::from_str::<serde_json::Value>(text) else {
            return (s, vec!["settings.json is not a JSON object; using defaults".into()]);
        };
        for (k, v) in map {
            let as_text = match &v {
                serde_json::Value::String(x) => x.clone(),
                other => other.to_string(),
            };
            // scrub.* lives in the nested object read above; a top-level dotted key is ignored so
            // this reader and the scrubber can never disagree.
            // vault is validated on `set`, not on load: a folder that later vanished must still be
            // reported as "the vault setting points at X", not silently swapped for the default.
            if k == "vault" {
                s.vault = v.as_str().filter(|x| !x.is_empty()).map(PathBuf::from);
                continue;
            }
            if KEYS.contains(&k.as_str()) && !k.starts_with("scrub.") {
                if let Err(e) = s.apply(&k, &as_text) {
                    warnings.push(format!("settings.json: {e:#}; using the default for {k}"));
                }
            }
        }
        (s, warnings)
    }

    /// Never fails: a missing file is the defaults, a bad value falls back per key.
    /// Warnings go to stderr once per load — a silently ignored setting is the worse outcome.
    pub fn load() -> Self {
        Self::load_from(&settings_path())
    }

    pub fn load_from(path: &std::path::Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let (s, warnings) = Self::from_json(&text);
                for w in warnings {
                    eprintln!("recall: {w}");
                }
                s
            }
            Err(_) => Self::default(),
        }
    }

    /// Validate and persist one key, preserving every other key already in the file.
    pub fn set(key: &str, value: &str) -> Result<Self> {
        Self::set_in(&settings_path(), key, value)
    }

    pub fn set_in(path: &std::path::Path, key: &str, value: &str) -> Result<Self> {
        let mut s = Self::load_from(path);
        s.apply(key, value)?;
        let mut map = match std::fs::read_to_string(path)
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        {
            Some(serde_json::Value::Object(m)) => m,
            _ => serde_json::Map::new(),
        };
        if let Some(sub) = key.strip_prefix("scrub.") {
            // nested: { "scrub": { "pii": false, ... } }, keeping the object's other keys
            let json_val = match sub {
                "custom" => serde_json::json!(s.scrub.custom),
                "secrets" => serde_json::json!(s.scrub.secrets),
                "pii" => serde_json::json!(s.scrub.pii),
                _ => serde_json::json!(s.scrub.store_redacted),
            };
            let obj = map.entry("scrub").or_insert_with(|| serde_json::json!({}));
            if !obj.is_object() {
                *obj = serde_json::json!({});
            }
            obj.as_object_mut().unwrap().insert(sub.to_string(), json_val);
        } else {
            let canonical = s.get(key).unwrap_or_default();
            let json_val = match key {
                "stale_penalty" | "recency_boost" => serde_json::json!(canonical.parse::<f64>()?),
                "vault" if canonical.is_empty() => serde_json::Value::Null,
                _ => serde_json::json!(canonical),
            };
            map.insert(key.to_string(), json_val);
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        }
        // Write-then-rename, so a query reading mid-save sees the old file or the new one, never a
        // truncated one (which would load defaults and drop the custom scrub list for that query).
        // Unique per writer: two processes sharing one temp name could rename the other's
        // half-written file into place.
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let tmp = path.with_extension(format!("json.{}.{nanos}.tmp", std::process::id()));
        std::fs::write(&tmp, serde_json::to_string_pretty(&serde_json::Value::Object(map))?)
            .with_context(|| format!("write {}", tmp.display()))?;
        if let Err(e) = std::fs::rename(&tmp, path) {
            let _ = std::fs::remove_file(&tmp); // a read-only settings.json must not collect one temp file per run
            return Err(e).with_context(|| format!("replace {}", path.display()));
        }
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refresh_parses_every_documented_form() {
        assert_eq!(RefreshMode::parse("nightly").unwrap(), RefreshMode::Nightly);
        assert_eq!(RefreshMode::parse("On-Change").unwrap(), RefreshMode::OnChange);
        assert_eq!(RefreshMode::parse("every 30 minutes").unwrap(), RefreshMode::Every(30));
        assert_eq!(RefreshMode::parse("every 2 hours").unwrap(), RefreshMode::Every(120));
        assert_eq!(RefreshMode::parse("every 90m").unwrap(), RefreshMode::Every(90));
        assert_eq!(RefreshMode::parse("6h").unwrap(), RefreshMode::Every(360));
        assert!(RefreshMode::parse("every 1 minute").is_err(), "below the 5 minute floor");
        assert!(RefreshMode::parse("every 9 days").is_err());
        assert!(RefreshMode::parse("sometimes").is_err());
    }

    #[test]
    fn defaults_are_nightly_lazy_and_both_ranking_knobs_off() {
        let s = Settings::default();
        assert_eq!(s.refresh, RefreshMode::Nightly);
        assert!(s.freshness_check);
        assert_eq!((s.stale_penalty, s.recency_boost), (0.0, 0.0));
        assert!(!s.closest_labels, "closest-match label ships off (cli-quality addendum A)");
    }

    #[test]
    fn closest_labels_round_trips() {
        let dir = std::env::temp_dir().join(format!("recall-settings-labels-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("settings.json");
        assert!(Settings::set_in(&path, "closest_labels", "maybe").is_err());
        Settings::set_in(&path, "closest_labels", "on").unwrap();
        assert!(Settings::load_from(&path).closest_labels);
        Settings::set_in(&path, "closest_labels", "off").unwrap();
        assert_eq!(Settings::load_from(&path).get("closest_labels").as_deref(), Some("off"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn set_validates_persists_and_keeps_other_keys() {
        let dir = std::env::temp_dir().join(format!("recall-settings-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("settings.json");
        assert!(Settings::set_in(&path, "refresh", "whenever").is_err());
        assert!(!path.exists(), "a rejected value must not create the file");
        Settings::set_in(&path, "refresh", "every 2 hours").unwrap();
        Settings::set_in(&path, "stale_penalty", "2").unwrap();
        let s = Settings::load_from(&path);
        assert_eq!(s.refresh, RefreshMode::Every(120));
        assert_eq!(s.stale_penalty, 2.0);
        assert!(Settings::set_in(&path, "stale_penalty", "99").is_err());
        assert!(Settings::set_in(&path, "nope", "1").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scrub_keys_round_trip_through_the_nested_object_the_scrubber_reads() {
        let dir = std::env::temp_dir().join(format!("recall-settings-scrub-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("settings.json");
        Settings::set_in(&path, "refresh", "on-change").unwrap();
        Settings::set_in(&path, "scrub.pii", "off").unwrap();
        Settings::set_in(&path, "scrub.custom", r#"["Project Falcon","re:ACME-\\d{4}"]"#).unwrap();
        assert!(Settings::set_in(&path, "scrub.secrets", "maybe").is_err());
        assert!(Settings::set_in(&path, "scrub.custom", r#"["re:("]"#).is_err(), "bad regex refused");
        assert!(Settings::set_in(&path, "scrub.custom", r#"["re:(?P<x>a)","re:(?P<x>b)"]"#).is_err(), "must combine");
        let cfg = crate::scrub::Config::load_from(&path);
        assert!(!cfg.pii && cfg.secrets && !cfg.store_redacted);
        assert_eq!(cfg.custom, vec!["Project Falcon".to_string(), r"re:ACME-\d{4}".to_string()]);
        let s = Settings::load_from(&path);
        assert_eq!(s.refresh, RefreshMode::OnChange, "other keys kept");
        assert_eq!((s.get("scrub.pii").unwrap(), s.get("scrub.secrets").unwrap()), ("off".into(), "on".into()));
        Settings::set_in(&path, "scrub.pii", "on").unwrap();
        Settings::set_in(&path, "scrub.custom", "").unwrap();
        let cfg = crate::scrub::Config::load_from(&path);
        assert!(cfg.pii && cfg.custom.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn vault_round_trips_validates_and_clears() {
        let dir = std::env::temp_dir().join(format!("recall-settings-vault-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let notes = dir.join("notes");
        std::fs::create_dir_all(&notes).unwrap();
        let path = dir.join("settings.json");
        assert!(Settings::set_in(&path, "vault", dir.join("nope").to_str().unwrap()).is_err(), "must exist");
        Settings::set_in(&path, "refresh", "on-change").unwrap();
        Settings::set_in(&path, "vault", notes.to_str().unwrap()).unwrap();
        let s = Settings::load_from(&path);
        assert_eq!(s.vault.as_deref(), Some(notes.as_path()));
        assert_eq!(s.refresh, RefreshMode::OnChange, "other keys kept");
        // a folder removed after `set` is still reported as the setting, not swapped for a default
        std::fs::remove_dir_all(&notes).unwrap();
        assert_eq!(Settings::load_from(&path).vault.as_deref(), Some(notes.as_path()));
        Settings::set_in(&path, "vault", "").unwrap();
        assert_eq!(Settings::load_from(&path).vault, None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_top_level_dotted_scrub_key_is_ignored_like_the_scrubber_ignores_it() {
        let text = r#"{"scrub":{"secrets":false},"scrub.secrets":true}"#;
        let (s, _) = Settings::from_json(text);
        assert!(!s.scrub.secrets && !crate::scrub::Config::from_json(text).secrets);
    }

    #[test]
    fn a_bad_value_in_the_file_falls_back_per_key_not_wholesale() {
        let (s, w) = Settings::from_json(r#"{"refresh":"every 2 hours","stale_penalty":"abc"}"#);
        assert_eq!(s.refresh, RefreshMode::Every(120));
        assert_eq!(s.stale_penalty, 0.0);
        assert_eq!(w.len(), 1);
    }
}
