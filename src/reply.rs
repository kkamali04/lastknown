// reply.rs — terminal output helper.
// Plain ASCII labels. Colour only where `init` allowed it, and every coloured span carries its own
// reset, so a line can never leave the terminal coloured.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

static OUT: AtomicBool = AtomicBool::new(false);
static ERR: AtomicBool = AtomicBool::new(false);
// Where a bare reset may be written: a coloured stream that is a terminal. Never a pipe or a file
// (it would add bytes to data, and a full pipe would block the interrupt handler).
static RESET_OUT: AtomicBool = AtomicBool::new(false);
static RESET_ERR: AtomicBool = AtomicBool::new(false);

const RESET: &str = "\x1b[0m";
// SGR codes from the terminal's own basic palette, so they follow a light or dark theme.
pub const BOLD: &str = "1";
pub const DIM: &str = "2";
pub const NOTE: &str = "33"; // yellow
pub const WARN: &str = "1;33"; // bold yellow
pub const BAD: &str = "1;31"; // bold red
pub const MARK: &str = "36"; // cyan

/// `--color`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum)]
pub enum Color {
    Auto,
    Always,
    Never,
}

/// The whole rule. `--color always|never` wins; otherwise NO_COLOR turns colour off, CLICOLOR_FORCE
/// turns it on, and without either it is on only for a terminal that reads escape sequences.
fn decide(when: Color, no_color: bool, force: bool, terminal: bool) -> bool {
    match when {
        Color::Always => true,
        Color::Never => false,
        Color::Auto => !no_color && (force || terminal),
    }
}

/// Decide colour for stdout and stderr separately, once, before anything is printed.
pub fn init(when: Color) {
    let set = |name: &str| std::env::var_os(name).is_some_and(|v| !v.is_empty());
    let (no_color, force) = (set("NO_COLOR"), std::env::var_os("CLICOLOR_FORCE").is_some_and(|v| !v.is_empty() && v != "0"));
    // TERM=dumb reads no escapes; off Windows, neither does a terminal with no TERM at all.
    let dumb = std::env::var_os("TERM").map_or(cfg!(not(windows)), |t| t == "dumb");
    for (stderr, tty, on, reset) in [
        (false, std::io::stdout().is_terminal(), &OUT, &RESET_OUT),
        (true, std::io::stderr().is_terminal(), &ERR, &RESET_ERR),
    ] {
        // Ask the console only while colour is still possible: asking switches a Windows console's mode.
        let terminal = decide(when, no_color, force, true) && tty && !dumb && console_reads_escapes(stderr);
        on.store(decide(when, no_color, force, terminal), Relaxed);
        reset.store(on.load(Relaxed) && tty, Relaxed);
    }
    if RESET_OUT.load(Relaxed) || RESET_ERR.load(Relaxed) {
        reset_on_interrupt();
    }
}

/// Windows: switch the console to escape-sequence processing; `false` when it cannot (an old
/// console would print the escapes literally). A terminal that is not a console (Git Bash's
/// mintty is a pipe) reads escapes itself.
#[cfg(windows)]
fn console_reads_escapes(stderr: bool) -> bool {
    use windows_sys::Win32::System::Console::{
        GetConsoleMode, GetStdHandle, SetConsoleMode, ENABLE_VIRTUAL_TERMINAL_PROCESSING, STD_ERROR_HANDLE, STD_OUTPUT_HANDLE,
    };
    unsafe {
        let handle = GetStdHandle(if stderr { STD_ERROR_HANDLE } else { STD_OUTPUT_HANDLE });
        let mut mode = 0;
        if GetConsoleMode(handle, &mut mode) == 0 {
            return true;
        }
        mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING != 0 || SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0
    }
}

#[cfg(not(windows))]
fn console_reads_escapes(_stderr: bool) -> bool {
    true
}

/// Ctrl+C between two writes of one long line: write the reset, then let the default action end
/// the process exactly as before.
#[cfg(unix)]
fn reset_on_interrupt() {
    extern "C" fn on_sigint(_: libc::c_int) {
        // Only async-signal-safe calls here.
        unsafe {
            for (on, fd) in [(&RESET_OUT, 1), (&RESET_ERR, 2)] {
                if on.load(Relaxed) {
                    libc::write(fd, RESET.as_ptr().cast(), RESET.len());
                }
            }
            libc::signal(libc::SIGINT, libc::SIG_DFL);
            libc::raise(libc::SIGINT);
        }
    }
    unsafe {
        // A process started with SIGINT ignored (a background job) keeps ignoring it.
        if libc::signal(libc::SIGINT, on_sigint as extern "C" fn(libc::c_int) as libc::sighandler_t) == libc::SIG_IGN {
            libc::signal(libc::SIGINT, libc::SIG_IGN);
        }
    }
}

#[cfg(windows)]
fn reset_on_interrupt() {
    unsafe extern "system" fn on_ctrl(_: u32) -> windows_sys::core::BOOL {
        reset();
        0 // not handled: the default handler still ends the process
    }
    unsafe {
        windows_sys::Win32::System::Console::SetConsoleCtrlHandler(Some(on_ctrl), 1);
    }
}

#[cfg(not(any(unix, windows)))]
fn reset_on_interrupt() {}

/// Write a bare reset to each coloured terminal, ignoring errors. For the exits that do not return
/// through a finished `println!`: a panic, and a reader that closed the pipe in the middle of a
/// coloured line (its half is already on the terminal that stderr may share).
pub fn reset() {
    if RESET_OUT.load(Relaxed) {
        let mut out = std::io::stdout();
        let _ = out.write_all(RESET.as_bytes()).and_then(|_| out.flush());
    }
    if RESET_ERR.load(Relaxed) {
        let _ = std::io::stderr().write_all(RESET.as_bytes());
    }
}

fn wrap(on: bool, style: &str, text: &str) -> String {
    if on && !text.is_empty() {
        format!("\x1b[{style}m{text}{RESET}")
    } else {
        text.to_string()
    }
}

/// `text` styled for stdout and closed by a reset; `text` itself when stdout colour is off.
pub fn paint(style: &str, text: &str) -> String {
    wrap(OUT.load(Relaxed), style, text)
}

/// [`paint`] for stderr.
pub fn paint_err(style: &str, text: &str) -> String {
    wrap(ERR.load(Relaxed), style, text)
}

/// Untrusted text for human output. Keep layout, but remove C0/C1 controls (including
/// ESC, carriage return and backspace) that can change terminal state or overwrite text.
pub fn plain(text: &str) -> String {
    text.chars().filter(|c| !c.is_control() || matches!(c, '\n' | '\t')).collect()
}

/// Colour the `[matched]` words of an excerpt. The brackets stay; with colour off nothing changes.
/// Only a bracketed word that starts like a word of `query` is coloured, so a `[EMAIL]` placeholder
/// or other bracketed text in the note is left alone.
pub fn marks(excerpt: &str, query: &str) -> String {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    if !OUT.load(Relaxed) {
        return excerpt.to_string();
    }
    let query = query.to_lowercase();
    let stems: Vec<String> = query.split(|c: char| !c.is_alphanumeric()).map(|w| w.chars().take(3).collect()).collect();
    let re = RE.get_or_init(|| regex::Regex::new(r"\[([^\s\[\]]+)\]").unwrap());
    re.replace_all(excerpt, |c: &regex::Captures| {
        let word = c[1].to_lowercase();
        wrap(stems.iter().any(|s| !s.is_empty() && word.starts_with(s.as_str())), MARK, &c[0])
    })
    .into_owned()
}

/// A warning above a result list.
pub fn warn(msg: &str) {
    println!("  {}\n", paint(WARN, &format!("Warning: {msg}")));
}

/// A closing summary: a title, then `label  value` rows aligned in two columns. A row whose flag is
/// false is printed only when its value is not zero.
pub fn summary(title: &str, rows: &[(&str, i64, bool)]) {
    let rows: Vec<_> = rows.iter().filter(|(_, n, always)| *always || *n != 0).collect();
    let label = rows.iter().map(|(l, ..)| l.len()).max().unwrap_or(0);
    let value = rows.iter().map(|(_, n, _)| n.to_string().len()).max().unwrap_or(0);
    println!("\n{}", paint(BOLD, title));
    for (l, n, _) in rows {
        println!("  {l:<label$}  {n:>value$}");
    }
    println!();
}

/// The label a section header prints as. "WHAT I DID" is the plain result and has none.
fn label(header: &str) -> Option<String> {
    match header {
        "WHAT I DID" => None,
        "BLOCKING" => Some(paint(BAD, "Refused:")),
        "YOU NEED TO DO" => Some(paint(WARN, "To do:")),
        "NEXT UP" => Some(paint(BOLD, "Next:")),
        other => Some(paint(BOLD, &format!("{other}:"))),
    }
}

/// One result line.
pub fn section(header: &str, msg: &str) {
    match label(header) {
        Some(l) => println!("{l} {msg}"),
        None => println!("{msg}"),
    }
}

/// A refusal on stderr.
pub fn blocking(_context: &str, msg: &str) {
    eprintln!("{} {}", paint_err(BAD, "error:"), msg);
}

/// Several lines per section, each section under its label.
pub fn card(sections: &[(&str, &[&str])]) {
    println!();
    for (header, lines) in sections {
        let label = label(header);
        let indent = if label.is_some() { "  " } else { "" };
        if let Some(l) = label {
            println!("{l}");
        }
        for line in *lines {
            println!("{indent}{line}");
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colour_rule() {
        use Color::*;
        // (when, NO_COLOR, CLICOLOR_FORCE, terminal) -> on
        for (when, no_color, force, terminal, on) in [
            (Auto, false, false, true, true),
            (Auto, false, false, false, false), // piped or redirected
            (Auto, true, false, true, false),   // NO_COLOR
            (Auto, false, true, false, true),   // CLICOLOR_FORCE, piped
            (Auto, true, true, true, false),    // NO_COLOR beats CLICOLOR_FORCE
            (Always, true, false, false, true), // the flag beats the environment
            (Never, false, true, true, false),
        ] {
            assert_eq!(decide(when, no_color, force, terminal), on, "{when:?} {no_color} {force} {terminal}");
        }
    }

    #[test]
    fn every_span_is_closed_and_plain_text_is_untouched() {
        assert_eq!(wrap(false, BOLD, "title"), "title");
        assert_eq!(wrap(true, BOLD, "title"), "\x1b[1mtitle\x1b[0m");
        assert_eq!(wrap(true, BOLD, ""), "");
        assert_eq!(plain("a\x1b[31mb"), "a[31mb");
    }

    #[test]
    fn note_text_cannot_move_the_cursor_or_send_terminal_controls() {
        let controls: String = (0..=0x9f).filter_map(char::from_u32)
            .filter(|c| c.is_control()).collect();
        assert_eq!(plain(&controls), "\t\n");
        assert_eq!(plain("Title\rforged\u{8}!\u{7f}\u{9b}31m\u{9d}title\u{7}"), "Titleforged!31mtitle");
        assert_eq!(plain("café 日本語\n\ttext"), "café 日本語\n\ttext");
    }
}
