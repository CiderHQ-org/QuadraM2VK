//! The consumer guide's example: `examples/api_preview.rs` runs, and every code block of
//! `docs/API.md` is quoted from it, so the guide's code compiles and runs in the gate.

use std::path::Path;
use std::process::Command;

#[path = "../examples/api_preview.rs"]
mod example;

/// The example runs to completion on the committed profiles, or is skipped visibly without
/// `llvm-as` (the owned fixtures are LLVM text).
#[test]
fn the_api_preview_example_runs() {
    if let Err(e) = Command::new("llvm-as").arg("--version").output() {
        eprintln!("SKIPPED: llvm-as not on PATH: {e}");
        return;
    }
    if let Err(e) = example::main() {
        panic!("examples/api_preview.rs failed: {e}");
    }
}

/// The trimmed, non-empty lines of `text`.
fn lines(text: &str) -> Vec<&str> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect()
}

/// Every ```` ```rust ```` block of `docs/API.md` is quoted from the example: each run of lines
/// between `// ...` elisions appears in it, contiguously and in order, ignoring indentation and
/// blank lines.
#[test]
fn the_guide_quotes_the_example() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let guide = std::fs::read_to_string(root.join("../../docs/API.md")).expect("docs/API.md");
    let source = std::fs::read_to_string(root.join("examples/api_preview.rs")).expect("example");
    let source = lines(&source);
    let mut blocks = 0;
    let mut rest = guide.as_str();
    while let Some(start) = rest.find("```rust\n") {
        let body = &rest[start + "```rust\n".len()..];
        let end = body
            .find("```")
            .expect("an unterminated code block in docs/API.md");
        let mut from = 0;
        for run in body[..end].split("// ...") {
            let run = lines(run);
            if run.is_empty() {
                continue;
            }
            let at = (from..=source.len().saturating_sub(run.len()))
                .find(|&i| source[i..i + run.len()] == run[..])
                .unwrap_or_else(|| {
                    panic!(
                        "docs/API.md quotes lines examples/api_preview.rs does not hold (in \
                         this order):\n{}",
                        run.join("\n")
                    )
                });
            from = at + run.len();
        }
        blocks += 1;
        rest = &body[end + 3..];
    }
    assert!(blocks > 0, "docs/API.md has no rust code block");
}
