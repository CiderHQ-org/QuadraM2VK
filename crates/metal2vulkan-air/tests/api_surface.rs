//! The public surface of `metal2vulkan-air`, read from source and pinned in
//! `tests/api_surface.txt`.
//!
//! The surface is every public module the crate root reaches through `pub mod`, and every public
//! item at each such module's root: functions, types (with their construction class), traits,
//! constants, statics, re-exports, and the types the crate's own declaring macros (`ir_id!`,
//! `flag_set!`, `metal_enum!`) generate, with a private module's items a glob re-exports. A
//! test-only module or item (`#[cfg(test)]`) is not part of it. Methods, fields, and variants are
//! not listed.
//!
//! The crate moves in lockstep with `metal2vulkan`, which pins it exactly (`=0.x.y`): a removed or
//! changed line is a breaking change of this crate, released as a minor version of both. On a
//! mismatch the test names each side's extra lines; `print_the_surface` prints the scanned table:
//!
//! ```sh
//! cargo test -p metal2vulkan-air --test api_surface -- --ignored --nocapture print_the_surface
//! ```
//!
//! Construction classes: `N` is `#[non_exhaustive]` (fields and variants may grow within a minor
//! version); `O` is opaque (a private field, so a dependent cannot build or destructure it); `E` is
//! exhaustive: every field public, or an enum a dependent matches completely.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// The attribute and doc lines directly above line `i`.
fn attrs_above(lines: &[&str], i: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut j = i;
    while j > 0 {
        let l = lines[j - 1];
        let attr_like = l.starts_with("#[") || l.starts_with("//") || l.starts_with(")]");
        if !attr_like {
            break;
        }
        out.push(l.trim().to_owned());
        j -= 1;
    }
    out
}

fn ident_prefix(s: &str) -> String {
    s.chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

fn test_only(attrs: &[String]) -> bool {
    attrs.iter().any(|a| a == "#[cfg(test)]")
}

/// The construction class of the struct or enum declared at line `i`.
fn class_at(lines: &[&str], i: usize, is_struct: bool) -> &'static str {
    if attrs_above(lines, i)
        .iter()
        .any(|a| a == "#[non_exhaustive]")
    {
        return "N";
    }
    if !is_struct {
        return "E";
    }
    let after_name = lines[i]
        .split_once("struct ")
        .map(|(_, r)| r)
        .unwrap_or_default();
    let rest = after_name.trim_start_matches(|c: char| c.is_alphanumeric() || c == '_');
    let rest = rest
        .strip_prefix('<')
        .and_then(|generic| generic.split_once('>'))
        .map_or(rest, |(_, after)| after);
    if let Some(tuple) = rest.strip_prefix('(') {
        let body = tuple.rsplit_once(')').map(|(b, _)| b).unwrap_or(tuple);
        let private = body
            .split(',')
            .map(str::trim)
            .filter(|f| !f.is_empty())
            .any(|f| !f.starts_with("pub ") || f.starts_with("pub("));
        return if private { "O" } else { "E" };
    }
    if rest.trim_start().starts_with(';') {
        return "E";
    }
    // A braced struct: its fields are the four-space-indented `name: Type` lines.
    for l in &lines[i + 1..] {
        if *l == "}" {
            break;
        }
        let Some(field) = l.strip_prefix("    ") else {
            continue;
        };
        if field.starts_with(' ') || field.starts_with("//") || field.starts_with('#') {
            continue;
        }
        if field.starts_with("pub(") || (!field.starts_with("pub ") && field.contains(':')) {
            return "O";
        }
    }
    "E"
}

/// The type a declaring macro invocation at line `i` generates: its name and kind.
fn generated_item(lines: &[&str], i: usize) -> Option<(String, String)> {
    let header = || {
        lines[i + 1..]
            .iter()
            .map(|l| l.trim())
            .take_while(|l| l.starts_with("///") || l.starts_with("#["))
    };
    let kind = match lines[i] {
        // A `u32` index behind a private field, and a flag set's private bits.
        "ir_id!(" | "flag_set!(" => "struct(O)".to_owned(),
        "metal_enum! {" if header().any(|l| l == "#[non_exhaustive]") => "enum(N)".to_owned(),
        "metal_enum! {" => "enum(E)".to_owned(),
        _ => return None,
    };
    let name = lines[i + 1..]
        .iter()
        .map(|l| l.trim())
        .find(|l| !l.starts_with("///") && !l.starts_with("#["))?;
    Some((ident_prefix(name), kind))
}

/// The leaves of a `pub use` tree (`a::{b::{C, D}, E as F}`), as (path, name).
fn use_leaves(prefix: &str, tree: &str, out: &mut Vec<(String, String)>) {
    let tree = tree.trim();
    if let Some(open) = tree.find('{') {
        let head = format!("{prefix}{}", &tree[..open]);
        let inner = &tree[open + 1..tree.rfind('}').expect("a closed use group")];
        let mut depth = 0;
        let mut start = 0;
        for (k, c) in inner.char_indices() {
            match c {
                '{' => depth += 1,
                '}' => depth -= 1,
                ',' if depth == 0 => {
                    use_leaves(&head, &inner[start..k], out);
                    start = k + 1;
                }
                _ => {}
            }
        }
        if !inner[start..].trim().is_empty() {
            use_leaves(&head, &inner[start..], out);
        }
        return;
    }
    let (path, name) = match tree.split_once(" as ") {
        Some((path, alias)) => (path.trim(), alias.trim().to_owned()),
        None => (tree, tree.rsplit("::").next().unwrap_or(tree).to_owned()),
    };
    out.push((format!("{prefix}{path}"), name));
}

/// Every surface line of module `module` (declared in `file`, its children in `dir`), and of the
/// public modules below it.
fn scan(module: &str, file: &Path, dir: &Path, out: &mut BTreeSet<String>) {
    let text = std::fs::read_to_string(file).unwrap_or_else(|e| panic!("{file:?}: {e}"));
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let attrs = attrs_above(&lines, i);
        if test_only(&attrs) {
            i += 1;
            continue;
        }
        if let Some((name, kind)) = generated_item(&lines, i) {
            out.insert(format!("{module} {name} {kind}"));
        } else if let Some(rest) = line.strip_prefix("pub use ") {
            let mut tree = rest.to_owned();
            while !tree.trim_end().ends_with(';') {
                i += 1;
                tree.push(' ');
                tree.push_str(lines[i].trim());
            }
            let tree = tree.trim_end().trim_end_matches(';');
            let mut leaves = Vec::new();
            use_leaves("", tree, &mut leaves);
            for (path, name) in leaves {
                let path = path.split_whitespace().collect::<String>();
                // A glob of a private child module (`mod generated; pub use generated::*;`) makes
                // that module's items this module's.
                if let Some(child) = path.strip_suffix("::*").filter(|c| !c.contains("::")) {
                    let child_file = if dir.join(child).join("mod.rs").is_file() {
                        dir.join(child).join("mod.rs")
                    } else {
                        dir.join(format!("{child}.rs"))
                    };
                    scan(module, &child_file, &dir.join(child), out);
                } else {
                    out.insert(format!("{module} {name} use({path})"));
                }
            }
        } else if let Some(rest) = line.strip_prefix("pub mod ") {
            let name = ident_prefix(rest);
            out.insert(format!("{module} {name} mod"));
            let child = if module == "crate" {
                name.clone()
            } else {
                format!("{module}::{name}")
            };
            let (child_file, child_dir) = if dir.join(&name).join("mod.rs").is_file() {
                (dir.join(&name).join("mod.rs"), dir.join(&name))
            } else {
                (dir.join(format!("{name}.rs")), dir.join(&name))
            };
            assert!(
                rest.trim_end().ends_with(';'),
                "{file:?}: inline `pub mod {name}` is not read"
            );
            scan(&child, &child_file, &child_dir, out);
        } else if let Some(rest) = line.strip_prefix("pub ") {
            let (word, name) = rest
                .strip_prefix("const fn ")
                .map(|r| ("fn", r))
                .or_else(|| rest.strip_prefix("unsafe fn ").map(|r| ("fn", r)))
                .or_else(|| rest.split_once(' '))
                .unwrap_or(("", ""));
            let kind = match word {
                "struct" => Some(format!("struct({})", class_at(&lines, i, true))),
                "enum" => Some(format!("enum({})", class_at(&lines, i, false))),
                "fn" | "trait" | "const" | "static" | "type" => Some(word.to_owned()),
                _ => None,
            };
            if let Some(kind) = kind {
                out.insert(format!("{module} {} {kind}", ident_prefix(name)));
            }
        }
        i += 1;
    }
}

fn scanned() -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    scan("crate", &src_dir().join("lib.rs"), &src_dir(), &mut out);
    out
}

fn table() -> BTreeSet<String> {
    let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/api_surface.txt");
    std::fs::read_to_string(&file)
        .unwrap_or_else(|e| panic!("{file:?}: {e}"))
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_owned)
        .collect()
}

/// The table equals the source: same modules, items, kinds, and construction classes.
#[test]
fn the_table_matches_the_source() {
    let (table, source) = (table(), scanned());
    let only_table: Vec<_> = table.difference(&source).collect();
    let only_source: Vec<_> = source.difference(&table).collect();
    assert!(
        only_table.is_empty() && only_source.is_empty(),
        "the public surface of metal2vulkan-air changed: update air/tests/api_surface.txt (a \
         removed or changed line is a breaking change)\nin the table, not \
         the source:\n  {}\nin the source, not the table:\n  {}",
        only_table
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n  "),
        only_source
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}

/// The scan reads what it claims: a test-only module stays out, the declaring macros' types and
/// the crate root's re-exports are in, and so are nested public modules.
#[test]
fn the_scan_reads_the_crate() {
    let source = scanned();
    for line in [
        "crate canon mod",
        "canon discard_sink mod",
        "ir::id FuncId struct(O)",
        "ir::inst FastMath struct(O)",
        "metal DataType enum(N)",
        "crate Entry use(canon::Entry)",
        "crate specialize_linked fn",
        "error Error enum(E)",
        "error InternalKind enum(N)",
    ] {
        assert!(source.contains(line), "{line}");
    }
    assert!(!source.iter().any(|l| l.contains("families_tests")));
}

/// Prints the scanned surface, to write `tests/api_surface.txt` from.
#[test]
#[ignore = "prints the table; run it with --ignored --nocapture"]
fn print_the_surface() {
    for line in scanned() {
        println!("{line}");
    }
}
