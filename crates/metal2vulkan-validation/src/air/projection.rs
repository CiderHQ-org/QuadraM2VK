//! One entry point's projection of a module that declares several.
//!
//! A source row of such a module stands for one of its entries, so every fact classification
//! derives for the row must be that entry's. The projection is the module's text cut down to what
//! the entry reaches, and classification reads it exactly as it reads a module with one entry.
//! It is a text slicer over `llvm-dis` output, so it places every top-level line it keeps by
//! rule, and refuses, naming the line, any it cannot place.
//!
//! Every projection keeps:
//! - what no entry owns: the header (`source_filename`, `target ...`), named types, comdats,
//!   attribute groups, and module `asm`;
//! - the module's static initializers: `@llvm.global_ctors` and `@llvm.global_dtors` with the
//!   functions they list, and every function defined in an `air.static_init` or
//!   `air.fc_initializer` section, listed or not (a lone `air.static_init` function needs no
//!   list). Metal runs them once per pipeline, before any entry (the product's
//!   `air::canon::static_init`; the oracle's fuel plan starts every function of either section),
//!   so every entry reaches them and each is charged with what they do.
//!
//! Each entry's projection also keeps what it reaches:
//! - the entry, and every global, alias, function, and declaration that kept text names
//!   (`@symbol`), through function bodies and global initializers alike: a function another
//!   function only takes the address of, or that a global's initializer names (a function table),
//!   is reached;
//! - the unnamed metadata that kept text and the entry's own `!air.*` node name, transitively,
//!   and every symbol that metadata names (the predicate global an argument's
//!   `air.function_constant` names is the entry's);
//! - of `!air.kernel`, `!air.vertex`, and `!air.fragment`, the entry's node alone;
//! - of every other named metadata, each operand whose metadata names only symbols the entry
//!   reaches, or none (module flags, compile options, versions). An operand that names a symbol
//!   the entry does not reach describes something that is not the entry's, so it is kept for
//!   exactly the entries that reach every symbol it names: an operand naming one helper of each of
//!   two entries is kept for neither, and one naming a helper both reach is kept for both. A named
//!   metadata left with no operand is dropped.
//!
//! Comments, `@llvm.used`, `@llvm.compiler.used` (liveness annotations no entry executes), and
//! everything no kept text reaches are dropped. A symbol or node that kept text names but the
//! module does not define, a top-level line of any other kind, and an entry the module does not
//! define or declares twice are errors.

use super::{module_entries, ModuleEntry};
use std::collections::{BTreeSet, HashMap};

/// The static-initializer lists every entry reaches.
const STATIC_INITIALIZERS: [&str; 2] = ["llvm.global_ctors", "llvm.global_dtors"];

/// The sections of the functions the runtime starts itself, before any entry, whether or not a
/// static-initializer list names them: a lone `air.static_init` function needs no list (the
/// product's `air::canon::static_init`), and the oracle's fuel plan starts every function of
/// either section (`oracle_fuel::plan`'s `runtime_started`).
const RUNTIME_STARTED_SECTIONS: [&str; 2] = [
    "section \"air.static_init\"",
    "section \"air.fc_initializer\"",
];

/// What one top-level item of the module's text is.
enum Item<'a> {
    /// In every projection: the header, a named type, a comdat, an attribute group, module `asm`.
    Module,
    /// A global, alias, function definition (with its body), or declaration: kept when the entry
    /// reaches it.
    Symbol,
    /// A named metadata list: its name and its node operands.
    Named { name: &'a str, operands: Vec<u32> },
    /// An unnamed metadata node and its id.
    Node(u32),
    /// A comment or a blank line: in no projection.
    Comment,
}

/// The module's top-level items, each with its text (whole lines, a function with its body).
struct Parsed<'a> {
    items: Vec<(Item<'a>, &'a str)>,
    symbols: HashMap<&'a str, usize>,
    nodes: HashMap<u32, usize>,
    /// The definitions in a [`RUNTIME_STARTED_SECTIONS`] section, which every entry reaches.
    runtime_started: Vec<usize>,
}

/// The projection of the module `ll` onto its entry `entry` (the module-level rules above).
pub fn entry_projection(ll: &str, entry: &str) -> Result<String, String> {
    let parsed = parse(ll)?;
    let declared = module_entries(ll)?
        .into_iter()
        .find(|declared| declared.name == entry)
        .ok_or_else(|| format!("the module declares no entry {entry}"))?;
    let entry_node = parsed.entry_node(entry)?;
    let entry_item = *parsed
        .symbols
        .get(entry)
        .ok_or_else(|| format!("the module does not define the entry {entry}"))?;

    // Reach: from the entry, the static initializers (the lists and every function the runtime
    // starts), and the entry's own node.
    let mut kept_items = vec![false; parsed.items.len()];
    let mut kept_nodes = BTreeSet::new();
    let mut pending_items = std::iter::once(entry_item)
        .chain(
            STATIC_INITIALIZERS
                .iter()
                .filter_map(|name| parsed.symbols.get(name).copied()),
        )
        .chain(parsed.runtime_started.iter().copied())
        .collect::<Vec<_>>();
    let mut pending_nodes = vec![entry_node];
    loop {
        let text = if let Some(item) = pending_items.pop() {
            if kept_items[item] {
                continue;
            }
            kept_items[item] = true;
            parsed.items[item].1
        } else if let Some(id) = pending_nodes.pop() {
            if !kept_nodes.insert(id) {
                continue;
            }
            parsed.node_text(id)?
        } else {
            break;
        };
        let mut missing = None;
        references(
            text,
            |name| match parsed.symbols.get(name) {
                Some(&item) => pending_items.push(item),
                None => {
                    missing.get_or_insert_with(|| format!("@{name}"));
                }
            },
            |id| pending_nodes.push(id),
        );
        if let Some(missing) = missing {
            return Err(format!(
                "kept text names {missing}, which the module does not define or declare"
            ));
        }
    }

    // Named metadata: the entry's node of the entry lists; any other operand only while every
    // symbol its metadata names is kept.
    let mut named = HashMap::new();
    for (item, _) in &parsed.items {
        let Item::Named { name, operands } = item else {
            continue;
        };
        let mut kept = Vec::new();
        for &id in operands {
            if super::is_entry_list(name) {
                if id == entry_node {
                    kept.push(id);
                }
                continue;
            }
            let (symbols, nodes) = parsed.closure(id)?;
            if symbols.iter().all(|name| {
                parsed
                    .symbols
                    .get(name)
                    .is_some_and(|&item| kept_items[item])
            }) {
                kept_nodes.extend(nodes);
                kept.push(id);
            }
        }
        named.insert(*name, kept);
    }

    let mut projection = String::with_capacity(ll.len());
    for (index, (item, text)) in parsed.items.iter().enumerate() {
        match item {
            Item::Comment => {}
            Item::Module => push_line(&mut projection, text),
            Item::Symbol => {
                if kept_items[index] {
                    push_line(&mut projection, text);
                }
            }
            Item::Named { name, .. } => {
                let operands = &named[name];
                if !operands.is_empty() {
                    let operands = operands
                        .iter()
                        .map(|id| format!("!{id}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    push_line(&mut projection, &format!("!{name} = !{{{operands}}}"));
                }
            }
            Item::Node(id) => {
                if kept_nodes.contains(id) {
                    push_line(&mut projection, text);
                }
            }
        }
    }
    let projected = module_entries(&projection)?;
    if projected
        != [ModuleEntry {
            stage: declared.stage,
            name: entry.to_string(),
        }]
    {
        return Err(format!(
            "the projection declares the entries {projected:?}, not exactly {entry}"
        ));
    }
    Ok(projection)
}

impl<'a> Parsed<'a> {
    /// The node of `entry` among the entry lists' operands: the one whose first symbol is it.
    fn entry_node(&self, entry: &str) -> Result<u32, String> {
        let mut found = None;
        for (item, _) in &self.items {
            let Item::Named { name, operands } = item else {
                continue;
            };
            if !super::is_entry_list(name) {
                continue;
            }
            for &id in operands {
                let mut first = None;
                references(
                    self.node_text(id)?,
                    |symbol| {
                        first.get_or_insert(symbol);
                    },
                    |_| {},
                );
                if first == Some(entry) && found.replace(id).is_some() {
                    return Err(format!("two entry-list nodes name the entry {entry}"));
                }
            }
        }
        found.ok_or_else(|| format!("no entry-list node names the entry {entry}"))
    }

    fn node_text(&self, id: u32) -> Result<&'a str, String> {
        self.nodes
            .get(&id)
            .map(|&item| self.items[item].1)
            .ok_or_else(|| format!("kept text names !{id}, which the module does not define"))
    }

    /// The symbols the metadata of node `id` names, through every node it names, and those nodes.
    fn closure(&self, id: u32) -> Result<(Vec<&'a str>, BTreeSet<u32>), String> {
        let (mut symbols, mut nodes, mut pending) = (Vec::new(), BTreeSet::new(), vec![id]);
        while let Some(id) = pending.pop() {
            if nodes.insert(id) {
                references(
                    self.node_text(id)?,
                    |symbol| symbols.push(symbol),
                    |id| pending.push(id),
                );
            }
        }
        Ok((symbols, nodes))
    }
}

fn push_line(projection: &mut String, text: &str) {
    projection.push_str(text.trim_end_matches('\n'));
    projection.push('\n');
}

/// The module's top-level items. A function's item runs from its `define` line through the line
/// `}` that closes its body.
fn parse(ll: &str) -> Result<Parsed<'_>, String> {
    let mut parsed = Parsed {
        items: Vec::new(),
        symbols: HashMap::new(),
        nodes: HashMap::new(),
        runtime_started: Vec::new(),
    };
    let mut lines = ll.split_inclusive('\n').scan(0usize, |offset, line| {
        let start = *offset;
        *offset += line.len();
        Some((start, line))
    });
    let mut number = 0usize;
    while let Some((start, line)) = lines.next() {
        number += 1;
        // Top-level LLVM assembly is whitespace-insensitive, so an item is placed by its text
        // whatever its indentation.
        let text = line.trim();
        let first_line = number;
        let refuse = |why: &str| format!("line {first_line} ({text:?}): {why}");
        let mut end = start + line.len();
        let item = if text.is_empty() || text.starts_with(';') {
            Item::Comment
        } else if text.starts_with("source_filename ")
            || text.starts_with("target ")
            || text.starts_with('%')
            || text.starts_with('$')
            || text.starts_with("attributes #")
            || text.starts_with("module asm ")
        {
            Item::Module
        } else if let Some(rest) = text.strip_prefix('@') {
            let name = symbol_name(rest).ok_or_else(|| refuse("a global without a name"))?;
            define(&mut parsed.symbols, name, parsed.items.len(), &refuse)?;
            Item::Symbol
        } else if text.starts_with("declare ") || text.starts_with("define ") {
            let name = text
                .find('@')
                .and_then(|at| symbol_name(&text[at + 1..]))
                .ok_or_else(|| refuse("a function without a name"))?;
            if text.starts_with("define ") && !text.ends_with('}') {
                if !text.ends_with('{') {
                    return Err(refuse("a definition whose body does not open at its end"));
                }
                // The body runs to the line `}` that closes it.
                loop {
                    let Some((body_start, body_line)) = lines.next() else {
                        return Err(refuse("a body that never closes"));
                    };
                    number += 1;
                    end = body_start + body_line.len();
                    if body_line.trim() == "}" {
                        break;
                    }
                }
            }
            if text.starts_with("define ")
                && RUNTIME_STARTED_SECTIONS
                    .iter()
                    .any(|section| text.contains(section))
            {
                parsed.runtime_started.push(parsed.items.len());
            }
            define(&mut parsed.symbols, name, parsed.items.len(), &refuse)?;
            Item::Symbol
        } else if let Some(rest) = text.strip_prefix('!') {
            if rest.starts_with(|ch: char| ch.is_ascii_digit()) {
                let id = node_id(text)?;
                if parsed.nodes.insert(id, parsed.items.len()).is_some() {
                    return Err(refuse("a node defined twice"));
                }
                Item::Node(id)
            } else {
                let (name, list) = rest
                    .split_once(" = !{")
                    .ok_or_else(|| refuse("named metadata that is no node list"))?;
                let list = list
                    .strip_suffix('}')
                    .ok_or_else(|| refuse("a node list that does not close"))?;
                let operands = list
                    .split(',')
                    .map(str::trim)
                    .filter(|operand| !operand.is_empty())
                    .map(|operand| {
                        operand
                            .strip_prefix('!')
                            .and_then(|id| id.parse().ok())
                            .ok_or_else(|| refuse("a named-metadata operand that is no node"))
                    })
                    .collect::<Result<Vec<u32>, _>>()?;
                Item::Named { name, operands }
            }
        } else {
            return Err(refuse("no top-level item the projection places"));
        };
        parsed.items.push((item, &ll[start..end]));
    }
    Ok(parsed)
}

fn define<'a>(
    symbols: &mut HashMap<&'a str, usize>,
    name: &'a str,
    item: usize,
    refuse: &impl Fn(&str) -> String,
) -> Result<(), String> {
    if symbols.insert(name, item).is_some() {
        return Err(refuse("a symbol defined twice"));
    }
    Ok(())
}

/// The id of the unnamed node whose line is `text` (`!12 = ...`).
fn node_id(text: &str) -> Result<u32, String> {
    text.strip_prefix('!')
        .and_then(|rest| rest.split_once(' '))
        .and_then(|(id, _)| id.parse().ok())
        .ok_or_else(|| format!("{text:?} is no unnamed node"))
}

/// The symbol name at the start of `text` (after its `@`): quoted (`"persona::k"`) or bare.
fn symbol_name(text: &str) -> Option<&str> {
    if let Some(quoted) = text.strip_prefix('"') {
        return quoted.find('"').map(|end| &quoted[..end]);
    }
    let end = text
        .find(|ch: char| !(ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '$' | '-')))
        .unwrap_or(text.len());
    (end != 0).then(|| &text[..end])
}

/// Every symbol (`@name`) and unnamed node (`!N`) `text` names, outside its string literals
/// (`c"..."`, `!"..."`, attribute strings) and comments. LLVM spells a quote inside a string
/// `\22`, so a string ends at the next quote.
fn references<'a>(text: &'a str, mut symbol: impl FnMut(&'a str), mut node: impl FnMut(u32)) {
    let bytes = text.as_bytes();
    let mut at = 0;
    while at < bytes.len() {
        match bytes[at] {
            b'"' => {
                at += 1;
                while at < bytes.len() && bytes[at] != b'"' {
                    at += 1;
                }
                at += 1;
            }
            b';' => {
                while at < bytes.len() && bytes[at] != b'\n' {
                    at += 1;
                }
            }
            b'@' => {
                let rest = &text[at + 1..];
                match symbol_name(rest) {
                    Some(name) => {
                        symbol(name);
                        // A quoted name also skips its two quotes.
                        at += 1 + name.len() + if rest.starts_with('"') { 2 } else { 0 };
                    }
                    None => at += 1,
                }
            }
            b'!' => {
                let digits = bytes[at + 1..]
                    .iter()
                    .take_while(|byte| byte.is_ascii_digit())
                    .count();
                if digits != 0 {
                    if let Ok(id) = text[at + 1..at + 1 + digits].parse() {
                        node(id);
                    }
                }
                at += 1 + digits;
            }
            _ => at += 1,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
