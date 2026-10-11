//! The descriptive `.ll` CFG-cycle fact (`has_cfg_cycle`) that structural triage persists.
//!
//! This is a line-based scan of sanitized text, kept only because the triage analyzer
//! (`ANALYZER_ABI`) records it as a descriptive structural fact. It decides nothing about
//! execution safety: the checker's classifier is [`super::loop_free`], over decoded AIR.

use std::collections::HashMap;

/// Whether any function in `ll` has a CFG cycle. This is the CFG half of the classifier only; it
/// ignores calls and terminator coverage and is kept as a descriptive structural fact.
///
/// It shares the scanner's blind spot for labels it cannot read (for example the comment form
/// `; <label>:5:`): an edge to such a block is dropped, so a loop through it reads as acyclic
/// here. The typed classifier ([`super::loop_free`]) decodes
/// the bitcode instead; use it, never this fact, for execution safety.
pub fn has_cfg_cycle(ll: &str) -> bool {
    let mut graph = HashMap::<String, Vec<String>>::new();
    let mut current = None::<String>;
    let mut in_function = false;
    let mut switch_cases = false;
    for line in ll.lines() {
        let line = line.trim();
        if line.starts_with("define ") {
            in_function = true;
            switch_cases = false;
            current = Some(ENTRY_PLACEHOLDER.into());
            graph.entry(ENTRY_PLACEHOLDER.into()).or_default();
            continue;
        }
        if !in_function {
            continue;
        }
        if line == "}" {
            if graph_has_cycle(&graph) {
                return true;
            }
            graph.clear();
            current = None;
            in_function = false;
            continue;
        }
        if let Some(label) = parse_label(line) {
            switch_cases = false;
            current = Some(label.clone());
            graph.entry(label).or_default();
            continue;
        }
        if let Some(block) = current.as_ref() {
            let destinations = branch_destinations(line, switch_cases);
            if line.split_whitespace().next() == Some("switch") {
                switch_cases = true;
            }
            if line.starts_with(']') {
                switch_cases = false;
            }
            graph.entry(block.clone()).or_default().extend(destinations);
        }
    }
    graph_has_cycle(&graph)
}

/// Name of the implicit entry block before any label line.
const ENTRY_PLACEHOLDER: &str = "<entry>";

fn parse_label(line: &str) -> Option<String> {
    let label = line.split(';').next()?.trim().strip_suffix(':')?;
    if label.is_empty() || label.contains(char::is_whitespace) {
        return None;
    }
    Some(label.trim_matches('"').into())
}

fn branch_destinations(line: &str, switch_cases: bool) -> Vec<String> {
    let instruction = line
        .split_once(" = ")
        .map_or(line, |(_, instruction)| instruction.trim_start());
    let opcode = instruction.split_whitespace().next().unwrap_or_default();
    if !switch_cases
        && !matches!(
            opcode,
            "br" | "switch"
                | "indirectbr"
                | "invoke"
                | "callbr"
                | "catchswitch"
                | "catchret"
                | "cleanupret"
        )
    {
        return Vec::new();
    }
    let bytes = instruction.as_bytes();
    let mut destinations = Vec::new();
    let mut offset = 0;
    while let Some(relative) = instruction[offset..].find("label %") {
        let start = offset + relative + "label %".len();
        let tail = &instruction[start..];
        let (name, consumed) = if let Some(quoted) = tail.strip_prefix('"') {
            match quoted.find('"') {
                Some(end) => (&quoted[..end], end + 2),
                None => break,
            }
        } else {
            let end = tail
                .find(|ch: char| {
                    !(ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '$' | '-'))
                })
                .unwrap_or(tail.len());
            (&tail[..end], end)
        };
        if !name.is_empty() {
            destinations.push(name.into());
        }
        offset = start + consumed;
        if offset >= bytes.len() {
            break;
        }
    }
    destinations
}

/// Kahn's algorithm; iterative, so deep graphs do not grow the stack.
fn graph_has_cycle(graph: &HashMap<String, Vec<String>>) -> bool {
    first_cyclic_node(graph).is_some()
}

/// A node Kahn's algorithm could not remove (on or downstream of a cycle), smallest by name so
/// the reported evidence is deterministic.
fn first_cyclic_node(graph: &HashMap<String, Vec<String>>) -> Option<String> {
    let mut incoming: HashMap<&str, usize> = graph.keys().map(|node| (node.as_str(), 0)).collect();
    for edges in graph.values() {
        for next in edges {
            if let Some(count) = incoming.get_mut(next.as_str()) {
                *count += 1;
            }
        }
    }
    let mut ready: Vec<_> = incoming
        .iter()
        .filter_map(|(&node, &count)| (count == 0).then_some(node))
        .collect();
    while let Some(node) = ready.pop() {
        incoming.remove(node);
        for next in &graph[node] {
            if let Some(count) = incoming.get_mut(next.as_str()) {
                *count -= 1;
                if *count == 0 {
                    ready.push(next.as_str());
                }
            }
        }
    }
    incoming.keys().min().map(|node| node.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn back_edges_including_switch_cases_are_cycles() {
        let acyclic = "define void @k() {\nentry:\n br label %done\ndone:\n ret void\n}";
        let cyclic = "define void @k() {\nentry:\n br label %loop\nloop: ; preds = %entry, %loop\n br i1 true, label %loop, label %done\ndone:\n ret void\n}";
        assert!(!has_cfg_cycle(acyclic));
        assert!(has_cfg_cycle(cyclic));
        let switch = "define void @k(i32 %value) {\nentry:\n switch i32 %value, label %done [\n i32 0, label %entry\n ]\ndone:\n ret void\n}";
        assert!(has_cfg_cycle(switch));
        assert!(!has_cfg_cycle(
            &switch.replace("i32 0, label %entry", "i32 0, label %done")
        ));
        // The documented blind spot: an unreadable comment-form label drops the edge.
        let commented = "define void @k() {\n  br label %1\n\n; <label>:1:\n  br i1 true, label %1, label %2\n\n; <label>:2:\n  ret void\n}";
        assert!(!has_cfg_cycle(commented));
    }

    #[test]
    fn deep_graphs_do_not_grow_the_stack() {
        let mut graph = (0..100_000)
            .map(|index| (index.to_string(), vec![(index + 1).to_string()]))
            .collect::<HashMap<_, _>>();
        assert!(!graph_has_cycle(&graph));
        graph.insert("100000".into(), vec!["0".into(), "0".into()]);
        assert!(graph_has_cycle(&graph));
        graph.insert("100000".into(), vec![]);
        assert!(!graph_has_cycle(&graph));
        graph.insert("disconnected".into(), vec!["disconnected".into()]);
        assert!(graph_has_cycle(&graph));
    }
}
