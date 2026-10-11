//! A buffer's access, tightened from the entry body: the old engine's refinement.
//!
//! A buffer parameter whose SSA name the entry body never mentions, or that LLVM marks
//! `readnone`, is unused; one LLVM marks `readonly` or `writeonly` is read-only or write-only.
//! Otherwise the declared access stands. Both facts are the compiler's own: the attributes of the
//! entry definition's parameter, and whether the body names it at all. The old engine read them
//! after lowering asynchronous and dynamic-length copies, which replaces each such call with a
//! call over the same operands; that leaves the definition's attributes alone and names the same
//! parameters, so the body read here, unlowered, gives the same answer.

use super::{ResourceAccess, ResourceKind, ShaderInterface};

/// Tighten declared buffer access from the entry definition's parameter attributes and uses.
pub(super) fn refine_buffer_access_from_entry(interface: &mut ShaderInterface, ll: &str) {
    let Some(entry) = interface.entry_point.as_deref() else {
        return;
    };
    let Some((args, body)) = llvm_entry_args_and_body(ll, entry) else {
        return;
    };
    for binding in &mut interface.bindings {
        if !matches!(
            binding.kind,
            ResourceKind::Buffer | ResourceKind::ThreadgroupBuffer
        ) {
            continue;
        }
        let Some(param_index) = binding
            .param_index
            .and_then(|idx| usize::try_from(idx).ok())
        else {
            continue;
        };
        let Some(arg) = args.get(param_index) else {
            continue;
        };
        let Some(name) = percent_tokens(arg).last().copied() else {
            continue;
        };
        if !ssa_token_occurs(body, name) || llvm_arg_has_attribute(arg, "readnone") {
            binding.access = Some(ResourceAccess::Unused);
        } else if llvm_arg_has_attribute(arg, "readonly") {
            binding.access = Some(ResourceAccess::ReadOnly);
        } else if llvm_arg_has_attribute(arg, "writeonly") {
            binding.access = Some(ResourceAccess::WriteOnly);
        }
    }
}

fn llvm_entry_args_and_body<'a>(ll: &'a str, entry: &str) -> Option<(Vec<&'a str>, &'a str)> {
    let plain = format!("@{entry}(");
    let quoted = format!("@\"{entry}\"(");
    let start = ll.lines().position(|line| {
        line.trim_start().starts_with("define ")
            && (line.contains(&plain) || line.contains(&quoted))
    })?;
    let byte_start = ll
        .lines()
        .take(start)
        .map(|line| line.len() + 1)
        .sum::<usize>();
    let function = &ll[byte_start..];
    // Search from the entry symbol, rather than taking the first `{` / `(` in the header: an LLVM
    // function may return an aggregate such as `<{ i32, i32 }>`, and those delimiters precede its
    // parameter list.
    let symbol = function.find(&plain).or_else(|| function.find(&quoted))?;
    let open = function[symbol..].find('(')? + symbol;
    let mut depth = 0u32;
    let mut close = None;
    for (offset, character) in function[open..].char_indices() {
        match character {
            '(' => depth += 1,
            ')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    close = Some(open + offset);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close?;
    let header_end = function[close + 1..].find('{')? + close + 1;
    let args = split_top_level_llvm(&function[open + 1..close]);
    let tail = &function[header_end + 1..];
    let body_end = tail.find("\n}").unwrap_or(tail.len());
    Some((args, &tail[..body_end]))
}

fn split_top_level_llvm(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut depth = 0i32;
    for (index, character) in text.char_indices() {
        match character {
            '(' | '[' | '{' | '<' => depth += 1,
            ')' | ']' | '}' | '>' => depth -= 1,
            ',' if depth == 0 => {
                out.push(text[start..index].trim());
                start = index + 1;
            }
            _ => {}
        }
    }
    if !text[start..].trim().is_empty() {
        out.push(text[start..].trim());
    }
    out
}

fn percent_tokens(text: &str) -> Vec<&str> {
    text.split('%')
        .skip(1)
        .filter_map(|tail| {
            let token = tail
                .split(|character: char| {
                    !(character.is_ascii_alphanumeric() || character == '_' || character == '.')
                })
                .next()?;
            (!token.is_empty()).then_some(token)
        })
        .collect()
}

fn ssa_token_occurs(text: &str, name: &str) -> bool {
    percent_tokens(text).contains(&name)
}

fn llvm_arg_has_attribute(arg: &str, attribute: &str) -> bool {
    arg.split(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
        .any(|token| token == attribute)
}
