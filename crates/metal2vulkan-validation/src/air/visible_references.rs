//! AIR's direct visible-function references (`.MTL_VISIBLE_FN_REF` stubs and the
//! `air.visible_function_reference` metadata naming each stub's logical symbol), and the check
//! that a case's authored linked modules close them. Validation's copy of the errors the old
//! engine's linker raised for them, without the linking.

use std::collections::{HashMap, HashSet};

/// One authored visible-function reference: the logical symbol and the module text defining it.
pub(crate) struct AuthoredReference<'a> {
    pub(crate) symbol: &'a str,
    pub(crate) module_ll: &'a str,
}

/// Whether the authored references close `module`'s visible-function references: every authored
/// reference is defined by its module and named once, and every reference `module` makes, and
/// every reference the modules it reaches make in turn, has an authored module. The old engine's
/// linker refused a case with the first of these errors.
pub(crate) fn validate_closure(
    module: &str,
    authored: &[AuthoredReference<'_>],
) -> Result<(), String> {
    let mut symbols = HashSet::new();
    for reference in authored {
        if !module_defines(reference.module_ll, reference.symbol) {
            return Err(format!(
                "linked module does not define authored visible function reference {:?}",
                reference.symbol
            ));
        }
        if !symbols.insert(reference.symbol) {
            return Err(format!(
                "duplicate authored visible function reference {:?}",
                reference.symbol
            ));
        }
    }
    let by_symbol = authored
        .iter()
        .map(|reference| (reference.symbol, reference))
        .collect::<HashMap<_, _>>();
    let mut visited = HashSet::<&str>::new();
    let mut pending = Vec::new();
    resolve(module, &by_symbol, &mut pending)?;
    while let Some(reference) = pending.pop() {
        if visited.insert(reference.module_ll) {
            resolve(reference.module_ll, &by_symbol, &mut pending)?;
        }
    }
    Ok(())
}

/// Queue the authored module of every visible-function reference `module` makes.
fn resolve<'a>(
    module: &str,
    authored: &HashMap<&str, &'a AuthoredReference<'a>>,
    pending: &mut Vec<&'a AuthoredReference<'a>>,
) -> Result<(), String> {
    for symbol in reference_symbols(module)? {
        let dependency = authored.get(symbol.as_str()).ok_or_else(|| {
            format!("AIR visible function reference {symbol:?} has no authored linked module")
        })?;
        llvm_global(&symbol)?;
        pending.push(dependency);
    }
    Ok(())
}

/// The logical symbols `module`'s `air.visible_function_reference` metadata names, in metadata
/// order, each stub once; a stub mapped to two symbols, or a malformed entry, is an error. On an
/// entry's projection (`air::entry_text`) these are the references the entry reaches, which
/// classification describes (`StructuralTriage::linked_functions`).
pub(crate) fn reference_symbols(module: &str) -> Result<Vec<String>, String> {
    const MARKER: &str = "!\"air.visible_function_reference\", ptr ";
    let mut symbols = Vec::new();
    let mut stubs = HashMap::<String, String>::new();
    for line in module.lines() {
        let Some(marker) = line.find(MARKER) else {
            continue;
        };
        let body = &line[marker + MARKER.len()..];
        let at = body
            .find('@')
            .ok_or_else(|| format!("AIR visible-function reference has no stub global: {line}"))?;
        let stub_end = metadata_global_end(body, at).ok_or_else(|| {
            format!("AIR visible-function reference has malformed stub global: {line}")
        })?;
        let stub_global = body[at..stub_end].to_string();
        let rest = &body[stub_end..];
        let name_start = rest.find(", !\"").ok_or_else(|| {
            format!("AIR visible-function reference has no logical symbol: {line}")
        })? + 4;
        let (encoded, _) = quoted_string(&rest[name_start..]).ok_or_else(|| {
            format!("AIR visible-function reference has malformed logical symbol: {line}")
        })?;
        let symbol = decode_string(encoded)?;
        if let Some(previous) = stubs.insert(stub_global.clone(), symbol.clone()) {
            if previous != symbol {
                return Err(format!(
                    "AIR visible-function stub {stub_global} maps to both {previous:?} and {symbol:?}"
                ));
            }
            continue;
        }
        symbols.push(symbol);
    }
    Ok(symbols)
}

fn metadata_global_end(text: &str, at: usize) -> Option<usize> {
    if text.as_bytes().get(at) != Some(&b'@') {
        return None;
    }
    if text.as_bytes().get(at + 1) != Some(&b'\"') {
        return text[at..]
            .find(|character: char| character == ',' || character.is_ascii_whitespace())
            .map(|relative| at + relative);
    }
    let (_, consumed) = quoted_string(&text[at + 2..])?;
    Some(at + 2 + consumed)
}

/// The encoded body of an LLVM quoted string and the bytes consumed, the closing quote included.
fn quoted_string(text: &str) -> Option<(&str, usize)> {
    let mut escaped = false;
    for (index, byte) in text.bytes().enumerate() {
        if byte == b'\"' && !escaped {
            return Some((&text[..index], index + 1));
        }
        escaped = byte == b'\\' && !escaped;
        if byte != b'\\' {
            escaped = false;
        }
    }
    None
}

fn decode_string(encoded: &str) -> Result<String, String> {
    let bytes = encoded.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] == b'\\' {
            let pair = bytes
                .get(index + 1..index + 3)
                .ok_or_else(|| format!("unterminated LLVM string escape in {encoded:?}"))?;
            let hex = std::str::from_utf8(pair)
                .ok()
                .and_then(|pair| u8::from_str_radix(pair, 16).ok())
                .ok_or_else(|| format!("invalid LLVM string escape in {encoded:?}"))?;
            decoded.push(hex);
            index += 3;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(decoded).map_err(|_| format!("non-UTF-8 LLVM symbol in {encoded:?}"))
}

/// The LLVM global spelling of a function symbol: bare when LLVM allows it, quoted otherwise.
fn llvm_global(symbol: &str) -> Result<String, String> {
    if symbol.is_empty() || symbol.contains(['\n', '\r', '\0']) {
        return Err(format!("invalid linked LLVM function symbol {symbol:?}"));
    }
    if symbol
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'$' | b'-'))
    {
        Ok(format!("@{symbol}"))
    } else {
        Ok(format!(
            "@\"{}\"",
            symbol.replace('\\', "\\5C").replace('"', "\\22")
        ))
    }
}

/// Whether `module` defines the function `symbol`.
fn module_defines(module: &str, symbol: &str) -> bool {
    llvm_global(symbol).is_ok_and(|global| {
        module.lines().any(|line| {
            let line = line.trim_start();
            line.starts_with("define ") && line.contains(&format!("{global}("))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENTRY: &str = r#"@k.MTL_VISIBLE_FN_REF = external global ptr
define void @k() {
  ret void
}
!air.visible_function_references = !{!0}
!0 = !{!"air.visible_function_reference", ptr @k.MTL_VISIBLE_FN_REF, !"helper"}
"#;
    const HELPER: &str = "define float @helper(float %x) {\n  ret float %x\n}\n";

    #[test]
    fn an_authored_module_closes_the_reference() {
        let authored = [AuthoredReference {
            symbol: "helper",
            module_ll: HELPER,
        }];
        assert_eq!(validate_closure(ENTRY, &authored), Ok(()));
    }

    #[test]
    fn a_reference_without_an_authored_module_is_refused() {
        assert_eq!(
            validate_closure(ENTRY, &[]),
            Err("AIR visible function reference \"helper\" has no authored linked module".into())
        );
    }

    #[test]
    fn a_module_that_does_not_define_its_symbol_is_refused() {
        let authored = [AuthoredReference {
            symbol: "helper",
            module_ll: "define void @other() {\n  ret void\n}\n",
        }];
        assert!(validate_closure(ENTRY, &authored)
            .unwrap_err()
            .contains("does not define authored visible function reference"));
    }

    #[test]
    fn a_reference_a_linked_module_makes_must_close_too() {
        let helper = format!(
            "{HELPER}@h.MTL_VISIBLE_FN_REF = external global ptr\n!0 = !{{!\"air.visible_function_reference\", ptr @h.MTL_VISIBLE_FN_REF, !\"deeper\"}}\n"
        );
        let authored = [AuthoredReference {
            symbol: "helper",
            module_ll: &helper,
        }];
        assert_eq!(
            validate_closure(ENTRY, &authored),
            Err("AIR visible function reference \"deeper\" has no authored linked module".into())
        );
    }
}
