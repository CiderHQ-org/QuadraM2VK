use crate::case::Stage;

pub mod calls;
pub(crate) mod fc_specialize;
pub mod meta;
pub(crate) mod projection;
pub mod reflect;
pub mod sanitize;
pub(crate) mod text;
pub(crate) mod visible_references;

pub use projection::entry_projection;

/// The stage label (`Kernel`) of the first entry list the module declares, reading the vertex,
/// fragment, kernel, object, then mesh list.
pub fn stage_label_from_ll(ll: &str) -> Option<&'static str> {
    [
        Stage::Vertex,
        Stage::Fragment,
        Stage::Kernel,
        Stage::Object,
        Stage::Mesh,
    ]
    .into_iter()
    .find(|stage| ll.contains(&format!("!air.{} =", stage.entry_list())))
    .map(Stage::metadata_label)
}

pub fn stage_from_ll(ll: &str) -> Stage {
    stage_label_from_ll(ll)
        .and_then(Stage::from_metadata_label)
        .unwrap_or(Stage::Kernel)
}

pub fn stage_name(stage: Stage) -> &'static str {
    stage.entry_list()
}

pub fn entry_name_from_ll(ll: &str) -> Option<String> {
    stage_entry_from_ll(ll)
        .map(|(_, entry)| entry)
        .or_else(|| first_define_name(ll))
}

/// The stage label and function of the first entry the module declares, reading its entry lists
/// in [`Stage::ALL`]'s order (`ENTRY_LISTS`).
pub fn stage_entry_from_ll(ll: &str) -> Option<(&'static str, String)> {
    for stage in ENTRY_LISTS {
        let needle = format!("!air.{} = !{{!", stage.entry_list());
        if let Some(pos) = ll.find(&needle) {
            let rest = &ll[pos + needle.len()..];
            let id: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if id.is_empty() {
                continue;
            }
            let node = format!("!{id} = !{{");
            if let Some(npos) = ll.find(&node) {
                let body = &ll[npos + node.len()..];
                if let Some(name) = symbol_after_ptr_at(body) {
                    return Some((stage.metadata_label(), name));
                }
            }
        }
    }
    None
}

/// The stages whose `!air.*` lists declare a module's entry points, in the order
/// [`stage_entry_from_ll`] reads them: every stage of the one vocabulary ([`Stage::ALL`]), so an
/// object or mesh function is a row like any other.
const ENTRY_LISTS: [Stage; 5] = Stage::ALL;

/// Whether the named metadata `name` (`air.kernel`) is one of the entry-point lists
/// ([`ENTRY_LISTS`]).
pub(crate) fn is_entry_list(name: &str) -> bool {
    name.strip_prefix("air.")
        .is_some_and(|key| ENTRY_LISTS.iter().any(|stage| stage.entry_list() == key))
}

/// The operand text of the module's `!air.<key>` list (`!0, !7`), if it declares one.
fn entry_list<'a>(ll: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("!air.{key} = !{{");
    let start = memchr::memmem::find(ll.as_bytes(), needle.as_bytes())? + needle.len();
    let list = &ll[start..];
    Some(&list[..list.find('}')?])
}

/// The node operands (`!N`) of an entry list's text.
fn list_operands(list: &str) -> impl Iterator<Item = &str> {
    list.split(',')
        .map(str::trim)
        .filter(|operand| operand.starts_with('!'))
}

/// How many entry points the module declares: the operands of its entry lists (`ENTRY_LISTS`),
/// the entry kinds [`stage_entry_from_ll`] reads. A module that
/// declares more than one is the module of several source rows, one per entry.
pub fn entry_count_from_ll(ll: &str) -> usize {
    ENTRY_LISTS
        .into_iter()
        .filter_map(|stage| entry_list(ll, stage.entry_list()))
        .map(|list| list_operands(list).count())
        .sum()
}

/// The stages whose entry lists the module declares with at least one entry, in [`Stage::ALL`]'s
/// order: what a stage filter of the harvest reads (`corpus-harvest --only-stages`).
pub fn listed_stages(ll: &str) -> Vec<Stage> {
    ENTRY_LISTS
        .into_iter()
        .filter(|stage| {
            entry_list(ll, stage.entry_list())
                .is_some_and(|list| list_operands(list).next().is_some())
        })
        .collect()
}

/// One entry point a module declares: the stage of the `!air.*` list that names it and the
/// function it names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleEntry {
    pub stage: &'static str,
    pub name: String,
}

/// Every entry point the module declares ([`entry_count_from_ll`]), in the order
/// [`stage_entry_from_ll`] reads them: the `!air.kernel`, `!air.vertex`, `!air.fragment`,
/// `!air.object`, then `!air.mesh` list, each in operand order. An operand whose node names no function, or a function declared
/// twice, is an error naming it.
pub fn module_entries(ll: &str) -> Result<Vec<ModuleEntry>, String> {
    let mut entries = Vec::<ModuleEntry>::new();
    for list_stage in ENTRY_LISTS {
        let (key, stage) = (list_stage.entry_list(), list_stage.metadata_label());
        let Some(list) = entry_list(ll, key) else {
            continue;
        };
        for operand in list_operands(list) {
            let node = format!("{operand} = !{{");
            let name = ll
                .find(&node)
                .and_then(|at| symbol_after_ptr_at(&ll[at + node.len()..]))
                .ok_or_else(|| format!("!air.{key} operand {operand} names no entry function"))?;
            if entries.iter().any(|entry| entry.name == name) {
                return Err(format!("the module declares the entry {name} twice"));
            }
            entries.push(ModuleEntry { stage, name });
        }
    }
    Ok(entries)
}

/// The text the facts of entry `entry` of the module `ll` are read from: the module itself when it
/// declares one entry, and otherwise the entry's projection of it ([`entry_projection`]), so no
/// fact of another entry is charged to this one.
pub fn entry_text<'a>(ll: &'a str, entry: &str) -> Result<std::borrow::Cow<'a, str>, String> {
    if entry_count_from_ll(ll) > 1 {
        entry_projection(ll, entry).map(std::borrow::Cow::Owned)
    } else {
        Ok(std::borrow::Cow::Borrowed(ll))
    }
}

fn first_define_name(ll: &str) -> Option<String> {
    for line in ll.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("define ") else {
            continue;
        };
        if let Some(at) = rest.find('@') {
            if let Some(name) = symbol_after_at(&rest[at..]) {
                return Some(name);
            }
        }
    }
    None
}

fn symbol_after_ptr_at(s: &str) -> Option<String> {
    // The entry node's first operand is the function pointer. Opaque-pointer AIR spells it
    // `ptr @k`; `xcrun metal -S -emit-llvm` still spells it `void (i32 addrspace(1)*, i32)* @k`,
    // where the pointer is the `*` closing the function type. Accept both, and take whichever
    // comes first so a later operand can never be read as the entry.
    let opaque = s.find("ptr @");
    let typed = s.find("* @");
    let p = match (opaque, typed) {
        (Some(a), Some(b)) => a.min(b),
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => return None,
    };
    symbol_after_at(s[p..s.len()].trim_start_matches(|c| c != '@'))
}

fn symbol_after_at(s: &str) -> Option<String> {
    let rest = s.strip_prefix('@')?;
    if let Some(rest) = rest.strip_prefix('"') {
        let mut out = String::new();
        let mut escaped = false;
        for ch in rest.chars() {
            if escaped {
                out.push(ch);
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => break,
                _ => out.push(ch),
            }
        }
        return (!out.is_empty()).then_some(out);
    }
    let name: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '.' || *c == '$')
        .collect();
    (!name.is_empty()).then_some(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stage is listed only by an entry list holding an entry: an empty list or another
    /// named metadata lists none.
    #[test]
    fn listed_stages_reads_every_nonempty_entry_list() {
        assert_eq!(listed_stages("define void @f() { ret void }"), []);
        assert_eq!(listed_stages("!air.mesh = !{}\n"), []);
        assert_eq!(listed_stages("!air.not_a_list = !{!0}\n"), []);
        assert_eq!(
            listed_stages("!air.mesh = !{!1}\n!air.object = !{!0}\n!air.kernel = !{!2}\n"),
            [Stage::Kernel, Stage::Object, Stage::Mesh]
        );
    }

    #[test]
    fn entry_count_sums_every_entry_list() {
        assert_eq!(entry_count_from_ll(""), 0);
        assert_eq!(
            entry_count_from_ll("!air.kernel = !{!0}\n!0 = !{ptr @k}"),
            1
        );
        assert_eq!(entry_count_from_ll("!air.kernel = !{!0, !7, !12}\n"), 3);
        assert_eq!(
            entry_count_from_ll("!air.vertex = !{!0, !4}\n!air.fragment = !{!9}\n"),
            3
        );
    }

    #[test]
    fn stage_detection_uses_air_metadata() {
        assert_eq!(stage_from_ll("!air.kernel = !{!0}"), Stage::Kernel);
        assert_eq!(stage_from_ll("!air.vertex = !{!0}"), Stage::Vertex);
        assert_eq!(stage_from_ll("!air.fragment = !{!0}"), Stage::Fragment);
        assert_eq!(stage_from_ll(""), Stage::Kernel);
    }

    #[test]
    fn entry_name_prefers_air_stage_metadata() {
        let ll = r#"
define void @helper() {
  ret void
}

define void @"persona::ksDepthDilate"(ptr addrspace(2) %0) {
  ret void
}

!air.kernel = !{!15}
!15 = !{ptr @"persona::ksDepthDilate", !16, !17}
!16 = !{}
!17 = !{!18}
!18 = !{i32 0, !"air.buffer", !"air.location_index", i32 0}
"#;

        assert_eq!(
            entry_name_from_ll(ll),
            Some("persona::ksDepthDilate".into())
        );
        assert_eq!(
            stage_entry_from_ll(ll),
            Some(("Kernel", "persona::ksDepthDilate".into()))
        );
    }

    #[test]
    fn stage_entry_requires_a_well_formed_stage_node() {
        let ll = "define void @k() { ret void }\n!air.kernel = !{!0}\n!0 = !{i32 1}";
        assert_eq!(stage_entry_from_ll(ll), None);
        assert_eq!(entry_name_from_ll(ll), Some("k".into()));
    }
}
