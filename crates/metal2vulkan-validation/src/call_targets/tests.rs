//! Synthetic `.ll` modules, assembled with `llvm-as`; each test skips visibly without it.

use super::*;
use crate::check::loop_free::{assemble, DecodedModule};

/// The decoded `ll`; `None` (a visible skip) without `llvm-as`.
fn decoded(test: &str, ll: &str) -> Option<DecodedModule> {
    match assemble(ll) {
        Ok(bitcode) => Some(DecodedModule::from_bitcode("synthetic", &bitcode).unwrap()),
        Err(why) if why.contains("unavailable") => {
            eprintln!("SKIP {test}: {why}");
            None
        }
        Err(why) => panic!("{test}: {why}"),
    }
}

/// The names of `funcs`.
fn names(module: &Module, funcs: &BTreeSet<FuncId>) -> Vec<String> {
    funcs
        .iter()
        .map(|&f| {
            let name = module.symbols().get(module.func(f).unwrap().name).unwrap();
            String::from_utf8_lossy(name).into_owned()
        })
        .collect()
}

/// Per closed call of `module`, its targets' names; and the escaped functions' names.
fn analysed(module: &Module) -> (Vec<Vec<String>>, Vec<String>) {
    let closed = closed_calls(module);
    let calls = closed
        .calls
        .values()
        .map(|targets| names(module, targets))
        .collect();
    (calls, names(module, &closed.escaped))
}

const TARGETS: &str = "define internal void @a(i32 %n) {\n ret void\n}\n\
define internal void @b(i32 %n) {\n ret void\n}\n\
define internal void @c(float %x) {\n ret void\n}\n\
declare void @d(i32)\n";

/// A kernel choosing `@a`, `@b` or nothing by `%p`/`%q`, testing the choice against null, and
/// calling it; `extra` is spliced in before the call.
fn chooser(extra: &str) -> String {
    format!(
        "{TARGETS}define void @k(i1 %p, i1 %q, i32 %n, ptr %out) {{\n\
entry:\n %s0 = select i1 %p, ptr @a, ptr null\n %s1 = select i1 %q, ptr @b, ptr %s0\n\
 %none = icmp eq ptr %s1, null\n br i1 %none, label %done, label %call\n\
call:\n{extra} call void %s1(i32 %n)\n br label %done\n\
done:\n ret void\n}}\n"
    )
}

#[test]
fn a_select_tree_of_function_addresses_and_null_is_a_closed_callee() {
    let test = "a_select_tree_of_function_addresses_and_null_is_a_closed_callee";
    let Some(module) = decoded(test, &chooser("")) else {
        return;
    };
    let (calls, escaped) = analysed(module.module());
    assert_eq!(calls, vec![vec!["a".to_string(), "b".to_string()]]);
    assert!(escaped.is_empty(), "{escaped:?}");
    // A function address called directly through a pointer is closed too.
    let direct = format!(
        "{TARGETS}define void @k(i32 %n) {{\n call void @a(i32 %n)\n call void (i32) @b(i32 %n)\n ret void\n}}\n"
    );
    let Some(module) = decoded(test, &direct) else {
        return;
    };
    assert!(analysed(module.module()).1.is_empty());
}

#[test]
fn an_address_used_beyond_a_closed_callee_escapes() {
    let test = "an_address_used_beyond_a_closed_callee_escapes";
    let cases = [
        // The chosen callee stored: the tree escapes, so the call is open and both leaves escape.
        (" store ptr %s1, ptr %out\n", vec![], vec!["a", "b"]),
        // Passed to the call it chooses: the same.
        ("", vec![], vec!["a", "b"]),
        // One leaf passed elsewhere: that leaf escapes, and the call stays closed.
        (
            " call void @d(i32 ptrtoint (ptr @a to i32))\n",
            vec![vec!["a", "b"]],
            vec!["a"],
        ),
        // A leaf stored on its own: it escapes; the call stays closed.
        (" store ptr @b, ptr %out\n", vec![vec!["a", "b"]], vec!["b"]),
        // The tree compared with something other than null.
        (" %same = icmp eq ptr %s1, %out\n", vec![], vec!["a", "b"]),
    ];
    for (index, (extra, calls, escaped)) in cases.into_iter().enumerate() {
        let ll = if index == 1 {
            chooser("").replace("call void %s1(i32 %n)", "call void %s1(ptr %s1)")
        } else {
            chooser(extra)
        };
        let Some(module) = decoded(test, &ll) else {
            return;
        };
        let (got_calls, got_escaped) = analysed(module.module());
        let calls = calls
            .into_iter()
            .map(|c| c.into_iter().map(String::from).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        if index == 1 {
            // The call's own type changed (it passes a pointer), so its leaves no longer match it.
            assert!(got_calls.is_empty(), "case {index}: {got_calls:?}");
        } else {
            assert_eq!(got_calls, calls, "case {index}");
        }
        assert_eq!(got_escaped, escaped, "case {index}");
    }
    // A global initializer takes an address.
    let global = format!("@slot = global ptr @a\n{}", chooser(""));
    let Some(module) = decoded(test, &global) else {
        return;
    };
    assert_eq!(analysed(module.module()).1, vec!["a".to_string()]);
}

#[test]
fn a_callee_the_tree_cannot_name_is_open() {
    let test = "a_callee_the_tree_cannot_name_is_open";
    let cases = [
        // A leaf of another type than the call's.
        chooser("").replace("ptr @b, ptr %s0", "ptr @c, ptr %s0"),
        // A leaf the module only declares.
        chooser("").replace("ptr @b, ptr %s0", "ptr @d, ptr %s0"),
        // A leaf loaded from memory.
        chooser("").replace(
            "%s1 = select i1 %q, ptr @b, ptr %s0",
            "%l = load ptr, ptr %out\n %s1 = select i1 %q, ptr %l, ptr %s0",
        ),
        // A phi merging the choice.
        chooser("")
            .replace(
                "call:\n",
                "call:\n br label %merge\nmerge:\n %m = phi ptr [ %s1, %call ]\n",
            )
            .replace("call void %s1(i32 %n)", "call void %m(i32 %n)"),
    ];
    for (index, ll) in cases.iter().enumerate() {
        let Some(module) = decoded(test, ll) else {
            return;
        };
        let (calls, escaped) = analysed(module.module());
        assert!(calls.is_empty(), "case {index}: {calls:?}");
        assert!(
            escaped.contains(&"a".to_string()),
            "case {index}: {escaped:?}"
        );
    }
}
