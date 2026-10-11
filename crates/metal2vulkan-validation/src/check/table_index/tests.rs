//! Synthetic `.ll` modules, assembled with `llvm-as`; each test skips visibly without it.

use super::*;
use crate::check::loop_free::assemble;

/// The decoded modules of `sources`, the entry's first; `None` (a visible skip) without
/// `llvm-as`.
fn decoded(test: &str, sources: &[(ModuleRole, &str)]) -> Option<Vec<(ModuleRole, DecodedModule)>> {
    let mut out = Vec::new();
    for (role, ll) in sources {
        match assemble(ll) {
            Ok(bitcode) => out.push((
                *role,
                DecodedModule::from_bitcode("synthetic", &bitcode).unwrap(),
            )),
            Err(why) if why.contains("unavailable") => {
                eprintln!("SKIP {test}: {why}");
                return None;
            }
            Err(why) => panic!("{test}: {why}"),
        }
    }
    Some(out)
}

/// The table accesses of entry `k`, whose parameter 0 is the table.
fn accesses(test: &str, sources: &[(ModuleRole, &str)]) -> Option<BTreeSet<(u32, u64)>> {
    let modules = decoded(test, sources)?;
    let borrowed = modules
        .iter()
        .map(|(role, module)| (*role, module))
        .collect::<Vec<_>>();
    Some(table_accesses("k", &borrowed, &BTreeSet::from([0])))
}

/// The entry builds an `array_ref`-like state holding the table, copies it (field by field, or
/// with `llvm.memcpy` under `MEMCPY`), and hands the copy to a visible-function stub.
fn entry(copy: &str) -> String {
    format!(
        "%ref = type {{ ptr, i64 }}\n\
         define void @k(ptr %table, float %v) {{\n\
         entry:\n\
         %s = alloca %ref, align 8\n\
         %f0 = getelementptr inbounds %ref, ptr %s, i64 0, i32 0\n\
         store ptr %table, ptr %f0, align 8\n\
         %copy = alloca %ref, align 8\n\
         {copy}\n\
         call void @postfix.MTL_VISIBLE_FN_REF(ptr %copy, float %v)\n\
         ret void\n\
         }}\n\
         declare void @postfix.MTL_VISIBLE_FN_REF(ptr, float) section \"air.externally_defined\"\n\
         declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)\n\
         !air.visible_function_references = !{{!4}}\n\
         !4 = !{{!\"air.visible_function_reference\", ptr @postfix.MTL_VISIBLE_FN_REF, !\"postfix\"}}\n"
    )
}

const FIELD_COPY: &str = "%t = load ptr, ptr %f0, align 8\n\
     %c0 = getelementptr inbounds %ref, ptr %copy, i64 0, i32 0\n\
     store ptr %t, ptr %c0, align 8";
const MEMCPY: &str = "call void @llvm.memcpy.p0.p0.i64(ptr %copy, ptr %s, i64 16, i1 false)";

/// A destination callback that writes `value` through table element `index` (a constant, or the
/// parameter `%i`).
fn postfix(index: &str) -> String {
    format!(
        "define void @postfix(ptr %state, float %value) {{\n\
         entry:\n\
         %data = load ptr, ptr %state, align 8\n\
         %slot = getelementptr inbounds ptr addrspace(1), ptr %data, i64 {index}\n\
         %dst = load ptr addrspace(1), ptr %slot, align 8\n\
         store float %value, ptr addrspace(1) %dst, align 4\n\
         ret void\n\
         }}\n"
    )
}

/// The shape of the qvm-t cases: the table reaches the linked callback through two thread-local
/// copies of its state and a visible-function stub, and the callback reads element 1.
#[test]
fn a_constant_index_reaches_the_table_through_private_copies_and_a_stub() {
    let test = "a_constant_index_reaches_the_table_through_private_copies_and_a_stub";
    for copy in [FIELD_COPY, MEMCPY] {
        let Some(found) = accesses(
            test,
            &[
                (ModuleRole::Entry, &entry(copy)),
                (ModuleRole::Linked, &postfix("1")),
            ],
        ) else {
            return;
        };
        assert_eq!(found, BTreeSet::from([(0, 8)]), "{copy}");
        assert_eq!(
            judge(&[(0, 4, 1)], &found),
            vec![
                "device-buffer array 4 binds 1 element, but the code the case runs reads \
                  element 1 of it with a constant index; bind 2 elements"
                    .to_string()
            ]
        );
        assert!(judge(&[(0, 4, 2)], &found).is_empty());
    }
}

/// An index that is data (a loop over the table) is not judged, a callback no call reaches is
/// not scanned, and without the linked module there is nothing to follow.
#[test]
fn only_reached_constant_indices_count() {
    let test = "only_reached_constant_indices_count";
    let dynamic = postfix("%i").replace("%slot =", "%i = fptoui float %value to i64\n%slot =");
    let Some(found) = accesses(
        test,
        &[
            (ModuleRole::Entry, &entry(FIELD_COPY)),
            (ModuleRole::Linked, &dynamic),
        ],
    ) else {
        return;
    };
    assert_eq!(found, BTreeSet::new());
    let unreached = postfix("1").replace("@postfix(", "@other(");
    let both = format!("{}{unreached}", postfix("0"));
    let Some(found) = accesses(
        test,
        &[
            (ModuleRole::Entry, &entry(FIELD_COPY)),
            (ModuleRole::Linked, &both),
        ],
    ) else {
        return;
    };
    assert_eq!(found, BTreeSet::from([(0, 0)]));
    let Some(found) = accesses(test, &[(ModuleRole::Entry, &entry(FIELD_COPY))]) else {
        return;
    };
    assert_eq!(found, BTreeSet::new());
}

/// In the entry itself: an element read directly, and one read through a pointer a defined
/// helper returns.
#[test]
fn direct_and_returned_table_pointers_are_followed() {
    let test = "direct_and_returned_table_pointers_are_followed";
    let ll = "define ptr @second(ptr %t) {\n\
              entry:\n\
              %p = getelementptr inbounds ptr addrspace(1), ptr %t, i64 1\n\
              ret ptr %p\n\
              }\n\
              define void @k(ptr %table, float %v) {\n\
              entry:\n\
              %third = getelementptr inbounds ptr addrspace(1), ptr %table, i64 2\n\
              %a = load ptr addrspace(1), ptr %third, align 8\n\
              store float %v, ptr addrspace(1) %a, align 4\n\
              %p = call ptr @second(ptr %table)\n\
              %b = load ptr addrspace(1), ptr %p, align 8\n\
              store float %v, ptr addrspace(1) %b, align 4\n\
              ret void\n\
              }\n";
    let Some(found) = accesses(test, &[(ModuleRole::Entry, ll)]) else {
        return;
    };
    assert_eq!(found, BTreeSet::from([(0, 8), (0, 16)]));
    assert_eq!(
        judge(&[(0, 0, 1)], &found),
        vec![
            "device-buffer array 0 binds 1 element, but the code the case runs reads \
              elements 1, 2 of it with a constant index; bind 3 elements"
                .to_string()
        ]
    );
}
