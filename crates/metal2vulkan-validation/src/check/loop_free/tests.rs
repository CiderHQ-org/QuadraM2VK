//! Synthetic `.ll` modules, assembled with `llvm-as` and classified through the typed decoder.
//! Each test skips visibly when `llvm-as` is not installed.

use super::*;
use ModuleRole::{Entry, Intersection, Linked};

/// Whether `llvm-as` is available; prints a visible skip line when it is not.
fn assembler(test: &str) -> bool {
    match assemble("define void @probe() {\n ret void\n}\n") {
        Ok(_) => true,
        Err(why) => {
            eprintln!("SKIP {test}: {why}");
            false
        }
    }
}

fn source(ll: &str) -> ModuleSource<'_> {
    ModuleSource {
        label: "synthetic",
        air_ll: ll,
        blob_b64: None,
    }
}

fn modules(entry: &str, sources: &[(ModuleRole, &str)]) -> LoopClass {
    classify_sources(entry, sources.iter().map(|&(role, ll)| (role, source(ll))))
}

fn class(ll: &str) -> LoopClass {
    modules("k", &[(Entry, ll)])
}

fn kind(ll: &str) -> Option<&'static str> {
    class(ll).evidence().map(CycleEvidence::kind)
}

fn cfg_cycle(function: &str) -> LoopClass {
    LoopClass::Cyclic(CycleEvidence::CfgCycle {
        function: function.into(),
    })
}

fn call_cycle(function: &str) -> LoopClass {
    LoopClass::Cyclic(CycleEvidence::CallCycle {
        function: function.into(),
    })
}

#[test]
fn back_edges_are_cfg_cycles() {
    if !assembler("back_edges_are_cfg_cycles") {
        return;
    }
    let acyclic = "define void @k() {\nentry:\n br label %done\ndone:\n ret void\n}";
    assert_eq!(class(acyclic), LoopClass::LoopFree);
    let cyclic = "define void @k(i1 %c) {\nentry:\n br label %loop\nloop:\n br i1 %c, label %loop, label %done\ndone:\n ret void\n}";
    assert_eq!(class(cyclic), cfg_cycle("k"));
    let switch = "define void @k(i32 %value) {\nentry:\n br label %head\nhead:\n switch i32 %value, label %done [\n i32 0, label %head\n ]\ndone:\n ret void\n}";
    assert_eq!(class(switch), cfg_cycle("k"));
    let forward = switch.replace("i32 0, label %head", "i32 0, label %done");
    assert_eq!(class(&forward), LoopClass::LoopFree);
}

#[test]
fn cycles_among_unreachable_blocks_do_not_execute() {
    if !assembler("cycles_among_unreachable_blocks_do_not_execute") {
        return;
    }
    // The line scanner refused this; the typed CFG proves the loop dead.
    let dead = "define void @k() {\nentry:\n ret void\ndead:\n br label %dead\n}";
    assert_eq!(class(dead), LoopClass::LoopFree);
}

#[test]
fn rows_without_decodable_bitcode_are_refused() {
    if !assembler("rows_without_decodable_bitcode_are_refused") {
        return;
    }
    assert_eq!(kind("define void @k() {"), Some("undecoded"));
    let bad_blob = classify_sources(
        "k",
        [(
            Entry,
            ModuleSource {
                label: "blob",
                air_ll: "define void @k() {\n ret void\n}",
                blob_b64: Some("bm90IGJpdGNvZGU="),
            },
        )],
    );
    assert_eq!(
        bad_blob.evidence().map(CycleEvidence::kind),
        Some("undecoded")
    );
    // The retained blob wins over the text: the text is never read when a blob exists.
    let assembled = assemble("define void @k() {\n ret void\n}").unwrap();
    let blob = {
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD.encode(assembled)
    };
    let from_blob = classify_sources(
        "k",
        [(
            Entry,
            ModuleSource {
                label: "blob",
                air_ll: "not LLVM text",
                blob_b64: Some(&blob),
            },
        )],
    );
    assert_eq!(from_blob, LoopClass::LoopFree);
}

#[test]
fn the_entry_must_exist_and_only_reachable_functions_count() {
    if !assembler("the_entry_must_exist_and_only_reachable_functions_count") {
        return;
    }
    let ll = "define void @k() {\nentry:\n ret void\n}\ndefine void @other(i1 %c) {\nentry:\n br label %loop\nloop:\n br i1 %c, label %loop, label %done\ndone:\n call void @missing()\n call void @other(i1 %c)\n ret void\n}\ndeclare void @missing()";
    assert_eq!(class(ll), LoopClass::LoopFree);
    assert_eq!(modules("other", &[(Entry, ll)]), cfg_cycle("other"));
    assert_eq!(
        modules("absent", &[(Entry, ll)]),
        LoopClass::Cyclic(CycleEvidence::MissingEntry {
            entry: "absent".into()
        })
    );
}

const HELPERS: &str = "define internal float @square(float %x) {\nentry:\n %y = fmul float %x, %x\n ret float %y\n}\n\n";

#[test]
fn loop_free_calls_to_defined_functions_and_intrinsics_are_loop_free() {
    if !assembler("loop_free_calls_to_defined_functions_and_intrinsics_are_loop_free") {
        return;
    }
    let module = format!(
        "{HELPERS}define void @k(ptr addrspace(1) %out) {{\nentry:\n %a = tail call float @square(float 2.0)\n %b = call fast float @llvm.fabs.f32(float %a)\n store float %b, ptr addrspace(1) %out\n call void @\"quoted helper\"()\n ret void\n}}\n\ndefine internal void @\"quoted helper\"() {{\n ret void\n}}\n\ndeclare float @llvm.fabs.f32(float)\n"
    );
    assert_eq!(class(&module), LoopClass::LoopFree);
}

#[test]
fn self_and_mutual_recursion_are_call_cycles() {
    if !assembler("self_and_mutual_recursion_are_call_cycles") {
        return;
    }
    let direct = "define void @k() {\nentry:\n call void @k()\n ret void\n}";
    assert_eq!(class(direct), call_cycle("k"));
    let mutual = "define void @k() {\nentry:\n call void @even(i32 3)\n ret void\n}\ndefine internal void @even(i32 %n) {\nentry:\n call void @odd(i32 %n)\n ret void\n}\ndefine internal void @odd(i32 %n) {\nentry:\n tail call void @even(i32 %n)\n ret void\n}";
    assert_eq!(class(mutual), call_cycle("even"));
    // Calls in a loop-free DAG (two paths to one helper) are not a cycle.
    let diamond = "define void @k() {\nentry:\n call void @a()\n call void @b()\n ret void\n}\ndefine internal void @a() {\n call void @c()\n ret void\n}\ndefine internal void @b() {\n call void @c()\n ret void\n}\ndefine internal void @c() {\n ret void\n}";
    assert_eq!(class(diamond), LoopClass::LoopFree);
}

#[test]
fn call_graph_spans_linked_modules() {
    if !assembler("call_graph_spans_linked_modules") {
        return;
    }
    let entry =
        "define void @k() {\nentry:\n call void @visible()\n ret void\n}\ndeclare void @visible()";
    let linked =
        "define void @visible() {\nentry:\n call void @k()\n ret void\n}\ndeclare void @k()";
    assert_eq!(
        class(entry),
        LoopClass::Cyclic(CycleEvidence::UnresolvedCall {
            function: "k".into(),
            callee: "visible".into()
        })
    );
    assert_eq!(
        modules("k", &[(Entry, entry), (Linked, linked)])
            .evidence()
            .map(CycleEvidence::kind),
        Some("call_cycle")
    );
    let linked_leaf = "define void @visible() {\nentry:\n ret void\n}";
    assert_eq!(
        modules("k", &[(Entry, entry), (Linked, linked_leaf)]),
        LoopClass::LoopFree
    );
    let linked_loop =
        "define void @visible() {\nentry:\n br label %spin\nspin:\n br label %spin\n}";
    assert_eq!(
        modules("k", &[(Entry, entry), (Linked, linked_loop)]),
        cfg_cycle("visible")
    );
}

#[test]
fn visible_function_reference_stubs_resolve_through_their_abi() {
    if !assembler("visible_function_reference_stubs_resolve_through_their_abi") {
        return;
    }
    let entry = "define void @k() {\nentry:\n %v = call i32 @linked.MTL_VISIBLE_FN_REF()\n ret void\n}\ndeclare i32 @linked.MTL_VISIBLE_FN_REF() section \"air.externally_defined\"\n!air.visible_function_references = !{!4}\n!4 = !{!\"air.visible_function_reference\", ptr @linked.MTL_VISIBLE_FN_REF, !\"linked\"}";
    assert_eq!(
        class(entry),
        LoopClass::Cyclic(CycleEvidence::UnresolvedCall {
            function: "k".into(),
            callee: "linked".into()
        })
    );
    let leaf = "define i32 @linked() {\nentry:\n ret i32 1\n}";
    assert_eq!(
        modules("k", &[(Entry, entry), (Linked, leaf)]),
        LoopClass::LoopFree
    );
    let recursive = "define i32 @linked() {\nentry:\n %r = call i32 @linked()\n ret i32 %r\n}";
    assert_eq!(
        modules("k", &[(Entry, entry), (Linked, recursive)]),
        call_cycle("linked")
    );
}

#[test]
fn indirect_calls_reach_every_linked_function_and_address_taken_entry_function() {
    if !assembler("indirect_calls_reach_every_linked_function_and_address_taken_entry_function") {
        return;
    }
    let entry = "define void @k(ptr %table) {\nentry:\n %fn = load ptr, ptr %table\n %r = call float %fn(float 1.0)\n ret void\n}";
    // No possible target: an indirect call cannot reach any code, so it adds no edge.
    assert_eq!(class(entry), LoopClass::LoopFree);
    let leaf = "define float @slot(float %x) {\nentry:\n ret float %x\n}";
    assert_eq!(
        modules("k", &[(Entry, entry), (Linked, leaf)]),
        LoopClass::LoopFree
    );
    // A table function that itself calls through a table reaches itself.
    let reentrant = "define float @slot(ptr %table) {\nentry:\n %fn = load ptr, ptr %table\n %r = call float %fn(float 1.0)\n ret float %r\n}";
    assert_eq!(
        modules("k", &[(Entry, entry), (Linked, reentrant)]),
        call_cycle("slot")
    );
    // An entry-module helper whose address escapes is a possible target too.
    let escaping = "define internal float @helper(ptr %table) {\nentry:\n %fn = load ptr, ptr %table\n %r = call float %fn(float 1.0)\n ret float %r\n}\ndefine void @k(ptr %table) {\nentry:\n store ptr @helper, ptr %table\n %r = call float @helper(ptr %table)\n ret void\n}";
    assert_eq!(class(escaping), call_cycle("helper"));
    // The same helper called only directly, and named only in metadata, is not a target.
    let direct_only = escaping.replace(" store ptr @helper, ptr %table\n", "")
        + "\n!air.kernel = !{!0}\n!0 = !{ptr @helper}\n";
    assert_eq!(class(&direct_only), LoopClass::LoopFree);
    // A global initializer takes an address too.
    let global = direct_only.replace(
        "define internal float @helper",
        "@slot = global ptr @helper\ndefine internal float @helper",
    );
    assert_eq!(class(&global), call_cycle("helper"));
}

#[test]
fn intersection_intrinsics_reach_linked_intersection_functions() {
    if !assembler("intersection_intrinsics_reach_linked_intersection_functions") {
        return;
    }
    let entry = "define void @k() {\nentry:\n call void @air.intersect.test()\n ret void\n}\ndeclare void @air.intersect.test()";
    let leaf = "define i1 @hit() {\nentry:\n ret i1 true\n}";
    assert_eq!(
        modules("k", &[(Entry, entry), (Intersection, leaf)]),
        LoopClass::LoopFree
    );
    let retracing = "define i1 @hit() {\nentry:\n call void @air.intersect.test()\n ret i1 true\n}\ndeclare void @air.intersect.test()";
    assert_eq!(
        modules("k", &[(Entry, entry), (Intersection, retracing)]),
        call_cycle("hit")
    );
    // The same module linked as a plain visible module is not reached from the intrinsic.
    assert_eq!(
        modules("k", &[(Entry, entry), (Linked, retracing)]),
        LoopClass::LoopFree
    );
}

#[test]
fn runtime_started_functions_are_roots() {
    if !assembler("runtime_started_functions_are_roots") {
        return;
    }
    let init = "define void @k() {\n ret void\n}\ndefine internal void @init(i1 %c) section \"air.static_init\" {\nentry:\n br label %spin\nspin:\n br i1 %c, label %spin, label %done\ndone:\n ret void\n}";
    assert_eq!(class(init), cfg_cycle("init"));
    let ctor = "@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @ctor, ptr null }]\ndefine void @k() {\n ret void\n}\ndefine internal void @ctor() {\n call void @ctor()\n ret void\n}";
    assert_eq!(class(ctor), call_cycle("ctor"));
}

#[test]
fn only_the_abi_named_control_point_fetch_is_a_runtime_leaf() {
    if !assembler("only_the_abi_named_control_point_fetch_is_a_runtime_leaf") {
        return;
    }
    let entry = "define void @k(ptr %patch) {\nentry:\n %p = tail call { <3 x float> } @fetch.MTL_CONTROL_POINT_FN(i32 0, ptr %patch)\n ret void\n}\ndeclare { <3 x float> } @fetch.MTL_CONTROL_POINT_FN(i32, ptr) section \"air.externally_defined\"\n";
    let named = format!(
        "{entry}!air.vertex = !{{!20}}\n!20 = !{{ptr @k, !{{}}, !{{!21}}, !23}}\n!21 = !{{i32 0, !\"air.patch_control_point_input\", !22, !24}}\n!22 = !{{!\"air.patch_control_point_function\", ptr @fetch.MTL_CONTROL_POINT_FN}}\n!24 = !{{!\"air.location_index\", i32 0, i32 1, !\"air.arg_type_name\", !\"float3\", !\"air.arg_name\", !\"p\"}}\n!23 = !{{!\"air.patch\", !\"triangle\", !\"air.patch_control_point\", i32 3}}\n"
    );
    assert_eq!(class(&named), LoopClass::LoopFree);
    assert_eq!(kind(entry), Some("unresolved_call"));
    // The fetch is taken from the decoded ABI: a module whose metadata ABI does not decode (here
    // a control-point input on a vertex function that is not post-tessellation) resolves
    // nothing, so the call stays unresolved.
    let malformed = named.replace(", !23}", "}");
    assert_eq!(kind(&malformed), Some("unresolved_call"));
}

#[test]
fn linked_modules_with_visible_declarations_resolve_stubs() {
    if !assembler("linked_modules_with_visible_declarations_resolve_stubs") {
        return;
    }
    let entry = "define void @k() {\nentry:\n %v = call i32 @linked.MTL_VISIBLE_FN_REF(i32 1)\n ret void\n}\ndeclare i32 @linked.MTL_VISIBLE_FN_REF(i32) section \"air.externally_defined\"\n!air.visible_function_references = !{!4}\n!4 = !{!\"air.visible_function_reference\", ptr @linked.MTL_VISIBLE_FN_REF, !\"linked\"}";
    // A linked `[[stitchable]]` module as the Metal compiler lists it: its `!air.visible`
    // declaration with stitching info decodes, and it names its own helpers.
    let stitchable = "define i32 @linked(i32 %x) {\nentry:\n ret i32 %x\n}\ndefine internal void @load(ptr %p) {\n ret void\n}\n!air.visible = !{!0}\n!0 = !{ptr @linked, !1, !3, !5}\n!1 = !{!2}\n!2 = !{!\"air.visible_output\", !\"air.arg_type_name\", !\"int\"}\n!3 = !{!4}\n!4 = !{i32 0, !\"air.visible_input\", !\"air.arg_type_name\", !\"int\", !\"air.arg_name\", !\"x\"}\n!5 = !{!\"air.stitching_info\", !6, !8}\n!6 = !{!\"air.stitching_type\", !7, ptr @load, ptr @load, ptr @load, null}\n!7 = !{!\"air.int_type\", i32 4, i32 4, null}\n!8 = !{!\"air.stitching_argument\", !6, !\"x\"}";
    assert_eq!(
        modules("k", &[(Entry, entry), (Linked, stitchable)]),
        LoopClass::LoopFree
    );
    let recursive = stitchable.replace(
        " ret i32 %x\n",
        " %r = call i32 @linked(i32 %x)\n ret i32 %r\n",
    );
    assert_eq!(
        modules("k", &[(Entry, entry), (Linked, &recursive)]),
        call_cycle("linked")
    );
    // An entry module whose metadata ABI does not decode resolves no stub.
    let malformed = format!("{entry}\n!air.kernel = !{{!9}}\n!9 = !{{ptr @k}}");
    assert_eq!(
        modules("k", &[(Entry, &malformed), (Linked, stitchable)]),
        LoopClass::Cyclic(CycleEvidence::UnresolvedCall {
            function: "k".into(),
            callee: "linked.MTL_VISIBLE_FN_REF".into()
        })
    );
}

fn fuel(entry: &str, sources: &[(ModuleRole, &str)]) -> Classification {
    classify_sources_with_fuel(entry, sources.iter().map(|&(role, ll)| (role, source(ll))))
}

fn uncovered_cycle(function: &str) -> FuelScope {
    FuelScope::Uncovered(CycleEvidence::CfgCycle {
        function: function.into(),
    })
}

const SPIN: &str = "br label %spin\nspin:\n %i = phi i32 [ 0, %entry ], [ %i1, %spin ]\n %i1 = add i32 %i, 1\n %c = icmp ult i32 %i1, %n\n br i1 %c, label %spin, label %done\ndone:\n";

#[test]
fn oracle_fuel_covers_cycles_only_direct_calls_from_the_entry_reach() {
    if !assembler("oracle_fuel_covers_cycles_only_direct_calls_from_the_entry_reach") {
        return;
    }
    // A loop in the entry, and one in a helper only the entry calls directly: covered.
    let entry_loop = format!("define void @k(i32 %n) {{\nentry:\n {SPIN} ret void\n}}");
    let own = fuel("k", &[(Entry, &entry_loop)]);
    assert_eq!(own.class, cfg_cycle("k"));
    assert_eq!(own.fuel, FuelScope::Covered);
    let helper = format!(
        "define internal void @h(i32 %n) {{\nentry:\n {SPIN} ret void\n}}\ndefine void @k(i32 %n) {{\n call void @h(i32 %n)\n call void @h(i32 %n)\n ret void\n}}"
    );
    assert_eq!(fuel("k", &[(Entry, &helper)]).fuel, FuelScope::Covered);
    // A loop-free module is covered trivially.
    let flat = "define void @k() {\n ret void\n}";
    assert_eq!(
        fuel("k", &[(Entry, flat)]),
        Classification {
            class: LoopClass::LoopFree,
            fuel: FuelScope::Covered
        }
    );
    // A loop in a linked module: the transform instruments the entry module only.
    let calls_linked =
        "define void @k(i32 %n) {\nentry:\n call void @visible(i32 %n)\n ret void\n}\ndeclare void @visible(i32)";
    let linked = format!("define void @visible(i32 %n) {{\nentry:\n {SPIN} ret void\n}}");
    assert_eq!(
        fuel("k", &[(Entry, calls_linked), (Linked, &linked)]).fuel,
        uncovered_cycle("visible")
    );
    // A loop in a helper whose address escapes: an indirect call cannot pass the state.
    let escaping = format!(
        "define internal void @h(i32 %n) {{\nentry:\n {SPIN} ret void\n}}\ndefine void @k(ptr %table, i32 %n) {{\n store ptr @h, ptr %table\n call void @h(i32 %n)\n ret void\n}}"
    );
    let escaped = fuel("k", &[(Entry, &escaping)]);
    assert_eq!(escaped.class, cfg_cycle("h"));
    assert_eq!(escaped.fuel, uncovered_cycle("h"));
    // A loop in an initializer the runtime runs itself.
    let init = format!(
        "define void @k() {{\n ret void\n}}\ndefine internal void @init(i32 %n) section \"air.static_init\" {{\nentry:\n {SPIN} ret void\n}}"
    );
    assert_eq!(fuel("k", &[(Entry, &init)]).fuel, uncovered_cycle("init"));
    // A call cycle is never covered, even behind a CFG cycle the class names first.
    let recursive = format!(
        "define internal void @r(i32 %n) {{\n call void @r(i32 %n)\n ret void\n}}\ndefine void @k(i32 %n) {{\nentry:\n {SPIN} call void @r(i32 %n)\n ret void\n}}"
    );
    let both = fuel("k", &[(Entry, &recursive)]);
    assert_eq!(both.class, cfg_cycle("k"));
    assert_eq!(
        both.fuel,
        FuelScope::Uncovered(CycleEvidence::CallCycle {
            function: "r".into()
        })
    );
}

/// An entry choosing `@spin` (a loop), `@flat` or nothing by `%p`/`%q` with a `select` tree,
/// testing the choice against null, and calling it; `extra` is spliced in before the test.
fn chooser(extra: &str) -> String {
    format!(
        "define internal void @spin(i32 %n) {{\nentry:\n {SPIN} ret void\n}}\n\
define internal void @flat(i32 %n) {{\n ret void\n}}\n\
define void @k(i1 %p, i1 %q, i32 %n, ptr %out) {{\nentry:\n\
 %s0 = select i1 %p, ptr @spin, ptr null\n %s1 = select i1 %q, ptr @flat, ptr %s0\n{extra}\
 %none = icmp eq ptr %s1, null\n br i1 %none, label %done, label %call\n\
call:\n call void %s1(i32 %n)\n br label %done\ndone:\n ret void\n}}"
    )
}

#[test]
fn a_closed_indirect_call_threads_fuel_to_each_of_its_targets() {
    if !assembler("a_closed_indirect_call_threads_fuel_to_each_of_its_targets") {
        return;
    }
    // Its targets are reached by a local call only: the loop is covered.
    let closed = fuel("k", &[(Entry, &chooser(""))]);
    assert_eq!(closed.class, cfg_cycle("spin"));
    assert_eq!(closed.fuel, FuelScope::Covered);
    // A target whose address escapes would take the state too (the call's targets keep one
    // prototype), so the loop of another target is not covered.
    let stored = chooser(" store ptr @flat, ptr %out\n");
    assert_eq!(fuel("k", &[(Entry, &stored)]).fuel, uncovered_cycle("spin"));
    // The choice itself escaping opens the call: every leaf may be reached from elsewhere.
    let open = chooser(" store ptr %s1, ptr %out\n");
    let open = fuel("k", &[(Entry, &open)]);
    assert_eq!(open.class, cfg_cycle("spin"));
    assert_eq!(open.fuel, uncovered_cycle("spin"));
    // A closed call reaches its targets only: a function whose address escapes elsewhere and
    // that loops back into the entry's callees is no target of it.
    let other = chooser("").replace(
        "define internal void @flat",
        "@slot = global ptr @again\ndefine internal void @again(i32 %n) {\n call void @again(i32 %n)\n ret void\n}\ndefine internal void @flat",
    );
    let other = fuel("k", &[(Entry, &other)]);
    assert_eq!(other.class, cfg_cycle("spin"));
    assert_eq!(other.fuel, FuelScope::Covered);
}

/// A kernel calling MetalPerformancePrimitives' `matmul2d` ABI: the cooperative-tensor helpers with
/// descriptor fields `{fields}` by value, and `run` with the descriptor `@desc` (whose initializer
/// is `{desc}`) copied into its stack object right before the call. `extra` is spliced into the
/// entry block before the copy.
fn matmul2d_kernel(fields: &str, desc: &str, extra: &str) -> String {
    format!(
        r#"%struct.desc = type {{ i32, i32, i32, i8, i8, i8, i32 }}
@desc = internal constant %struct.desc {{ {desc} }}
define void @k(ptr addrspace(1) %a) {{
entry:
  %d = alloca %struct.desc, align 4
  %ta = alloca [16 x i8], align 8
  %tb = alloca [16 x i8], align 8
  %size = call i64 @__tensorops_impl_matmul2d_op_cooperative_tensor_data_size(i32 2, {fields}, i32 268435488, i32 268435488, i32 268435488, i32 32)
  %ct = alloca i8, i64 %size, align 4
  call void @__tensorops_impl_matmul2d_op_cooperative_tensor_init(i32 2, {fields}, ptr %ct, i32 268435488, i32 268435488, i32 268435488, i32 32)
  %n = call i16 @__tensorops_impl_matmul2d_op_cooperative_tensor_num_elements(i32 2, {fields}, ptr %ct, i32 268435488, i32 268435488, i32 32)
  %e = call ptr @__tensorops_impl_matmul2d_op_cooperative_tensor_get_element_pointer(i32 2, {fields}, ptr %ct, i16 -1, i32 268435488, i32 268435488, i32 268435488)
{extra}  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr @desc, i64 20, i1 false)
  call void @__tensorops_impl_matmul2d_op_run_cooperative_dv_f32_dv_f32_f32(ptr %d, ptr %ta, i32 2, ptr %tb, i32 2, ptr %e, i32 32)
  ret void
}}
declare i64 @__tensorops_impl_matmul2d_op_cooperative_tensor_data_size(i32, i32, i32, i32, i8, i8, i8, i32, i32, i32, i32, i32) section "air.externally_defined"
declare void @__tensorops_impl_matmul2d_op_cooperative_tensor_init(i32, i32, i32, i32, i8, i8, i8, i32, ptr, i32, i32, i32, i32) section "air.externally_defined"
declare i16 @__tensorops_impl_matmul2d_op_cooperative_tensor_num_elements(i32, i32, i32, i32, i8, i8, i8, i32, ptr, i32, i32, i32) section "air.externally_defined"
declare ptr @__tensorops_impl_matmul2d_op_cooperative_tensor_get_element_pointer(i32, i32, i32, i32, i8, i8, i8, i32, ptr, i16, i32, i32, i32) section "air.externally_defined"
declare void @__tensorops_impl_matmul2d_op_run_cooperative_dv_f32_dv_f32_f32(ptr, ptr, i32, ptr, i32, ptr, i32) section "air.externally_defined"
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
"#
    )
}

const STATIC_FIELDS: &str = "i32 16, i32 16, i32 32, i8 0, i8 1, i8 0, i32 1";
const STATIC_DESC: &str = "i32 16, i32 16, i32 32, i8 0, i8 1, i8 0, i32 1";

fn unresolved(callee: &str) -> LoopClass {
    LoopClass::Cyclic(CycleEvidence::UnresolvedCall {
        function: "k".into(),
        callee: format!("__tensorops_impl_matmul2d_op_{callee}"),
    })
}

#[test]
fn matmul2d_calls_with_a_constant_descriptor_are_leaves() {
    if !assembler("matmul2d_calls_with_a_constant_descriptor_are_leaves") {
        return;
    }
    let ll = matmul2d_kernel(STATIC_FIELDS, STATIC_DESC, "");
    assert_eq!(class(&ll), LoopClass::LoopFree);
    let classification = classify_sources_with_fuel("k", [(Entry, source(&ll))]);
    assert!(classification.fuel.is_covered());
    // A loop around them is the entry's own CFG cycle, which fuel covers.
    let looped = ll.replace(
        "  ret void\n}",
        "  br label %entry2\nentry2:\n  br i1 true, label %entry2, label %done\ndone:\n  ret void\n}",
    );
    assert_eq!(class(&looped), cfg_cycle("k"));
    assert!(classify_sources_with_fuel("k", [(Entry, source(&looped))])
        .fuel
        .is_covered());
}

#[test]
fn matmul2d_calls_whose_work_no_constant_fixes_stay_unresolved() {
    if !assembler("matmul2d_calls_whose_work_no_constant_fixes_stay_unresolved") {
        return;
    }
    // A dynamic inner dimension (`dynamic_extent`) in the helpers' fields.
    let dynamic = "i32 16, i32 16, i32 -1, i8 0, i8 1, i8 0, i32 1";
    assert_eq!(
        class(&matmul2d_kernel(dynamic, STATIC_DESC, "")),
        unresolved("cooperative_tensor_data_size")
    );
    // ... or in the descriptor `run` reads.
    assert_eq!(
        class(&matmul2d_kernel(STATIC_FIELDS, dynamic, "")),
        unresolved("run_cooperative_dv_f32_dv_f32_f32")
    );
    // A field that is not a constant.
    let runtime = matmul2d_kernel(STATIC_FIELDS, STATIC_DESC, "")
        .replace(
            "(i32 2, i32 16, i32 16, i32 32",
            "(i32 2, i32 %m, i32 16, i32 32",
        )
        .replace(
            "entry:\n",
            "entry:\n  %m = ptrtoint ptr addrspace(1) %a to i32\n",
        );
    assert_eq!(class(&runtime), unresolved("cooperative_tensor_data_size"));
    // A descriptor that does not decode (a zero tile).
    let zero = "i32 0, i32 16, i32 32, i8 0, i8 1, i8 0, i32 1";
    assert_eq!(
        class(&matmul2d_kernel(zero, STATIC_DESC, "")),
        unresolved("cooperative_tensor_data_size")
    );
    // A store into `run`'s descriptor object.
    let stored = matmul2d_kernel(
        STATIC_FIELDS,
        STATIC_DESC,
        "  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr @desc, i64 20, i1 false)\n  %kf = getelementptr i8, ptr %d, i64 8\n  store i32 -1, ptr %kf\n",
    );
    assert_eq!(
        class(&stored),
        unresolved("run_cooperative_dv_f32_dv_f32_f32")
    );
    // A copy that is not a whole descriptor.
    let partial = matmul2d_kernel(STATIC_FIELDS, STATIC_DESC, "")
        .replace("ptr @desc, i64 20", "ptr @desc, i64 12");
    assert_eq!(
        class(&partial),
        unresolved("run_cooperative_dv_f32_dv_f32_f32")
    );
    // A copy from a global that is not constant.
    let mutable = matmul2d_kernel(STATIC_FIELDS, STATIC_DESC, "")
        .replace("@desc = internal constant", "@desc = internal global");
    assert_eq!(
        class(&mutable),
        unresolved("run_cooperative_dv_f32_dv_f32_f32")
    );
    // No copy before the call in its block.
    let late = matmul2d_kernel(STATIC_FIELDS, STATIC_DESC, "").replace(
        "  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr @desc, i64 20, i1 false)\n  call void @__tensorops_impl_matmul2d_op_run",
        "  br label %run\nrun:\n  call void @__tensorops_impl_matmul2d_op_run",
    ).replace(
        "entry:\n  %d = alloca %struct.desc, align 4\n",
        "entry:\n  %d = alloca %struct.desc, align 4\n  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr @desc, i64 20, i1 false)\n",
    );
    assert_eq!(
        class(&late),
        unresolved("run_cooperative_dv_f32_dv_f32_f32")
    );
}

#[test]
fn matmul2d_names_are_the_abi_only_as_declared_and_modeled() {
    if !assembler("matmul2d_names_are_the_abi_only_as_declared_and_modeled") {
        return;
    }
    let base = matmul2d_kernel(STATIC_FIELDS, STATIC_DESC, "");
    // Outside `air.externally_defined`.
    let unsectioned = base.replace(
        "(i32, i32, i32, i32, i8, i8, i8, i32, i32, i32, i32, i32) section \"air.externally_defined\"",
        "(i32, i32, i32, i32, i8, i8, i8, i32, i32, i32, i32, i32)",
    );
    assert_eq!(
        class(&unsectioned),
        unresolved("cooperative_tensor_data_size")
    );
    // A family name the contract does not model.
    let unmodeled = base.replace(
        "cooperative_tensor_num_elements",
        "cooperative_reduction_destination_tensor_num_elements",
    );
    assert_eq!(
        class(&unmodeled),
        unresolved("cooperative_reduction_destination_tensor_num_elements")
    );
    // A call whose argument count is not its form's.
    let arity = base
        .replace("ptr %e, i32 32)\n  ret void", "ptr %e)\n  ret void")
        .replace(
            "(ptr, ptr, i32, ptr, i32, ptr, i32) section",
            "(ptr, ptr, i32, ptr, i32, ptr) section",
        );
    assert_eq!(
        class(&arity),
        unresolved("run_cooperative_dv_f32_dv_f32_f32")
    );
}
