//! Where coordinate-mapping calls read their map, over synthetic module text and the public
//! fixtures.

use super::*;
use crate::air::reflect::EmbeddedArgBuffer;
use std::cell::Cell;

thread_local! {
    /// Root pairs `normalized` compared, for the boundedness tests.
    static COMPARED: Cell<usize> = const { Cell::new(0) };
}

/// Count `pairs` more root pairs `normalized` compared.
pub(super) fn compared(pairs: usize) {
    COMPARED.with(|compared| compared.set(compared.get() + pairs));
}

const PLACEMENT: Option<ToolingRequirement> =
    Some(ToolingRequirement::RasterizationRateMapPlacement);

/// A kernel reflection of entry `main` whose argument-buffer buffer members are `members`
/// (`(parameter, byte offset)`).
fn reflection(members: &[(u32, u32)]) -> ShaderInterface {
    let mut reflection = crate::air::reflect::interface_from_kernel(
        &crate::air::meta::KernMeta::default(),
        Some("main"),
    );
    reflection.argument_buffer_fields = members
        .iter()
        .enumerate()
        .map(|(ordinal, &(param, offset))| EmbeddedArgBuffer {
            buffer_param_index: param,
            buffer_index: param,
            field_offset: offset,
            field_ordinal: ordinal as u32,
            argument_index: ordinal as u32,
            resource_kind: Some(ArgumentBufferResourceKind::Buffer),
        })
        .collect();
    reflection
}

/// The declared reflection of kernel `entry` of `ll`.
fn inspected(ll: &str, entry: &str) -> ShaderInterface {
    crate::air::reflect::inspect(ll, crate::case::Stage::Kernel, entry, &[]).unwrap()
}

fn placement(places: &[MapPlace], unplaced: bool) -> RateMapPlacement {
    RateMapPlacement {
        places: places.iter().copied().collect(),
        unplaced,
    }
}

const BUFFER_0: MapPlace = MapPlace::Buffer { param: 0 };

/// The public fixture's map operand is its first parameter, through a no-op `bitcast`.
#[test]
fn the_public_fixture_reads_its_map_at_offset_0_of_a_buffer() {
    let ll = include_str!("../../../fixtures/public/kernel_rate_map_physical_to_screen.ll");
    let reflection = inspected(ll, "kernel_rate_map_physical_to_screen");
    assert_eq!(
        rate_map_placement(ll, &reflection),
        placement(&[BUFFER_0], false)
    );

    let offset = ll.replace(
        "%7 = bitcast ptr addrspace(2) %0 to ptr addrspace(2)",
        "%7 = getelementptr inbounds i8, ptr addrspace(2) %0, i64 16",
    );
    assert_ne!(offset, ll);
    assert_eq!(
        rate_map_placement_requirement(&offset, &reflection),
        PLACEMENT
    );
}

/// The public fixture whose maps are argument-buffer members reads them at bytes 8 and 16 of
/// its argument buffer, through a `select` of the members' addresses and a load; its plain-data
/// member at byte 0 is no map.
#[test]
fn the_public_fixture_reads_its_maps_from_argument_buffer_members() {
    let ll = include_str!("../../../fixtures/public/kernel_rate_map_argument_buffer.ll");
    let reflection = inspected(ll, "kernel_rate_map_argument_buffer");
    assert_eq!(
        rate_map_placement(ll, &reflection),
        placement(
            &[
                MapPlace::ArgumentBufferMember {
                    param: 0,
                    offset: 8
                },
                MapPlace::ArgumentBufferMember {
                    param: 0,
                    offset: 16
                },
            ],
            false
        )
    );
}

/// The public fixture that picks its map from a thread-memory array reads both buffers whose
/// pointers the array holds.
#[test]
fn the_public_fixture_reads_its_maps_through_a_pointer_array() {
    let ll = include_str!("../../../fixtures/public/kernel_rate_map_pointer_array.ll");
    let reflection = inspected(ll, "kernel_rate_map_pointer_array");
    assert_eq!(
        rate_map_placement(ll, &reflection),
        placement(&[BUFFER_0, MapPlace::Buffer { param: 1 }], false)
    );
}

/// A kernel whose map operand is `map`, defined by `prelude`, with `params` after the map's
/// buffer.
fn kernel(params: &str, prelude: &str, map: &str) -> String {
    format!(
        "%struct.Args = type {{ ptr addrspace(2), <3 x float>, ptr addrspace(2), [2 x ptr \
         addrspace(2)] }}\ndefine void @main(ptr addrspace(2) noundef readonly %0{params}) \
         {{\n{prelude}  %r = tail call fast <2 x float> \
         @air.map_screen_to_physical_coordinates.v2f32.p2i8.i32(<2 x float> zeroinitializer, ptr \
         addrspace(2) readonly captures(none) {map}, i32 0)\n  ret void\n}}\ndeclare <2 x float> \
         @air.map_screen_to_physical_coordinates.v2f32.p2i8.i32(<2 x float>, ptr addrspace(2), \
         i32)\n"
    )
}

#[test]
fn offset_0_of_an_entry_buffer_is_placed() {
    for (prelude, map) in [
        ("", "%0"),
        (
            "  %m = bitcast ptr addrspace(2) %0 to ptr addrspace(2)\n",
            "%m",
        ),
        (
            "  %m = getelementptr inbounds [4 x i32], ptr addrspace(2) %0, i64 0, i64 0\n",
            "%m",
        ),
        ("  %m = freeze ptr addrspace(2) %0\n", "%m"),
        // Out by 16 bytes and back.
        (
            "  %a = getelementptr inbounds i8, ptr addrspace(2) %0, i64 16\n  %m = getelementptr \
             inbounds <4 x float>, ptr addrspace(2) %a, i64 -1\n",
            "%m",
        ),
    ] {
        let ll = kernel("", prelude, map);
        assert_eq!(
            rate_map_placement(&ll, &reflection(&[])),
            placement(&[BUFFER_0], false),
            "{ll}"
        );
    }
    // Either of two buffers, each at offset 0.
    let ll = kernel(
        ", ptr addrspace(2) %1, i1 %c",
        "  %m = select i1 %c, ptr addrspace(2) %0, ptr addrspace(2) %1\n",
        "%m",
    );
    assert_eq!(
        rate_map_placement(&ll, &reflection(&[])),
        placement(&[BUFFER_0, MapPlace::Buffer { param: 1 }], false)
    );
}

#[test]
fn map_data_elsewhere_is_not_placed() {
    for (params, prelude, map) in [
        // Another offset of the buffer.
        (
            "",
            "  %m = getelementptr inbounds i8, ptr addrspace(2) %0, i64 16\n",
            "%m",
        ),
        // A dynamic offset.
        (
            ", i64 %i",
            "  %m = getelementptr inbounds i8, ptr addrspace(2) %0, i64 %i\n",
            "%m",
        ),
        // A pointer an argument buffer holds where the reflection declares no buffer member.
        (
            "",
            "  %m = load ptr addrspace(2), ptr addrspace(2) %0, align 8\n",
            "%m",
        ),
        // A buffer, or a pointer loaded from it.
        (
            ", i1 %c",
            "  %l = load ptr addrspace(2), ptr addrspace(2) %0, align 8\n  %m = select i1 %c, ptr \
             addrspace(2) %0, ptr addrspace(2) %l\n",
            "%m",
        ),
        // A table (`array_ref`) parameter is no buffer.
        (", ptr %t", "", "%t"),
        // A constant expression.
        ("", "", "getelementptr (i8, ptr addrspace(2) @g, i64 0)"),
        // An integer cast to a pointer.
        (
            ", i64 %a",
            "  %m = inttoptr i64 %a to ptr addrspace(2)\n",
            "%m",
        ),
        // A device buffer, which the shader may write.
        (
            ", ptr addrspace(1) %d",
            "  %m = addrspacecast ptr addrspace(1) %d to ptr addrspace(2)\n",
            "%m",
        ),
    ] {
        let ll = kernel(params, prelude, map);
        assert_eq!(
            rate_map_placement_requirement(&ll, &reflection(&[])),
            PLACEMENT,
            "{ll}"
        );
    }
    // A buffer member of a device argument buffer, into which the shader may store another
    // pointer; the same member of a constant argument buffer is placed.
    let member = |owner: &str| {
        kernel(
            &format!(", {owner} %d"),
            &format!("  %m = load ptr addrspace(2), {owner} %d, align 8\n"),
            "%m",
        )
    };
    for (owner, expected) in [
        ("ptr addrspace(1)", PLACEMENT),
        ("i8 addrspace(2)* addrspace(1)*", PLACEMENT),
        ("ptr addrspace(2)", None),
        ("i8 addrspace(2)* addrspace(2)*", None),
    ] {
        assert_eq!(
            rate_map_placement_requirement(&member(owner), &reflection(&[(1, 0)])),
            expected,
            "{owner}"
        );
    }
}

/// A map an argument-buffer buffer member holds is placed at the member's byte offset, which a
/// struct `getelementptr` reaches through the module's type layout.
#[test]
fn a_map_an_argument_buffer_member_holds_is_placed() {
    let member = |offset| MapPlace::ArgumentBufferMember { param: 0, offset };
    // `%struct.Args` is a pointer at 0, a `<3 x float>` at 16, a pointer at 32, and two pointers
    // at 40 and 48.
    for (prelude, places) in [
        (
            "  %a = getelementptr inbounds %struct.Args, ptr addrspace(2) %0, i64 0, i32 2\n  %m = \
             load ptr addrspace(2), ptr addrspace(2) %a, align 8\n",
            vec![member(32)],
        ),
        // The members' addresses selected, then loaded (one row's shape).
        (
            "  %a = getelementptr inbounds %struct.Args, ptr addrspace(2) %0, i64 0, i32 3, i64 0\n  \
             %b = getelementptr inbounds %struct.Args, ptr addrspace(2) %0, i64 0, i32 3, i64 1\n  \
             %s = select i1 %c, ptr addrspace(2) %a, ptr addrspace(2) %b\n  %m = load ptr \
             addrspace(2), ptr addrspace(2) %s, align 8\n",
            vec![member(40), member(48)],
        ),
        // Both members loaded, then selected (another row's shape).
        (
            "  %x = load ptr addrspace(2), ptr addrspace(2) %0, align 8\n  %b = getelementptr \
             inbounds i8, ptr addrspace(2) %0, i64 48\n  %y = load ptr addrspace(2), ptr \
             addrspace(2) %b, align 8\n  %m = select i1 %c, ptr addrspace(2) %x, ptr addrspace(2) \
             %y\n",
            vec![member(0), member(48)],
        ),
    ] {
        let ll = kernel(", i1 %c", prelude, "%m");
        assert_eq!(
            rate_map_placement(&ll, &reflection(&[(0, 0), (0, 32), (0, 40), (0, 48)])),
            placement(&places, false),
            "{ll}"
        );
        // A member the reflection does not declare a buffer is not placed.
        assert_eq!(
            rate_map_placement_requirement(&ll, &reflection(&[(0, 16)])),
            PLACEMENT,
            "{ll}"
        );
    }
    let ll = kernel(
        ", i64 %i",
        "  %a = getelementptr inbounds %struct.Args, ptr addrspace(2) %0, i64 0, i32 3, i64 %i\n  \
         %m = load ptr addrspace(2), ptr addrspace(2) %a, align 8\n",
        "%m",
    );
    // A member array's dynamic element.
    assert_eq!(
        rate_map_placement_requirement(&ll, &reflection(&[(0, 40), (0, 48)])),
        PLACEMENT
    );
    // A pointer loaded through a loaded pointer.
    let ll = kernel(
        "",
        "  %a = load ptr addrspace(2), ptr addrspace(2) %0, align 8\n  %m = load ptr addrspace(2), \
         ptr addrspace(2) %a, align 8\n",
        "%m",
    );
    assert_eq!(
        rate_map_placement_requirement(&ll, &reflection(&[(0, 0)])),
        PLACEMENT
    );
    // An argument buffer's own memory is no map.
    let ll = kernel("", "", "%0");
    assert_eq!(
        rate_map_placement_requirement(&ll, &reflection(&[(0, 0)])),
        PLACEMENT
    );
}

/// A map picked from a thread-memory array of pointers is any pointer stored into the array,
/// while the array's address goes nowhere else.
#[test]
fn a_map_a_thread_memory_array_holds_is_placed_by_its_stores() {
    let array = |extra: &str| {
        format!(
            "  %a = alloca [2 x ptr addrspace(2)], align 8\n  call void \
             @llvm.lifetime.start.p0(i64 16, ptr nonnull %a)\n  %e0 = getelementptr inbounds [2 x \
             ptr addrspace(2)], ptr %a, i64 0, i64 0\n  store ptr addrspace(2) %0, ptr %e0, align \
             8\n  %e1 = getelementptr inbounds [2 x ptr addrspace(2)], ptr %a, i64 0, i64 1\n  \
             store ptr addrspace(2) %1, ptr %e1, align 8\n{extra}  %e = getelementptr inbounds [2 \
             x ptr addrspace(2)], ptr %a, i64 0, i64 %i\n  %c = bitcast ptr %e to ptr\n  %m = load \
             ptr addrspace(2), ptr %c, align 8\n"
        )
    };
    let params = ", ptr addrspace(2) %1, i64 %i, ptr %t";
    let ll = kernel(params, &array(""), "%m");
    assert_eq!(
        rate_map_placement(&ll, &reflection(&[])),
        placement(&[BUFFER_0, MapPlace::Buffer { param: 1 }], false)
    );
    // A pointer loaded from an argument-buffer member, stored into the array.
    let ll = kernel(
        params,
        &array("").replace(
            "  store ptr addrspace(2) %1, ptr %e1, align 8\n",
            "  %l = load ptr addrspace(2), ptr addrspace(2) %1, align 8\n  store ptr addrspace(2) \
             %l, ptr %e1, align 8\n",
        ),
        "%m",
    );
    assert_eq!(
        rate_map_placement(&ll, &reflection(&[(1, 0)])),
        placement(
            &[
                BUFFER_0,
                MapPlace::ArgumentBufferMember {
                    param: 1,
                    offset: 0
                }
            ],
            false
        )
    );
    for (why, extra) in [
        (
            "the array's address is stored",
            "  store ptr %a, ptr %t, align 8\n",
        ),
        (
            "the array's address is passed to a call",
            "  call void @use(ptr %e1)\n",
        ),
        (
            "an integer is stored into it",
            "  store i64 7, ptr %e1, align 8\n",
        ),
        (
            "bytes are copied into it",
            "  call void @llvm.memcpy.p0.p2.i64(ptr %a, ptr addrspace(2) %1, i64 16, i1 false)\n",
        ),
        (
            "a pointer at another offset is stored",
            "  %o = getelementptr inbounds i8, ptr addrspace(2) %1, i64 8\n  store ptr \
             addrspace(2) %o, ptr %e1, align 8\n",
        ),
        (
            "its address is selected",
            "  %s = select i1 true, ptr %e0, ptr %e1\n",
        ),
    ] {
        let ll = kernel(params, &array(extra), "%m");
        assert_eq!(
            rate_map_placement_requirement(&ll, &reflection(&[])),
            PLACEMENT,
            "{why}"
        );
    }
    // An array nothing is stored into.
    let ll = kernel(
        ", i64 %i",
        "  %a = alloca [2 x ptr addrspace(2)], align 8\n  %e = getelementptr inbounds [2 x ptr \
         addrspace(2)], ptr %a, i64 0, i64 %i\n  %m = load ptr addrspace(2), ptr %e, align 8\n",
        "%m",
    );
    assert_eq!(
        rate_map_placement_requirement(&ll, &reflection(&[])),
        PLACEMENT
    );
}

/// The map reaches the call through a defined function's parameter, as two corpus rows'
/// non-inlined callees take it: placed when every direct call, from any defined function, passes
/// a placed value.
#[test]
fn a_callee_parameter_is_placed_by_every_call() {
    let callee = "define internal <2 x float> @decode(ptr addrspace(2) %map, <2 x float> %c) {\n  \
                  %r = tail call fast <2 x float> \
                  @air.map_physical_to_screen_coordinates.v2f32.p2i8.i32(<2 x float> %c, ptr \
                  addrspace(2) %map, i32 0)\n  ret <2 x float> %r\n}\n";
    let entry = |calls: &str| {
        format!(
            "define void @main(ptr addrspace(2) %0, ptr addrspace(2) %1) {{\n  %m = bitcast ptr \
             addrspace(2) %0 to ptr addrspace(2)\n{calls}  ret void\n}}\n{callee}"
        )
    };
    let placed = entry(
        "  %a = call <2 x float> @decode(ptr addrspace(2) %m, <2 x float> zeroinitializer)\n  %b \
         = call <2 x float> @decode(ptr addrspace(2) %1, <2 x float> zeroinitializer)\n",
    );
    assert_eq!(
        rate_map_placement(&placed, &reflection(&[])),
        placement(&[BUFFER_0, MapPlace::Buffer { param: 1 }], false)
    );

    let one_offset = entry(
        "  %a = call <2 x float> @decode(ptr addrspace(2) %m, <2 x float> zeroinitializer)\n  %o \
         = getelementptr inbounds i8, ptr addrspace(2) %1, i64 64\n  %b = call <2 x float> \
         @decode(ptr addrspace(2) %o, <2 x float> zeroinitializer)\n",
    );
    assert_eq!(
        rate_map_placement_requirement(&one_offset, &reflection(&[])),
        PLACEMENT
    );

    // A function no direct call reaches (a visible function) may be passed anything.
    let indirect = entry("");
    assert_eq!(
        rate_map_placement_requirement(&indirect, &reflection(&[])),
        PLACEMENT
    );
    // So may one a table names, whatever its direct calls pass.
    let named = format!(
        "{placed}!air.visible_function = !{{!9}}\n!9 = !{{ptr @decode}}\n",
        placed = placed
    );
    assert_eq!(
        rate_map_placement_requirement(&named, &reflection(&[])),
        PLACEMENT
    );
    // Without the entry the reflection names, nothing is placed.
    let mut other = reflection(&[]);
    other.entry_point = Some("other".into());
    assert_eq!(rate_map_placement_requirement(&placed, &other), PLACEMENT);
}

/// A call site in a function the entry does not call still passes its value: `@vf`, which `@main`
/// never calls, passes `@decode` a pointer it loaded (the RRM verifier's probe).
#[test]
fn every_defined_function_s_call_sites_count() {
    let ll = "define void @main(ptr addrspace(2) %0) {\n  %a = call <2 x float> @decode(ptr \
              addrspace(2) %0)\n  ret void\n}\ndefine void @vf(ptr addrspace(2) %p) {\n  %l = \
              load ptr addrspace(2), ptr addrspace(2) %p, align 8\n  %a = call <2 x float> \
              @decode(ptr addrspace(2) %l)\n  ret void\n}\ndefine internal <2 x float> \
              @decode(ptr addrspace(2) %map) {\n  %r = tail call fast <2 x float> \
              @air.map_physical_to_screen_coordinates.v2f32.p2i8.i32(<2 x float> \
              zeroinitializer, ptr addrspace(2) %map, i32 0)\n  ret <2 x float> %r\n}\n";
    assert_eq!(
        rate_map_placement_requirement(ll, &reflection(&[])),
        PLACEMENT
    );
    // Without `@vf`'s call, the map is the entry's buffer.
    let ll = ll.replace("  %a = call <2 x float> @decode(ptr addrspace(2) %l)\n", "");
    assert_eq!(
        rate_map_placement(&ll, &reflection(&[])),
        placement(&[BUFFER_0], false)
    );
}

/// The entry passes a pointer it loads from an argument-buffer member down a chain of callees
/// (one row's shape), and a callee loads the member itself.
#[test]
fn argument_buffer_members_reach_the_call_through_callees() {
    let ll = "%struct.Args = type { ptr addrspace(2), ptr addrspace(2) }\ndefine void @main(ptr \
              addrspace(2) %0) {\n  %a = getelementptr inbounds %struct.Args, ptr addrspace(2) %0, \
              i64 0, i32 1\n  %l = load ptr addrspace(2), ptr addrspace(2) %a, align 8\n  call \
              fastcc void @outer(ptr addrspace(2) %l, ptr addrspace(2) %0)\n  ret void\n}\ndefine \
              internal fastcc void @outer(ptr addrspace(2) %map, ptr addrspace(2) %args) {\n  %r = \
              call fastcc <2 x float> @inner(ptr addrspace(2) %map)\n  %s = load ptr \
              addrspace(2), ptr addrspace(2) %args, align 8\n  %t = call fastcc <2 x float> \
              @inner(ptr addrspace(2) %s)\n  ret void\n}\ndefine internal fastcc <2 x float> \
              @inner(ptr addrspace(2) %map) {\n  %r = tail call fast <2 x float> \
              @air.map_screen_to_physical_coordinates.v2f32.p2i8.i32(<2 x float> \
              zeroinitializer, ptr addrspace(2) %map, i32 0)\n  ret <2 x float> %r\n}\n";
    assert_eq!(
        rate_map_placement(ll, &reflection(&[(0, 0), (0, 8)])),
        placement(
            &[
                MapPlace::ArgumentBufferMember {
                    param: 0,
                    offset: 0
                },
                MapPlace::ArgumentBufferMember {
                    param: 0,
                    offset: 8
                },
            ],
            false
        )
    );
}

#[test]
fn a_module_without_coordinate_mapping_needs_nothing() {
    let ll = "define void @main(ptr addrspace(2) %0) {\n  ret void\n}\n";
    assert_eq!(
        rate_map_placement(ll, &reflection(&[])),
        RateMapPlacement::default()
    );
}

/// A chain of `functions` defined functions, each passing its map parameter to the next and the
/// last mapping through it, entered from `@main`.
fn chain(functions: usize) -> String {
    let mut ll =
        "define void @main(ptr addrspace(2) %0) {\n  call void @f0(ptr addrspace(2) %0)\n  \
                  ret void\n}\n"
            .to_string();
    for index in 0..functions {
        let next = if index + 1 == functions {
            "  %r = tail call fast <2 x float> \
             @air.map_screen_to_physical_coordinates.v2f32.p2i8.i32(<2 x float> zeroinitializer, \
             ptr addrspace(2) %m, i32 0)\n"
                .to_string()
        } else {
            format!("  call void @f{}(ptr addrspace(2) %m)\n", index + 1)
        };
        ll.push_str(&format!(
            "define internal void @f{index}(ptr addrspace(2) %p) {{\n  %m = bitcast ptr \
             addrspace(2) %p to ptr addrspace(2)\n{next}  ret void\n}}\n"
        ));
    }
    ll
}

/// The fixed point resolves each call site again only when its caller's parameters grow, so the
/// work over a chain of callees grows linearly with its length (the RRM verifier found the
/// earlier fixpoint about cubic: 12, 37, and 166 ms at 50, 100, and 200 functions).
#[test]
fn the_placement_is_linear_in_a_chain_of_callees() {
    let work = |functions: usize| {
        let mut work = Work::default();
        let placed = analyze(&chain(functions), &reflection(&[]), &mut work);
        assert_eq!(placed, placement(&[BUFFER_0], false), "{functions}");
        work.traced + work.visited + work.resolved
    };
    let counts = [50, 100, 200, 400].map(|functions| (functions, work(functions)));
    for (functions, count) in counts {
        // Four units of work per function, one of them the forged rule's resolution of each
        // reached function (203 at 50 functions when written), with slack for a second
        // resolution of each call site.
        assert!(count <= 6 * functions + 8, "{functions} functions: {count}");
    }
    assert!(counts[3].1 <= 2 * counts[2].1 + 8, "{counts:?}");
}

/// What a constant `getelementptr` adds, under AIR's layout and another one.
#[test]
fn constant_offsets_follow_the_type_layout() {
    let types = "%s = type { i8, <3 x float>, [2 x ptr addrspace(2)] }\n%\"p q\" = type <{ i8, \
                 i32, %s }>\n%r = type { double, i16, half }\n";
    let layouts = layout::Layouts::new(types);
    let offset = |source: &str, indices: &[&str]| layouts.gep_offset(source, indices);
    assert_eq!(offset("%s", &["i64 0", "i32 1"]), Some(16));
    assert_eq!(offset("%s", &["i64 0", "i32 2", "i64 1"]), Some(40));
    assert_eq!(offset("%s", &["i64 1"]), Some(48));
    assert_eq!(offset("%\"p q\"", &["i64 0", "i32 1"]), Some(1));
    assert_eq!(offset("%\"p q\"", &["i64 0", "i32 2", "i32 2"]), Some(37));
    assert_eq!(offset("%r", &["i64 1"]), Some(16));
    assert_eq!(offset("%r", &["i64 0", "i32 2"]), Some(10));
    assert_eq!(offset("<2 x float>", &["i64 3"]), Some(24));
    assert_eq!(offset("i8 addrspace(2)*", &["i64 2"]), Some(16));
    assert_eq!(offset("i8", &["i64 -4"]), Some(-4));
    // A dynamic index ranges over its array's extent, and one over the source type is unbounded.
    assert_eq!(
        layouts.gep_range("%s", &["i64 0", "i32 2", "i64 %i"]),
        Some((32, 40))
    );
    assert_eq!(layouts.gep_range("<2 x float>", &["i64 %i"]), None);
    assert_eq!(
        layouts.gep_range("<2 x float>", &["i64 0", "i32 %i"]),
        Some((0, 4))
    );
    // A dynamic index, an index past the struct, and an opaque struct have no constant offset.
    assert_eq!(offset("%s", &["i64 0", "i32 %i"]), None);
    assert_eq!(offset("%s", &["i64 0", "i32 3"]), None);
    let opaque = layout::Layouts::new("%o = type opaque\n");
    assert_eq!(opaque.gep_offset("%o", &["i64 1"]), None);
    assert_eq!(opaque.gep_offset("%o", &["i64 0"]), Some(0));
    // Another data layout lays out only bytes.
    let other_text = format!("target datalayout = \"e-p:32:32\"\n{types}");
    let other = layout::Layouts::new(&other_text);
    assert_eq!(other.gep_offset("i8", &["i64 3"]), Some(3));
    assert_eq!(other.gep_offset("%s", &["i64 0", "i32 1"]), None);
    assert_eq!(other.gep_offset("ptr", &["i64 1"]), None);
}

/// A value two paths share holds what both make of it: a `select` of a pointer and an offset from
/// it is both, whichever arm comes first (the RRMb verifier's probes).
#[test]
fn a_value_two_paths_share_keeps_both_offsets() {
    let select = |root: &str, arms: &str| {
        format!(
            "{root}  %g = getelementptr inbounds i8, ptr addrspace(2) %b, i64 16\n  %m = select i1 \
             %c, {arms}\n"
        )
    };
    let buffer = "  %b = bitcast ptr addrspace(2) %0 to ptr addrspace(2)\n";
    let member = "  %b = load ptr addrspace(2), ptr addrspace(2) %0, align 8\n";
    for (root, members) in [(buffer, vec![]), (member, vec![(0, 0)])] {
        for arms in [
            "ptr addrspace(2) %b, ptr addrspace(2) %g",
            "ptr addrspace(2) %g, ptr addrspace(2) %b",
        ] {
            let ll = kernel(", i1 %c", &select(root, arms), "%m");
            assert_eq!(
                rate_map_placement_requirement(&ll, &reflection(&members)),
                PLACEMENT,
                "{ll}"
            );
        }
        // The shared value alone is placed.
        let ll = kernel(
            ", i1 %c",
            &select(root, "ptr addrspace(2) %b, ptr addrspace(2) %b"),
            "%m",
        );
        assert_eq!(
            rate_map_placement_requirement(&ll, &reflection(&members)),
            None,
            "{ll}"
        );
    }
}

/// A loop that steps the map pointer reads it at every multiple of the step, so it may be
/// anything; a loop that keeps its offsets settles.
#[test]
fn a_pointer_stepped_around_a_loop_is_not_placed() {
    let looped = |step: &str, map: &str| {
        kernel(
            ", i1 %c",
            &format!(
                "  br label %L\nL:\n  %m = phi ptr addrspace(2) [ %0, %E ], [ %k, %L ]\n{step}  br \
                 i1 %c, label %L, label %X\nX:\n"
            ),
            map,
        )
    };
    let stepped = "  %k = getelementptr inbounds i8, ptr addrspace(2) %m, i64 16\n";
    let ll = looped(stepped, "%m");
    assert_eq!(
        rate_map_placement_requirement(&ll, &reflection(&[])),
        PLACEMENT
    );
    // Each value changes at most `MAX_POINTERS + 2` times before it may be anything, and each
    // change evaluates its users again: three uses (`%k` and the call use `%m`, `%m` uses `%k`)
    // after the first evaluation of each of the three definitions.
    let mut work = Work::default();
    analyze(&ll, &reflection(&[]), &mut work);
    assert!(work.traced <= 3 * (MAX_POINTERS + 2) + 3, "{}", work.traced);
    // Out by 16 bytes and back each time round, or not moved at all.
    let back =
        "  %n = getelementptr inbounds i8, ptr addrspace(2) %m, i64 16\n  %k = getelementptr \
                inbounds i8, ptr addrspace(2) %n, i64 -16\n";
    let kept = "  %k = bitcast ptr addrspace(2) %m to ptr addrspace(2)\n";
    for step in [back, kept] {
        assert_eq!(
            rate_map_placement(&looped(step, "%m"), &reflection(&[])),
            placement(&[BUFFER_0], false),
            "{step}"
        );
    }
    // The pointer 16 bytes in, on the way round.
    assert_eq!(
        rate_map_placement_requirement(&looped(back, "%n"), &reflection(&[])),
        PLACEMENT
    );
}

/// A thread-memory array is followed only while every load and store of it is a whole pointer
/// slot: a load across two slots reads part of each pointer.
#[test]
fn a_thread_memory_array_is_read_by_whole_slots() {
    let array = |load: &str, store: &str| {
        format!(
            "  %a = alloca [2 x ptr addrspace(2)], align 8\n  store ptr addrspace(2) %0, ptr %a, \
             align 8\n  %e1 = getelementptr inbounds i8, ptr %a, i64 {store}\n  store ptr \
             addrspace(2) %1, ptr %e1, align 8\n  %s = getelementptr inbounds {load}\n  %m = load \
             ptr addrspace(2), ptr %s, align 8\n"
        )
    };
    let params = ", ptr addrspace(2) %1, i64 %i";
    let both = placement(&[BUFFER_0, MapPlace::Buffer { param: 1 }], false);
    for (load, store, expected) in [
        ("i8, ptr %a, i64 8", "8", Some(&both)),
        (
            "[2 x ptr addrspace(2)], ptr %a, i64 0, i64 %i",
            "8",
            Some(&both),
        ),
        ("i8, ptr %a, i64 4", "8", None),
        ("i8, ptr %a, i64 8", "4", None),
        ("i32, ptr %a, i64 %i", "8", None),
    ] {
        let ll = kernel(params, &array(load, store), "%m");
        match expected {
            Some(expected) => {
                assert_eq!(&rate_map_placement(&ll, &reflection(&[])), expected, "{ll}")
            }
            None => assert_eq!(
                rate_map_placement_requirement(&ll, &reflection(&[])),
                PLACEMENT,
                "{ll}"
            ),
        }
    }
}

/// Memory some pointer that is not constant memory views may be written, so a map there may not
/// be what the case authored: a member's buffer loaded as a device pointer, or a constant buffer
/// or argument buffer cast to one (the RRMb verifier's probe).
#[test]
fn memory_a_device_pointer_views_holds_no_placed_map() {
    let member = "  %l = load ptr addrspace(2), ptr addrspace(2) %0, align 8\n";
    for (why, prelude, map) in [
        (
            "the member's buffer loaded as a device pointer and written",
            "  %d = load ptr addrspace(1), ptr addrspace(2) %0, align 8\n  store float 1.0, ptr \
             addrspace(1) %d, align 4\n  %r2 = load ptr addrspace(1), ptr addrspace(2) %0, align \
             8\n  %l = addrspacecast ptr addrspace(1) %r2 to ptr addrspace(2)\n",
            "%l",
        ),
        (
            "the member's buffer cast to a device pointer",
            "  %l = load ptr addrspace(2), ptr addrspace(2) %0, align 8\n  %d = addrspacecast ptr \
             addrspace(2) %l to ptr addrspace(1)\n",
            "%l",
        ),
        (
            "the argument buffer cast to a device pointer",
            "  %l = load ptr addrspace(2), ptr addrspace(2) %0, align 8\n  %d = addrspacecast ptr \
             addrspace(2) %0 to ptr addrspace(1)\n",
            "%l",
        ),
    ] {
        let ll = kernel("", prelude, map);
        assert_eq!(
            rate_map_placement_requirement(&ll, &reflection(&[(0, 0)])),
            PLACEMENT,
            "{why}"
        );
    }
    let ll = kernel("", member, "%l");
    assert_eq!(
        rate_map_placement_requirement(&ll, &reflection(&[(0, 0)])),
        None
    );
    // A constant map buffer cast to a device pointer.
    let ll = kernel(
        "",
        "  %d = addrspacecast ptr addrspace(2) %0 to ptr addrspace(1)\n",
        "%0",
    );
    assert_eq!(
        rate_map_placement_requirement(&ll, &reflection(&[])),
        PLACEMENT
    );
    // Another member's buffer, viewed as device memory, does not hold the map.
    let ll = kernel(
        "",
        "  %a = getelementptr inbounds i8, ptr addrspace(2) %0, i64 8\n  %d = load ptr \
         addrspace(1), ptr addrspace(2) %a, align 8\n  store float 1.0, ptr addrspace(1) %d, align \
         4\n  %l = load ptr addrspace(2), ptr addrspace(2) %0, align 8\n",
        "%l",
    );
    assert_eq!(
        rate_map_placement(&ll, &reflection(&[(0, 0), (0, 8)])),
        placement(
            &[MapPlace::ArgumentBufferMember {
                param: 0,
                offset: 0
            }],
            false
        )
    );
}

/// A callee's writable view of the map stays known whatever else it views: each root resolves on
/// its own (the third RRMb verify's first low).
#[test]
fn a_known_writable_view_survives_an_unknown_one_beside_it() {
    let module = |main: &str, other: &str| {
        format!(
            "define void @main(ptr addrspace(2) %0, i64 %a) {{\n{main}  call void @w(ptr \
             addrspace(2) %0, ptr addrspace(2) %u)\n  %r = tail call fast <2 x float> \
             @air.map_screen_to_physical_coordinates.v2f32.p2i8.i32(<2 x float> \
             zeroinitializer, ptr addrspace(2) %0, i32 0)\n  ret void\n}}\ndefine internal void \
             @w(ptr addrspace(2) %p, ptr addrspace(2) %q) {{\n  %d = addrspacecast ptr \
             addrspace(2) %p to ptr addrspace(1)\n  store float 1.0, ptr addrspace(1) %d, align \
             4\n{other}  ret void\n}}\n"
        )
    };
    let unknown = "  %u = inttoptr i64 %a to ptr addrspace(2)\n";
    let returned = "  %u = call ptr addrspace(2) @source()\n";
    let device_q =
        "  %e = addrspacecast ptr addrspace(2) %q to ptr addrspace(1)\n  store float 1.0, \
                    ptr addrspace(1) %e, align 4\n";
    let member_q = "  %l = load ptr addrspace(2), ptr addrspace(2) %q, align 8\n  %e = \
                    addrspacecast ptr addrspace(2) %l to ptr addrspace(1)\n  store float 1.0, ptr \
                    addrspace(1) %e, align 4\n";
    for (main, other) in [
        (returned, device_q),
        (returned, member_q),
        (unknown, device_q),
        (unknown, member_q),
        ("  %l = load ptr addrspace(2), ptr addrspace(2) %0, align 8\n  %u = load ptr addrspace(2), ptr addrspace(2) %l, align 8\n", member_q),
    ] {
        let ll = module(main, other);
        assert_eq!(
            rate_map_placement_requirement(&ll, &reflection(&[])),
            PLACEMENT,
            "{ll}"
        );
    }
    // Without `@w`'s view of `%p`, the map is placed whatever unknown pointer `%q` is, when
    // nothing writes through it ([`an_untrusted_device_write_places_no_map`]); a call of code
    // the module does not hold places no map ([`an_unknown_write_target_is_not_trusted`]).
    let ll = module(unknown, device_q)
        .replace(
            "  %d = addrspacecast ptr addrspace(2) %p to ptr addrspace(1)\n  store float 1.0, ptr \
             addrspace(1) %d, align 4\n",
            "",
        )
        .replace(
            "store float 1.0, ptr addrspace(1) %e, align 4",
            "%x = load float, ptr addrspace(1) %e, align 4",
        );
    assert_eq!(
        rate_map_placement(&ll, &reflection(&[])),
        placement(&[BUFFER_0], false)
    );
}

/// A writable view keeps its roots through what the placement itself gives up on: a dynamic
/// offset, a member at a dynamic slot whose range overlaps the map's member, a `select` with an
/// unknown arm, and a `volatile` or `atomic` load (the third RRMb verify's second low).
#[test]
fn writable_views_keep_their_roots_through_dynamic_offsets() {
    // `%struct.Args` is a pointer at 0, a `<3 x float>` at 16, a pointer at 32, and two pointers
    // at 40 and 48.
    for (why, params, prelude, map, members) in [
        (
            "a dynamic offset of the map's buffer viewed as device memory",
            ", i64 %i",
            "  %g = getelementptr i8, ptr addrspace(2) %0, i64 %i\n  %d = addrspacecast ptr \
             addrspace(2) %g to ptr addrspace(1)\n  store i8 0, ptr addrspace(1) %d, align 1\n",
            "%0",
            vec![],
        ),
        (
            "a device pointer loaded from the member array at a dynamic index",
            ", i64 %i",
            "  %e = getelementptr inbounds %struct.Args, ptr addrspace(2) %0, i64 0, i32 3, i64 \
             %i\n  %d = load ptr addrspace(1), ptr addrspace(2) %e, align 8\n  store float 1.0, \
             ptr addrspace(1) %d, align 4\n  %s = getelementptr inbounds %struct.Args, ptr \
             addrspace(2) %0, i64 0, i32 3, i64 1\n  %l = load ptr addrspace(2), ptr \
             addrspace(2) %s, align 8\n",
            "%l",
            vec![(0, 40), (0, 48)],
        ),
        (
            "a select of the map's buffer and an unknown pointer viewed as device memory",
            ", i1 %c, i64 %a",
            "  %u = inttoptr i64 %a to ptr addrspace(2)\n  %s = select i1 %c, ptr addrspace(2) %0, \
             ptr addrspace(2) %u\n  %d = addrspacecast ptr addrspace(2) %s to ptr addrspace(1)\n  \
             store i8 0, ptr addrspace(1) %d, align 1\n",
            "%0",
            vec![],
        ),
        (
            "a volatile load of the member as a device pointer",
            "",
            "  %d = load volatile ptr addrspace(1), ptr addrspace(2) %0, align 8\n  store float \
             1.0, ptr addrspace(1) %d, align 4\n  %l = load ptr addrspace(2), ptr addrspace(2) \
             %0, align 8\n",
            "%l",
            vec![(0, 0)],
        ),
        (
            "an atomic load of the member as a device pointer",
            "",
            "  %d = load atomic ptr addrspace(1), ptr addrspace(2) %0 unordered, align 8\n  store \
             float 1.0, ptr addrspace(1) %d, align 4\n  %l = load ptr addrspace(2), ptr \
             addrspace(2) %0, align 8\n",
            "%l",
            vec![(0, 0)],
        ),
    ] {
        let ll = kernel(params, prelude, map);
        assert_eq!(
            rate_map_placement_requirement(&ll, &reflection(&members)),
            PLACEMENT,
            "{why}"
        );
    }
    // A device pointer array beside the map's members, at slots that do not overlap them (two
    // rows' shape), leaves the map placed; so does a volatile load of the map itself.
    let ll = kernel(
        ", i64 %i",
        "  %e = getelementptr inbounds %struct.Args, ptr addrspace(2) %0, i64 0, i32 3, i64 %i\n  \
         %d = load ptr addrspace(1), ptr addrspace(2) %e, align 8\n  store float 1.0, ptr \
         addrspace(1) %d, align 4\n  %l = load volatile ptr addrspace(2), ptr addrspace(2) %0, \
         align 8\n",
        "%l",
    );
    assert_eq!(
        rate_map_placement(&ll, &reflection(&[(0, 0), (0, 40), (0, 48)])),
        placement(
            &[MapPlace::ArgumentBufferMember {
                param: 0,
                offset: 0
            }],
            false
        )
    );
}

/// A pointer the analysis stops following escapes, and memory it roots may be written through
/// any pointer: the map stored to device memory and loaded back, turned into an integer or an
/// aggregate, returned, or passed to a function the module does not define (the third RRMb
/// verify's second low, shape (c)).
#[test]
fn a_map_pointer_that_escapes_is_not_placed() {
    for (why, params, prelude) in [
        (
            "stored to device memory and loaded back as a device pointer",
            ", ptr addrspace(1) %b",
            "  store ptr addrspace(2) %0, ptr addrspace(1) %b, align 8\n  %x = load ptr \
             addrspace(1), ptr addrspace(1) %b, align 8\n  store float 1.0, ptr addrspace(1) %x, \
             align 4\n",
        ),
        (
            "turned into an integer",
            "",
            "  %i = ptrtoint ptr addrspace(2) %0 to i64\n",
        ),
        (
            "put into an aggregate",
            "",
            "  %v = insertvalue { ptr addrspace(2) } poison, ptr addrspace(2) %0, 0\n",
        ),
        (
            "passed to a function the module does not define",
            "",
            "  call void @external(ptr addrspace(2) %0)\n",
        ),
        (
            "passed to an intrinsic that returns a pointer",
            "",
            "  %q = call ptr addrspace(2) @llvm.ptrmask.p2.i64(ptr addrspace(2) %0, i64 -1)\n",
        ),
    ] {
        let ll = kernel(params, prelude, "%0");
        assert_eq!(
            rate_map_placement_requirement(&ll, &reflection(&[])),
            PLACEMENT,
            "{why}"
        );
    }
    // Returned from a defined function.
    let ll = "define void @main(ptr addrspace(2) %0) {\n  %p = call ptr addrspace(2) @id(ptr \
              addrspace(2) %0)\n  %r = tail call fast <2 x float> \
              @air.map_screen_to_physical_coordinates.v2f32.p2i8.i32(<2 x float> \
              zeroinitializer, ptr addrspace(2) %0, i32 0)\n  ret void\n}\ndefine internal ptr \
              addrspace(2) @id(ptr addrspace(2) %p) {\n  ret ptr addrspace(2) %p\n}\n";
    assert_eq!(
        rate_map_placement_requirement(ll, &reflection(&[])),
        PLACEMENT
    );
    // Compared, or stored into a followed thread-memory array, it stays placed.
    let ll = kernel(
        ", ptr addrspace(2) %1",
        "  %c = icmp eq ptr addrspace(2) %0, %1\n  %a = alloca ptr addrspace(2), align 8\n  store \
         ptr addrspace(2) %0, ptr %a, align 8\n  %m = load ptr addrspace(2), ptr %a, align 8\n",
        "%m",
    );
    assert_eq!(
        rate_map_placement(&ll, &reflection(&[])),
        placement(&[BUFFER_0], false)
    );
}

/// A member's pointer read as bytes (a `load` of anything but one pointer, or any call the
/// module does not define, whatever its name) may be turned back into any pointer, so its pointee
/// is viewed over every slot the read touches (all of them for a call); loads of the bytes beside
/// it leave the map placed.
#[test]
fn a_member_read_as_bytes_is_viewed_over_the_slots_it_touches() {
    let member =
        "  %s = getelementptr inbounds %struct.Args, ptr addrspace(2) %0, i64 0, i32 2\n  \
                  %l = load ptr addrspace(2), ptr addrspace(2) %s, align 8\n";
    for (why, prelude) in [
        (
            "the member read as an integer, cast to device memory, and written",
            "  %i = load i64, ptr addrspace(2) %s, align 8\n  %d = inttoptr i64 %i to ptr \
             addrspace(1)\n  store float 1.0, ptr addrspace(1) %d, align 4\n",
        ),
        (
            "an integer read across the member's first bytes",
            "  %h = getelementptr inbounds i8, ptr addrspace(2) %0, i64 28\n  %i = load i64, ptr \
             addrspace(2) %h, align 4\n",
        ),
        (
            "a vector of pointers read over the member",
            "  %v = load <2 x ptr addrspace(2)>, ptr addrspace(2) %s, align 16\n",
        ),
        (
            "the argument buffer copied into device memory",
            "  call void @llvm.memcpy.p1.p2.i64(ptr addrspace(1) %b, ptr addrspace(2) %0, i64 40, \
             i1 false)\n",
        ),
        (
            "other bytes of the argument buffer copied, by a call whose reach is unknown",
            "  %f = getelementptr inbounds %struct.Args, ptr addrspace(2) %0, i64 0, i32 1\n  \
             call void @llvm.memcpy.p1.p2.i64(ptr addrspace(1) %b, ptr addrspace(2) %f, i64 16, \
             i1 false)\n",
        ),
        (
            "the member read by a masked load",
            "  %x = call <1 x i64> @llvm.masked.load.v1i64.p2(ptr addrspace(2) %s, i32 8, <1 x \
             i1> <i1 true>, <1 x i64> poison)\n",
        ),
        (
            "the argument buffer moved by an unknown length",
            "  call void @llvm.memmove.p1.p2.i64(ptr addrspace(1) %b, ptr addrspace(2) %0, i64 %n, \
             i1 false)\n",
        ),
        (
            "the member's pointer stored in a thread-memory array read as an integer",
            "  %a = alloca ptr addrspace(2), align 8\n  store ptr addrspace(2) %l, ptr %a, align \
             8\n  %i = load i64, ptr %a, align 8\n",
        ),
    ] {
        let ll = kernel(
            ", ptr addrspace(1) %b, i64 %n",
            &format!("{member}{prelude}"),
            "%l",
        );
        assert_eq!(
            rate_map_placement_requirement(&ll, &reflection(&[(0, 32)])),
            PLACEMENT,
            "{why}"
        );
    }
    // The vector before the member, and the integer ending at its first byte.
    let ll = kernel(
        "",
        &format!(
            "{member}  %f = getelementptr inbounds %struct.Args, ptr addrspace(2) %0, i64 0, i32 \
             1\n  %v = load <3 x float>, ptr addrspace(2) %f, align 16\n  %h = getelementptr \
             inbounds i8, ptr addrspace(2) %0, i64 28\n  %i = load i32, ptr addrspace(2) %h, \
             align 4\n"
        ),
        "%l",
    );
    assert_eq!(
        rate_map_placement(&ll, &reflection(&[(0, 32)])),
        placement(
            &[MapPlace::ArgumentBufferMember {
                param: 0,
                offset: 32
            }],
            false
        )
    );
}

/// A vector of pointers is the union of its elements: a vector of the map's pointers viewed as
/// device memory, stored by an intrinsic, selected, or passed to a function that takes one apart
/// views the map's buffer (the RRMb verifier's third pass, first low).
#[test]
fn a_vector_of_pointers_is_followed_as_its_elements() {
    let vector = "  %v = getelementptr i8, ptr addrspace(2) %0, <2 x i64> zeroinitializer\n";
    for (why, rest) in [
        (
            "cast to device memory and scattered into",
            "  %d = addrspacecast <2 x ptr addrspace(2)> %v to <2 x ptr addrspace(1)>\n  call void \
             @llvm.masked.scatter.v2i8.v2p1(<2 x i8> zeroinitializer, <2 x ptr addrspace(1)> %d, \
             i32 1, <2 x i1> <i1 true, i1 true>)\n",
        ),
        (
            "stored by a masked store and loaded back as a device pointer",
            "  call void @llvm.masked.store.v2p2.p1(<2 x ptr addrspace(2)> %v, ptr addrspace(1) \
             %b, i32 8, <2 x i1> <i1 true, i1 true>)\n  %x = load ptr addrspace(1), ptr \
             addrspace(1) %b, align 8\n  store i8 0, ptr addrspace(1) %x, align 1\n",
        ),
        (
            "selected, cast to device memory, and taken apart",
            "  %s = select <2 x i1> <i1 true, i1 false>, <2 x ptr addrspace(2)> %v, <2 x ptr \
             addrspace(2)> %v\n  %d = addrspacecast <2 x ptr addrspace(2)> %s to <2 x ptr \
             addrspace(1)>\n  %e = extractelement <2 x ptr addrspace(1)> %d, i32 0\n  store i8 \
             0, ptr addrspace(1) %e, align 1\n",
        ),
        (
            "passed to a function that takes it apart, casts, and stores",
            "  call void @w(<2 x ptr addrspace(2)> %v)\n",
        ),
    ] {
        let ll = format!(
            "{}define internal void @w(<2 x ptr addrspace(2)> %p) {{\n  %e = extractelement <2 x \
             ptr addrspace(2)> %p, i32 1\n  %d = addrspacecast ptr addrspace(2) %e to ptr \
             addrspace(1)\n  store i8 0, ptr addrspace(1) %d, align 1\n  ret void\n}}\n",
            kernel(", ptr addrspace(1) %b", &format!("{vector}{rest}"), "%0")
        );
        assert_eq!(
            rate_map_placement_requirement(&ll, &reflection(&[])),
            PLACEMENT,
            "{why}"
        );
    }
    // Another buffer's pointers gathered through leave the map placed.
    let ll = kernel(
        ", ptr addrspace(2) %1",
        "  %v = getelementptr i8, ptr addrspace(2) %1, <2 x i64> zeroinitializer\n  %g = call <2 \
         x i8> @llvm.masked.gather.v2i8.v2p2(<2 x ptr addrspace(2)> %v, i32 1, <2 x i1> <i1 true, \
         i1 true>, <2 x i8> poison)\n",
        "%0",
    );
    assert_eq!(
        rate_map_placement(&ll, &reflection(&[])),
        placement(&[BUFFER_0], false)
    );
}

/// A typed pointer with attributes after its type is a pointer: a callee's `i8 addrspace(2)*
/// nocapture readonly %p` viewed as device memory views the map's buffer (the RRMb verifier's
/// third pass, third low).
#[test]
fn a_typed_pointer_parameter_with_attributes_is_a_pointer() {
    let ll = "define void @main(i8 addrspace(2)* noundef readonly %0) {\n  call void @w(i8 \
              addrspace(2)* %0)\n  %r = tail call fast <2 x float> \
              @air.map_screen_to_physical_coordinates.v2f32.p2i8.i32(<2 x float> \
              zeroinitializer, i8 addrspace(2)* %0, i32 0)\n  ret void\n}\ndefine internal void \
              @w(i8 addrspace(2)* nocapture readonly %p) {\n  %d = addrspacecast i8 addrspace(2)* \
              %p to i8 addrspace(1)*\n  store i8 0, i8 addrspace(1)* %d, align 1\n  ret void\n}\n";
    assert_eq!(
        rate_map_placement_requirement(ll, &reflection(&[])),
        PLACEMENT
    );
    // Without the callee's view, the typed buffer is placed.
    let ll = ll.replace(
        "  %d = addrspacecast i8 addrspace(2)* %p to i8 addrspace(1)*\n  store i8 0, i8 \
         addrspace(1)* %d, align 1\n",
        "",
    );
    assert_eq!(
        rate_map_placement(&ll, &reflection(&[])),
        placement(&[BUFFER_0], false)
    );
}

/// One parser names every pointer type, typed or opaque, alone or in a vector, with attributes
/// or a value after it, and its address space.
#[test]
fn pointer_types_are_read_with_their_attributes() {
    let pointer = |text: &str| layout::pointer_type(text).map(|p| (p.space, p.vector));
    assert_eq!(
        pointer("ptr addrspace(2) noundef readonly"),
        Some((2, false))
    );
    assert_eq!(pointer("ptr %x"), Some((0, false)));
    assert_eq!(
        pointer("i8 addrspace(2)* nocapture readonly"),
        Some((2, false))
    );
    assert_eq!(pointer("float* %f"), Some((0, false)));
    assert_eq!(pointer("i8 addrspace(2)* addrspace(1)*"), Some((1, false)));
    assert_eq!(pointer("<4 x float> addrspace(2)*"), Some((2, false)));
    assert_eq!(pointer("<2 x ptr addrspace(1)> %v"), Some((1, true)));
    assert_eq!(pointer("<2 x i8 addrspace(3)*>"), Some((3, true)));
    assert_eq!(pointer("<4 x float>"), None);
    assert_eq!(pointer("i64 noundef"), None);
    assert_eq!(pointer("%struct.Args"), None);
}

/// A write through a device pointer, or a device pointer handed to code the trace does not see,
/// in a function the entry may reach, must trace to an entry parameter or to a resource member
/// the product reflects of an argument buffer. Any other device pointer may be any device address,
/// the map's included, so no map is placed: one made from an integer or a constant, loaded from
/// memory not known to hold authored pointers, or a call's result (the RRMb verifier's fourth
/// pass). A read, a hand-off the callee promises only to read, and a function the entry cannot
/// reach leave the map placed.
#[test]
fn an_untrusted_device_write_places_no_map() {
    let write = "  store i8 0, ptr addrspace(1) %p, align 1\n";
    let callee = "define internal void @w(ptr %t) {\n  %p = addrspacecast ptr %t to ptr \
                  addrspace(1)\n  store i8 0, ptr addrspace(1) %p, align 1\n  ret void\n}\n";
    let global = "@g = internal addrspace(3) global i64 undef, align 8\n";
    let params = ", ptr addrspace(2) %1, ptr addrspace(1) %b, i64 %n";
    for (why, prelude) in [
        (
            "an aggregate loaded through a loaded pointer (B1)",
            "  %l = load ptr addrspace(2), ptr addrspace(2) %1, align 8\n  %s = load { ptr \
             addrspace(1) }, ptr addrspace(2) %l, align 8\n  %p = extractvalue { ptr addrspace(1) \
             } %s, 0\n",
        ),
        (
            "an array loaded through a loaded pointer (B1)",
            "  %l = load ptr addrspace(2), ptr addrspace(2) %1, align 8\n  %s = load [1 x ptr \
             addrspace(1)], ptr addrspace(2) %l, align 8\n  %p = extractvalue [1 x ptr \
             addrspace(1)] %s, 0\n",
        ),
        (
            "an integer stored into thread memory and loaded as a device pointer (F1)",
            "  %a = alloca i64, align 8\n  store i64 %n, ptr %a, align 8\n  %p = load ptr \
             addrspace(1), ptr %a, align 8\n",
        ),
        (
            "an integer stored into a threadgroup global and loaded as a device pointer (F2)",
            "  store i64 %n, ptr addrspace(3) @g, align 8\n  %p = load ptr addrspace(1), ptr \
             addrspace(3) @g, align 8\n",
        ),
        (
            "a device pointer loaded from a device buffer's data (F2c)",
            "  %p = load ptr addrspace(1), ptr addrspace(1) %b, align 8\n",
        ),
        (
            "a member's struct copied into thread memory and its pointer loaded (S3)",
            "  %l = load ptr addrspace(2), ptr addrspace(2) %1, align 8\n  %a = alloca { ptr \
             addrspace(1) }, align 8\n  call void @llvm.memcpy.p0.p2.i64(ptr %a, ptr addrspace(2) \
             %l, i64 8, i1 false)\n  %p = load ptr addrspace(1), ptr %a, align 8\n",
        ),
        (
            "a constant integer cast to a device pointer (L2)",
            "  %p = getelementptr i8, ptr addrspace(1) inttoptr (i64 4096 to ptr addrspace(1)), \
             i64 0\n",
        ),
        (
            "a dynamic offset from a null device pointer (L2)",
            "  %p = getelementptr i8, ptr addrspace(1) null, i64 %n\n",
        ),
        (
            "an external function's result (L1)",
            "  %p = call ptr addrspace(1) @external()\n",
        ),
        (
            "an integer cast to a device pointer",
            "  %p = inttoptr i64 %n to ptr addrspace(1)\n",
        ),
        (
            "an integer cast to a thread pointer, then to a device one",
            "  %t = inttoptr i64 %n to ptr\n  %p = addrspacecast ptr %t to ptr addrspace(1)\n",
        ),
        (
            "a device pointer loaded through a member's pointer",
            "  %l = load ptr addrspace(2), ptr addrspace(2) %1, align 8\n  %p = load ptr \
             addrspace(1), ptr addrspace(2) %l, align 8\n",
        ),
    ] {
        let ll = format!(
            "{global}{}",
            kernel(params, &format!("{prelude}{write}"), "%0")
        );
        assert_eq!(
            rate_map_placement_requirement(&ll, &reflection(&[])),
            PLACEMENT,
            "{why}"
        );
    }
    let forged = "  %p = inttoptr i64 %n to ptr addrspace(1)\n";
    for (why, rest) in [
        (
            "an atomic through it",
            "  %o = atomicrmw add ptr addrspace(1) %p, i32 1 monotonic, align 4\n",
        ),
        (
            "handed to an external function",
            "  call void @external(ptr addrspace(1) %p)\n",
        ),
        (
            "a thread pointer made from it written by a callee",
            "  %t = inttoptr i64 %n to ptr\n  call void @w(ptr %t)\n",
        ),
    ] {
        let ll = format!(
            "{}{callee}",
            kernel(params, &format!("{forged}{rest}"), "%0")
        );
        assert_eq!(
            rate_map_placement_requirement(&ll, &reflection(&[])),
            PLACEMENT,
            "{why}"
        );
    }
    for (why, members, prelude) in [
        (
            "a device pointer made from an integer, only read",
            vec![],
            "  %p = inttoptr i64 %n to ptr addrspace(1)\n  %x = load i8, ptr addrspace(1) %p, \
             align 1\n",
        ),
        (
            "a device pointer made from an integer, handed to an intrinsic that only reads it",
            vec![],
            "  %p = inttoptr i64 %n to ptr addrspace(1)\n  %x = call float @air.read(ptr \
             addrspace(1) readonly captures(none) %p)\n",
        ),
        (
            "a write through an argument buffer's buffer member",
            vec![(1, 0)],
            "  %p = load ptr addrspace(1), ptr addrspace(2) %1, align 8\n  store i8 0, ptr \
             addrspace(1) %p, align 1\n",
        ),
    ] {
        let ll = kernel(params, prelude, "%0");
        assert_eq!(
            rate_map_placement(&ll, &reflection(&members)),
            placement(&[BUFFER_0], false),
            "{why}"
        );
    }
    // A constant integer cast to a device pointer, written through as it is (L2).
    let ll = kernel(
        "",
        "  store i8 0, ptr addrspace(1) inttoptr (i64 4096 to ptr addrspace(1)), align 1\n",
        "%0",
    );
    assert_eq!(
        rate_map_placement_requirement(&ll, &reflection(&[])),
        PLACEMENT
    );
    // The argument buffer's bytes at a slot the product reflects no member at are data.
    let ll = kernel(
        params,
        "  %e = getelementptr inbounds i8, ptr addrspace(2) %1, i64 8\n  %p = load ptr \
         addrspace(1), ptr addrspace(2) %e, align 8\n  store i8 0, ptr addrspace(1) %p, align 1\n",
        "%0",
    );
    assert_eq!(
        rate_map_placement_requirement(&ll, &reflection(&[(1, 0)])),
        PLACEMENT
    );
    // A function the entry does not reach.
    let ll = format!(
        "{}define void @other(i64 %a) {{\n  %d = inttoptr i64 %a to ptr addrspace(1)\n  store \
         i8 0, ptr addrspace(1) %d, align 1\n  ret void\n}}\n",
        kernel("", "", "%0")
    );
    assert_eq!(
        rate_map_placement(&ll, &reflection(&[])),
        placement(&[BUFFER_0], false)
    );
}

/// An array's loads read one summary of what is stored into it, which grows a bounded number of
/// times, so interleaved stores and loads at distinct offsets take linear work (the RRMb
/// verifier's third pass, fourth low: each load named every stored value, (n + 1)² visits).
#[test]
fn a_thread_memory_array_s_stores_and_loads_are_solved_in_linear_work() {
    let interleaved = |pairs: usize| {
        let mut body = "  %a = alloca ptr addrspace(2), align 8\n".to_string();
        for index in 1..=pairs {
            body.push_str(&format!(
                "  %g{index} = getelementptr inbounds i8, ptr addrspace(2) %0, i64 {index}\n  \
                 store ptr addrspace(2) %g{index}, ptr %a, align 8\n  %l{index} = load ptr \
                 addrspace(2), ptr %a, align 8\n"
            ));
        }
        kernel("", &body, &format!("%l{pairs}"))
    };
    let work = |pairs: usize| {
        let mut work = Work::default();
        let placed = analyze(&interleaved(pairs), &reflection(&[]), &mut work);
        assert_eq!(placed, placement(&[], true), "{pairs}");
        work.traced + work.visited
    };
    let counts = [100, 200, 400].map(|pairs| (pairs, work(pairs)));
    for (pairs, count) in counts {
        // The summary grows about `MAX_POINTERS` times, each time visiting every load (6,967 at
        // 100 pairs when written).
        assert!(
            count <= (MAX_POINTERS + 8) * pairs,
            "{pairs} pairs: {count}"
        );
    }
    assert!(counts[2].1 <= 2 * counts[1].1 + 16, "{counts:?}");
}

/// A member slot's pointer is trusted only when its argument buffer is constant memory no
/// writable view reaches: a slot the shader may write holds what it wrote, not what the case
/// authored (the RRMb verifier's fifth pass, first blocking finding).
#[test]
fn a_member_slot_the_shader_may_write_is_not_trusted() {
    let write = "  %p = load ptr addrspace(1), ptr addrspace(1) %1, align 8\n  store i8 0, ptr \
                 addrspace(1) %p, align 1\n";
    let device = ", ptr addrspace(1) %1, ptr addrspace(1) %b, i64 %n";
    for (why, params, prelude, write) in [
        (
            "a forged pointer stored into the slot",
            device,
            "  %f = inttoptr i64 %n to ptr addrspace(1)\n  store ptr addrspace(1) %f, ptr \
             addrspace(1) %1, align 8\n",
            write,
        ),
        (
            "a forged pointer exchanged into the slot",
            device,
            "  %f = inttoptr i64 %n to ptr addrspace(1)\n  %x = cmpxchg ptr addrspace(1) %1, ptr \
             addrspace(1) null, ptr addrspace(1) %f monotonic monotonic, align 8\n",
            write,
        ),
        (
            "bytes copied into the slot",
            device,
            "  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %1, ptr addrspace(1) %b, i64 8, \
             i1 false)\n",
            write,
        ),
        (
            "a pointer loaded from data stored into the slot",
            device,
            "  %q = load ptr addrspace(1), ptr addrspace(1) %b, align 8\n  store ptr addrspace(1) \
             %q, ptr addrspace(1) %1, align 8\n",
            write,
        ),
        (
            "a constant argument buffer written through a cast to device memory",
            ", ptr addrspace(2) %1, ptr addrspace(1) %b, i64 %n",
            "  %f = inttoptr i64 %n to ptr addrspace(1)\n  %c = addrspacecast ptr addrspace(2) %1 \
             to ptr addrspace(1)\n  store ptr addrspace(1) %f, ptr addrspace(1) %c, align 8\n",
            "  %p = load ptr addrspace(1), ptr addrspace(2) %1, align 8\n  store i8 0, ptr \
             addrspace(1) %p, align 1\n",
        ),
    ] {
        let ll = kernel(params, &format!("{prelude}{write}"), "%0");
        assert_eq!(
            rate_map_placement_requirement(&ll, &reflection(&[(1, 0)])),
            PLACEMENT,
            "{why}"
        );
    }
}

/// A write is trusted only when every place it may land traces to a known root: an unknown
/// pointer (a parameter some indirect call may pass, a load through `null`, `poison`) is not
/// trusted, and neither is code the module does not hold (an indirect call, an external
/// function, a constant-expression callee), which may write anywhere (the RRMb verifier's fifth
/// pass, second blocking finding and two lows).
#[test]
fn an_unknown_write_target_is_not_trusted() {
    let table = "@table = internal addrspace(2) constant [1 x ptr] [ptr @f]\n";
    let callee = "define internal void @f(ptr addrspace(1) %p) {\n  %q = load ptr addrspace(1), \
                  ptr addrspace(1) %p, align 8\n  store i8 0, ptr addrspace(1) %q, align 1\n  ret \
                  void\n}\n";
    let params = ", ptr addrspace(1) %b, ptr addrspace(2) %c, ptr %fp";
    for (why, globals, prelude, rest) in [
        (
            "an indirect call to a function that writes through a pointer loaded from its \
             parameter",
            table,
            "  call void %fp(ptr addrspace(1) %b)\n",
            callee,
        ),
        (
            "a visible function some table may call",
            "",
            "",
            "define void @f(ptr addrspace(1) %p) {\n  %q = load ptr addrspace(1), ptr \
             addrspace(1) %p, align 8\n  store i8 0, ptr addrspace(1) %q, align 1\n  ret void\n}\n\
             !air.visible_function = !{!9}\n!9 = !{ptr @f}\n",
        ),
        (
            "a constant data buffer handed to an indirect call, whose callee writes through a \
             device pointer loaded from it",
            "@table = internal addrspace(2) constant [1 x ptr] [ptr @h]\n",
            "  call void %fp(ptr addrspace(2) %c)\n",
            "define internal void @h(ptr addrspace(2) %p) {\n  %q = load ptr addrspace(1), ptr \
             addrspace(2) %p, align 8\n  store i8 0, ptr addrspace(1) %q, align 1\n  ret void\n}\n",
        ),
        (
            "a device pointer loaded from a constant data buffer handed to an indirect call",
            "",
            "  %p = load ptr addrspace(1), ptr addrspace(2) %c, align 8\n  call void %fp(ptr \
             addrspace(1) %p)\n",
            "",
        ),
        (
            "a write through a pointer loaded through null",
            "",
            "  %p = load ptr addrspace(1), ptr addrspace(2) null, align 8\n  store i8 0, ptr \
             addrspace(1) %p, align 1\n",
            "",
        ),
        (
            "a write through a frozen poison pointer",
            "",
            "  %p = freeze ptr addrspace(1) poison\n  store i8 0, ptr addrspace(1) %p, align 1\n",
            "",
        ),
        (
            "an external function handed a trusted pointer",
            "",
            "  call void @external(ptr addrspace(1) %b)\n",
            "",
        ),
        (
            "a constant-expression callee",
            "",
            "  call void inttoptr (i64 4096 to ptr)(ptr addrspace(1) %b)\n",
            "",
        ),
    ] {
        let ll = format!("{globals}{}{rest}", kernel(params, prelude, "%0"));
        assert_eq!(
            rate_map_placement_requirement(&ll, &reflection(&[])),
            PLACEMENT,
            "{why}"
        );
    }
    // Calls of defined functions and of intrinsics, and writes through the entry's own
    // pointers, leave the map placed.
    let ll = format!(
        "{}{}",
        kernel(
            params,
            "  call void @g(ptr addrspace(1) %b)\n  call void @llvm.memset.p1.i64(ptr \
             addrspace(1) %b, i8 0, i64 4, i1 false)\n",
            "%0"
        ),
        "define internal void @g(ptr addrspace(1) %p) {\n  store i8 0, ptr addrspace(1) %p, \
         align 1\n  ret void\n}\n"
    );
    assert_eq!(
        rate_map_placement(&ll, &reflection(&[])),
        placement(&[BUFFER_0], false)
    );
}

/// A device pointer cast to thread or threadgroup space still reaches device memory, so a write
/// through the cast is a write through the device pointer; a pointer made from an integer may be
/// any address whatever its space. Thread and threadgroup memory the module holds (an `alloca`,
/// a global) is trusted (the RRMb verifier's fifth pass, third blocking finding).
#[test]
fn a_write_through_a_cast_of_a_device_pointer_is_a_device_write() {
    for (why, prelude) in [
        (
            "a forged device pointer cast to thread space",
            "  %f = inttoptr i64 %n to ptr addrspace(1)\n  %t = addrspacecast ptr addrspace(1) %f \
             to ptr\n  store i8 0, ptr %t, align 1\n",
        ),
        (
            "a forged device pointer cast to threadgroup space",
            "  %f = inttoptr i64 %n to ptr addrspace(1)\n  %t = addrspacecast ptr addrspace(1) %f \
             to ptr addrspace(3)\n  store i8 0, ptr addrspace(3) %t, align 1\n",
        ),
        (
            "an integer cast to a threadgroup pointer",
            "  %p = inttoptr i64 %n to ptr addrspace(3)\n  store i8 0, ptr addrspace(3) %p, align \
             1\n",
        ),
        (
            "an integer cast to a thread pointer",
            "  %p = inttoptr i64 %n to ptr\n  store i8 0, ptr %p, align 1\n",
        ),
    ] {
        let ll = kernel(", i64 %n", prelude, "%0");
        assert_eq!(
            rate_map_placement_requirement(&ll, &reflection(&[])),
            PLACEMENT,
            "{why}"
        );
    }
    let ll = format!(
        "@shared = internal addrspace(3) global [4 x float] undef, align 4\n{}",
        kernel(
            ", float %x",
            "  %a = alloca [2 x float], align 4\n  store float %x, ptr %a, align 4\n  store float \
             %x, ptr addrspace(3) getelementptr inbounds ([4 x float], ptr addrspace(3) @shared, \
             i64 0, i64 2), align 4\n  store float %x, ptr addrspace(3) @shared, align 4\n",
            "%0"
        )
    );
    assert_eq!(
        rate_map_placement(&ll, &reflection(&[])),
        placement(&[BUFFER_0], false)
    );
}

/// A pointer loaded where its 8 bytes may span two member slots is no member's, so a write
/// through it is not trusted: one read 4 bytes into a slot, a dynamic index over 4-byte elements,
/// an aggregate holding its pointer between slots, and a `select` of a slot index and an odd
/// offset (the RRMb verifier's fifth pass, straddling loads). A dynamic index over whole slots
/// stays trusted.
#[test]
fn a_pointer_loaded_across_two_member_slots_is_not_trusted() {
    let write = "  store i8 0, ptr addrspace(1) %p, align 1\n";
    for (why, prelude) in [
        (
            "a pointer read 4 bytes into a slot",
            "  %e = getelementptr inbounds i8, ptr addrspace(2) %1, i64 4\n  %p = load ptr \
             addrspace(1), ptr addrspace(2) %e, align 4\n",
        ),
        (
            "a dynamic index over 4-byte elements",
            "  %e = getelementptr inbounds [4 x i32], ptr addrspace(2) %1, i64 0, i64 %i\n  %p = \
             load ptr addrspace(1), ptr addrspace(2) %e, align 4\n",
        ),
        (
            "an aggregate loaded 4 bytes into a slot",
            "  %e = getelementptr inbounds i8, ptr addrspace(2) %1, i64 4\n  %v = load { ptr \
             addrspace(1) }, ptr addrspace(2) %e, align 4\n  %p = extractvalue { ptr \
             addrspace(1) } %v, 0\n",
        ),
        (
            "a packed aggregate holding its pointer 4 bytes into a slot",
            "  %v = load <{ i32, ptr addrspace(1) }>, ptr addrspace(2) %1, align 4\n  %p = \
             extractvalue <{ i32, ptr addrspace(1) }> %v, 1\n",
        ),
        (
            "a slot index or an odd offset",
            "  %a = getelementptr inbounds [2 x ptr addrspace(1)], ptr addrspace(2) %1, i64 0, \
             i64 %i\n  %b = getelementptr inbounds i8, ptr addrspace(2) %1, i64 4\n  %e = select \
             i1 %c, ptr addrspace(2) %a, ptr addrspace(2) %b\n  %p = load ptr addrspace(1), ptr \
             addrspace(2) %e, align 4\n",
        ),
    ] {
        let ll = kernel(
            ", ptr addrspace(2) %1, i64 %i, i1 %c",
            &format!("{prelude}{write}"),
            "%0",
        );
        assert_eq!(
            rate_map_placement_requirement(&ll, &reflection(&[(1, 0), (1, 8)])),
            PLACEMENT,
            "{why}"
        );
    }
    for (why, prelude) in [
        (
            "a dynamic index over pointers",
            "  %e = getelementptr inbounds [2 x ptr addrspace(1)], ptr addrspace(2) %1, i64 0, \
             i64 %i\n  %p = load ptr addrspace(1), ptr addrspace(2) %e, align 8\n",
        ),
        (
            "a dynamic index over 8-byte integers",
            "  %e = getelementptr inbounds [2 x i64], ptr addrspace(2) %1, i64 0, i64 %i\n  %p = \
             load ptr addrspace(1), ptr addrspace(2) %e, align 8\n",
        ),
        (
            "an aggregate of two pointers",
            "  %v = load { ptr addrspace(1), ptr addrspace(1) }, ptr addrspace(2) %1, align 8\n  \
             %p = extractvalue { ptr addrspace(1), ptr addrspace(1) } %v, 1\n",
        ),
    ] {
        let ll = kernel(
            ", ptr addrspace(2) %1, i64 %i, i1 %c",
            &format!("{prelude}{write}"),
            "%0",
        );
        assert_eq!(
            rate_map_placement(&ll, &reflection(&[(1, 0), (1, 8)])),
            placement(&[BUFFER_0], false),
            "{why}"
        );
    }
}

/// Each named type's size, pointers, and where they lie are worked out once, so a type tree whose
/// every level names the one below twice costs its depth, not two to its depth (the RRMb
/// verifier's fifth pass: 50.9 s in debug for 15 levels and 200 loads).
#[test]
fn a_named_type_tree_is_laid_out_once_per_type() {
    let tree = |levels: usize, loads: usize| {
        let mut ll = "%t0 = type { ptr addrspace(1), i32 }\n".to_string();
        for level in 1..=levels {
            ll.push_str(&format!("%t{level} = type {{ %t{0}, %t{0} }}\n", level - 1));
        }
        let mut body = String::new();
        for load in 0..loads {
            body.push_str(&format!(
                "  %v{load} = load %t{levels}, ptr addrspace(2) %1, align 8\n"
            ));
        }
        ll + &kernel(", ptr addrspace(2) %1", &body, "%0")
    };
    for (levels, loads) in [(15, 200), (30, 200)] {
        let ll = tree(levels, loads);
        let layouts = layout::Layouts::new(&ll);
        let mut work = Work::default();
        let placed = analyze(&ll, &reflection(&[]), &mut work);
        assert_eq!(placed, placement(&[BUFFER_0], false), "{levels}");
        let ty = format!("%t{levels}");
        for _ in 0..loads {
            let _ = layouts.pointer_spaces(&ty);
            let _ = layouts.size(&ty);
            let _ = layouts.pointers_aligned(&ty);
        }
        // Each level at most once for its size, once for its pointers, and once for where they
        // lie.
        assert!(
            layouts.expansions() <= 3 * (levels + 1),
            "{levels}: {}",
            layouts.expansions()
        );
    }
}

/// A `select` chain over many pointer parameters coarsens its roots past `MAX_POINTERS`, after
/// which no pair of them is compared again: the pairwise work stays within
/// `MAX_POINTERS²` per value (the RRMb verifier's fifth pass: about P³ before).
#[test]
fn a_select_chain_over_many_parameters_compares_bounded_pairs() {
    let chain = |count: usize| {
        let params = (1..=count)
            .map(|index| format!(", ptr addrspace(2) %p{index}"))
            .collect::<String>();
        let mut body =
            "  %s0 = select i1 true, ptr addrspace(2) %0, ptr addrspace(2) %0\n".to_string();
        for index in 1..=count {
            body.push_str(&format!(
                "  %s{index} = select i1 true, ptr addrspace(2) %s{}, ptr addrspace(2) %p{index}\n",
                index - 1
            ));
        }
        kernel(&params, &body, &format!("%s{count}"))
    };
    for count in [100, 200, 400] {
        COMPARED.with(|compared| compared.set(0));
        let mut work = Work::default();
        let placed = analyze(&chain(count), &reflection(&[]), &mut work);
        assert_eq!(placed, placement(&[], true), "{count}");
        let pairs = COMPARED.with(Cell::get);
        assert!(
            pairs <= (count + 8) * MAX_POINTERS * MAX_POINTERS,
            "{count} parameters: {pairs}"
        );
    }
}

/// A map that may be an object the module holds is not placed: the object is the module's own
/// bytes, not what the case authored, and it never vanishes beside an entry buffer in a
/// `select`, a `phi`, a thread-memory array, or a callee's parameter (the RRMb verifier's sixth
/// pass, first blocking finding).
#[test]
fn a_map_the_module_may_hold_is_not_placed() {
    let globals = "@gmap = internal addrspace(2) constant [64 x i8] zeroinitializer\n@tg = \
                   internal addrspace(3) global [64 x i8] undef\n";
    for (why, prelude, map) in [
        (
            "a select of a global",
            "  %m = select i1 %c, ptr addrspace(2) %0, ptr addrspace(2) @gmap\n",
            "%m",
        ),
        (
            "a phi of a global",
            "  br i1 %c, label %a, label %b\na:\n  br label %b\nb:\n  %m = phi ptr addrspace(2) \
             [ %0, %entry ], [ @gmap, %a ]\n",
            "%m",
        ),
        (
            "a thread-memory array holding a global",
            "  %t = alloca [2 x ptr addrspace(2)], align 8\n  store ptr addrspace(2) %0, ptr %t, \
             align 8\n  %e = getelementptr inbounds [2 x ptr addrspace(2)], ptr %t, i64 0, i64 \
             1\n  store ptr addrspace(2) @gmap, ptr %e, align 8\n  %s = getelementptr inbounds \
             [2 x ptr addrspace(2)], ptr %t, i64 0, i64 %i\n  %m = load ptr addrspace(2), ptr \
             %s, align 8\n",
            "%m",
        ),
        (
            "a constant offset into a global",
            "  %m = select i1 %c, ptr addrspace(2) %0, ptr addrspace(2) getelementptr inbounds \
             ([64 x i8], ptr addrspace(2) @gmap, i64 0, i64 0)\n",
            "%m",
        ),
        (
            "an alloca cast to constant space",
            "  %a = alloca [64 x i8], align 8\n  %k = addrspacecast ptr %a to ptr addrspace(2)\n  \
             %m = select i1 %c, ptr addrspace(2) %0, ptr addrspace(2) %k\n",
            "%m",
        ),
        (
            "a threadgroup global cast to constant space",
            "  %k = addrspacecast ptr addrspace(3) @tg to ptr addrspace(2)\n  %m = select i1 %c, \
             ptr addrspace(2) %0, ptr addrspace(2) %k\n",
            "%m",
        ),
    ] {
        let ll = format!(
            "{globals}{}",
            kernel(", i1 %c, i64 %i", prelude, map).replacen(") {\n", ") {\nentry:\n", 1)
        );
        assert_eq!(
            rate_map_placement_requirement(&ll, &reflection(&[])),
            PLACEMENT,
            "{why}"
        );
    }
    // A callee called once with the entry's buffer and once with a global.
    let ll = format!(
        "{globals}{}define internal void @use(ptr addrspace(2) %m) {{\n  %r = tail call fast <2 \
         x float> @air.map_screen_to_physical_coordinates.v2f32.p2i8.i32(<2 x float> \
         zeroinitializer, ptr addrspace(2) %m, i32 0)\n  ret void\n}}\n",
        kernel(
            "",
            "  call void @use(ptr addrspace(2) %0)\n  call void @use(ptr addrspace(2) @gmap)\n",
            "%0"
        )
    );
    assert_eq!(
        rate_map_placement_requirement(&ll, &reflection(&[])),
        PLACEMENT,
        "a callee's parameter"
    );
}

/// Only the object a constant names is memory the module holds: an offset from `null` or
/// `poison` (whatever global address it adds), and a global the module only declares, may be any
/// address, so a write through one places no map (the RRMb verifier's sixth pass, second
/// blocking finding).
#[test]
fn a_constant_the_module_does_not_hold_is_not_trusted() {
    let globals = "@g = internal addrspace(3) global i64 undef\n@ext = external addrspace(1) \
                   global i8\n@own = internal addrspace(1) global [4 x i8] zeroinitializer\n";
    for (why, prelude) in [
        (
            "an offset from null by a global's address",
            "  %p = getelementptr i8, ptr addrspace(1) getelementptr (i8, ptr addrspace(1) null, \
             i64 ptrtoint (ptr addrspace(3) @g to i64)), i64 %n\n  store i8 0, ptr addrspace(1) \
             %p, align 1\n",
        ),
        (
            "an offset from poison as a store address",
            "  store i8 0, ptr addrspace(1) getelementptr (i8, ptr addrspace(1) poison, i64 \
             ptrtoint (ptr addrspace(3) @g to i64)), align 1\n",
        ),
        (
            "a global the module only declares",
            "  store i8 0, ptr addrspace(1) @ext, align 1\n",
        ),
    ] {
        let ll = format!("{globals}{}", kernel(", i64 %n", prelude, "%0"));
        assert_eq!(
            rate_map_placement_requirement(&ll, &reflection(&[])),
            PLACEMENT,
            "{why}"
        );
    }
    // A global the module defines, and an offset into it, are its own memory.
    let ll = format!(
        "{globals}{}",
        kernel(
            ", i64 %n",
            "  store i8 0, ptr addrspace(1) @own, align 1\n  store i8 0, ptr addrspace(1) \
             getelementptr inbounds ([4 x i8], ptr addrspace(1) @own, i64 0, i64 2), align 1\n  \
             %p = getelementptr i8, ptr addrspace(1) @own, i64 %n\n  store i8 0, ptr \
             addrspace(1) %p, align 1\n",
            "%0"
        )
    );
    assert_eq!(
        rate_map_placement(&ll, &reflection(&[])),
        placement(&[BUFFER_0], false)
    );
}

/// A callee's pointer parameter that a call leaves out, or fills with no pointer, may be
/// anything, so a write through it places no map (the RRMb verifier's sixth pass, third blocking
/// finding).
#[test]
fn a_parameter_no_call_fills_is_unknown() {
    let callee = "define internal void @g(ptr addrspace(1) %p) {\n  store i8 0, ptr addrspace(1) \
                  %p, align 1\n  ret void\n}\n";
    for (why, prelude) in [
        ("an integer", "  call void @g(i64 %n)\n"),
        ("no argument", "  call void @g()\n"),
    ] {
        let ll = format!("{}{callee}", kernel(", i64 %n", prelude, "%0"));
        assert_eq!(
            rate_map_placement_requirement(&ll, &reflection(&[])),
            PLACEMENT,
            "{why}"
        );
    }
    let ll = format!(
        "{}{callee}",
        kernel(
            ", ptr addrspace(1) %b",
            "  call void @g(ptr addrspace(1) %b)\n",
            "%0"
        )
    );
    assert_eq!(
        rate_map_placement(&ll, &reflection(&[])),
        placement(&[BUFFER_0], false)
    );
}

/// An operand's value is read by the text's grammar: a type named like a parameter inside a
/// constant expression is no parameter, and a `;` inside a quoted name starts no comment (the
/// RRMb verifier's sixth pass, first two lows).
#[test]
fn operands_are_read_by_the_module_text_s_grammar() {
    // `%1` names a type and a buffer: the map is an offset into a global, not buffer 1.
    let ll = format!(
        "%1 = type {{ i32 }}\n@gmap = internal addrspace(2) constant [64 x i8] \
         zeroinitializer\n{}",
        kernel(
            ", ptr addrspace(2) %1",
            "",
            "getelementptr inbounds ({ %1 }, ptr addrspace(2) @gmap, i64 0, i32 0, i32 0)"
        )
    );
    assert_eq!(
        rate_map_placement_requirement(&ll, &reflection(&[])),
        PLACEMENT,
        "a type named like a parameter"
    );
    // The write through a forged pointer follows a stored value whose quoted name holds a `;`.
    let ll = kernel(
        ", ptr addrspace(1) %b, i64 %n",
        "  %f = inttoptr i64 %n to ptr addrspace(1)\n  %\"v;w\" = getelementptr i8, ptr \
         addrspace(1) %b, i64 0\n  store ptr addrspace(1) %\"v;w\", ptr addrspace(1) %f, align \
         8\n",
        "%0",
    );
    assert_eq!(
        rate_map_placement_requirement(&ll, &reflection(&[])),
        PLACEMENT,
        "a quoted name holding a `;`"
    );
    // A quoted name holding a space or a comma is one value.
    let ll = kernel(
        ", ptr addrspace(1) %b",
        "  %\"a, b\" = getelementptr i8, ptr addrspace(1) %b, i64 4\n  store i8 0, ptr \
         addrspace(1) %\"a, b\", align 1\n",
        "%0",
    );
    assert_eq!(
        rate_map_placement(&ll, &reflection(&[])),
        placement(&[BUFFER_0], false)
    );
    // A `)` inside a quoted name ends no call's operand list (the RRMb verifier's seventh pass,
    // second low).
    let ll = kernel(
        "",
        "  %\"m)n\" = bitcast ptr addrspace(2) %0 to ptr addrspace(2)\n",
        "%\"m)n\"",
    );
    assert_eq!(
        rate_map_placement(&ll, &reflection(&[])),
        placement(&[BUFFER_0], false),
        "a quoted name holding a `)`"
    );
}

/// A value or parameter no path gives a value is unknown once the trace settles, and what names
/// it, loads it, or is passed it takes that in: a pointer only ever stored back into the array it
/// was loaded from vanished in a `select` with a placed value, through a callee, and as a write
/// target, and a parameter only a cycle of calls reaches vanished in the callee it passes it to
/// (the RRMb verifier's seventh pass, first low).
#[test]
fn a_value_no_path_gives_is_unknown() {
    let looped = |space: u32| {
        format!(
            "  %t = alloca ptr addrspace({space}), align 8\n  %u = load ptr addrspace({space}), \
             ptr %t, align 8\n  store ptr addrspace({space}) %u, ptr %t, align 8\n"
        )
    };
    let ll = kernel(
        ", i1 %c",
        &format!(
            "{}  %s = select i1 %c, ptr addrspace(2) %0, ptr addrspace(2) %u\n",
            looped(2)
        ),
        "%s",
    );
    assert_eq!(
        rate_map_placement_requirement(&ll, &reflection(&[])),
        PLACEMENT,
        "the map"
    );
    // `@g` maps its parameter, which `@main` fills with buffer 0 and `@f` with what `@main`
    // passes `@f`.
    let callees = |f: &str| {
        format!(
            "define internal void @f(ptr addrspace(2) %p) {{\n{f}  call void @g(ptr addrspace(2) \
             %p)\n  ret void\n}}\ndefine internal void @g(ptr addrspace(2) %q) {{\n  %r = tail \
             call fast <2 x float> @air.map_screen_to_physical_coordinates.v2f32.p2i8.i32(<2 x \
             float> zeroinitializer, ptr addrspace(2) %q, i32 0)\n  ret void\n}}\n"
        )
    };
    let ll = format!(
        "{}{}",
        kernel(
            "",
            &format!(
                "{}  call void @g(ptr addrspace(2) %0)\n  call void @f(ptr addrspace(2) %u)\n",
                looped(2)
            ),
            "%0"
        ),
        callees("")
    );
    assert_eq!(
        rate_map_placement_requirement(&ll, &reflection(&[])),
        PLACEMENT,
        "through a callee"
    );
    // `@f` is called by itself alone.
    let ll = format!(
        "{}{}",
        kernel("", "  call void @g(ptr addrspace(2) %0)\n", "%0"),
        callees("  call void @f(ptr addrspace(2) %p)\n")
    );
    assert_eq!(
        rate_map_placement_requirement(&ll, &reflection(&[])),
        PLACEMENT,
        "a parameter only a cycle of calls fills"
    );
    // Without the cycle, `@f` is passed buffer 0 too.
    let ll = format!(
        "{}{}",
        kernel(
            "",
            "  call void @g(ptr addrspace(2) %0)\n  call void @f(ptr addrspace(2) %0)\n",
            "%0"
        ),
        callees("")
    );
    assert_eq!(
        rate_map_placement(&ll, &reflection(&[])),
        placement(&[BUFFER_0], false)
    );
    let write = |target: &str| {
        kernel(
            ", ptr addrspace(1) %b, i1 %c",
            &format!(
                "{}  %s = select i1 %c, ptr addrspace(1) %b, ptr addrspace(1) {target}\n  store \
                 i8 0, ptr addrspace(1) %s, align 1\n",
                looped(1)
            ),
            "%0",
        )
    };
    assert_eq!(
        rate_map_placement_requirement(&write("%u"), &reflection(&[])),
        PLACEMENT,
        "a write target"
    );
    assert_eq!(
        rate_map_placement(&write("%b"), &reflection(&[])),
        placement(&[BUFFER_0], false)
    );
}

/// A pointer stepped by a dynamic index around a loop grows its range each time round; past
/// `MAX_POINTERS` changes its offsets are forgotten, so the fixed point ends.
#[test]
fn a_range_that_grows_around_a_loop_is_widened() {
    let ll = kernel(
        ", i1 %c, i64 %i",
        "  br label %L\nL:\n  %m = phi ptr addrspace(2) [ %0, %E ], [ %k, %L ]\n  %k = \
         getelementptr inbounds [4 x i8], ptr addrspace(2) %m, i64 0, i64 %i\n  br i1 %c, label \
         %L, label %X\nX:\n",
        "%m",
    );
    let mut work = Work::default();
    let placed = analyze(&ll, &reflection(&[]), &mut work);
    assert_eq!(placed, placement(&[], true));
    assert!(
        work.traced + work.visited <= 8 * (MAX_POINTERS + 4),
        "{work:?}"
    );
}

/// One function of `values` chained definitions, each a `bitcast` of the one before, laid out
/// last first so each settles only after the one it names.
fn long_function(values: usize) -> String {
    let mut body = String::new();
    for index in (1..=values).rev() {
        body.push_str(&format!(
            "  %v{index} = bitcast ptr addrspace(2) %v{} to ptr addrspace(2)\n",
            index - 1
        ));
    }
    kernel("", &body.replace("%v0 ", "%0 "), &format!("%v{values}"))
}

/// Within one function a value is evaluated again only when one it names changed: the work,
/// counting every evaluation and every user visited after a change, grows linearly with the
/// function (the third RRMb verify's third low: re-queueing every definition on each change made
/// it quadratic and no test saw it).
#[test]
fn the_values_of_one_function_are_solved_in_linear_work() {
    let work = |values: usize| {
        let mut work = Work::default();
        let placed = analyze(&long_function(values), &reflection(&[]), &mut work);
        assert_eq!(placed, placement(&[BUFFER_0], false), "{values}");
        work.traced + work.visited
    };
    let counts = [100, 200, 400].map(|values| (values, work(values)));
    for (values, count) in counts {
        // Each value is evaluated once in order (before what it names), once more when what it
        // names changes, and visited once as its one user's operand.
        assert!(count <= 4 * values + 8, "{values} values: {count}");
    }
    assert!(counts[2].1 <= 2 * counts[1].1 + 8, "{counts:?}");
}

/// The addresses derived from a thread-memory array are found by a worklist, each user visited
/// once per mention, however the chain is laid out (the third RRMb verify's fourth low: a fixed
/// point over the definitions took one pass per link).
#[test]
fn a_thread_memory_array_s_addresses_are_derived_in_linear_work() {
    let chained = |links: usize| {
        let mut body = "  %a = alloca [2 x ptr addrspace(2)], align 8\n".to_string();
        for index in (1..=links).rev() {
            let base = if index == 1 {
                "%a".to_string()
            } else {
                format!("%g{}", index - 1)
            };
            body.push_str(&format!(
                "  %g{index} = getelementptr inbounds i8, ptr {base}, i64 0\n"
            ));
        }
        body.push_str(&format!(
            "  store ptr addrspace(2) %0, ptr %g{links}, align 8\n  %m = load ptr addrspace(2), \
             ptr %g{links}, align 8\n"
        ));
        kernel("", &body, "%m")
    };
    let work = |links: usize| {
        let mut work = Work::default();
        let placed = analyze(&chained(links), &reflection(&[]), &mut work);
        assert_eq!(placed, placement(&[BUFFER_0], false), "{links}");
        work.traced + work.visited
    };
    let counts = [100, 200, 400].map(|links| (links, work(links)));
    for (links, count) in counts {
        assert!(count <= 6 * links + 16, "{links} links: {count}");
    }
    assert!(counts[2].1 <= 2 * counts[1].1 + 16, "{counts:?}");
}

/// A ring of phis, each stepped around its own loop, changes each value about 64 times, one
/// offset at a time. A root at one offset covers only itself, so no change compares the set's
/// roots pairwise (the RRMb verifier's sixth pass: 30.8, 54.1, and 114.6 s in debug at 500,
/// 1000, and 2000 phis, comparing each change's whole set).
#[test]
fn a_ring_of_stepped_phis_compares_no_single_roots() {
    for count in [50, 100] {
        let mut body = "  br label %b0\n".to_string();
        for index in 0..count {
            let back = (index + count - 1) % count;
            let next = (index + 1) % count;
            body.push_str(&format!(
                "b{index}:\n  %p{index} = phi ptr addrspace(2) [ %d, %entry ], [ %q{index}, \
                 %b{index} ], [ %p{back}, %b{back} ]\n  %q{index} = getelementptr inbounds i8, \
                 ptr addrspace(2) %p{index}, i64 4\n  br i1 %c, label %b{index}, label \
                 %b{next}\n"
            ));
        }
        let ll = kernel(", ptr addrspace(2) %d, i1 %c", &body, "%0").replacen(
            ") {\n",
            ") {\nentry:\n",
            1,
        );
        COMPARED.with(|compared| compared.set(0));
        let mut work = Work::default();
        assert_eq!(
            analyze(&ll, &reflection(&[]), &mut work),
            placement(&[BUFFER_0], false)
        );
        assert_eq!(COMPARED.with(Cell::get), 0, "{count}");
    }
}

/// A `phi` of `incoming` definitions laid out after it, a chain of casts of buffer 0 laid out
/// last first, so they change one at a time after it was evaluated, each before the next.
fn wide_phi(incoming: usize) -> String {
    let pairs = (1..=incoming)
        .map(|index| format!("[ %v{index}, %b{index} ]"))
        .collect::<Vec<_>>()
        .join(", ");
    let mut body = format!("  %p = phi ptr addrspace(2) {pairs}\n");
    for index in (1..=incoming).rev() {
        body.push_str(&format!(
            "  %v{index} = bitcast ptr addrspace(2) %v{} to ptr addrspace(2)\n",
            index - 1
        ));
    }
    kernel("", &body.replace("%v0 ", "%0 "), "%p")
}

/// A `phi` takes in each changed operand alone: the operand joins grow linearly with its incoming
/// values (the RRMb verifier's seventh pass, fourth low: joining every incoming value each time
/// one changed took about 16.2M joins and 123 s in debug at 4000).
#[test]
fn a_phi_takes_in_each_changed_operand_alone() {
    let joined = |incoming: usize| {
        let mut work = Work::default();
        let placed = analyze(&wide_phi(incoming), &reflection(&[]), &mut work);
        assert_eq!(placed, placement(&[BUFFER_0], false), "{incoming}");
        work.joined
    };
    let counts = [500, 1000].map(|incoming| (incoming, joined(incoming)));
    for (incoming, count) in counts {
        // The `phi` reads each incoming value once when evaluated and takes in each once when it
        // changes; each cast reads its operand once in order and once more when it changes.
        assert!(count <= 5 * incoming, "{incoming} incoming: {count}");
    }
    assert!(counts[1].1 <= 2 * counts[0].1 + 8, "{counts:?}");
}

/// The trace's slowest synthetic shapes, timed (not asserted: wall time depends on the machine).
/// The census run takes it in release: a ring of phis each stepped around its own loop, whose
/// values change about 64 times each; a load and a write at each of many member slots; a `select`
/// chain over many parameters; a deep tree of named types; and a `phi` whose many incoming values
/// change one at a time.
#[test]
#[ignore = "timing for the census run, in release"]
fn the_trace_s_slowest_shapes_in_release() {
    let ring = |count: usize, space: u32| {
        let mut body = "  br label %b0\n".to_string();
        for index in 0..count {
            let back = (index + count - 1) % count;
            let next = (index + 1) % count;
            body.push_str(&format!(
                "b{index}:\n  %p{index} = phi ptr addrspace({space}) [ %d, %entry ], [ \
                 %q{index}, %b{index} ], [ %p{back}, %b{back} ]\n  %q{index} = getelementptr \
                 inbounds i8, ptr addrspace({space}) %p{index}, i64 4\n"
            ));
            if space == 1 {
                body.push_str(&format!(
                    "  store i8 0, ptr addrspace(1) %q{index}, align 1\n"
                ));
            }
            body.push_str(&format!("  br i1 %c, label %b{index}, label %b{next}\n"));
        }
        kernel(&format!(", ptr addrspace({space}) %d, i1 %c"), &body, "%0").replacen(
            ") {\n",
            ") {\nentry:\n",
            1,
        )
    };
    let time = |what: String, ll: &str, reflection: &ShaderInterface| {
        let start = std::time::Instant::now();
        let mut work = Work::default();
        let placed = analyze(ll, reflection, &mut work);
        println!(
            "TIMING {what} {:?} unplaced={} work={}",
            start.elapsed(),
            placed.unplaced,
            work.traced + work.visited + work.resolved + work.joined
        );
    };
    for (space, kind) in [(2, "constant"), (1, "device")] {
        for count in [500, 1000, 2000] {
            time(
                format!("{kind} phi ring N={count}"),
                &ring(count, space),
                &reflection(&[]),
            );
        }
    }
    for count in [1000, 2000, 4000] {
        let mut body = String::new();
        for slot in 0..count {
            body.push_str(&format!(
                "  %e{slot} = getelementptr inbounds i8, ptr addrspace(2) %1, i64 {}\n  %p{slot} \
                 = load ptr addrspace(1), ptr addrspace(2) %e{slot}, align 8\n  store i8 0, ptr \
                 addrspace(1) %p{slot}, align 1\n",
                slot * 8
            ));
        }
        let members = (0..count)
            .map(|slot| (1, u32::try_from(slot * 8).unwrap()))
            .collect::<Vec<_>>();
        time(
            format!("member slots N={count}"),
            &kernel(", ptr addrspace(2) %1", &body, "%0"),
            &reflection(&members),
        );
    }
    let count = 1000;
    let params = (1..=count)
        .map(|index| format!(", ptr addrspace(2) %p{index}"))
        .collect::<String>();
    let mut body = "  %s0 = select i1 true, ptr addrspace(2) %0, ptr addrspace(2) %0\n".to_string();
    for index in 1..=count {
        body.push_str(&format!(
            "  %s{index} = select i1 true, ptr addrspace(2) %s{}, ptr addrspace(2) %p{index}\n",
            index - 1
        ));
    }
    time(
        format!("select chain P={count}"),
        &kernel(&params, &body, &format!("%s{count}")),
        &reflection(&[]),
    );
    let mut ll = "%t0 = type { ptr addrspace(1), i32 }\n".to_string();
    for level in 1..=15 {
        ll.push_str(&format!("%t{level} = type {{ %t{0}, %t{0} }}\n", level - 1));
    }
    let mut body = String::new();
    for load in 0..200 {
        body.push_str(&format!(
            "  %v{load} = load %t15, ptr addrspace(2) %1, align 8\n"
        ));
    }
    let ll = ll + &kernel(", ptr addrspace(2) %1", &body, "%0");
    time("type tree D=15 loads=200".into(), &ll, &reflection(&[]));
    for incoming in [4000, 8000] {
        time(
            format!("phi of chained incoming L={incoming}"),
            &wide_phi(incoming),
            &reflection(&[]),
        );
    }
}
