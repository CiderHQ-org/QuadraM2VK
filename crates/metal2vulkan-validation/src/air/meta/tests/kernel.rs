use super::*;

// `!air.kernel`: buffers + compute builtins (mirrors multi_add.air + harvested app kernels).
const KERN_LL: &str = r#"
!air.kernel = !{!14}
!14 = !{ptr @k, !15, !16}
!15 = !{}
!16 = !{!17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30}
!17 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"float", !"air.arg_name", !"a"}
!18 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.arg_type_name", !"float", !"air.arg_name", !"b"}
!19 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.arg_type_name", !"float", !"air.arg_name", !"c"}
!20 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
!21 = !{i32 4, !"air.threads_per_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"lsize"}
!22 = !{i32 5, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"lid"}
!23 = !{i32 6, !"air.threadgroups_per_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gsize"}
!24 = !{i32 7, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!25 = !{i32 8, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"thread_id"}
!26 = !{i32 9, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"lane"}
!27 = !{i32 10, !"air.simdgroup_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"simd_group"}
!28 = !{i32 11, !"air.threads_per_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"grid_size"}
!29 = !{i32 12, !"air.threads_per_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"simd_width"}
!30 = !{i32 13, !"air.simdgroups_per_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"num_simd_groups"}
"#;

/// Both Metal spellings of the threadgroup size have a lowering, and they are not the same one.
///
/// They differ under `dispatchThreads:`, where a final partial threadgroup reports a smaller
/// `threads_per_threadgroup` than the dispatch asked for -- and this translator reproduces that
/// dispatch by giving each region of `KernelDispatchPlan` its own local size, so the difference
/// survives into Vulkan rather than collapsing there. What both roles must keep is a lowering at
/// all: a parameter that read a zero instead of the size divided by it in every shader that used it.
#[test]
fn each_threadgroup_size_role_has_its_own_lowering() {
    let ll = |role: &str| {
        format!(
            r#"
define void @K(i32 %size) {{
  ret void
}}

!air.kernel = !{{!0}}
!0 = !{{ptr @K, !1, !2}}
!1 = !{{}}
!2 = !{{!3}}
!3 = !{{i32 0, !"air.{role}", !"air.arg_type_name", !"uint", !"air.arg_name", !"size"}}
"#
        )
    };
    for (role, expected) in [
        ("threads_per_threadgroup", KernRole::ThreadsPerThreadgroup),
        (
            "dispatch_threads_per_threadgroup",
            KernRole::DispatchThreadsPerThreadgroup,
        ),
    ] {
        let meta = parse_air_kernel_meta(&ll(role)).0.expect("kernel metadata");
        assert_eq!(
            meta.role_of(0),
            Some(&expected),
            "`air.{role}` decodes as its own threadgroup-size role"
        );
        assert!(
            meta.unmodelled_input_params.is_empty(),
            "`air.{role}` has a lowering, so it must not be reported as unmodelled"
        );
    }
}

#[test]
fn kernel_roles() {
    let m = parse_air_kernel_meta(KERN_LL).0.unwrap();
    assert_eq!(m.role_of(0), Some(&KernRole::Buffer(0)));
    assert_eq!(m.role_of(1), Some(&KernRole::Buffer(1)));
    assert_eq!(m.role_of(2), Some(&KernRole::Buffer(2)));
    assert_eq!(m.buffer_address_space(0), Some(1));
    assert_eq!(m.buffer_address_space(1), Some(1));
    assert_eq!(m.buffer_address_space(2), Some(1));
    assert_eq!(m.buffer_type_name(0), Some("float"));
    assert_eq!(m.buffer_type_name(1), Some("float"));
    assert_eq!(m.buffer_type_name(2), Some("float"));
    assert_eq!(m.role_of(3), Some(&KernRole::ThreadPositionInGrid));
    assert_eq!(m.role_of(4), Some(&KernRole::ThreadsPerThreadgroup));
    assert_eq!(m.role_of(5), Some(&KernRole::ThreadPositionInThreadgroup));
    assert_eq!(m.role_of(6), Some(&KernRole::ThreadgroupsPerGrid));
    assert_eq!(m.role_of(7), Some(&KernRole::ThreadgroupPositionInGrid));
    assert_eq!(m.role_of(8), Some(&KernRole::ThreadIndexInThreadgroup));
    assert_eq!(
        m.role_of(9),
        Some(&KernRole::ExecutionGroup {
            fact: ExecutionGroupFact::ThreadIndexInGroup,
            lanes: 32,
        })
    );
    assert_eq!(
        m.role_of(10),
        Some(&KernRole::ExecutionGroup {
            fact: ExecutionGroupFact::GroupIndexInThreadgroup,
            lanes: 32,
        })
    );
    assert_eq!(m.role_of(11), Some(&KernRole::ThreadsPerGrid));
    assert_eq!(
        m.role_of(12),
        Some(&KernRole::ExecutionGroup {
            fact: ExecutionGroupFact::ThreadsPerGroup,
            lanes: 32,
        })
    );
    assert_eq!(
        m.role_of(13),
        Some(&KernRole::ExecutionGroup {
            fact: ExecutionGroupFact::GroupsPerThreadgroup,
            lanes: 32,
        })
    );
}

/// Every marker the execution-group family decodes is one some stage's inventory calls modelled,
/// and exactly the stages that can answer it.
///
/// The decoder and the inventory are separate statements of one fact -- [`execution_group_role`]
/// decides what the emitter can lower, [`air_input_role_is_modelled`] decides what it refuses -- and
/// a member missing from the second is refused despite having a lowering, while one missing from the
/// first is bound to a zero and never reported. `air.quadgroups_per_threadgroup` was in neither, and
/// the whole family was in the kernel's alone: a fragment declaring
/// `[[thread_index_in_simdgroup]]`, a lane index that needs no threadgroup, was refused outright.
#[test]
fn every_execution_group_marker_is_listed_as_modelled() {
    for &(group, lanes) in AIR_EXECUTION_GROUPS {
        for (marker, fact) in [
            (
                format!("thread_index_in_{group}"),
                ExecutionGroupFact::ThreadIndexInGroup,
            ),
            (
                format!("{group}_index_in_threadgroup"),
                ExecutionGroupFact::GroupIndexInThreadgroup,
            ),
            (
                format!("{group}s_per_threadgroup"),
                ExecutionGroupFact::GroupsPerThreadgroup,
            ),
            (
                format!("threads_per_{group}"),
                ExecutionGroupFact::ThreadsPerGroup,
            ),
        ] {
            assert_eq!(
                execution_group_role(&marker),
                Some((fact, lanes)),
                "`air.{marker}` names a {group}, which is {lanes} threads wide"
            );
            for stage in [Stage::Kernel, Stage::Fragment, Stage::Vertex] {
                let answerable = stage == Stage::Kernel || !fact.needs_a_threadgroup();
                assert_eq!(
                    air_input_role_is_modelled(stage, &marker),
                    answerable,
                    "`air.{marker}` at {stage:?}: a fact about the group is answered at every \
                     stage, a fact about the threadgroup only where there is one"
                );
                // `dispatch_` asks what the DISPATCH requested. The two differ only for a final
                // partial threadgroup, which `vkCmdDispatch` cannot issue.
                let dispatched = format!("dispatch_{marker}");
                assert_eq!(
                    air_input_role_is_modelled(stage, &dispatched),
                    answerable,
                    "`air.{dispatched}` is `air.{marker}` under Vulkan"
                );
            }
            // No inventory may spell the family out a second time; the decoder is the one statement.
            for inventory in [FRAGMENT_INPUT_ROLES, VERTEX_INPUT_ROLES, KERNEL_INPUT_ROLES] {
                assert!(
                    !inventory.contains(&marker.as_str()),
                    "`air.{marker}` is decoded from its shape and must not also be listed"
                );
            }
        }
    }
    // Neighbours whose markers END the same way but name no execution group.
    for marker in [
        "threads_per_threadgroup",
        "thread_index_in_threadgroup",
        "threadgroups_per_grid",
        "threads_per_grid",
        "thread_position_in_grid",
    ] {
        assert_eq!(
            execution_group_role(marker),
            None,
            "`air.{marker}` is not an execution-group fact"
        );
    }
}

#[test]
fn function_table_roles_preserve_metal_buffer_indices() {
    let ll = r#"
define void @k(ptr addrspace(1) %visible, ptr addrspace(1) %intersection) { ret void }
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.visible_function_table", !"air.location_index", i32 7}
!4 = !{i32 1, !"air.intersection_function_table", !"air.location_index", i32 9}
"#;
    let meta = parse_air_kernel_meta(ll).0.unwrap();
    assert_eq!(meta.role_of(0), Some(&KernRole::VisibleFunctionTable(7)));
    assert_eq!(
        meta.role_of(1),
        Some(&KernRole::IntersectionFunctionTable(9))
    );
}

#[test]
fn kernel_texture_sampler_roles() {
    let ll = r#"
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 5, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<uint, read>", !"air.arg_name", !"inTex"}
!4 = !{i32 1, !"air.sampler", !"air.location_index", i32 2, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
"#;
    let m = parse_air_kernel_meta(ll).0.unwrap();
    assert_eq!(m.role_of(0), Some(&KernRole::Texture(5)));
    assert_eq!(m.texture_type_name(0), Some("texture2d<uint, read>"));
    assert_eq!(m.role_of(1), Some(&KernRole::Sampler(2)));
}

#[test]
fn kernel_stage_in_is_preserved_as_role() {
    let ll = r#"
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.stage_in", !"air.location_index", i32 7, i32 1, !"air.arg_type_name", !"uint3", !"air.arg_name", !"pointIndices"}
"#;
    let m = parse_air_kernel_meta(ll).0.unwrap();
    assert_eq!(m.role_of(0), Some(&KernRole::StageInput(7)));
}

#[test]
fn kernel_indirect_buffer_is_a_buffer_role() {
    let ll = r#"
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!4 = !{i32 0, i32 4, i32 0, !"uint", !"count", !"air.indirect_argument", !5, i32 4, i32 4, i32 0, !"uint", !"stride", !"air.indirect_argument", !6}
!5 = !{}
!6 = !{}
"#;
    let m = parse_air_kernel_meta(ll).0.unwrap();
    assert_eq!(m.role_of(1), Some(&KernRole::Buffer(3)));
    assert_eq!(m.buffer_address_space(1), Some(2));
    assert_eq!(m.buffer_type_size(1), Some(8));
    assert!(matches!(m.layout_of(1), Some(AirType::Struct(fields)) if fields.len() == 2));
    assert_eq!(m.embedded_arguments.len(), 2);
    assert_eq!(m.embedded_arguments[0].buffer_param_index, 1);
    assert_eq!(m.embedded_arguments[0].buffer_index, 3);
    assert_eq!(m.embedded_arguments[0].field_ordinal, 0);
    assert_eq!(m.embedded_arguments[0].field_offset, 0);
    assert_eq!(m.embedded_arguments[1].field_ordinal, 1);
    assert_eq!(m.embedded_arguments[1].field_offset, 4);
}

#[test]
fn kernel_acceleration_structure_introspection_uses_shadow_role() {
    let ll = r#"
define void @k(ptr addrspace(1) %as, ptr addrspace(1) %out) {
  %count = call i32 @air.get_instance_count_instance_acceleration_structure(ptr addrspace(1) %as)
  store i32 %count, ptr addrspace(1) %out
  ret void
}
declare i32 @air.get_instance_count_instance_acceleration_structure(ptr addrspace(1))

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.instance_acceleration_structure", !"air.location_index", i32 8, i32 1, !"air.read", !"air.arg_type_name", !"acceleration_structure<instancing>", !"air.arg_name", !"as"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let m = parse_air_kernel_meta(ll).0.unwrap();
    assert_eq!(
        m.role_of(0),
        Some(&KernRole::AccelerationStructureShadow(8))
    );
    assert_eq!(m.role_of(1), Some(&KernRole::Buffer(0)));
}

#[test]
fn kernel_primitive_acceleration_structure_is_always_a_literal_geometry_resource() {
    let ll = r#"
define void @k(ptr addrspace(1) %as, ptr addrspace(1) %out) {
  store i32 7, ptr addrspace(1) %out
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.primitive_acceleration_structure", !"air.location_index", i32 5, i32 1, !"air.read", !"air.arg_type_name", !"acceleration_structure<>", !"air.arg_name", !"as"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let meta = parse_air_kernel_meta(ll).0.unwrap();
    assert_eq!(
        meta.role_of(0),
        Some(&KernRole::PrimitiveAccelerationStructure(5))
    );
    assert_eq!(meta.role_of(1), Some(&KernRole::Buffer(0)));
}

#[test]
fn kernel_threadgroup_buffer_records_address_space() {
    let ll = r#"
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 3, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.struct_type_info", !4, !"air.arg_type_name", !"Temp", !"air.arg_name", !"temp"}
!4 = !{i32 0, i32 4, i32 0, !"packed_float1", !"value"}
"#;
    let m = parse_air_kernel_meta(ll).0.unwrap();
    assert_eq!(m.role_of(3), Some(&KernRole::Buffer(0)));
    assert_eq!(m.buffer_address_space(3), Some(3));
    assert!(matches!(m.layout_of(3), Some(AirType::Struct(fields)) if fields.len() == 1));
}

#[test]
fn kernel_threadgroup_buffer_infers_address_space_from_signature() {
    let ll = r#"
define void @k(ptr addrspace(3) %temp, i32 %i) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"temp"}
!4 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
"#;
    let m = parse_air_kernel_meta(ll).0.unwrap();
    assert_eq!(m.role_of(0), Some(&KernRole::Buffer(0)));
    assert_eq!(m.buffer_address_space(0), Some(3));
}

#[test]
fn kernel_metadata_is_decoded_once_without_forcing_gated_buffers() {
    let ll = r#"
define void @k(ptr addrspace(1) %optional) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.function_constant", !4, !"air.buffer", !"air.location_index", i32 7, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"optional"}
!4 = !{i32 0}
"#;

    KERNEL_META_DECODE_COUNT.with(|count| count.set(0));
    let (meta, entry) = parse_air_kernel_meta(ll);
    assert_eq!(KERNEL_META_DECODE_COUNT.with(std::cell::Cell::get), 1);
    assert_eq!(entry.as_deref(), Some("k"));
    assert_eq!(meta.unwrap().role_of(0), Some(&KernRole::Other));
}

#[test]
fn quoted_kernel_entry_name() {
    let ll = r#"
define void @"re::df::pack"(ptr addrspace(1) %out) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @"re::df::pack", !1, !2}
!1 = !{}
!2 = !{!3}
!3 = distinct !{i32 0, !"air.buffer", !"air.location_index", i32 7, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let (meta, entry) = parse_air_kernel_meta(ll);
    assert_eq!(entry.as_deref(), Some("re::df::pack"));
    let meta = meta.expect("kernel metadata");
    assert_eq!(meta.role_of(0), Some(&KernRole::Buffer(7)));
    assert_eq!(meta.buffer_address_space(0), Some(1));
}

/// `air.location_index` carries a PAIR of operands -- the Metal slot and the descriptor count --
/// and either may be a literal or a pointer to a function-constant global. All four combinations
/// occur in the corpus (112946 / 3404 / 1447 / 439 over 14579 sources), so a decode that scans
/// forward for "the next i32" or "the next @" reads the COUNT whenever the slot is spelled the
/// other way.
#[test]
fn the_location_index_operand_pair_is_read_by_position() {
    use super::{location_operands, LocationOperand};
    let global = "@_ZL32__metal_implicit_attr_int_expr_2.150".to_string();
    for (body, index, count) in [
        (
            r#"i32 0, !"air.texture", !"air.location_index", i32 4, i32 2, !"air.sample""#,
            LocationOperand::Literal(4),
            Some(LocationOperand::Literal(2)),
        ),
        (
            r#"i32 0, !"air.buffer", !"air.location_index", i32 7, ptr addrspace(2) @_ZL32__metal_implicit_attr_int_expr_2.150, !"air.read_write""#,
            LocationOperand::Literal(7),
            Some(LocationOperand::Global(global.clone())),
        ),
        (
            r#"i32 0, !"air.texture", !"air.location_index", ptr addrspace(2) @_ZL32__metal_implicit_attr_int_expr_2.150, i32 1, !"air.sample""#,
            LocationOperand::Global(global.clone()),
            Some(LocationOperand::Literal(1)),
        ),
        (
            r#"i32 0, !"air.texture", !"air.location_index", ptr addrspace(2) @_ZL32__metal_implicit_attr_int_expr_2.150, ptr addrspace(2) @_ZL32__metal_implicit_attr_int_expr_2.150, !"air.write""#,
            LocationOperand::Global(global.clone()),
            Some(LocationOperand::Global(global.clone())),
        ),
    ] {
        let decoded = location_operands(body).expect("every shape decodes");
        assert_eq!(decoded.index, index, "slot operand of `{body}`");
        assert_eq!(decoded.count, count, "count operand of `{body}`");
    }

    // A string operand may itself contain commas, so splitting the list on every comma would shift
    // every position after it.
    let quoted = r#"i32 0, !"air.texture", !"air.arg_type_name", !"array<texture2d<half, sample>, 2>", !"air.location_index", i32 4, i32 2"#;
    let decoded = location_operands(quoted).expect("decodes past a comma-bearing string");
    assert_eq!(decoded.index, LocationOperand::Literal(4));
    assert_eq!(decoded.count, Some(LocationOperand::Literal(2)));
}

/// A kernel's stage-input ABI is a function of its SIGNATURE, not of the order AIR happens to list
/// its argument nodes.
///
/// The same three arguments, listed forwards and backwards, must hand a consumer the same synthetic
/// buffer slots. Allocation used to walk `roles` in list order, so a producer that emitted the
/// argument nodes in a different order would have given the same shader different slots -- an ABI
/// that depends on a metadata layout nothing else reads. Every kernel argument list in the local
/// corpus is already in parameter-index order, so this pins a property rather than fixing an
/// observed difference.
#[test]
fn kernel_stage_input_slots_do_not_depend_on_the_argument_list_order() {
    const FORWARD: &str = r#"
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"float", !"air.arg_name", !"params"}
!4 = !{i32 1, !"air.stage_in", !"air.location_index", i32 7, i32 1, !"air.arg_type_name", !"uint3", !"air.arg_name", !"first"}
!5 = !{i32 2, !"air.stage_in", !"air.location_index", i32 9, i32 1, !"air.arg_type_name", !"uint3", !"air.arg_name", !"second"}
"#;
    const REVERSED: &str = r#"
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!5, !4, !3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"float", !"air.arg_name", !"params"}
!4 = !{i32 1, !"air.stage_in", !"air.location_index", i32 7, i32 1, !"air.arg_type_name", !"uint3", !"air.arg_name", !"first"}
!5 = !{i32 2, !"air.stage_in", !"air.location_index", i32 9, i32 1, !"air.arg_type_name", !"uint3", !"air.arg_name", !"second"}
"#;
    let forward = parse_air_kernel_meta(FORWARD).0.expect("kernel metadata");
    let reversed = parse_air_kernel_meta(REVERSED).0.expect("kernel metadata");
    assert_eq!(
        forward.stage_input_bindings(),
        reversed.stage_input_bindings(),
        "the same kernel listed in a different order got different stage-input slots"
    );
    // ...and the slots skip the buffer the kernel already binds at 1.
    assert_eq!(
        forward.stage_input_bindings(),
        HashMap::from([(1, 0), (2, 2)]),
    );
}

/// The slot a role occupies in the Metal buffer table is one answer, and the synthetic stage-input
/// allocator is its only consumer. A role that names a buffer index but reports `None` here would
/// have a stage input allocated on top of it.
#[test]
fn every_kernel_role_that_names_a_buffer_index_reports_it() {
    for (role, slot) in [
        (KernRole::Buffer(3), Some(3)),
        (KernRole::AccelerationStructureShadow(4), Some(4)),
        (KernRole::PrimitiveAccelerationStructure(5), Some(5)),
        (KernRole::PrimitiveAccelerationStructureShadow(6), Some(6)),
        (KernRole::VisibleFunctionTable(7), Some(7)),
        (KernRole::IntersectionFunctionTable(8), Some(8)),
        // A texture or sampler index is a different table, and a stage input's own `u32` is the AIR
        // attribute location the synthetic slot stands in for.
        (KernRole::Texture(0), None),
        (KernRole::Sampler(0), None),
        (KernRole::StageInput(0), None),
        (KernRole::ThreadPositionInGrid, None),
        (KernRole::Other, None),
    ] {
        assert_eq!(role.buffer_table_slot(), slot, "{role:?}");
    }
}

#[test]
fn stage_attribute_boundaries_preserve_quoted_commas_and_empty_tails() {
    let nodes = HashMap::new();
    for source in [
        r#"ptr @"entry,with,commas", !1, !2, !"early_fragment_tests""#,
        r#"ptr @"entry,with,commas", !1, !2, !"early_fragment_tests", "#,
    ] {
        assert!(matches!(
            stage_root_attributes(source, &nodes).as_slice(),
            [StageAttribute::EarlyFragmentTests]
        ));
    }
    assert!(matches!(
        stage_root_attributes(r#"ptr @entry, !1, !2, !"unknown,attribute""#, &nodes).as_slice(),
        [StageAttribute::Unrecognized(value)] if value == r#"!"unknown,attribute""#
    ));
}
