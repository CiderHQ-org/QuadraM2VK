use super::*;

const VERT_LL: &str = r#"
!air.vertex = !{!15}
!15 = !{ptr @V, !16, !19}
!16 = !{!17, !18}
!19 = !{!20, !21, !22}
!20 = !{i32 0, !"air.vertex_input", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float3", !"air.arg_name", !"position"}
!21 = !{i32 1, !"air.vertex_input", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!22 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 64, !"air.location_index", i32 4, i32 1}
"#;

#[test]
fn vertex_roles() {
    let m = parse_air_vertex_meta(VERT_LL).0.unwrap();
    assert_eq!(m.role_of(0), Some(&VertRole::VertexInput(0)));
    assert_eq!(m.role_of(1), Some(&VertRole::VertexInput(1)));
    assert_eq!(m.role_of(2), Some(&VertRole::Buffer(4)));
    assert_eq!(m.vertex_input_type(0), Some("float3"));
    assert_eq!(
        m.vertex_input_names.get(&0).map(String::as_str),
        Some("position")
    );
    assert_eq!(m.vertex_input_type(1), Some("float2"));
    assert_eq!(m.vertex_input_names.get(&1).map(String::as_str), Some("uv"));
}

// `[[vertex_id]]` / `[[instance_id]]` builtins -> VertexId/InstanceId roles (mirrors render_tri.air).
const VERT_BUILTIN_LL: &str = r#"
!air.vertex = !{!14}
!14 = !{ptr @vmain, !15, !17}
!15 = !{!16, !20, !21}
!16 = !{!"air.position", !"air.arg_type_name", !"float4"}
!17 = !{!18, !19}
!18 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
!19 = !{i32 1, !"air.instance_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"iid"}
!20 = !{!"air.viewport_array_index", !"air.arg_type_name", !"uint", !"air.arg_name", !"viewport"}
!21 = !{!"air.clip_distance", !"air.arg_type_name", !"float", !"air.arg_name", !"clip"}
"#;

#[test]
fn vertex_builtin_roles() {
    let m = parse_air_vertex_meta(VERT_BUILTIN_LL).0.unwrap();
    assert_eq!(m.role_of(0), Some(&VertRole::VertexId));
    assert_eq!(m.role_of(1), Some(&VertRole::InstanceId));
    assert_eq!(m.output_role_of(0), Some(&VertOutRole::Position));
    assert_eq!(m.output_role_of(1), Some(&VertOutRole::ViewportArrayIndex));
    assert_eq!(m.output_role_of(2), Some(&VertOutRole::ClipDistance));
}

/// `air.invariant` is decoded onto the member it sits on and nowhere else.
#[test]
fn invariance_is_decoded_per_output_member() {
    let plain = parse_air_vertex_meta(VERT_BUILTIN_LL).0.unwrap();
    assert!(
        !plain.output_is_invariant(0),
        "a position without the marker is not invariant"
    );

    let ll = VERT_BUILTIN_LL.replace(
        r#"!16 = !{!"air.position","#,
        r#"!16 = !{!"air.position", !"air.invariant","#,
    );
    assert_ne!(ll, VERT_BUILTIN_LL);
    let m = parse_air_vertex_meta(&ll).0.unwrap();
    assert!(m.output_is_invariant(0));
    assert_eq!(
        m.output_role_of(0),
        Some(&VertOutRole::Position),
        "the marker must not displace the role it qualifies"
    );
    assert!(
        !m.output_is_invariant(1),
        "the viewport index is not marked"
    );
    assert!(!m.output_is_invariant(2), "the clip distance is not marked");
}

#[test]
fn vertex_patch_contract_preserves_domain_control_points_and_system_locations() {
    let ll = r#"
!air.vertex = !{!0}
!0 = !{ptr @vmain, !1, !2, !10}
!1 = !{!3}
!2 = !{!4, !7, !8, !9}
!3 = !{!"air.position", !"air.arg_type_name", !"float4"}
!4 = !{i32 0, !"air.patch_control_point_input", !5, !6}
!5 = !{!"air.patch_control_point_function", ptr @control.MTL_CONTROL_POINT_FN}
!6 = !{!"air.location_index", i32 2, i32 1, !"air.arg_type_name", !"float3"}
!7 = !{i32 1, !"air.function_constant", !11, !"air.patch_input", !"air.location_index", i32 5, i32 1, !"air.arg_type_name", !"float4"}
!8 = !{i32 2, !"air.instance_id", !"air.arg_type_name", !"uint"}
!9 = !{i32 3, !"air.amplification_count", !"air.arg_type_name", !"ushort"}
!10 = !{!"air.patch", !"quad", !"air.patch_control_point", i32 16}
!11 = !{ptr addrspace(2) @fc, !"bool", !"enabled"}
"#;
    let meta = parse_air_vertex_meta(ll).0.unwrap();
    let tessellation = meta.tessellation.as_ref().unwrap();
    assert_eq!(tessellation.domain, PatchDomain::Quad);
    assert_eq!(tessellation.control_point_count, 16);
    assert_eq!(
        tessellation.control_point_function.as_deref(),
        Some("control.MTL_CONTROL_POINT_FN")
    );
    assert_eq!(tessellation.control_point_fields[0].location, 2);
    assert_eq!(meta.role_of(0), Some(&VertRole::PatchControlPoints));
    assert_eq!(meta.role_of(1), Some(&VertRole::PatchInput(5)));
    assert_eq!(meta.role_of(2), Some(&VertRole::InstanceId));
    assert_eq!(meta.role_of(3), Some(&VertRole::AmplificationCount));
    assert_eq!(
        meta.tessellation_system_input_location(&VertRole::InstanceId),
        Some(6)
    );
    assert_eq!(
        meta.tessellation_system_input_location(&VertRole::AmplificationCount),
        Some(8)
    );
}

#[test]
fn vertex_render_target_array_index_role() {
    let ll = r#"
!air.vertex = !{!0}
!0 = !{ptr @vmain, !1, !5}
!1 = !{!2, !3, !4}
!2 = !{!"air.position", !"air.arg_type_name", !"float4"}
!3 = !{!"air.render_target_array_index", !"air.arg_type_name", !"uint", !"air.arg_name", !"layer"}
!4 = !{!"air.vertex_output", !"generated(7varyingf)", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"varying"}
!5 = !{}
"#;
    let m = parse_air_vertex_meta(ll).0.unwrap();
    assert_eq!(m.output_role_of(0), Some(&VertOutRole::Position));
    assert_eq!(
        m.output_role_of(1),
        Some(&VertOutRole::RenderTargetArrayIndex)
    );
    assert_eq!(m.output_role_of(2), Some(&VertOutRole::Varying(0)));
    assert_eq!(m.output_varying_type(0), Some("float"));
    assert_eq!(
        m.output_varying_names.get(&0).map(String::as_str),
        Some("varying")
    );
    assert_eq!(
        m.output_varying_user_semantics.get(&0).map(String::as_str),
        Some("generated(7varyingf)")
    );
}

/// A `[[function_constant]]`-gated return member keeps the varying slot it occupies.
///
/// Vertex→fragment linkage is by `Location`, and the two stages are translated as separate modules
/// from the same Metal varying struct. The fragment decode reads a gated `air.fragment_input` as
/// the varying it is and numbers it; if the vertex decode drops the same member, every later
/// varying in that struct is numbered one lower on the vertex side. Both modules still validate,
/// and the fragment silently reads the wrong interpolant. Only an AIR-specialized-off predicate
/// (`air.function_constant_disabled`) removes a member, and specialization is applied to both
/// stages alike. The old engine's `tests/gated_varying_locations.rs` (deleted with it in 0.2.0)
/// pinned the same rule end to end on the emitted `Location` decorations.
#[test]
fn a_gated_vertex_output_keeps_the_varying_slot_it_declares() {
    let ll = r#"
!air.vertex = !{!0}
!0 = !{ptr @vmain, !1, !6}
!1 = !{!2, !3, !5}
!2 = !{!"air.position", !"air.arg_type_name", !"float4"}
!3 = !{!"air.function_constant", !4, !"air.vertex_output", !"air.arg_type_name", !"float4", !"air.arg_name", !"optional"}
!4 = !{ptr addrspace(2) @__metal_implicit_fc_pred_0, !"bool", !"enabled"}
!5 = !{!"air.vertex_output", !"air.arg_type_name", !"float", !"air.arg_name", !"varying"}
!6 = !{}
"#;
    let m = parse_air_vertex_meta(ll).0.unwrap();
    assert_eq!(m.output_role_of(0), Some(&VertOutRole::Position));
    assert_eq!(m.output_role_of(1), Some(&VertOutRole::Varying(0)));
    assert_eq!(m.output_role_of(2), Some(&VertOutRole::Varying(1)));

    // The specialized-off form is the one that removes the member, and it removes it from every
    // stage: `specialize_function_constant_metadata` rewrites the wrapper for the whole module.
    let disabled = parse_air_vertex_meta(&ll.replace(
        "!\"air.function_constant\", !4,",
        "!\"air.function_constant_disabled\", !4,",
    ))
    .0
    .unwrap();
    assert_eq!(
        disabled.output_role_of(1),
        Some(&VertOutRole::FunctionConstantDisabled)
    );
    assert_eq!(disabled.output_role_of(2), Some(&VertOutRole::Varying(0)));
}
