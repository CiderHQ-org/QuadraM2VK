use super::*;

#[test]
fn attachmentless_fragment_executes_without_invented_output() {
    use crate::library_module::ResolvedLinkedFunctions;
    use crate::source::SourceRow;

    let air_ll = crate::case::ATTACHMENTLESS_FRAGMENT_AIR;
    let source = SourceRow {
        air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
        stage: "Fragment".into(),
        entry: "fragment_no_writes".into(),
        air_ll: air_ll.into(),
        blob_b64: Some(
            base64::engine::general_purpose::STANDARD
                .encode(compile_metal_fixture("fragment_no_writes")),
        ),
        lib_sha256s: vec!["10".repeat(32)],
        label: "test/fragment-no-writes.air".into(),
    };
    let case = crate::case::attachmentless_fragment_test_case(
        source.air_sha256.clone(),
        source.entry.clone(),
    );
    let reflection =
        crate::air::reflect::inspect_stage(source.air_ll.as_str(), crate::case::Stage::Fragment)
            .unwrap();
    let resources = LiteralResources::prepare(&case).unwrap();
    assert!(platform::execute(
        &case,
        as_compiled(&source, companion_of(&source, &case).as_ref()),
        &resources,
        &reflection,
        &interface_of(&source, &case),
        &ResolvedLinkedFunctions::default(),
        &BTreeMap::new(),
    )
    .unwrap()
    .is_empty());
}

#[test]
fn rasterization_disabled_vertex_executes_narrow_attributes() {
    use crate::library_module::ResolvedLinkedFunctions;
    use crate::source::SourceRow;

    let air_ll = include_str!("../../../fixtures/public/vertex_narrow_attributes.ll");
    let source = SourceRow {
        air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
        stage: "Vertex".into(),
        entry: "vertex_narrow_attributes".into(),
        air_ll: air_ll.into(),
        blob_b64: Some(
            base64::engine::general_purpose::STANDARD
                .encode(compile_metal_fixture("vertex_narrow_attributes")),
        ),
        lib_sha256s: vec!["11".repeat(32)],
        label: "test/vertex-side-effect.air".into(),
    };
    let case =
        crate::case::narrow_vertex_test_case(source.air_sha256.clone(), source.entry.clone());
    let reflection =
        crate::air::reflect::inspect_stage(source.air_ll.as_str(), crate::case::Stage::Vertex)
            .unwrap();
    let resources = LiteralResources::prepare(&case).unwrap();
    assert_eq!(
        platform::execute(
            &case,
            as_compiled(&source, companion_of(&source, &case).as_ref()),
            &resources,
            &reflection,
            &interface_of(&source, &case),
            &ResolvedLinkedFunctions::default(),
            &BTreeMap::new(),
        )
        .unwrap(),
        9u32.to_le_bytes()
    );
}

#[test]
fn combined_depth_stencil_outputs_use_separate_native_attachments() {
    use crate::library_module::ResolvedLinkedFunctions;
    use crate::source::SourceRow;

    let air_ll = include_str!("../../../fixtures/public/fragment_depth_stencil.ll");
    let source = SourceRow {
        air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
        stage: "Fragment".into(),
        entry: "fragment_depth_stencil".into(),
        air_ll: air_ll.into(),
        blob_b64: Some(
            base64::engine::general_purpose::STANDARD
                .encode(compile_metal_fixture("fragment_depth_stencil")),
        ),
        lib_sha256s: vec!["22".repeat(32)],
        label: "test/fragment-depth-stencil.air".into(),
    };
    let case = crate::case::combined_depth_stencil_test_case(
        source.air_sha256.clone(),
        source.entry.clone(),
    );
    let reflection =
        crate::air::reflect::inspect_stage(source.air_ll.as_str(), crate::case::Stage::Fragment)
            .unwrap();
    let resources = LiteralResources::prepare(&case).unwrap();
    assert_eq!(
        platform::execute(
            &case,
            as_compiled(&source, companion_of(&source, &case).as_ref()),
            &resources,
            &reflection,
            &interface_of(&source, &case),
            &ResolvedLinkedFunctions::default(),
            &BTreeMap::new(),
        )
        .unwrap(),
        [7]
    );
}

#[test]
fn authored_tessellation_executes_native_patch_inputs() {
    use crate::case::{
        AttributeFormat, AttributeInput, Comparison, ExecutionSafety, OutputSelection,
        RenderTargetResource, Stage, TessellationDraw, TessellationFactors, TextureFormat,
        VertexObservation,
    };
    use crate::library_module::ResolvedLinkedFunctions;
    use crate::source::SourceRow;

    // The entry's metadata as Metal's compiler writes it for `tessellation_literal.metal` (Xcode
    // 27, every deployment target from macOS 13 to 27): a control point's field and the patch
    // input carry a location index, the vertex output none.
    let air_ll = r#"
define <{ <4 x float>, <4 x float> }> @tessellation_literal(ptr %control, <4 x float> %patch, <2 x float> %coordinate) { ret <{ <4 x float>, <4 x float> }> zeroinitializer }
declare { <3 x float> } @control.MTL_CONTROL_POINT_FN(i32, ptr) section "air.externally_defined"
!air.vertex = !{!0}
!0 = !{ptr @tessellation_literal, !1, !2, !8}
!1 = !{!3, !12}
!2 = !{!4, !7, !9}
!3 = !{!"air.position", !"air.arg_type_name", !"float4"}
!4 = !{i32 0, !"air.patch_control_point_input", !5, !6}
!5 = !{!"air.patch_control_point_function", ptr @control.MTL_CONTROL_POINT_FN}
!6 = !{!"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float3"}
!7 = !{i32 1, !"air.patch_input", !"air.location_index", i32 4, i32 1, !"air.arg_type_name", !"float4"}
!8 = !{!"air.patch", !"quad", !"air.patch_control_point", i32 16}
!9 = !{i32 2, !"air.position_in_patch", !"air.arg_type_name", !"float2"}
!12 = !{!"air.vertex_output", !"user(locn0)", !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
"#;
    let source = SourceRow {
        air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
        stage: "Vertex".into(),
        entry: "tessellation_literal".into(),
        air_ll: air_ll.into(),
        blob_b64: Some(
            base64::engine::general_purpose::STANDARD
                .encode(compile_metal_fixture("tessellation_literal")),
        ),
        lib_sha256s: vec!["11".repeat(32)],
        label: "test/tessellation-literal.air".into(),
    };
    let case = AuthoredCase {
        air_sha256: source.air_sha256.clone(),
        case_id: "test-tessellation-literal".into(),
        name: "tessellation-literal-smoke".into(),
        entry: source.entry.clone(),
        stage: Stage::Vertex,
        buffers: vec![],
        argument_buffer_buffers: vec![],
        device_buffer_arrays: vec![],
        command_buffers: vec![],
        pipeline_states: vec![],
        threadgroup_memory: vec![],
        imageblock: None,
        fragment_imageblock: None,
        acceleration_structures: vec![],
        visible_function_references: vec![],
        visible_function_tables: vec![],
        intersection_function_tables: vec![],
        argument_buffer_intersection_function_tables: vec![],
        textures: vec![],
        texture_arrays: vec![],
        argument_buffer_textures: vec![],
        argument_buffer_samplers: vec![],
        samplers: vec![],
        render_targets: vec![RenderTargetResource {
            index: 0,
            format: TextureFormat::Rgba32Float,
            dimensions: [1, 1],
            initial_bytes_b64: base64::engine::general_purpose::STANDARD.encode([0; 16]),
        }],
        depth_stencil: None,
        vertex_inputs: vec![],
        vertex_observation: Some(VertexObservation::Varying { location: 0 }),
        varying_values: Vec::new(),
        kernel_stage_inputs: vec![],
        function_constants: vec![],
        dispatch: None,
        draw: None,
        tessellation: Some(TessellationDraw {
            factors: vec![TessellationFactors {
                edge_f16: vec![0x3c00; 4],
                inside_f16: vec![0x3c00; 2],
            }],
            instance_count: 1,
            amplification_count: 1,
            control_points: vec![AttributeInput {
                location: 0,
                format: AttributeFormat::Float3,
                stride: 12,
                dynamic_stride: false,
                bytes_b64: base64::engine::general_purpose::STANDARD.encode([0; 16 * 12]),
            }],
            patch_inputs: vec![AttributeInput {
                location: 4,
                format: AttributeFormat::Float4,
                stride: 16,
                dynamic_stride: false,
                bytes_b64: base64::engine::general_purpose::STANDARD
                    .encode([0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64]),
            }],
            partition_mode: Default::default(),
            output_winding: Default::default(),
            cull_mode: Default::default(),
        }),
        mesh_draw: None,
        output: OutputSelection::RenderTarget {
            index: 0,
            origin: [0, 0],
            dimensions: [1, 1],
        },
        compare: Comparison::Exact,
        execution_safety: ExecutionSafety::LoopFree,
        fuel: None,
        out_of_bounds: None,
        rationale: None,
        authored_by: Some("codex:gpt-5.6-sol".into()),
        render_pass_rate_map: None,
        raster_sample_count: None,
        public_source: None,
        no_effect: None,
        creation_bound: None,
    };
    let reflection =
        crate::air::reflect::inspect_stage(source.air_ll.as_str(), crate::case::Stage::Vertex)
            .unwrap();
    let resources = LiteralResources::prepare(&case).unwrap();
    let output = platform::execute(
        &case,
        as_compiled(&source, companion_of(&source, &case).as_ref()),
        &resources,
        &reflection,
        &interface_of(&source, &case),
        &ResolvedLinkedFunctions::default(),
        &BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(
        output,
        [0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64]
    );
}
