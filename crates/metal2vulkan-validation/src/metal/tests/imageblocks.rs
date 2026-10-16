use super::*;

#[test]
fn custom_fragment_imageblock_executes_on_metal() {
    use crate::library_module::ResolvedLinkedFunctions;
    use crate::source::SourceRow;

    let air_ll = include_str!("../../../fixtures/public/fragment_custom_imageblock.ll");
    let source = SourceRow {
        air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
        stage: "Fragment".into(),
        entry: "fragment_custom_imageblock".into(),
        air_ll: air_ll.into(),
        blob_b64: Some(
            base64::engine::general_purpose::STANDARD
                .encode(compile_metal_fixture("fragment_custom_imageblock")),
        ),
        lib_sha256s: vec!["12".repeat(32)],
        label: "test/fragment-custom-imageblock.air".into(),
    };
    let case =
        crate::case::fragment_imageblock_test_case(source.air_sha256.clone(), source.entry.clone());
    let reflection =
        crate::air::reflect::inspect_stage(source.air_ll.as_str(), crate::case::Stage::Fragment)
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
    assert_eq!(output, [0x00, 0x40]);
}

#[test]
fn implicit_imageblock_kernel_executes_as_a_tile_pipeline() {
    use crate::case::{
        BufferResource, Comparison, Dispatch, ExecutionSafety, ImageblockResource, OutputSelection,
        RenderTargetResource, ResourceRole, Stage, TextureFormat,
    };
    use crate::library_module::ResolvedLinkedFunctions;
    use crate::source::SourceRow;

    let air_ll = r#"
%imageblock = type opaque
define void @kernel_implicit_imageblock_half4(ptr addrspace(4) %block, <2 x i16> %position) {
entry:
  %value = call <4 x half> @air.load.implicit_imageblock.v4f16(i32 0, <2 x i16> %position, i32 0, i16 0)
  call void @air.store.implicit_imageblock.v4f16(<4 x half> %value, i32 0, <2 x i16> %position, i32 0, i16 0)
  ret void
}
declare <4 x half> @air.load.implicit_imageblock.v4f16(i32, <2 x i16>, i32, i16)
declare void @air.store.implicit_imageblock.v4f16(<4 x half>, i32, <2 x i16>, i32, i16)
!air.kernel = !{!0}
!0 = !{ptr @kernel_implicit_imageblock_half4, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.imageblock", !"implicit", !"air.struct_type_info", !4, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"imageblock<ImplicitColor, layout_implicit>"}
!4 = !{i32 0, i32 8, i32 0, !"half4", !"value", !"air.render_target", i32 0}
!5 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2"}
"#;
    let source = SourceRow {
        air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
        stage: "Kernel".into(),
        entry: "kernel_implicit_imageblock_half4".into(),
        air_ll: air_ll.into(),
        blob_b64: Some(
            base64::engine::general_purpose::STANDARD
                .encode(compile_metal_fixture("kernel_implicit_imageblock_half4")),
        ),
        lib_sha256s: vec!["11".repeat(32)],
        label: "test/kernel-implicit-imageblock-half4.air".into(),
    };
    let initial = [0x00, 0x3c, 0x00, 0x40, 0x00, 0x42, 0x00, 0x44];
    let case = AuthoredCase {
        air_sha256: source.air_sha256.clone(),
        case_id: "22".repeat(32),
        name: "implicit-imageblock-tile-smoke".into(),
        entry: source.entry.clone(),
        stage: Stage::Kernel,
        buffers: vec![BufferResource {
            binding: 0,
            role: ResourceRole::Output,
            bytes_b64: None,
            initial_bytes_b64: Some(
                base64::engine::general_purpose::STANDARD.encode([0u8; 16 * 16 * 8]),
            ),
            rasterization_rate_map: None,
        }],
        argument_buffer_buffers: vec![],
        device_buffer_arrays: vec![],
        command_buffers: vec![],
        pipeline_states: vec![],
        threadgroup_memory: vec![],
        imageblock: Some(ImageblockResource {
            dimensions: [16, 16],
            implicit_coverage: Some(crate::case::ImplicitImageblockCoverage::FullSingleSample),
        }),
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
            format: TextureFormat::Rgba16Float,
            dimensions: [16, 16],
            initial_bytes_b64: base64::engine::general_purpose::STANDARD
                .encode(initial.repeat(16 * 16)),
        }],
        depth_stencil: None,
        vertex_inputs: vec![],
        vertex_observation: None,
        varying_values: Vec::new(),
        kernel_stage_inputs: vec![],
        function_constants: vec![],
        dispatch: Some(Dispatch {
            grid: [16, 16, 1],
            threads_per_threadgroup: [16, 16, 1],
            stage_in_region: None,
        }),
        draw: None,
        tessellation: None,
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
    case.validate_literal_resources().unwrap();
    let reflection = crate::air::reflect::inspect(
        &source.air_ll,
        crate::case::Stage::Kernel,
        &source.entry,
        &[],
    )
    .unwrap();
    assert_eq!(reflection.implicit_imageblock_attachments.len(), 1);
    let resources = LiteralResources::prepare(&case).unwrap();
    let output = platform::execute(
        &case,
        as_compiled(&source, None),
        &resources,
        &reflection,
        &interface_of(&source, &case),
        &ResolvedLinkedFunctions::default(),
        &BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(output, [0x00, 0x40, 0x00, 0x42, 0x00, 0x44, 0x00, 0x45]);
}

#[test]
fn narrow_implicit_imageblock_executes_as_a_tile_pipeline() {
    use crate::library_module::ResolvedLinkedFunctions;
    use crate::source::SourceRow;
    use base64::Engine as _;

    let air_ll = include_str!("../../../fixtures/public/kernel_implicit_imageblock_half2.ll");
    let source = SourceRow {
        air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
        stage: "Kernel".into(),
        entry: "kernel_implicit_imageblock_half2".into(),
        air_ll: air_ll.into(),
        blob_b64: Some(
            base64::engine::general_purpose::STANDARD
                .encode(compile_metal_fixture("kernel_implicit_imageblock_half2")),
        ),
        lib_sha256s: vec!["11".repeat(32)],
        label: "test/kernel-implicit-imageblock-half2.air".into(),
    };
    let case = crate::case::narrow_implicit_imageblock_test_case(
        source.air_sha256.clone(),
        source.entry.clone(),
    );
    let reflection = crate::air::reflect::inspect(
        &source.air_ll,
        crate::case::Stage::Kernel,
        &source.entry,
        &[],
    )
    .unwrap();
    let resources = LiteralResources::prepare(&case).unwrap();
    let output = platform::execute(
        &case,
        as_compiled(&source, None),
        &resources,
        &reflection,
        &interface_of(&source, &case),
        &ResolvedLinkedFunctions::default(),
        &BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(output, [0x00, 0x3c, 0x00, 0x40]);
}
