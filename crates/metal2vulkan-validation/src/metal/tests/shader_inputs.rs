use super::*;

#[test]
fn vector_function_constant_executes_all_authored_lanes_on_metal() {
    use crate::library_module::ResolvedLinkedFunctions;
    use crate::source::SourceRow;

    let air_ll = include_str!("../../../fixtures/public/kernel_vector_function_constant.ll");
    let source = SourceRow {
        air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
        stage: "Kernel".into(),
        entry: "kernel_vector_function_constant".into(),
        air_ll: air_ll.into(),
        blob_b64: Some(
            base64::engine::general_purpose::STANDARD
                .encode(compile_metal_fixture("kernel_vector_function_constant")),
        ),
        lib_sha256s: vec!["33".repeat(32)],
        label: "test/kernel-vector-function-constant.air".into(),
    };
    let case = crate::case::vector_function_constant_test_case(
        source.air_sha256.clone(),
        source.entry.clone(),
    );
    let reflection =
        crate::air::reflect::inspect_stage(source.air_ll.as_str(), crate::case::Stage::Kernel)
            .unwrap();
    let resources = LiteralResources::prepare(&case).unwrap();
    assert_eq!(
        platform::execute(
            &case,
            as_compiled(&source, None),
            &resources,
            &reflection,
            &interface_of(&source, &case),
            &ResolvedLinkedFunctions::default(),
            &BTreeMap::new(),
        )
        .unwrap(),
        10u32.to_le_bytes()
    );
}

#[test]
fn visible_function_table_executes_a_separately_linked_air_function() {
    use crate::case::{
        BufferResource, Comparison, Dispatch, ExecutionSafety, FunctionTableEntry,
        FunctionTableResource, OutputSelection, ResourceRole, Stage,
    };
    use crate::library_module::{
        LibraryModuleRow, ResolvedFunctionEntry, ResolvedFunctionTable, ResolvedLinkedFunctions,
    };
    use crate::source::SourceRow;

    let entry_blob = compile_metal_fixture("kernel_visible_function_table_word");
    let helper_blob = compile_metal_fixture("visible_function_add_one");
    let air_ll = r#"
define void @kernel_visible_function_table_word(ptr addrspace(1) %output, ptr addrspace(1) %functions) {
entry:
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @kernel_visible_function_table_word, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"uint"}
!4 = !{i32 1, !"air.visible_function_table", !"air.location_index", i32 1, i32 1, !"air.read", !"air.arg_type_name", !"visible_function_table"}
"#;
    let module_ll = "define i32 @visible_function_add_one(i32 %value) { ret i32 %value }";
    let module_sha256 = crate::hash::sha256_bytes(module_ll.as_bytes());
    let source = SourceRow {
        air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
        stage: "Kernel".into(),
        entry: "kernel_visible_function_table_word".into(),
        air_ll: air_ll.into(),
        blob_b64: Some(base64::engine::general_purpose::STANDARD.encode(entry_blob)),
        lib_sha256s: vec!["11".repeat(32)],
        label: "test/kernel-visible-function-table.air".into(),
    };
    let case = AuthoredCase {
        air_sha256: source.air_sha256.clone(),
        case_id: "test-visible-function-table".into(),
        name: "visible-function-table-smoke".into(),
        entry: source.entry.clone(),
        stage: Stage::Kernel,
        buffers: vec![BufferResource {
            binding: 0,
            role: ResourceRole::Output,
            bytes_b64: None,
            initial_bytes_b64: Some("q6urqw==".into()),
            rasterization_rate_map: None,
        }],
        argument_buffer_buffers: vec![],
        device_buffer_arrays: vec![],
        command_buffers: vec![],
        pipeline_states: vec![],
        threadgroup_memory: vec![],
        imageblock: None,
        fragment_imageblock: None,
        acceleration_structures: vec![],
        visible_function_references: vec![],
        visible_function_tables: vec![FunctionTableResource {
            binding: 1,
            size: 1,
            entries: vec![FunctionTableEntry {
                index: 0,
                module_sha256: module_sha256.clone(),
                function: "visible_function_add_one".into(),
            }],
        }],
        intersection_function_tables: vec![],
        argument_buffer_intersection_function_tables: vec![],
        textures: vec![],
        texture_arrays: vec![],
        argument_buffer_textures: vec![],
        argument_buffer_samplers: vec![],
        samplers: vec![],
        render_targets: vec![],
        depth_stencil: None,
        vertex_inputs: vec![],
        vertex_observation: None,
        varying_values: Vec::new(),
        kernel_stage_inputs: vec![],
        function_constants: vec![],
        dispatch: Some(Dispatch {
            grid: [1, 1, 1],
            threads_per_threadgroup: [1, 1, 1],
            stage_in_region: None,
        }),
        draw: None,
        tessellation: None,
        mesh_draw: None,
        output: OutputSelection::Buffer {
            binding: 0,
            offset: 0,
            length: 4,
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
    let module = LibraryModuleRow {
        module_sha256,
        air_ll: module_ll.into(),
        blob_b64: base64::engine::general_purpose::STANDARD.encode(helper_blob),
        lib_sha256s: source.lib_sha256s.clone(),
        label: "test/visible-function-add-one.air".into(),
    };
    let function_tables = ResolvedLinkedFunctions {
        references: vec![],
        visible: vec![ResolvedFunctionTable {
            binding: 1,
            size: 1,
            entries: vec![ResolvedFunctionEntry {
                index: 0,
                function: "visible_function_add_one".into(),
                module,
            }],
        }],
        intersection: vec![],
    };
    let reflection =
        crate::air::reflect::inspect_stage(source.air_ll.as_str(), crate::case::Stage::Kernel)
            .unwrap();
    let resources = LiteralResources::prepare(&case).unwrap();
    let output = platform::execute(
        &case,
        as_compiled(&source, None),
        &resources,
        &reflection,
        &interface_of(&source, &case),
        &function_tables,
        &BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(output, 42u32.to_le_bytes());
}

#[test]
fn opaque_triangle_intersection_entry_uses_the_native_sentinel_contract() {
    use crate::case::{
        AccelerationStructureKind, AccelerationStructureResource, BufferResource, Comparison,
        Dispatch, ExecutionSafety, IntersectionFunctionSignature, IntersectionFunctionTableEntry,
        IntersectionFunctionTableResource, OutputSelection, ResourceRole, Stage,
    };
    use crate::library_module::{
        ResolvedIntersectionFunctionEntry, ResolvedIntersectionFunctionTable,
        ResolvedLinkedFunctions,
    };
    use crate::source::SourceRow;

    let air_ll = include_str!("../../../fixtures/public/kernel_instance_as_intersect.ll");
    let source = SourceRow {
        air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
        stage: "Kernel".into(),
        entry: "instance_as_intersect".into(),
        air_ll: air_ll.into(),
        blob_b64: Some(
            base64::engine::general_purpose::STANDARD
                .encode(compile_metal_fixture("kernel_instance_as_intersect")),
        ),
        lib_sha256s: vec!["11".repeat(32)],
        label: "test/kernel-instance-as-intersect.air".into(),
    };
    let signature = vec![
        IntersectionFunctionSignature::Instancing,
        IntersectionFunctionSignature::TriangleData,
        IntersectionFunctionSignature::IntersectionFunctionBuffer,
    ];
    let case = AuthoredCase {
        air_sha256: source.air_sha256.clone(),
        case_id: "test-opaque-intersection-table".into(),
        name: "opaque-intersection-table-smoke".into(),
        entry: source.entry.clone(),
        stage: Stage::Kernel,
        buffers: vec![BufferResource {
            binding: 0,
            role: ResourceRole::Output,
            bytes_b64: None,
            initial_bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([0xabu8; 36])),
            rasterization_rate_map: None,
        }],
        argument_buffer_buffers: vec![],
        device_buffer_arrays: vec![],
        command_buffers: vec![],
        pipeline_states: vec![],
        threadgroup_memory: vec![],
        imageblock: None,
        fragment_imageblock: None,
        acceleration_structures: vec![AccelerationStructureResource {
            binding: 5,
            kind: AccelerationStructureKind::Instance,
            primitive_triangles_b64: None,
            child_references: vec![0],
        }],
        visible_function_references: vec![],
        visible_function_tables: vec![],
        intersection_function_tables: vec![IntersectionFunctionTableResource {
            binding: 6,
            size: 1,
            entries: vec![IntersectionFunctionTableEntry::OpaqueTriangle {
                index: 0,
                signature: signature.clone(),
            }],
        }],
        argument_buffer_intersection_function_tables: vec![],
        textures: vec![],
        texture_arrays: vec![],
        argument_buffer_textures: vec![],
        argument_buffer_samplers: vec![],
        samplers: vec![],
        render_targets: vec![],
        depth_stencil: None,
        vertex_inputs: vec![],
        vertex_observation: None,
        varying_values: Vec::new(),
        kernel_stage_inputs: vec![],
        function_constants: vec![],
        dispatch: Some(Dispatch {
            grid: [1, 1, 1],
            threads_per_threadgroup: [1, 1, 1],
            stage_in_region: None,
        }),
        draw: None,
        tessellation: None,
        mesh_draw: None,
        output: OutputSelection::Buffer {
            binding: 0,
            offset: 0,
            length: 36,
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
    let function_tables = ResolvedLinkedFunctions {
        references: vec![],
        visible: vec![],
        intersection: vec![ResolvedIntersectionFunctionTable {
            location: crate::library_module::ResolvedIntersectionFunctionTableLocation::Direct {
                binding: 6,
            },
            size: 1,
            entries: vec![ResolvedIntersectionFunctionEntry::OpaqueTriangle {
                index: 0,
                signature,
            }],
        }],
    };
    let reflection =
        crate::air::reflect::inspect_stage(source.air_ll.as_str(), crate::case::Stage::Kernel)
            .unwrap();
    let resources = LiteralResources::prepare(&case).unwrap();
    let output = platform::execute(
        &case,
        as_compiled(&source, None),
        &resources,
        &reflection,
        &interface_of(&source, &case),
        &function_tables,
        &BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(u32::from_le_bytes(output[0..4].try_into().unwrap()), 1);
}

#[test]
fn argument_buffer_intersection_table_is_encoded_at_reflected_field() {
    use crate::case::{
        AccelerationStructureKind, AccelerationStructureResource,
        ArgumentBufferIntersectionFunctionTableResource, BufferResource, Comparison, Dispatch,
        ExecutionSafety, IntersectionFunctionSignature, IntersectionFunctionTableEntry,
        OutputSelection, ResourceRole, Stage,
    };
    use crate::library_module::{
        ResolvedIntersectionFunctionEntry, ResolvedIntersectionFunctionTable,
        ResolvedIntersectionFunctionTableLocation, ResolvedLinkedFunctions,
    };
    use crate::source::SourceRow;

    // The fixture's interface as Metal's compiler states it (`xcrun metal -c` of
    // `kernel_argument_buffer_intersection_table.metal`, metalfe-32023.921): the argument buffer's
    // `table` member is an `air.intersection_function_table`, the role the executor encodes it by.
    let air_ll = r#"
%struct.RayResources = type { ptr addrspace(1) }
define void @argument_buffer_intersection_table(ptr addrspace(1) %as, ptr addrspace(2) %resources, ptr addrspace(1) %output) {
entry:
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @argument_buffer_intersection_table, !1, !2}
!1 = !{}
!2 = !{!3, !4, !7}
!3 = !{i32 0, !"air.instance_acceleration_structure", !"air.location_index", i32 5, i32 1, !"air.read", !"air.arg_type_name", !"acceleration_structure<instancing>", !"air.arg_name", !"acceleration_structure"}
!4 = !{i32 1, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 6, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !5, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"RayResources", !"air.arg_name", !"resources"}
!5 = !{i32 0, i32 8, i32 0, !"intersection_function_table<instancing, triangle_data>", !"table", !"air.indirect_argument", !6}
!6 = !{i32 0, !"air.intersection_function_table", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"intersection_function_table<instancing, triangle_data>", !"air.arg_name", !"table"}
!7 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"output"}
"#;
    let source =
        SourceRow {
            air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
            stage: "Kernel".into(),
            entry: "argument_buffer_intersection_table".into(),
            air_ll: air_ll.into(),
            blob_b64: Some(base64::engine::general_purpose::STANDARD.encode(
                compile_metal_fixture("kernel_argument_buffer_intersection_table"),
            )),
            lib_sha256s: vec!["11".repeat(32)],
            label: "test/kernel-argument-buffer-intersection-table.air".into(),
        };
    let signature = vec![
        IntersectionFunctionSignature::Instancing,
        IntersectionFunctionSignature::TriangleData,
        IntersectionFunctionSignature::IntersectionFunctionBuffer,
    ];
    let case = AuthoredCase {
        air_sha256: source.air_sha256.clone(),
        case_id: "test-argument-buffer-intersection-table".into(),
        name: "argument-buffer-intersection-table-smoke".into(),
        entry: source.entry.clone(),
        stage: Stage::Kernel,
        buffers: vec![
            BufferResource {
                binding: 0,
                role: ResourceRole::Output,
                bytes_b64: None,
                initial_bytes_b64: Some(
                    base64::engine::general_purpose::STANDARD.encode([0xabu8; 4]),
                ),
                rasterization_rate_map: None,
            },
            BufferResource {
                binding: 6,
                role: ResourceRole::Input,
                bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([0u8; 256])),
                initial_bytes_b64: None,
                rasterization_rate_map: None,
            },
        ],
        argument_buffer_buffers: vec![],
        device_buffer_arrays: vec![],
        command_buffers: vec![],
        pipeline_states: vec![],
        threadgroup_memory: vec![],
        imageblock: None,
        fragment_imageblock: None,
        acceleration_structures: vec![AccelerationStructureResource {
            binding: 5,
            kind: AccelerationStructureKind::Instance,
            primitive_triangles_b64: None,
            child_references: vec![0],
        }],
        visible_function_references: vec![],
        visible_function_tables: vec![],
        intersection_function_tables: vec![],
        argument_buffer_intersection_function_tables: vec![
            ArgumentBufferIntersectionFunctionTableResource {
                buffer_binding: 6,
                field_offset: 0,
                size: 1,
                entries: vec![IntersectionFunctionTableEntry::OpaqueTriangle {
                    index: 0,
                    signature: signature.clone(),
                }],
            },
        ],
        textures: vec![],
        texture_arrays: vec![],
        argument_buffer_textures: vec![],
        argument_buffer_samplers: vec![],
        samplers: vec![],
        render_targets: vec![],
        depth_stencil: None,
        vertex_inputs: vec![],
        vertex_observation: None,
        varying_values: Vec::new(),
        kernel_stage_inputs: vec![],
        function_constants: vec![],
        dispatch: Some(Dispatch {
            grid: [1, 1, 1],
            threads_per_threadgroup: [1, 1, 1],
            stage_in_region: None,
        }),
        draw: None,
        tessellation: None,
        mesh_draw: None,
        output: OutputSelection::Buffer {
            binding: 0,
            offset: 0,
            length: 4,
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
    let function_tables = ResolvedLinkedFunctions {
        references: vec![],
        visible: vec![],
        intersection: vec![ResolvedIntersectionFunctionTable {
            location: ResolvedIntersectionFunctionTableLocation::ArgumentBuffer {
                buffer_binding: 6,
                field_offset: 0,
            },
            size: 1,
            entries: vec![ResolvedIntersectionFunctionEntry::OpaqueTriangle {
                index: 0,
                signature,
            }],
        }],
    };
    let reflection =
        crate::air::reflect::inspect_stage(source.air_ll.as_str(), crate::case::Stage::Kernel)
            .unwrap();
    assert_eq!(reflection.argument_buffer_fields[0].buffer_index, 6);
    assert_eq!(reflection.argument_buffer_fields[0].field_offset, 0);
    let resources = LiteralResources::prepare(&case).unwrap();
    let output = platform::execute(
        &case,
        as_compiled(&source, None),
        &resources,
        &reflection,
        &interface_of(&source, &case),
        &function_tables,
        &BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(u32::from_le_bytes(output.try_into().unwrap()), 1);
}
