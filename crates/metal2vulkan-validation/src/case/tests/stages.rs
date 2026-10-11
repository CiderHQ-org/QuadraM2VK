use super::*;

#[test]
fn fragment_depth_stencil_attachment_has_exact_aspect_bytes_and_selection() {
    let mut case = example();
    case.case_id = "00".repeat(32);
    case.stage = Stage::Fragment;
    case.buffers.clear();
    case.dispatch = None;
    case.draw = Some(Draw {
        primitive: Primitive::Triangle,
        vertex_start: 0,
        vertex_count: 3,
        instance_count: 1,
        index_buffer: None,
        base_vertex: 0,
        base_instance: 0,
    });
    case.depth_stencil = Some(DepthStencilResource {
        dimensions: [1, 1],
        initial_depth_b64: Some("AAAAAA==".into()),
        initial_stencil_b64: None,
        state: None,
    });
    case.output = OutputSelection::Depth {
        origin: [0, 0],
        dimensions: [1, 1],
    };
    assert_eq!(case.validate_literal_resources(), Ok(()));

    case.depth_stencil.as_mut().unwrap().initial_depth_b64 = Some("AA==".into());
    assert!(case
        .validate_literal_resources()
        .unwrap_err()
        .iter()
        .any(|error| error.contains("expected 4")));
}

#[test]
fn tessellation_system_values_have_one_integer_scalar_contract() {
    assert!(AttributeFormat::Ushort.supports_tessellation_system_value());
    assert!(AttributeFormat::Int.supports_tessellation_system_value());
    assert!(!AttributeFormat::Ushort2.supports_tessellation_system_value());
    assert!(!AttributeFormat::Float.supports_tessellation_system_value());
    assert!(!AttributeFormat::Uchar.supports_tessellation_system_value());
}

#[test]
fn captured_function_constant_abi_shapes_are_exactly_authorable() {
    assert_eq!(
        ScalarType::from_metal_abi_type_encoding("b"),
        Some((ScalarType::Bool, 1))
    );
    assert_eq!(
        ScalarType::from_metal_abi_type_encoding("Dv4_j"),
        Some((ScalarType::U32, 4))
    );
    assert_eq!(
        ScalarType::from_metal_abi_type_encoding("Dv3_Dh"),
        Some((ScalarType::F16, 3))
    );
    assert!(ScalarType::from_metal_abi_type_encoding("Dv8_j").is_none());
}

#[test]
fn vertex_side_effect_case_uses_draw_without_raster_attachments() {
    let mut case = example();
    case.stage = Stage::Vertex;
    case.dispatch = None;
    case.draw = Some(Draw {
        primitive: Primitive::Point,
        vertex_start: 0,
        vertex_count: 1,
        instance_count: 1,
        index_buffer: None,
        base_vertex: 0,
        base_instance: 0,
    });
    case.case_id = case.computed_case_id().unwrap();
    case.validate_literal_resources().unwrap();
    assert!(case.is_rasterization_disabled_vertex());

    case.render_targets.push(RenderTargetResource {
        index: 0,
        format: TextureFormat::Rgba32Float,
        dimensions: [1, 1],
        initial_bytes_b64: "q6urq6urq6urq6urq6urqw==".into(),
    });
    case.case_id = case.computed_case_id().unwrap();
    assert!(case
        .validate_literal_resources()
        .unwrap_err()
        .iter()
        .any(|error| error.contains("does not accept attachments")));
}

/// A draw's base vertex and base instance are a vertex case's, without vertex inputs, and a base
/// vertex an indexed draw's (a draw without an index buffer starts at its vertex start). Zero
/// bases serialize as before the fields existed, so a case without them keeps its identity; a
/// nonzero base is part of it.
#[test]
fn draw_bases_are_an_attribute_free_vertex_case_s() {
    let mut case = crate::case::narrow_vertex_test_case("11".repeat(32), "vertex_main".into());
    case.vertex_inputs.clear();
    let before = case.computed_case_id().unwrap();
    let json = serde_json::to_string(&case.draw).unwrap();
    assert!(!json.contains("base_"), "{json}");
    let errors = |case: &AuthoredCase| {
        let mut case = case.clone();
        case.case_id = case.computed_case_id().unwrap();
        case.validate_literal_resources()
            .err()
            .unwrap_or_default()
            .join("\n")
    };
    {
        let draw = case.draw.as_mut().unwrap();
        draw.vertex_start = 0xffff_fffa;
        draw.base_instance = 7;
    }
    assert_eq!(errors(&case), "");
    assert_ne!(case.computed_case_id().unwrap(), before);
    let mut based = case.clone();
    based.draw.as_mut().unwrap().base_vertex = -3;
    assert!(errors(&based).contains("draw.base_vertex is an indexed draw's"));
    based.draw.as_mut().unwrap().vertex_start = 0;
    based.draw.as_mut().unwrap().index_buffer = Some(IndexBufferLiteral {
        index_type: IndexType::Uint16,
        offset: 0,
        bytes_b64: "BQAGAAcACAA=".into(),
    });
    assert_eq!(errors(&based), "");
    let mut fetched = case.clone();
    fetched.vertex_inputs = vec![AttributeInput {
        location: 0,
        format: AttributeFormat::Uint,
        stride: 4,
        dynamic_stride: false,
        bytes_b64: "AAAAAA==".into(),
    }];
    assert!(errors(&fetched).contains("without vertex_inputs"));
    let mut kernel = example();
    kernel.draw = case.draw.clone();
    assert!(errors(&kernel).contains("are a vertex case's"));
}

/// A patch draw may run with rasterization disabled, its post-tessellation vertex function observed
/// through a buffer it stores to (the domain-point cases), under the same attachment rule as any
/// other rasterization-disabled vertex case.
#[test]
fn a_patch_draw_may_run_with_rasterization_disabled() {
    let mut case = example();
    case.stage = Stage::Vertex;
    case.dispatch = None;
    case.tessellation = Some(TessellationDraw {
        factors: vec![TessellationFactors {
            edge_f16: vec![0x3c00; 4],
            inside_f16: vec![0x3c00; 2],
        }],
        instance_count: 1,
        amplification_count: 1,
        control_points: vec![],
        patch_inputs: vec![],
        partition_mode: Default::default(),
        output_winding: Default::default(),
        cull_mode: Default::default(),
    });
    case.case_id = case.computed_case_id().unwrap();
    assert_eq!(case.validate_literal_resources(), Ok(()));
    assert!(case.is_rasterization_disabled_vertex());

    case.render_targets.push(RenderTargetResource {
        index: 0,
        format: TextureFormat::Rgba32Float,
        dimensions: [1, 1],
        initial_bytes_b64: "q6urq6urq6urq6urq6urqw==".into(),
    });
    case.case_id = case.computed_case_id().unwrap();
    assert!(case
        .validate_literal_resources()
        .unwrap_err()
        .iter()
        .any(|error| error.contains("does not accept attachments")));
}

#[test]
fn draw_inputs_and_bool_constants_are_range_checked() {
    let mut case = example();
    case.stage = Stage::Vertex;
    case.dispatch = None;
    case.draw = Some(Draw {
        primitive: Primitive::Triangle,
        vertex_start: 1,
        vertex_count: 3,
        instance_count: 1,
        index_buffer: None,
        base_vertex: 0,
        base_instance: 0,
    });
    case.vertex_inputs.push(VertexInput {
        location: 0,
        format: AttributeFormat::Float,
        stride: 4,
        dynamic_stride: false,
        bytes_b64: "AAAAAAAAAAA=".into(),
    });
    case.function_constants.push(FunctionConstant {
        index: 0,
        scalar_type: ScalarType::Bool,
        lanes: 1,
        bytes_b64: "Ag==".into(),
    });
    let errors = case.validate_literal_resources().unwrap_err().join("\n");
    assert!(
        errors.contains("execution requires at least 16"),
        "{errors}"
    );
    assert!(
        errors.contains("bool must be encoded as 0 or 1"),
        "{errors}"
    );
    // A vertex input may declare a dynamic stride; a tessellation input may not: no executor models
    // its layout so.
    case.vertex_inputs[0].dynamic_stride = true;
    let errors = case.validate_literal_resources().unwrap_err().join("\n");
    assert!(!errors.contains("dynamic stride"), "{errors}");
    case.tessellation = Some(crate::case::TessellationDraw {
        factors: vec![],
        instance_count: 1,
        amplification_count: 1,
        control_points: vec![case.vertex_inputs[0].clone()],
        patch_inputs: vec![],
        partition_mode: Default::default(),
        output_winding: Default::default(),
        cull_mode: Default::default(),
    });
    let errors = case.validate_literal_resources().unwrap_err().join("\n");
    assert!(
        errors.contains("tessellation control point 0 declares a dynamic stride"),
        "{errors}"
    );
}

#[test]
fn kernel_stage_inputs_use_the_products_runtime_array_stride() {
    let mut case = example();
    case.case_id = "22".repeat(32);
    let original_id = case.computed_case_id().unwrap();
    case.kernel_stage_inputs.push(KernelStageInput {
        location: 6,
        format: AttributeFormat::Uint3,
        stride: 12,
        dynamic_stride: false,
        bytes_b64: "AAAAAAAAAAAAAAAAAAAAAA==".into(),
    });
    let errors = case.validate_literal_resources().unwrap_err().join("\n");
    assert!(
        errors.contains("must equal product storage-buffer stride 16"),
        "{errors}"
    );
    case.kernel_stage_inputs[0].stride = 16;
    assert!(case.validate_literal_resources().is_ok());
    assert_ne!(original_id, case.computed_case_id().unwrap());

    case.kernel_stage_inputs[0].format = AttributeFormat::Half3;
    case.kernel_stage_inputs[0].stride = 8;
    case.kernel_stage_inputs[0].bytes_b64 = "AAAAAAAAAAA=".into();
    assert!(case.validate_literal_resources().is_ok());
    // A dynamic stride is a kernel stage input's, binds at the same stride, and is part of the
    // case's identity; absent, it does not serialize.
    let fixed_id = case.computed_case_id().unwrap();
    let fixed = serde_json::to_value(&case.kernel_stage_inputs[0]).unwrap();
    assert!(fixed.get("dynamic_stride").is_none());
    case.kernel_stage_inputs[0].dynamic_stride = true;
    assert!(case.validate_literal_resources().is_ok());
    assert_ne!(fixed_id, case.computed_case_id().unwrap());
}

/// A dispatch's stage-in region is a kernel stage input case's: its stage inputs must hold the
/// elements the region lets a thread fetch (its x origin plus the threads it keeps apart: the
/// grid's, or its size where smaller, a size of 0 clamping nothing). It is part of the case's
/// identity and, absent, does not serialize.
#[test]
fn a_dispatch_stage_in_region_counts_the_elements_its_threads_fetch() {
    use crate::case::indirect_commands::HostStageInRegion;
    let mut case = example();
    case.case_id = "33".repeat(32);
    let dispatch = case.dispatch.as_mut().unwrap();
    dispatch.grid = [8, 1, 1];
    dispatch.threads_per_threadgroup = [8, 1, 1];
    assert!(serde_json::to_value(&*dispatch)
        .unwrap()
        .get("stage_in_region")
        .is_none());
    let region = |origin: [u32; 3], size: [u32; 3]| Some(HostStageInRegion { origin, size });
    case.dispatch.as_mut().unwrap().stage_in_region = region([0; 3], [6, 1, 1]);
    let errors = case.validate_literal_resources().unwrap_err().join("\n");
    assert!(
        errors.contains("stage_in_region is valid only with kernel stage inputs"),
        "{errors}"
    );
    // Seven 4-byte elements.
    case.kernel_stage_inputs.push(KernelStageInput {
        location: 0,
        format: AttributeFormat::Uint,
        stride: 4,
        dynamic_stride: false,
        bytes_b64: "EAAAABEAAAASAAAAEwAAABQAAAAVAAAAFgAAAA==".into(),
    });
    for (origin, size, elements, valid) in [
        ([0, 0, 0], [6, 1, 1], 6, true),
        ([1, 0, 0], [6, 1, 1], 7, true),
        ([2, 0, 0], [6, 1, 1], 8, false),
        ([0, 5, 0], [0, 1, 1], 8, false),
        ([0, 0, 0], [7, 1, 1], 7, true),
        ([0, 0, 0], [9, 1, 1], 8, false),
    ] {
        case.dispatch.as_mut().unwrap().stage_in_region = region(origin, size);
        assert_eq!(
            case.dispatch.as_ref().unwrap().stage_in_elements(),
            Some(elements)
        );
        let result = case.validate_literal_resources();
        assert_eq!(result.is_ok(), valid, "{origin:?} {size:?}: {result:?}");
        if !valid {
            assert!(
                result
                    .unwrap_err()
                    .join("\n")
                    .contains("but execution requires at least"),
                "{origin:?} {size:?}"
            );
        }
    }
    case.dispatch.as_mut().unwrap().stage_in_region = region([u32::MAX, 0, 0], [6, 1, 1]);
    let errors = case.validate_literal_resources().unwrap_err().join("\n");
    assert!(errors.contains("overflows 32 bits"), "{errors}");
    case.dispatch.as_mut().unwrap().stage_in_region = region([0; 3], [6, 1, 1]);
    let with = case.computed_case_id().unwrap();
    case.dispatch.as_mut().unwrap().stage_in_region = None;
    assert_ne!(with, case.computed_case_id().unwrap());
    assert!(
        case.validate_literal_resources().is_err(),
        "eight threads need eight elements"
    );
}

#[test]
fn imageblock_dimensions_are_a_uniform_multiple_of_the_local_size() {
    let mut case = example();
    case.dispatch.as_mut().unwrap().threads_per_threadgroup = [2, 2, 1];
    case.imageblock = Some(ImageblockResource {
        dimensions: [2, 2],
        implicit_coverage: None,
    });
    case.case_id = case.computed_case_id().unwrap();
    case.validate_literal_resources().unwrap();

    // A thread may own a whole kxk block of the tile rather than one cell of it.
    case.imageblock.as_mut().unwrap().dimensions = [4, 4];
    case.case_id = case.computed_case_id().unwrap();
    case.validate_literal_resources().unwrap();

    // Not the same factor in both axes: no k names this tile.
    for dimensions in [[4, 2], [3, 3], [2, 0]] {
        case.imageblock.as_mut().unwrap().dimensions = dimensions;
        case.case_id = case.computed_case_id().unwrap();
        let errors = case.validate_literal_resources().unwrap_err().join("\n");
        assert!(
            errors.contains("must be the same whole multiple of the threadgroup"),
            "{dimensions:?}: {errors}"
        );
    }
}

/// A `uint` fragment imageblock member is a literal of its own, a 4-byte `R32Uint` plane; a reader
/// built before it refuses such a case as an unknown variant.
#[test]
fn a_uint_imageblock_member_is_its_own_literal() {
    let member = serde_json::json!({
        "semantic": "lo",
        "format": "uint",
        "role": "in_out",
        "initial_bytes_b64": "AQAAAA==",
    });
    let parsed: FragmentImageblockMemberResource = serde_json::from_value(member.clone()).unwrap();
    assert_eq!(parsed.format, FragmentImageblockFormat::Uint);
    assert_eq!(serde_json::to_value(&parsed).unwrap(), member);
    assert_eq!(
        FragmentImageblockFormat::from_air_type("uint", 4),
        Some(FragmentImageblockFormat::Uint)
    );
    assert_eq!(FragmentImageblockFormat::from_air_type("uint", 8), None);
    assert_eq!(FragmentImageblockFormat::from_air_type("int", 4), None);
    assert_eq!(FragmentImageblockFormat::Uint.air_type_name(), "uint");
    assert_eq!(FragmentImageblockFormat::Uint.byte_size(), 4);
    assert_eq!(
        FragmentImageblockFormat::Uint.texture_format(),
        TextureFormat::R32Uint
    );
}

/// A `float` fragment imageblock member is a literal of its own, a 4-byte `R32Float` plane. The
/// format is a closed enum with no catch-all, so a reader whose schema does not name a format
/// refuses the case as an unknown variant: an executor or oracle built before `float` refuses such
/// a case rather than misreading the member as another format.
#[test]
fn a_float_imageblock_member_is_its_own_literal_and_an_unknown_format_is_refused() {
    let member = |format: &str| {
        serde_json::json!({
            "semantic": "sum",
            "format": format,
            "role": "input",
            "bytes_b64": "AADAPw==",
        })
    };
    let parsed: FragmentImageblockMemberResource = serde_json::from_value(member("float")).unwrap();
    assert_eq!(parsed.format, FragmentImageblockFormat::Float);
    assert_eq!(serde_json::to_value(&parsed).unwrap(), member("float"));
    assert_eq!(
        FragmentImageblockFormat::from_air_type("float", 4),
        Some(FragmentImageblockFormat::Float)
    );
    assert_eq!(FragmentImageblockFormat::from_air_type("float", 8), None);
    assert_eq!(FragmentImageblockFormat::Float.air_type_name(), "float");
    assert_eq!(FragmentImageblockFormat::Float.byte_size(), 4);
    assert_eq!(
        FragmentImageblockFormat::Float.texture_format(),
        TextureFormat::R32Float
    );
    let unknown =
        serde_json::from_value::<FragmentImageblockMemberResource>(member("float2")).unwrap_err();
    assert!(
        unknown.to_string().contains("unknown variant `float2`"),
        "{unknown}"
    );
}

#[test]
fn varying_values_are_a_fragment_cases_and_part_of_its_identity() {
    let value = |lanes: &[u32]| VaryingValue {
        key: "generated(1af)".into(),
        lanes: lanes.to_vec(),
    };
    let plain = example();
    assert!(!serde_json::to_string(&plain)
        .unwrap()
        .contains("varying_values"));
    let mut authored = example();
    authored.varying_values = vec![value(&[0x4000_0000])];
    let id = authored.computed_case_id().unwrap();
    assert_ne!(id, plain.computed_case_id().unwrap());
    let mut other = authored.clone();
    other.varying_values = vec![value(&[0x3F80_0000])];
    assert_ne!(other.computed_case_id().unwrap(), id);

    // The kernel example has no varyings to give values to; a key twice and a value of no lanes
    // are refused too. The checker matches keys and lane types against the fragment's inputs.
    authored.case_id = id;
    authored.varying_values.push(value(&[]));
    let errors = authored
        .validate_literal_resources()
        .unwrap_err()
        .join("\n");
    for refusal in [
        "varying_values are valid only for fragment cases",
        "duplicate varying value key \"generated(1af)\"",
        "has 0 lanes; a varying has 1 to 4",
    ] {
        assert!(errors.contains(refusal), "{refusal}: {errors}");
    }
}
