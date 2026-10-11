use super::*;

#[test]
fn rg32_float_has_an_exact_literal() {
    assert_eq!(TextureFormat::Rg32Float.bytes_per_pixel(), 8);
}

/// `rg32_uint` is the format of Metal's `ulong` textures. Its texel is its two 32-bit words; it is
/// no executor's gap; and a case binds it to a single-sample texture alone, never a multisampled
/// texture or a render target.
#[test]
fn rg32_uint_is_the_ulong_texture_literal() {
    use base64::Engine as _;
    let format = TextureFormat::Rg32Uint;
    assert_eq!(format.bytes_per_pixel(), 8);
    assert!(format.is_ulong() && !format.sampled_only() && !format.is_depth());
    assert_eq!(serde_json::to_value(format).unwrap(), "rg32_uint");
    let mut case = example();
    case.textures.push(TextureResource {
        binding: 0,
        role: ResourceRole::InOut,
        texture_type: TextureType::D2,
        format,
        dimensions: [2, 2, 1],
        sample_count: 1,
        bytes_b64: None,
        initial_bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([0u8; 32])),
        sparse: None,
        mip_levels: 1,
    });
    case.case_id = case.computed_case_id().unwrap();
    case.validate_literal_resources().unwrap();
    assert!(crate::executor_contract::case_requirements(&case).is_empty());

    let mut multisampled = case.clone();
    multisampled.textures[0].role = ResourceRole::Input;
    multisampled.textures[0].texture_type = TextureType::D2Multisample;
    multisampled.textures[0].sample_count = 2;
    multisampled.textures[0].initial_bytes_b64 = None;
    multisampled.textures[0].bytes_b64 =
        Some(base64::engine::general_purpose::STANDARD.encode([0u8; 64]));
    multisampled.case_id = multisampled.computed_case_id().unwrap();
    let errors = multisampled
        .validate_literal_resources()
        .unwrap_err()
        .join("\n");
    assert!(
        errors.contains("Rg32Uint is the ulong texture format, bound only to a single-sample"),
        "{errors}"
    );

    let mut target = example();
    target.stage = Stage::Fragment;
    target.dispatch = None;
    target.render_targets.push(RenderTargetResource {
        index: 0,
        format,
        dimensions: [1, 1],
        initial_bytes_b64: base64::engine::general_purpose::STANDARD.encode([0u8; 8]),
    });
    target.case_id = target.computed_case_id().unwrap();
    let errors = target.validate_literal_resources().unwrap_err().join("\n");
    assert!(
        errors.contains("render target 0 format Rg32Uint is the ulong texture format"),
        "{errors}"
    );
}

#[test]
fn argument_buffer_textures_use_structural_identity_and_literal_output() {
    let mut case = example();
    case.argument_buffer_textures
        .push(ArgumentBufferTextureResource {
            buffer_binding: 0,
            field_offset: 16,
            role: ResourceRole::Output,
            texture_type: TextureType::D2,
            format: TextureFormat::R32Uint,
            dimensions: [1, 1, 1],
            sample_count: 1,
            bytes_b64: None,
            initial_bytes_b64: Some("q6urqw==".into()),
        });
    case.output = OutputSelection::ArgumentBufferTexture {
        buffer_binding: 0,
        field_offset: 16,
        origin: [0, 0, 0],
        dimensions: [1, 1, 1],
    };
    case.case_id = case.computed_case_id().unwrap();
    case.validate_literal_resources().unwrap();
    let resources = crate::literal::LiteralResources::prepare(&case).unwrap();
    assert_eq!(resources.argument_buffer_textures.len(), 1);
    assert_eq!(
        resources.argument_buffer_textures[0]
            .select([0, 0, 0], [1, 1, 1])
            .unwrap(),
        [0xab; 4]
    );
}

#[test]
fn texture_array_elements_are_ordered_literals_and_reserve_metal_slots() {
    let mut case = example();
    case.texture_arrays.push(TextureArrayResource {
        binding: 4,
        role: ResourceRole::Input,
        texture_type: TextureType::D2,
        format: TextureFormat::R32Uint,
        sample_count: 1,
        overrides_texture_at_base: false,
        elements: vec![
            TextureArrayElement {
                dimensions: [1, 1, 1],
                bytes_b64: Some("AQAAAA==".into()),
                initial_bytes_b64: None,
            },
            TextureArrayElement {
                dimensions: [1, 1, 1],
                bytes_b64: Some("AgAAAA==".into()),
                initial_bytes_b64: None,
            },
        ],
    });
    case.case_id = case.computed_case_id().unwrap();
    case.validate_literal_resources().unwrap();
    let resources = crate::literal::LiteralResources::prepare(&case).unwrap();
    assert_eq!(resources.texture_arrays[0].elements[0].bytes, [1, 0, 0, 0]);
    assert_eq!(resources.texture_arrays[0].elements[1].bytes, [2, 0, 0, 0]);

    case.textures.push(TextureResource {
        binding: 5,
        role: ResourceRole::Input,
        texture_type: TextureType::D2,
        format: TextureFormat::R32Uint,
        dimensions: [1, 1, 1],
        sample_count: 1,
        bytes_b64: Some("AwAAAA==".into()),
        initial_bytes_b64: None,
        sparse: None,
        mip_levels: 1,
    });
    let errors = case.validate_literal_resources().unwrap_err().join("\n");
    assert!(errors.contains("overlaps Metal texture slot 5"), "{errors}");
}

#[test]
fn function_constant_layered_and_fixed_texture_alternatives_may_share_the_base_slot() {
    let mut case = example();
    case.function_constants.push(FunctionConstant {
        index: 0,
        scalar_type: ScalarType::U32,
        lanes: 1,
        bytes_b64: "AAAAAA==".into(),
    });
    case.textures.push(TextureResource {
        binding: 0,
        role: ResourceRole::Input,
        texture_type: TextureType::D2Array,
        format: TextureFormat::R32Uint,
        dimensions: [1, 1, 1],
        sample_count: 1,
        bytes_b64: Some("BwAAAA==".into()),
        initial_bytes_b64: None,
        sparse: None,
        mip_levels: 1,
    });
    case.texture_arrays.push(TextureArrayResource {
        binding: 0,
        role: ResourceRole::Input,
        texture_type: TextureType::D2,
        format: TextureFormat::R32Uint,
        sample_count: 1,
        overrides_texture_at_base: true,
        elements: vec![TextureArrayElement {
            dimensions: [1, 1, 1],
            bytes_b64: Some("BwAAAA==".into()),
            initial_bytes_b64: None,
        }],
    });

    case.case_id = case.computed_case_id().unwrap();
    case.validate_literal_resources().unwrap();
    let resources = crate::literal::LiteralResources::prepare(&case).unwrap();
    assert!(resources.texture_arrays[0].overrides_texture_at_base);

    case.function_constants.clear();
    case.case_id = case.computed_case_id().unwrap();
    let errors = case.validate_literal_resources().unwrap_err().join("\n");
    assert!(errors.contains("overlaps Metal texture slot 0"), "{errors}");

    case.function_constants.push(FunctionConstant {
        index: 0,
        scalar_type: ScalarType::U32,
        lanes: 1,
        bytes_b64: "AAAAAA==".into(),
    });
    case.textures.clear();
    case.case_id = case.computed_case_id().unwrap();
    let errors = case.validate_literal_resources().unwrap_err().join("\n");
    assert!(
        errors.contains(
            "overrides_texture_at_base requires a matching input-only layered texture alternative"
        ),
        "{errors}"
    );
}

#[test]
fn multisample_array_literals_keep_layers_and_samples_independent() {
    use base64::Engine as _;

    let mut case = example();
    case.textures.push(TextureResource {
        binding: 3,
        role: ResourceRole::Input,
        texture_type: TextureType::D2MultisampleArray,
        format: TextureFormat::R32Uint,
        dimensions: [2, 1, 3],
        sample_count: 4,
        bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([0u8; 96])),
        initial_bytes_b64: None,
        sparse: None,
        mip_levels: 1,
    });
    case.case_id = case.computed_case_id().unwrap();
    case.validate_literal_resources().unwrap();
    let resources = crate::literal::LiteralResources::prepare(&case).unwrap();
    let layout = resources.textures[0].layout().unwrap();
    assert_eq!(layout.array_layers, 3);
    assert_eq!(layout.sample_count, 4);

    case.textures[0].role = ResourceRole::Output;
    let errors = case.validate_literal_resources().unwrap_err().join("\n");
    assert!(errors.contains("multisample texture binding 3 must have input role"));
}

/// A sampled-only format binds as a single-sample input texture, and is refused as an output, a
/// multisampled texture, a storage specialization, or a render target.
#[test]
fn sampled_only_formats_bind_only_as_single_sample_inputs() {
    use base64::Engine as _;

    let mut case = example();
    case.textures.push(TextureResource {
        binding: 3,
        role: ResourceRole::Input,
        texture_type: TextureType::D2,
        format: TextureFormat::Rgba16Snorm,
        dimensions: [2, 2, 1],
        sample_count: 1,
        bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([0u8; 32])),
        initial_bytes_b64: None,
        sparse: None,
        mip_levels: 1,
    });
    case.case_id = case.computed_case_id().unwrap();
    case.validate_literal_resources().unwrap();
    assert_eq!(TextureFormat::Rgba16Snorm.bytes_per_pixel(), 8);

    case.textures[0].role = ResourceRole::Output;
    case.textures[0].bytes_b64 = None;
    case.textures[0].initial_bytes_b64 =
        Some(base64::engine::general_purpose::STANDARD.encode([0u8; 32]));
    case.case_id = case.computed_case_id().unwrap();
    let errors = case.validate_literal_resources().unwrap_err().join("\n");
    assert!(
        errors.contains("Rgba16Snorm is a sampled-only format"),
        "{errors}"
    );

    case.textures[0].role = ResourceRole::Input;
    case.textures[0].initial_bytes_b64 = None;
    case.textures[0].texture_type = TextureType::D2Multisample;
    case.textures[0].sample_count = 4;
    case.textures[0].bytes_b64 = Some(base64::engine::general_purpose::STANDARD.encode([0u8; 128]));
    case.case_id = case.computed_case_id().unwrap();
    let errors = case.validate_literal_resources().unwrap_err().join("\n");
    assert!(
        errors.contains("Rgba16Snorm is a sampled-only format"),
        "{errors}"
    );
}

/// The formats the probe reads are sampled-only, spelled in snake case, read as `float` textures,
/// and take Metal's bytes per pixel: the packed 16-bit formats two, the packed 32-bit ones four.
#[test]
fn the_probed_formats_are_sampled_only_with_metals_sizes() {
    use base64::Engine as _;

    let formats = [
        (TextureFormat::Bgra8Unorm, "bgra8_unorm", 4),
        (TextureFormat::Bgra8UnormSrgb, "bgra8_unorm_srgb", 4),
        (TextureFormat::R8UnormSrgb, "r8_unorm_srgb", 1),
        (TextureFormat::Rg8Unorm, "rg8_unorm", 2),
        (TextureFormat::Rg8UnormSrgb, "rg8_unorm_srgb", 2),
        (TextureFormat::Rg8Snorm, "rg8_snorm", 2),
        (TextureFormat::Rg16Unorm, "rg16_unorm", 4),
        (TextureFormat::A8Unorm, "a8_unorm", 1),
        (TextureFormat::B5g6r5Unorm, "b5g6r5_unorm", 2),
        (TextureFormat::A1bgr5Unorm, "a1bgr5_unorm", 2),
        (TextureFormat::Abgr4Unorm, "abgr4_unorm", 2),
        (TextureFormat::Bgr5a1Unorm, "bgr5a1_unorm", 2),
        (TextureFormat::Rgb10a2Unorm, "rgb10a2_unorm", 4),
        (TextureFormat::Bgr10a2Unorm, "bgr10a2_unorm", 4),
        (TextureFormat::Rg11b10Float, "rg11b10_float", 4),
        (TextureFormat::Rgb9e5Float, "rgb9e5_float", 4),
    ];
    for (format, name, bytes) in formats {
        assert!(format.sampled_only(), "{name}");
        assert_eq!(format.bytes_per_pixel(), bytes, "{name}");
        assert_eq!(
            format.component_class(),
            crate::air::reflect::TextureComponent::Float,
            "{name}"
        );
        assert_eq!(serde_json::to_value(format).unwrap(), name);
        let mut case = example();
        case.textures.push(TextureResource {
            binding: 3,
            role: ResourceRole::Input,
            texture_type: TextureType::D2,
            format,
            dimensions: [2, 2, 1],
            sample_count: 1,
            bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode(vec![0u8; 4 * bytes])),
            initial_bytes_b64: None,
            sparse: None,
            mip_levels: 1,
        });
        case.case_id = case.computed_case_id().unwrap();
        case.validate_literal_resources()
            .unwrap_or_else(|errors| panic!("{name}: {}", errors.join("; ")));
        case.textures[0].role = ResourceRole::Output;
        case.textures[0].initial_bytes_b64 = case.textures[0].bytes_b64.take();
        case.case_id = case.computed_case_id().unwrap();
        let errors = case.validate_literal_resources().unwrap_err().join("\n");
        assert!(
            errors.contains("is a sampled-only format"),
            "{name}: {errors}"
        );
    }
}

/// An argument-buffer sampler is a literal of its owner's field. It round-trips with its comparison
/// function; a case without one serializes as before (no key), so every existing identity stands;
/// one joins both identities; and it shares its owner's fields with every other member and needs
/// its owner bound.
#[test]
fn argument_buffer_samplers_are_field_literals_in_the_identity() {
    let case = example();
    let json = serde_json::to_value(&case).unwrap();
    assert!(json.get("argument_buffer_samplers").is_none());
    let mut authored = case.clone();
    authored
        .argument_buffer_samplers
        .push(ArgumentBufferSamplerResource {
            buffer_binding: 0,
            field_offset: 8,
            address_mode: SamplerAddressMode::ClampToEdge,
            min_filter: SamplerFilter::Linear,
            mag_filter: SamplerFilter::Linear,
            mip_filter: SamplerMipFilter::NotMipmapped,
            normalized_coordinates: true,
            compare_function: Some(SamplerCompareFunction::Less),
        });
    let json = serde_json::to_value(&authored).unwrap();
    assert_eq!(
        json["argument_buffer_samplers"][0]["compare_function"],
        "less"
    );
    let back: AuthoredCase = serde_json::from_value(json).unwrap();
    assert_eq!(back, authored);
    assert_ne!(
        authored.computed_case_id().unwrap(),
        case.computed_case_id().unwrap()
    );
    assert_ne!(
        authored.computed_input_sha256().unwrap(),
        case.computed_input_sha256().unwrap()
    );
    // Absent is Metal's default, `never`, and is not written.
    let mut never = authored.clone();
    never.argument_buffer_samplers[0].compare_function = None;
    let json = serde_json::to_value(&never).unwrap();
    assert!(json["argument_buffer_samplers"][0]
        .get("compare_function")
        .is_none());
    authored.case_id = authored.computed_case_id().unwrap();
    authored.validate_literal_resources().unwrap();

    let errors = |case: &AuthoredCase| case.validate_literal_resources().unwrap_err();
    let mut twice = authored.clone();
    twice
        .argument_buffer_samplers
        .push(twice.argument_buffer_samplers[0].clone());
    twice.case_id = twice.computed_case_id().unwrap();
    assert!(
        errors(&twice)
            .iter()
            .any(|error| error == "duplicate argument-buffer sampler at buffer 0 offset 8"),
        "{:?}",
        errors(&twice)
    );
    let mut ownerless = authored.clone();
    ownerless.argument_buffer_samplers[0].buffer_binding = 5;
    ownerless.case_id = ownerless.computed_case_id().unwrap();
    assert!(
        errors(&ownerless)
            .iter()
            .any(|error| error == "argument-buffer sampler at offset 8 owns undeclared buffer 5"),
        "{:?}",
        errors(&ownerless)
    );
}
