use super::*;

#[test]
fn device_buffer_array_elements_are_sparse_bounded_literal_resources() {
    let mut case = example();
    case.case_id = "00".repeat(32);
    case.buffers.clear();
    case.device_buffer_arrays.push(DeviceBufferArrayResource {
        binding: 0,
        length: 3,
        elements: vec![
            DeviceBufferArrayElementResource {
                index: 0,
                role: ResourceRole::Input,
                bytes_b64: Some("AQAAAA==".into()),
                initial_bytes_b64: None,
            },
            DeviceBufferArrayElementResource {
                index: 2,
                role: ResourceRole::Output,
                bytes_b64: None,
                initial_bytes_b64: Some("q6urqw==".into()),
            },
        ],
    });
    case.output = OutputSelection::DeviceBufferArrayElement {
        binding: 0,
        element: 2,
        offset: 0,
        length: 4,
    };
    assert_eq!(case.validate_literal_resources(), Ok(()));
    case.threadgroup_memory.push(ThreadgroupMemoryResource {
        binding: 0,
        length: 8,
    });
    assert_eq!(case.validate_literal_resources(), Ok(()));

    case.device_buffer_arrays[0].elements[1].index = 3;
    assert!(case
        .validate_literal_resources()
        .unwrap_err()
        .iter()
        .any(|error| error.contains("exceeds length 3")));

    case.device_buffer_arrays[0].elements[1].index = 2;
    case.buffers.push(BufferResource {
        binding: 1,
        role: ResourceRole::Input,
        bytes_b64: Some("AQAAAA==".into()),
        initial_bytes_b64: None,
        rasterization_rate_map: None,
    });
    assert!(case
        .validate_literal_resources()
        .unwrap_err()
        .iter()
        .any(|error| error.contains("overlaps a buffer binding")));

    case.buffers.clear();
    case.device_buffer_arrays.push(DeviceBufferArrayResource {
        binding: 2,
        length: 1,
        elements: vec![DeviceBufferArrayElementResource {
            index: 0,
            role: ResourceRole::Input,
            bytes_b64: Some("AQAAAA==".into()),
            initial_bytes_b64: None,
        }],
    });
    assert!(case
        .validate_literal_resources()
        .unwrap_err()
        .iter()
        .any(|error| error.contains("overlaps another device-buffer array")));
}

#[test]
fn argument_buffer_buffers_are_owned_literals_and_selectable_outputs() {
    let mut case = example();
    case.argument_buffer_buffers
        .push(ArgumentBufferBufferResource {
            buffer_binding: 0,
            field_offset: 0,
            role: ResourceRole::Output,
            bytes_b64: None,
            initial_bytes_b64: Some("q6urqw==".into()),
            rasterization_rate_map: None,
        });
    case.output = OutputSelection::ArgumentBufferBuffer {
        buffer_binding: 0,
        field_offset: 0,
        offset: 0,
        length: 4,
    };
    case.case_id = case.computed_case_id().unwrap();
    case.validate_literal_resources().unwrap();
    let resources = crate::literal::LiteralResources::prepare(&case).unwrap();
    assert_eq!(resources.argument_buffer_buffers.len(), 1);
    assert_eq!(resources.argument_buffer_buffers[0].bytes, [0xab; 4]);
}

/// A rasterization-rate map is an input buffer's only contents, its descriptor is in Metal's
/// documented range, and it serializes only when present, so a case without one keeps its
/// identity. Its Vulkan bytes are the product's parameter data of the map.
#[test]
fn a_rate_map_is_an_input_buffers_only_contents() {
    let map = RateMapLiteral {
        screen_size: [37, 23],
        layers: vec![RateMapLayerLiteral {
            horizontal: vec![RateSample(0.5), RateSample(1.0)],
            vertical: vec![RateSample(1.0), RateSample(0.5)],
        }],
    };
    let mut case = example();
    let plain = serde_json::to_string(&case.buffers[0]).unwrap();
    assert!(!plain.contains("rasterization_rate_map"), "{plain}");
    case.buffers.push(BufferResource {
        binding: 1,
        role: ResourceRole::Input,
        bytes_b64: None,
        initial_bytes_b64: None,
        rasterization_rate_map: Some(map.clone()),
    });
    case.case_id = case.computed_case_id().unwrap();
    assert_eq!(case.validate_literal_resources(), Ok(()));
    let text = serde_json::to_string(&case.buffers[1]).unwrap();
    assert_eq!(
        serde_json::from_str::<BufferResource>(&text).unwrap(),
        case.buffers[1]
    );

    let literal = crate::literal::LiteralResources::prepare(&case).unwrap();
    let buffer = &literal.buffers[1];
    assert_eq!(buffer.rasterization_rate_map.as_ref(), Some(&map));
    let product = metal2vulkan::bind::RasterizationRateMap::new(&map.descriptor()).unwrap();
    assert_eq!(buffer.bytes, product.to_bytes());
    assert_eq!(product.physical_size(0), Some([36, 23]));

    let refused = |edit: &dyn Fn(&mut BufferResource)| {
        let mut bad = case.clone();
        edit(&mut bad.buffers[1]);
        bad.case_id = bad.computed_case_id().unwrap();
        bad.validate_literal_resources().unwrap_err()
    };
    let only = "rasterization_rate_map is an input's only contents";
    assert!(refused(&|b| b.bytes_b64 = Some("AAAAAA==".into()))[0].contains(only));
    assert!(refused(&|b| b.role = ResourceRole::InOut)
        .iter()
        .any(|error| error.contains(only)));
    let map_of = |b: &mut BufferResource| b.rasterization_rate_map.as_mut().unwrap().clone();
    let with = |edit: &dyn Fn(&mut RateMapLiteral)| {
        refused(&|b| {
            let mut map = map_of(b);
            edit(&mut map);
            b.rasterization_rate_map = Some(map);
        })
    };
    assert!(with(&|m| m.screen_size[0] = 0)[0].contains("screen extent 0"));
    assert!(with(&|m| m.screen_size[1] = 65_537)[0].contains("screen extent 65537"));
    assert!(with(&|m| m.layers.clear())[0].contains("has no layer"));
    assert!(with(&|m| m.layers[0].vertical.clear())[0].contains("axis 1 has no sample"));
    assert!(with(&|m| m.layers[0].horizontal[0] = RateSample(1.5))[0].contains("outside [0, 1]"));
    assert!(
        with(&|m| m.layers[0].horizontal[0] = RateSample(f32::NAN))[0].contains("outside [0, 1]")
    );
}

/// An argument-buffer member may hold a rasterization-rate map as a buffer does: it is the member's
/// only contents, serializes only when present, and its Vulkan bytes are the product's parameter
/// data of the map.
#[test]
fn a_rate_map_is_an_input_argument_buffer_members_only_contents() {
    let map = RateMapLiteral {
        screen_size: [37, 23],
        layers: vec![RateMapLayerLiteral {
            horizontal: vec![RateSample(0.5), RateSample(1.0)],
            vertical: vec![RateSample(1.0), RateSample(0.5)],
        }],
    };
    let mut case = example();
    let member = |map: Option<RateMapLiteral>| ArgumentBufferBufferResource {
        buffer_binding: 0,
        field_offset: 8,
        role: ResourceRole::Input,
        bytes_b64: map.is_none().then(|| "AAAAAA==".into()),
        initial_bytes_b64: None,
        rasterization_rate_map: map,
    };
    let plain = serde_json::to_string(&member(None)).unwrap();
    assert!(!plain.contains("rasterization_rate_map"), "{plain}");
    case.argument_buffer_buffers.push(member(Some(map.clone())));
    case.case_id = case.computed_case_id().unwrap();
    assert_eq!(case.validate_literal_resources(), Ok(()));
    let text = serde_json::to_string(&case.argument_buffer_buffers[0]).unwrap();
    assert_eq!(
        serde_json::from_str::<ArgumentBufferBufferResource>(&text).unwrap(),
        case.argument_buffer_buffers[0]
    );

    let literal = crate::literal::LiteralResources::prepare(&case).unwrap();
    let buffer = &literal.argument_buffer_buffers[0];
    assert_eq!(buffer.rasterization_rate_map.as_ref(), Some(&map));
    let product = metal2vulkan::bind::RasterizationRateMap::new(&map.descriptor()).unwrap();
    assert_eq!(buffer.bytes, product.to_bytes());

    let refused = |edit: &dyn Fn(&mut ArgumentBufferBufferResource)| {
        let mut bad = case.clone();
        edit(&mut bad.argument_buffer_buffers[0]);
        bad.case_id = bad.computed_case_id().unwrap();
        bad.validate_literal_resources().unwrap_err()
    };
    let only = "argument-buffer buffer at buffer 0 offset 8 rasterization_rate_map is an input's \
                only contents";
    assert!(refused(&|b| b.bytes_b64 = Some("AAAAAA==".into()))[0].contains(only));
    assert!(refused(&|b| b.role = ResourceRole::InOut)
        .iter()
        .any(|error| error.contains(only)));
    assert!(refused(&|b| {
        let mut map = b.rasterization_rate_map.clone().unwrap();
        map.layers.clear();
        b.rasterization_rate_map = Some(map);
    })[0]
        .contains("has no layer"));
}
