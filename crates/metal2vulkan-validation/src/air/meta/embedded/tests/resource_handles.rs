use super::*;

#[test]
fn a_c_array_of_buffer_handles_surfaces_one_resource_per_element() {
    // `device half4* bufs[2] [[id(12)]]` -- two device addresses, at consecutive argument ids
    // and consecutive 8-byte slots. Surfacing only the first left the second slot of the
    // argument buffer unwritten, so a runtime index into the member read an address nothing
    // ever populated.
    let mut nodes = HashMap::new();
    nodes.insert(
            10,
            r#"i32 0, i32 8, i32 2, !"half4", !"bufs", !"air.indirect_argument", !20, i32 16, i32 4, i32 4, !"uint", !"counts", !"air.indirect_argument", !21"#.to_string(),
        );
    nodes.insert(
            20,
            r#"i32 0, !"air.buffer", !"air.location_index", i32 12, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_name", !"half4", !"air.arg_name", !"bufs""#.to_string(),
        );
    // An inline `uint counts[4]` is one span of DATA, not four resources.
    nodes.insert(
            21,
            r#"i32 1, !"air.indirect_constant", !"air.location_index", i32 14, i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"counts""#.to_string(),
        );

    let arguments = detect_embedded_arguments(&nodes, &ArgumentBuffers::new(vec![(0, 3, 10)]));
    let coordinates = arguments
        .iter()
        .map(|argument| {
            (
                argument.field_offset,
                argument.argument_index,
                argument.resource_kind,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        coordinates,
        vec![
            (0, 12, Some(ArgumentBufferResourceKind::Buffer)),
            (8, 13, Some(ArgumentBufferResourceKind::Buffer)),
            (16, 14, None)
        ]
    );
}

#[test]
fn embedded_argument_detection_classifies_only_nested_buffers_as_device_resources() {
    let mut nodes = HashMap::new();
    nodes.insert(
            10,
            r#"i32 0, i32 4, i32 0, !"uint", !"count", !"air.indirect_argument", !20, i32 8, i32 8, i32 0, !"float", !"values", !"air.indirect_argument", !21"#.to_string(),
        );
    nodes.insert(
        20,
        r#"i32 0, !"air.indirect_constant", !"air.location_index", i32 2, i32 1"#.to_string(),
    );
    nodes.insert(
        21,
        r#"i32 1, !"air.buffer", !"air.location_index", i32 7, i32 1, !"air.address_space", i32 1"#
            .to_string(),
    );

    let arguments = detect_embedded_arguments(&nodes, &ArgumentBuffers::new(vec![(3, 5, 10)]));
    assert_eq!(arguments.len(), 2);
    assert_eq!(arguments[0].resource_kind, None);
    assert_eq!(arguments[1].buffer_param_index, 3);
    assert_eq!(arguments[1].buffer_index, 5);
    assert_eq!(arguments[1].field_offset, 8);
    assert_eq!(arguments[1].argument_index, 7);
    assert_eq!(
        arguments[1].resource_kind,
        Some(ArgumentBufferResourceKind::Buffer)
    );
    assert_eq!(arguments[1].resource_address_space, Some(1));
}
