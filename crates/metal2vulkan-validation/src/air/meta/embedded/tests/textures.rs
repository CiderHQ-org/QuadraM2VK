use super::*;

#[test]
fn embedded_texture_use_gate_includes_sampling_and_gathering() {
    assert!(body_uses_texture_intrinsic(
        "call <4 x float> @air.sample_texture_2d(...)"
    ));
    assert!(body_uses_texture_intrinsic(
        "call <4 x float> @air.gather_texture_2d(...)"
    ));
    assert!(!body_uses_texture_intrinsic(
        "declare void @unrelated_texture_helper()"
    ));
}

#[test]
fn embedded_texture_detection_includes_read_and_write_fields() {
    let mut nodes = HashMap::new();
    nodes.insert(
            10,
            r#"i32 0, i32 8, i32 0, !"texture2d<float, read>", !"input", !"air.indirect_argument", !20, i32 8, i32 8, i32 0, !"texture2d<float, write>", !"output", !"air.indirect_argument", !21"#.to_string(),
        );
    nodes.insert(
            20,
            r#"i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>""#.to_string(),
        );
    nodes.insert(
            21,
            r#"i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>""#.to_string(),
        );

    let textures = detect_embedded_textures(&nodes, &ArgumentBuffers::new(vec![(4, 0, 10)]), &[]);
    assert_eq!(textures.len(), 2);
    assert_eq!(textures[0].buffer_param_index, 4);
    assert_eq!(textures[0].field_offset, 0);
    assert_eq!(textures[0].argument_index, 0);
    assert_eq!(textures[0].storage_format, None);
    assert!(!textures[0].arrayed);
    assert_eq!(textures[0].synthetic_texture_index, 0);
    assert_eq!(textures[1].field_offset, 8);
    assert_eq!(textures[1].argument_index, 1);
    assert_eq!(textures[1].storage_format, Some(TextureFormat::R32f));
    assert_eq!(textures[1].array_length, None);
    assert!(!textures[1].arrayed);
    assert_eq!(textures[1].synthetic_texture_index, 1);
}

#[test]
fn embedded_fixed_depth_array_preserves_handle_count_and_image_shape() {
    let mut nodes = HashMap::new();
    nodes.insert(
            10,
            r#"i32 0, i32 16, i32 0, !"array<depth2d_array<float, sample>, 2>", !"depth", !"air.indirect_argument", !20"#.to_string(),
        );
    nodes.insert(
            20,
            r#"i32 0, !"air.texture", !"air.location_index", i32 4, i32 2, !"air.sample", !"air.arg_type_name", !"array<depth2d_array<float, sample>, 2>""#.to_string(),
        );

    let textures = detect_embedded_textures(&nodes, &ArgumentBuffers::new(vec![(1, 16, 10)]), &[]);
    assert_eq!(textures.len(), 1);
    assert_eq!(textures[0].argument_index, 4);
    assert_eq!(textures[0].array_length, Some(2));
    assert!(textures[0].arrayed);
    assert_eq!(textures[0].storage_format, None);
}

#[test]
fn embedded_c_array_member_takes_its_length_from_the_member_tuple() {
    // `texture2d<float, sample> slices[32]` inside an argument buffer: the type name is the
    // ELEMENT type and the node's count operand stays 1, so only the tuple's `array_len` slot
    // says 32. Reading the name alone surfaced one descriptor and let a runtime index sample
    // element 0.
    let mut nodes = HashMap::new();
    nodes.insert(
            10,
            r#"i32 0, i32 8, i32 32, !"texture2d<float, sample>", !"slices", !"air.indirect_argument", !20"#.to_string(),
        );
    nodes.insert(
            20,
            r#"i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"slices""#.to_string(),
        );

    let textures = detect_embedded_textures(&nodes, &ArgumentBuffers::new(vec![(5, 1, 10)]), &[]);
    assert_eq!(textures.len(), 1);
    assert_eq!(textures[0].array_length, Some(32));
    assert_eq!(textures[0].argument_index, 0);
    assert!(!textures[0].arrayed);
    // Expressible, so it is not also reported as a resource with no descriptor.
    assert!(
        unsurfaced_embedded_resources(&nodes, &ArgumentBuffers::new(vec![(5, 1, 10)])).is_empty()
    );
}

/// The synthetic embedded-texture index is a function of the entry's SIGNATURE, not of the order
/// the AIR metadata list names its argument-buffer nodes.
///
/// The walk hands out `K`, `K+1`, ... in list order, and a consumer binds by that index, so
/// listing the same two argument buffers the other way round used to move each embedded texture
/// onto the other one's descriptor -- in one corpus kernel, from `Binding 32` (sampled) to
/// `Binding 480` (storage). Parameter index orders them because it is unique per argument;
/// the Metal `[[buffer(N)]]` slot is not, since mutually exclusive function-constant
/// alternatives share one.
#[test]
fn the_synthetic_texture_index_does_not_depend_on_the_argument_list_order() {
    let nodes = HashMap::from([
            (
                10_u32,
                r#"i32 0, i32 8, i32 0, !"texture2d<float, sample>", !"first", !"air.indirect_argument", !12"#
                    .to_string(),
            ),
            (
                11_u32,
                r#"i32 0, i32 8, i32 0, !"texture2d<float, write>", !"second", !"air.indirect_argument", !13"#
                    .to_string(),
            ),
            (
                12_u32,
                r#"i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"first""#
                    .to_string(),
            ),
            (
                13_u32,
                r#"i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"second""#
                    .to_string(),
            ),
        ]);
    // Parameter 0 binds `[[buffer(7)]]`, parameter 1 binds `[[buffer(2)]]`, so list order,
    // parameter order and Metal-slot order are three different orders.
    let declared = vec![(0_u32, 7_u32, 10_u32), (1, 2, 11)];
    let forward = detect_embedded_textures(&nodes, &ArgumentBuffers::new(declared.clone()), &[]);
    let reversed = detect_embedded_textures(
        &nodes,
        &ArgumentBuffers::new(declared.into_iter().rev().collect()),
        &[],
    );
    assert_eq!(forward, reversed);
    assert_eq!(
        forward
            .iter()
            .map(|texture| (texture.buffer_param_index, texture.synthetic_texture_index))
            .collect::<Vec<_>>(),
        vec![(0, 0), (1, 1)],
    );
}
