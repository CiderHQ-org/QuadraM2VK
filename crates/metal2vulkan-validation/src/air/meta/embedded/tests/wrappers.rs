use super::*;

/// A wrapper's members carry an `air.location_index` RELATIVE to the wrapper's own id, so the
/// Metal `[[id(n)]]` adds down the nesting path -- 30 + 0 + 2 for a handle two wrappers deep.
#[test]
fn a_wrapped_texture_binds_at_the_summed_argument_id() {
    let nodes = HashMap::from([
            (
                10_u32,
                r#"!"air.struct_type_info", !11, i32 64, i32 8, i32 0, !"deep", !"d", !"air.indirect_argument", i32 30, i32 72, i32 8, i32 0, !"texture2d<float, sample>", !"flat", !"air.indirect_argument", !14"#.to_string(),
            ),
            (
                11_u32,
                r#"!"air.struct_type_info", !12, i32 0, i32 8, i32 0, !"inner", !"mid", !"air.indirect_argument", i32 0"#.to_string(),
            ),
            (
                12_u32,
                r#"i32 8, i32 8, i32 0, !"texture2d<float, sample>", !"y", !"air.indirect_argument", !13"#.to_string(),
            ),
            (
                13_u32,
                r#"i32 0, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>""#.to_string(),
            ),
            (
                14_u32,
                r#"i32 1, !"air.texture", !"air.location_index", i32 40, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>""#.to_string(),
            ),
        ]);
    let textures = detect_embedded_textures(&nodes, &ArgumentBuffers::new(vec![(0, 0, 10)]), &[]);
    assert_eq!(
        textures
            .iter()
            .map(|t| (t.field_ordinal, t.field_offset, t.argument_index))
            .collect::<Vec<_>>(),
        // The nested handle keeps the OUTER ordinal 0 (the first GEP index that reaches it) and
        // an ABSOLUTE byte offset 64 + 0 + 8; the flat sibling is untouched by the descent.
        vec![(0, 72, 32), (1, 72, 40)],
    );
}

/// A wrapper declared as an ARRAY of structs repeats its members once per element, at offsets
/// and argument ids this walk does not enumerate. Binding element 0 alone would leave the rest
/// reading black, so every member under it stays a refusal reason instead.
#[test]
fn an_arrayed_wrapper_is_refused_rather_than_bound_at_element_zero() {
    let nodes = HashMap::from([
            (
                10_u32,
                r#"!"air.struct_type_info", !11, i32 0, i32 48, i32 9, !"MaterialProperty", !"slot", !"air.indirect_argument", i32 0"#.to_string(),
            ),
            (
                11_u32,
                r#"i32 0, i32 8, i32 0, !"texture2d<float, sample>", !"tex", !"air.indirect_argument", !12"#.to_string(),
            ),
            (
                12_u32,
                r#"i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>""#.to_string(),
            ),
        ]);
    let buffers = ArgumentBuffers::new(vec![(0, 0, 10)]);
    assert_eq!(detect_embedded_textures(&nodes, &buffers, &[]), vec![]);
    assert_eq!(
        unsurfaced_embedded_resources(&nodes, &buffers),
        vec![
            "argument buffer parameter 0 holds an air.texture at byte offset 0 inside a nested \
                 struct member"
                .to_string()
        ],
    );
}

/// A wrapped `air.buffer` is a device address a consumer has to write into the owning argument
/// buffer, and nothing binds one, so it is named as unsurfaced rather than becoming a silent
/// `EmbeddedArgument` beside the flat members.
#[test]
fn a_wrapped_buffer_is_named_unsurfaced_and_is_not_an_embedded_argument() {
    let nodes = HashMap::from([
            (
                10_u32,
                r#"!"air.struct_type_info", !11, i32 16, i32 8, i32 0, !"wrapper", !"w", !"air.indirect_argument", i32 4, i32 24, i32 8, i32 0, !"float", !"values", !"air.indirect_argument", !13"#.to_string(),
            ),
            (
                11_u32,
                r#"i32 0, i32 8, i32 0, !"float", !"inner", !"air.indirect_argument", !12"#.to_string(),
            ),
            (
                12_u32,
                r#"i32 0, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.address_space", i32 1"#.to_string(),
            ),
            (
                13_u32,
                r#"i32 1, !"air.buffer", !"air.location_index", i32 7, i32 1, !"air.address_space", i32 1"#.to_string(),
            ),
        ]);
    let buffers = ArgumentBuffers::new(vec![(0, 0, 10)]);
    assert_eq!(
        detect_embedded_arguments(&nodes, &buffers)
            .iter()
            .map(|argument| (argument.field_offset, argument.resource_kind))
            .collect::<Vec<_>>(),
        vec![(24, Some(ArgumentBufferResourceKind::Buffer))],
    );
    assert_eq!(
        unsurfaced_embedded_resources(&nodes, &buffers),
        vec![
            "argument buffer parameter 0 holds an air.buffer at byte offset 16 inside a nested \
                 struct member"
                .to_string()
        ],
    );
}

/// A nested struct with no `air.indirect_argument` suffix is an inline value, not a wrapped
/// resource: the walk steps over it without disturbing the ordinals of the members around it.
#[test]
fn an_inline_nested_struct_does_not_shift_its_siblings() {
    let nodes = HashMap::from([
            (
                10_u32,
                r#"i32 0, i32 4, i32 0, !"uint", !"count", !"air.indirect_argument", !11, !"air.struct_type_info", !12, i32 8, i32 16, i32 0, !"Params", !"params", i32 24, i32 8, i32 0, !"texture2d<float, sample>", !"tex", !"air.indirect_argument", !13"#.to_string(),
            ),
            (
                11_u32,
                r#"i32 0, !"air.indirect_constant", !"air.location_index", i32 0, i32 1"#.to_string(),
            ),
            (
                12_u32,
                r#"i32 0, i32 16, i32 0, !"float4", !"a""#.to_string(),
            ),
            (
                13_u32,
                r#"i32 2, !"air.texture", !"air.location_index", i32 5, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>""#.to_string(),
            ),
        ]);
    let buffers = ArgumentBuffers::new(vec![(0, 0, 10)]);
    assert!(unsurfaced_embedded_resources(&nodes, &buffers).is_empty());
    let textures = detect_embedded_textures(&nodes, &buffers, &[]);
    assert_eq!(textures.len(), 1);
    assert_eq!(textures[0].field_ordinal, 2);
    assert_eq!(textures[0].field_offset, 24);
    assert_eq!(textures[0].argument_index, 5);
}
