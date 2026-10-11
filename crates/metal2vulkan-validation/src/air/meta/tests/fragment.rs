use super::*;

#[test]
fn fragment_roles() {
    let m = parse_air_fragment_meta(FRAG_LL).0.unwrap();
    assert_eq!(m.render_target_members.len(), 1);
    assert_eq!(
        m.render_target_members
            .iter()
            .map(|(_, location)| *location)
            .collect::<Vec<_>>(),
        vec![0]
    );
    assert_eq!(m.role_of(0), Some(&FragRole::Position));
    assert_eq!(m.role_of(1), Some(&FragRole::Varying(0)));
    assert_eq!(m.role_of(2), Some(&FragRole::Texture(0)));
    assert_eq!(m.role_of(3), Some(&FragRole::Buffer(5)));
    assert_eq!(m.role_of(4), Some(&FragRole::PointCoord));
    assert_eq!(m.role_of(5), Some(&FragRole::FrontFacing));
    assert_eq!(m.texture_type_name(2), Some("texture2d<float, sample>"));
    assert_eq!(m.varying_type(0), Some("float2"));
    assert_eq!(
        m.varying_names.get(&0).map(String::as_str),
        Some("texCoord")
    );
    assert_eq!(
        m.varying_user_semantics.get(&0).map(String::as_str),
        Some("generated")
    );
    assert!(!m.varying_is_flat(0));
}

/// Every AIR interpolation marker is decoded into [`VaryingInterpolation`] or listed here as
/// deliberately ignored.
///
/// `air.no_perspective` and `air.centroid` were silently dropped for as long as this decode read
/// only `air.flat`: a `[[center_no_perspective]]` varying translated to a perspective-correct one,
/// valid SPIR-V that renders differently. The inventory below is the ABI's whole interpolation
/// vocabulary, so a marker added to AIR later shows up as a case this test does not name rather
/// than as another silent default.
#[test]
fn every_interpolation_marker_is_decoded_or_deliberately_ignored() {
    let with = |markers: &str| {
        let ll = FRAG_LL.replace(
            r#"!"air.fragment_input", !"generated","#,
            &format!(r#"!"air.fragment_input", !"generated", {markers}"#),
        );
        parse_air_fragment_meta(&ll)
            .0
            .unwrap()
            .varying_interpolation(0)
    };

    // Decoded: the perspective axis.
    assert_eq!(
        with(r#"!"air.center", !"air.perspective","#),
        VaryingInterpolation::default(),
        "AIR's defaults are Vulkan's defaults and decode to no decoration"
    );
    assert!(with(r#"!"air.center", !"air.no_perspective","#).no_perspective);
    assert!(!with(r#"!"air.center", !"air.perspective","#).no_perspective);

    // Decoded: the sampling axis.
    assert_eq!(
        with(r#"!"air.center", !"air.perspective","#).sampling,
        VaryingSampling::Center
    );
    assert_eq!(
        with(r#"!"air.centroid", !"air.perspective","#).sampling,
        VaryingSampling::Centroid
    );
    assert_eq!(
        with(r#"!"air.sample", !"air.no_perspective","#).sampling,
        VaryingSampling::Sample
    );

    // Decoded: `air.flat`, which AIR states instead of the pair above.
    assert!(with(r#"!"air.flat","#).flat);
    assert!(!with(r#"!"air.center", !"air.perspective","#).flat);

    // Deliberately ignored: every other marker AIR states on the same node. `FRAG_LL` already
    // carries `air.fragment_input`, `air.arg_type_name` and `air.arg_name` on this argument, so a
    // node with no interpolation markers at all still decoding to the default is what pins that
    // none of them is mistaken for one.
    assert_eq!(with(""), VaryingInterpolation::default());
}

/// The two spellings of `[[sample_mask]]` are different roles: an argument is the coverage coming
/// in, a return member is the coverage going out. Reading either as the other loses a direction.
#[test]
fn the_two_sample_mask_directions_are_distinct_roles() {
    let ll = FRAG_LL
        .replace(
            r#"!20 = !{i32 1, !"air.fragment_input", !"generated","#,
            r#"!20 = !{i32 1, !"air.sample_mask_in","#,
        )
        .replace(
            r#"!17 = !{!"air.render_target", i32 0, i32 0}"#,
            r#"!17 = !{!"air.sample_mask", !"air.arg_type_name", !"uint"}"#,
        );
    let m = parse_air_fragment_meta(&ll).0.unwrap();
    assert_eq!(m.role_of(1), Some(&FragRole::SampleMaskIn));
    assert!(
        m.is_sample_mask_member(0),
        "and the return member is the output"
    );
    assert!(
        m.unmodelled_input_params.is_empty(),
        "`air.sample_mask_in` has a lowering"
    );
}

/// A gated-off argument is absent whichever builtin it names, so neither of the two roles that
/// declare an Input variable of their own may materialize one for it.
#[test]
fn a_disabled_builtin_argument_declares_no_input_variable() {
    for role in ["sample_mask_in", "barycentric_coord"] {
        let ll = FRAG_LL.replace(
            r#"!20 = !{i32 1, !"air.fragment_input", !"generated","#,
            &format!(
                r#"!20 = !{{i32 1, !"air.function_constant", !97, !"air.{role}", !"air.center", !"air.perspective","#
            ),
        ) + "\n!97 = !{ptr addrspace(2) @off.MTL_FC_INIT_0_b, !\"bool\", !\"off\", i32 0, i1 false}\n";
        let m = parse_air_fragment_meta(&ll).0.unwrap();
        assert_eq!(
            m.role_of(1),
            Some(&FragRole::Other),
            "an off-by-default `air.{role}` argument must not claim its builtin"
        );
    }
}

/// A barycentric argument carries its own perspective axis, and only the perspective axis.
#[test]
fn a_barycentric_argument_keeps_its_perspective_axis() {
    let with = |markers: &str| {
        let ll = FRAG_LL.replace(
            r#"!20 = !{i32 1, !"air.fragment_input", !"generated","#,
            &format!(r#"!20 = !{{i32 1, !"air.barycentric_coord", {markers}"#),
        );
        let m = parse_air_fragment_meta(&ll).0.unwrap();
        m.role_of(1).expect("decoded role").clone()
    };
    assert_eq!(
        with(r#"!"air.center", !"air.perspective","#),
        FragRole::BarycentricCoord {
            no_perspective: false
        }
    );
    assert_eq!(
        with(r#"!"air.center", !"air.no_perspective","#),
        FragRole::BarycentricCoord {
            no_perspective: true
        }
    );
    assert_eq!(
        with(r#"!"air.centroid", !"air.perspective","#),
        FragRole::BarycentricCoord {
            no_perspective: false
        },
        "SPIR-V has no centroid barycentric builtin, so the sampling axis is deliberately ignored"
    );
}

/// `air.sample` means per-sample interpolation on a varying and sample-read access on a texture.
/// The decode must not read one argument's markers as another's.
#[test]
fn the_sample_marker_is_read_per_argument_not_per_module() {
    let ll = FRAG_LL
        .replace(
            r#"!"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"texture2d<float, sample>""#,
            r#"!"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>""#,
        )
        .replace(
            r#"!"air.fragment_input", !"generated","#,
            r#"!"air.fragment_input", !"generated", !"air.center", !"air.perspective","#,
        );
    let m = parse_air_fragment_meta(&ll).0.unwrap();
    assert_eq!(m.role_of(2), Some(&FragRole::Texture(0)));
    assert_eq!(
        m.varying_interpolation(0).sampling,
        VaryingSampling::Center,
        "the texture argument's `air.sample` access qualifier is not the varying's sampling rate"
    );
}

#[test]
fn fragment_flat_varying_metadata() {
    let ll = FRAG_LL.replace(
        r#"!20 = !{i32 1, !"air.fragment_input", !"generated", !"air.arg_type_name", !"float2", !"air.arg_name", !"texCoord"}"#,
        r#"!20 = !{i32 1, !"air.fragment_input", !"generated", !"air.flat", !"air.arg_type_name", !"float2", !"air.arg_name", !"texCoord"}"#,
    );
    let m = parse_air_fragment_meta(&ll).0.unwrap();
    assert!(m.varying_is_flat(0));
}

#[test]
fn fragment_render_target_array_index_role() {
    let ll = r#"
!air.fragment = !{!0}
!0 = !{ptr @F, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0}
!3 = !{!4}
!4 = !{i32 0, !"air.render_target_array_index", !"air.arg_type_name", !"ushort", !"air.arg_name", !"layer"}
"#;
    let meta = parse_air_fragment_meta(ll).0.unwrap();
    assert_eq!(meta.role_of(0), Some(&FragRole::RenderTargetArrayIndex));
}

#[test]
fn fragment_color_input_uses_render_target_location() {
    let ll = r#"
!air.fragment = !{!0}
!0 = !{ptr @F, !1, !4}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!4 = !{!5}
!5 = !{i32 2, !"air.render_target", i32 2, !"air.arg_type_name", !"float", !"air.arg_name", !"d0"}
"#;
    let m = parse_air_fragment_meta(ll).0.unwrap();
    assert_eq!(m.role_of(2), Some(&FragRole::ColorInput(2)));
    assert_eq!(
        m.color_input_type_names.get(&2).map(String::as_str),
        Some("float")
    );
}

#[test]
fn fragment_render_target_indices() {
    let ll = FRAG_LL.replace(
        "!17 = !{!\"air.render_target\", i32 0, i32 0}",
        "!17 = !{!\"air.render_target\", i32 3, i32 0}",
    );
    let m = parse_air_fragment_meta(&ll).0.unwrap();
    assert_eq!(m.render_target_members.len(), 1);
    assert_eq!(m.render_target_members, vec![(0, 3)]);
    assert_eq!(
        m.render_target_members
            .iter()
            .map(|(_, location)| *location)
            .collect::<Vec<_>>(),
        vec![3]
    );
}

#[test]
fn fragment_render_target_type_names() {
    let ll = FRAG_LL.replace(
        "!17 = !{!\"air.render_target\", i32 0, i32 0}",
        "!17 = !{!\"air.render_target\", i32 0, i32 0, !\"air.arg_type_name\", !\"int4\"}",
    );
    let m = parse_air_fragment_meta(&ll).0.unwrap();
    assert_eq!(m.render_target_type_name(0), Some("int4"));
}

#[test]
fn fragment_stencil_output_is_not_a_render_target() {
    let ll = FRAG_LL.replace(
        "!17 = !{!\"air.render_target\", i32 0, i32 0}",
        "!17 = !{!\"air.stencil\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"stencil\"}",
    );
    let m = parse_air_fragment_meta(&ll).0.unwrap();
    assert_eq!(m.render_target_members.len(), 0);
    assert!(m.render_target_members.is_empty());
    assert!(m.render_target_members.is_empty());
    assert_eq!(m.stencil_members, vec![0]);
    assert!(m.is_stencil_member(0));
    assert!(!m.is_stencil_member(1));
}

/// Every non-color fragment output role is read by the same walk, so none can be the one nobody
/// copied it for — which is how `air.sample_mask` came to be dropped.
#[test]
fn each_non_color_fragment_output_role_is_recognised() {
    let with = |node: &str| {
        let ll = FRAG_LL.replace(r#"!17 = !{!"air.render_target", i32 0, i32 0}"#, node);
        parse_air_fragment_meta(&ll).0.unwrap()
    };

    let depth = with(r#"!17 = !{!"air.depth", !"air.arg_type_name", !"float"}"#);
    assert!(depth.is_depth_member(0));
    assert!(!depth.is_stencil_member(0) && !depth.is_sample_mask_member(0));

    let stencil = with(r#"!17 = !{!"air.stencil", !"air.arg_type_name", !"uint"}"#);
    assert!(stencil.is_stencil_member(0));
    assert!(!stencil.is_depth_member(0) && !stencil.is_sample_mask_member(0));

    let mask = with(r#"!17 = !{!"air.sample_mask", !"air.arg_type_name", !"uint"}"#);
    assert!(mask.is_sample_mask_member(0));
    assert!(!mask.is_depth_member(0) && !mask.is_stencil_member(0));
    assert_eq!(
        mask.render_target_members.len(),
        0,
        "a coverage mask is not a color attachment"
    );
    assert!(mask.render_target_members.is_empty());
}

/// A function-constant-gated output is off by default, and that gate is checked for every role.
#[test]
fn a_disabled_sample_mask_output_is_not_reported() {
    let ll = FRAG_LL.replace(
        r#"!17 = !{!"air.render_target", i32 0, i32 0}"#,
        r#"!17 = !{!"air.function_constant", !99, !"air.sample_mask", !"air.arg_type_name", !"uint"}
!99 = !{ptr addrspace(2) @off.MTL_FC_INIT_0_b, !"bool", !"off", i32 0, i1 false}"#,
    );
    let m = parse_air_fragment_meta(&ll).0.unwrap();
    assert!(
        !m.is_sample_mask_member(0),
        "an off-by-default mask must not claim the builtin"
    );
}
