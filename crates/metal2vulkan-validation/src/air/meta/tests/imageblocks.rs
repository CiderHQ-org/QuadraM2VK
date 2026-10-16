use super::*;

#[test]
fn kernel_imageblock_layout_is_not_a_buffer_layout() {
    let ll = r#"
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 16, !"air.struct_type_info", !4, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"imageblock<ImageBlockData, layout_explicit>", !"air.arg_name", !"imgBlk"}
!4 = !{i32 0, i32 16, i32 0, !"float4", !"v"}
"#;
    let m = parse_air_kernel_meta(ll).0.unwrap();
    assert_eq!(m.role_of(0), Some(&KernRole::Other));
    assert!(m.layout_of(0).is_none());
    assert!(matches!(
        m.imageblock_layouts.get(&0),
        Some(AirType::Struct(fields))
            if matches!(
                fields.as_slice(),
                [AirMember {
                    offset: 0,
                    ty: AirType::Vec {
                        scalar: Float,
                        lanes: 4
                    }
                }]
            )
    ));
}

#[test]
fn kernel_function_constant_imageblock_keeps_its_cell_layout() {
    let ll = r#"
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.function_constant", !4, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 16, !"air.struct_type_info", !5, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"imageblock<ImageBlockData, layout_explicit>", !"air.arg_name", !"imgBlk"}
!4 = !{ptr addrspace(2) @__metal_implicit_fc_pred_0, !"bool", !"enabled"}
!5 = !{i32 0, i32 16, i32 0, !"float4", !"v"}
"#;
    let m = parse_air_kernel_meta(ll).0.unwrap();
    assert_eq!(m.role_of(0), Some(&KernRole::Other));
    assert!(matches!(
        m.imageblock_layouts.get(&0),
        Some(AirType::Struct(fields))
            if matches!(
                fields.as_slice(),
                [AirMember {
                    offset: 0,
                    ty: AirType::Vec {
                        scalar: Float,
                        lanes: 4
                    }
                }]
            )
    ));
}

/// Each call's plane is its attachment's at its data rate, the `i32` before the last operand
/// (`(.., coord, data rate, index)`): a color-rate read and write of color 3 and 5 reach one
/// plane, a sample-rate read of a sample the shader computes another, and a plain read of the
/// pixel a third.
#[test]
fn kernel_implicit_imageblock_inventory_preserves_attachment_rate_and_access() {
    let ll = r#"
define void @k(i16 %s) {
  %a = call <4 x half> @air.load.implicit_imageblock.v4f16(i32 2, <2 x i16> zeroinitializer, i32 1, i16 3)
  call void @air.store.implicit_imageblock.v4f16(<4 x half> %a, i32 2, <2 x i16> zeroinitializer, i32 1, i16 5)
  %b = call <2 x half> @air.load.implicit_imageblock.v2f16(i32 3, <2 x i16> zeroinitializer, i32 0, i16 0)
  %c = call <2 x half> @air.load.implicit_imageblock.v2f16(i32 3, <2 x i16> zeroinitializer, i32 2, i16 %s)
  call void @air.store.implicit_imageblock.i32(i32 7, i32 4, <2 x i16> zeroinitializer, i32 0, i16 0)
  ret void
}

declare <4 x half> @air.load.implicit_imageblock.v4f16(i32, <2 x i16>, i32, i16)
declare void @air.store.implicit_imageblock.v4f16(<4 x half>, i32, <2 x i16>, i32, i16)
declare <2 x half> @air.load.implicit_imageblock.v2f16(i32, <2 x i16>, i32, i16)
declare void @air.store.implicit_imageblock.i32(i32, i32, <2 x i16>, i32, i16)
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !1}
!1 = !{}
"#;
    let meta = parse_air_kernel_meta(ll).0.unwrap();
    assert_eq!(
        meta.implicit_imageblock_attachments,
        [
            ImplicitImageblockAttachment {
                attachment: 2,
                data_rate: 1,
                format: TextureFormat::Rgba16f,
                reads: true,
                writes: true,
            },
            ImplicitImageblockAttachment {
                attachment: 3,
                data_rate: 0,
                format: TextureFormat::Rg16f,
                reads: true,
                writes: false,
            },
            ImplicitImageblockAttachment {
                attachment: 3,
                data_rate: 2,
                format: TextureFormat::Rg16f,
                reads: true,
                writes: false,
            },
            ImplicitImageblockAttachment {
                attachment: 4,
                data_rate: 0,
                format: TextureFormat::R32ui,
                reads: false,
                writes: true,
            },
        ]
    );
}

/// Each texel type of the implicit-imageblock family names its exact plane, through one table for
/// the intrinsic suffix and an aliased explicit imageblock's member: a `float2` color is an
/// `RG32Float` plane either way.
/// A store to the samples a coverage mask selects, `(value, attachment, coord, mask)`, writes its
/// attachment's plane in the texel's format, as a plain store of the pixel does; a load has no
/// masked form.
#[test]
fn coverage_masked_implicit_store_writes_its_attachment_plane() {
    let ll = r#"
define void @k(i16 %m) {
  call void @air.store.implicit_imageblock.mask.v4f16(<4 x half> zeroinitializer, i32 1, <2 x i16> zeroinitializer, i16 %m)
  call void @air.store.implicit_imageblock.mask.f32(float 0.0, i32 2, <2 x i16> zeroinitializer, i16 3)
  %a = call float @air.load.implicit_imageblock.f32(i32 2, <2 x i16> zeroinitializer, i32 0, i16 0)
  ret void
}

declare void @air.store.implicit_imageblock.mask.v4f16(<4 x half>, i32, <2 x i16>, i16)
declare void @air.store.implicit_imageblock.mask.f32(float, i32, <2 x i16>, i16)
declare float @air.load.implicit_imageblock.f32(i32, <2 x i16>, i32, i16)
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !1}
!1 = !{}
"#;
    let meta = parse_air_kernel_meta(ll).0.unwrap();
    assert_eq!(
        meta.implicit_imageblock_attachments,
        [
            ImplicitImageblockAttachment {
                attachment: 1,
                data_rate: 0,
                format: TextureFormat::Rgba16f,
                reads: false,
                writes: true,
            },
            ImplicitImageblockAttachment {
                attachment: 2,
                data_rate: 0,
                format: TextureFormat::R32f,
                reads: true,
                writes: true,
            },
        ]
    );
    assert_eq!(
        implicit_imageblock_texture_format("air.store.implicit_imageblock.mask.v2f32"),
        Ok(Some(TextureFormat::Rg32f))
    );
    assert!(implicit_imageblock_texture_format("air.load.implicit_imageblock.mask.f32").is_err());
    assert!(
        implicit_imageblock_texture_format("air.store.implicit_imageblock.mask.v3f16").is_err()
    );
}

#[test]
fn implicit_imageblock_texels_name_their_exact_planes() {
    for (suffix, format) in [
        ("f16", TextureFormat::R16f),
        ("v2f16", TextureFormat::Rg16f),
        ("v4f16", TextureFormat::Rgba16f),
        ("f32", TextureFormat::R32f),
        ("v2f32", TextureFormat::Rg32f),
        ("v4f32", TextureFormat::Rgba32f),
        ("i32", TextureFormat::R32ui),
    ] {
        for family in ["load", "store"] {
            assert_eq!(
                implicit_imageblock_texture_format(&format!(
                    "air.{family}.implicit_imageblock.{suffix}"
                )),
                Ok(Some(format)),
                "{family} {suffix}"
            );
        }
    }
    assert_eq!(
        implicit_imageblock_texture_format("air.read.texture_2d"),
        Ok(None)
    );

    let member = |offset, ty| AirMember { offset, ty };
    let planes = aliased_imageblock_planes(&AirType::Struct(vec![
        member(
            0,
            AirType::Vec {
                scalar: Float,
                lanes: 2,
            },
        ),
        member(8, AirType::Scalar(Half)),
    ]))
    .unwrap()
    .unwrap();
    assert_eq!(
        planes
            .iter()
            .map(|plane| (
                plane.attachment,
                plane.offset,
                plane.format,
                plane.intrinsic_suffix
            ))
            .collect::<Vec<_>>(),
        [
            (0, 0, TextureFormat::Rg32f, "v2f32"),
            (1, 8, TextureFormat::R16f, "f16"),
        ]
    );
    // A member no plane represents is an error, never a dropped plane.
    assert!(aliased_imageblock_planes(&AirType::Struct(vec![member(
        0,
        AirType::Vec {
            scalar: Half,
            lanes: 3,
        },
    )]))
    .is_err());
}

#[test]
fn unknown_implicit_imageblock_suffix_cannot_disappear_from_reflection() {
    let ll = r#"
define void @k() {
  %value = call <3 x half> @air.load.implicit_imageblock.v3f16(i32 0, <2 x i16> zeroinitializer, i32 0, i16 0)
  ret void
}
declare <3 x half> @air.load.implicit_imageblock.v3f16(i32, <2 x i16>, i32, i16)
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !1}
!1 = !{}
"#;
    assert!(parse_air_kernel_meta(ll).0.is_none());
    assert!(implicit_imageblock_texture_format("air.load.implicit_imageblock.v3f16").is_err());
}

#[test]
fn direct_fragment_imageblock_data_layout_is_its_own_master() {
    let ll = r#"
define { { <4 x half>, i16 } } @f({ <4 x half>, i16 } %tile) { ret { { <4 x half>, i16 } } poison }
!air.fragment = !{!0}
!0 = !{ptr @f, !1, !2}
!1 = !{!3}
!2 = !{!4}
!3 = !{!"air.imageblock_data", !"air.imageblock_data_size", i32 16, !"air.struct_type_info", !5}
!4 = !{i32 0, !"air.imageblock_data", !"air.imageblock_data_size", i32 16, !"air.struct_type_info", !5}
!5 = !{i32 0, i32 8, i32 0, !"half4", !"color", !"air.raster_order_group", i32 0, i32 8, i32 2, i32 0, !"ushort", !"depth", !"air.raster_order_group", i32 1}
"#;
    let imageblock = parse_air_fragment_meta(ll)
        .0
        .expect("fragment metadata")
        .fragment_imageblock
        .expect("direct imageblock master");
    assert_eq!(imageblock.sample_size, 16);
    assert_eq!(imageblock.members.len(), 2);
    assert_eq!(imageblock.members[0].type_name, "half4");
    assert_eq!(imageblock.members[1].semantic, "depth");
    assert_eq!(imageblock.members[1].raster_order_group, 1);
    assert_eq!(imageblock.inputs[0].members[1].master_member, 1);
    assert_eq!(imageblock.outputs[0].members[0].master_member, 0);
}

/// Views of two masters reach one cell, which spans the larger master. The reflection's master is
/// the larger, here the input view's `Wide` (8 bytes) over the output view's `Swapped` (4 bytes);
/// each view member reaches the master members its bytes overlap in its own master: the output's
/// `a` (byte 2 of `Swapped`) is `Wide`'s `b`, and the input's `w` of `Word` (bytes 0 to 4) is
/// `Wide`'s `a` and `b`.
#[test]
fn views_of_two_masters_reach_the_larger_masters_members_by_bytes() {
    // The output view `ViewA` of `Swapped` (!6) and the input `ViewA` of `Wide` (!7), and the
    // input `ViewW` of `Word` (!13).
    let layouts = r#"
!5 = !{i32 0, i32 2, i32 0, !"half", !"user(a)"}
!6 = !{i32 0, i32 2, i32 0, !"half", !"user(b)", !"air.raster_order_group", i32 0, i32 2, i32 2, i32 0, !"half", !"user(a)", !"air.raster_order_group", i32 0}
!7 = !{i32 0, i32 2, i32 0, !"half", !"user(a)", !"air.raster_order_group", i32 0, i32 2, i32 2, i32 0, !"half", !"user(b)", !"air.raster_order_group", i32 0, i32 4, i32 4, i32 0, !"uint", !"user(w)", !"air.raster_order_group", i32 0}
!13 = !{i32 0, i32 4, i32 0, !"uint", !"user(w)", !"air.raster_order_group", i32 0}
"#;
    let output = r#"!3 = !{!"air.imageblock_data", !"air.imageblock_data_size", i32 4, !"air.struct_type_info", !5, !"air.imageblock_master", !6, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"ViewA", !"air.arg_name", !"view"}"#;
    let reflect = |input: &str| {
        let ll = format!(
            "define {{ {{ half }} }} @f({{ half }} %in) {{ ret {{ {{ half }} }} poison }}\n\
             !air.fragment = !{{!0}}\n!0 = !{{ptr @f, !1, !2}}\n!1 = !{{!3}}\n!2 = !{{!4}}\n\
             {output}\n{input}\n{layouts}"
        );
        parse_air_fragment_meta(&ll)
            .0
            .expect("fragment metadata")
            .fragment_imageblock
            .expect("imageblock")
    };
    let semantics = |ib: &FragmentImageblock| -> Vec<String> {
        ib.members.iter().map(|m| m.semantic.clone()).collect()
    };
    let reached = |p: &FragmentImageblockProjection| -> Vec<(u32, u32)> {
        p.members
            .iter()
            .map(|m| (m.projection_member, m.master_member))
            .collect()
    };
    let wide = reflect(
        r#"!4 = !{i32 0, !"air.imageblock_data", !"air.imageblock_data_size", i32 8, !"air.struct_type_info", !5, !"air.imageblock_master", !7, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"ViewA", !"air.arg_name", !"in"}"#,
    );
    assert_eq!(wide.sample_size, 8);
    assert_eq!(semantics(&wide), ["user(a)", "user(b)", "user(w)"]);
    assert_eq!(reached(&wide.inputs[0]), [(0, 0)]);
    assert_eq!(reached(&wide.outputs[0]), [(0, 1)]);
    // Through `Word` (4 bytes, a tie with `Swapped`): the output view's master, whose `b` and `a`
    // the input's `w` overlaps.
    let word = reflect(
        r#"!4 = !{i32 0, !"air.imageblock_data", !"air.imageblock_data_size", i32 4, !"air.struct_type_info", !13, !"air.imageblock_master", !13, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"ViewW", !"air.arg_name", !"in"}"#,
    );
    assert_eq!(word.sample_size, 4);
    assert_eq!(semantics(&word), ["user(b)", "user(a)"]);
    assert_eq!(reached(&word.inputs[0]), [(0, 0), (0, 1)]);
    assert_eq!(reached(&word.outputs[0]), [(0, 1)]);
}
