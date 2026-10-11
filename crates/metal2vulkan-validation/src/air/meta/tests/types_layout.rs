use super::*;

#[test]
fn nested_bitfield_struct_type_info_uses_declared_storage() {
    let ll = r#"
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Header", !"air.arg_name", !"header"}
!4 = !{!"air.struct_type_info", !5, i32 0, i32 4, i32 0, !"Header::(anonymous)", !"flags", i32 4, i32 4, i32 0, !"float", !"tail"}
!5 = !{i32 0, i32 4, i32 0, !"uint", !"mode", i32 0, i32 1, i32 0, !"bool", !"enabled", i32 1, i32 4, i32 0, !"uint", !"depth"}
"#;
    let m = parse_air_kernel_meta(ll).0.unwrap();
    assert_eq!(
        m.layout_of(0),
        Some(&AirType::Struct(vec![
            AirMember {
                offset: 0,
                ty: AirType::Scalar(UInt)
            },
            AirMember {
                offset: 4,
                ty: AirType::Scalar(Float)
            }
        ]))
    );
}

/// AIR spells a C bitfield by giving each field the offset of the byte its run starts in beside
/// the size of the WHOLE underlying storage unit. Read member-by-member, `cv3d` corpus source
/// `5e3a0bbd` reconstructs `uint pixel_x` at 0, `uint pixel_y` at 1 and `uint matid` at 3 as a
/// seven-byte struct for a four-byte argument, which `validate_descriptor_abi` refuses -- so the
/// whole module failed to translate over a layout nobody asked for.
#[test]
fn top_level_bitfield_struct_type_info_has_no_reconstructed_layout() {
    let ll = r#"
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Packed", !"air.arg_name", !"packed"}
!4 = !{i32 0, i32 4, i32 0, !"uint", !"pixel_x", i32 1, i32 4, i32 0, !"uint", !"pixel_y", i32 3, i32 4, i32 0, !"uint", !"matid"}
"#;
    let m = parse_air_kernel_meta(ll).0.unwrap();
    assert_eq!(m.layout_of(0), None);
}

/// A union is the other shape with overlapping members, and it is refused for the same reason:
/// the layout's consumer builds a SPIR-V struct type from it, and a `Block` cannot decorate two
/// members onto the same bytes. Handing it one does not produce an overlapping struct -- it makes
/// the emitter address the WHOLE buffer as raw words. Reporting the union instead cost `87c1f8ca`
/// every typed access chain into its `PKMetalStrokeVertex` array; refusing it lets the emitter's
/// own reconstruction take over, which gained typed chains on 32 corpus sources and lost them on
/// none.
#[test]
fn union_struct_type_info_has_no_reconstructed_layout() {
    let ll = r#"
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!4 = !{i32 0, i32 4, i32 0, !"float", !"as_float", i32 0, i32 4, i32 0, !"int", !"as_int", i32 0, i32 4, i32 0, !"uint", !"as_uint"}
"#;
    let m = parse_air_kernel_meta(ll).0.unwrap();
    assert_eq!(m.layout_of(0), None);
}

/// A nested node that is refused leaves its member exactly as well described as before, because
/// AIR stated the member's size in the enclosing tuple. It must NOT take the whole enclosing
/// layout down with it.
#[test]
fn a_refused_nested_node_costs_only_its_own_member() {
    let ll = r#"
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 12, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Outer", !"air.arg_name", !"outer"}
!4 = !{i32 0, i32 4, i32 0, !"float", !"lead", !"air.struct_type_info", !5, i32 4, i32 4, i32 0, !"Bits", !"bits", i32 8, i32 4, i32 0, !"uint", !"tail"}
!5 = !{i32 0, i32 4, i32 0, !"uint", !"x", i32 1, i32 4, i32 0, !"uint", !"y"}
"#;
    let m = parse_air_kernel_meta(ll).0.unwrap();
    assert_eq!(
        m.layout_of(0),
        Some(&AirType::Struct(vec![
            AirMember {
                offset: 0,
                ty: AirType::Scalar(Float)
            },
            AirMember {
                offset: 4,
                ty: AirType::Scalar(UInt)
            },
            AirMember {
                offset: 8,
                ty: AirType::Scalar(UInt)
            }
        ]))
    );
}

// `air.struct_type_info` reconstruction, including a matrix (float3x4 -> { [3 x float4] }) and a
// NESTED struct (mirrors ColorCorrectionParametric { float3x4, ColorCurve{float3 x2} }).
const RECON_LL: &str = r#"
!air.fragment = !{!0}
!0 = !{ptr @F, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0}
!3 = !{!4, !5, !9}
!4 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 16, !"air.struct_type_info", !6, !"air.arg_type_name", !"Pair"}
!5 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 96, !"air.struct_type_info", !7, !"air.arg_type_name", !"PCC"}
!6 = !{i32 0, i32 8, i32 0, !"packed_float2", !"_a", i32 8, i32 8, i32 0, !"packed_float2", !"_b", i32 16, i32 4, i32 0, !"packed_float1", !"_c"}
!7 = !{i32 0, i32 48, i32 0, !"float3x4", !"_m", !"air.struct_type_info", !8, i32 48, i32 48, i32 0, !"Curve", !"_c"}
!8 = !{i32 0, i32 16, i32 0, !"float3", !"_x", i32 16, i32 16, i32 0, !"float3", !"_y"}
!9 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 32, !"air.struct_type_info", !10, !"air.arg_type_name", !"HistogramLayout"}
!10 = !{i32 0, i32 4, i32 4, !"uint", !"indices", i32 16, i32 4, i32 4, !"uint", !"counts"}
"#;

#[test]
fn struct_type_info_reconstruction() {
    let m = parse_air_fragment_meta(RECON_LL).0.unwrap();
    // Pair { packed_float2, packed_float2, packed_float1 }
    assert_eq!(
        m.layout_of(0),
        Some(&AirType::Struct(vec![
            AirMember {
                offset: 0,
                ty: AirType::PackedVec {
                    scalar: Float,
                    lanes: 2
                }
            },
            AirMember {
                offset: 8,
                ty: AirType::PackedVec {
                    scalar: Float,
                    lanes: 2
                }
            },
            AirMember {
                offset: 16,
                ty: AirType::PackedVec {
                    scalar: Float,
                    lanes: 1
                }
            }
        ]))
    );
    // PCC { float3x4 matrix, Curve { float3, float3 } }
    assert_eq!(
        m.layout_of(1),
        Some(&AirType::Struct(vec![
            AirMember {
                offset: 0,
                ty: AirType::Matrix {
                    scalar: Float,
                    cols: 3,
                    rows: 4
                }
            },
            AirMember {
                offset: 48,
                ty: AirType::Struct(vec![
                    AirMember {
                        offset: 0,
                        ty: AirType::Vec {
                            scalar: Float,
                            lanes: 3
                        }
                    },
                    AirMember {
                        offset: 16,
                        ty: AirType::Vec {
                            scalar: Float,
                            lanes: 3
                        }
                    }
                ])
            },
        ]))
    );
    // HistogramLayout { uint indices[4], uint counts[4] }
    assert_eq!(
        m.layout_of(2),
        Some(&AirType::Struct(vec![
            AirMember {
                offset: 0,
                ty: AirType::Array {
                    elem: Box::new(AirType::Scalar(UInt)),
                    len: 4
                }
            },
            AirMember {
                offset: 16,
                ty: AirType::Array {
                    elem: Box::new(AirType::Scalar(UInt)),
                    len: 4
                }
            },
        ]))
    );
}

/// Every arrayed texture spelling reflects as arrayed, including the two that no enumeration of
/// per-dimension substrings caught: `texture2d_ms_array` and `depth2d_ms_array` do not contain
/// `2d_array`, so they used to reflect as plain multisample 2D textures while the interface pass
/// built an arrayed `OpTypeImage` for them from the same name. One `_array` suffix answers the
/// question for all of them.
#[test]
fn every_arrayed_texture_spelling_reflects_as_arrayed() {
    use crate::air::meta::{texture_shape_from_name, TextureDimension};

    let arrayed: [(&str, TextureDimension, bool); 8] = [
        (
            "texture1d_array<float, sample>",
            TextureDimension::D1,
            false,
        ),
        (
            "texture2d_array<float, sample>",
            TextureDimension::D2,
            false,
        ),
        (
            "texture2d_ms_array<float, read>",
            TextureDimension::D2,
            true,
        ),
        ("depth2d_array<float, sample>", TextureDimension::D2, false),
        ("depth2d_ms_array<float, read>", TextureDimension::D2, true),
        (
            "texturecube_array<float, sample>",
            TextureDimension::Cube,
            false,
        ),
        (
            "depthcube_array<float, sample>",
            TextureDimension::Cube,
            false,
        ),
        (
            "array<texture2d_ms_array<float, read>, 4>",
            TextureDimension::D2,
            true,
        ),
    ];
    for (name, dimension, multisampled) in arrayed {
        let shape = texture_shape_from_name(name);
        assert!(shape.arrayed, "{name} is arrayed");
        assert_eq!(shape.dimension, dimension, "{name}");
        assert_eq!(shape.multisampled, multisampled, "{name}");
    }

    let single: [(&str, TextureDimension, bool); 7] = [
        ("texture1d<float, sample>", TextureDimension::D1, false),
        ("texture2d<float, sample>", TextureDimension::D2, false),
        ("texture2d_ms<float, read>", TextureDimension::D2, true),
        ("texture3d<float, sample>", TextureDimension::D3, false),
        ("texturecube<float, sample>", TextureDimension::Cube, false),
        ("depth2d<float, sample>", TextureDimension::D2, false),
        (
            "texture_buffer<float, read>",
            TextureDimension::Buffer,
            false,
        ),
    ];
    for (name, dimension, multisampled) in single {
        let shape = texture_shape_from_name(name);
        assert!(!shape.arrayed, "{name} is a single texture");
        assert_eq!(shape.dimension, dimension, "{name}");
        assert_eq!(shape.multisampled, multisampled, "{name}");
    }
}
