//! Mipmapped texture literals (`case::mips`).

use super::*;
use crate::case::mips::{full_chain, level_dimensions, literal_len};

fn mipmapped(texture_type: TextureType, dimensions: [u32; 3], levels: u32) -> AuthoredCase {
    let mut case = example();
    let len = literal_len(texture_type, dimensions, levels, 16).unwrap();
    case.textures.push(TextureResource {
        binding: 0,
        role: ResourceRole::Input,
        texture_type,
        format: TextureFormat::Rgba32Float,
        dimensions,
        sample_count: 1,
        bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode(vec![0u8; len])),
        initial_bytes_b64: None,
        sparse: None,
        mip_levels: levels,
    });
    case.case_id = case.computed_case_id().unwrap();
    case
}

/// Each level halves the axes Metal halves, floored at 1; layers and cube faces are every
/// level's; the full chain is `floor(log2(largest halved axis)) + 1`.
#[test]
fn levels_halve_the_axes_metal_halves() {
    assert_eq!(level_dimensions(TextureType::D2, [16, 4, 1], 3), [2, 1, 1]);
    assert_eq!(
        level_dimensions(TextureType::D2Array, [8, 8, 3], 2),
        [2, 2, 3]
    );
    assert_eq!(level_dimensions(TextureType::Cube, [4, 4, 6], 2), [1, 1, 6]);
    assert_eq!(level_dimensions(TextureType::D3, [8, 4, 2], 2), [2, 1, 1]);
    assert_eq!(level_dimensions(TextureType::D2, [16, 4, 1], 40), [1, 1, 1]);
    assert_eq!(full_chain(TextureType::D2, [16, 4, 1]), 5);
    assert_eq!(full_chain(TextureType::D2, [17, 1, 1]), 5);
    assert_eq!(full_chain(TextureType::D2Array, [1, 1, 9]), 1);
    assert_eq!(full_chain(TextureType::D3, [2, 2, 8]), 4);
    // 16x4 RGBA32F: 16*4 + 8*2 + 4*1 + 2*1 + 1*1 texels.
    assert_eq!(
        literal_len(TextureType::D2, [16, 4, 1], 5, 16),
        Some(87 * 16)
    );
}

/// A mipmapped literal holds every level, level 0 first; one level is the field's only spelling,
/// so a case that leaves it out keeps its identity.
#[test]
fn a_mipmapped_literal_holds_every_level() {
    let case = mipmapped(TextureType::D2, [4, 2, 1], 3);
    case.validate_literal_resources().unwrap();
    let value = serde_json::to_value(&case.textures[0]).unwrap();
    assert_eq!(value["mip_levels"], 3);
    let mut one = case.clone();
    one.textures[0].mip_levels = 1;
    one.textures[0].bytes_b64 = Some(base64::engine::general_purpose::STANDARD.encode([0u8; 128]));
    assert!(serde_json::to_value(&one.textures[0])
        .unwrap()
        .get("mip_levels")
        .is_none());

    let literal = crate::literal::LiteralResources::prepare(&case).unwrap();
    let texture = &literal.textures[0];
    assert_eq!(texture.mip_levels, 3);
    let (dimensions, bytes) = texture.level(0).unwrap();
    assert_eq!((dimensions, bytes.len()), ([4, 2, 1], 4 * 2 * 16));
    let (dimensions, bytes) = texture.level(2).unwrap();
    assert_eq!((dimensions, bytes.len()), ([1, 1, 1], 16));
    assert!(texture.level(3).is_err());

    let mut short = case.clone();
    short.textures[0].bytes_b64 =
        Some(base64::engine::general_purpose::STANDARD.encode(vec![0u8; 4 * 2 * 16]));
    short.case_id = short.computed_case_id().unwrap();
    assert!(short.validate_literal_resources().is_err());
}

/// Levels only where each executor makes them exactly: a single-sample input of a mipmappable
/// type, not sparse, within its full chain.
#[test]
fn levels_are_refused_where_no_executor_makes_them() {
    let refused = |case: AuthoredCase, why: &str| {
        let errors = case.validate_literal_resources().unwrap_err().join("\n");
        assert!(errors.contains(why), "{why}: {errors}");
    };
    refused(mipmapped(TextureType::D2, [4, 4, 1], 4), "has at most 3");
    refused(
        mipmapped(TextureType::D1, [8, 1, 1], 2),
        "a mipmapped texture is 2D, 2D array, 3D, cube, or cube array",
    );
    let mut written = mipmapped(TextureType::D2, [4, 4, 1], 2);
    written.textures[0].role = ResourceRole::InOut;
    written.textures[0].initial_bytes_b64 = written.textures[0].bytes_b64.take();
    written.case_id = written.computed_case_id().unwrap();
    refused(written, "a mipmapped texture is an input");
    let mut zero = mipmapped(TextureType::D2, [4, 4, 1], 1);
    zero.textures[0].mip_levels = 0;
    zero.case_id = zero.computed_case_id().unwrap();
    refused(zero, "mip_levels is at least 1");
}
