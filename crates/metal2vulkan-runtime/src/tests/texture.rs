//! The texture object's pure parts: image aspects, Metal regions in Vulkan's counts, the copy
//! regions of CPU copies, and the device's texture state.

use super::super::copy_plan::ImageRegion;
use super::super::texture::{
    buffer_image_copy, image_aspects, narrow_range, texture_region, TextureState,
};
use super::super::values::Region;
use super::image_plan::committed;
use ash::vk;
use metal2vulkan::vk::{Format, ImageAspectFlags};

#[test]
fn images_carry_their_formats_aspects() {
    use ImageAspectFlags as A;
    assert_eq!(image_aspects(Format::R8G8B8A8_UNORM), A::COLOR);
    assert_eq!(image_aspects(Format::BC1_RGBA_UNORM_BLOCK), A::COLOR);
    assert_eq!(image_aspects(Format::D32_SFLOAT), A::DEPTH);
    assert_eq!(image_aspects(Format::S8_UINT), A::STENCIL);
    assert_eq!(
        image_aspects(Format::D32_SFLOAT_S8_UINT),
        A::DEPTH | A::STENCIL
    );
}

#[test]
fn metal_regions_narrow_to_vulkans_counts_or_refuse() {
    let r = texture_region(&Region::new_3d(1, 2, 3, 4, 5, 6), 7, 8).unwrap();
    assert_eq!(
        (r.level, r.slice, r.origin, r.size),
        (7, 8, [1, 2, 3], [4, 5, 6])
    );
    assert!(texture_region(&Region::new_2d(1 << 32, 0, 1, 1), 0, 0).is_none());
    assert!(texture_region(&Region::new_1d(0, 1), 0, 1 << 40).is_none());
    assert_eq!(narrow_range(&(2..5)), Some(2..5));
    assert_eq!(narrow_range(&(0..(1 << 33))), None);
}

#[test]
fn cpu_copies_read_a_tightly_packed_buffer() {
    let copy = buffer_image_copy(&ImageRegion {
        aspect: ImageAspectFlags::COLOR,
        level: 2,
        layer: 5,
        layers: 1,
        origin: [1, 2, 0],
        size: [3, 4, 1],
    });
    assert_eq!(
        (
            copy.buffer_offset,
            copy.buffer_row_length,
            copy.buffer_image_height
        ),
        (0, 0, 0)
    );
    let s = copy.image_subresource;
    assert_eq!(
        (
            s.aspect_mask.as_raw(),
            s.mip_level,
            s.base_array_layer,
            s.layer_count
        ),
        (vk::ImageAspectFlags::COLOR.as_raw(), 2, 5, 1)
    );
    assert_eq!(
        (
            copy.image_offset.x,
            copy.image_offset.y,
            copy.image_offset.z
        ),
        (1, 2, 0)
    );
    assert_eq!(
        (
            copy.image_extent.width,
            copy.image_extent.height,
            copy.image_extent.depth
        ),
        (3, 4, 1)
    );
}

#[test]
fn the_granularity_is_a_whole_power_of_two() {
    let (_, desc, _) = committed().into_iter().next().unwrap();
    assert_eq!(TextureState::new(&desc, 0, false).granularity(), 1);
    assert_eq!(TextureState::new(&desc, 1024, false).granularity(), 1024);
    assert_eq!(TextureState::new(&desc, 1000, false).granularity(), 1024);
}

#[test]
fn copy_units_are_the_images_aspects_blocks() {
    use super::super::texture::copy_units;
    use metal2vulkan::metal::PixelFormat;
    use ImageAspectFlags as A;
    assert_eq!(
        copy_units(Format::R8G8B8A8_UNORM, PixelFormat::RGBA8Unorm),
        vec![(A::COLOR, 4, [1, 1])]
    );
    assert_eq!(
        copy_units(Format::BC1_RGBA_UNORM_BLOCK, PixelFormat::BC1RGBA),
        vec![(A::COLOR, 8, [4, 4])]
    );
    assert_eq!(
        copy_units(
            Format::D32_SFLOAT_S8_UINT,
            PixelFormat::Depth32FloatStencil8
        ),
        vec![(A::DEPTH, 4, [1, 1]), (A::STENCIL, 1, [1, 1])]
    );
    assert_eq!(
        copy_units(Format::D16_UNORM, PixelFormat::Depth16Unorm),
        vec![(A::DEPTH, 2, [1, 1])]
    );
}

#[test]
fn zero_copies_cover_every_texel_once_in_bands() {
    use super::super::texture::zero_copies;
    // A 2D array of 3 layers, 9x7 with 3 levels, of 4-byte texels, in bands of at most 2 rows
    // (2 rows of 36 bytes fit 80 bytes, 3 do not).
    let unit = [(ImageAspectFlags::COLOR, 4, [1, 1])];
    let (length, copies) = zero_copies([9, 7, 1], 3, 3, false, &unit, 80);
    assert_eq!(length, 72);
    let mut covered = std::collections::BTreeMap::new();
    for c in &copies {
        assert_eq!(
            (c.layers, c.origin[0], c.origin[2], c.size[2]),
            (1, 0, 0, 1)
        );
        let [w, h, _] = [9u32, 7, 1].map(|a| (a >> c.level).max(1));
        assert_eq!(c.size[0], w);
        assert!(u64::from(c.size[0]) * 4 * u64::from(c.size[1]) <= length);
        for y in c.origin[1]..c.origin[1] + c.size[1] {
            assert!(y < h);
            *covered.entry((c.level, c.layer, y)).or_insert(0) += 1;
        }
    }
    let rows: u32 = (0..3).map(|l| (7u32 >> l).max(1)).sum();
    assert_eq!(covered.len() as u32, rows * 3);
    assert!(covered.values().all(|&n| n == 1));
    // Compressed blocks: bands of block rows, a partial last block row reaching the edge.
    let bc = [(ImageAspectFlags::COLOR, 8, [4, 4])];
    let (length, copies) = zero_copies([10, 10, 1], 1, 1, false, &bc, 24);
    assert_eq!(length, 24);
    let bands: Vec<(u32, u32)> = copies.iter().map(|c| (c.origin[1], c.size[1])).collect();
    assert_eq!(bands, vec![(0, 4), (4, 4), (8, 2)]);
    // A 3D level: each depth slice; a row larger than the budget is one band of one row.
    let (length, copies) = zero_copies([64, 2, 3], 1, 1, true, &unit, 16);
    assert_eq!(length, 256);
    assert_eq!(copies.len(), 6);
    assert!(copies.iter().all(|c| c.size[1] == 1));
}
