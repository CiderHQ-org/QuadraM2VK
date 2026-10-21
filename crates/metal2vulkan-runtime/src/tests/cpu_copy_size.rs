//! FFI copy spans use the actual format and exclude trailing row/image padding.
use super::blit_holds::{device, Scratch};
use crate::{Region, ResourceOptions, TextureDescriptor};
use metal2vulkan::metal::{PixelFormat, TextureType};

#[test]
fn checked_cpu_spans_cover_formats_padding_volumes_and_empty_regions() {
    let scratch = Scratch::new("copy-span");
    let device = device(&scratch.0);
    for (format, width, height, row, expected) in [
        (PixelFormat::RGBA32Float, 4, 4, 80, 304),
        (PixelFormat::Depth32Float, 4, 4, 32, 112),
        (PixelFormat::RGBA8Unorm, 4, 4, 32, 112),
        (PixelFormat::BC1RGBA, 7, 5, 24, 40),
        (PixelFormat::BC3RGBA, 7, 5, 40, 72),
    ] {
        let mut desc = TextureDescriptor::texture_2d(format, width, height, true);
        desc.resource_options = ResourceOptions::shared();
        let texture = device.new_texture(&desc).unwrap();
        let region = Region::new_2d(0, 0, width, height);
        assert_eq!(
            texture.cpu_copy_byte_length(region, 0, 0, row, 0).unwrap(),
            expected,
            "{format:?}"
        );
        assert_eq!(
            texture
                .cpu_copy_byte_length(Region::new_2d(0, 0, 0, height), 0, 0, row, 0)
                .unwrap(),
            0
        );
        assert!(texture.cpu_copy_byte_length(region, 30, 0, row, 0).is_err());
        assert!(texture
            .cpu_copy_byte_length(region, 0, 0, u64::MAX, 0)
            .is_err());
    }
    let mut desc = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, true);
    desc.texture_type = TextureType::Type3D;
    desc.depth = 4;
    desc.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&desc).unwrap();
    let region = Region::new_3d(0, 0, 0, 2, 2, 2);
    assert_eq!(
        texture.cpu_copy_byte_length(region, 1, 0, 16, 64).unwrap(),
        88
    );
    assert!(texture
        .cpu_copy_byte_length(region, 1, 0, 16, u64::MAX)
        .is_err());
    let mut private = desc;
    private.resource_options = ResourceOptions::private();
    assert!(device
        .new_texture(&private)
        .unwrap()
        .cpu_copy_byte_length(region, 0, 0, 16, 64)
        .is_err());
}
