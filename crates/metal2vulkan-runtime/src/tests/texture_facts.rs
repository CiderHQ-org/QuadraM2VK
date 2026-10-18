//! A texture's binding facts (`Texture::facts`), worked out at the first read and kept: the kept
//! facts are the ones worked out again from what the texture was made with, after a view of it
//! was made, with a depth texture's twin, and a view's are its own. (Whether a storage binding
//! takes the usage shadow is a test of the image's and the texture's usage, made with them, not
//! of the shadow, which is made later.)
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::{Device, DeviceKind, DeviceOptions, ResourceOptions, Texture, TextureDescriptor};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU.
fn device() -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// A private 2D texture of `format`, 16 by 16, of `usage`.
fn texture(device: &Device, format: PixelFormat, usage: TextureUsage) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, 16, 16, false);
    descriptor.usage = usage;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// That `texture`'s kept facts are the ones worked out again now.
fn kept_as_worked_out(texture: &Texture, what: &str) {
    assert_eq!(
        texture.facts(),
        texture.worked_out_facts(),
        "{what}: the kept facts are not the ones worked out again"
    );
}

#[test]
fn a_texture_s_kept_facts_are_the_ones_worked_out_again_and_a_view_s_its_own() {
    let device = device();
    let read_write = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    let color = texture(
        &device,
        PixelFormat::RGBA8Unorm,
        read_write | TextureUsage::PIXEL_FORMAT_VIEW,
    );
    let first = color.facts();
    // A view made after the texture's facts were kept has its own, of its own format.
    let view = color.new_texture_view(PixelFormat::RGBA8Uint).unwrap();
    assert_eq!(view.facts().pixel_format, PixelFormat::RGBA8Uint);
    assert_eq!(first.pixel_format, PixelFormat::RGBA8Unorm);
    kept_as_worked_out(&view, "a view made after its texture's facts were kept");
    kept_as_worked_out(&color, "a texture after a view of it was made");
    assert_eq!(color.facts(), first, "kept once made");
    // A depth texture's twin is made with its image: its facts say so from the first read.
    let depth = texture(
        &device,
        PixelFormat::Depth32Float,
        TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
    );
    let facts = depth.facts();
    assert_eq!(facts.depth_twin, depth.twin_image().is_some());
    kept_as_worked_out(&depth, "a depth texture");
}
