//! Native BC storage avoids the decoded twin; the opt-in exact decoder stays available.
use crate::{Device, DeviceKind, DeviceOptions, Region, ResourceOptions, ShaderCacheOptions, TextureDescriptor};
use metal2vulkan::device::{BlockDecodePolicy, DriverFact};
use metal2vulkan::metal::PixelFormat;
use super::blit_holds::Scratch;

#[test]
fn native_bc_storage_keeps_no_twin_and_exact_keeps_the_measured_decode() {
    for (name, policy) in [("native", BlockDecodePolicy::NativeDevice), ("exact", BlockDecodePolicy::Exact)] {
        let scratch = Scratch::new(name);
        let mut options = DeviceOptions::default();
        options.kinds = if cfg!(m2v_cpu_icd) { vec![DeviceKind::Cpu] } else { vec![DeviceKind::Discrete, DeviceKind::Integrated, DeviceKind::Virtual] };
        options.shader_cache = ShaderCacheOptions { directory: Some(scratch.0.clone()), ..ShaderCacheOptions::default() };
        options.policy.block_decode = policy;
        let device = Device::system_default_with(&options).unwrap();
        let before = device.work_counts();
        for (format, block_bytes, fact) in [
            (PixelFormat::BC1RGBA, 8, DriverFact::ColorBlockDecodeIsMetals),
            (PixelFormat::BC3RGBA, 16, DriverFact::ColorBlockDecodeIsMetals),
            (PixelFormat::BC5RGUnorm, 16, DriverFact::ChannelBlockDecodeIsMetals),
        ] {
            let mut desc = TextureDescriptor::texture_2d(format, 8, 8, false);
            desc.resource_options = ResourceOptions::shared();
            let texture = device.new_texture(&desc).unwrap();
            let twin = policy == BlockDecodePolicy::Exact && !device.properties().driver_facts.contains(&fact);
            assert_eq!(texture.has_decoded_twin(), twin, "{name}: {format:?}");
            let blocks = vec![0x37; block_bytes * 4];
            let region = Region::new_2d(0, 0, 8, 8);
            let row = (block_bytes * 2) as u64;
            texture.replace_region(region, 0, 0, &blocks, row, 0).unwrap();
            let mut read = vec![0; blocks.len()];
            texture.get_bytes(&mut read, row, 0, region, 0, 0).unwrap();
            assert_eq!(blocks, read);
        }
        if policy == BlockDecodePolicy::NativeDevice {
            assert_eq!(device.work_counts().block_decode_passes, before.block_decode_passes);
            assert_eq!(device.work_counts().cpu_block_decodes, before.cpu_block_decodes);
        }
    }
}
