//! A game's terrain material array: 36 slices of 512x512 with a full chain of 10 levels, made and
//! filled one slice and level at a time (the way Roblox does after "Terrain TextureArray ...
//! arraySize: 36, mip: 10"), by the CPU into a `Managed` or `Shared` texture. Prints what each step
//! costs and how much host memory the texture takes, per format and storage mode.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch};
use crate::{
    CpuCacheMode, HazardTrackingMode, Region, ResourceOptions, StorageMode, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use std::time::Instant;

fn level_bytes(format: PixelFormat, width: u64, height: u64) -> (u64, u64) {
    let (block, bytes) = match format {
        PixelFormat::BC1RGBA => (4, 8),
        PixelFormat::BC3RGBA => (4, 16),
        _ => (1, 4),
    };
    let row = width.div_ceil(block) * bytes;
    (row, row * height.div_ceil(block))
}

#[test]
#[ignore = "makes a 36-slice texture per format and storage mode and prints timings; run it on purpose"]
fn a_terrain_texture_array_reports_what_filling_it_costs() {
    let scratch = Scratch::new("terrain-array");
    let device = device(&scratch.0);
    for format in [
        PixelFormat::RGBA8Unorm,
        PixelFormat::BC3RGBA,
        PixelFormat::BC1RGBA,
    ] {
        for storage in [
            StorageMode::Managed,
            StorageMode::Shared,
            StorageMode::Private,
        ] {
            let mut desc = TextureDescriptor::texture_2d(format, 512, 512, true);
            desc.texture_type = TextureType::Type2DArray;
            desc.array_length = 36;
            desc.usage = TextureUsage::SHADER_READ;
            desc.resource_options = ResourceOptions::new(
                storage,
                CpuCacheMode::DefaultCache,
                HazardTrackingMode::Default,
            );
            let before = device.memory_statistics();
            let start = Instant::now();
            let texture = match device.new_texture(&desc) {
                Ok(texture) => texture,
                Err(error) => {
                    eprintln!("TERRAIN: {format:?} {storage:?}: refused: {error}");
                    continue;
                }
            };
            let created = start.elapsed().as_secs_f64() * 1000.0;
            let after = device.memory_statistics();
            let (mut uploads, mut worst, mut bytes_in) = (0u32, 0.0f64, 0u64);
            let start = Instant::now();
            if storage != StorageMode::Private {
                for slice in 0..36 {
                    for level in 0..desc.mipmap_level_count {
                        let (width, height) = ((512u64 >> level).max(1), (512u64 >> level).max(1));
                        let (row, size) = level_bytes(format, width, height);
                        let data = vec![0x5au8; size as usize];
                        let one = Instant::now();
                        texture
                            .replace_region(
                                Region::new_2d(0, 0, width, height),
                                level,
                                slice,
                                &data,
                                row,
                                0,
                            )
                            .unwrap();
                        worst = worst.max(one.elapsed().as_secs_f64() * 1000.0);
                        uploads += 1;
                        bytes_in += size;
                    }
                }
            }
            let filled = start.elapsed().as_secs_f64() * 1000.0;
            let end = device.memory_statistics();
            eprintln!(
                "TERRAIN: {format:?} {storage:?}: create {created:.1} ms (+{} MiB live, +{} MiB device-local); {uploads} uploads of {} MiB in {filled:.0} ms (worst {worst:.1} ms); live now +{} MiB, device-local +{} MiB",
                (after.live_bytes - before.live_bytes) >> 20,
                (after.device_local_live_bytes - before.device_local_live_bytes) >> 20,
                bytes_in >> 20,
                (end.live_bytes as i64 - before.live_bytes as i64) >> 20,
                (end.device_local_live_bytes as i64 - before.device_local_live_bytes as i64) >> 20,
            );
            drop(texture);
        }
    }
}
