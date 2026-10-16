//! `Shared` and `Managed` textures whose CPU copies run on the host where the device copies there
//! (`runtime/src/texture/host_copy.rs`), read and written by the GPU between those copies, on the
//! device the cfg selects, under the validation layer: every texel `replace_region` writes reaches
//! the GPU's next read, and every texel the GPU writes reaches `get_bytes`. On ANV's host-cached
//! memory type without coherence (Mesa 26.2.4 on Arrow Lake-S's Xe-LPG), where such images lay
//! before, the GPU read stale 64-byte lines of them (4x4 texels of a Tile4 32-bit image;
//! `runtime/src/memory.rs`, `host_cached`).
//!
//! A blit's command buffer that fails says why, with Metal's description of the failure
//! (`CommandBuffer::error_description`). On the M3 Max this lane failed only beside
//! `host_memory`'s lanes that fault the GPU on purpose, whose recovery fails another process's
//! command buffers in flight (`Execution(1)`, "Discarded (victim of GPU error/recovery)"), so those
//! lanes run alone (`.config/nextest.toml`).

use super::{buffer_with, read_buffer, WAIT};
use crate::runtime_lane::host_memory::on_device;
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::{
    BlitOption, CommandQueue, CpuCacheMode, Device, HazardTrackingMode, Origin, Region,
    ResourceOptions, Size, StorageMode, Texture, TextureDescriptor,
};

/// The sides of the square `RGBA8Unorm` textures each round makes: one smaller than a 4-KiB
/// tile's row of texels, and two spanning many tiles.
const SIDES: [u64; 3] = [16, 256, 1024];

/// The rounds of copies each storage mode and side runs, each on a new texture.
const ROUNDS: u8 = 4;

/// `n` bytes of a pattern seeded by `seed`, no texel of it zero.
fn pattern(n: usize, seed: u8) -> Vec<u8> {
    (0..n)
        .map(|i| ((i * 29 + usize::from(seed) * 7) % 251) as u8 | 1)
        .collect()
}

/// The GPU's copy of `texture`'s `side` x `side` texels, through a blit into a new buffer.
fn gpu_read(device: &Device, queue: &CommandQueue, texture: &Texture, side: u64) -> Vec<u8> {
    let len = side * side * 4;
    let out = buffer_with(device, &vec![0xee; len as usize]).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    (blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(side, side, 1),
        &out,
        0,
        side * 4,
        len,
        BlitOption::None,
    ))
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.error(),
        None,
        "the blit's command buffer failed: {:?}",
        command_buffer.error_description()
    );
    read_buffer(&out, 0, len).unwrap()
}

/// The GPU's write of `bytes` over `texture`'s `side` x `side` texels, through a blit.
fn gpu_write(device: &Device, queue: &CommandQueue, texture: &Texture, side: u64, bytes: &[u8]) {
    let source = buffer_with(device, bytes).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    (blit.copy_from_buffer_to_texture(
        &source,
        0,
        side * 4,
        side * side * 4,
        Size::new(side, side, 1),
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    ))
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.error(),
        None,
        "the blit's command buffer failed: {:?}",
        command_buffer.error_description()
    );
}

/// The texels of `got` that differ from `want`'s, of a `side` x `side` 4-byte texel image: their
/// count and the first few, as `(x, y)`, else `None`.
fn differing(got: &[u8], want: &[u8], side: u64) -> Option<String> {
    let texels: Vec<u64> = (got.chunks(4).zip(want.chunks(4)).enumerate())
        .filter(|(_, (got, want))| got != want)
        .map(|(index, _)| index as u64)
        .collect();
    let first = texels.iter().take(4).map(|&t| (t % side, t / side));
    (!texels.is_empty()).then(|| {
        format!(
            "{} texels, from {:?}",
            texels.len(),
            first.collect::<Vec<_>>()
        )
    })
}

/// Each round's new texture is read by the GPU while it holds the zeros it was made with, then
/// takes a `replace_region` of every texel the GPU reads back, a GPU write `get_bytes` reads back,
/// and a `replace_region` of all but a 1-texel border (each 64-byte line of a tiled image along
/// the border written in part) that both read back over the GPU's border: every texel as written
/// last, in `Shared` and `Managed` textures of three sizes.
#[test]
fn host_copies_and_gpu_copies_see_each_others_texels() {
    on_device(|device| {
        let name = device.name();
        let queue = device.new_command_queue().unwrap();
        let managed = ResourceOptions::new(
            StorageMode::Managed,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        );
        let mut failures = Vec::new();
        let mut hosted = 0;
        for options in [ResourceOptions::shared(), managed] {
            let mode = options.storage_mode().unwrap();
            for side in SIDES {
                let len = (side * side * 4) as usize;
                let row = side * 4;
                let whole = Region::new_2d(0, 0, side, side);
                for round in 0..ROUNDS {
                    let seed = round * 8;
                    let mut descriptor =
                        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, side, side, false);
                    descriptor.resource_options = options;
                    let before = device.work_counts();
                    let texture = device.new_texture(&descriptor).unwrap();
                    let made = device.work_counts();
                    hosted += made.images_started_on_the_host - before.images_started_on_the_host;
                    let mut check = |step: &str, got: &[u8], want: &[u8]| {
                        if let Some(differ) = differing(got, want, side) {
                            failures.push(format!(
                                "{mode:?} {side}x{side} round {round}, {step}: {differ}"
                            ));
                        }
                    };

                    check(
                        "the GPU's read of the zeros",
                        &gpu_read(device, &queue, &texture, side),
                        &vec![0; len],
                    );

                    let written = pattern(len, seed + 1);
                    texture
                        .replace_region(whole, 0, 0, &written, row, 0)
                        .unwrap();
                    check(
                        "the GPU's read of replace_region",
                        &gpu_read(device, &queue, &texture, side),
                        &written,
                    );

                    let drawn = pattern(len, seed + 2);
                    gpu_write(device, &queue, &texture, side, &drawn);
                    let mut read = vec![0; len];
                    texture.get_bytes(&mut read, row, 0, whole, 0, 0).unwrap();
                    check("get_bytes of the GPU's write", &read, &drawn);

                    let inner = Region::new_2d(1, 1, side - 2, side - 2);
                    let inner_row = (side - 2) * 4;
                    let patch = pattern(((side - 2) * inner_row) as usize, seed + 3);
                    texture
                        .replace_region(inner, 0, 0, &patch, inner_row, 0)
                        .unwrap();
                    let mut want = drawn.clone();
                    for (y, line) in patch.chunks(inner_row as usize).enumerate() {
                        let at = ((y as u64 + 1) * row + 4) as usize;
                        want[at..at + line.len()].copy_from_slice(line);
                    }
                    check(
                        "the GPU's read of an inner replace_region",
                        &gpu_read(device, &queue, &texture, side),
                        &want,
                    );
                    let mut read = vec![0; len];
                    texture.get_bytes(&mut read, row, 0, whole, 0, 0).unwrap();
                    check("get_bytes of an inner replace_region", &read, &want);
                }
            }
        }
        eprintln!("HOST-COPIED-TEXTURES on {name}: {hosted} of the textures copied on the host");
        assert!(failures.is_empty(), "on {name}:\n{}", failures.join("\n"));
    });
}
