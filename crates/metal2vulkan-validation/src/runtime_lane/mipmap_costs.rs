//! The GPU cost of `generate_mipmaps` on the device the cfg selects, without the validation layer:
//! report only, in release builds, to weigh the path the runtime takes for a format (Metal's filter
//! computed by the runtime's kernel, or the device's blit chain) before and after a change to it.
//!
//! Each case times one command buffer holding one `generate_mipmaps` of a private texture, by the
//! command buffer's GPU start and end times, by the host's wall time from its commit until its
//! wait returns, and the call's encoding on the host, [`RUNS`] times after one untimed run (which
//! pays for the format's first-use check or the kernel's pipeline). The wall time bounds the GPU's
//! from above on a device whose timestamps are not the command buffer's whole work (MoltenVK's on
//! an Apple GPU gave a 4096x4096 `RGBA16Float` level in 6.4 us, under the memory's bandwidth).
//! The cases: `RGBA8Unorm` and `RGBA16Float`, a 4096x4096 texture's whole chain, and two-level
//! textures from 4096 down to 64 (one level's cost at each size). Printed per case: the median and
//! the fastest GPU time, the median and the fastest wall time, the median encoding time, and the
//! notices the case announced (each kind once per device, so at
//! its first case: `mipmaps-within-spread` for the blit chain, `mipmaps-as-apple-m4` for the
//! filter where an M3 Max's answer differs). A case the device refuses (for example a format
//! whose blit chain failed the runtime's check, on a device that runs no filter for it) is
//! printed with its refusal and skipped.
//!
//! Beside each case, the parts of the filter's command that move bytes, each timed alone the same
//! way in a command buffer of its own (the GPU-side follow-up): the first level copied into a
//! buffer, the later levels copied back from one, and a fill of the later levels' bytes. What the
//! filter's GPU time has beyond their sum is its dispatches'.
//!
//! Ignored by default: a measurement, not a check, and slow on a CPU device. Run it by name with
//! `--run-ignored all`.

use super::degrade_notices::counts;
use super::{validated_device_with, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    BlitCommandEncoder, BlitOption, Buffer, CommandQueue, Error, Origin, ResourceOptions, Size,
    Texture, TextureDescriptor,
};
use std::time::Instant;

/// The timed runs of each case.
const RUNS: usize = 9;

/// A private square texture of `pf`, `size` texels a side, of `levels` levels.
fn texture(
    device: &metal2vulkan_runtime::Device,
    pf: PixelFormat,
    size: u64,
    levels: u64,
) -> Texture {
    let mut d = TextureDescriptor::texture_2d(pf, size, size, false);
    d.mipmap_level_count = levels;
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    device.new_texture(&d).expect("a texture")
}

/// One timed run's GPU, wall and encoding times, in microseconds.
struct Times {
    gpu: f64,
    wall: f64,
    encoding: f64,
}

/// The times of one command buffer of `queue` generating `t`'s mipmaps; the refusal where the
/// device refuses them.
fn generate(queue: &CommandQueue, t: &Texture) -> Result<Times, Error> {
    timed(queue, |blit| blit.generate_mipmaps(t))
}

/// The times of one command buffer of `queue` holding the blit commands `encode` records; the
/// refusal where the device refuses them.
fn timed(
    queue: &CommandQueue,
    encode: impl FnOnce(&mut BlitCommandEncoder) -> Result<(), Error>,
) -> Result<Times, Error> {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let encoding = Instant::now();
    let generated = encode(&mut blit);
    let encoded = encoding.elapsed().as_secs_f64() * 1e6;
    blit.end_encoding().unwrap();
    generated?;
    let committed = Instant::now();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let wall = committed.elapsed().as_secs_f64() * 1e6;
    assert_eq!(command_buffer.error(), None);
    let start = command_buffer.gpu_start_time().unwrap();
    let end = command_buffer.gpu_end_time().unwrap();
    Ok(Times {
        gpu: (end - start) * 1e6,
        wall,
        encoding: encoded,
    })
}

/// The median GPU time of [`RUNS`] runs of `encode` after one untimed run.
fn gpu_median(
    queue: &CommandQueue,
    mut encode: impl FnMut(&mut BlitCommandEncoder) -> Result<(), Error>,
) -> f64 {
    timed(queue, &mut encode).expect("a part of the filter's command");
    let runs = (0..RUNS)
        .map(|_| timed(queue, &mut encode).expect("a part, made once").gpu)
        .collect();
    median_and_least(runs).0
}

/// The parts of the filter's command for `t` (of `levels` levels, `size` texels a side, `bytes`
/// a texel) that move bytes, each one's median GPU time (the module docs): the first level
/// copied into `scratch`, the later levels copied back from it, and a fill of their bytes.
fn parts(
    queue: &CommandQueue,
    t: &Texture,
    scratch: &Buffer,
    (size, levels, bytes): (u64, u64, u64),
) -> [f64; 3] {
    let extent = |level: u64| (size >> level).max(1);
    let level_bytes = |level: u64| extent(level) * extent(level) * bytes;
    let later: u64 = (1..levels).map(level_bytes).sum();
    let copy_in = gpu_median(queue, |blit| {
        blit.copy_from_texture_to_buffer(
            t,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(size, size, 1),
            scratch,
            0,
            size * bytes,
            size * size * bytes,
            BlitOption::None,
        )
    });
    let copies_back = gpu_median(queue, |blit| {
        let mut at = 0;
        for level in 1..levels {
            let e = extent(level);
            blit.copy_from_buffer_to_texture(
                scratch,
                at,
                e * bytes,
                e * e * bytes,
                Size::new(e, e, 1),
                t,
                0,
                level,
                Origin::new(0, 0, 0),
                BlitOption::None,
            )?;
            at += level_bytes(level);
        }
        Ok(())
    });
    let fill = gpu_median(queue, |blit| blit.fill_buffer(scratch, 0..later, 0));
    [copy_in, copies_back, fill]
}

/// The median and the smallest of `values`.
fn median_and_least(mut values: Vec<f64>) -> (f64, f64) {
    values.sort_by(f64::total_cmp);
    (values[values.len() / 2], values[0])
}

/// Every case's GPU times, printed (the module docs).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn generate_mipmaps_gpu_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
    })
    .unwrap();
    messages.take_notices();
    let queue = device.new_command_queue().unwrap();
    let mut cases: Vec<(u64, u64)> = vec![(4096, 13)];
    cases.extend((6..=12).rev().map(|log| (1u64 << log, 2)));
    for pf in [PixelFormat::RGBA8Unorm, PixelFormat::RGBA16Float] {
        for &(size, levels) in &cases {
            let t = texture(&device, pf, size, levels);
            if let Err(refusal) = generate(&queue, &t) {
                eprintln!(
                    "mipmap cost on {}: {} {size}x{size} levels {levels}: refused, {refusal:?}",
                    device.name(),
                    pf.name(),
                );
                continue;
            }
            let runs: Vec<Times> = (0..RUNS)
                .map(|_| generate(&queue, &t).expect("the mipmaps, made once"))
                .collect();
            let (gpu, gpu_least) = median_and_least(runs.iter().map(|r| r.gpu).collect());
            let (wall, wall_least) = median_and_least(runs.iter().map(|r| r.wall).collect());
            let (encoding, _) = median_and_least(runs.iter().map(|r| r.encoding).collect());
            let notices = counts(&messages.take_notices());
            eprintln!(
                "mipmap cost on {}: {} {size}x{size} levels {levels}: GPU median {gpu:.1} us, \
                 fastest {gpu_least:.1} us; wall median {wall:.1} us, fastest {wall_least:.1} us; \
                 encoding median {encoding:.1} us; announced {notices:?}",
                device.name(),
                pf.name(),
            );
            let bytes = if pf == PixelFormat::RGBA8Unorm { 4 } else { 8 };
            let scratch = device
                .new_buffer(size * size * bytes, ResourceOptions::private())
                .expect("a buffer of the first level's bytes");
            let [copy_in, copies_back, fill] = parts(&queue, &t, &scratch, (size, levels, bytes));
            eprintln!(
                "mipmap cost parts on {}: {} {size}x{size} levels {levels}: GPU medians: copy in \
                 {copy_in:.1} us, copies back {copies_back:.1} us, fill {fill:.1} us; the rest \
                 {:.1} us",
                device.name(),
                pf.name(),
                gpu - copy_in - copies_back - fill,
            );
        }
    }
}
