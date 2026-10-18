//! A blit copy between a buffer and a multisampled texture is emulated per sample
//! (`sample_interleave.rs`), or refused typed where the device cannot run the emulation.
//!
//! Vulkan copies no multisampled image to or from a buffer; Metal copies it per sample, the
//! pixel's samples consecutive in the buffer (texel `(x, y)` sample `s` at
//! `offset + y * bytesPerRow + (x * n + s) * bpp`, on an M4). The runtime moves
//! the samples through a wide image and a multisampled uint scratch. This test fills a buffer
//! with a distinct pattern laid out per sample, copies it into a multisampled uint texture and
//! back out into a second buffer, and checks the two buffers' covered bytes match: both
//! directions are exact inverses implementing the same `(x * n + s)` layout, bit for bit, which
//! with Metal's measured layout (the modules compute exactly that) verifies the emulation. On a
//! device that cannot run it the copy returns `NotImplemented(MultisampledBufferCopy)`, and the
//! case is skipped.
//!
//! Each case prints one `CASE` line: the format, the sample count and whether the device answers
//! `supportsTextureSampleCount:` yes for it, the region, and what happened (`RAN`, `REFUSED`, or
//! `NO-TEXTURE` where the device makes no such texture). The device lanes fail on a `REFUSED` or
//! `NO-TEXTURE` case at a supported count, so a lane cannot pass on skipped cases unnoticed.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::error::{Error, RuntimeGap};
use crate::{
    BlitOption, Buffer, CommandBufferStatus, Device, DeviceKind, DeviceOptions, Origin,
    ResourceOptions, ShaderCacheOptions, Size, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// The longest a copy's command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        let path = std::env::temp_dir().join(format!(
            "m2v-multisample-buffer-copy-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU, its shader cache
/// in `cache`.
fn device(cache: &Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// A `Shared` buffer of `len` bytes filled with `fill`.
fn buffer(device: &Device, len: u64, fill: impl Fn(usize) -> u8) -> Buffer {
    let bytes: Vec<u8> = (0..len as usize).map(fill).collect();
    device
        .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
        .unwrap()
}

/// The first `len` bytes of a buffer read back.
fn read(buffer: &Buffer, len: usize) -> Vec<u8> {
    let contents = buffer.contents().expect("a shared buffer");
    // SAFETY: the buffer holds at least `len` bytes at `contents`, alive with it; no command
    // buffer uses it now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), len) }.to_vec()
}

/// What one case did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    /// Both copies ran and the round trip was bit-exact.
    Ran,
    /// The device refused the copy (`MultisampledBufferCopy`).
    Refused,
    /// The device makes no texture of the format at the sample count.
    NoTexture,
}

/// One copy round trip of a `width` by `height` region of a `samples`-sample texture of
/// `(pixel_format, bpp)` at `origin`, with a tight `bytesPerRow`: fill a buffer per sample, copy
/// it in, copy it back out, and check the covered bytes match. Prints the case's `CASE` line.
fn round_trip(
    device: &Device,
    (pixel_format, bpp): (PixelFormat, u64),
    samples: u64,
    (width, height): (u64, u64),
    origin: Origin,
) -> Outcome {
    let supported = match device.properties().supports_texture_sample_count(samples) {
        Ok(true) => "supported",
        Ok(false) | Err(_) => "unsupported",
    };
    let outcome = copy_round_trip(
        device,
        (pixel_format, bpp),
        samples,
        (width, height),
        origin,
    );
    println!(
        "CASE {pixel_format:?} n={samples} ({supported}) {width}x{height}@({},{}): {}",
        origin.x,
        origin.y,
        match outcome {
            Outcome::Ran => "RAN",
            Outcome::Refused => "REFUSED",
            Outcome::NoTexture => "NO-TEXTURE",
        }
    );
    outcome
}

/// [`round_trip`]'s copies and check.
fn copy_round_trip(
    device: &Device,
    (pixel_format, bpp): (PixelFormat, u64),
    samples: u64,
    (width, height): (u64, u64),
    origin: Origin,
) -> Outcome {
    let mut descriptor =
        TextureDescriptor::texture_2d(pixel_format, origin.x + width, origin.y + height, false);
    descriptor.texture_type = TextureType::Type2DMultisample;
    descriptor.sample_count = samples;
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    let texture = match device.new_texture(&descriptor) {
        Ok(texture) => texture,
        Err(_) => return Outcome::NoTexture,
    };
    let row = width * samples * bpp;
    let len = row * height;
    let source = buffer(device, len, |i| ((i * 37 + 11) & 0xff) as u8);
    let out = buffer(device, len, |_| 0);
    let queue = device.new_command_queue().unwrap();

    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let into = blit.copy_from_buffer_to_texture(
        &source,
        0,
        row,
        len,
        Size::new(width, height, 1),
        &texture,
        0,
        0,
        origin,
        BlitOption::None,
    );
    match into {
        Ok(()) => {}
        Err(Error::NotImplemented(RuntimeGap::MultisampledBufferCopy)) => return Outcome::Refused,
        Err(other) => panic!("{pixel_format:?} {samples} into: {other}"),
    }
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);

    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &texture,
        0,
        0,
        origin,
        Size::new(width, height, 1),
        &out,
        0,
        row,
        len,
        BlitOption::None,
    )
    .expect("the reverse copy runs once the forward one did");
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);

    let (a, b) = (read(&source, len as usize), read(&out, len as usize));
    assert_eq!(
        a, b,
        "{pixel_format:?} {samples} {width}x{height}@{origin:?}: round trip not bit-exact"
    );
    Outcome::Ran
}

/// Each single-texel uint format of one size (so the round trip is bit-exact), at 2, 4 and 8
/// samples (Metal makes 8 on no Apple GPU, so a device answers no and the case prints
/// `NO-TEXTURE`), for a whole copy and a sub-region, the counts in turn on one device (so a
/// module made for one count and reused for another shows), and a texture 16384 wide at 4
/// samples, whose `width * samples` exceeds the largest side the runtime makes its wide image
/// with (Metal's 16384, or a smaller device 2D limit), so the wide image bands, round-trips
/// through the multisampled texture where the device runs the emulation.
#[test]
fn a_blit_between_a_buffer_and_a_multisampled_texture_round_trips_per_sample() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let formats = [
        (PixelFormat::R8Uint, 1u64),
        (PixelFormat::R16Uint, 2),
        (PixelFormat::R32Uint, 4),
        (PixelFormat::RG32Uint, 8),
        (PixelFormat::RGBA32Uint, 16),
    ];
    let mut outcomes = Vec::new();
    for (pf, bpp) in formats {
        for samples in [2u64, 4, 8] {
            outcomes.push(round_trip(
                &device,
                (pf, bpp),
                samples,
                (8, 8),
                Origin::new(0, 0, 0),
            ));
            outcomes.push(round_trip(
                &device,
                (pf, bpp),
                samples,
                (4, 3),
                Origin::new(2, 1, 0),
            ));
        }
    }
    // Wide textures whose `width * samples` (16384 * 4) exceeds the wide image's largest side,
    // so it bands, at the smallest and the largest texel; two rows and one keep memory small.
    outcomes.push(round_trip(
        &device,
        (PixelFormat::R8Uint, 1),
        4,
        (16384, 2),
        Origin::new(0, 0, 0),
    ));
    outcomes.push(round_trip(
        &device,
        (PixelFormat::RGBA32Uint, 16),
        4,
        (16384, 1),
        Origin::new(0, 0, 0),
    ));
    let ran = outcomes.iter().filter(|o| **o == Outcome::Ran).count();
    println!("CASES ran {ran} of {}", outcomes.len());
    // The emulation runs on lavapipe and the GPUs this test's venues use; if a device ran none,
    // the test proved nothing, so fail rather than pass vacuously.
    assert!(ran > 0, "the device ran no multisampled buffer copy");
}
