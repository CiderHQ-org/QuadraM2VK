//! Copies of row and column strips within one texture, of the shapes the consumer's Maps makes
//! (its diagnostic at r23: one region a call, into the same texture, `Private` and `ShaderRead`
//! only, `RGBA8Unorm` 256 x 256 and `RG8Unorm` 256 x 352; strips up to 29 texels long, most moved
//! to places they do not meet, four copied onto themselves at the origin), on the device the cfg
//! selects, under the validation layer.
//!
//! A strip copied onto itself keeps its texels and records nothing; a strip copied to a place it
//! does not meet moves its texels there. Neither overlaps, so neither is refused and neither
//! announces `overlapping-texture-copy-as-memmove`. The bytes are our own: every texel of each
//! texture holds a value no other texel of it holds.

use super::degrade_notices::counts;
use super::{validated_device, Messages, ScratchCache};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    BlitCommandEncoder, BlitOption, Buffer, CpuCacheMode, Degradation, Error, HazardTrackingMode,
    Origin, ResourceOptions, Size, StorageMode, Texture, TextureDescriptor,
};
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(60);

/// One copy: source origin, size, destination origin.
type Strip = ([u64; 2], [u64; 2], [u64; 2]);

/// The textures (format, bytes a texel, width, height) and their strips: the four copies onto
/// themselves the diagnostic lists, and strips moved to places they do not meet, some next to
/// their source.
fn cases() -> [(PixelFormat, u64, u64, u64, Vec<Strip>); 2] {
    [
        (
            PixelFormat::RGBA8Unorm,
            4,
            256,
            256,
            vec![
                ([0, 0], [29, 1], [0, 0]),
                ([0, 0], [1, 29], [0, 0]),
                ([0, 0], [9, 1], [236, 32]),
                ([0, 0], [29, 1], [100, 200]),
                ([0, 0], [1, 29], [200, 100]),
                ([0, 0], [29, 1], [29, 0]),
                ([0, 0], [29, 1], [0, 1]),
                ([0, 0], [1, 29], [1, 0]),
                ([17, 3], [9, 1], [17, 4]),
            ],
        ),
        (
            PixelFormat::RG8Unorm,
            2,
            256,
            352,
            vec![
                ([0, 0], [21, 1], [0, 0]),
                ([0, 0], [1, 25], [0, 0]),
                ([0, 0], [21, 1], [10, 300]),
                ([0, 0], [1, 25], [1, 0]),
                ([0, 0], [1, 25], [0, 25]),
                ([0, 0], [9, 1], [236, 32]),
            ],
        ),
    ]
}

/// Every strip of [`cases`] copied within its texture: each call is taken, and the texture then
/// holds its bytes with the strip moved (unchanged for a strip onto itself), without the
/// overlapping copy's announcement.
#[test]
fn maps_strip_copies_within_one_texture_are_taken_and_move_their_texels() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let mut wrong = Vec::new();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let queue = device.new_command_queue().unwrap();
        for (pf, per, width, height, strips) in cases() {
            let mut d = TextureDescriptor::texture_2d(pf, width, height, false);
            d.usage = TextureUsage::SHADER_READ;
            d.resource_options = ResourceOptions::new(
                StorageMode::Private,
                CpuCacheMode::DefaultCache,
                HazardTrackingMode::Default,
            );
            let texture = device.new_texture(&d).unwrap();
            let texels = width * height;
            let start: Vec<u8> = (0..texels)
                .flat_map(|i| {
                    let v = (i as u32).wrapping_mul(2_654_435_761) ^ 0x5a5a_5a5a;
                    v.to_le_bytes()[..per as usize].to_vec()
                })
                .collect();
            // Each texel's value: unique in the RGBA8 texture; in the RG8 one, whose two bytes
            // hold fewer values than its texels, texel `i` and `i + 65536` alike.
            let row = (width * per) as usize;
            let upload = device
                .new_buffer(start.len() as u64, ResourceOptions::shared())
                .unwrap();
            // SAFETY: the buffer holds `start.len()` bytes; no command buffer uses it yet.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    start.as_ptr(),
                    upload.contents().unwrap().as_ptr(),
                    start.len(),
                );
            }
            for (at, size, into) in strips {
                let out = device
                    .new_buffer(start.len() as u64, ResourceOptions::shared())
                    .unwrap();
                let command_buffer = queue.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                let recorded = strip(
                    &mut blit,
                    (&upload, &texture, &out),
                    [width, height, row as u64],
                    (at, size, into),
                );
                blit.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.error(), None);
                let what = format!(
                    "{pf:?} {}x{} at ({}, {}) to ({}, {})",
                    size[0], size[1], at[0], at[1], into[0], into[1]
                );
                if let Err(e) = recorded {
                    eprintln!("MAPS-STRIP {what} on {name}: refused {e:?}");
                    wrong.push(format!("{what}: refused {e:?}"));
                    continue;
                }
                let mut want = start.clone();
                for y in 0..size[1] {
                    for x in 0..size[0] {
                        let from = ((at[1] + y) * width + at[0] + x) * per;
                        let to = ((into[1] + y) * width + into[0] + x) * per;
                        let (from, to, n) = (from as usize, to as usize, per as usize);
                        want[to..to + n].copy_from_slice(&start[from..from + n]);
                    }
                }
                // SAFETY: the command buffer completed; the buffer holds `start.len()` bytes.
                let got = unsafe {
                    std::slice::from_raw_parts(out.contents().unwrap().as_ptr(), start.len())
                };
                let differ = got
                    .chunks_exact(per as usize)
                    .zip(want.chunks_exact(per as usize))
                    .filter(|(g, w)| g != w)
                    .count();
                eprintln!("MAPS-STRIP {what} on {name}: taken, {differ} texels off");
                if differ > 0 {
                    wrong.push(format!("{what}: {differ} texels off"));
                }
            }
        }
    }
    let notices = counts(&messages.take_notices());
    eprintln!("MAPS-STRIP notices {notices:?}");
    assert!(
        !notices.contains_key(&Degradation::OverlappingTextureCopyAsMemmove),
        "{notices:?}"
    );
    assert!(wrong.is_empty(), "strips not taken as Metal's: {wrong:#?}");
}

/// Fill `texture` (`width` x `height`, `row` bytes a row) from `upload`, copy one strip within
/// it, and read it whole into `out`.
fn strip(
    blit: &mut BlitCommandEncoder,
    (upload, texture, out): (&Buffer, &Texture, &Buffer),
    [width, height, row]: [u64; 3],
    (at, size, into): Strip,
) -> Result<(), Error> {
    blit.copy_from_buffer_to_texture(
        upload,
        0,
        row,
        0,
        Size::new(width, height, 1),
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    )?;
    blit.copy_from_texture_to_texture(
        texture,
        0,
        0,
        Origin::new(at[0], at[1], 0),
        Size::new(size[0], size[1], 1),
        texture,
        0,
        0,
        Origin::new(into[0], into[1], 0),
    )?;
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(width, height, 1),
        out,
        0,
        row,
        0,
        BlitOption::None,
    )
}
