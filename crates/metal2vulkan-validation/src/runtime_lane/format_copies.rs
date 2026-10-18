//! Blit copies between textures of different formats and within one texture on the device the cfg
//! selects, under the validation layer, against what Metal probes measured on an M3 Max (macOS
//! 26.7) and an M4 (macOS 27.0.1), alike without API validation:
//!
//! - Between a depth-only or stencil-only format, or the stencil view `X32Stencil8` of a
//!   `Depth32FloatStencil8` texture, and a colour format of its copy unit (2 bytes for
//!   `Depth16Unorm`, 4 for `Depth32Float`, 1 for stencil), and between stencil formats, Metal
//!   copies the bytes. The runtime copies each such pair through a scratch buffer of the encoder.
//! - A buffer copy through the stencil view reads or writes the stencil, a byte a texel, with any
//!   blit option; the depth stays. A `Depth32FloatStencil8` texture copies into another, both
//!   aspects.
//! - Between a compressed format and an uncompressed one whose texel is its block's bytes, Metal
//!   copies each block into one texel and back, the copy's origins and size in the destination's
//!   texels (our probe on an Apple M3 Max under macOS 27.0.1;
//!   [`compressed_blocks_copy_into_texels_of_their_bytes_and_back`]).
//! - Formats of different units, and a combined format with another, are Metal's to refuse
//!   (`Invalid`, `CopyProblem::Format`).
//! - A region copied onto itself keeps its texels; a copy between other overlapping regions of
//!   one texture gives `memmove`'s answer, Metal's for small regions (its copy races past a few
//!   hundred texels), announcing `overlapping-texture-copy-as-memmove` once.
//!
//! Every texture is `Private` and read back through a buffer, as the probes read theirs. Depth
//! values lie in `[0, 1]` wherever a depth image holds them, so no device's clamping decides
//! (`texture_blits` measures that).

use super::degrade_notices::assert_once;
use super::{validated_device, Messages, ScratchCache};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    BlitCommandEncoder, BlitOption, Buffer, CommandQueue, CopyProblem, CpuCacheMode, Degradation,
    Device, Error, HazardTrackingMode, Invalid, Origin, Problem, ResourceOptions, Size,
    StorageMode, Texture, TextureDescriptor,
};
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(60);

/// The kind [`a_copy_onto_its_own_texels_keeps_them_and_another_overlap_moves_as_memmove`] takes
/// once: a copy between overlapping regions of one texture, given `memmove`'s answer.
pub(super) const MEMMOVES: [Degradation; 1] = [Degradation::OverlappingTextureCopyAsMemmove];

/// The textures' width; each is one row.
const W: u64 = 4;

/// Four texels of each size, as the probes wrote them: `Depth32Float` values in `[0, 1]`, 16-bit
/// and 8-bit words with their top bits set and clear.
const FOUR: [u8; 16] = [
    0x00, 0x00, 0x80, 0x3e, 0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x40, 0x3f, 0x00, 0x00, 0x80, 0x3f,
];
const TWO: [u8; 8] = [0x34, 0x12, 0xff, 0xff, 0x01, 0x00, 0x00, 0x80];
const ONE: [u8; 4] = [0x5a, 0xff, 0x01, 0x80];

/// The device's textures and the encoder that copies between them.
struct Lane {
    device: Device,
    queue: CommandQueue,
}

impl Lane {
    /// A `Private` one-row texture of `pf`, `usage`.
    fn texture(&self, pf: PixelFormat, usage: TextureUsage) -> Texture {
        let mut d = TextureDescriptor::texture_2d(pf, W, 1, false);
        d.usage = usage;
        d.resource_options = ResourceOptions::new(
            StorageMode::Private,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        );
        self.device.new_texture(&d).unwrap()
    }

    /// A Shared buffer holding `bytes`.
    fn buffer(&self, bytes: &[u8]) -> Buffer {
        let buffer = self
            .device
            .new_buffer(bytes.len().max(16) as u64, ResourceOptions::shared())
            .unwrap();
        // SAFETY: the buffer holds at least `bytes.len()` bytes at its contents, and no command
        // buffer uses it yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buffer.contents().unwrap().as_ptr(),
                bytes.len(),
            );
        }
        buffer
    }

    /// Run `record` in one blit encoder; answer the bytes each buffer it returns holds (the
    /// lengths `record` names), or its refusal.
    fn run(
        &self,
        record: impl FnOnce(&mut BlitCommandEncoder) -> Result<Vec<(Buffer, usize)>, Error>,
    ) -> Result<Vec<Vec<u8>>, Error> {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let recorded = record(&mut blit);
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        Ok(recorded?
            .into_iter()
            .map(|(buffer, n)| {
                // SAFETY: the command buffer completed; the buffer holds at least `n` bytes.
                unsafe { std::slice::from_raw_parts(buffer.contents().unwrap().as_ptr(), n) }
                    .to_vec()
            })
            .collect())
    }
}

/// Fill `texture` from `bytes` (`per` bytes a texel) with `option`.
fn fill(
    lane: &Lane,
    blit: &mut BlitCommandEncoder,
    texture: &Texture,
    bytes: &[u8],
    option: BlitOption,
) -> Result<(), Error> {
    let per = bytes.len() as u64 / W;
    blit.copy_from_buffer_to_texture(
        &lane.buffer(bytes),
        0,
        per * W,
        0,
        Size::new(W, 1, 1),
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        option,
    )
}

/// Read `texture`'s row, `per` bytes a texel, with `option`, into a new buffer of 0xee bytes.
fn read(
    lane: &Lane,
    blit: &mut BlitCommandEncoder,
    texture: &Texture,
    per: u64,
    option: BlitOption,
) -> Result<(Buffer, usize), Error> {
    let buffer = lane.buffer(&[0xee; 64]);
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(W, 1, 1),
        &buffer,
        0,
        per * W,
        0,
        option,
    )?;
    Ok((buffer, (per * W) as usize))
}

/// `source`'s row into `destination`'s.
fn copy(
    blit: &mut BlitCommandEncoder,
    source: &Texture,
    destination: &Texture,
) -> Result<(), Error> {
    blit.copy_from_texture_to_texture(
        source,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(W, 1, 1),
        destination,
        0,
        0,
        Origin::new(0, 0, 0),
    )
}

/// Probes 8 and 8b: depth and stencil formats copy their bytes to and from colour formats of
/// their unit, in a depth image or (for a read-only `Depth32Float` on a device whose depth images
/// may not keep a `float`) a depth twin.
#[test]
fn depth_and_stencil_copy_their_bytes_with_colour_formats_of_their_unit() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane {
            queue: device.new_command_queue().unwrap(),
            device,
        };
        let read_only = TextureUsage::SHADER_READ;
        let target = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        use PixelFormat as P;
        let cases: [(P, P, &[u8], TextureUsage); 16] = [
            (P::Depth32Float, P::R32Float, &FOUR, read_only),
            (P::Depth32Float, P::R32Uint, &FOUR, target),
            (P::Depth32Float, P::RGBA8Unorm, &FOUR, read_only),
            (P::R32Float, P::Depth32Float, &FOUR, read_only),
            (P::R32Uint, P::Depth32Float, &FOUR, target),
            (P::RGBA8Unorm, P::Depth32Float, &FOUR, read_only),
            (P::Depth16Unorm, P::R16Unorm, &TWO, read_only),
            (P::Depth16Unorm, P::RG8Unorm, &TWO, target),
            (P::R16Uint, P::Depth16Unorm, &TWO, read_only),
            (P::R16Float, P::Depth16Unorm, &TWO, target),
            (P::Stencil8, P::R8Uint, &ONE, read_only),
            (P::Stencil8, P::R8Sint, &ONE, target),
            (P::R8Unorm, P::Stencil8, &ONE, read_only),
            (P::R8Snorm, P::Stencil8, &ONE, target),
            (P::Stencil8, P::Stencil8, &ONE, read_only),
            (P::Depth16Unorm, P::Depth16Unorm, &TWO, target),
        ];
        for (from, to, bytes, usage) in cases {
            let (source, destination) = (lane.texture(from, usage), lane.texture(to, usage));
            let per = bytes.len() as u64 / W;
            let got = lane.run(|blit| {
                fill(
                    &lane,
                    blit,
                    &destination,
                    &vec![0xcc; bytes.len()],
                    BlitOption::None,
                )?;
                fill(&lane, blit, &source, bytes, BlitOption::None)?;
                copy(blit, &source, &destination)?;
                Ok(vec![
                    read(&lane, blit, &destination, per, BlitOption::None)?,
                    read(&lane, blit, &source, per, BlitOption::None)?,
                ])
            });
            assert_eq!(
                got,
                Ok(vec![bytes.to_vec(), bytes.to_vec()]),
                "{from:?} into {to:?} ({usage:?}) on {name}"
            );
        }
        // Units of different sizes, and a combined format with another, are Metal's to refuse.
        let format = Error::Invalid(Invalid::new(
            "copy_from_texture_to_texture",
            Problem::Copy(CopyProblem::Format),
        ));
        for (from, to) in [
            (P::Depth32Float, P::Stencil8),
            (P::Depth16Unorm, P::Depth32Float),
            (P::Depth32FloatStencil8, P::RG32Uint),
            (P::RG32Uint, P::Depth32FloatStencil8),
        ] {
            let (source, destination) = (lane.texture(from, target), lane.texture(to, target));
            let refused = lane.run(|blit| copy(blit, &source, &destination).map(|()| Vec::new()));
            assert_eq!(
                refused,
                Err(format.clone()),
                "{from:?} into {to:?} on {name}"
            );
        }
    }
    assert_eq!(messages.take(), Vec::new(), "the validation layer reported");
}

/// Probes 8, 8b and 8c: the stencil view `X32Stencil8` of a `Depth32FloatStencil8` texture copies
/// the stencil, to and from colour and stencil formats, other stencil views, and buffers with any
/// blit option, its depth kept; a combined texture copies into another, both aspects.
#[test]
fn the_stencil_view_of_a_combined_texture_copies_its_stencil() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane {
            queue: device.new_command_queue().unwrap(),
            device,
        };
        let usage = TextureUsage::SHADER_READ
            | TextureUsage::RENDER_TARGET
            | TextureUsage::PIXEL_FORMAT_VIEW;
        let (depth, stencil) = (
            BlitOption::DepthFromDepthStencil,
            BlitOption::StencilFromDepthStencil,
        );
        // A combined texture holding `FOUR` and `ONE`, and its stencil view.
        let combined = || {
            let texture = lane.texture(PixelFormat::Depth32FloatStencil8, usage);
            let view = texture.new_texture_view(PixelFormat::X32Stencil8).unwrap();
            (texture, view)
        };
        let filled = |blit: &mut BlitCommandEncoder, texture: &Texture| -> Result<(), Error> {
            fill(&lane, blit, texture, &FOUR, depth)?;
            fill(&lane, blit, texture, &ONE, stencil)
        };
        let aspects = |blit: &mut BlitCommandEncoder, texture: &Texture| -> Result<_, Error> {
            Ok(vec![
                read(&lane, blit, texture, 1, stencil)?,
                read(&lane, blit, texture, 4, depth)?,
            ])
        };
        let written = [0x11u8, 0x22, 0x33, 0x44];
        // Out of the view into a colour texture, a stencil texture and another view.
        for into in [
            PixelFormat::R8Uint,
            PixelFormat::Stencil8,
            PixelFormat::X32Stencil8,
        ] {
            let (texture, view) = combined();
            let (other, other_view) = combined();
            let destination = match into {
                PixelFormat::X32Stencil8 => other_view,
                pf => lane.texture(pf, usage),
            };
            let got = lane.run(|blit| {
                filled(blit, &texture)?;
                fill(&lane, blit, &other, &[0; 16], depth)?;
                fill(&lane, blit, &other, &[9; 4], stencil)?;
                copy(blit, &view, &destination)?;
                let mut out = vec![read(&lane, blit, &destination, 1, BlitOption::None)?];
                out.extend(aspects(blit, &texture)?);
                Ok(out)
            });
            assert_eq!(
                got,
                Ok(vec![ONE.to_vec(), ONE.to_vec(), FOUR.to_vec()]),
                "the stencil view into {into:?} on {name}"
            );
        }
        // Into the view from a colour texture and a stencil texture.
        for from in [PixelFormat::R8Uint, PixelFormat::Stencil8] {
            let (texture, view) = combined();
            let source = lane.texture(from, usage);
            let got = lane.run(|blit| {
                filled(blit, &texture)?;
                fill(&lane, blit, &source, &written, BlitOption::None)?;
                copy(blit, &source, &view)?;
                aspects(blit, &texture)
            });
            assert_eq!(
                got,
                Ok(vec![written.to_vec(), FOUR.to_vec()]),
                "{from:?} into the stencil view on {name}"
            );
        }
        // Buffers into and out of the view, with each option.
        for option in [BlitOption::None, stencil, depth] {
            let (texture, view) = combined();
            let got = lane.run(|blit| {
                filled(blit, &texture)?;
                let mut out = vec![read(&lane, blit, &view, 1, option)?];
                fill(&lane, blit, &view, &written, option)?;
                out.extend(aspects(blit, &texture)?);
                Ok(out)
            });
            assert_eq!(
                got,
                Ok(vec![ONE.to_vec(), written.to_vec(), FOUR.to_vec()]),
                "buffer copies through the stencil view with {option:?} on {name}"
            );
        }
        // Probe 8c: a combined texture into another, both aspects, by region and whole.
        let (source, _) = combined();
        let (destination, _) = combined();
        let (whole, _) = combined();
        let got = lane.run(|blit| {
            filled(blit, &source)?;
            fill(&lane, blit, &destination, &[0; 16], depth)?;
            fill(&lane, blit, &destination, &[9; 4], stencil)?;
            blit.copy_from_texture_to_texture(
                &source,
                0,
                0,
                Origin::new(1, 0, 0),
                Size::new(2, 1, 1),
                &destination,
                0,
                0,
                Origin::new(0, 0, 0),
            )?;
            blit.copy_texture(&source, &whole)?;
            let mut out = aspects(blit, &destination)?;
            out.extend(aspects(blit, &whole)?);
            Ok(out)
        });
        let moved_depth: Vec<u8> = [&FOUR[4..12], &[0; 8][..]].concat();
        assert_eq!(
            got,
            Ok(vec![
                vec![0xff, 0x01, 9, 9],
                moved_depth,
                ONE.to_vec(),
                FOUR.to_vec()
            ]),
            "a combined texture into another on {name}"
        );
    }
    assert_eq!(messages.take(), Vec::new(), "the validation layer reported");
}

/// Probes 7, 7b and 8b: a region copied onto itself keeps its texels, the whole texture or a
/// part; between other overlapping regions of one texture, through the texture or a view of it,
/// the copy gives `memmove`'s answer (Metal's for these small regions), and the
/// device announces `overlapping-texture-copy-as-memmove` once.
#[test]
fn a_copy_onto_its_own_texels_keeps_them_and_another_overlap_moves_as_memmove() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane {
            queue: device.new_command_queue().unwrap(),
            device,
        };
        const SIDE: u64 = 64;
        let start: Vec<u8> = (0..SIDE * SIDE)
            .flat_map(|i| ((((i / SIDE) << 16) | (i % SIDE)) as u32).to_le_bytes())
            .collect();
        let mut d = TextureDescriptor::texture_2d(PixelFormat::R32Uint, SIDE, SIDE, false);
        d.usage = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
        d.resource_options = ResourceOptions::new(
            StorageMode::Private,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        );
        let texture = lane.device.new_texture(&d).unwrap();
        let view = texture.new_texture_view(PixelFormat::RGBA8Uint).unwrap();
        let region = |blit: &mut BlitCommandEncoder,
                      from: &Texture,
                      to: &Texture,
                      at: [u64; 2],
                      size: [u64; 2],
                      into: [u64; 2]| {
            blit.copy_from_texture_to_texture(
                from,
                0,
                0,
                Origin::new(at[0], at[1], 0),
                Size::new(size[0], size[1], 1),
                to,
                0,
                0,
                Origin::new(into[0], into[1], 0),
            )
        };
        let source = lane.buffer(&start);
        let got = lane.run(|blit| {
            blit.copy_from_buffer_to_texture(
                &source,
                0,
                SIDE * 4,
                0,
                Size::new(SIDE, SIDE, 1),
                &texture,
                0,
                0,
                Origin::new(0, 0, 0),
                BlitOption::None,
            )?;
            region(blit, &texture, &texture, [4, 4], [32, 32], [4, 4])?;
            region(blit, &view, &texture, [3, 7], [50, 50], [3, 7])?;
            blit.copy_texture(&texture, &texture)?;
            let out = lane.buffer(&vec![0xee; start.len()]);
            blit.copy_from_texture_to_buffer(
                &texture,
                0,
                0,
                Origin::new(0, 0, 0),
                Size::new(SIDE, SIDE, 1),
                &out,
                0,
                SIDE * 4,
                0,
                BlitOption::None,
            )?;
            Ok(vec![(out, start.len())])
        });
        assert_eq!(
            got,
            Ok(vec![start.clone()]),
            "copies onto themselves on {name}"
        );
        // `memmove`'s answer: the 48 x 48 source region as it was, written at the destination.
        let memmoved = |at: [u64; 2], into: [u64; 2]| -> Vec<u8> {
            let mut want = start.clone();
            for y in 0..48 {
                for x in 0..48 {
                    let from = (((at[1] + y) * SIDE + at[0] + x) * 4) as usize;
                    let to = (((into[1] + y) * SIDE + into[0] + x) * 4) as usize;
                    want[to..to + 4].copy_from_slice(&start[from..from + 4]);
                }
            }
            want
        };
        messages.take_notices();
        for (from, at, into) in [
            (&texture, [0, 0], [1, 0]),
            (&texture, [0, 1], [0, 0]),
            (&texture, [0, 0], [8, 8]),
            (&view, [8, 8], [0, 0]),
        ] {
            let got = lane.run(|blit| {
                blit.copy_from_buffer_to_texture(
                    &source,
                    0,
                    SIDE * 4,
                    0,
                    Size::new(SIDE, SIDE, 1),
                    &texture,
                    0,
                    0,
                    Origin::new(0, 0, 0),
                    BlitOption::None,
                )?;
                region(blit, from, &texture, at, [48, 48], into)?;
                let out = lane.buffer(&vec![0xee; start.len()]);
                blit.copy_from_texture_to_buffer(
                    &texture,
                    0,
                    0,
                    Origin::new(0, 0, 0),
                    Size::new(SIDE, SIDE, 1),
                    &out,
                    0,
                    SIDE * 4,
                    0,
                    BlitOption::None,
                )?;
                Ok(vec![(out, start.len())])
            });
            assert_eq!(
                got,
                Ok(vec![memmoved(at, into)]),
                "{at:?} onto {into:?} on {name}"
            );
        }
        let notices = assert_once(&messages.take_notices(), &MEMMOVES);
        assert_eq!(notices.len(), 1, "{notices:?} on {name}");
    }
    assert_eq!(messages.take(), Vec::new(), "the validation layer reported");
}

/// A unit's bytes: `tag`, its index `i`, then bytes that count up, `n` in all.
fn unit_bytes(i: usize, tag: u8, n: usize) -> Vec<u8> {
    let mut bytes = vec![tag, i as u8];
    bytes.extend((2..n).map(|k| (k as u8) ^ tag));
    bytes
}

/// Our probe on an Apple M3 Max under macOS 27.0.1: between a compressed format and an
/// uncompressed one whose texel is its block's bytes, each block copies into one texel and back,
/// the copy's origins and size in the destination's texels. Out of a texture of 8 x 8 blocks an
/// origin of (4, 4) and a size of 2 x 2 into an 8 x 8 texture at (1, 2) copy the blocks from
/// (4, 4) into texels (1, 2) to (2, 3); out of that texture an origin of (1, 2) and a size of
/// 2 x 2 into the compressed one at a block's texel extent (its second block across and down)
/// copy texel (0, 0) alone into that block. Every other byte stays. BC1 and BC7, and ETC2 and
/// ASTC 6x5, which a device without them holds in a carrier.
#[test]
fn compressed_blocks_copy_into_texels_of_their_bytes_and_back() {
    use PixelFormat as P;
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane {
            queue: device.new_command_queue().unwrap(),
            device,
        };
        let made = |pf: PixelFormat, width: u64, height: u64| {
            let mut d = TextureDescriptor::texture_2d(pf, width, height, false);
            d.usage = TextureUsage::SHADER_READ;
            d.resource_options = ResourceOptions::new(
                StorageMode::Private,
                CpuCacheMode::DefaultCache,
                HazardTrackingMode::Default,
            );
            lane.device.new_texture(&d).unwrap()
        };
        let cases: [(P, [u64; 2], P, usize); 4] = [
            (P::BC1RGBA, [4, 4], P::RG32Uint, 8),
            (P::BC7RGBAUnorm, [4, 4], P::RGBA32Uint, 16),
            (P::ETC2RGB8, [4, 4], P::RG32Uint, 8),
            (P::ASTC6x5LDR, [6, 5], P::RGBA32Uint, 16),
        ];
        for (c, [bw, bh], u, n) in cases {
            let compressed = made(c, 8 * bw, 8 * bh);
            let texels = made(u, 8, 8);
            let blocks: Vec<u8> = (0..64).flat_map(|i| unit_bytes(i, 0x10, n)).collect();
            let words: Vec<u8> = (0..64).flat_map(|i| unit_bytes(i, 0x80, n)).collect();
            let row = 8 * n as u64;
            let got = lane.run(|blit| {
                for (texture, bytes, size) in [
                    (&compressed, &blocks, Size::new(8 * bw, 8 * bh, 1)),
                    (&texels, &words, Size::new(8, 8, 1)),
                ] {
                    blit.copy_from_buffer_to_texture(
                        &lane.buffer(bytes),
                        0,
                        row,
                        8 * row,
                        size,
                        texture,
                        0,
                        0,
                        Origin::new(0, 0, 0),
                        BlitOption::None,
                    )?;
                }
                blit.copy_from_texture_to_texture(
                    &compressed,
                    0,
                    0,
                    Origin::new(4, 4, 0),
                    Size::new(2, 2, 1),
                    &texels,
                    0,
                    0,
                    Origin::new(1, 2, 0),
                )?;
                blit.copy_from_texture_to_texture(
                    &texels,
                    0,
                    0,
                    Origin::new(1, 2, 0),
                    Size::new(2, 2, 1),
                    &compressed,
                    0,
                    0,
                    Origin::new(bw, bh, 0),
                )?;
                let mut out = Vec::new();
                for (texture, size) in [
                    (&compressed, Size::new(8 * bw, 8 * bh, 1)),
                    (&texels, Size::new(8, 8, 1)),
                ] {
                    let buffer = lane.buffer(&vec![0xee; 64 * n]);
                    blit.copy_from_texture_to_buffer(
                        texture,
                        0,
                        0,
                        Origin::new(0, 0, 0),
                        size,
                        &buffer,
                        0,
                        row,
                        8 * row,
                        BlitOption::None,
                    )?;
                    out.push((buffer, 64 * n));
                }
                Ok(out)
            });
            let at = |x: usize, y: usize| (8 * y + x) * n..(8 * y + x + 1) * n;
            let mut want_blocks = blocks.clone();
            want_blocks[at(1, 1)].copy_from_slice(&words[at(0, 0)]);
            let mut want_words = words.clone();
            for (x, y) in [(1, 2), (2, 2), (1, 3), (2, 3)] {
                want_words[at(x, y)].copy_from_slice(&blocks[at(x + 3, y + 2)]);
            }
            assert_eq!(
                got,
                Ok(vec![want_blocks, want_words]),
                "{c:?} with {u:?} on {name}"
            );
        }
    }
    assert_eq!(messages.take(), Vec::new(), "the validation layer reported");
}
