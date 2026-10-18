//! ASTC LDR textures on the device the cfg selects, under the validation layer, against Metal's
//! decode (`metal2vulkan::bind::AstcDecode`, matched bit for bit against an Apple M3 Max by its
//! own tests).
//!
//! On a device without a format's ASTC block ([`decodes_astc`]) the texture holds its blocks in a
//! carrier and its reads go through a decoded twin, announced `carried-decoded-twin` once and, under
//! `DecodedTexelsPolicy::Compact`, `decoded-texel-precision` once ([`ASTC_TWINS`]). Every format's
//! texels read as Metal's after every kind of write ([`astc_texels_read_as_metal_s`], and
//! [`astc_texels_read_exactly_as_metal_s_under_exact`]): exactly under `DecodedTexelsPolicy::Exact`,
//! within 1/510 under the default, an sRGB form's exactly under either (its twin's 8-bit codes read
//! raw and decoded through Metal's table); a new texture's zero blocks as magenta; then blocks a CPU copy
//! wrote, a blit from a buffer, a fill of one block, and a copy from another ASTC texture, each
//! read by a command buffer after the write's, each decoded into the twin by a GPU decode pass (a
//! blit's always; a CPU copy's where it goes through a submission, else on the CPU, counted in
//! `WorkCounts`). The blocks each wrote come back unchanged from
//! `get_bytes` and from a blit into a buffer. A device with the ASTC block reads through its own
//! decode (not compared), announcing neither kind.
//!
//! On every device, the blocks of every level and slice of a mipmapped array and cube whose sides
//! are no multiple of the block, written by blits from a buffer at the level's texel size (Metal's
//! `sourceSize` at a level's edge, `bytesPerImage` the row's bytes times the block rows), come back
//! bit for bit from blits into a buffer and from `get_bytes`
//! ([`astc_blocks_round_trip_at_every_level_and_slice`]). On a device without ASTC a view of a
//! texture in its other colour space reads Metal's texels of the view's form through a second
//! decoded twin, kept in step with CPU copies and blits, a blit encoded before the view included
//! ([`astc_views_in_the_other_colour_space_read_as_metal_s`]); a heap's texture and a view of
//! another decode are refused, and creation with `ShaderWrite` and `RenderTarget` usage is not
//! ([`astc_textures_the_runtime_does_not_hold_are_refused`]).

use super::degrade_notices::{counts, on_device, on_device_with};
use super::{buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::{AstcDecode, AstcTexels};
use metal2vulkan::device::{DecodedTexelsPolicy, FormatUsage};
use metal2vulkan::metal::format::{vk_format, FormatEmulation};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitCommandEncoder, BlitOption, Buffer, CommandBufferStatus, ComputePipelineState, Degradation,
    Device, DeviceOptions, Error, HeapDescriptor, HeapType, Origin, PipelineOptions, Region,
    ResourceOptions, RuntimeGap, Size, Texture, TextureDescriptor,
};

/// The kinds an ASTC texture announces on a device without its ASTC block: the decoded twin, and
/// an LDR form's 8-bit codes under `DecodedTexelsPolicy::Compact`.
pub(super) const ASTC_TWINS: [Degradation; 2] = [
    Degradation::CarriedDecodedTwin,
    Degradation::DecodedTexelPrecision,
];

/// Every ASTC LDR format, linear and sRGB.
const FORMATS: [PixelFormat; 28] = [
    PixelFormat::ASTC4x4LDR,
    PixelFormat::ASTC5x4LDR,
    PixelFormat::ASTC5x5LDR,
    PixelFormat::ASTC6x5LDR,
    PixelFormat::ASTC6x6LDR,
    PixelFormat::ASTC8x5LDR,
    PixelFormat::ASTC8x6LDR,
    PixelFormat::ASTC8x8LDR,
    PixelFormat::ASTC10x5LDR,
    PixelFormat::ASTC10x6LDR,
    PixelFormat::ASTC10x8LDR,
    PixelFormat::ASTC10x10LDR,
    PixelFormat::ASTC12x10LDR,
    PixelFormat::ASTC12x12LDR,
    PixelFormat::ASTC4x4SRGB,
    PixelFormat::ASTC5x4SRGB,
    PixelFormat::ASTC5x5SRGB,
    PixelFormat::ASTC6x5SRGB,
    PixelFormat::ASTC6x6SRGB,
    PixelFormat::ASTC8x5SRGB,
    PixelFormat::ASTC8x6SRGB,
    PixelFormat::ASTC8x8SRGB,
    PixelFormat::ASTC10x5SRGB,
    PixelFormat::ASTC10x6SRGB,
    PixelFormat::ASTC10x8SRGB,
    PixelFormat::ASTC10x10SRGB,
    PixelFormat::ASTC12x10SRGB,
    PixelFormat::ASTC12x12SRGB,
];

/// The side, in texels, of every texture the lane reads.
const SIDE: u32 = 16;

/// The most an 8-bit code is off Metal's value (`decoded-texel-precision`), with room for `f32`.
const UNORM8_BOUND: f32 = 1.0 / 510.0 + 1e-6;

/// An LDR void-extent block of red, green, blue and alpha `0x4000`, `0x8000`, `0xc000` and
/// `0xffff`: every texel that color.
const VOID_EXTENT: [u8; 16] = [
    0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x40, 0x00, 0x80, 0x00, 0xc0, 0xff, 0xff,
];

/// Whether `device` holds a texture of `pf` in its ASTC block, decoding it its own way, rather
/// than in a carrier with a decoded twin.
pub(super) fn decodes_astc(device: &Device, pf: PixelFormat) -> bool {
    let usage = [FormatUsage::SampledImage].into();
    let mapping = vk_format(pf, &usage, device.compile_device().profile()).unwrap();
    mapping.emulation != FormatEmulation::Decoded
}

/// `count` 16-byte blocks of a seeded xorshift: random bits, of which most are error blocks and
/// the rest decode through every mode, every fourth an LDR void-extent block of a random color.
fn seeded_blocks(seed: u64, count: usize) -> Vec<u8> {
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_f491_4f6c_dd1d)
    };
    let mut blocks = Vec::with_capacity(16 * count);
    for index in 0..count {
        let mut block = [0; 16];
        block[..8].copy_from_slice(&next().to_le_bytes());
        block[8..].copy_from_slice(&next().to_le_bytes());
        if index % 4 == 3 {
            block[..8].copy_from_slice(&VOID_EXTENT[..8]);
        }
        blocks.extend_from_slice(&block);
    }
    blocks
}

/// The format scout's `scout_read_f` on `device`.
pub(super) fn reader(device: &Device) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("format_scout.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name("scout_read_f").unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// `bytes` as `float4` texels.
fn texels_of(bytes: &[u8]) -> Vec<[f32; 4]> {
    bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|t| {
            std::array::from_fn(|c| f32::from_le_bytes(t[4 * c..4 * c + 4].try_into().unwrap()))
        })
        .collect()
}

/// The `float4` texels of `texture` (`SIDE` x `SIDE`) as `state` ([`reader`]) reads them, in one
/// command buffer, waited for.
pub(super) fn read_whole(
    device: &Device,
    state: &ComputePipelineState,
    texture: &Texture,
) -> Vec<[f32; 4]> {
    let texels = u64::from(SIDE * SIDE);
    let out = buffer_with(device, &vec![0; 16 * texels as usize]).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    let side = u64::from(SIDE);
    encoder
        .dispatch_threads(Size::new(side, side, 1), Size::new(side, side, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    texels_of(&read_buffer(&out, 0, 16 * texels).unwrap())
}

/// One command buffer of blits `encode` encodes, committed and waited for.
pub(super) fn blit(device: &Device, encode: impl FnOnce(&mut BlitCommandEncoder)) {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// A texture of `pf`, `Shared`, of `descriptor`'s shape, made with `usage`.
fn made(device: &Device, mut descriptor: TextureDescriptor, usage: TextureUsage) -> Texture {
    descriptor.usage = usage;
    descriptor.resource_options = ResourceOptions::shared();
    device.new_texture(&descriptor).unwrap()
}

/// A `SIDE` x `SIDE` texture of `pf`, `Shared`, made with `usage`.
pub(super) fn texture(device: &Device, pf: PixelFormat, usage: TextureUsage) -> Texture {
    let descriptor = TextureDescriptor::texture_2d(pf, SIDE.into(), SIDE.into(), false);
    made(device, descriptor, usage)
}

/// The blocks across and down a level of `width` x `height` texels of `decode`'s.
fn blocks_of(decode: AstcDecode, width: u64, height: u64) -> [u64; 2] {
    let [bw, bh] = decode.block_extent().map(u64::from);
    [width.div_ceil(bw), height.div_ceil(bh)]
}

/// The bytes of a row of `decode`'s blocks across a `SIDE`-texel level.
fn row_bytes(decode: AstcDecode) -> u64 {
    blocks_of(decode, SIDE.into(), SIDE.into())[0] * u64::from(AstcDecode::BLOCK_BYTES)
}

/// The blocks of a `SIDE` x `SIDE` level of `decode`.
fn block_count(decode: AstcDecode) -> usize {
    let [across, down] = blocks_of(decode, SIDE.into(), SIDE.into());
    (across * down) as usize
}

/// Metal's texels of `blocks`, a `SIDE` x `SIDE` level of `decode`.
fn metal_texels(decode: AstcDecode, blocks: &[u8]) -> Vec<[f32; 4]> {
    texels_of(&decode.decode(AstcTexels::Float, blocks, [SIDE, SIDE, 1]))
}

/// A blit of `blocks` from a buffer into `level` and `slice` of `texture`, a level of `size`
/// texels, `row` bytes a row of blocks and `image` an image of them.
pub(super) fn upload(
    device: &Device,
    texture: &Texture,
    (level, slice): (u64, u64),
    size: Size,
    (row, image): (u64, u64),
    blocks: &[u8],
) {
    let source = buffer_with(device, blocks).unwrap();
    blit(device, |encoder| {
        encoder
            .copy_from_buffer_to_texture(
                &source,
                0,
                row,
                image,
                size,
                texture,
                slice,
                level,
                Origin::new(0, 0, 0),
                BlitOption::None,
            )
            .unwrap();
    });
}

/// The blocks of `level` and `slice` of `texture`, a level of `size` texels, blitted into a
/// buffer `row` bytes a row of blocks and `image` an image of them: `length` bytes.
pub(super) fn download(
    device: &Device,
    texture: &Texture,
    (level, slice): (u64, u64),
    size: Size,
    (row, image): (u64, u64),
    length: u64,
) -> Vec<u8> {
    let out: Buffer = buffer_with(device, &vec![0; length as usize]).unwrap();
    blit(device, |encoder| {
        encoder
            .copy_from_texture_to_buffer(
                texture,
                slice,
                level,
                Origin::new(0, 0, 0),
                size,
                &out,
                0,
                row,
                image,
                BlitOption::None,
            )
            .unwrap();
    });
    read_buffer(&out, 0, length).unwrap()
}

/// Every ASTC LDR format's texels read as Metal's on a device without its block, under `policy`
/// (the module docs): a new texture's, then after a CPU copy, a blit from a buffer, a fill and a
/// copy from another texture, the blocks of each coming back unchanged.
fn astc_texels_under(policy: DecodedTexelsPolicy) {
    let adjust = |options: &mut DeviceOptions| options.policy.decoded_texels = policy;
    on_device_with(adjust, |device, messages| {
        let name = device.name().to_owned();
        let state = reader(device);
        let mut wrong = Vec::new();
        let mut undecoded = Vec::new();
        let mut twinned = false;
        // The GPU decode passes and CPU decodes so far.
        let decodes = |device: &Device| {
            let counts = device.work_counts();
            (counts.block_decode_passes, counts.cpu_block_decodes)
        };
        for (f, pf) in FORMATS.into_iter().enumerate() {
            let decode = AstcDecode::of(pf).unwrap();
            let native = decodes_astc(device, pf);
            twinned |= !native;
            let row = row_bytes(decode);
            let side = Size::new(SIDE.into(), SIDE.into(), 1);
            let whole = Region::new_2d(0, 0, SIDE.into(), SIDE.into());
            let mut check = |what: &str, texture: &Texture, blocks: &[u8]| {
                let mut back = vec![0; blocks.len()];
                texture.get_bytes(&mut back, row, 0, whole, 0, 0).unwrap();
                if back != blocks {
                    wrong.push(format!("{pf:?} {what}: get_bytes changed the blocks"));
                }
                let length = blocks.len() as u64;
                if download(device, texture, (0, 0), side, (row, 0), length) != blocks {
                    wrong.push(format!(
                        "{pf:?} {what}: a blit into a buffer changed the blocks"
                    ));
                }
                if native {
                    return;
                }
                let got = read_whole(device, &state, texture);
                let want = metal_texels(decode, blocks);
                let mut most = 0.0f32;
                for (t, (got, want)) in got.iter().zip(&want).enumerate() {
                    for c in 0..4 {
                        let off = (got[c] - want[c]).abs();
                        most = most.max(off);
                        let fits = reads_as(policy, decode, got[c], want[c]);
                        if !fits && wrong.len() < 24 {
                            wrong.push(format!(
                                "{pf:?} {what} texel {t} channel {c}: {} against Metal's {}",
                                got[c], want[c]
                            ));
                        }
                    }
                }
                eprintln!("ASTCREAD {name} {policy:?} {pf:?} {what}: at most {most} off Metal's");
            };
            let texture = texture(device, pf, TextureUsage::SHADER_READ);
            // A new texture's zero blocks are error blocks: magenta.
            let count = block_count(decode);
            check("fresh", &texture, &vec![0; count * 16]);
            let seed = 0xa57c_0000 + 0x10 * f as u64;
            let written = seeded_blocks(seed, count);
            let before = decodes(device);
            texture
                .replace_region(whole, 0, 0, &written, row, 0)
                .unwrap();
            check("CPU copy", &texture, &written);
            // A CPU copy's decode: a GPU pass where it goes through a submission, else the CPU's.
            let after = decodes(device);
            if !native && after == before {
                undecoded.push(format!("{pf:?} CPU copy: neither a pass nor a CPU decode"));
            }
            let blitted = seeded_blocks(seed + 1, count);
            let before = decodes(device);
            upload(device, &texture, (0, 0), side, (row, 0), &blitted);
            check("blit", &texture, &blitted);
            let [across, down] = blocks_of(decode, SIDE.into(), SIDE.into());
            blit(device, |encoder| {
                let blocks = Region::new_2d(0, 0, across, down);
                encoder
                    .fill_texture(&texture, 0, 0, blocks, &VOID_EXTENT)
                    .unwrap();
            });
            check("fill", &texture, &VOID_EXTENT.repeat(count));
            let copy = self::texture(device, pf, TextureUsage::SHADER_READ);
            let source = seeded_blocks(seed + 2, count);
            texture
                .replace_region(whole, 0, 0, &source, row, 0)
                .unwrap();
            let blits = decodes(device);
            blit(device, |encoder| {
                let at = Origin::new(0, 0, 0);
                encoder
                    .copy_from_texture_to_texture(&texture, 0, 0, at, side, &copy, 0, 0, at)
                    .unwrap();
            });
            check("texture copy", &copy, &source);
            // Every blit's decode is a GPU pass: the upload's and the fill's, then the copy's.
            let after = decodes(device);
            if !native && (blits.0 < before.0 + 2 || after.0 == blits.0) {
                undecoded.push(format!(
                    "{pf:?} blits: GPU decode passes {before:?} -> {blits:?} -> {after:?}"
                ));
            }
        }
        assert!(wrong.is_empty(), "on {name}:\n{}", wrong.join("\n"));
        assert!(undecoded.is_empty(), "on {name}:\n{}", undecoded.join("\n"));
        let counts = counts(&messages.take_notices());
        let announced = |kind| counts.get(&kind).copied().unwrap_or(0);
        let precision = twinned && policy == DecodedTexelsPolicy::Compact;
        assert_eq!(
            (announced(ASTC_TWINS[0]), announced(ASTC_TWINS[1])),
            (usize::from(twinned), usize::from(precision)),
            "on {name}"
        );
    });
}

/// Whether `got` reads as Metal's `want` for a channel of a texel of `decode` under `policy` (the
/// module docs): bit for bit under `DecodedTexelsPolicy::Exact`, and an sRGB form's under either
/// (its twin's codes, decoded through Metal's table); an LDR form's within 1/510 under the
/// default.
fn reads_as(policy: DecodedTexelsPolicy, decode: AstcDecode, got: f32, want: f32) -> bool {
    match (policy, decode.srgb) {
        (DecodedTexelsPolicy::Exact, _) | (_, true) => got.to_bits() == want.to_bits(),
        _ => (got - want).abs() <= UNORM8_BOUND,
    }
}

/// [`astc_texels_under`] the default, `DecodedTexelsPolicy::Compact`.
#[test]
fn astc_texels_read_as_metal_s() {
    astc_texels_under(DecodedTexelsPolicy::Compact);
}

/// [`astc_texels_under`] `DecodedTexelsPolicy::Exact`, which `DevicePolicy::exact_answers` selects.
#[test]
fn astc_texels_read_exactly_as_metal_s_under_exact() {
    astc_texels_under(DecodedTexelsPolicy::Exact);
}

/// The other colour space's form of an ASTC LDR format: its sRGB form's LDR one, and the other
/// way.
fn other_space(pf: PixelFormat) -> PixelFormat {
    FORMATS
        .into_iter()
        .find(|&other| {
            let (a, b) = (AstcDecode::of(pf).unwrap(), AstcDecode::of(other).unwrap());
            a.srgb != b.srgb && a.block_extent() == b.block_extent()
        })
        .unwrap()
}

/// On a device without ASTC, a view of a texture in its other colour space (an sRGB texture as its
/// LDR form, an LDR one as its sRGB form) reads Metal's texels of the view's form, as a texture of
/// the view's format over the same blocks reads them (our probe on an M3 Max: bit for bit), and
/// the texture its own, through its second decoded twin (the module docs): after a CPU copy made
/// before the view (the twin's whole decode), a blit encoded before another texture's view was
/// made and committed after it, a blit, a CPU copy and a copy from another texture after the
/// view. A blit into a texture with no such view counts a check and records no decode
/// (`WorkCounts::second_twin_checks`, `second_twin_decodes`); one after the view records one.
#[test]
fn astc_views_in_the_other_colour_space_read_as_metal_s() {
    on_device(|device, _messages| {
        let name = device.name().to_owned();
        let state = reader(device);
        let policy = DecodedTexelsPolicy::Compact;
        let mut wrong = Vec::new();
        let formats = [
            PixelFormat::ASTC4x4SRGB,
            PixelFormat::ASTC6x5LDR,
            PixelFormat::ASTC12x12SRGB,
            PixelFormat::ASTC8x8LDR,
        ];
        for (f, pf) in formats.into_iter().enumerate() {
            if decodes_astc(device, pf) {
                continue;
            }
            let other = other_space(pf);
            let (own, theirs) = (AstcDecode::of(pf).unwrap(), AstcDecode::of(other).unwrap());
            let row = row_bytes(own);
            let count = block_count(own);
            let side = Size::new(SIDE.into(), SIDE.into(), 1);
            let whole = Region::new_2d(0, 0, SIDE.into(), SIDE.into());
            let check = |wrong: &mut Vec<String>,
                         what: &str,
                         texture: &Texture,
                         decode: AstcDecode,
                         blocks: &[u8]| {
                let got = read_whole(device, &state, texture);
                let want = metal_texels(decode, blocks);
                for (t, (got, want)) in got.iter().zip(&want).enumerate() {
                    for c in 0..4 {
                        let fits = reads_as(policy, decode, got[c], want[c]);
                        if !fits && wrong.len() < 24 {
                            wrong.push(format!(
                                "{pf:?} as {:?} {what} texel {t} channel {c}: {} against Metal's {}",
                                decode, got[c], want[c]
                            ));
                        }
                    }
                }
            };
            let seed = 0xa57c_4000 + 0x10 * f as u64;
            let texture = texture(device, pf, TextureUsage::SHADER_READ);
            // No view yet: a blit's write is checked and decodes into no second twin.
            let before = device.work_counts();
            upload(
                device,
                &texture,
                (0, 0),
                side,
                (row, 0),
                &seeded_blocks(seed, count),
            );
            let after = device.work_counts();
            if after.second_twin_checks <= before.second_twin_checks
                || after.second_twin_decodes != before.second_twin_decodes
            {
                wrong.push(format!(
                    "{pf:?} a blit without a view: checks {} -> {}, decodes {} -> {}",
                    before.second_twin_checks,
                    after.second_twin_checks,
                    before.second_twin_decodes,
                    after.second_twin_decodes
                ));
            }
            // A CPU copy, then the view: its second twin's whole decode.
            let copied = seeded_blocks(seed + 1, count);
            texture
                .replace_region(whole, 0, 0, &copied, row, 0)
                .unwrap();
            let view = texture.new_texture_view(other).unwrap();
            check(
                &mut wrong,
                "CPU copy before the view",
                &texture,
                own,
                &copied,
            );
            check(
                &mut wrong,
                "CPU copy before the view",
                &view,
                theirs,
                &copied,
            );
            // A blit after the view decodes into the second twin.
            let blitted = seeded_blocks(seed + 2, count);
            let before = device.work_counts();
            upload(device, &texture, (0, 0), side, (row, 0), &blitted);
            let after = device.work_counts();
            if after.second_twin_decodes <= before.second_twin_decodes {
                wrong.push(format!(
                    "{pf:?} a blit after the view decoded no second twin"
                ));
            }
            check(&mut wrong, "blit", &view, theirs, &blitted);
            check(&mut wrong, "blit", &texture, own, &blitted);
            // A CPU copy after the view.
            let copied = seeded_blocks(seed + 3, count);
            texture
                .replace_region(whole, 0, 0, &copied, row, 0)
                .unwrap();
            check(&mut wrong, "CPU copy", &view, theirs, &copied);
            // A copy from another texture.
            let source = self::texture(device, pf, TextureUsage::SHADER_READ);
            let blocks = seeded_blocks(seed + 4, count);
            source.replace_region(whole, 0, 0, &blocks, row, 0).unwrap();
            blit(device, |encoder| {
                let at = Origin::new(0, 0, 0);
                encoder
                    .copy_from_texture_to_texture(&source, 0, 0, at, side, &texture, 0, 0, at)
                    .unwrap();
            });
            check(&mut wrong, "texture copy", &view, theirs, &blocks);
            // A blit encoded before a texture's view was made, committed after it: the view's
            // twin reads the blocks the blit wrote.
            let later = self::texture(device, pf, TextureUsage::SHADER_READ);
            let blocks = seeded_blocks(seed + 5, count);
            let staged = buffer_with(device, &blocks).unwrap();
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.blit_command_encoder().unwrap();
            let at = Origin::new(0, 0, 0);
            encoder
                .copy_from_buffer_to_texture(
                    &staged,
                    0,
                    row,
                    0,
                    side,
                    &later,
                    0,
                    0,
                    at,
                    BlitOption::None,
                )
                .unwrap();
            encoder.end_encoding().unwrap();
            let later_view = later.new_texture_view(other).unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            check(
                &mut wrong,
                "blit encoded before the view",
                &later_view,
                theirs,
                &blocks,
            );
            check(
                &mut wrong,
                "blit encoded before the view",
                &later,
                own,
                &blocks,
            );
        }
        assert!(wrong.is_empty(), "on {name}:\n{}", wrong.join("\n"));
    });
}

/// Every level and slice of a mipmapped 2D array (two slices) and cube of sides no multiple of the
/// block, of a square, an oblong and an sRGB footprint: blocks blitted in from a buffer at the
/// level's texel size come back bit for bit from blits into a buffer and from `get_bytes` (the
/// module docs).
#[test]
fn astc_blocks_round_trip_at_every_level_and_slice() {
    on_device(|device, _messages| {
        let name = device.name().to_owned();
        let mut wrong = Vec::new();
        let formats = [
            PixelFormat::ASTC8x8LDR,
            PixelFormat::ASTC6x5LDR,
            PixelFormat::ASTC12x10SRGB,
        ];
        for (f, pf) in formats.into_iter().enumerate() {
            let decode = AstcDecode::of(pf).unwrap();
            let mut array = TextureDescriptor::texture_2d(pf, 37, 23, true);
            array.texture_type = TextureType::Type2DArray;
            array.array_length = 2;
            let cube = TextureDescriptor::texture_cube(pf, 29, true);
            for (shape, descriptor) in [("array", array), ("cube", cube)] {
                let texture = made(device, descriptor, TextureUsage::SHADER_READ);
                let slices = match descriptor.texture_type {
                    TextureType::Cube => 6,
                    _ => descriptor.array_length,
                };
                for level in 0..descriptor.mipmap_level_count {
                    let width = (descriptor.width >> level).max(1);
                    let height = (descriptor.height >> level).max(1);
                    let [across, down] = blocks_of(decode, width, height);
                    let row = across * u64::from(AstcDecode::BLOCK_BYTES);
                    let image = row * down;
                    let size = Size::new(width, height, 1);
                    for slice in 0..slices {
                        let seed = 0xa57c_2000 ^ ((f as u64) << 16) ^ (level << 8) ^ slice;
                        let blocks = seeded_blocks(seed, (across * down) as usize);
                        let at = (level, slice);
                        upload(device, &texture, at, size, (row, image), &blocks);
                        let what = format!("{pf:?} {shape} level {level} slice {slice}");
                        if download(device, &texture, at, size, (row, image), image) != blocks {
                            wrong.push(format!("{what}: a blit into a buffer changed the blocks"));
                        }
                        let mut back = vec![0; blocks.len()];
                        let region = Region::new_2d(0, 0, width, height);
                        texture
                            .get_bytes(&mut back, row, image, region, level, slice)
                            .unwrap();
                        if back != blocks {
                            wrong.push(format!("{what}: get_bytes changed the blocks"));
                        }
                    }
                }
            }
        }
        assert!(wrong.is_empty(), "on {name}:\n{}", wrong.join("\n"));
    });
}

/// On a device without ASTC: a texture of `ShaderWrite` and `RenderTarget` usage is made (as
/// Metal makes it); a placement heap's texture (whose size the device answers) and a view of the
/// texture's HDR form are refused, typed. Nothing on a device with the format's block.
#[test]
fn astc_textures_the_runtime_does_not_hold_are_refused() {
    on_device(|device, messages| {
        let pf = PixelFormat::ASTC8x8LDR;
        if decodes_astc(device, pf) {
            return;
        }
        let usage =
            TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET;
        let written = texture(device, pf, usage);
        let mut descriptor = TextureDescriptor::texture_2d(pf, SIDE.into(), SIDE.into(), false);
        descriptor.resource_options = ResourceOptions::shared();
        // Its heap size query is refused alike.
        let answer = device.heap_texture_size_and_align(&descriptor).map(drop);
        assert!(
            matches!(
                answer,
                Err(Error::NotImplemented(RuntimeGap::DecodedHeapTexture))
            ),
            "{answer:?}"
        );
        let heap_descriptor =
            HeapDescriptor::new(HeapType::Placement, 1 << 16, ResourceOptions::shared());
        let heap = device.new_heap(&heap_descriptor).unwrap();
        let placed = heap.new_texture_at_offset(&descriptor, 0).map(drop);
        assert!(
            matches!(
                placed,
                Err(Error::NotImplemented(RuntimeGap::DecodedHeapTexture))
            ),
            "{placed:?}"
        );
        // A view as its HDR form reads another decode; one as its sRGB form is made
        // ([`astc_views_in_the_other_colour_space_read_as_metal_s`]).
        let view = written.new_texture_view(PixelFormat::ASTC8x8HDR).map(drop);
        assert!(matches!(view, Err(Error::Unsupported(_))), "{view:?}");
        written.new_texture_view(PixelFormat::ASTC8x8SRGB).unwrap();
        let counts = counts(&messages.take_notices());
        assert_eq!(counts.get(&ASTC_TWINS[0]).copied(), Some(1));
    });
}
