//! How BC1 to BC5 texels decode (the consumer's x86 guests' BC textures) on the device the cfg
//! selects, under the validation layer, against Metal's decode ([`metal_texel`]).
//!
//! Metal's decode, fitted bit for bit on an Apple M4 (macOS 27.0.1; the format
//! scout's reads of random blocks of every BC format, which MoltenVK's equal): each endpoint
//! widened to 8 bits by replicating its bits; an index's weight `w` of 64, BC7's (21 and 43 for
//! the thirds, 9, 18, 27, 37, 46 and 55 for the sevenths, 13, 26, 38 and 51 for the fifths, 32
//! for BC1's three-color half), the sum `(64 - w) a + w b`; BC1 to BC3's channels and BC3's alpha
//! rounded half up to 8 bits (`(sum + 32) >> 6`), BC2's alpha its 4-bit code times 17, each code
//! `c / 255`; BC1's fourth three-color index black with alpha 0; BC4 and BC5's sum unrounded,
//! `sum / 16320` (signed `sum / 8128`, its six-value constants -1 and 1); an sRGB color channel
//! the 8-bit code through Metal's twelve-bit table (`k / 4095` for `k = round(decode(c / 255) ·
//! 4095)`), its alpha `c / 255`. Each value is correctly rounded to `f32`.
//!
//! Per format, [`ROUNDS`] 16 x 16 textures of random blocks are read whole by the format scout's
//! `scout_read_f` (a signed format's first block with an endpoint of -128, whose interpolated
//! texels the model clamps to -1 after the sum, an inference this lane's Metal venues check).
//! Every texel reads as Metal's: on a device whose driver has a format's fact ([`Kind::fact`]:
//! BC1 to BC3 `DriverFact::ColorBlockDecodeIsMetals`, BC4 and BC5
//! `DriverFact::ChannelBlockDecodeIsMetals`) through its own decode, elsewhere through the
//! texture's decoded twin, announced `decoded-block-twin` once ([`BLOCK_DECODES`]);
//! `compressed-texel-decode` is announced by no BC1 to BC5 texture of the lane. A BC6H or BC7
//! texture, which keeps no twin, announces it where the driver lacks the format's fact
//! ([`bc6h_and_bc7_announce_where_their_decode_is_unmeasured`]). The twins lane
//! ([`decoded_twins_follow_every_write`]) reads Metal's texels after every kind of write: a blit
//! into a level, a CPU copy of a region inside an array's slice, a copy into a cube's face, an sRGB
//! view, and a heap's textures, fresh and over the bytes a released one left. A texture in a heap
//! over the caller's host pages keeps no twin, and the heap rebinds with it
//! ([`a_host_heap_s_bc_textures_keep_no_twin_and_rebind`]).
//!
//! The ordering lane ([`bc_uploads_are_ordered_before_reads`]) reads BC1, BC3 and BC7 blocks
//! right after they move into a texture, unwaited ([`Order`]), against the same blocks uploaded
//! and waited for: the device's own decode both ways, so any difference is an ordering one. The
//! fresh-contents lane ([`new_bc_textures_read_zero_blocks`]) reads a new texture of every BC
//! format, made right after one of its shape held random blocks and was released, against zero
//! blocks read the same way (the zeros, through the decode).
//!
//! The lane of writes behind work in flight ([`writes_behind_work_in_flight_read_as_metal_s`])
//! writes a texture with `replaceRegion` while a command buffer that reads it waits behind the busy
//! kernel or a shared event, and reads Metal's decode of the new blocks; [`cpu_block_decode_costs`]
//! (ignored, release) times the CPU decode that takes.

use super::degrade_notices::{counts, on_device};
use super::first_use::{read_mark, release, scheduled, Busy, ENDED};
use super::host_memory::{recording, Mapping, Seen};
use super::no_copy_heaps::{no_copy_heap, shared, HEAP};
use super::rebinds::rebind_heap;
use super::{buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::device::DriverFact;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitCommandEncoder, BlitOption, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue,
    ComputePipelineState, Degradation, Device, EventRef, HeapDescriptor, HeapType, Origin,
    PipelineOptions, Region, ResourceOptions, SharedEvent, Size, Texture, TextureDescriptor,
};
use std::sync::Arc;
use std::time::Instant;

/// The kinds a BC1 to BC5 texture announces on a driver not measured to decode it as Metal does:
/// without a decoded twin, and with one.
pub(super) const BLOCK_DECODES: [Degradation; 2] = [
    Degradation::CompressedTexelDecode,
    Degradation::DecodedBlockTwin,
];

/// The kind a write behind work in flight into a texture with a decoded twin announces, once:
/// the region decoded on the CPU ([`writes_behind_work_in_flight_read_as_metal_s`]).
pub(super) const BEHIND_WRITES: [Degradation; 1] = [Degradation::CpuBlockDecode];

/// The textures read per format.
const ROUNDS: u64 = 4;

/// A format the lane reads: its block's bytes, and how Metal decodes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// BC1: one color block.
    Bc1,
    /// BC2: explicit 4-bit alpha, then a four-color block.
    Bc2,
    /// BC3: an interpolated alpha block, then a four-color block.
    Bc3,
    /// BC4 and BC5: one or two interpolated channel blocks.
    Channels { count: usize, signed: bool },
}

impl Kind {
    /// The driver fact under which the kind's texels read as Metal's.
    fn fact(self) -> DriverFact {
        match self {
            Kind::Bc1 | Kind::Bc2 | Kind::Bc3 => DriverFact::ColorBlockDecodeIsMetals,
            Kind::Channels { .. } => DriverFact::ChannelBlockDecodeIsMetals,
        }
    }
}

/// Every BC1 to BC5 format: how Metal decodes it, and whether its color is sRGB.
const FORMATS: [(PixelFormat, Kind, bool); 10] = [
    (PixelFormat::BC1RGBA, Kind::Bc1, false),
    (PixelFormat::BC1RGBASRGB, Kind::Bc1, true),
    (PixelFormat::BC2RGBA, Kind::Bc2, false),
    (PixelFormat::BC2RGBASRGB, Kind::Bc2, true),
    (PixelFormat::BC3RGBA, Kind::Bc3, false),
    (PixelFormat::BC3RGBASRGB, Kind::Bc3, true),
    (
        PixelFormat::BC4RUnorm,
        Kind::Channels {
            count: 1,
            signed: false,
        },
        false,
    ),
    (
        PixelFormat::BC4RSnorm,
        Kind::Channels {
            count: 1,
            signed: true,
        },
        false,
    ),
    (
        PixelFormat::BC5RGUnorm,
        Kind::Channels {
            count: 2,
            signed: false,
        },
        false,
    ),
    (
        PixelFormat::BC5RGSnorm,
        Kind::Channels {
            count: 2,
            signed: true,
        },
        false,
    ),
];

impl Kind {
    fn block_bytes(self) -> usize {
        match self {
            Kind::Bc1 | Kind::Channels { count: 1, .. } => 8,
            _ => 16,
        }
    }
}

/// The weights of 64 of a 2-bit (thirds) and a 3-bit (sevenths) index, BC7's.
const THIRDS: [u32; 4] = [0, 21, 43, 64];
const SEVENTHS: [u32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
/// The weights of the fifths of a six-value channel block.
const FIFTHS: [u32; 6] = [0, 13, 26, 38, 51, 64];

/// `bits` bits widened to 8 by replicating them.
fn widen(v: u32, bits: u32) -> u32 {
    (v << (8 - bits)) | (v >> (2 * bits - 8))
}

/// A 565 color's three channels widened to 8 bits.
fn rgb565(c: u32) -> [u32; 3] {
    [
        widen((c >> 11) & 31, 5),
        widen((c >> 5) & 63, 6),
        widen(c & 31, 5),
    ]
}

/// Texel `j` (0 to 15, row-major in its block) of a color block: its three 8-bit codes and
/// whether it is BC1's transparent black. `four` forces the four-color mode (BC2 and BC3).
fn color(block: &[u8], j: usize, four: bool) -> ([u32; 3], bool) {
    let c0 = u32::from(u16::from_le_bytes([block[0], block[1]]));
    let c1 = u32::from(u16::from_le_bytes([block[2], block[3]]));
    let indices = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    let i = (indices >> (2 * j)) & 3;
    let (e0, e1) = (rgb565(c0), rgb565(c1));
    let mix = |w: u32| std::array::from_fn(|k| ((64 - w) * e0[k] + w * e1[k] + 32) >> 6);
    match (i, four || c0 > c1) {
        (0, _) => (e0, false),
        (1, _) => (e1, false),
        (2, true) => (mix(THIRDS[1]), false),
        (3, true) => (mix(THIRDS[2]), false),
        (2, false) => (mix(32), false),
        _ => ([0; 3], true),
    }
}

/// Texel `j` of an interpolated channel block (BC3's alpha, BC4's and BC5's channels): the sum
/// of 64ths of its endpoints, or a six-value block's constant as `Err(high)`.
fn channel(block: &[u8], j: usize, signed: bool) -> Result<i32, bool> {
    let end = |b: u8| {
        if signed {
            i32::from(b as i8)
        } else {
            i32::from(b)
        }
    };
    let (a0, a1) = (end(block[0]), end(block[1]));
    let mut bits = [0u8; 8];
    bits[..6].copy_from_slice(&block[2..8]);
    let i = ((u64::from_le_bytes(bits) >> (3 * j)) & 7) as usize;
    let w = match (i, a0 > a1) {
        (0, _) => 0,
        (1, _) => 64,
        (_, true) => SEVENTHS[i - 1],
        (6, false) => return Err(false),
        (7, false) => return Err(true),
        (_, false) => FIFTHS[i - 1],
    } as i32;
    Ok((64 - w) * a0 + w * a1)
}

/// `n / d` correctly rounded to `f32` (`d` odd: no `f64` result lies on an `f32` tie).
fn ratio(n: i32, d: u32) -> f32 {
    (f64::from(n) / f64::from(d)) as f32
}

/// Metal's value of texel `(x, y)` of a 16 x 16 texture of `kind` whose blocks are `blocks`
/// (four across, row-major), `srgb` its color sRGB.
fn metal_texel(kind: Kind, srgb: bool, blocks: &[u8], x: usize, y: usize) -> [f32; 4] {
    let size = kind.block_bytes();
    let b = (y / 4) * 4 + x / 4;
    let block = &blocks[b * size..(b + 1) * size];
    let j = (y % 4) * 4 + x % 4;
    let code = |c: u32| ratio(c as i32, 255);
    let rgb = |codes: [u32; 3]| -> [f32; 3] {
        codes.map(|c| match srgb {
            true => ratio(super::srgb_step(c) as i32, 4095),
            false => code(c),
        })
    };
    match kind {
        Kind::Bc1 => {
            let (codes, black) = color(block, j, false);
            let [r, g, b] = rgb(codes);
            [r, g, b, if black { 0.0 } else { 1.0 }]
        }
        Kind::Bc2 => {
            let (codes, _) = color(&block[8..], j, true);
            let alpha = (u64::from_le_bytes(block[..8].try_into().unwrap()) >> (4 * j)) & 15;
            let [r, g, b] = rgb(codes);
            [r, g, b, code(alpha as u32 * 17)]
        }
        Kind::Bc3 => {
            let (codes, _) = color(&block[8..], j, true);
            let alpha = match channel(&block[..8], j, false) {
                Ok(sum) => (sum as u32 + 32) >> 6,
                Err(high) => u32::from(high) * 255,
            };
            let [r, g, b] = rgb(codes);
            [r, g, b, code(alpha)]
        }
        Kind::Channels { count, signed } => {
            let mut texel = [0.0, 0.0, 0.0, 1.0];
            for (c, value) in texel.iter_mut().take(count).enumerate() {
                let (low, d) = match signed {
                    true => (-1.0, 8128),
                    false => (0.0, 16320),
                };
                *value = match channel(&block[8 * c..8 * c + 8], j, signed) {
                    // A signed endpoint of -128 reads -1, as its code does.
                    Ok(sum) => ratio(sum, d).max(low),
                    Err(high) => [low, 1.0][usize::from(high)],
                };
            }
            texel
        }
    }
}

/// A splitmix64 stream of bytes.
struct Rng(u64);

impl Rng {
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n)
            .map(|_| {
                self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                (z ^ (z >> 31)) as u8
            })
            .collect()
    }
}

/// The format scout's `scout_read_f` on `device`.
fn reader(device: &Device) -> ComputePipelineState {
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

/// The device's texels of a 16 x 16 texture of `pf` holding `blocks`, read whole by `state`
/// ([`reader`]).
fn read_texels(
    device: &Device,
    state: &ComputePipelineState,
    pf: PixelFormat,
    blocks: &[u8],
) -> Vec<[f32; 4]> {
    let mut descriptor = TextureDescriptor::texture_2d(pf, 16, 16, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let row = blocks.len() as u64 / 4;
    texture
        .replace_region(Region::new_2d(0, 0, 16, 16), 0, 0, blocks, row, 0)
        .unwrap();
    read_whole(device, state, &texture)
        .as_chunks::<16>()
        .0
        .iter()
        .map(|t| {
            std::array::from_fn(|c| f32::from_le_bytes(t[4 * c..4 * c + 4].try_into().unwrap()))
        })
        .collect()
}

/// The bytes of the `float4` texels of `texture` (16 x 16) as `state` ([`reader`]) reads them,
/// in one command buffer, waited for.
fn read_whole(device: &Device, state: &ComputePipelineState, texture: &Texture) -> Vec<u8> {
    let out = buffer_with(device, &[0; 16 * 16 * 16]).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder
        .dispatch_threads(Size::new(16, 16, 1), Size::new(16, 16, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    read_buffer(&out, 0, 16 * 16 * 16).unwrap()
}

/// Every BC1 to BC5 texel reads as Metal's: through the device's decode where its driver has the
/// format's fact, else through the texture's decoded twin, a device lacking either fact
/// announcing `decoded-block-twin` once and `compressed-texel-decode` never (the module docs).
#[test]
fn bc1_to_bc5_texels_read_as_metal_s() {
    on_device(|device, messages| {
        let name = device.name().to_owned();
        let facts = &device.properties().driver_facts;
        let announces = FORMATS
            .iter()
            .any(|(_, kind, _)| !facts.contains(&kind.fact()));
        let state = reader(device);
        let mut wrong = Vec::new();
        for (f, (pf, kind, srgb)) in FORMATS.into_iter().enumerate() {
            let (mut differ, mut most) = (0, 0.0f32);
            for round in 0..ROUNDS {
                let mut rng = Rng(0x5eed_bc00 + 0x100 * f as u64 + round);
                let mut blocks = rng.bytes(16 * kind.block_bytes());
                if let (0, Kind::Channels { signed: true, .. }) = (round, kind) {
                    // The first block's first endpoint -128, which reads -1.
                    blocks[0] = 0x80;
                }
                let texels = read_texels(device, &state, pf, &blocks);
                for (t, got) in texels.iter().enumerate() {
                    let want = metal_texel(kind, srgb, &blocks, t % 16, t / 16);
                    for c in 0..4 {
                        if got[c].to_bits() != want[c].to_bits() {
                            differ += 1;
                            most = most.max((got[c] - want[c]).abs());
                            if wrong.len() < 24 {
                                wrong.push(format!(
                                    "{pf:?} round {round} texel {t} channel {c}: {} against \
                                     Metal's {}",
                                    got[c], want[c]
                                ));
                            }
                        }
                    }
                }
            }
            eprintln!(
                "BCDECODE {name} {pf:?}: {differ} of {} channels off Metal's, the largest by \
                 {most} ({:.3}/255)",
                ROUNDS * 16 * 16 * 4,
                most * 255.0
            );
        }
        assert!(wrong.is_empty(), "on {name}:\n{}", wrong.join("\n"));
        let counts = counts(&messages.take_notices());
        let announced = |kind| counts.get(&kind).copied().unwrap_or(0);
        assert_eq!(announced(BLOCK_DECODES[0]), 0, "on {name}");
        assert_eq!(
            announced(BLOCK_DECODES[1]),
            usize::from(announces),
            "on {name}"
        );
    });
}

/// A BC6H and a BC7 texture, which keep no decoded twin, announce `compressed-texel-decode`
/// once exactly where the device's driver lacks either format's fact
/// (`DriverFact::Bc6hBlockDecodeIsMetals`, `DriverFact::Bc7BlockDecodeIsMetals`, taken on a
/// measured identity alone), and `decoded-block-twin` never. The decode itself is not re-measured
/// here: the lane holds no model of BC6H's or BC7's (the facts' rows name the measurements).
#[test]
fn bc6h_and_bc7_announce_where_their_decode_is_unmeasured() {
    on_device(|device, messages| {
        let name = device.name().to_owned();
        let facts = &device.properties().driver_facts;
        drop(messages.take_notices());
        let mut lacks = false;
        let mut made = Vec::new();
        for (pf, fact) in [
            (
                PixelFormat::BC6HRGBUfloat,
                DriverFact::Bc6hBlockDecodeIsMetals,
            ),
            (
                PixelFormat::BC7RGBAUnorm,
                DriverFact::Bc7BlockDecodeIsMetals,
            ),
        ] {
            lacks |= !facts.contains(&fact);
            let mut descriptor = TextureDescriptor::texture_2d(pf, 16, 16, false);
            descriptor.usage = TextureUsage::SHADER_READ;
            let texture = device
                .new_texture(&descriptor)
                .unwrap_or_else(|error| panic!("on {name}: a {pf:?} texture: {error:?}"));
            made.push(texture);
        }
        let counts = counts(&messages.take_notices());
        let announced = |kind| counts.get(&kind).copied().unwrap_or(0);
        eprintln!(
            "BC6H-BC7 {name}: facts {:?}, compressed-texel-decode {}",
            facts
                .iter()
                .filter(|f| matches!(
                    f,
                    DriverFact::Bc6hBlockDecodeIsMetals | DriverFact::Bc7BlockDecodeIsMetals
                ))
                .collect::<Vec<_>>(),
            announced(Degradation::CompressedTexelDecode)
        );
        assert_eq!(announced(Degradation::DecodedBlockTwin), 0, "on {name}");
        assert_eq!(
            announced(Degradation::CompressedTexelDecode),
            usize::from(lacks),
            "on {name}"
        );
        drop(made);
    });
}

// ---- The ordering lane: uploads before reads, unwaited ----

/// How an ordering case moves the blocks into the texture it reads.
#[derive(Clone, Copy, Debug)]
enum Order {
    /// A blit from a buffer, then the read's compute encoder, in one command buffer.
    OneCommandBuffer,
    /// A blit from a buffer in one command buffer, the read in the next on the same queue,
    /// both committed before either completes.
    NextCommandBuffer,
    /// `replaceRegion` of a `Shared` texture, then the read's command buffer.
    ReplaceThenRead,
    /// A blit from a buffer into a `Private` texture, a copy of it into a second in the next
    /// command buffer, and the read of the second in the one after, none waited for.
    CopyChain,
}

/// The device's texels of `pf` holding `blocks` through `order`, read by `state` (16 x 16, one
/// `float4` a texel): every command buffer committed before the first completes, the last
/// waited for.
fn read_ordered(
    device: &Device,
    state: &ComputePipelineState,
    pf: PixelFormat,
    blocks: &[u8],
    order: Order,
) -> Vec<u8> {
    let texture = |options: ResourceOptions| {
        let mut descriptor = TextureDescriptor::texture_2d(pf, 16, 16, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        descriptor.resource_options = options;
        device.new_texture(&descriptor).unwrap()
    };
    let row = blocks.len() as u64 / 4;
    let source = buffer_with(device, blocks).unwrap();
    let out = buffer_with(device, &[0; 16 * 16 * 16]).unwrap();
    let queue = device.new_command_queue().unwrap();
    let upload = |command_buffer: &CommandBuffer, to: &Texture| {
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            0,
            row,
            0,
            Size::new(16, 16, 1),
            to,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
    };
    let read = |command_buffer: &CommandBuffer, from: &Texture| {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(state).unwrap();
        encoder.set_texture(Some(from), 0).unwrap();
        encoder.set_buffer(Some(&out), 0, 0).unwrap();
        encoder
            .dispatch_threads(Size::new(16, 16, 1), Size::new(16, 16, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
    };
    let mut buffers = Vec::new();
    match order {
        Order::OneCommandBuffer => {
            let target = texture(ResourceOptions::private());
            let command_buffer = queue.command_buffer().unwrap();
            upload(&command_buffer, &target);
            read(&command_buffer, &target);
            command_buffer.commit().unwrap();
            buffers.push(command_buffer);
        }
        Order::NextCommandBuffer => {
            let target = texture(ResourceOptions::private());
            let first = queue.command_buffer().unwrap();
            upload(&first, &target);
            first.commit().unwrap();
            let second = queue.command_buffer().unwrap();
            read(&second, &target);
            second.commit().unwrap();
            buffers.extend([first, second]);
        }
        Order::ReplaceThenRead => {
            let target = texture(ResourceOptions::shared());
            target
                .replace_region(Region::new_2d(0, 0, 16, 16), 0, 0, blocks, row, 0)
                .unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            read(&command_buffer, &target);
            command_buffer.commit().unwrap();
            buffers.push(command_buffer);
        }
        Order::CopyChain => {
            let staging = texture(ResourceOptions::private());
            let target = texture(ResourceOptions::private());
            let first = queue.command_buffer().unwrap();
            upload(&first, &staging);
            first.commit().unwrap();
            let second = queue.command_buffer().unwrap();
            let mut blit = second.blit_command_encoder().unwrap();
            let at = Origin::new(0, 0, 0);
            blit.copy_from_texture_to_texture(
                &staging,
                0,
                0,
                at,
                Size::new(16, 16, 1),
                &target,
                0,
                0,
                at,
            )
            .unwrap();
            blit.end_encoding().unwrap();
            second.commit().unwrap();
            let third = queue.command_buffer().unwrap();
            read(&third, &target);
            third.commit().unwrap();
            buffers.extend([first, second, third]);
        }
    }
    for command_buffer in &buffers {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }
    read_buffer(&out, 0, 16 * 16 * 16).unwrap()
}

/// A BC texture read right after the blocks move into it, by a blit in the same or the previous
/// command buffer, by `replaceRegion`, or through a texture copy, none waited for, reads what a
/// texture uploaded and waited for reads (the device's decode of the same blocks).
#[test]
fn bc_uploads_are_ordered_before_reads() {
    on_device(|device, messages| {
        let name = device.name().to_owned();
        let state = reader(device);
        let mut wrong = Vec::new();
        let formats = [
            (PixelFormat::BC1RGBA, 8),
            (PixelFormat::BC3RGBA, 16),
            (PixelFormat::BC7RGBAUnorm, 16),
        ];
        for (f, (pf, block)) in formats.into_iter().enumerate() {
            for round in 0..4u64 {
                let blocks = Rng(0x5eed_0bd0 + 0x100 * f as u64 + round).bytes(16 * block);
                let waited: Vec<u8> = read_texels(device, &state, pf, &blocks)
                    .iter()
                    .flatten()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                for order in [
                    Order::OneCommandBuffer,
                    Order::NextCommandBuffer,
                    Order::ReplaceThenRead,
                    Order::CopyChain,
                ] {
                    let got = read_ordered(device, &state, pf, &blocks, order);
                    let differ = got
                        .as_chunks::<16>()
                        .0
                        .iter()
                        .zip(waited.as_chunks::<16>().0.iter())
                        .filter(|(a, b)| a != b)
                        .count();
                    if differ > 0 {
                        wrong.push(format!(
                            "{pf:?} round {round} {order:?}: {differ} of 256 texels differ"
                        ));
                    }
                }
            }
        }
        eprintln!("BCORDER {name}: {} cases wrong", wrong.len());
        assert!(wrong.is_empty(), "on {name}:\n{}", wrong.join("\n"));
        drop(messages.take_notices());
    });
}

// ---- Fresh contents: a new texture's blocks are zeros ----

/// Every BC format and its block's bytes.
const EVERY_BC: [(PixelFormat, usize); 14] = [
    (PixelFormat::BC1RGBA, 8),
    (PixelFormat::BC1RGBASRGB, 8),
    (PixelFormat::BC2RGBA, 16),
    (PixelFormat::BC2RGBASRGB, 16),
    (PixelFormat::BC3RGBA, 16),
    (PixelFormat::BC3RGBASRGB, 16),
    (PixelFormat::BC4RUnorm, 8),
    (PixelFormat::BC4RSnorm, 8),
    (PixelFormat::BC5RGUnorm, 16),
    (PixelFormat::BC5RGSnorm, 16),
    (PixelFormat::BC6HRGBFloat, 16),
    (PixelFormat::BC6HRGBUfloat, 16),
    (PixelFormat::BC7RGBAUnorm, 16),
    (PixelFormat::BC7RGBAUnormSRGB, 16),
];

/// A new texture of every BC format, `Private` and `Shared`, made right after one of its shape
/// and storage held random blocks, was read and was released, reads what a texture written with
/// zero blocks reads (the device's decode of the same blocks): Metal's new texture holds zeros, so
/// no earlier resource's bytes show through as blocks.
#[test]
fn new_bc_textures_read_zero_blocks() {
    on_device(|device, messages| {
        let name = device.name().to_owned();
        let state = reader(device);
        let mut wrong = Vec::new();
        for (f, (pf, block)) in EVERY_BC.into_iter().enumerate() {
            let zeros: Vec<u8> = read_texels(device, &state, pf, &vec![0; 16 * block])
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            for (s, options) in [ResourceOptions::private(), ResourceOptions::shared()]
                .into_iter()
                .enumerate()
            {
                for round in 0..2u64 {
                    let seed = 0x5eed_f400 + 0x100 * f as u64 + 0x10 * s as u64 + round;
                    let blocks = Rng(seed).bytes(16 * block);
                    // The released texture: written by a blit (`Private`) or `replaceRegion`
                    // (`Shared`), read, and dropped inside.
                    if s == 0 {
                        read_ordered(device, &state, pf, &blocks, Order::OneCommandBuffer);
                    } else {
                        read_texels(device, &state, pf, &blocks);
                    }
                    let mut descriptor = TextureDescriptor::texture_2d(pf, 16, 16, false);
                    descriptor.usage = TextureUsage::SHADER_READ;
                    descriptor.resource_options = options;
                    let texture = device.new_texture(&descriptor).unwrap();
                    let got = read_whole(device, &state, &texture);
                    let differ = got
                        .as_chunks::<16>()
                        .0
                        .iter()
                        .zip(zeros.as_chunks::<16>().0.iter())
                        .filter(|(a, b)| a != b)
                        .count();
                    if differ > 0 {
                        wrong.push(format!(
                            "{pf:?} {:?} round {round}: {differ} of 256 texels differ from zero \
                             blocks'",
                            ["Private", "Shared"][s]
                        ));
                    }
                }
            }
        }
        eprintln!("BCFRESH {name}: {} cases wrong", wrong.len());
        assert!(wrong.is_empty(), "on {name}:\n{}", wrong.join("\n"));
        drop(messages.take_notices());
    });
}

/// The `float4` texels `state` ([`reader`]) reads of `texture` (16 x 16).
fn texels_of(device: &Device, state: &ComputePipelineState, texture: &Texture) -> Vec<[f32; 4]> {
    read_whole(device, state, texture)
        .as_chunks::<16>()
        .0
        .iter()
        .map(|t| {
            std::array::from_fn(|c| f32::from_le_bytes(t[4 * c..4 * c + 4].try_into().unwrap()))
        })
        .collect()
}

/// The texels of `got` (16 x 16) whose bits differ from Metal's decode of `blocks` (four
/// across) of `kind`, `srgb` its color sRGB.
fn differing(got: &[[f32; 4]], kind: Kind, srgb: bool, blocks: &[u8]) -> usize {
    got.iter()
        .enumerate()
        .filter(|(t, texel)| {
            let want = metal_texel(kind, srgb, blocks, t % 16, t / 16);
            (0..4).any(|c| texel[c].to_bits() != want[c].to_bits())
        })
        .count()
}

/// How many of `got`'s texels (16 x 16) are neither Metal's decode of the blocks `old` nor of
/// `new` at their place, as a read racing a write that lands while it runs may read either.
fn neither(got: &[[f32; 4]], kind: Kind, old: &[u8], new: &[u8]) -> usize {
    got.iter()
        .enumerate()
        .filter(|(t, texel)| {
            let bits =
                |blocks: &[u8]| metal_texel(kind, false, blocks, t % 16, t / 16).map(f32::to_bits);
            let got = texel.map(f32::to_bits);
            got != bits(old) && got != bits(new)
        })
        .count()
}

/// Run `encode` on a blit encoder of one command buffer, committed and waited for.
fn blit(device: &Device, encode: impl FnOnce(&mut BlitCommandEncoder)) {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// A blit of `blocks` (16 x 16 texels, four blocks across) from a buffer into `texture`'s
/// `level` and `slice` at its origin.
fn blit_in(device: &Device, texture: &Texture, blocks: &[u8], (level, slice): (u64, u64)) {
    let source = buffer_with(device, blocks).unwrap();
    blit(device, |encoder| {
        let at = Origin::new(0, 0, 0);
        encoder
            .copy_from_buffer_to_texture(
                &source,
                0,
                blocks.len() as u64 / 4,
                0,
                Size::new(16, 16, 1),
                texture,
                slice,
                level,
                at,
                BlitOption::None,
            )
            .unwrap();
    });
}

/// The sRGB form of a BC1 to BC3 unorm format.
fn srgb_form(pf: PixelFormat) -> Option<PixelFormat> {
    match pf {
        PixelFormat::BC1RGBA => Some(PixelFormat::BC1RGBASRGB),
        PixelFormat::BC2RGBA => Some(PixelFormat::BC2RGBASRGB),
        PixelFormat::BC3RGBA => Some(PixelFormat::BC3RGBASRGB),
        _ => None,
    }
}

/// The blocks `texture`'s level 0 of slice 0 (16 x 16) holds, read on the CPU.
fn blocks_of(texture: &Texture, size: usize) -> Vec<u8> {
    let mut blocks = vec![0; 16 * size];
    let row = 4 * size as u64;
    texture
        .get_bytes(&mut blocks, row, 0, Region::new_2d(0, 0, 16, 16), 0, 0)
        .unwrap();
    blocks
}

/// After every kind of write into a BC1 to BC5 texture its reads are Metal's decode of the blocks
/// it holds (the module docs): a blit into a level of a mipmapped `Private` texture, read through
/// a view of the level; a CPU copy of a region inside a slice of a `Shared` array, read through a
/// view of the slice; a copy into a cube's face, read through a view of the face; a BC1 to BC3
/// texture through its sRGB view; and a placement heap's texture, fresh, written, and placed again
/// over the bytes a released one left (its blocks read back on the CPU).
#[test]
fn decoded_twins_follow_every_write() {
    on_device(|device, messages| {
        let name = device.name().to_owned();
        let state = reader(device);
        let mut wrong = Vec::new();
        let mut check =
            |what: String, got: Vec<[f32; 4]>, kind: Kind, srgb: bool, blocks: &[u8]| {
                let n = differing(&got, kind, srgb, blocks);
                if n > 0 {
                    wrong.push(format!("{what}: {n} of 256 texels off Metal's"));
                }
            };
        let usage = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
        let made = |mut descriptor: TextureDescriptor, options: ResourceOptions| {
            descriptor.usage = usage;
            descriptor.resource_options = options;
            device.new_texture(&descriptor).unwrap()
        };
        for (f, (pf, kind, srgb)) in FORMATS.into_iter().enumerate() {
            if srgb {
                continue;
            }
            let size = kind.block_bytes();
            let row = 4 * size as u64;
            let mut rng = Rng(0x5eed_7a00 + 0x100 * f as u64);
            let blocks = rng.bytes(16 * size);
            let view = |texture: &Texture, levels, slices| {
                texture
                    .new_texture_view_with_levels_slices(pf, TextureType::Type2D, levels, slices)
                    .unwrap()
            };

            // A blit into the second level of a mipmapped `Private` texture.
            let descriptor = TextureDescriptor::texture_2d(pf, 32, 32, true);
            let mipped = made(descriptor, ResourceOptions::private());
            blit_in(device, &mipped, &blocks, (1, 0));
            let level = view(&mipped, 1..2, 0..1);
            let got = texels_of(device, &state, &level);
            check(format!("{pf:?} level 1"), got, kind, false, &blocks);

            // A CPU copy of the 8 x 8 texels inside a slice of a `Shared` array.
            let mut descriptor = TextureDescriptor::texture_2d(pf, 16, 16, false);
            descriptor.texture_type = TextureType::Type2DArray;
            descriptor.array_length = 3;
            let array = made(descriptor, ResourceOptions::shared());
            let whole = Region::new_2d(0, 0, 16, 16);
            array.replace_region(whole, 0, 2, &blocks, row, 0).unwrap();
            let inner = rng.bytes(4 * size);
            let inside = Region::new_2d(4, 4, 8, 8);
            array
                .replace_region(inside, 0, 2, &inner, 2 * size as u64, 0)
                .unwrap();
            let mut held = blocks.clone();
            for (b, block) in inner.chunks_exact(size).enumerate() {
                let at = ((1 + b / 2) * 4 + 1 + b % 2) * size;
                held[at..at + size].copy_from_slice(block);
            }
            let slice = view(&array, 0..1, 2..3);
            let got = texels_of(device, &state, &slice);
            check(format!("{pf:?} array slice 2"), got, kind, false, &held);

            // A copy from a 2D texture into a cube's fourth face.
            let flat = made(
                TextureDescriptor::texture_2d(pf, 16, 16, false),
                ResourceOptions::private(),
            );
            blit_in(device, &flat, &blocks, (0, 0));
            let cube = made(
                TextureDescriptor::texture_cube(pf, 16, false),
                ResourceOptions::private(),
            );
            blit(device, |encoder| {
                let at = Origin::new(0, 0, 0);
                let size = Size::new(16, 16, 1);
                (encoder.copy_from_texture_to_texture(&flat, 0, 0, at, size, &cube, 3, 0, at))
                    .unwrap();
            });
            let face = view(&cube, 0..1, 3..4);
            let got = texels_of(device, &state, &face);
            check(format!("{pf:?} cube face 3"), got, kind, false, &blocks);

            // The sRGB view of a BC1 to BC3 texture.
            if let Some(srgb_pf) = srgb_form(pf) {
                let srgb_view = flat.new_texture_view(srgb_pf).unwrap();
                let got = texels_of(device, &state, &srgb_view);
                check(format!("{srgb_pf:?} view"), got, kind, true, &blocks);
            }

            // A placement heap's texture: fresh, written, and placed again over the bytes the
            // released one left.
            let mut descriptor = TextureDescriptor::texture_2d(pf, 16, 16, false);
            descriptor.usage = usage;
            descriptor.resource_options = ResourceOptions::shared();
            let answer = device.heap_texture_size_and_align(&descriptor).unwrap();
            let heap_descriptor =
                HeapDescriptor::new(HeapType::Placement, answer.size, ResourceOptions::shared());
            let heap = device.new_heap(&heap_descriptor).unwrap();
            let first = heap.new_texture_at_offset(&descriptor, 0).unwrap();
            let got = texels_of(device, &state, &first);
            check(
                format!("{pf:?} heap fresh"),
                got,
                kind,
                false,
                &blocks_of(&first, size),
            );
            first.replace_region(whole, 0, 0, &blocks, row, 0).unwrap();
            let got = texels_of(device, &state, &first);
            check(format!("{pf:?} heap written"), got, kind, false, &blocks);
            drop(first);
            let again = heap.new_texture_at_offset(&descriptor, 0).unwrap();
            let got = texels_of(device, &state, &again);
            check(
                format!("{pf:?} heap again"),
                got,
                kind,
                false,
                &blocks_of(&again, size),
            );
        }
        eprintln!("BCTWINS {name}: {} cases wrong", wrong.len());
        assert!(wrong.is_empty(), "on {name}:\n{}", wrong.join("\n"));
        drop(messages.take_notices());
    });
}

/// A BC1 to BC5 texture placed in a heap over the caller's host pages
/// (`Device::new_heap_with_bytes_no_copy`) keeps no decoded twin: its texels are the caller's
/// bytes, which change without the runtime's writes. It reads the device's decode of what the pages
/// hold, announces `compressed-texel-decode` where the device lacks a format's fact and
/// `decoded-block-twin` never, and the heap rebinds (`rebind_bytes_no_copy`) with every such
/// texture placed in it, as the consumer's replace (other pages, here random bytes) and detach
/// (fresh pages of zeros) rebind it; after the detach each reads zero blocks as Metal decodes them.
/// `HOSTHEAP` prints how many textures read the replaced pages' bytes.
#[test]
fn a_host_heap_s_bc_textures_keep_no_twin_and_rebind() {
    on_device(|device, messages| {
        let name = device.name().to_owned();
        let facts = &device.properties().driver_facts;
        let lacks = FORMATS
            .iter()
            .any(|(_, kind, _)| !facts.contains(&kind.fact()));
        let seen: [Arc<Seen>; 3] = Default::default();
        let first = Mapping::anonymous(HEAP);
        let pointer = first.pointer();
        // SAFETY: the mapping holds `HEAP` writable bytes until its deallocator releases it.
        let made = unsafe {
            no_copy_heap(
                device,
                pointer,
                HeapType::Placement,
                Some(recording(&seen[0], first)),
            )
        };
        let Some(heap) = made else {
            return;
        };
        let mut placed = Vec::new();
        let mut offset = 0u64;
        for (pf, kind, srgb) in FORMATS {
            let mut descriptor = TextureDescriptor::texture_2d(pf, 16, 16, false);
            descriptor.usage = TextureUsage::SHADER_READ;
            descriptor.resource_options = shared();
            let answer = device.heap_texture_size_and_align(&descriptor).unwrap();
            offset = offset.next_multiple_of(answer.align);
            let texture = heap
                .new_texture_at_offset(&descriptor, offset)
                .unwrap_or_else(|error| panic!("on {name}: a {pf:?} texture: {error:?}"));
            offset += answer.size;
            placed.push((texture, pf, kind, srgb));
        }
        assert!(offset <= HEAP as u64, "on {name}: {offset} bytes placed");
        let state = reader(device);
        // Sixteen zero blocks of the widest, four across.
        let zeros = [0; 16 * 16];
        let zero_reads = |stage: &str, wrong: &mut Vec<String>| {
            for (texture, pf, kind, srgb) in &placed {
                let got = texels_of(device, &state, texture);
                let n = differing(&got, *kind, *srgb, &zeros);
                if n != 0 {
                    wrong.push(format!(
                        "{pf:?} {stage}: {n} texels off zero blocks' decode"
                    ));
                }
            }
        };
        let mut wrong = Vec::new();
        zero_reads("fresh", &mut wrong);
        // The consumer's replace: other pages, of random bytes.
        let other = Mapping::anonymous(HEAP);
        other.write(0, &Rng(0x5eed_4057).bytes(HEAP));
        rebind_heap(&heap, other, &seen[1])
            .unwrap_or_else(|error| panic!("on {name}: the replace: {error:?}"));
        let followed = placed
            .iter()
            .filter(|(texture, _, kind, srgb)| {
                differing(&texels_of(device, &state, texture), *kind, *srgb, &zeros) != 0
            })
            .count();
        // The consumer's detach: fresh pages, zeros.
        rebind_heap(&heap, Mapping::anonymous(HEAP), &seen[2])
            .unwrap_or_else(|error| panic!("on {name}: the detach: {error:?}"));
        zero_reads("detached", &mut wrong);
        eprintln!(
            "HOSTHEAP {name}: {} BC textures rebound twice; {followed} read the replaced pages' \
             bytes; {} cases wrong",
            placed.len(),
            wrong.len()
        );
        assert!(wrong.is_empty(), "on {name}:\n{}", wrong.join("\n"));
        let counts = counts(&messages.take_notices());
        let announced = |kind| counts.get(&kind).copied().unwrap_or(0);
        assert_eq!(announced(Degradation::DecodedBlockTwin), 0, "on {name}");
        assert_eq!(
            announced(Degradation::CompressedTexelDecode),
            usize::from(lacks),
            "on {name}"
        );
    });
}

/// The side of the textures [`decoded_twin_costs`] measures.
const COST_SIDE: u64 = 4096;

/// The blits [`decoded_twin_costs`] times per texture (the first one's first moves included).
const COST_ROUNDS: u64 = 9;

/// The median GPU time, in milliseconds, of [`COST_ROUNDS`] blits of `blocks` from a buffer into the
/// whole level 0 of `texture`, each in a command buffer of its own, waited for.
fn blit_ms(device: &Device, texture: &Texture, blocks: &[u8], row: u64, side: u64) -> f64 {
    let source = buffer_with(device, blocks).unwrap();
    let queue = device.new_command_queue().unwrap();
    let mut times: Vec<f64> = (0..COST_ROUNDS)
        .map(|_| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.blit_command_encoder().unwrap();
            let at = Origin::new(0, 0, 0);
            let size = Size::new(side, side, 1);
            encoder
                .copy_from_buffer_to_texture(
                    &source,
                    0,
                    row,
                    0,
                    size,
                    texture,
                    0,
                    0,
                    at,
                    BlitOption::None,
                )
                .unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            let start = command_buffer.gpu_start_time().unwrap();
            (command_buffer.gpu_end_time().unwrap() - start) * 1e3
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

/// The cost of a decoded twin, measured on the device: per BC1 to BC5 decode, a `Private` texture
/// of [`COST_SIDE`] texels square on a device of its own, the `decoded-block-twin` notice (its
/// twin's and blocks' bytes and placed memory), and the median GPU time of a blit of random blocks
/// into its whole level, against a blit of the same bytes into an `RG32Uint` or `RGBA32Uint`
/// texture of a quarter the side (no twin): `TWINCOST` lines.
#[test]
#[ignore = "a measurement of GPU time: the twin cost runs take it"]
fn decoded_twin_costs() {
    for (f, (pf, kind, srgb)) in FORMATS.into_iter().enumerate() {
        if srgb {
            continue;
        }
        on_device(|device, messages| {
            let size = kind.block_bytes();
            let side = COST_SIDE / 4;
            let blocks = Rng(0x5eed_c057 + f as u64).bytes((side * side) as usize * size);
            let row = side * size as u64;
            let mut descriptor = TextureDescriptor::texture_2d(pf, COST_SIDE, COST_SIDE, false);
            descriptor.usage = TextureUsage::SHADER_READ;
            descriptor.resource_options = ResourceOptions::private();
            let twinned = device.new_texture(&descriptor).unwrap();
            for message in messages.take_notices() {
                let Some(notice) = message.degradation.as_ref() else {
                    continue;
                };
                eprintln!("TWINCOST {} {pf:?} notice {notice:?}", device.name());
            }
            let plain_pf = match size {
                8 => PixelFormat::RG32Uint,
                _ => PixelFormat::RGBA32Uint,
            };
            let mut descriptor = TextureDescriptor::texture_2d(plain_pf, side, side, false);
            descriptor.usage = TextureUsage::SHADER_READ;
            descriptor.resource_options = ResourceOptions::private();
            let plain = device.new_texture(&descriptor).unwrap();
            let with_twin = blit_ms(device, &twinned, &blocks, row, COST_SIDE);
            let without = blit_ms(device, &plain, &blocks, row, side);
            eprintln!(
                "TWINCOST {} {pf:?} {COST_SIDE}x{COST_SIDE}: blit with the decode {with_twin:.3} ms, \
                 the same bytes into {plain_pf:?} {without:.3} ms, the decode pass about {:.3} ms",
                device.name(),
                with_twin - without
            );
            drop(messages.take_notices());
        });
    }
}

/// What holds a command buffer that reads a texture until after a CPU write into it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Behind {
    /// Committed after the busy kernel on its queue (`first_use`), submitted before the write
    /// and run after the kernel ends.
    Busy,
    /// Waiting for a shared event the test signals after the write.
    Event,
}

/// A 16 x 16 `Shared` texture of `pf` holding `blocks` (four across, `size` bytes each).
fn shared_texture(device: &Device, pf: PixelFormat, blocks: &[u8], size: usize) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(pf, 16, 16, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let whole = Region::new_2d(0, 0, 16, 16);
    (texture.replace_region(whole, 0, 0, blocks, 4 * size as u64, 0)).unwrap();
    texture
}

/// A command buffer on `queue` that reads `texture` (16 x 16) into `out` with `state`
/// ([`reader`]) after waiting for `hold` to reach 1 where given, committed.
fn read_after(
    queue: &CommandQueue,
    state: &ComputePipelineState,
    texture: &Texture,
    out: &Buffer,
    hold: Option<&SharedEvent>,
) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    if let Some(hold) = hold {
        (command_buffer.encode_wait_for_event(EventRef::from(hold), 1)).unwrap();
    }
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(out), 0, 0).unwrap();
    encoder
        .dispatch_threads(Size::new(16, 16, 1), Size::new(16, 16, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
}

/// `replaceRegion` of a BC1 to BC5 texture while a command buffer that reads it is committed and
/// not yet run: one held by a shared event signalled after the write reads the new blocks as
/// Metal decodes them, as Metal's command buffer reads its CPU write. One held behind the busy
/// kernel, submitted before the write, reads each texel as Metal decodes the old or the new blocks,
/// since the call no longer waits for the twin's decode (a race Metal's contract leaves to the
/// program); a command buffer committed after it reads the new ones. Where the device keeps the
/// format's decoded twin and the write goes on the host or the staging queue, the write takes the
/// CPU's decode (`WorkCounts::cpu_block_decodes`, counted once the later command buffer ran;
/// announced `cpu-block-decode` once, [`BEHIND_WRITES`]). `BCBEHIND` prints each case's route. A
/// case behind the kernel whose command buffer ran before the write is void: lavapipe runs a
/// submission inside `vkQueueSubmit`, so the runtime submits the reader only once the kernel ended,
/// and Metal's reader that ran before the write reads the old blocks.
#[test]
fn writes_behind_work_in_flight_read_as_metal_s() {
    on_device(|device, messages| {
        let name = device.name().to_owned();
        let facts = device.properties().driver_facts.clone();
        let state = reader(device);
        let queue = device.new_command_queue().unwrap();
        let busy = Busy::new(device, &queue);
        let mut wrong = Vec::new();
        let mut held = 0;
        for (f, (pf, kind, srgb)) in FORMATS.into_iter().enumerate() {
            if srgb {
                continue;
            }
            let size = kind.block_bytes();
            let twinned = !facts.contains(&kind.fact());
            let mut rng = Rng(0x5eed_f100 + 0x100 * f as u64);
            for behind in [Behind::Busy, Behind::Event] {
                let (old, new) = (rng.bytes(16 * size), rng.bytes(16 * size));
                let texture = shared_texture(device, pf, &old, size);
                let out = buffer_with(device, &[0; 16 * 16 * 16]).unwrap();
                let hold = device.new_shared_event().unwrap();
                let at = Instant::now();
                let (spin, command_buffer) = match behind {
                    Behind::Busy => {
                        let spin = busy.commit(device, &queue);
                        let reads = read_after(&queue, &state, &texture, &out, None);
                        scheduled(&reads);
                        (Some(spin), reads)
                    }
                    Behind::Event => {
                        let reads = read_after(&queue, &state, &texture, &out, Some(&hold));
                        (None, reads)
                    }
                };
                let queued = at.elapsed();
                let before = device.work_counts();
                let at = Instant::now();
                let whole = Region::new_2d(0, 0, 16, 16);
                (texture.replace_region(whole, 0, 0, &new, 4 * size as u64, 0)).unwrap();
                let took = at.elapsed();
                // The command buffer had not run: the kernel it waits behind still spins.
                let ran = spin
                    .as_ref()
                    .is_some_and(|(_, words)| read_mark(words) == ENDED)
                    || command_buffer.status() == CommandBufferStatus::Completed;
                match &spin {
                    Some((_, words)) => release(words),
                    None => hold.set_signaled_value(1),
                }
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.error(), None, "{pf:?} {behind:?}");
                if let Some((spun, _)) = &spin {
                    spun.wait_until_completed_for(WAIT).unwrap();
                }
                let texels = |out: &Buffer| -> Vec<[f32; 4]> {
                    read_buffer(out, 0, 16 * 16 * 16)
                        .unwrap()
                        .as_chunks::<16>()
                        .0
                        .iter()
                        .map(|t| {
                            std::array::from_fn(|c| {
                                f32::from_le_bytes(t[4 * c..4 * c + 4].try_into().unwrap())
                            })
                        })
                        .collect()
                };
                let got = texels(&out);
                // A command buffer committed after the write waits for its decode.
                let later_out = buffer_with(device, &[0; 16 * 16 * 16]).unwrap();
                let later = read_after(&queue, &state, &texture, &later_out, None);
                later.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(later.error(), None, "{pf:?} {behind:?}: the later reader");
                let after = device.work_counts();
                let cpu = after.cpu_block_decodes - before.cpu_block_decodes;
                let hosted = after.host_texel_copies > before.host_texel_copies;
                let staged = after.staging_submissions > before.staging_submissions;
                let route = match (hosted, staged) {
                    (true, _) => "on the host",
                    (false, true) => "on the staging queue",
                    (false, false) => "through a submission",
                };
                let void = match ran {
                    true => ", void: the reader ran before the write",
                    false => "",
                };
                eprintln!(
                    "BCBEHIND {name} {pf:?} {behind:?}: committed in {queued:?}, copied {route}, \
                     twin {twinned}, {cpu} CPU decodes, the call {took:?}{void}"
                );
                match (ran, behind) {
                    (true, Behind::Event) => {
                        wrong.push(format!("{pf:?} {behind:?}: read before the event"));
                        continue;
                    }
                    (true, Behind::Busy) => continue,
                    (false, _) => held += 1,
                }
                let n = match behind {
                    Behind::Event => differing(&got, kind, false, &new),
                    Behind::Busy => neither(&got, kind, &old, &new),
                };
                if n > 0 {
                    let stale = differing(&got, kind, false, &old);
                    wrong.push(format!(
                        "{pf:?} {behind:?}: {n} of 256 texels off Metal's decode of the new \
                         blocks ({} of the old)",
                        256 - stale
                    ));
                }
                let n = differing(&texels(&later_out), kind, false, &new);
                if n > 0 {
                    wrong.push(format!(
                        "{pf:?} {behind:?}: the later reader: {n} of 256 texels off Metal's \
                         decode of the new blocks"
                    ));
                }
                let decodes = twinned && behind == Behind::Busy && (hosted || staged);
                if decodes && cpu != 1 {
                    wrong.push(format!("{pf:?} {behind:?}: {cpu} CPU decodes, not 1"));
                }
                if !twinned && cpu != 0 {
                    wrong.push(format!(
                        "{pf:?} {behind:?}: {cpu} CPU decodes without a twin"
                    ));
                }
            }
        }
        eprintln!("BCBEHIND {name}: {} cases wrong, {held} held", wrong.len());
        assert!(wrong.is_empty(), "on {name}:\n{}", wrong.join("\n"));
        let counts = counts(&messages.take_notices());
        let announced = |kind| counts.get(&kind).copied().unwrap_or(0);
        let decoded = device.work_counts().cpu_block_decodes > 0;
        assert_eq!(
            announced(BEHIND_WRITES[0]),
            usize::from(decoded),
            "on {name}"
        );
    });
}

/// The textures' side [`cpu_block_decode_costs`] writes.
const DECODE_SIDE: u64 = 2048;

/// A `Shared` BC1 to BC5 texture whose decoded twin takes CPU copies (its blocks' copies go on
/// the host or the staging queue) is never written by GPU work: its making starts the twin on
/// the CPU and makes no decode pass, and a `replace_region` while nothing is in flight decodes
/// on the CPU too (one `WorkCounts::cpu_block_decodes` by the time a command buffer that reads
/// it completed, the read waits for it; no `block_decode_passes`), so no GPU decode of older blocks
/// can be in flight to land after a later write's CPU decode (master `7d426aff` on lavapipe under
/// the lane's load: `writes_behind_work_in_flight_read_as_metal_s` read a whole texture's old
/// texels). A command buffer then reads Metal's decode of the written blocks. A format the device
/// decodes as Metal does, or whose twin takes no CPU copies, prints its route alone (`BCROUTE`).
#[test]
fn host_written_twins_take_no_gpu_decode() {
    on_device(|device, messages| {
        let name = device.name().to_owned();
        let facts = device.properties().driver_facts.clone();
        let state = reader(device);
        let queue = device.new_command_queue().unwrap();
        let mut checked = 0;
        for (f, (pf, kind, srgb)) in FORMATS.into_iter().enumerate() {
            if srgb {
                continue;
            }
            let size = kind.block_bytes();
            let twinned = !facts.contains(&kind.fact());
            let blocks = Rng(0x5eed_f200 + 0x100 * f as u64).bytes(16 * size);
            let made = device.work_counts();
            let mut descriptor = TextureDescriptor::texture_2d(pf, 16, 16, false);
            descriptor.usage = TextureUsage::SHADER_READ;
            descriptor.resource_options = ResourceOptions::shared();
            let texture = device.new_texture(&descriptor).unwrap();
            let before = device.work_counts();
            let whole = Region::new_2d(0, 0, 16, 16);
            (texture.replace_region(whole, 0, 0, &blocks, 4 * size as u64, 0)).unwrap();
            let after = device.work_counts();
            let routed = after.host_texel_copies > before.host_texel_copies
                || after.staging_submissions > before.staging_submissions;
            let out = buffer_with(device, &[0; 16 * 16 * 16]).unwrap();
            let reads = read_after(&queue, &state, &texture, &out, None);
            reads.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(reads.error(), None, "{pf:?} on {name}");
            let read = device.work_counts();
            eprintln!("BCROUTE {name} {pf:?}: twin {twinned}, CPU copies {routed}");
            let got: Vec<[f32; 4]> = read_buffer(&out, 0, 16 * 16 * 16)
                .unwrap()
                .as_chunks::<16>()
                .0
                .iter()
                .map(|t| {
                    std::array::from_fn(|c| {
                        f32::from_le_bytes(t[4 * c..4 * c + 4].try_into().unwrap())
                    })
                })
                .collect();
            let off = differing(&got, kind, false, &blocks);
            assert_eq!(
                off, 0,
                "{pf:?} on {name}: {off} of 256 texels off Metal's decode"
            );
            if !(twinned && routed) {
                continue;
            }
            checked += 1;
            assert_eq!(
                (read.block_decode_passes - made.block_decode_passes),
                0,
                "{pf:?} on {name}: a GPU decode pass for a twin that takes CPU copies"
            );
            assert_eq!(
                read.cpu_block_decodes - before.cpu_block_decodes,
                1,
                "{pf:?} on {name}: the write's CPU decodes"
            );
        }
        eprintln!("BCROUTE {name}: {checked} formats' twins take CPU copies");
        drop(messages.take_notices());
    });
}

/// The cost of a CPU decode behind work in flight, measured on the device: per BC1 to BC5 decode, a
/// `Shared` texture of [`DECODE_SIDE`] texels square, the median wall time of its making, of
/// `replaceRegion` of its whole level while nothing reads it, of the first command buffer that
/// reads it then (from its commit to its completion: it waits at its submission for the decodes
/// queued) and of the next, and of `replaceRegion` behind the busy kernel, a command buffer that
/// reads it submitted (which waits for no decode): `CPUDECODE` lines; and the making of the
/// device's first such texture, which may make the device's decode scratch.
#[test]
#[ignore = "a measurement of wall time: the CPU decode runs take it in release"]
fn cpu_block_decode_costs() {
    on_device(|device, messages| {
        let name = device.name().to_owned();
        let state = reader(device);
        let queue = device.new_command_queue().unwrap();
        let busy = Busy::new(device, &queue);
        let median = |mut times: Vec<f64>| {
            times.sort_by(f64::total_cmp);
            times[times.len() / 2]
        };
        let mut first = None;
        for (f, (pf, kind, srgb)) in FORMATS.into_iter().enumerate() {
            if srgb {
                continue;
            }
            let size = kind.block_bytes();
            let side = DECODE_SIDE / 4;
            let blocks = Rng(0x5eed_dec0 + f as u64).bytes((side * side) as usize * size);
            let row = side * size as u64;
            let whole = Region::new_2d(0, 0, DECODE_SIDE, DECODE_SIDE);
            let mut descriptor = TextureDescriptor::texture_2d(pf, DECODE_SIDE, DECODE_SIDE, false);
            descriptor.usage = TextureUsage::SHADER_READ;
            descriptor.resource_options = ResourceOptions::shared();
            let mut made = Vec::new();
            let mut idle = Vec::new();
            let mut first_read = Vec::new();
            let mut next_read = Vec::new();
            let mut behind = Vec::new();
            let before = device.work_counts();
            for _ in 0..5 {
                let at = Instant::now();
                let texture = device.new_texture(&descriptor).unwrap();
                made.push(at.elapsed().as_secs_f64() * 1e3);
                first = first.or(made.last().copied());
                let at = Instant::now();
                (texture.replace_region(whole, 0, 0, &blocks, row, 0)).unwrap();
                idle.push(at.elapsed().as_secs_f64() * 1e3);
                // The first read of the texture waits for its decodes; the next reads them done.
                let out = buffer_with(device, &[0; 16 * 16 * 16]).unwrap();
                let at = Instant::now();
                let reads = read_after(&queue, &state, &texture, &out, None);
                reads.wait_until_completed_for(WAIT).unwrap();
                first_read.push(at.elapsed().as_secs_f64() * 1e3);
                let at = Instant::now();
                let reads = read_after(&queue, &state, &texture, &out, None);
                reads.wait_until_completed_for(WAIT).unwrap();
                next_read.push(at.elapsed().as_secs_f64() * 1e3);
                let (spun, words) = busy.commit(device, &queue);
                let reads = read_after(&queue, &state, &texture, &out, None);
                scheduled(&reads);
                let at = Instant::now();
                (texture.replace_region(whole, 0, 0, &blocks, row, 0)).unwrap();
                let took = at.elapsed().as_secs_f64() * 1e3;
                // A round whose reader ran before the write measured no CPU decode.
                if read_mark(&words) != ENDED {
                    behind.push(took);
                }
                release(&words);
                reads.wait_until_completed_for(WAIT).unwrap();
                spun.wait_until_completed_for(WAIT).unwrap();
            }
            if behind.is_empty() {
                eprintln!("CPUDECODE {name} {pf:?}: every reader ran before the write");
                continue;
            }
            let cpu = device.work_counts().cpu_block_decodes - before.cpu_block_decodes;
            let held = behind.len();
            let (made, idle, behind) = (median(made), median(idle), median(behind));
            let (first_read, next_read) = (median(first_read), median(next_read));
            eprintln!(
                "CPUDECODE {name} {pf:?} {DECODE_SIDE}x{DECODE_SIDE}: made {made:.3} ms, \
                 replace_region {idle:.3} ms idle, first read {first_read:.3} ms (the next \
                 {next_read:.3} ms), replace_region {behind:.3} ms behind a command buffer in \
                 flight ({cpu} CPU decodes in 5 rounds, {held} held)"
            );
        }
        if let Some(first) = first {
            eprintln!("CPUDECODE {name}: the first texture made in {first:.3} ms");
        }
        drop(messages.take_notices());
    });
}
