//! The block decode kernel on the CPU interpreter, against the decode Metal's reads were fitted
//! to.

use super::{typed_module, BlockDecode, BlockDecodeModule, BLOCKS, SHAPE, TEXELS};
use crate::metal::PixelFormat;
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};
use crate::vk::Format;

#[test]
fn the_block_decode_kernels_are_valid_on_every_committed_profile() {
    for (_, p) in committed_profiles().expect("profiles") {
        for decode in BlockDecode::ALL {
            assert_valid(typed_module(decode).expect("module"), &p);
        }
    }
}

/// The BC1 to BC5 formats and their sRGB forms decode, each sRGB form to its unorm form's codes;
/// BC6H, BC7 and every other format do not.
#[test]
fn the_bc1_to_bc5_formats_alone_decode() {
    use PixelFormat as P;
    let decoded: Vec<(PixelFormat, BlockDecode)> = PixelFormat::ALL
        .iter()
        .filter_map(|&pf| BlockDecode::of(pf).map(|d| (pf, d)))
        .collect();
    assert_eq!(
        decoded,
        [
            (P::BC1RGBA, BlockDecode::Bc1),
            (P::BC1RGBASRGB, BlockDecode::Bc1),
            (P::BC2RGBA, BlockDecode::Bc2),
            (P::BC2RGBASRGB, BlockDecode::Bc2),
            (P::BC3RGBA, BlockDecode::Bc3),
            (P::BC3RGBASRGB, BlockDecode::Bc3),
            (P::BC4RUnorm, BlockDecode::Bc4Unorm),
            (P::BC4RSnorm, BlockDecode::Bc4Snorm),
            (P::BC5RGUnorm, BlockDecode::Bc5Unorm),
            (P::BC5RGSnorm, BlockDecode::Bc5Snorm),
        ]
    );
    for decode in BlockDecode::ALL {
        let words = decode.texel_bytes() / 4;
        let (format, pf) = match words {
            2 => (Format::R32G32_SFLOAT, P::RG32Float),
            _ if decode.texel_format() == Format::R32_SFLOAT => (Format::R32_SFLOAT, P::R32Float),
            _ => (Format::R8G8B8A8_UNORM, P::RGBA8Unorm),
        };
        assert_eq!(
            (decode.texel_format(), decode.texel_pixel_format()),
            (format, pf)
        );
    }
}

/// Texels whose values the decode's rules state outright: zero blocks (a new texture's), BC1's
/// three-color half and transparent black, BC2's alpha codes, a six-value block's constants, and
/// a signed endpoint of -128, which reads -1 and is summed as it is.
#[test]
fn the_decode_answers_the_rules_edges() {
    let zero = [0u8; 16];
    assert_eq!(BlockDecode::Bc1.texel(&zero, 0), [0xff00_0000, 0]);
    assert_eq!(BlockDecode::Bc2.texel(&zero, 5), [0, 0]);
    assert_eq!(BlockDecode::Bc3.texel(&zero, 15), [0, 0]);
    assert_eq!(BlockDecode::Bc4Unorm.texel(&zero, 3), [0, 0]);
    assert_eq!(BlockDecode::Bc5Snorm.texel(&zero, 3), [0, 0]);
    // BC1, three-color (c0 <= c1): red 0xf800 and blue 0x001f; index 2 the half, 3 black.
    let mut block = [0u8; 8];
    block[..2].copy_from_slice(&0x001fu16.to_le_bytes());
    block[2..4].copy_from_slice(&0xf800u16.to_le_bytes());
    block[4] = 0b1110_0100; // texels 0 to 3: indices 0, 1, 2, 3
    assert_eq!(BlockDecode::Bc1.texel(&block, 0), [0xffff_0000, 0]);
    assert_eq!(BlockDecode::Bc1.texel(&block, 1), [0xff00_00ff, 0]);
    // (32 · 255 + 32 · 0 + 32) >> 6 = 128 for red and blue.
    assert_eq!(BlockDecode::Bc1.texel(&block, 2), [0xff80_0080, 0]);
    assert_eq!(BlockDecode::Bc1.texel(&block, 3), [0, 0]);
    // The same endpoints four-color (BC2's and BC3's color blocks): thirds of 21 and 43.
    let (rgb, black) = super::color(&block, 2, true);
    assert_eq!(
        (rgb, black),
        ([(21 * 255 + 32) >> 6, 0, (43 * 255 + 32) >> 6], false)
    );
    // BC2's alpha code 0xa of texel 1 reads 170.
    let mut bc2 = [0u8; 16];
    bc2[0] = 0xa0;
    assert_eq!(BlockDecode::Bc2.texel(&bc2, 1)[0] >> 24, 170);
    // A six-value channel block (a0 <= a1): index 6 the low constant, 7 the high one.
    let mut channel = [0u8; 8];
    channel[0] = 10;
    channel[1] = 20;
    // Texel 0 index 6, texel 1 index 7: bits 0-2 and 3-5 of the indices.
    channel[2] = 6 | 7 << 3;
    assert_eq!(
        BlockDecode::Bc4Unorm.texel(&channel, 0)[0],
        0.0f32.to_bits()
    );
    assert_eq!(
        BlockDecode::Bc4Unorm.texel(&channel, 1)[0],
        1.0f32.to_bits()
    );
    assert_eq!(
        BlockDecode::Bc4Snorm.texel(&channel, 0)[0],
        (-1.0f32).to_bits()
    );
    assert_eq!(
        BlockDecode::Bc3.texel(&[channel, [0; 8]].concat(), 1)[0] >> 24,
        255
    );
    // A signed endpoint of -128 reads -1, and its sevenths sum it as it is: index 2 of an
    // eight-value block (a0 = 1 > a1 = -128) is (55 · 1 + 9 · -128) / 8128.
    let mut signed = [0u8; 8];
    signed[0] = 1;
    signed[1] = 0x80;
    signed[2] = 1 | 2 << 3;
    assert_eq!(
        f32::from_bits(BlockDecode::Bc4Snorm.texel(&signed, 0)[0]),
        -1.0
    );
    assert_eq!(
        f32::from_bits(BlockDecode::Bc4Snorm.texel(&signed, 1)[0]),
        (f64::from(55 - 9 * 128) / 8128.0) as f32
    );
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

/// The kernel's texels of `blocks` over a region of `size`, run in `order`.
fn kernel_texels(decode: BlockDecode, blocks: &[u8], size: [u32; 3], order: Order) -> Vec<u8> {
    let m = typed_module(decode).expect("module");
    let [w, h, d] = size;
    let texels = (w * h * d * decode.texel_bytes()) as usize;
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: Some(SHAPE),
                address: None,
                bytes: BlockDecodeModule::shape(size).to_vec(),
            },
            Buffer {
                slot: Some(BLOCKS),
                address: None,
                bytes: blocks.to_vec(),
            },
            Buffer {
                slot: Some(TEXELS),
                address: None,
                bytes: vec![0xcd; texels + 64],
            },
        ],
        push: Vec::new(),
        spec: Default::default(),
    };
    let groups = |n: u32| n.div_ceil(BlockDecodeModule::LOCAL_SIZE) + 1;
    let cfg = Config {
        order,
        workgroups: [groups(w), groups(h), d + 1],
        ..Config::default()
    };
    let mut out = run(&m, &cfg, res).expect("run").buffers;
    let written = out.swap_remove(2);
    // Invocations outside the region write nothing past its texels.
    assert!(written[texels..].iter().all(|&b| b == 0xcd), "{decode:?}");
    written[..texels].to_vec()
}

/// Every texel the kernel of each decode writes is the CPU decode's, in both invocation orders,
/// over random blocks (a signed decode's with endpoints of -128 too) of regions whole and
/// partial in their blocks, of one and of two layers.
#[test]
fn the_kernel_is_the_decode_on_every_texel() {
    for (k, decode) in BlockDecode::ALL.into_iter().enumerate() {
        for size in [[8u32, 8, 1], [13, 7, 2], [3, 2, 1]] {
            let [w, h, d] = size;
            let blocks_n = (w.div_ceil(4) * h.div_ceil(4) * d) as usize;
            let size_b = decode.block_bytes() as usize;
            let mut blocks =
                Rng(0xb10c_0000 + k as u64 * 977 + u64::from(w)).bytes(blocks_n * size_b);
            if decode.signed() {
                // Every other block's first endpoint -128 (and the second's, in BC5's).
                for (b, block) in blocks.chunks_exact_mut(size_b).enumerate() {
                    if b % 2 == 0 {
                        block[0] = 0x80;
                        if size_b == 16 {
                            block[9] = 0x80;
                        }
                    }
                }
            }
            let want = decode.decode(&blocks, size);
            let forward = kernel_texels(decode, &blocks, size, Order::Forward);
            let reverse = kernel_texels(decode, &blocks, size, Order::Reverse);
            assert_eq!(forward, reverse, "{decode:?} {size:?}: order independent");
            assert_eq!(forward, want, "{decode:?} {size:?}");
        }
    }
}

/// The kernel's integer division gives a BC4 channel's sums the CPU decode's `float`s: every
/// first endpoint against six second ones, at each of the eight indices (signed: an endpoint of
/// -128 among them).
#[test]
fn the_kernel_divides_every_channel_sum_exactly() {
    for decode in [BlockDecode::Bc4Unorm, BlockDecode::Bc4Snorm] {
        // Blocks of endpoints (a0, a1) whose index 1 (weight 64) reads a1 and index 0 reads a0:
        // every endpoint's sum 64 a, and the sevenths' sums between.
        let mut blocks = Vec::new();
        for a0 in 0..=255u8 {
            for a1 in [0u8, 1, 127, 128, 200, 255] {
                let mut block = [0u8; 8];
                block[0] = a0;
                block[1] = a1;
                // Texels 0 to 7: indices 0 to 7.
                let indices: u64 = (0..8).fold(0, |acc, i| acc | (i << (3 * i)));
                block[2..8].copy_from_slice(&indices.to_le_bytes()[..6]);
                blocks.extend_from_slice(&block);
            }
        }
        let across = (blocks.len() / 8) as u32;
        let size = [across * 4, 4, 1];
        let want = decode.decode(&blocks, size);
        let got = kernel_texels(decode, &blocks, size, Order::Forward);
        assert_eq!(got, want, "{decode:?}");
    }
}

/// A 16-bit normalized texel holds no BC4 or BC5 channel exactly but the endpoints': of the sums
/// BC4's weights make, the nearest `R16_UNORM` code's value is the channel's `float` for 256 of
/// 16,077, and the nearest `R16_SNORM` code's for few of the signed ones. So the decoded twin holds
/// `float`s.
#[test]
fn no_sixteen_bit_texel_holds_the_channel_values() {
    let weights = [0u32, 64, 9, 18, 27, 37, 46, 55, 13, 26, 38, 51];
    let mut unsigned = std::collections::BTreeSet::new();
    let mut signed = std::collections::BTreeSet::new();
    for &w in &weights {
        for a in 0..=255i32 {
            for b in 0..=255i32 {
                unsigned.insert((64 - w as i32) * a + w as i32 * b);
                let (sa, sb) = (a - 128, b - 128);
                signed.insert((64 - w as i32) * sa + w as i32 * sb);
            }
        }
    }
    let held = |sums: &std::collections::BTreeSet<i32>, d: f64, scale: f64| {
        sums.iter()
            .filter(|&&s| {
                let want = ((f64::from(s) / d) as f32).max(-1.0);
                let code = (f64::from(s) * scale / d).round();
                let got = ((code / scale) as f32).max(-1.0);
                got == want
            })
            .count()
    };
    assert_eq!(
        (unsigned.len(), held(&unsigned, 16320.0, 65535.0)),
        (16077, 256)
    );
    let signed_held = held(&signed, 8128.0, 32767.0);
    assert!(
        signed_held * 10 < signed.len(),
        "{signed_held} of {}",
        signed.len()
    );
}

/// The block-by-block decode ([`BlockDecode::decode`], the runtime's CPU decode of a write)
/// answers each texel's own decode ([`BlockDecode::texel`]), partial blocks at a region's edges
/// and signed endpoints of -128 included.
#[test]
fn the_decode_is_each_texel_s() {
    for (k, decode) in BlockDecode::ALL.into_iter().enumerate() {
        for [w, h, d] in [[8u32, 8, 1], [13, 7, 2], [3, 2, 1], [1, 1, 1], [5, 9, 3]] {
            let size = decode.block_bytes() as usize;
            let (across, down) = (w.div_ceil(4) as usize, h.div_ceil(4) as usize);
            let n = across * down * d as usize;
            let mut blocks = Rng(0xdec0_0000 + k as u64 * 131 + u64::from(w)).bytes(n * size);
            if decode.signed() {
                for block in blocks.chunks_exact_mut(size).step_by(3) {
                    block[0] = 0x80;
                }
            }
            let words = (decode.texel_bytes() / 4) as usize;
            let mut want = Vec::new();
            for z in 0..d as usize {
                for y in 0..h as usize {
                    for x in 0..w as usize {
                        let b = (z * down + y / 4) * across + x / 4;
                        let block = &blocks[b * size..(b + 1) * size];
                        let texel = decode.texel(block, (y % 4) * 4 + x % 4);
                        for word in &texel[..words] {
                            want.extend_from_slice(&word.to_le_bytes());
                        }
                    }
                }
            }
            let got = decode.decode(&blocks, [w, h, d]);
            assert_eq!(got, want, "{decode:?} {w}x{h}x{d}");
        }
    }
}
