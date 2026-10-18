//! The PVRTC decode against Metal's reads of seeded random blocks on an Apple M3 Max (macOS
//! 27.0.1).

use super::kernel::typed_module;
use super::PvrtcDecode;
use crate::bind::block_decode::{BlockDecodeModule, BLOCKS, SHAPE, TEXELS};
use crate::metal::PixelFormat;
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};

/// The eight PVRTC formats decode, and no other format does.
#[test]
fn the_pvrtc_formats_alone_decode() {
    use PixelFormat as P;
    let decoded: Vec<PixelFormat> = PixelFormat::ALL
        .iter()
        .copied()
        .filter(|&pf| PvrtcDecode::of(pf).is_some())
        .collect();
    assert_eq!(
        decoded,
        [
            P::PVRTCRGB2BPP,
            P::PVRTCRGB2BPPSRGB,
            P::PVRTCRGB4BPP,
            P::PVRTCRGB4BPPSRGB,
            P::PVRTCRGBA2BPP,
            P::PVRTCRGBA2BPPSRGB,
            P::PVRTCRGBA4BPP,
            P::PVRTCRGBA4BPPSRGB,
        ]
    );
}

/// `n` bytes of the xorshift64* stream from `seed`, each word little-endian.
pub(crate) fn stream(seed: u64, n: usize) -> Vec<u8> {
    let mut x = seed;
    let mut out = Vec::with_capacity(n + 8);
    while out.len() < n {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        out.extend_from_slice(&x.wrapping_mul(0x2545_F491_4F6C_DD1D).to_le_bytes());
    }
    out.truncate(n);
    out
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The seeded blocks of a 256 x 256 level of `decode`: the stream of seed `0x9E3779B97F4A7C15`
/// (64 x 64 blocks of 4 bits a pixel) or `0x94D049BB133111EB` (32 x 64 of 2 bits).
pub(crate) fn seeded_level(decode: PvrtcDecode) -> Vec<u8> {
    match decode.two_bits {
        true => stream(0x94D0_49BB_1331_11EB, 16384),
        false => stream(0x9E37_79B9_7F4A_7C15, 32768),
    }
}

/// Every texel of a 256 x 256 level of seeded blocks of each format is Metal's read: the `float`
/// texels' FNV-1a digest equals the digest of an Apple M3 Max's `texture.read` of the same blocks
/// into `float4` (macOS 27.0.1, our probe, blocks copied in row by row).
#[test]
fn seeded_levels_read_as_metal_read_them() {
    use PixelFormat as P;
    const DIGESTS: [(PixelFormat, u64); 8] = [
        (P::PVRTCRGB2BPP, 0x1dcf_129a_314f_4a74),
        (P::PVRTCRGB2BPPSRGB, 0xf537_90ff_99cb_07a6),
        (P::PVRTCRGB4BPP, 0x6ff3_effe_37da_4b93),
        (P::PVRTCRGB4BPPSRGB, 0x3c5d_1352_5f6e_557d),
        (P::PVRTCRGBA2BPP, 0x2fc4_4e77_57f6_25b8),
        (P::PVRTCRGBA2BPPSRGB, 0xd579_d21f_5b64_7af2),
        (P::PVRTCRGBA4BPP, 0x990f_73ea_8038_c524),
        (P::PVRTCRGBA4BPPSRGB, 0xac54_5407_88bb_89b2),
    ];
    for (pf, digest) in DIGESTS {
        let decode = PvrtcDecode::of(pf).expect("a decode");
        let blocks = seeded_level(decode);
        let codes = decode.decode(&blocks, [256, 256, 1]);
        let values: Vec<u8> = (codes.iter().enumerate())
            .flat_map(|(i, &c)| decode.metal_value(i % 4, c).to_le_bytes())
            .collect();
        assert_eq!(fnv1a(&values), digest, "{pf:?}");
    }
}

/// An sRGB format's blocks decode to its linear form's codes, which Metal reads through its
/// table: one twin serves both forms' views.
#[test]
fn an_srgb_format_decodes_to_its_linear_form_s_codes() {
    for (srgb, linear) in [
        (PixelFormat::PVRTCRGBA4BPPSRGB, PixelFormat::PVRTCRGBA4BPP),
        (PixelFormat::PVRTCRGB2BPPSRGB, PixelFormat::PVRTCRGB2BPP),
    ] {
        let (srgb, linear) = (PvrtcDecode::of(srgb), PvrtcDecode::of(linear));
        let (srgb, linear) = (srgb.expect("a decode"), linear.expect("a decode"));
        assert_eq!(srgb.linear(), linear);
        let blocks = seeded_level(srgb);
        let size = [64, 64, 1];
        assert_eq!(srgb.decode(&blocks, size), linear.decode(&blocks, size));
        assert_ne!(srgb.metal_value(0, 128), linear.metal_value(0, 128));
        assert_eq!(srgb.metal_value(3, 128), linear.metal_value(3, 128));
    }
}

/// Two images of a level decode apart, each wrapping at its own blocks; a level smaller than a
/// block decodes its one block's texels; a band of rows reads what the whole image's decode holds
/// there.
#[test]
fn images_small_levels_and_bands_decode_apart() {
    let decode = PvrtcDecode::of(PixelFormat::PVRTCRGBA2BPP).expect("a decode");
    let blocks = seeded_level(decode);
    // A 32 x 16 level of 8 x 4 blocks: 4 x 4 of them, 128 bytes an image.
    let image = decode.level_bytes([32, 16]);
    assert_eq!(image, 128);
    let two = decode.decode(&blocks[..2 * image], [32, 16, 2]);
    let first = decode.decode(&blocks[..image], [32, 16, 1]);
    let second = decode.decode(&blocks[image..2 * image], [32, 16, 1]);
    assert_eq!(two, [first.clone(), second].concat());
    assert_eq!(decode.decode(&blocks[..8], [4, 4, 1]).len(), 64);
    let mut band = vec![0; 32 * 5 * 4];
    decode.decode_rows(&blocks[..image], [32, 16], [7, 5], &mut band);
    assert_eq!(band, first[32 * 7 * 4..32 * 12 * 4]);
}

#[test]
fn the_pvrtc_decode_kernels_are_valid_on_every_committed_profile() {
    let profiles = committed_profiles().expect("profiles");
    for pf in [PixelFormat::PVRTCRGBA2BPPSRGB, PixelFormat::PVRTCRGB4BPP] {
        let decode = PvrtcDecode::of(pf).expect("a decode");
        for (_, p) in &profiles {
            assert_valid(typed_module(decode).expect("module"), p);
        }
    }
}

/// The kernel's texels of rows `first` to `first + rows` of `images` images of a level `level`
/// of `decode` whose `blocks` lie row-major; the texel buffer's words past the band's stay
/// unwritten.
fn kernel_texels(
    decode: PvrtcDecode,
    blocks: &[u8],
    level: [u32; 2],
    band: [u32; 2],
    images: u32,
) -> Vec<u8> {
    let m = typed_module(decode).expect("module");
    let (w, rows) = (level[0], band[1]);
    let length = (w * rows * images * PvrtcDecode::TEXEL_BYTES) as usize;
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: Some(SHAPE),
                address: None,
                bytes: PvrtcDecode::shape(level, band, images).to_vec(),
            },
            Buffer {
                slot: Some(BLOCKS),
                address: None,
                bytes: blocks.to_vec(),
            },
            Buffer {
                slot: Some(TEXELS),
                address: None,
                bytes: vec![0xcd; length + 64],
            },
        ],
        push: Vec::new(),
        spec: Default::default(),
    };
    let groups = |n: u32| n.div_ceil(BlockDecodeModule::LOCAL_SIZE);
    let cfg = Config {
        order: Order::Forward,
        workgroups: [groups(w) + 1, groups(rows), images + 1],
        step_limit: 1 << 26,
        ..Config::default()
    };
    let mut out = run(&m, &cfg, res).expect("run").buffers;
    let written = out.swap_remove(2);
    assert!(written[length..].iter().all(|&b| b == 0xcd), "{decode:?}");
    written[..length].to_vec()
}

/// The kernel writes the CPU's decode (Metal's, `seeded_levels_read_as_metal_read_them`) on every
/// format: a level of seeded blocks whole, a band of its rows, two images, and levels no larger
/// than a block.
#[test]
fn the_pvrtc_kernel_is_the_decode_on_every_texel() {
    for &pf in PixelFormat::ALL {
        let Some(decode) = PvrtcDecode::of(pf) else {
            continue;
        };
        let blocks = seeded_level(decode);
        for level in [[32, 32], [16, 8], [8, 8], [4, 4], [2, 2], [1, 1]] {
            let image = decode.level_bytes(level);
            let cpu = decode.decode(&blocks[..2 * image], [level[0], level[1], 2]);
            let gpu = kernel_texels(decode, &blocks, level, [0, level[1]], 2);
            assert_eq!(gpu, cpu, "{pf:?} {level:?}");
        }
        let level = [32, 32];
        let mut cpu = vec![0; 32 * 9 * 4];
        decode.decode_rows(&blocks, level, [13, 9], &mut cpu);
        let gpu = kernel_texels(decode, &blocks, level, [13, 9], 1);
        assert_eq!(gpu, cpu, "{pf:?} band");
    }
}
