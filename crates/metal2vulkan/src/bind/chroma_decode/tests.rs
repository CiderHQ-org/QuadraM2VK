use super::{typed_module, ChromaDecode};
use crate::bind::block_decode::{BlockDecodeModule, BLOCKS, SHAPE, TEXELS};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};

/// `n` bytes of the xorshift64* stream from `seed`.
fn stream(seed: u64, n: usize) -> Vec<u8> {
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

/// Each texel reads its own green and its red and blue interpolated with the pair beside it,
/// `(3 c + n + 2) >> 2`, alpha 255, in either order; a row's end reads its own pair.
#[test]
fn a_texel_interpolates_red_and_blue_with_the_pair_beside_it() {
    // `GBGR422` (G0 B G1 R): pair 0 red 0x40 blue 0x20, pair 1 red 0x80 blue 0x61.
    let pairs = [0x10, 0x20, 0x30, 0x40, 0x50, 0x61, 0x70, 0x80];
    let red = |c: u32, n: u32| ((3 * c + n + 2) >> 2) as u8;
    assert_eq!(
        ChromaDecode::Gbgr.decode(&pairs, [4, 1, 1]),
        [
            [0x40, 0x10, 0x20, 0xff],
            [red(0x40, 0x80), 0x30, red(0x20, 0x61), 0xff],
            [red(0x80, 0x40), 0x50, red(0x61, 0x20), 0xff],
            [0x80, 0x70, 0x61, 0xff],
        ]
        .concat()
    );
    // `BGRG422` (B G0 R G1): pair 0 red 0x30 blue 0x10, pair 1 red 0x70 blue 0x50.
    assert_eq!(
        ChromaDecode::Bgrg.decode(&pairs, [4, 1, 1]),
        [
            [0x30, 0x20, 0x10, 0xff],
            [red(0x30, 0x70), 0x40, red(0x10, 0x50), 0xff],
            [red(0x70, 0x30), 0x61, red(0x50, 0x10), 0xff],
            [0x70, 0x80, 0x50, 0xff],
        ]
        .concat()
    );
    // One pair a row reads itself on both sides; rows read their own pairs alone.
    assert_eq!(
        ChromaDecode::Gbgr.decode(&pairs, [2, 2, 1]),
        [
            [0x40, 0x10, 0x20, 0xff],
            [0x40, 0x30, 0x20, 0xff],
            [0x80, 0x50, 0x61, 0xff],
            [0x80, 0x70, 0x61, 0xff],
        ]
        .concat()
    );
}

/// Metal's reads of our M3 probe's textures (16 x 4 texels of both formats over the same bytes,
/// each the top byte of one xorshift64* step from `0x422422422`): the FNV-1a digest of the codes
/// of every texel's `float` read (each `c / 255` exactly, row by row) is this decode's.
#[test]
fn the_probe_s_textures_decode_as_metal_reads_them() {
    let mut x: u64 = 0x4_2242_2422;
    let bytes: Vec<u8> = (0..16 * 4 * 2)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 56) as u8
        })
        .collect();
    let fnv = |codes: &[u8]| {
        codes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &c| {
            (h ^ u64::from(c)).wrapping_mul(0x0100_0000_01b3)
        })
    };
    for (decode, digest) in [
        (ChromaDecode::Gbgr, 0x9f08_e255_b1af_8809),
        (ChromaDecode::Bgrg, 0xb1c8_3abd_161a_5a90),
    ] {
        let codes = decode.decode(&bytes, [16, 4, 1]);
        assert_eq!(fnv(&codes), digest, "{decode:?}: {:02x?}", &codes[..16]);
    }
}

#[test]
fn the_chroma_decode_kernels_are_valid_on_every_committed_profile() {
    let profiles = committed_profiles().expect("profiles");
    for decode in ChromaDecode::ALL {
        for (_, p) in &profiles {
            assert_valid(typed_module(decode).expect("module"), p);
        }
    }
}

/// The kernel's texels of a region `size` of `decode`'s `pairs`; the texel buffer's bytes past the
/// region's stay unwritten.
fn kernel_texels(decode: ChromaDecode, pairs: &[u8], size: [u32; 3]) -> Vec<u8> {
    let m = typed_module(decode).expect("module");
    let [w, h, d] = size;
    let length = (w * h * d * ChromaDecode::TEXEL_BYTES) as usize;
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
                bytes: pairs.to_vec(),
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
        workgroups: [groups(w) + 1, groups(h), d + 1],
        ..Config::default()
    };
    let mut out = run(&m, &cfg, res).expect("run").buffers;
    let written = out.swap_remove(2);
    assert!(written[length..].iter().all(|&b| b == 0xcd), "{decode:?}");
    written[..length].to_vec()
}

/// The kernel writes the CPU's decode on every texel of both orders: random pairs, two images, odd
/// widths and heights, rows of one pair and of one texel.
#[test]
fn the_chroma_kernel_is_the_decode_on_every_texel() {
    for (k, decode) in ChromaDecode::ALL.into_iter().enumerate() {
        for size in [[16u32, 12, 2], [13, 9, 1], [7, 5, 2], [2, 1, 1], [1, 1, 1]] {
            let [w, h, d] = size;
            let n = (w.div_ceil(2) * h * d * ChromaDecode::BLOCK_BYTES) as usize;
            let pairs = stream(0x422_0000 + k as u64, n);
            let cpu = decode.decode(&pairs, size);
            let gpu = kernel_texels(decode, &pairs, size);
            assert_eq!(gpu, cpu, "{decode:?} {size:?}");
        }
    }
}
