use super::{rule, typed_module, TexelWords, UnusedBitsModule, SHAPE, TILES, WORDS};
use crate::metal::format::ExtendedRange;
use crate::metal::PixelFormat;
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};

/// The signed normalized formats Metal's tile store rewrites.
const SNORM: [PixelFormat; 6] = [
    PixelFormat::R8Snorm,
    PixelFormat::RG8Snorm,
    PixelFormat::RGBA8Snorm,
    PixelFormat::R16Snorm,
    PixelFormat::RG16Snorm,
    PixelFormat::RGBA16Snorm,
];

/// Every format and the words it is held in.
fn formats() -> Vec<(PixelFormat, TexelWords)> {
    let mut all = Vec::new();
    for (plain, srgb) in [
        (PixelFormat::BGR10XR, PixelFormat::BGR10XRSRGB),
        (PixelFormat::BGRA10XR, PixelFormat::BGRA10XRSRGB),
        (PixelFormat::RGB10A82PXR10, PixelFormat::RGB10A82PXR10SRGB),
    ] {
        all.push((plain, TexelWords::Raw));
        all.push((srgb, TexelWords::Raw));
    }
    for format in [PixelFormat::RGB10A82PXR10, PixelFormat::RGB10A82PXR10SRGB] {
        all.push((format, TexelWords::ColorPlane));
    }
    for format in [
        PixelFormat::R10UnormX6,
        PixelFormat::RG10UnormX12,
        PixelFormat::R10UnormX6SRGB,
        PixelFormat::RG10UnormX12SRGB,
        PixelFormat::R16Bfloat,
        PixelFormat::RG16Bfloat,
        PixelFormat::RGBA16Bfloat,
        PixelFormat::RGB9E5Float,
    ]
    .into_iter()
    .chain(SNORM)
    {
        all.push((format, TexelWords::Raw));
    }
    all
}

/// The module of `format` held as `words`.
fn module(format: PixelFormat, words: TexelWords) -> crate::spirv::ir::TypedModule {
    let rule = rule(format, words).expect("a rule");
    let bytes = UnusedBitsModule::texel_bytes(format, words).expect("texel bytes");
    typed_module(&rule, bytes).expect("module")
}

#[test]
fn the_unused_bits_kernel_is_valid_on_every_committed_profile() {
    for (_, p) in committed_profiles().expect("profiles") {
        for (format, words) in formats() {
            assert_valid(module(format, words), &p);
        }
    }
}

/// Only a two-plane format has color-plane words, and a format Metal's tile store keeps has no
/// rule.
#[test]
fn only_a_two_plane_format_has_color_plane_words() {
    for (format, words) in formats() {
        assert!(rule(format, words).is_some(), "{format:?} {words:?}");
        let plane = rule(format, TexelWords::ColorPlane).is_some();
        let two_planes = matches!(
            ExtendedRange::of(format),
            Some(ExtendedRange::Planes { .. })
        );
        assert_eq!(plane, two_planes, "{format:?}");
    }
    for format in [
        PixelFormat::RGBA8Unorm,
        PixelFormat::RGBA16Float,
        PixelFormat::RG11B10Float,
    ] {
        assert!(rule(format, TexelWords::Raw).is_none(), "{format:?}");
        assert!(UnusedBitsModule::texel_bytes(format, TexelWords::Raw).is_none());
    }
}

/// Metal's store of an `RGB9E5Float` word (the module docs), computed here apart from the
/// kernel: its value's own encoding, the exponent the least that holds the largest mantissa
/// below 512, at least 0.
fn shared_exponent(word: u32) -> u32 {
    let m = [word & 0x1ff, (word >> 9) & 0x1ff, (word >> 18) & 0x1ff];
    let mut e = word >> 27;
    let mut m = m;
    while e > 0 && m.iter().all(|&x| x < 256) {
        m = m.map(|x| x << 1);
        e -= 1;
    }
    m[0] | m[1] << 9 | m[2] << 18 | e << 27
}

/// Metal's store of `word`, the `odd` word of its texel, of `format` held as `words` (the module
/// docs), computed here apart from the kernel.
fn stored(format: PixelFormat, words: TexelWords, word: u32, odd: bool) -> u32 {
    let codes = 0x3fff_ffff;
    let Some(form) = ExtendedRange::of(format) else {
        if format == PixelFormat::RGB9E5Float {
            return shared_exponent(word);
        }
        let lane = match format {
            PixelFormat::R16Snorm | PixelFormat::RG16Snorm | PixelFormat::RGBA16Snorm => 16,
            _ => 8,
        };
        let mask = (1u32 << lane) - 1;
        return (0..32).step_by(lane).fold(0, |out, shift| {
            let code = (word >> shift) & mask;
            let code = if code == 1 << (lane - 1) {
                code + 1
            } else {
                code
            };
            out | code << shift
        });
    };
    match (form, words) {
        (ExtendedRange::Bgr10 { .. }, _) => word | !codes,
        (ExtendedRange::Planes { .. }, TexelWords::Raw) if odd => word,
        (ExtendedRange::Planes { .. }, _) => word & codes,
        (ExtendedRange::Bgra10 { .. }, _) if !odd => word & 0xffc0_ffc0,
        (ExtendedRange::Bgra10 { .. }, _) => (word & 0xffc0) | (word >> 22).clamp(384, 894) << 22,
        _ if form.bfloat() => {
            let code = |h: u32| form.load_code(h).expect("a bfloat code");
            code(word & 0xffff) | code(word >> 16) << 16
        }
        _ => {
            let half = |h: u32| form.load_code(h >> 6).expect("a padded code") << 6;
            half(word & 0xffff) | half(word >> 16) << 16
        }
    }
}

/// A splitmix64 stream: every bit of each word random.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        (z ^ (z >> 31)) as u32
    }
}

/// Run the kernel of `format` held as `words` over `bytes` with the mask `tiles` and the shape
/// `shape`, in `groups` workgroups, both invocation orders.
fn rewritten(
    (format, words): (PixelFormat, TexelWords),
    bytes: &[u8],
    tiles: &[u32],
    shape: [u8; 16],
    groups: u32,
) -> Vec<u8> {
    let m = module(format, words);
    let mut out = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let res = Resources {
            buffers: vec![
                Buffer {
                    slot: Some(WORDS),
                    address: None,
                    bytes: bytes.to_vec(),
                },
                Buffer {
                    slot: Some(TILES),
                    address: None,
                    bytes: tiles.iter().flat_map(|w| w.to_le_bytes()).collect(),
                },
                Buffer {
                    slot: Some(SHAPE),
                    address: None,
                    bytes: shape.to_vec(),
                },
            ],
            push: Vec::new(),
            spec: Default::default(),
        };
        let cfg = Config {
            order,
            workgroups: [groups, 1, 1],
            ..Config::default()
        };
        out.push(run(&m, &cfg, res).expect("run").buffers.swap_remove(0));
    }
    assert_eq!(out[0], out[1], "order independent");
    out.swap_remove(0)
}

/// A region of 70 texels a row (rows padded to a word), 40 rows a layer, 2 layers, bound from
/// row 3 on, under a mask of every third tile: each word of a set tile's texel takes Metal's
/// store, every other word stays, for every format, and invocations past the words write nothing.
#[test]
fn each_word_of_a_masked_tile_takes_metals_store_and_no_other_changes() {
    let (columns, layer_rows, layers, first_row) = (70u32, 40u32, 2u32, 3u32);
    let tiles_per_row = columns.div_ceil(UnusedBitsModule::TILE);
    let tile_rows = layer_rows.div_ceil(UnusedBitsModule::TILE);
    let tiles = layers * tile_rows * tiles_per_row;
    let mask: Vec<u32> = (0..tiles.div_ceil(32))
        .map(|at| {
            (0..32)
                .filter(|bit| (32 * at + bit) % 3 == 0)
                .fold(0, |m, bit| m | 1 << bit)
        })
        .collect();
    let set = |tile: u32| tile < tiles && mask[(tile / 32) as usize] >> (tile % 32) & 1 == 1;
    for (format, words) in formats() {
        let texel = UnusedBitsModule::texel_bytes(format, words).unwrap();
        let row_words = (columns * texel).div_ceil(4);
        let rows = layers * layer_rows - first_row;
        let mut rng = Rng(u64::from(texel) << 32 | u64::from(row_words));
        // Random words, a few holding a signed normalized minimum code in a lane, which random
        // bits rarely give a 16-bit lane.
        let before: Vec<u32> = (0..rows * row_words)
            .map(|i| match (i % 7, rng.next()) {
                (0, w) => w & 0xffff_0000 | 0x8000,
                (1, w) => w & 0x0000_ffff | 0x8000_0000,
                (2, w) => w & 0xffff_ff00 | 0x80,
                (_, w) => w,
            })
            .collect();
        let bytes: Vec<u8> = before.iter().flat_map(|w| w.to_le_bytes()).collect();
        let groups = (before.len() as u32).div_ceil(UnusedBitsModule::LOCAL_SIZE) + 1;
        let shape = UnusedBitsModule::shape(row_words, first_row, layer_rows, tiles_per_row);
        let after = rewritten((format, words), &bytes, &mask, shape, groups);
        let (mut changed, mut kept) = (0, 0);
        for (i, (&word, got)) in before.iter().zip(after.chunks(4)).enumerate() {
            let i = i as u32;
            let row = first_row + i / row_words;
            let w = i % row_words;
            let (layer, y) = (row / layer_rows, row % layer_rows);
            let x = 4 * w / texel;
            let tile = (layer * tile_rows + y / 32) * tiles_per_row + x / 32;
            let want = match set(tile) {
                true => stored(format, words, word, w % 2 == 1),
                false => word,
            };
            let got = u32::from_le_bytes([got[0], got[1], got[2], got[3]]);
            assert_eq!(
                got, want,
                "{format:?} {words:?} word {i} (row {row}, word {w}, tile {tile}): {word:08x}"
            );
            match got == word {
                true => kept += 1,
                false => changed += 1,
            }
        }
        assert!(changed > 0 && kept > 0, "{format:?} {words:?}");
    }
}

/// Every 16-bit word of a padded linear form keeps its top ten bits and loses its low six, and an
/// `_sRGB` twin's takes its code's load code (`ExtendedRange::load_code`: 23 codes of
/// `R10Unorm_X6_sRGB` move, 1021 of `RG10Unorm_X12_sRGB`), every value of a word among them,
/// under a mask of every tile, across several workgroups.
#[test]
fn every_padded_word_takes_its_code_s_load_code_with_its_low_six_bits_zeroed() {
    let halves: Vec<u16> = (0..=u16::MAX).collect();
    let bytes: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
    let words = (bytes.len() / 4) as u32;
    let groups = words.div_ceil(UnusedBitsModule::LOCAL_SIZE) + 1;
    for format in [
        PixelFormat::R10UnormX6,
        PixelFormat::RG10UnormX12,
        PixelFormat::R10UnormX6SRGB,
        PixelFormat::RG10UnormX12SRGB,
    ] {
        let form = ExtendedRange::of(format).expect("a padded form");
        let moved = (0..1024).filter(|&c| form.load_code(c) != Some(c)).count();
        let want_moved = match form {
            ExtendedRange::R10X6Srgb => 23,
            ExtendedRange::Rg10X12Srgb => 1021,
            _ => 0,
        };
        assert_eq!(moved, want_moved, "{form:?}");
        let want: Vec<u8> = halves
            .iter()
            .flat_map(|&h| {
                let c = form.load_code(u32::from(h >> 6)).expect("a padded code");
                (u16::try_from(c << 6).expect("a word")).to_le_bytes()
            })
            .collect();
        // One row of every word: its texels' tiles all set.
        let texel = UnusedBitsModule::texel_bytes(format, TexelWords::Raw).unwrap();
        let tiles_per_row = (4 * words / texel).div_ceil(32);
        let mask = vec![u32::MAX; tiles_per_row.div_ceil(32) as usize];
        let shape = UnusedBitsModule::shape(words, 0, 1, tiles_per_row);
        assert_eq!(
            rewritten((format, TexelWords::Raw), &bytes, &mask, shape, groups),
            want,
            "{form:?}"
        );
    }
}

/// Every code of each bfloat form takes its load code (`ExtendedRange::load_code`, measured for
/// every code), under a mask of every tile, across several workgroups: 56,829
/// codes move (the NaNs to `0x7fc0`, the magnitudes from 65,536 to infinity, those below
/// `2^-14` to `half`'s step), the 8,707 others stay.
#[test]
fn every_bfloat_code_takes_its_load_code() {
    let codes: Vec<u16> = (0..=u16::MAX).collect();
    let bytes: Vec<u8> = codes.iter().flat_map(|h| h.to_le_bytes()).collect();
    let words = (bytes.len() / 4) as u32;
    let groups = words.div_ceil(UnusedBitsModule::LOCAL_SIZE) + 1;
    for format in [
        PixelFormat::R16Bfloat,
        PixelFormat::RG16Bfloat,
        PixelFormat::RGBA16Bfloat,
    ] {
        let form = ExtendedRange::of(format).expect("a bfloat form");
        let load = |c: u16| form.load_code(u32::from(c)).expect("a bfloat code");
        let moved = codes.iter().filter(|&&c| load(c) != u32::from(c)).count();
        assert_eq!(moved, 56_829, "{form:?}");
        // A few of Metal's answers: the NaNs, an overflow, the codes `half`
        // holds, and the double rounding below `2^-14`.
        for (code, metal) in [
            (0x7f81, 0x7fc0),
            (0xffff, 0x7fc0),
            (0xff7a, 0xff80),
            (0x4780, 0x7f80),
            (0x477f, 0x477f),
            (0x3880, 0x3880),
            (0x8001, 0x8000),
            (0x0001, 0x0000),
            (0x3301, 0x3380),
        ] {
            assert_eq!(load(code), metal, "{form:?} {code:#06x}");
        }
        let want: Vec<u8> = codes
            .iter()
            .flat_map(|&c| (u16::try_from(load(c)).expect("a code")).to_le_bytes())
            .collect();
        let texel = UnusedBitsModule::texel_bytes(format, TexelWords::Raw).unwrap();
        let tiles_per_row = (4 * words / texel).div_ceil(32);
        let mask = vec![u32::MAX; tiles_per_row.div_ceil(32) as usize];
        let shape = UnusedBitsModule::shape(words, 0, 1, tiles_per_row);
        assert_eq!(
            rewritten((format, TexelWords::Raw), &bytes, &mask, shape, groups),
            want,
            "{form:?}"
        );
    }
}

/// Every 16-bit value of a signed normalized format's word takes Metal's store under a mask of
/// every tile: a lane at the minimum code (`0x80` of an 8-bit format's byte, `0x8000` of a 16-bit
/// one's half) becomes the code above it, every other lane stays (an 8-bit format's 256 minimum
/// pairs of bytes move, a 16-bit one's one value).
#[test]
fn every_signed_normalized_minimum_code_is_raised_to_the_code_above_it() {
    let halves: Vec<u16> = (0..=u16::MAX).collect();
    let bytes: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
    let words = (bytes.len() / 4) as u32;
    let groups = words.div_ceil(UnusedBitsModule::LOCAL_SIZE) + 1;
    for format in SNORM {
        let wide = matches!(
            format,
            PixelFormat::R16Snorm | PixelFormat::RG16Snorm | PixelFormat::RGBA16Snorm
        );
        let want: Vec<u8> = halves
            .iter()
            .flat_map(|&h| {
                let raised = match wide {
                    true if h == 0x8000 => 0x8001,
                    true => h,
                    false => u16::from_le_bytes(h.to_le_bytes().map(|b| match b {
                        0x80 => 0x81,
                        b => b,
                    })),
                };
                raised.to_le_bytes()
            })
            .collect();
        let moved = halves
            .iter()
            .zip(want.chunks(2))
            .filter(|(h, w)| h.to_le_bytes() != [w[0], w[1]])
            .count();
        assert_eq!(moved, if wide { 1 } else { 511 }, "{format:?}");
        let texel = UnusedBitsModule::texel_bytes(format, TexelWords::Raw).unwrap();
        let tiles_per_row = (4 * words / texel).div_ceil(32);
        let mask = vec![u32::MAX; tiles_per_row.div_ceil(32) as usize];
        let shape = UnusedBitsModule::shape(words, 0, 1, tiles_per_row);
        assert_eq!(
            rewritten((format, TexelWords::Raw), &bytes, &mask, shape, groups),
            want,
            "{format:?}"
        );
    }
}

/// Each `RGB9E5Float` word of every exponent and every largest mantissa (each mantissa's own
/// value, the others random below it) takes Metal's store under a mask of every tile: normalized
/// where its largest mantissa is below 256 under an exponent above 0, as the word Metal's pass left
/// for `1 | 2 << 9 | 3 << 18 | 20 << 27` (an Apple M3 Max under macOS 27.0.1), kept otherwise.
#[test]
fn every_unnormalized_shared_exponent_word_is_normalized() {
    // The M3 Max's answer: the largest mantissa 3 shifted up seven times, the exponent 13.
    let metal = 1 << 7 | 2 << 16 | 3 << 25 | 13 << 27;
    assert_eq!(shared_exponent(1 | 2 << 9 | 3 << 18 | 20 << 27), metal);
    let mut rng = Rng(0x9e5);
    let mut words = Vec::new();
    for e in 0..32u32 {
        for largest in 0..512u32 {
            let lane = rng.next() % 3;
            let m: [u32; 3] = std::array::from_fn(|k| {
                if k as u32 == lane {
                    largest
                } else {
                    rng.next() % (largest + 1)
                }
            });
            words.push(m[0] | m[1] << 9 | m[2] << 18 | e << 27);
        }
    }
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let want: Vec<u8> = words
        .iter()
        .flat_map(|&w| shared_exponent(w).to_le_bytes())
        .collect();
    let moved = words.iter().filter(|&&w| shared_exponent(w) != w).count();
    assert_eq!(
        moved,
        31 * 256,
        "every word below 256 under an exponent above 0"
    );
    let n = words.len() as u32;
    let groups = n.div_ceil(UnusedBitsModule::LOCAL_SIZE) + 1;
    let tiles_per_row = n.div_ceil(32);
    let mask = vec![u32::MAX; tiles_per_row.div_ceil(32) as usize];
    let shape = UnusedBitsModule::shape(n, 0, 1, tiles_per_row);
    let format = (PixelFormat::RGB9E5Float, TexelWords::Raw);
    assert_eq!(rewritten(format, &bytes, &mask, shape, groups), want);
}

/// A mask of no word leaves every word, and a tile past the mask's words reads as unset.
#[test]
fn a_tile_past_the_mask_is_not_rewritten() {
    let form = (PixelFormat::BGR10XR, TexelWords::Raw);
    // Two layers of one row of 64 texels, 40 tiles a row: the first row's tiles 0 and 1 are the
    // mask's one word's bits 0 and 1 (tile 0 set), and the second's, 40 and 41, lie past it.
    let bytes = vec![0u8; 4 * 128];
    let shape = UnusedBitsModule::shape(64, 0, 1, 40);
    let after = rewritten(form, &bytes, &[0b01], shape, 2);
    for (i, word) in after.chunks(4).enumerate() {
        let word = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
        let want = if i < 32 { 0xc000_0000 } else { 0 };
        assert_eq!(word, want, "word {i}");
    }
}
