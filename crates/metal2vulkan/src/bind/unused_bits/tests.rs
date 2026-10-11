use super::{rule, typed_module, TexelWords, UnusedBitsModule, SHAPE, TILES, WORDS};
use crate::metal::format::ExtendedRange;
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};

/// Every form and the words it is held in.
fn forms() -> Vec<(ExtendedRange, TexelWords)> {
    let mut all = Vec::new();
    for srgb in [false, true] {
        all.push((ExtendedRange::Bgr10 { srgb }, TexelWords::Raw));
        all.push((ExtendedRange::Bgra10 { srgb }, TexelWords::Raw));
        all.push((ExtendedRange::Planes { srgb }, TexelWords::Raw));
        all.push((ExtendedRange::Planes { srgb }, TexelWords::ColorPlane));
    }
    for form in [
        ExtendedRange::R10X6,
        ExtendedRange::Rg10X12,
        ExtendedRange::R10X6Srgb,
        ExtendedRange::Rg10X12Srgb,
        ExtendedRange::R16Bfloat,
        ExtendedRange::Rg16Bfloat,
        ExtendedRange::Rgba16Bfloat,
    ] {
        all.push((form, TexelWords::Raw));
    }
    all
}

/// The module of `form` held as `words`.
fn module(form: ExtendedRange, words: TexelWords) -> crate::spirv::ir::TypedModule {
    let rule = rule(form, words).expect("a rule");
    let bytes = UnusedBitsModule::texel_bytes(form, words).expect("texel bytes");
    typed_module(&rule, bytes).expect("module")
}

#[test]
fn the_unused_bits_kernel_is_valid_on_every_committed_profile() {
    for (_, p) in committed_profiles().expect("profiles") {
        for (form, words) in forms() {
            assert_valid(module(form, words), &p);
        }
    }
}

/// Only a two-plane format has color-plane words.
#[test]
fn only_a_two_plane_format_has_color_plane_words() {
    for (form, words) in forms() {
        assert!(rule(form, words).is_some(), "{form:?} {words:?}");
        let plane = rule(form, TexelWords::ColorPlane).is_some();
        assert_eq!(
            plane,
            matches!(form, ExtendedRange::Planes { .. }),
            "{form:?}"
        );
    }
}

/// Metal's store of `word`, the `odd` word of its texel, of `form` held as `words` (the module
/// docs), computed here apart from the kernel.
fn stored(form: ExtendedRange, words: TexelWords, word: u32, odd: bool) -> u32 {
    let codes = 0x3fff_ffff;
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

/// Run the kernel of `form` held as `words` over `bytes` with the mask `tiles` and the shape
/// `shape`, in `groups` workgroups, both invocation orders.
fn rewritten(
    (form, words): (ExtendedRange, TexelWords),
    bytes: &[u8],
    tiles: &[u32],
    shape: [u8; 16],
    groups: u32,
) -> Vec<u8> {
    let m = module(form, words);
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
/// store, every other word stays, for every form, and invocations past the words write nothing.
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
    for (form, words) in forms() {
        let texel = UnusedBitsModule::texel_bytes(form, words).unwrap();
        let row_words = (columns * texel).div_ceil(4);
        let rows = layers * layer_rows - first_row;
        let mut rng = Rng(u64::from(texel) << 32 | u64::from(row_words));
        let before: Vec<u32> = (0..rows * row_words).map(|_| rng.next()).collect();
        let bytes: Vec<u8> = before.iter().flat_map(|w| w.to_le_bytes()).collect();
        let groups = (before.len() as u32).div_ceil(UnusedBitsModule::LOCAL_SIZE) + 1;
        let shape = UnusedBitsModule::shape(row_words, first_row, layer_rows, tiles_per_row);
        let after = rewritten((form, words), &bytes, &mask, shape, groups);
        let (mut changed, mut kept) = (0, 0);
        for (i, (&word, got)) in before.iter().zip(after.chunks(4)).enumerate() {
            let i = i as u32;
            let row = first_row + i / row_words;
            let w = i % row_words;
            let (layer, y) = (row / layer_rows, row % layer_rows);
            let x = 4 * w / texel;
            let tile = (layer * tile_rows + y / 32) * tiles_per_row + x / 32;
            let want = match set(tile) {
                true => stored(form, words, word, w % 2 == 1),
                false => word,
            };
            let got = u32::from_le_bytes([got[0], got[1], got[2], got[3]]);
            assert_eq!(
                got, want,
                "{form:?} {words:?} word {i} (row {row}, word {w}, tile {tile}): {word:08x}"
            );
            match got == word {
                true => kept += 1,
                false => changed += 1,
            }
        }
        assert!(changed > 0 && kept > 0, "{form:?} {words:?}");
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
    for form in [
        ExtendedRange::R10X6,
        ExtendedRange::Rg10X12,
        ExtendedRange::R10X6Srgb,
        ExtendedRange::Rg10X12Srgb,
    ] {
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
        let texel = UnusedBitsModule::texel_bytes(form, TexelWords::Raw).unwrap();
        let tiles_per_row = (4 * words / texel).div_ceil(32);
        let mask = vec![u32::MAX; tiles_per_row.div_ceil(32) as usize];
        let shape = UnusedBitsModule::shape(words, 0, 1, tiles_per_row);
        assert_eq!(
            rewritten((form, TexelWords::Raw), &bytes, &mask, shape, groups),
            want,
            "{form:?}"
        );
    }
}

/// Every code of each bfloat form takes its load code (`ExtendedRange::load_code`, measured for
/// every code: EV FRT-bfp1-m4), under a mask of every tile, across several workgroups: 56,829
/// codes move (the NaNs to `0x7fc0`, the magnitudes from 65,536 to infinity, those below
/// `2^-14` to `half`'s step), the 8,707 others stay.
#[test]
fn every_bfloat_code_takes_its_load_code() {
    let codes: Vec<u16> = (0..=u16::MAX).collect();
    let bytes: Vec<u8> = codes.iter().flat_map(|h| h.to_le_bytes()).collect();
    let words = (bytes.len() / 4) as u32;
    let groups = words.div_ceil(UnusedBitsModule::LOCAL_SIZE) + 1;
    for form in [
        ExtendedRange::R16Bfloat,
        ExtendedRange::Rg16Bfloat,
        ExtendedRange::Rgba16Bfloat,
    ] {
        let load = |c: u16| form.load_code(u32::from(c)).expect("a bfloat code");
        let moved = codes.iter().filter(|&&c| load(c) != u32::from(c)).count();
        assert_eq!(moved, 56_829, "{form:?}");
        // A few of Metal's answers (EV FRT-bfp1-m4): the NaNs, an overflow, the codes `half`
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
        let texel = UnusedBitsModule::texel_bytes(form, TexelWords::Raw).unwrap();
        let tiles_per_row = (4 * words / texel).div_ceil(32);
        let mask = vec![u32::MAX; tiles_per_row.div_ceil(32) as usize];
        let shape = UnusedBitsModule::shape(words, 0, 1, tiles_per_row);
        assert_eq!(
            rewritten((form, TexelWords::Raw), &bytes, &mask, shape, groups),
            want,
            "{form:?}"
        );
    }
}

/// A mask of no word leaves every word, and a tile past the mask's words reads as unset.
#[test]
fn a_tile_past_the_mask_is_not_rewritten() {
    let form = (ExtendedRange::Bgr10 { srgb: false }, TexelWords::Raw);
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
