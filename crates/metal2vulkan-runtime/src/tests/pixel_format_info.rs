//! `pixel_format_info`: `MTLPixelFormatGetInfoForDevice` answers our own measurements at each Metal
//! level (an Apple M3 Max: our probe and the consumer's macOS 27 host lane, whose answers are an
//! M4's but from 630 to 645), the level describes a format exactly where the table has a row,
//! and `MTLGetTextureLevelInfoForDeviceWithOptions` answers the probes' layouts (on each).

use crate::{MetalLevel, PixelFormatInfo, TextureLevelQuery};
use metal2vulkan::metal::PixelFormat as P;

/// Every raw value the probes asked (0 to 2047) with a description at `level`.
fn described(level: MetalLevel) -> Vec<u64> {
    (0..2048)
        .filter(|&raw| level.pixel_format_info(raw).describes_a_format())
        .collect()
}

/// An Apple M3 Max describes the same 264 values under macOS 26.7 and 27.0.1, none from 630 to
/// 645 (which an Apple M4 describes under 27.0.1).
#[test]
fn each_level_answers_its_measured_rows() {
    let old = described(MetalLevel::MacOS26_7);
    let new = described(MetalLevel::MacOS27_0_1);
    assert_eq!(old.len(), 264);
    assert_eq!(new, old);
    assert!(!new.iter().any(|raw| (630..=645).contains(raw)));
    // Past every named value, and a value no format has, Metal answers Invalid.
    for raw in [0, 2, 9, 2047, 2048, u64::from(u32::MAX) + 10, u64::MAX] {
        for level in MetalLevel::ALL {
            assert_eq!(
                level.pixel_format_info(raw),
                PixelFormatInfo::INVALID,
                "{raw}"
            );
        }
    }
}

/// A level describes a format (`MetalLevel::describes`) exactly where `pixel_format_info` answers
/// one, and then under the format's own Metal name; 24-bit depth formats, which no Apple GPU has,
/// at no level.
#[test]
fn a_level_describes_exactly_the_formats_it_answers() {
    for level in MetalLevel::ALL {
        for &pf in P::ALL {
            let info = level.pixel_format_info(u64::from(pf.as_raw()));
            assert_eq!(
                level.describes(pf),
                info.describes_a_format(),
                "{level} {pf:?}"
            );
            if info.describes_a_format() {
                assert_eq!(info.name, pf.name(), "{level} {pf:?}");
            }
        }
        for pf in [P::Depth24UnormStencil8, P::X24Stencil8] {
            assert!(!level.describes(pf), "{level} {pf:?}");
        }
    }
}

/// Spot values, as the probes printed them: a format alike at both levels, a cast class numbered
/// anew, flags that differ, a compressed format's block, and a format only an Apple M4 describes.
#[test]
fn the_answers_are_the_probes() {
    let at = |level: MetalLevel, pf: P| level.pixel_format_info(u64::from(pf.as_raw()));
    for level in MetalLevel::ALL {
        let bgra8 = at(level, P::BGRA8Unorm);
        assert_eq!(bgra8.name, "MTLPixelFormatBGRA8Unorm");
        assert_eq!(bgra8.words[0], 4, "{level}");
    }
    // RG32Float: the same flags and words, cast class 4 at 26.7 and 6 at 27.0.1.
    let old = at(MetalLevel::MacOS26_7, P::RG32Float);
    let new = at(MetalLevel::MacOS27_0_1, P::RG32Float);
    assert_eq!(
        (old.flags, old.cast_class, old.words),
        (0x0403_019f, 4, [8, 1026, 8, 8])
    );
    assert_eq!(
        (new.flags, new.cast_class, new.words),
        (0x0403_019f, 6, [8, 1026, 8, 8])
    );
    // RGB10A2Unorm_sRGB: its flags differ by level.
    assert_eq!(
        at(MetalLevel::MacOS26_7, P::RGB10A2UnormSRGB).flags,
        0x0003_099d
    );
    assert_eq!(
        at(MetalLevel::MacOS27_0_1, P::RGB10A2UnormSRGB).flags,
        0x0703_099d
    );
    // BC1_RGBA: compressed (flags bit 0x400), an 8-byte 4x4x1 block.
    let bc1 = at(MetalLevel::MacOS27_0_1, P::BC1RGBA);
    assert_ne!(bc1.flags & 0x400, 0);
    assert_eq!(bc1.words, [8, 4, 4, 1]);
    // BGRA10Unorm (638): only an Apple M4 under macOS 27.0.1 describes it.
    for level in MetalLevel::ALL {
        assert_eq!(
            at(level, P::BGRA10Unorm),
            PixelFormatInfo::INVALID,
            "{level}"
        );
    }
}

/// A format of blocks (flags bit 0x400) states its block in the four words: bytes, then a width
/// and height of more than one texel, then a depth of 1. They are the compressed formats (80 at
/// 27.0.1, ASTC 4x2's 4 by 2 among them) and the 4:2:2 formats, of 4-byte blocks of 2 by 1.
#[test]
fn block_formats_state_their_block() {
    for level in MetalLevel::ALL {
        for raw in described(level) {
            let info = level.pixel_format_info(raw);
            if info.flags & 0x400 != 0 {
                let [bytes, width, height, depth] = info.words;
                assert!(bytes > 0 && width * height > 1, "{level} {}", info.name);
                assert_eq!(depth, 1, "{level} {}", info.name);
            }
        }
    }
    let at = |pf: P| MetalLevel::MacOS27_0_1.pixel_format_info(u64::from(pf.as_raw()));
    assert_eq!(at(P::GBGR422).words, [4, 2, 1, 1]);
    assert_eq!(at(P::BGRG422).words, [4, 2, 1, 1]);
    assert_eq!(at(P::ASTC4x4LDR).words[1..], [4, 4, 1]);
}

/// One measured answer of `MTLGetTextureLevelInfoForDeviceWithOptions`, as the probes printed it:
/// format, width, height, depth, array length, level, plane argument, then Metal's 13 words.
/// Each was answered alike on an Apple M3 Max under macOS 26.7 and an Apple M4 under 27.0.1.
type Measured = ([u64; 7], [u64; 13]);

/// Answers that state the rule: BGRA8Unorm (80) at levels 0, 1, 64 (the level modulo 64), 1000 and
/// 2^32, with an array length of 6 and a depth of 3; BC1_RGBA (130) and ASTC_12x10_LDR (217),
/// whose blocks cover the size rounded up; zero sizes, which are 1, and a zero array length,
/// which answers no bytes; values no format has, which answer no bytes.
#[rustfmt::skip]
const MEASURED: &[Measured] = &[
    ([80, 17, 9, 1, 1, 0, 0], [17, 9, 1, 17, 9, 1, 17, 9, 1, 4, 68, 612, 612]),
    ([80, 17, 9, 1, 1, 1, 0], [8, 4, 1, 8, 4, 1, 8, 4, 1, 4, 32, 128, 128]),
    ([80, 17, 9, 1, 1, 64, 0], [17, 9, 1, 17, 9, 1, 17, 9, 1, 4, 68, 612, 612]),
    ([80, 17, 9, 1, 1, 1000, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4]),
    ([80, 17, 9, 1, 1, 1 << 32, 0], [17, 9, 1, 17, 9, 1, 17, 9, 1, 4, 68, 612, 612]),
    ([80, 33, 17, 1, 6, 1, 0], [16, 8, 1, 16, 8, 1, 16, 8, 1, 24, 384, 3072, 3072]),
    ([80, 1000, 7, 3, 2, 0, 0], [1000, 7, 3, 1000, 7, 3, 1000, 7, 3, 8, 8000, 56000, 168000]),
    ([80, 1000, 7, 3, 2, 1, 0], [500, 3, 1, 500, 3, 1, 500, 3, 1, 8, 4000, 12000, 12000]),
    ([130, 17, 9, 1, 1, 0, 0], [17, 9, 1, 5, 3, 1, 20, 12, 1, 8, 40, 120, 120]),
    ([130, 1000, 7, 3, 2, 0, 0], [1000, 7, 3, 250, 2, 3, 1000, 8, 3, 16, 4000, 8000, 24000]),
    ([130, 1000, 7, 3, 2, 2, 0], [250, 1, 1, 63, 1, 1, 252, 4, 1, 16, 1008, 1008, 1008]),
    ([130, 5, 3, 1, 1, 2, 0], [1, 1, 1, 1, 1, 1, 4, 4, 1, 8, 8, 8, 8]),
    ([217, 17, 9, 1, 1, 0, 0], [17, 9, 1, 2, 1, 1, 24, 10, 1, 16, 32, 32, 32]),
    ([217, 1000, 7, 3, 2, 0, 0], [1000, 7, 3, 84, 1, 3, 1008, 10, 3, 32, 2688, 2688, 8064]),
    ([80, 0, 0, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4]),
    ([130, 0, 5, 1, 1, 0, 0], [1, 5, 1, 1, 2, 1, 4, 8, 1, 8, 8, 16, 16]),
    ([80, 17, 9, 0, 1, 0, 0], [17, 9, 1, 17, 9, 1, 17, 9, 1, 4, 68, 612, 612]),
    ([130, 17, 9, 1, 0, 0, 0], [17, 9, 1, 5, 3, 1, 20, 12, 1, 0, 0, 0, 0]),
    ([0, 64, 64, 2, 3, 1, 1], [32, 32, 1, 32, 32, 1, 32, 32, 1, 0, 0, 0, 0]),
    ([99999, 5, 3, 1, 1, 0, 2], [5, 3, 1, 5, 3, 1, 5, 3, 1, 0, 0, 0, 0]),
];

/// The block's bytes (word 9) the depth-and-stencil formats answer at 17 by 9, array length 1,
/// by plane argument 0 to 7: Depth24Unorm_Stencil8 (255), Depth32Float_Stencil8 (260),
/// X32_Stencil8 (261) and X24_Stencil8 (262).
const PLANES: &[(u64, [u64; 8])] = &[
    (255, [0, 4, 1, 0, 0, 0, 0, 0]),
    (260, [0, 4, 1, 8, 8, 8, 8, 8]),
    (261, [1, 1, 1, 1, 1, 1, 1, 1]),
    (262, [1, 0, 1, 0, 0, 0, 0, 0]),
];

fn query(args: [u64; 7]) -> TextureLevelQuery {
    let [pixel_format, width, height, depth, array_length, level, planes] = args;
    TextureLevelQuery {
        pixel_format,
        width,
        height,
        depth,
        array_length,
        level,
        planes,
    }
}

/// Each level answers the probes' layouts word for word.
#[test]
fn texture_level_info_answers_the_probes() {
    for level in MetalLevel::ALL {
        for &(args, words) in MEASURED {
            assert_eq!(
                level.texture_level_info(&query(args)).words(),
                words,
                "{level} {args:?}"
            );
        }
        for &(raw, by_plane) in PLANES {
            for (planes, &bytes) in (0..).zip(by_plane.iter()) {
                let info = level.texture_level_info(&query([raw, 17, 9, 1, 1, 0, planes]));
                assert_eq!(info.bytes_per_block, bytes, "{level} {raw} {planes}");
                assert_eq!(info.bytes, bytes * 17 * 9, "{level} {raw} {planes}");
            }
        }
    }
}

/// A 2D plane's query is level 0 of one image, plane argument 0, and answers as the full query.
#[test]
fn a_plane_query_is_level_zero_of_one_image() {
    assert_eq!(
        TextureLevelQuery::plane_2d(80, 17, 9),
        query([80, 17, 9, 1, 1, 0, 0])
    );
    let info = MetalLevel::MacOS27_0_1.texture_level_info(&TextureLevelQuery::plane_2d(80, 17, 9));
    assert_eq!(
        (info.bytes_per_row, info.aligned_height, info.bytes),
        (68, 9, 612)
    );
}

/// `u64::MAX`, the largest extent.
const ALL_ONES: u64 = u64::MAX;
/// `2^62 + 3`: a block count past `2^60`, and a row of `2^63` bytes and more, without a wrap of
/// the add.
const NEAR_TOP: u64 = (1 << 62) + 3;

/// Answers near `2^64`, by the rule: Metal counts a level's blocks as `(extent + block - 1) /
/// block` with an add that wraps, so an extent past `2^64 - block` has no blocks (the consumer's
/// comparison on an Apple M3 Max under macOS 26.7: at a width of `u64::MAX` and a 4-wide block
/// Metal answers 0 blocks, where rounding up gives `2^62`), and the fields that follow from the
/// count follow it: the size the blocks cover, the bytes of a row, an image and the level. BC1_RGBA
/// (130: 4 by 4, 8 bytes), ASTC_12x10_LDR (217: 12 by 10, 16 bytes), GBGR422 (240: 2 by 1, 4 bytes)
/// and BGRA8Unorm (80: one texel, 4 bytes), at all ones, at the first extents whose add wraps and
/// the last that do not, at `2^62 + 3`, and at level 1, where the shifted extent never wraps. The
/// words are the rule's, computed apart from the runtime's code; our own probe on an Apple M4 under
/// macOS 27.0.1 answers the rule at every one of its 495 queries, most of these
/// among them.
#[rustfmt::skip]
const NEAR_TWO_TO_THE_64: &[Measured] = &[
    // A 4-wide block: all ones and the two below it wrap; three below it does not.
    ([130, ALL_ONES, 17, 1, 1, 0, 0], [ALL_ONES, 17, 1, 0, 5, 1, 0, 20, 1, 8, 0, 0, 0]),
    ([130, ALL_ONES - 2, 17, 1, 1, 0, 0], [ALL_ONES - 2, 17, 1, 0, 5, 1, 0, 20, 1, 8, 0, 0, 0]),
    ([130, ALL_ONES - 3, 17, 1, 1, 0, 0],
     [ALL_ONES - 3, 17, 1, (1 << 62) - 1, 5, 1, ALL_ONES - 3, 20, 1, 8, ALL_ONES - 7, ALL_ONES - 39,
      ALL_ONES - 39]),
    ([130, 17, ALL_ONES, 1, 1, 0, 0], [17, ALL_ONES, 1, 5, 0, 1, 20, 0, 1, 8, 40, 0, 0]),
    ([130, ALL_ONES, ALL_ONES, 1, 1, 0, 0], [ALL_ONES, ALL_ONES, 1, 0, 0, 1, 0, 0, 1, 8, 0, 0, 0]),
    ([130, NEAR_TOP, ALL_ONES, 1, 1, 0, 0],
     [NEAR_TOP, ALL_ONES, 1, (1 << 60) + 1, 0, 1, (1 << 62) + 4, 0, 1, 8, (1 << 63) + 8, 0, 0]),
    ([130, NEAR_TOP, 17, 1, 1, 0, 0],
     [NEAR_TOP, 17, 1, (1 << 60) + 1, 5, 1, (1 << 62) + 4, 20, 1, 8, (1 << 63) + 8, (1 << 63) + 40,
      (1 << 63) + 40]),
    // Level 1 halves all ones to `2^63 - 1`, whose add does not wrap.
    ([130, ALL_ONES, 17, 1, 1, 1, 0],
     [(1 << 63) - 1, 8, 1, 1 << 61, 2, 1, 1 << 63, 8, 1, 8, 0, 0, 0]),
    ([130, 17, ALL_ONES, 1, 1, 1, 0],
     [8, (1 << 63) - 1, 1, 2, 1 << 61, 1, 8, 1 << 63, 1, 8, 16, 0, 0]),
    // A 12-by-10 block: ten below all ones wraps across, eleven does not; eight below wraps down,
    // nine does not.
    ([217, ALL_ONES, ALL_ONES, 1, 1, 0, 0], [ALL_ONES, ALL_ONES, 1, 0, 0, 1, 0, 0, 1, 16, 0, 0, 0]),
    ([217, ALL_ONES - 10, 17, 1, 1, 0, 0], [ALL_ONES - 10, 17, 1, 0, 2, 1, 0, 20, 1, 16, 0, 0, 0]),
    ([217, ALL_ONES - 11, 17, 1, 1, 0, 0],
     [ALL_ONES - 11, 17, 1, 1_537_228_672_809_129_301, 2, 1, ALL_ONES - 3, 20, 1, 16,
      6_148_914_691_236_517_200, 12_297_829_382_473_034_400, 12_297_829_382_473_034_400]),
    ([217, 17, ALL_ONES - 8, 1, 1, 0, 0], [17, ALL_ONES - 8, 1, 2, 0, 1, 24, 0, 1, 16, 32, 0, 0]),
    ([217, 17, ALL_ONES - 9, 1, 1, 0, 0],
     [17, ALL_ONES - 9, 1, 2, 1_844_674_407_370_955_161, 1, 24, ALL_ONES - 5, 1, 16, 32,
      3_689_348_814_741_910_304, 3_689_348_814_741_910_304]),
    // A 2-by-1 block wraps at all ones across; a block one texel high never adds.
    ([240, ALL_ONES, 17, 1, 1, 0, 0], [ALL_ONES, 17, 1, 0, 17, 1, 0, 17, 1, 4, 0, 0, 0]),
    ([240, ALL_ONES - 1, 17, 1, 1, 0, 0],
     [ALL_ONES - 1, 17, 1, (1 << 63) - 1, 17, 1, ALL_ONES - 1, 17, 1, 4, ALL_ONES - 3,
      ALL_ONES - 67, ALL_ONES - 67]),
    ([240, 17, ALL_ONES, 1, 1, 0, 0],
     [17, ALL_ONES, 1, 9, ALL_ONES, 1, 18, ALL_ONES, 1, 4, 36, ALL_ONES - 35, ALL_ONES - 35]),
    // One texel: the count is the extent, and only the products wrap.
    ([80, ALL_ONES, 17, 1, 1, 0, 0],
     [ALL_ONES, 17, 1, ALL_ONES, 17, 1, ALL_ONES, 17, 1, 4, ALL_ONES - 3, ALL_ONES - 67,
      ALL_ONES - 67]),
    ([80, NEAR_TOP, NEAR_TOP, 1, 1, 0, 0],
     [NEAR_TOP, NEAR_TOP, 1, NEAR_TOP, NEAR_TOP, 1, NEAR_TOP, NEAR_TOP, 1, 4, 12, 36, 36]),
    ([80, ALL_ONES, ALL_ONES, 1, 1, 0, 0],
     [ALL_ONES, ALL_ONES, 1, ALL_ONES, ALL_ONES, 1, ALL_ONES, ALL_ONES, 1, 4, ALL_ONES - 3, 4, 4]),
];

/// Each level answers the wrapped block counts and the fields that follow from them.
#[test]
fn extents_near_two_to_the_64_wrap_the_block_count() {
    for level in MetalLevel::ALL {
        for &(args, words) in NEAR_TWO_TO_THE_64 {
            assert_eq!(
                level.texture_level_info(&query(args)).words(),
                words,
                "{level} {args:?}"
            );
        }
    }
}

/// Every ASTC LDR format, and its sRGB form, is described as Metal describes it whatever the
/// device holds its blocks in (its ASTC format, or a carrier, `astc.rs`): a format of blocks (flags
/// bit 0x400) of 16 bytes, its footprint, depth 1; and a level's layout counts its blocks, a side
/// no multiple of the block rounded up, rows of 16-byte blocks, an image of those rows.
#[test]
fn astc_formats_answer_metal_s_blocks_and_levels() {
    use metal2vulkan::bind::AstcDecode;
    let formats = P::ALL
        .iter()
        .filter_map(|&pf| AstcDecode::of(pf).map(|decode| (pf, decode)));
    let mut seen = 0;
    for (pf, decode) in formats {
        seen += 1;
        let [bw, bh] = decode.block_extent().map(u64::from);
        for level in MetalLevel::ALL {
            let raw = u64::from(pf.as_raw());
            let info = level.pixel_format_info(raw);
            assert_ne!(info.flags & 0x400, 0, "{level} {pf:?}");
            assert_eq!(info.words, [16, bw, bh, 1], "{level} {pf:?}");
            for (width, height, mip) in [(37, 23, 0), (37, 23, 2), (1000, 7, 1), (1, 1, 0)] {
                let at = |side: u64| (side >> mip).max(1);
                let (w, h) = (at(width), at(height));
                let [across, down] = [w.div_ceil(bw), h.div_ceil(bh)];
                let mut query = TextureLevelQuery::plane_2d(raw, width, height);
                query.level = mip;
                let answer = level.texture_level_info(&query);
                let what = format!("{level} {pf:?} {width}x{height} level {mip}");
                assert_eq!([answer.width, answer.height], [w, h], "{what}");
                assert_eq!(
                    [answer.blocks_across, answer.blocks_down],
                    [across, down],
                    "{what}"
                );
                assert_eq!(answer.bytes_per_block, 16, "{what}");
                assert_eq!(answer.bytes_per_row, 16 * across, "{what}");
                assert_eq!(answer.bytes_per_image, 16 * across * down, "{what}");
            }
        }
    }
    assert_eq!(seen, 28);
}
