//! Apple's parameter-data layout and decoders: the arithmetic's edges. The bytes and answers Metal
//! gives for whole maps are checked against `bind::RasterizationRateMap`.

use super::*;

/// The map's index truncates in range, clamps both ends, and reads `NaN` and infinities as
/// bounds; a shader's saturates and is not clamped.
#[test]
fn indices_clamp_or_saturate() {
    assert_eq!(index(2.75, 9), 2);
    assert_eq!(index(-0.5, 9), 0);
    assert_eq!(index(f32::NAN, 9), 0);
    assert_eq!(index(f32::INFINITY, 9), 9);
    assert_eq!(index(f32::NEG_INFINITY, 9), 0);
    assert_eq!(index(1.0e30, 9), 9);
    assert_eq!(shader_index(513.9), 513);
    assert_eq!(shader_index(-0.5), 0);
    assert_eq!(shader_index(f32::NAN), 0);
    assert_eq!(shader_index(3.0e7), 30_000_000);
    assert_eq!(shader_index(SHADER_INDEX_SATURATION), u32::MAX);
    assert_eq!(shader_index(f32::INFINITY), u32::MAX);
    assert_eq!(
        shader_index(SHADER_INDEX_SATURATION.next_down()),
        4_294_967_040
    );
}

/// The layout's tables, and the last entries the encoder writes.
#[test]
fn tables_and_entries() {
    assert_eq!(WORDS, 6156);
    assert_eq!(table(0, 0, false), 0);
    assert_eq!(table(1, 0, false), 1026);
    assert_eq!(table(0, 0, true), 2052);
    assert_eq!(table(1, 0, true), 3078);
    assert_eq!(table(7, 1, false), 4104);
    assert_eq!(table(7, 1, true), 5130);
    // A shader's layer 2 reads the screen-to-physical tables as physical-to-screen ones.
    assert_eq!(table(2, 0, false), table(0, 0, true));
    assert_eq!(last_screen_entry(1), 2);
    assert_eq!(last_screen_entry(16384), 512);
    assert_eq!(last_physical_entry(528), 17);
    assert_eq!(last_physical_entry(16384), 512);
}

/// A tile's line: the scale is `K / 2^24` (finer than an `f32` quotient's below 1/2), and the
/// offset the exact line's at screen 0 in 1/256 pixels, rounded to nearest.
#[test]
fn tile_lines_are_apples_fixed_point() {
    let identity = tile_line(64, 64, 32, 32);
    assert_eq!(identity.scale, 1.0);
    assert_eq!(identity.reciprocal, 1.0);
    assert_eq!(identity.offset, 0);
    // 63 screen pixels to 32: K = round(2^29 / 63) = 8521761.
    let half = tile_line(63, 32, 63, 32);
    assert_eq!(half.scale, 8_521_761.0 / 16_777_216.0);
    assert_eq!(half.reciprocal, 1.0 / half.scale);
    assert_eq!(half.offset, 0);
    // A quarter: 126 to 32, K = round(2^29 / 126) = 4260880; the f32 quotient is finer.
    let quarter = tile_line(0, 0, 126, 32);
    assert_eq!(quarter.scale, 4_260_880.0 / 16_777_216.0);
    assert_ne!(quarter.scale, 32.0f32 / 126.0);
    // The offset is 256 · (P - S · p / s) rounded: 7731 to 3584 at 41:32 is
    // -627187.512..., where the fixed-point scale's line gives -627187.4546.
    assert_eq!(tile_line(7731, 3584, 41, 32).offset, -627_188);
    assert_eq!(tile_line(1024, 416, 85, 32).offset, 7806);
}

/// An offset exactly halfway between two steps is a tie Apple's compile never makes; one is
/// reachable by explicit tiles.
#[test]
fn offset_ties() {
    // 256 · (0 - 1 · 1 / 512) = -1/2.
    assert!(offset_tie(1, 0, 512, 1));
    assert!(!offset_tie(1, 0, 512, 2));
    assert!(!offset_tie(7731, 3584, 41, 32));
    // Every tile before an axis's last is 32 physical pixels from a multiple of 32.
    for s in 32..=3200 {
        for start in [0, 1, 4096, 8192, 12_345] {
            assert!(
                !offset_tie(start, 32 * (start / 64), s, 32),
                "{s} at {start}"
            );
        }
    }
}

/// The encoder writes `⌈P/32⌉ + 1` physical-to-screen entries and `⌈W/32⌉ + 2`
/// screen-to-physical ones (the last tile's line after the extent), each boundary byte in half
/// pixels, and zeros after.
#[test]
fn an_axis_encodes_as_apples_tables() {
    let (mut to_screen, mut to_physical) =
        (vec![0; TABLE_WORDS as usize], vec![0; TABLE_WORDS as usize]);
    // Tiles of 40 then 36 screen pixels to 32 and 20 physical: 76 to 52.
    encode_axis(&[(40, 32), (36, 20)], &mut to_screen, &mut to_physical);
    let first = tile_line(0, 0, 40, 32);
    let last = tile_line(40, 32, 36, 20);
    let pair = |line: TileLine| {
        [
            line.reciprocal.to_bits(),
            (line.offset as f32 / 256.0).to_bits(),
        ]
    };
    assert_eq!(
        to_screen[..6],
        [pair(first), pair(last), pair(last)].concat()
    );
    assert!(to_screen[6..].iter().all(|&w| w == 0));
    let packed = |line: TileLine, boundary: u32| ((line.offset as u32) << 8) | boundary;
    // Granule 1 (screen 32 to 64) holds the boundary at 40: 16 half pixels in.
    assert_eq!(
        to_physical[..10],
        [
            first.scale.to_bits(),
            packed(first, NO_BOUNDARY),
            first.scale.to_bits(),
            packed(first, 16),
            last.scale.to_bits(),
            packed(last, NO_BOUNDARY),
            last.scale.to_bits(),
            packed(last, NO_BOUNDARY),
            last.scale.to_bits(),
            packed(last, NO_BOUNDARY),
        ]
    );
    assert!(to_physical[10..].iter().all(|&w| w == 0));
}

/// No words decode to zeros, the sign of the coordinate's zero kept where IEEE keeps it.
#[test]
fn zero_words_decode_to_zero() {
    let zeros = [0u32; 8];
    assert_eq!(
        screen_to_physical(&zeros, 0, [5.5, -2.0], [64, 64]).map(f32::to_bits),
        [0, 0]
    );
    assert_eq!(
        shader_physical_to_screen(&[], 3, [5.5, -7.0]).map(f32::to_bits),
        [0, 0x8000_0000]
    );
}

/// A shader reads a denormal as the zero of its sign, and answers Metal's `NaN`; the map keeps
/// both.
#[test]
fn a_shader_flushes_denormals_and_answers_metals_nan() {
    let mut words = vec![0u32; WORDS as usize];
    let (to_screen, to_physical) = words.split_at_mut(HORIZONTAL_TO_PHYSICAL as usize);
    encode_axis(
        &[(32, 32)],
        &mut to_screen[..TABLE_WORDS as usize],
        to_physical,
    );
    let tiny = -f32::from_bits(1);
    let shader = shader_physical_to_screen(&words, 0, [tiny, 0.0]);
    assert_eq!(shader[0].to_bits(), 0x8000_0000);
    let map = physical_to_screen(&words, 0, [tiny, 0.0], [32, 32]);
    assert_eq!(map[0], tiny);
    let nan = f32::from_bits(0x7fa0_0000);
    let shader = shader_screen_to_physical(&words, 0, [nan, 0.0]);
    assert_eq!(shader[0].to_bits(), 0x7fc0_0000);
    assert!(screen_to_physical(&words, 0, [nan, 0.0], [32, 32])[0].is_nan());
}
